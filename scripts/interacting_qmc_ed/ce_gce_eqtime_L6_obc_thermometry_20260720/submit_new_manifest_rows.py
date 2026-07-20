#!/usr/bin/env python3
"""Idempotently submit only never-submitted rows from an L=6 OBC manifest."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
from pathlib import Path
import socket
import subprocess
from typing import Iterable

HERE = Path(__file__).resolve().parent
PRIMARY = (
    "equal_time_kinetic_per_site_qmc.tsv",
    "equal_time_double_occupancy_per_site_qmc.tsv",
    "equal_time_nn_spin_qmc.tsv",
    "equal_time_nn_connected_charge_qmc.tsv",
)
LEDGER_FIELDS = (
    "timestamp_utc", "source_manifest", "delta_manifest", "root", "job_id",
    "array_task", "job_name", "wrapper", "account", "partition", "qos",
)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write(path: Path, rows: Iterable[dict[str, str]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def strict_final(root: Path, row: dict[str, str], root_field: str) -> bool:
    if root_field == "outdir":
        expected = int(row["expected_ranks"])
        return (
            (root / "obc_thermometry_complete.txt").is_file()
            and len(list(root.glob("ranks/rank_*/checkpoint_complete.txt"))) == expected
            and len(list(root.glob("ranks/rank_*/equal_time_site_density_qmc.tsv"))) == expected
            and all((root / name).is_file() for name in PRIMARY)
        )
    if "mu_final" in row:
        return (root / "dqmc_gce_obc_eqtime_complete.txt").is_file()
    completes = list(root.glob("complete_*_obc_*"))
    return len(completes) == 1 and (completes[0] / "global_stats.csv").is_file()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root-field", choices=("outdir", "out_parent"), required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--ledger", type=Path, default=HERE / "status_source" / "submission_ledger.tsv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if os.environ.get("USER") != "9pm" or "or-" not in socket.gethostname().lower():
        raise SystemExit("run as 9pm on a CADES host reached through lowercase SSH host cades")
    manifest = args.manifest.resolve()
    wrapper = args.wrapper.resolve()
    rows = read(manifest)
    if not rows:
        print(f"NO_ROWS {manifest}"); return
    if args.root_field not in rows[0]:
        raise SystemExit(f"{manifest}: missing root field {args.root_field}")
    roots = [row[args.root_field] for row in rows]
    if len(set(roots)) != len(roots):
        raise SystemExit(f"{manifest}: duplicate output roots")
    for row, root in zip(rows, roots):
        if row.get("boundary") != "open" or "_obc_" not in root:
            raise SystemExit(f"{manifest}: non-OBC row/root refused: {root}")
        if row.get("partition", "burst") != "burst" or row.get("qos", "default") != "default":
            raise SystemExit(f"{manifest}: row not on burst/default: {root}")

    accounts = {row.get("account", "ccsd") for row in rows}
    if not accounts or not accounts <= {"ccsd", "cnms"}:
        raise SystemExit(f"{manifest}: mixed or invalid accounts {sorted(accounts)}")
    ledger_rows = read(args.ledger) if args.ledger.is_file() else []
    submitted = {row["root"] for row in ledger_rows}
    delta: list[dict[str, str]] = []
    source_indices: list[int] = []
    for source_idx, row in enumerate(rows):
        root = Path(row[args.root_field])
        marker = root / ".l6_obc_submission_jobid"
        if str(root) in submitted or marker.is_file() or strict_final(root, row, args.root_field):
            continue
        delta.append(dict(row)); source_indices.append(source_idx)
    if not delta:
        print(f"NO_NEW_ROWS {manifest}"); return
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    grouped: dict[str, list[tuple[int, dict[str, str]]]] = {}
    for source_idx, row in zip(source_indices, delta):
        grouped.setdefault(row.get("account", "ccsd"), []).append((source_idx, row))
    for account in sorted(grouped):
        group = grouped[account]
        group_rows = [dict(row) for _, row in group]
        for idx, row in enumerate(group_rows):
            row["idx"] = str(idx)
        delta_path = HERE / "status_source" / "submission_manifests" / (
            f"{manifest.stem}_{account}_delta_{stamp}.tsv"
        )
        write(delta_path, group_rows, group_rows[0].keys())
        command = [
            "sbatch", "--parsable", "-A", account, "-p", "burst", "--qos=default",
            f"--job-name={args.job_name}", f"--array=0-{len(group_rows)-1}", str(wrapper),
        ]
        print("SUBMIT", " ".join(command), f"MANIFEST={delta_path}")
        if args.dry_run:
            continue
        env = dict(os.environ)
        env.update({
            "MANIFEST": str(delta_path),
            "CHECKPOINT_RESET_ACCUMULATORS": "false",
            "AUTO_RESUBMIT": "true",
        })
        job_id = subprocess.check_output(command, text=True, env=env).strip().split(";")[0]
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        additions: list[dict[str, str]] = []
        for task, ((source_idx, _), row) in enumerate(zip(group, group_rows)):
            root = Path(row[args.root_field]); root.mkdir(parents=True, exist_ok=True)
            (root / ".l6_obc_submission_jobid").write_text(
                f"job_id={job_id}\narray_task={task}\nsource_row={source_idx}\n"
                f"source_manifest={manifest}\ndelta_manifest={delta_path}\naccount={account}\n"
            )
            additions.append({
                "timestamp_utc": now, "source_manifest": str(manifest),
                "delta_manifest": str(delta_path), "root": str(root), "job_id": job_id,
                "array_task": str(task), "job_name": args.job_name, "wrapper": str(wrapper),
                "account": account, "partition": "burst", "qos": "default",
            })
        ledger_rows.extend(additions)
        write(args.ledger, ledger_rows, LEDGER_FIELDS)
        print(f"SUBMITTED job={job_id} rows={len(group_rows)} array=0-{len(group_rows)-1} account={account} unthrottled=true")


if __name__ == "__main__":
    main()
