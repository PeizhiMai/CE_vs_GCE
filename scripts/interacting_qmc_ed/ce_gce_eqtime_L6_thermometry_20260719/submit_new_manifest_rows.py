#!/usr/bin/env python3
"""Submit only previously unsubmitted rows from a generated L=6 manifest.

The on-CADES ledger and per-root marker make repeated stage advancement
idempotent.  Arrays are deliberately unthrottled and fixed to
ccsd/burst/default.  This helper is for newly generated pilot/tuning/production
manifests; checkpoint repairs remain a separate audited action.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
from pathlib import Path
import socket
import subprocess


HERE = Path(__file__).resolve().parent
LEDGER_FIELDS = ("timestamp_utc", "source_manifest", "delta_manifest", "root", "job_id", "job_name", "wrapper")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def strict_final(root: Path, root_field: str) -> bool:
    if root_field == "outdir":
        tables = (
            "equal_time_observables_qmc.tsv", "equal_time_charge_spin_wedge_qmc.tsv",
            "equal_time_structure_factors_qmc.tsv", "equal_time_neighbor_shells_qmc.tsv",
        )
        return (root / "checkpoint_complete.txt").is_file() and all((root / x).is_file() for x in tables)
    if (root / "dqmc_gce_eqtime_complete.txt").is_file():
        return True
    return any((p / "global_stats.csv").is_file() for p in root.glob("complete_*"))


def active_manifest_paths() -> str:
    user = subprocess.check_output(["id", "-un"], text=True).strip()
    ids = subprocess.check_output(["squeue", "-r", "-u", user, "-h", "-o", "%A"], text=True).split()
    lines = []
    for job_id in sorted(set(ids)):
        proc = subprocess.run(["scontrol", "show", "job", "-o", job_id], text=True, capture_output=True)
        if proc.returncode == 0:
            lines.append(proc.stdout)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--root-field", choices=("outdir", "out_parent"), required=True)
    ap.add_argument("--wrapper", type=Path, required=True)
    ap.add_argument("--job-name", required=True)
    ap.add_argument("--ledger", type=Path, default=HERE / "status_source" / "submission_ledger.tsv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if os.environ.get("USER") != "9pm" or "or-" not in socket.gethostname().lower():
        raise SystemExit("this submitter must run as 9pm on a CADES host")
    manifest = args.manifest.resolve(); wrapper = args.wrapper.resolve()
    rows = read_tsv(manifest)
    if not rows:
        print(f"no rows: {manifest}"); return
    if args.root_field not in rows[0]:
        raise SystemExit(f"{manifest}: root field {args.root_field} missing")
    active = active_manifest_paths()
    if str(manifest) in active:
        print(f"SKIP active/pending scheduler job already references {manifest}")
        return
    ledger_rows = read_tsv(args.ledger) if args.ledger.is_file() else []
    submitted = {r["root"] for r in ledger_rows}
    delta: list[dict[str, str]] = []
    for row in rows:
        root = Path(row[args.root_field])
        marker = root / ".l6_submission_jobid"
        if str(root) in submitted or marker.is_file() or strict_final(root, args.root_field):
            continue
        delta.append(dict(row))
    if not delta:
        print(f"no new rows to submit from {manifest}")
        return
    for i, row in enumerate(delta):
        if "idx" in row:
            row["idx"] = str(i)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    delta_path = HERE / "status_source" / "submission_manifests" / f"{manifest.stem}_delta_{stamp}.tsv"
    write_tsv(delta_path, delta, list(delta[0].keys()))
    command = [
        "sbatch", "--parsable", "-A", "ccsd", "-p", "burst", "--qos=default",
        f"--job-name={args.job_name}", f"--array=0-{len(delta)-1}",
        f"--export=ALL,MANIFEST={delta_path}", str(wrapper),
    ]
    print("SUBMIT", " ".join(command))
    if args.dry_run:
        return
    job_id = subprocess.check_output(command, text=True).strip().split(";")[0]
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    new_ledger = []
    for row in delta:
        root = Path(row[args.root_field]); root.mkdir(parents=True, exist_ok=True)
        (root / ".l6_submission_jobid").write_text(f"job_id={job_id}\nsource_manifest={manifest}\ndelta_manifest={delta_path}\n")
        new_ledger.append({
            "timestamp_utc": now, "source_manifest": str(manifest), "delta_manifest": str(delta_path),
            "root": str(root), "job_id": job_id, "job_name": args.job_name, "wrapper": str(wrapper),
        })
    write_tsv(args.ledger, ledger_rows + new_ledger, list(LEDGER_FIELDS))
    print(f"submitted job={job_id} rows={len(delta)} manifest={delta_path}")


if __name__ == "__main__":
    main()
