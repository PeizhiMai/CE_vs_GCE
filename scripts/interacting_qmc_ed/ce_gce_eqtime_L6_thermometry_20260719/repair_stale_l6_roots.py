#!/usr/bin/env python3.11
"""Safely resume stale L=6 roots with fresh, complete checkpoints.

The simulation wrappers intentionally stop before the Slurm wall clock, but
MPI teardown can occasionally last until Slurm terminates the allocation.  In
that case the wrapper never gets a chance to submit its own continuation.
This helper is the conservative backstop used by the hourly monitor:

* it never submits while any job with the same workflow job name is queued;
* it skips strict-final roots;
* it resumes only complete, recent checkpoint sets (or complete GCE data that
  merely still needs export/validation);
* every repair uses a new delta manifest and an append-only ledger;
* all arrays remain unthrottled on ccsd/burst/default.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import os
from pathlib import Path
import re
import socket
import subprocess
import time
from typing import Iterable


HERE = Path(__file__).resolve().parent
MANIFESTS = HERE / "manifests"
STATUS = HERE / "status_source"
LOGS = Path("/home/9pm/nUHubbard/logs")
TABLES = (
    "equal_time_observables_qmc.tsv",
    "equal_time_charge_spin_wedge_qmc.tsv",
    "equal_time_structure_factors_qmc.tsv",
    "equal_time_neighbor_shells_qmc.tsv",
)
LEDGER_FIELDS = (
    "timestamp_utc", "job_name", "job_id", "continuation_count",
    "source_manifests", "delta_manifest", "root", "reason",
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def queued_names() -> set[str]:
    user = subprocess.check_output(["id", "-un"], text=True).strip()
    out = subprocess.check_output(["squeue", "-r", "-u", user, "-h", "-o", "%j"], text=True)
    return {x.strip() for x in out.splitlines() if x.strip()}


def recent_complete(paths: list[Path], expected: int, cutoff: float) -> bool:
    return len(paths) == expected and all(p.stat().st_mtime >= cutoff for p in paths)


def status_valid(path: Path) -> bool:
    data: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            data[key] = value
    try:
        int(data.get("warmups_completed", "-1"))
        int(data.get("nsamples", data.get("completed_batches", "-1")))
    except ValueError:
        return False
    return data.get("reason", "") in {
        "runtime_limit", "runtime_limit_warmup", "runtime_limit_thermalization", "periodic"
    }


def ce_strict_final(row: dict[str, str]) -> bool:
    root = Path(row["outdir"])
    expected = int(row["expected_ranks"])
    ranks = list(root.glob("ranks/rank_*/checkpoint_complete.txt"))
    return (
        (root / "checkpoint_complete.txt").is_file()
        and len(ranks) == expected
        and all((root / name).is_file() for name in TABLES)
    )


def ce_eligible(row: dict[str, str], cutoff: float) -> tuple[bool, str]:
    if ce_strict_final(row):
        return False, "strict_final"
    root = Path(row["outdir"])
    expected = int(row["expected_ranks"])
    checkpoints = sorted(root.glob("ranks/rank_*/checkpoint.jls"))
    statuses = sorted(root.glob("ranks/rank_*/checkpoint.jls.status"))
    if not recent_complete(checkpoints, expected, cutoff):
        return False, f"checkpoint_coverage={len(checkpoints)}/{expected}_or_stale"
    if not recent_complete(statuses, expected, cutoff):
        return False, f"status_coverage={len(statuses)}/{expected}_or_stale"
    if not all(status_valid(path) for path in statuses):
        return False, "invalid_status_reason_or_progress"
    return True, "fresh_full_ce_checkpoint_status"


def probe_base(row: dict[str, str]) -> str:
    prefix = "attractive_hubbard_rect" if float(row["U"]) < 0 else "hubbard_spin_hs_rect"
    return (
        f"{prefix}_U{float(row['U']):.2f}_tp0.00_mu{float(row['mu_probe']):.2f}_"
        f"Lx{int(row['Lx'])}_Ly{int(row['Ly'])}_b{float(row['beta']):.2f}-{int(row['sid'])}"
    )


def probe_eligible(row: dict[str, str], cutoff: float) -> tuple[bool, str]:
    parent = Path(row["out_parent"])
    base = probe_base(row)
    if (parent / f"complete_{base}" / "global_stats.csv").is_file():
        return False, "strict_final"
    checkpoints = sorted((parent / base).glob("checkpoint_pID-*.jld2"))
    if not recent_complete(checkpoints, 32, cutoff):
        return False, f"checkpoint_coverage={len(checkpoints)}/32_or_stale"
    return True, "fresh_full_gce_probe_checkpoint"


def production_base(row: dict[str, str]) -> str:
    prefix = "attractive_hubbard_rect" if float(row["U"]) < 0 else "hubbard_spin_hs_rect"
    return (
        f"{prefix}_U{float(row['U']):.2f}_tp0.00_mu{float(row['mu_final']):.2f}_"
        f"Lx{int(row['Lx'])}_Ly{int(row['Ly'])}_b{float(row['beta']):.2f}-{int(row['sid'])}"
    )


def production_strict_final(row: dict[str, str]) -> bool:
    parent = Path(row["out_parent"])
    complete = parent / f"complete_{production_base(row)}"
    export = parent / "export"
    ranks = list(complete.glob("simulation_info_sID-*_pID-*.toml"))
    achieved = parent / "achieved_density.tsv"
    density_ok = False
    if achieved.is_file():
        try:
            density_ok = abs(float(read_tsv(achieved)[0]["delta_N"])) <= float(row["density_tolerance"])
        except (KeyError, ValueError, IndexError):
            density_ok = False
    return (
        (parent / "dqmc_gce_eqtime_complete.txt").is_file()
        and len(ranks) == int(row["expected_ranks"])
        and all((export / name).is_file() for name in TABLES)
        and density_ok
    )


def production_eligible(row: dict[str, str], cutoff: float) -> tuple[bool, str]:
    parent = Path(row["out_parent"])
    if production_strict_final(row):
        return False, "strict_final"
    if (parent / "density_tolerance_failed.txt").is_file():
        return False, "density_tolerance_failed_requires_retune"
    expected = int(row["expected_ranks"])
    base = production_base(row)
    complete = parent / f"complete_{base}"
    if (complete / "global_stats.csv").is_file():
        ranks = list(complete.glob("simulation_info_sID-*_pID-*.toml"))
        if len(ranks) == expected:
            return True, "complete_data_needs_export_or_validation"
        return False, f"complete_data_wrong_rank_coverage={len(ranks)}/{expected}"
    checkpoints = sorted((parent / base).glob("checkpoint_pID-*.jld2"))
    if not recent_complete(checkpoints, expected, cutoff):
        return False, f"checkpoint_coverage={len(checkpoints)}/{expected}_or_stale"
    return True, "fresh_full_gce_production_checkpoint"


def observed_max_continuation(job_name: str) -> int:
    maximum = 0
    pattern = re.compile(rb"continuation=(\d+)/(\d+)")
    for path in LOGS.glob(f"{job_name}_*.out"):
        try:
            with path.open("rb") as f:
                head = f.read(32_000)
            for match in pattern.finditer(head):
                maximum = max(maximum, int(match.group(1)))
        except OSError:
            pass
    return maximum


def submit_delta(
    *, job_name: str, wrapper: Path, source_paths: list[Path], rows: list[dict[str, str]],
    roots: list[str], reasons: list[str], ledger: list[dict[str, str]], dry_run: bool,
) -> list[dict[str, str]]:
    accounts = {row.get("account", "ccsd") for row in rows}
    if len(accounts) != 1 or not accounts <= {"ccsd", "cnms"}:
        raise SystemExit(f"refusing mixed or invalid repair accounts for {job_name}: {sorted(accounts)}")
    account = next(iter(accounts))
    prior = [int(x["continuation_count"]) for x in ledger if x.get("job_name") == job_name]
    next_count = max([observed_max_continuation(job_name), *prior, 0]) + 1
    cap = 80 if job_name.startswith("muL6") else 120
    if next_count > cap:
        print(f"BLOCK {job_name}: continuation cap {cap} reached")
        return ledger
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    delta = STATUS / "repair_manifests" / f"{job_name}_repair_{stamp}.tsv"
    copied = [dict(row) for row in rows]
    for idx, row in enumerate(copied):
        if "idx" in row:
            row["idx"] = str(idx)
    write_tsv(delta, copied, copied[0].keys())
    command = [
        "sbatch", "--parsable", "-A", account, "-p", "burst", "--qos=default",
        f"--job-name={job_name}", f"--array=0-{len(copied)-1}",
        "--export=ALL," + f"MANIFEST={delta},RESUBMIT_COUNT={next_count},"
        "CHECKPOINT_RESET_ACCUMULATORS=false,AUTO_RESUBMIT=true",
        str(wrapper),
    ]
    print("REPAIR_SUBMIT", " ".join(command))
    if dry_run:
        return ledger
    job_id = subprocess.check_output(command, text=True).strip().split(";")[0]
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    sources = ",".join(str(x) for x in source_paths)
    for root, reason in zip(roots, reasons):
        ledger.append({
            "timestamp_utc": now, "job_name": job_name, "job_id": job_id,
            "continuation_count": str(next_count), "source_manifests": sources,
            "delta_manifest": str(delta), "root": root, "reason": reason,
        })
    write_tsv(STATUS / "repair_ledger.tsv", ledger, LEDGER_FIELDS)
    print(f"REPAIRED job={job_id} name={job_name} rows={len(rows)} continuation={next_count}/{cap}")
    return ledger


def process_spec(
    *, job_name: str, wrapper: Path, source_paths: list[Path], root_field: str,
    eligibility, queue: set[str], cutoff: float, ledger: list[dict[str, str]], dry_run: bool,
) -> list[dict[str, str]]:
    if not source_paths:
        return ledger
    if job_name in queue:
        print(f"SKIP {job_name}: active/pending scheduler leg exists")
        return ledger
    unique: dict[str, dict[str, str]] = {}
    for path in source_paths:
        for row in read_tsv(path):
            unique.setdefault(row[root_field], row)
    eligible_rows: list[dict[str, str]] = []
    roots: list[str] = []
    reasons: list[str] = []
    blocked: list[tuple[str, str]] = []
    for root, row in unique.items():
        ok, reason = eligibility(row, cutoff)
        if ok:
            eligible_rows.append(row); roots.append(root); reasons.append(reason)
        elif reason != "strict_final":
            blocked.append((root, reason))
    if blocked:
        print(f"BLOCKED {job_name} count={len(blocked)}")
        for root, reason in blocked[:12]:
            print(f"  {reason} {root}")
    if not eligible_rows:
        print(f"NO_REPAIR {job_name}: no eligible stale roots")
        return ledger
    # A user-directed account move can make a family contain both ccsd and
    # cnms roots.  Preserve each root's intended account in future repairs and
    # never put mixed-account rows in one delta manifest.
    by_account: dict[str, list[int]] = {}
    for idx, row in enumerate(eligible_rows):
        by_account.setdefault(row.get("account", "ccsd"), []).append(idx)
    for account, indices in sorted(by_account.items()):
        ledger = submit_delta(
            job_name=job_name, wrapper=wrapper, source_paths=source_paths,
            rows=[eligible_rows[i] for i in indices], roots=[roots[i] for i in indices],
            reasons=[reasons[i] for i in indices], ledger=ledger, dry_run=dry_run,
        )
    return ledger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-age-hours", type=float, default=72.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if os.environ.get("USER") != "9pm" or "or-" not in socket.gethostname().lower():
        raise SystemExit("run as 9pm on a CADES host")
    STATUS.mkdir(parents=True, exist_ok=True)
    lock_path = STATUS / ".repair_stale_l6_roots.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("SKIP: another L6 repair audit is running")
            return
        cutoff = time.time() - args.max_age_hours * 3600
        queue = queued_names()
        ledger_path = STATUS / "repair_ledger.tsv"
        ledger = read_tsv(ledger_path) if ledger_path.is_file() else []

        ce_specs = [
            ("ceL6Attr", "ce_L6_attractive_beta_le10_r32_m50000.tsv", "job_ce_L6_r32_cades.sbatch"),
            ("ceL6A20", "ce_L6_attractive_beta20_r64_m30000.tsv", "job_ce_L6_r64_cades.sbatch"),
            ("ceL6Pos", "ce_L6_positive_beta_le4_r32_m50000.tsv", "job_ce_L6_r32_cades.sbatch"),
            ("ceL6Pilot", "ce_L6_positive_pilot_beta5_6p7_10_r32_m10000.tsv", "job_ce_L6_r32_cades.sbatch"),
            ("ceL6PosProd", "ce_L6_positive_admitted_r32_m50000.tsv", "job_ce_L6_r32_cades.sbatch"),
            ("ceL6PosHi", "ce_L6_positive_admitted_r64_m30000.tsv", "job_ce_L6_r64_cades.sbatch"),
        ]
        for job, manifest_name, wrapper_name in ce_specs:
            path = MANIFESTS / manifest_name
            if path.is_file():
                ledger = process_spec(
                    job_name=job, wrapper=HERE / wrapper_name, source_paths=[path],
                    root_field="outdir", eligibility=ce_eligible, queue=queue,
                    cutoff=cutoff, ledger=ledger, dry_run=args.dry_run,
                )

        for family, job, wrapper_name in [
            ("attractive", "muL6Attr", "job_gce_mu_L6_attractive_cades.sbatch"),
            ("spinHS", "muL6Spin", "job_gce_mu_L6_spinHS_cades.sbatch"),
        ]:
            if family == "attractive":
                # Account-move manifests come first so setdefault() preserves
                # the user-directed account for those output roots.
                paths = (
                    sorted((STATUS / "account_moves").glob("gce_mu_probe_L6_attractive*_account_move*.tsv"))
                    + sorted(MANIFESTS.glob("gce_mu_probe_L6_attractive*.tsv"))
                )
            else:
                paths = (
                    sorted((STATUS / "account_moves").glob("gce_mu_probe_L6_spinHS*_account_move*.tsv"))
                    + sorted({
                    *MANIFESTS.glob("gce_mu_probe_L6_positive_spinHS*.tsv"),
                    *MANIFESTS.glob("gce_mu_probe_L6_spinHS_followup_*.tsv"),
                    })
                )
            ledger = process_spec(
                job_name=job, wrapper=HERE / wrapper_name, source_paths=paths,
                root_field="out_parent", eligibility=probe_eligible, queue=queue,
                cutoff=cutoff, ledger=ledger, dry_run=args.dry_run,
            )

        for family, job, manifest_name, wrapper_name in [
            ("attractive", "gceL6Attr", "gce_prod_L6_attractive_confirmed.tsv", "job_gce_prod_L6_attractive_cades.sbatch"),
            ("spinHS", "gceL6Spin", "gce_prod_L6_spinHS_confirmed.tsv", "job_gce_prod_L6_spinHS_cades.sbatch"),
        ]:
            path = MANIFESTS / manifest_name
            if path.is_file():
                ledger = process_spec(
                    job_name=job, wrapper=HERE / wrapper_name, source_paths=[path],
                    root_field="out_parent", eligibility=production_eligible, queue=queue,
                    cutoff=cutoff, ledger=ledger, dry_run=args.dry_run,
                )


if __name__ == "__main__":
    main()
