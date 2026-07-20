#!/usr/bin/env python3
"""Root-aware repair of orphaned L=6 OBC checkpoint sets.

A repair is submitted only when the root is required, not strict-final, has a
fresh full checkpoint set (and CE status set), and has no active or pending
scheduler element mapped to that exact root.  Unknown workflow jobs and
multiple queued legs for one root are fatal, so this helper fails closed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import os
from pathlib import Path
import socket
import subprocess
import time
from collections import Counter, defaultdict
from typing import Iterable

HERE = Path(__file__).resolve().parent
MANIFESTS = HERE / "manifests"
STATUS = HERE / "status_source"
PRIMARY = (
    "equal_time_kinetic_per_site_qmc.tsv",
    "equal_time_double_occupancy_per_site_qmc.tsv",
    "equal_time_nn_spin_qmc.tsv",
    "equal_time_nn_connected_charge_qmc.tsv",
)
JOB_NAMES = {
    "ceL6OAtr", "ceL6OA20", "ceL6OPos", "ceL6OPil", "ceL6OPN", "ceL6OPH",
    "muL6OAttr", "muL6OSpin", "gceL6OAttr", "gceL6OSpin",
}
REPAIR_FIELDS = (
    "timestamp_utc", "job_name", "job_id", "array_task", "continuation_count",
    "source_manifest", "delta_manifest", "root", "reason", "account",
    "partition", "qos",
)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write(path: Path, rows: Iterable[dict[str, str]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def recent_full(paths: list[Path], expected: int, cutoff: float) -> bool:
    try:
        return (
            len(paths) == expected
            and all(path.is_file() and path.stat().st_size > 0 and path.stat().st_mtime >= cutoff for path in paths)
        )
    except OSError:
        return False


def ce_status_valid(path: Path) -> bool:
    values: dict[str, str] = {}
    try:
        for line in path.read_text(errors="replace").splitlines():
            if "=" in line:
                key, value = line.split("=", 1); values[key] = value
        int(values.get("warmups_completed", "-1"))
        int(values.get("nsamples", values.get("completed_batches", "-1")))
    except (OSError, ValueError):
        return False
    return values.get("reason") in {
        "runtime_limit", "runtime_limit_warmup", "runtime_limit_thermalization", "periodic"
    }


def obc_base(row: dict[str, str], mu_field: str) -> str:
    family = row["family"]
    prefix = "attractive_hubbard_obc_rect" if family == "attractive" else "hubbard_spin_hs_obc_rect"
    return (
        f"{prefix}_U{float(row['U']):.2f}_tp0.00_mu{float(row[mu_field]):.2f}_"
        f"Lx{int(row['Lx'])}_Ly{int(row['Ly'])}_b{float(row['beta']):.2f}-{int(row['sid'])}"
    )


def ce_final(row: dict[str, str]) -> bool:
    root = Path(row["outdir"]); expected = int(row["expected_ranks"])
    return (
        (root / "obc_thermometry_complete.txt").is_file()
        and len(list(root.glob("ranks/rank_*/checkpoint_complete.txt"))) == expected
        and len(list(root.glob("ranks/rank_*/equal_time_site_density_qmc.tsv"))) == expected
        and all((root / name).is_file() for name in PRIMARY)
    )


def ce_eligible(row: dict[str, str], cutoff: float) -> tuple[bool, str]:
    if ce_final(row): return False, "strict_final"
    root = Path(row["outdir"]); expected = int(row["expected_ranks"])
    checkpoints = sorted(root.glob("ranks/rank_*/checkpoint.jls"))
    statuses = sorted(root.glob("ranks/rank_*/checkpoint.jls.status"))
    if not recent_full(checkpoints, expected, cutoff):
        return False, f"checkpoint_coverage={len(checkpoints)}/{expected}_or_stale"
    if not recent_full(statuses, expected, cutoff):
        return False, f"status_coverage={len(statuses)}/{expected}_or_stale"
    if not all(ce_status_valid(path) for path in statuses):
        return False, "invalid_status_reason_or_progress"
    return True, "fresh_full_ce_checkpoint_status"


def probe_final(row: dict[str, str]) -> bool:
    parent = Path(row["out_parent"]); complete = parent / f"complete_{obc_base(row, 'mu_probe')}"
    return (
        (complete / "global_stats.csv").is_file()
        and len(list(complete.glob("simulation_info_sID-*_pID-*.toml"))) == 32
        and (parent / "probe_achieved_density.tsv").is_file()
    )


def probe_eligible(row: dict[str, str], cutoff: float) -> tuple[bool, str]:
    if probe_final(row): return False, "strict_final"
    root = Path(row["out_parent"]) / obc_base(row, "mu_probe")
    checkpoints = sorted(root.glob("checkpoint_pID-*.jld2"))
    if not recent_full(checkpoints, 32, cutoff):
        return False, f"checkpoint_coverage={len(checkpoints)}/32_or_stale"
    return True, "fresh_full_gce_probe_checkpoint"


def production_bracket_valid(row: dict[str, str]) -> bool:
    try:
        lo, high = sorted((float(row["mu_bracket_low"]), float(row["mu_bracket_high"])))
        fitted = float(row["mu_fitted"]); final_mu = float(row["mu_final"])
    except (KeyError, ValueError):
        return False
    return (
        high - lo <= 0.020000000001
        and lo - 1e-12 <= fitted <= high + 1e-12
        and lo - 1e-12 <= final_mu <= high + 1e-12
    )


def production_final(row: dict[str, str]) -> bool:
    parent = Path(row["out_parent"]); expected = int(row["expected_ranks"])
    complete = parent / f"complete_{obc_base(row, 'mu_final')}"
    achieved = parent / "achieved_density.tsv"
    density_ok = False
    if achieved.is_file():
        try:
            data = read(achieved)[0]
            density_ok = abs(float(data["delta_N"])) <= float(row["density_tolerance"])
        except (IndexError, KeyError, ValueError):
            pass
    return (
        (parent / "dqmc_gce_obc_eqtime_complete.txt").is_file()
        and len(list(complete.glob("obc_equal_time_rank_pID-*.tsv"))) == expected
        and len(list(complete.glob("obc_equal_time_site_rank_pID-*.tsv"))) == expected
        and all((complete / name).is_file() for name in PRIMARY)
        and density_ok
        and production_bracket_valid(row)
    )


def production_eligible(row: dict[str, str], cutoff: float) -> tuple[bool, str]:
    parent = Path(row["out_parent"]); expected = int(row["expected_ranks"])
    if not production_bracket_valid(row):
        return False, "superseded_bad_final_bracket"
    if production_final(row): return False, "strict_final"
    if (parent / "density_tolerance_failed.txt").is_file():
        return False, "density_tolerance_failed_requires_retune"
    base = obc_base(row, "mu_final"); complete = parent / f"complete_{base}"
    if (complete / "global_stats.csv").is_file():
        ranks = len(list(complete.glob("simulation_info_sID-*_pID-*.toml")))
        return (ranks == expected, "complete_data_needs_export_or_validation" if ranks == expected else f"complete_wrong_ranks={ranks}/{expected}")
    checkpoints = sorted((parent / base).glob("checkpoint_pID-*.jld2"))
    if not recent_full(checkpoints, expected, cutoff):
        return False, f"checkpoint_coverage={len(checkpoints)}/{expected}_or_stale"
    return True, "fresh_full_gce_production_checkpoint"


def authoritative_rows() -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    specs = [
        (sorted(MANIFESTS.glob("ce_L6_obc_attractive_*.tsv")), "ce", "outdir"),
        ([MANIFESTS / "ce_L6_obc_positive_beta_le4_r32_m50000.tsv", MANIFESTS / "ce_L6_obc_positive_pilot_beta5_6p7_10_r32_m10000.tsv"] + sorted(MANIFESTS.glob("ce_L6_obc_positive_admitted_r*_m*.tsv")), "ce", "outdir"),
        (sorted(MANIFESTS.glob("gce_mu_probe_L6_obc_*.tsv")), "probe", "out_parent"),
        (sorted(MANIFESTS.glob("gce_prod_L6_obc_*_confirmed.tsv")), "production", "out_parent"),
    ]
    for paths, kind, root_field in specs:
        for path in paths:
            if not path.is_file(): continue
            for row in read(path):
                root = row.get(root_field, "")
                if not root: continue
                if row.get("boundary") != "open" or "_obc_" not in root:
                    raise SystemExit(f"non-OBC authoritative row {root} in {path}")
                candidate = {**row, "_kind": kind, "_root_field": root_field, "_source": str(path)}
                prior = records.get(root)
                if prior is not None and prior != candidate:
                    raise SystemExit(f"conflicting duplicate authoritative root {root}")
                records[root] = candidate
    return records


def known_job_elements(records: dict[str, dict[str, str]]) -> dict[tuple[str, str], str]:
    mapping: dict[tuple[str, str], str] = {}
    for ledger in (STATUS / "submission_ledger.tsv", STATUS / "repair_ledger.tsv"):
        if not ledger.is_file(): continue
        for row in read(ledger):
            job = row.get("job_id", ""); task = row.get("array_task", ""); root = row.get("root", "")
            if job and task and root:
                mapping[(job, task)] = root
    for root in records:
        ledger = Path(root) / ".l6_obc_continuation_ledger.tsv"
        if not ledger.is_file(): continue
        for row in read(ledger):
            job = row.get("job_id", ""); task = row.get("array_task", "")
            if job and task:
                prior = mapping.get((job, task))
                if prior is not None and prior != root:
                    raise SystemExit(f"conflicting continuation mapping job={job}_{task}")
                mapping[(job, task)] = root
    return mapping


def active_roots(records: dict[str, dict[str, str]]) -> set[str]:
    mapping = known_job_elements(records)
    user = subprocess.check_output(["id", "-un"], text=True).strip()
    output = subprocess.check_output(
        ["squeue", "-r", "-h", "-u", user, "-o", "%F|%K|%j|%T|%a|%P|%q"], text=True
    )
    roots: list[str] = []; unknown: list[str] = []
    for line in output.splitlines():
        base, task, name, state, account, partition, qos = line.split("|", 6)
        if name not in JOB_NAMES: continue
        root = mapping.get((base, task))
        if root is None:
            unknown.append(f"{base}_{task} {name} {state} {account}/{partition}/{qos}")
            continue
        authoritative = records.get(root)
        if authoritative is None:
            unknown.append(f"{base}_{task} maps to non-authoritative root {root}")
            continue
        expected = (authoritative.get("account", "ccsd"), authoritative.get("partition", "burst"), authoritative.get("qos", "default"))
        if (account, partition, qos) != expected:
            unknown.append(f"{base}_{task} resource drift {(account,partition,qos)} != {expected} root={root}")
            continue
        roots.append(root)
    if unknown:
        raise SystemExit("refusing repair; active workflow jobs could not be mapped:\n  " + "\n  ".join(unknown))
    duplicates = [root for root, count in Counter(roots).items() if count > 1]
    if duplicates:
        raise SystemExit("duplicate active roots detected:\n  " + "\n  ".join(duplicates))
    return set(roots)


def confirmed_targets() -> set[str]:
    path = STATUS / "mu_tuning" / "mu_target_status.tsv"
    if not path.is_file(): return set()
    return {row["target_key"] for row in read(path) if row.get("status") == "confirmed_within_abs_N_0p03"}


def continuation_count(root: str) -> int:
    path = Path(root) / ".l6_obc_continuation_ledger.tsv"
    values = [int(row["continuation"]) for row in read(path) if row.get("continuation", "").isdigit()] if path.is_file() else []
    repairs = STATUS / "repair_ledger.tsv"
    if repairs.is_file():
        values.extend(
            int(row["continuation_count"])
            for row in read(repairs)
            if row.get("root") == root and row.get("continuation_count", "").isdigit()
        )
    return max(values, default=0)


def config(row: dict[str, str]) -> tuple[str, Path, int]:
    kind = row["_kind"]
    if kind == "ce":
        ranks = int(row["expected_ranks"])
        stage = row["stage"]
        if stage == "attractive": name = "ceL6OAtr"
        elif stage == "attractive_beta20": name = "ceL6OA20"
        elif stage == "positive_lowbeta": name = "ceL6OPos"
        elif stage == "positive_pilot": name = "ceL6OPil"
        elif ranks == 32: name = "ceL6OPN"
        else: name = "ceL6OPH"
        wrapper = HERE / ("job_ce_L6_r32_cades.sbatch" if ranks == 32 else "job_ce_L6_r64_cades.sbatch")
        return name, wrapper, 120
    family = row["family"]
    if kind == "probe":
        return (
            "muL6OAttr" if family == "attractive" else "muL6OSpin",
            HERE / ("job_gce_mu_L6_attractive_cades.sbatch" if family == "attractive" else "job_gce_mu_L6_spinHS_cades.sbatch"),
            80,
        )
    return (
        "gceL6OAttr" if family == "attractive" else "gceL6OSpin",
        HERE / ("job_gce_prod_L6_attractive_cades.sbatch" if family == "attractive" else "job_gce_prod_L6_spinHS_cades.sbatch"),
        120,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-age-hours", type=float, default=72.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if os.environ.get("USER") != "9pm" or "or-" not in socket.gethostname().lower():
        raise SystemExit("run as 9pm on CADES through lowercase SSH host cades")
    STATUS.mkdir(parents=True, exist_ok=True)
    with (STATUS / ".repair_stale_l6_obc.lock").open("w") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("SKIP: another L6 OBC repair audit is running"); return
        records = authoritative_rows(); active = active_roots(records)
        confirmed = confirmed_targets(); cutoff = time.time() - args.max_age_hours * 3600
        eligible: list[tuple[str, dict[str, str], str]] = []; blocked = Counter()
        for root, row in records.items():
            if root in active: continue
            if row["_kind"] == "probe" and row["target_key"] in confirmed: continue
            if row["_kind"] == "ce": ok, reason = ce_eligible(row, cutoff)
            elif row["_kind"] == "probe": ok, reason = probe_eligible(row, cutoff)
            else: ok, reason = production_eligible(row, cutoff)
            if ok: eligible.append((root, row, reason))
            elif reason != "strict_final": blocked[reason] += 1
        print(f"REPAIR_AUDIT required_roots={len(records)} active_roots={len(active)} eligible={len(eligible)} blocked={dict(blocked)}")
        if not eligible: return

        groups: dict[tuple[str, str, str, str, int], list[tuple[str, dict[str, str], str]]] = defaultdict(list)
        for item in eligible:
            root, row, _ = item; name, wrapper, cap = config(row)
            next_count = continuation_count(root) + 1
            groups[(name, str(wrapper), row.get("account", "ccsd"), row.get("partition", "burst"), next_count)].append(item)
        prior = read(STATUS / "repair_ledger.tsv") if (STATUS / "repair_ledger.tsv").is_file() else []
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        for (name, wrapper_text, account, partition, next_count), items in sorted(groups.items()):
            cap = config(items[0][1])[2]
            if next_count > cap:
                print(f"BLOCK continuation_cap root={items[0][0]} next={next_count}/{cap}"); continue
            if account not in {"ccsd", "cnms"} or partition != "burst" or any(row.get("qos", "default") != "default" for _, row, _ in items):
                raise SystemExit(f"invalid repair resources {account}/{partition}/default")
            rows = [dict(row) for _, row, _ in items]
            for idx, row in enumerate(rows):
                row.pop("_kind", None); row.pop("_root_field", None); row.pop("_source", None); row["idx"] = str(idx)
            delta = STATUS / "repair_manifests" / f"{name}_repair_{stamp}_{next_count}.tsv"
            write(delta, rows, rows[0].keys())
            command = [
                "sbatch", "--parsable", "-A", account, "-p", "burst", "--qos=default",
                f"--job-name={name}", f"--array=0-{len(rows)-1}", wrapper_text,
            ]
            print("REPAIR_SUBMIT", " ".join(command), f"MANIFEST={delta} continuation={next_count}/{cap}")
            if args.dry_run: continue
            env = dict(os.environ); env.update({
                "MANIFEST": str(delta), "RESUBMIT_COUNT": str(next_count),
                "CHECKPOINT_RESET_ACCUMULATORS": "false", "AUTO_RESUBMIT": "true",
            })
            job_id = subprocess.check_output(command, text=True, env=env).strip().split(";")[0]
            now = dt.datetime.now(dt.timezone.utc).isoformat()
            for task, (root, row, reason) in enumerate(items):
                prior.append({
                    "timestamp_utc": now, "job_name": name, "job_id": job_id,
                    "array_task": str(task), "continuation_count": str(next_count),
                    "source_manifest": row["_source"], "delta_manifest": str(delta),
                    "root": root, "reason": reason, "account": account,
                    "partition": "burst", "qos": "default",
                })
            write(STATUS / "repair_ledger.tsv", prior, REPAIR_FIELDS)
            print(f"REPAIRED job={job_id} rows={len(rows)} name={name} continuation={next_count}/{cap}")


if __name__ == "__main__":
    main()
