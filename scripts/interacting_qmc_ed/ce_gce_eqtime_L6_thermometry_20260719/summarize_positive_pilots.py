#!/usr/bin/env python3.11
"""Classify L=6 positive-U CE sign pilots and emit admitted production rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
from collections import Counter
from pathlib import Path


CE_FIELDS = [
    "idx", "stage", "U_label", "U", "beta", "target_T", "actual_T", "Ltau",
    "Ntot", "Nup", "Ndn", "density", "account", "job_tag", "outdir", "seed",
    "nwarmups", "max_batches", "batch_nsamples", "measure_interval", "cluster_size",
    "Lx", "Ly", "dtau", "expected_ranks", "phase_reweighted", "force_symmetry",
]
TABLES = [
    "equal_time_observables_qmc.tsv",
    "equal_time_charge_spin_wedge_qmc.tsv",
    "equal_time_structure_factors_qmc.tsv",
    "equal_time_neighbor_shells_qmc.tsv",
]


def stable_int(label: str, base: int) -> int:
    return base + int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big") % 80_000_000


def beta_label(beta: float) -> str:
    return f"{beta:.1f}".replace(".", "p")


def read_first(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        return next(csv.DictReader(f, delimiter="\t"))


def all_finite_primary(row: dict[str, str]) -> bool:
    required = [
        "kinetic_per_site", "kinetic_stderr",
        "double_occupancy_per_site", "double_occupancy_stderr",
    ]
    return all(k in row and math.isfinite(float(row[k])) for k in required)


def inspect(row: dict[str, str]) -> dict[str, object]:
    root = Path(row["outdir"])
    rank_complete = len(list(root.glob("ranks/rank_*/checkpoint_complete.txt")))
    strict = (
        (root / "checkpoint_complete.txt").is_file()
        and rank_complete == int(row["expected_ranks"])
        and all((root / name).is_file() for name in TABLES)
    )
    avg = sem = math.nan
    rank_phase: list[float] = []
    global_finite = False
    phase_pooling_valid = False
    nsamples = 0
    if (root / TABLES[0]).is_file():
        q = read_first(root / TABLES[0])
        avg = float(q.get("average_phase", "nan"))
        nsamples = int(float(q.get("nsamples", 0)))
        phase_pooling_valid = q.get("phase_reweighted", "").lower() == "true" and "phase_sum" in q
        global_finite = all_finite_primary(q)
    for p in sorted(root.glob("ranks/rank_*/equal_time_observables_qmc.tsv")):
        try:
            q = read_first(p)
            n = float(q["nsamples"])
            if n > 0 and q.get("phase_reweighted", "").lower() == "true":
                rank_phase.append(float(q["phase_sum"]) / n)
        except (OSError, KeyError, ValueError, StopIteration):
            pass
    if len(rank_phase) >= 2:
        sem = statistics.stdev(rank_phase) / math.sqrt(len(rank_phase))
    abs_phase = abs(avg) if math.isfinite(avg) else math.nan
    if not strict:
        tier = "unfinished"
    elif not phase_pooling_valid or len(rank_phase) != int(row["expected_ranks"]):
        tier = "invalid_phase_pooling_or_rank_coverage"
    elif not global_finite:
        tier = "invalid_nonfinite_equal_time"
    elif abs_phase >= 0.02:
        tier = "normal"
    elif abs_phase >= 0.002:
        tier = "high_stat"
    else:
        tier = "sign_limited"
    return {
        **row,
        "strict_final": int(strict),
        "rank_complete": rank_complete,
        "rank_phase_coverage": len(rank_phase),
        "average_phase": avg,
        "abs_average_phase": abs_phase,
        "phase_sem_rank": sem,
        "phase_95ci_low": avg - 1.96 * sem if math.isfinite(sem) else math.nan,
        "phase_95ci_high": avg + 1.96 * sem if math.isfinite(sem) else math.nan,
        "nsamples_total": nsamples,
        "phase_pooling_valid": int(phase_pooling_valid),
        "primary_observables_finite": int(global_finite),
        "tier": tier,
    }


def production_row(pilot: dict[str, object], idx: int, ranks: int, measurements: int, tier: str) -> dict[str, object]:
    beta = float(pilot["beta"]); ulabel = str(pilot["U_label"]); ntot = int(pilot["Ntot"])
    label = beta_label(beta)
    stage = f"positive_admitted_{tier}"
    seed = stable_int(f"L6CEprod:{ulabel}:{ntot}:{beta}:{tier}", 1_800_000_000)
    root = (
        f"/home/9pm/nUHubbard/runs/ce_eqtime_L6_thermometry_20260719/{stage}/{ulabel}/"
        f"Ntot{ntot:03d}_n{float(pilot['density']):.6f}_b{label}_T{float(pilot['actual_T']):.6f}"
        f"_Nup{int(pilot['Nup']):03d}_Ndn{int(pilot['Ndn']):03d}_r{ranks}_w5000_m{measurements}_seed{seed}"
    )
    return {
        "idx": idx, "stage": stage, "U_label": ulabel, "U": pilot["U"],
        "beta": pilot["beta"], "target_T": pilot["target_T"], "actual_T": pilot["actual_T"],
        "Ltau": pilot["Ltau"], "Ntot": ntot, "Nup": pilot["Nup"], "Ndn": pilot["Ndn"],
        "density": pilot["density"], "account": "ccsd",
        "job_tag": f"ceL6{ulabel}N{ntot}b{label}{'Hi' if ranks == 64 else 'Norm'}",
        "outdir": root, "seed": seed, "nwarmups": 5000, "max_batches": measurements,
        "batch_nsamples": 1, "measure_interval": 3, "cluster_size": 36,
        "Lx": 6, "Ly": 6, "dtau": "0.1", "expected_ranks": ranks,
        "phase_reweighted": "true", "force_symmetry": "false",
    }


def write(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w=csv.DictWriter(f, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest-dir", type=Path, required=True)
    ap.add_argument("--write-production", action="store_true")
    args=ap.parse_args()
    with args.manifest.open(newline="") as f:
        pilots=[inspect(r) for r in csv.DictReader(f, delimiter="\t")]
    fields=list(pilots[0].keys())
    write(args.out, pilots, fields)
    print(f"PILOTS final={sum(int(r['strict_final']) for r in pilots)}/{len(pilots)} tiers={dict(Counter(str(r['tier']) for r in pilots))}")
    if not args.write_production:
        return
    if any(r["tier"] == "unfinished" for r in pilots):
        raise SystemExit("not writing production manifests until all pilots are strict-final")
    invalid=[r for r in pilots if str(r["tier"]).startswith("invalid")]
    if invalid:
        raise SystemExit(f"not writing production manifests: {len(invalid)} invalid pilot rows")
    r32=[]; r64=[]; limited=[]
    for r in pilots:
        tier=str(r["tier"])
        if tier == "sign_limited":
            limited.append(r); continue
        beta=float(r["beta"])
        # beta=5 normal-tier rows use the standard 32x50k design.  Every
        # high-stat row and all admitted beta=6.7/10 rows use 64x30k.
        if beta <= 5.0 + 1e-12 and tier == "normal":
            r32.append(production_row(r, len(r32), 32, 50_000, tier))
        else:
            r64.append(production_row(r, len(r64), 64, 30_000, tier))
    write(args.manifest_dir / "ce_L6_positive_admitted_r32_m50000.tsv", r32, CE_FIELDS)
    write(args.manifest_dir / "ce_L6_positive_admitted_r64_m30000.tsv", r64, CE_FIELDS)
    write(args.manifest_dir / "ce_L6_positive_sign_limited.tsv", limited, fields)
    print(f"PRODUCTION admitted_r32={len(r32)} admitted_r64={len(r64)} sign_limited={len(limited)}")


if __name__ == "__main__":
    main()
