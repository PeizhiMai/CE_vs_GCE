#!/usr/bin/env python3
"""Classify L=6 OBC positive-U CE pilots and emit fresh production rows.

Pilot samples are used only for the sign/phase gate and are never merged into
production.  Every accepted pilot is revalidated from the four direct OBC
primary tables and from the global signed-numerator/phase-denominator pool.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable

PRIMARY = (
    "equal_time_kinetic_per_site_qmc.tsv",
    "equal_time_double_occupancy_per_site_qmc.tsv",
    "equal_time_nn_spin_qmc.tsv",
    "equal_time_nn_connected_charge_qmc.tsv",
)
RUN_BASE = Path("/home/9pm/nUHubbard_obc_runs/ce_L6_obc_thermometry_20260720")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_one(path: Path) -> dict[str, str]:
    rows = read_rows(path)
    if len(rows) != 1:
        raise ValueError(f"{path}: expected one row, found {len(rows)}")
    return rows[0]


def write(path: Path, rows: Iterable[dict[str, object]], fields: Iterable[str]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, delimiter="\t", fieldnames=list(fields), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(materialized)


def stable_int(label: str, base: int = 1_800_000_000) -> int:
    return base + int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big") % 80_000_000


def beta_label(beta: float) -> str:
    return f"{beta:.1f}".replace(".", "p")


def finite(value: object) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite value {value!r}")
    return result


def inspect(row: dict[str, str]) -> dict[str, object]:
    root = Path(row["outdir"])
    expected = int(row["expected_ranks"])
    rank_complete = len(list(root.glob("ranks/rank_*/checkpoint_complete.txt")))
    rank_sites = len(list(root.glob("ranks/rank_*/equal_time_site_density_qmc.tsv")))
    table_count = sum((root / name).is_file() for name in PRIMARY)
    strict = (
        (root / "obc_thermometry_complete.txt").is_file()
        and rank_complete == expected
        and rank_sites == expected
        and table_count == 4
    )
    problem = ""
    avg = sem = pooled = math.nan
    rank_phase: list[float] = []
    nsamples = 0.0
    phase_pooling_valid = False
    primary_finite = False
    try:
        combined = [read_one(root / name) for name in PRIMARY]
        for item in combined:
            if item.get("boundary") != "open":
                raise ValueError("non-OBC primary table")
            if int(float(item["nranks"])) != expected:
                raise ValueError("combined primary wrong-rank count")
            finite(item["value"]); finite(item["stderr"])
        kinetic = combined[0]
        nsamples = finite(kinetic["nsamples"])
        phase_sum = finite(kinetic["phase_sum"])
        avg = finite(kinetic["average_phase"])
        if nsamples <= 0:
            raise ValueError("nonpositive pilot sample count")
        if kinetic.get("phase_reweighted", "").lower() != "true":
            raise ValueError("positive-U pilot not phase reweighted")
        pooled_phase_sum = 0.0
        pooled_samples = 0.0
        for path in sorted(root.glob("ranks/rank_*/equal_time_kinetic_per_site_qmc.tsv")):
            item = read_one(path)
            n = finite(item["nsamples"])
            p = finite(item["phase_sum"])
            if item.get("boundary") != "open" or item.get("phase_reweighted", "").lower() != "true" or n <= 0:
                raise ValueError(f"invalid rank pilot table {path}")
            rank_phase.append(p / n)
            pooled_phase_sum += p
            pooled_samples += n
        if len(rank_phase) != expected:
            raise ValueError(f"rank phase coverage {len(rank_phase)}/{expected}")
        pooled = pooled_phase_sum / pooled_samples
        if not math.isclose(nsamples, pooled_samples, rel_tol=0, abs_tol=0.5):
            raise ValueError("combined nsamples != summed rank nsamples")
        if not math.isclose(avg, phase_sum / nsamples, rel_tol=2e-10, abs_tol=2e-12):
            raise ValueError("average_phase != phase_sum/nsamples")
        if not math.isclose(avg, pooled, rel_tol=2e-9, abs_tol=2e-11):
            raise ValueError("combined phase != global rank phase pool")
        sem = statistics.stdev(rank_phase) / math.sqrt(expected) if expected > 1 else 0.0
        phase_pooling_valid = True
        primary_finite = True
    except (OSError, KeyError, ValueError, StopIteration) as exc:
        problem = str(exc)

    abs_phase = abs(avg) if math.isfinite(avg) else math.nan
    if not strict:
        tier = "unfinished"
    elif not phase_pooling_valid:
        tier = "invalid_phase_pooling_or_rank_coverage"
    elif not primary_finite:
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
        "rank_site_coverage": rank_sites,
        "primary_table_coverage": f"{table_count}/4",
        "rank_phase_coverage": len(rank_phase),
        "average_phase": avg,
        "abs_average_phase": abs_phase,
        "pooled_rank_average_phase": pooled,
        "phase_sem_rank": sem,
        "phase_95ci_low": avg - 1.96 * sem if math.isfinite(sem) else math.nan,
        "phase_95ci_high": avg + 1.96 * sem if math.isfinite(sem) else math.nan,
        "nsamples_total": nsamples,
        "phase_pooling_valid": int(phase_pooling_valid),
        "primary_observables_finite": int(primary_finite),
        "problem": problem,
        "tier": tier,
    }


def production_row(pilot: dict[str, object], idx: int, ranks: int, measurements: int, tier: str) -> dict[str, object]:
    result = {
        key: value for key, value in pilot.items()
        if key not in {
            "strict_final", "rank_complete", "rank_site_coverage", "primary_table_coverage",
            "rank_phase_coverage", "average_phase", "abs_average_phase",
            "pooled_rank_average_phase", "phase_sem_rank", "phase_95ci_low",
            "phase_95ci_high", "nsamples_total", "phase_pooling_valid",
            "primary_observables_finite", "problem", "tier",
        }
    }
    beta = float(pilot["beta"])
    ntot = int(pilot["Ntot"])
    ulabel = str(pilot["U_label"])
    label = beta_label(beta)
    stage = f"positive_admitted_{tier}"
    seed = stable_int(f"L6OBC:CE:production:{ulabel}:{ntot}:{beta}:{tier}:{ranks}:{measurements}")
    root = (
        RUN_BASE / stage / ulabel /
        f"Ntot{ntot:03d}_n{float(pilot['density']):.6f}_b{label}_T{float(pilot['actual_T']):.6f}"
        f"_Nup{int(pilot['Nup']):03d}_Ndn{int(pilot['Ndn']):03d}_r{ranks}_w5000_m{measurements}_seed{seed}_obc_prod"
    )
    result.update({
        "idx": idx,
        "stage": stage,
        "job_tag": f"ceL6O{ulabel}N{ntot}b{label}{'H' if ranks == 64 else 'N'}",
        "outdir": str(root),
        "seed": seed,
        "nwarmups": 5000,
        "max_batches": measurements,
        "batch_nsamples": 1,
        "expected_ranks": ranks,
        "phase_reweighted": "true",
        "force_symmetry": "false",
        "boundary": "open",
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--write-production", action="store_true")
    args = parser.parse_args()

    source = read_rows(args.manifest)
    if len(source) != 24:
        raise SystemExit(f"expected 24 positive-U OBC pilots, found {len(source)}")
    pilots = [inspect(row) for row in source]
    write(args.out, pilots, pilots[0].keys())
    counts = Counter(str(row["tier"]) for row in pilots)
    print(f"PILOTS final={sum(int(row['strict_final']) for row in pilots)}/24 tiers={dict(counts)}")
    if not args.write_production:
        return
    if any(row["tier"] == "unfinished" for row in pilots):
        raise SystemExit("not writing production manifests until all 24 pilots are strict-final")
    invalid = [row for row in pilots if str(row["tier"]).startswith("invalid")]
    if invalid:
        raise SystemExit(f"not writing production manifests: {len(invalid)} invalid pilots")

    r32: list[dict[str, object]] = []
    r64: list[dict[str, object]] = []
    limited: list[dict[str, object]] = []
    for row in pilots:
        tier = str(row["tier"])
        if tier == "sign_limited":
            limited.append(row)
        elif float(row["beta"]) <= 5.0 + 1e-12 and tier == "normal":
            r32.append(production_row(row, len(r32), 32, 50_000, tier))
        else:
            r64.append(production_row(row, len(r64), 64, 30_000, tier))

    manifest_fields = list(source[0].keys())
    write(args.manifest_dir / "ce_L6_obc_positive_admitted_r32_m50000.tsv", r32, manifest_fields)
    write(args.manifest_dir / "ce_L6_obc_positive_admitted_r64_m30000.tsv", r64, manifest_fields)
    write(args.manifest_dir / "ce_L6_obc_positive_sign_limited.tsv", limited, pilots[0].keys())
    print(f"PRODUCTION admitted_r32={len(r32)} admitted_r64={len(r64)} sign_limited={len(limited)}")


if __name__ == "__main__":
    main()
