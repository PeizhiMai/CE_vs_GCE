#!/usr/bin/env python3
"""Collect only strict-final L=6 CE/GCE equal-time rows.

The collector is intentionally conservative.  It accepts a canonical row only
after all expected rank completion markers and all four combined tables exist.
It accepts a grand-canonical row only after the tuned production manifest,
32-rank metadata coverage, four exported tables, and the achieved-density
tolerance marker exist.  Positive-U canonical rows additionally recheck the
global signed-numerator/phase-denominator pool against the rank outputs.

Run this on CADES, where the manifest roots are directly visible.  Exact U=0
rows are read from the deterministic snapshot copied into status_source.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable


HERE = Path(__file__).resolve().parent
MANIFESTS = HERE / "manifests"
TABLES = (
    "equal_time_observables_qmc.tsv",
    "equal_time_charge_spin_wedge_qmc.tsv",
    "equal_time_structure_factors_qmc.tsv",
    "equal_time_neighbor_shells_qmc.tsv",
)
SNAPSHOT_FIELDS = (
    "L", "U", "ensemble", "Ntot", "target_density", "beta", "T",
    "N_mean", "density", "kinetic", "kinetic_err", "double_occupancy",
    "double_occupancy_err", "nn_spin", "nn_spin_err",
    "nn_charge_connected", "nn_charge_connected_err", "nnn_spin",
    "nnn_spin_err", "nnn_charge_connected", "nnn_charge_connected_err",
    "nsamples", "batches", "nranks", "average_phase", "average_phase_err",
    "mu", "mu_L8_reference", "L8_reference_Ntot", "final", "sign_limited",
    "source", "workflow",
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_first(path: Path) -> dict[str, str]:
    rows = read_tsv(path)
    if not rows:
        raise ValueError(f"empty TSV: {path}")
    return rows[0]


def write_tsv(path: Path, rows: Iterable[dict[str, object]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def finite(value: object) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"non-finite value: {value!r}")
    return x


def key(u: object, ntot: object, beta: object) -> tuple[float, int, float]:
    return (round(float(u), 10), int(ntot), round(float(beta), 10))


def phase_sem_from_ranks(root: Path, expected: int) -> tuple[float, int, float, float]:
    phases: list[float] = []
    phase_sum = 0.0
    nsamples = 0.0
    rank_files = sorted(root.glob("ranks/rank_*/equal_time_observables_qmc.tsv"))
    for path in rank_files:
        row = read_first(path)
        n = finite(row["nsamples"])
        p = finite(row["phase_sum"])
        if n <= 0:
            raise ValueError(f"{path}: nonpositive nsamples")
        phases.append(p / n)
        phase_sum += p
        nsamples += n
    if len(phases) != expected:
        raise ValueError(f"{root}: rank phase coverage {len(phases)}/{expected}")
    sem = statistics.stdev(phases) / math.sqrt(len(phases)) if len(phases) > 1 else 0.0
    return phase_sum / nsamples, len(phases), sem, nsamples


def shell_values(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    rows = {int(r["shell"]): r for r in read_tsv(path)}
    if 1 not in rows or 2 not in rows:
        raise ValueError(f"{path}: shell 1/2 missing")
    return rows[1], rows[2]


def inspect_ce(row: dict[str, str]) -> tuple[dict[str, object] | None, dict[str, object]]:
    root = Path(row["outdir"])
    expected = int(row["expected_ranks"])
    rank_complete = len(list(root.glob("ranks/rank_*/checkpoint_complete.txt")))
    table_count = sum((root / name).is_file() for name in TABLES)
    strict = (root / "checkpoint_complete.txt").is_file() and rank_complete == expected and table_count == 4
    status = {
        "root": str(root), "rank_coverage": f"{rank_complete}/{expected}",
        "table_coverage": f"{table_count}/4", "strict_final": int(strict), "problem": "",
    }
    if not strict:
        return None, status
    try:
        scalar = read_first(root / TABLES[0])
        nn, nnn = shell_values(root / TABLES[3])
        if int(float(scalar.get("nranks", expected))) != expected:
            raise ValueError(f"combined nranks={scalar.get('nranks')} expected={expected}")
        u = float(row["U"])
        avg_phase = finite(scalar.get("average_phase", 1.0))
        phase_err = 0.0
        if u > 0:
            if scalar.get("phase_reweighted", "").lower() != "true":
                raise ValueError("positive-U combined table is not phase reweighted")
            if "phase_sum" not in scalar:
                raise ValueError("positive-U combined table lacks phase_sum")
            combined_ratio = finite(scalar["phase_sum"]) / finite(scalar["nsamples"])
            pooled, coverage, phase_err, rank_nsamples = phase_sem_from_ranks(root, expected)
            if coverage != expected:
                raise ValueError(f"phase rank coverage={coverage}/{expected}")
            if not math.isclose(avg_phase, combined_ratio, rel_tol=2e-10, abs_tol=2e-12):
                raise ValueError("average_phase != combined phase_sum/nsamples")
            if not math.isclose(avg_phase, pooled, rel_tol=2e-9, abs_tol=2e-11):
                raise ValueError("combined average_phase != global rank phase-sum pool")
            if not math.isclose(finite(scalar["nsamples"]), rank_nsamples, rel_tol=0, abs_tol=0.5):
                raise ValueError("combined nsamples != summed rank nsamples")
        out = {
            "L": 6, "U": u, "ensemble": "CE", "Ntot": int(row["Ntot"]),
            "target_density": float(row["density"]), "beta": float(row["beta"]),
            "T": 1.0 / float(row["beta"]), "N_mean": finite(scalar["ntotal"]),
            "density": finite(scalar["density"]), "kinetic": finite(scalar["kinetic_per_site"]),
            "kinetic_err": finite(scalar["kinetic_stderr"]),
            "double_occupancy": finite(scalar["double_occupancy_per_site"]),
            "double_occupancy_err": finite(scalar["double_occupancy_stderr"]),
            "nn_spin": finite(nn["spin_corr_s_s"]), "nn_spin_err": finite(nn["spin_corr_s_s_stderr"]),
            "nn_charge_connected": finite(nn["charge_corr_connected"]),
            "nn_charge_connected_err": finite(nn["charge_corr_connected_stderr"]),
            "nnn_spin": finite(nnn["spin_corr_s_s"]), "nnn_spin_err": finite(nnn["spin_corr_s_s_stderr"]),
            "nnn_charge_connected": finite(nnn["charge_corr_connected"]),
            "nnn_charge_connected_err": finite(nnn["charge_corr_connected_stderr"]),
            "nsamples": finite(scalar["nsamples"]), "batches": finite(scalar["batches"]),
            "nranks": expected, "average_phase": avg_phase, "average_phase_err": phase_err,
            "mu": "", "mu_L8_reference": "", "L8_reference_Ntot": "", "final": 1,
            "sign_limited": 0, "source": str(root), "workflow": "L6_CE_QMC_20260719",
        }
        return out, status
    except Exception as exc:  # keep the audit table useful instead of silently accepting a bad row
        status["strict_final"] = 0
        status["problem"] = str(exc)
        return None, status


def inspect_gce(row: dict[str, str]) -> tuple[dict[str, object] | None, dict[str, object]]:
    root = Path(row["out_parent"])
    expected = int(row["expected_ranks"])
    export = root / "export"
    table_count = sum((export / name).is_file() for name in TABLES)
    complete_dirs = [p for p in root.glob("complete_*") if p.is_dir()]
    rank_info = sum(len(list(p.glob("simulation_info_sID-*_pID-*.toml"))) for p in complete_dirs)
    achieved_path = root / "achieved_density.tsv"
    marker = root / "dqmc_gce_eqtime_complete.txt"
    strict = marker.is_file() and table_count == 4 and rank_info == expected and achieved_path.is_file()
    status = {
        "root": str(root), "rank_coverage": f"{rank_info}/{expected}",
        "table_coverage": f"{table_count}/4", "strict_final": int(strict), "problem": "",
    }
    if not strict:
        return None, status
    try:
        achieved = read_first(achieved_path)
        achieved_n = finite(achieved["achieved_N"])
        tol = finite(achieved["tolerance"])
        target = int(row["Ntot_target"])
        if abs(achieved_n - target) > tol + 1e-12 or tol > 0.0300000001:
            raise ValueError(f"achieved N={achieved_n} outside target={target} tolerance={tol}")
        scalar = read_first(export / TABLES[0])
        nn, nnn = shell_values(export / TABLES[3])
        out = {
            "L": 6, "U": float(row["U"]), "ensemble": "GCE", "Ntot": target,
            "target_density": float(row["target_density"]), "beta": float(row["beta"]),
            "T": 1.0 / float(row["beta"]), "N_mean": achieved_n,
            "density": finite(scalar["density"]), "kinetic": finite(scalar["kinetic_per_site"]),
            "kinetic_err": finite(scalar["kinetic_stderr"]),
            "double_occupancy": finite(scalar["double_occupancy_per_site"]),
            "double_occupancy_err": finite(scalar["double_occupancy_stderr"]),
            "nn_spin": finite(nn["spin_corr_s_s"]), "nn_spin_err": finite(nn["spin_corr_s_s_stderr"]),
            "nn_charge_connected": finite(nn["charge_corr_connected"]),
            "nn_charge_connected_err": finite(nn["charge_corr_connected_stderr"]),
            "nnn_spin": finite(nnn["spin_corr_s_s"]), "nnn_spin_err": finite(nnn["spin_corr_s_s_stderr"]),
            "nnn_charge_connected": finite(nnn["charge_corr_connected"]),
            "nnn_charge_connected_err": finite(nnn["charge_corr_connected_stderr"]),
            "nsamples": finite(scalar.get("nsamples", row["nmeasurements"])),
            "batches": finite(scalar.get("batches", row["nbins"])), "nranks": expected,
            "average_phase": float(scalar.get("average_sign", "nan")),
            "average_phase_err": float(scalar.get("average_sign_stderr", "nan")),
            "mu": finite(row["mu_final"]), "mu_L8_reference": finite(row["mu_L8_reference"]),
            "L8_reference_Ntot": int(row["L8_reference_Ntot"]), "final": 1,
            "sign_limited": 0, "source": str(root), "workflow": "L6_GCE_QMC_mu_tuned_20260719",
        }
        return out, status
    except Exception as exc:
        status["strict_final"] = 0
        status["problem"] = str(exc)
        return None, status


def load_manifest_paths(names: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for name in names:
        paths.extend(sorted(MANIFESTS.glob(name)))
    return paths


def indexed_rows(paths: Iterable[Path], root_field: str, key_fields: tuple[str, str, str]) -> dict[tuple[float, int, float], dict[str, str]]:
    out: dict[tuple[float, int, float], dict[str, str]] = {}
    roots: set[str] = set()
    for path in paths:
        for row in read_tsv(path):
            root = row[root_field]
            if root in roots:
                raise ValueError(f"duplicate run root in accepted manifests: {root}")
            roots.add(root)
            k = key(row[key_fields[0]], row[key_fields[1]], row[key_fields[2]])
            if k in out:
                raise ValueError(f"duplicate physical condition in accepted manifests: {k}")
            row = {**row, "_manifest": str(path)}
            out[k] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exact-snapshot", type=Path, default=HERE / "status_source" / "exact_u0_L6_snapshot.tsv")
    ap.add_argument("--snapshot", type=Path, default=HERE / "status_source" / "analysis" / "l6_snapshot_current.tsv")
    ap.add_argument("--status", type=Path, default=HERE / "status_source" / "analysis" / "l6_condition_status.tsv")
    ap.add_argument("--require-complete", action="store_true")
    args = ap.parse_args()

    grid = read_tsv(MANIFESTS / "L6_full_condition_grid.tsv")
    if len(grid) != 192:
        raise SystemExit(f"expected 192 condition rows, found {len(grid)}")
    exact = read_tsv(args.exact_snapshot)
    if len(exact) != 80:
        raise SystemExit(f"expected 80 exact ensemble rows, found {len(exact)}")

    ce_paths = load_manifest_paths((
        "ce_L6_attractive_beta_le10_r32_m50000.tsv",
        "ce_L6_attractive_beta20_r64_m30000.tsv",
        "ce_L6_positive_beta_le4_r32_m50000.tsv",
        "ce_L6_positive_admitted_r32_m50000.tsv",
        "ce_L6_positive_admitted_r64_m30000.tsv",
    ))
    ce = indexed_rows(ce_paths, "outdir", ("U", "Ntot", "beta"))
    gce_paths = load_manifest_paths(("gce_prod_L6_attractive_confirmed.tsv", "gce_prod_L6_spinHS_confirmed.tsv"))
    gce = indexed_rows(gce_paths, "out_parent", ("U", "Ntot_target", "beta"))

    sign_limited: set[tuple[float, int, float]] = set()
    limited_path = MANIFESTS / "ce_L6_positive_sign_limited.tsv"
    if limited_path.is_file():
        sign_limited = {key(r["U"], r["Ntot"], r["beta"]) for r in read_tsv(limited_path)}

    snapshots: list[dict[str, object]] = []
    for row in exact:
        snapshots.append({k: row.get(k, "") for k in SNAPSHOT_FIELDS})
    statuses: list[dict[str, object]] = []
    status_counts: Counter[str] = Counter()
    for condition in grid:
        k = key(condition["U"], condition["Ntot"], condition["beta"])
        u = float(condition["U"])
        for ensemble in ("CE", "GCE"):
            base: dict[str, object] = {
                "U": u, "Ntot": int(condition["Ntot"]), "beta": float(condition["beta"]),
                "T": 1.0 / float(condition["beta"]), "ensemble": ensemble,
                "target_density": float(condition["density"]), "status": "", "root": "",
                "rank_coverage": "", "table_coverage": "", "strict_final": 0,
                "sign_limited": int(ensemble == "CE" and k in sign_limited), "problem": "",
            }
            if u == 0:
                base.update(status="exact_final", strict_final=1, rank_coverage="exact", table_coverage="4/4")
            elif ensemble == "CE" and k in sign_limited:
                base.update(status="sign_limited", problem="abs_average_phase_below_0p002")
            elif ensemble == "CE" and k in ce:
                result, audit = inspect_ce(ce[k])
                base.update(audit)
                base["status"] = "strict_final" if result else ("invalid" if audit["problem"] else "incomplete")
                if result:
                    snapshots.append(result)
            elif ensemble == "GCE" and k in gce:
                result, audit = inspect_gce(gce[k])
                base.update(audit)
                base["status"] = "strict_final" if result else ("invalid" if audit["problem"] else "incomplete")
                if result:
                    snapshots.append(result)
            else:
                base["status"] = "waiting_manifest"
            status_counts[str(base["status"])] += 1
            statuses.append(base)

    snapshot_keys = [(float(r["U"]), r["ensemble"], int(r["Ntot"]), float(r["beta"])) for r in snapshots]
    if len(snapshot_keys) != len(set(snapshot_keys)):
        dup = [k for k, n in Counter(snapshot_keys).items() if n > 1]
        raise SystemExit(f"duplicate snapshot rows: {dup[:5]}")
    snapshots.sort(key=lambda r: (float(r["U"]), int(r["Ntot"]), float(r["beta"]), str(r["ensemble"])))
    statuses.sort(key=lambda r: (float(r["U"]), int(r["Ntot"]), float(r["beta"]), str(r["ensemble"])))
    write_tsv(args.snapshot, snapshots, SNAPSHOT_FIELDS)
    write_tsv(args.status, statuses, statuses[0].keys())
    print(f"snapshot rows={len(snapshots)}/384; statuses={dict(status_counts)}")
    print(f"wrote {args.snapshot}")
    print(f"wrote {args.status}")

    invalid = [r for r in statuses if r["status"] == "invalid"]
    unfinished_required = [r for r in statuses if r["status"] not in ("strict_final", "exact_final", "sign_limited")]
    if invalid:
        raise SystemExit(f"invalid strict-final candidates: {len(invalid)}; first={invalid[0]}")
    if args.require_complete and unfinished_required:
        raise SystemExit(f"required ensemble rows unfinished: {len(unfinished_required)}")


if __name__ == "__main__":
    main()
