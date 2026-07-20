#!/usr/bin/env python3
"""Audit OBC QMC coverage and compare dtau^2 extrapolations with ED."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tomllib
from collections import defaultdict
from pathlib import Path

import numpy as np


PRIMARY_FILES = {
    "kinetic_per_site": "equal_time_kinetic_per_site_qmc.tsv",
    "double_occupancy_per_site": "equal_time_double_occupancy_per_site_qmc.tsv",
    "nn_spin_s_s": "equal_time_nn_spin_qmc.tsv",
    "nn_connected_charge": "equal_time_nn_connected_charge_qmc.tsv",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ce-manifest", type=Path, required=True)
    parser.add_argument("--gce-manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--root-remap-from")
    parser.add_argument("--root-remap-to", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def remap(path: str, args: argparse.Namespace) -> Path:
    if args.root_remap_from and args.root_remap_to:
        prefix = args.root_remap_from.rstrip("/")
        if path == prefix or path.startswith(prefix + "/"):
            suffix = path[len(prefix) :].lstrip("/")
            return args.root_remap_to / suffix
    return Path(path)


def one_row(path: Path) -> dict[str, str]:
    rows = read_tsv(path)
    if len(rows) != 1:
        raise RuntimeError(f"expected one row in {path}, found {len(rows)}")
    return rows[0]


def rank_ratio_jackknife(
    data_dir: Path, name: str, *, scale: float = 1.0
) -> tuple[float, float]:
    """Pool signed rank accumulators and jackknife their ratio.

    The finite-dtau achieved density is a diagnostic, not a per-run rejection
    criterion: an ED-tuned chemical potential generally acquires an O(dtau^2)
    density shift.  This helper supplies the uncertainty needed to extrapolate
    that shift to zero time step.
    """
    rank_rows: list[tuple[complex, complex]] = []
    for path in sorted(data_dir.glob("obc_equal_time_rank_pID-*.tsv")):
        matches = [row for row in read_tsv(path) if row["name"] == name]
        if len(matches) != 1:
            raise RuntimeError(f"expected one {name!r} accumulator in {path}")
        row = matches[0]
        phase = complex(float(row["phase_sum_real"]), float(row["phase_sum_imag"]))
        signed = complex(float(row["signed_sum_real"]), float(row["signed_sum_imag"]))
        rank_rows.append((phase, signed))
    if not rank_rows:
        raise RuntimeError(f"no rank accumulators below {data_dir}")
    phase_total = sum((phase for phase, _ in rank_rows), 0j)
    signed_total = sum((signed for _, signed in rank_rows), 0j)
    if abs(phase_total) <= 100 * np.finfo(float).eps:
        raise RuntimeError("numerical-zero global phase in rank accumulator")
    ratio = signed_total / phase_total
    if abs(ratio.imag) > 1e-8 * max(1.0, abs(ratio.real)):
        raise RuntimeError(f"non-negligible imaginary pooled {name} ratio: {ratio}")
    value = float(ratio.real * scale)
    if len(rank_rows) == 1:
        return value, math.nan
    leave_one = []
    for phase, signed in rank_rows:
        denominator = phase_total - phase
        if abs(denominator) <= 100 * np.finfo(float).eps:
            raise RuntimeError("numerical-zero leave-one-rank phase denominator")
        leave_one.append(float(((signed_total - signed) / denominator).real * scale))
    samples = np.asarray(leave_one)
    error = math.sqrt(
        (len(samples) - 1) / len(samples) * float(np.sum((samples - np.mean(samples)) ** 2))
    )
    return value, error


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate_data_dir(row: dict[str, str], args: argparse.Namespace) -> Path | None:
    if row["ensemble"] == "CE":
        root = remap(row["outdir"], args)
        return root if root.is_dir() else None
    root = remap(row["out_parent"], args)
    complete = sorted(root.glob("complete_*")) if root.is_dir() else []
    if len(complete) > 1:
        raise RuntimeError(f"duplicate complete GCE roots below {root}")
    return complete[0] if complete else None


def audit_ce(
    row: dict[str, str], data_dir: Path, expected_ranks: int
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    rank_dirs = sorted(data_dir.glob("ranks/rank_*"))
    complete = [path for path in rank_dirs if (path / "checkpoint_complete.txt").is_file()]
    if len(rank_dirs) != expected_ranks or len(complete) != expected_ranks:
        issues.append(
            f"CE rank coverage dirs={len(rank_dirs)} complete={len(complete)} expected={expected_ranks}"
        )
    for rank_dir in rank_dirs:
        metadata_path = rank_dir / "metadata.toml"
        if not metadata_path.is_file():
            issues.append(f"missing {metadata_path}")
            continue
        metadata = tomllib.loads(metadata_path.read_text())
        dependencies = metadata.get("dependencies", {})
        lattice = metadata.get("lattice", {})
        equal_time = metadata.get("equal_time", {})
        if lattice.get("boundary") != "open":
            issues.append(f"non-OBC metadata in {rank_dir}")
        for field in ("site_count", "nn_bond_count", "nnn_bond_count"):
            if int(lattice.get(field, -1)) != int(row[field]):
                issues.append(f"CE geometry {field} mismatch in {rank_dir}")
        for field in (
            "kinetic_normalization",
            "double_occupancy_normalization",
            "nn_normalization",
            "nnn_normalization",
        ):
            if lattice.get(field) != row[field]:
                issues.append(f"CE estimator {field} mismatch in {rank_dir}")
        if dependencies.get("smoqydqmc_obc_fork_commit") != row["smoqydqmc_commit"]:
            issues.append(f"CE dependency provenance mismatch in {rank_dir}")
        if dependencies.get("smoqydqmc_reference_version") != row["smoqydqmc_version"]:
            issues.append(f"CE SmoQy reference version mismatch in {rank_dir}")
        for field in (
            "canensafqmc_base_commit",
            "canensafqmc_current_response_patch_sha256",
            "canensafqmc_obc_patch_sha256",
        ):
            if dependencies.get(field) != row[field]:
                issues.append(f"CE dependency {field} mismatch in {rank_dir}")
        expected_pair_estimator = row.get("ce_same_spin_estimator", "")
        if expected_pair_estimator and (
            equal_time.get("obc_same_spin_estimator") != expected_pair_estimator
        ):
            issues.append(f"CE canonical pair-estimator provenance mismatch in {rank_dir}")
    if any(data_dir.glob("equal_time_charge_spin_wedge_qmc.tsv")):
        issues.append("translational CE correlation output found under OBC")
    return not issues, issues


def audit_gce(
    row: dict[str, str], data_dir: Path, expected_ranks: int
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    rank_accum = list(data_dir.glob("obc_equal_time_rank_pID-*.tsv"))
    site_accum = list(data_dir.glob("obc_equal_time_site_rank_pID-*.tsv"))
    rank_info = list(data_dir.glob("simulation_info_sID-*_pID-*.toml"))
    if not (
        len(rank_accum) == len(site_accum) == len(rank_info) == expected_ranks
    ):
        issues.append(
            "GCE rank coverage "
            f"observable={len(rank_accum)} site={len(site_accum)} metadata={len(rank_info)} "
            f"expected={expected_ranks}"
        )
    for metadata_path in rank_info:
        document = tomllib.loads(metadata_path.read_text())
        metadata = document.get("metadata", document)
        if metadata.get("boundary") != "open":
            issues.append(f"non-OBC metadata in {metadata_path}")
        if metadata.get("smoqydqmc_version") != row["smoqydqmc_version"]:
            issues.append(f"SmoQy version mismatch in {metadata_path}")
        if metadata.get("smoqydqmc_commit") != row["smoqydqmc_commit"]:
            issues.append(f"SmoQy commit mismatch in {metadata_path}")
        for field in ("site_count", "nn_bond_count", "nnn_bond_count"):
            if int(metadata.get(f"geometry_{field}", -1)) != int(row[field]):
                issues.append(f"GCE geometry {field} mismatch in {metadata_path}")
        for field in (
            "kinetic_normalization",
            "double_occupancy_normalization",
            "nn_normalization",
            "nnn_normalization",
        ):
            if metadata.get(f"geometry_{field}") != row[field]:
                issues.append(f"GCE estimator {field} mismatch in {metadata_path}")
    forbidden = [
        path
        for path in data_dir.rglob("*")
        if path.is_file()
        and any(
            token in str(path.relative_to(data_dir)).lower()
            for token in ("time-displaced", "integrated", "bkt", "greens_")
        )
    ]
    if forbidden:
        issues.append(f"forbidden OBC translational/unequal-time outputs: {len(forbidden)}")
    return not issues, issues


def collect_rows(
    manifest_path: Path, args: argparse.Namespace
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    collected: list[dict[str, object]] = []
    density_rows: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    for row in read_tsv(manifest_path):
        expected_ranks = int(row["expected_ranks"])
        reference_path = Path(row["ed_reference_file"])
        if not reference_path.is_absolute():
            reference_path = Path.cwd() / reference_path
        if (
            not reference_path.is_file()
            or sha256(reference_path) != row["ed_reference_sha256"]
        ):
            missing.append(
                {
                    "ensemble": row["ensemble"],
                    "idx": row["idx"],
                    "run_id": row["run_id"],
                    "reason": "ED reference file is missing or has a SHA-256 mismatch",
                }
            )
            continue
        if float(row["ed_tail_weight_bound"]) > 1e-8:
            missing.append(
                {
                    "ensemble": row["ensemble"],
                    "idx": row["idx"],
                    "run_id": row["run_id"],
                    "reason": "ED omitted-Boltzmann-weight bound exceeds 1e-8",
                }
            )
            continue
        if float(row.get("ed_max_eigen_residual", "inf")) > 1e-8:
            missing.append(
                {
                    "ensemble": row["ensemble"],
                    "idx": row["idx"],
                    "run_id": row["run_id"],
                    "reason": "ED eigenpair residual exceeds 1e-8",
                }
            )
            continue
        data_dir = locate_data_dir(row, args)
        if data_dir is None:
            missing.append(
                {
                    "ensemble": row["ensemble"],
                    "idx": row["idx"],
                    "run_id": row["run_id"],
                    "reason": "run root is not complete",
                }
            )
            continue
        audit_ok, issues = (
            audit_ce(row, data_dir, expected_ranks)
            if row["ensemble"] == "CE"
            else audit_gce(row, data_dir, expected_ranks)
        )
        values: dict[str, tuple[float, float]] = {}
        table_problem = False
        for observable, filename in PRIMARY_FILES.items():
            path = data_dir / filename
            if not path.is_file():
                issues.append(f"missing primary table {filename}")
                table_problem = True
                continue
            primary = one_row(path)
            if primary.get("boundary") != "open":
                issues.append(f"{filename} does not identify open boundary")
            if int(primary.get("nranks", expected_ranks)) != expected_ranks:
                issues.append(f"{filename} has wrong nranks")
            phase = float(
                primary.get("average_phase", primary.get("average_phase_abs", "1"))
            )
            if not math.isfinite(phase) or abs(phase) <= 100 * np.finfo(float).eps:
                issues.append(f"{filename} has numerical-zero phase")
            values[observable] = (float(primary["value"]), float(primary["stderr"]))
        achieved_n = float(row["target_N"])
        achieved_n_stderr = 0.0
        if row["ensemble"] == "GCE":
            density_tolerance = float(row.get("density_tolerance", "0.01"))
            ed_density_error = abs(
                float(row.get("ed_achieved_N", "nan")) - float(row["target_N"])
            )
            if not math.isfinite(ed_density_error) or ed_density_error > density_tolerance:
                issues.append(
                    "ED-tuned GCE chemical potential misses target density "
                    f"by {ed_density_error} > {density_tolerance}"
                )
            density_path = data_dir / "equal_time_observables_obc_qmc.tsv"
            if density_path.is_file():
                density = one_row(density_path)
                achieved_n = float(density["achieved_N"])
                try:
                    pooled_n, achieved_n_stderr = rank_ratio_jackknife(
                        data_dir, "density", scale=float(row["site_count"])
                    )
                except (KeyError, RuntimeError, ValueError) as error:
                    issues.append(f"cannot pool GCE achieved N: {error}")
                else:
                    if not math.isclose(
                        pooled_n, achieved_n, rel_tol=1e-10, abs_tol=1e-10
                    ):
                        issues.append(
                            f"GCE achieved-N pooled/table mismatch {pooled_n} vs {achieved_n}"
                        )
                if not math.isfinite(achieved_n):
                    issues.append(
                        f"non-finite GCE achieved N in {density_path}"
                    )
            else:
                issues.append("missing GCE achieved-N table")
        if table_problem or len(values) != len(PRIMARY_FILES):
            audit_ok = False
        if issues:
            audit_ok = False
        if not audit_ok:
            missing.append(
                {
                    "ensemble": row["ensemble"],
                    "idx": row["idx"],
                    "run_id": row["run_id"],
                    "reason": "; ".join(issues),
                }
            )
            continue
        for observable, (value, stderr) in values.items():
            collected.append(
                {
                    "ensemble": row["ensemble"],
                    "L": int(row["L"]),
                    "U": float(row["U"]),
                    "beta": float(row["beta"]),
                    "nup": int(row["nup"]),
                    "ndn": int(row["ndn"]),
                    "target_N": int(row["target_N"]),
                    "dtau": float(row["dtau"]),
                    "seed_index": int(row["seed_index"]),
                    "observable": observable,
                    "value": value,
                    "stderr": stderr,
                    "ed_value": float(row[f"ed_{observable}"]),
                    "achieved_N": achieved_n,
                    "delta_N": achieved_n - float(row["target_N"]),
                    "run_id": row["run_id"],
                }
            )
        if row["ensemble"] == "GCE":
            density_rows.append(
                {
                    "ensemble": row["ensemble"],
                    "L": int(row["L"]),
                    "U": float(row["U"]),
                    "beta": float(row["beta"]),
                    "nup": int(row["nup"]),
                    "ndn": int(row["ndn"]),
                    "target_N": int(row["target_N"]),
                    "dtau": float(row["dtau"]),
                    "seed_index": int(row["seed_index"]),
                    "observable": "achieved_N",
                    "value": achieved_n,
                    "stderr": achieved_n_stderr,
                    "ed_value": float(row["ed_achieved_N"]),
                    "density_tolerance": float(row.get("density_tolerance", "0.01")),
                    "delta_N": achieved_n - float(row["target_N"]),
                    "run_id": row["run_id"],
                }
            )
    return collected, density_rows, missing


def combine_seed_rows(rows: list[dict[str, object]]) -> tuple[float, float]:
    values = np.asarray([float(row["value"]) for row in rows])
    errors = np.asarray([float(row["stderr"]) for row in rows])
    finite = np.isfinite(errors) & (errors > 0)
    if np.any(finite):
        weights = np.where(finite, 1.0 / np.maximum(errors, 1e-15) ** 2, 0.0)
        mean = float(np.sum(weights * values) / np.sum(weights))
        internal_variance = float(1.0 / np.sum(weights))
    else:
        mean = float(np.mean(values))
        internal_variance = 0.0
    between_sem = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, math.sqrt(internal_variance + between_sem**2)


def fit_dtau_squared(rows: list[dict[str, object]]) -> tuple[float, float, float, float]:
    x = np.asarray([float(row["dtau"]) ** 2 for row in rows])
    y = np.asarray([float(row["value"]) for row in rows])
    sigma = np.asarray([max(float(row["stderr"]), 1e-12) for row in rows])
    design = np.column_stack((np.ones_like(x), x))
    weights = 1.0 / sigma**2
    normal = design.T @ (weights[:, None] * design)
    covariance = np.linalg.inv(normal)
    coefficients = covariance @ (design.T @ (weights * y))
    residual = y - design @ coefficients
    chi2 = float(np.sum((residual / sigma) ** 2))
    reduced = chi2 / max(len(y) - 2, 1)
    covariance *= max(1.0, reduced)
    return (
        float(coefficients[0]),
        float(math.sqrt(max(covariance[0, 0], 0.0))),
        float(coefficients[1]),
        reduced,
    )


def main() -> None:
    args = parse_args()
    ce_manifest_rows = read_tsv(args.ce_manifest)
    gce_manifest_rows = read_tsv(args.gce_manifest)
    manifest_issues: list[str] = []
    for ensemble, rows, root_field in (
        ("CE", ce_manifest_rows, "outdir"),
        ("GCE", gce_manifest_rows, "out_parent"),
    ):
        if len(rows) != 72:
            manifest_issues.append(f"{ensemble} manifest has {len(rows)} rows, expected 72")
        if len({row["run_id"] for row in rows}) != len(rows):
            manifest_issues.append(f"{ensemble} manifest has duplicate run_id values")
        if len({row[root_field] for row in rows}) != len(rows):
            manifest_issues.append(f"{ensemble} manifest has duplicate run roots")
        rank_streams: set[int] = set()
        for row in rows:
            streams = {
                int(row["seed"]) + p_id
                for p_id in range(int(row["expected_ranks"]))
            }
            if rank_streams & streams:
                manifest_issues.append(
                    f"{ensemble} manifest has overlapping seed+pID MPI rank streams"
                )
                break
            rank_streams.update(streams)
    all_rows: list[dict[str, object]] = []
    density_rows: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    for manifest in (args.ce_manifest, args.gce_manifest):
        collected, densities, absent = collect_rows(manifest, args)
        all_rows.extend(collected)
        density_rows.extend(densities)
        missing.extend(absent)

    seed_groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in all_rows:
        key = tuple(
            row[column]
            for column in (
                "ensemble",
                "L",
                "U",
                "beta",
                "nup",
                "ndn",
                "target_N",
                "observable",
                "dtau",
            )
        )
        seed_groups[key].append(row)

    dtau_rows: list[dict[str, object]] = []
    for key, rows in sorted(seed_groups.items(), key=lambda item: tuple(map(str, item[0]))):
        if len(rows) != 2 or {int(row["seed_index"]) for row in rows} != {0, 1}:
            continue
        mean, stderr = combine_seed_rows(rows)
        dtau_rows.append(
            {
                "ensemble": key[0],
                "L": key[1],
                "U": key[2],
                "beta": key[3],
                "nup": key[4],
                "ndn": key[5],
                "target_N": key[6],
                "observable": key[7],
                "dtau": key[8],
                "value": mean,
                "stderr": stderr,
                "ed_value": rows[0]["ed_value"],
                "seed_replicates": 2,
            }
        )

    fit_groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in dtau_rows:
        key = tuple(
            row[column]
            for column in (
                "ensemble",
                "L",
                "U",
                "beta",
                "nup",
                "ndn",
                "target_N",
                "observable",
            )
        )
        fit_groups[key].append(row)

    extrapolations: list[dict[str, object]] = []
    for key, rows in sorted(fit_groups.items(), key=lambda item: tuple(map(str, item[0]))):
        if len(rows) != 3 or {float(row["dtau"]) for row in rows} != {0.2, 0.1, 0.05}:
            continue
        intercept, intercept_error, slope, reduced_chi2 = fit_dtau_squared(rows)
        ed_value = float(rows[0]["ed_value"])
        delta = intercept - ed_value
        combined_error = max(intercept_error, 1e-12)
        passed = abs(delta) <= 3.0 * combined_error
        extrapolations.append(
            {
                "ensemble": key[0],
                "L": key[1],
                "U": key[2],
                "beta": key[3],
                "nup": key[4],
                "ndn": key[5],
                "target_N": key[6],
                "observable": key[7],
                "qmc_dtau2_to_zero": intercept,
                "qmc_intercept_stderr": intercept_error,
                "ed_value": ed_value,
                "qmc_minus_ed": delta,
                "combined_stderr": combined_error,
                "z_score": delta / combined_error,
                "slope_dtau2": slope,
                "reduced_chi2": reduced_chi2,
                "passed_3sigma": passed,
            }
        )

    density_seed_groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in density_rows:
        key = tuple(
            row[column]
            for column in ("L", "U", "beta", "nup", "ndn", "target_N", "dtau")
        )
        density_seed_groups[key].append(row)

    density_dtau_rows: list[dict[str, object]] = []
    for key, rows in sorted(
        density_seed_groups.items(), key=lambda item: tuple(map(str, item[0]))
    ):
        if len(rows) != 2 or {int(row["seed_index"]) for row in rows} != {0, 1}:
            continue
        mean, stderr = combine_seed_rows(rows)
        density_dtau_rows.append(
            {
                "ensemble": "GCE",
                "L": key[0],
                "U": key[1],
                "beta": key[2],
                "nup": key[3],
                "ndn": key[4],
                "target_N": key[5],
                "dtau": key[6],
                "achieved_N": mean,
                "achieved_N_stderr": stderr,
                "delta_N": mean - float(key[5]),
                "density_tolerance": rows[0]["density_tolerance"],
                "seed_replicates": 2,
            }
        )

    density_fit_groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in density_dtau_rows:
        key = tuple(
            row[column]
            for column in ("L", "U", "beta", "nup", "ndn", "target_N")
        )
        density_fit_groups[key].append(row)

    density_extrapolations: list[dict[str, object]] = []
    for key, rows in sorted(
        density_fit_groups.items(), key=lambda item: tuple(map(str, item[0]))
    ):
        if len(rows) != 3 or {float(row["dtau"]) for row in rows} != {0.2, 0.1, 0.05}:
            continue
        fit_rows = [
            {
                "dtau": row["dtau"],
                "value": row["achieved_N"],
                "stderr": row["achieved_N_stderr"],
            }
            for row in rows
        ]
        intercept, intercept_error, slope, reduced_chi2 = fit_dtau_squared(fit_rows)
        target = float(key[5])
        delta = intercept - target
        tolerance = float(rows[0]["density_tolerance"])
        threshold = max(tolerance, 3.0 * max(intercept_error, 1e-12))
        density_extrapolations.append(
            {
                "ensemble": "GCE",
                "L": key[0],
                "U": key[1],
                "beta": key[2],
                "nup": key[3],
                "ndn": key[4],
                "target_N": key[5],
                "qmc_achieved_N_dtau2_to_zero": intercept,
                "qmc_intercept_stderr": intercept_error,
                "qmc_minus_target_N": delta,
                "density_tolerance": tolerance,
                "three_sigma": 3.0 * max(intercept_error, 1e-12),
                "acceptance_threshold": threshold,
                "slope_dtau2": slope,
                "reduced_chi2": reduced_chi2,
                "passed_density_gate": abs(delta) <= threshold,
            }
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.outdir / "run_observables.tsv", all_rows)
    write_tsv(args.outdir / "gce_density_run_diagnostics.tsv", density_rows)
    write_tsv(args.outdir / "missing_or_invalid_runs.tsv", missing)
    write_tsv(args.outdir / "seed_combined_dtau.tsv", dtau_rows)
    write_tsv(args.outdir / "dtau2_extrapolation_vs_ed.tsv", extrapolations)
    write_tsv(args.outdir / "gce_density_seed_combined_dtau.tsv", density_dtau_rows)
    write_tsv(args.outdir / "gce_density_dtau2_extrapolation.tsv", density_extrapolations)
    expected_runs = 144
    expected_fits = 2 * 12 * len(PRIMARY_FILES)
    summary = {
        "schema_version": 1,
        "expected_runs": expected_runs,
        "complete_valid_runs": len(all_rows) // len(PRIMARY_FILES),
        "missing_or_invalid_runs": len(missing),
        "expected_extrapolations": expected_fits,
        "completed_extrapolations": len(extrapolations),
        "failed_3sigma": sum(not bool(row["passed_3sigma"]) for row in extrapolations),
        "expected_gce_density_extrapolations": 12,
        "completed_gce_density_extrapolations": len(density_extrapolations),
        "failed_gce_density_gates": sum(
            not bool(row["passed_density_gate"]) for row in density_extrapolations
        ),
        "finite_dtau_gce_rows_outside_density_tolerance": sum(
            abs(float(row["delta_N"])) > float(row["density_tolerance"])
            for row in density_rows
        ),
        "all_four_primary_observables": list(PRIMARY_FILES),
        "manifest_issues": manifest_issues,
    }
    summary["accepted"] = (
        summary["complete_valid_runs"] == expected_runs
        and not manifest_issues
        and not missing
        and len(extrapolations) == expected_fits
        and summary["failed_3sigma"] == 0
        and len(density_extrapolations) == summary["expected_gce_density_extrapolations"]
        and summary["failed_gce_density_gates"] == 0
    )
    (args.outdir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_complete and not summary["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
