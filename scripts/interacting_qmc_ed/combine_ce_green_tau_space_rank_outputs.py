#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union


FLOAT_EPSILON = 2.220446049250313e-16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Combine independent MPI-rank CE-QMC G(r,tau)/G(k,tau) summary TSV files."
    )
    p.add_argument("--root-dir", type=Path, required=True, help="MPI run root containing rank output dirs.")
    p.add_argument("--rank-glob", default="ranks/rank_*", help="Glob below root-dir for rank output dirs.")
    p.add_argument("--outdir", type=Path, default=None, help="Where to write combined TSVs; default is root-dir.")
    return p.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_rows(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def finite_float(x: Union[str, float]) -> float:
    v = float(x)
    return v if math.isfinite(v) else float("nan")


def combine_stats(entries: List[Tuple[int, float, float]]) -> Tuple[float, float, int]:
    """Combine (n, mean, stderr) groups into (mean, stderr, n_total)."""
    entries = [(n, m, se) for n, m, se in entries if n > 0 and math.isfinite(m)]
    n_total = sum(n for n, _, _ in entries)
    if n_total == 0:
        return float("nan"), float("nan"), 0
    mean = sum(n * m for n, m, _ in entries) / n_total
    if n_total <= 1:
        return mean, float("nan"), n_total
    ss = 0.0
    for n, m, se in entries:
        if n > 1 and math.isfinite(se):
            s2 = (se * math.sqrt(n)) ** 2
            ss += (n - 1) * s2
        ss += n * (m - mean) ** 2
    var = max(ss / (n_total - 1), 0.0)
    return mean, math.sqrt(var / n_total), n_total


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def rows_are_phase_reweighted(rows: List[Dict[str, str]]) -> bool:
    return bool(rows) and all(
        parse_bool(row.get("phase_reweighted", "false")) and "phase_sum" in row
        for row in rows
    )


def combine_phase_reweighted_rows(
    rows: List[Dict[str, str]], value_col: str, err_col: str
) -> Tuple[float, float, int]:
    """Combine rank ratios exactly and estimate uncertainty by rank jackknife."""
    valid = []
    for row in rows:
        n = int(row["nsamples"])
        denominator = finite_float(row["phase_sum"])
        value = finite_float(row[value_col])
        signed_sum_col = f"{value_col}_signed_sum"
        signed_numerator = finite_float(row.get(signed_sum_col, float("nan")))
        if not math.isfinite(signed_numerator) and math.isfinite(value):
            # Backward compatibility for old rank files whose denominator was
            # nonzero and therefore had a finite rank-local ratio.
            signed_numerator = denominator * value
        if n > 0 and math.isfinite(denominator) and math.isfinite(signed_numerator):
            valid.append((n, denominator, signed_numerator, finite_float(row[err_col])))
    n_total = sum(n for n, _, _, _ in valid)
    denominator_total = sum(d for _, d, _, _ in valid)
    numerator_total = sum(num for _, _, num, _ in valid)
    if n_total == 0 or abs(denominator_total) <= 10 * FLOAT_EPSILON:
        return float("nan"), float("nan"), n_total
    mean = numerator_total / denominator_total
    leave_one_out = []
    if len(valid) >= 2:
        for _, denominator, numerator, _ in valid:
            d_loo = denominator_total - denominator
            if abs(d_loo) > 10 * FLOAT_EPSILON:
                leave_one_out.append((numerator_total - numerator) / d_loo)
    if len(leave_one_out) == len(valid) and len(leave_one_out) >= 2:
        center = sum(leave_one_out) / len(leave_one_out)
        err = math.sqrt(
            (len(leave_one_out) - 1)
            / len(leave_one_out)
            * sum((x - center) ** 2 for x in leave_one_out)
        )
    elif len(valid) == 1:
        err = valid[0][3]
    else:
        err = float("nan")
    return mean, err, n_total


def collect_by_key(rank_dirs: List[Path], filename: str, key_cols: List[str]) -> Dict[Tuple[str, ...], List[Dict[str, str]]]:
    grouped = {}  # type: Dict[Tuple[str, ...], List[Dict[str, str]]]
    for rank_dir in rank_dirs:
        path = rank_dir / filename
        if not path.exists():
            continue
        for row in read_rows(path):
            key = tuple(row[c] for c in key_cols)
            grouped.setdefault(key, []).append(row)
    return grouped


def combine_tau_file(rank_dirs: List[Path], outdir: Path, filename: str, value_col: str) -> None:
    key_cols = ["slice", "tau"]
    grouped = collect_by_key(rank_dirs, filename, key_cols)
    if not grouped:
        return
    rows = []  # type: List[Dict[str, object]]
    for key in sorted(grouped, key=lambda k: int(k[0])):
        entries = []
        batches = []
        for row in grouped[key]:
            n = int(row["nsamples"])
            entries.append((n, finite_float(row[value_col]), finite_float(row[value_col.replace("_mean", "_stderr")])))
            batches.append(int(row["batches"]))
        mean, err, n_total = combine_stats(entries)
        rows.append(
            {
                "slice": key[0],
                "tau": key[1],
                value_col: mean,
                value_col.replace("_mean", "_stderr"): err,
                "nsamples": n_total,
                "batches": max(batches) if batches else 0,
                "nranks": len(entries),
            }
        )
    write_rows(outdir / filename, key_cols + [value_col, value_col.replace("_mean", "_stderr"), "nsamples", "batches", "nranks"], rows)


def combine_complex_file(rank_dirs: List[Path], outdir: Path, filename: str, key_cols: List[str], passthrough_cols: List[str]) -> None:
    grouped = collect_by_key(rank_dirs, filename, key_cols)
    if not grouped:
        return
    rows = []  # type: List[Dict[str, object]]
    def sort_key(k):
        return tuple(int(float(x)) if i == 0 or x.replace(".", "", 1).replace("-", "", 1).isdigit() else x for i, x in enumerate(k))

    for key in sorted(grouped, key=sort_key):
        re_entries = []
        im_entries = []
        batches = []
        for row in grouped[key]:
            n = int(row["nsamples"])
            re_entries.append((n, finite_float(row["Greal_mean"]), finite_float(row["Greal_stderr"])))
            im_entries.append((n, finite_float(row["Gimag_mean"]), finite_float(row["Gimag_stderr"])))
            batches.append(int(row["batches"]))
        re_mean, re_err, n_total = combine_stats(re_entries)
        im_mean, im_err, _ = combine_stats(im_entries)
        first = grouped[key][0]
        out = {c: first[c] for c in passthrough_cols}
        out.update(
            {
                "Greal_mean": re_mean,
                "Greal_stderr": re_err,
                "Gimag_mean": im_mean,
                "Gimag_stderr": im_err,
                "nsamples": n_total,
                "batches": max(batches) if batches else 0,
                "nranks": len(re_entries),
            }
        )
        rows.append(out)
    write_rows(
        outdir / filename,
        passthrough_cols
        + ["Greal_mean", "Greal_stderr", "Gimag_mean", "Gimag_stderr", "nsamples", "batches", "nranks"],
        rows,
    )


def maybe_add_value_error_cols(rows: List[Dict[str, str]], pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    if not rows:
        return []
    first = rows[0]
    return [(v, e) for v, e in pairs if v in first and e in first]


def combine_bkt_file(rank_dirs: List[Path], outdir: Path) -> None:
    filename = "bkt_observables_qmc.tsv"
    rows_by_rank = []
    for rank_dir in rank_dirs:
        path = rank_dir / filename
        if not path.exists():
            continue
        rows = read_rows(path)
        if rows:
            rows_by_rank.append(rows[0])
    if not rows_by_rank:
        return

    value_error_cols = maybe_add_value_error_cols(rows_by_rank, [
        ("lambda_longitudinal_qmin0", "lambda_longitudinal_stderr"),
        ("lambda_transverse_0qmin", "lambda_transverse_stderr"),
        ("Kx_per_site", "Kx_stderr"),
        ("diamagnetic_minus_Kx_per_site", "diamagnetic_minus_Kx_stderr"),
        ("rho_s_current", "rho_s_current_stderr"),
        ("rho_s_diamagnetic", "rho_s_diamagnetic_stderr"),
        ("bkt_residual_current", "bkt_residual_current_stderr"),
        ("bkt_residual_diamagnetic", "bkt_residual_diamagnetic_stderr"),
    ])
    first = rows_by_rank[0]
    out = {
        "beta": first["beta"],
        "temperature": first["temperature"],
        "bkt_universal_jump_2T_over_pi": first["bkt_universal_jump_2T_over_pi"],
        "time_slices": first["time_slices"],
    }  # type: Dict[str, object]
    n_total_ref = None
    for value_col, err_col in value_error_cols:
        entries = []
        for row in rows_by_rank:
            n = int(row["nsamples"])
            entries.append((n, finite_float(row[value_col]), finite_float(row[err_col])))
        mean, err, n_total = combine_stats(entries)
        out[value_col] = mean
        out[err_col] = err
        if n_total_ref is None:
            n_total_ref = n_total
    out["nsamples"] = n_total_ref if n_total_ref is not None else 0
    out["batches"] = max(int(row["batches"]) for row in rows_by_rank)
    out["nranks"] = len(rows_by_rank)

    fieldnames = [
        "beta",
        "temperature",
        "bkt_universal_jump_2T_over_pi",
        "lambda_longitudinal_qmin0",
        "lambda_longitudinal_stderr",
        "lambda_transverse_0qmin",
        "lambda_transverse_stderr",
        "Kx_per_site",
        "Kx_stderr",
        "diamagnetic_minus_Kx_per_site",
        "diamagnetic_minus_Kx_stderr",
        "rho_s_current",
        "rho_s_current_stderr",
        "rho_s_diamagnetic",
        "rho_s_diamagnetic_stderr",
        "bkt_residual_current",
        "bkt_residual_current_stderr",
        "bkt_residual_diamagnetic",
        "bkt_residual_diamagnetic_stderr",
        "time_slices",
        "nsamples",
        "batches",
        "nranks",
    ]
    write_rows(outdir / filename, fieldnames, [out])


def combine_equal_time_file(rank_dirs: List[Path], outdir: Path) -> None:
    filename = "equal_time_observables_qmc.tsv"
    rows_by_rank = []
    for rank_dir in rank_dirs:
        path = rank_dir / filename
        if not path.exists():
            continue
        rows = read_rows(path)
        if rows:
            rows_by_rank.append(rows[0])
    if not rows_by_rank:
        return

    value_error_cols = maybe_add_value_error_cols(rows_by_rank, [
        ("kinetic_per_site", "kinetic_stderr"),
        ("interaction_per_site", "interaction_stderr"),
        ("total_per_site", "total_stderr"),
        ("double_occupancy_per_site", "double_occupancy_stderr"),
        ("local_moment_z", "local_moment_z_stderr"),
        ("Kx_per_site", "Kx_stderr"),
        ("diamagnetic_minus_Kx_per_site", "diamagnetic_minus_Kx_stderr"),
    ])
    first = rows_by_rank[0]
    out = {
        "beta": first["beta"],
        "temperature": first["temperature"],
        "nup": first["nup"],
        "ndn": first["ndn"],
        "ntotal": first["ntotal"],
        "density": first["density"],
        "time_slices": first["time_slices"],
    }  # type: Dict[str, object]
    n_total_ref = None
    phase_reweighted = rows_are_phase_reweighted(rows_by_rank)
    for value_col, err_col in value_error_cols:
        if phase_reweighted:
            mean, err, n_total = combine_phase_reweighted_rows(rows_by_rank, value_col, err_col)
        else:
            entries = []
            for row in rows_by_rank:
                n = int(row["nsamples"])
                entries.append((n, finite_float(row[value_col]), finite_float(row[err_col])))
            mean, err, n_total = combine_stats(entries)
        out[value_col] = mean
        out[err_col] = err
        if n_total_ref is None:
            n_total_ref = n_total
    out["nsamples"] = n_total_ref if n_total_ref is not None else 0
    out["batches"] = max(int(row["batches"]) for row in rows_by_rank)
    out["nranks"] = len(rows_by_rank)
    out["phase_reweighted"] = phase_reweighted
    out["phase_sum"] = sum(finite_float(row.get("phase_sum", row["nsamples"])) for row in rows_by_rank)
    out["average_phase"] = out["phase_sum"] / out["nsamples"] if out["nsamples"] else float("nan")

    fieldnames = [
        "beta",
        "temperature",
        "nup",
        "ndn",
        "ntotal",
        "density",
        "kinetic_per_site",
        "kinetic_stderr",
        "interaction_per_site",
        "interaction_stderr",
        "total_per_site",
        "total_stderr",
        "double_occupancy_per_site",
        "double_occupancy_stderr",
        "local_moment_z",
        "local_moment_z_stderr",
        "Kx_per_site",
        "Kx_stderr",
        "diamagnetic_minus_Kx_per_site",
        "diamagnetic_minus_Kx_stderr",
        "time_slices",
        "nsamples",
        "batches",
        "nranks",
        "phase_reweighted",
        "phase_sum",
        "average_phase",
    ]
    write_rows(outdir / filename, fieldnames, [out])


def sort_key_numeric_tuple(k: Tuple[str, ...]) -> Tuple[float, ...]:
    vals = []
    for item in k:
        try:
            vals.append(float(item))
        except ValueError:
            vals.append(float("inf"))
    return tuple(vals)


def combine_keyed_scalar_file(
    rank_dirs: List[Path],
    outdir: Path,
    filename: str,
    key_cols: List[str],
    passthrough_cols: List[str],
    value_error_cols: List[Tuple[str, str]],
) -> None:
    grouped = collect_by_key(rank_dirs, filename, key_cols)
    if not grouped:
        return
    rows = []  # type: List[Dict[str, object]]
    for key in sorted(grouped, key=sort_key_numeric_tuple):
        key_rows = grouped[key]
        phase_reweighted = rows_are_phase_reweighted(key_rows)
        entries_by_value = {value_col: [] for value_col, _ in value_error_cols}
        batches = []
        for row in key_rows:
            n = int(row["nsamples"])
            for value_col, err_col in value_error_cols:
                entries_by_value[value_col].append((n, finite_float(row[value_col]), finite_float(row[err_col])))
            batches.append(int(row["batches"]))
        first = key_rows[0]
        out = {c: first[c] for c in passthrough_cols}
        n_total_ref = None
        for value_col, err_col in value_error_cols:
            if phase_reweighted:
                mean, err, n_total = combine_phase_reweighted_rows(key_rows, value_col, err_col)
            else:
                mean, err, n_total = combine_stats(entries_by_value[value_col])
            out[value_col] = mean
            out[err_col] = err
            if n_total_ref is None:
                n_total_ref = n_total
        out["nsamples"] = n_total_ref if n_total_ref is not None else 0
        out["batches"] = max(batches) if batches else 0
        out["nranks"] = len(key_rows)
        out["phase_reweighted"] = phase_reweighted
        out["phase_sum"] = sum(finite_float(row.get("phase_sum", row["nsamples"])) for row in key_rows)
        out["average_phase"] = out["phase_sum"] / out["nsamples"] if out["nsamples"] else float("nan")
        rows.append(out)
    write_rows(
        outdir / filename,
        passthrough_cols + [c for pair in value_error_cols for c in pair] + [
            "nsamples", "batches", "nranks", "phase_reweighted", "phase_sum", "average_phase"
        ],
        rows,
    )


def combine_equal_time_correlation_files(rank_dirs: List[Path], outdir: Path) -> None:
    combine_keyed_scalar_file(
        rank_dirs,
        outdir,
        "equal_time_charge_spin_wedge_qmc.tsv",
        ["dx", "dy"],
        ["dx", "dy"],
        [
            ("charge_corr_raw", "charge_corr_raw_stderr"),
            ("charge_corr_connected", "charge_corr_connected_stderr"),
            ("spin_corr_s_s", "spin_corr_s_s_stderr"),
            ("spin_corr_SzSz", "spin_corr_SzSz_stderr"),
        ],
    )
    combine_keyed_scalar_file(
        rank_dirs,
        outdir,
        "equal_time_structure_factors_qmc.tsv",
        ["mx", "my"],
        ["mx", "my", "qx", "qy"],
        [
            ("charge_structure_raw", "charge_structure_raw_stderr"),
            ("charge_structure_connected", "charge_structure_connected_stderr"),
            ("spin_structure_s_s", "spin_structure_s_s_stderr"),
            ("spin_structure_SzSz", "spin_structure_SzSz_stderr"),
        ],
    )
    combine_keyed_scalar_file(
        rank_dirs,
        outdir,
        "equal_time_neighbor_shells_qmc.tsv",
        ["shell"],
        ["shell", "r2", "vectors"],
        [
            ("charge_corr_raw", "charge_corr_raw_stderr"),
            ("charge_corr_connected", "charge_corr_connected_stderr"),
            ("spin_corr_s_s", "spin_corr_s_s_stderr"),
            ("spin_corr_SzSz", "spin_corr_SzSz_stderr"),
        ],
    )


def main() -> None:
    args = parse_args()
    outdir = args.outdir or args.root_dir
    rank_dirs = sorted([d for d in args.root_dir.glob(args.rank_glob) if d.is_dir()])
    if not rank_dirs:
        raise SystemExit(f"no rank dirs found below {args.root_dir} with glob {args.rank_glob!r}")

    combine_tau_file(rank_dirs, outdir, "greens_tau0_qmc.tsv", "Ctau_mean")
    combine_tau_file(rank_dirs, outdir, "greens_tau0_remove_qmc.tsv", "Rtau_mean")
    combine_complex_file(
        rank_dirs,
        outdir,
        "greens_r_tau_add_qmc.tsv",
        ["slice", "tau", "dx", "dy"],
        ["slice", "tau", "dx", "dy"],
    )
    combine_complex_file(
        rank_dirs,
        outdir,
        "greens_r_tau_remove_qmc.tsv",
        ["slice", "tau", "dx", "dy"],
        ["slice", "tau", "dx", "dy"],
    )
    combine_complex_file(
        rank_dirs,
        outdir,
        "greens_k_tau_add_qmc.tsv",
        ["slice", "tau", "nx", "ny"],
        ["slice", "tau", "nx", "ny", "kx", "ky"],
    )
    combine_complex_file(
        rank_dirs,
        outdir,
        "greens_k_tau_remove_qmc.tsv",
        ["slice", "tau", "nx", "ny"],
        ["slice", "tau", "nx", "ny", "kx", "ky"],
    )
    combine_equal_time_file(rank_dirs, outdir)
    combine_equal_time_correlation_files(rank_dirs, outdir)
    combine_bkt_file(rank_dirs, outdir)


if __name__ == "__main__":
    main()
