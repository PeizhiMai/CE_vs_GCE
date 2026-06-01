#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union


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

    value_error_cols = [
        ("lambda_longitudinal_qmin0", "lambda_longitudinal_stderr"),
        ("lambda_transverse_0qmin", "lambda_transverse_stderr"),
        ("Kx_per_site", "Kx_stderr"),
        ("diamagnetic_minus_Kx_per_site", "diamagnetic_minus_Kx_stderr"),
        ("rho_s_current", "rho_s_current_stderr"),
        ("rho_s_diamagnetic", "rho_s_diamagnetic_stderr"),
        ("bkt_residual_current", "bkt_residual_current_stderr"),
        ("bkt_residual_diamagnetic", "bkt_residual_diamagnetic_stderr"),
    ]
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

    value_error_cols = [
        ("kinetic_per_site", "kinetic_stderr"),
        ("interaction_per_site", "interaction_stderr"),
        ("total_per_site", "total_stderr"),
        ("double_occupancy_per_site", "double_occupancy_stderr"),
        ("Kx_per_site", "Kx_stderr"),
        ("diamagnetic_minus_Kx_per_site", "diamagnetic_minus_Kx_stderr"),
    ]
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
        "Kx_per_site",
        "Kx_stderr",
        "diamagnetic_minus_Kx_per_site",
        "diamagnetic_minus_Kx_stderr",
        "time_slices",
        "nsamples",
        "batches",
        "nranks",
    ]
    write_rows(outdir / filename, fieldnames, [out])


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
    combine_bkt_file(rank_dirs, outdir)


if __name__ == "__main__":
    main()
