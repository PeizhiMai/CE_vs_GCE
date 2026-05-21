#!/usr/bin/env python3
"""Summarize SmoQyDQMC density-tuning runs and interpolate mu for a target density.

Example on CADES:

    python3.11 scripts/interacting_qmc_ed/summarize_density_tuning.py \
      /home/9pm/nUHubbard/runs/density_mu_search_L12_n05_stage1_T0142 \
      --target-density 0.5

The script reads each run's aggregated global_stats.csv (or, for legacy output,
the first global_stats_pID-*.csv), extracts the density row, writes all scan
points to density_tuning_summary.tsv, and writes one interpolated
chemical-potential estimate per temperature to mu_estimates_n*.tsv.
"""

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


RUN_RE = re.compile(
    r"(?:complete_)?attractive_hubbard_rect_"
    r"U(?P<U>[-+0-9.]+)_tp(?P<tprime>[-+0-9.]+)_mu(?P<mu>[-+0-9.]+)"
    r"_Lx(?P<Lx>\d+)_Ly(?P<Ly>\d+)_b(?P<beta>[-+0-9.]+)-(?P<sid>\d+)$"
)


def read_global_row(path, measurement):
    with path.open() as f:
        reader = csv.DictReader(f, delimiter=" ", skipinitialspace=True)
        for row in reader:
            if row.get("MEASUREMENT") == measurement:
                return float(row["MEAN_REAL"]), float(row["STD"])
    raise KeyError("No measurement {!r} in {}".format(measurement, path))


def find_global_stats(run_dir):
    aggregated = run_dir / "global_stats.csv"
    if aggregated.is_file():
        return aggregated
    files = sorted(run_dir.glob("global_stats_pID-*.csv"))
    return files[0] if files else None


def collect_rows(parent):
    rows = []
    for run_dir in sorted(parent.iterdir()):
        if not run_dir.is_dir():
            continue
        m = RUN_RE.match(run_dir.name)
        if not m:
            continue
        global_file = find_global_stats(run_dir)
        if global_file is None:
            continue
        density, density_err = read_global_row(global_file, "density")
        sign, sign_err = read_global_row(global_file, "sgn")
        vals = m.groupdict()
        beta = float(vals["beta"])
        rows.append(
            {
                "run_dir": run_dir,
                "U": float(vals["U"]),
                "tprime": float(vals["tprime"]),
                "mu": float(vals["mu"]),
                "Lx": int(vals["Lx"]),
                "Ly": int(vals["Ly"]),
                "beta": beta,
                "T": 1.0 / beta if beta != 0 else math.nan,
                "sid": int(vals["sid"]),
                "density": density,
                "density_err": density_err,
                "sgn": sign,
                "sgn_err": sign_err,
                "complete": run_dir.name.startswith("complete_"),
                "global_file": global_file,
            }
        )
    return rows


def bracket_or_nearest(points, target):
    """Return interpolation information for one beta group."""
    points = sorted(points, key=lambda r: r["mu"])
    if not points:
        return None
    for a, b in zip(points[:-1], points[1:]):
        da = a["density"] - target
        db = b["density"] - target
        if da == 0:
            return a["mu"], "exact", a, a
        if da * db <= 0 and b["density"] != a["density"]:
            frac = (target - a["density"]) / (b["density"] - a["density"])
            mu = a["mu"] + frac * (b["mu"] - a["mu"])
            # Rough propagated uncertainty from density errors and local slope.
            slope = (b["density"] - a["density"]) / (b["mu"] - a["mu"])
            err = math.sqrt(a["density_err"] ** 2 + b["density_err"] ** 2) / abs(slope)
            return mu, "bracket", a, b, err
    nearest = min(points, key=lambda r: abs(r["density"] - target))
    return nearest["mu"], "nearest_no_bracket", nearest, nearest, math.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parent", type=Path, help="Parent directory containing tuning run folders")
    ap.add_argument("--target-density", type=float, default=0.5)
    ap.add_argument("--out-summary", type=Path, default=None)
    ap.add_argument("--out-mu", type=Path, default=None)
    args = ap.parse_args()

    parent = args.parent.resolve()
    rows = collect_rows(parent)
    if not rows:
        raise SystemExit("No completed global_stats files found under {}".format(parent))

    summary_path = args.out_summary or parent / "density_tuning_summary.tsv"
    target_label = "{:.4g}".format(args.target_density).replace(".", "p")
    mu_path = args.out_mu or parent / ("mu_estimates_n{}.tsv".format(target_label))

    with summary_path.open("w") as f:
        f.write(
            "T\tbeta\tmu\tdensity\tdensity_err\tsgn\tsgn_err\tLx\tLy\tU\ttprime\tsid\tcomplete\trun_dir\n"
        )
        for r in sorted(rows, key=lambda x: (x["beta"], x["mu"])):
            f.write(
                "{T:.10g}\t{beta:.10g}\t{mu:.10g}\t{density:.12g}\t{density_err:.12g}\t"
                "{sgn:.12g}\t{sgn_err:.12g}\t{Lx}\t{Ly}\t{U:.10g}\t{tprime:.10g}\t{sid}\t{complete}\t{run_dir}\n".format(
                    **r
                )
            )

    groups = defaultdict(list)
    for r in rows:
        # Group by rounded beta because the folder name stores beta with two decimals.
        groups[round(r["beta"], 8)].append(r)

    with mu_path.open("w") as f:
        f.write(
            "T\tbeta\ttarget_density\tmu_estimate\tmu_err_est\tstatus\t"
            "mu_low\tn_low\tn_low_err\tmu_high\tn_high\tn_high_err\n"
        )
        for beta in sorted(groups):
            points = groups[beta]
            result = bracket_or_nearest(points, args.target_density)
            if result is None:
                continue
            mu_est, status, low, high, mu_err = result
            f.write(
                "{T:.10g}\t{beta:.10g}\t{target:.10g}\t{mu:.12g}\t{mu_err:.4g}\t{status}\t"
                "{mu_low:.10g}\t{n_low:.12g}\t{n_low_err:.12g}\t"
                "{mu_high:.10g}\t{n_high:.12g}\t{n_high_err:.12g}\n".format(
                    T=1.0 / beta,
                    beta=beta,
                    target=args.target_density,
                    mu=mu_est,
                    mu_err=mu_err,
                    status=status,
                    mu_low=low["mu"],
                    n_low=low["density"],
                    n_low_err=low["density_err"],
                    mu_high=high["mu"],
                    n_high=high["density"],
                    n_high_err=high["density_err"],
                )
            )

    print("wrote {}".format(summary_path))
    print("wrote {}".format(mu_path))


if __name__ == "__main__":
    main()
