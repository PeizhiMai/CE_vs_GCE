#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_single_row(path: Path) -> dict[str, str]:
    with path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return next(reader)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def f(x: str) -> float:
    return float(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare 3x3 CE-QMC benchmark output against low-temperature ED.")
    p.add_argument("--ed-dir", type=Path, required=True)
    p.add_argument("--qmc-dir", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    ed_summary = read_single_row(args.ed_dir / "summary.tsv")
    ed_corr = read_rows(args.ed_dir / "correlations.tsv")
    qmc_summary = read_single_row(args.qmc_dir / "summary.tsv")
    qmc_corr = read_rows(args.qmc_dir / "correlations.tsv")

    with (args.outdir / "summary_comparison.tsv").open("w", newline="") as fh:
        cols = ["observable", "ed", "qmc", "qmc_stderr", "delta_qmc_minus_ed"]
        wr = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        wr.writeheader()
        rows = [
            ("energy_per_site", f(ed_summary["energy_per_site"]), f(qmc_summary["total_per_site"]), f(qmc_summary["total_stderr"])),
            ("kinetic_per_site", f(ed_summary["kinetic_per_site"]), f(qmc_summary["kinetic_per_site"]), f(qmc_summary["kinetic_stderr"])),
            ("potential_per_site", f(ed_summary["potential_per_site"]), f(qmc_summary["potential_per_site"]), f(qmc_summary["potential_stderr"])),
            ("double_occupancy_per_site", f(ed_summary["double_occupancy_per_site"]), f(qmc_summary["double_occupancy_per_site"]), f(qmc_summary["double_occupancy_stderr"])),
        ]
        for name, ed_val, qmc_val, qmc_err in rows:
            wr.writerow(
                {
                    "observable": name,
                    "ed": ed_val,
                    "qmc": qmc_val,
                    "qmc_stderr": qmc_err,
                    "delta_qmc_minus_ed": qmc_val - ed_val,
                }
            )

    qmc_corr_by_key = {(row["dx"], row["dy"]): row for row in qmc_corr}
    with (args.outdir / "correlation_comparison.tsv").open("w", newline="") as fh:
        cols = [
            "dx",
            "dy",
            "observable",
            "ed",
            "qmc",
            "qmc_stderr",
            "delta_qmc_minus_ed",
        ]
        wr = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        wr.writeheader()
        for ed_row in ed_corr:
            key = (ed_row["dx"], ed_row["dy"])
            qmc_row = qmc_corr_by_key[key]
            for obs, qmc_err_key in [
                ("charge_corr", "charge_stderr"),
                ("spin_z_corr", "spin_z_stderr"),
                ("pair_corr", "pair_stderr"),
            ]:
                ed_val = f(ed_row[obs])
                qmc_val = f(qmc_row[obs])
                qmc_err = f(qmc_row[qmc_err_key])
                wr.writerow(
                    {
                        "dx": ed_row["dx"],
                        "dy": ed_row["dy"],
                        "observable": obs,
                        "ed": ed_val,
                        "qmc": qmc_val,
                        "qmc_stderr": qmc_err,
                        "delta_qmc_minus_ed": qmc_val - ed_val,
                    }
                )

    print(f"Wrote {args.outdir / 'summary_comparison.tsv'}")
    print(f"Wrote {args.outdir / 'correlation_comparison.tsv'}")


if __name__ == "__main__":
    main()
