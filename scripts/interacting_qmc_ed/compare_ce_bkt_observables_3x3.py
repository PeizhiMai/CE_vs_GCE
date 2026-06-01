#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from typing import Dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare CE-QMC and ED BKT/superfluid-stiffness observables.")
    p.add_argument("--qmc-bkt", type=Path, required=True, help="Path to bkt_observables_qmc.tsv")
    p.add_argument("--ed-bkt", type=Path, required=True, help="Path to bkt_observables_ed.tsv")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def read_single_row(path: Path) -> Dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if len(rows) != 1:
        raise SystemExit(f"expected exactly one row in {path}, found {len(rows)}")
    return rows[0]


def f(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    val = row.get(key, "")
    if val == "":
        return default
    return float(val)


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    qmc = read_single_row(args.qmc_bkt)
    ed = read_single_row(args.ed_bkt)

    observables = [
        ("lambda_longitudinal_qmin0", "lambda_longitudinal_stderr"),
        ("lambda_transverse_0qmin", "lambda_transverse_stderr"),
        ("Kx_per_site", "Kx_stderr"),
        ("diamagnetic_minus_Kx_per_site", "diamagnetic_minus_Kx_stderr"),
        ("rho_s_current", "rho_s_current_stderr"),
        ("rho_s_diamagnetic", "rho_s_diamagnetic_stderr"),
        ("bkt_residual_current", "bkt_residual_current_stderr"),
        ("bkt_residual_diamagnetic", "bkt_residual_diamagnetic_stderr"),
    ]

    with args.out.open("w", newline="") as fh:
        cols = [
            "observable",
            "ed",
            "qmc",
            "qmc_stderr",
            "delta_qmc_minus_ed",
            "z_abs",
        ]
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for obs, err_col in observables:
            ed_val = f(ed, obs)
            qmc_val = f(qmc, obs)
            qmc_err = f(qmc, err_col)
            delta = qmc_val - ed_val
            z = abs(delta) / qmc_err if qmc_err > 0.0 and math.isfinite(qmc_err) else float("nan")
            w.writerow(
                {
                    "observable": obs,
                    "ed": ed_val,
                    "qmc": qmc_val,
                    "qmc_stderr": qmc_err,
                    "delta_qmc_minus_ed": delta,
                    "z_abs": z,
                }
            )

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
