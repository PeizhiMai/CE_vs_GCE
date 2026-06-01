#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_single_row(path: Path) -> dict[str, str]:
    with path.open() as fh:
        return next(csv.DictReader(fh, delimiter="\t"))


def f(x: str) -> float:
    return float(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare CE-QMC current-response benchmark against ED.")
    p.add_argument("--ed-summary", type=Path, required=True)
    p.add_argument("--qmc-summary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    ed = read_single_row(args.ed_summary)
    qmc = read_single_row(args.qmc_summary)

    rows = [
        ("lambda_longitudinal_qmin0", f(ed["lambda_longitudinal_qmin0"]), f(qmc["lambda_longitudinal_qmin0"]), f(qmc["lambda_longitudinal_stderr"])),
        ("lambda_transverse_0qmin", f(ed["lambda_transverse_0qmin"]), f(qmc["lambda_transverse_0qmin"]), f(qmc["lambda_transverse_stderr"])),
        ("Kx_per_site", f(ed["Kx_per_site"]), f(qmc["Kx_per_site"]), f(qmc["Kx_stderr"])),
        ("rho_s_current", f(ed["rho_s_current"]), f(qmc["rho_s_current"]), f(qmc["rho_s_current_stderr"])),
        ("rho_s_diamagnetic", f(ed["rho_s_diamagnetic"]), f(qmc["rho_s_diamagnetic"]), f(qmc["rho_s_diamagnetic_stderr"])),
    ]

    with args.out.open("w", newline="") as fh:
        wr = csv.DictWriter(
            fh,
            fieldnames=["observable", "ed", "qmc", "qmc_stderr", "delta_qmc_minus_ed"],
            delimiter="\t",
        )
        wr.writeheader()
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

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
