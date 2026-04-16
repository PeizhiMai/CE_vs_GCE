#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/Users/cosdis/Desktop/projects/CE_GCE")
DEFAULT_INPUT = ROOT / "results" / "interacting_qmc_ed" / "ed_hubbard_4x2_half_filling_beta_0p1_grid" / "energy_vs_beta.tsv"
DEFAULT_OUTDIR = ROOT / "results" / "interacting_qmc_ed" / "ed_hubbard_4x2_half_filling_beta_0p1_grid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ED Hubbard energy vs beta.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input TSV path.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory for plots.")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "beta": float(row["beta"]),
                    "total_energy": float(row["total_energy"]),
                    "total_energy_per_site": float(row["total_energy_per_site"]),
                    "total_energy_per_particle": float(row["total_energy_per_particle"]),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)

    beta = [row["beta"] for row in rows]
    e_particle = [row["total_energy_per_particle"] for row in rows]
    e_total = [row["total_energy"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), dpi=180)

    axes[0].plot(beta, e_particle, color="#1b6ca8", linewidth=2.0)
    axes[0].set_xlabel("beta")
    axes[0].set_ylabel("E/N")
    axes[0].set_title("ED Hubbard Energy per Particle")
    axes[0].grid(False)

    axes[1].plot(beta, e_total, color="#b03a2e", linewidth=2.0)
    axes[1].set_xlabel("beta")
    axes[1].set_ylabel("E")
    axes[1].set_title("ED Hubbard Total Energy")
    axes[1].grid(False)

    fig.suptitle("4x2 Hubbard ED, Nup=Ndn=4, U=4, t=1")
    fig.tight_layout()

    args.outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.outdir / "energy_vs_beta.png", bbox_inches="tight")
    fig.savefig(args.outdir / "energy_vs_beta.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
