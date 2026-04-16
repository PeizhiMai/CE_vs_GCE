#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/Users/cosdis/Desktop/projects/CE_GCE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay 4x2 ED and CE QMC energies.")
    parser.add_argument("--ed-path", type=Path, required=True)
    parser.add_argument("--qmc-path", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def read_ed_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "beta": float(row["beta"]),
                    "e_particle": float(row["total_energy_per_particle"]),
                }
            )
    return rows


def read_qmc_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "beta": float(row["beta"]),
                    "e_particle": float(row["total_per_particle"]),
                    "stderr": float(row["total_stderr"]),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    ed_rows = read_ed_rows(args.ed_path)
    qmc_rows = read_qmc_rows(args.qmc_path)

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=180)
    ax.plot(
        [row["beta"] for row in ed_rows],
        [row["e_particle"] for row in ed_rows],
        color="#1b6ca8",
        linewidth=2.0,
        label="ED",
    )

    if qmc_rows:
        ax.errorbar(
            [row["beta"] for row in qmc_rows],
            [row["e_particle"] for row in qmc_rows],
            yerr=[row["stderr"] for row in qmc_rows],
            color="#b03a2e",
            marker="o",
            markersize=5,
            linestyle="none",
            capsize=3,
            label="CE QMC",
        )

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("E/N")
    ax.set_xlim(0.0, 12.0)
    ax.set_title("4x2 Hubbard: ED vs CE QMC")
    ax.text(
        0.97,
        0.97,
        "4x2 lattice\nNup=Ndn=4\nU=4, t=1",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.75", alpha=0.9),
    )
    ax.grid(False)
    ax.legend(frameon=False, loc="center right")

    args.outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.outdir / "ed_vs_ce_qmc.png", bbox_inches="tight")
    fig.savefig(args.outdir / "ed_vs_ce_qmc.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
