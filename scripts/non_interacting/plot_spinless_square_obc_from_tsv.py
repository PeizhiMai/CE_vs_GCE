#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

ROOT = Path("/Users/cosdis/Desktop/projects/CE_GCE")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULTS_DIR = ROOT / "results" / "non_interacting" / "spinless_square_obc_comparison_by_L"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def column_as_float(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def make_plot(
    path: Path,
    L: int,
    nsites: int,
    canonical_n: int,
    temperatures: np.ndarray,
    gce_values: np.ndarray,
    ce_values: np.ndarray,
    ylabel: str,
) -> None:
    order = np.argsort(temperatures)
    temps = temperatures[order]
    note = f"L={L}, V={nsites}, N={canonical_n}, GCE tuned to <N>={canonical_n}"
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=180)
    ax.plot(
        temps,
        gce_values[order],
        color="#8c510a",
        linewidth=2.4,
        linestyle="-",
        label="GCE",
    )
    ax.plot(
        temps,
        ce_values[order],
        color="#01665e",
        linewidth=2.2,
        linestyle="--",
        label="CE",
    )
    ax.set_xlabel(r"$T = 1/\beta$", fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.set_title(note, fontsize=18)
    ax.grid(True, alpha=0.18)
    ax.legend(frameon=False, fontsize=18)
    ax.tick_params(labelsize=18)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def replot_L(results_dir: Path, L: int) -> None:
    l_dir = results_dir / f"L_{L}"
    thermo_rows = read_tsv(l_dir / "square_obc_thermo_vs_beta.tsv")
    specific_heat_rows = read_tsv(l_dir / "square_obc_specific_heat_vs_temperature.tsv")

    temperatures = column_as_float(thermo_rows, "temperature")
    nsites = int(thermo_rows[0]["nsites"])
    canonical_n = int(thermo_rows[0]["canonical_n"])

    make_plot(
        l_dir / "energy_vs_temperature.png",
        L=L,
        nsites=nsites,
        canonical_n=canonical_n,
        temperatures=temperatures,
        gce_values=column_as_float(thermo_rows, "energy_per_particle_grand_canonical"),
        ce_values=column_as_float(thermo_rows, "energy_per_particle_canonical"),
        ylabel="Energy per particle",
    )
    make_plot(
        l_dir / "specific_heat_vs_temperature.png",
        L=L,
        nsites=nsites,
        canonical_n=canonical_n,
        temperatures=column_as_float(specific_heat_rows, "temperature"),
        gce_values=column_as_float(
            specific_heat_rows, "specific_heat_per_particle_grand_canonical"
        ),
        ce_values=column_as_float(
            specific_heat_rows, "specific_heat_per_particle_canonical"
        ),
        ylabel="Specific heat per particle",
    )
    make_plot(
        l_dir / "compressibility_vs_temperature.png",
        L=L,
        nsites=nsites,
        canonical_n=canonical_n,
        temperatures=temperatures,
        gce_values=column_as_float(thermo_rows, "kappa_grand_canonical"),
        ce_values=column_as_float(thermo_rows, "kappa_canonical"),
        ylabel="Compressibility",
    )
    make_plot(
        l_dir / "entropy_vs_temperature.png",
        L=L,
        nsites=nsites,
        canonical_n=canonical_n,
        temperatures=temperatures,
        gce_values=column_as_float(thermo_rows, "entropy_per_particle_grand_canonical"),
        ce_values=column_as_float(thermo_rows, "entropy_per_particle_canonical"),
        ylabel="Entropy per particle",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replot saved square-lattice OBC CE/GCE results from existing TSV files."
    )
    parser.add_argument(
        "--Ls",
        default="4,6,8,12,16,24,32",
        help="Comma-separated linear system sizes to replot.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing existing square-lattice OBC result folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Ls = [int(item.strip()) for item in args.Ls.split(",") if item.strip()]
    for L in Ls:
        replot_L(args.results_dir, L)
    print("Replotted saved square-lattice OBC figures from TSV data")
    print(f"  L values = {', '.join(str(L) for L in Ls)}")
    print(f"  results directory = {args.results_dir}")


if __name__ == "__main__":
    main()
