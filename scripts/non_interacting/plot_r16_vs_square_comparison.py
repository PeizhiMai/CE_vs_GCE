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


DISK_DIR = ROOT / "results" / "non_interacting" / "spinless_disk_comparison_by_radius" / "radius_16"
SQUARE_RESULTS_DIR = ROOT / "results" / "non_interacting" / "spinless_square_pbc_comparison_by_L"
DEFAULT_OUT_ROOT = ROOT / "results" / "non_interacting"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def column_as_float(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def disk_data() -> dict[str, np.ndarray | int]:
    energy_rows = read_tsv(DISK_DIR / "energy_vs_beta.tsv")
    specific_heat_rows = read_tsv(DISK_DIR / "specific_heat_vs_beta.tsv")
    thermo_rows = read_tsv(DISK_DIR / "thermo_vs_beta.tsv")
    return {
        "temperature": column_as_float(energy_rows, "temperature"),
        "energy_gce": column_as_float(energy_rows, "energy_per_particle_grand_canonical"),
        "energy_ce": column_as_float(energy_rows, "energy_per_particle_canonical"),
        "specific_heat_gce": column_as_float(
            specific_heat_rows, "specific_heat_per_particle_grand_canonical"
        ),
        "specific_heat_ce": column_as_float(
            specific_heat_rows, "specific_heat_per_particle_canonical"
        ),
        "kappa_gce": column_as_float(thermo_rows, "kappa_grand_canonical"),
        "kappa_ce": column_as_float(thermo_rows, "kappa_canonical"),
        "entropy_gce": column_as_float(thermo_rows, "entropy_per_particle_grand_canonical"),
        "entropy_ce": column_as_float(thermo_rows, "entropy_per_particle_canonical"),
        "nsites": int(energy_rows[0]["nsites"]),
        "canonical_n": int(energy_rows[0]["canonical_n"]),
    }


def square_data(L: int) -> dict[str, np.ndarray | int]:
    l_dir = SQUARE_RESULTS_DIR / f"L_{L}"
    thermo_rows = read_tsv(l_dir / "square_pbc_thermo_vs_beta.tsv")
    specific_heat_rows = read_tsv(l_dir / "square_pbc_specific_heat_vs_temperature.tsv")
    return {
        "temperature": column_as_float(thermo_rows, "temperature"),
        "energy_gce": column_as_float(thermo_rows, "energy_per_particle_grand_canonical"),
        "energy_ce": column_as_float(thermo_rows, "energy_per_particle_canonical"),
        "specific_heat_gce": column_as_float(
            specific_heat_rows, "specific_heat_per_particle_grand_canonical"
        ),
        "specific_heat_ce": column_as_float(
            specific_heat_rows, "specific_heat_per_particle_canonical"
        ),
        "kappa_gce": column_as_float(thermo_rows, "kappa_grand_canonical"),
        "kappa_ce": column_as_float(thermo_rows, "kappa_canonical"),
        "entropy_gce": column_as_float(thermo_rows, "entropy_per_particle_grand_canonical"),
        "entropy_ce": column_as_float(thermo_rows, "entropy_per_particle_canonical"),
        "nsites": int(thermo_rows[0]["nsites"]),
        "canonical_n": int(thermo_rows[0]["canonical_n"]),
    }


def make_plot(
    path: Path,
    temperatures: np.ndarray,
    disk_gce: np.ndarray,
    disk_ce: np.ndarray,
    square_gce: np.ndarray,
    square_ce: np.ndarray,
    ylabel: str,
    title: str,
    note: str,
) -> None:
    order = np.argsort(temperatures)
    temps = temperatures[order]
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=180)
    ax.plot(temps, disk_gce[order], color="#1b9e77", linewidth=2.0, label="Disk OBC GCE")
    ax.plot(temps, disk_ce[order], color="#d95f02", linewidth=2.0, linestyle="--", label="Disk OBC CE")
    ax.plot(temps, square_gce[order], color="#7570b3", linewidth=2.0, label="Square PBC GCE")
    ax.plot(temps, square_ce[order], color="#e7298a", linewidth=2.0, linestyle="--", label="Square PBC CE")
    ax.set_xlabel(r"$T = 1/\beta$", fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.set_title(title, fontsize=24)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=18)
    ax.tick_params(labelsize=18)
    ax.text(
        0.5,
        0.98,
        note,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=18,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_metadata(
    path: Path,
    square_L: int,
    disk_nsites: int,
    disk_canonical_n: int,
    square_nsites: int,
    square_canonical_n: int,
    temperatures: np.ndarray,
) -> None:
    lines = [
        f'comparison = "Disk OBC R=16 versus square PBC L={square_L}"',
        'x_axis = "temperature T = 1/beta"',
        'quantities = "energy per particle, specific heat per particle, thermodynamic compressibility, entropy per particle"',
        'curves = "disk_gce, disk_ce, square_gce, square_ce"',
        'disk_geometry = "2D square-lattice disk"',
        'disk_boundary = "open"',
        "disk_radius = 16",
        f"disk_nsites = {disk_nsites}",
        f"disk_canonical_n = {disk_canonical_n}",
        'square_geometry = "LxL square lattice"',
        'square_boundary = "periodic"',
        f"square_L = {square_L}",
        f"square_nsites = {square_nsites}",
        f"square_canonical_n = {square_canonical_n}",
        f'temperature_grid = "{",".join(f"{value:.12f}" for value in temperatures)}"',
        f'data_sources = "results/non_interacting/spinless_disk_comparison_by_radius/radius_16 and results/non_interacting/spinless_square_pbc_comparison_by_L/L_{square_L}"',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create R=16 disk versus square-PBC comparison plots from saved TSV data."
    )
    parser.add_argument("--square-L", type=int, required=True, help="Square-lattice linear size.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory. Defaults to results/r16_vs_l{L}_comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    disk = disk_data()
    square = square_data(args.square_L)

    disk_temperatures = np.asarray(disk["temperature"], dtype=float)
    square_temperatures = np.asarray(square["temperature"], dtype=float)
    if disk_temperatures.shape != square_temperatures.shape or not np.allclose(
        disk_temperatures, square_temperatures, atol=1e-12, rtol=0.0
    ):
        raise RuntimeError("Disk and square temperature grids do not match.")

    out_dir = args.out_dir or (DEFAULT_OUT_ROOT / f"r16_vs_l{args.square_L}_comparison")
    note = (
        f"Disk: R=16, V={disk['nsites']}, N={disk['canonical_n']}   |   "
        f"Square PBC: L={args.square_L}, V={square['nsites']}, N={square['canonical_n']}"
    )

    make_plot(
        out_dir / "energy_vs_temperature.png",
        disk_temperatures,
        np.asarray(disk["energy_gce"], dtype=float),
        np.asarray(disk["energy_ce"], dtype=float),
        np.asarray(square["energy_gce"], dtype=float),
        np.asarray(square["energy_ce"], dtype=float),
        ylabel="Energy per particle",
        title=f"R=16 Disk vs L={args.square_L} Square PBC Energy",
        note=note,
    )
    make_plot(
        out_dir / "specific_heat_vs_temperature.png",
        disk_temperatures,
        np.asarray(disk["specific_heat_gce"], dtype=float),
        np.asarray(disk["specific_heat_ce"], dtype=float),
        np.asarray(square["specific_heat_gce"], dtype=float),
        np.asarray(square["specific_heat_ce"], dtype=float),
        ylabel="Specific heat per particle",
        title=f"R=16 Disk vs L={args.square_L} Square PBC Specific Heat",
        note=note,
    )
    make_plot(
        out_dir / "compressibility_vs_temperature.png",
        disk_temperatures,
        np.asarray(disk["kappa_gce"], dtype=float),
        np.asarray(disk["kappa_ce"], dtype=float),
        np.asarray(square["kappa_gce"], dtype=float),
        np.asarray(square["kappa_ce"], dtype=float),
        ylabel="Compressibility",
        title=f"R=16 Disk vs L={args.square_L} Square PBC Compressibility",
        note=note,
    )
    make_plot(
        out_dir / "entropy_vs_temperature.png",
        disk_temperatures,
        np.asarray(disk["entropy_gce"], dtype=float),
        np.asarray(disk["entropy_ce"], dtype=float),
        np.asarray(square["entropy_gce"], dtype=float),
        np.asarray(square["entropy_ce"], dtype=float),
        ylabel="Entropy per particle",
        title=f"R=16 Disk vs L={args.square_L} Square PBC Entropy",
        note=note,
    )
    write_metadata(
        out_dir / "comparison_metadata.toml",
        square_L=args.square_L,
        disk_nsites=int(disk["nsites"]),
        disk_canonical_n=int(disk["canonical_n"]),
        square_nsites=int(square["nsites"]),
        square_canonical_n=int(square["canonical_n"]),
        temperatures=disk_temperatures,
    )


if __name__ == "__main__":
    main()
