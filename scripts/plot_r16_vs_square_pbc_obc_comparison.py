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

SQUARE_PBC_ROOT = ROOT / "results" / "spinless_square_pbc_comparison_by_L"
SQUARE_OBC_ROOT = ROOT / "results" / "spinless_square_obc_comparison_by_L"
DEFAULT_OUT_ROOT = ROOT / "results"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def column_as_float(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def disk_data(radius: int) -> dict[str, np.ndarray | int]:
    disk_dir = ROOT / "results" / "spinless_disk_comparison_by_radius" / f"radius_{radius}"
    energy_rows = read_tsv(disk_dir / "energy_vs_beta.tsv")
    specific_heat_rows = read_tsv(disk_dir / "specific_heat_vs_beta.tsv")
    thermo_rows = read_tsv(disk_dir / "thermo_vs_beta.tsv")
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
        "radius": radius,
    }


def square_pbc_data(L: int) -> dict[str, np.ndarray | int]:
    base = SQUARE_PBC_ROOT / f"L_{L}"
    thermo_rows = read_tsv(base / "square_pbc_thermo_vs_beta.tsv")
    specific_heat_rows = read_tsv(base / "square_pbc_specific_heat_vs_temperature.tsv")
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


def square_obc_data(L: int) -> dict[str, np.ndarray | int]:
    base = SQUARE_OBC_ROOT / f"L_{L}"
    thermo_rows = read_tsv(base / "square_obc_thermo_vs_beta.tsv")
    specific_heat_rows = read_tsv(base / "square_obc_specific_heat_vs_temperature.tsv")
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
    pbc_gce: np.ndarray,
    pbc_ce: np.ndarray,
    obc_gce: np.ndarray,
    obc_ce: np.ndarray,
    ylabel: str,
    title: str,
) -> None:
    order = np.argsort(temperatures)
    temps = temperatures[order]
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=180)
    ax.plot(temps, disk_gce[order], color="#1b9e77", linewidth=2.0, label="Disk OBC GCE")
    ax.plot(temps, disk_ce[order], color="#d95f02", linewidth=2.0, linestyle="--", label="Disk OBC CE")
    ax.plot(temps, pbc_gce[order], color="#7570b3", linewidth=2.0, label="Square PBC GCE")
    ax.plot(temps, pbc_ce[order], color="#e7298a", linewidth=2.0, linestyle="--", label="Square PBC CE")
    ax.plot(temps, obc_gce[order], color="#66a61e", linewidth=2.0, label="Square OBC GCE")
    ax.plot(temps, obc_ce[order], color="#e6ab02", linewidth=2.0, linestyle="--", label="Square OBC CE")
    ax.set_xlabel(r"$T = 1/\beta$", fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.set_title(title, fontsize=18)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=15, ncol=1)
    ax.tick_params(labelsize=18)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_metadata(
    path: Path,
    disk_radius: int,
    square_L: int,
    disk_nsites: int,
    disk_canonical_n: int,
    pbc_nsites: int,
    pbc_canonical_n: int,
    obc_nsites: int,
    obc_canonical_n: int,
    temperatures: np.ndarray,
) -> None:
    lines = [
        f'comparison = "Disk OBC R={disk_radius} versus square L={square_L} PBC and square L={square_L} OBC"',
        'x_axis = "temperature T = 1/beta"',
        'quantities = "energy per particle, specific heat per particle, thermodynamic compressibility, entropy per particle"',
        'curves = "disk_gce, disk_ce, square_pbc_gce, square_pbc_ce, square_obc_gce, square_obc_ce"',
        'disk_geometry = "2D square-lattice disk"',
        'disk_boundary = "open"',
        f"disk_radius = {disk_radius}",
        f"disk_nsites = {disk_nsites}",
        f"disk_canonical_n = {disk_canonical_n}",
        'square_pbc_geometry = "LxL square lattice"',
        'square_pbc_boundary = "periodic"',
        f"square_pbc_L = {square_L}",
        f"square_pbc_nsites = {pbc_nsites}",
        f"square_pbc_canonical_n = {pbc_canonical_n}",
        'square_obc_geometry = "LxL square lattice"',
        'square_obc_boundary = "open"',
        f"square_obc_L = {square_L}",
        f"square_obc_nsites = {obc_nsites}",
        f"square_obc_canonical_n = {obc_canonical_n}",
        f'temperature_grid = "{",".join(f"{value:.12f}" for value in temperatures)}"',
        f'data_sources = "results/spinless_disk_comparison_by_radius/radius_{disk_radius}, results/spinless_square_pbc_comparison_by_L/L_{square_L}, results/spinless_square_obc_comparison_by_L/L_{square_L}"',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create disk versus square-PBC/square-OBC comparison plots from saved TSV data."
    )
    parser.add_argument("--disk-radius", type=int, default=16, help="Disk radius.")
    parser.add_argument("--square-L", type=int, required=True, help="Square-lattice linear size.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory. Defaults to results/r{R}_vs_l{L}pbc_vs_l{L}obc_comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    disk = disk_data(args.disk_radius)
    pbc = square_pbc_data(args.square_L)
    obc = square_obc_data(args.square_L)

    temperatures = np.asarray(disk["temperature"], dtype=float)
    for label, other in [("square PBC", pbc), ("square OBC", obc)]:
        other_temperatures = np.asarray(other["temperature"], dtype=float)
        if temperatures.shape != other_temperatures.shape or not np.allclose(
            temperatures, other_temperatures, atol=1e-12, rtol=0.0
        ):
            raise RuntimeError(f"Temperature grid does not match for {label}.")

    out_dir = args.out_dir or (
        DEFAULT_OUT_ROOT / f"r{args.disk_radius}_vs_l{args.square_L}pbc_vs_l{args.square_L}obc_comparison"
    )
    title = (
        f"Disk: R={args.disk_radius}, V={disk['nsites']}   |   "
        f"Square PBC/OBC: L={args.square_L}, V={pbc['nsites']}"
    )

    make_plot(
        out_dir / "energy_vs_temperature.png",
        temperatures,
        np.asarray(disk["energy_gce"], dtype=float),
        np.asarray(disk["energy_ce"], dtype=float),
        np.asarray(pbc["energy_gce"], dtype=float),
        np.asarray(pbc["energy_ce"], dtype=float),
        np.asarray(obc["energy_gce"], dtype=float),
        np.asarray(obc["energy_ce"], dtype=float),
        ylabel="Energy per particle",
        title=title,
    )
    make_plot(
        out_dir / "specific_heat_vs_temperature.png",
        temperatures,
        np.asarray(disk["specific_heat_gce"], dtype=float),
        np.asarray(disk["specific_heat_ce"], dtype=float),
        np.asarray(pbc["specific_heat_gce"], dtype=float),
        np.asarray(pbc["specific_heat_ce"], dtype=float),
        np.asarray(obc["specific_heat_gce"], dtype=float),
        np.asarray(obc["specific_heat_ce"], dtype=float),
        ylabel="Specific heat per particle",
        title=title,
    )
    make_plot(
        out_dir / "compressibility_vs_temperature.png",
        temperatures,
        np.asarray(disk["kappa_gce"], dtype=float),
        np.asarray(disk["kappa_ce"], dtype=float),
        np.asarray(pbc["kappa_gce"], dtype=float),
        np.asarray(pbc["kappa_ce"], dtype=float),
        np.asarray(obc["kappa_gce"], dtype=float),
        np.asarray(obc["kappa_ce"], dtype=float),
        ylabel="Compressibility",
        title=title,
    )
    make_plot(
        out_dir / "entropy_vs_temperature.png",
        temperatures,
        np.asarray(disk["entropy_gce"], dtype=float),
        np.asarray(disk["entropy_ce"], dtype=float),
        np.asarray(pbc["entropy_gce"], dtype=float),
        np.asarray(pbc["entropy_ce"], dtype=float),
        np.asarray(obc["entropy_gce"], dtype=float),
        np.asarray(obc["entropy_ce"], dtype=float),
        ylabel="Entropy per particle",
        title=title,
    )
    write_metadata(
        out_dir / "comparison_metadata.toml",
        disk_radius=args.disk_radius,
        square_L=args.square_L,
        disk_nsites=int(disk["nsites"]),
        disk_canonical_n=int(disk["canonical_n"]),
        pbc_nsites=int(pbc["nsites"]),
        pbc_canonical_n=int(pbc["canonical_n"]),
        obc_nsites=int(obc["nsites"]),
        obc_canonical_n=int(obc["canonical_n"]),
        temperatures=temperatures,
    )


if __name__ == "__main__":
    main()
