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


PBC_ROOT = ROOT / "results" / "non_interacting" / "spinless_square_pbc_comparison_by_L"
OBC_ROOT = ROOT / "results" / "non_interacting" / "spinless_square_obc_comparison_by_L"
DEFAULT_OUT_ROOT = ROOT / "results" / "non_interacting"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def column_as_float(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def square_pbc_data(L: int) -> dict[str, np.ndarray | int]:
    base = PBC_ROOT / f"L_{L}"
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
    base = OBC_ROOT / f"L_{L}"
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
    obc_gce: np.ndarray,
    obc_ce: np.ndarray,
    pbc_gce: np.ndarray,
    pbc_ce: np.ndarray,
    ylabel: str,
    title: str,
) -> None:
    order = np.argsort(temperatures)
    temps = temperatures[order]
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=180)
    ax.plot(temps, obc_gce[order], color="#8c510a", linewidth=2.0, label="OBC GCE")
    ax.plot(temps, obc_ce[order], color="#01665e", linewidth=2.0, linestyle="--", label="OBC CE")
    ax.plot(temps, pbc_gce[order], color="#7570b3", linewidth=2.0, label="PBC GCE")
    ax.plot(temps, pbc_ce[order], color="#e7298a", linewidth=2.0, linestyle="--", label="PBC CE")
    ax.set_xlabel(r"$T = 1/\beta$", fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.set_title(title, fontsize=18)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, fontsize=16, ncol=1)
    ax.tick_params(labelsize=18)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_metadata(
    path: Path,
    square_L: int,
    pbc_nsites: int,
    pbc_canonical_n: int,
    obc_nsites: int,
    obc_canonical_n: int,
    temperatures: np.ndarray,
) -> None:
    lines = [
        f'comparison = "Square L={square_L} OBC versus square L={square_L} PBC"',
        'x_axis = "temperature T = 1/beta"',
        'quantities = "energy per particle, specific heat per particle, thermodynamic compressibility, entropy per particle"',
        'curves = "square_obc_gce, square_obc_ce, square_pbc_gce, square_pbc_ce"',
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
        f'data_sources = "results/non_interacting/spinless_square_pbc_comparison_by_L/L_{square_L}, results/non_interacting/spinless_square_obc_comparison_by_L/L_{square_L}"',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def compare_L(L: int, out_root: Path) -> None:
    pbc = square_pbc_data(L)
    obc = square_obc_data(L)
    temperatures = np.asarray(pbc["temperature"], dtype=float)
    obc_temperatures = np.asarray(obc["temperature"], dtype=float)
    if temperatures.shape != obc_temperatures.shape or not np.allclose(
        temperatures, obc_temperatures, atol=1e-12, rtol=0.0
    ):
        raise RuntimeError(f"Temperature grid does not match for L={L}")

    out_dir = out_root / f"l{L}_pbc_vs_obc_comparison"
    title = f"Square: L={L}, V={pbc['nsites']} | OBC vs PBC"

    make_plot(
        out_dir / "energy_vs_temperature.png",
        temperatures,
        np.asarray(obc["energy_gce"], dtype=float),
        np.asarray(obc["energy_ce"], dtype=float),
        np.asarray(pbc["energy_gce"], dtype=float),
        np.asarray(pbc["energy_ce"], dtype=float),
        ylabel="Energy per particle",
        title=title,
    )
    make_plot(
        out_dir / "specific_heat_vs_temperature.png",
        temperatures,
        np.asarray(obc["specific_heat_gce"], dtype=float),
        np.asarray(obc["specific_heat_ce"], dtype=float),
        np.asarray(pbc["specific_heat_gce"], dtype=float),
        np.asarray(pbc["specific_heat_ce"], dtype=float),
        ylabel="Specific heat per particle",
        title=title,
    )
    make_plot(
        out_dir / "compressibility_vs_temperature.png",
        temperatures,
        np.asarray(obc["kappa_gce"], dtype=float),
        np.asarray(obc["kappa_ce"], dtype=float),
        np.asarray(pbc["kappa_gce"], dtype=float),
        np.asarray(pbc["kappa_ce"], dtype=float),
        ylabel="Compressibility",
        title=title,
    )
    make_plot(
        out_dir / "entropy_vs_temperature.png",
        temperatures,
        np.asarray(obc["entropy_gce"], dtype=float),
        np.asarray(obc["entropy_ce"], dtype=float),
        np.asarray(pbc["entropy_gce"], dtype=float),
        np.asarray(pbc["entropy_ce"], dtype=float),
        ylabel="Entropy per particle",
        title=title,
    )
    write_metadata(
        out_dir / "comparison_metadata.toml",
        square_L=L,
        pbc_nsites=int(pbc["nsites"]),
        pbc_canonical_n=int(pbc["canonical_n"]),
        obc_nsites=int(obc["nsites"]),
        obc_canonical_n=int(obc["canonical_n"]),
        temperatures=temperatures,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create square PBC versus OBC comparison plots from saved TSV data."
    )
    parser.add_argument(
        "--Ls",
        default="4,8,16,32,72",
        help="Comma-separated linear system sizes.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Root directory for comparison folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Ls = [int(item.strip()) for item in args.Ls.split(",") if item.strip()]
    for L in Ls:
        compare_L(L, args.out_root)
    print("Created square PBC vs OBC comparison plots")
    print(f"  L values = {', '.join(str(L) for L in Ls)}")
    print(f"  output root = {args.out_root}")


if __name__ == "__main__":
    main()
