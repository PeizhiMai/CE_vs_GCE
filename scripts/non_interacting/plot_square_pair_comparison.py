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


DEFAULT_OUT_ROOT = ROOT / "results" / "non_interacting"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def column_as_float(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def load_square_data(base: Path, boundary: str) -> dict[str, np.ndarray | int | str]:
    if boundary == "pbc":
        thermo_path = base / "square_pbc_thermo_vs_beta.tsv"
        specific_heat_path = base / "square_pbc_specific_heat_vs_temperature.tsv"
    else:
        thermo_path = base / "square_obc_thermo_vs_beta.tsv"
        specific_heat_path = base / "square_obc_specific_heat_vs_temperature.tsv"

    thermo_rows = read_tsv(thermo_path)
    specific_heat_rows = read_tsv(specific_heat_path)
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
        "folder": base.name,
    }


def make_plot(
    path: Path,
    temperatures: np.ndarray,
    a_gce: np.ndarray,
    a_ce: np.ndarray,
    b_gce: np.ndarray,
    b_ce: np.ndarray,
    ylabel: str,
    title: str,
    label_a: str,
    label_b: str,
) -> None:
    order = np.argsort(temperatures)
    temps = temperatures[order]
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=180)
    ax.plot(temps, a_gce[order], color="#1b9e77", linewidth=2.1, label=f"{label_a} GCE")
    ax.plot(temps, a_ce[order], color="#d95f02", linewidth=2.1, linestyle="--", label=f"{label_a} CE")
    ax.plot(temps, b_gce[order], color="#7570b3", linewidth=2.1, label=f"{label_b} GCE")
    ax.plot(temps, b_ce[order], color="#e7298a", linewidth=2.1, linestyle="--", label=f"{label_b} CE")
    ax.set_xlabel(r"$T = 1/\beta$", fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.set_title(title, fontsize=18)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=14)
    ax.tick_params(labelsize=18)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_metadata(
    path: Path,
    boundary: str,
    folder_a: str,
    folder_b: str,
    nsites_a: int,
    canonical_n_a: int,
    nsites_b: int,
    canonical_n_b: int,
    temperatures: np.ndarray,
) -> None:
    lines = [
        f'comparison = "Square {boundary.upper()} comparison: {folder_a} versus {folder_b}"',
        'x_axis = "temperature T = 1/beta"',
        'quantities = "energy per particle, specific heat per particle, thermodynamic compressibility, entropy per particle"',
        'curves = "system_a_gce, system_a_ce, system_b_gce, system_b_ce"',
        f'boundary = "{boundary}"',
        f'system_a_folder = "{folder_a}"',
        f"system_a_nsites = {nsites_a}",
        f"system_a_canonical_n = {canonical_n_a}",
        f'system_b_folder = "{folder_b}"',
        f"system_b_nsites = {nsites_b}",
        f"system_b_canonical_n = {canonical_n_b}",
        f'temperature_grid = "{",".join(f"{value:.12f}" for value in temperatures)}"',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create comparison plots between two saved square-lattice CE/GCE result folders."
    )
    parser.add_argument("--boundary", choices=["pbc", "obc"], required=True, help="Boundary condition.")
    parser.add_argument("--dir-a", type=Path, required=True, help="First square result folder.")
    parser.add_argument("--dir-b", type=Path, required=True, help="Second square result folder.")
    parser.add_argument("--label-a", required=True, help="Legend label for the first folder.")
    parser.add_argument("--label-b", required=True, help="Legend label for the second folder.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for comparison plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_a = load_square_data(args.dir_a.resolve(), args.boundary)
    data_b = load_square_data(args.dir_b.resolve(), args.boundary)

    temperatures_a = np.asarray(data_a["temperature"], dtype=float)
    temperatures_b = np.asarray(data_b["temperature"], dtype=float)
    if temperatures_a.shape != temperatures_b.shape or not np.allclose(
        temperatures_a, temperatures_b, atol=1e-12, rtol=0.0
    ):
        raise RuntimeError("Temperature grids do not match between the two inputs.")

    boundary_label = args.boundary.upper()
    title_prefix = f"Square {boundary_label}: {args.label_a} vs {args.label_b}"

    make_plot(
        args.out_dir / "energy_vs_temperature.png",
        temperatures_a,
        np.asarray(data_a["energy_gce"], dtype=float),
        np.asarray(data_a["energy_ce"], dtype=float),
        np.asarray(data_b["energy_gce"], dtype=float),
        np.asarray(data_b["energy_ce"], dtype=float),
        ylabel="Energy per particle",
        title=f"{title_prefix} Energy",
        label_a=args.label_a,
        label_b=args.label_b,
    )
    make_plot(
        args.out_dir / "specific_heat_vs_temperature.png",
        temperatures_a,
        np.asarray(data_a["specific_heat_gce"], dtype=float),
        np.asarray(data_a["specific_heat_ce"], dtype=float),
        np.asarray(data_b["specific_heat_gce"], dtype=float),
        np.asarray(data_b["specific_heat_ce"], dtype=float),
        ylabel="Specific heat per particle",
        title=f"{title_prefix} Specific Heat",
        label_a=args.label_a,
        label_b=args.label_b,
    )
    make_plot(
        args.out_dir / "compressibility_vs_temperature.png",
        temperatures_a,
        np.asarray(data_a["kappa_gce"], dtype=float),
        np.asarray(data_a["kappa_ce"], dtype=float),
        np.asarray(data_b["kappa_gce"], dtype=float),
        np.asarray(data_b["kappa_ce"], dtype=float),
        ylabel="Compressibility",
        title=f"{title_prefix} Compressibility",
        label_a=args.label_a,
        label_b=args.label_b,
    )
    make_plot(
        args.out_dir / "entropy_vs_temperature.png",
        temperatures_a,
        np.asarray(data_a["entropy_gce"], dtype=float),
        np.asarray(data_a["entropy_ce"], dtype=float),
        np.asarray(data_b["entropy_gce"], dtype=float),
        np.asarray(data_b["entropy_ce"], dtype=float),
        ylabel="Entropy per particle",
        title=f"{title_prefix} Entropy",
        label_a=args.label_a,
        label_b=args.label_b,
    )
    write_metadata(
        args.out_dir / "comparison_metadata.toml",
        boundary=args.boundary,
        folder_a=str(data_a["folder"]),
        folder_b=str(data_b["folder"]),
        nsites_a=int(data_a["nsites"]),
        canonical_n_a=int(data_a["canonical_n"]),
        nsites_b=int(data_b["nsites"]),
        canonical_n_b=int(data_b["canonical_n"]),
        temperatures=temperatures_a,
    )
    print(f"Created square {args.boundary.upper()} comparison in {args.out_dir}")


if __name__ == "__main__":
    main()
