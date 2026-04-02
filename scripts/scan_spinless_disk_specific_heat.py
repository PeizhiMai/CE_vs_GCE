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

from scan_spinless_disk_energy import (
    canonical_energy_per_particle,
    grand_canonical_energy_per_particle,
    parse_beta_grid,
)
from disk_spectrum import build_disk_hamiltonian, single_particle_spectrum


DEFAULT_OUT_DIR = ROOT / "results" / "spinless_disk_comparison_by_radius"


def specific_heat_from_energy_curve(betas: np.ndarray, energies: np.ndarray) -> np.ndarray:
    dedbeta = np.gradient(energies, betas, edge_order=2)
    return -(betas ** 2) * dedbeta


def write_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def make_radius_plot(
    path: Path,
    radius: float,
    nsites: int,
    canonical_n: int,
    temperatures: np.ndarray,
    cv_gc: np.ndarray,
    cv_ce: np.ndarray,
) -> None:
    order = np.argsort(temperatures)
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=180)
    ax.plot(temperatures[order], cv_gc[order], color="#1b9e77", linewidth=2.0, label="GCE")
    ax.plot(temperatures[order], cv_ce[order], color="#d95f02", linewidth=2.0, linestyle="--", label="CE")
    ax.set_xlabel(r"$T = 1/\beta$", fontsize=22)
    ax.set_ylabel("Specific heat per particle", fontsize=22)
    ax.set_title(f"Spinless Disk Specific Heat vs T, R={int(radius) if float(radius).is_integer() else radius}", fontsize=24)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=18)
    ax.tick_params(labelsize=18)
    note = f"V={nsites}, N={canonical_n}, GCE tuned to <N>={canonical_n}"
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


def write_metadata(path: Path, radius: float, nsites: int, canonical_n: int, beta_grid: list[float]) -> None:
    lines = [
        'model = "spinless non-interacting tight-binding"',
        'geometry = "2D square-lattice disk"',
        'boundary = "open"',
        'quantity = "specific heat per particle"',
        'definition = "C/N = -beta^2 d(E/N)/d beta evaluated on the sampled beta grid"',
        'comparison = "canonical fixed N and grand canonical tuned to <N>=N"',
        f"radius = {radius}",
        f"nsites = {nsites}",
        f"canonical_n = {canonical_n}",
        f'beta_grid = "{",".join(str(beta) for beta in beta_grid)}"',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def format_radius_dir(radius: float) -> str:
    if float(radius).is_integer():
        return f"radius_{int(radius)}"
    return f"radius_{str(radius).replace('.', 'p')}"


def solve_radius(radius: float, beta_grid: list[float], out_dir: Path) -> tuple[int, int, float, float]:
    system = build_disk_hamiltonian(radius=radius, hopping=1.0, chemical_potential=0.0)
    energies = single_particle_spectrum(system)
    nsites = len(system.sites)
    canonical_n = (nsites + 1) // 2

    betas = np.asarray(beta_grid, dtype=float)
    temperatures = 1.0 / betas
    energy_gc = np.zeros_like(betas)
    energy_ce = np.zeros_like(betas)
    mu_gc = np.zeros_like(betas)
    avg_n_gc = np.zeros_like(betas)

    for idx, beta in enumerate(betas):
        mu, avg_n, e_gc = grand_canonical_energy_per_particle(beta, energies, canonical_n)
        e_ce = canonical_energy_per_particle(beta, energies, canonical_n)
        mu_gc[idx] = mu
        avg_n_gc[idx] = avg_n
        energy_gc[idx] = e_gc
        energy_ce[idx] = e_ce

    cv_gc = specific_heat_from_energy_curve(betas, energy_gc)
    cv_ce = specific_heat_from_energy_curve(betas, energy_ce)

    radius_dir = out_dir / format_radius_dir(radius)
    rows: list[list[object]] = []
    for idx in range(len(betas)):
        rows.append(
            [
                f"{betas[idx]:.6f}",
                f"{temperatures[idx]:.12f}",
                nsites,
                canonical_n,
                f"{mu_gc[idx]:.12f}",
                f"{avg_n_gc[idx]:.12f}",
                f"{energy_gc[idx]:.12f}",
                f"{energy_ce[idx]:.12f}",
                f"{cv_gc[idx]:.12f}",
                f"{cv_ce[idx]:.12f}",
            ]
        )

    write_tsv(
        radius_dir / "specific_heat_vs_beta.tsv",
        [
            "beta",
            "temperature",
            "nsites",
            "canonical_n",
            "mu_grand_canonical",
            "avg_n_grand_canonical",
            "energy_per_particle_grand_canonical",
            "energy_per_particle_canonical",
            "specific_heat_per_particle_grand_canonical",
            "specific_heat_per_particle_canonical",
        ],
        rows,
    )
    write_metadata(radius_dir / "specific_heat_metadata.toml", radius, nsites, canonical_n, beta_grid)
    make_radius_plot(
        radius_dir / "specific_heat_vs_temperature.png",
        radius=radius,
        nsites=nsites,
        canonical_n=canonical_n,
        temperatures=temperatures,
        cv_gc=cv_gc,
        cv_ce=cv_ce,
    )
    return nsites, canonical_n, float(cv_gc[0]), float(cv_ce[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exact specific-heat scan for a spinless tight-binding disk, "
            "comparing canonical and grand-canonical ensembles."
        )
    )
    parser.add_argument(
        "--radii",
        default="4,6,8,10,12,14,16,18,20",
        help="Comma-separated radii list.",
    )
    parser.add_argument(
        "--beta-grid",
        default="1:100:1",
        help="Either start:stop:step or a comma-separated list.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    radii = [float(item.strip()) for item in args.radii.split(",") if item.strip()]
    beta_grid = parse_beta_grid(args.beta_grid)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[list[object]] = []
    for radius in radii:
        nsites, canonical_n, cv_gc_beta1, cv_ce_beta1 = solve_radius(radius, beta_grid, out_dir)
        summary_rows.append(
            [
                str(int(radius)) if float(radius).is_integer() else str(radius),
                nsites,
                canonical_n,
                f"{cv_gc_beta1:.12f}",
                f"{cv_ce_beta1:.12f}",
            ]
        )

    write_tsv(
        out_dir / "radius_summary.tsv",
        [
            "radius",
            "nsites",
            "canonical_n",
            "specific_heat_per_particle_grand_canonical_beta1",
            "specific_heat_per_particle_canonical_beta1",
        ],
        summary_rows,
    )

    print("Computed spinless disk specific-heat scan")
    print(f"  radii = {', '.join(str(int(radius)) if float(radius).is_integer() else str(radius) for radius in radii)}")
    print(f"  beta grid = {beta_grid[0]} ... {beta_grid[-1]} ({len(beta_grid)} points)")
    print(f"  output directory = {out_dir}")


if __name__ == "__main__":
    main()
