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

from scan_spinless_disk_energy import parse_beta_grid, solve_mu_for_target_n
from disk_spectrum import build_disk_hamiltonian, single_particle_spectrum


DEFAULT_OUT_DIR = ROOT / "results" / "spinless_disk_comparison_by_radius"


def fermi_dirac(beta: float, energies: np.ndarray, mu: float) -> np.ndarray:
    x = np.clip(beta * (energies - mu), -700.0, 700.0)
    return 1.0 / (np.exp(x) + 1.0)


def canonical_sector_data(
    beta: float,
    energies: np.ndarray,
    sectors: list[int],
    mu_ref: float,
) -> dict[int, dict[str, float]]:
    max_sector = max(sectors)
    x = -beta * (energies - mu_ref)
    log_prefactor = float(np.sum(np.logaddexp(0.0, x)))
    probs = 1.0 / (np.exp(np.clip(-x, -700.0, 700.0)) + 1.0)
    qvals = 1.0 - probs

    prob = np.zeros(max_sector + 1, dtype=float)
    energy_sum = np.zeros(max_sector + 1, dtype=float)
    prob[0] = 1.0

    for idx, energy in enumerate(energies):
        p = float(probs[idx])
        q = float(qvals[idx])
        upper = min(idx + 1, max_sector)
        for n in range(upper, 0, -1):
            old_prob_n = prob[n]
            old_energy_n = energy_sum[n]
            old_prob_nm1 = prob[n - 1]
            old_energy_nm1 = energy_sum[n - 1]
            prob[n] = q * old_prob_n + p * old_prob_nm1
            energy_sum[n] = q * old_energy_n + p * (old_energy_nm1 + energy * old_prob_nm1)
        prob[0] *= q
        energy_sum[0] *= q

    out: dict[int, dict[str, float]] = {}
    for n in sectors:
        if prob[n] <= 0.0:
            raise RuntimeError(f"Canonical sector probability underflowed for N={n}")
        log_z = np.log(prob[n]) + log_prefactor - beta * mu_ref * n
        free_energy = -log_z / beta
        total_energy = energy_sum[n] / prob[n]
        out[n] = {
            "log_z": float(log_z),
            "free_energy": float(free_energy),
            "energy": float(total_energy),
        }
    return out


def gce_observables(
    beta: float,
    energies: np.ndarray,
    target_n: int,
    nsites: int,
) -> dict[str, float]:
    mu = solve_mu_for_target_n(beta, energies, target_n)
    occ = fermi_dirac(beta, energies, mu)
    avg_n = float(np.sum(occ))
    total_energy = float(np.dot(energies, occ))
    variance_n = float(np.sum(occ * (1.0 - occ)))
    omega = -float(np.sum(np.logaddexp(0.0, -beta * (energies - mu)))) / beta
    temperature = 1.0 / beta
    entropy = (total_energy - omega - mu * avg_n) / temperature
    kappa = beta * variance_n / nsites
    return {
        "mu": float(mu),
        "avg_n": avg_n,
        "energy_per_particle": total_energy / avg_n,
        "entropy_per_particle": entropy / avg_n,
        "kappa": float(kappa),
    }


def write_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def make_plot(
    path: Path,
    radius: float,
    nsites: int,
    canonical_n: int,
    temperatures: np.ndarray,
    gce_values: np.ndarray,
    ce_values: np.ndarray,
    ylabel: str,
    title_prefix: str,
) -> None:
    order = np.argsort(temperatures)
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=180)
    ax.plot(temperatures[order], gce_values[order], color="#1b9e77", linewidth=2.0, label="GCE")
    ax.plot(temperatures[order], ce_values[order], color="#d95f02", linewidth=2.0, linestyle="--", label="CE")
    ax.set_xlabel(r"$T = 1/\beta$", fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.set_title(f"{title_prefix}, R={int(radius) if float(radius).is_integer() else radius}", fontsize=24)
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
        'compressibility_definition_ce = "kappa = 1 / (V * (F(N+1)-2F(N)+F(N-1)))"',
        'compressibility_definition_gce = "kappa = beta/V * (<N^2>-<N>^2) = dn/dmu"',
        'entropy_definition_ce = "S/N = (E-F)/(T N)"',
        'entropy_definition_gce = "S/N = (E-Omega-mu N)/(T N)"',
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


def solve_radius(radius: float, beta_grid: list[float], out_dir: Path) -> tuple[int, int, float, float, float, float]:
    system = build_disk_hamiltonian(radius=radius, hopping=1.0, chemical_potential=0.0)
    energies = single_particle_spectrum(system)
    nsites = len(system.sites)
    canonical_n = (nsites + 1) // 2
    temperatures = 1.0 / np.asarray(beta_grid, dtype=float)

    kappa_gce = np.zeros(len(beta_grid), dtype=float)
    kappa_ce = np.zeros(len(beta_grid), dtype=float)
    entropy_gce = np.zeros(len(beta_grid), dtype=float)
    entropy_ce = np.zeros(len(beta_grid), dtype=float)
    mu_gce = np.zeros(len(beta_grid), dtype=float)
    avg_n_gce = np.zeros(len(beta_grid), dtype=float)

    rows: list[list[object]] = []
    sectors = [canonical_n - 1, canonical_n, canonical_n + 1]

    for idx, beta in enumerate(beta_grid):
        temperature = 1.0 / beta
        mu_ref = 0.5 * (energies[canonical_n - 1] + energies[canonical_n]) if canonical_n < nsites else energies[-1] + 1.0
        ce_data = canonical_sector_data(beta, energies, sectors, mu_ref)
        free_minus = ce_data[canonical_n - 1]["free_energy"]
        free_zero = ce_data[canonical_n]["free_energy"]
        free_plus = ce_data[canonical_n + 1]["free_energy"]
        energy_zero = ce_data[canonical_n]["energy"]

        d2f = free_plus - 2.0 * free_zero + free_minus
        kappa_ce[idx] = 1.0 / (nsites * d2f)
        entropy_ce[idx] = (energy_zero - free_zero) / (temperature * canonical_n)

        gce = gce_observables(beta, energies, canonical_n, nsites)
        kappa_gce[idx] = gce["kappa"]
        entropy_gce[idx] = gce["entropy_per_particle"]
        mu_gce[idx] = gce["mu"]
        avg_n_gce[idx] = gce["avg_n"]

        rows.append(
            [
                f"{beta:.6f}",
                f"{temperature:.12f}",
                nsites,
                canonical_n,
                f"{mu_gce[idx]:.12f}",
                f"{avg_n_gce[idx]:.12f}",
                f"{kappa_gce[idx]:.12f}",
                f"{kappa_ce[idx]:.12f}",
                f"{entropy_gce[idx]:.12f}",
                f"{entropy_ce[idx]:.12f}",
            ]
        )

    radius_dir = out_dir / format_radius_dir(radius)
    write_tsv(
        radius_dir / "thermo_vs_beta.tsv",
        [
            "beta",
            "temperature",
            "nsites",
            "canonical_n",
            "mu_grand_canonical",
            "avg_n_grand_canonical",
            "kappa_grand_canonical",
            "kappa_canonical",
            "entropy_per_particle_grand_canonical",
            "entropy_per_particle_canonical",
        ],
        rows,
    )
    write_tsv(
        radius_dir / "compressibility_vs_temperature.tsv",
        [
            "temperature",
            "kappa_grand_canonical",
            "kappa_canonical",
        ],
        [
            [
                f"{temperatures[idx]:.12f}",
                f"{kappa_gce[idx]:.12f}",
                f"{kappa_ce[idx]:.12f}",
            ]
            for idx in range(len(beta_grid))
        ],
    )
    write_tsv(
        radius_dir / "entropy_vs_temperature.tsv",
        [
            "temperature",
            "entropy_per_particle_grand_canonical",
            "entropy_per_particle_canonical",
        ],
        [
            [
                f"{temperatures[idx]:.12f}",
                f"{entropy_gce[idx]:.12f}",
                f"{entropy_ce[idx]:.12f}",
            ]
            for idx in range(len(beta_grid))
        ],
    )
    write_metadata(radius_dir / "thermo_metadata.toml", radius, nsites, canonical_n, beta_grid)
    make_plot(
        radius_dir / "compressibility_vs_temperature.png",
        radius=radius,
        nsites=nsites,
        canonical_n=canonical_n,
        temperatures=temperatures,
        gce_values=kappa_gce,
        ce_values=kappa_ce,
        ylabel="Compressibility",
        title_prefix="Spinless Disk Compressibility vs T",
    )
    make_plot(
        radius_dir / "entropy_vs_temperature.png",
        radius=radius,
        nsites=nsites,
        canonical_n=canonical_n,
        temperatures=temperatures,
        gce_values=entropy_gce,
        ce_values=entropy_ce,
        ylabel="Entropy per particle",
        title_prefix="Spinless Disk Entropy vs T",
    )

    return (
        nsites,
        canonical_n,
        float(kappa_gce[0]),
        float(kappa_ce[0]),
        float(entropy_gce[0]),
        float(entropy_ce[0]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exact thermodynamic compressibility and entropy scan for a spinless "
            "tight-binding disk, comparing canonical and grand-canonical ensembles."
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
        nsites, canonical_n, kappa_gc_beta1, kappa_ce_beta1, s_gc_beta1, s_ce_beta1 = solve_radius(
            radius, beta_grid, out_dir
        )
        summary_rows.append(
            [
                str(int(radius)) if float(radius).is_integer() else str(radius),
                nsites,
                canonical_n,
                f"{kappa_gc_beta1:.12f}",
                f"{kappa_ce_beta1:.12f}",
                f"{s_gc_beta1:.12f}",
                f"{s_ce_beta1:.12f}",
            ]
        )

    write_tsv(
        out_dir / "radius_summary.tsv",
        [
            "radius",
            "nsites",
            "canonical_n",
            "kappa_grand_canonical_beta1",
            "kappa_canonical_beta1",
            "entropy_per_particle_grand_canonical_beta1",
            "entropy_per_particle_canonical_beta1",
        ],
        summary_rows,
    )

    print("Computed spinless disk thermodynamic scan")
    print(f"  radii = {', '.join(str(int(radius)) if float(radius).is_integer() else str(radius) for radius in radii)}")
    print(f"  beta grid = {beta_grid[0]} ... {beta_grid[-1]} ({len(beta_grid)} points)")
    print(f"  output directory = {out_dir}")


if __name__ == "__main__":
    main()
