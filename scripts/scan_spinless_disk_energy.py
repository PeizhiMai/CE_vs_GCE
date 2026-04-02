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

from disk_spectrum import build_disk_hamiltonian, single_particle_spectrum


DEFAULT_OUT_DIR = ROOT / "results" / "spinless_disk_comparison_by_radius"


def parse_beta_grid(text: str) -> list[float]:
    if ":" in text:
        parts = [part.strip() for part in text.split(":")]
        if len(parts) != 3:
            raise SystemExit("Beta grid with ':' must be start:stop:step.")
        start, stop, step = (float(part) for part in parts)
        if step <= 0:
            raise SystemExit("Beta-grid step must be positive.")
        values: list[float] = []
        current = start
        while current <= stop + 1e-12:
            values.append(round(current, 10))
            current += step
        return values

    return [float(item.strip()) for item in text.split(",") if item.strip()]


def fermi_dirac(beta: float, energies: np.ndarray, mu: float) -> np.ndarray:
    x = beta * (energies - mu)
    clipped = np.clip(x, -700.0, 700.0)
    return 1.0 / (np.exp(clipped) + 1.0)


def grand_canonical_particle_number(beta: float, energies: np.ndarray, mu: float) -> float:
    return float(np.sum(fermi_dirac(beta, energies, mu)))


def solve_mu_for_target_n(beta: float, energies: np.ndarray, target_n: float) -> float:
    emin = float(np.min(energies))
    emax = float(np.max(energies))
    margin = max(10.0, 40.0 / beta)
    lo = emin - margin
    hi = emax + margin

    n_lo = grand_canonical_particle_number(beta, energies, lo)
    n_hi = grand_canonical_particle_number(beta, energies, hi)
    if not (n_lo <= target_n <= n_hi):
        raise RuntimeError(
            f"Failed to bracket target N: N(lo)={n_lo}, N(hi)={n_hi}, target={target_n}"
        )

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        n_mid = grand_canonical_particle_number(beta, energies, mid)
        if abs(n_mid - target_n) < 1e-13:
            return mid
        if n_mid < target_n:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def grand_canonical_energy_per_particle(beta: float, energies: np.ndarray, target_n: int) -> tuple[float, float, float]:
    mu = solve_mu_for_target_n(beta, energies, target_n)
    occ = fermi_dirac(beta, energies, mu)
    avg_n = float(np.sum(occ))
    energy = float(np.dot(energies, occ))
    return mu, avg_n, energy / avg_n


def canonical_energy_per_particle(beta: float, energies: np.ndarray, target_n: int) -> float:
    nlevels = len(energies)
    if target_n == 0:
        return 0.0

    μref = 0.5 * (energies[target_n - 1] + energies[target_n]) if target_n < nlevels else energies[-1] + 1.0
    probs = fermi_dirac(beta, energies, μref)

    prob = np.zeros((nlevels + 1, target_n + 1), dtype=float)
    energy_sum = np.zeros((nlevels + 1, target_n + 1), dtype=float)
    prob[0, 0] = 1.0

    for i in range(1, nlevels + 1):
        p = probs[i - 1]
        q = 1.0 - p
        ε = energies[i - 1]
        prob[i, 0] = q * prob[i - 1, 0]
        energy_sum[i, 0] = q * energy_sum[i - 1, 0]
        max_n = min(i, target_n)
        for n in range(1, max_n + 1):
            prob[i, n] = q * prob[i - 1, n] + p * prob[i - 1, n - 1]
            energy_sum[i, n] = (
                q * energy_sum[i - 1, n]
                + p * (energy_sum[i - 1, n - 1] + ε * prob[i - 1, n - 1])
            )

    prob_total = prob[nlevels, target_n]
    if prob_total <= 0.0:
        raise RuntimeError("Canonical conditioning probability underflowed to zero.")

    energy = energy_sum[nlevels, target_n] / prob_total
    return float(energy / target_n)


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
    energy_gc: list[float],
    energy_ce: list[float],
) -> None:
    order = np.argsort(temperatures)
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=180)
    ax.plot(temperatures[order], np.asarray(energy_gc)[order], color="#1b9e77", linewidth=2.0, label="GCE")
    ax.plot(temperatures[order], np.asarray(energy_ce)[order], color="#d95f02", linewidth=2.0, linestyle="--", label="CE")
    ax.set_xlabel(r"$T = 1/\beta$", fontsize=22)
    ax.set_ylabel("Energy per particle", fontsize=22)
    ax.set_title(f"Spinless Disk Energy vs T, R={int(radius) if float(radius).is_integer() else radius}", fontsize=24)
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

    radius_dir = out_dir / format_radius_dir(radius)
    rows: list[list[object]] = []
    gc_curve: list[float] = []
    ce_curve: list[float] = []

    for beta in beta_grid:
        temperature = 1.0 / beta
        mu_gc, avg_n_gc, energy_per_particle_gc = grand_canonical_energy_per_particle(beta, energies, canonical_n)
        energy_per_particle_ce = canonical_energy_per_particle(beta, energies, canonical_n)
        gc_curve.append(energy_per_particle_gc)
        ce_curve.append(energy_per_particle_ce)
        rows.append(
            [
                f"{beta:.6f}",
                f"{temperature:.12f}",
                nsites,
                canonical_n,
                f"{mu_gc:.12f}",
                f"{avg_n_gc:.12f}",
                f"{energy_per_particle_gc:.12f}",
                f"{energy_per_particle_ce:.12f}",
            ]
        )

    write_tsv(
        radius_dir / "energy_vs_beta.tsv",
        [
            "beta",
            "temperature",
            "nsites",
            "canonical_n",
            "mu_grand_canonical",
            "avg_n_grand_canonical",
            "energy_per_particle_grand_canonical",
            "energy_per_particle_canonical",
        ],
        rows,
    )
    write_metadata(radius_dir / "energy_metadata.toml", radius, nsites, canonical_n, beta_grid)
    make_radius_plot(
        radius_dir / "energy_vs_temperature.png",
        radius=radius,
        nsites=nsites,
        canonical_n=canonical_n,
        temperatures=1.0 / np.asarray(beta_grid, dtype=float),
        energy_gc=gc_curve,
        energy_ce=ce_curve,
    )
    return nsites, canonical_n, gc_curve[0], ce_curve[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exact energy-per-particle scan for a spinless tight-binding disk, "
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
        nsites, canonical_n, energy_gc_beta1, energy_ce_beta1 = solve_radius(radius, beta_grid, out_dir)
        summary_rows.append(
            [
                str(int(radius)) if float(radius).is_integer() else str(radius),
                nsites,
                canonical_n,
                f"{energy_gc_beta1:.12f}",
                f"{energy_ce_beta1:.12f}",
            ]
        )

    write_tsv(
        out_dir / "radius_summary.tsv",
        [
            "radius",
            "nsites",
            "canonical_n",
            "energy_per_particle_grand_canonical_beta1",
            "energy_per_particle_canonical_beta1",
        ],
        summary_rows,
    )

    print("Computed spinless disk energy scan")
    print(f"  radii = {', '.join(str(int(radius)) if float(radius).is_integer() else str(radius) for radius in radii)}")
    print(f"  beta grid = {beta_grid[0]} ... {beta_grid[-1]} ({len(beta_grid)} points)")
    print(f"  output directory = {out_dir}")


if __name__ == "__main__":
    main()
