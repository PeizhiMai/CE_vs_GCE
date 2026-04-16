#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from disk_spectrum import build_disk_hamiltonian, single_particle_spectrum


ROOT = Path("/Users/cosdis/Desktop/projects/CE_GCE")
DEFAULT_OUT_DIR = ROOT / "results" / "non_interacting" / "spinless_disk_compressibility_r4_to_r20_step2"


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
            f"Failed to bracket the target particle number: "
            f"N(lo)={n_lo}, N(hi)={n_hi}, target={target_n}"
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


def grand_canonical_compressibility(
    beta: float,
    energies: np.ndarray,
    mu: float,
    nsites: int,
) -> tuple[float, float]:
    occupations = fermi_dirac(beta, energies, mu)
    avg_n = float(np.sum(occupations))
    variance_n = float(np.sum(occupations * (1.0 - occupations)))
    kappa = beta * variance_n / nsites
    return avg_n, kappa


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
    betas: list[float],
    kappa_gc: list[float],
    kappa_ce: list[float],
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=180)
    ax.plot(betas, kappa_gc, color="#1b9e77", linewidth=2.0, label="GCE")
    ax.plot(betas, kappa_ce, color="#d95f02", linewidth=2.0, linestyle="--", label="CE")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\kappa$")
    ax.set_title(f"Spinless Disk Compressibility, R={int(radius) if float(radius).is_integer() else radius}")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)

    note = f"V={nsites}, N_CE={canonical_n}, GCE tuned to <N>={canonical_n}"
    ax.text(
        0.02,
        0.98,
        note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_metadata(
    path: Path,
    radius: float,
    nsites: int,
    canonical_n: int,
    beta_grid: list[float],
) -> None:
    lines = [
        'model = "spinless non-interacting tight-binding"',
        'geometry = "2D square-lattice disk"',
        'boundary = "open"',
        'compressibility_definition_grand_canonical = "beta/V * (<N^2>-<N>^2)"',
        'compressibility_definition_canonical = "beta/V * (<N^2>-<N>^2)"',
        'canonical_note = "For fixed N, the global compressibility is identically zero."',
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
    target_density = canonical_n / nsites

    radius_dir = out_dir / format_radius_dir(radius)
    rows: list[list[object]] = []
    kappa_gc_values: list[float] = []
    kappa_ce_values: list[float] = []

    for beta in beta_grid:
        temperature = 1.0 / beta
        mu_gc = solve_mu_for_target_n(beta, energies, canonical_n)
        avg_n_gc, kappa_gc = grand_canonical_compressibility(beta, energies, mu_gc, nsites)
        kappa_ce = 0.0
        kappa_gc_values.append(kappa_gc)
        kappa_ce_values.append(kappa_ce)

        rows.append(
            [
                f"{beta:.6f}",
                f"{temperature:.12f}",
                nsites,
                canonical_n,
                f"{target_density:.12f}",
                f"{mu_gc:.12f}",
                f"{avg_n_gc:.12f}",
                f"{kappa_gc:.12e}",
                f"{kappa_ce:.12e}",
            ]
        )

    write_tsv(
        radius_dir / "compressibility_vs_beta.tsv",
        [
            "beta",
            "temperature",
            "nsites",
            "canonical_n",
            "target_density",
            "mu_grand_canonical",
            "avg_n_grand_canonical",
            "kappa_grand_canonical",
            "kappa_canonical",
        ],
        rows,
    )
    write_metadata(radius_dir / "metadata.toml", radius, nsites, canonical_n, beta_grid)
    make_radius_plot(
        radius_dir / "compressibility_vs_beta.png",
        radius=radius,
        nsites=nsites,
        canonical_n=canonical_n,
        betas=beta_grid,
        kappa_gc=kappa_gc_values,
        kappa_ce=kappa_ce_values,
    )

    return nsites, canonical_n, kappa_gc_values[0], kappa_gc_values[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exact compressibility scan for a spinless tight-binding disk, "
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
        nsites, canonical_n, kappa_beta1, kappa_beta100 = solve_radius(radius, beta_grid, out_dir)
        summary_rows.append(
            [
                str(int(radius)) if float(radius).is_integer() else str(radius),
                nsites,
                canonical_n,
                f"{canonical_n / nsites:.12f}",
                f"{kappa_beta1:.12e}",
                f"{kappa_beta100:.12e}",
            ]
        )

    write_tsv(
        out_dir / "radius_summary.tsv",
        [
            "radius",
            "nsites",
            "canonical_n",
            "target_density",
            "kappa_grand_canonical_beta1",
            "kappa_grand_canonical_beta100",
        ],
        summary_rows,
    )

    print("Computed spinless disk compressibility scan")
    print(f"  radii = {', '.join(str(int(radius)) if float(radius).is_integer() else str(radius) for radius in radii)}")
    print(f"  beta grid = {beta_grid[0]} ... {beta_grid[-1]} ({len(beta_grid)} points)")
    print(f"  output directory = {out_dir}")


if __name__ == "__main__":
    main()
