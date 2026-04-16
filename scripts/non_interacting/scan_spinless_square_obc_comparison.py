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
from scan_spinless_disk_specific_heat import specific_heat_from_energy_curve
from scan_spinless_disk_thermo import canonical_sector_data, gce_observables


DEFAULT_OUT_DIR = ROOT / "results" / "non_interacting" / "spinless_square_obc_comparison_by_L"


def square_obc_energies(L: int, t_hop: float = 1.0) -> np.ndarray:
    modes = np.arange(1, L + 1, dtype=float)
    momenta = np.pi * modes / (L + 1)
    energies = [
        -2.0 * t_hop * (np.cos(kx) + np.cos(ky))
        for kx in momenta
        for ky in momenta
    ]
    return np.sort(np.asarray(energies, dtype=float))


def write_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def output_subdir_name(L: int, canonical_n: int, default_n: int) -> str:
    if canonical_n == default_n:
        return f"L_{L}"
    return f"L_{L}_N_{canonical_n}"


def summary_filename(canonical_n: int | None) -> str:
    if canonical_n is None:
        return "L_summary.tsv"
    return f"L_summary_N_{canonical_n}.tsv"


def make_plot(
    path: Path,
    L: int,
    nsites: int,
    canonical_n: int,
    temperatures: np.ndarray,
    gce_values: np.ndarray,
    ce_values: np.ndarray,
    ylabel: str,
    title_prefix: str,
) -> None:
    order = np.argsort(temperatures)
    note = f"L={L}, V={nsites}, N={canonical_n}, GCE tuned to <N>={canonical_n}"
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=180)
    ax.plot(
        temperatures[order],
        gce_values[order],
        color="#8c510a",
        linewidth=2.4,
        linestyle="-",
        label="GCE",
    )
    ax.plot(
        temperatures[order],
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
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_metadata(
    path: Path,
    L: int,
    nsites: int,
    canonical_n: int,
    beta_grid: list[float],
    filling: str,
) -> None:
    lines = [
        'model = "spinless non-interacting tight-binding"',
        'geometry = "LxL square lattice"',
        'boundary = "open"',
        f'filling = "{filling}"',
        'compressibility_definition_ce = "kappa = 1 / (V * (F(N+1)-2F(N)+F(N-1)))"',
        'compressibility_definition_gce = "kappa = beta/V * (<N^2>-<N>^2) = dn/dmu"',
        'entropy_definition_ce = "S/N = (E-F)/(T N)"',
        'entropy_definition_gce = "S/N = (E-Omega-mu N)/(T N)"',
        f"L = {L}",
        f"nsites = {nsites}",
        f"canonical_n = {canonical_n}",
        f'beta_grid = "{",".join(str(beta) for beta in beta_grid)}"',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def solve_L(
    L: int,
    beta_grid: list[float],
    out_dir: Path,
    t_hop: float = 1.0,
    canonical_n: int | None = None,
) -> dict[str, float]:
    energies = square_obc_energies(L, t_hop=t_hop)
    nsites = L * L
    default_n = nsites // 2
    canonical_n = default_n if canonical_n is None else canonical_n
    if not (1 <= canonical_n < nsites):
        raise SystemExit(f"canonical_n must satisfy 1 <= N < V; got N={canonical_n}, V={nsites}")
    temperatures = 1.0 / np.asarray(beta_grid, dtype=float)

    energy_gce = np.zeros(len(beta_grid), dtype=float)
    energy_ce = np.zeros(len(beta_grid), dtype=float)
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

        mu_gc, avg_n, e_gc = grand_canonical_energy_per_particle(beta, energies, canonical_n)
        e_ce = canonical_energy_per_particle(beta, energies, canonical_n)
        energy_gce[idx] = e_gc
        energy_ce[idx] = e_ce
        mu_gce[idx] = mu_gc
        avg_n_gce[idx] = avg_n

        mu_ref = 0.5 * (energies[canonical_n - 1] + energies[canonical_n]) if canonical_n < nsites else energies[-1] + 1.0
        ce_data = canonical_sector_data(beta, energies, sectors, mu_ref=mu_ref)
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

        rows.append(
            [
                f"{beta:.6f}",
                f"{temperature:.12f}",
                nsites,
                canonical_n,
                f"{mu_gce[idx]:.12f}",
                f"{avg_n_gce[idx]:.12f}",
                f"{energy_gce[idx]:.12f}",
                f"{energy_ce[idx]:.12f}",
                f"{kappa_gce[idx]:.12f}",
                f"{kappa_ce[idx]:.12f}",
                f"{entropy_gce[idx]:.12f}",
                f"{entropy_ce[idx]:.12f}",
            ]
        )

    specific_heat_gce = specific_heat_from_energy_curve(np.asarray(beta_grid, dtype=float), energy_gce)
    specific_heat_ce = specific_heat_from_energy_curve(np.asarray(beta_grid, dtype=float), energy_ce)

    l_dir = out_dir / output_subdir_name(L, canonical_n, default_n)
    write_tsv(
        l_dir / "square_obc_thermo_vs_beta.tsv",
        [
            "beta",
            "temperature",
            "nsites",
            "canonical_n",
            "mu_grand_canonical",
            "avg_n_grand_canonical",
            "energy_per_particle_grand_canonical",
            "energy_per_particle_canonical",
            "kappa_grand_canonical",
            "kappa_canonical",
            "entropy_per_particle_grand_canonical",
            "entropy_per_particle_canonical",
        ],
        rows,
    )
    write_tsv(
        l_dir / "square_obc_specific_heat_vs_temperature.tsv",
        [
            "temperature",
            "specific_heat_per_particle_grand_canonical",
            "specific_heat_per_particle_canonical",
        ],
        [
            [f"{temperatures[idx]:.12f}", f"{specific_heat_gce[idx]:.12f}", f"{specific_heat_ce[idx]:.12f}"]
            for idx in range(len(beta_grid))
        ],
    )
    filling = "half-filling" if canonical_n == default_n else f"fixed N={canonical_n}"
    write_metadata(l_dir / "metadata.toml", L, nsites, canonical_n, beta_grid, filling)
    make_plot(
        l_dir / "energy_vs_temperature.png",
        L=L,
        nsites=nsites,
        canonical_n=canonical_n,
        temperatures=temperatures,
        gce_values=energy_gce,
        ce_values=energy_ce,
        ylabel="Energy per particle",
        title_prefix="Spinless Square OBC Energy vs T",
    )
    make_plot(
        l_dir / "specific_heat_vs_temperature.png",
        L=L,
        nsites=nsites,
        canonical_n=canonical_n,
        temperatures=temperatures,
        gce_values=specific_heat_gce,
        ce_values=specific_heat_ce,
        ylabel="Specific heat per particle",
        title_prefix="Spinless Square OBC Specific Heat vs T",
    )
    make_plot(
        l_dir / "compressibility_vs_temperature.png",
        L=L,
        nsites=nsites,
        canonical_n=canonical_n,
        temperatures=temperatures,
        gce_values=kappa_gce,
        ce_values=kappa_ce,
        ylabel="Compressibility",
        title_prefix="Spinless Square OBC Compressibility vs T",
    )
    make_plot(
        l_dir / "entropy_vs_temperature.png",
        L=L,
        nsites=nsites,
        canonical_n=canonical_n,
        temperatures=temperatures,
        gce_values=entropy_gce,
        ce_values=entropy_ce,
        ylabel="Entropy per particle",
        title_prefix="Spinless Square OBC Entropy vs T",
    )

    return {
        "L": float(L),
        "nsites": float(nsites),
        "canonical_n": float(canonical_n),
        "energy_gce_beta1": float(energy_gce[0]),
        "energy_ce_beta1": float(energy_ce[0]),
        "cv_gce_beta1": float(specific_heat_gce[0]),
        "cv_ce_beta1": float(specific_heat_ce[0]),
        "kappa_gce_beta1": float(kappa_gce[0]),
        "kappa_ce_beta1": float(kappa_ce[0]),
        "entropy_gce_beta1": float(entropy_gce[0]),
        "entropy_ce_beta1": float(entropy_ce[0]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exact CE vs GCE comparison for a spinless open-boundary LxL square lattice "
            "at half-filling."
        )
    )
    parser.add_argument(
        "--Ls",
        default="4,6,8,12,16",
        help="Comma-separated linear system sizes.",
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
    parser.add_argument(
        "--t",
        type=float,
        default=1.0,
        help="Nearest-neighbor hopping amplitude.",
    )
    parser.add_argument(
        "--canonical-n",
        type=int,
        help="Optional fixed canonical particle number N. Defaults to floor(V/2).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Ls = [int(item.strip()) for item in args.Ls.split(",") if item.strip()]
    beta_grid = parse_beta_grid(args.beta_grid)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[list[object]] = []
    for L in Ls:
        result = solve_L(L, beta_grid, out_dir, t_hop=args.t, canonical_n=args.canonical_n)
        summary_rows.append(
            [
                L,
                int(result["nsites"]),
                int(result["canonical_n"]),
                f"{result['energy_gce_beta1']:.12f}",
                f"{result['energy_ce_beta1']:.12f}",
                f"{result['cv_gce_beta1']:.12f}",
                f"{result['cv_ce_beta1']:.12f}",
                f"{result['kappa_gce_beta1']:.12f}",
                f"{result['kappa_ce_beta1']:.12f}",
                f"{result['entropy_gce_beta1']:.12f}",
                f"{result['entropy_ce_beta1']:.12f}",
            ]
        )

    write_tsv(
        out_dir / summary_filename(args.canonical_n),
        [
            "L",
            "nsites",
            "canonical_n",
            "energy_per_particle_grand_canonical_beta1",
            "energy_per_particle_canonical_beta1",
            "specific_heat_per_particle_grand_canonical_beta1",
            "specific_heat_per_particle_canonical_beta1",
            "kappa_grand_canonical_beta1",
            "kappa_canonical_beta1",
            "entropy_per_particle_grand_canonical_beta1",
            "entropy_per_particle_canonical_beta1",
        ],
        summary_rows,
    )

    print("Computed spinless square-lattice OBC comparison")
    print(f"  L values = {', '.join(str(L) for L in Ls)}")
    if args.canonical_n is not None:
        print(f"  canonical N = {args.canonical_n}")
    print(f"  beta grid = {beta_grid[0]} ... {beta_grid[-1]} ({len(beta_grid)} points)")
    print(f"  output directory = {out_dir}")


if __name__ == "__main__":
    main()
