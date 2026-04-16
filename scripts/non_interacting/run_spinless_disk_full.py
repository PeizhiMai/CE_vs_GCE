#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from scan_spinless_disk_energy import (
    DEFAULT_OUT_DIR,
    load_or_compute_disk_spectrum,
    parse_beta_grid,
    solve_radius as solve_energy_radius,
)
from scan_spinless_disk_specific_heat import solve_radius as solve_specific_heat_radius
from scan_spinless_disk_thermo import solve_radius as solve_thermo_radius


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full spinless disk workflow serially: diagonalize once, cache the "
            "single-particle spectrum, then compute energy, specific heat, and thermo."
        )
    )
    parser.add_argument(
        "--radii",
        required=True,
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

    for radius in radii:
        energies, nsites = load_or_compute_disk_spectrum(radius, out_dir)
        print(
            f"Prepared disk spectrum for radius {int(radius) if float(radius).is_integer() else radius}: "
            f"V={nsites}, levels={len(energies)}"
        )
        solve_energy_radius(radius, beta_grid, out_dir)
        print(f"Computed energy scan for radius {int(radius) if float(radius).is_integer() else radius}")
        solve_specific_heat_radius(radius, beta_grid, out_dir)
        print(f"Computed specific-heat scan for radius {int(radius) if float(radius).is_integer() else radius}")
        solve_thermo_radius(radius, beta_grid, out_dir)
        print(f"Computed thermodynamic scan for radius {int(radius) if float(radius).is_integer() else radius}")

    print("Completed full spinless disk workflow")
    print(f"  radii = {', '.join(str(int(radius)) if float(radius).is_integer() else str(radius) for radius in radii)}")
    print(f"  beta grid = {beta_grid[0]} ... {beta_grid[-1]} ({len(beta_grid)} points)")
    print(f"  output directory = {out_dir}")


if __name__ == "__main__":
    main()
