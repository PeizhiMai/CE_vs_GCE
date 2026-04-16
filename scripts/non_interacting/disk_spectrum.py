#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path("/Users/cosdis/Desktop/projects/CE_GCE")
DEFAULT_OUT_DIR = ROOT / "results" / "non_interacting" / "disk_spectrum_python"


@dataclass(frozen=True)
class DiskSystem:
    radius: float
    sites: list[tuple[int, int]]
    hamiltonian: np.ndarray


def format_tag(value: float) -> str:
    return str(round(float(value), 6)).replace("-", "m").replace(".", "p")


def parse_radii(args: argparse.Namespace) -> list[float]:
    if args.radii:
        radii = [float(item.strip()) for item in args.radii.split(",") if item.strip()]
    elif args.radius is not None:
        radii = [float(args.radius)]
    else:
        raise SystemExit("Provide either --radius or --radii.")

    if any(radius <= 0 for radius in radii):
        raise SystemExit("All radii must be positive.")

    return radii


def build_disk_sites(radius: float) -> list[tuple[int, int]]:
    bound = int(np.floor(radius))
    radius_sq = radius * radius
    sites: list[tuple[int, int]] = []

    for y in range(-bound, bound + 1):
        for x in range(-bound, bound + 1):
            if x * x + y * y <= radius_sq + 1e-12:
                sites.append((x, y))

    sites.sort(key=lambda site: (site[1], site[0]))
    return sites


def build_disk_hamiltonian(
    radius: float,
    hopping: float = 1.0,
    chemical_potential: float = 0.0,
) -> DiskSystem:
    sites = build_disk_sites(radius)
    nsites = len(sites)
    if nsites == 0:
        raise ValueError(f"Radius {radius} contains no lattice sites.")

    site_to_index = {site: idx for idx, site in enumerate(sites)}
    hamiltonian = np.zeros((nsites, nsites), dtype=float)

    for idx, (x, y) in enumerate(sites):
        hamiltonian[idx, idx] = -chemical_potential
        for neighbor in ((x + 1, y), (x, y + 1)):
            jdx = site_to_index.get(neighbor)
            if jdx is not None:
                hamiltonian[idx, jdx] = -hopping
                hamiltonian[jdx, idx] = -hopping

    return DiskSystem(radius=radius, sites=sites, hamiltonian=hamiltonian)


def single_particle_spectrum(system: DiskSystem) -> np.ndarray:
    return np.linalg.eigvalsh(system.hamiltonian)


def write_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def write_two_row_radius_table(path: Path, radii: list[float], site_counts: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        formatted_radii: list[str] = []
        for radius in radii:
            if float(radius).is_integer():
                formatted_radii.append(str(int(radius)))
            else:
                formatted_radii.append(f"{radius:.6f}")
        writer.writerow(["radius", *formatted_radii])
        writer.writerow(["n_sites", *[str(count) for count in site_counts]])


def write_metadata(
    path: Path,
    radius: float,
    hopping: float,
    chemical_potential: float,
    nsites: int,
    eigenvalues: np.ndarray,
) -> None:
    lines = [
        'model = "non-interacting tight-binding"',
        'geometry = "2D square-lattice disk"',
        'boundary = "open"',
        'inclusion_rule = "x^2 + y^2 <= R^2 with integer lattice sites"',
        f"radius = {radius}",
        f"hopping = {hopping}",
        f"mu = {chemical_potential}",
        f"nsites = {nsites}",
        f"lowest_eigenvalue = {float(eigenvalues[0])}",
        f"highest_eigenvalue = {float(eigenvalues[-1])}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def solve_one_radius(
    radius: float,
    hopping: float,
    chemical_potential: float,
    out_dir: Path,
) -> dict[str, float]:
    system = build_disk_hamiltonian(
        radius=radius,
        hopping=hopping,
        chemical_potential=chemical_potential,
    )
    eigenvalues = single_particle_spectrum(system)
    radius_dir = out_dir / f"radius_{format_tag(radius)}"

    write_metadata(
        radius_dir / "metadata.toml",
        radius=radius,
        hopping=hopping,
        chemical_potential=chemical_potential,
        nsites=len(system.sites),
        eigenvalues=eigenvalues,
    )
    write_tsv(
        radius_dir / "sites.tsv",
        ["site_index", "x", "y"],
        [[idx + 1, x, y] for idx, (x, y) in enumerate(system.sites)],
    )
    write_tsv(
        radius_dir / "eigenvalues.tsv",
        ["level_index", "eigenvalue"],
        [[idx + 1, f"{value:.12f}"] for idx, value in enumerate(eigenvalues)],
    )

    return {
        "radius": radius,
        "nsites": float(len(system.sites)),
        "emin": float(eigenvalues[0]),
        "emax": float(eigenvalues[-1]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Python solver for the non-interacting single-particle "
            "tight-binding spectrum on a 2D disk-shaped lattice."
        )
    )
    parser.add_argument("--radius", type=float, help="Solve one radius.")
    parser.add_argument(
        "--radii",
        help="Comma-separated list of radii, for example 3,4,5,6.",
    )
    parser.add_argument(
        "--t",
        type=float,
        default=1.0,
        help="Nearest-neighbor hopping amplitude.",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.0,
        help="Diagonal shift added as -mu on each site.",
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
    radii = parse_radii(args)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[list[object]] = []
    site_counts: list[int] = []
    for radius in radii:
        result = solve_one_radius(
            radius=radius,
            hopping=args.t,
            chemical_potential=args.mu,
            out_dir=out_dir,
        )
        site_counts.append(int(result["nsites"]))
        summary_rows.append(
            [
                f"{result['radius']:.6f}",
                int(result["nsites"]),
                f"{result['emin']:.12f}",
                f"{result['emax']:.12f}",
            ]
        )

    write_tsv(
        out_dir / "radius_summary.tsv",
        ["radius", "nsites", "lowest_eigenvalue", "highest_eigenvalue"],
        summary_rows,
    )
    write_two_row_radius_table(out_dir / "radius_nsites_table.tsv", radii, site_counts)

    print("Computed disk tight-binding spectra")
    print(f"  radii = {', '.join(str(radius) for radius in radii)}")
    print(f"  hopping t = {args.t}")
    print(f"  mu = {args.mu}")
    print(f"  output directory = {out_dir}")


if __name__ == "__main__":
    main()
