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


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def column_as_float(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def load_thermo(path: Path) -> dict[str, np.ndarray]:
    rows = read_tsv(path)
    return {
        "temperature": column_as_float(rows, "temperature"),
        "kappa_gce": column_as_float(rows, "kappa_grand_canonical"),
        "kappa_ce": column_as_float(rows, "kappa_canonical"),
    }


def interp_kappa_from_temperature(
    sample_temperatures: np.ndarray,
    temperatures: np.ndarray,
    kappas: np.ndarray,
) -> np.ndarray:
    order = np.argsort(temperatures)
    log_t = np.log(temperatures[order])
    return np.interp(np.log(sample_temperatures), log_t, kappas[order])


def invert_temperature_from_kappa(
    kappa_values: np.ndarray,
    temperatures: np.ndarray,
    kappas: np.ndarray,
) -> np.ndarray:
    order = np.argsort(kappas)
    kappa_sorted = kappas[order]
    log_t_sorted = np.log(temperatures[order])
    unique_kappa, unique_indices = np.unique(kappa_sorted, return_index=True)
    unique_log_t = log_t_sorted[unique_indices]

    out = np.full(kappa_values.shape, np.nan, dtype=float)
    valid = (kappa_values >= unique_kappa[0]) & (kappa_values <= unique_kappa[-1])
    if np.any(valid):
        out[valid] = np.exp(np.interp(kappa_values[valid], unique_kappa, unique_log_t))
    return out


def match_temperatures(
    disk_temperatures: np.ndarray,
    disk_kappas: np.ndarray,
    square_temperatures: np.ndarray,
    square_kappas: np.ndarray,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    tmin = max(float(np.min(disk_temperatures)), 1e-12)
    tmax = float(np.max(disk_temperatures))
    sample_t = np.geomspace(tmin, tmax, num_points)
    sample_kappa = interp_kappa_from_temperature(sample_t, disk_temperatures, disk_kappas)
    mapped_t = invert_temperature_from_kappa(sample_kappa, square_temperatures, square_kappas)
    valid = np.isfinite(mapped_t)
    return sample_t[valid], mapped_t[valid]


def plot_panel(
    path: Path,
    x_obc: np.ndarray,
    y_obc: np.ndarray,
    x_pbc: np.ndarray,
    y_pbc: np.ndarray,
    disk_radius: int,
    square_L: int,
    disk_ens: str,
    square_ens: str,
    log_scale: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=180)
    ax.plot(x_obc, y_obc, color="#8c510a", linewidth=2.2, label="Square OBC")
    ax.plot(x_pbc, y_pbc, color="#7570b3", linewidth=2.2, label="Square PBC")

    finite_x = np.concatenate([x_obc, x_pbc]) if len(x_obc) and len(x_pbc) else (x_obc if len(x_obc) else x_pbc)
    finite_y = np.concatenate([y_obc, y_pbc]) if len(y_obc) and len(y_pbc) else (y_obc if len(y_obc) else y_pbc)
    if len(finite_x) and len(finite_y):
        line_min = max(float(min(np.min(finite_x), np.min(finite_y))), 0.0)
        line_max = float(max(np.max(finite_x), np.max(finite_y)))
        if line_max > line_min:
            if log_scale:
                line_min = max(line_min, 1e-6)
                line = np.geomspace(line_min, line_max, 200)
            else:
                line = np.linspace(line_min, line_max, 200)
            ax.plot(line, line, color="#666666", linewidth=1.0, linestyle="--", label="y = x")

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel(
        rf"$T_{{\mathrm{{disk}},\,R={disk_radius},\,{disk_ens.upper()}}}$",
        fontsize=18,
    )
    ax.set_ylabel(
        rf"$T_{{\mathrm{{square}},\,L={square_L},\,{square_ens.upper()}}}$",
        fontsize=18,
    )
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=16)
    ax.legend(frameon=False, fontsize=15)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map square L=8 temperatures to disk R=16 temperatures at equal compressibility."
    )
    parser.add_argument("--disk-radius", type=int, default=16)
    parser.add_argument("--square-L", type=int, default=8)
    parser.add_argument("--num-points", type=int, default=1500)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "non_interacting" / "r16_vs_l8pbc_vs_l8obc_kappa_temperature_map",
    )
    args = parser.parse_args()

    disk = load_thermo(
        ROOT / "results" / "non_interacting" / "spinless_disk_comparison_by_radius" / f"radius_{args.disk_radius}" / "thermo_vs_beta.tsv"
    )
    pbc = load_thermo(
        ROOT / "results" / "non_interacting" / "spinless_square_pbc_comparison_by_L" / f"L_{args.square_L}" / "square_pbc_thermo_vs_beta.tsv"
    )
    obc = load_thermo(
        ROOT / "results" / "non_interacting" / "spinless_square_obc_comparison_by_L" / f"L_{args.square_L}" / "square_obc_thermo_vs_beta.tsv"
    )

    panels = [
        ("gce", "gce", "Disk GCE to Square GCE"),
        ("gce", "ce", "Disk GCE to Square CE"),
        ("ce", "gce", "Disk CE to Square GCE"),
        ("ce", "ce", "Disk CE to Square CE"),
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    matched_rows: list[list[object]] = []

    for disk_ens, square_ens, panel_title in panels:
        disk_kappa = disk[f"kappa_{disk_ens}"]
        pbc_kappa = pbc[f"kappa_{square_ens}"]
        obc_kappa = obc[f"kappa_{square_ens}"]

        x_pbc, y_pbc = match_temperatures(
            disk["temperature"], disk_kappa, pbc["temperature"], pbc_kappa, args.num_points
        )
        x_obc, y_obc = match_temperatures(
            disk["temperature"], disk_kappa, obc["temperature"], obc_kappa, args.num_points
        )

        panel_slug = f"disk_{disk_ens}_to_square_{square_ens}"
        plot_panel(
            args.out_dir / f"{panel_slug}.png",
            x_obc,
            y_obc,
            x_pbc,
            y_pbc,
            args.disk_radius,
            args.square_L,
            disk_ens,
            square_ens,
            log_scale=False,
        )
        plot_panel(
            args.out_dir / f"{panel_slug}_loglog.png",
            x_obc,
            y_obc,
            x_pbc,
            y_pbc,
            args.disk_radius,
            args.square_L,
            disk_ens,
            square_ens,
            log_scale=True,
        )

        for t_disk, t_square in zip(x_obc, y_obc):
            matched_rows.append([panel_title, "square_obc", f"{t_disk:.12f}", f"{t_square:.12f}"])
        for t_disk, t_square in zip(x_pbc, y_pbc):
            matched_rows.append([panel_title, "square_pbc", f"{t_disk:.12f}", f"{t_square:.12f}"])

    write_tsv(
        args.out_dir / "kappa_temperature_map.tsv",
        ["panel", "square_geometry", "temperature_disk", "temperature_square"],
        matched_rows,
    )

    metadata = "\n".join(
        [
            'comparison = "Equal-compressibility temperature map"',
            f"disk_radius = {args.disk_radius}",
            f"square_L = {args.square_L}",
            f"num_interpolation_points = {args.num_points}",
            'panels = "disk_gce_to_square_gce, disk_gce_to_square_ce, disk_ce_to_square_gce, disk_ce_to_square_ce"',
            'square_curves = "square_obc, square_pbc"',
            'method = "log-temperature interpolation of kappa(T), then inverse interpolation T(kappa)"',
            f'data_sources = "results/non_interacting/spinless_disk_comparison_by_radius/radius_{args.disk_radius}/thermo_vs_beta.tsv, results/non_interacting/spinless_square_pbc_comparison_by_L/L_{args.square_L}/square_pbc_thermo_vs_beta.tsv, results/non_interacting/spinless_square_obc_comparison_by_L/L_{args.square_L}/square_obc_thermo_vs_beta.tsv"',
        ]
    )
    (args.out_dir / "metadata.toml").write_text(metadata + "\n")

    print("Created equal-compressibility temperature map")
    print(f"  output directory = {args.out_dir}")


if __name__ == "__main__":
    main()
