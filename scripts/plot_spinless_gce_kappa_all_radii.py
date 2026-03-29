#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/Users/cosdis/Desktop/projects/CE_GCE")
DEFAULT_INPUT_DIR = ROOT / "results" / "spinless_disk_compressibility_r4_to_r20_step2"
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "gce_kappa_vs_beta_all_radii.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot grand-canonical compressibility vs beta for all radii together."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing radius_*/compressibility_vs_beta.tsv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PNG path.",
    )
    return parser.parse_args()


def radius_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.name.split("_", 1)[1]
    return int(float(suffix)), suffix


def read_curve(path: Path) -> tuple[list[float], list[float]]:
    betas: list[float] = []
    kappas: list[float] = []
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            betas.append(float(row["beta"]))
            kappas.append(float(row["kappa_grand_canonical"]))
    return betas, kappas


def main() -> None:
    args = parse_args()
    radius_dirs = sorted(
        [path for path in args.input_dir.iterdir() if path.is_dir() and path.name.startswith("radius_")],
        key=radius_sort_key,
    )
    if not radius_dirs:
        raise SystemExit(f"No radius_* directories found in {args.input_dir}")

    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=180)

    for radius_dir in radius_dirs:
        radius_label = radius_dir.name.split("_", 1)[1]
        beta_path = radius_dir / "compressibility_vs_beta.tsv"
        betas, kappas = read_curve(beta_path)
        ax.plot(betas, kappas, linewidth=1.8, label=f"R={radius_label}")

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\kappa_{\mathrm{GCE}}$")
    ax.set_title("Spinless Disk: GCE Compressibility vs Beta")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=3)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
