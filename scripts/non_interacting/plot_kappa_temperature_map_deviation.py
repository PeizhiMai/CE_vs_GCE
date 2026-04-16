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

PANEL_ORDER = [
    "Disk GCE to Square GCE",
    "Disk GCE to Square CE",
    "Disk CE to Square GCE",
    "Disk CE to Square CE",
]

PANEL_TITLE = {
    "Disk GCE to Square GCE": "Disk GCE -> Square GCE",
    "Disk GCE to Square CE": "Disk GCE -> Square CE",
    "Disk CE to Square GCE": "Disk CE -> Square GCE",
    "Disk CE to Square CE": "Disk CE -> Square CE",
}

PANEL_SLUG = {
    "Disk GCE to Square GCE": "disk_gce_to_square_gce",
    "Disk GCE to Square CE": "disk_gce_to_square_ce",
    "Disk CE to Square GCE": "disk_ce_to_square_gce",
    "Disk CE to Square CE": "disk_ce_to_square_ce",
}

COLOR_MAP = {
    "square_obc": "#8c510a",
    "square_pbc": "#7570b3",
}

LABEL_MAP = {
    "square_obc": "Square OBC",
    "square_pbc": "Square PBC",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def read_metadata_value(path: Path, key: str) -> str | None:
    for line in path.read_text().splitlines():
        if line.startswith(f"{key} = "):
            value = line.split("=", 1)[1].strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            return value
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot linear-scale deviation from y=x for equal-compressibility temperature maps."
    )
    parser.add_argument(
        "--map-dir",
        type=Path,
        required=True,
        help="Directory containing kappa_temperature_map.tsv and metadata.toml.",
    )
    args = parser.parse_args()

    map_dir = args.map_dir
    rows = read_tsv(map_dir / "kappa_temperature_map.tsv")
    metadata = map_dir / "metadata.toml"
    disk_radius = read_metadata_value(metadata, "disk_radius") or "?"
    square_L = read_metadata_value(metadata, "square_L") or "?"

    deviation_rows: list[list[object]] = []
    grouped: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    for panel in PANEL_ORDER:
        for geom in ("square_obc", "square_pbc"):
            subset = [row for row in rows if row["panel"] == panel and row["square_geometry"] == geom]
            x = np.asarray([float(row["temperature_disk"]) for row in subset], dtype=float)
            y = np.asarray([float(row["temperature_square"]) for row in subset], dtype=float)
            order = np.argsort(x)
            x = x[order]
            y = y[order]
            grouped[(panel, geom)] = (x, y - x)
            for xd, yd in zip(x, y):
                deviation_rows.append([panel, geom, f"{xd:.12f}", f"{yd:.12f}", f"{(yd - xd):.12f}", f"{abs(yd - xd):.12f}"])

    for panel in PANEL_ORDER:
        fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=180)
        for geom in ("square_obc", "square_pbc"):
            x, delta = grouped[(panel, geom)]
            ax.plot(x, delta, color=COLOR_MAP[geom], linewidth=2.2, label=LABEL_MAP[geom])
        ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
        disk_ens = "GCE" if "Disk GCE" in panel else "CE"
        square_ens = "GCE" if "Square GCE" in panel else "CE"
        ax.set_xlabel(rf"$T_{{\mathrm{{disk}},\,R={disk_radius},\,{disk_ens}}}$", fontsize=17)
        ax.set_ylabel(
            rf"$\Delta T = T_{{\mathrm{{square}},\,L={square_L},\,{square_ens}}} - T_{{\mathrm{{disk}},\,R={disk_radius},\,{disk_ens}}}$",
            fontsize=17,
        )
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=15)
        ax.legend(frameon=False, fontsize=15)
        fig.tight_layout()
        fig.savefig(map_dir / f"{PANEL_SLUG[panel]}_deviation.png", bbox_inches="tight")
        plt.close(fig)

    write_tsv(
        map_dir / "kappa_temperature_deviation.tsv",
        [
            "panel",
            "square_geometry",
            "temperature_disk",
            "temperature_square",
            "delta_temperature",
            "abs_delta_temperature",
        ],
        deviation_rows,
    )

    print("Created kappa temperature deviation outputs")
    print(f"  map directory = {map_dir}")


if __name__ == "__main__":
    main()
