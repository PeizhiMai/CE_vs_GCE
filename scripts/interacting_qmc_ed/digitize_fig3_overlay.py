#!/usr/bin/env python3
from __future__ import annotations

import csv
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/Users/cosdis/Desktop/projects/CE_GCE")
SVG_PATH = ROOT / "external/papers-code-CanEnsAFQMC/figures/Energy_Lx6Ly6.svg"
DEFAULT_RUN_PATH = ROOT / "results/interacting_qmc_ed/fig3_ce_energy_scan/ce_energy.tsv"
DEFAULT_OUT_DIR = ROOT / "results/interacting_qmc_ed/fig3_digitized_overlay"

# Main-axes limits encoded in the published SVG and plotting notebook.
XMIN_PX = 274.889
XMAX_PX = 2352.76
YTOP_PX = 47.2441
YBOT_PX = 1414.54
XMIN = 0.0
XMAX = 13.0
YMIN = -0.9
YMAX = -0.1


def px_to_x(x_px: float) -> float:
    return XMIN + (x_px - XMIN_PX) * (XMAX - XMIN) / (XMAX_PX - XMIN_PX)


def px_to_y(y_px: float) -> float:
    return YMAX - (y_px - YTOP_PX) * (YMAX - YMIN) / (YBOT_PX - YTOP_PX)


def extract_marker_centers(svg_text: str, color: str) -> list[tuple[float, float]]:
    pattern = re.compile(
        rf'<path clip-path="url\(#clip103\)" d="([^"]+)" fill="{re.escape(color)}" '
        r'fill-rule="evenodd" fill-opacity="1" stroke="none"/>',
        re.IGNORECASE,
    )
    centers: list[tuple[float, float]] = []
    for d in pattern.findall(svg_text):
        nums = [float(x) for x in re.findall(r"[-+]?[0-9]*\.?[0-9]+", d)]
        xs = nums[0::2]
        ys = nums[1::2]
        centers.append(((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0))
    centers.sort(key=lambda item: item[0])
    return centers


def write_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def read_run_results(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "beta": float(row["beta"]),
                    "total_per_particle": float(row["total_per_particle"]),
                    "total_stderr": float(row["total_stderr"]),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay current CE run against digitized Fig. 3 points.")
    parser.add_argument(
        "--run-path",
        type=Path,
        default=DEFAULT_RUN_PATH,
        help="Path to the CE run TSV to overlay.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for generated TSV and plot outputs.",
    )
    parser.add_argument(
        "--run-label",
        default="This run CE",
        help="Legend label for the overlaid CE run.",
    )
    parser.add_argument(
        "--extra-run",
        action="append",
        default=[],
        help="Additional run overlay in the form path:label",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    svg_text = SVG_PATH.read_text()

    digitized = {
        "ce": extract_marker_centers(svg_text, "#009af9"),
        "gce": extract_marker_centers(svg_text, "#e26f46"),
    }

    digitized_rows: dict[str, list[list[object]]] = {}
    for label, centers in digitized.items():
        rows = []
        for x_px, y_px in centers:
            rows.append(
                [
                    f"{px_to_x(x_px):.6f}",
                    f"{px_to_y(y_px):.6f}",
                    f"{x_px:.3f}",
                    f"{y_px:.3f}",
                ]
            )
        digitized_rows[label] = rows
        write_tsv(
            out_dir / f"fig3_{label}_digitized.tsv",
            ["beta_digitized", "energy_per_particle_digitized", "x_px", "y_px"],
            rows,
        )

    run_rows = read_run_results(args.run_path)
    extra_runs: list[tuple[list[dict[str, float]], str]] = []
    for item in args.extra_run:
        path_str, label = item.split(":", 1)
        extra_runs.append((read_run_results(Path(path_str)), label))

    ce_digitized = [
        {"beta": float(row[0]), "energy": float(row[1])}
        for row in digitized_rows["ce"]
        if float(row[0]) <= 6.0 + 1e-9
    ]

    comparison_rows: list[list[object]] = []
    for run in run_rows:
        beta = run["beta"]
        nearest = min(ce_digitized, key=lambda row: abs(row["beta"] - beta))
        comparison_rows.append(
            [
                f"{beta:.1f}",
                f"{nearest['beta']:.6f}",
                f"{nearest['energy']:.6f}",
                f"{run['total_per_particle']:.6f}",
                f"{run['total_stderr']:.6f}",
                f"{run['total_per_particle'] - nearest['energy']:.6f}",
            ]
        )

    write_tsv(
        out_dir / "fig3_ce_vs_run.tsv",
        [
            "beta_run",
            "beta_digitized",
            "fig3_ce_digitized",
            "run_ce_total",
            "run_ce_stderr",
            "run_minus_digitized",
        ],
        comparison_rows,
    )

    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=180)

    ce_x = [float(row[0]) for row in digitized_rows["ce"]]
    ce_y = [float(row[1]) for row in digitized_rows["ce"]]
    gce_x = [float(row[0]) for row in digitized_rows["gce"]]
    gce_y = [float(row[1]) for row in digitized_rows["gce"]]
    run_x = [row["beta"] for row in run_rows]
    run_y = [row["total_per_particle"] for row in run_rows]
    run_err = [row["total_stderr"] for row in run_rows]

    ax.plot(ce_x, ce_y, color="#009af9", marker="s", markersize=5, linewidth=1.8, label="Fig. 3 CE (digitized)")
    ax.plot(gce_x, gce_y, color="#e26f46", marker="s", markersize=5, linewidth=1.8, label="Fig. 3 GCE (digitized)")
    ax.errorbar(
        run_x,
        run_y,
        yerr=run_err,
        color="#1b9e77",
        marker="o",
        markersize=5,
        linewidth=1.6,
        capsize=3,
        label=args.run_label,
    )

    extra_colors = ["#d95f02", "#7570b3", "#66a61e", "#e7298a"]
    for idx, (rows, label) in enumerate(extra_runs):
        color = extra_colors[idx % len(extra_colors)]
        ax.errorbar(
            [row["beta"] for row in rows],
            [row["total_per_particle"] for row in rows],
            yerr=[row["total_stderr"] for row in rows],
            color=color,
            marker="^",
            markersize=5,
            linewidth=1.4,
            capsize=3,
            label=label,
        )

    ax.set_xlim(0, 13)
    ax.set_ylim(-0.9, -0.1)
    ax.set_xlabel("beta")
    ax.set_ylabel("E/N")
    ax.grid(False)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Fig. 3 Digitized Points vs Current CE Run")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig3_overlay.png", bbox_inches="tight")
    fig.savefig(out_dir / "fig3_overlay.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
