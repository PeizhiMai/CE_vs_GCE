#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare CE-QMC and ED G(r,tau)/G(k,tau) tables.")
    p.add_argument("--qmc-dir", type=Path, required=True)
    p.add_argument("--ed-dir", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--branch", choices=["add"], default="add")
    p.add_argument("--space", choices=["r", "k", "both"], default="both")
    p.add_argument("--min-slice", type=int, default=None)
    p.add_argument("--max-slice", type=int, default=None)
    return p.parse_args()


def compare_one(
    qmc_dir: Path,
    ed_dir: Path,
    outdir: Path,
    space: str,
    branch: str,
    min_slice: Optional[int],
    max_slice: Optional[int],
) -> Dict[str, float]:
    qmc_path = qmc_dir / f"greens_{space}_tau_{branch}_qmc.tsv"
    ed_path = ed_dir / f"greens_{space}_tau_{branch}_ed.tsv"

    if space == "r":
        keys = ["slice", "dx", "dy"]
    else:
        keys = ["slice", "nx", "ny"]

    def read_rows(path: Path) -> List[Dict[str, str]]:
        with path.open(newline="") as f:
            return list(csv.DictReader(f, delimiter="\t"))

    def key_for(row: Dict[str, str]) -> Tuple[int, ...]:
        return tuple(int(row[k]) for k in keys)

    def to_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
        val = row.get(key, "")
        if val == "" or val is None:
            return default
        return float(val)

    qmc_rows = read_rows(qmc_path)
    ed_by_key = {key_for(row): row for row in read_rows(ed_path)}
    out_rows = []
    for qrow in qmc_rows:
        sl = int(qrow["slice"])
        if min_slice is not None and sl < min_slice:
            continue
        if max_slice is not None and sl > max_slice:
            continue
        erow = ed_by_key.get(key_for(qrow))
        if erow is None:
            continue
        gre_q = to_float(qrow, "Greal_mean")
        gim_q = to_float(qrow, "Gimag_mean")
        gre_e = to_float(erow, "Greal")
        gim_e = to_float(erow, "Gimag")
        delta_re = gre_q - gre_e
        delta_im = gim_q - gim_e
        delta_abs = math.hypot(delta_re, delta_im)
        stderr_abs = math.hypot(to_float(qrow, "Greal_stderr"), to_float(qrow, "Gimag_stderr"))
        z_abs = delta_abs / stderr_abs if stderr_abs > 0.0 else float("nan")
        row = {}
        for k in keys:
            row[k] = qrow[k]
        row.update({
            "tau": qrow.get("tau", erow.get("tau", "")),
            "Greal_mean": qrow.get("Greal_mean", ""),
            "Greal_stderr": qrow.get("Greal_stderr", ""),
            "Gimag_mean": qrow.get("Gimag_mean", ""),
            "Gimag_stderr": qrow.get("Gimag_stderr", ""),
            "Greal": erow.get("Greal", ""),
            "Gimag": erow.get("Gimag", ""),
            "delta_re": "{:.17g}".format(delta_re),
            "delta_im": "{:.17g}".format(delta_im),
            "delta_abs": "{:.17g}".format(delta_abs),
            "stderr_abs": "{:.17g}".format(stderr_abs),
            "z_abs": "{:.17g}".format(z_abs),
        })
        if space == "k":
            row["kx_qmc"] = qrow.get("kx", "")
            row["ky_qmc"] = qrow.get("ky", "")
        out_rows.append(row)

    keep = keys + ["tau", "Greal_mean", "Greal_stderr", "Gimag_mean", "Gimag_stderr", "Greal", "Gimag", "delta_re", "delta_im", "delta_abs", "stderr_abs", "z_abs"]
    if space == "k":
        keep = ["slice", "tau", "nx", "ny", "kx_qmc", "ky_qmc"] + keep[4:]
    with (outdir / f"greens_{space}_tau_{branch}_comparison.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keep, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    delta_abs_values = [float(row["delta_abs"]) for row in out_rows]
    valid_z = [
        float(row["z_abs"])
        for row in out_rows
        if float(row["stderr_abs"]) > 1e-12 and math.isfinite(float(row["z_abs"]))
    ]
    return {
        "nrows": float(len(out_rows)),
        "max_abs_delta": max(delta_abs_values) if delta_abs_values else float("nan"),
        "rms_abs_delta": math.sqrt(sum(x * x for x in delta_abs_values) / len(delta_abs_values)) if delta_abs_values else float("nan"),
        "max_z_abs": max(valid_z) if valid_z else float("nan"),
        "rms_z_abs": math.sqrt(sum(x * x for x in valid_z) / len(valid_z)) if valid_z else float("nan"),
    }


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    spaces = ["r", "k"] if args.space == "both" else [args.space]
    rows = []
    for space in spaces:
        stats = compare_one(args.qmc_dir, args.ed_dir, args.outdir, space, args.branch, args.min_slice, args.max_slice)
        row = {"space": space}
        row.update(stats)
        rows.append(row)
    with (args.outdir / "green_tau_space_comparison_summary.tsv").open("w", newline="") as f:
        fieldnames = ["space", "nrows", "max_abs_delta", "rms_abs_delta", "max_z_abs", "rms_z_abs"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
