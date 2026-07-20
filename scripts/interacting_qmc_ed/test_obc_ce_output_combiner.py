#!/usr/bin/env python3
"""Regression test for nonlinear OBC connected-charge pooling across CE ranks."""

from __future__ import annotations

import csv
import math
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        coordinates = {1: (0, 0), 2: (1, 0), 3: (0, 1), 4: (1, 1)}
        densities = ([0.2, 0.4, 0.6, 0.8], [1.0, 1.2, 1.4, 1.6])
        raw_values = (0.5, 1.5)
        for rank in range(2):
            directory = root / "ranks" / f"rank_{rank:05d}"
            directory.mkdir(parents=True)
            denominator = 10.0
            global_row = {
                "beta": 2.0,
                "temperature": 0.5,
                "nup": 1,
                "ndn": 1,
                "ntotal": 2,
                "density": 0.5,
                "kinetic_per_site": -1.0,
                "kinetic_stderr": 0.1,
                "interaction_per_site": -0.2,
                "interaction_stderr": 0.1,
                "total_per_site": -1.2,
                "total_stderr": 0.1,
                "double_occupancy_per_site": 0.1,
                "double_occupancy_stderr": 0.01,
                "local_moment_z": 0.3,
                "local_moment_z_stderr": 0.01,
                "Kx_per_site": -0.5,
                "Kx_stderr": 0.1,
                "diamagnetic_minus_Kx_per_site": 0.5,
                "diamagnetic_minus_Kx_stderr": 0.1,
                "time_slices": 10,
                "nsamples": 10,
                "batches": 1,
                "phase_reweighted": False,
                "phase_sum": denominator,
                "average_phase": 1.0,
                "kinetic_per_site_signed_sum": -10.0,
                "interaction_per_site_signed_sum": -2.0,
                "total_per_site_signed_sum": -12.0,
                "double_occupancy_per_site_signed_sum": 1.0,
                "local_moment_z_signed_sum": 3.0,
                "Kx_per_site_signed_sum": -5.0,
                "diamagnetic_minus_Kx_per_site_signed_sum": 5.0,
            }
            write_rows(directory / "equal_time_observables_qmc.tsv", [global_row])
            bond_rows = []
            for shell, count in (("NN", 4), ("NNN", 2)):
                raw = raw_values[rank]
                bond_rows.append(
                    {
                        "shell": shell,
                        "bond_count": count,
                        "charge_corr_raw": raw,
                        "charge_corr_raw_stderr": 0.1,
                        "charge_corr_connected": 0.0,
                        "charge_corr_connected_stderr": 0.1,
                        "spin_corr_s_s": -0.2,
                        "spin_corr_s_s_stderr": 0.1,
                        "spin_corr_SzSz": -0.05,
                        "spin_corr_SzSz_stderr": 0.025,
                        "nsamples": 10,
                        "batches": 1,
                        "phase_reweighted": False,
                        "phase_sum": denominator,
                        "average_phase": 1.0,
                        "charge_corr_raw_signed_sum": raw * denominator,
                        "spin_corr_s_s_signed_sum": -0.2 * denominator,
                        "estimator_normalization": "existing undirected physical bonds",
                    }
                )
            write_rows(directory / "equal_time_bond_observables_qmc.tsv", bond_rows)
            site_rows = []
            for site, density in enumerate(densities[rank], start=1):
                x, y = coordinates[site]
                site_rows.append(
                    {
                        "site": site,
                        "x": x,
                        "y": y,
                        "density": density,
                        "density_stderr": math.nan,
                        "nsamples": 10,
                        "batches": 1,
                        "phase_reweighted": False,
                        "phase_sum": denominator,
                        "average_phase": 1.0,
                        "density_signed_sum": density * denominator,
                    }
                )
            write_rows(directory / "equal_time_site_density_qmc.tsv", site_rows)

        subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "combine_ce_green_tau_space_rank_outputs.py"),
                "--root-dir",
                str(root),
            ],
            check=True,
        )
        rows = list(
            csv.DictReader(
                (root / "equal_time_bond_observables_qmc.tsv").open(),
                delimiter="\t",
            )
        )
        nn = next(row for row in rows if row["shell"] == "NN")
        # Global site means are [0.6,0.8,1.0,1.2], so the NN disconnected
        # average is 0.81 and the globally pooled raw pair value is 1.0.
        assert abs(float(nn["charge_corr_connected"]) - 0.19) < 1e-12
        assert int(nn["nranks"]) == 2
        primary = list(
            csv.DictReader(
                (root / "equal_time_nn_connected_charge_qmc.tsv").open(),
                delimiter="\t",
            )
        )[0]
        assert abs(float(primary["value"]) - 0.19) < 1e-12
    print("OBC CE output combiner: PASS")


if __name__ == "__main__":
    main()
