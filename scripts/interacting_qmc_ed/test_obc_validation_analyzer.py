#!/usr/bin/env python3
"""Focused tests for the OBC validation acceptance analyzer."""

from __future__ import annotations

import importlib.util
import csv
import math
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parent
    / "obc_ce_gce_validation_20260720"
    / "analyze_validation.py"
)
SPEC = importlib.util.spec_from_file_location("obc_validation_analyzer", MODULE_PATH)
assert SPEC and SPEC.loader
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


class ValidationAnalyzerTests(unittest.TestCase):
    def test_final_manifests_have_disjoint_rank_rng_streams(self) -> None:
        manifest_dir = (
            Path(__file__).resolve().parent
            / "obc_ce_gce_validation_20260720"
            / "manifests_independent_rdm2_fix2"
        )
        for name in ("ce_validation_manifest.tsv", "gce_validation_manifest.tsv"):
            with (manifest_dir / name).open(newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 72)
            streams: list[int] = []
            for row in rows:
                streams.extend(
                    int(row["seed"]) + p_id
                    for p_id in range(int(row["expected_ranks"]))
                )
            self.assertEqual(len(streams), len(set(streams)))

    def test_signed_rank_density_ratio_and_jackknife(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rank, density in enumerate((0.4, 0.5, 0.6, 0.7)):
                (root / f"obc_equal_time_rank_pID-{rank}.tsv").write_text(
                    "name\tnsamples\tphase_sum_real\tphase_sum_imag\tabs_phase_sum\t"
                    "signed_sum_real\tsigned_sum_imag\traw_sum\traw_sumsq\n"
                    f"density\t10\t10\t0\t10\t{10*density}\t0\t0\t0\n"
                )
            achieved_n, error = ANALYZER.rank_ratio_jackknife(
                root, "density", scale=4.0
            )
            self.assertAlmostEqual(achieved_n, 2.2)
            self.assertTrue(math.isfinite(error))
            self.assertGreater(error, 0.0)

    def test_linear_dtau_squared_density_extrapolation(self) -> None:
        rows = [
            {"dtau": dtau, "value": 2.0 + 0.8 * dtau**2, "stderr": 0.01}
            for dtau in (0.2, 0.1, 0.05)
        ]
        intercept, error, slope, reduced_chi2 = ANALYZER.fit_dtau_squared(rows)
        self.assertAlmostEqual(intercept, 2.0)
        self.assertAlmostEqual(slope, 0.8)
        self.assertGreater(error, 0.0)
        self.assertLess(reduced_chi2, 1e-20)

    def test_equal_work_seed_combination_does_not_precision_weight(self) -> None:
        rows = [
            {"value": 1.0, "stderr": 0.1},
            {"value": 3.0, "stderr": 0.4},
        ]
        value, error = ANALYZER.combine_seed_rows(rows)
        within_sem = math.sqrt(0.1**2 + 0.4**2) / 2
        between_sem = 1.0
        self.assertAlmostEqual(value, 2.0)
        self.assertAlmostEqual(error, math.hypot(within_sem, between_sem))


if __name__ == "__main__":
    unittest.main()
