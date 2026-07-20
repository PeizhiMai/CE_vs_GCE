#!/usr/bin/env python3
"""Focused tests for the OBC validation acceptance analyzer."""

from __future__ import annotations

import importlib.util
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


if __name__ == "__main__":
    unittest.main()
