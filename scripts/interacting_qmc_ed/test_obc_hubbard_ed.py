#!/usr/bin/env python3
"""Fast deterministic checks for the OBC exact-diagonalization oracle."""

from __future__ import annotations

import math

import numpy as np

import obc_hubbard_ed as ed


def assert_summary_close(left: dict[str, object], right: dict[str, object]) -> None:
    for name in (
        "kinetic_per_site",
        "double_occupancy_per_site",
        "nn_spin_s_s",
        "nn_connected_charge",
        "nnn_spin_s_s",
        "nnn_connected_charge",
    ):
        assert math.isclose(
            float(left[name]), float(right[name]), rel_tol=1e-11, abs_tol=1e-11
        ), (name, left[name], right[name])


def main() -> None:
    geometry = ed.build_square_geometry(2, 2)
    assert len(geometry.nn_bonds) == 4
    assert len(geometry.nnn_bonds) == 2

    for interaction in (-3.0, 3.0):
        # Force the C4-resolved branch even though the 2x2 sector is small,
        # and compare it with complete unsymmetrized dense diagonalization.
        dense = ed.diagonalize_sector(
            geometry, 2, 2, interaction, 5.0, dense_threshold=1000
        )
        c4 = ed.diagonalize_sector(
            geometry,
            2,
            2,
            interaction,
            5.0,
            dense_threshold=1,
            initial_k=64,
            max_k=128,
        )
        assert dense.complete and c4.complete
        assert np.allclose(dense.eigenvalues, c4.eigenvalues, rtol=0, atol=2e-11)
        assert_summary_close(
            ed.canonical_summary(geometry, dense, 5.0),
            ed.canonical_summary(geometry, c4, 5.0),
        )

        # Check the spin-swap plus bipartite particle-hole reuse against an
        # independently diagonalized N>V sector.
        transformed = ed.get_spectrum(
            geometry,
            3,
            3,
            interaction,
            5.0,
            cache_dir=None,
            boltzmann_tolerance=1e-12,
            dense_threshold=1000,
            initial_k=64,
            max_k=128,
        )
        direct = ed.diagonalize_sector(
            geometry, 3, 3, interaction, 5.0, dense_threshold=1000
        )
        assert np.allclose(
            transformed.eigenvalues, direct.eigenvalues, rtol=0, atol=2e-11
        )
        assert_summary_close(
            ed.canonical_summary(geometry, transformed, 5.0),
            ed.canonical_summary(geometry, direct, 5.0),
        )

    print("OBC Hubbard ED symmetry/reuse checks: PASS")


if __name__ == "__main__":
    main()
