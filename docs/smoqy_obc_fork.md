# Pinned SmoQyDQMC v2.0.12 OBC fork

Date: 2026-07-20

## Pin and compatibility boundary

- Official base: SmoQyDQMC tag `v2.0.12`, commit
  `2a643a9655225e9eec3f6a689e73343f27a54b80`.
- OBC fork: `PeizhiMai/SmoQyDQMC.jl`, branch `obc-v2.0.12`, commit
  `c5f0c81bc98029bae585e0cb283428e293553999`.
- The parent project uses the fork through `external/SmoQyDQMC`; every OBC
  output records the package version and exact commit.
- Existing v2.0.11 checkpoints remain tied to
  `julia_env/legacy/smoqy-v2.0.11`. They must never be opened by this fork.
  OBC always starts in a fresh run root containing `_obc_`.

## Implemented package behavior

The fork permits periodic, open, and mixed flags in `ModelGeometry`. For an
open direction it:

1. rejects a nonzero twist;
2. builds each hopping bond type independently;
3. concatenates only physical bonds that remain inside the finite lattice;
4. records variable-length cumulative `bond_slices`; and
5. initializes one hopping value per existing bond.

The original all-periodic construction is retained as a separate fast path.
Dense tight-binding/Hubbard propagation is the only validated OBC path.
Checkerboard propagation, electron-phonon and extended-Hubbard models, and
SmoQy's translational correlation initializers fail explicitly under OBC.
The project drivers additionally reject unequal-time, momentum-space, and BKT
profiles under OBC.

## Deterministic validation

- `2x2`: 4 NN and 2 NNN undirected physical bonds.
- `3x3`: 12 NN and 8 NNN undirected physical bonds.
- No wrap edges, Hermitian hopping, x-fastest site ordering, and mixed-boundary
  behavior are covered by package tests.
- A conventional one-orbital OBC geometry and a periodic single-supercell
  representation produce identical one-body Hamiltonians for `2x2` and `3x3`.
- The full fork test suite passed 66/66 tests with Julia 1.12.1.

## Post-patch PBC regression

Fixed-seed one-rank attractive-Hubbard and positive-U spin-HS `2x2` PBC
calculations were repeated against the unmodified registry v2.0.12 package.
All six scientific CSV files in each run were byte-identical, including global
and local statistics and equal-time density/spin files. Per-file hashes are in
`docs/validation/smoqy_obc_postfork_pbc_regression_20260720.json`.

This result is a regression gate, not permission to mix checkpoints: resumed
runs still require the identical fork commit recorded in their checkpoint.
