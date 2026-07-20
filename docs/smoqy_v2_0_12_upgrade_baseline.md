# SmoQyDQMC v2.0.12 upgrade baseline

Date: 2026-07-20

## Environment policy

- New development is pinned to SmoQyDQMC `2.0.12` and Julia `1.12.x`.
- `julia_env/legacy/smoqy-v2.0.11` preserves the pre-upgrade lock for old
  production checkpoints.
- A v2.0.11 checkpoint must never be loaded by the v2.0.12 or OBC
  environments. OBC calculations always use fresh run roots.

## Package validation

The registry release was instantiated and precompiled with Julia 1.12.1.
`Pkg.test("SmoQyDQMC")` passed all 16 upstream tests.

## PBC scientific regression

One-rank `2x2`, `beta=1`, `dtau=0.1` fixed-seed (`123`) smoke calculations
were run with both v2.0.11 and v2.0.12:

| Driver | U | warmups | measurements | profile | Result |
|---|---:|---:|---:|---|---|
| Attractive Hubbard | -3 | 2 | 4 | equal-time-only | byte-identical CSV outputs |
| Spin-HS Hubbard | +3 | 2 | 4 | equal-time-only | byte-identical CSV outputs |

For each driver, `global_stats`, `local_stats`, position-space density, and
position-space spin-z CSV files had identical SHA-256 hashes between package
versions. This establishes the PBC baseline before any OBC package patch is
introduced.

Checkpoint/resume equivalence is a separate gate and must be repeated after
the OBC fork is pinned, because JLD2 state is only supported within the exact
same package commit.
