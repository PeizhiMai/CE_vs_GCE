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

The checkpoint gate was then repeated with the final project drivers, using
separate roots and OpenBLAS for both package versions. For attractive and
spin-HS PBC cases under both v2.0.11 and v2.0.12, an intentional warmup stop
returned exit code 13 and resumed to the same 4 thermalization and 4
measurement counters as an uninterrupted run. All six scientific CSVs and
the geometry/model summary were byte-identical between uninterrupted and
resumed runs. They were also byte-identical across v2.0.11 and v2.0.12 for
both execution modes. Stable geometry, boundary, hopping-model, seed, and
counter metadata matched exactly; version/commit provenance remained
intentionally distinct. Full hashes and field checks are in
`docs/validation/smoqy_v2011_v2012_pbc_checkpoint_regression_20260720.json`.

## OBC-fork follow-up

The pinned OBC fork is documented in `docs/smoqy_obc_fork.md`. After applying
the fork, the same fixed-seed attractive and spin-HS PBC cases produced six
byte-identical scientific CSV files per case relative to the unmodified
registry v2.0.12 package. Exact per-file hashes are stored in
`docs/validation/smoqy_obc_postfork_pbc_regression_20260720.json`.
