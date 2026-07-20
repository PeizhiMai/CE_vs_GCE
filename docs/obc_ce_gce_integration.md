# Open-boundary CE/GCE integration and validation

Date: 2026-07-20

## Compatibility boundary

This implementation is isolated from checkpointed production. Existing
SmoQyDQMC v2.0.11 runs remain on `julia_env/legacy/smoqy-v2.0.11`. New work
uses Julia 1.12.1, SmoQyDQMC v2.0.12, and the pinned OBC fork commit
`c5f0c81bc98029bae585e0cb283428e293553999`. `julia_env/LocalPreferences.toml`
selects `OpenMPI_jll`, including on CADES. A checkpoint may be resumed only by
the same boundary setting and exact dependency commit/patch set.

The CADES validation defaults are isolated as well: source code lives in
`/home/9pm/nUHubbard_obc_dev` and run data in
`/home/9pm/nUHubbard_obc_runs`. They never reuse production run roots.

The CE backend remains CanEnsAFQMC base commit
`21b4f6815d0b836973064ff8401fb2ba9c23b802`, reconstructed by applying the
existing current-response patch followed by `patches/CanEnsAFQMC-obc.patch`.
`bootstrap_canensafqmc.jl` verifies the base commit, applies both patches
idempotently, and develops that exact tree into `julia_env`.

## Public interface and supported scope

Both production drivers accept `--boundary=periodic|open`; the default is
unchanged PBC. OBC v1 means open in both spatial directions and requires a
fresh output root containing `_obc_`.

- CE uses an explicit no-wrap CanEnsAFQMC hopping matrix and direct real-space
  estimators. Positive-U CE is rejected unless it uses spin-channel Hirsch HS,
  `force_symmetry=false`, and global phase reweighting.
- Attractive and spin-HS GCE use the patched SmoQyDQMC geometry and the same
  shared physical bond lists. Checkpoints include the custom equal-time
  accumulator and exact fork provenance.
- Dense tight-binding/Hubbard propagation and equal-time or density-only GCE
  profiles are supported.
- OBC checkerboard, electron-phonon/SSH, extended-Hubbard, translational/FFT,
  unequal-time, momentum-space, and BKT paths fail explicitly.

The unchanged PBC path was retested after project integration. Four CE
scientific TSVs and all six CSVs from both attractive and spin-HS GCE
fixed-seed runs were byte-identical to the pre-integration drivers.
Every MPI runner selects the project explicitly with
`mpiexecjl --project=julia_env`, preventing a global MPI launcher from being
mixed with the pinned `OpenMPI_jll` runtime.

## Shared Hamiltonian and estimators

`square_lattice_geometry.jl` is the single project-level definition of
x-fastest site ordering, NN/NNN physical bonds, hopping, and normalization.
The CE, SmoQyDQMC, and Python ED one-body Hamiltonians agree entry by entry.
The patched conventional SmoQyDQMC OBC geometry also agrees with its
single-supercell validation oracle up to site permutation.

For OBC, each ensemble measures:

1. kinetic energy divided by the number of sites;
2. double occupancy divided by the number of sites;
3. NN spin correlation, using
   `(n_up-n_dn)_i (n_up-n_dn)_j`, averaged over existing undirected NN bonds;
4. NN connected charge, with the disconnected term formed from globally
   phase-reweighted site means and averaged over existing NN bonds;
5. the analogous NNN spin and connected-charge quantities;
6. total/interaction energies, local moment, site densities, achieved `N`,
   and phase/sign diagnostics.

CE and GCE both pool the signed observable numerator and phase denominator
globally. Connected charge is recomputed only after pooling site means, rather
than averaging nonlinear rank-local connected values. Acceptance errors use a
leave-one-rank jackknife. A numerical-zero global or leave-one-rank phase
denominator is fatal.

Every accepted row must have full expected-rank coverage, per-rank site
accumulators, and these four primary files:

- `equal_time_kinetic_per_site_qmc.tsv`
- `equal_time_double_occupancy_per_site_qmc.tsv`
- `equal_time_nn_spin_qmc.tsv`
- `equal_time_nn_connected_charge_qmc.tsv`

## Deterministic and checkpoint validation

- `2x2` OBC has 4 NN and 2 NNN undirected bonds.
- `3x3` OBC has 12 NN and 8 NNN undirected bonds.
- Geometry checks cover Hermiticity, no wrap edges, exact site ordering,
  CanEns/SmoQy/shared Hamiltonian identity, and the single-supercell oracle.
- A forced C4-sector decomposition and the spin-swap/particle-hole reuse path
  reproduce complete unsymmetrized `2x2` ED spectra and observables for both
  `U=-3` and `U=+3`.
- A synthetic two-rank CE combiner test verifies the nonlinear connected-charge
  pooling result.
- A two-rank MPI accumulator test verifies global phase pooling, site pooling,
  and rank jackknives.
- A CE checkpoint stop during warmup resumed to seven byte-identical scientific
  tables relative to an uninterrupted run, with 4/4 samples retained.
- An attractive-GCE checkpoint stop resumed with all counters retained; ten
  scientific/rank tables agree with an uninterrupted run to a maximum absolute
  difference of `1.83e-14`.

Exact hashes and test counts are recorded in
`docs/validation/obc_ce_gce_local_regression_20260720.json`. The end-to-end
two-rank smoke runner additionally exercised CE and attractive-GCE checkpoint
resume plus a fresh positive-U spin-HS GCE run and wrote all four primary
tables.

## ED oracle and QMC acceptance workflow

`obc_hubbard_ed.py` supplies complete thermal ED for `2x2`, exact one-body
CE/GCE traces at `U=0`, and C4-sector-resolved low-energy ED for `3x3`. Spin
interchange and the exact bipartite particle-hole map reduce duplicate sectors.
Every truncated sector records the number of omitted states, a conservative
first-omitted energy lower guard, eigen-residuals, and an omitted Boltzmann
weight bound. No `3x3` reference is accepted above `1e-8`.
Eigenpair residuals are independently required to remain below `1e-8`.

The frozen reference set in
`docs/validation/obc_ed_validation_20260720` passed all deterministic gates:
2 geometry rows, 4 exact `U=0` checks, 12 interacting CE conditions, and 12
ED-tuned GCE conditions. Across the accepted interacting references, the
largest omitted-Boltzmann-weight bound is `8.945e-11`, the largest CE
eigenpair residual is `3.224e-12`, the largest GCE eigenpair residual is
`2.992e-11`, and the largest GCE particle-number mismatch is `2.352e-12`.
The largest exact-`U=0` particle-number mismatch is `2.842e-14`.

The validation workflow under
`scripts/interacting_qmc_ed/obc_ce_gce_validation_20260720` generates 72 unique
CE and 72 unique GCE roots: three `dtau` values and two seeds for each of 12
physical conditions. GCE chemical potentials come from the ED grand-canonical
trace and satisfy `|<N>-N_CE| <= 0.01`. The analyzer requires provenance,
geometry, rank/site coverage, all four primary tables, and no forbidden
translational output. Equal-work independent seed replicates are combined by
an arithmetic mean, with within-run and between-seed SEMs added in quadrature;
the resulting points enter weighted linear fits in `dtau^2`. All 96 zero-step
intercepts must agree with ED within three standard errors.

The final fresh validation arrays were CADES jobs `5486493` (CE) and `5486494`
(GCE). They completed 144/144 roots with full four-rank and four-primary-table
coverage. All 96/96 observable extrapolations and all 12/12 GCE density
extrapolations passed; the maximum absolute observable discrepancy was
`2.5124215563` standard errors. Full provenance, Slurm coverage, phase minima,
superseded-generation notes, and tabulated results are in
`docs/validation/obc_ce_gce_cades_validation_20260720`.

The ED reference, PBC regression, package/fork tests, checkpoint-resume smoke,
and final CADES QMC gates now all pass. OBC production may use only the
explicitly supported dense Hubbard paths, exact pinned dependencies, and fresh
`_obc_` roots. PBC or v2.0.11 checkpoints remain incompatible.
