# L=6 OBC CE/GCE equal-time thermometry (2026-07-20)

This workflow is the isolated open-boundary mirror of the active L=6 PBC
calculation.  It never reads a PBC checkpoint and never writes below the PBC
production tree.

## Fixed provenance and paths

- Project base: `8361604f9d1e0df7da347b2af010862c99626262` plus this workflow commit.
- SmoQyDQMC: official v2.0.12 with pinned OBC fork commit
  `c5f0c81bc98029bae585e0cb283428e293553999`.
- Julia 1.12.1 and pinned `OpenMPI_jll`.
- CADES wrapper validation is pinned to `/usr/bin/python3.11` (rather than the
  login-node `python3`, which is Python 3.6 and lacks `tomllib`).
- CADES code: `/home/9pm/nUHubbard_obc_dev`.
- CADES data: `/home/9pm/nUHubbard_obc_runs`.
- Every run root contains `_obc_`; every CE/GCE invocation explicitly uses
  `--boundary=open`.
- CADES scheduling is `ccsd/burst/default`, with unthrottled arrays.  Account
  moves are never automatic.

Each manifest records the boundary, project/fork commits, CanEns patch hashes,
36 sites, 60 undirected NN bonds, 50 undirected NNN bonds, estimator
normalizations, and chemical-potential provenance.

## Physics grid

- Open 6x6 lattice, `t=1`, `t'=0`, `dtau=0.1`.
- `Ntot = 12,18,26,32`; balanced CE sectors are `(6,6)`, `(9,9)`, `(13,13)`,
  and `(16,16)`.
- `U = -5,-3,0,+3,+5`.
- `beta = 2,2.2,2.5,2.9,3.3,4,5,6.7,10`, plus `beta=20` for `U<=0`.
- 192 physical conditions: 152 interacting QMC and 40 exact U=0.
- 384 ensemble rows after CE and GCE are counted separately.

`exact_u0_l6_obc.py` evaluates all 40 exact conditions for both ensembles by
one-body OBC diagonalization and elementary-symmetric-polynomial occupation
moments; it does not enumerate `binomial(36,N)` configurations.

## CE production and positive-U pilot gate

Immediate CE arrays are:

| Stage | Rows | Ranks | Warmups | Measurements/rank |
|---|---:|---:|---:|---:|
| Attractive, beta<=10 | 72 | 32 | 5,000 | 50,000 |
| Attractive, beta=20 | 8 | 64 | 5,000 | 30,000 |
| Positive U, beta<=4 | 48 | 32 | 5,000 | 50,000 |
| Positive-U beta=5,6.7,10 pilot | 24 | 32 | 2,000 | 10,000 |

Positive-U pilots are classified independently of PBC:

- `abs(<phase>) >= 0.02`: admitted normal tier.
- `0.002 <= abs(<phase>) < 0.02`: admitted high-statistics tier.
- `abs(<phase>) < 0.002`: terminal `sign_limited`.

Admitted beta=5 normal rows use 32 ranks and 50,000 measurements/rank.  All
admitted beta=6.7/10 rows and all high-statistics rows use 64 ranks and 30,000
measurements/rank.  Pilot samples are never combined with production samples.
Positive-U CE uses spin-channel Hirsch HS, `force_symmetry=false`,
`phase_reweighted=true`, and global signed-numerator/global-phase-denominator
pooling.

## OBC GCE chemical-potential tuning

`import_pbc_mu_references.py` imports only L=6 PBC rows whose status is
`confirmed_within_abs_N_0p03`.  Imported values are frozen; a later changed PBC
value is treated as a provenance error.  Every imported row retains both the
L=6 PBC value and its inherited L=8 source.

The normative search policy is documented in `MU_TUNING_RULE.md`, adapted from
the PBC workflow rule at
`/home/9pm/nUHubbard/scripts/interacting_qmc_ed/ce_gce_eqtime_L6_thermometry_20260719/MU_TUNING_RULE.md`.
For each interacting target, OBC probes start at `mu_PBC` and
`mu_PBC +/- 0.02`.  If those points do not bracket the target, `tune_mu.py`
recenters on the completed OBC point closest in density and submits only one
directed 0.02 step; it never adds a mirrored point for symmetry.  A coarse
density-straddling bracket is refined one interior 0.02 step at a time.
Production is admitted only from a bracket no wider than 0.02, with both the
secant fit and independent confirmation inside that bracket and
`abs(N_OBC-Ntarget)<=0.03`.  Production must independently satisfy the same
particle-number tolerance.  Failed or wide-bracket production is superseded,
excluded from analysis/repair, and replaced only after a fresh confirmation.

All interacting GCE production uses 32 ranks, 5,000 warmups, and 50,000
measurements/rank.  GCE continues for all 152 targets even when the matching CE
condition is sign-limited.

## Checkpoint and queue contract

- Measurement interval 3, 1:30 walltime, 1.25-hour internal stop, hourly
  checkpoints.
- `checkpoint_reset_accumulators=false` for every fresh and resumed leg.
- Continuations and repairs preserve account/partition/QOS and exact resources.
- User-directed pending-task moves are recorded in
  `status_source/account_moves/active_account_overrides.tsv`; subsequent
  hourly imports preserve the destination account for those exact roots and
  their continuations.
- `repair_stale_l6_roots.py` repairs a root only when it is required,
  non-final, has fresh full checkpoints/statuses, and has no active or pending
  leg.  Queue/root mapping is ledger based and fails closed on unknown jobs,
  duplicates, or resource drift.
- Per-root continuation ledgers and append-only submission/repair ledgers make
  stage advancement idempotent.

## Measured equal-time quantities

CE and GCE use identical physical OBC site/bond lists:

1. Kinetic energy/site.
2. Double occupancy/site.
3. NN spin correlation averaged over 60 existing NN bonds.
4. NN connected-charge correlation averaged over 60 existing NN bonds.
5. NNN spin and connected-charge correlations averaged over 50 existing NNN
   bonds.
6. Total/interaction energies, local moment, site densities, achieved N, and
   phase/sign diagnostics.

Connected charge is formed only after global phase pooling and global
site-density pooling.  Translational wedges, FFT/momentum outputs, periodic
shell estimators, and unequal-time paths are forbidden.

## Build and local validation

From this workflow directory, using the requested virtual environment:

```bash
~/.venvs/myenv/bin/python prepare_workflow.py
~/.venvs/myenv/bin/python validate_workflow.py
~/.venvs/myenv/bin/python exact_u0_l6_obc.py \
  --outdir status_source/exact_u0_obc \
  --snapshot status_source/exact_u0_L6_obc_snapshot.tsv
```

The validator checks the 192/152/40 counts, unique roots, balanced sectors,
36/60/50 geometry, no wrap bonds, exact U=0 moments, PBC-mu provenance,
positive-U phase flags, launchers, and thermometry line-break behavior.

## CADES smoke and staged launch

After deploying the exact workflow commit and initializing pinned submodules:

```bash
./submit_smoke_cades.sh
./check_smoke_cades.sh
./submit_initial_cades.sh
./install_hourly_driver_cades.sh
```

The smoke gate runs two-rank attractive/spin-HS CE and GCE OBC cases and
requires strict OBC geometry/provenance/table outputs.  Initial production
submits the four CE arrays plus all currently available PBC-seeded OBC probes.

`advance_l6_stages_cades.sh` runs hourly at minute 17.  It imports newly
confirmed PBC seeds, submits only new probes, repairs eligible stale roots,
classifies completed pilots, advances secant/confirmation probes, submits
confirmed GCE and admitted CE production, and writes an audit.  Repeated calls
do not duplicate roots.

## Strict collection and thermometry

```bash
python3.11 collect_l6_results.py \
  --exact-snapshot status_source/exact_u0_L6_obc_snapshot.tsv \
  --snapshot status_source/analysis/l6_obc_snapshot_current.tsv \
  --status status_source/analysis/l6_obc_condition_status_current.tsv
```

Strict-final QMC rows require complete expected-rank, checkpoint/status,
rank-local site-accumulator, and four primary-table coverage.  The collector
recomputes global phase pools and connected charge, rejects incorrect bond
counts or forbidden periodic outputs, and enforces the GCE density tolerance.

Thermometry uses actual `T=1/beta` and

```text
(T_GCE - T_CE) / T_CE
```

for kinetic energy/site, double occupancy/site, NN spin, and NN connected
charge.  Plots join only actual simulated points with a unique inferred
temperature.  Curves break at no-solution, multiple-solution,
flat-calibration, unfinished, or sign-limited points.

## Final deliverables

After both L=6 PBC and OBC grids are terminal, run locally:

```bash
./finalize_l6_obc_results_local.sh
```

The finalizer performs strict remote collection for both boundaries, creates
the 14 OBC analysis panels, builds a standalone 18-slide PowerPoint with
artifact-tool, renders all 18 slides, runs overflow QA, creates PDF/TSV/PNG
outputs, and mirrors the timestamped package to:

- `/Users/cosdis/Desktop/projects/CE_GCE/results`
- `/Users/cosdis/CE_GCE_no_icloud/results`

Slides 2-15 are the four primary comparisons, two NNN comparisons, three
mismatch-versus-T panels, thermometry scorecard, and four large slide-wide/tall
thermometer panels.  Slides 16-18 compare PBC/OBC thermometry, report boundary
coverage/sign/density tuning, and conclude.  The final mirrored no_icloud PPTX
is opened only after strict collection, rendering, and QA succeed.
