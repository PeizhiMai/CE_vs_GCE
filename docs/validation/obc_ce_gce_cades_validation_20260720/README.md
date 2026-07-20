# Final CADES OBC CE/GCE validation

Date: 2026-07-20
Status: **ACCEPTED**

This is the final validation gate for the pinned SmoQyDQMC v2.0.12 OBC
implementation and its CE/GCE integration. It uses the fresh
`independent_rdm2fix2` manifests; every earlier validation generation is
superseded and is not an input to this result.

## Provenance

- Julia: `1.12.1`
- SmoQyDQMC upstream version: `2.0.12`
- SmoQyDQMC OBC fork commit:
  `c5f0c81bc98029bae585e0cb283428e293553999`
- CanEnsAFQMC base commit:
  `21b4f6815d0b836973064ff8401fb2ba9c23b802`
- CanEnsAFQMC OBC response-patch SHA-256:
  `8e65be6f9f9bb010d444b45847b77cf56835b937cec54f62d794b25daf77a4c0`
- Remote source tree: `/home/9pm/nUHubbard_obc_dev`
- Remote run root:
  `/home/9pm/nUHubbard_obc_runs/obc_ce_gce_validation_independent_rdm2_20260720`

The production checkout and all v2.0.11 checkpoint roots were untouched.

## CADES execution and coverage

| Ensemble | Job | Array | State | Elapsed range | Median | Resources |
|---|---:|---:|---|---:|---:|---|
| CE | `5486493` | `0-71` | 72/72 completed, exit `0:0` | 80-599 s | 176.5 s | `ccsd/burst/default`, 4 ranks, 20G |
| GCE | `5486494` | `0-71` | 72/72 completed, exit `0:0` | 153-254 s | 197 s | `ccsd/burst/default`, 4 ranks, 20G |

All 144 roots have the four primary equal-time tables. CE has 288/288
rank-complete markers. GCE has 288/288 rank observable accumulators, 288/288
rank site accumulators, and 288/288 rank metadata files. No continuation was
needed. There were no fatal, JLD2, MPI, wrong-rank, numerical-zero-phase,
checkpoint-drift, or duplicate-root signatures. GCE stderr contained BLAS
configuration messages and 65 nonfatal adaptive stabilization-frequency
warnings; all affected tasks completed normally.

## Statistical acceptance

Each of the 12 physical conditions uses `dtau={0.20,0.10,0.05}` and two
independent equal-work seed replicates. The 288 `seed+pID` rank streams in
each ensemble are unique. Each run uses 4 ranks, 2,000 warmups, and 10,000
measurements per rank.

Because the seed replicates have identical work, their central estimates are
combined with an arithmetic mean rather than noisy inverse-SEM weights. The
reported seed-combined uncertainty is the quadrature sum of the propagated
within-run SEM and the between-seed SEM. Each individual run has already
performed the required global signed-numerator/phase-denominator pooling.
This policy is fixed in the analyzer and covered by a regression test.

- **96/96** primary-observable `dtau^2 -> 0` extrapolations agree with ED
  within three combined standard errors.
- The largest absolute discrepancy is `2.5124215563 sigma`, for `2x2` CE,
  `U=+3`, `beta=5`, `(Nup,Ndn)=(1,1)`, double occupancy/site.
- **12/12** GCE zero-step density extrapolations pass the density gate.
- The largest zero-step density point-estimate offset is `0.0226949539` for
  `2x2`, `U=-3`, `beta=5`, target `N=4`; its SEM is `0.0142215758`, so it is
  statistically compatible with the target. The underlying ED-tuned chemical
  potentials themselves miss their target by at most `2.352e-12`.
- 24 finite-`dtau` GCE rows lie outside `|N-N_target|<=0.01`; these are retained
  diagnostics because the ED-tuned chemical potential acquires an expected
  `O(dtau^2)` density shift.
- Minimum pooled average phase: CE `0.90315`; GCE `0.968`.

## Superseded diagnostics

The first 144-root validation generation is invalid because it combined an
incorrect canonical CE same-spin Wick estimator with overlapping replicate
rank streams. The short-lived `rdm2fix1` CE attempt still had overlapping
streams. Neither generation may be resumed, pooled, or cited as acceptance
evidence.

The final generation fixes both issues:

1. CE same-spin bond terms use CanEnsAFQMC's canonical two-body RDM `rho2`.
2. Every base seed is separated sufficiently that all `seed+pID` streams are
   disjoint.

An initial analysis of the final data also exposed an analyzer-only mistake:
inverse-variance weighting of two equal-work seed means let a noisy four-rank
SEM estimate move the central value. The final, unit-tested equal-work policy
above replaces that aggregation. No QMC row was discarded or replaced.

## Files

- `validation_summary.json`: machine-readable gate result.
- `cades_validation_audit.json`: Slurm, coverage, phase, provenance, and hash
  audit.
- `dtau2_extrapolation_vs_ed.tsv`: all 96 observable acceptance rows.
- `gce_density_dtau2_extrapolation.tsv`: all 12 density acceptance rows.
- `run_observables.tsv` and `gce_density_run_diagnostics.tsv`: per-run inputs
  to the extrapolations.
- `seed_combined_dtau.tsv` and `gce_density_seed_combined_dtau.tsv`: the
  equal-work seed-combined finite-step estimates.

All three implementation gates now pass. OBC production is permitted only
for the explicitly supported dense Hubbard paths, using fresh `_obc_` roots
and the exact dependency commits above; no PBC or v2.0.11 checkpoint may be
reused.
