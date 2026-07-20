# CE/GCE OBC validation workflow

This directory implements the `2x2`/`3x3` open-boundary validation gate. It is
not an OBC production workflow. Production remains forbidden until the ED,
checkpoint, rank-coverage, and three-sigma extrapolation gates all pass.

## Physics matrix

- Square OBC, `t=1`, `t'=0`, x-fastest site ordering.
- `U = {-3,+3}`.
- `2x2`: CE sectors `(1,1)` and `(2,2)`, `beta={2,5}`.
- `3x3`: CE sectors `(2,2)` and `(4,4)`, `beta=5`.
- GCE chemical potentials are tuned by ED to the matching CE total particle
  number with absolute error at most `0.01`.
- `dtau={0.20,0.10,0.05}` and two independent base seeds.
- Default QMC validation statistics are 4 MPI ranks, 2,000 warmups, and 10,000
  measurements per rank with interval 3. Each small-cluster array task uses
  one node and 20 GiB on `ccsd/burst/default`.

The four acceptance observables are kinetic energy/site, double
occupancy/site, NN spin `(n_up-n_dn)_i(n_up-n_dn)_j`, and NN connected charge.
Every bond estimator is averaged over existing undirected physical bonds.
For CE, the same-spin pair term is evaluated with CanEnsAFQMC's canonical
two-body RDM. A Wick contraction of the already number-projected one-body RDM
is mathematically invalid and is rejected by estimator provenance checks.

## ED oracle

Run from the project root with the required Python environment:

```bash
~/.venvs/myenv/bin/python scripts/interacting_qmc_ed/obc_hubbard_ed.py \
  --outdir results/interacting_qmc_ed/obc_ed_validation_20260720 \
  --cache-dir results/interacting_qmc_ed/obc_ed_spectrum_cache_v3_20260720 \
  --sizes 2 3 --interactions -3 3 --boltzmann-tolerance 1e-8 \
  --eigen-residual-tolerance 1e-8 \
  --dense-threshold 1800 --initial-k 96 --max-k 2048
```

`2x2` uses complete many-body thermal ED. Large `3x3` sectors are resolved in
all four C4 rotation sectors before Lanczos truncation. Each omitted subspace
has a conservative dimension times lower-energy Boltzmann bound; no reference
is accepted above `1e-8`, and no reference with a maximum eigenpair residual
above `1e-8` is accepted. The `U=0` gate separately checks exact CE and GCE
one-body traces.

## Manifests and CADES jobs

First run the two-rank CADES smoke gate from the isolated OBC worktree:

```bash
mkdir -p /home/9pm/nUHubbard_obc_dev/logs \
         /home/9pm/nUHubbard_obc_runs/obc_ce_gce_validation_20260720
sbatch scripts/interacting_qmc_ed/obc_ce_gce_validation_20260720/job_obc_smoke_cades.sbatch
```

It verifies `OpenMPI_jll`, a real two-rank MPI world, the shared `2x2` OBC
Hamiltonian, CE and attractive-GCE checkpoint/resume without counter resets,
a fresh positive-U spin-HS GCE run, rank/site coverage, and all four primary
tables. It writes `smoke_complete.txt` only after every check passes.

After that gate passes, generate the full validation manifests:

```bash
~/.venvs/myenv/bin/python \
  scripts/interacting_qmc_ed/obc_ce_gce_validation_20260720/generate_validation_manifests.py \
  --ed-dir docs/validation/obc_ed_validation_20260720 \
  --outdir scripts/interacting_qmc_ed/obc_ce_gce_validation_20260720/manifests
```

This creates 72 unique CE roots and 72 unique GCE roots. The two Slurm files
are unthrottled arrays pinned to `ccsd/burst/default`. Their runners reject a
wrong rank count, a root without `_obc_`, accumulator resets, or dependency
provenance changes. Healthy runtime-limit checkpoints may continue only the
same manifest row and run root. Continuations deliberately use environment
prefix assignments rather than `sbatch --export`, avoiding the CADES site
OpenMPI environment that conflicts with Julia's MPI.jl runtime.

The default source worktree is `/home/9pm/nUHubbard_obc_dev`; validation data
go under `/home/9pm/nUHubbard_obc_runs`. Neither default touches the frozen
production checkout or its checkpoint roots.

Submit only after the local and CADES smoke gates pass:

```bash
sbatch scripts/interacting_qmc_ed/obc_ce_gce_validation_20260720/job_ce_validation_cades.sbatch
sbatch scripts/interacting_qmc_ed/obc_ce_gce_validation_20260720/job_gce_validation_cades.sbatch
```

The first CE validation generation exposed the projected-1-RDM Wick error.
Those CE roots are retained as failed diagnostics and are never resumed or
accepted. The fresh corrective CE generation is
`manifests_ce_rdm2_fix1/ce_validation_manifest.tsv`, submitted with:

```bash
sbatch scripts/interacting_qmc_ed/obc_ce_gce_validation_20260720/job_ce_validation_rdm2_fix1_cades.sbatch
```

## Acceptance analysis

```bash
~/.venvs/myenv/bin/python \
  scripts/interacting_qmc_ed/obc_ce_gce_validation_20260720/analyze_validation.py \
  --ce-manifest scripts/interacting_qmc_ed/obc_ce_gce_validation_20260720/manifests_ce_rdm2_fix1/ce_validation_manifest.tsv \
  --gce-manifest scripts/interacting_qmc_ed/obc_ce_gce_validation_20260720/manifests/gce_validation_manifest.tsv \
  --outdir results/interacting_qmc_ed/obc_ce_gce_validation_analysis_20260720 \
  --require-complete
```

The analyzer requires complete rank/site-accumulator coverage and all four
primary tables, combines the two seed replicates, performs weighted linear
fits in `dtau^2`, and requires every zero-step intercept to agree with ED
within three combined standard errors. It also fails on numerical-zero phase,
dependency drift, or hidden translational/unequal-time output.

Finite-time-step GCE densities are retained rather than rejected point by
point: an ED-tuned chemical potential can have an expected `O(dtau^2)` density
shift. The analyzer pools rank signed-numerator/phase-denominator accumulators,
fits achieved `N` versus `dtau^2`, and requires the zero-step result to satisfy
the ED tuning tolerance or three-sigma uncertainty, whichever is wider.
