# CADES checkpoint notes for canonical-ensemble QMC

CE-QMC now follows the same CADES wrapper pattern that worked for the DQMC runs:

- launch with `mpiexecjl`, not bare `srun`,
- one MPI rank is one independent Markov chain,
- each rank writes its own restart checkpoint,
- a nonzero checkpoint exit (`exit(13)`, often surfaced by `mpiexecjl` as `rc=1`) is treated as resumable only when all expected rank checkpoints exist,
- auto-resubmit uses `RESUBMIT_COUNT=... sbatch script` with no explicit `--export`, and
- the wrapper unsets site OpenMPI variables before calling `mpiexecjl`.

## Driver flags

Typical checkpointed CE-QMC invocation uses:

```bash
--checkpoint-enable=true \
--checkpoint-file=checkpoint.jls \
--checkpoint-every-batches=1 \
--checkpoint-freq-hours=0.5 \
--runtime-limit-hours=3.75 \
--checkpoint-keep=true
```

The MPI wrapper automatically adds:

```bash
--checkpoint-world-size=<nranks>
--checkpoint-root-dir=<absolute root output directory>
```

so ranks can wait for peer checkpoint sidecars before exiting through the
runtime-limit path.

## Per-rank files

For output root `runs/<run>/`, rank `r` writes

```text
runs/<run>/ranks/rank_0000r/checkpoint.jls
runs/<run>/ranks/rank_0000r/checkpoint.jls.status
```

The `.jls` payload stores the Markov-chain walker, Julia RNG state, completed
batch count, total sample count, and measurement accumulators.  The `.status`
sidecar is a small text file used by the Slurm wrapper and peer synchronization.

## CADES scripts

Current CE-QMC checkpoint scripts:

```text
scripts/interacting_qmc_ed/job_ce_bkt_only_checkpoint_smoke_3x3_cades.sbatch
scripts/interacting_qmc_ed/job_ce_bkt_only_3x3_um5_beta10_dtau01_mpi32_checkpoint_cades.sbatch
```

The production template is BKT/equal-time only and intentionally disables the
expensive one-particle unequal-time Green-function output:

```bash
--measure-greens=false \
--measure-bkt=true \
--measure-equal-time=true
```
