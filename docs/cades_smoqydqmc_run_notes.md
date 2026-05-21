# CADES notes for SmoQyDQMC attractive-Hubbard runs

These notes record the CADES-specific issues and fixes found while setting up
the `n=0.5`, `L=12`, attractive Hubbard benchmark in `/home/9pm/nUHubbard`.

## 1. Launch Julia MPI with `mpiexecjl`, not bare `srun`

On CADES, the Julia `MPI.jl` environment in this project uses the Julia MPI
wrapper.  Launching with

```bash
srun -n 32 julia --project="${PROJECT}/julia_env" ...
```

appeared to allocate 32 Slurm tasks, but each Julia process behaved like an
independent single-rank MPI job:

```text
pID = 0
MPI size = 1
```

This caused all tasks to write the same output paths, for example
`bins/pID-0/bin-19.h5`, leading to HDF5 file-creation collisions and invalid
`complete_*` folders containing only `pID-0` output.

Use `mpiexecjl` instead:

```bash
export PATH="$HOME/.julia/bin:$HOME/.juliaup/bin:$PATH"

mpiexecjl -n "${SLURM_NTASKS:-32}" \
  julia --project="${PROJECT}/julia_env" \
  "${PROJECT}/scripts/interacting_qmc_ed/run_smoqydqmc_attractive_hubbard_checkpoint.jl" \
  "${SID}" "${U}" "${TPRIME}" "${MU}" "${LX}" "${BETA}" \
  "${N_THERM}" "${N_MEASUREMENTS}" "${N_BINS}" "${N_UPDATES}" \
  "${CHECKPOINT_FREQ_HOURS}" "${RUNTIME_LIMIT_HOURS}" \
  "${PH_SYM_FORM}" "${OUT_PARENT}" "${LY}" "${MEASUREMENT_PROFILE}" \
  "${DELTA_TAU}" "${N_STAB}" "${DG_MAX}" \
  "${USE_REFLECTION_UPDATE}" "${UPDATE_STABILIZATION_FREQUENCY}" "${N_STAB_MIN}"
```

For one node with 32 Slurm tasks, this gives 32 MPI ranks, i.e. 32 independent
Markov chains for the same parameter point.

## 2. How to verify MPI launched correctly

After a valid 32-rank run finishes, the complete data folder should contain
one simulation-info file per MPI rank:

```bash
cd /home/9pm/nUHubbard

for d in runs/density_tuning_L12_n05/complete_attractive_hubbard_rect_*; do
  echo -n "$(basename "$d") "
  find "$d" -maxdepth 1 -name "simulation_info_sID-*_pID-*.toml" | wc -l
done
```

Expected count for 32 MPI ranks:

```text
32
```

If the count is `1` and only `pID-0` files exist, the run was not a real MPI
run and should be discarded.

## 3. Checkpointing and auto-resubmission on CADES

Use the dedicated smoke test before relying on checkpoint/resume for production:

```bash
cd /home/9pm/nUHubbard
sbatch scripts/interacting_qmc_ed/job_checkpoint_smoke_L12_n05_burst_cades.sbatch
```

The smoke test intentionally:

1. writes 32 per-rank JLD2 checkpoint files,
2. exits through the runtime-limit/checkpoint path,
3. auto-resubmits once,
4. resumes from those checkpoint files,
5. completes and renames the output folder to `complete_*`.

Successful reference test on CADES:

```text
5379922  chkptSmokeL12  COMPLETED  first leg wrote 32 checkpoints and resubmitted
5379924  chkptSmokeL12  COMPLETED  resumed from checkpoint and completed
```

Verify:

```bash
find runs/checkpoint_smoke_L12_n05/complete_*129900 \
  -maxdepth 1 -name "simulation_info_sID-*_pID-*.toml" | wc -l
```

Expected:

```text
32
```

### Important: do not resubmit with `sbatch --export=...`

On CADES, explicit Slurm export options such as

```bash
sbatch --export=ALL,RESUBMIT_COUNT=1 script.sbatch
sbatch --export=RESUBMIT_COUNT=1 script.sbatch
```

caused Slurm to load the site user environment, including system OpenMPI
variables such as:

```text
LD_LIBRARY_PATH=/sw/cades-open/.../openmpi-4.1.6.../lib:...
OMPI_MCA_pml=ucx
OMPI_MCA_btl=^vader,openib,tcp
MPI_ROOT=/sw/cades-open/.../openmpi-4.1.6...
```

Those conflict with Julia's `OpenMPI_jll`/`mpiexecjl` stack and produced
MPI initialization failures:

```text
mca_base_framework_open on ompi_pml failed
An error occurred in MPI_Init_thread
ProcessExited(14)
```

The working auto-resubmit pattern is:

```bash
RESUBMIT_COUNT="${next_count}" sbatch "${script_path}"
```

with no `--export` option.  The job scripts also defensively unset the
site-OpenMPI variables before calling `mpiexecjl`:

```bash
unset LD_LIBRARY_PATH MPI_PATH MPI_ROOT MPICC MPICXX MPIF77 MPIF90 MPIFC
unset OMPI_MCA_pml OMPI_MCA_btl
```

### `mpiexecjl` exit code caveat

When the Julia driver exits with `exit(13)` at a checkpoint/runtime limit,
`mpiexecjl` may return shell `rc=1` while printing `ProcessExited(13)` in the
`.err` log.  Therefore, wrapper scripts should treat `rc=1` as a checkpoint
stop only if all expected per-rank checkpoint files are present:

```bash
checkpoint_count=$(find "${incomplete_dir}" -maxdepth 1 \
  -name 'checkpoint_pID-*.jld2' | wc -l | tr -d ' ')
```

For a 32-rank run, require `checkpoint_count >= 32` before auto-resubmitting.

For short density-tuning or smoke-test jobs that do not need restart, it is
still fine to disable checkpointing:

```bash
CHECKPOINT_FREQ_HOURS=0
AUTO_RESUBMIT=false
```

This avoids JLD2 checkpoint writes altogether.

## 4. Use commensurate `beta` and `dtau`

SmoQyDQMC requires:

```text
beta = L_tau * dtau
```

Do not pass an arbitrary `beta = 1/T` if it is not commensurate with `dtau`.
For example, with target `T/t = 0.142` and `dtau = 0.1`:

```text
exact beta = 1 / 0.142 = 7.04225352
L_tau      = round(beta / dtau) = 70
used beta  = 70 * 0.1 = 7.0
T_eff      = 1 / 7.0 = 0.14285714
```

The job scripts compute this rounded, commensurate beta before launching.

## 5. Lightweight density-tuning settings

For a quick MPI smoke/coarse density test at `T/t ≈ 0.142`, use the lighter
density-only settings:

```bash
N_THERM=100
N_MEASUREMENTS=200
N_BINS=4
N_UPDATES=2
MEASUREMENT_PROFILE=density-only
CHECKPOINT_FREQ_HOURS=0
```

For production statistics, increase `N_THERM`, `N_MEASUREMENTS`, and `N_BINS`
after the MPI launch and output are verified.

## 6. MKL and threading

The Julia driver loads MKL:

```julia
using MKL
```

For MPI-only runs, keep one CPU thread per rank:

```bash
export JULIA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
```

The log should show `libmkl_rt.so` in the BLAS/LAPACK configuration.

## 7. Useful Slurm/log checks

Check current jobs:

```bash
squeue -u 9pm -o "%.18i %.9P %.24j %.8u %.2t %.10M %.6D %R"
```

Check completed job states:

```bash
sacct -j <jobids> --format=JobID,JobName%24,Partition,State,ExitCode,Elapsed,NNodes,NodeList%20
```

Inspect logs:

```bash
cd /home/9pm/nUHubbard
tail -f logs/mu0142m0p40-<jobid>.out
tail -f logs/mu0142m0p40-<jobid>.err
```

Look for HDF5 collisions, segmentation faults, or bus errors:

```bash
grep -nE "ERROR|HDF5|Segmentation|signal 11|Bus error|srun: error" logs/*.err
```
