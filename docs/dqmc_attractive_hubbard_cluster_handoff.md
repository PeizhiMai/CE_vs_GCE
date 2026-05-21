# Cluster handoff: SmoQyDQMC attractive-Hubbard production runs with superfluid density

This note captures the current DQMC setup in this repository so a fresh thread/window can continue the cluster setup without rediscovering the local context.

Repository root used locally:

```text
/Users/cosdis/Desktop/projects/CE_GCE
```

Main calculation: 2D attractive Hubbard model on an `Lx × Ly` periodic square lattice, with integrated current-current measurements for the finite-size superfluid density.

---

## 1. Files to use

### Production/checkpoint DQMC driver

```text
scripts/interacting_qmc_ed/run_smoqydqmc_attractive_hubbard_checkpoint.jl
```

Use this on the cluster. It supports checkpoint/restart and MPI.

### Non-checkpoint/local test driver

```text
scripts/interacting_qmc_ed/run_smoqydqmc_attractive_hubbard.jl
```

Use this only for short local smoke tests.

### Superfluid-density postprocessor

```text
scripts/interacting_qmc_ed/compute_smoqydqmc_superfluid_density.py
```

This reads SmoQyDQMC `stats_pID-*.h5` first, not rounded CSV, and writes:

```text
superfluid_density.tsv
```

### Background implementation note

```text
docs/dqmc_superfluid_density_implementation.md
```

---

## 2. Model/measurement currently implemented

Hamiltonian conventions are controlled by `ph_sym_form`:

- `ph_sym_form = true`:

  \[
  H = K + U\sum_i (n_{i\uparrow}-1/2)(n_{i\downarrow}-1/2) - \mu \sum_{i\sigma} n_{i\sigma}.
  \]

- `ph_sym_form = false`:

  \[
  H = K + U\sum_i n_{i\uparrow}n_{i\downarrow} - \mu \sum_{i\sigma} n_{i\sigma}.
  \]

The DQMC driver uses a density-channel Hirsch HST for `U < 0`.

Nearest-neighbor hopping IDs in the driver:

| HOPPING_ID | bond |
|---:|---|
| 1 | `+x` NN hopping |
| 2 | `+y` NN hopping |
| 3 | `+x + y` NNN hopping, amplitude `t'` |
| 4 | `+x - y` NNN hopping, amplitude `t'` |

For the current benchmark and intended production, use `t = 1`, `t' = 0`, `U = -5` unless the production scan changes these.

---

## 3. Superfluid-density measurement

The driver has the needed current-current measurement:

```julia
initialize_correlation_measurements!(
    measurement_container = measurement_container,
    model_geometry = model_geometry,
    correlation = "current",
    time_displaced = false,
    integrated = true,
    pairs = [(1, 1)]
)
```

Important: `integrated = true` is the key option. It outputs the imaginary-time-integrated static response

\[
\Lambda_{xx}(\mathbf q,0)
= \frac{1}{N}\int_0^\beta d\tau\,
\langle J_x(\mathbf q,\tau)J_x(-\mathbf q,0)\rangle.
\]

So we do **not** need to store the full unequal-time correlator for superfluid density. Full `time_displaced = true` current data would be needed for DC-conductivity midpoint estimates, but not for \(\rho_s\).

The finite-size estimator used is

\[
\rho_s(L,T)=\frac14\left[\Lambda_{xx}(q_{\min},0,0)-\Lambda_{xx}(0,q_{\min},0)\right],
\quad q_{\min}=2\pi/L.
\]

The postprocessor also reports the diamagnetic estimator

\[
\rho_s^{\rm dia}=\frac14\left[\frac{-K_x}{N}-\Lambda_{xx}(0,q_{\min},0)\right].
\]

Output file from postprocessing:

```text
<run-datafolder>/superfluid_density.tsv
```

---

## 4. Driver command-line arguments

Checkpoint driver positional arguments:

```text
1  sID              integer simulation ID; use a fixed ID for restartable production
2  U                Hubbard interaction, e.g. -5.0
3  tprime           next-nearest-neighbor hopping, e.g. 0.0
4  mu               chemical potential
5  L                Lx
6  beta             inverse temperature
7  N_therm          number of thermalization sweeps
8  N_measurements   number of measurements
9  N_bins           number of bins; N_measurements must be divisible by N_bins
10 N_updates        updates/sweeps between measurements
11 checkpoint_freq  hours between checkpoint writes, e.g. 0.5 or 1.0
12 runtime_limit    hours before clean checkpoint/exit, e.g. 3.75 for a 4h job
13 ph_sym_form      true or false
14 filepath         output parent directory
15 Ly               optional Ly; defaults to Lx if omitted
```

`N_updates` means the number of full update sweeps between measurements. Total measurement-stage update sweeps are roughly `N_measurements × N_updates`.

Recommended for 4-hour walltime jobs:

- scheduler walltime: `04:00:00`
- `runtime_limit`: `3.7` to `3.8` hours
- `checkpoint_freq`: `0.5` to `1.0` hours
- use a fixed `sID`, not `0`, for production/requeue/restart jobs

The same command with the same `sID`, parameters, and output path resumes an incomplete checkpointed simulation.

---

## 5. Example cluster command

Example for a square `L × L` run with `ph_sym_form=true`:

```bash
julia --project=/path/to/CE_GCE/julia_env \
  /path/to/CE_GCE/scripts/interacting_qmc_ed/run_smoqydqmc_attractive_hubbard_checkpoint.jl \
  1 \
  -5.0 0.0 -1.0 \
  8 10.0 \
  5000 50000 50 5 \
  0.5 3.75 \
  true \
  /path/to/scratch/dqmc_attractive_Um5 \
  8
```

MPI/Slurm style:

```bash
srun -n ${SLURM_NTASKS:-1} julia --project=/path/to/CE_GCE/julia_env \
  /path/to/CE_GCE/scripts/interacting_qmc_ed/run_smoqydqmc_attractive_hubbard_checkpoint.jl \
  ${SID} \
  ${U} ${TPRIME} ${MU} \
  ${LX} ${BETA} \
  ${N_THERM} ${N_MEASUREMENTS} ${N_BINS} ${N_UPDATES} \
  ${CHECKPOINT_FREQ_HOURS} ${RUNTIME_LIMIT_HOURS} \
  ${PH_SYM_FORM} \
  ${OUT_PARENT} \
  ${LY}
```

Minimal Slurm skeleton:

```bash
#!/bin/bash
#SBATCH --job-name=dqmc-ah
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/dqmc-%A_%a.out
#SBATCH --error=logs/dqmc-%A_%a.err

set -euo pipefail

PROJECT=/path/to/CE_GCE
OUT_PARENT=/path/to/scratch/dqmc_attractive_Um5
mkdir -p logs "${OUT_PARENT}"

SID=${SLURM_ARRAY_TASK_ID:-1}
U=-5.0
TPRIME=0.0
MU=-1.0
LX=8
LY=8
BETA=10.0
N_THERM=5000
N_MEASUREMENTS=50000
N_BINS=50
N_UPDATES=5
CHECKPOINT_FREQ_HOURS=0.5
RUNTIME_LIMIT_HOURS=3.75
PH_SYM_FORM=true

srun -n ${SLURM_NTASKS:-1} julia --project=${PROJECT}/julia_env \
  ${PROJECT}/scripts/interacting_qmc_ed/run_smoqydqmc_attractive_hubbard_checkpoint.jl \
  ${SID} ${U} ${TPRIME} ${MU} ${LX} ${BETA} \
  ${N_THERM} ${N_MEASUREMENTS} ${N_BINS} ${N_UPDATES} \
  ${CHECKPOINT_FREQ_HOURS} ${RUNTIME_LIMIT_HOURS} \
  ${PH_SYM_FORM} ${OUT_PARENT} ${LY}
```

Before first cluster run, instantiate/precompile the Julia environment:

```bash
julia --project=/path/to/CE_GCE/julia_env -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

Depending on the cluster MPI setup, `MPI.jl` may need local configuration. Start with `srun -n 1` smoke tests first.

---

## 6. Output layout and postprocessing

During a checkpointed run, the data folder is under:

```text
<OUT_PARENT>/attractive_hubbard_rect_U-5.00_tp0.00_mu-1.00_Lx<LX>_Ly<LY>_b<BETA>-<sID>
```

After successful completion, the checkpoint driver renames it to begin with:

```text
complete_attractive_hubbard_rect_...
```

Expected important files:

```text
stats_pID-0.h5
local_stats_pID-0.csv
global_stats_pID-0.csv
simulation_info_sID-<sID>_pID-0.toml
integrated/current/current_momentum_integrated_stats_pID-0.csv
```

Run the superfluid-density postprocessor after completion:

```bash
python /path/to/CE_GCE/scripts/interacting_qmc_ed/compute_smoqydqmc_superfluid_density.py \
  /path/to/the/run-datafolder
```

Locally, use this Python if working in the Codex desktop environment:

```bash
~/.venvs/myenv/bin/python scripts/interacting_qmc_ed/compute_smoqydqmc_superfluid_density.py <run-datafolder>
```

The postprocessor writes:

```text
<run-datafolder>/superfluid_density.tsv
```

Columns include:

```text
lambda_longitudinal_qmin0
lambda_transverse_0qmin
rho_s_current
diamagnetic_minus_Kx_per_site
rho_s_diamagnetic
```

---

## 7. Energy and basic observable reconstruction

Do **not** use any old/artificial corrected energy quantity. Use SmoQy local measurements directly.

For `t'=0`, energy per site from DQMC CSV/HDF5 is

```text
E/site = hopping_energy(HOPPING_ID=1)
       + hopping_energy(HOPPING_ID=2)
       + hubbard_energy(HUBBARD_ID=1)
       + onsite_energy(ORBITAL_ID=1)
```

`HOPPING_ID=3,4` are zero when `t'=0`; include them if `t' != 0`.

Useful global observables:

```text
global_stats_pID-0.csv: density, double_occ, compressibility, sgn
local_stats_pID-0.csv: hopping_energy, hubbard_energy, onsite_energy
simulation_info_sID-*_pID-0.toml: dG, n_stab_final, acceptance rates, seed
```

Sanity checks:

- `dG` should stay below `dG_max`.
- `sgn` should be near 1 for attractive Hubbard.
- acceptance rates should not be pathological.
- For superfluid checks, compare `lambda_longitudinal_qmin0` with `-Kx/N`; they should be close, with finite-size/statistical differences.

---

## 8. Known caveats

1. **Avoid Lx or Ly = 2 for production benchmarks.**  We found that the SmoQy periodic-bond convention for size 2 does not duplicate the two opposite periodic bonds the same way a standard tight-binding ED convention would. For `Lx,Ly >= 3`, this issue is absent.

2. **`ph_sym_form=false` can have ergodicity problems near full/empty filling.**  At `U=-5`, `mu=-1`, `beta=10`, the conventional-form 3×3 system is almost fully filled. DQMC stayed essentially saturated with tiny current response. This is a physics/ergodicity warning for that regime, not a failure of the `ph_sym_form=true` production setup.

3. **Current measurement here is for superfluid density, not regular DC conductivity.**  For DC conductivity midpoint estimates, add a full time-displaced current measurement and use \(\Lambda_{xx}(q=0,\tau=\beta/2)\). The current setup only stores the integrated static response needed for \(\rho_s\).

---

## 9. Local validation already completed

3×3 PBC benchmark, parameters:

```text
Lx=3, Ly=3, t=1, t'=0, U=-5, mu=-1, beta=10
DQMC: N_therm=5000, N_measurements=50000, N_bins=50, N_updates=5, Delta_tau=0.05
```

### `ph_sym_form=true`

| observable | ED | DQMC |
|---|---:|---:|
| E/site | -1.480655701 | -1.481315128 ± 0.001043527 |
| density | 0.222439518 | 0.222427252 ± 0.000204957 |
| double occ/site | 0.041511463 | 0.041979337 ± 0.000130297 |
| compressibility | 0.004341456 | 0.004102076 ± 0.004100877 |
| Lambda_L | 0.400818349 | 0.397511055 ± 0.000843052 |
| Lambda_T | 0.001214041 | 0.001555032 ± 0.000537116 |
| rho_s current | 0.099901077 | 0.098989006 ± 0.000249904 |
| -Kx/N | 0.400818349 | 0.400029166 ± 0.000639622 |
| rho_s diamagnetic | 0.099901077 | 0.099618533 ± 0.000208808 |

### `ph_sym_form=false`

| observable | ED | DQMC |
|---|---:|---:|
| E/site | -2.999988235 | -2.9999999996 ± 0.0000000123 |
| density | 1.999974348 | 1.9999999993 ± 0.0000000022 |
| double occ/site | 0.999984146 | 0.9999999993 ± 0.0000000022 |
| compressibility | 0.000512989 | 0.0000000071 ± 0.0000000220 |
| Lambda_L | 0.000020926 | 0.00000000094 ± 0.00000000418 |
| Lambda_T | 0.000006180 | -0.0000000157 ± 0.0000000254 |
| rho_s current | 0.000003686 | 0.00000000417 ± 0.00000000645 |
| -Kx/N | 0.000020926 | 0.00000000036 ± 0.00000000380 |
| rho_s diamagnetic | 0.000003686 | 0.00000000402 ± 0.00000000643 |

Benchmark summary file:

```text
results/interacting_qmc_ed/benchmark_3x3_t1_tp0_Um5_mu_m1_beta10_ed_dqmc.tsv
```

---

## 10. Suggested new-thread prompt

Paste this into a fresh thread if setting up on the cluster:

```text
We are setting up production SmoQyDQMC runs for the attractive Hubbard model in the CE_GCE repo. Read docs/dqmc_attractive_hubbard_cluster_handoff.md first. Use scripts/interacting_qmc_ed/run_smoqydqmc_attractive_hubbard_checkpoint.jl as the production driver and scripts/interacting_qmc_ed/compute_smoqydqmc_superfluid_density.py for postprocessing rho_s. The target observable is superfluid density from integrated current-current correlations, not DC conductivity. Use Lx,Ly >= 3, checkpoint/restart under 4-hour jobs, and preserve fixed sID/output path for restarts.
```
