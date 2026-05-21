#!/bin/bash
#SBATCH -A ccsd
#SBATCH -p high_mem
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -c 1
#SBATCH -J dqmc-ah
#SBATCH --mem=100G
#SBATCH -t 6:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cosdis123@gmail.com
#SBATCH --chdir=/home/9pm/nUHubbard
#SBATCH --output=/home/9pm/nUHubbard/logs/dqmc-ah-%A_%a.out
#SBATCH --error=/home/9pm/nUHubbard/logs/dqmc-ah-%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=peizhimai@gmail.com

# CADES Slurm launcher for the checkpointed attractive-U Hubbard SMOQyDQMC driver.
#
# Submit default single-run production template:
#   sbatch scripts/interacting_qmc_ed/submit_smoqydqmc_attractive_hubbard_cades.sbatch
#
# Override parameters without editing the file:
#   sbatch --export=ALL,SID=2,LX=12,LY=12,BETA=8.0,MU=-1.0 scripts/interacting_qmc_ed/submit_smoqydqmc_attractive_hubbard_cades.sbatch
#
# Submit a cheap smoke test:
#   sbatch --export=ALL,JOB_MODE=smoke scripts/interacting_qmc_ed/submit_smoqydqmc_attractive_hubbard_cades.sbatch

set -euo pipefail

PROJECT=${PROJECT:-/home/9pm/nUHubbard}
export PATH="$HOME/.juliaup/bin:$PATH"
export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export MKL_DYNAMIC=${MKL_DYNAMIC:-FALSE}

LOG_DIR=${LOG_DIR:-${PROJECT}/logs}
OUT_PARENT=${OUT_PARENT:-${PROJECT}/runs/dqmc_attractive_Um5}
mkdir -p "${LOG_DIR}" "${OUT_PARENT}"

JOB_MODE=${JOB_MODE:-production}

SID=${SID:-${SLURM_ARRAY_TASK_ID:-1}}
U=${U:--5.0}
TPRIME=${TPRIME:-0.0}
MU=${MU:--1.0}
# Keep the interaction convention fixed to
#   U * (n_up - 1/2) * (n_dn - 1/2)
# i.e. the checkpoint driver argument ph_sym_form=true.
PH_SYM_FORM=true
MEASUREMENT_PROFILE=${MEASUREMENT_PROFILE:-full}
DELTA_TAU=${DELTA_TAU:-0.1}
N_STAB=${N_STAB:-10}
N_STAB_MIN=${N_STAB_MIN:-6}
DG_MAX=${DG_MAX:-1e-5}
USE_REFLECTION_UPDATE=${USE_REFLECTION_UPDATE:-false}
UPDATE_STABILIZATION_FREQUENCY=${UPDATE_STABILIZATION_FREQUENCY:-true}

if [[ "${JOB_MODE}" == "smoke" ]]; then
  LX=${LX:-3}
  LY=${LY:-${LX}}
  BETA=${BETA:-2.0}
  N_THERM=${N_THERM:-20}
  N_MEASUREMENTS=${N_MEASUREMENTS:-40}
  N_BINS=${N_BINS:-4}
  N_UPDATES=${N_UPDATES:-1}
  CHECKPOINT_FREQ_HOURS=${CHECKPOINT_FREQ_HOURS:-0.1}
  RUNTIME_LIMIT_HOURS=${RUNTIME_LIMIT_HOURS:-0.25}
else
  LX=${LX:-8}
  LY=${LY:-${LX}}
  BETA=${BETA:-10.0}
  N_THERM=${N_THERM:-5000}
  N_MEASUREMENTS=${N_MEASUREMENTS:-50000}
  N_BINS=${N_BINS:-50}
  N_UPDATES=${N_UPDATES:-5}
  CHECKPOINT_FREQ_HOURS=${CHECKPOINT_FREQ_HOURS:-0.5}
  RUNTIME_LIMIT_HOURS=${RUNTIME_LIMIT_HOURS:-5.75}
fi

echo "[$(date)] Starting ${JOB_MODE} attractive-Hubbard SMOQyDQMC run"
echo "HOST=$(hostname)"
echo "PROJECT=${PROJECT}"
echo "OUT_PARENT=${OUT_PARENT}"
echo "SID=${SID} U=${U} TPRIME=${TPRIME} MU=${MU} LX=${LX} LY=${LY} BETA=${BETA}"
echo "N_THERM=${N_THERM} N_MEASUREMENTS=${N_MEASUREMENTS} N_BINS=${N_BINS} N_UPDATES=${N_UPDATES}"
echo "CHECKPOINT_FREQ_HOURS=${CHECKPOINT_FREQ_HOURS} RUNTIME_LIMIT_HOURS=${RUNTIME_LIMIT_HOURS}"
echo "PH_SYM_FORM=${PH_SYM_FORM}"
echo "MEASUREMENT_PROFILE=${MEASUREMENT_PROFILE}"
echo "DELTA_TAU=${DELTA_TAU}"
echo "N_STAB=${N_STAB} N_STAB_MIN=${N_STAB_MIN} DG_MAX=${DG_MAX}"
echo "USE_REFLECTION_UPDATE=${USE_REFLECTION_UPDATE} UPDATE_STABILIZATION_FREQUENCY=${UPDATE_STABILIZATION_FREQUENCY}"
echo "JULIA=$(command -v julia)"
echo "JULIA_NUM_THREADS=${JULIA_NUM_THREADS} OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} MKL_DYNAMIC=${MKL_DYNAMIC}"

srun -n "${SLURM_NTASKS:-1}" julia --project="${PROJECT}/julia_env" \
  "${PROJECT}/scripts/interacting_qmc_ed/run_smoqydqmc_attractive_hubbard_checkpoint.jl" \
  "${SID}" "${U}" "${TPRIME}" "${MU}" "${LX}" "${BETA}" \
  "${N_THERM}" "${N_MEASUREMENTS}" "${N_BINS}" "${N_UPDATES}" \
  "${CHECKPOINT_FREQ_HOURS}" "${RUNTIME_LIMIT_HOURS}" \
  "${PH_SYM_FORM}" "${OUT_PARENT}" "${LY}" "${MEASUREMENT_PROFILE}" "${DELTA_TAU}" "${N_STAB}" "${DG_MAX}" \
  "${USE_REFLECTION_UPDATE}" "${UPDATE_STABILIZATION_FREQUENCY}" "${N_STAB_MIN}"

echo "[$(date)] Finished"
