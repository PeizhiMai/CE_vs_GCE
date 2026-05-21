#!/bin/bash
# Submit packed L=12 n=0.5 density-tuning jobs.
#
# This helper chunks the (T, mu) tuning grid into Slurm jobs where each job
# requests GROUPS_PER_JOB nodes and launches GROUPS_PER_JOB independent
# SmoQyDQMC groups concurrently:
#
#   one group = one node = 32 MPI ranks/chains for one (T, mu)
#
# Dry-run by default.  Example:
#
#   GROUPS_PER_JOB=5 DRY_RUN=0 \
#     scripts/interacting_qmc_ed/submit_density_tuning_L12_n05_pack_cades.sh

set -euo pipefail

PROJECT=${PROJECT:-/home/9pm/nUHubbard}
PACK_SCRIPT=${PACK_SCRIPT:-${PROJECT}/scripts/interacting_qmc_ed/run_density_tuning_L12_n05_pack_cades.sbatch}
OUT_PARENT=${OUT_PARENT:-${PROJECT}/runs/density_tuning_L12_n05}
LOG_DIR=${LOG_DIR:-${PROJECT}/logs}

L=${L:-12}
LY=${LY:-${L}}
U=${U:--5.0}
TPRIME=${TPRIME:-0.0}

T_LIST=${T_LIST:-"0.142 0.150 0.156 0.161 0.166"}
MU_LIST=${MU_LIST:-"-0.80 -0.70 -0.60 -0.50 -0.40"}

N_THERM=${N_THERM:-2000}
N_MEASUREMENTS=${N_MEASUREMENTS:-10000}
N_BINS=${N_BINS:-20}
N_UPDATES=${N_UPDATES:-2}
DELTA_TAU=${DELTA_TAU:-0.1}
N_STAB=${N_STAB:-10}
N_STAB_MIN=${N_STAB_MIN:-6}
DG_MAX=${DG_MAX:-1e-5}
USE_REFLECTION_UPDATE=${USE_REFLECTION_UPDATE:-false}
UPDATE_STABILIZATION_FREQUENCY=${UPDATE_STABILIZATION_FREQUENCY:-true}
CHECKPOINT_FREQ_HOURS=${CHECKPOINT_FREQ_HOURS:-0.5}
RUNTIME_LIMIT_HOURS=${RUNTIME_LIMIT_HOURS:-5.75}
TASKS_PER_GROUP=${TASKS_PER_GROUP:-32}
SID_BASE=${SID_BASE:-120500}
GROUPS_PER_JOB=${GROUPS_PER_JOB:-5}
AUTO_RESUBMIT=${AUTO_RESUBMIT:-true}
MAX_RESUBMITS=${MAX_RESUBMITS:-50}
DRY_RUN=${DRY_RUN:-1}

mkdir -p "${LOG_DIR}" "${OUT_PARENT}"

read -r -a T_ARRAY <<< "${T_LIST}"
read -r -a MU_ARRAY <<< "${MU_LIST}"
TOTAL_GROUPS=$((${#T_ARRAY[@]} * ${#MU_ARRAY[@]}))

echo "Packed density tuning submitter"
echo "T_LIST=${T_LIST}"
echo "MU_LIST=${MU_LIST}"
echo "TOTAL_GROUPS=${TOTAL_GROUPS}"
echo "GROUPS_PER_JOB=${GROUPS_PER_JOB}"
echo "TASKS_PER_GROUP=${TASKS_PER_GROUP}"
echo "DELTA_TAU=${DELTA_TAU} N_STAB=${N_STAB} N_STAB_MIN=${N_STAB_MIN} DG_MAX=${DG_MAX}"
echo "USE_REFLECTION_UPDATE=${USE_REFLECTION_UPDATE} UPDATE_STABILIZATION_FREQUENCY=${UPDATE_STABILIZATION_FREQUENCY}"
echo "AUTO_RESUBMIT=${AUTO_RESUBMIT} MAX_RESUBMITS=${MAX_RESUBMITS}"
echo "Each submitted job requests GROUP_COUNT nodes and GROUP_COUNT*${TASKS_PER_GROUP} tasks."
echo "DRY_RUN=${DRY_RUN}"

offset=0
job_idx=0
while (( offset < TOTAL_GROUPS )); do
  count=${GROUPS_PER_JOB}
  if (( offset + count > TOTAL_GROUPS )); then
    count=$((TOTAL_GROUPS - offset))
  fi
  ntasks=$((count * TASKS_PER_GROUP))
  EXPORT_VARS="ALL,PROJECT=${PROJECT},OUT_PARENT=${OUT_PARENT},LOG_DIR=${LOG_DIR}"
  EXPORT_VARS+=",L=${L},LY=${LY},U=${U},TPRIME=${TPRIME}"
  EXPORT_VARS+=",T_LIST=${T_LIST},MU_LIST=${MU_LIST}"
  EXPORT_VARS+=",N_THERM=${N_THERM},N_MEASUREMENTS=${N_MEASUREMENTS},N_BINS=${N_BINS},N_UPDATES=${N_UPDATES}"
  EXPORT_VARS+=",DELTA_TAU=${DELTA_TAU},N_STAB=${N_STAB},N_STAB_MIN=${N_STAB_MIN},DG_MAX=${DG_MAX},USE_REFLECTION_UPDATE=${USE_REFLECTION_UPDATE},UPDATE_STABILIZATION_FREQUENCY=${UPDATE_STABILIZATION_FREQUENCY},CHECKPOINT_FREQ_HOURS=${CHECKPOINT_FREQ_HOURS},RUNTIME_LIMIT_HOURS=${RUNTIME_LIMIT_HOURS}"
  EXPORT_VARS+=",TASKS_PER_GROUP=${TASKS_PER_GROUP},SID_BASE=${SID_BASE},GROUP_OFFSET=${offset},GROUP_COUNT=${count}"
  EXPORT_VARS+=",AUTO_RESUBMIT=${AUTO_RESUBMIT},MAX_RESUBMITS=${MAX_RESUBMITS},RESUBMIT_COUNT=0"
  CMD=(
    sbatch
    --nodes="${count}"
    --ntasks="${ntasks}"
    --ntasks-per-node="${TASKS_PER_GROUP}"
    --cpus-per-task=1
    --job-name="muTunePack_${job_idx}"
    --export="${EXPORT_VARS}"
    "${PACK_SCRIPT}"
  )
  printf '[pack %02d] offset=%d count=%d nodes=%d ntasks=%d: ' "${job_idx}" "${offset}" "${count}" "${count}" "${ntasks}"
  if [[ "${DRY_RUN}" == "0" ]]; then
    "${CMD[@]}"
  else
    printf '%q ' "${CMD[@]}"
    printf '\n'
  fi
  offset=$((offset + count))
  job_idx=$((job_idx + 1))
done
