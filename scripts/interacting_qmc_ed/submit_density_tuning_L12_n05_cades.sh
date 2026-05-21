#!/bin/bash
# Submit short L=12 attractive-Hubbard density-tuning jobs on CADES.
#
# Goal: tune chemical potential for <n>=0.5 before full rho_s production.
# This helper submits one Slurm job per (T/t, mu) point using the main CADES
# checkpoint launcher.  It is dry-run by default; set DRY_RUN=0 to submit.
#
# Default benchmark temperatures are read from the L=12 points in the target
# figure.  Override, for example:
#   T_LIST="0.142 0.150" MU_LIST="-0.75 -0.65 -0.55" DRY_RUN=0 \
#     ./scripts/interacting_qmc_ed/submit_density_tuning_L12_n05_cades.sh

set -euo pipefail

PROJECT=${PROJECT:-/home/9pm/nUHubbard}
LAUNCHER=${LAUNCHER:-${PROJECT}/scripts/interacting_qmc_ed/submit_smoqydqmc_attractive_hubbard_cades.sbatch}
OUT_PARENT=${OUT_PARENT:-${PROJECT}/runs/density_tuning_L12_n05}
LOG_DIR=${LOG_DIR:-${PROJECT}/logs}

L=${L:-12}
LY=${LY:-${L}}
U=${U:--5.0}
TPRIME=${TPRIME:-0.0}
TARGET_DENSITY=${TARGET_DENSITY:-0.5}

# L=12 temperatures from the benchmark figure, in units of t.  We omit the two
# hottest points for the first n=0.5 benchmark pass.
T_LIST=${T_LIST:-"0.142 0.150 0.156 0.161 0.166"}

# Coarse first bracket for the particle-hole-symmetric chemical potential.
# For U=-5 and ph_sym_form=true, half filling is MU=0; n=0.5 is at negative MU.
MU_LIST=${MU_LIST:-"-0.80 -0.70 -0.60 -0.50 -0.40"}

# Short tuning statistics.  Final production should be larger.
N_THERM=${N_THERM:-2000}
N_MEASUREMENTS=${N_MEASUREMENTS:-10000}
N_BINS=${N_BINS:-20}
N_UPDATES=${N_UPDATES:-2}
DELTA_TAU=${DELTA_TAU:-0.1}
CHECKPOINT_FREQ_HOURS=${CHECKPOINT_FREQ_HOURS:-0.5}
RUNTIME_LIMIT_HOURS=${RUNTIME_LIMIT_HOURS:-5.75}
NTASKS=${NTASKS:-32}
SID_BASE=${SID_BASE:-120500}
DRY_RUN=${DRY_RUN:-1}

mkdir -p "${LOG_DIR}" "${OUT_PARENT}"

read -r -a T_ARRAY <<< "${T_LIST}"
read -r -a MU_ARRAY <<< "${MU_LIST}"

echo "Density tuning for target n=${TARGET_DENSITY}, L=${L}x${LY}, U=${U}, t'=${TPRIME}"
echo "T_LIST=${T_LIST}"
echo "MU_LIST=${MU_LIST}"
echo "N_THERM=${N_THERM} N_MEASUREMENTS=${N_MEASUREMENTS} N_BINS=${N_BINS} N_UPDATES=${N_UPDATES}"
echo "DELTA_TAU=${DELTA_TAU}"
echo "NTASKS=${NTASKS} OUT_PARENT=${OUT_PARENT}"
echo "DRY_RUN=${DRY_RUN}"

idx=0
for i in "${!T_ARRAY[@]}"; do
  T=${T_ARRAY[$i]}
  BETA=$(python3.11 -c "print(f'{1.0/float(${T}):.8f}')")
  for j in "${!MU_ARRAY[@]}"; do
    MU=${MU_ARRAY[$j]}
    SID=$((SID_BASE + 100 * i + j))
    JOB_NAME=$(printf "muTune_L%s_T%s_mu%s" "${L}" "${T}" "${MU}" | tr '.-' 'pm')
    EXPORT_VARS="ALL,PROJECT=${PROJECT},OUT_PARENT=${OUT_PARENT},LOG_DIR=${LOG_DIR},JOB_MODE=production"
    EXPORT_VARS+=",SID=${SID},U=${U},TPRIME=${TPRIME},MU=${MU},LX=${L},LY=${LY},BETA=${BETA}"
    EXPORT_VARS+=",N_THERM=${N_THERM},N_MEASUREMENTS=${N_MEASUREMENTS},N_BINS=${N_BINS},N_UPDATES=${N_UPDATES}"
    EXPORT_VARS+=",CHECKPOINT_FREQ_HOURS=${CHECKPOINT_FREQ_HOURS},RUNTIME_LIMIT_HOURS=${RUNTIME_LIMIT_HOURS},MEASUREMENT_PROFILE=full,DELTA_TAU=${DELTA_TAU}"
    CMD=(
      sbatch
      --job-name="${JOB_NAME}"
      --ntasks="${NTASKS}"
      --cpus-per-task=1
      --export="${EXPORT_VARS}"
      "${LAUNCHER}"
    )
    printf '[%02d] T=%s beta=%s mu=%s SID=%s: ' "${idx}" "${T}" "${BETA}" "${MU}" "${SID}"
    if [[ "${DRY_RUN}" == "0" ]]; then
      "${CMD[@]}"
    else
      printf '%q ' "${CMD[@]}"
      printf '\n'
    fi
    idx=$((idx + 1))
  done
done
