#!/bin/bash
# Submit lightweight autocorrelation probes after density tuning.
#
# These jobs use MEASUREMENT_PROFILE=autocorr-current, which records global/local
# observables plus the integrated current-current measurement needed for rho_s,
# but skips expensive time-displaced Green's functions and pair/spin structure
# factors.  Use N_BINS=N_MEASUREMENTS so each stored bin is one measured sample.
#
# Example:
#   T_LIST="0.150" MU_LIST="-0.61" DRY_RUN=0 \
#     ./scripts/interacting_qmc_ed/submit_autocorr_probe_L12_n05_cades.sh
#
# Or use estimates from summarize_density_tuning.py:
#   T_MU_FILE=/home/9pm/nUHubbard/runs/density_tuning_L12_n05/mu_estimates_n0p5.tsv DRY_RUN=0 \
#     ./scripts/interacting_qmc_ed/submit_autocorr_probe_L12_n05_cades.sh

set -euo pipefail

PROJECT=${PROJECT:-/home/9pm/nUHubbard}
LAUNCHER=${LAUNCHER:-${PROJECT}/scripts/interacting_qmc_ed/submit_smoqydqmc_attractive_hubbard_cades.sbatch}
OUT_PARENT=${OUT_PARENT:-${PROJECT}/runs/autocorr_L12_n05}
LOG_DIR=${LOG_DIR:-${PROJECT}/logs}

L=${L:-12}
LY=${LY:-${L}}
U=${U:--5.0}
TPRIME=${TPRIME:-0.0}

T_LIST=${T_LIST:-"0.150"}
MU_LIST=${MU_LIST:-"-0.60"}
T_MU_FILE=${T_MU_FILE:-}

N_THERM=${N_THERM:-5000}
N_MEASUREMENTS=${N_MEASUREMENTS:-5000}
N_BINS=${N_BINS:-${N_MEASUREMENTS}}
N_UPDATES=${N_UPDATES:-1}
DELTA_TAU=${DELTA_TAU:-0.1}
CHECKPOINT_FREQ_HOURS=${CHECKPOINT_FREQ_HOURS:-0.5}
RUNTIME_LIMIT_HOURS=${RUNTIME_LIMIT_HOURS:-5.75}
NTASKS=${NTASKS:-16}
SID_BASE=${SID_BASE:-121500}
DRY_RUN=${DRY_RUN:-1}

mkdir -p "${LOG_DIR}" "${OUT_PARENT}"

if [[ -n "${T_MU_FILE}" ]]; then
  mapfile -t PAIRS < <(awk 'BEGIN{FS="\t"} NR>1 && ($6=="bracket" || $6=="exact") {print $1 " " $4}' "${T_MU_FILE}")
else
  read -r -a T_ARRAY <<< "${T_LIST}"
  read -r -a MU_ARRAY <<< "${MU_LIST}"
  if [[ ${#MU_ARRAY[@]} -eq 1 && ${#T_ARRAY[@]} -gt 1 ]]; then
    PAIRS=()
    for T in "${T_ARRAY[@]}"; do PAIRS+=("${T} ${MU_ARRAY[0]}"); done
  elif [[ ${#MU_ARRAY[@]} -eq ${#T_ARRAY[@]} ]]; then
    PAIRS=()
    for i in "${!T_ARRAY[@]}"; do PAIRS+=("${T_ARRAY[$i]} ${MU_ARRAY[$i]}"); done
  else
    echo "MU_LIST must contain one value or the same number of values as T_LIST" >&2
    exit 2
  fi
fi

echo "Autocorrelation probes, L=${L}x${LY}, U=${U}, t'=${TPRIME}"
echo "N_THERM=${N_THERM} N_MEASUREMENTS=${N_MEASUREMENTS} N_BINS=${N_BINS} N_UPDATES=${N_UPDATES}"
echo "DELTA_TAU=${DELTA_TAU}"
echo "NTASKS=${NTASKS} OUT_PARENT=${OUT_PARENT}"
echo "DRY_RUN=${DRY_RUN}"

idx=0
for pair in "${PAIRS[@]}"; do
  read -r T MU <<< "${pair}"
  BETA=$(python3.11 -c "print(f'{1.0/float(${T}):.8f}')")
  SID=$((SID_BASE + idx))
  JOB_NAME=$(printf "auto_L%s_T%s_mu%s" "${L}" "${T}" "${MU}" | tr '.-' 'pm')
  EXPORT_VARS="ALL,PROJECT=${PROJECT},OUT_PARENT=${OUT_PARENT},LOG_DIR=${LOG_DIR},JOB_MODE=production"
  EXPORT_VARS+=",SID=${SID},U=${U},TPRIME=${TPRIME},MU=${MU},LX=${L},LY=${LY},BETA=${BETA}"
  EXPORT_VARS+=",N_THERM=${N_THERM},N_MEASUREMENTS=${N_MEASUREMENTS},N_BINS=${N_BINS},N_UPDATES=${N_UPDATES}"
  EXPORT_VARS+=",CHECKPOINT_FREQ_HOURS=${CHECKPOINT_FREQ_HOURS},RUNTIME_LIMIT_HOURS=${RUNTIME_LIMIT_HOURS},MEASUREMENT_PROFILE=autocorr-current,DELTA_TAU=${DELTA_TAU}"
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
