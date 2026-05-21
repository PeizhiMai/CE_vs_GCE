#!/bin/bash
#SBATCH -A ccsd
#SBATCH -p high_mem
#SBATCH -J muTunePack
#SBATCH -t 6:00:00
#SBATCH --mem=100G
#SBATCH --chdir=/home/9pm/nUHubbard
#SBATCH --output=/home/9pm/nUHubbard/logs/muTunePack-%A.out
#SBATCH --error=/home/9pm/nUHubbard/logs/muTunePack-%A.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=peizhimai@gmail.com

# Packed CADES density-tuning launcher.
#
# One node is one parameter group:
#   1 node -> 32 MPI ranks -> 32 independent Markov chains at one (T, mu)
#
# Submit this directly only if you also pass --nodes/--ntasks large enough.
# Usually use submit_density_tuning_L12_n05_pack_cades.sh.

set -euo pipefail

PROJECT=${PROJECT:-/home/9pm/nUHubbard}
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
GROUP_OFFSET=${GROUP_OFFSET:-0}
GROUP_COUNT=${GROUP_COUNT:-1}
AUTO_RESUBMIT=${AUTO_RESUBMIT:-false}
MAX_RESUBMITS=${MAX_RESUBMITS:-50}
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}

# Avoid CADES user-env/system-OpenMPI contamination when using Julia OpenMPI_jll.
unset LD_LIBRARY_PATH MPI_PATH MPI_ROOT MPICC MPICXX MPIF77 MPIF90 MPIFC
unset OMPI_MCA_pml OMPI_MCA_btl

export PATH="$HOME/.julia/bin:$HOME/.juliaup/bin:$PATH"
export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export MKL_DYNAMIC=${MKL_DYNAMIC:-FALSE}

mkdir -p "${LOG_DIR}" "${OUT_PARENT}"

read -r -a T_ARRAY <<< "${T_LIST}"
read -r -a MU_ARRAY <<< "${MU_LIST}"
N_T=${#T_ARRAY[@]}
N_MU=${#MU_ARRAY[@]}
TOTAL_GROUPS=$((N_T * N_MU))

echo "[$(date)] Starting packed density tuning"
echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none} SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-unknown}"
echo "PROJECT=${PROJECT}"
echo "OUT_PARENT=${OUT_PARENT}"
echo "T_LIST=${T_LIST}"
echo "MU_LIST=${MU_LIST}"
echo "GROUP_OFFSET=${GROUP_OFFSET} GROUP_COUNT=${GROUP_COUNT} TOTAL_GROUPS=${TOTAL_GROUPS}"
echo "AUTO_RESUBMIT=${AUTO_RESUBMIT} RESUBMIT_COUNT=${RESUBMIT_COUNT} MAX_RESUBMITS=${MAX_RESUBMITS}"
echo "TASKS_PER_GROUP=${TASKS_PER_GROUP}"
echo "L=${L} LY=${LY} U=${U} TPRIME=${TPRIME}"
echo "N_THERM=${N_THERM} N_MEASUREMENTS=${N_MEASUREMENTS} N_BINS=${N_BINS} N_UPDATES=${N_UPDATES} DELTA_TAU=${DELTA_TAU}"
echo "N_STAB=${N_STAB} N_STAB_MIN=${N_STAB_MIN} DG_MAX=${DG_MAX}"
echo "USE_REFLECTION_UPDATE=${USE_REFLECTION_UPDATE} UPDATE_STABILIZATION_FREQUENCY=${UPDATE_STABILIZATION_FREQUENCY}"
echo "JULIA=$(command -v julia)"
echo "JULIA_NUM_THREADS=${JULIA_NUM_THREADS} OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} MKL_DYNAMIC=${MKL_DYNAMIC}"

if (( GROUP_OFFSET < 0 || GROUP_COUNT < 1 || GROUP_OFFSET >= TOTAL_GROUPS )); then
  echo "Invalid GROUP_OFFSET/GROUP_COUNT for TOTAL_GROUPS=${TOTAL_GROUPS}" >&2
  exit 2
fi

last=$((GROUP_OFFSET + GROUP_COUNT))
if (( last > TOTAL_GROUPS )); then
  last=${TOTAL_GROUPS}
fi

running=0
declare -a pids=()
declare -a labels=()
declare -a group_ids=()

run_folder_names() {
  local sid=$1
  local u=$2
  local tprime=$3
  local mu=$4
  local lx=$5
  local ly=$6
  local beta=$7
  python3.11 - "$sid" "$u" "$tprime" "$mu" "$lx" "$ly" "$beta" <<'PY'
import sys
sid, u, tp, mu, lx, ly, beta = sys.argv[1:]
base = f"attractive_hubbard_rect_U{float(u):.2f}_tp{float(tp):.2f}_mu{float(mu):.2f}_Lx{int(lx)}_Ly{int(ly)}_b{float(beta):.2f}-{int(sid)}"
print(base)
print("complete_" + base)
PY
}

group_complete() {
  local sid=$1
  local u=$2
  local tprime=$3
  local mu=$4
  local lx=$5
  local ly=$6
  local beta=$7
  local names base complete
  names=$(run_folder_names "$sid" "$u" "$tprime" "$mu" "$lx" "$ly" "$beta")
  base=$(printf '%s\n' "$names" | sed -n '1p')
  complete=$(printf '%s\n' "$names" | sed -n '2p')
  [[ -d "${OUT_PARENT}/${complete}" ]]
}

for ((g = GROUP_OFFSET; g < last; g++)); do
  ti=$((g / N_MU))
  mi=$((g % N_MU))
  T=${T_ARRAY[$ti]}
  MU=${MU_ARRAY[$mi]}
  BETA=$(python3.11 -c "print(f'{1.0/float(${T}):.8f}')")
  SID=$((SID_BASE + 100 * ti + mi))
  label=$(printf "g%03d_L%s_T%s_mu%s_sid%s" "${g}" "${L}" "${T}" "${MU}" "${SID}" | tr '.-' 'pm')
  stdout="${LOG_DIR}/muTunePack-${SLURM_JOB_ID:-manual}-${label}.out"
  stderr="${LOG_DIR}/muTunePack-${SLURM_JOB_ID:-manual}-${label}.err"

  if group_complete "${SID}" "${U}" "${TPRIME}" "${MU}" "${L}" "${LY}" "${BETA}"; then
    echo "[$(date)] skipping complete group ${g}: T=${T} beta=${BETA} mu=${MU} SID=${SID}"
    continue
  fi

  echo "[$(date)] launching group ${g}: T=${T} beta=${BETA} mu=${MU} SID=${SID}"
  srun --exclusive -N1 -n "${TASKS_PER_GROUP}" --cpus-per-task=1 \
    julia --project="${PROJECT}/julia_env" \
      "${PROJECT}/scripts/interacting_qmc_ed/run_smoqydqmc_attractive_hubbard_checkpoint.jl" \
      "${SID}" "${U}" "${TPRIME}" "${MU}" "${L}" "${BETA}" \
      "${N_THERM}" "${N_MEASUREMENTS}" "${N_BINS}" "${N_UPDATES}" \
      "${CHECKPOINT_FREQ_HOURS}" "${RUNTIME_LIMIT_HOURS}" \
      true "${OUT_PARENT}" "${LY}" full "${DELTA_TAU}" "${N_STAB}" "${DG_MAX}" \
      "${USE_REFLECTION_UPDATE}" "${UPDATE_STABILIZATION_FREQUENCY}" "${N_STAB_MIN}" \
      >"${stdout}" 2>"${stderr}" &
  pids+=("$!")
  labels+=("${label}")
  group_ids+=("${g}")
  running=$((running + 1))
done

rc=0
saw_runtime_limit=false
saw_hard_failure=false
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[$(date)] completed ${labels[$i]}"
  else
    step_rc=$?
    echo "[$(date)] FAILED ${labels[$i]} rc=${step_rc}" >&2
    rc=${step_rc}
    if [[ "${step_rc}" == "13" ]]; then
      saw_runtime_limit=true
    else
      saw_hard_failure=true
    fi
  fi
done

incomplete=0
for ((g = GROUP_OFFSET; g < last; g++)); do
  ti=$((g / N_MU))
  mi=$((g % N_MU))
  T=${T_ARRAY[$ti]}
  MU=${MU_ARRAY[$mi]}
  BETA=$(python3.11 -c "print(f'{1.0/float(${T}):.8f}')")
  SID=$((SID_BASE + 100 * ti + mi))
  if ! group_complete "${SID}" "${U}" "${TPRIME}" "${MU}" "${L}" "${LY}" "${BETA}"; then
    echo "[$(date)] incomplete group remains: g=${g} T=${T} beta=${BETA} mu=${MU} SID=${SID}"
    incomplete=$((incomplete + 1))
  fi
done

if [[ "${AUTO_RESUBMIT}" == "true" && "${incomplete}" -gt 0 && "${saw_hard_failure}" == "false" ]]; then
  if (( RESUBMIT_COUNT < MAX_RESUBMITS )); then
    next_count=$((RESUBMIT_COUNT + 1))
    ntasks=$((GROUP_COUNT * TASKS_PER_GROUP))
    export_vars="ALL,PROJECT=${PROJECT},OUT_PARENT=${OUT_PARENT},LOG_DIR=${LOG_DIR}"
    export_vars+=",L=${L},LY=${LY},U=${U},TPRIME=${TPRIME},T_LIST=${T_LIST},MU_LIST=${MU_LIST}"
    export_vars+=",N_THERM=${N_THERM},N_MEASUREMENTS=${N_MEASUREMENTS},N_BINS=${N_BINS},N_UPDATES=${N_UPDATES}"
    export_vars+=",DELTA_TAU=${DELTA_TAU},N_STAB=${N_STAB},N_STAB_MIN=${N_STAB_MIN},DG_MAX=${DG_MAX}"
    export_vars+=",USE_REFLECTION_UPDATE=${USE_REFLECTION_UPDATE},UPDATE_STABILIZATION_FREQUENCY=${UPDATE_STABILIZATION_FREQUENCY}"
    export_vars+=",CHECKPOINT_FREQ_HOURS=${CHECKPOINT_FREQ_HOURS},RUNTIME_LIMIT_HOURS=${RUNTIME_LIMIT_HOURS}"
    export_vars+=",TASKS_PER_GROUP=${TASKS_PER_GROUP},SID_BASE=${SID_BASE},GROUP_OFFSET=${GROUP_OFFSET},GROUP_COUNT=${GROUP_COUNT}"
    export_vars+=",AUTO_RESUBMIT=${AUTO_RESUBMIT},MAX_RESUBMITS=${MAX_RESUBMITS},RESUBMIT_COUNT=${next_count}"
    echo "[$(date)] Auto-resubmitting ${incomplete} incomplete group(s), attempt ${next_count}/${MAX_RESUBMITS}"
    sbatch \
      --nodes="${GROUP_COUNT}" \
      --ntasks="${ntasks}" \
      --ntasks-per-node="${TASKS_PER_GROUP}" \
      --cpus-per-task=1 \
      --job-name="${SLURM_JOB_NAME:-muTunePack}_r${next_count}" \
      --export="${export_vars}" \
      "${PROJECT}/scripts/interacting_qmc_ed/run_density_tuning_L12_n05_pack_cades.sbatch"
    # Treat an intentional checkpoint/runtime-limit exit as successful once the
    # continuation job is queued, so Slurm does not report this job as failed.
    rc=0
  else
    echo "[$(date)] Auto-resubmit limit reached; ${incomplete} incomplete group(s) remain." >&2
    rc=${rc:-13}
    [[ "${rc}" == "0" ]] && rc=13
  fi
fi

echo "[$(date)] Packed density tuning finished with rc=${rc}, incomplete=${incomplete}"
exit "${rc}"
