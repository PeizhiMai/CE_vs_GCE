#!/bin/bash
#SBATCH -A ccsd
#SBATCH -p burst
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --ntasks-per-node=32
#SBATCH -c 1
#SBATCH -J auto0142m0p684
#SBATCH --mem=100G
#SBATCH -t 4:00:00
#SBATCH --chdir=/home/9pm/nUHubbard
#SBATCH --output=/home/9pm/nUHubbard/logs/auto0142m0p684-%j.out
#SBATCH --error=/home/9pm/nUHubbard/logs/auto0142m0p684-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peizhimai@gmail.com

# Autocorrelation probe at the stage-2 interpolated n=0.5 chemical potential.
#
# One node = 32 MPI ranks = 32 independent Markov chains with rank-dependent
# seeds.  We set N_BINS=N_MEASUREMENTS so each stored bin is one measured
# configuration; analyze_smoqydqmc_autocorr.jl then estimates tau_int in update
# sweep units.

set -u -o pipefail

PROJECT=/home/9pm/nUHubbard
OUT_PARENT=${PROJECT}/runs/autocorr_L12_n05_T0142
LOG_DIR=${PROJECT}/logs
mkdir -p "${OUT_PARENT}" "${LOG_DIR}"

SID=122000
BASE_SEED=$((88200000 + SID))
U=-5.0
TPRIME=0.0
MU=-0.684072277424
LX=12
LY=12
T_TARGET=0.142
DELTA_TAU=0.1
LTAU=$(python3.11 -c "import sys; print(round(1.0/float(sys.argv[1])/float(sys.argv[2])))" "${T_TARGET}" "${DELTA_TAU}")
BETA=$(python3.11 -c "import sys; print(f'{int(sys.argv[1]) * float(sys.argv[2]):.8f}')" "${LTAU}" "${DELTA_TAU}")
T_EFF=$(python3.11 -c "import sys; print(f'{1.0/float(sys.argv[1]):.8f}')" "${BETA}")

N_THERM=5000
N_MEASUREMENTS=5000
N_BINS=5000
N_UPDATES=1
PH_SYM_FORM=true
MEASUREMENT_PROFILE=autocorr-current

N_STAB=10
N_STAB_MIN=6
DG_MAX=1e-5
USE_REFLECTION_UPDATE=false
UPDATE_STABILIZATION_FREQUENCY=true

CHECKPOINT_FREQ_HOURS=0.5
RUNTIME_LIMIT_HOURS=3.75
AUTO_RESUBMIT=true
MAX_RESUBMITS=${MAX_RESUBMITS:-10}
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

run_base=$(python3.11 - "$SID" "$U" "$TPRIME" "$MU" "$LX" "$LY" "$BETA" <<'PY2'
import sys
sid, u, tp, mu, lx, ly, beta = sys.argv[1:]
print(f"attractive_hubbard_rect_U{float(u):.2f}_tp{float(tp):.2f}_mu{float(mu):.2f}_Lx{int(lx)}_Ly{int(ly)}_b{float(beta):.2f}-{int(sid)}")
PY2
)
complete_dir="${OUT_PARENT}/complete_${run_base}"
incomplete_dir="${OUT_PARENT}/${run_base}"

run_analysis() {
  local dir="$1"
  echo "[$(date)] Running autocorrelation analysis in ${dir}"
  julia --project="${PROJECT}/julia_env" \
    "${PROJECT}/scripts/interacting_qmc_ed/analyze_smoqydqmc_autocorr.jl" \
    "${dir}" --observable rho_s_current --max-lag 1000 \
    --out "${dir}/autocorr_rho_s_current.tsv"
  julia --project="${PROJECT}/julia_env" \
    "${PROJECT}/scripts/interacting_qmc_ed/analyze_smoqydqmc_autocorr.jl" \
    "${dir}" --observable density --max-lag 1000 \
    --out "${dir}/autocorr_density.tsv"
  julia --project="${PROJECT}/julia_env" \
    "${PROJECT}/scripts/interacting_qmc_ed/analyze_smoqydqmc_autocorr.jl" \
    "${dir}" --observable minus_kx --max-lag 1000 \
    --out "${dir}/autocorr_minus_kx.tsv"
}

if [[ -d "${complete_dir}" ]]; then
  echo "[$(date)] Complete run already exists: ${complete_dir}"
  run_analysis "${complete_dir}"
  exit 0
fi

echo "[$(date)] Starting autocorrelation probe"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-manual} RESUBMIT_COUNT=${RESUBMIT_COUNT}/${MAX_RESUBMITS}"
echo "SID=${SID} BASE_SEED=${BASE_SEED} T_TARGET=${T_TARGET} T_EFF=${T_EFF} BETA=${BETA} LTAU=${LTAU} MU=${MU} LX=${LX} LY=${LY} U=${U}"
echo "OUT_PARENT=${OUT_PARENT}"
echo "N_THERM=${N_THERM} N_MEASUREMENTS=${N_MEASUREMENTS} N_BINS=${N_BINS} N_UPDATES=${N_UPDATES} PROFILE=${MEASUREMENT_PROFILE}"
echo "DELTA_TAU=${DELTA_TAU} N_STAB=${N_STAB} N_STAB_MIN=${N_STAB_MIN} DG_MAX=${DG_MAX}"
echo "USE_REFLECTION_UPDATE=${USE_REFLECTION_UPDATE} UPDATE_STABILIZATION_FREQUENCY=${UPDATE_STABILIZATION_FREQUENCY}"
echo "RUNTIME_LIMIT_HOURS=${RUNTIME_LIMIT_HOURS} CHECKPOINT_FREQ_HOURS=${CHECKPOINT_FREQ_HOURS}"
echo "JULIA=$(command -v julia)"
echo "MPIEXECJL=$(command -v mpiexecjl || true)"

mpiexecjl -n "${SLURM_NTASKS:-32}" \
  julia --project="${PROJECT}/julia_env" \
  "${PROJECT}/scripts/interacting_qmc_ed/run_smoqydqmc_attractive_hubbard_checkpoint.jl" \
  "${SID}" "${U}" "${TPRIME}" "${MU}" "${LX}" "${BETA}" \
  "${N_THERM}" "${N_MEASUREMENTS}" "${N_BINS}" "${N_UPDATES}" \
  "${CHECKPOINT_FREQ_HOURS}" "${RUNTIME_LIMIT_HOURS}" \
  "${PH_SYM_FORM}" "${OUT_PARENT}" "${LY}" "${MEASUREMENT_PROFILE}" \
  "${DELTA_TAU}" "${N_STAB}" "${DG_MAX}" \
  "${USE_REFLECTION_UPDATE}" "${UPDATE_STABILIZATION_FREQUENCY}" "${N_STAB_MIN}" \
  "${BASE_SEED}"
rc=$?

echo "[$(date)] launcher exit code: ${rc}"

if [[ -d "${complete_dir}" ]]; then
  echo "[$(date)] Complete: ${complete_dir}"
  run_analysis "${complete_dir}"
  exit 0
fi

checkpoint_count=0
if [[ -d "${incomplete_dir}" ]]; then
  checkpoint_count=$(find "${incomplete_dir}" -maxdepth 1 -name 'checkpoint_pID-*.jld2' | wc -l | tr -d ' ')
fi
expected_checkpoints=${SLURM_NTASKS:-32}
echo "[$(date)] checkpoint_count=${checkpoint_count}/${expected_checkpoints}"

checkpoint_stop=false
if [[ "${checkpoint_count}" -ge "${expected_checkpoints}" ]]; then
  if [[ "${rc}" == "13" || "${rc}" == "1" || "${rc}" == "0" ]]; then
    checkpoint_stop=true
  fi
fi

if [[ "${AUTO_RESUBMIT}" == "true" && "${checkpoint_stop}" == "true" ]]; then
  if (( RESUBMIT_COUNT < MAX_RESUBMITS )); then
    next_count=$((RESUBMIT_COUNT + 1))
    script_path=$(readlink -f "${BASH_SOURCE[0]}")
    echo "[$(date)] Checkpoint written/incomplete; resubmitting ${script_path} attempt ${next_count}/${MAX_RESUBMITS}"
    RESUBMIT_COUNT="${next_count}" sbatch "${script_path}"
    exit 0
  else
    echo "[$(date)] Incomplete and MAX_RESUBMITS reached. Incomplete dir: ${incomplete_dir}" >&2
    exit 13
  fi
fi

echo "[$(date)] Autocorrelation probe failed or incomplete without checkpoint-resubmit. rc=${rc} incomplete_dir=${incomplete_dir}" >&2
exit "${rc}"
