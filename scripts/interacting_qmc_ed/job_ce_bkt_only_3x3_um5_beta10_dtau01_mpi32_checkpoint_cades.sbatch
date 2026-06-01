#!/bin/bash
#SBATCH -A ccsd
#SBATCH -p burst
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --ntasks-per-node=32
#SBATCH --exclusive
#SBATCH -c 1
#SBATCH --mem=100G
#SBATCH -t 04:00:00
#SBATCH --chdir=/home/9pm/nUHubbard
#SBATCH --job-name=ce_bkt3x3_chk100k
#SBATCH --output=/home/9pm/nUHubbard/logs/%x_%j.out
#SBATCH --error=/home/9pm/nUHubbard/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peizhimai@gmail.com

# Checkpointed 32-rank CE-QMC production template for equal-time + BKT observables only.
# Unequal-time one-particle G is intentionally disabled (--measure-greens=false).

set -u -o pipefail

PROJECT=/home/9pm/nUHubbard
LOG_DIR=${PROJECT}/logs
mkdir -p "${LOG_DIR}"
cd "${PROJECT}"

OUTDIR="runs/ce_bkt_only_3x3_um5_beta10_dtau01_MPI32_CHK_Ntherm10000_Nmeas100000perrank_seed20260530"
ED_BKT_DIR="runs/full_ed_3x3_um5_beta10_dtau01_bkt"
COMPARE_BKT="runs/compare_ce_bkt_only_3x3_um5_dtau01_MPI32_CHK_100000perrank_vs_full_ed.tsv"
RANKS=${SLURM_NTASKS:-32}
NWARMUPS=10000
BATCH_NSAMPLES=1000
MAX_BATCHES=100
MEASURE_INTERVAL=2
CHECKPOINT_FREQ_HOURS=0.5
RUNTIME_LIMIT_HOURS=3.75
CHECKPOINT_EVERY_BATCHES=1
AUTO_RESUBMIT=true
MAX_RESUBMITS=${MAX_RESUBMITS:-10}
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}

unset LD_LIBRARY_PATH MPI_PATH MPI_ROOT MPICC MPICXX MPIF77 MPIF90 MPIFC
unset OMPI_MCA_pml OMPI_MCA_btl
export PATH="$HOME/.julia/bin:$HOME/.juliaup/bin:$PATH"
export JULIA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

success_outputs_present() {
  [[ -f "${OUTDIR}/bkt_observables_qmc.tsv" && -f "${OUTDIR}/equal_time_observables_qmc.tsv" ]]
}

run_bkt_compare() {
  if [[ -f "${OUTDIR}/bkt_observables_qmc.tsv" && -f "${ED_BKT_DIR}/bkt_observables_ed.tsv" ]]; then
    python3 "${PROJECT}/scripts/interacting_qmc_ed/compare_ce_bkt_observables_3x3.py" \
      --qmc-bkt "${OUTDIR}/bkt_observables_qmc.tsv" \
      --ed-bkt "${ED_BKT_DIR}/bkt_observables_ed.tsv" \
      --out "${COMPARE_BKT}" || true
    if [[ -f "${COMPARE_BKT}" ]]; then
      echo "=== BKT comparison summary ==="
      cat "${COMPARE_BKT}"
    fi
  fi
}

if success_outputs_present; then
  echo "[$(date -Is)] checkpointed CE BKT-only run already complete: ${OUTDIR}"
  run_bkt_compare
  exit 0
fi

mkdir -p "${OUTDIR}"

START_EPOCH=$(date +%s)
echo "[$(date -Is)] starting checkpointed MPI32 CE equal-time+BKT-only QMC"
echo "host=$(hostname) job=${SLURM_JOB_ID:-manual} ranks=${RANKS} resubmit=${RESUBMIT_COUNT}/${MAX_RESUBMITS} outdir=${OUTDIR}"
echo "settings: U=-5 beta=10 dtau=0.1 warmup/rank=${NWARMUPS} measurements/rank=$((BATCH_NSAMPLES * MAX_BATCHES)) chunks/rank=${MAX_BATCHES} chunk_size=${BATCH_NSAMPLES}"
echo "checkpoint: runtime_limit_hours=${RUNTIME_LIMIT_HOURS} checkpoint_freq_hours=${CHECKPOINT_FREQ_HOURS} keep=true"
echo "current_helper_sha256=$(sha256sum scripts/interacting_qmc_ed/ce_unequal_time_current_helpers.jl | awk '{print $1}')"
echo "JULIA=$(command -v julia)"
echo "MPIEXECJL=$(command -v mpiexecjl || true)"

julia --project="${PROJECT}/julia_env" \
  "${PROJECT}/scripts/interacting_qmc_ed/validate_ce_current_trace_formula_3x3.jl"

rc=0
mpiexecjl -n "${RANKS}" \
  julia --project="${PROJECT}/julia_env" \
  "${PROJECT}/scripts/interacting_qmc_ed/benchmark_ce_green_tau_space_mpi_3x3.jl" \
  --u=-5 \
  --dtau=0.1 \
  --beta=10 \
  --nwarmups="${NWARMUPS}" \
  --batch-nsamples="${BATCH_NSAMPLES}" \
  --measure-interval="${MEASURE_INTERVAL}" \
  --max-batches="${MAX_BATCHES}" \
  --num-fourier-points=10 \
  --nfreq=1 \
  --use-lowrank=false \
  --use-charge-hs=false \
  --sys-type=complex \
  --measure-greens=false \
  --measure-bkt=true \
  --measure-equal-time=true \
  --checkpoint-enable=true \
  --checkpoint-freq-hours="${CHECKPOINT_FREQ_HOURS}" \
  --checkpoint-every-batches="${CHECKPOINT_EVERY_BATCHES}" \
  --runtime-limit-hours="${RUNTIME_LIMIT_HOURS}" \
  --checkpoint-keep=true \
  --checkpoint-sync-timeout-seconds=600 \
  --checkpoint-sync-poll-seconds=5 \
  --seed=20260530 \
  --python=python3 \
  --output-dir="${OUTDIR}" || rc=$?

END_EPOCH=$(date +%s)
echo "[$(date -Is)] launcher exit code: ${rc} runtime_seconds=$((END_EPOCH - START_EPOCH))"

if success_outputs_present; then
  echo "[$(date -Is)] checkpointed CE BKT-only run complete: ${OUTDIR}"
  ls -lh "${OUTDIR}"
  run_bkt_compare
  exit 0
fi

checkpoint_count=0
if [[ -d "${OUTDIR}/ranks" ]]; then
  checkpoint_count=$(find "${OUTDIR}/ranks" -mindepth 2 -maxdepth 2 -name 'checkpoint.jls' | wc -l | tr -d ' ')
fi
echo "[$(date -Is)] checkpoint_count=${checkpoint_count}/${RANKS}"

checkpoint_stop=false
if [[ "${checkpoint_count}" -ge "${RANKS}" ]]; then
  if [[ "${rc}" == "13" || "${rc}" == "1" || "${rc}" == "0" ]]; then
    checkpoint_stop=true
  fi
fi

if [[ "${AUTO_RESUBMIT}" == "true" && "${checkpoint_stop}" == "true" ]]; then
  if (( RESUBMIT_COUNT < MAX_RESUBMITS )); then
    next_count=$((RESUBMIT_COUNT + 1))
    script_path=$(readlink -f "${BASH_SOURCE[0]}")
    echo "[$(date -Is)] checkpointed CE BKT-only run incomplete; resubmitting ${script_path} attempt ${next_count}/${MAX_RESUBMITS}"
    RESUBMIT_COUNT="${next_count}" sbatch "${script_path}"
    exit 0
  else
    echo "[$(date -Is)] checkpointed CE BKT-only run incomplete and MAX_RESUBMITS reached: ${OUTDIR}" >&2
    exit 13
  fi
fi

echo "[$(date -Is)] checkpointed CE BKT-only run failed/incomplete without valid checkpoint stop. rc=${rc}" >&2
exit "${rc}"
