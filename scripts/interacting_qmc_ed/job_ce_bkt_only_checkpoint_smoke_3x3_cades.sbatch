#!/bin/bash
#SBATCH -A ccsd
#SBATCH -p burst
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH --chdir=/home/9pm/nUHubbard
#SBATCH --job-name=ce_bkt_chk_smoke
#SBATCH --output=/home/9pm/nUHubbard/logs/%x_%j.out
#SBATCH --error=/home/9pm/nUHubbard/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peizhimai@gmail.com

# CE-QMC checkpoint/resume smoke test.
# First leg intentionally exits through the checkpoint runtime-limit path;
# the wrapper then resubmits once and the resumed leg should finish and combine.

set -u -o pipefail

PROJECT=/home/9pm/nUHubbard
LOG_DIR=${PROJECT}/logs
mkdir -p "${LOG_DIR}"
cd "${PROJECT}"

OUTDIR="runs/ce_bkt_only_checkpoint_smoke_3x3_20260530"
RANKS=${SLURM_NTASKS:-4}
NWARMUPS=1
BATCH_NSAMPLES=1
MAX_BATCHES=2
MEASURE_INTERVAL=1
CHECKPOINT_FREQ_HOURS=0
RUNTIME_LIMIT_HOURS=0.0001
CHECKPOINT_EVERY_BATCHES=1
AUTO_RESUBMIT=true
MAX_RESUBMITS=${MAX_RESUBMITS:-4}
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

if success_outputs_present; then
  echo "[$(date -Is)] CE checkpoint smoke already complete: ${OUTDIR}"
  cat "${OUTDIR}/bkt_observables_qmc.tsv"
  exit 0
fi

rm -f "${OUTDIR}/checkpoint_stop.request" 2>/dev/null || true
mkdir -p "${OUTDIR}"

START_EPOCH=$(date +%s)
echo "[$(date -Is)] starting CE checkpoint smoke"
echo "host=$(hostname) job=${SLURM_JOB_ID:-manual} ranks=${RANKS} resubmit=${RESUBMIT_COUNT}/${MAX_RESUBMITS} outdir=${OUTDIR}"
echo "runtime_limit_hours=${RUNTIME_LIMIT_HOURS} checkpoint_freq_hours=${CHECKPOINT_FREQ_HOURS}"
echo "JULIA=$(command -v julia)"
echo "MPIEXECJL=$(command -v mpiexecjl || true)"

rc=0
mpiexecjl -n "${RANKS}" \
  julia --project="${PROJECT}/julia_env" \
  "${PROJECT}/scripts/interacting_qmc_ed/benchmark_ce_green_tau_space_mpi_3x3.jl" \
  --u=-5 \
  --dtau=0.1 \
  --beta=0.2 \
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
  --checkpoint-sync-timeout-seconds=120 \
  --checkpoint-sync-poll-seconds=1 \
  --seed=20260530 \
  --python=python3 \
  --output-dir="${OUTDIR}" || rc=$?

END_EPOCH=$(date +%s)
echo "[$(date -Is)] launcher exit code: ${rc} runtime_seconds=$((END_EPOCH - START_EPOCH))"

if success_outputs_present; then
  echo "[$(date -Is)] CE checkpoint smoke complete after resume/combine: ${OUTDIR}"
  cat "${OUTDIR}/bkt_observables_qmc.tsv"
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
    echo "[$(date -Is)] checkpoint smoke incomplete; resubmitting ${script_path} attempt ${next_count}/${MAX_RESUBMITS}"
    RESUBMIT_COUNT="${next_count}" sbatch "${script_path}"
    exit 0
  else
    echo "[$(date -Is)] checkpoint smoke incomplete and MAX_RESUBMITS reached: ${OUTDIR}" >&2
    exit 13
  fi
fi

echo "[$(date -Is)] CE checkpoint smoke failed/incomplete without valid checkpoint stop. rc=${rc}" >&2
exit "${rc}"
