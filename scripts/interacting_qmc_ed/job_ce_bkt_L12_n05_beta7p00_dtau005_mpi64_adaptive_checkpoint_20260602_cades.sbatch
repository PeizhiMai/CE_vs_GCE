#!/bin/bash
#SBATCH -A ccsd
#SBATCH -p burst
#SBATCH -N 2
#SBATCH -n 64
#SBATCH --ntasks-per-node=32
#SBATCH --exclusive
#SBATCH -c 1
#SBATCH --mem=100G
#SBATCH -t 04:00:00
#SBATCH --chdir=/home/9pm/nUHubbard
#SBATCH --job-name=ceL12n05b7p00
#SBATCH --output=/home/9pm/nUHubbard/logs/%x_%j.out
#SBATCH --error=/home/9pm/nUHubbard/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peizhimai@gmail.com

# Checkpointed adaptive canonical-ensemble QMC BKT production point.
# L=12x12, U=-5, density n=0.5 -> Nup=Ndn=36, beta=7.00 (T=0.142857142857).
# One MPI rank = one independent Markov chain. Measurements are per rank.
# This is BKT/equal-time only; unequal-time one-particle G is intentionally disabled.

set -u -o pipefail

PROJECT=/home/9pm/nUHubbard
LOG_DIR=${PROJECT}/logs
mkdir -p "${LOG_DIR}"
cd "${PROJECT}"

OUTDIR="runs/ce_bkt_L12x12_Um5_n05_beta7p00_dtau0p05_MPI64_ADAPTIVE_Ntherm5000_Nmeas50000perrank_Nupd3_seed2026060905"
RANKS=${SLURM_NTASKS:-64}
LX=12
LY=12
NUP=36
NDN=36
U=-5
BETA=7.00
DTAU=0.05
NWARMUPS=5000
BATCH_NSAMPLES=1
MAX_BATCHES=50000
MEASURE_INTERVAL=3
BASE_SEED=2026060905
CHECKPOINT_FREQ_HOURS=0.50
RUNTIME_LIMIT_HOURS=3.75
CHECKPOINT_EVERY_BATCHES=1
CHECKPOINT_WARMUP_CHUNK=1
CLUSTER_SIZE=144
NUM_FOURIER_POINTS=145
AUTO_RESUBMIT=true
MAX_RESUBMITS=${MAX_RESUBMITS:-1000}
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
  [[ -f "${OUTDIR}/checkpoint_complete.txt" && -f "${OUTDIR}/bkt_observables_qmc.tsv" && -f "${OUTDIR}/equal_time_observables_qmc.tsv" ]]
}

if success_outputs_present; then
  echo "[$(date -Is)] CE-QMC run already complete: ${OUTDIR}"
  echo "=== bkt_observables_qmc.tsv ==="
  cat "${OUTDIR}/bkt_observables_qmc.tsv"
  echo "=== equal_time_observables_qmc.tsv ==="
  cat "${OUTDIR}/equal_time_observables_qmc.tsv"
  exit 0
fi

mkdir -p "${OUTDIR}"

START_EPOCH=$(date +%s)
echo "[$(date -Is)] starting checkpointed adaptive CE-QMC equal-time+BKT-only production run"
echo "host=$(hostname) job=${SLURM_JOB_ID:-manual} nodes=${SLURM_JOB_NUM_NODES:-2} ranks=${RANKS} resubmit=${RESUBMIT_COUNT}/${MAX_RESUBMITS} outdir=${OUTDIR}"
echo "settings: Lx=${LX} Ly=${LY} U=${U} beta=${BETA} T=0.142857142857 dtau=${DTAU} Ltau=140 Nup=${NUP} Ndn=${NDN} warmup/rank=${NWARMUPS} measurements/rank=$((BATCH_NSAMPLES * MAX_BATCHES)) nupdate=${MEASURE_INTERVAL} batches=${MAX_BATCHES} batch_size=${BATCH_NSAMPLES}"
echo "checkpoint: slurm_walltime=04:00:00 runtime_limit_hours=${RUNTIME_LIMIT_HOURS} checkpoint_freq_hours=${CHECKPOINT_FREQ_HOURS} keep=true auto_resubmit=${AUTO_RESUBMIT}"
echo "optimization: cluster_size=${CLUSTER_SIZE} num_fourier_points=${NUM_FOURIER_POINTS} bkt_current_estimator=propagated adaptive_refresh=true refresh_tol=1e-6"
echo "current_helper_sha256=$(sha256sum scripts/interacting_qmc_ed/ce_unequal_time_current_helpers.jl | awk '{print $1}')"
echo "driver_sha256=$(sha256sum scripts/interacting_qmc_ed/benchmark_ce_green_matsubara_3x3.jl | awk '{print $1}')"
echo "mpi_wrapper_sha256=$(sha256sum scripts/interacting_qmc_ed/benchmark_ce_green_tau_space_mpi_3x3.jl | awk '{print $1}')"
echo "JULIA=$(command -v julia)"
echo "MPIEXECJL=$(command -v mpiexecjl || true)"

rc=0
mpiexecjl -n "${RANKS}"   julia --project="${PROJECT}/julia_env"   "${PROJECT}/scripts/interacting_qmc_ed/benchmark_ce_green_tau_space_mpi_3x3.jl"   --lx="${LX}"   --ly="${LY}"   --nup="${NUP}"   --ndn="${NDN}"   --u="${U}"   --dtau="${DTAU}"   --beta="${BETA}"   --nwarmups="${NWARMUPS}"   --batch-nsamples="${BATCH_NSAMPLES}"   --measure-interval="${MEASURE_INTERVAL}"   --max-batches="${MAX_BATCHES}"   --cluster-size="${CLUSTER_SIZE}"   --num-fourier-points="${NUM_FOURIER_POINTS}"   --nfreq=1   --use-lowrank=false   --use-charge-hs=false   --sys-type=complex   --measure-greens=false   --measure-bkt=true   --measure-equal-time=true   --bkt-current-estimator=propagated   --bkt-adaptive-refresh=true   --bkt-refresh-interval=10   --bkt-refresh-tol=1e-6   --bkt-refresh-min=1   --bkt-refresh-max=20   --bkt-refresh-growth-patience=3   --checkpoint-enable=true   --checkpoint-file=checkpoint.jls   --checkpoint-freq-hours="${CHECKPOINT_FREQ_HOURS}"   --checkpoint-every-batches="${CHECKPOINT_EVERY_BATCHES}"   --checkpoint-warmup-chunk="${CHECKPOINT_WARMUP_CHUNK}"   --runtime-limit-hours="${RUNTIME_LIMIT_HOURS}"   --checkpoint-keep=true   --checkpoint-sync-timeout-seconds=900   --checkpoint-sync-poll-seconds=5   --seed="${BASE_SEED}"   --python=python3   --output-dir="${OUTDIR}" || rc=$?

END_EPOCH=$(date +%s)
echo "[$(date -Is)] launcher exit code: ${rc} runtime_seconds=$((END_EPOCH - START_EPOCH))"
echo "runtime_minutes=$(awk -v s=$((END_EPOCH - START_EPOCH)) 'BEGIN{printf "%.3f", s/60.0}')"

if success_outputs_present; then
  echo "[$(date -Is)] CE-QMC run complete: ${OUTDIR}"
  ls -lh "${OUTDIR}"
  echo "=== bkt_observables_qmc.tsv ==="
  cat "${OUTDIR}/bkt_observables_qmc.tsv"
  echo "=== equal_time_observables_qmc.tsv ==="
  cat "${OUTDIR}/equal_time_observables_qmc.tsv"
  exit 0
fi

checkpoint_count=0
status_count=0
if [[ -d "${OUTDIR}/ranks" ]]; then
  checkpoint_count=$(find "${OUTDIR}/ranks" -mindepth 2 -maxdepth 2 -name 'checkpoint.jls' | wc -l | tr -d ' ')
  status_count=$(find "${OUTDIR}/ranks" -mindepth 2 -maxdepth 2 -name 'checkpoint.jls.status' | wc -l | tr -d ' ')
fi
echo "[$(date -Is)] checkpoint_count=${checkpoint_count}/${RANKS} status_count=${status_count}/${RANKS}"

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
    echo "[$(date -Is)] CE-QMC run incomplete; resubmitting ${script_path} attempt ${next_count}/${MAX_RESUBMITS}"
    RESUBMIT_COUNT="${next_count}" sbatch "${script_path}"
    exit 0
  else
    echo "[$(date -Is)] CE-QMC run incomplete and MAX_RESUBMITS reached: ${OUTDIR}" >&2
    exit 13
  fi
fi

echo "[$(date -Is)] CE-QMC run failed/incomplete without valid checkpoint stop. rc=${rc}" >&2
exit "${rc}"
