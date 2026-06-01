#!/bin/bash
#SBATCH -A ccsd
#SBATCH -p burst
#SBATCH -N 2
#SBATCH -n 64
#SBATCH --ntasks-per-node=32
#SBATCH --exclusive
#SBATCH -c 1
#SBATCH --mem=100G
#SBATCH -t 01:00:00
#SBATCH --chdir=/home/9pm/nUHubbard
#SBATCH --job-name=ce_L12b6_chk3k
#SBATCH --output=/home/9pm/nUHubbard/logs/%x_%j.out
#SBATCH --error=/home/9pm/nUHubbard/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peizhimai@gmail.com

# Checkpointed canonical-ensemble QMC run requested 2026-05-30.
# 12x12 attractive Hubbard, U=-5, beta=6, Nup=Ndn=36.
# 2 CADES nodes, 64 MPI ranks = 64 independent Markov chains.
# Measurements are per rank: 500 warmup sweeps, 3000 measured configurations.
# One-particle unequal-time G is skipped; equal-time + BKT/current observables are measured.

set -u -o pipefail

PROJECT=/home/9pm/nUHubbard
LOG_DIR=${PROJECT}/logs
mkdir -p "${LOG_DIR}"
cd "${PROJECT}"

OUTDIR="runs/ce_bkt_only_L12x12_Um5_beta6_dtau01_Nup36_Ndn36_MPI64_CHK_CL144_Ntherm500_Nmeas3000perrank_seed2026053006"
RANKS=${SLURM_NTASKS:-64}
LX=12
LY=12
NUP=36
NDN=36
U=-5
BETA=6
DTAU=0.1
NWARMUPS=500
BATCH_NSAMPLES=1
MAX_BATCHES=3000
MEASURE_INTERVAL=2
BASE_SEED=2026053006
CHECKPOINT_FREQ_HOURS=0.10
RUNTIME_LIMIT_HOURS=0.80
CHECKPOINT_EVERY_BATCHES=1
CHECKPOINT_WARMUP_CHUNK=1
CLUSTER_SIZE=144
# Exact canonical Fourier projection for a 12x12 lattice would use V+1=145
# points.  This is currently prohibitively slow for BKT/current measurements;
# keep this variable explicit so we do not silently reuse the 3x3 value.
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
  [[ -f "${OUTDIR}/bkt_observables_qmc.tsv" && -f "${OUTDIR}/equal_time_observables_qmc.tsv" ]]
}

if success_outputs_present; then
  echo "[$(date -Is)] checkpointed CE-QMC run already complete: ${OUTDIR}"
  echo "=== bkt_observables_qmc.tsv ==="
  cat "${OUTDIR}/bkt_observables_qmc.tsv"
  echo "=== equal_time_observables_qmc.tsv ==="
  cat "${OUTDIR}/equal_time_observables_qmc.tsv"
  exit 0
fi

mkdir -p "${OUTDIR}"

START_EPOCH=$(date +%s)
echo "[$(date -Is)] starting checkpointed CE-QMC equal-time+BKT-only run"
echo "host=$(hostname) job=${SLURM_JOB_ID:-manual} nodes=${SLURM_JOB_NUM_NODES:-2} ranks=${RANKS} resubmit=${RESUBMIT_COUNT}/${MAX_RESUBMITS} outdir=${OUTDIR}"
echo "settings: Lx=${LX} Ly=${LY} U=${U} beta=${BETA} dtau=${DTAU} Nup=${NUP} Ndn=${NDN} warmup/rank=${NWARMUPS} measurements/rank=$((BATCH_NSAMPLES * MAX_BATCHES)) batches=${MAX_BATCHES} batch_size=${BATCH_NSAMPLES} measure_interval=${MEASURE_INTERVAL}"
echo "checkpoint: slurm_walltime=1h runtime_limit_hours=${RUNTIME_LIMIT_HOURS} checkpoint_freq_hours=${CHECKPOINT_FREQ_HOURS} keep=true auto_resubmit=${AUTO_RESUBMIT}"
echo "checkpoint_warmup_chunk=${CHECKPOINT_WARMUP_CHUNK}; batch_nsamples=1 so runtime/checkpoint is checked after every measured configuration"
echo "optimization: cluster_size=${CLUSTER_SIZE} num_fourier_points=${NUM_FOURIER_POINTS}"
echo "current_helper_sha256=$(sha256sum scripts/interacting_qmc_ed/ce_unequal_time_current_helpers.jl | awk '{print $1}')"
echo "driver_sha256=$(sha256sum scripts/interacting_qmc_ed/benchmark_ce_green_matsubara_3x3.jl | awk '{print $1}')"
echo "mpi_wrapper_sha256=$(sha256sum scripts/interacting_qmc_ed/benchmark_ce_green_tau_space_mpi_3x3.jl | awk '{print $1}')"
echo "JULIA=$(command -v julia)"
echo "MPIEXECJL=$(command -v mpiexecjl || true)"

rc=0
mpiexecjl -n "${RANKS}" \
  julia --project="${PROJECT}/julia_env" \
  "${PROJECT}/scripts/interacting_qmc_ed/benchmark_ce_green_tau_space_mpi_3x3.jl" \
  --lx="${LX}" \
  --ly="${LY}" \
  --nup="${NUP}" \
  --ndn="${NDN}" \
  --u="${U}" \
  --dtau="${DTAU}" \
  --beta="${BETA}" \
  --nwarmups="${NWARMUPS}" \
  --batch-nsamples="${BATCH_NSAMPLES}" \
  --measure-interval="${MEASURE_INTERVAL}" \
  --max-batches="${MAX_BATCHES}" \
  --cluster-size="${CLUSTER_SIZE}" \
  --num-fourier-points="${NUM_FOURIER_POINTS}" \
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
  --checkpoint-warmup-chunk="${CHECKPOINT_WARMUP_CHUNK}" \
  --runtime-limit-hours="${RUNTIME_LIMIT_HOURS}" \
  --checkpoint-keep=true \
  --checkpoint-sync-timeout-seconds=600 \
  --checkpoint-sync-poll-seconds=5 \
  --seed="${BASE_SEED}" \
  --python=python3 \
  --output-dir="${OUTDIR}" || rc=$?

END_EPOCH=$(date +%s)
echo "[$(date -Is)] launcher exit code: ${rc} runtime_seconds=$((END_EPOCH - START_EPOCH))"

if success_outputs_present; then
  echo "[$(date -Is)] checkpointed CE-QMC run complete: ${OUTDIR}"
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
    echo "[$(date -Is)] checkpointed CE-QMC run incomplete; resubmitting ${script_path} attempt ${next_count}/${MAX_RESUBMITS}"
    RESUBMIT_COUNT="${next_count}" sbatch "${script_path}"
    exit 0
  else
    echo "[$(date -Is)] checkpointed CE-QMC run incomplete and MAX_RESUBMITS reached: ${OUTDIR}" >&2
    exit 13
  fi
fi

echo "[$(date -Is)] checkpointed CE-QMC run failed/incomplete without valid checkpoint stop. rc=${rc}" >&2
exit "${rc}"
