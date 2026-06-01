#!/bin/bash
#SBATCH -A ccsd
#SBATCH -p burst
#SBATCH -N 2
#SBATCH -n 64
#SBATCH --ntasks-per-node=32
#SBATCH --exclusive
#SBATCH -c 1
#SBATCH --mem=100G
#SBATCH -t 10:00:00
#SBATCH --chdir=/home/9pm/nUHubbard
#SBATCH --job-name=ce_bkt3x3_d005_ad30k
#SBATCH --output=/home/9pm/nUHubbard/logs/%x_%j.out
#SBATCH --error=/home/9pm/nUHubbard/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peizhimai@gmail.com

set -euo pipefail

# Adaptive propagated canonical Fourier-projection BKT-current rerun (2026-06-01).
# BKT/equal-time only; intentionally skip unequal-time one-particle G.

cd /home/9pm/nUHubbard
export PATH="$HOME/.julia/bin:$HOME/.juliaup/bin:$PATH"
export JULIA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
unset LD_LIBRARY_PATH MPI_PATH MPI_ROOT MPICC MPICXX MPIF77 MPIF90 MPIFC || true
unset OMPI_MCA_pml OMPI_MCA_btl || true

OUTDIR="runs/ce_bkt_3x3_um5_beta10_dtau005_MPI64_ADAPTIVE_BKTONLY_Ntherm5000_Nmeas30000perrank_seed20260601"
ED_BKT_DIR="runs/full_ed_3x3_um5_beta10_dtau01_bkt"
COMPARE_BKT="runs/compare_ce_bkt_3x3_um5_dtau005_MPI64_ADAPTIVE_BKTONLY_30000perrank_vs_full_ed.tsv"
mkdir -p "${OUTDIR}"

START_EPOCH=$(date +%s)
echo "[$(date -Is)] starting ADAPTIVE_BKTONLY MPI64 CE BKT QMC"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} outdir=${OUTDIR}"
echo "settings: nodes=${SLURM_JOB_NUM_NODES:-2} ranks=${SLURM_NTASKS:-64} U=-5 beta=10 dtau=0.05 warmup/rank=5000 measurements/rank=30000 total=1920000 chunks/rank=30 chunk_size=1000 measure_greens=false measure_bkt=true"
echo "adaptive_current: estimator=propagated interval=10 tol=1e-6 min=1 max=20 growth_patience=3"
echo "helper_sha256=$(sha256sum scripts/interacting_qmc_ed/ce_unequal_time_current_helpers.jl | awk '{print $1}')"

echo "[$(date -Is)] validating current estimators before production"
julia --project=/home/9pm/nUHubbard/julia_env \
  /home/9pm/nUHubbard/scripts/interacting_qmc_ed/validate_ce_current_trace_formula_3x3.jl
julia --project=/home/9pm/nUHubbard/julia_env \
  /home/9pm/nUHubbard/scripts/interacting_qmc_ed/validate_ce_current_propagated_3x3.jl

mpiexecjl -n "${SLURM_NTASKS:-64}" \
  julia --project=/home/9pm/nUHubbard/julia_env \
  /home/9pm/nUHubbard/scripts/interacting_qmc_ed/benchmark_ce_green_tau_space_mpi_3x3.jl \
  --u=-5 \
  --dtau=0.05 \
  --beta=10 \
  --nwarmups=5000 \
  --batch-nsamples=1000 \
  --measure-interval=2 \
  --max-batches=30 \
  --num-fourier-points=10 \
  --nfreq=1 \
  --use-lowrank=false \
  --use-charge-hs=false \
  --sys-type=complex \
  --measure-greens=false \
  --measure-bkt=true \
  --measure-equal-time=true \
  --bkt-current-estimator=propagated \
  --bkt-adaptive-refresh=true \
  --bkt-refresh-interval=10 \
  --bkt-refresh-tol=1e-6 \
  --bkt-refresh-min=1 \
  --bkt-refresh-max=20 \
  --bkt-refresh-growth-patience=3 \
  --seed=20260601 \
  --python=python3 \
  --output-dir="${OUTDIR}"

END_EPOCH=$(date +%s)
RUNTIME=$((END_EPOCH - START_EPOCH))
echo "[$(date -Is)] finished ADAPTIVE_BKTONLY MPI64 CE BKT QMC"
echo "runtime_seconds=${RUNTIME}"
echo "runtime_minutes=$(awk -v s=${RUNTIME} 'BEGIN{printf \"%.3f\", s/60.0}')"
ls -lh "${OUTDIR}"

if [[ -f "${OUTDIR}/bkt_observables_qmc.tsv" && -f "${ED_BKT_DIR}/bkt_observables_ed.tsv" ]]; then
  python3 /home/9pm/nUHubbard/scripts/interacting_qmc_ed/compare_ce_bkt_observables_3x3.py \
    --qmc-bkt "${OUTDIR}/bkt_observables_qmc.tsv" \
    --ed-bkt "${ED_BKT_DIR}/bkt_observables_ed.tsv" \
    --out "${COMPARE_BKT}" || true
  if [[ -f "${COMPARE_BKT}" ]]; then
    echo "=== BKT comparison summary ==="
    cat "${COMPARE_BKT}"
  fi
fi
