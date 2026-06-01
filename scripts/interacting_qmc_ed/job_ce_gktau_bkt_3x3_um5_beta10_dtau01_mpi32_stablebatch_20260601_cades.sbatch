#!/bin/bash
#SBATCH -A ccsd
#SBATCH -p burst
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --ntasks-per-node=32
#SBATCH --exclusive
#SBATCH -c 1
#SBATCH --mem=100G
#SBATCH -t 16:00:00
#SBATCH --chdir=/home/9pm/nUHubbard
#SBATCH --job-name=ce_bkt3x3_d01_stab100k
#SBATCH --output=/home/9pm/nUHubbard/logs/%x_%j.out
#SBATCH --error=/home/9pm/nUHubbard/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peizhimai@gmail.com

set -euo pipefail

# Re-run after stable batched canonical Fourier-projection current-response optimization (2026-06-01).

cd /home/9pm/nUHubbard
export PATH="$HOME/.julia/bin:$HOME/.juliaup/bin:$PATH"
export JULIA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
unset LD_LIBRARY_PATH MPI_PATH MPI_ROOT MPICC MPICXX MPIF77 MPIF90 MPIFC || true
unset OMPI_MCA_pml OMPI_MCA_btl || true

OUTDIR="runs/ce_gktau_bkt_3x3_um5_beta10_dtau01_MPI32_STABLEBATCH_Ntherm10000_Nmeas100000perrank_seed20260601"
ED_G_DIR="runs/full_ed_3x3_um5_beta10_dtau01"
ED_BKT_DIR="runs/full_ed_3x3_um5_beta10_dtau01_bkt"
COMPARE_G_DIR="runs/compare_ce_gktau_bkt_3x3_um5_dtau01_MPI32_STABLEBATCH_100000perrank_vs_full_ed"
COMPARE_BKT="runs/compare_ce_bkt_3x3_um5_dtau01_MPI32_STABLEBATCH_100000perrank_vs_full_ed.tsv"
mkdir -p "${OUTDIR}"

START_EPOCH=$(date +%s)
echo "[$(date -Is)] starting STABLEBATCH MPI32 CE unequal-time G(r,tau)/G(k,tau)+BKT QMC"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} outdir=${OUTDIR}"
echo "settings: nodes=${SLURM_JOB_NUM_NODES:-1} ranks=${SLURM_NTASKS:-32} U=-5 beta=10 dtau=0.1 warmup/rank=10000 measurements/rank=100000 total=3200000 chunks/rank=100 chunk_size=1000 measure_bkt=true"
echo "stablebatch_helper_sha256=$(sha256sum scripts/interacting_qmc_ed/ce_unequal_time_current_helpers.jl | awk '{print $1}')"

echo "[$(date -Is)] validating current trace formula before production"
julia --project=/home/9pm/nUHubbard/julia_env \
  /home/9pm/nUHubbard/scripts/interacting_qmc_ed/validate_ce_current_trace_formula_3x3.jl

mpiexecjl -n "${SLURM_NTASKS:-32}" \
  julia --project=/home/9pm/nUHubbard/julia_env \
  /home/9pm/nUHubbard/scripts/interacting_qmc_ed/benchmark_ce_green_tau_space_mpi_3x3.jl \
  --u=-5 \
  --dtau=0.1 \
  --beta=10 \
  --nwarmups=10000 \
  --batch-nsamples=1000 \
  --measure-interval=2 \
  --max-batches=100 \
  --num-fourier-points=10 \
  --nfreq=1 \
  --use-lowrank=false \
  --use-charge-hs=false \
  --sys-type=complex \
  --measure-bkt=true \
  --seed=20260601 \
  --python=python3 \
  --output-dir="${OUTDIR}"

END_EPOCH=$(date +%s)
RUNTIME=$((END_EPOCH - START_EPOCH))
echo "[$(date -Is)] finished STABLEBATCH MPI32 CE unequal-time G(r,tau)/G(k,tau)+BKT QMC"
echo "runtime_seconds=${RUNTIME}"
echo "runtime_minutes=$(awk -v s=${RUNTIME} 'BEGIN{printf \"%.3f\", s/60.0}')"
ls -lh "${OUTDIR}"

if [[ -d "${ED_G_DIR}" ]]; then
  python3 /home/9pm/nUHubbard/scripts/interacting_qmc_ed/compare_ce_green_tau_space_vs_ed_3x3.py \
    --qmc-dir "${OUTDIR}" \
    --ed-dir "${ED_G_DIR}" \
    --outdir "${COMPARE_G_DIR}" \
    --min-slice=1 \
    --max-slice=50 || true
  if [[ -f "${COMPARE_G_DIR}/green_tau_space_comparison_summary.tsv" ]]; then
    echo "=== G tau-space comparison summary (dtau=0.1, slice 1..50) ==="
    cat "${COMPARE_G_DIR}/green_tau_space_comparison_summary.tsv"
  fi
fi

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
