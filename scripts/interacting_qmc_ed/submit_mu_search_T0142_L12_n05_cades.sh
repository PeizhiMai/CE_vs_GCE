#!/bin/bash
# Submit the five stage-1 T/t≈0.142 chemical-potential search jobs.
#
# Dry-run by default.  To submit on CADES:
#
#   cd /home/9pm/nUHubbard
#   DRY_RUN=0 scripts/interacting_qmc_ed/submit_mu_search_T0142_L12_n05_cades.sh
#
# Each job requests one burst node and uses 32 MPI ranks/chains.

set -euo pipefail

PROJECT=${PROJECT:-/home/9pm/nUHubbard}
SCRIPT_DIR=${SCRIPT_DIR:-${PROJECT}/scripts/interacting_qmc_ed}
DRY_RUN=${DRY_RUN:-1}

scripts=(
  job_mu_search_T0142_L12_n05_mu_m0p80_burst_cades.sbatch
  job_mu_search_T0142_L12_n05_mu_m0p70_burst_cades.sbatch
  job_mu_search_T0142_L12_n05_mu_m0p60_burst_cades.sbatch
  job_mu_search_T0142_L12_n05_mu_m0p50_burst_cades.sbatch
  job_mu_search_T0142_L12_n05_mu_m0p40_burst_cades.sbatch
)

echo "Stage-1 mu search submitter"
echo "PROJECT=${PROJECT}"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "DRY_RUN=${DRY_RUN}"
echo "Submitting ${#scripts[@]} jobs; one node/job, 32 MPI ranks/job."

for script in "${scripts[@]}"; do
  path="${SCRIPT_DIR}/${script}"
  if [[ ! -f "${path}" ]]; then
    echo "missing script: ${path}" >&2
    exit 2
  fi
  if [[ "${DRY_RUN}" == "0" ]]; then
    sbatch "${path}"
  else
    printf 'sbatch %q\n' "${path}"
  fi
done

echo
echo "After the jobs complete, summarize with:"
echo "  python3.11 ${SCRIPT_DIR}/summarize_density_tuning.py ${PROJECT}/runs/density_mu_search_L12_n05_stage1_T0142 --target-density 0.5"
