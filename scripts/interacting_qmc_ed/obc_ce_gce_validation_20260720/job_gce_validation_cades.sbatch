#!/usr/bin/env bash
#SBATCH -A ccsd
#SBATCH -p burst
#SBATCH --qos=default
#SBATCH --job-name=obcGCEVal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=01:30:00
#SBATCH --array=0-71
#SBATCH --output=/home/9pm/nUHubbard_obc_dev/logs/obcGCEVal_%A_%a.out
#SBATCH --error=/home/9pm/nUHubbard_obc_dev/logs/obcGCEVal_%A_%a.err
set -u
echo "This first-generation GCE validation is superseded by job_gce_validation_independent_rdm2_fix2_cades.sbatch." >&2
exit 2
PROJECT=${PROJECT:-/home/9pm/nUHubbard_obc_dev}
export SBATCH_SCRIPT=${SBATCH_SCRIPT:-${PROJECT}/scripts/interacting_qmc_ed/obc_ce_gce_validation_20260720/job_gce_validation_cades.sbatch}
exec "${PROJECT}/scripts/interacting_qmc_ed/obc_ce_gce_validation_20260720/run_gce_validation_task.sh"
