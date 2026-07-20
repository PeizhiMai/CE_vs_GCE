#!/usr/bin/env bash
#SBATCH -A ccsd
#SBATCH -p burst
#SBATCH --qos=default
#SBATCH --job-name=obcCERDM2
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=01:30:00
#SBATCH --array=0-71
#SBATCH --output=/home/9pm/nUHubbard_obc_dev/logs/obcCERDM2_%A_%a.out
#SBATCH --error=/home/9pm/nUHubbard_obc_dev/logs/obcCERDM2_%A_%a.err
set -u
PROJECT=${PROJECT:-/home/9pm/nUHubbard_obc_dev}
export MANIFEST=${MANIFEST:-${PROJECT}/scripts/interacting_qmc_ed/obc_ce_gce_validation_20260720/manifests_ce_rdm2_fix1/ce_validation_manifest.tsv}
export SBATCH_SCRIPT=${SBATCH_SCRIPT:-${PROJECT}/scripts/interacting_qmc_ed/obc_ce_gce_validation_20260720/job_ce_validation_rdm2_fix1_cades.sbatch}
exec "${PROJECT}/scripts/interacting_qmc_ed/obc_ce_gce_validation_20260720/run_ce_validation_task.sh"
