#!/usr/bin/env bash
#SBATCH -A ccsd
#SBATCH -p burst
#SBATCH --qos=default
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --ntasks-per-node=32
#SBATCH -c 1
#SBATCH --mem=100G
#SBATCH -t 01:30:00
#SBATCH --chdir=/home/9pm/nUHubbard_obc_dev
#SBATCH --job-name=gceL6OSpin
#SBATCH --output=/home/9pm/nUHubbard_obc_dev/logs/%x_%A_%a_%j.out
#SBATCH --error=/home/9pm/nUHubbard_obc_dev/logs/%x_%A_%a_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cosdis123@gmail.com
#SBATCH --array=0-0

set -euo pipefail
PROJECT=${PROJECT:-/home/9pm/nUHubbard_obc_dev}
export GCE_FAMILY=spinHS
export RESUBMIT_SCRIPT=$(readlink -f "${BASH_SOURCE[0]}")
exec "${PROJECT}/scripts/interacting_qmc_ed/ce_gce_eqtime_L6_obc_thermometry_20260720/run_gce_production_manifest_task.sh"
