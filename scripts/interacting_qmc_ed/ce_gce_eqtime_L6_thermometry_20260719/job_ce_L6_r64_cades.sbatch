#!/usr/bin/env bash
#SBATCH -A ccsd
#SBATCH -p burst
#SBATCH --qos=default
#SBATCH -N 2
#SBATCH -n 64
#SBATCH --ntasks-per-node=32
#SBATCH -c 1
#SBATCH --mem=100G
#SBATCH -t 01:30:00
#SBATCH --chdir=/home/9pm/nUHubbard
#SBATCH --job-name=ceL6Eq64
#SBATCH --output=/home/9pm/nUHubbard/logs/%x_%A_%a_%j.out
#SBATCH --error=/home/9pm/nUHubbard/logs/%x_%A_%a_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cosdis123@gmail.com
#SBATCH --array=0-0

set -euo pipefail
PROJECT=${PROJECT:-/home/9pm/nUHubbard}
SELF=$(readlink -f "${BASH_SOURCE[0]}")
export RESUBMIT_SCRIPT="${SELF}"
exec "${PROJECT}/scripts/interacting_qmc_ed/ce_gce_eqtime_L6_thermometry_20260719/run_ce_manifest_task.sh"
