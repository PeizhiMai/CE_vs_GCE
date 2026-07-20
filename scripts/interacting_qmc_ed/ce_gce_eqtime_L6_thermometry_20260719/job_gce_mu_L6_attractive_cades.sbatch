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
#SBATCH --chdir=/home/9pm/nUHubbard
#SBATCH --job-name=muL6Attr
#SBATCH --output=/home/9pm/nUHubbard/logs/%x_%A_%a_%j.out
#SBATCH --error=/home/9pm/nUHubbard/logs/%x_%A_%a_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cosdis123@gmail.com
#SBATCH --array=0-0

set -euo pipefail
PROJECT=${PROJECT:-/home/9pm/nUHubbard}
export GCE_FAMILY=attractive
export RESUBMIT_SCRIPT=$(readlink -f "${BASH_SOURCE[0]}")
exec "${PROJECT}/scripts/interacting_qmc_ed/ce_gce_eqtime_L6_thermometry_20260719/run_gce_mu_manifest_task.sh"
