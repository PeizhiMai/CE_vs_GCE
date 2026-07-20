#!/usr/bin/env bash
#SBATCH -A ccsd
#SBATCH -p burst
#SBATCH --qos=default
#SBATCH --job-name=obcSmoke
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=/home/9pm/nUHubbard_obc_dev/logs/obcSmoke_%j.out
#SBATCH --error=/home/9pm/nUHubbard_obc_dev/logs/obcSmoke_%j.err
set -euo pipefail
PROJECT=${PROJECT:-/home/9pm/nUHubbard_obc_dev}
exec "${PROJECT}/scripts/interacting_qmc_ed/obc_ce_gce_validation_20260720/run_obc_smoke_cades.sh"
