#!/bin/bash
#SBATCH -A ccsd
#SBATCH -p high_mem
#SBATCH -J muT0142L12
#SBATCH -N 5
#SBATCH -n 160
#SBATCH --ntasks-per-node=32
#SBATCH -c 1
#SBATCH --mem=100G
#SBATCH -t 6:00:00
#SBATCH --chdir=/home/9pm/nUHubbard
#SBATCH --output=/home/9pm/nUHubbard/logs/muT0142L12-%j.out
#SBATCH --error=/home/9pm/nUHubbard/logs/muT0142L12-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=peizhimai@gmail.com

# Anchor chemical-potential scan for the L=12, n=0.5 benchmark.
#
# Purpose:
#   Locate mu such that n(mu, T/t=0.142) ~= 0.5.
#
# Layout:
#   5 nodes total.
#   Each node runs one independent SmoQyDQMC parameter group.
#   Each group uses 32 MPI ranks = 32 independent Markov chains.
#
# Parameter groups:
#   T/t = 0.142
#   mu = -0.80, -0.70, -0.60, -0.50, -0.40
#
# Submit:
#   cd /home/9pm/nUHubbard
#   sbatch scripts/interacting_qmc_ed/job_mu_anchor_T0142_L12_n05_cades.sbatch

set -euo pipefail

export PROJECT=/home/9pm/nUHubbard
export OUT_PARENT=${PROJECT}/runs/density_tuning_L12_n05
export LOG_DIR=${PROJECT}/logs

export L=12
export LY=12
export U=-5.0
export TPRIME=0.0

export T_LIST="0.142"
export MU_LIST="-0.80 -0.70 -0.60 -0.50 -0.40"

export N_THERM=2000
export N_MEASUREMENTS=10000
export N_BINS=20
export N_UPDATES=2
export DELTA_TAU=0.1
export N_STAB=10
export N_STAB_MIN=6
export DG_MAX=1e-5
export USE_REFLECTION_UPDATE=false
export UPDATE_STABILIZATION_FREQUENCY=true

export CHECKPOINT_FREQ_HOURS=0.5
export RUNTIME_LIMIT_HOURS=5.75

export TASKS_PER_GROUP=32
export SID_BASE=120500
export GROUP_OFFSET=0
export GROUP_COUNT=5
export AUTO_RESUBMIT=true
export MAX_RESUBMITS=50
export RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}

exec "${PROJECT}/scripts/interacting_qmc_ed/run_density_tuning_L12_n05_pack_cades.sbatch"
