#!/bin/bash
#SBATCH -A ccsd
#SBATCH -p burst
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --ntasks-per-node=32
#SBATCH -c 1
#SBATCH -J chkptCompareL12
#SBATCH --mem=100G
#SBATCH -t 0:25:00
#SBATCH --chdir=/home/9pm/nUHubbard
#SBATCH --output=/home/9pm/nUHubbard/logs/chkptCompareL12-%j.out
#SBATCH --error=/home/9pm/nUHubbard/logs/chkptCompareL12-%j.err

# Compare a lightweight uninterrupted run to a checkpoint/resume run.
# Both use the same fixed base RNG seed and rank-dependent seeds, so the final
# density-only statistics and bin data should match exactly.

set -u -o pipefail

PROJECT=/home/9pm/nUHubbard
OUT_PARENT=${PROJECT}/runs/checkpoint_compare_L12_n05
LOG_DIR=${PROJECT}/logs
mkdir -p "${OUT_PARENT}" "${LOG_DIR}"

# Avoid CADES user-env/system-OpenMPI contamination when using Julia OpenMPI_jll.
unset LD_LIBRARY_PATH MPI_PATH MPI_ROOT MPICC MPICXX MPIF77 MPIF90 MPIFC
unset OMPI_MCA_pml OMPI_MCA_btl

export PATH="$HOME/.julia/bin:$HOME/.juliaup/bin:$PATH"
export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export MKL_DYNAMIC=${MKL_DYNAMIC:-FALSE}

U=-5.0
TPRIME=0.0
MU=-0.60
LX=12
LY=12
T_TARGET=0.142
DELTA_TAU=0.1
LTAU=$(python3.11 -c "import sys; print(round(1.0/float(sys.argv[1])/float(sys.argv[2])))" "${T_TARGET}" "${DELTA_TAU}")
BETA=$(python3.11 -c "import sys; print(f'{int(sys.argv[1]) * float(sys.argv[2]):.8f}')" "${LTAU}" "${DELTA_TAU}")

N_THERM=20
N_MEASUREMENTS=40
N_BINS=4
N_UPDATES=1
PH_SYM_FORM=true
MEASUREMENT_PROFILE=density-only
N_STAB=10
N_STAB_MIN=6
DG_MAX=1e-5
USE_REFLECTION_UPDATE=false
UPDATE_STABILIZATION_FREQUENCY=true
BASE_SEED=24681357

SID_NOCHK=129920
SID_CHK=129921

run_base() {
  python3.11 - "$1" "$U" "$TPRIME" "$MU" "$LX" "$LY" "$BETA" <<'PY2'
import sys
sid, u, tp, mu, lx, ly, beta = sys.argv[1:]
print(f"attractive_hubbard_rect_U{float(u):.2f}_tp{float(tp):.2f}_mu{float(mu):.2f}_Lx{int(lx)}_Ly{int(ly)}_b{float(beta):.2f}-{int(sid)}")
PY2
}

archive_existing() {
  local sid="$1"
  local base
  base=$(run_base "$sid")
  local stamp
  stamp=$(date +%Y%m%d_%H%M%S)
  local archive="${OUT_PARENT}/superseded_${stamp}"
  for d in "${OUT_PARENT}/${base}" "${OUT_PARENT}/complete_${base}"; do
    if [[ -d "$d" ]]; then
      mkdir -p "${archive}"
      mv "$d" "${archive}/"
    fi
  done
}

launch_driver() {
  local sid="$1"
  local checkpoint_freq="$2"
  local runtime_limit="$3"
  echo "[$(date)] launch sid=${sid} checkpoint_freq=${checkpoint_freq} runtime_limit=${runtime_limit}"
  mpiexecjl -n "${SLURM_NTASKS:-32}" \
    julia --project="${PROJECT}/julia_env" \
    "${PROJECT}/scripts/interacting_qmc_ed/run_smoqydqmc_attractive_hubbard_checkpoint.jl" \
    "${sid}" "${U}" "${TPRIME}" "${MU}" "${LX}" "${BETA}" \
    "${N_THERM}" "${N_MEASUREMENTS}" "${N_BINS}" "${N_UPDATES}" \
    "${checkpoint_freq}" "${runtime_limit}" \
    "${PH_SYM_FORM}" "${OUT_PARENT}" "${LY}" "${MEASUREMENT_PROFILE}" \
    "${DELTA_TAU}" "${N_STAB}" "${DG_MAX}" \
    "${USE_REFLECTION_UPDATE}" "${UPDATE_STABILIZATION_FREQUENCY}" "${N_STAB_MIN}" \
    "${BASE_SEED}"
}

echo "[$(date)] Starting checkpoint-vs-no-checkpoint comparison"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-manual}"
echo "BETA=${BETA} LTAU=${LTAU} MU=${MU} BASE_SEED=${BASE_SEED}"
echo "N_THERM=${N_THERM} N_MEASUREMENTS=${N_MEASUREMENTS} N_BINS=${N_BINS} N_UPDATES=${N_UPDATES}"
echo "JULIA=$(command -v julia)"
echo "MPIEXECJL=$(command -v mpiexecjl || true)"

archive_existing "${SID_NOCHK}"
archive_existing "${SID_CHK}"

# 1. Uninterrupted no-checkpoint run.
launch_driver "${SID_NOCHK}" 0 0.30
rc_nochk=$?
echo "[$(date)] no-checkpoint rc=${rc_nochk}"
if [[ "${rc_nochk}" != "0" ]]; then
  echo "No-checkpoint run failed" >&2
  exit "${rc_nochk}"
fi

# 2. First checkpoint leg.  This intentionally stops immediately after a valid
# 32-rank checkpoint.  mpiexecjl reports this as rc=1 with ProcessExited(13).
launch_driver "${SID_CHK}" 0.001 0.0005
rc_chk1=$?
base_chk=$(run_base "${SID_CHK}")
incomplete_dir="${OUT_PARENT}/${base_chk}"
checkpoint_count=0
if [[ -d "${incomplete_dir}" ]]; then
  checkpoint_count=$(find "${incomplete_dir}" -maxdepth 1 -name 'checkpoint_pID-*.jld2' | wc -l | tr -d ' ')
fi
echo "[$(date)] checkpoint first-leg rc=${rc_chk1} checkpoint_count=${checkpoint_count}/${SLURM_NTASKS:-32}"
if [[ "${checkpoint_count}" -lt "${SLURM_NTASKS:-32}" ]]; then
  echo "Checkpoint first leg did not create all checkpoint files" >&2
  exit 13
fi

# 3. Resume from checkpoint and finish.
launch_driver "${SID_CHK}" 0 0.30
rc_chk2=$?
echo "[$(date)] checkpoint resume rc=${rc_chk2}"
if [[ "${rc_chk2}" != "0" ]]; then
  echo "Checkpoint resume run failed" >&2
  exit "${rc_chk2}"
fi

# 4. Compare outputs.  The summarized CSV files should be byte-identical.
# HDF5 files are compared via h5dump content because the raw file bytes include
# the input filename in the dump/header path and can differ in harmless metadata.
python3.11 - "${OUT_PARENT}" "$(run_base "${SID_NOCHK}")" "$(run_base "${SID_CHK}")" <<'PY'
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
nochk = root / ("complete_" + sys.argv[2])
chk = root / ("complete_" + sys.argv[3])

if not nochk.is_dir():
    raise SystemExit(f"missing no-checkpoint complete dir: {nochk}")
if not chk.is_dir():
    raise SystemExit(f"missing checkpoint complete dir: {chk}")

for label, folder in [("no-checkpoint", nochk), ("checkpoint", chk)]:
    n_pids = len(list(folder.glob("simulation_info_sID-*_pID-*.toml")))
    print(f"{label} pID files: {n_pids}")
    if n_pids != 32:
        raise SystemExit(f"{label} expected 32 pID files, found {n_pids}")

def read_space_csv(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter=" ", skipinitialspace=True))

def compare_csv(name: str, atol: float = 0.0, rtol: float = 0.0):
    a_path = nochk / name
    b_path = chk / name
    if not a_path.is_file() or not b_path.is_file():
        raise SystemExit(f"missing comparison file {name}: {a_path} {b_path}")
    a = read_space_csv(a_path)
    b = read_space_csv(b_path)
    if len(a) != len(b):
        raise SystemExit(f"{name}: row count differs {len(a)} != {len(b)}")
    max_abs = 0.0
    max_field = None
    for i, (ra, rb) in enumerate(zip(a, b), start=1):
        if ra.keys() != rb.keys():
            raise SystemExit(f"{name}: columns differ")
        for key in ra:
            va, vb = ra[key], rb[key]
            try:
                fa, fb = float(va), float(vb)
            except ValueError:
                if va != vb:
                    raise SystemExit(f"{name}: row {i} column {key} differs: {va!r} != {vb!r}")
                continue
            diff = abs(fa - fb)
            if diff > max_abs:
                max_abs = diff
                max_field = (i, key, fa, fb)
            if not math.isclose(fa, fb, rel_tol=rtol, abs_tol=atol):
                raise SystemExit(f"{name}: row {i} column {key} differs: {fa} != {fb}, diff={diff}")
    print(f"{name}: MATCH max_abs_diff={max_abs:.3e} at {max_field}")

compare_csv("global_stats.csv")
compare_csv("local_stats.csv")
print("CHECKPOINT_COMPARE_PASS")
PY

for f in global_stats.csv local_stats.csv; do
  if ! cmp -s "${OUT_PARENT}/complete_$(run_base "${SID_NOCHK}")/${f}" \
             "${OUT_PARENT}/complete_$(run_base "${SID_CHK}")/${f}"; then
    echo "${f}: byte comparison failed" >&2
    exit 1
  fi
  echo "${f}: byte-identical"
done

for f in stats.h5 $(python3.11 - <<'PY'
for p in range(32):
    print(f"bins/bins_pID-{p}.h5")
PY
); do
  h5dump "${OUT_PARENT}/complete_$(run_base "${SID_NOCHK}")/${f}" | tail -n +2 > "/tmp/chkpt_compare_nochk_${SLURM_JOB_ID:-manual}.txt"
  h5dump "${OUT_PARENT}/complete_$(run_base "${SID_CHK}")/${f}" | tail -n +2 > "/tmp/chkpt_compare_chk_${SLURM_JOB_ID:-manual}.txt"
  if ! diff -q "/tmp/chkpt_compare_nochk_${SLURM_JOB_ID:-manual}.txt" "/tmp/chkpt_compare_chk_${SLURM_JOB_ID:-manual}.txt" >/dev/null; then
    echo "${f}: h5dump content differs" >&2
    exit 1
  fi
done
echo "stats.h5 and all 32 bin HDF5 files: h5dump content identical"

echo "[$(date)] Checkpoint comparison passed"
