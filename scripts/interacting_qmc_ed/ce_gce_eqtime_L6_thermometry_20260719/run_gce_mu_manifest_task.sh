#!/usr/bin/env bash
# Shared density-only GCE chemical-potential probe runner for L=6.

set -u -o pipefail
PROJECT=${PROJECT:-/home/9pm/nUHubbard}
MANIFEST=${MANIFEST:?MANIFEST must point to an L6 GCE probe TSV}
GCE_FAMILY=${GCE_FAMILY:?GCE_FAMILY must be attractive or spinHS}
RESUBMIT_SCRIPT=${RESUBMIT_SCRIPT:?RESUBMIT_SCRIPT must point to a thin Slurm wrapper}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
LOG_DIR=${PROJECT}/logs
cd "${PROJECT}" || exit 2
mkdir -p "${LOG_DIR}"

row_tsv=$(python3 - "${MANIFEST}" "${TASK_ID}" <<'PY'
import csv, sys
path, task = sys.argv[1], int(sys.argv[2])
with open(path, newline="") as f:
    rows=list(csv.DictReader(f, delimiter="\t"))
if not 0 <= task < len(rows):
    raise SystemExit(f"task {task} out of range 0..{len(rows)-1}")
r=rows[task]
fields=[
    "idx","U_label","U","Ntot_target","target_density","beta","T",
    "mu_L8_reference","L8_reference_Ntot","L8_reference_density","mu_probe","mu_label",
    "probe_offset","probe_role","Lx","Ly","dtau","ntherm","nmeasurements","nbins",
    "nupdates","account","out_parent","sid","seed","target_key","tuning_status",
    "source_manifests","job_tag",
]
missing=[k for k in fields if k not in r]
if missing: raise SystemExit(f"manifest missing fields: {missing}")
print("\t".join(r[k] for k in fields))
PY
) || exit $?
IFS=$'\t' read -r IDX U_LABEL U NTOT_TARGET TARGET_DENSITY BETA T_TARGET MU_L8_REFERENCE L8_REFERENCE_NTOT L8_REFERENCE_DENSITY MU MU_LABEL PROBE_OFFSET PROBE_ROLE LX LY DTAU N_THERM N_MEASUREMENTS N_BINS N_UPDATES ACCOUNT OUT_PARENT SID BASE_SEED TARGET_KEY TUNING_STATUS SOURCE_MANIFESTS JOB_TAG <<< "${row_tsv}"

RANKS=${SLURM_NTASKS:-32}
SMOKE_MODE=${SMOKE_MODE:-false}
if [[ "${ACCOUNT}" != "ccsd" && "${ACCOUNT}" != "cnms" ]]; then
  echo "ERROR: unapproved L6 account in manifest: ${ACCOUNT}" >&2
  exit 2
fi
if [[ "${SLURM_JOB_ACCOUNT:-unset}" != "${ACCOUNT}" ]]; then
  echo "ERROR: manifest/Slurm account mismatch: manifest=${ACCOUNT} Slurm=${SLURM_JOB_ACCOUNT:-unset}" >&2
  exit 2
fi
if [[ "${SLURM_JOB_PARTITION:-burst}" != "burst" || "${SLURM_JOB_QOS:-default}" != "default" ]]; then
  echo "ERROR: L6 workflow requires burst/default" >&2; exit 2
fi
if (( RANKS != 32 )) && [[ "${SMOKE_MODE}" != "true" ]]; then echo "ERROR: GCE probes require 32 ranks, got ${RANKS}" >&2; exit 2; fi
if (( LX != 6 || LY != 6 )) && [[ "${SMOKE_MODE}" != "true" ]]; then echo "ERROR: GCE probes require L=6" >&2; exit 2; fi
if [[ "${CHECKPOINT_RESET_ACCUMULATORS:-false}" != "false" ]]; then
  echo "ERROR: checkpoint accumulator reset is forbidden" >&2; exit 2
fi

case "${GCE_FAMILY}" in
  attractive)
    python3 - "${U}" <<'PY' || exit $?
import sys
assert float(sys.argv[1]) < 0, "attractive wrapper received nonnegative U"
PY
    DRIVER=${PROJECT}/scripts/interacting_qmc_ed/run_smoqydqmc_attractive_hubbard_checkpoint.jl
    RUN_PREFIX=attractive_hubbard_rect
    UPDATE_STABILIZATION_FREQUENCY=${UPDATE_STABILIZATION_FREQUENCY:-true}
    ;;
  spinHS)
    python3 - "${U}" <<'PY' || exit $?
import sys
assert float(sys.argv[1]) > 0, "spinHS wrapper received nonpositive U"
PY
    DRIVER=${PROJECT}/scripts/interacting_qmc_ed/run_smoqydqmc_hubbard_spin_hs_checkpoint.jl
    RUN_PREFIX=hubbard_spin_hs_rect
    # Match production: adapt the stabilization cadence when the Green-matrix
    # drift check exceeds DG_MAX.  Leaving this disabled created very large
    # warning logs and is unnecessarily risky for density interpolation.
    UPDATE_STABILIZATION_FREQUENCY=${UPDATE_STABILIZATION_FREQUENCY:-true}
    ;;
  *) echo "ERROR: invalid GCE_FAMILY=${GCE_FAMILY}" >&2; exit 2;;
esac

TPRIME=${TPRIME:-0.0}
PH_SYM_FORM=${PH_SYM_FORM:-true}
MEASUREMENT_PROFILE=density-only
N_STAB=${N_STAB:-10}
N_STAB_MIN=${N_STAB_MIN:-6}
DG_MAX=${DG_MAX:-1e-5}
USE_REFLECTION_UPDATE=${USE_REFLECTION_UPDATE:-false}
CHECKPOINT_FREQ_HOURS=${CHECKPOINT_FREQ_HOURS:-1.0}
RUNTIME_LIMIT_HOURS=${RUNTIME_LIMIT_HOURS:-1.25}
CHECKPOINT_EVERY_N_MEASUREMENTS=${CHECKPOINT_EVERY_N_MEASUREMENTS:-10}
REQUESTED_WALLTIME=${REQUESTED_WALLTIME:-01:30:00}
AUTO_RESUBMIT=${AUTO_RESUBMIT:-true}
MAX_RESUBMITS=${MAX_RESUBMITS:-80}
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}
RESUBMIT_JOB_NAME=${SLURM_JOB_NAME:-muL6Probe}
MIN_RUNTIME_STOP_SECONDS=${MIN_RUNTIME_STOP_SECONDS:-300}
export CHECKPOINT_RESET_ACCUMULATORS=false

unset LD_LIBRARY_PATH MPI_PATH MPI_ROOT MPICC MPICXX MPIF77 MPIF90 MPIFC || true
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,tcp
export OMPI_MCA_btl_tcp_if_include=mgmt0
export OMPI_MCA_oob_tcp_if_include=mgmt0
export PRTE_MCA_oob_tcp_if_include=mgmt0
export PMIX_MCA_ptl_tcp_if_include=mgmt0
export PATH="$HOME/.julia/bin:$HOME/.juliaup/bin:$PATH"
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-${PROJECT}/.julia_depot:$HOME/.julia}
export JULIA_PROJECT=${JULIA_PROJECT:-${PROJECT}/julia_env}
export JULIA_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE
MPIEXECJL=${MPIEXECJL:-$(command -v mpiexecjl)}
[[ -n "${MPIEXECJL}" ]] || { echo "ERROR: mpiexecjl not found" >&2; exit 2; }
mkdir -p "${OUT_PARENT}"

run_base=$(python3 - "${RUN_PREFIX}" "${SID}" "${U}" "${TPRIME}" "${MU}" "${LX}" "${LY}" "${BETA}" <<'PY'
import sys
prefix,sid,u,tp,mu,lx,ly,beta=sys.argv[1:]
print(f"{prefix}_U{float(u):.2f}_tp{float(tp):.2f}_mu{float(mu):.2f}_Lx{int(lx)}_Ly{int(ly)}_b{float(beta):.2f}-{int(sid)}")
PY
)
complete_dir=${OUT_PARENT}/complete_${run_base}
incomplete_dir=${OUT_PARENT}/${run_base}
if [[ -d "${complete_dir}" && -f "${complete_dir}/global_stats.csv" ]]; then
  echo "[$(date -Is)] already complete: ${complete_dir}"
  exit 0
fi

START_EPOCH=$(date +%s)
job_start_marker=${OUT_PARENT}/.checkpoint_job_start_${SLURM_JOB_ID:-manual}_${TASK_ID}_${START_EPOCH}.marker
date > "${job_start_marker}"
echo "[$(date -Is)] L6 GCE mu probe family=${GCE_FAMILY} task=${TASK_ID} job=${SLURM_JOB_ID:-manual} target=${TARGET_KEY} role=${PROBE_ROLE}"
echo "U=${U} beta=${BETA} T=${T_TARGET} Ntarget=${NTOT_TARGET} density_target=${TARGET_DENSITY} mu8=${MU_L8_REFERENCE} mu=${MU} offset=${PROBE_OFFSET} ranks=${RANKS} warmups=${N_THERM} measurements/rank=${N_MEASUREMENTS} continuation=${RESUBMIT_COUNT}/${MAX_RESUBMITS}"
sha256sum "${DRIVER}" || exit 2

rc=0
"${MPIEXECJL}" -n "${RANKS}" julia --project="${JULIA_PROJECT}" "${DRIVER}" \
  "${SID}" "${U}" "${TPRIME}" "${MU}" "${LX}" "${BETA}" \
  "${N_THERM}" "${N_MEASUREMENTS}" "${N_BINS}" "${N_UPDATES}" \
  "${CHECKPOINT_FREQ_HOURS}" "${RUNTIME_LIMIT_HOURS}" \
  "${PH_SYM_FORM}" "${OUT_PARENT}" "${LY}" "${MEASUREMENT_PROFILE}" \
  "${DTAU}" "${N_STAB}" "${DG_MAX}" \
  "${USE_REFLECTION_UPDATE}" "${UPDATE_STABILIZATION_FREQUENCY}" "${N_STAB_MIN}" \
  "${BASE_SEED}" "${CHECKPOINT_EVERY_N_MEASUREMENTS}" || rc=$?

END_EPOCH=$(date +%s); RUNTIME_SECONDS=$((END_EPOCH - START_EPOCH))
if [[ -d "${complete_dir}" && -f "${complete_dir}/global_stats.csv" ]]; then
  echo "[$(date -Is)] complete: ${complete_dir}"
  exit 0
fi

checkpoint_count=0; fresh_checkpoint_count=0
if [[ -d "${incomplete_dir}" ]]; then
  checkpoint_count=$(find "${incomplete_dir}" -maxdepth 1 -name 'checkpoint_pID-*.jld2' | wc -l | tr -d ' ')
  fresh_checkpoint_count=$(find "${incomplete_dir}" -maxdepth 1 -name 'checkpoint_pID-*.jld2' -newer "${job_start_marker}" | wc -l | tr -d ' ')
fi
error_glob="${LOG_DIR}/${SLURM_JOB_NAME:-muL6Probe}_*_${SLURM_JOB_ID:-manual}.err"
problem_count=0
for error_log in ${error_glob}; do
  [[ -f "${error_log}" ]] || continue
  n=$(grep -a -Eiv 'ProcessExited\((13|9)\)' "${error_log}" | grep -a -Eci 'JLD2|MPI_ERRORS_ARE_FATAL|Socket closed|failed to TCP connect|No route to host|drift too large|wrong[- ]rank|ERROR:|LoadError|BoundsError|MethodError|OutOfMemory|StackOverflow' || true)
  problem_count=$((problem_count + n))
done
echo "[$(date -Is)] incomplete rc=${rc} runtime_s=${RUNTIME_SECONDS} checkpoints=${checkpoint_count}/${RANKS} fresh=${fresh_checkpoint_count}/${RANKS} problems=${problem_count} dir=${incomplete_dir}"

checkpoint_stop=false
if [[ "${checkpoint_count}" -eq "${RANKS}" && "${fresh_checkpoint_count}" -eq "${RANKS}" && "${problem_count}" -eq 0 && "${RUNTIME_SECONDS}" -ge "${MIN_RUNTIME_STOP_SECONDS}" ]]; then
  case "${rc}" in 0|1|9|13) checkpoint_stop=true;; esac
fi
if [[ "${AUTO_RESUBMIT}" == "true" && "${checkpoint_stop}" == "true" && "${RESUBMIT_COUNT}" -lt "${MAX_RESUBMITS}" ]]; then
  next_count=$((RESUBMIT_COUNT + 1))
  echo "[$(date -Is)] healthy full checkpoint set; submitting continuation ${next_count}/${MAX_RESUBMITS}"
  sbatch -A "${ACCOUNT}" -p burst --qos=default --job-name="${RESUBMIT_JOB_NAME}" \
    --nodes=1 --ntasks="${RANKS}" --ntasks-per-node="${RANKS}" --cpus-per-task=1 --mem=100G --time="${REQUESTED_WALLTIME}" \
    --export=ALL,MANIFEST="${MANIFEST}",GCE_FAMILY="${GCE_FAMILY}",RESUBMIT_SCRIPT="${RESUBMIT_SCRIPT}",RESUBMIT_COUNT="${next_count}",CHECKPOINT_RESET_ACCUMULATORS=false,SMOKE_MODE="${SMOKE_MODE}" \
    --array="${TASK_ID}" "${RESUBMIT_SCRIPT}"
  exit 0
fi
echo "ERROR: GCE probe incomplete without safe continuation: ${TARGET_KEY} ${PROBE_ROLE} rc=${rc}" >&2
if [[ "${rc}" -eq 0 ]]; then rc=1; fi
exit "${rc}"
