#!/usr/bin/env bash
# Shared checkpoint/resume and strict-export runner for tuned L=6 GCE production.

set -u -o pipefail
PROJECT=${PROJECT:-/home/9pm/nUHubbard}
MANIFEST=${MANIFEST:?MANIFEST must point to a confirmed L6 GCE production TSV}
GCE_FAMILY=${GCE_FAMILY:?GCE_FAMILY must be attractive or spinHS}
RESUBMIT_SCRIPT=${RESUBMIT_SCRIPT:?RESUBMIT_SCRIPT must point to a thin Slurm wrapper}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
LOG_DIR=${PROJECT}/logs
cd "${PROJECT}" || exit 2
mkdir -p "${LOG_DIR}"

row_tsv=$(python3 - "${MANIFEST}" "${TASK_ID}" <<'PY'
import csv, sys
path, task = sys.argv[1], int(sys.argv[2])
rows=list(csv.DictReader(open(path), delimiter="\t"))
if not 0 <= task < len(rows): raise SystemExit(f"task {task} out of range 0..{len(rows)-1}")
r=rows[task]
fields=[
 "idx","U_label","U","Ntot_target","target_density","beta","T","mu_L8_reference",
 "L8_reference_Ntot","L8_reference_density","probe_bracket","mu_bracket_low",
 "density_bracket_low","mu_bracket_high","density_bracket_high","mu_fitted","mu_final",
 "mu_label","confirmation_density","confirmation_density_err","confirmation_N","confirmation_N_err",
 "density_tolerance","tuning_status","source_manifests","Lx","Ly","dtau","ntherm",
 "nmeasurements","nbins","nupdates","expected_ranks","account","out_parent","sid","seed",
 "target_key","job_tag",
]
missing=[k for k in fields if k not in r]
if missing: raise SystemExit(f"manifest missing fields: {missing}")
print("\t".join(r[k] for k in fields))
PY
) || exit $?
IFS=$'\t' read -r IDX U_LABEL U NTOT_TARGET TARGET_DENSITY BETA T_TARGET MU_L8_REFERENCE L8_REFERENCE_NTOT L8_REFERENCE_DENSITY PROBE_BRACKET MU_BRACKET_LOW DENSITY_BRACKET_LOW MU_BRACKET_HIGH DENSITY_BRACKET_HIGH MU_FITTED MU MU_LABEL CONFIRMATION_DENSITY CONFIRMATION_DENSITY_ERR CONFIRMATION_N CONFIRMATION_N_ERR DENSITY_TOLERANCE TUNING_STATUS SOURCE_MANIFESTS LX LY DTAU N_THERM N_MEASUREMENTS N_BINS N_UPDATES EXPECTED_RANKS ACCOUNT OUT_PARENT SID BASE_SEED TARGET_KEY JOB_TAG <<< "${row_tsv}"

RANKS=${SLURM_NTASKS:-${EXPECTED_RANKS}}
if [[ "${ACCOUNT}" != "ccsd" || "${SLURM_JOB_ACCOUNT:-ccsd}" != "ccsd" ]]; then echo "ERROR: production is pinned to ccsd" >&2; exit 2; fi
if [[ "${SLURM_JOB_PARTITION:-burst}" != "burst" || "${SLURM_JOB_QOS:-default}" != "default" ]]; then echo "ERROR: production requires burst/default" >&2; exit 2; fi
if (( RANKS != EXPECTED_RANKS || RANKS != 32 )); then echo "ERROR: wrong rank count ${RANKS}; expected ${EXPECTED_RANKS}=32" >&2; exit 2; fi
if (( LX != 6 || LY != 6 || N_UPDATES != 3 )); then echo "ERROR: expected L=6 and measurement interval 3" >&2; exit 2; fi
if [[ "${TUNING_STATUS}" != "confirmed_within_abs_N_0p03" ]]; then echo "ERROR: production row lacks accepted confirmation" >&2; exit 2; fi
if [[ "${CHECKPOINT_RESET_ACCUMULATORS:-false}" != "false" ]]; then echo "ERROR: accumulator reset is forbidden" >&2; exit 2; fi

case "${GCE_FAMILY}" in
 attractive)
   python3 - "${U}" <<'PY' || exit $?
import sys; assert float(sys.argv[1]) < 0
PY
   DRIVER=${PROJECT}/scripts/interacting_qmc_ed/run_smoqydqmc_attractive_hubbard_checkpoint.jl
   RUN_PREFIX=attractive_hubbard_rect
   UPDATE_STABILIZATION_FREQUENCY=${UPDATE_STABILIZATION_FREQUENCY:-true}
   ;;
 spinHS)
   python3 - "${U}" <<'PY' || exit $?
import sys; assert float(sys.argv[1]) > 0
PY
   DRIVER=${PROJECT}/scripts/interacting_qmc_ed/run_smoqydqmc_hubbard_spin_hs_checkpoint.jl
   RUN_PREFIX=hubbard_spin_hs_rect
   UPDATE_STABILIZATION_FREQUENCY=${UPDATE_STABILIZATION_FREQUENCY:-true}
   ;;
 *) echo "ERROR: invalid GCE_FAMILY=${GCE_FAMILY}" >&2; exit 2;;
esac

TPRIME=0.0
PH_SYM_FORM=true
MEASUREMENT_PROFILE=equal-time-only
N_STAB=${N_STAB:-10}; N_STAB_MIN=${N_STAB_MIN:-6}; DG_MAX=${DG_MAX:-1e-5}
USE_REFLECTION_UPDATE=${USE_REFLECTION_UPDATE:-false}
CHECKPOINT_FREQ_HOURS=${CHECKPOINT_FREQ_HOURS:-1.0}
RUNTIME_LIMIT_HOURS=${RUNTIME_LIMIT_HOURS:-1.25}
CHECKPOINT_EVERY_N_MEASUREMENTS=${CHECKPOINT_EVERY_N_MEASUREMENTS:-10}
REQUESTED_WALLTIME=${REQUESTED_WALLTIME:-01:30:00}
AUTO_RESUBMIT=${AUTO_RESUBMIT:-true}; MAX_RESUBMITS=${MAX_RESUBMITS:-120}; RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}
RESUBMIT_JOB_NAME=${SLURM_JOB_NAME:-gceL6Eq}; MIN_RUNTIME_STOP_SECONDS=${MIN_RUNTIME_STOP_SECONDS:-300}
export CHECKPOINT_RESET_ACCUMULATORS=false

unset LD_LIBRARY_PATH MPI_PATH MPI_ROOT MPICC MPICXX MPIF77 MPIF90 MPIFC || true
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,tcp
export OMPI_MCA_btl_tcp_if_include=mgmt0 OMPI_MCA_oob_tcp_if_include=mgmt0 PRTE_MCA_oob_tcp_if_include=mgmt0 PMIX_MCA_ptl_tcp_if_include=mgmt0
export PATH="$HOME/.julia/bin:$HOME/.juliaup/bin:$PATH"
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-${PROJECT}/.julia_depot:$HOME/.julia}
export JULIA_PROJECT=${JULIA_PROJECT:-${PROJECT}/julia_env}
export JULIA_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE
mkdir -p "${OUT_PARENT}"

run_base=$(python3 - "${RUN_PREFIX}" "${SID}" "${U}" "${MU}" "${LX}" "${LY}" "${BETA}" <<'PY'
import sys
p,sid,u,mu,lx,ly,b=sys.argv[1:]
print(f"{p}_U{float(u):.2f}_tp0.00_mu{float(mu):.2f}_Lx{int(lx)}_Ly{int(ly)}_b{float(b):.2f}-{int(sid)}")
PY
)
complete_dir=${OUT_PARENT}/complete_${run_base}
incomplete_dir=${OUT_PARENT}/${run_base}
export_dir=${OUT_PARENT}/export
required_tables=(equal_time_observables_qmc.tsv equal_time_charge_spin_wedge_qmc.tsv equal_time_structure_factors_qmc.tsv equal_time_neighbor_shells_qmc.tsv)
strict_final() {
  [[ -f "${OUT_PARENT}/dqmc_gce_eqtime_complete.txt" ]] || return 1
  local f; for f in "${required_tables[@]}"; do [[ -f "${export_dir}/${f}" ]] || return 1; done
  [[ -d "${complete_dir}" ]] || return 1
  [[ $(find "${complete_dir}" -maxdepth 1 -name 'simulation_info_sID-*_pID-*.toml' | wc -l | tr -d ' ') -eq "${RANKS}" ]]
}
if strict_final; then echo "[$(date -Is)] already strict-final: ${OUT_PARENT}"; exit 0; fi

START_EPOCH=$(date +%s)
job_start_marker=${OUT_PARENT}/.checkpoint_job_start_${SLURM_JOB_ID:-manual}_${TASK_ID}_${START_EPOCH}.marker
date > "${job_start_marker}"
echo "[$(date -Is)] L6 GCE production family=${GCE_FAMILY} task=${TASK_ID} job=${SLURM_JOB_ID:-manual} target=${TARGET_KEY}"
echo "U=${U} beta=${BETA} Ntarget=${NTOT_TARGET} mu8=${MU_L8_REFERENCE} mu_fit=${MU_FITTED} mu=${MU} confirmation_N=${CONFIRMATION_N}+/-${CONFIRMATION_N_ERR} ranks=${RANKS} warmups=${N_THERM} measurements/rank=${N_MEASUREMENTS} interval=${N_UPDATES} continuation=${RESUBMIT_COUNT}/${MAX_RESUBMITS}"
sha256sum "${DRIVER}" "${PROJECT}/scripts/interacting_qmc_ed/export_dqmc_equal_time_observables.py" || exit 2

rc=0
if [[ ! -d "${complete_dir}" ]]; then
  mpiexecjl -n "${RANKS}" julia --project="${JULIA_PROJECT}" "${DRIVER}" \
    "${SID}" "${U}" "${TPRIME}" "${MU}" "${LX}" "${BETA}" \
    "${N_THERM}" "${N_MEASUREMENTS}" "${N_BINS}" "${N_UPDATES}" \
    "${CHECKPOINT_FREQ_HOURS}" "${RUNTIME_LIMIT_HOURS}" "${PH_SYM_FORM}" "${OUT_PARENT}" \
    "${LY}" "${MEASUREMENT_PROFILE}" "${DTAU}" "${N_STAB}" "${DG_MAX}" \
    "${USE_REFLECTION_UPDATE}" "${UPDATE_STABILIZATION_FREQUENCY}" "${N_STAB_MIN}" \
    "${BASE_SEED}" "${CHECKPOINT_EVERY_N_MEASUREMENTS}" || rc=$?
fi

END_EPOCH=$(date +%s); RUNTIME_SECONDS=$((END_EPOCH - START_EPOCH))
if [[ ! -d "${complete_dir}" ]]; then
  checkpoint_count=0; fresh_checkpoint_count=0
  if [[ -d "${incomplete_dir}" ]]; then
    checkpoint_count=$(find "${incomplete_dir}" -maxdepth 1 -name 'checkpoint_pID-*.jld2' | wc -l | tr -d ' ')
    fresh_checkpoint_count=$(find "${incomplete_dir}" -maxdepth 1 -name 'checkpoint_pID-*.jld2' -newer "${job_start_marker}" | wc -l | tr -d ' ')
  fi
  problem_count=0
  for error_log in ${LOG_DIR}/${SLURM_JOB_NAME:-gceL6Eq}_*_${SLURM_JOB_ID:-manual}.err; do
    [[ -f "${error_log}" ]] || continue
    n=$(grep -a -Eiv 'ProcessExited\((13|9)\)' "${error_log}" | grep -a -Eci 'JLD2|MPI_ERRORS_ARE_FATAL|Socket closed|failed to TCP connect|No route to host|drift too large|wrong[- ]rank|ERROR:|LoadError|BoundsError|MethodError|OutOfMemory|StackOverflow' || true)
    problem_count=$((problem_count+n))
  done
  echo "[$(date -Is)] incomplete rc=${rc} runtime_s=${RUNTIME_SECONDS} checkpoints=${checkpoint_count}/${RANKS} fresh=${fresh_checkpoint_count}/${RANKS} problems=${problem_count}"
  checkpoint_stop=false
  if [[ "${checkpoint_count}" -eq "${RANKS}" && "${fresh_checkpoint_count}" -eq "${RANKS}" && "${problem_count}" -eq 0 && "${RUNTIME_SECONDS}" -ge "${MIN_RUNTIME_STOP_SECONDS}" ]]; then
    case "${rc}" in 0|1|9|13) checkpoint_stop=true;; esac
  fi
  if [[ "${AUTO_RESUBMIT}" == true && "${checkpoint_stop}" == true && "${RESUBMIT_COUNT}" -lt "${MAX_RESUBMITS}" ]]; then
    next_count=$((RESUBMIT_COUNT+1))
    sbatch -A ccsd -p burst --qos=default --job-name="${RESUBMIT_JOB_NAME}" \
      --nodes=1 --ntasks=32 --ntasks-per-node=32 --cpus-per-task=1 --mem=100G --time="${REQUESTED_WALLTIME}" \
      --export=ALL,MANIFEST="${MANIFEST}",GCE_FAMILY="${GCE_FAMILY}",RESUBMIT_SCRIPT="${RESUBMIT_SCRIPT}",RESUBMIT_COUNT="${next_count}",CHECKPOINT_RESET_ACCUMULATORS=false \
      --array="${TASK_ID}" "${RESUBMIT_SCRIPT}"
    exit 0
  fi
  echo "ERROR: GCE production incomplete without safe continuation rc=${rc}" >&2
  if [[ "${rc}" -eq 0 ]]; then rc=1; fi
  exit "${rc}"
fi

rank_info_count=$(find "${complete_dir}" -maxdepth 1 -name 'simulation_info_sID-*_pID-*.toml' | wc -l | tr -d ' ')
if [[ "${rank_info_count}" -ne "${RANKS}" ]]; then
  echo "ERROR: complete directory has wrong rank metadata coverage ${rank_info_count}/${RANKS}" >&2; exit 2
fi
python3 "${PROJECT}/scripts/interacting_qmc_ed/export_dqmc_equal_time_observables.py" \
  --datafolder "${OUT_PARENT}" --outdir "${export_dir}" --u "${U}" --beta "${BETA}" \
  --mu "${MU}" --dtau "${DTAU}" --lx "${LX}" --ly "${LY}" --overwrite || exit $?
for f in "${required_tables[@]}"; do [[ -s "${export_dir}/${f}" ]] || { echo "ERROR: missing export ${f}" >&2; exit 2; }; done

python3 - "${complete_dir}/global_stats.csv" "${NTOT_TARGET}" "${DENSITY_TOLERANCE}" "${OUT_PARENT}" <<'PY'
import csv, math, pathlib, sys
path, target, tol, root = pathlib.Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), pathlib.Path(sys.argv[4])
if not path.is_file(): raise SystemExit(f"missing {path}")
stats={}
with path.open() as f:
    for r in csv.DictReader(f, delimiter=" ", skipinitialspace=True):
        if r.get("MEASUREMENT"): stats[r["MEASUREMENT"]]=(float(r["MEAN_REAL"]),float(r.get("STD") or 0))
d,de=stats["density"]; achieved=36*d; err=36*de; delta=achieved-target
(root/"achieved_density.tsv").write_text("target_N\tachieved_N\tachieved_N_err\tdelta_N\ttolerance\n"+f"{target}\t{achieved:.12f}\t{err:.12f}\t{delta:.12f}\t{tol:.12f}\n")
print(f"achieved_N={achieved:.12f} +/- {err:.12f}; target={target}; delta={delta:.12f}; tolerance={tol}")
if abs(delta) > tol:
    (root/"density_tolerance_failed.txt").write_text(f"target={target}\nachieved={achieved:.12f}\ndelta={delta:.12f}\ntolerance={tol}\n")
    raise SystemExit(42)
PY
validation_rc=$?
if [[ "${validation_rc}" -ne 0 ]]; then exit "${validation_rc}"; fi

bad_count=$(find "${OUT_PARENT}" -type f \( -path '*/pair/*' -o -path '*/greens/*' -o -path '*/current/*' -o -path '*/d-wave/*' -o -path '*/time-displaced/*' -o -path '*/integrated/*' -o -name bkt_observables_qmc.tsv \) | wc -l | tr -d ' ')
if [[ "${bad_count}" -ne 0 ]]; then echo "ERROR: found ${bad_count} forbidden non-equal-time outputs" >&2; exit 2; fi
touch "${OUT_PARENT}/dqmc_gce_eqtime_complete.txt"
strict_final || { echo "ERROR: strict-final validation failed after export" >&2; exit 2; }
echo "[$(date -Is)] strict-final GCE production: ${OUT_PARENT}"
