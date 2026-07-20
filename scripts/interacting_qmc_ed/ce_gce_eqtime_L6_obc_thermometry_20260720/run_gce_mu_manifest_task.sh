#!/usr/bin/env bash
# Checkpoint-safe density-only L=6 OBC GCE chemical-potential probe.
set -u -o pipefail

PROJECT=${PROJECT:-/home/9pm/nUHubbard_obc_dev}
WF=${PROJECT}/scripts/interacting_qmc_ed/ce_gce_eqtime_L6_obc_thermometry_20260720
MANIFEST=${MANIFEST:?MANIFEST must point to an OBC GCE probe manifest}
GCE_FAMILY=${GCE_FAMILY:?GCE_FAMILY must be attractive or spinHS}
RESUBMIT_SCRIPT=${RESUBMIT_SCRIPT:?RESUBMIT_SCRIPT is required}
TASK_ID=${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}
cd "${PROJECT}" || exit 2
mkdir -p "${PROJECT}/logs"

row=$(python3 - "${MANIFEST}" "${TASK_ID}" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter="\t")); i=int(sys.argv[2])
if not 0 <= i < len(rows): raise SystemExit(f"task {i} outside 0..{len(rows)-1}")
r=rows[i]
fields=("idx","target_key","family","U_label","U","Ntot_target","target_density","beta","T",
        "mu_L6_PBC_reference","mu_L8_reference","mu_probe","probe_role","Lx","Ly","dtau",
        "ntherm","nmeasurements","nbins","nupdates","account","partition","qos","out_parent",
        "sid","seed","tuning_status","boundary","project_commit","smoqydqmc_version",
        "smoqydqmc_commit","site_count","nn_bond_count","nnn_bond_count")
missing=[k for k in fields if k not in r]
if missing: raise SystemExit(f"manifest missing fields {missing}")
print("\t".join(r[k] for k in fields))
PY
) || exit $?
IFS=$'\t' read -r IDX TARGET_KEY FAMILY U_LABEL U NTOT_TARGET TARGET_DENSITY BETA TARGET_T MU_PBC MU_L8 MU PROBE_ROLE LX LY DTAU N_THERM N_MEASUREMENTS N_BINS N_UPDATES ACCOUNT PARTITION QOS OUT_PARENT SID BASE_SEED TUNING_STATUS BOUNDARY EXPECTED_PROJECT_COMMIT EXPECTED_SMOQY_VERSION EXPECTED_SMOQY_COMMIT SITE_COUNT NN_COUNT NNN_COUNT <<< "${row}"

RANKS=${SLURM_NTASKS:-32}; SMOKE_MODE=${SMOKE_MODE:-false}
[[ "${BOUNDARY}" == open && "${OUT_PARENT}" == *"_obc_"* ]] || { echo "OBC boundary/root guard failed" >&2; exit 2; }
[[ "${FAMILY}" == "${GCE_FAMILY}" ]] || { echo "manifest/wrapper GCE family mismatch" >&2; exit 2; }
[[ "${ACCOUNT}" == ccsd || "${ACCOUNT}" == cnms ]] || { echo "unapproved account" >&2; exit 2; }
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  [[ "${SLURM_JOB_ACCOUNT:-}" == "${ACCOUNT}" ]] || { echo "account mismatch" >&2; exit 2; }
  [[ "${SLURM_JOB_PARTITION:-}" == "${PARTITION}" && "${SLURM_JOB_QOS:-}" == "${QOS}" ]] || { echo "partition/QOS mismatch" >&2; exit 2; }
fi
[[ "${PARTITION}" == burst && "${QOS}" == default ]] || exit 2
if [[ "${SMOKE_MODE}" != true ]]; then
  (( RANKS == 32 && LX == 6 && LY == 6 && SITE_COUNT == 36 && NN_COUNT == 60 && NNN_COUNT == 50 )) || { echo "L6 OBC probe layout guard failed" >&2; exit 2; }
fi
[[ "${CHECKPOINT_RESET_ACCUMULATORS:-false}" == false ]] || { echo "accumulator reset forbidden" >&2; exit 2; }

ACTUAL_PROJECT_COMMIT=$(git rev-parse HEAD)
ACTUAL_SMOQY_COMMIT=$(git -C "${PROJECT}/external/SmoQyDQMC" rev-parse HEAD)
[[ "${ACTUAL_PROJECT_COMMIT}" == "${EXPECTED_PROJECT_COMMIT}" && "${ACTUAL_SMOQY_COMMIT}" == "${EXPECTED_SMOQY_COMMIT}" ]] || { echo "project/dependency commit drift" >&2; exit 2; }
[[ "${EXPECTED_SMOQY_VERSION}" == 2.0.12 ]] || exit 2
case "${GCE_FAMILY}" in
  attractive)
    python3 - "${U}" <<'PY' || exit $?
import sys; assert float(sys.argv[1]) < 0
PY
    DRIVER=${PROJECT}/scripts/interacting_qmc_ed/run_smoqydqmc_attractive_hubbard_checkpoint.jl ;;
  spinHS)
    python3 - "${U}" <<'PY' || exit $?
import sys; assert float(sys.argv[1]) > 0
PY
    DRIVER=${PROJECT}/scripts/interacting_qmc_ed/run_smoqydqmc_hubbard_spin_hs_checkpoint.jl ;;
  *) echo "invalid family ${GCE_FAMILY}" >&2; exit 2;;
esac

AUTO_RESUBMIT=${AUTO_RESUBMIT:-true}; MAX_RESUBMITS=${MAX_RESUBMITS:-80}; RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}
CHECKPOINT_FREQ_HOURS=${CHECKPOINT_FREQ_HOURS:-1.0}; RUNTIME_LIMIT_HOURS=${RUNTIME_LIMIT_HOURS:-1.25}
CHECKPOINT_EVERY_N_MEASUREMENTS=${CHECKPOINT_EVERY_N_MEASUREMENTS:-10}; REQUESTED_WALLTIME=${REQUESTED_WALLTIME:-01:30:00}
MIN_RUNTIME_STOP_SECONDS=${MIN_RUNTIME_STOP_SECONDS:-300}; RESUBMIT_JOB_NAME=${SLURM_JOB_NAME:-muL6OBC}
N_STAB=${N_STAB:-10}; N_STAB_MIN=${N_STAB_MIN:-6}; DG_MAX=${DG_MAX:-1e-5}

unset LD_LIBRARY_PATH MPI_PATH MPI_ROOT MPICC MPICXX MPIF77 MPIF90 MPIFC OMPI_MCA_pml OMPI_MCA_btl || true
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,tcp
export OMPI_MCA_btl_tcp_if_include=${OMPI_MCA_btl_tcp_if_include:-mgmt0}
export OMPI_MCA_oob_tcp_if_include=${OMPI_MCA_oob_tcp_if_include:-mgmt0}
export PRTE_MCA_oob_tcp_if_include=${PRTE_MCA_oob_tcp_if_include:-mgmt0}
export PMIX_MCA_ptl_tcp_if_include=${PMIX_MCA_ptl_tcp_if_include:-mgmt0}
export PATH="${PROJECT}/.julia_depot/bin:$HOME/.julia/bin:$HOME/.juliaup/bin:$PATH"
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-${PROJECT}/.julia_depot:$HOME/.julia}
export JULIA_PROJECT=${JULIA_PROJECT:-${PROJECT}/julia_env}
export JULIA_BINDIR=${JULIA_BINDIR:-/home/9pm/.julia/juliaup/julia-1.12.1+0.x64.linux.gnu/bin}
JULIA_BIN=${JULIA_BIN:-${JULIA_BINDIR}/julia}
export JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE JULIA_PKG_PRECOMPILE_AUTO=0 CHECKPOINT_RESET_ACCUMULATORS=false
ACTUAL_VERSION=$("${JULIA_BIN}" --project="${JULIA_PROJECT}" -e 'using SmoQyDQMC; print(Base.pkgversion(SmoQyDQMC))')
[[ "${ACTUAL_VERSION}" == "${EXPECTED_SMOQY_VERSION}" ]] || { echo "SmoQy version drift" >&2; exit 2; }

complete_dir() { find "${OUT_PARENT}" -mindepth 1 -maxdepth 1 -type d -name 'complete_*_obc_*' -print 2>/dev/null; }
strict_final() {
  local complete count
  complete=$(complete_dir)
  [[ -n "${complete}" && $(printf '%s\n' "${complete}" | wc -l | tr -d ' ') -eq 1 ]] || return 1
  [[ -s "${complete}/global_stats.csv" ]] || return 1
  count=$(find "${complete}" -maxdepth 1 -name 'simulation_info_sID-*_pID-*.toml' | wc -l | tr -d ' ')
  [[ "${count}" -eq "${RANKS}" ]] || return 1
  python3 - "${complete}" "${RANKS}" "${SITE_COUNT}" "${NN_COUNT}" "${NNN_COUNT}" <<'PY'
import pathlib,sys,tomllib
root=pathlib.Path(sys.argv[1]); ranks=int(sys.argv[2]); site=int(sys.argv[3]); nn=int(sys.argv[4]); nnn=int(sys.argv[5])
paths=sorted(root.glob("simulation_info_sID-*_pID-*.toml")); assert len(paths)==ranks
for p in paths:
 d=tomllib.loads(p.read_text())["metadata"]
 assert d["boundary"]=="open" and d["geometry_boundary"]=="open"
 assert d["geometry_site_count"]==site and d["geometry_nn_bond_count"]==nn and d["geometry_nnn_bond_count"]==nnn
 assert d["smoqydqmc_version"]=="2.0.12"
PY
}

write_achieved() {
  local complete
  complete=$(complete_dir)
  python3 - "${complete}/global_stats.csv" "${NTOT_TARGET}" "${SITE_COUNT}" "${OUT_PARENT}/probe_achieved_density.tsv" <<'PY'
import csv,pathlib,sys
path=pathlib.Path(sys.argv[1]); target=int(sys.argv[2]); sites=int(sys.argv[3]); out=pathlib.Path(sys.argv[4])
stats={}
with path.open() as f:
 for r in csv.DictReader(f,delimiter=" ",skipinitialspace=True):
  if r.get("MEASUREMENT"): stats[r["MEASUREMENT"]]=(float(r["MEAN_REAL"]),float(r.get("STD") or 0))
d,de=stats["density"]; n=sites*d; ne=sites*de
out.write_text("target_N\tachieved_N\tachieved_N_err\tdelta_N\tsite_count\n"+f"{target}\t{n:.12f}\t{ne:.12f}\t{n-target:.12f}\t{sites}\n")
print(f"probe achieved N={n:.8f}+/-{ne:.8f} target={target}")
PY
}
if strict_final; then write_achieved; echo "[$(date -Is)] already complete ${TARGET_KEY} ${PROBE_ROLE}"; exit 0; fi

mkdir -p "${OUT_PARENT}"
START=$(date +%s)
marker=${OUT_PARENT}/.job_start_${SLURM_JOB_ID:-manual}_${TASK_ID}_${START}
date > "${marker}"
echo "[$(date -Is)] L6 OBC GCE probe ${TARGET_KEY} role=${PROBE_ROLE} family=${GCE_FAMILY} mu=${MU} PBC_seed=${MU_PBC} ranks=${RANKS} continuation=${RESUBMIT_COUNT}/${MAX_RESUBMITS}"
rc=0
mpiexecjl --project="${JULIA_PROJECT}" -n "${RANKS}" "${JULIA_BIN}" --project="${JULIA_PROJECT}" "${DRIVER}" \
  "${SID}" "${U}" 0.0 "${MU}" "${LX}" "${BETA}" "${N_THERM}" "${N_MEASUREMENTS}" "${N_BINS}" "${N_UPDATES}" \
  "${CHECKPOINT_FREQ_HOURS}" "${RUNTIME_LIMIT_HOURS}" true "${OUT_PARENT}" "${LY}" density-only \
  "${DTAU}" "${N_STAB}" "${DG_MAX}" false true "${N_STAB_MIN}" "${BASE_SEED}" "${CHECKPOINT_EVERY_N_MEASUREMENTS}" --boundary=open || rc=$?

if [[ "${rc}" -eq 0 ]] && strict_final; then write_achieved; echo "[$(date -Is)] complete ${TARGET_KEY} ${PROBE_ROLE}"; exit 0; fi
incomplete=$(find "${OUT_PARENT}" -mindepth 1 -maxdepth 1 -type d ! -name 'complete_*' -print 2>/dev/null | head -1)
checkpoint_count=0; fresh_count=0
if [[ -n "${incomplete}" ]]; then
  checkpoint_count=$(find "${incomplete}" -maxdepth 1 -name 'checkpoint_pID-*.jld2' | wc -l | tr -d ' ')
  fresh_count=$(find "${incomplete}" -maxdepth 1 -name 'checkpoint_pID-*.jld2' -newer "${marker}" | wc -l | tr -d ' ')
fi
elapsed=$(( $(date +%s)-START ))
fresh=false
if [[ "${rc}" -eq 13 && "${checkpoint_count}" -eq "${RANKS}" && "${fresh_count}" -eq "${RANKS}" ]]; then fresh=true; fi
if [[ "${fresh}" == true && "${elapsed}" -ge "${MIN_RUNTIME_STOP_SECONDS}" && "${AUTO_RESUBMIT}" == true && "${RESUBMIT_COUNT}" -lt "${MAX_RESUBMITS}" ]]; then
  next=$((RESUBMIT_COUNT+1))
  echo "[$(date -Is)] healthy OBC probe checkpoint; continuation ${next}/${MAX_RESUBMITS}"
  jid=$(MANIFEST="${MANIFEST}" GCE_FAMILY="${GCE_FAMILY}" RESUBMIT_COUNT="${next}" RESUBMIT_SCRIPT="${RESUBMIT_SCRIPT}" CHECKPOINT_RESET_ACCUMULATORS=false SMOKE_MODE="${SMOKE_MODE}" \
    sbatch --parsable -A "${ACCOUNT}" -p burst --qos=default --job-name="${RESUBMIT_JOB_NAME}" --nodes=1 --ntasks="${RANKS}" \
      --ntasks-per-node="${RANKS}" --cpus-per-task=1 --mem=100G --time="${REQUESTED_WALLTIME}" --array="${TASK_ID}" "${RESUBMIT_SCRIPT}") || exit $?
  jid=${jid%%;*}
  ledger=${OUT_PARENT}/.l6_obc_continuation_ledger.tsv
  [[ -e "${ledger}" ]] || printf 'timestamp\tjob_id\tarray_task\tcontinuation\tjob_name\taccount\tmanifest\n' > "${ledger}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${jid}" "${TASK_ID}" "${next}" "${RESUBMIT_JOB_NAME}" "${ACCOUNT}" "${MANIFEST}" >> "${ledger}"
  echo "[$(date -Is)] submitted continuation job=${jid} task=${TASK_ID} root=${OUT_PARENT}"
  exit 0
fi
echo "ERROR: OBC GCE probe incomplete without safe continuation ${TARGET_KEY} ${PROBE_ROLE} rc=${rc} checkpoints=${checkpoint_count}/${RANKS}" >&2
[[ "${rc}" -ne 0 ]] || rc=1
exit "${rc}"
