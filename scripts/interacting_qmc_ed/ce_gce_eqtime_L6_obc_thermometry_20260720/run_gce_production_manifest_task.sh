#!/usr/bin/env bash
# Checkpoint-safe equal-time L=6 OBC GCE production runner.
set -u -o pipefail

PROJECT=${PROJECT:-/home/9pm/nUHubbard_obc_dev}
WF=${PROJECT}/scripts/interacting_qmc_ed/ce_gce_eqtime_L6_obc_thermometry_20260720
MANIFEST=${MANIFEST:?MANIFEST must point to a confirmed OBC GCE production manifest}
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
        "mu_L6_PBC_reference","mu_L8_reference","mu_fitted","mu_final","confirmation_N",
        "confirmation_N_err","density_tolerance","tuning_status","Lx","Ly","dtau","ntherm",
        "nmeasurements","nbins","nupdates","expected_ranks","account","partition","qos",
        "out_parent","sid","seed","boundary","project_commit","smoqydqmc_version",
        "smoqydqmc_commit","site_count","nn_bond_count","nnn_bond_count")
missing=[k for k in fields if k not in r]
if missing: raise SystemExit(f"manifest missing fields {missing}")
print("\t".join(r[k] for k in fields))
PY
) || exit $?
IFS=$'\t' read -r IDX TARGET_KEY FAMILY U_LABEL U NTOT_TARGET TARGET_DENSITY BETA TARGET_T MU_PBC MU_L8 MU_FITTED MU CONF_N CONF_N_ERR DENSITY_TOL TUNING_STATUS LX LY DTAU N_THERM N_MEASUREMENTS N_BINS N_UPDATES EXPECTED_RANKS ACCOUNT PARTITION QOS OUT_PARENT SID BASE_SEED BOUNDARY EXPECTED_PROJECT_COMMIT EXPECTED_SMOQY_VERSION EXPECTED_SMOQY_COMMIT SITE_COUNT NN_COUNT NNN_COUNT <<< "${row}"

RANKS=${SLURM_NTASKS:-${EXPECTED_RANKS}}
[[ "${BOUNDARY}" == open && "${OUT_PARENT}" == *"_obc_"* ]] || { echo "OBC boundary/root guard failed" >&2; exit 2; }
[[ "${FAMILY}" == "${GCE_FAMILY}" ]] || { echo "GCE family mismatch" >&2; exit 2; }
[[ "${TUNING_STATUS}" == confirmed_within_abs_N_0p03 ]] || { echo "production row lacks accepted OBC confirmation" >&2; exit 2; }
(( RANKS == EXPECTED_RANKS && RANKS == 32 && LX == 6 && LY == 6 && N_UPDATES == 3 )) || { echo "production layout guard failed" >&2; exit 2; }
(( SITE_COUNT == 36 && NN_COUNT == 60 && NNN_COUNT == 50 )) || exit 2
[[ "${ACCOUNT}" == ccsd || "${ACCOUNT}" == cnms ]] || exit 2
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  [[ "${SLURM_JOB_ACCOUNT:-}" == "${ACCOUNT}" ]] || { echo "account mismatch" >&2; exit 2; }
  [[ "${SLURM_JOB_PARTITION:-}" == "${PARTITION}" && "${SLURM_JOB_QOS:-}" == "${QOS}" ]] || { echo "partition/QOS mismatch" >&2; exit 2; }
fi
[[ "${PARTITION}" == burst && "${QOS}" == default && "${CHECKPOINT_RESET_ACCUMULATORS:-false}" == false ]] || exit 2

ACTUAL_PROJECT_COMMIT=$(git rev-parse HEAD); ACTUAL_SMOQY_COMMIT=$(git -C "${PROJECT}/external/SmoQyDQMC" rev-parse HEAD)
[[ "${ACTUAL_PROJECT_COMMIT}" == "${EXPECTED_PROJECT_COMMIT}" && "${ACTUAL_SMOQY_COMMIT}" == "${EXPECTED_SMOQY_COMMIT}" && "${EXPECTED_SMOQY_VERSION}" == 2.0.12 ]] || { echo "project/dependency drift" >&2; exit 2; }
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
 *) exit 2;;
esac

AUTO_RESUBMIT=${AUTO_RESUBMIT:-true}; MAX_RESUBMITS=${MAX_RESUBMITS:-120}; RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}
CHECKPOINT_FREQ_HOURS=${CHECKPOINT_FREQ_HOURS:-1.0}; RUNTIME_LIMIT_HOURS=${RUNTIME_LIMIT_HOURS:-1.25}
CHECKPOINT_EVERY_N_MEASUREMENTS=${CHECKPOINT_EVERY_N_MEASUREMENTS:-10}; REQUESTED_WALLTIME=${REQUESTED_WALLTIME:-01:30:00}
MIN_RUNTIME_STOP_SECONDS=${MIN_RUNTIME_STOP_SECONDS:-300}; RESUBMIT_JOB_NAME=${SLURM_JOB_NAME:-gceL6OBC}
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
[[ "${ACTUAL_VERSION}" == "${EXPECTED_SMOQY_VERSION}" ]] || exit 2

complete_dir() { find "${OUT_PARENT}" -mindepth 1 -maxdepth 1 -type d -name 'complete_*_obc_*' -print 2>/dev/null; }
required=(equal_time_kinetic_per_site_qmc.tsv equal_time_double_occupancy_per_site_qmc.tsv equal_time_nn_spin_qmc.tsv equal_time_nn_connected_charge_qmc.tsv)
strict_final() {
  [[ -f "${OUT_PARENT}/dqmc_gce_obc_eqtime_complete.txt" ]] || return 1
  local complete count f
  complete=$(complete_dir)
  [[ -n "${complete}" && $(printf '%s\n' "${complete}" | wc -l | tr -d ' ') -eq 1 ]] || return 1
  count=$(find "${complete}" -maxdepth 1 -name 'obc_equal_time_rank_pID-*.tsv' | wc -l | tr -d ' ')
  [[ "${count}" -eq "${RANKS}" ]] || return 1
  count=$(find "${complete}" -maxdepth 1 -name 'obc_equal_time_site_rank_pID-*.tsv' | wc -l | tr -d ' ')
  [[ "${count}" -eq "${RANKS}" ]] || return 1
  for f in "${required[@]}"; do [[ -s "${complete}/${f}" ]] || return 1; done
  [[ -s "${OUT_PARENT}/achieved_density.tsv" ]]
}
if strict_final; then echo "[$(date -Is)] already strict-final ${TARGET_KEY}"; exit 0; fi

mkdir -p "${OUT_PARENT}"
START=$(date +%s); marker=${OUT_PARENT}/.job_start_${SLURM_JOB_ID:-manual}_${TASK_ID}_${START}; date > "${marker}"
echo "[$(date -Is)] L6 OBC GCE production ${TARGET_KEY} family=${GCE_FAMILY} mu=${MU} confirmation=${CONF_N}+/-${CONF_N_ERR} continuation=${RESUBMIT_COUNT}/${MAX_RESUBMITS}"
rc=0
mpiexecjl --project="${JULIA_PROJECT}" -n "${RANKS}" "${JULIA_BIN}" --project="${JULIA_PROJECT}" "${DRIVER}" \
  "${SID}" "${U}" 0.0 "${MU}" "${LX}" "${BETA}" "${N_THERM}" "${N_MEASUREMENTS}" "${N_BINS}" "${N_UPDATES}" \
  "${CHECKPOINT_FREQ_HOURS}" "${RUNTIME_LIMIT_HOURS}" true "${OUT_PARENT}" "${LY}" equal-time-only \
  "${DTAU}" "${N_STAB}" "${DG_MAX}" false true "${N_STAB_MIN}" "${BASE_SEED}" "${CHECKPOINT_EVERY_N_MEASUREMENTS}" --boundary=open || rc=$?

complete=$(complete_dir)
if [[ "${rc}" -eq 0 && -n "${complete}" ]]; then
  python3 - "${complete}" "${NTOT_TARGET}" "${DENSITY_TOL}" "${OUT_PARENT}" "${RANKS}" <<'PY' || exit $?
import csv,pathlib,sys,tomllib
root=pathlib.Path(sys.argv[1]); target=int(sys.argv[2]); tol=float(sys.argv[3]); parent=pathlib.Path(sys.argv[4]); ranks=int(sys.argv[5])
summary=list(csv.DictReader(open(root/"equal_time_observables_obc_qmc.tsv"),delimiter="\t"))[0]
n=float(summary["achieved_N"]); delta=n-target
site_files=sorted(root.glob("obc_equal_time_site_rank_pID-*.tsv")); assert len(site_files)==ranks
infos=sorted(root.glob("simulation_info_sID-*_pID-*.toml")); assert len(infos)==ranks
for p in infos:
 d=tomllib.loads(p.read_text())["metadata"]
 assert d["boundary"]=="open" and d["geometry_site_count"]==36 and d["geometry_nn_bond_count"]==60 and d["geometry_nnn_bond_count"]==50
parent.joinpath("achieved_density.tsv").write_text("target_N\tachieved_N\tdelta_N\ttolerance\n"+f"{target}\t{n:.12f}\t{delta:.12f}\t{tol:.12f}\n")
if abs(delta)>tol:
 parent.joinpath("density_tolerance_failed.txt").write_text(f"target={target}\nachieved={n}\ndelta={delta}\ntolerance={tol}\n")
 raise SystemExit(42)
for name in ("equal_time_kinetic_per_site_qmc.tsv","equal_time_double_occupancy_per_site_qmc.tsv","equal_time_nn_spin_qmc.tsv","equal_time_nn_connected_charge_qmc.tsv"):
 r=list(csv.DictReader(open(root/name),delimiter="\t")); assert len(r)==1 and r[0]["boundary"]=="open" and int(r[0]["nranks"])==ranks
PY
  bad=$(find "${complete}" -type f \( -path '*/pair/*' -o -path '*/greens/*' -o -path '*/current/*' -o -path '*/time-displaced/*' -o -name 'bkt_observables_qmc.tsv' -o -name 'equal_time_structure_factors_qmc.tsv' -o -name 'equal_time_charge_spin_wedge_qmc.tsv' \) | wc -l | tr -d ' ')
  [[ "${bad}" -eq 0 ]] || { echo "forbidden OBC translational/momentum outputs found" >&2; exit 2; }
  touch "${OUT_PARENT}/dqmc_gce_obc_eqtime_complete.txt"
  strict_final || { echo "strict-final validation failed" >&2; exit 2; }
  echo "[$(date -Is)] strict-final ${TARGET_KEY}"; exit 0
fi

incomplete=$(find "${OUT_PARENT}" -mindepth 1 -maxdepth 1 -type d ! -name 'complete_*' -print 2>/dev/null | head -1)
checkpoint_count=0; fresh_count=0
if [[ -n "${incomplete}" ]]; then
 checkpoint_count=$(find "${incomplete}" -maxdepth 1 -name 'checkpoint_pID-*.jld2' | wc -l | tr -d ' ')
 fresh_count=$(find "${incomplete}" -maxdepth 1 -name 'checkpoint_pID-*.jld2' -newer "${marker}" | wc -l | tr -d ' ')
fi
elapsed=$(( $(date +%s)-START )); fresh=false
if [[ "${rc}" -eq 13 && "${checkpoint_count}" -eq "${RANKS}" && "${fresh_count}" -eq "${RANKS}" ]]; then fresh=true; fi
if [[ "${fresh}" == true && "${elapsed}" -ge "${MIN_RUNTIME_STOP_SECONDS}" && "${AUTO_RESUBMIT}" == true && "${RESUBMIT_COUNT}" -lt "${MAX_RESUBMITS}" ]]; then
 next=$((RESUBMIT_COUNT+1))
 jid=$(MANIFEST="${MANIFEST}" GCE_FAMILY="${GCE_FAMILY}" RESUBMIT_COUNT="${next}" RESUBMIT_SCRIPT="${RESUBMIT_SCRIPT}" CHECKPOINT_RESET_ACCUMULATORS=false \
  sbatch --parsable -A "${ACCOUNT}" -p burst --qos=default --job-name="${RESUBMIT_JOB_NAME}" --nodes=1 --ntasks=32 --ntasks-per-node=32 \
    --cpus-per-task=1 --mem=100G --time="${REQUESTED_WALLTIME}" --array="${TASK_ID}" "${RESUBMIT_SCRIPT}") || exit $?
 jid=${jid%%;*}
 ledger=${OUT_PARENT}/.l6_obc_continuation_ledger.tsv
 [[ -e "${ledger}" ]] || printf 'timestamp\tjob_id\tarray_task\tcontinuation\tjob_name\taccount\tmanifest\n' > "${ledger}"
 printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${jid}" "${TASK_ID}" "${next}" "${RESUBMIT_JOB_NAME}" "${ACCOUNT}" "${MANIFEST}" >> "${ledger}"
 echo "[$(date -Is)] submitted continuation job=${jid} task=${TASK_ID} root=${OUT_PARENT}"
 exit 0
fi
echo "ERROR: OBC GCE production incomplete without safe continuation ${TARGET_KEY} rc=${rc} checkpoints=${checkpoint_count}/${RANKS}" >&2
[[ "${rc}" -ne 0 ]] || rc=1
exit "${rc}"
