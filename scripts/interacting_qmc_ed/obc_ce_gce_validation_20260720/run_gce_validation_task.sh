#!/usr/bin/env bash
# Checkpoint-safe per-row runner for the GCE OBC/ED validation matrix.
set -u -o pipefail

PROJECT=${PROJECT:-/home/9pm/nUHubbard_obc_dev}
WORKFLOW=${PROJECT}/scripts/interacting_qmc_ed/obc_ce_gce_validation_20260720
MANIFEST=${MANIFEST:-${WORKFLOW}/manifests/gce_validation_manifest.tsv}
TASK_ID=${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}

row=$(python3 - "${MANIFEST}" "${TASK_ID}" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter="\t")); i=int(sys.argv[2])
if not 0 <= i < len(rows): raise SystemExit(f"task {i} outside 0..{len(rows)-1}")
r=rows[i]
fields=("idx","run_id","L","U","beta","dtau","target_N","seed","expected_ranks",
        "warmups","measurements_per_rank","measurement_interval","family","mu_ph_symmetric",
        "density_tolerance","sid","out_parent","smoqydqmc_version","smoqydqmc_commit")
print("\t".join(r[k] for k in fields))
PY
) || exit $?
IFS=$'\t' read -r IDX RUN_ID L U BETA DTAU TARGET_N SEED EXPECTED_RANKS WARMUPS MEASUREMENTS INTERVAL FAMILY MU DENSITY_TOL SID OUT_PARENT SMOQY_VERSION SMOQY_COMMIT <<< "${row}"

RANKS=${SLURM_NTASKS:-${EXPECTED_RANKS}}
[[ "${RANKS}" -eq "${EXPECTED_RANKS}" ]] || { echo "wrong rank count ${RANKS}/${EXPECTED_RANKS}" >&2; exit 2; }
[[ "${OUT_PARENT}" == *"_obc_"* ]] || { echo "OBC root lacks _obc_: ${OUT_PARENT}" >&2; exit 2; }
ACTUAL_SMOQY_COMMIT=$(git -C "${PROJECT}/external/SmoQyDQMC" rev-parse HEAD)
[[ "${ACTUAL_SMOQY_COMMIT}" == "${SMOQY_COMMIT}" ]] || {
  echo "SmoQyDQMC commit drift: ${ACTUAL_SMOQY_COMMIT} != ${SMOQY_COMMIT}" >&2; exit 2;
}
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  [[ "${SLURM_JOB_ACCOUNT:-}" == ccsd ]] || { echo "GCE validation must remain on ccsd" >&2; exit 2; }
  [[ "${SLURM_JOB_PARTITION:-}" == burst && "${SLURM_JOB_QOS:-}" == default ]] || { echo "GCE validation requires burst/default" >&2; exit 2; }
fi
case "${FAMILY}" in
  attractive) DRIVER=${PROJECT}/scripts/interacting_qmc_ed/run_smoqydqmc_attractive_hubbard_checkpoint.jl ;;
  spin_hs) DRIVER=${PROJECT}/scripts/interacting_qmc_ed/run_smoqydqmc_hubbard_spin_hs_checkpoint.jl ;;
  *) echo "invalid GCE family ${FAMILY}" >&2; exit 2;;
esac

required=(equal_time_kinetic_per_site_qmc.tsv equal_time_double_occupancy_per_site_qmc.tsv equal_time_nn_spin_qmc.tsv equal_time_nn_connected_charge_qmc.tsv)
complete_dir() { find "${OUT_PARENT}" -mindepth 1 -maxdepth 1 -type d -name 'complete_*' -print 2>/dev/null; }
strict_final() {
  local complete count f
  complete=$(complete_dir)
  [[ -n "${complete}" && $(printf '%s\n' "${complete}" | wc -l | tr -d ' ') -eq 1 ]] || return 1
  count=$(find "${complete}" -maxdepth 1 -name 'obc_equal_time_rank_pID-*.tsv' | wc -l | tr -d ' ')
  [[ "${count}" -eq "${EXPECTED_RANKS}" ]] || return 1
  count=$(find "${complete}" -maxdepth 1 -name 'obc_equal_time_site_rank_pID-*.tsv' | wc -l | tr -d ' ')
  [[ "${count}" -eq "${EXPECTED_RANKS}" ]] || return 1
  for f in "${required[@]}"; do [[ -s "${complete}/${f}" ]] || return 1; done
  python3 - "${complete}" "${EXPECTED_RANKS}" <<'PY'
import csv,pathlib,sys
root=pathlib.Path(sys.argv[1]); ranks=int(sys.argv[2])
for name in ("equal_time_kinetic_per_site_qmc.tsv","equal_time_double_occupancy_per_site_qmc.tsv",
             "equal_time_nn_spin_qmc.tsv","equal_time_nn_connected_charge_qmc.tsv"):
    rows=list(csv.DictReader(open(root/name),delimiter="\t"))
    assert len(rows)==1 and rows[0]["boundary"]=="open" and int(rows[0]["nranks"])==ranks
d=list(csv.DictReader(open(root/"equal_time_observables_obc_qmc.tsv"),delimiter="\t"))[0]
assert float(d["achieved_N"]) == float(d["achieved_N"])
PY
}
if strict_final; then echo "[$(date -Is)] already strict-final ${RUN_ID}"; exit 0; fi

mkdir -p "${OUT_PARENT}" "${PROJECT}/logs"
cd "${PROJECT}" || exit 2
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
export JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE JULIA_PKG_PRECOMPILE_AUTO=0
export CHECKPOINT_RESET_ACCUMULATORS=false
ACTUAL_SMOQY_VERSION=$("${JULIA_BIN}" --project="${JULIA_PROJECT}" -e 'using SmoQyDQMC; print(Base.pkgversion(SmoQyDQMC))')
[[ "${ACTUAL_SMOQY_VERSION}" == "${SMOQY_VERSION}" ]] || {
  echo "SmoQyDQMC version drift: ${ACTUAL_SMOQY_VERSION} != ${SMOQY_VERSION}" >&2; exit 2;
}

N_BINS=100
START=$(date +%s)
echo "[$(date -Is)] GCE OBC validation task=${TASK_ID} ${RUN_ID} ranks=${RANKS} U=${U} beta=${BETA} dtau=${DTAU} targetN=${TARGET_N} mu=${MU} seed=${SEED}"
rc=0
mpiexecjl --project="${JULIA_PROJECT}" -n "${RANKS}" "${JULIA_BIN}" --project="${JULIA_PROJECT}" "${DRIVER}" \
  "${SID}" "${U}" 0.0 "${MU}" "${L}" "${BETA}" "${WARMUPS}" "${MEASUREMENTS}" \
  "${N_BINS}" "${INTERVAL}" 1.0 1.25 true "${OUT_PARENT}" "${L}" equal-time-only \
  "${DTAU}" 10 1e-6 false true 1 "${SEED}" 100 --boundary=open || rc=$?

if [[ "${rc}" -eq 0 ]] && strict_final; then
  touch "${OUT_PARENT}/obc_validation_complete.txt"
  echo "[$(date -Is)] strict-final ${RUN_ID}"
  exit 0
fi

incomplete=$(find "${OUT_PARENT}" -mindepth 1 -maxdepth 1 -type d ! -name 'complete_*' -print 2>/dev/null | head -1)
checkpoint_count=0
[[ -n "${incomplete}" ]] && checkpoint_count=$(find "${incomplete}" -maxdepth 1 -name 'checkpoint_pID-*.jld2' | wc -l | tr -d ' ')
elapsed=$(( $(date +%s) - START ))
fresh_runtime_checkpoints=false
if [[ "${rc}" -eq 13 && "${checkpoint_count}" -eq "${EXPECTED_RANKS}" && -n "${incomplete}" ]]; then
  if python3 - "${incomplete}" "${EXPECTED_RANKS}" "${START}" <<'PY'
import pathlib, sys
root=pathlib.Path(sys.argv[1]); expected=int(sys.argv[2]); started=float(sys.argv[3])
checkpoints=sorted(root.glob("checkpoint_pID-*.jld2"))
assert len(checkpoints)==expected
assert all(p.stat().st_size > 0 and p.stat().st_mtime >= started for p in checkpoints)
PY
  then fresh_runtime_checkpoints=true; fi
fi
if [[ "${fresh_runtime_checkpoints}" == true && "${RESUBMIT_COUNT:-0}" -lt "${MAX_RESUBMITS:-20}" && "${AUTO_RESUBMIT:-true}" == true && -n "${SBATCH_SCRIPT:-}" ]]; then
  next=$(( ${RESUBMIT_COUNT:-0} + 1 ))
  echo "[$(date -Is)] checkpointed continuation ${next} for ${RUN_ID}; rc=${rc} elapsed=${elapsed}s"
  # Do not use sbatch --export on CADES: it imports site OpenMPI variables
  # that conflict with Julia's MPI.jl runtime.  Prefix assignments are
  # inherited by sbatch under the site's default export policy.
  MANIFEST="${MANIFEST}" RESUBMIT_COUNT="${next}" SBATCH_SCRIPT="${SBATCH_SCRIPT}" \
    sbatch --array="${TASK_ID}" "${SBATCH_SCRIPT}"
  exit 0
fi
echo "GCE OBC validation incomplete without safe continuation: ${RUN_ID} rc=${rc} fresh_runtime_checkpoints=${fresh_runtime_checkpoints} checkpoints=${checkpoint_count}/${EXPECTED_RANKS}" >&2
[[ "${rc}" -ne 0 ]] || rc=1
exit "${rc}"
