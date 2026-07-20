#!/usr/bin/env bash
# Checkpoint-safe per-row runner for the CE OBC/ED validation matrix.
set -u -o pipefail

PROJECT=${PROJECT:-/home/9pm/nUHubbard_obc_dev}
WORKFLOW=${PROJECT}/scripts/interacting_qmc_ed/obc_ce_gce_validation_20260720
MANIFEST=${MANIFEST:-${WORKFLOW}/manifests/ce_validation_manifest.tsv}
TASK_ID=${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}
EXPECTED_ACCOUNT=ccsd

row=$(python3 - "${MANIFEST}" "${TASK_ID}" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter="\t")); i=int(sys.argv[2])
if not 0 <= i < len(rows): raise SystemExit(f"task {i} outside 0..{len(rows)-1}")
r=rows[i]
fields=("idx","run_id","L","U","beta","dtau","nup","ndn","seed","expected_ranks",
        "warmups","measurements_per_rank","measurement_interval","phase_reweighted",
        "force_symmetry","outdir","smoqydqmc_commit")
print("\t".join(r[k] for k in fields))
PY
) || exit $?
IFS=$'\t' read -r IDX RUN_ID L U BETA DTAU NUP NDN SEED EXPECTED_RANKS WARMUPS MEASUREMENTS INTERVAL PHASE_REWEIGHTED FORCE_SYMMETRY OUTDIR SMOQY_COMMIT <<< "${row}"

RANKS=${SLURM_NTASKS:-${EXPECTED_RANKS}}
[[ "${RANKS}" -eq "${EXPECTED_RANKS}" ]] || { echo "wrong rank count ${RANKS}/${EXPECTED_RANKS}" >&2; exit 2; }
[[ "${OUTDIR}" == *"_obc_"* ]] || { echo "OBC root lacks _obc_: ${OUTDIR}" >&2; exit 2; }
ACTUAL_SMOQY_COMMIT=$(git -C "${PROJECT}/external/SmoQyDQMC" rev-parse HEAD)
[[ "${ACTUAL_SMOQY_COMMIT}" == "${SMOQY_COMMIT}" ]] || {
  echo "SmoQyDQMC commit drift: ${ACTUAL_SMOQY_COMMIT} != ${SMOQY_COMMIT}" >&2; exit 2;
}
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  [[ "${SLURM_JOB_ACCOUNT:-}" == "${EXPECTED_ACCOUNT}" ]] || { echo "CE validation must remain on ccsd" >&2; exit 2; }
  [[ "${SLURM_JOB_PARTITION:-}" == "burst" && "${SLURM_JOB_QOS:-}" == "default" ]] || { echo "CE validation requires burst/default" >&2; exit 2; }
fi

required=(equal_time_kinetic_per_site_qmc.tsv equal_time_double_occupancy_per_site_qmc.tsv equal_time_nn_spin_qmc.tsv equal_time_nn_connected_charge_qmc.tsv)
strict_final() {
  [[ -f "${OUTDIR}/obc_validation_complete.txt" ]] || return 1
  local count f
  count=$(find "${OUTDIR}/ranks" -mindepth 2 -maxdepth 2 -name checkpoint_complete.txt 2>/dev/null | wc -l | tr -d ' ')
  [[ "${count}" -eq "${EXPECTED_RANKS}" ]] || return 1
  for f in "${required[@]}"; do [[ -s "${OUTDIR}/${f}" ]] || return 1; done
  python3 - "${OUTDIR}" "${EXPECTED_RANKS}" "${MANIFEST}" "${TASK_ID}" <<'PY'
import csv,pathlib,sys
root=pathlib.Path(sys.argv[1]); expected=int(sys.argv[2])
manifest=list(csv.DictReader(open(sys.argv[3]),delimiter="\t")); row=manifest[int(sys.argv[4])]
for name in ("equal_time_kinetic_per_site_qmc.tsv","equal_time_double_occupancy_per_site_qmc.tsv",
             "equal_time_nn_spin_qmc.tsv","equal_time_nn_connected_charge_qmc.tsv"):
    rows=list(csv.DictReader(open(root/name),delimiter="\t"))
    assert len(rows)==1 and rows[0]["boundary"]=="open" and int(rows[0]["nranks"])==expected
pair_estimator=row.get("ce_same_spin_estimator","")
if pair_estimator:
    metadata=sorted(root.glob("ranks/rank_*/metadata.toml"))
    assert len(metadata)==expected
    for path in metadata:
        needle='obc_same_spin_estimator = "{}"'.format(pair_estimator)
        assert needle in path.read_text()
PY
}
if strict_final; then echo "[$(date -Is)] already strict-final ${RUN_ID}"; exit 0; fi

mkdir -p "${OUTDIR}" "${PROJECT}/logs"
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
ACTUAL_SMOQY_VERSION=$("${JULIA_BIN}" --project="${JULIA_PROJECT}" -e 'using SmoQyDQMC; print(Base.pkgversion(SmoQyDQMC))')
[[ "${ACTUAL_SMOQY_VERSION}" == "2.0.12" ]] || {
  echo "SmoQyDQMC version drift: ${ACTUAL_SMOQY_VERSION} != 2.0.12" >&2; exit 2;
}

BATCH_NSAMPLES=100
MAX_BATCHES=$((MEASUREMENTS / BATCH_NSAMPLES))
START=$(date +%s)
echo "[$(date -Is)] CE OBC validation task=${TASK_ID} ${RUN_ID} ranks=${RANKS} U=${U} beta=${BETA} dtau=${DTAU} sector=(${NUP},${NDN}) seed=${SEED}"
rc=0
mpiexecjl --project="${JULIA_PROJECT}" -n "${RANKS}" "${JULIA_BIN}" --project="${JULIA_PROJECT}" \
  "${PROJECT}/scripts/interacting_qmc_ed/benchmark_ce_green_tau_space_mpi_3x3.jl" \
  --lx="${L}" --ly="${L}" --boundary=open --nup="${NUP}" --ndn="${NDN}" \
  --u="${U}" --beta="${BETA}" --dtau="${DTAU}" --nwarmups="${WARMUPS}" \
  --batch-nsamples="${BATCH_NSAMPLES}" --max-batches="${MAX_BATCHES}" \
  --measure-interval="${INTERVAL}" --cluster-size="$((L*L))" --num-fourier-points=auto \
  --nfreq=1 --use-lowrank=false --use-charge-hs=false --sys-type=complex \
  --force-symmetry="${FORCE_SYMMETRY}" --phase-reweight="${PHASE_REWEIGHTED}" \
  --measure-greens=false --measure-bkt=false --measure-equal-time=true \
  --measure-equal-time-correlations=true --checkpoint-enable=true --checkpoint-file=checkpoint.jls \
  --checkpoint-freq-hours=1.0 --checkpoint-every-batches=10 --checkpoint-warmup-chunk=10 \
  --runtime-limit-hours=1.25 --checkpoint-keep=true --checkpoint-reset-accumulators=false \
  --checkpoint-sync-timeout-seconds=900 --checkpoint-sync-poll-seconds=5 \
  --seed="${SEED}" --python=python3 --output-dir="${OUTDIR}" || rc=$?

if [[ "${rc}" -eq 0 ]]; then
  complete_count=$(find "${OUTDIR}/ranks" -mindepth 2 -maxdepth 2 -name checkpoint_complete.txt 2>/dev/null | wc -l | tr -d ' ')
  if [[ "${complete_count}" -eq "${EXPECTED_RANKS}" ]]; then
    touch "${OUTDIR}/obc_validation_complete.txt"
    strict_final || { echo "CE outputs failed strict validation" >&2; exit 2; }
    echo "[$(date -Is)] strict-final ${RUN_ID}"
    exit 0
  fi
fi

checkpoint_count=$(find "${OUTDIR}/ranks" -mindepth 2 -maxdepth 2 -name checkpoint.jls 2>/dev/null | wc -l | tr -d ' ')
status_count=$(find "${OUTDIR}/ranks" -mindepth 2 -maxdepth 2 -name checkpoint.jls.status 2>/dev/null | wc -l | tr -d ' ')
elapsed=$(( $(date +%s) - START ))
fresh_runtime_checkpoints=false
if [[ "${rc}" -eq 13 && "${checkpoint_count}" -eq "${EXPECTED_RANKS}" && "${status_count}" -eq "${EXPECTED_RANKS}" ]]; then
  if python3 - "${OUTDIR}" "${EXPECTED_RANKS}" "${START}" <<'PY'
import pathlib, sys
root=pathlib.Path(sys.argv[1]); expected=int(sys.argv[2]); started=float(sys.argv[3])
checkpoints=sorted(root.glob("ranks/rank_*/checkpoint.jls"))
statuses=sorted(root.glob("ranks/rank_*/checkpoint.jls.status"))
assert len(checkpoints)==expected and len(statuses)==expected
assert all(p.stat().st_mtime >= started for p in checkpoints+statuses)
for path in statuses:
    fields=dict(line.strip().split("=",1) for line in path.read_text().splitlines() if "=" in line)
    assert fields.get("reason") in {"runtime_limit", "runtime_limit_warmup"}
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
echo "CE OBC validation incomplete without safe continuation: ${RUN_ID} rc=${rc} fresh_runtime_checkpoints=${fresh_runtime_checkpoints} checkpoints=${checkpoint_count}/${EXPECTED_RANKS}" >&2
[[ "${rc}" -ne 0 ]] || rc=1
exit "${rc}"
