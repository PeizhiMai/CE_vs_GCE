#!/usr/bin/env bash
# Shared checkpoint/resume runner for the L=6 canonical equal-time workflow.
# A thin Slurm wrapper supplies RESUBMIT_SCRIPT and the requested rank layout.

set -u -o pipefail

PROJECT=${PROJECT:-/home/9pm/nUHubbard}
WORKFLOW_REL=scripts/interacting_qmc_ed/ce_gce_eqtime_L6_thermometry_20260719
MANIFEST=${MANIFEST:?MANIFEST must point to an L6 CE TSV manifest}
RESUBMIT_SCRIPT=${RESUBMIT_SCRIPT:?RESUBMIT_SCRIPT must point to the thin Slurm wrapper}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
LOG_DIR=${PROJECT}/logs
mkdir -p "${LOG_DIR}"
cd "${PROJECT}" || exit 2

row_tsv=$(python3 - "${MANIFEST}" "${TASK_ID}" <<'PY'
import csv, sys
path, task = sys.argv[1], int(sys.argv[2])
with open(path, newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
if not 0 <= task < len(rows):
    raise SystemExit(f"task {task} out of range 0..{len(rows)-1}")
r = rows[task]
fields = [
    "idx", "stage", "U_label", "U", "beta", "target_T", "actual_T", "Ltau",
    "Ntot", "Nup", "Ndn", "density", "account", "job_tag", "outdir", "seed",
    "nwarmups", "max_batches", "batch_nsamples", "measure_interval", "cluster_size",
    "Lx", "Ly", "dtau", "expected_ranks", "phase_reweighted", "force_symmetry",
]
missing = [k for k in fields if k not in r]
if missing:
    raise SystemExit(f"manifest missing fields: {missing}")
print("\t".join(r[k] for k in fields))
PY
) || exit $?
IFS=$'\t' read -r IDX STAGE U_LABEL U BETA TARGET_T ACTUAL_T LTAU NTOT NUP NDN DENSITY ACCOUNT JOB_TAG OUTDIR BASE_SEED NWARMUPS MAX_BATCHES BATCH_NSAMPLES MEASURE_INTERVAL CLUSTER_SIZE LX LY DTAU EXPECTED_RANKS PHASE_REWEIGHTED FORCE_SYMMETRY <<< "${row_tsv}"

RANKS=${SLURM_NTASKS:-${EXPECTED_RANKS}}
SMOKE_MODE=${SMOKE_MODE:-false}
if [[ "${ACCOUNT}" != "ccsd" || "${SLURM_JOB_ACCOUNT:-ccsd}" != "ccsd" ]]; then
  echo "ERROR: L6 workflow is pinned to ccsd; manifest=${ACCOUNT} Slurm=${SLURM_JOB_ACCOUNT:-unset}" >&2
  exit 2
fi
if [[ "${SLURM_JOB_PARTITION:-burst}" != "burst" || "${SLURM_JOB_QOS:-default}" != "default" ]]; then
  echo "ERROR: L6 workflow requires burst/default; got ${SLURM_JOB_PARTITION:-unset}/${SLURM_JOB_QOS:-unset}" >&2
  exit 2
fi
if (( RANKS != EXPECTED_RANKS )); then
  echo "ERROR: wrong rank count: Slurm=${RANKS} expected=${EXPECTED_RANKS}" >&2
  exit 2
fi
if (( NUP != NDN || NUP + NDN != NTOT )); then
  echo "ERROR: CE sector must be balanced and sum to Ntot" >&2
  exit 2
fi
if (( LX != 6 || LY != 6 || CLUSTER_SIZE != 36 )) && [[ "${SMOKE_MODE}" != "true" ]]; then
  echo "ERROR: production workflow requires Lx=Ly=6 and cluster_size=36" >&2
  exit 2
fi

CHECKPOINT_RESET_ACCUMULATORS=${CHECKPOINT_RESET_ACCUMULATORS:-false}
if [[ "${CHECKPOINT_RESET_ACCUMULATORS}" != "false" ]]; then
  echo "ERROR: accumulator reset is forbidden for this workflow" >&2
  exit 2
fi

is_positive=$(python3 - "${U}" <<'PY'
import sys
print("true" if float(sys.argv[1]) > 0 else "false")
PY
)
if [[ "${is_positive}" == "true" ]]; then
  [[ "${PHASE_REWEIGHTED}" == "true" && "${FORCE_SYMMETRY}" == "false" ]] || {
    echo "ERROR: positive-U CE requires phase_reweighted=true and force_symmetry=false" >&2; exit 2;
  }
else
  [[ "${PHASE_REWEIGHTED}" == "false" ]] || {
    echo "ERROR: nonpositive-U CE manifest unexpectedly requests phase reweighting" >&2; exit 2;
  }
fi

AUTO_RESUBMIT=${AUTO_RESUBMIT:-true}
MAX_RESUBMITS=${MAX_RESUBMITS:-120}
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}
REQUESTED_WALLTIME=${REQUESTED_WALLTIME:-01:30:00}
CHECKPOINT_FREQ_HOURS=${CHECKPOINT_FREQ_HOURS:-1.0}
RUNTIME_LIMIT_HOURS=${RUNTIME_LIMIT_HOURS:-1.25}
CHECKPOINT_EVERY_BATCHES=${CHECKPOINT_EVERY_BATCHES:-10}
CHECKPOINT_WARMUP_CHUNK=${CHECKPOINT_WARMUP_CHUNK:-1}
MIN_RUNTIME_STOP_SECONDS=${MIN_RUNTIME_STOP_SECONDS:-300}
DIAGNOSE_METROPOLIS=${DIAGNOSE_METROPOLIS:-false}
RESUBMIT_JOB_NAME=${SLURM_JOB_NAME:-ceL6Eq}
NODES=$(( (RANKS + 31) / 32 ))
TPN=32

unset LD_LIBRARY_PATH MPI_PATH MPI_ROOT MPICC MPICXX MPIF77 MPIF90 MPIFC || true
unset OMPI_MCA_pml OMPI_MCA_btl || true
export OMPI_MCA_btl_tcp_if_include=${OMPI_MCA_btl_tcp_if_include:-mgmt0}
export OMPI_MCA_oob_tcp_if_include=${OMPI_MCA_oob_tcp_if_include:-mgmt0}
export PRTE_MCA_oob_tcp_if_include=${PRTE_MCA_oob_tcp_if_include:-mgmt0}
export PMIX_MCA_ptl_tcp_if_include=${PMIX_MCA_ptl_tcp_if_include:-mgmt0}
export PATH="$HOME/.julia/bin:$HOME/.juliaup/bin:$PATH"
export JULIA_BINDIR=${JULIA_BINDIR:-/home/9pm/.julia/juliaup/julia-1.12.1+0.x64.linux.gnu/bin}
JULIA_BIN=${JULIA_BIN:-${JULIA_BINDIR}/julia}
export CE_USE_MKL=${CE_USE_MKL:-true}
export JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE JULIA_PKG_PRECOMPILE_AUTO=0

required_tables=(
  equal_time_observables_qmc.tsv
  equal_time_charge_spin_wedge_qmc.tsv
  equal_time_structure_factors_qmc.tsv
  equal_time_neighbor_shells_qmc.tsv
)
success_outputs_present() {
  [[ -f "${OUTDIR}/checkpoint_complete.txt" ]] || return 1
  local f
  for f in "${required_tables[@]}"; do [[ -f "${OUTDIR}/${f}" ]] || return 1; done
}
if success_outputs_present; then
  echo "[$(date -Is)] already strict-final: ${OUTDIR}"
  exit 0
fi

mkdir -p "${OUTDIR}"
START_EPOCH=$(date +%s)
echo "[$(date -Is)] L6 CE start stage=${STAGE} task=${TASK_ID} job=${SLURM_JOB_ID:-manual} tag=${JOB_TAG}"
echo "U=${U} beta=${BETA} actual_T=${ACTUAL_T} N=${NTOT} (${NUP},${NDN}) ranks=${RANKS} warmups=${NWARMUPS} measurements/rank=$((MAX_BATCHES * BATCH_NSAMPLES)) interval=${MEASURE_INTERVAL}"
echo "phase_reweighted=${PHASE_REWEIGHTED} force_symmetry=${FORCE_SYMMETRY} reset_accumulators=${CHECKPOINT_RESET_ACCUMULATORS} continuation=${RESUBMIT_COUNT}/${MAX_RESUBMITS}"
sha256sum \
  "${PROJECT}/scripts/interacting_qmc_ed/benchmark_ce_green_matsubara_3x3.jl" \
  "${PROJECT}/scripts/interacting_qmc_ed/combine_ce_green_tau_space_rank_outputs.py" \
  "${PROJECT}/scripts/interacting_qmc_ed/ce_unequal_time_current_helpers.jl" || exit 2

extra_args=(--use-charge-hs=false --sys-type=complex)
if [[ "${is_positive}" == "true" ]]; then
  extra_args+=(--force-symmetry=false --phase-reweight=true)
else
  extra_args+=(--phase-reweight=false)
fi

rc=0
mpiexecjl -n "${RANKS}" "${JULIA_BIN}" --project="${PROJECT}/julia_env" \
  "${PROJECT}/scripts/interacting_qmc_ed/benchmark_ce_green_tau_space_mpi_3x3.jl" \
  --lx="${LX}" --ly="${LY}" --nup="${NUP}" --ndn="${NDN}" --u="${U}" \
  --dtau="${DTAU}" --beta="${BETA}" --nwarmups="${NWARMUPS}" \
  --batch-nsamples="${BATCH_NSAMPLES}" --measure-interval="${MEASURE_INTERVAL}" \
  --max-batches="${MAX_BATCHES}" --cluster-size="${CLUSTER_SIZE}" \
  --num-fourier-points=auto --nfreq=1 --use-lowrank=false \
  --measure-greens=false --measure-bkt=false --measure-equal-time=true --measure-equal-time-correlations=true \
  --checkpoint-enable=true --checkpoint-file=checkpoint.jls --checkpoint-freq-hours="${CHECKPOINT_FREQ_HOURS}" \
  --checkpoint-every-batches="${CHECKPOINT_EVERY_BATCHES}" --checkpoint-warmup-chunk="${CHECKPOINT_WARMUP_CHUNK}" \
  --runtime-limit-hours="${RUNTIME_LIMIT_HOURS}" --checkpoint-keep=true \
  --checkpoint-sync-timeout-seconds=900 --checkpoint-sync-poll-seconds=5 \
  --diagnose-metropolis="${DIAGNOSE_METROPOLIS}" --checkpoint-reset-accumulators=false \
  --seed="${BASE_SEED}" --python=python3 --output-dir="${OUTDIR}" "${extra_args[@]}" || rc=$?

rank_complete_count=0
if [[ -d "${OUTDIR}/ranks" ]]; then
  rank_complete_count=$(find "${OUTDIR}/ranks" -mindepth 2 -maxdepth 2 -name checkpoint_complete.txt | wc -l | tr -d ' ')
fi
tables_complete=true
for f in "${required_tables[@]}"; do [[ -f "${OUTDIR}/${f}" ]] || tables_complete=false; done
if [[ "${rc}" -eq 0 && "${rank_complete_count}" -eq "${RANKS}" && "${tables_complete}" == "true" ]]; then
  touch "${OUTDIR}/checkpoint_complete.txt"
fi

END_EPOCH=$(date +%s)
RUNTIME_SECONDS=$((END_EPOCH - START_EPOCH))
checkpoint_count=0; status_count=0; fresh_checkpoint_count=0; fresh_status_count=0; runtime_stop_count=0; fresh_runtime_stop_count=0
if [[ -d "${OUTDIR}/ranks" ]]; then
  checkpoint_count=$(find "${OUTDIR}/ranks" -mindepth 2 -maxdepth 2 -name checkpoint.jls | wc -l | tr -d ' ')
  status_count=$(find "${OUTDIR}/ranks" -mindepth 2 -maxdepth 2 -name checkpoint.jls.status | wc -l | tr -d ' ')
  fresh_checkpoint_count=$(find "${OUTDIR}/ranks" -mindepth 2 -maxdepth 2 -name checkpoint.jls -newermt "@${START_EPOCH}" | wc -l | tr -d ' ')
  fresh_status_count=$(find "${OUTDIR}/ranks" -mindepth 2 -maxdepth 2 -name checkpoint.jls.status -newermt "@${START_EPOCH}" | wc -l | tr -d ' ')
  runtime_stop_count=$(find "${OUTDIR}/ranks" -mindepth 2 -maxdepth 2 -name checkpoint.jls.status -exec grep -El '^reason=runtime_limit(_warmup|_thermalization)?$' {} \; | wc -l | tr -d ' ')
  fresh_runtime_stop_count=$(find "${OUTDIR}/ranks" -mindepth 2 -maxdepth 2 -name checkpoint.jls.status -newermt "@${START_EPOCH}" -exec grep -El '^reason=runtime_limit(_warmup|_thermalization)?$' {} \; | wc -l | tr -d ' ')
fi

error_glob="${LOG_DIR}/${SLURM_JOB_NAME:-ceL6Eq}_*_${SLURM_JOB_ID:-manual}.err"
problem_count=0
for error_log in ${error_glob}; do
  [[ -f "${error_log}" ]] || continue
  n=$(grep -a -Eiv 'ProcessExited\((13|9)\)' "${error_log}" | grep -a -Eci 'numerical[- ]zero[- ]sign|JLD2|MPI_ERRORS_ARE_FATAL|Socket closed|failed to TCP connect|No route to host|drift too large|wrong[- ]rank|ERROR:|LoadError|BoundsError|MethodError|OutOfMemory|StackOverflow' || true)
  problem_count=$((problem_count + n))
done
echo "[$(date -Is)] rc=${rc} runtime_s=${RUNTIME_SECONDS} ranks_complete=${rank_complete_count}/${RANKS} checkpoints=${checkpoint_count}/${RANKS} fresh_checkpoints=${fresh_checkpoint_count}/${RANKS} statuses=${status_count}/${RANKS} fresh_statuses=${fresh_status_count}/${RANKS} runtime_stops=${runtime_stop_count}/${RANKS} fresh_runtime_stops=${fresh_runtime_stop_count}/${RANKS} problems=${problem_count}"

python3 - "${OUTDIR}" "${RUNTIME_SECONDS}" "${RANKS}" "${NWARMUPS}" "$((MAX_BATCHES * BATCH_NSAMPLES))" <<'PY'
import pathlib, statistics, sys
root=pathlib.Path(sys.argv[1]); runtime=max(float(sys.argv[2]),1.0); ranks=int(sys.argv[3]); target_w=int(sys.argv[4]); target_m=int(sys.argv[5])
rows=[]
for p in root.glob("ranks/rank_*/checkpoint.jls.status"):
    d={}
    for line in p.read_text(errors="replace").splitlines():
        if "=" in line:
            k,v=line.split("=",1); d[k]=v
    try:
        rows.append((int(d.get("warmups_completed",-1)), int(d.get("nsamples",d.get("completed_batches",-1))), d.get("reason","")))
    except ValueError:
        pass
if rows:
    w=[x[0] for x in rows]; m=[x[1] for x in rows]; med=statistics.median(m); speed=med/(runtime/3600)
    eta=(target_m-med)/speed if speed > 0 and med > 0 else float("nan")
    print(f"progress ranks={len(rows)}/{ranks} warmup_min_med_max={min(w)},{statistics.median(w)},{max(w)}/{target_w} measurements_min_med_max={min(m)},{med},{max(m)}/{target_m} speed_per_rank_h={speed:.2f} eta_h={eta:.2f}")
else:
    print("progress no rank status files")
PY

if success_outputs_present; then
  echo "[$(date -Is)] strict-final: ${OUTDIR}"
  exit 0
fi

checkpoint_stop=false
if [[ "${checkpoint_count}" -eq "${RANKS}" && "${status_count}" -eq "${RANKS}" \
      && "${fresh_checkpoint_count}" -eq "${RANKS}" && "${fresh_status_count}" -eq "${RANKS}" \
      && "${runtime_stop_count}" -eq "${RANKS}" && "${fresh_runtime_stop_count}" -eq "${RANKS}" \
      && "${problem_count}" -eq 0 && "${RUNTIME_SECONDS}" -ge "${MIN_RUNTIME_STOP_SECONDS}" ]]; then
  case "${rc}" in 0|1|9|13) checkpoint_stop=true;; esac
fi

if [[ "${AUTO_RESUBMIT}" == "true" && "${checkpoint_stop}" == "true" && "${RESUBMIT_COUNT}" -lt "${MAX_RESUBMITS}" ]]; then
  next_count=$((RESUBMIT_COUNT + 1))
  echo "[$(date -Is)] healthy runtime checkpoint; submitting continuation ${next_count}/${MAX_RESUBMITS}"
  sbatch -A ccsd -p burst --qos=default --job-name="${RESUBMIT_JOB_NAME}" \
    --nodes="${NODES}" --ntasks="${RANKS}" --ntasks-per-node="${TPN}" --cpus-per-task=1 --mem=100G --time="${REQUESTED_WALLTIME}" \
    --export=ALL,MANIFEST="${MANIFEST}",RESUBMIT_COUNT="${next_count}",RESUBMIT_SCRIPT="${RESUBMIT_SCRIPT}",CHECKPOINT_RESET_ACCUMULATORS=false,SMOKE_MODE="${SMOKE_MODE}" \
    --array="${TASK_ID}" "${RESUBMIT_SCRIPT}"
  exit 0
fi

echo "ERROR: CE root incomplete without a valid safe continuation: ${OUTDIR} rc=${rc}" >&2
if [[ "${rc}" -eq 0 ]]; then rc=1; fi
exit "${rc}"
