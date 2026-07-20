#!/usr/bin/env bash
# Checkpoint-safe L=6 OBC canonical equal-time runner.
set -u -o pipefail

PROJECT=${PROJECT:-/home/9pm/nUHubbard_obc_dev}
WF=${PROJECT}/scripts/interacting_qmc_ed/ce_gce_eqtime_L6_obc_thermometry_20260720
MANIFEST=${MANIFEST:?MANIFEST must point to an L6 OBC CE manifest}
RESUBMIT_SCRIPT=${RESUBMIT_SCRIPT:?RESUBMIT_SCRIPT must point to the Slurm wrapper}
TASK_ID=${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}
PYTHON=${PYTHON:-/usr/bin/python3.11}
LOG_DIR=${PROJECT}/logs
mkdir -p "${LOG_DIR}"
cd "${PROJECT}" || exit 2

row=$("${PYTHON}" - "${MANIFEST}" "${TASK_ID}" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter="\t")); i=int(sys.argv[2])
if not 0 <= i < len(rows): raise SystemExit(f"task {i} outside 0..{len(rows)-1}")
r=rows[i]
fields=("idx","target_key","stage","U_label","U","beta","actual_T","Ntot","Nup","Ndn",
        "account","partition","qos","outdir","seed","nwarmups","max_batches","batch_nsamples",
        "measure_interval","cluster_size","Lx","Ly","dtau","expected_ranks","phase_reweighted",
        "force_symmetry","boundary","project_commit","smoqydqmc_version","smoqydqmc_commit",
        "site_count","nn_bond_count","nnn_bond_count","canensafqmc_base_commit",
        "canensafqmc_current_patch_sha256","canensafqmc_obc_patch_sha256")
missing=[k for k in fields if k not in r]
if missing: raise SystemExit(f"manifest missing fields {missing}")
print("\t".join(r[k] for k in fields))
PY
) || exit $?
IFS=$'\t' read -r IDX TARGET_KEY STAGE U_LABEL U BETA ACTUAL_T NTOT NUP NDN ACCOUNT PARTITION QOS OUTDIR BASE_SEED NWARMUPS MAX_BATCHES BATCH_NSAMPLES MEASURE_INTERVAL CLUSTER_SIZE LX LY DTAU EXPECTED_RANKS PHASE_REWEIGHTED FORCE_SYMMETRY BOUNDARY EXPECTED_PROJECT_COMMIT EXPECTED_SMOQY_VERSION EXPECTED_SMOQY_COMMIT SITE_COUNT NN_COUNT NNN_COUNT CANENS_BASE CANENS_CURRENT_PATCH CANENS_OBC_PATCH <<< "${row}"

RANKS=${SLURM_NTASKS:-${EXPECTED_RANKS}}
SMOKE_MODE=${SMOKE_MODE:-false}
[[ "${BOUNDARY}" == open && "${OUTDIR}" == *"_obc_"* ]] || { echo "OBC boundary/root guard failed" >&2; exit 2; }
[[ "${ACCOUNT}" == ccsd || "${ACCOUNT}" == cnms ]] || { echo "unapproved account ${ACCOUNT}" >&2; exit 2; }
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  [[ "${SLURM_JOB_ACCOUNT:-}" == "${ACCOUNT}" ]] || { echo "manifest/Slurm account mismatch" >&2; exit 2; }
  [[ "${SLURM_JOB_PARTITION:-}" == "${PARTITION}" && "${SLURM_JOB_QOS:-}" == "${QOS}" ]] || { echo "partition/QOS mismatch" >&2; exit 2; }
fi
[[ "${PARTITION}" == burst && "${QOS}" == default ]] || { echo "L6 OBC requires burst/default" >&2; exit 2; }
(( RANKS == EXPECTED_RANKS )) || { echo "wrong rank count ${RANKS}/${EXPECTED_RANKS}" >&2; exit 2; }
(( NUP == NDN && NUP + NDN == NTOT )) || { echo "unbalanced canonical sector" >&2; exit 2; }
if [[ "${SMOKE_MODE}" != true ]]; then
  (( LX == 6 && LY == 6 && CLUSTER_SIZE == 36 && SITE_COUNT == 36 && NN_COUNT == 60 && NNN_COUNT == 50 )) || {
    echo "L=6 OBC geometry manifest guard failed" >&2; exit 2;
  }
fi
[[ "${CHECKPOINT_RESET_ACCUMULATORS:-false}" == false ]] || { echo "accumulator reset forbidden" >&2; exit 2; }

ACTUAL_PROJECT_COMMIT=$(git rev-parse HEAD)
ACTUAL_SMOQY_COMMIT=$(git -C "${PROJECT}/external/SmoQyDQMC" rev-parse HEAD)
[[ "${ACTUAL_PROJECT_COMMIT}" == "${EXPECTED_PROJECT_COMMIT}" ]] || { echo "project commit drift ${ACTUAL_PROJECT_COMMIT} != ${EXPECTED_PROJECT_COMMIT}" >&2; exit 2; }
[[ "${ACTUAL_SMOQY_COMMIT}" == "${EXPECTED_SMOQY_COMMIT}" ]] || { echo "SmoQy commit drift" >&2; exit 2; }
[[ "${EXPECTED_SMOQY_VERSION}" == 2.0.12 ]] || { echo "manifest SmoQy version drift" >&2; exit 2; }
[[ "${CANENS_BASE}" == 21b4f6815d0b836973064ff8401fb2ba9c23b802 ]] || { echo "CanEns base drift" >&2; exit 2; }

is_positive=$("${PYTHON}" - "${U}" <<'PY'
import sys; print("true" if float(sys.argv[1]) > 0 else "false")
PY
)
if [[ "${is_positive}" == true ]]; then
  [[ "${PHASE_REWEIGHTED}" == true && "${FORCE_SYMMETRY}" == false ]] || { echo "positive-U OBC CE requires signed spin-HS pooling" >&2; exit 2; }
else
  [[ "${PHASE_REWEIGHTED}" == false ]] || { echo "attractive CE must not request phase pooling" >&2; exit 2; }
fi

AUTO_RESUBMIT=${AUTO_RESUBMIT:-true}; MAX_RESUBMITS=${MAX_RESUBMITS:-120}; RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}
REQUESTED_WALLTIME=${REQUESTED_WALLTIME:-01:30:00}; RUNTIME_LIMIT_HOURS=${RUNTIME_LIMIT_HOURS:-1.25}
CHECKPOINT_FREQ_HOURS=${CHECKPOINT_FREQ_HOURS:-1.0}; CHECKPOINT_EVERY_BATCHES=${CHECKPOINT_EVERY_BATCHES:-10}
CHECKPOINT_WARMUP_CHUNK=${CHECKPOINT_WARMUP_CHUNK:-10}; MIN_RUNTIME_STOP_SECONDS=${MIN_RUNTIME_STOP_SECONDS:-300}
RESUBMIT_JOB_NAME=${SLURM_JOB_NAME:-ceL6OBC}; NODES=$(( (RANKS + 31) / 32 ))

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
export JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE JULIA_PKG_PRECOMPILE_AUTO=0 CE_USE_MKL=true
ACTUAL_SMOQY_VERSION=$("${JULIA_BIN}" --project="${JULIA_PROJECT}" -e 'using SmoQyDQMC; print(Base.pkgversion(SmoQyDQMC))')
[[ "${ACTUAL_SMOQY_VERSION}" == "${EXPECTED_SMOQY_VERSION}" ]] || { echo "SmoQy version drift" >&2; exit 2; }

required=(equal_time_kinetic_per_site_qmc.tsv equal_time_double_occupancy_per_site_qmc.tsv equal_time_nn_spin_qmc.tsv equal_time_nn_connected_charge_qmc.tsv)
strict_final() {
  [[ -f "${OUTDIR}/obc_thermometry_complete.txt" ]] || return 1
  local count f
  count=$(find "${OUTDIR}/ranks" -mindepth 2 -maxdepth 2 -name checkpoint_complete.txt 2>/dev/null | wc -l | tr -d ' ')
  [[ "${count}" -eq "${EXPECTED_RANKS}" ]] || return 1
  count=$(find "${OUTDIR}/ranks" -mindepth 2 -maxdepth 2 -name equal_time_site_density_qmc.tsv 2>/dev/null | wc -l | tr -d ' ')
  [[ "${count}" -eq "${EXPECTED_RANKS}" ]] || return 1
  for f in "${required[@]}"; do [[ -s "${OUTDIR}/${f}" ]] || return 1; done
  [[ -s "${OUTDIR}/equal_time_bond_observables_qmc.tsv" && -s "${OUTDIR}/equal_time_site_density_qmc.tsv" ]] || return 1
  count=$(find "${OUTDIR}" -type f \( -path '*/pair/*' -o -path '*/greens/*' -o -path '*/current/*' -o -path '*/time-displaced/*' \
    -o -name 'bkt_observables_qmc.tsv' -o -name 'equal_time_structure_factors_qmc.tsv' \
    -o -name 'equal_time_charge_spin_wedge_qmc.tsv' -o -name 'equal_time_neighbor_shells_qmc.tsv' \) | wc -l | tr -d ' ')
  [[ "${count}" -eq 0 ]] || return 1
  "${PYTHON}" - "${OUTDIR}" "${EXPECTED_RANKS}" "${SITE_COUNT}" "${NN_COUNT}" "${NNN_COUNT}" <<'PY'
import csv,pathlib,sys
root=pathlib.Path(sys.argv[1]); ranks=int(sys.argv[2]); sites_expected=int(sys.argv[3]); nn=int(sys.argv[4]); nnn=int(sys.argv[5])
for name in ("equal_time_kinetic_per_site_qmc.tsv","equal_time_double_occupancy_per_site_qmc.tsv",
             "equal_time_nn_spin_qmc.tsv","equal_time_nn_connected_charge_qmc.tsv"):
    rows=list(csv.DictReader(open(root/name),delimiter="\t"))
    assert len(rows)==1 and rows[0]["boundary"]=="open" and int(rows[0]["nranks"])==ranks
bond=list(csv.DictReader(open(root/"equal_time_bond_observables_qmc.tsv"),delimiter="\t"))
assert [(row["shell"],int(row["bond_count"])) for row in bond]==[("NN",nn),("NNN",nnn)]
sites=list(csv.DictReader(open(root/"equal_time_site_density_qmc.tsv"),delimiter="\t"))
assert len(sites)==sites_expected
PY
}
if strict_final; then echo "[$(date -Is)] already strict-final ${TARGET_KEY}"; exit 0; fi

mkdir -p "${OUTDIR}"
START=$(date +%s)
echo "[$(date -Is)] L6 OBC CE ${TARGET_KEY} stage=${STAGE} ranks=${RANKS} U=${U} beta=${BETA} N=(${NUP},${NDN}) continuation=${RESUBMIT_COUNT}/${MAX_RESUBMITS}"
sha256sum "${PROJECT}/patches/CanEnsAFQMC-current-response.patch" "${PROJECT}/patches/CanEnsAFQMC-obc.patch" || exit 2
[[ $(sha256sum "${PROJECT}/patches/CanEnsAFQMC-current-response.patch" | awk '{print $1}') == "${CANENS_CURRENT_PATCH}" ]] || exit 2
[[ $(sha256sum "${PROJECT}/patches/CanEnsAFQMC-obc.patch" | awk '{print $1}') == "${CANENS_OBC_PATCH}" ]] || exit 2

extra=(--use-charge-hs=false --sys-type=complex --boundary=open)
if [[ "${is_positive}" == true ]]; then extra+=(--force-symmetry=false --phase-reweight=true); else extra+=(--phase-reweight=false); fi
rc=0
mpiexecjl --project="${JULIA_PROJECT}" -n "${RANKS}" "${JULIA_BIN}" --project="${JULIA_PROJECT}" \
  "${PROJECT}/scripts/interacting_qmc_ed/benchmark_ce_green_tau_space_mpi_3x3.jl" \
  --lx="${LX}" --ly="${LY}" --nup="${NUP}" --ndn="${NDN}" --u="${U}" --beta="${BETA}" --dtau="${DTAU}" \
  --nwarmups="${NWARMUPS}" --batch-nsamples="${BATCH_NSAMPLES}" --max-batches="${MAX_BATCHES}" \
  --measure-interval="${MEASURE_INTERVAL}" --cluster-size="${CLUSTER_SIZE}" --num-fourier-points=auto --nfreq=1 \
  --use-lowrank=false --measure-greens=false --measure-bkt=false --measure-equal-time=true \
  --measure-equal-time-correlations=true --checkpoint-enable=true --checkpoint-file=checkpoint.jls \
  --checkpoint-freq-hours="${CHECKPOINT_FREQ_HOURS}" --checkpoint-every-batches="${CHECKPOINT_EVERY_BATCHES}" \
  --checkpoint-warmup-chunk="${CHECKPOINT_WARMUP_CHUNK}" --runtime-limit-hours="${RUNTIME_LIMIT_HOURS}" \
  --checkpoint-keep=true --checkpoint-reset-accumulators=false --checkpoint-sync-timeout-seconds=900 \
  --checkpoint-sync-poll-seconds=5 --seed="${BASE_SEED}" --python="${PYTHON}" --output-dir="${OUTDIR}" "${extra[@]}" || rc=$?

if [[ "${rc}" -eq 0 ]]; then
  complete=$(find "${OUTDIR}/ranks" -mindepth 2 -maxdepth 2 -name checkpoint_complete.txt 2>/dev/null | wc -l | tr -d ' ')
  if [[ "${complete}" -eq "${EXPECTED_RANKS}" ]]; then
    touch "${OUTDIR}/obc_thermometry_complete.txt"
    strict_final || { echo "strict-final output validation failed" >&2; exit 2; }
    echo "[$(date -Is)] strict-final ${TARGET_KEY}"; exit 0
  fi
fi

checkpoint_count=$(find "${OUTDIR}/ranks" -mindepth 2 -maxdepth 2 -name checkpoint.jls 2>/dev/null | wc -l | tr -d ' ')
status_count=$(find "${OUTDIR}/ranks" -mindepth 2 -maxdepth 2 -name checkpoint.jls.status 2>/dev/null | wc -l | tr -d ' ')
fresh=false
if [[ "${rc}" -eq 13 && "${checkpoint_count}" -eq "${EXPECTED_RANKS}" && "${status_count}" -eq "${EXPECTED_RANKS}" ]]; then
  if "${PYTHON}" - "${OUTDIR}" "${EXPECTED_RANKS}" "${START}" <<'PY'
import pathlib,sys
root=pathlib.Path(sys.argv[1]); n=int(sys.argv[2]); started=float(sys.argv[3])
cp=sorted(root.glob("ranks/rank_*/checkpoint.jls")); st=sorted(root.glob("ranks/rank_*/checkpoint.jls.status"))
assert len(cp)==n and len(st)==n and all(p.stat().st_mtime>=started for p in cp+st)
for p in st:
 d=dict(line.split("=",1) for line in p.read_text().splitlines() if "=" in line)
 assert d.get("reason") in {"runtime_limit","runtime_limit_warmup","runtime_limit_thermalization"}
PY
  then fresh=true; fi
fi
elapsed=$(( $(date +%s) - START ))
if [[ "${fresh}" == true && "${elapsed}" -ge "${MIN_RUNTIME_STOP_SECONDS}" && "${AUTO_RESUBMIT}" == true && "${RESUBMIT_COUNT}" -lt "${MAX_RESUBMITS}" ]]; then
  next=$((RESUBMIT_COUNT+1))
  echo "[$(date -Is)] healthy OBC CE checkpoint; continuation ${next}/${MAX_RESUBMITS}"
  jid=$(MANIFEST="${MANIFEST}" RESUBMIT_COUNT="${next}" RESUBMIT_SCRIPT="${RESUBMIT_SCRIPT}" CHECKPOINT_RESET_ACCUMULATORS=false SMOKE_MODE="${SMOKE_MODE}" \
    sbatch --parsable -A "${ACCOUNT}" -p burst --qos=default --job-name="${RESUBMIT_JOB_NAME}" --nodes="${NODES}" \
      --ntasks="${RANKS}" --ntasks-per-node=32 --cpus-per-task=1 --mem=100G --time="${REQUESTED_WALLTIME}" \
      --array="${TASK_ID}" "${RESUBMIT_SCRIPT}") || exit $?
  jid=${jid%%;*}
  ledger=${OUTDIR}/.l6_obc_continuation_ledger.tsv
  [[ -e "${ledger}" ]] || printf 'timestamp\tjob_id\tarray_task\tcontinuation\tjob_name\taccount\tmanifest\n' > "${ledger}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${jid}" "${TASK_ID}" "${next}" "${RESUBMIT_JOB_NAME}" "${ACCOUNT}" "${MANIFEST}" >> "${ledger}"
  echo "[$(date -Is)] submitted continuation job=${jid} task=${TASK_ID} root=${OUTDIR}"
  exit 0
fi
echo "ERROR: OBC CE root incomplete without safe continuation ${TARGET_KEY} rc=${rc} checkpoints=${checkpoint_count}/${EXPECTED_RANKS}" >&2
[[ "${rc}" -ne 0 ]] || rc=1
exit "${rc}"
