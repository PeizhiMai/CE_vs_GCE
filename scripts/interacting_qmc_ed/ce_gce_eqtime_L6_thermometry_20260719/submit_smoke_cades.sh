#!/usr/bin/env bash
set -euo pipefail
PROJECT=${PROJECT:-/home/9pm/nUHubbard}
WF=${PROJECT}/scripts/interacting_qmc_ed/ce_gce_eqtime_L6_thermometry_20260719
STATUS=${WF}/status_source
MARKER=${STATUS}/smoke_submission_jobs.tsv
PARTIAL=${MARKER}.partial
mkdir -p "${STATUS}" "${PROJECT}/logs"
if [[ -e "${MARKER}" || -e "${PARTIAL}" ]]; then
  echo "ERROR: smoke submission record already exists (${MARKER} or .partial); refusing duplicates" >&2
  exit 2
fi
printf 'stage\tjob_id\tarray\taccount\tpartition\tqos\tmanifest\n' > "${PARTIAL}"

submit() {
  local stage=$1 array=$2 manifest=$3 script=$4 name=$5 extra=$6
  local jid
  # shellcheck disable=SC2086
  jid=$(sbatch --parsable -A ccsd -p burst --qos=default -N1 -n2 --ntasks-per-node=2 -c1 --mem=8G -t 00:15:00 \
    --job-name="${name}" --array="${array}" \
    --export=ALL,MANIFEST="${manifest}",AUTO_RESUBMIT=false,CHECKPOINT_RESET_ACCUMULATORS=false,${extra} "${script}")
  jid=${jid%%;*}
  printf '%s\t%s\t%s\tccsd\tburst\tdefault\t%s\n' "${stage}" "${jid}" "${array}" "${manifest}" >> "${PARTIAL}"
  echo "submitted ${stage}: ${jid} array=${array}"
}

submit ce_smoke 0-1 "${WF}/manifests/smoke_ce_L6_two_rank.tsv" "${WF}/job_ce_L6_r32_cades.sbatch" ceL6Smoke "MIN_RUNTIME_STOP_SECONDS=0,SMOKE_MODE=true"
submit gce_attractive_smoke 0 "${WF}/manifests/smoke_gce_mu_L6_two_rank.tsv" "${WF}/job_gce_mu_L6_attractive_cades.sbatch" muL6SmA "GCE_FAMILY=attractive,SMOKE_MODE=true,MIN_RUNTIME_STOP_SECONDS=0"
submit gce_spinHS_smoke 1 "${WF}/manifests/smoke_gce_mu_L6_two_rank.tsv" "${WF}/job_gce_mu_L6_spinHS_cades.sbatch" muL6SmP "GCE_FAMILY=spinHS,SMOKE_MODE=true,MIN_RUNTIME_STOP_SECONDS=0"
mv "${PARTIAL}" "${MARKER}"
echo "recorded ${MARKER}"
