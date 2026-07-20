#!/usr/bin/env bash
# Submit the four two-rank OBC workflow smoke modes exactly once.
set -euo pipefail
PROJECT=${PROJECT:-/home/9pm/nUHubbard_obc_dev}
WF=${PROJECT}/scripts/interacting_qmc_ed/ce_gce_eqtime_L6_obc_thermometry_20260720
STATUS=${WF}/status_source
MARKER=${STATUS}/smoke_submission_jobs.tsv
PARTIAL=${MARKER}.partial
mkdir -p "${STATUS}" "${PROJECT}/logs"
[[ $(id -un) == 9pm && $(hostname -s) == or-* ]] || { echo "run through lowercase SSH host cades" >&2; exit 2; }
if [[ -e "${MARKER}" || -e "${PARTIAL}" ]]; then
  echo "ERROR: smoke submission record already exists; refusing duplicates" >&2; exit 2
fi
printf 'stage\tjob_id\tarray\taccount\tpartition\tqos\tmanifest\n' > "${PARTIAL}"

submit() {
  local stage=$1 array=$2 manifest=$3 script=$4 name=$5 family=${6:-}
  local jid
  if [[ -n ${family} ]]; then
    jid=$(MANIFEST="${manifest}" GCE_FAMILY="${family}" AUTO_RESUBMIT=false CHECKPOINT_RESET_ACCUMULATORS=false \
      SMOKE_MODE=true MIN_RUNTIME_STOP_SECONDS=0 \
      sbatch --parsable -A ccsd -p burst --qos=default -N1 -n2 --ntasks-per-node=2 -c1 --mem=8G -t 00:15:00 \
      --job-name="${name}" --array="${array}" "${script}")
  else
    jid=$(MANIFEST="${manifest}" AUTO_RESUBMIT=false CHECKPOINT_RESET_ACCUMULATORS=false \
      SMOKE_MODE=true MIN_RUNTIME_STOP_SECONDS=0 \
      sbatch --parsable -A ccsd -p burst --qos=default -N1 -n2 --ntasks-per-node=2 -c1 --mem=8G -t 00:15:00 \
      --job-name="${name}" --array="${array}" "${script}")
  fi
  jid=${jid%%;*}
  printf '%s\t%s\t%s\tccsd\tburst\tdefault\t%s\n' "${stage}" "${jid}" "${array}" "${manifest}" >> "${PARTIAL}"
  echo "submitted ${stage}: ${jid} array=${array}"
}

submit ce_smoke 0-1 "${WF}/manifests/smoke_ce_L6_obc_two_rank.tsv" "${WF}/job_ce_L6_r32_cades.sbatch" ceL6OSmk
submit gce_attractive_smoke 0 "${WF}/manifests/smoke_gce_L6_obc_two_rank.tsv" "${WF}/job_gce_mu_L6_attractive_cades.sbatch" muL6OSmA attractive
submit gce_spinHS_smoke 1 "${WF}/manifests/smoke_gce_L6_obc_two_rank.tsv" "${WF}/job_gce_mu_L6_spinHS_cades.sbatch" muL6OSmP spinHS
mv "${PARTIAL}" "${MARKER}"
echo "recorded ${MARKER}"
