#!/usr/bin/env bash
# Submit only the immediate, non-gated L=6 stages.  Positive high-beta
# production and all GCE production remain gated by pilots/mu confirmation.
set -euo pipefail
PROJECT=${PROJECT:-/home/9pm/nUHubbard}
WF=${PROJECT}/scripts/interacting_qmc_ed/ce_gce_eqtime_L6_thermometry_20260719
STATUS=${WF}/status_source
SMOKE_MARKER=${STATUS}/smoke_passed.txt
MARKER=${STATUS}/initial_submission_jobs.tsv
PARTIAL=${MARKER}.partial
mkdir -p "${STATUS}" "${PROJECT}/logs"
[[ -f "${SMOKE_MARKER}" ]] || { echo "ERROR: ${SMOKE_MARKER} missing; pass all four smoke modes first" >&2; exit 2; }
if [[ -e "${MARKER}" || -e "${PARTIAL}" ]]; then
  echo "ERROR: initial submission record already exists; refusing duplicate roots" >&2; exit 2
fi
printf 'stage\tjob_id\tarray\ttasks\tranks_per_task\taccount\tpartition\tqos\tmanifest\n' > "${PARTIAL}"

submit() {
  local stage=$1 manifest=$2 script=$3 name=$4 ranks=$5 nodes=$6
  local tasks array jid
  tasks=$(( $(wc -l < "${manifest}") - 1 )); array="0-$((tasks-1))"
  jid=$(sbatch --parsable -A ccsd -p burst --qos=default --nodes="${nodes}" --ntasks="${ranks}" --ntasks-per-node=32 -c1 --mem=100G -t 01:30:00 \
    --job-name="${name}" --array="${array}" \
    --export=ALL,MANIFEST="${manifest}",CHECKPOINT_RESET_ACCUMULATORS=false "${script}")
  jid=${jid%%;*}
  printf '%s\t%s\t%s\t%s\t%s\tccsd\tburst\tdefault\t%s\n' "${stage}" "${jid}" "${array}" "${tasks}" "${ranks}" "${manifest}" >> "${PARTIAL}"
  echo "submitted ${stage}: ${jid} array=${array} tasks=${tasks} ranks/task=${ranks}"
}

submit ce_attractive_beta_le10 "${WF}/manifests/ce_L6_attractive_beta_le10_r32_m50000.tsv" "${WF}/job_ce_L6_r32_cades.sbatch" ceL6Attr 32 1
submit ce_attractive_beta20 "${WF}/manifests/ce_L6_attractive_beta20_r64_m30000.tsv" "${WF}/job_ce_L6_r64_cades.sbatch" ceL6A20 64 2
submit ce_positive_beta_le4 "${WF}/manifests/ce_L6_positive_beta_le4_r32_m50000.tsv" "${WF}/job_ce_L6_r32_cades.sbatch" ceL6Pos 32 1
submit ce_positive_pilots "${WF}/manifests/ce_L6_positive_pilot_beta5_6p7_10_r32_m10000.tsv" "${WF}/job_ce_L6_r32_cades.sbatch" ceL6Pilot 32 1
submit gce_mu_attractive "${WF}/manifests/gce_mu_probe_L6_attractive_L8seed_pm0p02.tsv" "${WF}/job_gce_mu_L6_attractive_cades.sbatch" muL6Attr 32 1
submit gce_mu_spinHS "${WF}/manifests/gce_mu_probe_L6_positive_spinHS_L8seed_pm0p02.tsv" "${WF}/job_gce_mu_L6_spinHS_cades.sbatch" muL6Spin 32 1
mv "${PARTIAL}" "${MARKER}"
echo "recorded ${MARKER}"
