#!/usr/bin/env bash
# Hourly idempotent stage driver for L=6 OBC CE/GCE thermometry.
set -euo pipefail
PROJECT=${PROJECT:-/home/9pm/nUHubbard_obc_dev}
WF=${PROJECT}/scripts/interacting_qmc_ed/ce_gce_eqtime_L6_obc_thermometry_20260720
M=${WF}/manifests
STATUS=${WF}/status_source
PYTHON=${PYTHON:-/usr/bin/python3.11}
mkdir -p "${STATUS}" "${M}" "${PROJECT}/logs"
cd "${PROJECT}"
[[ $(id -un) == 9pm && $(hostname -s) == or-* ]] || { echo "run through lowercase SSH host cades" >&2; exit 2; }
exec 9>"${STATUS}/.advance_l6_obc.lock"
flock -n 9 || { echo "SKIP: another L6 OBC stage driver is active"; exit 0; }
[[ -f ${STATUS}/smoke_passed.txt ]] || { echo "WAIT: smoke gate not passed"; exit 0; }

echo "=== L6 OBC stage advance $(date -Is) ==="
# Import only PBC values already confirmed to |N-Ntarget| <= 0.03.  Previously
# frozen values are immutable and every import retains its inherited L=8 provenance.
"${PYTHON}" "${WF}/import_pbc_mu_references.py" --manifest-dir "${M}"

submit() { "${PYTHON}" "${WF}/submit_new_manifest_rows.py" "$@"; }
# Cumulative initial manifests are safe to revisit: the submission ledger and
# root markers permit only newly imported targets to enter the queue.
for family in attractive spinHS; do
  manifest=${M}/gce_mu_probe_L6_obc_${family}_from_PBC.tsv
  [[ -s ${manifest} ]] || continue
  if [[ ${family} == attractive ]]; then wrapper=${WF}/job_gce_mu_L6_attractive_cades.sbatch; name=muL6OAttr
  else wrapper=${WF}/job_gce_mu_L6_spinHS_cades.sbatch; name=muL6OSpin; fi
  submit "${manifest}" --root-field out_parent --wrapper "${wrapper}" --job-name "${name}"
done

# Repair only fresh full-checkpoint roots that have no root-matched queued leg.
"${PYTHON}" "${WF}/repair_stale_l6_roots.py"

# Positive-U production is generated only after all 24 OBC pilots are strict-final.
if "${PYTHON}" "${WF}/summarize_positive_pilots.py" \
    "${M}/ce_L6_obc_positive_pilot_beta5_6p7_10_r32_m10000.tsv" \
    --out "${STATUS}/positive_pilot_status.tsv" --manifest-dir "${M}" --write-production; then
  [[ ! -f ${M}/ce_L6_obc_positive_admitted_r32_m50000.tsv ]] || \
    submit "${M}/ce_L6_obc_positive_admitted_r32_m50000.tsv" --root-field outdir \
      --wrapper "${WF}/job_ce_L6_r32_cades.sbatch" --job-name ceL6OPN
  [[ ! -f ${M}/ce_L6_obc_positive_admitted_r64_m30000.tsv ]] || \
    submit "${M}/ce_L6_obc_positive_admitted_r64_m30000.tsv" --root-field outdir \
      --wrapper "${WF}/job_ce_L6_r64_cades.sbatch" --job-name ceL6OPH
else
  echo "WAIT: positive-U OBC pilots are not all strict-final"
fi

mapfile -t PROBES < <(find "${M}" -maxdepth 1 -type f -name 'gce_mu_probe_L6_obc_*.tsv' | sort)
if ((${#PROBES[@]})); then
  "${PYTHON}" "${WF}/tune_mu.py" "${PROBES[@]}" \
    --outdir "${STATUS}/mu_tuning" --manifest-dir "${M}" --write-next
fi
for manifest in "${M}"/gce_mu_probe_L6_obc_attractive_followup_*.tsv; do
  [[ -f ${manifest} ]] || continue
  submit "${manifest}" --root-field out_parent --wrapper "${WF}/job_gce_mu_L6_attractive_cades.sbatch" --job-name muL6OAttr
done
for manifest in "${M}"/gce_mu_probe_L6_obc_spinHS_followup_*.tsv; do
  [[ -f ${manifest} ]] || continue
  submit "${manifest}" --root-field out_parent --wrapper "${WF}/job_gce_mu_L6_spinHS_cades.sbatch" --job-name muL6OSpin
done
if [[ -f ${M}/gce_prod_L6_obc_attractive_confirmed.tsv ]]; then
  submit "${M}/gce_prod_L6_obc_attractive_confirmed.tsv" --root-field out_parent \
    --wrapper "${WF}/job_gce_prod_L6_attractive_cades.sbatch" --job-name gceL6OAttr
fi
if [[ -f ${M}/gce_prod_L6_obc_spinHS_confirmed.tsv ]]; then
  submit "${M}/gce_prod_L6_obc_spinHS_confirmed.tsv" --root-field out_parent \
    --wrapper "${WF}/job_gce_prod_L6_spinHS_cades.sbatch" --job-name gceL6OSpin
fi

"${PYTHON}" "${WF}/audit_status.py"
echo "=== L6 OBC stage advance complete $(date -Is) ==="
