#!/usr/bin/env bash
# Submit only ungated OBC CE production, positive-U pilots, and available PBC-seeded probes.
set -euo pipefail
PROJECT=${PROJECT:-/home/9pm/nUHubbard_obc_dev}
WF=${PROJECT}/scripts/interacting_qmc_ed/ce_gce_eqtime_L6_obc_thermometry_20260720
M=${WF}/manifests
PYTHON=${PYTHON:-/usr/bin/python3.11}
[[ -f ${WF}/status_source/smoke_passed.txt ]] || { echo "smoke gate has not passed" >&2; exit 2; }

submit() { "${PYTHON}" "${WF}/submit_new_manifest_rows.py" "$@"; }
submit "${M}/ce_L6_obc_attractive_beta_le10_r32_m50000.tsv" --root-field outdir --wrapper "${WF}/job_ce_L6_r32_cades.sbatch" --job-name ceL6OAtr
submit "${M}/ce_L6_obc_attractive_beta20_r64_m30000.tsv" --root-field outdir --wrapper "${WF}/job_ce_L6_r64_cades.sbatch" --job-name ceL6OA20
submit "${M}/ce_L6_obc_positive_beta_le4_r32_m50000.tsv" --root-field outdir --wrapper "${WF}/job_ce_L6_r32_cades.sbatch" --job-name ceL6OPos
submit "${M}/ce_L6_obc_positive_pilot_beta5_6p7_10_r32_m10000.tsv" --root-field outdir --wrapper "${WF}/job_ce_L6_r32_cades.sbatch" --job-name ceL6OPil

for family in attractive spinHS; do
  manifest=${M}/gce_mu_probe_L6_obc_${family}_from_PBC.tsv
  [[ -s ${manifest} ]] || continue
  if [[ ${family} == attractive ]]; then wrapper=${WF}/job_gce_mu_L6_attractive_cades.sbatch; name=muL6OAttr
  else wrapper=${WF}/job_gce_mu_L6_spinHS_cades.sbatch; name=muL6OSpin; fi
  submit "${manifest}" --root-field out_parent --wrapper "${wrapper}" --job-name "${name}"
done
