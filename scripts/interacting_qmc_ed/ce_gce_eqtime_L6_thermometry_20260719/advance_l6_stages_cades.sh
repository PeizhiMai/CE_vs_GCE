#!/usr/bin/env bash
# Idempotently advance completed L=6 pilot/probe stages on CADES.

set -euo pipefail
PROJECT=${PROJECT:-/home/9pm/nUHubbard}
WF=${PROJECT}/scripts/interacting_qmc_ed/ce_gce_eqtime_L6_thermometry_20260719
M=${WF}/manifests
cd "${PROJECT}"

if [[ $(id -un) != 9pm || $(hostname -s) != or-* ]]; then
  echo "ERROR: run as 9pm on lowercase SSH host cades" >&2; exit 2
fi

PYTHON=${PYTHON:-/usr/bin/python3.11}

# Backstop for roots that wrote a fresh, complete checkpoint set but reached
# the Slurm wall clock before MPI teardown returned control to the wrapper.
# The helper is locked, queue-aware, strict-final-aware, and ledgered.
"${PYTHON}" "${WF}/repair_stale_l6_roots.py"

# Positive-U pilot gating.  The production manifests are written only when all
# 24 pilot roots are strict-final and phase pooling/rank coverage are valid.
if "${PYTHON}" "${WF}/summarize_positive_pilots.py" \
    "${M}/ce_L6_positive_pilot_beta5_6p7_10_r32_m10000.tsv" \
    --out "${WF}/status_source/positive_pilot_status.tsv" --manifest-dir "${M}" \
    --write-production; then
  [[ ! -f ${M}/ce_L6_positive_admitted_r32_m50000.tsv ]] || \
    "${PYTHON}" "${WF}/submit_new_manifest_rows.py" "${M}/ce_L6_positive_admitted_r32_m50000.tsv" \
      --root-field outdir --wrapper "${WF}/job_ce_L6_r32_cades.sbatch" --job-name ceL6PosProd
  [[ ! -f ${M}/ce_L6_positive_admitted_r64_m30000.tsv ]] || \
    "${PYTHON}" "${WF}/submit_new_manifest_rows.py" "${M}/ce_L6_positive_admitted_r64_m30000.tsv" \
      --root-field outdir --wrapper "${WF}/job_ce_L6_r64_cades.sbatch" --job-name ceL6PosHi
else
  echo "positive-U pilots are not yet all strict-final; no admitted CE production submitted"
fi

# Every target advances independently: symmetric extensions, a secant
# confirmation, or a confirmed production row.  Existing incomplete probe
# roots prevent duplicate follow-up generation.
mapfile -t PROBES < <(find "${M}" -maxdepth 1 -type f -name 'gce_mu_probe_L6_*.tsv' | sort)
if ((${#PROBES[@]})); then
  "${PYTHON}" "${WF}/tune_mu.py" "${PROBES[@]}" \
    --outdir "${WF}/status_source/mu_tuning" --manifest-dir "${M}" --write-next
fi

for manifest in "${M}"/gce_mu_probe_L6_attractive_followup_*.tsv; do
  [[ -f ${manifest} ]] || continue
  "${PYTHON}" "${WF}/submit_new_manifest_rows.py" "${manifest}" --root-field out_parent \
    --wrapper "${WF}/job_gce_mu_L6_attractive_cades.sbatch" --job-name muL6Attr
done
for manifest in "${M}"/gce_mu_probe_L6_spinHS_followup_*.tsv; do
  [[ -f ${manifest} ]] || continue
  "${PYTHON}" "${WF}/submit_new_manifest_rows.py" "${manifest}" --root-field out_parent \
    --wrapper "${WF}/job_gce_mu_L6_spinHS_cades.sbatch" --job-name muL6Spin
done

if [[ -f ${M}/gce_prod_L6_attractive_confirmed.tsv ]]; then
  "${PYTHON}" "${WF}/submit_new_manifest_rows.py" "${M}/gce_prod_L6_attractive_confirmed.tsv" \
    --root-field out_parent --wrapper "${WF}/job_gce_prod_L6_attractive_cades.sbatch" --job-name gceL6Attr
fi
if [[ -f ${M}/gce_prod_L6_spinHS_confirmed.tsv ]]; then
  "${PYTHON}" "${WF}/submit_new_manifest_rows.py" "${M}/gce_prod_L6_spinHS_confirmed.tsv" \
    --root-field out_parent --wrapper "${WF}/job_gce_prod_L6_spinHS_cades.sbatch" --job-name gceL6Spin
fi

"${PYTHON}" "${WF}/audit_status.py"
