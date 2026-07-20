#!/usr/bin/env bash
# Strict final collection, OBC/PBC analysis, standalone 18-slide deck build,
# render/overflow QA, timestamped mirroring, and opening of the no_icloud PPTX.
# The script intentionally refuses to finalize through any unfinished required
# OBC or PBC condition.

set -euo pipefail

OBC_PROJECT_LOCAL=${OBC_PROJECT_LOCAL:-/Users/cosdis/Desktop/projects/CE_GCE_l6_obc}
RESULT_PROJECT_LOCAL=${RESULT_PROJECT_LOCAL:-/Users/cosdis/Desktop/projects/CE_GCE}
OBC_PROJECT_REMOTE=${OBC_PROJECT_REMOTE:-/home/9pm/nUHubbard_obc_dev}
PBC_PROJECT_REMOTE=${PBC_PROJECT_REMOTE:-/home/9pm/nUHubbard}
OBC_WF_REL=scripts/interacting_qmc_ed/ce_gce_eqtime_L6_obc_thermometry_20260720
PBC_WF_REL=scripts/interacting_qmc_ed/ce_gce_eqtime_L6_thermometry_20260719
OBC_WF_LOCAL=${OBC_PROJECT_LOCAL}/${OBC_WF_REL}
PBC_WF_LOCAL=${RESULT_PROJECT_LOCAL}/${PBC_WF_REL}
OBC_WF_REMOTE=${OBC_PROJECT_REMOTE}/${OBC_WF_REL}
PBC_WF_REMOTE=${PBC_PROJECT_REMOTE}/${PBC_WF_REL}
MIRROR_ROOT=${MIRROR_ROOT:-/Users/cosdis/CE_GCE_no_icloud/results}
PYTHON=${PYTHON:-$HOME/.venvs/myenv/bin/python}
NODE=${NODE:-/Users/cosdis/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node}
SOFFICE=${SOFFICE:-/Users/cosdis/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/soffice}
SKILL_DIR=${SKILL_DIR:-/Users/cosdis/.codex/plugins/cache/openai-primary-runtime/presentations/26.715.12143/skills/presentations}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
RESULT_NAME=ce_gce_eqtime_L6_obc_thermometry_complete_${STAMP}
OUT=${RESULT_PROJECT_LOCAL}/results/${RESULT_NAME}
MIRROR=${MIRROR_ROOT}/${RESULT_NAME}
SCRATCH_ROOT=${SCRATCH_ROOT:-$(${NODE} -p "require('node:os').tmpdir()")}
WORKSPACE=${SCRATCH_ROOT}/codex-presentations/${CODEX_THREAD_ID:-manual-${STAMP}}/l6-obc-thermometry-final
TMP_DIR=${WORKSPACE}/tmp
mkdir -p "${OUT}/data" "${OUT}/qa" "${OUT}/workflow" "${TMP_DIR}" "${MIRROR_ROOT}"

command -v ssh >/dev/null
command -v rsync >/dev/null
test -x "${PYTHON}"
test -x "${NODE}"
test -x "${SOFFICE}"

# Collect both boundaries with their own strict collector.  This is read-only
# with respect to production roots and does not submit or resume anything.
ssh cades "cd '${OBC_PROJECT_REMOTE}' && python3.11 '${OBC_WF_REMOTE}/collect_l6_results.py' \
  --exact-snapshot '${OBC_WF_REMOTE}/status_source/exact_u0_L6_obc_snapshot.tsv' \
  --snapshot '${OBC_WF_REMOTE}/status_source/analysis/l6_obc_snapshot_final.tsv' \
  --status '${OBC_WF_REMOTE}/status_source/analysis/l6_obc_condition_status_final.tsv' \
  --require-complete"

ssh cades "cd '${PBC_PROJECT_REMOTE}' && python3.11 '${PBC_WF_REMOTE}/collect_l6_results.py' \
  --exact-snapshot '${PBC_WF_REMOTE}/status_source/exact_u0_L6_snapshot.tsv' \
  --snapshot '${PBC_WF_REMOTE}/status_source/analysis/l6_pbc_snapshot_final_for_obc.tsv' \
  --status '${PBC_WF_REMOTE}/status_source/analysis/l6_pbc_condition_status_final_for_obc.tsv' \
  --require-complete"

rsync -av \
  "cades:${OBC_WF_REMOTE}/status_source/analysis/l6_obc_snapshot_final.tsv" \
  "cades:${OBC_WF_REMOTE}/status_source/analysis/l6_obc_condition_status_final.tsv" \
  "${OUT}/data/"
rsync -av \
  "cades:${PBC_WF_REMOTE}/status_source/analysis/l6_pbc_snapshot_final_for_obc.tsv" \
  "cades:${PBC_WF_REMOTE}/status_source/analysis/l6_pbc_condition_status_final_for_obc.tsv" \
  "${OUT}/data/"

"${PYTHON}" "${OBC_WF_LOCAL}/analyze_l6_obc_thermometry.py" \
  --snapshot "${OUT}/data/l6_obc_snapshot_final.tsv" \
  --status "${OUT}/data/l6_obc_condition_status_final.tsv" \
  --outdir "${OUT}/analysis"

# Reuse the final PBC inversion implementation so the boundary comparison is
# algorithmically identical rather than reimplementing it in the deck code.
"${PYTHON}" "${PBC_WF_LOCAL}/analyze_l6_thermometry.py" \
  --snapshot "${OUT}/data/l6_pbc_snapshot_final_for_obc.tsv" \
  --status "${OUT}/data/l6_pbc_condition_status_final_for_obc.tsv" \
  --outdir "${OUT}/pbc_reference_analysis"

"${NODE}" "${SKILL_DIR}/container_tools/setup_artifact_tool_workspace.mjs" --workspace "${TMP_DIR}"
cp "${OBC_WF_LOCAL}/build_l6_obc_18slide_deck.mjs" "${TMP_DIR}/build_l6_obc_18slide_deck.mjs"
FINAL_PPTX=${OUT}/L6_OBC_CE_GCE_equal_time_thermometry_18slides_${STAMP}.pptx
(cd "${TMP_DIR}" && "${NODE}" build_l6_obc_18slide_deck.mjs \
  "${OUT}/analysis" \
  "${FINAL_PPTX}" \
  "${OUT}/pbc_reference_analysis/data/l6_thermometry_quantity_summary.tsv" \
  "${OUT}/data/l6_pbc_snapshot_final_for_obc.tsv" \
  "${OUT}/data/l6_pbc_condition_status_final_for_obc.tsv")

"${PYTHON}" "${SKILL_DIR}/container_tools/render_slides.py" "${FINAL_PPTX}"
RENDERED=${FINAL_PPTX%.pptx}
"${PYTHON}" "${SKILL_DIR}/container_tools/create_montage.py" \
  --input_dir "${RENDERED}" --output_file "${OUT}/qa/18slide_montage.png"
"${PYTHON}" "${SKILL_DIR}/container_tools/slides_test.py" "${FINAL_PPTX}" \
  | tee "${OUT}/qa/slides_test.txt"
cp "${FINAL_PPTX}.inspect.ndjson" "${OUT}/qa/deck_inspection.ndjson"
find "${RENDERED}" -maxdepth 1 -type f -name 'slide-*.png' -print | sort \
  | awk 'BEGIN{print "rendered_slide"}{print}' > "${OUT}/qa/rendered_slide_manifest.tsv"
test "$(find "${RENDERED}" -maxdepth 1 -type f -name 'slide-*.png' | wc -l | tr -d ' ')" = 18

"${SOFFICE}" --headless --convert-to pdf --outdir "${OUT}" "${FINAL_PPTX}" \
  >"${OUT}/qa/soffice_pdf.txt" 2>&1
FINAL_PDF=${FINAL_PPTX%.pptx}.pdf
test -s "${FINAL_PDF}"

# Preserve exact workflow/manifests and the remote audit ledgers that produced
# the final snapshot.  These are evidence, not inputs to the deck renderer.
rsync -a "${OBC_WF_LOCAL}/" "${OUT}/workflow/"
rsync -a "cades:${OBC_WF_REMOTE}/manifests/" "${OUT}/workflow/manifests/"
rsync -a --include='*/' --include='*.tsv' --include='*.txt' --exclude='*' \
  "cades:${OBC_WF_REMOTE}/status_source/" "${OUT}/workflow/status_source/"

rsync -a "${OUT}/" "${MIRROR}/"
ln -sfn "${RESULT_NAME}" "${RESULT_PROJECT_LOCAL}/results/ce_gce_eqtime_L6_obc_thermometry_complete_latest"
ln -sfn "${RESULT_NAME}" "${MIRROR_ROOT}/ce_gce_eqtime_L6_obc_thermometry_complete_latest"
ln -sfn "${RESULT_NAME}/$(basename "${FINAL_PPTX}")" "${RESULT_PROJECT_LOCAL}/results/L6_OBC_CE_GCE_equal_time_thermometry_latest.pptx"
ln -sfn "${RESULT_NAME}/$(basename "${FINAL_PPTX}")" "${MIRROR_ROOT}/L6_OBC_CE_GCE_equal_time_thermometry_latest.pptx"
ln -sfn "${RESULT_NAME}/$(basename "${FINAL_PDF}")" "${RESULT_PROJECT_LOCAL}/results/L6_OBC_CE_GCE_equal_time_thermometry_latest.pdf"
ln -sfn "${RESULT_NAME}/$(basename "${FINAL_PDF}")" "${MIRROR_ROOT}/L6_OBC_CE_GCE_equal_time_thermometry_latest.pdf"

MIRROR_PPTX=${MIRROR}/$(basename "${FINAL_PPTX}")
printf 'FINAL_PPTX=%s\nFINAL_PDF=%s\nMIRROR_PPTX=%s\nMONTAGE=%s\n' \
  "${FINAL_PPTX}" "${FINAL_PDF}" "${MIRROR_PPTX}" "${OUT}/qa/18slide_montage.png"

# User explicitly requested that the final mirrored no_icloud PowerPoint be
# opened after validation.
open "${MIRROR_PPTX}"
