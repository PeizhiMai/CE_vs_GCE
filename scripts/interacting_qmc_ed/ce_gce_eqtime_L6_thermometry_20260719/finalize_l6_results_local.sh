#!/usr/bin/env bash
# Final strict collection, 14-slide asset build, 43-slide deck merge, and QA.
# This script intentionally refuses to run through unfinished production rows.

set -euo pipefail

PROJECT_LOCAL=${PROJECT_LOCAL:-/Users/cosdis/Desktop/projects/CE_GCE}
PROJECT_REMOTE=${PROJECT_REMOTE:-/home/9pm/nUHubbard}
WF_REL=scripts/interacting_qmc_ed/ce_gce_eqtime_L6_thermometry_20260719
WF_LOCAL=${PROJECT_LOCAL}/${WF_REL}
WF_REMOTE=${PROJECT_REMOTE}/${WF_REL}
CORE29=${CORE29:-${PROJECT_LOCAL}/results/L12_plus_L8_CE_GCE_thermometry_core_with_NNN_latest.pptx}
MIRROR_ROOT=${MIRROR_ROOT:-/Users/cosdis/CE_GCE_no_icloud/results}
PYTHON=${PYTHON:-$HOME/.venvs/myenv/bin/python}
NODE=${NODE:-/Users/cosdis/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node}
SKILL_DIR=${SKILL_DIR:-/Users/cosdis/.codex/plugins/cache/openai-primary-runtime/presentations/26.715.12143/skills/presentations}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
RESULT_NAME=ce_gce_eqtime_L6_thermometry_complete_${STAMP}
OUT=${PROJECT_LOCAL}/results/${RESULT_NAME}
MIRROR=${MIRROR_ROOT}/${RESULT_NAME}
SCRATCH_ROOT=${SCRATCH_ROOT:-$(${NODE} -p "require('node:os').tmpdir()")}
WORKSPACE=${SCRATCH_ROOT}/codex-presentations/${CODEX_THREAD_ID:-manual-${STAMP}}/l6-thermometry-final
TMP_DIR=${WORKSPACE}/tmp
mkdir -p "${OUT}/data" "${OUT}/qa" "${TMP_DIR}" "${MIRROR_ROOT}"

ssh cades "cd '${PROJECT_REMOTE}' && python3.11 '${WF_REMOTE}/collect_l6_results.py' \
  --exact-snapshot '${WF_REMOTE}/status_source/exact_u0_L6_snapshot.tsv' \
  --snapshot '${WF_REMOTE}/status_source/analysis/l6_snapshot_current.tsv' \
  --status '${WF_REMOTE}/status_source/analysis/l6_condition_status.tsv' \
  --require-complete"

rsync -av \
  "cades:${WF_REMOTE}/status_source/analysis/l6_snapshot_current.tsv" \
  "cades:${WF_REMOTE}/status_source/analysis/l6_condition_status.tsv" \
  "${OUT}/data/"

"${PYTHON}" "${WF_LOCAL}/analyze_l6_thermometry.py" \
  --snapshot "${OUT}/data/l6_snapshot_current.tsv" \
  --status "${OUT}/data/l6_condition_status.tsv" \
  --outdir "${OUT}/analysis"

"${NODE}" "${SKILL_DIR}/container_tools/setup_artifact_tool_workspace.mjs" --workspace "${TMP_DIR}"
cp "${WF_LOCAL}/build_l6_43slide_deck.mjs" "${TMP_DIR}/build_l6_43slide_deck.mjs"
FINAL_PPTX=${OUT}/L6_L8_L12_CE_GCE_equal_time_thermometry_43slides_${STAMP}.pptx
(cd "${TMP_DIR}" && "${NODE}" build_l6_43slide_deck.mjs "${CORE29}" "${OUT}/analysis" "${FINAL_PPTX}")

"${PYTHON}" "${SKILL_DIR}/container_tools/render_slides.py" "${FINAL_PPTX}"
RENDERED=${FINAL_PPTX%.pptx}
"${PYTHON}" "${SKILL_DIR}/container_tools/create_montage.py" \
  --input_dir "${RENDERED}" --output_file "${OUT}/qa/43slide_montage.png"
"${PYTHON}" "${SKILL_DIR}/container_tools/slides_test.py" "${FINAL_PPTX}" \
  | tee "${OUT}/qa/slides_test.txt"
cp "${FINAL_PPTX}.inspect.ndjson" "${OUT}/qa/deck_inspection.ndjson"
cp -R "${WF_LOCAL}/manifests" "${OUT}/manifests"
cp "${WF_LOCAL}/README.md" "${OUT}/workflow_README.txt"

rsync -a "${OUT}/" "${MIRROR}/"
ln -sfn "${RESULT_NAME}" "${PROJECT_LOCAL}/results/ce_gce_eqtime_L6_thermometry_complete_latest"
ln -sfn "${RESULT_NAME}" "${MIRROR_ROOT}/ce_gce_eqtime_L6_thermometry_complete_latest"
ln -sfn "${RESULT_NAME}/$(basename "${FINAL_PPTX}")" "${PROJECT_LOCAL}/results/L6_L8_L12_CE_GCE_equal_time_thermometry_latest.pptx"
ln -sfn "${RESULT_NAME}/$(basename "${FINAL_PPTX}")" "${MIRROR_ROOT}/L6_L8_L12_CE_GCE_equal_time_thermometry_latest.pptx"

printf 'FINAL_PPTX=%s\nMIRROR_PPTX=%s\nMONTAGE=%s\n' \
  "${FINAL_PPTX}" "${MIRROR}/$(basename "${FINAL_PPTX}")" "${OUT}/qa/43slide_montage.png"
