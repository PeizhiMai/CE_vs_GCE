#!/usr/bin/env bash
# Install exactly one hourly cron entry for the idempotent L=6 OBC stage driver.
set -euo pipefail
PROJECT=${PROJECT:-/home/9pm/nUHubbard_obc_dev}
WF=${PROJECT}/scripts/interacting_qmc_ed/ce_gce_eqtime_L6_obc_thermometry_20260720
TAG='L6_OBC_THERMOMETRY_20260720_HOURLY'
ENTRY="17 * * * * /bin/bash ${WF}/advance_l6_stages_cades.sh >> ${WF}/status_source/hourly_stage_driver.log 2>&1 # ${TAG}"
[[ $(id -un) == 9pm && $(hostname -s) == or-* ]] || { echo "run on CADES as 9pm" >&2; exit 2; }
current=$(crontab -l 2>/dev/null || true)
filtered=$(printf '%s\n' "${current}" | grep -v "${TAG}" || true)
printf '%s\n%s\n' "${filtered}" "${ENTRY}" | sed '/^[[:space:]]*$/d' | crontab -
echo "installed hourly driver: ${ENTRY}"
