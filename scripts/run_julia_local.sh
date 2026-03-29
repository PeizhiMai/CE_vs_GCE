#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export JULIAUP_DEPOT_PATH="$ROOT/.juliaup"
export JULIA_DEPOT_PATH="$ROOT/.julia_depot"
export JULIA_PROJECT="$ROOT/julia_env"

mkdir -p "$JULIAUP_DEPOT_PATH" "$JULIA_DEPOT_PATH" "$JULIA_PROJECT"

exec /Users/cosdis/.juliaup/bin/julia "$@"
