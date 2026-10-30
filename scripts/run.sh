#!/usr/bin/env bash
set -euo pipefail
runtime_dir=$(cd "$(dirname "$0")/.." && pwd)
# Package parsing, validation and launch are shared; default is dry-run.
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$runtime_dir/src${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m piper_vla_common.checkpoint_package \
  --package "${HOME}/piper_checkpoints/current" "$@" \
  --robot single --runtime-dir "$runtime_dir"
