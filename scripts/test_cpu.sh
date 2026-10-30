#!/usr/bin/env bash
# No ROS master, Docker, GPU, model import or hardware access. Dependencies must
# already be installed in the selected CPU Python environment.
set -euo pipefail
runtime_dir=$(cd "$(dirname "$0")/.." && pwd)
cd "$runtime_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$runtime_dir/src:$runtime_dir/tests/dual:$runtime_dir/vendor/openpi_pbl5/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
test_python=${PYTHON:-python3}
"$test_python" -m unittest discover -s tests/dual -p 'test_*.py'
"$test_python" tests/dual/run_single_regression.py
