#!/usr/bin/env bash
set -euo pipefail
runtime_dir=$(cd "$(dirname "$0")/.." && pwd)
# Explicit setup only. Model switching never builds or pulls an image.
docker build -f "$runtime_dir/docker/Dockerfile" --target dual-policy \
  -t "${PIPER_PACKAGE_IMAGE:-piper-vla-dual-policy:local}" "$runtime_dir"
docker build -f "$runtime_dir/docker/Dockerfile" --target dual-client \
  -t "${PIPER_PACKAGE_CLIENT_IMAGE:-piper-vla-dual-client:local}" "$runtime_dir"
