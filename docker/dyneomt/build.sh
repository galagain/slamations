#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker build -f "$here/Dockerfile.torch27-triton33" -t dyneomt-torch27-triton33:jp6 "$here"
docker build -f "$here/Dockerfile.dyneomt-benchmark" -t "${DYNSLAM_DYNEOMT_IMAGE:-dyneomt-benchmark:jp6}" "$here"
