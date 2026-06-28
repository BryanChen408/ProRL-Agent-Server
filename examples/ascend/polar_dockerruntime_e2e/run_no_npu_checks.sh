#!/usr/bin/env bash
# Stable no-NPU check entrypoint for CI/local review. This wraps the default
# preflight and focused Polar E2E tests; it does not start training, services,
# Docker containers, cleanup scripts, or the runtime image gate.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLAR_ROOT="$(cd "${ROOT}/../../.." && pwd)"

cd "${POLAR_ROOT}"

bash "${ROOT}/preflight.sh"
pytest -q -p no:cacheprovider \
  tests/examples/test_polar_render_contract.py \
  tests/examples/test_polar_gen_op_assets.py \
  tests/examples/test_polar_pipeline_budget_watcher.py
