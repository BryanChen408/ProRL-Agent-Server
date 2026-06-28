#!/usr/bin/env bash
# Sync the Polar repository as one deployable unit.
#
# This replaces the old multi-step flow that separately copied polar_e2e,
# readonly_tools, tools, op assets, and skills. It intentionally excludes
# runtime output; generate or sync datasets separately.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/_paths.sh"

usage() {
  cat <<'EOF'
Usage:
  deploy/ascend_operator/sync_polar_repo.sh <user@host:/path/to/ProRL-Agent-Server/>

Environment:
  POLAR_SYNC_DRY_RUN=1          Print the rsync plan without copying.
  POLAR_SYNC_DELETE=1           Delete files at the destination that no longer exist locally.
  POLAR_SYNC_ALLOW_LEGACY_DEST=1 Allow destination paths containing /polar_e2e.

Notes:
  - Syncs the Polar repo root as the deployable unit.
  - Excludes .git and output/.
  - Does not sync operator datasets; use OPERATOR_TASK_JSONL/OPERATOR_TASKS_DIR on the Slime side.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

DEST="${1:-}"
if [[ -z "${DEST}" ]]; then
  usage >&2
  exit 2
fi

if [[ "${DEST}" == *"/polar_e2e"* && "${POLAR_SYNC_ALLOW_LEGACY_DEST:-0}" != "1" ]]; then
  cat >&2 <<EOF
[fatal] refusing to sync to legacy polar_e2e destination:
  ${DEST}

Use a repo-shaped destination, for example:
  root@host:/home/docker/ProRL-Agent-Server/

Set POLAR_SYNC_ALLOW_LEGACY_DEST=1 only for a deliberate compatibility run.
EOF
  exit 2
fi

RSYNC_ARGS=(
  -a
  --human-readable
  --info=stats2,progress2
  --exclude=.git/
  --exclude=output/
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='.pytest_cache/'
  --exclude='.mypy_cache/'
  --exclude='.ruff_cache/'
)

if [[ "${POLAR_SYNC_DELETE:-0}" == "1" ]]; then
  RSYNC_ARGS+=(--delete)
fi

if [[ "${POLAR_SYNC_DRY_RUN:-0}" == "1" ]]; then
  RSYNC_ARGS+=(--dry-run)
fi

echo "[sync] source=${POLAR_REPO_ROOT}/"
echo "[sync] dest=${DEST}"
echo "[sync] delete=${POLAR_SYNC_DELETE:-0} dry_run=${POLAR_SYNC_DRY_RUN:-0}"

exec rsync "${RSYNC_ARGS[@]}" "${POLAR_REPO_ROOT}/" "${DEST}"
