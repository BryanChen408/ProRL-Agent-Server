#!/usr/bin/env bash
set -uo pipefail
_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OP_NAME="${1:-}"
[[ -z "$OP_NAME" ]] && { echo "[selfcheck] usage: bash tools/ascendc_selfcheck.sh <op_name>" >&2; exit 1; }
shift || true
echo "[selfcheck] 本入口已合并 → 转发到 tools/ascendc_eval_pipeline.sh(与 judge 判分同一个脚本)"
exec bash "$_DIR/ascendc_eval_pipeline.sh" --op_name "$OP_NAME" \
     --impl "output/submission/${OP_NAME}_impl.tar.gz" --out_dir judge_out "$@"
