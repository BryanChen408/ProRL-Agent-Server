#!/usr/bin/env bash
# =============================================================================
# ascendc_selfcheck.sh — **已合并进 ascendc_eval_pipeline.sh**,此处只做转发。
#
# 【为什么合并】agent 自检和 judge 判分曾是两个脚本,结果两边持续漂移:judge 侧的退化闸门
# 退化成 grep(agent 侧却用真 AST 检查器)、wheel 安装只补了 judge 一边、基准注入只有 judge
# 有 …… 对抗审查一次查出 22 条,其中 10 条 critical 有 6 条集中在这条缝上。triton 从一开始
# 就是**同一个脚本**(tools/triton_eval_pipeline.sh 既是 agent 固定入口也是 judge_command),
# 这不是巧合。现在 ascendc 对齐:一个入口,一条命令,两侧行为按上下文自适应。
# =============================================================================
set -uo pipefail
_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OP_NAME="${1:-}"
[[ -z "$OP_NAME" ]] && { echo "[selfcheck] usage: bash tools/ascendc_selfcheck.sh <op_name>" >&2; exit 1; }
shift || true
echo "[selfcheck] 本入口已合并 → 转发到 tools/ascendc_eval_pipeline.sh(与 judge 判分同一个脚本)"
exec bash "$_DIR/ascendc_eval_pipeline.sh" --op_name "$OP_NAME" \
     --impl "output/submission/${OP_NAME}_impl.tar.gz" --out_dir judge_out "$@"
