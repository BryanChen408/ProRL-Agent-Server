#!/usr/bin/env bash
# judge_best.sh — t3a judge 侧判优:按 ranked list 顺序验收,AST/状态化打掉顺延,
# 直到一个候选通过我方判分链(ascendc_eval_pipeline.sh,AGENT_SIDE=0 纯判分)。
# 用法: judge_best.sh --op_name OP [--out_dir judge_out]
# 候选发现链(首个命中即用):
#   $POLAR_T3A_CANDIDATES_DIR > $ARTIFACTS_DIR/t3a_candidates > $WORKDIR/output/.t3a/t3a_candidates
# 兜底:无任何候选时,若 $WORKDIR/{op}/ 存在则直接打包它当唯一候选;都没有 → 写 metrics 失败退出。
set -uo pipefail

OP_NAME="" OUT_DIR="judge_out"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --op_name) OP_NAME="$2"; shift 2;;
    --out_dir) OUT_DIR="$2"; shift 2;;
    *) echo "[judge-best] unknown arg: $1" >&2; exit 1;;
  esac
done
[[ -z "$OP_NAME" ]] && { echo "[judge-best] --op_name required" >&2; exit 1; }

_SRC="${BASH_SOURCE[0]}"
command -v readlink >/dev/null 2>&1 && _SRC="$(readlink -f "$_SRC")"
JUDGE_DIR="$(cd "$(dirname "$_SRC")" && pwd)"
WORKDIR="$(cd "$JUDGE_DIR/.." && pwd)"
PIPELINE="$JUDGE_DIR/ascendc_eval_pipeline.sh"
[[ -f "$PIPELINE" ]] || { echo "[judge-best] pipeline missing: $PIPELINE" >&2; exit 1; }

CAND_DIR=""
for cand in "${POLAR_T3A_CANDIDATES_DIR:-}" "${ARTIFACTS_DIR:-}/t3a_candidates" "$WORKDIR/output/.t3a/t3a_candidates"; do
  [[ -n "$cand" && -f "$cand/index.json" ]] && { CAND_DIR="$cand"; break; }
done

# [R5-G4] 隐藏 {op}/,强制 pipeline 走 AGENT_SIDE=0 纯判分:
# workdir 里存在 {op}/ 时 AGENT_SIDE=1,pipeline 的 dedup(CUR_HASH vs .last.hash)
# 会把 ranked 第 2..N 名全部短路成「复用第 1 名的旧结论」——判优走查形同虚设;
# 且 budget/pack/promote 语义都会误伤判分。隐藏后:无 dedup、无预算、无打包,
# 每个候选都 fresh 解包、独立判分。judge 完毕原样恢复。
HIDDEN_OP=""
if [[ -d "$WORKDIR/$OP_NAME" ]]; then
  HIDDEN_OP="$WORKDIR/.${OP_NAME}.judge_hide"
  mv "$WORKDIR/$OP_NAME" "$HIDDEN_OP"
fi
_restore_op() {
  [[ -n "$HIDDEN_OP" && -d "$HIDDEN_OP" ]] && mv "$HIDDEN_OP" "$WORKDIR/$OP_NAME" || true
}
trap _restore_op EXIT

LIST_FILE="$JUDGE_DIR/.judge_best_list.$$"
: > "$LIST_FILE"
if [[ -n "$CAND_DIR" ]]; then
  echo "[judge-best] candidates from $CAND_DIR"
  python3 - "$CAND_DIR" "$LIST_FILE" <<'PY'
import json, sys
cand_dir, out = sys.argv[1], sys.argv[2]
entries = json.load(open(f"{cand_dir}/index.json", encoding="utf-8"))
with open(out, "w", encoding="utf-8") as fh:
    for e in sorted(entries, key=lambda x: x.get("rank", 99)):
        fh.write(f"{e.get('file')}\t{e.get('sha256')}\n")
PY
elif [[ -n "$HIDDEN_OP" && -d "$HIDDEN_OP" ]]; then
  echo "[judge-best] no candidates index; fallback: pack final $OP_NAME/"
  FB_TAR="$WORKDIR/output/submission/${OP_NAME}_impl.tar.gz"
  mkdir -p "$(dirname "$FB_TAR")"
  tar czf "$FB_TAR" --exclude=build --exclude=dist --exclude='*.so' --exclude='*.o' \
      --exclude='*.a' --exclude='*.whl' --exclude=__pycache__ --exclude='*.egg-info' \
      --exclude=judge_out --exclude=output --exclude=.git \
      -C "$HIDDEN_OP" .
  echo -e "$FB_TAR\t$(sha256sum "$FB_TAR" | cut -d' ' -f1)" > "$LIST_FILE"
else
  echo "[judge-best] FATAL: no candidates and no $WORKDIR/$OP_NAME" >&2
  exit 1
fi

ACCEPTED=0
PIPELINE_RC=0
while IFS=$'\t' read -r FILE SHA; do
  [[ -z "$FILE" ]] && continue
  if [[ ! -f "$FILE" ]]; then
    echo "[judge-best] candidate missing on disk: $FILE — skip"
    continue
  fi
  ACTUAL="$(sha256sum "$FILE" | cut -d' ' -f1)"
  if [[ -n "$SHA" && "$ACTUAL" != "$SHA" ]]; then
    echo "[judge-best] sha256 mismatch(tampered?): $FILE — reject"
    continue
  fi
  echo "[judge-best] judging: $FILE"
  bash "$PIPELINE" --op_name "$OP_NAME" --impl "$FILE" --out_dir "$OUT_DIR" || true
  # [R5-G2] judge 侧 process_info 中和:pipeline 每次运行会把它自己的单次判分
  # 写进 $ARTIFACTS_DIR/process_info.json —— 那是 judge 的一次验证,不是 agent 的
  # 解题过程。operator_judge 的 _load_process_events 就认这个路径,留着它会把
  # 「judge 一次过」误当成 agent 的过程分。挪走改名,过程分只认 attempt stream
  # 合成的 process_reward.json(t3a_process_reward.py 产出)。
  if [[ -n "${ARTIFACTS_DIR:-}" && -f "$ARTIFACTS_DIR/process_info.json" ]]; then
    mv -f "$ARTIFACTS_DIR/process_info.json" "$ARTIFACTS_DIR/process_info.judge.$(date +%s).json" 2>/dev/null || true
  fi
  VERDICT=$(python3 -c "
import json
try:
    d = json.load(open('$OUT_DIR/metrics.json'))
    et = d.get('error_type') or ''
    ok = d.get('success') is True or (d.get('correctness_ok') is True)
    blocked = et in ('stateful_impl_detected', 'ast_check_failed')
    print('accept' if ok and not blocked else ('blocked' if blocked else 'reject'))
except Exception as e:
    print('reject')
" 2>/dev/null)
  case "$VERDICT" in
    accept) echo "[judge-best] ACCEPTED: $FILE"; ACCEPTED=1; break;;
    blocked) echo "[judge-best] rejected by judge gate (ast/stateful): $FILE — try next";;
    *) echo "[judge-best] not accepted: $FILE — try next";;
  esac
done < "$LIST_FILE"
rm -f "$LIST_FILE"

# [R5] 过程分合成:attempt stream + 最终 metrics → judge_out/process_reward.json
# 无论候选是否被接受都合成:全挂轨迹的过程反馈同样是信号(此时 metrics 为最后一次判分)。
STREAM=""
for sp in "${POLAR_T3A_CANDIDATES_DIR:-}/../t3a_attempt_stream.jsonl" \
          "${ARTIFACTS_DIR:-}/t3a_attempt_stream.jsonl" \
          "$WORKDIR/output/.t3a/t3a_attempt_stream.jsonl"; do
  [[ -f "$sp" ]] && STREAM="$sp" && break
done
if [[ -n "$STREAM" && -f "$JUDGE_DIR/t3a_process_reward.py" ]]; then
  python3 "$JUDGE_DIR/t3a_process_reward.py" "$STREAM" "$OUT_DIR/metrics.json" \
      > "$OUT_DIR/process_reward.json" 2>/dev/null \
    && echo "[judge-best] process_reward.json written from $STREAM" \
    || echo "[judge-best] process reward synthesis skipped (non-fatal)"
fi

if [[ "$ACCEPTED" != "1" ]]; then
  echo "[judge-best] no candidate accepted; metrics.json reflects the last judged candidate"
  exit 1
fi
exit 0
