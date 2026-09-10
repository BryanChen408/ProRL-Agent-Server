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
[[ "$OP_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "[judge-best] valid --op_name required" >&2; exit 1; }

_SRC="${BASH_SOURCE[0]}"
command -v readlink >/dev/null 2>&1 && _SRC="$(readlink -f "$_SRC")"
JUDGE_DIR="$(cd "$(dirname "$_SRC")" && pwd)"
WORKDIR="$(cd "$JUDGE_DIR/.." && pwd)"
cd "$WORKDIR" || exit 1
mkdir -p "$OUT_DIR/candidates"
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
  echo "[judge-best] no candidates and no $WORKDIR/$OP_NAME" >&2
fi

ACCEPTED=0
# Start a new selection; per-candidate directories are fresh on every invocation.
rm -f "$OUT_DIR/metrics.json" "$OUT_DIR/metrics_error.log"
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
  CAND_OUT=$(mktemp -d "$OUT_DIR/candidates/${ACTUAL}.XXXXXX") || exit 1
  bash "$PIPELINE" --op_name "$OP_NAME" --impl "$FILE" --out_dir "$CAND_OUT" 2>&1 | tee "$CAND_OUT/pipeline.log"
  PIPELINE_RC=${PIPESTATUS[0]}
  # [R5-G2] judge 侧 process_info 中和:pipeline 每次运行会把它自己的单次判分
  # 写进 $ARTIFACTS_DIR/process_info.json —— 那是 judge 的一次验证,不是 agent 的
  # 解题过程。operator_judge 的 _load_process_events 就认这个路径,留着它会把
  # 「judge 一次过」误当成 agent 的过程分。挪走改名,过程分只认 attempt stream
  # 合成的 process_reward.json(t3a_process_reward.py 产出)。
  if [[ -n "${ARTIFACTS_DIR:-}" && -f "$ARTIFACTS_DIR/process_info.json" ]]; then
    mv -f "$ARTIFACTS_DIR/process_info.json" "$ARTIFACTS_DIR/process_info.judge.$(date +%s).json" 2>/dev/null || true
  fi
  VERDICT=$(python3 - "$CAND_OUT" "$OUT_DIR" "$ACTUAL" "$FILE" "$PIPELINE_RC" <<'PYRESULT'
import json
from pathlib import Path
import shutil
import sys
candidate_dir, output = map(Path, sys.argv[1:3])
sha, filename, rc = sys.argv[3], sys.argv[4], int(sys.argv[5])
try:
    d = json.loads((candidate_dir / 'metrics.json').read_text())
    if not isinstance(d, dict):
        raise ValueError('metrics must be an object')
except (OSError, ValueError) as exc:
    d = {'success': False, 'correctness_ok': False, 'ast_check_ok': False,
         'error_type': 'judge_no_metrics', 'error': str(exc)}
    shutil.copyfile(candidate_dir / 'pipeline.log', candidate_dir / 'metrics_error.log')
if d.get('evaluated_candidate_sha256') not in (None, '', sha):
    d.update(success=False, correctness_ok=False, error_type='judge_metrics_unreadable',
             error='candidate hash in metrics does not match the evaluated tarball')
d['evaluated_candidate_sha256'] = sha
ok = (rc == 0 and d.get('success') is True and d.get('correctness_ok') is True
      and d.get('ast_check_ok') is True and not d.get('error_type'))
if not ok and d.get('success') is True:
    d.update(success=False, error_type=d.get('error_type') or 'judge_metrics_unreadable')
(candidate_dir / 'metrics.json').write_text(json.dumps(d, ensure_ascii=False, indent=2))
selected_path = output / 'metrics.json'
selected = json.loads(selected_path.read_text()) if selected_path.exists() else None
summary = {k: d.get(k) for k in ('success', 'error_type', 'correctness_ok', 'cases_passed',
                              'cases_total', 'evaluated_candidate_sha256')}
summary.update(candidate=filename, pipeline_exit_code=rc)
history = (selected or {}).get('judge_candidates', []) + [summary]
# Keep the first ranked substantive failure; layout/infra-only results must
# not hide an implementation error. A later accepted candidate always wins.
no_submission = {'submission_missing', 'judge_no_metrics', 'judge_metrics_unreadable'}
choose = (selected is None or ok or
          (selected.get('error_type') in no_submission and d.get('error_type') not in no_submission))
if choose:
    selected = d
    error_path = output / 'metrics_error.log'
    if (candidate_dir / 'metrics_error.log').exists():
        shutil.copyfile(candidate_dir / 'metrics_error.log', error_path)
    else:
        error_path.write_text(d.get('error') or '')
selected['judge_candidates'] = history
selected_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2))
print('accept' if ok else 'reject')
PYRESULT
  ) || exit 1
  case "$VERDICT" in
    accept) echo "[judge-best] ACCEPTED: $FILE"; ACCEPTED=1; break;;
    blocked) echo "[judge-best] rejected by judge gate (ast/stateful): $FILE — try next";;
    *) echo "[judge-best] not accepted: $FILE — try next";;
  esac
done < "$LIST_FILE"
rm -f "$LIST_FILE"
if [[ ! -f "$OUT_DIR/metrics.json" ]]; then
  python3 - "$OUT_DIR/metrics.json" <<'PYNONE'
import json, sys
with open(sys.argv[1], 'w') as f:
    json.dump({'success': False, 'correctness_ok': False, 'ast_check_ok': False,
               'error_type': 'submission_missing', 'error': 'No readable, hash-verified candidate',
               'evaluated_candidate_sha256': None, 'judge_candidates': []}, f)
PYNONE
fi

# [R5] 过程分合成:attempt stream + 最终 metrics → judge_out/process_reward.json
# 无论候选是否被接受都合成:全挂轨迹的过程反馈同样是信号(此时 metrics 保留排名靠前的真实失败)。
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
  echo "[judge-best] no candidate accepted; metrics.json preserves the selected failure; all verdicts are in judge_candidates"
  exit 1
fi
exit 0
