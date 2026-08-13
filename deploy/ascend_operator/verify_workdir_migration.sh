#!/usr/bin/env bash
# Step 1 验证：workdir 从 /opt/workspace/agent_workdir 迁到 /polar/session/agent_workdir
# 之后，一个 session 能否走到 judge 出分。**仍用 backend=docker** —— 这样把「workdir 迁移」
# 和「LocalRuntime」两个变量分开，任一出问题都能立刻归因。
#
# 前提（缺一不可）：
#   1. 本机能跑 docker，且 ascendc-tilelang:v1（或 profile 里的 image）在位
#   2. 有一个 OpenAI 兼容的 LLM 端点 —— agent 要真写出 kernel 才有 submission 可判。
#      没有真 LLM 时 agent 写不出东西，judge 必然报 submission_missing 出 0.2 地板分，
#      而那与「路径断了」的表现完全一样，测不出结论。
#   3. NPU 可用（judge 要编译对拍）
#
# 用法：
#   LLM_HOST=<ip> LLM_PORT=<port> MODEL_SERVED=<模型名> NPU_POOL='[0, 1, 2, 3]' \
#     bash deploy/ascend_operator/verify_workdir_migration.sh [op_name]
#
# MODEL_SERVED 必须与该端点注册的模型名逐字相同（curl <host>:<port>/v1/models 可查）。
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "${HERE}/../.." && pwd -P)"
OP_NAME="${1:-}"
ROLLOUT_PORT="${ROLLOUT_PORT:-12345}"
POLL_MAX="${POLL_MAX:-3600}"

for v in LLM_HOST LLM_PORT MODEL_SERVED NPU_POOL; do
  [[ -n "${!v:-}" ]] || { echo "ERROR: 需要 $v。见脚本头部用法。" >&2; exit 1; }
done

# ─────── 1. 起 polar（跳过 vime handoff）───────────────────────────────────────
echo "=== 1. 起 polar（VIME_NODE_IP 指向 LLM 端点，跳过 handoff）==="
VIME_NODE_IP="${LLM_HOST}" \
VLLM_ROUTER_PORT="${LLM_PORT}" \
MODEL_SERVED="${MODEL_SERVED}" \
NPU_POOL="${NPU_POOL}" \
ROLLOUT_PORT="${ROLLOUT_PORT}" \
  bash "${HERE}/launch_polar_and_guide.sh" || { echo "polar 起不来"; exit 1; }

ROLLOUT="http://127.0.0.1:${ROLLOUT_PORT}"
for i in $(seq 1 60); do
  curl -fsS "${ROLLOUT}/health" >/dev/null 2>&1 && break
  [[ $i == 60 ]] && { echo "ERROR: rollout ${ROLLOUT} 未就绪"; exit 1; }
  sleep 2
done
echo "rollout ok"

# ─────── 2. 渲染产物自查：workdir 必须已是新值 ─────────────────────────────────
echo
echo "=== 2. 渲染产物里的 workdir（必须是 /polar/session/agent_workdir）==="
RUNTIME_PROFILE="${POLAR_PROFILE_RUNTIME:-/tmp/polar_profile_runtime.yaml}"
python3 - "$REPO" "$RUNTIME_PROFILE" <<'PY'
import subprocess, sys, re
repo, profile = sys.argv[1], sys.argv[2]
out = subprocess.run(
    [sys.executable, f"{repo}/deploy/ascend_operator/tools/load_polar_profile.py",
     "--profile", profile, "--repo-root", repo],
    capture_output=True, text=True).stdout
old = out.count("/opt/workspace/agent_workdir")
new = out.count("/polar/session/agent_workdir")
print(f"  旧路径出现 {old} 次，新路径出现 {new} 次")
for line in out.splitlines():
    if "agent_workdir" in line:
        print("   ", line.strip()[:150])
if old:
    sys.exit("FAIL: 渲染产物里仍有 /opt/workspace/agent_workdir")
if not new:
    sys.exit("FAIL: 渲染产物里没有 /polar/session/agent_workdir")
print("  OK")
PY
[[ $? == 0 ]] || exit 1

# ─────── 3. 挑一个算子 ─────────────────────────────────────────────────────────
if [[ -z "${OP_NAME}" ]]; then
  TASKS_DIR="$(sed -n 's|^\s*task_assets_dir:\s*||p' "${RUNTIME_PROFILE}" | head -1)"
  OP_NAME="$(ls "${TASKS_DIR}"/*.py 2>/dev/null | head -1 | xargs -r basename | sed 's/\.py$//')"
  [[ -n "${OP_NAME}" ]] || { echo "ERROR: ${TASKS_DIR} 里没有 .py"; exit 1; }
fi
echo
echo "=== 3. 提交算子 ${OP_NAME} ==="

TASK_ID="verify-workdir-$(date +%s)"
RESP="$(curl -fsS -X POST "${ROLLOUT}/rollout/operator_samples/submit" \
  -H 'content-type: application/json' \
  -d "{\"task_id\":\"${TASK_ID}\",\"instruction\":\"Implement the operator.\",\"num_samples\":1,\"sample\":{\"op_name\":\"${OP_NAME}\"}}")" \
  || { echo "提交失败"; exit 1; }
echo "  ${RESP}"
TASK_ID="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['task_id'])" "$RESP")"

# ─────── 4. 轮询到终态 ─────────────────────────────────────────────────────────
echo
echo "=== 4. 轮询（最长 ${POLL_MAX}s，单 session 可能几十分钟）==="
for (( t = 0; t < POLL_MAX; t += 10 )); do
  ST="$(curl -fsS "${ROLLOUT}/rollout/task/${TASK_ID}" 2>/dev/null)"
  S="$(python3 -c "import json,sys;print(json.loads(sys.argv[1]).get('status',''))" "$ST" 2>/dev/null)"
  printf '\r  t=%ss status=%s   ' "$t" "$S"
  [[ "$S" =~ ^(COMPLETED|ERROR|TIMEOUT)$ ]] && break
  sleep 10
done
echo

# ─────── 5. 三条判据 ──────────────────────────────────────────────────────────
echo
echo "=== 5. 判据 ==="
SESS_JSON="$(curl -fsS "${ROLLOUT}/tasks/${TASK_ID}/sessions" 2>/dev/null)"
echo "$SESS_JSON" > /tmp/verify_sessions.json
OUT_ROOT="$(sed -n 's|^\s*output_dir:\s*||p' "${RUNTIME_PROFILE}" | head -1)"
SESS_DIR="$(find "${REPO}/${OUT_ROOT}" -maxdepth 4 -type d -name "session-*" -newermt '-6 hours' 2>/dev/null | tail -1)"
echo "session 目录: ${SESS_DIR:-未找到}"

echo
echo "[判据 1] prepare 未报 exit 1"
grep -riE "exit 1|prepare.*fail" "${SESS_DIR}"/logs/* 2>/dev/null | head -3 || echo "  未见 prepare 失败"

echo
echo "[判据 2] agent_workdir 落在 session 内（新路径生效的直接证据）"
ls -d "${SESS_DIR}/agent_workdir" 2>/dev/null && ls "${SESS_DIR}/agent_workdir/input/" 2>/dev/null | head -3
echo "  eval 实例的独立 workdir（应与上面不是同一个目录）:"
ls -d "${SESS_DIR}/eval_runtime/agent_workdir" 2>/dev/null || echo "  (无 eval_runtime，可能 judge 未跑到)"

echo
echo "[判据 3] judge 出分 —— 非 0.2 才算路径通"
python3 - <<'PY'
import json
try:
    d = json.load(open("/tmp/verify_sessions.json"))
except Exception as e:
    raise SystemExit(f"  读 sessions 失败: {e}")
items = d if isinstance(d, list) else d.get("sessions", [])
for s in items:
    r = s.get("reward")
    if r is None:
        r = (s.get("result") or {}).get("reward")
    print(f"  session={str(s.get('session_id'))[:40]} status={s.get('status')} reward={r}")
    if r == 0.2:
        print("  !! 0.2 是 submission_missing 地板分 —— judge 没取到 submission。")
        print("     若 agent 日志显示它确实写出了 tarball，问题就在跨实例取文件那条路径。")
PY

echo
echo "完整 session json: /tmp/verify_sessions.json"
echo "agent 日志: ${SESS_DIR}/logs/agent/claude-code.txt"
