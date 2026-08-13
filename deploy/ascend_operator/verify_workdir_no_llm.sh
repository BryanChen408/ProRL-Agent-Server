#!/usr/bin/env bash
# Step 1 验证（**不需要大模型服务**）：workdir 迁到 /polar/session/agent_workdir 之后，
# prepare 能否跑通、judge 能否取到 agent 写的 submission。仍用 backend=docker。
#
# 怎么绕开 LLM：agent 换成 shell harness（AgentSpec.harness="shell" + custom_shell，
# 见 src/polar/agent/presets/shell.py），用一条 shell 命令冒充 agent —— 它只做一件事：
# 往 judge 会去找的相对路径写一个假 tarball。整条 prepare → agent → judge 链路照跑。
#
# 判据不看 reward 看 error_type。**0.2 有两个来源**（operator_reward.py:72-80）：
#   submission_missing  = judge 没找到文件  → 路径断了，是我们要抓的
#   ast_check_failed 等 = 找到了但内容不行  → 路径通了，假 tarball 本就编不过
# 只看 0.2 会把两者混为一谈。真正的判据是 error_type != submission_missing。
#
# 用法：
#   NPU_POOL='[0, 1, 2, 3]' bash deploy/ascend_operator/verify_workdir_no_llm.sh [op_name]
#
# 无需 LLM_HOST / MODEL_SERVED：agent 不发 LLM 请求。NPU 也只在 judge 编译时用到，
# 假 tarball 编不过就返回，通常连卡都不会真占。
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "${HERE}/../.." && pwd -P)"
OP_NAME="${1:-}"
ROLLOUT_PORT="${ROLLOUT_PORT:-12345}"
POLL_MAX="${POLL_MAX:-1800}"
: "${NPU_POOL:?需要 NPU_POOL，例如 NPU_POOL='[0, 1, 2, 3]'}"

# ─────── 1. 起 polar（LLM 端点填 127.0.0.1，反正不会被访问）────────────────────
echo "=== 1. 起 polar ==="
VIME_NODE_IP=127.0.0.1 \
MODEL_SERVED="${MODEL_SERVED:-unused-no-llm}" \
NPU_POOL="${NPU_POOL}" \
ROLLOUT_PORT="${ROLLOUT_PORT}" \
  bash "${HERE}/launch_polar_and_guide.sh" || { echo "polar 起不来"; exit 1; }

ROLLOUT="http://127.0.0.1:${ROLLOUT_PORT}"
for i in $(seq 1 60); do
  curl -fsS "${ROLLOUT}/health" >/dev/null 2>&1 && break
  [[ $i == 60 ]] && { echo "ERROR: rollout 未就绪"; exit 1; }
  sleep 2
done
echo "rollout ok"

# ─────── 2. 构造 TaskRequest：复用 topology 里的 runtime/evaluator，agent 换 shell ──
echo
echo "=== 2. 构造请求（runtime/evaluator 取自渲染产物，agent 换成 shell）==="
python3 - "$REPO" "${OP_NAME}" "$ROLLOUT" <<'PY' > /tmp/verify_req.json || exit 1
import json, os, subprocess, sys, yaml, pathlib
repo, op_name, rollout = sys.argv[1], sys.argv[2], sys.argv[3]

# 从 loader 的 export 里拿 POLAR_TOPOLOGY —— 那份 yaml 含完整 operator profile
prof = os.environ.get("POLAR_PROFILE") or "/tmp/polar_profile_runtime.yaml"
out = subprocess.run(
    [sys.executable, f"{repo}/deploy/ascend_operator/tools/load_polar_profile.py",
     "--profile", prof, "--repo-root", repo],
    capture_output=True, text=True, check=True).stdout
env = {}
for line in out.splitlines():
    if line.startswith("export "):
        k, _, v = line[len("export "):].partition("=")
        env[k] = v.strip("'")
topo = yaml.safe_load(pathlib.Path(env["POLAR_TOPOLOGY"]).read_text())
profiles = topo["rollout"]["operator_profiles"]
op_profile = profiles[topo["rollout"]["default_operator_profile"]]

runtime = op_profile["runtime"]
evaluator = op_profile["evaluator"]
workdir = runtime["workdir"]
print(f"  workdir = {workdir}", file=sys.stderr)
if "/opt/workspace" in workdir:
    sys.exit("FAIL: workdir 还是旧值，改动没生效")

if not op_name:
    tasks = runtime.get("kwargs", {})
    tad = None
    for line in yaml.safe_dump(topo).splitlines():
        pass
    tad = yaml.safe_load(pathlib.Path(prof).read_text())["operator_runtime"].get("task_assets_dir")
    cand = sorted(pathlib.Path(tad).glob("*.py")) if tad else []
    if not cand:
        sys.exit(f"FAIL: {tad} 里没有 .py，用参数显式给 op_name")
    op_name = cand[0].stem
print(f"  op_name = {op_name}", file=sys.stderr)

# judge 找的相对路径（profile 的 submission_path），由 _abs() 拼上 workdir
sub_rel = evaluator["config"].get("submission_path", "output/submission/{op_name}_impl.tar.gz")
sub_rel = sub_rel.replace("{op_name}", op_name)

# 冒充 agent 的一条命令：建目录 + 写一个内容无效但**存在**的 tarball。
# judge 找得到 → error_type 不是 submission_missing → 路径通。编不过是预期的。
fake = (
    f"set -x; mkdir -p $(dirname {sub_rel}); "
    f"mkdir -p /tmp/fake_impl/{op_name} && echo 'not a real kernel' > /tmp/fake_impl/{op_name}/main.cpp && "
    f"tar -czf {sub_rel} -C /tmp/fake_impl {op_name} && "
    f"ls -l {sub_rel} && echo FAKE_SUBMISSION_WRITTEN"
)

req = {
    "task_id": f"verify-nollm-{op_name}",
    "instruction": "no-op shell agent",
    "num_samples": 1,
    "timeout_seconds": float(op_profile.get("timeout_seconds", 1800)),
    "runtime": runtime,
    "agent": {"harness": "shell",
              "custom_shell": {"command": fake, "cwd": workdir}},
    "evaluator": evaluator,
}
# prepare 里的 {op_name} 占位符要替换
req_s = json.dumps(req).replace("{op_name}", op_name)
print(req_s)
PY
[[ -s /tmp/verify_req.json ]] || { echo "构造请求失败"; exit 1; }

# ─────── 3. 提交 ──────────────────────────────────────────────────────────────
echo
echo "=== 3. 提交 ==="
RESP="$(curl -fsS -X POST "${ROLLOUT}/rollout/task/submit" \
  -H 'content-type: application/json' -d @/tmp/verify_req.json)" \
  || { echo "提交失败"; exit 1; }
echo "  ${RESP}"
TASK_ID="$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['task_id'])" "$RESP")"

# ─────── 4. 轮询 ──────────────────────────────────────────────────────────────
echo
echo "=== 4. 轮询（最长 ${POLL_MAX}s；无 LLM，应该几分钟内结束）==="
for (( t = 0; t < POLL_MAX; t += 5 )); do
  ST="$(curl -fsS "${ROLLOUT}/rollout/task/${TASK_ID}" 2>/dev/null)"
  S="$(python3 -c "import json,sys;print(json.loads(sys.argv[1]).get('status',''))" "$ST" 2>/dev/null)"
  printf '\r  t=%ss status=%s   ' "$t" "$S"
  [[ "$S" =~ ^(COMPLETED|ERROR|TIMEOUT)$ ]] && break
  sleep 5
done
echo

# ─────── 5. 判据 ──────────────────────────────────────────────────────────────
echo
echo "=== 5. 判据 ==="
curl -fsS "${ROLLOUT}/tasks/${TASK_ID}/sessions" > /tmp/verify_sessions.json 2>/dev/null
SESS_BASE="$(sed -n 's|^\s*session_base_dir:\s*||p' /tmp/polar_profile_runtime.yaml 2>/dev/null)"
SESS_DIR="$(find "${REPO}/output" -maxdepth 6 -type d -name "session-*" -newermt '-2 hours' 2>/dev/null | tail -1)"
echo "session 目录: ${SESS_DIR:-未找到}"

echo
echo "[1] agent 与 eval 各有独立 workdir（新路径生效的直接证据）"
for d in "${SESS_DIR}/agent_workdir" "${SESS_DIR}/eval_runtime/agent_workdir"; do
  if [[ -d "$d" ]]; then echo "  OK  $d"; else echo "  --  $d 不存在"; fi
done

echo
echo "[2] 假 tarball 写出来了"
find "${SESS_DIR}" -name "*_impl.tar.gz" 2>/dev/null | head -3 || echo "  未找到"
grep -l FAKE_SUBMISSION_WRITTEN "${SESS_DIR}"/logs/**/* 2>/dev/null | head -2

echo
echo "[3] error_type 是否 submission_missing —— 这是真判据"
python3 - <<'PY'
import json, pathlib
try:
    d = json.load(open("/tmp/verify_sessions.json"))
except Exception as e:
    raise SystemExit(f"  读 sessions 失败: {e}")
for s in d.get("sessions", []):
    print(f"  status={s.get('status')} reward={s.get('reward')} error={str(s.get('error'))[:80]}")
blob = pathlib.Path("/tmp/verify_sessions.json").read_text()
if "submission_missing" in blob:
    print("  FAIL: 出现 submission_missing —— judge 没取到 agent 写的文件。")
    print("        这正是 workdir 迁移最可能踩的坑（agent 与 judge 各有一份 workdir）。")
elif "submission_fetch_failed" in blob:
    print("  FAIL: submission_fetch_failed —— 找到了但传输失败。")
else:
    print("  PASS: 无 submission_missing —— judge 取到了文件，路径通。")
    print("        reward 低/编译失败是预期的（假 tarball 本就不是真 kernel）。")
PY

echo
echo "详细：/tmp/verify_sessions.json"
echo "eval 日志：${SESS_DIR}/logs/eval/"
