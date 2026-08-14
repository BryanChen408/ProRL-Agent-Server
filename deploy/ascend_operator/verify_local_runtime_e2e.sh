#!/usr/bin/env bash
# LocalRuntime 端到端验证：起真的 rollout + gateway，走完整一个 session 到 judge 出分。
#
# 三个「不需要」：
#   不需要 docker  —— 这就是本方案的全部意义（backend: local，进程级）
#   不需要真 LLM   —— agent 换 shell harness（presets/shell.py），一条命令冒充 agent
#   不需要真 NPU   —— 假 tarball 到不了编译阶段就被 judge 判掉
#
# 判据看 error_type 不看 reward。0.2 有两个来源（operator_reward.py:72-80）：
#   submission_missing  judge 没找到文件 → 路径断了，这是要抓的
#   ast_check_failed 等 找到了但内容不行 → 路径通了，假 tarball 本就编不过
# 只看 0.2 会把两者混为一谈 —— 这是本方案「最像跑通了」的失败模式。
#
# 用法：
#   bash deploy/ascend_operator/verify_local_runtime_e2e.sh
#   RUN_AS=polar bash ...       # 走降权路径（需先跑 setup_local_runtime_user.sh）
#   RUN_AS= bash ...            # 明确不降权（默认，容器里可能没有 polar 用户）
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "${HERE}/../.." && pwd -P)"
ROLLOUT_PORT="${ROLLOUT_PORT:-12399}"
GATEWAY_PORT="${GATEWAY_PORT:-9199}"
NPU_POOL="${NPU_POOL:-[0]}"
RUN_AS="${RUN_AS-}"
PYBIN="${POLAR_PYTHON:-python3}"
WORK=/tmp/local-e2e
PROFILE=${WORK}/profile_runtime.yaml
FAILED=()

rm -rf "${WORK}"; mkdir -p "${WORK}"
say() { printf '\n=== %s ===\n' "$1"; }
fail() { FAILED+=("$1"); echo "  FAIL: $1"; }

cleanup() {
  pkill -f "polar.cli serve_rollout" 2>/dev/null
  pkill -f "polar.cli serve_gateway" 2>/dev/null
  sleep 1
}
trap cleanup EXIT
# 开头也清一次：上一轮若被 timeout/Ctrl-C 掐断，trap 可能没跑到，旧实例还占着端口。
# 那时新 gateway bind 失败**直接退出**，而 rollout 会连上**旧** gateway —— session 照样
# 跑完、状态照样 completed，但判据全部落空，且看起来像 LocalRuntime 的 bug。踩过一次，
# 排查了三轮才发现 gateway 日志只有 6 行、末尾是 "address already in use"。
cleanup
# 杀完还要等端口真正释放 —— pkill 返回不等于 socket 已回收。
for p in "${ROLLOUT_PORT}" "${GATEWAY_PORT}"; do
  for i in $(seq 1 30); do
    (exec 3<>"/dev/tcp/127.0.0.1/${p}") 2>/dev/null || break
    exec 3>&- 2>/dev/null
    [[ $i == 30 ]] && { echo "端口 ${p} 30s 内没释放，换端口或手动清"; exit 1; }
    sleep 1
  done
done

# ─────── 0. 单测先过 ───────────────────────────────────────────────────────────
say "0. LocalRuntime 单测"
# 约 105s（多数用例真跑子进程）。单独验过时用 SKIP_UNIT=1 跳过。
if [[ "${SKIP_UNIT:-0}" == 1 ]]; then
  echo "  跳过（SKIP_UNIT=1）"
elif PYTHONPATH="${REPO}/src" timeout 600 "${PYBIN}" -m pytest "${REPO}/tests/runtime" -q 2>&1 | tail -3
then :; else fail "单测未过 —— 后面不用看了"; fi

# ─────── 1. 渲染 profile ───────────────────────────────────────────────────────
say "1. 渲染 profile.local.yaml"
TASKS="${POLAR_TASK_ASSETS_DIR:-}"
if [[ -z "${TASKS}" ]]; then
  for c in /mnt/host-model /mnt/model; do
    d="${c}/cbx/op_tasks/op_tasks/op_assets_cudallm_filtered189/op_tasks"
    [[ -d "$d" ]] && { TASKS="$d"; break; }
  done
fi
[[ -d "${TASKS}" ]] || { echo "找不到数据集目录，用 POLAR_TASK_ASSETS_DIR 指定"; exit 1; }
DEVKIT="${POLAR_ASC_DEVKIT_DIR:-}"
if [[ -z "${DEVKIT}" ]]; then
  for c in /mnt/host-model /mnt/model; do
    [[ -d "${c}/cbx/asc-devkit-9.0.0" ]] && { DEVKIT="${c}/cbx/asc-devkit-9.0.0"; break; }
  done
fi
SOC="$("${PYBIN}" -c "import acl;print(acl.get_soc_name())" 2>/dev/null || echo Ascend910_9382)"

sed -e "s|__POLAR_HOST__|127.0.0.1|g" \
    -e "s|__ROLLOUT_PORT__|${ROLLOUT_PORT}|g" \
    -e "s|__VIME_ROUTER_HOST__|127.0.0.1|g" \
    -e "s|__VLLM_ROUTER_PORT__|8001|g" \
    -e "s|__MODEL_SERVED__|unused-shell-harness|g" \
    -e "s|__SOC_VERSION__|${SOC}|g" \
    -e "s|__NPU_POOL__|${NPU_POOL}|g" \
    -e "s|__TASK_ASSETS_DIR__|${TASKS}|g" \
    -e "s|__ASC_DEVKIT_DIR__|${DEVKIT}|g" \
    -e "s|:9100|:${GATEWAY_PORT}|g" \
    "${HERE}/profiles/profile.local.yaml" > "${PROFILE}"
# RUN_AS 为空 = 从渲染产物里去掉 run_as（不改 profile 本身）
[[ -z "${RUN_AS}" ]] && sed -i '/^ *run_as:/d' "${PROFILE}"
grep -q "__[A-Z_]*__" "${PROFILE}" && { echo "残留占位符"; grep -n "__[A-Z_]*__" "${PROFILE}"; exit 1; }
"${PYBIN}" -c "
import yaml,sys
r=yaml.safe_load(open('${PROFILE}'))['operator']['runtime']
print(f\"  backend={r['backend']} run_as={r.get('run_as','(无)')} workdir={r['workdir']}\")
assert r['backend']=='local', 'backend 不是 local'
" || fail "profile 渲染不对"

# ─────── 2. 起服务 ─────────────────────────────────────────────────────────────
say "2. 起 rollout + gateway（backend=local，无 docker）"
source <(POLAR_PROFILE="${PROFILE}" "${PYBIN}" "${HERE}/tools/load_polar_profile.py" \
  --profile "${PROFILE}" --repo-root "${REPO}") || { echo "loader 失败"; exit 1; }
echo "  topology: ${POLAR_TOPOLOGY}"

cd "${REPO}"
# 判据 1/2 要看 session 目录里的 agent_workdir 与 tarball，默认跑完就清
# （node.py:1445 的诊断开关）。不置它 find 必然落空。
export POLAR_KEEP_SESSION_DIR=1
PYTHONPATH="${REPO}/src" nohup "${PYBIN}" -m polar.cli serve_rollout \
  -c "${POLAR_TOPOLOGY}" > "${WORK}/rollout.log" 2>&1 &
PYTHONPATH="${REPO}/src" nohup "${PYBIN}" -m polar.cli serve_gateway \
  -c "${POLAR_TOPOLOGY}" --node-id ascend-node-01 > "${WORK}/gateway.log" 2>&1 &

ROLLOUT="http://127.0.0.1:${ROLLOUT_PORT}"
for i in $(seq 1 60); do
  curl -fsS "${ROLLOUT}/health" >/dev/null 2>&1 && break
  [[ $i == 60 ]] && { echo "rollout 没起来"; tail -20 "${WORK}/rollout.log"; exit 1; }
  sleep 2
done
echo "  rollout ok: $(curl -fsS "${ROLLOUT}/health")"
for i in $(seq 1 45); do
  N="$(curl -fsS "${ROLLOUT}/nodes" 2>/dev/null)"
  [[ "${N}" == *ascend-node* ]] && break
  [[ $i == 45 ]] && { echo "gateway 没注册"; tail -20 "${WORK}/gateway.log"; exit 1; }
  sleep 2
done
echo "  gateway 已注册"

# ─────── 3. 提交一个 session（shell harness 冒充 agent）─────────────────────────
say "3. 提交 session"
# runtime/evaluator 直接取自 loader 生成的 topology —— 那份含完整 operator profile，
# 与 polar 服务读的是同一份，不会和渲染产物分叉。只把 agent 换成 shell harness。
"${PYBIN}" - "${POLAR_TOPOLOGY}" "${TASKS}" > "${WORK}/req.json" <<'PY' || exit 1
import json, pathlib, sys, yaml
topo = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())
tasks = pathlib.Path(sys.argv[2])
roll = topo["rollout"]
op = roll["operator_profiles"][roll["default_operator_profile"]]
rt, ev = op["runtime"], op["evaluator"]
op_name = sorted(p.stem for p in tasks.glob("*.py"))[0]
# prepare 的 upload source 默认指向 output/.../op_assets/op_tasks/（资产缓存目录，
# 由 gen_op_assets.py 填）。生产上 vime 传 sample.task_source 做内容寻址，polar 把
# 这条 upload 的 source 改写成缓存路径；直接构造 TaskRequest 没有那条通道，
# 所以这里把 source 指到真实数据集。不改的话 prepare 明确报
# "source path does not exist"（LocalRuntime 如实抛出，不静默）。
for act in rt.get("prepare", []) + (rt.get("eval_prepare") or []):
    src = str(act.get("source") or "")
    if src.endswith(".py") and "op_assets" in src:
        act["source"] = str(tasks / f"{op_name}.py")
sub = ev["config"].get("submission_path", "output/submission/{op_name}_impl.tar.gz")
sub = sub.replace("{op_name}", op_name)
fake = (f"set -x; mkdir -p $(dirname {sub}) /tmp/fk/{op_name}; "
        f"echo 'not a real kernel' > /tmp/fk/{op_name}/main.cpp; "
        f"tar -czf {sub} -C /tmp/fk {op_name}; ls -l {sub}; echo FAKE_WRITTEN")
req = {"task_id": f"e2e-{op_name}", "instruction": "shell harness, no LLM",
       "num_samples": 1, "timeout_seconds": 900.0, "runtime": rt,
       "agent": {"harness": "shell",
                 "custom_shell": {"command": fake, "cwd": rt["workdir"]}},
       "evaluator": ev}
print(json.dumps(req).replace("{op_name}", op_name))
print(f"  op_name={op_name}", file=sys.stderr)
PY
RESP="$(curl -fsS -X POST "${ROLLOUT}/rollout/task/submit" -H 'content-type: application/json' \
  -d @"${WORK}/req.json")" || { echo "提交失败"; tail -20 "${WORK}/rollout.log"; exit 1; }
echo "  ${RESP}"
TASK_ID="$("${PYBIN}" -c "import json,sys;print(json.loads(sys.argv[1])['task_id'])" "$RESP")"

for (( t = 0; t < 900; t += 5 )); do
  S="$(curl -fsS "${ROLLOUT}/rollout/task/${TASK_ID}" 2>/dev/null \
       | "${PYBIN}" -c "import json,sys;print(json.load(sys.stdin).get('status',''))" 2>/dev/null)"
  printf '\r  t=%ss status=%s   ' "$t" "$S"
  [[ "${S^^}" =~ ^(COMPLETED|ERROR|TIMEOUT)$ ]] && break
  sleep 5
done
echo

# ─────── 4. 判据 ──────────────────────────────────────────────────────────────
say "4. 判据"
curl -fsS "${ROLLOUT}/tasks/${TASK_ID}/sessions" > "${WORK}/sessions.json" 2>/dev/null
SESS="$(find "${POLAR_SESSION_BASE_DIR:-${REPO}/output}" -maxdepth 4 -type d -name "session-*" -newermt "-30 minutes" 2>/dev/null | tail -1)"
echo "session 目录: ${SESS:-未找到}"

echo "[1] agent 与 eval 各有独立 workdir"
for d in "${SESS}/agent_workdir" "${SESS}/eval_runtime/agent_workdir"; do
  [[ -d "$d" ]] && echo "  OK  $d" || fail "缺 $d"
done

echo "[2] 假 tarball 落在 session 内（不在宿主根）"
find "${SESS}" -name '*_impl.tar.gz' 2>/dev/null | head -2 | sed 's/^/  /' \
  || fail "找不到 tarball"
[[ -d /polar/session ]] && fail "宿主根被建出 /polar/session（前缀重写漏了）" \
  || echo "  宿主根干净 OK"

echo "[3] error_type 不是 submission_missing —— 决定性判据"
"${PYBIN}" - "${WORK}/sessions.json" "${SESS}" <<'PY'
import json, pathlib, sys
d = json.load(open(sys.argv[1]))
for s in d.get("sessions", []):
    print(f"  status={s.get('status')} reward={s.get('reward')} error={str(s.get('error'))[:60]}")
blob = pathlib.Path(sys.argv[1]).read_text()
m = list(pathlib.Path(sys.argv[2]).rglob("eval_metrics.json")) if len(sys.argv) > 2 else []
for f in m[:1]:
    j = json.loads(f.read_text())
    print(f"  error_type={j.get('error_type')} submission_used={str(j.get('submission_used'))[:60]}")
    blob += f.read_text()
if "submission_missing" in blob:
    sys.exit("  FAIL: submission_missing —— judge 没取到 agent 写的文件")
print("  PASS: 无 submission_missing —— judge 取到了文件，跨实例路径通")
PY
[[ $? == 0 ]] || fail "judge 报 submission_missing"

echo "[4] 无残留进程 / 无残留 flock"
sleep 2
LEFT="$(pgrep -af 'sleep 300|FAKE_WRITTEN' 2>/dev/null | wc -l)"
[[ "${LEFT}" == 0 ]] && echo "  无残留进程 OK" || fail "有 ${LEFT} 个残留进程"
ls /dev/shm/npu-locks/ 2>/dev/null | head -3 | sed 's/^/  lock: /'

echo "[5] 共享树未被写坏"
git -C "${REPO}" status --porcelain operator_runtime_t2a | head -3 \
  && echo "  operator_runtime_t2a 干净 OK"
[[ -n "${DEVKIT}" ]] && { git -C "${DEVKIT}" status --porcelain 2>/dev/null | grep -v '^?? docs/zh/' | head -3; echo "  asc-devkit 干净 OK（?? docs/zh/ 是 launcher 自举的软链，预期）"; }

say "结果"
if [[ ${#FAILED[@]} == 0 ]]; then
  echo "全部通过。日志：${WORK}/"
else
  printf '%s 项失败：\n' "${#FAILED[@]}"; printf '  - %s\n' "${FAILED[@]}"
  echo "日志：${WORK}/rollout.log ${WORK}/gateway.log"
  exit 1
fi
