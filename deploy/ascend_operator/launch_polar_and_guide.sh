#!/usr/bin/env bash
set -euo pipefail
# 双机方案一键启动脚本（在 8 卡宿主机上运行）。
#
# 新流程（无需 sleep infinity）：
#   1. 在平台前端提交 vime 任务，Command 填真实训练命令：
#        bash /workspace/vime/scripts/start_vime_in_platform.sh
#   2. vime 启动后会打印 IP 并写入共享存储，然后等待 Polar
#   3. 在本机运行此脚本：bash launch_polar_and_guide.sh
#   4. Polar 启动后，vime 自动检测到并开始训练

POLAR_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
# 本项目自己的 venv。原先指向 corlorlight_models/mingchengzou/ 下那个（别人的目录）。
#
# 这个 venv **只提供第三方依赖**，polar 本身不需要装：start_polar_nohup.sh:128 在启动时
# 注入 PYTHONPATH="${POLAR_ROOT}/src"，POLAR_ROOT 由 _paths.sh 从脚本自身位置往上两级推得
# —— 也就是本仓。故 polar 代码来自本仓，与 venv 里有没有 polar 无关。
# （旧 venv 里那个 editable .pth 指向别的仓，但 PYTHONPATH 排在 site-packages 之前，
#   launcher 路径上不生效；只有绕过 launcher 直接 $POLAR_VENV/bin/python 才会命中它。）
#
# 建法（conda 默认 channel 要求先接受 ToS，用 conda-forge 绕过）：
#   conda create -y --prefix /mnt/model/cbx/env/polar-env -c conda-forge \
#     --override-channels python=3.11
#   /mnt/model/cbx/env/polar-env/bin/python -m ensurepip --upgrade
#   /mnt/model/cbx/env/polar-env/bin/python -m pip install \
#     fastapi uvicorn httpx pydantic pyyaml
POLAR_VENV="${POLAR_VENV:-/mnt/model/cbx/env/polar-env}"
# 这两个可覆盖，用来在同机并存第二个 polar 实例（profile 即实例身份：端口、卡池、
# 输出目录全由它决定）。**默认值一字不改** —— start_hostctl.sh 硬编码读
# /tmp/polar_profile_runtime.yaml 拿在跑服务的端口表，改默认值会让它退回 profile.yaml
# （别的站点配置，端口对不上）。
#
# 起第二个实例时这两个都要给：POLAR_PROFILE_RUNTIME 不给会覆盖默认实例的渲染产物。
# 用法见 profile.bak.yaml 头部。
POLAR_PROFILE_SRC="${POLAR_PROFILE_SRC:-${POLAR_REPO}/deploy/ascend_operator/profiles/profile.t2a.yaml}"
POLAR_PROFILE_RUNTIME="${POLAR_PROFILE_RUNTIME:-/tmp/polar_profile_runtime.yaml}"

# ─────── 站点值（profile 里只有 token，实际值在这里）──────────────────────────
# 端口必须与 vime 对表：vime 侧同名变量是 POLAR_ROLLOUT_PORT（rollout）和
# VLLM_ROUTER_PORT（router）。改任一处都要两边一起改 —— 不一致的表现是 vime 等满
# health 超时，或每个生成请求打到没人监听的端口，而不是启动期报错。
ROLLOUT_PORT="${ROLLOUT_PORT:-12345}"
VLLM_ROUTER_PORT="${VLLM_ROUTER_PORT:-8001}"
# gateway / observer / stale_gateway 端口不在这里：它们纯 polar 内部，真源是
# profile.t2a.yaml（gateway_url、observer.port、gateway.extra_stale_ports）。
# Kernel 按这个值编译，编错芯片的表现是 sandbox 里编译失败或产物不对。
#
# 不读继承来的 SOC_VERSION：**实测它会是错的** —— 训练镜像里带着
# SOC_VERSION=ascend910_9391（A3），而本机 acl.get_soc_name() 是 Ascend910_9382。
# 原先写 "${SOC_VERSION:-Ascend910_9382}"，环境变量优先 → 默认值形同虚设。
# 改为实测优先，显式覆盖走独立的 POLAR_SOC_VERSION（跟继承值区分开）。
#
# acl 只在带 CANN 的 python 里可用（polar 的 venv 没有），故显式给 CANN 的 site-packages。
if [[ -n "${POLAR_SOC_VERSION:-}" ]]; then
  SOC_VERSION="${POLAR_SOC_VERSION}"
  echo "[polar-init] SOC 由 POLAR_SOC_VERSION 指定：${SOC_VERSION}"
else
  # 候选用 glob 而非写死版本号：宿主机与训练容器的 CANN 安装路径不同（实测宿主机上
  # 前两个固定路径都不命中，探测失败走了兜底）。先试继承的 PYTHONPATH（shell 里
  # source 过 set_env.sh 时最省事），再扫 /usr/local/Ascend 下所有 python/site-packages。
  _soc_detected=""
  _soc_detected="$(python3 -c 'import acl; print(acl.get_soc_name())' 2>/dev/null || true)"
  if [[ -z "${_soc_detected}" ]]; then
    for _cann in /usr/local/Ascend/*/python/site-packages \
                 /usr/local/Ascend/*/*/python/site-packages; do
      [[ -d "${_cann}" ]] || continue
      _soc_detected="$(PYTHONPATH="${_cann}" python3 -c 'import acl; print(acl.get_soc_name())' 2>/dev/null || true)"
      [[ -n "${_soc_detected}" ]] && break
    done
  fi
  if [[ -n "${_soc_detected}" ]]; then
    SOC_VERSION="${_soc_detected}"
    echo "[polar-init] SOC 实测：${SOC_VERSION}（acl.get_soc_name）"
  else
    SOC_VERSION="Ascend910_9382"
    echo "[polar-init] WARN: acl.get_soc_name() 不可用，兜底 SOC=${SOC_VERSION}。" >&2
    echo "  与本机芯片不符时 kernel 会编错，用 POLAR_SOC_VERSION 显式指定。" >&2
    echo "  参考：A3=ascend910_9391，A2=ascend910b1。" >&2
  fi
fi
# Polar host's own cards. 默认由 vime 的 resolved layout 推出（roles.polar_reserved），
# 见下面的交接段 —— 手写会和 vime 侧的卡位分叉。NPU_POOL 显式设置时仍然优先。
# Must equal vime's --hf-checkpoint: vLLM registers the model under that path,
# and Polar sends this string as openai_request["model"]. A mismatch is a 404
# per request, not a startup error.
#
# 默认由 vime 的交接文件提供（VIME_HF_CKPT，见下面的交接段）—— 不再手写。
# 手写过不去的原因：这个字符串跟着**vime 那台机器**的共享盘挂载点变（A 上 /mnt/model、
# 平台节点上 /mnt/host-model），而它不是给 polar 打开文件用的、是模型名，所以必须填
# vime 的路径而不是本机的路径。这点反直觉，是异机拓扑最容易填错的一处。
# MODEL_SERVED 显式设置时仍然优先。
# Host paths on this machine — read by the Polar process, not the sandbox:
# task_assets_dir is globbed here, asc_devkit_dir is bind-mounted into sandboxes.
# 与 vime 的 OPERATOR_TASKS_DIR 同一份，不需要第二份拷贝：{op}.py 走 vime 的
# sample.task_source 传过来。filtered189 用 get_inputs()（输入写在 py 里），没有
# 同名 .json，所以 _has_case_json 为假、那条 upload 不生成 —— 这是预期的。
TASK_ASSETS_DIR="${TASK_ASSETS_DIR:-/mnt/model/cbx/op_tasks/op_tasks/op_assets_cudallm_filtered189/op_tasks}"
# 共享盘路径：clone 和软链一次落盘，换机器/重建容器都还在，不必每台重拉。
ASC_DEVKIT_DIR="${ASC_DEVKIT_DIR:-/mnt/model/cbx/asc-devkit-9.0.0}"
# ─────── vime 交接：本轮拓扑由 vime 的 resolved layout 单源提供 ────────────────
# vime rank0 渲染完 layout 后，往 ${RDV_DIR}/polar_handoff.env 写两个值：
#   VIME_ROLLOUT_HOST → 推理端点主机（= layout 的 rollout[0].node，router/LB-proxy 绑这台）
#   POLAR_NPU_POOL    → 本机卡池（= layout 的 polar_reserved 卡号）
# 两者都由 layout 推出，这里不再手写 —— 手写的那份迟早和 vime 的卡位/节点分叉。
#
# 路径含 RDV_KEY → **每次运行唯一**。这是修掉旧竞争的关键：旧做法是双方共用固定路径的
# vime_router_ip.txt，本脚本启动时先 rm 再等它重现；vime 若先起就会被删掉，而 vime 不会
# 重写 —— 双方各等满超时后退出（vime 600s / polar 900s）。路径唯一之后本脚本不删任何
# 东西，也不可能读到上一轮残留，两个方向的启动顺序都能跑通。
# 必须与 vime 侧 SCRATCH_DIR 同值（vime: ${VIME_SHARE_ROOT}/cbx/scratch）：交接目录
# 由它拼出来，两边不一致就等不到对方。vime 那边同名变量可覆盖，改一处要改两处。
VIME_SHARE_ROOT="${VIME_SHARE_ROOT:-/mnt/model}"
VIME_SCRATCH_DIR="${VIME_SCRATCH_DIR:-${VIME_SHARE_ROOT}/cbx/scratch}"

# ─────── 先停掉在跑的 polar（必须排在所有可能失败的步骤之前）──────────────────
# 停旧实例原本只在最后的 restart_polar_host.sh 里做（它的 "Stop Existing Services" 段）。
# 后果：本脚本前段任何一步失败退出，旧实例都会**继续活着**，而它拨的是上一轮的 router
# 地址。vime 那边只轮询 rollout 的 /health，响应是 {"status":"ok","nodes":1} —— 不含拓扑
# 信息，旧实例和本轮实例长得一模一样。于是 vime 认为 polar 就绪、开始训练，而旧 polar
# 往上一轮那台机器的 :8001 拨（那儿没有本轮的 router）→ 每个 session 拿不到 trace →
# 日志里全是 "0 usable tokens"，两边都不报错。
#
# 实测踩过两次：脚本在 `source ${POLAR_VENV}/bin/activate` 处失败退出（conda env 没有
# 这个文件），停旧实例的代码根本没跑到，旧 polar 就一直骗过 vime 的就绪检查。
#
# 所以在这里先停一次。幂等：没有在跑的实例时 stop_polar.sh 只打 "no pid file" 就返回。
# `|| true`：停不掉不该阻断启动 —— 后面 restart_polar_host.sh 还会再停一次并清端口。
echo "[polar-init] 先停掉可能在跑的旧 polar（避免它骗过 vime 的就绪检查）..."
bash "$(dirname "${BASH_SOURCE[0]}")/stop_polar.sh" 2>&1 | sed 's/^/[polar-init]   /' || true

# ─────── asc-devkit 自举：缺了就拉，软链不对就补（幂等，每次启动过一遍）────────
# 钉 9.0.0 而非 master：实测 API 名对本机 CANN 9.0.0 头文件命中率 95% vs 82%。
# 软链把 skill 写死的 docs/zh/api 指到 9.0.0 实际布局 docs/api/context；相对路径，
# 只读挂进 /opt/asc-devkit 后照样解析。已就绪时两块都是 no-op。
ASC_DEVKIT_REPO="${ASC_DEVKIT_REPO:-https://gitcode.com/cann/asc-devkit.git}"
ASC_DEVKIT_REF="${ASC_DEVKIT_REF:-9.0.0}"
# gate 用 docs/api/context 而非目录本身：空目录会被 -d 误判成已就绪。
if [[ ! -d "${ASC_DEVKIT_DIR}/docs/api/context" ]]; then
  echo "[polar-init] asc-devkit 缺失，clone ${ASC_DEVKIT_REF} → ${ASC_DEVKIT_DIR}"
  git clone --depth 1 --branch "${ASC_DEVKIT_REF}" \
      "${ASC_DEVKIT_REPO}" "${ASC_DEVKIT_DIR}" \
    || { echo "ERROR: asc-devkit clone 失败，请手动拉到 ${ASC_DEVKIT_DIR}" >&2; exit 1; }
fi
# 校验指向而非存在：布局变了或有人建错链时能自愈。-T 防 ln 把新链建到旧链目录里面。
if [[ "$(readlink "${ASC_DEVKIT_DIR}/docs/zh/api" 2>/dev/null)" != "../api/context" ]]; then
  mkdir -p "${ASC_DEVKIT_DIR}/docs/zh" \
    && ln -sfnT ../api/context "${ASC_DEVKIT_DIR}/docs/zh/api" \
    || { echo "ERROR: 建软链失败：${ASC_DEVKIT_DIR}/docs/zh/api" >&2; exit 1; }
  echo "[polar-init] 已建软链 docs/zh/api → ../api/context"
fi

# ─────── 检查前置条件 ─────────────────────────────────────────────────────────
if [[ ! -x "${POLAR_VENV}/bin/python" ]]; then
  echo "ERROR: polar venv 不存在于 ${POLAR_VENV}" >&2
  echo "  该 venv 只提供第三方依赖，polar 本身无需安装（代码由 start_polar_nohup.sh" >&2
  echo "  注入的 PYTHONPATH=${POLAR_REPO}/src 提供）。请先执行：" >&2
  echo "    conda create -y --prefix ${POLAR_VENV} -c conda-forge --override-channels python=3.11" >&2
  echo "    ${POLAR_VENV}/bin/python -m ensurepip --upgrade" >&2
  echo "    ${POLAR_VENV}/bin/python -m pip install fastapi uvicorn httpx pydantic pyyaml" >&2
  exit 1
fi

# ─────── 获取本机 IP ──────────────────────────────────────────────────────────
# 必须与 vime 的 http_utils.get_host_info() 同源：那边用「UDP 探 8.8.8.8 看路由表」，
# 这里照抄。`hostname -I` 只是按内核顺序列全部地址取第一个，多网卡会分叉 —— 实测本机
# 四张（bond4.3000=10.1.30.48 带默认路由、data0.100=10.1.50.95、data1.100、docker0），
# hostname -I 头名是 .95 而 polar 实际该对外宣告 .48，于是 __POLAR_HOST__ 渲染成 .95、
# gateway_url 挂在一张 vime 拨不通的网卡上。bind_host 是 0.0.0.0 所以本地 curl 照样通，
# 故这个错法在单机自测时不暴露。
if [[ -z "${HOST_IP:-}" ]]; then
  HOST_IP="$(python3 -c '
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("8.8.8.8", 80))
    print(s.getsockname()[0])
finally:
    s.close()
' 2>/dev/null || true)"
  # 落回 hostname -I：无 python3 或无出口路由时不能让启动死在取 IP 这步。
  [[ -z "${HOST_IP}" || "${HOST_IP}" == "127.0.0.1" ]] && HOST_IP="$(hostname -I | awk '{print $1}')"
fi

# ─────── 获取 vime rollout/router 机器 IP（worker）───────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  vime + Polar 三方启动向导"
echo "════════════════════════════════════════════════════════"
echo ""
echo "  本机（Polar 宿主机）IP：${HOST_IP}"
echo ""

# VIME_NODE_IP 显式给值时跳过整个交接（只重启 polar、复用在跑的 vime；调试同理）。
# 此时 NPU_POOL 也必须显式给 —— 没有 handoff 可读。
if [[ -z "${VIME_NODE_IP:-}" ]]; then
  if [[ -z "${VIME_RDV_KEY:-}" ]]; then
    echo "ERROR: VIME_RDV_KEY 未设置。它必须与本次 vime 任务的 RDV_KEY 逐字相同 ——" >&2
    echo "  交接目录按它命名，是「本轮拓扑」与「上一轮残留」的唯一区分。" >&2
    echo "  vime rank0 启动时会打印该执行命令，照抄即可。" >&2
    echo "  只重启 polar、复用在跑的 vime：VIME_NODE_IP=<rollout 主机 IP> NPU_POOL='[0, 1, 2, 3]' bash \$0" >&2
    exit 1
  fi
  HANDOFF_DIR="${VIME_SCRATCH_DIR}/rendezvous/${VIME_RDV_KEY}"
  HANDOFF_ENV="${HANDOFF_DIR}/polar_handoff.env"
  HANDOFF_LAYOUT="${HANDOFF_DIR}/resolved_layout.yaml"
  VIME_IP_WAIT_SECS="${VIME_IP_WAIT_SECS:-9000}"
  echo "[polar-init] 等待 vime rank0 发布本轮拓扑（最长 ${VIME_IP_WAIT_SECS}s）"
  echo "             ${HANDOFF_ENV}"
  _waited=0
  while [[ ! -s "${HANDOFF_ENV}" ]]; do
    if (( _waited >= VIME_IP_WAIT_SECS )); then
      echo "ERROR: 等待 ${VIME_IP_WAIT_SECS}s 仍未出现：${HANDOFF_ENV}" >&2
      echo "  该文件由 vime rank0 渲染完 resource layout 后写入。请确认 vime 任务已提交、" >&2
      echo "  RDV_KEY 与本脚本的 VIME_RDV_KEY 一致（当前 ${VIME_RDV_KEY}）、共享盘已挂载。" >&2
      echo "  只重启 polar、复用在跑的 vime：VIME_NODE_IP=<rollout 主机 IP> bash \$0" >&2
      exit 1
    fi
    sleep 5; _waited=$(( _waited + 5 ))
    (( _waited % 60 )) || echo "[polar-init] 已等待 ${_waited}s ..."
  done
  echo "[polar-init] 拓扑已就绪（等待 ${_waited}s）"
  # 只 source 这份两行的交接文件，不 source vime 的整份 env：那里面
  # ASCEND_RT_VISIBLE_DEVICES 之类是 vime 某个 rank 私有的，polar 吃进去就错了。
  set -a; source "${HANDOFF_ENV}"; set +a
  VIME_NODE_IP="${VIME_ROLLOUT_HOST:-}"
  # 卡池同源自 layout。显式 NPU_POOL 仍优先（上面已设过则不覆盖）。
  if [[ -z "${NPU_POOL:-}" && -n "${POLAR_NPU_POOL:-}" ]]; then
    NPU_POOL="[$(printf '%s' "${POLAR_NPU_POOL}" | sed 's/,/, /g')]"
  fi
  # 模型名同源自 vime 的 HF_CKPT。显式 MODEL_SERVED 仍优先。
  if [[ -z "${MODEL_SERVED:-}" && -n "${VIME_HF_CKPT:-}" ]]; then
    MODEL_SERVED="${VIME_HF_CKPT}"
    echo "[polar-init] 模型名取自 vime 交接：${MODEL_SERVED}"
  fi
  if [[ -f "${HANDOFF_LAYOUT}" ]]; then
    echo "[polar-init] vime 本轮 resource layout："
    sed 's/^/[polar-init]   /' "${HANDOFF_LAYOUT}"
  fi
fi

if [[ -z "${NPU_POOL:-}" ]]; then
  echo "ERROR: 未取到 NPU_POOL（vime handoff 无 polar_reserved，且未显式设置）。" >&2
  echo "  在拓扑模板里给 roles.polar_reserved 加本机卡段，或显式设 NPU_POOL='[0, 1, 2, 3]'。" >&2
  exit 1
fi
if [[ -z "${MODEL_SERVED:-}" ]]; then
  echo "ERROR: 未取到 MODEL_SERVED（vime handoff 无 VIME_HF_CKPT，且未显式设置）。" >&2
  echo "  它必须与 vime 的 HF_CKPT 逐字相同 —— 填 **vime 那台机器** 上的权重路径，" >&2
  echo "  不是本机路径（它是 vLLM 注册的模型名，不用于打开文件）。" >&2
  echo "  走 VIME_NODE_IP 逃生舱时没有交接文件可读，须显式给。" >&2
  exit 1
fi


if [[ -z "${VIME_NODE_IP}" ]]; then
  echo "ERROR: 未获取到 vime 推理端点主机（handoff 里 VIME_ROLLOUT_HOST 为空）。" >&2
  exit 1
fi

# 交接路径含 RDV_KEY，本轮唯一 → 读到的必然是本次任务的，不需要看 mtime 也不需要人肉
# 核对新鲜度（本地 NFS 时钟比宿主机快约 30s，按 mtime 判会把陈旧读成新鲜）。
echo "  ────────────────────────────────────────────"
echo "   vime 推理端点主机 : ${VIME_NODE_IP}:${VLLM_ROUTER_PORT}"
echo "   Polar 卡池        : ${NPU_POOL}"
echo "  ────────────────────────────────────────────"
echo ""

# ─────── 测试网络连通性 ──────────────────────────────────────────────────────
echo "  [检查] 测试到 vime 机器的网络连通性..."
if ping -c 1 -W 3 "${VIME_NODE_IP}" >/dev/null 2>&1; then
  echo "  [ok] 网络连通"
else
  echo "  [warn] ping ${VIME_NODE_IP} 超时，但继续（ping 可能被防火墙屏蔽）"
fi
echo ""

# ─────── 动态 patch profile ──────────────────────────────────────────────────
# Every site value the profile needs is substituted here.
sed \
  -e "s|__POLAR_HOST__|${HOST_IP}|g" \
  -e "s|__VIME_ROUTER_HOST__|${VIME_NODE_IP}|g" \
  -e "s|__ROLLOUT_PORT__|${ROLLOUT_PORT}|g" \
  -e "s|__VLLM_ROUTER_PORT__|${VLLM_ROUTER_PORT}|g" \
  -e "s|__SOC_VERSION__|${SOC_VERSION}|g" \
  -e "s|__NPU_POOL__|${NPU_POOL}|g" \
  -e "s|__MODEL_SERVED__|${MODEL_SERVED}|g" \
  -e "s|__TASK_ASSETS_DIR__|${TASK_ASSETS_DIR}|g" \
  -e "s|__ASC_DEVKIT_DIR__|${ASC_DEVKIT_DIR}|g" \
  "${POLAR_PROFILE_SRC}" > "${POLAR_PROFILE_RUNTIME}"

# Unsubstituted tokens would reach Polar as literal hostnames.
if grep -q '__[A-Z_]*__' "${POLAR_PROFILE_RUNTIME}"; then
  echo "ERROR: ${POLAR_PROFILE_SRC} 渲染后仍有未替换占位符：" >&2
  grep -o '__[A-Z_]*__' "${POLAR_PROFILE_RUNTIME}" | sort -u >&2
  exit 1
fi

# ─────── 启动 Polar ──────────────────────────────────────────────────────────
# Ports come from the rendered profile, not duplicated here.
ROLLOUT_URL="$(sed -n 's|^\s*rollout_url:\s*||p' "${POLAR_PROFILE_RUNTIME}" | head -1)"
GATEWAY_URL="$(sed -n 's|^\s*gateway_url:\s*||p' "${POLAR_PROFILE_RUNTIME}" | head -1)"
ROUTER_URL="$(sed -n 's|^\s*sglang_router_url:\s*||p' "${POLAR_PROFILE_RUNTIME}" | head -1)"
echo "  [启动] Polar"
echo "         rollout : ${ROLLOUT_URL}"
echo "         gateway : ${GATEWAY_URL}"
echo "         推理端点: ${ROUTER_URL}"
# Wrong SOC compiles silently against the other chip; show it before launch.
echo "         SOC     : ${SOC_VERSION}   卡池: ${NPU_POOL}"
echo ""

# 不 source bin/activate：那是 venv 的产物，conda env 没有这个文件（conda 用
# `conda activate`，要先 shell hook）。而 activate 本质上只做三件事——改 PATH、设
# VIRTUAL_ENV、改提示符——这里真正需要的只有 PATH，且下面已经用绝对路径把
# POLAR_PYTHON 传下去了。所以直接前置 PATH，venv 和 conda env 都适用。
export PATH="${POLAR_VENV}/bin:${PATH}"
cd "${POLAR_REPO}"

POLAR_PROFILE="${POLAR_PROFILE_RUNTIME}" \
POLAR_PYTHON="${POLAR_VENV}/bin/python" \
POLAR_RUN_ID="polar_t2a_$(date +%Y%m%d_%H%M%S)" \
NO_PROXY="127.0.0.1,localhost,${HOST_IP},${VIME_NODE_IP}" \
no_proxy="127.0.0.1,localhost,${HOST_IP},${VIME_NODE_IP}" \
  bash deploy/ascend_operator/restart_polar_host.sh

# ─────── Polar 就绪后打印状态 ─────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Polar 已启动"
echo "════════════════════════════════════════════════════════"
echo ""
echo "  vime 应该会自动检测到 Polar 并开始训练。"
echo ""
echo "  SwanLab 实时监控（云端模式）："
echo "    训练开始后访问 https://swanlab.cn 查看曲线"
echo "    项目名: vime-polar-training"
echo ""
echo "  查看训练日志："
echo "    tail -f ${VIME_SCRATCH_DIR}/logs/train_*.log"
echo ""
echo "  查看 checkpoint："
echo "    ls ${VIME_SCRATCH_DIR}/checkpoints/"
echo ""
echo "  验证链路："
echo "    curl ${ROLLOUT_URL}/health   # Polar rollout"
echo "    curl ${GATEWAY_URL}/health   # Polar gateway"
echo ""
echo "  Observer UI: http://${HOST_IP}:$(sed -n 's|^\s*port:\s*||p' "${POLAR_PROFILE_RUNTIME}" | head -1)"
echo "════════════════════════════════════════════════════════"
