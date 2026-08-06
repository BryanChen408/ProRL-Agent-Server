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
POLAR_VENV="${POLAR_VENV:-/mnt/model/corlorlight_models/mingchengzou/ProRL2/polar-venv}"
POLAR_PROFILE_SRC="${POLAR_REPO}/deploy/ascend_operator/profile.t2a.yaml"
POLAR_PROFILE_RUNTIME=/tmp/polar_profile_runtime.yaml
# vime rollout/router 节点（worker）IP —— Polar 的推理端点主机部分。
# master pod 在启动时把 worker IP 写到这个共享存储文件。
VIME_ROUTER_IP_FILE="/mnt/model/corlorlight_models/mingchengzou/ProRL2/scratch/vime_router_ip.txt"

# ─────── 检查前置条件 ─────────────────────────────────────────────────────────
if [[ ! -x "${POLAR_VENV}/bin/python" ]]; then
  echo "ERROR: polar venv 不存在于 ${POLAR_VENV}" >&2
  echo "  请先执行：" >&2
  echo "    python3 -m venv ${POLAR_VENV}" >&2
  echo "    source ${POLAR_VENV}/bin/activate" >&2
  echo "    pip install -e ${POLAR_REPO}" >&2
  exit 1
fi

# ─────── 获取本机 IP ──────────────────────────────────────────────────────────
HOST_IP="${HOST_IP:-$(hostname -I | awk '{print $1}')}"

# ─────── 获取 vime rollout/router 机器 IP（worker）───────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  vime + Polar 三方启动向导"
echo "════════════════════════════════════════════════════════"
echo ""
echo "  本机（Polar 宿主机）IP：${HOST_IP}"
echo ""

# VIME_NODE_IP env var overrides the file (debugging).
if [[ -z "${VIME_NODE_IP:-}" ]]; then
  # Missing file = this run's vime master isn't ready. No interactive prompt:
  # platform jobs have no tty, so read would hang until timeout.
  if [[ ! -s "${VIME_ROUTER_IP_FILE}" ]]; then
    echo "ERROR: 未找到 vime router IP 文件：${VIME_ROUTER_IP_FILE}" >&2
    echo "  该文件由 vime master 完成 IP 交会后写入。请确认 vime 任务已提交、" >&2
    echo "  master pod 已打印出 rollout 节点 IP、共享盘已挂载。" >&2
    echo "  调试可绕过：VIME_NODE_IP=<worker IP> bash \$0" >&2
    exit 1
  fi
  VIME_NODE_IP="$(tr -d '[:space:]' < "${VIME_ROUTER_IP_FILE}")"
fi

if [[ -z "${VIME_NODE_IP}" ]]; then
  echo "ERROR: 未获取到 vime worker IP，退出。" >&2
  exit 1
fi

# Show what we read — this script is run by hand, so a human catches a wrong IP
# on the spot. Cheaper and more reliable than any automatic freshness check.
echo "  ────────────────────────────────────────────"
echo "   vime rollout/router IP : ${VIME_NODE_IP}"
echo "   写入时间       : $(date -r "${VIME_ROUTER_IP_FILE}" '+%F %T' 2>/dev/null || echo 未知)"
echo "  ────────────────────────────────────────────"
echo "  ↑ 确认是本次 vime 任务打印的 IP，不对就 Ctrl-C。"

echo ""
echo "  vime worker（rollout/router）IP：${VIME_NODE_IP}"
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
# Substitute host tokens only; ports stay wherever the profile declares them.
sed \
  -e "s|__POLAR_HOST__|${HOST_IP}|g" \
  -e "s|__VIME_ROUTER_HOST__|${VIME_NODE_IP}|g" \
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
echo ""

source "${POLAR_VENV}/bin/activate"
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
echo "    tail -f /mnt/model/corlorlight_models/mingchengzou/ProRL2/scratch/logs/train_*.log"
echo ""
echo "  查看 checkpoint："
echo "    ls /mnt/model/corlorlight_models/mingchengzou/ProRL2/scratch/checkpoints/"
echo ""
echo "  验证链路："
echo "    curl ${ROLLOUT_URL}/health   # Polar rollout"
echo "    curl ${GATEWAY_URL}/health   # Polar gateway"
echo ""
echo "  Observer UI: http://${HOST_IP}:$(sed -n 's|^\s*port:\s*||p' "${POLAR_PROFILE_RUNTIME}" | head -1)"
echo "════════════════════════════════════════════════════════"
