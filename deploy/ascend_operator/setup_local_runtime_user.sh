#!/usr/bin/env bash
# LocalRuntime 的降权前置：在容器内建一次即可，幂等，可反复跑。
#
# 为什么要降权（LOCAL_RUNTIME_DESIGN.md §5.1）：进程级 runtime 把 kwargs.volumes 实现
# 成软链，而软链没有 docker `:ro` 的内核语义 —— agent 以 root 跑就能写穿它改坏共享树。
# 后果不是崩而是**静默漂移**：asc-devkit 被改坏 → docs-search 返回错内容 → agent 照着
# 写出编不过的 kernel，且一份坏了污染后续所有 session。事后 git status 只能检测不能防护。
#
# 可行的关键（沙箱实测）：NPU 设备属主是 uid/gid 1000、权限 0666，**非 root 用卡不需要
# 任何 capability** —— capability 恰恰是受限环境最难拿的（同一次实测里 CAP_SYS_ADMIN
# 就是没有的）。
#
# 用法：bash deploy/ascend_operator/setup_local_runtime_user.sh [用户名]
set -uo pipefail

USER_NAME="${1:-${POLAR_RUN_AS:-polar}}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "${HERE}/../.." && pwd -P)"

echo "=== 1. 建用户 ${USER_NAME} ==="
if id "${USER_NAME}" >/dev/null 2>&1; then
  echo "  已存在：$(id "${USER_NAME}")"
else
  useradd -M -s /bin/bash "${USER_NAME}" || { echo "useradd 失败"; exit 1; }
  echo "  已建：$(id "${USER_NAME}")"
fi

echo
echo "=== 2. 加 NPU 设备属组 ==="
# 设备属主实测是 uid/gid 1000（沙箱 0666、训练容器 0660）。0666 时不加组也能用，
# 但权限位随镜像变，加了不会错。
DEVGID=""
for dev in /dev/davinci0 /dev/davinci_manager; do
  [[ -e "$dev" ]] && { DEVGID="$(stat -c %g "$dev")"; break; }
done
if [[ -n "${DEVGID}" ]]; then
  usermod -aG "${DEVGID}" "${USER_NAME}" 2>/dev/null \
    && echo "  已加入 gid ${DEVGID}（$(getent group "${DEVGID}" | cut -d: -f1 || echo 无名)）" \
    || echo "  加组失败（0666 下不影响用卡）"
  ls -l /dev/davinci0 2>/dev/null | sed 's/^/  /'
else
  echo "  本机无 /dev/davinci* —— 非 NPU 机器，跳过（判分时才需要卡）"
fi

echo
echo "=== 3. NPU 租约锁目录（npu_lease_exec.py 要 flock，必须可写）==="
LOCK_DIR="${POLAR_NPU_LOCK_DIR:-/dev/shm/npu-locks}"
install -d -m 0775 "${LOCK_DIR}" && chgrp "${DEVGID:-0}" "${LOCK_DIR}" 2>/dev/null
chmod 0777 "${LOCK_DIR}"   # 多用户共抢同一池，简单起见给全权
ls -ld "${LOCK_DIR}" | sed 's/^/  /'

echo
echo "=== 4. 共享树置 0555（内核拒写，这是真防护）==="
# 顺序要紧：0555 之后连 polar 自己也改不了这两棵树，而 launcher 的 asc-devkit 自举
# （浅克隆 + docs/zh/api 软链修复）需要写权限。所以自举必须在本脚本之前跑完 ——
# 那是 run 间隙动作，不冲突。若 asc-devkit 尚未自举，先跑一次 launcher 再回来。
if [[ -n "${SHARE_ROOT:-}" ]] || true; then :; fi
# 只读挂载的三个源。asc-devkit 路径与 launcher 的探测保持一致。
SHARE_ROOT=""
for cand in /mnt/host-model /mnt/model /models; do
  [[ -d "${cand}/cbx" ]] && { SHARE_ROOT="${cand}"; break; }
done
TREES=("${REPO}/operator_runtime_t2a")
[[ -n "${SHARE_ROOT}" && -d "${SHARE_ROOT}/cbx/asc-devkit-9.0.0" ]] \
  && TREES+=("${SHARE_ROOT}/cbx/asc-devkit-9.0.0")
for tree in "${TREES[@]}"; do
  if [[ ! -d "${tree}" ]]; then echo "  跳过（不存在）：${tree}"; continue; fi
  # 目录要 r-x（进得去），文件 r--；保留可执行位（tools/*.sh 要能跑）。
  chown -R root:root "${tree}" 2>/dev/null
  find "${tree}" -type d -exec chmod 0555 {} + 2>/dev/null
  find "${tree}" -type f -exec chmod a-w {} + 2>/dev/null
  echo "  已锁：${tree}"
  echo "    $(ls -ld "${tree}" | awk '{print $1, $3, $4}')"
done

echo
echo "=== 5. 验证：非 root 能读、不能写 ==="
T="${TREES[0]}"
runuser -u "${USER_NAME}" -- test -r "${T}" \
  && echo "  可读 OK" || echo "  !! 不可读 —— 降权后 agent 拿不到 canonical"
if runuser -u "${USER_NAME}" -- touch "${T}/.write_probe" 2>/dev/null; then
  echo "  !! 可写 —— 防护没生效"; rm -f "${T}/.write_probe"
else
  echo "  拒写 OK（内核级，非事后检测）"
fi
# a-w 只去 w 不动 x，所以仓里带 x 的文件仍可执行。canonical tools 在仓里是 0644
# （git ls-files -s 显示 100644）—— 它们从来不靠 x 位跑，judge_command 是
# `bash tools/ascendc_eval_pipeline.sh`，显式经 bash 调。所以这里只报告不判失败。
for probe in "${T}/tools/env.sh" "${T}/tools/ascendc_eval_pipeline.sh"; do
  [[ -f "${probe}" ]] && echo "  $(stat -c '%A' "${probe}") $(basename "${probe}")"
done
echo "  （仓里是 0644，靠 bash 显式调用，不需要 x 位）"

echo
echo "=== 6. 非 root 能否真用 NPU（核心判据）==="
if [[ -n "${DEVGID}" ]]; then
  CARD="${CARD:-0}"
  install -d -o "${USER_NAME}" -m 0755 "/tmp/${USER_NAME}-home"
  runuser -u "${USER_NAME}" -- env "HOME=/tmp/${USER_NAME}-home" \
    ASCEND_RT_VISIBLE_DEVICES="${CARD}" python3 -c "
import torch, torch_npu
print('  device_count', torch.npu.device_count())
torch.npu.set_device(0)
print('  nonroot tensor ok', (torch.ones(8, device='npu:0') + 1).sum().item())
" 2>&1 | grep -vE "^/etc/profile|Warning:" | tail -3
  echo "  （device_count 0 = 卡被别的进程占着或未分配，非权限问题）"
else
  echo "  跳过：本机无 NPU 设备"
fi

echo
echo "完成。profile.local.yaml 的 operator.runtime.run_as 应为 ${USER_NAME}。"
echo "不想先验降权就删掉那一行 —— 链路照跑，只是丢掉写保护。"
