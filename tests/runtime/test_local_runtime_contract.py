"""LocalRuntime contract tests.

只测「错了会静默」的路径 —— 会明确报错的靠实跑自然暴露，不值得写测试。
LocalRuntime 比 DockerRuntime 好测:它本来就是跑本地子进程,多数用例直接真跑 bash。

用例与它们各自防的问题见 LOCAL_RUNTIME_DESIGN.md §8.1。
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
from pathlib import Path

import pytest

from polar.runtime.factory import create_runtime
from polar.runtime.local import LocalRuntime
from polar.runtime.models import RuntimeSpec

WORKDIR = "/polar/session/agent_workdir"


def _spec(**over) -> RuntimeSpec:
    base = dict(backend="local", image="unused-process-level", workdir=WORKDIR)
    base.update(over)
    return RuntimeSpec(**base)


def _rt(tmp_path: Path, name: str = "sk-polar-test", **over) -> LocalRuntime:
    return LocalRuntime(_spec(**over), name, tmp_path / name)


# ── 1. agent 与 eval 两个实例必须落在不同宿主目录 ────────────────────────────────
# 防的是 §3 的核心风险:gateway 每 session 建两个 runtime(agent 用 session_dir、
# fresh eval judge 用 session_dir/eval_runtime),docker 下各自 bind 到同名
# /polar/session。没有 mount namespace 时若不重写,两者共用一个目录,
# 「judge 在 fresh 容器上判分」的语义静默作废。

def test_two_instances_resolve_to_different_host_dirs(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    agent = LocalRuntime(_spec(), "sid", session_dir)
    judge = LocalRuntime(_spec(), "sid-eval", session_dir / "eval_runtime")

    assert agent._rewrite(WORKDIR) == str(session_dir / "agent_workdir")
    assert judge._rewrite(WORKDIR) == str(session_dir / "eval_runtime" / "agent_workdir")
    assert agent._rewrite(WORKDIR) != judge._rewrite(WORKDIR)

    async def go() -> None:
        for rt in (agent, judge):
            await rt.start()
            r = await rt.exec("pwd > marker.txt && cat marker.txt")
            assert r.return_code == 0, r.stderr
            assert r.stdout is not None and r.stdout.strip() == rt._rewrite(WORKDIR)

    asyncio.run(go())
    # 两份 marker 各在自己的目录里,内容不同 —— 隔离的直接证据
    a = (session_dir / "agent_workdir" / "marker.txt").read_text().strip()
    j = (session_dir / "eval_runtime" / "agent_workdir" / "marker.txt").read_text().strip()
    assert a != j


# ── 2. cwd / env value / 命令字符串三处前缀都要重写 ─────────────────────────────
# 漏任一处就走错目录,且不报错。三处分别对应:agent 的 workdir、
# claude_code.py:24 的 CLAUDE_CONFIG_DIR、:118 的日志 tee 目标。

def test_prefix_rewritten_in_cwd_env_and_command(tmp_path: Path) -> None:
    rt = _rt(tmp_path, env={"CLAUDE_CONFIG_DIR": "/polar/session/.claude"})

    async def go() -> None:
        await rt.start()
        # 命令字符串里的绝对路径(模拟 agent 日志 tee 的目标)
        r = await rt.exec(
            "mkdir -p /polar/session/logs/agent && "
            "echo hi > /polar/session/logs/agent/claude-code.txt && "
            'echo "cfg=$CLAUDE_CONFIG_DIR" && pwd'
        )
        assert r.return_code == 0, r.stderr
        out = r.stdout or ""
        # env value 被重写
        assert f"cfg={rt.session_dir}/.claude" in out
        # cwd 被重写
        assert str(rt.session_dir / "agent_workdir") in out
        # 命令里的路径被重写 —— 文件真落在 session 内,而不是宿主的 /polar/session
        assert (rt.session_dir / "logs" / "agent" / "claude-code.txt").read_text() == "hi\n"

    asyncio.run(go())
    assert not Path("/polar/session").exists(), "重写漏了:真在宿主根上建了 /polar/session"


# ── 3. stop() 之后孙进程也必须没了 ───────────────────────────────────────────────
# 这是全套里最贵的 bug。docker 下「拆容器即全杀」是免费的;进程级下 base.cancel()
# 只杀直接子进程,而 agent 会派生 CLI、编译器、eval 子进程。漏一个就攥着 NPU flock
# (npu_lease_exec.py)不放,几个 session 之后卡池空转。

def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def test_stop_kills_grandchildren(tmp_path: Path) -> None:
    rt = _rt(tmp_path)
    pid_file = tmp_path / "pids.txt"

    async def go() -> list[int]:
        await rt.start()
        # 直接子进程是 bash;它再派生两个 sleep = 孙进程。只杀直接子进程的话
        # 这两个 sleep 会活下来。
        await rt.exec(
            f"( sleep 300 & echo $! >> {pid_file}; sleep 300 & echo $! >> {pid_file}; ) ; "
            "sleep 0.2",
            timeout_sec=30,
        )
        pids = [int(x) for x in pid_file.read_text().split()]
        assert len(pids) == 2 and all(_alive(p) for p in pids), "前置不成立:孙进程没起来"
        await rt.stop()
        return pids

    pids = asyncio.run(go())
    deadline = time.time() + 15
    while time.time() < deadline and any(_alive(p) for p in pids):
        time.sleep(0.3)
    leaked = [p for p in pids if _alive(p)]
    for p in leaked:  # 别把残留留给后续测试
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass
    assert not leaked, f"孙进程未被 stop() 清掉: {leaked}"


# ── 4. factory 不能因 kwargs.ascend 拒掉 local ──────────────────────────────────
# supports_ascend 忘了置 True 时,operator rollout 在本 backend 上永远起不来
# (factory.py:55 硬拒)。反面:apptainer 仍应被拒,闸门强度不变。

def test_factory_accepts_local_with_ascend_kwargs(tmp_path: Path) -> None:
    spec = _spec(kwargs={"ascend": {"pool": "0,1", "lock_dir": "/dev/shm/npu-locks",
                                    "lease_at_start": False}})
    assert isinstance(create_runtime(spec, "sid", tmp_path / "s"), LocalRuntime)

    with pytest.raises(ValueError, match="Ascend NPU passthrough"):
        create_runtime(
            RuntimeSpec(backend="apptainer", image="x", kwargs={"ascend": {"pool": "0"}}),
            "sid", tmp_path / "s2",
        )


# ── 5. agent 实例写的文件,judge 实例要能取到 ────────────────────────────────────
# 前缀重写最刁的一处。agent 的 workdir 重写到 <session>/agent_workdir,judge 的重写到
# <session>/eval_runtime/agent_workdir —— 两个不同宿主目录。而 operator_judge._abs()
# 拼出的绝对路径会按 **judge 自己的实例**重写,落进 eval_runtime/ 下面,可 submission
# 其实是 agent 写在另一处的。docker 下靠 gateway 显式传 submission_host_path 转移
# (node.py:878);这里验的是那条转移路径在 LocalRuntime 下同样有效 ——
# download_file 用宿主绝对路径取 agent 的产物,不被 judge 的重写带偏。
# 判据在实跑侧是 judge 出分非 0.2(submission_missing 地板分)。

def test_judge_instance_can_fetch_agent_artifact(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    agent = LocalRuntime(_spec(), "sid", session_dir)
    judge = LocalRuntime(_spec(), "sid-eval", session_dir / "eval_runtime")
    rel = "output/submission/op_impl.tar.gz"

    async def go() -> None:
        await agent.start()
        await judge.start()
        r = await agent.exec(f"mkdir -p $(dirname {rel}) && echo payload > {rel}")
        assert r.return_code == 0, r.stderr

        # gateway 传的是**宿主绝对路径**(submission_host_path),不是容器内路径
        host_path = agent.session_dir / "agent_workdir" / rel
        assert host_path.is_file(), "前置不成立:agent 没写出产物"

        fetched = tmp_path / "fetched.tar.gz"
        await judge.download_file(str(host_path), str(fetched))
        assert fetched.read_text() == "payload\n"

        # 反面:judge 按**自己**的 /polar/session 去找,必然找不到 —— 这正是
        # submission_missing 的成因,确认它确实是两个不同目录。
        with pytest.raises(FileNotFoundError):
            await judge.download_file(f"{WORKDIR}/{rel}", str(tmp_path / "nope.gz"))

    asyncio.run(go())


# ── 附:volumes 软链 ─────────────────────────────────────────────────────────────
# kwargs.volumes 契约不改,由 LocalRuntime 解释成软链。session 内的按 session 建,
# 全局的共享。源不存在要明确报错 —— docker 会静默建空目录,asc-devkit 就这样丢过。

def test_volumes_become_symlinks(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    (canonical / "tools").mkdir(parents=True)
    (canonical / "tools" / "env.sh").write_text("echo env\n")
    rt = _rt(tmp_path, kwargs={"volumes": [
        f"{canonical}:/opt/canonical_test:ro",
        f"{canonical / 'tools'}:{WORKDIR}/tools:ro",
    ]})

    async def go() -> None:
        await rt.start()
        # session 内那条必须是**真目录拷贝**而非软链 —— tools/ascendc_eval_pipeline.sh
        # 会 readlink -f 自己再取上一级当 WORK_ROOT，软链会让它解析到共享树，
        # judge_out 写进 operator_runtime_t2a（实测踩过）。docker 下那是 bind 挂载。
        tools = rt.session_dir / "agent_workdir" / "tools"
        assert tools.is_dir() and not tools.is_symlink(), "session 内的 volume 必须是拷贝"
        assert (tools / "env.sh").is_file()
        # 解析后仍在 session 内 —— 这是防回归的关键断言
        assert str((tools / "env.sh").resolve()).startswith(str(rt.session_dir))
        # 拷贝是只读的（profile 标 :ro）
        assert not (tools / "env.sh").stat().st_mode & 0o222
        r = await rt.exec("cat tools/env.sh")
        assert r.return_code == 0 and "echo env" in (r.stdout or "")
        # 全局那条仍是软链
        assert Path("/opt/canonical_test").is_symlink()
        await rt.stop()
        # stop() 恢复写位，否则 gateway 的 rmtree(session_dir) 清不掉、每 rollout 漏一个
        assert (tools / "env.sh").stat().st_mode & 0o200, "写位没恢复，session 目录清不掉"

    asyncio.run(go())
    for stale in (Path("/opt/canonical_test"),):
        if stale.is_symlink():
            stale.unlink()


def test_missing_volume_source_is_loud(tmp_path: Path, caplog) -> None:
    rt = _rt(tmp_path, kwargs={"volumes": [f"{tmp_path}/does-not-exist:{WORKDIR}/tools:ro"]})
    with caplog.at_level("ERROR"):
        asyncio.run(rt.start())
    assert any("volume source does not exist" in r.getMessage() for r in caplog.records)
    assert not (rt.session_dir / "agent_workdir" / "tools").exists()
