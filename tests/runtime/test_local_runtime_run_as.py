"""`run_as` 降权用例 —— §5.1 的写保护是否真的成立。

需要预先建好 polar 用户（deploy/ascend_operator/setup_local_runtime_user.sh），
没有就 skip 而不是让整套测试红。
"""

from __future__ import annotations

import asyncio
import pwd
from pathlib import Path

import pytest

from polar.runtime.local import LocalRuntime
from polar.runtime.models import RuntimeSpec

WORKDIR = "/polar/session/agent_workdir"
RUN_AS = "polar"


def _has_user() -> bool:
    try:
        pwd.getpwnam(RUN_AS)
    except KeyError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _has_user(), reason=f"需要 {RUN_AS} 用户：先跑 setup_local_runtime_user.sh"
)


def _rt(tmp_path: Path, volumes: list[str] | None = None) -> LocalRuntime:
    spec = RuntimeSpec(
        backend="local", image="unused", workdir=WORKDIR,
        kwargs={"run_as": RUN_AS, **({"volumes": volumes} if volumes else {})},
    )
    return LocalRuntime(spec, "sid-runas", tmp_path / "session")


def test_runs_as_unprivileged_user_and_owns_its_workdir(tmp_path: Path) -> None:
    """降权生效，且 agent 能写自己的 workdir。

    后半句是踩过的坑：start() 的 chown 跑在前，而 workdir 原先由 exec() 以 root 身份
    mkdir 出来 —— agent 拿到一个 root:root 755 的工作目录，第一次写就 Permission
    denied。所以 workdir 必须在 chown **之前**建好。
    """
    rt = _rt(tmp_path)

    async def go() -> None:
        await rt.start()
        r = await rt.exec('id -un; echo "HOME=$HOME"; echo payload > probe.txt; cat probe.txt')
        assert r.return_code == 0, f"降权后写自己的 workdir 失败: {r.stderr}"
        out = r.stdout or ""
        assert RUN_AS in out, f"没跑在 {RUN_AS} 身份下: {out}"
        assert f"HOME={rt.session_dir}" in out, "HOME 没指进 session（登录 shell 冲掉了？）"
        assert "payload" in out
        await rt.stop()

    asyncio.run(go())
    written = rt.session_dir / "agent_workdir" / "probe.txt"
    assert written.is_file()
    assert written.owner() == RUN_AS, "产物属主不是降权用户"


def test_shared_tree_is_write_protected(tmp_path: Path) -> None:
    """降权用户改不了只读 volume 拷贝，也改不到源树。

    这是 §5.1 的全部意义：软链/拷贝没有 docker `:ro` 的内核语义，root 能写穿它改坏
    共享树，而后果是静默漂移（asc-devkit 被改坏 → docs-search 返回错内容 → agent
    照着写出编不过的 kernel，一份坏了污染后续所有 session）。
    """
    src = tmp_path / "canonical_tools"
    src.mkdir()
    (src / "env.sh").write_text("echo original\n")
    rt = _rt(tmp_path, volumes=[f"{src}:{WORKDIR}/tools:ro"])

    async def go() -> None:
        await rt.start()
        # 读得到
        r = await rt.exec("cat tools/env.sh")
        assert r.return_code == 0 and "original" in (r.stdout or "")
        # 改不了（内核拒写，不是事后检测）。
        # 注意重定向失败由 **shell 本身**报到它自己的 stderr，命令里的 2>&1 拦不到
        # （那只作用于 echo，而 echo 根本没跑起来）—— 所以查 stderr + 内容未变。
        r = await rt.exec("echo hacked > tools/env.sh; echo ---; cat tools/env.sh")
        combined = (r.stdout or "") + (r.stderr or "")
        assert "Permission denied" in combined, f"降权用户竟能写只读拷贝: {combined[:200]}"
        assert "hacked" not in (r.stdout or ""), "内容被改了"
        # 加不了新文件（touch 自己报错，2>&1 拦得到）
        r = await rt.exec("touch tools/hack.sh 2>&1; echo done")
        assert "Permission denied" in ((r.stdout or "") + (r.stderr or "")), \
            f"能往只读拷贝里加文件: {r.stdout}"
        await rt.stop()

    asyncio.run(go())
    # 源树一字未改
    assert (src / "env.sh").read_text() == "echo original\n"
    assert not (src / "hack.sh").exists()
