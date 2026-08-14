"""LocalRuntime 并发用例：多 session 同容器共享全局软链。

Docker 下 N 个 session = N 个容器，各自 bind 自己的挂载，永远不会互相干扰。
LocalRuntime 下全局 volume（/opt/canonical、/opt/asc-devkit）是**一份共享软链**，
N 个 session 的 start() 会并发去建同一个 dest —— 这是 docker 路径上不存在的竞态。

实测规模参考 polar-session-scale：单 run 37 session / 2.5 小时，gateway 的
max_init_workers 默认 8，所以并发 start() 是常态而非边缘情况。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from polar.runtime.local import LocalRuntime
from polar.runtime.models import RuntimeSpec

WORKDIR = "/polar/session/agent_workdir"
N = 8  # = gateway.max_init_workers 默认值


def _make(tmp_path: Path, i: int, shared_src: Path, global_dst: Path) -> LocalRuntime:
    spec = RuntimeSpec(
        backend="local", image="unused", workdir=WORKDIR,
        kwargs={"volumes": [
            f"{shared_src}:{global_dst}:ro",             # 全局：N 个 session 抢同一个 dest
            f"{shared_src / 'tools'}:{WORKDIR}/tools:ro",  # session 内：各自一份，不该冲突
        ]},
    )
    return LocalRuntime(spec, f"sid-{i}", tmp_path / f"session-{i}")


def test_concurrent_start_shares_global_symlink(tmp_path: Path) -> None:
    """N 个 session 同时 start()：全局软链只有一份且指向正确，无一失败。

    防的是 _link_volumes 里 is_symlink() 检查与 symlink_to() 之间的窗口 ——
    另一个 session 抢先建好，本 session 撞 FileExistsError，异常从 start() 抛出，
    session init 挂掉。只在并发下出现，单测跑单个实例永远发现不了。
    """
    src = tmp_path / "canonical"
    (src / "tools").mkdir(parents=True)
    (src / "tools" / "env.sh").write_text("echo shared\n")
    global_dst = tmp_path / "opt_canonical_shared"

    runtimes = [_make(tmp_path, i, src, global_dst) for i in range(N)]

    async def go() -> list[BaseException | None]:
        results = await asyncio.gather(
            *(rt.start() for rt in runtimes), return_exceptions=True
        )
        return [r if isinstance(r, BaseException) else None for r in results]

    errors = [e for e in asyncio.run(go()) if e is not None]
    assert not errors, f"并发 start() 有 {len(errors)} 个失败: {errors[:3]}"

    # 全局软链恰好一份，指向真实源
    assert global_dst.is_symlink()
    assert global_dst.resolve() == src.resolve()
    assert (global_dst / "tools" / "env.sh").read_text() == "echo shared\n"

    # session 内那份各自独立
    for rt in runtimes:
        link = rt.session_dir / "agent_workdir" / "tools"
        assert link.is_symlink(), f"{rt.session_id} 缺 session 内软链"
        assert (link / "env.sh").is_file()

    async def stop_all() -> None:
        await asyncio.gather(*(rt.stop() for rt in runtimes), return_exceptions=True)

    asyncio.run(stop_all())

    # session 内的清掉；全局的**留着** —— 别的 session 可能仍在用
    for rt in runtimes:
        assert not (rt.session_dir / "agent_workdir" / "tools").is_symlink()
    assert global_dst.is_symlink(), "全局软链被某个 session 的 stop() 误删"
    global_dst.unlink()


def test_concurrent_exec_isolated(tmp_path: Path) -> None:
    """N 个 session 并发 exec：各自的 cwd / HOME / TMPDIR 不串。

    进程级下这些是同一个文件系统里的路径，写错前缀就会互相覆盖，
    而表现是内容错乱而非报错。
    """
    src = tmp_path / "canonical"
    (src / "tools").mkdir(parents=True)
    runtimes = [_make(tmp_path, i, src, tmp_path / "gdst") for i in range(N)]

    async def go() -> list[str]:
        await asyncio.gather(*(rt.start() for rt in runtimes))
        cmds = [
            rt.exec(
                f'echo "{rt.session_id}" > mine.txt && '
                'echo "$HOME|$TMPDIR|$(pwd)|$(cat mine.txt)"'
            )
            for rt in runtimes
        ]
        results = await asyncio.gather(*cmds)
        for r in results:
            assert r.return_code == 0, r.stderr
        return [(r.stdout or "").strip() for r in results]

    lines = asyncio.run(go())
    assert len(set(lines)) == N, f"并发 exec 输出串了: {lines}"
    for rt, line in zip(runtimes, lines):
        home, tmpdir, cwd, mine = line.split("|")
        assert home == str(rt.session_dir)
        assert tmpdir == str(rt.session_dir / "tmp")
        assert cwd == str(rt.session_dir / "agent_workdir")
        assert mine == rt.session_id, "各 session 的文件互相覆盖了"

    async def stop_all() -> None:
        await asyncio.gather(*(rt.stop() for rt in runtimes), return_exceptions=True)

    asyncio.run(stop_all())
    g = tmp_path / "gdst"
    if g.is_symlink():
        g.unlink()
