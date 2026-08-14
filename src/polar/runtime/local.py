"""Process-level rollout runtime — sessions are subprocesses, not nested containers.

WHY THIS EXISTS. The Docker/Apptainer backends both need to *create* a container. In a
restricted environment we only get in-container access: no host root, no docker socket,
and no CAP_SYS_ADMIN (measured: ``CapEff=00000000a80425fb``, so DinD, DooD **and**
``unshare --mount`` are all out). This backend drops the nesting entirely and runs each
session as a subprocess of the gateway, inside the gateway's own container.

WHAT IS AND ISN'T LOST versus DockerRuntime:

  * NPU card scoping — **unchanged**. The profile sets ``lease_at_start=False``, so the
    card is leased *inside* the sandbox by ``tools/npu_lease_exec.py`` (flock, driven by
    ``POLAR_NPU_LEASE_POOL`` / ``POLAR_NPU_LOCK_DIR`` in the runtime env). DockerRuntime's
    only extra job was mounting the driver — which a process here inherits for free. The
    existing scheme was always *soft* isolation anyway (all cards visible, scoped by
    ``ASCEND_RT_VISIBLE_DEVICES``; see ``ascend.py``), so nothing regresses.
  * Network isolation and resource quotas — not lost, **never used**: the profile sets
    only ``image`` and ``network: host``, with no cpus/memory_mb/storage_mb.
  * Read-only mounts — genuinely lost; a symlink cannot enforce ``:ro``. Recovered by
    running the agent as a non-root user with the shared trees at 0555 (see
    ``LOCAL_RUNTIME_DESIGN.md`` §5.1) — feasible because the NPU devices are owned by
    uid/gid 1000, not root, so an unprivileged user needs no capability to use a card.
  * Process cleanup — the one real regression. "Tear down the container" was free; here we
    must kill a process group by hand (see ``stop``).

THE CORE MECHANISM (§3 of the design doc) is ``/polar/session`` prefix rewriting. The
gateway builds **two** runtimes per session — the agent on ``session_dir`` and a fresh
eval judge on ``session_dir/eval_runtime`` (``gateway/node.py:554``) — and under Docker
each bind-mounts its own host directory at the *same* path. With no mount namespace we
cannot reproduce that, so instead every ``/polar/session`` occurrence in a command
string, an env value, or a cwd is rewritten to *this instance's* real host directory.
The reference surface is small and was enumerated exhaustively: the constant itself,
``agent/presets/claude_code.py:24`` (``_config_dir``) and ``:118`` (agent log tee), plus
the cannbot-only prepare in ``load_polar_profile.py`` (we run ``legacy``).

Skipping the rewrite would not raise: the two instances would silently share one
directory and quietly void "the judge scores in a fresh container".
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import signal
from pathlib import Path

from polar.runtime.base import BaseRuntime
from polar.runtime.models import ExecResult, RuntimeSpec

logger = logging.getLogger(__name__)


class LocalRuntime(BaseRuntime):
    """Run each rollout session as a subprocess in the gateway's own container."""

    _STOP_GRACE = 5.0  # seconds between SIGTERM and SIGKILL on the process group

    def __init__(self, spec: RuntimeSpec, session_id: str, session_dir: Path) -> None:
        super().__init__(spec, session_id, session_dir)
        self._pgids: set[int] = set()
        # Non-root execution guard for the shared read-only trees. Empty = run as-is
        # (tests, or a deployment that accepts the exposure).
        self._run_as = str(spec.kwargs.get("run_as") or "").strip()
        self._symlinks: list[Path] = []
        self._copies: list[Path] = []

    @property
    def runtime_id(self) -> str:
        return f"local-{self.session_id}"

    # ── capabilities ────────────────────────────────────────────────────────────
    # Ordered as in BaseRuntime so the diff against docker.py stays readable.

    @property
    def supports_gpus(self) -> bool:
        return True

    @property
    def can_disable_internet(self) -> bool:
        # No netns available (no CAP_SYS_ADMIN). The profile sets allow_internet=true,
        # so factory._validate_runtime_capabilities never reaches this gate; returning
        # False keeps us honest if some future profile does ask for isolation.
        return False

    @property
    def supports_ascend(self) -> bool:
        """True with a different meaning than DockerRuntime's.

        There it means "I mount the driver and scope the card". Here it means "the
        Ascend environment is *already* present, inherited from the gateway's own
        container" — no ``--privileged``, no ``-v /dev:/dev``, nothing to set up.
        The flag must still be True or ``factory.py:55`` hard-rejects ``kwargs.ascend``
        and an operator rollout can never run on this backend.
        """
        return True

    # ── prefix rewriting: the heart of this backend ─────────────────────────────

    def _rewrite(self, text: str) -> str:
        """Map ``/polar/session`` to THIS instance's host directory.

        Applied to command strings, env values and cwd. Docker gets this from the bind
        mount; without a mount namespace it has to be textual.
        """
        return text.replace(self.runtime_session_dir, str(self.session_dir))

    def _rewrite_env(self, env: dict[str, str]) -> dict[str, str]:
        return {key: self._rewrite(str(value)) for key, value in env.items()}

    # ── lifecycle ───────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._destroyed:
            raise RuntimeError("local runtime was already destroyed")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("logs/agent", "logs/eval", "eval_artifacts", "tmp"):
            (self.session_dir / sub).mkdir(parents=True, exist_ok=True)
        # Create the workdir HERE, before the chown below — otherwise exec() creates it
        # later as root and the unprivileged agent cannot write to its own workdir
        # (measured: `touch probe.txt` -> Permission denied on a root:root 755 dir).
        if self.spec.workdir:
            Path(self._rewrite(self.spec.workdir)).mkdir(parents=True, exist_ok=True)
        self._link_volumes()
        if self._run_as:
            # The agent runs unprivileged; its own session tree must stay writable.
            # -R covers the volume copies too, which is fine: they are mode 0555, so
            # ownership alone does not make them writable.
            await self._run_local_command(
                "chown", "-R", f"{self._run_as}:", str(self.session_dir), capture=True
            )

    def _link_volumes(self) -> None:
        """Realise ``kwargs.volumes`` (``src:dst[:opts]``).

        The profile contract is unchanged; only the mechanism differs. Two mechanisms,
        picked by where the destination lands:

        * **inside the session → real copy.** A symlink is wrong here, and subtly so:
          ``tools/ascendc_eval_pipeline.sh`` runs ``readlink -f "$BASH_SOURCE"`` *on
          purpose* (to survive being invoked through a symlink) and derives
          ``WORK_ROOT="$_SCRIPT_DIR/.."``. Under Docker ``<workdir>/tools`` is a bind
          mount — a real directory — so ``WORK_ROOT`` comes out as ``<workdir>``. If we
          symlink it to the repo tree instead, ``readlink -f`` resolves *through* the
          link and ``WORK_ROOT`` becomes the shared tree: ``judge_out/`` then gets
          written into ``operator_runtime_t2a/`` (measured), and the judge looks for the
          agent's project in the wrong place. Copying restores Docker's semantics
          exactly. Cheap: the only session-scoped volume is ``tools`` at 260K / 8 files.
        * **outside the session → shared symlink.** ``/opt/canonical`` (21M) and
          ``/opt/asc-devkit`` (317M) are read-only reference trees, far too big to copy
          per session, and nothing resolves paths *through* them the way the eval
          pipeline does with its own directory.
        """
        for volume in self.spec.kwargs.get("volumes", []) or []:
            parts = str(volume).split(":")
            if len(parts) < 2:
                logger.warning("skipping malformed volume %r", volume)
                continue
            src, dst = parts[0], self._rewrite(parts[1])
            source, dest = Path(src), Path(dst)
            if not source.exists():
                # Docker would silently create an empty dir here; that is exactly how
                # asc-devkit went missing and 46 skill references dangled. Be loud.
                logger.error("volume source does not exist: %s (dest %s)", src, dst)
                continue
            if self._is_session_scoped(dest):
                self._copy_volume(source, dest)
                continue
            if dest.is_symlink():
                if dest.resolve() == source.resolve():
                    continue  # already linked (shared global mount, another session did it)
                logger.warning(
                    "volume dest %s already links elsewhere (%s); relinking to %s",
                    dest, os.readlink(dest), source,
                )
                dest.unlink()
            elif dest.exists():
                logger.warning("volume dest %s exists and is not a symlink; leaving it", dest)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(source, target_is_directory=source.is_dir())
            self._symlinks.append(dest)

    def _is_session_scoped(self, dest: Path) -> bool:
        try:
            dest.relative_to(self.session_dir)
        except ValueError:
            return False
        return True

    def _copy_volume(self, source: Path, dest: Path) -> None:
        """Copy a session-scoped volume, then drop write bits to mimic ``:ro``.

        The profile marks these ``:ro`` and prepare re-copies what it needs into the
        workdir, so nothing should write here. Read-only also keeps a rogue agent from
        editing the eval entry point it is about to be judged by.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            if dest.is_symlink() or dest.is_file():
                dest.unlink()
            else:
                shutil.rmtree(dest)
        if source.is_dir():
            shutil.copytree(source, dest, symlinks=True)
        else:
            shutil.copy2(source, dest)
        for path in ([dest, *dest.rglob("*")] if dest.is_dir() else [dest]):
            try:
                path.chmod(path.stat().st_mode & ~0o222)
            except OSError:
                pass
        self._copies.append(dest)

    async def stop(self) -> None:
        """Kill every process group we started, then drop per-session symlinks.

        This is the one place LocalRuntime is weaker than Docker, where discarding the
        container killed everything for free. ``base.cancel()`` only kills the direct
        child, but the agent forks a CLI, compilers and eval helpers — a missed process
        keeps its NPU flock (``npu_lease_exec.py``) and the pool spins down to empty
        after a few sessions.
        """
        if self._destroyed:
            return
        self._destroyed = True
        for pgid in list(self._pgids):
            self._signal_group(pgid, signal.SIGTERM)
        if self._pgids:
            import asyncio

            await asyncio.sleep(self._STOP_GRACE)
        for pgid in list(self._pgids):
            self._signal_group(pgid, signal.SIGKILL)
        self._pgids.clear()
        # Restore write bits on the read-only volume copies: the gateway's
        # _remove_session_dir_best_effort does shutil.rmtree(session_dir), which fails on
        # a read-only subtree and would leak a session dir per rollout.
        for copied in self._copies:
            try:
                for path in ([copied, *copied.rglob("*")] if copied.is_dir() else [copied]):
                    path.chmod(path.stat().st_mode | 0o200)
            except OSError:
                logger.warning("failed to restore write bits on %s", copied)
        self._copies.clear()
        # Only per-session links; a shared global link may still be in use elsewhere.
        for link in self._symlinks:
            try:
                if link.is_symlink() and str(link).startswith(str(self.session_dir)):
                    link.unlink()
            except OSError:
                logger.warning("failed to unlink %s", link)
        self._symlinks.clear()

    @staticmethod
    def _signal_group(pgid: int, sig: int) -> None:
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    async def cancel(self) -> None:
        # Override base: it only kills _active_process, which leaves grandchildren.
        await self.stop()

    # ── exec ────────────────────────────────────────────────────────────────────

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ) -> ExecResult:
        effective_env = self._session_env({**self.spec.env, **(env or {})})
        effective_workdir = self._rewrite(
            cwd or self.spec.workdir or self.runtime_session_dir
        )
        # A cwd created here belongs to the gateway (root); hand it to run_as or the
        # unprivileged command cannot write in its own working directory. Only touch
        # directories we just made, and only inside the session.
        cwd_path = Path(effective_workdir)
        fresh = not cwd_path.exists()
        cwd_path.mkdir(parents=True, exist_ok=True)
        if fresh and self._run_as and self._is_session_scoped(cwd_path):
            await self._run_local_command(
                "chown", f"{self._run_as}:", str(cwd_path), capture=True
            )
        wrapped = self._prepend_pinned_exports(self._rewrite(command), effective_env)
        if self._run_as:
            # Drop privileges here, not in the agent preset: this single point covers
            # prepare, agent and judge. Wrapping in the preset would miss the other two.
            wrapped = (
                f"runuser -u {shlex.quote(self._run_as)} -- "
                f"bash -lc {shlex.quote(wrapped)}"
            )
        rc, stdout, stderr = await self._run_in_process_group(
            "bash", "-lc", wrapped,
            cwd=effective_workdir, env=effective_env, timeout=timeout_sec,
        )
        return ExecResult(stdout=stdout, stderr=stderr, return_code=rc)

    # 这三个必须在登录 profile 跑完之后再设一遍，见 _prepend_pinned_exports。
    _PINNED = ("HOME", "TMPDIR", "ASCEND_PROCESS_LOG_PATH")

    def _prepend_pinned_exports(self, command: str, env: dict[str, str]) -> str:
        """Re-export the session-scoped paths *after* the login profile has run.

        ``bash -lc`` is a login shell and resets ``HOME`` from the passwd database —
        measured: passing ``HOME=/tmp/x`` to ``bash -lc`` yields ``/root``, while
        ``bash -c`` keeps it (``-p`` does not help either). So everything
        ``_session_env`` pins via ``env=`` is silently discarded by the time the
        command runs: CANN keeps logging to ``/root/ascend/log`` and, worse, every
        concurrent session shares one ``HOME`` and one ``TMPDIR`` (the eval pipeline's
        ``/tmp/npu``) — the exact collisions the pinning existed to prevent.

        Dropping ``-l`` is not an option: DockerRuntime uses ``bash -lc`` too, and the
        CANN/conda environment only lands on PATH via the login profile. So instead of
        fighting the login shell, re-assert the values after it.

        Only a concurrent test catches this: with one instance ``HOME=/root`` raises
        nothing at all.
        """
        exports = " ".join(
            f"export {key}={shlex.quote(str(env[key]))};"
            for key in self._PINNED if key in env
        )
        return f"{exports} {command}" if exports else command

    def _session_env(self, env: dict[str, str]) -> dict[str, str]:
        """Rewrite prefixes, then pin the three paths that would otherwise be shared."""
        out = self._rewrite_env(env)
        # HOME: useradd -M creates no home, and CANN then logs
        # "can not create directory: /home/<user>/ascend/log" on every exec (measured).
        out.setdefault("HOME", str(self.session_dir))
        out.setdefault("ASCEND_PROCESS_LOG_PATH", str(self.session_dir / "logs" / "ascend"))
        # TMPDIR: the eval pipeline writes /tmp/npu, which would collide across the
        # concurrent sessions that now share one container.
        out.setdefault("TMPDIR", str(self.session_dir / "tmp"))
        return out

    async def _run_in_process_group(
        self, *args: str, cwd: str, env: dict[str, str], timeout: float | None,
    ) -> tuple[int, str | None, str | None]:
        """Like ``base._run_local_command`` but in a NEW process group.

        ``base`` does not set ``start_new_session``, so a kill only reaches the direct
        child. Everything the agent forks would survive — see ``stop``.
        """
        import asyncio

        process = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            env={**os.environ, **env},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self._active_process = process
        pgid: int | None = None
        try:
            pgid = os.getpgid(process.pid)
            self._pgids.add(pgid)
        except (ProcessLookupError, OSError):
            pass
        try:
            if timeout is None:
                stdout_bytes, stderr_bytes = await process.communicate()
            else:
                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        process.communicate(), timeout=timeout
                    )
                except asyncio.TimeoutError:
                    # Whole group, not just the child: a timed-out agent leaves a tree.
                    if pgid is not None:
                        self._signal_group(pgid, signal.SIGKILL)
                    else:
                        process.kill()
                    try:
                        await process.wait()
                    except ProcessLookupError:
                        pass
                    return -1, None, None
        finally:
            self._active_process = None
            if pgid is not None:
                self._pgids.discard(pgid)
        rc = process.returncode or 0
        stdout = stdout_bytes.decode(errors="replace") if stdout_bytes else None
        stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else None
        return rc, stdout, stderr

    # ── transfers: no container boundary to cross ───────────────────────────────
    # base.py's bind-mount helpers already cover /polar/session; outside it these are
    # plain host copies. This is the cheapest part of the backend.

    async def upload_file(self, local_path: str, remote_path: str) -> None:
        target = Path(self._rewrite(remote_path))
        if self._copy_to_bind_mount(local_path, remote_path):
            return
        source = Path(local_path)
        if not source.exists():
            raise FileNotFoundError(f"source path does not exist: {local_path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    async def upload_dir(self, local_path: str, remote_path: str) -> None:
        target = Path(self._rewrite(remote_path))
        if self._copy_to_bind_mount(local_path, remote_path):
            return
        source = Path(local_path)
        if not source.exists():
            raise FileNotFoundError(f"source path does not exist: {local_path}")
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)

    async def download_file(self, remote_path: str, local_path: str) -> None:
        if self._copy_from_bind_mount(remote_path, Path(local_path)):
            return
        source = Path(self._rewrite(remote_path))
        if not source.is_file():
            raise FileNotFoundError(f"runtime path does not exist: {remote_path}")
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, Path(local_path))

    async def download_dir(self, remote_path: str, local_path: str) -> None:
        if self._copy_from_bind_mount(remote_path, Path(local_path)):
            return
        source = Path(self._rewrite(remote_path))
        if not source.is_dir():
            raise FileNotFoundError(f"runtime path does not exist: {remote_path}")
        target = Path(local_path)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
