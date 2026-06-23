from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from polar.runtime.docker import DockerRuntime
from polar.runtime.models import RuntimeSpec


class _FakeLock:
    def __init__(self, device_id: str) -> None:
        self.device_id = device_id
        self.released = False

    def release(self) -> None:
        self.released = True


def _values(args: list[str], flag: str) -> list[str]:
    return [args[i + 1] for i, item in enumerate(args) if item == flag and i + 1 < len(args)]


def test_docker_start_builds_mainline_ascend_create_args(monkeypatch, tmp_path: Path) -> None:
    asyncio.run(_test_docker_start_builds_mainline_ascend_create_args(monkeypatch, tmp_path))


async def _test_docker_start_builds_mainline_ascend_create_args(monkeypatch, tmp_path: Path) -> None:
    created_commands: list[list[str]] = []
    lock = _FakeLock("11")

    def fake_acquire_card(pool, lock_dir):
        assert pool == ["8", "9", "11"]
        assert lock_dir == "/locks"
        return lock

    async def fake_run_local_command(self, *args, **kwargs):
        created_commands.append(list(args))
        if args[:2] == ("docker", "exec") and args[-1:] == ("id", "-u"):
            return 0, "0\n", ""
        return 0, "", ""

    monkeypatch.setattr("polar.runtime.docker.acquire_card", fake_acquire_card)
    monkeypatch.setattr(DockerRuntime, "_run_local_command", fake_run_local_command)

    spec = RuntimeSpec(
        backend="docker",
        image="sandbox:v1",
        network="host",
        workdir="/opt/workspace/agent_workdir",
        kwargs={
            "volumes": ["/skills:/opt/canonical:ro"],
            "ascend": {
                "pool": "8,9,11",
                "lock_dir": "/locks",
                "env": {"ASCEND_RT_VISIBLE_DEVICES": "0", "CUSTOM_FLAG": "1"},
            },
        },
    )
    runtime = DockerRuntime(spec, "session/eval", tmp_path)

    await runtime.start()

    create = created_commands[0]
    assert create[:4] == ["docker", "create", "--name", "polar-session-eval"]
    assert create[-3:] == ["sandbox:v1", "sleep", "infinity"]
    assert "--network" in create
    assert "host" in _values(create, "--network")
    volumes = _values(create, "-v")
    assert f"{tmp_path}:/polar/session" in volumes
    assert "/skills:/opt/canonical:ro" in volumes
    assert "/dev:/dev" in volumes
    assert "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro" in volumes
    assert "--privileged" in create
    env = dict(item.split("=", 1) for item in _values(create, "-e"))
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "11"
    assert env["CUSTOM_FLAG"] == "1"

    await runtime.stop()

    assert lock.released is True
    assert ["docker", "kill", "polar-session-eval"] in created_commands


def test_docker_start_releases_lock_when_start_fails(monkeypatch, tmp_path: Path) -> None:
    asyncio.run(_test_docker_start_releases_lock_when_start_fails(monkeypatch, tmp_path))


async def _test_docker_start_releases_lock_when_start_fails(monkeypatch, tmp_path: Path) -> None:
    lock = _FakeLock("9")

    def fake_acquire_card(pool, lock_dir):
        return lock

    async def fake_run_local_command(self, *args, **kwargs):
        if args[:2] == ("docker", "start"):
            return 1, "", "start failed"
        return 0, "", ""

    monkeypatch.setattr("polar.runtime.docker.acquire_card", fake_acquire_card)
    monkeypatch.setattr(DockerRuntime, "_run_local_command", fake_run_local_command)

    spec = RuntimeSpec(
        backend="docker",
        image="sandbox:v1",
        kwargs={"ascend": {"pool": "9", "lock_dir": "/locks"}},
    )
    runtime = DockerRuntime(spec, "session", tmp_path)

    with pytest.raises(RuntimeError, match="docker start failed"):
        await runtime.start()

    assert lock.released is True


def test_docker_start_can_mount_ascend_without_start_lease(monkeypatch, tmp_path: Path) -> None:
    asyncio.run(_test_docker_start_can_mount_ascend_without_start_lease(monkeypatch, tmp_path))


async def _test_docker_start_can_mount_ascend_without_start_lease(
    monkeypatch, tmp_path: Path
) -> None:
    created_commands: list[list[str]] = []

    def fake_acquire_card(pool, lock_dir):
        pytest.fail("lease_at_start=false must not acquire a card")

    async def fake_run_local_command(self, *args, **kwargs):
        created_commands.append(list(args))
        if args[:2] == ("docker", "exec") and args[-1:] == ("id", "-u"):
            return 0, "0\n", ""
        return 0, "", ""

    monkeypatch.setattr("polar.runtime.docker.acquire_card", fake_acquire_card)
    monkeypatch.setattr(DockerRuntime, "_run_local_command", fake_run_local_command)

    spec = RuntimeSpec(
        backend="docker",
        image="sandbox:v1",
        network="host",
        workdir="/opt/workspace/agent_workdir",
        kwargs={
            "volumes": ["/skills:/opt/canonical:ro"],
            "ascend": {
                "pool": "8,9",
                "lock_dir": "/locks",
                "lease_at_start": False,
                "env": {"ASCEND_RT_VISIBLE_DEVICES": "8", "CUSTOM_FLAG": "1"},
                "mounts": ["/locks:/locks"],
            },
        },
    )
    runtime = DockerRuntime(spec, "session", tmp_path)

    await runtime.start()

    create = created_commands[0]
    volumes = _values(create, "-v")
    assert f"{tmp_path}:/polar/session" in volumes
    assert "/skills:/opt/canonical:ro" in volumes
    assert "/dev:/dev" in volumes
    assert "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro" in volumes
    assert "/locks:/locks" in volumes
    assert "--privileged" in create
    env = dict(item.split("=", 1) for item in _values(create, "-e"))
    assert env == {"CUSTOM_FLAG": "1"}
    assert "ASCEND_RT_VISIBLE_DEVICES" not in env

    await runtime.stop()

    assert ["docker", "kill", "polar-session"] in created_commands


def test_docker_start_releases_lock_when_create_fails(monkeypatch, tmp_path: Path) -> None:
    asyncio.run(_test_docker_start_releases_lock_when_create_fails(monkeypatch, tmp_path))


async def _test_docker_start_releases_lock_when_create_fails(
    monkeypatch, tmp_path: Path
) -> None:
    lock = _FakeLock("9")

    def fake_acquire_card(pool, lock_dir):
        return lock

    async def fake_run_local_command(self, *args, **kwargs):
        if args[:3] == ("docker", "create", "--name"):
            return 1, "", "create failed"
        return 0, "", ""

    monkeypatch.setattr("polar.runtime.docker.acquire_card", fake_acquire_card)
    monkeypatch.setattr(DockerRuntime, "_run_local_command", fake_run_local_command)

    spec = RuntimeSpec(
        backend="docker",
        image="sandbox:v1",
        kwargs={"ascend": {"pool": "9", "lock_dir": "/locks"}},
    )
    runtime = DockerRuntime(spec, "session", tmp_path)

    with pytest.raises(RuntimeError, match="docker create failed"):
        await runtime.start()

    assert lock.released is True
