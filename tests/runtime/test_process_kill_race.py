import asyncio
import sys
from unittest.mock import AsyncMock, Mock

import pytest

from polar.runtime.docker import DockerRuntime
from polar.runtime.models import RuntimeSpec


@pytest.mark.parametrize("returncode", [None, 0])
def test_timeout_and_cancel_tolerate_exited_process(monkeypatch, tmp_path, returncode):
    async def check():
        runtime = DockerRuntime(RuntimeSpec(image="unused"), "race", tmp_path)
        process = Mock(returncode=returncode)
        process.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
        process.kill = Mock(side_effect=ProcessLookupError)
        process.wait = AsyncMock(return_value=0)
        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", AsyncMock(return_value=process)
        )
        runtime.stop = AsyncMock()

        assert await runtime._run_local_command("unused", timeout=1) == (-1, None, None)
        assert runtime._active_process is None
        runtime.stop.assert_not_awaited()
        for _ in range(2):
            runtime._active_process = process
            await runtime.cancel()
        assert runtime.stop.await_count == 2
        assert process.wait.await_count == 3
        assert process.kill.call_count == (3 if returncode is None else 0)

    asyncio.run(check())


def test_real_process_completion_and_timeout(tmp_path):
    async def check():
        runtime = DockerRuntime(RuntimeSpec(image="unused"), "real", tmp_path)
        assert await runtime._run_local_command(
            sys.executable, "-c", "print('ok')", capture=True
        ) == (0, "ok\n", None)
        assert await runtime._run_local_command(
            sys.executable, "-c", "import time; time.sleep(60)", timeout=0.05,
            capture=True,
        ) == (-1, None, None)
        assert runtime._active_process is None

    asyncio.run(check())


def test_kill_does_not_hide_other_errors(tmp_path):
    runtime = DockerRuntime(RuntimeSpec(image="unused"), "error", tmp_path)
    process = Mock(returncode=None)
    process.kill.side_effect = PermissionError("denied")
    with pytest.raises(PermissionError, match="denied"):
        asyncio.run(runtime._kill_and_wait(process))
