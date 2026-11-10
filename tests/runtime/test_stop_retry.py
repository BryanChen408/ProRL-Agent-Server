import asyncio,sys
from pathlib import Path
from unittest.mock import AsyncMock,Mock
import pytest
from polar.runtime.docker import DockerRuntime

def runtime(command):
    r=object.__new__(DockerRuntime);r._destroyed=False;r._chmod_needed=False;r._container_name='isolated';r._npu_lock=Mock();r._run_local_command=command
    return r

def test_rm_failure_retains_lease_until_retry_succeeds():
    async def run():
        cmd=AsyncMock(side_effect=[(0,'',''),(1,'','daemon unavailable'),(0,'',''),(0,'','')]);r=runtime(cmd);lease=r._npu_lock
        with pytest.raises(RuntimeError):await r.stop()
        assert not r._destroyed;lease.release.assert_not_called()
        await r.stop();assert r._destroyed;lease.release.assert_called_once()
    asyncio.run(run())

def test_concurrent_stop_only_releases_once():
    async def run():
        cmd=AsyncMock(return_value=(0,'',''));r=runtime(cmd);lease=r._npu_lock
        await asyncio.gather(*(r.stop() for _ in range(64)))
        assert cmd.await_count==2;lease.release.assert_called_once()
    asyncio.run(run())

def test_already_removed_is_idempotent():
    async def run():
        cmd=AsyncMock(return_value=(1,'','Error response from daemon: No such container: isolated'));r=runtime(cmd);lease=r._npu_lock
        await r.stop();await r.stop();assert r._destroyed;assert cmd.await_count==2;lease.release.assert_called_once()
    asyncio.run(run())

def test_cancelled_cleanup_remains_retryable():
    async def run():
        started=asyncio.Event()
        async def blocked(*a,**kw):started.set();await asyncio.Event().wait()
        r=runtime(blocked);lease=r._npu_lock;t=asyncio.create_task(r.stop());await started.wait();t.cancel()
        with pytest.raises(asyncio.CancelledError):await t
        assert not r._destroyed;lease.release.assert_not_called()
        r._run_local_command=AsyncMock(return_value=(0,'',''));await r.stop();lease.release.assert_called_once()
    asyncio.run(run())
