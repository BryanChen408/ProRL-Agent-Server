import asyncio,sys,time
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
import pytest
from test_discard_judge import setup,fake_runtime
from polar.gateway import server
from polar.gateway.session import SessionRegistry
from polar.gateway.storage import SessionStore
from polar.gateway.control import GatewayControlStore
from polar.gateway.proxy import InferenceClient
from polar.gateway.engine import VLLMEngine

@pytest.mark.parametrize('first',['group_unacceptable','policy_cutoff'])
def test_pause_does_not_wait_for_group_teardown_and_keeps_ack(tmp_path,monkeypatch,first):
    async def run():
        node,m=setup(tmp_path);gate=asyncio.Event();entered=asyncio.Event()
        async def stop():entered.set();await gate.wait()
        m.runtime=fake_runtime();m.runtime.cancel=AsyncMock(side_effect=stop)
        registry=SessionRegistry();registry.register(m.session_id,task_id='task')
        storage=SessionStore();storage.ensure_session(m.session_id,None,None,None,task_id='task')
        node.release_session_affinity_best_effort=AsyncMock(return_value=True)
        proxy=InferenceClient('http://127.0.0.1:1',VLLMEngine())
        state=NS(node_manager=node,session_registry=registry,storage=storage,discarded_sessions=set(),inflight=NS(status=lambda:{'entries':[]},close_session=AsyncMock()),control=GatewayControlStore(None),inference=proxy)
        monkeypatch.setattr(server,'get_state',lambda:state)
        await node.cancel(m.session_id,reason=first);await entered.wait()
        await node.cancel(m.session_id,reason='group_unacceptable')
        assert m.discard_cleanup_task and not m.discard_cleanup_task.done()
        pending=await server.delete_session(m.session_id,reason='group_unacceptable');assert pending.status_code==202
        # Global policy pause should acknowledge without waiting for Docker/judge cleanup.
        paused=await asyncio.wait_for(server.pause_inference_generation(timeout_seconds=0,wait_for_drain=False,transition_id='test-transition'),1)
        assert paused['paused'] and not m.discard_cleanup_task.done()
        policy_ack=await server.delete_session(m.session_id,reason='policy_cutoff');assert policy_ack.deleted
        assert registry.get(m.session_id) is not None
        pending=await server.delete_session(m.session_id,reason='group_unacceptable');assert pending.status_code==202
        gate.set();await asyncio.wait_for(m.discard_cleanup_task,1)
        ack=await server.delete_session(m.session_id,reason='group_unacceptable');assert ack.deleted
        m.runtime.cancel.assert_awaited_once()
        await proxy.resume_generation();assert not proxy.generation_status()['paused'];await proxy.close()
    asyncio.run(run())

