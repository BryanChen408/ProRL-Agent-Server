import asyncio,sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
import httpx,pytest
from polar.gateway import server
from polar.gateway.session import SessionRegistry
from polar.gateway.storage import SessionStore
from polar.rollout.pipeline import Pipeline
from polar.rollout.models import SessionContext

def test_gateway_202_until_cleanup_done_and_affinity_released(monkeypatch):
    async def run():
        gate=asyncio.Event();cleanup=asyncio.create_task(gate.wait())
        managed=NS(discard_cleanup_task=cleanup)
        node=NS(cancel=AsyncMock(return_value=True),_dispatcher=NS(get_session=AsyncMock(return_value=managed)),release_session_affinity_best_effort=AsyncMock(return_value=False))
        registry=SessionRegistry();registry.register('session-a',task_id='task-a')
        storage=SessionStore();storage.ensure_session('session-a',None,None,None,task_id='task-a')
        state=NS(node_manager=node,session_registry=registry,storage=storage,discarded_sessions=set(),inflight=NS(status=lambda:{'entries':[]},close_session=AsyncMock()))
        monkeypatch.setattr(server,'get_state',lambda:state)
        result=await server.delete_session('session-a',reason='group_unacceptable');assert result.status_code==202
        gate.set();await cleanup
        result=await server.delete_session('session-a',reason='group_unacceptable');assert result.status_code==202
        assert registry.get('session-a') is not None
        node.release_session_affinity_best_effort.return_value=True
        result=await server.delete_session('session-a',reason='group_unacceptable');assert result.deleted
        assert registry.get('session-a') is None
        # Lost 200 response: replay returns the same completion acknowledgement.
        result=await server.delete_session('session-a',reason='group_unacceptable');assert result.deleted
        node._dispatcher.get_session.return_value=None;state.discarded_sessions.clear()
        # After gateway restart/lost provenance, do not fake a release acknowledgement.
        with pytest.raises(server.HTTPException) as exc:await server.delete_session('session-a',reason='group_unacceptable')
        assert exc.value.status_code==404
    asyncio.run(run())

@pytest.mark.parametrize('status',[202,404,503])
def test_pipeline_does_not_accept_unconfirmed_cleanup(status):
    async def run():
        p=Pipeline.__new__(Pipeline)
        p._new_client=lambda:httpx.AsyncClient(transport=httpx.MockTransport(lambda req:httpx.Response(status,json={})))
        session=NS(session_id='s',gateway_url='http://isolated',cancel_acknowledged=False,cancel_error=None,
                   group_cleanup_acknowledged=False,group_cleanup_error=None)
        with pytest.raises(Exception):await p._cleanup_session(session,reason='group_unacceptable',strict=True)
        assert not session.group_cleanup_acknowledged and session.group_cleanup_error
        assert not session.cancel_acknowledged and session.cancel_error is None
    asyncio.run(run())


@pytest.mark.parametrize('first_reason', [None, 'policy_cutoff'])
def test_coordinator_waits_for_cancelled_task_to_finish_unwinding(first_reason):
    from polar.rollout.manager import RolloutManager
    from polar.rollout.balancer import NodeScheduler
    from polar.rollout.models import TaskRequest
    from polar.runtime.models import RuntimeSpec
    from polar.agent.models import AgentSpec
    async def run():
        started=asyncio.Event();finish=asyncio.Event()
        class FakePipeline:
            async def run_batch(self,sessions,**kwargs):
                started.set()
                try:await asyncio.Event().wait()
                finally:await finish.wait()
            def result_path_for(self,*a,**kw):return None
            async def _cleanup_session(self,*a,**kw):return None
            async def _record_cancel_before_enqueue(self,*a,**kw):return None
        manager=RolloutManager(pipeline=FakePipeline(),scheduler=NodeScheduler())
        await manager.submit_task(TaskRequest(task_id='task-pending',instruction='test',num_samples=1,runtime=RuntimeSpec(image='none'),agent=AgentSpec(harness='codex')))
        await started.wait()
        if first_reason is not None:
            await manager.cancel_tasks(['task-pending'], reason=first_reason)
        first=await manager.cancel_tasks(['task-pending'],reason='group_unacceptable')
        assert not first['cleanup_confirmed']
        assert not first['cancel_accepted']
        finish.set();await asyncio.sleep(.01)
        second=await manager.cancel_tasks(['task-pending'],reason='group_unacceptable')
        assert second['cleanup_confirmed']
        assert second['cancel_accepted']
    asyncio.run(run())


def test_cancel_before_enqueue_creates_cleanup_receipt(tmp_path):
    from test_discard_judge import setup
    async def run():
        node, managed = setup(tmp_path)
        node.session_registry = SessionRegistry()
        node.storage = SessionStore()
        node._session_base_dir = tmp_path
        node._completed_runtime_cleanups = set()
        await node._dispatcher.start()
        node._dispatcher._sessions.pop(managed.session_id, None)
        await node._dispatcher.cancel(managed.session_id, reason='group_unacceptable')
        with pytest.raises(ValueError, match='cancelled before enqueue'):
            await node.dispatch(managed.request)
        await node._dispatcher.stop()
        assert managed.session_id in node._completed_runtime_cleanups
    asyncio.run(run())


def test_group_postrun_creates_cleanup_receipt(tmp_path):
    from test_discard_judge import setup
    async def run():
        node, managed = setup(tmp_path)
        node.session_registry = SessionRegistry()
        node.storage = SessionStore()
        node.inflight = NS(close_session=AsyncMock())
        node.node_id = 'test-node'
        node._session_affinity_release_url = None
        node._completed_runtime_cleanups = set()
        managed.cancel_requested = True
        managed.cancel_reason = 'group_unacceptable'
        managed.discard_cleanup_task = asyncio.create_task(asyncio.sleep(0))
        await node._handle_postrun(managed)
        assert managed.session_id in node._completed_runtime_cleanups
    asyncio.run(run())
