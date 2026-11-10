import asyncio,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock,Mock
import pytest
from polar.gateway.node import GatewayNodeManager
from polar.gateway.dispatcher import ManagedSession,SessionDispatcher
from polar.rollout.models import SessionDispatchRequest
from polar.rollout.timer import StageTimer
from polar.runtime.models import RuntimeSpec,ExecInput
from polar.agent.models import AgentSpec,AgentRunResult
from polar.trajectory.models import EvaluatorSpec,Trajectory

def setup(tmp_path):
    req=SessionDispatchRequest(session_id='discard-test',task_id='task',instruction='test',remaining_timeout_seconds=60,runtime=RuntimeSpec(image='never-started'),agent=AgentSpec(harness='codex'),evaluator=EvaluatorSpec(strategy='operator_judge'))
    m=ManagedSession(request=req,timer=StageTimer(),session_dir=tmp_path,artifacts_dir=tmp_path)
    node=GatewayNodeManager.__new__(GatewayNodeManager)
    node._dispatcher=SessionDispatcher(max_init_workers=1,max_run_workers=1,max_postrun_workers=1)
    node._dispatcher._sessions[m.session_id]=m
    node.inflight=SimpleNamespace(close_session=AsyncMock());node._policy_cleanup_tasks=set()
    return node,m

def fake_runtime():
    return SimpleNamespace(runtime_id='fake',cancel=AsyncMock(),exec=AsyncMock())

@pytest.mark.parametrize('initial_reason',[None,'pipeline_budget_exceeded'])
def test_discard_stops_agent_and_judge_once_and_overrides_budget(tmp_path,initial_reason):
    async def run():
        node,m=setup(tmp_path);m.runtime=fake_runtime();m.eval_runtime=fake_runtime()
        m.cancel_requested=bool(initial_reason);m.cancel_reason=initial_reason
        await asyncio.gather(*(node.cancel(m.session_id,reason='group_unacceptable') for _ in range(8)))
        await node.cancel(m.session_id,reason='policy_cutoff')
        await m.discard_cleanup_task
        assert m.cancel_reason=='group_unacceptable' and m.discard_event.is_set()
        m.runtime.cancel.assert_awaited_once();m.eval_runtime.cancel.assert_awaited_once()
    asyncio.run(run())

def test_discard_interrupts_inflight_judge_and_never_retries(tmp_path):
    async def run():
        node,m=setup(tmp_path);m.runtime=fake_runtime();m.eval_runtime=fake_runtime();node._start_execution_deadline(m)
        started=asyncio.Event();closed=asyncio.Event()
        async def judge():
            started.set()
            try:await asyncio.Event().wait()
            finally:closed.set()
        t=asyncio.create_task(node._await_with_budget(judge(),m));await started.wait()
        await node.cancel(m.session_id,reason='group_unacceptable')
        with pytest.raises(RuntimeError,match='discarded'):await asyncio.wait_for(t,1)
        await m.discard_cleanup_task
        assert closed.is_set() and not m.discard_work
    asyncio.run(run())

def test_discard_during_runtime_creation_stops_creation_before_teardown(tmp_path):
    async def run():
        node,m=setup(tmp_path);node._start_execution_deadline(m);m.runtime=fake_runtime();creating=asyncio.Event();events=[]
        async def start():
            creating.set()
            try:await asyncio.Event().wait()
            finally:events.append('creation_stopped')
        async def cancel():events.append('runtime_cancel')
        m.runtime.cancel=cancel
        t=asyncio.create_task(node._await_with_budget(start(),m));await creating.wait()
        await node.cancel(m.session_id,reason='group_unacceptable')
        await asyncio.gather(t,return_exceptions=True);await m.discard_cleanup_task
        assert events==['creation_stopped','runtime_cancel']
    asyncio.run(run())

@pytest.mark.parametrize('method',['_prepare_eval_runtime','_run_postrun_steps'])
def test_discard_does_not_prepare_judge_or_run_postrun(tmp_path,method):
    async def run():
        node,m=setup(tmp_path);m.cancel_requested=True;m.cancel_reason='group_unacceptable';m.runtime=fake_runtime()
        m.postrun_steps=[ExecInput(command='must-not-run')]
        await getattr(node,method)(m);m.runtime.exec.assert_not_awaited()
    asyncio.run(run())

def test_discard_skips_eval_even_if_successful_candidate_exists(tmp_path):
    async def run():
        node,m=setup(tmp_path);m.cancel_requested=True;m.cancel_reason='group_unacceptable';node.evaluators=Mock()
        t=Trajectory(task_id='task',session_id=m.session_id,status='COMPLETED',traces=[])
        result=await node._run_eval(m.request,t,agent_result=AgentRunResult(status='completed',return_code=0),managed=m)
        assert result.status=='ERROR';node.evaluators.create.assert_not_called()
    asyncio.run(run())

def test_cleanup_failure_retries_and_retains_pending_state(tmp_path):
    async def run():
        node,m=setup(tmp_path);m.runtime=fake_runtime();m.runtime.cancel=AsyncMock(side_effect=[RuntimeError('busy'),None])
        await node.cancel(m.session_id,reason='group_unacceptable');await asyncio.sleep(.01)
        assert not m.discard_cleanup_task.done()
        await asyncio.wait_for(m.discard_cleanup_task,2)
        assert m.runtime.cancel.await_count==2
    asyncio.run(run())


@pytest.mark.parametrize('reason,expected',[('group_unacceptable',0),('pipeline_budget_exceeded',1)])
def test_eval_prepare_keeps_budget_judge_but_skips_discard(tmp_path,reason,expected):
    async def run():
        node,m=setup(tmp_path);m.cancel_requested=True;m.cancel_reason=reason;node._start_execution_deadline(m)
        rt=fake_runtime();rt.runtime_session_dir=str(tmp_path);rt.exec.return_value=SimpleNamespace(return_code=0,stdout='',stderr='')
        node._runtime_env=Mock(return_value={});node._write_exec_log=Mock()
        action=SimpleNamespace(type='exec',env={},cwd=None,command='isolated-compile')
        await node._run_runtime_prepare(rt,m.request.runtime,m.request,m,actions=[action],honor_cancel=False)
        assert rt.exec.await_count==expected
    asyncio.run(run())
