import asyncio
import json
import time
from types import SimpleNamespace

import httpx
import pytest

from polar.agent.models import AgentSpec
from polar.rollout.models import SessionContext, SessionResult, TaskRequest
from polar.rollout.pipeline import Pipeline
from polar.trajectory.models import Trajectory


def session(sid):
    request = TaskRequest(task_id=sid, instruction='work', timeout_seconds=30,
                          agent=AgentSpec(harness='codex'))
    return SessionContext(session_id=sid, task_id=sid, request=request,
                          deadline_monotonic=time.monotonic() + 30)


def result(sid):
    return SessionResult(session_id=sid, task_id=sid, status='COMPLETED',
                         trajectory=Trajectory(status='COMPLETED', traces=[]))


def scheduler(url):
    return SimpleNamespace(acquire_node=lambda: SimpleNamespace(node_id='node', gateway_url=url))


def test_real_http_poll_reuse_and_cancel_isolation():
    async def run():
        events, clients, handlers = [], [], set()
        changed = asyncio.Condition()
        release_a = asyncio.Event()
        finish_b = False
        evaluating_b = False

        async def serve(reader, writer):
            task = asyncio.current_task()
            handlers.add(task)
            peer = writer.get_extra_info('peername')
            try:
                while True:
                    head = await reader.readuntil(b'\r\n\r\n')
                    lines = head.decode().split('\r\n')
                    method, target, _ = lines[0].split()
                    headers = dict(l.lower().split(': ', 1) for l in lines[1:] if ': ' in l)
                    raw = await reader.readexactly(int(headers.get('content-length', '0')))
                    body = json.loads(raw) if raw else {}
                    sid = body['session_id'] if method == 'POST' else target.split('?')[0].rsplit('/', 1)[1]
                    async with changed:
                        events.append((method, sid, peer, target))
                        changed.notify_all()
                    if method == 'POST':
                        payload = {'session_id': sid}
                    elif method == 'DELETE':
                        if sid == 'a':
                            release_a.set()
                        payload = {'deleted': True}
                    elif sid == 'a':
                        await release_a.wait()
                        payload = {'status': 'RUNNING'}
                    elif finish_b:
                        payload = {'status': 'COMPLETED', 'result': result(sid).model_dump(mode='json')}
                    else:
                        payload = {'status': 'EVALUATING' if evaluating_b else 'RUNNING'}
                    data = json.dumps(payload).encode()
                    writer.write(f'HTTP/1.1 200 OK\r\nContent-Length: {len(data)}\r\n\r\n'.encode() + data)
                    await writer.drain()
            except (asyncio.IncompleteReadError, ConnectionError):
                pass
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except ConnectionError:
                    pass
                handlers.discard(task)

        async def wait_until(predicate):
            async with changed:
                await asyncio.wait_for(changed.wait_for(predicate), 5)

        server = await asyncio.start_server(serve, '127.0.0.1', 0)
        url = f'http://127.0.0.1:{server.sockets[0].getsockname()[1]}'
        pipeline = Pipeline(callback_url='http://unused', save_dir=None,
                            scheduler=scheduler(url), dispatch_poll_interval_seconds=.01)
        original = pipeline._new_client
        def factory():
            client = original()
            clients.append(client)
            return client
        pipeline._new_client = factory
        a, b = session('a'), session('b')
        try:
            await pipeline.start()
            jobs = [asyncio.create_task(pipeline._dispatch_and_collect(s, None, lambda *a: None)) for s in [a, b]]
            await wait_until(lambda: any(m == 'GET' and sid == 'a' for m, sid, _, _ in events)
                             and sum(m == 'GET' and sid == 'b' for m, sid, _, _ in events) >= 3)
            assert len(clients) == 2 and not any(m == 'DELETE' for m, *_ in events)
            b_client = pipeline._clients['b']
            a.cancel_requested, a.cancel_reason = True, 'policy_cutoff'
            jobs[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(jobs[0], 5)
            assert a.cancel_acknowledged and a.cancel_error is None
            assert 'a' not in pipeline._clients and not jobs[1].done()
            assert pipeline._clients['b'] is b_client and not b_client.is_closed
            assert not any(m == 'DELETE' and sid == 'b' for m, sid, _, _ in events)
            evaluating_b = True
            previous = len(events)
            await wait_until(lambda: len(events) > previous)
            assert not any(m == 'DELETE' and sid == 'b' for m, sid, _, _ in events)
            finish_b = True
            assert (await asyncio.wait_for(jobs[1], 5)).status == 'COMPLETED'
            b_requests = [peer for method, sid, peer, _ in events if sid == 'b' and method != 'DELETE']
            assert len(set(b_requests)) == 1  # POST and repeated GETs reuse the same socket.
            assert any(m == 'DELETE' and sid == 'b' for m, sid, _, _ in events)
            assert len(clients) == 4 and all(c.is_closed for c in clients)
            assert not pipeline._clients and not pipeline._pending
        finally:
            release_a.set()
            await pipeline.close()
            server.close()
            await server.wait_closed()
            if handlers:
                await asyncio.wait_for(asyncio.gather(*tuple(handlers)), 5)

    asyncio.run(run())


def test_many_session_completions_and_cancellations_leave_no_shared_pool():
    async def run():
        clients, deleted = [], []
        polls = 0
        changed = asyncio.Condition()

        async def handler(request):
            nonlocal polls
            sid = request.url.path.rsplit('/', 1)[-1]
            if request.method == 'POST':
                return httpx.Response(200, json={})
            if request.method == 'DELETE':
                deleted.append(sid)
                return httpx.Response(200, json={})
            if sid.startswith('cancel'):
                async with changed:
                    polls += 1
                    changed.notify_all()
                await asyncio.Event().wait()
            return httpx.Response(200, json={'status': 'COMPLETED', 'result': result(sid).model_dump(mode='json')})

        pipeline = Pipeline(callback_url='http://unused', save_dir=None,
                            scheduler=scheduler('http://gateway'), dispatch_poll_interval_seconds=.001)
        def factory():
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            clients.append(client)
            return client
        pipeline._new_client = factory
        await pipeline.start()
        for wave in range(6):
            complete = [session(f'ok-{wave}-{i}') for i in range(10)]
            assert all(r.status == 'COMPLETED' for r in await pipeline.run_batch(
                complete, on_state=lambda *a: None))
            cancelled = [session(f'cancel-{wave}-{i}') for i in range(10)]
            target = polls + 10
            jobs = [asyncio.create_task(pipeline._dispatch_and_collect(s, None, lambda *a: None)) for s in cancelled]
            async with changed:
                await asyncio.wait_for(changed.wait_for(lambda: polls >= target), 5)
            for s, job in zip(cancelled, jobs):
                s.cancel_requested, s.cancel_reason = True, 'policy_cutoff'
                job.cancel()
            outcomes = await asyncio.gather(*jobs, return_exceptions=True)
            assert all(isinstance(x, asyncio.CancelledError) for x in outcomes)
            assert all(s.cancel_acknowledged for s in cancelled)
            assert not pipeline._clients and not pipeline._pending
            assert all(c.is_closed for c in clients)
        assert len(deleted) == 120 and len(set(deleted)) == 120
        assert len(clients) == 240  # One poll client + one terminal DELETE client per session.
        await pipeline.close()
        await pipeline.start()
        assert (await pipeline.run_batch([session('after-restart')], on_state=lambda *a: None))[0].status == 'COMPLETED'
        await pipeline.close()
        assert all(c.is_closed for c in clients)

    asyncio.run(run())
