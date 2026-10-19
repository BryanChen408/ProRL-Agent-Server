"""Real local HTTP checks: request ownership and terminal-only affinity release."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from polar.gateway.engine import VLLMEngine
from polar.gateway.http_client import closing_client
from polar.gateway.inflight import InflightGenerationTracker
from polar.gateway.node import GatewayNodeManager
from polar.gateway.proxy import InferenceClient, UpstreamHTTPError, UpstreamTimeoutError
from polar.gateway.session import SessionRegistry
from polar.gateway.storage import SessionStore
from polar.rollout.models import SessionResult, SessionStatus
from polar.rollout.timer import StageTimer
from polar.trajectory.models import Trajectory


def test_real_http_concurrency_timeouts_cancellation_and_affinity():
    async def run():
        requests = []
        handlers = set()
        changed = asyncio.Condition()
        gate = asyncio.Event()
        health_blocked = False
        clients = []

        async def serve(reader, writer):
            task = asyncio.current_task()
            handlers.add(task)
            try:
                head = (await reader.readuntil(b'\r\n\r\n')).decode()
                lines = head.split('\r\n')
                path = lines[0].split()[1]
                headers = dict(line.lower().split(': ', 1) for line in lines[1:] if ': ' in line)
                raw = await reader.readexactly(int(headers.get('content-length', '0')))
                body = json.loads(raw) if raw else {}
                async with changed:
                    requests.append((path, headers, body))
                    changed.notify_all()
                if body.get('hold') or path == '/health' and health_blocked:
                    await reader.read()  # Client cancellation must close this socket.
                    return
                if body.get('gate'):
                    await gate.wait()
                payload = {'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}]}
                if path == '/health':
                    payload = {'status': 'ok'}
                elif path == '/v1/models':
                    payload = {'data': [{'id': 'served'}]}
                status = 503 if body.get('fail') else 200
                data = json.dumps(payload).encode()
                writer.write(f'HTTP/1.1 {status} Test\r\nContent-Length: {len(data)}\r\nConnection: close\r\n\r\n'.encode() + data)
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

        async def reached(count):
            async with changed:
                await asyncio.wait_for(changed.wait_for(lambda: len(requests) >= count), 5)

        server = await asyncio.start_server(serve, '127.0.0.1', 0)
        url = f'http://127.0.0.1:{server.sockets[0].getsockname()[1]}'
        proxy = InferenceClient(url, VLLMEngine(), liveness_timeout_seconds=2)
        factory = proxy._new_client

        def tracked():
            client = factory()
            clients.append(client)
            return client

        proxy._new_client = tracked
        try:
            # A timed-out health probe must not cancel either live generation.
            jobs = [asyncio.create_task(proxy.completion(
                {'model': 'served', 'messages': [], 'gate': True},
                trace_headers={'x-polar-trace-id': f'session-{i}:1'},
            )) for i in range(2)]
            await reached(2)
            health_blocked = True
            proxy._HEALTH_TIMEOUT_SECONDS = .03
            with pytest.raises(UpstreamTimeoutError):
                await proxy.health()
            assert all(not job.done() for job in jobs)
            assert proxy.generation_status()['inflight'] == 2
            gate.set()
            assert all(r['choices'][0]['message']['content'] == 'ok' for r in await asyncio.gather(*jobs))
            health_blocked = False
            assert await proxy.health() == {'status': 'ok'}
            assert (await proxy.list_models())['data'][0]['id'] == 'served'

            # Cancelling one generation must leave another session's request alive.
            gate.clear()
            target = len(requests) + 2
            survivor = asyncio.create_task(proxy.completion({'messages': [], 'gate': True}))
            victim = asyncio.create_task(proxy.completion({'messages': [], 'hold': True}))
            await reached(target)
            victim.cancel()
            with pytest.raises(asyncio.CancelledError):
                await victim
            assert not survivor.done() and proxy.generation_status()['inflight'] == 1
            gate.set()
            assert (await survivor)['choices'][0]['message']['content'] == 'ok'

            # More than the former shared pool's 100 slots; no cancelled request
            # may consume a slot belonging to a later request.
            for _ in range(12):
                target = len(requests) + 10
                jobs = [asyncio.create_task(proxy.completion({'messages': [], 'hold': True})) for _ in range(10)]
                await reached(target)
                for job in jobs:
                    job.cancel()
                results = await asyncio.gather(*jobs, return_exceptions=True)
                assert all(isinstance(r, asyncio.CancelledError) for r in results)
                assert proxy.generation_status()['inflight'] == 0
                assert not proxy._clients
            result = await proxy.completion({'messages': []}, trace_headers={'x-polar-trace-id': 'session-0:2'})
            assert result['choices'][0]['message']['content'] == 'ok'
            headers = [h for path, h, _ in requests if h.get('x-session-id') == 'session-0']
            assert len(headers) == 2  # New HTTP clients preserve the same routing identity.
            assert {h['x-polar-trace-id'] for h in headers} == {'session-0:1', 'session-0:2'}
            assert all(body.get('stream') is False for path, _, body in requests if path.endswith('completions'))

            with pytest.raises(UpstreamHTTPError):
                await proxy.completion({'messages': [], 'fail': True})
            proxy._liveness_timeout_seconds = .03
            with pytest.raises(UpstreamTimeoutError):
                await proxy.completion({'messages': [], 'hold': True})
            assert not proxy._clients
            assert proxy.generation_status()['inflight'] == 0
            assert all(c.is_closed and not c._transport._pool.connections for c in clients)
            assert len({id(c) for c in clients}) == len(clients)
        finally:
            await proxy.close()
            server.close()
            await server.wait_closed()
            if handlers:
                await asyncio.wait_for(asyncio.gather(*tuple(handlers)), 5)

    asyncio.run(run())


def test_repeated_cancel_waits_for_owned_client_cleanup():
    async def run():
        started, closing, allow_close = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class Client:
            closed = False
            async def aclose(self):
                closing.set()
                await allow_close.wait()
                self.closed = True

        client = Client()
        async def work():
            async with closing_client(client):
                started.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(work())
        await started.wait()
        task.cancel()
        await closing.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not client.closed
        allow_close.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client.closed

    asyncio.run(run())


def test_affinity_released_only_after_evaluation_and_teardown(monkeypatch, tmp_path):
    async def run():
        evaluated, allow_evaluation = asyncio.Event(), asyncio.Event()
        stopping, allow_stop = asyncio.Event(), asyncio.Event()
        calls, clients = [], []
        manager = GatewayNodeManager.__new__(GatewayNodeManager)
        manager.node_id = 'test'
        manager.storage = SessionStore()
        manager.storage.ensure_session('s', None, None, None)
        manager.session_registry = SessionRegistry()
        manager.session_registry.register('s', status=SessionStatus.RUNNING)
        manager.inflight = InflightGenerationTracker()
        manager._session_affinity_release_url = 'http://router/release'
        request = SimpleNamespace(session_id='s', callback_url='http://rollout/callback')
        managed = SimpleNamespace(request=request, final_result=None, cancel_requested=False,
                                  runtime=object(), eval_runtime=None, timer=StageTimer(), session_dir=tmp_path)
        result = SessionResult(session_id='s', task_id='t', status=SessionStatus.COMPLETED,
                               trajectory=Trajectory(status='COMPLETED', traces=[]))

        async def build(_):
            evaluated.set()
            await allow_evaluation.wait()
            return result
        async def stop(*_):
            stopping.set()
            await allow_stop.wait()
        async def noop(*_):
            return None
        manager._start_postrun_deadline = lambda _: None
        manager._build_session_result = build
        manager._run_postrun_steps = noop
        manager._drain_eval_prewarm_task = noop
        manager._stop_runtime_best_effort = stop
        manager._remove_session_dir_best_effort = noop
        original = httpx.AsyncClient

        def handle(req):
            assert allow_stop.is_set()
            assert manager.storage.is_session_closed('s')
            assert manager.session_registry.get('s').status == SessionStatus.COMPLETED
            calls.append((req.url.path, json.loads(req.content)))
            return httpx.Response(200, json={'status': 'ok'})
        def factory(**kwargs):
            client = original(transport=httpx.MockTransport(handle), **kwargs)
            clients.append(client)
            return client
        monkeypatch.setattr('polar.gateway.node.httpx.AsyncClient', factory)

        # Even an accidental direct call cannot release a running session.
        await manager.release_session_affinity_best_effort('s')
        assert not clients
        task = asyncio.create_task(manager._handle_postrun(managed))
        await evaluated.wait()
        assert not calls and not manager.storage.is_session_closed('s')
        allow_evaluation.set()
        await stopping.wait()
        assert not calls and not manager.storage.is_session_closed('s')
        allow_stop.set()
        await task
        assert [path for path, _ in calls] == ['/release', '/callback']
        assert calls[0][1] == {'session_id': 's'}
        assert calls[1][1]['status'] == 'COMPLETED'
        assert manager.session_registry.get('s').result is None
        assert all(client.is_closed for client in clients)
        # Terminal cleanup can be called twice without reopening or altering sessions.
        await manager.release_session_affinity_best_effort('s')
        assert calls[-1] == ('/release', {'session_id': 's'})
        assert len(clients) == 3 and all(client.is_closed for client in clients)

    asyncio.run(run())


def test_callback_failure_cannot_poison_release_or_next_callback(monkeypatch):
    async def run():
        manager = GatewayNodeManager.__new__(GatewayNodeManager)
        manager.storage = SessionStore()
        manager.storage.mark_session_closed('s')
        manager._session_affinity_release_url = 'http://router/release'
        result = SessionResult(session_id='s', task_id='t', status=SessionStatus.COMPLETED,
                               trajectory=Trajectory(status='COMPLETED', traces=[]))
        clients, calls = [], []
        original = httpx.AsyncClient

        def handler(request):
            calls.append(request.url.path)
            return httpx.Response(503 if len(calls) == 1 else 200, json={'status': 'ok'})
        def factory(**kwargs):
            client = original(transport=httpx.MockTransport(handler), **kwargs)
            clients.append(client)
            return client
        monkeypatch.setattr('polar.gateway.node.httpx.AsyncClient', factory)
        assert await manager._push_result('http://rollout/callback', result) is False
        assert clients[0].is_closed
        await manager.release_session_affinity_best_effort('s')
        assert await manager._push_result('http://rollout/callback', result) is True
        assert calls == ['/callback', '/release', '/callback']
        assert len(clients) == 3 and all(c.is_closed for c in clients)

    asyncio.run(run())


def test_startup_pause_drain_epoch_change_and_resume():
    async def run():
        proxy = InferenceClient('http://engine', VLLMEngine(), initially_paused=True)
        tracker = InflightGenerationTracker()
        epoch = 0
        wire = []
        started, closing = asyncio.Event(), asyncio.Event()
        allow_close = asyncio.Event()
        allow_close.set()
        clients = []

        async def handler(request):
            if request.url.path == '/health':
                return httpx.Response(200, json={'status': 'ok'})
            body = json.loads(request.content)
            wire.append(body)
            if body.get('hold'):
                started.set()
                await asyncio.Event().wait()
            return httpx.Response(200, json={'choices': [{'message': {'content': 'ok'}}]})

        class Client(httpx.AsyncClient):
            async def aclose(self):
                if not allow_close.is_set():
                    closing.set()
                await allow_close.wait()
                await super().aclose()

        def factory():
            client = Client(base_url='http://engine', transport=httpx.MockTransport(handler))
            clients.append(client)
            return client
        proxy._new_client = factory

        # Bootstrap: no generation/client is admitted before the explicit resume.
        first = asyncio.create_task(proxy.completion({'messages': [], 'epoch': 0}))
        await asyncio.sleep(0)
        assert not first.done() and not clients and not wire
        assert await proxy.health() == {'status': 'ok'}
        await proxy.resume_generation()
        assert (await first)['choices'][0]['message']['content'] == 'ok'

        for transition in range(3):
            started.clear()
            closing.clear()
            old_epoch = epoch
            sid = f'old-{transition}'
            req = {'messages': [], 'hold': True, 'epoch': old_epoch}
            old = asyncio.create_task(tracker.run(
                sid, req, lambda: proxy.completion(req),
            ))
            await started.wait()
            state = await proxy.pause_generation(wait_for_drain=False)
            assert state['paused'] and not state['drained'] and state['inflight'] == 1

            # A late old-policy request must stay behind the pause fence.
            sent = len(wire)
            stale = asyncio.create_task(proxy.completion(
                {'messages': [], 'epoch': old_epoch},
                generation_guard=lambda: epoch == old_epoch,
            ))
            await asyncio.sleep(0)
            assert len(wire) == sent and not stale.done()

            # Simulate cutoff cancellation while HTTP socket cleanup is still pending.
            allow_close.clear()
            await tracker.close_session(sid, reason='policy_cutoff')
            await asyncio.wait_for(closing.wait(), 1)
            state = await proxy.pause_generation(timeout_seconds=.01)
            assert state['paused'] and state['timed_out'] and not state['drained']
            allow_close.set()
            state = await proxy.pause_generation(timeout_seconds=1)
            assert state['drained'] and state['inflight'] == 0
            from polar.gateway.proxy import UpstreamError
            with pytest.raises(UpstreamError, match='session closed'):
                await old
            assert not proxy._clients
            assert all(c.is_closed for c in clients)

            # Engine weight synchronization is represented by the epoch change.
            # Resume must reject the old waiter, then accept a fresh session.
            epoch += 1
            await proxy.resume_generation()
            with pytest.raises(UpstreamError, match='policy epoch fence'):
                await stale
            assert len(wire) == sent
            with pytest.raises(UpstreamError, match='session closed'):
                await tracker.run(sid, req, lambda: proxy.completion(req))
            result = await tracker.run(
                f'new-{transition}', {'epoch': epoch},
                lambda: proxy.completion({'messages': [], 'epoch': epoch}),
            )
            assert result.response['choices'][0]['message']['content'] == 'ok'
            assert wire[-1]['epoch'] == epoch
            assert proxy.generation_status()['inflight'] == 0
            assert not proxy._clients
        await proxy.close()

    asyncio.run(run())


def test_independent_clients_share_only_tls_configuration(monkeypatch):
    from polar.gateway import http_client

    original = httpx.AsyncClient
    seen = []
    def factory(**kwargs):
        seen.append(kwargs['verify'])
        return original(**kwargs)
    monkeypatch.setattr(http_client.httpx, 'AsyncClient', factory)

    async def run():
        a, b = http_client.new_client(), http_client.new_client()
        try:
            assert a is not b
            assert a._transport._pool is not b._transport._pool
            assert seen[0] is seen[1]
            assert seen[0].check_hostname
        finally:
            await a.aclose()
            await b.aclose()
    asyncio.run(run())
