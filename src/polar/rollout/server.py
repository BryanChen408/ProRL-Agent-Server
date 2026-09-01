"""FastAPI server for rollout orchestration."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
import time

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from polar.config import RolloutServiceConfig, TopologyConfig
from polar.platform.events import SSE_HEADERS, EventBus
from polar.rollout.balancer import NodeScheduler
from polar.rollout.manager import RolloutManager
from polar.rollout.models import (
    GatewayNodeInfo,
    NodeHeartbeatRequest,
    NodeRegistrationRequest,
    OperatorSampleRequest,
    PolicyBootstrapBeginRequest,
    PolicyEpochInitializeRequest,
    PolicyQuiesceRequest,
    PolicyTransitionAbortRequest,
    PolicyTransitionBeginRequest,
    PolicyTransitionCommitRequest,
    PolicyTransitionDrainRequest,
    PolicyTransitionFailRequest,
    SessionResult,
    TaskCancelRequest,
    TaskRequest,
    TaskStatus,
)
from polar.rollout.operator_profile import expand_operator_sample_request
from polar.rollout.pipeline import Pipeline
from polar.rollout.policy_transition import (
    PolicyTransitionError,
    PolicyTransitionKind,
    PolicyTransitionPhase,
    PolicyTransitionRecord,
    PolicyTransitionStore,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

_POLICY_CUTOFF_RESUME_FENCE_TIMEOUT_SECONDS = 5.0


@dataclass(slots=True)
class RolloutState:
    topology: TopologyConfig
    rollout: RolloutServiceConfig
    scheduler: NodeScheduler
    pipeline: Pipeline
    manager: RolloutManager
    event_bus: EventBus
    policy_transitions: PolicyTransitionStore
    policy_transition_lock: asyncio.Lock


_state: RolloutState | None = None
_configured_topology_path: str | None = None


def configure_server(topology_path: str = "topology.yaml") -> None:
    global _configured_topology_path, _state
    _configured_topology_path = topology_path
    _state = None


def _build_state(topology: TopologyConfig) -> RolloutState:
    rollout = topology.rollout
    scheduler = NodeScheduler(
        bootstrap_nodes=topology.bootstrap_nodes,
        allow_dynamic_nodes=False,
    )
    event_bus = EventBus()
    pipeline = Pipeline(
        callback_url=f"{rollout.public_url}/callbacks/session_result",
        save_dir=rollout.save_dir,
        scheduler=scheduler,
        dispatch_poll_interval_seconds=rollout.dispatch_poll_interval_seconds,
        callback_grace_seconds=rollout.callback_grace_seconds,
        event_bus=event_bus,
    )
    manager = RolloutManager(pipeline=pipeline, scheduler=scheduler, event_bus=event_bus)
    control_dir = os.environ.get("POLAR_CONTROL_DIR")
    transition_path = (
        Path(control_dir) / "policy-transition.json"
        if control_dir
        else (
            Path(rollout.save_dir) / "_control" / "policy-transition.json"
            if rollout.save_dir
            else None
        )
    )
    policy_transitions = PolicyTransitionStore(transition_path)
    snapshot = policy_transitions.snapshot()
    if snapshot.current is not None:
        closed = snapshot.current.phase not in {
            PolicyTransitionPhase.SERVING,
            PolicyTransitionPhase.ABORTED,
        }
        manager.restore_policy_admission(
            policy_namespace=snapshot.active_namespace,
            epoch=snapshot.active_epoch,
            closed=closed,
            transition_id=snapshot.current.transition_id,
        )
    return RolloutState(
        topology=topology,
        rollout=rollout,
        scheduler=scheduler,
        pipeline=pipeline,
        manager=manager,
        event_bus=event_bus,
        policy_transitions=policy_transitions,
        policy_transition_lock=asyncio.Lock(),
    )


def get_state() -> RolloutState:
    global _state
    if _state is None:
        topology_path = _configured_topology_path or os.environ.get(
            "POLAR_TOPOLOGY",
            "topology.yaml",
        )
        _state = _build_state(TopologyConfig.load(topology_path))
    return _state


@asynccontextmanager
async def _lifespan(_: FastAPI):
    state = get_state()
    await state.pipeline.start()
    try:
        yield
    finally:
        await state.pipeline.close()


app = FastAPI(title="Polar Rollout", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
async def health():
    state = get_state()
    return {"status": "ok", "nodes": len(state.scheduler.list_nodes())}


@app.post("/rollout/task/submit")
async def submit_task_async(request: TaskRequest):
    """Non-blocking task submission. Returns immediately with task_id.

    Poll ``GET /rollout/task/{task_id}`` until status becomes terminal.
    """
    state = get_state()
    try:
        task_id = await state.manager.submit_task(request)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"task_id": task_id, "status": "running"}


@app.post("/rollout/operator_samples/submit")
async def submit_operator_samples_async(request: OperatorSampleRequest):
    """Submit a thin operator sample request using a Polar-owned profile."""
    state = get_state()
    try:
        task_request = expand_operator_sample_request(request, state.rollout)
        task_id = await state.manager.submit_task(task_request)
    except ValueError as exc:
        message = str(exc)
        status_code = 409 if "already running" in message else 400
        raise HTTPException(status_code=status_code, detail=message) from exc
    return {"task_id": task_id, "status": "running"}


@app.get("/rollout/task/{task_id}", response_model=TaskStatus)
async def get_task(task_id: str):
    task = get_state().manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.post("/rollout/admin/tasks/cancel")
async def cancel_rollout_tasks(request: TaskCancelRequest):
    """Cancel trainer-owned tasks without waiting for sessions to finish naturally."""
    result = await get_state().manager.cancel_tasks(
        request.task_ids,
        reason=request.reason,
    )
    if not result["all_acknowledged"]:
        raise HTTPException(status_code=409, detail=result)
    logger.info(
        "Cancelled rollout tasks at policy boundary: requested=%s cancelled=%s "
        "terminal=%s sessions=%s fence_pending=%s",
        result["requested"],
        result["cancelled"],
        result["already_terminal"],
        result["sessions_cancel_requested"],
        result["fence_pending"],
    )
    return result


@app.get("/rollout/status")
async def rollout_status():
    return get_state().manager.status()


def _transition_response(state: RolloutState, record: PolicyTransitionRecord) -> dict[str, object]:
    return {
        **record.model_dump(mode="json"),
        "active_epoch": state.policy_transitions.snapshot().active_epoch,
        "active_namespace": state.policy_transitions.snapshot().active_namespace,
        "admission_closed": record.phase in {
            PolicyTransitionPhase.QUIESCED,
            PolicyTransitionPhase.ADMISSION_CLOSED,
            PolicyTransitionPhase.READY_FOR_TRAINING,
            PolicyTransitionPhase.COMMITTING,
            PolicyTransitionPhase.ABORTING,
            PolicyTransitionPhase.RECOVERY_REQUIRED,
        },
        "ready_for_training": record.phase == PolicyTransitionPhase.READY_FOR_TRAINING,
        "serving": record.phase == PolicyTransitionPhase.SERVING,
    }


def _gateway_state_matches(
    payload: dict[str, object],
    *,
    action: str,
    transition_id: str,
    policy_namespace: str | None,
    epoch: int | None,
) -> bool:
    if payload.get("transition_id") != transition_id:
        return False
    if action == "pause":
        return payload.get("paused") is True
    if action == "set_epoch":
        return (
            payload.get("policy_namespace") == policy_namespace
            and
            payload.get("policy_version") == epoch
            and payload.get("epoch_enforced") is True
        )
    if action == "resume":
        return (
            payload.get("paused") is False
            and payload.get("policy_namespace") == policy_namespace
            and payload.get("policy_version") == epoch
        )
    raise ValueError(f"unknown gateway control action: {action}")


async def _apply_gateway_control(
    *,
    action: str,
    transition_id: str,
    policy_namespace: str | None = None,
    epoch: int | None = None,
    allow_paused_transition_takeover: bool = False,
) -> list[dict[str, object]]:
    """Apply desired state and reconcile an ambiguous/lost acknowledgement."""
    state = get_state()
    timeout = httpx.Timeout(10.0, read=10.0)

    if action == "pause":
        path = "/admin/inference/pause"
        params: dict[str, object] = {
            "timeout_seconds": 0.0,
            "wait_for_drain": False,
            "transition_id": transition_id,
            "allow_paused_transition_takeover": allow_paused_transition_takeover,
        }
    elif action == "set_epoch":
        path = "/admin/policy_version"
        params = {
            "version": epoch,
            "policy_namespace": policy_namespace,
            "transition_id": transition_id,
            "enforce_epoch": True,
        }
    elif action == "resume":
        path = "/admin/inference/resume"
        params = {
            "transition_id": transition_id,
            "expected_policy_namespace": policy_namespace,
            "expected_policy_version": epoch,
        }
    else:
        raise ValueError(f"unknown gateway control action: {action}")

    async def apply_one(node) -> dict[str, object]:  # noqa: ANN001
        post_error: str | None = None
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(f"{node.public_url}{path}", params=params)
            except httpx.RequestError as exc:
                post_error = f"{type(exc).__name__}: {exc}"
            else:
                if 400 <= response.status_code < 500:
                    # A gateway 4xx is an authoritative protocol rejection.  A GET
                    # describing an older state must not turn that rejection into a
                    # successful coordinator acknowledgement.
                    return {
                        "node_id": node.id,
                        "status": "error",
                        "source": "post",
                        "post_error": f"HTTP {response.status_code}: {response.text}",
                    }
                if response.status_code >= 500:
                    post_error = f"HTTP {response.status_code}: {response.text}"
                else:
                    try:
                        payload = response.json()
                    except ValueError as exc:
                        post_error = f"{type(exc).__name__}: {exc}"
                    else:
                        if isinstance(payload, dict) and _gateway_state_matches(
                            payload,
                            action=action,
                            transition_id=transition_id,
                            policy_namespace=policy_namespace,
                            epoch=epoch,
                        ):
                            return {
                                "node_id": node.id,
                                "status": "ok",
                                "source": "post",
                                "response": payload,
                            }
                        return {
                            "node_id": node.id,
                            "status": "error",
                            "source": "post",
                            "response": payload,
                            "post_error": "gateway returned an unexpected control state",
                        }

            # The POST may have committed before its response was lost.  The GET is
            # the authoritative reconciliation step and makes retries idempotent.
            try:
                response = await client.get(f"{node.public_url}/admin/inference/status")
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict) and _gateway_state_matches(
                    payload,
                    action=action,
                    transition_id=transition_id,
                    policy_namespace=policy_namespace,
                    epoch=epoch,
                ):
                    return {
                        "node_id": node.id,
                        "status": "ok",
                        "source": "reconciled_status",
                        "response": payload,
                        "post_error": post_error,
                    }
                return {
                    "node_id": node.id,
                    "status": "pending",
                    "source": "status",
                    "response": payload,
                    "post_error": post_error,
                }
            except Exception as exc:
                return {
                    "node_id": node.id,
                    "status": "error",
                    "post_error": post_error,
                    "status_error": f"{type(exc).__name__}: {exc}",
                }

    return list(await asyncio.gather(*(apply_one(node) for node in state.topology.gateway.nodes)))


async def _observe_gateway_control() -> list[dict[str, object]]:
    state = get_state()

    async def observe_one(node) -> dict[str, object]:  # noqa: ANN001
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{node.public_url}/admin/inference/status")
                response.raise_for_status()
                payload = response.json()
            return {"node_id": node.id, "status": "ok", "response": payload}
        except Exception as exc:
            return {
                "node_id": node.id,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }

    return list(await asyncio.gather(*(observe_one(node) for node in state.topology.gateway.nodes)))


def _all_gateways(
    nodes: list[dict[str, object]],
    predicate,
) -> bool:
    return bool(nodes) and all(
        item.get("status") == "ok"
        and isinstance(item.get("response"), dict)
        and predicate(item["response"])
        for item in nodes
    )


def _validate_engine_versions(
    engine_versions: dict[str, str],
    *,
    previous: dict[str, str] | None = None,
    require_advanced: bool = False,
) -> dict[str, str]:
    normalized = {
        str(engine_id).strip(): str(version).strip()
        for engine_id, version in engine_versions.items()
    }
    if not normalized or any(not key or not value for key, value in normalized.items()):
        raise PolicyTransitionError("engine version evidence must contain non-empty ids and versions")
    if len(set(normalized.values())) != 1:
        raise PolicyTransitionError(f"rollout engines have mixed weight versions: {normalized}")
    if previous is not None:
        if set(normalized) != set(previous):
            raise PolicyTransitionError(
                "rollout engine set changed across the policy transition: "
                f"before={sorted(previous)} after={sorted(normalized)}"
            )
        if require_advanced:
            try:
                previous_version = int(next(iter(previous.values())))
                current_version = int(next(iter(normalized.values())))
            except (TypeError, ValueError) as exc:
                raise PolicyTransitionError(
                    "rollout engine weight versions must be integer counters when "
                    "advancement is required"
                ) from exc
            if current_version <= previous_version:
                raise PolicyTransitionError(
                    "rollout engine weight version did not advance monotonically: "
                    f"before={previous_version} after={current_version}"
                )
    return normalized


async def _drive_quiescing(
    state: RolloutState,
    record: PolicyTransitionRecord,
    *,
    reset_epoch: bool = False,
) -> PolicyTransitionRecord:
    if reset_epoch:
        task_ids = state.manager.reset_policy_admission(
            record.from_epoch,
            policy_namespace=record.policy_namespace,
            transition_id=record.transition_id,
        )
    else:
        task_ids = state.manager.close_policy_admission(
            from_epoch=record.from_epoch,
            policy_namespace=record.policy_namespace,
            transition_id=record.transition_id,
        )
    cancellation = await state.manager.cancel_tasks(task_ids, reason="policy_cutoff")
    record = state.policy_transitions.update(
        record.transition_id,
        phase=PolicyTransitionPhase.QUIESCING,
        cancellation=cancellation,
        clear_error=True,
    )
    nodes = await _apply_gateway_control(
        action="pause",
        transition_id=record.transition_id,
        allow_paused_transition_takeover=reset_epoch,
    )
    record = state.policy_transitions.update(record.transition_id, gateway_nodes={
        str(item["node_id"]): item for item in nodes
    })
    if not _all_gateways(
        nodes,
        lambda payload: _gateway_state_matches(
            payload,
            action="pause",
            transition_id=record.transition_id,
            policy_namespace=None,
            epoch=None,
        ),
    ):
        return state.policy_transitions.update(
            record.transition_id,
            last_error="not every gateway acknowledged closed admission",
        )
    # Do not wait for drain here. VIME must regain control so it can abort every
    # serving engine; only the explicit confirm-drained phase below may wait.
    return state.policy_transitions.update(
        record.transition_id,
        phase=PolicyTransitionPhase.ADMISSION_CLOSED,
        clear_error=True,
    )


async def _drive_confirm_drained(
    state: RolloutState,
    record: PolicyTransitionRecord,
    *,
    wait_timeout_seconds: float,
) -> PolicyTransitionRecord:
    deadline = time.monotonic() + max(0.0, wait_timeout_seconds)
    while True:
        observed = await _observe_gateway_control()
        all_paused = _all_gateways(
            observed,
            lambda payload: (
                payload.get("paused") is True
                and payload.get("transition_id") == record.transition_id
            ),
        )
        naturally_drained = all_paused and _all_gateways(
            observed,
            lambda payload: (
                payload.get("paused") is True
                and payload.get("drained") is True
                and int(payload.get("inflight", 0) or 0) == 0
                and payload.get("transition_id") == record.transition_id
            ),
        )
        record = state.policy_transitions.update(
            record.transition_id,
            gateway_nodes={str(item["node_id"]): item for item in observed},
        )
        if all_paused and (record.engine_abort_confirmed or naturally_drained):
            return state.policy_transitions.update(
                record.transition_id,
                phase=PolicyTransitionPhase.READY_FOR_TRAINING,
                clear_error=True,
            )
        if time.monotonic() >= deadline:
            return state.policy_transitions.update(
                record.transition_id,
                phase=PolicyTransitionPhase.ADMISSION_CLOSED,
                last_error=(
                    "gateway admission is not durably paused, or neither an all-engine "
                    "abort proof nor a natural gateway drain has been observed"
                ),
            )
        await asyncio.sleep(min(0.1, max(0.0, deadline - time.monotonic())))


async def _drive_commit(
    state: RolloutState,
    record: PolicyTransitionRecord,
) -> PolicyTransitionRecord:
    if record.verified_policy_epoch != record.to_epoch:
        return state.policy_transitions.update(
            record.transition_id,
            phase=PolicyTransitionPhase.RECOVERY_REQUIRED,
            last_error=(
                f"verified policy epoch {record.verified_policy_epoch} does not prove "
                f"target {record.to_epoch}"
            ),
        )
    try:
        _validate_engine_versions(
            record.engine_versions,
            previous=(record.from_engine_versions or None),
            require_advanced=record.kind == PolicyTransitionKind.UPDATE,
        )
    except PolicyTransitionError as exc:
        return state.policy_transitions.update(
            record.transition_id,
            phase=PolicyTransitionPhase.RECOVERY_REQUIRED,
            last_error=str(exc),
        )
    state.manager.close_policy_admission(
        from_epoch=record.from_epoch,
        policy_namespace=record.policy_namespace,
        transition_id=record.transition_id,
    )
    record = state.policy_transitions.update(
        record.transition_id,
        phase=PolicyTransitionPhase.COMMITTING,
        clear_error=True,
    )
    nodes = await _apply_gateway_control(
        action="set_epoch",
        transition_id=record.transition_id,
        policy_namespace=record.policy_namespace,
        epoch=record.to_epoch,
    )
    if not _all_gateways(
        nodes,
        lambda payload: _gateway_state_matches(
            payload,
            action="set_epoch",
            transition_id=record.transition_id,
            policy_namespace=record.policy_namespace,
            epoch=record.to_epoch,
        ),
    ):
        return state.policy_transitions.update(
            record.transition_id,
            gateway_nodes={str(item["node_id"]): item for item in nodes},
            last_error="target policy epoch is not acknowledged by every gateway",
        )

    nodes = await _apply_gateway_control(
        action="resume",
        transition_id=record.transition_id,
        policy_namespace=record.policy_namespace,
        epoch=record.to_epoch,
    )
    if not _all_gateways(
        nodes,
        lambda payload: _gateway_state_matches(
            payload,
            action="resume",
            transition_id=record.transition_id,
            policy_namespace=record.policy_namespace,
            epoch=record.to_epoch,
        ),
    ):
        # A partial multi-gateway resume must be compensated immediately.  Standard
        # rollout admission is still closed, and re-pausing prevents direct clients
        # from observing a half-open serving fleet while the coordinator retries.
        compensated = await _apply_gateway_control(
            action="pause",
            transition_id=record.transition_id,
        )
        return state.policy_transitions.update(
            record.transition_id,
            gateway_nodes={str(item["node_id"]): item for item in compensated},
            last_error="not every gateway acknowledged target-epoch resume",
        )

    # Durable SERVING precedes reopening the in-memory task gate.  A crash between
    # them is liveness-only: restart reconstructs the open gate from this record.
    record = state.policy_transitions.update(
        record.transition_id,
        phase=PolicyTransitionPhase.SERVING,
        gateway_nodes={str(item["node_id"]): item for item in nodes},
        clear_error=True,
    )
    state.manager.open_policy_admission(
        record.to_epoch,
        policy_namespace=record.policy_namespace,
        transition_id=record.transition_id,
    )
    return record


async def _drive_abort(
    state: RolloutState,
    record: PolicyTransitionRecord,
) -> PolicyTransitionRecord:
    if record.verified_policy_epoch != record.from_epoch:
        return state.policy_transitions.update(
            record.transition_id,
            phase=PolicyTransitionPhase.RECOVERY_REQUIRED,
            last_error=(
                f"cannot restore epoch {record.from_epoch}; verified policy epoch is "
                f"{record.verified_policy_epoch}"
            ),
        )
    if record.engine_versions != record.from_engine_versions:
        return state.policy_transitions.update(
            record.transition_id,
            phase=PolicyTransitionPhase.RECOVERY_REQUIRED,
            last_error="cannot restore old policy: rollout engine versions changed",
        )
    state.manager.close_policy_admission(
        from_epoch=record.from_epoch,
        policy_namespace=record.policy_namespace,
        transition_id=record.transition_id,
    )
    record = state.policy_transitions.update(
        record.transition_id,
        phase=PolicyTransitionPhase.ABORTING,
    )
    nodes = await _apply_gateway_control(
        action="set_epoch",
        transition_id=record.transition_id,
        policy_namespace=record.policy_namespace,
        epoch=record.from_epoch,
    )
    if _all_gateways(
        nodes,
        lambda payload: _gateway_state_matches(
            payload,
            action="set_epoch",
            transition_id=record.transition_id,
            policy_namespace=record.policy_namespace,
            epoch=record.from_epoch,
        ),
    ):
        nodes = await _apply_gateway_control(
            action="resume",
            transition_id=record.transition_id,
            policy_namespace=record.policy_namespace,
            epoch=record.from_epoch,
        )
    if not _all_gateways(
        nodes,
        lambda payload: _gateway_state_matches(
            payload,
            action="resume",
            transition_id=record.transition_id,
            policy_namespace=record.policy_namespace,
            epoch=record.from_epoch,
        ),
    ):
        # A subset may already have resumed. Re-close every gateway so retrying
        # abort recovery cannot expose a half-open serving fleet.
        compensated = await _apply_gateway_control(
            action="pause",
            transition_id=record.transition_id,
        )
        return state.policy_transitions.update(
            record.transition_id,
            gateway_nodes={str(item["node_id"]): item for item in compensated},
            last_error="old-epoch abort recovery is not acknowledged by every gateway",
        )
    record = state.policy_transitions.update(
        record.transition_id,
        phase=PolicyTransitionPhase.ABORTED,
        gateway_nodes={str(item["node_id"]): item for item in nodes},
        clear_error=True,
    )
    state.manager.open_policy_admission(
        record.from_epoch,
        policy_namespace=record.policy_namespace,
        transition_id=record.transition_id,
    )
    return record


async def _reconcile_transition(
    state: RolloutState,
    record: PolicyTransitionRecord,
    *,
    wait_timeout_seconds: float = 0.0,
) -> PolicyTransitionRecord:
    if record.phase == PolicyTransitionPhase.QUIESCING:
        return await _drive_quiescing(state, record)
    if record.phase == PolicyTransitionPhase.COMMITTING:
        return await _drive_commit(state, record)
    if record.phase == PolicyTransitionPhase.ABORTING:
        return await _drive_abort(state, record)
    if record.phase == PolicyTransitionPhase.RECOVERY_REQUIRED:
        state.manager.close_policy_admission(
            from_epoch=record.from_epoch,
            policy_namespace=record.policy_namespace,
            transition_id=record.transition_id,
        )
        nodes = await _apply_gateway_control(
            action="pause",
            transition_id=record.transition_id,
        )
        return state.policy_transitions.update(
            record.transition_id,
            gateway_nodes={str(item["node_id"]): item for item in nodes},
        )
    if record.phase == PolicyTransitionPhase.ADMISSION_CLOSED:
        # GET reconciliation must make progress after a confirm-drained response or
        # acknowledgement is delayed. A zero-wait observation is non-blocking and
        # prevents the old "cleaned just after timeout but status never advances" bug.
        return await _drive_confirm_drained(
            state,
            record,
            wait_timeout_seconds=wait_timeout_seconds,
        )
    if record.phase == PolicyTransitionPhase.READY_FOR_TRAINING:
        observed = await _observe_gateway_control()
        all_paused = _all_gateways(
            observed,
            lambda payload: (
                payload.get("paused") is True
                and payload.get("transition_id") == record.transition_id
            ),
        )
        naturally_drained = all_paused and _all_gateways(
            observed,
            lambda payload: (
                payload.get("drained") is True
                and int(payload.get("inflight", 0) or 0) == 0
            ),
        )
        if not all_paused or not (record.engine_abort_confirmed or naturally_drained):
            record = state.policy_transitions.update(
                record.transition_id,
                phase=PolicyTransitionPhase.ADMISSION_CLOSED,
                last_error="gateway drifted from the ready-for-training fence",
            )
            return await _drive_confirm_drained(
                state,
                record,
                wait_timeout_seconds=wait_timeout_seconds,
            )
    return record


@app.post("/rollout/admin/policy/initialize")
async def initialize_policy_epoch(request: PolicyEpochInitializeRequest):
    state = get_state()
    async with state.policy_transition_lock:
        try:
            snapshot = state.policy_transitions.snapshot()
            if (
                snapshot.active_epoch == request.epoch
                and snapshot.active_namespace == request.policy_namespace
                and snapshot.current is not None
                and snapshot.current.phase == PolicyTransitionPhase.SERVING
                and snapshot.current.transition_id == request.transition_id
            ):
                evidence = _validate_engine_versions(request.engine_versions)
                if snapshot.current.engine_versions != evidence:
                    raise PolicyTransitionError(
                        "policy initialization was retried with different engine evidence"
                    )
                state.manager.initialize_policy_admission(
                    request.epoch,
                    policy_namespace=request.policy_namespace,
                    transition_id=snapshot.current.transition_id,
                )
                return _transition_response(state, snapshot.current)
            record = state.policy_transitions.start(
                transition_id=request.transition_id,
                policy_namespace=request.policy_namespace,
                from_epoch=request.epoch,
                to_epoch=request.epoch,
                from_engine_versions=_validate_engine_versions(request.engine_versions),
                allow_epoch_reset=True,
                kind=PolicyTransitionKind.INITIALIZE,
            )
            if record.phase == PolicyTransitionPhase.SERVING:
                state.manager.initialize_policy_admission(
                    request.epoch,
                    policy_namespace=request.policy_namespace,
                    transition_id=request.transition_id,
                )
                return _transition_response(state, record)
            record = await _drive_quiescing(state, record, reset_epoch=True)
            if record.phase == PolicyTransitionPhase.ADMISSION_CLOSED:
                record = await _drive_confirm_drained(
                    state,
                    record,
                    wait_timeout_seconds=30.0,
                )
            if record.phase != PolicyTransitionPhase.READY_FOR_TRAINING:
                return _transition_response(state, record)
            record = state.policy_transitions.update(
                record.transition_id,
                phase=PolicyTransitionPhase.COMMITTING,
                verified_policy_epoch=request.epoch,
                engine_versions=_validate_engine_versions(request.engine_versions),
            )
            record = await _drive_commit(state, record)
            return _transition_response(state, record)
        except (PolicyTransitionError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/rollout/admin/policy/bootstrap/begin")
async def begin_policy_bootstrap(request: PolicyBootstrapBeginRequest):
    """Durably close Polar before VIME performs its first weight synchronization."""
    state = get_state()
    async with state.policy_transition_lock:
        try:
            record = state.policy_transitions.start(
                transition_id=request.transition_id,
                policy_namespace=request.policy_namespace,
                from_epoch=request.epoch,
                to_epoch=request.epoch,
                from_engine_versions={},
                allow_epoch_reset=True,
                kind=PolicyTransitionKind.BOOTSTRAP,
            )
            if record.kind != PolicyTransitionKind.BOOTSTRAP:
                raise PolicyTransitionError("bootstrap transition kind mismatch")
            record = await _reconcile_transition(state, record)
            return _transition_response(state, record)
        except (PolicyTransitionError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/rollout/admin/policy-transitions/begin")
async def begin_policy_transition(request: PolicyTransitionBeginRequest):
    if request.to_epoch <= request.from_epoch:
        raise HTTPException(status_code=422, detail="to_epoch must be greater than from_epoch")
    state = get_state()
    async with state.policy_transition_lock:
        try:
            record = state.policy_transitions.start(
                transition_id=request.transition_id,
                policy_namespace=request.policy_namespace,
                from_epoch=request.from_epoch,
                to_epoch=request.to_epoch,
                from_engine_versions=_validate_engine_versions(request.engine_versions),
                kind=PolicyTransitionKind.UPDATE,
            )
            record = await _reconcile_transition(
                state,
                record,
            )
            return _transition_response(state, record)
        except (PolicyTransitionError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/rollout/admin/policy-transitions/{transition_id}")
async def get_policy_transition(transition_id: str):
    state = get_state()
    async with state.policy_transition_lock:
        snapshot = state.policy_transitions.snapshot()
        record = snapshot.current
        if record is None or record.transition_id != transition_id:
            raise HTTPException(status_code=404, detail="policy transition not found")
        record = await _reconcile_transition(state, record)
        return _transition_response(state, record)


@app.post("/rollout/admin/policy-transitions/{transition_id}/confirm-drained")
async def confirm_policy_transition_drained(
    transition_id: str,
    request: PolicyTransitionDrainRequest,
):
    state = get_state()
    async with state.policy_transition_lock:
        try:
            snapshot = state.policy_transitions.snapshot()
            record = snapshot.current
            if record is None or record.transition_id != transition_id:
                raise PolicyTransitionError(f"unknown policy transition: {transition_id}")
            if record.phase == PolicyTransitionPhase.READY_FOR_TRAINING:
                return _transition_response(state, record)
            if record.phase == PolicyTransitionPhase.QUIESCING:
                record = await _drive_quiescing(state, record)
            if record.phase != PolicyTransitionPhase.ADMISSION_CLOSED:
                raise PolicyTransitionError(
                    f"transition is {record.phase}, not admission_closed"
                )
            if request.engine_abort_confirmed:
                # Persist the VIME all-engine abort acknowledgement before deriving
                # READY. A lost HTTP response can then be reconciled safely by GET.
                record = state.policy_transitions.update(
                    record.transition_id,
                    engine_abort_confirmed=True,
                    clear_error=True,
                )
            record = await _drive_confirm_drained(
                state,
                record,
                wait_timeout_seconds=request.wait_timeout_seconds,
            )
            return _transition_response(state, record)
        except (PolicyTransitionError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/rollout/admin/policy-transitions/{transition_id}/commit")
async def commit_policy_transition(
    transition_id: str,
    request: PolicyTransitionCommitRequest,
):
    state = get_state()
    async with state.policy_transition_lock:
        try:
            snapshot = state.policy_transitions.snapshot()
            record = snapshot.current
            if record is None or record.transition_id != transition_id:
                raise PolicyTransitionError(f"unknown policy transition: {transition_id}")
            if record.phase == PolicyTransitionPhase.SERVING:
                if (
                    record.policy_namespace != request.policy_namespace
                    or
                    record.verified_policy_epoch != request.verified_policy_epoch
                    or record.engine_versions != request.engine_versions
                ):
                    raise PolicyTransitionError(
                        "committed transition was retried with different engine evidence"
                    )
                return _transition_response(state, record)
            # Re-observe READY as well: actor_model.update_weights resumes the vLLM
            # engines internally, so the gateway pause is the last admission fence
            # and must still hold immediately before publishing the new identity.
            record = await _reconcile_transition(state, record)
            if record.phase != PolicyTransitionPhase.READY_FOR_TRAINING:
                raise PolicyTransitionError(
                    f"transition is {record.phase}, not ready_for_training"
                )
            if record.policy_namespace != request.policy_namespace:
                raise PolicyTransitionError(
                    "commit policy_namespace does not match the active transition"
                )
            record = state.policy_transitions.update(
                transition_id,
                phase=PolicyTransitionPhase.COMMITTING,
                verified_policy_epoch=request.verified_policy_epoch,
                engine_versions=_validate_engine_versions(
                    request.engine_versions,
                    previous=(record.from_engine_versions or None),
                    require_advanced=record.kind == PolicyTransitionKind.UPDATE,
                ),
            )
            record = await _drive_commit(state, record)
            return _transition_response(state, record)
        except (PolicyTransitionError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/rollout/admin/policy-transitions/{transition_id}/abort")
async def abort_policy_transition(
    transition_id: str,
    request: PolicyTransitionAbortRequest,
):
    del request.reason
    state = get_state()
    async with state.policy_transition_lock:
        try:
            snapshot = state.policy_transitions.snapshot()
            record = snapshot.current
            if record is None or record.transition_id != transition_id:
                raise PolicyTransitionError(f"unknown policy transition: {transition_id}")
            if record.phase in {PolicyTransitionPhase.ABORTED, PolicyTransitionPhase.SERVING}:
                return _transition_response(state, record)
            if record.kind == PolicyTransitionKind.BOOTSTRAP:
                raise PolicyTransitionError(
                    "bootstrap cannot restore an unverified pre-sync engine state"
                )
            if record.policy_namespace != request.policy_namespace:
                raise PolicyTransitionError(
                    "abort policy_namespace does not match the active transition"
                )
            record = state.policy_transitions.update(
                transition_id,
                phase=PolicyTransitionPhase.ABORTING,
                verified_policy_epoch=request.verified_policy_epoch,
                engine_versions=_validate_engine_versions(
                    request.engine_versions,
                    previous=record.from_engine_versions,
                ),
            )
            record = await _drive_abort(state, record)
            return _transition_response(state, record)
        except (PolicyTransitionError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/rollout/admin/policy-transitions/{transition_id}/fail")
async def fail_policy_transition(
    transition_id: str,
    request: PolicyTransitionFailRequest,
):
    state = get_state()
    async with state.policy_transition_lock:
        try:
            snapshot = state.policy_transitions.snapshot()
            record = snapshot.current
            if record is None or record.transition_id != transition_id:
                raise PolicyTransitionError(f"unknown policy transition: {transition_id}")
            if record.phase == PolicyTransitionPhase.SERVING:
                return _transition_response(state, record)
            record = state.policy_transitions.update(
                transition_id,
                phase=PolicyTransitionPhase.RECOVERY_REQUIRED,
                last_error=request.reason,
            )
            record = await _reconcile_transition(state, record)
            return _transition_response(state, record)
        except (PolicyTransitionError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/rollout/admin/policy/quiesce")
async def quiesce_policy(request: PolicyQuiesceRequest):
    """Close the current serving policy when its VIME trainer is disposing."""
    state = get_state()
    async with state.policy_transition_lock:
        try:
            snapshot = state.policy_transitions.snapshot()
            record = snapshot.current
            if record is None:
                raise PolicyTransitionError("no initialized policy can be quiesced")
            if (
                snapshot.active_namespace != request.policy_namespace
                or snapshot.active_epoch != request.epoch
            ):
                raise PolicyTransitionError(
                    "quiesce identity mismatch: "
                    f"stored={snapshot.active_namespace}/{snapshot.active_epoch} "
                    f"requested={request.policy_namespace}/{request.epoch}"
                )
            if record.phase == PolicyTransitionPhase.QUIESCED:
                return _transition_response(state, record)
            if record.phase != PolicyTransitionPhase.SERVING:
                raise PolicyTransitionError(
                    f"transition is {record.phase}, not serving"
                )
            task_ids = state.manager.close_policy_admission(
                from_epoch=request.epoch,
                policy_namespace=request.policy_namespace,
                transition_id=record.transition_id,
            )
            cancellation = await state.manager.cancel_tasks(
                task_ids,
                reason="policy_cutoff",
            )
            nodes = await _apply_gateway_control(
                action="pause",
                transition_id=record.transition_id,
            )
            if not _all_gateways(
                nodes,
                lambda payload: _gateway_state_matches(
                    payload,
                    action="pause",
                    transition_id=record.transition_id,
                    policy_namespace=None,
                    epoch=None,
                ),
            ):
                return _transition_response(
                    state,
                    state.policy_transitions.update(
                        record.transition_id,
                        gateway_nodes={str(item["node_id"]): item for item in nodes},
                        cancellation=cancellation,
                        last_error="not every gateway acknowledged trainer quiesce",
                    ),
                )
            record = state.policy_transitions.update(
                record.transition_id,
                phase=PolicyTransitionPhase.QUIESCED,
                gateway_nodes={str(item["node_id"]): item for item in nodes},
                cancellation=cancellation,
                clear_error=True,
            )
            return _transition_response(state, record)
        except (PolicyTransitionError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/rollout/admin/inference/pause")
async def pause_gateway_generation(
    timeout_seconds: float = 300.0,
    wait_for_drain: bool = True,
):
    result = await _forward_gateway_admin(
        "/admin/inference/pause",
        params={
            "timeout_seconds": timeout_seconds,
            "wait_for_drain": wait_for_drain,
        },
    )
    nodes = result["nodes"]
    successful = [
        item["response"]
        for item in nodes
        if item["status"] == "ok" and isinstance(item.get("response"), dict)
    ]
    all_paused = len(successful) == len(nodes) and all(
        response.get("paused") is True for response in successful
    )
    all_drained = all_paused and all(
        response.get("drained") is True for response in successful
    )
    inflight = sum(
        int(response.get("inflight", 0) or 0)
        for response in successful
    )
    return {
        "all_paused": all_paused,
        "all_drained": all_drained,
        "inflight": inflight,
        "nodes": nodes,
    }


@app.post("/rollout/admin/inference/resume")
async def resume_gateway_generation():
    fence = await get_state().manager.wait_for_policy_cutoff_fences(
        timeout_seconds=_POLICY_CUTOFF_RESUME_FENCE_TIMEOUT_SECONDS,
    )
    if not fence["all_fenced"]:
        logger.error("Refusing inference resume before policy cutoff fence: %s", fence)
        raise HTTPException(
            status_code=409,
            detail={
                "message": "old-policy sessions are not fully fenced",
                **fence,
            },
        )
    logger.info("Policy-cutoff session fence verified before inference resume: %s", fence)
    result = await _forward_gateway_admin("/admin/inference/resume")
    nodes = result["nodes"]
    successful = [
        item["response"]
        for item in nodes
        if item["status"] == "ok" and isinstance(item.get("response"), dict)
    ]
    all_resumed = len(successful) == len(nodes) and all(
        response.get("paused") is False for response in successful
    )
    return {"all_resumed": all_resumed, "nodes": nodes}


@app.post("/rollout/admin/policy_version")
async def set_gateway_policy_version(version: int):
    # version-span: forward the trainer's new weight version to the gateway(s) so per-turn
    # policy_version stamping + span rejection can fire. Mirrors pause/resume forwarding.
    result = await _forward_gateway_admin(
        "/admin/policy_version",
        params={"version": version},
    )
    nodes = result["nodes"]
    successful = [
        item["response"]
        for item in nodes
        if item["status"] == "ok" and isinstance(item.get("response"), dict)
    ]
    all_updated = len(successful) == len(nodes) and all(
        response.get("policy_version") == version for response in successful
    )
    return {
        "all_updated": all_updated,
        "policy_version": version,
        "nodes": nodes,
    }


@app.post("/nodes/register", response_model=GatewayNodeInfo)
async def register_node(request: NodeRegistrationRequest):
    try:
        return get_state().scheduler.register_node(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/nodes/{node_id}/heartbeat", response_model=GatewayNodeInfo)
async def node_heartbeat(node_id: str, request: NodeHeartbeatRequest):
    try:
        return get_state().scheduler.heartbeat(node_id, metrics=request.metrics)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/nodes", response_model=list[GatewayNodeInfo])
async def list_nodes():
    return get_state().scheduler.list_nodes()


@app.get("/nodes/{node_id}", response_model=GatewayNodeInfo)
async def get_node(node_id: str):
    node = get_state().scheduler.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return node


@app.delete("/nodes/{node_id}", response_model=GatewayNodeInfo)
async def drain_node(node_id: str):
    try:
        return get_state().scheduler.drain_node(node_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/callbacks/session_result")
async def session_result_callback(result: SessionResult):
    await get_state().pipeline.accept_callback_result(result)
    return {"status": "accepted"}


@app.get("/tasks")
async def list_tasks(
    status: str | None = Query(default=None),
    harness: str | None = Query(default=None),
    since: float | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
):
    """List tasks tracked in-memory by RolloutManager."""
    state = get_state()
    tasks = state.manager.list_tasks()
    if status:
        tasks = [t for t in tasks if t["status"] == status]
    if harness:
        tasks = [t for t in tasks if t.get("harness") == harness]
    if since is not None:
        tasks = [t for t in tasks if (t.get("updated_at") or 0) >= since]
    tasks.sort(key=lambda t: (t.get("updated_at") or 0), reverse=True)
    return {"tasks": tasks[:limit]}


@app.get("/tasks/{task_id}/sessions")
async def list_task_sessions(task_id: str):
    """Per-session summaries for a task currently in memory."""
    state = get_state()
    sessions = state.manager.list_sessions_for(task_id)
    if sessions is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task_id, "sessions": sessions}


@app.get("/events")
async def stream_events(request: Request):
    state = get_state()

    async def iterator():
        async for chunk in state.event_bus.stream_events(heartbeat_seconds=15.0):
            if await request.is_disconnected():
                break
            yield chunk

    return StreamingResponse(
        iterator(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


async def _forward_gateway_admin(
    path: str,
    *,
    params: dict[str, object] | None = None,
) -> dict[str, object]:
    state = get_state()
    nodes = list(state.topology.gateway.nodes)
    if not nodes:
        raise HTTPException(status_code=503, detail="No gateway nodes configured")

    wait_for_drain = bool(params.get("wait_for_drain", True)) if params else False
    drain_timeout = float(params.get("timeout_seconds", 0)) if params and wait_for_drain else 0.0
    timeout = httpx.Timeout(10.0, read=max(drain_timeout + 5.0, 10.0))
    async with httpx.AsyncClient(timeout=timeout) as client:
        responses = []
        for node in nodes:
            try:
                response = await client.post(f"{node.public_url}{path}", params=params)
                response.raise_for_status()
                responses.append({"node_id": node.id, "status": "ok", "response": response.json()})
            except Exception as exc:
                responses.append({
                    "node_id": node.id,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                })

    if all(item["status"] == "error" for item in responses):
        raise HTTPException(status_code=502, detail={"nodes": responses})
    return {"nodes": responses}


def serve(topology_path: str = "topology.yaml", *, log_level: str = "info") -> None:
    import uvicorn

    configure_server(topology_path)
    state = get_state()
    uvicorn.run(
        app,
        host=state.rollout.host,
        port=state.rollout.port,
        log_level=log_level,
    )


def main() -> None:
    serve(os.environ.get("POLAR_TOPOLOGY", "topology.yaml"))


if __name__ == "__main__":
    main()
