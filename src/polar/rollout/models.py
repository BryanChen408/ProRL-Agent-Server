"""Shared data models for rollout orchestration and gateway-node execution."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

from polar.agent.models import AgentSpec
from polar.runtime.models import RuntimeSpec
from polar.trajectory.models import EvaluatorSpec, StrategySpec, Trajectory

if TYPE_CHECKING:
    from polar.rollout.timer import StageTimer


class SessionStatus(StrEnum):
    """Canonical session lifecycle statuses.

    StrEnum instances serialize to their string values, so wire compatibility
    with older clients that read plain status strings is preserved.
    """

    REGISTERED = "REGISTERED"
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    RUNNING = "RUNNING"
    POST_RUN = "POST_RUN"
    BUILDING = "BUILDING"
    EVALUATING = "EVALUATING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"

    @classmethod
    def terminal(cls) -> frozenset["SessionStatus"]:
        return frozenset({cls.COMPLETED, cls.ERROR, cls.TIMEOUT})

    @classmethod
    def active(cls) -> frozenset["SessionStatus"]:
        return frozenset(set(cls) - cls.terminal())


def _new_stage_timer() -> "StageTimer":
    from polar.rollout.timer import StageTimer

    return StageTimer()


def _default_builder_spec() -> StrategySpec:
    return StrategySpec(strategy="per_request")


class TaskRequest(BaseModel):
    """Task submitted by the trainer."""

    task_id: str
    instruction: str
    num_samples: int = Field(default=1, ge=1)
    timeout_seconds: float = Field(default=600.0, gt=0)
    runtime: RuntimeSpec | None = None
    agent: AgentSpec
    builder: StrategySpec = Field(default_factory=_default_builder_spec)
    evaluator: EvaluatorSpec | None = None
    callback_url: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class OperatorSample(BaseModel):
    """Logical operator sample submitted by a trainer."""

    op_name: str
    group_index: int | None = None
    index: int | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    task_source: str | None = None
    task_source_sha256: str | None = None

    @field_validator("op_name")
    @classmethod
    def _validate_op_name(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("op_name must be non-empty")
        if "/" in text or "\\" in text or text in {".", ".."}:
            raise ValueError("op_name must be a file stem, not a path")
        return text

    @field_validator("task_source")
    @classmethod
    def _validate_task_source(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value:
            raise ValueError("task_source must be non-empty when provided")
        return value

    @field_validator("task_source_sha256")
    @classmethod
    def _validate_task_source_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", text):
            raise ValueError("task_source_sha256 must be a hex sha256 digest")
        return text


class OperatorSampleRequest(BaseModel):
    """Thin operator rollout request expanded by Polar-owned profiles."""

    task_id: str
    instruction: str
    num_samples: int = Field(default=1, ge=1)
    profile: str | None = None
    timeout_seconds: float | None = Field(default=None, gt=0)
    sample: OperatorSample
    metadata: dict[str, object] = Field(default_factory=dict)


class SessionDispatchRequest(BaseModel):
    """Session lifecycle request sent from the rollout server to a gateway node.

    `remaining_timeout_seconds` is the execution budget the gateway starts
    counting when the session enters INIT. The field name is kept for wire
    compatibility with existing clients.
    """

    session_id: str
    task_id: str
    instruction: str
    remaining_timeout_seconds: float = Field(gt=0)
    runtime: RuntimeSpec | None = None
    agent: AgentSpec
    builder: StrategySpec = Field(default_factory=_default_builder_spec)
    evaluator: EvaluatorSpec | None = None
    callback_url: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class SessionDispatchResponse(BaseModel):
    """Acknowledgement returned by a gateway node when a session is accepted."""

    session_id: str
    task_id: str
    status: SessionStatus
    node_id: str | None = None


class SessionTiming(BaseModel):
    """Per-session durations in milliseconds.

    Top-level fields (init_ms, run_ms, postrun_ms) are kept for backward
    compatibility and aggregate the finer breakdown fields below.  Custom
    profilers can read the `_<stage>_ms` fields directly to get per-phase
    durations without parsing marks; the coarse fields are computed as sums
    of their children when the children are populated.
    """

    model_config = ConfigDict(extra="forbid")

    # ── coarse (backward-compatible aggregates) ──
    register_to_init_queue_ms: float = 0.0
    init_ms: float = 0.0
    run_ms: float = 0.0
    postrun_ms: float = 0.0

    # ── INIT breakdown ──
    init_runtime_create_ms: float = 0.0   # create_runtime + runtime.start()
    init_docker_create_ms: float = 0.0    # docker create (subset of runtime_create)
    init_docker_start_ms: float = 0.0     # docker start  (subset of runtime_create)
    init_prepare_ms: float = 0.0          # prepare actions (upload + exec)

    # ── READY wait ──
    ready_wait_ms: float = 0.0            # init finished → run started

    # ── RUN breakdown ──
    run_harness_setup_ms: float = 0.0     # harness.setup()
    run_agent_exec_ms: float = 0.0        # agent command execution (total)
    run_harness_postprocess_ms: float = 0.0  # harness.postprocess()

    # ── LLM interaction (within run) ──
    llm_call_count: int = 0               # total LLM calls this session
    llm_total_ms: float = 0.0             # aggregate SGLang wait time
    llm_request_total_ms: float = 0.0     # aggregate full round-trip (gateway→SGLang→gateway)
    llm_agent_side_total_ms: float = 0.0  # aggregate tool + client overhead between LLM calls
    llm_calls: list[dict] = []            # per-call timing + classified agent_actions
    tool_execs: list[dict] = []           # per-tool-exec detail: [{"idx":0, "command":"bash ...", "duration_ms":..., "exit_code":0}, ...]

    # ── POSTRUN breakdown ──
    postrun_build_ms: float = 0.0         # trajectory building
    postrun_eval_ms: float = 0.0          # evaluator.evaluate()
    postrun_teardown_ms: float = 0.0      # stop runtimes + cleanup
    postrun_docker_kill_ms: float = 0.0   # docker kill  (subset of teardown)
    postrun_docker_rm_ms: float = 0.0     # docker rm -f (subset of teardown)
    postrun_push_result_ms: float = 0.0   # POST callback to rollout server

    # ── wall-clock total ──
    total_ms: float = 0.0                 # dispatch_started → return_finished


class SessionResult(BaseModel):
    """Terminal node result returned to the rollout server."""

    session_id: str
    task_id: str
    status: SessionStatus
    trajectory: Trajectory
    timing: SessionTiming = Field(default_factory=SessionTiming)
    node_id: str | None = None
    error: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class TaskResult(BaseModel):
    """Blocking response returned once all rollout sessions resolve."""

    task_id: str
    status: str  # Task-level status vocabulary: "running" | "completed" | "failed"
    results: list[SessionResult]
    result_paths: list[str] = Field(default_factory=list)


class TaskStatus(BaseModel):
    """Monitoring view for a task that may still be running."""

    task_id: str
    status: str
    total_sessions: int
    completed_sessions: int
    results: list[SessionResult] = Field(default_factory=list)
    result_paths: list[str] = Field(default_factory=list)


class TaskCancelRequest(BaseModel):
    """Cancel trainer-owned rollout tasks at a policy boundary."""

    task_ids: list[str] = Field(min_length=1)
    reason: str = "policy_cutoff"

    @field_validator("task_ids")
    @classmethod
    def _validate_task_ids(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            task_id = str(value).strip()
            if not task_id:
                raise ValueError("task_ids must not contain empty values")
            if task_id not in seen:
                normalized.append(task_id)
                seen.add(task_id)
        return normalized


class PolicyTransitionBeginRequest(BaseModel):
    transition_id: str = Field(min_length=1, max_length=256)
    policy_namespace: str = Field(default="legacy", min_length=1, max_length=128)
    from_epoch: int = Field(ge=0)
    to_epoch: int = Field(ge=0)
    engine_versions: dict[str, str] = Field(min_length=1)


class PolicyBootstrapBeginRequest(BaseModel):
    """Close an existing serving namespace before the first VIME weight load."""

    transition_id: str = Field(min_length=1, max_length=256)
    policy_namespace: str = Field(min_length=1, max_length=128)
    epoch: int = Field(ge=0)


class PolicyTransitionDrainRequest(BaseModel):
    wait_timeout_seconds: float = Field(default=30.0, ge=0, le=300.0)
    # Set only after VIME's /pause?mode=abort fanout has succeeded on every
    # serving engine. This is the NPU-safety proof; gateway/session teardown is
    # deliberately not part of the synchronous training boundary.
    engine_abort_confirmed: bool = False


class PolicyEpochInitializeRequest(BaseModel):
    transition_id: str = Field(min_length=1, max_length=256)
    policy_namespace: str = Field(default="legacy", min_length=1, max_length=128)
    epoch: int = Field(ge=0)
    engine_versions: dict[str, str] = Field(min_length=1)


class PolicyTransitionCommitRequest(BaseModel):
    verified_policy_epoch: int = Field(ge=0)
    policy_namespace: str = Field(default="legacy", min_length=1, max_length=128)
    engine_versions: dict[str, str] = Field(min_length=1)


class PolicyTransitionAbortRequest(BaseModel):
    verified_policy_epoch: int = Field(ge=0)
    policy_namespace: str = Field(default="legacy", min_length=1, max_length=128)
    engine_versions: dict[str, str] = Field(min_length=1)
    reason: str = "prepare_failed"


class PolicyTransitionFailRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=2000)


class PolicyQuiesceRequest(BaseModel):
    """Leave the current policy durably closed when its trainer exits."""

    policy_namespace: str = Field(min_length=1, max_length=128)
    epoch: int = Field(ge=0)


class NodeRegistrationRequest(BaseModel):
    """Payload sent by a gateway node when registering with the rollout server."""

    node_id: str
    gateway_url: str
    max_init_workers: int = Field(ge=1)
    max_run_workers: int = Field(ge=1)
    max_postrun_workers: int = Field(ge=1)
    heartbeat_interval_seconds: int = Field(default=30, ge=1)


class NodeStageMetrics(BaseModel):
    """Per-node stage occupancy and queue depths."""

    init_queue_depth: int = Field(default=0, ge=0)
    init_inflight: int = Field(default=0, ge=0)
    ready_depth: int = Field(default=0, ge=0)
    run_inflight: int = Field(default=0, ge=0)
    postrun_queue_depth: int = Field(default=0, ge=0)
    postrun_inflight: int = Field(default=0, ge=0)

    @property
    def total_sessions(self) -> int:
        return (
            self.init_queue_depth
            + self.init_inflight
            + self.ready_depth
            + self.run_inflight
            + self.postrun_queue_depth
            + self.postrun_inflight
        )


class NodeHeartbeatRequest(BaseModel):
    """Heartbeat payload sent by a gateway node."""

    metrics: NodeStageMetrics = Field(default_factory=NodeStageMetrics)


class GatewayNodeInfo(BaseModel):
    """External view of one schedulable gateway node."""

    node_id: str
    gateway_url: str
    max_init_workers: int
    max_run_workers: int
    max_postrun_workers: int
    metrics: NodeStageMetrics = Field(default_factory=NodeStageMetrics)
    dispatch_reservations: int = Field(default=0, ge=0)
    healthy: bool
    draining: bool = False
    heartbeat_interval_seconds: int
    last_heartbeat: datetime


@dataclass(slots=True)
class SessionContext:
    """Internal state that flows through dispatch and collection."""

    session_id: str
    task_id: str
    request: TaskRequest
    deadline_monotonic: float = field(default_factory=time.monotonic)
    node_id: str | None = None
    gateway_url: str | None = None
    timer: "StageTimer" = field(default_factory=_new_stage_timer)
    rollout_result: SessionResult | None = None
    completion_future: asyncio.Future[SessionResult] | None = field(
        default=None,
        repr=False,
    )
    cancel_requested: bool = False
    cancel_reason: str | None = None
    cancel_acknowledged: bool = False
    cancel_error: str | None = None
