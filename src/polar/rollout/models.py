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

    Public session timing used for rollout metrics. `register_to_init_queue_ms`
    captures gateway-side waiting before INIT starts; the other three fields
    cover runtime startup + prepare, agent harness execution, and post-run work
    (build/eval/teardown).
    """

    model_config = ConfigDict(extra="forbid")

    register_to_init_queue_ms: float = 0.0
    init_ms: float = 0.0
    run_ms: float = 0.0
    postrun_ms: float = 0.0


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
