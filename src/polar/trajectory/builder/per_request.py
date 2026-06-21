"""Built-in trajectory builder that emits one trace per request/completion."""

from __future__ import annotations

from polar.trajectory.builder.base import BaseTrajectoryBuilder
from polar.trajectory.builder.record_filters import filter_trainable_completions
from polar.trajectory.builder.record_utils import build_trace_from_completion
from polar.trajectory.models import CompletionSession, Trajectory


class PerRequestBuilder(BaseTrajectoryBuilder):
    """Convert every stored completion record into one trajectory trace."""

    async def build(self, session: CompletionSession) -> Trajectory:
        filter_result = filter_trainable_completions(session.completions)
        if not session.completions:
            return Trajectory(
                status="ERROR",
                metadata={
                    "builder": "per_request",
                    "session_id": session.session_id,
                    "task_metadata": dict(session.metadata),
                    "record_count": 0,
                    **_top_level_scheduler_metadata(session.metadata),
                },
                traces=[],
                error="no completions",
            )
        if not filter_result.kept:
            return Trajectory(
                status="ERROR",
                metadata={
                    "builder": "per_request",
                    "session_id": session.session_id,
                    "task_metadata": dict(session.metadata),
                    "record_count": len(session.completions),
                    "completion_filter": filter_result.metadata,
                    **_top_level_scheduler_metadata(session.metadata),
                },
                traces=[],
                error="no trainable completions after completion filter",
            )

        return Trajectory(
            status="COMPLETED",
            metadata={
                "builder": "per_request",
                "session_id": session.session_id,
                "task_id": session.task_id,
                "api_type": session.api_type,
                "model_requested": session.model_requested,
                "model_used": session.model_used,
                "record_count": len(session.completions),
                "task_metadata": dict(session.metadata),
                "trace_count": len(filter_result.kept),
                "completion_filter": filter_result.metadata,
                **_top_level_scheduler_metadata(session.metadata),
            },
            traces=[
                build_trace_from_completion(completion)
                for completion in filter_result.kept
            ],
        )


def _top_level_scheduler_metadata(metadata: dict) -> dict:
    keys = {"group_id", "policy_version", "rollout_step"}
    return {key: metadata[key] for key in keys if key in metadata}
