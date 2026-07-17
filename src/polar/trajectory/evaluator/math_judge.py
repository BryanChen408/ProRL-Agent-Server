"""Rule-based math reward evaluator (DAPO-style verifiable answers).

Mirrors ``operator_judge`` but far simpler: no compile/benchmark subprocess —
just extract the agent's final ``\\boxed{}`` answer from the trajectory and
compare to a judge-only ground truth loaded per ``task_id``.

Leakage-safe: the ground truth lives in a judge-only ``answers_dir`` keyed by
``task_id`` (mirrors operator's ``verify_dir``); it is NEVER placed in the
agent-visible prompt.

StrategySpec config keys::

    answers_dir:  str    judge-only dir; per-task ``<task_id>.txt`` or an
                         ``answers.jsonl`` of ``{"task_id":..., "answer":...}``
    format_bonus: float  optional small reward for emitting a \\boxed{} (default 0)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from polar.trajectory.evaluator.base import BaseTrajectoryEvaluator
from polar.trajectory.models import EvalResult, Trajectory

# DAPO 题面要求 "put your answer on its own line after 'Answer:'"(常见 "Answer: 34" 或
# "Answer: $\boxed{34}$")→ 优先认 Answer: 行,再认 boxed,最后兜底最后一个整数。
_ANSWER = re.compile(r"Answer:\s*[^\n]*?(-?\d+)", re.I)
_BOXED = re.compile(r"\\boxed\{\s*(-?\d+)\s*\}")
_INT = re.compile(r"-?\d+")


def _msg_text(msg: dict[str, Any]) -> str:
    """Extract text from an OpenAI-style message (content str OR list of blocks)."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                parts.append(str(b.get("text", "")))
            else:
                parts.append(str(b))
        return " ".join(parts)
    return "" if content is None else str(content)


def _final_text(trajectory: Trajectory) -> str:
    """Concatenate all assistant response text across traces (answer is in the last \\boxed{})."""
    parts: list[str] = []
    for tr in trajectory.traces:
        for msg in tr.response_messages:
            if isinstance(msg, dict):
                parts.append(_msg_text(msg))
    return "\n".join(p for p in parts if p)


def _answer_from_trajectory(trajectory: Trajectory) -> str | None:
    """Ground truth travels judge-only via task metadata (never in the agent prompt).

    polar's per_request builder puts ``task_metadata = dict(session.metadata)`` into
    ``trajectory.metadata`` (and each trace's metadata). vime places the answer there
    via the task-request payload metadata. Agent only sees ``instruction`` → no leak.
    """
    def _dig(md: dict[str, Any]) -> str | None:
        if not isinstance(md, dict):
            return None
        tm = md.get("task_metadata")
        if isinstance(tm, dict) and tm.get("answer") is not None:
            return str(tm["answer"])
        if md.get("answer") is not None:  # tolerate answer placed directly
            return str(md["answer"])
        return None

    ans = _dig(trajectory.metadata)
    if ans is not None:
        return ans
    for tr in trajectory.traces:
        ans = _dig(getattr(tr, "metadata", {}) or {})
        if ans is not None:
            return ans
    return None


def _extract_answer(text: str) -> str | None:
    for pat in (_ANSWER, _BOXED):  # DAPO 显式格式优先
        m = pat.findall(text)
        if m:
            return m[-1].strip()
    nums = _INT.findall(text)  # 兜底:最后一个整数
    return nums[-1] if nums else None


def _norm(s: str | None) -> str | None:
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace("$", "").replace("\\", "").strip()
    try:
        return str(int(float(s)))
    except Exception:
        return s


class MathJudgeEvaluator(BaseTrajectoryEvaluator):
    """Return outcome_reward 1.0 for a correct boxed answer, else 0.0 (+ optional format bonus)."""

    def __init__(
        self,
        *,
        answers_dir: str | None = None,
        format_bonus: float = 0.0,
        **_ignored: Any,
    ) -> None:
        self._answers_dir = Path(answers_dir) if answers_dir else None
        self._format_bonus = float(format_bonus)
        self._index: dict[str, str] | None = None

    def _load_answer(self, task_id: str | None) -> str | None:
        if task_id is None or self._answers_dir is None:
            return None
        per_task = self._answers_dir / f"{task_id}.txt"
        if per_task.is_file():
            return per_task.read_text().strip()
        if self._index is None:
            self._index = {}
            idx = self._answers_dir / "answers.jsonl"
            if idx.is_file():
                for ln in idx.read_text().splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    rec = json.loads(ln)
                    self._index[str(rec["task_id"])] = str(rec["answer"])
        return self._index.get(str(task_id))

    async def evaluate(self, trajectory: Trajectory, **runtime: Any) -> EvalResult:
        task_id = runtime.get("task_id")
        # 主:答案随 task metadata 进 trajectory(judge-only);备:answers_dir by task_id
        gt = _answer_from_trajectory(trajectory)
        if gt is None:
            gt = self._load_answer(task_id)
        text = _final_text(trajectory)
        pred = _extract_answer(text)
        has_boxed = "\\boxed{" in text
        correct = gt is not None and pred is not None and _norm(pred) == _norm(gt)
        reward = (1.0 if correct else 0.0) + (self._format_bonus if has_boxed else 0.0)
        return EvalResult(
            outcome_reward=reward,
            metadata={
                "task_id": task_id,
                "pred": pred,
                "gt": gt,
                "correct": bool(correct),
                "has_boxed": has_boxed,
                "gt_found": gt is not None,  # False = 答案没 staged 到 answers_dir(排查用,别当解错)
            },
        )
