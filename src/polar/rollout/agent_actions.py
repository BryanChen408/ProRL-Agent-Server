"""Extract and classify agent actions from LLM API requests.

Agent CLIs send the completed tool result(s) back in the next LLM request.  The
request therefore contains both the original tool call and its result, which is
enough to label the agent-side gap without parsing CLI-specific log files.
"""

from __future__ import annotations

import json
import re
from pathlib import PurePosixPath
from typing import Any

_COMMAND_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "eval",
        re.compile(
            r"(?:^|[\s/;&|])(?:pytest|py\.test|unittest|benchmark(?:\.py)?|"
            r"verify(?:\.py)?|ascendc_eval_pipeline\.sh)(?=$|[\s;&|])",
            re.IGNORECASE,
        ),
    ),
    (
        "build",
        re.compile(
            r"(?:^|[\s/;&|])(?:cmake|make|ninja|meson|gcc|g\+\+|clang|clang\+\+|"
            r"build\.sh|setup\.py\s+(?:build|bdist|develop))(?=$|[\s;&|])",
            re.IGNORECASE,
        ),
    ),
    (
        "write",
        re.compile(
            r"(?:^|[\s/;&|])(?:cat|tee|printf|echo)\b[^;&|]*"
            r"(?:(?<!\d)>{1,2}[ \t]*(?!/?dev/null\b)|<<)",
            re.IGNORECASE,
        ),
    ),
    ("python", re.compile(r"(?:^|[\s/;&|])python(?:3(?:\.\d+)?)?(?=$|[\s;&|])")),
    ("grep", re.compile(r"(?:^|[\s/;&|])(?:rg|grep|egrep|fgrep)(?=$|[\s;&|])")),
    ("find", re.compile(r"(?:^|[\s/;&|])(?:find|fd|ls|tree)(?=$|[\s;&|])")),
    ("git", re.compile(r"(?:^|[\s/;&|])git(?=$|[\s;&|])")),
    (
        "package",
        re.compile(
            r"(?:^|[\s/;&|])(?:pip3?|uv\s+pip|apt(?:-get)?|yum|dnf|conda|mamba|npm|pnpm|"
            r"yarn)(?=$|[\s;&|])"
        ),
    ),
    ("network", re.compile(r"(?:^|[\s/;&|])(?:curl|wget)(?=$|[\s;&|])")),
    (
        "fs",
        re.compile(
            r"(?:^|[\s/;&|])(?:mkdir|cp|mv|rm|touch|chmod|chown|ln|tar|unzip|zip)"
            r"(?=$|[\s;&|])"
        ),
    ),
    ("edit", re.compile(r"(?:^|[\s/;&|])sed\s+(?:-[^\s]*i[^\s]*|--in-place)")),
    ("read", re.compile(r"(?:^|[\s/;&|])(?:cat|head|tail|less|more|sed|awk)(?=$|[\s;&|])")),
)

_DIRECT_TOOL_KINDS: dict[str, str] = {
    "read": "read",
    "write": "write",
    "edit": "edit",
    "notebookedit": "edit",
    "grep": "grep",
    "glob": "find",
    "find": "find",
    "skill": "skill",
    "webfetch": "network",
    "websearch": "network",
    "task": "subagent",
    "agent": "subagent",
    "workflow": "subagent",
}

_SHELL_TOOL_NAMES = frozenset({"bash", "shell", "local_shell", "computer"})
_AGENT_RUNNERS = re.compile(
    r"(?:^|[;&|]\s*|\s)(?:claude|codex\s+exec|gemini|opencode|qwen|hermes|"
    r"openclaw\s+agent|pi\s+--print|[^\s]*openhands-sdk[^\s]*)(?:\s|$)",
    re.IGNORECASE,
)


def classify_shell_command(command: str) -> str:
    """Return the most useful semantic label for a shell command."""
    normalized = _classification_preamble(command).replace("'", " ").replace('"', " ")
    if not normalized:
        return "shell"
    for kind, pattern in _COMMAND_RULES:
        if pattern.search(normalized):
            return kind
    return "shell"


def is_agent_runner_command(command: str) -> bool:
    """Whether an ExecInput launches an agent CLI rather than an inner tool."""
    return bool(_AGENT_RUNNERS.search(command))


def classify_tool(tool_name: str, tool_input: dict[str, Any] | None = None) -> str:
    """Classify a native agent tool, drilling into shell commands when possible."""
    normalized = tool_name.strip().lower()
    if normalized in _SHELL_TOOL_NAMES:
        command = _command_from_input(tool_input or {})
        return classify_shell_command(command)
    return _DIRECT_TOOL_KINDS.get(normalized, "other")


def extract_completed_agent_actions(request: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the tool results completed immediately before *request*.

    Anthropic Messages and OpenAI Chat Completions request shapes are supported.
    Each returned dictionary is intentionally compact: file contents and tool
    output are excluded from trace metadata.
    """
    messages = request.get("messages")
    if not isinstance(messages, list):
        return []

    actions = _extract_anthropic_actions(messages)
    if actions:
        return actions
    return _extract_openai_actions(messages)


def _extract_anthropic_actions(messages: list[Any]) -> list[dict[str, Any]]:
    result_index = -1
    results: list[dict[str, Any]] = []
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        found = [
            block
            for block in content
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        if found:
            result_index = index
            results = found
            break
    if result_index < 0:
        return []

    calls: dict[str, dict[str, Any]] = {}
    for message in messages[:result_index]:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tool_use_id = str(block.get("id") or "")
            if tool_use_id:
                calls[tool_use_id] = block

    actions: list[dict[str, Any]] = []
    for result in results:
        tool_use_id = str(result.get("tool_use_id") or "")
        call = calls.get(tool_use_id)
        if call is None:
            continue
        actions.append(
            _make_action(
                tool_use_id=tool_use_id,
                tool_name=str(call.get("name") or ""),
                tool_input=_as_dict(call.get("input")),
                is_error=bool(result.get("is_error", False)),
            )
        )
    return actions


def _extract_openai_actions(messages: list[Any]) -> list[dict[str, Any]]:
    result_messages: list[dict[str, Any]] = []
    result_ids: set[str] = set()
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "tool":
            result_messages.append(message)
            result_ids.add(str(message.get("tool_call_id") or ""))
            continue
        if result_messages:
            break
    if not result_messages:
        return []

    calls: dict[str, dict[str, Any]] = {}
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            tool_use_id = str(call.get("id") or "")
            if tool_use_id in result_ids:
                calls[tool_use_id] = call
        if len(calls) == len(result_ids):
            break

    actions: list[dict[str, Any]] = []
    for result in reversed(result_messages):
        tool_use_id = str(result.get("tool_call_id") or "")
        call = calls.get(tool_use_id)
        if call is None:
            continue
        function = _as_dict(call.get("function"))
        actions.append(
            _make_action(
                tool_use_id=tool_use_id,
                tool_name=str(function.get("name") or ""),
                tool_input=_parse_arguments(function.get("arguments")),
                is_error=False,
            )
        )
    return actions


def _make_action(
    *,
    tool_use_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    is_error: bool,
) -> dict[str, Any]:
    kind = classify_tool(tool_name, tool_input)
    command = _command_from_input(tool_input)
    target = _target_from_input(tool_input)
    description = str(tool_input.get("description") or "")
    summary = command.splitlines()[0].strip() if command else target or description or tool_name

    action: dict[str, Any] = {
        "kind": kind,
        "tool_name": tool_name,
        "tool_use_id": tool_use_id,
        "summary": _truncate(summary, 256),
        "is_error": is_error,
    }
    if command:
        action["command"] = _truncate(command.splitlines()[0].strip(), 512)
    if target:
        action["target"] = _truncate(target, 512)
    if description:
        action["description"] = _truncate(description, 256)
    return action


def _command_from_input(tool_input: dict[str, Any]) -> str:
    for key in ("command", "cmd", "script"):
        value = tool_input.get(key)
        if isinstance(value, str):
            return value
    return ""


def _classification_preamble(command: str) -> str:
    """Drop comments and heredoc bodies before matching command semantics."""
    lines: list[str] = []
    for line in command.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
        if "<<" in stripped:
            break
    return " ".join(" ".join(lines).split())


def _target_from_input(tool_input: dict[str, Any]) -> str:
    for key in ("file_path", "path", "notebook_path", "pattern", "query", "skill"):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            if key in {"file_path", "path", "notebook_path"}:
                name = PurePosixPath(value).name
                return name or value
            return value
    return ""


def _parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."
