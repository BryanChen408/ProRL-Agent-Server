"""Unit tests for the claude_code preset's CLI flag wiring (no runtime, no Claude Code binary).

Standalone: `python tests/agent/test_claude_code_preset.py`  | or via pytest.

Locks in the RL tool-restriction fix (audit 2026-06-09): agent.settings.disallowed_tools must flow
to `--disallowedTools` on the claude CLI (bare tool names there are removed from the model's context
entirely; deny precedence holds under --dangerously-skip-permissions). The SEMANTICS were confirmed
against Claude Code docs; this test locks the WIRING (settings -> flag).
"""

from __future__ import annotations

import os
import sys

_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    from polar.agent.models import AgentSpec
    from polar.agent.presets.claude_code import ClaudeCodeHarness
    _DEPS = True
except Exception as _exc:  # noqa: BLE001 — polar/pydantic not importable on a bare host
    _DEPS = False
    _IMPORT_ERR = _exc

_BAN = "Agent WebSearch WebFetch CronCreate TaskCreate"


def _cmd(settings: dict | None = None, model_name: str | None = "qwen35") -> str:
    spec = AgentSpec(harness="claude_code", model_name=model_name, settings=settings or {})
    return ClaudeCodeHarness(spec).run_steps("Implement Abs")[0].command


def test_disallowed_tools_wired_to_flag():
    cmd = _cmd({"disallowed_tools": _BAN})
    assert "--disallowedTools" in cmd
    # value is shlex.quoted as one arg, so the whole ban list stays together
    assert _BAN in cmd


def test_allowed_tools_wired_to_flag():
    cmd = _cmd({"allowed_tools": "Read Write Edit Bash Glob Grep Skill"})
    assert "--allowedTools" in cmd and "Skill" in cmd


def test_no_tool_flags_when_unset():
    # default (no settings) must NOT inject tool flags -> Claude Code's full default toolset.
    # This is exactly the pre-fix behavior the operator rollout had (WebSearch et al. offered).
    cmd = _cmd({})
    assert "--disallowedTools" not in cmd and "--allowedTools" not in cmd


def test_baseline_flags_preserved():
    # the fix must not disturb the existing non-interactive invocation
    cmd = _cmd({"disallowed_tools": _BAN})
    assert "--dangerously-skip-permissions" in cmd
    assert "--output-format=stream-json" in cmd
    assert "--model" in cmd and "qwen35" in cmd


def test_stop_supervisor_enabled_by_default():
    cmd = _cmd({"disallowed_tools": _BAN})
    assert "/polar/session/.polar/STOP_NOW" in cmd
    assert "/polar/session/.polar/STOP_VIOLATION" in cmd
    assert "polar-supervisor" in cmd
    assert "stop_grace_seconds=20" in cmd
    assert "terminate_claude()" in cmd
    assert "stop_violation_file" in cmd
    assert "stop_violation_log" in cmd
    assert "stop_grace_timeout" in cmd
    assert "returning success after policy stop" in cmd
    assert "exit 0" in cmd
    assert "claude " in cmd


def test_stop_supervisor_can_be_disabled():
    cmd = _cmd({"stop_supervisor": False})
    assert "polar-supervisor" not in cmd
    assert "/polar/session/.polar/STOP_NOW" not in cmd
    assert "2>&1 | tee /polar/session/logs/agent/claude-code.txt" in cmd


def test_stop_supervisor_custom_stop_file():
    cmd = _cmd({
        "stop_file": "/tmp/custom-stop",
        "stop_poll_seconds": 1,
        "stop_grace_seconds": 3,
        "stop_kill_grace_seconds": 2,
        "stop_violation_file": "/tmp/custom-violation",
        "stop_violation_pattern": "verify[.]py",
    })
    assert "/tmp/custom-stop" in cmd
    assert "/tmp/custom-violation" in cmd
    assert "poll_seconds=1" in cmd
    assert "stop_grace_seconds=3" in cmd
    assert "stop_kill_grace_seconds=2" in cmd
    assert "stop_violation_pattern='verify[.]py'" in cmd


if __name__ == "__main__":
    if not _DEPS:
        print(f"[skip] polar/pydantic not importable: {_IMPORT_ERR}")
        sys.exit(0)
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [OK] {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  [XX] {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
