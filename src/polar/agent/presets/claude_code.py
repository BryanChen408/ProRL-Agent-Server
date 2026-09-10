"""Claude Code harness — https://docs.anthropic.com/en/docs/claude-code"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from polar.agent.base import BaseHarness
from polar.agent.models import AgentSpec
from polar.runtime.base import BaseRuntime, RUNTIME_AGENT_LOG_DIR, RUNTIME_SESSION_DIR
from polar.runtime.models import ExecInput
from polar.trajectory.models import CompletionSession


class ClaudeCodeHarness(BaseHarness):
    """Run Claude Code CLI in non-interactive mode."""

    _DEFAULT_API_TIMEOUT_MS = "14400000"  # 4h; avoid 10-minute long-context retries.
    _DEFAULT_MAX_RETRIES = "1"

    def __init__(self, agent_spec: AgentSpec) -> None:
        super().__init__(agent_spec)
        # Absolute path outside the workspace — $HOME won't expand in docker
        # exec -e, and a literal "$HOME" dir would get swept into git add -A.
        self._config_dir = f"{RUNTIME_SESSION_DIR}/.claude"

    async def setup(self, runtime: BaseRuntime) -> None:
        await runtime.exec(f"mkdir -p {self._config_dir}")

        # Register MCP servers
        if self.mcp_servers:
            mcp_config: dict[str, dict] = {}
            for server in self.mcp_servers:
                entry: dict = {}
                if server.transport == "stdio":
                    entry["command"] = server.command
                    if server.args:
                        entry["args"] = server.args
                    entry["type"] = "stdio"
                else:
                    entry["url"] = server.url
                    entry["type"] = server.transport
                mcp_config[server.name] = entry
            config = {"mcpServers": mcp_config}
            config_json = json.dumps(config)
            await runtime.exec(
                f"cat > {self._config_dir}/.claude.json << 'POLARCFG'\n{config_json}\nPOLARCFG"
            )

        # Copy skills
        if self.skills_path:
            await runtime.exec(
                f"mkdir -p {self._config_dir}/skills && "
                f"cp -r {shlex.quote(self.skills_path)}/* {self._config_dir}/skills/ 2>/dev/null || true"
            )

    def run_steps(self, instruction: str) -> list[ExecInput]:
        escaped = shlex.quote(instruction)

        flags: list[str] = [
            "--verbose",
            "--output-format=stream-json",
            "--dangerously-skip-permissions",
        ]
        for key, cli in [
            ("max_turns", "--max-turns"),
            ("reasoning_effort", "--effort"),
            ("max_budget_usd", "--max-budget-usd"),
            ("fallback_model", "--fallback-model"),
            ("append_system_prompt", "--append-system-prompt"),
            ("allowed_tools", "--allowedTools"),
            ("disallowed_tools", "--disallowedTools"),
        ]:
            value = self.settings.get(key)
            if value is not None:
                flags.append(f"{cli} {shlex.quote(str(value))}")

        flags_str = " ".join(flags)
        env: dict[str, str] = {
            **self.env,
            "CLAUDE_CONFIG_DIR": self._config_dir,
            # Allow --dangerously-skip-permissions / bypassPermissions inside
            "IS_SANDBOX": "1",
            # Suppress Statsig / telemetry calls that the CLI otherwise makes
            # to api.anthropic.com even when ANTHROPIC_BASE_URL points elsewhere.
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
        env.setdefault("API_TIMEOUT_MS", self._DEFAULT_API_TIMEOUT_MS)
        env.setdefault("CLAUDE_CODE_MAX_RETRIES", self._DEFAULT_MAX_RETRIES)
        # Bash tool timeouts. NOTE these are `setdefault` over `self.env` (AgentSpec.env)
        # and the result is handed to ExecInput below, which OVERRIDES the container env —
        # so a deployment that wants a different value must set it in the profile's
        # `agent.env`, NOT in `runtime.env`. Putting it in `runtime.env` silently loses:
        # run 133937 configured BASH_MAX_TIMEOUT_MS=1800000 there, the container really had
        # it, and the CLI still advertised "up to 600000ms" because these two lines wrote
        # 600000 on top. 92 of that run's 125 pipeline calls dutifully requested 1800000
        # and were clamped back to 10 min.
        #
        # 600000 is also the CLI's own built-in ceiling (its Bash tool resolves max as
        # `max(builtin, default)`), so as a *default* both lines are no-ops — which is why
        # this never showed up until a non-default value was tried. They stay as an
        # explicit floor for harnesses whose CLI default is lower (120000 upstream).
        env.setdefault("BASH_DEFAULT_TIMEOUT_MS", "600000")
        env.setdefault("BASH_MAX_TIMEOUT_MS", "600000")
        if self.settings.get("max_thinking_tokens"):
            env["MAX_THINKING_TOKENS"] = str(self.settings["max_thinking_tokens"])

        # Model config: if model_name is set, use --model flag and pin all tier
        # aliases to the same model so claude-code doesn't try to route a
        # sub-agent / fallback request back to api.anthropic.com.
        model_flag = ""
        if self.model_name:
            model_flag = f" --model {shlex.quote(self.model_name)}"
            for alias in (
                "ANTHROPIC_MODEL",
                "ANTHROPIC_DEFAULT_SONNET_MODEL",
                "ANTHROPIC_DEFAULT_OPUS_MODEL",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                "CLAUDE_CODE_SUBAGENT_MODEL",
            ):
                env[alias] = self.model_name

        return [
            ExecInput(
                command=(
                    f"set -o pipefail; claude {flags_str}{model_flag} -p {escaped} "
                    f"2>&1 | tee {RUNTIME_AGENT_LOG_DIR}/claude-code.txt"
                ),
                env=env,
            )
        ]


def annotate_completion_roles(session: CompletionSession, session_dir: Path) -> None:
    """Join native Claude identities to captured completions by message ID.

    Transcripts survive compaction and include subagents' full messages. The CLI
    stream is a fallback for a transcript tail not yet flushed at timeout.
    Missing or conflicting identities remain unknown; never infer a role from text.
    """
    identities: dict[str, tuple[str, str | None]] = {}
    projects = session_dir / ".claude" / "projects"
    paths = [*sorted(projects.rglob("*.jsonl")), session_dir / "logs/agent/claude-code.txt"]
    for path in paths:
        try:
            stream = path.open(encoding="utf-8", errors="replace")
        except OSError:
            continue
        with stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if not isinstance(event, dict) or event.get("type") != "assistant":
                    continue
                message = event.get("message")
                message_id = message.get("id") if isinstance(message, dict) else None
                if not isinstance(message_id, str) or not message_id:
                    continue
                native_transcript = type(event.get("isSidechain")) is bool
                if native_transcript:
                    role = "sub" if event["isSidechain"] else "main"
                    actor = str(event.get("agentId") or path.relative_to(session_dir)) if role == "sub" else "main"
                elif "parent_tool_use_id" in event:
                    parent = event["parent_tool_use_id"]
                    if parent is not None and (not isinstance(parent, str) or not parent):
                        continue
                    role, actor = ("sub", parent) if parent is not None else ("main", "main")
                else:
                    continue
                previous = identities.get(message_id)
                if previous is None:
                    identities[message_id] = (role, actor)
                elif previous[0] != role or (native_transcript and previous[1] != actor):
                    identities[message_id] = ("unknown", None)

    for completion in session.completions:
        response_id = completion.response.get("id")
        role, actor = identities.get(f"msg_{response_id}", ("unknown", None))
        completion.metadata.update({
            "chain_role": role,
            "chain_role_source": "claude_native" if role != "unknown" else "unresolved",
            "agent_chain_id": actor,
        })
