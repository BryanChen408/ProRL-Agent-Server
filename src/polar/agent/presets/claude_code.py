"""Claude Code harness — https://docs.anthropic.com/en/docs/claude-code"""

from __future__ import annotations

import json
import shlex

from polar.agent.base import BaseHarness
from polar.agent.models import AgentSpec
from polar.runtime.base import BaseRuntime, RUNTIME_AGENT_LOG_DIR, RUNTIME_SESSION_DIR
from polar.runtime.models import ExecInput


class ClaudeCodeHarness(BaseHarness):
    """Run Claude Code CLI in non-interactive mode."""

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
                    f"claude {flags_str}{model_flag} -p {escaped} "
                    f"2>&1 | tee {RUNTIME_AGENT_LOG_DIR}/claude-code.txt"
                ),
                env=env,
            )
        ]
