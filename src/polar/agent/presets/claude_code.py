"""Claude Code harness — https://docs.anthropic.com/en/docs/claude-code"""

from __future__ import annotations

import json
import shlex
import textwrap

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

        claude_command = f"claude {flags_str}{model_flag} -p {escaped}"
        if self.settings.get("stop_supervisor", True):
            command = self._supervised_command(claude_command)
        else:
            command = f"{claude_command} 2>&1 | tee {RUNTIME_AGENT_LOG_DIR}/claude-code.txt"

        return [ExecInput(command=command, env=env)]

    def _supervised_command(self, claude_command: str) -> str:
        stop_file = str(
            self.settings.get("stop_file")
            or f"{RUNTIME_SESSION_DIR}/.polar/STOP_NOW"
        )
        poll_seconds = str(self.settings.get("stop_poll_seconds", 5))
        stop_grace_seconds = str(self.settings.get("stop_grace_seconds", 20))
        stop_kill_grace_seconds = str(self.settings.get("stop_kill_grace_seconds", 5))
        stop_violation_file = str(
            self.settings.get("stop_violation_file")
            or f"{RUNTIME_SESSION_DIR}/.polar/STOP_VIOLATION"
        )
        stop_violation_pattern = str(
            self.settings.get("stop_violation_pattern")
            or r'"command"[[:space:]]*:[[:space:]]*"[^"]*(tools/triton_eval_pipeline[.]sh|triton_eval_pipeline[.]sh|verify[.]py|benchmark[.]py)'
        )
        log_path = f"{RUNTIME_AGENT_LOG_DIR}/claude-code.txt"
        supervisor_log = f"{RUNTIME_AGENT_LOG_DIR}/claude-supervisor.txt"
        script = f"""
            set -euo pipefail
            mkdir -p {shlex.quote(RUNTIME_AGENT_LOG_DIR)}
            stop_file={shlex.quote(stop_file)}
            stop_violation_file={shlex.quote(stop_violation_file)}
            poll_seconds={shlex.quote(poll_seconds)}
            stop_grace_seconds={shlex.quote(stop_grace_seconds)}
            stop_kill_grace_seconds={shlex.quote(stop_kill_grace_seconds)}
            stop_violation_pattern={shlex.quote(stop_violation_pattern)}
            log_path={shlex.quote(log_path)}
            supervisor_log={shlex.quote(supervisor_log)}
            set +e
            (
              set -m
              terminate_claude() {{
                reason="$1"
                echo "[polar-supervisor] terminating claude reason=${{reason}} at $(date -Is)" >> "$supervisor_log"
                kill -TERM -- "-$claude_pid" 2>/dev/null || kill -TERM "$claude_pid" 2>/dev/null || true
                sleep "$stop_kill_grace_seconds"
                if kill -0 "$claude_pid" 2>/dev/null; then
                  kill -KILL -- "-$claude_pid" 2>/dev/null || kill -KILL "$claude_pid" 2>/dev/null || true
                fi
              }}
              ( exec {claude_command} ) > >(tee "$log_path") 2>&1 &
              claude_pid=$!
              echo "[polar-supervisor] claude_pid=$claude_pid stop_file=$stop_file violation_file=$stop_violation_file poll=${{poll_seconds}}s stop_grace=${{stop_grace_seconds}}s" >> "$supervisor_log"
              stop_requested=0
              stop_deadline=0
              log_marker=0
              while kill -0 "$claude_pid" 2>/dev/null; do
                now="$(date +%s)"
                if [ "$stop_requested" = "0" ] && [ -f "$stop_file" ]; then
                  stop_requested=1
                  echo "[polar-supervisor] STOP_NOW detected at $(date -Is)" >> "$supervisor_log"
                  stop_deadline=$((now + stop_grace_seconds))
                  log_marker=$(wc -l < "$log_path" 2>/dev/null || printf '0')
                fi
                if [ "$stop_requested" = "1" ]; then
                  if [ -f "$stop_violation_file" ]; then
                    terminate_claude "stop_violation_file"
                    break
                  fi
                  if [ -f "$log_path" ]; then
                    if tail -n +"$((log_marker + 1))" "$log_path" 2>/dev/null | grep -E "$stop_violation_pattern" >/dev/null 2>&1; then
                      terminate_claude "stop_violation_log"
                      break
                    fi
                  fi
                  if [ "$stop_grace_seconds" -ge 0 ] && [ "$now" -ge "$stop_deadline" ]; then
                    terminate_claude "stop_grace_timeout"
                    break
                  fi
                fi
                sleep "$poll_seconds"
              done
              wait "$claude_pid"
              rc=$?
              if [ "$stop_requested" = "1" ]; then
                echo "[polar-supervisor] returning success after policy stop; claude_rc=$rc" >> "$supervisor_log"
                exit 0
              fi
              echo "[polar-supervisor] claude exited rc=$rc" >> "$supervisor_log"
              exit "$rc"
            )
        """
        return textwrap.dedent(script).strip()
