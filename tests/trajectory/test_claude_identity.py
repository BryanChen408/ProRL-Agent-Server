from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from polar.agent.presets.claude_code import ClaudeCodeHarness
from polar.agent.models import AgentSpec
from polar.gateway.node import GatewayNodeManager
from polar.rollout.models import SessionDispatchRequest
from polar.trajectory.models import StrategySpec
from polar.trajectory.registry import default_builder_registry
from polar.gateway.transform.anthropic import AnthropicTransformer
from polar.trajectory.models import CompletionRecord, CompletionSession


def record(name, prompt=None, system='same system'):
    return CompletionRecord(
        completion_id=name,
        request={'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': 'solve'}]},
        response={'id': name, 'choices': [{
            'input_token_ids': prompt or [1, 2],
            'message': {'role': 'assistant', 'content': name}, 'finish_reason': 'stop',
            'logprobs': {'content': [{'token_id': 10, 'logprob': -0.1}, {'token_id': 99, 'logprob': -0.2}]},
        }]},
        metadata={'policy_version': 7},
    )


def native(name, side=False, agent=None):
    return {'type': 'assistant', 'message': {'id': f'msg_{name}'}, 'isSidechain': side, 'agentId': agent}


def build(tmp_path, records, events):
    transcript = tmp_path / '.claude/projects/work/session.jsonl'
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text('\n'.join(json.dumps(e) for e in events) + '\n{"partial":')
    session = CompletionSession(session_id='s', completions=records)
    manager = GatewayNodeManager.__new__(GatewayNodeManager)
    manager.storage = SimpleNamespace(load_completion_session=lambda session_id: session)
    manager.builders = default_builder_registry()
    request = SessionDispatchRequest(
        session_id='s', task_id='t', instruction='solve', remaining_timeout_seconds=60,
        agent=AgentSpec(harness='claude_code'),
        builder=StrategySpec(strategy='prefix_merging', config={'end_of_turn_token_id': 99}),
    )
    return manager._build_trajectory(request, tmp_path)


@pytest.mark.parametrize('root_filtered', [False, True])
def test_native_identity_survives_same_prompts_and_filtered_root(tmp_path, root_filtered):
    root, child = record('00-root'), record('01-child')
    if root_filtered:
        root.response['choices'] = []
    trajectory = build(tmp_path, [root, child], [native('00-root'), native('01-child', True, 'worker')])
    roles = [t.metadata['chain_role'] for t in trajectory.traces]
    assert roles == (['sub'] if root_filtered else ['main', 'sub'])
    assert trajectory.traces[-1].metadata['chain_role_source'] == 'claude_native'
    assert trajectory.traces[-1].metadata['agent_chain_id'] == 'worker'
    assert trajectory.traces[-1].response_logprobs == [-0.1, -0.2]


def test_compaction_and_identical_subagents_keep_native_roles(tmp_path):
    records = [record('00-root'), record('01-worker'), record('02-worker'),
               record('03-compact', [50, 60], system='changed after compaction')]
    trajectory = build(tmp_path, records, [native('00-root'), native('01-worker', True, 'a'),
                                          native('02-worker', True, 'b'), native('03-compact')])
    assert [t.metadata['chain_role'] for t in trajectory.traces] == ['main', 'sub', 'sub', 'main']
    assert [t.metadata['agent_chain_id'] for t in trajectory.traces] == ['main', 'a', 'b', 'main']


def test_missing_conflicting_and_stream_fallback_identities(tmp_path):
    records = [record('00-missing'), record('01-conflict'), record('02-stream')]
    stream = tmp_path / 'logs/agent/claude-code.txt'
    stream.parent.mkdir(parents=True)
    stream.write_text(json.dumps({'type': 'assistant', 'message': {'id': 'msg_02-stream'},
                                  'parent_tool_use_id': 'tool-worker'}) + '\n')
    trajectory = build(tmp_path, records, [native('01-conflict'), native('01-conflict', True, 'a')])
    roles = {cid: trace.metadata['chain_role'] for trace in trajectory.traces
             for cid in trace.metadata['source_completion_ids']}
    assert roles == {'00-missing': 'unknown', '01-conflict': 'unknown', '02-stream': 'sub'}


def test_stream_and_nonstream_message_ids_match_native_transcript_join(tmp_path):
    response = record('completion-1').response
    transformer = AnthropicTransformer()
    original = {'model': 'claude-test'}
    nonstream_id = transformer.transform_response(response, original)['id']
    events = transformer.create_stream_state(original).process_chunk(
        {'id': response['id'], 'choices': []}, is_first=True,
    )
    assert events[0]['message']['id'] == nonstream_id == 'msg_completion-1'


@pytest.mark.parametrize("exit_code", [0, 7, 124])
def test_cli_exit_code_survives_log_capture(tmp_path, exit_code):
    from polar.runtime.base import RUNTIME_AGENT_LOG_DIR

    command = ClaudeCodeHarness(AgentSpec(harness="claude_code")).run_steps("solve")[0].command.replace(RUNTIME_AGENT_LOG_DIR, str(tmp_path))
    result = subprocess.run(
        ["bash", "-lc", f"claude() {{ echo cli-output; return {exit_code}; }}; " + command],
        capture_output=True, text=True,
    )
    assert result.returncode == exit_code
    assert (tmp_path / "claude-code.txt").read_text() == "cli-output\n"
