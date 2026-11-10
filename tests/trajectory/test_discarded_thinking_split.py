"""Only proven orphan-thinking deletion may introduce a new chain boundary."""
from copy import deepcopy

import pytest

from polar.trajectory.builder import prefix_merging as pm
from test_prefix_merging_builder import _record, _build, EOT


def records():
    history = [{"role": "user", "content": "task"}]
    first = _record('00', [1, 2], [10, EOT], prompt_messages=history)
    first.response['choices'][0]['message'] = {
        'role': 'assistant', 'content': None, 'reasoning_content': 'reasoning B',
    }
    second = _record('01', [1, 2, 50, EOT, 51], [20, EOT],
                     prompt_messages=history + [{'role': 'user', 'content': 'continue'}])
    third = _record('02', [1, 2, 50, EOT, 51, 20, EOT, 61], [30, EOT],
                    prompt_messages=second.request['messages'] + [second.response['choices'][0]['message'],
                                    {'role': 'user', 'content': 'again'}])
    return [first, second, third]


def valid_targets(trajectory):
    return [(token, lp) for t in trajectory.traces
            for token, lp, mask in zip(t.response_ids, t.response_logprobs, t.loss_mask)
            if mask]


def baseline(monkeypatch, rs):
    with monkeypatch.context() as m:
        m.setattr(pm, '_dropped_thinking_reply', lambda *_: False)
        return _build(rs)


def test_only_boundary_splits_preserving_thinking_and_learning_tokens(monkeypatch):
    rs = records()
    old = baseline(monkeypatch, rs)
    new = _build(rs)
    assert len(old.traces) == 1
    assert len(new.traces) == 2
    assert new.traces[0].response_ids == [10, EOT]
    assert new.traces[1].prompt_ids == rs[1].response['choices'][0]['input_token_ids']
    assert new.traces[1].metadata['source_completion_ids'] == ['01', '02']
    assert valid_targets(new) == valid_targets(old)
    assert new.traces[0].metadata['break_reason'] == 'discarded_thinking_context'


@pytest.mark.parametrize('case', ['whitespace', 'text', 'tool', 'no_reasoning',
                                  'retained', 'rewritten', 'length', 'all_masked', 'matching_tokens'])
def test_unrelated_trajectories_unchanged(monkeypatch, case):
    rs = records()
    message = rs[0].response['choices'][0]['message']
    if case == 'whitespace': message['content'] = '\n\n'
    if case == 'text': message['content'] = 'answer'
    if case == 'tool': message['tool_calls'] = [{'id': 'call', 'type': 'function', 'function': {'name': 'Read', 'arguments': '{}'}}]
    if case == 'no_reasoning': message['reasoning_content'] = ''
    if case == 'retained': rs[1].request['messages'].insert(1, deepcopy(message))
    if case == 'rewritten': rs[1].request['messages'][0] = {'role': 'user', 'content': 'different'}
    if case == 'length': rs[0].response['choices'][0]['finish_reason'] = 'length'
    if case == 'all_masked':
        for r in rs[1:]: r.response['choices'][0]['finish_reason'] = 'length'
    if case == 'matching_tokens':
        rs = rs[:2]
        rs[1].response['choices'][0]['input_token_ids'] = [1, 2, 10, EOT, 51]
    assert _build(rs).model_dump() == baseline(monkeypatch, rs).model_dump()


@pytest.mark.parametrize('mode', ['post_best', 'post_best_all_masked', 'budget', 'budget_all_masked'])
def test_split_preserves_existing_mask_boundaries(monkeypatch, mode):
    monkeypatch.setenv('POLAR_ATTEMPT_CREDIT', '1')
    monkeypatch.setenv('POLAR_POST_BEST_MASK', '1')
    rs = records()
    state = {
        'best_ordinal': 0,
        'ordinal_by_completion_id': {'00': (0, 'best'), '02': (1, 'later')},
        'prev_ordinal_by_completion_id': {'00': -1, '01': 0, '02': 0},
        'total_events': 2,
        'verdict_by_call_id': {},
        'score_by_call_id': {'best': 1.0, 'later': 0.0},
    }
    if mode == 'post_best_all_masked':
        state['ordinal_by_completion_id'] = {'00': (0, 'best'), '01': (1, 'later')}
        state['prev_ordinal_by_completion_id']['02'] = 1
    if mode.startswith('budget'):
        state['best_ordinal'] = None
        state['budget_completion_ids'] = {'00'} if mode.endswith('all_masked') else {'00', '01'}
    monkeypatch.setattr(pm, '_prepare_attempt_span_state', lambda *args: state)
    old = baseline(monkeypatch, rs)
    new = _build(rs)
    assert valid_targets(old) == valid_targets(new)
    if mode.endswith('all_masked'):
        assert new.model_dump() == old.model_dump()
    else:
        assert len(new.traces) == 2
        assert new.traces[1].loss_mask[-2:] == [0, 0]
