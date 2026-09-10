#!/usr/bin/env python3
"""T3A role/dispatch checks using cannbot's PreToolUse agent_id protocol.

The task text comes from the existing converter and native task-prompts template;
this hook does not implement an operator workflow or execute evaluations.
"""
import json
import os
import re
from pathlib import Path
import sys
import subprocess

DEVELOPER = 'tilelang2ascendc-kernel-generator'


def handle(data: dict, root: Path) -> dict:
    contract = json.loads((root / '.claude/task_contract.json').read_text())
    if data.get('hook_event_name') == 'UserPromptSubmit':
        return {'hookSpecificOutput': {'hookEventName': 'UserPromptSubmit',
            'additionalContext': '本会话使用当前 T3A 任务模板；旧任务说明中的初始化和派发规则由以下规则替代：\n'
                                 + contract['main_prompt']}}

    name = data.get('tool_name')
    args = data.get('tool_input') or {}
    # cannbot model-infer-optimize/hooks/pre_tool_use.py uses these native
    # fields for role protection; transcript path also covers older clients.
    sub = bool(data.get('agent_id')) or '/subagents/' in str(data.get('transcript_path', ''))
    output = {'hookEventName': 'PreToolUse', 'permissionDecision': 'allow'}
    reason = None
    if name in ('Agent', 'Task'):
        if sub:
            reason = '开发子链必须自行执行已加载的工作流；设计/转译/诊断用 Skill，禁止继续嵌套 Agent。'
        elif args.get('subagent_type') != DEVELOPER:
            reason = f'唯一注册的开发 Agent 是 {DEVELOPER}；请将端到端开发派发给它。'
        else:
            # Preserve resume IDs and diagnostic context, but always transmit
            # the canonical task paths/policies after any generated dispatch.
            marker = '\n\n[T3A canonical task]\n'
            prompt = str(args.get('prompt') or '').split(marker, 1)[0]
            output['updatedInput'] = {**args, 'prompt': prompt + marker + contract['developer_prompt']}
    elif not sub and name in ('Write', 'Edit', 'MultiEdit', 'Bash', 'Skill'):
        reason = f'主链仅调度和读取；请派发 {DEVELOPER} 开发，用 Read/Glob/Grep 检查产物。'
    elif name == 'Bash':
        command = str(args.get('command') or '')
        if re.search(r'\bpython(?:3(?:\.\d+)?)?\s+-c\s', command) and re.search(
                r'torch\.npu\.(?:is_available|device_count|current_device|set_device)\s*\(', command):
            reason = ('禁止用未经过租约的设备探针判断无 NPU；请执行原生 evaluate/verification，'
                      '设备由租约执行器分配，不得因裸探针结果跳过验证。')
    elif name == 'Skill':
        skill = args.get('skill') or ''
        registered = {p.name for p in (root / '.claude/skills').iterdir() if (p / 'SKILL.md').is_file()}
        if skill not in registered:
            reason = f'未注册 Skill: {skill}。从 .claude/skills 中选择真实 Skill；开发 Agent 不是 Skill。'
        elif skill == 'tilelang2ascend-case-simplifier' and contract['case_mode'] == 'simple':
            reason = '本任务已经是 simple 的全部 5 条用例，不再精简；备份工作 JSON 后继续 Phase 3。'
    elif name in ('Write', 'Edit', 'MultiEdit'):
        path = Path(args.get('file_path') or '')
        if not path.is_absolute():
            path = Path(data.get('cwd') or root) / path
        if not path.resolve().is_relative_to((root / contract['op_name']).resolve()):
            reason = '开发修改必须位于当前算子输出目录内；input、skill、hook 和评测工具不可修改。'
    if reason:
        output.update(permissionDecision='deny', permissionDecisionReason=reason)
    return {'hookSpecificOutput': output}


def main():
    data = json.load(sys.stdin)
    root = Path(os.environ.get('CLAUDE_PROJECT_DIR') or Path(__file__).resolve().parents[2])
    result = handle(data, root)
    if data.get('tool_name') == 'Bash' and result['hookSpecificOutput'].get('permissionDecision') == 'allow':
        # Claude runs separate hooks concurrently. Delegate only AFTER the
        # role check so a denied main-chain Bash cannot execute in a sibling hook.
        proc = subprocess.run([sys.executable, str(Path(__file__).with_name('skill_script_hook.py'))],
                              input=json.dumps(data), text=True)
        raise SystemExit(proc.returncode)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
