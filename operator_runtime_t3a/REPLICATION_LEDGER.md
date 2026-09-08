# REPLICATION_LEDGER — 复刻对账表

- 源: /home/docker/cannbot-skills @ `13b2ae5652c75fe83a3e4114552a6477d3a01f3d`
- 目标: /home/docker/polar_can/ProRL-Agent-Server/operator_runtime_t3a

## 白名单补丁点(允许存在的全部差异)
- `agents/tilelang2ascendc-kernel-generator.md`:R2a 删 SOC 优先级 2/3(npu-smi 违禁)
- `skills/tilelang2ascend-tilelang-designer/references/attention-patterns`:官方 init.sh 的安装动作(从 translator 复制)
- `skills/tilelang2ascend-translator/scripts/evaluate_ascendc.sh`:R2b 设备获取改租约
- `hooks/skill_script_hook.py`:R3 代跑外包租约
- `hooks/doc_gate.py`:R4 文档路径重映射+降级
- `CLAUDE.md`:global 形态路径重写(官方 init.sh 语义)+ R1 尾部三条覆盖
- `settings.json.template` ← hooks/hooks.json(按 init.sh 的 settings 生成语义)
- `tools/npu_lease_exec.py`、`tools/npu_wrap.sh`:RL 卡池接线(我方件)
- `hooks/t3a_running_best.py`:R5 attempt stream+running-best(我方件)
- `hooks/skill_script_hook.py` 的 record_attempt 调用点:R5 接线(白名单内)
- `hooks/skill_script_hook.py` 的 should_intercept 内联 -c 匹配:R6(python -c 自造 wrapper 丢判决,Abs 冒烟实证)
- `hooks/skill_script_hook.py` 的 _lease_wrap_command:R8(直接形态 NPU 脚本外层包租约,acos 冒烟实证)
- `judge/`(t2a 判分链脚本 + judge_best.sh):R5 judge 侧接线(我方件,不进 agent workdir)
- `runtime/prepare_operator_workdir.py`:t3a prepare(我方件,CLI 与 t2a 同形)
- 4 个脚本的 R-paths 补丁(verification_ascendc/validate_ascendc_impl/verification_tilelang/validate_tilelang_impl):跨 skill import 改向上搜/内联

## 逐件对账

- **skills/npu-arch**: ✅ 一致
- **skills/ascendc-api-best-practices**: ✅ 一致
- **skills/ascendc-docs-search**: ✅ 一致
- **skills/ascendc-tiling-design**: ✅ 一致
- **skills/tilelang2ascend-case-simplifier**: ✅ 一致
- **skills/tilelang2ascend-operator-project-init**: ✅ 一致
- **skills/ops-profiling**: ✅ 一致
- **skills/ascendc-precision-debug**: ✅ 一致
- **skills/tilelang2ascend-precision-tuning**: ✅ 一致
- **skills/tilelang2ascend-tilelang-designer**: ⚠️ 3 处差异(应全部在白名单内)
    - `Only in /home/docker/polar_can/ProRL-Agent-Server/operator_runtime_t3a/skills/tilelang2ascend-tilelang-designer: references`
    - `Files /home/docker/cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-tilelang-designer/scripts/validate_tilelang_impl.py and /home/docker/polar_can/ProRL-Agent-Server/operator_runtime_t3a/skills/tilelang2ascend-tilelang-designer/scripts/validate_tilelang_impl.py differ`
    - `Files /home/docker/cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-tilelang-designer/scripts/verification_tilelang.py and /home/docker/polar_can/ProRL-Agent-Server/operator_runtime_t3a/skills/tilelang2ascend-tilelang-designer/scripts/verification_tilelang.py differ`
- **skills/tilelang2ascend-translator**: ⚠️ 3 处差异(应全部在白名单内)
    - `Files /home/docker/cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/scripts/evaluate_ascendc.sh and /home/docker/polar_can/ProRL-Agent-Server/operator_runtime_t3a/skills/tilelang2ascend-translator/scripts/evaluate_ascendc.sh differ`
    - `Files /home/docker/cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/scripts/validate_ascendc_impl.py and /home/docker/polar_can/ProRL-Agent-Server/operator_runtime_t3a/skills/tilelang2ascend-translator/scripts/validate_ascendc_impl.py differ`
    - `Files /home/docker/cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/scripts/verification_ascendc.py and /home/docker/polar_can/ProRL-Agent-Server/operator_runtime_t3a/skills/tilelang2ascend-translator/scripts/verification_ascendc.py differ`
- **skills/tilelang2ascend-trace-recorder**: ✅ 一致
- **skills/tilelang-op-design**: ✅ 一致
- **skills/tilelang-op-develop**: ✅ 一致
- **skills/tilelang-perf-optimization**: ✅ 一致
- **skills/ascendc-perf-optimize**: ✅ 一致
- **agents/(白名单:R2a)**: ⚠️ 1 处差异(应全部在白名单内)
    - `Files /home/docker/cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/agents/tilelang2ascendc-kernel-generator.md and /home/docker/polar_can/ProRL-Agent-Server/operator_runtime_t3a/agents/tilelang2ascendc-kernel-generator.md differ`
- **hooks/(白名单:R3/R4)**: ⚠️ 3 处差异(应全部在白名单内)
    - `Files /home/docker/cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/hooks/doc_gate.py and /home/docker/polar_can/ProRL-Agent-Server/operator_runtime_t3a/hooks/doc_gate.py differ`
    - `Files /home/docker/cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/hooks/skill_script_hook.py and /home/docker/polar_can/ProRL-Agent-Server/operator_runtime_t3a/hooks/skill_script_hook.py differ`
    - `Only in /home/docker/polar_can/ProRL-Agent-Server/operator_runtime_t3a/hooks: t3a_running_best.py`
- **workflows/**: ✅ 一致

**差异总数 10(每一行都应能对应到白名单某一条;对不上的=夹带或漏拷,必须清零)。**
