---
name: tilelang2ascend-operator-project-init
description: "初始化 AscendC 算子工程并创建可编译的算子骨架。触发场景：(1) 用户要求创建新算子；(2) 关键词：ascendc算子、新建算子、算子目录、算子初始化；(3) 需要基于 ascend-kernel 模板快速落地。本 skill 不只建目录，还输出“可继续开发”的标准文件与检查清单。"
---

# T3A 工程初始化

仅开发子 agent 执行；主 agent 应派发 tilelang2ascendc-kernel-generator。
以下逐字复用开发子 agent Phase 1.2；固定模板保留在本 skill 的 templates/。


创建 `{output_dir}/kernel/` 目录骨架并复制固定工具文件：

```bash
mkdir -p {output_dir}/kernel/op_host
mkdir -p {output_dir}/kernel/op_kernel
mkdir -p {output_dir}/kernel/utils
# 从模板复制固定工具文件（不生成，内容固定）
cp /opt/workspace/agent_workdir/.claude/skills/tilelang2ascend-operator-project-init/templates/ascend-kernel/csrc/utils/torch_kernel_helper.h {output_dir}/kernel/utils/
```

kernel 目录结构（后续 Phase 4 由 Developer / translator skill 填充）：
```
{output_dir}/kernel/
├── CMakeLists.txt           # cmake 编译配置
├── setup.py                 # whl 打包（NpuExtension + build_lib 指向 build/）
├── ops.h                    # 算子声明 (namespace ascend_kernel)
├── register.cpp             # torch.ops.npu.* 注册
├── op_host/
│   └── <op_name>.cpp        # Host 端: tiling + EXEC_KERNEL_CMD 启动
├── op_kernel/
│   └── <op_name>.cpp        # Device 端: CopyIn → Compute → CopyOut
└── utils/
    └── torch_kernel_helper.h # EXEC_KERNEL_CMD 宏
```
