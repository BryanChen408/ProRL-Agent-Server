# szai 拓扑集成契约（已迁出）

本文档的全部内容已并入 vime 仓的单一交接文档，此处不再维护副本，避免两份漂移：

```
/workspace/vime/HANDOFF_SZAI_TOPOLOGY.md        （训练容器内活目录）
```

对应关系：

| 原本节 | 现在位置 |
|---|---|
| 拓扑、集成契约、交会文件新鲜度 | §1、§5 |
| 为什么渲染步骤必需 + 十个 token 明细 | §3 |
| 真跑之前必须先做的事 | §7 |
| 同事 `luoss-lzc` 方案为何不合并 | §8 |
| 提交清单、当前状态 | §5、§6 |

迁出时修正了三处过时内容：gateway 默认值 `8200`→`8100`；`model_served` 默认值改为
`/models/Qwen3.6-35B-A3B`（对齐 vime `start_vime_in_platform.sh:248` 的 `HF_CKPT`）；
`task_assets_dir` 的 `glob("*.json")` 由"隐蔽的看起来成功"更正为**刻意加的护栏**
（见 `tools/load_polar_profile.py:107-112` 的注释）。

改 profile 或 launcher 之前请读那一份。
