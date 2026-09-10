---
name: ascendc-docs-search
description: Ascend C 开发资源检索技能。通过本地 API 文档索引、示例代码映射和在线文档兜底搜索定位开发资料，优先查本地、缺失时再查在线。当需要查询 API 用法、示例代码、兼容性信息、官方资料入口或定位文档来源时使用。
permission:
  external_directory: allow
---

# Ascend C 开发资源

## 概述

本技能提供"本地优先，在线兜底"的文档搜索能力：
- **本地资源**：当前挂载版本的 API 文档、示例代码、实现参考
- **在线搜索**：华为昇腾社区文档搜索（仅在本地资源不足时使用）

## 官方资源路径

先检查 `test -d "$ASC_DEVKIT_DIR"`。T2A 的资料挂载在 `/opt/asc-devkit`；
无需从旧 cannbot 安装目录或当前目录寻找另一个 asc-devkit。

| 资源类型 | 路径 | 说明 |
| --- | --- | --- |
| API 文档 | `$ASC_DEVKIT_DIR/docs/zh/api/` | 用 find 搜索全部同名变体，不猜子目录 |
| 官方示例 | `$ASC_DEVKIT_DIR/examples/` | 实际入口见 [示例代码目录](references/example-catalog.md) |
| 完整文档 | `$ASC_DEVKIT_DIR/docs/` | 开发说明与平台限制 |
| API 实现 | `$ASC_DEVKIT_DIR/impl/` | 文档不足时查源码 |
| HCCL 头文件 | `$ASC_DEVKIT_DIR/include/adv_api/hccl/` | 文档仍从 API 根目录按 HCCL/API 名搜索 |

官方 OPP 算子实现、编译器类型定义和部分 SDK 头文件不在 devkit 中。需要这些资料时，
Read `.claude/workflows/cannbot-reference-index.md` 的「当前工具链源码」，复用评测环境
定位真实 CANN/OPP 根目录；此范围允许只读检索，不允许修改或执行 SDK 内测试。

## 资料查找优先级

```
0. 先查 [api-index.md](references/api-index.md) 选择 API 关键词
         ↓
1. $ASC_DEVKIT_DIR/docs/zh/api/ 下的所有 .md 文档（从此根目录 find 搜索）
         ↓ 找不到
2. $ASC_DEVKIT_DIR/examples/ (按实际目录查找)
         ↓ 找不到
3. $ASC_DEVKIT_DIR/impl/ (实现代码)
         ↓ 找不到
4. 当前工具链 headers/OPP 源码（按 cannbot-reference-index.md 定位）
         ↓ 本地资料仍不足
5. 在线搜索（华为昇腾社区）
   使用 scripts/ 中的 Python 脚本
   版本过滤以实际 CANN 版本为准，不使用旧文档的版本常量
```

## ⚠️ API 变体搜索指南（重要）

**问题**：Ascend C 存在 **240+ 个带数字后缀的 API 变体**（如 `Add-25.md`），同名 API 的不同变体功能可能完全不同。

**典型案例**：
- `Add.md` - 基础版本
- `Add-25.md` - 变体版本（支持更多功能）

### 强制搜索步骤

1. **列出所有变体**：
   ```bash
   # 搜索某个 API 的所有变体（将 APIName 替换为实际 API 名称）
   find "$ASC_DEVKIT_DIR/docs/zh/api/" -name "${APIName}*.md"

   # 示例：
   # APIName.md        ← 基础版本
   # APIName-25.md     ← 变体版本
   # APIName-91.md     ← 变体版本
   ```

2. **逐一确认功能**：每个变体的函数签名、参数、功能可能完全不同

### 变体命名规律

| 后缀 | 含义 | 示例 |
|------|------|------|
| 无后缀 | 基础版本 | `Add.md` |
| `-数字` | 变体版本（数字无语义，功能可能完全不同） | `Add-25.md` |

### 变体检测命令

```bash
# 查找某个 API 的所有变体（强制，将 APIName 替换为实际名称）
find "$ASC_DEVKIT_DIR/docs/zh/api/" -name "${APIName}*.md"

# 在所有变体中搜索特定关键词
grep -rl "关键词" "$ASC_DEVKIT_DIR/docs/zh/api/" --include="*.md"
```

## 环境兼容性

T2A 当前 profile 使用 9.0.0 配套 devkit；目标芯片以注入的 `SOC_VERSION` 为准。
不要从历史对话或服务器名称猜硬件型号，详见 [环境兼容性表](references/compatibility.md)。

查阅资料时必须确认 API/方法适用于当前环境。

## 当前会话直接检索

T2A 中由当前 agent 使用 Glob/Grep/Read 或 Bash 的 find/rg 查找，不派发 Task/Explore。
先按文件名搜索 API 的所有变体；仍无结果再扩大到整个 `$ASC_DEVKIT_DIR` 按内容搜索。
Read 找不到文件后先列所在目录，选择实际文件；不连续猜目录层级或数字后缀。
已读且未变化的文档不重复加载。示例的真实入口统一见
[示例代码目录](references/example-catalog.md)。

## 在线搜索

**适用情况**：
- 本地 $ASC_DEVKIT_DIR 文档未覆盖相关 API
- 需要更详细的官方说明或最新版本信息
- 本地文档版本过旧或不完整

**快速搜索**：
```bash
# 基础搜索（推荐使用中文关键词）
python3 .claude/skills/ascendc-docs-search/scripts/ascend_search_client.py "Ascend C 临时内存申请" --max_results 5

# 搜索 API 文档
python3 .claude/skills/ascendc-docs-search/scripts/ascend_search_client.py "AscendC::Add 接口原型" --max_results 8

# 带版本过滤（仅当实际 CANN 为 9.0.0）
python3 .claude/skills/ascendc-docs-search/scripts/ascend_search_client.py "Ascend C API" --version "9.0.0"
```

**获取详细内容**：
```bash
python3 .claude/skills/ascendc-docs-search/scripts/ascend_content_fetcher.py <URL>
```

**参数说明**：
- `keyword`：搜索关键词（必需），建议使用中文
- `--max_results`：返回结果数量（1-10，默认 10）
- `--lang`：语言设置（zh/en，默认 zh）
- `--version`：版本过滤字符串（如 "9.0.0"）
- `--doc_type`：文档类型（DOC/API，默认 DOC）

在线脚本仅在现有环境已具备依赖与网络时作为兜底；失败时返回本地资料检索，不在 rollout 中安装依赖或修改脚本。

## 参考资料

- [API 文档索引](references/api-index.md)
- [示例代码目录](references/example-catalog.md)
- [环境兼容性表](references/compatibility.md)
