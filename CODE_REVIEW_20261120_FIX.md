# Code Review Report — cage-attention (2026-11-20)

## 审查范围
46 个 Python 源文件，涵盖训练、评估、指标计算与实验管线。

## 发现的问题

### 1. [CRITICAL] 语法错误：空 `with torch.no_grad():` 块
- **文件**: `cage/_train_recipe.py:658`
- **问题**: `with torch.no_grad():` 语句体为空，紧接着是同级缩进的 `with torch.inference_mode():`。Python 解析失败，整个训练诊断模块不可用。
- **根因**: 新增 `inference_mode()` 时保留了冗余的 `no_grad()` 行，但未正确嵌套或删除。
- **修复**: 删除空 `with torch.no_grad():` 行。`torch.inference_mode()` 已包含 no_grad 语义且更严格。

### 静态分析误报（经验证为正确代码）
| 模式 | 数量 | 验证结论 |
|------|------|----------|
| `eval_call` | 12 | 均为正确的推理模式切换 |
| `unprotected_division` | 22 | 均使用 `clamp_min`、`max(x, 1)`、条件表达式保护 |
| `no_grad_context_missing` | 4 | 均在训练/审计上下文中，backward 是必需的 |
| `exec_call` | 1 | `_fragments.py` 中的 exec 用于组合模块片段，仅读取本地项目文件，安全 |
| `hardcoded_seed` | 1 | `ci_gate.py` 中的 `torch.manual_seed(0)` 用于 CI 门控测试的确定性，合理 |

### 科研方法审查
- **实验设计**: 使用多 seed 重复实验，计算置信区间和标准误，方法论正确。
- **分层抽样**: 使用 Kish 有效样本量校正加权估计的方差，统计学上合理。
- **鲁棒性指标**: 跨 seed 的方差分析正确区分了模型内在变异和数据变异。
- **数据审计**: 包含完整的分布漂移检测（TVD）、近邻去重、数据污染检查。

## 修复摘要
| 文件 | 修复类型 | 影响 |
|------|----------|------|
| `cage/_train_recipe.py` | 语法错误修复 | 恢复训练诊断功能 |
