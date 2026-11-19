# Code Review Report — cage-attention (Round 11)

**日期**: 2026-11-19
**审查方法**: 深度逐行审查 + 模式扫描 + 跨仓库对比分析
**审查范围**: 全部 46 个 Python 源文件

## 审查结论

发现并修复了 **4 处真实 bug**，均为 `model.eval()` 后紧跟 `model.train()` 使评估模式失效。

## 发现的问题

### Bug: `model.eval()` 后立即 `model.train()` 导致 eval 模式失效

**严重程度**: 中等（影响评估/测量的正确性）

**影响**: 评估函数在训练模式下执行前向传播，dropout 保持活跃，导致：
- 验证 accuracy 带噪声（dropout 随机关闭部分神经元）
- 效率测量不准确
- 返回的模型处于错误的训练模式

| 文件 | 行号 | 影响 |
|---|---|---|
| `cage/_train_eval.py` | 283–284 | 评估函数: 验证 accuracy 在 train 模式下计算 |
| `cage/baselines.py` | 279–280 | 训练后返回的模型处于 train 模式而非 eval |
| `cage/efficiency.py` | 145–146 | 效率测量: 延迟和内存数据在 train 模式下采集 |
| `cage/train.py` | 567–568 | `eval_accuracy()`: 评估在 train 模式下执行 |

**注意**: cage-attention 在之前的审查轮次（commit 16be21d）已修复了 `_train_loop.py` 和 `_train_recipe.py` 中的 2 处同类 bug。本次发现的是其余文件中残留的同类问题。

## 静态分析标志复核

| 标志模式 | 数量 | 复核结果 |
|---|---|---|
| `eval_call` | 12 | 全部为标准 PyTorch 用法 |
| `unprotected_division` | 22 | 全部有 clamp_min 或三元保护 |
| `exec_call` | 1 | 本地模块组合，有 `# nosec B102` 注释 |
| `hardcoded_seed` | 1 | CI 确定性需要 |
| `no_grad_context_missing` | 2 | 均在 enable_grad() 中 |
| `todo_fixme` | 2 | CI 改进备注 |

## 科研方法论审查

- 模型架构（CAGE 门控）: 正确实现
- 统计推断: 分层推断 + Simes 检验 + BH 校正
- 数据审计: 卡方置换检验
- 训练管线: 梯度流监控 + 非有限值诊断

## 结论

代码质量优秀。本次修复了 4 处残留的 eval/train 排序 bug，加上之前已修复的 2 处，该类型问题已全部修复。
