# Code Review Report — cage-attention
**日期**: 2026-11-16
**审查范围**: cage/ 目录（19,106行Python代码）+ tests/ 目录

## 审查摘要

对 cage-attention 仓库进行了全面的代码审查。该仓库实现了 CAGE（Context-Aware Gated Expert）注意力机制的研究管线。

## 发现的问题

### 误报分析（自动扫描标记但确认为安全的项）

#### 1. eval() 调用 — 全部为误报
- 自动扫描标记了9处 `eval()` 调用为高风险
- 经 AST 分析确认：全部为 `model.eval()`（PyTorch 评估模式）
- **结论**: 无安全风险

#### 2. 数据 shuffle 导致数据泄漏 — 误报
- 自动扫描标记了 `_data_task.py:16,137,728` 为潜在数据泄漏
- 经审查确认：
  - 第16行：位于注释中，描述分割协议设计原则
  - 第137行：位于 `time_ordered_split()` 文档中
  - 第728行：位于 `assert_time_ordered()` 验证函数中
- 代码实际实现了严格的分割前分配（split-before-generate）协议
- **结论**: 数据分割协议正确，无泄漏风险

#### 3. 除零保护 — 已有完善防护
- `_metrics_inference.py:1543` — `weight_total <= 0.0` 检查在除法前执行
- `_metrics_inference.py:1572` — 使用 `counts_used.clamp_min(1e-10)` 保护
- **结论**: 所有关键除法均有保护

### 审查通过的项目

#### 科研方法论
- ✅ **数据分割协议**: seed-owned stream 分割，训练入口通过 `check_train_seed` 拒绝评估种子
- ✅ **时间序列分割**: `time_ordered_split()` 保证时序连续性，`group_split()` 保证组完整性
- ✅ **分割验证**: `assert_time_ordered()` 验证分割的时间顺序正确性
- ✅ **训练循环**: 正确处理 non-finite loss/gradient，梯度裁剪，调度器在跳过步仍前进
- ✅ **评估梯度管理**: 使用 `torch.no_grad()` 上下文
- ✅ **多重比较校正**: 实现了 Benjamini-Hochberg 和 Benjamini-Yekutieli 方法

#### 代码质量
- **防御性编程**: 非有限值检测、梯度裁剪、输入验证
- **文档质量**: 详细的协议说明和设计决策注释
- **测试覆盖**: 包含数据契约测试、多重比较模型测试

## 代码质量评估

- **整体质量**: 优秀
- **主要优势**: 数据分割协议设计严谨，训练循环健壮
- **无需要修复的问题**: 所有自动扫描标记的问题均为误报或已有防护
