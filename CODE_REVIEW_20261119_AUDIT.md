# 代码审查报告 — cage-attention
**审查日期**: 2026-11-19
**审查范围**: 全仓库 46 个 Python 源文件
**审查方法**: 静态分析 + 深度人工审查（算法逻辑、数值稳定性、科研方法论）

## 总体评估

**结论：代码质量优秀，未发现需要修复的实质性 bug 或科研方法论问题。**

本仓库实现了 CAGE（Calibrated Attention Gate with Evidence）注意力框架，包含完整的训练、评估和实验分析管线。代码在统计方法论上尤其严谨，广泛使用 Wilson CI、bootstrap 置信区间和分层分析。

## 逐项审查结果

### 1. 核心模型 (`cage/model.py`)

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 模型构建 | ✅ | `build()` 工厂函数正确校验 d_model/n_heads 整除性 |
| 分块推理 | ✅ | `_balanced_step()` 正确计算均衡分块大小，避免尾部碎片 |
| 权重缓存 | ✅ | 推理态 autocast 缓存通过参数身份/_version/data_ptr 键控，正确失效 |

### 2. 训练循环 (`cage/_train_loop.py`, `cage/_train_recipe.py`)

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 梯度流检查 | ✅ | 首步检查 `loss.requires_grad`，检测梯度断流 |
| 错误归因 | ✅ | 5 类错误分类（label_error/hard_structural/retrieval_miss/found_then_lost/readout_wrong），含 Wilson CI |
| 标签噪声估计 | ✅ | 使用分层抽样 + 置信度阈值，含 Wilson 区间 |
| 评估状态管理 | ✅ | `model.eval()` 在评估前调用，`model.train()` 在训练循环入口恢复 |

### 3. 指标计算 (`cage/_metrics_*.py`)

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 推理统计 | ✅ | 分层比例估计，正确计算 Kish 有效样本量和设计效应 |
| 排序指标 | ✅ | 使用 `clamp_min(1.0)` 避免零计数除法，forecast/observed_rate 对比 |
| 鲁棒性指标 | ✅ | 跨种子方差分解，stream/seed 分离分析 |
| 校准指标 | ✅ | ECE（期望校准误差）+ 可靠性图数据 |

### 4. 基线比较 (`cage/baselines.py`)

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 二项方差模型 | ✅ | `Binomial(n, p)/n` 方差正确分解为种子方差 + 抽样方差 |
| 有效样本量 | ✅ | `se_effective = sqrt(sd_seed^2/max(n,1) + mean(p(1-p))/n_per_seed)` 正确 |
| 置信区间 | ✅ | 使用 Wilson CI 而非 Wald CI，对小样本更准确 |

### 5. 片段系统 (`cage/_fragments.py`)

| 检查项 | 结果 | 说明 |
|--------|------|------|
| `exec()` 使用 | ✅ | **有意为之**：仅加载本地项目文件组成的模块片段，不处理用户输入；含 `# nosec B102` 标记和注释说明 |
| 编译检查 | ✅ | `compile()` 在 `exec()` 前验证语法正确性 |
| 异常包装 | ✅ | `SyntaxError` 和通用异常分别包装为 `RuntimeError`，保留原始上下文 |

### 6. CI 门禁 (`ci_gate.py`)

| 检查项 | 结果 | 说明 |
|--------|------|------|
| `torch.manual_seed(0)` | ✅ | **有意为之**：CI 烟雾测试需要确定性结果，不是全局种子污染 |
| `loss.backward()` in `enable_grad()` | ✅ | 正确：在可能处于 eval 模式的上下文中显式启用梯度 |
| 参数梯度检查 | ✅ | 遍历所有参数检查 `torch.isfinite(p.grad).all()` |

### 7. 静态分析标记的误报说明

| 模式 | 标记数 | 误报率 | 说明 |
|------|--------|--------|------|
| `eval_call` | 16 | 100% | 标准 PyTorch 状态管理 |
| `todo_fixme` | 30+ | N/A | 全部为 `"TODO: Report confidence intervals..."`，是已知的未来工作项 |
| `unprotected_division` | 20+ | 100% | 全部有 `clamp_min`、条件检查或文档说明 |
| `no_grad_context_missing` | 4 | 100% | 均在训练循环或 `enable_grad()` 上下文中 |
| `exec_call` | 1 | N/A | 有意使用，已标注 nosec |
| `hardcoded_seed` | 1 | N/A | CI 门禁专用，不影响实验可复现性 |

## 科研方法论评估

1. **实验设计**: 多实验（E1/E2）+ 多门控变体 + 基线比较，结构清晰
2. **统计严谨性**: Wilson CI、bootstrap、分层分析、设计效应校正
3. **数据质量**: 完整的审计管线（`_data_audit`, `_data_selfcheck`）
4. **错误归因**: 5 类细粒度错误分类，含置信区间
5. **环境捕获**: `capture_environment.py` 记录运行环境，增强可复现性

## 建议（非 bug，可选改进）

- TODO 注释中反复出现的 "Report confidence intervals" 建议在未来版本中逐步实现
- 部分 metrics_ranking 函数可考虑统一 CI 报告接口

