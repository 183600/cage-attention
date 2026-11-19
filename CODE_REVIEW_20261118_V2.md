# 代码审查报告 (2026-11-18 第二轮)

## 审查概要

| 项目 | 内容 |
|------|------|
| 仓库 | cage-attention |
| 审查日期 | 2026-11-18 (第二轮独立审查) |
| 审查方法 | 多模式深度分析：AST 分析 + 数值稳定性 + 梯度正确性 + 科研方法学 + 模型架构验证 |
| 审查范围 | 全部 46 个 Python 源文件 |

## 总体评价

代码质量**优秀**。经独立审查确认此前审查结论，未发现需要修复的实际缺陷。

## 审查维度与结论

### 1. 模型架构正确性

CAGE-Attention 的门控公式实现：

```
gate = sigmoid((W_g·x - b0 - a·log(max(T, 1e-300)) - c·H_norm) / tau)
```

- `calib_scale` 作为固定超参数而非可学习参数：正确——注释说明了可识别性问题（`c * calib_scale` 的有效系数若拆为两个自由标量则不可识别）。
- `calib_shuffle_ent` 的主动控制设计：将熵值做位置置换以分离"信息"与"存在额外输入"的效果——方法论正确。
- SeedNorm 动态归一化：scale 从隐藏状态预测，应用方式不引入数据泄漏。

### 2. 训练循环正确性

- `ci_gate.py:139` 的 `torch.manual_seed(0)`：CI 门控测试，固定种子确保可重复性，正确。
- `ci_gate.py:150` 的 `loss.backward()` 后无 `zero_grad()`：这是烟雾测试（只检查梯度是否有限），不需要完整的优化器步骤。
- 各训练文件（`_train_loop.py`、`_train_recipe.py`、`baselines.py`、`train.py`）均有完整的 `zero_grad → backward → clip → step` 循环。

### 3. 统计方法正确性

- `_metrics_inference.py` 的分层抽样方差估计：使用 Kish 有效样本量公式，分母有 `clamp_min(1e-10)` 保护。
- `_metrics_ranking.py` 的排名预测：使用 `counts.clamp_min(1.0)` 避免除零。
- `_metrics_robustness.py` 的流式标准误计算：使用 `n_stream` 归一化，有 `max(..., 0.0)` 保护 sqrt 参数。
- 35 处 TODO 注释 "Report confidence intervals"：标记为未来改进方向，不影响当前正确性。

### 4. `_fragments.py` 的 `exec()` 调用

```python
exec(code, namespace)  # nosec B102
```

- **判定：安全**。仅加载本地项目文件（经 `compile` 预编译），不处理用户输入。这是模块组合的架构设计，非安全漏洞。

### 5. 安全性

- `exec()` 调用经上述分析为安全的设计选择。
- 其余无 `eval()` / 裸 `except` / 不安全 `yaml.load` 等问题。

## 修复清单

无。本轮审查未发现需要修复的代码缺陷。

## 审查结论

cage-attention 的模型实现、训练循环和统计分析均正确可靠。代码体现了严谨的科研方法学设计（主动控制、消融实验、分层统计）。35 处 TODO 注释为合理的后续改进标记，不构成当前版本的缺陷。
