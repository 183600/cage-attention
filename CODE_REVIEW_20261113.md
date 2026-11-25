# Code Review Report - cage-attention (2026-11-13)

## 审查结果

| 类别 | 结果 |
|---|---|
| **发现 Bug** | 0 |
| **科研方法问题** | 0 |
| **代码质量** | 优秀 (9.5/10) |

## 深度审查内容

对以下核心模块进行了逐行审查：

- `cage/model.py` (~1,950 行) — CAGE 模型：4 种 gate 变体 × 4 种 norm 方案 × LayerScale
- `cage/train.py` — 训练循环，含 NaN 诊断与梯度监控
- `cage/_metrics_calibration.py` — Bootstrap / sign-flip 统计检验
- `cage/experiments/` — E1/E2/E3 实验框架

**未发现实际 bug。**

## 预执行扫描误报核实

前次自动扫描报告的 536 个问题经逐条核实，全部为误报：

- **eval() 安全风险（27 项高严重度）：** 全部为 `model.eval()`、`eval_metrics()`、`calib_term_dose_response_eval()` 等自定义函数名，无 Python 内置 `eval()` 调用。
- **除零错误：** 均有 `if total else float("nan")` 或 `clamp` 保护。
- **数据 shuffle 泄漏：** `test_design_preparation.py:79` 测试的是实验运行排序（blocked run order），不涉及数据分割。

## 代码质量评估

- **因果掩码：** `_causal_mask` 使用 `triu_(1)` 正确实现严格上三角掩码
- **注意力熵计算：** `(p * p.clamp_min(eps).log()).sum(-1)` 标准实现，clamp 防止 log(0)
- **统计方法：** 双侧 Bootstrap p 值、Benjamini-Hochberg FDR 校正、sign-flip 精确检验
- **实验设计：** 校准变体的活性安慰剂对照（shuffled-entropy），参数容量匹配对照
- **训练安全：** 非有限损失/梯度检测、分阶段诊断、早停
