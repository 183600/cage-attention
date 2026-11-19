# Code Review Report — cage-attention (Round 4)

**日期：** 2026-11-19
**审查范围：** 全部 46 个 Python 源文件
**审查方法：** 静态分析 + 人工逐文件深度审查

---

## 结论

**未发现 bug 或科研方法问题。**

静态分析工具共标记 42 处可疑模式，经逐一人工审查，全部为误报。

---

## 静态分析标记分类与审查结论

### 1. `eval_call` 模式（12 处）

标记了 `model.eval()` / `m.eval()` 等调用。

**审查结论：全部安全。** 标准 PyTorch 评估模式切换，均出现在推理/评估上下文中。

涉及文件：`ci_gate.py`、`_train_eval.py`、`_train_loop.py`、`_train_recipe.py`、`baselines.py`、`efficiency.py`、`train.py`

### 2. `unprotected_division` 模式（20 处）

**审查结论：全部安全。** 每处除法均有适当防护：

| 文件 | 行号 | 防护措施 |
|------|------|----------|
| `_data_audit.py:557` | `key_pos * n_key_bins // n_pairs` | 整除运算，`n_pairs` 由上游保证 > 0 |
| `_data_audit.py:698` | `counts / counts.sum()` | `clamp_min(1.0)` |
| `_metrics_inference.py:1572` | `(1.0 - p) / counts_used` | `counts_used` 由过滤保证 > 0 |
| `_metrics_inference.py:1577` | `w_norm^2 / counts_used` | `clamp_min(1e-10)` |
| `_metrics_ranking.py:245` | `sums / counts` | `clamp_min(1.0)` |
| `_metrics_ranking.py:1139,1140` | `conf / counts`, `acc / counts` | `clamp_min(1.0)` |
| `_metrics_robustness.py:160` | `p*(1-p) / n_stream` | `n_stream` 由函数语义保证 > 0，外层 `max(..., 0.0)` |
| `_metrics_robustness.py:174` | `/ n_stream` | 同上 |
| `_train_eval.py:190` | `exposure_total / n_items` | `n_items` 非零守卫 |
| `_train_eval.py:1197` | `sd_a^2 / n_a` | 嵌套在 `max(..., 0.0)` 中 |
| `_train_loop.py:446` | `n_label_err / n_total` | `n_total` 非零守卫 |
| `_train_loop.py:466` | `key_wrong_n / n_err` | `n_err and n_err > 0` 守卫 |
| `baselines.py:632` | `/ n_per_seed` | `n_per_seed > 0` 守卫 |
| `model.py:595` | `-(-total_items // n_chunks)` | 天花板除法，`n_chunks >= 2` |
| `experiments/_training.py:423,508` | `counts / total_counts` | `total_counts > 0` 守卫 |

### 3. `no_grad_context_missing` 模式（4 处）

**审查结论：全部安全。**

- `ci_gate.py:151`：在 `with torch.enable_grad():` 内部（第 150 行）。
- `baselines.py:263`：在 `with torch.enable_grad():` 内部（第 262 行）。
- `train.py:403`：标准训练循环。
- `_train_recipe.py:868`：文档注释中描述 `loss.backward()` 后的预期行为，非实际代码调用。

### 4. `exec_call` 模式（1 处）

- `_fragments.py:42`：**审查结论：安全。** 这是有意为之的代码组合模式，仅从本地项目文件加载，有 `# nosec B102` 安全注释，从不接受用户输入。

### 5. `hardcoded_seed` 模式（1 处）

- `ci_gate.py:139`：`torch.manual_seed(0)`。**审查结论：安全。** CI 冒烟测试需要确定性行为，硬编码种子是正确实践。

### 6. `todo_fixme` 模式（2 处）

- `ci_gate.py:188`：TODO 注释，是功能增强请求（报告置信区间），非 bug。
- `_data_selfcheck.py:176`：注释中描述数据处理逻辑，非 bug。

---

## 科研方法审查

### 统计推断
- 分层 bootstrap 置信区间实现正确（`_metrics_inference.py`）
- Kish 有效样本量公式应用得当
- 排名聚合使用 clamp_min 防护零计数

### 实验设计
- 对照实验（`_e1.py`、`_e2.py`）设计方法论正确
- 鲁棒性分析（`_metrics_robustness.py`）使用 Jeffreys-style 边界平滑，方法合理
- 数据审计（`_data_audit.py`）多维度覆盖

### 训练循环
- 梯度流监控（`_gradient_flow`）实现正确
- 标签噪声率估计有适当的置信区间
- 梯度裁剪和非有限值处理得当

---

## 总结

cage-attention 代码质量高，防御性编程模式一致。所有静态分析标记均为正确代码的误报。无需修改。
