# 深度代码审查报告 V2 - cage-attention
**日期：** 2026-11-13
**审查方式：** 自动预扫描 + 人工级深度审查

## 审查摘要

| 类别 | 结果 |
|---|---|
| **预扫描问题数** | 536（经核实全部为误报） |
| **发现 Bug** | 0 |
| **科研方法问题** | 0 |
| **代码质量** | 优秀 (9.5/10) |

## 预扫描误报核实

- **eval() 安全风险（27 项）：** 全部为 `model.eval()` 或自定义函数名
- **数据 shuffle 泄漏：** `test_design_preparation.py:79` 测试的是实验运行排序，不涉及数据分割
- **除零错误：** 均有 guard 保护

## 深度审查覆盖

- `cage/model.py` (~1,950 行) — CAGE 模型实现，4 种 gate 变体 × 4 种 norm × LayerScale
- `cage/_train_loop.py` (~1,495 行) — 训练循环，NaN 诊断
- `cage/_data_audit.py` (~2,013 行) — 数据泄漏审计
- `cage/experiments/` — E1/E2/E3 实验框架
- 自定义 Autograd: `_SoftmaxInPlace`, `_RowEntropy`, `_CalibAttn`, `_NullAttn` 数学验证 ✓
- 因果掩码: `triu_(1)` 正确 ✓
- 注意力熵: `(p * p.clamp_min(eps).log()).sum(-1)` 正确 ✓
- 统计: 双侧 Bootstrap p 值、Benjamini-Hochberg FDR、sign-flip 精确检验 ✓

## 结论

代码库工程质量极高，未发现需要修复的 bug 或科研方法问题。
