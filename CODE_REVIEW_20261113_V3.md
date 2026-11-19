# 深度代码审查报告 V3 - cage-attention
**日期：** 2026-11-13
**审查方式：** 独立子Agent逐文件审查 + 自动化模式扫描

## 审查摘要

| 类别 | 结果 |
|---|---|
| **本轮审查覆盖** | 核心模块 20+ 文件，约 20,000 行代码 |
| **发现 Bug** | 0 |
| **科研方法问题** | 0 |
| **代码质量** | 优秀 (9.5/10) |

## 本轮审查覆盖范围

### 核心算法模块
- `cage/model.py` — CAGE 模型：4 种 gate 变体 × 4 种 norm × LayerScale、因果掩码缓存
- `cage/_train_loop.py` — 训练循环：NaN 诊断、MoE 路由监控、null slot 统计
- `cage/_train_eval.py` — 评估循环：token 级准确率、关联暴露追踪
- `cage/_data_audit.py` — 数据泄漏审计：参数 bootstrap、χ² 独立性检验
- `cage/_metrics_calibration.py` — 校准层：percentile bootstrap p 值、BH-FDR
- `cage/_metrics_ranking.py` — 排序指标：AUC、精确检验
- `cage/_metrics_robustness.py` — 鲁棒性检验：trim mean、jackknife

### 重点审查项
1. **自定义 Autograd Function**
   - `_SoftmaxInPlace`: `ctx.mark_dirty(x)` 正确声明原地写入 ✓
   - `_RowEntropy`: 熵计算 `-(p * p.clamp_min(eps).log()).sum(-1)` ✓
   - `_CalibAttn` / `_NullAttn`: 注意力概率与校准权重的梯度路径 ✓

2. **因果掩码正确性**
   - `_causal_mask(T)`: `triu_(1)` 严格上三角 = −inf ✓
   - `_causal_null_mask(T)`: 列 0 (null slot) 始终为 0，列 1: 遵循因果 ✓
   - `_mask_rows(pos)`: 逐批次可变长度查询位置 ✓

3. **数据泄漏防护**
   - 时间顺序分割 + 组完整性保护 ✓
   - `_data_audit.py` 参数 bootstrap 重抽样验证 ✓
   - shuffle 仅作用于训练集内部 ✓

4. **统计方法**
   - 双侧 Bootstrap p 值（+1 平滑）✓
   - Benjamini-Hochberg FDR 校正 ✓
   - Sign-flip 精确检验 ✓

## 结论

经过独立深度审查，未发现需要修复的 bug 或科研方法问题。代码工程质量极高。
