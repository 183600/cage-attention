# Code Review — cage-attention (2026-11-27)

## 审查范围

核心源代码文件：
- `cage/model.py` — GPT 模型（可插拔门控变体 + 归一化方案）
- `cage/_train_loop.py` — 训练循环与诊断统计
- `cage/_metrics_inference.py` — 统计推断（配对 bootstrap、等价检验、多重比较校正）
- `cage/_metrics_ranking.py` — 排序指标
- `cage/_metrics_calibration.py` — 校准指标
- `cage/_metrics_robustness.py` — 稳健性指标
- `cage/_data_task.py` — 任务数据生成
- `cage/_data_selfcheck.py` — 数据自检
- `cage/baselines.py` — 基线方法
- `cage/experiments/` — 实验设计（E1/E2/E3）

## 审查结论：代码质量优秀，未发现实质性 bug 或科研方法问题

### 1. 核心算法正确性

**校准门控公式** — `gate = sigmoid((W_g x - b0 - a*log(max(T, 1e-300)) - c*H_norm) / tau)` 实现正确。其中 `H_norm = H / log T` 的归一化使得集中度信号与序列长度解耦，消融开关（`calib_use_len`/`calib_use_ent`/`calib_shuffle_ent`/`calib_ent_norm`/`calib_pos_norm`）设计合理。

**主动控制（安慰剂）** — `calib_shuffle_ent` 将熵替换为固定位置置换版本（相同边际分布，破坏行-熵关联），正确隔离了"信息贡献"与"容量贡献"。

**归一化方案** — Pre-LN / Post-LN / Mix-LN / SeeDNorm 四种方案实现完备，Mix-LN 在前半深度用 Pre-LN、后半深度用 Post-LN。

**数据自检** — `validate_batch` 在训练门控处验证数据契约（键唯一性、答案配对），`_label_audit` 在诊断路径中对每个样本做标签完整性审计。

### 2. 统计方法

- `paired_bootstrap_ci` 实现了完整的配对差异 bootstrap：百分位置信区间、p 值（+1 平滑）、MDE80、Cohen's d / Hedges g 效应量、符号一致性、留一法 jackknife、BCA 区间、sign-flip p 值
- `bca_agrees` 字段核对 BCA 与百分位区间是否给出相同的排除零决策
- `bootstrap_below_sign_flip_floor` 标记 bootstrap p 值低于精确检验分辨率的情况
- 多重比较校正提供了 Holm-Bonferroni（FWER）与 Benjamini-Hochberg（FDR）两套方案
- 等价检验（TOST）使用双单侧检验，与配对设计互补

### 3. 实验设计

- E1（单因素消融 + 主动控制）正确分离了校准门控的两个贡献来源
- E2（归一化消融）测试了长度归一化的必要性
- E3（跨条件泛化）验证结论的稳健性
- 错误分类学（`_ERROR_TYPES`）将模型错误细分为互斥类别（标签错误 > 硬结构 > 检索遗漏 > 找到后丢失 > 读出错误），每个类别带 Wilson 区间

### 4. 代码质量

- `_zero_scalar_cache` 按 (device, dtype) 缓存零标量，使用 `inference_mode(False)` 创建以避免 autograd 问题
- `_servable` 正确判断张量是否可在当前模式下使用
- 配置验证全面（布尔字段严格校验类型，几何字段验证整数性）

### 5. 注意事项（非 bug）

- `Cfg.__post_init__` 中 `calib_scale` 被声明为非学习参数（固定超参数），因为 `c * calib_scale` 的乘积才可辨识，文档解释清晰

## 无需修改

代码无需修改，审查通过。
