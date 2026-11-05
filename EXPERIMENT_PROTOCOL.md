# CAGE-Attention 实验协议文档

## 版本控制

| 版本 | 日期 | 变更说明 | Git SHA |
|------|------|----------|---------|
| 1.0 | 2026-11-03 | 初始版本，对应预注册假设注册表 v1 | - |
| 1.1 | 2026-11-04 | 新增 §3.5：单因素对照的**设计期证书**（把既有"受控对比只改一个因子"的要求从行级审计前移到开跑前；不新增/修改任何假设、臂、对比对、SESOI 或判定规则，故 `_config.py:HYPOTHESES` 与 `docs/design_notes.md` 无需改动） | - |
| 1.2 | 2026-11-04 | 修正 §2.3 的 E2 臂表：第 8 臂按实现为 `null`（原文误列为 E1 的安慰剂 `calib_shuf_ent`，而 E2 预注册决策规则中没有安慰剂对比）；文档与实现对齐，不改任何假设/对比/规则 | - |
| 1.3 | 2026-11-04 | 新增 §3.6：**研究级分析计划**（跨 E1/E2/E3 的预注册确认性对比集 + 声明式合取 + 合并族 FDR 裁决）；修正 §2.1 封存测试种子（456 → **2024**，与 `cage/_data_task.py:TEST_SEED` 对齐）与 §2.2 网格默认值（与 `Settings` 默认值对齐）；§5.5/附录 C 登记新增制品 | - |
| 1.4 | 2026-11-04 | 在 §3.6 增加**研究级单因素证书**：确认性对比集里的每条 `one_factor` 腿必须恰好变动一个解析后的模型因子、且是其家族已声明的受控对比对；`interaction` 腿必须是两个已声明单因素边际之差再跨两个上下文相减（两轴因子不同）。证书写入 `study_design_preflight.json`，不通过即拒绝开跑；不新增/修改任何假设、对比对、SESOI 或判定规则 | - |
| 1.5 | 2026-11-05 | 新增 **H1h 位置混淆消融臂**（`calib_pos_norm`）与研究级腿 **P10**：注册门控用全局 `log(T)` 归一化每一行的注意力熵，而因果行 t 最多只能看到 t+1 个键，其熵上限是 `log(t+1)`——全局除数把**位置相关的开放度调度**混进了集中度信号。新臂改用每行可见 token 数 `log(max(t+1,2))` 做除数与长度项（架构/参数/计算/配方不变），在最终 query 行与注册臂数值相同（`log(max(T,2)) = log T`），因此只改动中间行。判定为**单侧非劣性**（`calib_pos_norm - calib` 的 95% CI 下界 ≥ -0.01）+ 机制行（位置中性臂自身的 \|r\|(gate, entropy) ≥ 0.05）。E1 网格 8→9 cell、单因素对 7→8 对、确认性腿 9→10（声明 C1 增加 P10）；其余假设/对比/SESOI 不变 | - |
| 1.6 | 2026-11-05 | 可复现性与工程规范修订（**不改动任何假设、臂、对比对、SESOI、判定规则与统计方法**）：①新增声明依赖 vs 实装依赖的一致性检查（`--check-dependencies` / `make verify-deps` / `make preflight`），并可用归档环境记录重算历史结果的依赖符合性；②`capture_environment.py` 改为复用包内唯一的环境记录实现，两套指纹不再互不相容；③`Dockerfile.repro` 改为从 `requirements-ci.txt` 安装（此前固定的 torch 版本在所选 cu118 索引中不存在，且 `pip install -e .` 会重新解析依赖），并加构建期版本自检；④修正附录 A 中并不存在的 lr 敏感性开关（该伴随读出随 E1 自动产出）；⑤新增文档-代码漂移守卫测试（引用的版本、CLI 参数、make 目标、容器配方、文档链接） | - |

> **重要**：本协议文档与 `cage/experiments/_config.py:HYPOTHESES`、`docs/design_notes.md` 及
> `cage/experiments/_analysis_plan.py:_PRIMARY_CONTRASTS`/`_STUDY_CLAIMS`
> 形成**四位一体**的预注册体系：前三者定义单条假设与其判定规则，第四者定义
> **哪些对比承担研究的确认性主张**以及它们如何合成研究级声明。任何实验协议变更
> 必须同步更新相应文档，并通过 `ci_gate.py` 设计固定项检查。

---

## 1. 研究问题与假设体系

### 1.1 核心研究问题

1. **RQ1（门控新颖性）**：门控注意力是否只是已有门控方案（Gated Attention 家族）的微小变体？
2. **RQ2（深层协同）**：深层残差流贡献不足时，门控与归一化应如何协同设计？

### 1.2 假设层级

| 层级 | 编号范围 | 性质 | 实验 |
|------|----------|------|------|
| 基线前提 | HB0–HB7 | 管道有效性/任务合理性/复杂度控制 | baselines |
| 机制核心 | H1a–H1h | 校准门控的注意力分布使用、错误抑制、消融、安慰剂、位置混淆控制 | E1 |
| 泛化机制 | H2a–H2g | 长度校准的显式修正、系数正向性、归一化负载 | E2 |
| 协同设计 | H3a–H3c | 动态归一化深层写入、LayerScale 交互、校准×归一化交互 | E3 |

### 1.3 预注册决策规则统一表

| 假设 | 指标族 | FDR 家族 | SESOI | 方向 | 判定规则 |
|------|--------|----------|-------|------|----------|
| HB1 | accuracy | baselines | 0.01 | — | rule_acc < 0.99 → invalidates all |
| HB2 | probe_margin | baselines | — | — | CI upper < +chance (certified) |
| HB3 | overlap/shift | baselines | — | — | any shared sample or adj. p < FDR → invalidates all |
| HB4 | control_acc | baselines | 0.02 (tol) | — | CI upper < +tol |
| HB5 | train_loss/acc | baselines | — | — | loss ≤ 0.1 & acc = 1.0 |
| HB6 | 1L_acc vs ceiling | baselines | 0.01 | — | 95% upper bound ≥ ceiling - SESOI (depth measurably required) |
| HB7 | saturation_depth | baselines | 0.01 | — | d* > 2 & HB0 competent |
| H1a | \|gate_vs_entropy\|, \|gate_vs_maxprob\|, accuracy | E1 | 0.05, 0.05, 0.01 | +, +, noninferior | paired bootstrap CI ≠ 0 after BH (q=0.05); accuracy CI lower ≥ -SESOI |
| H1b | gate_margin | E1 | 0.02 | + | paired bootstrap CI ≠ 0 after BH |
| H1c | null_mass_error_margin, null_mass_false_conc_margin, accuracy(null-none) | E1 | 0.02, 0.02, 0.01 | +, +, noninferior | paired bootstrap CI ≠ 0 after BH; null-none CI lower ≥ -SESOI |
| H1d | accuracy, gate_margin, \|gate_vs_entropy\| (ablations) | E1 | 0.01, 0.02, 0.05 | +, +, + | each single-term ablation CI ≠ 0 after BH; interaction two-sided ≥ SESOI |
| H1e | prauc_lift, recall@10% | E1 | 0.1, 0.02 | +, + | paired bootstrap CI ≠ 0 after BH |
| H1f | accuracy, gate_margin, \|gate_vs_entropy\| (calib vs placebo) | E1 | 0.01, 0.02, 0.05 | +, +, + | paired bootstrap CI ≠ 0 after BH |
| H1g | calib_abs_*, query_abs_* | E1 | 0.05, 0.05 / 0.10, 0.10 | +, + / equivalence | calib: CI ≠ 0 & ≥ SESOI; query: CI inside ±bound (TOST) |
| H1h | accuracy (calib_pos_norm − calib), \|pos_norm_abs_gate_vs_entropy\| | E1 | 0.01, 0.05 | noninferior, + | 非劣性：95% CI 下界 ≥ −SESOI（位置中性臂不显著更差）；机制行：单样本 CI ≠ 0 且 ≥ SESOI |
| H2a | accuracy_at_longest, accuracy_drop, accuracy_slope_log_len | E2 | 0.01, 0.01, 0.01 | +, -, + | paired bootstrap CI ≠ 0 after BH |
| H2b | entropy_drift, entropy_norm_drift, gate_drift | E2 | 0.1, 0.05, 0.02 | + / equivalence, equivalence | raw: CI ≠ 0 & >0; norm/gate: CI inside ±bound (TOST) |
| H2c | accuracy_at_longest, accuracy_drop (calib vs ent_only); calib vs len_only (reverse) | E2 | 0.01, 0.01 / 0.01, 0.01 | +, - / equivalence | calib-ent_only CI ≠ 0; calib-len_only CI inside ±bound (TOST) |
| H2d | accuracy_at_longest, accuracy_drop, accuracy_slope_log_len (calib vs query) | E2 | 0.01, 0.01, 0.01 | +, -, + | paired bootstrap CI ≠ 0 after BH |
| H2e | concentration main effect, interaction | E2 | 0.01, 0.01 | +, - / two-sided | main: CI ≠ 0 after BH; interaction: two-sided ≥ SESOI |
| H2f | a_mean, c_effective_mean | E2 | 0.01, 0.01 | +, + | one-sample bootstrap CI ≠ 0 after BH |
| H2g | accuracy_at_longest, accuracy_drop (calib vs raw_ent) | E2 | 0.01, 0.01 | +, - | paired bootstrap CI ≠ 0 after BH |
| H3a | gate_depth_ratio, resid_norm_last_over_first, accuracy | E3 | 0.05, 0.05, 0.01 | +, +, noninferior | paired bootstrap CI ≠ 0 after BH; accuracy CI lower ≥ -SESOI |
| H3b | resid_norm_last_over_first, gate_depth_ratio | E3 | 0.05, 0.05 | primary/secondary | primary CI ≠ 0 after BH |
| H3c | accuracy, gate_depth_ratio (DiD: seednorm vs pre) | E3 | 0.01, 0.05 | +, + | paired bootstrap CI ≠ 0 after BH |

---

## 2. 实验设计细节

### 2.1 任务：联想回忆

**格式**：`[k1 v1 k2 v2 ... kn vn, SEP, qi, <predict vi>]`
- vocab: `0..V-1`，其中 `0=PAD`, `1=SEP`, `2..V-1` 为内容 token
- 每序列 `n_pairs` 个键值对，键在序列内唯一
- 监督位置：query token 后的单一位置（预测对应的 value）

**数据流分离**（构造时固定，不可更改）：
| 流 | 种子 | 用途 | 保护 |
|----|------|------|------|
| train | 用户种子集 | 训练批次生成 | 拒绝 VAL/TEST/STATS 种子 |
| val | `VAL_SEED=123` | 学习率/配方选择 | 仅选择，不评估 |
| test | `TEST_SEED=2024` | 最终报告数字、预注册门控规则 | 封存，从不用于选择 |
| stats | `STATS_SEED=7` | 机制统计（熵、门控、错误挖掘） | 固定，与 test 独立 |

**HB3 分裂审计**：训练流与每个保留流的精确重复重叠=0；分布偏移卡方检验 BH 校正后 p ≥ FDR。

### 2.2 E1：门控对比（4 变体 + 消融 + 安慰剂 + 位置控制）

| 变体 | gate | 关键配置 | 说明 |
|------|------|----------|------|
| none | `none` | — | 标准注意力（无门控） |
| query | `query` | — | 纯 query 依赖门控：`σ(W_g x)` |
| calib | `calib` | 默认全开 | 校准门控：`σ((W_g x − a·log T − c·H̄)/τ)` |
| null | `null` | — | 显式 null KV 槽位竞争 |
| calib_len_only | `calib` | `calib_use_len=True, calib_use_ent=False` | 仅长度项 |
| calib_ent_only | `calib` | `calib_use_len=False, calib_use_ent=True` | 仅集中度项 |
| calib_no_terms | `calib` | `calib_use_len=False, calib_use_ent=False` | 两项全关（退化为 query 偏置） |
| calib_shuf_ent | `calib` | `calib_shuffle_ent=True` | **安慰剂**：熵位置固定打乱，架构/参数/计算/边缘分布不变 |
| calib_pos_norm | `calib` | `calib_pos_norm=True` | **位置中性对照**：除数与长度项改用每行可见 token 数 `log(max(t+1,2))`，去掉全局 `log(T)` 混入的位置开放度调度 |

**位置混淆消融（H1h）**：注册门控对**每一行**用全局 `log(T)` 归一化注意力熵并减去 `a·log(T)`，而因果行 t 最多只能看到 t+1 个键（其熵上限为 `log(t+1)`）。因此同一份注意力分布在不同位置会被读成不同的开放度——这是集中度信号里混入的位置信息。`calib_pos_norm` 把除数与长度项换成每行自己的可见 token 数：架构、参数、计算与配方不变（单因素 `calib_pos_norm`），且在最终 query 行与注册臂**数值相同**（`log(max(T,2)) = log T`），差异只出现在中间行的门控上。判定为**单侧非劣性**（位置中性臂不显著更差 ⇒ 优势不依赖位置调度；若显著更差，则"读集中度"的主张必须条件化表述）+ 机制行（位置中性臂自身仍读出注意力熵关联，排除"退化臂"对照）。

**共享调度**：上述 9 个 cell 进入**同一个**随机化完全区组设计（每种子一个区组，每区组 9 个 cell 位置随机排列），避免消融对比被机器漂移混淆。

**网格配置**（`Settings` 默认值，可通过 CLI 覆盖）：
```python
# 共享冻结配方：在 none 上用 VAL_SEED 从 lr_grid 调得后冻结（lr_grid=(1e-3, 3e-4, 1e-4)）
steps=1500, batch=64, n_pairs=16, vocab=512
d_model=256, n_layers=8, n_heads=4, max_len=256
dropout=0.0, norm=pre, layer_scale=None
# calib 专用
calib_a=1.0, calib_c=1.0, calib_scale=1.0, calib_tau_init=0.9
# 其余运行参数
seeds=(0,1,2,3,4), fdr_q=0.05, min_seeds_for_claim=3
design_alpha=0.05, power_target=0.8, run_order_seed=20240517
```

**读出指标**（每层 Fisher z 平均后的 Pearson 相关）：
- `gate_vs_entropy`：门控值 vs 归一化注意力熵
- `gate_vs_maxprob`：门控值 vs 最大注意力概率
- `false_conc_rate`：高 max-prob (>0.5) 样本中的错误率
- `gate_on_false_conc` / `gate_on_correct`：错误集中 vs 正确集中的门控均值
- `gate_margin = gate_on_correct - gate_on_false_conc`
- `null_mass`：null 变体中 null 槽位分走的注意力权重占比
- `calib_coefficients`：学得的 `a_mean`、`c_effective_mean` 等

### 2.3 E2：长度泛化

**训练长度**：固定 `n_pairs=16`（序列长度 ~34）

**扫描长度（请求网格 vs 实际执行网格）**：
- **请求网格** = `length_sweep_lengths(base=16, factor=2, n=4, max_len=256)`
  的几何序列 `16 · 2^i`（i < 4），即 `[16, 32, 64, 128]`。
- **实际执行网格** = `[16, 32, 64]`。长度 `L` 需要 `2L+2` 个序列位置；
  `L=128` 需要 258 个位置，超过注册的 `max_len=256`（模型的位置嵌入表
  没有这些位置，任何在该长度上的前向都会索引越界），因此该长度以及
  更长的长度一律丢弃。**丢弃是显式记录、不是隐式截断**：
  `cage.data.length_sweep_plan` 返回请求网格、实际网格、被丢弃的长度、
  它需要的序列位置与原因；`e2_summary.json` / `e2_runs.jsonl` /
  `e2_rule_ceiling.json` 都携带该计划，实际网格与请求网格不一致时
  运行器会在控制台打印截断说明。
- **实际覆盖范围** = 34 → 130 个序列位置（3.82×；键值对 16 → 64，4×）。
  所有长度泛化结论必须对照这个**实际测量范围**读取，不能对照请求网格。
- **最少实际长度 = 3**（`Settings.e2_sweep_min_lengths`，`length_sweep_plan`
  的 `min_lengths`）。理由：H2a 同时读取 `accuracy_drop` 与
  `accuracy_slope_log_len`，后者存在的意义正是发现"中段反转"——端点
  下降不变但中段方向相反的情形；只有 2 个实际长度时，斜率只是端点差的
  线性缩放、没有任何内部点，中段反转在结构上不可检测。因此无法给出
  3 个实际长度的几何在开跑前直接拒绝，而不是产出一条无法回答问题的
  斜率读数。
- **执行几何的唯一来源**是 `cage.experiments._e2_length_sweep_plan`：
  E2 运行器与评估答案均衡审计共用同一份计划，两份记录不可能对"实际
  跑了哪些长度"产生分歧。

<!-- e2-sweep-contract
base=16
factor=2
n=4
max_len=256
min_lengths=3
realized=16,32,64
-->
> 上面的 `e2-sweep-contract` 注释块是**可机检**的预注册记录：
> `tests/test_length_sweep_contract.py` 用默认 `Settings` 重新计算
> `length_sweep_plan` 并逐项比对，协议文档与代码一旦对"实际跑了哪些
> 长度"产生分歧，测试立即失败。

**臂**（8 臂完整 2×2×2 设计）：
| 臂 | gate | calib_use_len | calib_use_ent | calib_ent_norm | 说明 |
|----|------|---------------|---------------|----------------|------|
| none | none | — | — | — | 基线 |
| query | query | — | — | — | query 门控基线 |
| calib | calib | True | True | True | **完整校准门控** |
| calib_len_only | calib | True | False | True | 仅长度项 |
| calib_ent_only | calib | False | True | True | 仅集中度项 |
| calib_no_terms | calib | False | False | True | 两项全关 |
| calib_raw_ent | calib | True | True | **False** | **归一化消融**：读原始行均值熵而非 H/log T |
| null | null | — | — | — | 显式 null KV 槽位基线（进入逐变体公平性伴随，不进入预注册决策行） |

> **臂表说明（v1.2 修正）**：本表原先把 E1 的安慰剂臂 `calib_shuf_ent` 列在 E2 中，
> 但 E2 的预注册决策规则（§1.3 H2a–H2g）没有任何安慰剂对比，代码实现里 E2 也没有该
> 臂；E2 的实际第 8 臂是 `null`（`cage/experiments/_e2.py:_E2_VARIANTS`）。此处按实现
> 修正文档，不新增/删除任何臂、对比对、假设或判定规则。

**位置编码外推控制**（探索性伴随，同一封闭评估流）：
- `fold`：外推位置折叠回训练区间（模 训练最大位置）
- `zero`：外推位置零化
- 若 fold 能恢复大部分退化 → 退化主要来自位置外推而非机制

**剂量/反向干预**（探索性伴随）：
- 将学得系数缩放为 `0 / 0.5 / 1 / 2 / -1`，检查配对准确率响应
- 用于区分"有符号的标定机制在起作用" vs "任何门控扰动都会同样伤害"

### 2.4 E3：归一化/激活协同

**网格**：`gate ∈ {query, calib} × norm ∈ {pre, post, mix, seednorm} × LayerScale ∈ {None, 1e-2}` = 16 个 cell

**读出**：
- 逐层残差范数 `resid_norm`（注意：post-norm 的是 LN 后范数，pre/seednorm 是原始残差流）
- `gate_depth_ratio` = 深层门控均值 / 浅层门控均值（层索引按深度分半）
- `attn_write_ratio` / `mlp_write_ratio`：各分支相对其写入流的范数比（跨归一化方案可比）
- `layer_knockout_eval`：逐层替换为恒等后测准确率（因果性证据）
- LayerScale 的 `ls1`/`ls2` 学得值分别读出

**跨归一化比较规则**：
- **禁止**直接比较 `resid_norm`（量级被归一化钉住 vs 随深度增长）
- **必须**用 `attn_write_ratio` / `mlp_write_ratio` + `layer_knockout_eval`

---

## 3. 统计推断协议

### 3.1 种子级配对 Bootstrap

```python
# cage.metrics.paired_bootstrap_ci
# 输入：两臂在相同种子下的指标值列表（长度相同，按种子配对）
# 输出：mean_diff, 95% CI [lo, hi], p_value (percentile), seeds_used, seeds_dropped
# 并行：BCa 区间、Jackknife 诊断、精确配对符号翻转检验
```

**配对原则**：
- 同一实验内的所有对比**必须**按种子配对
- Curve-invalid 运行（plateau/diverging/numerical）从配对中排除，记录 `seeds_dropped`
- 最少配对种子数：`min_seeds_for_claim=3`（Settings 可配置）

### 3.2 多重比较校正

- **家族定义**：每个实验（E1/E2/E3/baselines）的所有预注册对比构成一个 FDR 家族
- **方法**：Benjamini-Hochberg (q=0.05，可通过 `--fdr-q` 覆盖)
- **BY 伴随**：任意依赖下的 FCR 区间使用 Benjamini-Yekutieli 调和因子分母

### 3.3 SESOI 与实质显著性

| 判定 | 条件 |
|------|------|
| **supported** | FDR 显著 + 方向一致 + 效应量 ≥ SESOI |
| **refuted** | FDR 显著 + 方向相反（无论效应量） |
| **practically negligible** | FDR 显著 + 方向一致 + 效应量 < SESOI |
| **inconclusive** | FDR 不显著（CI 包含 0）或 CI 宽度跨越 SESOI 边界 |
| **ineligible** | 配对种子数 < `min_seeds_for_claim` |

**等价性/非劣效性检验**（`sesoi_test` 注册为 `"equivalence"` 或 `"noninferior"`）：
- 使用 `cage.metrics.equivalence_test` (bootstrap TOST) 或 `noninferiority_test`
- **仅当 95% CI 完全落在 ±SESOI 内（或单侧下界 ≥ -SESOI）才判定 supported**
- CI 仅包含 0 而跨越界 → `inconclusive`（非证据）

### 3.4 功效与设计计划

设计计划在首次运行前生成，包含：
- 将 SESOI 换算为计划 N 能容忍的最大种子级 SD
- 报告**配对差异方差**（而非单臂方差）
- 有限评估流的二项式下界（更多训练种子无法消除）
- Curve-invalid 率损耗修正
- 精确配对符号翻转检验的可达 p-value 下界（5 种子双侧最小 0.0625，无法在 q=0.05 确认）
- 计划冻结后修改种子/SESOI/FDR 抛出 `DesignPlanError`

### 3.5 单因素对照的设计期证书

预注册的受控对比必须是**单因素**实验：两条臂之间只允许有**一个**解析后的模型因子不同（`gate`/`norm`/`layer_scale`/`calib_use_len`/`calib_use_ent`/`calib_shuffle_ent`/`calib_ent_norm`/`calib_pos_norm`）。

- **设计期（开跑前）**：每个实验用**解析后的 `Cfg`** 计算各实验单元的因子向量，并声明它的受控对比对（E1 8 对、E2 5 对、E3 20 对），逐对检查"恰好一个因子不同"；证书写入 `<tag>_single_factor_design.json`（注册进制品索引），控制台打印摘要。
- **失败即停**：任一声明的对比对变化了多于一个因子时抛出 `ValueError`，网格**不启动**——多因子操作无法读作单因素结论，先跑再拒只会白花算力。
- **运行期防漂移**：决策行必须从证书声明的对比对里取臂（`_design_pair`）；未声明的对比对会中止运行，因此"实际执行的对比"与"认证过的设计"始终是同一集合。
- **因子格角落**：`calib_no_terms` 同时关闭两个校准项，是**双因子**单元，只通过已注册的 2×2 主效应/交互（聚合行）读取，**不得**作为单因素决策行；证书把它列在 `cells_without_one_factor_contrast` 中。
- **非受控探索对比**：安慰剂臂与 `query` 的对比同时改变 `gate` 与 `calib_shuffle_ent`，显式记入证书的 `exploratory_pairs`（`controlled: false`），只在决策日志之外打印。
- **与行级审计并存**：设计期证书不替代逐行的受控变量审计（`varying_factors` / `declared_label_diffs`）；两处都必须通过。

### 3.6 研究级分析计划（跨家族多重性）

E1/E2/E3 各自控制一个 FDR 家族，但**这些保证不复合**：三个家族各自在 q=0.05 上做 BH，
并不意味着它们发现的并集是 FDR-0.05 集合。研究的头条主张同时在多个家族里被检验
（校准门控在 E1 **和** E2 中都与 query 门控对比），所以"在自家家族里成立"这句话
会让读者读到一个从未被控制过的研究级错误率。

因此研究在**开跑前**注册一份研究级分析计划（`cage/experiments/_analysis_plan.py`）：

- **确认性对比集**（`_PRIMARY_CONTRASTS`，10 条腿 P1–P10）：每条腿用
  `(experiment, hypothesis, metric, rule)` 三元组标识运行器**实际产出的**那一行，
  并显式写出它对比的网格单元 `arms`（如 P1 = `("calib", "query")`）与
  `kind`（`one_factor` 对比两个单元 / `interaction` 对比两个差之差，共四个单元）。
  其中 **P10** 是 H1h 的位置中性腿：`arms = ("calib_pos_norm", "calib")`，
  `rule = "calib_pos_norm - calib"`，`kind = one_factor`（顺序即差值方向，
  与单因素证书的声明一致），并入声明 C1。
- **研究级声明**（`_STUDY_CLAIMS`，C1–C4）：每条声明是其各腿的**合取**
  （intersection-union, IUT）。IUT 下拒绝一个合取需要拒绝它的**每一条**腿，
  因此声明之间不需要额外校正；声明的全局 p 值是其各腿单侧 p 值的最大值。
- **注册表审计**：每条腿的假设、实验归属、指标 SESOI、方向/零预测检验方式
  必须在 `_config.py:HYPOTHESES` 中存在，否则计划无法发布。
- **可达性审计**（`_primary_contrast_reachability`）：每条腿的 `arms` 必须真的
  是它所属家族的网格单元（`_e1_cell_plan` / `_E2_VARIANTS` / E3 网格）。
  不可达的腿**永远无法被测量**——研究会先花掉算力，再以"不可裁决"收场，
  而原因在开跑前就写在源码里。不可达即**拒绝开跑**（`DesignPlanError`）。
- **研究级单因素证书**（`_primary_contrast_single_factor`）：§3.5 的家族级证书
  只认证家族**自己声明**的对比对，而确认性对比集是**第二份**注册。证书把每条腿
  的 `arms` 解析回其家族的单因素细胞配置并逐条认证：`one_factor` 腿必须恰好变动
  **一个**解析后的模型因子，且必须是该家族已声明的受控对比对（方向也由声明固定）；
  `interaction` 腿必须是两个**同一**单因子边际之差再跨两个**同一**其他因子上下文的
  差之差（即运行时 `_paired_did` 在训练后强制的同一结构），四条边也都必须是已声明
  的受控对比对。任一腿不满足即**拒绝开跑**（`DesignPlanError`）：一条会变动两个因子
  的确认性腿不能被读作单因素主张，先跑再拒只会白花算力。
- **冻结**：`analysis_plan.json` 写入内容哈希（`analysis_plan_sha256`），
  **任何家族产出结果行之后**修改确认性对比集、其多重性水平或计划种子数
  都会抛出 `DesignPlanError`（那是看过数据之后重选确认性对比 = 事后重设计）。
  开跑前的修订保留在 `analysis_plan_history.jsonl`。
- **研究级功效**（`_primary_contrast_power`）：单家族预算通过**不等于**研究级
  主张可裁决——合并后的确认性家族更大，BH 的最坏情况阈值是 `q/m`。计划同时报告
  `claim_level`（未膨胀的 q）与 `pooled_family`（q/m）两个阈值下的种子数需求与功效。
  只有 accuracy 量纲的腿有经验 pilot 方差，其余腿报 `no_pilot` 而不是编造种子数。
- **裁决**（`_study_level_adjudication` → `study_level_adjudication.json`）：
  读取三个家族的 `<tag>_contrasts.json`，报告每条腿的家族内与合并族判定、
  每条声明的 IUT 判定、**家族缺失/腿缺失/腿歧义/计划与注册表不一致**等偏离，
  以及研究级状态。缺家族的运行报 **incomplete**，不得当作完整研究汇报。
  在自家家族成立但在合并族不成立的腿/声明分别记为 `family_inflated`。
- **不合格的腿不能裁决声明**：配对种子数低于 `min_seeds_for_claim`、曲线失效、
  评估流不平衡或非受控对比的行，其 `claim_eligible=false`，因此既不能支持
  也不能**反驳**一条声明（与 §5.3 假设综合的约定一致）。

**注册与刷新时机**：E1、E2、E3 **每个家族**在首次训练前都发布/复核这份计划
（因此"先跑 E2"与"先跑 E1"等价），并在各自结束时刷新研究级裁决；
`--only all` 在三个家族之后最后再写一次，因此完整运行留下的
`study_level_adjudication.json` 是**完整研究**的读数，而不是第一个家族结束时的部分读数。

**研究级设计预检**（`_study_design_preflight` → `study_design_preflight.json`）：
在开跑前把注册表一致性、可达性、合并族功效预算和**算力预算**（由 `Settings`
与网格计划推导出的运行数与解析 FLOPs）汇总成一份准备证书。算力预算取代
手写的运行次数：提案里引用的运行数应当来自这份制品，而不是一个人工抄写的数字。

---

## 4. 基线与合理性门禁（运行前必须通过）

| 门禁 | 检查内容 | 失败后果 |
|------|----------|----------|
| HB1 | 规则准确率 ~1.0（Wilson 95% CI） | < 0.99 → 任务构造失效，整个实验无效 |
| HB2 | 线性探针在调优后仍接近 chance（bootstrap CI 上界 < +chance） | CI 含 +chance → "inconclusive"；CI 全在 +chance 以上 → "violated"，答案泄露 |
| HB3 | 训练/验证/测试/统计流无重叠、无分布偏移（BH 调整） | 任一失败 → 所有报告数字无效 |
| HB4 | 负控制：标签打乱训练后测试流准确率在 chance 附近（CI 上界 < +0.02） | 失败 → 答案信息泄露，所有准确率无效 |
| HB5 | 微小批次（dropout=0）能否将训练损失压至 ~0、批次准确率=1.0 | 失败 → 训练循环/梯度/精度路径坏掉 |
| HB6 | 1 层标准 Transformer 是否已饱和（95% 上界在 ceiling - SESOI 内） | 饱和 → E3 深层动机需标注 "1层已足够" |
| HB7 | 阶梯饱和深度 d*（最小胜任深度且 95% 下界在 ceiling - SESOI 内） | d* ≤ 2 → E3 深层动机需标注 "2层已饱和"；无胜任层 → "not determinable" |

**基线裁决输出**：`baselines_verdict.json`（含 rule ceiling Wilson CI、probe bootstrap CI、split audit、HB4/HB5/HB6/HB7 详细读数）。E1/E2/E3 运行前自动读取并作为正向/反向对照。

---

## 5. 数据记录与制品规范

### 5.1 运行记录必含字段

每条 `*_runs.jsonl` 行必须包含（`_RECORD_PROVENANCE_KEYS`）：

```json
{
  "run_id": "uuid4",
  "schema_version": 1,
  "setup_sha256": "16字符内容哈希",
  "git_sha": "完整提交哈希或 unknown",
  "source_tree_sha256": "16字符源码树哈希",
  "hypotheses_sha256": "16字符假设注册表哈希",
  "config": { ...完整配置... },
  "data_fingerprints": { "train": "...", "val": "...", "test": "...", "stats": "..." },
  "environment": { ...完整环境块... },
  "seed": 42,
  "lr": 3e-4,
  "gate": "calib",
  "accuracy": 0.87,
  "abs_gate_vs_entropy": 0.42,
  "curve_verdict": "healthy",
  "created_utc": "2026-11-03T12:34:56Z"
}
```

### 5.2 制品索引（`artifact_index.json`）

累积索引，每次调用更新：
```json
{
  "experiment": "e1",
  "completed": true,
  "elapsed_seconds": 123.456,
  "artifact_count": 12,
  "current_artifact_count": 3,
  "updated_entries": ["e1_runs.jsonl", "e1_error_cases.jsonl", "e1_manifest.json"],
  "artifacts": [
    {"file": "e1_runs.jsonl", "bytes": 45678, "sha256": "...", "records": 40},
    {"file": "environment.json", "bytes": 2345, "sha256": "..."}
  ],
  "run_id": "...", "git_sha": "...", "setup_sha256": "...", "source_tree_sha256": "..."
}
```

### 5.3 运行生命周期（`run_index.jsonl`）

```jsonl
{"run_id": "...", "event": "started", "experiment": "e1", "git_sha": "...", "setup_sha256": "...", "created_utc": "..."}
{"run_id": "...", "event": "completed", "experiment": "e1", "reason": "all cells finished", "created_utc": "..."}
```
终止事件：`completed` / `aborted` / `failed`。**只有 completed 可被引用为证据**。

### 5.4 环境历史（`environment_history.jsonl`）

```jsonl
{"run_id": "...", "compatible": true, "identical": true, "differences": [], "previous_environment_sha256": "...", "current_environment_sha256": "..."}
{"run_id": "...", "compatible": false, "differences": [{"field": "torch", "severity": "critical", "previous": "2.3.0", "current": "2.4.0"}]}
```

### 5.5 研究级制品（跨家族）

| 制品 | 写入时机 | 内容 |
|------|----------|------|
| `study_design_preflight.json` | 每个家族开跑前 | 注册表一致性 + 腿可达性 + 研究级单因素证书（每条确认性腿的因子变动）+ 合并族功效预算 + 算力预算（运行数、解析 FLOPs） |
| `analysis_plan.json` | 每个家族开跑前 | 预注册研究级分析计划：确认性对比集 P1–P9、声明 C1–C4、合并 FDR 水平、计划种子数、内容哈希 |
| `analysis_plan_history.jsonl` | 计划创建/修订时 | 每次计划事件的追加式历史（`created` / `amended` / `replaced_malformed`） |
| `study_level_adjudication.json` | 每个家族结束时（`--only all` 结束后再写一次） | 每条腿的家族内/合并族判定、每条声明的 IUT 判定、研究级状态与偏离清单 |

`analysis_plan.json` 的哈希覆盖**设计相关**部分（对比集、声明、合并 FDR 水平、
声明门槛、计划种子数、假设注册表哈希），不覆盖每次调用都变的溯源信息。

---

## 6. 复现与验证检查表

### 6.1 发布前自检

- [ ] `make preflight` 通过（依赖一致性 + 语法 + 设计固定项 + 冒烟 + 全测试）
- [ ] 正式实验使用非 `--quick` 设置、足够种子数（≥5）
- [ ] 结果目录完整保留（未手动删除/修改任何文件）
- [ ] `make verify OUT=results` 全部通过
- [ ] `baselines_verdict.json` 所有门禁通过
- [ ] 设计计划冻结无冲突（`_design_plan_history.jsonl` 无错误）
- [ ] 研究级分析计划已注册且冻结（`analysis_plan.json` 存在，`analysis_plan_history.jsonl` 无意外修订）
- [ ] `study_design_preflight.json` 中无不可达腿，功效预算已复核
- [ ] `study_level_adjudication.json` 的 `study_complete=true`（三个家族齐备），否则报告必须写明"研究不完整"

### 6.2 独立复现核对

- [ ] 使用 `requirements-ci.txt` 精确依赖版本
- [ ] 依赖一致性 `verify-deps` 通过（实装版本 == 声明的固定版本）
- [ ] 环境指纹 `verify-env` 通过（critical 字段一致）
- [ ] 制品完整性 `verify-artifacts` 通过
- [ ] 来源溯源 `verify-provenance` 通过
- [ ] 运行完整性 `verify-runs` 通过（无 aborted/failed，环境同质）
- [ ] 源码一致性 `verify-source` 通过
- [ ] 可选：容器化运行（Dockerfile.repro）跨机器验证

### 6.3 重放验证

- [ ] `python -m cage.experiments --replay-manifest results/e1_manifest.json --out results_replay`
- [ ] `make verify-replay SOURCE=results REPLAY_OUT=results_replay` 返回 0
- [ ] 数值差异在浮点非确定性可接受范围内（无系统性偏移）

---

## 7. 结果报告模板

### 7.1 对比表格（每个预注册对比）

| 对比 | 指标 | 均值差 | 95% CI | p值 | seeds_used/dropped | FDR | SESOI判定 | 状态 |
|------|------|--------|--------|-----|-------------------|-----|-----------|------|
| calib - query | accuracy | +0.032 | [+0.011, +0.054] | 0.008 | 5/0 | ✓ | >0.01 | supported |
| calib - query | \|gate_vs_entropy\| | +0.18 | [+0.09, +0.27] | 0.002 | 5/0 | ✓ | >0.05 | supported |

### 7.2 假设综合表

| 假设 | 注册行 | 全部 supported | 部分 supported | 有 refuted | 综合判定 |
|------|--------|----------------|----------------|------------|----------|
| H1a | 3 | ✓ | — | — | **supported** |
| H1d | 6 | — | 4/6 | — | **partial** |
| H2f | 2 | ✓ | — | — | **supported** |

### 7.3 伴读表（不改变行级裁决）

| 伴读 | 说明 |
|------|------|
| 复制区间 | 未来同量级研究均值的 t 区间（sqrt(2/n)），保持符号概率 |
| FCR 区间 | BH 选中集合的假覆盖率区间（BY 分母） |
| 功效分析 | 观测方差下达到 80% 功效所需种子数 |

### 7.4 研究级声明表（§3.6）

| 声明 | 腿 | 家族内 IUT | 合并族 IUT | 家族膨胀 | 说明 |
|------|----|-----------|-----------|---------|------|
| C1 | P1, P2, P3, P10 | supported | supported | 否 | 校准门控确实读取注意力分布（且不依赖全局归一化混入的位置调度） |
| C2 | P4, P5 | partial | inconclusive | 是 | 读分布是否改变门控行为 |
| C3 | P6, P7 | inconclusive | inconclusive | 否 | 长度鲁棒性是否超出 query 门控 |
| C4 | P8, P9 | supported | partial | 是 | 门控与动态归一化是否协同 |

> 表中数值为**格式示例**，不是结果。真实读数见 `study_level_adjudication.json`；
> 研究级状态为 `incomplete`（有家族缺失）或任一声明的合并族读数为非 supported 时，
> 不得把研究写成"结论成立"。

---

## 8. 变更控制

### 8.1 协议修改流程

1. 在 `docs/design_notes.md` 修改对应 `【固定】` 条目
2. 同步更新 `cage/experiments/_config.py:HYPOTHESES`（假设注册表）
3. 若改动影响**确认性对比集或研究级声明**，同步更新
   `cage/experiments/_analysis_plan.py:_PRIMARY_CONTRASTS`/`_STUDY_CLAIMS`
   （两者的内容哈希进入 `analysis_plan.json`，改动会使既有结果目录的
   `--verify-provenance` 失效，这是预期行为）
4. 同步更新本文档 `EXPERIMENT_PROTOCOL.md`
5. 运行 `make check` 确保 CI 门禁通过
6. 提交并移动固定标签：`git tag -f design-pin <新提交sha> && git push origin design-pin`

> **注（v1.5 的执行偏差）**：`docs/design_notes.md` 与 `ci_gate.py` 由 CI 以
> `design-pin` 标签为基准自动还原（改动会被丢弃），因此本次 H1h 注册**只能**落在
> `_config.py`、`_analysis_plan.py` 与本文档三处；第 1、6 步（写 `design_notes.md`
> 并移动标签）需要具备推送权限的人执行。在标签移动之前，H1h 的登记以本文档
> §1.3/§2.2/§3.6 与 `_config.py:HYPOTHESES["H1h"]` 为准，`design_notes.md`
> 仍是 v1.4 的固定条目——这一偏差是**显式记录**的，而不是静默跳过。

### 8.2 版本兼容性

| 协议版本 | 兼容结果目录 | 说明 |
|----------|--------------|------|
| 1.0 | ≥ v1.0 | 向前兼容：旧结果可读、可验证、可重放 |
| 1.x | ≥ v1.x | 次版本仅增量添加字段/伴读，不改变决策规则 |

### 8.3 弃用与移除

- 移除任何预注册假设/指标/对比需：同步四文档 + 移动 design-pin 标签 + 标记为 `deprecated` 而非直接删除（保留历史记录可读性）
- 移除研究级确认性对比（P*）或声明（C*）需：同时更新 `_analysis_plan.py` 与本节 §3.6，
  并在 `analysis_plan_history.jsonl` 留痕；**结果产出之后**的移除会被
  `DesignPlanError` 拒绝
- 移除手写 autograd 核需：通过 gradcheck + 参考实现对照 + 多块一致性 + 训练/评估路径一致性四重验证

---

## 附录 A：CLI 参数速查

```bash
# 核心运行
python -m cage.experiments --only {sanity,baselines,e1,e2,e3,all} --seeds N --out DIR
python -m cage.experiments --only e1 --seeds 5 --out results --fdr-q 0.05 --deterministic

# 设计与调试
python -m cage.experiments --only e1 --quick --out results_quick     # 流水线检查
python -m cage.experiments --only e1 --per-variant-tuning            # 逐变体调参（探索性）
# lr 敏感性是 E1 网格的固定伴随读出，随 E1 自动产出（e1_lr_sensitivity.json），
# 没有单独的开关。

# 验证
python -m cage.experiments --verify-artifacts --out results
python -m cage.experiments --check-environment --out results
python -m cage.experiments --verify-provenance --out results
python -m cage.experiments --verify-runs --out results
python -m cage.experiments --check-source --out results
python -m cage.experiments --check-dependencies --out results
python -m cage.experiments --verify-replay results --out results_replay

# 重放
python -m cage.experiments --replay-manifest results/e1_manifest.json --out results_replay

# Makefile 封装
make check | gate | test | sanity | verify | verify-replay SOURCE=results
make verify-deps | preflight
```

---

## 附录 B：环境变量约定

| 变量 | 默认 | 说明 |
|------|------|------|
| `OMP_NUM_THREADS` | 1 | OpenMP 线程数（记录在 numeric_env） |
| `MKL_NUM_THREADS` | 1 | MKL 线程数 |
| `OPENBLAS_NUM_THREADS` | 1 | OpenBLAS 线程数 |
| `NUMEXPR_NUM_THREADS` | 1 | NumExpr 线程数 |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | 确定性 CUDA 时设置 |
| `PYTHONHASHSEED` | 运行种子 | `seed_everything` 自动导出并记录 |

---

## 附录 C：文件命名约定

| 模式 | 含义 | 示例 |
|------|------|------|
| `{tag}_runs.jsonl` | 主结果记录 | `e1_runs.jsonl` |
| `{tag}_error_cases.jsonl` | 错误案例（含注意力分布） | `e1_error_cases.jsonl` |
| `{tag}_design_plan.json` | 冻结设计计划 | `e1_design_plan.json` |
| `{tag}_design_plan_history.jsonl` | 计划修订历史 | `e1_design_plan_history.jsonl` |
| `{tag}_run_order.json` | 计划调度 | `e1_run_order.json` |
| `{tag}_run_order_realized.json` | 实测调度+位置效应 | `e1_run_order_realized.json` |
| `{tag}_manifest.json` | 重放清单 | `e1_manifest.json` |
| `{tag}_per_variant_tuning.json` | 逐变体调参伴读 | `e1_per_variant_tuning.json` |
| `{tag}_lr_sensitivity.json` | lr 敏感性伴读 | `e1_lr_sensitivity.json` |
| `study_design_preflight.json` | 研究级设计预检（可达性 + 功效 + 算力） | — |
| `analysis_plan.json` | 预注册研究级分析计划 | — |
| `analysis_plan_history.jsonl` | 研究级计划修订历史 | — |
| `study_level_adjudication.json` | 研究级裁决（IUT + 合并族） | — |
| `baselines_verdict.json` | 基线裁决 | — |
| `environment.json` | 当前环境记录 | — |
| `environment_history.jsonl` | 环境漂移历史 | — |
| `artifact_index.json` | 累积制品索引 | — |
| `run_index.jsonl` | 运行生命周期 | — |
| `replay.json` | 重放记录 | — |