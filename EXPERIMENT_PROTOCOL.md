# CAGE-Attention 实验协议文档

## 版本控制

| 版本 | 日期 | 变更说明 | Git SHA |
|------|------|----------|---------|
| 1.0 | 2026-11-03 | 初始版本，对应预注册假设注册表 v1 | - |

> **重要**：本协议文档与 `cage/experiments/_config.py:HYPOTHESES` 及 `docs/design_notes.md` 形成三位一体的预注册体系。任何实验协议变更必须同步更新三处，并通过 `ci_gate.py` 设计固定项检查。

---

## 1. 研究问题与假设体系

### 1.1 核心研究问题

1. **RQ1（门控新颖性）**：门控注意力是否只是已有门控方案（Gated Attention 家族）的微小变体？
2. **RQ2（深层协同）**：深层残差流贡献不足时，门控与归一化应如何协同设计？

### 1.2 假设层级

| 层级 | 编号范围 | 性质 | 实验 |
|------|----------|------|------|
| 基线前提 | HB0–HB7 | 管道有效性/任务合理性/复杂度控制 | baselines |
| 机制核心 | H1a–H1g | 校准门控的注意力分布使用、错误抑制、消融、安慰剂 | E1 |
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
| test | `TEST_SEED=456` | 最终报告数字、预注册门控规则 | 封存，从不用于选择 |
| stats | `STATS_SEED=7` | 机制统计（熵、门控、错误挖掘） | 固定，与 test 独立 |

**HB3 分裂审计**：训练流与每个保留流的精确重复重叠=0；分布偏移卡方检验 BH 校正后 p ≥ FDR。

### 2.2 E1：门控对比（4 变体 + 消融 + 安慰剂）

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

**共享调度**：上述 8 个 cell 进入**同一个**随机化完全区组设计（每种子一个区组，每区组 8 个 cell 位置随机排列），避免消融对比被机器漂移混淆。

**网格配置**（默认，可通过 CLI 覆盖）：
```python
# 共享冻结配方（在 none 上用 VAL_SEED 调得）
steps=2000, lr=3e-4, batch=32, n_pairs=16, vocab=64
d_model=256, n_layers=8, n_heads=4, max_len=256
dropout=0.0, norm=pre, layer_scale=None
# calib 专用
calib_a=1.0, calib_c=1.0, calib_scale=1.0, calib_tau_init=0.9
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
**扫描长度**：`length_sweep_lengths(base=16, factor=2, n=4, max_len=256)` → `[16, 32, 64, 128]`（需至少 2 个拟合长度）

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
| calib_shuf_ent | calib | True | True | True | 安慰剂（同 E1） |

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

---

## 6. 复现与验证检查表

### 6.1 发布前自检

- [ ] `make check` 通过（语法 + 设计固定项 + 冒烟 + 全测试）
- [ ] 正式实验使用非 `--quick` 设置、足够种子数（≥5）
- [ ] 结果目录完整保留（未手动删除/修改任何文件）
- [ ] `make verify OUT=results` 全部通过
- [ ] `baselines_verdict.json` 所有门禁通过
- [ ] 设计计划冻结无冲突（`_design_plan_history.jsonl` 无错误）

### 6.2 独立复现核对

- [ ] 使用 `requirements-ci.txt` 精确依赖版本
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

---

## 8. 变更控制

### 8.1 协议修改流程

1. 在 `docs/design_notes.md` 修改对应 `【固定】` 条目
2. 同步更新 `cage/experiments/_config.py:HYPOTHESES`（假设注册表）
3. 同步更新本文档 `EXPERIMENT_PROTOCOL.md`
4. 运行 `make check` 确保 CI 门禁通过
5. 提交并移动固定标签：`git tag -f design-pin <新提交sha> && git push origin design-pin`

### 8.2 版本兼容性

| 协议版本 | 兼容结果目录 | 说明 |
|----------|--------------|------|
| 1.0 | ≥ v1.0 | 向前兼容：旧结果可读、可验证、可重放 |
| 1.x | ≥ v1.x | 次版本仅增量添加字段/伴读，不改变决策规则 |

### 8.3 弃用与移除

- 移除任何预注册假设/指标/对比需：同步三文档 + 移动 design-pin 标签 + 标记为 `deprecated` 而非直接删除（保留历史记录可读性）
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
python -m cage.experiments --only e1 --lr-sensitivity                # lr 敏感性（探索性）

# 验证
python -m cage.experiments --verify-artifacts --out results
python -m cage.experiments --check-environment --out results
python -m cage.experiments --verify-provenance --out results
python -m cage.experiments --verify-runs --out results
python -m cage.experiments --check-source --out results
python -m cage.experiments --verify-replay results --out results_replay

# 重放
python -m cage.experiments --replay-manifest results/e1_manifest.json --out results_replay

# Makefile 封装
make check | gate | test | sanity | verify | verify-replay SOURCE=results
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
| `baselines_verdict.json` | 基线裁决 | — |
| `environment.json` | 当前环境记录 | — |
| `environment_history.jsonl` | 环境漂移历史 | — |
| `artifact_index.json` | 累积制品索引 | — |
| `run_index.jsonl` | 运行生命周期 | — |
| `replay.json` | 重放记录 | — |