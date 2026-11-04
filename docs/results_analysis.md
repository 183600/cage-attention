# CAGE-Attention 结果分析方法（固定版）

> **本文档说明"一份结果文件该怎么读"。** 协议（做什么、怎么注册）
> 在 `EXPERIMENT_PROTOCOL.md`，假设注册表在
> `cage/experiments/_config.py:HYPOTHESES`，研究级确认性对比集在
> `cage/experiments/_analysis_plan.py`。本文件解释这些注册内容如何变成
> 一个可辩护的结论，以及每一步的输出落在哪个制品里。

---

## 0. 为什么有这份文档

同一个数字，在不同的读法下可以是"支持""不显著""无效""不可裁决"。
本项目的判定链是**分层**的，每一层都有明确的注册依据和明确的制品：

| 层次 | 问题 | 判据 | 制品 |
|------|------|------|------|
| 行级 | 这一条对比是否成立？ | 受控审计 + 方向 + SESOI + 种子门槛 + 家族内 FDR | `<tag>_contrasts.json` 的 `rows[]` |
| 假设级 | 这一条**假设**是否成立？ | 该假设全部注册行的合取 | `<tag>_contrasts.json` 的 `hypothesis_synthesis` |
| 研究级 | 这一**研究**的头条主张是否成立？ | 预注册确认性腿的 IUT + 合并族 FDR | `study_level_adjudication.json` |

层次不能跳过。一条行 `supported` 不等于一条假设 `supported`；一条假设
`supported` 也不等于研究的主张 `supported`。

---

## 1. 行级判定：一条对比何时算"成立"

一行进入家族的唯一入口是 `ContrastLog.print` / `print_equivalence` /
`print_equivalence_ci`（`cage/experiments/_contrastlog.py`）。它把以下
注册信息与统计量绑定在一起，**判定规则不在判定处临时决定**：

1. **注册来源**：SESOI、方向（`+` / `-` / 无）、零预测检验方式
   （`equivalence` / `noninferior`）全部来自 `_config.py:HYPOTHESES`。
   注册表里没有的方向键会被拒绝。
2. **受控变量审计**（`_contrasts.py::_controlled_audit`）：两臂的优化配方、
   优化器、评估流、配对种子数、变化的模型因子逐项比对。配方不同、流不同、
   种子未配对或改动多于一个因子 → `controlled=false`，该行不得支持任何结论。
3. **方向**：方向相反的显著结果一律是 **refuted**，无论效应量多大。
4. **SESOI**：统计显著但效应量 < SESOI → `practically negligible`，
   不是"支持"。
5. **种子门槛**：`min_seeds_for_claim`（默认 3）个独立配对种子是声明的
   最低门槛；不足的行标 `INELIGIBLE`，`claim_eligible=false`。
   被丢弃的种子记录在 `seeds_dropped` / `seeds_unpaired` / `seeds_undefined`。
6. **曲线失效**：任一侧含 plateau / diverging / numerical 的运行 →
   该行标注曲线警告；这类行常常只是"没学会"，不是"没差别"。
7. **家族内 FDR**：Benjamini-Hochberg（`--fdr-q`，默认 0.05）在该实验的
   家族上校正。

### 1.1 判定语义

| 判定 | 条件 |
|------|------|
| `supported` | FDR 显著 + 方向一致 + 效应量 ≥ SESOI + 行可声明 |
| `refuted` | FDR 显著 + 方向相反 |
| `practically negligible` | FDR 显著 + 方向一致 + 效应量 < SESOI |
| `inconclusive` | FDR 不显著，或 CI 宽度跨越 SESOI 边界 |
| `INELIGIBLE` | 配对种子数 < `min_seeds_for_claim`，或前提被推翻 |

**"CI 包含 0"永远不是"没有效应"的证据。** 它只说明这个设计在这个样本量下
没有分辨出来；计划阶段的 MDE80 / 所需种子数就是为此报告的。

### 1.2 零预测（等价性 / 非劣效性）

注册为 `equivalence` / `noninferior` 的行必须用 **TOST** 读：
只有当 95% CI **完全落在 ±SESOI 内**（等价）或**单侧下界 ≥ -SESOI**（非劣）
才算支持。"CI 包含 0"不是等价性证据。

---

## 2. 行级伴随（不改变判定，只让判定更脆的地方可见）

伴随读数**永远不替换**预注册判定；它们的用途是让读者看见判定的脆弱点：

| 伴随 | 读什么 | 何时必须警惕 |
|------|--------|-------------|
| BCa 区间 | 偏倚校正 + 加速后的区间 | 与百分位区间结论不一致 |
| Jackknife（留一） | 去掉单个种子后结论是否翻转 | 翻转 → 结论由一个种子决定 |
| 精确配对符号翻转 | 小 n 下可达的最小精确 p | 5 种子双侧最小 p = 0.0625，**在 q=0.05 下无法确认** |
| McNemar（逐条样本） | 同一封闭流上的逐样本一致性 | 与种子级结论方向相反 |
| 复制区间 | 未来同量级研究均值的区间（sqrt(2/n)） | "复制不确定" ≠ "稳健" |
| FCR 区间 | 选中集合的假覆盖率 | 区间判决翻转 → 选择敏感 |
| 受控审计 | 变化了哪些因子 | `controlled=false` → 该行不可用 |

**只有 bootstrap 显著而精确检验不显著的行，报告为"暂定"而不是"确认"。**

---

## 3. 假设级综合：一条假设是若干注册行的合取

一条假设往往由多行共同支撑（例如 H1a 同时有
`accuracy`（非劣）与 `abs_gate_vs_entropy`（方向性））。因此：

- 该假设的**全部**注册行都 `supported` → 假设 `supported`；
- 部分支持 → `partial`；
- 出现方向相反的显著行 → `refuted`；
- 无显著行 → `inconclusive`；
- 某条注册行根本没被执行 → `no_contrast`（**不是"没发现"**）。

单行存活不得读作"假设被确认"。`no_contrast` 尤其重要：它说明实验没有
测到该假设声称的东西，而不是测了没测出来。

---

## 4. 研究级分析计划：跨家族的主张

### 4.1 为什么需要它

E1/E2/E3 各自控制一个 FDR 家族。**三个家族各自在 q=0.05 上做 BH，
不意味着它们发现的并集是 FDR-0.05 集合**；而研究的头条主张在多个家族里
被重复检验。只读"在自家家族里成立"，等于读了一个从未被控制过的错误率。

### 4.2 注册了什么

`_analysis_plan.py` 预注册两样东西：

- **确认性对比集**（`_PRIMARY_CONTRASTS`，P1–P9）：每条腿用
  `(experiment, hypothesis, metric, rule)` 标识运行器实际产出的那一行，
  并写出它对比的网格单元 `arms` 与 `kind`（`one_factor` / `interaction`）。
- **研究级声明**（`_STUDY_CLAIMS`，C1–C4）：每条声明是其腿的**合取**
  （intersection-union）。IUT 下拒绝一个合取需要拒绝它的每一条腿，
  因此声明之间不需要额外校正；声明的全局 p 值是其各腿单侧 p 值的最大值。

### 4.3 开跑前的三道审计

| 审计 | 拒绝什么 | 为什么 |
|------|---------|--------|
| 注册表一致性 | 假设/实验/指标 SESOI/方向在 `HYPOTHESES` 中不存在的腿 | 裁决一条没人注册过的比较 |
| **可达性** | `arms` 不是该家族网格单元的腿 | 这条腿**永远无法被测量** |
| 合并族功效 | 计划种子数下 q/m 阈值解不开 SESOI 的腿 | 合并家族更大，单家族通过不等于研究级可裁决 |

可达性不可达时**拒绝开跑**（`DesignPlanError`）：让研究先花算力再以
"不可裁决"收场，而原因在源码里已经写着，是没有理由的。

### 4.4 冻结

`analysis_plan.json` 带内容哈希（`analysis_plan_sha256`，覆盖对比集、声明、
合并 FDR 水平、声明门槛、计划种子数与假设注册表哈希）。**任何家族写出结果行
之后**，修改确认性对比集、其多重性水平或计划种子数都会抛 `DesignPlanError`——
那是看过数据之后重选确认性对比，属于事后重设计。开跑前的修订保留在
`analysis_plan_history.jsonl`。

冻结是**研究级**的，不是实验级的：E1 不能用一套主对比、E2 用另一套。
因此 E1/E2/E3 每个家族在开跑前都发布/复核同一份计划，"先跑哪个家族"是等价的。

### 4.5 两个研究级读数

`study_level_adjudication.json` 同时给出：

- `study_status`：按**各家族自己的**预注册判定读的 IUT 结果；
- `study_status_pooled_family`：同样的腿在**合并的确认性家族**（BH，q）
  里读的 IUT 结果。

两者不一致的腿/声明记为 `family_inflated`（家族膨胀）：
**在自家家族里成立，在合并族里不成立**。这类结论必须以"研究级不确定"报告，
不能写成"成立"。

另外，**缺家族**的运行报 `study_status=incomplete`，并列出
"研究家族缺失 / 腿无对应行 / 腿匹配到多行 / 计划与注册表不一致"等偏离。
部分家族的结果**不得**当作完整研究汇报。

### 4.6 不合格的腿不能裁决声明

配对种子数不足、曲线失效、评估流不平衡或非受控对比的行，
`claim_eligible=false`，因此既不能支持也**不能反驳**一条声明——
与 §3 假设综合的约定一致。一个 2 种子快速检查里的"反向显著"不是反驳。

---

## 5. 读法指南

**可以说**：

- "在 E1 家族内，calib 与 query 的 `abs_gate_vs_entropy` 差值为
  +0.18 [+0.09, +0.27]，BH 校正后 p=0.002，效应量 ≥ SESOI 0.05，
  5/5 种子同号，结论为 supported。"
- "研究级声明 C1 的三条腿在合并族中全部成立；研究状态 supported。"
- "研究不完整：E3 未运行，因此本研究只能报告 E1/E2 的部分结论。"

**不能说**：

- "E1 通过了，所以机制成立。"（单家族 ≠ 研究级）
- "CI 包含 0，说明没有效应。"（只是没分辨出来）
- "方向相反但幅度不大，所以只是噪声。"（方向相反的显著结果是 refuted）
- "两行里有一行显著，所以 H1a 成立。"（假设是合取）
- "`--quick` 的 2 种子结果说明机制成立。"（低于 `min_seeds_for_claim`，
  只是流水线检查）

---

## 6. 制品对照表

| 想回答的问题 | 看哪个文件 |
|-------------|-----------|
| 这一条对比成立吗？ | `<tag>_contrasts.json` → `rows[]` |
| 这一条假设成立吗？ | `<tag>_contrasts.json` → `hypothesis_synthesis` |
| 研究的主张成立吗？ | `study_level_adjudication.json` → `claims[]` |
| 确认性对比集是什么？ | `analysis_plan.json` → `primary_contrasts` |
| 设计为什么长这样？ | `analysis_plan.json` + `analysis_plan_history.jsonl` |
| 开跑前认证过什么？ | `study_design_preflight.json` |
| 结论会不会因一个种子翻转？ | `<tag>_contrasts.json` → 每行的 jackknife / BCa / sign-flip 字段 |
| 换台机器还能复现吗？ | `make verify OUT=results`（环境/制品/溯源/运行/源码五项） |
| 这份结果是哪个协议产生的？ | `<tag>_manifest.json` + `run_index.jsonl` + `artifact_index.json` |
