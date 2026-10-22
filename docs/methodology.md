# CAGE-Attention 科研方法与统计推断约定

> 本文档补充 `docs/design_notes.md`（设计宪法，固定项不得由 CI 修改）。
> 它说明**结果如何被读出**：分析单位、决策层级、多重比较、样本量与
> 小样本下的推断边界。新增代码或报告必须与本文一致；改变决策规则属于
> 预注册变更，必须同时更新 `cage/experiments.py::HYPOTHESES` 的哈希内容，
> 不能只改解释文字。

## 1. 分析单位与配对设计

- **独立单位是训练种子**，不是评测样本。同一训练种子下的所有评测样本共享
  同一模型；不同训练种子共享同一条 sealed 评测流。因此所有主对照都按种子
  配对（同一 `seed` 下 A/B 两侧作差），并且：
  - 训练种子必须互不重复、不得与 `VAL_SEED` / `TEST_SEED` / `STATS_SEED`
    别名（见 `cage.data.check_train_seed`）；
  - 评测流复用带来的测量误差用 seed-clustered 区间与逐项 McNemar 诊断
    单独报告，不能把同一批评测样本当成独立伯努利试验来“增加 n”；
  - 单种子删除后的结论是否翻转由 jackknife 诊断；BCa 区间作为小样本
    偏差/偏度校正的伴随读数。
- 选择（学习率、`calib_tau_init`、配方、基线）只允许读 `VAL_SEED`；
  `TEST_SEED` 只用于最终报告与固定的预注册门禁；`STATS_SEED` 只用于机制
  统计。这一封印在代码层强制，而不是靠约定。

## 2. 决策层级（从强到弱）

1. **预注册的最低效应量 SESOI**：统计显著但小于 SESOI 读作
   *practically negligible*，不算支持机制。
2. **预注册方向**：方向相反且 FDR 显著读作 *refutation*，永远不算支持。
3. **Benjamini-Hochberg FDR**：每个实验的整个对照族一起校正
   （`--fdr-q`，默认 0.05）；不能逐条挑显著。
4. **种子下限**：`min_seeds_for_claim`（默认 3）以下的对照标记
   `INELIGIBLE`，无论 p 值多小。
5. **前提门禁**：HB0 能力前提、设计覆盖/种子平衡、曲线有效性
   （plateau/diverging/数值不稳定）、评测流答案平衡、逐项符号一致性。
   任一门禁失败，对照不能支持机制结论。
6. **小样本稳健性**：jackknife、BCa、精确 sign-flip 作为*伴随诊断*。
   它们不替换第 3 步的预注册 bootstrap 决策，但会显式标出
   “bootstrap 显著、精确检验不显著”或“该设计在原理上无法拒绝”的行。

## 3. 精确配对符号翻转检验

`cage.metrics.sign_flip_test` 在零假设“配对差关于 0 对称”下枚举全部
`2**n` 个符号分配（n 超过 `max_exact` 时用固定种子的 Monte-Carlo 近似）。
它回答的是：**这个种子数下，数据最多能提供多强的证据？**

| n（配对种子） | 单侧最小 p | 双侧最小 p |
|---:|---:|---:|
| 3 | 0.125 | 0.25 |
| 4 | 0.0625 | 0.125 |
| 5 | 0.03125 | 0.0625 |
| 6 | 0.015625 | 0.03125 |

由此得到两条硬约束：

- 双侧读法在 n ≤ 5 时**不可能**达到 0.05；方向性假设在 n ≤ 4 时也
  不可能达到 0.05。这样的行会被标记
  `sign_flip_cannot_reject` / `n_sign_flip_unfalsifiable`。
- 五种子全同向时，bootstrap 可能给出 `p ≈ 2e-4`，而精确单侧 p 只有
  `1/32 ≈ 0.031`。报告应同时给出精确 p 与其 floor，不能把 bootstrap
  的数值当作设计分辨率。

每个 bootstrap 对照现在自动携带：

```text
sign_flip_p_greater / sign_flip_p_less / sign_flip_p_two_sided
sign_flip_method           exact | monte-carlo
sign_flip_min_p_*          该备择假设下的最小可达 p
sign_flip_p_adj            精确检验族的 BH 校正 p（伴随读数）
sign_flip_supports         精确检验下是否支持（伴随读数）
sign_flip_disagrees        bootstrap 与精确检验结论是否相反
```

等价的零预测（TOST）与非劣性检验也携带精确 sign-flip 版本
（`sign_flip_p` / `sign_flip_p_lower` / `sign_flip_p_upper`）。

## 4. 预设功效与批次顺序

- 每个网格在训练前写 `<tag>_design_plan.json`：用基线阶梯的 pilot 方差把
  SESOI 翻译成“计划种子数能容忍的最大 SD”，并对欠功效项提前报警。
- 运行顺序是**随机完整区组**（每个训练种子一个 block，每个 cell 恰好一次），
  记录在 `<tag>_run_order.json`，避免机器漂移与处理臂别名。
- 运行记录带代码版本/源码哈希、完整配方、数据流指纹、线程与确定性开关；
  `artifact_index.json` 跨调用累积，可用
  `python3 -m cage.experiments --verify-artifacts --out <dir>` 校验历史结果
  是否被截断或改动。

## 5. 报告清单

每个进入结论的对照至少报告：

1. 配对种子数 `n`、`seeds_used` / `seeds_dropped`（含原因）；
2. 效应量、配对 SD、MDE80、SESOI；
3. bootstrap 95% CI 与 BH 校正 p；
4. 精确 sign-flip p 与最小可达 p；
5. jackknife 区间、BCa 是否与主决策一致；
6. 前提门禁状态（能力、覆盖、曲线、答案平衡、逐项符号）。

结论措辞：

- `SURVIVES` + 精确检验同向：可以写“在该设计下支持”；
- 只有 bootstrap 支持、精确检验不支持：写“暂定/需更多种子”，不能写成
  机制已被证实；
- CI 包含 0 且 |effect| < MDE80：写“不确定（design cannot resolve）”，
  不能写“无效应”；
- 零预测只有在整个 CI 落在等价界内时才写“等价”，不能用“不显著”代替。

## 6. 复现

```bash
make check                                  # 语法 + 设计固定项 + 冒烟 + unittest
python3 -m cage.metrics                     # 统计层自检
python3 -m cage.experiments --only e1 --seeds 5
python3 -m cage.experiments --verify-artifacts --out results
```

统计层自检不依赖 pytest / numpy；所有随机过程都使用显式 canonical seed。
