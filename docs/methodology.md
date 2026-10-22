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

## 3.5 评测数据的流指纹与 fold 稳健性

种子层推断把训练种子当作独立单位，但所有种子共享**同一条有限评测流**。
因此还需要在“数据”这一层回答两个问题：两条记录是否真的评了同一批样本？
结论是否被评测流中的某一小段样本单独撑起来？

- **流内容指纹**：每次 `eval_metrics` 对实际消费的整条评测流（所有 batch、
  上下文、标签、`query_pos` / `key_pos` / `answer`）计算
  `stream_digest`，而不是只散列第一批。逐项 McNemar / fold 伴随读数按位置
  比较正确性；若两侧指纹不一致，伴随读数直接
  `applicable=False`，不会把两组不同样本当成同一批 item 配对。旧记录缺少
  指纹时保留读数但标记 `digest_verified=False`，不冒充已验证。
- **fold jackknife**：把评测 item 按消费顺序切成 `k` 个连续块
  （`k=min(8, n_items//2)`），报告每块内的配对效应、逐种子 fold 符号是否
  翻转、去掉任意一块后的效应范围，以及最大块占全部绝对效应的比例。
  `fold_audit.fragile=True` 表示“去掉一块就翻符号”或“单块贡献超过一半”；
  `n_unit_fold_sign_flips` 表示有多少训练种子的 fold 效应方向不一致。
- **解释层级**：上述两项都是**数据层敏感性伴随读数**，不替代第 2 节的
  预注册种子层决策，也不改变 FDR 结论。支持的对照若 fold 脆弱，应写成
  “在该评测流上支持，但数据层泛化需更宽或独立的评测流”，不能写成机制已
  被普遍证实。逐项 McNemar 的 pooled p 仍只作描述，因为同一条评测流会被
  每个种子重复评分。

实现位于 `cage.data.stream_content_digest`、`cage.metrics.evaluation_fold_audit`
与 `cage.experiments._paired_stream_test`；相应字段随每个对照写入
`<tag>_contrasts.json` 的 `stream_test` / `stream_fold_audit`。

## 3.6 种子 × 评测项交叉 bootstrap：区分“在这条流上”与“在任务总体上”

第 1 节把训练种子当作独立单位，但预注册的种子层 bootstrap 在重采样种子时
**固定**了整条 sealed 评测流。它回答的是“换一批训练种子、仍在这条流上评，
效应会不会复现”，而不是“换一批训练种子、再换一条有限评测流，效应会不会
复现”。后者多了一层 item 抽样方差；在 5 个种子、有限评测集的规模下，这一
层可能正好决定显著性。

`cage.metrics.crossed_cluster_bootstrap_ci` 对“种子 × item”两个轴同时有放回
重采样：每个 replicate 先重采样训练种子，再重采样评测 item，然后在重采样后的
种子上平均配对正确率差。返回的百分位区间是**数据层泛化伴随读数**：

- `seed_lo` / `seed_hi`：只重采样种子、保留全部 item（对应“这条流上”的
  种子层问题）；
- `item_lo` / `item_hi`：只重采样 item、保留全部种子；
- `sd_seed` / `sd_item` / `sd_total` 与 `variance_share_seed`：哪一轴主导
  不确定性；`variance_share_seed` 只是描述性比例，不是 ANOVA 分解，因为
  交叉 bootstrap 还包含 seed × item 交互；
- `ci_excludes_zero` 与调用层写入的 `seed_ci_excludes_zero`、
  `decision_agrees`、`stream_limited`。`stream_limited=True` 表示种子层
  95% CI 排除 0，但种子 × item CI 包含 0：效应只在这条已实现的评测流上
  成立，不能写成任务总体上的机制结论。

两个轴都至少需要 2 个水平才能估计各自的方差；输入畸形、非 0/1、空矩阵或
单种子/单 item 时返回 `valid=False` 与原因，不伪造区间。该伴随读数不替代
第 2 节的预注册种子层决策，也不改变 FDR 结论；它只把“结论对评测流内容的
依赖”显式化。字段随每个对照写入 `<tag>_contrasts.json` 的
`stream_test.crossed_bootstrap` 及行级 `stream_crossed_*`。

## 4. 预设功效与批次顺序

- 每个网格在训练前写 `<tag>_design_plan.json`：用基线阶梯的 pilot 方差把
  SESOI 翻译成“计划种子数能容忍的最大 SD”，并对欠功效项提前报警。
- **配对设计必须按配对差的正态尺度做功效规划**。只用一个臂的 seed 间 SD
  隐含“两臂的 seed 效应完全相关”，会在两臂独立时高估功效。规划器因此使用
  `max(相邻深度在同一 seed 上的配对差 SD, sqrt(2) × 最大单臂 SD)`：
  前者是基线阶梯里唯一的经验配对方差，后者是两臂独立时的
  `Var(A-B)=Var(A)+Var(B)` 界；单臂 SD 只作为兼容字段
  `pilot_sd_single_arm` 与对照 `required_seeds_single_arm` 保留。
  逐行字段为 `pilot_paired_sd` / `pilot_paired_source` /
  `pilot_paired_components` / `pilot_paired_inflation`。
- **有限评测流有独立于种子数的精度下限**。在“两臂逐 item 误差独立”的
  最坏情形下，配对差的 item 方差为 `2p(1-p)`，`N` 个 item 的均值标准误为
  `sqrt(2p(1-p)/N)`；因此设计计划同时报告
  `planned_eval_items`、`stream_item_sd_floor`、
  `stream_items_for_sesoi` 与 `stream_limited_worst_case`。这不是把
  item 方差重复加进 seed 方差（seed 层差已包含该已实现流的噪声），
  而是一个独立的下限诊断：`stream_limited_worst_case=True` 时，继续加
  training seeds 不能消除该下限，应加宽 `eval_batch × eval_batches`，并用
  3.6 节的种子 × item 交叉 bootstrap 检查实际 item 方差是否主导。
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
6. 前提门禁状态（能力、覆盖、曲线、答案平衡、逐项符号）；
7. 评测流指纹是否验证（`stream_digest_verified`）、fold 稳健性
   （`stream_fold_loo_sign_flip`、`stream_fold_dominant_share`、
   `stream_fold_fragile`）与种子 × item 交叉 bootstrap（`stream_crossed_lo` /
   `stream_crossed_hi` / `stream_crossed_stream_limited`）；脆弱或
   `stream_limited` 时给出“只在该评测流上成立”的解释边界。

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
