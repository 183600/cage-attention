# 数据层反事实评测：回答究竟来自检索、记忆还是位置？

> 本文档补充 `docs/design_notes.md`（固定设计，不得修改）与
> `docs/results_analysis.md`（统计读数）。它只增加**数据层操控检验**，
> 不改变任何预注册假设、FDR 族、SESOI 或能力门禁。

## 0. 为什么还需要这一层

联想回忆任务的目标是让模型**在上下文中检索**与 query key 配对的 value。
但同一份评测准确率可以来自至少四种不同的规则：

1. 真正按 query key 检索上下文中的配对 value；
2. 记住训练中见过的 `key -> value` 关联，直接输出记忆值；
3. 复制某个固定位置（例如“最后一个 value”或“紧跟某个模式的 token”）；
4. 对输入编辑不敏感，总是输出高频 token。

只看 accuracy 无法区分它们。尤其当训练步数足以覆盖大部分
`(key, value)` 组合时，第 2 种解释必须在数据层被单独排除，而不是靠
事后的相关性猜测。

## 1. 三个确定性的最小编辑

`cage.train.eval_metrics` 在**已经生成并通过 `validate_batch` 的同一个
评测 batch** 上构造三个反事实副本。编辑完全确定、不消耗 RNG，也不改变
评测流的其余内容：

| probe | 数据编辑 | 正确行为 | 它排除的捷径 |
|---|---|---|---|
| `value_swap` | 把 query key 配对的 value 换成另一个可选的 value | 答案跟随新 value | 记忆 `key -> value`、固定位置复制 |
| `distractor_swap` | 把某个**非 query** pair 的 value 换掉 | 答案基本不变 | 对任意 token 编辑都过度敏感 |
| `pair_relocation` | 把整个 `(key, value)` pair 移到另一个 pair 槽位 | 答案不变 | 只按位置/槽位复制，而非按 key 内容检索 |

实现：`cage.train._counterfactual_edits`。替代 value 由
`_counterfactual_alternative` 在 `[2, vocab)` 内循环一位得到，保证：

- 仍是生成器可能产生的 selectable token（不会出现 PAD/SEP）；
- 与旧 value 不同；
- key 仍然唯一、SEP 仍在固定槽位、query 仍在最后位置；
- 编辑后的 batch 会再次通过 `validate_batch`，因此 probe 不会在畸形
  数据上比较预测。

`n_pairs == 1` 时没有第二个 pair，`distractor_swap` 与
`pair_relocation` 直接 abstain（相应 `n = 0`），不会偷偷编辑 query pair。

## 2. 读数

每次评测记录新增：

```text
counterfactual.value_swap
  n / n_original_correct
  n_follow / follow_rate          # 干净预测正确且跟随新 value
  n_stubborn / stubborn_rate      # 干净预测正确但仍输出旧 value
  n_correct_after_edit / accuracy_after_edit
  flip_rate                       # 编辑后预测发生变化的行占比
counterfactual.distractor_swap
  n / n_unchanged / unchanged_rate
  n_correct_after_edit / accuracy_after_edit
counterfactual.pair_relocation
  n / n_unchanged / unchanged_rate
  n_correct_after_edit / accuracy_after_edit
```

同时写入扁平字段，便于 `_aggregate_runs` 与汇总 JSON 直接使用：

```text
cf_value_follow_rate          cf_value_stubborn_rate
cf_value_accuracy             cf_value_flip_rate
cf_distractor_unchanged_rate  cf_distractor_accuracy
cf_relocation_unchanged_rate  cf_relocation_accuracy
cf_n                          cf_n_original_correct
cf_n_distractor               cf_n_relocation
```

`cf_n*` 是分母：`follow_rate` 条件在 3 个干净正确行上，与条件在 300 个
样本上不是同等证据；分母为零时比率为 `NaN`，但 `cf_n_original_correct=0`
会显式保留这一事实。

### 2.1 条件化与跨臂可比性

`follow_rate` 与 `stubborn_rate` 以**干净 pass 中预测正确的行**为分母：
“编辑后是否跟随”只在这些行上有可解释的含义。若干净模型一行都没答对，
`follow_rate = NaN`，而不是 0；未定义的诊断不能被当成“模型在记忆”的
证据。

但“干净正确”是模型输出后的条件，**不是处理前协变量**。不同臂的正确
子集可能由不同难度的 item 组成，直接比较 `follow_rate` 会把“哪些题做对
了”的选择差异混进来。因此：

- **跨臂主读数**用 `cf_value_accuracy`（value swap 后整批 item 的准确率）
  与 `cf_distractor_accuracy` / `cf_relocation_accuracy`（整批 item 的
  安慰剂准确率）。这些量在编辑后的完整评测流上定义，分母固定，不随臂
  的干净正确率变化；
- `follow_rate` / `stubborn_rate` 是**臂内描述**：它们回答“该臂答对的
  那些题，是否真的跟着配对 value 变化”，不作为跨臂显著性主张；
- `distractor_swap` / `pair_relocation` 的 `unchanged_rate` 同样保留为
  描述性稳定性读数；一个恒定输出的模型可以在这两个稳定性读数上取高值，
  所以不能单独用它证明“检索”。

### 2.2 预期模式

一个真正按 key 检索的模型应当同时满足：

- `cf_value_accuracy` 高（换 value 后在整批 item 上仍能答对新 value）；
- `cf_value_follow_rate` 高、`cf_value_stubborn_rate` 低（在原本答对的
  子集上不固守旧 value）；
- `cf_distractor_accuracy` 与干净 accuracy 接近（改无关 value 不影响）；
- `cf_relocation_accuracy` 与干净 accuracy 接近（移动 pair 不影响）。

反之：

- value swap 后整批准确率接近机会水平，而干净准确率很高 → 更像记忆或
  位置复制，而不是按上下文 value 作答；
- `distractor` / `relocation` 后准确率大幅下降 → 对无关编辑过度敏感；
- relocation 后准确率大幅下降 → 依赖绝对/相对槽位。

这些是**描述性操控检验**，不是新的显著性决策。它们不进入 FDR 族，也
不能把一行 `SURVIVES` 改成 `REFUTED`；但报告机制结论时必须先看它们：
若某臂 accuracy 更高而 `cf_value_accuracy` 没有提高，不能写成“该臂
更好地利用了上下文配对信息”。

## 3. 汇总与输出

- `<tag>_runs.jsonl` 的每条记录带上述 `counterfactual` 块与 `cf_*` 标量；
- `<tag>_summary.json` 的每臂聚合表自动包含 `cf_*` 的均值/标准差；
- 额外写 `<tag>_counterfactual.json`：逐臂均值与相对参考臂
  （E1/E2 为 `none`；E3 为同 `norm`/`layer_scale` 的 `query` 臂）的
  配对 bootstrap 区间；value swap 的跨臂主对比是 `cf_value_accuracy`，
  follow/stubborn 与两个 `unchanged_rate` 仅作描述；
- 控制台打印逐臂 `clean_acc / value_cf_acc / value_follow /
  distractor_acc / relocation_acc` 表。

E2 的每个长度都独立记录一套 probe（`metrics_by_length[L]`），因此可以
检查“长序列准确率下降”是否伴随检索行为的下降，而不是只比较 accuracy。

## 4. 解释边界

- 单 token 最小编辑不是形式化因果证明：模型可能同时使用多种部分线索，
  也可能对编辑产生非局部表示变化。三个 probe 联合读，比单个 probe 更
  可信。
- `pair_relocation` 会改变位置编码，因此“完全不变”不是必需条件；
  它只用于排除**纯位置复制**这一极端解释。
- value swap 的新 value 可能同时出现在其他位置；由于 key 唯一，正确
  pair 仍然唯一，不会把任务变成不可定义。
- 这些读数与 `cage.data` 的切分/分布审计互补：前者问“模型用了哪条
  规则”，后者问“数据是否支持受控比较”。

## 5. 复现

```bash
python3 -m unittest tests.test_counterfactual_eval -v
python3 -m cage.experiments --only e1 --quick --out results_quick
python3 -m cage.experiments --only e2 --quick --out results_quick
python3 -m cage.experiments --only e3 --quick --out results_quick
make check
```

所有 probe 都是确定性的，不消耗 RNG 流；同一 seed、同一评测流下重复
评测得到同一组计数。

## 6. 关联暴露审计：行级去重之外的必要一层

### 6.1 为什么整行重复不够

`split_leakage_audit` / `actual_stream_overlap_audit` 的整行指纹回答的是
“生成器是否抽到了同一条完整序列”。在合成任务里这个事件几乎不发生：
两条序列只要其他 pair 不同，整行就不同；但**同一个 query key 与同一个
value 的关联**（下称 pair）仍可能同时出现在训练流和评测流。模型完全
可以记住 `key -> value`，在干净评测上答对，而整行泄漏审计仍显示 0。

因此数据层需要把“评测 item 的精确 `(query key, answer value)` 关联是否
在训练流中出现过”作为独立读数。它回答的是**记忆机会**，不是模型是否
真的在记忆；后者由第 1–5 节的反事实 probe 回答。

### 6.2 实现

- `cage.data.association_codes` 从已通过 `validate_batch` 的 batch 中取出
  query pair，编码为 `key * vocab + value`；同时校验 `key_pos`、最终
  query token 与 `meta['answer']` 一致，畸形 batch 直接 `ValueError`。
- `actual_stream_overlap_audit` 在重放每个消费流时顺带记录 pair code 与
  query key 集合，训练组（`group="train"`）的并集作为参考；held-out 流
  只被**打分**，不参与参考拟合。
- `eval_metrics(..., train_association_codes=...)` 用同一参考把每个评测
  item 分为：
  - `seen`：精确 pair 在训练流中出现过，记忆即可作答；
  - `unseen`：精确 pair 未出现，必须从上下文读取。
  该划分只依赖训练数据，不依赖模型输出，是**处理前**协变量。

### 6.3 读数

`association_exposure`（审计 JSON，逐 held-out 流）：

```text
n_items / n_unique_pairs / n_unique_keys
pair_seen_train / pair_seen_train_rate      # 精确关联已见率
key_seen_train / key_seen_train_rate        # key 已见（任意 value）
pair_unseen_key_seen / ..._rate             # key 已见但该 value 未配对
pair_seen_unique / ..._rate                 # 去重 pair 口径
```

`train_association_summary` 报告训练流的 pair 空间覆盖：

```text
n_items / n_unique_pairs / n_unique_keys
n_possible_pairs / n_possible_keys / pair_space_coverage
```

`eval_metrics` 的 `association_exposure` 块逐 cohort 报告：

```text
seen/unseen: n, n_correct, accuracy,
             cf_value_accuracy, cf_value_follow_rate,
             cf_value_stubborn_rate,
             distractor_accuracy, relocation_accuracy
accuracy_gap_seen_minus_unseen
```

扁平字段 `assoc_exposure_*` 供 `_aggregate_runs` 与汇总表直接使用；分母
为零时比率为 `NaN`，不把“该 cohort 没有样本”读成 0。

### 6.4 解释边界

- `pair_seen_train_rate` 是**记忆机会的上界读数**，不是泄漏判定：共享
  vocabulary 下部分关联重合是任务构造的必然结果。跨组整行重合、跨组
  seed 复用仍是硬失败；关联暴露只披露、不改变 HB3 的通过条件。
- `seen` 与 `unseen` 两个 cohort 的 item 组成不同（例如已见 pair 可能
  集中在更常见的 key 上），所以 `accuracy_gap_seen_minus_unseen` 是
  描述性读数，不是因果效应；要判断“准确率是否来自上下文检索”，必须
  同时看两个 cohort 的 `cf_value_accuracy` / `cf_value_follow_rate`。
- 一个健康的检索模型应在 `unseen` cohort 上仍有高于机会的
  `cf_value_accuracy`；若干净 accuracy 的增益集中在 `seen`，而
  `seen` 的 value-swap follow rate 很低，则不能把增益写成“更好地利用
  了上下文配对信息”。

### 6.5 复现

```bash
python3 -m unittest tests.test_association_audit -v
python3 -m cage.experiments --only sanity --quick --out results_quick
make check
```

关联重放与 probe 一样不消耗 RNG；同一 Settings / 同一训练几何下重复
审计得到同一组计数。
