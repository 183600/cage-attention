# 结果分析：选择调整区间、复现预测与假设级证据合成

> 本文档补充 `docs/design_notes.md`（设计宪法，固定项不得修改）。它只说明
> **已有结果应当如何被读出**，不改变任何预注册判定：`supports_hypothesis`、
> FDR 族、SESOI、能力门禁等规则保持原样。新增的三个读数都是**伴随读数**
> （companion），用于暴露"统计上支持"与"可作为机制结论"之间的距离。
>
> 实现：`cage/metrics.py::false_coverage_ci`、
> `cage/metrics.py::replication_interval`、
> `cage/experiments/_synthesis.py::_hypothesis_evidence_synthesis`
> 与 `ContrastLog.finalize`。测试：`tests/test_false_coverage.py`、
> `tests/test_replication_synthesis.py`。

## 0. 为什么还需要这一层

项目已经按训练种子配对，报告 bootstrap 95% CI、MDE80、SESOI、精确
sign-flip、jackknife、BCa、评测流 fold/交叉 bootstrap，并用 BH 控制每个
实验族。这些读数回答的是：

1. 当前这次运行的**平均效应**是否与 0（或等价界）相容；
2. 结论对**单个种子**、**区间构造**、**有限评测流切片**有多敏感。

但结果分析还有三个未被显式回答的问题：

- **选择问题**：BH 只报告"哪些对照被选中"，却没有修正这些被选中对照
  的区间。被选中的集合富集了大观测效应，普通 95% CI 不再以名义覆盖率
  覆盖被选参数。
- **复现问题**：如果换一批同样数量的训练种子重做一次，效应会不会仍然
  同号？当前 CI 是 `mean ± t·sd/√n`，它随 n 收缩；而一次新实验的均值
  自身也带抽样方差。把当前 CI 当成"重复实验也会显著"是把
  `Var(mean_old)` 当成了 `Var(mean_new - mean_old)`。
- **合取问题**：预注册假设通常是"A 且 B 且 C"，判定表却是逐行
  （row-level）的。只要一行 `SURVIVES`，报告就可能写成"该假设得到支持"，
  把合取偷换成析取。

## 1. 选择调整区间（false-coverage-rate, FCR）

### 1.1 问题

设一个实验族有 `m` 个可检验对照，BH 在 FDR 水平 `q` 下选中了其中
`R` 个。报告表通常只展示这 `R` 行，并为每行附上普通 95% CI。条件在
"被选中"上，这个区间的覆盖率不再是 95%：选择过程偏向大观测效应，即
winner's curse。FCR 控制的是**被选中集合**中错误覆盖参数的比例期望，
与 BH 的 FDR 目标一致。

### 1.2 定义

对第 `i` 个被选中对照，其 bootstrap 均值分布与
`paired_bootstrap_ci` 相同，但分位点水平改为：

```text
PRDS / 独立：      1 - q · R / m
任意依赖（BY）：   1 - q · R / (m · H_m)
H_m = sum(1/i, i=1..m)
```

因此每侧尾部概率为 `alpha/2`，其中
`alpha = qR/m`（PRDS）或 `alpha = qR/(mH_m)`（任意依赖）。
`R = m` 时 PRDS 区间退化为普通 `1 - q` 区间；`R < m` 时区间变宽。
任意依赖版本用调和因子 `H_m` 作分母，比 PRDS 版本更保守。

### 1.3 判定与解释

- `fcr_decision_holds`：把同一预注册判定规则（方向、等价界、
  非劣界）重新套在 FCR 区间上。
- `fcr_supports`：`fcr_decision_holds` 且该行通过 SESOI/claim gate，
  且没有被 controlled-variable audit 判为 confound。
- `fcr_disagrees`：该行原本 `supports_hypothesis=True`，但
  `fcr_supports=False`。这类行应写成**选择敏感**（selection-sensitive），
  不能写成"稳健"，也不能写成"被证伪"。
- 普通 `supports_hypothesis` 和 BH 判定**不变**。FCR 是报告 companion，
  不是新的显著性检验，也不进入 FDR 族。
- `R = 0` 时没有可报告的 FCR 区间；`R = m` 且 PRDS 时与普通区间相同。
- 若某行种子数不足、bootstrap 退化或输入非有限，函数返回
  `valid=False`；`finalize` 会把该行计入 unavailable 并告警，不会把
  缺失区间当作支持。

### 1.4 报告字段

每个对照行新增：

```text
bh_selected                  BH 原始选择事件（不含 confound/claim gate）
fcr_selected                 是否被 BH 选中
fcr_n_selected / n_tests     R 与 m
fcr                          PRDS 完整读数
fcr_by                       任意依赖（BY）完整读数
fcr_lo / fcr_hi              PRDS 选择调整区间
fcr_adjusted_alpha           调整后的总尾部概率
fcr_confidence_level         1 - fcr_adjusted_alpha
fcr_decision_holds           调整区间是否仍支持预注册判定
fcr_supports                 调整区间 + SESOI/claim gate
fcr_disagrees                原支持与 FCR 支持是否不一致
fcr_by_lo / fcr_by_hi        任意依赖区间
fcr_by_adjusted_alpha        任意依赖调整水平
fcr_by_decision_holds        任意依赖区间是否仍支持
fcr_by_supports              任意依赖区间 + SESOI/claim gate
fcr_by_disagrees             原支持与 BY-FCR 支持是否不一致
```

`<tag>_contrasts.json` 的汇总字段：
`n_fcr_selected`、`n_fcr_tests`、`n_fcr_decision_sensitive`、
`n_fcr_by_decision_sensitive`、`n_fcr_unavailable`、
`n_fcr_by_unavailable`、`fcr_decision_sensitive_contrasts`、
`fcr_by_decision_sensitive_contrasts`。

### 1.5 解释边界

- FCR 保证的是被选中集合的**期望错误覆盖比例**，不是每一行都有 95%
  覆盖率；单行读数仍应结合效应量、SESOI 和复现区间。
- bootstrap 分位点依赖重采样近似；种子数很少时区间本身不稳定，
  应与 jackknife/BCa 读数一起看。
- 任意依赖版本使用 BY 调和因子，代价是区间明显更宽；若 PRDS 版本
  仍支持而 BY 版本不支持，应写成"依赖模型敏感"，而不是"效应不存在"。
- FCR 区间不替代预注册的 BH 判定，也不改变假设级合成中的
  `supports_hypothesis` 真值。

## 2. 复现区间（same-size replication interval）

### 2.1 定义

对 n 个配对的种子级差值 `d_1..d_n`，令 `m`、`s` 为样本均值与无偏标准差。
同规模重复实验均值的 95% 区间为

```text
m ± t_{0.975, n-1} · s · sqrt(2/n)
```

因为两次独立、同规模实验均值之差

```text
Var(mean_new - mean_old) = s²/n + s²/n = 2s²/n
```

`sqrt(2/n)` 才是"重复一次"的尺度；当前均值的 CI 用 `sqrt(1/n)`。
单颗新种子的预测区间更宽，使用 `sqrt(1 + 1/n)`，作为次要读数
（`lo_single` / `hi_single`）单独报告。

`t` 分位数由仓库内的不完全 Beta 函数实现，不依赖 SciPy/NumPy。

### 2.2 判定与概率

- `replicates=True`：同规模重复区间整体不跨 0，且在预注册方向上
  （`direction="+"` 要求 `lo > 0`，`direction="-"` 要求 `hi < 0`）。
- 零预测：`mode="equivalence"` 要求区间整体落在 `±bound` 内；
  `mode="noninferiority"` 要求 `lo > -bound`。
- 未注册方向的两侧读数：`replicates` 表示重复区间排除 0，只作描述。
- `prob_replicates`：在 t 假设下，同规模重复实验均值落在预注册方向的
  概率。它不是 p 值，也不进入 FDR 族。
- `prob_single_same_sign`：单颗新种子保持同号的后验式概率，仅作参考。

### 2.3 解释边界

- 该区间假设种子级效应独立、同方差、近似正态；在 5 个种子时这是明确的
  建模假设，不是无分布保证。
- 它是**伴随读数**：一行仍可按预注册规则 `SURVIVES`，同时被标为
  `replication_uncertain=True`。后者表示"当前均值支持该方向，但同规模
  重复在 95% 下可能反向/跨 0"，不是"该行不成立"。
- 若该行存在曲线无效（plateau/diverging/...）的种子，复现区间会在
  `finalize` 中用与曲线敏感性检查相同的**有效种子子集**重算，结果存于
  `replication_curve_valid` / `replication_effective`，避免用未学会任务的
  运行去预测复现。

### 2.4 报告字段

每个对照行新增：

```text
replication                 完整读数（valid/reason/mean/sd/n/df/t_crit/
                            lo/hi/lo_single/hi_single/prob_replicates/
                            prob_single_same_sign/replicates/...）
replication_lo / hi         同规模重复区间
replication_lo_single       单颗新种子预测区间
replication_prob            同规模重复保持预注册方向的概率
replication_replicates      区间级复现判定（同伴）
replication_uncertain       supports_hypothesis 且 replication_replicates=False
replication_curve_valid     剔除曲线无效种子后的重算（如有）
replication_effective       最终用于 uncertain 判定的读数
replication_reason          无法估计时的原因（n<3、非有限值等）
```

`<tag>_contrasts.json` 的汇总字段：
`n_replication_uncertain`、`replication_uncertain_contrasts`。

## 3. 假设级证据合成（hypothesis-level synthesis）

### 3.1 为什么逐行判定不够

以 H2a 为例，预注册预测是

```text
accuracy_at_longest(calib) > accuracy_at_longest(none)
且 accuracy_drop(calib) < accuracy_drop(none)
且 accuracy_slope_log_len(calib) > accuracy_slope_log_len(none)
```

三行各自通过 FDR 后，才算这个合取成立。如果只有第一行 `SURVIVES`，
把 H2a 写成"supported"就是把"且"读成"或"。同理，H1d 要求每个单因子
消融都显示贡献；H1g 同时要求 calib 的相关为正、query 的相关落在近零界内。

### 3.2 哪些行进入合取

只有**预注册了方向或零预测**的行进入合成：

- `direction` 为 `+`/`-` 的差值行；
- `test` 为 `equivalence` / `noninferiority` 的零预测行。

没有注册方向的两侧 companion（例如 H3b 的 LayerScale 主读数、H2e 的
交互项、H3a 的 post/mix 对照）保持逐行描述，不强行纳入合取；否则会把
"探索性同伴"当成"必须成立"。

### 3.3 状态

| status | 含义 |
|---|---|
| `supported` | 所有进入合取的已运行行都 `supports_hypothesis=True`，且无 refutation |
| `partial` | 至少一行支持，但至少一行仍未确认（不显著/CI 跨界/无判定） |
| `refuted` | 至少一行是预注册方向的反向显著结果，或零预测的 CI 整体越界；无论其它行多显著 |
| `inconclusive` | 没有行支持，也没有行构成 refutation |
| `undirected` | 只有两侧/描述性行，没有可合取的方向或界；逐行读数仍有效，但不计为"已确认的合取" |
| `no_contrast` | 该预注册假设没有任何对照被执行 |

`refuted` 优先于 `partial`/`supported`：合取中一个真值为假的项就足以
否定整个合取，不能被其它支持行抵消。

### 3.4 报告字段

`<tag>_contrasts.json` 新增：

```text
hypothesis_synthesis                  {hyp: {status, n_rows, n_required,
                                       n_confirmed, n_refuted,
                                       n_inconclusive, n_descriptive,
                                       n_descriptive_supporting,
                                       n_replication_uncertain,
                                       n_practically_provisional,
                                       refuted_contrasts,
                                       unconfirmed_contrasts,
                                       replication_uncertain_contrasts,
                                       practically_provisional_contrasts}}
n_hypotheses_supported / partial / refuted / inconclusive /
n_hypotheses_undirected / no_contrast
n_hypotheses_replication_uncertain
partial_hypotheses / refuted_hypotheses /
replication_uncertain_hypotheses
```

控制台会打印合成摘要，并对 `partial`、`refuted`、复现不确定的假设分别
告警。该合成不改变任何行的 `supports_hypothesis`，也不改变 BH 族。

## 4. 结果写法

在原有报告清单（见 `docs/design_notes.md` 与各实验输出）之后，结论句还
必须区分：

1. **行级支持**：`SURVIVES` / `EQUIVALENT`；
2. **选择稳健性**：`fcr_supports` 与 `fcr_by_supports`（原支持行在
   FCR 区间下不再支持时，写"选择敏感"，不能写"稳健"）；
3. **实际大小**：`minimum_effect_established` 与
   `sesoi_point_estimate_only`（点估计过 SESOI 但区间未过界时，只能写
   "实际大小暂定"）；
4. **复现强度**：`replication_replicates` 与 `prob_replicates`
   （复现区间跨 0 时，写"当前运行支持，复现不确定"，不能写"稳健"）；
5. **假设级结论**：只有 `hypothesis_synthesis[hyp].status == "supported"`
   才能写"该假设得到支持"；`partial` 写"部分支持/合取未确认"；
   `refuted` 写"被证伪"，不能因为其它行显著而改写。

建议句式：

- `supported` + `replication_replicates=True` + `fcr_supports=True`：
  "在该设计下，该假设的全部预注册预测均得到支持，选择调整区间仍排除
  0（或仍落在等价界内），且同规模重复区间不跨 0。"
- `supported` + `fcr_supports=False`：
  "当前运行按预注册 BH 规则支持该方向，但该行是 BH 选中的对照之一；
  在 `1 - qR/m` 的 FCR 区间下该方向不再成立，属于选择敏感，需报告
  FCR 区间并避免写成稳健机制结论。"
- `supported` + `replication_replicates=False`：
  "当前运行支持该方向，但同规模重复的 95% 区间跨 0（P(同向)≈x），属于
  复现不确定，需更多种子或更大规模重复。"
- `partial`：
  "该假设是合取；n/m 个预注册预测得到支持，另有 k 个未确认，故只能写
  部分支持。"
- `refuted`：
  "至少一个预注册预测方向被显著反向结果证伪；其它支持行不能抵消。"

## 5. 复现与检查

```bash
python3 -m unittest tests.test_false_coverage -v
python3 -m unittest tests.test_replication_synthesis -v
python3 -m cage.metrics          # 统计层自检（含 FCR/t 分位数检查）
python3 -m unittest discover -s tests -q
python3 ci_gate.py
```

所有随机过程仍使用显式 canonical seed；新增读数本身是确定性的，不消耗
任何 RNG 流。

## 6. 数据层反事实操控检验（补充阅读顺序）

种子层推断回答“换训练种子是否复现”，评测流层回答“换 item 是否复现”，
但都没有回答“模型究竟是按上下文配对检索，还是靠记忆/位置/词频作答”。
`cage.train.eval_metrics` 因此对每个已实现的评测 batch 做三个确定性最小
编辑：交换 query key 配对的 value、交换一个无关 value、移动整个 query
pair。读法：

1. `cf_value_accuracy`：交换 value 后整批 item 的准确率，是跨臂可比的
   主读数；
2. `cf_distractor_accuracy` / `cf_relocation_accuracy`：安慰剂编辑后的
   整批准确率，应与干净 accuracy 接近；
3. `cf_value_follow_rate` / `cf_value_stubborn_rate`：只在干净答对子集上
   定义，是臂内描述（该子集是输出后选择，不能直接跨臂比较）。

若 accuracy 提升但 `cf_value_accuracy` 没有提升，不能把结果写成“更好地
利用了上下文配对”；只能写成“在该数据分布上准确率更高，但检索操控检验
未同步改善”。详细定义、字段与边界见 `docs/data_methodology.md`。

## 7. 关联暴露与 exposure-stratified accuracy（补充阅读顺序）

整行重复为 0 不代表评测 item 的精确 `(query key, answer value)` 关联没有
在训练中出现过。训练流重放现在同时记录每个 item 的 pair code，并在
`<tag>_split_audit.json` 的 `association_exposure` 中逐 held-out 流报告
`pair_seen_train_rate` / `key_seen_train_rate` / `pair_unseen_key_seen_rate`。
`eval_metrics` 用训练 pair 集合把每个评测 item 分为 `seen` / `unseen`，
在 `association_exposure` 与扁平 `assoc_exposure_*` 字段中分别报告
clean accuracy、value-swap accuracy 与 follow/stubborn rate。

读法：

1. `pair_seen_train_rate` 高说明“干净准确率可以由记忆解释”的机会大，
   本身不是泄漏；它提醒必须读反事实 probe。
2. `unseen` cohort 的 `cf_value_accuracy` 才是必须在上下文中完成检索的
   读数；它接近机会水平时，不能声称模型学会了按 key 检索。
3. `accuracy_gap_seen_minus_unseen` 是描述性差异（两个 cohort 的 item
   组成不同），不是因果效应，也不进入 FDR 族或预注册判定。
4. 若 accuracy 提升集中在 `seen` 且 `seen` 的 `cf_value_follow_rate` 低，
   应写“准确率提升伴随对已见关联的依赖”，而不是“检索能力提升”。
