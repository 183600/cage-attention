# 结果分析：复现预测与假设级证据合成

> 本文档补充 `docs/design_notes.md`（设计宪法，固定项不得修改）。它只说明
> **已有结果应当如何被读出**，不改变任何预注册判定：`supports_hypothesis`、
> FDR 族、SESOI、能力门禁等规则保持原样。新增的两个读数都是**伴随读数**
> （companion），用于暴露"统计上支持"与"可作为机制结论"之间的距离。
>
> 实现：`cage/metrics.py::replication_interval`、
> `cage/experiments.py::_hypothesis_evidence_synthesis` 与
> `ContrastLog.finalize`。测试：`tests/test_replication_synthesis.py`。

## 0. 为什么还需要这一层

项目已经按训练种子配对，报告 bootstrap 95% CI、MDE80、SESOI、精确
sign-flip、jackknife、BCa、评测流 fold/交叉 bootstrap，并用 BH 控制每个
实验族。这些读数回答的是：

1. 当前这次运行的**平均效应**是否与 0（或等价界）相容；
2. 结论对**单个种子**、**区间构造**、**有限评测流切片**有多敏感。

但结果分析还有两个未被显式回答的问题：

- **复现问题**：如果换一批同样数量的训练种子重做一次，效应会不会仍然
  同号？当前 CI 是 `mean ± t·sd/√n`，它随 n 收缩；而一次新实验的均值
  自身也带抽样方差。把当前 CI 当成"重复实验也会显著"是把
  `Var(mean_old)` 当成了 `Var(mean_new - mean_old)`。
- **合取问题**：预注册假设通常是"A 且 B 且 C"，判定表却是逐行
  （row-level）的。只要一行 `SURVIVES`，报告就可能写成"该假设得到支持"，
  把合取偷换成析取。

## 1. 复现区间（same-size replication interval）

### 1.1 定义

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

### 1.2 判定与概率

- `replicates=True`：同规模重复区间整体不跨 0，且在预注册方向上
  （`direction="+"` 要求 `lo > 0`，`direction="-"` 要求 `hi < 0`）。
- 零预测：`mode="equivalence"` 要求区间整体落在 `±bound` 内；
  `mode="noninferiority"` 要求 `lo > -bound`。
- 未注册方向的两侧读数：`replicates` 表示重复区间排除 0，只作描述。
- `prob_replicates`：在 t 假设下，同规模重复实验均值落在预注册方向的
  概率。它不是 p 值，也不进入 FDR 族。
- `prob_single_same_sign`：单颗新种子保持同号的后验式概率，仅作参考。

### 1.3 解释边界

- 该区间假设种子级效应独立、同方差、近似正态；在 5 个种子时这是明确的
  建模假设，不是无分布保证。
- 它是**伴随读数**：一行仍可按预注册规则 `SURVIVES`，同时被标为
  `replication_uncertain=True`。后者表示"当前均值支持该方向，但同规模
  重复在 95% 下可能反向/跨 0"，不是"该行不成立"。
- 若该行存在曲线无效（plateau/diverging/...）的种子，复现区间会在
  `finalize` 中用与曲线敏感性检查相同的**有效种子子集**重算，结果存于
  `replication_curve_valid` / `replication_effective`，避免用未学会任务的
  运行去预测复现。

### 1.4 报告字段

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

## 2. 假设级证据合成（hypothesis-level synthesis）

### 2.1 为什么逐行判定不够

以 H2a 为例，预注册预测是

```text
accuracy_at_longest(calib) > accuracy_at_longest(none)
且 accuracy_drop(calib) < accuracy_drop(none)
且 accuracy_slope_log_len(calib) > accuracy_slope_log_len(none)
```

三行各自通过 FDR 后，才算这个合取成立。如果只有第一行 `SURVIVES`，
把 H2a 写成"supported"就是把"且"读成"或"。同理，H1d 要求每个单因子
消融都显示贡献；H1g 同时要求 calib 的相关为正、query 的相关落在近零界内。

### 2.2 哪些行进入合取

只有**预注册了方向或零预测**的行进入合成：

- `direction` 为 `+`/`-` 的差值行；
- `test` 为 `equivalence` / `noninferiority` 的零预测行。

没有注册方向的两侧 companion（例如 H3b 的 LayerScale 主读数、H2e 的
交互项、H3a 的 post/mix 对照）保持逐行描述，不强行纳入合取；否则会把
"探索性同伴"当成"必须成立"。

### 2.3 状态

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

### 2.4 报告字段

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

## 3. 结果写法（新增部分）

在原有报告清单（见 `docs/design_notes.md` 与各实验输出）之后，结论句还
必须区分：

1. **行级支持**：`SURVIVES` / `EQUIVALENT`；
2. **实际大小**：`minimum_effect_established` 与
   `sesoi_point_estimate_only`（点估计过 SESOI 但区间未过界时，只能写
   "实际大小暂定"）；
3. **复现强度**：`replication_replicates` 与 `prob_replicates`
   （复现区间跨 0 时，写"当前运行支持，复现不确定"，不能写"稳健"）；
4. **假设级结论**：只有 `hypothesis_synthesis[hyp].status == "supported"`
   才能写"该假设得到支持"；`partial` 写"部分支持/合取未确认"；
   `refuted` 写"被证伪"，不能因为其它行显著而改写。

建议句式：

- `supported` + `replication_replicates=True`：
  "在该设计下，该假设的全部预注册预测均得到支持，且同规模重复区间不跨
  0。"
- `supported` + `replication_replicates=False`：
  "当前运行支持该方向，但同规模重复的 95% 区间跨 0（P(同向)≈x），属于
  复现不确定，需更多种子或更大规模重复。"
- `partial`：
  "该假设是合取；n/m 个预注册预测得到支持，另有 k 个未确认，故只能写
  部分支持。"
- `refuted`：
  "至少一个预注册预测方向被显著反向结果证伪；其它支持行不能抵消。"

## 4. 复现与检查

```bash
python3 -m unittest tests.test_replication_synthesis -v
python3 -m cage.metrics          # 统计层自检（含 t 分位数/区间检查）
python3 -m unittest discover -s tests -q
python3 ci_gate.py
```

所有随机过程仍使用显式 canonical seed；新增读数本身是确定性的，不消耗
任何 RNG 流。

## 5. 数据层反事实操控检验（补充阅读顺序）

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
