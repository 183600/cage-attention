# CAGE-Attention：校准门控注意力 × 归一化协同

**中文** | [English](README_en.md)

面向门控注意力机制研究的受控实验代码库。它围绕两个评审关切展开：

1. 门控注意力是否只是已有门控方案的小变体？
2. 深层残差流贡献不足时，门控与归一化应如何协同？

核心思想是：**门控不应只依赖 query（Gated Attention 风格），还应按注意力分布的
集中度（熵）与序列长度做校准**，并与 Pre/Post/Mix-LN、SeeDNorm 类动态归一化
和 LayerScale 联合设计，使深层残差流能够写入有效内容。

设计边界与实验契约见 [docs/design_notes.md](docs/design_notes.md)。其中标记
【固定】的条目用于保证跨实现、跨提交的数值路径一致，修改代码时必须遵守。

## 目录结构

| 路径 | 作用 |
|---|---|
| `cage/model.py` | CAGE 模型：4 种 gate 变体 × 4 种 norm 方案 × 可选 LayerScale |
| `cage/data.py` | 联想回忆（associative recall）合成任务与长度扫描 |
| `cage/train.py` | 训练、评估、门控统计与假集中分析 |
| `cage/metrics.py` | 仅依赖 torch 的统计与效应量工具 |
| `cage/baselines.py` | 规则、线性探针与深度阶梯等简单基线 |
| `cage/experiments/` | E1/E2/E3 实验运行器、结果记录与来源校验 |
| `tests/` | 标准库 `unittest` 测试套件 |
| `ci_gate.py` | 语法、设计固定项与冒烟测试门禁 |
| `docs/design_notes.md` | 固定设计决策与实验契约 |

## 环境与安装

需要 Python 3.10+ 与 PyTorch。安装运行依赖：

```bash
python3 -m pip install -e .
```

只在 CI 中复现固定依赖时，可使用：

```bash
python3 -m pip install -r requirements-ci.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

## 快速开始

运行质量门禁与测试：

```bash
make check          # ci_gate.py + unittest
make gate           # 只运行设计门禁与冒烟测试
make test           # 只运行 unittest
```

也可以直接调用门禁：

```bash
python3 ci_gate.py
python3 -m unittest discover -s tests
```

先用最小配置检查完整流程：

```bash
python3 -m cage.experiments --only sanity --quick --out results_quick
```

再按需运行正式实验：

```bash
python3 -m cage.experiments --only e1 --seeds 5 --out results
python3 -m cage.experiments --only e2 --seeds 5 --out results
python3 -m cage.experiments --only e3 --seeds 5 --out results
```

`--quick` 只用于流水线检查；正式结论应使用默认（非 quick）设置并保留完整的
`--out` 目录。实验产物默认写入 `results/`，该目录已加入 `.gitignore`。

## 直接在代码中调用

```python
from cage.model import build
from cage.train import collect_stats, eval_accuracy, train_model

model = build({
    "vocab_size": 64,
    "d_model": 32,
    "n_heads": 4,
    "n_layers": 2,
    "max_len": 64,
    "gate": "calib",
})

hist = train_model(model, steps=3, batch=4, n_pairs=6, vocab=64,
                   log_every=1)
accuracy = eval_accuracy(model, 6, 64, "cpu", batch=4, batches=1)
stats = collect_stats(model, 6, 64, "cpu", batch=4, batches=1)
print(hist[-1], accuracy, sorted(stats))
```

`train_model` 的 `seed` 同时约束数据流与 dropout 随机性；评估与统计默认使用
固定种子，因此同一配置可以重复对照。

## E1：严格对比的 4 个变体

| gate | 说明 |
|---|---|
| `none`  | 标准 Attention（无门控） |
| `query` | 纯 query 依赖门控（Gated Attention 风格），`σ(W_g x)` |
| `calib` | 本项目的长度 + 集中度校准门控，`σ((W_g x − a·log T − c·H̄)/τ)`，其中 `H̄` 为按 `log T` 归一化的注意力熵 |
| `null`  | 显式空注意力 / 稀疏选择基线：可学习的 null KV 槽位与真实 token 竞争 |

`calib` 的消融开关保持默认值时复现完整门控；它们只用于 E1/E2 的单因素对照：

| 配置项 | 默认 | 作用 |
|---|---|---|
| `calib_use_len` | `True` | 是否使用长度项 `a·log T` |
| `calib_use_ent` | `True` | 是否使用集中度项 `c·H̄` |
| `calib_ent_norm` | `True` | 集中度按 `log T` 归一化；`False` 表示原始行均值熵 |
| `calib_pos_norm` | `False` | 位置混淆消融：除数与长度项改用每行可见 token 数 `log(max(t+1,2))`；末位 query 行两臂严格一致 |
| `calib_shuffle_ent` | `False` | 主动对照：只打乱门控读取的熵位置，保留同分布信号 |

## 实验设计

- **E1 门控对比**：比较联想回忆准确率；对 `query` / `calib` 报告门控值与
  注意力熵、max-prob 的逐层 Pearson 相关（Fisher z 平均）；分析
  “假集中”（max-prob 高但回答错误）时门控是否关闭，另报门控分布分位数
  （误关正确答案的代价）；`null` 变体另报 `null_mass`，并给出含 null
  槽位的全分布假集中读出。E1 记录同时携带学得的校准系数
  （`calib_coefficients`，与 E2 同一读出）。
- **E2 长度泛化**：固定训练长度，在按 2 倍间隔扩展的长度扫描上评估；校准门控
  显式修正 `log T` 引起的熵增长，因此预期退化更慢。报告各长度准确率以及
  熵、max-prob、门控值的漂移；附带位置编码外推控制
  （`pos_extrapolation_control`，见下方方法学边界）。
- **E3 归一化/激活协同**：`query` / `calib` × `pre` / `post` / `mix` /
  `seednorm` × LayerScale（无 / `1e-2`）。报告逐层残差范数与
  `gate_depth_ratio`（深层门控均值 / 浅层门控均值）；另报跨归一化方案
  可比的逐层相对写入量（`attn_write_ratio` / `mlp_write_ratio`）与
  逐层关闭干预（`layer_knockout_eval`，见下方方法学边界）。

## 关键指标

| 指标 | 读法 |
|---|---|
| `gate_vs_entropy` / `gate_vs_maxprob` | 门控是否真正使用了注意力分布信息；`query` 不应与熵强相关，`calib` 应显著相关 |
| `false_conc_rate` | 高 max-prob 样本中的错误率（null 变体另报全分布 `false_conc_rate_full` 与 `mean_null_mass_on_high`） |
| `gate_on_false_conc` vs `gate_on_correct` | 门控在错误集中时是否更倾向于关闭（另报两组的门控分位数 `*_q10/q50/q90`） |
| `gate_depth_ratio` | 大于 1 表示深层比浅层更愿意写内容（门值比率，不是层重要性；重要性证据看 `layer_knockout_eval` 与逐层写入量） |
| `null_mass` | `null` 变体中 null 槽位分走的注意力权重占比（null KV 可学习，非固定空槽） |

## 方法学边界与对照

本节明确各结论的证据边界与已知限制，避免超出对照设计的解读。

1. **相关 ≠ 机制**。`calib` 门控公式内含熵项，因此 gate 与熵的相关
   部分是构造便有的。支持“校准门控识别错误”的证据链是消融与主动对照
   （H1d 单项消融、H1f 打乱熵安慰剂，容量强制匹配）加反事实探针，
   不是裸相关系数。
2. **熵归一化的位置语义**。登记门控对每一行都用全序列 `log T` 作除数；
   因果第 t 行只能看到 t+1 个键，其熵上限是 `log(t+1)`。本任务只监督
   且只统计末位 query 行（该行可见键数恰为 T，`log T` 精确），但前向中
   非末行的门控确实混入位置信息——可用 `calib_pos_norm=True` 消融检验
   （末行两臂严格一致，仅非末行不同）。
3. **E2 的位置编码混淆与控制**。模型用可学习绝对位置 embedding，固定
   长度训练后扫描更长序列时，超出训练长度的位置向量从未被训练。
   长度退化可能来自门控/注意力行为，也可能来自未训练位置向量。
   E2 现在对每个外推长度额外评估 `fold`（位置折叠回训练区间）与
   `zero`（外推位置零化）两种读法（`pos_extrapolation_control`，
   探索性伴随，同一封闭评估流）：若折叠能恢复大部分退化，则退化
   主要来自位置外推而非机制本身。
4. **参数可识别性**。门控中只有 `calib_c × calib_scale` 的乘积起作用，
   因此 `calib_scale` 是固定超参（buffer，非可学习参数），可学习的
   集中度系数就是 `calib_c`（`c_effective = calib_c × calib_scale`）。
   E1 与 E2 均记录学得系数（`a_mean` / `c_effective_mean` 等，H2f 操纵检查）。
5. **初始门开度匹配**。`query` 门偏置初始化为 2（开度 ≈0.88），
   `calib` 为 6：在默认训练长度（T=34）下减去 `a·log T + c·H̄` 后两臂
   初始开度相近（≈0.8–0.9），这是偏置取值的依据；两侧参数量差
   （calib 每层多 3 个标量）在对照日志中展示。
6. **`null` 变体的条件量读法**。`null` 变体报告的熵 / max-prob 是
   “未选 null”条件下的真实 token 分布（与其它变体可比），
   `null_mass` 同步报告；`false_conc_rate_full` 给出含 null 槽位的
   全分布版本，避免把“几乎不写但写错”误读为“自信地写错”。
7. **E3 跨归一化方案的残差范数不可直接比较**。post-norm 的
   `resid_norm` 是 LN 之后的范数（量级被归一化钉住），pre/seednorm
   是原始残差流范数（随深度增长）。跨方案比较“深层写入”请用
   `attn_write_ratio` / `mlp_write_ratio`（各分支相对其写入流的范数比）
   与 `layer_knockout_eval`（逐层替换为恒等后测准确率，因果性证据）。
   LayerScale 的注意力/MLP 两分支是独立参数（ls1/ls2），其学得值
   可分别读出。
8. **新颖性边界**。query 门控属于 Gated Attention 家族
   （*Gated Attention for Large Language Models: Non-linearity,
   Sparsity, and Attention-Sink-Free*, Qiu et al., 2025,
   arXiv:2505.06708，NeurIPS 2025；已研究 query 依赖的 sigmoid 门控、
   稀疏门控与长上下文外推）。
   本项目的新颖性主张仅限熵 × 长度校准项，且必须由 H1d/H1f/H2g 消融
   支撑，不能把“门控有效”或“门控改善长上下文”作为新颖性主张。
9. **手写核的梯度验证**。四个自定义 autograd 核（`_SoftmaxInPlace` /
   `_RowEntropy` / `_CalibAttn` / `_NullAttn`）在
   `tests/test_gradcheck.py` 中以 float64 gradcheck、参考实现对照、
   多块分解路径与训练/评估路径一致性四种方式验证。

## 复现性与结果校验

- `make check` 依次运行 `ci_gate.py`（语法、设计固定项、冒烟）与 `tests/`
  下的标准库 `unittest` 套件；测试不依赖 pytest 或 numpy。
  `tests/test_gradcheck.py` 对手写 autograd 核做双精度 gradcheck 与
  参考实现对照。
- 所有 RNG 与数据流使用同一个 32 位 canonical seed：`s` 与 `s + 2**32`
  表示同一次运行；bool 或小数 seed 会被拒绝，而不是静默截断。
- 每条结果记录都携带代码版本或源码哈希、完整配置、数据流指纹，以及线程与
  确定性开关等环境快照。
- `artifact_index.json` 是跨多次调用累积的制品清单。以下命令可独立校验历史
  结果目录：

```bash
python3 -m cage.experiments --verify-artifacts --out results
python3 -m cage.experiments --check-environment --out results
python3 -m cage.experiments --verify-provenance --out results
python3 -m cage.experiments --check-source --out results
```

`make verify` 会依次执行环境、制品、来源与源码检查。

## 设计约束

代码修改必须遵守 [docs/design_notes.md](docs/design_notes.md) 中的【固定】条目。
提交前至少运行一次 `python3 ci_gate.py`；涉及实验流程的改动还应运行完整测试
套件。
