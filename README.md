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
| `cage/experiments/` | E1/E2/E3 实验运行器、结果记录与来源校验 |
| `tests/` | 标准库 `unittest` 测试套件 |
| `ci_gate.py` | 语法、设计固定项与冒烟测试门禁 |
| `docs/design_notes.md` | 固定设计决策与实验契约 |

## 快速开始

安装运行依赖：

```bash
python3 -m pip install -e .
```

运行质量门禁与测试：

```bash
make check          # ci_gate.py + unittest
make gate           # 只运行设计门禁与冒烟测试
make test           # 只运行 unittest
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

## 实验设计

- **E1 门控对比**：比较联想回忆准确率；对 `query` / `calib` 报告门控值与
  注意力熵、max-prob 的逐层 Pearson 相关（Fisher z 平均）；分析
  “假集中”（max-prob 高但回答错误）时门控是否关闭；`null` 变体另报
  `null_mass`。
- **E2 长度泛化**：固定训练长度，在按 2 倍间隔扩展的长度扫描上评估；校准门控
  显式修正 `log T` 引起的熵增长，因此预期退化更慢。报告各长度准确率以及
  熵、max-prob、门控值的漂移。
- **E3 归一化/激活协同**：`query` / `calib` × `pre` / `post` / `mix` /
  `seednorm` × LayerScale（无 / `1e-2`）。报告逐层残差范数与
  `gate_depth_ratio`（深层门控均值 / 浅层门控均值），用于观察深层是否更愿意
  写入有效内容。

## 关键指标

| 指标 | 读法 |
|---|---|
| `gate_vs_entropy` / `gate_vs_maxprob` | 门控是否真正使用了注意力分布信息；`query` 不应与熵强相关，`calib` 应显著相关 |
| `false_conc_rate` | 高 max-prob 样本中的错误率 |
| `gate_on_false_conc` vs `gate_on_correct` | 门控在错误集中时是否更倾向于关闭 |
| `deep_gate_ratio` | 大于 1 表示深层比浅层更愿意写内容 |
| `null_mass` | `null` 变体中 null 槽位分走的注意力权重占比 |

## 复现性与结果校验

- `make check` 依次运行 `ci_gate.py`（语法、设计固定项、冒烟）与 `tests/`
  下的标准库 `unittest` 套件；测试不依赖 pytest 或 numpy。
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
