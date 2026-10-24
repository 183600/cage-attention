# CAGE-Attention: 校准门控注意力 × 归一化/激活协同

**中文** | [English](README_en.md)


针对"稀疏/门控注意力是不是已有门控的小变体"以及"深层贡献不足"两个 reviewer
关切的实验代码库。核心思想：**门控不应只依赖 query（Gated Attention 风格），
还要按注意力分布的集中度（熵）与序列长度做校准**；并与 Pre/Post/Mix-LN、
SeeDNorm 类动态归一化、LayerScale 联合设计，让深层残差流"愿意写有效内容"。

## 结构

```
cage/
  model.py   # CAGE 模型：4 种 gate 变体 × 4 种 norm 方案 × 可选 LayerScale
  data.py    # 联想回忆（associative recall）合成任务，可度量"正确答案"
  train.py   # 训练、评估、门控-熵/最大概率相关性、假集中分析
```

## 严格对比的 4 个变体（E1）

| gate | 说明 |
|---|---|
| `none`  | 标准 Attention（无门控） |
| `query` | 纯 query-dependent 门控（Gated Attention 风格），`σ(W_g x)` |
| `calib` | ** ours：长度 + 集中度校准门控**，`σ((W_g x − a·log T − c·H̄)/τ)`，H̄ 为按 log T 归一化的注意力熵 |
| `null`  | 显式空注意力 / 稀疏选择基线（可学习 null KV 槽位与真实 token 竞争） |

## 实验

- **E1 门控对比**：联想回忆准确率、门控值 vs 注意力熵 / max-prob 的
  Pearson 相关、以及"假集中"（max-prob 高但回答错误）时门控是否关闭——
  校准门控应在假集中时给出更低的门控值。
- **E2 长度泛化**：固定训练长度，在 2 倍间隔的长度扫描上评估；校准门控
  显式修正 log T 熵增长，应退化更慢，并报告长序列下的熵/max-prob/门控值漂移。
- **E3 归一化/激活协同**：`query`/`calib` × `pre`/`post`/`mix`/`seednorm`
  × LayerScale(无/1e-2)，报告逐层残差范数增长与
  **deep/early gate ratio**（深层门控均值 / 浅层门控均值），衡量深层是否
  更愿意"写有效内容"。

## 关键指标解读

- `gate_vs_entropy` / `gate_vs_maxprob`：门控是否真正利用注意力分布信息
  （`query` 门控理论上不应与熵强相关；`calib` 应显著相关）。
- `false_conc_rate`：高 max-prob 样本中的错误率；配合
  `gate_on_false_conc` vs `gate_on_correct` 看门控的"自我怀疑"能力。
- `deep_gate_ratio > 1`：协同归一化后深层比浅层更愿意写内容，说明缓解了
  Pre-LN 深层贡献不足的问题。

## 复现性与工程检查

- `make check`：先运行 `ci_gate.py`（语法 + 设计固定项 + 冒烟），再运行
  `tests/` 下的标准库 `unittest` 套件（种子规范化、数据契约、环境快照与
  结果目录完整性）。测试不依赖 pytest/numpy。
- 所有 RNG 与数据流使用同一个 32 位 canonical seed：`s` 与 `s + 2**32`
  是同一次运行；bool/小数 seed 会被拒绝而不是静默截断。
- 每次实验的结果记录附带代码版本/源码哈希、完整配置、数据流指纹、
  线程与确定性开关等环境快照；`artifact_index.json` 是跨多次调用的累积
  清单，`python3 -m cage.experiments --verify-artifacts --out <dir>` 可校验
  历史结果文件是否被截断或篡改。`--check-environment` 对照当前软件/数值
  环境，`--check-source` 对照当前 `cage/*.py` 源码哈希；`make verify`
  依次执行环境、制品、来源与源码检查。
