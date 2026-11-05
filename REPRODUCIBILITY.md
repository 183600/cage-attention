# CAGE-Attention 可复现性指南

## 概述

本文档描述了 CAGE-Attention 项目的可复现性框架、工程规范以及如何在不同环境中完整复现实验结果。项目采用**假设驱动**、**受控对比**、**全链路溯源**的科研方法论，确保每一个实验结论都可以被独立验证。

---

## 1. 核心可复现性保证

### 1.1 确定性种子体系

所有随机性来源使用统一的 32 位规范种子：

```python
from cage.repro import seed_everything, canonical_seed

# 同一运行的所有 RNG（Python、NumPy、PyTorch CPU/CUDA）消费同一个 32 位键
info = seed_everything(42, deterministic=True)
# info 包含：requested_seed, seed_key, python_random, numpy, torch, cuda_devices, 
# deterministic_algorithms, cudnn, tf32, torch_num_threads, pythonhashseed
```

**关键属性**：
- `s` 和 `s + 2**32` 表示同一次运行（由 `canonical_seed` 统一规范化）
- 布尔值、浮点数等非整数种子被**拒绝**而非静默截断
- 数据流使用独立的 `torch.Generator`（通过 `stream_generator` 创建），不读取全局 RNG 状态

### 1.2 环境指纹与漂移检测

每次运行自动记录完整的软件/数值环境：

```bash
# 自动捕获的环境字段
python_full_version, torch, torch_git_version, platform, machine,
cpu_model, torch_num_threads, deterministic_algorithms, cublas_workspace_config,
tf32, numeric_env (OMP/MKL/OPENBLAS/NUMEXPR/CUBLAS 线程数),
pythonhashseed, hash_randomization, argv, cwd, repo_root, git_sha, git_dirty
```

**环境兼容性判定**：
- **Critical drift**（不兼容）：PyTorch 版本、Python 版本、依赖清单、数值环境变量变化
- **Advisory drift**（兼容但警告）：硬件平台、CPU 型号等非数值字段变化
- 结果目录包含 `environment_history.jsonl`，追踪每次调用的环境漂移

**单一事实来源**：环境记录只有一份实现——`cage.experiments._environment_record`。
`capture_environment.py` 是它的命令行外壳，写出的文件与结果目录里
`environment.json` 的环境块逐字段同构，`environment_sha256` 因此可以直接互比
（两种文件形态都能读）：

```bash
python3 capture_environment.py --output /tmp/env.json
python3 capture_environment.py --compare /tmp/env.json     # 与 --check-environment 同一判据
python3 -m cage.experiments --check-environment --out results
```

> 历史教训：独立实现曾让同一台机器产生两个互不相容的 `environment_sha256`，
> "比对两个环境"实际上在比两个不相关的对象。

**声明 vs 实装**：环境记录说明"装了什么"，`--check-dependencies` 回答
"装的是不是我们声明的那个"（详见 §4.4）。

### 1.3 制品完整性链

```
artifact_index.json (累积索引)
├── 每个文件的：相对路径、字节数、SHA-256、行数(JSONL)
├── 保留历史条目（未重写的文件保持原证书）
├── previous_index_error 标记历史索引不可读
└── 当前调用的 run_id、git_sha、setup_sha256、source_tree_sha256

verify-artifacts: 逐文件校验 size/hash/行数
verify-provenance: 语义级校验（每行 setup_sha256 从自身 config/data/hypotheses 重算）
verify-runs:      拒绝未完成/失败的运行、关键环境漂移历史
```

### 1.4 来源溯源

每条结果记录携带完整的来源信息：
- `run_id`：唯一运行标识
- `git_sha` / `git_dirty`：代码版本
- `source_tree_sha256`：源码树内容哈希
- `setup_sha256`：配置/数据/假设注册表的内容哈希
- `hypotheses_sha256`：预注册假设注册表的内容哈希
- `data_fingerprints`：训练/验证/测试/统计流的指纹

---

## 2. 实验协议与预注册

### 2.1 假设注册表

所有科学假设在 `cage/experiments/_config.py:HYPOTHESES` 中预注册，包含：

```python
HYPOTHESES = {
    "H1a": {
        "experiment": "e1",
        "question": "Does the calibrated gate use attention concentration / length information beyond the query?",
        "hypothesis": "...",
        "prediction": "...",
        "decision": "paired bootstrap 95% CI excludes 0 after BH FDR control (q=0.05)",
        "sesoi": {"accuracy": 0.01, "abs_gate_vs_entropy": 0.05, "abs_gate_vs_maxprob": 0.05},
        "direction": {"abs_gate_vs_entropy": "+", "abs_gate_vs_maxprob": "+"},
        "sesoi_test": {"accuracy": "noninferior"},
    },
    # ... HB0-HB7, H1a-H1g, H2a-H2g, H3a-H3c
}
```

**预注册要素**：
- **SESOI** (Smallest Effect Size of Interest)：统计显著但效应量小于 SESOI 被判定为"实际上可忽略"
- **方向性预测**：反方向的显著效应直接**反驳**假设
- **等价性/非劣效性检验**：空假设预测使用 bootstrap TOST，CI 必须完全落在界内
- **FDR 家族**：每个实验的对比构成一个家族，Benjamini-Hochberg 控制 (q=0.05)

### 2.2 设计计划冻结

每个网格在首次运行前生成 `<tag>_design_plan.json`：
- 将 SESOI 换算为计划种子数能容忍的最大种子级 SD
- 报告配对差异方差（而非单臂方差）、有限评估流的二项式下界
- 应用基线阶梯观测到的 curve-invalid 率进行损耗修正
- 翻译注册方向为精确配对符号翻转检验的可达下界
- **冻结机制**：首个结果行存在后计划锁定；改变种子/SESOI/FDR 水平抛出 `DesignPlanError`
- 历史版本保存在 `<tag>_design_plan_history.jsonl`

### 2.3 单因素对照的设计期证书

每个网格在**首次训练之前**生成 `<tag>_single_factor_design.json`：
- 用解析后的 `Cfg` 计算每个实验单元的因子向量，声明该实验的受控对比对
- 逐对检查"恰好一个模型因子不同"；多因子对比对会在开跑前抛 `ValueError`，网格不启动
- 决策行从证书声明的对比对取臂，未声明的对比对会中止运行（执行与设计不漂移）
- 双因子角落（`calib_no_terms`）只通过已注册的 2×2 聚合行读取，不作为单因素决策行
- 故意非受控的探索对比（安慰剂 vs `query`）记在 `exploratory_pairs`（`controlled: false`）

### 2.4 随机化完全区组设计

```python
# 每个训练种子 = 一个区组
# 每个区组内：所有声明的 cell 各跑一次，位置随机排列
# 排列确定性来源：schedule_seed + block_index（不触碰全局 RNG）
# 记录：<tag>_run_order.json（计划）、<tag>_run_order_realized.json（实测+位置效应诊断）
```

**位置效应诊断**：区组内槽位回归斜率的随机化检验（2000 次排列），BH 校正后若显著则警告该实测调度仍存在时间混淆。

---

## 3. 标准复现工作流

### 3.1 环境准备

```bash
# 1. 克隆仓库
git clone https://github.com/your-org/cage-attention.git
cd cage-attention

# 2. 安装精确依赖（CI 使用的冻结版本）
python3 -m pip install -r requirements-ci.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu

# 3. 确认装出来的版本确实等于声明的固定版本
make verify-deps

# 4. 验证环境指纹
python3 -m cage.experiments --check-environment --out results
```

### 3.2 运行质量门禁

```bash
# 语法 + 设计固定项 + 冒烟测试
make gate

# 完整测试套件（标准库 unittest，无 pytest 依赖）
make test

# 组合检查
make check

# 正式实验前的完整预检（依赖一致性 + 门禁 + 测试）
make preflight
```

### 3.3 运行实验

```bash
# 快速流水线检查（非正式结论）
python3 -m cage.experiments --only sanity --quick --out results_quick

# 正式实验（保留完整 --out 目录）
python3 -m cage.experiments --only e1 --seeds 5 --out results
python3 -m cage.experiments --only e2 --seeds 5 --out results
python3 -m cage.experiments --only e3 --seeds 5 --out results

# 或一次性运行所有
python3 -m cage.experiments --only all --seeds 5 --out results
```

### 3.4 验证结果目录

```bash
# 完整验证链
make verify OUT=results

# 单项验证
make verify-env OUT=results        # 环境兼容性
make verify-artifacts OUT=results  # 制品完整性
make verify-provenance OUT=results # 来源溯源
make verify-runs OUT=results       # 运行完整性+环境同质性
make verify-source OUT=results     # 归档代码哈希与当前检出比对

# 声明 vs 实装：这份结果当初是不是在声明的依赖集下跑出来的
python3 -m cage.experiments --check-dependencies --out results
```

### 3.5 独立重放（Replication）

```bash
# 从已完成的结果目录生成冻结清单并重跑
python3 -m cage.experiments --replay-manifest results/e1_manifest.json --out results_replay

# 比较源目录与重放目录的科研载荷
python3 -m cage.experiments --verify-replay results --out results_replay
# 或
make verify-replay SOURCE=results REPLAY_OUT=results_replay
```

**重放比较范围**：覆盖该实验的全部 JSON/JSONL 证据，仅剔除：
- run_id、时间戳、git 修订与脏标记、启动 argv/cwd、哈希随机化种子
- 覆盖这些字段的 setup/环境指纹
- 成本伴随读出的墙钟测量（`latency_ms_*` / `throughput_sequences_per_s`）：
  它们测量的是执行重放的那台机器，每次重放都会有几个百分点的差异，不是科研载荷；
  解析成本（FLOPs、参数量）仍在比较范围内
- 数值差异、缺失证据、未完成重放、无法关联的 replay.json 均返回非零

---

## 4. 工程规范

### 4.1 代码组织原则

```
cage/
├── model.py          # 模型定义、四种 gate、四种 norm、LayerScale、手写 autograd 核
├── data.py           # 联想回忆合成任务、长度扫描、流审计（拆分为 _data_task/_data_audit/_data_selfcheck）
├── train.py          # 训练循环、评估、统计量、曲线裁决、假集中/错误挖掘
├── metrics.py        # 仅依赖 torch 的统计工具：bootstrap CI、FDR、等价检验、功效分析
├── baselines.py      # 规则基线、线性探针、深度阶梯
├── repro.py          # 种子一切、确定性配置、流生成器
├── runlock.py        # 结果目录锁（防并发写入）
├── efficiency.py     # 显存/吞吐测量
└── experiments/      # E1/E2/E3 运行器、设计、对比、制品、环境、审计
    ├── _config.py    # 假设注册表、种子校验、Settings
    ├── _analysis_plan.py # 研究级分析计划：确认性对比集、声明、冻结、合并族裁决
    ├── _design.py    # 运行调度、难度标记、覆盖率
    ├── _planning.py  # 设计计划冻结、功效换算
    ├── _study_design.py # 研究级设计预检：腿可达性、算力预算
    ├── _contrasts.py # 配对 bootstrap、FDR、SESOI、符号翻转
    ├── _synthesis.py # 假设层面综合、复制区间、FCR
    ├── _contrastlog.py / _contrastlog_finalize.py # 决策日志与家族内裁决
    ├── _artifacts.py # 结果写入、制品索引、来源审计
    ├── _environment.py # 环境指纹、漂移检测、历史
    ├── _gates.py     # 基线门禁、曲线裁决
    ├── _training.py  # 单运行训练+评估
    ├── _audits.py    # 数据/生成器/分裂/校准审计
    ├── _baselines.py # 基线阶梯、规则/探针/负控制
    ├── _e1.py / _e1_runner.py # E1 细胞计划、逐变体调参、lr 敏感性、运行器
    ├── _e2.py / _e2_runner.py # E2 长度扫描、系数读出、剂量反应、运行器
    ├── _single_factor.py # 单因素设计期证书、对比对守卫
    ├── _e3_cli.py    # E3 网格、逐层写入、knockout、CLI 入口
    └── __init__.py   # 片段加载、方法学总纲
```

### 4.2 设计固定项（CI 强制）

`docs/design_notes.md` 中标记 `【固定】` 的决策由 `ci_gate.py` 源码检查强制：

| 编号 | 决策 | 检查内容 |
|------|------|----------|
| D1 | 统一手写注意力 kernel | 禁止 `scaled_dot_product_attention`、`use_sdpa` 双路径 |
| D2 | 模块级因果掩码缓存 | `_causal_mask_cache` 存在 |
| D3 | 查询分块公式括号 | `max_rows = (1 << 20) // (bh * T)` 显式括号 |
| D4 | `calib_tau` 初始化 | `math.log(math.expm1(cfg.calib_tau_init))` |
| D5 | 优化器工厂回退链 | `_make_adamw` 尝试 `fused=True` → `foreach=True` → 默认 |
| D6 | 四个手写 autograd 核 | `_SoftmaxInPlace`、`_RowEntropy`、`_CalibAttn`、`_NullAttn` 存在；MLP 用 `_GeluLinear` |

### 4.3 测试规范

- **仅使用标准库 `unittest`**，无 pytest/numpy 依赖
- 测试分类（完整清单见 `tests/` 目录）：
  - `test_reproducibility.py`：种子规范化、环境记录
  - `test_reproducibility_contract.py`：环境指纹、漂移、历史
  - `test_environment_record.py`：环境块可重建、git 溯源
  - `test_provenance_audit.py`：语义溯源、run_id 一致性
  - `test_gradcheck.py`：四个手写核的 float64 gradcheck、参考实现对照、多块分解一致性
  - `test_methodology_controls.py`：方法学对照（HB0-HB7、H1f 安慰剂等）
  - `test_ablation_dose_response.py`：消融/剂量反应
  - `test_false_coverage.py`：FCR 区间
  - `test_counterfactual_eval.py`：反事实评估
  - `test_run_audit.py`：运行审计
  - `test_replication_synthesis.py`：复制综合
  - `test_design_preparation.py`：区组调度、功效计划、曲线失效损耗
  - `test_single_factor_design.py` / `test_single_factor_contract.py`：单因素设计期证书
  - `test_study_design.py`：研究级分析计划（注册表一致性、腿可达性、冻结、IUT 裁决、算力预算）
  - `test_dependency_conformance.py`：声明依赖 vs 实装依赖（版本比较、归档记录重算、CLI 退出码）
  - `test_documentation_integrity.py`：文档-代码漂移守卫（引用的版本、CLI 参数、make 目标、容器配方、文档链接）

### 4.4 依赖管理

```toml
# pyproject.toml
[project]
dependencies = ["torch>=2.0"]

[project.optional-dependencies]
dev = ["pytest>=7", "ruff>=0.5"]

# requirements-ci.txt (CI 冻结版本)
torch==2.14.1
# 仅在 CI 复现固定依赖时使用
```

**两处声明的分工**：`pyproject.toml` 声明"本包需要什么"（版本范围），
`requirements-ci.txt` 声明"本次研究用什么"（冻结版本）。冻结版本是**唯一
事实来源**：容器镜像也从它安装依赖，文档不再重复写版本号。

**依赖一致性检查**（新增）：环境记录会写下"装了什么"，但不会自动回答
"装的东西是不是我们声明的那个"。`--check-dependencies` 做这件事：

```bash
make verify-deps     # 等价于 python3 -m cage.experiments --check-dependencies
```

- 比对**当前解释器**与声明（`requirements-ci.txt` 的 `==` 固定版本、
  `pyproject.toml` 的运行期要求）；
- 带 `--out results` 时，额外用结果目录里归档的环境记录重算一遍，回答
  "这份历史结果当初是不是在声明的依赖集下跑出来的"；
- 退出码：`0` 一致、`1` 不一致（版本不符或缺包）、`2` 无声明可查；
- 规范比较器只认得 `== / != / >= / <= / > / <`，其余写法（如 `~=`）标为
  **未判定**并如实报告，绝不当作通过。

正式实验前的推荐顺序：`make preflight`（= `verify-deps` + `check`）。
`run_experiment.py` 也会在预检阶段打印同一份一致性结论（不阻断，
但会明确警告结果不可与冻结配方下的运行直接比较）。

---

## 5. 结果解读规范

### 5.1 对比读取规则

每个预注册对比输出：
```
mean ± std (n seeds)
paired bootstrap 95% CI: [lo, hi]
p_value (percentile bootstrap)
seeds_used / seeds_dropped (curve-invalid 排除)
sign_flip_p (exact paired sign-flip), sign_flip_floor
BCa CI, jackknife flag
FDR-adjusted: supported / not supported
SES0I check: practically negligible / above SESOI / below -SESOI
state: "supported" / "refuted" / "inconclusive" / "practically negligible" / "ineligible"
```

**决策优先级**：
1. `curve-invalid` 运行被配对差异排除（记录 `seeds_dropped`）
2. 配对种子数 < `min_seeds_for_claim` (默认 3) → `ineligible`
3. FDR 校正后 CI 不排除 0 → `inconclusive`
4. FDR 显著但效应量 < SESOI → `practically negligible`
5. FDR 显著且方向与预注册相反 → `refuted`
6. FDR 显著、方向一致、效应量 ≥ SESOI → `supported`

### 5.2 假设层面综合

`_synthesis.py` 在不改变行级裁决的前提下给出三个伴读：
1. **假设合取**：某假设的所有注册行均 `supported` 才标记 `supported`；部分通过=`partial`，有反驳=`refuted`
2. **同量复制区间**：未来同量级研究均值的 t 区间（`sqrt(2/n)` 而非 `sqrt(1/n)`）与保持符号概率
3. **FCR 区间**：被 BH 选中的集合的假覆盖率区间（BY 调和因子分母）

### 5.3 方法学边界（避免过度解读）

README.md "方法学边界与对照" 节明确列出 9 条边界，关键者：
1. **相关 ≠ 机制**：`gate_vs_entropy` 相关部分源于构造；证据链靠消融(H1d) + 安慰剂(H1f) + 反事实探针
2. **熵归一化的位置语义**：非末行门控混入位置信息；E1 训练位置中性臂 `calib_pos_norm`（H1h，末位 query 行两臂一致）并以单侧非劣性判定
3. **E2 位置编码混淆**：`pos_extrapolation_control`（fold/zero 读法）区分机制退化 vs 位置外推
4. **参数可识别性**：只有 `c × calib_scale` 乘积起作用；`calib_scale` 固定为 buffer
5. **新颖性边界**：query 门控属 Gated Attention 家族；本项目新颖性仅限熵×长度校准项，需 H1d/H1f/H1h/H2g 支撑

---

## 6. 扩展与贡献指南

### 6.1 新增实验/假设

1. 在 `HYPOTHESES` 注册新假设（含 SESOI、方向、sesoi_test、decision）
2. 在 `_e1.py`/`_e2.py`/`_e3_cli.py` 或新建 `_e4.py` 中实现细胞计划与运行器
3. 在 `__init__.py` 的 `_FRAGMENTS` 注册新片段
4. 运行 `make check` 确保门禁通过
5. 更新 `README.md` 实验设计表与指标表

### 6.2 改依赖 / 改文档时的纪律

1. **依赖版本只在 `requirements-ci.txt` 写一次**。文档、容器配方、CI
   安装命令都引用它，不复制版本号；`tests/test_documentation_integrity.py`
   会检查文档里出现的 `torch==...` 仍然是声明的那个版本。
2. **文档里出现的命令必须真的存在**：CLI 参数由
   `tests/test_documentation_integrity.py` 对照 `_e3_cli.py` 的解析器核对，
   `make <目标>` 对照 `Makefile` 核对（历史上协议文档曾写了一个并不存在的
   lr 敏感性开关）。
3. **容器配方必须装同一份冻结依赖**：`Dockerfile.repro` 不得写死 torch
   版本，`pip install -e .` 必须带 `--no-deps`，并保留构建期的版本自检。
4. 改完运行 `make check`（门禁 + 全量测试，含上述漂移守卫）。

### 6.3 修改固定设计

1. 直接修改 `docs/design_notes.md` 对应条目
2. 同步移动固定标签：`git tag -f design-pin <新提交sha> && git push origin design-pin`
3. 未移动标签的修改会在下一次 CI 被自动回滚

### 6.4 数值核修改

四个手写 autograd 核（`_SoftmaxInPlace`、`_RowEntropy`、`_CalibAttn`、`_NullAttn`）任何修改必须：
- 通过 `tests/test_gradcheck.py` 的 float64 gradcheck
- 通过参考实现对照（数值一致性）
- 通过多块分解路径与训练/评估路径一致性
- 不改变 `_EPS = 1e-9` 量级

---

## 7. 常见问题与排查

### 7.1 "artifact index certifies it but file is missing"

```bash
# 原因：结果目录被截断/手动删除文件
# 解决：从备份恢复该文件，或删除整个结果目录重跑
rm -rf results
python3 -m cage.experiments --only e1 --seeds 5 --out results
```

### 7.2 "run_id claimed by 2 different setups"

```bash
# 原因：同一 results 目录混入了不同配置/代码版本的运行
# 解决：使用独立的 --out 目录，或清理后重跑
```

### 7.3 "critical environment drift"

```bash
# 原因：PyTorch/Python/依赖版本变化导致数值环境不兼容
# 解决：使用 requirements-ci.txt 精确复现环境，或在容器中运行
python3 -m pip install -r requirements-ci.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
make verify-deps          # 确认装出来的版本确实等于声明的固定版本
```

### 7.4 "design plan frozen with different seeds/SESOI"

```bash
# 原因：同一实验标签下改变了种子集合或 SESOI
# 解决：使用新的 --out 目录，或删除旧结果目录
```

### 7.5 "依赖一致性检查失败"

```bash
# 症状：[dep-check] FAIL: the environment does not match the declared
#        dependency set
# 原因：解释器里的 torch 不是 requirements-ci.txt 固定的那个版本
#       （常见于 pip install -e . 之后解析器把 torch 升到了最新版）
# 解决：按 requirements-ci.txt 重装，并复核：
python3 -m pip install -r requirements-ci.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
make verify-deps
```

---

## 8. 容器化复现（推荐用于跨机器精确复现）

`Dockerfile.repro` 从**同一份冻结依赖**（`requirements-ci.txt`）安装，
因此容器里的数值环境与 CI 一致。镜像内不再写死任何 torch 版本号——
版本只存在于 `requirements-ci.txt` 一处。

```bash
# 构建（CPU；与 CI 的 wheel 索引一致）
docker build -f Dockerfile.repro -t cage-attention:repro .

# 构建（GPU）：选择确实提供该固定版本的索引
docker build -f Dockerfile.repro -t cage-attention:repro-gpu \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 .

# 运行（结果挂载到宿主机）
docker run --rm -v $(pwd)/results:/work/results cage-attention:repro \
  --only e1 --seeds 5 --out results
```

镜像的三条工程约束（每条都对应一个真实故障）：

1. **索引必须提供固定版本**：torch 2.14.1 的 wheel 只发布在
   `cpu` / `cu126` / `cu130` 索引（`cu118` 最高只到 2.7.1）。换索引前
   先确认，否则构建会在装依赖时失败。
2. **`pip install -e .` 必须带 `--no-deps`**：否则 pip 会按
   `pyproject.toml` 的 `torch>=2.0` 重新解析依赖，把刚装好的固定版本
   换成别的版本。
3. **构建末尾自检**：装出来的 torch 与固定版本不一致就让构建失败，
   而不是产出一个"看起来能用、其实换了配方"的镜像。

---

## 9. 引用与致谢

本框架的方法学设计参考了以下最佳实践：

- **预注册与 SESOI**：Lakens et al., "Justify Your Alpha" (2018); "Equivalence Testing" (2017)
- **FDR 控制**：Benjamini & Hochberg (1995); Benjamini & Yekutieli (2001)
- **配对 Bootstrap 与 BCa**：Efron & Tibshirani (1993); DiCiccio & Efron (1996)
- **精确符号翻转检验**：Rosenberger & Geller (1983); Hemerik & Goeman (2018)
- **随机化完全区组设计**：Cox (1958); Hinkelmann & Kempthorne (2008)
- **可复现性工程**：Piccolo & Frampton (2016); "Reproducibility in ML" (NeurIPS 2019+)

---

## 附录：快速参考卡

| 命令 | 作用 |
|------|------|
| `make check` | 语法+设计固定项+冒烟+完整测试套件 |
| `make verify-deps` | 实装依赖 vs 声明依赖（固定版本是否真的装上了） |
| `make preflight` | `verify-deps` + `check`（正式实验前的推荐入口） |
| `make sanity` | 快速预检（HB5 overfit gate） |
| `python -m cage.experiments --only e1 --seeds 5 --out results` | 运行 E1 正式实验 |
| `make verify OUT=results` | 完整验证链（环境+制品+来源+运行+源码） |
| `python -m cage.experiments --replay-manifest results/e1_manifest.json --out results_replay` | 独立重放 |
| `make verify-replay SOURCE=results REPLAY_OUT=results_replay` | 重放比较 |
| `python3 capture_environment.py --compare results/environment.json` | 独立环境快照比对（与 `--check-environment` 同一套判据） |
| `python ci_gate.py --syntax-only` | 仅语法检查（关机 trap 用） |

**结果目录结构**：
```
results/
├── environment.json              # 当前环境记录
├── environment_history.jsonl     # 环境漂移历史
├── artifact_index.json           # 累积制品索引
├── run_index.jsonl               # 运行生命周期
├── study_design_preflight.json   # 研究级设计预检（可达性+功效+算力，开跑前）
├── analysis_plan.json            # 预注册研究级分析计划（确认性对比集+声明）
├── analysis_plan_history.jsonl   # 研究级计划修订历史
├── study_level_adjudication.json # 研究级裁决（IUT + 合并族）
├── e1_design_plan.json           # E1 设计计划（冻结）
├── e1_design_plan_history.jsonl  # 计划修订历史
├── e1_single_factor_design.json  # E1 单因素设计证书（开跑前）
├── e1_run_order.json             # E1 计划调度
├── e1_run_order_realized.json    # E1 实测调度+位置效应
├── e1_manifest.json              # E1 冻结清单（供重放）
├── e1_runs.jsonl                 # E1 完整结果记录
├── e1_error_cases.jsonl          # E1 错误案例
├── e1_per_variant_tuning.json    # E1 逐变体调参（探索性）
├── e1_lr_sensitivity.json        # E1 lr 敏感性（探索性）
├── baselines_verdict.json        # 基线裁决（HB1-HB7）
└── ... (e2_*, e3_* 同理)
```