# CAGE-Attention 假设预注册模板

## 使用说明

本模板用于在实验运行**前**预注册科学假设。所有预注册假设必须：
1. 记录在 `cage/experiments/_config.py:HYPOTHESES` 字典中
2. 包含完整的 SESOI、方向性、判定规则
3. 通过 `hypotheses_sha256` 内容哈希锁定，任何事后修改都会被 `verify-provenance` 检测到

## 模板结构

```python
"HYPOTHESIS_ID": {
    "experiment": "e1|e2|e3|baselines",     # 所属实验（决定 FDR 家族）
    "question": "核心科学问题，单句陈述",
    "hypothesis": "机制假设，解释为什么预期会观测到该效应",
    "prediction": "可测量的定量预测，包含指标名、预期方向、效应量",
    "decision": "完整的统计判定规则：CI 类型、FDR 控制、SESOI 阈值、方向性要求",
    "sesoi": {                               # 必填：每个指标的最小关注效应量
        "metric_name": 0.01,                 # 单位：该指标的原始尺度
    },
    "direction": {                           # 方向性预测（有方向性假设时必填）
        "metric_name": "+",                  # "+" 正向，"-" 负向，None 双向/无方向
    },
    "sesoi_test": {                          # 空假设/等价性预测时必填
        "metric_name": "equivalence|noninferior",  # 等价性检验或非劣效性检验
    },
    "context_metrics": ("metric_name",),     # 可选：伴随指标（不参与决策）
}
```

## 字段详细规范

### 1. `experiment` - FDR 家族归属
- `baselines`：基线前提假设（HB0-HB7），独立家族
- `e1`：门控对比实验假设（H1a-H1g），共享 E1 FDR 家族
- `e2`：长度泛化实验假设（H2a-H2g），共享 E2 FDR 家族
- `e3`：协同设计实验假设（H3a-H3c），共享 E3 FDR 家族

### 2. `sesoi` - 最小关注效应量 (SESOI)
**必须为每个决策指标指定 SESOI**。统计显著但效应量 < SESOI 判定为 "practically negligible"。

| 指标类型 | SESOI 单位 | 设定原则 |
|----------|------------|----------|
| 准确率/错误率 | 概率点 (0.01 = 1%) | 基于任务随机基线 1/(V-2) 的实质意义 |
| 相关系数 | 相关系数单位 (0.05) | 基于注意力熵/max-prob 典型变异范围 |
| 比率类 | 比率单位 (0.05) | 基于层深比率的实质变化阈值 |
| 系数类 | 前sigmoid门控单位 (0.01) | 基于校准项系数的可识别性下界 |

### 3. `direction` - 方向性预测
- `"+"`：预期效应为正（如 `calib - query > 0`）
- `"-"`：预期效应为负（如 `accuracy_drop(calib) < accuracy_drop(none)`）
- `None`：双向/无方向预测（如交互项、完整性检查）
- **反方向的 FDR 显著效应直接判定为 `refuted`，无论效应量大小**

### 4. `sesoi_test` - 空假设/等价性检验类型
| 类型 | 适用场景 | 判定规则 |
|------|----------|----------|
| `"equivalence"` | 预测效应在 ±SESOI 内（长度稳定性、归一化稳定性） | Bootstrap TOST：95% CI 完全落在 ±SESOI 内 |
| `"noninferior"` | 预测不劣于对照（校准门控不损失准确率、null 不损失准确率） | 单侧非劣效性：95% CI 下界 ≥ -SESOI |

### 5. `decision` - 完整判定规则文本
必须包含：
- CI 类型（配对 bootstrap 95% CI / 单样本 bootstrap 95% CI / bootstrap TOST）
- FDR 控制（Benjamini-Hochberg q=0.05，家族内）
- SESOI 门槛引用
- 方向性要求
- 最小配对种子数要求（`min_seeds_for_claim`，默认 3）

## 示例：完整预注册条目

### HB1 - 规则天花板（基线前提）
```python
"HB1": {
    "experiment": "baselines",
    "question": "Is the recall task well posed with a known solution?",
    "hypothesis": "The pair structure admits an exact parameter-free lookup rule because keys are distinct per sequence.",
    "prediction": "rule accuracy ~ 1.0 (ceiling) and unique_key_rate = 1.0",
    "decision": "rule accuracy < 0.99 (point estimate; the Wilson 95% CI is reported to show the precision of the finite evaluation stream) indicates a broken task/data construction and invalidates everything above it",
    "sesoi": {"accuracy": 0.01},
}
```

### H1a - 校准门控使用注意力分布信息（核心机制）
```python
"H1a": {
    "experiment": "e1",
    "question": "Does the calibrated gate use attention concentration / length information beyond the query?",
    "hypothesis": "The query-only gate has no access to the attention distribution, so |gate_vs_entropy| and |gate_vs_maxprob| are both clearly larger for calib than for query.",
    "prediction": "|gate_vs_entropy|(calib) - |gate_vs_entropy|(query) > 0 and |gate_vs_maxprob|(calib) - |gate_vs_maxprob|(query) > 0; the same direction holds for the rank (Spearman) correlations as a robustness check. The registered accuracy companion (calib - none and calib - query) is a one-sided non-inferiority check: the calibrated gate must not lose accuracy beyond the registered SESOI to its controls - a loss larger than that bound refutes the mechanism, while a gain is not required",
    "decision": "paired bootstrap 95% CI of the seed differences excludes 0 after Benjamini-Hochberg FDR control across the E1 contrasts (q=0.05); the accuracy companions are one-sided non-inferiority tests against the registered SESOI (lower bound above -SESOI), not directional superiority claims",
    "sesoi": {"accuracy": 0.01, "abs_gate_vs_entropy": 0.05, "abs_gate_vs_maxprob": 0.05},
    "direction": {"abs_gate_vs_entropy": "+", "abs_gate_vs_maxprob": "+", "query_none_accuracy": None},
    "sesoi_test": {"accuracy": "noninferior"},
}
```

### H2b - 熵归一化的长度稳定性（等价性预测）
```python
"H2b": {
    "experiment": "e2",
    "question": "Do the attention statistics drift with length as the calibration assumes?",
    "hypothesis": "Raw entropy grows with log(T), while the quantity the gate normalises by, H/log(T), is length-stable.",
    "prediction": "raw entropy drift (longest - base) > 0 (directional) and the normalised entropy drift H/log(T) lies inside the pre-registered equivalence bound, with the calibrated gate drift inside its bound too",
    "decision": "paired bootstrap 95% CIs, FDR-controlled across the E2 contrasts (q=0.05): the raw entropy drift CI excludes 0 with a positive mean (difference test); the H/log(T) drift and the gate drift are *equivalence* (TOST) tests - each is supported only when its CI lies inside +-SESOI, so a CI that merely includes 0 while spanning a drift larger than the bound is 'not established', not length-stability evidence",
    "sesoi": {"entropy_drift": 0.1, "entropy_norm_drift": 0.05, "gate_drift": 0.02},
    "direction": {"entropy_drift": "+"},
    "sesoi_test": {"entropy_norm_drift": "equivalence", "gate_drift": "equivalence"},
}
```

### H1f - 主动安慰剂对照（机制特异性）
```python
"H1f": {
    "experiment": "e1",
    "question": "Is the calibrated gate's association with the attention distribution carried by the real entropy information, or only by the presence of an extra data-dependent input (active placebo control)?",
    "hypothesis": "An active-control arm with the identical calib architecture, parameter count, computation and optimiser, but whose gate reads a fixed position-shuffled copy of the attention entropy (same values and marginal distribution, decorrelated from each queried row's own attention), must lose the gate/entropy association and must not reproduce the full gate's advantage over the query-only gate.",
    "prediction": "|gate_vs_entropy|(calib) - |gate_vs_entropy|(calib_shuf_ent) > 0 (the shuffled signal contains no information about the read row, so the association must collapse), and accuracy and gate_margin of full calib exceed the placebo arm's - a placebo that matches full calib would refute the calibration story and show the effect is capacity, not attention-distribution information",
    "decision": "paired bootstrap 95% CIs of the three calib - placebo differences exclude 0 in the pre-registered directions after Benjamini-Hochberg FDR control across the E1 family (q=0.05). The mechanism leg validates that the placebo actually severed the signal pathway; the accuracy/gate_margin legs are the specificity test - a reversed or null read there means the calibration benefit is not attributable to the entropy information",
    "sesoi": {"accuracy": 0.01, "gate_margin": 0.02, "abs_gate_vs_entropy": 0.05},
    "direction": {"accuracy": "+", "gate_margin": "+", "abs_gate_vs_entropy": "+"},
}
```

## 预注册检查清单（运行前必须确认）

- [ ] 所有新假设已添加到 `HYPOTHESES` 字典
- [ ] 每个假设包含 `experiment`、`question`、`hypothesis`、`prediction`、`decision`
- [ ] 每个决策指标在 `sesoi` 中有数值
- [ ] 有方向性预测的指标在 `direction` 中标注 `+`/`-`/`None`
- [ ] 等价性/非劣效性预测的指标在 `sesoi_test` 中标注 `"equivalence"`/`"noninferior"`
- [ ] `direction` 与 `sesoi_test` 的键不冲突（同一指标不能既有方向又做等价性检验，除非用不同键名区分如 `calib_abs_*` vs `query_abs_*`）
- [ ] FDR 家族划分正确（同一实验的所有预注册对比在同一家族）
- [ ] 运行 `python3 -c "from cage.experiments import HYPOTHESES; import json; print(json.dumps({k: v.get('sesoi') for k,v in HYPOTHESES.items()}, indent=2))"` 验证 SESOI 完整性
- [ ] 运行 `make check` 确保 CI 门禁通过（含设计固定项检查）

## 版本控制

| 版本 | 日期 | 变更 | Git SHA |
|------|------|------|---------|
| 1.0 | 2026-11-03 | 初始版本，对应当前 HYPOTHESES 注册表 | - |

> **重要**：任何预注册假设的修改（增/删/改 SESOI/方向/判定规则）都会改变 `hypotheses_sha256`，导致：
> 1. 新运行的 `setup_sha256` 与旧结果目录不同
> 2. `--verify-provenance` 会报告不匹配
> 3. `--replay-manifest` 会拒绝重放（清单哈希不匹配）
> 
> 这是**有意设计**：预注册一旦锁定不可事后更改。如需修正，必须作为新版本预注册并在文档中说明变更理由。