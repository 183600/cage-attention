# 结果分析视角的科研方法改进建议

本文件在不修改 `.github` CI 配置与 `README*.md` 说明文档的前提下，对 CAGE-Attention 项目的科研方法提出可落地的改进。所有建议均围绕“结果必须可解释、可复现、可被多重检验所证伪”展开。

## 1. 现状与风险点

- 当前实验已具备种子级配对Bootstrap、FDR、BCA、Jackknife、精确符号翻转、跨种子-样本交叉Bootstrap、复制区间、选择调整置信区间等完整工具链。
- 仍存在三类结果解读风险：
  1. **流样本有限性未在层级结论中显式量化**：统计种子CI条件于密封评估流，跨种子-样本交叉Bootstrap仅作伴随注释，未在假设结论表中强制披露“STREAM-LIMITED”比例。
  2. **机制相关性指标易被结构性关联误导**：`gate_vs_entropy` 包含构造性成分，当前已提供偏相关与分层内相关，但未在项目级证据合成中强制要求“偏相关+主动对照”同时成立。
  3. **多重检验模型敏感性缺少聚合视图**：每行对比均报告BH/BY/Holm，但项目合成仅计数，未提供“模型敏感性热力图”与“选择敏感性比例”。

## 2. 改进措施

### 2.1 结果质量仪表盘
在 `cage/experiments/_synthesis.py` 中新增 `_result_quality_dashboard`，对每份 `<tag>_contrasts.json` 输出统一质量指标：
- `replication_uncertain_rate`：支持但复制区间跨零的比例
- `bca_disagree_rate`：BCA 与百分位CI决策不一致的比例
- `jackknife_sign_flip_rate`：单种子移除导致符号翻转的比例
- `stream_limited_rate`：种子CI排除0但交叉Bootstrap包含0的比例
- `selection_sensitive_rate`：FCR调整后结论消失的比例
- `exact_floor_violation`：Bootstrap p < 精确符号翻转最小可达p的对比数

仪表盘写入 `results/<tag>_quality.json`，并在 `project_evidence.json` 中汇总跨实验质量。

### 2.2 机制证据门槛强化
对 H1a/H1d/H1f 类机制假设，增加“双重成立”规则：
- 要求 `abs_gate_vs_entropy` 与 `gate_vs_entropy_partial_maxprob` 同时在FDR后支持，且主动对照 `calib_shuffle_ent` 的效应方向相反
- 偏相关必须伴随层级内一致性：`gate_vs_entropy_partial_maxprob_layers` ≥ 0.8 *总层数
- 在 `ContrastLog.finalize` 中增加 `mechanism_gate` 标志，违例时在控制台 `WARNING: mechanism evidence incomplete` 并在JSON中记录 `mechanism_sufficient=False`

### 2.3 流样本敏感性强制披露
修改 `cage/experiments/_contrastlog.py` 的打印与写入逻辑：
- 对 `stream_crossed_stream_limited=True` 的行，强制在 `supports_hypothesis` 为True时追加 `claim_eligible=False` 并标记 `stream_limited`
- 在项目合成中统计 `stream_limited_claims`，并在 `project_evidence.json` 中单独列出

### 2.4 效应量标准化报告
在每份对比行中已提供 Cohen's dz 与 Hedges' g 区间，建议：
- 在 `finalize` 输出中增加 `standardized_effect_by_seed` 分布摘要：中位数、四分位、最大最小
- 对 `accuracy` 类指标，除相对效应外，强制报告Wilson区间与种子级聚类比例 `clustered_proportion_ci` 的对比，以区分“数值改善”与“样本构成改善”

### 2.5 设计-结果闭环
利用现有 `design_plan.json`，在实验完成时生成 `design_plan_vs_outcome.json`，对比：
- 计划的最小可探测效应 MDE80 与实际观测效应
- 计划的精确检验分辨率与实际p值
- 计划的流样本边界与实际 `stream_limited` 比例
未达标项自动写入 `WARNING` 并建议增量种子或扩大评估流

## 3. 实现路径

- 新增 `cage/experiments/_result_quality_report.py` 实现仪表盘，入口函数 `build_quality_report(out_dir, tags)`
- 修改 `main()` 在 `--synthesize-results` 后调用该函数，输出 `project_quality.json`
- 不修改 `docs/design_notes.md` 中的【固定】决策，仅在 `docs/` 新增分析说明文件
- 所有改动保持 `ci_gate.py` 的设计固定项检查通过：不引入 SDPA、不改动注意力内核、保留种子控制

## 4. 预期收益

- 结果解读从“单点显著”转向“稳健性矩阵”：显著性、实用性、可复制性、流样本独立性、模型敏感性五维度齐备
- 机制声称需同时通过统计与结构性对照，避免构造性相关被误读
- 项目级证据合成可直接回答：哪些结论在更严格的多重检验模型下仍成立，哪些是流样本限制下的假阳性

以上改进可在不触碰 `.github` 与 `README*.md` 的约束下完成，且与现有 `cage.metrics` 工具链完全兼容。
