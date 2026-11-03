# 结果分析视角下的科研方法改进建议

> 本文件基于对现有实验体系的审查，从结果分析与统计推断的严谨性出发，提出可在不触动 `.github` CI 配置及 README 文档的前提下增强的改进方向。所有建议保持与 `docs/design_notes.md` 中【固定】设计决策一致，不修改数值路径，只扩展分析层的透明度与鲁棒性。

## 现状总结

当前项目已具备业内领先的统计基础设施：

- 配对自助 + BCa/ jackknife/ leave-one-seed-out 鲁棒性伴随检验
- 精确配对符号翻转检验暴露五种子样本的最小可达 p 值
- Benjamini-Hochberg / Benjamini-Yekutieli / Holm-Bonferroni 多重检验模型并行报告，配合 Simes 族检验
- SESOI 最小效应、等价性 TOST、最小效应检验、复制区间
- 跨种子×评价流的交叉聚类自助、评价流折叠审计、校准误差分解
- 预注册假设表含 `sesoi` 与等价边界，结果记录哈希化

这些已显著降低了“裸相关”与“显著但无意义”误读的风险。

## 可改进的分析维度

### 1. 种子分辨率与统计功效
- **问题**：默认 5 个训练种子 => 双侧符号翻转最小 p = 2/32 ≈ 0.0625，无法达到常规 0.05。置信区间一阶近似较差，BCa 常因单种子主导而失效。
- **改进**：
  - 在实验规划阶段使用 `cage.metrics.seeds_for_power` 对每条预注册对比做功效预审，将种子数作为设计变量而非默认常量。建议对核心家族 H1a/H1b/H2a 设置 `min_seeds_for_claim ≥ 7`。
  - 引入自适应停止：以序列 FDR 控制的 alpha-spending 函数累积证据，避免一次性设定种子数过高。
  - 对每条对比报告 `sign_flip_min_units` 与 `sign_flip_resolution`，在结果表中直接标注“本设计无法分辨的效应大小”，防止把“未显著”误读为“无效应”。

### 2. 多重检验的层次结构
- **问题**：E1 已对 4 个 gate 变体+消融臂做 BH 控制，但 `false_concentration_sweep` 在 0.3/0.5/0.7/0.9 多阈值上重复评估同一假设，形成隐式多重检验。
- **改进**：
  - 将阈值扫描设为探索性伴随，并在 `contrastlog` 中标记 `exploratory=True`，不进入预注册 FDR 家族。如需正式比较，选择一个预注册阈值，其余阈值以 FDR 报告 `selection_at_fraction` 曲线展示。
  - 对逐层指标 `gate_vs_entropy`、`attn_write_ratio` 等实施层级 FDR：先在模型层面上控制，再在层内报告局部错误发现率，避免层数随深度增加带来的假阳性膨胀。

### 3. 构造性相关的部分相关控制
- **问题**：`calib` 门控公式显含熵项，导致 `gate_vs_entropy` 相关部分源于构造。现有文档已提示，但数值报告仍给出裸相关。
- **改进**：
  - 增加部分相关估计：在控制 `W_g x` 与 `log T` 后，估计熵对门控的增量解释方差。可用残差回归实现，保持数值路径不变，仅在 `_train_eval` 中新增统计块。
  - 对 `calib_shuffle_ent` 安慰剂臂报告信息增益的零效应区间，而不仅是点估计，以量化“容量 vs 信息”分解。

### 4. 效应量标准化的可解释性
- **问题**：准确率差异以绝对百分点计，但在不同长度、不同归一化下基线漂移会影响解读。
- **改进**：
  - 对每条对比同时报告配对 Cohen's dz 与 Hedges' g 的 BCa 区间，并给出基于任务难度标准化的效应：`(Δacc) / (rule_ceiling - chance)`，使跨实验的效应可比。
  - 对 `gate_depth_ratio` 报告对数比并给出置换检验 p 值，避免比率的偏态影响。

### 5. 方差分解与流稳定性的可视证据
- **问题**：`crossed_cluster_bootstrap_ci` 已区分种子与评价流方差，但结果记录中仅保留区间，未保留方差成分比例。
- **改进**：
  - 在每条结果记录中加入 `var_seed / var_item / var_residual` 分解比例，以及 `evaluation_fold_audit` 的脆弱性分数。对 `stream-limited` 的结论，强制要求同时报告交叉区间与种子区间。
  - 对 E2 长度泛化，增加每长度点的 `pos_extrapolation_control` 三读法的差异分解：`fold-zero` 恢复比例 = `(acc_fold - acc_zero) / (acc_train - acc_zero)`，以量化位置外推混淆的贡献。

### 6. 超参数敏感性与交互效应
- **问题**：E1/E3 使用固定 lr 与 `calib_tau_init`。`per_variant_tuning` 仅作为探索性伴随。
- **改进**：
  - 将 lr × tau 的敏感性网格结果汇总为响应面摘要：对每个 gate 报告效应随 lr 的单调性与拐点。敏感性结果以 `exploratory` 标签存档，并在主表中引用“效应方向在 0.5× / 2× lr 下保持”。
  - 对 E3 的三因素设计 `gate × norm × LayerScale`，增加交互效应估计：报告 `gate × norm` 的配对差异差异，并用分层 FDR 控制交互检验，避免把主效应误读为普遍有效。

### 7. 校准系数的统计稳定性
- **问题**：`calib_coefficients` 记录均值，但未评估跨种子收敛性。
- **改进**：
  - 对 `a_mean` 与 `c_effective_mean` 报告跨种子变异系数与置换检验的零假设 `c_effective = 0`。若系数符号在种子间不一致，需标记该机制的学得不稳定。
  - 增加系数与性能的种子级相关：若系数大小与准确率提升不相关，则“校准起作用”需依赖其他证据。

### 8. 结果报告的决策透明度
- **问题**：多重检验模型并行报告后，读者仍需自行判断哪个是“主要决定”。
- **改进**：
  - 在 `contrastlog_finalize` 输出中加入决策溯源字段：`decision_basis = {"primary": "BH-95CI+SESOI", "companions": ["BCa_agrees", "sign_flip_valid", "replication_interval"]}`，并在最终表格中直接标注 `selection-sensitive`、`replication-uncertain`、`stream-limited` 等标签。
  - 禁止仅以点估计排序：所有摘要表必须包含区间与伴随标签。

### 9. 文档化分析假设
- **建议**：在 `docs/` 下新增 `analysis_assumptions.md`，显式列出所有分析层假设：自助重采样独立性、种子交换性、评价流静态性、相关系数 Fisher z 变换的正态性假设等。便于后续审阅时快速定位假设违反。

## 不触动的原则
- 不修改 `.github/` 任何文件。
- 不删除或移动任何 `README*.md`。
- 不修改 `docs/design_notes.md` 中的【固定】决策。
- 所有新增分析仅作为伴随或探索性输出，不改变预注册决策规则。

以上改进可在现有统计框架上渐进实施，显著提升结果的可重复解释力与抗误读能力，同时保持实验设计的数值一致性。
