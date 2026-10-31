# 校准项的模型内 knockout（补强消融）

## 目的

E1/E2 现有的校准项消融（`calib_len_only` / `calib_ent_only` /
`calib_no_terms`）会**重新训练**一个去掉对应项的模型。因此“去掉项 → 精度
变化”混合了两个来源：

1. 该项在已训练模型中的实际贡献；
2. 去掉该项后优化轨迹改变带来的差异。

`calib_term_knockout_eval` 提供互补的**模型内干预**：固定已训练权重，只把
选定的可学习系数临时置零，再在**同一条封闭评测流**上重新打分。这样
baseline 与 knockout 是逐样本配对的，精度差可直接读作“该已学习项在推理时
被使用的程度”。

## 实现与记录

- 入口：`cage/train.py::calib_term_knockout_eval`。
- `terms="length"` 将各层 `calib_a` 置零（`-a*log T` 项）。
- `terms="concentration"` 将各层 `calib_c` 置零（`-c*scale*Hbar` 项）。
- `terms="both"` 将两类系数全部置零。
- 参数在 `finally` 中恢复；恢复值来自置零前的 clone，因此是精确恢复。
- 非 `calib` 门控返回 `applicable=False`；架构中不存在的项在 `by_term`
  中单独返回 `applicable=False`，不会伪造基线。
- E2 在 `calib` 和 `calib_raw_ent` 两条完整双臂上，于训练长度与最长扫描
  长度写入 `calib_term_knockout`，并输出
  `e2_term_knockout.json` / `_term_knockout` 汇总。

## 解释边界

- 这是**探索性证伪/机制检查**，不进入 H2 的 FDR 决策族，也不能替代
  预注册的重新训练消融。它回答的是“训练后的模型是否使用该项”，不是
  “没有该项时重新优化能达到什么”。
- 负的 delta 表示去掉该项后精度下降，即该项在该长度上被使用；接近零
  表示训练后的该项对该评测流没有可测影响。两者都不单独证明因果机制，
  需与 `calib_coefficients`、重新训练消融和 H2 主检验一起读。
- 只置零系数、不改 `calib_tau` / `calib_scale` 与其它权重；结论不能外推
  到“整个门控模块无用”。
- 评测流种子、batch 几何与记录中的 base 评测一致；若 baseline 不能复现
  记录精度，记录会给出警告并标记 `calib_knockout_base_matches_raw=False`。

## 与已有设计的关系

该补强不修改 `docs/design_notes.md` 的固定决策，也不改变任何预注册假设、
SESOI、FDR 族或 runner 的判定路径。它只是在 E2 的已有运行上增加一个
配对的、无额外训练成本的读出，补上“重新训练消融”无法区分的
“项是否被训练后的模型实际使用”这一问题。
