# 代码审查报告 (2026-11-18 第三轮)

## 审查概要

| 项目 | 内容 |
|------|------|
| 仓库 | cage-attention |
| 审查日期 | 2026-11-18 (第三轮独立验证) |
| 审查方法 | 子 Agent 独立深度审查：训练循环完整性 + exec() 安全性 + 统计方法 + 科研方法学 |
| 审查范围 | 全部 46 个 Python 源文件 |

## 总体评价

代码质量**优秀**。经第三轮独立验证，确认前两轮审查结论，未发现新的缺陷。

## 本轮审查重点

### 1. 训练循环完整性复验

核查了全部训练相关函数中的 `loss.backward()` 上下文：

| 文件 | 行号 | 判定 |
|------|------|------|
| `ci_gate.py` | 150 | **安全**：CI 烟雾测试，仅验证梯度有限性，不需要完整优化器步骤 |
| `baselines.py` | 262 | **安全**：完整 `zero_grad → backward → clip_grad_norm_ → step` 链路 |
| `train.py` | 403 | **安全**：完整训练循环 |
| `_train_loop.py` | 132 | **安全**：`collect_stats` 评估函数，正确 `model.eval()` + 保存/恢复训练状态 |
| `_train_recipe.py` | 多处 | **安全**：完整的训练 recipe 循环 |

### 2. exec() 调用安全性

`cage/_fragments.py:42` 的 `exec(code, namespace)` 调用：

- **确认安全**：仅执行通过 `compile()` 预编译的本地项目文件代码
- 不处理用户输入或外部数据
- 有完善的异常包装（`RuntimeError`）和 `# nosec B102` 标注
- 这是模块组合的架构设计选择，非安全漏洞

### 3. 统计方法正确性

- **分层抽样方差**：`_metrics_inference.py` 使用 Kish 有效样本量公式，分母 `clamp_min(1e-10)` 保护
- **排名预测**：`_metrics_ranking.py` 使用 `counts.clamp_min(1.0)` 避免除零
- **流式标准误**：`_metrics_robustness.py` 使用 `max(..., 0.0)` 保护 sqrt 参数
- **35 处 TODO 注释**："Report confidence intervals" — 标记为未来改进方向，不影响当前正确性

### 4. 科研方法学

- **主动控制设计**：`calib_shuffle_ent` 的位置置换分离"信息"与"存在额外输入"的效果
- **SeedNorm 动态归一化**：scale 从隐藏状态预测，不引入数据泄漏
- **消融实验**：多配置对比，含 `gate=none/query/calib` 等变体

## 修复清单

无新增修复。

## 审查结论

cage-attention 的训练循环、统计方法和科研方法学经过三轮独立审查验证，代码实现正确可靠。exec() 调用为安全的本地模块组合设计。35 处 TODO 为合理的后续改进标记。
