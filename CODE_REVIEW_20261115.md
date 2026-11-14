# 代码审查报告 - cage-attention
**审查日期**: 2026-11-15  
**审查范围**: 核心代码（cage/）和 CI 门禁（ci_gate.py）  
**审查深度**: 人工逐行审查核心模块 + AST 静态分析  

## 审查结论

**代码质量评级：优秀**。未发现需要立即修复的关键代码 bug 或科研方法问题。

## 详细审查结果

### 1. 校准门控机制 (cage/model.py)

**门控公式实现** ✅ 正确
- 校准门控 `gate = sigmoid((W_g x - b0 - a·log(T) - c·H_norm) / tau)` 实现与文档一致
- 熵归一化 `H_norm = H / log(T)` 正确实现长度校准
- 消融开关 (calib_use_len, calib_use_ent, calib_shuffle_ent) 正确隔离各组件
- 安慰剂对照 (calib_shuffle_ent) 正确实现位置置换，保持边际分布不变

**Null 注意力** ✅ 正确
- Null 槽的联合 softmax 正确实现：`p_null = sigmoid(a_null - lse)`
- 反向传播正确处理 null 槽梯度
- 诊断统计基于 dropout 前的 softmax 分布（避免重标定失真）

**分块计算** ✅ 正确
- 查询分块路径正确处理因果掩码
- 反向传播正确累积跨块梯度
- 内存管理合理（scratch buffer 复用）

### 2. 数据处理 (cage/_data_task.py)

**数据泄漏防护** ✅ 完善
- `time_ordered_split()` 正确实现时序切分（过去→训练，最新→测试）
- `group_split()` 正确保证群组不跨侧
- 合成任务的 i.i.d. 性质通过 seed-owned stream 实现隔离
- `check_train_seed()` 正确拒绝使用保留评估种子训练

**Shuffle 警告澄清** ✅
- 代码注释明确区分"划分前 shuffle"（禁止）和"训练批内 shuffle"（允许）
- 实际实现中训练数据在划分后才进行批内 shuffle，无泄漏风险

### 3. 训练循环 (cage/train.py, cage/_train_loop.py)

**损失计算** ✅ 正确
- 训练损失正确累积并除以有效 token 数
- 验证损失正确使用均值和标准误
- 梯度裁剪和早停机制正确实现

### 4. 统计分析 (cage/_metrics_*.py)

**除零保护** ✅ 全面
- `_metrics_inference.py` 中的除法均有 `max(denom, 1e-10)` 保护
- `_metrics_ranking.py` 中的相关系数计算正确处理零方差情况
- `_metrics_robustness.py` 中的分位数计算正确处理空数组

### 5. 静态分析结果

- **eval() 调用**: 全部为 `model.eval()`，无安全风险
- **数据 shuffle**: 均在划分后执行，无泄漏风险
- **除零风险**: 已有适当保护

## 审查中发现的非关键观察

1. 校准门控的 tau 参数化使用 `log(expm1(tau_init))` 保证 tau > 0，设计巧妙
2. 实验设计 (E1/E2/E3) 文档完善，对照设计严谨

## 结论

代码实现与论文方法一致，科研方法设计严谨（主动对照、安慰剂对照、消融实验），无需修改。
