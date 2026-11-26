# Code Review — cage-attention (2026-11-26 Agent)

## 审查范围
- 校准注意力门控 (`model.py`) — calib gate 公式, null attention, entropy 归一化
- 训练管线 (`train.py`, `_train_loop.py`) — 训练循环, 非有限损失处理, 验证评测
- 统计指标 (`metrics.py`) — 配对 bootstrap CI, sign-flip test, 等效性检验
- 数据管线 (`data.py`) — 合成数据生成, 验证, 标签噪声审计

## 审查结果：未发现 bug 或科研方法问题

### 校准注意力门控
- 门控公式 `sigmoid((W_g x - a*log T - c*scale*Hbar) / tau)` 数学正确
- `calib_scale` 正确设为不可学习参数（避免与 `calib_c` 不可辨识）
- 熵归一化 `Hbar = H / log T` 正确实现长度稳定的集中度信号
- 主动对照（`calib_shuffle_ent`）：确定性位置置换保持边际分布不变，只破坏行-熵关联
- 位置混淆消融（`calib_pos_norm`）：正确隔离全局 log T 与逐行可见 token 长度

### 训练管线
- 余弦退火调度器在跳过非有限步时正确推进（保持 T_max=steps 对齐）
- 梯度范数追踪与区间峰值检测正确
- 标签置换负对照：独立生成器保证相同数据流，仅标签通道不同
- eval/train 状态管理正确（保存/恢复训练状态）

### 统计方法
- `_self_test` 自检查：所有断言通过（配对 CI, sign-flip, BCa, 等效性, FDR）
- 小样本稳健性伴侣：jackknife + BCa 区间与 percentile CI 并行报告
- 精确配对 sign-flip 随机化检验：小 n 时枚举全部 2^n 符号分配

### 数据管线
- `validate_batch` 结构验证门控：无效批次中止而非静默处理
- 标签噪声率作为测量声明（非假设 0）— 方法论正确

## 结论
校准注意力门控实现数学正确，误差分类学严谨（带 Wilson 区间），训练管线正确处理边界情况。未发现需要修复的 bug 或方法论问题。
