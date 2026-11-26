# Code Review — cage-attention (2026-11-26 第二轮独立审查)

## 审查范围
- `cage/model.py` — CAGE 校准注意力门控、null attention、entropy 归一化、位置编码
- `cage/train.py` — 训练管线、AdamW 参数分组、验证评测、标签置换负对照
- `cage/metrics.py` — 配对 bootstrap CI、sign-flip test、等效性检验

## 审查方法
独立逐文件审查，重点检查：
1. 校准门控公式的数学正确性和参数可辨识性
2. 训练循环的边界处理和非有限损失处理
3. 统计方法的正确性
4. 消融实验设计的严谨性

## 审查结果：未发现 bug 或科研方法问题

### 校准门控 (`model.py`)
- 门控公式 `sigmoid((W_g x - a*log T - c*scale*Hbar) / tau)` 数学正确
- `calib_scale` 正确设为不可学习 buffer（避免与 `calib_c` 不可辨识）
- `calib_tau` 通过 `softplus + 0.1` 参数化确保正值，`tau_init` 经 `log(expm1(x))` 反算保证初始值正确
- `_log_len_cache` / `_log_vis_cache` / `_ent_perm_cache` 缓存设计正确
- 位置置换 (`_ent_perm`) 使用确定性种子 `(layer_idx, T)`，保证同层同长度使用相同置换
- `_CalibAttn` 自定义 autograd.Function 的前向/反向实现正确
- `_NullAttn` 的 null 槽概率计算和梯度回传正确

### 位置编码外推控制
- `set_pos_extrapolation` 提供 learned/fold/zero 三种模式，正确隔离位置向量外推与注意力/门控行为
- fold 模式 `p % trained_len` 使外推行复用已训练的位置嵌入

### 训练管线 (`train.py`)
- AdamW 参数分组：≥2-D 权重矩阵有衰减，bias/norm/embedding/scalars 无衰减
- 余弦退火调度器正确处理跳过非有限步的情况
- 标签置换负对照使用独立生成器，保证相同数据流只改标签通道

### 统计方法 (`metrics.py`)
- Wilson CI、BCa bootstrap、精确 sign-flip 随机化检验均正确实现
- 小样本稳健性：jackknife + BCa 与 percentile CI 并行报告

## 结论
校准门控实现数学正确，参数可辨识性设计严谨，训练管线和统计方法无问题。
