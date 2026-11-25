# Code Review Report - 2026-11-26

## 审查范围

核心模块：`cage/model.py`（Attention、CAGE、Block）、`cage/train.py`（_make_adamw）

## 审查结果

### 总体评价：代码质量优秀，自定义 autograd Function 实现精确

本次审查聚焦于：
1. 校准门控（calibrated gate）的数学正确性
2. 自定义 `_CalibAttn` / `_NullAttn` 的前向/反向一致性
3. scratch buffer 管理的安全性
4. 位置外推控制（pos_extrapolation）的正确性
5. 优化器参数分组

### 详细发现

#### 1. 校准门控公式 — 正确 ✓

`gate = sigmoid((W_g x - a·log T - c·scale·H̄) / τ)` 实现：
- 温度参数通过 `softplus(calib_tau) + 0.1` 参数化，保证 τ > 0.1
- 长度归一化 `H̄ = H / log T` 可选关闭（`calib_ent_norm=False` 消融臂）
- 位置混淆消融 `calib_pos_norm` 正确使用逐行可见长度 `log(max(t+1, 2))`
- 主动对照 `calib_shuffle_ent`（位置置换）正确保持边际分布不变

#### 2. `_CalibAttn` 自定义内核 — 正确 ✓

- 前向使用 `baddbmm` + `_softmax_inplace`，分块路径正确处理块间数据流
- 反向 `_softmax_grad_from_p` 正确实现 `d_s = p · (d_p - <d_p, p>)`
- 熵梯度 `-(log(p) + 1)` 正确处理 `p > eps` 选择
- `ctx.set_materialize_grads(False)` 优化零梯度路径

#### 3. `_NullAttn` — 正确 ✓

- null 槽不参与因果掩码（`_causal_null_mask` 第 0 列为 0）
- `_null_probs` 的联合 softmax 数值稳定（先 max 再 exp）
- 反向正确处理 null key/value 的梯度累加（跨块 accumulate）

#### 4. Scratch Buffer 管理 — 安全 ✓

- `_get_scratch` 使用 `(name, device, dtype)` 键，精确大小匹配或切片
- 过大缓冲（> 4× 请求大小）自动缩容
- `release_eval_scratch()` 在评测后释放，避免训练缓冲被评测残留占用
- `_servable` 检查 inference tensor 是否可在当前模式使用

#### 5. 优化器参数分组 — 正确 ✓

`_make_adamw` 按维度分组：
- `p.ndim >= 2` 且非 norm/bias → weight_decay = 0.01
- 1-D 标量、bias、norm → weight_decay = 0
- 与 lega-attention、ada-squash 的分组约定一致

#### 6. 轻微建议（非 bug）

- `Attention.forward` 的 dropout 分支（`dropping=True`）与无 dropout 分支（`dropping=False`）代码路径差异较大，建议增加注释说明两者的数值对齐策略。

## 结论

自定义 autograd Function 实现精确，校准门控与消融设计科学严谨，未发现需要修复的问题。
