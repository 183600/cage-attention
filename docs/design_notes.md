# CAGE-Attention 设计说明（固定版）

> **本文档是本仓库的"设计宪法"。** 其中标记【固定】的决策为最终决定，
> CI 循环（含 perf / bug / clean / apply 各阶段）**必须遵守，不得替换、
> 不得删除、不得修改本文件**。CI 会在每次提交前运行 `ci_gate.py` 校验
> 这些决策；违反固定设计的改动会被整段撤销（`git reset --hard`）。
>
> 人工变更流程：直接修改本文件并提交后，**必须**同步移动固定标签：
> `git tag -f design-pin <新提交sha> && git push origin design-pin`。
> 未移动标签的修改会在下一次 CI 运行时被自动回滚。

## 0. 为什么有这份文档

自主 CI 循环使用随机更换的免费模型驱动。不同模型对"什么是最优实现"
判断不一致，曾造成核心实现反复横跳：注意力在 SDPA 与手写路径之间来回
切换 12+ 次、`calib_tau` 初始化被等价重写 4 次、`foreach=True` 加了又拆。
为终止这种布朗运动，现将关键设计固定如下，并附上选择理由。

## 1. 研究目标（背景，不可漂移）

针对两个评审关切做受控机制研究：

1. **门控注意力是否只是已有门控的小变体？** —— 需要严格的 4 变体对照（E1）；
2. **深层残差贡献不足**（Pre-LN 深层只做微小更新）—— 需要门控 × 归一化
   协同设计（E3），并以长度泛化（E2）佐证。

核心思想：门控不应只依赖 query（Gated Attention 风格），还要按注意力分布的
集中度（熵）与序列长度做校准：`σ((W_g x − a·log T − c·H̄)/τ)`，
其中 `H̄` 为按 `log T` 归一化的逐头注意力熵均值。

## 2.【固定】设计决策

### D1. 注意力实现：统一的分块手写 kernel 路径

- **决策**：四种 gate 变体（`none` / `query` / `calib` / `null`）全部走同一套
  手写 kernel：
  - 无 attention dropout 时：`_CalibAttn`（none/query/calib 共用，熵为免费
    副产物）与 `_NullAttn`（null 变体专用）；
  - 有 attention dropout 时：统一显式 math 路径（stats 从干净概率读取，
    dropout 作用于概率）。
- **禁止**：引入 `F.scaled_dot_product_attention`（SDPA）或任何其它注意力
  实现替换主路径；禁止恢复按 gate 变体分派的 `use_sdpa` 双路径。
- **理由**：E1 是受控对比——四个变体必须共享同一数值路径（同一掩码构造、
  同一 softmax 计算顺序），否则 kernel 差异成为混淆因子；SDPA 不返回注意力
  概率，无法产出校准门控所需的熵。性能优化只能在分块大小、内存布局、
  原地操作、缓存等维度进行，不得更换实现族。
- **震荡史**：SDPA ↔ 手写路径来回切换 12+ 次；`use_sdpa` 双路径被反复
  增删。

### D2. 因果掩码：模块级缓存

- **决策**：`_causal_mask` 以 `(T, device, dtype)` 为键缓存于模块级
  `_causal_mask_cache`（LRU 淘汰，上限 `_MAX_MASK_CACHE=64`）；掩码为只读
  共享张量。查询行掩码 `_mask_rows` 依赖 `pos` 数值，按调用重建。
- **禁止**：逐调用重建全量掩码；删除缓存。
- **震荡史**：`lru_cache` 装饰器与手写 dict 缓存之间被等价重写过多次。

### D3. 查询分块公式

- **决策**：`max_rows = (1 << 20) // (bh * T)`，下限 32；`T <= 32` 时整块。
- **禁止**：去掉外层括号（`1 << 20 // x` 是运算符优先级 bug，历史上
  曾真实出现过并静默退化为 32 行下限）。
- **理由**：控制 `(B*H, T, T)` 概率矩阵的峰值显存，同时保证小序列不分块。

### D4. `calib_tau` 初始化

- **决策**：`tau_init = math.log(math.expm1(cfg.calib_tau_init))`，
  语义为 `softplus(tau) + 0.1 = calib_tau_init`（默认 0.9，即初始温度 1.0）。
- **禁止**：任何"等价重写"（换 torch API 写法、内联常量等均无意义且已造成
  过 4 次无净变化的提交）。
- **理由**：软温度保证 τ 恒正；`+0.1` 防止 softplus(τ)→0 时门控饱和。

### D5. 优化器构造

- **决策**：`train.py::_make_adamw` 工厂：CUDA 参数优先 `fused=True`
  （失败回退），否则 `foreach=True`（失败回退），最后回退默认构造；
  超参固定 `weight_decay=0.01, betas=(0.9, 0.95)`，配合
  `CosineAnnealingLR(T_max=steps, eta_min=0.1*lr)`。
- **禁止**：在训练循环内联优化器构造；去掉 try/except 回退链。

### D6. 自定义 autograd 数值核（省显存设计）

- **决策**：保留 `_SoftmaxInPlace`、`_RowEntropy`、`_CalibAttn`、
  `_NullAttn` 四个手写 autograd Function：反向时按查询块重算概率，
  不保存概率矩阵。MLP 的 GELU 通过 `_GeluLinear`（`nn.Linear` 子类，
  `F.linear(F.gelu(h), ...)`）实现。
- **禁止**：删除或绕过这些 kernel 改用会物化注意力矩阵的实现；
  禁止把 `_GeluLinear` 换回手写 backward 的 `_GeluAffine`（曾引入数值
  bug，被 bug 阶段反复修复）。
- **理由**：熵项参与梯度（校准门控），标准 autograd 无法直接表达；
  重计算设计是 E2 长序列扫描能跑起来的前提。

### D7. 数值稳定约定

- **决策**：`_EPS = 1e-9`；熵计算前 `p.clamp_min(_EPS)`；梯度中
  `p > _EPS` 布尔掩码过滤零概率项（bool 张量与 float 相加时按 0/1 提升）。
- **禁止**：调整 `_EPS` 量级（会改变熵/梯度的数值行为，破坏跨提交可比性）。

### D8. 训练可复现性

- **决策**：`train_model` 的 `seed` 同时控制数据生成器（`torch.Generator`）
  与全局 RNG（dropout 掩码）；`eval_accuracy` / `collect_stats` 使用独立
  固定种子（123 / 7）。
- **理由**：无此约定则同配置两次运行结果不同，所有对照实验失效。

### D9. 编译与检查点

- **决策**：`torch.compile` 仅经 `build(compile=True)` 显式开启（默认关），
  不可用时打印警告并跳过；梯度检查点经 `Cfg.use_gradient_checkpointing`
  显式开启（默认关）。
- **理由**：两者都改变数值/内存行为，作为可选项保持实验主路径纯净。

## 3. 实验契约（E1/E2/E3）

| 实验 | 网格 | 关键指标 |
|---|---|---|
| E1 门控对比 | `gate ∈ {none, query, calib, null}` | 联想回忆准确率；`gate_vs_entropy` / `gate_vs_maxprob`（逐层 Pearson，Fisher z 平均）；`false_conc_rate`；`gate_on_false_conc` vs `gate_on_correct`；null 变体另报 `null_mass` |
| E2 长度泛化 | 固定训练长度 × 2 倍间隔长度扫描 | 各长度准确率退化曲线；熵 / max-prob / 门控值漂移 |
| E3 协同设计 | `gate ∈ {query, calib} × norm ∈ {pre, post, mix, seednorm} × LayerScale ∈ {无, 1e-2}` | 逐层残差范数；`gate_depth_ratio`（深层/浅层门控均值比） |

指标定义见 `cage/train.py` 的 docstring；实现修改必须保持定义不变。

## 4. 质量门禁（ci_gate.py）

CI 在每次阶段提交前运行 `python3 ci_gate.py`，包含三层：

1. **语法检查**：仓库内全部 `.py` 可解析；
2. **设计固定项**：D1–D6 以源码检查方式强制（如"不得出现
   `scaled_dot_product_attention`"）；
3. **冒烟测试**：四种 gate × 前向/反向有限性、seednorm、dropout 路径、
   LayerScale、梯度检查点、3 步端到端训练与统计收集。

任一层失败 → 该阶段全部改动被 `git reset --hard` 撤销。`--syntax-only`
模式供 runner 关闭时的 trap 使用。

## 5. 目录

```
cage/model.py        # 模型与门控/归一化实现（D1–D7 所在）
cage/data.py        # 联想回忆合成任务 + 长度扫描
cage/train.py        # 训练、评估与统计量（D5、D8）
ci_gate.py           # 质量门禁（第 4 节）
docs/design_notes.md # 本文档
算力申请.md           # 算力申请材料（apply 阶段检查修复，不得删除）
```
