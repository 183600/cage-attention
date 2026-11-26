# 独立代码审查报告 (2026-11-27)

## 审查范围
- `cage/model.py` - CAGE 校准注意力门控，gate 变体（none/query/calib/null）
- `cage/train.py` - 训练循环，AdamW 权重衰减分组，早停与曲线监控
- `cage/data.py` - 关联召回任务数据生成，长度泛化扫描计划
- `cage/metrics.py` - 配对 bootstrap CI，BCa 区间，精确 sign-flip 检验

## 审查结论

**未发现需要修复的 bug 或科研方法问题。**

### 代码质量亮点

1. **实验设计严谨**：`length_sweep_plan` 记录完整决策（请求/实现/丢弃的长度），防止文档与设计漂移
2. **统计方法全面**：paired bootstrap + BCa + jackknife + sign-flip 四重 companion，FDR 校正含 BH/BY/Holm
3. **配置验证严格**：`Cfg.__post_init__` 拒绝 bool 冒充整数、非有限浮点、非法 gate/norm 组合
4. **消融控制精确**：`calib_shuffle_ent` 用位置置换保留边际分布但破坏关联，隔离信息 vs 容量

### 已审查的关键逻辑点
- [x] 校准门控公式：sigmoid((W_g x - a*log T - c*scale*H/log T)/tau) 的温度参数
- [x] 关联召回数据：key-value 对生成、query 选择、dense_lm 模式
- [x] 训练曲线监控：val_loss 窗口平均、学习率证明退火、过/欠拟合判定
- [x] sign-flip 检验：exact 枚举 vs Monte Carlo 的阈值切换，最小可达 p 值
- [x] AdamW 参数分组：≥2-D 权重加衰减，bias/norm/embedding/scalar 不加
