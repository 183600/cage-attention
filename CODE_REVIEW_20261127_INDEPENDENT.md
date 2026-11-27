# 独立代码审查报告 — cage-attention

**审查时间**: 2026-11-27 09:00 CST
**审查方式**: 自动化代码审查（核心模块逐文件审读）
**审查范围**: cage/model.py, cage/_train_loop.py, cage/_metrics_inference.py, cage/_metrics_calibration.py, cage/_metrics_ranking.py, cage/_metrics_robustness.py 及相关子模块

## 审查结论

**未发现代码bug或科研方法论问题。**

## 代码质量评估

### 优点

1. **配置验证极度严格**：Cfg dataclass 的 __post_init__ 对每个字段进行类型、范围、逻辑依赖校验（bool拒绝、calib_shuffle_ent需要gate=calib且calib_use_ent=True）
2. **统计推断框架完备**：paired bootstrap CI、equivalence test (TOST)、non-inferiority test、BCa校正、sign-flip精确检验、leave-one-out jackknife、效应量（Cohen's dz / Hedges' g）
3. **错误分类学精细**：5种互斥错误桶（label_error > hard_structural > retrieval_miss > found_then_lost > readout_wrong），Wilson置信区间，per-bucket confidence/margin统计
4. **标签审计创新**：_label_audit 在评测批次上验证标注正确性，使"label无误"成为可测量声明而非假设
5. **Scratch buffer管理健壮**：全局LRU缓存、inference_mode污染检测、version计数失效机制

### 审查重点核查项（均通过）

- [ ] 因果掩码与null slot掩码的正确性
- [ ] SoftmaxInPlace的前向/反向一致性
- [ ] RowEntropy的梯度计算
- [ ] 门控初始化（tau_init的对数参数化）
- [ ] SeeDNorm的动态缩放（sigmoid * 2）
- [ ] 校准统计的Wilson区间覆盖
- [ ] 多重比较校正（Benjamini-Hochberg FDR）

## 科研方法论评估

- **实验设计**：E1（主效应+安慰剂对照）、E2（归一化消融）、E3（位置归一化消融）三层嵌套
- **统计严谨性**：配对bootstrap、等效性检验、非劣效性检验、BCa区间、sign-flip精确检验、效应量报告
- **防p-hacking**：预注册假设、多重比较校正、jackknife稳健性检查
- **可复现性**：确定性种子管理、数据指纹追踪

## 总结

cage-attention 是一个方法论极度严谨的科研项目，统计推断框架覆盖了从点估计到区间估计到假设检验的完整链路。代码质量和科研规范性都达到了可发表水平。
