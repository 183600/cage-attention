# Code Review Report - cage-attention
**审查日期**: 2026-11-26
**审查者**: AI Sub-Agent #2 (Automated)

## 1. 测试套件结果

```
    self.assertTrue(all(entry["status"] == "satisfied"
E   AssertionError: False is not true
=========================== short test summary info ============================
FAILED tests/test_dependency_conformance.py::ConformanceReportTest::test_the_installed_torch_satisfies_the_frozen_pin
!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!
```

**失败测试**:
```
FAILED tests/test_dependency_conformance.py::ConformanceReportTest::test_the_installed_torch_satisfies_the_frozen_pin
```

## 2. 静态分析发现的Bug

\n- **_train_eval.py**: 使用了 global 关键字: 233:    and global edit sensitivity without drawing a second evaluation stream;\n- **_train_recipe.py**: 使用了 global 关键字: 202:    ``late_overfit`` catches the regression the global slope dilutes: a
215:    final regime matches the global slope.  ``val_loss_se_last`` is the
227:    the global verdict: it is set only when the train/val divergence is
480:            f"(@ step {out['best_val_step']}) - the global slope hides it, "\n- **model.py**: 使用了 global 关键字: 25:                 visible-key count equals T and the global divisor is exact;
50:                 ``log(max(t+1, 2))`` instead of the global ``log T``.  On\n- **repro.py**: 使用了 global 关键字: 11:Data order is *not* part of the global RNG: ``cage.data.make_batch`` never
12:reads global state - every split/stream passes its own ``torch.Generator``
15:re-seeding the global RNG elsewhere cannot shuffle the data underneath a
21:the global RNGs as well as for the data streams.  A bool or fractional
93:    one kept the global flag on while its records claimed the opposite.
186:    global kernel flags into a later one.
203:    global _HASHSEED_PINNED
241:    the global RNG state.  The seed is canonicalised with\n- **train.py**: 使用了 global 关键字: 143:    global RNG (dropout) and/or the data stream would silently differ.\n- **_analysis_plan.py**: 使用了 global 关键字: 185:                "information that the global log(max(T, 1e-300)) normalisation mixes into "
822:      every one of its legs is, and the claim's global p-value is the\n- **_config.py**: 使用了 global 关键字: 496:                    "position information that the global log(max(T, 1e-300)) "
500:                      "entropy by the global log(max(T, 1e-300)) and subtracts a*log(max(T, 1e-300)), "\n- **_contrastlog_finalize.py**: 使用了 global 关键字: 1156:        simes_s = (f"global Simes p={float(family_simes['p']):.4f}"
1158:                   "global Simes p=n/a")\n- **_contrasts.py**: 使用了可变默认参数\n- **_design.py**: 使用了 global 关键字: 109:    and never touches the global RNG, so reruns reproduce the same order
916:                 "slots within blocks and never touches the global RNG"),\n- **_e1_runner.py**: 使用了 global 关键字: 251:          "own visible-token count log(max(t+1, 2)) instead of the global "
253:          "gate's global divisor mixes a position-dependent open-ness "
1369:          "the position schedule the global divisor mixes in.  The "
1412:          "late = U-shaped val curves where the global slope hides the "\n- **_environment.py**: 使用了 global 关键字: 25:    global _SOURCE_TREE_SHA
53:    global _RUN_ID, _RUN_STARTED_UTC
314:    call (a previous run in the same process may have left the global flag
920:        "global RNG), so data order cannot be perturbed by other seeding"),\n- **_gates.py**: 使用了 global 关键字: 908:     "data draws do not consume the global RNG (stream-owns-its-state)"),
910:     "accuracy/metrics/statistics evaluation does not consume the global "
1057:    * **RNG isolation**: drawing data leaves the global Python and Torch
1063:      trajectory never reads the global Torch RNG, so it cannot catch a
1072:      which *does* consume the global RNG: only that run verifies the
1104:        the global Torch RNG, so they cannot fail if ``torch.manual_seed``
1203:        """Evaluation must neither consume the global RNG nor change modes.
1206:        an evaluation helper quietly drew from the global generator (or\n- **_synthesis.py**: 使用了可变默认参数\n- **_synthesis.py**: 使用了 global 关键字: 19:    the contrast within-cell instead of comparing against one global arm.
205:    verifies one global recipe and one seed set across all four cells - a

## 3. 科研方法论检查

\n- ✅ 包含对照/消融实验设计\n- ✅ 包含随机化机制\n- ✅ 包含统计检验\n- ✅ 包含可复现性保障机制

## 4. 深度代码审查

\n### model.py\n函数 _null_probs 缺少返回类型注解 (行 908)
函数 __post_init__ 缺少返回类型注解 (行 171)
函数 backward 缺少返回类型注解 (行 497)
函数 forward 缺少返回类型注解 (行 704)
函数 backward 缺少返回类型注解 (行 784)
函数 forward 缺少返回类型注解 (行 951)
函数 backward 缺少返回类型注解 (行 1016)
函数 __init__ 缺少返回类型注解 (行 1132)
函数 forward 缺少返回类型注解 (行 1347)
函数 __init__ 缺少返回类型注解 (行 1736)\n### train.py\n未检测到梯度累积配置\n### metrics.py\n指标实现检查通过

## 5. 总结

共发现 15 个潜在代码问题。

## 6. 修复措施

本次审查已完成自动化检查，发现的潜在问题已记录。

---
*审查完成于 Thu Nov 26 22:09:51 CST 2026*
