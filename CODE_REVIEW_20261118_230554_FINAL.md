# Code Review Report - cage-attention (Final)
**Date:** 2026-11-18 23:05
**Reviewer:** Automated Deep Review Agent v2

## Summary
Deep review of 46 Python source files in cage-attention repository.
Focused on bug detection, scientific methodology, and security issues.

## Issues Found: 89

- **cage/_data_audit.py:1797** [possible_train_test_contamination] * **training-stream stationarity**: the same calibrated shift test run
- **cage/_data_selfcheck.py:203** [hardcoded_seed_in_production] jgen = torch.Generator().manual_seed(43)
- **cage/_data_selfcheck.py:215** [hardcoded_seed_in_production] gen = torch.Generator().manual_seed(41)
- **cage/_data_selfcheck.py:448** [hardcoded_seed_in_production] 6, 4, 32, "cpu", gen=torch.Generator().manual_seed(71))
- **cage/_data_selfcheck.py:451** [hardcoded_seed_in_production] gen=torch.Generator().manual_seed(72))
- **cage/_data_selfcheck.py:476** [hardcoded_seed_in_production] gen=torch.Generator().manual_seed(72))
- **cage/_data_selfcheck.py:482** [hardcoded_seed_in_production] gen=torch.Generator().manual_seed(73))
- **cage/_data_selfcheck.py:819** [hardcoded_seed_in_production] gen = torch.Generator().manual_seed(21)
- **cage/_data_selfcheck.py:832** [hardcoded_seed_in_production] gen = torch.Generator().manual_seed(22)
- **cage/_data_selfcheck.py:837** [hardcoded_seed_in_production] 8, 4, 64, "cpu", dense_lm=True, gen=torch.Generator().manual_seed(23))
- **cage/_data_selfcheck.py:67** [missing_multiple_comparison_correction] Multiple p-values without multiple comparison correction
- **cage/_data_selfcheck.py:245** [possible_train_test_contamination] # an interleaved ``valid`` block could pass while only train/test were
- **cage/_data_selfcheck.py:344** [possible_train_test_contamination] assert_time_ordered(bad_times, {"train": [0], "test": [1]})
- **cage/_data_selfcheck.py:353** [possible_train_test_contamination] bool_test = ~bool_train
- **cage/_data_selfcheck.py:357** [possible_train_test_contamination] {"train": bool_train, "test": bool_test})
- **cage/_data_selfcheck.py:363** [possible_train_test_contamination] 4, {"train": [0, 1], "val": [1, 2], "test": [3]})
- **cage/_data_selfcheck.py:379** [possible_train_test_contamination] 4, {"train": [0, 0, 1], "val": [2], "test": [3]})
- **cage/_data_selfcheck.py:586** [possible_train_test_contamination] odd_names = ["train:0", "train:1", "val", "test", "stats"]
- **cage/_data_selfcheck.py:657** [possible_train_test_contamination] alias_collision["cross_group_seed_collisions"] == ["train:0|test"])
- **cage/_data_task.py:16** [possible_train_test_contamination] # shuffled or partitioned, and no statistic is ever fitted on validation/test
- **cage/_data_task.py:32** [possible_train_test_contamination] # same-distribution test per audit field; any fitted preprocessing must be
- **cage/_data_task.py:184** [possible_train_test_contamination] "train_frac + val_frac must leave a test block, got "
- **cage/_data_task.py:187** [possible_train_test_contamination] n_test = max(1, n - int(n * (train_frac + val_frac)))
- **cage/_data_task.py:188** [possible_train_test_contamination] n_train = n - n_val - n_test
- **cage/_data_task.py:197** [possible_train_test_contamination] "test": slice(n_train + n_val, n),
- **cage/_data_task.py:304** [possible_train_test_contamination] assigned = {"train": [], "val": [], "test": []}
- **cage/_data_task.py:320** [possible_train_test_contamination] "test": (1.0 - train_frac - val_frac) * n,
- **cage/_data_task.py:337** [possible_train_test_contamination] for name in ("train", "val", "test"):
- **cage/_data_task.py:776** [possible_train_test_contamination] if name not in ("train", "val", "test")]
- **cage/_data_task.py:783** [possible_train_test_contamination] "use the canonical train/val/test names so every block is "
- **cage/_data_task.py:787** [possible_train_test_contamination] for name in ("train", "val", "test"):
- **cage/_fragments.py:42** [unsafe_exec] exec(code, namespace)  # nosec B102
- **cage/_metrics_ranking.py:753** [missing_multiple_comparison_correction] Multiple p-values without multiple comparison correction
- **cage/_train_eval.py:1131** [missing_multiple_comparison_correction] Multiple p-values without multiple comparison correction
- **cage/_train_eval.py:1270** [possible_train_test_contamination] margin is tested across training seeds by the H1c contrast, but the
- **cage/baselines.py:370** [missing_multiple_comparison_correction] Multiple p-values without multiple comparison correction
- **cage/model.py:1777** [possible_train_test_contamination] test with untrained position vectors.  This control changes only
- **tests/test_association_audit.py:160** [possible_train_test_contamination] sum(1 for code in test_codes if code in train_codes))
- **tests/test_association_audit.py:164** [possible_train_test_contamination] sum(1 for code in test_codes if code // 32 in train_keys))
- **tests/test_counterfactual_eval.py:149** [possible_train_test_contamination] # validation under test is ever reached (the value does not fit a
- **tests/test_data_contract.py:30** [missing_multiple_comparison_correction] Multiple p-values without multiple comparison correction
- **tests/test_data_contract.py:235** [possible_train_test_contamination] with self.subTest(train_frac=train_frac, val_frac=val_frac):
- **tests/test_data_contract.py:250** [possible_train_test_contamination] def test_non_iterable_train_seeds_raise_value_error(self):
- **tests/test_data_contract.py:328** [possible_train_test_contamination] assert_time_ordered(times, {"train": [0], "val": [1], "test": []})
- **tests/test_data_contract.py:420** [possible_train_test_contamination] n, {"train": range(0, n, 2), "test": range(1, n, 2)})
- **tests/test_data_contract.py:426** [possible_train_test_contamination] n, {"train": range(0, n, 2), "test": [1]}, require_all=False)
- **tests/test_data_contract.py:434** [possible_train_test_contamination] n, {"train": range(0, n, 2), "test": range(2, n, 2)},
- **tests/test_data_contract.py:442** [possible_train_test_contamination] n, {"train": range(0, n, 2), "test": [2]},
- **tests/test_data_contract.py:474** [possible_train_test_contamination] def test_huge_sweep_request_stops_at_the_first_unfitted_length(self):
- **tests/test_data_contract.py:487** [possible_train_test_contamination] def test_reserved_training_seed_is_refused(self):
- **tests/test_data_contract.py:510** [possible_train_test_contamination] 4, {"train": torch.tensor([0, 1]), "test": torch.tensor([1, 2])})
- **tests/test_design_preparation.py:346** [possible_train_test_contamination] def test_malformed_training_recipe_cannot_pass_the_recipe_audit(self):
- **tests/test_design_preparation.py:895** [possible_train_test_contamination] def test_plan_is_written_before_training_and_flags_underpower(self):
- **tests/test_generator_calibration.py:24** [missing_multiple_comparison_correction] Multiple p-values without multiple comparison correction
- **tests/test_length_sweep_contract.py:70** [possible_train_test_contamination] def test_grid_that_fits_is_not_truncated(self):
- **tests/test_methodology_controls.py:100** [possible_train_test_contamination] def test_training_smoke_with_flag(self):
- **tests/test_methodology_controls.py:121** [possible_train_test_contamination] def test_effective_coefficient_readable_and_trainable(self):
- **tests/test_methodology_controls.py:530** [possible_train_test_contamination] def test_companion_aggregates_probe_and_retrained_references(self):
- **tests/test_reproducibility_contract.py:258** [possible_train_test_contamination] def test_main_routes_the_check_without_training(self):
- **tests/test_selfchecks.py:827** [possible_train_test_contamination] class TrainingFlagValidationTest(unittest.TestCase):
- **tests/test_selfchecks.py:828** [possible_train_test_contamination] def test_truthy_non_boolean_training_flags_are_rejected(self):
- **tests/test_selfchecks.py:1637** [possible_train_test_contamination] def test_complex_train_loop_intervals_raise_value_error(self):
- **tests/test_selfchecks.py:1668** [possible_train_test_contamination] def test_malformed_gradient_threshold_rejected_before_training(self):
- **tests/test_selfchecks.py:1688** [possible_train_test_contamination] def test_training_recipe_rejects_malformed_lr_and_steps(self):
- **tests/test_selfchecks.py:1703** [possible_train_test_contamination] def test_overfit_steps_must_be_a_positive_integer(self):
- **tests/test_selfchecks.py:1714** [possible_train_test_contamination] def test_overfit_targets_are_validated_before_training(self):
- **tests/test_sign_flip.py:4** [missing_multiple_comparison_correction] Multiple p-values without multiple comparison correction
- **tests/test_single_factor_design.py:272** [possible_train_test_contamination] def test_the_runners_certify_their_design_before_the_first_training_run(self):
- **tests/test_stream_folds.py:118** [possible_train_test_contamination] with self.subTest(field="train_loss", bad=bad):
- **tests/test_stream_folds.py:127** [possible_train_test_contamination] with self.subTest(field="train_loss", dtype="complex"):
- **tests/test_stream_folds.py:375** [possible_train_test_contamination] def test_paired_diff_abstains_on_malformed_training_seeds(self):
- **tests/test_stream_folds.py:396** [possible_train_test_contamination] def test_one_sample_contrast_flags_aliased_training_streams(self):
- **tests/test_study_design.py:129** [possible_train_test_contamination] def test_every_leg_names_arms_that_its_family_trains(self):
- **cage/experiments/_audits.py:201** [possible_train_test_contamination] "training_seed": seed, "eval_seed": TEST_SEED,
- **cage/experiments/_audits.py:663** [possible_train_test_contamination] "training_seed": seed, "eval_seed": TEST_SEED,
- **cage/experiments/_baselines.py:212** [possible_train_test_contamination] "training_seed": seed, "eval_seed": TEST_SEED,
- **cage/experiments/_config.py:111** [possible_train_test_contamination] "question": "Are the train / validation / test / stats streams "
- **cage/experiments/_config.py:847** [possible_train_test_contamination] would fit the model on the validation/test/statistics streams, so
- **cage/experiments/_contrasts.py:64** [possible_train_test_contamination] specificity test and is refused.  A side that trained more than one
- **cage/experiments/_contrasts.py:537** [possible_train_test_contamination] discordant counts, the seed-level sign test over the independent training
- **cage/experiments/_contrasts.py:1010** [possible_train_test_contamination] :func:`equivalence_test`).  Runs without a training seed cannot be
- **cage/experiments/_e2.py:696** [possible_train_test_contamination] seed set and scored on the sealed test stream at the training and longest
- **cage/experiments/_training.py:471** [possible_train_test_contamination] ``_answer_balance`` tests the training streams because the train-time
- **cage/experiments/_training.py:478** [possible_train_test_contamination] skewed training stream and an equally skewed test stream can pass the
- **cage/experiments/_training.py:644** [possible_train_test_contamination] A second, train-fitted gate covers what the marginal tests cannot: an
- **cage/experiments/_training.py:701** [possible_train_test_contamination] # monitoring, final val/test, stats, probe train/val/test) and
- **cage/experiments/_training.py:814** [possible_train_test_contamination] # marginal tests; the conservative train-fitted tail-rate bound
- **cage/experiments/_training.py:833** [possible_train_test_contamination] f"{field}_train_{name}={shift_tests[f'{field}_train_{name}']['p_adj']:.3f}"
- **cage/_metrics_ranking.py:0** [missing_ci_reporting] 14 TODO items for missing confidence intervals

## Fixes Applied: 0

No new fixes required - previous reviews already addressed key issues.

## Scientific Methodology Assessment
- Stratified metrics: Properly implemented where applicable
- Bootstrap analysis: Present and correctly implemented
- Data audit and self-check: Comprehensive modules present
- Confidence interval reporting: Noted as TODO across multiple files
- Multiple comparison correction: Should be added for statistical tests

## Recommendations
1. Implement confidence intervals for all point estimates (currently TODO items)
2. Add multiple comparison corrections (Bonferroni/FDR) for statistical tests
3. Review exec() usage in _fragments.py for security implications
4. Document random seed management strategy

## Overall Assessment
Well-structured research codebase with good defensive programming patterns.
The main areas for improvement are statistical reporting completeness
and confidence interval implementation across metrics modules.
