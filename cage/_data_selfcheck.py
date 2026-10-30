# cage data: internal self-test.
# Executed into the wrapper module namespace by the loader;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541


def _self_test() -> int:
    """Deterministic self-checks of the split-audit machinery.

    Pins the two properties the distribution-shift decision rests on:
    a same-source held-out draw must not be flagged (calibration), and a
    held-out split drawn from half the class range must be (power) even
    though its total-variation gap stays below the historical
    ``3 x noise floor`` tolerance - the blindness that motivated the
    Monte Carlo-calibrated chi-square.  Also pins the duplicate-row
    accounting used by the split audit, including duplicates that only
    appear when two batches are concatenated.  Run with
    ``python3 -m cage.data``.
    """
    failures = []

    def check(name: str, cond: bool) -> None:
        if not cond:
            failures.append(name)

    vocab, classes = 512, 510
    n_h, n_ref = 1024, 5120

    def draw_counts(k, n, seed):
        gen = torch.Generator().manual_seed(seed)
        return torch.bincount(torch.randint(0, k, (n,), generator=gen),
                              minlength=vocab).to(torch.int64)

    def _tv_between_counts(a, b):
        pa = a.to(torch.float64) / a.sum()
        pb = b.to(torch.float64) / b.sum()
        return 0.5 * float((pa - pb).abs().sum())

    reference = draw_counts(classes, n_ref, 11)
    same_source = draw_counts(classes, n_h, 12)
    p_same, _ = _distribution_shift_p(same_source, reference, n_sim=400,
                                      seed=90210)
    check("same_source_not_flagged", p_same > 0.05)

    half_range = draw_counts(classes // 2, n_h, 13)
    p_shift, chi2_shift = _distribution_shift_p(half_range, reference,
                                                n_sim=400, seed=90210)
    check("half_range_shift_detected", p_shift <= 1.0 / 401.0)
    check("shift_chi2_large", chi2_shift > 900.0)

    floor = _tv_between_counts(same_source, draw_counts(classes, n_h, 14))
    tv_shift = _tv_between_counts(half_range, reference)
    check("old_tv_tolerance_would_pass", tv_shift < 3.0 * floor)

    # Answer-uniformity decision: the balance claim behind "accuracy is the
    # headline metric" must survive a uniform draw and reject a
    # half-range draw at the default audit sample size (5 seeds x 4
    # batches x 256 ~ 5120 samples, ~10 per class).
    n_audit = 5 * 4 * 256
    uni_counts = draw_counts(classes, n_audit, 31)[:classes]
    uni = answer_uniformity_test(uni_counts, n_sim=400, seed=90210)
    check("uniform_draw_supported", uni["uniform_supported"])
    check("uniform_chi2_finite", math.isfinite(uni["chi2"]))
    skewed_counts = draw_counts(classes // 2, n_audit, 32)[:classes]
    skewed = answer_uniformity_test(skewed_counts, n_sim=400, seed=90210)
    check("skewed_draw_rejected", not skewed["uniform_supported"])
    check("skewed_p_below_floor", skewed["p_value"] <= 1.0 / 401.0)
    degenerate = answer_uniformity_test(torch.zeros(1, dtype=torch.int64))
    check("degenerate_input_not_supported",
          not degenerate["uniform_supported"]
          and math.isnan(degenerate["p_value"]))
    malformed = answer_uniformity_test(object())
    check("malformed_answer_counts_abstain",
          not malformed["uniform_supported"]
          and math.isnan(malformed["p_value"]))
    try:
        answer_uniformity_test(torch.tensor([True, False]))
        check("boolean_answer_counts_rejected", False)
    except ValueError:
        check("boolean_answer_counts_rejected", True)
    for bad_counts in (torch.tensor([-1, 2]),
                       torch.tensor([float("nan"), 1.0]),
                       torch.tensor([0.5, 1.5])):
        try:
            answer_uniformity_test(bad_counts)
            check("invalid_answer_counts_rejected", False)
        except ValueError:
            check("invalid_answer_counts_rejected", True)
    try:
        _distribution_shift_p(torch.tensor([1.0, 2.0]),
                              torch.tensor([1.0, 2.0, 3.0]))
        check("mismatched_shift_counts_rejected", False)
    except ValueError:
        check("mismatched_shift_counts_rejected", True)
    try:
        _distribution_shift_p(torch.tensor([0.5, 1.5]),
                              torch.tensor([1.0, 2.0]))
        check("fractional_shift_counts_rejected", False)
    except ValueError:
        check("fractional_shift_counts_rejected", True)
    try:
        _counts(torch.tensor([0, 3]), 3)
        check("out_of_range_audit_value_rejected", False)
    except ValueError:
        check("out_of_range_audit_value_rejected", True)
    try:
        _counts(torch.tensor([0.5, 1.5]), 3)
        check("fractional_audit_value_rejected", False)
    except ValueError:
        check("fractional_audit_value_rejected", True)

    # Generator calibration: a coupled key/value stream must be rejected by
    # the permutation independence test, an independent stream must not be
    # flagged, and the declared generator must pass its own family.  These
    # are the model-free checks that catch a defect shared by every split.
    coupled = _categorical_independence_test(
        torch.arange(3000) % 10, (torch.arange(3000) + 1) % 10,
        10, 10, n_sim=200, seed=0)
    check("generator_coupling_detected",
          coupled["valid"] and coupled["p_value"] <= 1.0 / 201.0)
    independent = _categorical_independence_test(
        torch.arange(3000) % 10, torch.arange(3000) % 7,
        10, 10, n_sim=200, seed=0)
    check("generator_independence_not_flagged",
          independent["valid"] and independent["p_value"] > 0.05)
    calibration = generator_calibration_audit(
        8, 64, "cpu", batch=64, batches=2, seed=0, n_sim=200)
    check("declared_generator_calibrated",
          calibration["calibration_supported"] is True
          and calibration["family"]["abstained"] == [])
    check("value_repeat_birthday_probability",
          abs(_expected_value_repeat_rate(1, 4) - 0.0) < 1e-12
          and abs(_expected_value_repeat_rate(2, 4) - 0.5) < 1e-12
          and _expected_value_repeat_rate(5, 4) == 1.0)

    # Structural difficulty covariates: hand-built sequences pin where the
    # matching key sits, whether the answer value repeats, and whether the
    # naive last-match rule is misled by the query token appearing later
    # as a value.
    feat_split = {
        "idx": torch.tensor([[5, 10, 7, 7, 9, 12, SEP_ID, 7],
                             [3, 20, 4, 20, 5, 9, SEP_ID, 4]]),
        "answer": torch.tensor([7, 20]),
    }
    feat = _difficulty_features(feat_split, 3)
    check("difficulty_key_pos",
          feat["key_pos"].tolist() == [1, 1])
    check("difficulty_value_repeats",
          feat["value_repeats"].tolist() == [0, 1])
    check("difficulty_naive_confusable",
          feat["naive_confusable"].tolist() == [1, 0])
    check("difficulty_joint_encoding",
          _joint_difficulty_values(feat).tolist() == [5, 6])
    # Difficulty strata are the *pre-treatment* covariate used to
    # post-stratify accuracy across lengths.  With one bin per pair the
    # stratum must reproduce the historical joint encoding exactly; with
    # three relative bins the same samples collapse into the middle bin.
    strata_exact = difficulty_strata(
        feat_split["idx"], feat_split["answer"], 3, n_key_bins=3)
    check("difficulty_strata_matches_joint",
          strata_exact["stratum"].tolist() == [5, 6])
    check("difficulty_strata_labels_complete",
          len(strata_exact["labels"]) == 12
          and [row["id"] for row in strata_exact["labels"]]
          == list(range(12)))
    check("difficulty_strata_bounds_cover_pairs",
          strata_exact["key_bin_bounds"][0][0] == 0
          and strata_exact["key_bin_bounds"][-1][1] == 3)
    # A malformed width must be refused rather than silently slicing the
    # wrong columns into an "easy" label.
    try:
        difficulty_features(feat_split["idx"][:, :-1],
                            feat_split["answer"], 3)
        check("difficulty_strata_geometry_rejected", False)
    except ValueError:
        check("difficulty_strata_geometry_rejected", True)

    # Joint-difficulty shift: the 2x2 marginals are identical in both
    # samples, so the separate field tests are blind by construction; the
    # joint test must still detect the reversed dependence.  A same-source
    # draw must not be flagged.
    joint_ref = torch.tensor([300, 200, 200, 300], dtype=torch.int64)
    joint_obs = torch.tensor([200, 300, 300, 200], dtype=torch.int64)

    def _joint_marginals(c):
        return (torch.stack([c[0] + c[1], c[2] + c[3]]),
                torch.stack([c[0] + c[2], c[1] + c[3]]))

    ref_rows, ref_cols = _joint_marginals(joint_ref)
    obs_rows, obs_cols = _joint_marginals(joint_obs)
    check("joint_shift_marginals_equal",
          torch.equal(ref_rows, obs_rows) and torch.equal(ref_cols, obs_cols))
    p_joint, chi2_joint = _distribution_shift_p(
        joint_obs, joint_ref, n_sim=400, seed=90210)
    check("joint_shift_detected", p_joint <= 1.0 / 401.0)
    check("joint_shift_chi2_large", chi2_joint > 100.0)
    jgen = torch.Generator().manual_seed(43)
    joint_same = torch.bincount(
        torch.multinomial(joint_ref.to(torch.float64) / joint_ref.sum(),
                          1000, replacement=True, generator=jgen),
        minlength=4).to(torch.int64)
    p_joint_same, _ = _distribution_shift_p(
        joint_same, joint_ref, n_sim=400, seed=90210)
    check("joint_same_source_not_flagged", p_joint_same > 0.05)

    # A shift in retrieval distance must be detected by the same calibrated
    # test used for the token marginals, even when only the *position*
    # distribution changes.
    gen = torch.Generator().manual_seed(41)
    ref_kp = torch.bincount(torch.randint(0, 16, (4096,), generator=gen),
                            minlength=16)
    obs_kp = torch.bincount(torch.randint(0, 8, (1024,), generator=gen),
                            minlength=16)
    p_kp, _ = _distribution_shift_p(obs_kp, ref_kp, n_sim=400, seed=90210)
    check("key_pos_shift_detected", p_kp <= 1.0 / 401.0)
    same_kp = torch.bincount(torch.randint(0, 16, (1024,), generator=gen),
                             minlength=16)
    p_kp_same, _ = _distribution_shift_p(same_kp, ref_kp, n_sim=400,
                                         seed=90210)
    check("key_pos_same_source_not_flagged", p_kp_same > 0.05)

    # Time-order split contract: contiguous, chronological blocks with no
    # future sample in train, and a crossing assignment must be rejected.
    times = torch.arange(100, dtype=torch.float64)
    t_split = time_ordered_split(100)
    assert_time_ordered(times, t_split)
    check("time_split_contiguous_chronological",
          t_split["train"] == slice(0, 70)
          and t_split["val"] == slice(70, 85)
          and t_split["test"] == slice(85, 100))
    try:
        assert_time_ordered(
            times, {"train": slice(0, 60), "val": slice(50, 80),
                    "test": slice(80, 100)})
        check("time_leakage_rejected", False)
    except ValueError:
        check("time_leakage_rejected", True)
    # A misspelled split name used to be skipped by the chronology loop, so
    # an interleaved ``valid`` block could pass while only train/test were
    # compared.  Unknown names must be refused instead.
    try:
        assert_time_ordered(
            times, {"train": slice(0, 60), "valid": slice(50, 80),
                    "test": slice(80, 100)})
        check("time_unknown_split_name_rejected", False)
    except ValueError:
        check("time_unknown_split_name_rejected", True)
    # Integer timestamps above 2**53 must be compared exactly.  Casting
    # them to float64 collapses adjacent nanosecond instants and would let
    # a training row newer than a validation row pass the chronology gate.
    ns_times = torch.tensor([1_700_000_000_000_000_001,
                             1_700_000_000_000_000_000], dtype=torch.int64)
    try:
        assert_time_ordered(ns_times, {"train": [0], "val": [1],
                                       "test": []})
        check("time_int64_precision_leakage_rejected", False)
    except ValueError:
        check("time_int64_precision_leakage_rejected", True)

    # Group split contract: whole groups on one side, full coverage, and a
    # per-sample crossing assignment must be rejected.
    groups = torch.arange(200, dtype=torch.long) // 4
    g_split = group_disjoint_split(groups, seed=3)
    assert_group_disjoint(groups, g_split)
    check("group_split_covers_every_sample",
          sum(int(v.numel()) for v in g_split.values()) == 200)
    check("group_split_nonempty",
          all(int(v.numel()) > 0 for v in g_split.values()))
    g_split_again = group_disjoint_split(groups, seed=3)
    check("group_split_deterministic",
          all(torch.equal(g_split[k], g_split_again[k]) for k in g_split))
    # A dominant group can make the midpoint pass leave a split empty.  The
    # fallback must donate a group that improves the mass allocation, not
    # the arbitrary last group (which could be the dominant one and leave
    # train with only a single small group).
    imbalanced = torch.arange(3).repeat_interleave(
        torch.tensor([10, 10, 80]))
    imbalanced_split = group_disjoint_split(imbalanced, seed=1)
    check("group_split_fallback_preserves_train_mass",
          int(imbalanced_split["train"].numel()) == 80)
    assert_group_disjoint(imbalanced, imbalanced_split)
    # Non-empty is not enough: the mass rebalance must also avoid putting
    # the dominant group in val/test when the midpoint pass picked a poor
    # permutation (the old fallback produced 10/80/10 here).
    mixed = torch.arange(3).repeat_interleave(torch.tensor([54, 17, 19]))
    mixed_split = group_disjoint_split(mixed, seed=1)
    check("group_split_mass_rebalanced",
          int(mixed_split["train"].numel()) == 54)
    assert_group_disjoint(mixed, mixed_split)
    g_split_alias = group_disjoint_split(groups, seed=3 + 2 ** 32)
    check("group_split_seed_alias_canonical",
          all(torch.equal(g_split[k], g_split_alias[k]) for k in g_split))
    try:
        assert_group_disjoint(
            groups, {"train": torch.arange(0, 100),
                     "val": torch.arange(100, 150),
                     "test": torch.arange(150, 200)})
        check("group_crossing_rejected", False)
    except ValueError:
        check("group_crossing_rejected", True)

    # String group ids (user/document names) must be usable; the encoder
    # maps them to dense codes and the assertion audits every named split.
    str_groups = [f"user{i // 2}" for i in range(20)]
    str_split = group_disjoint_split(str_groups, seed=5)
    assert_group_disjoint(str_groups, str_split)
    check("string_group_split_covers",
          sum(len(str_split[name]) for name in ("train", "val", "test"))
          == len(str_groups))
    try:
        assert_group_disjoint(
            ["alice", "alice", "bob"],
            {"train": [0], "holdout": [1, 2]})
        check("group_crossing_arbitrary_name_rejected", False)
    except ValueError:
        check("group_crossing_arbitrary_name_rejected", True)
    try:
        group_disjoint_split([["unhashable"], ["group"], ["ids"]])
        check("unhashable_group_rejected", False)
    except ValueError:
        check("unhashable_group_rejected", True)
    try:
        group_disjoint_split(3)
        check("scalar_group_rejected", False)
    except ValueError:
        check("scalar_group_rejected", True)

    try:
        assert_time_ordered(torch.tensor([0.0, float("nan"), 2.0]),
                            time_ordered_split(3))
        check("time_nonfinite_rejected", False)
    except ValueError:
        check("time_nonfinite_rejected", True)
    for bad_times in (["2020-01-01", "2020-01-02"],
                      torch.tensor([True, False]),
                      torch.tensor([0.0 + 0.0j, 1.0 + 0.0j])):
        try:
            assert_time_ordered(bad_times, {"train": [0], "test": [1]})
            check("time_non_numeric_rejected", False)
        except ValueError:
            check("time_non_numeric_rejected", True)

    # Boolean split masks must select the masked rows, not be cast to 0/1
    # positions; duplicated or missing sample assignments must be rejected
    # instead of silently auditing the wrong data.
    bool_train = torch.tensor([False, False, True, True])
    bool_test = ~bool_train
    try:
        assert_group_disjoint(
            torch.tensor([0, 0, 1, 1]),
            {"train": bool_train, "test": bool_test})
        check("bool_mask_split_read_correctly", True)
    except ValueError:
        check("bool_mask_split_read_correctly", False)
    try:
        assert_split_partition(
            4, {"train": [0, 1], "val": [1, 2], "test": [3]})
        check("split_duplicate_rejected", False)
    except ValueError:
        check("split_duplicate_rejected", True)
    try:
        assert_split_partition(4, {"train": [0, 1], "val": [2]})
        check("split_missing_sample_rejected", False)
    except ValueError:
        check("split_missing_sample_rejected", True)
    try:
        assert_split_partition(4, {"train": torch.tensor([0.5, 1.0])})
        check("split_float_index_rejected", False)
    except ValueError:
        check("split_float_index_rejected", True)
    try:
        assert_split_partition(
            4, {"train": [0, 0, 1], "val": [2], "test": [3]})
        check("within_split_duplicate_rejected", False)
    except ValueError:
        check("within_split_duplicate_rejected", True)
    try:
        assert_split_partition(
            4, {"train": [0, 1], "valid": [1, 2]})
        check("arbitrary_split_name_audited", False)
    except ValueError:
        check("arbitrary_split_name_audited", True)
    for bad_n in (True, 3.5, 0):
        try:
            assert_split_partition(bad_n, {"train": [0]})
            check("split_bad_n_rejected", False)
        except ValueError:
            check("split_bad_n_rejected", True)
    # An empty split container is a legitimate assignment (the function
    # documents empty splits as allowed), including ``[]`` whose inferred
    # dtype is float32.  It must not be rejected as a floating index array.
    for empty in ([], (), torch.tensor([])):
        try:
            assert_split_partition(4, {"train": [0, 1, 2, 3],
                                       "test": empty})
            check("empty_split_container_allowed", True)
        except ValueError:
            check("empty_split_container_allowed", False)
    try:
        time_ordered_split(10.5)
        check("time_split_fractional_n_rejected", False)
    except ValueError:
        check("time_split_fractional_n_rejected", True)
    try:
        group_disjoint_split(torch.tensor([0.0, 1.0, float("nan"), 2.0, 3.0]))
        check("group_nonfinite_rejected", False)
    except ValueError:
        check("group_nonfinite_rejected", True)

    # Data-geometry contract: too many keys for the selectable ids, or a
    # position budget that cannot hold the sequence, must fail explicitly.
    check("geometry_valid_accepted",
          check_data_geometry(3, 5, max_len=8) is None)
    try:
        check_data_geometry(5, 6)
        check("geometry_too_many_keys_rejected", False)
    except ValueError:
        check("geometry_too_many_keys_rejected", True)
    try:
        check_data_geometry(16, 512, max_len=33)
        check("geometry_max_len_rejected", False)
    except ValueError:
        check("geometry_max_len_rejected", True)
    for geo in ((3.5, 64), (3, 64.0), (3, 64, 7.5)):
        try:
            check_data_geometry(*geo)
            check(f"geometry_fractional_rejected_{geo}", False)
        except ValueError:
            check(f"geometry_fractional_rejected_{geo}", True)
    for args in ((2, 0, 64), (2, -1, 64), (2, 4.0, 64), (2, 4, -1)):
        try:
            make_batch(*args, device="cpu")
            check(f"batch_geometry_rejected_{args}", False)
        except ValueError:
            check(f"batch_geometry_rejected_{args}", True)

    # Within-item length extension: the queried association must survive,
    # keys must stay distinct, and chaining must keep the shorter sequence
    # as a prefix (the nested design behind the E2 linked companion).
    try:
        base_idx, base_tgt, base_meta = make_batch(
            6, 4, 32, "cpu", gen=torch.Generator().manual_seed(71))
        ext_idx, ext_tgt, ext_meta = make_length_linked_batch(
            base_idx, base_meta, 4, 8, 32,
            gen=torch.Generator().manual_seed(72))
        validate_batch(ext_idx, ext_tgt, ext_meta, 32)
        check("linked_width", int(ext_idx.size(1)) == 2 * 8 + 2)
        check("linked_prefix_preserved",
              torch.equal(ext_idx[:, :2 * 4], base_idx[:, :2 * 4]))
        check("linked_query_preserved",
              torch.equal(ext_idx[:, -1], base_idx[:, -1]))
        check("linked_answer_preserved",
              torch.equal(ext_meta["answer"], base_meta["answer"]))
        check("linked_key_pos_preserved",
              torch.equal(ext_meta["key_pos"], base_meta["key_pos"]))
        base_code = association_codes(base_idx, base_meta, 32)["code"]
        check("linked_association_preserved",
              torch.equal(association_codes(ext_idx, ext_meta, 32)["code"],
                          base_code))
        linked_keys = ext_idx[:, :2 * 8:2]
        ordered_keys = linked_keys.sort(dim=1).values
        check("linked_keys_distinct",
              bool((ordered_keys[:, 1:] != ordered_keys[:, :-1]).all()))
        check("linked_new_keys_fresh",
              not bool((
                  ext_idx[:, 2 * 4:2 * 8:2].unsqueeze(2)
                  == base_idx[:, :2 * 4:2].unsqueeze(1)).any()))
        repeat_idx, repeat_tgt, _ = make_length_linked_batch(
            base_idx, base_meta, 4, 8, 32,
            gen=torch.Generator().manual_seed(72))
        check("linked_deterministic",
              torch.equal(ext_idx, repeat_idx)
              and torch.equal(ext_tgt, repeat_tgt))
        nested_idx, nested_tgt, nested_meta = make_length_linked_batch(
            ext_idx, ext_meta, 8, 12, 32,
            gen=torch.Generator().manual_seed(73))
        validate_batch(nested_idx, nested_tgt, nested_meta, 32)
        check("linked_nested_prefix",
              torch.equal(nested_idx[:, :2 * 8], ext_idx[:, :2 * 8]))
        check("linked_nested_association",
              torch.equal(
                  association_codes(nested_idx, nested_meta, 32)["code"],
                  base_code))
        check("linked_same_length_noop",
              torch.equal(
                  make_length_linked_batch(base_idx, base_meta, 4, 4, 32)[0],
                  base_idx))
        try:
            make_length_linked_batch(base_idx, base_meta, 4, 3, 32)
            check("linked_shorter_rejected", False)
        except ValueError:
            check("linked_shorter_rejected", True)
        try:
            make_length_linked_batch(base_idx, base_meta, 4, 40, 32)
            check("linked_geometry_rejected", False)
        except ValueError:
            check("linked_geometry_rejected", True)
    except Exception as exc:  # noqa: BLE001 - surface unexpected failures
        failures.append(f"linked_length_batch:{exc}")

    # Length-sweep geometry: a one-point "sweep" would make the base-to-
    # longest drop and entropy drift identically zero, turning H2b's
    # equivalence test into a tautology.  The helper must reject it.
    check("length_sweep_default",
          length_sweep_lengths(16, 2, 4, 256) == [16, 32, 64])
    try:
        length_sweep_lengths(16, 2, 4, 34)
        check("length_sweep_single_point_rejected", False)
    except ValueError:
        check("length_sweep_single_point_rejected", True)
    zero_sim = answer_uniformity_test(torch.tensor([5, 5]), n_sim=0)
    check("uniformity_zero_sim_no_claim",
          math.isnan(zero_sim["p_value"])
          and not zero_sim["uniform_supported"])
    for args in ((0, 2, 4, 256), (16, 1, 4, 256), (16, 0, 4, 256),
                 (16, 2, 0, 256)):
        try:
            length_sweep_lengths(*args)
            check(f"length_sweep_invalid_rejected_{args}", False)
        except ValueError:
            check(f"length_sweep_invalid_rejected_{args}", True)

    # Outlier/tail audit: a same-distribution held-out draw must not be
    # flagged, while a held-out draw pushed into the rare tail of the same
    # train-fitted reference must be.
    rare_probs = {
        "answer": torch.tensor(
            [0.5, 0.2, 0.1, 0.07, 0.05, 0.04, 0.03, 0.01],
            dtype=torch.float64),
        "key_pos": torch.tensor([0.7, 0.2, 0.08, 0.02], dtype=torch.float64),
        "value_repeats": torch.tensor([0.9, 0.1], dtype=torch.float64),
        "naive_confusable": torch.tensor([0.8, 0.2], dtype=torch.float64),
    }

    def draw_rare_split(seed, n=2048):
        gen = torch.Generator().manual_seed(seed)
        return {field: torch.multinomial(p, n, replacement=True,
                                         generator=gen)
                for field, p in rare_probs.items()}

    odd_values = {}
    for i, name in enumerate(("train:0", "train:1")):
        for field, vals in draw_rare_split(50 + i).items():
            odd_values[(name, field)] = vals
    for seed, name in ((60, "val"), (61, "test")):
        for field, vals in draw_rare_split(seed).items():
            odd_values[(name, field)] = vals
    tail = draw_rare_split(62)
    tail["answer"] = torch.full((2048,), 7, dtype=torch.long)
    tail["key_pos"] = torch.full((2048,), 3, dtype=torch.long)
    tail["value_repeats"] = torch.ones(2048, dtype=torch.long)
    tail["naive_confusable"] = torch.ones(2048, dtype=torch.long)
    for field, vals in tail.items():
        odd_values[("stats", field)] = vals
    odd_names = ["train:0", "train:1", "val", "test", "stats"]
    odd = _outlier_audit(odd_values, ["train:0", "train:1"], odd_names,
                         rare_probs)
    check("outlier_threshold_is_train_tail", odd["rate_train"] <= 0.05)
    check("outlier_same_source_not_flagged",
          not any(odd["shift_by_split"][n] for n in ("val", "test")))
    check("outlier_tail_shift_detected", odd["shift_by_split"]["stats"])
    check("outlier_tail_rate_near_one",
          odd["rate_by_split"]["stats"] > 0.9)

    # Actual-stream replay: same seed and geometry is caught as overlap,
    # two different seeds are disjoint, and re-reading one split at two
    # geometries is same-group and must not fail the leakage gate.
    train_spec = {"seed": 0, "batch": 8, "steps": 3, "group": "train"}
    spec_overlap = actual_stream_overlap_audit(
        4, 64, "cpu",
        train_specs={"train:0": train_spec},
        heldout_specs={
            "val": {"seed": VAL_SEED, "batch": 8, "batches": 2,
                    "group": "val"},
            "test": {"seed": TEST_SEED, "batch": 8, "batches": 2,
                     "group": "test"},
        })
    check("actual_overlap_zero_between_seeds",
          spec_overlap["max_overlap"] == 0)
    same_stream = actual_stream_overlap_audit(
        4, 64, "cpu",
        train_specs={"train:0": {"seed": 5, "batch": 8, "steps": 2,
                                 "group": "train"}},
        heldout_specs={"leaky": {"seed": 5, "batch": 8, "batches": 2,
                                 "group": "val"}})
    check("actual_overlap_detects_same_stream",
          same_stream["max_overlap"]
          == same_stream["streams"]["train:0"]["n_rows"] > 0)
    same_split = actual_stream_overlap_audit(
        4, 64, "cpu",
        train_specs={"train:0": train_spec},
        heldout_specs={
            "val_monitor": {"seed": VAL_SEED, "batch": 8, "batches": 2,
                            "group": "val"},
            "val_final": {"seed": VAL_SEED, "batch": 8, "batches": 3,
                          "group": "val"},
        })
    same_pair = same_split["pairwise_overlap"]["val_monitor|val_final"]
    check("same_group_prefix_overlap_observed", same_pair > 0)
    check("same_group_overlap_not_leakage",
          same_split["max_overlap"] == 0 and same_pair > 0)
    # A split is a seed-owned stream, but make_batch's draws are
    # shape-dependent: the same seed at a different batch size produces a
    # disjoint realization.  Reusing a training seed as a held-out stream
    # at another geometry therefore has zero content overlap while still
    # training and evaluating on the same stream.  The canonical seed key
    # must flag the cross-group collision on its own.
    seed_collision = actual_stream_overlap_audit(
        4, 64, "cpu",
        train_specs={"train:0": {"seed": 6, "batch": 8, "steps": 2,
                                 "group": "train"}},
        heldout_specs={"val": {"seed": 6, "batch": 9, "batches": 2,
                               "group": "val"}})
    check("seed_collision_has_no_content_overlap",
          seed_collision["max_overlap"] == 0
          and seed_collision["pairwise_overlap"]["train:0|val"] == 0)
    check("cross_group_seed_collision_detected",
          seed_collision["cross_group_seed_collisions"] == ["train:0|val"])
    alias_collision = actual_stream_overlap_audit(
        4, 64, "cpu",
        train_specs={"train:0": {"seed": 6, "batch": 8, "steps": 2,
                                 "group": "train"}},
        heldout_specs={"test": {"seed": 6 + 2 ** 32, "batch": 9,
                                "batches": 2, "group": "test"}})
    check("seed_alias_collision_detected",
          alias_collision["cross_group_seed_collisions"] == ["train:0|test"])
    same_group_seed = actual_stream_overlap_audit(
        4, 64, "cpu",
        train_specs={"train:0": {"seed": 6, "batch": 8, "steps": 2,
                                 "group": "train"}},
        heldout_specs={"train_geom2": {"seed": 6, "batch": 9, "batches": 2,
                                       "group": "train"}})
    check("same_group_seed_reuse_not_leakage",
          same_group_seed["cross_group_seed_collisions"] == [])
    # A shared name would overwrite one stream in the replay map and make
    # the train-vs-held-out pairwise lookup raise ``KeyError``; names must
    # be rejected explicitly while the caller can still fix the spec.
    try:
        actual_stream_overlap_audit(
            4, 64, "cpu",
            train_specs={"shared": {"seed": 0, "batch": 8, "steps": 1}},
            heldout_specs={"shared": {"seed": 1, "batch": 8, "batches": 1}})
        check("duplicate_stream_name_rejected", False)
    except ValueError:
        check("duplicate_stream_name_rejected", True)
    # Pair keys are flattened as "<name>|<name>"; a delimiter inside a name
    # can make two different pairs share one key, so reject it explicitly.
    try:
        actual_stream_overlap_audit(
            4, 64, "cpu",
            train_specs={"train|0": {"seed": 0, "batch": 8, "steps": 1}},
            heldout_specs={"val": {"seed": 1, "batch": 8, "batches": 1}})
        check("stream_name_delimiter_rejected", False)
    except ValueError:
        check("stream_name_delimiter_rejected", True)

    for label, kwargs in (
            ("split_zero_batches",
             dict(batch=8, batches=0)),
            ("split_zero_batch",
             dict(batch=0, batches=1)),
    ):
        try:
            split_leakage_audit(4, 64, "cpu", train_seeds=(0,), **kwargs)
            check(f"{label}_rejected", False)
        except ValueError:
            check(f"{label}_rejected", True)
    for label, train_spec, heldout_spec in (
            ("actual_zero_steps",
             {"seed": 0, "batch": 8, "steps": 0},
             {"seed": 1, "batch": 8, "batches": 1}),
            ("actual_zero_batches",
             {"seed": 0, "batch": 8, "steps": 1},
             {"seed": 1, "batch": 8, "batches": 0}),
            ("actual_zero_batch",
             {"seed": 0, "batch": 0, "steps": 1},
             {"seed": 1, "batch": 8, "batches": 1}),
    ):
        try:
            actual_stream_overlap_audit(
                4, 64, "cpu",
                train_specs={"train:0": train_spec},
                heldout_specs={"val": heldout_spec})
            check(f"{label}_rejected", False)
        except ValueError:
            check(f"{label}_rejected", True)

    for label, train_spec in (
            ("actual_nonmapping_spec", None),
            ("actual_missing_seed",
             {"batch": 8, "steps": 1}),
            ("actual_null_seed",
             {"seed": None, "batch": 8, "steps": 1}),
            ("actual_missing_batch",
             {"seed": 0, "steps": 1}),
            ("actual_fractional_seed",
             {"seed": 0.5, "batch": 8, "steps": 1}),
            ("actual_fractional_batch",
             {"seed": 0, "batch": 8.5, "steps": 1}),
    ):
        try:
            actual_stream_overlap_audit(
                4, 64, "cpu",
                train_specs={"train:0": train_spec},
                heldout_specs={"val": {"seed": 1, "batch": 8, "batches": 1}})
            check(f"{label}_rejected", False)
        except ValueError:
            check(f"{label}_rejected", True)

    # Dataset-versioning fingerprint must depend on the batch geometry the
    # stream is realized with, not only on the seed.
    fp_a = stream_fingerprint(4, 64, "cpu", batch=4, seed=77)
    fp_b = stream_fingerprint(4, 64, "cpu", batch=8, seed=77)
    check("fingerprint_geometry_sensitive",
          fp_a["sha256"] != fp_b["sha256"])
    check("fingerprint_repeatable",
          fp_a == stream_fingerprint(4, 64, "cpu", batch=4, seed=77))
    fp_alias = stream_fingerprint(4, 64, "cpu", batch=4,
                                  seed=77 + 2 ** 32)
    check("fingerprint_seed_alias_canonical",
          fp_alias["sha256"] == fp_a["sha256"]
          and fp_alias["stream_key"] == fp_a["stream_key"])
    # A non-integral seed must not be truncated into a different stream
    # before the digest is computed.
    for bad_seed in (77.5, 77.0, "77", True, None):
        try:
            stream_fingerprint(4, 64, "cpu", batch=4, seed=bad_seed)
            check(f"fingerprint_bad_seed_rejected_{bad_seed!r}", False)
        except ValueError:
            check(f"fingerprint_bad_seed_rejected_{bad_seed!r}", True)
    # Versioning must not hash an empty or unrealizable draw: such a digest
    # would certify a dataset the cleaning gate rejects.
    for bad_batch in (0, -1, True, 4.0, "4"):
        try:
            stream_fingerprint(4, 64, "cpu", batch=bad_batch, seed=77)
            check(f"fingerprint_bad_batch_rejected_{bad_batch!r}", False)
        except ValueError:
            check(f"fingerprint_bad_batch_rejected_{bad_batch!r}", True)
    try:
        stream_fingerprint(5, 6, "cpu", batch=4, seed=77)
        check("fingerprint_impossible_geometry_rejected", False)
    except ValueError:
        check("fingerprint_impossible_geometry_rejected", True)

    # A full-stream digest must cover every consumption batch, not only the
    # first one, and must be stable for the same generator sequence.
    def _digest_batches(seed):
        gen = torch.Generator().manual_seed(seed)
        return [make_batch(4, 3, 32, "cpu", gen=gen) for _ in range(2)]

    digest_a = stream_content_digest(_digest_batches(123))
    digest_b = stream_content_digest(_digest_batches(123))
    check("stream_digest_repeatable", digest_a == digest_b)
    check("stream_digest_batch_count",
          digest_a["n_batches"] == 2 and digest_a["n_rows"] == 8)
    batches_rev = list(reversed(_digest_batches(123)))
    check("stream_digest_order_sensitive",
          stream_content_digest(batches_rev)["sha256"]
          != digest_a["sha256"])
    tampered = [(idx.clone(), tgt.clone(), dict(meta))
                for idx, tgt, meta in _digest_batches(123)]
    tampered[-1][1][0, 0] = (tampered[-1][1][0, 0] + 1) % 30
    check("stream_digest_covers_later_batches",
          stream_content_digest(tampered)["sha256"] != digest_a["sha256"])
    try:
        stream_content_digest([(_digest_batches(123)[0][0].reshape(-1),) +
                               _digest_batches(123)[0][1:]])
        check("stream_digest_bad_shape_rejected", False)
    except ValueError:
        check("stream_digest_bad_shape_rejected", True)

    # Sealed-stream guard: selection may read the validation stream only.
    # Both the test and the statistics streams are held out, so using
    # either for a choice would turn it into an implicit validation set;
    # reading them *without* selection (final report / pre-registered gate
    # rules / mechanism statistics) stays allowed.
    check_eval_seed(VAL_SEED, selection=True)
    for bad in (TEST_SEED, STATS_SEED):
        try:
            check_eval_seed(bad, selection=True)
            check(f"selection_refused_on_stream_{bad}", False)
        except ValueError:
            check(f"selection_refused_on_stream_{bad}", True)
        check_eval_seed(bad)

    # Stream-level duplicate accounting: rows repeated across *different*
    # batches of one stream must be counted, not only within one batch.
    gen = torch.Generator().manual_seed(21)
    batch_a, _, _ = make_batch(8, 4, 64, "cpu", gen=gen)
    batch_b, _, _ = make_batch(8, 4, 64, "cpu", gen=gen)
    check("clean_batch_has_no_duplicates",
          _count_duplicate_rows(batch_a) == 0)
    check("stream_duplicates_counted",
          _count_duplicate_rows(torch.cat([batch_a, batch_a[:3], batch_b])) == 3)

    # Label-alignment and reserved-id gates: freshly drawn sparse and dense
    # batches pass the cleaning gate; a target that is neither ``-1`` nor
    # the true continuation is rejected (off-by-one / foreign-label
    # misalignment), and a reserved answer id (PAD/SEP) is rejected even
    # when the paired value and the supervised entry agree with it.
    gen = torch.Generator().manual_seed(22)
    sparse_idx, sparse_tgt, sparse_meta = make_batch(8, 4, 64, "cpu", gen=gen)
    check("validate_accepts_sparse_targets",
          validate_batch(sparse_idx, sparse_tgt, sparse_meta, 64)["n"] == 8)
    dense_idx, dense_tgt, dense_meta = make_batch(
        8, 4, 64, "cpu", dense_lm=True, gen=torch.Generator().manual_seed(23))
    check("validate_accepts_dense_targets",
          validate_batch(dense_idx, dense_tgt, dense_meta, 64)["n"] == 8)

    # The cleaning gate's contract is ValueError for *every* malformed batch.
    # A missing or non-mapping meta object used to escape as KeyError /
    # TypeError from the strict checker's raw subscript, so callers that
    # handle the documented ValueError did not catch it.
    for label, bad_meta in (
            ("missing_query_pos", {"answer": sparse_meta["answer"]}),
            ("missing_answer", {"query_pos": sparse_meta["query_pos"]}),
            ("not_a_mapping", None)):
        try:
            validate_batch(sparse_idx, sparse_tgt, bad_meta, 64)
            check(f"validate_rejects_{label}", False)
        except ValueError:
            check(f"validate_rejects_{label}", True)

    misaligned = sparse_tgt.clone()
    misaligned[:, 0] = (sparse_idx[:, 1] + 1) % 64
    try:
        validate_batch(sparse_idx, misaligned, sparse_meta, 64)
        check("validate_rejects_misaligned_target", False)
    except ValueError:
        check("validate_rejects_misaligned_target", True)

    rows = torch.arange(8)
    reserved_idx = sparse_idx.clone()
    reserved_idx[rows, sparse_meta["key_pos"] + 1] = 0
    reserved_meta = {"query_pos": sparse_meta["query_pos"],
                     "key_pos": sparse_meta["key_pos"],
                     "answer": torch.zeros(8, dtype=torch.long)}
    reserved_tgt = torch.full_like(sparse_tgt, -1)
    reserved_tgt[rows, reserved_meta["query_pos"]] = 0
    try:
        validate_batch(reserved_idx, reserved_tgt, reserved_meta, 64)
        check("validate_rejects_reserved_answer", False)
    except ValueError:
        check("validate_rejects_reserved_answer", True)

    # PAD is equally reserved on the input side: a PAD query/key or a PAD
    # value is not produced by the generator and must not pass the cleaning
    # gate merely because the answer pairing still happens to agree.
    pad_query_idx = sparse_idx.clone()
    pad_query_idx[rows, sparse_meta["query_pos"]] = 0
    pad_query_idx[rows, sparse_meta["key_pos"]] = 0
    try:
        validate_batch(pad_query_idx, sparse_tgt, sparse_meta, 64)
        check("validate_rejects_pad_query_key", False)
    except ValueError:
        check("validate_rejects_pad_query_key", True)

    pad_value_idx = sparse_idx.clone()
    pad_value_col = torch.where(sparse_meta["key_pos"] != 0,
                                torch.ones_like(rows), torch.full_like(rows, 3))
    pad_value_idx[rows, pad_value_col] = 0
    try:
        validate_batch(pad_value_idx, sparse_tgt, sparse_meta, 64)
        check("validate_rejects_pad_value", False)
    except ValueError:
        check("validate_rejects_pad_value", True)

    # A repeated non-query key violates the generator's without-replacement
    # contract even though the queried key is still unique; the cleaning gate
    # must reject it (the old check only looked at the query key).
    dup_nonquery = torch.tensor(
        [[2, 10, 3, 11, 3, 12, 4, 13, SEP_ID, 4]], dtype=torch.long)
    dup_nonquery_meta = {
        "query_pos": torch.tensor([9], dtype=torch.long),
        "key_pos": torch.tensor([6], dtype=torch.long),
        "answer": torch.tensor([13], dtype=torch.long),
    }
    try:
        validate_batch(dup_nonquery, None, dup_nonquery_meta, 32)
        check("validate_rejects_duplicate_nonquery_key", False)
    except ValueError:
        check("validate_rejects_duplicate_nonquery_key", True)

    # The predicate that guards every training batch must agree with the
    # strict checker exactly: it may not accept a batch the strict checker
    # rejects (that would silently disable the cleaning gate), and it may
    # not reject a valid one.
    def gate_agrees(i_, t_, m_, vocab_):
        held = _batch_invariants_hold(i_, t_, m_, vocab_)
        try:
            _validate_batch_strict(i_, t_, m_, vocab_)
            strict = True
        except Exception:  # noqa: BLE001 - any raise is a rejection
            strict = False
        return held == strict

    nan_idx = sparse_idx.clone().float()
    nan_idx[0, 0] = float("nan")
    bad_qp = sparse_meta["query_pos"] + 1
    meta_qp = {"query_pos": bad_qp, "key_pos": sparse_meta["key_pos"],
               "answer": sparse_meta["answer"]}
    meta_kp = {"query_pos": sparse_meta["query_pos"],
               "key_pos": sparse_meta["key_pos"] + 2,
               "answer": sparse_meta["answer"]}
    meta_qp_at_key = {"query_pos": sparse_meta["key_pos"],
                      "key_pos": sparse_meta["key_pos"],
                      "answer": sparse_meta["answer"]}
    soft_tgt = sparse_tgt.clone()
    soft_tgt[0, 0] = -2
    final_tgt = sparse_tgt.clone()
    final_tgt[:, -1] = sparse_meta["answer"] + 1
    dup_key = sparse_idx.clone()
    dup_key[:, 2] = sparse_idx[:, 0]
    for name, (i_, t_, m_) in {
            "valid_sparse": (sparse_idx, sparse_tgt, sparse_meta),
            "valid_dense": (dense_idx, dense_tgt, dense_meta),
            "misaligned_target": (sparse_idx, misaligned, sparse_meta),
            "reserved_answer": (reserved_idx, reserved_tgt, reserved_meta),
            "pad_query_key": (pad_query_idx, sparse_tgt, sparse_meta),
            "pad_value": (pad_value_idx, sparse_tgt, sparse_meta),
            "idx_out_of_range": (sparse_idx.clone().fill_(64), sparse_tgt,
                                 sparse_meta),
            "idx_not_2d": (sparse_idx.reshape(-1), sparse_tgt, sparse_meta),
            "idx_with_nan": (nan_idx, sparse_tgt, sparse_meta),
            "query_pos_out_of_range": (sparse_idx, sparse_tgt, meta_qp),
            "query_pos_not_final": (sparse_idx, None, meta_qp_at_key),
            "key_pos_disagrees": (sparse_idx, sparse_tgt, meta_kp),
            "key_not_unique": (dup_key, sparse_tgt, sparse_meta),
            "target_shape_wrong": (sparse_idx, sparse_tgt.reshape(-1),
                                   sparse_meta),
            "target_out_of_range": (sparse_idx, soft_tgt, sparse_meta),
            "final_target_wrong": (sparse_idx, final_tgt, sparse_meta),
    }.items():
        check(f"invariants_agree_{name}", gate_agrees(i_, t_, m_, 64))

    # End-to-end wiring of the train-fitted sections: a small deterministic
    # same-distribution audit must expose the atypical-tail, coverage and
    # stationarity readouts and must not trip the tail gate.
    e2e = split_leakage_audit(8, 32, "cpu", batch=32, batches=8,
                              train_seeds=(0,))
    check("audit_e2e_outlier_fields",
          e2e["outlier_audit"]["fields"] == list(_OUTLIER_FIELDS))
    check("audit_e2e_no_tail_shift",
          not any(e2e["outlier_audit"]["shift_by_split"].values()))
    check("audit_e2e_joint_field_present",
          "difficulty_joint" in e2e["audit_fields"]
          and set(e2e["shift_p_value"]["difficulty_joint"])
          == {"val", "test", "stats"})
    check("audit_e2e_stationarity_present",
          len(e2e["train_stationarity"]) == len(_STATIONARITY_FIELDS))
    cov = e2e["support_coverage"]
    check("audit_e2e_coverage_sane",
          all(0 <= cov[field]["n_seen_train"] <= cov[field]["n_categories"]
              and all(0.0 <= cov[field]["unseen_by_split"][name]["rate"] <= 1.0
                      for name in ("val", "test", "stats"))
              for field in ("answer", "token", "key", "value")))
    check("audit_e2e_outlier_rates_sane",
          all(0.0 <= v <= 1.0
              for v in e2e["outlier_audit"]["rate_by_split"].values()))

    if failures:
        print("data self-test FAILED: " + ", ".join(failures))
        return 1
    print(f"data self-test: all checks passed "
          f"(same-source p={p_same:.3f}, half-range p={p_shift:.4f}, "
          f"chi2={chi2_shift:.1f}, tv={tv_shift:.3f} vs 3x floor="
          f"{3.0 * floor:.3f}; uniformity p={uni['p_value']:.3f} "
          f"chi2={uni['chi2']:.1f}, skewed p={skewed['p_value']:.4f}; "
          f"key-pos shift p={p_kp:.4f} vs same-source "
          f"p={p_kp_same:.3f})")
    return 0
