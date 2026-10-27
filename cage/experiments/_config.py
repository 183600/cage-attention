# cage/experiments/_config.py - Configuration: hypothesis registry, seed validation and Settings.
# Executed into the cage.experiments namespace by __init__.py;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541

# Seeds with fixed roles (independent of every training seed) are defined in
# ``cage.data`` so that every entry point shares one source of truth:
#   VAL_SEED   - model selection only (learning-rate tuning, baseline checks)
#   TEST_SEED  - the final reported numbers *and* the pre-registered gate
#                rules fixed in this module (HB1 rule ceiling, HB4 permuted-
#                label control, HB0/HB6/HB7 competence/saturation marks);
#                never read by any model/hyperparameter/recipe choice
#   STATS_SEED - mechanism statistics (matches the collect_stats default)
# Seal = no *selection* reads these streams; the fixed rules above are
# re-measured on TEST when the CI loop re-runs `--only baselines`, which is
# not re-selection.  See the sealing-note paragraph in ``cage.data``.

HYPO_IDS = ("HB0", "HB1", "HB2", "HB3", "HB4", "HB5", "HB6", "HB7",
            "H1a", "H1b",
            "H1c", "H1d", "H1e", "H1f", "H1g", "H2a", "H2b", "H2c", "H2d",
            "H2e", "H2f", "H2g", "H3a", "H3b", "H3c")

# Every hypothesis entry may carry ``"sesoi"``: the smallest effect size of
# interest per metric (same metric keys as the pre-registered contrasts).
# A contrast is only read as supporting its hypothesis when it is both
# FDR-significant *and* at least the SESOI: at the seed counts used here an
# effect of 0.1 accuracy points can be statistically detectable while being
# scientifically meaningless, and "significant but tiny" must not be
# reported as "the mechanism works".  SESOI units follow the metric: accuracy
# and error-margin shares in probability points, correlations/ratios in
# their own scale (see each entry).  The registry is content-hashed into
# every result record, so the SESOI is part of the pre-registration.
#
# A hypothesis whose prediction is a *null/equivalence* claim (a quantity is
# expected to stay inside a bound, e.g. H2b's length-stable H/log(T)) lists
# the metric under ``"sesoi_test"`` with value ``"equivalence"``: the SESOI
# then acts as the equivalence bound and the contrast is supported only when
# the 95% CI of the effect lies inside +-bound (bootstrap TOST,
# ``cage.metrics.equivalence_test``).  "Not significant" on a difference
# test is *not* evidence for a null prediction, so the two decision rules
# must not be conflated.
HYPOTHESES = {
    "HB0": {
        "experiment": "all",
        "question": "Is a mechanism claim readable at all - did the models "
                    "under test actually learn the task at this step count?",
        "hypothesis": "The depth/mechanism contrasts are only informative "
                      "about the mechanism when the trained arms are in a "
                      "competent regime; an underfit grid produces "
                      "'no difference' or 'depth needed' readings for "
                      "reasons unrelated to the mechanism.",
        "prediction": "at least one trained arm (a ladder rung in "
                      "baselines, or a grid arm in E1/E2/E3) has a "
                      "seed-clustered one-sided 95% family-wise lower bound "
                      "(Bonferroni over arms) above chance "
                      "1/(vocab - 2) on the sealed test stream",
        "decision": "the competence prerequisite is a precondition, not a "
                    "mechanism test: when no arm clears chance, every "
                    "mechanism contrast in that family is marked "
                    "claim-ineligible (no claim may be read from an "
                    "underfit grid) and the depth requirement (HB7) is "
                    "'not determinable' rather than 'required'.  The lower "
                    "bound resamples training seeds and keeps the shared "
                    "evaluation-stream measurement error, so re-scoring more "
                    "seeds on the same test stream cannot buy precision",
    },
    "HB1": {
        "experiment": "baselines",
        "question": "Is the recall task well posed with a known solution?",
        "hypothesis": "The pair structure admits an exact parameter-free "
                      "lookup rule because keys are distinct per sequence.",
        "prediction": "rule accuracy ~ 1.0 (ceiling) and unique_key_rate = 1.0",
        "decision": "rule accuracy < 0.99 (point estimate; the Wilson 95% "
                    "CI is reported to show the precision of the finite "
                    "evaluation stream) indicates a broken task/data "
                    "construction and invalidates everything above it",
        "sesoi": {"accuracy": 0.01},
    },
    "HB2": {
        "experiment": "baselines",
        "question": "Does the task require content-based lookup rather than "
                    "a simple linear decision?",
        "hypothesis": "A linear model on bag-of-token counts cannot express "
                      "key-value lookup; its accuracy stays near chance "
                      "1/(vocab-2) even after tuning.",
        "prediction": "the 95% bootstrap CI of (tuned probe accuracy - chance) "
                      "stays below +chance, i.e. the probe is confidently "
                      "below 2x chance; the CI resamples seeds and adds one "
                      "shared Binomial(n_per_seed) scoring draw per "
                      "replication on the common evaluation stream (not one "
                      "draw per seed), so the bound is certified against "
                      "measurement noise without pseudo-replicating the "
                      "stream; the one-sided bootstrap p-value for H0: "
                      "margin >= +chance certifies this at adequate power",
        "decision": "The quantitative near-chance test "
                    "(cage.baselines.probe_vs_chance_test) must be "
                    "*certified* (CI upper bound < +chance); a CI that "
                    "merely straddles the bound is 'inconclusive' - not "
                    "certified, possibly underpowered - and the reported "
                    "seeds_for_certification (planned on seed spread plus "
                    "the non-shrinking common-stream noise floor) says "
                    "whether more seeds/the probe stream can settle it; only "
                    "a CI "
                    "entirely at or above +chance (state 'violated') is "
                    "evidence that answer information leaks into the "
                    "bag-of-tokens features (or a shortcut exists), which "
                    "invalidates the simple-baseline comparison",
    },
    "HB3": {
        "experiment": "baselines",
        "question": "Are the train / validation / test / stats streams "
                    "disjoint (no data leakage)?",
        "hypothesis": "The split is assigned before generation and each split "
                      "owns a disjoint RNG stream, so no sample is shared "
                      "between splits and no fitted statistic is computed on "
                      "held-out data.",
        "prediction": "exact-duplicate overlap between the training stream "
                      "and each held-out stream is 0; no audit field shows a "
                      "distribution gap beyond same-stream sampling noise "
                      "(Monte Carlo-calibrated chi-square, Benjamini-"
                      "Hochberg-adjusted over the field x split family)",
        "decision": "a shared sample, or an adjusted distribution-shift "
                    "p-value below the pre-registered FDR, invalidates every "
                    "reported number",
    },
    "HB4": {
        "experiment": "baselines",
        "question": "Does the pipeline leak the answer into the training "
                    "signal (negative control)?",
        "hypothesis": "With the supervised answers permuted across the batch "
                      "during training, no true answer can be learned, so "
                      "accuracy on the unchanged test stream must stay at "
                      "chance.",
        "prediction": "the upper bound of the seed-clustered 95% bootstrap "
                      "CI of (control accuracy - chance) stays below the "
                      "pre-registered absolute tolerance "
                      "(Settings.negative_control_tol, default 0.02); the "
                      "clustered interval resamples training seeds and keeps "
                      "the shared finite-stream measurement error, so the "
                      "control cannot pass by scoring a lucky small test "
                      "stream (the seed-only and pooled Wilson intervals are "
                      "descriptive companions)",
        "decision": "a control clearly above chance means answer "
                    "information leaks through the data/pipeline (or a "
                    "shortcut exists) and every accuracy reported above it "
                    "is invalid",
    },
    "HB5": {
        "experiment": "baselines",
        "question": "Can the training loop fit anything at all "
                    "(gradient/loss sanity)?",
        "hypothesis": "A small model trained on one fixed tiny batch with "
                      "dropout off should drive the training loss to ~0 and "
                      "the batch accuracy to 1.0; if it cannot, gradients, "
                      "the loss or the precision path are broken.",
        "prediction": "final train loss <= 0.1 and train accuracy = 1.0",
        "decision": "failure invalidates the training loop and blocks the "
                    "interpretation of every grid result (necessary, not "
                    "sufficient: data/label problems can survive "
                    "memorisation)",
    },
    "HB6": {
        "experiment": "baselines",
        "question": "Does the simplest learned model (one attention layer) "
                    "already solve the recall task, making depth (and the "
                    "E3 deep-contribution premise) unnecessary?",
        "hypothesis": "The recall readout needs match-then-shift "
                      "composition (find the key that equals the query, "
                      "then read the token one position later), which a "
                      "single attention layer cannot express, so the tuned "
                      "1-layer transformer should stay clearly below the "
                      "parameter-free rule ceiling.",
        "prediction": "the tuned 1-layer standard transformer is competent "
                      "(seed-clustered 95% lower bound above chance) and its "
                      "seed-clustered 95% *upper* bound on the test stream "
                      "lies at least the pre-registered SESOI (0.01 "
                      "accuracy) below the rule-ceiling point estimate - "
                      "i.e. the plausible gap is practically meaningful, "
                      "not merely statistically non-zero",
        "decision": "the baseline is a complexity control, not a pipeline "
                    "validity gate.  The rung is read as *practically at the "
                    "ceiling* when its seed-clustered 95% *lower* bound "
                    "lies within SESOI of the ceiling (one-sided equivalence: the "
                    "whole interval sits inside the practically-saturated "
                    "band), in which case every depth-based motivation (E3) "
                    "must be annotated that a 1-layer model suffices at this "
                    "scale.  By contrast, 'depth measurably required' needs "
                    "the seed-clustered 95% *upper* bound to stay at least "
                    "SESOI below the ceiling: a statistically detectable but "
                    "practically negligible gap does not earn the complexity. "
                    "The bare 'CI includes the ceiling point estimate' "
                    "criterion is absence of evidence, so it is recorded as a "
                    "companion (hb6_supported_ci) and never decides.  A rung "
                    "at chance (HB0 competence prerequisite failed) shows "
                    "only that this step count did not solve the task - it is "
                    "not evidence that depth is required",
        "sesoi": {"accuracy": 0.01},
    },
    "HB7": {
        "experiment": "baselines",
        "question": "At which depth does the tuned standard transformer "
                    "saturate on the recall task, and is the 8-layer grid "
                    "depth-justified at this scale?",
        "hypothesis": "Depth helps the match-then-shift readout, but it "
                      "saturates at some small depth: once a rung's accuracy "
                      "is no longer distinguishable from the "
                      "parameter-free ceiling, deeper grids (E1/E3 at 8 "
                      "layers) can only test what depth unlocks, not whether "
                      "depth is needed.",
        "prediction": "the ladder's saturation depth d* - the smallest "
                      "*competent* depth whose seed-clustered 95% lower "
                      "bound lies within the pre-registered SESOI (0.01 accuracy) "
                      "of the rule-ceiling point estimate (one-sided "
                      "equivalence: the whole interval is inside the "
                      "practically-saturated band) - is reported (None when "
                      "no ladder rung is equivalent-within-SESOI); adjacent-"
                      "depth paired bootstrap CIs plus bootstrap-TOST "
                      "equivalence checks against the same SESOI show where "
                      "the gain stops being practically meaningful; the depth "
                      "demand is only readable when at least one rung is "
                      "competent (HB0: seed-clustered 95% lower bound above "
                      "chance)",
        "decision": "pre-registered readout: depth beyond two layers is "
                    "'measurably required' only when d* is None or d* > 2 "
                    "*and* at least one rung is above chance; for d* <= 2 "
                    "every depth-based motivation (E3's deep-contribution "
                    "premise, the 8-layer E1 setting) must carry the "
                    "annotation that a 2-layer standard transformer already "
                    "saturates the task at this scale; when no rung clears "
                    "chance the ladder did not solve the task at this "
                    "step count, so the depth demand is 'not determinable' "
                    "(never 'required') and E3's depth framing is "
                    "unverified.  Saturation is an equivalence claim with the "
                    "registered SESOI bound, never the bare 'CI reaches the "
                    "ceiling' (absence of evidence) reading",
        "sesoi": {"accuracy": 0.01},
    },
    "H1a": {
        "experiment": "e1",
        "question": "Does the calibrated gate use attention concentration / "
                    "length information beyond the query?",
        "hypothesis": "The query-only gate has no access to the attention "
                      "distribution, so |gate_vs_entropy| and "
                      "|gate_vs_maxprob| are both clearly larger for calib "
                      "than for query.",
        "prediction": "|gate_vs_entropy|(calib) - |gate_vs_entropy|(query) > 0 "
                      "and |gate_vs_maxprob|(calib) - "
                      "|gate_vs_maxprob|(query) > 0; the same direction holds "
                      "for the rank (Spearman) correlations as a robustness "
                      "check.  The registered "
                      "accuracy companion (calib - none and calib - query) "
                      "is a one-sided non-inferiority check: the calibrated "
                      "gate must not lose accuracy beyond the registered "
                      "SESOI to its controls - a loss larger than that bound "
                      "refutes the mechanism, while a gain is not required",
        "decision": "paired bootstrap 95% CI of the seed differences "
                    "excludes 0 after Benjamini-Hochberg FDR control across "
                    "the E1 contrasts (q=0.05); the accuracy companions are "
                    "one-sided non-inferiority tests against the registered "
                    "SESOI (lower bound above -SESOI), not directional "
                    "superiority claims",
        # Both distribution readouts are in the pre-registered family: the
        # claim is that calib reads the attention distribution while query
        # does not, and max-prob is the other observable summary of that
        # distribution.  Registering only entropy would leave half of the
        # stated evidence exploratory.
        "sesoi": {"accuracy": 0.01, "abs_gate_vs_entropy": 0.05,
                  "abs_gate_vs_maxprob": 0.05},
        # The one-sided accuracy companion covers the registered calib
        # contrasts (calib - none, calib - query) and is registered as
        # non-inferiority under ``sesoi_test``, not as a direction.  The
        # query-vs-none completeness companion measures whether *any* gate
        # moves accuracy and is also not a signed claim, so it registers its
        # own undirected key.
        "direction": {"abs_gate_vs_entropy": "+",
                      "abs_gate_vs_maxprob": "+",
                      "query_none_accuracy": None},
        "sesoi_test": {"accuracy": "noninferior"},
    },
    "H1b": {
        "experiment": "e1",
        "question": "Does the calibrated gate 'doubt' itself on false "
                    "concentration?",
        "hypothesis": "On samples with high max-prob but wrong answers the "
                      "calibrated gate closes more than on high-max-prob "
                      "correct samples, more so than the query gate.",
        "prediction": "gate_margin = gate_on_correct - gate_on_false_conc "
                      "is larger for calib than for query, and the sign is "
                      "stable when the high-maxprob threshold is swept "
                      "(0.3/0.5/0.7/0.9)",
        "decision": "paired bootstrap 95% CI excludes 0 after "
                    "Benjamini-Hochberg FDR control across the E1 contrasts "
                    "(q=0.05); the pre-registered threshold 0.5 is the "
                    "decision point, the sweep only reports robustness",
        "sesoi": {"gate_margin": 0.02},
        "direction": {"gate_margin": "+"},
    },
    "H1c": {
        "experiment": "e1",
        "question": "Does the explicit null slot suppress writes when the "
                    "model is wrong, and does the null arm stay at least as "
                    "accurate as the no-gate baseline?",
        "hypothesis": "null_mass is higher on errors than on correct "
                      "answers (the null slot absorbs the write).  Because "
                      "the null arm claims a dedicated slot is a benign, "
                      "useful addition (not an accuracy trade), its accuracy "
                      "must not fall below the no-gate ``none`` arm - a "
                      "significant accuracy loss would mean the null slot "
                      "buys its suppression by breaking recall.",
        "prediction": "null_mass_error_margin > 0 (and the same margin "
                      "remains positive on the high-maxprob false-"
                      "concentration subset); and the null arm's "
                      "accuracy relative to ``none`` stays inside the "
                      "pre-registered non-inferiority bound (null - none "
                      ">= -SESOI, i.e. the 95% CI of null - none lies above "
                      "the negative bound rather than materially below "
                      "``none``)",
        "decision": "paired bootstrap 95% CI excludes 0 for the margin after "
                    "Benjamini-Hochberg FDR control across the E1 contrasts "
                    "(q=0.05); the null-vs-none accuracy row is a *null* "
                    "reverse check, read as a one-sided non-inferiority test: "
                    "the lower bound of the CI of null - none must stay above "
                    "the registered bound (-accuracy SESOI), so a confidence "
                    "interval entirely below it refutes the claim rather than "
                    "being read as a null result",
        "sesoi": {"null_mass_error_margin": 0.02,
                  "null_mass_false_conc_margin": 0.02, "accuracy": 0.01},
        "direction": {"null_mass_error_margin": "+",
                      "null_mass_false_conc_margin": "+"},
        # The accuracy companion is a null/non-inferiority prediction: its
        # bound is the accuracy SESOI above, tested one-sided (CI above the
        # negative bound), not a directional superiority claim.
        "sesoi_test": {"accuracy": "noninferior"},
    },
    "H1d": {
        "experiment": "e1",
        "question": "Does each calibration term carry its own contribution "
                    "(no dead component)?",
        "hypothesis": "The calib gate claims two additive terms (length "
                      "-a*log T and concentration -c*Hbar).  Within the "
                      "identical calib architecture, removing the "
                      "concentration term must dismantle the gate/entropy "
                      "association, and removing a term that does work must "
                      "show up in accuracy or in the false-concentration "
                      "margin - a component removable without effect is not "
                      "doing work.",
        "prediction": "|gate_vs_entropy| collapses toward the query gate in "
                      "the no-concentration arms and stays clearly positive "
                      "in full calib; accuracy and gate_margin of full calib "
                      "are at least as high as in the ablated arms "
                      "(the fixed-length main effect of the length term is "
                      "expected to be small because at a fixed T it is a "
                      "constant absorbed by the gate bias; H2c tests it "
                      "where it matters, across lengths).  The complete 2x2 "
                      "additionally reports the two-term interaction as a "
                      "two-sided additivity check: a near-zero interaction "
                      "supports reading the terms as additive, a large one "
                      "requires each contribution to be stated conditional "
                      "on the other",
        "decision": "paired bootstrap 95% CIs of accuracy(calib - ablated "
                    "arm) and gate_margin(calib - ablated arm) exclude 0 "
                    "for each *single-term* ablated arm after "
                    "Benjamini-Hochberg FDR control across the E1 family "
                    "(q=0.05); the joint both-terms-off cell is not a "
                    "single-factor row and enters only the registered 2x2 "
                    "factorial design.  That design is read completely: the "
                    "concentration main effect is directional, and the "
                    "length x concentration interaction is a two-sided "
                    "additivity check (no registered sign) that must still "
                    "exceed its SESOI to count as practically meaningful - "
                    "a large interaction means each term's contribution is "
                    "conditional on the other, so the additive claim is "
                    "not supported as stated.  The mechanism check "
                    "validates that each ablation removed the intended "
                    "pathway and is reported alongside, not as a decision "
                    "criterion",
        "sesoi": {"accuracy": 0.01, "gate_margin": 0.02,
                  "abs_gate_vs_entropy": 0.05},
        "direction": {"accuracy": "+", "gate_margin": "+",
                      "abs_gate_vs_entropy": "+",
                      # The interaction rows reuse the same display metrics
                      # but are explicitly two-sided: an undirected key keeps
                      # them from inheriting the main-effect sign.
                      "interaction_accuracy": None,
                      "interaction_gate_margin": None,
                      "interaction_abs_gate_vs_entropy": None},
    },
    "H1e": {
        "experiment": "e1",
        "question": "Does the calibrated gate rank the model's own errors "
                    "above the query-only gate's ranking (the operational, "
                    "imbalance-aware reading of 'the gate doubts itself')?",
        "hypothesis": "Errors are the rare class after convergence, so the "
                      "mechanism's operational value is a rare-positive "
                      "ranking claim: low calibrated-gate values should "
                      "concentrate on wrong answers more than the query "
                      "gate's values do.  Under this imbalance the metric "
                      "choice matters: PR-AUC is the headline (ROC-AUC stays "
                      "optimistic as the negative population grows), and the "
                      "recall of the most doubtful 10% is its operational "
                      "companion.",
        "prediction": "prauc_lift_gate_to_error(calib) - "
                      "prauc_lift_gate_to_error(query) > 0 and "
                      "recall_at_10pct_gate_to_error(calib) - "
                      "recall_at_10pct_gate_to_error(query) > 0 (lift = "
                      "PR-AUC / prevalence, 1.0 = random ranking; ROC-AUC is "
                      "reported as the optimistic companion, not the "
                      "decision metric)",
        "decision": "paired bootstrap 95% CI of the pre-registered "
                    "imbalance-aware readout excludes 0 after "
                    "Benjamini-Hochberg FDR control across the E1 contrasts "
                    "(q=0.05); seeds whose stats stream contains no errors "
                    "leave the metric undefined and are recorded as such, "
                    "never imputed",
        "sesoi": {"prauc_lift_gate_to_error": 0.1,
                  "recall_at_10pct_gate_to_error": 0.02},
        "direction": {"prauc_lift_gate_to_error": "+",
                      "recall_at_10pct_gate_to_error": "+"},
    },
    "H1f": {
        "experiment": "e1",
        "question": "Is the calibrated gate's association with the attention "
                    "distribution carried by the real entropy *information*, "
                    "or only by the presence of an extra data-dependent "
                    "input (active placebo control)?",
        "hypothesis": "An active-control arm with the identical calib "
                      "architecture, parameter count, computation and "
                      "optimiser, but whose gate reads a fixed "
                      "position-shuffled copy of the attention entropy (same "
                      "values and marginal distribution, decorrelated from "
                      "each queried row's own attention), must lose the "
                      "gate/entropy association and must not reproduce the "
                      "full gate's advantage over the query-only gate.  The "
                      "leave-one-out ablations remove the term *and* its "
                      "parameter, so they cannot separate 'the gate reads "
                      "the concentration information' from 'an extra "
                      "data-dependent input helps'; this control can.",
        "prediction": "|gate_vs_entropy|(calib) - |gate_vs_entropy|("
                      "calib_shuf_ent) > 0 (the shuffled signal contains no "
                      "information about the read row, so the association "
                      "must collapse), and accuracy and gate_margin of full "
                      "calib exceed the placebo arm's - a placebo that "
                      "matches full calib would refute the calibration "
                      "story and show the effect is capacity, not "
                      "attention-distribution information",
        "decision": "paired bootstrap 95% CIs of the three calib - placebo "
                    "differences exclude 0 in the pre-registered directions "
                    "after Benjamini-Hochberg FDR control across the E1 "
                    "family (q=0.05).  The mechanism leg validates that the "
                    "placebo actually severed the signal pathway; the "
                    "accuracy/gate_margin legs are the specificity test - a "
                    "reversed or null read there means the calibration "
                    "benefit is not attributable to the entropy information",
        "sesoi": {"accuracy": 0.01, "gate_margin": 0.02,
                  "abs_gate_vs_entropy": 0.05},
        "direction": {"accuracy": "+", "gate_margin": "+",
                      "abs_gate_vs_entropy": "+"},
    },
    "H1g": {
        "experiment": "e1",
        "question": "Is the variant difference in attention-distribution use "
                    "visible at the correlation level itself: calib's gate "
                    "is associated with entropy / max-prob, while the "
                    "query-only gate is not strongly associated?",
        "hypothesis": "H1a tests that calib's |r| is larger than query's.  "
                      "A difference of two near-zero associations would not "
                      "show that calib reads the distribution, and "
                      "'query not significant' would not show that it does "
                      "not.  This correlation-level claim makes both halves "
                      "falsifiable: calib's one-sample |r| must clear a "
                      "practical floor, and query's must lie inside a "
                      "pre-registered near-zero band.",
        "prediction": "calib |r|(gate, entropy) and |r|(gate, max-prob) > 0 "
                      "by at least the SESOI; query's two |r| values lie "
                      "inside the equivalence bound 0.10",
        "decision": "one-sample bootstrap 95% CIs through the same "
                    "Benjamini-Hochberg E1 family (q=0.05): calib rows are "
                    "directional (CI excludes 0 and effect >= SESOI), query "
                    "rows are bootstrap-TOST equivalence tests (CI inside "
                    "+-bound).  A query CI that merely includes 0 while "
                    "spanning the bound is 'not established', not evidence "
                    "of no association",
        "sesoi": {"calib_abs_gate_vs_entropy": 0.05,
                  "calib_abs_gate_vs_maxprob": 0.05,
                  "query_abs_gate_vs_entropy": 0.10,
                  "query_abs_gate_vs_maxprob": 0.10},
        "direction": {"calib_abs_gate_vs_entropy": "+",
                      "calib_abs_gate_vs_maxprob": "+"},
        "sesoi_test": {"query_abs_gate_vs_entropy": "equivalence",
                       "query_abs_gate_vs_maxprob": "equivalence"},
    },
    "H2a": {
        "experiment": "e2",
        "question": "Does length calibration slow the degradation beyond "
                    "the training length?",
        "hypothesis": "Because the gate subtracts a*log(T) and the "
                      "log(T)-normalised entropy, calib degrades less than "
                      "none when evaluated at 2x-spaced lengths.",
        "prediction": "accuracy_drop(calib) < accuracy_drop(none) and "
                      "accuracy@longest(calib) > accuracy@longest(none); "
                      "the fitted accuracy slope on log(T) across the whole "
                      "sweep is also less negative for calib (endpoint-only "
                      "reads can hide a mid-sweep reversal)",
        "decision": "paired bootstrap 95% CI of the differences excludes 0 "
                    "after Benjamini-Hochberg FDR control across the E2 "
                    "contrasts (q=0.05)",
        "sesoi": {"accuracy_at_longest": 0.01, "accuracy_drop": 0.01,
                  "accuracy_slope_log_len": 0.01},
        "direction": {"accuracy_at_longest": "+", "accuracy_drop": "-",
                      "accuracy_slope_log_len": "+"},
    },
    "H2b": {
        "experiment": "e2",
        "question": "Do the attention statistics drift with length as the "
                    "calibration assumes?",
        "hypothesis": "Raw entropy grows with log(T), while the quantity the "
                      "gate normalises by, H/log(T), is length-stable.",
        "prediction": "raw entropy drift (longest - base) > 0 (directional) "
                      "and the normalised entropy drift H/log(T) lies inside "
                      "the pre-registered equivalence bound, with the "
                      "calibrated gate drift inside its bound too",
        "decision": "paired bootstrap 95% CIs, FDR-controlled across the E2 "
                    "contrasts (q=0.05): the raw entropy drift CI excludes 0 "
                    "with a positive mean (difference test); the H/log(T) "
                    "drift and the gate drift are *equivalence* (TOST) tests "
                    "- each is supported only when its CI lies inside "
                    "+-SESOI, so a CI that merely includes 0 while spanning "
                    "a drift larger than the bound is 'not established', "
                    "not length-stability evidence",
        "sesoi": {"entropy_drift": 0.1, "entropy_norm_drift": 0.05,
                  "gate_drift": 0.02},
        "direction": {"entropy_drift": "+"},
        # Metrics whose pre-registered prediction is a *null/equivalence*
        # claim: the SESOI above is then the bound the 95% CI must lie
        # inside (metrics.equivalence_test), not a minimum meaningful
        # effect.
        "sesoi_test": {"entropy_norm_drift": "equivalence",
                       "gate_drift": "equivalence"},
    },
    "H2c": {
        "experiment": "e2",
        "question": "Is the length term (not the concentration term) the "
                    "specific cause of calib's length robustness?",
        "hypothesis": "The advantage of calib over none beyond the training "
                      "length comes from subtracting a*log(T): removing the "
                      "length term (concentration-only arm) must lose the "
                      "advantage, while removing the concentration term "
                      "(length-only arm) must keep it.",
        "prediction": "accuracy_at_longest(calib) > accuracy_at_longest("
                      "calib_ent_only) and accuracy_drop(calib) < "
                      "accuracy_drop(calib_ent_only); calib_len_only stays "
                      "close to the full calib arm (within the registered "
                      "bound on both accuracy_at_longest and accuracy_drop)",
        "decision": "paired bootstrap 95% CI of calib - calib_ent_only on "
                    "accuracy_at_longest excludes 0 after "
                    "Benjamini-Hochberg FDR control across the E2 contrasts "
                    "(q=0.05).  The calib_len_only arm is a *null* reverse "
                    "check, not a superiority row: its 95% CI for calib - "
                    "calib_len_only must lie inside +-SESOI on both "
                    "accuracy_at_longest and accuracy_drop (bootstrap TOST, "
                    "same FDR family); a non-significant difference is not "
                    "enough to call the term dispensable",
        "sesoi": {"accuracy_at_longest": 0.01, "accuracy_drop": 0.01,
                  # Separately registered bounds for the reverse check:
                  # the displayed metrics are shared with the directional
                  # calib - calib_ent_only rows, so the null prediction
                  # needs its own keys in the registry.
                  "calib_minus_len_only_accuracy_at_longest": 0.01,
                  "calib_minus_len_only_accuracy_drop": 0.01},
        "direction": {"accuracy_at_longest": "+", "accuracy_drop": "-"},
        "sesoi_test": {
            "calib_minus_len_only_accuracy_at_longest": "equivalence",
            "calib_minus_len_only_accuracy_drop": "equivalence",
        },
    },
    "H2d": {
        "experiment": "e2",
        "question": "Does calib's length robustness exceed the closest "
                    "simpler gate (query), not only the no-gate baseline?",
        "hypothesis": "The length-robustness claim is only about the "
                      "calibration terms if the calibrated gate degrades "
                      "less than the query-only gate, which shares the same "
                      "attention and gating architecture but lacks the "
                      "length and concentration terms.  A gain over "
                      "``none`` alone could be produced by the presence of "
                      "any learned gate.",
        "prediction": "accuracy_at_longest(calib) > "
                      "accuracy_at_longest(query) and "
                      "accuracy_drop(calib) < accuracy_drop(query), with "
                      "the same sign on the whole-sweep accuracy slope on "
                      "log(T)",
        "decision": "paired bootstrap 95% CI excludes 0 in the "
                    "pre-registered directions after Benjamini-Hochberg FDR "
                    "control across the E2 contrasts (q=0.05)",
        "sesoi": {"accuracy_at_longest": 0.01, "accuracy_drop": 0.01,
                  "accuracy_slope_log_len": 0.01},
        "direction": {"accuracy_at_longest": "+", "accuracy_drop": "-",
                      "accuracy_slope_log_len": "+"},
    },
    "H2e": {
        "experiment": "e2",
        "question": "Does the concentration term carry an average "
                    "contribution across the length sweep, and is the two-"
                    "term design additive (full 2x2 over the length sweep)?",
        "hypothesis": "With all four calibration-term cells trained on the "
                      "same length sweep, the concentration term's effect "
                      "averaged over both length-term levels must be "
                      "measured as well as the length term's (H2c); the "
                      "length x concentration interaction is the "
                      "two-sided companion that checks the terms are not "
                      "merely interchangeable.",
        "prediction": "concentration main effect on accuracy_at_longest > 0 "
                      "and on accuracy_drop < 0; the interaction "
                      "(length effect under concentration on minus under "
                      "concentration off) is reported as a two-sided "
                      "readout, since the design tests additivity rather "
                      "than a signed interaction",
        "decision": "paired bootstrap 95% CIs through the same "
                    "Benjamini-Hochberg family as the H2c rows (q=0.05): "
                    "the concentration main effect carries a registered "
                    "direction and SESOI; the interaction is two-sided "
                    "(no registered direction) but must still exceed its "
                    "SESOI to be practically meaningful",
        "sesoi": {"accuracy_at_longest": 0.01, "accuracy_drop": 0.01},
        # Direction is keyed separately from the display metric so the
        # two-sided interaction rows (same metrics) do not inherit the
        # directional prediction of the main-effect rows.
        "direction": {"concentration_accuracy_at_longest": "+",
                      "concentration_accuracy_drop": "-"},
    },
    "H2f": {
        "experiment": "e2",
        "question": "Does the full calibrated gate actually learn the "
                    "explicit corrections its mechanism claim names?",
        "hypothesis": "The calibrated gate is defined as "
                      "sigmoid((W_g x - a*log T - c*scale*Hbar)/tau).  The "
                      "mechanism claim is therefore an explicit correction "
                      "with a>0 (subtract the log-length entropy growth) and "
                      "c*scale>0 (subtract the normalised concentration).  "
                      "A full-calib arm whose learned a or c*scale is "
                      "non-positive is not implementing that correction, "
                      "whatever its endpoint accuracy.",
        "prediction": "one-sample across training seeds: a_mean > 0 and "
                      "c_effective_mean > 0; a gate that passes the "
                      "endpoint accuracy test with a non-positive "
                      "coefficient is a capacity effect, not the claimed "
                      "explicit calibration",
        "decision": "one-sample bootstrap 95% CI excludes 0 in the "
                    "registered direction after Benjamini-Hochberg FDR "
                    "control across the E2 contrasts (q=0.05).  This is a "
                    "mechanism-manipulation check: failure annotates H2a/H2b "
                    "(and the E2 coefficients table) as not showing the "
                    "named explicit correction, even when an endpoint "
                    "accuracy contrast passes",
        # Coefficients are in pre-sigmoid gate units; 0.01 is a deliberately
        # small practical floor so a numerically non-zero but negligible
        # coefficient is not sold as an implemented mechanism.
        "sesoi": {"a_mean": 0.01, "c_effective_mean": 0.01},
        "direction": {"a_mean": "+", "c_effective_mean": "+"},
    },
    "H2g": {
        "experiment": "e2",
        "question": "Is the log-T normalisation of the concentration term "
                    "load-bearing, or would raw entropy do just as well "
                    "across lengths?",
        "hypothesis": "The calibrated gate claims a length-stable "
                      "concentration signal: Hbar = H / log T.  H2b measures "
                      "that H/log T is stable, but a stable *statistic* does "
                      "not show that the model needs the normalisation.  An "
                      "arm with the identical calib architecture, parameter "
                      "count, computation and length term, but reading raw "
                      "row-mean entropy (Cfg.calib_ent_norm=False), therefore "
                      "isolates the normalisation.  At the training length "
                      "the two parameterisations differ by a constant factor "
                      "that the learned c can absorb, so the informative "
                      "divergence is at the extrapolation lengths, where H "
                      "keeps growing with log T and an unnormalised term "
                      "over-corrects relative to the registered gate.",
        "prediction": "accuracy_at_longest(calib) > "
                      "accuracy_at_longest(calib_raw_ent) and "
                      "accuracy_drop(calib) < accuracy_drop(calib_raw_ent); "
                      "the full arm's concentration coefficient was already "
                      "shown positive by H2f, so this is a comparison between "
                      "two implemented corrections, not between a term and "
                      "no term",
        "decision": "paired bootstrap 95% CI of calib - calib_raw_ent "
                    "excludes 0 in the pre-registered directions after "
                    "Benjamini-Hochberg FDR control across the E2 contrasts "
                    "(q=0.05); a CI including 0 while spanning the SESOI is "
                    "'inconclusive', a confidently positive effect below the "
                    "SESOI is 'practically negligible', and an FDR-significant "
                    "effect in the opposite direction refutes the "
                    "normalisation claim rather than being read as a null "
                    "result",
        "sesoi": {"accuracy_at_longest": 0.01, "accuracy_drop": 0.01},
        "direction": {"accuracy_at_longest": "+", "accuracy_drop": "-"},
    },
    "H3a": {
        "experiment": "e3",
        "question": "Does dynamic normalisation make deep layers write more "
                    "into the residual stream?",
        "hypothesis": "seednorm (per-token dynamic scale) raises the "
                      "deep/early gate ratio relative to pre-LN.",
        "prediction": "gate_depth_ratio(seednorm) > gate_depth_ratio(pre) "
                      "and the deep/early residual-norm ratio "
                      "resid_norm_last_over_first is larger under seednorm "
                      "than under pre-LN; the registered "
                      "accuracy companion (calib - query within each norm "
                      "and LayerScale level) is a one-sided non-inferiority "
                      "check: the calibrated gate must not lose accuracy "
                      "beyond the registered SESOI to the query-only gate - "
                      "a loss larger than that bound refutes the co-design "
                      "story, while a gain is not required",
        "decision": "paired bootstrap 95% CI excludes 0 after "
                    "Benjamini-Hochberg FDR control across the E3 contrasts "
                    "(q=0.05); the accuracy companion is a one-sided "
                    "non-inferiority test against the registered SESOI "
                    "(lower bound above -SESOI), not a directional "
                    "superiority claim",
        "sesoi": {"gate_depth_ratio": 0.05,
                  "resid_norm_last_over_first": 0.05,
                  "accuracy": 0.01,
                  "norm_gate_depth_ratio": 0.05,
                  "norm_resid_norm_last_over_first": 0.05,
                  "norm_accuracy": 0.01},
        # The average-own-level main effects (each norm level vs pre-LN,
        # collapsed over gate and LayerScale) reuse the H3a mechanism
        # direction for seednorm; post/mix are two-sided companions that
        # measure whether a normalisation level moves the readout at all.
        # The companions register their own (undirected) keys so they cannot
        # inherit seednorm's sign through the shared display metric.  The
        # accuracy companion is non-inferiority under ``sesoi_test``, not a
        # directional superiority claim, so it has no direction entry.
        "direction": {"gate_depth_ratio": "+",
                      "resid_norm_last_over_first": "+",
                      "norm_gate_depth_ratio": "+",
                      "norm_resid_norm_last_over_first": "+",
                      "postmix_norm_gate_depth_ratio": None,
                      "postmix_norm_resid_norm_last_over_first": None},
        "sesoi_test": {"accuracy": "noninferior"},
    },
    "H3b": {
        "experiment": "e3",
        "question": "Does LayerScale change how strongly deep layers write?",
        "hypothesis": "LayerScale (1e-2) damps the initial write strength "
                      "and thereby changes the residual-norm profile and "
                      "the gate depth ratio.",
        "prediction": "the primary readout "
                      "resid_norm_last_over_first shifts between LayerScale "
                      "off and on; gate_depth_ratio is a secondary readout",
        "decision": "paired bootstrap 95% CI of the primary readout "
                    "excludes 0 after Benjamini-Hochberg FDR control across "
                    "the E3 contrasts (q=0.05); the secondary readout is "
                    "reported for context and is not a decision criterion",
        "sesoi": {"resid_norm_last_over_first": 0.05,
                  "gate_depth_ratio": 0.05},
    },
    "H3c": {
        "experiment": "e3",
        "question": "Is the calibrated gate's benefit specific to the "
                    "dynamic (seednorm) normalisation - a co-design "
                    "interaction - rather than uniform across norms?",
        "hypothesis": "The calib-over-query accuracy advantage is larger "
                      "under seednorm than under pre-LN (positive "
                      "interaction), because the learned calibration terms "
                      "read the same per-token dynamic scale; the "
                      "deep/early gate ratio shows the same interaction.",
        "prediction": "difference-in-differences "
                      "[acc(calib|seednorm) - acc(query|seednorm)] - "
                      "[acc(calib|pre) - acc(query|pre)] > 0 at LayerScale "
                      "off, with the same sign for gate_depth_ratio - and "
                      "the marginal calib-vs-query contrasts must remain "
                      "controlled for the interaction to be attributable",
        "decision": "paired bootstrap 95% CI of the per-seed "
                    "difference-in-differences excludes 0 after "
                    "Benjamini-Hochberg FDR control across the E3 contrasts "
                    "(q=0.05)",
        "sesoi": {"accuracy": 0.01, "gate_depth_ratio": 0.05},
        "direction": {"accuracy": "+", "gate_depth_ratio": "+"},
    },
}


def _validate_training_seeds(seeds):
    """Validate the configured training-seed set; return it as an int tuple.

    A training-seed set must be non-empty, integral, duplicate-free and
    disjoint from the reserved held-out evaluation streams:

    * **duplicates** break the premises the seed-level statistics rest on.
      A paired contrast pairs runs by seed (``_paired_diff``); a seed that
      appears twice on a side is counted twice in ``diffs``, so ``n`` is
      inflated and the bootstrap CI too narrow, and the "independent
      seeds" behind a Mean +- Std are not independent at all;
    * **reserved seeds** (``VAL_SEED`` / ``TEST_SEED`` / ``STATS_SEED``)
      would fit the model on the validation/test/statistics streams, so
      the reported numbers stop being held-out.

    Comparisons use :func:`cage.data.stream_seed_key`, the 32-bit key the
    CPU stream generators actually consume: ``0`` and ``2**32`` are the
    same stream and count as duplicates, and a reserved-seed alias is
    refused exactly like the reserved seed itself.

    Checked both at ``Settings`` construction and from ``_run_context``
    (which every runner reaches before training), so a post-construction
    attribute assignment - the CLI applies ``--seeds`` that way - cannot
    bypass it either.
    """
    # A bare integer / None is not a seed sequence; string-like and mapping
    # objects are iterable but do not name seeds either (``bytes`` would
    # silently become its integer byte values).  Translate these into the
    # documented ValueError instead of leaking ``tuple()``'s TypeError.
    if isinstance(seeds, (str, bytes)) or isinstance(seeds, Mapping):
        raise ValueError(
            "seeds must be an iterable of training seeds, got "
            f"{type(seeds).__name__}")
    try:
        seeds = tuple(seeds)
    except TypeError as exc:
        raise ValueError(
            "seeds must be an iterable of training seeds, got "
            f"{type(seeds).__name__}") from exc
    if not seeds:
        raise ValueError(
            "seeds must be a non-empty sequence of training seeds")
    if any(isinstance(s, bool) or not isinstance(s, int) for s in seeds):
        raise ValueError(
            f"training seeds must be plain integers, got {seeds!r}")
    seeds = tuple(int(s) for s in seeds)
    seen, duplicates = {}, []
    for s in seeds:
        key = stream_seed_key(s)
        if key in seen:
            duplicates.extend((seen[key], s))
        else:
            seen[key] = s
    if duplicates:
        raise ValueError(
            f"duplicate training seeds/streams {sorted(set(duplicates))} in "
            f"{list(seeds)}: a repeated seed is the same run twice, so paired "
            f"contrasts would double-count it (inflated n, too-narrow CI) and "
            f"the seed-level Mean +- Std would not average independent runs")
    reserved = sorted(
        s for s in seeds
        if any(stream_seed_key(s) == stream_seed_key(r)
               for r in RESERVED_EVAL_SEEDS))
    if reserved:
        raise ValueError(
            f"training seeds {reserved} are reserved held-out evaluation "
            f"streams {RESERVED_EVAL_SEEDS}: training on them would fit the "
            f"model on validation/test data")
    return seeds


@dataclass
class Settings:
    device: str = "cpu"
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 8
    max_len: int = 256
    vocab: int = 512
    batch: int = 64
    n_pairs: int = 16
    steps: int = 1500
    seeds: tuple = (0, 1, 2, 3, 4)
    eval_batch: int = 256
    eval_batches: int = 4
    stats_batch: int = 128
    stats_batches: int = 4
    probe_steps: int = 400
    probe_batch: int = 64
    # The simple-baseline controls are tuned harder than the mechanism grids
    # on purpose: "the complex model beats a tuned simple baseline" is only
    # credible when the simple baseline got its best shot.  The probe grid
    # spans 30x (a diverged high-lr point is recorded and excluded, not
    # read as "cannot do better"); the depth ladder re-tunes the learning
    # rate and step budget per depth so the shallow rungs are not
    # handicapped by inheriting an under-trained recipe.
    #
    # The probe evaluation stream is deliberately wider than the grid
    # evaluation stream: HB2 certifies "the tuned probe stays below
    # 2x chance", and with a narrow stream the per-seed binomial noise
    # (sd ~ sqrt(p(1-p)/n)) can exceed the chance-level margin, making the
    # certificate fail for power reasons on a clean pipeline.  512x16
    # samples shrink that noise well below the margin; the one-sided
    # p-value and the seeds_for_certification power plan are reported next
    # to the decision so residual underpower is visible either way.
    probe_eval_batch: int = 512
    probe_eval_batches: int = 16
    probe_lr_grid: tuple = (1e-2, 3e-3, 1e-3, 3e-4)
    # The simple baseline is tuned over (lr, steps), not lr alone: a fixed
    # 400-step budget can make a linear probe look weaker than it is.  An
    # empty tuple keeps the legacy single-budget behaviour for callers that
    # pass ``probe_steps`` explicitly.
    probe_steps_grid: tuple = (200, 400, 800)
    depth_ladder: tuple = (1, 2, 4)
    # The depth ladder tunes (lr, steps) per rung.  A fixed step budget can
    # make a shallow rung look weaker merely because its best learning rate
    # was selected at an under-trained budget; the multipliers are relative
    # to settings.steps so a --steps override propagates.  The multi-axis
    # boundary audit flags a winner at the 2x edge instead of silently
    # treating it as resolved.
    depth_step_multipliers: tuple = (1.0, 2.0)
    dropout: float = 0.0        # model regularisation (recorded in the recipe)
    # Deterministic kernels are on by default: fixed seeds alone do not make
    # CUDA runs repeatable when an op may pick a non-deterministic algorithm.
    # The official grids therefore run under the stricter mode unless the
    # caller explicitly opts out (--no-deterministic), and the resolved value
    # is stamped into every record.
    deterministic: bool = True
    lr: float | None = None
    lr_grid: tuple = (1e-3, 3e-4, 1e-4)
    # Second, deliberately separate accounting of baseline fairness: the
    # controlled grid above freezes ONE recipe (tuned on gate=none) for every
    # arm, which makes the gate contrast clean but can handicap a baseline
    # whose best lr/temperature differs.  This companion gives every arm its
    # own validation-selected recipe (lr, and calib_tau_init for calib) and
    # then re-pairs the arms on the sealed test stream.  It is an exploratory
    # benchmark, never a replacement for the pre-registered controlled
    # family: a calib gain that survives both readings is the robust claim.
    per_variant_tuning: bool = True
    # Selection seeds for the per-variant tuned account.  Default to the
    # pre-registered claim floor: the shared/controlled recipe already
    # refuses to freeze a point on fewer seeds than a claim would need, and
    # the tuned baseline comparison should not be decided by a smaller
    # (noisier) argmax than that.  ``min(per_variant_tune_seeds,
    # len(seeds))`` is used, so quick configs still work.
    per_variant_tune_seeds: int = 3
    calib_tau_grid: tuple = (0.5, 0.9, 1.5)
    curve_points: int = 10      # validation checks per run (0 disables)
    val_batches: int = 2        # validation batches per check (fixed stream)
    max_error_cases: int = 64   # per gate x seed, dumped to disk
    fdr_q: float = 0.05         # BH false-discovery rate for contrast families
    # Pre-registered floor on the number of independent training seeds a
    # contrast needs before it may support a claim: a 0.2% effect next to a
    # 0.5% seed-level std is not evidence, and a two-seed "win" is noise.
    min_seeds_for_claim: int = 3
    # Prospective design plan (written before the first grid run): the
    # two-sided per-test alpha and target power used to translate the
    # pre-registered SESOI into a sensitivity budget.  These are planning
    # parameters, not a second decision rule - the FDR-controlled contrasts
    # still carry the pre-registered readout.
    design_alpha: float = 0.05
    power_target: float = 0.8
    # Run-order randomisation: every seed is a complete block containing all
    # planned cells once, and the cell order inside each block is a fixed
    # pseudo-random permutation.  The schedule is deterministic and recorded,
    # so it is reproducible while ensuring that a time-varying machine state
    # cannot be aliased with one treatment arm.
    randomize_run_order: bool = True
    run_order_seed: int = 20240517
    overfit_steps: int = 300    # HB5 tiny-batch overfit check
    overfit_batch: int = 8
    overfit_lr: float = 3e-3
    negative_control_tol: float = 0.02  # HB4 pre-registered tolerance
    out_dir: pathlib.Path = field(default_factory=lambda: pathlib.Path("results"))

    def __post_init__(self) -> None:
        # Fail at construction on an invalid training-seed set instead of
        # letting a duplicated reserve-dataset seed flow into a grid; the
        # runner entry point re-checks (see ``_validate_training_seeds``)
        # because ``--seeds`` is applied to the attribute afterwards.
        self.seeds = _validate_training_seeds(self.seeds)
        # Numeric geometry and loop sizes are used as lengths, ranges and
        # loop bounds downstream.  A bool, fractional value or string would
        # otherwise be coerced somewhere deeper (or fail halfway through a
        # run), so validate and canonicalise them at the configuration
        # boundary like ``Cfg`` does.
        def _positive_int_field(name):
            raw = getattr(self, name)
            if isinstance(raw, bool):
                raise ValueError(f"{name} must be an integer, not bool")
            try:
                value = operator.index(raw)
            except TypeError as exc:
                raise ValueError(
                    f"{name} must be an integer, got {raw!r}") from exc
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {raw!r}")
            setattr(self, name, int(value))
            return int(value)

        for name in ("d_model", "n_heads", "n_layers", "batch", "steps",
                     "eval_batch", "eval_batches", "stats_batch",
                     "stats_batches", "val_batches", "probe_batch",
                     "probe_steps", "probe_eval_batch",
                     "probe_eval_batches", "overfit_steps",
                     "overfit_batch"):
            _positive_int_field(name)
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model={self.d_model} is not a positive multiple of "
                f"n_heads={self.n_heads}")
        raw_curve_points = self.curve_points
        if isinstance(raw_curve_points, bool):
            raise ValueError(
                f"curve_points must be a non-negative integer, got "
                f"{raw_curve_points!r}")
        try:
            curve_points = operator.index(raw_curve_points)
        except TypeError as exc:
            raise ValueError(
                f"curve_points must be a non-negative integer, got "
                f"{raw_curve_points!r}") from exc
        if curve_points < 0:
            raise ValueError(
                f"curve_points must be a non-negative integer, got "
                f"{raw_curve_points!r}")
        self.curve_points = int(curve_points)
        # The per-run error-case cap is zero-or-positive and is passed
        # straight to ``collect_stats_and_errors``, which requires an
        # integer.  Validate it here as well so a malformed cap cannot
        # abort a grid after its training runs have already been written.
        raw_max_cases = self.max_error_cases
        if isinstance(raw_max_cases, bool):
            raise ValueError(
                f"max_error_cases must be a non-negative integer, got "
                f"{raw_max_cases!r}")
        try:
            max_cases = operator.index(raw_max_cases)
        except TypeError as exc:
            raise ValueError(
                f"max_error_cases must be a non-negative integer, got "
                f"{raw_max_cases!r}") from exc
        if max_cases < 0:
            raise ValueError(
                f"max_error_cases must be a non-negative integer, got "
                f"{raw_max_cases!r}")
        self.max_error_cases = int(max_cases)

        def _positive_int_tuple(name, allow_empty=False):
            raw = getattr(self, name)
            # ``bytes`` is iterable and would silently become its byte
            # values; a string/mapping is also not a tuple of settings.
            if isinstance(raw, (str, bytes)) or isinstance(raw, Mapping):
                raise ValueError(
                    f"{name} must be an iterable of positive integers, "
                    f"got {type(raw).__name__}")
            try:
                values = tuple(raw)
            except TypeError as exc:
                raise ValueError(
                    f"{name} must be an iterable of positive integers, "
                    f"got {raw!r}") from exc
            if not values and not allow_empty:
                raise ValueError(f"{name} must not be empty")
            checked = []
            for value in values:
                if isinstance(value, bool):
                    raise ValueError(
                        f"{name} entries must be integers, got {value!r}")
                try:
                    item = operator.index(value)
                except TypeError as exc:
                    raise ValueError(
                        f"{name} entries must be integers, got {value!r}"
                    ) from exc
                if item < 1:
                    raise ValueError(
                        f"{name} entries must be >= 1, got {value!r}")
                checked.append(int(item))
            setattr(self, name, tuple(checked))

        _positive_int_tuple("depth_ladder")
        _positive_int_tuple("probe_steps_grid", allow_empty=True)
        raw_multipliers = self.depth_step_multipliers
        if isinstance(raw_multipliers, (str, bytes)) \
                or isinstance(raw_multipliers, Mapping):
            raise ValueError(
                "depth_step_multipliers must be an iterable of finite "
                f"positive numbers, got {type(raw_multipliers).__name__}")
        try:
            multipliers = tuple(raw_multipliers)
        except TypeError as exc:
            raise ValueError(
                "depth_step_multipliers must be an iterable of finite "
                f"positive numbers, got {raw_multipliers!r}") from exc
        checked_multipliers = []
        for value in multipliers:
            if isinstance(value, bool):
                raise ValueError(
                    "depth_step_multipliers entries must be numbers, "
                    f"got {value!r}")
            try:
                number = float(value)
            except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
                raise ValueError(
                    "depth_step_multipliers entries must be finite "
                    f"positive numbers, got {value!r}") from exc
            if not (math.isfinite(number) and number > 0.0):
                raise ValueError(
                    "depth_step_multipliers entries must be finite "
                    f"positive numbers, got {value!r}")
            # The multiplier is applied to ``steps`` to build the depth
            # ladder's step grid.  A finite multiplier can still overflow
            # that product (1e308 * 1500 -> inf), which used to surface as
            # an incidental OverflowError from ``round`` during the audit,
            # after the baselines runner had already started.  Reject the
            # unrepresentable combination at the configuration boundary.
            try:
                scaled = number * float(self.steps)
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError(
                    "depth_step_multipliers must keep steps * multiplier "
                    f"finite, got {value!r} at steps={self.steps}") from exc
            if not math.isfinite(scaled):
                raise ValueError(
                    "depth_step_multipliers must keep steps * multiplier "
                    f"finite, got {value!r} at steps={self.steps}")
            checked_multipliers.append(number)
        self.depth_step_multipliers = tuple(checked_multipliers)
        for name in ("dropout", "negative_control_tol"):
            raw = getattr(self, name)
            if isinstance(raw, bool):
                raise ValueError(f"{name} must be a finite number, got {raw!r}")
            try:
                number = float(raw)
            except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
                raise ValueError(
                    f"{name} must be a finite number, got {raw!r}") from exc
            if not math.isfinite(number):
                raise ValueError(
                    f"{name} must be a finite number, got {raw!r}")
            if name == "dropout" and not 0.0 <= number <= 1.0:
                raise ValueError(
                    f"dropout must be in [0, 1], got {raw!r}")
            if name == "negative_control_tol" and number < 0.0:
                raise ValueError(
                    f"negative_control_tol must be >= 0, got {raw!r}")
            setattr(self, name, number)
        for name in ("lr", "overfit_lr"):
            raw = getattr(self, name)
            if raw is None and name == "lr":
                continue
            if isinstance(raw, bool):
                raise ValueError(
                    f"{name} must be a finite positive number, got {raw!r}")
            try:
                number = float(raw)
            except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
                raise ValueError(
                    f"{name} must be a finite positive number, got "
                    f"{raw!r}") from exc
            if not (math.isfinite(number) and number > 0.0):
                raise ValueError(
                    f"{name} must be a finite positive number, got {raw!r}")
            setattr(self, name, number)
        check_data_geometry(self.n_pairs, self.vocab, self.max_len)
        raw_q = self.fdr_q
        try:
            q = float(raw_q)
        except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
            raise ValueError(
                f"fdr_q must be a finite number strictly between 0 and 1, "
                f"got {raw_q!r}") from exc
        if not (math.isfinite(q) and 0.0 < q < 1.0):
            raise ValueError(
                f"fdr_q must be a finite number strictly between 0 and 1, "
                f"got {raw_q!r}")
        self.fdr_q = q
        for name in ("design_alpha", "power_target"):
            raw = getattr(self, name)
            try:
                value = float(raw)
            except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
                raise ValueError(
                    f"{name} must be a finite number strictly between 0 and "
                    f"1, got {raw!r}") from exc
            if not (math.isfinite(value) and 0.0 < value < 1.0):
                raise ValueError(
                    f"{name} must be a finite number strictly between 0 and "
                    f"1, got {raw!r}")
            setattr(self, name, value)
        if not isinstance(self.randomize_run_order, bool):
            raise ValueError(
                f"randomize_run_order must be a bool, got "
                f"{self.randomize_run_order!r}")
        # ``configure_determinism`` coerces its argument with ``bool``, so a
        # string such as ``"False"`` would silently *enable* deterministic
        # kernels (and TF32 suppression) while the run record still showed
        # the caller's value.  Validate the semantic switch at the
        # configuration boundary, like the Cfg ablation flags.
        if not isinstance(self.deterministic, bool):
            raise ValueError(
                f"deterministic must be a bool, got {self.deterministic!r}")
        if isinstance(self.run_order_seed, bool) or not isinstance(
                self.run_order_seed, int):
            raise ValueError(
                f"run_order_seed must be a plain integer, got "
                f"{self.run_order_seed!r}")
        floor = _positive_integer(self.min_seeds_for_claim)
        if floor is None:
            raise ValueError(
                f"min_seeds_for_claim must be a positive integer, got "
                f"{self.min_seeds_for_claim!r}")
        self.min_seeds_for_claim = int(floor)
        if not isinstance(self.per_variant_tuning, bool):
            raise ValueError(
                f"per_variant_tuning must be a bool, got "
                f"{self.per_variant_tuning!r}")
        if _positive_integer(self.per_variant_tune_seeds) is None:
            raise ValueError(
                f"per_variant_tune_seeds must be a positive integer, got "
                f"{self.per_variant_tune_seeds!r}")
        for name in ("lr_grid", "probe_lr_grid", "calib_tau_grid"):
            raw = getattr(self, name)
            if isinstance(raw, (str, bytes)) or isinstance(raw, Mapping):
                raise ValueError(
                    f"{name} must be a non-empty tuple of finite positive "
                    f"numbers, got {type(raw).__name__}")
            try:
                values = tuple(raw)
            except TypeError as exc:
                raise ValueError(
                    f"{name} must be a non-empty tuple of finite positive "
                    f"numbers, got {raw!r}") from exc
            if not values:
                raise ValueError(
                    f"{name} must be a non-empty tuple of finite positive "
                    f"numbers, got {values!r}")
            checked = []
            for value in values:
                if not isinstance(value, (int, float)) \
                        or isinstance(value, bool):
                    raise ValueError(
                        f"{name} must be a non-empty tuple of finite positive "
                        f"numbers, got {values!r}")
                try:
                    number = float(value)
                except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
                    # A JSON integer can be arbitrarily large; converting it
                    # to float raises OverflowError.  That is a malformed
                    # configuration value, not a reason to leak the
                    # conversion error out of Settings.
                    raise ValueError(
                        f"{name} must be a non-empty tuple of finite positive "
                        f"numbers, got {values!r}") from exc
                if not (math.isfinite(number) and number > 0.0):
                    raise ValueError(
                        f"{name} must be a non-empty tuple of finite positive "
                        f"numbers, got {values!r}")
                if name == "calib_tau_grid":
                    # These values are passed to ``Cfg.calib_tau_init``,
                    # whose softplus-inverse initialisation is
                    # ``log(expm1(x))``.  A finite-but-huge value overflows
                    # that expression, so the per-variant tuning pass would
                    # fail only after the baseline and main grids had run.
                    # Reject it at the configuration boundary instead.
                    try:
                        math.log(math.expm1(number))
                    except (OverflowError, ValueError) as exc:
                        raise ValueError(
                            f"{name} entries must keep "
                            f"log(expm1(calib_tau_init)) finite, got "
                            f"{value!r}") from exc
                checked.append(value)
            # Canonicalise so a caller's list cannot be mutated after
            # validation and silently change the grid used downstream.
            setattr(self, name, tuple(checked))

    @classmethod
    def quick(cls, device: str = "cpu") -> "Settings":
        """Tiny pipeline-check configuration (never for reported numbers)."""
        return cls(
            device=device, d_model=48, n_heads=4, n_layers=4, max_len=64,
            vocab=64, batch=16, n_pairs=6, steps=12, seeds=(0, 1),
            eval_batch=64, eval_batches=2, stats_batch=32, stats_batches=2,
            probe_steps=40, probe_steps_grid=(20, 40, 80),
            probe_batch=16, probe_eval_batch=64,
            probe_eval_batches=2, overfit_steps=100,
        )
