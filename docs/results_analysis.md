# Results analysis and claim levels

This document describes the analysis layers that sit above the raw
contrast table.  The design itself is fixed in `docs/design_notes.md`; this
document does not change the hypotheses, the pre-registered directions, or
the per-experiment FDR rules.

The central rule is:

> The row-level pre-registered decision remains the decision unit.  The
> hypothesis-level and project-level analyses are companions.  They can
> withhold a claim, but they cannot create or rewrite a row-level
> discovery.

## Inputs and outputs

| Path | Written by | Contents |
| --- | --- | --- |
| `e1_contrasts.json` | E1 finalize | E1 contrast rows and family decisions |
| `e2_contrasts.json` | E2 finalize | E2 contrast rows and family decisions |
| `e3_contrasts.json` | E3 finalize | E3 contrast rows and family decisions |
| `project_evidence.json` | `--synthesize-results` or `run_all` | Pooled multiplicity and hypothesis-level synthesis across experiments |
| `artifact_index.json` | run finalization | Hashes of files written by the run |

`run_all()` calls the project synthesis after E3.  An existing results
directory can be synthesized separately:

```bash
python3 -m cage.experiments --only all --quick --out results/quick
python3 -m cage.experiments --synthesize-results --out results/quick --fdr-q 0.05
python3 -m cage.experiments --verify-artifacts --out results/quick
```

The standalone command verifies an existing `artifact_index.json` first,
takes the results lock, hashes the source contrast files, writes
`project_evidence.json`, and then updates the artifact index.  It refuses
to mix a synthesis into a directory whose prior artifacts do not match
their index.

## Row-level decisions

Each contrast row records the effect, interval, p-value, registered
direction or null-prediction bound, SESOI, and the controlled-variable
audit.  A row is read as supporting its prediction only when the relevant
pre-registered rule is met:

- **Directional rows** need an FDR-significant effect in the registered
  direction and an effect at least as large as the registered SESOI.  A
  significant effect in the opposite direction is a refutation, not a null
  result.  A significant effect below the SESOI is statistically
  detectable but practically negligible.
- **Equivalence rows** need the whole confidence interval inside the
  registered bound.
- **Non-inferiority rows** need the confidence-interval lower bound above
  the negative registered bound.

The primary family procedure is Benjamini-Hochberg at the configured
`fdr_q`.  Rows that fail the controlled-variable audit, the competence
prerequisite, or another claim-eligibility gate are reported but cannot
support a mechanism claim.

Several companion procedures annotate the same family:

- Benjamini-Yekutieli for arbitrary dependence.
- Holm-Bonferroni for family-wise error control.
- Simes for a global omnibus test.
- A paired sign-flip family for an exact-test view of the direction.
- False-coverage-rate intervals for the BH-selected set, with a PRDS and a
  BY variant.

These companions do not replace the BH decision.  A verdict that changes
under BY or Holm is reported as multiplicity-model sensitive.  A bootstrap
p-value below the exact sign-flip floor is reported as an overstatement of
the design's resolution, because the exact test cannot attain that
p-value at the current number of seeds.

## Hypothesis-level synthesis

The hypotheses are conjunctions: a hypothesis is confirmed only when all
of its registered predictions hold.  A single surviving row is not enough.

### Required registry coverage

For each hypothesis the analysis derives the set of decision keys from:

- `sesoi` metrics,
- `sesoi_test` null-prediction metrics,
- registered directional keys with sign `+` or `-`.

A contrast row records every pre-registration key it satisfies in
`registry_key` and `registry_keys`.  This matters when the displayed metric
and the registered key differ, for example when a main effect is displayed
on an accuracy metric but registered under a separate concentration key.

If a required key appears in no executed row, the hypothesis is reported
with `registry_complete: false` and `claim_level_status:
"partial_registry"`.  This is deliberate: a conjunction cannot be fully
confirmed when one of its registered outcomes was never printed.  Legacy
results that predate registry-key recording will look partial until they
are rerun.

Metrics listed as `context_metrics` are not required decisions.  H3b's
secondary `gate_depth_ratio` readout is the current example: it carries a
SESOI annotation but the hypothesis registers the residual-norm readout as
the primary decision.

### Conjunction test

For the executed registered rows, the intersection-union p-value is the
maximum component p-value:

```text
p_conjunction = max(p_1, ..., p_k)
```

This is valid without an independence assumption.  It is conservative.
A non-finite component leaves the conjunction undecided rather than
silently dropping that prediction.  Descriptive and robustness rows with
no registered direction or null-prediction test do not enter the
conjunction.

The conjunction p-values are then corrected across the hypotheses in the
same family with BH, BY, and Holm.  The pre-registered row-level family is
unchanged.

### Claim-level status

`claim_level_supported` is true only when all of the following hold:

1. The hypothesis status is `supported`: every executed required row
   supports the prediction.
2. `registry_complete` is true.
3. The conjunction survives the hypothesis-family BH correction.

The status is otherwise one of the explicit non-claim states:

| `claim_level_status` | Meaning |
| --- | --- |
| `supported` | All three conditions above hold |
| `partial_registry` | Executed rows support the hypothesis, but a registered decision metric is missing |
| `not_family_significant` | Rows support the hypothesis, but the conjunction does not survive the hypothesis-family correction |
| `partial` | At least one required row is inconclusive |
| `refuted` | At least one required row is a refutation |
| `inconclusive` | No required row supports or refutes |
| `undirected` | Only descriptive or two-sided rows were registered or executed |
| `no_contrast` | No contrast exists for the hypothesis |

`claim_level_supported` does not turn a row-level companion or an
exploratory analysis into a pre-registered confirmation.

## Project-level synthesis

Within-family FDR control does not compose across experiments.  E1, E2,
and E3 each controlling at `q = 0.05` does not make the union of their
reported discoveries an FDR-0.05 set.  The project synthesis reads the
three finalized contrast files from one results directory and applies BH,
BY, and Holm to the pooled rows.

Every copied row receives:

- `experiment`: source family,
- `project_p_adj`, `project_fdr_reject`,
- `project_by_p_adj`, `project_by_reject`,
- `project_holm_p_adj`, `project_holm_reject`,
- `project_supports`,
- `project_family_support_lost`,
- `project_multiplicity_model_sensitive`.

`project_supports` requires both the source row's
`supports_hypothesis` and a pooled-BH rejection, and respects the existing
claim-eligibility gates.  A row that supports its hypothesis in its own
family but not after pooling is reported as `project_family_support_lost`
and counted as project-level uncertain.  It is not erased or relabelled
as a null result.  A project-supported row that fails BY or Holm is marked
multiplicity-model sensitive.

The payload records the SHA-256, byte count, recorded q, and row count for
every source file.  A missing, malformed, or mixed-q source is visible
instead of being silently pooled.  The source contrast files are not
rewritten.

The pooled hypothesis synthesis repeats the registry-coverage,
conjunction, and hypothesis-family correction across all pooled rows.  Its
output is the `hypothesis_synthesis` block in `project_evidence.json`.
The key summary fields are:

- `n_family_supporting`
- `n_project_supporting`
- `n_family_support_lost_globally`
- `n_project_by_reject`
- `n_project_holm_reject`
- `n_project_multiplicity_model_sensitive`
- `n_hypotheses_registry_incomplete`
- `n_hypotheses_claim_level_supported`

## Reporting order

A defensible results report reads the layers in this order:

1. **Validity gates.**  Check the split audit, the competence prerequisite,
   controlled-variable audits, curve filters, and the exact-test seed
   floor.  A failed gate blocks mechanism interpretation before any
   p-value is read.
2. **Row-level decisions.**  Report the pre-registered directional,
   equivalence, and non-inferiority decisions with their effect sizes,
   intervals, SESOI, and practical-size caveats.
3. **Hypothesis status.**  Report conjunctions as supported, partial,
   refuted, inconclusive, undirected, or no-contrast.  Do not credit a
   hypothesis on one surviving row.
4. **Coverage.**  Report partial-registry status separately from
   statistical support.  A missing registered metric is an omission, not
   evidence in favour of the hypothesis.
5. **Multiplicity sensitivity.**  Report BY and Holm disagreements and
   selection-adjusted interval losses as companions, not as alternative
   discovery claims.
6. **Project-level status.**  Report a within-family claim that does not
   survive the pooled family as project-level uncertain.  Report a
   project-supported hypothesis only when its registered-metric coverage
   is complete.

## Limitations

- The hypothesis conjunction is an intersection-union test and is
  therefore conservative.  Failure to reject does not prove the
  hypothesis is false.
- The conjunction is over executed registered rows, not a per-key
  aggregation.  Repeated rows for the same registered construct each enter
  the maximum.
- Hypothesis-family correction applies to hypotheses with a finite
  conjunction p-value; a hypothesis with no evaluable conjunction is
  reported by its non-claim status instead.
- The pooled BH family treats all eligible rows as one family.  Dependence
  remains, which is why BY and Holm are reported.
- Source families retain their own q levels.  A mixed-q project ledger is
  recorded rather than repaired.
- Source hashes certify the files read by the synthesis; they do not
  re-derive raw-data statistics or verify the upstream run's scientific
  choices.
- `context_metrics` are excluded from required coverage but are still
  printed and annotated.  They are context, not confirmation.
