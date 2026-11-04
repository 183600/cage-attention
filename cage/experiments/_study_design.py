# cage/experiments/_study_design.py - the study-level design pre-flight.
# Executed into the cage.experiments namespace by __init__.py;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541
"""Preparation-time certificates for the *study*, not for one family.

The per-experiment machinery already certifies a single family before it
trains: the sensitivity plan (does this family resolve its SESOI?), the
single-factor certificate (does every declared contrast vary exactly one
resolved model factor?), the blocked schedule (can machine drift be
aliased with an arm?) and the baseline gates (is the task solvable at all?).

The study reads across three families, and two questions only exist at that
level:

* **Is every confirmatory leg reachable?**  A primary contrast names the
  grid cells it contrasts.  If one of those cells is not in the family's
  grid, the leg can never be measured: the study would spend its compute
  and then report the claim as "not adjudicable" for a reason that was
  visible in the source before the first run.  The reachability check
  resolves each leg's arms against the *authoritative* cell plan of its
  family (``_e1_cell_plan``, ``_E2_VARIANTS``, the E3 grid), so a renamed
  or dropped cell is caught at preparation time.
* **What will the study cost?**  The compute request, the milestone plan
  and the decision to add seeds are all made against a run count that was
  written down by hand.  The cost plan derives the run inventory and the
  analytic training FLOPs from the settings and the same cell plans, so
  the number in the proposal and the number the code executes come from
  one source.

Both live in one artifact (``study_design_preflight.json``) because they
answer one question: *before the first training run, can this study
deliver what it claims, and what will it cost to try?*  Neither changes a
pre-registered decision rule; they only refuse a design that cannot be
executed as registered.
"""

# The number of grid cells each family trains, and the gate/normalisation
# each cell uses.  The FLOP-relevant part of a cell is its gate (the
# calibration terms and the placebo share the ``calib`` architecture) and
# its normalisation (the dynamic normalisation adds two scale matmuls per
# block); the LayerScale level and the calibration-term switches change the
# parameters, not the matmul count.
_E3_GRID_GATES = ("query", "calib")


def _family_cell_labels(experiment):
    """The grid-cell labels one family actually trains.

    The cell plans are the single source of truth for what a grid contains:
    the runners schedule exactly these labels, so a primary contrast that
    names a label outside them cannot be measured.
    """
    if experiment == "e1":
        return tuple(_e1_cell_plan()["cells"])
    if experiment == "e2":
        return tuple(label for label, *_ in _E2_VARIANTS)
    if experiment == "e3":
        return tuple(_e3_single_factor_design()["cells"])
    return ()


def _primary_contrast_reachability(legs=None):
    """Whether every registered primary contrast has the cells it needs.

    A leg is reachable when all the arms it names are cells of its own
    family.  The check is deliberately about *existence*, not about
    significance: an unreachable leg is a registration defect, and the
    remedy is to fix the registration before the grid runs, never to drop
    the leg after its result is known.
    """
    legs = _analysis_plan_legs(legs)
    rows = []
    for leg in legs:
        experiment = leg.get("experiment")
        arms = list(leg.get("arms") or ())
        cells = set(_family_cell_labels(experiment))
        missing = [arm for arm in arms if arm not in cells]
        rows.append({
            "id": leg.get("id"),
            "experiment": experiment,
            "kind": leg.get("kind"),
            "arms": arms,
            "n_arms": len(arms),
            "n_family_cells": len(cells),
            "missing_arms": missing,
            "reachable": bool(arms) and not missing,
        })
    unreachable = [row["id"] for row in rows if not row["reachable"]]
    return {
        "ok": bool(rows) and not unreachable,
        "n_legs": len(rows),
        "legs": rows,
        "unreachable_legs": unreachable,
        "note": ("a leg is reachable when every arm it names is a cell of "
                 "its own family's grid; reachability says nothing about "
                 "the size or sign of the effect"),
    }


def _study_cost_plan(settings):
    """The compute the registered study is going to spend, before it spends it.

    The inventory is derived from the settings and the cell plans, so it
    moves when the design moves.  Three kinds of number appear:

    * ``exact`` - the run count is fixed by the design (the grids, the
      learning-rate sensitivity sweep, the capacity-matched references and
      the validation-selection runs of the per-variant account);
    * ``range`` - the run count depends on which candidate recipe wins the
      validation selection, so the plan reports the interval it can fall
      in (the per-variant final runs are reused when the frozen reference
      already wins);
    * the FLOP estimate - analytic forward matmul FLOPs per sequence,
      multiplied by the training steps and the batch size and by the usual
      3x for forward plus backward.  It excludes softmax, normalisation,
      activation and elementwise work, exactly like
      ``cage.efficiency.analytic_matmul_flops``.

    The baselines ladder (the tuned probe grid and the depth ladder) and
    the evaluation-only sweeps are *not* included: they are reported by
    ``baselines_verdict.json`` and by the length-efficiency companion, and
    mixing them into one number would make the total unverifiable.
    """
    n_seeds = len(settings.seeds)
    tune_seeds = max(1, min(int(settings.per_variant_tune_seeds), n_seeds))
    candidate_lrs = len({float(v) for v in settings.lr_grid}) + 1
    candidate_taus = len({float(v) for v in settings.calib_tau_grid}) + 1
    # The reference point is a candidate too, but it may coincide with a
    # grid entry, so the upper bound is used as the conservative count.
    selection_runs = len(GATES) * candidate_lrs * tune_seeds
    tau_selection_max = candidate_taus * tune_seeds
    final_runs_max = len(GATES) * n_seeds

    def _cfg(gate, norm):
        return Cfg(vocab_size=int(settings.vocab),
                   d_model=int(settings.d_model),
                   n_heads=int(settings.n_heads),
                   n_layers=int(settings.n_layers),
                   max_len=int(settings.max_len),
                   gate=gate, norm=norm)

    # (cells, gate, norm) for every cell each family trains.
    e1_groups = [(1, "none", "pre"), (1, "query", "pre"),
                 (len(_E1_ABLATION_LABELS) + 1, "calib", "pre"),
                 (1, "null", "pre")]
    e2_groups = []
    for label, gate, _extra, _overrides in _E2_VARIANTS:
        for index, (count, g, _n) in enumerate(e2_groups):
            if g == gate:
                e2_groups[index] = (count + 1, g, "pre")
                break
        else:
            e2_groups.append((1, gate, "pre"))
    e3_groups = [(len(_E3_LAYER_SCALES), gate, norm)
                 for gate in _E3_GRID_GATES for norm in NORMS]

    seq_len = 2 * int(settings.n_pairs) + 2
    families = {
        "e1": {
            "cells": len(_e1_cell_plan()["cells"]),
            "groups": e1_groups,
            "exact_runs": [
                {"name": "primary grid", "runs": len(_e1_cell_plan()["cells"])
                 * n_seeds},
                {"name": "learning-rate sensitivity", "runs":
                 len(_E1_LR_SENSITIVITY_FACTORS) * len(GATES) * n_seeds},
                {"name": "per-variant tuning (validation selection)",
                 "runs": selection_runs},
            ],
            "range_runs": [
                {"name": "per-variant tuning (calib temperature "
                         "selection)", "runs_min": 0,
                 "runs_max": tau_selection_max},
                {"name": "per-variant tuning (final re-training)",
                 "runs_min": 0, "runs_max": final_runs_max},
            ],
        },
        "e2": {
            "cells": len(_E2_VARIANTS),
            "groups": e2_groups,
            "exact_runs": [
                {"name": "primary grid", "runs": len(_E2_VARIANTS) * n_seeds},
                {"name": "per-variant tuning (validation selection)",
                 "runs": selection_runs},
            ],
            "range_runs": [
                {"name": "per-variant tuning (calib temperature "
                         "selection)", "runs_min": 0,
                 "runs_max": tau_selection_max},
                {"name": "per-variant tuning (final re-training)",
                 "runs_min": 0, "runs_max": final_runs_max},
            ],
        },
        "e3": {
            "cells": len(_E3_GRID_GATES) * len(NORMS)
            * len(_E3_LAYER_SCALES),
            "groups": e3_groups,
            "exact_runs": [
                {"name": "primary grid", "runs":
                 len(_E3_GRID_GATES) * len(NORMS)
                 * len(_E3_LAYER_SCALES) * n_seeds},
                {"name": "capacity-matched pre-LN references", "runs":
                 len(_E3_GRID_GATES) * len(_E3_LAYER_SCALES) * n_seeds},
            ],
            "range_runs": [],
        },
    }

    total_exact = 0
    total_min = 0
    total_max = 0
    total_flops = 0.0
    for tag, plan in families.items():
        exact = sum(int(item["runs"]) for item in plan["exact_runs"])
        low = exact + sum(int(item["runs_min"]) for item in plan["range_runs"])
        high = exact + sum(int(item["runs_max"])
                           for item in plan["range_runs"])
        flops = 0.0
        per_cell = []
        for count, gate, norm in plan["groups"]:
            try:
                forward = analytic_matmul_flops(_cfg(gate, norm), seq_len,
                                                batch=1, head_tokens=1)
            except Exception as exc:  # noqa: BLE001 - report, do not abort
                per_cell.append({"gate": gate, "norm": norm, "cells": count,
                                 "error": f"{type(exc).__name__}: {exc}"})
                continue
            training = forward * int(settings.steps) * int(settings.batch) * 3
            # One run per cell *per training seed*: the cost of a group is
            # its number of cells times the seed count, not its number of
            # cells alone.
            flops += training * count * n_seeds
            per_cell.append({
                "gate": gate, "norm": norm, "cells": count,
                "runs": count * n_seeds,
                "forward_flops_per_sequence": int(forward),
                "training_flops_per_run": int(training),
                "training_tflops": training * count * n_seeds / 1e12,
            })
        plan["exact_runs_total"] = exact
        plan["runs_min"] = low
        plan["runs_max"] = high
        plan["training_tflops"] = flops / 1e12
        plan["cells_detail"] = per_cell
        total_exact += exact
        total_min += low
        total_max += high
        total_flops += flops
    return {
        "planned_seeds": list(settings.seeds),
        "n_seeds": n_seeds,
        "steps": int(settings.steps),
        "batch": int(settings.batch),
        "train_seq_len": seq_len,
        "families": families,
        "total_exact_runs": total_exact,
        "total_runs_min": total_min,
        "total_runs_max": total_max,
        "total_training_tflops": total_flops / 1e12,
        "scope": ("the three mechanism grids and their directly attached "
                  "companions; the baselines ladder (tuned probe grid, depth "
                  "ladder) and the evaluation-only length sweep are "
                  "excluded and are reported by their own artifacts"),
        "flops_scope": ("analytic forward matmul FLOPs (2 x MACs) times the "
                        "training steps and batch size times 3 for forward "
                        "plus backward; softmax, normalisation, activation "
                        "and elementwise work are excluded"),
        "note": ("the run inventory moves with the settings and the cell "
                 "plans, so a proposal that quotes a run count should quote "
                 "this artifact rather than a hand-written number"),
    }


def _study_design_preflight(settings, out_dir, power=None, verdict=None):
    """Certify the study-level design before any family trains.

    The certificate answers three preparation questions at once:

    * does the pre-registered analysis plan agree with the hypothesis
      registry (``_analysis_plan_audit``)?
    * can every registered primary contrast actually be measured by its
      family (``_primary_contrast_reachability``)?
    * can the study deliver its claims at the planned seed count
      (``_primary_contrast_power``), and what will it cost
      (``_study_cost_plan``)?

    A registry disagreement or an unreachable leg stops the run: both are
    defects of the registration itself, and fixing them after the data
    exist would be a post-hoc redesign.  The power and cost blocks are
    reported, not enforced - an underpowered or expensive design is a
    decision the researcher has to make, not an error the code should hide.
    """
    audit = _analysis_plan_audit()
    reach = _primary_contrast_reachability()
    if power is None:
        power = _primary_contrast_power(settings, verdict)
    try:
        cost = _study_cost_plan(settings)
    except Exception as exc:  # noqa: BLE001 - a cost estimate never blocks
        cost = {"available": False,
                "reason": f"{type(exc).__name__}: {exc}"}
    issues = list(audit["issues"])
    for row in reach["legs"]:
        if row["reachable"]:
            continue
        missing = ", ".join(row["missing_arms"]) or "no arms"
        issues.append(
            f"{row['id']}: the arms {missing} are not cells of the "
            f"{row['experiment']} grid, so this primary contrast can never "
            f"be measured")
    if issues:
        raise DesignPlanError(
            "the study-level design is not executable as registered: "
            + "; ".join(issues) + ".  Fix the registration before the grid "
            "runs; changing the confirmatory contrasts after the data "
            "exist would be a post-hoc redesign.")
    payload = {
        "schema_version": 1,
        "description": ("Preparation-time certificate of the study-level "
                        "design: the confirmatory contrasts are consistent "
                        "with the hypothesis registry, every one of them is "
                        "reachable in its family's grid, the pooled "
                        "primary family is budgeted against the planned "
                        "seed count, and the compute the study will spend "
                        "is derived from the settings and the cell plans."),
        "experiments": list(_STUDY_EXPERIMENTS),
        "registry_consistent": True,
        "registry_issues": [],
        "reachable": bool(reach["ok"]),
        "unreachable_legs": list(reach["unreachable_legs"]),
        "primary_contrasts": reach["legs"],
        "project_family": power,
        "compute": cost,
        "_provenance": _stamp(settings),
    }
    _write_json(pathlib.Path(out_dir) / "study_design_preflight.json",
                payload)
    print(f"[study] design pre-flight: {len(reach['legs'])} pre-registered "
          f"primary contrast(s) over {len(payload['experiments'])} "
          f"families are consistent with the hypothesis registry and "
          f"reachable in their grids")
    if isinstance(cost, Mapping) and cost.get("available") is not False:
        print(f"[study] compute: {cost.get('total_exact_runs')} fixed "
              f"training run(s), {cost.get('total_runs_min')}-"
              f"{cost.get('total_runs_max')} including the recipe "
              f"selection bounds; {float(cost.get('total_training_tflops', 0.0)):.1f}"
              f" TFLOP of training matmuls at n={cost.get('n_seeds')}")
    unpiloted = (power or {}).get("legs_without_pilot") or []
    if unpiloted:
        print(f"[study] NOTE: {len(unpiloted)} primary contrast(s) "
              f"({', '.join(unpiloted)}) have no empirical pilot variance, "
              f"so their study-level power is not certified")
    return payload
