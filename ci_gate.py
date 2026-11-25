#!/usr/bin/env python3
"""CI gate for the CAGE-Attention continuous loop.

Exit 0 = pass. Any other exit code = the calling workflow must discard the
phase's changes (git reset --hard).

Layers
------
1. Syntax: every .py file parses (always run, no dependencies).
2. Design pins: docs/design_notes.md 【固定】 decisions are enforced by
   source inspection (D1-D6). Only run against files that exist.
3. Smoke: tiny end-to-end forward/backward for every gate variant plus
   the dropout path, LayerScale, gradient checkpointing and a 3-step
   training loop. Skipped (with a warning) when torch is unavailable so
   a network hiccup can never brick the loop.

--syntax-only runs layer 1 only (used by the shutdown trap).
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

FAILURES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"[gate] FAIL: {msg}")


def ok(msg: str) -> None:
    print(f"[gate] ok: {msg}")


# --------------------------------------------------------------------------
# Layer 1: syntax
# --------------------------------------------------------------------------
def check_syntax() -> None:
    n = 0
    for py in sorted(ROOT.rglob("*.py")):
        if any(part in (".git", "__pycache__", ".venv", "venv") for part in py.parts):
            continue
        try:
            ast.parse(py.read_text(encoding="utf-8"))
            n += 1
        except SyntaxError as e:
            fail(f"syntax error in {py.relative_to(ROOT)}: {e}")
    if n:
        ok(f"syntax: {n} python files parse")


# --------------------------------------------------------------------------
# Layer 2: design pins (docs/design_notes.md 【固定】 decisions)
# --------------------------------------------------------------------------
def read(rel: str) -> str:
    p = ROOT / rel
    return p.read_text(encoding="utf-8") if p.exists() else ""


def check_pins() -> None:
    model = read("cage/model.py")
    train = read("cage/train.py")

    if not model or not train:
        fail("cage/model.py or cage/train.py is missing")
        return

    # D1: unified hand-written attention path, no SDPA, no dual-path flag.
    if "scaled_dot_product_attention" in model:
        fail("D1: F.scaled_dot_product_attention must not appear in cage/model.py "
             "(all four gate variants share one hand-written kernel family)")
    if "use_sdpa" in model:
        fail("D1: the use_sdpa dual-path dispatch must not come back")
    for cls in ("class _CalibAttn", "class _NullAttn", "class _SoftmaxInPlace",
                "class _RowEntropy"):
        if cls not in model:
            fail(f"D6: {cls} kernel is missing from cage/model.py")

    # D2: module-level causal-mask cache.
    if "_causal_mask_cache" not in model:
        fail("D2: the module-level _causal_mask_cache is missing")

    # D3: parenthesised block formula, no shift/div precedence bug.
    if not re.search(r"max_rows\s*=\s*\(\s*1\s*<<\s*2\d\s*\)\s*//\s*\(\s*bh\s*\*\s*T\s*\)", model):
        fail("D3: max_rows must be (1 << 2N) // (bh * T) with explicit parentheses")
    if re.search(r"<<\s*\d+\s*//", model):
        fail("D3: unparenthesised '<< N //' precedence bug present")

    # D4: calib_tau softplus-inverse initialisation.
    if "math.log(math.expm1(cfg.calib_tau_init))" not in model:
        fail("D4: calib_tau must be initialised via math.log(math.expm1(cfg.calib_tau_init))")

    # D5: optimizer factory with fallback chain.
    if "def _make_adamw" not in train:
        fail("D5: train.py must build the optimizer via _make_adamw")
    if "foreach=True" not in train or "fused=True" not in train:
        fail("D5: _make_adamw must try fused=True then foreach=True before falling back")

    # D6: MLP GELU stays the simple _GeluLinear (no hand-written _GeluAffine).
    if "class _GeluLinear" not in model:
        fail("D6: the MLP GELU must stay folded into _GeluLinear")
    if "_GeluAffine" in model:
        fail("D6: _GeluAffine (hand-written GELU backward) must not come back")

    # API contract used by scripts/docs.
    for rel, names in (
        ("cage/model.py", ("def build", "class CAGE", "class Attention", "class Cfg")),
        ("cage/train.py", ("def train_model", "def eval_accuracy", "def collect_stats")),
        ("cage/data.py", ("def make_batch", "def length_sweep_lengths")),
    ):
        src = read(rel)
        for name in names:
            if name not in src:
                fail(f"API: {name} disappeared from {rel}")

    ok("design pins D1-D6 + API contract hold")


# --------------------------------------------------------------------------
# Layer 3: smoke test (requires torch; skipped gracefully otherwise)
# --------------------------------------------------------------------------
def check_smoke() -> None:
    try:
        import torch  # noqa: F401
    except Exception as e:  # network / wheel problems must not brick the loop
        print(f"[gate] warn: torch unavailable ({e}); smoke test skipped")
        return

    from cage.data import make_batch
    from cage.model import build
    from cage.train import collect_stats, eval_accuracy, train_model

    torch.manual_seed(0)
    base = dict(d_model=32, n_layers=2, n_heads=4, max_len=64, vocab_size=64)
    idx, tgt, meta = make_batch(3, 6, 64, "cpu")

    def fwd_bwd(cfg, train_mode=False):
        m = build(cfg, compile=False)
        if train_mode:
            m.train()
        logits, loss, _ = m(idx, tgt, sup_pos=meta["query_pos"])
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss for {cfg}")
        loss.backward()
        for p in m.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise RuntimeError(f"non-finite grad for {cfg}")
        m.eval()
        _, _, layer_stats = m(idx, stat_pos=meta["query_pos"])
        for s in layer_stats:
            for k, v in s.items():
                if isinstance(v, torch.Tensor) and v.numel() and not torch.isfinite(v).all():
                    raise RuntimeError(f"non-finite stat {k} for {cfg}")

    cases = [
        ("gate=none", {**base, "gate": "none"}, False),
        ("gate=query", {**base, "gate": "query"}, False),
        ("gate=calib", {**base, "gate": "calib"}, False),
        ("gate=null", {**base, "gate": "null"}, False),
        ("norm=seednorm", {**base, "gate": "calib", "norm": "seednorm"}, False),
        ("dropout path (calib)", {**base, "gate": "calib", "dropout": 0.1}, True),
        ("dropout path (none)", {**base, "gate": "none", "dropout": 0.1}, True),
        ("dropout path (null)", {**base, "gate": "null", "dropout": 0.1}, True),
        ("layerscale+ckpt", {**base, "gate": "query", "layer_scale": 1e-2,
                             "use_gradient_checkpointing": True}, False),
    ]
    for name, cfg, train_mode in cases:
        try:
            fwd_bwd(cfg, train_mode)
        except Exception as e:
            fail(f"smoke [{name}]: {e}")

    # end-to-end: 3 training steps, eval, stats collection
    try:
        m = build({**base, "gate": "calib", "d_model": 48}, compile=False)
        hist = train_model(m, steps=3, batch=4, n_pairs=6, vocab=64, log_every=1)
        if len(hist) != 3:
            raise RuntimeError("train_model did not run 3 steps")
        acc = eval_accuracy(m, 6, 64, "cpu", batch=4, batches=1)
        if not 0.0 <= acc <= 1.0:
            raise RuntimeError(f"eval_accuracy out of range: {acc}")
        st = collect_stats(m, 6, 64, "cpu", batch=4, batches=1)
        if not torch.isfinite(st["entropy"]).all():
            raise RuntimeError("collect_stats produced non-finite entropy")
    except Exception as e:
        fail(f"smoke [end-to-end]: {e}")

    if not any(f.startswith("smoke") for f in FAILURES):
        ok("smoke: 4 gate variants + seednorm + dropout path + layerscale + "
           "checkpointing + 3-step training all finite")


def main() -> int:
    syntax_only = "--syntax-only" in sys.argv
    check_syntax()
    if not syntax_only:
        check_pins()
        check_smoke()
    if FAILURES:
        print(f"\n[gate] REJECTED — {len(FAILURES)} failure(s)")
        return 1
    print("\n[gate] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
