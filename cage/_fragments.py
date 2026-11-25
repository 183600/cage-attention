"""Execute fragment source files into a wrapper module's namespace.

The data, training, metrics and experiment wrappers are assembled from
several focused source files.  Running those fragments in the wrapper's own
namespace keeps every helper's public name and module identity (for example
``cage.data.make_batch`` rather than ``cage._data_task.make_batch``), which
callers, tests and archived provenance records rely on.
"""

from __future__ import annotations

import __future__
from pathlib import Path
# NOTE: Module-level seed setting removed to prevent silent RNG state
# reset on import. Use the centralized seed function at the experiment
# entry point instead (e.g., repro.set_all_seeds / seed_everything).


def load_fragments(namespace: dict, names: tuple[str, ...]) -> None:
    """Compile and execute ``names`` into ``namespace`` in order.

    ``namespace`` is normally the wrapper module's ``globals()``.  Fragments
    are read as UTF-8 and compiled with future annotations enabled and
    ``dont_inherit=True``, matching the wrapper's own parsing mode without
    letting a fragment change the wrapper's future flags.
    """
    base = Path(namespace["__file__"]).resolve().parent
    flags = __future__.annotations.compiler_flag
    for name in names:
        path = base / name
        source = path.read_text(encoding="utf-8")
        try:
            code = compile(source, str(path), "exec",
                           flags=flags, dont_inherit=True)
        except SyntaxError as exc:
            raise RuntimeError(
                f"Failed to compile fragment {name!r} ({path}): {exc}"
            ) from exc
        try:
            exec(code, namespace)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to execute fragment {name!r} ({path}): {exc}"
            ) from exc
