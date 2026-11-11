"""Run-level reproducibility: one place that seeds every RNG a run draws from.

What is seeded
--------------
* Python ``random`` (stdlib helpers and any third-party code reading it);
* NumPy ``numpy.random`` (only when NumPy is installed - PyTorch works
  without it, so the helper must not hard-depend on it);
* PyTorch CPU and CUDA: ``torch.manual_seed`` seeds the default generator
  and ``torch.cuda.manual_seed_all`` makes the all-device coverage explicit.

Data order is *not* part of the global RNG: ``cage.data.make_batch`` never
reads global state - every split/stream passes its own ``torch.Generator``
(created here via :func:`stream_generator`).  That generator is the direct
equivalent of a ``DataLoader(generator=...)``: the stream owns its seed, so
re-seeding the global RNG elsewhere cannot shuffle the data underneath a
run, and a future ``worker_init_fn`` would derive per-worker seeds from the
same value.

Seeds are canonicalised to the 32-bit key every stream generator uses
(:func:`canonical_seed`), so ``s`` and ``s + 2**32`` are the same run for
the global RNGs as well as for the data streams.  A bool or fractional
seed is rejected instead of being silently truncated.

Scope of the guarantee
----------------------
Running the same configuration twice on the same machine with the same
PyTorch build is run-to-run stable.  Exact bit-for-bit reproduction across
different hardware/instruction sets is *not* promised (reduction order in
matmuls/atomics differs); ``deterministic=True`` additionally asks PyTorch
for deterministic kernels where they exist.

What is declared
----------------
Seeding fixes *which* numbers a run draws; it says nothing about *which
library* produced them.  The second half of run-level reproducibility is
therefore a dependency conformance check: the versions the project freezes
(``requirements-ci.txt``, the ``pyproject.toml`` runtime requirement) must
match the versions that are actually installed, otherwise a result is
produced under a recipe nobody declared.  :func:`check_dependency_conformance`
answers that question for the running interpreter, and
:func:`dependency_conformance_from_record` answers it for an archived
environment record - so an old result directory can be audited for "was this
run executed with the dependency set the project declares?" without rerunning
anything.
"""

from __future__ import annotations

import operator
import os
import pathlib
import random
import re
from importlib import metadata as importlib_metadata

import torch

# Last PYTHONHASHSEED value pinned by :func:`seed_everything` (None when the
# variable was absent or set externally).  Lets repeated calls in one process
# refresh their own pin without ever overwriting an operator-set value.
_HASHSEED_PINNED: str | None = None


# PyTorch's documented defaults for reduced-precision matrix products.
# Restored on every non-deterministic call so a previous deterministic run
# cannot leak its precision mode into a later one.
_TF32_DEFAULTS = {"matmul": False, "cudnn": True}


def _tf32_modules():
    """The reduced-precision knobs PyTorch exposes (may be absent)."""
    modules = {}
    cuda_backend = getattr(torch.backends, "cuda", None)
    matmul = getattr(cuda_backend, "matmul", None)
    if matmul is not None and hasattr(matmul, "allow_tf32"):
        modules["matmul"] = matmul
    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is not None and hasattr(cudnn, "allow_tf32"):
        modules["cudnn"] = cudnn
    return modules


def _tf32_state() -> dict:
    return {name: bool(module.allow_tf32)
            for name, module in _tf32_modules().items()}


def configure_determinism(deterministic: bool) -> dict:
    """Apply the *whole* determinism switch, including turning it off.

    ``seed_everything`` used to only ever enable deterministic kernels, so
    a process that ran one deterministic job and then a non-deterministic
    one kept the global flag on while its records claimed the opposite.
    This helper makes the flag follow the requested value on every call:
    off restores PyTorch's default (non-deterministic kernels), while
    cuDNN benchmark autotuning is always disabled because timing-based
    algorithm selection is a run-to-run variance source even when kernels
    themselves are otherwise deterministic.

    Reduced-precision TF32 matrix products are disabled under
    ``deterministic=True`` and restored to the PyTorch defaults otherwise.
    TF32 keeps only 10 mantissa bits, so its use changes the numerical
    trajectory; leaving it enabled would make a ``deterministic`` run
    fast-but-reduced-precision while its record claimed a reproducible
    recipe. The resolved values are returned and stamped into the run
    environment.

    ``CUBLAS_WORKSPACE_CONFIG`` is only ever *added* (via ``setdefault``)
    before CUDA initialises; it cannot be unset safely afterwards, so the
    resolved value is reported and recorded rather than pretending the
    environment was restored.
    """
    if not isinstance(deterministic, bool):
        raise ValueError(
            f"deterministic must be a bool, got {deterministic!r}: a truthy "
            f"string such as 'False' would silently request deterministic "
            f"kernels and TF32 suppression")
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except (AttributeError, TypeError):
        import logging as _logging; _logging.warning("Suppressed exception in repro.py:122")
    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is not None:
        cudnn.deterministic = deterministic
        cudnn.benchmark = False
    wanted_tf32 = (not deterministic)
    for name, module in _tf32_modules().items():
        module.allow_tf32 = bool(_TF32_DEFAULTS[name]) if wanted_tf32 else False
    return {
        "deterministic": deterministic,
        "deterministic_algorithms": bool(getattr(
            torch, "are_deterministic_algorithms_enabled", lambda: False)()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn": None if cudnn is None else {
            "deterministic": bool(cudnn.deterministic),
            "benchmark": bool(cudnn.benchmark),
        },
        "tf32": _tf32_state(),
        "torch_num_threads": int(torch.get_num_threads()),
    }


def canonical_seed(seed) -> tuple[int, int]:
    """Return ``(requested_seed, canonical_key)`` for a run seed.

    Every RNG a run draws from must consume the *same* integer.  PyTorch's
    data-stream generators keep only the low 32 bits of their seed
    (:func:`cage.data.stream_seed_key`), and NumPy's legacy
    ``numpy.random.seed`` rejects values outside ``[0, 2**32)``, so feeding
    the unrestricted Python integer to some RNGs and the 32-bit key to
    others makes aliased seeds (``s`` and ``s + 2**32``) draw identical data
    but different dropout masks.  Canonicalising once here means an alias
    is the same run, and a negative / oversized seed is accepted
    consistently instead of failing only when NumPy happens to be
    installed.

    ``seed`` must be a true integer (``bool`` and floats are rejected): a
    silent truncation would run a different stream than the one the caller
    named.
    """
    if isinstance(seed, bool):
        raise ValueError(f"run seed must be an integer, got {seed!r}")
    try:
        requested = operator.index(seed)
    except TypeError as exc:
        raise ValueError(
            f"run seed must be an integer, got {seed!r}") from exc
    return int(requested), int(requested) & 0xFFFFFFFF


def seed_everything(seed: int, deterministic: bool = False) -> dict:
    """Seed Python, NumPy (if present) and PyTorch CPU + all CUDA devices.

    Returns a small dict describing what was seeded, so a run record can
    document the seeding state instead of assuming it.  ``deterministic``
    is opt-in: it calls
    ``torch.use_deterministic_algorithms(True, warn_only=True)`` (an op
    without a deterministic kernel warns instead of aborting a long grid),
    pins cuDNN to deterministic kernels with benchmark autotuning off, and
    sets the cuBLAS workspace config *before* any CUDA call - cuBLAS reads
    that variable when it initialises, so setting it afterwards is a no-op.
    Passing ``False`` explicitly restores the non-deterministic default, so
    a previous deterministic run in the same process cannot leak its
    global kernel flags into a later one.

    ``PYTHONHASHSEED`` is exported for child processes and recorded.  The
    hash seed of the *running* interpreter is fixed at launch, so a run
    that depends on set/dict iteration order must start with it exported.
    A value already present in the environment is left untouched; a value
    this helper pinned itself is refreshed on every call, so the recorded
    and exported value always describes the seed of the current call
    instead of the first one.

    The seed is canonicalised with :func:`canonical_seed` before any RNG
    is touched: ``seed`` and ``seed + 2**32`` are the same run, matching
    the per-stream data generators.  The returned record carries both the
    caller's integer (``requested_seed``) and the key actually used
    (``seed`` / ``seed_key``), so an aliased invocation is visible rather
    than silently relabelled.
    """
    global _HASHSEED_PINNED
    requested, seed = canonical_seed(seed)
    determinism_info = configure_determinism(deterministic)
    random.seed(seed)
    info = {
        "seed": seed,
        "seed_key": seed,
        "requested_seed": requested,
        "python_random": True,
        "numpy": False,
        "torch": True,
        "cuda_devices": 0,
    }
    current = os.environ.get("PYTHONHASHSEED")
    if current is None or current == _HASHSEED_PINNED:
        _HASHSEED_PINNED = str(seed)
        os.environ["PYTHONHASHSEED"] = _HASHSEED_PINNED
    info["pythonhashseed"] = os.environ.get("PYTHONHASHSEED")
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)
        info["numpy"] = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        info["cuda_devices"] = int(torch.cuda.device_count())
    info.update(determinism_info)
    return info


def stream_generator(seed: int, device="cpu") -> torch.Generator:
    """Deterministic per-stream data generator (DataLoader equivalent).

    Every synthetic split/training seed owns one of these; ``make_batch``
    draws exclusively through it, so data order is fixed independently of
    the global RNG state.  The seed is canonicalised with
    :func:`cage.data.stream_seed_key` (the low 32 bits torch's CPU
    generators consume), so aliased seeds cannot fingerprint as a different
    dataset than the stream they actually draw.
    """
    from .data import stream_seed_key

    return torch.Generator(device=device).manual_seed(stream_seed_key(seed))


# ---------------------------------------------------------------------------
# Declared-vs-installed dependency conformance
# ---------------------------------------------------------------------------
#
# The frozen dependency set (``requirements-ci.txt``) and the declared
# runtime requirement (``pyproject.toml``) describe the recipe a result is
# supposed to belong to.  The interpreter that actually runs the experiment
# can disagree with them - a stale virtualenv, an editable install that
# re-resolved a dependency, a wheel index that no longer carries the pinned
# version.  Such a run is not reproducible even when every seed matches,
# because the numbers come from different code.  The helpers below make that
# disagreement visible instead of leaving it to be noticed months later.

# Files whose content declares what the project depends on.
_DEPENDENCY_DECLARATION_FILES = ("requirements-ci.txt", "requirements.txt",
                                  "pyproject.toml")

# A requirement line: name, optional extras, optional version specifier.
_REQUIREMENT_LINE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?:\[[^\]]*\])?"
    r"(?P<spec>[<>=!~^].*)?$")

# A single, supported version specifier.  Anything else is reported as
# "unchecked" instead of being silently treated as satisfied.
_SINGLE_SPECIFIER = re.compile(
    r"^(?P<op>===|==|>=|<=|!=|>|<)\s*(?P<version>[^\s,]+)$")

# The release part of a version, without a local ("+cpu") or a pre/post/dev
# suffix.  ``None`` means the version cannot be ordered exactly.
_PUBLIC_RELEASE = re.compile(r"^(?P<release>\d+(?:\.\d+)*)$")


def _normalise_distribution_name(name: str) -> str:
    """Canonical spelling of a distribution name.

    ``Foo.Bar``, ``foo_bar`` and ``foo-bar`` are the same distribution; the
    environment record already normalises the installed inventory this way,
    and the declared names have to be normalised the same way for the two
    sides to be comparable.
    """
    return name.strip().lower().replace("_", "-").replace(".", "-")


def _public_release(version: str) -> tuple[int, ...] | None:
    """Release segments of a version, or ``None`` when it is not exact.

    A local version label (``2.14.1+cpu``) is ignored: PEP 440 says
    ``==2.14.1`` is satisfied by ``2.14.1+cpu``.  A pre/post/dev suffix
    (``2.14.1rc1``) is *not* ignored - it orders differently from the final
    release - so such a version is reported as unevaluable instead of being
    compared by its numeric prefix.
    """
    public = str(version).split("+", 1)[0].strip()
    match = _PUBLIC_RELEASE.match(public)
    if not match:
        return None
    return tuple(int(part) for part in match.group("release").split("."))


def _compare_releases(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    """Compare two release tuples, padding the shorter one with zeros."""
    length = max(len(left), len(right))
    left = left + (0,) * (length - len(left))
    right = right + (0,) * (length - len(right))
    return (left > right) - (left < right)


def parse_requirement_specifiers(text: str) -> list[tuple[str, str]]:
    """Parse ``(name, specifier)`` pairs out of a requirements-style body.

    Comments, blank lines and pip options (``-r other.txt``, ``--index-url``)
    are skipped.  A requirement without a version specifier is returned with
    an empty specifier, which means "any version satisfies this".
    """
    entries: list[tuple[str, str]] = []
    for raw in str(text).splitlines():
        line = re.sub(r"(^|\s)#.*$", "", raw).strip()
        if not line or line.startswith("-"):
            continue
        match = _REQUIREMENT_LINE.match(line)
        if not match:
            continue
        name = _normalise_distribution_name(match.group("name"))
        specifier = (match.group("spec") or "").strip()
        entries.append((name, specifier))
    return entries


def _pyproject_dependencies(text: str) -> list[tuple[str, str]]:
    """Runtime requirements declared in a ``pyproject.toml`` body.

    ``tomllib`` is used when the interpreter provides it (Python 3.11+); the
    fallback reads the ``[project] dependencies`` array directly so the check
    also works on Python 3.10, which the package still supports.
    """
    try:
        import tomllib
    except ModuleNotFoundError:
        tomllib = None
    if tomllib is not None:
        try:
            data = tomllib.loads(text)
        except Exception:  # noqa: BLE001 - malformed metadata is not fatal
            data = {}
        declared = data.get("project", {}).get("dependencies", [])
        if isinstance(declared, list):
            entries: list[tuple[str, str]] = []
            for item in declared:
                entries.extend(parse_requirement_specifiers(str(item)))
            return entries
    match = re.search(r"^dependencies\s*=\s*\[(?P<body>.*?)\]", text,
                      re.S | re.M)
    if not match:
        return []
    body = " ".join(re.findall(r"[\"']([^\"']+)[\"']", match.group("body")))
    return parse_requirement_specifiers(body)


def declared_requirements(root=None) -> dict[str, list[tuple[str, str]]]:
    """The dependency declarations of this checkout.

    Returns ``{file name: [(distribution, specifier), ...]}`` for every
    declaration file that exists.  The file name (not its absolute path) is
    the key, so the verdict does not depend on where the checkout lives.
    """
    base = pathlib.Path(__file__).resolve().parent.parent if root is None \
        else pathlib.Path(root)
    declared: dict[str, list[tuple[str, str]]] = {}
    for name in _DEPENDENCY_DECLARATION_FILES:
        path = base / name
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if name.endswith(".toml"):
            declared[name] = _pyproject_dependencies(text)
        else:
            declared[name] = parse_requirement_specifiers(text)
    return declared


def _higher_version(current: str, candidate: str) -> str:
    """Deterministically pick the higher of two version strings.

    ``max`` on the raw strings orders lexicographically, so for a
    distribution that appears twice in the metadata scan (e.g. a user-site
    copy next to a system copy) ``"2.9.0"`` would beat ``"2.14.1"`` and the
    conformance verdict could be decided by scan order.  Compare the PEP 440
    public releases when both parse; fall back to the raw string order
    (still deterministic) when either version carries a suffix this module
    cannot order exactly.
    """
    left = _public_release(current)
    right = _public_release(candidate)
    if left is not None and right is not None:
        order = _compare_releases(left, right)
        if order != 0:
            return current if order > 0 else candidate
    return max(current, candidate)


def installed_distribution_versions() -> dict[str, str]:
    """Installed distributions and their versions, normalised and sorted.

    Package metadata is untrusted input: a broken distribution must not
    make the conformance check fail with a traceback.  When the same
    distribution is reported more than once, the semantically highest
    version is kept so the result does not depend on the metadata scan
    order.
    """
    versions: dict[str, str] = {}
    try:
        distributions = list(importlib_metadata.distributions())
    except Exception:  # noqa: BLE001 - metadata is a best-effort probe
        return {}
    for dist in distributions:
        try:
            name = dist.metadata["Name"]
            version = dist.version
        except Exception:  # noqa: BLE001 - skip malformed distributions
            continue
        if not isinstance(name, str) or not isinstance(version, str):
            continue
        key = _normalise_distribution_name(name)
        if not key or not version:
            continue
        previous = versions.get(key)
        versions[key] = (version if previous is None
                         else _higher_version(previous, version))
    return dict(sorted(versions.items()))


def evaluate_requirement(name: str, specifier: str,
                         installed_version: str | None) -> dict:
    """Decide whether one installed version satisfies one declaration.

    The verdict is one of ``satisfied`` / ``unsatisfied`` / ``missing`` /
    ``unchecked``.  ``unchecked`` is used for specifiers this module cannot
    evaluate exactly (compound ranges, compatible releases, pre-release
    versions): the requirement is reported, never counted as satisfied.
    """
    entry = {
        "name": name,
        "required": specifier or "(any)",
        "installed": installed_version,
        "status": "unchecked",
        "detail": "",
    }
    if installed_version is None:
        entry["status"] = "missing"
        entry["detail"] = "distribution is not installed"
        return entry
    if not specifier:
        entry["status"] = "satisfied"
        entry["detail"] = "no version constraint"
        return entry
    match = _SINGLE_SPECIFIER.match(specifier)
    if not match:
        entry["detail"] = (f"unsupported specifier {specifier!r}; "
                           "reported, not certified")
        return entry
    operator_text = match.group("op")
    required = match.group("version")
    if operator_text == "===":
        satisfied = installed_version == required
        entry["detail"] = ("exact match" if satisfied
                           else f"expected exactly {required}")
        entry["status"] = "satisfied" if satisfied else "unsatisfied"
        return entry
    left = _public_release(installed_version)
    right = _public_release(required)
    if left is None or right is None:
        entry["detail"] = (f"cannot compare {installed_version!r} with "
                           f"{required!r} exactly")
        return entry
    order = _compare_releases(left, right)
    if operator_text == "==":
        satisfied = order == 0
    elif operator_text == "!=":
        satisfied = order != 0
    elif operator_text == ">=":
        satisfied = order >= 0
    elif operator_text == "<=":
        satisfied = order <= 0
    elif operator_text == ">":
        satisfied = order > 0
    else:  # "<"
        satisfied = order < 0
    entry["status"] = "satisfied" if satisfied else "unsatisfied"
    entry["detail"] = (f"installed {installed_version} vs required "
                       f"{specifier}")
    return entry


def _conformance_report(declared: dict[str, list[tuple[str, str]]],
                        installed: dict[str, str]) -> dict:
    """Aggregate per-requirement verdicts into one conformance report."""
    entries: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for source, requirements in declared.items():
        for name, specifier in requirements:
            key = (name, specifier)
            if key in seen:
                continue
            seen.add(key)
            entry = evaluate_requirement(name, specifier,
                                         installed.get(name))
            entry["source"] = source
            entries.append(entry)
    entries.sort(key=lambda item: (item["source"], item["name"]))
    unsatisfied = [e["name"] for e in entries
                   if e["status"] == "unsatisfied"]
    missing = [e["name"] for e in entries if e["status"] == "missing"]
    unchecked = [e["name"] for e in entries if e["status"] == "unchecked"]
    return {
        "conforms": not unsatisfied and not missing,
        "complete": not unsatisfied and not missing and not unchecked,
        "entries": entries,
        "unsatisfied": unsatisfied,
        "missing": missing,
        "unchecked": unchecked,
        "declared": {source: [f"{name}{specifier}"
                               for name, specifier in requirements]
                     for source, requirements in declared.items()},
        "installed": {name: installed[name] for name, _ in seen
                      if name in installed},
    }


def check_dependency_conformance(root=None, installed=None) -> dict:
    """Do the installed distributions match the declared dependency set?

    The report names every declaration, the installed version and the
    verdict.  ``conforms`` is false when a declaration is unsatisfied or its
    distribution is missing; ``complete`` additionally requires that every
    declaration could be evaluated exactly (an unsupported specifier is
    reported, never counted as a pass).
    """
    declared = declared_requirements(root)
    versions = installed_distribution_versions() if installed is None \
        else dict(installed)
    return _conformance_report(declared, versions)


def dependency_conformance_from_record(record, root=None) -> dict:
    """Recompute the conformance verdict of an archived environment record.

    An archived record carries the installed package inventory, so the
    question "was this result produced with the dependency set the project
    declares?" can be answered from the evidence alone - no rerun and no
    access to the machine that produced it.
    """
    if not isinstance(record, dict):
        return {
            "conforms": False,
            "complete": False,
            "entries": [],
            "unsatisfied": [],
            "missing": [],
            "unchecked": [],
            "declared": {},
            "installed": {},
            "error": "environment record is missing or malformed",
        }
    inventory = record.get("package_inventory")
    if not isinstance(inventory, dict):
        return {
            "conforms": False,
            "complete": False,
            "entries": [],
            "unsatisfied": [],
            "missing": [],
            "unchecked": [],
            "declared": {},
            "installed": {},
            "error": "environment record carries no package inventory",
        }
    installed = {}
    for name, version in inventory.items():
        if isinstance(name, str) and isinstance(version, str):
            installed[_normalise_distribution_name(name)] = version
    return _conformance_report(declared_requirements(root), installed)
