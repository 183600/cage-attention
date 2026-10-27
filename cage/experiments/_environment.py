# cage/experiments/_environment.py - Source/environment fingerprints, context and manifest stamps.
# Executed into the cage.experiments namespace by __init__.py;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541
def _source_tree_sha256() -> str:
    """Content hash of the training package (cage/**/*.py).

    ``git_sha`` identifies the committed revision, but a run from a dirty
    tree is not rebuildable from that revision alone and the dirty flag does
    not say *what* code ran.  Hashing the actual source files gives every
    record a content-addressed code version that survives uncommitted
    edits, so two runs can be compared (and a stale baseline verdict can be
    refused) even when both report the same ``git_sha``.
    """
    global _SOURCE_TREE_SHA
    if _SOURCE_TREE_SHA is None:
        h = hashlib.sha256()
        for path in sorted(pathlib.Path(__file__).resolve().parent.parent.rglob("*.py")):
            h.update(path.name.encode("utf-8") + b"\0")
            h.update(path.read_bytes())
            h.update(b"\0")
        _SOURCE_TREE_SHA = h.hexdigest()[:16]
    return _SOURCE_TREE_SHA


def _run_identity() -> tuple[str, str]:
    """Per-process run id and start timestamp (excluded from setup hash).

    One invocation can write many JSONL records; the run id groups them and
    links the compact provenance stamp on error-case rows back to the full
    manifest, so stale rows from an earlier invocation are never mistaken
    for part of the current one.
    """
    global _RUN_ID, _RUN_STARTED_UTC
    if _RUN_ID is None:
        _RUN_ID = (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                   + "-" + uuid.uuid4().hex[:8])
        _RUN_STARTED_UTC = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return _RUN_ID, _RUN_STARTED_UTC


def _hypotheses_sha() -> str:
    """Content hash of the pre-registered hypothesis registry.

    Records carry the hash so a result can be matched to the exact
    decision rules in force when it was produced: if the registry is
    edited mid-analysis, old and new records can no longer be silently
    pooled under "the same" hypotheses.
    """
    payload = json.dumps(HYPOTHESES, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# Fields that describe *how this invocation was launched* rather than the
# numerical environment a rerun has to match.  They are deliberately kept in
# the record (they are useful provenance) but excluded from the environment
# fingerprint and from drift comparison: a rerun from another directory or
# with another argv is still the same environment.
_ENV_IGNORED_FIELDS = frozenset({
    "argv", "cwd", "repo_root", "git_dirty", "pythonhashseed",
    "environment_sha256",
})

# Hardware/toolchain descriptors that are reported as advisories instead of
# hard incompatibilities.  A different CPU model can change the last bits of
# a float, but it does not by itself mean that a *protocol* was changed;
# critical fields below identify the software/numerical recipe.
_ENV_ADVISORY_FIELDS = frozenset({
    "platform", "machine", "processor", "os_release", "cpu_model",
    "cpu_count", "cuda_device_count", "cuda_devices", "python_executable",
    "hash_randomization",
})

# Declared dependencies are content-addressed: the installed inventory alone
# cannot tell whether the project's own requirement range changed.
_DEPENDENCY_FILES = ("pyproject.toml", "requirements-ci.txt",
                     "requirements.txt")


def _package_inventory() -> dict:
    """Installed distribution name -> version, deterministically sorted.

    ``torch.__version__`` pins the framework but not the rest of the
    process.  An optional dependency (NumPy, a BLAS wheel, a plugin) can
    change numerics or import behaviour, so the environment record keeps the
    full installed inventory.  Package metadata is untrusted input: a broken
    distribution must not make the provenance probe fail a run.
    """
    packages: dict[str, str] = {}
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
        key = name.strip().lower().replace("_", "-").replace(".", "-")
        if not key or not version:
            continue
        # Two spellings of the same distribution should not make the
        # inventory order-dependent: keep the lexicographically largest
        # version so the result is stable regardless of scan order.
        previous = packages.get(key)
        packages[key] = version if previous is None else max(previous, version)
    return dict(sorted(packages.items()))


def _dependency_manifest() -> dict:
    """Content hashes of the project's declared dependency files.

    The installed inventory answers "what ran"; these hashes answer "what
    was requested".  A requirements/pyproject edit changes the latter even
    when the current interpreter already has the new versions installed.
    """
    manifest = {}
    for name in _DEPENDENCY_FILES:
        path = _REPO_ROOT / name
        try:
            data = path.read_bytes()
        except OSError:
            continue
        manifest[name] = {
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    return manifest


def _environment_fingerprint(record: Mapping) -> str:
    """Stable content id for the reproducibility-relevant environment.

    All non-volatile fields participate, so a field added to the record
    automatically strengthens the fingerprint instead of being silently
    ignored.  Volatile launch metadata (argv/cwd/git status) is excluded:
    two runs from different directories still share an environment.
    """
    payload = {
        key: value for key, value in record.items()
        if key not in _ENV_IGNORED_FIELDS
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _mapping_delta(recorded: Mapping, current: Mapping) -> dict:
    """Summarise a nested mapping difference without dumping both copies."""
    recorded_keys = set(recorded)
    current_keys = set(current)
    return {
        "missing_keys": sorted(recorded_keys - current_keys),
        "extra_keys": sorted(current_keys - recorded_keys),
        "changed_keys": sorted(
            key for key in recorded_keys & current_keys
            if recorded[key] != current[key]),
    }


def _environment_drift(recorded, current) -> dict:
    """Compare a stored environment record with the current process.

    Returns a structured diagnosis rather than a bare boolean:

    * ``compatible`` - no *critical* field differs (the software/numerical
      recipe is unchanged); advisory hardware fields may still differ;
    * ``identical`` - no recorded field differs at all;
    * ``critical_differences`` / ``advisory_differences`` - the evidence for
      either verdict, with nested mappings summarised by missing/extra/
      changed keys instead of two full copies.

    Missing fields fail closed when they are critical: an old record that
    predates the package inventory cannot certify the packages that ran.
    A malformed record is itself an incompatibility, never an empty match.
    """
    result = {
        "compatible": False,
        "identical": False,
        "recorded_environment_sha256": None,
        "current_environment_sha256": None,
        "differences": [],
        "critical_differences": [],
        "advisory_differences": [],
    }
    if not isinstance(recorded, Mapping) or not isinstance(current, Mapping) \
            or not recorded:
        result["reason"] = "recorded or current environment is missing/malformed"
        return result
    recorded_sha = recorded.get("environment_sha256")
    current_sha = current.get("environment_sha256")
    if not isinstance(current_sha, str) or not current_sha:
        current_sha = _environment_fingerprint(current)
    result["recorded_environment_sha256"] = recorded_sha
    result["current_environment_sha256"] = current_sha
    for field in sorted((set(recorded) | set(current)) - _ENV_IGNORED_FIELDS):
        severity = "advisory" if field in _ENV_ADVISORY_FIELDS else "critical"
        if field not in recorded:
            difference = {
                "field": field, "kind": "missing_recorded",
                "severity": severity,
                "current": (current.get(field)
                            if not isinstance(current.get(field), Mapping)
                            else _mapping_delta({}, current[field])),
            }
        elif field not in current:
            difference = {
                "field": field, "kind": "missing_current",
                "severity": severity,
                "recorded": (recorded.get(field)
                             if not isinstance(recorded.get(field), Mapping)
                             else _mapping_delta(recorded[field], {})),
            }
        elif recorded[field] != current[field]:
            if isinstance(recorded[field], Mapping) \
                    and isinstance(current[field], Mapping):
                detail = _mapping_delta(recorded[field], current[field])
            else:
                detail = {"recorded": recorded[field],
                          "current": current[field]}
            difference = {
                "field": field, "kind": "changed",
                "severity": severity, **detail,
            }
        else:
            continue
        result["differences"].append(difference)
        result[f"{severity}_differences"].append(difference)
    # A hash mismatch with no field-level difference means the fingerprint
    # algorithm or its field set changed.  Treat that as a critical schema
    # mismatch instead of claiming the environments are identical.  A
    # malformed hash is likewise not allowed to pass as "no hash".
    valid_recorded_hash = (
        isinstance(recorded_sha, str) and len(recorded_sha) == 16
        and all(ch in "0123456789abcdef" for ch in recorded_sha))
    if recorded_sha is not None and not valid_recorded_hash:
        difference = {
            "field": "environment_sha256", "kind": "malformed",
            "severity": "critical",
            "recorded": recorded_sha, "current": current_sha,
        }
        result["differences"].append(difference)
        result["critical_differences"].append(difference)
    elif (isinstance(recorded_sha, str) and recorded_sha
          and recorded_sha != current_sha and not result["differences"]):
        difference = {
            "field": "environment_sha256", "kind": "fingerprint_schema",
            "severity": "critical",
            "recorded": recorded_sha, "current": current_sha,
        }
        result["differences"].append(difference)
        result["critical_differences"].append(difference)
    result["compatible"] = not result["critical_differences"]
    result["identical"] = not result["differences"]
    if result["identical"]:
        result["reason"] = "environment matches the recorded one"
    elif result["compatible"]:
        result["reason"] = ("critical software/numerical fields match; "
                            "advisory hardware fields differ")
    else:
        result["reason"] = ("critical software/numerical fields differ: "
                            + ", ".join(
                                d["field"] for d in
                                result["critical_differences"]))
    return result


def _extract_environment_record(payload):
    """Accept either a bare environment block or a run-context document."""
    if not isinstance(payload, Mapping):
        return None, None
    nested = payload.get("environment")
    if isinstance(nested, Mapping):
        return nested, payload
    if "torch" in payload and "python" in payload:
        return payload, None
    return None, payload


def _environment_record(settings: Settings, git_dirty: bool | None = None) -> dict:
    """Software/hardware provenance for reproducibility.

    Beyond versions the record pins the numerical-behaviour switches a
    rerun must match: CPU thread count and the BLAS/OpenMP environment
    overrides that determine it (parallel reductions change the summation
    order), the CPU model and OS release, cuDNN determinism/benchmark,
    whether deterministic algorithms are on, the cuBLAS workspace config,
    the torch build identity/backends and the NumPy version when installed.
    The determinism switch is applied from ``Settings.deterministic``
    *before* it is read, so the recorded state is the one this run will
    train under even when the context is built before the first training
    call (a previous run in the same process may have left the global flag
    in the other position).

    The record additionally carries the full installed package inventory,
    content hashes of the declared dependency files, and a stable
    ``environment_sha256`` over the non-volatile fields.  The hash makes
    "same environment" a one-field comparison; :func:`_environment_drift`
    explains a mismatch field by field.
    """
    determinism = configure_determinism(settings.deterministic)
    numeric_env = {}
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "CUBLAS_WORKSPACE_CONFIG"):
        numeric_env[name] = os.environ.get(name)
    env = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "torch": torch.__version__,
        "torch_git_version": getattr(torch.version, "git_version", None),
        "torch_debug": bool(getattr(torch.version, "debug", False)),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "os_release": platform.release(),
        "device": settings.device,
        "cpu_count": os.cpu_count(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "torch_num_threads": determinism["torch_num_threads"],
        "deterministic_requested": bool(settings.deterministic),
        "deterministic_algorithms": determinism["deterministic_algorithms"],
        "cublas_workspace_config": determinism["cublas_workspace_config"],
        "tf32": determinism["tf32"],
        # Thread-count overrides are read by the BLAS/OpenMP runtimes before
        # torch is imported, so they are part of the numerical recipe even
        # though ``torch_num_threads`` records the resulting value.
        "numeric_env": numeric_env,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "hash_randomization": bool(sys.flags.hash_randomization),
        # How the run was launched is part of the recipe: the same config
        # can mean different code paths through CLI flags, and a result is
        # only rebuildable when its invocation is recorded.
        "argv": [str(arg) for arg in sys.argv],
        "cwd": os.getcwd(),
        "repo_root": str(_REPO_ROOT),
        "git_dirty": (_git_dirty() if git_dirty is None else bool(git_dirty)),
    }
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    env["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    # ``/proc/cpuinfo`` is absent on some platforms (and ARM kernels may not
    # expose a ``model name`` line).  Keep the record schema stable: a
    # missing probe is ``None``, not a missing key that breaks consumers
    # expecting the documented field.
    env.setdefault("cpu_model", None)
    for name in ("mkldnn", "openmp"):
        backend = getattr(torch.backends, name, None)
        probe = getattr(backend, "is_available", None)
        try:
            env[f"torch_{name}"] = bool(probe()) if probe is not None else None
        except Exception:  # noqa: BLE001 - probe must not fail a record
            env[f"torch_{name}"] = None
    try:
        env["torch_num_interop_threads"] = int(torch.get_num_interop_threads())
    except Exception:  # noqa: BLE001 - older/limited builds may not expose it
        env["torch_num_interop_threads"] = None
    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is not None:
        env["cudnn_deterministic"] = bool(cudnn.deterministic)
        env["cudnn_benchmark"] = bool(cudnn.benchmark)
        try:
            env["cudnn_version"] = cudnn.version()
        except Exception:  # noqa: BLE001 - probe must not fail a record
            env["cudnn_version"] = None
    try:
        import numpy as np
        env["numpy"] = np.__version__
    except ImportError:
        env["numpy"] = None
    if torch.cuda.is_available():
        try:
            count = int(torch.cuda.device_count())
            devices = []
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                devices.append({
                    "index": i,
                    "name": props.name,
                    "capability": f"{props.major}.{props.minor}",
                    "total_memory_bytes": int(props.total_memory),
                })
            env["cuda_device_count"] = count
            env["cuda_devices"] = devices
        except Exception as exc:  # noqa: BLE001 - probe must not fail a record
            env["cuda_devices_error"] = f"{type(exc).__name__}: {exc}"
    # Full interpreter string (the short ``python`` field omits the build),
    # the complete package set, and content ids for the declared dependency
    # files.  Computed last so the fingerprint below covers every probe that
    # actually ran.
    env["python_full_version"] = sys.version
    env["package_inventory"] = _package_inventory()
    env["dependency_manifest"] = _dependency_manifest()
    env["environment_sha256"] = _environment_fingerprint(env)
    return env


def _read_environment_file(path) -> tuple[dict | None, dict | None, str | None]:
    """Read an ``environment.json`` or run manifest.

    Returns ``(environment_block, whole_payload, error)``.  A malformed or
    schema-less file is an error rather than an empty record: treating a
    corrupt provenance file as "no previous environment" would let a rerun
    silently bless an unverifiable results directory.
    """
    try:
        payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        return None, None, f"{type(exc).__name__}: {exc}"
    environment, context = _extract_environment_record(payload)
    if environment is None:
        return None, payload if isinstance(payload, dict) else None, (
            "file has no environment block")
    return environment, context, None


def _record_environment_history(out_dir, context: Mapping) -> dict:
    """Compare this invocation to the previous environment in ``out_dir``.

    A results directory is routinely appended to across days, package
    upgrades and machines.  Silently mixing records from different
    environments makes the artifact index certify a directory whose numbers
    were not produced by one recipe.  Every invocation therefore appends a
    compatibility row to ``environment_history.jsonl`` before the new
    ``environment.json`` replaces the old one; a critical drift is printed
    and (with ``--strict-environment``) aborts the run, while an advisory
    hardware drift is recorded but allowed.

    The previous file is the *only* source of the old record: it is read
    before the caller writes the new one.  If it is missing or malformed the
    history still records that fact (``compatible=False`` for malformed,
    ``None`` for a first run) so the provenance chain has no silent gap.
    """
    out_dir = pathlib.Path(out_dir)
    previous_path = out_dir / "environment.json"
    previous, previous_payload, previous_error = (None, None, None)
    if previous_path.exists():
        previous, previous_payload, previous_error = _read_environment_file(
            previous_path)
    current = context.get("environment") if isinstance(context, Mapping) else None
    if not isinstance(current, Mapping):
        drift = {
            "compatible": False,
            "identical": False,
            "reason": "current environment record is missing",
            "differences": [], "critical_differences": [],
            "advisory_differences": [],
        }
    elif previous_error is not None:
        drift = {
            "compatible": False,
            "identical": False,
            "reason": f"previous environment record is unreadable: "
                      f"{previous_error}",
            "differences": [], "critical_differences": [],
            "advisory_differences": [],
        }
    elif previous is None:
        drift = {
            "compatible": None,
            "identical": None,
            "reason": "no previous environment record (first run in this "
                      "results directory)",
            "differences": [], "critical_differences": [],
            "advisory_differences": [],
        }
    else:
        drift = _environment_drift(previous, current)
    record = {
        "run_id": context.get("run_id"),
        "schema_version": context.get("schema_version"),
        "setup_sha256": context.get("setup_sha256"),
        "git_sha": context.get("git_sha"),
        "source_tree_sha256": context.get("source_tree_sha256"),
        "event": "environment_checked",
        "previous_environment_sha256": (
            previous.get("environment_sha256")
            if isinstance(previous, Mapping) else None),
        "current_environment_sha256": (
            current.get("environment_sha256")
            if isinstance(current, Mapping) else None),
        "previous_setup_sha256": (
            previous_payload.get("setup_sha256")
            if isinstance(previous_payload, Mapping) else None),
        "previous_git_sha": (
            previous_payload.get("git_sha")
            if isinstance(previous_payload, Mapping) else None),
        "compatible": drift.get("compatible"),
        "identical": drift.get("identical"),
        "reason": drift.get("reason"),
        "differences": drift.get("differences", []),
    }
    try:
        _write_result(out_dir / "environment_history.jsonl", record)
    except TrackingError:
        # A certificate mismatch is not a transient I/O problem: continuing
        # would let the end-of-run index writer re-bless edited evidence.
        raise
    except Exception as exc:  # noqa: BLE001 - history must not kill a run
        print(f"[runner] WARNING: could not record environment history: {exc}")
    return drift


def _describe_environment_difference(difference: Mapping) -> str:
    """One readable line for an environment drift entry."""
    field = difference.get("field", "<unknown>")
    kind = difference.get("kind")
    severity = difference.get("severity", "critical")
    if kind == "missing_recorded":
        detail = "missing from the recorded environment"
    elif kind == "missing_current":
        detail = "missing from the current environment"
    elif "changed_keys" in difference:
        detail = ("missing={missing_keys} extra={extra_keys} "
                  "changed={changed_keys}").format(**difference)
    elif kind == "fingerprint_schema":
        detail = (f"fingerprint {difference.get('recorded')} != "
                  f"{difference.get('current')}")
    else:
        detail = (f"recorded={difference.get('recorded')!r} "
                  f"current={difference.get('current')!r}")
    return f"[{severity}] {field}: {detail}"


def _check_environment(out_dir) -> int:
    """Compare the environment stored in ``out_dir`` with this process.

    Exit 0 when the critical software/numerical fields match, 1 when they
    drift, 2 when no usable environment record exists.  The check is
    read-only: it writes nothing into the results directory.
    """
    root = pathlib.Path(out_dir)
    candidates = []
    primary = root / "environment.json"
    if primary.is_file():
        candidates.append(primary)
    if not candidates and root.is_dir():
        candidates.extend(sorted(root.glob("*_manifest.json")))
    if not candidates:
        print(f"[env-check] no environment.json or run manifest under {root}; "
              f"nothing to check")
        return 2
    path = candidates[0]
    recorded, payload, error = _read_environment_file(path)
    if error is not None:
        print(f"[env-check] cannot read {path}: {error}")
        return 2
    device = recorded.get("device") if isinstance(
        recorded.get("device"), str) else "cpu"
    deterministic = recorded.get("deterministic_requested")
    if not isinstance(deterministic, bool):
        deterministic = True
    try:
        current = _environment_record(
            Settings(device=device, deterministic=deterministic),
            git_dirty=None)
    except Exception as exc:  # noqa: BLE001 - a bad record is a check error
        print(f"[env-check] cannot build the current environment record: "
              f"{type(exc).__name__}: {exc}")
        return 2
    drift = _environment_drift(recorded, current)
    run_label = (payload.get("run_id") if isinstance(payload, Mapping)
                 else None)
    print(f"[env-check] recorded={path} run={run_label} "
          f"recorded_env={drift.get('recorded_environment_sha256')} "
          f"current_env={drift.get('current_environment_sha256')}")
    for difference in drift["differences"]:
        print("    " + _describe_environment_difference(difference))
    print(f"[env-check] {drift.get('reason')}")
    if drift["compatible"]:
        print("[env-check] PASS: critical software/numerical fields match")
        return 0
    print("[env-check] FAIL: the stored result was produced under a "
          "different software/numerical environment")
    return 1


def _source_hash_of(record):
    """Code-content hash carried by one archived record, if any.

    Full run records and manifests carry ``source_tree_sha256`` at the top
    level; analysis companions nest the same stamp under ``_provenance``.
    Documents without either are ignored here; :func:`_check_source` still
    fails closed when a record declares a ``run_id`` or ``setup_sha256``
    without a source hash.
    """
    if not isinstance(record, Mapping):
        return None
    value = record.get("source_tree_sha256")
    if value is None:
        provenance = record.get("_provenance")
        if isinstance(provenance, Mapping):
            value = provenance.get("source_tree_sha256")
    return value


def _source_hash_values(record) -> list:
    """Every code hash carried by one archived record.

    A record can carry the stamp at the top level, under ``_provenance``,
    or both.  Both copies are returned so a stale nested companion cannot
    hide behind a current top-level stamp.
    """
    if not isinstance(record, Mapping):
        return []
    values = []
    if "source_tree_sha256" in record:
        values.append(record.get("source_tree_sha256"))
    provenance = record.get("_provenance")
    if isinstance(provenance, Mapping) \
            and "source_tree_sha256" in provenance:
        values.append(provenance.get("source_tree_sha256"))
    return values


def _declares_code_provenance(record) -> bool:
    """Whether a record claims provenance that should carry a code hash.

    Analysis companions put the stamp under ``_provenance``, so the
    declaration may live one level below the record itself.
    """
    if not isinstance(record, Mapping):
        return False
    if "run_id" in record or "setup_sha256" in record:
        return True
    provenance = record.get("_provenance")
    return (isinstance(provenance, Mapping)
            and ("run_id" in provenance or "setup_sha256" in provenance))


def _iter_archived_source_record_mappings(document):
    """Yield every record mapping in a parsed JSON artifact.

    Most artifacts are objects, but pooled manifests may be top-level
    arrays (a JSONL line can likewise hold one), and tuning/analysis
    companions may nest per-run records under a mapping key (for example
    ``{"runs": [...]}``).  A hash stored on those records must still be
    checked; descending only through arrays would silently skip every
    record nested one level below an object.
    """
    pending = [document]
    while pending:
        item = pending.pop()
        if isinstance(item, Mapping):
            yield item
            pending.extend(reversed([
                value for key, value in item.items()
                if key != "_provenance"]))
        elif isinstance(item, (list, tuple)):
            pending.extend(reversed(item))


def _iter_archived_source_records(root: pathlib.Path):
    """Yield ``(relative, line, record, error)`` for archived JSON documents.

    Every ``*.json`` / ``*.jsonl`` artifact under the results directory is
    scanned, so a code hash is checked wherever an experiment chose to store
    it.  A malformed document is yielded with ``record=None`` and an error
    string instead of being skipped: an unreadable provenance file cannot
    certify the code that produced it.
    """
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in (".json", ".jsonl"):
            continue
        if path.name.startswith("."):
            continue
        relative = str(path.relative_to(root))
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            yield relative, None, None, f"{type(exc).__name__}: {exc}"
            continue
        if path.suffix == ".jsonl":
            for line_number, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except (TypeError, ValueError) as exc:
                    yield (relative, line_number, None,
                           f"invalid JSON: {exc}")
                    continue
                yield relative, line_number, record, None
            continue
        try:
            record = json.loads(text)
        except (TypeError, ValueError) as exc:
            yield relative, None, None, f"invalid JSON: {exc}"
            continue
        yield relative, None, record, None


def _check_source(out_dir) -> int:
    """Compare archived code-content hashes with this checkout.

    ``--verify-artifacts`` proves the bytes did not change and
    ``--verify-provenance`` proves each row is internally consistent, but
    neither says whether the *current* package source is the code that
    produced the stored numbers.  A result directory is routinely reused
    after a code edit; appending to it or pooling its rows without checking
    the code version silently mixes protocols.  This read-only audit reads
    every archived ``source_tree_sha256`` and compares it with the hash of
    the current ``cage/*.py`` tree.

    Exit codes: 0 = every archived code hash matches this checkout, 1 = a
    mismatched or malformed hash (or an unreadable document), 2 = no
    archived code-provenance records.  Nothing is written.
    """
    root = pathlib.Path(out_dir)
    if not root.is_dir():
        print(f"[source-check] no results directory at {root}")
        return 2
    current = _source_tree_sha256()
    git_sha = _git_sha()
    git_dirty = _git_dirty()
    checked = 0
    problems = []
    versions: dict[str, list[str]] = {}
    for relative, line_number, document, error in (
            _iter_archived_source_records(root)):
        where = relative if line_number is None else f"{relative}:{line_number}"
        if error is not None:
            problems.append(f"{where}: {error}")
            continue
        for record in _iter_archived_source_record_mappings(document):
            values = _source_hash_values(record)
            provenance = record.get("_provenance")
            nested_declares = (
                isinstance(provenance, Mapping)
                and ("run_id" in provenance
                     or "setup_sha256" in provenance))
            if not values:
                if _declares_code_provenance(record):
                    checked += 1
                    problems.append(
                        f"{where}: code-provenance record has no "
                        f"source_tree_sha256")
                continue
            checked += 1
            if nested_declares and "source_tree_sha256" not in provenance:
                problems.append(
                    f"{where}: nested provenance has no "
                    f"source_tree_sha256")
            valid = []
            for value in values:
                if not _is_short_sha(value):
                    problems.append(
                        f"{where}: source_tree_sha256 is malformed "
                        f"({value!r})")
                    continue
                if value not in valid:
                    valid.append(value)
                    versions.setdefault(value, []).append(where)
                if value != current:
                    run_id = record.get("run_id")
                    if not run_id and isinstance(provenance, Mapping):
                        run_id = provenance.get("run_id")
                    problems.append(
                        f"{where}: source_tree_sha256 {value} != "
                        f"current {current}"
                        + (f" (run {run_id})" if run_id else ""))
            if len(valid) > 1:
                problems.append(
                    f"{where}: conflicting source_tree_sha256 values "
                    f"{sorted(valid)}")
    print(f"[source-check] current source_tree_sha256={current} "
          f"git_sha={git_sha} dirty={git_dirty}")
    if not checked and not problems:
        print(f"[source-check] no archived code-provenance records under "
              f"{root}")
        return 2
    print(f"[source-check] scanned {checked} code-provenance record(s) "
          f"across {len(versions)} distinct source version(s)")
    for value in sorted(versions):
        label = "current" if value == current else "other"
        print(f"    {value} ({label}): {len(versions[value])} record(s)")
    if problems:
        print(f"[source-check] FAIL: {len(problems)} issue(s):")
        for problem in problems:
            print(f"    {problem}")
        return 1
    print("[source-check] PASS: every archived result was produced by the "
          "current package source")
    return 0


_SEEDING_POLICY = {
    "every_training_run": (
        "cage.repro.seed_everything(seed): python random + numpy (when "
        "installed) + torch CPU and all CUDA devices; deterministic kernels "
        "are requested by default (Settings.deterministic=True)"),
    "seed_canonicalisation": (
        "every RNG (Python, NumPy, torch, CUDA) and every data-stream "
        "generator consumes the same low-32-bit seed key, so s and "
        "s + 2**32 are the same run; bool/fractional seeds are rejected "
        "instead of silently truncated"),
    "determinism_switch": (
        "Settings.deterministic is applied by every seeding call, including "
        "train_model; it defaults to True and can be opted out via "
        "--no-deterministic; False explicitly restores non-deterministic "
        "kernels so a prior deterministic run cannot leak into a later one"),
    "data_streams": (
        "each split and training seed owns a dedicated torch.Generator "
        "(DataLoader generator= equivalent; make_batch never reads the "
        "global RNG), so data order cannot be perturbed by other seeding"),
    "eval_stream_seeds": {"val": VAL_SEED, "test": TEST_SEED,
                          "stats": STATS_SEED},
    "reserved_eval_seeds": list(RESERVED_EVAL_SEEDS),
    "scope": ("run-to-run stable on the same machine/build; bit-exact "
              "reproduction across different hardware or kernel libraries "
              "is not promised"),
}


def _config_cache_key(settings: Settings) -> str:
    """Content key for per-``Settings`` caches.

    The runners mutate ``Settings`` in place (``--seeds``, CLI overrides,
    verdict marks), so object identity alone must never decide whether a
    cached audit is still valid: a mutated config would otherwise reuse
    results computed for a different recipe and silently certify the wrong
    setup.  Every cache below stores this key beside its value.
    """
    return json.dumps(
        {name: getattr(settings, name)
         for name in Settings.__dataclass_fields__ if name != "out_dir"},
        ensure_ascii=False, sort_keys=True, default=str)


def _run_context(settings: Settings) -> dict:
    """Self-contained provenance block attached to every result record.

    Code revision, hypothesis-registry hash, resolved configuration (every
    hyperparameter and seed actually used), software environment, the
    seeding policy and the data-stream fingerprints travel *with* each
    record, so a result file is traceable without relying on defaults that
    may have changed since.  The ``setup_sha256`` hash over exactly
    {revision, source-tree hash, hypotheses, config, data fingerprints}
    gives every record a compact id for "same code + same recipe + same
    data": records carrying different ids were not produced by the same
    setup and cannot be pooled silently.  ``run_id`` groups all records of
    one invocation.  Cached on the Settings object and invalidated when the
    resolved config changes, so a mutated Settings cannot reuse a stale
    provenance block.

    The configured training-seed set is re-validated on every call (cheap,
    before the cache lookup): every official entry point builds the run
    context before training, so a seed set mutated after construction -
    the runner applies ``--seeds`` that way - is still refused when it is
    not duplicate-free / disjoint from the reserved evaluation streams.
    """
    # Keep the validated/normalised tuple: a caller may have assigned a
    # one-shot iterable after construction, and validating it would consume
    # the iterator before the config/fingerprint loops below read it.
    settings.seeds = _validate_training_seeds(settings.seeds)
    config = {name: getattr(settings, name)
              for name in Settings.__dataclass_fields__ if name != "out_dir"}
    cache_key = _config_cache_key(settings)
    ctx = getattr(settings, "_run_context_cache", None)
    if ctx is not None and getattr(settings, "_run_context_cache_key", None) \
            == cache_key:
        return ctx
    # Fingerprint each stream at the geometry its consumer actually uses:
    # make_batch's draws are shape-dependent, so one seed at two batch
    # sizes is two realizations.  Training reads settings.batch; validation
    # monitoring reads settings.batch too (same fixed VAL stream), final
    # val/test read eval_batch, mechanism statistics read stats_batch, the
    # linear probe trains at probe_batch and selects/scores at
    # probe_eval_batch, and the overfit sanity check reads overfit_batch.
    fingerprints = {
        "val": stream_fingerprint(
            settings.n_pairs, settings.vocab, settings.device,
            batch=settings.eval_batch, seed=VAL_SEED),
        "val_monitor": stream_fingerprint(
            settings.n_pairs, settings.vocab, settings.device,
            batch=settings.batch, seed=VAL_SEED),
        "test": stream_fingerprint(
            settings.n_pairs, settings.vocab, settings.device,
            batch=settings.eval_batch, seed=TEST_SEED),
        "stats": stream_fingerprint(
            settings.n_pairs, settings.vocab, settings.device,
            batch=settings.stats_batch, seed=STATS_SEED),
        "probe_val": stream_fingerprint(
            settings.n_pairs, settings.vocab, settings.device,
            batch=settings.probe_eval_batch, seed=VAL_SEED),
        "probe_test": stream_fingerprint(
            settings.n_pairs, settings.vocab, settings.device,
            batch=settings.probe_eval_batch, seed=TEST_SEED),
        "overfit_sanity": stream_fingerprint(
            settings.n_pairs, settings.vocab, settings.device,
            batch=settings.overfit_batch, seed=SANITY_SEED),
    }
    fingerprints["train"] = {
        str(seed): stream_fingerprint(
            settings.n_pairs, settings.vocab, settings.device,
            batch=settings.batch, seed=seed)
        for seed in settings.seeds
    }
    fingerprints["probe_train"] = {
        str(seed): stream_fingerprint(
            settings.n_pairs, settings.vocab, settings.device,
            batch=settings.probe_batch, seed=seed)
        for seed in settings.seeds
    }
    git_sha = _git_sha()
    git_dirty = _git_dirty()
    source_tree = _source_tree_sha256()
    run_id, run_started_utc = _run_identity()
    payload = json.dumps({
        "git_sha": git_sha,
        "source_tree_sha256": source_tree,
        "hypotheses_sha256": _hypotheses_sha(),
        "config": config,
        "data_fingerprints": fingerprints,
    }, ensure_ascii=False, sort_keys=True, default=str)
    ctx = {
        "run_id": run_id,
        "schema_version": _RECORD_SCHEMA,
        "run_started_utc": run_started_utc,
        "git_sha": git_sha,
        "git_dirty": bool(git_dirty),
        "source_tree_sha256": source_tree,
        "hypotheses_sha256": _hypotheses_sha(),
        "setup_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16],
        "environment": _environment_record(settings, git_dirty),
        "config": config,
        "seeding": dict(_SEEDING_POLICY),
        "data_fingerprints": fingerprints,
        # Frozen prospective design-plan id (``None`` before the plan is
        # published, e.g. while the baselines runner is producing its
        # verdict).  Excluded from the setup hash: the plan is a design
        # commitment, not a numerical recipe.
        "design_sha256": getattr(settings, "_design_sha256", None),
        # Validity-gate summary stamped by ``_enforce_baseline_verdict``
        # (None for the baselines runner itself, which is the producer).
        # Deliberately excluded from the setup hash above: the verdict
        # describes whether this protocol passed its gates, not which
        # protocol it is.
        "baseline_validity": getattr(settings, "_baseline_validity", None),
    }
    settings._run_context_cache = ctx
    settings._run_context_cache_key = cache_key
    return ctx


def _stamp(settings: Settings) -> dict:
    """Compact provenance stamp for auxiliary/error-case records.

    Full provenance lives in the run manifest; the stamp carries just the
    ids needed to join a raw case row back to it without repeating the
    environment and every hyperparameter on each row.
    """
    ctx = _run_context(settings)
    return {key: ctx[key] for key in
            ("run_id", "schema_version", "setup_sha256", "git_sha",
             "source_tree_sha256")}


def _protocol_sha256(ctx: dict) -> str:
    """Hash over exactly the resolved config and data-stream fingerprints.

    Excludes the code revision, hypothesis registry and environment on
    purpose: those do not change *which data* a check was run on.  The
    source-tree hash is checked separately by
    :func:`_verdict_provenance_note`, because the gate *implementation*
    must also match.  Any config or stream-fingerprint change invalidates
    the stored HB marks.
    """
    payload = json.dumps({"config": ctx["config"],
                          "data_fingerprints": ctx["data_fingerprints"]},
                         ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _verdict_provenance_note(verdict: dict, settings: Settings) -> str | None:
    """Explain why a stored baselines verdict may not apply to this run.

    A verdict written under a different config/data protocol (or before
    the stamp existed) cannot annotate the current grids: its HB1/HB2/HB4/
    HB5/HB6/HB7 marks were measured on different streams or a different
    recipe.  The source-tree hash is checked separately: the gates are
    implemented in this package, so a verdict produced by older code is not
    evidence about the current code even when config and data match (a
    dirty tree with the same ``git_sha`` makes that distinction necessary).
    """
    stored = verdict.get("protocol_sha256")
    if stored is None:
        return ("baselines verdict predates protocol stamping; re-run "
                "`--only baselines` so its HB marks describe the current "
                "data/recipe")
    current = _protocol_sha256(_run_context(settings))
    if stored != current:
        return (f"baselines verdict protocol {stored} != current {current}; "
                f"it was produced under a different config/data protocol, so "
                f"its HB marks do not describe this run - re-run "
                f"`--only baselines`")
    stored_source = verdict.get("source_tree_sha256")
    current_source = _source_tree_sha256()
    if stored_source is None:
        return ("baselines verdict predates source-tree stamping; re-run "
                "`--only baselines` so its HB marks were produced by the "
                "current gate code")
    if stored_source != current_source:
        return (f"baselines verdict source {stored_source} != current "
                f"{current_source}; it was produced by older gate code, so "
                f"its HB marks do not certify this run - re-run "
                f"`--only baselines`")
    return None


def _write_manifest(settings: Settings, out_dir: pathlib.Path, tag: str) -> dict:
    """Per-experiment provenance manifest, written before any training.

    One ``<tag>_manifest.json`` per experiment pins the invocation run id,
    the code revision (plus dirty flag and content hash), the
    pre-registered hypotheses and their hash, every hyperparameter and
    seed, the data-stream fingerprints (including each training stream) and
    the software environment.  Run records repeat this context inline; the
    manifest is the single artifact that makes a results directory
    traceable back to a reproducible setup without joining files by hand.
    """
    manifest = {
        "experiment": tag,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **_run_context(settings),
    }
    _write_json(out_dir / f"{tag}_manifest.json", manifest)
    return manifest


def _load_baseline_verdict(out_dir: pathlib.Path):
    """The HB1-HB5 verdicts written by ``run_baselines``, if present.

    E1/E2/E3 read the learned model against the simple baselines, so a
    missing verdict file is surfaced as a note instead of silently
    treating the baseline check as passed.  A verdict file written by an
    older version lacks the HB4/HB5 keys or the protocol/source stamps; the
    marks are reported as ``n/a`` (not ``FAILED``) and a provenance warning
    asks for a re-run, so neither a missing check nor a stale one is
    mistaken for a pass.
    """
    path = out_dir / "baselines_verdict.json"
    if not path.exists():
        return None
    try:
        verdict = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    # A syntactically valid JSON array/scalar is not a verdict record; treat
    # it like a corrupt/missing file so the caller raises the documented
    # BaselineVerdictError instead of failing on ``.get``.
    return verdict if isinstance(verdict, dict) else None


class BaselineVerdictError(RuntimeError):
    """A baseline validity gate failed (or its verdict is missing/stale).

    HB1 (task construction), HB3 (split disjointness), HB4 (label-leak
    negative control) and HB5 (training loop) pre-register that their
    failure invalidates every result above them.  The E1/E2/E3 runners
    enforce that semantics by refusing to train on a protocol whose
    verdict is absent, stale, or failed - the same hard-gate policy as
    the reproducibility and overfit pre-flights.
    """


# Validity gates whose failure invalidates every grid result (the
# pre-registered "invalidate everything above it" semantics of HB1/HB3,
# plus the loop gate).  HB2 is deliberately not in this set: its
# registered semantics invalidate the *simple-baseline comparison*, so its
# state is stamped and displayed (certified/inconclusive/violated) instead
# of blocking the grid.  HB4 is also special-cased below: only a
# *violated* negative control blocks, while an inconclusive one is
# stamped (with a finite stream, "not shown" is not "leakage shown").
_BLOCKING_VERDICT_GATES = ("hb1_supported", "hb3_supported",
                           "hb5_supported")


def _verdict_seed_floor(verdict: dict, settings: Settings) -> dict:
    """The pre-registered seed floor as applied to the verdict's marks.

    ``min_seeds_for_claim`` (Settings, hashed) says a seed-level claim
    needs at least that many independent training seeds.  The baselines
    verdict applies the same floor to HB2/HB4/HB6/HB7: a supported mark
    resting on fewer seeds is *provisional* (displayed and stamped as
    such, never silently certified), matching the decision layer of the
    contrast families where a 2-seed "win" is marked INELIGIBLE.
    """
    if not isinstance(verdict, dict):
        verdict = {}
    floor = int(settings.min_seeds_for_claim)
    stored = verdict.get("seed_floor")
    if not isinstance(stored, dict):
        stored = {}
    if "n_seeds" not in stored:
        total = len(settings.seeds)
    else:
        total = _positive_integer(stored["n_seeds"])
        # A corrupted count is not a licence to fall back to the configured
        # seed set: mark the affected floor ineligible so a hand-edited
        # verdict cannot silently certify a claim.
        if total is None:
            total = 0
    out = {"min_seeds_for_claim": floor, "n_seeds": int(total)}
    for key in ("hb2", "hb4", "hb6", "hb7"):
        key_n = f"{key}_n_seeds"
        if key_n not in stored:
            n = total
        else:
            n = _positive_integer(stored[key_n])
            if n is None:
                n = 0
        flag_key = f"{key}_claim_eligible"
        if flag_key in stored:
            flag = stored[flag_key]
            # The stored mark cannot override the seed floor: if it says a
            # claim is eligible while its own count is below the registered
            # minimum, the verdict is internally inconsistent (or was
            # hand-edited) and must fail closed.  A genuine ``False`` is
            # still respected.
            eligible = (isinstance(flag, bool) and flag
                        and int(n) >= floor)
        else:
            eligible = bool(int(n) >= floor)
        out[f"{key}_n_seeds"] = int(n)
        out[f"{key}_claim_eligible"] = bool(eligible)
    return out
