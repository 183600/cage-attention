# cage/experiments/_artifacts.py - Result writing, artifact index and provenance audit.
# Executed into the cage.experiments namespace by __init__.py;
# split for readability, sharing the module globals.
# ruff: noqa: F821, F541
def _format_case(case) -> str:
    """One-line, human-readable bad case for the console."""
    ctx = case["context"]
    n_pairs = max(0, (len(ctx) - 2) // 2)
    pairs = " ".join(f"{ctx[2 * i]}:{ctx[2 * i + 1]}" for i in range(n_pairs))
    flags = []
    if case["confidence"] >= 0.9:
        flags.append("high_conf")
    if case["confidence"] < 0.5:
        flags.append("low_conf")
    if case["pred_in_context"]:
        flags.append("pred_in_context")
    if case["naive_confusable"]:
        flags.append("naive_confusable")
    if case["value_repeats"]:
        flags.append("value_repeats")
    by_layer = case.get("attn_mass_on_key_by_layer") or []
    profile = ""
    if by_layer:
        profile = " mass_by_layer=[" + ",".join(
            f"{v:.2f}" for v in by_layer) + "]"
    return (f"pair#{case['key_pair_idx']} key={case['query_key']} "
            f"ans={case['answer']} pred={case['pred']} "
            f"conf={case['confidence']:.2f} "
            f"mass_key={case['attn_mass_on_key']:.2f}{profile} "
            f"type={case['error_type']} "
            f"flags=[{','.join(flags) or 'plain'}] pairs=[{pairs}]")


def _print_conf_by_error_type(named_runs, title="[e1]"):
    """Mean confidence inside each error bucket, per arm / grid cell.

    The bucket shares say what failed; this table says whether the model
    was confident while failing.  A high-confidence ``retrieval_miss`` is a
    systematic blind spot worth a mechanism change, while a low-confidence
    ``hard_structural`` error is ambiguity the task imposes and no
    architecture fix will remove.
    """
    print(f"\n{title} mean answer confidence inside each error bucket "
          f"(high = confident blind spot; low = ambiguous sample); 'n/a' "
          f"where the bucket is empty:")
    width = max(16, max(len(t) for t in _ERROR_TYPES) + 1)
    print(f"{'arm':>{width}}" + "".join(f"{t:>{width}}" for t in _ERROR_TYPES))
    for item in _materialize_runs(named_runs):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            continue
        label, rs = item
        rs = _mapping_runs(rs)
        cells = []
        for etype in _ERROR_TYPES:
            vals = []
            for r in rs:
                bucket = r.get("err_conf_by_error_type")
                if not isinstance(bucket, Mapping):
                    continue
                value = bucket.get(etype)
                if _finite(value):
                    vals.append(float(value))
            cells.append(f"{sum(vals) / len(vals):.3f}"
                         if vals else "n/a")
        print(f"{label:>{width}}" + "".join(f"{c:>{width}}" for c in cells))


def _print_structural_decomposition(named_runs, title="[e1]"):
    """Disjoint decomposition of the ``hard_structural`` error bucket.

    ``naive_confusable`` (the query token also appears as a later value)
    and ``value_repeats`` (the answer value occurs more than once) are
    overlapping task properties; the primary taxonomy keeps their union as
    ``hard_structural``.  This companion reports the disjoint naive-only /
    repeat-only / both shares so one source cannot hide behind the other.
    """
    print(f"\n{title} hard-structural decomposition (share of errors; "
          f"disjoint components of the unchanged ``hard_structural`` "
          f"bucket):")
    print(f"{'arm':>23} {'naive_only':>12} {'repeat_only':>12} "
          f"{'both':>12}")
    for item in _materialize_runs(named_runs):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            continue
        label, rs = item
        rs = _mapping_runs(rs)
        cells = []
        for key in ("naive_only", "repeat_only", "both"):
            agg = _mean_std([
                r.get(f"err_taxonomy_hard_structural_{key}") for r in rs])
            cells.append(f"{agg['mean']:.3f}" if _finite(agg["mean"])
                         else "n/a")
        print(f"{str(label):>23} " + " ".join(
            f"{cell:>12}" for cell in cells))


def _announce(hyp_ids):
    for hid in hyp_ids:
        h = HYPOTHESES[hid]
        print(f"[hypothesis {hid}] {h['question']}")
        print(f"    prediction: {h['prediction']}")
        print(f"    decision:   {h['decision']}")
        sesoi = h.get("sesoi") or {}
        equiv = h.get("sesoi_test") or {}
        directional = {k: v for k, v in sorted(sesoi.items())
                       if k not in equiv}
        direction = h.get("direction") or {}
        if directional:
            terms = []
            for k, v in directional.items():
                sign = direction.get(k)
                if sign == "+":
                    terms.append(f"{k} >= +{v:g}")
                elif sign == "-":
                    terms.append(f"{k} <= -{v:g}")
                else:
                    terms.append(f"|{k}| >= {v:g} (undirected)")
            print(f"    SESOI (smallest effect size of interest): "
                  f"{', '.join(terms)} - statistical significance below "
                  f"this is reported as practically negligible, not as "
                  f"support; a significant effect opposite to a registered "
                  f"direction is a refutation, whatever its size")
        for k in sorted(equiv):
            if k not in sesoi:
                continue
            if equiv[k] == "noninferior":
                print(f"    non-inferiority bound (one-sided null "
                      f"prediction, bootstrap): {k} >= -{sesoi[k]:g} - "
                      f"supported only when the 95% CI's lower bound lies "
                      f"above the negative bound; a CI entirely below it is "
                      f"a refutation, not an inconclusive result")
            else:
                print(f"    equivalence bound (null prediction, bootstrap "
                      f"TOST): |{k}| <= {sesoi[k]:g} - supported only when "
                      f"the 95% CI of the effect lies inside this bound; a "
                      f"CI that merely includes 0 is not enough")


def _run_event(settings: Settings, experiment: str, event: str, **details) -> None:
    """Append one lifecycle row to ``run_index.jsonl``.

    A results directory may accumulate many invocations (baselines, sanity,
    E1..E3, retries).  The index records started/completed/aborted/failed
    per invocation with the same ``run_id`` as its manifest and records, so
    a half-finished grid can be told apart from a complete one; metadata
    write failures are warnings rather than experiment failures.
    """
    try:
        try:
            ctx = _run_context(settings)
        except Exception as e:  # noqa: BLE001 - keep the failure event
            ctx = _fallback_context(e)
        record = {
            "run_id": ctx["run_id"],
            "schema_version": ctx["schema_version"],
            "run_started_utc": ctx["run_started_utc"],
            "experiment": experiment,
            "event": event,
            "git_sha": ctx["git_sha"],
            "git_dirty": ctx["git_dirty"],
            "source_tree_sha256": ctx["source_tree_sha256"],
            "setup_sha256": ctx["setup_sha256"],
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **details,
        }
        path = pathlib.Path(settings.out_dir) / "run_index.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        _assert_append_target_intact(path)
        # Append directly rather than through ``_write_result``: a fallback
        # context has no setup hash, and this lifecycle row must still be
        # written when the environment probe itself failed.
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False,
                                sort_keys=True) + "\n")
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:  # noqa: PERF203 - durability is best-effort
                pass
        _register_artifact(path)
    except Exception as e:  # noqa: BLE001 - tracking must not kill a run
        print(f"[runner] WARNING: could not write run_index event "
              f"{event}: {e}")


_RECORD_PROVENANCE_KEYS = (
    "run_id", "schema_version", "setup_sha256", "git_sha",
    "source_tree_sha256",
)


class TrackingError(RuntimeError):
    """A result row cannot be joined to its run manifest / reproducible setup."""


def _validate_result_record(record) -> None:
    """Refuse to persist a row without the provenance needed to trace it.

    Every result row must be joinable to the run manifest that describes the
    code revision, resolved config, seeds and data fingerprints behind it.
    The runners pass ``**_run_context`` (full rows) or ``**_stamp`` (auxiliary
    error-case rows); a future call site that forgets either would otherwise
    append an untraceable number to the results file, silently violating the
    "every run records code/config/seed/data/curve/result" contract.  The
    check is deliberately limited to the join keys, so auxiliary rows with
    different payloads remain valid.
    """
    if not isinstance(record, dict):
        raise TrackingError(
            f"result record must be a dict, got {type(record).__name__}")
    missing = [key for key in _RECORD_PROVENANCE_KEYS
               if record.get(key) in (None, "")]
    if missing:
        raise TrackingError(
            f"result record is missing provenance {missing}; stamp it with "
            f"**_run_context(settings) or **_stamp(settings) so it can be "
            f"joined to its manifest")
    try:
        json.dumps(record, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise TrackingError(
            f"result record is not JSON-serializable: {exc}") from exc


def _write_result(path: pathlib.Path, record: dict) -> None:
    """Append one JSONL run record, stamped with its creation time.

    The timestamp is injected here rather than by each runner, so every
    record on disk carries "when" next to the ``git_sha`` / ``setup_sha256``
    provenance it already stores inline.  The write is refused (instead of
    appending an orphan row) when the provenance join keys are absent.

    An append to a file that was already certified by an earlier
    ``artifact_index.json`` is refused when its current bytes no longer
    match that certificate.  Without this guard the next index write would
    simply hash the tampered file and launder it into a fresh "verified"
    entry, so a single edited row could silently re-enter the cumulative
    chain of custody.
    """
    _validate_result_record(record)
    record.setdefault("created_utc",
                      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_append_target_intact(path)
    existed = path.exists()
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:  # noqa: PERF203 - durability is best-effort
            pass
    if not existed:
        # A durable file is not durable until its directory entry is too:
        # otherwise a crash right after the first append can lose the file
        # even though the row itself was fsynced.
        _fsync_directory(path.parent)
    _register_artifact(path)


def _fsync_directory(path: pathlib.Path) -> None:
    """Best-effort fsync of a directory entry (POSIX; no-op elsewhere).

    ``os.replace`` and the first append update a directory entry, and the
    file-level ``fsync`` does not flush that entry on every filesystem.
    Failing to do this is not fatal for a run (the evidence may still be
    present), so every error is swallowed after the probe.
    """
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - some filesystems reject dir fsync
        pass
    finally:
        try:
            os.close(fd)
        except OSError:  # pragma: no cover - nothing useful to retry
            pass


def _atomic_write_text(path: pathlib.Path, text: str) -> None:
    """Write ``text`` to ``path`` without exposing a half-written file.

    Result and index files are evidence: a crash or full disk in the middle
    of a plain ``write_text`` leaves a truncated JSON document that looks
    like a corrupted run.  Writing a sibling temporary file and renaming it
    over the target makes each artifact appear atomically.  The containing
    directory is fsynced after the rename, because a rename is only durable
    once its directory entry is.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:  # noqa: PERF203 - durability is best-effort
                pass
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def _write_json(path: pathlib.Path, payload) -> None:
    _atomic_write_text(
        path, json.dumps(payload, ensure_ascii=False, indent=2,
                         sort_keys=True))
    _register_artifact(path)


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def _git_sha() -> str:
    """Commit hash of the repository that contains this package.

    The probe is anchored at the package root instead of the process CWD:
    a run launched from another checkout (or from a subdirectory that is
    itself inside a different repository) would otherwise stamp a
    ``git_sha`` that does not identify the code that produced the record.
    """
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"],
                              cwd=str(_REPO_ROOT),
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    """Whether the working tree differs from the recorded revision.

    A result produced from a dirty tree cannot be rebuilt from its
    ``git_sha`` alone, so every run record says so explicitly instead of
    silently attributing the numbers to the committed revision.  The
    status is read from the package's repository root for the same reason
    as :func:`_git_sha`; a non-repository location fails closed (dirty).
    """
    try:
        out = subprocess.run(["git", "status", "--porcelain"],
                             cwd=str(_REPO_ROOT),
                             capture_output=True, text=True, check=True).stdout
        return bool(out.strip())
    except Exception:
        return True


_SOURCE_TREE_SHA: str | None = None
_RUN_ID: str | None = None
_RUN_STARTED_UTC: str | None = None
_RECORD_SCHEMA = 1
# Dedicated exit code for "the existing results directory already failed its
# artifact certificate"; distinct from the lock conflict (4) and the
# pre-grid scientific gates (3) so a caller can tell the failure modes apart.
ARTIFACT_INTEGRITY_EXIT_CODE = 5
_RUN_ARTIFACTS: set[pathlib.Path] = set()


def _register_artifact(path) -> None:
    """Remember a file written by this invocation for the artifact index."""
    try:
        _RUN_ARTIFACTS.add(pathlib.Path(path).resolve())
    except Exception:  # noqa: BLE001 - tracking must not kill a run
        pass


def _fallback_context(error: Exception) -> dict:
    """Minimal provenance when the full context itself failed.

    Setup problems (e.g. asking for CUDA on a CPU-only build) can make
    ``_run_context`` raise before any run event is written.  Recording a
    minimal block keeps the failure traceable instead of losing it just
    because the environment probe failed too.
    """
    run_id, run_started_utc = _run_identity()
    return {
        "run_id": run_id,
        "schema_version": _RECORD_SCHEMA,
        "run_started_utc": run_started_utc,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "source_tree_sha256": _source_tree_sha256(),
        "setup_sha256": None,
        "context_error": f"{type(error).__name__}: {error}",
    }


def _read_artifact_entries(index_path: pathlib.Path) -> tuple[dict, str | None]:
    """Read the entries of an existing index, or return no entries + why.

    A malformed index is not silently treated as empty: the new index keeps
    the ``previous_index_error`` so a reader knows that artifacts from
    earlier invocations are no longer covered.  The previous list is
    dropped as a whole rather than partially trusted - a partially readable
    integrity list is not a certificate.
    """
    if not index_path.exists():
        return {}, None
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("index must be a JSON object")
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError("artifacts must be a list")
        entries = {}
        for entry in artifacts:
            if not isinstance(entry, dict):
                raise ValueError("every artifact entry must be an object")
            name = entry.get("file")
            if not isinstance(name, str) or not name:
                raise ValueError("every artifact entry needs a non-empty file")
            if name in entries:
                # A duplicated entry is not a partially trustworthy list:
                # silently keeping the last copy would let the next index
                # write launder a state that ``_verify_artifacts`` rejects.
                raise ValueError(
                    f"duplicate artifact entry for file {name!r}")
            # A structurally valid entry with an unusable byte count or
            # digest is still a malformed integrity record.  Keeping it
            # would write a new index that ``_verify_artifacts`` rejects,
            # while the merge loop below would pretend the earlier history
            # was preserved; drop the whole previous list and report it.
            size = entry.get("bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError(
                    f"artifact entry {name!r} has an invalid byte count")
            digest = entry.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64 \
                    or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError(
                    f"artifact entry {name!r} has an invalid sha256")
            if name.endswith(".jsonl"):
                records = entry.get("records")
                if isinstance(records, bool) or not isinstance(records, int) \
                        or records < 0:
                    raise ValueError(
                        f"artifact entry {name!r} has an invalid line count")
            entries[name] = entry
        return entries, None
    except (OSError, ValueError, TypeError, UnicodeError) as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def _find_artifact_root(path: pathlib.Path) -> pathlib.Path | None:
    """Nearest ancestor directory that carries an ``artifact_index.json``."""
    for candidate in (path.parent, *path.parents):
        if (candidate / "artifact_index.json").is_file():
            return candidate
    return None


def _jsonl_tail_problem(path: pathlib.Path) -> str | None:
    """Describe a torn append boundary, or return ``None``.

    Every record this runner appends is one JSON serialisation followed by
    ``\\n``.  A crash between the data write and the newline can therefore
    leave a file whose final line is a fragment.  Appending after that
    fragment would splice the next record onto it, producing a permanently
    unparseable JSONL file that still ends in a newline.  Checking the last
    byte is sufficient for this writer (a JSON string escapes embedded
    newlines), cheap for large logs, and does not reject legacy JSONL files
    whose older records are not JSON objects.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        return f"cannot stat it ({type(exc).__name__}: {exc})"
    if size == 0:
        return None
    try:
        with path.open("rb") as fh:
            fh.seek(size - 1)
            if fh.read(1) != b"\n":
                return "it does not end with a newline"
    except OSError as exc:
        return f"cannot read it ({type(exc).__name__}: {exc})"
    return None


def _assert_append_target_intact(path) -> None:
    """Refuse to append to a previously certified but changed file.

    ``_write_artifact_index`` hashes whatever is on disk when an invocation
    ends.  Without this pre-append check, appending to a JSONL file that an
    earlier invocation certified - after someone edited or truncated it -
    would let the next index write re-certify the modified history.  The
    check is intentionally limited to files that appear in an existing
    index; a brand-new or legacy untracked file is allowed through and will
    be covered by the index written at the end of the invocation.

    Independently of the certificate, the current tail must be a complete
    JSONL record.  A crash can leave a torn final line *before* any index
    is written, so an untracked file is not implicitly safe to append to.
    """
    path = pathlib.Path(path)
    if path.resolve() in _RUN_ARTIFACTS:
        # This process already wrote the file (and therefore knows its
        # current state); re-hashing a growing JSONL before every row would
        # make long grids quadratic.
        return
    if path.suffix == ".jsonl" and path.exists():
        problem = _jsonl_tail_problem(path)
        if problem is not None:
            raise TrackingError(
                f"refusing to append to {path}: {problem}; the previous "
                f"writer may have crashed mid-record, and appending would "
                f"splice a new row onto a torn line. Restore the file or "
                f"rebuild the results directory deliberately")
    root = _find_artifact_root(path)
    if root is None:
        return
    entries, error = _read_artifact_entries(root / "artifact_index.json")
    if error is not None:
        raise TrackingError(
            f"refusing to append to {path}: the previous artifact index in "
            f"{root} is unreadable ({error}); the cumulative chain of "
            f"custody cannot be checked")
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return
    entry = entries.get(relative)
    if entry is None:
        return
    if not path.exists():
        raise TrackingError(
            f"refusing to recreate {path}: the artifact index certifies it "
            f"({entry['bytes']} bytes, {entry['sha256'][:12]}) but the file "
            f"is missing; restore it or rebuild the results directory "
            f"deliberately instead of letting a new run overwrite the "
            f"missing history")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise TrackingError(
            f"refusing to append to {path}: cannot read it to check the "
            f"previous certificate ({exc})") from exc
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != entry["bytes"] or digest != entry["sha256"]:
        raise TrackingError(
            f"refusing to append to {path}: its current size/hash "
            f"({len(data)} bytes, {digest[:12]}) does not match the "
            f"certified entry ({entry['bytes']} bytes, "
            f"{entry['sha256'][:12]}); re-run ``--verify-artifacts`` and "
            f"restore the file instead of extending a tampered history")


def _write_artifact_index(out_dir: pathlib.Path, settings: Settings,
                          experiment: str, completed: bool,
                          elapsed_seconds: float) -> dict:
    """Hash every artifact written so far into one cumulative index file.

    A results directory is append-only across invocations and can be
    truncated or edited after the fact.  The index records the size, line
    count and SHA-256 of each file written by *this* run **and keeps the
    entries of earlier runs that this invocation did not rewrite**, so
    ``--verify-artifacts`` certifies the whole directory rather than only
    its last invocation.  A rewritten/append-appended file replaces its own
    entry (the current hash always wins).  The index itself is written last
    and is not listed.

    Entries are stored with paths relative to ``out_dir`` so artifacts in
    nested directories are covered too.  A previous index that cannot be
    parsed is reported in ``previous_index_error`` instead of being
    mistaken for an empty directory.
    """
    root = pathlib.Path(out_dir).resolve()
    previous, previous_error = _read_artifact_entries(
        root / "artifact_index.json")
    current = {}
    for path in sorted(_RUN_ARTIFACTS):
        try:
            relative = path.resolve().relative_to(root)
        except (OSError, RuntimeError, ValueError):
            continue
        if not path.is_file():
            continue
        data = path.read_bytes()
        entry = {
            "file": relative.as_posix(),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        if path.suffix == ".jsonl":
            entry["records"] = data.count(b"\n")
        current[entry["file"]] = entry
    merged = {**previous, **current}
    entries = [merged[name] for name in sorted(merged)]
    try:
        context = _run_context(settings)
    except Exception as e:  # noqa: BLE001 - an index must still be written
        context = _fallback_context(e)
    payload = {
        "experiment": experiment,
        "completed": bool(completed),
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "artifact_count": len(entries),
        "current_artifact_count": len(current),
        "updated_entries": sorted(current),
        "artifacts": entries,
    }
    if previous_error is not None:
        payload["previous_index_error"] = previous_error
    payload.update(context)
    _atomic_write_text(
        root / "artifact_index.json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return payload


def _verify_artifacts(out_dir) -> int:
    """Check a results directory against its cumulative artifact index.

    Exit 0 when every listed file exists with the recorded size and hash,
    1 when any file is missing/changed or an entry is malformed, 2 when no
    usable index exists.  Because the index is cumulative, a file from an
    earlier invocation is checked too; a missing entry therefore means the
    results directory was truncated after that invocation.
    """
    root = pathlib.Path(out_dir).resolve()
    index_path = root / "artifact_index.json"
    if not index_path.exists():
        print(f"[verify] no artifact index at {index_path}; nothing to check")
        return 2
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("index must be a JSON object")
        artifacts = payload["artifacts"]
        if not isinstance(artifacts, list):
            raise ValueError("artifacts must be a list")
    except (OSError, ValueError, KeyError, TypeError, UnicodeError) as e:
        print(f"[verify] unreadable artifact index {index_path}: {e}")
        return 2
    # A previous index that could not be read means the cumulative history
    # was dropped when this index was rewritten.  The surviving list can
    # still be verified, but it no longer certifies the whole results
    # directory, so failing closed is the only honest verdict.
    if payload.get("previous_index_error") is not None:
        print(f"[verify] FAILED: artifact history is incomplete: "
              f"{payload.get('previous_index_error')}")
        return 1
    failures = []
    seen = set()
    for entry in artifacts:
        if not isinstance(entry, dict):
            failures.append(f"<malformed entry>: not an object")
            continue
        name = entry.get("file")
        if not isinstance(name, str) or not name:
            failures.append("<malformed entry>: missing non-empty file")
            continue
        if name in seen:
            failures.append(f"{name}: duplicate entry in index")
            continue
        seen.add(name)
        try:
            path = (root / name).resolve()
            path.relative_to(root)
        except (OSError, RuntimeError, ValueError):
            failures.append(f"{name}: outside results directory")
            continue
        if not path.is_file():
            failures.append(f"{name}: missing")
            continue
        if isinstance(entry.get("bytes"), bool) \
                or not isinstance(entry.get("bytes"), int) \
                or entry.get("bytes") < 0:
            failures.append(f"{name}: invalid byte count")
            continue
        digest_expected = entry.get("sha256")
        if not isinstance(digest_expected, str) \
                or len(digest_expected) != 64 \
                or any(c not in "0123456789abcdef" for c in digest_expected):
            failures.append(f"{name}: invalid sha256")
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            failures.append(f"{name}: unreadable ({exc})")
            continue
        if len(data) != entry["bytes"]:
            failures.append(f"{name}: size {len(data)} != "
                            f"{entry['bytes']}")
        digest = hashlib.sha256(data).hexdigest()
        if digest != digest_expected:
            failures.append(f"{name}: sha256 {digest[:12]} != "
                            f"{digest_expected[:12]}")
        lines = data.count(b"\n")
        if path.suffix == ".jsonl":
            expected_lines = entry.get("records")
            if isinstance(expected_lines, bool) \
                    or not isinstance(expected_lines, int) \
                    or expected_lines < 0:
                failures.append(f"{name}: invalid line count")
            elif lines != expected_lines:
                failures.append(f"{name}: line count {lines} != "
                                f"{expected_lines}")
    if failures:
        print(f"[verify] FAILED ({len(failures)} issue(s)) for run "
              f"{payload.get('run_id')} in {root}:")
        for failure in failures:
            print(f"    {failure}")
        return 1
    print(f"[verify] ok: {len(artifacts)} artifact(s) match run "
          f"{payload.get('run_id')} (completed={payload.get('completed')}, "
          f"schema={payload.get('schema_version')}, "
          f"current={payload.get('current_artifact_count', 'n/a')})")
    return 0


_TERMINAL_RUN_EVENTS = frozenset({"completed", "aborted", "failed"})


def _read_jsonl_objects(path: pathlib.Path):
    """Read ``path`` as JSONL objects, pairing each row with its line number."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (TypeError, ValueError) as exc:
                rows.append((line_number, None,
                             f"invalid JSON ({type(exc).__name__}: {exc})"))
                continue
            if not isinstance(row, Mapping):
                rows.append((line_number, None,
                             f"row is not a JSON object "
                             f"({type(row).__name__})"))
                continue
            rows.append((line_number, row, None))
    return rows


def _verify_runs(out_dir) -> int:
    """Check that a results directory holds only completed, homogeneous runs.

    ``--verify-artifacts`` proves the bytes have not changed since they were
    certified, and ``--verify-provenance`` proves each row is internally
    joinable to its setup.  Neither answers the evidence question: was every
    invocation in this directory actually *completed*, and was the whole
    directory produced under one critical software/numerical environment?

    ``run_index.jsonl`` records the started/terminal lifecycle of each
    invocation and ``environment_history.jsonl`` records every drift verdict
    against the previous environment.  This audit reads both:

    * every ``started`` run must have exactly one terminal event, and only a
      ``completed`` terminal event can be cited as evidence; an ``aborted``
      or ``failed`` invocation is reported as a defect rather than hidden
      behind a passing artifact hash;
    * a critical (software/numerical) environment drift in the history makes
      the directory non-homogeneous, because its rows were not all produced
      by one numerical recipe; advisory hardware drift is reported as a
      warning;
    * a malformed or truncated lifecycle/history line fails closed;
    * when ``artifact_index.json`` exists, its last run must also be marked
      ``completed``.

    Exit codes: 0 = every run completed under a compatible environment,
    1 = incomplete/non-homogeneous evidence, 2 = no results directory or no
    usable run lifecycle log.  The check is read-only.
    """
    root = pathlib.Path(out_dir)
    if not root.is_dir():
        print(f"[runs] no results directory at {root}; nothing to check")
        return 2
    run_log = root / "run_index.jsonl"
    if not run_log.is_file():
        print(f"[runs] no run lifecycle log at {run_log}; nothing to check")
        return 2
    try:
        rows = _read_jsonl_objects(run_log)
    except (OSError, UnicodeError) as exc:
        print(f"[runs] cannot read {run_log}: {exc}")
        return 2
    if not rows:
        print(f"[runs] run lifecycle log {run_log} is empty; nothing to check")
        return 2

    problems: list[str] = []
    warnings: list[str] = []
    events_by_run: dict[str, list[tuple[int, str, Mapping]]] = {}
    for line_number, row, error in rows:
        if error is not None:
            problems.append(f"run_index.jsonl:{line_number}: {error}")
            continue
        run_id = row.get("run_id")
        event = row.get("event")
        if not isinstance(run_id, str) or not run_id:
            problems.append(
                f"run_index.jsonl:{line_number}: run_id must be a non-empty "
                f"string")
            continue
        if not isinstance(event, str) or not event:
            problems.append(
                f"run_index.jsonl:{line_number}: event must be a non-empty "
                f"string")
            continue
        events_by_run.setdefault(run_id, []).append(
            (line_number, event, row))

    completed_runs = set()
    for run_id in sorted(events_by_run):
        events = events_by_run[run_id]
        started = [(line, row) for line, event, row in events
                   if event == "started"]
        terminals = [(line, event, row) for line, event, row in events
                     if event in _TERMINAL_RUN_EVENTS]
        if not started:
            problems.append(
                f"run {run_id}: terminal event without a started event")
        elif len(started) > 1:
            problems.append(
                f"run {run_id}: {len(started)} started events "
                f"(first at run_index.jsonl:{started[0][0]})")
        if not terminals:
            first = started[0][0] if started else events[0][0]
            problems.append(
                f"run {run_id}: started at run_index.jsonl:{first} but has "
                f"no completed/aborted/failed terminal event")
            continue
        if started and terminals[0][0] < started[0][0]:
            problems.append(
                f"run {run_id}: terminal event at "
                f"run_index.jsonl:{terminals[0][0]} precedes its started "
                f"event at run_index.jsonl:{started[0][0]}")
        if len(terminals) > 1:
            problems.append(
                f"run {run_id}: {len(terminals)} terminal events "
                f"(first at run_index.jsonl:{terminals[0][0]})")
        for line_number, event, row in terminals:
            if event == "completed":
                completed_runs.add(run_id)
                continue
            reason = row.get("reason") or row.get("error")
            detail = f": {reason}" if reason else ""
            problems.append(
                f"run {run_id}: {event} at run_index.jsonl:{line_number}"
                f"{detail}; a non-completed invocation cannot be cited as "
                f"evidence")

    if not completed_runs:
        problems.append("run_index.jsonl contains no completed run")

    history_path = root / "environment_history.jsonl"
    history_rows = 0
    if history_path.is_file():
        try:
            history = _read_jsonl_objects(history_path)
        except (OSError, UnicodeError) as exc:
            problems.append(f"cannot read environment_history.jsonl: {exc}")
            history = []
        for line_number, row, error in history:
            where = f"environment_history.jsonl:{line_number}"
            if error is not None:
                problems.append(f"{where}: {error}")
                continue
            history_rows += 1
            run_id = row.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                problems.append(
                    f"{where}: run_id must be a non-empty string")
            if "compatible" not in row:
                problems.append(
                    f"{where}: missing compatible verdict; a history row "
                    f"must state true, false or null explicitly")
                continue
            compatible = row["compatible"]
            if compatible is not None and not isinstance(compatible, bool):
                problems.append(
                    f"{where}: compatible must be true, false or null")
                continue
            differences = row.get("differences")
            if not isinstance(differences, list):
                problems.append(
                    f"{where}: differences must be a list")
                continue
            critical = [difference for difference in differences
                        if isinstance(difference, Mapping)
                        and difference.get("severity") == "critical"]
            malformed = [difference for difference in differences
                         if not isinstance(difference, Mapping)
                         or difference.get("severity") not in
                         ("critical", "advisory")]
            if malformed:
                problems.append(
                    f"{where}: every difference must be an object with "
                    f"severity critical or advisory")
                continue
            if compatible is False:
                reason = row.get("reason") or "critical environment drift"
                problems.append(
                    f"{where}: run {run_id} was produced under a "
                    f"non-compatible environment ({reason})")
            elif compatible is None and differences:
                problems.append(
                    f"{where}: a first-run null verdict cannot carry "
                    f"environment differences")
            elif compatible is True and critical:
                fields = ", ".join(
                    str(difference.get("field")) for difference in critical)
                problems.append(
                    f"{where}: compatible=true contradicts recorded critical "
                    f"difference(s) ({fields})")
            elif compatible is True and differences:
                warnings.append(
                    f"{where}: advisory environment drift for run {run_id} "
                    f"({row.get('reason') or 'differences recorded'})")
    else:
        warnings.append(
            "environment_history.jsonl is missing; the directory predates "
            "the environment-compatibility contract")

    index_path = root / "artifact_index.json"
    if index_path.is_file():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            problems.append(
                f"artifact_index.json: unreadable "
                f"({type(exc).__name__}: {exc})")
        else:
            if not isinstance(payload, Mapping):
                problems.append("artifact_index.json: not a JSON object")
            elif payload.get("completed") is not True:
                problems.append(
                    f"artifact_index.json: last run "
                    f"{payload.get('run_id')} is marked "
                    f"completed={payload.get('completed')!r}")
            else:
                index_run = payload.get("run_id")
                if "run_id" in payload and (
                        not isinstance(index_run, str) or not index_run):
                    problems.append(
                        "artifact_index.json: run_id must be a non-empty "
                        "string when present")
                elif isinstance(index_run, str) and index_run \
                        and index_run not in completed_runs:
                    problems.append(
                        f"artifact_index.json: last run {index_run} has no "
                        f"completed lifecycle event in run_index.jsonl")
    else:
        warnings.append(
            "artifact_index.json is missing; artifact integrity is not "
            "certified")

    if problems:
        print(f"[runs] FAILED ({len(problems)} issue(s)) in {root}:")
        for problem in problems:
            print(f"    {problem}")
        for warning in warnings:
            print(f"    warning: {warning}")
        return 1
    print(f"[runs] ok: {len(completed_runs)} completed run(s) cite a "
          f"compatible environment ({history_rows} history row(s) checked)")
    for warning in warnings:
        print(f"[runs] warning: {warning}")
    return 0


# ---------------------------------------------------------------------------
# Record-level provenance audit
# ---------------------------------------------------------------------------
# The artifact index proves that a file's bytes have not changed since it was
# certified; it does not prove that the *records inside* are joinable to the
# setup and environment they claim.  ``--verify-provenance`` closes that gap
# by recomputing the two content ids from each record's own payload, checking
# that one ``run_id`` never names two different setups, and reporting result
# files that pool rows from more than one protocol/environment.

_RESULT_RECORD_SUFFIXES = ("_runs.jsonl", "_error_cases.jsonl")

_PROVENANCE_REQUIRED_KEYS = (
    "run_id", "schema_version", "setup_sha256", "git_sha",
    "source_tree_sha256",
)

# The exact inputs ``_run_context`` hashes into ``setup_sha256``.  A full
# record carries all of them, so its declared id can be recomputed; a compact
# ``_stamp`` row carries only the resulting id and is checked against the
# lifecycle/manifest anchor for its ``run_id`` instead.
_SETUP_HASH_INPUT_KEYS = (
    "git_sha", "source_tree_sha256", "hypotheses_sha256", "config",
    "data_fingerprints",
)


def _is_short_sha(value) -> bool:
    """Whether ``value`` is one of the repository's 16-hex content ids."""
    return (isinstance(value, str) and len(value) == 16
            and all(ch in "0123456789abcdef" for ch in value))


def _setup_fingerprint(record) -> str | None:
    """Recompute ``setup_sha256`` from a record's own stored inputs.

    Returns ``None`` when the record is a compact stamp (no ``config`` /
    ``data_fingerprints`` / ``hypotheses_sha256``), which cannot be
    self-verified but can still be matched to its run anchor.  The payload is
    built exactly as :func:`_run_context` builds it, so a record produced by
    this code must reproduce its declared id.
    """
    if not isinstance(record, Mapping):
        return None
    if not all(key in record for key in _SETUP_HASH_INPUT_KEYS):
        return None
    payload = {key: record[key] for key in _SETUP_HASH_INPUT_KEYS}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _provenance_record_errors(record) -> list[str]:
    """Structural and self-consistency defects in one result row."""
    if not isinstance(record, Mapping):
        return [f"record is not a JSON object ({type(record).__name__})"]
    problems = []
    missing = [key for key in _PROVENANCE_REQUIRED_KEYS
               if key not in record]
    if missing:
        problems.append(f"missing provenance keys {missing}")
    run_id = record.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        problems.append("run_id must be a non-empty string")
    schema = record.get("schema_version")
    if isinstance(schema, bool) or not isinstance(schema, int):
        problems.append("schema_version must be an integer")
    elif schema != _RECORD_SCHEMA:
        problems.append(
            f"schema_version {schema} is not the current {_RECORD_SCHEMA}")
    setup = record.get("setup_sha256")
    if not _is_short_sha(setup):
        problems.append("setup_sha256 must be a 16-character lowercase hex id")
    git_sha = record.get("git_sha")
    if not isinstance(git_sha, str) or not git_sha:
        problems.append("git_sha must be a non-empty string")
    source = record.get("source_tree_sha256")
    if not _is_short_sha(source):
        problems.append(
            "source_tree_sha256 must be a 16-character lowercase hex id")
    present = [key for key in ("hypotheses_sha256", "config",
                               "data_fingerprints") if key in record]
    if present and len(present) != 3:
        problems.append(
            f"incomplete full-context block: has {present} but needs "
            f"hypotheses_sha256, config and data_fingerprints")
    elif present:
        # The full-context block is itself typed provenance: a string/list
        # where a mapping belongs cannot be silently hashed as if it were a
        # valid config or fingerprint table.  Fail closed before the setup
        # hash comparison so the defect is named explicitly.
        for key in ("config", "data_fingerprints"):
            if not isinstance(record[key], Mapping):
                problems.append(f"{key} must be a JSON object")
        if not _is_short_sha(record["hypotheses_sha256"]):
            problems.append(
                "hypotheses_sha256 must be a 16-character lowercase hex id")
        expected = _setup_fingerprint(record)
        if expected is not None and expected != setup:
            problems.append(
                f"setup_sha256 {setup} does not match the hash of this "
                f"record's own config/data/hypotheses ({expected})")
    environment = record.get("environment")
    if "environment" in record:
        if not isinstance(environment, Mapping):
            problems.append("environment must be a JSON object")
        else:
            declared = environment.get("environment_sha256")
            if not _is_short_sha(declared):
                problems.append(
                    "environment.environment_sha256 must be a 16-character "
                    "lowercase hex id")
            else:
                recomputed = _environment_fingerprint(environment)
                if recomputed != declared:
                    problems.append(
                        f"environment_sha256 {declared} does not match the "
                        f"hash of the environment block itself ({recomputed})")
    return problems


def _audit_provenance(out_dir, require_homogeneous: bool = False) -> int:
    """Verify that every result row can be joined to its declared setup.

    The checks are semantic, complementing :func:`_verify_artifacts`:

    * every row carries the record schema and provenance join keys;
    * a full row's ``setup_sha256`` is recomputed from its own
      code/config/data/hypotheses fields and must match;
    * a full row's environment block is re-hashed and must match its declared
      ``environment_sha256``;
    * one ``run_id`` may not name two different setups or environments
      (checked across all result files, manifests and the lifecycle index);
    * a compact stamp row must share the setup of the same ``run_id`` seen in
      the lifecycle index or a full row;
    * rows inside one result file are expected to describe one protocol and
      one environment.  Mixing them is reported as a warning by default and
      as an error with ``require_homogeneous=True``, because such a file
      cannot be pooled for analysis without splitting it first.

    Exit codes: 0 = every row traceable and internally consistent, 1 = a
    defect was found (or a mixed file under ``require_homogeneous``),
    2 = no result records to audit.
    """
    root = pathlib.Path(out_dir)
    if not root.is_dir():
        print(f"[provenance] no results directory at {root}")
        return 2

    errors: list[str] = []
    warnings: list[str] = []

    def _problem(message: str) -> None:
        if message not in errors:
            errors.append(message)

    def _warning(message: str) -> None:
        if message not in warnings:
            warnings.append(message)

    # ``run_id -> {(setup, git_sha, source_tree_sha256): first location}``
    # and the analogous environment map.  A run_id may legitimately appear in
    # many files, but never with two different provenance tuples.
    run_base: dict[str, dict[tuple, str]] = {}
    run_env: dict[str, dict[str, str]] = {}

    def _add_run(record, where: str) -> None:
        if not isinstance(record, Mapping):
            return
        run_id = record.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            return
        setup = record.get("setup_sha256")
        git_sha = record.get("git_sha")
        source = record.get("source_tree_sha256")
        if _is_short_sha(setup) and isinstance(git_sha, str) and git_sha \
                and isinstance(source, str) and source:
            run_base.setdefault(run_id, {}).setdefault(
                (setup, git_sha, source), where)
        environment = record.get("environment")
        if isinstance(environment, Mapping):
            declared = environment.get("environment_sha256")
            if _is_short_sha(declared):
                run_env.setdefault(run_id, {}).setdefault(declared, where)

    def _read_json_document(path: pathlib.Path, where: str):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            _problem(f"{where}: unreadable provenance document "
                     f"({type(exc).__name__}: {exc})")
            return None
        if not isinstance(payload, Mapping):
            _problem(f"{where}: provenance document is not a JSON object")
            return None
        return payload

    # ---- anchors: environment.json and every manifest -----------------
    environment_path = root / "environment.json"
    directory_environment = None
    anchor_paths = []
    if environment_path.is_file():
        anchor_paths.append(environment_path)
    anchor_paths.extend(sorted(root.glob("*_manifest.json")))
    for path in anchor_paths:
        relative = path.relative_to(root).as_posix()
        payload = _read_json_document(path, relative)
        if payload is None:
            continue
        environment, context = _extract_environment_record(payload)
        if environment is None:
            _problem(f"{relative}: has no environment block")
            continue
        declared = environment.get("environment_sha256")
        if not _is_short_sha(declared):
            _problem(f"{relative}: environment_sha256 is malformed")
        else:
            recomputed = _environment_fingerprint(environment)
            if recomputed != declared:
                _problem(
                    f"{relative}: environment_sha256 {declared} does not "
                    f"match the hash of its environment block ({recomputed})")
        if path is environment_path and _is_short_sha(declared):
            directory_environment = declared
        if isinstance(context, Mapping):
            _add_run(context, relative)

    # ---- lifecycle index: the anchor for compact error-case stamps ----
    run_index_path = root / "run_index.jsonl"
    if run_index_path.is_file():
        relative = run_index_path.name
        with run_index_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except (TypeError, ValueError) as exc:
                    _problem(f"{relative}:{line_number}: invalid JSON "
                             f"({type(exc).__name__}: {exc})")
                    continue
                if not isinstance(row, Mapping):
                    _problem(f"{relative}:{line_number}: row is not a "
                             f"JSON object")
                    continue
                # A failed environment probe writes a fallback context
                # with setup_sha256=None; that is an honest failure row,
                # not a provenance defect in an experiment result.
                _add_run(row, f"{relative}:{line_number}")

    # ---- result files --------------------------------------------------
    # Runs already anchored by environment.json, a manifest or the lifecycle
    # index.  Full result rows add themselves; compact stamps only point at
    # one of these.
    anchored_runs = set(run_base)
    stamp_rows: list[tuple[str, str]] = []
    result_files = sorted(
        path for path in root.rglob("*.jsonl")
        if path.is_file() and path.name.endswith(_RESULT_RECORD_SUFFIXES)
        and path != run_index_path)
    if not result_files:
        if errors:
            # A malformed anchor (environment/manifest/lifecycle index) is a
            # provenance defect even when there are no result rows to scan;
            # returning the "nothing to audit" code would hide it.
            print(f"[provenance] FAILED ({len(errors)} issue(s)):")
            for error in errors:
                print(f"    {error}")
            return 1
        print(f"[provenance] no result records under {root}; nothing to audit")
        return 2

    total_records = 0
    file_summaries = []
    for path in result_files:
        relative = path.relative_to(root).as_posix()
        file_setups: set[str] = set()
        file_environments: set[str] = set()
        n_records = 0
        try:
            handle = path.open(encoding="utf-8")
        except OSError as exc:
            _problem(f"{relative}: cannot read ({type(exc).__name__}: {exc})")
            continue
        with handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                n_records += 1
                try:
                    record = json.loads(line)
                except (TypeError, ValueError) as exc:
                    _problem(f"{relative}:{line_number}: invalid JSON "
                             f"({type(exc).__name__}: {exc})")
                    continue
                where = f"{relative}:{line_number}"
                for problem in _provenance_record_errors(record):
                    _problem(f"{where}: {problem}")
                if not isinstance(record, Mapping):
                    continue
                setup = record.get("setup_sha256")
                if _is_short_sha(setup):
                    file_setups.add(setup)
                environment = record.get("environment")
                if isinstance(environment, Mapping):
                    declared = environment.get("environment_sha256")
                    if _is_short_sha(declared):
                        file_environments.add(declared)
                _add_run(record, where)
                # A compact stamp has no full context of its own; it is only
                # traceable through the same run's anchor.  Defer the verdict
                # until every file has been read, so a full row in a later
                # file still counts as the anchor.
                run_id = record.get("run_id")
                if _setup_fingerprint(record) is None:
                    if isinstance(run_id, str) and run_id:
                        stamp_rows.append((where, run_id))
                elif isinstance(run_id, str) and run_id:
                    anchored_runs.add(run_id)
        total_records += n_records
        message = None
        if len(file_setups) > 1:
            message = (f"{relative} mixes {len(file_setups)} setups "
                       f"({', '.join(sorted(file_setups))})")
        if len(file_environments) > 1:
            extra = (f"{relative} mixes {len(file_environments)} "
                     f"environments ({', '.join(sorted(file_environments))})")
            message = f"{message}; {extra}" if message else extra
        if directory_environment is not None and file_environments and \
                file_environments != {directory_environment}:
            extra = (f"{relative} was not produced under the environment in "
                     f"environment.json ({', '.join(sorted(file_environments))} "
                     f"vs {directory_environment})")
            message = f"{message}; {extra}" if message else extra
        if message is not None:
            if require_homogeneous:
                _problem(message)
            else:
                _warning(message)
        file_summaries.append((relative, n_records, file_setups,
                               file_environments))

    for where, run_id in stamp_rows:
        if run_id not in anchored_runs:
            _warning(
                f"{where}: compact row for run {run_id} has no full-context "
                f"anchor in this directory")

    # ---- cross-file run consistency -----------------------------------
    for run_id, variants in sorted(run_base.items()):
        if len(variants) > 1:
            detail = "; ".join(
                f"{key[0]} at {where}"
                for key, where in sorted(variants.items(),
                                         key=lambda item: item[1]))
            _problem(f"run {run_id} is claimed by {len(variants)} different "
                     f"setups: {detail}")
    for run_id, environments in sorted(run_env.items()):
        if len(environments) > 1:
            detail = "; ".join(
                f"{sha} at {where}"
                for sha, where in sorted(environments.items(),
                                         key=lambda item: item[1]))
            _problem(f"run {run_id} is claimed by {len(environments)} "
                     f"different environments: {detail}")

    if total_records == 0:
        _problem("result files exist but contain no records")

    print(f"[provenance] scanned {len(result_files)} result file(s), "
          f"{total_records} record(s), {len(run_base)} run(s) under {root}")
    for relative, n_records, setups, environments in file_summaries:
        print(f"    {relative}: {n_records} record(s), "
              f"{len(setups)} setup(s), {len(environments)} environment(s)")
        for setup in sorted(setups):
            print(f"        setup {setup}")
        for sha in sorted(environments):
            print(f"        env   {sha}")
    for warning in warnings:
        print(f"[provenance] WARNING: {warning}")
    if errors:
        print(f"[provenance] FAILED ({len(errors)} issue(s)):")
        for error in errors:
            print(f"    {error}")
        return 1
    print("[provenance] PASS: every result row is traceable to one declared "
          "setup and environment")
    if warnings:
        print(f"[provenance] note: {len(warnings)} warning(s) above; use "
              f"--require-homogeneous to make mixed-protocol files an error")
    return 0
