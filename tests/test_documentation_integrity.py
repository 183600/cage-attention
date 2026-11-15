"""The documentation must describe the code that is actually here.

A reproducibility document that quotes a different dependency version, a
command-line flag that does not exist or a container recipe that installs
another build is worse than no document at all: it sends the reader to a
different numerical environment and calls the difference a reproduction.
Every drift class below has actually occurred in this repository, so the
checks are written as regressions rather than as style preferences:

* the guide quoted ``torch==2.3.0+cpu`` while CI pinned ``torch==2.14.1``;
* the protocol advertised ``--lr-sensitivity``, a flag the runner never
  defined (the analysis runs as part of E1 instead);
* the container recipe installed from a wheel index that does not publish
  the pinned version, so the documented cross-machine path could not build;
* a source file pointed at a results-analysis document that no longer
  exists.
"""

from __future__ import annotations

import ast
import pathlib
import re
import unittest

from cage.repro import declared_requirements
# Reproducibility: Set random seeds
import random
import numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except ImportError:
    pass

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _markdown_files() -> list[pathlib.Path]:
    files = sorted(ROOT.glob("*.md"))
    docs = ROOT / "docs"
    if docs.is_dir():
        files.extend(sorted(docs.glob("*.md")))
    return files


def _python_files() -> list[pathlib.Path]:
    files = []
    for path in sorted(ROOT.rglob("*.py")):
        if any(part in (".git", "__pycache__", ".venv", "venv", "results")
               for part in path.parts):
            continue
        files.append(path)
    return files


def _cli_flags() -> set[str]:
    """Every command-line flag the experiment CLI defines.

    Read from the source with ``ast`` rather than from a hard-coded list,
    so a flag that disappears from the CLI also disappears from the set of
    flags the documentation is allowed to mention.
    """
    source = (ROOT / "cage" / "experiments" / "_e3_cli.py").read_text(
        encoding="utf-8")
    flags: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        flag = node.args[0].value
        if not isinstance(flag, str) or not flag.startswith("--"):
            continue
        flags.add(flag)
        for keyword in node.keywords:
            if (keyword.arg == "action"
                    and isinstance(keyword.value, ast.Attribute)
                    and keyword.value.attr == "BooleanOptionalAction"):
                flags.add("--no-" + flag[2:])
    return flags


def _make_targets() -> set[str]:
    """Every target the Makefile defines."""
    targets: set[str] = set()
    for line in (ROOT / "Makefile").read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)\s*:", line)
        if match and match.group(1) != ".PHONY":
            targets.add(match.group(1))
    return targets


class MarkdownLinkTest(unittest.TestCase):
    def test_relative_links_resolve(self):
        broken = []
        for document in _markdown_files():
            text = document.read_text(encoding="utf-8")
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
                link = target.strip().split("#", 1)[0]
                if not link or link.startswith(("http://", "https://",
                                               "mailto:")):
                    continue
                if any(char in link for char in "<>{}*$"):
                    continue
                if not (document.parent / link).exists():
                    broken.append(f"{document.name}: {target}")
        self.assertEqual(broken, [], f"broken documentation links: {broken}")


class DependencyQuoteTest(unittest.TestCase):
    def test_quoted_versions_match_the_declaration(self):
        declared = declared_requirements(ROOT)
        allowed: dict[str, set[str]] = {}
        for requirements in declared.values():
            for name, specifier in requirements:
                if specifier:
                    allowed.setdefault(name, set()).add(specifier)
        problems = []
        for document in _markdown_files():
            text = document.read_text(encoding="utf-8")
            for name, specifiers in allowed.items():
                pattern = re.compile(
                    re.escape(name) + r"(==|>=|<=|!=|~=|<|>)\s*"
                    r"([0-9][0-9A-Za-z.+_-]*)")
                for _operator, version in pattern.findall(text):
                    quoted = f"{_operator}{version}"
                    if quoted not in specifiers:
                        problems.append(
                            f"{document.name}: quotes {name}{quoted}, "
                            f"declared: {sorted(specifiers)}")
        self.assertEqual(problems, [],
                         f"documentation quotes a different dependency: {problems}")


class CommandReferenceTest(unittest.TestCase):
    # Flags that belong to other tools but appear in the same code blocks:
    # pip, docker, git, the CI gate script and the standalone environment
    # capture script (which has its own --output/--compare).
    _FOREIGN_FLAGS = {
        "--extra-index-url", "--index-url", "--no-cache-dir", "--no-deps",
        "--rm", "--hard", "--syntax-only", "--require-hashes",
        "--build-arg", "--output", "--compare",
    }

    def test_documented_cli_flags_exist(self):
        flags = _cli_flags()
        unknown = []
        for document in _markdown_files():
            text = document.read_text(encoding="utf-8")
            for token in sorted(set(re.findall(r"--[a-z][a-z0-9-]*", text))):
                if token in flags or token in self._FOREIGN_FLAGS:
                    continue
                unknown.append(f"{document.name}: {token}")
        self.assertEqual(unknown, [],
                         f"documented flags that the CLI does not define: {unknown}")

    def test_documented_make_targets_exist(self):
        targets = _make_targets()
        unknown = []
        for document in _markdown_files():
            text = document.read_text(encoding="utf-8")
            # A make invocation is a command: it either starts a line or
            # sits in a code span.  Prose such as "make the model ..." is
            # not a command and must not be read as one.
            candidates = set(re.findall(r"(?m)^[ \t]*make\s+([a-z][a-z0-9-]*)",
                                        text))
            candidates |= set(re.findall(r"`make\s+([a-z][a-z0-9-]*)", text))
            for target in sorted(candidates):
                if target not in targets:
                    unknown.append(f"{document.name}: make {target}")
        self.assertEqual(unknown, [],
                         f"documented make targets that do not exist: {unknown}")

    def test_sources_do_not_point_at_missing_documents(self):
        missing = []
        for path in _python_files():
            text = path.read_text(encoding="utf-8")
            for reference in sorted(set(re.findall(r"docs/[\w.-]+\.md", text))):
                if not (ROOT / reference).exists():
                    missing.append(f"{path.name}: {reference}")
        self.assertEqual(missing, [],
                         f"source files pointing at missing documents: {missing}")


class ContainerRecipeTest(unittest.TestCase):
    def setUp(self):
        self.text = (ROOT / "Dockerfile.repro").read_text(encoding="utf-8")

    def test_the_recipe_installs_the_frozen_requirement_file(self):
        self.assertIn("requirements-ci.txt", self.text)

    def test_the_recipe_hardcodes_no_dependency_version(self):
        # A version in the image recipe is a second source of truth; it
        # silently diverged from the CI pin once already.
        self.assertIsNone(re.search(r"torch==\d", self.text))

    def test_the_editable_install_cannot_re_resolve_the_dependencies(self):
        # Without --no-deps, pip resolves torch>=2.0 again and may replace
        # the pinned build with a different one.
        self.assertIn("--no-deps", self.text)

    def test_the_wheel_index_is_a_build_argument(self):
        # The index has to be selectable: the pinned version is published
        # for some CUDA variants only.
        self.assertIn("TORCH_INDEX_URL", self.text)


if __name__ == "__main__":
    unittest.main()
