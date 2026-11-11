"""Declared-vs-installed dependency conformance.

The seeds fix *which* numbers a run draws; the dependency set fixes *which
library* produces them.  A run whose interpreter contradicts the frozen
requirements is not a reproduction of the declared recipe, even when every
seed matches, so the disagreement has to be visible - not discovered months
later when two result directories refuse to be pooled.

These tests cover the parsing, the version comparison and the two ways the
verdict is produced: for the running interpreter and for an archived
environment record.
"""

from __future__ import annotations

import contextlib
import io
import pathlib
import tempfile
import unittest
from unittest import mock

from cage import experiments
from cage.repro import (
    check_dependency_conformance,
    declared_requirements,
    dependency_conformance_from_record,
    evaluate_requirement,
    installed_distribution_versions,
    parse_requirement_specifiers,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent


class RequirementParsingTest(unittest.TestCase):
    def test_comments_options_and_blanks_are_skipped(self):
        text = (
            "# CI-only dependency set (CPU, Python 3.12).\n"
            "\n"
            "--extra-index-url https://download.pytorch.org/whl/cpu\n"
            "-r other-requirements.txt\n"
            "torch==2.14.1  # the frozen build\n"
            "numpy>=1.26\n"
        )
        self.assertEqual(parse_requirement_specifiers(text),
                         [("torch", "==2.14.1"), ("numpy", ">=1.26")])

    def test_names_are_normalised_and_extras_are_dropped(self):
        text = "Foo.Bar[extra]==1.2.3\nfoo_bar==1.2.3\n"
        self.assertEqual(parse_requirement_specifiers(text),
                         [("foo-bar", "==1.2.3"), ("foo-bar", "==1.2.3")])

    def test_a_requirement_without_a_version_is_kept(self):
        self.assertEqual(parse_requirement_specifiers("torch\n"),
                         [("torch", "")])

    def test_the_checkout_declares_its_frozen_dependency_set(self):
        declared = declared_requirements(ROOT)
        self.assertIn("requirements-ci.txt", declared)
        self.assertIn(("torch", "==2.14.1"),
                      declared["requirements-ci.txt"])
        self.assertIn(("torch", ">=2.0"), declared["pyproject.toml"])


class VersionEvaluationTest(unittest.TestCase):
    def test_a_local_version_label_still_satisfies_the_plain_pin(self):
        entry = evaluate_requirement("torch", "==2.14.1", "2.14.1+cpu")
        self.assertEqual(entry["status"], "satisfied")

    def test_a_different_release_does_not_satisfy_the_pin(self):
        entry = evaluate_requirement("torch", "==2.14.1", "2.15.0")
        self.assertEqual(entry["status"], "unsatisfied")

    def test_zero_padding_does_not_change_the_release(self):
        entry = evaluate_requirement("torch", "==2.14.1", "2.14.1.0")
        self.assertEqual(entry["status"], "satisfied")

    def test_a_prerelease_is_not_certified_by_its_numeric_prefix(self):
        entry = evaluate_requirement("torch", ">=2.14.1", "2.14.1rc1")
        self.assertEqual(entry["status"], "unchecked")

    def test_an_unsupported_specifier_is_reported_not_certified(self):
        entry = evaluate_requirement("torch", "~=2.14.0", "2.14.1")
        self.assertEqual(entry["status"], "unchecked")

    def test_a_missing_distribution_is_reported(self):
        entry = evaluate_requirement("torch", "==2.14.1", None)
        self.assertEqual(entry["status"], "missing")

    def test_ordering_comparisons(self):
        self.assertEqual(
            evaluate_requirement("torch", ">=2.0", "2.14.1")["status"],
            "satisfied")
        self.assertEqual(
            evaluate_requirement("torch", "<2.0", "2.14.1")["status"],
            "unsatisfied")
        self.assertEqual(
            evaluate_requirement("torch", "!=2.14.1", "2.14.1")["status"],
            "unsatisfied")


class DuplicateDistributionOrderTest(unittest.TestCase):
    """A distribution reported twice must resolve order-independently."""

    class _Dist:
        def __init__(self, name, version):
            self.metadata = {"Name": name}
            self.version = version

    def _install(self, entries):
        dists = [self._Dist(name, version) for name, version in entries]
        return mock.patch("cage.repro.importlib_metadata.distributions",
                          return_value=iter(dists))

    def test_scan_order_does_not_change_the_kept_version(self):
        # Lexicographic order would keep 2.9.0 over 2.14.1; the semantic
        # release order must keep the higher one regardless of scan order.
        for order in ([("torch", "2.9.0"), ("torch", "2.14.1")],
                      [("torch", "2.14.1"), ("torch", "2.9.0")]):
            with self.subTest(order=order), self._install(order):
                self.assertEqual(
                    installed_distribution_versions()["torch"], "2.14.1")

    def test_different_name_spellings_collapse_to_one_entry(self):
        with self._install([("Foo.Bar", "1.0"), ("foo_bar", "2.0")]):
            versions = installed_distribution_versions()
            self.assertEqual(versions["foo-bar"], "2.0")


class ConformanceReportTest(unittest.TestCase):
    def test_the_running_environment_conforms_to_the_declaration(self):
        report = check_dependency_conformance(ROOT)
        self.assertTrue(report["conforms"], report)
        self.assertEqual(report["unsatisfied"], [])
        self.assertEqual(report["missing"], [])

    def test_the_installed_torch_satisfies_the_frozen_pin(self):
        installed = installed_distribution_versions()
        self.assertIn("torch", installed)
        report = check_dependency_conformance(ROOT, installed=installed)
        torch_entries = [entry for entry in report["entries"]
                         if entry["name"] == "torch"]
        self.assertTrue(torch_entries)
        self.assertTrue(all(entry["status"] == "satisfied"
                            for entry in torch_entries))

    def test_a_mismatching_version_is_reported(self):
        report = check_dependency_conformance(ROOT, installed={"torch": "1.0"})
        self.assertFalse(report["conforms"])
        self.assertIn("torch", report["unsatisfied"])

    def test_a_missing_distribution_is_reported(self):
        report = check_dependency_conformance(ROOT, installed={})
        self.assertFalse(report["conforms"])
        self.assertIn("torch", report["missing"])

    def test_an_archived_record_can_be_audited_on_its_own(self):
        report = dependency_conformance_from_record({
            "package_inventory": {"torch": "2.14.1+cpu"}})
        self.assertTrue(report["conforms"], report)

        drifted = dependency_conformance_from_record({
            "package_inventory": {"torch": "2.13.0"}})
        self.assertFalse(drifted["conforms"])
        self.assertIn("torch", drifted["unsatisfied"])

    def test_a_record_without_an_inventory_is_not_a_pass(self):
        report = dependency_conformance_from_record({"python": "3.12"})
        self.assertFalse(report["conforms"])
        self.assertIn("error", report)

    def test_a_malformed_record_is_not_a_pass(self):
        for bad in (None, [], "not-a-record"):
            with self.subTest(bad=bad):
                report = dependency_conformance_from_record(bad)
                self.assertFalse(report["conforms"])


class DependencyCheckCommandTest(unittest.TestCase):
    def _run(self, argv):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = experiments.main(argv)
        return code, out.getvalue()

    def test_the_command_passes_in_this_checkout(self):
        code, text = self._run(["--check-dependencies"])
        self.assertEqual(code, 0, text)
        self.assertIn("PASS", text)

    def test_the_command_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(experiments, "_write_result") as write:
                code, _ = self._run(["--check-dependencies", "--out", tmp])
            self.assertEqual(code, 0)
            write.assert_not_called()

    def test_a_mismatch_fails_the_command(self):
        drifted = check_dependency_conformance(ROOT, installed={"torch": "1.0"})
        with mock.patch.object(experiments, "check_dependency_conformance",
                               return_value=drifted):
            code, text = self._run(["--check-dependencies"])
        self.assertEqual(code, 1)
        self.assertIn("FAIL", text)

    def test_an_archived_environment_record_is_audited(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            record = experiments._environment_record(
                experiments.Settings(device="cpu"), git_dirty=False)
            (root / "environment.json").write_text(
                experiments.json.dumps(record), encoding="utf-8")
            code, text = self._run(["--check-dependencies", "--out", str(root)])
            self.assertEqual(code, 0, text)
            self.assertIn("archived record", text)

    def test_a_check_never_starts_a_grid(self):
        # The read-only checks follow the runner's convention: they win over
        # a requested experiment and no training is started.
        with mock.patch.object(experiments, "_write_result") as write:
            code, text = self._run(["--check-dependencies", "--only", "e1"])
        self.assertEqual(code, 0, text)
        write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
