"""
The server's M5 settings (M5-1): read from the environment at each use, never a
default path, and one directory rule shared with `--registry`.
=================================================================================
`src/config/server_settings.py`. What these pin:

- **the deployment mode** enables operator controls only when it is exactly
  `M1`; unset or anything else is read-only, and the reason never repeats the
  value set;
- **a missing setting turns its feature off with a reason**, and nothing falls
  back to a default path (the repository's `data/workspaces/` included);
- **the registry and the intake directory** must be absolute, existing, and
  outside the repository (a symlink into it too), by the same rule
  `validate_registry` applies to `--registry`;
- **the evidence directories** are `os.pathsep`-separated, and one bad entry
  refuses the whole setting;
- **the ordering roles** default to none, and `institutional_acceptance` is
  refused even when listed (§4.9 question 1);
- **each read sees the environment as it is now**: no value is cached.

Module: tests/unit/test_server_settings.py
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.config import server_settings as ss
from src.config.server_settings import REPO_ROOT


def _env(**values: str) -> dict:
    return {getattr(ss, f"ENV_{key}"): value for key, value in values.items()}


# ---------------------------------------------------------------------------
# The deployment mode
# ---------------------------------------------------------------------------
class TestDeploymentMode:

    def test_m1_enables(self):
        setting = ss.deployment_mode(_env(DEPLOYMENT_MODE="M1"))
        assert setting.available and setting.value == "M1"
        assert setting.variable == "SHEPHERD_DEPLOYMENT_MODE"

    @pytest.mark.parametrize("environ", [{}, _env(DEPLOYMENT_MODE="")], ids=["unset", "empty"])
    def test_unset_is_read_only(self, environ):
        setting = ss.deployment_mode(environ)
        assert not setting.available and setting.value is None
        assert "not set" in setting.problem and "read-only" in setting.problem

    @pytest.mark.parametrize("value", ["m1", " M1", "M1 ", "M2", "M3", "M1,M2", "yes"])
    def test_anything_else_is_read_only_and_not_repeated(self, value):
        setting = ss.deployment_mode(_env(DEPLOYMENT_MODE=value))
        assert not setting.available and setting.value is None
        assert "read-only" in setting.problem and "compared exactly" in setting.problem
        assert repr(value) not in setting.problem
        if value.strip() not in ("M1", ""):
            assert value.strip() not in setting.problem.replace("M1", "")

    def test_it_reads_the_environment_at_each_use(self, monkeypatch):
        monkeypatch.delenv(ss.ENV_DEPLOYMENT_MODE, raising=False)
        assert ss.deployment_mode().value is None
        monkeypatch.setenv(ss.ENV_DEPLOYMENT_MODE, "M1")
        assert ss.deployment_mode().value == "M1"
        monkeypatch.setenv(ss.ENV_DEPLOYMENT_MODE, "M3")
        assert ss.deployment_mode().value is None


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
class TestTheWorkspacesRoot:

    def test_unset_lists_nothing_and_assumes_no_default(self):
        setting = ss.workspaces_root({})
        assert setting.value is None
        assert "SHEPHERD_WORKSPACES_ROOT is not set" in setting.problem
        assert "no default" in setting.problem

    def test_an_existing_absolute_directory_is_resolved(self, tmp_path):
        (tmp_path / "real").mkdir()
        (tmp_path / "link").symlink_to(tmp_path / "real")
        setting = ss.workspaces_root(_env(WORKSPACES_ROOT=str(tmp_path / "link")))
        assert setting.available and setting.value == (tmp_path / "real").resolve()

    def test_inside_the_repository_is_allowed_when_named(self):
        """`data/workspaces/` is the usual root; it holds no patient data."""
        setting = ss.workspaces_root(_env(WORKSPACES_ROOT=str(REPO_ROOT / "tests")))
        assert setting.available and setting.value == (REPO_ROOT / "tests").resolve()

    @pytest.mark.parametrize("given, fragment", [
        (lambda tmp: "data/workspaces", "not an absolute path"),
        (lambda tmp: str(tmp / "missing"), "does not exist"),
        (lambda tmp: str(tmp / "file"), "not a directory"),
    ], ids=["relative", "missing", "file"])
    def test_a_bad_value_is_refused_with_its_reason(self, tmp_path, given, fragment):
        (tmp_path / "file").write_text("x")
        setting = ss.workspaces_root(_env(WORKSPACES_ROOT=given(tmp_path)))
        assert setting.value is None
        assert setting.problem.startswith("SHEPHERD_WORKSPACES_ROOT is refused: ")
        assert fragment in setting.problem
        assert "lists no workspace" in setting.problem


@pytest.mark.parametrize("reader, variable", [
    (ss.cohort_registry, "SHEPHERD_REGISTRY"),
    (ss.intake_directory, "SHEPHERD_INTAKE_DIR"),
], ids=["registry", "intake"])
class TestThePatientLevelDirectories:

    def test_unset_turns_the_feature_off(self, reader, variable):
        setting = reader({})
        assert setting.value is None and f"{variable} is not set" in setting.problem

    def test_an_existing_absolute_directory_outside_the_repository(
            self, reader, variable, tmp_path):
        setting = reader({variable: str(tmp_path)})
        assert setting.available and setting.value == tmp_path.resolve()

    @pytest.mark.parametrize("given, fragment", [
        (lambda tmp: "relative/dir", "not an absolute path"),
        (lambda tmp: str(REPO_ROOT / "data"), "inside the repository"),
        (lambda tmp: str(tmp / "into-repo"), "inside the repository"),
        (lambda tmp: str(tmp / "missing"), "does not exist"),
    ], ids=["relative", "inside", "link-inside", "missing"])
    def test_a_bad_value_is_refused(self, reader, variable, tmp_path, given, fragment):
        (tmp_path / "into-repo").symlink_to(REPO_ROOT / "src")
        setting = reader({variable: given(tmp_path)})
        assert setting.value is None
        assert setting.problem.startswith(f"{variable} is refused: ")
        assert fragment in setting.problem


class TestOneRuleForTheRegistry:
    """`--registry` and `SHEPHERD_REGISTRY` are refused, and accepted, alike."""

    @pytest.mark.parametrize("given", [
        lambda tmp: str(tmp),
        lambda tmp: "relative/registry",
        lambda tmp: str(REPO_ROOT / "data"),
        lambda tmp: str(tmp / "missing"),
        lambda tmp: str(tmp / "file"),
        lambda tmp: str(tmp / "into-repo"),
    ])
    def test_the_cli_rule_and_the_setting_agree(self, tmp_path, given):
        from src.evaluation.cohort_registry import RegistryRefused, validate_registry

        (tmp_path / "file").write_text("x")
        (tmp_path / "into-repo").symlink_to(REPO_ROOT)
        value = given(tmp_path)
        setting = ss.cohort_registry(_env(REGISTRY=value))
        try:
            resolved = validate_registry(value)
        except RegistryRefused as exc:
            assert setting.value is None and str(exc) in setting.problem
        else:
            assert setting.value == resolved

    def test_validate_registry_applies_the_shared_rule(self, monkeypatch, tmp_path):
        """Not a copy of it: replacing the shared rule changes what
        `validate_registry` accepts."""
        import src.evaluation.cohort_registry as registry_module

        calls = []

        def rule(path):
            calls.append(path)
            raise ss.DirectoryRefused("refused by the shared rule")

        monkeypatch.setattr(registry_module, "check_registry_root", rule)
        with pytest.raises(registry_module.RegistryRefused, match="refused by the shared rule"):
            registry_module.validate_registry(tmp_path)
        assert calls == [tmp_path]

    def test_a_non_path_is_refused(self):
        with pytest.raises(ss.DirectoryRefused, match="must be a path"):
            ss.check_registry_root(None)


# ---------------------------------------------------------------------------
# The evidence directories
# ---------------------------------------------------------------------------
class TestTrainingEvidenceDirs:

    def test_unset_is_none_and_usable(self):
        setting = ss.training_evidence_dirs({})
        assert setting.available and setting.value == ()

    def test_entries_are_resolved_in_order_and_kept_once(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        raw = os.pathsep.join([str(b), "", str(a), str(b), ""])
        setting = ss.training_evidence_dirs(_env(TRAINING_EVIDENCE_DIRS=raw))
        assert setting.available
        assert setting.value == (b.resolve(), a.resolve())

    def test_one_bad_entry_refuses_the_whole_setting(self, tmp_path):
        good = tmp_path / "good"
        good.mkdir()
        raw = os.pathsep.join([str(good), str(tmp_path / "missing")])
        setting = ss.training_evidence_dirs(_env(TRAINING_EVIDENCE_DIRS=raw))
        assert setting.value is None
        assert "evidence directory 2" in setting.problem and "does not exist" in setting.problem

    def test_a_relative_entry_is_refused_not_read_from_the_working_directory(self, tmp_path):
        setting = ss.training_evidence_dirs(_env(TRAINING_EVIDENCE_DIRS="evidence"))
        assert setting.value is None and "not an absolute path" in setting.problem


# ---------------------------------------------------------------------------
# The ordering roles
# ---------------------------------------------------------------------------
class TestOrderingRoles:

    def test_default_is_none(self):
        setting = ss.ordering_roles({})
        assert setting.available and setting.value == () and setting.note is None
        assert not ss.role_offers_ordering("mygene2", {})

    def test_listed_roles_in_order_once(self):
        setting = ss.ordering_roles(_env(ORDERING_ROLES=" mygene2, synthetic ,,mygene2"))
        assert setting.value == ("mygene2", "synthetic")

    def test_institutional_acceptance_is_refused_even_when_listed(self):
        environ = _env(ORDERING_ROLES="institutional_acceptance, mygene2")
        setting = ss.ordering_roles(environ)
        assert setting.value == ("mygene2",)
        assert "institutional_acceptance" in setting.note and "ignored" in setting.note
        assert not ss.role_offers_ordering("institutional_acceptance", environ)
        assert ss.role_offers_ordering("mygene2", environ)

    @pytest.mark.parametrize("role", ["other", None, 3, ""])
    def test_an_unlisted_role_offers_none(self, role):
        assert not ss.role_offers_ordering(role, _env(ORDERING_ROLES="mygene2"))


def test_all_settings_in_the_documented_order():
    names = [setting.variable for setting in ss.all_settings({})]
    assert names == [
        "SHEPHERD_DEPLOYMENT_MODE", "SHEPHERD_WORKSPACES_ROOT", "SHEPHERD_REGISTRY",
        "SHEPHERD_INTAKE_DIR", "SHEPHERD_TRAINING_EVIDENCE_DIRS", "SHEPHERD_ORDERING_ROLES",
    ]


def test_no_setting_is_read_at_import(monkeypatch, tmp_path):
    """Read at use: setting a variable after the module was imported takes
    effect on the next read."""
    monkeypatch.delenv(ss.ENV_WORKSPACES_ROOT, raising=False)
    assert ss.workspaces_root().value is None
    monkeypatch.setenv(ss.ENV_WORKSPACES_ROOT, str(tmp_path))
    assert ss.workspaces_root().value == tmp_path.resolve()


def test_both_guides_document_every_setting():
    """Each variable is in both deployment guides."""
    for guide in ("deployment-guide.en.md", "deployment-guide.md"):
        text = (REPO_ROOT / guide).read_text(encoding="utf-8")
        for setting in ss.all_settings({}):
            assert setting.variable in text, f"{guide} does not mention {setting.variable}"


def test_the_module_is_importable_without_the_service():
    """It sits below `src.utils` so the D8 check and the GPU lock can read it:
    it imports nothing of the project."""
    import ast

    tree = ast.parse(Path(ss.__file__).read_text(encoding="utf-8"))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                 for alias in node.names}
    assert not any(name and name.startswith("src") for name in imported), imported
