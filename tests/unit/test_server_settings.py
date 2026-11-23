"""
The server's M5 settings (M5-1; UI-06): one resolution at each use, never a
default path, and one directory rule shared with `--registry`.
=================================================================================
`src/config/server_settings.py`. What these pin:

- **the deployment mode** enables operator controls only when it is exactly
  `M1`, or, unset, by the local start's default (the launcher's mark on a
  loopback bind); anything else is read-only and never upgraded, and the
  reason never repeats the value set;
- **a data location** comes from its environment variable when it is set,
  else from the saved settings file's applied value, else nothing (UI-06,
  2026-11-16; `TestTheOneResolution` and the classes after it); a saved value
  is checked by the same rule at each read, and an unreadable file is a
  problem, never read as empty;
- **a missing setting turns its feature off with a reason**, and nothing falls
  back to a default path (the repository's `data/workspaces/` included);
- **the registry and the intake directory** must be absolute, existing, and
  outside the repository (a symlink into it too), by the same rule
  `validate_registry` applies to `--registry`;
- **the evidence directories** are `os.pathsep`-separated, and one entry known
  not to be a directory refuses the whole setting; one whose state cannot be
  learned is kept, in its place, and noted (K8-6 completion review,
  2026-11-08), while a required directory in that state is still refused;
- **whether a named path is a directory** is one explicit stat
  (`is_directory`), never `Path.is_dir`, which from Python 3.14 reads an
  unknown state as absent;
- **each read sees the environment as it is now**: no value is cached (the
  saved file is read again whenever it changes).

Module: tests/unit/test_server_settings.py
"""
from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from src.config import server_settings as ss
from src.config.server_settings import REPO_ROOT


def _env(**values: str) -> dict:
    return {getattr(ss, f"ENV_{key}"): value for key, value in values.items()}


#: How a refused value set in the environment is introduced, per setting
#: (UI-06: the plain name, where it came from, the variable as its Ref).
REFUSED_AT_START = {
    "SHEPHERD_WORKSPACES_ROOT": "The workspaces folder set at start "
                                "(Ref: SHEPHERD_WORKSPACES_ROOT) is refused: ",
    "SHEPHERD_REGISTRY": "The results registry set at start (Ref: SHEPHERD_REGISTRY) is refused: ",
    "SHEPHERD_INTAKE_DIR": "The import folder set at start (Ref: SHEPHERD_INTAKE_DIR) is refused: ",
    "SHEPHERD_TRAINING_EVIDENCE_DIRS": "The training evidence folders set at start "
                                       "(Ref: SHEPHERD_TRAINING_EVIDENCE_DIRS) are refused: ",
}


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
        assert not setting.available and setting.value is None and setting.unset
        assert setting.problem == ss.MODE_NOT_SET and "read-only" in setting.problem
        assert setting.source is None

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
        assert setting.value is None and setting.unset and setting.source is None
        assert setting.problem.startswith("No workspaces folder is set")
        assert "Settings › Data locations" in setting.problem
        assert setting.problem.endswith("Ref: SHEPHERD_WORKSPACES_ROOT")
        assert "No default is assumed" in setting.problem
        # T08: the one folder Settings suggests, never the repository's.
        assert ("Create suggested folder makes SHEPHERD-data/workspaces in the service "
                "account's home folder") in setting.problem
        assert "repository" not in setting.problem

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
        assert setting.problem.startswith(REFUSED_AT_START["SHEPHERD_WORKSPACES_ROOT"])
        assert fragment in setting.problem and setting.source == ss.SOURCE_ENVIRONMENT
        assert "lists no workspace" in setting.problem


@pytest.mark.parametrize("reader, variable", [
    (ss.cohort_registry, "SHEPHERD_REGISTRY"),
    (ss.intake_directory, "SHEPHERD_INTAKE_DIR"),
], ids=["registry", "intake"])
class TestThePatientLevelDirectories:

    def test_unset_turns_the_feature_off(self, reader, variable):
        setting = reader({})
        assert setting.value is None and setting.unset
        assert setting.problem.startswith("No ") and " is set, so " in setting.problem
        assert setting.problem.endswith(f"Ref: {variable}")

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
        assert setting.problem.startswith(REFUSED_AT_START[variable])
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


class TestADirectoryThatCannotBeChecked:
    """A parent the service account may not enter (the guide's `mkdir -m 700`
    for another account): the stat the directory rule makes (`is_directory`,
    through `Path.stat`) raises EACCES rather than answer. Each required
    setting is refused with why, never a bare error (M5-1 review), and its
    words are as they were; the optional evidence setting keeps the entry and
    notes it (K8-6 completion review, 2026-11-08)."""

    @pytest.fixture
    def locked(self, tmp_path, monkeypatch):
        locked = (tmp_path / "locked").resolve()
        locked.mkdir()
        original = Path.stat

        def stat(self, *args, **kwargs):
            if self == locked:
                raise PermissionError(13, "Permission denied", str(self))
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", stat)
        return locked

    def test_the_rule_refuses_it_and_says_why(self, locked):
        with pytest.raises(ss.DirectoryUncheckable) as refused:
            ss.check_operator_directory(locked, what="the registry", holds="x",
                                        outside_repository=True)
        assert isinstance(refused.value, ss.DirectoryRefused)
        assert str(refused.value) == (
            f"the registry {locked} cannot be checked (PermissionError: Permission denied). "
            "The account that runs this must be able to reach it and every directory above it.")
        assert (refused.value.path, refused.value.reason) == (
            locked, "PermissionError: Permission denied")

    @pytest.mark.parametrize("reader, key", [
        (ss.workspaces_root, "WORKSPACES_ROOT"), (ss.cohort_registry, "REGISTRY"),
        (ss.intake_directory, "INTAKE_DIR"),
    ])
    def test_each_required_setting_is_refused_not_raised(self, locked, reader, key):
        setting = reader(_env(**{key: str(locked)}))
        assert setting.value is None and "cannot be checked" in setting.problem
        assert not setting.available and setting.note is None

    def test_the_optional_evidence_setting_keeps_it_and_says_why(self, locked, tmp_path):
        """P2-B of the K8-6 completion review. The entry is optional evidence:
        it stays in the value, in its place, the setting stays available, and
        the note names it, its reason and what a run does with it: R10 names it
        only for inputs found nowhere else, never for every run (review
        round). HEAD: the whole setting refused ("cannot be checked")."""
        good = (tmp_path / "good").resolve()
        good.mkdir()
        setting = ss.training_evidence_dirs(
            _env(TRAINING_EVIDENCE_DIRS=os.pathsep.join([str(good), str(locked)])))
        assert setting.available and setting.problem is None
        assert setting.value == (good, locked)
        assert setting.note == (
            f"Cannot be checked: {locked} (PermissionError: Permission denied). R10 cannot "
            "look in it: inputs found nowhere else read unverifiable, naming it.")

    def test_a_status_view_still_reads_every_setting(self, locked):
        settings = ss.all_settings(_env(REGISTRY=str(locked), WORKSPACES_ROOT=str(locked)))
        assert len(settings) == 5
        assert "cannot be checked" in settings[2].problem

    def test_validate_registry_refuses_it_too(self, locked):
        from src.evaluation.cohort_registry import RegistryRefused, validate_registry

        with pytest.raises(RegistryRefused, match="cannot be checked"):
            validate_registry(locked)


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

    @pytest.mark.parametrize("kind", ["missing", "file", "name-too-long", "beside-unknown"])
    def test_an_entry_known_not_to_be_a_directory_refuses_the_whole_setting(
            self, tmp_path, monkeypatch, kind):
        """Guard (K8-6 completion review, 2026-11-08). A missing path, a file
        and a name too long to exist (a real ENAMETOOLONG) refuse the setting
        as before, beside an entry whose state cannot be learned too: Start
        stays disabled. Under Python 3.14's `Path.is_dir` too."""
        from tests.unit.test_r10 import _pathlib_as_in_python_3_14, _stat_fails

        unknown = tmp_path / "unknown"
        unknown.mkdir()
        _stat_fails(monkeypatch, unknown, PermissionError(13, "Permission denied"))
        _pathlib_as_in_python_3_14(monkeypatch)
        bad = tmp_path / ("n" * 300 if kind == "name-too-long" else "bad")
        if kind == "file":
            bad.write_text("x")
        entries = [str(unknown), str(bad)] if kind == "beside-unknown" else [str(bad)]
        setting = ss.training_evidence_dirs(
            _env(TRAINING_EVIDENCE_DIRS=os.pathsep.join(entries)))
        assert setting.value is None and not setting.available and setting.note is None
        assert setting.problem == (
            f"{REFUSED_AT_START['SHEPHERD_TRAINING_EVIDENCE_DIRS']}evidence directory {len(entries)} "
            f"{bad} does not exist or is not a directory. The operator creates it; nothing "
            "creates one, so a mistyped path cannot become a directory nobody chose. No "
            "evidence directory is offered until it is corrected.")

    @pytest.mark.parametrize("error, says", [
        (PermissionError(13, "Permission denied"), "PermissionError: Permission denied"),
        (OSError(5, "Input/output error"), "OSError: Input/output error"),
    ], ids=["permission", "io-error"])
    def test_an_entry_whose_state_cannot_be_learned_is_kept_in_its_place(
            self, tmp_path, monkeypatch, error, says):
        """P2-B of the K8-6 completion review (2026-11-08): neither assumed to
        exist nor assumed absent. The entry stays in the value, in its place,
        resolved, kept once; the setting is available; the note names each
        such entry and its reason, and says what a run does with it. Under
        Python 3.14's `Path.is_dir` too, which would read it as absent. HEAD:
        the whole setting refused."""
        from tests.unit.test_r10 import _pathlib_as_in_python_3_14, _stat_fails

        first, good, second = (tmp_path / name for name in ("first", "good", "second"))
        for directory in (first, good, second):
            directory.mkdir()
        for directory in (first, second):
            _stat_fails(monkeypatch, directory.resolve(), error)
        _pathlib_as_in_python_3_14(monkeypatch)
        raw = os.pathsep.join([str(first), str(good), str(first), str(second)])

        setting = ss.training_evidence_dirs(_env(TRAINING_EVIDENCE_DIRS=raw))

        assert setting.available and setting.problem is None
        assert setting.value == (first.resolve(), good.resolve(), second.resolve())
        assert setting.note == (
            f"Cannot be checked: {first.resolve()} ({says}); {second.resolve()} ({says}). "
            "R10 cannot look in them: inputs found nowhere else read unverifiable, naming "
            "them.")

    def test_an_unknown_entry_named_through_a_link_is_kept_once_by_its_resolved_path(
            self, tmp_path, monkeypatch):
        """Review round of the K8-6 completion review (2026-11-08). The entry
        is kept by the path the rule checked, resolved, not as given: named
        through a link and then directly, it is one directory, kept and noted
        once. The run is given that path, as for an entry that can be
        checked."""
        from tests.unit.test_r10 import _stat_fails

        real = (tmp_path / "real").resolve()
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        _stat_fails(monkeypatch, real, PermissionError(13, "Permission denied"))

        setting = ss.training_evidence_dirs(
            _env(TRAINING_EVIDENCE_DIRS=os.pathsep.join([str(link), str(real)])))

        assert setting.available and setting.value == (real,)
        assert setting.note == (
            f"Cannot be checked: {real} (PermissionError: Permission denied). R10 cannot "
            "look in it: inputs found nowhere else read unverifiable, naming it.")

    def test_an_entry_whose_resolve_fails_is_kept_as_given(self, tmp_path, monkeypatch):
        """Its resolve failing for a reason that does not say it is absent is
        the same unknown state: the entry is kept under the path given. A
        required directory in that state is still refused, in the words it
        had ("cannot be resolved")."""
        entry = tmp_path / "remote"
        real = Path.resolve

        def resolve(self, strict=False):
            if self == entry:
                raise OSError(5, "Input/output error", str(self))
            return real(self, strict)

        monkeypatch.setattr(Path, "resolve", resolve)

        setting = ss.training_evidence_dirs(_env(TRAINING_EVIDENCE_DIRS=str(entry)))
        assert setting.available and setting.value == (entry,)
        assert setting.note == (f"Cannot be checked: {entry} (OSError: Input/output error). "
                                "R10 cannot look in it: inputs found nowhere else read "
                                "unverifiable, naming it.")
        registry = ss.cohort_registry(_env(REGISTRY=str(entry)))
        assert registry.value is None and registry.problem.startswith(
            f"{REFUSED_AT_START['SHEPHERD_REGISTRY']}the registry {entry} cannot be resolved "
            "(OSError: ")

    @pytest.mark.parametrize("error", [
        FileNotFoundError(errno.ENOENT, "No such file or directory"),
        OSError(errno.ELOOP, "Too many levels of symbolic links"),
    ], ids=["absent", "link-loop"])
    def test_a_resolve_error_that_says_the_entry_cannot_exist_refuses_the_setting(
            self, tmp_path, monkeypatch, error):
        """Guard, review round of the K8-6 completion review (2026-11-08). Only
        a resolve error that leaves the state unknown keeps the entry; one that
        says it is absent or a link loop is the known-invalid entry it names,
        and refuses the whole setting as a missing directory does."""
        entry = tmp_path / "gone"
        real = Path.resolve

        def resolve(self, strict=False):
            if self == entry:
                raise type(error)(error.errno, error.strerror, str(self))
            return real(self, strict)

        monkeypatch.setattr(Path, "resolve", resolve)

        setting = ss.training_evidence_dirs(_env(TRAINING_EVIDENCE_DIRS=str(entry)))

        assert setting.value is None and not setting.available and setting.note is None
        assert setting.problem.startswith(
            f"{REFUSED_AT_START['SHEPHERD_TRAINING_EVIDENCE_DIRS']}evidence directory 1 {entry} "
            f"cannot be resolved ({type(error).__name__}: ")


class TestIsDirectory:
    """`is_directory`, the one explicit stat both the setting and R10's
    `--training-evidence` check use (K8-6 completion review, 2026-11-08)."""

    def test_a_directory_and_a_link_to_one_are_directories(self, tmp_path):
        (tmp_path / "d").mkdir()
        (tmp_path / "link").symlink_to(tmp_path / "d")
        assert ss.is_directory(tmp_path / "d") is True
        assert ss.is_directory(str(tmp_path / "link")) is True

    @pytest.mark.parametrize("kind", ["missing", "file", "under-a-file", "loop",
                                      "name-too-long", "null-byte"])
    def test_what_cannot_be_a_directory_is_false(self, tmp_path, monkeypatch, kind):
        from tests.unit.test_r10 import _pathlib_as_in_python_3_14

        (tmp_path / "file").write_text("x")
        (tmp_path / "loop").symlink_to(tmp_path / "loop")
        path = {"missing": tmp_path / "missing", "file": tmp_path / "file",
                "under-a-file": tmp_path / "file" / "below", "loop": tmp_path / "loop",
                "name-too-long": tmp_path / ("n" * 300),
                "null-byte": f"{tmp_path}/bad\0name"}[kind]
        _pathlib_as_in_python_3_14(monkeypatch)
        assert ss.is_directory(path) is False

    @pytest.mark.parametrize("error", [PermissionError(13, "Permission denied"),
                                       OSError(5, "Input/output error")],
                             ids=["permission", "io-error"])
    def test_an_unknown_state_is_raised_never_false(self, tmp_path, monkeypatch, error):
        """A9: under Python 3.14's `Path.is_dir`, which answers False for it,
        the state is still unknown, never absent."""
        from tests.unit.test_r10 import _pathlib_as_in_python_3_14, _stat_fails

        target = tmp_path / "d"
        target.mkdir()
        _stat_fails(monkeypatch, target, error)
        _pathlib_as_in_python_3_14(monkeypatch)
        assert target.is_dir() is False, "the simulation reads it as absent"
        with pytest.raises(type(error)):
            ss.is_directory(target)


def test_all_settings_in_the_documented_order():
    names = [setting.variable for setting in ss.all_settings({})]
    assert names == [
        "SHEPHERD_DEPLOYMENT_MODE", "SHEPHERD_WORKSPACES_ROOT", "SHEPHERD_REGISTRY",
        "SHEPHERD_INTAKE_DIR", "SHEPHERD_TRAINING_EVIDENCE_DIRS",
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
    of the project it imports only the saved settings file's module
    (UI-06), which imports nothing of the project (the launcher reads both
    before anything starts)."""
    import ast

    from src.config import deployment_settings

    def imports(module):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        found = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        found |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                  for alias in node.names}
        return {name for name in found if name and name.startswith("src")}

    assert imports(ss) == {"src.config", "src.config.deployment_settings"}, imports(ss)
    assert imports(deployment_settings) == set(), imports(deployment_settings)


# ---------------------------------------------------------------------------
# Loopback, and the network bind (SPEC_4 §3.1, B1; M5-1)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text, loopback", [
    ("127.0.0.1", True), ("127.255.0.9", True), ("::1", True), ("::ffff:127.0.0.1", True),
    ("192.0.2.10", False), ("0.0.0.0", False), ("::", False), ("::ffff:192.0.2.1", False),
    ("localhost", False), ("", False), (None, False), (b"127.0.0.1", False),
])
def test_a_loopback_address_is_an_ip_literal_on_the_loopback_network(text, loopback):
    assert ss.is_loopback_ip(text) is loopback


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "LOCALHOST", "::1", "[::1]",
                                  "127.0.0.2"])
def test_a_loopback_bind_is_allowed_and_says_nothing(host):
    decision = ss.network_bind(host, {})
    assert decision.allowed and decision.message is None and ss.bind_is_loopback(host)


@pytest.mark.parametrize("host, where", [("0.0.0.0", "all interfaces"), ("::", "all interfaces"),
                                         ("", "all interfaces"), ("192.0.2.10", "192.0.2.10"),
                                         ("myhost.lan", "myhost.lan")])
def test_a_network_bind_is_refused_unless_the_risk_is_accepted(host, where):
    refused = ss.network_bind(host, {})
    assert not refused.allowed
    assert refused.message.startswith(f"refusing to bind {where}:")
    assert "SHEPHERD_ACCEPT_UNAUTHENTICATED_NETWORK_BIND=1" in refused.message
    assert "Nothing was started." in refused.message

    accepted = ss.network_bind(host, {ss.ENV_ACCEPT_UNAUTHENTICATED_NETWORK_BIND: "1"})
    assert accepted.allowed and accepted.host == host
    assert "risk acceptance, not a control" in accepted.message
    assert "Operator controls stay refused" in accepted.message


@pytest.mark.parametrize("value", ["", "0", "true", "yes", "1 ", "01"])
def test_only_the_exact_value_one_accepts_the_risk(value):
    assert not ss.network_bind("0.0.0.0",
                               {ss.ENV_ACCEPT_UNAUTHENTICATED_NETWORK_BIND: value}).allowed


def test_the_default_bind_is_loopback():
    assert ss.DEFAULT_BIND_HOST == "127.0.0.1" and ss.bind_is_loopback(ss.DEFAULT_BIND_HOST)


def test_an_inherited_socket_counts_as_a_network_bind():
    """Its address is not read, so it takes the same acceptance as a network
    bind, with the same flag (uvicorn --fd; systemd socket activation)."""
    refused = ss.inherited_socket_bind(3, {})
    assert not refused.allowed
    assert refused.message.startswith(
        "refusing to serve on the inherited socket (file descriptor 3):")
    assert "SHEPHERD_ACCEPT_UNAUTHENTICATED_NETWORK_BIND=1" in refused.message
    assert "Nothing was started." in refused.message
    for value in ("", "0", "true", " 1"):
        assert not ss.inherited_socket_bind(3, {
            ss.ENV_ACCEPT_UNAUTHENTICATED_NETWORK_BIND: value}).allowed
    accepted = ss.inherited_socket_bind(3, {ss.ENV_ACCEPT_UNAUTHENTICATED_NETWORK_BIND: "1"})
    assert accepted.allowed and "risk acceptance, not a control" in accepted.message


def test_startup_checks_the_host_passed_in_and_nothing_else():
    """SPEC_4 §3.1's third enforcement point: the host the launcher or
    `main()` passed in, through the same rule; none passed in, no check."""
    assert ss.startup_bind({}) is None
    assert ss.startup_bind({ss.ENV_BIND_HOST: "127.0.0.1"}).allowed
    for host in ("0.0.0.0", "", "192.0.2.10"):
        decision = ss.startup_bind({ss.ENV_BIND_HOST: host})
        assert not decision.allowed and decision.message.startswith("refusing to bind")
        assert ss.startup_bind({ss.ENV_BIND_HOST: host,
                                ss.ENV_ACCEPT_UNAUTHENTICATED_NETWORK_BIND: "1"}).allowed


def test_the_deployment_mode_is_documented_as_applied():
    """WP1 said the mode was not applied yet; D8 applies it since M5-1."""
    text = Path(ss.__file__).read_text(encoding="utf-8")
    assert "Not applied yet" not in text and "once D8's check applies it" not in text
    assert "src/utils/operator_access.py" in text


# ---------------------------------------------------------------------------
# `python -m src.api.main` applies the same bind rule
# ---------------------------------------------------------------------------
@pytest.fixture
def uvicorn_runs(monkeypatch):
    pytest.importorskip("uvicorn")
    import uvicorn

    runs = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: runs.append((app, kwargs)))
    return runs


def test_the_api_entry_point_binds_loopback_without_proxy_headers(uvicorn_runs):
    import os

    from src.api import main

    main.main()
    ((app, kwargs),) = uvicorn_runs
    assert app == "src.api.main:app"
    assert kwargs["host"] == "127.0.0.1" and kwargs["proxy_headers"] is False
    # Passed to application startup, which checks it again (SPEC_4 §3.1).
    assert os.environ[ss.ENV_BIND_HOST] == "127.0.0.1"


def test_application_startup_refuses_a_network_bind_passed_in_without_acceptance(
        monkeypatch):
    """The lifespan applies `startup_bind` before anything else: a refused
    host ends startup, so uvicorn exits without serving; an accepted one
    starts with the warning logged."""
    from fastapi.testclient import TestClient

    from src.api import main

    monkeypatch.setenv(ss.ENV_BIND_HOST, "0.0.0.0")
    with pytest.raises(RuntimeError, match="Startup refused: SHEPHERD_BIND_HOST"):
        with TestClient(main.app):
            pass
    monkeypatch.setenv(ss.ENV_ACCEPT_UNAUTHENTICATED_NETWORK_BIND, "1")
    with TestClient(main.app) as client:
        assert client.get("/health").status_code == 200


def test_the_api_entry_point_refuses_a_network_bind_without_acceptance(uvicorn_runs,
                                                                       monkeypatch):
    from src.api import main

    with pytest.raises(SystemExit, match="refusing to bind all interfaces"):
        main.main(host="0.0.0.0")
    assert uvicorn_runs == []
    monkeypatch.setenv(ss.ENV_ACCEPT_UNAUTHENTICATED_NETWORK_BIND, "1")
    main.main(host="0.0.0.0")
    assert uvicorn_runs[0][1]["host"] == "0.0.0.0"


# ---------------------------------------------------------------------------
# The saved settings file and the one resolution (UI-06, 2026-11-16)
# ---------------------------------------------------------------------------
from src.config import deployment_settings as ds  # noqa: E402
from src.config.deployment_settings import ENV_SETTINGS_FILE, FORMAT  # noqa: E402


def _write_saved(path: Path, applied: dict, saved: dict | None = None, **extra) -> Path:
    """A saved settings file as Settings writes it (`settings_store`)."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"format": FORMAT, "applied": applied,
                "saved": dict(applied) if saved is None else saved, **extra}
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


@pytest.fixture
def saved_file(tmp_path):
    """A settings file of the test's own, named explicitly, and its environ."""
    path = tmp_path / "settings" / "deployment.json"
    return path, {ENV_SETTINGS_FILE: str(path)}


@pytest.fixture
def folders(tmp_path):
    """Directories outside the repository for each data location."""
    made = {}
    for name in ("workspaces", "registry", "intake", "evidence-a", "evidence-b", "other"):
        (tmp_path / name).mkdir()
        made[name] = tmp_path / name
    return made


class TestTheOneResolution:
    """The environment wins, then the saved file's applied values, then
    nothing; each value says where it came from (`Setting.source`)."""

    READERS = [
        (ss.workspaces_root, "WORKSPACES_ROOT", "workspaces_root", "workspaces"),
        (ss.cohort_registry, "REGISTRY", "registry", "registry"),
        (ss.intake_directory, "INTAKE_DIR", "intake_dir", "intake"),
    ]

    @pytest.mark.parametrize("reader, env, key, folder", READERS,
                             ids=["workspaces", "registry", "intake"])
    def test_a_saved_value_is_used_when_the_environment_sets_none(
            self, saved_file, folders, reader, env, key, folder):
        path, environ = saved_file
        _write_saved(path, {key: str(folders[folder])})
        setting = reader(environ)
        assert setting.available and setting.value == folders[folder].resolve()
        assert setting.source == ss.SOURCE_SAVED
        # An empty variable is not set: the saved value is used (as `_raw` reads it).
        assert reader({**environ, getattr(ss, f"ENV_{env}"): ""}).source == ss.SOURCE_SAVED

    @pytest.mark.parametrize("reader, env, key, folder", READERS,
                             ids=["workspaces", "registry", "intake"])
    def test_an_explicit_environment_value_wins_over_the_saved_one(
            self, saved_file, folders, reader, env, key, folder):
        path, environ = saved_file
        _write_saved(path, {key: str(folders["other"])})
        setting = reader({**environ, getattr(ss, f"ENV_{env}"): str(folders[folder])})
        assert setting.value == folders[folder].resolve()
        assert setting.source == ss.SOURCE_ENVIRONMENT

    def test_only_the_applied_values_are_in_effect(self, saved_file, folders):
        """A saved change waiting to be applied changes nothing in use, so a
        job and a command read what the service uses."""
        path, environ = saved_file
        _write_saved(path, {"workspaces_root": str(folders["workspaces"])},
                     saved={"workspaces_root": str(folders["other"])})
        assert ss.workspaces_root(environ).value == folders["workspaces"].resolve()
        assert ss.gpu_lock_location(environ).value == (
            folders["workspaces"].resolve() / ss.GPU_LOCK_FILE_NAME)
        assert ds.read_saved(environ).pending == ("workspaces_root",)

    def test_the_gpu_lock_is_under_the_saved_root_and_says_so(self, saved_file, folders):
        path, environ = saved_file
        _write_saved(path, {"workspaces_root": str(folders["workspaces"])})
        lock = ss.gpu_lock_location(environ)
        assert lock.value == folders["workspaces"].resolve() / ss.GPU_LOCK_FILE_NAME
        assert lock.source == ss.SOURCE_SAVED

    def test_a_saved_value_is_checked_by_the_same_rule_at_each_read(
            self, saved_file, folders):
        """Never trusted because it was checked when saved: a folder removed
        since, or one inside the repository, is refused as an environment
        value is, saying where it came from and where to correct it."""
        path, environ = saved_file
        _write_saved(path, {"registry": str(REPO_ROOT / "data"),
                            "workspaces_root": str(folders["workspaces"] / "gone")})
        registry = ss.cohort_registry(environ)
        assert registry.value is None and registry.source == ss.SOURCE_SAVED
        assert registry.problem.startswith("The results registry saved in Settings is refused: ")
        assert "inside the repository" in registry.problem
        assert registry.problem.endswith("Correct it in Settings › Data locations.")
        root = ss.workspaces_root(environ)
        assert root.value is None and "does not exist" in root.problem and not root.unset
        lock = ss.gpu_lock_location(environ)
        assert lock.value is None and not lock.unset and "GPU lock under it" in lock.problem

    def test_saved_evidence_folders_are_a_list_read_by_the_same_rule(self, saved_file, folders):
        path, environ = saved_file
        entries = [str(folders["evidence-a"]), "", str(folders["evidence-b"]),
                   str(folders["evidence-a"])]
        _write_saved(path, {"training_evidence_dirs": entries})
        setting = ss.training_evidence_dirs(environ)
        assert setting.value == (folders["evidence-a"].resolve(), folders["evidence-b"].resolve())
        assert setting.source == ss.SOURCE_SAVED
        _write_saved(path, {"training_evidence_dirs": [str(folders["other"] / "gone")]})
        refused = ss.training_evidence_dirs(environ)
        assert refused.value is None and refused.problem.startswith(
            "The training evidence folders saved in Settings are refused: evidence directory 1 ")

    def test_the_deployment_registry_rule_reads_the_saved_registry(self, saved_file, folders):
        path, environ = saved_file
        _write_saved(path, {"registry": str(folders["registry"])})
        assert ss.deployment_registry_refusal(folders["registry"].resolve(), environ) is None
        said = ss.deployment_registry_refusal(folders["other"].resolve(), environ)
        assert said.startswith(f"the registry {folders['other'].resolve()} is not the "
                               "deployment's: the results registry saved in Settings is ")
        assert said.endswith("choose it in Settings › Data locations")

    def test_nothing_set_is_still_off_with_its_reason_never_a_default(self, saved_file):
        path, environ = saved_file
        _write_saved(path, {})
        for setting in ss.all_settings(environ)[1:4]:
            assert setting.value is None and setting.unset and setting.source is None
        assert ss.training_evidence_dirs(environ).value == ()

    def test_a_mapping_without_a_location_has_no_saved_file(self, tmp_path, folders,
                                                            monkeypatch):
        """Existing callers that pass `environ={...}` keep their meaning: the
        location comes from that mapping only, never from the process."""
        process_file = _write_saved(tmp_path / "process.json",
                                    {"workspaces_root": str(folders["workspaces"])})
        monkeypatch.setenv(ENV_SETTINGS_FILE, str(process_file))
        monkeypatch.delenv(ss.ENV_WORKSPACES_ROOT, raising=False)
        assert ss.workspaces_root().source == ss.SOURCE_SAVED           # the process's
        assert ss.workspaces_root({}).unset                            # the mapping's: none
        assert ds.settings_location({}).path is None

    def test_nothing_writes_a_resolved_value_into_the_environment(self, saved_file, folders,
                                                                   monkeypatch):
        path, environ = saved_file
        monkeypatch.setenv(ENV_SETTINGS_FILE, str(path))
        monkeypatch.delenv(ss.ENV_WORKSPACES_ROOT, raising=False)
        _write_saved(path, {"workspaces_root": str(folders["workspaces"])})
        before = dict(os.environ)
        assert ss.workspaces_root().source == ss.SOURCE_SAVED
        ss.all_settings()
        assert dict(os.environ) == before


class TestTheSavedFileLocation:

    @pytest.mark.parametrize("platform, environ, expected", [
        ("linux", {"HOME": "/home/u"}, "/home/u/.config/shepherd-advanced/deployment.json"),
        ("linux", {"HOME": "/home/u", "XDG_CONFIG_HOME": "/cfg"},
         "/cfg/shepherd-advanced/deployment.json"),
        ("linux", {"HOME": "/home/u", "XDG_CONFIG_HOME": "relative"},
         "/home/u/.config/shepherd-advanced/deployment.json"),
        ("darwin", {"HOME": "/Users/u"},
         "/Users/u/Library/Application Support/shepherd-advanced/deployment.json"),
        ("win32", {"LOCALAPPDATA": "/appdata/local", "HOME": "/ignored"},
         "/appdata/local/shepherd-advanced/deployment.json"),
        ("linux", {"HOME": "/home/u", ENV_SETTINGS_FILE: "/slots/B/deployment.json"},
         "/slots/B/deployment.json"),
    ], ids=["linux-home", "linux-xdg", "linux-relative-xdg", "macos", "windows", "explicit"])
    def test_the_user_config_dir_per_platform_or_the_explicit_file(
            self, platform, environ, expected):
        location = ds.settings_location(environ, platform=platform)
        assert location.path == Path(expected) and location.problem is None
        assert location.explicit is (ENV_SETTINGS_FILE in environ)

    @pytest.mark.parametrize("environ", [{}, {"HOME": "relative"}, {"XDG_CONFIG_HOME": ""}],
                             ids=["nothing", "relative-home", "empty-xdg"])
    def test_no_place_is_no_saved_file(self, environ):
        assert ds.settings_location(environ, platform="linux") == ds.Location(None)
        assert ds.read_saved(environ).path is None and ds.read_saved(environ).problem is None

    def test_a_relative_explicit_file_is_refused_for_every_setting_it_would_give(self):
        environ = {ENV_SETTINGS_FILE: "relative/deployment.json"}
        location = ds.settings_location(environ)
        assert location.path is None and "is not an absolute path" in location.problem
        root = ss.workspaces_root(environ)
        assert root.value is None and not root.unset
        assert "the workspaces folder is not known" in root.problem

    def test_the_file_is_outside_the_repository_by_default(self):
        location = ds.settings_location({"HOME": "/home/u"}, platform="linux")
        assert REPO_ROOT not in location.path.parents

    def test_each_test_has_a_settings_file_of_its_own(self):
        """Item 11: no test reads the account's saved data locations."""
        location = ds.settings_location()
        assert location.explicit and not location.path.exists()
        assert Path.home() not in location.path.parents or "pytest" in str(location.path)


class TestAnUnreadableFile:

    @pytest.mark.parametrize("content, says", [
        (b"{ not json", "it is not valid JSON"),
        (b"[1, 2]", "it does not hold a JSON object"),
        (b'{"applied": {}}', "it names no format"),
        (b'{"format": "shepherd-deployment-settings/9", "applied": {}}',
         "its format is not shepherd-deployment-settings/1"),
        (b'{"format": "shepherd-deployment-settings/1", "applied": []}',
         "its 'applied' part is not a JSON object"),
        (b'{"format": "shepherd-deployment-settings/1", "applied": {"registry": 7}}',
         "its 'applied' part is of the wrong shape"),
        (b'{"format": "shepherd-deployment-settings/1", '
         b'"applied": {"training_evidence_dirs": "/a"}}', "must be a list of paths"),
        (b"\xff\xfe", "it is not UTF-8 text"),
    ], ids=["not-json", "not-object", "no-format", "other-format", "applied-list",
            "wrong-value", "evidence-string", "not-utf8"])
    def test_every_setting_it_would_give_is_a_problem_naming_the_file(
            self, saved_file, content, says):
        path, environ = saved_file
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
        read = ds.read_saved(environ)
        assert read.problem.startswith(f"The saved settings file {path} cannot be read (")
        assert says in read.problem and read.repairable
        for setting, name in zip(ss.all_settings(environ)[1:],
                                 ("workspaces folder", "results registry", "import folder",
                                  "training evidence folders")):
            assert setting.value is None and not setting.unset and setting.source is None
            assert setting.problem == (f"{read.problem}: the {name} is not known. Fix it in "
                                       "Settings › Data locations.")
        lock = ss.gpu_lock_location(environ)
        assert lock.value is None and not lock.unset
        assert ss.deployment_registry_refusal(Path("/x"), environ).startswith(read.problem)

    def test_the_environment_still_wins_over_an_unreadable_file(self, saved_file, folders):
        path, environ = saved_file
        path.parent.mkdir(parents=True)
        path.write_text("{ not json")
        setting = ss.workspaces_root({**environ, ss.ENV_WORKSPACES_ROOT: str(folders["other"])})
        assert setting.available and setting.source == ss.SOURCE_ENVIRONMENT

    def test_a_file_that_cannot_be_opened_is_not_repairable(self, saved_file, monkeypatch):
        path, environ = saved_file
        path.parent.mkdir(parents=True)
        path.write_text("{}")
        real = Path.read_bytes

        def denied(self):
            if self == path:
                raise PermissionError(13, "Permission denied", str(self))
            return real(self)

        monkeypatch.setattr(Path, "read_bytes", denied)
        read = ds.read_saved(environ)
        assert "PermissionError: Permission denied" in read.problem and not read.repairable
        assert "is not known" in ss.cohort_registry(environ).problem

    def test_reads_are_cached_and_a_rewrite_is_read_again(self, saved_file, folders,
                                                         monkeypatch):
        """One stat per read while the file is unchanged; a replacement (a new
        inode, as the atomic writer makes) is read again, even at the same
        size and within the same clock tick."""
        from src.config.atomic_json import write_json_atomic

        path, environ = saved_file
        a, b = folders["evidence-a"], folders["evidence-b"]       # the same length
        write_json_atomic(path, {"format": FORMAT, "applied": {"workspaces_root": str(a)}})
        assert ss.workspaces_root(environ).value == a.resolve()
        reads = []
        real = ds.read_file
        monkeypatch.setattr(ds, "read_file", lambda p: reads.append(p) or real(p))
        for _ in range(5):
            ss.workspaces_root(environ)
        assert reads == []
        write_json_atomic(path, {"format": FORMAT, "applied": {"workspaces_root": str(b)}})
        assert ss.workspaces_root(environ).value == b.resolve() and len(reads) == 1


class TestTheLocalStartsDefaultMode:
    """D06: `M1` from the local start only with its mark and a loopback bind;
    an explicit mode is never upgraded. D8's per-request check is
    test_operator_access's."""

    LOCAL = {ss.ENV_LOCAL_ENTRY: "1", ss.ENV_BIND_HOST: "127.0.0.1"}

    def test_the_local_start_gives_m1_and_says_so(self):
        setting = ss.deployment_mode(self.LOCAL)
        assert setting.available and setting.value == "M1"
        assert setting.source == ss.SOURCE_LOCAL_DEFAULT

    @pytest.mark.parametrize("environ", [
        {ss.ENV_LOCAL_ENTRY: "1"},                                        # no bind host
        {ss.ENV_LOCAL_ENTRY: "1", ss.ENV_BIND_HOST: "0.0.0.0"},
        {ss.ENV_LOCAL_ENTRY: "1", ss.ENV_BIND_HOST: ""},
        {ss.ENV_LOCAL_ENTRY: "1", ss.ENV_BIND_HOST: "192.0.2.10"},
        {ss.ENV_LOCAL_ENTRY: "yes", ss.ENV_BIND_HOST: "127.0.0.1"},
        {ss.ENV_LOCAL_ENTRY: "", ss.ENV_BIND_HOST: "127.0.0.1"},
    ], ids=["no-bind-host", "all-interfaces", "empty-host", "lan", "not-1", "empty"])
    def test_without_its_mark_and_a_loopback_bind_it_is_read_only(self, environ):
        setting = ss.deployment_mode(environ)
        assert setting.value is None and setting.unset and setting.problem == ss.MODE_NOT_SET

    @pytest.mark.parametrize("mode", ["read-only", "m1", "M3", " M1"])
    def test_an_explicit_mode_is_never_upgraded(self, mode):
        setting = ss.deployment_mode({**self.LOCAL, ss.ENV_DEPLOYMENT_MODE: mode})
        assert setting.value is None and not setting.unset
        assert setting.problem == ss.MODE_NOT_M1 and setting.source == ss.SOURCE_ENVIRONMENT

    def test_an_explicit_m1_is_the_environment_s(self):
        setting = ss.deployment_mode({**self.LOCAL, ss.ENV_DEPLOYMENT_MODE: "M1"})
        assert setting.value == "M1" and setting.source == ss.SOURCE_ENVIRONMENT

    def test_the_mode_is_never_read_from_the_saved_file(self, saved_file):
        path, environ = saved_file
        _write_saved(path, {}, deployment_mode="M1", applied_mode="M1")
        assert ss.deployment_mode(environ).value is None


class TestSavingApplyingAndRepairing:
    """`src/utils/settings_store.py`: every write under the file's lock, by
    the one atomic writer; a save checks each value; an apply only while no
    GPU lock the old values name is held; an unreadable file is never
    overwritten. The job runner's half of the apply is test_backend_control's."""

    @pytest.fixture
    def store(self):
        from src.utils import settings_store

        return settings_store

    def test_a_save_writes_saved_only_and_an_apply_makes_it_applied(
            self, store, saved_file, folders):
        import json

        path, environ = saved_file
        saved = store.save({"workspaces_root": str(folders["workspaces"]),
                            "training_evidence_dirs": [str(folders["evidence-a"])]}, environ)
        assert saved.saved, saved.message
        document = json.loads(path.read_text())
        assert document["format"] == FORMAT and document["applied"] == {}
        assert document["saved"] == {"workspaces_root": str(folders["workspaces"]),
                                     "training_evidence_dirs": [str(folders["evidence-a"])]}
        assert ss.workspaces_root(environ).unset                  # nothing applied yet
        applied = store.apply_saved(environ)
        assert applied.applied and not applied.pending, applied.message
        assert ss.workspaces_root(environ).value == folders["workspaces"].resolve()
        assert ds.read_saved(environ).pending == ()
        assert store.apply_saved(environ).applied is False          # nothing pending now

    def test_the_file_is_private_and_its_new_directory_too(self, store, tmp_path, folders):
        import stat as stat_module

        path = tmp_path / "fresh" / "config" / "deployment.json"
        assert store.save({"registry": str(folders["registry"])},
                          {ENV_SETTINGS_FILE: str(path)}).saved
        assert stat_module.S_IMODE(path.stat().st_mode) == 0o600
        assert stat_module.S_IMODE(path.parent.stat().st_mode) == 0o700
        assert stat_module.S_IMODE(path.parent.parent.stat().st_mode) == 0o700

    def test_one_refused_value_refuses_the_whole_save_with_a_reason_per_field(
            self, store, saved_file, folders):
        path, environ = saved_file
        outcome = store.save({"workspaces_root": str(folders["workspaces"]),
                              "registry": str(REPO_ROOT / "data"),
                              "intake_dir": "relative/intake", "colour": "blue"}, environ)
        assert not outcome.saved and not path.exists()
        assert set(outcome.refusals) == {"registry", "intake_dir", "colour"}
        assert "inside the repository" in outcome.refusals["registry"]
        assert "not an absolute path" in outcome.refusals["intake_dir"]

    def test_a_save_keeps_other_keys_and_unknown_ones(self, store, saved_file, folders):
        import json

        path, environ = saved_file
        _write_saved(path, {"registry": str(folders["registry"]), "future": 1},
                     saved={"registry": str(folders["registry"]), "future": 1}, extra_top="kept")
        assert store.save({"intake_dir": str(folders["intake"])}, environ).saved
        document = json.loads(path.read_text())
        assert document["extra_top"] == "kept" and document["saved"]["future"] == 1
        assert document["saved"]["registry"] == str(folders["registry"])
        assert document["saved"]["intake_dir"] == str(folders["intake"])
        assert store.save({"registry": None}, environ).saved          # cleared
        assert "registry" not in json.loads(path.read_text())["saved"]

    def test_an_unreadable_file_is_never_overwritten_and_is_renamed_only_by_repair(
            self, store, saved_file, folders):
        path, environ = saved_file
        path.parent.mkdir(parents=True)
        path.write_text("{ not json")
        for outcome in (store.save({"registry": str(folders["registry"])}, environ),
                        store.apply_saved(environ)):
            assert "cannot be read" in outcome.message
        assert path.read_text() == "{ not json"
        repaired = store.replace_unreadable(environ)
        assert repaired.replaced and repaired.kept.read_text() == "{ not json"
        assert repaired.kept.name.startswith("deployment.json.unreadable-")
        assert str(repaired.kept) in repaired.message and not path.exists()
        assert store.save({"registry": str(folders["registry"])}, environ).saved
        assert store.replace_unreadable(environ).replaced is False      # readable now

    def test_a_file_that_cannot_be_opened_is_never_renamed(self, store, saved_file,
                                                           monkeypatch):
        path, environ = saved_file
        path.parent.mkdir(parents=True)
        path.write_text("{}")
        real = Path.read_bytes
        monkeypatch.setattr(Path, "read_bytes", lambda self: (
            (_ for _ in ()).throw(PermissionError(13, "Permission denied", str(self)))
            if self == path else real(self)))
        outcome = store.replace_unreadable(environ)
        assert not outcome.replaced and "never renamed" in outcome.message
        assert path.exists() and list(path.parent.glob("*.unreadable-*")) == []

    def test_an_apply_waits_while_a_gpu_lock_the_old_values_name_is_held(
            self, store, saved_file, folders):
        """A command started by hand holds the lock under the applied root:
        the change stays pending, saying why, and applies once it ends."""
        from src.utils.gpu_lock import GpuLock

        path, environ = saved_file
        _write_saved(path, {"workspaces_root": str(folders["workspaces"])},
                     saved={"workspaces_root": str(folders["other"])})
        old_lock = folders["workspaces"].resolve() / ss.GPU_LOCK_FILE_NAME
        with GpuLock(old_lock, job_type="training", command="train_model.py by hand").acquire():
            outcome = store.apply_saved(environ)
            assert not outcome.applied and outcome.pending
            assert "train_model.py by hand" in outcome.message
            assert "press Apply then, or it applies at the next start" in outcome.message
            assert ss.workspaces_root(environ).value == folders["workspaces"].resolve()
        assert store.apply_saved(environ).applied
        assert ss.workspaces_root(environ).value == folders["other"].resolve()

    def test_an_apply_waits_for_the_lock_in_effect_too(self, store, saved_file, folders):
        """With the root set at start, the registry still comes from the file:
        a job under the environment's lock keeps it until it ends."""
        from src.utils.gpu_lock import GpuLock

        path, environ = saved_file
        env = {**environ, ss.ENV_WORKSPACES_ROOT: str(folders["workspaces"])}
        _write_saved(path, {"registry": str(folders["registry"])},
                     saved={"registry": str(folders["other"])})
        lock = folders["workspaces"].resolve() / ss.GPU_LOCK_FILE_NAME
        with GpuLock(lock, job_type="test", command="measure_scorer.py").acquire():
            assert store.apply_saved(env).pending
        assert store.apply_saved(env).applied

    def test_a_value_the_service_start_up_setting_overrides_is_not_applied_there(
            self, store, saved_file, folders):
        """T03 review (P2): a service started with the workspaces root in its
        environment saves another folder in Settings. Applied there, the
        service would keep its own lock while a command run by hand with no
        variable took the saved folder's: both held at once. The overridden
        key is held back (pending, its applied value kept); the others apply;
        the hand command's lock is unchanged (refused, as before the save)."""
        from src.utils.gpu_lock import GpuLockNotConfigured, take_gpu_lock

        path, environ = saved_file
        service = {**environ, ss.ENV_WORKSPACES_ROOT: str(folders["workspaces"])}
        hand = dict(environ)
        assert ss.gpu_lock_location(hand).unset
        assert store.save({"workspaces_root": str(folders["other"]),
                           "registry": str(folders["registry"])}, service).saved
        outcome = store.apply_saved(service)
        assert outcome.applied and outcome.pending, outcome.message
        assert "a start-up setting sets the workspaces folder for this service" in outcome.message
        saved = ds.read_saved(environ)
        assert saved.pending == ("workspaces_root",)
        assert "workspaces_root" not in saved.applied
        assert ss.cohort_registry(hand).value == folders["registry"].resolve()
        assert ss.gpu_lock_location(hand).unset                 # not moved to the saved folder
        assert ss.gpu_lock_location(service).value == (folders["workspaces"].resolve()
                                                       / ss.GPU_LOCK_FILE_NAME)
        with take_gpu_lock("training", "the service's job", environ=service) as job:
            assert job.held
            with pytest.raises(GpuLockNotConfigured):
                take_gpu_lock("training", "train_model.py by hand", environ=hand)
        again = store.apply_saved(service)                      # the startup Apply too
        assert not again.applied and again.pending
        assert "applies at a start without that setting" in again.message
        # A start without the setting applies it, as for any pending change.
        assert store.apply_saved(hand).applied
        assert ss.gpu_lock_location(hand).value == (folders["other"].resolve()
                                                    / ss.GPU_LOCK_FILE_NAME)

    def test_a_saved_value_equal_to_the_start_up_setting_is_applied(
            self, store, saved_file, folders):
        """The same folder (compared as a path) gives every process one lock,
        so it is applied as usual, and a list of evidence folders alike."""
        path, environ = saved_file
        service = {**environ, ss.ENV_WORKSPACES_ROOT: str(folders["workspaces"]) + "/",
                   ss.ENV_TRAINING_EVIDENCE_DIRS: os.pathsep.join(
                       [str(folders["evidence-a"]), "", str(folders["evidence-b"])])}
        assert store.save({"workspaces_root": str(folders["workspaces"]),
                           "training_evidence_dirs": [str(folders["evidence-a"]),
                                                      str(folders["evidence-b"])]},
                          service).saved
        outcome = store.apply_saved(service)
        assert outcome.applied and not outcome.pending, outcome.message
        assert ss.gpu_lock_location(environ).value == ss.gpu_lock_location(service).value

    def test_a_crash_between_the_write_and_the_replace_leaves_the_old_file(
            self, saved_file, folders):
        """The one writer stages the new bytes beside the file and replaces it
        last: a process killed before the replace leaves the old file whole
        (and a stray staging file, which no reader opens)."""
        import subprocess
        import sys

        path, environ = saved_file
        _write_saved(path, {"registry": str(folders["registry"])})
        old = path.read_bytes()
        script = (
            "import os, sys\n"
            "from src.config import atomic_json\n"
            "os.replace = lambda *a: os._exit(9)\n"
            "atomic_json.write_json_atomic(sys.argv[1], {'format': 'x', 'applied': {}})\n")
        done = subprocess.run([sys.executable, "-c", script, str(path)], cwd=REPO_ROOT,
                              capture_output=True, timeout=120)
        assert done.returncode == 9, done.stderr
        assert path.read_bytes() == old
        assert ss.cohort_registry(environ).value == folders["registry"].resolve()
        assert [p.name for p in path.parent.iterdir() if p.name.endswith(".tmp")]

    def test_a_failed_write_removes_its_staging_file(self, saved_file, folders, monkeypatch):
        from src.config import atomic_json

        path, environ = saved_file
        _write_saved(path, {"registry": str(folders["registry"])})
        old = path.read_bytes()

        def refuse(*args):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(atomic_json.os, "replace", refuse)
        with pytest.raises(OSError):
            atomic_json.write_json_atomic(path, {"format": FORMAT})
        assert path.read_bytes() == old and sorted(path.parent.iterdir()) == [path]

    def test_concurrent_saves_never_lose_a_value(self, store, saved_file, folders):
        """Two pages saving at once: each read-modify-write is under the
        file's lock, so both values are kept and the file is always whole."""
        import threading

        path, environ = saved_file
        keys = {"workspaces_root": "workspaces", "registry": "registry", "intake_dir": "intake"}
        errors = []

        def save(key, folder):
            for _ in range(15):
                outcome = store.save({key: str(folders[folder])}, environ)
                if not outcome.saved:
                    errors.append(outcome.message)

        threads = [threading.Thread(target=save, args=item) for item in keys.items()]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        assert errors == []
        read = ds.read_file(path)
        assert read.problem is None
        assert read.saved == {key: str(folders[folder]) for key, folder in keys.items()}


class TestATakerFollowsAnApply:
    """The GPU lock's location is read again once the lock is held, so a
    taker that read the old root just before an apply never runs under it."""

    def test_a_location_changed_while_it_was_taken_is_followed(self, tmp_path, monkeypatch):
        from src.utils import gpu_lock

        old, new = tmp_path / "old.lock", tmp_path / "new.lock"
        answers = iter([old, new, new, new])
        monkeypatch.setattr(gpu_lock, "lock_location", lambda environ=None: next(answers))
        with gpu_lock.take_gpu_lock("training", "train_model.py") as lock:
            assert lock.path == new and gpu_lock.probe(new) is not None
            assert gpu_lock.probe(old) is None, "the old lock was kept"

    def test_a_location_that_keeps_changing_is_refused(self, tmp_path, monkeypatch):
        from src.utils import gpu_lock

        counter = iter(range(100))
        monkeypatch.setattr(gpu_lock, "lock_location",
                            lambda environ=None: tmp_path / f"{next(counter)}.lock")
        with pytest.raises(gpu_lock.GpuLockMisconfigured, match="changed 3 times"):
            gpu_lock.take_gpu_lock("training", "train_model.py")
        assert all(gpu_lock.probe(p) is None for p in tmp_path.glob("*.lock"))


class TestAChildAndAHandCommandReadTheSameSettings:

    def test_a_process_without_the_variables_reads_the_applied_values_and_lock(
            self, saved_file, folders, monkeypatch):
        """The job runner's children and a shell of the same account: no
        `SHEPHERD_*` data location in their environment, the applied values
        and the GPU lock of the saved file."""
        import json
        import subprocess
        import sys

        path, environ = saved_file
        _write_saved(path, {"workspaces_root": str(folders["workspaces"]),
                            "registry": str(folders["registry"])},
                     saved={"workspaces_root": str(folders["other"])})
        env = {k: v for k, v in os.environ.items() if not k.startswith("SHEPHERD_")}
        env[ENV_SETTINGS_FILE] = str(path)
        code = ("import json\n"
                "from src.config import server_settings as ss\n"
                "from src.utils.gpu_lock import lock_location\n"
                "print(json.dumps([str(ss.workspaces_root().value), ss.workspaces_root().source,"
                " str(ss.cohort_registry().value), str(lock_location())]))\n")
        done = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env,
                              capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr
        root, source, registry, lock = json.loads(done.stdout.strip().splitlines()[-1])
        assert (root, source) == (str(folders["workspaces"].resolve()), "saved")
        assert registry == str(folders["registry"].resolve())
        assert lock == str(folders["workspaces"].resolve() / ss.GPU_LOCK_FILE_NAME)


class TestTheOneViewOfTheDataLocations:
    """`src/utils/deployment_view.py`: Settings, ``GET
    /api/v1/settings/deployment`` and ``scripts/deployment_settings.py show``
    say the same thing about each location, read through the one resolution
    (UI-06, item 10)."""

    def test_each_location_says_what_is_in_effect_from_where_and_what_is_saved(
            self, saved_file, folders):
        from src.utils.deployment_view import deployment_view

        path, environ = saved_file
        _write_saved(path, {"workspaces_root": str(folders["workspaces"]),
                            "registry": str(folders["registry"])},
                     saved={"workspaces_root": str(folders["other"]),
                            "registry": str(folders["registry"])})
        environ = {**environ, ss.ENV_REGISTRY: str(folders["intake"])}
        view = deployment_view(environ)
        root = view.field("workspaces_root")
        assert (root.value, root.source) == (str(folders["workspaces"].resolve()), "saved")
        assert root.pending and root.saved == str(folders["other"]) and not root.overridden
        registry = view.field("registry")
        assert (registry.value, registry.source) == (str(folders["intake"].resolve()),
                                                     "environment")
        assert registry.overridden and not registry.pending
        intake = view.field("intake_dir")
        assert intake.unset and intake.source is None and intake.value is None
        assert view.pending == ("workspaces_root",)
        assert view.pending_reason == "Press Apply, or it applies at the next start."
        assert view.gpu_lock.value == folders["workspaces"].resolve() / ss.GPU_LOCK_FILE_NAME
        assert not view.nothing_set

    def test_a_held_back_change_is_said_as_such_and_not_offered_to_apply(
            self, saved_file, folders):
        """A pending change a start-up setting of this process holds back is
        named (`held`), and the reason says so, not "press Apply"."""
        from src.utils.deployment_view import as_dict, deployment_view

        path, environ = saved_file
        _write_saved(path, {}, saved={"workspaces_root": str(folders["other"])})
        service = {**environ, ss.ENV_WORKSPACES_ROOT: str(folders["workspaces"])}
        view = deployment_view(service)
        assert view.pending == ("workspaces_root",) and view.held == ("workspaces_root",)
        assert view.applicable == ()
        assert view.pending_reason == (
            "Saved, not applied yet: a start-up setting sets the workspaces folder for this "
            "service, so the saved value applies at a start without that setting.")
        assert as_dict(view, paths_shown=False)["held_back"] == ["workspaces_root"]
        alone = deployment_view(environ)                        # no start-up setting here
        assert alone.held == () and alone.applicable == ("workspaces_root",)
        assert alone.pending_reason == "Press Apply, or it applies at the next start."

    def test_a_refused_value_is_shown_with_what_was_set_never_as_not_set(
            self, saved_file, tmp_path):
        from src.utils.deployment_view import deployment_view

        path, environ = saved_file
        gone = tmp_path / "gone"
        _write_saved(path, {"registry": str(gone)})
        registry = deployment_view(environ).field("registry")
        assert registry.value is None and registry.configured == str(gone)
        assert registry.source == "saved" and not registry.unset
        assert "saved in Settings is refused" in registry.problem

    def test_hidden_paths_leave_what_is_set_and_from_where(self, saved_file, folders):
        import json

        from src.utils.deployment_view import as_dict, deployment_view

        path, environ = saved_file
        _write_saved(path, {"registry": str(folders["registry"]) + "/../registry"})
        answer = as_dict(deployment_view(environ), paths_shown=False)
        said = json.dumps(answer)
        assert str(folders["registry"].parent) not in said and str(path) not in said
        registry = next(f for f in answer["fields"] if f["key"] == "registry")
        assert registry["set"] is True and registry["source"] == "saved"
        assert registry["in_effect"] is None and registry["saved"] is None

    def test_a_refused_value_is_never_quoted_to_a_viewer_whose_paths_are_hidden(
            self, saved_file):
        """T03 review (P3): a refusal quotes the value as given, and the one
        redaction finds an absolute path only, so a relative or ``~`` value
        was shown to a read-only request ("the registry 'clinic/MRN
        0012345/registry' is not an absolute path"). Hidden, a refused value
        is said in fixed words; an operator still reads the refusal's own."""
        import json

        from src.utils.deployment_view import as_dict, deployment_view

        path, environ = saved_file
        _write_saved(path, {"workspaces_root": "rel/ws"})
        environ = {**environ, ss.ENV_REGISTRY: "clinic/MRN 0012345/registry",
                   ss.ENV_INTAKE_DIR: "~/intake/Jane Doe",
                   ss.ENV_TRAINING_EVIDENCE_DIRS: "ev/MRN 99"}
        view = deployment_view(environ)
        hidden = as_dict(view, paths_shown=False)
        said = json.dumps(hidden, ensure_ascii=False)
        for given in ("rel/ws", "clinic", "MRN", "Jane", "Doe", "ev/"):
            assert given not in said, given
        problems = {f["key"]: f["problem"] for f in hidden["fields"]}
        assert problems["registry"] == ("Results registry set when the service started is "
                                        "refused: an operator sees why in Settings › Data "
                                        "locations.")
        assert problems["workspaces_root"].startswith("Workspaces folder saved in Settings is "
                                                      "refused")
        assert problems["training_evidence_dirs"].startswith(
            "Training evidence folders set when the service started are refused")
        assert hidden["gpu_lock"]["problem"].startswith("The GPU lock cannot be used")
        assert all(f["set"] for f in hidden["fields"])
        shown = {f["key"]: f["problem"] for f in as_dict(view, paths_shown=True)["fields"]}
        assert "'clinic/MRN 0012345/registry' is not an absolute path" in shown["registry"]

    def test_a_settings_file_named_at_start_and_refused_is_never_quoted_either(self):
        """A relative `SHEPHERD_SETTINGS_FILE` is refused in words that quote it,
        and every location that would come from it repeats them."""
        import json

        from src.utils.deployment_view import as_dict, deployment_view

        view = deployment_view({ENV_SETTINGS_FILE: "clinic/MRN 7/deployment.json"})
        assert view.file.problem and "MRN 7" in view.file.problem
        said = json.dumps(as_dict(view, paths_shown=False), ensure_ascii=False)
        assert "MRN" not in said and "clinic" not in said
        answer = as_dict(view, paths_shown=False)
        assert answer["settings_file"]["problem"].startswith(
            "The settings file named when the service started is refused")
        root = next(f for f in answer["fields"] if f["key"] == "workspaces_root")
        assert root["problem"].startswith("Workspaces folder is not known")

    @pytest.mark.parametrize("case", ["mode not set", "mode not M1", "no workspaces folder",
                                      "workspaces folder refused", "settings file refused",
                                      "held back"])
    def test_the_text_form_says_every_reason_the_json_form_carries(self, saved_file, folders,
                                                                   tmp_path, case):
        """T08 flow (T08-flow-8): ``deployment_settings.py show`` read
        "Deployment mode: read-only (source: not set)" with no reason, while
        its ``--json`` (the route's view) said why ("They turn on when the
        service is started from its desktop entry or launch_shepherd.sh on this
        computer."); the GPU lock's problem was dropped the same way ("GPU lock:
        not configured"). The one view says the same facts in each form: every
        reason the JSON form carries (a problem, a note, why a change waits) is
        in the text form."""
        from src.utils.deployment_view import as_dict, deployment_view, text_lines

        path, environ = saved_file
        _write_saved(path, {"workspaces_root": str(folders["workspaces"])})
        if case == "mode not M1":
            environ = {**environ, ss.ENV_DEPLOYMENT_MODE: "read-only"}
        elif case == "no workspaces folder":
            _write_saved(path, {})
        elif case == "workspaces folder refused":
            environ = {**environ, ss.ENV_WORKSPACES_ROOT: str(tmp_path / "gone")}
        elif case == "settings file refused":
            environ = {ENV_SETTINGS_FILE: "clinic/deployment.json"}
        elif case == "held back":
            _write_saved(path, {}, saved={"workspaces_root": str(folders["other"])})
            environ = {**environ, ss.ENV_WORKSPACES_ROOT: str(folders["workspaces"])}
        view = deployment_view(environ)
        answer = as_dict(view, paths_shown=True)
        assert answer["deployment_mode"]["problem"], "each case is read in a plain shell"
        reasons = [answer["settings_file"]["problem"], answer["deployment_mode"]["problem"],
                   answer["gpu_lock"]["problem"], answer["pending_reason"],
                   *(field[key] for field in answer["fields"] for key in ("problem", "note"))]
        text = "\n".join(text_lines(view))
        for reason in filter(None, reasons):
            assert reason in text, reason

    def test_the_cli_prints_the_same_view_a_service_reads(self, saved_file, folders):
        """A shell of the account, with no `SHEPHERD_*` data location set, and
        the service read the same values: the CLI's ``--json`` is the route's
        view for this environment, and its text names each source."""
        import json
        import subprocess
        import sys

        from src.utils.deployment_view import as_dict, deployment_view

        path, environ = saved_file
        _write_saved(path, {"workspaces_root": str(folders["workspaces"])})
        env = {k: v for k, v in os.environ.items() if not k.startswith("SHEPHERD_")}
        env[ENV_SETTINGS_FILE] = str(path)
        script = str(REPO_ROOT / "scripts" / "deployment_settings.py")
        done = subprocess.run([sys.executable, script, "show", "--json"], cwd="/", env=env,
                              capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr
        assert json.loads(done.stdout) == as_dict(deployment_view(environ), paths_shown=True)
        text = subprocess.run([sys.executable, script, "show"], cwd="/", env=env,
                              capture_output=True, text=True, timeout=120)
        assert text.returncode == 0, text.stderr
        assert f"in effect: {folders['workspaces'].resolve()}" in text.stdout
        assert "source:    saved settings file (applied)" in text.stdout
        assert f"Settings file: {path} (named by SHEPHERD_SETTINGS_FILE)" in text.stdout


# ---------------------------------------------------------------------------
# The notices: a setting's state in parts, the whole sentences from the same parts
# ---------------------------------------------------------------------------
#: The whole sentences the CLI, the API and the logs say, byte for byte (the
#: notices add parts; these never change: the UI sweep of 2026-11-22).
WHOLE = {
    "WORKSPACES_ROOT_NOT_SET": (
        "No workspaces folder is set, so the server lists no workspace. Choose the folder whose "
        "subfolders are the workspaces in Settings › Data locations; its Create suggested folder "
        "makes SHEPHERD-data/workspaces in the service account's home folder. No default is "
        "assumed. Ref: SHEPHERD_WORKSPACES_ROOT"),
    "GPU_LOCK_NOT_CONFIGURED": (
        "No workspaces folder is set, so the GPU lock is not configured and every managed GPU "
        "job is refused rather than run without excluding the others. Choose the folder in "
        "Settings › Data locations: the lock is .shepherd-gpu.lock in it, the one file every "
        "process that uses the folder takes (deployment guide §4.11). Ref: "
        "SHEPHERD_WORKSPACES_ROOT"),
    "REGISTRY_NOT_SET": (
        "No results registry is set, so the features that read the cohort registry are off. "
        "Choose it in Settings › Data locations: a folder outside the repository, the one the "
        "commands take as --registry. Ref: SHEPHERD_REGISTRY"),
    "INTAKE_DIR_NOT_SET": (
        "No import folder is set, so no import source is offered. Choose it in Settings › Data "
        "locations: a folder outside the repository where you place shepherd-cases/1 files. "
        "Ref: SHEPHERD_INTAKE_DIR"),
}
#: Each notice beside its whole sentence, and the facts the sentence says that
#: its parts must say too.
NOTICE_OF = {
    "WORKSPACES_ROOT_NOT_SET": ("WORKSPACES_ROOT_NOTICE", (
        "No workspaces folder is set", "Settings › Data locations",
        "SHEPHERD-data/workspaces in the service account's home folder",
        "Create suggested folder", "the server lists no workspace", "SHEPHERD_WORKSPACES_ROOT")),
    "GPU_LOCK_NOT_CONFIGURED": ("GPU_LOCK_NOTICE", (
        "Settings › Data locations", ".shepherd-gpu.lock", "§4.11", "SHEPHERD_WORKSPACES_ROOT",
        "every process that uses the folder")),
    "REGISTRY_NOT_SET": ("REGISTRY_NOTICE", (
        "No results registry is set", "Settings › Data locations",
        "a folder outside the repository", "--registry", "SHEPHERD_REGISTRY")),
    "INTAKE_DIR_NOT_SET": ("INTAKE_DIR_NOTICE", (
        "No import folder is set", "Settings › Data locations", "no import source is offered",
        "a folder outside the repository", "shepherd-cases/1", "SHEPHERD_INTAKE_DIR")),
}


def _parts(notice) -> str:
    return "\n".join((notice.summary, notice.short, notice.action, *notice.details, *notice.ref))


@pytest.mark.parametrize("name", sorted(WHOLE))
def test_each_whole_sentence_is_unchanged_and_its_notice_says_every_fact(name):
    """The CLI's and the API's sentence is kept byte for byte, built from the
    parts its notice shares (the folder's state, Data locations, the suggested
    folder, the lock file, the guide section, the variable, ``--registry``),
    and every one of those facts is in the notice's own parts too: the WebUI's
    short form drops none of them, it moves them behind "Details ›"."""
    assert getattr(ss, name) == WHOLE[name]
    notice_name, facts = NOTICE_OF[name]
    parts = _parts(getattr(ss, notice_name))
    for fact in facts:
        assert fact.lower() in parts.lower(), (name, fact)
        assert fact in WHOLE[name] or fact.lower() in WHOLE[name].lower(), (name, fact)
    # The readers still answer with the whole sentence.
    assert ss.workspaces_root({}).problem == WHOLE["WORKSPACES_ROOT_NOT_SET"]
    assert ss.gpu_lock_location({}).problem == WHOLE["GPU_LOCK_NOT_CONFIGURED"]
    assert ss.cohort_registry({}).problem == WHOLE["REGISTRY_NOT_SET"]
    assert ss.intake_directory({}).problem == WHOLE["INTAKE_DIR_NOT_SET"]


def test_the_notices_parts_are_short_front_loaded_and_name_no_server_path():
    """The text brief's message shape: a summary of one front-loaded sentence
    (its first two words carry the state) of 60 characters or fewer; a short
    form a relation reads ("Start disabled: no workspaces folder is
    set."); the next step by its place in the GUI; the details as term rows,
    each term said once across the three notices (one screen may merge them);
    the codes only in the Ref items; and no server path anywhere."""
    notices = (ss.WORKSPACES_ROOT_NOTICE, ss.GPU_LOCK_NOTICE, ss.REGISTRY_NOTICE, ss.INTAKE_DIR_NOTICE)
    assert [n.key for n in notices] == ["workspaces_root", "gpu_lock", "registry", "intake_dir"]
    assert ss.GPU_LOCK_NOTICE.cause == "workspaces_root"
    assert ss.WORKSPACES_ROOT_NOTICE.cause is None and ss.REGISTRY_NOTICE.cause is None
    assert ss.INTAKE_DIR_NOTICE.cause is None
    terms = []
    for n in notices:
        assert n.summary.endswith(".") and len(n.summary) <= 60, n.summary
        assert n.summary.split()[:2] in (["No", "workspaces"], ["No", "results"], ["GPU", "jobs"],
                                         ["No", "import"])
        assert n.short == n.short.strip() and not n.short.endswith(".") and len(n.short) <= 40
        assert n.short[0].islower() or n.short.startswith("GPU"), n.short
        assert n.action == "Choose one in Settings › Data locations"
        assert n.target == "settings#shep-dl"
        for row in n.details:
            term, _, text = row.partition(": ")
            assert term and text and term[0].isupper() and len(term) <= 32, row
            assert not text.endswith(".") and ";" not in row, row
            terms.append(term)
        assert n.ref and all("SHEPHERD_" not in row for row in (n.summary, n.short, n.action))
        for part in (n.summary, n.short, n.action, *n.details, *n.ref):
            assert "/home" not in part and not part.startswith("/") and " /" not in part, part
    assert len(terms) == len(set(terms)), terms
