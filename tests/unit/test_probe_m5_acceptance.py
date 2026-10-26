"""The deployment probe's M5 acceptance: its parts that are not phases.

The phases themselves (W1-W11 in phase M, RW1-RW3 in phase R) run end to end in
tests/unit/test_probe_m2_phase.py. This shows, without starting a service or a
job, that what they rest on holds and that their checks fire:

- every M5 probe is required, so a missing one leaves the acceptance incomplete;
- A5 records the installed packages against the lock beside the checkout and
  states a set that differs (the homelab's: huggingface-hub 1.33.0, click
  8.5.0, hf-xet 1.7.0, aiofiles and ffmpy left over), never hides it: a
  package installed twice, the deploy scripts' packages told from leftovers,
  whether the lock is the committed one, all said again at the end of a run;
  without a lock it is skipped, so an acceptance is incomplete;
- a temporary directory in which a DataLoader worker's socket cannot be made
  is said, as the Python that runs the jobs answers it, and RW3 stops early;
- the M5 settings are the probe's own while it runs and are put back, so an
  operator's shell holding the deployment's settings changes nothing; the
  deployment's own lock is read and stated, once in each phase and as it is
  at that moment, as the probe's docstring says (verification of INT-2, P-2);
- the service is started as the launcher starts it, with the probe's settings,
  and ends with the probe; SIGTERM ends the probe through its cleanups;
- D8's, CORS's and the route guard's checks each fail on the answer they exist
  to catch; every gated API route and the gated dashboard events are found;
  the URL-fetch count fails when the guard is gone;
- a training stop is required not to reach a test run;
- W11's and RW3's training job must run and be the lock's recorded holder
  before a test start is tried and after the measurement CLI is, one bracket
  around the whole check, shown on the real job runner: a job that ended at
  any step fails naming the action during which it ended and how the runner
  says it ended, never as a start, a cancel, a stop reaching training or the
  lock failing to refuse; a test start, a test cancel or the CLI after which
  the runner says training was stopped fails as that action reaching it; a
  test run a start not refused began is stopped; the job is held by
  configuration in the one function that starts it, whatever its settings
  say, and without os.waitid no job is started (review of 8242b47, W11/RW3;
  root-cause round, W11-1, W11-2, W11-4, and its verification);
- phases M and R name the probe's own GPU lock, and the operator's lock is
  still stated and its settings put back (P2-4).

Module: tests/unit/test_probe_m5_acceptance.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tests.fixtures.job_processes import live  # noqa: E402


def _probe():
    spec = importlib.util.spec_from_file_location(
        "probe_deployment", REPO / "scripts" / "probe_deployment.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


@pytest.fixture(scope="module")
def probe():
    return _probe()


# --------------------------------------------------------------------------- required
def test_every_m5_probe_is_required_and_a_missing_one_leaves_its_part_incomplete(probe):
    assert {"A5", *probe.M5_DEMO_PROBES} <= set(probe.M2_DEMO_REQUIRED)
    assert set(probe.M5_REAL_PROBES) <= set(probe.M2_REAL_REQUIRED)
    assert probe.M5_DEMO_PROBES == tuple(f"W{i}" for i in range(1, 12))
    assert len(set(probe.M2_DEMO_REQUIRED)) == len(probe.M2_DEMO_REQUIRED)
    assert len(set(probe.M2_REAL_REQUIRED)) == len(probe.M2_REAL_REQUIRED)
    report = probe.Report()
    for probe_id in (*probe.M2_DEMO_REQUIRED, *probe.M2_REAL_REQUIRED):
        report.probes.append(probe.Probe(probe_id, "m2", "x", status="passed"))
    assert probe.m2_acceptance(report, True)["complete"]
    report.probes = [p for p in report.probes if p.probe_id not in ("W4", "RW3")]
    acceptance = probe.m2_acceptance(report, True)
    assert acceptance["demo"]["not_passed"] == {"W4": "not run"}
    assert acceptance["real_data"]["not_passed"] == {"RW3": "not run"}
    assert "M5" in acceptance["scope"] and "no long training" in acceptance["scope"]


# --------------------------------------------------------------------------- A5
def _lock(path: Path, packages) -> Path:
    lines = ["version = 1", 'requires-python = ">=3.12"', ""]
    for name, version in packages:
        lines += ["[[package]]", f'name = "{name}"', f'version = "{version}"', ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _installed(monkeypatch, distributions):
    """Fake installed distributions: (name, version) or (name, version,
    requires)."""
    import importlib.metadata

    fakes = [SimpleNamespace(metadata={"Name": entry[0]}, version=entry[1],
                             requires=list(entry[2]) if len(entry) > 2 else None)
             for entry in distributions]
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: iter(fakes))


def test_a_set_that_differs_from_the_lock_is_stated_never_hidden(probe, tmp_path, monkeypatch):
    """The homelab's check set against the committed lock (M5-0): three
    versions moved, two leftovers of Gradio 5.29.1, torch_geometric at another
    version than the lock's (BACKLOG 50) with what it requires, a native
    extension the deploy scripts installed, an extra not installed, and a stale
    second copy of gradio. A package the lock pins at two versions (one per
    platform) matches either; versions compare as PEP 440 does."""
    lock = _lock(tmp_path / "uv.lock", [
        ("huggingface-hub", "1.16.1"), ("click", "8.3.3"), ("hf-xet", "1.5.0"),
        ("gradio", "6.30.0"), ("numpy", "2.3.4"), ("numpy", "2.2.6"),
        ("torch", "2.10.0+cu130"), ("tqdm", "4.67.1"), ("spacy", "3.8.14"),
        ("torch-geometric", "2.8.1")])
    _installed(monkeypatch, [
        ("huggingface_hub", "1.33.0"), ("Click", "8.5.0"), ("hf_xet", "1.7.0"),
        ("gradio", "6.30.0"), ("numpy", "2.2.6", ["numpy-helper"]), ("numpy_helper", "1.0"),
        ("torch", "2.10.0+cu130"),
        ("tqdm", "4.67.1.0"), ("aiofiles", "24.1.0"), ("ffmpy", "1.0.0", ["old-helper"]),
        ("torch_geometric", "2.8.0.post1",
         ["aiohttp", "numpy", "xxhash; python_version >= '3'", "pytest; extra == 'test'"]),
        ("aiohttp", "3.13.0", ["Multidict>=4.5"]), ("multidict", "6.7.0"),
        ("xxhash", "3.6.0"), ("pytest", "9.0.0"), ("old_helper", "0.1"),
        ("pyg_lib", "0.6.0"),
        ("gradio", "5.29.1")])            # a stale .dist-info beside the current one

    recorded = probe.installed_against_lock(lock)

    assert recorded["lock_sha256"] == hashlib.sha256(lock.read_bytes()).hexdigest()
    assert recorded["lock_is_the_committed_one"] is None          # not in a checkout
    assert recorded["versions_match_lock"] is False
    assert recorded["version_differs"] == [
        {"name": "click", "installed": "8.5.0", "locked": ["8.3.3"]},
        {"name": "gradio", "installed": "5.29.1", "locked": ["6.30.0"]},
        {"name": "hf-xet", "installed": "1.7.0", "locked": ["1.5.0"]},
        {"name": "huggingface-hub", "installed": "1.33.0", "locked": ["1.16.1"]},
        {"name": "torch-geometric", "installed": "2.8.0.post1", "locked": ["2.8.1"]},
    ]
    assert recorded["installed_more_than_once"] == [
        {"name": "gradio", "versions": ["5.29.1", "6.30.0"]}]
    deploy, required, leftover = (probe.OUTSIDE_LOCK_DEPLOY, probe.OUTSIDE_LOCK_REQUIRED,
                                  probe.OUTSIDE_LOCK_LEFTOVER)
    assert recorded["installed_not_in_lock"] == [
        {"name": "aiofiles", "version": "24.1.0", "why": leftover},
        # A locked package's requirements: the deploy scripts' closure does not
        # pass through what the sync installed.
        {"name": "aiohttp", "version": "3.13.0", "why": required},
        {"name": "ffmpy", "version": "1.0.0", "why": leftover},
        {"name": "multidict", "version": "6.7.0", "why": required},
        {"name": "numpy-helper", "version": "1.0", "why": required},
        {"name": "old-helper", "version": "0.1", "why": required},
        {"name": "pyg-lib", "version": "0.6.0", "why": deploy},
        {"name": "pytest", "version": "9.0.0", "why": leftover},     # only an extra asks
        {"name": "xxhash", "version": "3.6.0", "why": required}]
    assert recorded["locked_not_installed"] == ["spacy"]
    assert recorded["declared_not_installed"] == []      # this lock names no project
    assert (recorded["lock_packages"], recorded["installed_distributions"]) == (9, 17)
    assert str(tmp_path) not in json.dumps(recorded)

    said = probe.lock_sentence(recorded)
    assert "5 version(s) differ (click 8.5.0, lock 8.3.3, gradio 5.29.1, lock 6.30.0" in said
    assert "torch-geometric 2.8.0.post1, lock 2.8.1" in said
    assert "installed more than once: gradio (5.29.1, 6.30.0)" in said
    assert "9 installed package(s) are not in the lock: 1 from the deploy scripts" in said
    assert "required by nothing installed: aiofiles 24.1.0, ffmpy 1.0.0, pytest 9.0.0" in said
    assert ("required by another installed package: aiohttp 3.13.0, multidict 6.7.0, "
            "numpy-helper 1.0, old-helper 0.1, xxhash 3.6.0") in said
    assert "whether it is the committed one could not be read" in said


def test_a_second_installed_copy_is_stated_whatever_the_directory_order(
        probe, tmp_path, monkeypatch):
    """Two real .dist-info directories of one package in one site directory:
    importlib.metadata finds both in directory order; A5 compares both and
    says the package is installed twice (it kept the first one found, and
    said nothing of the other, before M5-3's review)."""
    from importlib import metadata

    site = tmp_path / "site-packages"
    for version in ("6.30.0", "5.29.1"):
        info = site / f"gradio-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: gradio\nVersion: {version}\n", encoding="utf-8")
    lock = _lock(tmp_path / "uv.lock", [("gradio", "6.30.0")])
    found = metadata.distributions
    monkeypatch.setattr(metadata, "distributions", lambda: found(path=[str(site)]))

    recorded = probe.installed_against_lock(lock)

    assert recorded["installed_more_than_once"] == [
        {"name": "gradio", "versions": ["5.29.1", "6.30.0"]}]
    assert recorded["version_differs"] == [
        {"name": "gradio", "installed": "5.29.1", "locked": ["6.30.0"]}]
    assert recorded["installed_distributions"] == 1


def _metadata_dir(path: Path, name: str, version: str, *, editable_of: Path = None) -> None:
    """A real metadata directory: a ``.dist-info`` (METADATA, and with
    `editable_of` the ``direct_url.json`` an editable install writes) or an
    ``.egg-info`` (PKG-INFO, as an editable build leaves one in ``src/``)."""
    path.mkdir(parents=True)
    text = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    (path / ("PKG-INFO" if path.suffix == ".egg-info" else "METADATA")).write_text(
        text, encoding="utf-8")
    if editable_of is not None:
        (path / "direct_url.json").write_text(json.dumps(
            {"url": editable_of.resolve().as_uri(), "dir_info": {"editable": True}}),
            encoding="utf-8")


def test_the_projects_editable_install_and_its_source_tree_metadata_are_one(
        probe, tmp_path, monkeypatch):
    """BACKLOG 59 (Batch 1's GB10 check): an editable install of this checkout
    and the ``src/<name>.egg-info`` its build leaves are one installation,
    stated in ``counted_once``; a stale second ``.dist-info`` of the project,
    a source-tree copy at another version and any other package found twice
    are still installed more than once."""
    from importlib import metadata

    checkout, site = tmp_path / "checkout", tmp_path / "env" / "site-packages"
    lock = checkout / "uv.lock"
    checkout.mkdir()
    lock.write_text("\n".join([
        "version = 1", 'requires-python = ">=3.12"', "",
        "[[package]]", 'name = "shepherd-advanced"', 'version = "0.1.0"',
        'source = { editable = "." }', "",
        "[[package]]", 'name = "gradio"', 'version = "6.30.0"', ""]), encoding="utf-8")
    _metadata_dir(site / "shepherd_advanced-0.1.0.dist-info", "shepherd-advanced", "0.1.0",
                  editable_of=checkout)
    _metadata_dir(checkout / "src" / "shepherd_advanced.egg-info", "shepherd-advanced", "0.1.0")
    for version in ("6.30.0", "5.29.1"):
        _metadata_dir(site / f"gradio-{version}.dist-info", "gradio", version)
    found = metadata.distributions
    paths = [str(site), str(checkout / "src")]
    monkeypatch.setattr(metadata, "distributions", lambda: found(path=paths))

    recorded = probe.installed_against_lock(lock)

    assert recorded["installed_more_than_once"] == [
        {"name": "gradio", "versions": ["5.29.1", "6.30.0"]}]
    assert recorded["counted_once"] == [
        {"name": "shepherd-advanced", "version": "0.1.0",
         "why": "its editable install and the metadata its build left in the checkout's "
                "source tree"}]
    assert "installed more than once: gradio (5.29.1, 6.30.0)" in probe.lock_sentence(recorded)
    assert str(tmp_path) not in json.dumps(recorded)

    # A source-tree copy at another version is stale metadata, still stated.
    shutil.rmtree(checkout / "src" / "shepherd_advanced.egg-info")
    _metadata_dir(checkout / "src" / "shepherd_advanced.egg-info", "shepherd-advanced", "0.0.9")
    stale = probe.installed_against_lock(lock)
    assert {"name": "shepherd-advanced", "versions": ["0.0.9", "0.1.0"]} in \
        stale["installed_more_than_once"] and stale["counted_once"] == []
    # So is a second installed .dist-info of the project beside the editable one.
    shutil.rmtree(checkout / "src" / "shepherd_advanced.egg-info")
    _metadata_dir(site / "shepherd_advanced-0.0.9.dist-info", "shepherd-advanced", "0.1.0")
    second = probe.installed_against_lock(lock)
    assert {"name": "shepherd-advanced", "versions": ["0.1.0", "0.1.0"]} in \
        second["installed_more_than_once"] and second["counted_once"] == []


def _git(cwd, *args):
    import subprocess

    subprocess.run(["git", "-c", "user.name=probe", "-c", "user.email=probe@example.invalid",
                    "-c", "commit.gpgsign=false", *args], cwd=cwd, check=True,
                   capture_output=True)


def test_whether_the_lock_read_is_the_committed_one_is_said(probe, tmp_path, monkeypatch):
    """A lock made again on the machine and not committed is not taken for the
    committed one, a line-ending conversion git would undo is no difference,
    and outside a checkout it cannot be said."""
    _installed(monkeypatch, [("gradio", "6.30.0")])
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    lock = _lock(checkout / "uv.lock", [("gradio", "6.30.0")])
    (checkout / ".gitattributes").write_text("* text=auto\n", encoding="utf-8")
    _git(checkout, "init", "-q")
    assert probe.installed_against_lock(lock)["lock_is_the_committed_one"] is None  # no commit
    (checkout / "README").write_text("x\n", encoding="utf-8")
    _git(checkout, "add", "README", ".gitattributes")
    _git(checkout, "commit", "-q", "-m", "no lock yet")
    assert probe.installed_against_lock(lock)["lock_is_the_committed_one"] is False
    _git(checkout, "add", "uv.lock")
    _git(checkout, "commit", "-q", "-m", "lock")
    recorded = probe.installed_against_lock(lock)
    assert recorded["lock_is_the_committed_one"] is True
    assert "(the one committed at the tested commit)" in probe.lock_sentence(recorded)
    lock.write_bytes(lock.read_bytes().replace(b"\n", b"\r\n"))
    assert probe.installed_against_lock(lock)["lock_is_the_committed_one"] is True
    _lock(lock, [("gradio", "6.30.0"), ("click", "8.5.0")])       # locked again, not committed
    recorded = probe.installed_against_lock(lock)
    assert recorded["lock_is_the_committed_one"] is False
    assert "which is NOT the one committed at the tested commit" in probe.lock_sentence(recorded)


def test_the_deploy_scripts_install_the_roots_a5_names(probe):
    """`DEPLOY_INSTALLED_ROOTS` are what deploy.sh installs after the sync, so
    A5's "from the deploy scripts" follows them."""
    script = (REPO / "deploy.sh").read_text(encoding="utf-8")
    spelled = {probe._normalised(word)
               for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]*[A-Za-z0-9]", script)}
    assert set(probe.DEPLOY_INSTALLED_ROOTS) <= spelled


def test_the_pyg_core_is_locked_and_no_deploy_script_installs_it_unpinned(probe):
    """BACKLOG 50: torch_geometric is declared once, in pyproject.toml as an
    exact pin, and the lock holds that version; deploy.sh and deploy.cmd install
    it through the sync, never again with no version, and take the PyG wheel
    index from the installed torch rather than writing the torch version a
    second time. No version number is written here either: an upgrade changes
    pyproject.toml and the lock, not this test."""
    import tomllib

    declared = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    pins = [d for d in declared["project"]["dependencies"]
            if probe._normalised(re.split(r"[=<>!~ ;\[]", d, maxsplit=1)[0]) == "torch-geometric"]
    assert len(pins) == 1 and re.fullmatch(r"torch-geometric==[0-9][^,;\s]*", pins[0]), pins
    lock = tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))
    assert [p["version"] for p in lock["package"] if p["name"] == "torch-geometric"] == [
        pins[0].split("==", 1)[1]]
    assert "torch-geometric" not in probe.DEPLOY_INSTALLED_ROOTS
    for script in ("deploy.sh", "deploy.cmd"):
        text = (REPO / script).read_text(encoding="utf-8")
        installs = re.findall(r"uv pip install[^\n]*", text)
        assert installs, script
        assert not [line for line in installs
                    if re.search(r"torch[_-]geometric", line, re.IGNORECASE)], script
        assert "data.pyg.org/whl/torch-2" not in text, script     # derived, not written


def test_a_declared_dependency_that_is_not_installed_is_not_the_locks_set(
        probe, tmp_path, monkeypatch):
    """Missing is not matching (BACKLOG 50): a dependency the project declares,
    its marker true here, that is not installed is stated with the version the
    lock pins; one whose marker is false here is not asked for."""
    lock = tmp_path / "uv.lock"
    lock.write_text("\n".join([
        "version = 1", "",
        "[[package]]", 'name = "shepherd-advanced"', 'version = "0.2.0"',
        'source = { editable = "." }',
        "dependencies = [",
        '    { name = "torch" },',
        '    { name = "torch-geometric" },',
        '    { name = "pywin32", marker = "sys_platform == \'win32\' and sys_platform == \'linux\'" },',
        "]", "",
        "[[package]]", 'name = "torch"', 'version = "2.10.0+cu130"', "",
        "[[package]]", 'name = "torch-geometric"', 'version = "2.8.1"', "",
        "[[package]]", 'name = "pywin32"', 'version = "311"', ""]), encoding="utf-8")
    _installed(monkeypatch, [("torch", "2.10.0+cu130")])

    recorded = probe.installed_against_lock(lock)

    assert recorded["version_differs"] == []
    assert recorded["declared_not_installed"] == [{"name": "torch-geometric", "locked": ["2.8.1"]}]
    assert recorded["versions_match_lock"] is False
    said = probe.lock_sentence(recorded)
    assert "1 declared dependency is not installed (torch-geometric, lock 2.8.1)" in said
    report = probe.Report()
    report.probes.append(probe.Probe("A5", "environment", "x", status="passed", facts=recorded))
    verdict = probe.m2_acceptance(report, False)["environment"]
    assert verdict["declared_not_installed"] == ["torch-geometric"]
    assert verdict["versions_match_lock"] is False

    _installed(monkeypatch, [("torch", "2.10.0+cu130"), ("torch_geometric", "2.8.1")])
    matching = probe.installed_against_lock(lock)
    assert matching["declared_not_installed"] == [] and matching["versions_match_lock"] is True


def test_a_matching_set_says_so(probe, tmp_path, monkeypatch):
    lock = _lock(tmp_path / "uv.lock", [("gradio", "6.30.0"), ("torch-geometric", "2.8.0-post1")])
    _installed(monkeypatch, [("gradio", "6.30.0"), ("torch_geometric", "2.8.0.post1")])
    recorded = probe.installed_against_lock(lock)
    assert recorded["versions_match_lock"] is True
    assert recorded["version_differs"] == recorded["installed_not_in_lock"] == []


def test_without_a_lock_the_record_is_skipped_and_the_acceptance_incomplete(probe, tmp_path):
    with pytest.raises(probe.SkipProbe, match="no uv.lock"):
        probe.installed_against_lock(tmp_path / "uv.lock")


def _differing():
    return {"lock_sha256": "0" * 64, "lock_is_the_committed_one": False, "lock_packages": 2,
            "installed_distributions": 4, "versions_match_lock": False,
            "version_differs": [{"name": "click", "installed": "8.5.0", "locked": ["8.3.3"]}],
            "installed_more_than_once": [],
            "installed_not_in_lock": [
                {"name": "aiofiles", "version": "24.1.0", "why": "required_by_nothing_installed"},
                {"name": "pyg-lib", "version": "0.6.0", "why": "deploy_scripts"}],
            "locked_not_installed": []}


def test_a5_states_a_differing_set_on_the_console_and_in_the_environment(
        probe, monkeypatch, capsys):
    monkeypatch.setattr(probe, "installed_against_lock", lambda path: _differing())
    report = probe.Report()

    environment = probe.phase_environment(report, "cpu", describe_only=True)

    a5 = {p.probe_id: p for p in report.probes}["A5"]
    assert a5.status == "passed" and "1 installed version(s) differ" in a5.detail
    said = capsys.readouterr().out
    assert "click 8.5.0, lock 8.3.3" in said and "aiofiles 24.1.0" in said
    assert "NOT the one committed at the tested commit" in said
    assert "1 from the deploy scripts" in said and "pyg-lib" not in said
    assert environment["installed_against_lock"]["version_differs"] == _differing()[
        "version_differs"]
    assert "socket_fits" in environment["installed_against_lock"]["temporary_directory"]


def test_a5_is_said_again_beside_the_acceptance_verdict(probe, monkeypatch):
    """At the end of the run and in the report's `m2_acceptance`, not only on
    a line an hour up the console."""
    report = probe.Report()
    assert probe.environment_line(report) is None
    assert probe.m2_acceptance(report, False)["environment"] == {"A5": "not run"}
    report.probes.append(probe.Probe("A5", "environment", "x", status="passed",
                                     facts=_differing()))

    verdict = probe.m2_acceptance(report, False)["environment"]
    line = probe.environment_line(report)

    assert verdict == {
        "A5": "passed", "lock_sha256": "0" * 64, "lock_is_the_committed_one": False,
        "versions_match_lock": False, "version_differs": ["click"],
        "declared_not_installed": [], "installed_more_than_once": [], "not_in_lock": 2, "not_in_lock_from_deploy_scripts": 1,
        "not_in_lock_required_by_nothing_installed": ["aiofiles"]}
    assert line.startswith("A5, installed packages against this checkout's uv.lock, which is "
                           "NOT the one committed")
    assert "click 8.5.0, lock 8.3.3" in line and "aiofiles 24.1.0" in line
    report.probes[-1] = probe.Probe("A5", "environment", "x", status="skipped",
                                    detail="there is no uv.lock beside this checkout")
    assert "skipped (there is no uv.lock" in probe.environment_line(report)
    assert probe.m2_acceptance(report, False)["environment"] == {"A5": "skipped"}


def test_a5_runs_after_the_device_probes_in_a_full_run(probe, monkeypatch):
    monkeypatch.setattr(probe, "installed_against_lock",
                        lambda path: {"version_differs": [], "installed_not_in_lock": []})
    report = probe.Report()
    probe.phase_environment(report, "cpu")
    assert [p.probe_id for p in report.probes] == ["A1", "A2", "A3", "A4", "A5"]


def test_the_committed_lock_is_read_as_it_is(probe):
    """The repository's own uv.lock and this interpreter's packages: whatever
    they are, each difference is a name with its versions, and gradio, which
    M5-0 pinned, is among the lock's packages."""
    recorded = probe.installed_against_lock(REPO / "uv.lock")
    assert recorded["lock_sha256"] == hashlib.sha256((REPO / "uv.lock").read_bytes()).hexdigest()
    assert recorded["lock_packages"] > 100 and "gradio" not in recorded["locked_not_installed"]
    assert recorded["lock_is_the_committed_one"] in (True, False, None)
    assert all(set(entry) == {"name", "installed", "locked"}
               for entry in recorded["version_differs"])
    assert all(set(entry) == {"name", "version", "why"} and entry["why"] in (
        probe.OUTSIDE_LOCK_DEPLOY, probe.OUTSIDE_LOCK_REQUIRED, probe.OUTSIDE_LOCK_LEFTOVER)
        for entry in recorded["installed_not_in_lock"])


def _long_directory(base: Path, length: int) -> Path:
    """A real directory whose path is `length` bytes long, under `base`."""
    pad = length - len(str(base)) - 1
    if pad < 1:
        pytest.skip("the test's temporary directory is already too long")
    path = base / ("d" * pad)
    path.mkdir()
    assert len(str(path)) == length
    return path


@pytest.mark.skipif(sys.platform != "linux", reason="Linux's AF_UNIX path limit")
def test_a_temporary_directory_a_worker_socket_cannot_be_made_in_is_said(
        probe, tmp_path, monkeypatch):
    """Asked of Python, not computed: a listener opened and closed under the
    temporary directory, as multiprocessing opens one for DataLoader workers."""
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    short = probe._socket_room()
    assert short["socket_fits"] is True and short["path_bytes"] == len(str(tmp_path))
    if sys.version_info[:2] != (3, 12):
        pytest.skip("the failing case is Python 3.12's")
    monkeypatch.setattr(tempfile, "tempdir", str(_long_directory(tmp_path, 100)))
    long = probe._socket_room()
    assert long == {"path_bytes": 100, "socket_fits": False, "refusal": "AF_UNIX path too long",
                    "checked": long["checked"]}
    assert str(tmp_path) not in json.dumps(long)


@pytest.mark.skipif(sys.platform != "linux" or not Path("/usr/bin/python3.13").exists(),
                    reason="needs Linux and a Python 3.13 beside this one")
def test_python_3_13_makes_its_sockets_under_tmp_whatever_the_temporary_directory(
        probe, tmp_path, monkeypatch):
    """`requires-python` is >=3.12: on 3.13 a long TMPDIR does not fail
    training with workers (its sockets fall back to /tmp), so A5 and RW3 must
    not say it does. A computed path length said it did."""
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(_long_directory(tmp_path, 100)))
    assert probe._socket_room("/usr/bin/python3.13")["socket_fits"] is True


def test_a_socket_check_that_cannot_run_says_so(probe, tmp_path):
    room = probe._socket_room(str(tmp_path / "no-such-python"))
    assert room["socket_fits"] is None and room["checked"] == "the check could not run"


# --------------------------------------------------------------------------- settings
def test_the_m5_settings_cover_every_deployment_setting(probe):
    from src.config import server_settings
    from src.utils.gpu_lock import WORKSPACES_ROOT_ENV

    names = {setting.variable for setting in server_settings.all_settings()}
    names |= {WORKSPACES_ROOT_ENV, server_settings.ENV_BIND_HOST,
              server_settings.ENV_ACCEPT_UNAUTHENTICATED_NETWORK_BIND, "SHEPHERD_DEVICE",
              "SHEPHERD_KG_PATH", "SHEPHERD_CHECKPOINT_PATH", "SHEPHERD_DATA_DIR"}
    assert names <= set(probe._M5_SETTINGS)
    assert "SHEPHERD_ALLOC_SOURCE" not in probe._M5_SETTINGS     # the allocator's, left as it is


def test_the_m5_settings_are_the_probe_s_own_while_it_runs_and_put_back(probe, monkeypatch):
    import os

    monkeypatch.setenv("SHEPHERD_REGISTRY", "/deployment/registry")
    monkeypatch.setenv("SHEPHERD_INTAKE_DIR", "/deployment/intake")
    monkeypatch.setenv("SHEPHERD_ALLOC_SOURCE", "preset")
    monkeypatch.delenv("SHEPHERD_WORKSPACES_ROOT", raising=False)

    with pytest.raises(RuntimeError):
        with probe._deployment_settings({"SHEPHERD_WORKSPACES_ROOT": "/probe/root"}):
            assert os.environ["SHEPHERD_WORKSPACES_ROOT"] == "/probe/root"
            assert "SHEPHERD_REGISTRY" not in os.environ
            assert "SHEPHERD_INTAKE_DIR" not in os.environ
            assert os.environ["SHEPHERD_ALLOC_SOURCE"] == "preset"
            raise RuntimeError("a probe that raised")

    assert os.environ["SHEPHERD_REGISTRY"] == "/deployment/registry"
    assert os.environ["SHEPHERD_INTAKE_DIR"] == "/deployment/intake"
    assert "SHEPHERD_WORKSPACES_ROOT" not in os.environ
    with pytest.raises(AssertionError, match="not an M5 setting"):
        with probe._deployment_settings({"SHEPHERD_ALLOC_SOURCE": "x"}):
            pass


def test_the_service_is_started_as_the_launcher_starts_it(probe, tmp_path, monkeypatch):
    """uvicorn on 127.0.0.1 with the launcher's own app and forced arguments,
    startup's bind check given its host, the probe's settings and none of the
    operator's, its Gradio and temporary files under its directory."""
    import subprocess

    from scripts.launch.shep_launch import UVICORN_APP, UVICORN_FORCED_ARGS

    monkeypatch.setenv("SHEPHERD_KG_PATH", "/deployment/kg.json")
    monkeypatch.setenv("SHEPHERD_DEPLOYMENT_MODE", "M1")
    monkeypatch.setenv("SHEPHERD_ALLOC_SOURCE", "preset")
    started = {}

    class _Popen:
        def __init__(self, argv, **kwargs):
            started.update(argv=argv, **kwargs)
            self.pid = 1

    monkeypatch.setattr(subprocess, "Popen", _Popen)
    server = probe._ProbeServer("read-only", tmp_path / "service",
                                {"SHEPHERD_WORKSPACES_ROOT": "/probe/root"})
    server.start()
    server._log.close()

    argv, env = started["argv"], started["env"]
    assert argv[1:4] == ["-m", "uvicorn", UVICORN_APP]
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == str(server.port) and server.port > 0
    assert argv[-len(UVICORN_FORCED_ARGS):] == UVICORN_FORCED_ARGS
    assert env["SHEPHERD_BIND_HOST"] == "127.0.0.1"
    assert env["SHEPHERD_WORKSPACES_ROOT"] == "/probe/root"
    assert "SHEPHERD_KG_PATH" not in env and "SHEPHERD_DEPLOYMENT_MODE" not in env
    assert env["SHEPHERD_ALLOC_SOURCE"] == "preset"
    assert env["GRADIO_TEMP_DIR"] == str(tmp_path / "service" / "gradio")
    assert env["TMPDIR"] == str(tmp_path / "service" / "tmp")
    assert started["cwd"] == probe.project_root
    # It ends with the probe, however the probe ends (Linux).
    assert callable(started["preexec_fn"]) == sys.platform.startswith("linux")


_CHILD = "import time; time.sleep(120)"


def _run_parent(script: str, stderr=None):
    """A Python process running `script` with the probe module as `probe`;
    returns it and the child pid it prints first. `stderr` is passed to
    `Popen` (inherited unless given)."""
    import subprocess

    source = str(REPO / "scripts" / "probe_deployment.py")
    code = (
        "import importlib.util, os, subprocess, sys, time\n"
        f"spec = importlib.util.spec_from_file_location('probe', {source!r})\n"
        "probe = sys.modules['probe'] = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(probe)\n"
        + script)
    parent = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                              stderr=stderr, text=True)
    child = int(parent.stdout.readline())
    return parent, child


def _stop(pid: int) -> None:
    import os
    import signal

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
@pytest.mark.parametrize("name", ["SIGTERM", "SIGHUP", "SIGQUIT"])
def test_a_termination_signal_ends_the_probe_through_its_cleanups(probe, name):
    """`kill <pid>` (SIGTERM), a closed terminal (SIGHUP) and its Ctrl-\\
    (SIGQUIT) run the probe's `finally` blocks, which stop the services and
    jobs it started (`_job_ended_on_exit`), as Ctrl-C does; without the
    handler the process ends at once and a child it started keeps running.
    SIGQUIT did, until the signals were taken from the runner's one list of
    those a job no longer gets (root-cause review, P23-2).

    **The probe is started with the signal at its default action and not
    blocked**, the case this test is about: a disposition and a signal mask
    are inherited across exec, and the probe deliberately leaves a signal
    already set to be ignored as it is (`test_a_signal_set_to_be_ignored_is_
    left_ignored`), so a test process that ignores or blocks the signal would
    otherwise leave the probe asleep and time out without saying why (one
    unexplained 60 s timeout was seen for SIGHUP while closing out the review
    of 2887506; not reproduced since). The parent says on stderr whether the
    probe's handler was installed, and a failure shows what it said."""
    import signal
    import subprocess

    number = getattr(signal, name)
    parent, child = _run_parent(
        "import signal\n"
        f"signal.pthread_sigmask(signal.SIG_UNBLOCK, [{int(number)}])\n"
        f"signal.signal({int(number)}, signal.SIG_DFL)\n"
        "probe._stop_on_termination_signals()\n"
        f"installed = signal.getsignal({int(number)}) is probe._exit_on_signal\n"
        "print(f'handler installed: {installed}', file=sys.stderr, flush=True)\n"
        f"started = subprocess.Popen([sys.executable, '-c', {_CHILD!r}],\n"
        "                           stdout=subprocess.DEVNULL)\n"
        "try:\n"
        "    print(started.pid, flush=True)\n"         # inside: the signal comes after it
        "    time.sleep(120)\n"
        "finally:\n"
        "    started.kill(); started.wait()\n"
        "    print('cleaned up', flush=True)\n", stderr=subprocess.PIPE)
    try:
        parent.send_signal(number)
        try:
            said, told = parent.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            _stop(parent.pid)
            said, told = parent.communicate()
            pytest.fail(f"the probe did not end within 60 s of {name}; it said: {told!r}")
        assert "handler installed: True" in told, told
        assert parent.returncode == 128 + number and "cleaned up" in said, (said, told)
        assert not live(child)
    finally:
        _stop(child)
        _stop(parent.pid)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_the_probe_routes_exactly_the_signals_a_runner_job_no_longer_gets(probe):
    """One definition, beside the spawn (`JOB_DETACHED_SIGNALS`): each of them
    at its default action ends the probe through its cleanups; Ctrl-C keeps
    Python's own handler, which does already."""
    import signal

    from src.api.services.training_manager import JOB_DETACHED_SIGNALS

    present = [getattr(signal, name) for name in JOB_DETACHED_SIGNALS if hasattr(signal, name)]
    saved = {number: signal.getsignal(number) for number in present}
    try:
        for number in present:
            if number != signal.SIGINT:
                signal.signal(number, signal.SIG_DFL)
        probe._stop_on_termination_signals()
        for number in present:
            expected = (signal.default_int_handler if number == signal.SIGINT
                        else probe._exit_on_signal)
            assert signal.getsignal(number) is expected, signal.Signals(number).name
    finally:
        for number, handler in saved.items():
            signal.signal(number, handler)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_a_signal_set_to_be_ignored_is_left_ignored(probe):
    """Under nohup SIGHUP is ignored; the probe keeps it so."""
    import signal

    saved = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGHUP)}
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        probe._stop_on_termination_signals()
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
        assert signal.getsignal(signal.SIGTERM) is probe._exit_on_signal
    finally:
        for number, handler in saved.items():
            signal.signal(number, handler)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux's")
def test_a_service_started_by_the_probe_ends_when_the_probe_is_killed(probe):
    """SIGKILL runs no cleanup; the kernel's parent-death signal still ends
    the service (`_ProbeServer.start` passes `_with_parent_death_signal`)."""
    import time

    parent, child = _run_parent(
        f"started = subprocess.Popen([sys.executable, '-c', {_CHILD!r}],\n"
        "                           stdout=subprocess.DEVNULL,\n"
        "                           preexec_fn=probe._with_parent_death_signal())\n"
        "print(started.pid, flush=True)\n"
        "time.sleep(120)\n")
    try:
        _stop(parent.pid)
        parent.wait(timeout=30)
        deadline = time.monotonic() + 20
        while live(child) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not live(child), "the child outlived the killed probe"
    finally:
        _stop(child)


class _Runner:
    """A job runner's status word and stops, for the probe's job helpers."""

    def __init__(self, words, stop_answer=None):
        self.words, self.stopped, self.stop_answer = dict(words), [], stop_answer

    def get_job_status_word(self, job_type):
        return self.words.get(job_type, "idle")

    def stop(self, job_type):
        self.stopped.append(job_type)
        self.words[job_type] = "cancelled"
        return {"success": True}

    def stop_training(self):
        self.stopped.append("training")
        if self.stop_answer is None:                  # a stop that reaches whatever runs
            self.words["test"] = "cancelled"
            return {"success": True, "status": "stopped"}
        return self.stop_answer


def test_a_job_still_running_when_a_probe_ends_is_stopped(probe):
    runner = _Runner({"test": "running"})
    with pytest.raises(SystemExit):
        with probe._job_ended_on_exit(runner, "test"):
            raise SystemExit(143)                     # SIGTERM through `_exit_on_signal`
    assert runner.stopped == ["test"]
    runner = _Runner({"test": "completed"})
    with probe._job_ended_on_exit(runner, "test"):
        pass
    assert runner.stopped == []


def test_a_training_stop_is_required_not_to_reach_a_test_run(probe):
    refused = {"success": False, "error": "No training is running: the GPU job running is a "
                                          "test run, which this stop does not reach"}
    assert "refused" in probe._training_stop_does_not_reach_a_test_run(
        _Runner({"test": "running"}, stop_answer=refused))
    with pytest.raises(AssertionError, match="training stop was not refused"):
        probe._training_stop_does_not_reach_a_test_run(_Runner({"test": "running"}))
    with pytest.raises(AssertionError, match="reached the test run"):
        probe._training_stop_does_not_reach_a_test_run(
            _Runner({"test": "cancelled"}, stop_answer=refused))


@pytest.mark.skipif(sys.platform == "win32", reason="the holder is a POSIX process here")
def test_the_deployment_s_own_lock_is_read_and_stated(probe, tmp_path, monkeypatch, capsys):
    """M5's probes run on their own lock; whether a job holds the lock the
    operator's settings name when it is read is stated, the holder named on
    the console only."""
    import subprocess

    from src.utils.gpu_lock import WORKSPACES_ROOT_ENV

    monkeypatch.delenv(WORKSPACES_ROOT_ENV, raising=False)
    assert probe._deployment_gpu_lock() == {"configured": False, "held": False,
                                            "unreadable": False}
    monkeypatch.setenv(WORKSPACES_ROOT_ENV, str(tmp_path))
    holder = subprocess.Popen([sys.executable, "-c", probe._LOCK_HOLDER], cwd=REPO,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().startswith("held ")
        assert probe._deployment_gpu_lock() == {"configured": True, "held": True,
                                                "unreadable": False}
        assert f"pid {holder.pid}" in capsys.readouterr().err
    finally:
        holder.kill()
        holder.wait(timeout=30)
        holder.stdin.close()
        holder.stdout.close()
    assert probe._deployment_gpu_lock()["held"] is False


def test_the_deployment_s_lock_is_read_once_per_phase_as_the_docstring_says(probe):
    """Verification of INT-2, P-2: the deployment's lock is read once in
    phase M (W1) and once in phase R (RW1), each time saying whether a job
    holds it at that moment, so the probe's docstring says so, and not that
    those reads tell whether a job held it meanwhile: a job that ran during
    the training probes and ended before W1 is not seen."""
    import ast

    tree = ast.parse((REPO / "scripts" / "probe_deployment.py").read_text(encoding="utf-8"))
    readers = sorted(function.name for function in ast.walk(tree)
                     if isinstance(function, ast.FunctionDef)
                     for node in ast.walk(function)
                     if isinstance(node, ast.Call)
                     and getattr(node.func, "id", None) == "_deployment_gpu_lock")
    assert readers == ["_m5_demo", "_m5_real"]
    said = " ".join(probe.__doc__.split())
    assert ("The deployment's lock is only read, once at W1 and once at RW1, to say whether a "
            "job holds it at that moment; a job that ran during the training probes and ended "
            "before is not seen") in said
    assert ("W1 and RW1 read the lock the operator's settings name once, as M5's probes begin, "
            "and state whether a job holds it then") in said
    assert "held it meanwhile" not in said and "was held meanwhile" not in said


def test_rw3_stops_before_training_when_no_worker_socket_can_be_made(
        probe, tmp_path, monkeypatch):
    """Phase R's RW3 with workers in its settings and a temporary directory in
    which their sockets cannot be made: failed early, saying to use a short
    TMPDIR, no training job started."""
    import src.api.services.training_manager as training_manager

    started = []

    class _NoJobs:
        def __init__(self, *args, **kwargs):
            started.append(1)
            raise AssertionError("a job runner was used")

    monkeypatch.setattr(training_manager, "TrainingManager", _NoJobs)
    monkeypatch.setattr(probe, "_socket_room", lambda python=None: {
        "path_bytes": 120, "socket_fits": False, "refusal": "AF_UNIX path too long",
        "checked": "an AF_UNIX listener opened and closed under it"})
    monkeypatch.setattr(probe, "_deployment_gpu_lock", lambda: {
        "configured": False, "held": False, "unreadable": False})
    real = tmp_path / "workspaces" / "real"
    real.mkdir(parents=True)
    report = probe.Report()

    probe._m5_real(report, tmp_path / "work", real, tmp_path / "workspaces", "cpu",
                   {"read": {}, "settings": {"num_workers": 2}, "m4": {}})

    rw3 = {p.probe_id: p for p in report.probes}["RW3"]
    assert rw3.status == "failed" and "short TMPDIR" in rw3.detail and "120 bytes" in rw3.detail
    assert started == []


def test_a_service_that_exits_before_it_answers_is_a_failure_in_the_probe_s_words(
        probe, tmp_path):
    server = probe._ProbeServer("M1", tmp_path, {})
    (tmp_path / "server.log").write_text("Traceback ... /secret/path\n")
    server.process = SimpleNamespace(poll=lambda: 3, returncode=3)
    server.port = 1
    with pytest.raises(AssertionError, match="exited") as failed:
        server.wait_ready(timeout=5)
    assert "/secret/path" not in str(failed.value)


def test_a_port_taken_meanwhile_is_tried_once_more(probe, tmp_path, monkeypatch):
    server = probe._ProbeServer("M1", tmp_path, {})
    (tmp_path / "server.log").write_text("ERROR: [Errno 98] Address already in use\n")
    server.process = SimpleNamespace(poll=lambda: 1, returncode=1)
    restarts = []
    monkeypatch.setattr(server, "start", lambda: restarts.append(1))
    with pytest.raises(AssertionError, match="exited"):
        server.wait_ready(timeout=5)
    assert restarts == [1]


# --------------------------------------------------------------------------- checks fire
LISTED = "a" * 64
REFUSAL = '{"error": "Refused: words. Nothing was done."}'
EVENTS = [("training_console._on_stop", "_on_stop", [{}], "No training is running"),
          ("test_page._on_start", "_on_start_1", [{}] * 7, "Nothing was started."),
          ("test_page._on_register", "_on_register", [{}] * 5, "Nothing was registered.")]
STOP = "POST /api/v1/training/stop"
LIST = "GET /api/v1/workspaces/{name}/checkpoints"
RELOAD = "POST /api/v1/pipeline/reload"


def _response(status, body):
    return (status, {}, body.encode("utf-8"))


def _routes(probe, answer):
    return {f"{method} {path}": answer(method, path, expected)
            for method, path, expected in probe.OPERATOR_ROUTES}


def _refused(probe, condition="forwarding_header", routes=None, events=None):
    seen = {
        "access": {"enabled": False, "condition": condition, "reason": "words"},
        "routes": _routes(probe, lambda method, path, expected: _response(403, REFUSAL)),
        "events": {label: '["Refused: words. Nothing was done.", null]'
                   for label, _, _, _ in EVENTS},
    }
    seen["routes"].update(routes or {})
    seen["events"].update(events or {})
    return seen


def _allowed_answer(method, path, expected):
    if path.endswith("/{name}/checkpoints"):
        return _response(200, json.dumps({"checkpoints": [{"sha256": LISTED}]}))
    if path.endswith("/training/stop"):
        return _response(409, '{"error": "No training is running"}')
    return _response(expected, '{"detail": "words"}')


def _allowed(probe, access=None, routes=None, events=None):
    seen = {
        "access": access or {"enabled": True, "condition": "enabled", "reason": "words"},
        "routes": _routes(probe, _allowed_answer),
        "events": {"training_console._on_stop": '["Not stopped: No training is running", null]',
                   "test_page._on_start": '["Choose a cohort version. Nothing was started."]',
                   "test_page._on_register": '["Select a kept report. Nothing was registered."]'},
    }
    seen["routes"].update(routes or {})
    seen["events"].update(events or {})
    return seen


def test_d8_s_refusal_check_passes_a_refusal(probe):
    facts = probe._check_refused(_refused(probe), "forwarding_header", ("203.0.113.9",), LISTED)
    assert facts["condition"] == "forwarding_header"
    assert facts["routes"] == dict.fromkeys(_routes(probe, lambda *a: None), 403)
    assert facts["gradio_events"] == dict.fromkeys([e[0] for e in EVENTS], "refused")


@pytest.mark.parametrize("change, message", [
    ({"condition": "cross_site"}, "not forwarding_header"),
    ({"routes": {LIST: _response(200, json.dumps({"checkpoints": [{"sha256": LISTED}]}))}},
     "GET /api/v1/workspaces/{name}/checkpoints answered 200"),
    ({"routes": {STOP: _response(409, '{"error": "No training is running"}')}},
     "POST /api/v1/training/stop answered 409"),
    # A reload that lost D8's dependency answers the wrong-shaped body in
    # validation: refused for the wrong reason, which is not D8's refusal.
    ({"routes": {RELOAD: _response(422, '{"detail": "Input should be a valid dict"}')}},
     "POST /api/v1/pipeline/reload answered 422"),
    ({"routes": {"POST /api/v1/later/route": _response(200, "{}")}},
     "POST /api/v1/later/route answered 200"),
    ({"events": {"test_page._on_start": '["Choose a cohort version. Nothing was started."]'}},
     "test_page._on_start was not refused"),
    ({"routes": {LIST: _response(403, f'{{"error": "Refused: {LISTED}. Nothing was done."}}')}},
     "showed a checkpoint"),
    ({"routes": {STOP: _response(403, '{"error": "Refused: 203.0.113.9. Nothing was done."}')}},
     "repeated a header's value"),
])
def test_d8_s_refusal_check_fails_on_what_it_exists_to_catch(probe, change, message):
    with pytest.raises(AssertionError, match=re.escape(message)):
        probe._check_refused(_refused(probe, **change), "forwarding_header", ("203.0.113.9",),
                             LISTED)


def test_d8_s_refusal_check_requires_every_operator_route_and_an_event(probe):
    seen = _refused(probe)
    del seen["routes"][RELOAD]
    with pytest.raises(AssertionError, match="not asked"):
        probe._check_refused(seen, "forwarding_header", (), LISTED)
    seen = _refused(probe)
    seen["events"] = {}
    with pytest.raises(AssertionError, match="not asked"):
        probe._check_refused(seen, "forwarding_header", (), LISTED)


def test_d8_s_allowance_check_passes_controls_that_ran(probe):
    facts = probe._check_allowed(_allowed(probe), LISTED, "loopback", EVENTS)
    assert facts["gradio_events"] == dict.fromkeys([e[0] for e in EVENTS], "ran")
    assert facts["routes"][RELOAD] == 422 and facts["routes"][STOP] == 409


@pytest.mark.parametrize("change, message", [
    ({"access": {"enabled": False, "condition": "deployment_mode"}}, "D8 refused"),
    ({"routes": {LIST: _response(403, "Refused: words")}}, "answered 403"),
    ({"routes": {LIST: _response(200, json.dumps({"checkpoints": [{"sha256": "b" * 64}]}))}},
     "does not hold the listed checkpoint"),
    ({"routes": {STOP: _response(403, "Refused: words")}}, "D8 did not let it through"),
    ({"routes": {RELOAD: _response(403, '{"error": "Refused: words"}')}},
     "POST /api/v1/pipeline/reload answered 403"),
    ({"routes": {RELOAD: _response(200, '{"success": true}')}},     # it ran: nothing may
     "POST /api/v1/pipeline/reload answered 200"),
    ({"events": {"test_page._on_register": '["Refused: words. Nothing was done."]'}},
     "test_page._on_register did not run its handler"),
])
def test_d8_s_allowance_check_fails_on_a_refused_control(probe, change, message):
    with pytest.raises(AssertionError, match=re.escape(message)):
        probe._check_allowed(_allowed(probe, **change), LISTED, "loopback", EVENTS)


def test_every_gated_api_route_is_d8_s_and_found(probe):
    """The routes D8's dependency gates are the six its contract names
    (src/api/operator_access.py); the probe finds them from the application,
    so one added later is asked too, and asks the six whatever is found."""
    import src.api.main as api

    assert probe._gated_api_routes(api.app) == sorted(
        (method, path) for method, path, _ in probe.OPERATOR_ROUTES)


def test_the_gated_dashboard_events_are_found_by_their_handlers(probe):
    import src.api.main as api

    found = probe._gated_events(api.app)
    assert [(label, data) for label, _, data, _ in found] == [
        (f"{module}.{name}", list(data)) for module, name, data, _ in probe.GATED_EVENTS]
    names = [api_name for _, api_name, _, _ in found]
    assert names[0] == "_on_stop" and len(set(names)) == 3


def test_an_event_registered_otherwise_than_the_probe_sends_it_fails(probe, monkeypatch):
    monkeypatch.setattr(probe, "GATED_EVENTS", (("test_page", "_on_start", ({},), "x"),))
    import src.api.main as api

    with pytest.raises(AssertionError, match="other inputs than the probe sends"):
        probe._gated_events(api.app)


def test_the_url_fetch_count_is_where_a_fetch_happens(probe, monkeypatch):
    """In process: through the application the route guard answers every
    file= URL form and nothing reaches Gradio's URL fetch; with the guard's
    refusal taken away (the defect W4 exists to catch), the forms reach it.
    The control past the guard reaches it either way."""
    import src.api.main as api
    from src.api.middleware.gradio_routes import RefuseUnusedGradioRoutes

    forms = ("file=http://198.51.100.7/probe.png", "file=%68ttp://198.51.100.7/probe.png",
             "file=//198.51.100.7/probe.png")
    held = probe._url_fetches(api.app, forms)
    assert held == {"through_the_guard": 0, "past_the_guard_control": 1, "not_the_guard": []}

    monkeypatch.setattr(RefuseUnusedGradioRoutes, "refusal", lambda self, scope: None)
    gone = probe._url_fetches(api.app, forms)
    assert gone["through_the_guard"] >= 2 and gone["not_the_guard"]
    assert gone["past_the_guard_control"] == 1


def test_a_cors_grant_is_seen_whatever_its_case(probe):
    assert probe._cors_grants({"vary": "Origin"}) == []
    assert probe._cors_grants({"access-control-allow-origin": "https://other.example",
                               "access-control-allow-credentials": "true"}) == [
        "access-control-allow-credentials", "access-control-allow-origin"]


GUARD = "Refused: the dashboard takes no file uploads; nothing was written."


def _raw(status_line, headers, body=b""):
    return (status_line + b"\r\n" + b"".join(h + b"\r\n" for h in headers) + b"\r\n" + body)


def test_the_route_guard_check_passes_its_refusal(probe):
    probe._check_guard_refusal(
        _raw(b"HTTP/1.1 403 Forbidden", [b"content-type: text/plain", b"connection: close"],
             GUARD.encode()), GUARD, "upload")
    probe._check_guard_refusal(
        _raw(b"HTTP/1.1 403 Forbidden", [b"connection: close"]), None, "a HEAD")


@pytest.mark.parametrize("raw, message", [
    (_raw(b"HTTP/1.1 200 OK", [b"connection: close"], GUARD.encode()), "answered 200"),
    (_raw(b"HTTP/1.1 403 Forbidden", [b"connection: keep-alive"], GUARD.encode()),
     "connection closed"),
    (_raw(b"HTTP/1.1 403 Forbidden", [b"connection: close"], b"File not allowed"),
     "not by the route guard"),
    (b"HTTP/1.1 100 Continue\r\n\r\n" + _raw(b"HTTP/1.1 403 Forbidden",
                                            [b"connection: close"], GUARD.encode()),
     "answered 100"),
    (b"", "answered None"),
])
def test_the_route_guard_check_fails_on_what_it_exists_to_catch(probe, raw, message):
    with pytest.raises(AssertionError, match=message):
        probe._check_guard_refusal(raw, GUARD, "upload")


def test_a_holder_is_named_by_its_pid_exactly(probe):
    said = "a training job (train_model.py, pid 4242 on host, since 2026-10-21T09:00:00+00:00"
    assert probe._names_pid(said, 4242)
    assert not probe._names_pid(said, 424)
    assert not probe._names_pid(said.replace(" on host", ""), 42)


def test_a_list_item_s_identity_fields_leave_its_reasons_out(probe):
    item = {"name": "ws", "problem": "/srv/x is unreadable", "build": {
        "state": "not_recorded", "reason": "/srv/x/workspace_build.json is absent"},
        "not_trainable": "/srv/x: no manifest", "items": [{"problem": "/srv/y"}]}
    assert probe._identity_fields(item) == {"name": "ws", "build": {"state": "not_recorded"},
                                            "items": [{}]}


# --------------------------------------------------------------------------- W11 / RW3's hold
#: The training job's stand-in: it takes the test's own GPU lock through the
#: lock's entry point, as `train_model.py` does, and holds it until it ends.
_HOLDER_JOB = (
    "import time\n"
    "from src.utils.gpu_lock import take_gpu_lock\n"
    "lock = take_gpu_lock('training', 'train_model.py')\n"
    "time.sleep(120)\n"
)
#: A test run's stand-in, for a start that is not refused: a stop's SIGINT
#: ends it as a stop ends a test run, so the runner says "cancelled".
_TEST_RUN = [sys.executable, "-c",
             "import signal, time; signal.signal(signal.SIGINT, signal.SIG_DFL); time.sleep(120)"]


class _RunInputs:
    """A test run's inputs, as far as `start_test` reads them."""

    def __init__(self, reports: Path):
        self.reports = reports

    def reports_dir(self):
        return self.reports

    def summary(self):
        return {}


@pytest.fixture
def held(probe, monkeypatch, tmp_path):
    """The real job runner, its training job a real child holding the test's
    GPU lock (started as the runner starts every job, `_spawn_locked`), and a
    measurement CLI refused by the lock naming its holder while it is held,
    measuring (exit 0) once it is free. `ends_job` ends the training job by
    a signal sent outside the runner's stop (SIGKILL, as a crash or the
    out-of-memory killer does, unless another is given), waiting until the
    runner's monitor has recorded it; `during_cli`, when set, runs as the CLI
    starts."""
    import signal
    import time

    import src.api.services.training_manager as training_manager
    from src.api.services.training_manager import JOB_TYPES, TrainingManager
    from src.utils.gpu_lock import lock_state

    monkeypatch.delenv("SHEPHERD_DEVICE", raising=False)
    monkeypatch.setattr(training_manager, "build_test_argv", lambda *args: list(_TEST_RUN))
    manager = TrainingManager()
    with manager._lock:
        job, error = manager._spawn_locked(JOB_TYPES["training"],
                                           [sys.executable, "-c", _HOLDER_JOB], inputs={})
    assert job is not None, error
    pid = job.process.pid
    deadline = time.monotonic() + 60
    while not probe._names_pid(lock_state().get("held_by") or "", pid):
        assert job.process.poll() is None and time.monotonic() < deadline, (
            "the training job's stand-in did not take the GPU lock")
        time.sleep(0.05)

    def ends_job(number=signal.SIGKILL):
        os.kill(pid, number)
        assert job.done.wait(timeout=30), "the runner did not record the job's end"

    state = {"manager": manager, "pid": pid, "ends_job": ends_job, "during_cli": None,
             "cli_calls": 0, "outcome": tmp_path / "outcome.json",
             "inputs": _RunInputs(tmp_path)}

    def cli(argv):
        state["cli_calls"] += 1
        if state["during_cli"] is not None:
            state["during_cli"]()
        held_by = lock_state().get("held_by")
        if held_by:
            return 1, (f"measure_scorer: refused: the GPU lock x is held by {held_by}. One "
                       "managed GPU job runs at a time. Nothing was written.")
        return 0, "measured"

    monkeypatch.setattr(probe, "_cli_measure", cli)
    try:
        yield state
    finally:
        for job_type in ("test", "training"):
            if manager.get_job_status_word(job_type) == "running":
                manager.stop(job_type)
        if job.process.poll() is None:
            job.process.kill()
        job.done.wait(timeout=30)


def _exclusion(probe, held):
    return probe._exclusion_while_training(held["manager"], held["inputs"], held["pid"], [],
                                           held["outcome"])


def _after(held, method, then):
    """`held`'s runner with `method` answering as it does, then `then()`."""
    manager = held["manager"]
    real = getattr(manager, method)

    def answered(*args):
        answer = real(*args)
        then()
        return answer

    setattr(manager, method, answered)


def _before(held, method, first):
    """`held`'s runner with `first()` run before `method` answers."""
    manager = held["manager"]
    real = getattr(manager, method)

    def answered(*args):
        first()
        return real(*args)

    setattr(manager, method, answered)


WAITID = pytest.mark.skipif(not hasattr(os, "waitid"), reason="os.waitid")


@WAITID
def test_the_hold_passes_when_the_job_runs_and_is_named_throughout(probe, held):
    facts = _exclusion(probe, held)
    assert facts["cli_exit"] == 1 and held["cli_calls"] == 1
    assert held["manager"].get_job_status_word("training") == "running"
    assert held["manager"].get_job_status_word("test") == "idle"


@WAITID
def test_a_job_that_ended_before_the_check_fails_as_could_not_hold(probe, held):
    """Review of 8242b47, W11/RW3: the demo job ended by early stopping and
    released the lock before the CLI was tried, which then measured (exit 0)
    and read as the lock failing. A job that ended before the check is said as
    one that could not hold the lock, and nothing is tried: a test start, which
    the free slot would not refuse, would start a real test run."""
    held["ends_job"]()
    starts = []
    _before(held, "start_test", lambda: starts.append(1))
    with pytest.raises(AssertionError, match="could not hold the GPU lock for the check: before "
                                             "a test start was tried, its process had ended"):
        _exclusion(probe, held)
    assert starts == [] and held["cli_calls"] == 0


@WAITID
@pytest.mark.parametrize("moment, during", [
    ("as the test start is tried", "the test start"),
    ("after a refused start", "the test start"),
    ("after the cancel", "the test cancel"),
    ("while the CLI is tried", "the measurement CLI"),
])
def test_a_job_that_ends_during_the_check_is_said_with_when_and_how(probe, held, moment,
                                                                      during):
    """Root-cause round, W11-1, and its verification, P-1: one bracket around
    the whole check. However the job ends between the two holder checks (here
    by a signal sent outside the runner's stop, so nothing asked the runner to
    stop it), every answer meanwhile is judged only after the second check,
    and the failure names the action during which the runner's record first
    showed it ended, and how it says it ended: never "a test start was not
    refused", "a test cancel was not refused", a stop reaching training or
    the lock failing to refuse. A test run the free slot let a start begin is
    stopped before the cancel is tried."""
    if moment == "as the test start is tried":
        _before(held, "start_test", held["ends_job"])
    elif moment == "after a refused start":
        _after(held, "start_test", held["ends_job"])
    elif moment == "after the cancel":
        _after(held, "cancel_test", held["ends_job"])
    else:
        held["during_cli"] = held["ends_job"]
    with pytest.raises(AssertionError) as failed:
        _exclusion(probe, held)
    said = str(failed.value)
    assert said == (f"the training job ended during {during} with no stop asked of the runner "
                    "(the runner says failed, ended by SIGKILL): a crash, or a signal sent "
                    "outside the runner's stop, which its record does not tell apart; the "
                    "exclusion was not exercised"), said
    assert held["cli_calls"] == 1
    word = held["manager"].get_job_status_word("test")
    assert word == ("cancelled" if moment == "as the test start is tried" else "idle"), word


@WAITID
def test_a_cancel_that_signals_training_itself_is_said_as_ending_it_then(probe, held):
    """Verification of W11-1, P-1, its case (b): a test cancel that signals the
    training job's process itself, as the runner's stop would but without
    asking it (a product defect the runner's record cannot attribute), once
    read as a job that could not hold the lock. It is said as ended during
    the test cancel with no stop asked, the signal named as the runner records
    it."""
    import signal

    _after(held, "cancel_test", lambda: held["ends_job"](signal.SIGINT))
    with pytest.raises(AssertionError) as failed:
        _exclusion(probe, held)
    said = str(failed.value)
    assert said.startswith("the training job ended during the test cancel with no stop asked of "
                           "the runner (the runner says failed, ended by SIGINT)"), said
    assert "could not hold" not in said


@WAITID
@pytest.mark.parametrize("action, answer", [
    ("the test start", "answered as it does"),
    ("the test cancel", "stopped, saying so"),
    ("the test cancel", "stopped, claiming a refusal"),
    ("the measurement CLI", "answered as it does"),
])
def test_an_action_that_stops_training_fails_as_reaching_it(probe, held, action, answer):
    """The counter-example the attribution exists for: one of the check's
    actions ends the training job through the runner's own stop (a product
    defect: a new job replacing the old, a cancel reaching the other job
    type). The runner's own record says a stop ended it ("stopped", which only
    a stop asked of it gives) right after that action and not before, so it
    fails as that action reaching training, never as a job that could not
    hold the lock or as another action, whatever the action answered. The
    check still observes every action before it judges any."""
    manager = held["manager"]
    if action == "the test start":
        _before(held, "start_test", lambda: manager.stop("training"))
    elif action == "the measurement CLI":
        held["during_cli"] = lambda: manager.stop("training")
    else:
        def cancel():
            stopped = manager.stop("training")
            if answer == "stopped, saying so":
                return stopped
            return {"success": False, "error": "No test run is running: the GPU job running "
                                               "is training, which this stop does not reach"}

        manager.cancel_test = cancel
    with pytest.raises(AssertionError) as failed:
        _exclusion(probe, held)
    assert str(failed.value) == (f"{action} reached training: the runner says stopped right "
                                 "after it, which only a stop asked of the runner gives")
    assert held["cli_calls"] == 1


@WAITID
def test_a_start_not_refused_while_the_job_holds_fails_and_its_run_is_stopped(
        probe, held, monkeypatch):
    """The counter-example the start's check exists for: the job runs and holds
    the lock, and a test start is not refused (the runner's slot check gone).
    It fails as not refused, and the test run it began is stopped before the
    cancel is tried."""
    manager = held["manager"]
    real = manager._slot_refusal_locked
    monkeypatch.setattr(manager, "_slot_refusal_locked",
                        lambda kind: None if kind.name == "test" else real(kind))
    with pytest.raises(AssertionError, match="a test start was not refused"):
        _exclusion(probe, held)
    assert manager.get_job_status_word("test") == "cancelled"
    assert manager.get_job_status_word("training") == "running"


@WAITID
def test_a_lock_held_by_another_than_the_job_fails_the_hold(probe, held, monkeypatch):
    import src.utils.gpu_lock as gpu_lock

    monkeypatch.setattr(gpu_lock, "lock_state", lambda environ=None: {
        "held_by": "a training job (train_model.py, pid 1 on host, as it described itself)"})
    with pytest.raises(AssertionError, match="holder record did not name its process"):
        _exclusion(probe, held)
    assert held["cli_calls"] == 0


def test_without_os_waitid_the_hold_fails_saying_why(probe, held, monkeypatch):
    """Windows, and macOS before Python 3.13, have no os.waitid: W11 and RW3
    fail with a sentence that says so, never pass or skip."""
    monkeypatch.delattr(os, "waitid", raising=False)
    with pytest.raises(AssertionError, match=r"no os\.waitid \(Windows, or macOS before Python "
                                             r"3\.13\)"):
        _exclusion(probe, held)
    assert held["cli_calls"] == 0


class _StartsRecorded:
    """A job runner that records each training start asked of it, the settings
    it was given, and refuses it; `made` collects every runner made."""

    made: list = []

    def __init__(self, *args, **kwargs):
        self.starts = []
        _StartsRecorded.made.append(self)

    def get_job_status_word(self, job_type):
        return "idle"

    def start_training(self, config, expected_dataset_fingerprint=None):
        self.starts.append(dict(config))
        return {"success": False, "error": "refused by the stand-in"}


def _rw3(probe, tmp_path, monkeypatch, settings):
    """Phase R's M5 part with a recording runner (`_StartsRecorded`); its RW3."""
    import src.api.services.training_manager as training_manager

    _StartsRecorded.made = []
    monkeypatch.setattr(training_manager, "TrainingManager", _StartsRecorded)
    monkeypatch.setattr(probe, "_read_run_checkpoint", lambda path: {"sha256": "c" * 64})
    monkeypatch.setattr(probe, "_deployment_gpu_lock", lambda: {
        "configured": False, "held": False, "unreadable": False})
    real = tmp_path / "workspaces" / "real"
    real.mkdir(parents=True)
    version = SimpleNamespace(identity=SimpleNamespace(sha256="e" * 64))
    report = probe.Report()
    probe._m5_real(report, tmp_path / "work", real, tmp_path / "workspaces", "cpu",
                   {"read": {"split_manifest.json": "f" * 64}, "parent": tmp_path / "p.pt",
                    "settings": settings, "m4": {"version": version}})
    return {p.probe_id: p for p in report.probes}["RW3"]


#: Settings that would end the job by itself or make it a continuation.
_SELF_ENDING = {"num_epochs": 3, "early_stopping_patience": 2, "pause_after_epoch": 1,
                "resume_from": "parent.pt", "extend_epochs": 1, "extend_lr": 0.01}


def test_the_held_job_is_held_by_configuration_whatever_its_settings_say(probe, tmp_path,
                                                                         monkeypatch):
    """Root-cause round, W11-2: the hold is `_start_training_job`'s, the one
    function W11 and RW3 start their job through, so neither caller can
    start it otherwise. Whatever settings it is given, the runner gets
    `_HOLD_EPOCHS` epochs and patience, no pause, no parent, no extension, and
    every other setting as given."""
    if not hasattr(os, "waitid"):                     # pinned on every platform
        monkeypatch.setattr(os, "waitid", lambda *args: None, raising=False)
    runner = _StartsRecorded()
    others = {**probe._DEMO_TRAINING, "data_dir": "/w", "run_dir": "/r", "device": "cpu"}
    with pytest.raises(AssertionError, match="the runner refused the training start"):
        probe._start_training_job(runner, {**others, **_SELF_ENDING}, "f" * 64,
                                  tmp_path / "gpu.lock")
    assert runner.starts == [{**others, "num_epochs": probe._HOLD_EPOCHS,
                              "early_stopping_patience": probe._HOLD_EPOCHS}]
    assert probe._HOLD_EPOCHS >= 1_000_000


@WAITID
def test_rw3_starts_its_job_held_whatever_the_real_settings_say(probe, tmp_path, monkeypatch):
    """RW3's job, from real settings that would let it end by itself during the
    check (the reviewer's run) or continue a parent, reaches the runner held."""
    rw3 = _rw3(probe, tmp_path, monkeypatch, {"num_workers": 0, **_SELF_ENDING})
    assert rw3.status == "failed" and "the runner refused the training start" in rw3.detail
    (config,) = [start for runner in _StartsRecorded.made for start in runner.starts]
    assert config["num_epochs"] == config["early_stopping_patience"] == probe._HOLD_EPOCHS
    assert not set(config) & {"pause_after_epoch", "resume_from", "extend_epochs", "extend_lr"}


def test_without_os_waitid_no_held_job_is_started(probe, tmp_path, monkeypatch):
    """Root-cause round, W11-4: whether the job still runs cannot be told
    without os.waitid, and that is known before anything starts, so W11's and
    RW3's one start (`_start_training_job`) fails saying so before it asks the
    runner for a job: RW3 fails with the sentence, no training started."""
    monkeypatch.delattr(os, "waitid", raising=False)
    runner = _StartsRecorded()
    with pytest.raises(AssertionError, match=r"no os\.waitid"):
        probe._start_training_job(runner, {}, "f" * 64, tmp_path / "gpu.lock")
    assert runner.starts == []
    rw3 = _rw3(probe, tmp_path, monkeypatch, {"num_workers": 0})
    assert rw3.status == "failed" and "no os.waitid" in rw3.detail
    assert [start for made in _StartsRecorded.made for start in made.starts] == []


def test_the_probe_s_own_lock_is_named_and_the_operator_s_is_still_stated(probe, tmp_path,
                                                                          monkeypatch, capsys):
    """Review of 8242b47, P2-4: the formal entry points the probe runs in its
    own process refuse without a lock, so phases M and R name one of the
    probe's own under the work directory; W1 and RW1 still state the
    operator's lock, and the operator's settings come back afterwards.

    The operator's shell names the deployment's root, as the guide prepares
    it, and the root is the lock's one name (root-cause review, P24-S1 and
    its verification): the probe's own lock is the one under a workspaces
    root of its own, an empty directory, named in the operator's place
    meanwhile."""
    from src.utils.gpu_lock import LOCK_FILE_NAME, WORKSPACES_ROOT_ENV, GpuLock, lock_location

    operator_lock = tmp_path / LOCK_FILE_NAME
    monkeypatch.setenv(WORKSPACES_ROOT_ENV, str(tmp_path))
    work = tmp_path / "work"
    work.mkdir()
    own = work / probe.PROBE_GPU_LOCK_ROOT
    with GpuLock(operator_lock, job_type="training", command="deployment-job").acquire():
        with probe._probes_own_gpu_lock(work):
            assert os.environ[WORKSPACES_ROOT_ENV] == str(own)
            assert lock_location() == own.resolve() / LOCK_FILE_NAME
            assert list(own.iterdir()) == [], "the probe's own root lists a workspace"
            assert probe._deployment_gpu_lock() == {"configured": True, "held": True,
                                                    "unreadable": False}
    assert "deployment-job" in capsys.readouterr().err
    assert os.environ[WORKSPACES_ROOT_ENV] == str(tmp_path)
    assert probe._OPERATOR_LOCK_SETTINGS is None
    monkeypatch.delenv(WORKSPACES_ROOT_ENV)
    with probe._probes_own_gpu_lock(work):
        assert probe._deployment_gpu_lock() == {"configured": False, "held": False,
                                                "unreadable": False}
    assert WORKSPACES_ROOT_ENV not in os.environ
