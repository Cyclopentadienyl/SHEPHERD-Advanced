"""The deployment probe's download phase (G), without a network.

Phase G exists to show that a deployment machine can complete the production
fetch path — configured sources, destination rules, staging, the imports and
role checks, publish — on its real network. These tests replace only the
transfer (`download_ontology`) and DNS, and check the three things the phase
owes its reader:

  * the category it reports is the one the production path decided, for a
    success and for each kind of failure — a transfer that succeeded is not a
    pass unless the file loaded, passed both checks and was published as the
    bytes that were verified;
  * nothing outside `--work-dir` is written, and a write there is caught;
  * the report carries no path, no source host and no proxy credential.

Module: tests/unit/test_probe_download_phase.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest

pytest.importorskip("pronto")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MONDO = (b"format-version: 1.2\ndata-version: releases/2026-09-01\nontology: mondo\n\n"
         b"[Term]\nid: MONDO:0000001\nname: d\n\n[Term]\nid: MONDO:0000002\nname: e\n")
HPO = (b"format-version: 1.2\ndata-version: hp/releases/2026-09-01\nontology: hp\n\n"
       b"[Term]\nid: HP:0000001\nname: p\n")
MONDO_OWL = (
    b'<?xml version="1.0"?>\n<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
    b'xmlns:owl="http://www.w3.org/2002/07/owl#">\n'
    b'<owl:Ontology rdf:about="http://purl.obolibrary.org/obo/mondo.owl"/>\n'
    b'<owl:Class rdf:about="http://purl.obolibrary.org/obo/MONDO_0000001"/>\n</rdf:RDF>\n'
)
SOURCES = {
    "mondo": ("https://purl.example/obo/mondo.obo", "https://purl.example/obo/mondo.owl"),
    "hpo": ("https://purl.example/obo/hp.obo",),
    "go": (), "mp": (),
}


def _probe():
    spec = importlib.util.spec_from_file_location(
        "probe_deployment", REPO / "scripts" / "probe_deployment.py")
    module = importlib.util.module_from_spec(spec)
    # Its dataclasses resolve their own module by name while being built.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A configured root and a default cache that must stay untouched, no
    network at all, and a transfer the test decides."""
    import src.ontology.download as download_module
    import src.ontology.loader as loader_module
    import src.ontology.settings as settings_module

    for name in ("http_proxy", "https_proxy", "ftp_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)

    def no_network(*args, **kwargs):
        raise socket.gaierror(socket.EAI_NONAME, "no network in a unit test")

    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)

    root = tmp_path / "configured_root"
    root.mkdir()
    (root / "mondo.obo").write_bytes(MONDO.replace(b"2026-09-01", b"2025-01-01"))
    home_cache = tmp_path / "home_cache"
    home_cache.mkdir()
    (home_cache / "hpo.obo").write_bytes(HPO)
    monkeypatch.setattr(loader_module, "default_cache_dir", lambda: home_cache)

    settings = {"value": settings_module.OntologySettings(roots=(root,), sources=dict(SOURCES))}
    monkeypatch.setattr(settings_module, "load_ontology_settings",
                        lambda config_path=None: settings["value"])

    plan = {"mondo.obo": MONDO, "mondo.owl": MONDO_OWL, "hp.obo": HPO}
    calls = []

    def transfer(url, destination, **kwargs):
        calls.append(url)
        outcome = plan[url.rsplit("/", 1)[-1]]
        if isinstance(outcome, BaseException):
            raise outcome
        Path(destination).write_bytes(outcome)
        return Path(destination)

    monkeypatch.setattr(download_module, "download_ontology", transfer)
    work = tmp_path / "work"
    work.mkdir()
    return {"plan": plan, "calls": calls, "root": root, "home_cache": home_cache,
            "work": work, "settings": settings, "settings_module": settings_module}


def _run_phase(world):
    probe = _probe()
    report = probe.Report()
    probe.phase_download(report, world["work"])
    return {p.probe_id: p for p in report.probes}


class TestASuccessIsMoreThanATransfer:

    def test_both_ontologies_are_published_as_the_bytes_that_were_verified(self, world):
        probes = _run_phase(world)

        assert {k: p.status for k, p in probes.items()} == {
            "G0": "passed", "G1": "passed", "G2": "passed", "G9": "passed"}
        mondo, hpo = probes["G1"].facts, probes["G2"].facts
        assert mondo["outcome"] == hpo["outcome"] == "published"
        assert mondo["source_digest"] == hashlib.sha256(MONDO).hexdigest()
        assert hpo["source_digest"] == hashlib.sha256(HPO).hexdigest()
        assert mondo["data_version"] == "releases/2026-09-01"
        assert mondo["attempts"] == [{"index": 1, "format": "obo", "outcome": "delivered"}]
        assert mondo["term_count"] == 2 and mondo["format"] == "obo"
        cache = world["work"] / "download_cache"
        assert sorted(p.name for p in cache.iterdir()) == ["hpo.obo", "mondo.obo"]

    def test_it_forces_a_download_even_with_candidates_on_disk(self, world):
        """The configured root holds a MONDO and the default cache an HPO; a
        phase that selected them would report a download that never happened."""
        _run_phase(world)

        assert world["calls"] == ["https://purl.example/obo/mondo.obo",
                                  "https://purl.example/obo/hp.obo"]

    def test_a_fallback_is_recorded_source_by_source(self, world):
        from src.ontology.download import OntologyHostUnresolved

        world["plan"]["mondo.obo"] = OntologyHostUnresolved("no answer")

        mondo = _run_phase(world)["G1"].facts

        assert mondo["outcome"] == "published" and mondo["format"] == "owl"
        assert mondo["attempts"] == [
            {"index": 1, "format": "obo", "outcome": "host_unresolved"},
            {"index": 2, "format": "owl", "outcome": "delivered"},
        ]

    def test_the_network_is_described_per_source_without_hosts(self, world):
        mondo = _run_phase(world)["G1"].facts

        assert mondo["network_at_probe_time"] == [
            {"index": 1, "scheme": "https", "host_resolves_here": False,
             "goes_through_proxy": False, "host_allow_listed": False},
            {"index": 2, "scheme": "https", "host_resolves_here": False,
             "goes_through_proxy": False, "host_allow_listed": False},
        ]
        assert mondo["sources_configured"] == 2
        assert mondo["sources_are_project_defaults"] is False


class TestEachFailureIsNamedByWhatDecidedIt:

    @pytest.mark.parametrize("make,outcome,attempt", [
        (lambda d: d.OntologyDestinationRefused("refused"), "refused_by_policy", "refused_by_policy"),
        (lambda d: d.OntologyHostUnresolved("no answer"), "transfer_failed", "host_unresolved"),
        (lambda d: d.OntologyTruncatedError("short"), "transfer_failed", "truncated"),
        (lambda d: d.OntologyDownloadError("reset"), "transfer_failed", "transfer_failed"),
    ], ids=["policy", "dns", "truncated", "transfer"])
    def test_a_fetch_that_delivered_nothing(self, world, make, outcome, attempt):
        import src.ontology.download as download_module

        world["plan"]["mondo.obo"] = make(download_module)
        world["plan"]["mondo.owl"] = make(download_module)

        g1 = _run_phase(world)["G1"]

        assert g1.status == "failed"
        assert g1.facts["outcome"] == outcome
        assert [a["outcome"] for a in g1.facts["attempts"]] == [attempt, attempt]

    def test_the_loader_keeps_a_failed_fetchs_attempts_too(self, world):
        """`last_fetch_attempts` is documented as set on failure as well as
        success; the first version set it only on success."""
        import src.ontology.download as download_module
        from src.ontology.loader import OntologyFetchError, OntologyLoader

        world["plan"]["mondo.obo"] = download_module.OntologyDownloadError("reset")
        world["plan"]["mondo.owl"] = download_module.OntologyDownloadError("reset")
        loader = OntologyLoader(cache_dir=world["work"] / "c")

        with pytest.raises(OntologyFetchError) as caught:
            loader._fetch_ontology("mondo", True, roots=[])

        assert loader.last_fetch_attempts == caught.value.attempts
        assert [a.outcome for a in loader.last_fetch_attempts] == ["transfer_failed"] * 2

    @pytest.mark.parametrize("body,outcome", [
        (HPO, "wrong_ontology"),
        (MONDO.replace(b"ontology: mondo\n", b"ontology: mondo\nimport: http://x.invalid/y.obo\n"),
         "imports_declared"),
        (b"format-version: 1.2\n[Term\nthis is not obo\n", "failed_otherwise"),
    ], ids=["role", "imports", "unparseable"])
    def test_a_transfer_that_succeeded_is_not_a_pass(self, world, body, outcome):
        world["plan"]["mondo.obo"] = body

        g1 = _run_phase(world)["G1"]

        assert g1.status == "failed"
        assert g1.facts["outcome"] == outcome
        assert g1.facts["attempts"] == [{"index": 1, "format": "obo", "outcome": "delivered"}]
        cache = world["work"] / "download_cache"
        assert not (cache / "mondo.obo").exists(), "a rejected download was published"
        assert not [p for p in cache.iterdir() if p.name.endswith(".staged")]

    def test_an_unreadable_configuration_is_said_once_and_the_rest_skip(self, world, monkeypatch):
        settings_module = world["settings_module"]

        def broken(config_path=None):
            raise settings_module.OntologySettingsError("ontology.allowed_hosts lists 'x://y'")

        monkeypatch.setattr(settings_module, "load_ontology_settings", broken)

        probes = _run_phase(world)

        assert probes["G0"].status == "failed"
        assert probes["G0"].facts == {"outcome": "configuration_invalid",
                                      "error_type": "OntologySettingsError"}
        assert {probes[k].status for k in ("G1", "G2", "G9")} == {"skipped"}

    def test_no_configured_source_is_its_own_category(self, world):
        world["settings"]["value"] = world["settings_module"].OntologySettings(
            roots=(world["root"],), sources={"mondo": (), "hpo": (), "go": (), "mp": ()})

        g1 = _run_phase(world)["G1"]

        assert g1.facts["outcome"] == "no_source" and g1.facts["attempts"] == []


class TestNothingOutsideTheWorkDirectory:

    def test_the_configured_root_and_the_default_cache_are_left_alone(self, world):
        before = {p: p.read_bytes() for d in (world["root"], world["home_cache"])
                  for p in d.iterdir()}

        assert _run_phase(world)["G9"].status == "passed"
        assert {p: p.read_bytes() for p in before} == before

    @pytest.mark.parametrize("where", ["root", "home_cache"])
    def test_a_write_there_is_caught(self, world, monkeypatch, where):
        """The check has to be able to fail, for each place it guards: a
        transfer that also drops a file there is reported, not passed."""
        import src.ontology.download as download_module

        stubbed = download_module.download_ontology

        def leaky(url, destination, **kwargs):
            (world[where] / "stray.obo").write_bytes(HPO)
            return stubbed(url, destination, **kwargs)

        monkeypatch.setattr(download_module, "download_ontology", leaky)

        g9 = _run_phase(world)["G9"]

        assert g9.status == "failed" and "created" in g9.detail


class TestThePhaseChecksWhatItReports:
    """Two claims a pass makes that the production path always satisfies, so
    only a broken production path can show the phase would notice."""

    def test_published_bytes_that_are_not_the_verified_ones_fail(self, world, monkeypatch):
        import src.ontology.loader as loader_module

        genuine = loader_module.os.replace

        def swap_on_publish(src, dst):
            genuine(src, dst)
            if str(dst).endswith("mondo.obo"):
                Path(dst).write_bytes(MONDO + b"\n")

        monkeypatch.setattr(loader_module.os, "replace", swap_on_publish)

        g1 = _run_phase(world)["G1"]

        assert g1.status == "failed" and "not the bytes that were verified" in g1.detail

    def test_a_staging_file_left_behind_fails(self, world, monkeypatch):
        import src.ontology.download as download_module

        stubbed = download_module.download_ontology

        def leaves_one(url, destination, **kwargs):
            (Path(destination).parent / ".mondo.obo.leftover.staged").write_bytes(b"")
            return stubbed(url, destination, **kwargs)

        monkeypatch.setattr(download_module, "download_ontology", leaves_one)

        g1 = _run_phase(world)["G1"]

        assert g1.status == "failed" and "staging file" in g1.detail


class TestTheCommandLine:

    def _quiet_phases(self, probe, monkeypatch, *, forbid=()):
        def explode(*args, **kwargs):
            raise AssertionError("a phase ran that the flags excluded")

        def environment(report, device):
            return {"resolved_device": "cpu"}

        monkeypatch.setattr(probe, "phase_environment", environment)
        for name in ("phase_writer", "phase_workspace", "phase_training", "phase_serving",
                     "phase_real_build", "phase_download"):
            monkeypatch.setattr(probe, name, explode if name in forbid else (lambda *a, **k: None))

    def test_the_download_is_opt_in(self, tmp_path, monkeypatch):
        probe = _probe()
        self._quiet_phases(probe, monkeypatch, forbid=("phase_download",))

        assert probe.main(["--work-dir", str(tmp_path / "w"), "--report",
                           str(tmp_path / "r.json")]) == 0

    def test_download_only_runs_nothing_else(self, tmp_path, monkeypatch):
        probe = _probe()
        self._quiet_phases(probe, monkeypatch, forbid=(
            "phase_writer", "phase_workspace", "phase_training", "phase_serving",
            "phase_real_build"))
        ran = []
        monkeypatch.setattr(probe, "phase_download", lambda report, work: ran.append(work))

        assert probe.main(["--work-dir", str(tmp_path / "w"), "--report",
                           str(tmp_path / "r.json"), "--download-only"]) == 0
        assert ran == [tmp_path / "w"]
        settings = json.loads((tmp_path / "r.json").read_text())["settings"]
        assert settings["download_only"] is True and settings["download_requested"] is True

    def test_download_only_with_a_real_build_is_refused(self):
        with pytest.raises(SystemExit) as caught:
            _probe().parse_args(["--work-dir", "w", "--download-only", "--external-dir", "e"])

        assert caught.value.code == 2


def test_the_report_names_no_path_no_host_and_no_credential(world, tmp_path, monkeypatch):
    """End to end through `main`, with a proxy whose URL carries a password.
    The report may say a proxy was involved; it may not say which, or how to
    log in to it, and it may name no source host and no directory."""
    probe = _probe()
    monkeypatch.setattr(probe, "phase_environment",
                        lambda report, device: {"resolved_device": "cpu"})
    monkeypatch.setenv("https_proxy", "http://probe-user:s3cret@proxy.hospital.internal:3128")
    report_path = tmp_path / "report.json"

    status = probe.main(["--work-dir", str(tmp_path / "w"), "--report", str(report_path),
                         "--download-only"])

    text = report_path.read_text()
    payload = json.loads(text)
    assert status == 0, [(p["id"], p["status"], p["detail"]) for p in payload["probes"]]
    for forbidden in (str(tmp_path), "purl.example", "proxy.hospital.internal", "s3cret",
                      "probe-user", str(Path.home())):
        assert forbidden not in text, f"the report carries {forbidden!r}"
    g1 = next(p for p in payload["probes"] if p["id"] == "G1")
    assert g1["facts"]["network_at_probe_time"][0]["goes_through_proxy"] is True
