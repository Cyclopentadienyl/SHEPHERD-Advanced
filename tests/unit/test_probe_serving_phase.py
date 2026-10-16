"""The deployment probe's serving phase (E1-E8), on CPU, and contract M3c's checks.

On the GB10 phase E runs on CUDA; here it runs on CPU over the demo workspace
and a model the production trainer trained on it (phase D), so the serving
build, the status encoder, the reload route and a diagnosis are driven end to
end and every probe in the phase must pass. Contract M3c rides on these probes:
E1 requires the served identity to name what the probe itself verified, E2 that
the status reports it unchanged, E5 that a refused reload leaves it as it was,
and E7 that a diagnosis carries it. M4 and R6 ride on `_serve_and_diagnose`,
which requires a diagnosis to name what the caller's workspace check read and
to carry the identity serving reports; it runs here over a copy of the
workspace given a shortest-path table, as M4 serves it. Each of those checks is
also shown to fire: a probe that cannot fail proves nothing.

Module: tests/unit/test_probe_serving_phase.py
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

E_PROBES = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"]


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


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """The demo workspace and one epoch of the production trainer on it (D1),
    shared by this file's tests: serving reads both and writes neither."""
    from src.kg.workspace import write_workspace

    probe = _probe()
    work = tmp_path_factory.mktemp("serving")
    sound = work / "sound_workspace"
    write_workspace(probe._demo_kg(), sound, feature_dim=probe.FEATURE_DIM,
                    samples=probe._budget())
    report = probe.Report()
    checkpoint = probe.phase_training(report, work, sound, device="cpu", epochs=1)
    assert checkpoint is not None, [p.detail for p in report.probes if p.status != "passed"]
    return work, sound, checkpoint


@pytest.fixture
def served_state(monkeypatch):
    """E4 publishes and reloads into the service's shared state; this puts the
    served fields back after the test (the flags are the conftest's)."""
    import src.api.main as api

    for name in ("pipeline", "kg", "_current_data_dir", "_current_checkpoint_path"):
        monkeypatch.setattr(api.app_state, name, getattr(api.app_state, name, None),
                            raising=False)
    return api


def _phase(probe, trained):
    work, sound, checkpoint = trained
    report = probe.Report()
    probe.phase_serving(report, work, sound, checkpoint, device="cpu")
    return {p.probe_id: p for p in report.probes}


# ------------------------------------------------------------------ the phase
def test_the_serving_phase_passes_and_identifies_what_it_serves(trained, served_state):
    probe = _probe()
    _, sound, checkpoint = trained

    probes = _phase(probe, trained)

    assert list(probes) == E_PROBES
    assert {k: p.status for k, p in probes.items()} == dict.fromkeys(E_PROBES, "passed"), (
        {k: p.detail for k, p in probes.items() if p.status != "passed"}
    )
    # E1's evidence: the identity, by the digests of the bytes the probe's own
    # reads took -- checked here against the files themselves.
    identity = probes["E1"].facts["served_identity"]

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    assert identity["checkpoint"] == {"state": "verified", "sha256": digest(checkpoint)}
    assert identity["graph"] == {
        "state": "verified", "kg_sha256": digest(sound / "kg.json"),
        "split_manifest_sha256": digest(sound / "split_manifest.json")}
    assert identity["dataset_record"] == {
        "state": "verified", "origin": "synthetic",
        "sha256": digest(sound / "kg.provenance.json")}
    # The demo workspace holds no shortest-path table: stated, not left blank.
    assert identity["shortest_paths"] == {"state": "absent", "build_id": None, "max_hops": None}
    assert identity["scoring_mode"] == "gnn_only"
    assert (identity["settings"]["device"], identity["settings"]["device_requested"]) == (
        "cpu", "cpu")
    assert str(trained[0]) not in json.dumps(identity), "the identity carries a path"
    assert probes["E5"].facts == {"served_identity_unchanged": True}
    assert probes["E7"].facts["served_identity_is_the_built_pipelines"] is True


def test_e1_fails_when_the_identity_names_another_checkpoint(trained, served_state,
                                                             monkeypatch):
    probe = _probe()
    real = probe._read_run_checkpoint
    monkeypatch.setattr(probe, "_read_run_checkpoint",
                        lambda path: {**real(path), "sha256": "0" * 64})

    probes = _phase(probe, trained)

    assert probes["E1"].status == "failed"
    assert "not the checkpoint this probe read" in probes["E1"].detail
    assert list(probes) == ["E1"], "the phase went on without a verified build"


def _status_without_the_identity(status):
    return status.model_copy(update={"served_identity": None})


def _status_with_a_default_device_request(status):
    identity = status.served_identity
    settings = identity.settings.model_copy(update={"device_requested": "auto"})
    return status.model_copy(
        update={"served_identity": identity.model_copy(update={"settings": settings})})


@pytest.mark.parametrize("change", [
    _status_without_the_identity, _status_with_a_default_device_request])
def test_e2_fails_when_the_status_does_not_report_the_built_identity(trained, served_state,
                                                                     monkeypatch, change):
    """The regression this catches: the status renders the configuration and
    drops the served identity (`_status_of` no longer passes it on), or reports
    a value other than the one served (here a default device request). The
    pipeline built is the same, so E1 passes and only E2 can see it."""
    import src.api.routes.pipeline as routes

    probe = _probe()
    real_status = routes._status_of
    monkeypatch.setattr(routes, "_status_of", lambda *args: change(real_status(*args)))

    probes = _phase(probe, trained)

    assert probes["E1"].status == "passed"
    assert (probes["E2"].status, probes["E2"].detail) == (
        "failed", "the status does not report the served identity the pipeline built")


def test_e5_fails_when_a_refused_reload_moves_the_served_identity(trained, served_state,
                                                                  monkeypatch):
    """The refusal keeps the same pipeline object and changes what it reports:
    only the identity check can see that."""
    import src.api.routes.pipeline as routes

    probe = _probe()
    api = served_state
    real_reload = routes.reload_pipeline

    async def reload_then_drift(request):
        response = await real_reload(request)
        if not response.success:
            api.app_state.pipeline._device_requested = "cuda"
        return response

    monkeypatch.setattr(routes, "reload_pipeline", reload_then_drift)

    probes = _phase(probe, trained)

    assert probes["E4"].status == "passed"
    assert (probes["E5"].status, probes["E5"].detail) == (
        "failed", "the served identity moved")


def test_e7_fails_when_a_diagnosis_carries_another_identity(trained, served_state,
                                                            monkeypatch):
    from src.inference.pipeline import DiagnosisPipeline

    probe = _probe()
    real_run = DiagnosisPipeline.run

    def run_elsewhere(self, *args, **kwargs):
        result = real_run(self, *args, **kwargs)
        identity = result.served_identity
        result.served_identity = dataclasses.replace(
            identity, settings=dataclasses.replace(identity.settings, device_requested="auto"))
        return result

    monkeypatch.setattr(DiagnosisPipeline, "run", run_elsewhere)

    probes = _phase(probe, trained)

    assert probes["E7"].status == "failed"
    assert "another served identity" in probes["E7"].detail


# --------------------------------------------- the comparison, part by part
@pytest.fixture(scope="module")
def verified(trained):
    """One serving build of the trained model, and what the probe's own reads
    say it should identify."""
    import src.api.main as api

    probe = _probe()
    _, sound, checkpoint = trained
    # `build_pipeline` sets this flag, and a module-scoped fixture runs outside
    # the conftest's per-test reset, so it is put back here.
    requested = api.app_state.real_pipeline_requested
    try:
        bundle = api.build_pipeline(kg_path=str(sound / "kg.json"),
                                    checkpoint_path=str(checkpoint),
                                    data_dir=str(sound), device="cpu")
    finally:
        api.app_state.real_pipeline_requested = requested
    _, _, read, record = probe.verify_workspace(sound)
    expected = {"checkpoint_sha256": probe._read_run_checkpoint(checkpoint)["sha256"],
                "workspace_digests": read, "record": record, "sp_build_id": None,
                "device": "cpu"}
    return probe, bundle.config["served_identity"], expected


def _changed(identity, part, **values):
    copy = json.loads(json.dumps(identity))
    if part is None:
        copy.update(values)
    else:
        copy[part].update(values)
    return copy


@pytest.mark.parametrize("part, values, fragment", [
    ("checkpoint", {"sha256": "1" * 64}, "not the checkpoint this probe read"),
    ("checkpoint", {"state": "supplied_in_memory", "sha256": None},
     "not the checkpoint this probe read"),
    ("graph", {"kg_sha256": "2" * 64}, "not the workspace this probe verified"),
    ("graph", {"split_manifest_sha256": "3" * 64}, "not the workspace this probe verified"),
    ("graph", {"state": "unverified", "split_manifest_sha256": None},
     "not the workspace this probe verified"),
    ("dataset_record", {"origin": "files"}, "not the record the workspace check verified"),
    ("dataset_record", {"sha256": "4" * 64}, "not the record the workspace check verified"),
    ("shortest_paths", {"state": "verified", "build_id": "a" * 32, "max_hops": 5},
     "for a workspace that holds none"),
    (None, {"scoring_mode": "gnn_plus_shortest_path"}, "scoring mode"),
    ("settings", {"device": "cuda:0"}, "not the requested cpu"),
    ("settings", {"device_requested": "auto"}, "not the requested cpu"),
])
def test_each_part_of_the_identity_is_compared_with_the_probes_reads(verified, part, values,
                                                                     fragment):
    probe, identity, expected = verified
    assert probe._assert_served_identity(identity, **expected) == identity

    with pytest.raises(AssertionError, match=fragment):
        probe._assert_served_identity(_changed(identity, part, **values), **expected)


def test_no_identity_and_a_table_not_served_are_refused(verified):
    probe, identity, expected = verified
    with pytest.raises(AssertionError, match="reports no served identity"):
        probe._assert_served_identity(None, **expected)
    # The sidecar declares a table, and the identity says none was served.
    with pytest.raises(AssertionError, match="not the table whose sidecar declares"):
        probe._assert_served_identity(identity, **{**expected, "sp_build_id": "b" * 32})


def test_the_sidecar_build_id_is_read_through_the_sidecars_reader(tmp_path):
    probe = _probe()
    assert probe._sidecar_build_id(tmp_path) is None, "no table is not a table"
    (tmp_path / "shortest_paths.pt").write_bytes(b"not read here")
    with pytest.raises(AssertionError, match="has no sidecar"):
        probe._sidecar_build_id(tmp_path)
    (tmp_path / "shortest_paths.meta.json").write_text(json.dumps(
        {"schema_version": 1, "build_id": "c" * 32, "kg_digest": "d" * 64, "max_hops": 5}))
    assert probe._sidecar_build_id(tmp_path) == "c" * 32


# ------------------------------------- M4 and R6: the diagnosis, checked
@pytest.fixture(scope="module")
def with_table(trained, tmp_path_factory):
    """The trained model's workspace copied and given a shortest-path table by
    the production producer, as M4 serves it, with what the workspace check
    reads of the copy: the record and the per-file digests the caller hands
    `_serve_and_diagnose`."""
    probe = _probe()
    _, sound, checkpoint = trained
    served = probe._workspace_copy(sound, tmp_path_factory.mktemp("with_table") / "ws")
    produced = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "compute_shortest_paths.py"),
         "--kg-path", str(served / "kg.json"), "--output-dir", str(served),
         "--max-hops", "5", "--workers", "1"],
        cwd=REPO, capture_output=True, text=True, timeout=600,
    )
    assert produced.returncode == 0, produced.stderr[-2000:]
    _, _, read, record = probe.verify_workspace(served)
    return served, checkpoint, read, record


def _serve(probe, with_table, **changes):
    served, checkpoint, read, record = with_table
    arguments = {"record": record, "workspace_digests": read, **changes}
    return probe._serve_and_diagnose(served / "kg.json", checkpoint, served, "cpu", **arguments)


def test_serve_and_diagnose_passes_over_the_workspace_its_caller_verified(with_table):
    """The control for the failures below: the same build and diagnosis, handed
    the caller's own reads, pass and name the table and the checkpoint."""
    from src.kg.artifacts import GRAPH_ARTIFACTS

    probe = _probe()
    _, checkpoint, read, _ = with_table

    identity = _serve(probe, with_table)["served_identity"]

    assert identity["scoring_mode"] == "gnn_plus_shortest_path"
    assert identity["shortest_paths"]["state"] == "verified"
    assert identity["checkpoint"] == {
        "state": "verified", "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    assert identity["graph"]["kg_sha256"] == read[GRAPH_ARTIFACTS["kg"]]


def test_serve_and_diagnose_fails_when_the_diagnosis_names_another_workspace(with_table):
    """The regression this catches: `_serve_and_diagnose` stops comparing the
    diagnosis's identity with the caller's workspace reads. Serving and the
    diagnosis still agree with each other, so only that comparison can see a
    graph other than the one the caller verified."""
    from src.kg.artifacts import GRAPH_ARTIFACTS

    probe = _probe()
    _, _, read, _ = with_table
    elsewhere = {**read, GRAPH_ARTIFACTS["kg"]: "5" * 64}

    with pytest.raises(AssertionError, match="not the workspace this probe verified"):
        _serve(probe, with_table, workspace_digests=elsewhere)


def test_serve_and_diagnose_fails_when_the_diagnosis_names_another_checkpoint(with_table,
                                                                              monkeypatch):
    """The same regression, seen from the checkpoint: the digest the probe's own
    read took is not the one the diagnosis names."""
    probe = _probe()
    real = probe._read_run_checkpoint
    monkeypatch.setattr(probe, "_read_run_checkpoint",
                        lambda path: {**real(path), "sha256": "6" * 64})

    with pytest.raises(AssertionError, match="not the checkpoint this probe read"):
        _serve(probe, with_table)


@pytest.mark.parametrize("part, values, fragment", [
    # A part the probe's reads determine: refused as not what the caller verified.
    ("dataset_record", {"origin": "files"}, "not the record the workspace check verified"),
    # A setting no read of the probe's determines: refused as not what serving reports.
    ("settings", {"max_paths_per_candidate": 1},
     "another served identity than serving reports"),
])
def test_serve_and_diagnose_fails_when_the_diagnosis_carries_another_identity(
    with_table, monkeypatch, part, values, fragment
):
    """The regressions this catches: the comparison is applied to what serving
    reports instead of to the diagnosis's own identity, or the diagnosis is no
    longer required to carry the identity serving reports. The build is the
    same, so the record serving publishes still matches; only the diagnosis
    differs."""
    from src.inference.pipeline import DiagnosisPipeline

    probe = _probe()
    real_run = DiagnosisPipeline.run

    def run_elsewhere(self, *args, **kwargs):
        result = real_run(self, *args, **kwargs)
        identity = result.served_identity
        result.served_identity = dataclasses.replace(
            identity, **{part: dataclasses.replace(getattr(identity, part), **values)})
        return result

    monkeypatch.setattr(DiagnosisPipeline, "run", run_elsewhere)

    with pytest.raises(AssertionError, match=fragment):
        _serve(probe, with_table)
