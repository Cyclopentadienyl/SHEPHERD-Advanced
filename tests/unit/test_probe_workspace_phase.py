"""The deployment probe's workspace phase (C), on CPU.

Phase C provokes the workspace contract's refusals on real files: a replaced
graph artifact, a missing manifest and another workspace's `kg.json`, each with
the sound workspace as its control. Since contract M2.1's S9 it checks them the
way every graph consumer does, from one read of each file (`verify_workspace`:
the shared readers and the read-form verifiers), so these tests run the phase
end to end and require every probe in it to pass.

Module: tests/unit/test_probe_workspace_phase.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


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
def written(tmp_path):
    probe = _probe()
    report = probe.Report()
    sound = probe.phase_writer(report, tmp_path)
    assert sound is not None, [p.detail for p in report.probes if p.status != "passed"]
    return probe, report, sound, tmp_path


def test_every_workspace_probe_passes_from_one_read_of_each_file(written):
    probe, report, sound, work = written
    before = len(report.probes)

    probe.phase_workspace(report, work, sound)

    phase = report.probes[before:]
    assert [p.probe_id for p in phase] == [
        "C1", "C2.edge_indices", "C2.kg", "C2.node_features", "C2.num_nodes", "C3", "C4",
        # Contract M3b: each way the dataset record can fail to bind.
        "C5.missing", "C5.replaced", "C5.other_graph", "C5.unbound", "C5.unreadable",
        "C5.unidentified",
    ]
    assert {p.probe_id: p.status for p in phase} == {p.probe_id: "passed" for p in phase}, (
        {p.probe_id: p.detail for p in phase if p.status != "passed"}
    )
    kg_digest = hashlib.sha256((sound / "kg.json").read_bytes()).hexdigest()
    assert phase[0].facts["kg_digest"] == kg_digest
    # The demo's record, shown with its origin, never as plain verified.
    assert phase[0].facts["dataset_record"] == {"binding": "verified", "origin": "synthetic"}
    assert {p.probe_id: p.facts["binding"] for p in phase if p.probe_id.startswith("C5.")} == {
        "C5.missing": "unverifiable", "C5.replaced": "mismatch",
        "C5.other_graph": "mismatch", "C5.unbound": "unrecorded",
        "C5.unreadable": "unverifiable", "C5.unidentified": "verified",
    }


def test_the_checks_open_each_workspace_file_once(written):
    from tests.fixtures.opens import count_opens

    probe, _, sound, _ = written

    with count_opens() as opens:
        bound, cohorts, read, record = probe.verify_workspace(sound)

    names = ("kg.json", "node_features.pt", "edge_indices.pt", "num_nodes.json",
             "train_samples.json", "val_samples.json", "split_manifest.json",
             "kg.provenance.json")
    for name in names:
        assert opens.opens(sound / name) == 1, name
    assert cohorts.verified == ("train", "val")
    assert bound["kg"] == hashlib.sha256((sound / "kg.json").read_bytes()).hexdigest()
    for name in names:
        assert read[name] == hashlib.sha256((sound / name).read_bytes()).hexdigest(), name
    assert (record.binding, record.origin) == ("verified", "synthetic")


def test_a_replacement_after_the_manifest_read_is_refused_naming_the_file(
    written, tmp_path, monkeypatch
):
    """The checks compare reads with one manifest reading, so a file swapped
    between the two is refused rather than verified against a newer manifest."""
    from src.kg.workspace import write_workspace
    from tests.fixtures.replacement import replace_after_read

    probe, _, sound, _ = written
    other = tmp_path / "other"

    write_workspace(probe._other_kg(), other, feature_dim=probe.FEATURE_DIM,
                    samples=probe._budget())
    target = sound / "node_features.pt"
    replace_after_read(
        monkeypatch, after=sound / "split_manifest.json", target=target,
        data=(other / "node_features.pt").read_bytes(), how="rename",
    )

    with pytest.raises(ValueError, match="is not the node_features artifact") as refused:
        probe.verify_workspace(sound)
    assert str(target) in str(refused.value)


def test_the_training_phase_checks_the_checkpoint_from_one_read(written):
    """Phase D on CPU, one epoch: D2 checks the trained checkpoint's recorded graph
    against the workspace from `read_checkpoint` and `verify_workspace`, the reads
    S9 moved it onto, and every probe in the phase passes."""
    probe, report, sound, work = written
    before = len(report.probes)

    selected = probe.phase_training(report, work, sound, device="cpu", epochs=1)

    phase = report.probes[before:]
    assert {"D1", "D2"} <= {p.probe_id for p in phase}, [p.probe_id for p in phase]
    assert {p.probe_id: p.status for p in phase} == {p.probe_id: "passed" for p in phase}, (
        {p.probe_id: p.detail for p in phase if p.status != "passed"}
    )
    assert selected is not None
