"""
Measurement reads each recorded input once, and records those reads — Mode C.
=============================================================================
Contract M2.1, S7, and M4 §6. Before any model is built or output written,
`main` reads the manifest, the graph tensors, the cohort's samples and the
checkpoint once each (`read_measurement_inputs`), and verifies from those reads
with the `kg.json` identification check (decision 1). A generated split is
resolved first (stat only) and its sibling split is read for R2; a registered
cohort version is resolved by one read of its `cohort.json`, its `samples.json`
is read once against the digest that manifest binds, and the workspace's own
`train_samples.json` and `val_samples.json` are read only as R10 candidates.
The manifest's `artifact_digests` are built only from those reads
(`measurement_input_digests`); its `kg` role is the manifest-bound source graph
(decision 2). The acceptance shown here, at the real entry point (`main`, with
Mode C's real scoring), for a generated split (`--split val` on the synthetic
workspace) and a registered version (`--cohort` on the demo workspace,
`tests/fixtures/registered_cohort.py`):

- each input is opened once, and the record equals the digests of the files;
- a graph tensor, the cohort's sample file or the checkpoint replaced after its
  read, by atomic rename or in-place rewrite, is the one encoded, scored or
  built and the one recorded; a graph file (`kg.json` included) or a generated
  split replaced after the manifest read is refused, the refusal names it and the
  manifest, and no model is built, no checkpoint read and nothing written; a
  version's samples replaced after its `cohort.json` was read are refused by
  digest, unparsed;
- a republished manifest is refused before the file it covers is read, is
  irrelevant after both reads (for `kg.json`, after its check), and is read and
  recorded when it lands before the run of a generated split, while a version
  mapped against the graph before the republication is refused by its binding;
- the cohort is resolved before the workspace is opened; a workspace with no
  manifest, or with no `kg.json`, is refused before the checkpoint is read;
- with every path hash poisoned except the one the `kg.json` identification
  check takes (it records nothing), the run completes and no poison reaches the
  record;
- graph tensors load on CPU and safely; the checkpoint keeps this site's options;
  the samples are read with the default fields.

Module: tests/unit/test_measurement_reads_once.py
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tests.fixtures.opens import count_opens  # noqa: E402
from tests.fixtures.path_forms import POISON, poison_path_forms  # noqa: E402
from tests.fixtures.registered_cohort import (  # noqa: E402
    build_registered_cohort,
    copy_registered_cohort,
)
from tests.fixtures.replacement import (  # noqa: E402
    HOWS,
    hook_read_once,
    replace_after_read,
    replace_file,
)
from tests.fixtures.synthetic_workspace import build_workspace  # noqa: E402

BATCH_SIZE = 3
GRAPH_FILES = {"node_features": "node_features.pt", "edge_indices": "edge_indices.pt",
               "num_nodes": "num_nodes.json"}
KINDS = ("generated", "registered")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def registered_base(tmp_path_factory):
    """Built once per module; each test measures its own copy."""
    return build_registered_cohort(tmp_path_factory.mktemp("registered_base"))


def _generated(root):
    data_dir, checkpoint = build_workspace(root)
    return SimpleNamespace(kind="generated", data_dir=data_dir, checkpoint=checkpoint,
                           cohort_samples=data_dir / "val_samples.json",
                           output=root / "out" / "measurement.json")


def _registered(base, root):
    cohort = copy_registered_cohort(base, root)
    return SimpleNamespace(kind="registered", data_dir=cohort.data_dir,
                           checkpoint=cohort.checkpoint, registry=cohort.registry,
                           version=cohort.version, label=cohort.version.manifest["label"],
                           cohort_samples=cohort.version.samples_path,
                           cohort_manifest=cohort.version.directory / "cohort.json")


@pytest.fixture
def cli():
    import scripts.measure_scorer as module

    return module


@pytest.fixture(params=KINDS)
def ws(request, tmp_path, registered_base):
    if request.param == "generated":
        return _generated(tmp_path / "a")
    return _registered(registered_base, tmp_path / "a")


@pytest.fixture
def generated(tmp_path):
    return _generated(tmp_path / "a")


@pytest.fixture
def registered(tmp_path, registered_base):
    return _registered(registered_base, tmp_path / "a")


@pytest.fixture
def other(tmp_path):
    """Another workspace: the same layout, different bytes in every graph file
    (five diseases instead of four). Its files are what a replacement writes;
    they parse, so a refusal is the binding's, not a parse. Its `test` file is
    the replacement for a samples file: its generated cut is the same as this
    workspace's, so its `train` and `val` would be no replacement at all."""
    data_dir, checkpoint = build_workspace(tmp_path / "b", n_diseases=5)
    return data_dir


def _other_checkpoint(checkpoint, path):
    """The same architecture with shifted weights and another epoch."""
    contents = torch.load(checkpoint, map_location="cpu", weights_only=True)
    contents["state_dict"] = {
        name: value + 1.0 if value.is_floating_point() else value
        for name, value in contents["state_dict"].items()
    }
    contents["epoch"] = 9
    torch.save(contents, path)
    return path


def _argv(ws, *, split=None, cohort=None, registry=None, extra=()):
    common = ["--checkpoint", str(ws.checkpoint), "--data-dir", str(ws.data_dir),
              "--batch-size", str(BATCH_SIZE), "--device", "cpu"]
    if ws.kind == "generated":
        return [*common, "--split", split or "val", "--output", str(ws.output), *extra]
    return [*common, "--cohort", cohort or ws.label, "--registry", str(registry or ws.registry),
            *extra]


def _main(cli, ws, **kwargs):
    return cli.main(_argv(ws, **kwargs))


def _report_path(ws):
    if ws.kind == "generated":
        return ws.output
    reports = sorted((ws.version.directory / "reports").glob("*/report.json"))
    assert len(reports) == 1, reports
    return reports[0]


def _ranks_path(ws):
    report = _report_path(ws)
    if ws.kind == "generated":
        return report.with_name("measurement_ranks.json")
    return report.with_name("ranks.json")


def _recorded(ws):
    return json.loads(_report_path(ws).read_text())["manifest"]["artifact_digests"]


def _expected(ws):
    """The record a run must write: the digest of each file it read, and `kg` as
    the manifest binds it; `split_manifest` for a generated split only."""
    manifest = json.loads((ws.data_dir / "split_manifest.json").read_text())
    expected = {
        "checkpoint": _sha(ws.checkpoint),
        "samples": _sha(ws.cohort_samples),
        **{role: _sha(ws.data_dir / name) for role, name in GRAPH_FILES.items()},
        "kg": manifest["artifacts"]["kg"],
    }
    if ws.kind == "generated":
        expected["split_manifest"] = _sha(ws.data_dir / "split_manifest.json")
    return expected


@pytest.fixture
def seen(monkeypatch):
    """What the model was built from and what was encoded. `main` imports both
    inside the function, so the patches land on the source modules."""
    import src.evaluation.measurement as measurement
    import src.models.gnn.shepherd_gnn as shepherd_gnn

    seen = {}
    real_build, real_encode = shepherd_gnn.build_shepherd_model, measurement.encode_full_graph

    def build(checkpoint, graph_data, device=None):
        seen["checkpoint"] = checkpoint
        return real_build(checkpoint, graph_data, device)

    def encode(model, graph_data, device):
        seen["graph_data"] = graph_data
        return real_encode(model, graph_data, device)

    monkeypatch.setattr(shepherd_gnn, "build_shepherd_model", build)
    monkeypatch.setattr(measurement, "encode_full_graph", encode)
    return seen


@pytest.fixture
def no_model(monkeypatch):
    """Building or encoding a model fails if it is reached."""
    import src.evaluation.measurement as measurement
    import src.models.gnn.shepherd_gnn as shepherd_gnn

    def reached(*args, **kwargs):
        raise AssertionError("a model was built or encoded after a refusal")

    monkeypatch.setattr(shepherd_gnn, "build_shepherd_model", reached)
    monkeypatch.setattr(measurement, "encode_full_graph", reached)


def _nothing_written(ws):
    """No report anywhere it could go, and no ledger beside the checkpoint."""
    if ws.kind == "generated":
        return not ws.output.parent.exists()
    return (not (ws.version.directory / "reports").exists()
            and not (ws.checkpoint.parent / "evaluations.json").exists())


# ---------------------------------------------------------------------------
# One read each, and the record is those reads
# ---------------------------------------------------------------------------
def test_each_input_is_opened_once(cli, ws, tmp_path):
    """Both kinds: every workspace file the run reads, once, and the checkpoint
    once. A generated split reads its sibling for R2. A registered version reads
    its `cohort.json` and `samples.json` once each, and the workspace's `train`
    and `val` only as R10 candidates, once each; the fixture checkpoint records
    both, so once both are matched no candidate in an evidence directory is
    opened at all."""
    extra = ()
    evidence = tmp_path / "evidence"
    if ws.kind == "registered":
        evidence.mkdir()
        for name in ("train_samples.json", "val_samples.json"):
            (evidence / f"copy_{name}").write_bytes((ws.data_dir / name).read_bytes())
        extra = ("--training-evidence", str(evidence))

    with count_opens() as opens:
        assert _main(cli, ws, extra=extra) == 0

    read = ["kg.json", "split_manifest.json", "kg.provenance.json", *GRAPH_FILES.values(),
            "train_samples.json", "val_samples.json"]
    for name in read:
        assert opens.opens(ws.data_dir / name) == 1, name
    assert opens.opens(ws.checkpoint) == 1
    if ws.kind == "generated":
        assert opens.opens(ws.data_dir / "test_samples.json") == 0, "no input of this run"
    else:
        assert opens.opens(ws.cohort_manifest) == 1
        assert opens.opens(ws.cohort_samples) == 1
        for name in ("train_samples.json", "val_samples.json"):
            assert opens.opens(evidence / f"copy_{name}") == 0, (
                "an evidence file was read after both scopes were matched"
            )


def test_the_record_is_the_digests_of_the_bytes_read(cli, ws):
    """Exactly these roles: the five files read, `split_manifest` for a generated
    split only, and `kg` as the manifest binds it."""
    assert _main(cli, ws) == 0

    assert _recorded(ws) == _expected(ws)


def test_nothing_else_in_the_directory_is_opened_or_becomes_a_role(cli, ws):
    """A record of the directory is not a record of the run. For a registered
    version the workspace's other `*_samples.json` files are not R10 candidates
    either: only its own `train` and `val` are."""
    (ws.data_dir / "notes.txt").write_bytes(b"scratch")
    (ws.data_dir / "mygene2_samples.json").write_bytes(b"[]")

    with count_opens() as opens:
        assert _main(cli, ws) == 0

    assert opens.opens(ws.data_dir / "notes.txt") == 0
    assert opens.opens(ws.data_dir / "mygene2_samples.json") == 0
    assert _recorded(ws) == _expected(ws)


# ---------------------------------------------------------------------------
# Replacements: used and recorded, or refused
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("how", HOWS)
def test_a_graph_tensor_replaced_after_its_read_is_encoded_and_recorded(
    cli, ws, other, seen, monkeypatch, how
):
    target = ws.data_dir / "node_features.pt"
    read = _sha(target)
    tensors = torch.load(target, weights_only=True)
    record = replace_after_read(monkeypatch, after=target, target=target,
                                data=(other / "node_features.pt").read_bytes(), how=how)

    assert _main(cli, ws) == 0

    assert record.fired == 1
    assert _recorded(ws)["node_features"] == read != _sha(target)
    for node_type, features in tensors.items():
        assert torch.equal(seen["graph_data"]["x_dict"][node_type], features)


@pytest.mark.parametrize("how", HOWS)
def test_the_cohort_replaced_after_its_read_is_scored_and_recorded(
    cli, ws, other, monkeypatch, how
):
    target = ws.cohort_samples
    read = _sha(target)
    patients = [row["patient_id"] for row in json.loads(target.read_text())]
    record = replace_after_read(monkeypatch, after=target, target=target,
                                data=(other / "test_samples.json").read_bytes(), how=how)

    assert _main(cli, ws) == 0

    assert record.fired == 1
    assert _recorded(ws)["samples"] == read != _sha(target)
    ranks = json.loads(_ranks_path(ws).read_text())
    assert [row["sample_id"] for row in ranks] == patients


@pytest.mark.parametrize("how", HOWS)
def test_the_checkpoint_replaced_after_its_read_builds_the_model_read(
    cli, ws, seen, monkeypatch, tmp_path, how
):
    read = _sha(ws.checkpoint)
    state = torch.load(ws.checkpoint, weights_only=True)["state_dict"]
    _other_checkpoint(ws.checkpoint, tmp_path / "b.pt")
    record = replace_after_read(monkeypatch, after=ws.checkpoint, target=ws.checkpoint,
                                data=(tmp_path / "b.pt").read_bytes(), how=how)

    assert _main(cli, ws) == 0

    assert record.fired == 1
    assert _recorded(ws)["checkpoint"] == read != _sha(ws.checkpoint)
    assert seen["checkpoint"]["epoch"] == 1, "the checkpoint built is A, not B"
    for name, value in state.items():
        assert torch.equal(seen["checkpoint"]["state_dict"][name], value)


def _refused_after_the_manifest_read(cli, ws, monkeypatch, target, data, how):
    with count_opens() as opens:
        record = replace_after_read(monkeypatch, after=ws.data_dir / "split_manifest.json",
                                    target=target, data=data, how=how)
        with pytest.raises(ValueError) as refused:
            _main(cli, ws)

    assert record.fired == 1
    assert str(target) in str(refused.value)
    assert "split_manifest.json" in str(refused.value)
    assert opens.opens(ws.checkpoint) == 0, "a refused workspace reads no checkpoint"
    assert _nothing_written(ws)
    return refused


@pytest.mark.parametrize("how", HOWS)
@pytest.mark.parametrize("name", [*GRAPH_FILES.values(), "kg.json"])
def test_a_graph_file_replaced_after_the_manifest_read_is_refused_naming_it(
    cli, ws, other, no_model, monkeypatch, name, how
):
    """`kg.json` alone included: measurement never parses it, and the
    identification check compares it with the run's one manifest reading."""
    data = b'{"synthetic": "another graph"}' if name == "kg.json" else (other / name).read_bytes()
    _refused_after_the_manifest_read(cli, ws, monkeypatch, ws.data_dir / name, data, how)


@pytest.mark.parametrize("how", HOWS)
@pytest.mark.parametrize("name", ["val_samples.json", "train_samples.json"])
def test_a_generated_split_replaced_after_the_manifest_read_is_refused_naming_it(
    cli, generated, other, no_model, monkeypatch, name, how
):
    """The cohort scored and its sibling, which is read for its binding and the
    disjointness check."""
    _refused_after_the_manifest_read(cli, generated, monkeypatch, generated.data_dir / name,
                                     (other / "test_samples.json").read_bytes(), how)


@pytest.mark.parametrize("how", HOWS)
def test_a_versions_samples_replaced_after_its_manifest_read_are_refused_unparsed(
    cli, registered, no_model, monkeypatch, how
):
    """`cohort.json` is read first and binds `samples.json` by digest; samples
    replaced in between are refused on that digest, and the refusal names both
    files. The replacement is not even JSON, so a parse before the comparison
    would surface as a parse error instead of this refusal."""
    with count_opens() as opens:
        record = replace_after_read(monkeypatch, after=registered.cohort_manifest,
                                    target=registered.cohort_samples,
                                    data=b"\xff not a samples file", how=how)
        with pytest.raises(ValueError, match="is not the samples") as refused:
            _main(cli, registered)

    assert record.fired == 1
    assert str(registered.cohort_samples) in str(refused.value)
    assert str(registered.cohort_manifest) in str(refused.value)
    assert "Nothing was written" in str(refused.value)
    assert opens.opens(registered.checkpoint) == 0
    assert _nothing_written(registered)


@pytest.mark.parametrize("how", HOWS)
def test_a_manifest_republished_before_the_tensor_read_is_refused(
    cli, ws, other, no_model, monkeypatch, how
):
    """The run's one reading of the manifest is A; the tensor read after the
    republication is B; the comparison is between those reads."""
    target = ws.data_dir / "node_features.pt"
    record = replace_after_read(monkeypatch, after=ws.data_dir / "split_manifest.json",
                                target=target, data=(other / "node_features.pt").read_bytes(),
                                how=how, republish="node_features")

    with pytest.raises(ValueError, match="is not the node_features artifact"):
        _main(cli, ws)

    assert record.fired == 1
    assert _nothing_written(ws)


@pytest.mark.parametrize("how", HOWS)
def test_a_manifest_republished_after_both_reads_changes_nothing_recorded(
    cli, ws, other, monkeypatch, how
):
    manifest = ws.data_dir / "split_manifest.json"
    target = ws.data_dir / "node_features.pt"
    expected = _expected(ws)
    record = replace_after_read(monkeypatch, after=target, target=target,
                                data=(other / "node_features.pt").read_bytes(), how=how,
                                republish="node_features")

    assert _main(cli, ws) == 0

    assert record.fired == 1
    assert _recorded(ws) == expected
    if ws.kind == "generated":
        assert expected["split_manifest"] != _sha(manifest)


@pytest.mark.parametrize("how", HOWS)
def test_kg_json_republished_after_its_check_leaves_the_run_on_what_was_checked(
    cli, ws, monkeypatch, how
):
    """The checkpoint is read after the identification check, so a republication
    landing after that read comes after both the manifest read and the check."""
    expected = _expected(ws)
    record = replace_after_read(monkeypatch, after=ws.checkpoint, target=ws.data_dir / "kg.json",
                                data=b'{"synthetic": "republished"}', how=how, republish="kg")

    assert _main(cli, ws) == 0

    assert record.fired == 1
    assert _recorded(ws) == expected
    assert expected["kg"] != _sha(ws.data_dir / "kg.json")


def _republish_node_features_before_the_run(ws, data):
    """B's tensor, a manifest binding it, and a checkpoint trained on it."""
    target, manifest_path = ws.data_dir / "node_features.pt", ws.data_dir / "split_manifest.json"
    replace_file(target, data, "rename")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["node_features"] = hashlib.sha256(data).hexdigest()
    replace_file(manifest_path, json.dumps(manifest).encode("utf-8"), "rename")
    contents = torch.load(ws.checkpoint, map_location="cpu", weights_only=True)
    contents["training_input_digests"]["node_features"] = hashlib.sha256(data).hexdigest()
    torch.save(contents, ws.checkpoint)


def test_a_republication_before_the_run_is_read_and_recorded(cli, generated, other):
    """Fired before either read, the run reads and records B.

    The checkpoint is one trained on the republished graph: a checkpoint that
    records A's tensors is refused against B (M2.4), which is not this test's
    subject."""
    data = (other / "node_features.pt").read_bytes()
    _republish_node_features_before_the_run(generated, data)

    assert _main(cli, generated) == 0

    assert _recorded(generated)["node_features"] == hashlib.sha256(data).hexdigest()
    assert _recorded(generated) == _expected(generated)


def test_a_republication_before_the_run_refuses_a_version_mapped_against_the_old_graph(
    cli, registered, other, no_model
):
    """The same republication, for a version imported before it: the workspace
    and the checkpoint agree on B, but the version's indices were mapped against
    A, so its binding refuses before the checkpoint is read."""
    _republish_node_features_before_the_run(registered,
                                            (other / "node_features.pt").read_bytes())

    with count_opens() as opens:
        with pytest.raises(ValueError, match="mapped against another graph") as refused:
            _main(cli, registered)

    assert refused.value.roles == ("node_features",)
    assert opens.opens(registered.checkpoint) == 0
    assert _nothing_written(registered)


@pytest.mark.parametrize("target", ["cohort samples", "checkpoint"])
def test_a_file_removed_after_its_read_is_still_recorded_never_as_none(
    cli, ws, monkeypatch, target
):
    target = ws.checkpoint if target == "checkpoint" else ws.cohort_samples
    read = _sha(target)

    def remove(path):
        if Path(path).resolve() == target.resolve():
            target.unlink()

    hook_read_once(monkeypatch, on_return=remove)

    assert _main(cli, ws) == 0

    assert not target.exists(), "the hook did not fire"
    recorded = _recorded(ws)
    assert None not in recorded.values()
    assert recorded["checkpoint" if target == ws.checkpoint else "samples"] == read


# ---------------------------------------------------------------------------
# Refused before anything is read, built or written
# ---------------------------------------------------------------------------
def _opened_under(opens, *roots):
    watched = tuple(os.path.realpath(root) for root in roots)
    return [path for path, n in opens.counts.items() if n and path.startswith(watched)]


@pytest.mark.parametrize("case", ["absolute", "not_generated", "supplied_name", "missing"])
def test_a_generated_split_is_resolved_before_anything_is_read(
    cli, generated, no_model, tmp_path, case
):
    """An absolute split used to open and parse a file outside the workspace
    before the cohort was resolved. A name that is not the generator's -- the
    loose supplied cohort `test` the workspace still holds, or any other -- is
    refused pointing at the registered path, before anything is opened."""
    outside = tmp_path / "outside_samples.json"
    outside.write_text((generated.data_dir / "test_samples.json").read_text())
    if case == "missing":
        (generated.data_dir / "val_samples.json").unlink()
    split, message = {
        "absolute": (str(tmp_path / "outside"), "must be an identifier"),
        "not_generated": ("test", "not one of the generator's splits"),
        "supplied_name": ("mygene2", "--cohort LABEL --registry DIR"),
        "missing": ("val", "does not exist"),
    }[case]

    with count_opens() as opens:
        with pytest.raises(ValueError, match=message):
            _main(cli, generated, split=split)

    opened = _opened_under(opens, generated.data_dir, outside, generated.checkpoint)
    assert opened == [], f"opened before the cohort was resolved: {opened}"
    assert _nothing_written(generated)


@pytest.mark.parametrize("case", [
    "label_path", "no_such_label", "interrupted", "relative_registry", "registry_in_repo",
    "missing_registry", "missing_evidence_dir",
])
def test_a_registered_cohort_is_resolved_before_the_workspace_is_read(
    cli, registered, no_model, tmp_path, case
):
    """The registry, the label and the evidence directories are the operator's
    arguments, and each is refused before any workspace file or the checkpoint
    is opened, with nothing written: a label that is a path, a version that is
    not there, an interrupted import, a registry that is relative, inside the
    repository or missing, an evidence directory that does not exist."""
    from src.evaluation.cohort_registry import REPO_ROOT

    cohort, registry, extra = registered.label, registered.registry, ()
    if case == "label_path":
        cohort, message = str(tmp_path / "x"), "cohort label is refused"
    elif case == "no_such_label":
        cohort, message = "never-imported", "holds no cohort version"
    elif case == "interrupted":
        (registered.registry / "versions" / "half-done").mkdir()
        cohort, message = "half-done", "interrupted import"
    elif case == "relative_registry":
        registry, message = Path("registry"), "not an absolute path"
    elif case == "registry_in_repo":
        registry, message = REPO_ROOT / "tests", "inside the repository"
    elif case == "missing_registry":
        registry, message = tmp_path / "no-registry", "does not exist"
    else:
        extra = ("--training-evidence", str(tmp_path / "no-evidence"))
        message = "Training evidence directory"

    with count_opens() as opens:
        with pytest.raises(ValueError, match=message):
            _main(cli, registered, cohort=cohort, registry=registry, extra=extra)

    opened = _opened_under(opens, registered.data_dir, registered.checkpoint)
    assert opened == [], f"opened before the cohort was resolved: {opened}"
    assert _nothing_written(registered)
    assert not (tmp_path / "no-registry").exists(), "a registry root was created"


def test_a_missing_sibling_split_is_refused_before_anything_is_read(cli, generated, no_model):
    (generated.data_dir / "train_samples.json").unlink()

    with count_opens() as opens:
        with pytest.raises(ValueError, match="train_samples.json does not exist"):
            _main(cli, generated)

    assert not _opened_under(opens, generated.data_dir)
    assert _nothing_written(generated)


def test_the_sibling_split_is_released_before_the_checkpoint_is_read(cli, generated, monkeypatch):
    """The sibling is checked, not consumed: none of its samples is still alive
    when the checkpoint is read, so the run does not hold both at once."""
    import gc
    import weakref

    import src.kg.storage.file_storage as file_storage
    import src.utils.checkpoint_io as checkpoint_io

    sibling = []
    alive_at_checkpoint = []
    real_read_samples, real_read_checkpoint = (
        file_storage.read_samples, checkpoint_io.read_checkpoint)

    def read_samples(data_dir, split, **kwargs):
        read = real_read_samples(data_dir, split, **kwargs)
        if split != "val":
            sibling.extend(weakref.ref(sample) for sample in read.samples)
        return read

    def read_checkpoint(*args, **kwargs):
        gc.collect()
        alive_at_checkpoint.append(sum(ref() is not None for ref in sibling))
        return real_read_checkpoint(*args, **kwargs)

    monkeypatch.setattr(file_storage, "read_samples", read_samples)
    monkeypatch.setattr(checkpoint_io, "read_checkpoint", read_checkpoint)

    assert _main(cli, generated) == 0

    assert sibling, "the sibling split was read"
    assert alive_at_checkpoint == [0]


def test_a_workspace_without_a_manifest_is_refused_before_anything_is_opened(
    cli, ws, no_model
):
    (ws.data_dir / "split_manifest.json").unlink()

    with count_opens() as opens:
        with pytest.raises(ValueError, match="split_manifest.json"):
            _main(cli, ws)

    assert not _opened_under(opens, ws.data_dir, ws.checkpoint)
    assert _nothing_written(ws)


def test_an_absent_kg_json_is_refused_naming_it(cli, ws, no_model):
    """Measurement never parses `kg.json`, but the identification check still
    needs the file the manifest binds to be there."""
    (ws.data_dir / "kg.json").unlink()

    with count_opens() as opens:
        with pytest.raises(ValueError, match="kg.json is absent"):
            _main(cli, ws)

    assert opens.opens(ws.checkpoint) == 0, "a refused workspace reads no checkpoint"
    assert _nothing_written(ws)


def test_the_sibling_split_puts_disjointness_in_scope(cli, generated, no_model):
    """Both generated splits are verified, so the manifest's `disjoint` claim is
    checked; scoped to the cohort alone it would not be."""
    path = generated.data_dir / "split_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["disjoint"] = False
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="claims disjoint=False"):
        _main(cli, generated)

    assert _nothing_written(generated)


def test_a_graph_file_from_another_workspace_is_refused(cli, ws, other, no_model):
    """The graph binding, on a valid file: junk bytes now fail in the reader's
    parse instead (next test)."""
    target = ws.data_dir / "edge_indices.pt"
    target.write_bytes((other / "edge_indices.pt").read_bytes())

    with pytest.raises(ValueError, match="is not the edge_indices artifact") as refused:
        _main(cli, ws)

    assert str(target) in str(refused.value)
    assert _nothing_written(ws)


def test_a_tensor_file_that_does_not_parse_is_refused_naming_it(cli, ws, no_model):
    target = ws.data_dir / "edge_indices.pt"
    target.write_bytes(b"from another workspace")

    with pytest.raises(ValueError, match="could not be parsed") as refused:
        _main(cli, ws)

    assert str(target) in str(refused.value)
    assert _nothing_written(ws)


# ---------------------------------------------------------------------------
# Only from the readers
# ---------------------------------------------------------------------------
def test_no_path_form_is_reached_and_no_path_digest_is_recorded(cli, ws, monkeypatch):
    """S9's dynamic pin for measurement: every path hash is poisoned except the
    `kg.json` identification check's, and the record is still the digests of
    the bytes read. `measure_scorer` binds no path hash at import since S9
    removed the `file_sha256` re-export, so none is out of the poison's reach."""
    for name in ("artifact_digests", "verify_graph_artifacts", "verify_generated_cohorts",
                 "compute_input_digests", "file_sha256", "_file_sha256"):
        assert not hasattr(cli, name), f"measure_scorer binds {name} at import"
    poison_path_forms(monkeypatch, true_digest_for=[ws.data_dir / "kg.json"])

    assert _main(cli, ws) == 0

    recorded = _recorded(ws)
    assert POISON not in recorded.values()
    assert recorded == _expected(ws)


def test_graph_tensors_load_on_cpu_and_safely_and_the_checkpoint_keeps_its_options(
    cli, ws, monkeypatch
):
    """Raw values, not their strings: `str(torch.device("cpu")) == "cpu"`, so a
    graph read moved to the run's device, or a checkpoint read moved off it, would
    pass a string comparison on a CPU-only machine."""
    real_load = torch.load
    calls = []

    def recording_load(*args, **kwargs):
        caller = sys._getframe(1).f_globals.get("__name__")
        calls.append((caller, kwargs.get("map_location"), kwargs.get("weights_only")))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)

    assert _main(cli, ws) == 0

    graph = [(m, type(m), w) for c, m, w in calls if c == "src.kg.storage.file_storage"]
    checkpoint = [(m, type(m), w) for c, m, w in calls if c == "src.utils.checkpoint_io"]
    assert graph == [("cpu", str, True)] * 2
    assert checkpoint == [(torch.device("cpu"), torch.device, True)]
    assert len(calls) == 3, f"a load outside the readers: {calls}"


def test_the_samples_are_read_with_the_default_fields(cli, ws, monkeypatch):
    """Mode C scores phenotypes against every disease; the two fields only
    training reads stay unread, for the cohort and for R10's evidence alike.
    Every samples read goes through `read_samples_file`, the one parser."""
    import src.evaluation.r10 as r10
    import src.kg.storage.file_storage as file_storage

    real = file_storage.read_samples_file
    seen = []

    def spy(path, **kwargs):
        seen.append((Path(path).name, kwargs.get("training_fields", False)))
        return real(path, **kwargs)

    monkeypatch.setattr(file_storage, "read_samples_file", spy)
    monkeypatch.setattr(r10, "read_samples_file", spy)

    assert _main(cli, ws) == 0

    expected = ({"val_samples.json", "train_samples.json"} if ws.kind == "generated"
                else {"samples.json", "train_samples.json", "val_samples.json"})
    assert {name for name, _ in seen} == expected
    assert all(fields is False for _, fields in seen), seen


def test_the_read_and_the_record_hash_no_path(cli):
    """Static half of the pin: neither function, nor any helper the read calls,
    names a path form, a path hash or `torch.load`."""
    gone = {"verify_graph_artifacts", "verify_generated_cohorts",
            "verify_graph_source", "compute_input_digests", "file_sha256", "_file_sha256",
            # Any other way of reading or hashing a file by path.
            "hashlib", "file_digest", "read_once", "read_bytes", "read_text", "open"}
    functions = (cli.read_measurement_inputs, cli.measurement_input_digests,
                 cli.cohort_version_block, cli._resolve_generated, cli._resolve_registered,
                 cli._read_generated, cli._read_version_samples, cli._verify_generated,
                 cli._verify_version_binding, cli._r10)
    for function in functions:
        tree = ast.parse(inspect.getsource(function).lstrip())
        body = tree.body[0]
        names = {node.id for node in ast.walk(body) if isinstance(node, ast.Name)}
        names |= {node.attr for node in ast.walk(body) if isinstance(node, ast.Attribute)}
        names |= {alias.name for node in ast.walk(body)
                  if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
        assert not names & gone, f"{function.__name__} names {sorted(names & gone)}"
        assert "torch.load" not in ast.unparse(body), function.__name__
