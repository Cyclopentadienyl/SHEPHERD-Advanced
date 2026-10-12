"""
Measurement reads each recorded input once, and records those reads — Mode C.
=============================================================================
Contract M2.1, S7. Before any model is built or output written, `main`
resolves the cohort (stat only), then reads the manifest, the graph tensors, the
cohort's samples, for a generated cohort the sibling split, and the checkpoint
once each (`read_measurement_inputs`), and verifies from those reads with the
`kg.json` identification check (decision 1).
The manifest's `artifact_digests` are built only from those reads
(`measurement_input_digests`); its `kg` role is the manifest-bound source graph
(decision 2). The acceptance shown here, at the real entry point (`main`, with
Mode C's real scoring on the synthetic workspace), for a supplied and a
generated cohort:

- each input is opened once, and the record equals the digests of the files;
- a graph tensor, the cohort's sample file or the checkpoint replaced after its
  read, by atomic rename or in-place rewrite, is the one encoded, scored or
  built and the one recorded; a graph file (`kg.json` included) or a generated
  split replaced after the manifest read is refused, the refusal names it and the
  manifest, and no model is built, no checkpoint read and nothing written;
- a republished manifest is refused before the file it covers is read, is
  irrelevant after both reads (for `kg.json`, after its check), and is read and
  recorded when it lands before the run;
- the cohort is resolved before anything is opened; a workspace with no
  manifest, or with no `kg.json`, is refused before the checkpoint is read;
- with every path form raising and every path hash poisoned except the one the
  `kg.json` identification check takes (it records nothing), the run completes
  and no poison reaches the record;
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

from tests.fixtures.generated_workspace import write_generated_workspace  # noqa: E402
from tests.fixtures.opens import count_opens  # noqa: E402
from tests.fixtures.path_forms import POISON, poison_path_forms  # noqa: E402
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
#: split, cohort kind
COHORTS = {"supplied": ("test", "supplied"), "generated": ("val", "generated")}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _recut(data_dir):
    """Generated train and val whose ids fit the synthetic graph (three
    phenotypes, four diseases), bound by a real manifest. `build_workspace`'s own
    generated cohorts exist only to sit beside a supplied one."""
    def rows(split, ids):
        return [{"patient_id": f"{split}-{d}", "phenotype_ids": [d % 3, (d + 1) % 3],
                 "disease_id": d} for d in ids]

    write_generated_workspace(
        data_dir, train_ids=[0, 1], val_ids=[2, 3],
        profiles={d: {"phenotype_ids": [d % 3], "gene_ids": [0]} for d in range(4)},
        train_samples=rows("train", [0, 1]), val_samples=rows("val", [2, 3]),
    )


def _workspace(root, kind):
    data_dir, checkpoint = build_workspace(root)
    if kind == "generated":
        _recut(data_dir)
    split, kind = COHORTS[kind]
    return SimpleNamespace(data_dir=data_dir, checkpoint=checkpoint, split=split, kind=kind,
                           output=root / "out" / "measurement.json")


@pytest.fixture
def cli():
    import scripts.measure_scorer as module

    return module


@pytest.fixture(params=sorted(COHORTS))
def ws(request, tmp_path):
    return _workspace(tmp_path / "a", request.param)


@pytest.fixture
def generated(tmp_path):
    return _workspace(tmp_path / "a", "generated")


@pytest.fixture
def supplied(tmp_path):
    return _workspace(tmp_path / "a", "supplied")


@pytest.fixture
def other(tmp_path):
    """Another workspace: the same layout, different bytes in every graph file
    and every sample file (five diseases instead of four)."""
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


def _main(cli, ws, *, split=None, kind=None):
    return cli.main([
        "--checkpoint", str(ws.checkpoint), "--data-dir", str(ws.data_dir),
        "--split", split or ws.split, "--cohort-kind", kind or ws.kind,
        "--output", str(ws.output), "--batch-size", str(BATCH_SIZE),
        "--device", "cpu",
    ])


def _recorded(ws):
    return json.loads(ws.output.read_text())["manifest"]["artifact_digests"]


def _expected(ws):
    """The record a run must write: the digest of each file it read, and `kg` as
    the manifest binds it."""
    manifest = json.loads((ws.data_dir / "split_manifest.json").read_text())
    expected = {
        "checkpoint": _sha(ws.checkpoint),
        "samples": _sha(ws.data_dir / f"{ws.split}_samples.json"),
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
    return not ws.output.parent.exists()


# ---------------------------------------------------------------------------
# One read each, and the record is those reads
# ---------------------------------------------------------------------------
def test_each_input_is_opened_once(cli, ws):
    with count_opens() as opens:
        assert _main(cli, ws) == 0

    read = ["kg.json", "split_manifest.json", *GRAPH_FILES.values(), f"{ws.split}_samples.json"]
    if ws.kind == "generated":
        read.append("train_samples.json")
    for name in read:
        assert opens.opens(ws.data_dir / name) == 1, name
    assert opens.opens(ws.checkpoint) == 1
    unread = {"train_samples.json", "val_samples.json", "test_samples.json"} - set(read)
    for name in unread:
        assert opens.opens(ws.data_dir / name) == 0, f"{name} is no input of this run"


def test_the_record_is_the_digests_of_the_bytes_read(cli, ws):
    """Exactly these roles: the five files read, `split_manifest` for a generated
    cohort only, and `kg` as the manifest binds it."""
    assert _main(cli, ws) == 0

    assert _recorded(ws) == _expected(ws)


def test_nothing_else_in_the_directory_is_opened_or_becomes_a_role(cli, ws):
    """A record of the directory is not a record of the run."""
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
    target = ws.data_dir / f"{ws.split}_samples.json"
    read = _sha(target)
    patients = [row["patient_id"] for row in json.loads(target.read_text())]
    record = replace_after_read(monkeypatch, after=target, target=target,
                                data=(other / f"{ws.split}_samples.json").read_bytes(), how=how)

    assert _main(cli, ws) == 0

    assert record.fired == 1
    assert _recorded(ws)["samples"] == read != _sha(target)
    ranks = json.loads(ws.output.with_name("measurement_ranks.json").read_text())
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
                                     (other / name).read_bytes(), how)


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


def test_a_republication_before_the_run_is_read_and_recorded(cli, ws, other):
    """Fired before either read, the run reads and records B."""
    target, manifest_path = ws.data_dir / "node_features.pt", ws.data_dir / "split_manifest.json"
    data = (other / "node_features.pt").read_bytes()
    replace_file(target, data, "rename")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["node_features"] = hashlib.sha256(data).hexdigest()
    replace_file(manifest_path, json.dumps(manifest).encode("utf-8"), "rename")

    assert _main(cli, ws) == 0

    assert _recorded(ws)["node_features"] == hashlib.sha256(data).hexdigest()
    assert _recorded(ws) == _expected(ws)


@pytest.mark.parametrize("name", ["test_samples.json", "checkpoint"])
def test_a_file_removed_after_its_read_is_still_recorded_never_as_none(
    cli, supplied, monkeypatch, name
):
    target = supplied.checkpoint if name == "checkpoint" else supplied.data_dir / name
    read = _sha(target)

    def remove(path):
        if Path(path).resolve() == target.resolve():
            target.unlink()

    hook_read_once(monkeypatch, on_return=remove)

    assert _main(cli, supplied) == 0

    assert not target.exists(), "the hook did not fire"
    recorded = _recorded(supplied)
    assert None not in recorded.values()
    assert recorded["checkpoint" if name == "checkpoint" else "samples"] == read


# ---------------------------------------------------------------------------
# Refused before anything is read, built or written
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", ["absolute", "reserved", "not_generated", "missing"])
def test_the_cohort_is_resolved_before_anything_is_read(cli, supplied, no_model, tmp_path, case):
    """An absolute split used to open and parse a file outside the workspace
    before the cohort was resolved; a split of the wrong kind used to be refused
    only after the model was built and encoded."""
    outside = tmp_path / "outside_samples.json"
    outside.write_text((supplied.data_dir / "test_samples.json").read_text())
    split, kind, message = {
        "absolute": (str(tmp_path / "outside"), "supplied", "must be an identifier"),
        "reserved": ("val", "supplied", "generator's own split names"),
        "not_generated": ("test", "generated", "not one of the generator's splits"),
        "missing": ("mygene2", "supplied", "does not exist"),
    }[case]

    with count_opens() as opens:
        with pytest.raises(ValueError, match=message):
            _main(cli, supplied, split=split, kind=kind)

    watched = (os.path.realpath(supplied.data_dir), os.path.realpath(outside),
               os.path.realpath(supplied.checkpoint))
    opened = [path for path, n in opens.counts.items() if n and path.startswith(watched)]
    assert opened == [], f"opened before the cohort was resolved: {opened}"
    assert _nothing_written(supplied)


def test_a_missing_sibling_split_is_refused_before_anything_is_read(cli, generated, no_model):
    (generated.data_dir / "train_samples.json").unlink()

    with count_opens() as opens:
        with pytest.raises(ValueError, match="train_samples.json does not exist"):
            _main(cli, generated)

    root = os.path.realpath(generated.data_dir)
    assert not [path for path, n in opens.counts.items() if n and path.startswith(root)]
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
        if split != generated.split:
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

    roots = (os.path.realpath(ws.data_dir), os.path.realpath(ws.checkpoint))
    assert not [path for path, n in opens.counts.items() if n and path.startswith(roots)]
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


def test_a_graph_file_from_another_workspace_is_refused_for_a_supplied_cohort(
    cli, supplied, other, no_model
):
    """The supplied-cohort case of the graph binding, on a valid file: junk bytes
    now fail in the reader's parse instead (next test)."""
    target = supplied.data_dir / "edge_indices.pt"
    target.write_bytes((other / "edge_indices.pt").read_bytes())

    with pytest.raises(ValueError, match="is not the edge_indices artifact") as refused:
        _main(cli, supplied)

    assert str(target) in str(refused.value)
    assert _nothing_written(supplied)


def test_a_tensor_file_that_does_not_parse_is_refused_naming_it(cli, supplied, no_model):
    target = supplied.data_dir / "edge_indices.pt"
    target.write_bytes(b"from another workspace")

    with pytest.raises(ValueError, match="could not be parsed") as refused:
        _main(cli, supplied)

    assert str(target) in str(refused.value)
    assert _nothing_written(supplied)


# ---------------------------------------------------------------------------
# Only from the readers
# ---------------------------------------------------------------------------
def test_no_path_form_is_reached_and_no_path_digest_is_recorded(cli, ws, monkeypatch):
    """S9's dynamic pin, brought forward for measurement. `poison_path_forms`
    patches the source modules, which cannot reach a name `measure_scorer` bound
    at import. The only such names left are the `file_sha256` re-export (S9's to
    remove), so those are made to fail here too, and the path forms are asserted
    not to be bound here at all, which would put them out of the poison's reach."""
    for name in ("artifact_digests", "verify_graph_artifacts", "verify_generated_cohorts",
                 "compute_input_digests"):
        assert not hasattr(cli, name), f"measure_scorer binds {name} at import"
    poison_path_forms(monkeypatch, true_digest_for=[ws.data_dir / "kg.json"])

    def reached(*args, **kwargs):
        raise AssertionError("a measurement reached a path hash")

    for name in ("file_sha256", "_file_sha256"):
        monkeypatch.setattr(cli, name, reached)

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
    training reads stay unread."""
    import src.kg.storage.file_storage as file_storage

    real = file_storage.read_samples
    seen = []

    def spy(data_dir, split, **kwargs):
        seen.append((split, kwargs.get("training_fields", False)))
        return real(data_dir, split, **kwargs)

    monkeypatch.setattr(file_storage, "read_samples", spy)

    assert _main(cli, ws) == 0

    assert seen and all(fields is False for _, fields in seen), seen


def test_the_read_and_the_record_hash_no_path(cli):
    """Static half of the pin: neither function names a path form, a path hash
    or `torch.load`."""
    gone = {"verify_graph_artifacts", "verify_generated_cohorts",
            "verify_graph_source", "compute_input_digests", "file_sha256", "_file_sha256",
            # Any other way of reading or hashing a file by path.
            "hashlib", "file_digest", "read_once", "read_bytes", "read_text", "open"}
    for function in (cli.read_measurement_inputs, cli.measurement_input_digests):
        tree = ast.parse(inspect.getsource(function).lstrip())
        body = tree.body[0]
        names = {node.id for node in ast.walk(body) if isinstance(node, ast.Name)}
        names |= {node.attr for node in ast.walk(body) if isinstance(node, ast.Attribute)}
        names |= {alias.name for node in ast.walk(body)
                  if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
        assert not names & gone, f"{function.__name__} names {sorted(names & gone)}"
        assert "torch.load" not in ast.unparse(body), function.__name__
