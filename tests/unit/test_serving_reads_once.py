"""
Serving reads each recorded input once, and checks its graph reads.
===================================================================
Contract M2.1, S6. `build_pipeline` reads `kg.json` with `read_json` and passes
the `GraphRead` whole; the pipeline reads the manifest and the graph tensors once
each, verifies the graph from those reads and keeps all four bound digests; only
then does it read the checkpoint, once, and build the model from that read. M2.1
checks nothing about the checkpoint; comparing it with the graph is M2.4's. The
acceptance shown here:

- a file replaced after its read, by atomic rename or in-place rewrite, is the
  one used and recorded; one replaced after the manifest read, or a graph whose
  manifest is republished after the read, is refused, and the refusal names it;
- each input is opened once, and with every path hash poisoned no poisoned
  digest reaches what the pipeline keeps;
- graph tensors and the checkpoint load with ``map_location="cpu"`` and
  ``weights_only=True`` (the checkpoint since contract decision 10, M3a-1); a
  checkpoint holding anything but tensors and plain data is refused;
- the `GraphRead` arrives whole, and the pipeline refuses a graph and an
  identity given apart;
- an external `kg.json` that matches the manifest decides startup; one that
  does not is refused; a refused reload keeps what is serving;
- the model is built from the final checkpoint read, never from a selection;
- a reload reports the architecture the loaded spec names, for a task or a
  path alike, wherever the file is filed (K8-4 review P3-1);
- a checkpoint whose record names its graph two ways is refused and what is
  serving stays (K8-4 review P2-2).

Every build that reaches a model serves the session's trained disease child
(`child`; `child_copy` where a test changes a file), K8-2a: a checkpoint without
a model spec is not what serving is shown with. The synthetic `workspace` stays
where the build is refused before a model is built.

Module: tests/unit/test_serving_reads_once.py
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")

from src.kg.graph import GraphRead, KnowledgeGraph  # noqa: E402
from tests.fixtures.opens import count_opens  # noqa: E402
from tests.fixtures.replacement import HOWS, replace_after_read  # noqa: E402
from tests.fixtures.synthetic_workspace import build_workspace  # noqa: E402

GRAPH_FILES = {"kg": "kg.json", "node_features": "node_features.pt",
               "edge_indices": "edge_indices.pt", "num_nodes": "num_nodes.json"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def api(monkeypatch):
    """App state as a fresh process has it, and no deployment in the environment."""
    import src.api.main as api_main

    for field in ("pipeline", "kg", "_current_data_dir", "_current_checkpoint_path"):
        monkeypatch.setattr(api_main.app_state, field, None, raising=False)
    monkeypatch.setattr(api_main.app_state, "is_ready", False, raising=False)
    for name in ("SHEPHERD_KG_PATH", "SHEPHERD_DATA_DIR", "SHEPHERD_CHECKPOINT_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SHEPHERD_DEVICE", "cpu")
    return api_main


@pytest.fixture
def workspace(tmp_path):
    return build_workspace(tmp_path / "a")


@pytest.fixture
def child(trained_disease_child):
    """The session's trained disease child, ``(data_dir, checkpoint)`` as
    `workspace` gives them. Read-only: a test that changes a file takes
    `child_copy`."""
    return trained_disease_child["data_dir"], trained_disease_child["disease"]


@pytest.fixture
def child_copy(trained_disease_child, tmp_path):
    """`child` copied into this test's directory (``a/data``, ``a/checkpoint.pt``),
    for a test that replaces, removes or adds a file."""
    data_dir, checkpoint = tmp_path / "a" / "data", tmp_path / "a" / "checkpoint.pt"
    shutil.copytree(trained_disease_child["data_dir"], data_dir)
    shutil.copyfile(trained_disease_child["disease"], checkpoint)
    return data_dir, checkpoint


def _other_checkpoint(checkpoint, path):
    """The same architecture with shifted weights and another epoch."""
    contents = torch.load(checkpoint, map_location="cpu", weights_only=False)
    contents["state_dict"] = {
        name: value + 1.0 if value.is_floating_point() else value
        for name, value in contents["state_dict"].items()
    }
    contents["epoch"] = 9
    torch.save(contents, path)
    return path


def _build(api, data_dir, checkpoint, kg_path=None):
    return api.build_pipeline(
        kg_path=str(kg_path or data_dir / "kg.json"),
        checkpoint_path=str(checkpoint),
        data_dir=str(data_dir),
    )


# ---------------------------------------------------------------------------
# Replacements: used and recorded, or refused
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("how", HOWS)
def test_kg_json_replaced_after_the_api_read_serves_and_records_what_was_read(
    api, child_copy, tmp_path, monkeypatch, how
):
    from src.core.types import DataSource, NodeType
    from src.kg.graph import Node, NodeID

    data_dir, checkpoint = child_copy
    kg_file = data_dir / "kg.json"
    read = _sha(kg_file)
    nodes = KnowledgeGraph.read_json(kg_file).kg.total_nodes
    other = KnowledgeGraph()
    other.add_node(Node(id=NodeID(source=DataSource.MONDO, local_id="MONDO:9"),
                        node_type=NodeType.DISEASE, name="other"))
    other.save_json(str(tmp_path / "b.json"))
    record = replace_after_read(monkeypatch, after=kg_file, target=kg_file,
                                data=(tmp_path / "b.json").read_bytes(), how=how)

    pipeline = _build(api, data_dir, checkpoint).pipeline

    assert record.fired == 1
    assert pipeline.kg.total_nodes == nodes != other.total_nodes, (
        "the graph served is the one read, not B")
    assert pipeline._graph_identity.sha256 == read
    assert pipeline._graph_kg_digest == read
    assert pipeline._bound_digests["kg"] == read


@pytest.mark.parametrize("how", HOWS)
def test_kg_json_republished_after_its_read_is_refused(api, workspace, monkeypatch, how):
    """The manifest now binds B and the graph in hand is A: not the same graph."""
    data_dir, checkpoint = workspace
    kg_file = data_dir / "kg.json"
    record = replace_after_read(monkeypatch, after=kg_file, target=kg_file,
                                data=b'{"nodes": [], "edges": [], "other": 1}', how=how,
                                republish="kg")

    with pytest.raises(ValueError, match="is not the graph") as refused:
        _build(api, data_dir, checkpoint)

    assert record.fired == 1
    assert str(kg_file) in str(refused.value)


@pytest.mark.parametrize("how", HOWS)
def test_a_tensor_replaced_after_its_read_serves_and_records_what_was_read(
    api, child_copy, tmp_path, monkeypatch, how
):
    data_dir, checkpoint = child_copy
    other, _ = build_workspace(tmp_path / "b", n_diseases=5)
    target = data_dir / "node_features.pt"
    read = _sha(target)
    tensors = torch.load(target, weights_only=True)
    record = replace_after_read(monkeypatch, after=target, target=target,
                                data=(other / "node_features.pt").read_bytes(), how=how)

    pipeline = _build(api, data_dir, checkpoint).pipeline

    assert record.fired == 1
    assert pipeline._bound_digests["node_features"] == read
    for node_type, features in tensors.items():
        assert torch.equal(pipeline._graph_data["x_dict"][node_type], features)


@pytest.mark.parametrize("how", HOWS)
def test_a_tensor_replaced_after_the_manifest_read_is_refused_naming_it(
    api, workspace, tmp_path, monkeypatch, how
):
    data_dir, checkpoint = workspace
    other, _ = build_workspace(tmp_path / "b", n_diseases=5)
    target = data_dir / "edge_indices.pt"
    record = replace_after_read(monkeypatch, after=data_dir / "split_manifest.json",
                                target=target, data=(other / "edge_indices.pt").read_bytes(),
                                how=how)

    with pytest.raises(ValueError, match="is not the edge_indices artifact") as refused:
        _build(api, data_dir, checkpoint)

    assert record.fired == 1
    assert str(target) in str(refused.value)


@pytest.mark.parametrize("how", HOWS)
def test_the_checkpoint_replaced_after_its_read_builds_the_model_read(
    api, child_copy, tmp_path, monkeypatch, how
):
    data_dir, checkpoint = child_copy
    read = _sha(checkpoint)
    contents = torch.load(checkpoint, weights_only=False)
    state = contents["state_dict"]
    _other_checkpoint(checkpoint, tmp_path / "b.pt")
    record = replace_after_read(monkeypatch, after=checkpoint, target=checkpoint,
                                data=(tmp_path / "b.pt").read_bytes(), how=how)

    bundle = _build(api, data_dir, checkpoint)

    assert record.fired == 1
    assert bundle.pipeline._checkpoint_identity.sha256 == read
    served = bundle.pipeline.model.state_dict()
    for name, value in state.items():
        assert torch.equal(served[name].cpu(), value)
    assert bundle.config["checkpoint_meta"]["epoch"] == contents["epoch"] != 9, (
        "the metadata is A's too")


def test_a_tensor_file_that_does_not_parse_is_refused_naming_it(api, workspace):
    """Serving parses before it compares; the reader names the file it could not
    parse, where the parser's own message names nothing."""
    data_dir, checkpoint = workspace
    target = data_dir / "node_features.pt"
    target.write_bytes(b"same shape, another workspace")

    with pytest.raises(ValueError, match="could not be parsed") as refused:
        _build(api, data_dir, checkpoint)

    assert str(target) in str(refused.value)


# ---------------------------------------------------------------------------
# One read each; nothing hashed by path; load options
# ---------------------------------------------------------------------------
def test_each_input_is_opened_once(api, child):
    data_dir, checkpoint = child

    with count_opens() as opens:
        _build(api, data_dir, checkpoint)

    for name in (*GRAPH_FILES.values(), "split_manifest.json"):
        assert opens.opens(data_dir / name) == 1, name
    assert opens.opens(checkpoint) == 1


def test_no_path_digest_is_kept(api, child, monkeypatch):
    """S9's dynamic pin for serving: every path hash is poisoned, with no
    exception (serving parses `kg.json`, so nothing here is identified by path),
    and the build still succeeds with the true digests of the bytes it read."""
    from tests.fixtures.path_forms import POISON, poison_path_forms

    data_dir, checkpoint = child
    poison_path_forms(monkeypatch)

    pipeline = _build(api, data_dir, checkpoint).pipeline

    kept = [*pipeline._bound_digests.values(), pipeline._graph_kg_digest,
            pipeline._checkpoint_identity.sha256]
    assert POISON not in kept
    for role, name in GRAPH_FILES.items():
        assert pipeline._bound_digests[role] == _sha(data_dir / name), role
    assert pipeline._graph_kg_digest == _sha(data_dir / "kg.json")
    assert pipeline._checkpoint_identity.sha256 == _sha(checkpoint)


def test_graph_tensors_and_the_checkpoint_load_on_cpu_and_safely(
    api, child, monkeypatch
):
    import sys

    data_dir, checkpoint = child
    real_load = torch.load
    calls = []

    def recording_load(*args, **kwargs):
        caller = sys._getframe(1).f_globals.get("__name__")
        calls.append((caller, kwargs.get("map_location"), kwargs.get("weights_only")))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)

    _build(api, data_dir, checkpoint)

    graph = [call for call in calls if call[0] == "src.kg.storage.file_storage"]
    assert graph == [("src.kg.storage.file_storage", "cpu", True)] * 2
    loaded = [call for call in calls if call[0] == "src.utils.checkpoint_io"]
    assert loaded == [("src.utils.checkpoint_io", "cpu", True)]
    assert all(call[0] in {"src.kg.storage.file_storage", "src.utils.checkpoint_io"}
               for call in calls), f"a load outside the readers: {calls}"


# ---------------------------------------------------------------------------
# The graph and its identity travel together
# ---------------------------------------------------------------------------
def test_the_graph_read_arrives_whole(api, child, monkeypatch):
    import src.inference.pipeline as pipeline_module

    data_dir, checkpoint = child
    real_read = KnowledgeGraph.read_json.__func__
    returned = []

    def read_json(cls, path):
        result = real_read(cls, path)
        returned.append(result)
        return result

    monkeypatch.setattr(KnowledgeGraph, "read_json", classmethod(read_json))
    seen = {}
    real_factory = pipeline_module.create_diagnosis_pipeline

    def factory(**kwargs):
        seen.update(kwargs)
        return real_factory(**kwargs)

    monkeypatch.setattr(pipeline_module, "create_diagnosis_pipeline", factory)

    bundle = _build(api, data_dir, checkpoint)

    assert len(returned) == 1
    assert seen["graph"] is returned[0] and isinstance(seen["graph"], GraphRead)
    assert "kg" not in seen and "kg_path" not in seen
    assert bundle.pipeline.kg is returned[0].kg and bundle.kg is returned[0].kg


def test_a_graph_and_an_identity_cannot_be_given_apart(workspace):
    from src.inference.pipeline import DiagnosisPipeline, create_diagnosis_pipeline

    data_dir, checkpoint = workspace
    graph = KnowledgeGraph.read_json(data_dir / "kg.json")

    with pytest.raises(TypeError, match="GraphRead"):
        DiagnosisPipeline(graph=(graph.kg, graph.identity.sha256),
                          checkpoint_path=str(checkpoint), data_dir=str(data_dir))
    for construct in (DiagnosisPipeline.__init__, create_diagnosis_pipeline):
        parameters = inspect.signature(construct).parameters
        assert not {"kg", "kg_path"} & set(parameters), (
            "a graph or a path option kept beside the read would be a second way in"
        )
        # K7: the graph read, the checkpoint and the workspace are all required.
        for name in ("graph", "checkpoint_path", "data_dir"):
            assert parameters[name].default is inspect.Parameter.empty, name


def test_a_workspace_build_without_the_graph_read_is_refused(workspace, monkeypatch):
    """A graph built in memory claims no persisted source, so it cannot be
    paired with a workspace's tensors: refused before any workspace file is
    opened."""
    from src.inference.pipeline import DiagnosisPipeline

    data_dir, checkpoint = workspace

    # `device="cpu"`: with none named a checkpoint asks for CUDA and, on a host
    # without it, is refused first (contract M3c, §5.2). This test is about the
    # graph= refusal, so it names the development device to reach it.
    with count_opens() as opens:
        with pytest.raises(TypeError, match="GraphRead"):
            DiagnosisPipeline(graph=KnowledgeGraph(), data_dir=str(data_dir),
                              checkpoint_path=str(checkpoint), device="cpu")

    assert not [path for path in data_dir.iterdir() if opens.opens(path)]


# ---------------------------------------------------------------------------
# An external kg.json, and what a refused reload keeps
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("beside_the_tensors", ["absent", "different"])
def test_an_external_kg_json_decides_startup_and_a_refused_reload_keeps_serving(
    api, child_copy, tmp_path, monkeypatch, beside_the_tensors
):
    from fastapi.testclient import TestClient

    data_dir, checkpoint = child_copy
    external = tmp_path / "mounted" / "kg.json"
    external.parent.mkdir()
    shutil.copyfile(data_dir / "kg.json", external)
    if beside_the_tensors == "absent":
        (data_dir / "kg.json").unlink()
    else:
        (data_dir / "kg.json").write_text(json.dumps({"nodes": [], "edges": [], "other": True}))
    monkeypatch.setenv("SHEPHERD_KG_PATH", str(external))
    monkeypatch.setenv("SHEPHERD_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SHEPHERD_CHECKPOINT_PATH", str(checkpoint))
    # The reload is an operator control (D8): mode M1, loopback requests.
    from tests.fixtures.operator_access import OPERATOR_BASE_URL, operator_mode

    operator_mode(monkeypatch)

    with TestClient(api.app, base_url=OPERATOR_BASE_URL) as client:
        assert client.get("/ready").status_code == 200
        served = api.app_state.pipeline
        assert served._graph_identity.path == external, "startup used the external file"

        # A reload still derives its graph from the data directory.
        reply = client.post(
            "/api/v1/pipeline/reload",
            json={"data_dir": str(data_dir), "checkpoint_path": str(checkpoint)},
        ).json()

        assert reply["success"] is False
        expected = ("Missing required files" if beside_the_tensors == "absent"
                    else "is not the graph")
        assert expected in reply["message"]
        assert api.app_state.pipeline is served, "a refused reload keeps what is serving"
        assert client.get("/ready").status_code == 200


def test_an_external_kg_json_that_does_not_match_is_refused(api, workspace, tmp_path):
    data_dir, checkpoint = workspace
    external = tmp_path / "kg.json"
    external.write_text(json.dumps({"nodes": [], "edges": [], "other": True}))

    with pytest.raises(ValueError, match="is not the graph"):
        _build(api, data_dir, checkpoint, kg_path=external)


# ---------------------------------------------------------------------------
# The model comes from the final read (decision 4)
# ---------------------------------------------------------------------------
def test_the_model_is_built_from_the_final_read_not_from_the_selection(
    api, child_copy, tmp_path, monkeypatch
):
    """The reload route selects a checkpoint by reading candidates (a task's
    selection: the disease child, filed under its encoder kind). The file is
    replaced after that selection; what is served, and what is kept as its
    identity, is what the build read."""
    data_dir, checkpoint = child_copy
    architecture = data_dir / "runs" / "gatv2" / "20261015-000000_run_aaaaaa" / "checkpoints"
    architecture.mkdir(parents=True)
    shutil.copyfile(checkpoint, architecture / "model.pt")
    _other_checkpoint(checkpoint, tmp_path / "b.pt")
    replacement = _sha(tmp_path / "b.pt")
    real_build = api.build_pipeline

    def replace_then_build(**kwargs):
        shutil.copyfile(tmp_path / "b.pt", kwargs["checkpoint_path"])
        return real_build(**kwargs)

    monkeypatch.setattr(api, "build_pipeline", replace_then_build)
    from src.api.routes.pipeline import PipelineReloadRequest, reload_pipeline

    result = asyncio.run(reload_pipeline(PipelineReloadRequest(data_dir=str(data_dir),
                                                               task="disease")))

    assert result.success, result.message
    assert api.app_state.pipeline._checkpoint_identity.sha256 == replacement
    assert result.status.checkpoint_meta["epoch"] == 9
    # Contract M3c (BACKLOG 19.1, 19.2): the served identity names the bytes the
    # build read, and the reason says the choice was made on earlier bytes.
    assert result.status.served_identity.checkpoint.sha256 == replacement
    assert "replaced after it was scored" in result.selection_reason
    assert f"loaded sha256 {replacement}" in result.selection_reason


def test_an_auto_selected_file_swapped_for_a_link_out_of_the_workspace_is_never_read_or_published(
    api, child_copy, tmp_path, monkeypatch
):
    """The reviewer's K8-4 P2-1 (probe.py, "swap after selection"): a task's
    selection chooses the workspace's disease child, and between the selection
    and the build's checkpoint read the chosen ``last.pt`` is replaced by a link
    to another child outside the workspace. Before the fix the build read the
    outside file and the reload served its SHA-256. Now the build's one read is
    bound to the workspace by the file it opened: the reload is refused, not one
    read of the outside file returns, and the model published before (by the
    same selection, the positive control) goes on serving."""
    from src.api.routes.pipeline import PipelineReloadRequest, run_reload
    from tests.fixtures.replacement import hook_read_once

    data_dir, checkpoint = child_copy
    run = data_dir / "runs" / "gatv2" / "20261015-000000_run_aaaaaa"
    selected = run / "checkpoints" / "last.pt"
    selected.parent.mkdir(parents=True)
    shutil.copyfile(checkpoint, selected)
    (tmp_path / "outside").mkdir()
    outside = _other_checkpoint(checkpoint, tmp_path / "outside" / "b.pt")
    request = PipelineReloadRequest(data_dir=str(data_dir), task="disease")

    first = run_reload(request)
    assert first.success, first.message
    prior = api.app_state.pipeline
    assert prior.served_checkpoint_read().sha256 == _sha(checkpoint)

    real_build, swaps = api.build_pipeline, []

    def swap_then_build(**kwargs):
        assert kwargs["checkpoint_path"] == str(selected)
        swaps.append(kwargs["checkpoint_path"])
        selected.unlink()
        selected.symlink_to(outside)
        return real_build(**kwargs)

    monkeypatch.setattr(api, "build_pipeline", swap_then_build)
    returned = []
    hook_read_once(monkeypatch, on_return=lambda path: returned.append(Path(path).resolve()))

    result = run_reload(request)

    assert swaps == [str(selected)]
    assert result.success is False
    assert f"{selected} was opened outside the workspace" in result.message
    assert "nothing was read from it or published" in result.message
    assert result.message.endswith("The previously loaded pipeline is still being served.")
    assert api.app_state.pipeline is prior
    assert prior.served_checkpoint_read().sha256 == _sha(checkpoint)
    assert outside.resolve() not in returned, "the outside file was read"


def test_a_reload_reports_the_architecture_its_loaded_spec_names_wherever_the_file_is_filed(
    api, child_copy
):
    """The reviewer's K8-4 P3-1 (probe.py, ``architecture``): the trained
    disease child, a GATv2 model, filed under ``runs/hgt/``. Before the fix an
    automatic reload reported the directory's name (``hgt``) as the
    architecture, and its reason read "latest run '…' (hgt)", while a reload by
    the explicit path reported none. The response's ``architecture`` is the
    loaded model spec's encoder kind, the same for both; the directory only
    locates the file, so the reason names the run by its location, and the
    model is served wherever it is filed."""
    from src.api.routes.pipeline import PipelineReloadRequest, run_reload

    data_dir, checkpoint = child_copy
    run = "20261015-000000_review_aaaaaa"
    filed = data_dir / "runs" / "hgt" / run / "checkpoints" / "last.pt"
    filed.parent.mkdir(parents=True)
    shutil.copyfile(checkpoint, filed)

    auto = run_reload(PipelineReloadRequest(data_dir=str(data_dir), task="disease"))

    assert auto.success, auto.message
    assert auto.checkpoint_path == str(filed)
    assert auto.architecture == "gatv2" == api.app_state.pipeline.model.spec.encoder.kind
    assert f"task disease: auto: latest run 'hgt/{run}'" in auto.selection_reason
    assert "(hgt)" not in auto.selection_reason and "(hgt)" not in auto.message

    explicit = run_reload(PipelineReloadRequest(data_dir=str(data_dir),
                                                checkpoint_path=str(filed)))

    assert explicit.success, explicit.message
    assert explicit.architecture == "gatv2" == api.app_state.pipeline.model.spec.encoder.kind
    assert explicit.selection_reason == "explicit checkpoint_path"


def test_a_child_whose_data_roles_kg_contradicts_its_training_record_is_refused_and_nothing_is_published(
    api, child_copy, tmp_path
):
    """The reviewer's K8-4 P2-2, inference counterexample (probe.py,
    ``disease_conflicting_kg``): the trained child with only ``data_roles.kg``
    changed, loaded by an explicit path with its correct SHA-256. Its
    ``training_input_digests.kg`` matches the workspace, so the graph
    comparison passes it; before the fix the reload published it. Now the
    shared record reader refuses a record that names the graph two ways, and
    the model published before goes on serving."""
    from src.api.routes.pipeline import PipelineReloadRequest, run_reload

    data_dir, checkpoint = child_copy
    first = run_reload(PipelineReloadRequest(data_dir=str(data_dir),
                                             checkpoint_path=str(checkpoint)))
    assert first.success, first.message
    prior = api.app_state.pipeline
    contents = torch.load(checkpoint, map_location="cpu", weights_only=True)
    recorded = contents["training_input_digests"]["kg"]
    assert contents["data_roles"]["kg"] == recorded
    contents["data_roles"]["kg"] = "0" * 64
    conflicting = tmp_path / "conflicting.pt"
    torch.save(contents, conflicting)

    result = run_reload(PipelineReloadRequest(
        data_dir=str(data_dir), checkpoint_path=str(conflicting),
        expected_checkpoint_sha256=_sha(conflicting)))

    assert result.success is False
    assert "the record contradicts itself" in result.message
    assert (f"data_roles.kg {'0' * 12}... differs from training_input_digests.kg "
            f"{recorded[:12]}...") in result.message
    assert result.message.endswith("The previously loaded pipeline is still being served.")
    assert api.app_state.pipeline is prior
    assert prior.served_checkpoint_read().sha256 == _sha(checkpoint)


def test_a_checkpoint_holding_arbitrary_objects_is_refused(api, workspace, tmp_path):
    """`weights_only=True` (contract decision 10): a pickled object other than
    tensors and plain data is not loaded, so a file a request names cannot run
    code in the service. Refused like any unreadable checkpoint, with no second
    attempt under the unsafe loader."""
    from pathlib import Path

    data_dir, checkpoint = workspace
    contents = torch.load(checkpoint, map_location="cpu", weights_only=False)
    contents["not_plain"] = Path("x")
    unsafe = tmp_path / "unsafe.pt"
    torch.save(contents, unsafe)

    with pytest.raises(Exception, match="could not be read"):
        _build(api, data_dir, unsafe)
