"""Automatic selection reads each checkpoint's record first, and the lists show
it (provenance §9.3, hazard 3; README §1).

A reload that names a task serves the latest run holding a model finetuned for
it, judged by each file's model record, never by its directory: a newer base,
another task's child and a checkpoint without a model spec are passed over;
with none it is refused naming the task, and an explicit checkpoint of another
task, or one without a model spec, is refused with nothing published. A reload
that names neither a task nor a checkpoint is refused before anything is read
(SG2), and a field the request does not declare (the retired ``conv_type``, a
misspelt ``expected_checkpoint_sha``) is refused with HTTP 422 before any
candidate is selected (Q17 (b)). The run listing carries each
file's record from its one read; the lists show it through the one reader of
records (`listed_record`), and a bundle run resolves by the key it is listed
under (``gatv2``). Automatic selection never reads test results (D4): a ledger
beside a task child's checkpoints that favours another file does not change
what the reload or `select_auto_run_checkpoint` choose (the import-linter
contract `selection-ignores-test-results` keeps the modules apart).

Module: tests/unit/test_task_first_selection.py
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from src.api.routes.pipeline import (  # noqa: E402
    REQUIRED_DATA_FILES,
    PipelineReloadRequest,
    run_reload,
)
from src.core.types import (  # noqa: E402
    SCORING_MODE_TASKS,
    SERVED_SCORING_MODES,
    TASK_SCORING_MODES,
)
from src.models.bundle import build_bundle, finetune_spec_from_parent, reference_spec  # noqa: E402
from src.models.bundle.record import (  # noqa: E402
    HISTORY_ONLY,
    LISTED_WITHOUT_SPEC,
    LISTED_RECORD_FIELDS,
    MODEL_RECORD_VERSION,
    ModelRecord,
    listed_record,
)
from src.models.bundle.spec import InputSpec, ModelSpec  # noqa: E402
from src.utils.checkpoint_io import write_checkpoint  # noqa: E402
from src.utils.fingerprint import file_sha256  # noqa: E402

VIEW = {"digest": "e" * 64, "num_nodes": 4, "node_types": ["gene", "phenotype"],
        "relations": ["gene|has|phenotype", "phenotype|rev_has|gene"]}
BASE_RUN = "20261001-000000_base_aaaaaa"


def _base_spec():
    spec = reference_spec("base")
    return dataclasses.replace(
        spec, input=InputSpec(dim=4),
        encoder=dataclasses.replace(spec.encoder, layers=1, hidden=2, output=4, heads=2)
    ).validate()


def _record_payload(task=None, *, logs=None, parent=None, family_id=BASE_RUN):
    """A bundle checkpoint's payload: a base (`task` None) or a child of
    `parent` (`{sha256, run}`; by default a file of `BASE_RUN`), its record as
    the training payload writes it."""
    base = _base_spec()
    spec = base if task is None else finetune_spec_from_parent(base, task)
    bundle = build_bundle(spec, VIEW)
    lineage = (dict(operation="new", family_id=family_id) if task is None else
               dict(operation="finetune", family_id=family_id,
                    parent=parent or {"sha256": "a" * 64, "run": BASE_RUN}))
    record = ModelRecord(model_spec=spec, graph_view=bundle.view,
                         data_roles={"kg": "k" * 64, "split": "s" * 64},
                         graph_exposure={"base": {"message_passing": "train_edges"}},
                         loss_identity={"terms": ["lp"], "lp_loss_weight": 1.0}, **lineage)
    return {"record_version": MODEL_RECORD_VERSION, "state_dict": bundle.state_dict(),
            "logs": dict(logs or {}), **record.to_payload()}


#: A ShepherdGNN checkpoint: no model spec.
GNN_PAYLOAD = {"checkpoint_kind": "training_state", "epoch": 3, "logs": {"val_mrr": 0.9}}


def _put(data_dir, key, run, payload, mtime, name="last.pt"):
    path = data_dir / "runs" / key / run / "checkpoints" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    write_checkpoint(payload, path)
    os.utime(path, (mtime, mtime))
    return path


# ---------------------------------------------------------------------------
# The reload: a fake build, so what is tested is what is chosen and published
# ---------------------------------------------------------------------------
def _identity(mode, sha256):
    from src.core.types import (
        ServedCheckpoint,
        ServedDatasetRecord,
        ServedGraph,
        ServedIdentity,
        ServedSettings,
        ServedShortestPaths,
    )

    return ServedIdentity(
        scoring_mode=mode,
        checkpoint=ServedCheckpoint(state="verified", sha256=sha256),
        graph=ServedGraph(state="verified", kg_sha256="1" * 64, split_manifest_sha256="2" * 64),
        shortest_paths=ServedShortestPaths(state="computed" if mode == "gene_task" else "absent"),
        dataset_record=ServedDatasetRecord(state="verified", origin="synthetic", sha256="3" * 64),
        settings=ServedSettings(
            device="cpu", device_requested="cpu", eta=1.0, max_path_length=4,
            path_length_penalty=0.9,
            max_paths_per_candidate=10, use_direct_path_optimization=True,
            phenotype_normalisation_version=1),
    ).to_dict()


@pytest.fixture
def service(tmp_path, monkeypatch):
    """A workspace the reload accepts as far as the build, and a build that
    serves what the chosen file's own record says (a base is not served, and a
    file without a model spec is refused by its model record, as the real
    build refuses it), with publication recorded instead of performed."""
    import src.api.main as api

    data_dir = tmp_path / "ws"
    data_dir.mkdir()
    for name in REQUIRED_DATA_FILES:
        (data_dir / name).write_text("placeholder", encoding="utf-8")
    built, published = [], []

    def build(**kwargs):
        path = kwargs["checkpoint_path"]
        built.append(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if "model_spec" not in payload:
            from src.models.bundle.record import read_model_record

            read_model_record(payload)       # refuses it: HISTORY_ONLY (SG1)
        if payload.get("task") is None:
            raise RuntimeError("a base model is not served")
        else:
            mode = TASK_SCORING_MODES[payload["task"]]
        config = {"served_identity": _identity(mode, file_sha256(path)), "scoring_mode": mode,
                  "gnn_ready": True, "has_model": True, "checkpoint_meta": dict(payload["logs"])}
        # The built model carries the chosen file's own spec, as the real
        # build's `ModelBundle` does: the reload reports its encoder kind.
        model = SimpleNamespace(spec=ModelSpec.from_dict(payload["model_spec"]))
        return SimpleNamespace(config=config, kg=None, pipeline=SimpleNamespace(model=model),
                               data_dir=kwargs["data_dir"])

    monkeypatch.setenv("SHEPHERD_DEVICE", "cpu")
    monkeypatch.setattr(api, "build_pipeline", build)
    monkeypatch.setattr(api, "publish_pipeline", published.append)
    monkeypatch.setattr(api.app_state, "pipeline", None, raising=False)
    monkeypatch.setattr(api.app_state, "real_pipeline_requested", False, raising=False)
    return SimpleNamespace(data_dir=data_dir, built=built, published=published)


def _reload(service, **kwargs):
    return run_reload(PipelineReloadRequest(data_dir=str(service.data_dir), **kwargs))


@pytest.fixture
def family(service):
    """A base and its three children in the bundle layout, the base newest and
    the disease child oldest, beside a ShepherdGNN run newer than all of them."""
    d = service.data_dir
    return {
        "gnn": _put(d, "hgt", "20261009-000000_gnn_eeeeee", GNN_PAYLOAD, 900),
        "base": _put(d, "gatv2", BASE_RUN, _record_payload(logs={"val_lp_accuracy": 0.7}), 800),
        "gene": _put(d, "gatv2", "20261003-000000_gene_cccccc",
                     _record_payload("gene", logs={"val_pooled_mrr": 0.2}), 700),
        "disease": _put(d, "gatv2", "20261002-000000_disease_bbbbbb",
                        _record_payload("disease", logs={"val_pooled_mrr": 0.3}), 600),
    }


def test_a_task_is_served_by_its_own_child_whatever_is_newer(service, family):
    """Catches selection by recency or directory: the base, the gene child and
    the ShepherdGNN run are all newer than the disease child."""
    disease = _reload(service, task="disease")
    assert disease.success, disease.message
    assert disease.checkpoint_path == str(family["disease"])
    assert disease.selection_reason.startswith(
        "task disease: auto: latest run 'gatv2/20261002-000000_disease_bbbbbb'")
    assert "best val_pooled_mrr=0.3000" in disease.selection_reason
    assert disease.architecture == "gatv2"
    gene = _reload(service, task="gene")
    assert gene.success and gene.checkpoint_path == str(family["gene"])
    assert len(service.published) == 2
    assert service.built == [str(family["disease"]), str(family["gene"])]


def test_a_task_no_run_holds_is_refused_naming_it(service, family):
    result = _reload(service, task="patients_like_me")
    assert not result.success and service.built == [] and service.published == []
    assert "No run holds a model finetuned for the patients_like_me task" in result.message
    assert result.selection_reason == "task patients_like_me: no run has a checkpoint"
    assert result.files_found["checkpoint"] is False


def test_a_file_the_record_does_not_place_is_never_a_task_candidate(service, family):
    """A newer unreadable file in the child's run is not a candidate, nor is
    a run holding nothing else."""
    run = family["disease"].parent
    (run / "torn.pt").write_bytes(b"not a checkpoint")
    other = service.data_dir / "runs" / "gatv2" / "20261008-000000_torn_ffffff" / "checkpoints"
    other.mkdir(parents=True)
    (other / "last.pt").write_bytes(b"not a checkpoint")
    result = _reload(service, task="disease")
    assert result.success and result.checkpoint_path == str(family["disease"])


def test_an_explicit_checkpoint_of_another_task_publishes_nothing(service, family):
    refused = _reload(service, task="disease", checkpoint_path=str(family["gene"]))
    assert not refused.success and service.published == []
    assert (f"{family['gene']} serves the gene task, not the disease task asked for. "
            "Nothing was published.") in refused.message
    gnn = _reload(service, task="disease", checkpoint_path=str(family["gnn"]))
    assert not gnn.success and service.published == []
    assert HISTORY_ONLY in gnn.message
    served = _reload(service, task="gene", checkpoint_path=str(family["gene"]))
    assert served.success and len(service.published) == 1


def _observed_reads(monkeypatch):
    """The paths the reload's scan reads (`read_checkpoint`, imported where it
    runs)."""
    import src.utils.checkpoint_io as checkpoint_io

    original, reads = checkpoint_io.read_checkpoint, []

    def observing(path, *args, **kwargs):
        reads.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(checkpoint_io, "read_checkpoint", observing)
    return reads


def test_a_reload_naming_neither_a_task_nor_a_path_is_refused_and_reads_nothing(
        service, family, monkeypatch):
    """SG2: automatic selection serves the latest model finetuned for the task
    named; with neither a task nor a path nothing is chosen. Refused after the
    file check (the file report stays) and before the scan: though the
    workspace holds a disease child and a newer file without a model spec, no
    candidate is read, built or published."""
    reads = _observed_reads(monkeypatch)

    result = _reload(service)

    assert not result.success
    assert result.message.startswith(
        f"Name a task ({', '.join(TASK_SCORING_MODES)}) or a checkpoint_path: automatic "
        "selection serves the latest model finetuned for the task named, and nothing is "
        "chosen without one.")
    assert result.selection_reason == "neither task nor checkpoint_path"
    assert all(result.files_found[name] for name in REQUIRED_DATA_FILES)
    assert reads == [] and service.built == [] and service.published == []


@pytest.mark.parametrize("linked", ["run", "layout"])
def test_a_task_child_linked_outside_the_workspace_is_never_chosen_or_read(
        service, family, monkeypatch, tmp_path, linked):
    """The reviewer's K8-1 recheck (directory_sim): the listing refuses a
    checkpoint that resolves outside the workspace, and before the fix the
    task's automatic selection read, chose and published it, newer than the
    workspace's own child. Now selection keeps the listing's boundary: a
    disease child in a run directory (or a layout directory) linked out of
    the workspace is never read or chosen, the workspace's own disease child
    is served, and the reason names the file in the listing's words; with
    nothing left inside, the hint says the runs resolve outside the workspace
    instead of suggesting a finetune. An explicit `checkpoint_path` stays as it
    is: operator-only (D8), read by its bytes."""
    elsewhere = tmp_path / "elsewhere"
    outside = _put(elsewhere, "gatv2", "20261010-000000_far_ffffff",
                   _record_payload("disease", logs={"val_pooled_mrr": 0.9}), 1000)
    run = outside.parent.parent
    if linked == "run":
        (service.data_dir / "runs" / "gatv2" / run.name).symlink_to(
            run, target_is_directory=True)
    else:
        # Every bundle run of the workspace moves out behind its layout key.
        workspace_gatv2 = service.data_dir / "runs" / "gatv2"
        for child in workspace_gatv2.iterdir():
            child.rename(elsewhere / "runs" / "gatv2" / child.name)
        workspace_gatv2.rmdir()
        workspace_gatv2.symlink_to(elsewhere / "runs" / "gatv2", target_is_directory=True)
    reads = _observed_reads(monkeypatch)

    result = _reload(service, task="disease")

    home = service.data_dir.resolve()
    assert reads and all(Path(read).resolve().is_relative_to(home) for read in reads)
    reason = f"{run.name}/last.pt resolves outside the workspace, so it is not read"
    if linked == "run":
        assert result.success and result.checkpoint_path == str(family["disease"])
        assert service.built == [str(family["disease"])]
    else:
        # Nothing of the workspace's own is left under the linked key. The
        # hint names the boundary first, then what the task lacks inside the
        # workspace; it never says "Finetune one": behind a linked layout
        # directory training refuses to write too (`run_dir_outside`), and
        # where a run can be trained depends on where the link is.
        assert not result.success and service.built == [] and service.published == []
        assert ("checkpoint file(s) of the workspace's runs resolve outside the workspace, so "
                "automatic selection does not read them; keep the runs inside the workspace"
                in result.message)
        assert "or pass an explicit checkpoint_path" in result.message
        assert ("No run inside the workspace holds a model finetuned for the disease task."
                in result.message)
        assert "Finetune one" not in result.message
    assert reason in result.selection_reason
    # The explicit path is unchanged: named, it is read by its bytes.
    explicit = _reload(service, task="disease", checkpoint_path=str(outside))
    assert explicit.success and explicit.selection_reason == "explicit checkpoint_path"


@pytest.mark.parametrize("case", ["the only candidate", "another candidate in the run"])
def test_a_candidate_swapped_out_of_the_workspace_after_the_boundary_check_is_not_read_or_chosen(
        service, family, monkeypatch, tmp_path, case):
    """The reviewer's K8-4 P2-1, at the scan: the disease child's `last.pt` is
    replaced by a link to a disease child outside the workspace right after the
    boundary check judged its name (`checkpoint_not_read`) and before the scan
    reads it. Before the fix the scan read the outside file, accepted it, ranked
    it first (0.9) and the build built it. Now the scan's read is bound to the
    file it opens: the outside file is never read, chosen, built or published,
    and the reason names it in the listing's words, whether another candidate
    of the run is served or none is left."""
    import src.utils.checkpoint_paths as checkpoint_paths
    from tests.fixtures.replacement import hook_read_once

    swapped = family["disease"]
    outside = _put(tmp_path / "outside", "gatv2", "20261010-000000_far_ffffff",
                   _record_payload("disease", logs={"val_pooled_mrr": 0.9}), 1000)
    outside_sha = file_sha256(outside)
    best = None
    if case == "another candidate in the run":
        best = _put(service.data_dir, "gatv2", swapped.parent.parent.name,
                    _record_payload("disease", logs={"val_pooled_mrr": 0.1}), 600,
                    name="best.pt")
    real_check, fired = checkpoint_paths.checkpoint_not_read, []

    def check_then_swap(run_dir, path, within):
        answer = real_check(run_dir, path, within)
        if path == swapped and not fired:
            fired.append(path)
            os.unlink(swapped)
            os.symlink(outside, swapped)
        return answer

    monkeypatch.setattr(checkpoint_paths, "checkpoint_not_read", check_then_swap)
    returned = []
    hook_read_once(monkeypatch, on_return=lambda path: returned.append(os.path.realpath(path)))

    result = _reload(service, task="disease")

    assert fired == [swapped], "the swap fell between the boundary check and the scan's read"
    assert os.path.realpath(outside) not in returned, "the outside file was read"
    assert all(item.config["served_identity"]["checkpoint"]["sha256"] != outside_sha
               for item in service.published)
    words = (f"{swapped.parent.parent.name}/last.pt was opened outside the workspace, "
             "so it is not read")
    assert f"not read: {words}" in result.selection_reason
    if best is None:
        assert not result.success and service.built == [] and service.published == []
        assert "1 checkpoint file(s) of the workspace's runs" in result.message
    else:
        assert result.success, result.message
        assert service.built == [str(best)] and result.checkpoint_path == str(best)
        assert f"not read: {words}" in result.message


def test_an_explicit_path_that_is_missing_is_named_with_a_task_too(service, family):
    missing = service.data_dir / "runs" / "gatv2" / "nope.pt"
    result = _reload(service, task="disease", checkpoint_path=str(missing))
    assert not result.success and service.built == []
    assert f"{missing} does not exist." in result.message
    assert "No run holds" not in result.message and "pass an explicit" not in result.message


def test_a_malformed_record_is_listed_as_refused_and_no_list_fails(tmp_path):
    """A loadable file whose record parts are not plain data of their kind
    (a parent as a string, no parent run, data roles null) is refused whole
    with its reason; the checkpoint lists and the family view still answer."""
    from src.api.services import server_lists as sl
    from src.config import server_settings as ss

    good = _record_payload("disease")
    malformed = {
        "a parent as a string": {**good, "parent": "a" * 64},
        "a parent without its run": {**good, "parent": {"sha256": "a" * 64}},
        "data roles null": {**good, "data_roles": None},
        "a model spec of the wrong shape": {**good, "model_spec": ["not", "a", "mapping"]},
    }
    for case, payload in malformed.items():
        summary = listed_record(_listed_form(payload))
        assert summary["record"] == "refused", case
        assert summary["record_problem"] and summary["task"] is None, case
    root = tmp_path / "root"
    ws = root / "ws"
    for n, payload in enumerate(malformed.values()):
        _put(ws, "gatv2", f"2026100{n + 1}-000000_bad_{n}{n}{n}{n}{n}{n}", payload, 100 + n)
    environ = {ss.ENV_WORKSPACES_ROOT: str(root)}
    listing = sl.list_checkpoints("ws", environ=environ)
    assert listing.available, listing.problem
    assert [item.record for item in listing.items] == ["refused"] * len(malformed)
    family = sl.list_family("ws", environ=environ)
    assert family.available, family.problem


# ---------------------------------------------------------------------------
# What the lists show of a record
# ---------------------------------------------------------------------------
def _listed_form(payload):
    """The record as the run listing carries it: `record_version` and every
    other key the reader reads, no weights (`ListedCheckpoint.model_record`)."""
    from src.config.model_types import MODEL_RECORD_READ_KEYS

    return {"record_version": payload["record_version"],
            **{k: payload[k] for k in MODEL_RECORD_READ_KEYS if k in payload}}


def test_the_listed_record_states():
    child = listed_record(_listed_form(_record_payload("disease")))
    assert set(child) == set(LISTED_RECORD_FIELDS)
    assert child == {"record": "model", "record_problem": None, "learning_stage": "finetune",
                     "task": "disease", "operation": "finetune", "family_id": BASE_RUN,
                     "parent_sha256": "a" * 64, "parent_run": BASE_RUN, "encoder": "gatv2",
                     "reference": False, "differences": SMALL_SPEC_DIFFERENCES}
    base = listed_record(_listed_form(_record_payload()))
    assert (base["learning_stage"], base["task"], base["operation"], base["parent_sha256"]) == (
        "base", None, "new", None)
    for history in (None, {"record_version": 1, "epoch": 2}):
        assert listed_record(history) == {**dict.fromkeys(LISTED_RECORD_FIELDS),
                                          "record": "history", "record_problem": LISTED_WITHOUT_SPEC}
    refused = listed_record({**_listed_form(_record_payload("disease")), "record_version": 1})
    assert refused["record"] == "refused" and "record version" in refused["record_problem"]
    assert refused["task"] is None, "a refused record states nothing it could not read"
    assert refused["differences"] is None and refused["reference"] is None


#: The fields `_base_spec` changes from the reference recipe (README §5, axis 2).
SMALL_SPEC_DIFFERENCES = ("encoder.heads", "encoder.hidden", "encoder.layers", "encoder.output",
                          "input.dim")


def test_the_listed_record_names_the_fields_that_differ_from_the_reference():
    """README §5 axis 2 lists the differing items, not a bool: a reference spec
    lists none and is the reference; a changed one names each field, sorted."""
    assert LISTED_RECORD_FIELDS[-1] == "differences"
    for task in (None, "gene"):
        spec = reference_spec("base")
        if task is not None:
            spec = finetune_spec_from_parent(spec, task)
        bundle = build_bundle(spec, VIEW)
        record = ModelRecord(
            model_spec=spec, graph_view=bundle.view, data_roles={"kg": "k" * 64},
            graph_exposure={}, loss_identity={},
            **(dict(operation="new", family_id=BASE_RUN) if task is None else
               dict(operation="finetune", family_id=BASE_RUN,
                    parent={"sha256": "a" * 64, "run": BASE_RUN})))
        listed = listed_record({"record_version": MODEL_RECORD_VERSION, **record.to_payload()})
        assert (listed["differences"], listed["reference"]) == ((), True)
    changed = listed_record(_listed_form(_record_payload()))
    assert (changed["differences"], changed["reference"]) == (SMALL_SPEC_DIFFERENCES, False)
    assert changed["differences"] == tuple(sorted(_base_spec().differences_from_reference()))


def test_one_task_scoring_mode_table_and_its_inverse():
    """Decision 7: the pipeline, the reload route and the pages read one table."""
    import inspect

    import src.api.routes.pipeline as reload_route
    import src.inference.pipeline as served
    from src.models.bundle.spec import TASKS

    assert tuple(TASK_SCORING_MODES) == TASKS
    assert set(TASK_SCORING_MODES.values()) <= set(SERVED_SCORING_MODES)
    assert {SCORING_MODE_TASKS[mode]: mode for mode in SCORING_MODE_TASKS} == TASK_SCORING_MODES
    assert all(SCORING_MODE_TASKS[TASK_SCORING_MODES[t]] == t for t in TASKS)
    assert not hasattr(reload_route, "_TASK_OF_MODE")
    assert '"disease_task"' not in inspect.getsource(served.DiagnosisPipeline._scoring_mode)


def test_the_listing_carries_the_record_from_its_one_read(tmp_path, monkeypatch):
    from collections import OrderedDict

    import src.utils.checkpoint_paths as cp

    monkeypatch.setattr(cp, "_READS", OrderedDict())
    monkeypatch.setattr(cp, "SETTLED_NS", 0)
    payload = _record_payload("gene", logs={"val_pooled_mrr": 0.25})
    _put(tmp_path, "gatv2", "20261002-000000_gene_bbbbbb", payload, 200)
    _put(tmp_path, "hgt", "20261001-000000_gnn_aaaaaa", GNN_PAYLOAD, 100)
    child, gnn = cp.read_run_checkpoints(tmp_path, within=tmp_path)
    assert child.model_record == _listed_form(payload) and "state_dict" not in child.model_record
    assert child.validation == ("val_pooled_mrr", 0.25)
    assert gnn.model_record is None
    child.model_record["task"] = "disease"  # a caller's change never reaches the kept read
    again, _ = cp.read_run_checkpoints(tmp_path, within=tmp_path)
    assert again.model_record["task"] == "gene"


def test_each_stage_is_listed_by_its_own_selection_metric():
    """The base and finetune selection metrics are among the listing's
    ranking keys, and neither stage's validation logs a key listed before
    them, which would otherwise be shown and ranked in their place."""
    from src.models.bundle.metrics import QueryOutcome, lp_metrics, ranking_report
    from src.training.finetune_task import FINETUNE_SELECTION_METRIC
    from src.training.lp_task import BASE_SELECTION_METRIC
    from src.utils.checkpoint_paths import RANKING_SCORE_KEYS, ranking_score_detail

    for metric in (BASE_SELECTION_METRIC, FINETUNE_SELECTION_METRIC):
        earlier = RANKING_SCORE_KEYS[:RANKING_SCORE_KEYS.index(metric)]
        assert metric in RANKING_SCORE_KEYS
        logged = {f"val_{k}" for k in (
            lp_metrics([1.0, 0.2], [0.1, 0.5]) if metric == BASE_SELECTION_METRIC
            else ranking_report([QueryOutcome("q", "ranked", [2.0])], (1, 10)))}
        assert metric in logged and not logged & set(earlier), metric
    assert ranking_score_detail({"val_lp_accuracy": 0.7, "val_lp_roc_auc": 0.8}) == (
        "val_lp_accuracy", 0.7)
    assert ranking_score_detail({"val_pooled_mrr": 0.3, "val_macro_mrr": 0.4}) == (
        "val_pooled_mrr", 0.3)


def test_the_lists_show_a_bundle_run_and_resolve_it_by_its_key(tmp_path):
    from src.api.services import server_lists as sl
    from src.config import server_settings as ss
    from src.api.services.training_manager import training_manager

    root = tmp_path / "root"
    ws = root / "ws"
    payload = _record_payload("disease", logs={"val_pooled_mrr": 0.3})
    child = _put(ws, "gatv2", "20261002-000000_disease_bbbbbb", payload, 200)
    _put(ws, "hgt", "20261001-000000_gnn_aaaaaa", GNN_PAYLOAD, 100)
    environ = {ss.ENV_WORKSPACES_ROOT: str(root)}
    for architecture in (None, "gatv2"):
        listing = sl.list_checkpoints("ws", architecture, environ=environ)
        assert listing.available, listing.problem
        item = listing.items[0]
        assert (item.architecture, item.run, item.filename) == (
            "gatv2", "20261002-000000_disease_bbbbbb", "last.pt")
        assert (item.record, item.learning_stage, item.task, item.operation, item.family_id,
                item.parent_sha256, item.parent_run, item.encoder, item.reference) == (
            "model", "finetune", "disease", "finetune", BASE_RUN, "a" * 64, BASE_RUN,
            "gatv2", False)
        assert item.validation == {"metric": "val_pooled_mrr", "value": 0.3}
    gnn_item = sl.list_checkpoints("ws", "hgt", environ=environ).items[0]
    assert (gnn_item.record, gnn_item.record_problem) == ("history", LISTED_WITHOUT_SPEC)
    assert sl.resolve_checkpoint("ws", "gatv2", "20261002-000000_disease_bbbbbb", "last.pt",
                                 environ=environ) == child
    by_manager = {c["architecture"]: c for c in training_manager.get_checkpoints(str(ws))}
    assert (by_manager["gatv2"]["task"], by_manager["gatv2"]["parent_sha256"]) == (
        "disease", "a" * 64)
    assert by_manager["hgt"]["record"] == "history"


# ---------------------------------------------------------------------------
# Automatic selection never reads test results (D4)
# ---------------------------------------------------------------------------
def test_a_ledger_does_not_change_what_automatic_selection_chooses(service):
    """Two checkpoints of one disease child run, the validation metric
    (``val_pooled_mrr``) favouring `best.pt`; a ledger beside them whose
    test result (a registered disease task record) favours `other.pt`. The
    reload's task-first selection and `select_auto_run_checkpoint` choose
    `best.pt`, with or without the ledger (the import-linter contract
    `selection-ignores-test-results` keeps the modules apart). Moved from
    `test_test_score_ordering` at K8-2a, from a ShepherdGNN run's history
    files to a task child's."""
    import hashlib
    import json

    from src.evaluation.sidecar import (
        LEDGER_FILENAME,
        append_record,
        build_record,
        empty_ledger,
        read_ledger,
        write_ledger,
    )
    from src.evaluation.task_measurement import disease_semantics
    from src.utils.checkpoint_io import read_checkpoint
    from src.utils.checkpoint_paths import ranking_score_detail, select_auto_run_checkpoint
    from tests.unit.test_evaluation_sidecar import registered_report

    run = "20261002-000000_disease_bbbbbb"
    # `best.pt` older, so recency would not favour it.
    best = _put(service.data_dir, "gatv2", run,
                _record_payload("disease", logs={"val_pooled_mrr": 0.6}), 1, name="best.pt")
    other = _put(service.data_dir, "gatv2", run,
                 _record_payload("disease", logs={"val_pooled_mrr": 0.2}), 900, name="other.pt")

    def score(path):
        logs = read_checkpoint(path, map_location="cpu", weights_only=True).checkpoint["logs"]
        return ranking_score_detail(logs)[1]

    def chosen():
        reload = _reload(service, task="disease")
        assert reload.success, reload.message
        assert "best val_pooled_mrr=0.6000" in reload.selection_reason
        return (select_auto_run_checkpoint(service.data_dir, score_fn=score)[0],
                Path(reload.checkpoint_path))

    without = chosen()
    ledger = empty_ledger()
    for n, (path, pooled_mrr) in enumerate(((other, 0.99), (best, 0.01)), start=1):
        report = registered_report(metrics={"pooled_mrr": pooled_mrr}, **disease_semantics(32))
        report["manifest"]["artifact_digests"]["checkpoint"] = file_sha256(path)
        digest = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
        run_id = f"20261020T090000Z-{file_sha256(path)[:12]}-0000000{n}"
        ledger = append_record(ledger, build_record(report, digest), run_id=run_id,
                               registered_at="2026-10-20T09:00:00Z")[1]
    write_ledger(best.parent / LEDGER_FILENAME, ledger, None)
    tested = {record["checkpoint_digest"]: record["metrics"]["pooled_mrr"]
              for record in read_ledger(best.parent / LEDGER_FILENAME)[0]["records"]}
    assert tested == {file_sha256(other): 0.99, file_sha256(best): 0.01}

    assert without == chosen() == (best, best)


# ---------------------------------------------------------------------------
# The request declares five fields; any other is refused before anything is
# read (Q17 (b))
# ---------------------------------------------------------------------------
@pytest.fixture
def client(service, monkeypatch):
    """The real route over `service`, as an operator (D8 passes, so a refusal
    below is the request's own), with a pipeline served (a sentinel) and every
    checkpoint read observed, where the scan reads (`checkpoint_io`) and where
    the build reads (`src.inference.pipeline`)."""
    from fastapi.testclient import TestClient

    import src.api.main as api
    import src.inference.pipeline as served_pipeline
    import src.utils.checkpoint_io as checkpoint_io
    from tests.fixtures.operator_access import OPERATOR_BASE_URL, operator_mode

    operator_mode(monkeypatch)
    sentinel = object()
    monkeypatch.setattr(api.app_state, "pipeline", sentinel, raising=False)
    reads = []
    for module in (checkpoint_io, served_pipeline):
        def observing(path, *args, _read=module.read_checkpoint, **kwargs):
            reads.append(str(path))
            return _read(path, *args, **kwargs)

        monkeypatch.setattr(module, "read_checkpoint", observing)
    return SimpleNamespace(http=TestClient(api.app, base_url=OPERATOR_BASE_URL), api=api,
                           sentinel=sentinel, reads=reads)


#: The bodies the retired scripts sent (cases 1-4 of the guides' old-scripts
#: note) and a `conv_type` alone, each with the one field the request does not
#: declare: (case, body without `data_dir`, that field).
OLD_SCRIPT_BODIES = [
    ("1: an explicit path with conv_type auto", {"checkpoint_path": "disease",
                                                 "conv_type": "auto"}, "conv_type"),
    ("1: an explicit path with conv_type hgt", {"checkpoint_path": "disease",
                                                "conv_type": "hgt"}, "conv_type"),
    ("2: a task with conv_type auto", {"task": "disease", "conv_type": "auto"}, "conv_type"),
    ("2: a task with conv_type null", {"task": "disease", "conv_type": None}, "conv_type"),
    ("3: a task with conv_type hgt", {"task": "disease", "conv_type": "hgt"}, "conv_type"),
    ("4: the digest misspelt", {"checkpoint_path": "disease",
                                "expected_checkpoint_sha": "digest"},
     "expected_checkpoint_sha"),
    ("neither task nor path, conv_type hgt", {"conv_type": "hgt"}, "conv_type"),
]


def _body(service, family, body):
    """`body` with the workspace and the disease child's path and digest
    filled in."""
    filled = {"data_dir": str(service.data_dir)}
    for key, value in body.items():
        if key == "checkpoint_path":
            value = str(family["disease"])
        elif value == "digest":
            value = file_sha256(family["disease"])
        filled[key] = value
    return filled


@pytest.mark.parametrize("case, body, field", OLD_SCRIPT_BODIES,
                         ids=[case for case, _, _ in OLD_SCRIPT_BODIES])
def test_an_old_scripts_request_is_refused_with_422_before_anything_is_read(
        client, service, family, case, body, field):
    """Each request an old script sends carries a field this service does not
    declare: the architecture is read from the checkpoint's own model spec,
    never from the request, and a misspelt digest is one the reload would
    otherwise not check. Refused by request validation (422, the field named),
    before any candidate is selected: nothing read, built or published, and
    the pipeline served stays. Before K8-4 cases 1, 2 and 4 and the last were
    served, and case 3 refused with a 200."""
    response = client.http.post("/api/v1/pipeline/reload", json=_body(service, family, body))

    assert response.status_code == 422, response.text
    (detail,) = response.json()["detail"]
    assert detail["type"] == "extra_forbidden" and detail["loc"] == ["body", field]
    assert client.reads == [] and service.built == [] and service.published == []
    assert client.api.app_state.pipeline is client.sentinel


@pytest.mark.parametrize("body, served", [
    ({"checkpoint_path": "disease"}, "an explicit path"),
    ({"task": "disease"}, "the disease child, selected by its task"),
    ({"checkpoint_path": "disease", "expected_checkpoint_sha256": "digest"}, "the bound path"),
], ids=["1", "2", "4"])
def test_the_same_requests_without_the_undeclared_field_are_served(
        client, service, family, body, served):
    """The positive control, through the same client: cases 1, 2 and 4
    without the field (4 with `expected_checkpoint_sha256` spelt exactly)
    answer 200 and publish the disease child once."""
    response = client.http.post("/api/v1/pipeline/reload", json=_body(service, family, body))

    assert response.status_code == 200, response.text
    answer = response.json()
    assert answer["success"] is True, (served, answer["message"])
    assert answer["checkpoint_path"] == str(family["disease"])
    assert service.built == [str(family["disease"])] and len(service.published) == 1


def test_a_request_with_an_undeclared_field_cannot_be_built_in_process():
    """The same refusal for a caller in process (the dashboard's shared Load):
    the request model refuses the field, with the type the route answers."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as refused:
        PipelineReloadRequest(data_dir="/ws", task="disease", conv_type="auto")
    assert refused.value.errors()[0]["type"] == "extra_forbidden"
    assert set(PipelineReloadRequest.model_fields) == {
        "data_dir", "checkpoint_path", "task", "device", "expected_checkpoint_sha256"}
