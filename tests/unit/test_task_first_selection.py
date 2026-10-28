"""Automatic selection reads each checkpoint's record first, and the lists show
it (provenance §9.3, hazard 3; README §1).

A reload that names a task serves the latest run holding a model finetuned for
it, judged by each file's model record, never by its directory: a newer base,
another task's child and a checkpoint without a model spec are passed over;
with none it is refused naming the task, and an explicit checkpoint of another
task is refused with nothing published. A reload without a task never chooses
a model bundle, not even one filed under a ShepherdGNN conv type's directory
(an encoder kind may share it: ``hgt``, ``sage``). The run listing carries each
file's record from its one read; the lists show it through the one reader of
records (`listed_record`), and a bundle run resolves by the key it is listed
under (``gatv2``).

Module: tests/unit/test_task_first_selection.py
"""
from __future__ import annotations

import dataclasses
import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from src.api.routes.pipeline import (  # noqa: E402
    REQUIRED_DATA_FILES,
    PipelineReloadRequest,
    run_reload,
)
from src.models.bundle import build_bundle, finetune_spec_from_parent, reference_spec  # noqa: E402
from src.models.bundle.record import (  # noqa: E402
    HISTORY_ONLY,
    LISTED_RECORD_FIELDS,
    MODEL_RECORD_VERSION,
    ModelRecord,
    listed_record,
)
from src.models.bundle.spec import InputSpec  # noqa: E402
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


def _record_payload(task=None, *, logs=None):
    """A bundle checkpoint's payload: a base (`task` None) or a child of
    `BASE_RUN`, its record as the training payload writes it."""
    base = _base_spec()
    spec = base if task is None else finetune_spec_from_parent(base, task)
    bundle = build_bundle(spec, VIEW)
    lineage = (dict(operation="new", family_id=BASE_RUN) if task is None else
               dict(operation="finetune", family_id=BASE_RUN,
                    parent={"sha256": "a" * 64, "run": BASE_RUN}))
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
_MODE_OF_TASK = {"disease": "disease_task", "gene": "gene_task",
                 "patients_like_me": "patients_like_me"}


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
            path_length_penalty=0.9, aggregation_method="weighted_sum",
            max_paths_per_candidate=10, use_direct_path_optimization=True,
            phenotype_normalisation_version=1),
    ).to_dict()


@pytest.fixture
def service(tmp_path, monkeypatch):
    """A workspace the reload accepts as far as the build, and a build that
    serves what the chosen file's own record says (a base is not served), with
    publication recorded instead of performed."""
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
            mode = "gnn_only"
        elif payload.get("task") is None:
            raise RuntimeError("a base model is not served")
        else:
            mode = _MODE_OF_TASK[payload["task"]]
        config = {"served_identity": _identity(mode, file_sha256(path)), "scoring_mode": mode,
                  "gnn_ready": True, "has_model": True, "checkpoint_meta": dict(payload["logs"])}
        return SimpleNamespace(config=config, kg=None, pipeline=object(),
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
        "gnn": _put(d, "gat", "20261009-000000_gnn_eeeeee", GNN_PAYLOAD, 900),
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
        "task disease: auto: latest run '20261002-000000_disease_bbbbbb' (gatv2)")
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


def test_a_task_names_no_conv_type(service, family):
    result = _reload(service, task="disease", conv_type="gat")
    assert not result.success and service.built == []
    assert "conv_type names a ShepherdGNN architecture" in result.message


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
    assert "serves scoring mode 'gnn_only', not the disease task" in gnn.message
    served = _reload(service, task="gene", checkpoint_path=str(family["gene"]))
    assert served.success and len(service.published) == 1


def test_a_reload_without_a_task_never_chooses_a_bundle(service):
    """Hazard 3: a disease child filed under a conv type's directory, newer
    than the ShepherdGNN run beside it, is passed over by the record; with no
    ShepherdGNN file left, nothing is chosen."""
    d = service.data_dir
    gnn = _put(d, "sage", "20261001-000000_gnn_aaaaaa", GNN_PAYLOAD, 100)
    _put(d, "sage", "20261002-000000_child_bbbbbb", _record_payload("disease"), 900)
    _put(d, "gatv2", "20261003-000000_child_cccccc", _record_payload("disease"), 950)
    auto = _reload(service)
    assert auto.success and auto.checkpoint_path == str(gnn) and auto.architecture == "sage"
    named = _reload(service, conv_type="sage")
    assert named.success and named.checkpoint_path == str(gnn)
    gnn.unlink()
    refused = _reload(service)
    assert not refused.success and refused.files_found["checkpoint"] is False
    assert "a model bundle is selected by its task (task)" in refused.message
    assert service.built == [str(gnn), str(gnn)]


# ---------------------------------------------------------------------------
# What the lists show of a record
# ---------------------------------------------------------------------------
def _listed_form(payload):
    """The record as the run listing carries it: `record_version` and the
    record's keys, no weights (`ListedCheckpoint.model_record`)."""
    from src.config.model_types import MODEL_RECORD_KEYS

    return {"record_version": payload["record_version"],
            **{k: payload[k] for k in MODEL_RECORD_KEYS if k in payload}}


def test_the_listed_record_states():
    child = listed_record(_listed_form(_record_payload("disease")))
    assert set(child) == set(LISTED_RECORD_FIELDS)
    assert child == {"record": "model", "record_problem": None, "learning_stage": "finetune",
                     "task": "disease", "operation": "finetune", "family_id": BASE_RUN,
                     "parent_sha256": "a" * 64, "parent_run": BASE_RUN, "encoder": "gatv2",
                     "reference": False}
    base = listed_record(_listed_form(_record_payload()))
    assert (base["learning_stage"], base["task"], base["operation"], base["parent_sha256"]) == (
        "base", None, "new", None)
    for history in (None, {"record_version": 1, "epoch": 2}):
        assert listed_record(history) == {**dict.fromkeys(LISTED_RECORD_FIELDS),
                                          "record": "history", "record_problem": HISTORY_ONLY}
    refused = listed_record({**_listed_form(_record_payload("disease")), "record_version": 1})
    assert refused["record"] == "refused" and "record version" in refused["record_problem"]
    assert refused["task"] is None, "a refused record states nothing it could not read"


def test_the_listing_carries_the_record_from_its_one_read(tmp_path, monkeypatch):
    from collections import OrderedDict

    import src.utils.checkpoint_paths as cp

    monkeypatch.setattr(cp, "_READS", OrderedDict())
    monkeypatch.setattr(cp, "SETTLED_NS", 0)
    payload = _record_payload("gene", logs={"val_pooled_mrr": 0.25})
    _put(tmp_path, "gatv2", "20261002-000000_gene_bbbbbb", payload, 200)
    _put(tmp_path, "gat", "20261001-000000_gnn_aaaaaa", GNN_PAYLOAD, 100)
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
    _put(ws, "gat", "20261001-000000_gnn_aaaaaa", GNN_PAYLOAD, 100)
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
    gnn_item = sl.list_checkpoints("ws", "gat", environ=environ).items[0]
    assert (gnn_item.record, gnn_item.record_problem) == ("history", HISTORY_ONLY)
    assert sl.resolve_checkpoint("ws", "gatv2", "20261002-000000_disease_bbbbbb", "last.pt",
                                 environ=environ) == child
    by_manager = {c["architecture"]: c for c in training_manager.get_checkpoints(str(ws))}
    assert (by_manager["gatv2"]["task"], by_manager["gatv2"]["parent_sha256"]) == (
        "disease", "a" * 64)
    assert by_manager["gat"]["record"] == "history"
