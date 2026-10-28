"""
The loaded checkpoint's training metrics reach the Diagnosis page.
==================================================================
The pipeline copied `mrr`, `hits_at_1` and `hits_at_10` from a checkpoint's
`logs` -- names no trainer in this repository has written -- while the trainer
writes `val_mrr` and `val_hits@k` there. The status line therefore showed the
losses and none of the ranking metrics.

The ranking keys now come from `RANKING_SCORE_KEYS`, the list auto-selection ranks
checkpoints by, so there is one definition of "a checkpoint's ranking metrics".

The keys the trainer writes are taken from the trainer rather than typed here: the
training keys from the real `Trainer._train_epoch`, the validation keys from the
real `Trainer._validate` over the real `RankingMetrics`, and, for a model bundle
run, over the real reports its stage's task validates with (the base's LP
metrics, a finetune's ranking report). A rename on either side fails this file
instead of silently emptying the display again. One step is copied rather than
run: `_run_evaluation_pass` needs a model and data, so its
`RankingMetrics().compute_all(...)` call, with default k values, is repeated here.
"""
from types import SimpleNamespace

import pytest


def _current_trainer_logs():
    """What `ModelCheckpoint._save_checkpoint` stores as `logs` for one epoch.

    `Trainer.train` passes `{**train_metrics, **val_metrics}` to the callbacks,
    and the checkpoint callback saves it verbatim. Both halves come from the real
    methods, run against a stub that supplies only what each reads:
    `train_metrics` from `Trainer._train_epoch` over an empty training set, and
    `val_metrics` from `Trainer._validate`.
    """
    pytest.importorskip("torch")
    from src.training.trainer import Trainer
    from src.utils.metrics import RankingMetrics

    ranking = RankingMetrics().compute_all([["d1", "d2"], ["d3", "d1"]], ["d1", "d1"])
    stub = SimpleNamespace(
        model=SimpleNamespace(eval=lambda: None),
        callbacks=SimpleNamespace(
            on_validation_begin=lambda trainer: None,
            on_validation_end=lambda trainer, metrics: None,
        ),
        val_dataloader=None,
        task=None,
        _run_evaluation_pass=lambda loader: SimpleNamespace(
            ranking_metrics=ranking, mean_loss=0.25
        ),
        config=SimpleNamespace(early_stopping_monitor="val_mrr", early_stopping_mode="max"),
        state=SimpleNamespace(best_metric=None, best_epoch=None, val_metric_history=[]),
    )
    stub._is_best_metric = lambda value: Trainer._is_best_metric(stub, value)
    val_metrics = Trainer._validate(stub, 0)

    train_stub = SimpleNamespace(
        model=SimpleNamespace(train=lambda: None),
        callbacks=SimpleNamespace(on_epoch_begin=lambda trainer, epoch: None),
        train_dataloader=[],
        task=None,
        state=SimpleNamespace(train_loss_history=[]),
        optimizer=SimpleNamespace(param_groups=[{"lr": 1e-3}]),
        config=SimpleNamespace(gradient_accumulation_steps=1),
    )
    train_metrics = Trainer._train_epoch(train_stub, 0)
    return {**train_metrics, **val_metrics}


def _task_validation_logs():
    """What `Trainer._validate` writes for a model bundle run, whose stage's
    task validates itself: the base's LP metrics (`lp_metrics`) and a
    finetune's ranking report (`finetune_task._report`), from the real
    functions the tasks return."""
    pytest.importorskip("torch")
    from src.models.bundle.metrics import QueryOutcome, lp_metrics
    from src.training.finetune_task import _report
    from src.training.trainer import Trainer

    logs = {}
    for report in (lp_metrics([1.0, 0.2], [0.1, 0.5]),
                   _report([QueryOutcome("q", "ranked", [2.0])], [], (1, 10))):
        stub = SimpleNamespace(
            model=SimpleNamespace(eval=lambda: None), device="cpu",
            callbacks=SimpleNamespace(on_validation_begin=lambda trainer: None,
                                      on_validation_end=lambda trainer, metrics: None),
            task=SimpleNamespace(validate=lambda model, device, report=report: dict(report)),
            config=SimpleNamespace(early_stopping_monitor="val_mrr", early_stopping_mode="max"),
            state=SimpleNamespace(best_metric=None, best_epoch=None, val_metric_history=[]),
        )
        stub._is_best_metric = lambda value, stub=stub: Trainer._is_best_metric(stub, value)
        logs.update(Trainer._validate(stub, 0))
    return logs


def test_every_metric_the_pipeline_copies_is_one_the_trainer_writes():
    """No guessed names: a key the trainer never writes is a lookup that can only
    come back empty, which is how the display lost its ranking metrics. Written
    is what either validation path writes: the ShepherdGNN pass or a model
    bundle stage's task."""
    from src.inference.pipeline import CHECKPOINT_LOG_METRICS

    written = {**_current_trainer_logs(), **_task_validation_logs()}

    missing = [key for key in CHECKPOINT_LOG_METRICS if key not in written]
    assert not missing, f"the pipeline copies keys the trainer does not write: {missing}"


def test_the_ranking_metrics_shown_are_the_ones_selection_ranks_by():
    from src.inference.pipeline import CHECKPOINT_LOG_METRICS
    from src.utils.checkpoint_paths import RANKING_SCORE_KEYS

    assert set(RANKING_SCORE_KEYS) <= set(CHECKPOINT_LOG_METRICS)
    assert {"val_mrr", "val_hits@1", "val_hits@10"} <= set(RANKING_SCORE_KEYS)


def test_the_ranking_keys_are_not_listed_a_second_time():
    """One definition. The two tests above catch a list that drifts from what the
    trainer writes or from what selection ranks by, but an identical hand-typed
    copy would pass both. So this reads the assignment itself: it must splice
    `RANKING_SCORE_KEYS` in and must not name a ranking key on its own."""
    import ast
    from pathlib import Path

    from src.utils.checkpoint_paths import RANKING_SCORE_KEYS

    source = Path(__file__).resolve().parents[2] / "src" / "inference" / "pipeline.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    (value,) = [
        node.value
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == "CHECKPOINT_LOG_METRICS"
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
    ]
    assert isinstance(value, ast.Tuple)
    spliced = [
        element.value.id
        for element in value.elts
        if isinstance(element, ast.Starred) and isinstance(element.value, ast.Name)
    ]
    assert spliced == ["RANKING_SCORE_KEYS"]
    literals = {element.value for element in value.elts if isinstance(element, ast.Constant)}
    assert not literals & set(RANKING_SCORE_KEYS), (
        f"ranking keys typed a second time: {sorted(literals & set(RANKING_SCORE_KEYS))}"
    )
    # And the name spliced in is the imported object, not a local copy that shadows it.
    from src.inference import pipeline
    from src.utils import checkpoint_paths

    assert pipeline.RANKING_SCORE_KEYS is checkpoint_paths.RANKING_SCORE_KEYS


def test_a_current_trainer_checkpoint_shows_its_ranking_metrics(tmp_path):
    """End to end: logs as the trainer writes them, loaded by the pipeline,
    returned by the status route's model, shown by the Diagnosis page (the
    card and Model Facts)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("gradio")
    from datetime import datetime

    from fastapi.encoders import jsonable_encoder

    from src.api.routes.pipeline import _status_of
    from src.inference.pipeline import DiagnosisPipeline
    from src.kg.graph import KnowledgeGraph
    from src.webui.components import diagnosis_card as card
    from src.webui.components import diagnosis_panel as panel
    from tests.fixtures.synthetic_workspace import build_workspace

    logs = _current_trainer_logs()
    data_dir, checkpoint_path = build_workspace(tmp_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["logs"] = logs
    torch.save(checkpoint, checkpoint_path)

    # From the workspace, as the service builds it: a checkpoint is loaded only
    # over a graph verified from one (contract M2.4).
    pipeline = DiagnosisPipeline(
        graph=KnowledgeGraph.read_json(data_dir / "kg.json"),
        data_dir=str(data_dir),
        checkpoint_path=str(checkpoint_path),
        device="cpu",
    )

    meta = pipeline.get_pipeline_config()["checkpoint_meta"]
    for key in ("val_mrr", "val_hits@1", "val_hits@10"):
        assert meta[key] == pytest.approx(logs[key])
    # Copied by choice, not wholesale: ndcg, epoch time and the learning rate
    # are in the logs and are not status-line material.
    assert "val_ndcg@1" not in meta and "learning_rate" not in meta

    status = jsonable_encoder(_status_of(pipeline.get_pipeline_config(), None, None))
    state = card.advance(None, status, None, now=datetime(2026, 10, 20, 10, 0), operator=True,
                         d8_short=None, open_checkpoints=False, facts=_no_listing)
    details = panel.model_identity_html(state)

    # Model Facts' Evidence: the Trainer's selection metric, named as such.
    assert (f"Validation MRR (Trainer, top-20) {logs['val_mrr']:.4f} — selection metric on "
            "val, not an independent test") in details

    # The same status names what the loaded pipeline serves (contract M3c),
    # taken from the pipeline's own served identity: the digest of the
    # checkpoint read it was built from, and the workspace's synthetic origin.
    served = pipeline.served_identity()
    assert served.checkpoint.state == "verified"
    body = card.body_html(state)
    assert f">{served.checkpoint.sha256[:12]}</code>" in body and "synthetic data" in body
    assert "SYNTHETIC WORKSPACE: not a clinical result." in details
    assert panel.IDENTITY_CLAIM_BOUNDARY in details


def _no_listing(status, **kwargs):
    """What the card adds, without a workspaces root (no list read)."""
    from src.webui.components import diagnosis_card as card

    return {"key": card.served_key(status), "operator": kwargs.get("operator"), "workspaces": [],
            "serving": None, "item": None, "item_problem": None, "recorded": None,
            "trained": None}


@pytest.mark.parametrize("meta, shown", [
    ({"epoch": 3, "params": 1000, "device": "cpu", "val_loss": 0.25, "train_loss": 0.5,
      "val_mrr": 0.5, "val_hits@10": 0.75, "val_hits@1": 0.25, "flag": True,
      "note": "not a number"}, "Validation MRR (Trainer, top-20) 0.5000"),
    ({"epoch": 3, "val_hits@10": 0.75, "val_hits@1": 0.25}, "Validation Hits@10 (Trainer) 0.7500"),
    ({"epoch": 3, "val_loss": 0.25}, "Validation (Trainer): not recorded in this checkpoint"),
])
def test_details_show_the_selection_metric_by_the_ranking_keys(meta, shown):
    """The figure is the first of `RANKING_SCORE_KEYS` the status's
    ``checkpoint_meta`` carries (the one definition of a checkpoint's ranking
    metric); losses and descriptive keys are not a validation figure."""
    pytest.importorskip("gradio")
    from datetime import datetime

    from src.webui.components import diagnosis_card as card

    status = {"initialized": True, "gnn_ready": True, "checkpoint_meta": meta,
              "served_identity": {
                  "scoring_mode": "gnn_only",
                  "checkpoint": {"state": "verified", "sha256": "a" * 64},
                  "graph": {"state": "verified", "kg_sha256": "b" * 64,
                            "split_manifest_sha256": "c" * 64},
                  "shortest_paths": {"state": "absent", "build_id": None, "max_hops": None},
                  "dataset_record": {"state": "verified", "origin": "files", "sha256": "d" * 64},
                  "settings": {}}}
    state = card.advance(None, status, None, now=datetime(2026, 10, 20), operator=True,
                         d8_short=None, open_checkpoints=False, facts=_no_listing)
    details = card.served_facts_html(state)
    assert shown in details
    assert "Val Loss" not in details and "Train Loss" not in details
