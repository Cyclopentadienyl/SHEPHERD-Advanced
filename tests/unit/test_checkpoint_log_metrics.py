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
training keys from the real `Trainer._train_epoch`, and the validation keys from
the real `Trainer._validate` over the real reports a stage's task validates with
(the base's LP metrics, a finetune's ranking report). A rename on either side
fails this file instead of silently emptying the display again. The check runs
one way: what this trainer writes for the status line must be copied. A key
the pipeline copies that this trainer does not write (one an existing
checkpoint's logs may carry) is a reading of what is there, not an error.
"""
from types import SimpleNamespace

import pytest


def _training_logs():
    """What `Trainer._train_epoch` adds to an epoch's logs (`train_loss` and
    the rest), from the real method over an empty training set, against a stub
    that supplies only what it reads."""
    pytest.importorskip("torch")
    from src.training.trainer import Trainer

    train_stub = SimpleNamespace(
        model=SimpleNamespace(train=lambda: None),
        callbacks=SimpleNamespace(on_epoch_begin=lambda trainer, epoch: None),
        train_dataloader=[],
        task=SimpleNamespace(),
        state=SimpleNamespace(train_loss_history=[]),
        optimizer=SimpleNamespace(param_groups=[{"lr": 1e-3}]),
        config=SimpleNamespace(gradient_accumulation_steps=1),
    )
    return Trainer._train_epoch(train_stub, 0)


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
            config=SimpleNamespace(early_stopping_monitor="val_pooled_mrr",
                                   early_stopping_mode="max"),
            state=SimpleNamespace(best_metric=None, best_epoch=None, val_metric_history=[]),
        )
        stub._is_best_metric = lambda value, stub=stub: Trainer._is_best_metric(stub, value)
        logs.update(Trainer._validate(stub, 0))
    return logs


def test_the_metrics_the_trainer_writes_for_the_status_line_are_the_ones_copied():
    """No guessed names: the display lost its ranking metrics when the pipeline
    asked for names the trainer never wrote. The training loss and each
    stage's selection metric, under the names the training epoch and a model
    bundle stage's task write them, are among the keys the pipeline copies."""
    from src.inference.pipeline import CHECKPOINT_LOG_METRICS
    from src.training.finetune_task import FINETUNE_SELECTION_METRIC
    from src.training.lp_task import BASE_SELECTION_METRIC

    written = {**_training_logs(), **_task_validation_logs()}
    shown = {"train_loss", BASE_SELECTION_METRIC, FINETUNE_SELECTION_METRIC}

    assert shown <= set(written), f"the trainer does not write {sorted(shown - set(written))}"
    assert shown <= set(CHECKPOINT_LOG_METRICS), (
        f"the pipeline does not copy {sorted(shown - set(CHECKPOINT_LOG_METRICS))}")


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


def test_a_current_trainer_checkpoint_shows_its_ranking_metrics(trained_disease_child,
                                                                 disease_pipeline):
    """End to end: logs as the trainer writes them, loaded by the pipeline,
    returned by the status route's model, shown by the Diagnosis page (the
    card and Model Facts). The checkpoint is a trained task model (K8-2a): the
    session's disease child, whose `logs` the real Trainer wrote in its run; a
    task model's ranking metric is `val_pooled_mrr`."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("gradio")
    from datetime import datetime

    from fastapi.encoders import jsonable_encoder

    from src.api.routes.pipeline import _status_of
    from src.webui.components import diagnosis_card as card
    from src.webui.components import diagnosis_panel as panel

    logs = torch.load(trained_disease_child["disease"], map_location="cpu",
                      weights_only=False)["logs"]

    # From the workspace, as the service builds it (`disease_pipeline`): a
    # checkpoint is loaded only over a graph verified from one (contract M2.4).
    pipeline = disease_pipeline

    meta = pipeline.get_pipeline_config()["checkpoint_meta"]
    assert meta["val_pooled_mrr"] == pytest.approx(logs["val_pooled_mrr"])
    # Copied by choice, not wholesale: the ranking report's other figures, epoch
    # time and the learning rate are in the logs and are not status-line material.
    descriptive = {"val_pooled_top1", "val_macro_mrr", "epoch_time", "learning_rate"}
    assert descriptive <= set(logs) and not descriptive & set(meta)

    status = jsonable_encoder(_status_of(pipeline.get_pipeline_config(), None, None))
    state = card.advance(None, status, None, now=datetime(2026, 10, 20, 10, 0), operator=True,
                         d8_short=None, open_checkpoints=False, facts=_no_listing)
    details = panel.model_identity_html(state)

    # Model Facts' Evidence: the Trainer's selection metric, named as such.
    assert (f"Validation pooled MRR (Trainer) {logs['val_pooled_mrr']:.4f} — selection metric "
            "on val, not an independent test") in details

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
      "val_pooled_mrr": 0.5, "val_lp_accuracy": 0.75, "val_pooled_top1": 0.25, "flag": True,
      "note": "not a number"}, "Validation pooled MRR (Trainer) 0.5000"),
    ({"epoch": 3, "val_lp_accuracy": 0.75, "val_pooled_top1": 0.25},
     "Validation LP accuracy (Trainer) 0.7500"),
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
                  "scoring_mode": "disease_task",
                  "checkpoint": {"state": "verified", "sha256": "a" * 64},
                  "graph": {"state": "verified", "kg_sha256": "b" * 64,
                            "split_manifest_sha256": "c" * 64},
                  "shortest_paths": {"state": "absent"},
                  "dataset_record": {"state": "verified", "origin": "files", "sha256": "d" * 64},
                  "settings": {}}}
    state = card.advance(None, status, None, now=datetime(2026, 10, 20), operator=True,
                         d8_short=None, open_checkpoints=False, facts=_no_listing)
    details = card.served_facts_html(state)
    assert shown in details
    assert "Val Loss" not in details and "Train Loss" not in details
