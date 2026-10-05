"""
The loaded checkpoint's training metrics reach the Diagnosis tab.
=================================================================
The pipeline copied `mrr`, `hits_at_1` and `hits_at_10` from a checkpoint's
`logs` -- names no trainer in this repository has written -- while the trainer
writes `val_mrr` and `val_hits@k` there. The status line therefore showed the
losses and none of the ranking metrics.

The ranking keys now come from `RANKING_SCORE_KEYS`, the list auto-selection ranks
checkpoints by, so there is one definition of "a checkpoint's ranking metrics".

The keys the trainer writes are taken from the trainer rather than typed here: the
training keys from the real `Trainer._train_epoch`, the validation keys from the
real `Trainer._validate` over the real `RankingMetrics`. A rename on either side
fails this file instead of silently emptying the display again. One step is
copied rather than run: `_run_evaluation_pass` needs a model and data, so its
`RankingMetrics().compute_all(...)` call, with default k values, is repeated here.
"""
import json
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
        state=SimpleNamespace(train_loss_history=[]),
        optimizer=SimpleNamespace(param_groups=[{"lr": 1e-3}]),
    )
    train_metrics = Trainer._train_epoch(train_stub, 0)
    return {**train_metrics, **val_metrics}


def test_every_metric_the_pipeline_copies_is_one_the_trainer_writes():
    """No guessed names: a key the trainer never writes is a lookup that can only
    come back empty, which is how the display lost its ranking metrics."""
    from src.inference.pipeline import CHECKPOINT_LOG_METRICS

    written = _current_trainer_logs()

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


def test_a_current_trainer_checkpoint_shows_its_ranking_metrics(tmp_path):
    """End to end: logs as the trainer writes them, loaded by the pipeline,
    returned by the status route's model, rendered by the Diagnosis tab."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("gradio")
    from fastapi.encoders import jsonable_encoder

    from src.api.routes.pipeline import _status_of
    from src.inference.pipeline import DiagnosisPipeline
    from src.kg.graph import KnowledgeGraph
    from src.webui.components import diagnosis_panel as panel
    from tests.fixtures.synthetic_workspace import build_workspace

    logs = _current_trainer_logs()
    data_dir, checkpoint_path = build_workspace(tmp_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["logs"] = logs
    torch.save(checkpoint, checkpoint_path)

    pipeline = DiagnosisPipeline(
        kg=KnowledgeGraph(),
        graph_data={
            "x_dict": torch.load(data_dir / "node_features.pt", weights_only=False),
            "edge_index_dict": torch.load(data_dir / "edge_indices.pt", weights_only=False),
            "num_nodes_dict": json.loads((data_dir / "num_nodes.json").read_text()),
        },
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
    rendered = panel._format_pipeline_status(status)

    assert f"Val MRR: {logs['val_mrr']:.4f}" in rendered
    assert f"Val Hits@1: {logs['val_hits@1']:.4f}" in rendered
    assert f"Val Hits@10: {logs['val_hits@10']:.4f}" in rendered
    assert f"Val Loss: {logs['val_loss']:.4f}" in rendered
    assert f"Train Loss: {logs['train_loss']:.4f}" in rendered


def test_the_status_line_labels_the_metrics_and_skips_the_descriptive_ones():
    pytest.importorskip("gradio")
    from src.webui.components import diagnosis_panel as panel

    rendered = panel._format_pipeline_status({
        "initialized": True,
        "gnn_ready": True,
        "checkpoint_meta": {
            "epoch": 3,
            "params": 1000,
            "device": "cpu",
            "val_loss": 0.25,
            "train_loss": 0.5,
            "val_mrr": 0.5,
            "val_hits@10": 0.75,
            "val_hits@1": 0.25,
            "flag": True,
            "note": "not a number",
        },
    })

    checkpoint_line = next(
        line for line in rendered.splitlines() if line.startswith("- Checkpoint:")
    )
    assert checkpoint_line == (
        "- Checkpoint: Epoch 3 | 1,000 params | device=cpu | Val Loss: 0.2500 | "
        "Train Loss: 0.5000 | Val MRR: 0.5000 | Val Hits@10: 0.7500 | "
        "Val Hits@1: 0.2500"
    )
