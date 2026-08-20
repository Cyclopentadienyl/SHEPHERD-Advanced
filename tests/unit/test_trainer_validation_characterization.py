"""Backlog item 1b — freeze what `Trainer._validate` and `Trainer.evaluate` do.

`_validate` (`trainer.py:615-681`) and `evaluate` (`:773-849`) duplicate the same
calculation. Item 1c extracts the shared pass and item 1d compares it against the
Mode A harness. **This file exists so that extraction can be shown to preserve
behaviour rather than asserted to.**

**These are characterization tests, not specifications.** They record what the
code does today, including things that are arguably wrong — AMP silently
disabling itself on CPU, the two entry points disagreeing about metric prefixes,
a top-20 truncation that discards the rest of the ranking. Freezing a defect is
the point: 1c must not change any of it by accident, and a deliberate change
should have to edit a test that says what it is changing.

**Scope, bounded on purpose.** No extraction, no `Trainer` redesign, no
evaluation framework, no differential calibration. One file, and nothing under
`src/` is touched.

**Every assertion here was checked against a mutation that should break it**,
because a characterization test that passes on changed code is worse than none —
it certifies a change it did not examine. Six mutations were run: dropping the
`val_` prefix, truncating at 10 instead of 20, replacing the mean loss with the
last batch's, moving the loss outside `autocast`, clamping only the targets, and
restoring the whole pre-fix clamp. Five failed the test that names them. The
sixth — clamping the targets alone — passed everything, which turned out to be a
finding rather than a gap and is now pinned by the last test in this file.

Module: tests/unit/test_trainer_validation_characterization.py
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest
import torch
import torch.nn as nn

from src.training.callbacks import Callback
from src.training.trainer import Trainer, TrainerConfig

#: 25, not 5, so the `[:20]` truncation is **observable**. At five diseases a
#: truncated and an untruncated ranking are the same list and the test would pass
#: against code that had dropped the truncation entirely.
N_DISEASES = 25
N_PHENOTYPES = 7
HIDDEN = 4


class RecordingModel(nn.Module):
    """Deterministic embeddings, and a forward counter.

    The output ignores its inputs so that two batches differ only in their
    labels: that isolates the aggregation being characterized from the model.
    It still owns a parameter, because `Trainer` builds an optimizer over
    `model.parameters()`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(HIDDEN))
        self.forward_calls = 0

    def forward(self, x_dict, edge_index_dict) -> Dict[str, torch.Tensor]:
        self.forward_calls += 1
        generator = torch.Generator().manual_seed(0)
        return {
            "disease": torch.randn(N_DISEASES, HIDDEN, generator=generator) @ self.weight,
            "phenotype": torch.randn(N_PHENOTYPES, HIDDEN, generator=generator) @ self.weight,
        }


class RecordingCallback(Callback):
    """Every validation hook, in call order, with the keys it was handed."""

    def __init__(self) -> None:
        self.events: List[Any] = []

    def on_validation_begin(self, trainer, **kwargs) -> None:
        self.events.append("validation_begin")

    def on_validation_end(self, trainer, logs, **kwargs) -> None:
        self.events.append(("validation_end", sorted(logs)))


def make_batch(disease_ids: List[int]) -> Dict[str, Any]:
    """One yielded dict in the shape `DiagnosisDataLoader._process_batch` returns."""
    return {
        "batch": {
            "phenotype_ids": torch.tensor([[0, 1, -1], [2, 3, -1]]),
            "phenotype_mask": torch.tensor([[True, True, False], [True, True, False]]),
            "disease_ids": torch.tensor(disease_ids),
            "patient_ids": ["p0", "p1"],
        },
        "subgraph_x_dict": {},
        "subgraph_edge_index_dict": {},
    }


def make_trainer(val_batches, callbacks=None, **config_overrides) -> Trainer:
    """A CPU trainer with no default callbacks and no scheduler.

    `callbacks` is passed non-empty on purpose: `Trainer.__init__` does
    `callbacks or self._create_default_callbacks()`, so an empty list would
    silently install checkpointing and early stopping, which write files.
    """
    settings = dict(device="cpu", use_amp=True, scheduler_type="none", seed=0)
    settings.update(config_overrides)
    return Trainer(
        model=RecordingModel(),
        train_dataloader=[],
        val_dataloader=val_batches,
        config=TrainerConfig(**settings),
        callbacks=callbacks or [RecordingCallback()],
    )


# ---------------------------------------------------------------------------
# 1. Metric keys and prefixes — the two entry points disagree
# ---------------------------------------------------------------------------
RANKING_KEYS = {
    "mrr",
    *(f"hits@{k}" for k in (1, 3, 5, 10, 20)),
    *(f"ndcg@{k}" for k in (1, 3, 5, 10, 20)),
}


def test_validate_prefixes_every_ranking_key_and_adds_val_loss():
    trainer = make_trainer([make_batch([0, 1])])
    metrics = trainer._validate(epoch=1)

    assert set(metrics) == {f"val_{k}" for k in RANKING_KEYS} | {"val_loss"}


def test_evaluate_returns_the_same_metrics_unprefixed_and_calls_the_loss_key_loss():
    """**The two contracts differ, and 1c must preserve the difference.**

    `_validate` emits `val_mrr` and `val_loss`; `evaluate` emits `mrr` and
    `loss`. A shared pass that returned one shape for both callers would silently
    break `early_stopping_monitor="val_mrr"`, which reads the prefixed name.
    """
    trainer = make_trainer([make_batch([0, 1])])
    metrics = trainer.evaluate()

    assert set(metrics) == RANKING_KEYS | {"loss"}
    assert not any(k.startswith("val_") for k in metrics)


# ---------------------------------------------------------------------------
# 2. Loss aggregation — the mean over batches
# ---------------------------------------------------------------------------
def test_val_loss_is_the_mean_of_the_per_batch_losses():
    """Computed independently, not read back from the same accumulator."""
    batches = [make_batch([0, 1]), make_batch([7, 13])]
    trainer = make_trainer(batches)

    expected = []
    with torch.no_grad():
        for batch_data in batches:
            batch = batch_data["batch"]
            embeddings = trainer.model(batch_data["subgraph_x_dict"], {})
            outputs = trainer._compute_model_outputs(embeddings, batch, {}, {})
            loss, _ = trainer.loss_fn(batch, outputs)
            expected.append(float(loss))
    trainer.model.forward_calls = 0

    metrics = trainer._validate(epoch=1)

    assert metrics["val_loss"] == pytest.approx(sum(expected) / len(expected))
    assert expected[0] != pytest.approx(expected[1]), (
        "the two batches must differ, or this test cannot tell a mean from a last value"
    )


def test_an_empty_dataloader_divides_by_one_rather_than_raising():
    """`total_loss / max(num_batches, 1)`, frozen as observed."""
    trainer = make_trainer([])
    metrics = trainer._validate(epoch=1)

    assert metrics == {"val_loss": 0.0}, "no ranking metrics without predictions"


# ---------------------------------------------------------------------------
# 3. Callback order and count
# ---------------------------------------------------------------------------
def test_validation_calls_begin_once_then_end_once_with_the_metrics():
    recorder = RecordingCallback()
    trainer = make_trainer([make_batch([0, 1]), make_batch([2, 3])], callbacks=[recorder])

    metrics = trainer._validate(epoch=1)

    assert len(recorder.events) == 2, "one begin and one end, regardless of batch count"
    assert recorder.events[0] == "validation_begin"
    name, logs = recorder.events[1]
    assert name == "validation_end"
    assert logs == sorted(metrics), "the end hook receives the returned metrics"


def test_evaluate_fires_no_validation_callbacks():
    """`evaluate` is not `_validate`: it drives no hooks and touches no state."""
    recorder = RecordingCallback()
    trainer = make_trainer([make_batch([0, 1])], callbacks=[recorder])

    trainer.evaluate()

    assert recorder.events == []
    assert trainer.state.val_metric_history == []
    assert trainer.state.best_metric is None


# ---------------------------------------------------------------------------
# 4. Best metric and epoch updates
# ---------------------------------------------------------------------------
def test_best_metric_and_epoch_track_the_monitored_key():
    trainer = make_trainer([make_batch([0, 1])])
    first = trainer._validate(epoch=3)

    assert trainer.state.best_metric == first["val_mrr"]
    assert trainer.state.best_epoch == 3
    assert len(trainer.state.val_metric_history) == 1


def test_a_non_improving_epoch_leaves_best_alone_but_still_appends_history():
    """`early_stopping_mode="max"`, so an equal value is not an improvement."""
    trainer = make_trainer([make_batch([0, 1])])
    trainer._validate(epoch=3)
    trainer._validate(epoch=4)  # identical batch, identical metric

    assert trainer.state.best_epoch == 3, "equal is not better"
    assert len(trainer.state.val_metric_history) == 2, "history records every call"


def test_a_missing_monitor_key_leaves_best_metric_unset():
    """`early_stopping_monitor` naming a key nobody emits is silently ignored."""
    trainer = make_trainer([make_batch([0, 1])], early_stopping_monitor="val_nonexistent")
    trainer._validate(epoch=1)

    assert trainer.state.best_metric is None
    assert len(trainer.state.val_metric_history) == 1


# ---------------------------------------------------------------------------
# 5. Model forward count
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_batches", [1, 2, 3])
def test_one_forward_per_batch(n_batches):
    trainer = make_trainer([make_batch([0, 1]) for _ in range(n_batches)])
    trainer._validate(epoch=1)

    assert trainer.model.forward_calls == n_batches


def test_evaluate_prefers_its_argument_over_the_val_dataloader():
    trainer = make_trainer([make_batch([0, 1]) for _ in range(5)])
    trainer.evaluate([make_batch([0, 1]) for _ in range(2)])

    assert trainer.model.forward_calls == 2


def test_evaluate_with_no_dataloader_at_all_returns_empty():
    trainer = make_trainer(None)

    assert trainer.evaluate() == {}
    assert trainer.model.forward_calls == 0


# ---------------------------------------------------------------------------
# 6. The local top-20 rows and truth ids handed to the metric
# ---------------------------------------------------------------------------
def _capture_metric_inputs(monkeypatch):
    """Record what `RankingMetrics.compute_all` is called with.

    The prediction lists are local to `_validate`, and they are the artifact 1d
    will compare against Mode A's `legacy_top_k_local`. Capturing the call is the
    only way to freeze their shape without changing the code under test.
    """
    import src.training.trainer as trainer_module

    seen: Dict[str, Any] = {}
    original = trainer_module.RankingMetrics

    class Capturing(original):  # type: ignore[misc,valid-type]
        def compute_all(self, predictions, ground_truths, k_values=None):
            seen["predictions"] = predictions
            seen["ground_truths"] = ground_truths
            return super().compute_all(predictions, ground_truths, k_values)

    monkeypatch.setattr(trainer_module, "RankingMetrics", Capturing)
    return seen


def test_predictions_are_local_column_indices_as_strings_truncated_to_twenty(monkeypatch):
    seen = _capture_metric_inputs(monkeypatch)
    trainer = make_trainer([make_batch([4, 9])])
    trainer._validate(epoch=1)

    predictions = seen["predictions"]
    assert len(predictions) == 2, "one row per sample"
    for row in predictions:
        assert len(row) == 20, f"truncated to 20 of {N_DISEASES}"
        assert all(isinstance(v, str) for v in row)
        assert {int(v) for v in row} <= set(range(N_DISEASES)), "local column indices"
        assert len(set(row)) == 20, "a ranking has no repeats"


def test_ground_truths_are_the_stringified_local_disease_ids(monkeypatch):
    seen = _capture_metric_inputs(monkeypatch)
    trainer = make_trainer([make_batch([4, 9])])
    trainer._validate(epoch=1)

    assert seen["ground_truths"] == ["4", "9"]


def test_rows_accumulate_across_batches_in_dataloader_order(monkeypatch):
    seen = _capture_metric_inputs(monkeypatch)
    trainer = make_trainer([make_batch([4, 9]), make_batch([1, 2])])
    trainer._validate(epoch=1)

    assert seen["ground_truths"] == ["4", "9", "1", "2"]
    assert len(seen["predictions"]) == 4


# ---------------------------------------------------------------------------
# 7. AMP resolution and placement
# ---------------------------------------------------------------------------
def test_amp_disables_itself_on_cpu_whatever_the_config_asks_for():
    """**Frozen as a fact, not endorsed.** `_setup_amp` is
    `config.use_amp and device.type == "cuda"` (`trainer.py:380`), so a CPU run
    requesting float16 silently gets float32 and no scaler. Any measurement taken
    on CPU is therefore not the deployment numeric context, and 1d's differential
    calibration has to say which one it ran in.
    """
    trainer = make_trainer([], use_amp=True, amp_dtype="float16")

    assert trainer.use_amp is False
    assert trainer.amp_dtype is torch.float32
    assert trainer.scaler is None


def test_the_forward_and_the_loss_are_inside_autocast_and_the_metrics_are_not(monkeypatch):
    """Placement, not just settings — 1c must not move either boundary."""
    import src.training.trainer as trainer_module

    events: List[str] = []
    real_autocast = trainer_module.autocast

    class Tracing:
        def __init__(self, *args, **kwargs):
            self._inner = real_autocast(*args, **kwargs)
            events.append(f"autocast(enabled={kwargs.get('enabled')})")

        def __enter__(self):
            events.append("enter")
            return self._inner.__enter__()

        def __exit__(self, *exc):
            events.append("exit")
            return self._inner.__exit__(*exc)

    monkeypatch.setattr(trainer_module, "autocast", Tracing)

    original_model_forward = RecordingModel.forward

    def tracing_forward(self, *args, **kwargs):
        events.append("forward")
        return original_model_forward(self, *args, **kwargs)

    monkeypatch.setattr(RecordingModel, "forward", tracing_forward)
    seen = _capture_metric_inputs(monkeypatch)

    trainer = make_trainer([make_batch([0, 1])])

    def tracing_loss(batch, outputs, _original=None):
        events.append("loss")
        return _original(batch, outputs)

    real_loss = trainer.loss_fn
    trainer.loss_fn = lambda b, o: tracing_loss(b, o, _original=real_loss)
    trainer._validate(epoch=1)
    events.append("metrics" if "predictions" in seen else "no-metrics")

    assert events == [
        "autocast(enabled=False)",
        "enter",
        "forward",
        "loss",
        "exit",
        "metrics",
    ], events


# ---------------------------------------------------------------------------
# 8. A malformed truth never reaches metric aggregation
# ---------------------------------------------------------------------------
def test_a_malformed_truth_raises_before_any_metric_is_recorded(monkeypatch):
    """The claim `test_trainer_truth_invariant.py` could only make at source level.

    That file establishes refusal *at the loss* in isolation. This drives the
    real `_validate` and asserts the consequence: the run dies, no metric is
    computed, no history is appended, and `on_validation_end` never fires — so a
    malformed truth cannot become a `val_mrr` of 0.0.
    """
    seen = _capture_metric_inputs(monkeypatch)
    recorder = RecordingCallback()
    trainer = make_trainer([make_batch([0, N_DISEASES])], callbacks=[recorder])

    with pytest.raises((IndexError, RuntimeError)):
        trainer._validate(epoch=1)

    assert seen == {}, "no metric aggregation was reached"
    assert recorder.events == ["validation_begin"], "end never fired"
    assert trainer.state.val_metric_history == []
    assert trainer.state.best_metric is None


def test_the_two_malformed_id_paths_refuse_at_different_places(monkeypatch):
    """Found by mutation, and pinned because 1c could silently remove one.

    `_compute_model_outputs` refuses a malformed truth from **two independent
    places**, and which one fires depends on the sign:

      - `id >= n_rows` — the gather `disease_emb[disease_ids]` raises `IndexError`
        on its own, before the loss is reached;
      - `id == -1` — the gather **wraps** to the last row and does not raise, so
        only `DiagnosisLoss` refuses.

    A mutation that clamped the *targets* but left the gather unclamped still
    passed every other test in this file, because the gather was doing the work.
    An extraction that clamps or reorders the gather would remove one guard while
    the outcome tests keep passing on the other — until someone also touches the
    loss, at which point both are gone at once.
    """
    embeddings = {
        "disease": torch.randn(N_DISEASES, HIDDEN),
        "phenotype": torch.randn(N_PHENOTYPES, HIDDEN),
    }

    with pytest.raises(IndexError):
        Trainer._compute_model_outputs(
            None, embeddings, make_batch([0, N_DISEASES])["batch"], {}, {}
        )

    # `-1` gets through the gather by wrapping, and the outputs look ordinary.
    outputs = Trainer._compute_model_outputs(
        None, embeddings, make_batch([0, -1])["batch"], {}, {}
    )
    assert outputs["diagnosis_targets"].tolist() == [0, -1], "unclamped, deliberately"
    assert torch.equal(
        outputs["disease_embeddings"][1], embeddings["disease"][-1]
    ), "the gather wrapped rather than refusing"

    # Only the loss refuses it, which is why the loss must stay on this path.
    from src.training.loss_functions import LossConfig, MultiTaskLoss

    with pytest.raises((IndexError, RuntimeError)):
        MultiTaskLoss(LossConfig())(make_batch([0, -1])["batch"], outputs)
