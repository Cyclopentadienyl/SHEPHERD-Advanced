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

**Each contract group was mutation-checked against a representative defect** —
not every assertion independently, and the difference matters: a characterization
test that passes on changed code is worse than none, because it certifies a change
it did not examine. The mutations run were: dropping the `val_` prefix,
truncating at 10 instead of 20, replacing the mean loss with the last batch's,
moving the loss outside `autocast`, moving metric aggregation *inside* it,
clamping only `diagnosis_targets`, and restoring the whole pre-fix clamp. Each
failed the group that names it.

Clamping only `diagnosis_targets` is the one worth recording. It first passed
everything, because `disease_emb[disease_ids]` raises on `id >= n_rows` before
the loss is reached. Driving the **complete entry points** with `-1` as well
catches it: `-1` wraps through the gather and only the loss refuses, so a clamped
target lets the run finish and reach metric aggregation. That is why the
malformed-truth tests are parameterized over both signs rather than over the
convenient one.

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


#: The two public entry points over the pass item 1c extracts. Behaviour that
#: belongs to the *shared* pass is characterized through **both**; behaviour that
#: is caller-specific — prefixes, callbacks, state — is tested separately below.
#: Running the whole file twice would only make the caller contracts harder to
#: see.
ENTRY_POINTS = ("validate", "evaluate")
LOSS_KEY = {"validate": "val_loss", "evaluate": "loss"}


def run_entry_point(trainer: Trainer, entry: str, batches=None) -> Dict[str, float]:
    """Drive one entry point over the trainer's val batches."""
    if entry == "validate":
        return trainer._validate(epoch=1)
    return trainer.evaluate(batches)


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
@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_loss_is_the_mean_of_the_per_batch_losses(entry):
    """Computed independently, not read back from the same accumulator.

    Shared-pass behaviour, so both entry points are driven: 1c extracts one
    aggregation and both callers must keep getting the mean.
    """
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

    metrics = run_entry_point(trainer, entry)

    assert metrics[LOSS_KEY[entry]] == pytest.approx(sum(expected) / len(expected))
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
@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("n_batches", [1, 2, 3])
def test_one_forward_per_batch(entry, n_batches):
    trainer = make_trainer([make_batch([0, 1]) for _ in range(n_batches)])
    run_entry_point(trainer, entry)

    assert trainer.model.forward_calls == n_batches


def test_evaluate_prefers_its_argument_over_the_val_dataloader():
    trainer = make_trainer([make_batch([0, 1]) for _ in range(5)])
    trainer.evaluate([make_batch([0, 1]) for _ in range(2)])

    assert trainer.model.forward_calls == 2


def test_evaluate_with_no_dataloader_at_all_returns_empty():
    trainer = make_trainer(None)

    assert trainer.evaluate() == {}
    assert trainer.model.forward_calls == 0


def test_an_explicitly_empty_evaluate_argument_falls_back_to_the_val_dataloader():
    """**Frozen as observed, not endorsed.** `evaluate` selects its dataloader
    with `test_dataloader or self.val_dataloader` (`trainer.py:813`), so an
    explicitly supplied `[]` is falsy and silently becomes the validation set —
    a caller asking to evaluate nothing gets a full validation pass instead.

    1c must not quietly change this to an `is None` contract. That would be a
    defensible fix, but it is a **behaviour change** and has to be made by
    editing this test rather than by an extraction nobody re-read.
    """
    trainer = make_trainer([make_batch([0, 1]), make_batch([2, 3])])

    metrics = trainer.evaluate([])

    assert trainer.model.forward_calls == 2, "the two val batches ran"
    assert "loss" in metrics, "and produced a full result"


# ---------------------------------------------------------------------------
# 6. The local top-20 rows and truth ids handed to the metric
# ---------------------------------------------------------------------------
def _capture_metric_inputs(monkeypatch, events: List[str] | None = None):
    """Record what `RankingMetrics.compute_all` is called with, and **when**.

    The prediction lists are local to the entry points, and they are the artifact
    1d will compare against Mode A's `legacy_top_k_local`. Capturing the call is
    the only way to freeze their shape without changing the code under test.

    When `events` is supplied the call appends `"metrics"` **from inside**
    `compute_all`. Appending it after the entry point returns would prove only
    that aggregation happened at some point, which is true however the code is
    ordered — it would not show that aggregation ran after `autocast` exited.
    """
    import src.training.trainer as trainer_module

    seen: Dict[str, Any] = {}
    original = trainer_module.RankingMetrics

    class Capturing(original):  # type: ignore[misc,valid-type]
        def compute_all(self, predictions, ground_truths, k_values=None):
            if events is not None:
                events.append("metrics")
            seen["predictions"] = predictions
            seen["ground_truths"] = ground_truths
            return super().compute_all(predictions, ground_truths, k_values)

    monkeypatch.setattr(trainer_module, "RankingMetrics", Capturing)
    return seen


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_predictions_are_local_column_indices_as_strings_truncated_to_twenty(entry, monkeypatch):
    seen = _capture_metric_inputs(monkeypatch)
    trainer = make_trainer([make_batch([4, 9])])
    run_entry_point(trainer, entry)

    predictions = seen["predictions"]
    assert len(predictions) == 2, "one row per sample"
    for row in predictions:
        assert len(row) == 20, f"truncated to 20 of {N_DISEASES}"
        assert all(isinstance(v, str) for v in row)
        assert {int(v) for v in row} <= set(range(N_DISEASES)), "local column indices"
        assert len(set(row)) == 20, "a ranking has no repeats"


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_ground_truths_are_the_stringified_local_disease_ids(entry, monkeypatch):
    seen = _capture_metric_inputs(monkeypatch)
    trainer = make_trainer([make_batch([4, 9])])
    run_entry_point(trainer, entry)

    assert seen["ground_truths"] == ["4", "9"]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_rows_accumulate_across_batches_in_dataloader_order(entry, monkeypatch):
    seen = _capture_metric_inputs(monkeypatch)
    trainer = make_trainer([make_batch([4, 9]), make_batch([1, 2])])
    run_entry_point(trainer, entry)

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


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_the_forward_and_the_loss_are_inside_autocast_and_the_metrics_are_not(entry, monkeypatch):
    """Placement, not just settings — 1c must not move either boundary.

    `metrics` is appended from **inside** the capturing `compute_all`, so the
    assertion sees where aggregation really sits relative to `exit`.
    """
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
    _capture_metric_inputs(monkeypatch, events)

    trainer = make_trainer([make_batch([0, 1])])
    real_loss = trainer.loss_fn

    def tracing_loss(batch, outputs):
        events.append("loss")
        return real_loss(batch, outputs)

    trainer.loss_fn = tracing_loss
    run_entry_point(trainer, entry)

    assert events == [
        "autocast(enabled=False)",
        "enter",
        "forward",
        "loss",
        "exit",
        "metrics",
    ], events


# ---------------------------------------------------------------------------
# 8. A malformed truth never reaches metric aggregation — both signs, both callers
# ---------------------------------------------------------------------------
#: The two ways a remapped truth can be malformed. They refuse in **different
#: places**, which is why testing only one is not enough:
#:
#:   `n_rows` — `disease_emb[disease_ids]` raises at the gather, before the loss;
#:   `-1`     — the gather wraps to the last row and only `DiagnosisLoss` refuses.
#:
#: A mutation that clamps `diagnosis_targets` but leaves the gather alone passes
#: every `n_rows` test, because the gather is doing the work. It fails on `-1`,
#: where the clamped target lets the run finish and reach aggregation.
MALFORMED_IDS = (-1, N_DISEASES)


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("bad_id", MALFORMED_IDS)
def test_a_malformed_truth_raises_before_any_metric_is_recorded(entry, bad_id, monkeypatch):
    """The claim `test_trainer_truth_invariant.py` could only make at source level.

    That file establishes refusal *at the loss* in isolation. This drives the real
    entry points and asserts the consequence: the run dies, no metric is computed,
    and for `_validate` no history is appended and `on_validation_end` never fires
    — so a malformed truth cannot become a `val_mrr` of 0.0.
    """
    seen = _capture_metric_inputs(monkeypatch)
    recorder = RecordingCallback()
    trainer = make_trainer([make_batch([0, bad_id])], callbacks=[recorder])

    with pytest.raises((IndexError, RuntimeError)):
        run_entry_point(trainer, entry)

    assert seen == {}, "no metric aggregation was reached"
    assert trainer.state.val_metric_history == []
    assert trainer.state.best_metric is None

    if entry == "validate":
        assert recorder.events == ["validation_begin"], "end never fired"
    else:
        assert recorder.events == [], "evaluate drives no validation hooks"
