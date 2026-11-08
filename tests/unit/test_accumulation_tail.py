"""The gradient-accumulation tail, and progress counted three ways (contract M3a-1).

With accumulation, the batches after an epoch's last full group used to leave
their gradients in `.grad` for the next epoch's first update to absorb (the
last epoch's were dropped), while the schedule's step count assumed a fraction
of an update. The tail is now one more update at the epoch's end, rescaled to
the mean of its batches, and an epoch takes `ceil(batches / accumulation)`
updates, which the schedule counts. Optimizer attempts, successful updates and
scheduler steps are counted apart: they differ when AMP's scaler skips an
update.

A one-parameter model and plain SGD make every update computable by hand. The
Trainer is driven through a task module (README §3.1), its one training path:
the task's step returns the loss `w * v`, so each update's gradient is visible.

Module: tests/unit/test_accumulation_tail.py
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))


class _Task:
    """The smallest task module the Trainer drives: a step's loss is
    `loss(w, v)`, by default `w * v`, so d(loss)/dw = v; nothing to validate.
    Like every task it names its selection metric, which it never reports:
    these tests select nothing."""

    selection_metric = "val_toy_metric"

    def __init__(self, loss=lambda w, v: w * v):
        self.loss = loss

    def training_step(self, model, batch, device):
        from src.training.lp_task import StepResult

        return StepResult(self.loss(model.w.sum(), batch["v"].sum()))

    def validate(self, model, device):
        return {}


def _batches(values):
    return [{"v": torch.tensor([float(v)])} for v in values]


def _trainer(tmp_path, values, *, accumulation, scheduler_type="none", num_epochs=1,
             max_grad_norm=0.0, task=None):
    from src.training.trainer import Trainer, TrainerConfig

    model = _Toy()
    return Trainer(
        model=model,
        train_dataloader=_batches(values),
        optimizer=torch.optim.SGD(model.parameters(), lr=1.0),
        config=TrainerConfig(
            device="cpu", use_amp=False, scheduler_type=scheduler_type, warmup_steps=0,
            gradient_accumulation_steps=accumulation, max_grad_norm=max_grad_norm,
            num_epochs=num_epochs, checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "logs"),
        ),
        task=task or _Task(),
    )


def test_a_trainer_without_a_task_is_refused(tmp_path):
    """TM21: the task module is the Trainer's one training path (no default
    loss is built for a Trainer without one): a Trainer asked for without it,
    or with `None`, is refused before anything is built."""
    from src.training.trainer import Trainer, TrainerConfig

    config = TrainerConfig(device="cpu", use_amp=False, scheduler_type="none",
                           checkpoint_dir=str(tmp_path / "ckpt"), log_dir=str(tmp_path / "logs"))
    with pytest.raises(TypeError):
        Trainer(model=_Toy(), train_dataloader=_batches([1]), config=config)
    with pytest.raises(TypeError, match="task module"):
        Trainer(model=_Toy(), train_dataloader=_batches([1]), config=config, task=None)
    assert not (tmp_path / "ckpt").exists()


def test_the_tail_is_one_more_update_applying_the_mean_of_its_batches(tmp_path):
    trainer = _trainer(tmp_path, [1, 2, 3], accumulation=2)

    trainer._train_epoch(0)

    # Group (1, 2): mean gradient 1.5, w = -1.5. Tail (3): mean gradient 3,
    # w = -4.5. Before the tail was applied, w stayed at -1.5 with 1.5 left
    # in `.grad` for the next epoch.
    assert trainer.model.w.item() == pytest.approx(-4.5)
    assert trainer.state.global_step == trainer.state.optimizer_updates == 2
    assert trainer.model.w.grad is None or float(trainer.model.w.grad.abs().sum()) == 0.0


def test_no_gradient_crosses_an_epoch_boundary(tmp_path):
    trainer = _trainer(tmp_path, [1, 2, 3], accumulation=2, num_epochs=2)

    trainer._train_epoch(0)
    after_first = trainer.model.w.item()
    trainer._train_epoch(1)

    # The second epoch repeats the first's updates exactly: nothing carried.
    assert trainer.model.w.item() == pytest.approx(2 * after_first)


def test_the_update_clips_the_gradient_itself(tmp_path):
    """K7 (X12) removed the GradientClipping callback, which only logged a norm.
    The clip is the Trainer's own, in `_optimizer_update`: a gradient of 5
    under max_grad_norm 1 moves w by 1, not 5."""
    trainer = _trainer(tmp_path, [5], accumulation=1, max_grad_norm=1.0)

    trainer._train_epoch(0)

    assert trainer.model.w.item() == pytest.approx(-1.0)


def test_an_accumulation_that_divides_is_unchanged(tmp_path):
    trainer = _trainer(tmp_path, [1, 2, 3, 4], accumulation=2)

    trainer._train_epoch(0)

    assert trainer.model.w.item() == pytest.approx(-(1.5 + 3.5))
    assert trainer.state.optimizer_updates == 2


@pytest.mark.parametrize("batches,accumulation,epochs,expected", [
    (3, 2, 2, 4), (4, 2, 1, 2), (5, 1, 3, 15), (7, 3, 2, 6),
])
def test_the_schedule_counts_the_updates_taken(tmp_path, batches, accumulation, epochs, expected):
    trainer = _trainer(tmp_path, range(1, batches + 1), accumulation=accumulation,
                       scheduler_type="linear", num_epochs=epochs)

    assert trainer.schedule_record["total_steps"] == expected
    for epoch in range(epochs):
        trainer._train_epoch(epoch)
    assert trainer.state.optimizer_updates == trainer.state.scheduler_steps == expected


class _SkippingScaler:
    """A GradScaler that skips every update, as AMP does on non-finite
    gradients: `step` does not run the optimizer and `update` lowers the scale."""

    def __init__(self):
        self._scale = 1024.0

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        pass

    def update(self):
        self._scale /= 2

    def get_scale(self):
        return self._scale


def test_attempts_updates_and_scheduler_steps_are_counted_apart(tmp_path):
    trainer = _trainer(tmp_path, [1, 2, 3], accumulation=2, scheduler_type="linear")
    trainer.scaler = _SkippingScaler()

    trainer._train_epoch(0)

    assert trainer.state.global_step == 2
    assert trainer.state.optimizer_updates == 0
    assert trainer.state.scheduler_steps == 0
    assert trainer.model.w.item() == 0.0


@pytest.mark.parametrize("accumulation", [1, 2, 3])
def test_train_loss_is_the_mean_loss_per_batch(tmp_path, accumulation):
    """Batches with losses 1, 2 and 3 report 2.0 whatever the accumulation: one
    unit and one denominator, kept apart from the number of updates. With an
    accumulation of 2 this was 3.0 (a group sum averaged with a tail mean)."""
    trainer = _trainer(tmp_path, [1, 2, 3], accumulation=accumulation,
                       task=_Task(lambda w, v: w * 0 + v))

    metrics = trainer._train_epoch(0)

    assert metrics["train_loss"] == pytest.approx(2.0)


def test_the_task_step_runs_inside_autocast_and_validation_outside(tmp_path, monkeypatch):
    """AMP's placement on the task path: each step runs under autocast with
    the trainer's own setting and dtype; validation runs outside it. On CPU
    `_setup_amp` turns AMP off, so the setting is put on the built trainer
    (CPU autocast takes bfloat16) and autocast is traced."""
    import src.training.trainer as trainer_module

    events = []
    real = trainer_module.autocast

    class _Traced:
        def __init__(self, *args, **kwargs):
            self._inner = real(*args, **kwargs)
            events.append(("autocast", kwargs.get("enabled"), kwargs.get("dtype")))

        def __enter__(self):
            events.append("enter")
            return self._inner.__enter__()

        def __exit__(self, *exc):
            events.append("exit")
            return self._inner.__exit__(*exc)

    class _Recording(_Task):
        def training_step(self, model, batch, device):
            events.append("step")
            return super().training_step(model, batch, device)

        def validate(self, model, device):
            events.append("validate")
            return {}

    monkeypatch.setattr(trainer_module, "autocast", _Traced)
    trainer = _trainer(tmp_path, [1], accumulation=1, task=_Recording())
    trainer.use_amp, trainer.amp_dtype = True, torch.bfloat16

    trainer._train_epoch(0)
    trainer._validate(0)

    assert events == [("autocast", True, torch.bfloat16), "enter", "step", "exit", "validate"]
