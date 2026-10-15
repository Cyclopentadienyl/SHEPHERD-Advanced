"""The gradient-accumulation tail, and progress counted three ways (contract M3a-1).

With accumulation, the batches after an epoch's last full group used to leave
their gradients in `.grad` for the next epoch's first update to absorb (the
last epoch's were dropped), while the schedule's step count assumed a fraction
of an update. The tail is now one more update at the epoch's end, rescaled to
the mean of its batches, and an epoch takes `ceil(batches / accumulation)`
updates, which the schedule counts. Optimizer attempts, successful updates and
scheduler steps are counted apart: they differ when AMP's scaler skips an
update.

A one-parameter model and plain SGD make every update computable by hand.

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

    def forward(self, x_dict, edge_index_dict):
        return {"w": self.w}


def _batches(values):
    return [{"batch": {"v": torch.tensor([float(v)])}, "subgraph_x_dict": {},
             "subgraph_edge_index_dict": {}} for v in values]


def _trainer(tmp_path, values, *, accumulation, scheduler_type="none", num_epochs=1):
    from src.training.trainer import Trainer, TrainerConfig

    model = _Toy()
    trainer = Trainer(
        model=model,
        train_dataloader=_batches(values),
        optimizer=torch.optim.SGD(model.parameters(), lr=1.0),
        config=TrainerConfig(
            device="cpu", use_amp=False, scheduler_type=scheduler_type, warmup_steps=0,
            gradient_accumulation_steps=accumulation, max_grad_norm=0.0,
            num_epochs=num_epochs, checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "logs"),
        ),
    )
    # loss = w * v, so d(loss)/dw = v: each update's gradient is visible.
    trainer._compute_model_outputs = lambda emb, batch, x, e: emb
    trainer.loss_fn = lambda batch, outputs: (outputs["w"].sum() * batch["v"].sum(), {})
    return trainer


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
