"""
Training's seed fixes the initial weights, and the stream training draws from.
=============================================================================
Contract M3a: the seed is applied before the data loaders and the model are
built. It used to be applied first in `Trainer.__init__`, after both, so the
model's initial weights came from whatever state the process's random
generators were in, and two runs with the same `--seed` started from different
weights. The Trainer still applies the seed itself, so the stream training
draws from (negatives, shuffling, dropout) still starts from the seed.

Shown here at the real entry point (`train`, a base run of the demo's small
spec, with `Trainer.train` replaced by a recorder), with the process's
generators deliberately left in a different state before each run:

- the same seed gives the same initial weights;
- a different seed gives different ones;
- when training starts, all three generators are in the state the seed gives,
  so the first batch training draws is the same for the same seed.

What it does not show, and does not promise: bit-identical training on a GPU,
or that a resumed run restores the random state it stopped in.

Module: tests/unit/test_training_seed.py
"""
from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from tests.fixtures.base_workspace import write_base_workspace  # noqa: E402

SEED = 1234


@pytest.fixture
def train_model():
    import scripts.train_model as module

    return module


@pytest.fixture
def started(monkeypatch):
    """`Trainer.train` replaced by a recorder of what training would start from:
    the initial weights, the generators' state, and the first training batch."""
    import numpy as np

    from src.training import trainer as trainer_module

    runs = []

    def train(self):
        state = {
            "weights": {k: v.detach().clone() for k, v in self.model.state_dict().items()},
            "torch_seed": torch.initial_seed(),
            "numpy_seed": int(np.random.get_state()[1][0]),
            "python_state": random.getstate(),
        }
        # A base step's batch: the subgraph's nodes, the positives drawn and
        # each one's corrupted tail.
        batch = next(iter(self.train_dataloader))
        state["first_batch"] = {
            "node_ids": batch.node_ids.clone(),
            "positives": torch.stack([batch.head, batch.relation, batch.tail]).clone(),
            "negatives": batch.negative.clone(),
        }
        runs.append(state)
        return {"val_loss": 1.0}

    monkeypatch.setattr(trainer_module.Trainer, "train", train)
    return runs


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """A workspace the base stage trains on (`write_base_workspace`); only read."""
    return write_base_workspace(tmp_path_factory.mktemp("base"))


def _run(train_model, workspace, run, seed, disturbance):
    """One base run at the demo's base sizes (`demo_task_training`: its small
    spec, batch size and fan-out), after leaving the process's generators
    somewhere else. `train` is told that exactly the fields set here are the
    operator's (contract TM1: the explicit set is stated, never inferred)."""
    import numpy as np

    from scripts.setup_demo import demo_task_training

    random.seed(disturbance)
    np.random.seed(disturbance)
    torch.manual_seed(disturbance)
    torch.rand(disturbance % 7 + 1)

    fields = {"data_dir": str(workspace), "run_dir": str(run), "learning_stage": "base",
              "num_epochs": 1, "device": "cpu", "seed": seed,
              **demo_task_training()["base"]}
    train_model.train(train_model.TrainConfig(**fields), set(fields))


def _same_weights(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_the_same_seed_gives_the_same_initial_weights(train_model, started, workspace, tmp_path):
    _run(train_model, workspace, tmp_path / "a", SEED, disturbance=11)
    _run(train_model, workspace, tmp_path / "b", SEED, disturbance=29)

    first, second = started
    assert _same_weights(first["weights"], second["weights"]), (
        "two runs with the same seed started from different weights"
    )


def test_a_different_seed_gives_different_initial_weights(
    train_model, started, workspace, tmp_path
):
    """The half that makes the test above mean something: weights that never
    depended on the seed would also be equal."""
    _run(train_model, workspace, tmp_path / "a", SEED, disturbance=11)
    _run(train_model, workspace, tmp_path / "b", SEED + 1, disturbance=11)

    first, second = started
    assert not _same_weights(first["weights"], second["weights"])


def test_training_starts_from_the_seed_and_draws_the_same_first_batch(
    train_model, started, workspace, tmp_path
):
    """Seeding before construction must not move the stream training draws
    from: when training starts, every generator is in the state the seed gives,
    whatever building the loaders and the model consumed."""
    _run(train_model, workspace, tmp_path / "a", SEED, disturbance=11)
    _run(train_model, workspace, tmp_path / "b", SEED, disturbance=29)

    for state in started:
        assert state["torch_seed"] == SEED
        assert state["numpy_seed"] == SEED
        assert state["python_state"] == random.Random(SEED).getstate()

    first, second = started
    for part, value in first["first_batch"].items():
        assert torch.equal(value, second["first_batch"][part]), part
