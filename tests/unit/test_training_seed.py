"""
Training's seed fixes the initial weights, and the stream training draws from.
=============================================================================
Contract M3a: the seed is applied before the data loaders and the model are
built. It used to be applied first in `Trainer.__init__`, after both, so the
model's initial weights came from whatever state the process's random
generators were in, and two runs with the same `--seed` started from different
weights. The Trainer still applies the seed itself, so the stream training
draws from (negatives, shuffling, dropout) still starts from the seed.

Shown here at the real entry point (`train`, with `Trainer.train` replaced by a
recorder), with the process's generators deliberately left in a different
state before each run:

- the same seed gives the same initial weights;
- a different seed gives different ones;
- when training starts, all three generators are in the state the seed gives,
  so the first batch training draws is the same for the same seed.

What it does not show, and does not promise: bit-identical training on a GPU,
or that a resumed run restores the random state it stopped in.

Module: tests/unit/test_training_seed.py
"""
from __future__ import annotations

import json
import random

import pytest

torch = pytest.importorskip("torch")

from tests.fixtures.generated_workspace import write_generated_workspace  # noqa: E402
from tests.fixtures.synthetic_workspace import N_GENES, _graph  # noqa: E402

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
        batch = next(iter(self.train_dataloader))["batch"]
        state["first_batch"] = {
            "patient_ids": list(batch["patient_ids"]),
            "negatives": batch["negative_disease_ids"].clone(),
        }
        runs.append(state)
        return {"val_loss": 1.0}

    monkeypatch.setattr(trainer_module.Trainer, "train", train)
    return runs


@pytest.fixture
def workspace(tmp_path):
    """A generated workspace whose every sample id is inside its graph, so the
    first training batch can be drawn (the generated fixture's default profiles
    name phenotypes outside the synthetic graph)."""
    n_phenotypes, n_diseases = 6, 6
    data_dir = tmp_path / "ws"
    data_dir.mkdir()
    x_dict, edge_index_dict, num_nodes = _graph(n_phenotypes, n_diseases)
    torch.save(x_dict, data_dir / "node_features.pt")
    torch.save(edge_index_dict, data_dir / "edge_indices.pt")
    (data_dir / "num_nodes.json").write_text(json.dumps(num_nodes))
    (data_dir / "kg.json").write_text(json.dumps({"synthetic": True}))
    profiles = {
        d: {"phenotype_ids": [d % n_phenotypes, (d + 1) % n_phenotypes],
            "gene_ids": [d % N_GENES]}
        for d in range(n_diseases)
    }

    def samples(split, ids):
        return [
            {"patient_id": f"{split}-{d}-{i}", "disease_id": d,
             "phenotype_ids": list(profiles[d]["phenotype_ids"]),
             "gene_ids": list(profiles[d]["gene_ids"]),
             "candidate_disease_ids": sorted(ids)}
            for d in ids for i in range(3)
        ]

    train_ids, val_ids = [0, 1, 2, 3], [4, 5]
    write_generated_workspace(
        data_dir, train_ids=train_ids, val_ids=val_ids, profiles=profiles,
        train_samples=samples("train", train_ids), val_samples=samples("val", val_ids),
    )
    return data_dir


def _run(train_model, workspace, run, seed, disturbance):
    """One training run, after leaving the process's generators somewhere else."""
    import numpy as np

    random.seed(disturbance)
    np.random.seed(disturbance)
    torch.manual_seed(disturbance)
    torch.rand(disturbance % 7 + 1)

    train_model.train(train_model.TrainConfig(
        data_dir=str(workspace), run_dir=str(run), device="cpu", hidden_dim=16,
        num_layers=1, num_heads=2, num_workers=0, use_amp=False, seed=seed,
    ))


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
    assert first["first_batch"]["patient_ids"] == second["first_batch"]["patient_ids"]
    assert torch.equal(first["first_batch"]["negatives"], second["first_batch"]["negatives"])
