"""Trainer validation reports the ranking the served primitives give.

`Trainer._run_evaluation_pass` ranks every candidate disease of a batch by the
cosine between the masked mean of the patient's phenotype embeddings and each
disease embedding, keeps the first twenty columns of the **descending** ranking,
and computes `mrr` on those rows (`src/training/trainer.py:684-689`, `:801-805`,
`:836-838`). **This file checks those values, not their shape**: the shape is
`test_trainer_validation_characterization.py`'s.

The expected rows are computed from the model's own embeddings with the served
primitives `masked_mean_pool` and `cosine_score_matrix`
(`src/inference/scoring.py`), never with Trainer code. The two are separately
written, and the `layers` and `scorer-independence` contracts in
`.import-linter.ini` keep them so; a comparison with itself would pass on any
defect.

Every cohort is a real `DiagnosisDataLoader` over `build_workspace`, scored by a
model built with the production `build_shepherd_model`, and materialised once.
Each test asserts inline the precondition that gives it something to catch:

  - rows wider than twenty, or a window keeping the wrong twenty columns would
    pass;
  - padded phenotype positions that would change the ranking, or a dropped mask
    would pass;
  - disease embeddings of unequal norm that a raw dot product would reorder, or
    a dropped normalisation would pass. The fixture checkpoint's final
    `LayerNorm` is untrained and gives every row the norm `sqrt(hidden_dim)`, so
    that cohort sets the affine parameters a trained checkpoint would carry;
  - a candidate universe that is not the identity map, or a truth reported in
    the wrong space would pass.

Mutation-checked: a reversed sort, a plain mean over padded positions, a raw dot
product in place of the cosine, and `pred_indices[-20:]` each fail a test here.

Module: tests/unit/test_trainer_validation_values.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from src.inference.scoring import cosine_score_matrix, masked_mean_pool
from src.kg.data_loader import DataLoaderConfig, DiagnosisSample, create_diagnosis_dataloader
from src.kg.storage.file_storage import read_graph_artifacts, read_samples
from src.models.gnn.shepherd_gnn import build_shepherd_model
from src.training.callbacks import Callback
from src.training.trainer import Trainer, TrainerConfig
from tests.fixtures.synthetic_workspace import build_workspace

#: The Trainer's validation window, a literal at `trainer.py:689` and one here.
TOP_K = 20

#: Above `TOP_K`, so the window cuts every row.
WIDE_DISEASES = TOP_K + 5

#: Every hop's limit clears a gene's degree in the wide workspace (its phenotypes
#: plus its diseases), so the two-hop expansion reaches every disease whatever
#: the sampler draws.
WIDE_NEIGHBORS = [64, 64, 64]

#: Patients of unequal length in every batch of three (one, three and two
#: phenotypes, then one, three and one), so `diagnosis_collate_fn` pads each batch,
#: and truths at ids 18 and above, which a universe of at most eighteen seeded
#: diseases cannot reach from 0.
UNEVEN_SAMPLES = [
    ("U00", [2], 24),
    ("U01", [0, 1, 2], 19),
    ("U02", [1, 0], 21),
    ("U03", [0], 18),
    ("U04", [2, 1, 0], 23),
    ("U05", [1], 20),
]


@dataclass
class Cohort:
    """A model, the batches drawn once, and the samples in loader order."""

    model: Any
    batches: list[dict[str, Any]]
    samples: list[Any]


def build_cohort(data_dir, checkpoint, samples, num_neighbors) -> Cohort:
    graph_data = read_graph_artifacts(data_dir).graph_data
    model = build_shepherd_model(checkpoint, graph_data, device=torch.device("cpu"))
    loader = create_diagnosis_dataloader(
        samples=samples,
        graph_data=graph_data,
        config=DataLoaderConfig(
            batch_size=3, num_workers=0, shuffle=False, num_neighbors=num_neighbors
        ),
    )
    return Cohort(model=model, batches=list(loader), samples=samples)


def load_checkpoint(path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def uneven_samples() -> list[DiagnosisSample]:
    return [
        DiagnosisSample(patient_id=pid, phenotype_ids=phenotypes, disease_id=disease)
        for pid, phenotypes, disease in UNEVEN_SAMPLES
    ]


@pytest.fixture(scope="module")
def wide_workspace(tmp_path_factory):
    data_dir, checkpoint = build_workspace(
        tmp_path_factory.mktemp("trainer_values_wide"), n_diseases=WIDE_DISEASES
    )
    return data_dir, checkpoint


@pytest.fixture(scope="module")
def default_cohort(tmp_path_factory) -> Cohort:
    """Four diseases: every row is narrower than the window."""
    data_dir, checkpoint = build_workspace(tmp_path_factory.mktemp("trainer_values_default"))
    return build_cohort(
        data_dir, load_checkpoint(checkpoint), read_samples(data_dir, "test").samples,
        [15, 10, 5],
    )


@pytest.fixture(scope="module")
def wide_cohort(wide_workspace) -> Cohort:
    data_dir, checkpoint = wide_workspace
    return build_cohort(
        data_dir, load_checkpoint(checkpoint), read_samples(data_dir, "test").samples,
        WIDE_NEIGHBORS,
    )


@pytest.fixture(scope="module")
def padded_cohort(wide_workspace) -> Cohort:
    data_dir, checkpoint = wide_workspace
    return build_cohort(
        data_dir, load_checkpoint(checkpoint), uneven_samples(), WIDE_NEIGHBORS
    )


@pytest.fixture(scope="module")
def unequal_norm_cohort(wide_workspace) -> Cohort:
    """The wide cohort, with the final disease `LayerNorm` given the non-uniform
    affine parameters a trained checkpoint carries."""
    data_dir, checkpoint = wide_workspace
    cohort = build_cohort(
        data_dir, load_checkpoint(checkpoint), read_samples(data_dir, "test").samples,
        WIDE_NEIGHBORS,
    )
    norm = cohort.model.final_norms["disease"]
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.25, 4.0, norm.weight.numel()))
        norm.bias.copy_(torch.linspace(-1.0, 1.0, norm.bias.numel()))
    return cohort


@pytest.fixture(scope="module")
def remapped_cohort(wide_workspace) -> Cohort:
    """No expansion past the first hop, so the candidates are the seeded diseases:
    the truths and their negatives, at most eighteen of twenty-five."""
    data_dir, checkpoint = wide_workspace
    return build_cohort(data_dir, load_checkpoint(checkpoint), uneven_samples(), [64, 0])


def make_trainer(cohort: Cohort) -> Trainer:
    """A CPU trainer over the cohort's model, in eval mode.

    `callbacks` is non-empty on purpose: `Trainer.__init__` does
    `callbacks or self._create_default_callbacks()`, so an empty list would
    install checkpointing and early stopping, which write files.
    """
    trainer = Trainer(
        model=cohort.model,
        train_dataloader=[],
        val_dataloader=cohort.batches,
        config=TrainerConfig(device="cpu", use_amp=True, scheduler_type="none", seed=None),
        callbacks=[Callback()],
    )
    trainer.model.eval()
    return trainer


def embed(cohort: Cohort, batch_data: dict[str, Any]) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        return cohort.model(
            batch_data["subgraph_x_dict"], batch_data["subgraph_edge_index_dict"]
        )


def pooled_patients(embeddings, batch) -> torch.Tensor:
    """`(B, hidden)` masked means from the served primitive. A padded position
    gathers an arbitrary row, which the mask then discards."""
    ids, mask = batch["phenotype_ids"], batch["phenotype_mask"]
    gathered = embeddings["phenotype"][torch.where(mask, ids, torch.zeros_like(ids))]
    return masked_mean_pool(gathered, mask)


def patient_scores(embeddings, batch) -> torch.Tensor:
    """`(B, D)` cosines from the served primitives."""
    return cosine_score_matrix(pooled_patients(embeddings, batch), embeddings["disease"])


def top_columns(row: list[float]) -> list[int]:
    return sorted(range(len(row)), key=row.__getitem__, reverse=True)[:TOP_K]


def served_expectation(cohort: Cohort):
    """Rows, truths and MRR the served primitives give, in loader order.

    The truth is the position of the sample's own disease in the batch's
    candidate universe, read from the sample and `original_indices`, not from the
    batch's remapped `disease_ids`.
    """
    rows: list[list[str]] = []
    truths: list[str] = []
    reciprocal: list[float] = []
    samples = iter(cohort.samples)
    for batch_data in cohort.batches:
        universe = batch_data["original_indices"]["disease"].tolist()
        scores = patient_scores(embed(cohort, batch_data), batch_data["batch"])
        patient_ids = batch_data["batch"]["patient_ids"]
        for patient_id, row in zip(patient_ids, scores.tolist(), strict=True):
            sample = next(samples)
            assert patient_id == sample.patient_id, "batches follow the sample order"
            assert len(set(row)) == len(row), "a tie would leave the order to the sort"
            truth = universe.index(sample.disease_id)
            rank = 1 + sum(score > row[truth] for score in row)
            rows.append([str(column) for column in top_columns(row)])
            truths.append(str(truth))
            reciprocal.append(1.0 / rank if rank <= TOP_K else 0.0)
    return rows, truths, sum(reciprocal) / len(reciprocal)


def assert_validation_matches_the_served_ranking(cohort: Cohort) -> None:
    rows, truths, mrr = served_expectation(cohort)

    result = make_trainer(cohort)._run_evaluation_pass(cohort.batches)

    assert result.num_batches == len(cohort.batches)
    assert result.predictions == rows
    assert result.ground_truths == truths
    assert result.ranking_metrics["mrr"] == pytest.approx(mrr)


def test_validation_ranks_like_the_served_scorer_on_the_default_workspace(default_cohort):
    assert_validation_matches_the_served_ranking(default_cohort)


def test_validation_keeps_the_top_twenty_of_rankings_wider_than_twenty(wide_cohort):
    """The default workspace cannot catch a window that keeps twenty columns
    other than the first twenty; this cohort's width is the point of the test, so
    it is asserted rather than assumed."""
    universes = [b["original_indices"]["disease"].numel() for b in wide_cohort.batches]
    assert min(universes) > TOP_K, f"candidate universes are {universes}"

    assert_validation_matches_the_served_ranking(wide_cohort)

    _, _, mrr = served_expectation(wide_cohort)
    assert make_trainer(wide_cohort)._validate(epoch=1)["val_mrr"] == pytest.approx(mrr)


def test_padded_phenotype_positions_do_not_reach_the_validation_ranking(padded_cohort):
    """Whichever phenotype row a padded position gathered, an unmasked mean over
    it would move some row of the ranking."""
    for batch_data in padded_cohort.batches:
        assert not bool(batch_data["batch"]["phenotype_mask"].all()), "every batch is padded"
    embeddings = [embed(padded_cohort, b) for b in padded_cohort.batches]
    n_phenotype_rows = embeddings[0]["phenotype"].size(0)
    for filler in range(n_phenotype_rows):
        moved = False
        for batch_data, emb in zip(padded_cohort.batches, embeddings, strict=True):
            ids = batch_data["batch"]["phenotype_ids"]
            mask = batch_data["batch"]["phenotype_mask"]
            unmasked = emb["phenotype"][torch.where(mask, ids, torch.full_like(ids, filler))]
            leaked = cosine_score_matrix(unmasked.mean(dim=1), emb["disease"])
            served = patient_scores(emb, batch_data["batch"])
            moved |= any(
                top_columns(a) != top_columns(b)
                for a, b in zip(leaked.tolist(), served.tolist(), strict=True)
            )
        assert moved, f"padding that gathers phenotype row {filler} changes no ranking"

    assert_validation_matches_the_served_ranking(padded_cohort)


def test_validation_ranks_by_cosine_when_disease_norms_differ(unequal_norm_cohort):
    """A raw dot product would reorder some row here; on the untrained fixture
    checkpoint every disease row has the same norm and it could not."""
    reordered = False
    for batch_data in unequal_norm_cohort.batches:
        emb = embed(unequal_norm_cohort, batch_data)
        norms = emb["disease"].norm(dim=-1)
        assert float(norms.max() / norms.min()) > 1.5, "disease norms are unequal"
        batch = batch_data["batch"]
        raw = pooled_patients(emb, batch) @ emb["disease"].t()
        served = patient_scores(emb, batch)
        reordered |= any(
            top_columns(a) != top_columns(b)
            for a, b in zip(raw.tolist(), served.tolist(), strict=True)
        )
    assert reordered, "a raw dot product would rank these rows as the cosine does"

    assert_validation_matches_the_served_ranking(unequal_norm_cohort)


def test_validation_truths_index_the_batch_candidates_where_local_is_not_global(remapped_cohort):
    """The Trainer reads no `original_indices`: its truths and rows are positions
    in the batch's own candidate universe. Under an identity map a truth in the
    wrong space would still look right, so the map is asserted not to be one."""
    for batch_data in remapped_cohort.batches:
        universe = batch_data["original_indices"]["disease"].tolist()
        assert universe != list(range(len(universe))), universe

    assert_validation_matches_the_served_ranking(remapped_cohort)
