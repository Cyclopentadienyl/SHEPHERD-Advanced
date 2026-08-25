"""Backlog item 1d — the same-batch differential calibration, on a real cohort.

`src/evaluation/differential.py` claims that `Trainer._run_evaluation_pass` and
the Mode A harness compute the same thing. **This file exists to make that claim
falsifiable**, which needs two halves and is worthless with only the first:

  - that the two paths agree on a real workspace, sample for sample;
  - that the comparison would have **noticed** if they had not. Every contract
    group below is mutation-checked against a representative defect, because a
    calibration that passes on broken code certifies a comparison it did not make.

The workspace is `tests/fixtures/synthetic_workspace.py` — the same one the Mode A
tests use — driven through a real `DiagnosisDataLoader` and a model built by the
production `build_shepherd_model`. Nothing here is a stand-in for the pipeline:
the point of a differential test is that both sides see the objects production
would hand them.

**The batches are materialised once and both paths get that list.** That is the
whole design, and `_materialize_batches` is what stops it degrading into two
independent draws with no shared cohort — by freezing the stream rather than by
inspecting what the caller passed.

**AMP is off here because the device is CPU** (`trainer.py:380` resolves
`use_amp = config.use_amp and device.type == "cuda"`), so these tests measure the
bit-exact question. The CUDA question — whether autocast at `float16` reorders
anything near a tie — is item 7a's institutional run and cannot be asked here.
A test asserts the observed AMP state rather than leaving it implied.

Module: tests/unit/test_differential_calibration.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

torch = pytest.importorskip("torch")

from src.evaluation.differential import (  # noqa: E402
    DifferentialResult,
    compare_trainer_against_mode_a,
)
from src.evaluation.measurement import LEGACY_TRUNCATION_K, run_mode_a  # noqa: E402
from src.training.trainer import Trainer, TrainerConfig  # noqa: E402
from tests.fixtures.synthetic_workspace import build_workspace  # noqa: E402


@dataclass
class Cohort:
    """One materialised cohort plus everything both paths need to score it."""

    model: Any
    batches: List[Dict[str, Any]]
    manifest: Any


def build_cohort(
    root, *, n_diseases: int, n_phenotypes: int, n_samples: int,
    num_neighbors: List[int], batch_size: int,
) -> Cohort:
    """A real workspace, a production-built model, and the batches drawn once.

    Drawn **once** and reused: Mode A's candidate universe is the batch's
    subgraph, and the negative sampler draws from Python's `random`, so a second
    pass over the loader would build a different universe. Two paths compared on
    two universes would measure the sampler, not the scorer.

    The workspace's seed-independence is asserted here under **this** loader
    config, because stability is a property of the graph and the sampling limits
    together — see `synthetic_workspace._graph`.
    """
    import argparse

    from scripts.measure_scorer import build_manifest
    from src.kg.data_loader import DataLoaderConfig, create_diagnosis_dataloader
    from src.kg.storage.file_storage import read_graph_artifacts, read_samples
    from src.models.gnn.shepherd_gnn import build_shepherd_model
    from tests.fixtures.synthetic_workspace import assert_candidate_universe_is_stable

    data_dir, checkpoint_path = build_workspace(
        root, n_phenotypes=n_phenotypes, n_diseases=n_diseases, n_samples=n_samples,
    )
    device = torch.device("cpu")
    loader_config = DataLoaderConfig(
        batch_size=batch_size, num_workers=0, shuffle=False,
        num_neighbors=num_neighbors,
    )
    assert_candidate_universe_is_stable(data_dir, "test", config=loader_config)

    graph_data = read_graph_artifacts(data_dir)
    samples = read_samples(data_dir, "test")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_shepherd_model(checkpoint, graph_data, device=device)

    loader = create_diagnosis_dataloader(
        samples=samples, graph_data=graph_data, config=loader_config
    )
    batches = list(loader)
    args = argparse.Namespace(
        checkpoint=checkpoint_path, data_dir=data_dir, split="test",
        batch_size=batch_size, num_workers=0, seed=None,
    )
    manifest = build_manifest(args, graph_data, len(samples), device, loader_config)
    return Cohort(model=model, batches=batches, manifest=manifest)


@pytest.fixture(scope="module")
def cohort(tmp_path_factory) -> Cohort:
    """The default fixture: four diseases, so every ranking is **narrower** than
    `LEGACY_TRUNCATION_K` and the truncation is a no-op on both sides.

    That is a real production shape — a subgraph can be smaller than K — and it is
    the shape every other Mode A test runs on. It is not sufficient on its own,
    which is what `wide_cohort` is for.
    """
    return build_cohort(
        tmp_path_factory.mktemp("differential_narrow"),
        n_diseases=4, n_phenotypes=3, n_samples=6,
        num_neighbors=[15, 10, 5], batch_size=3,
    )


@pytest.fixture(scope="module")
def wide_cohort(tmp_path_factory) -> Cohort:
    """More diseases than `LEGACY_TRUNCATION_K`, so the truncation is real.

    **This exists because the narrow cohort cannot fail the truncation contract.**
    At four candidate columns a top-20 slice and no slice at all are the same list,
    so an agreement there says nothing about whether the two paths truncate to the
    same depth — and the trainer's hardcoded `[:20]` and Mode A's
    `LEGACY_TRUNCATION_K` are two separately written constants.

    `num_neighbors` is widened to match, and **every hop's limit** must clear the
    disease count, not just the first: a gene's out-degree in this fixture is the
    disease count, and the gene->disease expansion happens at hop 1. A first draft
    of this fixture used `[40, 20, 10]` and the stability check rejected it — at 25
    diseases the hop-1 limit of 20 made the expansion a real draw and the candidate
    universe moved with the seed. That check runs under this config, not the
    default one, which is why it caught it.
    """
    return build_cohort(
        tmp_path_factory.mktemp("differential_wide"),
        n_diseases=LEGACY_TRUNCATION_K + 5, n_phenotypes=3, n_samples=6,
        num_neighbors=[64, 64, 64], batch_size=3,
    )


def make_trainer(cohort: Cohort) -> Trainer:
    """A CPU trainer wrapping the cohort's model.

    `callbacks` is non-empty on purpose: `Trainer.__init__` does
    `callbacks or self._create_default_callbacks()`, so an empty list would
    install checkpointing and early stopping, which write files.
    """
    from src.training.callbacks import Callback

    return Trainer(
        model=cohort.model,
        train_dataloader=[],
        val_dataloader=cohort.batches,
        config=TrainerConfig(device="cpu", use_amp=True, scheduler_type="none", seed=0),
        callbacks=[Callback()],
    )


def run(cohort: Cohort, **kwargs) -> DifferentialResult:
    return compare_trainer_against_mode_a(
        make_trainer(cohort), cohort.batches, cohort.manifest,
        device=torch.device("cpu"), **kwargs,
    )


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------
def test_the_two_paths_agree_sample_for_sample(cohort):
    """The calibration itself. Everything else in this file exists to give this
    line meaning."""
    result = run(cohort)

    assert result.disagreements == []
    assert result.agreed is True
    assert result.n_disagreements_by_kind == {}


def test_the_cohort_is_the_whole_cohort(cohort):
    """A comparison over a subset would agree more easily and say less."""
    expected = sum(int(b["batch"]["disease_ids"].size(0)) for b in cohort.batches)

    result = run(cohort)

    assert result.n_samples == expected
    assert expected > 0


def test_the_two_aggregate_mrrs_are_identical(cohort):
    """Exactly identical, not close. Both reach the same
    `RankingMetrics.mean_reciprocal_rank` over rows that have just been asserted
    equal, so any gap here would mean the aggregation disagrees with its own
    inputs."""
    result = run(cohort)

    assert result.trainer_mrr == result.mode_a_mrr
    assert result.mrr_absolute_difference == 0.0


def test_the_predictions_are_truncated_rows_of_local_indices(cohort):
    """What was compared, stated rather than assumed: `LEGACY_TRUNCATION_K`
    subgraph-local column indices, or fewer where the subgraph is smaller."""
    result = run(cohort)
    trainer = make_trainer(cohort)
    trainer.model.eval()
    trainer_pass = trainer._run_evaluation_pass(cohort.batches)

    assert result.n_samples == len(trainer_pass.predictions)
    for row in trainer_pass.predictions:
        assert len(row) <= LEGACY_TRUNCATION_K
        assert all(isinstance(value, str) for value in row)


# ---------------------------------------------------------------------------
# The conditions the agreement holds under
# ---------------------------------------------------------------------------
def test_amp_is_off_on_cpu_so_the_question_asked_was_the_exact_one(cohort):
    """`TrainerConfig(use_amp=True)` above is deliberate: the trainer resolves it
    against the device (`trainer.py:380`) and lands on False. Asserting the
    *resolved* state is what stops these tests from silently becoming a
    tolerance comparison if that resolution ever changes."""
    result = run(cohort)

    assert result.amp_enabled is False
    assert result.bit_exact_contract is True
    assert result.device == "cpu"


def test_the_result_serialises_what_a_reader_needs_to_interpret_it(cohort):
    report = run(cohort).to_dict()

    assert report["agreed"] is True
    assert report["n_disagreements"] == 0
    assert report["aggregate_mrr_agreed"] is True
    assert report["bit_exact_contract"] is True
    assert report["amp_enabled"] is False
    assert "trainer_mrr" in report and "mode_a_mrr" in report


# ---------------------------------------------------------------------------
# The guards against a comparison that passes by measuring nothing
# ---------------------------------------------------------------------------
def test_a_one_shot_iterator_is_frozen_rather_than_refused(cohort):
    """The hazard is real and the snapshot is what removes it.

    Handed straight to both paths, a generator is drained by whichever runs first;
    the second sees an empty stream, and on one of the two orderings that produces
    two empty cohorts and a cheerful `agreed=True`. Frozen once, it is simply a
    cohort — so this is accepted, not refused.

    Two earlier versions rejected it. Rejecting cost the caller a `list(...)` they
    should not have had to write, and bought nothing the snapshot does not already
    guarantee.
    """
    result = compare_trainer_against_mode_a(
        make_trainer(cohort), (b for b in cohort.batches), cohort.manifest,
        device=torch.device("cpu"),
    )

    assert result.agreed is True
    assert result.n_samples > 0


def test_a_rebuilding_iterable_is_frozen_rather_than_refused(cohort):
    """A container that hands back new objects each pass is only dangerous if it is
    traversed twice. It is traversed once."""

    class Rebuilding(list):
        def __iter__(self):
            return iter([dict(batch) for batch in list.__iter__(self)])

    result = compare_trainer_against_mode_a(
        make_trainer(cohort), Rebuilding(cohort.batches), cohort.manifest,
        device=torch.device("cpu"),
    )

    assert result.agreed is True


def test_a_re_iterable_that_is_neither_list_nor_tuple_is_accepted(cohort):
    """A real cohort may reasonably arrive in a container of the caller's own. The
    first version of this guard turned those away on type alone."""

    class Holder:
        def __init__(self, items):
            self._items = items

        def __iter__(self):
            return iter(self._items)

    result = compare_trainer_against_mode_a(
        make_trainer(cohort), Holder(cohort.batches), cohort.manifest,
        device=torch.device("cpu"),
    )

    assert result.agreed is True


def test_the_caller_is_traversed_exactly_once(cohort):
    """The reason the identity check went. A second pass over a live dataloader
    re-samples subgraphs — it is not free, and on a real cohort it is not cheap.
    """

    class Counting:
        def __init__(self, items):
            self._items = items
            self.traversals = 0

        def __iter__(self):
            self.traversals += 1
            return iter(self._items)

    counting = Counting(cohort.batches)

    compare_trainer_against_mode_a(
        make_trainer(cohort), counting, cohort.manifest, device=torch.device("cpu"),
    )

    assert counting.traversals == 1


def test_something_that_is_not_iterable_at_all_is_refused(cohort):
    """One of the two refusals a snapshot cannot repair."""
    with pytest.raises(TypeError, match="must be iterable"):
        compare_trainer_against_mode_a(
            make_trainer(cohort), 7, cohort.manifest, device=torch.device("cpu"),
        )


def test_an_empty_cohort_is_refused(cohort):
    with pytest.raises(ValueError, match="no batches to compare"):
        compare_trainer_against_mode_a(
            make_trainer(cohort), [], cohort.manifest, device=torch.device("cpu"),
        )


def test_a_cohort_that_scores_no_rows_is_refused(cohort):
    """The reachable route to an empty cohort, and it is not an empty batch list.

    `_compute_model_outputs` returns early when the encoder yields no disease or
    phenotype embeddings, and `_run_evaluation_pass` then completes normally with
    no predictions and an empty metric dict — it does not raise.

    **The old comparator did not return agreement here**, and this test must not be
    read as though it did. Only the trainer's side is broken in this mutation, so
    Mode A still returns the full cohort and the `n_trainer != n_mode_a` check
    refuses the mismatch. What the new guard adds is an earlier and
    comparator-specific refusal: it fails before a wasted Mode A pass, and it says
    "no scored rows" instead of a count mismatch that leaves the reader to work out
    why one side was empty.
    """
    trainer = make_trainer(cohort)
    trainer._compute_model_outputs = lambda node_embeddings, *a, **k: {
        "node_embeddings": node_embeddings
    }

    with pytest.raises(ValueError, match="no scored rows"):
        compare_trainer_against_mode_a(
            trainer, cohort.batches, cohort.manifest, device=torch.device("cpu"),
        )


def test_mode_a_cannot_be_supplied_by_the_caller(cohort):
    """The acceptance gate does not accept evidence about what it is gating.

    An earlier draft took an optional `mode_a_result` to save a forward pass. No
    amount of field checking repairs that seam — patient ids and row counts can
    all match while the supplied result came from a different negative draw over
    the same patients — so it is gone, and the saving is served instead by
    `DifferentialResult.mode_a_result` carrying the run this function performed.
    """
    import inspect

    parameters = inspect.signature(compare_trainer_against_mode_a).parameters

    assert "mode_a_result" not in parameters


def test_the_verdict_carries_the_mode_a_run_it_performed(cohort):
    """So a caller needing both the artifact and the verdict pays for one pass."""
    result = run(cohort)

    assert result.mode_a_result is not None
    assert result.mode_a_result.sample_ids == [
        pid for b in cohort.batches for pid in b["batch"]["patient_ids"]
    ]
    assert "mode_a_result" not in result.to_dict()


# ---------------------------------------------------------------------------
# Mutation verification — the comparison can fail
# ---------------------------------------------------------------------------
def test_a_perturbed_mode_a_score_is_caught(cohort, monkeypatch):
    """The scoring leg. `run_modes_ab` imports the primitives *inside* the
    function, so patching the module attribute reaches it.

    A sign flip rather than noise: it reverses the ranking outright, so the
    failure cannot be a floating-point coincidence."""
    import src.inference.scoring as scoring

    original = scoring.cosine_score_matrix
    monkeypatch.setattr(
        scoring, "cosine_score_matrix", lambda p, c: -original(p, c)
    )

    result = run(cohort)

    assert result.agreed is False
    assert result.n_disagreements_by_kind.get("top_k", 0) > 0


def test_a_perturbed_mode_a_pooling_is_caught(cohort, monkeypatch):
    """The pooling leg, isolated from the scoring leg. The trainer's inline
    masked mean and `masked_mean_pool` are two implementations of one operation;
    this is the test that says so."""
    import src.inference.scoring as scoring

    original = scoring.masked_mean_pool
    monkeypatch.setattr(
        scoring, "masked_mean_pool", lambda e, m: original(e, m).flip(-1)
    )

    result = run(cohort)

    assert result.agreed is False
    assert result.n_disagreements_by_kind.get("top_k", 0) > 0


def test_a_trainer_that_reports_a_different_truth_is_caught(cohort):
    """The truth leg. Rolling `diagnosis_targets` by one row leaves the scores
    untouched, so only the `truth` comparison can catch it — which is exactly the
    defect a shared-`disease_ids` assumption would hide."""
    trainer = make_trainer(cohort)
    inner = trainer._compute_model_outputs

    def rolled(*args, **kwargs):
        outputs = inner(*args, **kwargs)
        if "diagnosis_targets" in outputs:
            outputs["diagnosis_targets"] = outputs["diagnosis_targets"].roll(1)
        return outputs

    trainer._compute_model_outputs = rolled

    result = compare_trainer_against_mode_a(
        trainer, cohort.batches, cohort.manifest, device=torch.device("cpu"),
    )

    assert result.agreed is False
    assert result.n_disagreements_by_kind.get("truth", 0) > 0


def test_a_truncation_disagreement_is_caught(cohort, monkeypatch):
    """The truncation leg, which none of the mutations above reach.

    The trainer hardcodes `pred_indices[:20]`; Mode A slices with the module
    global. Moving only the global makes the two disagree about *how much* of the
    ranking to compare while agreeing about the ranking itself — a defect that
    leaves both MRRs plausible and only the row lengths wrong.

    Skipped rather than faked where the synthetic subgraph is narrower than the
    reduced K, because then both sides emit the whole row and there is nothing to
    truncate. Saying so beats an assertion that passes for the wrong reason.
    """
    import src.evaluation.measurement as measurement

    trainer = make_trainer(cohort)
    trainer.model.eval()
    widest = max(len(row) for row in trainer._run_evaluation_pass(cohort.batches).predictions)
    reduced = widest - 1
    if reduced < 1:
        pytest.skip("subgraph too narrow for a truncation change to be observable")

    monkeypatch.setattr(measurement, "LEGACY_TRUNCATION_K", reduced)

    result = run(cohort)

    assert result.agreed is False
    assert result.n_disagreements_by_kind.get("top_k", 0) > 0


# ---------------------------------------------------------------------------
# The truncation contract, which only a cohort wider than K can exercise
# ---------------------------------------------------------------------------
def test_the_wide_cohort_actually_truncates(wide_cohort):
    """The precondition for the test below, asserted rather than assumed.

    If this stopped holding — a narrower subgraph, a changed sampling limit — the
    agreement test after it would silently go back to proving nothing about
    truncation while still passing.
    """
    universe = {int(b["original_indices"]["disease"].numel()) for b in wide_cohort.batches}

    assert min(universe) > LEGACY_TRUNCATION_K, (
        f"candidate universes are {sorted(universe)}; a cohort at or below "
        f"{LEGACY_TRUNCATION_K} cannot exercise the truncation"
    )


def test_the_two_paths_agree_where_the_ranking_is_truncated(wide_cohort):
    """The truncation contract: two separately written constants — the trainer's
    hardcoded `[:20]` and Mode A's `LEGACY_TRUNCATION_K` — cutting the same
    ranking to the same depth, on rows long enough for a difference to show."""
    result = run(wide_cohort)

    assert result.disagreements == []
    assert result.agreed is True
    assert result.trainer_mrr == result.mode_a_mrr


def test_both_sides_emit_exactly_k_predictions_on_the_wide_cohort(wide_cohort):
    """Stated in the artifact, not only in the comparison: a run that agreed on
    rows both sides had truncated to ten would satisfy the test above."""
    trainer = make_trainer(wide_cohort)
    trainer.model.eval()
    trainer_pass = trainer._run_evaluation_pass(wide_cohort.batches)
    mode_a = run_mode_a(
        wide_cohort.model, wide_cohort.batches, wide_cohort.manifest,
        device=torch.device("cpu"),
    )

    assert {len(row) for row in trainer_pass.predictions} == {LEGACY_TRUNCATION_K}
    assert {len(row) for row in mode_a.legacy_top_k_local} == {LEGACY_TRUNCATION_K}


def test_a_truncation_disagreement_is_caught_on_the_wide_cohort(wide_cohort, monkeypatch):
    """The same mutation as on the narrow cohort, but here the reduced depth is
    still a genuine truncation of a longer ranking rather than a slice that
    happens to bite because the subgraph is tiny."""
    import src.evaluation.measurement as measurement

    monkeypatch.setattr(measurement, "LEGACY_TRUNCATION_K", LEGACY_TRUNCATION_K - 5)

    result = run(wide_cohort)

    assert result.agreed is False
    assert result.n_disagreements_by_kind.get("top_k", 0) > 0


def test_an_aggregate_only_disagreement_is_caught(cohort, monkeypatch):
    """The fourth check, which the first draft listed and did not enforce.

    The mutation moves **only** the aggregate: every sample id, top-20 row and
    truth is preserved, so all three per-sample comparisons still agree and
    `disagreements` stays empty. A verdict of `agreed = not disagreements` would
    pass this, which is exactly the false pass being closed.

    `dataclasses.replace` on the frozen `ModeAResult` is what makes the mutation
    surgical — patching `RankingMetrics` instead would move both sides at once
    and prove nothing about the verdict.
    """
    import dataclasses

    import src.evaluation.differential as differential

    real = differential.run_mode_a

    def shifted(*args, **kwargs):
        result = real(*args, **kwargs)
        key = differential.legacy_mrr_key()
        return dataclasses.replace(
            result, legacy_metrics={**result.legacy_metrics, key: result.legacy_metrics[key] + 0.5}
        )

    monkeypatch.setattr(differential, "run_mode_a", shifted)

    result = run(cohort)

    assert result.disagreements == [], "the mutation must not disturb any sample"
    assert result.aggregate_mrr_agreed is False
    assert result.agreed is False
    assert result.mrr_absolute_difference == pytest.approx(0.5)


def test_the_verdict_records_the_compile_wrapper_it_observed(cohort):
    """Backlog item D2, on the artifact where the numeric regime actually varies.

    It cannot make the two paths disagree — they are handed the same model object,
    so it is wrapped for both or neither. It is recorded for the same reason
    `amp_dtype` is: a verdict that does not say what regime produced it cannot be
    compared with a later one.

    `False` here is an observation. `None` would mean nothing was observed. Neither
    says anything about whether a compiled graph executed.
    """
    result = run(cohort)

    assert result.torch_compile_wrapped is False
    assert result.to_dict()["torch_compile_wrapped"] is False


# ---------------------------------------------------------------------------
# D3's remaining item: legal-truth equality where local and global differ
# ---------------------------------------------------------------------------
#
# The two cohorts above cannot test this. Their subgraphs reach every disease, so
# `original_indices["disease"]` is `[0, 1, ..., n-1]` — an identity map, under
# which a side reading the wrong space produces the right numbers anyway. That is
# the "Direction" failure `tests/unit/test_measurement_ranking.py` names: local and
# global ids are both plausible integers, so a wrong-direction translation yields
# valid-looking, wrong metrics that nothing downstream can detect.
#
# So this section builds one batch by hand with a deliberately non-identity,
# non-monotonic map. It is not an end-to-end cohort and is not trying to be; it
# exists to make local != global so the equality has something to say.

GLOBAL_DISEASE_IDS = [7, 2, 9, 4]   # non-identity and non-sorted, both on purpose
LOCAL_TRUTHS = [0, 2, 3]            # -> globals 7, 9, 4
HIDDEN = 4


class _StubEncoder(torch.nn.Module):
    """Deterministic embeddings that ignore their inputs.

    The comparison under test is about identifier spaces, not about the encoder,
    and an input-dependent model would make a failure ambiguous between the two.
    It still owns a parameter, because `Trainer` builds an optimizer over
    `model.parameters()`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(HIDDEN))

    def forward(self, x_dict, edge_index_dict):
        generator = torch.Generator().manual_seed(11)
        return {
            "disease": torch.randn(len(GLOBAL_DISEASE_IDS), HIDDEN, generator=generator),
            "phenotype": torch.randn(5, HIDDEN, generator=generator),
        }


def _remapped_batch():
    return {
        "batch": {
            "phenotype_ids": torch.tensor([[0, 1, -1], [2, 3, -1], [1, 4, -1]]),
            "phenotype_mask": torch.tensor(
                [[True, True, False], [True, True, False], [True, True, False]]
            ),
            "disease_ids": torch.tensor(LOCAL_TRUTHS),
            "patient_ids": ["r0", "r1", "r2"],
        },
        "subgraph_x_dict": {},
        "subgraph_edge_index_dict": {},
        "original_indices": {"disease": torch.tensor(GLOBAL_DISEASE_IDS)},
    }


def _remapped_trainer():
    from src.training.callbacks import Callback

    return Trainer(
        model=_StubEncoder(),
        train_dataloader=[],
        val_dataloader=[],
        config=TrainerConfig(device="cpu", use_amp=True, scheduler_type="none", seed=0),
        callbacks=[Callback()],
    )


def _remapped_manifest(cohort: Cohort):
    """One real manifest with its cohort size corrected, rather than a hand-built
    one: `MeasurementManifest` has thirty-odd required fields and a second
    construction site would be a copy to keep in step for no gain."""
    import dataclasses

    return dataclasses.replace(cohort.manifest, n_samples=len(LOCAL_TRUTHS))


def test_the_map_this_section_uses_is_really_not_the_identity(cohort):
    """The precondition. If it ever became the identity, the test below would go
    on passing while testing nothing — which is how the gap it closes arose."""
    assert GLOBAL_DISEASE_IDS != list(range(len(GLOBAL_DISEASE_IDS)))
    assert GLOBAL_DISEASE_IDS != sorted(GLOBAL_DISEASE_IDS)
    assert all(
        list(b["original_indices"]["disease"]) == list(range(b["original_indices"]["disease"].numel()))
        for b in cohort.batches
    ), "the shared cohort is identity-mapped, which is why this section exists"


def test_the_two_paths_agree_about_the_truth_where_local_is_not_global(cohort):
    """D3's legal-truth equality: the trainer reports subgraph-local truths, Mode A
    reports global ones, and they describe the same diseases.

    Under a non-identity map this can fail; under the identity map of the shared
    cohorts it could not.
    """
    batches = [_remapped_batch()]

    result = compare_trainer_against_mode_a(
        _remapped_trainer(), batches, _remapped_manifest(cohort),
        device=torch.device("cpu"),
    )

    assert result.agreed is True
    assert result.mode_a_result.truth_global_ids == [
        GLOBAL_DISEASE_IDS[local] for local in LOCAL_TRUTHS
    ]


def test_a_trainer_reporting_global_truths_is_caught(cohort):
    """The mutation the identity map would have hidden.

    A trainer emitting **global** ids where local ones belong is the exact
    wrong-direction defect: both are plausible integers, so nothing downstream can
    tell them apart from the values alone.

    **Which boundary refuses it, checked rather than predicted.** The first draft
    of this test expected the harness's `to_global_ids` range check. It is refused
    earlier than that, by `DiagnosisLoss._label_smoothed_ce`'s `scatter_` — global
    id 7 into a four-column target. That is the second of D3's three boundaries
    (BACKLOG §2.3) firing before the harness sees anything, and it matches what
    §2.3 already measured: the trainer refuses through the loss, not through the
    metric. The harness boundary is tested where it lives, in
    `tests/unit/test_measurement_ranking.py`.

    Under an identity map this mutation is literally a no-op, which is asserted
    below so the reason this test needs the remapped batch is not left implicit.
    """
    identity = list(range(len(GLOBAL_DISEASE_IDS)))
    assert [identity[local] for local in LOCAL_TRUTHS] == LOCAL_TRUTHS, (
        "under an identity map this mutation changes nothing; the non-identity "
        "map is what gives it something to break"
    )
    trainer = _remapped_trainer()
    inner = trainer._compute_model_outputs
    lookup = torch.tensor(GLOBAL_DISEASE_IDS)

    def as_globals(*args, **kwargs):
        outputs = inner(*args, **kwargs)
        if "diagnosis_targets" in outputs:
            outputs["diagnosis_targets"] = lookup[outputs["diagnosis_targets"]]
        return outputs

    trainer._compute_model_outputs = as_globals

    with pytest.raises(RuntimeError, match="index 7 is out of bounds"):
        compare_trainer_against_mode_a(
            trainer, [_remapped_batch()], _remapped_manifest(cohort),
            device=torch.device("cpu"),
        )


def test_disagreeing_samples_are_counted_as_samples_not_as_disagreements(cohort, monkeypatch):
    """One sample can raise up to three disagreements, so the counts by kind
    cannot be added into a rate.

    This is what an experimenter reads to characterise an AMP-on run: how much of
    the cohort moved, not how many assertions fired. The harness reports the number
    and the rate and takes no position on whether either is acceptable.
    """
    import src.inference.scoring as scoring

    original = scoring.cosine_score_matrix
    monkeypatch.setattr(scoring, "cosine_score_matrix", lambda p, c: -original(p, c))

    result = run(cohort)

    assert result.agreed is False
    assert 0 < result.n_samples_disagreeing <= result.n_samples
    assert result.n_samples_disagreeing <= sum(result.n_disagreements_by_kind.values())
    assert result.to_dict()["sample_disagreement_rate"] == (
        result.n_samples_disagreeing / result.n_samples
    )


def test_an_agreeing_run_reports_a_zero_rate(cohort):
    result = run(cohort)

    assert result.n_samples_disagreeing == 0
    assert result.to_dict()["sample_disagreement_rate"] == 0.0
