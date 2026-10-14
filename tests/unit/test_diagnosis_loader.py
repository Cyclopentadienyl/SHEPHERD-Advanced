"""
What a diagnosis batch is made of: its negatives, its random stream, its subgraph and its truth.
==============================================================================================
`DiagnosisDataLoader` builds every batch training trains on and validation ranks
(`scripts/train_model.py`, `create_dataloaders`). Four of its guarantees decide
what those batches mean, and none of them is the Trainer's or a measurement's:

  - **every sample carries exactly `num_negative_samples` negatives**
    (`DiagnosisDataset.__getitem__`), wired from the config by
    `create_diagnosis_dataloader`;
  - **worker-drawn negatives follow the parent seed**, and the worker count is
    part of that stream rather than a throughput setting;
  - **each batch's subgraph is the 2-hop neighbourhood of its seeds**
    (`DIAGNOSIS_SUBGRAPH_HOPS`). Validation ranks the subgraph's diseases, so the
    hop count is the size of what is ranked;
  - **the truth is a seed of its own subgraph** (`_get_seed_nodes`), so it is a
    row of the batch whether or not the graph connects it to the patient.

The shared synthetic workspace cannot show the last two. Every gene there links
every phenotype to every disease (`tests/fixtures/synthetic_workspace.py`), so a
truth left out of the seeds is reached anyway, and with four diseases the
negatives alone seed nearly all of them, so the hop count barely changes what a
batch holds. The graphs below are built so the distances are the point: a node
two hops from the seeds and one three hops away, and a truth no patient can
reach.

The parent is seeded the way training seeds it -- `random`, NumPy and torch,
the three generators `Trainer._set_seed` sets -- and the loaders are built
through `create_diagnosis_dataloader`, the function training calls.

Module: tests/unit/test_diagnosis_loader.py
"""
import pytest

torch = pytest.importorskip("torch")

from src.kg.data_loader import (  # noqa: E402
    DataLoaderConfig,
    DiagnosisSample,
    create_diagnosis_dataloader,
)
from tests.fixtures.synthetic_workspace import _graph, _samples  # noqa: E402

#: The value the worker-stream tests used before they moved here.
SEED = 0

PHENOTYPE_GENE = ("phenotype", "associated_with", "gene")
GENE_DISEASE = ("gene", "causes", "disease")


def _graph_data(num_nodes, edges):
    """A graph in the shape `read_graph_artifacts` hands training.

    Each node's single feature is its own global id, so a row the batch hands the
    model names the node it came from.
    """
    return {
        "x_dict": {
            node_type: torch.arange(n, dtype=torch.float32).unsqueeze(1)
            for node_type, n in num_nodes.items()
        },
        "edge_index_dict": {
            edge_type: torch.tensor(pairs, dtype=torch.long).t()
            for edge_type, pairs in edges.items()
        },
        "num_nodes_dict": dict(num_nodes),
    }


# ---------------------------------------------------------------------------
# Negatives
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_negative_samples", [1, 3, 7])
def test_every_sample_carries_exactly_the_configured_number_of_negatives(
    num_negative_samples,
):
    """Neither of the two defaults: the dataset's own is 10 and the config's is
    5, so a count that came from either rather than from the config would show.
    Seven is more than the fixture has diseases besides the truth, so the count
    holds even when the draws must repeat."""
    x_dict, edge_index_dict, num_nodes_dict = _graph()
    samples = [
        DiagnosisSample(patient_id=s["patient_id"], phenotype_ids=s["phenotype_ids"],
                        disease_id=s["disease_id"])
        for s in _samples()
    ]
    loader = create_diagnosis_dataloader(
        samples=samples,
        graph_data={"x_dict": x_dict, "edge_index_dict": edge_index_dict,
                    "num_nodes_dict": num_nodes_dict},
        config=DataLoaderConfig(batch_size=4, num_workers=0, shuffle=False,
                                num_negative_samples=num_negative_samples),
    )

    drawn = 0
    for batch_data in loader:
        batch = batch_data["batch"]
        diseases = batch_data["original_indices"]["disease"]
        negatives = batch["negative_disease_ids"]

        assert negatives.shape == (len(batch["patient_ids"]), num_negative_samples)
        assert ((negatives >= 0) & (negatives < diseases.numel())).all()
        # Back in global space, no negative is its own sample's truth.
        assert (diseases[negatives] != diseases[batch["disease_ids"]].unsqueeze(1)).all()
        drawn += negatives.numel()

    assert drawn == len(samples) * num_negative_samples


class TestTheWorkerStreamFollowsTheParentSeed:
    """Training builds its loaders with worker processes and seeds only the
    parent, so a run's negatives are reproducible only if the parent's seed
    reaches those processes."""

    @staticmethod
    def _worker_negatives(seed: int, workers: int):
        """Seed the parent as training does, then draw negatives through the
        loader training builds. Negatives come from `random.randint` inside
        `DiagnosisDataset.__getitem__`, which runs in the worker processes;
        `num_diseases` is large relative to the batch so the draws really vary.
        The finite `timeout` turns a hung worker into a failure instead of a
        stalled test run."""
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        samples = [
            DiagnosisSample(patient_id=f"P{i}", phenotype_ids=[i % 3, (i + 1) % 3],
                            disease_id=i % 7)
            for i in range(24)
        ]
        loader = create_diagnosis_dataloader(
            samples=samples,
            graph_data=_graph_data({"phenotype": 3, "disease": 500}, {}),
            config=DataLoaderConfig(batch_size=4, num_workers=workers, shuffle=False),
        )
        # Read by the torch DataLoader when each iteration starts, so setting it
        # after construction applies to the run below.
        loader._dataloader.timeout = 60 if workers > 0 else 0
        return [
            batch_data["original_indices"]["disease"][
                batch_data["batch"]["negative_disease_ids"]
            ].tolist()
            for batch_data in loader
        ]

    @staticmethod
    def _default_workers() -> int:
        """The loader's default, which is also training's (`TrainConfig.num_workers`
        in `scripts/train_model.py`)."""
        return DataLoaderConfig().num_workers

    def test_worker_drawn_negatives_follow_the_parent_seed(self):
        """PyTorch seeds each worker as `base_seed + worker_id`, with `base_seed`
        drawn from the parent's torch RNG. Seeding the parent therefore reaches
        into the worker processes, which is the only reason a multi-worker run can
        be compared at all. The second assertion is the half that makes the first
        mean something: a constant reproduces perfectly too."""
        workers = self._default_workers()

        first = self._worker_negatives(SEED, workers)
        assert first == self._worker_negatives(SEED, workers)
        assert first != self._worker_negatives(SEED + 1, workers)

    def test_the_worker_count_is_part_of_the_stream(self):
        """Not a throughput setting. PyTorch hands batch i to worker i mod W, whose
        stream is seeded base_seed + (i mod W), so changing the count can move
        batches onto different streams; loading in-process (0) uses the parent's
        own stream instead. That case always differs and is the one pinned here."""
        workers = self._default_workers()
        assert workers > 0

        assert self._worker_negatives(SEED, workers) != (
            self._worker_negatives(SEED, 0)
        )


# ---------------------------------------------------------------------------
# The subgraph
# ---------------------------------------------------------------------------
#: phenotype 0 -- gene 0 -- phenotype 1 -- gene 1 -- phenotype 2, and gene 0 --
#: disease 1. From the patient's phenotype 0: gene 0 is one hop, phenotype 1 and
#: disease 1 two, gene 1 three. Every other disease, the truth included, has no
#: edge, so a negative adds only itself, and disease 1 drawn as a negative
#: reaches nothing phenotype 0 does not.
TWO_HOP_GRAPH = _graph_data(
    {"phenotype": 3, "gene": 2, "disease": 6},
    {PHENOTYPE_GENE: [(0, 0), (1, 0), (1, 1), (2, 1)], GENE_DISEASE: [(0, 1)]},
)


def test_the_subgraph_is_the_two_hop_neighbourhood_of_the_seeds():
    """One hop fewer and phenotype 1 is not in the batch, nor disease 1 unless it
    was drawn as a negative; one more and gene 1 is. Validation ranks the
    subgraph's diseases, so either change would move what `val_mrr` is computed
    over."""
    samples = [DiagnosisSample(patient_id="P0", phenotype_ids=[0], disease_id=0)]
    loader = create_diagnosis_dataloader(
        samples=samples, graph_data=TWO_HOP_GRAPH,
        config=DataLoaderConfig(batch_size=1, num_workers=0, shuffle=False),
    )

    (batch_data,) = list(loader)
    nodes = batch_data["original_indices"]

    assert nodes["phenotype"].tolist() == [0, 1]
    assert nodes["gene"].tolist() == [0]
    assert 1 in nodes["disease"].tolist()


# ---------------------------------------------------------------------------
# The truth
# ---------------------------------------------------------------------------
TRUTH = 5
TRUTH_GENE = 1
TRUTH_PHENOTYPE = 2

#: Two components. The patients' phenotypes 0 and 1 and gene 0 link diseases
#: 0-3; the truth links only to its own gene and phenotype, which no patient
#: carries. Disease 4 has no edge. Nothing a patient carries reaches the truth at
#: any distance, and no other disease does either, so the truth is in a batch
#: only if it is seeded.
UNREACHABLE_TRUTH_GRAPH = _graph_data(
    {"phenotype": 3, "gene": 2, "disease": 6},
    {
        PHENOTYPE_GENE: [(0, 0), (1, 0), (TRUTH_PHENOTYPE, TRUTH_GENE)],
        GENE_DISEASE: [(0, 0), (0, 1), (0, 2), (0, 3), (TRUTH_GENE, TRUTH)],
    },
)
TRUTH_COMPONENT = {
    ("disease", TRUTH), ("gene", TRUTH_GENE), ("phenotype", TRUTH_PHENOTYPE),
}


def test_the_truth_is_a_row_of_its_batch_even_when_no_patient_reaches_it():
    """Every patient shares one truth, so no patient's negative can be it -- each
    sample's draws exclude its own truth -- and the only way into the batch is
    the seed. The truth's local id has to name the truth's own row: the row the
    model is handed under that id carries the truth's feature."""
    samples = [
        DiagnosisSample(patient_id=f"P{i}", phenotype_ids=[0, 1][: 1 + i % 2],
                        disease_id=TRUTH, gene_ids=[0])
        for i in range(4)
    ]
    # The test's own preconditions. If an edge left the truth's component, or a
    # patient carried one of its nodes, the truth would be reachable and a green
    # run would prove nothing.
    for (src_type, _, dst_type), pairs in UNREACHABLE_TRUTH_GRAPH["edge_index_dict"].items():
        for src, dst in pairs.t().tolist():
            assert ((src_type, src) in TRUTH_COMPONENT) == ((dst_type, dst) in TRUTH_COMPONENT)
    carried = {("phenotype", p) for s in samples for p in s.phenotype_ids}
    carried |= {("gene", g) for s in samples for g in s.gene_ids}
    assert not carried & TRUTH_COMPONENT

    loader = create_diagnosis_dataloader(
        samples=samples, graph_data=UNREACHABLE_TRUTH_GRAPH,
        config=DataLoaderConfig(batch_size=2, num_workers=0, shuffle=False),
    )

    batches = list(loader)
    assert len(batches) == 2
    for batch_data in batches:
        batch = batch_data["batch"]
        diseases = batch_data["original_indices"]["disease"]
        truths = batch["disease_ids"]

        assert TRUTH not in diseases[batch["negative_disease_ids"]].flatten().tolist()
        assert TRUTH in diseases.tolist()
        assert ((truths >= 0) & (truths < diseases.numel())).all()
        assert diseases[truths].tolist() == [TRUTH] * len(batch["patient_ids"])
        assert batch_data["subgraph_x_dict"]["disease"][truths, 0].tolist() == (
            [float(TRUTH)] * len(batch["patient_ids"])
        )


def test_a_truth_left_out_of_its_subgraph_is_refused_by_the_loader(monkeypatch):
    """The run-time half of the same guarantee. A batch whose truth the seeding
    lost must stop at the loader, not reach the Trainer as `-1` -- which a gather
    reads as the last disease row. The seeds are cut down in the test, on the
    graph that cannot reach the truth any other way."""
    samples = [DiagnosisSample(patient_id="P0", phenotype_ids=[0], disease_id=TRUTH)]
    loader = create_diagnosis_dataloader(
        samples=samples, graph_data=UNREACHABLE_TRUTH_GRAPH,
        config=DataLoaderConfig(batch_size=1, num_workers=0, shuffle=False),
    )
    seeds_of = loader._get_seed_nodes

    def without_the_truth(batch):
        seeds = seeds_of(batch)
        seeds["disease"] = seeds["disease"][seeds["disease"] != TRUTH]
        return seeds

    monkeypatch.setattr(loader, "_get_seed_nodes", without_the_truth)

    with pytest.raises(ValueError, match="disease truth out of range"):
        list(loader)
