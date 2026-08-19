"""
Ranking for offline measurement — two streams, deliberately.
===========================================================
Module: src/evaluation/measurement.py

Mode A has to satisfy two requirements that one ranking cannot:

  - **reproduce the frozen evaluator exactly**, so its aggregate number can be
    compared against the historical one and the harness itself calibrated;
  - **rank deterministically**, so numbers from modes A, B, C and D can be
    compared with each other.

These conflict whenever scores tie. The frozen evaluator's order at a tie is
whatever `Tensor.sort` produced on that machine; a comparison across modes needs
an order that does not depend on which machine, which batch composition or which
input ordering produced it. So there are two streams, named for what each is for,
and neither is used for the other's purpose.

What the ranking functions below do **not** do: load data, run a model, or decide
what to do about a ground truth that is absent from the candidate set. They rank,
and they map identifiers. Absence is a mode-level decision — Mode A treats it as
a harness failure, because there the truth is one of the subgraph's own seeds —
and it is made in the driver at the bottom of this file, not in the primitives.

Dependencies: torch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch import Tensor

__all__ = [
    "LEGACY_TRUNCATION_K",
    "MeasurementManifest",
    "ModeAResult",
    "ModeResult",
    "to_global_ids",
    "canonical_ranking",
    "legacy_ranking",
    "ranks_of_truth",
    "run_mode_a",
    "run_modes_ab",
    "encode_full_graph",
    "run_mode_c",
    "assert_constructions_agree",
]


def _require_integer_ids(ids: Tensor, name: str) -> None:
    """Identifiers must be integer tensors, and `bool` is not an integer here.

    ``Tensor.long()`` turns 1.7 into 1 and ``True`` into 1, so a float or boolean
    tensor wired in by mistake becomes a plausible index rather than an error —
    the same silent-wrong-answer shape as a float mask reaching `masked_mean_pool`.
    """
    if ids.dtype == torch.bool or torch.is_floating_point(ids) or torch.is_complex(ids):
        raise ValueError(f"{name} must be an integer tensor, not {ids.dtype}")


def to_global_ids(original_indices: Tensor, local_ids: Tensor) -> Tensor:
    """Translate subgraph-local node indices into global knowledge-graph ids.

    Used for diseases when ranking candidates and for phenotypes when Mode B
    pools a patient from full-graph embeddings. It was named for diseases while
    they were its only caller; nothing about it was ever disease-specific, and a
    disease-shaped name on the phenotype path would have been the misleading kind
    of accurate.

    ``original_indices`` is the dataloader's ``original_indices["disease"]``
    — it is *already* the local-to-global direction: built as
    ``torch.tensor(sorted(nodes))`` and indexed by local position
    (`src/kg/data_loader.py:336-342`), so entry *i* holds the global index of local
    node *i*. The sibling ``node_mapping`` is the **opposite** direction, a
    global-to-local dict; inverting it here would be work with an extra chance of
    getting the direction backwards.

    This exists as a named function rather than a bare index because a silent
    direction error produces plausible ids and therefore plausible, wrong metrics.

    Local indices are only comparable within the batch that produced them, so
    everything persisted or aggregated must be global.
    """
    _require_integer_ids(original_indices, "original_indices")
    _require_integer_ids(local_ids, "local_ids")
    if original_indices.dim() != 1:
        raise ValueError(
            f"original_indices must be 1-D; got {tuple(original_indices.shape)}"
        )
    if local_ids.numel() and int(local_ids.max()) >= original_indices.numel():
        raise ValueError(
            f"local id {int(local_ids.max())} is outside the subgraph's "
            f"{original_indices.numel()} disease nodes"
        )
    if local_ids.numel() and int(local_ids.min()) < 0:
        raise ValueError("local ids must be non-negative")

    return original_indices.to(local_ids.device)[local_ids.long()]


def canonical_ranking(scores: Tensor, global_disease_ids: Tensor) -> Tensor:
    """Rank by score descending, breaking ties by ascending global disease id.

    ``scores`` is ``(B, D)``, ``global_disease_ids`` is ``(D,)``; returns
    ``(B, D)`` of global ids, best first.

    **The authoritative stream.** Given identical score values and global ids, its
    order — including its tie rule — is independent of the order the candidates
    arrived in and of batch composition. That is what makes a number from Mode A
    comparable with one from Mode C. It is *not* a claim that different hardware
    produces identical scores; if the scores differ the ranking may differ, and
    that is a property of the scores, not of this function.

    **Non-finite scores are rejected.** A ``NaN`` sorts unpredictably and would
    turn an invalid score into a plausible rank, and from there into a plausible
    mean rank and MRR. `legacy_ranking` deliberately does not reject them, because
    its contract is to reproduce the oracle; this stream's contract is to be
    correct.

    The tie rule is obtained from PyTorch's own stable sort rather than a custom
    comparator: order the candidates by global id first, then sort by score with
    ``stable=True``, and stability preserves ascending-id order inside every
    equal-score group.
    """
    if scores.dim() != 2:
        raise ValueError(f"scores must be (B, D); got {tuple(scores.shape)}")
    if global_disease_ids.dim() != 1:
        raise ValueError(
            f"global_disease_ids must be (D,); got {tuple(global_disease_ids.shape)}"
        )
    if global_disease_ids.numel() != scores.size(1):
        raise ValueError(
            f"{global_disease_ids.numel()} global ids for {scores.size(1)} score columns"
        )
    if global_disease_ids.unique().numel() != global_disease_ids.numel():
        raise ValueError("global_disease_ids contains duplicates; the ranking would be ambiguous")
    _require_integer_ids(global_disease_ids, "global_disease_ids")
    if not torch.isfinite(scores).all():
        raise ValueError("scores contain NaN or infinity; the canonical ranking would be meaningless")

    id_order = torch.argsort(global_disease_ids)
    ordered_ids = global_disease_ids[id_order]
    ordered_scores = scores[:, id_order.to(scores.device)]

    rank_order = torch.argsort(ordered_scores, dim=-1, descending=True, stable=True)
    return ordered_ids.to(rank_order.device)[rank_order]


def legacy_ranking(scores: Tensor) -> Tensor:
    """Reproduce the frozen evaluator's order. ``(B, D)`` of **subgraph-local**
    column indices.

    `scripts/evaluate_model.py:295` sorts with ``scores.sort(dim=-1,
    descending=True)`` over the subgraph-local columns, so this calls the same
    thing on the same columns. Tie behaviour is therefore whatever that call does
    — **which is the point**: this stream exists to match the historical number,
    not to be well defined. Non-finite scores are *not* rejected here for the same
    reason: reproducing the oracle means reproducing whatever it did.

    **Local, deliberately.** The only per-sample artifact the frozen oracle writes
    is ``predictions[i][:20]`` — subgraph-local column indices as strings
    (`scripts/evaluate_model.py:505-519`) — and it does **not** persist the
    ``original_indices`` needed to translate them. Local space is therefore the
    only space in which the two can be compared at all, and returning global ids
    from here would make the one comparison this function exists for impossible.

    Translate explicitly where global identity is wanted:

        local_ids  = legacy_ranking(scores)
        global_ids = to_global_ids(original_indices, local_ids)

    The local order is for exact oracle calibration; the translated order is for
    persistence and rank extraction. Neither requires a second sort.

    **This function has a deletion date.** It exists solely so the institutional
    Mode A calibration can be checked against the frozen oracle, and it is removed
    together with `scripts/evaluate_model.py` once that calibration succeeds.
    Nothing else may depend on legacy tie behaviour.
    """
    if scores.dim() != 2:
        raise ValueError(f"scores must be (B, D); got {tuple(scores.shape)}")

    _, local_order = scores.sort(dim=-1, descending=True)
    return local_order


def ranks_of_truth(ranked_global_ids: Tensor, truth_global_ids: Tensor) -> List[Optional[int]]:
    """1-based rank of each row's ground truth, or ``None`` where it is absent.

    ``ranked_global_ids`` is ``(B, D)``; ``truth_global_ids`` is ``(B,)``.

    **``None`` rather than a sentinel rank.** A ground truth outside the candidate
    set has no rank, and encoding that as a large integer would let it flow into a
    mean as though it were a measurement. `RankingMetrics.compute_from_ranks`
    refuses to invent a value for an empty cohort for the same reason; this is the
    same refusal one level up. What absence *means* is the caller's decision: in
    Mode A it is impossible and therefore fatal, but a mode that scores a
    restricted candidate universe may legitimately observe it.

    A Python list, not a tensor, because the value is genuinely optional and
    because the consumer accumulates across batches into a list anyway.
    """
    if ranked_global_ids.dim() != 2:
        raise ValueError(f"ranked_global_ids must be (B, D); got {tuple(ranked_global_ids.shape)}")
    if truth_global_ids.dim() != 1:
        raise ValueError(f"truth_global_ids must be (B,); got {tuple(truth_global_ids.shape)}")
    if truth_global_ids.numel() != ranked_global_ids.size(0):
        raise ValueError(
            f"{truth_global_ids.numel()} truths for {ranked_global_ids.size(0)} ranked rows"
        )
    _require_integer_ids(ranked_global_ids, "ranked_global_ids")
    _require_integer_ids(truth_global_ids, "truth_global_ids")

    truth = truth_global_ids.to(ranked_global_ids.device).unsqueeze(-1)
    matches = ranked_global_ids == truth
    found = matches.any(dim=-1)
    # argmax over a boolean row gives the first True, which is the best rank a
    # duplicated id could hold. Duplicates are rejected upstream, so this is the
    # only position.
    positions = matches.to(torch.uint8).argmax(dim=-1)

    return [
        int(position) + 1 if bool(is_found) else None
        for position, is_found in zip(positions.tolist(), found.tolist())
    ]


# =============================================================================
# Mode A — the calibration control
# =============================================================================
# Mode A reproduces what the legacy evaluator measures, deliberately including
# what is wrong with it: a per-batch subgraph whose disease candidates are seeded
# from the answers, and pure cosine with no shortest-path term. A control that has
# been improved is not a control, so nothing here corrects the candidate
# construction or the sampling policy.
#
# What it adds is honesty about the numbers: the legacy truncated metric is
# reported as such, and the authoritative untruncated metrics are computed
# alongside it from the canonical ranking.

LEGACY_TRUNCATION_K = 20
"""Twenty, from two independent places in the frozen evaluator that happen to agree.

Its predictions file is hardcoded `predictions[i][:20]`
(`scripts/evaluate_model.py:513`). Its **MRR** is computed over lists truncated at
`max(top_k_values)` (`:324`), and `top_k_values` defaults to `[1, 3, 5, 10, 20]`
(`:116`) — so 20 again, but by a default rather than a constant. There is no
`--top-k-values` flag and no `--config` flag, so nothing on the command line can
move it; only editing the frozen file could, which is forbidden.

The distinction matters to calibration and is therefore checked rather than
assumed: the oracle's report echoes its config, and `scripts/calibrate_mode_a.py`
refuses to compare the two MRRs unless `max(top_k_values)` is this value. Two
numbers truncated at different K are not the same quantity."""


@dataclass(frozen=True)
class MeasurementManifest:
    """What has to be recorded for a number to mean anything later.

    Not a schema framework — a frozen dataclass and `dataclasses.asdict`. Every
    field is something that changes what the measurement *is*, so omitting one
    would make two runs incomparable without saying so.

    `batch_size` is here as **semantics, not performance**: Mode A's candidate
    universe is the batch's subgraph, so changing the batch size changes what was
    measured.
    """

    mode: str
    split: str
    n_samples: int
    # candidate construction
    candidate_construction: str
    negative_sampling_strategy: str
    num_negative_samples: int
    subgraph_strategy: str
    subgraph_hops: int
    num_neighbors: List[int]
    max_subgraph_nodes: int
    """The **configured** ceiling, from `DataLoaderConfig.max_subgraph_nodes`.

    Distinct from `sampler_evidence["max_subgraph_nodes"]`, which is the per-type
    maximum actually reached. A run that never approached the cap and a run that
    was truncated by it look the same in the observation alone; only the pair says
    which happened."""
    batch_size: int
    shuffle: bool
    num_workers: int
    # scoring and ranking
    score_semantics: str
    model_construction: str
    """Which loader built the model: the frozen evaluator's or production's.

    Mode A mirrors the oracle deliberately; B, C and D use production semantics.
    Recorded because a reader comparing two modes has to know whether the encoder
    difference they are looking at is the only difference — and because
    `assert_constructions_agree` passing is a fact about one checkpoint, not a
    permanent property."""
    legacy_truncation_k: int
    legacy_tie_policy: str
    canonical_tie_policy_version: str
    # artifacts
    checkpoint_path: str
    data_dir: str
    graph_fingerprint: Dict[str, Any]
    artifact_digests: Dict[str, Optional[str]]
    """Raw SHA-256 of every file the measurement consumed, keyed by role.

    `graph_fingerprint` above is a **structural** identity — node types, counts,
    feature dimensions. Two different checkpoints trained on the same graph share
    it, and so do two different sample files of the same shape. A path is not an
    identity either: `checkpoints/best.pt` names a different file every time
    training improves. These digests are what let a number be traced back to the
    exact bytes that produced it. `None` where the file was absent."""
    cuda_executed: bool
    """Whether this run executed on CUDA. **That is all it claims.**

    It was called `calibration_eligible`, which overclaimed: a synthetic workspace
    on a CUDA machine would have set it true. Eligibility for institutional
    acceptance is not a property this process can observe — it depends on whether
    `artifact_digests` are the institution's real checkpoint and cohort, which no
    code here can verify. Recorded as the narrow, checkable fact, so the broad
    claim has to be made by a person who can actually make it."""
    # runtime
    software_revision: Optional[str]
    torch_version: str
    cuda_version: Optional[str]
    device: str
    dtype: str
    amp_enabled: bool
    deterministic_algorithms: bool
    cudnn_deterministic: Optional[bool]
    cudnn_benchmark: Optional[bool]
    python_seed: Optional[int]
    numpy_seed: Optional[int]
    torch_seed: Optional[int]


@dataclass(frozen=True)
class ModeResult:
    """One measurement run, in any mode.

    Everything here is comparable **across** modes, which is the whole point of
    the ladder: the same cohort in the same order, each sample's truth rank under
    that mode's encoder and candidate universe. `canonical_ranks` is aligned with
    `sample_ids`, so A→B→C can be read per patient and not only in aggregate — an
    aggregate that moved by a little can hide a cohort where half the ranks
    improved and half collapsed.
    """

    manifest: MeasurementManifest
    authoritative_metrics: Dict[str, float]
    n_ranked: int
    n_ground_truth_absent: int
    sampler_evidence: Dict[str, Any]
    sample_ids: List[str]
    truth_global_ids: List[int]
    canonical_ranks: List[int]

    def to_dict(self) -> Dict[str, Any]:
        """The measurement report. **Per-sample rows are deliberately not here** —
        they are their own artifact, because one is a summary a human reads and
        the other is a bulk list a comparison consumes."""
        return {
            "manifest": asdict(self.manifest),
            "authoritative_metrics": self.authoritative_metrics,
            "n_ranked": self.n_ranked,
            "n_ground_truth_absent": self.n_ground_truth_absent,
            "sampler_evidence": self.sampler_evidence,
        }

    def to_ranks(self) -> List[Dict[str, Any]]:
        """Per-sample ranks, for comparing this mode against another."""
        return [
            {"sample_id": sample_id, "ground_truth": truth, "rank": rank}
            for sample_id, truth, rank in zip(
                self.sample_ids, self.truth_global_ids, self.canonical_ranks
            )
        ]


@dataclass(frozen=True)
class ModeAResult(ModeResult):
    """Mode A, which carries what no other mode can: the frozen oracle's own
    artifacts.

    The two metric families are kept apart in the type, not merged into one dict,
    because they answer different questions and only one of them is comparable
    across modes. `legacy_metrics` exists **only** here — B and C have no frozen
    oracle to be compared against, and a `legacy_mrr` on them would invite exactly
    the comparison that means nothing.
    """

    legacy_metrics: Dict[str, float]
    legacy_top_k_local: List[List[int]]
    """Per sample, the frozen oracle's observable artifact: subgraph-local column
    indices, truncated to `LEGACY_TRUNCATION_K`. This is what a comparison against
    `--save-predictions` output is made of."""

    def to_dict(self) -> Dict[str, Any]:
        report = super().to_dict()
        report["legacy_metrics"] = self.legacy_metrics
        return report

    def to_predictions(self) -> List[Dict[str, Any]]:
        """Per-sample rows in the frozen oracle's own artifact shape.

        `scripts/evaluate_model.py:508-519` writes `sample_id`, `ground_truth` and
        `predictions` — and mixes spaces while doing it: `ground_truth` is the
        **global** disease id straight from the samples file, while `predictions`
        are **subgraph-local** column indices rendered as strings. That is
        reproduced here rather than tidied, because the only purpose of this
        artifact is to be diffed against that one, and a tidier shape would not
        diff.
        """
        return [
            {
                "sample_id": sample_id,
                "ground_truth": truth,
                "predictions": [str(index) for index in row],
            }
            for sample_id, truth, row in zip(
                self.sample_ids, self.truth_global_ids, self.legacy_top_k_local
            )
        ]


@dataclass
class _SamplerEvidence:
    """What the sampler actually produced, as opposed to what the manifest claims.

    The manifest records the **configuration** — negatives requested, hop count,
    neighbour limits. That is a statement of intent. These are observations from
    the batches that were really scored, and the pair is only worth anything
    together: a manifest claiming 1000 negatives beside an observed universe of 40
    disease columns says the run did not measure what it says it measured.

    Counters and one summary method. Not telemetry: nothing is emitted, sampled or
    aggregated over time, and the whole thing lives and dies inside one run.
    """

    n_batches: int = 0
    candidate_counts: List[int] = field(default_factory=list)
    max_subgraph_nodes: Dict[str, int] = field(default_factory=dict)
    negatives_seen: bool = False
    negatives_drawn: int = 0
    negatives_within_sample_duplicates: int = 0
    _negative_global_ids: set = field(default_factory=set)

    def observe(self, batch_data: Dict[str, Any], n_candidates: int) -> None:
        self.n_batches += 1
        self.candidate_counts.append(int(n_candidates))

        for node_type, features in batch_data["subgraph_x_dict"].items():
            size = int(features.size(0))
            if size > self.max_subgraph_nodes.get(node_type, 0):
                self.max_subgraph_nodes[node_type] = size

        negatives_local = batch_data["batch"].get("negative_disease_ids")
        if negatives_local is None:
            return
        self.negatives_seen = True

        # The batch dict is remapped to subgraph-local indices before it is
        # handed over (`src/kg/data_loader.py:965-970`), so a global count has to
        # translate first — with the same function the ranking path uses, not a
        # second index expression.
        globals_2d = to_global_ids(
            batch_data["original_indices"]["disease"].to(negatives_local.device),
            negatives_local,
        )
        self.negatives_drawn += int(globals_2d.numel())
        for row in globals_2d.tolist():
            self.negatives_within_sample_duplicates += len(row) - len(set(row))
            self._negative_global_ids.update(row)

    def summary(self) -> Dict[str, Any]:
        counts = self.candidate_counts
        unique = len(self._negative_global_ids)
        return {
            "n_batches": self.n_batches,
            "candidate_columns": {
                "min": min(counts) if counts else None,
                "max": max(counts) if counts else None,
                "mean": sum(counts) / len(counts) if counts else None,
            },
            "max_subgraph_nodes": dict(self.max_subgraph_nodes),
            "negative_sampling": {
                "observed": self.negatives_seen,
                "total_drawn": self.negatives_drawn,
                "unique_global_ids": unique,
                # Two different collisions, kept apart. A draw repeated inside one
                # patient's negative set is the sampler failing to deduplicate
                # (`data_loader.py:665-671` appends without checking); the same id
                # reappearing across patients is expected and says nothing.
                "repeat_draws_across_run": self.negatives_drawn - unique,
                "repeat_draws_within_sample": self.negatives_within_sample_duplicates,
            },
        }


def _authoritative(ranks: List[int]) -> Dict[str, float]:
    """The metric family every mode is compared on.

    `mrr` is renamed `untruncated_mrr` because the legacy family also has an
    `mrr`, and two numbers under one name in adjacent columns of a report is how
    a truncated metric ends up quoted as an untruncated one.
    """
    from src.utils.metrics import RankingMetrics

    return {
        f"untruncated_{name}" if name == "mrr" else name: value
        for name, value in RankingMetrics()
        .compute_from_ranks(ranks, k_values=(1, 5, 10, 20, 50, 100))
        .items()
    }


def encode_full_graph(model: Any, graph_data: Dict[str, Any], device: Any) -> Dict[str, Any]:
    """Embed every node once, the way the deployed pipeline does.

    `src/inference/pipeline.py:_precompute_node_embeddings` is the same three
    operations — move the tensors to the device, one forward under `no_grad`,
    keep the result. It is not called directly because reaching it means
    constructing a `DiagnosisPipeline`, which loads a knowledge-graph object, a
    shortest-path table and the path-finding and explanation machinery that modes
    B and C do not use. An integration test asserts that this produces what the
    pipeline caches, so "the same three operations" is checked rather than
    asserted in a comment.
    """
    import torch as _torch

    x_dict = {k: v.to(device) for k, v in graph_data["x_dict"].items()}
    edge_index_dict = {k: v.to(device) for k, v in graph_data["edge_index_dict"].items()}
    model.eval()
    with _torch.no_grad():
        return model(x_dict, edge_index_dict)


def _score_from_full_graph(
    embeddings: Dict[str, Any],
    batch_data: Dict[str, Any],
    global_disease_ids: Tensor,
    device: Any,
) -> "tuple":
    """Mode B's score matrix for one batch, and the candidate tensor it used.

    Returns the candidate tensor it was handed, so the caller can check that it
    is the same object Mode A scored rather than trusting the arrangement of the
    code.

    **Both index spaces appear here and they must not be mixed.** The batch dict
    is remapped to subgraph-local indices before it is handed over
    (`src/kg/data_loader.py:945-970`), while these embeddings are indexed
    globally. Phenotype ids are therefore translated back through the same
    local-to-global map the disease path uses. Padded positions are clamped to a
    valid row and then discarded by the mask, exactly as the subgraph path does —
    they contribute nothing either way, and clamping keeps the gather in bounds.
    """
    from src.inference.scoring import cosine_score_matrix, masked_mean_pool

    phenotype_emb = embeddings["phenotype"]
    disease_emb = embeddings["disease"]
    batch = batch_data["batch"]

    local_phenotypes = batch["phenotype_ids"]
    phenotype_map = batch_data["original_indices"]["phenotype"]
    safe_local = local_phenotypes.clamp(min=0, max=phenotype_map.numel() - 1)
    phenotype_ids = to_global_ids(phenotype_map, safe_local.reshape(-1)).reshape(
        local_phenotypes.shape
    ).to(device)
    mask = batch["phenotype_mask"].to(device)

    valid = phenotype_ids.clamp(min=0, max=phenotype_emb.size(0) - 1)
    patient_phenotypes = phenotype_emb.to(device)[valid.reshape(-1)].reshape(
        phenotype_ids.size(0), phenotype_ids.size(1), -1
    )
    patients = masked_mean_pool(patient_phenotypes, mask)
    candidates = disease_emb.to(device)[global_disease_ids.to(device).long()]
    return cosine_score_matrix(patients, candidates), global_disease_ids


def _assert_cohort_is_intact(
    manifest: MeasurementManifest,
    n_legacy_rows: int,
    n_sample_ids: int,
    n_canonical_ranks: int,
    n_absent: int,
) -> None:
    """Refuse to report a number computed over a cohort nobody chose.

    Every way a Mode A run can quietly measure fewer samples than it claims —
    a skipped batch, a dropped last batch, a truth outside the candidate set —
    ends here rather than in a plausible metric. The failures are separated
    because they have different causes and a single "count mismatch" message
    would send the reader to the wrong place.
    """
    if manifest.n_samples <= 0:
        raise ValueError(
            f"manifest claims {manifest.n_samples} samples; there is nothing to "
            "measure and every metric would be fabricated"
        )
    if n_absent:
        raise ValueError(
            f"{n_absent} ground truth(s) were absent from the candidate set. In "
            "Mode A the truth is one of the subgraph's own seed nodes, so this is "
            "impossible unless the harness is wrong — the candidate universe, the "
            "id translation or the batch wiring. It is not a property of the model "
            "and must not be reported as one"
        )
    if not (n_legacy_rows == n_sample_ids == n_canonical_ranks == manifest.n_samples):
        raise ValueError(
            "cohort shrinkage: the run produced "
            f"{n_legacy_rows} legacy rows, {n_sample_ids} sample ids and "
            f"{n_canonical_ranks} canonical ranks for {manifest.n_samples} declared "
            "samples. A metric over a subset of the declared cohort is not the "
            "metric the manifest describes"
        )


def run_mode_a(
    model: Any,
    dataloader: Iterable[Dict[str, Any]],
    manifest: MeasurementManifest,
    device: Optional[Any] = None,
) -> ModeAResult:
    """Mode A alone. See `run_modes_ab`, of which this is the one-mode case."""
    result_a, _ = run_modes_ab(model, dataloader, manifest, device=device)
    return result_a


def run_modes_ab(
    model: Any,
    dataloader: Iterable[Dict[str, Any]],
    manifest_a: MeasurementManifest,
    manifest_b: Optional[MeasurementManifest] = None,
    full_graph_embeddings: Optional[Dict[str, Any]] = None,
    device: Optional[Any] = None,
) -> "tuple":
    """Score a cohort exactly as the legacy evaluator does, and report honestly.

    Per batch: forward the subgraph, pool the patient's phenotypes with the
    padding mask, and cosine-score against the subgraph's disease embeddings —
    the same three operations, through the shared primitives rather than a second
    copy of them.

    Then two rankings from the one score matrix:

      - `legacy_ranking` in **local** space, truncated to `LEGACY_TRUNCATION_K`,
        which is the only representation the frozen oracle emits and therefore the
        only one a calibration can compare;
      - `canonical_ranking` in **global** space, untruncated, which is what modes
        A, B, C and D are compared with each other on.

    A ground truth absent from the candidate set is a **hard failure**, not a
    reported statistic. In Mode A the truth is one of the subgraph's own seeds
    (`src/kg/data_loader.py:916-926`), so absence is impossible unless the harness
    is wrong — and a report that quietly drops the affected samples would answer a
    question about a cohort nobody chose. The same applies to any shrinkage:
    legacy rows, canonical ranks and `manifest.n_samples` must all agree.

    **Mode B rides the same traversal, deliberately.** B is defined as *the same
    candidates as A, scored from full-graph embeddings*, so A→B is only a
    measurement of encoder scope if the candidate sets are genuinely identical.
    Running B separately — even from the same seed — would not establish that:
    the calibration evidence available compares the aggregate MRR and the local
    top-20, and two runs can agree on both while their candidate universes differ
    **outside** the top 20. Here both modes are handed the *same tensor*, and the
    identity is asserted rather than left as a property of how the code happens to
    be arranged today.

    Pass `manifest_b` and `full_graph_embeddings` to get B; omit them for A alone.
    Returns `(mode_a, mode_b_or_None)`.
    """
    import torch as _torch

    from src.inference.scoring import cosine_score_matrix, masked_mean_pool
    from src.utils.metrics import RankingMetrics

    device = _torch.device(device) if device is not None else _torch.device("cpu")
    model.eval()

    want_b = manifest_b is not None
    if want_b and full_graph_embeddings is None:
        raise ValueError(
            "Mode B needs full_graph_embeddings; without them it would fall back "
            "to the subgraph encoder and silently be Mode A under another name"
        )

    legacy_top_k: List[List[int]] = []
    legacy_truth_local: List[int] = []
    sample_ids: List[str] = []
    truth_global_ids: List[int] = []
    canonical_ranks: List[int] = []
    canonical_ranks_b: List[int] = []
    evidence = _SamplerEvidence()
    absent = 0
    absent_b = 0

    with _torch.no_grad():
        for batch_data in dataloader:
            batch = batch_data["batch"]
            subgraph_x = {k: v.to(device) for k, v in batch_data["subgraph_x_dict"].items()}
            subgraph_edges = {
                k: v.to(device) for k, v in batch_data["subgraph_edge_index_dict"].items()
            }

            node_embeddings = model(subgraph_x, subgraph_edges)
            disease_emb = node_embeddings.get("disease")
            phenotype_emb = node_embeddings.get("phenotype")
            if disease_emb is None or phenotype_emb is None:
                raise ValueError(
                    "the model produced no disease or phenotype embeddings; Mode A "
                    "cannot score this batch and must not silently skip it"
                )

            phenotype_ids = batch["phenotype_ids"].to(device)
            disease_ids_local = batch["disease_ids"].to(device)
            mask = batch["phenotype_mask"].to(device)

            # **The clamp is oracle parity, not defensiveness, and it stays.**
            # `diagnosis_collate_fn` pads phenotype ids with -1 and `_remap_indices`
            # leaves those positions at -1, so the frozen evaluator clamps before
            # gathering and reads row 0 for every padded slot. Indexing with -1
            # instead would read the *last* row through Python negative indexing —
            # a different operation, cancelled by the mask only for finite values,
            # and a NaN survives multiplication by zero. Mode A may not swap the
            # oracle's index semantics for an assumed downstream cancellation.
            #
            # This is the opposite decision from Mode C's, and deliberately so:
            # there every id is a real value under a true mask, so clamping would
            # score a different patient. Here every out-of-range value is padding
            # the mask already discards. Both go when the frozen oracle goes.
            valid = phenotype_ids.clamp(min=0, max=phenotype_emb.size(0) - 1)
            patient_phenotypes = phenotype_emb[valid.reshape(-1)].reshape(
                phenotype_ids.size(0), phenotype_ids.size(1), -1
            )
            patients = masked_mean_pool(patient_phenotypes, mask)
            scores = cosine_score_matrix(patients, disease_emb)

            global_ids = batch_data["original_indices"]["disease"].to(device)
            evidence.observe(batch_data, n_candidates=scores.size(1))

            local_order = legacy_ranking(scores)
            legacy_top_k.extend(local_order[:, :LEGACY_TRUNCATION_K].tolist())
            legacy_truth_local.extend(disease_ids_local.tolist())
            sample_ids.extend(batch["patient_ids"])

            canonical = canonical_ranking(scores, global_ids)
            truth_global = to_global_ids(global_ids, disease_ids_local)
            truth_global_ids.extend(truth_global.tolist())
            for rank in ranks_of_truth(canonical, truth_global):
                if rank is None:
                    absent += 1
                else:
                    canonical_ranks.append(rank)

            if not want_b:
                continue

            # Mode B: the SAME candidates, encoded over the whole graph instead
            # of this batch's subgraph. `global_ids` is the one tensor both modes
            # index by, and the assertion below says so in code — "they share a
            # variable" is a structural claim, and structural claims decay.
            b_scores, b_ids = _score_from_full_graph(
                full_graph_embeddings, batch_data, global_ids, device
            )
            if b_ids is not global_ids:
                raise AssertionError(
                    "Mode B scored a different candidate tensor from Mode A; A->B "
                    "would then measure encoder scope and candidate construction "
                    "together, which is not what it is for"
                )
            for rank in ranks_of_truth(canonical_ranking(b_scores, b_ids), truth_global):
                if rank is None:
                    absent_b += 1
                else:
                    canonical_ranks_b.append(rank)

    _assert_cohort_is_intact(
        manifest=manifest_a,
        n_legacy_rows=len(legacy_top_k),
        n_sample_ids=len(sample_ids),
        n_canonical_ranks=len(canonical_ranks),
        n_absent=absent,
    )
    if want_b:
        _assert_cohort_is_intact(
            manifest=manifest_b,
            n_legacy_rows=len(sample_ids),
            n_sample_ids=len(sample_ids),
            n_canonical_ranks=len(canonical_ranks_b),
            n_absent=absent_b,
        )

    result_a = ModeAResult(
        manifest=manifest_a,
        legacy_metrics={
            # Through `RankingMetrics.mean_reciprocal_rank` — the same call the
            # frozen evaluator reaches via `compute_all` (`evaluate_model.py:285,
            # 366`) — over the same truncated lists it builds, as ids rather than
            # as the strings it renders them to. A second implementation of
            # `1/rank` here would have to be trusted to agree with that one; this
            # cannot disagree with it.
            f"legacy_mrr_truncated_at_{LEGACY_TRUNCATION_K}":
                RankingMetrics().mean_reciprocal_rank(legacy_top_k, legacy_truth_local),
        },
        authoritative_metrics=_authoritative(canonical_ranks),
        n_ranked=len(canonical_ranks),
        n_ground_truth_absent=absent,
        sampler_evidence=evidence.summary(),
        sample_ids=sample_ids,
        truth_global_ids=truth_global_ids,
        canonical_ranks=canonical_ranks,
        legacy_top_k_local=legacy_top_k,
    )
    if not want_b:
        return result_a, None

    result_b = ModeResult(
        manifest=manifest_b,
        authoritative_metrics=_authoritative(canonical_ranks_b),
        n_ranked=len(canonical_ranks_b),
        n_ground_truth_absent=absent_b,
        # The same observation, because it is the same traversal. Recording it on
        # both is what lets a reader check that claim from the artifacts alone.
        sampler_evidence=evidence.summary(),
        sample_ids=sample_ids,
        truth_global_ids=truth_global_ids,
        canonical_ranks=canonical_ranks_b,
    )
    return result_a, result_b


def assert_constructions_agree(legacy_model: Any, production_model: Any) -> None:
    """Refuse to read A→B as encoder scope unless the two models are the same model.

    Mode A builds through the frozen evaluator's loader; modes B and C build
    through production's. If those disagree — a different conv type recovered, a
    different layer count, different weights — then A→B is encoder scope *plus*
    architecture resolution, and the ladder's first rung measures two things at
    once while reporting one.

    A difference here is a finding, not a nuisance: it means the served model and
    the historically evaluated model were never the same, which is worth knowing
    before any number is published.
    """
    import torch as _torch

    differences: List[str] = []
    left, right = getattr(legacy_model, "metadata", None), getattr(production_model, "metadata", None)
    if left != right:
        differences.append(f"metadata: legacy {left} vs production {right}")

    left_state = legacy_model.state_dict()
    right_state = production_model.state_dict()
    only_legacy = sorted(set(left_state) - set(right_state))
    only_production = sorted(set(right_state) - set(left_state))
    if only_legacy or only_production:
        differences.append(
            f"parameter names: {len(only_legacy)} only in legacy {only_legacy[:3]}, "
            f"{len(only_production)} only in production {only_production[:3]}"
        )

    mismatched = [
        name for name in sorted(set(left_state) & set(right_state))
        if left_state[name].shape != right_state[name].shape
        or not _torch.equal(left_state[name].cpu(), right_state[name].cpu())
    ]
    if mismatched:
        differences.append(f"{len(mismatched)} parameter tensors differ, e.g. {mismatched[:3]}")

    if differences:
        raise ValueError(
            "the legacy and production model constructions disagree, so A→B would "
            "measure encoder scope and architecture resolution together:\n  "
            + "\n  ".join(differences)
        )


def _assert_ids_in_range(samples: Any, n_phenotypes: int, n_diseases: int) -> None:
    """Every id a sample carries must index the graph it is being scored against.

    Out of range is fatal and names the patient and the value. The alternative —
    clamping, dropping, or substituting — turns a data error into a plausible
    rank for a patient who was never scored on their own phenotypes, and nothing
    downstream can tell that apart from a real result.

    A sample with **no** phenotypes is refused for the same reason: pooling over
    an empty mask yields the zero vector, whose cosine against every disease is
    zero, whose ranking is then an arbitrary tie-break over the whole graph.
    """
    for sample in samples:
        if not sample.phenotype_ids:
            raise ValueError(
                f"sample {sample.patient_id!r} has no phenotypes; it cannot be "
                "scored, and pooling nothing would rank the whole graph by a "
                "tie-break"
            )
        for phenotype_id in sample.phenotype_ids:
            if not 0 <= phenotype_id < n_phenotypes:
                raise ValueError(
                    f"sample {sample.patient_id!r} has phenotype id {phenotype_id}, "
                    f"outside the graph's {n_phenotypes} phenotype nodes. Refusing "
                    "rather than clamping: a clamped id scores a different patient"
                )
        if not 0 <= sample.disease_id < n_diseases:
            raise ValueError(
                f"sample {sample.patient_id!r} has ground-truth disease id "
                f"{sample.disease_id}, outside the graph's {n_diseases} disease "
                "nodes. Its rank would be meaningless"
            )


def run_mode_c(
    full_graph_embeddings: Dict[str, Any],
    samples: Iterable[Any],
    manifest: MeasurementManifest,
    device: Optional[Any] = None,
    batch_size: int = 32,
) -> ModeResult:
    """Score every patient against **every disease in the graph**.

    This is what the reference method does — *"we calculate a patient's
    similarity to all disease nodes in the KG at inference time"* (npj Digital
    Medicine 8:380, 2025, Methods) — and what nothing in this system did before.

    **No subgraph sampler and no dataloader.** Mode C's candidate universe is all
    diseases, so there is nothing for the sampler to select and no reason to pay
    for it; and running it would consume the random draws that make A and B
    reproducible without using any of them. Patients come straight from the
    samples, which is where the dataloader gets them too.

    Padding is done here rather than through `diagnosis_collate_fn` for the same
    reason: that function builds items through `DiagnosisDataset`, which draws
    negative samples this mode discards. Eight lines of `pad` and `stack` against
    a dependency that consumes randomness for nothing is not a close call.

    Absence is impossible by construction — the truth is a disease, and every
    disease is a candidate — so a non-zero absence count means the ids are wrong,
    and `_assert_cohort_is_intact` treats it as fatal.

    **Every real id is validated before it is used, and out-of-range is fatal.**
    Mode A clamps, and is right to: its out-of-range values are the dataloader's
    `-1` padding, already excluded by a `False` mask, and clamping only keeps the
    gather in bounds. Here every id is a real value from a real patient with
    `mask=True`, so the same clamp would score phenotype `-3` as phenotype 0 and
    phenotype 999999 as the last node in the graph — a plausible patient, a
    plausible rank, and a wrong one. Padding still uses a safe value under a
    `False` mask; **real values are never clamped, dropped or substituted.**
    """
    import torch as _torch
    import torch.nn.functional as _F

    from src.inference.scoring import cosine_score_matrix, masked_mean_pool

    device = _torch.device(device) if device is not None else _torch.device("cpu")
    phenotype_emb = full_graph_embeddings["phenotype"].to(device)
    disease_emb = full_graph_embeddings["disease"].to(device)
    all_disease_ids = _torch.arange(disease_emb.size(0), device=device)

    materialised = list(samples)
    sample_ids: List[str] = []
    truth_global_ids: List[int] = []
    canonical_ranks: List[int] = []
    absent = 0
    candidate_counts: List[int] = []

    with _torch.no_grad():
        for start in range(0, len(materialised), batch_size):
            chunk = materialised[start:start + batch_size]
            _assert_ids_in_range(chunk, phenotype_emb.size(0), disease_emb.size(0))
            widest = max(len(sample.phenotype_ids) for sample in chunk)
            ids, masks = [], []
            for sample in chunk:
                row = _torch.tensor(sample.phenotype_ids, dtype=_torch.long)
                pad = widest - row.numel()
                # Padding is 0 — a valid row that the mask discards. Real values
                # reached here already validated, so nothing below alters one.
                ids.append(_F.pad(row, (0, pad)))
                masks.append(
                    _torch.cat([
                        _torch.ones(row.numel(), dtype=_torch.bool),
                        _torch.zeros(pad, dtype=_torch.bool),
                    ])
                )
            phenotype_ids = _torch.stack(ids).to(device)
            mask = _torch.stack(masks).to(device)

            # No clamp: every entry is either a real id already validated above or
            # a padded 0 the mask removes. A clamp here is the defect this mode was
            # corrected for — it would score a different patient rather than fail.
            patient_phenotypes = phenotype_emb[phenotype_ids.reshape(-1)].reshape(
                phenotype_ids.size(0), phenotype_ids.size(1), -1
            )
            patients = masked_mean_pool(patient_phenotypes, mask)
            scores = cosine_score_matrix(patients, disease_emb)
            candidate_counts.append(int(scores.size(1)))

            truth = _torch.tensor([s.disease_id for s in chunk], dtype=_torch.long, device=device)
            ranked = canonical_ranking(scores, all_disease_ids)
            sample_ids.extend(s.patient_id for s in chunk)
            truth_global_ids.extend(truth.tolist())
            for rank in ranks_of_truth(ranked, truth):
                if rank is None:
                    absent += 1
                else:
                    canonical_ranks.append(rank)

    _assert_cohort_is_intact(
        manifest=manifest,
        n_legacy_rows=len(sample_ids),
        n_sample_ids=len(sample_ids),
        n_canonical_ranks=len(canonical_ranks),
        n_absent=absent,
    )

    return ModeResult(
        manifest=manifest,
        authoritative_metrics=_authoritative(canonical_ranks),
        n_ranked=len(canonical_ranks),
        n_ground_truth_absent=absent,
        sampler_evidence={
            "n_batches": len(candidate_counts),
            "candidate_columns": {
                "min": min(candidate_counts) if candidate_counts else None,
                "max": max(candidate_counts) if candidate_counts else None,
                "mean": (sum(candidate_counts) / len(candidate_counts))
                if candidate_counts else None,
            },
            "max_subgraph_nodes": {},
            # No sampler ran, and saying so is the point: an empty structure here
            # would read as "not recorded" rather than "not applicable".
            "negative_sampling": {"observed": False, "reason": "mode C scores every disease"},
        },
        sample_ids=sample_ids,
        truth_global_ids=truth_global_ids,
        canonical_ranks=canonical_ranks,
    )
