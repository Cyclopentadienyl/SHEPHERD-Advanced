"""
Mode C: every patient scored against every disease in the graph.
=================================================================
Module: src/evaluation/measurement.py

The formal Test's measurement (`docs/working/PLAN_TEST_RESULTS.md`, "This is
Mode C"): full-graph embeddings from the production model, a masked mean of each
patient's phenotypes, raw cosine against every disease, and one ranking.

**The ranking is deterministic.** Scores are ranked descending and ties break by
ascending global disease id, so the order does not depend on the machine, the
chunking or the order the candidates arrived in (`canonical_ranking`).

What the ranking functions below do **not** do: load data, run a model, or decide
what to do about a ground truth that is absent from the candidate set. They rank.
Absence is decided in `run_mode_c`, where it is fatal: every disease is a
candidate, so an absent truth means the ids are wrong.

Dependencies: torch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch import Tensor

# These live in `src.utils.run_environment`, which the task measurement reads
# too; imported here, where Mode C's callers have always found them.
from src.utils.run_environment import AutocastRegime, observe_autocast_regime

__all__ = [
    "AutocastRegime",
    "EncodedGraph",
    "MeasurementManifest",
    "ModeResult",
    "RELATION_NAMES",
    "RELATION_STATES",
    "canonical_ranking",
    "expected_relations",
    "ranks_of_truth",
    "encode_full_graph",
    "run_mode_c",
]


def _require_integer_ids(ids: Tensor, name: str) -> None:
    """Identifiers must be integer tensors, and `bool` is not an integer here.

    ``Tensor.long()`` turns 1.7 into 1 and ``True`` into 1, so a float or boolean
    tensor wired in by mistake becomes a plausible index rather than an error —
    the same silent-wrong-answer shape as a float mask reaching `masked_mean_pool`.
    """
    if ids.dtype == torch.bool or torch.is_floating_point(ids) or torch.is_complex(ids):
        raise ValueError(f"{name} must be an integer tensor, not {ids.dtype}")


def canonical_ranking(scores: Tensor, global_disease_ids: Tensor) -> Tensor:
    """Rank by score descending, breaking ties by ascending global disease id.

    ``scores`` is ``(B, D)``, ``global_disease_ids`` is ``(D,)``; returns
    ``(B, D)`` of global ids, best first.

    Given identical score values and global ids, its order — including its tie
    rule — is independent of the order the candidates arrived in and of how the
    cohort was chunked, so two Mode C runs over the same scores rank alike. It is
    *not* a claim that different hardware produces identical scores; if the
    scores differ the ranking may differ, and that is a property of the scores,
    not of this function.

    **Non-finite scores are rejected.** A ``NaN`` sorts unpredictably and would
    turn an invalid score into a plausible rank, and from there into a plausible
    mean rank and MRR.

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


def ranks_of_truth(ranked_global_ids: Tensor, truth_global_ids: Tensor) -> List[Optional[int]]:
    """1-based rank of each row's ground truth, or ``None`` where it is absent.

    ``ranked_global_ids`` is ``(B, D)``; ``truth_global_ids`` is ``(B,)``.

    **``None`` rather than a sentinel rank.** A ground truth outside the candidate
    set has no rank, and encoding that as a large integer would let it flow into a
    mean as though it were a measurement. `RankingMetrics.compute_from_ranks`
    refuses to invent a value for an empty cohort for the same reason; this is the
    same refusal one level up. What absence *means* is the caller's decision: in
    Mode C, which ranks every disease, it is impossible and therefore fatal.

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


#: Where `torch.compile`'s wrapper class is exported. The package-level name is
#: tried first because it is the less private of the two. Both resolve to the same
#: class in the pinned torch, so this buys compatibility with a release that moves
#: one of them, not two different answers.
_WRAPPER_IMPORT_PATHS = (
    ("torch._dynamo", "OptimizedModule"),
    ("torch._dynamo.eval_frame", "OptimizedModule"),
)


def _resolve_wrapper_classes() -> tuple:
    """Every wrapper class that could be imported. Empty when none could.

    **All of them, not the first.** Returning on the first path that resolves
    would answer `False` for a model wrapped by a class only the second path
    exports — which is the exact situation two paths exist for. They are the same
    class today; aggregating is what makes the claim true rather than incidental.
    """
    classes: list = []
    for module_name, attribute in _WRAPPER_IMPORT_PATHS:
        try:
            candidate = getattr(__import__(module_name, fromlist=[attribute]), attribute)
        except Exception:  # noqa: BLE001 - a moved internal is not an answer
            continue
        if isinstance(candidate, type) and candidate not in classes:
            classes.append(candidate)
    return tuple(classes)


def observe_torch_compile_wrapper(model: Any) -> Optional[bool]:
    """Whether `model` **is a `torch.compile` wrapper object**. An execution-side
    observation of the object, and deliberately not more than that.

    **What this does not claim.** It does not say a compiled graph ran. Graph
    breaks, guard failures and `suppress_errors` all leave a wrapped model
    executing eagerly, so `True` here means "wrapped", never "the scores came out
    of fused kernels". The field it feeds is named `torch_compile_wrapped` for the
    same reason `cuda_executed` was renamed from `calibration_eligible`: a field
    whose name claims more than its probe can see gets read for the claim, not for
    the probe. Establishing actual compiled execution would need Dynamo counters
    and graph-break accounting, which backlog item D2 rules out and which nothing
    in this phase needs.

    Still worth recording. It remains an **execution** fact rather than a config
    value — `src/config/training_fields.py` carries a compile toggle and the
    Runtime Settings tab writes it, and a run configured to compile is not a run
    that was handed a compiled model.

    **isinstance is the only evidence, and the tri-state carries the rest:**

    | Situation | Result |
    |---|---|
    | no model supplied | `None` |
    | a wrapper class resolved and the model is one | `True` |
    | a wrapper class resolved and the model is not | `False` |
    | no wrapper class resolved | `None` |

    **There is no attribute fallback, and two earlier versions of this function
    are why.** The first promoted `hasattr(model, "_orig_mod")` to `True`, and the
    test written for it asserted that an arbitrary class carrying that attribute
    was a compile wrapper — freezing a false positive as a specification. The
    second required the class to be defined under `torch._dynamo` as well, which
    fails twice over: `startswith` also matches `torch._dynamoevil`, and
    `__module__` is assignable class metadata that any object can declare rather
    than provenance of how it was built. `_orig_mod` is a *necessary* marker of
    the known wrapper, never sufficient evidence of wrapper identity, and no
    rearrangement of weak signals makes it so. When the exact classes cannot be
    imported the honest answer is that nothing was observed.

    **This never raises.** It is the thing that says what the run was, so a moved
    torch internal or an object with unusual attribute behaviour must come back as
    `None` rather than as a traceback that kills the measurement for the sake of
    one field.
    """
    if model is None:
        return None

    classes = _resolve_wrapper_classes()
    if not classes:
        return None
    try:
        return isinstance(model, classes)
    except Exception:  # noqa: BLE001 - upholds the never-raises contract; a
        # custom `__instancecheck__` or `__class__` is the only way here, and an
        # ordinary object does not reach it.
        return None


@dataclass(frozen=True)
class EncodedGraph:
    """Full-graph embeddings **and the regime they were computed under**.

    The pair travels together because the second cannot be recovered from the
    first, and because observing it anywhere other than at the computation gives
    the wrong answer. Mode C scores from these embeddings but does not produce
    them, and its manifest is built afterwards — an earlier design observed the
    regime at manifest construction, which is a different context and would have
    labelled these embeddings with it.

    A frozen dataclass rather than a bare tuple: callers read its named fields,
    and `encoded.embeddings` says what a `[0]` does not. It travels **whole** into
    `run_mode_c`, which is what lets it verify that its manifest
    describes the regime these embeddings were actually computed under — splitting
    it at the caller left that association unverifiable.
    """

    embeddings: Dict[str, Any]
    regime: AutocastRegime


def assert_manifest_describes_regime(
    manifest: Any, device: Any, encoded: "EncodedGraph"
) -> None:
    """Refuse when the manifest's recorded regime is not the one now in force.

    **A consistency check, not an absence check.** The version this replaces
    refused *any* autocast, because `build_manifest` wrote `amp_enabled=False` as a
    literal and a run under autocast would have shifted every score while the
    artifact claimed fp32. That was a real ground — inability to produce an honest
    artifact — but a removable one: recording what applied removes it, and
    refusing instead cost the harness a legitimate research question, *"what is
    this mode's MRR under the AMP setting the deployment actually uses?"*

    What remains refused is narrower and is the case a single field genuinely
    cannot describe: numbers produced under one regime and recorded under
    another. For Mode C that includes embeddings computed in one context and
    scored in a different one — its manifest carries the **embedding** regime,
    so scoring under a different one is refused rather than averaged into a
    single claim.

    **Two associations, not one.** `encoded` is the full-graph embeddings Mode C
    scores from, and its regime is the one they were computed under. Checking
    only the current scoring context leaves the other half unverified: fp32
    embeddings scored under bfloat16 would pass, which is precisely the
    mixed-regime run this function exists to refuse. An earlier version took only
    the current context — and the Mode C test written for it asserted success on
    exactly that run, freezing the defect as a specification.

    The cost is one thread-local read. Nothing here touches a tensor or a
    device.
    """
    recorded = AutocastRegime(enabled=manifest.amp_enabled, dtype=manifest.amp_dtype)

    if encoded.regime != recorded:
        raise RuntimeError(
            f"the full-graph embeddings were computed under {encoded.regime} while "
            f"this manifest records {recorded}. Mode C scores numbers these "
            "embeddings produced, so a manifest describing a different regime "
            "describes a run that did not happen"
        )

    observed = observe_autocast_regime(device)
    if observed != recorded:
        raise RuntimeError(
            f"this measurement is running under {observed} while its manifest "
            f"records {recorded}. The artifact would describe a run that did not "
            "happen. Build the manifest in the same context the computation runs "
            "in — for Mode C that is the context the full-graph embeddings were "
            "computed in, which `EncodedGraph` carries for exactly this reason"
        )


#: Bumped when the metric set or any metric's definition changes: a name added or
#: removed, a denominator redefined, a `hits@K` membership change, an aggregation
#: rule change. Lives beside `_authoritative`, which is what it versions.
METRIC_SCHEMA_VERSION = 1

#: The relations a Mode C run checks on the bytes it read, by the names its
#: manifest records them under (contract §5; `PLAN_TEST_RESULTS.md` §6, phase 1,
#: step 2): R1 the graph files against the workspace manifest, R2 a generated
#: split's samples against the same manifest, R3 the checkpoint's recorded graph
#: against the graph read, R5 the workspace's dataset record, and `cohort_graph` a
#: registered cohort version's graph and dataset record against the ones read.
#:
#: **Recorded per relation because a report used to keep only R5's state.** R1 and
#: R3 refused or passed and left nothing behind, so a reader could not tell from
#: the report which relations had been checked on it.
RELATION_NAMES = ("R1", "R2", "R3", "R5", "cohort_graph")

#: The only states a written manifest holds for a relation. **There is no
#: "failed" and no "not checked":** a relation that applies and does not hold
#: refuses the run before a manifest exists, so a report can only say that a
#: relation was verified on its bytes or that it does not apply to its cohort.
RELATION_STATES = ("verified", "not_applicable")


def expected_relations(*, registered: bool) -> Dict[str, str]:
    """The relation states a run earns, by the kind of cohort it measured.

    A generated split is bound to the workspace manifest by its sample digests
    (R2) and has no cohort version to bind; a registered cohort version is bound
    to the graph by its own manifest (`cohort_graph`) and was cut by no allocation
    of this workspace, so R2 does not apply. R1, R3 and R5 apply to both.

    A fresh dict each call, so a caller that keeps or edits it cannot change what
    the next manifest is checked against.
    """
    states = {name: "verified" for name in RELATION_NAMES}
    states["R2" if registered else "cohort_graph"] = "not_applicable"
    return states


@dataclass(frozen=True)
class MeasurementManifest:
    """What has to be recorded for a number to mean anything later.

    Not a schema framework — a frozen dataclass and `dataclasses.asdict`. Every
    field is something that changes what the measurement *is*, so omitting one
    would make two runs incomparable without saying so.

    `batch_size` is here as **semantics, not performance**: it is the chunk
    Mode C scores at once, and it can move the numbers at floating-point level.
    """

    mode: str
    split: str
    metric_schema_version: int
    """Which set of metrics, under which definitions, `authoritative_metrics` holds.

    **Separate from `canonical_tie_policy_version`, which cannot stand in for it.**
    The tie policy versions how equal scores become ranks. It says nothing about
    which metric names are emitted, what each denominator counts, which K values
    `hits@K` covers, or how per-sample values are aggregated -- all of which change
    what a number means while leaving the tie policy untouched. Bump this whenever
    any of those changes; two records under one version must be comparable."""
    cohort_kind: str
    """Whether this cohort came from this project's generator or from outside.

    `generated` is one of the generator's own splits (`train`, `val`),
    disease-disjoint by construction, and carries a `split_manifest` digest
    below; `supplied` is a registered cohort version (`cohort_version` below),
    an institutional or external patient set that was never cut from this
    disease universe, so its overlap with training is an open measurement (R10)
    rather than a contract.

    **Stated on the manifest rather than inferred downstream from whether a
    `split_manifest` digest is present.** Absence of a role could mean several
    things later; this field means one, and `__post_init__` refuses it unless it
    agrees with `cohort_version`."""
    n_samples: int
    # what was ranked, and in what chunks
    candidate_construction: str
    batch_size: int
    # scoring and ranking
    score_semantics: str
    model_construction: str
    """Which builder made the scored model: `build_shepherd_model`, the one the
    deployed pipeline uses. Recorded so a reader does not have to assume it."""
    canonical_tie_policy_version: str
    # artifacts
    checkpoint_path: str
    data_dir: str
    graph_fingerprint: Dict[str, Any]
    artifact_digests: Dict[str, str]
    """Raw SHA-256 of every file the measurement consumed, keyed by role, plus
    `kg`.

    `graph_fingerprint` above is a **structural** identity — node types, counts,
    feature dimensions. Two different checkpoints trained on the same graph share
    it, and so do two different sample files of the same shape. A path is not an
    identity either: `checkpoints/best.pt` names a different file every time
    training improves. These digests are what let a number be traced back to the
    exact bytes that produced it.

    They are the digests of the bytes the run parsed (contract M2.1, S7), and
    the `kg` role is not a file it read: measurement never parses `kg.json`, so
    `kg` is the digest the workspace's manifest binds for the source graph (M2.1
    decision 2)."""
    dataset_record: Dict[str, Any]
    """The workspace's dataset record as verified (contract M3b): its binding
    state and its origin, kept apart, the digest of its bytes, the sources it
    names and what it says it does not establish.

    **A number over a synthetic workspace says so here.** The binding is always
    `verified` in a manifest that exists, since anything else refuses the run;
    `origin` is what tells a synthetic acceptance run's number from one over a
    graph built from real files."""
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
    amp_dtype: Optional[str]
    """The autocast dtype in force, or `None` when AMP is off.

    The boolean alone cannot say what a run computed in. `amp_enabled=False` with
    no dtype leaves a reader unable to tell an fp32 run from one whose dtype was
    simply not recorded, and a float16 and a bfloat16 run are different numbers.

    **Observed, not assumed.** An earlier revision wrote both fields as literals
    and said "no traversal here opens an autocast context, therefore False". No
    traversal here does open one — but a *caller* can, and the literals then
    described a run that did not happen. They now carry what
    `observe_autocast_regime` read at the computation that produced the
    numbers: the full-graph embedding call. `assert_manifest_describes_regime`
    refuses a run whose regime differs from what its manifest records.

    **A harness-boundary regime.** An autocast context opened inside a model is
    not observable from these boundaries and is not claimed."""

    torch_compile_wrapped: Optional[bool]
    """Whether the model that ran **was a `torch.compile` wrapper object** —
    observed, not configured. `None` means not observed, which is a different
    claim from `False`.

    Named for what the probe can see. It does **not** say a compiled graph
    executed: graph breaks and eager fallback leave a wrapped model running
    eagerly. See `observe_torch_compile_wrapper`."""

    deterministic_algorithms: bool
    cudnn_deterministic: Optional[bool]
    cudnn_benchmark: Optional[bool]
    # which cohort version, the relations checked, R10, and the environment (M4)
    cohort_version: Optional[Dict[str, Any]]
    """The registered cohort version this run scored, or `None` for a generated
    split.

    For a version: its label and role, the digest of its manifest as this run
    read it (`manifest_sha256`, the ledger's cohort identity), the source, samples
    and dataset record digests and the graph digests it was mapped against, and
    its mapping policy and counts with their denominators. **Aggregates only:**
    no case id, which stays in the version on the operator's machine (D7)."""
    relations: Dict[str, str]
    """Each relation in `RELATION_NAMES` and its state on this run's bytes, one of
    `RELATION_STATES`. Fixed by the kind of cohort (`expected_relations`); any
    other combination is refused when the manifest is built."""
    r10: Optional[Dict[str, Any]]
    """How the cohort relates to the training and validation inputs the
    checkpoint records (contract §5.4): two scopes, two labels each, with the
    evidence that decided each scope. `None` for a generated split, whose
    relation to training is the allocation's."""
    environment: Dict[str, Any]
    """The facts a run was produced under (contract §5.3), from the one collector
    training uses: `facts` is `collect_environment` for the device in effect,
    `code` is `code_revision` of the repository, dirty state included.

    **Recorded, never enforced** (R11). The facts that can move a number are
    semantic fields above; the GPU model, driver, cuDNN, TF32 and PyG facts inform
    an investigation of two disagreeing runs and do not key them."""

    def __post_init__(self) -> None:
        """Refuse a manifest whose relation states are not the ones its run earned.

        **So a report cannot carry a state no check produced.** The states are
        decided by the kind of cohort, not passed through from whatever a caller
        assembled: a generated split with no cohort version, or a supplied cohort
        with one. A supplied cohort without a version is refused here because
        nothing binds its samples to the graph scored, so its report could not
        state that relation at all.
        """
        registered = self.cohort_version is not None
        if registered and self.cohort_kind != "supplied":
            raise ValueError(
                f"a registered cohort version is a supplied cohort; this manifest "
                f"records cohort_kind {self.cohort_kind!r} beside one"
            )
        if not registered and self.cohort_kind != "generated":
            raise ValueError(
                f"a {self.cohort_kind!r} cohort with no registered cohort version: "
                "nothing binds its samples to the graph scored, so no report can "
                "state that relation. Measure a supplied cohort as a version "
                "registered against this graph (--cohort LABEL --registry DIR)"
            )
        expected = expected_relations(registered=registered)
        if not isinstance(self.relations, dict) or self.relations != expected:
            raise ValueError(
                f"relation states {self.relations!r} are not the ones a "
                f"{'registered cohort version' if registered else 'generated split'} "
                f"run earns ({expected}); a relation that does not hold refuses the "
                "run, so no other state is written"
            )
        if registered != (self.r10 is not None):
            raise ValueError(
                "R10 is recorded for a registered cohort version and only for one; "
                f"this manifest has cohort_version {'set' if registered else 'None'} "
                f"and r10 {'None' if self.r10 is None else 'set'}"
            )


@dataclass(frozen=True)
class ModeResult:
    """One measurement run.

    `canonical_ranks` is aligned with `sample_ids`, so a run can be read per
    patient and not only in aggregate — an aggregate that moved by a little can
    hide a cohort where half the ranks improved and half collapsed.
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
        """Per-sample ranks, one row per patient in the cohort's order."""
        return [
            {"sample_id": sample_id, "ground_truth": truth, "rank": rank}
            for sample_id, truth, rank in zip(
                self.sample_ids, self.truth_global_ids, self.canonical_ranks
            )
        ]


def _authoritative(ranks: List[int]) -> Dict[str, float]:
    """The metric family a Mode C report carries.

    `mrr` is reported as `untruncated_mrr` so it cannot be read as the Trainer's
    `val_mrr`, which is computed over a top-20 window: two numbers under one name
    is how a truncated metric ends up quoted as an untruncated one.
    """
    from src.utils.metrics import RankingMetrics

    return {
        f"untruncated_{name}" if name == "mrr" else name: value
        for name, value in RankingMetrics()
        .compute_from_ranks(ranks, k_values=(1, 5, 10, 20, 50, 100))
        .items()
    }


def encode_full_graph(model: Any, graph_data: Dict[str, Any], device: Any) -> EncodedGraph:
    """Embed every node once, the way the deployed pipeline does.

    `src/inference/pipeline.py:_precompute_node_embeddings` is the same three
    operations — move the tensors to the device, one forward under `no_grad`,
    keep the result. It is not called directly because reaching it means
    constructing a `DiagnosisPipeline`, which loads a knowledge-graph object, a
    shortest-path table and the path-finding and explanation machinery that
    Mode C does not use. An integration test asserts that this produces what the
    pipeline caches, so "the same three operations" is checked rather than
    asserted in a comment.
    """
    import torch as _torch

    x_dict = {k: v.to(device) for k, v in graph_data["x_dict"].items()}
    edge_index_dict = {k: v.to(device) for k, v in graph_data["edge_index_dict"].items()}
    model.eval()
    # Observed **here**, inside the function that performs the forward, so a caller
    # who wraps this call in an autocast context is described by it. Observing at
    # the caller — or worse, at manifest construction later — reads a different
    # context and would label these embeddings with it.
    regime = observe_autocast_regime(device)
    with _torch.no_grad():
        return EncodedGraph(embeddings=model(x_dict, edge_index_dict), regime=regime)


def _assert_cohort_is_intact(
    manifest: MeasurementManifest,
    n_sample_ids: int,
    n_canonical_ranks: int,
    n_absent: int,
) -> None:
    """Refuse to report a number computed over a cohort nobody chose.

    Every way a run can quietly measure fewer samples than it claims — a
    dropped chunk, a duplicated or lost row, a truth outside the candidate set —
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
            f"{n_absent} ground truth(s) were absent from the candidate set. "
            "Mode C ranks every disease, so this is impossible unless the ids or "
            "the harness are wrong. It is not a property of the model and must "
            "not be reported as one"
        )
    if not (n_sample_ids == n_canonical_ranks == manifest.n_samples):
        raise ValueError(
            "cohort shrinkage: the run produced "
            f"{n_sample_ids} sample ids and {n_canonical_ranks} canonical ranks "
            f"for {manifest.n_samples} declared samples. A metric over a subset of "
            "the declared cohort is not the metric the manifest describes"
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
    full_graph_embeddings: "EncodedGraph",
    samples: Iterable[Any],
    manifest: MeasurementManifest,
    device: Optional[Any] = None,
    batch_size: int = 32,
) -> ModeResult:
    """Score every patient against **every disease in the graph**.

    Its **candidate universe** is the reference method's — *"we calculate a
    patient's similarity to all disease nodes in the KG at inference time"* (npj
    Digital Medicine 8:380, 2025, Methods). The score is not: it is raw cosine,
    the interim disease score (`docs/DISEASE_SCORER_POLICY.md`), where the paper
    uses a squared distance.

    **No subgraph sampler and no dataloader.** Mode C's candidate universe is all
    diseases, so there is nothing for a sampler to select and no reason to pay
    for it. Patients come straight from the samples, which is where the
    dataloader gets them too.

    Padding is done here rather than through `diagnosis_collate_fn` for the same
    reason: that function builds items through `DiagnosisDataset`, which draws
    negative samples this mode discards. Eight lines of `pad` and `stack` against
    a dependency that consumes randomness for nothing is not a close call.

    Absence is impossible by construction — the truth is a disease, and every
    disease is a candidate — so a non-zero absence count means the ids are wrong,
    and `_assert_cohort_is_intact` treats it as fatal.

    **Every real id is validated before it is used, and out-of-range is fatal.**
    Every id is a real value from a real patient with `mask=True`, so a clamp
    would score phenotype `-3` as phenotype 0 and phenotype 999999 as the last
    node in the graph — a plausible patient, a plausible rank, and a wrong one.
    Padding uses a safe value under a `False` mask; **real values are never
    clamped, dropped or substituted.**
    """
    import torch as _torch
    import torch.nn.functional as _F

    from src.inference.scoring import cosine_score_matrix, masked_mean_pool

    device = _torch.device(device) if device is not None else _torch.device("cpu")
    assert_manifest_describes_regime(manifest, device, full_graph_embeddings)
    phenotype_emb = full_graph_embeddings.embeddings["phenotype"].to(device)
    disease_emb = full_graph_embeddings.embeddings["disease"].to(device)
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
        },
        sample_ids=sample_ids,
        truth_global_ids=truth_global_ids,
        canonical_ranks=canonical_ranks,
    )
