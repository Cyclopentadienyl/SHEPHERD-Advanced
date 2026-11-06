"""
The measurement report: its manifest, its result and the ranks rule.
====================================================================
Module: src/evaluation/measurement.py

The format a measurement report is written in: `MeasurementManifest` (what a
number needs recorded to mean anything later, with the relation states a run
earns, `RELATION_NAMES` and `expected_relations`), `ModeResult` (one run: the
report and its per-case ranks) and the ranks rule its ranks rows are held to
(`check_ranks_rows`). The task evaluators (`src/evaluation/task_measurement.py`)
fill them; `observe_torch_compile_wrapper` is an execution fact the manifest
records.

Mode C's executor, which ranked a ShepherdGNN rebuild by raw cosine against
every disease, retired in K8-6b: only the old pipeline built its model. A kept
Mode C report keeps this format.

Dependencies: the standard library; torch only where
`observe_torch_compile_wrapper` looks up its wrapper class.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "MeasurementManifest",
    "ModeResult",
    "RELATION_NAMES",
    "RELATION_STATES",
    "RANKS_PROBLEMS",
    "RanksRowsRefused",
    "check_ranks_rows",
    "expected_relations",
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


#: The relations a measurement run checks on the bytes it read, by the names its
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
    of cases a run scores at once, and it can move the numbers at
    floating-point level.
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
    """Which builder made the scored model (a task model's bundle builder; a
    kept Mode C report names `build_shepherd_model`). Recorded so a reader does
    not have to assume it."""
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
    `observe_autocast_regime` read for the run's device before its evaluator
    ran (`scripts/measure_scorer.py`), and the task evaluators refuse a run
    whose regime differs from what its manifest records
    (`task_measurement._check_regime`).

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
        """Per-sample ranks, one row per patient in the cohort's order, held
        to the ranks rule before anyone writes them (`check_ranks_rows`).

        **Each case has exactly one ground truth and one rank**: the three
        lists are the same length, checked before any row is built, so none is
        cut to the shortest (a rank or truth list longer, or shorter, than the
        cohort used to be truncated silently by `zip`).

        Raises:
            ValueError: the lists differ in length, naming each length.
        """
        lengths = (len(self.sample_ids), len(self.truth_global_ids), len(self.canonical_ranks))
        if len(set(lengths)) != 1:
            raise ValueError(
                f"the result's lists differ in length (sample ids {lengths[0]}, ground truths "
                f"{lengths[1]}, ranks {lengths[2]}); each case has exactly one of each, so no "
                "ranks row is written")
        rows = [
            {"sample_id": sample_id, "ground_truth": truth, "rank": rank}
            for sample_id, truth, rank in zip(
                self.sample_ids, self.truth_global_ids, self.canonical_ranks, strict=True
            )
        ]
        check_ranks_rows(rows, self.sample_ids)
        return rows


#: The parts of the ranks rule a row can break (`RanksRowsRefused.problem`):
#: rows that are not a list, a row that is not a mapping with a string
#: `sample_id` and a rank that is null or at least 1, a row naming no case of
#: the cohort, a case named again, a case with no row, and the cohort's cases
#: in another order.
RANKS_PROBLEMS = ("not_a_list", "malformed", "foreign", "duplicate", "missing", "misordered")


class RanksRowsRefused(ValueError):
    """Ranks rows that break the ranks rule (`check_ranks_rows`). `problem` is
    the part they break, one of `RANKS_PROBLEMS`, and `position` where it was
    found: the row's index, or for a case with no row its index in the cohort.
    A caller can say what failed by these two without repeating an identifier."""

    def __init__(self, problem: str, position: int, message: str):
        super().__init__(message)
        self.problem = problem
        self.position = position


def check_ranks_rows(rows: Any, sample_ids: Sequence[str]) -> List[str]:
    """**The ranks rule**, stated once: a ranks file holds one row per case of
    the cohort it measured, in the cohort's order (`sample_ids`), each with
    its `sample_id` and its `rank`, null for a case the model could not rank
    (a task model's case its graph view does not bind), so no case is dropped,
    repeated, replaced or moved. `ModeResult.to_ranks` holds what it writes to
    it; a reader of a kept ranks file checks the file against the sample ids
    its cohort binds by calling it. Returns the sample ids of the rows with a
    rank, in the cohort's order.

    Raises:
        RanksRowsRefused: naming the first part broken, row by row (a row that
            is malformed, foreign or a repeat), then a case with no row, then
            the order.
    """
    import math
    from collections import Counter

    if not isinstance(rows, list):
        raise RanksRowsRefused("not_a_list", 0, f"ranks rows are {type(rows).__name__}, "
                                                "not a list of rows")
    for index, row in enumerate(rows):
        rank = row.get("rank", ...) if isinstance(row, dict) else ...
        ranked = (isinstance(rank, (int, float)) and not isinstance(rank, bool)
                  and math.isfinite(rank) and rank >= 1)
        if not (isinstance(row, dict) and isinstance(row.get("sample_id"), str)
                and (rank is None or ranked)):
            raise RanksRowsRefused("malformed", index, (
                f"ranks row {index} is not a row with a sample_id and a rank (null, or a "
                "number of at least 1)"))
    ids = [row["sample_id"] for row in rows]
    expected = list(sample_ids)
    if ids != expected:
        cohort, seen = Counter(expected), Counter()
        for index, sample_id in enumerate(ids):
            seen[sample_id] += 1
            if sample_id not in cohort:
                raise RanksRowsRefused("foreign", index, (
                    f"ranks row {index} names {sample_id!r}, which is not a case of the "
                    "cohort"))
            if seen[sample_id] > cohort[sample_id]:
                raise RanksRowsRefused("duplicate", index, (
                    f"ranks row {index} names {sample_id!r} again; each case has one row"))
        for index, sample_id in enumerate(expected):
            if seen[sample_id] < cohort[sample_id]:
                raise RanksRowsRefused("missing", index, (
                    f"case {index} of the cohort, {sample_id!r}, has no ranks row; an "
                    "unrankable case keeps its row, with a null rank"))
        index = next(i for i, (got, want) in enumerate(zip(ids, expected, strict=True))
                     if got != want)
        raise RanksRowsRefused("misordered", index, (
            f"ranks row {index} names {ids[index]!r} where the cohort's order has "
            f"{expected[index]!r}; the rows follow the cohort's order"))
    return [row["sample_id"] for row in rows if row["rank"] is not None]

