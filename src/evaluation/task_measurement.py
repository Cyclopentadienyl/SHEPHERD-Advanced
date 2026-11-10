"""
Task measurement: each protocol's evaluator, the one way.
========================================================
`docs/working/PLAN_TEST_RESULTS.md` §8, `docs/working/scorer-retraining/README.md`
E1–E6, U6, G4; provenance contract §9.3 hazard 2; research comparison §5
Stage 5 (the common KG link-prediction measurement).

The run reuses the measurement entry's reads, report, store, usage history and
ledger (`scripts/measure_scorer.py`): the manifest has the same fields as
every measurement, its metric schema and scoring identity given as values by
its evaluator (`metric_schema_version`, `candidate_construction`,
`score_semantics`, `model_construction`, `canonical_tie_policy_version`,
`batch_size`). Its `mode` is the protocol the run was asked for, which the
entry sets (`src.evaluation.protocols`): a protocol names the evaluator here
that runs it, and the evaluators' names ("disease", "gene",
"patients_like_me", "kg_link_prediction") key `TASK_RUNS` and
`task_semantics`.

**What is computed.** The bundle's node vectors by the full-view rule over the
whole view (G4), every case bound to the view (F7), ranked against every
disease of the view by the shared scorer (`src.models.tasks.scoring`), average
ranks (E3), and the one metrics module's report (E1, E2, E5): pooled MRR and
top-k, macro recall, best-positive MRR, coverage and the conservative reading.
A case excluded at binding is counted by its reason and enters the
conservative reading as 0 (E4, E5); a contract error refuses (E6).

**The KG link-prediction measurement** (`run_kg_link_prediction`; owner's O3:
one evaluator for a base and its finetunes, the formal KG test and a family's
common read-out alike). A KG edge set's stored pairs, each positive edge
against its one rule-generated negative, are scored by the LP head through the
one LP core (`src.models.bundle.lp_head.score_lp_pairs`), over the full-view
vectors of the version's message-passing graph (its split's train part, with
the view's node types) for every model; there is no full-view variant. Before
anything is scored, the version's split must rebuild from the view and its
pairs' positives must be edges of its role's part: a version whose record
contradicts its bytes is refused. It reports metric schema 3
(`LP_METRIC_SCHEMA_VERSION`): lp_accuracy (raw > 0 a positive's decision, raw
<= 0 a negative's), ROC-AUC (average ranks), AP and lp_loss over the whole set,
each with its unit and denominator; a metric that cannot be computed is a
status, never a number. The run keeps each pair's actual raw scores and state
(`LinkPredictionResult.scores_chunks`, the LP core's scores table) and its
exposure (`src.evaluation.exposure`), a top-level report block.

Module: src/evaluation/task_measurement.py
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

from src.evaluation.protocols import KG_EDGES
from src.models.bundle.metrics import ranking_report

#: The ranking report's schema for task measurements (E1–E5), apart from the
#: metric schema 1 a kept Mode C report records.
TASK_METRIC_SCHEMA_VERSION = 2
TASK_KS = (1, 3, 5, 10, 20, 50, 100)

#: The manifest values of the disease evaluator (its `mode` is the protocol's).
DISEASE_SEMANTICS: Dict[str, Any] = {
    "metric_schema_version": TASK_METRIC_SCHEMA_VERSION,
    "candidate_construction": "every disease of the model's graph view",
    "score_semantics": "disease head: -||z_d - z_P||^2 on L2-normalised projected vectors",
    "model_construction": "model bundle (src.models.bundle.build_bundle) from its model spec",
    "canonical_tie_policy_version": "average-rank/v1",
}


def disease_semantics(batch_size: int) -> Dict[str, Any]:
    return {**DISEASE_SEMANTICS, "batch_size": int(batch_size)}


def gene_semantics(batch_size: int, bundle: Any) -> Dict[str, Any]:
    """The gene evaluator's manifest values, η, the SP bounds and the
    augmentation named from the model's spec (PLAN_TEST_RESULTS §8.3: they are
    part of the scoring identity). Its `mode` is the protocol's."""
    gene = bundle.task_head.spec.gene
    augmentation = (f"Eq 10 augmentation K={gene.augmentation_k}, omega={gene.augmentation_omega}"
                    if gene.augmentation else "no augmentation")
    return {
        "metric_schema_version": TASK_METRIC_SCHEMA_VERSION,
        "candidate_construction": "each case's own candidate list, no upsampling",
        "score_semantics": (f"gene head: eta*EMBSIM + (1-eta)*SPLSIM, eta={gene.eta}; SPLSIM on "
                            f"exact view hops, bounds [{gene.sp_lower}, {gene.sp_upper}], no "
                            f"clipping; {augmentation}"),
        "model_construction": DISEASE_SEMANTICS["model_construction"],
        "canonical_tie_policy_version": "average-rank/v1",
        "batch_size": int(batch_size),
    }


def _mode_result(*, manifest: Any, report: Dict[str, Any], ranked: Sequence[Any], binding: Any,
                 samples: Sequence[Any], truth: Dict[str, int], evidence: Dict[str, Any]) -> Any:
    """The one `ModeResult` of a task measurement, from its report."""
    from src.evaluation.measurement import ModeResult

    metrics = {k: float(v) for k, v in report.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)
               and k not in ("units", "ranked_units", "positive_pairs",
                             "conservative_denominator_pairs")}
    by_case = {o.query_id: o for o in ranked if o.ranks}
    sample_ids = [s.patient_id for s in samples]
    ranks: List[Optional[float]] = [
        (min(by_case[i].ranks) if i in by_case else None) for i in sample_ids]
    evidence = {
        **evidence,
        "binding": dict(binding.counts),
        "units_by_status": dict(report["units_by_status"]),
        "units": report["units"],
        "positive_pairs": report["positive_pairs"],
        "conservative_denominator_pairs": report["conservative_denominator_pairs"],
    }
    return ModeResult(
        manifest=manifest, authoritative_metrics=metrics, n_ranked=int(report["ranked_units"]),
        n_ground_truth_absent=int(binding.counts.get("excluded_truth_outside_view", 0)),
        sampler_evidence=evidence, sample_ids=sample_ids,
        truth_global_ids=[truth.get(i) for i in sample_ids], canonical_ranks=ranks)


def _require_bound(binding: Any, task: str) -> None:
    """A cohort none of whose cases binds is refused before anything is written
    (E6): it is not this task's cohort or not this graph's, and a report would
    only say coverage 0."""
    if not binding.cases:
        raise ValueError(f"no case of the cohort can be ranked by the {task} model: "
                         f"{binding.counts}; nothing was written")


def _check_regime(device: torch.device, manifest: Any) -> None:
    from src.utils.run_environment import AutocastRegime, observe_autocast_regime

    recorded = AutocastRegime(enabled=manifest.amp_enabled, dtype=manifest.amp_dtype)
    if observe_autocast_regime(device) != recorded:
        raise RuntimeError("this measurement runs under another autocast regime than its "
                           "manifest records")


def run_disease_measurement(*, bundle: Any, view: Any, samples: Sequence[Any],
                            manifest: Any, device: torch.device, batch_size: int) -> Any:
    """Rank `samples` with a finetuned disease bundle; a `ModeResult`."""
    from src.models.tasks.scoring import bind_disease_cases, full_view_vectors, rank_cases

    if bundle.task != "disease":
        raise ValueError(f"a {bundle.task!r} model does not rank diseases")
    binding = bind_disease_cases(view, samples)
    _require_bound(binding, "disease")
    _check_regime(device, manifest)
    z = full_view_vectors(bundle, view.edge_index(), view.rel, device, node_type=view.node_type)
    diseases = view.type_nodes("disease")
    ranked = rank_cases(bundle, z, binding.cases, diseases, case_chunk=batch_size)
    report = ranking_report(ranked + binding.excluded_outcomes(), TASK_KS)
    return _mode_result(
        manifest=manifest, report=report, ranked=ranked, binding=binding, samples=samples,
        truth={c.case_id: int(c.truths[0]) for c in binding.cases},
        evidence={"candidate_columns": {"min": int(diseases.numel()),
                                        "max": int(diseases.numel())},
                  "graph_view": view.digest})


def run_gene_measurement(*, bundle: Any, view: Any, samples: Sequence[Any],
                         manifest: Any, device: torch.device, batch_size: int) -> Any:
    """Rank each case's own candidate list with a finetuned gene bundle
    (PLAN_TEST_RESULTS §8.3); a `ModeResult`. Every causal gene is a positive
    (F9); a case with an undefined distance is not scorable and stays in the
    denominator (§6 rule 7)."""
    from src.models.tasks.gene import (
        bind_gene_cases,
        distance_table,
        neighbour_table,
        rank_gene_cases,
    )
    from src.models.tasks.scoring import full_view_vectors

    if bundle.task != "gene":
        raise ValueError(f"a {bundle.task!r} model does not rank candidate genes")
    binding = bind_gene_cases(view, samples)
    _require_bound(binding, "gene")
    _check_regime(device, manifest)
    gene = bundle.task_head.spec.gene
    cases = binding.cases
    distances = distance_table(view, torch.cat([c.phenotypes for c in cases]),
                               torch.cat([c.candidates for c in cases]))
    neighbours = neighbour_table(view, gene.augmentation_k) if gene.augmentation else None
    z = full_view_vectors(bundle, view.edge_index(), view.rel, device, node_type=view.node_type)
    ranked = rank_gene_cases(bundle, z, cases, distances, neighbours, case_chunk=batch_size)
    lengths = [int(c.candidates.numel()) for c in cases]
    evidence: Dict[str, Any] = {
        "graph_view": view.digest,
        "candidate_columns": {"min": min(lengths), "max": max(lengths)},
        "derived_tables": {"sp_distances": distances.digest,
                           "augmentation_neighbours": (neighbours.record["digest"]
                                                       if neighbours is not None else None)}}
    report = ranking_report(ranked + binding.excluded_outcomes(), TASK_KS)
    # Each case's best-ranked causal gene, the one its canonical rank belongs to
    # (the ranks follow the candidates' order, `rank_gene_cases`).
    truth: Dict[str, int] = {}
    for case, outcome in zip(cases, ranked, strict=True):
        if outcome.ranks:
            genes = case.candidates[torch.isin(case.candidates, case.truths)].tolist()
            truth[case.case_id] = int(genes[min(range(len(outcome.ranks)),
                                                key=outcome.ranks.__getitem__)])
    return _mode_result(
        manifest=manifest, report=report, ranked=ranked, binding=binding, samples=samples,
        truth=truth, evidence=evidence)


#: The patients-like-me evaluator's manifest values (PLAN_TEST_RESULTS §8.4);
#: its `mode` is the protocol's.
PATIENTS_LIKE_ME_SEMANTICS: Dict[str, Any] = {
    "metric_schema_version": TASK_METRIC_SCHEMA_VERSION,
    "candidate_construction": "every other patient of the measured cohort; self excluded by case id",
    "score_semantics": ("patient tower: -||z_a - z_b||^2 on L2-normalised projected vectors; a "
                        "pair is positive when the declared causal-gene sets intersect: the "
                        "same gene of the graph, or the same normalised identifier outside it "
                        "(no alias resolved)"),
    "model_construction": DISEASE_SEMANTICS["model_construction"],
    "canonical_tie_policy_version": "average-rank/v1",
}


def run_patients_like_me_measurement(*, bundle: Any, view: Any, samples: Sequence[Any],
                                     manifest: Any, device: torch.device, batch_size: int) -> Any:
    """Rank every patient of the cohort against all the others (PLAN_TEST_RESULTS
    §8.4); a `ModeResult`. A patient with no other positive, or without a
    causal gene, keeps its own state and is never a recall of 0 (E4)."""
    from src.models.tasks.patients_like_me import bind_patients, rank_patients
    from src.models.tasks.scoring import full_view_vectors

    if bundle.task != "patients_like_me":
        raise ValueError(f"a {bundle.task!r} model does not rank patients")
    binding = bind_patients(view, samples)
    _require_bound(binding, "patients-like-me")
    if not any(c.genes for c in binding.cases):
        # E4/E6: without a causal gene no pair is positive; a report would
        # only say that no patient has a reliable truth.
        raise ValueError("no patient of the cohort declares a causal gene, so no pair can be "
                         "positive; nothing was written")
    _check_regime(device, manifest)
    cases = binding.cases
    z = full_view_vectors(bundle, view.edge_index(), view.rel, device, node_type=view.node_type)
    ranked = rank_patients(bundle, z, cases, cases, chunk=batch_size,
                           unranked=binding.unranked_positives)
    report = ranking_report(ranked + binding.excluded_outcomes(), TASK_KS)
    return _mode_result(
        manifest=manifest, report=report, ranked=ranked, binding=binding, samples=samples,
        truth={}, evidence={"graph_view": view.digest,
                            "candidate_columns": {"min": max(len(cases) - 1, 0),
                                                  "max": max(len(cases) - 1, 0)}})


# ---------------------------------------------------------------------------
# The common KG link-prediction measurement (research comparison §5 Stage 5)
# ---------------------------------------------------------------------------
#: The KG link-prediction measurement's metric schema: whole-set LP metrics,
#: each with its unit and denominator, and a status where one cannot be
#: computed; another schema than task ranking's (`TASK_METRIC_SCHEMA_VERSION`).
LP_METRIC_SCHEMA_VERSION = 3

#: Metric schema 3's metrics, in the report's order: each one's unit and its
#: denominator's unit (`lp_metric_statuses` counts it).
LP_METRICS: Dict[str, Dict[str, str]] = {
    "lp_accuracy": {
        "unit": ("share of raw scores on their side of the decision: a positive's above 0, a "
                 "negative's at or below 0"),
        "denominator": "raw scores: each pair's positive and its negative"},
    "lp_roc_auc": {
        "unit": ("probability that a positive's raw score is above a negative's, a tie "
                 "counting half (average ranks)"),
        "denominator": "(positive, negative) raw-score comparisons over the whole set"},
    "lp_average_precision": {
        "unit": ("step-sum average precision over the distinct raw-score thresholds, the "
                 "positives the relevant class"),
        "denominator": "positive raw scores (recall's denominator)"},
    "lp_loss": {
        "unit": "mean of the LP head's hinge over each pair's tanh scores",
        "denominator": "(positive, negative) pairs"},
}

#: The run's counts, each with its unit.
LP_COUNT_UNITS: Dict[str, str] = {
    "positive_edges": ("directed graph-view edges of the version's part (the measured units), "
                       "each a positive"),
    "pairs": "(positive, negative) pairs stored, one per positive edge that has a negative",
    "edges_without_negative": ("positive edges the pair rule left out: no other tail of their "
                               "relation in their batch to draw a negative from"),
    "pairs_scored": "pairs whose two raw scores are finite",
    "pairs_not_finite": "pairs with a raw score that is not finite (NaN or infinite)",
    "message_passing_edges": "directed graph-view edges of the version's message-passing graph",
}


def kg_link_prediction_semantics(batch_size: int, bundle: Any) -> Dict[str, Any]:
    """The KG link-prediction evaluator's manifest values, the LP head's margin
    named from the model's spec (it is part of what lp_loss means). Its
    `mode` is the protocol's; `batch_size` is the pairs scored at once."""
    margin = bundle.lp_head.spec.margin
    return {
        "metric_schema_version": LP_METRIC_SCHEMA_VERSION,
        "candidate_construction": (
            "each positive edge of the KG edge set against its one negative tail, as its pairs "
            "file stores them (the version's pair rule: another tail of the same relation in "
            "the edge's batch, drawn once); negatives are rule-generated, never proven "
            "non-relations"),
        "score_semantics": (
            "LP head: DistMult raw score sum(h * w_r * t) over full-view vectors of the "
            "version's message-passing graph (its split's train part) with the view's node "
            "types; decision raw > 0; lp_loss the head's hinge "
            f"max(0, {margin} - (tanh(raw+) - tanh(raw-)))"),
        "model_construction": DISEASE_SEMANTICS["model_construction"],
        "canonical_tie_policy_version": "average-rank/v1",
        "batch_size": int(batch_size),
    }


def lp_metric_statuses(metrics: Dict[str, Any], *, pairs: int, not_finite: int
                       ) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    """Metric schema 3 from the LP core's whole-set metrics (`score_lp_pairs`)
    over `pairs` pairs, `not_finite` of them with a raw score that is not
    finite: the metrics that could be computed, as numbers, and every
    metric's status, unit and denominator.

    **A metric that cannot be computed is a status, never a number:** with a
    raw score that is not finite in the set, no whole-set metric is defined;
    ROC-AUC and AP need both a positive and a negative (`lp_metrics` would
    give AP 1.0 over positives alone: a stored pair always holds both, and
    the status does not rest on that); any value that is not a finite
    number is not one either.
    """
    positives = negatives = int(pairs)
    counts = {"lp_accuracy": positives + negatives, "lp_roc_auc": positives * negatives,
              "lp_average_precision": positives, "lp_loss": int(pairs)}
    values: Dict[str, float] = {}
    statuses: Dict[str, Dict[str, Any]] = {}
    for name, words in LP_METRICS.items():
        value = metrics.get(name)
        if not_finite:
            reason = (f"{not_finite} of {pairs} pairs have a raw score that is not finite, so no "
                      "whole-set value is defined")
        elif name in ("lp_roc_auc", "lp_average_precision") and not (positives and negatives):
            reason = "it needs at least one positive and one negative"
        elif (isinstance(value, bool) or not isinstance(value, (int, float))
              or not math.isfinite(value)):
            reason = f"its value {value!r} is not a finite number"
        else:
            reason = None
            values[name] = float(value)
        statuses[name] = {"status": "computed" if reason is None else "not_computable",
                          "unit": words["unit"],
                          "denominator": {"count": counts[name], "unit": words["denominator"]},
                          "reason": reason}
    return values, statuses


@dataclass(frozen=True)
class LinkPredictionResult:
    """One KG link-prediction run: its report (`to_dict`: the manifest, the
    metrics that could be computed, the counts, the evidence with every
    metric's status, and the exposure block) and its actual outputs, each
    pair's raw scores and state in pair order, kept as a scores table
    (`scores_chunks`, `scores_block`)."""

    manifest: Any
    authoritative_metrics: Dict[str, float]
    #: The units scored: positive edges with a negative (the pairs).
    n_ranked: int
    #: Every measured edge is in the view by construction: 0.
    n_ground_truth_absent: int
    sampler_evidence: Dict[str, Any]
    exposure: Dict[str, Any]
    positive: torch.Tensor
    negative: torch.Tensor
    state: torch.Tensor
    #: The scores table's layout (`lp_scores_layout`).
    layout: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """The measurement report, without its ``scores`` entry, which the
        publication adds once the scores file is written (as a task run's
        ``ranks``)."""
        return {
            "manifest": asdict(self.manifest),
            "authoritative_metrics": self.authoritative_metrics,
            "n_ranked": self.n_ranked,
            "n_ground_truth_absent": self.n_ground_truth_absent,
            "sampler_evidence": self.sampler_evidence,
            "exposure": self.exposure,
        }

    def scores_chunks(self, rows: int = 65536) -> Iterator[bytes]:
        """The scores table, chunk by chunk (`lp_scores_chunks`): one row per
        pair, keyed to its row of the pairs file."""
        from src.models.bundle.lp_head import lp_scores_chunks

        return lp_scores_chunks(self.positive, self.negative, self.state, self.layout, rows=rows)

    def scores_block(self, file: str, sha256: str, size: int) -> Dict[str, Any]:
        """The report's ``scores`` entry for the table published as `file`
        with the digest `sha256` of its `size` bytes: its rows, layout, the
        dtype the scores were computed in, and the pairs file its rows key."""
        pairs = self.manifest.cohort_version["pairs"]
        return {"file": file, "sha256": sha256, "rows": int(self.positive.numel()),
                "bytes": int(size), "layout": dict(self.layout),
                "score_dtype": str(self.positive.dtype),
                "keyed_to": {"file": pairs["file"], "sha256": pairs["sha256"],
                             "lp_pairs_digest": pairs["lp_pairs_digest"]}}


def run_kg_link_prediction(*, bundle: Any, view: Any, pairs: Any, record: Any, manifest: Any,
                           device: torch.device, batch_size: int) -> LinkPredictionResult:
    """Score a KG edge set's stored `pairs` (`LPPairs`, as `read_kg_edge_pairs`
    returned them) with `bundle`'s LP head; a `LinkPredictionResult`.

    The version is the manifest's (`cohort_version`, its report block). In
    order, before anything is scored: its recorded split must rebuild from
    `view` (`rebuild_split`), and every positive of its pairs must be an edge
    of that split's part of its role; either refusal says the version's record
    contradicts its bytes and that nothing was written. Then the exposure
    block, from the model's record `record` (`src.evaluation.exposure`); the
    regime check; and the scoring, `batch_size` pairs at a time, over the
    version's message-passing graph (the split's train part) with the view's
    node types. Draws from no random generator.
    """
    from src.evaluation.exposure import kg_edges_exposure, measured_in_parts
    from src.kg.graph_view import GraphViewError, rebuild_split
    from src.models.bundle.lp_head import lp_score_states, lp_scores_layout, score_lp_pairs

    version = manifest.cohort_version
    if manifest.cohort_kind != KG_EDGES or not isinstance(version, dict):
        raise ValueError("the KG link-prediction measurement measures a KG edge set version; "
                         "nothing was written")
    label, role = version["label"], version["role"]
    try:
        split = rebuild_split(view, version["split"])
    except GraphViewError as exc:
        raise ValueError(
            f"the KG edge set {label!r} records an edge split that does not rebuild from this "
            f"graph's view ({exc}): its record contradicts the graph it binds, so it is not "
            "measured. Nothing was written.") from None
    n = int(pairs.head.numel())
    in_version = measured_in_parts(view, split, pairs)
    outside = int((~in_version[role]).sum())
    if outside:
        raise ValueError(
            f"{outside} of the {n} positive edges in the KG edge set {label!r}'s pairs file are "
            f"not edges of its split's {role} part: the pairs contradict the version's record, "
            "so they are not measured. Nothing was written.")
    exposure = kg_edges_exposure(
        record=record, checkpoint_sha256=manifest.artifact_digests["checkpoint"],
        version=version, view=view, version_split=split, pairs=pairs, in_version=in_version)
    _check_regime(device, manifest)
    passing = split.mask("train")
    scored = score_lp_pairs(bundle, pairs, view.edge_index(passing), view.rel[passing], device,
                            node_type=view.node_type, pair_chunk=batch_size)
    state = lp_score_states(scored.positive, scored.negative)
    not_finite = int(state.sum())
    values, statuses = lp_metric_statuses(scored.metrics, pairs=n, not_finite=not_finite)
    counts = version["counts"]
    evidence = {
        "graph_view": view.digest,
        "message_passing": {"part": "train", "split_digest": split.digest,
                            "edges": int(passing.sum()),
                            "measured_edges_in_it": int(in_version["train"].sum()),
                            "includes_measured_edges": bool(
                                version["message_passing"]["includes_measured_edges"])},
        "pairs": {"rule": version["pairs"]["rule"], "rows": n,
                  "lp_pairs_digest": pairs.digest, "pairs_scored_at_once": int(batch_size)},
        "counts": {"positive_edges": int(counts["positive_edges"]), "pairs": n,
                   "edges_without_negative": int(counts["edges_without_negative"]),
                   "pairs_scored": n - not_finite, "pairs_not_finite": not_finite,
                   "message_passing_edges": int(passing.sum())},
        "count_units": dict(LP_COUNT_UNITS),
        "metrics": statuses,
    }
    return LinkPredictionResult(
        manifest=manifest, authoritative_metrics=values, n_ranked=n, n_ground_truth_absent=0,
        sampler_evidence=evidence, exposure=exposure, positive=scored.positive,
        negative=scored.negative, state=state, layout=lp_scores_layout(scored.positive.dtype))


#: The evaluators this entry runs, keyed by evaluator: a protocol names its
#: evaluator (`src.evaluation.protocols.Protocol.evaluator`), and the run and
#: its semantics (`task_semantics`) are found under that name, not under the
#: protocol's name or the model's task (today a protocol and its evaluator
#: share a name; nothing relies on it). The three task evaluators take the
#: samples of a patient cohort; the KG link-prediction evaluator takes a KG
#: edge set's pairs and the model's record (`scripts/measure_scorer.py`
#: dispatches by the data kind measured).
TASK_RUNS = {"disease": run_disease_measurement, "gene": run_gene_measurement,
             "patients_like_me": run_patients_like_me_measurement,
             "kg_link_prediction": run_kg_link_prediction}


def task_semantics(evaluator: str, batch_size: int, bundle: Any) -> Dict[str, Any]:
    """The manifest values of the evaluator `evaluator` (a `TASK_RUNS` key),
    every semantic field but `mode`, which is the protocol's."""
    if evaluator == "disease":
        return disease_semantics(batch_size)
    if evaluator == "gene":
        return gene_semantics(batch_size, bundle)
    if evaluator == "patients_like_me":
        return {**PATIENTS_LIKE_ME_SEMANTICS, "batch_size": int(batch_size)}
    if evaluator == "kg_link_prediction":
        return kg_link_prediction_semantics(batch_size, bundle)
    raise ValueError(f"no task measurement for {evaluator!r}")


__all__ = ["DISEASE_SEMANTICS", "LP_COUNT_UNITS", "LP_METRICS", "LP_METRIC_SCHEMA_VERSION",
           "LinkPredictionResult", "PATIENTS_LIKE_ME_SEMANTICS", "TASK_KS",
           "TASK_METRIC_SCHEMA_VERSION", "TASK_RUNS", "disease_semantics", "gene_semantics",
           "kg_link_prediction_semantics", "lp_metric_statuses", "run_disease_measurement",
           "run_gene_measurement", "run_kg_link_prediction",
           "run_patients_like_me_measurement", "task_semantics"]
