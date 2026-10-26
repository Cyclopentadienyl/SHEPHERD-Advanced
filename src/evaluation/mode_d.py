"""
Mode D: the served ranking, measured (B-0.5).
=============================================
`docs/working/scorer-measurement/PLAN_B05.md` is the design; this module is all
of Mode D's own code, so retiring it later is this module, one CLI option and
one test file (§10).

**What it measures.** How well what the Diagnosis page serves today ranks the
truth: the diseases path discovery reaches, scored by the deployment's scoring
mode, ties kept in discovery order. It calls the served pipeline's own
`DiagnosisPipeline.rank()`, which `run()` itself calls, so nothing constructs
candidates a second way, and it builds no model: the pipeline was built by the
service's builder (`build_served_pipeline`).

**What it sets beside it, in the same run.** The formal Mode C, `run_mode_c`
itself, on the embeddings the pipeline already holds (no second model, no
second evaluator), and derived readings from the scores that run computed, one
change at a time: G1 the served candidates by raw cosine with Mode C's tie
rule, G2 the same with discovery order for ties. C to G1 changes the candidate
set, G1 to G2 the tie policy, G2 to D the score. C to D is their joint
difference, and is labelled so (§5).

**The case sets** (§4). Both modes rank exactly the import-eligible cases of a
registered cohort version (set I, `counts.ranked_eligible`). A case excluded at
import is counted beside the figures, never as a failure and never as absent:
whether it was eligible beforehand with a reliable truth (set E) is not
recorded by any version, so no end-to-end figure is given.

**Dependencies run one way** (§10): this module calls `rank()` and
`run_mode_c`; nothing in serving, training, Mode C or B-1 imports it.

Module: src/evaluation/mode_d.py
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.core.types import DataSource, NodeID, PatientPhenotypes
from src.evaluation.measurement import (
    EncodedGraph,
    MeasurementManifest,
    ModeResult,
    assert_manifest_describes_regime,
    run_mode_c,
)

MODE = "D"
#: The next value of the one integer sequence `MeasurementManifest`'s field
#: holds: Mode C's family is 1 (`measurement.METRIC_SCHEMA_VERSION`), and Mode
#: D's family (post-import, among reached, coverage) is a different metric set.
METRIC_SCHEMA_VERSION = 2
CANDIDATE_CONSTRUCTION_VERSION = "served-path-discovery/v1"
TIE_POLICY = "served-stable-discovery-order/v1"
#: One case per `rank()` call: the served path ranks one input at a time.
BATCH_SIZE = 1
K_VALUES: Tuple[int, ...] = (1, 5, 10, 20, 50, 100)

RANKED = "ranked"
NOT_REACHED = "not_reached"
REFUSED = "refused"
OUTCOMES: Tuple[str, ...] = (RANKED, NOT_REACHED, REFUSED)

#: The serving settings that are not the kind of scoring: where the model ran
#: is a regime field of the manifest (`device`), not a property of the policy.
_REGIME_SETTINGS = ("device", "device_requested")

COMPARISON_STATEMENT = (
    "Mode C and Mode D are different ranking policies: Mode C ranks every disease by raw "
    "cosine; Mode D measures what the Diagnosis page serves today. Both are measured on "
    "the same import-eligible cases; the cases excluded at import are counted beside them."
)
END_TO_END_ABSENT = (
    "not determinable: the cohort version records its source cases and its import "
    "exclusions, not which source cases were eligible beforehand with a reliable truth "
    "(BACKLOG 8a defines that set), so a figure over them would guess its denominator"
)


class ModeDRefused(ValueError):
    """A Mode D run that cannot measure what it is asked to, refused before any
    case is ranked. A `ValueError`, so the measurement CLI prints it as a
    refusal with nothing written."""


# ---------------------------------------------------------------------------
# What kind of scoring this is (the manifest's semantic values, §6)
# ---------------------------------------------------------------------------
def served_semantics(served_identity: Any) -> Dict[str, Any]:
    """Mode D's values for the manifest's existing semantic fields, from the
    served identity of the pipeline it measures.

    `candidate_construction` names the discovery settings by a digest of the
    served settings (the device excepted, which is a regime field), so a change
    of settings is a different measurement. `score_semantics` names the scoring
    mode, the mixing weight in effect and the shortest-path table by its build
    id. Refuses the path-reasoning fallback: it ranks without a model, so there
    is nothing to compare with Mode C (§3).
    """
    identity = served_identity.to_dict() if hasattr(served_identity, "to_dict") else dict(
        served_identity)
    mode = identity["scoring_mode"]
    if mode == "path_reasoning_fallback":
        raise ModeDRefused(
            "the pipeline serves path_reasoning_fallback: it ranks without a model, so "
            "there is no checkpoint and no embeddings to set beside Mode C. Measure a "
            "deployment that serves a checkpoint."
        )
    settings = {k: v for k, v in identity["settings"].items() if k not in _REGIME_SETTINGS}
    digest = hashlib.sha256(
        json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    build_id = identity["shortest_paths"].get("build_id") or "none"
    return {
        "mode": MODE,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "candidate_construction": f"{CANDIDATE_CONSTRUCTION_VERSION}:settings={digest}",
        "batch_size": BATCH_SIZE,
        "score_semantics": f"served:{mode}:eta={settings['eta']!r}:sp={build_id}",
        "canonical_tie_policy_version": TIE_POLICY,
    }


# ---------------------------------------------------------------------------
# One case through the served path (§2, §3, §4)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ServedCase:
    """One import-eligible case's served outcome.

    `served_ids` are the served candidates' global disease indices in the served
    order, `discovery_ordinals` and `served_scores` beside them (`rank()`'s). The
    truth's `served_rank` is its 1-based position in that order, or None."""

    sample_id: str
    ground_truth: int
    outcome: str
    no_candidates: bool
    served_rank: Optional[int]
    served_ids: Tuple[int, ...]
    discovery_ordinals: Tuple[int, ...]
    served_scores: Tuple[float, ...]


def served_input(sample: Any, phenotype_names: Mapping[int, str]) -> PatientPhenotypes:
    """A sample's phenotype indices as the served input form: the HPO terms the
    graph's own reverse mapping names, the mapping the importer used in the
    other direction. No source string is read. A row the graph does not name,
    or one that is not an HPO term, refuses: the cohort was bound to this graph,
    so either means the binding does not describe what is scored."""
    terms: List[str] = []
    for index in sample.phenotype_ids:
        name = phenotype_names.get(int(index))
        if name is None:
            raise ModeDRefused(
                f"sample {sample.patient_id!r} names phenotype row {index}, which this "
                "graph does not have; the cohort's binding does not describe this graph"
            )
        node = NodeID.from_string(name)
        if node.source != DataSource.HPO:
            raise ModeDRefused(
                f"sample {sample.patient_id!r} names phenotype row {index} ({name}), which "
                "is not an HPO term the served input can carry"
            )
        terms.append(node.local_id)
    return PatientPhenotypes(patient_id=str(sample.patient_id), phenotypes=terms)


def rank_served_cases(pipeline: Any, samples: Sequence[Any]) -> List[ServedCase]:
    """Every sample through `pipeline.rank()`, in cohort order, with its outcome:
    ``ranked`` (the truth is among the served candidates), ``not_reached``
    (discovery did not reach it; `no_candidates` when it reached nothing) or
    ``refused`` (the served validation refused the input)."""
    mappings = pipeline.kg.get_node_id_mapping()
    disease_index = mappings["disease"]
    phenotype_names = pipeline.kg.get_reverse_node_mapping()["phenotype"]
    cases: List[ServedCase] = []
    for sample in samples:
        truth = int(sample.disease_id)
        ranking = pipeline.rank(served_input(sample, phenotype_names))
        if ranking.outcome == "refused":
            cases.append(ServedCase(str(sample.patient_id), truth, REFUSED, False, None,
                                    (), (), ()))
            continue
        served_ids: List[int] = []
        for candidate in ranking.candidates:
            key = str(candidate.disease_id)
            if key not in disease_index:
                raise ModeDRefused(
                    f"the served candidate {key} is not a disease row of this graph; the "
                    "pipeline and the graph it was verified against disagree"
                )
            served_ids.append(int(disease_index[key]))
        rank = served_ids.index(truth) + 1 if truth in served_ids else None
        cases.append(ServedCase(
            sample_id=str(sample.patient_id),
            ground_truth=truth,
            outcome=RANKED if rank is not None else NOT_REACHED,
            no_candidates=not served_ids,
            served_rank=rank,
            served_ids=tuple(served_ids),
            discovery_ordinals=tuple(ranking.discovery_ordinals),
            served_scores=tuple(float(c.confidence_score) for c in ranking.candidates),
        ))
    return cases


# ---------------------------------------------------------------------------
# Metrics (§5)
# ---------------------------------------------------------------------------
def post_import_metrics(ranks: Sequence[Optional[int]]) -> Dict[str, float]:
    """Over every import-eligible case: 1/r for a ranked case and 0 for one that
    was not reached or was refused; hits@K the fraction ranked at K or better;
    `coverage` the fraction ranked at all."""
    n = len(ranks)
    if n == 0:
        raise ModeDRefused("there are no import-eligible cases, so every figure would be "
                           "fabricated")
    reached = [r for r in ranks if r is not None]
    metrics = {"post_import_mrr": sum(1.0 / r for r in reached) / n}
    for k in K_VALUES:
        metrics[f"post_import_hits@{k}"] = sum(1 for r in reached if r <= k) / n
    metrics["coverage"] = len(reached) / n
    return metrics


def reached_metrics(ranks: Sequence[Optional[int]]) -> Dict[str, float]:
    """Over the reached cases only (`RankingMetrics.compute_from_ranks`): empty
    when none was reached, which the report says."""
    from src.utils.metrics import RankingMetrics

    reached = [int(r) for r in ranks if r is not None]
    if not reached:
        return {}
    computed = RankingMetrics().compute_from_ranks(reached, k_values=K_VALUES)
    return {f"reached_{name}": value for name, value in computed.items()}


def _rank_of(order: Sequence[int], truth: int) -> Optional[int]:
    return order.index(truth) + 1 if truth in order else None


def derived_ranks(case: ServedCase, cosines: Optional[Sequence[float]]) -> Tuple[
        Optional[int], Optional[int]]:
    """G1 and G2's rank of the truth (§5): the served candidates ordered by raw
    cosine, ties to the lower global id (G1) or kept in discovery order by the
    discovery ordinal (G2), never by re-sorting the served order."""
    if not case.served_ids:
        return None, None
    if cosines is None or len(cosines) != len(case.served_ids):
        raise ModeDRefused(f"no cosine scores were observed for case {case.sample_id!r}'s "
                           "served candidates")
    rows = list(zip(case.served_ids, case.discovery_ordinals, cosines))
    g1 = [disease for disease, _, _ in sorted(rows, key=lambda r: (-r[2], r[0]))]
    g2 = [disease for disease, _, _ in sorted(rows, key=lambda r: (-r[2], r[1]))]
    return _rank_of(g1, case.ground_truth), _rank_of(g2, case.ground_truth)


def _quantiles(values: Sequence[int]) -> Optional[Dict[str, int]]:
    """Nearest-rank quantiles of a rank difference; None when there is none."""
    if not values:
        return None
    ordered = sorted(values)

    def at(q: float) -> int:
        return ordered[min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))]

    return {"min": ordered[0], "q25": at(0.25), "median": at(0.5), "q75": at(0.75),
            "max": ordered[-1]}


def paired(before: Sequence[Optional[int]], after: Sequence[Optional[int]]) -> Dict[str, Any]:
    """How the truth's rank moved from one step to the next, case by case."""
    both = [(a, b) for a, b in zip(before, after) if a is not None and b is not None]
    return {
        "n_both_ranked": len(both),
        "better": sum(1 for a, b in both if b < a),
        "equal": sum(1 for a, b in both if b == a),
        "worse": sum(1 for a, b in both if b > a),
        "ranked_then_not": sum(1 for a, b in zip(before, after) if a is not None and b is None),
        "rank_difference": _quantiles([b - a for a, b in both]),
    }


# ---------------------------------------------------------------------------
# The run (§4-§6)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ServedModeResult:
    """One Mode D run: the report and its per-case rows.

    The report keeps the keys every measurement report has (`manifest`,
    `authoritative_metrics`, `n_ranked`) and adds its own blocks, which
    `build_record` ignores: `outcomes`, `import_counts`, `served` (the served
    identity whole) and `comparison` (§6). No manifest field is added."""

    manifest: MeasurementManifest
    authoritative_metrics: Dict[str, float]
    n_ranked: int
    outcomes: Dict[str, int]
    import_counts: Dict[str, Any]
    served: Dict[str, Any]
    comparison: Dict[str, Any]
    rows: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """The measurement report; the per-case rows are their own artifact."""
        return {
            "manifest": asdict(self.manifest),
            "authoritative_metrics": self.authoritative_metrics,
            "n_ranked": self.n_ranked,
            "outcomes": self.outcomes,
            "import_counts": self.import_counts,
            "served": self.served,
            "comparison": self.comparison,
        }

    def to_ranks(self) -> List[Dict[str, Any]]:
        """One row per case in cohort order: the served outcome and each step's
        rank of the truth (None where it has none)."""
        return list(self.rows)


def import_counts_block(cohort_version: Mapping[str, Any]) -> Dict[str, Any]:
    """The version's import counts beside the figures (§4), with the reason no
    end-to-end figure is given."""
    counts = cohort_version.get("counts") or {}
    return {
        "source_cases": counts.get("source_cases"),
        "ranked_eligible": counts.get("ranked_eligible"),
        "excluded": dict(counts.get("excluded") or {}),
        "n_excluded": counts.get("n_excluded"),
        "end_to_end": END_TO_END_ABSENT,
    }


def run_mode_d(
    pipeline: Any,
    samples: Sequence[Any],
    manifest: MeasurementManifest,
    mode_c_manifest: MeasurementManifest,
    *,
    device: Any,
    batch_size: int,
    cohort_version: Mapping[str, Any],
    kept_mode_c: Optional[Callable[[Mapping[str, int]], Sequence[Mapping[str, Any]]]] = None,
) -> Tuple[ServedModeResult, ModeResult]:
    """Measure the served ranking of every import-eligible case, and set Mode C,
    run on the same embeddings, beside it.

    `manifest` is Mode D's (its semantic values from `served_semantics`, its
    regime the pipeline's embedding regime); `mode_c_manifest` is the C side's,
    built as any Mode C run's manifest is, for the device and `batch_size` it
    runs with. Both are checked against the embeddings' regime before anything
    is ranked. `kept_mode_c`, given the C side's ranks by sample id, returns
    the cross-check with kept Mode C reports for the same checkpoint and version
    (`kept_mode_c_reports`), recorded as it returns it.

    Returns Mode D's result and the C side's `ModeResult`; the C side is not a
    report of its own, and its figures are in Mode D's comparison block.
    """
    samples = list(samples)
    expected = served_semantics(pipeline.served_identity())
    recorded = {field: getattr(manifest, field) for field in expected}
    if recorded != expected:
        raise ModeDRefused(f"the manifest's semantic values {recorded} are not the served "
                           f"pipeline's {expected}")
    if manifest.n_samples != len(samples) or mode_c_manifest.n_samples != len(samples):
        raise ModeDRefused(
            f"the manifests claim {manifest.n_samples} and {mode_c_manifest.n_samples} samples "
            f"and the cohort holds {len(samples)}"
        )
    embeddings = pipeline.node_embeddings
    regime = pipeline.embedding_regime
    if embeddings is None or regime is None:
        raise ModeDRefused("the pipeline holds no embeddings, so there is nothing to measure")
    encoded = EncodedGraph(embeddings=embeddings, regime=regime)
    # Mode D's own manifest describes the embeddings' regime, as Mode C's must.
    assert_manifest_describes_regime(manifest, device, encoded)

    cases = rank_served_cases(pipeline, samples)

    # The C side: the formal Mode C on the same embeddings. Its observer keeps,
    # per case, only the raw cosines of that case's served candidates.
    cosines: Dict[int, List[float]] = {}

    def observe(first: int, _sample_ids: List[str], scores: Any) -> None:
        import torch

        for row in range(scores.size(0)):
            case = cases[first + row]
            if case.served_ids:
                columns = torch.tensor(case.served_ids, dtype=torch.long, device=scores.device)
                cosines[first + row] = scores[row].index_select(0, columns).tolist()

    mode_c = run_mode_c(encoded, samples, mode_c_manifest, device=device,
                        batch_size=batch_size, on_scores=observe)
    if mode_c.sample_ids != [case.sample_id for case in cases]:
        raise ModeDRefused("the C side ranked the cases in another order than Mode D did")

    rows: List[Dict[str, Any]] = []
    steps: Dict[str, List[Optional[int]]] = {"C": [], "G1": [], "G2": [], "D": []}
    for index, (case, c_rank) in enumerate(zip(cases, mode_c.canonical_ranks)):
        g1, g2 = derived_ranks(case, cosines.get(index))
        for name, rank in (("C", c_rank), ("G1", g1), ("G2", g2), ("D", case.served_rank)):
            steps[name].append(rank)
        rows.append({
            "sample_id": case.sample_id,
            "ground_truth": case.ground_truth,
            "outcome": case.outcome,
            "served_rank": case.served_rank,
            "n_served_candidates": len(case.served_ids),
            "c_rank": c_rank,
            "g1_rank": g1,
            "g2_rank": g2,
        })

    outcomes = {name: sum(1 for c in cases if c.outcome == name) for name in OUTCOMES}
    outcomes["no_candidates"] = sum(1 for c in cases if c.no_candidates)
    outcomes["n_import_eligible"] = len(cases)
    if sum(outcomes[name] for name in OUTCOMES) != len(cases):
        raise ModeDRefused(f"the outcomes {outcomes} do not add up to {len(cases)} cases")

    served_ranks = steps["D"]
    metrics = {**post_import_metrics(served_ranks), **reached_metrics(served_ranks)}
    comparison = {
        "statement": COMPARISON_STATEMENT,
        "case_set": "import-eligible (the version's counts.ranked_eligible)",
        "mode_c": {
            "authoritative_metrics": mode_c.authoritative_metrics,
            "n_ranked": mode_c.n_ranked,
            "run_conditions": {
                "device": mode_c_manifest.device,
                "batch_size": mode_c_manifest.batch_size,
                "dtype": mode_c_manifest.dtype,
                "amp_enabled": mode_c_manifest.amp_enabled,
                "amp_dtype": mode_c_manifest.amp_dtype,
            },
        },
        "steps": {
            "order": ["C", "G1", "G2", "D"],
            "what_changes": {
                "G1": "the candidate set: the served candidates, raw cosine, ties to the "
                      "lower global id",
                "G2": "the tie policy: the same scores, ties kept in discovery order",
                "D": "the score: the served score, ties in discovery order",
            },
            "post_import": {name: post_import_metrics(ranks) for name, ranks in steps.items()},
            "paired": {f"{a}->{b}": paired(steps[a], steps[b])
                       for a, b in (("C", "G1"), ("G1", "G2"), ("G2", "D"), ("C", "D"))},
            "d_not_reached_while_c_ranked": {
                f"top_{k}": sum(1 for c, d in zip(steps["C"], steps["D"])
                                if c is not None and c <= k and d is None)
                for k in (10, 100)
            },
        },
        "reached_metrics_omitted": (None if served_ranks and any(r is not None for r in served_ranks)
                                    else "no case was reached, so there is no rank to average"),
        "kept_mode_c_reports": (
            list(kept_mode_c(dict(zip(mode_c.sample_ids, mode_c.canonical_ranks))))
            if kept_mode_c is not None else []
        ),
    }
    result = ServedModeResult(
        manifest=manifest,
        authoritative_metrics=metrics,
        n_ranked=outcomes[RANKED],
        outcomes=outcomes,
        import_counts=import_counts_block(cohort_version),
        served=pipeline.served_identity().to_dict(),
        comparison=comparison,
        rows=rows,
    )
    return result, mode_c


# ---------------------------------------------------------------------------
# The cross-check with kept Mode C reports (§5)
# ---------------------------------------------------------------------------
def kept_mode_c_reports(version: Any, checkpoint_sha256: str,
                        c_ranks: Mapping[str, int]) -> List[Dict[str, Any]]:
    """For each kept Mode C report of the same checkpoint and version, how many
    of its ranks equal the C side's, case by case, with its run conditions.

    A cross-check under possibly different execution conditions (device,
    `batch_size`, regime), each named: a difference is stated, never fatal, and
    never read as an error in either run. A kept report whose ranks cannot be
    read as it names them is listed with the reason.
    """
    from src.evaluation.cohort_registry import (
        REPORT_FILE,
        report_run_directory,
        report_store_directory,
    )
    from src.evaluation.report_store import (
        _NotTheExpectedBytes,
        read_report,
        scan_report_store,
    )

    cohort_digest = version.identity.sha256
    found: List[Dict[str, Any]] = []
    scan = scan_report_store(version)
    reports_dir = report_store_directory(version, done="Nothing is read from there.")
    for entries in scan.reports.values():
        for run_id, record in entries.values():
            if (record["scoring"].get("mode") != "C"
                    or record["checkpoint_digest"] != checkpoint_sha256
                    or record["cohort_digest"] != cohort_digest):
                continue
            entry: Dict[str, Any] = {"run_id": run_id, "regime": dict(record["regime"])}
            try:
                run_dir = report_run_directory(reports_dir, run_id)
                report, _ = read_report(run_dir / REPORT_FILE,
                                        expected_sha256=record["source_artifact_digest"])
                named = report.get("ranks") or {}
                # The kept report's one reader, its digest compared before parsing.
                rows, _ = read_report(run_dir / str(named.get("file")),
                                      expected_sha256=str(named.get("sha256")))
                kept = {row["sample_id"]: row["rank"] for row in rows}
            except (OSError, ValueError, KeyError, TypeError, _NotTheExpectedBytes) as exc:
                entry["unreadable"] = f"{type(exc).__name__}: {exc}"
                found.append(entry)
                continue
            shared = set(kept) & set(c_ranks)
            entry.update({
                "n_compared": len(shared),
                "n_equal": sum(1 for sample in shared if kept[sample] == c_ranks[sample]),
            })
            found.append(entry)
    return found


__all__ = [
    "BATCH_SIZE",
    "COMPARISON_STATEMENT",
    "METRIC_SCHEMA_VERSION",
    "MODE",
    "ModeDRefused",
    "OUTCOMES",
    "ServedCase",
    "ServedModeResult",
    "TIE_POLICY",
    "derived_ranks",
    "import_counts_block",
    "kept_mode_c_reports",
    "paired",
    "post_import_metrics",
    "rank_served_cases",
    "reached_metrics",
    "run_mode_d",
    "served_input",
    "served_semantics",
]
