"""
Task measurement: a finetuned bundle's ranking of a cohort, the one way.
=======================================================================
`docs/working/PLAN_TEST_RESULTS.md` §8, `docs/working/scorer-retraining/README.md`
E1–E6, U6, G4; provenance contract §9.3 hazard 2.

The run reuses the measurement entry's reads, report, store, usage history and
ledger (`scripts/measure_scorer.py`): the manifest has the same fields as
every measurement, its task, metric schema and scoring identity given as values
(`mode`, `metric_schema_version`, `candidate_construction`, `score_semantics`,
`model_construction`, `canonical_tie_policy_version`), so existing ledgers and
reports stay readable and a task record never groups with a Mode C/D one.

**What is computed.** The bundle's node vectors by the full-view rule over the
whole view (G4), every case bound to the view (F7), ranked against every
disease of the view by the shared scorer (`src.models.tasks.scoring`), average
ranks (E3), and the one metrics module's report (E1, E2, E5): pooled MRR and
top-k, macro recall, best-positive MRR, coverage and the conservative reading.
A case excluded at binding is counted by its reason and enters the
conservative reading as 0 (E4, E5); a contract error refuses (E6).

Module: src/evaluation/task_measurement.py
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from src.models.bundle.metrics import INPUT_EXCLUDED, QueryOutcome, ranking_report

#: The ranking report's schema for task measurements (E1–E5), apart from Mode
#: C's metric schema 1 (`src.evaluation.measurement.METRIC_SCHEMA_VERSION`).
TASK_METRIC_SCHEMA_VERSION = 2
TASK_KS = (1, 3, 5, 10, 20, 50, 100)

#: The manifest values of a disease task measurement.
DISEASE_SEMANTICS: Dict[str, Any] = {
    "mode": "disease",
    "metric_schema_version": TASK_METRIC_SCHEMA_VERSION,
    "candidate_construction": "every disease of the model's graph view",
    "score_semantics": "disease head: -||z_d - z_P||^2 on L2-normalised projected vectors",
    "model_construction": "model bundle (src.models.bundle.build_bundle) from its model spec",
    "canonical_tie_policy_version": "average-rank/v1",
}


def disease_semantics(batch_size: int) -> Dict[str, Any]:
    return {**DISEASE_SEMANTICS, "batch_size": int(batch_size)}


def run_disease_measurement(*, bundle: Any, view: Any, samples: Sequence[Any],
                            manifest: Any, device: torch.device, batch_size: int) -> Any:
    """Rank `samples` with a finetuned disease bundle; a `ModeResult`."""
    from src.evaluation.measurement import ModeResult
    from src.models.tasks.scoring import bind_disease_cases, full_view_vectors, rank_cases
    from src.utils.run_environment import AutocastRegime, observe_autocast_regime

    if bundle.task != "disease":
        raise ValueError(f"a {bundle.task!r} model does not rank diseases")
    recorded = AutocastRegime(enabled=manifest.amp_enabled, dtype=manifest.amp_dtype)
    if observe_autocast_regime(device) != recorded:
        raise RuntimeError("this measurement runs under another autocast regime than its "
                           "manifest records")
    binding = bind_disease_cases(view, samples)
    z = full_view_vectors(bundle, view.edge_index(), view.rel, device)
    diseases = view.type_nodes("disease")
    ranked = rank_cases(bundle, z, binding.cases, diseases, case_chunk=batch_size)
    excluded: List[QueryOutcome] = [
        QueryOutcome(case_id, INPUT_EXCLUDED, n_positives=1, reason=reason)
        for case_id, reason in binding.excluded]
    report = ranking_report(ranked + excluded, TASK_KS)

    metrics = {k: float(v) for k, v in report.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)
               and k not in ("units", "ranked_units", "positive_pairs",
                             "conservative_denominator_pairs")}
    by_case = {o.query_id: o for o in ranked}
    truth = {c.case_id: int(c.truths[0]) for c in binding.cases}
    sample_ids = [s.patient_id for s in samples]
    ranks: List[Optional[float]] = [
        (min(by_case[i].ranks) if i in by_case else None) for i in sample_ids]
    truths = [truth.get(i) for i in sample_ids]
    evidence = {
        "candidate_columns": {"min": int(diseases.numel()), "max": int(diseases.numel())},
        "binding": dict(binding.counts),
        "units_by_status": dict(report["units_by_status"]),
        "units": report["units"],
        "positive_pairs": report["positive_pairs"],
        "conservative_denominator_pairs": report["conservative_denominator_pairs"],
        "graph_view": view.digest,
    }
    return ModeResult(
        manifest=manifest, authoritative_metrics=metrics, n_ranked=int(report["ranked_units"]),
        n_ground_truth_absent=int(binding.counts.get("excluded_truth_outside_view", 0)),
        sampler_evidence=evidence, sample_ids=sample_ids, truth_global_ids=truths,
        canonical_ranks=ranks)


__all__ = ["DISEASE_SEMANTICS", "TASK_KS", "TASK_METRIC_SCHEMA_VERSION", "disease_semantics",
           "run_disease_measurement"]
