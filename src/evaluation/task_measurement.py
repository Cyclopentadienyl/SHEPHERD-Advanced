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

from src.models.bundle.metrics import ranking_report

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


def gene_semantics(batch_size: int, bundle: Any) -> Dict[str, Any]:
    """The gene task's manifest values, η, the SP bounds and the augmentation
    named from the model's spec (PLAN_TEST_RESULTS §8.3: they are part of the
    scoring identity)."""
    gene = bundle.task_head.spec.gene
    augmentation = (f"Eq 10 augmentation K={gene.augmentation_k}, omega={gene.augmentation_omega}"
                    if gene.augmentation else "no augmentation")
    return {
        "mode": "gene",
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
    z = full_view_vectors(bundle, view.edge_index(), view.rel, device)
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
    z = full_view_vectors(bundle, view.edge_index(), view.rel, device)
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


#: The patients-like-me manifest values (PLAN_TEST_RESULTS §8.4).
PATIENTS_LIKE_ME_SEMANTICS: Dict[str, Any] = {
    "mode": "patients_like_me",
    "metric_schema_version": TASK_METRIC_SCHEMA_VERSION,
    "candidate_construction": "every other patient of the measured cohort; self excluded by case id",
    "score_semantics": ("patient tower: -||z_a - z_b||^2 on L2-normalised projected vectors; a "
                        "pair is positive when the causal-gene sets intersect"),
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
        raise ValueError("no patient of the cohort records a causal gene, so no pair can be "
                         "positive; nothing was written")
    _check_regime(device, manifest)
    cases = binding.cases
    z = full_view_vectors(bundle, view.edge_index(), view.rel, device)
    ranked = rank_patients(bundle, z, cases, cases, chunk=batch_size,
                           unranked=binding.unranked_positives)
    report = ranking_report(ranked + binding.excluded_outcomes(), TASK_KS)
    return _mode_result(
        manifest=manifest, report=report, ranked=ranked, binding=binding, samples=samples,
        truth={}, evidence={"graph_view": view.digest,
                            "candidate_columns": {"min": max(len(cases) - 1, 0),
                                                  "max": max(len(cases) - 1, 0)}})


#: The task measurements this entry runs: semantics and the run, by task.
TASK_RUNS = {"disease": run_disease_measurement, "gene": run_gene_measurement,
             "patients_like_me": run_patients_like_me_measurement}


def task_semantics(task: str, batch_size: int, bundle: Any) -> Dict[str, Any]:
    if task == "disease":
        return disease_semantics(batch_size)
    if task == "gene":
        return gene_semantics(batch_size, bundle)
    if task == "patients_like_me":
        return {**PATIENTS_LIKE_ME_SEMANTICS, "batch_size": int(batch_size)}
    raise ValueError(f"no task measurement for {task!r}")


__all__ = ["DISEASE_SEMANTICS", "PATIENTS_LIKE_ME_SEMANTICS", "TASK_KS",
           "TASK_METRIC_SCHEMA_VERSION", "TASK_RUNS", "disease_semantics", "gene_semantics",
           "run_disease_measurement", "run_gene_measurement",
           "run_patients_like_me_measurement", "task_semantics"]
