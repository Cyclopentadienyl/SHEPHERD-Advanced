"""Backlog item 1d — same-batch differential calibration.

The acceptance gate for the Mode A harness, after frozen-evaluator bit parity was
found to be unexecutable. No checkpoint in the scanned family carries the
`metadata` / `in_channels_dict` keys either loader needs (BACKLOG §3.1), so the
oracle cannot be run and cannot be reproduced. This module is the replacement
acceptance: hand **the same batches** to the trainer's own validation calculation
and to the Mode A harness, and compare them per sample.

**Why this is not a tautology.** The two paths compute the same quantity through
two separately written implementations, and the separation is enforced by the
build rather than by discipline. `.import-linter.ini` orders `src.evaluation`
above `src.inference` above `src.training`, and a lower layer may not import a
higher one — so `src/training/` **cannot** reach `masked_mean_pool` or
`cosine_score_matrix` even if someone wanted it to. The trainer keeps its own
inline `F.normalize` + `torch.mm`; Mode A goes through the served primitives.
`make lint-imports` fails if that ever stops being true.

What is deliberately *shared* is the **metric**: both sides reach
`RankingMetrics.mean_reciprocal_rank`. A second expression of `1/rank` here would
have to be trusted to agree with the one the trainer already uses, and a
calibration whose two sides disagree about arithmetic measures nothing useful.

**What agreement is checked on**, per sample and in this order — the first three
are per-sample and the fourth is the aggregate they imply:

  1. the top-`LEGACY_TRUNCATION_K` row of subgraph-local column indices;
  2. the ground-truth id, compared in **global** space. Both sides are translated
     through the *same* `original_indices` gather, so this does **not** test the
     translation — that cancels. It tests that the trainer's `diagnosis_targets`
     and Mode A's `disease_ids_local` are the same ids: the gather is injective
     (each subgraph row maps to one distinct global id), so equality after
     translation holds exactly when equality before it does. `to_global_ids` is
     tested on its own in `tests/unit/test_measurement_ranking.py`;
  3. the reciprocal rank in truncated-local space;
  4. aggregate MRR — the trainer's `compute_all()["mrr"]` against Mode A's
     `legacy_mrr_truncated_at_{K}`.

**Bit-exactness is a contract only when AMP is off, and that is not a caveat —
it is the thing this module has to report.** `Trainer._run_evaluation_pass`
forwards inside `autocast(..., enabled=self.use_amp)`; the Mode A traversal has no
autocast at all. `trainer.py:380` resolves `use_amp = config.use_amp and
device.type == "cuda"`, so on CPU the two are both fp32 and may be compared
exactly, while on CUDA the default `float16` autocast makes the trainer's scores
differ in the last bits and reorders anything close to a tie. A disagreement in
that state is a **measurement of AMP's effect on the ranking**, not a harness
fault, so `DifferentialResult` records the observed AMP state beside the verdict
and `bit_exact_contract` says which of the two questions the run answered.
Deciding what to do about that is item 1e's `amp_dtype` manifest field; refusing
to run under AMP would only make the effect unmeasurable.

**Scope.** One comparison function and its result type. No evaluation framework,
no protocol hierarchy, no runner, no artifact registry, and nothing under
`src/training/` or `src/evaluation/measurement.py` is modified — this module is
purely additive so that a failure here can never be a failure it introduced
elsewhere. The institutional CUDA run that consumes it is item 7a, which is
blocked on a designated loadable checkpoint and on item 10's evidence.

Module: src/evaluation/differential.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.evaluation.measurement import (
    MeasurementManifest,
    ModeAResult,
    run_mode_a,
    to_global_ids,
)

def legacy_mrr_key() -> str:
    """The key `run_modes_ab` builds its legacy metric under.

    Resolved **per call**, not frozen at import. `run_modes_ab` builds this key
    from the module global with an f-string every time it runs, so a module-level
    constant here would go stale the moment anything changed the truncation — and
    would then look up a key that no longer exists while still claiming to report
    a comparison. Reading it through the module is what keeps the two in step.
    """
    from src.evaluation import measurement as _measurement

    return f"legacy_mrr_truncated_at_{_measurement.LEGACY_TRUNCATION_K}"


@dataclass(frozen=True)
class SampleDisagreement:
    """One sample on which the two paths did not produce the same thing.

    `kind` separates the failures because they have different causes and a single
    "sample 41 differs" would send the reader to the wrong place: a `top_k`
    difference is scoring or ranking, a `truth` difference is id translation or
    batch wiring, and a `reciprocal_rank` difference with an equal `top_k` is the
    metric being fed differently.
    """

    index: int
    sample_id: str
    kind: str
    trainer_value: Any
    mode_a_value: Any

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "sample_id": self.sample_id,
            "kind": self.kind,
            "trainer": self.trainer_value,
            "mode_a": self.mode_a_value,
        }


@dataclass(frozen=True)
class DifferentialResult:
    """What one same-batch comparison established, and under what conditions.

    `agreed` is the narrow machine-checkable fact: zero disagreements over the
    whole cohort. It is deliberately *not* named "calibration_passed" — whether a
    run is fit for institutional acceptance depends on the checkpoint and cohort
    it consumed, which nothing here can observe.
    """

    n_samples: int
    agreed: bool
    disagreements: List[SampleDisagreement]
    trainer_mrr: float
    mode_a_mrr: float
    mrr_absolute_difference: float
    amp_enabled: bool
    amp_dtype: str
    bit_exact_contract: bool
    """True exactly when AMP was off, i.e. when an exact comparison was the
    question being asked. Under AMP the two paths run at different precisions by
    construction and `agreed` reports whether that happened to change the ranking
    on this cohort — a different and still worth-recording question."""

    device: str
    n_disagreements_by_kind: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "agreed": self.agreed,
            "n_disagreements": len(self.disagreements),
            "n_disagreements_by_kind": self.n_disagreements_by_kind,
            "disagreements": [d.to_dict() for d in self.disagreements],
            "trainer_mrr": self.trainer_mrr,
            "mode_a_mrr": self.mode_a_mrr,
            "mrr_absolute_difference": self.mrr_absolute_difference,
            "amp_enabled": self.amp_enabled,
            "amp_dtype": self.amp_dtype,
            "bit_exact_contract": self.bit_exact_contract,
            "device": self.device,
        }


def _require_rerunnable(batches: Any) -> Sequence[Dict[str, Any]]:
    """Refuse anything that can only be traversed once.

    **This is the guard the whole module rests on.** Both paths iterate the batch
    stream independently, so a generator would be drained by whichever ran first
    and the second would see an empty stream. That does not fail loudly: Mode A's
    `_assert_cohort_is_intact` would raise on one ordering, but on the other the
    trainer pass returns no predictions and an empty metric dict, and a comparison
    of two empty cohorts is trivially "agreed". A calibration that passes by
    measuring nothing is the exact failure this module exists to make impossible.
    """
    if isinstance(batches, (str, bytes)) or not isinstance(batches, Sequence):
        raise TypeError(
            f"batches must be a re-iterable sequence (list/tuple), not "
            f"{type(batches).__name__}. Both paths traverse it independently, so a "
            "one-shot iterator would hand the second caller an empty stream and the "
            "comparison would pass by measuring nothing"
        )
    if len(batches) == 0:
        raise ValueError(
            "no batches to compare; an empty cohort would report agreement without "
            "having compared anything"
        )
    return batches


def _trainer_global_truths(
    batches: Sequence[Dict[str, Any]], local_truths: Sequence[str]
) -> List[int]:
    """Translate the trainer's local truth ids to global space, batch by batch.

    The trainer emits `str(disease_ids[i])` in **subgraph-local** space and keeps
    no `original_indices`; Mode A records `truth_global_ids`. Comparing them
    therefore needs one translation, and it goes through `to_global_ids` — the
    same function the ranking path uses, which already refuses out-of-range and
    non-integer ids — rather than a second index expression written here.

    Row order is the trainer's own: it appends `range(scores.size(0))` per batch
    in dataloader order, which is what item 1b's batch-ordering test freezes.
    """
    import torch as _torch

    global_truths: List[int] = []
    offset = 0
    for batch_data in batches:
        original = batch_data["original_indices"]["disease"]
        n_rows = int(batch_data["batch"]["disease_ids"].size(0))
        window = local_truths[offset : offset + n_rows]
        if len(window) != n_rows:
            raise ValueError(
                f"the trainer produced {len(local_truths)} truth rows but the batch "
                f"stream declares at least {offset + n_rows}. The two are not "
                "describing the same cohort and no per-sample comparison is meaningful"
            )
        locals_tensor = _torch.tensor(
            [int(value) for value in window],
            dtype=original.dtype,
            device=original.device,
        )
        global_truths.extend(to_global_ids(original, locals_tensor).tolist())
        offset += n_rows

    if offset != len(local_truths):
        raise ValueError(
            f"the batch stream accounts for {offset} rows but the trainer produced "
            f"{len(local_truths)}. A comparison over mismatched cohorts would be "
            "silently misaligned rather than wrong in an obvious place"
        )
    return global_truths


def _batch_local_truths(batches: Sequence[Dict[str, Any]]) -> List[int]:
    """The subgraph-local truth ids, read straight off the batch stream.

    `ModeAResult` does not carry its `legacy_truth_local` list — it consumes it to
    build the legacy metric and keeps only the global translation. This reads the
    **input** that Mode A read (`batch["disease_ids"]`, flattened in dataloader
    order, which is what `run_modes_ab` extends its own list from), rather than
    reconstructing Mode A's output or widening `ModeAResult` to expose it. Adding a
    field to a reviewed result type in order to test it is how a schema grows to
    fit its tests.

    This is only used for the per-sample reciprocal rank. Whether the *trainer*
    agrees about the truth is checked separately and in global space, so a wrong
    truth cannot hide behind this being read from the same place both paths read it.
    """
    truths: List[int] = []
    for batch_data in batches:
        truths.extend(int(value) for value in batch_data["batch"]["disease_ids"].tolist())
    return truths


def compare_trainer_against_mode_a(
    trainer: Any,
    batches: Sequence[Dict[str, Any]],
    manifest: MeasurementManifest,
    device: Optional[Any] = None,
    mode_a_result: Optional[ModeAResult] = None,
) -> DifferentialResult:
    """Run both paths over one materialised batch list and compare them per sample.

    `trainer` is a real `Trainer`; its extracted `_run_evaluation_pass` is the
    reference, deliberately, because writing a fourth expression of a calculation
    that already exists three times would leave the newest copy the only untested
    one. Reaching a private method across a package boundary is the one thing this
    module is permitted to do that ordinary callers are not — the alternative is a
    reimplementation, which is worse.

    **Execution order does not matter and is not relied upon.** Both paths put the
    model in eval mode, neither consumes RNG (the batches are already drawn), and
    neither mutates the batch dicts. Order-independence is a property of the
    inputs, not something the caller has to arrange.

    Pass `mode_a_result` to reuse a Mode A run the caller already has; it must have
    come from these same batches, and nothing here can check that, which is why the
    default is to run it.
    """
    import torch as _torch

    batches = _require_rerunnable(batches)

    trainer.model.eval()
    trainer_pass = trainer._run_evaluation_pass(batches)
    if mode_a_result is None:
        mode_a_result = run_mode_a(trainer.model, batches, manifest, device=device)

    n_trainer = len(trainer_pass.predictions)
    n_mode_a = len(mode_a_result.legacy_top_k_local)
    if n_trainer != n_mode_a:
        raise ValueError(
            f"the trainer scored {n_trainer} samples and Mode A scored {n_mode_a} "
            "over the same batches. That is a harness fault, not a model "
            "disagreement, and no per-sample alignment exists to report"
        )
    if n_trainer != len(mode_a_result.sample_ids):
        raise ValueError(
            f"Mode A reported {len(mode_a_result.sample_ids)} sample ids for "
            f"{n_mode_a} ranked rows; the cohort is not internally consistent"
        )

    trainer_globals = _trainer_global_truths(batches, trainer_pass.ground_truths)
    local_truths = _batch_local_truths(batches)
    if len(local_truths) != n_trainer:
        raise ValueError(
            f"the batch stream carries {len(local_truths)} truth rows for a cohort "
            f"of {n_trainer} scored samples; the comparison would be misaligned"
        )

    from src.utils.metrics import RankingMetrics

    metrics = RankingMetrics()
    disagreements: List[SampleDisagreement] = []

    for i in range(n_trainer):
        sample_id = mode_a_result.sample_ids[i]

        trainer_row = [int(value) for value in trainer_pass.predictions[i]]
        mode_a_row = list(mode_a_result.legacy_top_k_local[i])
        if trainer_row != mode_a_row:
            disagreements.append(
                SampleDisagreement(
                    index=i,
                    sample_id=sample_id,
                    kind="top_k",
                    trainer_value=trainer_row,
                    mode_a_value=mode_a_row,
                )
            )

        trainer_truth = trainer_globals[i]
        mode_a_truth = int(mode_a_result.truth_global_ids[i])
        if trainer_truth != mode_a_truth:
            disagreements.append(
                SampleDisagreement(
                    index=i,
                    sample_id=sample_id,
                    kind="truth",
                    trainer_value=trainer_truth,
                    mode_a_value=mode_a_truth,
                )
            )

        # Through the shared metric, on one-sample cohorts, so that a per-sample
        # disagreement is reported where it happened rather than only as a shifted
        # aggregate. Both sides are fed in their own native representation — the
        # trainer's strings and Mode A's ints — because that is exactly what each
        # path hands the metric in production, and normalising them here would test
        # a conversion this module invented.
        trainer_rr = metrics.mean_reciprocal_rank(
            [trainer_pass.predictions[i]], [trainer_pass.ground_truths[i]]
        )
        mode_a_rr = metrics.mean_reciprocal_rank([mode_a_row], [local_truths[i]])
        if trainer_rr != mode_a_rr:
            disagreements.append(
                SampleDisagreement(
                    index=i,
                    sample_id=sample_id,
                    kind="reciprocal_rank",
                    trainer_value=trainer_rr,
                    mode_a_value=mode_a_rr,
                )
            )

    trainer_mrr = float(trainer_pass.ranking_metrics.get("mrr", 0.0))
    mode_a_mrr = float(mode_a_result.legacy_metrics[legacy_mrr_key()])

    by_kind: Dict[str, int] = {}
    for item in disagreements:
        by_kind[item.kind] = by_kind.get(item.kind, 0) + 1

    amp_enabled = bool(getattr(trainer, "use_amp", False))
    amp_dtype = str(getattr(trainer, "amp_dtype", _torch.float32))

    return DifferentialResult(
        n_samples=n_trainer,
        agreed=not disagreements,
        disagreements=disagreements,
        trainer_mrr=trainer_mrr,
        mode_a_mrr=mode_a_mrr,
        mrr_absolute_difference=abs(trainer_mrr - mode_a_mrr),
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        bit_exact_contract=not amp_enabled,
        device=str(trainer.device),
        n_disagreements_by_kind=by_kind,
    )
