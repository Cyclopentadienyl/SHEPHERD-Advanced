"""
The one metrics module: link prediction and task ranking.
=========================================================
`docs/working/scorer-retraining/README.md` A5, F3, F4 and §2.6 (E1–E7).
Training validation, measurement and reports all call this module; no entry
computes these numbers a second way.

**Link prediction (whole-set, F4).** Accuracy at raw score > 0 (equal to the
binary micro-F1 the paper selects on, A5), ROC-AUC by the Mann–Whitney
statistic with average ranks for ties, and AP as the step sum
`Σ (R_n − R_{n−1}) P_n` over distinct thresholds (scikit-learn's definition).
A class missing makes AUC and AP undefined: `None`, never a number.

**Ranking (E1–E5).** Ranks are average ranks over the candidate set
(scipy `rankdata(method="average")`, E3), so tied candidates share a rank and a
tie straddling `k` is decided by its average rank. Readings:

- *pooled*: over every (query, positive) pair, MRR = mean 1/rank and top-k =
  share with rank ≤ k (E1, the source's `train_utils.py:17-24`);
- *macro recall@k*: per query, positives within the top-k threshold ÷ all its
  relevant items (`n_positives`, ranked or not), averaged over eligible
  queries (E2); *macro precision@k* is the same count ÷ k, never called recall;
- *best-positive MRR*: per query, 1 / its best positive rank, averaged (an E7
  oracle reading, not a selection metric);
- *coverage and conservative pooled MRR*: units fixed in advance whose truth
  is assessable but the model could not rank, and relevant items a ranked
  query could not rank, contribute 0 (E5).

States are kept apart (E4): only `ranked` queries enter pooled and macro
figures; the others are counted by status, and none becomes a recall of 0.

Module: src/models/bundle/metrics.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

#: Query states (E4). `ranked` is the only one with ranks.
RANKED = "ranked"
NO_OTHER_POSITIVE = "no_other_positive"
NO_RELIABLE_TRUTH = "no_reliable_truth"
INPUT_EXCLUDED = "input_excluded"
NOT_SCORABLE = "not_scorable"
MODEL_FAILED = "model_failed"
QUERY_STATES = (RANKED, NO_OTHER_POSITIVE, NO_RELIABLE_TRUTH, INPUT_EXCLUDED, NOT_SCORABLE,
                MODEL_FAILED)
#: States whose truth is assessable, so a conservative reading counts them as 0.
ASSESSABLE_UNRANKED = (NOT_SCORABLE, MODEL_FAILED, INPUT_EXCLUDED)

DEFAULT_KS = (1, 3, 5, 10)


def lp_metrics(positive_raw: Sequence[float], negative_raw: Sequence[float]) -> Dict[str, Any]:
    """Whole-set LP metrics from raw (pre-tanh) scores."""
    pos = np.asarray(positive_raw, dtype=np.float64)
    neg = np.asarray(negative_raw, dtype=np.float64)
    n_pos, n_neg = pos.size, neg.size
    total = n_pos + n_neg
    if total == 0:
        raise ValueError("no LP pair was evaluated; there is nothing to measure")
    correct = int((pos > 0).sum() + (neg <= 0).sum())
    return {
        "lp_accuracy": correct / total,
        "lp_roc_auc": roc_auc(pos, neg),
        "lp_average_precision": average_precision(pos, neg),
        "lp_positives": int(n_pos),
        "lp_negatives": int(n_neg),
    }


def roc_auc(pos: np.ndarray, neg: np.ndarray) -> Optional[float]:
    from scipy.stats import rankdata

    if pos.size == 0 or neg.size == 0:
        return None
    ranks = rankdata(np.concatenate([pos, neg]), method="average")
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def average_precision(pos: np.ndarray, neg: np.ndarray) -> Optional[float]:
    if pos.size == 0:
        return None
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
    order = np.argsort(-scores, kind="stable")
    scores, labels = scores[order], labels[order]
    # The last index of each run of equal scores is a threshold.
    distinct = np.flatnonzero(np.diff(scores)) if scores.size > 1 else np.array([], dtype=int)
    thresholds = np.concatenate([distinct, [scores.size - 1]])
    tp = np.cumsum(labels)[thresholds]
    fp = (thresholds + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / pos.size
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def average_ranks(scores: Sequence[float]) -> np.ndarray:
    """Average ranks, 1 for the highest score (E3). Exact ties only: nearly
    equal scores are never merged."""
    from scipy.stats import rankdata

    return rankdata(-np.asarray(scores, dtype=np.float64), method="average")


def positive_ranks(scores: Sequence[float], positive_index: Iterable[int]) -> List[float]:
    ranks = average_ranks(scores)
    return [float(ranks[i]) for i in positive_index]


@dataclass
class QueryOutcome:
    """One evaluation unit (a case, a query patient)."""

    query_id: str
    status: str
    ranks: List[float] = field(default_factory=list)
    #: Positives the truth names, ranked or not (for the conservative reading).
    n_positives: int = 0
    reason: Optional[str] = None
    candidates: Optional[int] = None

    def __post_init__(self) -> None:
        if self.status not in QUERY_STATES:
            raise ValueError(f"unknown query state {self.status!r}")
        if self.status == RANKED and not self.ranks:
            raise ValueError("a ranked query has at least one positive rank")
        if self.status != RANKED and self.ranks:
            raise ValueError("only a ranked query carries ranks")
        if self.status == RANKED and self.n_positives == 0:
            self.n_positives = len(self.ranks)
        if self.n_positives < len(self.ranks):
            raise ValueError("a query has fewer relevant items than ranked positives")


def ranking_report(outcomes: Sequence[QueryOutcome], ks: Sequence[int] = DEFAULT_KS
                   ) -> Dict[str, Any]:
    """E1, E2 and E5 over `outcomes`, with every count named."""
    by_status = {s: 0 for s in QUERY_STATES}
    for o in outcomes:
        by_status[o.status] += 1
    ranked = [o for o in outcomes if o.status == RANKED]
    pairs = [r for o in ranked for r in o.ranks]
    report: Dict[str, Any] = {
        "units": len(outcomes),
        "units_by_status": by_status,
        "ranked_units": len(ranked),
        "positive_pairs": len(pairs),
        "coverage": (len(ranked) / len(outcomes)) if outcomes else None,
    }
    if pairs:
        arr = np.asarray(pairs, dtype=np.float64)
        report["pooled_mrr"] = float(np.mean(1.0 / arr))
        for k in ks:
            report[f"pooled_top{k}"] = float(np.mean(arr <= k))
        for k in ks:
            hits = [int(np.sum(np.asarray(o.ranks) <= k)) for o in ranked]
            report[f"macro_recall@{k}"] = float(np.mean(
                [h / o.n_positives for h, o in zip(hits, ranked)]))
            report[f"macro_precision@{k}"] = float(np.mean([h / k for h in hits]))
        report["macro_mrr"] = float(np.mean([np.mean(1.0 / np.asarray(o.ranks)) for o in ranked]))
        report["best_positive_mrr"] = float(np.mean([1.0 / min(o.ranks) for o in ranked]))
    else:
        report["pooled_mrr"] = None
        for k in ks:
            report[f"pooled_top{k}"] = None
            report[f"macro_recall@{k}"] = None
            report[f"macro_precision@{k}"] = None
        report["macro_mrr"] = None
        report["best_positive_mrr"] = None
    # Conservative pooled MRR (E5): assessable but unranked units, and the
    # unranked relevant items of ranked units, add a reciprocal rank of 0.
    denominator = (sum(o.n_positives for o in ranked)
                   + sum(o.n_positives for o in outcomes if o.status in ASSESSABLE_UNRANKED))
    report["conservative_pooled_mrr"] = (
        float(np.sum(1.0 / np.asarray(pairs, dtype=np.float64)) / denominator)
        if denominator else None)
    report["conservative_denominator_pairs"] = int(denominator)
    return report


__all__ = [
    "ASSESSABLE_UNRANKED",
    "DEFAULT_KS",
    "INPUT_EXCLUDED",
    "MODEL_FAILED",
    "NOT_SCORABLE",
    "NO_OTHER_POSITIVE",
    "NO_RELIABLE_TRUTH",
    "QUERY_STATES",
    "QueryOutcome",
    "RANKED",
    "average_precision",
    "average_ranks",
    "lp_metrics",
    "positive_ranks",
    "ranking_report",
    "roc_auc",
]
