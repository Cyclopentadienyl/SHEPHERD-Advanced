"""
SHEPHERD-Advanced Evaluation Metrics
====================================
Ranking metrics for the ShepherdGNN path's disease ranking.

Module: src/utils/metrics.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/utils/metrics.py

Purpose:
    Compute the standard IR metrics of a ranked candidate list.

Components:
    - RankingMetrics: Hits@k, MRR, NDCG for candidate ranking

Dependencies:
    - numpy: Array operations and statistical computations
    - typing: Type hints

Input:
    - predictions: Sequence[Sequence[str]] - Ranked candidate lists per sample
    - ground_truths: Sequence[str] - Correct answers per sample

Output:
    - Dict[str, float]: Metric name to value mapping

Called by:
    - src/training/trainer.py (the ShepherdGNN validation pass)
    - src/evaluation/measurement.py (Mode C)

Version: 1.0.0
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Type Aliases
# =============================================================================
ScoreArray = NDArray[np.float64]
RankArray = NDArray[np.int64]


# =============================================================================
# Core Ranking Metrics
# =============================================================================
@dataclass
class RankingMetrics:
    """
    排序評估指標計算器

    適用於疾病診斷排序任務：給定患者表型，排序候選疾病
    """

    # Default k values for Hits@k
    default_k_values: Tuple[int, ...] = (1, 3, 5, 10, 20)

    def compute_from_ranks(
        self,
        ranks: Sequence[int],
        k_values: Optional[Sequence[int]] = None,
    ) -> Dict[str, float]:
        """Rank-based metrics from ranks that were already determined elsewhere.

        The prediction-list entry points above take truncated top-k lists. A truth
        at the last kept position still contributes ``1/K``; what a truncated list
        loses is everything past it — **every true rank beyond K, and absence from
        the candidate set entirely, all collapse to a contribution of zero and
        become indistinguishable from each other.** This entry point takes ranks
        directly and so can report an untruncated MRR, a mean rank, and Hits@k for
        any k — including k larger than any list a caller would carry.

        **Ranking policy deliberately lives outside this module.** Score ordering,
        tie breaking, candidate identity and what to do about a ground truth that
        is absent from the candidate set are measurement decisions; they belong to
        the caller that knows which measurement mode it is running. This function
        receives ranks and nothing else, and stays unaware of any of that.

        Args:
            ranks: 1-based rank of the ground truth, one per sample. Every sample
                that has a rank appears here; a sample whose ground truth was not
                in the candidate set has no rank and is **excluded by the caller**,
                which reports that outcome separately.
            k_values: k values for Hits@k. Defaults to ``default_k_values``.

        Returns:
            ``{"mean_rank", "mrr", "hits@k"...}`` — all over the same denominator,
            ``len(ranks)``.

        Raises:
            ValueError: on empty input, or a rank or k that is not a positive
                integer.
        """
        k_values = tuple(k_values) if k_values is not None else self.default_k_values

        # An empty cohort has no mean rank and no MRR. Returning 0.0 would be a
        # fabricated number, which is the defect this project has already had to
        # correct once in the evaluation report.
        if len(ranks) == 0:
            raise ValueError("compute_from_ranks requires at least one rank")

        # bool is a subclass of int, so `isinstance(True, int)` is True and a
        # boolean would otherwise sail through as rank 1.
        for label, values in (("rank", ranks), ("k", k_values)):
            for value in values:
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                    raise ValueError(
                        f"{label} values must be integers, not {type(value).__name__}: {value!r}"
                    )
                if value < 1:
                    raise ValueError(f"{label} values must be >= 1; got {value}")

        rank_array = np.asarray(ranks, dtype=np.int64)
        n = float(rank_array.size)

        metrics: Dict[str, float] = {
            "mean_rank": float(rank_array.mean()),
            "mrr": float((1.0 / rank_array).sum() / n),
        }
        for k in k_values:
            metrics[f"hits@{k}"] = float((rank_array <= k).sum() / n)
        return metrics

    def hits_at_k(
        self,
        predictions: Sequence[Sequence[str]],
        ground_truths: Sequence[str],
        k: int,
    ) -> float:
        """
        計算 Hits@k：正確答案出現在前 k 個預測中的比例

        Args:
            predictions: 每個樣本的排序預測列表 [(pred1, pred2, ...), ...]
            ground_truths: 每個樣本的正確答案
            k: 取前 k 個預測

        Returns:
            Hits@k 分數 (0-1)
        """
        if len(predictions) == 0:
            return 0.0

        hits = 0
        for preds, truth in zip(predictions, ground_truths):
            top_k = list(preds)[:k]
            if truth in top_k:
                hits += 1

        return hits / len(predictions)

    def mean_reciprocal_rank(
        self,
        predictions: Sequence[Sequence[str]],
        ground_truths: Sequence[str],
    ) -> float:
        """
        計算 Mean Reciprocal Rank (MRR)

        MRR = (1/N) * Σ(1/rank_i)

        Args:
            predictions: 每個樣本的排序預測列表
            ground_truths: 每個樣本的正確答案

        Returns:
            MRR 分數 (0-1)
        """
        if len(predictions) == 0:
            return 0.0

        reciprocal_ranks = []
        for preds, truth in zip(predictions, ground_truths):
            try:
                rank = list(preds).index(truth) + 1  # 1-indexed
                reciprocal_ranks.append(1.0 / rank)
            except ValueError:
                # 正確答案不在預測列表中
                reciprocal_ranks.append(0.0)

        return float(np.mean(reciprocal_ranks))

    def ndcg_at_k(
        self,
        predictions: Sequence[Sequence[str]],
        ground_truths: Sequence[str],
        k: int,
        relevance_scores: Optional[Sequence[Dict[str, float]]] = None,
    ) -> float:
        """
        計算 Normalized Discounted Cumulative Gain (NDCG@k)

        對於單一正確答案的情況 (binary relevance):
        - 正確答案 relevance = 1, 其他 = 0
        - DCG@k = rel_i / log2(rank_i + 1)
        - IDCG@k = 1 / log2(2) = 1 (如果 k >= 1)

        Args:
            predictions: 每個樣本的排序預測列表
            ground_truths: 每個樣本的正確答案
            k: 取前 k 個預測
            relevance_scores: 可選，每個樣本的相關性分數字典

        Returns:
            NDCG@k 分數 (0-1)
        """
        if len(predictions) == 0:
            return 0.0

        ndcg_scores = []

        for i, (preds, truth) in enumerate(zip(predictions, ground_truths)):
            top_k = list(preds)[:k]

            if relevance_scores is not None and i < len(relevance_scores):
                # 使用提供的相關性分數
                rel_dict = relevance_scores[i]
                rels = [rel_dict.get(pred, 0.0) for pred in top_k]
            else:
                # Binary relevance: 正確答案 = 1, 其他 = 0
                rels = [1.0 if pred == truth else 0.0 for pred in top_k]

            # DCG@k
            dcg = self._dcg(rels)

            # IDCG@k (理想情況：相關項目排在最前面)
            ideal_rels = sorted(rels, reverse=True)
            idcg = self._dcg(ideal_rels)

            # 對於 binary relevance 且只有一個正確答案
            if idcg == 0:
                # 如果 truth 不在 top_k 中，計算假設它排第一的 IDCG
                if truth in list(preds):
                    idcg = 1.0  # 1 / log2(2)
                else:
                    idcg = 1.0  # 避免除以零

            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)

        return float(np.mean(ndcg_scores))

    def _dcg(self, relevances: Sequence[float]) -> float:
        """計算 Discounted Cumulative Gain"""
        dcg = 0.0
        for i, rel in enumerate(relevances):
            # 使用 log2(i + 2) 因為 i 是 0-indexed
            dcg += rel / np.log2(i + 2)
        return dcg

    def compute_all(
        self,
        predictions: Sequence[Sequence[str]],
        ground_truths: Sequence[str],
        k_values: Optional[Tuple[int, ...]] = None,
    ) -> Dict[str, float]:
        """
        計算所有排序指標

        Args:
            predictions: 每個樣本的排序預測列表
            ground_truths: 每個樣本的正確答案
            k_values: Hits@k 的 k 值列表

        Returns:
            所有指標的字典
        """
        k_values = k_values or self.default_k_values

        metrics = {
            "mrr": self.mean_reciprocal_rank(predictions, ground_truths),
        }

        for k in k_values:
            metrics[f"hits@{k}"] = self.hits_at_k(predictions, ground_truths, k)
            metrics[f"ndcg@{k}"] = self.ndcg_at_k(predictions, ground_truths, k)

        return metrics


# =============================================================================
# Module Exports
# =============================================================================
__all__ = [
    "RankingMetrics",
]
