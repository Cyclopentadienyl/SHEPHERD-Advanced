"""
SHEPHERD-Advanced Evaluation Metrics
====================================
用於疾病診斷模型的評估指標

包含:
- Hits@k: 前 k 個候選中的命中率
- MRR: Mean Reciprocal Rank
- NDCG: Normalized Discounted Cumulative Gain
- 本體約束違反率
- 證據覆蓋率

版本: 1.0.0
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
# Medical-Specific Metrics
# =============================================================================
@dataclass
class OntologyViolationMetrics:
    """
    本體約束違反指標

    檢查預測是否違反本體結構約束（如 HPO 的 is-a 關係）
    """

    def phenotype_consistency(
        self,
        predicted_phenotypes: Sequence[str],
        ontology_ancestors: Dict[str, set],
    ) -> float:
        """
        計算表型一致性：預測的表型是否與本體層次一致

        如果預測了一個表型及其祖先，這是冗餘的

        Args:
            predicted_phenotypes: 預測的表型 ID 列表
            ontology_ancestors: {phenotype_id: set(ancestor_ids)}

        Returns:
            一致性分數 (0-1)，1 表示完全無冗餘
        """
        if not predicted_phenotypes:
            return 1.0

        pred_set = set(predicted_phenotypes)
        redundant_count = 0

        for pheno in predicted_phenotypes:
            ancestors = ontology_ancestors.get(pheno, set())
            if ancestors & pred_set:
                # 該表型有祖先也在預測中，冗餘
                redundant_count += 1

        return 1.0 - (redundant_count / len(predicted_phenotypes))

    def disease_gene_consistency(
        self,
        predicted_diseases: Sequence[str],
        predicted_genes: Sequence[str],
        disease_gene_map: Dict[str, set],
    ) -> float:
        """
        計算疾病-基因一致性：預測的基因是否與預測的疾病相關

        Args:
            predicted_diseases: 預測的疾病 ID 列表
            predicted_genes: 預測的基因列表
            disease_gene_map: {disease_id: set(associated_gene_ids)}

        Returns:
            一致性分數 (0-1)
        """
        if not predicted_genes:
            return 1.0

        # 收集所有預測疾病的相關基因
        expected_genes = set()
        for disease in predicted_diseases:
            expected_genes.update(disease_gene_map.get(disease, set()))

        if not expected_genes:
            return 0.0

        # 計算預測基因與期望基因的重疊
        overlap = len(set(predicted_genes) & expected_genes)
        return overlap / len(predicted_genes)

    def compute_violation_rate(
        self,
        predictions: Sequence[Dict[str, Any]],
        ontology_ancestors: Optional[Dict[str, set]] = None,
        disease_gene_map: Optional[Dict[str, set]] = None,
    ) -> Dict[str, float]:
        """
        計算整體約束違反率

        Args:
            predictions: 預測結果列表，每個包含 phenotypes, diseases, genes
            ontology_ancestors: 本體祖先映射
            disease_gene_map: 疾病-基因映射

        Returns:
            違反率指標字典
        """
        metrics = {}

        if ontology_ancestors is not None:
            consistencies = []
            for pred in predictions:
                if "phenotypes" in pred:
                    c = self.phenotype_consistency(
                        pred["phenotypes"], ontology_ancestors
                    )
                    consistencies.append(c)
            if consistencies:
                metrics["phenotype_consistency"] = float(np.mean(consistencies))
                metrics["phenotype_violation_rate"] = 1.0 - metrics["phenotype_consistency"]

        if disease_gene_map is not None:
            consistencies = []
            for pred in predictions:
                if "diseases" in pred and "genes" in pred:
                    c = self.disease_gene_consistency(
                        pred["diseases"], pred["genes"], disease_gene_map
                    )
                    consistencies.append(c)
            if consistencies:
                metrics["disease_gene_consistency"] = float(np.mean(consistencies))
                metrics["disease_gene_violation_rate"] = 1.0 - metrics["disease_gene_consistency"]

        return metrics


@dataclass
class EvidenceCoverageMetrics:
    """
    證據覆蓋率指標

    評估推理路徑對輸入表型的解釋程度
    """

    def phenotype_coverage(
        self,
        input_phenotypes: Sequence[str],
        explained_phenotypes: Sequence[str],
    ) -> float:
        """
        計算表型覆蓋率：有多少輸入表型被推理路徑解釋

        Args:
            input_phenotypes: 輸入的表型 ID 列表
            explained_phenotypes: 推理路徑中涉及的表型

        Returns:
            覆蓋率 (0-1)
        """
        if not input_phenotypes:
            return 1.0

        covered = len(set(input_phenotypes) & set(explained_phenotypes))
        return covered / len(input_phenotypes)

    def evidence_strength(
        self,
        path_confidences: Sequence[float],
    ) -> Dict[str, float]:
        """
        計算證據強度統計

        Args:
            path_confidences: 每條推理路徑的置信度

        Returns:
            證據強度指標
        """
        if not path_confidences:
            return {
                "mean_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "std_confidence": 0.0,
            }

        arr = np.array(path_confidences)
        return {
            "mean_confidence": float(np.mean(arr)),
            "min_confidence": float(np.min(arr)),
            "max_confidence": float(np.max(arr)),
            "std_confidence": float(np.std(arr)),
        }

    def compute_all(
        self,
        results: Sequence[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        計算所有證據覆蓋指標

        Args:
            results: 推理結果列表，每個包含:
                - input_phenotypes: 輸入表型
                - explained_phenotypes: 被解釋的表型
                - path_confidences: 路徑置信度列表

        Returns:
            覆蓋率指標字典
        """
        coverages = []
        all_confidences = []

        for result in results:
            if "input_phenotypes" in result and "explained_phenotypes" in result:
                cov = self.phenotype_coverage(
                    result["input_phenotypes"],
                    result["explained_phenotypes"],
                )
                coverages.append(cov)

            if "path_confidences" in result:
                all_confidences.extend(result["path_confidences"])

        metrics = {}

        if coverages:
            metrics["mean_phenotype_coverage"] = float(np.mean(coverages))
            metrics["min_phenotype_coverage"] = float(np.min(coverages))

        if all_confidences:
            strength = self.evidence_strength(all_confidences)
            metrics.update({f"evidence_{k}": v for k, v in strength.items()})

        return metrics


# =============================================================================
# Model Training Metrics
# =============================================================================
@dataclass
class TrainingMetrics:
    """
    訓練過程監控指標
    """

    def __init__(self):
        self.history: Dict[str, List[float]] = {}
        self._step = 0

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        記錄指標

        Args:
            metrics: 指標字典
            step: 步數（可選，自動遞增）
        """
        if step is not None:
            self._step = step
        else:
            self._step += 1

        for name, value in metrics.items():
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(value)

    def get_latest(self, metric_name: str) -> Optional[float]:
        """獲取最新的指標值"""
        if metric_name in self.history and self.history[metric_name]:
            return self.history[metric_name][-1]
        return None

    def get_best(self, metric_name: str, mode: str = "max") -> Optional[float]:
        """
        獲取最佳指標值

        Args:
            metric_name: 指標名稱
            mode: "max" 或 "min"
        """
        if metric_name not in self.history or not self.history[metric_name]:
            return None

        if mode == "max":
            return max(self.history[metric_name])
        else:
            return min(self.history[metric_name])

    def get_moving_average(
        self,
        metric_name: str,
        window: int = 10,
    ) -> Optional[float]:
        """獲取移動平均"""
        if metric_name not in self.history or not self.history[metric_name]:
            return None

        values = self.history[metric_name][-window:]
        return float(np.mean(values))

    def is_improving(
        self,
        metric_name: str,
        mode: str = "max",
        patience: int = 5,
        min_delta: float = 1e-4,
    ) -> bool:
        """
        檢查指標是否在改善

        Args:
            metric_name: 指標名稱
            mode: "max" 或 "min"
            patience: 容忍的不改善步數
            min_delta: 最小改善量

        Returns:
            是否在改善
        """
        if metric_name not in self.history:
            return True

        values = self.history[metric_name]
        if len(values) <= patience:
            return True

        recent = values[-patience:]
        best_recent = max(recent) if mode == "max" else min(recent)
        earlier = values[:-patience]
        best_earlier = max(earlier) if mode == "max" else min(earlier)

        if mode == "max":
            return best_recent > best_earlier + min_delta
        else:
            return best_recent < best_earlier - min_delta

    def summary(self) -> Dict[str, Dict[str, float]]:
        """獲取所有指標的摘要統計"""
        summary = {}
        for name, values in self.history.items():
            if values:
                arr = np.array(values)
                summary[name] = {
                    "latest": float(values[-1]),
                    "best": float(np.max(arr)),
                    "worst": float(np.min(arr)),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                }
        return summary


# =============================================================================
# Link Prediction Metrics (for KG embedding evaluation)
# =============================================================================
@dataclass
class LinkPredictionMetrics:
    """
    知識圖譜連結預測指標

    用於評估 GNN 嵌入質量
    """

    def mean_rank(
        self,
        ranks: Sequence[int],
    ) -> float:
        """
        計算平均排名

        Args:
            ranks: 正確連結的排名列表 (1-indexed)

        Returns:
            平均排名
        """
        if not ranks:
            return float("inf")
        return float(np.mean(ranks))

    def mean_reciprocal_rank(
        self,
        ranks: Sequence[int],
    ) -> float:
        """
        計算 MRR

        Args:
            ranks: 正確連結的排名列表 (1-indexed)

        Returns:
            MRR 分數
        """
        if not ranks:
            return 0.0
        return float(np.mean([1.0 / r for r in ranks]))

    def hits_at_k(
        self,
        ranks: Sequence[int],
        k: int,
    ) -> float:
        """
        計算 Hits@k

        Args:
            ranks: 正確連結的排名列表 (1-indexed)
            k: 閾值

        Returns:
            Hits@k 分數
        """
        if not ranks:
            return 0.0
        return float(np.mean([1.0 if r <= k else 0.0 for r in ranks]))

    def compute_from_scores(
        self,
        positive_scores: NDArray[np.float64],
        negative_scores: NDArray[np.float64],
        k_values: Tuple[int, ...] = (1, 3, 5, 10),
    ) -> Dict[str, float]:
        """
        從分數計算連結預測指標

        Args:
            positive_scores: (N,) 正確連結的分數
            negative_scores: (N, num_negatives) 負樣本分數
            k_values: Hits@k 的 k 值列表

        Returns:
            指標字典
        """
        # 計算排名
        # 排名 = 分數高於正樣本的負樣本數 + 1
        ranks = []
        for i, pos_score in enumerate(positive_scores):
            neg_scores = negative_scores[i]
            rank = int((neg_scores > pos_score).sum()) + 1
            ranks.append(rank)

        metrics = {
            "mean_rank": self.mean_rank(ranks),
            "mrr": self.mean_reciprocal_rank(ranks),
        }

        for k in k_values:
            metrics[f"hits@{k}"] = self.hits_at_k(ranks, k)

        return metrics


# =============================================================================
# Unified Metrics Calculator
# =============================================================================
class DiagnosisMetrics:
    """
    統一的診斷評估指標計算器

    整合排序指標、本體違反指標和證據覆蓋指標
    """

    def __init__(self):
        self.ranking = RankingMetrics()
        self.ontology = OntologyViolationMetrics()
        self.evidence = EvidenceCoverageMetrics()
        self.link_prediction = LinkPredictionMetrics()
        self.training = TrainingMetrics()

    def evaluate_predictions(
        self,
        predictions: Sequence[Dict[str, Any]],
        ground_truths: Sequence[str],
        ontology_ancestors: Optional[Dict[str, set]] = None,
        disease_gene_map: Optional[Dict[str, set]] = None,
        k_values: Tuple[int, ...] = (1, 3, 5, 10, 20),
    ) -> Dict[str, float]:
        """
        完整評估診斷預測

        Args:
            predictions: 預測結果列表，每個包含:
                - candidates: 排序的候選疾病 ID 列表
                - supporting_genes: 支持基因列表 (可選)
                - input_phenotypes: 輸入表型 (可選)
                - explained_phenotypes: 被解釋的表型 (可選)
                - path_confidences: 路徑置信度 (可選)
            ground_truths: 正確疾病 ID 列表
            ontology_ancestors: 本體祖先映射 (可選)
            disease_gene_map: 疾病-基因映射 (可選)
            k_values: Hits@k 的 k 值列表

        Returns:
            完整的評估指標字典
        """
        metrics = {}

        # 1. 排序指標
        pred_lists = [p["candidates"] for p in predictions if "candidates" in p]
        if pred_lists and ground_truths:
            ranking_metrics = self.ranking.compute_all(
                pred_lists, ground_truths, k_values
            )
            metrics.update(ranking_metrics)

        # 2. 本體違反指標
        violation_metrics = self.ontology.compute_violation_rate(
            predictions, ontology_ancestors, disease_gene_map
        )
        metrics.update(violation_metrics)

        # 3. 證據覆蓋指標
        evidence_metrics = self.evidence.compute_all(predictions)
        metrics.update(evidence_metrics)

        return metrics

    def format_report(
        self,
        metrics: Dict[str, float],
        title: str = "Evaluation Report",
    ) -> str:
        """
        格式化評估報告

        Args:
            metrics: 指標字典
            title: 報告標題

        Returns:
            格式化的報告字符串
        """
        lines = [
            f"\n{'='*60}",
            f"  {title}",
            f"{'='*60}",
        ]

        # 分組顯示指標
        groups = {
            "Ranking Metrics": ["mrr", "hits@", "ndcg@"],
            "Ontology Metrics": ["consistency", "violation"],
            "Evidence Metrics": ["coverage", "confidence", "evidence_"],
        }

        for group_name, prefixes in groups.items():
            group_metrics = {
                k: v for k, v in metrics.items()
                if any(p in k for p in prefixes)
            }
            if group_metrics:
                lines.append(f"\n  {group_name}:")
                lines.append(f"  {'-'*40}")
                for name, value in sorted(group_metrics.items()):
                    lines.append(f"    {name:<30}: {value:.4f}")

        # 其他指標
        shown = set()
        for prefixes in groups.values():
            for k in metrics.keys():
                if any(p in k for p in prefixes):
                    shown.add(k)

        other = {k: v for k, v in metrics.items() if k not in shown}
        if other:
            lines.append("\n  Other Metrics:")
            lines.append(f"  {'-'*40}")
            for name, value in sorted(other.items()):
                lines.append(f"    {name:<30}: {value:.4f}")

        lines.append(f"\n{'='*60}\n")

        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================
def compute_hits_at_k(
    predictions: Sequence[Sequence[str]],
    ground_truths: Sequence[str],
    k: int = 10,
) -> float:
    """便捷函數：計算 Hits@k"""
    return RankingMetrics().hits_at_k(predictions, ground_truths, k)


def compute_mrr(
    predictions: Sequence[Sequence[str]],
    ground_truths: Sequence[str],
) -> float:
    """便捷函數：計算 MRR"""
    return RankingMetrics().mean_reciprocal_rank(predictions, ground_truths)


def compute_ndcg(
    predictions: Sequence[Sequence[str]],
    ground_truths: Sequence[str],
    k: int = 10,
) -> float:
    """便捷函數：計算 NDCG@k"""
    return RankingMetrics().ndcg_at_k(predictions, ground_truths, k)


# =============================================================================
# Module Exports
# =============================================================================
__all__ = [
    "RankingMetrics",
    "OntologyViolationMetrics",
    "EvidenceCoverageMetrics",
    "TrainingMetrics",
    "LinkPredictionMetrics",
    "DiagnosisMetrics",
    "compute_hits_at_k",
    "compute_mrr",
    "compute_ndcg",
]
