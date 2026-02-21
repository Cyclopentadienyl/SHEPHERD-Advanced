"""
SHEPHERD-Advanced Loss Functions
================================
Multi-task loss functions for diagnosis model training.

Module: src/training/loss_functions.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/training/loss_functions.py

Purpose:
    Define loss functions for the multi-task learning objective:
    1. Disease diagnosis ranking (primary task)
    2. Knowledge graph link prediction (auxiliary task)
    3. Contrastive learning for discriminative embeddings
    4. Ortholog consistency for cross-species knowledge transfer (P1 feature)

Components:
    - LossConfig: Hyperparameters for loss weights and margins
    - DiagnosisLoss: Cross-entropy + margin ranking for disease prediction
    - LinkPredictionLoss: DistMult-based KG embedding loss
    - ContrastiveLoss: InfoNCE loss for phenotype-disease matching
    - OrthologConsistencyLoss: Embedding alignment for ortholog gene pairs
    - MultiTaskLoss: Weighted combination with uncertainty weighting option

Dependencies:
    - torch: Tensor operations
    - torch.nn: Module base class, ParameterDict
    - torch.nn.functional: Activation functions, cross_entropy, relu

Input (MultiTaskLoss.forward):
    - batch: Dict containing task-specific tensors:
        - diagnosis_scores: (B, num_diseases) prediction logits
        - diagnosis_targets: (B,) ground truth indices
        - positive_triples: (B, 3) KG positive edges
        - negative_triples: (B, num_neg, 3) KG negative edges
        - patient_embeddings: (B, dim) aggregated phenotype embeddings
        - disease_embeddings: (B, dim) disease node embeddings
        - ortholog_pairs: (num_pairs, 2) human-mouse gene index pairs
    - model_outputs: Dict containing:
        - node_embeddings: {node_type: Tensor}
        - relation_embeddings: (num_relations, dim)

Output:
    - Tuple[Tensor, Dict[str, float]]: (total_loss, {loss_name: value})

Called by:
    - src/training/trainer.py (training step)

Version: 1.0.0
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class LossConfig:
    """損失函數配置"""
    # 任務權重
    diagnosis_weight: float = 1.0
    link_prediction_weight: float = 0.5
    contrastive_weight: float = 0.3
    ortholog_weight: float = 0.2

    # 診斷損失設定
    margin: float = 1.0
    label_smoothing: float = 0.1

    # 對比學習設定
    temperature: float = 0.07
    use_hard_negatives: bool = True

    # 連結預測設定
    negative_sample_ratio: int = 5


# =============================================================================
# Diagnosis Loss
# =============================================================================
class DiagnosisLoss(nn.Module):
    """
    疾病診斷損失

    結合排序損失和分類損失，確保正確疾病排名靠前
    """

    def __init__(
        self,
        margin: float = 1.0,
        label_smoothing: float = 0.1,
        reduction: str = "mean",
    ):
        """
        Args:
            margin: 排序損失的 margin
            label_smoothing: 標籤平滑係數
            reduction: 損失歸約方式 ("mean", "sum", "none")
        """
        super().__init__()
        self.margin = margin
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(
        self,
        scores: Tensor,
        targets: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        計算診斷損失

        Args:
            scores: (batch_size, num_diseases) 預測分數
            targets: (batch_size,) 正確疾病索引
            mask: (batch_size, num_diseases) 可選的遮罩

        Returns:
            損失值
        """
        batch_size, num_diseases = scores.shape

        # 1. 分類損失 (Cross Entropy with label smoothing)
        if self.label_smoothing > 0:
            ce_loss = self._label_smoothed_ce(scores, targets, num_diseases)
        else:
            ce_loss = F.cross_entropy(scores, targets, reduction=self.reduction)

        # 2. 排序損失 (Margin Ranking Loss)
        # 確保正確答案的分數比其他候選高出 margin
        positive_scores = scores.gather(1, targets.unsqueeze(1))  # (batch, 1)

        # 創建負樣本遮罩
        neg_mask = torch.ones_like(scores, dtype=torch.bool)
        neg_mask.scatter_(1, targets.unsqueeze(1), False)

        # 對每個負樣本計算 margin loss
        negative_scores = scores.masked_select(neg_mask).view(batch_size, -1)

        # max(0, margin - (positive - negative))
        margin_loss = F.relu(
            self.margin - positive_scores + negative_scores
        )

        if mask is not None:
            neg_mask_flat = mask.masked_select(neg_mask.unsqueeze(-1) if mask.dim() == 3 else neg_mask)
            margin_loss = margin_loss * neg_mask_flat.view(batch_size, -1)

        if self.reduction == "mean":
            margin_loss = margin_loss.mean()
        elif self.reduction == "sum":
            margin_loss = margin_loss.sum()

        # 組合損失
        total_loss = ce_loss + 0.5 * margin_loss

        return total_loss

    def _label_smoothed_ce(
        self,
        scores: Tensor,
        targets: Tensor,
        num_classes: int,
    ) -> Tensor:
        """帶標籤平滑的交叉熵"""
        log_probs = F.log_softmax(scores, dim=-1)

        # 創建平滑標籤
        smooth_targets = torch.full_like(
            log_probs, self.label_smoothing / (num_classes - 1)
        )
        smooth_targets.scatter_(
            1, targets.unsqueeze(1), 1.0 - self.label_smoothing
        )

        loss = -torch.sum(smooth_targets * log_probs, dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# =============================================================================
# Link Prediction Loss
# =============================================================================
class LinkPredictionLoss(nn.Module):
    """
    知識圖譜連結預測損失

    用於訓練 GNN 學習有意義的節點嵌入
    """

    def __init__(
        self,
        margin: float = 1.0,
        negative_sample_ratio: int = 5,
        reduction: str = "mean",
    ):
        """
        Args:
            margin: 排序損失的 margin
            negative_sample_ratio: 負樣本比例
            reduction: 損失歸約方式
        """
        super().__init__()
        self.margin = margin
        self.negative_sample_ratio = negative_sample_ratio
        self.reduction = reduction

    def forward(
        self,
        head_embeddings: Tensor,
        tail_embeddings: Tensor,
        relation_embeddings: Tensor,
        positive_samples: Tensor,
        negative_samples: Tensor,
    ) -> Tensor:
        """
        計算連結預測損失 (DistMult scoring)

        Args:
            head_embeddings: (num_nodes, dim) 頭節點嵌入
            tail_embeddings: (num_nodes, dim) 尾節點嵌入
            relation_embeddings: (num_relations, dim) 關係嵌入
            positive_samples: (batch, 3) 正樣本 [head, relation, tail]
            negative_samples: (batch, num_neg, 3) 負樣本

        Returns:
            損失值
        """
        # 正樣本分數
        pos_heads = head_embeddings[positive_samples[:, 0]]
        pos_rels = relation_embeddings[positive_samples[:, 1]]
        pos_tails = tail_embeddings[positive_samples[:, 2]]
        pos_scores = self._score(pos_heads, pos_rels, pos_tails)

        # 負樣本分數
        batch_size, num_neg = negative_samples.shape[:2]
        neg_flat = negative_samples.view(-1, 3)
        neg_heads = head_embeddings[neg_flat[:, 0]]
        neg_rels = relation_embeddings[neg_flat[:, 1]]
        neg_tails = tail_embeddings[neg_flat[:, 2]]
        neg_scores = self._score(neg_heads, neg_rels, neg_tails)
        neg_scores = neg_scores.view(batch_size, num_neg)

        # Margin ranking loss
        # 正樣本分數應該比負樣本高出 margin
        loss = F.relu(self.margin - pos_scores.unsqueeze(1) + neg_scores)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def _score(
        self,
        heads: Tensor,
        relations: Tensor,
        tails: Tensor,
    ) -> Tensor:
        """DistMult 評分函數: score = <h, r, t>"""
        return (heads * relations * tails).sum(dim=-1)


# =============================================================================
# Contrastive Loss
# =============================================================================
class ContrastiveLoss(nn.Module):
    """
    對比學習損失

    用於學習判別性的表型-疾病嵌入
    """

    def __init__(
        self,
        temperature: float = 0.07,
        use_hard_negatives: bool = True,
        reduction: str = "mean",
    ):
        """
        Args:
            temperature: softmax 溫度參數
            use_hard_negatives: 是否使用困難負樣本
            reduction: 損失歸約方式
        """
        super().__init__()
        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives
        self.reduction = reduction

    def forward(
        self,
        anchor_embeddings: Tensor,
        positive_embeddings: Tensor,
        negative_embeddings: Optional[Tensor] = None,
    ) -> Tensor:
        """
        計算對比損失 (InfoNCE)

        Args:
            anchor_embeddings: (batch, dim) 錨點嵌入 (患者表型)
            positive_embeddings: (batch, dim) 正樣本嵌入 (正確疾病)
            negative_embeddings: (batch, num_neg, dim) 可選的困難負樣本

        Returns:
            損失值
        """
        batch_size = anchor_embeddings.shape[0]

        # 正規化嵌入
        anchor_norm = F.normalize(anchor_embeddings, p=2, dim=-1)
        positive_norm = F.normalize(positive_embeddings, p=2, dim=-1)

        # 正樣本相似度
        pos_sim = torch.sum(anchor_norm * positive_norm, dim=-1) / self.temperature

        if negative_embeddings is not None and self.use_hard_negatives:
            # 使用提供的困難負樣本
            negative_norm = F.normalize(negative_embeddings, p=2, dim=-1)
            neg_sim = torch.bmm(
                anchor_norm.unsqueeze(1),
                negative_norm.transpose(1, 2),
            ).squeeze(1) / self.temperature  # (batch, num_neg)

            # InfoNCE loss
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
            labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        else:
            # 批次內負樣本 (in-batch negatives)
            sim_matrix = torch.mm(anchor_norm, positive_norm.t()) / self.temperature
            labels = torch.arange(batch_size, device=sim_matrix.device)
            loss = F.cross_entropy(sim_matrix, labels, reduction=self.reduction)

        return loss


# =============================================================================
# Ortholog Consistency Loss
# =============================================================================
class OrthologConsistencyLoss(nn.Module):
    """
    同源基因一致性損失

    確保同源基因對有相似的嵌入
    """

    def __init__(
        self,
        margin: float = 0.5,
        reduction: str = "mean",
    ):
        """
        Args:
            margin: 一致性 margin
            reduction: 損失歸約方式
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        human_gene_embeddings: Tensor,
        ortholog_embeddings: Tensor,
        ortholog_pairs: Tensor,
        confidence_scores: Optional[Tensor] = None,
    ) -> Tensor:
        """
        計算同源基因一致性損失

        Args:
            human_gene_embeddings: (num_human_genes, dim)
            ortholog_embeddings: (num_ortholog_genes, dim)
            ortholog_pairs: (num_pairs, 2) [human_idx, ortholog_idx]
            confidence_scores: (num_pairs,) 可選的置信度權重

        Returns:
            損失值
        """
        if ortholog_pairs.shape[0] == 0:
            return torch.tensor(0.0, device=human_gene_embeddings.device, requires_grad=True)

        # 獲取同源對的嵌入
        human_emb = human_gene_embeddings[ortholog_pairs[:, 0]]
        ortholog_emb = ortholog_embeddings[ortholog_pairs[:, 1]]

        # 正規化
        human_norm = F.normalize(human_emb, p=2, dim=-1)
        ortholog_norm = F.normalize(ortholog_emb, p=2, dim=-1)

        # 餘弦相似度
        similarity = (human_norm * ortholog_norm).sum(dim=-1)

        # 損失: 希望相似度接近 1
        loss = F.relu(self.margin - similarity)

        if confidence_scores is not None:
            loss = loss * confidence_scores

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# =============================================================================
# Multi-Task Loss
# =============================================================================
class MultiTaskLoss(nn.Module):
    """
    多任務損失組合

    整合診斷、連結預測、對比學習和同源基因一致性損失
    """

    def __init__(self, config: Optional[LossConfig] = None):
        """
        Args:
            config: 損失函數配置
        """
        super().__init__()
        self.config = config or LossConfig()

        # 初始化各個損失函數
        self.diagnosis_loss = DiagnosisLoss(
            margin=self.config.margin,
            label_smoothing=self.config.label_smoothing,
        )

        self.link_prediction_loss = LinkPredictionLoss(
            margin=self.config.margin,
            negative_sample_ratio=self.config.negative_sample_ratio,
        )

        self.contrastive_loss = ContrastiveLoss(
            temperature=self.config.temperature,
            use_hard_negatives=self.config.use_hard_negatives,
        )

        self.ortholog_loss = OrthologConsistencyLoss(margin=0.5)

        # 可學習的任務權重 (uncertainty weighting)
        self.log_vars = nn.ParameterDict({
            "diagnosis": nn.Parameter(torch.zeros(1)),
            "link_prediction": nn.Parameter(torch.zeros(1)),
            "contrastive": nn.Parameter(torch.zeros(1)),
            "ortholog": nn.Parameter(torch.zeros(1)),
        })

    def forward(
        self,
        batch: Dict[str, Any],
        model_outputs: Dict[str, Tensor],
        use_uncertainty_weighting: bool = False,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        計算多任務損失

        Args:
            batch: 原始批次數據，包含:
                - positive_triples: (batch, 3) - 用於連結預測
                - negative_triples: (batch, num_neg, 3) - 負樣本
                - ortholog_pairs: (num_pairs, 2) - 同源基因對
                - ortholog_confidences: (num_pairs,) - 同源置信度
            model_outputs: 模型計算輸出，包含:
                - node_embeddings: {node_type: (num_nodes, dim)}
                - relation_embeddings: (num_relations, dim)
                - diagnosis_scores: (batch, num_diseases) - 診斷分數
                - diagnosis_targets: (batch,) - 診斷目標
                - patient_embeddings: (batch, dim) - 患者嵌入
                - disease_embeddings: (batch, dim) - 疾病嵌入
            use_uncertainty_weighting: 是否使用不確定性加權

        Returns:
            (total_loss, loss_dict) 總損失和各項損失
        """
        loss_dict = {}

        # Get device from model_outputs (handle nested dict for node_embeddings)
        device = None
        for v in model_outputs.values():
            if isinstance(v, torch.Tensor):
                device = v.device
                break
            elif isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, torch.Tensor):
                        device = vv.device
                        break
                if device is not None:
                    break
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # 1. 診斷損失
        # diagnosis_scores 和 diagnosis_targets 由 model 計算，存放在 model_outputs 中
        if "diagnosis_scores" in model_outputs and "diagnosis_targets" in model_outputs:
            diag_loss = self.diagnosis_loss(
                model_outputs["diagnosis_scores"],
                model_outputs["diagnosis_targets"],
            )
            loss_dict["diagnosis"] = diag_loss.item()

            if use_uncertainty_weighting:
                precision = torch.exp(-self.log_vars["diagnosis"])
                weighted_loss = precision * diag_loss + self.log_vars["diagnosis"]
            else:
                weighted_loss = self.config.diagnosis_weight * diag_loss
            total_loss = total_loss + weighted_loss

        # 2. 連結預測損失
        if "positive_triples" in batch and "negative_triples" in batch:
            node_emb = model_outputs.get("node_embeddings", {})
            if "gene" in node_emb and "disease" in node_emb:
                lp_loss = self.link_prediction_loss(
                    head_embeddings=node_emb["gene"],
                    tail_embeddings=node_emb["disease"],
                    relation_embeddings=model_outputs.get(
                        "relation_embeddings",
                        torch.zeros(10, node_emb["gene"].shape[-1], device=total_loss.device),
                    ),
                    positive_samples=batch["positive_triples"],
                    negative_samples=batch["negative_triples"],
                )
                loss_dict["link_prediction"] = lp_loss.item()

                if use_uncertainty_weighting:
                    precision = torch.exp(-self.log_vars["link_prediction"])
                    weighted_loss = precision * lp_loss + self.log_vars["link_prediction"]
                else:
                    weighted_loss = self.config.link_prediction_weight * lp_loss
                total_loss = total_loss + weighted_loss

        # 3. 對比學習損失
        # patient_embeddings 和 disease_embeddings 由 model 計算，存放在 model_outputs 中
        if "patient_embeddings" in model_outputs and "disease_embeddings" in model_outputs:
            contrastive_loss = self.contrastive_loss(
                anchor_embeddings=model_outputs["patient_embeddings"],
                positive_embeddings=model_outputs["disease_embeddings"],
                negative_embeddings=model_outputs.get("negative_disease_embeddings"),
            )
            loss_dict["contrastive"] = contrastive_loss.item()

            if use_uncertainty_weighting:
                precision = torch.exp(-self.log_vars["contrastive"])
                weighted_loss = precision * contrastive_loss + self.log_vars["contrastive"]
            else:
                weighted_loss = self.config.contrastive_weight * contrastive_loss
            total_loss = total_loss + weighted_loss

        # 4. 同源基因一致性損失
        if "ortholog_pairs" in batch:
            node_emb = model_outputs.get("node_embeddings", {})
            if "gene" in node_emb and "mouse_gene" in node_emb:
                ortholog_loss = self.ortholog_loss(
                    human_gene_embeddings=node_emb["gene"],
                    ortholog_embeddings=node_emb["mouse_gene"],
                    ortholog_pairs=batch["ortholog_pairs"],
                    confidence_scores=batch.get("ortholog_confidences"),
                )
                loss_dict["ortholog"] = ortholog_loss.item()

                if use_uncertainty_weighting:
                    precision = torch.exp(-self.log_vars["ortholog"])
                    weighted_loss = precision * ortholog_loss + self.log_vars["ortholog"]
                else:
                    weighted_loss = self.config.ortholog_weight * ortholog_loss
                total_loss = total_loss + weighted_loss

        loss_dict["total"] = total_loss.item()

        return total_loss, loss_dict


# =============================================================================
# Module Exports
# =============================================================================
__all__ = [
    "LossConfig",
    "DiagnosisLoss",
    "LinkPredictionLoss",
    "ContrastiveLoss",
    "OrthologConsistencyLoss",
    "MultiTaskLoss",
]
