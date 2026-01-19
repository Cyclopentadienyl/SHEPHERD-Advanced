"""
# ==============================================================================
# Module: src/models/decoders/heads.py
# ==============================================================================
# Purpose: Prediction heads for various downstream tasks
#
# Dependencies:
#   - External: torch (>=2.9)
#   - Internal: None
#
# Exports:
#   - DiagnosisHead: Disease ranking for patient phenotypes
#   - LinkPredictionHead: Missing edge prediction
#   - NodeClassificationHead: Node property prediction
#   - ExplanationHead: Generate interpretable explanations
#
# Design Notes:
#   - All heads are torch.compile() compatible
#   - Support both training and inference modes
#   - Include uncertainty estimation where applicable
# ==============================================================================
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DiagnosisHead(nn.Module):
    """
    Prediction head for rare disease diagnosis.

    Takes patient phenotype embeddings and disease embeddings,
    outputs ranked disease predictions with confidence scores.

    Features:
    - Multiple similarity computation methods
    - Uncertainty estimation via Monte Carlo dropout
    - Evidence aggregation from multiple reasoning paths
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: Optional[int] = None,
        dropout: float = 0.1,
        similarity_type: str = "learned",  # "learned", "cosine", "euclidean"
        use_uncertainty: bool = True,
    ):
        """
        Args:
            hidden_dim: Input embedding dimension
            num_classes: Optional fixed number of disease classes
            dropout: Dropout for uncertainty estimation
            similarity_type: How to compute phenotype-disease similarity
            use_uncertainty: Whether to compute uncertainty estimates
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.similarity_type = similarity_type
        self.use_uncertainty = use_uncertainty

        # Phenotype aggregation (patient profile encoder)
        self.phenotype_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Disease representation refinement
        self.disease_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Similarity computation
        if similarity_type == "learned":
            self.similarity_net = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),  # concat + element-wise product
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        # Optional: fixed classification layer
        if num_classes is not None:
            self.classifier = nn.Linear(hidden_dim, num_classes)

        # For attention-based phenotype aggregation
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        phenotype_embeddings: Tensor,
        disease_embeddings: Tensor,
        phenotype_mask: Optional[Tensor] = None,
        return_patient_profile: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Compute disease ranking scores.

        Args:
            phenotype_embeddings: (batch, num_phenotypes, hidden_dim) or (num_phenotypes, hidden_dim)
            disease_embeddings: (num_diseases, hidden_dim)
            phenotype_mask: Optional (batch, num_phenotypes) padding mask
            return_patient_profile: Whether to also return patient profile

        Returns:
            scores: (batch, num_diseases) or (num_diseases,) disease scores
            patient_profile: Optional (batch, hidden_dim) patient representation
        """
        # Handle unbatched input
        unbatched = phenotype_embeddings.dim() == 2
        if unbatched:
            phenotype_embeddings = phenotype_embeddings.unsqueeze(0)
            if phenotype_mask is not None:
                phenotype_mask = phenotype_mask.unsqueeze(0)

        batch_size = phenotype_embeddings.size(0)
        num_diseases = disease_embeddings.size(0)

        # 1. Aggregate phenotypes to patient profile
        patient_profile = self._aggregate_phenotypes(
            phenotype_embeddings, phenotype_mask
        )  # (batch, hidden_dim)

        # 2. Encode patient profile
        patient_profile = self.phenotype_encoder(patient_profile)

        # 3. Encode disease embeddings
        disease_repr = self.disease_encoder(disease_embeddings)  # (num_diseases, hidden_dim)

        # 4. Compute similarity scores
        scores = self._compute_scores(patient_profile, disease_repr)

        if unbatched:
            scores = scores.squeeze(0)
            patient_profile = patient_profile.squeeze(0)

        if return_patient_profile:
            return scores, patient_profile
        return scores

    def _aggregate_phenotypes(
        self,
        phenotype_embeddings: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        """
        Aggregate phenotype embeddings to patient profile using attention.

        Args:
            phenotype_embeddings: (batch, num_phenotypes, hidden_dim)
            mask: Optional (batch, num_phenotypes) mask

        Returns:
            (batch, hidden_dim) aggregated patient profile
        """
        # Compute attention weights
        attn_scores = self.attention(phenotype_embeddings).squeeze(-1)  # (batch, num_phenotypes)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, num_phenotypes)

        # Weighted sum
        patient_profile = torch.bmm(
            attn_weights.unsqueeze(1),
            phenotype_embeddings
        ).squeeze(1)  # (batch, hidden_dim)

        return patient_profile

    def _compute_scores(
        self,
        patient_profile: Tensor,
        disease_repr: Tensor,
    ) -> Tensor:
        """
        Compute similarity scores between patient and diseases.

        Args:
            patient_profile: (batch, hidden_dim)
            disease_repr: (num_diseases, hidden_dim)

        Returns:
            (batch, num_diseases) scores
        """
        batch_size = patient_profile.size(0)
        num_diseases = disease_repr.size(0)

        if self.similarity_type == "cosine":
            p_norm = F.normalize(patient_profile, p=2, dim=-1)
            d_norm = F.normalize(disease_repr, p=2, dim=-1)
            return torch.mm(p_norm, d_norm.t())

        elif self.similarity_type == "euclidean":
            # Negative distance as score
            diff = patient_profile.unsqueeze(1) - disease_repr.unsqueeze(0)
            return -torch.norm(diff, p=2, dim=-1)

        else:  # learned
            # Expand for pairwise computation
            p_exp = patient_profile.unsqueeze(1).expand(-1, num_diseases, -1)
            d_exp = disease_repr.unsqueeze(0).expand(batch_size, -1, -1)

            # Element-wise product
            prod = p_exp * d_exp

            # Concatenate features
            combined = torch.cat([p_exp, d_exp, prod], dim=-1)

            return self.similarity_net(combined).squeeze(-1)

    def predict_with_uncertainty(
        self,
        phenotype_embeddings: Tensor,
        disease_embeddings: Tensor,
        phenotype_mask: Optional[Tensor] = None,
        num_samples: int = 10,
    ) -> Tuple[Tensor, Tensor]:
        """
        Predict with Monte Carlo dropout for uncertainty estimation.

        Args:
            phenotype_embeddings: (batch, num_phenotypes, hidden_dim)
            disease_embeddings: (num_diseases, hidden_dim)
            phenotype_mask: Optional mask
            num_samples: Number of MC samples

        Returns:
            mean_scores: (batch, num_diseases) mean predictions
            uncertainty: (batch, num_diseases) prediction uncertainty
        """
        self.train()  # Enable dropout

        samples = []
        for _ in range(num_samples):
            with torch.no_grad():
                scores = self.forward(
                    phenotype_embeddings, disease_embeddings, phenotype_mask
                )
                samples.append(scores)

        samples = torch.stack(samples, dim=0)
        mean_scores = samples.mean(dim=0)
        uncertainty = samples.std(dim=0)

        self.eval()
        return mean_scores, uncertainty


class LinkPredictionHead(nn.Module):
    """
    Head for link prediction in the knowledge graph.

    Used for:
    - Predicting missing gene-disease associations
    - Predicting phenotype-disease annotations
    - Knowledge graph completion
    """

    def __init__(
        self,
        hidden_dim: int,
        num_edge_types: int = 1,
        dropout: float = 0.1,
        decoder_type: str = "distmult",  # "distmult", "transe", "mlp"
    ):
        """
        Args:
            hidden_dim: Embedding dimension
            num_edge_types: Number of distinct edge/relation types
            dropout: Dropout rate
            decoder_type: Type of link decoder
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.decoder_type = decoder_type

        if decoder_type == "distmult":
            # Relation-specific diagonal matrices
            self.relation_embeddings = nn.Embedding(num_edge_types, hidden_dim)

        elif decoder_type == "transe":
            # Relation-specific translation vectors
            self.relation_embeddings = nn.Embedding(num_edge_types, hidden_dim)

        elif decoder_type == "mlp":
            # MLP decoder
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src_embeddings: Tensor,
        dst_embeddings: Tensor,
        edge_type: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute link prediction scores.

        Args:
            src_embeddings: (num_edges, hidden_dim) source node embeddings
            dst_embeddings: (num_edges, hidden_dim) destination node embeddings
            edge_type: Optional (num_edges,) edge type indices

        Returns:
            (num_edges,) link prediction scores
        """
        if self.decoder_type == "distmult":
            if edge_type is not None:
                relation = self.relation_embeddings(edge_type)
            else:
                relation = torch.ones_like(src_embeddings)
            scores = (src_embeddings * relation * dst_embeddings).sum(dim=-1)

        elif self.decoder_type == "transe":
            if edge_type is not None:
                relation = self.relation_embeddings(edge_type)
            else:
                relation = torch.zeros_like(src_embeddings)
            # Score = -||h + r - t||
            scores = -torch.norm(src_embeddings + relation - dst_embeddings, p=2, dim=-1)

        else:  # mlp
            combined = torch.cat([src_embeddings, dst_embeddings], dim=-1)
            scores = self.decoder(combined).squeeze(-1)

        return scores

    def compute_loss(
        self,
        pos_scores: Tensor,
        neg_scores: Tensor,
        margin: float = 1.0,
    ) -> Tensor:
        """
        Compute margin-based ranking loss.

        Args:
            pos_scores: (num_pos,) positive edge scores
            neg_scores: (num_neg,) negative edge scores
            margin: Margin for ranking loss

        Returns:
            Scalar loss
        """
        # Margin ranking loss
        pos_scores = pos_scores.unsqueeze(1)  # (num_pos, 1)
        neg_scores = neg_scores.unsqueeze(0)  # (1, num_neg)

        loss = F.relu(margin - pos_scores + neg_scores).mean()
        return loss


class NodeClassificationHead(nn.Module):
    """
    Head for node classification tasks.

    Used for:
    - Predicting node properties (e.g., gene function)
    - Multi-label classification
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        multi_label: bool = False,
    ):
        """
        Args:
            hidden_dim: Input embedding dimension
            num_classes: Number of output classes
            num_layers: Number of MLP layers
            dropout: Dropout rate
            multi_label: Whether this is multi-label classification
        """
        super().__init__()
        self.multi_label = multi_label

        layers = []
        in_dim = hidden_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, node_embeddings: Tensor) -> Tensor:
        """
        Compute class logits.

        Args:
            node_embeddings: (num_nodes, hidden_dim)

        Returns:
            (num_nodes, num_classes) logits
        """
        return self.classifier(node_embeddings)

    def compute_loss(
        self,
        logits: Tensor,
        targets: Tensor,
        class_weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute classification loss.

        Args:
            logits: (num_nodes, num_classes) predictions
            targets: (num_nodes,) or (num_nodes, num_classes) labels
            class_weights: Optional class weights for imbalanced data

        Returns:
            Scalar loss
        """
        if self.multi_label:
            return F.binary_cross_entropy_with_logits(
                logits, targets.float(),
                pos_weight=class_weights,
            )
        else:
            return F.cross_entropy(logits, targets, weight=class_weights)


class ExplanationHead(nn.Module):
    """
    Generates explanations for disease predictions.

    Produces:
    - Phenotype importance scores
    - Key genes contributing to prediction
    - Reasoning path highlights
    """

    def __init__(
        self,
        hidden_dim: int,
        max_phenotypes: int = 50,
        max_genes: int = 20,
    ):
        """
        Args:
            hidden_dim: Embedding dimension
            max_phenotypes: Maximum phenotypes to explain
            max_genes: Maximum genes to highlight
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Phenotype importance scorer
        self.phenotype_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Gene relevance scorer
        self.gene_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        patient_profile: Tensor,
        disease_embedding: Tensor,
        phenotype_embeddings: Tensor,
        gene_embeddings: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Generate explanation scores.

        Args:
            patient_profile: (hidden_dim,) patient representation
            disease_embedding: (hidden_dim,) predicted disease
            phenotype_embeddings: (num_phenotypes, hidden_dim)
            gene_embeddings: Optional (num_genes, hidden_dim)

        Returns:
            Dictionary with explanation scores
        """
        explanations = {}

        # Phenotype importance
        disease_exp = disease_embedding.unsqueeze(0).expand(
            phenotype_embeddings.size(0), -1
        )
        pheno_combined = torch.cat([phenotype_embeddings, disease_exp], dim=-1)
        explanations["phenotype_importance"] = self.phenotype_scorer(pheno_combined).squeeze(-1)

        # Gene relevance
        if gene_embeddings is not None:
            disease_exp = disease_embedding.unsqueeze(0).expand(
                gene_embeddings.size(0), -1
            )
            gene_combined = torch.cat([gene_embeddings, disease_exp], dim=-1)
            explanations["gene_relevance"] = self.gene_scorer(gene_combined).squeeze(-1)

        return explanations
