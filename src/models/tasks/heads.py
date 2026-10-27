"""
The patient encoder and the three task heads of a finetuned bundle.
==================================================================
`docs/working/scorer-retraining/README.md` A6–A9, C5, C9, U3, U10, U11, U13.

**Patient encoder (A6, U11).** The patient's phenotype vectors, padded to the
batch maximum with a mask (never truncated), pass a Transformer encoder with
torch's `TransformerEncoderLayer` defaults (d_ff 2048, dropout 0.1, ReLU,
post-norm, no final norm; `up:task_heads/patient_nca.py:51-52`), then attention
pooling `softmax(aᵀ W x_p)` over the real positions with one learned vector `a`
(`:29-33,64-66`). The source's bilinear bias is dropped: it cancels in the
softmax (U11). A padded position gets zero weight.

**Towers (C5, U13).** One two-layer projection per entity, `Linear → leaky
ReLU(0.1) → Linear`, as separate parameter sets (`:40-46,69-70`;
`up:task_heads/gp_aligner.py:46-54,81-86`).

**Scores.**
- disease and patients-like-me: `−‖ẑ_a − ẑ_b‖²` (U10, `src.models.tasks.losses`);
- gene EMBSIM: `tanh(ẑ_Pᵀ W ẑ_g + b)` on L2-normalised vectors (C9,
  `gp_aligner.py:95-101`). The SP mix and Eq 10 augmentation are the gene task's
  (README §6, F10), outside the module: they have no trainable parameters.

Module: src/models/tasks/heads.py
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.bundle.spec import PatientEncoderSpec, SpecError, TaskHeadSpec
from src.models.tasks.losses import normalised_sq_distance


class PatientEncoder(nn.Module):
    def __init__(self, spec: PatientEncoderSpec, width: int):
        super().__init__()
        self.spec = spec
        self.transformer: Optional[nn.TransformerEncoder] = None
        if spec.transformer_layers > 0:
            if width % spec.transformer_heads:
                raise SpecError(f"the patient Transformer's width {width} is not divisible by "
                                f"its {spec.transformer_heads} heads")
            layer = nn.TransformerEncoderLayer(
                d_model=width, nhead=spec.transformer_heads, dim_feedforward=spec.feedforward,
                dropout=spec.dropout, activation="relu", batch_first=True, norm_first=False)
            self.transformer = nn.TransformerEncoder(layer, num_layers=spec.transformer_layers,
                                                     enable_nested_tensor=False)
        # allennlp's BilinearAttention initialisation (xavier on W) and the
        # source's xavier query vector.
        self.query = nn.Parameter(torch.empty(1, width))
        self.pool_weight = nn.Parameter(torch.empty(width, width))
        nn.init.xavier_uniform_(self.query)
        nn.init.xavier_uniform_(self.pool_weight)

    def forward(self, phenotypes: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """`[B, P, W]` vectors and a `[B, P]` mask (True = a real phenotype) →
        the pooled `[B, W]` patient vector and the `[B, P]` pooling weights."""
        if bool((~mask.any(dim=1)).any()):
            raise ValueError("a patient without a phenotype has nothing to encode; refused at "
                             "data binding")
        x = phenotypes
        if self.transformer is not None:
            x = self.transformer(x, src_key_padding_mask=~mask)
        logits = (self.query @ self.pool_weight).unsqueeze(1) @ x.transpose(1, 2)
        logits = logits.squeeze(1).masked_fill(~mask, float("-inf"))
        weights = torch.softmax(logits, dim=1)
        return (weights.unsqueeze(1) @ x).squeeze(1), weights


class ProjectionTower(nn.Module):
    def __init__(self, width: int, slope: float):
        super().__init__()
        self.first = nn.Linear(width, width)
        self.second = nn.Linear(width, width)
        self.slope = slope

    def forward(self, x: Tensor) -> Tensor:
        return self.second(F.leaky_relu(self.first(x), negative_slope=self.slope))


class TaskHead(nn.Module):
    """The patient tower plus the task's candidate tower and scoring rule."""

    def __init__(self, spec: TaskHeadSpec, width: int):
        super().__init__()
        self.spec = spec
        self.patient_tower = ProjectionTower(width, spec.projection_slope)

    @property
    def task(self) -> str:
        return self.spec.task

    def project_candidates(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def scores(self, patients: Tensor, candidates: Tensor) -> Tensor:
        """`[B, W]` projected patients × `[M, W]` projected candidates → `[B, M]`."""
        raise NotImplementedError


class DiseaseHead(TaskHead):
    def __init__(self, spec: TaskHeadSpec, width: int):
        super().__init__(spec, width)
        self.disease_tower = ProjectionTower(width, spec.projection_slope)

    def project_candidates(self, x: Tensor) -> Tensor:
        return self.disease_tower(x)

    def scores(self, patients: Tensor, candidates: Tensor) -> Tensor:
        return -normalised_sq_distance(patients, candidates, self.spec.distance_normalisation_eps)


class PatientsLikeMeHead(TaskHead):
    """Candidates are other patients, through the same patient tower (A9)."""

    def project_candidates(self, x: Tensor) -> Tensor:
        return self.patient_tower(x)

    def scores(self, patients: Tensor, candidates: Tensor) -> Tensor:
        return -normalised_sq_distance(patients, candidates, self.spec.distance_normalisation_eps)


class GeneHead(TaskHead):
    def __init__(self, spec: TaskHeadSpec, width: int):
        super().__init__(spec, width)
        self.gene_tower = ProjectionTower(width, spec.projection_slope)
        self.decoder_weight = nn.Parameter(torch.empty(width, width))
        self.decoder_bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.decoder_weight)

    def project_candidates(self, x: Tensor) -> Tensor:
        return self.gene_tower(x)

    def scores(self, patients: Tensor, candidates: Tensor) -> Tensor:
        """EMBSIM; the gene task mixes it with SPLSIM (README §6)."""
        p = F.normalize(patients, p=2, dim=-1)
        g = F.normalize(candidates, p=2, dim=-1)
        return torch.tanh(p @ self.decoder_weight @ g.T + self.decoder_bias)

    def list_scores(self, patients: Tensor, candidates: Tensor) -> Tensor:
        """EMBSIM of each patient's own list: `[B, W]` × `[B, M, W]` → `[B, M]`."""
        p = F.normalize(patients, p=2, dim=-1)
        g = F.normalize(candidates, p=2, dim=-1)
        return torch.tanh(torch.einsum("bw,wv,bmv->bm", p, self.decoder_weight, g)
                          + self.decoder_bias)


_HEADS = {"disease": DiseaseHead, "gene": GeneHead, "patients_like_me": PatientsLikeMeHead}


def build_task_head(spec: TaskHeadSpec, width: int) -> TaskHead:
    return _HEADS[spec.task](spec, width)


__all__ = ["DiseaseHead", "GeneHead", "PatientEncoder", "PatientsLikeMeHead", "ProjectionTower",
           "TaskHead", "build_task_head"]
