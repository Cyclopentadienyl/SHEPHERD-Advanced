"""
SHEPHERD-Advanced Diagnosis Samples
===================================
The sample a samples file's row is read into.

Module: src/kg/data_loader.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/kg/data_loader.py

Purpose:
    `DiagnosisSample`, one patient's row of a samples file: the patient, the
    phenotypes as graph indices, the true disease and, for the gene and
    patients-like-me tasks, the gene fields. Every samples file is parsed into
    it by the one samples parser (`src.kg.storage.file_storage.read_samples_file`),
    which checks its ids; each task binds it to its graph view
    (`src.models.tasks`).

    The training-time loader stack that lived here (the subgraph sampler, the
    negative sampler, the diagnosis dataset and loader, and their
    configuration) retired with the ShepherdGNN training path (K8-7): a model
    bundle trains through its stage's own sampler (`src.training.lp_task`,
    `src.training.finetune_task`), in the training process, with no loader
    workers.

Version: 1.0.0
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DiagnosisSample:
    """單個診斷樣本"""
    patient_id: str
    phenotype_ids: List[int]  # 在圖中的索引
    disease_id: int  # 正確疾病在圖中的索引
    #: A gene-task row's candidate genes (README §8, PLAN_TEST_RESULTS §8.3). A
    #: generated disease cohort writes the disease's genes here; the disease
    #: task reads none of them.
    gene_ids: Optional[List[int]] = None
    #: The causal genes, all of them positives of the gene task (README F9); a
    #: patients-like-me pair is positive when two patients' sets intersect.
    causal_gene_ids: Optional[List[int]] = None
    #: The declared causal genes that name no single gene of the graph (not
    #: in it, or a symbol several gene nodes carry), as the importer
    #: normalised them (stripped, upper-case), distinct, in source order;
    #: written only when there are some. With `causal_gene_ids` they are the
    #: case's declared truth: each stays a positive it cannot rank (E5, F9),
    #: and two patients share one when they declared the same identifier.
    causal_genes_unmapped: Optional[List[str]] = None


__all__ = ["DiagnosisSample"]
