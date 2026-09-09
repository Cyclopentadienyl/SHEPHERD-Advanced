"""Producing a workspace: the one sequence that produces one.

A workspace is not a directory that happens to contain the right filenames. It
is a **single production event** — a graph cut into a disease-level allocation,
exported to tensors, and generated into cohorts — recorded by a manifest that
binds all six artifacts to each other by digest. Every consumer verifies that
binding: training, measurement, and the clinical inference path all refuse a
directory whose parts came from different events.

That makes the ordering load-bearing, and the ordering is easy to get wrong in a
way nothing detects until much later:

  1. the allocation is cut from the in-memory graph, and every refusal that can
     be decided from it happens **before the first byte is written**;
  2. `kg.json`, then the three tensors, are written;
  3. their digests are taken **by the writer**, from the files it just wrote;
  4. the cohorts are generated and the manifest records all six together.

Steps 3 and 4 are why this cannot be assembled by a caller from the outside: a
generator that hashed the artifacts itself would be digesting whatever files
happen to sit in the directory, and recording them beside this allocation's
universe digest as though they were one chain.

**This module exists so there is exactly one implementation of that sequence.**
It used to live inside `build_knowledge_graph.py`, which made "the production
path" a property of one script rather than of the project: `setup_demo.py` wrote
a workspace too, with its own sample generator that split patients rather than
diseases — the very defect this branch exists to remove — and no manifest at
all. It produced a directory that looked right, trained, and then could not be
served, because the verifier correctly refused it. Deleting that second producer
was not the fix; having one producer is.

Module: src/kg/workspace.py
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional

logger = logging.getLogger(__name__)


class SampleBudget(NamedTuple):
    """How many patients to simulate, and how to cut the diseases they come from.

    ``val_disease_fraction`` is a *disease* fraction, not a patient fraction:
    the partitions are disjoint at the disease level by construction, which is
    the property a patient-level slice cannot provide (EVALUATION_COHORTS §6.2).
    """

    num_train: int
    num_val: int
    val_disease_fraction: float = 0.15
    seed: int = 42
    min_phenotypes: int = 2


class WorkspaceWrite(NamedTuple):
    """What one production event produced."""

    graph_digests: Dict[str, str]
    allocation: Optional[Any]
    train_samples: List[Dict[str, Any]]
    val_samples: List[Dict[str, Any]]
    manifest: Optional[Dict[str, Any]]


def write_workspace(
    kg: Any,
    workspace: Path,
    *,
    feature_dim: int = 128,
    samples: Optional[SampleBudget] = None,
    preflight: Optional[Callable[[Any], None]] = None,
) -> WorkspaceWrite:
    """Write `kg.json`, the graph tensors, and — when asked — bound cohorts.

    Args:
        kg: the finished in-memory KnowledgeGraph. Nothing here builds or edits
            it; the caller owns what goes into the graph.
        workspace: the directory to write. Created if absent.
        feature_dim: node feature width for the export.
        samples: when given, cut an allocation and generate cohorts with a
            `split_manifest.json`. When omitted, only the graph is written — a
            directory no consumer accepts on its own, which is why every
            production caller supplies budgets.
        preflight: called with the allocation once it is cut and **before any
            byte is written**, so a caller can refuse in its own vocabulary (a
            command-line tool naming its flags, say) while the ordering that
            makes the refusal free stays here. The generator refuses an
            undersized budget too, but only after the graph has been saved.

    Returns:
        The digests, the allocation, both cohorts and the manifest.
    """
    from src.kg.artifacts import GRAPH_ARTIFACTS
    from src.kg.sample_generator import refuse_if_checkpoints_exist
    from src.utils.fingerprint import file_sha256

    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    # **A workspace under trained checkpoints is not rewritable.** Rebuilding
    # the graph beneath them leaves those checkpoints paired with a graph they
    # were never trained on, and nothing in the checkpoint says so. Callers that
    # can check earlier still should — an ontology load is minutes — but the
    # authoritative refusal belongs to the writer, because it is the writer that
    # would do the damage.
    refuse_if_checkpoints_exist(workspace)

    allocation = None
    if samples is not None:
        from src.kg import allocate_diseases, build_eligible_disease_profiles

        eligible = build_eligible_disease_profiles(
            kg, min_phenotypes=samples.min_phenotypes
        )
        allocation = allocate_diseases(
            eligible, samples.val_disease_fraction, seed=samples.seed
        )
        if preflight is not None:
            preflight(allocation)

    kg_path = workspace / GRAPH_ARTIFACTS["kg"]
    kg.save_json(str(kg_path))
    logger.info("KG saved to %s", kg_path)

    kg.export_graph_data(output_dir=workspace, feature_dim=feature_dim)
    logger.info("Graph data exported to %s (feature_dim=%d)", workspace, feature_dim)

    graph_digests = {
        role: file_sha256(workspace / filename)
        for role, filename in GRAPH_ARTIFACTS.items()
    }

    if samples is None:
        return WorkspaceWrite(graph_digests, None, [], [], None)

    from src.kg.sample_generator import generate_training_samples

    train_samples, val_samples, manifest = generate_training_samples(
        kg=kg,
        allocation=allocation,
        num_train=samples.num_train,
        num_val=samples.num_val,
        min_phenotypes=samples.min_phenotypes,
        output_dir=workspace,
        graph_digests=graph_digests,
    )
    logger.info(
        "Generated %d train samples over %d diseases, %d val over %d — disjoint: %s",
        len(train_samples), manifest["realised"]["train_diseases"],
        len(val_samples), manifest["realised"]["val_diseases"], manifest["disjoint"],
    )
    return WorkspaceWrite(
        graph_digests, allocation, train_samples, val_samples, manifest
    )


__all__ = ["SampleBudget", "WorkspaceWrite", "write_workspace"]
