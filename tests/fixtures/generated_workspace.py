"""A workspace in the shape the generator leaves one, for tests that consume it.

**Built through `build_split_manifest`, not by writing a plausible manifest.**
Every entry point now verifies the manifest against the exact sample bytes and
the disease sets they hold, so a hand-written manifest in a fixture would encode
this file's belief about that shape and keep passing after the real shape moved --
which is the class of defect the verification exists to catch, reproduced inside
its own tests.

Module: tests/fixtures/generated_workspace.py
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_CONFIG = {"min_phenotypes": 2, "max_phenotypes": 15, "phenotype_drop_rate": 0.3}


def default_graph_export() -> Dict[str, Any]:
    """The recipe a persisted schema-3 manifest carries, taken from the writer.

    **Read from the graph module rather than spelled out here.** A schema-3
    manifest promises `node_features.pt` can be rebuilt, and the reader collects
    that promise, so a fixture that omitted the recipe would build a workspace
    no consumer accepts -- which is exactly what this fixture exists not to do.
    Sourcing the values from the module that performs the draw means a field
    added there arrives here without this file being edited, and a field whose
    domain moves cannot leave the fixture producing something the reader
    refuses.

    The fixture's own export (`write_graph_export`) draws its features with this
    recipe, so for the files it writes the recipe is true. A test that writes its
    own tensors gets the recipe stated anyway: it is recorded rather than
    verified against the tensors -- only the writer of an export knows what it
    passed.
    """
    from src.kg.graph import (
        DEFAULT_FEATURE_SEED,
        FEATURE_INITIALISATION,
        FEATURE_INITIALISATION_VERSION,
    )

    return {
        "feature_dim": 8,
        "feature_seed": DEFAULT_FEATURE_SEED,
        "initialisation": FEATURE_INITIALISATION,
        "initialisation_version": FEATURE_INITIALISATION_VERSION,
    }


def profiles_for(disease_ids: Iterable[int]) -> Dict[int, Dict[str, List[int]]]:
    """A profile per disease, with phenotypes derived from the id."""
    return {
        int(d): {"phenotype_ids": [int(d), int(d) + 100], "gene_ids": [int(d)]}
        for d in disease_ids
    }


def one_sample_per_disease(split: str, ids: Sequence[int], profiles) -> List[Dict[str, Any]]:
    return [
        {"patient_id": f"SECRET-{split}-{i}",
         "phenotype_ids": list(profiles[d]["phenotype_ids"]),
         "disease_id": int(d)}
        for i, d in enumerate(ids)
    ]


def training_samples_per_disease(
    split: str, ids: Sequence[int], profiles
) -> List[Dict[str, Any]]:
    """`one_sample_per_disease`, with the two optional fields training reads.

    Training reads `candidate_disease_ids` and `gene_ids` as well
    (`scripts/train_model.py:463-464`); measurement reads neither. Every row here
    carries both, so a test of training's read is about the fields and not only
    about the flag that asks for them.
    """
    candidates = sorted({int(d) for d in ids})
    rows = one_sample_per_disease(split, ids, profiles)
    for row, d in zip(rows, ids, strict=True):
        row["gene_ids"] = list(profiles[d]["gene_ids"])
        row["candidate_disease_ids"] = list(candidates)
    return rows


def write_graph_export(root: Path, roles: Iterable[str]) -> None:
    """Write the named graph files the way a build writes them, for this workspace.

    **Real files, through the real writers.** `kg.json` comes from
    `KnowledgeGraph.save_json` and the three tensor files from
    `export_graph_data`, with the recipe `default_graph_export()` records, so a
    reader that parses them (contract M2.1) gets what a build would give it.
    Stand-in bytes satisfied a verifier that hashed paths, and would fail a
    parsing reader for the wrong reason.

    **Their content differs per workspace.** Node names carry the directory's
    name, so `kg.json` always differs; the number of nodes of each type is drawn
    from that name, so the tensors and `num_nodes.json` differ too unless two
    names draw the same three counts (one chance in 262,144). A file copied from
    another workspace is that workspace's file, never this one's by accident.

    Writing a tensor file needs torch, and a test that has none is skipped here,
    including one that only checks digests.
    """
    from src.core.types import DataSource, Edge, EdgeType, Node, NodeID, NodeType
    from src.kg import KnowledgeGraph
    from src.kg.artifacts import GRAPH_ARTIFACTS

    wanted = list(roles)
    if not wanted:
        return
    draw = hashlib.sha256(root.name.encode("utf-8")).digest()
    kinds = (
        (NodeType.PHENOTYPE, DataSource.HPO, 1 + draw[0] % 64),
        (NodeType.DISEASE, DataSource.MONDO, 1 + draw[1] % 64),
        (NodeType.GENE, DataSource.DISGENET, 1 + draw[2] % 64),
    )
    kg = KnowledgeGraph()
    nodes: Dict[Any, List[Any]] = {}
    for node_type, source, count in kinds:
        nodes[node_type] = [
            Node(id=NodeID(source=source, local_id=f"{node_type.value}-{i}-of-{root.name}"),
                 node_type=node_type, name=f"{node_type.value} {i} of {root.name}")
            for i in range(count)
        ]
        for node in nodes[node_type]:
            kg.add_node(node)
    diseases = nodes[NodeType.DISEASE]
    for node_type, edge_type in ((NodeType.PHENOTYPE, EdgeType.PHENOTYPE_OF_DISEASE),
                                 (NodeType.GENE, EdgeType.GENE_ASSOCIATED_WITH_DISEASE)):
        for i, node in enumerate(nodes[node_type]):
            kg.add_edge(Edge(source_id=node.id, target_id=diseases[i % len(diseases)].id,
                             edge_type=edge_type))

    recipe = default_graph_export()
    with tempfile.TemporaryDirectory() as staging_dir:
        staging = Path(staging_dir)
        kg.save_json(str(staging / GRAPH_ARTIFACTS["kg"]))
        if any(role != "kg" for role in wanted):
            import pytest

            pytest.importorskip("torch")
            kg.export_graph_data(output_dir=staging, feature_dim=recipe["feature_dim"],
                                 feature_seed=recipe["feature_seed"])
        for role in wanted:
            filename = GRAPH_ARTIFACTS[role]
            (root / filename).write_bytes((staging / filename).read_bytes())


def write_generated_workspace(
    root: Path,
    *,
    train_ids: Sequence[int],
    val_ids: Sequence[int],
    profiles: Optional[Dict[int, Dict[str, Any]]] = None,
    train_samples: Optional[List[Dict[str, Any]]] = None,
    val_samples: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
    graph_bytes: Optional[Dict[str, bytes]] = None,
    graph_export: Optional[Dict[str, Any]] = None,
    training_fields: bool = False,
) -> Tuple[Path, Dict[str, Any]]:
    """Write `train_samples.json`, `val_samples.json` and a real manifest.

    The manifest's artifact digests are taken from the files after they are
    written, exactly as `generate_training_samples` takes them, so the workspace
    satisfies `verify_generated_cohorts` without the fixture knowing what that
    checks.

    Graph files a test has already written, or passes in `graph_bytes`, are kept;
    the rest are written by `write_graph_export`. `training_fields` gives the
    default rows the two fields only training reads.
    """
    from src.kg.artifacts import GRAPH_ARTIFACTS
    from src.kg.disease_allocation import DiseaseAllocation, universe_digest
    from src.kg.sample_generator import build_split_manifest
    from src.utils.fingerprint import file_sha256

    root.mkdir(parents=True, exist_ok=True)
    # The graph export is part of the production event the manifest binds, so a
    # fixture that omitted it would build a workspace no consumer accepts. Written
    # first, then digested, exactly as `build_knowledge_graph` does it.
    supplied = dict(graph_bytes or {})
    for role, data in supplied.items():
        (root / GRAPH_ARTIFACTS[role]).write_bytes(data)
    write_graph_export(root, [
        role for role, filename in GRAPH_ARTIFACTS.items()
        if role not in supplied and not (root / filename).exists()
    ])
    train_ids, val_ids = [int(d) for d in train_ids], [int(d) for d in val_ids]
    profiles = profiles or profiles_for(train_ids + val_ids)
    rows_for = training_samples_per_disease if training_fields else one_sample_per_disease
    rows = {
        "train": train_samples if train_samples is not None
        else rows_for("train", train_ids, profiles),
        "val": val_samples if val_samples is not None
        else rows_for("val", val_ids, profiles),
    }
    for split, samples in rows.items():
        (root / f"{split}_samples.json").write_text(json.dumps(samples))

    ordered = sorted(train_ids + val_ids)
    allocation = DiseaseAllocation(
        train=tuple((d, profiles[d]) for d in sorted(train_ids)),
        val=tuple((d, profiles[d]) for d in sorted(val_ids)),
        val_fraction_requested=len(val_ids) / max(len(ordered), 1),
        seed=0,
        universe_digest=universe_digest([(d, profiles[d]) for d in ordered]),
    )
    manifest = build_split_manifest(
        allocation=allocation,
        train_samples=rows["train"], val_samples=rows["val"],
        config=dict(config or DEFAULT_CONFIG),
        num_train=len(rows["train"]), num_val=len(rows["val"]),
        artifacts={
            "train_samples": file_sha256(root / "train_samples.json"),
            "val_samples": file_sha256(root / "val_samples.json"),
            **{role: file_sha256(root / filename)
               for role, filename in GRAPH_ARTIFACTS.items()},
        },
        graph_export=dict(graph_export or default_graph_export()),
    )
    (root / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return root, manifest
