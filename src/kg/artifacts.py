"""
What a built workspace holds, named once.
=========================================
`KnowledgeGraph.save_json` writes one file and `export_graph_data` writes three
more. Those four are one production event from one in-memory graph, and three
places need to agree on their names: the writer that produces them, the manifest
builder that binds their digests, and every consumer that verifies them.

The map lived in `src/evaluation/cohort.py`, which put a fact about
`src.kg`'s own output above `src.kg` in the layer order — so `sample_generator`
importing it broke the layered-architecture contract. It belongs here, at the
layer that writes the files, and the verifier above imports it.

No torch, so a consumer that needs these does not pull in the graph machinery to
get them — which is what lets `src.inference` verify a workspace without reaching
above its own layer.

Module: src/kg/artifacts.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

#: Manifest role → filename, for the artifacts a graph consumer reads.
#:
#: `kg.json` is the serialised graph; the other three are the PyG export a model
#: actually loads. Binding only the first left the tensors bound to nothing, and
#: `graph_fingerprint` does not close that — it is structural, so a same-shaped
#: `node_features.pt` from another workspace shares it.
GRAPH_ARTIFACTS: Dict[str, str] = {
    "kg": "kg.json",
    "node_features": "node_features.pt",
    "edge_indices": "edge_indices.pt",
    "num_nodes": "num_nodes.json",
}

#: The file the generator writes to record how a workspace was cut.
MANIFEST_FILENAME = "split_manifest.json"

#: Bumped when the manifest's shape changes in a way that makes an old file
#: unreadable under the new rules.
#:
#: v2 — the manifest binds the **exported graph artifacts** as well as `kg.json`
#: and the sample files. v1 bound only `kg.json` and the samples, so a workspace
#: built under it cannot show that the tensors a model consumes are this graph's
#: export. Bumped rather than extended in place, and refused rather than migrated:
#: the missing digests cannot be recovered after the fact, because only the writer
#: could have vouched for them.
#:
#: **Lives here rather than in `sample_generator`** so that this module — which
#: every graph consumer imports, down to the clinical inference pipeline — does
#: not have to reach up into the generator to know what schema it is reading.
SPLIT_MANIFEST_SCHEMA_VERSION = 2


def verify_graph_artifacts(data_dir: Path) -> Dict[str, str]:
    """The graph a run consumes must be the export this workspace's manifest names.

    **A separate contract from the cohort one, and it applies to every graph
    consumer.** A supplied institutional cohort carries no allocation and is never
    subject to the generated splits' disjointness — but it is scored against
    ``node_features.pt`` and ``edge_indices.pt`` exactly like a generated one. Had
    graph binding lived inside ``verify_generated_cohorts``, generated validation
    would have been protected while supplied evaluation went on consuming a mixed
    workspace, which is the case the whole distinction exists to keep straight.

    **Why the whole set rather than the roles a caller opens.** Cohort
    verification is scoped because a *use* can legitimately not involve `val`.
    No use can legitimately involve a workspace missing one of these four: they
    are written together by one export from one graph, and a workspace that has a
    manifest has all four. Verifying the set therefore blocks nothing, and it is
    what makes the transitive claim available — training reads only the three
    tensors, and only the manifest ties them to the `kg.json` they were exported
    from.

    ``graph_fingerprint`` does not substitute for this. It is structural — node
    types, counts, feature dimensions — so a same-shaped ``node_features.pt`` from
    another workspace shares it and passes.

    Raises:
        ValueError: naming the artifact whose bytes are not the ones recorded.
    """
    from src.utils.fingerprint import file_sha256

    manifest_path = data_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ValueError(
            f"{data_dir} has no {MANIFEST_FILENAME}, so nothing records which "
            "graph export its artifacts are. Rebuild it with "
            "scripts/build_knowledge_graph.py --generate-samples."
        )
    manifest = json.loads(manifest_path.read_text())
    require_manifest_schema(manifest, manifest_path)
    artifacts = manifest.get("artifacts", {})

    observed: Dict[str, str] = {}
    for role, filename in GRAPH_ARTIFACTS.items():
        recorded = artifacts.get(role)
        if recorded is None:
            raise ValueError(
                f"{manifest_path} records no digest for {role}. A manifest that "
                "does not bind the graph its cohorts were cut from cannot say the "
                "tensors beside it are that graph's export. Rebuild the workspace."
            )
        digest = file_sha256(data_dir / filename)
        if digest != recorded:
            raise ValueError(
                f"{data_dir / filename} is not the {role} artifact "
                f"{manifest_path} records ({str(recorded)[:12]}... vs "
                f"{str(digest)[:12]}...). The graph export, the allocation and the "
                "samples are one production event; this file came from another."
            )
        observed[role] = digest
    return observed


def require_manifest_schema(manifest: Dict[str, Any], manifest_path: Path) -> None:
    version = manifest.get("schema_version")
    if version != SPLIT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"{manifest_path} is split-manifest schema {version!r}; this code "
            f"reads {SPLIT_MANIFEST_SCHEMA_VERSION}. Schema 1 did not bind the "
            "exported graph artifacts, so a workspace built under it cannot show "
            "that its tensors are this graph's export. Rebuild it with "
            "scripts/build_knowledge_graph.py --generate-samples; there is no "
            "migration and no unbound-digest path."
        )


__all__ = [
    "GRAPH_ARTIFACTS",
    "MANIFEST_FILENAME",
    "SPLIT_MANIFEST_SCHEMA_VERSION",
    "require_manifest_schema",
    "verify_graph_artifacts",
]
