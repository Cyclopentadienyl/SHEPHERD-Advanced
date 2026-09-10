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
from typing import Any, Dict, Tuple

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
SPLIT_MANIFEST_SCHEMA_VERSION = 3


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

    **A bounded limitation, stated rather than engineered around.** The digest is
    taken at one instant and the loader reads at another, so a file replaced
    between the two is not detected. Closing that needs locking or a read-then-
    hash of the same handle, and neither is built: the failure this exists to stop
    is a workspace assembled wrongly, not one edited mid-run.

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


#: The graph-export recipe a persisted schema-3 manifest must carry. The point
#: of schema 3 is that `node_features.pt` can be rebuilt, not merely recognised,
#: so a manifest that omits these is not a schema-3 workspace whatever its
#: version field says. Unknown keys beside them are welcome: additive
#: description that an old reader can ignore is not a schema change.
GRAPH_EXPORT_REQUIRED: Tuple[str, ...] = (
    "feature_dim",
    "feature_seed",
    "initialisation",
    "initialisation_version",
)


def validate_graph_export_recipe(recipe: Any) -> Dict[str, Any]:
    """The recipe's own rules, with no manifest and no file around it.

    **One rule, two boundaries.** The reader below re-frames these refusals with
    the path it was reading; the writer in `sample_generator` raises them as they
    are, before it has a path to name. Keeping the rules in one pure function is
    what stops the two from drifting into a state where the writer produces a
    recipe the reader rejects — an artifact guaranteed to be refused is a defect
    whichever side is "right".

    **Shape, never membership.** `initialisation` must name something; it is not
    checked against a list of names this revision knows how to execute. A future
    producer's well-formed, digest-bound artifact must stay readable here — the
    tool asked to *run* a recipe is where capability belongs, and a reader that
    refused unknown names would make every new initialisation a breaking change.

    Raises:
        ValueError: naming the field and what was wrong with it.
    """
    from src.kg.graph import validate_feature_dim, validate_feature_seed

    if not isinstance(recipe, dict) or not recipe:
        raise ValueError(
            "records no graph_export recipe, so its node_features.pt can be "
            "recognised and not rebuilt"
        )
    missing = [name for name in GRAPH_EXPORT_REQUIRED if name not in recipe]
    if missing:
        raise ValueError(
            f"graph_export is missing {missing}; a recipe without them cannot "
            "reproduce the export it describes"
        )

    try:
        validate_feature_dim(recipe["feature_dim"])
        validate_feature_seed(recipe["feature_seed"])
    except ValueError as exc:
        raise ValueError(f"graph_export is malformed: {exc}") from exc

    name = recipe["initialisation"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            "graph_export names no initialisation, so the recipe's numbers "
            f"describe an unknown draw (got {name!r})"
        )
    version = recipe["initialisation_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError(
            "graph_export initialisation_version must be a positive integer, "
            f"got {version!r}"
        )
    return dict(recipe)


def require_graph_export_recipe(
    manifest: Dict[str, Any], manifest_path: Path
) -> Dict[str, Any]:
    """The recipe a persisted manifest promises, checked where it is read.

    **The writer being correct is not the guarantee.** `generate_training_samples`
    will persist a workspace with `graph_export={}` if a caller passes no recipe,
    and every consumer used to accept it: the version check reads a number and
    the artifact check reads digests, so a schema-3 manifest could promise a
    reproducible export and carry none. A promise nothing reads is a comment.

    The rules are `validate_graph_export_recipe`'s, so the reader cannot come to
    require something the writer would not produce; what this adds is the path,
    which is the only part of the message a reader can supply and the writer
    cannot.
    """
    try:
        return validate_graph_export_recipe(manifest.get("graph_export"))
    except ValueError as exc:
        raise ValueError(
            f"{manifest_path} is schema {SPLIT_MANIFEST_SCHEMA_VERSION} and "
            f"{exc}. Rebuild the workspace with "
            "scripts/build_knowledge_graph.py --generate-samples."
        ) from exc


def require_manifest_schema(manifest: Dict[str, Any], manifest_path: Path) -> None:
    version = manifest.get("schema_version")
    if version != SPLIT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"{manifest_path} is split-manifest schema {version!r}; this code "
            f"reads {SPLIT_MANIFEST_SCHEMA_VERSION}. "
            + (
                "Schema 1 did not bind the exported graph artifacts, so a "
                "workspace built under it cannot show that its tensors are this "
                "graph's export. "
                if version == 1
                else "Schema 2 bound the graph artifacts but recorded no export "
                "recipe, so its `node_features.pt` can be identified and not "
                "rebuilt. "
                if version == 2
                else "That version is not one this revision knows how to read, so "
                "which of its fields still mean what they say is a guess. "
            )
            + "Rebuild it with scripts/build_knowledge_graph.py "
            "--generate-samples; there is no migration and no unbound-digest path."
        )
    require_graph_export_recipe(manifest, manifest_path)


def verify_graph_source(kg_path: Path, data_dir: Path) -> Dict[str, str]:
    """The graph object's source file must be this workspace's bound `kg.json`.

    **The composition, not the components.** ``verify_graph_artifacts`` proves a
    workspace is internally consistent, and two workspaces can each be internally
    consistent while a caller pairs one's `kg.json` with the other's tensors. The
    clinical path did exactly that: the API resolves ``SHEPHERD_KG_PATH`` and
    ``SHEPHERD_DATA_DIR`` independently, loads the ``KnowledgeGraph`` from the
    first and hands it to a pipeline pointed at the second. Embedding rows then
    come from one graph and the node-id mapping that interprets them from another,
    and same-shaped workspaces pass every structural check on the way.

    **Compared by digest rather than by path.** Requiring ``kg_path`` to *be*
    ``data_dir/kg.json`` would also close it, and would break a deployment that
    mounts or copies the file elsewhere for reasons of its own. A path is not an
    identity in this project; the bytes are. Any location holding the bound bytes
    is the bound graph.

    This is the one thing a caller must state rather than the code recover: an
    in-memory ``KnowledgeGraph`` has no source digest, so a caller who has only an
    object cannot be checked and is not pretended to be.

    Raises:
        ValueError: if the workspace is unsound, or if ``kg_path``'s bytes are not
            the ones its manifest binds.
    """
    from src.utils.fingerprint import file_sha256

    bound = verify_graph_artifacts(data_dir)
    observed = file_sha256(kg_path)
    if observed != bound["kg"]:
        raise ValueError(
            f"{kg_path} is not the graph {data_dir} was built from "
            f"({str(observed)[:12]}... vs {str(bound['kg'])[:12]}...). Its "
            "embeddings would be computed from one graph's tensors and read "
            "through another graph's node identifiers, which no structural check "
            "can see. Point both at one workspace."
        )
    return bound


__all__ = [
    "GRAPH_ARTIFACTS",
    "MANIFEST_FILENAME",
    "SPLIT_MANIFEST_SCHEMA_VERSION",
    "GRAPH_EXPORT_REQUIRED",
    "validate_graph_export_recipe",
    "require_graph_export_recipe",
    "require_manifest_schema",
    "verify_graph_artifacts",
    "verify_graph_source",
]
