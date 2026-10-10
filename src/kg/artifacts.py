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
from typing import Any, Dict, Mapping, NamedTuple, Tuple

from src.utils.fingerprint import ReadIdentity

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

#: The graph files every graph consumer parses. `kg.json` is parsed only where a
#: graph object is built (serving, the SP producer); training and measurement
#: compute from these three, and identify `kg.json` with
#: `check_unparsed_kg_json` instead (contract M2.1, decision 1).
GRAPH_TENSOR_ROLES: Tuple[str, ...] = ("node_features", "edge_indices", "num_nodes")

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

_REBUILD = "Rebuild it with scripts/build_knowledge_graph.py --generate-samples."


class ManifestRead(NamedTuple):
    """A split manifest as parsed, and the identity of the bytes it was parsed from.

    A run reads its manifest once and compares every file's digest with this one
    reading (contract M2.1), so a file replaced between two reads shows up as a
    mismatch instead of being checked against a second, newer manifest.
    """

    manifest: Dict[str, Any]
    identity: ReadIdentity


def read_split_manifest(data_dir: Path) -> ManifestRead:
    """Read `split_manifest.json` once, and refuse one this revision cannot read.

    The refusals are the ones each verifier made when it parsed the manifest
    itself: an absent manifest, a schema other than the current one, and a
    missing or malformed export recipe. A file that is not UTF-8 JSON, or not a
    JSON object, is refused naming the file; it used to surface as a bare
    decoding or attribute error.

    Decoded as UTF-8, as the generator writes it; the bytes are released before
    the text is parsed.
    """
    from src.utils.fingerprint import read_once

    manifest_path = data_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ValueError(
            f"{data_dir} has no {MANIFEST_FILENAME}, so nothing records which "
            f"graph export its artifacts are. {_REBUILD}"
        )
    read = read_once(manifest_path)
    identity = read.identity
    try:
        text = read.data.decode("utf-8")
        del read
        manifest = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"{manifest_path} is not UTF-8 JSON ({exc}). {_REBUILD}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(
            f"{manifest_path} is not a JSON object, so it records nothing this "
            f"revision can read. {_REBUILD}"
        )
    require_manifest_schema(manifest, manifest_path)
    return ManifestRead(manifest, identity)


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

    **The path form, a migration aid until contract M2.1's S9.** It hashes each
    file by path, at one instant, while the caller loads it at another, so a file
    replaced between the two is not detected. `verify_graph_reads` closes that: it
    compares the identities of the bytes the run parsed with the run's one
    manifest reading. Callers move to it in S6-S8, and S9 deletes this form.

    Raises:
        ValueError: naming the artifact whose bytes are not the ones recorded.
    """
    from src.utils.fingerprint import file_sha256

    manifest_read = read_split_manifest(data_dir)
    manifest_path = manifest_read.identity.path
    artifacts = manifest_read.manifest.get("artifacts", {})

    observed: Dict[str, str] = {}
    for role, filename in GRAPH_ARTIFACTS.items():
        recorded = artifacts.get(role)
        if recorded is None:
            raise _no_digest(manifest_path, role)
        digest = file_sha256(data_dir / filename)
        if digest != recorded:
            raise _not_the_artifact(data_dir / filename, role, manifest_path, recorded, digest)
        observed[role] = digest

    # **Provenance is deliberately not checked here, and that is a correction.**
    # An earlier revision raised from this function when a declared record was
    # missing or did not match. This function is not only an audit: it is called
    # from `verify_graph_source`, which `DiagnosisPipeline._init_gnn_inference`
    # calls before it loads a tensor — so a lost record stopped a model from
    # initialising, and through the API a cold start then answered `/diagnose`
    # with 503. The tensors were sound; only the note about where they came from
    # was not.
    #
    # Gating service on that is a policy nobody approved, and the plan says the
    # opposite: what a graph was built from is reported, not enforced. So it
    # moved to `workspace_provenance_status`, which returns a state instead of
    # raising, and callers decide. The four bindings above are unchanged and
    # still raise — those say the tensors are not this graph's, which is a
    # different claim entirely.
    return observed


def _no_digest(manifest_path: Path, role: str) -> ValueError:
    return ValueError(
        f"{manifest_path} records no digest for {role}. A manifest that "
        "does not bind the graph its cohorts were cut from cannot say the "
        "tensors beside it are that graph's export. Rebuild the workspace."
    )


def _not_the_artifact(
    path: Path, role: str, manifest_path: Path, recorded: Any, observed: Any
) -> ValueError:
    return ValueError(
        f"{path} is not the {role} artifact {manifest_path} records "
        f"({str(recorded)[:12]}... vs {str(observed)[:12]}...). The graph export, "
        "the allocation and the samples are one production event; this file came "
        "from another."
    )


def _bound_graph_digests(manifest: ManifestRead) -> Dict[str, str]:
    """The digest the manifest records for each of the four graph roles."""
    artifacts = manifest.manifest.get("artifacts", {})
    bound: Dict[str, str] = {}
    for role in GRAPH_ARTIFACTS:
        recorded = artifacts.get(role)
        if recorded is None:
            raise _no_digest(manifest.identity.path, role)
        bound[role] = recorded
    return bound


def verify_graph_reads(
    manifest: ManifestRead, reads: Mapping[str, ReadIdentity]
) -> Dict[str, str]:
    """The graph files a run read must be the export its one manifest reading binds.

    **Compares, and reads nothing** (contract M2.1). `reads` are the identities
    the readers returned with what they parsed (`read_graph_artifacts`, keyed by
    role), and `manifest` is the run's one reading of its manifest. A file
    replaced between the manifest read and its own read therefore shows up here
    as a mismatch, and the run refuses instead of mixing.

    **Every tensor role must have been read** (`GRAPH_TENSOR_ROLES`). An absent
    file has no identity, and is refused rather than skipped: the manifest binds
    all four files of one export. `kg` is compared when it is in `reads`; a run
    that parses `kg.json` passes its identity to `verify_graph_source_read`, and
    one that does not identifies the file with `check_unparsed_kg_json`.

    Returns the manifest-bound digest of all four graph roles, as the path form
    does.

    Raises:
        ValueError: naming the file and the manifest.
    """
    manifest_path = manifest.identity.path
    bound = _bound_graph_digests(manifest)
    unknown = sorted(set(reads) - set(GRAPH_ARTIFACTS))
    if unknown:
        raise ValueError(
            f"{unknown} are not graph artifacts {manifest_path} binds; its roles "
            f"are {list(GRAPH_ARTIFACTS)}"
        )
    unread = [role for role in GRAPH_TENSOR_ROLES if role not in reads]
    if unread:
        files = [str(manifest_path.parent / GRAPH_ARTIFACTS[role]) for role in unread]
        raise ValueError(
            f"{files} ({', '.join(unread)}) was not read, so nothing shows it is "
            f"the export {manifest_path} binds. The manifest binds all four files "
            "of one export; a workspace missing one is not that export."
        )
    for role in GRAPH_ARTIFACTS:
        identity = reads.get(role)
        if identity is not None and identity.sha256 != bound[role]:
            raise _not_the_artifact(
                identity.path, role, manifest_path, bound[role], identity.sha256
            )
    return bound


def verify_graph_source_read(
    manifest: ManifestRead, graph: ReadIdentity, reads: Mapping[str, ReadIdentity]
) -> Dict[str, str]:
    """The graph object's source bytes must be the `kg.json` this manifest binds.

    `verify_graph_source`'s composition check, for a consumer that parses
    `kg.json`: `graph` is the identity `KnowledgeGraph.read_json` returned with
    the graph, so the comparison is of the bytes the graph was built from, at any
    path. Reads nothing.

    Raises:
        ValueError: naming the file and the manifest.
    """
    bound = verify_graph_reads(manifest, reads)
    if graph.sha256 != bound["kg"]:
        raise ValueError(
            f"{graph.path} is not the graph {manifest.identity.path} binds "
            f"({graph.sha256[:12]}... vs {bound['kg'][:12]}...). Its embeddings "
            "would be computed from one graph's tensors and read through another "
            "graph's node identifiers, which no structural check can see. Point "
            "both at one workspace."
        )
    return bound


def check_unparsed_kg_json(data_dir: Path, manifest: ManifestRead) -> None:
    """Identify `kg.json` before a run that does not parse it (M2.1, decision 1).

    Training and measurement compute from the tensors and never parse `kg.json`;
    the role they record is the manifest-bound source graph. This hashes the file
    and compares it with the run's one manifest reading, so a workspace whose
    `kg.json` is not the graph its manifest binds is refused.

    **Narrow on purpose.** It shows only that the file on disk matched at that
    moment: not that the run consumed it, and not that it will load at serving
    later. It never reads the manifest itself and returns nothing to record. A
    consumer that parses `kg.json` compares the identity of its own read instead
    (`verify_graph_source_read`); this is not a second verification API for it.

    Raises:
        ValueError: naming the file and the manifest.
    """
    from src.utils.fingerprint import file_sha256

    manifest_path = manifest.identity.path
    recorded = manifest.manifest.get("artifacts", {}).get("kg")
    if recorded is None:
        raise _no_digest(manifest_path, "kg")
    path = data_dir / GRAPH_ARTIFACTS["kg"]
    observed = file_sha256(path)
    if observed is None:
        raise ValueError(
            f"{path} is absent, so it cannot be the kg artifact {manifest_path} "
            "records. The manifest binds all four files of one export."
        )
    if observed != recorded:
        raise _not_the_artifact(path, "kg", manifest_path, recorded, observed)


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


def workspace_provenance_status(data_dir: Path) -> "ProvenanceStatus":
    """What a workspace records about the inputs its graph was built from.

    **Reports; does not refuse.** Every state this can return is a fact about a
    note beside the graph, not about whether the graph is usable — so nothing
    here raises, and no caller is forced to stop. A caller that decides a
    missing record should block something is making a policy choice, and it
    makes it explicitly by reading this.

    Reads the manifest, when there is one, to learn whether a record was ever
    declared. That is what separates "this build predates provenance" from "it
    had one and it is gone": the file's absence alone cannot tell them apart,
    and giving both the same name would make a broken workspace read as an old
    one. A graph-only workspace has no manifest, and its record is still checked
    against the graph.

    **Not raising is a promise about the disk too.** Every file this touches can
    fail for reasons that say nothing about provenance, and a caller told this
    reports rather than refuses will not have wrapped it. Those return
    `unreadable` with the cause in the detail.

    **Including the `is_file` probes, which is what a first attempt missed.**
    `Path.is_file` calls `stat` and `pathlib` re-raises EACCES -- only ENOENT,
    ENOTDIR, EBADF and ELOOP come back as `False`. A workspace directory this
    process may not traverse therefore fails at the question *is there a graph
    here*, before any read. Reproduced as a real non-root process: with the
    directory at mode 000 this raised `PermissionError` naming `kg.json`, out of
    a function documented never to raise. The boundary below is around the whole
    body for that reason -- a guard per read is a list someone has to keep
    complete, and the probes were not on it.
    """
    from src.kg.provenance import ProvenanceStatus

    try:
        return _workspace_provenance_status(data_dir)
    except OSError as exc:
        return ProvenanceStatus(
            "unreadable", None,
            f"this workspace could not be inspected ({type(exc).__name__}: "
            f"{exc}); what its graph was built from cannot be established, "
            "which is not the same as its being unrecorded",
        )


def _workspace_provenance_status(data_dir: Path) -> "ProvenanceStatus":
    """`workspace_provenance_status`'s body. Free to raise `OSError`.

    The guards inside are not redundant with the boundary around it: they name
    which file failed and what that leaves unestablished, which a catch-all
    cannot. The boundary is the contract; these are the message.
    """
    from src.kg.provenance import ProvenanceStatus, provenance_status
    from src.utils.fingerprint import file_sha256

    kg_path = data_dir / GRAPH_ARTIFACTS["kg"]
    if not kg_path.is_file():
        return ProvenanceStatus(
            "absent", None, f"{data_dir} has no {GRAPH_ARTIFACTS['kg']} to describe"
        )

    declared = None
    manifest_path = data_dir / MANIFEST_FILENAME
    if manifest_path.is_file():
        # **A manifest that cannot be read is not a manifest that declared
        # nothing.** `declared = None` is the value that means "nothing was ever
        # declared", and feeding it here made an unreadable manifest with a
        # missing record report `absent` -- whose detail says the workspace
        # predates provenance, about a workspace that may have declared one and
        # lost it. What is true is narrower: the declaration could not be read.
        try:
            declared = json.loads(manifest_path.read_text()).get("artifacts", {}).get(
                "provenance"
            )
        except (OSError, ValueError, AttributeError) as exc:
            return ProvenanceStatus(
                "unreadable", None,
                f"{MANIFEST_FILENAME} is here and could not be read "
                f"({type(exc).__name__}: {exc}), so whether a record was ever "
                "declared cannot be established. Refusing the workspace over "
                "that manifest is `verify_graph_artifacts`'s to do, not this.",
            )

    try:
        kg_digest = file_sha256(kg_path)
    except OSError as exc:
        return ProvenanceStatus(
            "unreadable", None,
            f"{GRAPH_ARTIFACTS['kg']} is here and could not be read "
            f"({type(exc).__name__}: {exc}), so no record can be checked "
            "against the graph it claims to describe",
        )

    return provenance_status(data_dir, kg_digest, declared)


def require_graph_export_recipe(
    manifest: Dict[str, Any], manifest_path: Path
) -> Dict[str, Any]:
    """The recipe a persisted manifest promises, checked where it is read.

    **The writer being correct is not the guarantee, which is why this stays.**
    `generate_training_samples` once persisted a workspace with `graph_export={}`
    whenever a caller passed no recipe, and every consumer accepted it: the
    version check reads a number and the artifact check reads digests, so a
    schema-3 manifest could promise a reproducible export and carry none. That
    writer now refuses before it creates anything — but a manifest can reach a
    reader from a workspace built by an older revision, a partial copy, or a hand
    edit, and none of those went through it. A promise only its producer checks
    is a promise about that producer.

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

    **The path form, a migration aid until contract M2.1's S9.** It hashes
    ``kg_path`` again rather than the bytes the graph was built from;
    ``verify_graph_source_read`` compares the identity of that read.

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
    "GRAPH_TENSOR_ROLES",
    "ManifestRead",
    "read_split_manifest",
    "verify_graph_reads",
    "verify_graph_source_read",
    "check_unparsed_kg_json",
    "validate_graph_export_recipe",
    "require_graph_export_recipe",
    "require_manifest_schema",
    "workspace_provenance_status",
    "verify_graph_artifacts",
    "verify_graph_source",
]
