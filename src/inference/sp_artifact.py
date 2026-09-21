"""
The shortest-path artifact: the pair, what binds it, and who may read it.
=========================================================================
`shortest_paths.pt` and `shortest_paths.meta.json` are two files written by one
run and read as though they were one artifact. Nothing recorded that they belong
together, and nothing recorded which graph the distances were computed from.
Both gaps are silent: a new tensor beside a previous run's sidecar is scored
against the old ceiling, and a table computed from another graph resolves node
indices to different nodes. Every existing check asks *is this table well
formed*; these ask **is it the table that belongs here**.

**One module because there is one schema.** The producer writes these fields and
three programs read them — the served pipeline, the benchmark and the
reachability audit. A schema defined at the writer and re-derived at each reader
is the drift this project has repaired more than once, most recently as a
pre-write gate and a writer that encoded with different arguments and disagreed.

**What `build_id` establishes, and nothing more.** Two files declare the same
publication. It is not a checksum of the tensor's bytes, not tamper detection,
and not a freshness claim — two consistent *old* files are consistent. A digest
of the tensor would be the stronger claim and is not taken, because verifying it
means re-reading a multi-gigabyte file on every cold start to detect a threat
that is not the one observed: runs are interrupted and directories are mixed up,
nobody is editing tensors.

**Rule 0, because the states overlap otherwise.** A pair is *legacy* only when
neither file declares this protocol. A declaration on either side makes it a
new-format pair and it is validated as one; nothing falls back to legacy from a
partial state — not a missing field, not an empty value, not an unknown schema,
not a one-sided `build_id`. Legacy artifacts still face every check that already
existed, and are reported as **unrecorded**, never as verified.

**Validation does not vary by caller; use does.** Anything known-invalid raises,
so no caller can proceed over it by forgetting to look. The one state that is
*not* a verdict on the artifact — a binding that cannot be checked because no
comparable graph was supplied — is returned rather than raised, and the caller
decides: a consumer whose ranking depends on the table refuses, while a tool
measuring only the integer ids inside the tensor may proceed and must say so in
its output.

Module: src/inference/sp_artifact.py

Dependencies: torch, and `src.inference.sp_index` for the error type. Import
lazily from anywhere that must stay importable without torch.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from src.inference.sp_index import SPArtifactError

__all__ = [
    "ACCEPTED_SCHEMA_VERSIONS",
    "SPBinding",
    "SP_BINDING_FIELDS",
    "SP_SCHEMA_VERSION",
    "kg_binding_state",
    "new_build_id",
    "publish_sp_artifact",
    "read_binding",
    "read_sidecar",
    "sidecar_path",
]

#: What this producer writes.
SP_SCHEMA_VERSION = 1

#: Compared by membership, never for truthiness. `0`, `""` and `False` are all
#: falsy and none of them means "absent".
ACCEPTED_SCHEMA_VERSIONS = (1,)

#: The three fields that make a pair new-format. Any one of them, on either
#: file, and Rule 0 says the pair is declared.
SP_BINDING_FIELDS = ("schema_version", "build_id", "kg_digest")

#: The key the `.pt` dict carries. The tensor's half of the pairing.
TENSOR_BUILD_ID_KEY = "build_id"

_BUILD_ID = re.compile(r"[0-9a-f]{32}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class SPBinding:
    """What the pair declares about itself.

    `recorded` False is Rule 0's legacy: neither file declared anything, so the
    other fields are None and the artifact is **unrecorded**, which is not the
    same as verified and must never be reported as it.
    """

    recorded: bool
    schema_version: Optional[int] = None
    build_id: Optional[str] = None
    kg_digest: Optional[str] = None


def new_build_id() -> str:
    """A fresh 128-bit token, one per publication."""
    return secrets.token_hex(16)


def sidecar_path(tensor_path: Any) -> Path:
    """The sidecar beside a `.pt`. One definition, so no caller spells it."""
    return Path(tensor_path).with_suffix(".meta.json")


def read_sidecar(tensor_path: Any) -> Optional[Dict[str, Any]]:
    """The sidecar as a mapping, or None when there is no file.

    **Present and unusable raises.** A sidecar that is here and cannot be parsed
    is a broken deployment, not an absent one: the previous shape read it inside
    `except Exception: pass`, so a missing file, malformed JSON, an absent key
    and a value the producer would never write all arrived at the same silent
    default.
    """
    path = sidecar_path(tensor_path)
    try:
        present = path.is_file()
    except OSError as exc:
        raise SPArtifactError(
            f"{path} could not be looked up ({type(exc).__name__}: {exc}); "
            "whether a sidecar is there is unknown, which is not the same as "
            "its being absent"
        ) from exc
    if not present:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise SPArtifactError(
            f"{path} is present and could not be read as UTF-8 text "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    try:
        meta = json.loads(text)
    except ValueError as exc:
        raise SPArtifactError(
            f"{path} is present but not readable JSON ({type(exc).__name__}); "
            "the hop bound it declares cannot be recovered from the tensors"
        ) from exc
    if not isinstance(meta, dict):
        raise SPArtifactError(f"{path} is not a JSON object, so it describes no artifact")
    return meta


def read_binding(
    sidecar: Optional[Dict[str, Any]],
    tensor_keys: Iterable[str],
    *,
    source: str = "the shortest-path artifact",
) -> SPBinding:
    """Rule 0, then the full new-format check. **Known-invalid raises.**

    Args:
        sidecar: the parsed sidecar, or None when there is no file.
        tensor_keys: the keys of the loaded `.pt` mapping.
        source: what to name in a refusal.

    Returns:
        `SPBinding(recorded=False)` for a legacy pair, or a fully validated
        binding. It never returns a partially populated one — that state raises,
        because a pair that declares half of a protocol has not satisfied it.
    """
    meta = sidecar or {}
    tensor_keys = set(tensor_keys)
    declared_in_sidecar = [field for field in SP_BINDING_FIELDS if field in meta]
    declared_in_tensor = TENSOR_BUILD_ID_KEY in tensor_keys

    if not declared_in_sidecar and not declared_in_tensor:
        return SPBinding(recorded=False)

    # From here the pair is new-format and every field is required. Partial is
    # a refusal, never a fall back to legacy: the state this exists to catch —
    # a new tensor beside an old sidecar — is exactly a partial declaration.
    version = meta.get("schema_version")
    if version not in ACCEPTED_SCHEMA_VERSIONS:
        raise SPArtifactError(
            f"{source} declares schema_version {version!r}; this code reads "
            f"{list(ACCEPTED_SCHEMA_VERSIONS)}. Which of its fields still mean "
            "what they say is a guess."
        )

    sidecar_build_id = meta.get("build_id")
    if not isinstance(sidecar_build_id, str) or not _BUILD_ID.match(sidecar_build_id):
        raise SPArtifactError(
            f"{source}: the sidecar's build_id is not a 32-character hexadecimal "
            f"token ({sidecar_build_id!r}); the pair cannot be checked"
        )
    if not declared_in_tensor:
        raise SPArtifactError(
            f"{source}: the sidecar declares build_id {sidecar_build_id[:12]}... "
            "and the tensor carries none. One side of a pair is not a pair — "
            "this is a table published without its sidecar, or a sidecar "
            "written for a table that never landed."
        )

    kg_digest = meta.get("kg_digest")
    if not isinstance(kg_digest, str) or not _DIGEST.match(kg_digest):
        raise SPArtifactError(
            f"{source}: the sidecar's kg_digest is not a SHA-256 hexdigest "
            f"({kg_digest!r}); which graph these distances describe is unstated"
        )

    return SPBinding(
        recorded=True,
        schema_version=version,
        build_id=sidecar_build_id,
        kg_digest=kg_digest,
    )


def require_paired(
    binding: SPBinding,
    tensor_build_id: Any,
    *,
    source: str = "the shortest-path artifact",
) -> None:
    """The two halves name the same publication. **Refuses for every caller.**

    This is the check the whole protocol exists for: a publication interrupted
    between its two replacements leaves one new file and one old one, and
    nothing else can tell.
    """
    if not binding.recorded:
        return
    if not isinstance(tensor_build_id, str) or not _BUILD_ID.match(tensor_build_id):
        raise SPArtifactError(
            f"{source}: the tensor's build_id is not a 32-character hexadecimal "
            f"token ({tensor_build_id!r})"
        )
    if tensor_build_id != binding.build_id:
        raise SPArtifactError(
            f"{source}: the tensor and its sidecar were published by different "
            f"runs (tensor {tensor_build_id[:12]}..., sidecar "
            f"{str(binding.build_id)[:12]}...). One of them is left over from an "
            "interrupted or mixed publication, and which is current cannot be "
            "decided from the files."
        )


def kg_binding_state(
    binding: SPBinding,
    expected_kg_digest: Optional[str],
    *,
    source: str = "the shortest-path artifact",
) -> str:
    """Which graph this was computed from, against the graph being consumed.

    **The comparison is to the graph that supplies the node mapping**, never to
    whatever `kg.json` happens to sit beside the artifact. A caller that builds
    a pipeline from an in-memory graph skips the file-backed verification by
    design; binding to a file next to the table would let the check be bypassed
    by deleting one.

    Returns:
        ``"unrecorded"`` — legacy pair, nothing to compare.
        ``"verified"`` — declared and equal to the consumed graph.
        ``"unverifiable"`` — declared, and no trustworthy digest was supplied
        for the consumed graph. **Not a verdict on the artifact**, which is why
        it is returned rather than raised: a consumer whose ranking depends on
        this table must refuse, and a tool reading only the integer ids inside
        the tensor may proceed and must label the limitation.

    Raises:
        SPArtifactError: both are known and they differ.
    """
    if not binding.recorded or binding.kg_digest is None:
        return "unrecorded"
    if expected_kg_digest is None:
        return "unverifiable"
    if binding.kg_digest != expected_kg_digest:
        raise SPArtifactError(
            f"{source} was computed from a different graph "
            f"({binding.kg_digest[:12]}... vs {expected_kg_digest[:12]}...). "
            "Its node indices are positions in that graph's mapping, so every "
            "distance would be read against the wrong nodes — which no "
            "structural check can see."
        )
    return "verified"


def publish_sp_artifact(
    sp_data: Dict[str, Any],
    tensor_path: Any,
    metadata: Dict[str, Any],
) -> None:
    """Write both files whole, then move each into place.

    **Two whole-file replacements plus a reader that checks the pairing — not a
    two-file transaction**, and the difference bounds what is promised. A
    failure while writing either temporary file leaves the live pair untouched
    and still exactly as valid as it was. A failure between the two replacements
    leaves a mixed pair, and `require_paired` refuses it rather than serving it.
    Which file is replaced first is therefore not load-bearing, which is the
    point: ordering alone only moves *which* file is stale.

    **The sidecar is encoded before the tensor is written**, and the bytes that
    proved to encode are the bytes written. A pre-write check that re-encodes is
    a second encoder, and the two drift — measured once already, on a pre-write
    gate and a writer whose `json.dumps` arguments differed.

    Durability across power loss is **not** claimed: that needs a directory
    `fsync` and a statement about the filesystem. What is covered is write
    errors and process death.
    """
    import torch

    tensor_path = Path(tensor_path)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = sidecar_path(tensor_path)

    payload = json.dumps(metadata, indent=2, sort_keys=True)

    staged = []
    try:
        tensor_tmp = _stage(tensor_path, lambda handle: torch.save(sp_data, handle))
        staged.append(tensor_tmp)
        meta_tmp = _stage(meta_path, lambda handle: handle.write(payload.encode("utf-8")))
        staged.append(meta_tmp)

        os.replace(tensor_tmp, tensor_path)
        staged.remove(tensor_tmp)
        os.replace(meta_tmp, meta_path)
        staged.remove(meta_tmp)
    finally:
        for leftover in staged:
            Path(leftover).unlink(missing_ok=True)


def _stage(target: Path, write) -> str:
    """One file, written whole to a temporary name beside its destination."""
    handle = tempfile.NamedTemporaryFile(
        "wb", dir=str(target.parent), prefix=target.name, suffix=".tmp", delete=False
    )
    try:
        with handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return handle.name
