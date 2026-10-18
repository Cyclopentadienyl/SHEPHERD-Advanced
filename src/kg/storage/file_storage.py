"""
Reading the knowledge graph's on-disk layout — one implementation.
==================================================================
The KG is written by `src/kg/builder.py` as files under
`data/workspaces/<kg>/`, and read back independently by every consumer that
needs it. This is the shared reader those copies are meant to collapse into.

**This is the P1's first callers, not the P1.** The package docstring records the
whole job: become the single reader of this layout, replacing all six copies.
The measurement harness was the first migrated here, because B-0.3 needed a
reader that does **not** retire with the frozen evaluator, and adding another
copy to get one would have been the opposite of the point. Serving
(`src/inference/pipeline.py`) and training (`scripts/train_model.py`) moved here
in contract M2.1's S6 and S8, and `scripts/setup_demo.py` reads through
training. `scripts/build_index.py` and `scripts/spikes/validate_fast_subgraph.py`
still have their own; migrating them belongs to P1 and is not smuggled in here.

**No abstraction beyond the readers below.** No `Storage` Protocol, no
backend registry, no adapter hierarchy, no migration framework. The package name
says "backends" plural and that plural is aspirational; the shape a second
backend needs will be known when there is one. `read_samples_file` is the one
samples parser; `read_samples` only names a workspace split's file for it.

**Each file is parsed from one read, and its digest comes from that read**
(contract M2.1). A reader returns what it parsed and a `ReadIdentity`, never the
bytes; each buffer is released before the next file is read, which the release
check in `tests/fixtures/release.py` shows. Torch files are parsed through this
module's own `BytesIO`, which that check replaces to watch the wrapper.

**A caller that knows which bytes it wants says so, and the comparison comes
before the parse** (M4, R10). `read_samples_file(..., expected_sha256=...)`
compares the digest of its one read with the digests given and refuses other
bytes undecoded (`SamplesDigestMismatch`), so a file that is not the one asked
for never fails as a parse error. A file that was read and does not parse is
refused as `UnparseableFile`, which carries that read's identity, so a caller
that asked for several digests knows which one the unparseable bytes were.

Module: src/kg/storage/file_storage.py
"""
from __future__ import annotations

import json
from collections.abc import Set as AbstractSet
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, NamedTuple, Optional, Tuple, Union

from src.kg.artifacts import GRAPH_ARTIFACTS
from src.kg.phenotype_normalisation import PHENOTYPE_NORMALISATION_VERSION, normalise_phenotypes
from src.utils.fingerprint import ReadIdentity

__all__ = [
    "GraphArtifactsRead",
    "SamplesDigestMismatch",
    "SamplesRead",
    "UnparseableFile",
    "read_graph_artifacts",
    "read_samples",
    "read_samples_file",
]


class GraphArtifactsRead(NamedTuple):
    """The graph tensors a consumer loads, and the identity of each file read.

    `reads` is keyed by manifest role (`GRAPH_ARTIFACTS`) and holds exactly the
    files that were read, so an absent file has neither a key in `graph_data`
    nor an identity.
    """

    graph_data: Dict[str, Any]
    reads: Dict[str, ReadIdentity]


class SamplesRead(NamedTuple):
    """One samples file's samples, and the identity of the file they were parsed from."""

    samples: List[Any]
    identity: ReadIdentity


class SamplesDigestMismatch(ValueError):
    """A samples file whose bytes are not one of the digests the caller asked for.

    Raised **before the bytes are decoded or parsed**, from the same read whose
    digest was compared, and after that read's buffer is released, so neither
    the refusal nor its traceback keeps the bytes.

    **Why before the parse.** R10 reads every candidate file to find the one a
    checkpoint names, and most candidates are not it: another workspace's split,
    another cohort's format, a file that is not JSON at all. Parsing first would
    turn each of those into a parse error that answers a question nobody asked,
    and would end the search at the first stranger. `expected` is a frozenset
    because such a caller may still want two files (training and validation)
    from one read of each candidate.
    """

    def __init__(self, path: Path, sha256: str, expected: FrozenSet[str]) -> None:
        self.path = path
        self.sha256 = sha256
        self.expected = expected
        wanted = " or ".join(sorted(expected))
        super().__init__(
            f"{path}: its SHA-256 is {sha256}, not {wanted}; its bytes were neither "
            "decoded nor parsed"
        )


class UnparseableFile(ValueError):
    """A file that was read but does not parse, refused under its own name.

    `identity` is the read's: the file was read and its digest taken, so a caller
    that asked for several digests can tell which file it was. `what` names what
    the bytes failed to parse as ("UTF-8 JSON", "a samples file (a list of sample
    rows)", ...), without any of their content, for a caller that records the
    failure where row data must not go (R10's evidence problem).

    "Parse" is the reader's whole parse, from bytes to the objects it returns,
    including the integer-id check that makes a samples row usable and the
    phenotype normal-form check (N2) that a stored case lists each phenotype once.
    """

    def __init__(self, message: str, *, identity: ReadIdentity, what: str) -> None:
        super().__init__(message)
        self.identity = identity
        self.what = what


def _unparseable(
    path: Path, what: str, exc: Exception, identity: ReadIdentity
) -> UnparseableFile:
    """A file that was read but does not parse, refused under its own name.

    **Why the readers say this, not the caller.** An entry point reads, then
    verifies the digest against the manifest. A file that does not parse fails
    in the parse, before the comparison that would have named it, and torch's
    or json's own message (`pop from empty list`, `Expecting value`) names
    nothing. The cause stays chained for whoever debugs it.
    """
    return UnparseableFile(
        f"{path} could not be parsed as {what} ({type(exc).__name__}: {exc})",
        identity=identity,
        what=what,
    )


def _read_json(
    path: Path, expected_sha256: Optional[FrozenSet[str]] = None
) -> Tuple[Any, ReadIdentity]:
    """Decoded as UTF-8, as the workspace writers write it; the bytes are released
    before the text is parsed.

    With `expected_sha256`, the digest of this read is compared first: bytes that
    are none of those digests are released and refused (`SamplesDigestMismatch`)
    without being decoded, so the comparison and the parse are of one buffer and
    a stranger's bytes never reach the decoder.
    """
    from src.utils.fingerprint import read_once

    read = read_once(path)
    identity = read.identity
    if expected_sha256 is not None and identity.sha256 not in expected_sha256:
        del read
        raise SamplesDigestMismatch(path, identity.sha256, expected_sha256)
    try:
        text = read.data.decode("utf-8")
        del read
        parsed = json.loads(text)
    except ValueError as exc:  # UnicodeDecodeError and JSONDecodeError are both ValueErrors
        raise _unparseable(path, "UTF-8 JSON", exc, identity) from exc
    return parsed, identity


def _read_tensors(path: Path, map_location: Any) -> Tuple[Any, ReadIdentity]:
    import torch

    from src.utils.fingerprint import read_once

    read = read_once(path)
    try:
        tensors = torch.load(BytesIO(read.data), weights_only=True, map_location=map_location)
    except MemoryError:
        # Not a property of the file. Converting it would send a capacity
        # shortfall to whoever reads the message as a corrupt export.
        raise
    except RuntimeError as exc:
        # Neither is torch's own report of a failed allocation: c10's CPU
        # allocator raises a plain RuntimeError ("DefaultCPUAllocator: can't
        # allocate memory"), and a device raises `torch.OutOfMemoryError`.
        if isinstance(exc, torch.OutOfMemoryError) or "can't allocate memory" in str(exc):
            raise
        raise _unparseable(path, "a tensor file", exc, read.identity) from exc
    except Exception as exc:  # noqa: BLE001 — torch raises a different type for each kind of junk
        raise _unparseable(path, "a tensor file", exc, read.identity) from exc
    return tensors, read.identity


def read_graph_artifacts(data_dir: Path, *, map_location: Any = "cpu") -> GraphArtifactsRead:
    """Load `node_features.pt`, `edge_indices.pt` and `num_nodes.json`.

    Absent files are omitted rather than defaulted: a caller that needs
    `x_dict` should fail on its absence with its own message, which says what
    the caller was trying to do, rather than receive an empty dict that behaves
    like a graph with no nodes.

    `weights_only=True` — these are tensor files, and loading them must not be
    able to execute code. They re-enter a clinical tool. `map_location="cpu"`,
    as serving already passes: an export is saved from CPU, so this matches
    loading with no `map_location` for every export the writer produces.
    """
    graph_data: Dict[str, Any] = {}
    reads: Dict[str, ReadIdentity] = {}
    for role, key in (("node_features", "x_dict"), ("edge_indices", "edge_index_dict")):
        path = data_dir / GRAPH_ARTIFACTS[role]
        if path.exists():
            graph_data[key], reads[role] = _read_tensors(path, map_location)
    num_nodes = data_dir / GRAPH_ARTIFACTS["num_nodes"]
    if num_nodes.exists():
        graph_data["num_nodes_dict"], reads["num_nodes"] = _read_json(num_nodes)
    return GraphArtifactsRead(graph_data, reads)


def read_samples(data_dir: Path, split: str, *, training_fields: bool = False) -> SamplesRead:
    """Load `<split>_samples.json` as `DiagnosisSample` objects.

    Missing is an error, not an empty cohort: an empty list would flow into a
    measurement and produce metrics over nobody.

    **The error names the splits that do exist, and never falls back to one.**
    `src/kg/sample_generator.py` writes `train_samples.json` and
    `val_samples.json` only, so a caller asking for `test` on an ordinary
    workspace gets a file that was never generated. Listing what is present
    turns that from "which path did I mistype" into "this workspace has no test
    split" — and choosing a substitute here would silently answer a question
    the caller has to answer, since `val` is the checkpoint-selection split
    rather than held-out data.

    **Three fields by default; `training_fields=True` adds the two only
    training reads,** `candidate_disease_ids` and `gene_ids`. Measurement and
    the verifiers read the three default fields; training passes its two extra
    fields explicitly.

    The file is parsed by `read_samples_file`, the one samples parser; this
    function only names a workspace split's file and explains its absence.
    """
    path = data_dir / f"{split}_samples.json"
    if not path.exists():
        available = sorted(
            candidate.name[: -len("_samples.json")]
            # `is_file()`: a directory named `foo_samples.json` is not a split,
            # and reporting one as available would send the caller after a name
            # that can never load.
            for candidate in data_dir.glob("*_samples.json")
            if candidate.is_file()
        )
        detail = (
            f"this workspace has: {', '.join(available)}"
            if available
            else "this workspace has no *_samples.json files at all"
        )
        raise FileNotFoundError(f"Samples file not found: {path} — {detail}")
    return read_samples_file(path, training_fields=training_fields)


def read_samples_file(
    path: Path,
    *,
    training_fields: bool = False,
    expected_sha256: Union[str, AbstractSet[str], None] = None,
) -> SamplesRead:
    """Load one samples file, a JSON list of sample rows, as `DiagnosisSample` objects.

    The one samples parser: `read_samples` (a workspace split), the registered
    cohort version's `samples.json` and R10's evidence candidates all come here.
    One `read_once`; the identity returned is the digest of the bytes parsed.

    **`expected_sha256` compares before it parses.** Given a digest or a set of
    them, bytes whose digest is not among them raise `SamplesDigestMismatch`
    before they are decoded, from the same read. Bytes that match, or every
    file when nothing is expected, are parsed and checked exactly as
    `read_samples` always has, and refused as `UnparseableFile` under the file's
    name. An empty expectation is refused before anything is read: nothing could
    match it, so the read could only be thrown away.

    Missing is `FileNotFoundError` naming the path, never an empty list.
    `training_fields` is as `read_samples` describes.
    """
    from src.kg.data_loader import DiagnosisSample

    expected = _expected_digests(path, expected_sha256)
    try:
        entries, identity = _read_json(path, expected)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Samples file not found: {path}") from exc
    # A file that is JSON but not a list of sample rows is refused under its
    # name too: without `expected_sha256`, the entry point compares digests only
    # after this parse.
    try:
        if training_fields:
            samples = [
                DiagnosisSample(
                    patient_id=entry["patient_id"],
                    phenotype_ids=entry["phenotype_ids"],
                    disease_id=entry["disease_id"],
                    candidate_disease_ids=entry.get("candidate_disease_ids"),
                    gene_ids=entry.get("gene_ids"),
                )
                for entry in entries
            ]
        else:
            samples = [
                DiagnosisSample(
                    patient_id=entry["patient_id"],
                    phenotype_ids=entry["phenotype_ids"],
                    disease_id=entry["disease_id"],
                )
                for entry in entries
            ]
    except (KeyError, TypeError, AttributeError) as exc:
        raise _unparseable(path, "a samples file (a list of sample rows)", exc, identity) from exc
    _refuse_non_integer_ids(path, samples, identity)
    _refuse_phenotype_repeats(path, samples, identity)
    return SamplesRead(samples, identity)


def _expected_digests(
    path: Path, expected_sha256: Union[str, AbstractSet[str], None]
) -> Optional[FrozenSet[str]]:
    """`None` (parse whatever is read), or the non-empty set of digests to accept.

    A bare string is one digest. Anything else, an empty set or a value that is
    not a non-empty string, is a caller's error and refused before the read.
    """
    if expected_sha256 is None:
        return None
    if isinstance(expected_sha256, str):
        expected_sha256 = {expected_sha256}
    elif not isinstance(expected_sha256, AbstractSet):
        raise TypeError(
            f"expected_sha256 for {path} must be a digest string or a set of them, "
            f"not {type(expected_sha256).__name__}; nothing was read"
        )
    digests = frozenset(expected_sha256)
    if not digests or not all(isinstance(d, str) and d for d in digests):
        raise ValueError(
            f"expected_sha256 for {path} names no usable digest ({sorted(map(repr, digests))}); "
            "nothing could match it, so nothing was read"
        )
    return digests


def _refuse_non_integer_ids(path: Path, samples: List[Any], identity: ReadIdentity) -> None:
    """Every id a sample row carries must be a JSON integer; anything else is
    refused under the file's name and the patient's.

    **Refused, not converted.** Each caller turns these ids into an index tensor
    (`torch.tensor(..., dtype=torch.long)` or the like), which truncates `3.7` to
    `3` and reads `true` as `1`: the converted id names a node the file did not,
    and the run scores a plausible patient who is not this one. A boolean is
    refused although Python counts it as an `int`. Checked here because measurement,
    training, the cohort verifiers and the audits all read samples through this
    function, so the check comes before any of them converts. It says nothing
    about range, which needs the graph: Mode C's `_assert_ids_in_range` and the
    training loader's truth check (D3) stay where they are.

    The refusal is an `UnparseableFile` carrying the read's identity; its `what`
    names the rule, never the patient or the value, which only the message does.
    """
    def refuse(index: int, sample: Any, field: str, value: Any, what: str) -> None:
        raise UnparseableFile(
            f"{path}: sample {index} (patient {sample.patient_id!r}) has {field} "
            f"{value!r}, which is not {what}. Ids are refused rather than converted: "
            "a float or a boolean turned into an index names a different node",
            identity=identity,
            what="a samples file with integer ids",
        )

    def is_id(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    for index, sample in enumerate(samples):
        if not is_id(sample.disease_id):
            refuse(index, sample, "disease_id", sample.disease_id, "an integer id")
        lists = {"phenotype_ids": sample.phenotype_ids}
        for field in ("candidate_disease_ids", "gene_ids"):
            if getattr(sample, field) is not None:
                lists[field] = getattr(sample, field)
        for field, values in lists.items():
            if not isinstance(values, list):
                refuse(index, sample, field, values, "a list of integer ids")
            for value in values:
                if not is_id(value):
                    refuse(index, sample, field, value, "an integer id")


def _refuse_phenotype_repeats(path: Path, samples: List[Any], identity: ReadIdentity) -> None:
    """Every sample's `phenotype_ids` must be in phenotype normal form (N2); a file
    with a case that lists one phenotype twice is refused under the file's name
    and the patient's.

    **Checked, never normalised here.** Producers normalise and readers check
    (`docs/working/PLAN_PHENOTYPE_NORMALISATION.md` §6): the generator and the
    importer write each case's phenotypes once, and training and measurement
    record the digest of the bytes this read parsed. Removing a repeat on the way
    would score a list other than the one recorded. The rule is
    `normalise_phenotypes`, called rather than restated, so this check and the
    producers cannot disagree about what a repeat is. Only `phenotype_ids` is a
    phenotype list (§4.1): `gene_ids` and `candidate_disease_ids` are not checked.

    **After the integer check, never before it.** The rule compares identities in
    a set, where `True` and `1` are one entry, so a boolean id would be reported
    as a repeat instead of as the id that is not an integer.

    The refusal is an `UnparseableFile` carrying the read's identity, as the
    integer check's is, so R10 records such evidence as unreadable rather than
    refusing its run; its `what` names the rule, never the patient or the count,
    which only the message does.
    """
    for index, sample in enumerate(samples):
        repeats = normalise_phenotypes(sample.phenotype_ids).repeats_removed
        if repeats:
            raise UnparseableFile(
                f"{path}: sample {index} (patient {sample.patient_id!r}) is not in "
                f"phenotype normal form: its phenotype_ids repeat an id already listed "
                f"{repeats} {'time' if repeats == 1 else 'times'} (rule version "
                f"{PHENOTYPE_NORMALISATION_VERSION}: each phenotype once, first occurrence "
                "kept). A stored file is refused rather than normalised on read, because "
                "its digest is what a run records. Rebuild it with its producer: "
                "scripts/build_knowledge_graph.py --generate-samples for a workspace split, "
                "scripts/import_cohort.py for a test cohort version",
                identity=identity,
                what="a samples file in phenotype normal form",
            )
