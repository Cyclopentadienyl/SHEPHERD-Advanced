"""
Data Fingerprint Utilities
===========================
Compute the structural fingerprint a training run records for its graph
data, and compare a checkpoint's training record with the digests a consumer
holds (`verify_training_record`).

Module: src/utils/fingerprint.py

**Two identities live here, and they answer different questions.**

A *fingerprint* captures the structural identity of a graph dataset:
  - Node types and their counts
  - Edge types (including reverse edges)
  - Feature dimensions per node type
  - Total KG node/edge counts

It is a recorded structural summary of the graph a run read, compared by nothing.
It cannot answer "which exact inputs produced this checkpoint" — two runs over
different sample files share it, and sample files do not enter it at all.

*Content digests* (`read_once`, `file_sha256`) answer the second question. They are
kept separate rather than folded into the fingerprint, because
a structural identity that also changed whenever a byte moved would stop being a
compatibility check.

`read_once` is the content digest of an input that is about to be parsed: it
returns the bytes and the digest of those same bytes, so what a reader parses and
what a record names cannot come from two different reads (contract M2.1).
`file_sha256` identifies a file nobody is about to parse. Within the static
pin's scope (`src/**`, the entry points, the scripts M2.1's S9 moved and M4's
`import_cohort.py` and `record_evaluation.py`), every remaining caller is a
named exception in `tests/unit/test_provenance_pins.py`. Since contract M4 the
evaluation ledger and the report a record is derived from are each parsed and
identified from one read (BACKLOG item 18); the ledger's stale-write guard still
compares the file by path just before replacing it, and the ledger stays
single-writer. BACKLOG item 19's other scripts outside that scope still hash by
path, each with its home there.

`compute_fingerprint` is the structural summary a run records
(`data_fingerprint` in its checkpoint), compared by nothing at load.

**Identity is `verify_training_record`'s** (contract M2.4): the one comparator
between a checkpoint's `training_input_digests` and the digests a consumer holds
for the same roles. It raises; it does not warn.
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def stat_identity(st: os.stat_result) -> Tuple[int, ...]:
    """What tells one state of a file from another without reading it: its
    device, inode, size, and modification and change times (the change time
    moves with any write or metadata change and cannot be set back)."""
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


@dataclass(frozen=True)
class ReadIdentity:
    """Which file was read, and the SHA-256 of the bytes that read returned.

    It carries no bytes, so a reader can keep it beside what it parsed and let the
    raw buffer go. `opened` is the `stat_identity` of the file the read's one
    open opened, taken from that open descriptor once the bytes were read (never
    by looking the name up again), or None where no read made it; it names a
    state of the file, not the bytes, so two reads compare by path and digest.
    """

    path: Path
    sha256: str
    opened: Optional[Tuple[int, ...]] = field(default=None, compare=False)


@dataclass(frozen=True)
class FileRead:
    """One read of one file: its bytes, and the SHA-256 of exactly those bytes.

    A reader parses `data`, keeps `identity`, and drops this object, so the buffer
    is released once the parse is done.
    """

    path: Path
    data: bytes
    sha256: str
    #: The file this read opened (`ReadIdentity.opened`).
    opened: Optional[Tuple[int, ...]] = None

    @property
    def identity(self) -> ReadIdentity:
        return ReadIdentity(path=self.path, sha256=self.sha256, opened=self.opened)


def lies_within(path: Path, within: Path) -> bool:
    """Whether `path` lies inside the directory `within`, both already resolved:
    the one workspace boundary rule, for a name (`checkpoint_paths`) and for an
    opened file (`read_once(within=)`) alike. `within` itself is not inside."""
    return path != within and within in path.parents


class ReadOutside(ValueError):
    """A bounded read (`read_once(within=)`) opened a file that does not lie
    inside `within`, or one it cannot place; raised before a byte is read.

    It carries the facts: ``path`` (the name opened), ``within``, ``opened``
    (where the open descriptor says the file lies, or None) and ``why`` (set
    when ``opened`` is None). Its message states only them; each caller words
    its own refusal."""

    def __init__(self, path: Union[str, Path], within: Path, opened: Optional[Path],
                 why: Optional[str] = None) -> None:
        self.path = Path(path)
        self.within = Path(within)
        self.opened = opened
        self.why = why
        if opened is not None:
            message = f"{self.path} was opened at {opened}, outside {self.within}."
        else:
            message = f"{self.path} cannot be placed inside {self.within}: {why}."
        super().__init__(message)


def _opened_path(fd: int) -> Tuple[Optional[Path], Optional[str]]:
    """Where the file open on `fd` lies, asked of the descriptor itself: no
    name is looked up, so a name swapped before or after the open cannot
    answer for the file. ``(path, None)``, or ``(None, why)`` when there is
    no location.

    The location is taken first and the link count after it, so a file whose
    last name went away before or during the query is refused, never placed
    by the name it used to have. An `OSError` from the query (no `/proc`, say)
    reaches the caller, which refuses."""
    if sys.platform.startswith("linux"):
        opened: Optional[Path] = Path(os.readlink(f"/proc/self/fd/{fd}"))
    elif sys.platform == "win32":  # pragma: no cover - not exercised on Linux
        import ctypes
        import msvcrt
        from ctypes import wintypes

        final_path = ctypes.WinDLL("kernel32", use_last_error=True).GetFinalPathNameByHandleW
        final_path.argtypes = (wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD,
                               wintypes.DWORD)
        final_path.restype = wintypes.DWORD
        handle, size = msvcrt.get_osfhandle(fd), 1024
        while True:
            buffer = ctypes.create_unicode_buffer(size)
            # FILE_NAME_NORMALIZED | VOLUME_NAME_DOS
            needed = final_path(handle, buffer, size, 0)
            if needed == 0:
                return None, "the descriptor query failed"
            if needed < size:
                break
            size = needed + 1
        text = buffer.value
        # As `ntpath.realpath` does, so the result compares with a resolved path.
        if text.startswith("\\\\?\\UNC\\"):
            text = "\\\\" + text[len("\\\\?\\UNC\\"):]
        elif text.startswith("\\\\?\\"):
            text = text[len("\\\\?\\"):]
        opened = Path(text)
    elif sys.platform == "darwin":  # pragma: no cover - not exercised on Linux
        import fcntl

        raw = fcntl.fcntl(fd, fcntl.F_GETPATH, bytes(1024))
        opened = Path(os.fsdecode(raw.split(b"\0", 1)[0]))
    else:  # pragma: no cover - not exercised on Linux
        return None, "this platform gives no location for an open file"
    if os.fstat(fd).st_nlink == 0:
        return None, "the file opened has no name left"
    return opened, None


def _refuse_outside(handle: Any, path: Path, within: Path) -> None:
    """Raise `ReadOutside` unless the file open on `handle` lies inside
    `within` (`lies_within`), by its descriptor's location (`_opened_path`).
    No location, the query failing included, is a refusal."""
    try:
        opened, why = _opened_path(handle.fileno())
    except OSError as exc:
        opened, why = None, f"the descriptor query failed ({type(exc).__name__})"
    if opened is None:
        raise ReadOutside(path, within, None, why)
    if not lies_within(opened, within):
        raise ReadOutside(path, within, opened)


def read_once(path: Union[str, Path], *, within: Optional[Path] = None) -> FileRead:
    """Read a file once and return its bytes with their SHA-256.

    **One open, one full read, and the digest is taken over that same bytes
    object.** Hashing a path and then loading it reads the file twice, and a file
    replaced in between is parsed as one thing and recorded as another. Hashing an
    open handle and then parsing from it survives an atomic rename but not an
    in-place rewrite, which is how `torch.save` writes; hashing the buffer that is
    parsed survives both.

    **A missing file raises** (`FileNotFoundError`), unlike `file_sha256`. A file
    deleted between two reads is then refused instead of recorded as absent.

    **`within` bounds the read to a directory** (already resolved): the file
    this one open opened must lie inside it, or `ReadOutside` is raised before a
    byte is read. Where it lies is asked of the open descriptor, never of the
    name (`_opened_path`: `/proc/self/fd` on Linux, `GetFinalPathNameByHandleW`
    on Windows, `F_GETPATH` on macOS), because a name can point elsewhere at
    every lookup; a check of the name before or after the open leaves the open
    itself to land outside. So a file opened outside is refused whatever its
    name says afterwards, and a file opened inside is read even if its name
    moves out afterwards: its bytes are inside. Where no location can be had
    (another platform, no `/proc`, a file with no name left), the bounded read
    refuses. Without `within` nothing is checked, as before; automatic
    selection and the run listing pass it (`read_checkpoint(within=)`).

    **The file read is named by its open descriptor** (`FileRead.opened`, its
    `stat_identity` from `os.fstat` once the bytes are read), so a caller that
    keeps what a read found by the file's identity compares the file this read
    actually opened, never what its name pointed at before or after the open.

    No retry, no fallback. The read is alone on its line: the readers' release
    check (contract M2.1, S2) finds the raw buffer by the allocation that line
    makes, so nothing a reader keeps may be made on it.
    """
    path = Path(path)
    with path.open("rb") as handle:
        if within is not None:
            _refuse_outside(handle, path, within)
        data = handle.read()
        opened = stat_identity(os.fstat(handle.fileno()))
    sha256 = hashlib.sha256(data).hexdigest()
    return FileRead(path=path, data=data, sha256=sha256, opened=opened)


def file_sha256(path: Union[str, Path]) -> Optional[str]:
    """Raw content digest, or ``None`` if the file is not there.

    `hashlib.file_digest` (stdlib, 3.11+) reads in chunks, so a multi-gigabyte
    checkpoint is not loaded into memory to be identified. This hashes **bytes**
    and nothing else: no canonical form, no key ordering, no serialisation policy.
    Two runs quoting the same digest consumed the same file, which is the entire
    claim being made.

    **Lives here rather than in a script.** It was defined in
    `scripts/measure_scorer.py` and imported from there by other scripts and a
    test — scripts importing a script, which makes an entry point into a library.
    `src.utils` is the bottom layer, so every caller can reach this without
    reaching through anything else.

    **Absent is `None`, not an error.** A role a run did not consume, and a file
    that is genuinely missing, are both recorded rather than raised on; the caller
    decides what an absent input means. Note that the digest is taken at a
    different instant from any load, so a file changed in between would be recorded
    incorrectly — a limitation, not something locking is added for.
    """
    path = Path(path)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class DigestingFile:
    """A binary file being written, and the SHA-256 of every byte written to it.

    **The writer's side of `read_once`** (contract M3b; BACKLOG 19.16, 19.17).
    A writer that hashes its file by path after writing it records whatever the
    path holds by then; one that hashes what it serialises records the bytes it
    wrote, which is what a manifest binding claims. The serialiser writes
    through this object, and each piece is hashed as it passes, so nothing is
    held in memory to be hashed later.

    `torch.save` and the JSON writers need only `write` and `flush`.
    """

    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle
        self._digest = hashlib.sha256()

    def write(self, data: Any) -> int:
        self._digest.update(data)
        return self._handle.write(data)

    def flush(self) -> None:
        self._handle.flush()

    @property
    def sha256(self) -> str:
        """The digest of everything written so far."""
        return self._digest.hexdigest()


def write_digested(path: Union[str, Path], serialise: Callable[[DigestingFile], None]) -> str:
    """Write `path` through `serialise` and return the SHA-256 of exactly the
    bytes it wrote (contract M3b).

    Opened for binary writing, so no newline translation stands between what
    was hashed and what is on disk. In place, as the graph writers always wrote:
    the workspace writer refuses before its first byte, not by replacing files
    atomically.
    """
    path = Path(path)
    with path.open("wb") as handle:
        sink = DigestingFile(handle)
        serialise(sink)
    return sink.sha256



class TrainingRecordMismatch(ValueError):
    """A checkpoint's training record does not name the inputs it is compared
    with, or records none."""


def verify_training_record(
    checkpoint: Mapping[str, Any],
    current: Mapping[str, str],
    roles: Iterable[str],
    *,
    recorded_by: str,
    compared_with: str,
) -> None:
    """Refuse unless `checkpoint` records, for every role in `roles`, the digest
    `current` holds for it (contract M2.4).

    **The one model-to-graph comparator.** Serving and measurement compare the
    four graph roles with the workspace they verified, and the deployment probe
    asserts through this function; resume compares its parent with the run over
    the graph and data roles (M2.3). The set of roles is the argument and the
    rule is the same for every caller.

    **Present on both sides and equal, or refused.** A checkpoint with no
    `training_input_digests` (one from before the contract), a role it does not
    record, a role `current` does not hold and a different digest are each
    refused by name, so an empty record cannot pass vacuously. `current` comes
    from the caller's reads; nothing here opens a file.

    Args:
        checkpoint: the checkpoint as read.
        current: role -> SHA-256 the consumer holds for the inputs in front of it.
        roles: the roles compared; must not be empty.
        recorded_by: names the checkpoint in a refusal ("checkpoint <path>").
        compared_with: names `current` in a refusal ("the workspace <path>").

    Raises:
        TrainingRecordMismatch: any role is missing on either side or differs.
    """
    roles = tuple(roles)
    if not roles:
        raise ValueError("no roles to compare; an empty comparison passes anything")
    record = (
        checkpoint.get("training_input_digests")
        if isinstance(checkpoint, Mapping) else None
    )
    if not isinstance(record, Mapping) or not record:
        raise TrainingRecordMismatch(
            f"{recorded_by} records no training_input_digests, so the inputs it was "
            f"trained on cannot be compared with {compared_with}. A checkpoint from "
            "before the provenance contract is not used; retrain it."
        )
    problems: List[str] = []
    for role in roles:
        recorded, held = record.get(role), current.get(role)
        if not recorded:
            problems.append(f"{role} is not recorded")
        elif not held:
            problems.append(f"{role} is recorded and {compared_with} has none")
        elif recorded != held:
            problems.append(
                f"{role} differs (recorded {str(recorded)[:12]}..., "
                f"{compared_with} has {str(held)[:12]}...)"
            )
    if problems:
        raise TrainingRecordMismatch(
            f"{recorded_by} does not match {compared_with}: "
            + "; ".join(problems)
            + ". Its record names the bytes it was trained on, and these are not "
            "they."
        )


def compute_fingerprint(
    graph_data: Dict[str, Any],
    kg_total_nodes: Optional[int] = None,
    kg_total_edges: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute a structural fingerprint from graph data tensors.

    Args:
        graph_data: dict with keys "x_dict", "edge_index_dict", "num_nodes_dict"
        kg_total_nodes: total KG node count (optional, for extra validation)
        kg_total_edges: total KG edge count (optional, for extra validation)

    Returns:
        Fingerprint dict suitable for embedding in a checkpoint.
    """
    x_dict = graph_data.get("x_dict", {})
    edge_index_dict = graph_data.get("edge_index_dict", {})
    num_nodes_dict = graph_data.get("num_nodes_dict", {})

    node_types = sorted(x_dict.keys())

    feature_dims = {}
    for nt, tensor in x_dict.items():
        if hasattr(tensor, "size") and tensor.dim() >= 2:
            feature_dims[nt] = tensor.size(-1)

    edge_types = sorted(
        [list(k) if isinstance(k, tuple) else k for k in edge_index_dict.keys()],
        key=str,
    )

    node_counts = {k: int(v) for k, v in num_nodes_dict.items()}

    fp: Dict[str, Any] = {
        "node_types": node_types,
        "node_counts": node_counts,
        "feature_dims": feature_dims,
        "edge_types": edge_types,
        "num_edge_types": len(edge_types),
    }

    if kg_total_nodes is not None:
        fp["kg_total_nodes"] = kg_total_nodes
    if kg_total_edges is not None:
        fp["kg_total_edges"] = kg_total_edges

    return fp
