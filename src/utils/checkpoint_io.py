"""
Reading a checkpoint once, with the identity of the bytes loaded.
=================================================================
Contract M2.1: a checkpoint whose file a record or a check names is parsed from
the bytes `read_once` returned, so the digest used is that of the checkpoint
actually loaded. Loading by path and hashing by path read the file twice, and
`torch.save` rewrites a file in place, so the two could describe different
checkpoints.

**Not in `checkpoint_paths.py`,** which promises no torch so that the API can
import it. torch is imported here only when a checkpoint is read or written.

**One writer** (`write_checkpoint`, contract M3a-1): the payload is serialised
to bytes once, its SHA-256 taken from those bytes, and the file staged and
renamed into place, so a reader never sees a half-written checkpoint and the
digest is that of the bytes written, not of a later read of the path.

**Each site keeps its own loading options.** `map_location` and `weights_only`
have no defaults, so a caller states the ones it already uses instead of
inheriting one it did not choose. Nothing here retries, falls back to another
setting, or handles devices.

Module: src/utils/checkpoint_io.py
"""
from __future__ import annotations

import hashlib
import os
from io import BytesIO
from pathlib import Path
from typing import Any, NamedTuple, Optional

from src.utils.fingerprint import ReadIdentity

#: The checkpoint record's schema (contract M3a). 1 is M3a's first. Written by
#: `src.training.checkpoint_payload`; read by `src.utils.run_plan` to decide
#: whether a checkpoint holds a state a run can continue from.
CHECKPOINT_RECORD_VERSION = 1

#: **The training rules a model bundle's training-state checkpoint was written
#: under** (its `training_rules`), read by `src.utils.run_plan` before a run
#: continues one: a resume, an extension or a branch keeps its parent's rules,
#: so a parent written under other rules is refused, never carried across
#: (the reviewer's K5 review, P2-1). Rules 1 are K5's: early stopping is on
#: exactly when a patience is in effect, and the base validates on the pairs its
#: split fixes (seeded by the split's seed, in fixed batches), whatever a run's
#: seed and batch size. Raise it when a rule a continued run depends on changes.
TRAINING_RULES_VERSION = 1

__all__ = ["CheckpointChanged", "CheckpointRead", "TRAINING_RULES_VERSION", "read_checkpoint",
           "write_checkpoint"]


class CheckpointRead(NamedTuple):
    """A loaded checkpoint, and the identity of the bytes it was loaded from."""

    checkpoint: Any
    identity: ReadIdentity


class CheckpointChanged(ValueError):
    """The file is not the checkpoint a caller expects: `read_checkpoint` was
    given its SHA-256 (``expected_sha256``) and its one read has other bytes
    (``found``), or finds no file (``found`` is None); raised before anything
    is unpickled (root-cause round, P22-S1). A ``ValueError``, like the
    reader's other refusals.

    It carries the facts (``path``, ``expected``, ``found``) and its message
    states only them. What the difference means and what to do depend on
    where the expectation came from (a parent the operator chose from a list
    or by a typed path, a file a parent's record names), so each caller words
    its own refusal from these fields."""

    def __init__(self, path: str | Path, expected: str, found: Optional[str]) -> None:
        self.path = Path(path)
        self.expected = expected
        self.found = found
        held = (f"holds {found[:12]}..., not the {expected[:12]}... expected"
                if found is not None else f"is gone; {expected[:12]}... was expected")
        super().__init__(f"{self.path} {held}.")


def read_checkpoint(
    path: str | Path, *, map_location: Any, weights_only: bool,
    expected_sha256: Optional[str] = None,
) -> CheckpointRead:
    """Load a checkpoint from one read of `path`, with that read's identity.

    torch parses the bytes through this module's `BytesIO`, which the readers'
    release check replaces to see that no wrapper outlives the load. The raw
    buffer is released when this returns; the loaded checkpoint is the caller's.

    **`expected_sha256` compares before it loads** (the checkpoint a caller
    expects): other bytes, a valid checkpoint or not, raise `CheckpointChanged`
    and are never unpickled; a missing file raises the same, since the bytes
    expected are no longer there. As `read_split_manifest` and `read_report` do.

    Raises:
        FileNotFoundError: if there is no file at `path` and nothing is expected.
        CheckpointChanged: if `expected_sha256` is given and the read is not it.
        Whatever `torch.load` raises for these bytes and options, unchanged.
    """
    import torch

    from src.utils.fingerprint import read_once

    try:
        read = read_once(path)
    except FileNotFoundError:
        if expected_sha256 is None:
            raise
        raise CheckpointChanged(path, expected_sha256, None) from None
    if expected_sha256 is not None and read.sha256 != expected_sha256:
        found = read.sha256
        del read
        raise CheckpointChanged(path, expected_sha256, found)
    return CheckpointRead(
        torch.load(BytesIO(read.data), map_location=map_location, weights_only=weights_only),
        read.identity,
    )


def write_checkpoint(payload: Any, path: str | Path) -> str:
    """Write `payload` to `path` whole, and return the SHA-256 of the bytes written.

    Serialised once to memory; the bytes are written to a staging file in the
    same directory (`create_staging_file`: an unused name, the target's existing
    mode kept), flushed to disk and renamed over `path`. A failure leaves `path`
    as it was and removes the staging file.
    """
    import torch

    from src.utils.file_modes import create_staging_file

    buffer = BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    digest = hashlib.sha256(data).hexdigest()

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, staged = create_staging_file(target, binary=True)
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, target)
    except BaseException:
        Path(staged).unlink(missing_ok=True)
        raise
    return digest
