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
from typing import Any, NamedTuple

from src.utils.fingerprint import ReadIdentity

__all__ = ["CheckpointRead", "read_checkpoint", "write_checkpoint"]


class CheckpointRead(NamedTuple):
    """A loaded checkpoint, and the identity of the bytes it was loaded from."""

    checkpoint: Any
    identity: ReadIdentity


def read_checkpoint(
    path: str | Path, *, map_location: Any, weights_only: bool
) -> CheckpointRead:
    """Load a checkpoint from one read of `path`, with that read's identity.

    torch parses the bytes through this module's `BytesIO`, which the readers'
    release check replaces to see that no wrapper outlives the load. The raw
    buffer is released when this returns; the loaded checkpoint is the caller's.

    Raises:
        FileNotFoundError: if there is no file at `path`.
        Whatever `torch.load` raises for these bytes and options, unchanged.
    """
    import torch

    from src.utils.fingerprint import read_once

    read = read_once(path)
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
