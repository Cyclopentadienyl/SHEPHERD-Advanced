"""
Count how many times each file is opened through Python's io and os layers.
===========================================================================
The "read once" assertions of contract M2.1 count opens, not calls to one
function. A `sys.addaudithook` hook sees the `open` audit event that `open()`,
`Path.open`, `Path.read_text` and `os.open` all raise, and so `torch.load(path)`'s
own open and `json.load(open(path))` too. A reader cannot pass by opening its
file a second time through any of those.

**What it does not see:** an open made inside a C extension, such as torch's
`PyTorchFileReader` or `UntypedStorage.from_file` given a path, and an
`os.open(..., dir_fd=...)`, whose relative path is keyed against the working
directory. For torch readers, the release check's tracked-`BytesIO` rule is what
keeps them on the hashed buffer.

Audit hooks cannot be removed. One hook is installed for the process, the first
time a counter is used, and it returns at once unless a counter is active.

Module: tests/fixtures/opens.py
"""
from __future__ import annotations

import contextlib
import os
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

_active: list[OpenCounter] = []
_installed = False


def _key(path: str | bytes | os.PathLike) -> str:
    return os.path.realpath(os.fsdecode(path))


def _hook(event: str, args: tuple) -> None:
    if event != "open" or not _active:
        return
    target = args[0] if args else None
    if target is None or isinstance(target, int):
        return
    try:
        key = _key(target)
    except (TypeError, ValueError):
        return
    for counter in _active:
        counter.counts[key] += 1


class OpenCounter:
    """Opens seen while the counter was active, keyed by real path."""

    def __init__(self) -> None:
        self.counts: Counter = Counter()

    def opens(self, path: str | Path) -> int:
        return self.counts[_key(path)]


@contextlib.contextmanager
def count_opens() -> Iterator[OpenCounter]:
    """Count every open in the block. Nests: each active counter sees each open."""
    global _installed
    if not _installed:
        sys.addaudithook(_hook)
        _installed = True
    counter = OpenCounter()
    _active.append(counter)
    try:
        yield counter
    finally:
        _active.remove(counter)
