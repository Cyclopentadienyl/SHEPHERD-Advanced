"""
`read_once`: one open, one read, and the digest of exactly the bytes returned.
==============================================================================
Contract M2.1's primitive (`src/utils/fingerprint.py`). Every reader whose input a
record names will parse `FileRead.data` and keep `FileRead.identity`, so these
tests pin what the readers rely on:

- the digest is the digest of the bytes returned, not of the path read again;
- the file is opened once, through Python's io and os layers;
- a missing file raises; there is no `None`, unlike `file_sha256`;
- replacing the file afterwards, by atomic rename or in-place rewrite, changes
  neither the bytes nor the digest already returned (a lazy or memory-mapped
  read would fail the rewrite);
- a kept `ReadIdentity` holds nothing allocated by the read, found with
  `tracemalloc` filtered on the read's own line, which the readers' release check
  (contract M2.1, S2) depends on.

Module: tests/unit/test_read_once.py
"""
from __future__ import annotations

import gc
import hashlib
import inspect
import os
import tracemalloc
from pathlib import Path

import pytest

from src.utils import fingerprint
from src.utils.fingerprint import FileRead, ReadIdentity, read_once
from tests.fixtures.opens import count_opens

CONTENT = b"shepherd read once\n" * 4096


def _read_line() -> int:
    """The line of `read_once` that reads the file, which must do nothing else."""
    lines, start = inspect.getsourcelines(read_once)
    hits = [i for i, line in enumerate(lines) if ".read()" in line]
    assert len(hits) == 1, "read_once must read the file on exactly one line"
    assert lines[hits[0]].strip() == "data = handle.read()"
    return start + hits[0]


def test_the_digest_is_the_digest_of_the_bytes_returned(tmp_path):
    path = tmp_path / "input.bin"
    path.write_bytes(CONTENT)

    read = read_once(path)

    assert isinstance(read, FileRead)
    assert read.path == path
    assert read.data == CONTENT
    assert read.sha256 == hashlib.sha256(read.data).hexdigest()


def test_the_identity_carries_the_digest_and_no_bytes(tmp_path):
    path = tmp_path / "input.bin"
    path.write_bytes(CONTENT)

    identity = read_once(path).identity

    assert identity == ReadIdentity(path=path, sha256=hashlib.sha256(CONTENT).hexdigest())
    assert not any(isinstance(v, (bytes, bytearray, memoryview)) for v in vars(identity).values())


def test_the_file_is_opened_once(tmp_path):
    path = tmp_path / "input.bin"
    path.write_bytes(CONTENT)

    with count_opens() as opens:
        read_once(path)

    assert opens.opens(path) == 1


def test_a_missing_file_raises(tmp_path):
    """A file deleted between two reads is refused, never recorded as absent."""
    with pytest.raises(FileNotFoundError):
        read_once(tmp_path / "absent.bin")


@pytest.mark.parametrize("how", ["atomic_rename", "in_place_rewrite"])
def test_a_later_replacement_changes_neither_bytes_nor_digest(tmp_path, how):
    path = tmp_path / "input.bin"
    path.write_bytes(CONTENT)
    inode = path.stat().st_ino

    read = read_once(path)

    replacement = b"another file entirely\n" * 1024
    if how == "atomic_rename":
        staged = tmp_path / "staged.bin"
        staged.write_bytes(replacement)
        os.replace(staged, path)
        assert path.stat().st_ino != inode
    else:
        with path.open("r+b") as handle:
            handle.write(replacement)
            handle.truncate()
        assert path.stat().st_ino == inode
    assert path.read_bytes() == replacement

    assert read.data == CONTENT
    assert read.sha256 == hashlib.sha256(CONTENT).hexdigest()


def _live_bytes_from_the_read(path: Path, keep) -> int:
    """Bytes still allocated from `read_once`'s read line while the caller keeps
    `keep(read)`; the `FileRead` itself is dropped before the snapshot."""
    line_filter = tracemalloc.Filter(
        True, fingerprint.__file__, lineno=_read_line(), all_frames=True
    )
    tracemalloc.start(25)
    try:
        kept = keep(read_once(path))
        gc.collect()
        snapshot = tracemalloc.take_snapshot().filter_traces([line_filter])
        live = sum(stat.size for stat in snapshot.statistics("filename"))
        del kept
        return live
    finally:
        tracemalloc.stop()


def test_a_kept_identity_holds_no_allocation_from_the_read(tmp_path):
    """The positive control shows the filter finds the buffer; the identity holds none of it."""
    if tracemalloc.is_tracing():
        pytest.skip("tracemalloc is already tracing in this process")
    path = tmp_path / "input.bin"
    path.write_bytes(CONTENT)

    assert _live_bytes_from_the_read(path, lambda read: read) >= len(CONTENT)
    assert _live_bytes_from_the_read(path, lambda read: read.identity) == 0
