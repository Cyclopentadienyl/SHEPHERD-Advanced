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
  (contract M2.1, S2) depends on;
- a bounded read (`within=`, automatic selection's) judges the file its one open
  opened, by the descriptor: a file opened outside is never read, whatever its name
  says afterwards or to any lookup; a file opened inside is read even when its name
  moves out afterwards; a file it cannot place is refused (K8-4r, P2-1).

Module: tests/unit/test_read_once.py
"""
from __future__ import annotations

import gc
import hashlib
import inspect
import os
import pathlib
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("bounded", [False, True], ids=["unbounded", "bounded"])
def test_the_file_is_opened_once(tmp_path, bounded):
    """Asking the descriptor where the file lies opens nothing more."""
    path = tmp_path / "input.bin"
    path.write_bytes(CONTENT)

    with count_opens() as opens:
        read = read_once(path, within=tmp_path.resolve()) if bounded else read_once(path)

    assert opens.opens(path) == 1
    assert read.data == CONTENT


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


# ---------------------------------------------------------------------------
# A bounded read judges the file it opened, never its name (K8-4r, P2-1)
# ---------------------------------------------------------------------------
class _CountedHandle:
    """The real handle, its reads counted: `read_once` reads through this."""

    def __init__(self, real):
        self._real = real
        self.reads = 0

    def fileno(self):
        return self._real.fileno()

    def read(self, *args):
        self.reads += 1
        return self._real.read(*args)

    def close(self):
        self._real.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._real.close()
        return False


def _hook_open(monkeypatch, name, after_open):
    """Wrap `Path.open`: right after the real open of `name` returns, call
    `after_open()` and hand back the handle with its reads counted. Returns
    what it saw (`fired`, `handles`), so a test cannot pass because the open
    it hooks never happened."""
    real_open = pathlib.Path.open
    seen = SimpleNamespace(fired=0, handles=[])

    def opening(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        if os.fspath(self) != os.fspath(name):
            return handle
        seen.fired += 1
        after_open()
        counted = _CountedHandle(handle)
        seen.handles.append(counted)
        return counted

    monkeypatch.setattr(pathlib.Path, "open", opening)
    return seen


def _reads(seen) -> int:
    return sum(handle.reads for handle in seen.handles)


def _hook_name_lookups(monkeypatch, after):
    """Wrap `Path.resolve`, `os.path.realpath` and `os.stat`: each calls
    `after()` once it returns, so any lookup of a name an implementation makes
    after the open is answered and then followed by a swap."""
    real_resolve, real_realpath, real_stat = pathlib.Path.resolve, os.path.realpath, os.stat

    def resolving(self, *args, **kwargs):
        try:
            return real_resolve(self, *args, **kwargs)
        finally:
            after()

    def realpath(*args, **kwargs):
        try:
            return real_realpath(*args, **kwargs)
        finally:
            after()

    def stat(*args, **kwargs):
        try:
            return real_stat(*args, **kwargs)
        finally:
            after()

    monkeypatch.setattr(pathlib.Path, "resolve", resolving)
    monkeypatch.setattr(os.path, "realpath", realpath)
    monkeypatch.setattr(os, "stat", stat)


class _Swap:
    """Puts an inside regular file under a name that is a link out of the
    workspace, and the link back, by `os.unlink`, `os.symlink`, `os.rename`
    and `os.link` only (none of them wrapped), behind a re-entry guard.

    ``final``: the name is a link to an outside file, and its inside form is
    a hard link of a file prepared inside. ``directory``: the name's run
    directory is a link to an outside directory, and its inside form is a real
    directory renamed into its place."""

    def __init__(self, how, ws, out):
        self.how, self.busy, self.inside = how, False, False
        if how == "final":
            self.outside_file = out / "last.pt"
            self.outside_file.write_bytes(b"B outside")
            self.prepared = ws / "prepared.pt"
            self.prepared.write_bytes(b"A inside")
            self.name = ws / "last.pt"
            os.symlink(self.outside_file, self.name)
        else:
            outside_run = out / "run1"
            outside_run.mkdir()
            self.outside_file = outside_run / "last.pt"
            self.outside_file.write_bytes(b"D outside")
            self.outside_run, self.link = outside_run, ws / "run1"
            self.inside_run = ws / "run1-inside"
            self.inside_run.mkdir()
            (self.inside_run / "last.pt").write_bytes(b"A inside")
            self.name = self.link / "last.pt"
            os.symlink(outside_run, self.link, target_is_directory=True)

    def to_inside(self):
        if self.busy or self.inside:
            return
        self.busy = True
        try:
            if self.how == "final":
                os.unlink(self.name)
                os.link(self.prepared, self.name)
            else:
                os.unlink(self.link)
                os.rename(self.inside_run, self.link)
            self.inside = True
        finally:
            self.busy = False

    def to_outside(self):
        if self.busy or not self.inside:
            return
        self.busy = True
        try:
            if self.how == "final":
                os.unlink(self.name)
                os.symlink(self.outside_file, self.name)
            else:
                os.rename(self.link, self.inside_run)
                os.symlink(self.outside_run, self.link, target_is_directory=True)
            assert os.path.lexists(self.name)
            self.inside = False
        finally:
            self.busy = False


@pytest.mark.parametrize("how", ["final", "directory"])
def test_a_bound_read_never_reads_a_file_opened_outside_whatever_its_name_says_afterwards(
        tmp_path, monkeypatch, how):
    """The reviewer's double swap (K8-4r map review, race.py): the name is a link
    out of the workspace when it is opened; right after the open it becomes an
    inside regular file, and after every later lookup of a name (`resolve`,
    `realpath`, `stat`) it is put back. A check that asks the name, before or
    after the open, can be answered by an inside file while the bytes come from
    the outside one. The read asks the descriptor, so it refuses, naming the
    outside file it opened, and reads not one byte of it."""
    ws = (tmp_path / "ws").resolve()
    ws.mkdir()
    out = (tmp_path / "out").resolve()
    out.mkdir()
    swap = _Swap(how, ws, out)
    expected_opened = Path(os.path.realpath(swap.outside_file))
    seen = _hook_open(monkeypatch, swap.name, swap.to_inside)
    _hook_name_lookups(monkeypatch, swap.to_outside)

    with pytest.raises(fingerprint.ReadOutside) as refused:
        read_once(swap.name, within=ws)

    assert seen.fired == 1, "the open the swap hangs on happened"
    assert _reads(seen) == 0, "not one byte of the outside file was read"
    assert refused.value.opened == expected_opened
    assert refused.value.why is None and refused.value.within == ws
    assert str(expected_opened) in str(refused.value)


def test_a_bound_read_reads_a_file_opened_inside_even_when_its_name_moves_out_afterwards(
        tmp_path, monkeypatch):
    """What the bound means: the file the bytes come from. The name is a link
    to an inside file when it is opened, and a link out of the workspace by
    the time the read is checked: the inside file is read, with the digest of
    its bytes. A link inside the workspace to an inside file, left alone,
    reads too (the positive control)."""
    ws = (tmp_path / "ws").resolve()
    ws.mkdir()
    outside = tmp_path / "out.pt"
    outside.write_bytes(b"B outside")
    target = ws / "runs" / "a.pt"
    target.parent.mkdir()
    target.write_bytes(b"A inside")
    name = ws / "last.pt"
    name.symlink_to(target)

    def name_moves_out():
        os.unlink(name)
        os.symlink(outside, name)

    seen = _hook_open(monkeypatch, name, name_moves_out)
    read = read_once(name, within=ws)

    assert seen.fired == 1 and os.readlink(name) == str(outside)
    assert read.data == b"A inside"
    assert read.sha256 == hashlib.sha256(b"A inside").hexdigest()

    link = ws / "link.pt"
    link.symlink_to(target)
    assert read_once(link, within=ws).data == b"A inside"


@pytest.mark.parametrize("case", ["no location", "no name left"])
def test_a_bound_read_refuses_a_file_it_cannot_place(tmp_path, monkeypatch, case):
    """Fail closed: where the descriptor gives no location (the platform
    dispatch's fallback, tested without a second platform), or the file opened
    has no name left, the bounded read refuses before a byte is read."""
    ws = (tmp_path / "ws").resolve()
    ws.mkdir()
    name = ws / "last.pt"
    name.write_bytes(b"A inside")
    if case == "no location":
        why = "no location in this test"
        monkeypatch.setattr(fingerprint, "_opened_path", lambda fd: (None, why))
        seen = _hook_open(monkeypatch, name, lambda: None)
    else:
        why = "the file opened has no name left"
        seen = _hook_open(monkeypatch, name, lambda: os.unlink(name))

    with pytest.raises(fingerprint.ReadOutside) as refused:
        read_once(name, within=ws)

    assert seen.fired == 1 and _reads(seen) == 0
    assert refused.value.opened is None and refused.value.why == why
    assert why in str(refused.value)


def test_an_unbounded_read_is_unchanged_by_where_the_file_lies(tmp_path):
    """Without `within` nothing is asked: a link out of any directory reads as
    before (explicit paths, training, measurement, lists)."""
    outside = tmp_path / "out.pt"
    outside.write_bytes(b"B outside")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "last.pt").symlink_to(outside)

    assert read_once(ws / "last.pt").data == b"B outside"
    with pytest.raises(fingerprint.ReadOutside):
        read_once(ws / "last.pt", within=ws.resolve())
