"""
The readers' release check: no raw buffer reaches the next read or the caller.
==============================================================================
Contract M2.1's readers parse the bytes `read_once` returns and keep only what
they parsed and a `ReadIdentity`. That the raw buffer is then released is shown
here, on fixture-sized files on CPU, never inferred from resident memory: the
allocator may keep a freed buffer's pages resident, and live parsed objects may
pin them.

**When it checks.** On entry to each `read_once` call the reader makes, covering
the reads before it, and again after the reader returns. Each time:

- a `tracemalloc` snapshot holds no live allocation whose traceback runs through
  `read_once`'s read line. That catches the raw buffer whoever holds it: a local
  still bound inside a reader of several files (a loop that rebinds one name
  still holds the previous buffer during the next read), the return value, a
  closure or module state. `bytes` cannot be weakly referenced, so the buffer is
  found through its allocation;
- every `BytesIO` the reader built over an earlier read is gone. Each module
  named in `bytesio_modules` has its `BytesIO` replaced by a subclass that
  records a weak reference; a `BytesIO` that copied the bytes (as `getbuffer()`
  makes it do) no longer shares the buffer, so only this catches it.

**Its settings let it see the buffer.** Tracing starts before the reader is
called. The filter names `read_once`'s file and the line of its read, and matches
it in any frame (`all_frames=True`). `read_once` reads on a line that does
nothing else, so a correct reader's identities never match. Garbage collection is
off while the reader runs, so a buffer kept only by a reference cycle is seen
rather than collected by chance; `torch.load` over a `BytesIO` leaves no cycle.

**A torch file read without a tracked `BytesIO` fails** instead of passing
unseen, so a reader that builds its wrapper some other way is caught.

**What it does not see:** a copy the reader makes into another object, such as a
`bytearray` or a JSON file's decoded text, and torch's tensor storages. Those are
parsed results or copies, not the raw buffer; the readers' results are listed in
S10's readings, which explain peaks against them. **Nor does it see a buffer kept
between its parse and the reader's return:** it checks at the next read and after
the return, so a reader that keeps its last buffer through the rest of its own
work, and drops it before returning, passes. That the readers release the bytes
before parsing is their code (`del read`), not something this check shows. It is
a test, not a capacity reading, and it never runs inside a measured run.

Module: tests/fixtures/release.py
"""
from __future__ import annotations

import gc
import inspect
import io
import linecache
import tracemalloc
import weakref
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from src.utils import fingerprint
from tests.fixtures.replacement import hook_read_once

TORCH_SUFFIXES = (".pt",)


class ReleaseCheckError(AssertionError):
    """A reader kept a raw buffer, or a wrapper over one, past where it may."""


def read_line() -> tuple[str, int]:
    """`read_once`'s file and the line that reads the file, which does nothing else."""
    real = inspect.unwrap(fingerprint.read_once)
    lines, start = inspect.getsourcelines(real)
    hits = [i for i, line in enumerate(lines) if ".read()" in line]
    if len(hits) != 1 or lines[hits[0]].strip() != "data = handle.read()":
        raise ReleaseCheckError(
            "read_once must read the file on one line that does nothing else, "
            "or this check cannot tell the raw buffer from what a reader keeps"
        )
    return inspect.getsourcefile(real), start + hits[0]


def check_release(
    monkeypatch,
    call: Callable[[], Any],
    *,
    bytesio_modules: Sequence[ModuleType] = (),
    torch_suffixes: Sequence[str] = TORCH_SUFFIXES,
) -> tuple[Any, tuple[Path, ...]]:
    """Run `call()` (a reader) under the check. Returns its result and the paths read.

    Raises `ReleaseCheckError` if a raw buffer from an earlier read is alive at a
    later read or after the reader returns, if a tracked `BytesIO` outlives its
    read, if a torch file was read with no tracked `BytesIO`, or if the reader
    never called `read_once` at all.
    """
    filename, lineno = read_line()
    line_filter = tracemalloc.Filter(True, filename, lineno=lineno, all_frames=True)
    reads: list[Path] = []
    wrappers: list[tuple[int, weakref.ref]] = []

    class TrackedBytesIO(io.BytesIO):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            wrappers.append((len(reads) - 1, weakref.ref(self)))

    def verify(moment: str) -> None:
        snapshot = tracemalloc.take_snapshot().filter_traces([line_filter])
        stats = snapshot.statistics("traceback")
        if stats:
            worst = stats[0]
            where = "\n".join(
                f"  {frame.filename}:{frame.lineno}: "
                f"{linecache.getline(frame.filename, frame.lineno).strip()}"
                for frame in worst.traceback
            )
            raise ReleaseCheckError(
                f"{moment}: {sum(s.size for s in stats)} bytes allocated by "
                f"read_once's read are still alive (reads so far: "
                f"{[str(p) for p in reads]}). Largest block's traceback:\n{where}"
            )
        alive = [str(reads[index]) for index, ref in wrappers if ref() is not None]
        if alive:
            raise ReleaseCheckError(f"{moment}: a BytesIO over {alive} is still alive")

    def on_entry(path: Path) -> None:
        if reads:
            verify(f"before reading {path}")
        reads.append(path)

    with monkeypatch.context() as patch:
        for module in bytesio_modules:
            patch.setattr(module, "BytesIO", TrackedBytesIO)
        hook_read_once(patch, on_entry=on_entry)
        started = not tracemalloc.is_tracing()
        if started:
            tracemalloc.start(25)
        collecting = gc.isenabled()
        gc.disable()
        try:
            result = call()
            verify("after the reader returned")
        finally:
            if collecting:
                gc.enable()
            if started:
                tracemalloc.stop()

    if not reads:
        raise ReleaseCheckError("the reader never called read_once")
    wrapped = {index for index, _ in wrappers}
    unwrapped = [
        str(path) for index, path in enumerate(reads)
        if path.suffix in torch_suffixes and index not in wrapped
    ]
    if unwrapped:
        raise ReleaseCheckError(
            f"{unwrapped} were read with no tracked BytesIO, so whether a wrapper "
            "kept their bytes cannot be seen. Build it with the reader module's "
            "own BytesIO and name that module in bytesio_modules."
        )
    return result, tuple(reads)
