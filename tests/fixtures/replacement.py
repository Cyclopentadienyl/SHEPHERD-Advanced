"""
Replace an input between its read and everything after it.
==========================================================
Contract M2.1's acceptance runs a test double at each entry point: it replaces a
file right after a named read returns, once by atomic rename (a new inode) and
once by in-place rewrite (the same inode, which is how `torch.save` writes). The
run must then use and record the bytes it read, or refuse; it must never read A
and record B.

Both the double and the readers' release check (`tests/fixtures/release.py`)
work through `hook_read_once`, which wraps `read_once` wherever it is bound. The
hooks are given the path, never the `FileRead`, so neither they nor the wrapper
can hold a reader's buffer.

Module: tests/fixtures/replacement.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from src.utils import fingerprint

HOWS = ("rename", "rewrite")


def hook_read_once(
    monkeypatch,
    *,
    on_entry: Callable[[Path], None] | None = None,
    on_return: Callable[[Path], None] | None = None,
) -> None:
    """Wrap `read_once` in every module that has it bound, for this test.

    A reader may import the function at module level or inside its body; both
    see the wrapper, because every module whose `read_once` is the current one
    is patched and `src.utils.fingerprint` itself is among them. Hooks stack: a
    second call wraps the first wrapper.

    **Inert once the test's patches are undone.** A module imported for the
    first time while the hook is active, which binds `read_once` at module level,
    copies the wrapper, and monkeypatch never set that attribute so it cannot
    restore it. The wrapper therefore checks a flag that monkeypatch itself
    resets on undo, and from then on calls straight through, so a later test is
    never run through this test's hooks.
    """
    current = fingerprint.read_once
    state = SimpleNamespace(active=False)
    monkeypatch.setattr(state, "active", True)

    def wrapper(path):
        if not state.active:
            return current(path)
        path = Path(path)
        if on_entry is not None:
            on_entry(path)
        result = current(path)
        if on_return is not None:
            on_return(path)
        return result

    wrapper.__wrapped__ = current
    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", None)
        if namespace is not None and namespace.get("read_once") is current:
            monkeypatch.setattr(module, "read_once", wrapper)


def replace_file(target: Path, data: bytes, how: str) -> None:
    """Replace `target`'s content with `data`, by atomic rename or in place."""
    if how not in HOWS:
        raise ValueError(f"how must be one of {HOWS}, got {how!r}")
    inode = target.stat().st_ino
    if how == "rename":
        fd, staged = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(staged, target)
        assert target.stat().st_ino != inode, "a rename must leave a new inode"
    else:
        with target.open("r+b") as handle:
            handle.write(data)
            handle.truncate()
        assert target.stat().st_ino == inode, "a rewrite must keep the inode"


@dataclass
class Replacement:
    """What the double did: how many times it fired (at most once)."""

    after: Path
    target: Path
    fired: int = 0


def replace_after_read(
    monkeypatch,
    *,
    after: Path,
    target: Path,
    data: bytes,
    how: str,
    republish: str | None = None,
) -> Replacement:
    """Replace `target` with `data` once, right after the first read of `after`.

    `republish` names the manifest role that binds `target`. When it is given,
    the workspace's manifest beside `target` is rewritten in the same moment, the
    same way, to bind the replacement's digest: the publication of a new file
    together with a manifest that names it.

    Returns a record whose `fired` count a test asserts, so a test cannot pass
    because the named read never happened.
    """
    from src.kg.artifacts import MANIFEST_FILENAME

    if how not in HOWS:
        raise ValueError(f"how must be one of {HOWS}, got {how!r}")
    record = Replacement(after=Path(after), target=Path(target))
    after_key = os.path.realpath(after)

    def on_return(path: Path) -> None:
        if record.fired or os.path.realpath(path) != after_key:
            return
        record.fired += 1
        replace_file(record.target, data, how)
        if republish is not None:
            manifest_path = record.target.parent / MANIFEST_FILENAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"][republish] = hashlib.sha256(data).hexdigest()
            replace_file(
                manifest_path,
                json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
                how,
            )

    hook_read_once(monkeypatch, on_return=on_return)
    return record
