"""
Path hashes that lie.
=====================
Contract M2.1's dynamic pin (S9), used at each entry point and at the scripts S9
moved: run it with every path hash returning a poison digest, then assert the run
completes and no poison reaches its record. A record built only from the
readers' results passes; one that took a digest by hashing a path fails.

**The poison is put on `hashlib.file_digest`,** which `fingerprint.file_sha256`
calls, so it also reaches a `file_sha256` bound at import in another module, and
any direct `hashlib.file_digest` of a path. A path hashed another way (reading
the file and calling `hashlib.sha256` on it) is not caught here; that is the
static pin's (`tests/unit/test_provenance_pins.py`). The path-form verifiers this
fixture used to make fail are gone since S9, and the static pin pins their
absence by name.

`true_digest_for` names files whose path hash must stay true: the `kg.json`
identification check (`check_unparsed_kg_json`, M2.1 decision 1) hashes the file
by path on purpose, records nothing, and would otherwise refuse the run.

Module: tests/fixtures/path_forms.py
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

POISON = "0" * 64


class _Poisoned:
    """What a lying `hashlib.file_digest` returns: a digest object whose hex
    form is `POISON`."""

    def hexdigest(self) -> str:
        return POISON

    def digest(self) -> bytes:
        return bytes.fromhex(POISON)


def poison_path_forms(monkeypatch, *, true_digest_for: Iterable[Path] = ()) -> None:
    """Make every path hash return `POISON`, for the rest of the test, except
    for the files in `true_digest_for`."""
    import hashlib

    real_file_digest = hashlib.file_digest
    keep = {os.path.realpath(path) for path in true_digest_for}

    def file_digest(fileobj, digest, /, **kwargs):
        name = getattr(fileobj, "name", None)
        if isinstance(name, (str, bytes, os.PathLike)) and os.path.realpath(name) in keep:
            return real_file_digest(fileobj, digest, **kwargs)
        return _Poisoned()

    monkeypatch.setattr(hashlib, "file_digest", file_digest)
