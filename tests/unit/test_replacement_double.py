"""
The replacement double does what contract M2.1's acceptance relies on.
======================================================================
`tests/fixtures/replacement.py` replaces a file right after a named read, by
atomic rename or in-place rewrite, and can republish the manifest with it. The
entry points' acceptance tests (S3, S6-S8) mean something only if the double
fires exactly where it is told to, once, in the way it is told to, and is seen by
a reader however that reader imports `read_once`.

Module: tests/unit/test_replacement_double.py
"""
from __future__ import annotations

import hashlib
import json

import pytest

from src.utils import fingerprint
from src.utils.fingerprint import read_once
from tests.fixtures.replacement import replace_after_read


@pytest.fixture
def two_files(tmp_path):
    named = tmp_path / "named.bin"
    target = tmp_path / "target.bin"
    named.write_bytes(b"named file")
    target.write_bytes(b"original target")
    return named, target


@pytest.mark.parametrize("how,same_inode", [("rename", False), ("rewrite", True)])
def test_it_replaces_the_target_right_after_the_named_read(monkeypatch, two_files, how, same_inode):
    named, target = two_files
    inode = target.stat().st_ino
    record = replace_after_read(monkeypatch, after=named, target=target,
                                data=b"replacement", how=how)

    assert record.fired == 0
    read_once(target)  # not the named read
    assert record.fired == 0 and target.read_bytes() == b"original target"

    read = read_once(named)

    assert read.data == b"named file"
    assert record.fired == 1
    assert target.read_bytes() == b"replacement"
    assert (target.stat().st_ino == inode) is same_inode


def test_it_fires_once(monkeypatch, two_files):
    named, target = two_files
    record = replace_after_read(monkeypatch, after=named, target=target,
                                data=b"first", how="rewrite")
    read_once(named)
    target.write_bytes(b"restored by the test")

    read_once(named)

    assert record.fired == 1
    assert target.read_bytes() == b"restored by the test"


def test_a_reader_importing_read_once_late_sees_it_too(monkeypatch, two_files):
    """Readers import `read_once` inside their bodies as well as at module level."""
    named, target = two_files
    record = replace_after_read(monkeypatch, after=named, target=target,
                                data=b"late", how="rename")

    from src.utils.fingerprint import read_once as late

    late(named)
    assert record.fired == 1
    assert fingerprint.read_once is late


@pytest.mark.parametrize("how", ["rename", "rewrite"])
def test_a_republish_binds_the_replacement_in_the_manifest(monkeypatch, tmp_path, how):
    named = tmp_path / "split_manifest.json"
    target = tmp_path / "node_features.pt"
    target.write_bytes(b"A")
    manifest = {"schema_version": 3, "artifacts": {"node_features": hashlib.sha256(b"A").hexdigest(),
                                                  "kg": "kept"}}
    named.write_text(json.dumps(manifest), encoding="utf-8")
    record = replace_after_read(monkeypatch, after=named, target=target, data=b"B",
                                how=how, republish="node_features")

    read = read_once(named)

    assert json.loads(read.data)["artifacts"]["node_features"] == hashlib.sha256(b"A").hexdigest()
    assert record.fired == 1
    assert target.read_bytes() == b"B"
    republished = json.loads(named.read_text(encoding="utf-8"))
    assert republished["artifacts"] == {"node_features": hashlib.sha256(b"B").hexdigest(),
                                        "kg": "kept"}
    assert republished["schema_version"] == 3


def test_an_unknown_way_is_refused(monkeypatch, two_files):
    named, target = two_files
    with pytest.raises(ValueError, match="how must be one of"):
        replace_after_read(monkeypatch, after=named, target=target, data=b"", how="copy")
