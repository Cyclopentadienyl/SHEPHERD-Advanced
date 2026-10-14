"""
The release check fails on every way of keeping a buffer it claims to catch.
============================================================================
`tests/fixtures/release.py` is what shows contract M2.1's readers release their
raw buffers, so it is shown to fail first. Every reader here uses the real
`read_once` on a non-empty file. The negative controls keep the bytes on a module
global, in a closure, on the result and across the next read; one keeps a
`BytesIO` holding its own copy of the bytes, which only the weak reference sees.
The positive controls are correct readers of one file and of several, which also
show the wrapper itself keeps nothing.

Module: tests/unit/test_release_check.py
"""
from __future__ import annotations

import json
import sys
from io import BytesIO

import pytest

from src.utils.fingerprint import read_once
from tests.fixtures.release import ReleaseCheckError, check_release
from tests.fixtures.replacement import replace_after_read

torch = pytest.importorskip("torch")

THIS = sys.modules[__name__]
PAYLOAD = {"rows": list(range(4096))}
_KEPT: list = []


@pytest.fixture
def files(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    tensors = tmp_path / "tensors.pt"
    first.write_text(json.dumps(PAYLOAD), encoding="utf-8")
    second.write_text(json.dumps({"other": list(range(2048))}), encoding="utf-8")
    torch.save({"x": torch.arange(4096, dtype=torch.float32)}, tensors)
    return first, second, tensors


def _json(path):
    read = read_once(path)
    return json.loads(read.data.decode("utf-8")), read.identity


def _torch(path):
    read = read_once(path)
    return torch.load(BytesIO(read.data), weights_only=True, map_location="cpu"), read.identity


# ---------------------------------------------------------------------------
# Positive controls
# ---------------------------------------------------------------------------
def test_a_correct_reader_of_one_file_passes(monkeypatch, files):
    first, _, _ = files

    (parsed, identity), reads = check_release(monkeypatch, lambda: _json(first))

    assert parsed == PAYLOAD
    assert identity.path == first
    assert reads == (first,)


def test_a_correct_reader_of_several_files_passes(monkeypatch, files):
    first, second, tensors = files

    def reader():
        return [_json(first), _json(second), _torch(tensors)]

    result, reads = check_release(monkeypatch, reader, bytesio_modules=[THIS])

    assert reads == (first, second, tensors)
    assert torch.equal(result[2][0]["x"], torch.arange(4096, dtype=torch.float32))


def test_the_replacement_double_keeps_no_buffer_either(monkeypatch, files):
    """Stacked with the double, a correct reader still passes: neither wrapper holds a read."""
    first, second, _ = files
    record = replace_after_read(monkeypatch, after=first, target=second,
                                data=b"[]", how="rewrite")

    result, _ = check_release(monkeypatch, lambda: [_json(first), _json(second)])

    assert record.fired == 1
    assert result[1][0] == []


# ---------------------------------------------------------------------------
# Negative controls: each must fail
# ---------------------------------------------------------------------------
def test_bytes_kept_on_a_module_global_fail(monkeypatch, files):
    first, _, _ = files

    def reader():
        read = read_once(first)
        _KEPT.append(read.data)
        return json.loads(read.data.decode("utf-8"))

    try:
        with pytest.raises(ReleaseCheckError, match="after the reader returned"):
            check_release(monkeypatch, reader)
    finally:
        _KEPT.clear()


def test_bytes_kept_in_a_closure_fail(monkeypatch, files):
    first, _, _ = files

    def reader():
        data = read_once(first).data
        return lambda: data

    with pytest.raises(ReleaseCheckError, match="after the reader returned"):
        check_release(monkeypatch, reader)


def test_bytes_kept_on_the_result_fail(monkeypatch, files):
    first, _, _ = files

    def reader():
        read = read_once(first)
        return {"parsed": json.loads(read.data.decode("utf-8")), "raw": read.data}

    with pytest.raises(ReleaseCheckError, match="after the reader returned"):
        check_release(monkeypatch, reader)


def test_bytes_kept_across_the_next_read_fail(monkeypatch, files):
    """A local still bound while the next file is read: released at return, caught on entry."""
    first, second, _ = files

    def reader():
        one = read_once(first)
        parsed_one = json.loads(one.data.decode("utf-8"))
        two = read_once(second)
        return parsed_one, json.loads(two.data.decode("utf-8"))

    with pytest.raises(ReleaseCheckError, match="before reading"):
        check_release(monkeypatch, reader)


def test_bytes_kept_only_by_a_reference_cycle_fail(monkeypatch, files):
    """Garbage collection is off while the reader runs, so a cycle cannot hide a
    buffer by being collected at a lucky moment. The reader drops its cycle and
    then allocates enough containers to trigger an automatic collection, which
    would free the buffer if collection were on."""
    first, _, _ = files

    def reader():
        box = {}
        box["self"] = box
        box["read"] = read_once(first)
        parsed = json.loads(box["read"].data.decode("utf-8"))
        del box
        churn = [[] for _ in range(50_000)]
        del churn
        return parsed

    with pytest.raises(ReleaseCheckError, match="after the reader returned"):
        check_release(monkeypatch, reader)


def test_a_kept_bytesio_with_its_own_copy_fails(monkeypatch, files):
    """`getbuffer()` makes the BytesIO copy the bytes, so the raw buffer is gone and only
    the weak reference sees what is kept."""
    _, _, tensors = files

    def reader():
        read = read_once(tensors)
        wrapper = BytesIO(read.data)
        wrapper.getbuffer().release()
        del read
        loaded = torch.load(wrapper, weights_only=True, map_location="cpu")
        return loaded, wrapper

    with pytest.raises(ReleaseCheckError, match="BytesIO over"):
        check_release(monkeypatch, reader, bytesio_modules=[THIS])


def test_a_kept_bytesio_sharing_the_bytes_fails_through_them(monkeypatch, files):
    _, _, tensors = files

    def reader():
        wrapper = BytesIO(read_once(tensors).data)
        return torch.load(wrapper, weights_only=True, map_location="cpu"), wrapper

    with pytest.raises(ReleaseCheckError, match="bytes allocated by read_once"):
        check_release(monkeypatch, reader, bytesio_modules=[THIS])


def test_a_torch_file_read_without_a_tracked_bytesio_fails(monkeypatch, files):
    """A wrapper built some other way would be invisible, so it is refused, not passed."""
    import io

    _, _, tensors = files

    def reader():
        read = read_once(tensors)
        return torch.load(io.BytesIO(read.data), weights_only=True, map_location="cpu")

    with pytest.raises(ReleaseCheckError, match="no tracked BytesIO"):
        check_release(monkeypatch, reader, bytesio_modules=[THIS])


def test_a_reader_that_never_calls_read_once_fails(monkeypatch, files):
    first, _, _ = files

    with pytest.raises(ReleaseCheckError, match="never called read_once"):
        check_release(monkeypatch, lambda: json.loads(first.read_text()))
