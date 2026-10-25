"""
`workspace_build.json`: when a workspace was built, descriptive and unbound (M5-1).
==================================================================================
The owner, 2026-10-19: the workspace builder records the build time from now on,
beside the manifest, with the code revision; it is descriptive and **not bound by
the manifest**, so two byte-identical builds keep one dataset fingerprint; a
workspace built before it shows "not recorded", never a modification time.

What these pin:

- `write_workspace` writes it, after the manifest, with `built_at` in UTC (ISO
  8601), the code revision (`code_revision`, training's producer collector) and
  schema version 1; a graph-only write writes none;
- the manifest does not name it, and two builds of the same inputs have
  byte-identical manifests whatever their build times;
- it names the manifest it describes, so the reader never shows a build time
  beside another manifest;
- the reader tolerates its absence and every malformed case as "not
  recorded", with the reason, and reads no modification time.

Module: tests/unit/test_workspace_build_record.py
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone

import pytest

from src.kg.artifacts import (
    WORKSPACE_BUILD_FILENAME,
    encode_workspace_build,
    read_workspace_build,
)
from src.config.server_settings import REPO_ROOT
from src.utils.fingerprint import file_sha256

_UTC = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_REVISION = {"state": "recorded", "revision": "a" * 40, "dirty": False,
             "modified_tracked": 0, "untracked": 0}


def _write(workspace, *, samples=True):
    from scripts.setup_demo import build_demo_kg
    from src.kg.workspace import SampleBudget, write_workspace

    budget = SampleBudget(num_train=20, num_val=5, val_disease_fraction=0.2) if samples else None
    return write_workspace(build_demo_kg(), workspace, feature_dim=8, samples=budget)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("built") / "ws"
    before = datetime.now(timezone.utc).replace(microsecond=0)
    _write(workspace)
    after = datetime.now(timezone.utc)
    return workspace, before, after


class TestTheBuilderWritesIt:

    def test_beside_the_manifest_with_when_and_which_code(self, built):
        from src.utils.run_environment import code_revision

        workspace, before, after = built
        record = json.loads((workspace / WORKSPACE_BUILD_FILENAME).read_text(encoding="utf-8"))
        assert record["schema_version"] == 1
        assert _UTC.fullmatch(record["built_at"])
        built_at = datetime.strptime(record["built_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc)
        assert before <= built_at <= after
        # Training's producer collector, of this repository's checkout.
        expected = code_revision(REPO_ROOT)
        assert set(record["code_revision"]) == set(expected)
        assert record["code_revision"]["state"] == expected["state"]
        assert record["code_revision"]["revision"] == expected["revision"]
        assert record["split_manifest_sha256"] == file_sha256(workspace / "split_manifest.json")

    def test_the_reader_reads_it_back(self, built):
        workspace, _, _ = built
        read = read_workspace_build(workspace, file_sha256(workspace / "split_manifest.json"))
        assert read.state == "recorded" and read.reason is None
        assert _UTC.fullmatch(read.built_at)
        assert isinstance(read.code_revision, dict)

    def test_a_graph_only_write_writes_none(self, tmp_path):
        _write(tmp_path / "ws", samples=False)
        assert not (tmp_path / "ws" / WORKSPACE_BUILD_FILENAME).exists()


class TestNotBoundByTheManifest:

    def test_the_manifest_does_not_name_it(self, built):
        workspace, _, _ = built
        text = (workspace / "split_manifest.json").read_text(encoding="utf-8")
        assert WORKSPACE_BUILD_FILENAME not in text and "built_at" not in text

    def test_two_builds_of_the_same_inputs_keep_one_fingerprint(self, built, tmp_path,
                                                                monkeypatch):
        """Another build time, the same manifest bytes: the build record is
        outside the dataset's identity."""
        import src.kg.workspace as workspace_module

        first, _, _ = built
        original = workspace_module.write_workspace_build
        later = datetime(2030, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        monkeypatch.setattr(workspace_module, "write_workspace_build",
                            lambda ws: original(ws, now=later))
        _write(tmp_path / "ws")
        second = tmp_path / "ws"
        assert file_sha256(second / "split_manifest.json") == file_sha256(
            first / "split_manifest.json")
        records = [json.loads((ws / WORKSPACE_BUILD_FILENAME).read_text(encoding="utf-8"))
                   for ws in (first, second)]
        assert records[1]["built_at"] == "2030-01-02T03:04:05Z"
        assert records[0]["built_at"] != records[1]["built_at"]


def test_the_time_is_recorded_in_utc(tmp_path):
    from src.kg.workspace import write_workspace_build

    _write(tmp_path / "ws")
    local = datetime(2026, 10, 20, 10, 30, 0, tzinfo=timezone(timedelta(hours=8)))
    write_workspace_build(tmp_path / "ws", now=local)
    record = json.loads((tmp_path / "ws" / WORKSPACE_BUILD_FILENAME).read_text(encoding="utf-8"))
    assert record["built_at"] == "2026-10-20T02:30:00Z"


class TestTheReaderSaysNotRecorded:

    def _copy(self, built, tmp_path):
        import shutil

        workspace, _, _ = built
        copy = tmp_path / "ws"
        shutil.copytree(workspace, copy)
        return copy, file_sha256(copy / "split_manifest.json")

    def test_absent_beside_a_manifest_names_both_causes(self, built, tmp_path):
        """A revision before M5-1, or a build stopped between its manifest and
        this record: the reader cannot tell which, and says both. Never a
        modification time, whatever the files' times are."""
        workspace, digest = self._copy(built, tmp_path)
        (workspace / WORKSPACE_BUILD_FILENAME).unlink()
        os.utime(workspace / "split_manifest.json", (1_600_000_000, 1_600_000_000))
        read = read_workspace_build(workspace, digest)
        assert read.state == "not_recorded" and read.built_at is None
        assert read.reason == (
            "no workspace_build.json beside this split_manifest.json: it was built by a "
            "revision before M5-1, or its build stopped before writing one")
        assert "2020" not in read.reason

    def test_a_build_stopped_before_its_record_does_not_claim_an_older_revision(
            self, tmp_path, monkeypatch):
        """The first build of a workspace, stopped after its manifest: trainable,
        and its reason is not "built before build times were recorded"."""
        import src.kg.workspace as workspace_module

        def stopped(workspace, **kwargs):
            raise KeyboardInterrupt("stopped")

        monkeypatch.setattr(workspace_module, "write_workspace_build", stopped)
        with pytest.raises(KeyboardInterrupt):
            _write(tmp_path / "ws")
        read = read_workspace_build(tmp_path / "ws",
                                    file_sha256(tmp_path / "ws" / "split_manifest.json"))
        assert read.state == "not_recorded"
        assert "its build stopped before writing one" in read.reason
        assert "built before build times were recorded" not in read.reason

    def test_a_record_of_another_manifest_is_not_this_build(self, built, tmp_path):
        """A rebuild that stopped after its manifest, or one by a revision that
        does not write the record, leaves the old record beside a new manifest."""
        workspace, _ = self._copy(built, tmp_path)
        read = read_workspace_build(workspace, "b" * 64)
        assert read.state == "not_recorded" and read.built_at is None
        assert "describes another split_manifest.json" in read.reason

    def test_without_a_manifest_nothing_is_paired(self, built, tmp_path):
        workspace, _ = self._copy(built, tmp_path)
        read = read_workspace_build(workspace, None)
        assert read.state == "not_recorded" and read.reason == (
            "there is no readable split_manifest.json for a workspace_build.json to describe")

    @pytest.mark.parametrize("record", [None, b"[]", b"{not json"],
                             ids=["absent", "not-an-object", "not-json"])
    def test_without_a_manifest_the_record_is_not_read(self, tmp_path, monkeypatch, record):
        """The missing manifest is the reason, whatever is or is not beside it;
        the record is not even opened."""
        import src.utils.fingerprint as fingerprint

        if record is not None:
            (tmp_path / WORKSPACE_BUILD_FILENAME).write_bytes(record)
        monkeypatch.setattr(fingerprint, "read_once",
                            lambda path: pytest.fail(f"{path} was read"))
        read = read_workspace_build(tmp_path, None)
        assert read.state == "not_recorded"
        assert "no readable split_manifest.json" in read.reason

    def test_a_graph_only_workspace_says_it_has_no_manifest(self, tmp_path):
        """A graph-only build today writes neither file; it was not "built
        before build times were recorded"."""
        _write(tmp_path / "ws", samples=False)
        read = read_workspace_build(tmp_path / "ws", None)
        assert "no readable split_manifest.json" in read.reason
        assert "M5-1" not in read.reason

    @pytest.mark.parametrize("content, fragment", [
        (b"\xff\xfe not utf-8", "not UTF-8 JSON"),
        (b"{not json", "not UTF-8 JSON"),
        (b"[]", "not a JSON object"),
        (json.dumps({"schema_version": 2, "built_at": "2026-10-20T00:00:00Z",
                     "code_revision": {}, "split_manifest_sha256": "c" * 64}).encode(),
         "schema_version is 2"),
        (json.dumps({"schema_version": 1, "built_at": "yesterday",
                     "code_revision": {}, "split_manifest_sha256": "c" * 64}).encode(),
         "built_at is not a UTC time"),
        (json.dumps({"schema_version": 1, "built_at": "2026-10-20T00:00:00Z",
                     "code_revision": "abc", "split_manifest_sha256": "c" * 64}).encode(),
         "code_revision is not an object"),
        (json.dumps({"schema_version": 1, "built_at": "2026-10-20T00:00:00Z",
                     "code_revision": {}}).encode(),
         "split_manifest_sha256 is not a SHA-256"),
    ])
    def test_a_malformed_record_is_not_recorded(self, tmp_path, content, fragment):
        (tmp_path / WORKSPACE_BUILD_FILENAME).write_bytes(content)
        read = read_workspace_build(tmp_path, "c" * 64)
        assert read.state == "not_recorded" and read.built_at is None
        assert fragment in read.reason

    def test_a_well_formed_record_of_this_manifest_is_recorded(self, tmp_path):
        (tmp_path / WORKSPACE_BUILD_FILENAME).write_bytes(encode_workspace_build(
            built_at="2026-10-20T00:00:00Z", code_revision=_REVISION,
            split_manifest_sha256="c" * 64))
        read = read_workspace_build(tmp_path, "c" * 64)
        assert read == ("recorded", "2026-10-20T00:00:00Z", _REVISION, None)


def test_the_writer_refuses_what_the_reader_would_not_read():
    with pytest.raises(ValueError, match="built_at is not a UTC time"):
        encode_workspace_build(built_at="2026-10-20 00:00", code_revision=_REVISION,
                               split_manifest_sha256="c" * 64)
    with pytest.raises(ValueError, match="split_manifest_sha256"):
        encode_workspace_build(built_at="2026-10-20T00:00:00Z", code_revision=_REVISION,
                               split_manifest_sha256="short")
