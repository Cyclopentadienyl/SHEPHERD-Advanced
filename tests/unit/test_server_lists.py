"""
The server's lists (M5-1): workspaces, their checkpoints, cohort versions.
=========================================================================
`src/api/services/server_lists.py` and the routes over it. What these pin:

- **from server settings only**: no root, no list, and why; never a default;
- **items by name and digest, never a path**: the dataset fingerprint is the
  full SHA-256 of `split_manifest.json`, a checkpoint's the SHA-256 of the bytes
  read, a version's that of its `cohort.json`; a trainable workspace's item holds
  no server path, and `resolve_*` turns only a listed name into a path;
- **the existing readers, each file once**: the dataset record's origin and
  binding state apart, the build record or "not recorded" (never a
  modification time), each checkpoint read once and a load problem kept as a
  value, each version through `resolve_version`;
- **trainable only with a manifest, a verified dataset record and its files**,
  with why not otherwise;
- **cohort versions carry aggregates, never a case identifier** (D7), and the
  report store's runs, whose kept and registered state comes only from the
  reader given;
- **one unreadable entry never fails a list**: a malformed manifest, a file or
  directory this account may not check, a checkpoint link whose target is gone;
- **a checkpoint outside the workspace is listed and never opened**, and items
  name the workspace as the server lists it;
- **the checkpoint route is gated by D8** (SPEC_4 C3): refused, opening nothing,
  without mode M1 and a loopback request; the workspace and cohort lists stay
  open, and the cohort route reads each run's state through the registration.

Module: tests/unit/test_server_lists.py
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pytest

from src.api.services import server_lists as sl
from src.config import server_settings as ss
from src.config.model_types import WRITTEN_BEFORE_SPEC
from src.utils.fingerprint import file_sha256


def _roots(root: Path, **more: str) -> dict:
    return {ss.ENV_WORKSPACES_ROOT: str(root), **more}


@pytest.fixture(scope="module")
def demo_workspace(tmp_path_factory):
    """One complete demo workspace, built once; tests copy it."""
    from tests.fixtures.registered_cohort import write_demo_workspace

    return write_demo_workspace(tmp_path_factory.mktemp("demo") / "ws")


def _copy(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    return target


# ---------------------------------------------------------------------------
# Workspaces
# ---------------------------------------------------------------------------
class TestWorkspaces:

    def test_no_root_no_list_and_why(self):
        listing = sl.list_workspaces(environ={})
        assert listing.items == () and not listing.available
        assert "SHEPHERD_WORKSPACES_ROOT is not set" in listing.problem

    def test_a_refused_root_lists_nothing(self, tmp_path):
        listing = sl.list_workspaces(environ=_roots(tmp_path / "missing"))
        assert listing.items == () and "does not exist" in listing.problem

    def test_a_complete_workspace(self, demo_workspace, tmp_path):
        root = tmp_path / "root"
        ws = _copy(demo_workspace, root / "demo")
        manifest = json.loads((ws / "split_manifest.json").read_text(encoding="utf-8"))

        (item,) = sl.list_workspaces(environ=_roots(root)).items

        assert item.name == "demo"
        assert item.dataset_fingerprint == file_sha256(ws / "split_manifest.json")
        assert len(item.dataset_fingerprint) == 64
        assert item.trainable and item.not_trainable is None
        assert item.dataset_record == {
            "binding": "verified", "origin": "synthetic",
            "sha256": file_sha256(ws / "kg.provenance.json"), "problem": None}
        assert item.declared_releases == ()  # a synthetic build names no source
        assert item.build["state"] == "recorded" and item.build["reason"] is None
        assert item.build["built_at"].endswith("Z")
        assert (item.train_samples, item.val_samples) == (
            manifest["generation"]["num_train"], manifest["generation"]["num_val"])
        assert (item.train_samples, item.val_samples) == (20, 5)

    def test_the_item_carries_no_server_path(self, demo_workspace, tmp_path):
        root = tmp_path / "root"
        _copy(demo_workspace, root / "demo")
        (item,) = sl.list_workspaces(environ=_roots(root)).items
        assert str(tmp_path) not in json.dumps(asdict(item))

    def test_declared_releases_come_from_the_dataset_record(self, demo_workspace, tmp_path,
                                                             monkeypatch):
        """The record's sources as each declares its release; read through the
        existing reader, not reparsed here."""
        import src.kg.artifacts as artifacts

        record = artifacts.DatasetRecord(
            {"origin": "files", "sources": [
                {"role": "hpo", "filename": "hp.obo", "digest": "d" * 64,
                 "declared_version": "2026-09-01"},
                {"role": "mondo", "filename": "mondo.obo", "digest": "e" * 64,
                 "declared_version": None}]},
            artifacts.ReadIdentity(Path("kg.provenance.json"), "f" * 64))
        monkeypatch.setattr(artifacts, "verify_dataset_record", lambda manifest, kg: record)
        root = tmp_path / "root"
        _copy(demo_workspace, root / "demo")
        (item,) = sl.list_workspaces(environ=_roots(root)).items
        assert item.declared_releases == (
            {"role": "hpo", "declared_version": "2026-09-01"},
            {"role": "mondo", "declared_version": None})
        assert item.dataset_record["origin"] == "files"

    def test_no_manifest_is_not_trainable_and_says_why(self, tmp_path):
        root = tmp_path / "root"
        (root / "graph-only").mkdir(parents=True)
        (item,) = sl.list_workspaces(environ=_roots(root)).items
        assert item.dataset_fingerprint is None and not item.trainable
        assert "has no split_manifest.json" in item.not_trainable
        assert item.build["state"] == "not_recorded"
        assert "no readable split_manifest.json" in item.build["reason"]
        assert item.dataset_record["binding"] is None

    def test_a_workspace_built_before_build_records(self, demo_workspace, tmp_path):
        root = tmp_path / "root"
        ws = _copy(demo_workspace, root / "old")
        (ws / "workspace_build.json").unlink()
        (item,) = sl.list_workspaces(environ=_roots(root)).items
        assert item.trainable  # a missing build record is descriptive only
        assert item.build == {
            "state": "not_recorded", "built_at": None, "code_revision": None,
            "reason": ("no workspace_build.json beside this split_manifest.json: it was "
                       "built by a revision before M5-1, or its build stopped before "
                       "writing one")}

    def test_a_build_record_of_another_manifest_is_not_shown(self, demo_workspace, tmp_path):
        root = tmp_path / "root"
        ws = _copy(demo_workspace, root / "rebuilt")
        record = json.loads((ws / "workspace_build.json").read_text(encoding="utf-8"))
        record["split_manifest_sha256"] = "0" * 64
        (ws / "workspace_build.json").write_text(json.dumps(record), encoding="utf-8")
        (item,) = sl.list_workspaces(environ=_roots(root)).items
        assert item.build["state"] == "not_recorded" and item.build["built_at"] is None
        assert "describes another split_manifest.json" in item.build["reason"]

    def test_a_replaced_dataset_record_is_a_mismatch(self, demo_workspace, tmp_path):
        root = tmp_path / "root"
        ws = _copy(demo_workspace, root / "tampered")
        record = json.loads((ws / "kg.provenance.json").read_text(encoding="utf-8"))
        record["counters"] = {"edited": 1}
        (ws / "kg.provenance.json").write_text(json.dumps(record), encoding="utf-8")
        (item,) = sl.list_workspaces(environ=_roots(root)).items
        assert item.dataset_record["binding"] == "mismatch"
        assert not item.trainable and "mismatch" in item.not_trainable
        assert item.declared_releases == ()

    def test_a_missing_graph_file_is_not_trainable(self, demo_workspace, tmp_path):
        root = tmp_path / "root"
        ws = _copy(demo_workspace, root / "partial")
        (ws / "node_features.pt").unlink()
        (item,) = sl.list_workspaces(environ=_roots(root)).items
        assert item.dataset_record["binding"] == "verified"
        assert not item.trainable and "node_features.pt" in item.not_trainable

    def test_hidden_entries_and_files_are_not_workspaces(self, tmp_path):
        root = tmp_path / "root"
        for name in ("b", "a", ".cache"):
            (root / name).mkdir(parents=True)
        (root / "notes.txt").write_text("x")
        names = [item.name for item in sl.list_workspaces(environ=_roots(root)).items]
        assert names == ["a", "b"]

    def test_a_link_out_of_the_root_is_listed_and_refused(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        (tmp_path / "elsewhere").mkdir()
        (root / "away").symlink_to(tmp_path / "elsewhere")
        (item,) = sl.list_workspaces(environ=_roots(root)).items
        assert item.name == "away" and not item.trainable
        assert "outside the workspaces root" in item.not_trainable
        with pytest.raises(sl.ServerListRefused, match="outside the workspaces root"):
            sl.resolve_workspace("away", environ=_roots(root))

    def test_a_manifest_nested_too_deep_is_one_unreadable_workspace(self, demo_workspace,
                                                                      tmp_path):
        """`json` meets nesting deeper than the interpreter's limit with a
        `RecursionError`; the reader refuses it like any malformed manifest, so
        the other workspaces are still listed (M5-1 review)."""
        root = tmp_path / "root"
        _copy(demo_workspace, root / "good")
        (root / "deep").mkdir()
        (root / "deep" / "split_manifest.json").write_text("[" * 100_000, encoding="utf-8")
        listing = sl.list_workspaces(environ=_roots(root))
        assert listing.available
        items = {item.name: item for item in listing.items}
        assert items["good"].trainable
        assert not items["deep"].trainable and "not UTF-8 JSON" in items["deep"].not_trainable
        assert sl.list_checkpoints("deep", environ=_roots(root)).available

    def test_a_bound_file_that_cannot_be_checked_is_not_trainable(
            self, demo_workspace, tmp_path, monkeypatch):
        """`Path.is_file` raises EACCES rather than answer (a link into a
        directory this account may not enter): the workspace says so."""
        root = tmp_path / "root"
        ws = _copy(demo_workspace, root / "demo").resolve()
        original = Path.is_file

        def is_file(self, *args, **kwargs):
            if self == ws / "node_features.pt":
                raise PermissionError(13, "Permission denied", str(self))
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Path, "is_file", is_file)
        (item,) = sl.list_workspaces(environ=_roots(root)).items
        assert not item.trainable
        assert item.not_trainable == ("files its split_manifest.json binds cannot be "
                                      "checked (node_features.pt: PermissionError)")

    def test_an_entry_that_cannot_be_checked_does_not_fail_the_list(
            self, demo_workspace, tmp_path, monkeypatch):
        """One entry the account may not check is listed with why; the rest of
        the list stands."""
        root = tmp_path / "root"
        _copy(demo_workspace, root / "good")
        locked = root / "locked"
        locked.mkdir()
        originals = {name: getattr(Path, name) for name in ("is_dir", "is_file")}

        def refusing(name):
            def check(self, *args, **kwargs):
                if self == locked or locked in self.parents:
                    raise PermissionError(13, "Permission denied", str(self))
                return originals[name](self, *args, **kwargs)
            return check

        for name in originals:
            monkeypatch.setattr(Path, name, refusing(name))
        listing = sl.list_workspaces(environ=_roots(root))
        assert listing.available
        items = {item.name: item for item in listing.items}
        assert set(items) == {"good", "locked"} and items["good"].trainable
        assert not items["locked"].trainable
        assert "Permission denied" in items["locked"].not_trainable

    def test_each_call_reads_the_setting_again(self, tmp_path, monkeypatch):
        first, second = tmp_path / "one", tmp_path / "two"
        (first / "x").mkdir(parents=True)
        (second / "y").mkdir(parents=True)
        monkeypatch.setenv(ss.ENV_WORKSPACES_ROOT, str(first))
        assert [i.name for i in sl.list_workspaces().items] == ["x"]
        monkeypatch.setenv(ss.ENV_WORKSPACES_ROOT, str(second))
        assert [i.name for i in sl.list_workspaces().items] == ["y"]


class TestResolveWorkspace:

    def test_a_listed_name_resolves(self, tmp_path):
        (tmp_path / "root" / "demo").mkdir(parents=True)
        assert sl.resolve_workspace("demo", environ=_roots(tmp_path / "root")) == (
            tmp_path / "root" / "demo").resolve()

    @pytest.mark.parametrize("name", ["missing", "../root", ".", "", ".hidden", None,
                                      "demo/../demo", "/etc"])
    def test_only_a_listed_name_resolves(self, tmp_path, name):
        root = tmp_path / "root"
        (root / "demo").mkdir(parents=True)
        (root / ".hidden").mkdir()
        with pytest.raises(sl.ServerListRefused, match="is not a workspace the server lists"):
            sl.resolve_workspace(name, environ=_roots(root))

    def test_no_root_refuses(self):
        with pytest.raises(sl.ServerListRefused, match="SHEPHERD_WORKSPACES_ROOT is not set"):
            sl.resolve_workspace("demo", environ={})

    def test_a_long_name_is_not_echoed_whole(self, tmp_path):
        (tmp_path / "root").mkdir()
        with pytest.raises(sl.ServerListRefused) as refused:
            sl.resolve_workspace("x" * 500, environ=_roots(tmp_path / "root"))
        assert "x" * 100 not in str(refused.value)


class TestResolveWorkspaces:
    """Every name at once, from one listing of the root (M5-2): what
    `resolve_workspace` answers for each, and nothing it refuses."""

    def test_each_listed_name_resolves_as_resolve_workspace_says(self, tmp_path, monkeypatch):
        root = tmp_path / "root"
        for name in ("demo", "other", ".hidden"):
            (root / name).mkdir(parents=True)
        (root / "alias").symlink_to(root / "other", target_is_directory=True)
        (tmp_path / "away").mkdir()
        (root / "out").symlink_to(tmp_path / "away", target_is_directory=True)
        listed = []
        real = sl._workspace_entries
        monkeypatch.setattr(sl, "_workspace_entries",
                            lambda r: listed.append(r) or real(r))
        resolved = sl.resolve_workspaces(environ=_roots(root))
        assert len(listed) == 1                      # one listing, not one per name
        assert resolved == {name: sl.resolve_workspace(name, environ=_roots(root))
                            for name in ("alias", "demo", "other")}
        assert "out" not in resolved and ".hidden" not in resolved

    def test_no_root_refuses(self):
        with pytest.raises(sl.ServerListRefused, match="SHEPHERD_WORKSPACES_ROOT is not set"):
            sl.resolve_workspaces(environ={})


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
class NotPlainData:
    """A pickled class, which torch's weights-only loader refuses."""


def _run(workspace: Path, architecture: str, name: str) -> Path:
    directory = workspace / "runs" / architecture / name / "checkpoints"
    directory.mkdir(parents=True)
    return directory


@pytest.fixture(scope="module")
def checkpoint_root(demo_workspace, tmp_path_factory):
    """A root with one workspace holding: a checkpoint trained on it (a file
    without a model spec, recording this workspace's training inputs), one
    recording another dataset, one with no validation figure, one torch's safe
    loader refuses, and one that is not a checkpoint at all."""
    import torch

    from src.utils.checkpoint_io import write_checkpoint
    from tests.fixtures.registered_cohort import write_checkpoint_without_model_spec

    root = tmp_path_factory.mktemp("ckroot")
    ws = _copy(demo_workspace, root / "demo")
    digests = {}
    newest = _run(ws, "hgt", "20261020-120000_new_aaaaaa")
    digests["trained"] = write_checkpoint_without_model_spec(newest / "best.pt", data_dir=ws)
    digests["other"] = write_checkpoint(
        {"checkpoint_kind": "training_state", "epoch": 7,
         "logs": {"val_loss": 0.1, "val_pooled_mrr": 0.4},
         "training_input_digests": {"split_manifest": "9" * 64},
         "progress": {"completed_epochs": 8, "end": "paused"}}, newest / "other.pt")
    older = _run(ws, "sage", "20261019-120000_old_bbbbbb")
    digests["weights"] = write_checkpoint({"checkpoint_kind": "weights", "epoch": 2},
                                          older / "weights.pt")

    torch.save({"model": NotPlainData()}, older / "pickled.pt")
    (older / "garbage.pt").write_bytes(b"not a checkpoint")
    return root, ws, digests


class TestCheckpoints:

    def _items(self, checkpoint_root, architecture=None):
        root, _, _ = checkpoint_root
        listing = sl.list_checkpoints("demo", architecture, environ=_roots(root))
        assert listing.available, listing.problem
        return {item.filename: item for item in listing.items}

    def test_every_file_is_listed_none_dropped(self, checkpoint_root):
        items = self._items(checkpoint_root)
        assert set(items) == {"best.pt", "other.pt", "weights.pt", "pickled.pt", "garbage.pt"}

    def test_a_checkpoint_trained_on_this_dataset(self, checkpoint_root):
        _, ws, digests = checkpoint_root
        item = self._items(checkpoint_root)["best.pt"]
        assert item.sha256 == digests["trained"] == file_sha256(
            ws / "runs/hgt/20261020-120000_new_aaaaaa/checkpoints/best.pt")
        assert (item.workspace, item.architecture, item.run) == (
            "demo", "hgt", "20261020-120000_new_aaaaaa")
        # Written before the model spec: its record is refused with the one
        # phrase, its digest and figures still listed; its `val_mrr` is no
        # figure this build ranks by, so it shows none (Keep: the field stays).
        assert (item.record, item.record_problem) == ("refused", WRITTEN_BEFORE_SPEC)
        assert item.validation is None
        assert item.epoch == 1
        assert item.dataset_fingerprint == file_sha256(ws / "split_manifest.json")
        assert item.trained_on_this_dataset is True
        assert item.problem is None and item.size_bytes > 0

    def test_a_checkpoint_of_another_dataset_says_so(self, checkpoint_root):
        item = self._items(checkpoint_root)["other.pt"]
        assert item.dataset_fingerprint == "9" * 64
        assert item.trained_on_this_dataset is False
        assert item.validation == {"metric": "val_pooled_mrr", "value": 0.4}
        assert item.checkpoint_kind == "training_state" and item.epoch == 7

    def test_the_progress_record_is_the_files_own_never_derived(self, checkpoint_root):
        """The epochs a file completed and how its stage ended, as its progress
        record says (the GB10 WebUI recheck: the list said "epoch 194" of a run
        whose curves showed 195 completed); a file with no progress record has
        none, never its epoch index plus one."""
        items = self._items(checkpoint_root)
        assert (items["other.pt"].epoch, items["other.pt"].completed_epochs,
                items["other.pt"].end) == (7, 8, "paused")
        assert (items["weights.pt"].epoch, items["weights.pt"].completed_epochs,
                items["weights.pt"].end) == (2, None, None)
        assert items["garbage.pt"].completed_epochs is None and items["garbage.pt"].end is None

    def test_no_validation_figure_is_none_not_zero(self, checkpoint_root):
        item = self._items(checkpoint_root)["weights.pt"]
        assert item.validation is None and item.problem is None
        assert item.dataset_fingerprint is None and item.trained_on_this_dataset is None
        assert item.sha256 is not None

    @pytest.mark.parametrize("filename", ["pickled.pt", "garbage.pt"])
    def test_an_unreadable_file_is_listed_with_its_problem(self, checkpoint_root, filename):
        item = self._items(checkpoint_root)[filename]
        assert item.problem and item.sha256 is None and item.validation is None
        assert "\n" not in item.problem and len(item.problem) < 400

    def test_one_architecture(self, checkpoint_root):
        assert set(self._items(checkpoint_root, "hgt")) == {"best.pt", "other.pt"}

    def test_the_listing_order_is_kept(self, checkpoint_root):
        """The one listing's order (latest run first, newest file first)."""
        from src.utils.checkpoint_paths import list_run_checkpoints

        root, ws, _ = checkpoint_root
        listed = [(item.run, item.filename)
                  for item in sl.list_checkpoints("demo", environ=_roots(root)).items]
        assert listed == [(run.name, path.name) for run, path in list_run_checkpoints(ws)]

    def test_each_file_is_read_once(self, checkpoint_root, monkeypatch):
        import src.utils.checkpoint_io as checkpoint_io

        root, _, _ = checkpoint_root
        original = checkpoint_io.read_checkpoint
        reads = []

        def counting(path, **kwargs):
            reads.append(Path(path).name)
            assert kwargs == {"map_location": "cpu", "weights_only": True}
            return original(path, **kwargs)

        monkeypatch.setattr(checkpoint_io, "read_checkpoint", counting)
        sl.list_checkpoints("demo", environ=_roots(root))
        assert sorted(reads) == sorted(
            ["best.pt", "other.pt", "weights.pt", "pickled.pt", "garbage.pt"])

    def test_a_name_off_the_list(self, checkpoint_root):
        root, _, _ = checkpoint_root
        listing = sl.list_checkpoints("missing", None, environ=_roots(root))
        assert listing.items == () and "is not a workspace the server lists" in listing.problem

    @pytest.mark.parametrize("architecture", ["gatv9", ".", "..", "hgt/..", "../demo/runs/hgt",
                                              "HGT", "hgt/"])
    def test_a_directory_name_no_directory_has_lists_nothing(self, checkpoint_root,
                                                             architecture):
        """N4 (C1, C7): the filter names a directory under runs/ by equality
        with the listed names, never joined onto a path, so ``.``, ``..`` or a
        name with a slash reaches no other directory; a name no directory has
        lists nothing (the list is available, as for a workspace without that
        directory), and no file is read."""
        root, _, _ = checkpoint_root
        listing = sl.list_checkpoints("demo", architecture, environ=_roots(root))
        assert listing.available and listing.items == ()

    def test_no_item_carries_a_path(self, checkpoint_root):
        root, _, _ = checkpoint_root
        listing = sl.list_checkpoints("demo", environ=_roots(root))
        assert str(root) not in json.dumps(listing.as_dict("checkpoints"))

    def test_one_run_is_listed_alone_and_only_its_files_are_opened(self, checkpoint_root,
                                                                   monkeypatch):
        """`run=` (a Fine-tune child's products): that run directory's files,
        each read once, and no other file opened."""
        from collections import OrderedDict

        import src.utils.checkpoint_io as checkpoint_io
        import src.utils.checkpoint_paths as cp

        root, _, _ = checkpoint_root
        monkeypatch.setattr(cp, "_READS", OrderedDict())
        original, reads = checkpoint_io.read_checkpoint, []

        def counting(path, **kwargs):
            reads.append(Path(path).name)
            return original(path, **kwargs)

        monkeypatch.setattr(checkpoint_io, "read_checkpoint", counting)
        listing = sl.list_checkpoints("demo", run="20261019-120000_old_bbbbbb",
                                      environ=_roots(root))
        assert listing.available, listing.problem
        assert {item.run for item in listing.items} == {"20261019-120000_old_bbbbbb"}
        assert sorted(item.filename for item in listing.items) == sorted(reads) == [
            "garbage.pt", "pickled.pt", "weights.pt"]
        assert sl.list_checkpoints("demo", run="missing", environ=_roots(root)).items == ()


def _fixture_for(workspace: Path, manifest_sha256: str) -> None:
    """A functional fixture record bound to `manifest_sha256`, through the
    fixture's own publisher (its sample files are empty: readiness reads the
    record alone)."""
    from src.kg.task_fixture import publish_gene_fixture

    publish_gene_fixture(workspace, {"train": [], "val": []},
                         {"workspace": {"split_manifest": manifest_sha256}}, replace=True)


class TestTaskReadiness:
    """`task_readiness` (the Fine-tune page's task options): JSON reads only,
    no checkpoint opened, words that name no path."""

    @pytest.fixture(autouse=True)
    def _no_checkpoint_read(self, monkeypatch):
        import src.utils.checkpoint_io as checkpoint_io

        monkeypatch.setattr(checkpoint_io, "read_checkpoint",
                            lambda *a, **k: pytest.fail("readiness opened a checkpoint"))

    def test_disease_is_ready_on_a_trainable_workspace_and_the_fixture_tasks_need_it(
            self, demo_workspace, tmp_path):
        root = tmp_path / "root"
        ws = _copy(demo_workspace, root / "demo")
        assert sl.task_readiness("demo", environ=_roots(root)) == {
            "disease": None, "gene": "no functional fixture",
            "patients_like_me": "no functional fixture"}
        _fixture_for(ws, "0" * 64)
        assert sl.task_readiness("demo", environ=_roots(root)) == {
            "disease": None, "gene": "fixture from another build",
            "patients_like_me": "fixture from another build"}
        _fixture_for(ws, file_sha256(ws / "split_manifest.json"))
        assert sl.task_readiness("demo", environ=_roots(root)) == dict.fromkeys(
            ("disease", "gene", "patients_like_me"))
        (ws / "derived" / "gene_fixture" / "task_fixture.json").write_text("{torn")
        assert sl.task_readiness("demo", environ=_roots(root))["gene"] == "fixture not readable"

    def test_no_task_is_ready_on_a_workspace_that_is_not_trainable_or_not_listed(
            self, demo_workspace, tmp_path):
        root = tmp_path / "root"
        ws = _copy(demo_workspace, root / "demo")
        _fixture_for(ws, file_sha256(ws / "split_manifest.json"))
        (ws / "edge_indices.pt").unlink()
        assert set(sl.task_readiness("demo", environ=_roots(root)).values()) == {
            "dataset not trainable"}
        for name, environ in (("missing", _roots(root)), ("demo", {})):
            readiness = sl.task_readiness(name, environ=environ)
            assert set(readiness) == {"disease", "gene", "patients_like_me"}
            assert set(readiness.values()) == {"not a listed workspace"}
            assert str(tmp_path) not in json.dumps(readiness)


class TestResolveCheckpoint:

    def test_a_listed_checkpoint_resolves(self, checkpoint_root):
        root, ws, _ = checkpoint_root
        path = sl.resolve_checkpoint("demo", "hgt", "20261020-120000_new_aaaaaa", "best.pt",
                                     environ=_roots(root))
        assert path == ws / "runs/hgt/20261020-120000_new_aaaaaa/checkpoints/best.pt"

    @pytest.mark.parametrize("architecture, run, filename, fragment", [
        ("hgt", "20261020-120000_new_aaaaaa", "missing.pt", "is listed"),
        ("sage", "20261020-120000_new_aaaaaa", "best.pt", "is listed"),
        ("hgt", "../sage/20261019-120000_old_bbbbbb", "weights.pt", "is listed"),
        ("", "20261020-120000_new_aaaaaa", "best.pt", "is not an architecture"),
        (None, "20261020-120000_new_aaaaaa", "best.pt", "is not an architecture"),
        ("gatv9", "20261020-120000_new_aaaaaa", "best.pt", "of 'gatv9' is listed"),
        ("..", "20261020-120000_new_aaaaaa", "best.pt", "is listed"),
        (".", "20261020-120000_new_aaaaaa", "best.pt", "is listed"),
        ("runs/hgt", "20261020-120000_new_aaaaaa", "best.pt", "is listed"),
        ("hgt/../hgt", "20261020-120000_new_aaaaaa", "best.pt", "is listed"),
    ])
    def test_anything_else_is_refused(self, checkpoint_root, architecture, run, filename,
                                      fragment):
        root, _, _ = checkpoint_root
        with pytest.raises(sl.ServerListRefused, match=fragment):
            sl.resolve_checkpoint("demo", architecture, run, filename, environ=_roots(root))

    def test_a_run_linked_out_of_the_workspace_is_refused(self, demo_workspace, tmp_path):
        from src.utils.checkpoint_io import write_checkpoint

        root = tmp_path / "root"
        ws = _copy(demo_workspace, root / "demo")
        outside = tmp_path / "outside" / "checkpoints"
        outside.mkdir(parents=True)
        write_checkpoint({"epoch": 1}, outside / "x.pt")
        (ws / "runs" / "gatv2").mkdir(parents=True)
        (ws / "runs" / "gatv2" / "20261020-000000_far_cccccc").symlink_to(outside.parent)
        with pytest.raises(sl.ServerListRefused, match=(
                "in the workspace 'demo', 20261020-000000_far_cccccc/x.pt resolves outside the "
                "workspace, so it is not read")):
            sl.resolve_checkpoint("demo", "gatv2", "20261020-000000_far_cccccc", "x.pt",
                                  environ=_roots(root))
        (item,) = sl.list_checkpoints("demo", environ=_roots(root)).items
        assert "outside the workspace" in item.problem
        assert (item.sha256, item.size_bytes, item.epoch) == (None, None, None)


class TestCheckpointsInAndOut:
    """What the checkpoint list opens, and what it names (M5-1 review)."""

    def test_a_file_linked_out_of_the_workspace_is_never_opened(
            self, demo_workspace, tmp_path, monkeypatch):
        import src.utils.checkpoint_io as checkpoint_io
        from src.utils.checkpoint_io import write_checkpoint

        root = tmp_path / "root"
        ws = _copy(demo_workspace, root / "demo")
        secret = tmp_path / "outside" / "secret.pt"
        secret.parent.mkdir()
        write_checkpoint({"epoch": 9, "logs": {"val_pooled_mrr": 0.9},
                          "training_input_digests": {"split_manifest": "1" * 64}}, secret)
        run = _run(ws, "gatv2", "20261020-000000_x_aaaaaa")
        write_checkpoint({"epoch": 1}, run / "inside.pt")
        (run / "link.pt").symlink_to(secret)

        original = checkpoint_io.read_checkpoint
        reads = []

        def counting(path, **kwargs):
            reads.append(Path(path).resolve())
            return original(path, **kwargs)

        monkeypatch.setattr(checkpoint_io, "read_checkpoint", counting)
        items = {i.filename: i for i in sl.list_checkpoints("demo", environ=_roots(root)).items}
        assert secret.resolve() not in reads and len(reads) == 1
        link = items["link.pt"]
        assert "resolves outside the workspace" in link.problem
        assert (link.sha256, link.size_bytes, link.checkpoint_kind, link.epoch,
                link.validation, link.dataset_fingerprint,
                link.trained_on_this_dataset) == (None,) * 7
        assert items["inside.pt"].problem is None and items["inside.pt"].sha256

    def test_a_link_whose_target_is_gone_is_listed_with_its_problem(
            self, demo_workspace, tmp_path):
        from src.utils.checkpoint_io import write_checkpoint

        root = tmp_path / "root"
        ws = _copy(demo_workspace, root / "demo")
        run = _run(ws, "gatv2", "20261020-000000_x_aaaaaa")
        write_checkpoint({"epoch": 1}, run / "last.pt")
        (run / "gone.pt").symlink_to(run / "removed.pt")
        listing = sl.list_checkpoints("demo", environ=_roots(root))
        assert listing.available
        items = {i.filename: i for i in listing.items}
        assert items["last.pt"].problem is None
        assert items["gone.pt"].problem.startswith("FileNotFoundError")
        assert items["gone.pt"].sha256 is None

    def test_an_item_names_the_workspace_as_listed(self, demo_workspace, tmp_path):
        """`current -> builds/2026-10-20`: the item says `current`, the name the
        server lists, so it resolves again by its own name."""
        from src.utils.checkpoint_io import write_checkpoint

        root = tmp_path / "root"
        ws = _copy(demo_workspace, root / "builds" / "2026-10-20")
        (root / "current").symlink_to(ws)
        run = _run(ws, "gatv2", "20261020-000000_x_aaaaaa")
        write_checkpoint({"epoch": 1}, run / "last.pt")

        (item,) = sl.list_checkpoints("current", environ=_roots(root)).items
        assert item.workspace == "current"
        assert sl.resolve_checkpoint(item.workspace, item.architecture, item.run,
                                     item.filename, environ=_roots(root)) == run / "last.pt"
        with pytest.raises(sl.ServerListRefused, match="in the workspace 'current'"):
            sl.resolve_checkpoint("current", "gatv2", item.run, "missing.pt",
                                  environ=_roots(root))

    def test_runs_that_cannot_be_listed_say_why(self, demo_workspace, tmp_path, monkeypatch):
        import src.utils.checkpoint_paths as checkpoint_paths

        root = tmp_path / "root"
        _copy(demo_workspace, root / "demo")

        def refusing(*args, **kwargs):
            raise PermissionError(13, "Permission denied", "runs/gatv2")

        monkeypatch.setattr(checkpoint_paths, "list_runs", refusing)
        listing = sl.list_checkpoints("demo", environ=_roots(root))
        assert listing.items == ()
        assert listing.problem == ("the runs of the workspace 'demo' cannot be listed "
                                   "(PermissionError: Permission denied)")
        with pytest.raises(sl.ServerListRefused, match="cannot be listed"):
            sl.resolve_checkpoint("demo", "gatv2", "r", "x.pt", environ=_roots(root))

    def test_a_directory_under_runs_that_cannot_be_listed_hides_nothing_else(
            self, demo_workspace, tmp_path, monkeypatch):
        """N6 (C3): every directory under runs/ is listed now, so one this
        account may not enter (another account's ``mkdir -m 700``; simulated,
        as root ignores modes) is listed as one item with why, and every other
        run is still listed, read and selected. Only runs/ itself failing
        makes the list unavailable."""
        from src.utils.checkpoint_io import write_checkpoint
        from src.utils.checkpoint_paths import list_runs, select_auto_run_checkpoint
        from src.webui.components import training_console as tc

        root = tmp_path / "root"
        ws = _copy(demo_workspace, root / "demo")
        run = _run(ws, "gatv2", "20261020-000000_x_aaaaaa")
        write_checkpoint({"epoch": 1}, run / "last.pt")
        (ws / "runs" / "someone_elses" / "20261021-000000_y_bbbbbb").mkdir(parents=True)
        _refuse_iterdir(monkeypatch, "someone_elses")

        listing = sl.list_checkpoints("demo", environ=_roots(root))
        assert listing.available, listing.problem
        items = {i.filename: i for i in listing.items}
        assert items["last.pt"].problem is None and items["last.pt"].sha256
        unlisted = items["someone_elses"]
        assert unlisted.problem == ("runs/someone_elses cannot be listed (PermissionError: "
                                    "Permission denied), so no run in it is read")
        assert unlisted.sha256 is None and unlisted.record is None
        assert [r.name for r in list_runs(ws)] == [run.parent.name]
        # Selection serves the latest run it read and says what it did not
        # read (the review of C3, listing 1: it named an older run "the latest
        # run" and said nothing of the directory).
        not_read = []
        path, _key, reason = select_auto_run_checkpoint(ws, not_read=not_read)
        assert path == run / "last.pt"
        assert reason == (f"auto: latest run 'gatv2/{run.parent.name}'; not read: "
                          f"{unlisted.problem}")
        assert not_read == [unlisted.problem]
        assert select_auto_run_checkpoint(ws, accept=lambda p: False)[2] == (
            f"no run has a checkpoint; not read: {unlisted.problem}")
        assert tc._listed_runs(ws)[0] == f"gatv2/{run.parent.name}"
        with pytest.raises(sl.ServerListRefused, match="is listed"):
            sl.resolve_checkpoint("demo", unlisted.architecture, unlisted.run,
                                  unlisted.filename, environ=_roots(root))

        _refuse_iterdir(monkeypatch, "runs")
        refused = sl.list_checkpoints("demo", environ=_roots(root))
        assert refused.items == () and "the runs of the workspace 'demo' cannot be listed" in (
            refused.problem)

    def test_directories_that_cannot_be_listed_come_in_name_order(self, tmp_path, monkeypatch):
        """The review of C3 (listing nit): the entries for directories under
        runs/ that cannot be listed follow their names, not the filesystem's
        listing order (here the reverse of it)."""
        from src.utils.checkpoint_paths import read_run_checkpoints

        for name in ("zz", "aa", "mm"):
            (tmp_path / "runs" / name / "20261001-000000_r_111111").mkdir(parents=True)
        original = Path.iterdir

        def reversed_runs(self):
            if self.name == "runs":
                return iter(sorted(original(self), reverse=True))
            return original(self)

        monkeypatch.setattr(Path, "iterdir", reversed_runs)
        _refuse_iterdir(monkeypatch, "zz")
        _refuse_iterdir(monkeypatch, "aa")
        entries = read_run_checkpoints(tmp_path, within=tmp_path)
        assert [(e.path.name, e.problem is None) for e in entries] == [
            ("aa", False), ("zz", False)], "mm holds no checkpoint, so it lists none"

    def test_a_directory_under_runs_whose_name_is_not_text_is_one_entry_and_every_list_answers(
            self, demo_workspace, tmp_path, monkeypatch):
        """Checkpoint B (C3, 11_nontext_probe.txt): since every directory under
        runs/ is listed, one whose name is not valid text (a byte that is not
        UTF-8, which Python reads as a lone surrogate no answer can encode)
        made ``GET /workspaces/<ws>/checkpoints`` answer HTTP 500, every other
        file with it, where af3a8a9 ignored that directory. It now follows
        C3's rule for a directory under runs/ that cannot be listed: one entry,
        named with backslash escapes, saying why; nothing in it is read or
        selected or resolved; every other run is listed, and the server's
        list, its family view, the training service's list, their routes and
        the WebUI's lists answer."""
        import json
        import os
        from datetime import datetime

        from fastapi.testclient import TestClient

        from src.api.main import app
        from src.api.routes import training as training_routes
        from src.api.services.training_manager import TrainingManager
        from src.utils.checkpoint_io import write_checkpoint
        from src.utils.checkpoint_paths import read_run_checkpoints, select_auto_run_checkpoint
        from src.webui.components import model_list as ml
        from src.webui.components import training_console as tc
        from tests.fixtures.operator_access import (
            OPERATOR_BASE_URL,
            operator_mode,
            operator_request,
        )

        root = tmp_path / "root"
        ws = _copy(demo_workspace, root / "demo")
        run = _run(ws, "gatv2", "20261020-000000_x_aaaaaa")
        write_checkpoint({"epoch": 1}, run / "last.pt")
        os.utime(run / "last.pt", (1_700_000_000, 1_700_000_000))
        bad = os.fsdecode(b"bad\xff")
        hidden = _run(ws, bad, "20261021-000000_y_bbbbbb")
        write_checkpoint({"epoch": 2}, hidden / "last.pt")  # newer: the latest, were it read
        problem = (r"runs/bad\xff cannot be listed (its name is not valid text, shown here with "
                   r"backslash escapes; it cannot be opened by name), so no run in it is read")

        operator_mode(monkeypatch)
        monkeypatch.setenv(ss.ENV_WORKSPACES_ROOT, str(root))
        monkeypatch.setattr(training_routes.training_manager, "_config", {"data_dir": str(ws)})
        client = TestClient(app, base_url=OPERATOR_BASE_URL, raise_server_exceptions=False)
        for route, rows in (("/api/v1/workspaces/demo/checkpoints", "checkpoints"),
                            ("/api/v1/workspaces/demo/family", "rows"),
                            ("/api/v1/training/checkpoints", None)):
            response = client.get(route)  # 500 for the first, before the fix
            assert response.status_code == 200, (route, response.text[:300])
            listed = response.json() if rows is None else response.json()[rows]
            assert len(listed) == 2, (route, listed)
            unlisted = listed[1].get("item", listed[1])
            assert problem in unlisted.values(), (route, unlisted)

        def answers(value):  # what a JSON answer does with it: UTF-8 text
            return json.loads(json.dumps(value, ensure_ascii=False).encode("utf-8"))

        entries = read_run_checkpoints(ws, within=ws)
        assert [(e.names[1], e.names[2], e.problem) for e in entries] == [
            (run.parent.name, "last.pt", None), (r"bad\xff", r"bad\xff", problem)]
        # Its identity is its own name under runs/, by its bytes, apart from
        # its shown name (BACKLOG 66).
        assert entries[1].key == "/626164ff" and entries[1].run_key == "/626164ff"
        not_read = []
        path, _key, reason = select_auto_run_checkpoint(ws, not_read=not_read)
        assert path == run / "last.pt" and not_read == [problem]
        assert answers(reason) == f"auto: latest run 'gatv2/{run.parent.name}'; not read: {problem}"
        assert tc._listed_runs(ws)[0] == f"gatv2/{run.parent.name}"
        with pytest.raises(sl.ServerListRefused, match="is listed"):
            sl.resolve_checkpoint("demo", bad, hidden.parent.name, "last.pt", environ=_roots(root))

        listing = sl.list_checkpoints("demo", environ=_roots(root))
        assert listing.available, listing.problem
        items = answers(listing.as_dict("checkpoints"))["checkpoints"]
        assert [(i["architecture"], i["run"], i["filename"], i["problem"]) for i in items] == [
            ("gatv2", run.parent.name, "last.pt", None),
            ("runs", r"bad\xff", r"bad\xff", problem)]
        assert items[0]["sha256"] == file_sha256(run / "last.pt") and items[1]["sha256"] is None

        training = answers(TrainingManager().get_checkpoints(str(ws)))
        assert [(t["run"], t["filename"], t["metadata_problem"]) for t in training] == [
            (run.parent.name, "last.pt", None), (r"bad\xff", r"bad\xff", problem)]
        # Its time is the directory's own, whatever its name (the guides say so).
        assert training[1]["modified"] == datetime.fromtimestamp(
            hidden.parent.parent.stat().st_mtime).isoformat()

        monkeypatch.setattr(ml, "read_status", lambda: (None, "not read"))
        listing_ui = ml.list_models("demo", operator_request(monkeypatch), family=True)
        assert [i["run"] for i in answers(listing_ui["items"])] == [run.parent.name, r"bad\xff"]
        assert answers(listing_ui["family"])
        assert r"bad\xff" in ml.list_html(listing_ui, None, ml.PRESETS["research"])
        assert list(answers(tc.checkpoint_items(listing)[0])) == [
            f"gatv2/{run.parent.name}/last.pt", "/626164ff"]

    def test_a_run_or_file_whose_name_is_not_text_is_listed_and_every_list_answers(
            self, demo_workspace, tmp_path, monkeypatch):
        """BACKLOG 66 (the run and file cases; the directory case above): a
        run directory or a checkpoint file whose name is not valid text made
        ``GET /workspaces/<ws>/checkpoints`` and ``GET /training/checkpoints``
        answer HTTP 500 (UnicodeEncodeError), every other file with it. At the
        list root each is one entry with its state, shown with backslash
        escapes, identified by its bytes, never opened, chosen or resolved;
        every other file is listed and read; the server's list, its family
        view, the training service's list, their routes and the WebUI lists
        answer."""
        import json
        import os

        import src.utils.checkpoint_io as checkpoint_io
        from fastapi.testclient import TestClient

        from src.api.main import app
        from src.api.routes import training as training_routes
        from src.api.services.training_manager import TrainingManager
        from src.utils.checkpoint_io import write_checkpoint
        from src.utils.checkpoint_paths import read_run_checkpoints, select_auto_run_checkpoint
        from src.webui.components import model_list as ml
        from src.webui.components import test_page as tp
        from src.webui.components import training_console as tc
        from tests.fixtures.operator_access import (
            OPERATOR_BASE_URL,
            operator_mode,
            operator_request,
        )

        root = tmp_path / "root"
        ws = _copy(demo_workspace, root / "demo")
        older = _run(ws, "gatv2", "20261019-000000_x_aaaaaa")
        write_checkpoint({"epoch": 1}, older / "last.pt")
        os.utime(older / "last.pt", (1_700_000_000, 1_700_000_000))
        newer = _run(ws, "gatv2", "20261020-000000_z_cccccc")
        write_checkpoint({"epoch": 2}, newer / "last.pt")
        os.utime(newer / "last.pt", (1_700_000_100, 1_700_000_100))
        bad_file = newer / os.fsdecode(b"bad\xff.pt")
        write_checkpoint({"epoch": 9}, bad_file)  # the newest file, were it read
        bad_run = _run(ws, "gatv2", os.fsdecode(b"20261021-000000_y_\xff"))
        write_checkpoint({"epoch": 3}, bad_run / "last.pt")
        why = "its name is not valid text, shown here with backslash escapes; it cannot be " \
              "opened by name"
        run_problem = (rf"runs/gatv2/20261021-000000_y_\xff cannot be listed ({why}), so no "
                       "file in it is read")
        file_problem = (rf"runs/gatv2/{newer.parent.name}/checkpoints/bad\xff.pt is not listed "
                        f"({why}), so it is not read")

        original, reads = checkpoint_io.read_checkpoint, []

        def counting(path, **kwargs):
            reads.append(Path(path).name)
            return original(path, **kwargs)

        monkeypatch.setattr(checkpoint_io, "read_checkpoint", counting)

        def answers(value):  # what a JSON answer does with it: UTF-8 text
            return json.loads(json.dumps(value, ensure_ascii=False).encode("utf-8"))

        entries = read_run_checkpoints(ws, within=ws)
        assert [(e.names, e.problem) for e in entries] == [
            (("gatv2", newer.parent.name, "last.pt"), None),
            (("gatv2", older.parent.name, "last.pt"), None),
            (("gatv2", r"20261021-000000_y_\xff", r"20261021-000000_y_\xff"), run_problem),
            (("gatv2", newer.parent.name, r"bad\xff.pt"), file_problem)]
        assert reads == ["last.pt", "last.pt"], "a name that is not text was opened"
        assert [e.key for e in entries][2:] == [
            "gatv2//" + b"20261021-000000_y_\xff".hex(),
            f"gatv2/{newer.parent.name}//" + b"bad\xff.pt".hex()]
        assert [e.run_key for e in entries][2:] == [
            "gatv2//" + b"20261021-000000_y_\xff".hex(), f"gatv2/{newer.parent.name}"]
        assert all(e.sha256 is None for e in entries[2:])

        not_read = []
        path, _key, reason = select_auto_run_checkpoint(ws, not_read=not_read)
        assert path == newer / "last.pt" and not_read == [run_problem, file_problem]
        assert answers(reason) == (f"auto: latest run 'gatv2/{newer.parent.name}'; not read: "
                                   f"{run_problem}; {file_problem}")
        assert tc._listed_runs(ws)[0] == f"gatv2/{newer.parent.name}"
        for run, filename in ((r"20261021-000000_y_\xff", "last.pt"),
                              (newer.parent.name, r"bad\xff.pt")):
            with pytest.raises(sl.ServerListRefused, match="is listed"):
                sl.resolve_checkpoint("demo", "gatv2", run, filename, environ=_roots(root))

        operator_mode(monkeypatch)
        monkeypatch.setenv(ss.ENV_WORKSPACES_ROOT, str(root))
        monkeypatch.setattr(training_routes.training_manager, "_config", {"data_dir": str(ws)})
        client = TestClient(app, base_url=OPERATOR_BASE_URL, raise_server_exceptions=False)
        for route, rows in (("/api/v1/workspaces/demo/checkpoints", "checkpoints"),
                            ("/api/v1/workspaces/demo/family", "rows"),
                            ("/api/v1/training/checkpoints", None)):
            response = client.get(route)  # HTTP 500 for the first and the last, before
            assert response.status_code == 200, (route, response.text[:300])
            listed = response.json() if rows is None else response.json()[rows]
            listed = [row.get("item", row) for row in listed]
            assert len(listed) == 4, (route, listed)
            assert sorted(row["key"] for row in listed) == sorted(e.key for e in entries)
            for row, problem in ((listed[2], run_problem), (listed[3], file_problem)):
                assert problem in row.values(), (route, row)

        training = answers(TrainingManager().get_checkpoints(str(ws)))
        assert [(t["run"], t["filename"], t["metadata_problem"]) for t in training][2:] == [
            (r"20261021-000000_y_\xff", r"20261021-000000_y_\xff", run_problem),
            (newer.parent.name, r"bad\xff.pt", file_problem)]
        assert training[3]["filepath"].endswith(r"/checkpoints/bad\xff.pt")

        monkeypatch.setattr(ml, "read_status", lambda: (None, "not read"))
        listing_ui = ml.list_models("demo", operator_request(monkeypatch), family=True)
        items = answers(listing_ui["items"])
        assert [ml.item_key(i) for i in items] == [e.key for e in entries]
        html = ml.list_html(listing_ui, None, ml.PRESETS["research"])
        assert r"20261021-000000_y_\xff" in html and r"bad\xff.pt" in html
        assert answers(listing_ui["family"])
        assert len(answers(tc.checkpoint_items(sl.list_checkpoints("demo"))[0])) == 4
        assert len(answers(sorted(_checkpoint_values(listing_ui)))) == 4

    def test_names_shown_alike_keep_their_own_identity_and_selection(
            self, demo_workspace, tmp_path, monkeypatch):
        """BACKLOG 66, the collision: a run and a file whose names are text
        and read ``bad\\xff`` literally (a backslash, "x", "f", "f"), beside a
        run and a file whose names are the byte ``\\xff``, shown the same way.
        Each keeps its own identity (`CheckpointItem.key`), so each list holds
        all four and none hides another; selecting a key finds that entry and
        never the other (the WebUI's shared list, the Test page's model select,
        the console's items, the family view); a request by the shown names
        resolves the text-named file only."""
        import json
        import os

        from src.utils.checkpoint_io import write_checkpoint
        from src.utils.fingerprint import file_sha256
        from src.webui.components import model_list as ml
        from src.webui.components import test_page as tp
        from src.webui.components import training_console as tc
        from tests.fixtures.operator_access import operator_request

        root = tmp_path / "root"
        ws = _copy(demo_workspace, root / "demo")
        literal = _run(ws, "gatv2", r"bad\xff")
        write_checkpoint({"epoch": 1}, literal / "last.pt")
        write_checkpoint({"epoch": 2}, literal / r"bad\xff.pt")
        write_checkpoint({"epoch": 3}, literal / os.fsdecode(b"bad\xff.pt"))
        write_checkpoint({"epoch": 4}, _run(ws, "gatv2", os.fsdecode(b"bad\xff")) / "last.pt")

        listing = sl.list_checkpoints("demo", environ=_roots(root))
        items = json.loads(json.dumps(listing.as_dict("checkpoints"),
                                      ensure_ascii=False).encode("utf-8"))["checkpoints"]
        shown = [(i["architecture"], i["run"], i["filename"]) for i in items]
        assert sorted(shown) == sorted([
            ("gatv2", r"bad\xff", "last.pt"), ("gatv2", r"bad\xff", r"bad\xff.pt"),
            ("gatv2", r"bad\xff", r"bad\xff.pt"), ("gatv2", r"bad\xff", r"bad\xff")])
        keys = [i["key"] for i in items]
        assert len(set(keys)) == 4, keys
        text_file = next(i for i in items if i["key"] == r"gatv2/bad\xff/bad\xff.pt")
        other_file = next(i for i in items if i["key"] == "gatv2/bad\\xff//"
                          + b"bad\xff.pt".hex())
        assert text_file["sha256"] == file_sha256(literal / r"bad\xff.pt")
        assert text_file["problem"] is None
        assert other_file["sha256"] is None and "not valid text" in other_file["problem"]
        assert (text_file["filename"], other_file["filename"]) == (r"bad\xff.pt", r"bad\xff.pt")

        # Selecting one never picks the other, wherever a key selects.
        monkeypatch.setattr(ml, "read_status", lambda: (None, "not read"))
        listing_ui = ml.list_models("demo", operator_request(monkeypatch), family=True,
                                    environ=_roots(root))
        assert sorted(ml.item_key(i) for i in listing_ui["items"]) == sorted(keys)
        for item in items:
            assert ml.find_item(listing_ui, item["key"])["sha256"] == item["sha256"]
            assert tp.find_model(listing_ui, item["key"])["problem"] == item["problem"]
        assert len(_checkpoint_values(listing_ui)) == 4
        console = tc.checkpoint_items(listing)[0]
        assert sorted(console) == sorted(keys)
        assert console[text_file["key"]]["sha256"] == text_file["sha256"]
        family = [row["key"] for view in listing_ui["family"].values() for row in view]
        assert sorted(set(family)) == sorted(keys)
        assert sl.resolve_checkpoint("demo", "gatv2", r"bad\xff", r"bad\xff.pt",
                                     environ=_roots(root)) == literal / r"bad\xff.pt"

        # The two runs shown alike are two runs wherever a list groups by run
        # (the review of Stage 7): the Models list draws two run sections, each
        # with its own files, and counts two runs; the console groups them apart.
        assert sorted(i["run_key"] for i in items) == sorted(
            ["gatv2/bad\\xff"] * 3 + ["gatv2//" + b"bad\xff".hex()])
        groups = ml._runs_in_order(listing_ui["items"])
        assert sorted((label, len(group)) for label, group in groups) == [
            (r"gatv2/bad\xff", 1), (r"gatv2/bad\xff", 3)]
        by_run = ml.Preset("by_run", None, "In use", layout="rows")  # every row, by run
        html = ml.list_html(listing_ui, None, by_run)
        # Each group's heading, outside its list (which names it in its
        # aria-label too, since B0a).
        assert html.count(r"<b>Run gatv2/bad\xff</b>") == 2, html
        assert sorted(len(v) for v in ml.runs_of(console).values()) == [1, 3]

    def test_a_directory_entry_never_shares_a_key_with_a_run_or_file_named_alike(
            self, tmp_path, monkeypatch):
        """The review of Stage 7 (BACKLOG 66): an entry is identified by its
        own names under runs/, one for a directory directly under it, two for
        a run directory, three for a file. A directory under runs/ that cannot
        be listed was keyed by three names, ``runs/X/X``, the key of a run or
        a file in an architecture directory literally named ``runs``: two
        entries shared it, so the console's items dropped one and selecting
        the second found the first. Each now has its own."""
        import os

        from src.utils.checkpoint_io import write_checkpoint
        from src.utils.checkpoint_paths import read_run_checkpoints

        # A name that is not text: runs/<bad> beside runs/runs/<bad>/ (a run).
        bad = os.fsdecode(b"bad\xff")
        ws = tmp_path / "bytes"
        (ws / "runs" / bad).mkdir(parents=True)
        write_checkpoint({"epoch": 1}, _run(ws, "runs", bad) / "last.pt")
        write_checkpoint({"epoch": 2}, _run(ws, "gatv2", "20261020-000000_z_cccccc") / "last.pt")
        entries = read_run_checkpoints(ws, within=ws)
        assert [(e.key, "no run" in e.problem if e.problem else None) for e in entries] == [
            ("gatv2/20261020-000000_z_cccccc/last.pt", None),
            ("/626164ff", True), ("runs//626164ff", False)]
        assert len({e.run_key for e in entries}) == 3

        # Names that are text: runs/x.pt (a directory this account may not
        # enter) beside runs/runs/x.pt/checkpoints/x.pt, a file that is read.
        ws = tmp_path / "text"
        (ws / "runs" / "x.pt").mkdir(parents=True)
        write_checkpoint({"epoch": 1}, _run(ws, "runs", "x.pt") / "x.pt")
        locked = ws / "runs" / "x.pt"
        original = Path.iterdir

        def iterdir(self):
            if self == locked:
                raise PermissionError(13, "Permission denied", str(self))
            return original(self)

        monkeypatch.setattr(Path, "iterdir", iterdir)
        entries = read_run_checkpoints(ws, within=ws)
        assert [(e.key, e.sha256 is not None) for e in entries] == [
            ("runs/x.pt/x.pt", True), ("x.pt", False)]
        assert entries[1].problem.startswith("runs/x.pt cannot be listed (PermissionError")

    def test_the_latest_run_is_one_inside_the_workspace(self, tmp_path):
        """The review of C3 (listing nit): the Training page's latest run is
        the newest run inside the workspace, as `resolve_run` finds one; a
        newer run behind a link out of it, in any directory under runs/, is
        not it (it named that run, which `resolve_run` refuses, so the page
        showed no run), nor among the runs its Viewing select offers."""
        import os

        from src.webui.components import training_console as tc

        ws = tmp_path / "ws"
        inside = _run(ws, "gatv2", "20261020-000000_x_aaaaaa")
        (inside / "last.pt").write_bytes(b"x")
        os.utime(inside / "last.pt", (1_700_000_000, 1_700_000_000))
        far = tmp_path / "outside" / "20261021-000000_y_bbbbbb" / "checkpoints"
        far.mkdir(parents=True)
        (far / "last.pt").write_bytes(b"y")
        os.utime(far / "last.pt", (1_700_000_900, 1_700_000_900))
        (ws / "runs" / "elsewhere").symlink_to(far.parent.parent, target_is_directory=True)

        assert tc._listed_runs(ws) == [f"gatv2/{inside.parent.name}"]

    def test_the_training_services_list_never_reads_a_run_linked_out(
            self, demo_workspace, tmp_path, monkeypatch):
        """N7 (C4): the training service's list (GET /training/checkpoints)
        is bound to its workspace as the server's list is: a run directory
        linked out of it is listed with why, its file never opened."""
        import src.utils.checkpoint_io as checkpoint_io
        from src.api.services.training_manager import TrainingManager
        from src.utils.checkpoint_io import write_checkpoint

        root = tmp_path / "root"
        ws = _copy(demo_workspace, root / "demo")
        far = tmp_path / "outside" / "20261009-000000_far_ffffff" / "checkpoints"
        far.mkdir(parents=True)
        write_checkpoint({"epoch": 9, "logs": {"val_pooled_mrr": 0.9}}, far / "secret.pt")
        (ws / "runs" / "gatv2").mkdir(parents=True)
        (ws / "runs" / "gatv2" / far.parent.name).symlink_to(far.parent,
                                                             target_is_directory=True)
        original, reads = checkpoint_io.read_checkpoint, []

        def counting(path, **kwargs):
            reads.append(Path(path).resolve())
            return original(path, **kwargs)

        monkeypatch.setattr(checkpoint_io, "read_checkpoint", counting)
        (entry,) = TrainingManager().get_checkpoints(str(ws))
        assert reads == []
        assert entry["filename"] == "secret.pt" and entry["sha256"] is None
        assert "resolves outside the workspace, so it is not read" in entry["metadata_problem"]
        assert entry.get("metrics") is None and entry.get("record") is None


# ---------------------------------------------------------------------------
# Cohort versions
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def registered(tmp_path_factory):
    from tests.fixtures.registered_cohort import build_registered_cohort

    return build_registered_cohort(tmp_path_factory.mktemp("registered"))


def _registry_copy(registered, tmp_path):
    from tests.fixtures.registered_cohort import copy_registered_cohort

    return copy_registered_cohort(registered, tmp_path / "copy")


class TestCohortVersions:

    def test_no_registry_no_list_and_why(self):
        listing = sl.list_cohort_versions(environ={})
        assert listing.items == () and "SHEPHERD_REGISTRY is not set" in listing.problem

    def test_a_registry_inside_the_repository_is_refused(self):
        listing = sl.list_cohort_versions(
            environ={ss.ENV_REGISTRY: str(ss.REPO_ROOT / "data")})
        assert listing.items == () and "inside the repository" in listing.problem

    def test_a_version_by_label_and_digest_with_aggregates_only(self, registered, tmp_path):
        cohort = _registry_copy(registered, tmp_path)
        listing = sl.list_cohort_versions(environ={ss.ENV_REGISTRY: str(cohort.registry)})
        (item,) = listing.items
        manifest_path = cohort.registry / "versions" / item.label / "cohort.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert item.label == manifest["label"] and item.role == manifest["role"]
        assert item.manifest_sha256 == file_sha256(manifest_path)
        assert item.counts == manifest["counts"] and item.graph == manifest["graph"]
        assert item.dataset_record == manifest["dataset_record"]
        assert item.source_format == "shepherd-cases/1"
        assert item.source_sha256 == manifest["source"]["sha256"]
        assert item.imported_at == manifest["imported"]["at"]
        assert item.problem is None and item.reports == ()
        # D7: no case identifier, and no exclusion list, leaves the machine.
        dumped = json.dumps(listing.as_dict("versions"))
        assert manifest["exclusions"]
        for exclusion in manifest["exclusions"]:
            assert exclusion["case_id"] not in dumped
        assert "exclusions" not in dumped and str(tmp_path) not in dumped

    def test_a_version_carries_no_ordering_field_and_the_retired_setting_changes_nothing(
            self, registered, tmp_path):
        """K8-6d: GET /cohorts' items carry no ``offers_ordering``, and a
        deployment that still sets SHEPHERD_ORDERING_ROLES lists the same items
        (the setting is not read)."""
        cohort = _registry_copy(registered, tmp_path)
        role = cohort.version.manifest["role"]
        base = {ss.ENV_REGISTRY: str(cohort.registry)}
        without = sl.list_cohort_versions(environ=base)
        assert sl.list_cohort_versions(environ={**base, "SHEPHERD_ORDERING_ROLES": role}) == without
        (item,) = without.as_dict("versions")["versions"]
        assert "offers_ordering" not in item and item["role"] == role

    def test_an_interrupted_import_is_listed_with_its_reason(self, registered, tmp_path):
        cohort = _registry_copy(registered, tmp_path)
        (cohort.registry / "versions" / "half-done").mkdir()
        items = {item.label: item
                 for item in sl.list_cohort_versions(
                     environ={ss.ENV_REGISTRY: str(cohort.registry)}).items}
        assert items["half-done"].manifest_sha256 is None
        assert "interrupted import" in items["half-done"].problem
        assert items[cohort.version.manifest["label"]].problem is None

    def test_runs_take_their_state_only_from_the_reader(self, registered, tmp_path):
        cohort = _registry_copy(registered, tmp_path)
        reports = cohort.version.directory / "reports"
        older = "20261019T120000Z-aaaaaaaaaaaa-00000001"
        newer = "20261020T120000Z-bbbbbbbbbbbb-00000002"
        for run in (older, newer, "notes"):
            (reports / run).mkdir(parents=True)
        environ = {ss.ENV_REGISTRY: str(cohort.registry)}

        (unread,) = sl.list_cohort_versions(environ=environ).items
        assert [r.run_id for r in unread.reports] == ["notes", newer, older]
        assert all(r.state is None for r in unread.reports)
        assert "not named as a run id" in unread.reports[0].problem
        assert "not read here" in unread.reports[1].problem

        seen = []

        def reader(directory):
            seen.append(directory)
            if directory.name == older:
                raise OSError("cannot read")
            return {"kept": True, "registered": False}

        (read,) = sl.list_cohort_versions(environ=environ, report_state=reader).items
        assert seen == [(reports / newer).resolve(), (reports / older).resolve()]
        states = {r.run_id: r for r in read.reports}
        assert states[newer].state == {"kept": True, "registered": False}
        assert states[newer].problem is None
        assert states[older].state is None and "could not be read" in states[older].problem

    def test_a_run_linked_out_of_the_store_is_refused(self, registered, tmp_path):
        cohort = _registry_copy(registered, tmp_path)
        reports = cohort.version.directory / "reports"
        reports.mkdir()
        (tmp_path / "elsewhere").mkdir()
        run = "20261020T120000Z-bbbbbbbbbbbb-00000002"
        (reports / run).symlink_to(tmp_path / "elsewhere")
        (item,) = sl.list_cohort_versions(
            environ={ss.ENV_REGISTRY: str(cohort.registry)},
            report_state=lambda d: pytest.fail("a run outside the store was read")).items
        (listed,) = item.reports
        assert listed.state is None and "outside" in listed.problem


def _refuse_iterdir(monkeypatch, name):
    """`iterdir` on a directory named `name` raises EACCES, as a directory
    another account made with `mkdir -m 700` does."""
    original = Path.iterdir

    def iterdir(self):
        if self.name == name:
            raise PermissionError(13, "Permission denied", str(self))
        return original(self)

    monkeypatch.setattr(Path, "iterdir", iterdir)


class TestUnreadableRegistryDirectories:
    """One directory the service account may not read never fails the whole
    cohort list with a bare error (M5-1 review)."""

    def test_a_report_store_that_cannot_be_listed_is_that_versions_problem(
            self, registered, tmp_path, monkeypatch):
        cohort = _registry_copy(registered, tmp_path)
        (cohort.version.directory / "reports").mkdir()
        _refuse_iterdir(monkeypatch, "reports")
        listing = sl.list_cohort_versions(environ={ss.ENV_REGISTRY: str(cohort.registry)})
        assert listing.available
        (item,) = listing.items
        assert item.problem is None and item.manifest_sha256 is not None
        assert item.reports == ()
        assert "reports cannot be listed (PermissionError: Permission denied)" in (
            item.reports_problem)

    def test_versions_that_cannot_be_listed_say_why(self, registered, tmp_path, monkeypatch):
        cohort = _registry_copy(registered, tmp_path)
        _refuse_iterdir(monkeypatch, "versions")
        listing = sl.list_cohort_versions(environ={ss.ENV_REGISTRY: str(cohort.registry)})
        assert listing.items == ()
        assert "versions cannot be listed (PermissionError" in listing.problem
        assert "No version was read." in listing.problem

    def test_a_version_that_cannot_be_checked_is_listed_with_why(
            self, registered, tmp_path, monkeypatch):
        import src.evaluation.cohort_registry as registry_module

        cohort = _registry_copy(registered, tmp_path)
        (cohort.registry / "versions" / "locked").mkdir()
        original = registry_module.resolve_version

        def resolve(registry, label):
            if label == "locked":
                raise PermissionError(13, "Permission denied", label)
            return original(registry, label)

        monkeypatch.setattr(registry_module, "resolve_version", resolve)
        items = {item.label: item for item in sl.list_cohort_versions(
            environ={ss.ENV_REGISTRY: str(cohort.registry)}).items}
        assert items[cohort.version.manifest["label"]].problem is None
        assert "cannot be read (PermissionError: Permission denied)" in items["locked"].problem

    def test_a_run_that_cannot_be_checked_is_listed_with_why(
            self, registered, tmp_path, monkeypatch):
        from src.evaluation.cohort_registry import list_report_runs

        cohort = _registry_copy(registered, tmp_path)
        run = "20261020T120000Z-bbbbbbbbbbbb-00000002"
        target = (cohort.version.directory / "reports" / run)
        target.mkdir(parents=True)
        resolved = target.resolve()
        original = Path.is_dir

        def is_dir(self, *args, **kwargs):
            if self == resolved:
                raise PermissionError(13, "Permission denied", str(self))
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Path, "is_dir", is_dir)
        (listed,) = list_report_runs(cohort.version)
        assert listed.directory is None
        assert "cannot be checked (PermissionError: Permission denied)" in listed.problem


class TestTheRegistryListing:
    """`list_versions` and `list_report_runs` in the registry module."""

    def test_list_versions_reads_each_manifest_through_resolve_version(
            self, registered, tmp_path, monkeypatch):
        import src.evaluation.cohort_registry as registry_module

        cohort = _registry_copy(registered, tmp_path)
        calls = []
        original = registry_module.resolve_version

        def counting(registry, label):
            calls.append(label)
            return original(registry, label)

        monkeypatch.setattr(registry_module, "resolve_version", counting)
        listed = registry_module.list_versions(cohort.registry)
        assert [entry.label for entry in listed] == calls == [cohort.version.manifest["label"]]

    def test_an_empty_registry_lists_nothing(self, tmp_path):
        from src.evaluation.cohort_registry import list_versions

        assert list_versions(tmp_path) == []

    def test_a_version_without_reports_lists_no_run(self, registered, tmp_path):
        from src.evaluation.cohort_registry import list_report_runs

        cohort = _registry_copy(registered, tmp_path)
        assert list_report_runs(cohort.version) == []

    def test_the_stores_own_files_are_not_runs_and_are_not_listed(self, registered, tmp_path):
        """The usage history and the investigation notes live beside the run
        directories (`report_store`): the listing leaves them out, so no
        caller (the API's version list, the Test page) has to know the store's
        layout to skip them; anything else that is not a run is still listed
        with why."""
        from datetime import UTC, datetime

        from src.evaluation.cohort_registry import create_report_dir, list_report_runs
        from src.evaluation.report_store import NOTES_FILE, STORE_FILES, USAGE_FILE

        assert set(STORE_FILES) == {USAGE_FILE, NOTES_FILE}
        cohort = _registry_copy(registered, tmp_path)
        run_id, _directory = create_report_dir(cohort.version, "a" * 64,
                                               datetime(2026, 10, 20, tzinfo=UTC))
        reports = cohort.version.directory / "reports"
        for name in (USAGE_FILE, NOTES_FILE, "stray.txt"):
            (reports / name).write_text("{}\n")
        listed = list_report_runs(cohort.version)
        assert [run.run_id for run in listed] == ["stray.txt", run_id]
        assert "not named as a run id" in listed[0].problem
        (item,) = sl.list_cohort_versions(
            environ={ss.ENV_REGISTRY: str(cohort.registry)}).items
        assert [r.run_id for r in item.reports] == ["stray.txt", run_id]

    def test_the_stores_lock_files_are_not_runs_and_are_not_listed(self, registered, tmp_path):
        """F-k (K8-6r.1): each append to the usage history or the notes holds
        a lock file beside it (`report_store.STORE_LOCK_FILES`), kept after
        use. The listing leaves them out with the store's data files, so
        neither the API's version list nor the Test page shows them as runs;
        a stray file is still listed with why."""
        from datetime import UTC, datetime

        from src.evaluation.cohort_registry import create_report_dir, list_report_runs
        from src.evaluation.report_store import STORE_LOCK_FILES

        assert STORE_LOCK_FILES == ("usage.jsonl.lock", "notes.jsonl.lock")
        cohort = _registry_copy(registered, tmp_path)
        run_id, _directory = create_report_dir(cohort.version, "a" * 64,
                                               datetime(2026, 10, 20, tzinfo=UTC))
        reports = cohort.version.directory / "reports"
        for name in (*STORE_LOCK_FILES, "stray.txt"):
            (reports / name).write_text("")
        listed = list_report_runs(cohort.version)
        assert [run.run_id for run in listed] == ["stray.txt", run_id]
        assert "not named as a run id" in listed[0].problem
        (item,) = sl.list_cohort_versions(
            environ={ss.ENV_REGISTRY: str(cohort.registry)}).items
        assert [r.run_id for r in item.reports] == ["stray.txt", run_id]

    def test_the_run_id_pattern_matches_what_create_report_dir_makes(self, registered,
                                                                     tmp_path):
        from datetime import datetime, timezone

        from src.evaluation.cohort_registry import create_report_dir, list_report_runs

        cohort = _registry_copy(registered, tmp_path)
        run_id, directory = create_report_dir(cohort.version, "a" * 64,
                                              datetime(2026, 10, 20, tzinfo=timezone.utc))
        (listed,) = list_report_runs(cohort.version)
        assert (listed.run_id, listed.directory, listed.problem) == (run_id, directory, None)


# ---------------------------------------------------------------------------
# The routes
# ---------------------------------------------------------------------------
def _lists_app():
    from fastapi import FastAPI

    from src.api.routes import lists

    app = FastAPI()
    app.include_router(lists.router, prefix="/api/v1")
    return app


@pytest.fixture
def client():
    """Starlette's default client: its requests arrive on ``testserver``, not
    loopback, so D8 refuses an operator route whatever the mode."""
    from fastapi.testclient import TestClient

    return TestClient(_lists_app())


@pytest.fixture
def operator_client(monkeypatch):
    """A client D8 allows: mode M1, loopback arrival and Host, no Origin."""
    from fastapi.testclient import TestClient

    from tests.fixtures.operator_access import OPERATOR_BASE_URL, operator_mode

    operator_mode(monkeypatch)
    return TestClient(_lists_app(), base_url=OPERATOR_BASE_URL)


class TestRoutes:

    def test_workspaces_without_a_root(self, client, monkeypatch):
        monkeypatch.delenv(ss.ENV_WORKSPACES_ROOT, raising=False)
        response = client.get("/api/v1/workspaces")
        assert response.status_code == 200
        assert response.json()["workspaces"] == []
        assert "SHEPHERD_WORKSPACES_ROOT is not set" in response.json()["problem"]

    def test_workspaces(self, client, checkpoint_root, monkeypatch):
        root, ws, _ = checkpoint_root
        monkeypatch.setenv(ss.ENV_WORKSPACES_ROOT, str(root))
        body = client.get("/api/v1/workspaces").json()
        assert body["problem"] is None
        assert [w["name"] for w in body["workspaces"]] == ["demo"]
        assert body["workspaces"][0]["dataset_fingerprint"] == file_sha256(
            ws / "split_manifest.json")

    @pytest.mark.parametrize("mode, base_url, says", [
        (None, "http://127.0.0.1", "SHEPHERD_DEPLOYMENT_MODE is not set"),
        ("M3", "http://127.0.0.1", "not to M1"),
        ("M1", "http://192.0.2.10", "did not arrive on a loopback address"),
        ("M1", "http://testserver", "did not arrive on a loopback address"),
    ], ids=["mode-unset", "mode-m3", "lan-address", "not-an-address"])
    def test_the_checkpoint_route_is_refused_without_m1_on_loopback(
            self, checkpoint_root, monkeypatch, mode, base_url, says):
        """It opens every checkpoint (SPEC_4 C3), which D8 offers only where the
        operator controls are: refused with 403, naming the condition, and
        nothing is listed or opened."""
        from fastapi.testclient import TestClient

        from src.api.services import server_lists

        root, _, _ = checkpoint_root
        monkeypatch.setenv(ss.ENV_WORKSPACES_ROOT, str(root))
        if mode is not None:
            monkeypatch.setenv(ss.ENV_DEPLOYMENT_MODE, mode)
        monkeypatch.setattr(server_lists, "list_checkpoints",
                            lambda *a, **k: pytest.fail("a refused request listed checkpoints"))
        response = TestClient(_lists_app(), base_url=base_url).get(
            "/api/v1/workspaces/demo/checkpoints")
        assert response.status_code == 403
        assert says in response.json()["detail"] and "Nothing was done" in response.text

    def test_the_checkpoint_route_lists_for_an_operator_on_loopback(
            self, operator_client, checkpoint_root, monkeypatch):
        root, _, digests = checkpoint_root
        monkeypatch.setenv(ss.ENV_WORKSPACES_ROOT, str(root))
        body = operator_client.get("/api/v1/workspaces/demo/checkpoints",
                                   params={"architecture": "hgt"}).json()
        assert body["problem"] is None
        assert {c["filename"]: c["sha256"] for c in body["checkpoints"]} == {
            "best.pt": digests["trained"], "other.pt": digests["other"]}

    def test_the_handler_the_route_mounts(self, checkpoint_root, monkeypatch):
        from src.api.routes import lists

        root, _, digests = checkpoint_root
        monkeypatch.setenv(ss.ENV_WORKSPACES_ROOT, str(root))
        listed = lists.get_workspace_checkpoints("demo", "hgt")
        assert listed["problem"] is None
        assert {c["filename"]: c["sha256"] for c in listed["checkpoints"]} == {
            "best.pt": digests["trained"], "other.pt": digests["other"]}

    @pytest.mark.parametrize("name", ["nope", ".cache", "..", "../demo", "demo/"])
    def test_the_handler_takes_only_a_listed_workspace(self, checkpoint_root, monkeypatch,
                                                       name):
        from src.api.routes import lists

        root, _, _ = checkpoint_root
        monkeypatch.setenv(ss.ENV_WORKSPACES_ROOT, str(root))
        body = lists.get_workspace_checkpoints(name)
        assert body["checkpoints"] == [] and "is not a workspace" in body["problem"]

    def test_cohorts(self, client, registered, tmp_path, monkeypatch):
        cohort = _registry_copy(registered, tmp_path)
        monkeypatch.setenv(ss.ENV_REGISTRY, str(cohort.registry))
        body = client.get("/api/v1/cohorts").json()
        assert body["problem"] is None
        assert [v["label"] for v in body["versions"]] == [cohort.version.manifest["label"]]
        assert "offers_ordering" not in body["versions"][0]           # retired in K8-6d

    def test_the_cohorts_route_lists_only_runs_after_appends_took_their_locks(
            self, client, tmp_path, monkeypatch):
        """F-k (K8-6r.1), with the locks in place: after a ``test_run``
        append and an investigation note into one version, its ``reports/``
        holds both lock files, and GET /api/v1/cohorts lists the run alone."""
        import hashlib
        import json

        from src.evaluation.report_store import add_investigation_note, record_test_run
        from src.evaluation.sidecar import build_record, record_key
        from tests.fixtures.report_store import store_report, version_of
        from tests.unit.test_evaluation_sidecar import registered_report

        report = store_report(tmp_path / "registry", registered_report())
        version = version_of(report)
        record_test_run(version, run_id=report.parent.name, report_sha256="e" * 64,
                        checkpoint_sha256="c" * 64, metrics={"pooled_mrr": 0.5}, n_ranked=8,
                        deterministic=True, device="cpu", record_requested=False,
                        mode="disease")
        data = report.read_bytes()
        key = record_key(build_record(json.loads(data), hashlib.sha256(data).hexdigest()))
        add_investigation_note(version, key, who="ops", what="compared the environments")
        assert sorted(p.name for p in report.parents[1].iterdir()) == sorted(
            [report.parent.name, "notes.jsonl", "notes.jsonl.lock", "usage.jsonl",
             "usage.jsonl.lock"])
        monkeypatch.setenv(ss.ENV_REGISTRY, str(tmp_path / "registry"))
        (listed,) = client.get("/api/v1/cohorts").json()["versions"]
        assert [r["run_id"] for r in listed["reports"]] == [report.parent.name]
        assert [r["problem"] for r in listed["reports"]] == [None]

    def test_the_cohort_route_reads_each_runs_state_through_the_registration(
            self, client, registered, tmp_path, monkeypatch):
        """The registration's one helper, given a run directory (`stored_report_state`): a
        run directory with no report reads as not kept, never as a state the
        list supposed, and the route passes the helper rather than nothing."""
        cohort = _registry_copy(registered, tmp_path)
        run = "20261020T120000Z-bbbbbbbbbbbb-00000002"
        (cohort.version.directory / "reports" / run).mkdir(parents=True)
        monkeypatch.setenv(ss.ENV_REGISTRY, str(cohort.registry))
        (version,) = client.get("/api/v1/cohorts").json()["versions"]
        (listed,) = version["reports"]
        assert listed["problem"] is None
        assert listed["state"]["kept"] == "not_kept"
        assert listed["state"]["registered"] == "cannot_be_checked"
        assert "report.json" in listed["state"]["why_kept"]
        assert set(listed["state"]) == {"kept", "why_kept", "registered", "why_registered",
                                        "report_sha256", "sequence", "historical_overlap",
                                        "replaced_after_registration"}

    def test_a_kept_report_not_found_in_its_ledger_is_never_listed_as_not_registered(
            self, tmp_path):
        """The list knows only the checkpoint the run read; a registration
        filed beside another copy of the same weights is not looked for, so
        no record beside the read one is "cannot be checked", saying which
        ledger was read (review of M5-1 WP2)."""
        import hashlib

        from src.api.services.server_lists import stored_report_state
        from tests.fixtures.report_store import store_report
        from tests.unit.test_evaluation_sidecar import registered_report

        weights = tmp_path / "runs" / "r1" / "checkpoints" / "best.pt"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"the weights")
        report = registered_report(checkpoint_path=str(weights))
        report["manifest"]["artifact_digests"]["checkpoint"] = hashlib.sha256(
            b"the weights").hexdigest()
        stored = store_report(tmp_path / "registry", report)

        state = stored_report_state(stored.parent)
        assert (state["kept"], state["registered"]) == ("kept", "cannot_be_checked")
        assert "not registered beside the checkpoint this run read" in state["why_registered"]
        assert str(weights.parent.resolve() / "evaluations.json") in state["why_registered"]

    def test_the_cohort_list_stays_open(self, client, registered, tmp_path, monkeypatch):
        """Read-only: no D8 dependency, even for a request D8 refuses."""
        cohort = _registry_copy(registered, tmp_path)
        monkeypatch.setenv(ss.ENV_REGISTRY, str(cohort.registry))
        assert client.get("/api/v1/workspaces").status_code == 200
        assert client.get("/api/v1/cohorts").status_code == 200

    def test_the_lists_are_mounted_on_the_service(self):
        from src.api.main import app
        from src.api.operator_access import require_operator_access

        routes = {getattr(route, "path", None): route for route in app.routes}
        assert {"/api/v1/workspaces", "/api/v1/cohorts"} <= set(routes)
        gated = routes["/api/v1/workspaces/{name}/checkpoints"]
        assert [d.dependency for d in gated.dependencies] == [require_operator_access]
        for open_path in ("/api/v1/workspaces", "/api/v1/cohorts"):
            assert routes[open_path].dependencies == []


def _checkpoint_values(listing):
    """Every checkpoint the Test page's Run → Checkpoint pair offers for a
    listing, by its option's value (the listing's key): each run's options
    (`model_list.checkpoint_choices`, over `test_page.model_parents`)."""
    from src.webui.components import model_list as ml
    from src.webui.components import test_page as tp

    parents = tp.model_parents(listing)
    return {value for keys in ml.runs_of(parents).values()
            for _label, value in ml.checkpoint_choices(parents, keys)}
