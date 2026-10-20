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
  reader given.

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
        assert item.dataset_record["binding"] is None

    def test_a_workspace_built_before_build_records(self, demo_workspace, tmp_path):
        root = tmp_path / "root"
        ws = _copy(demo_workspace, root / "old")
        (ws / "workspace_build.json").unlink()
        (item,) = sl.list_workspaces(environ=_roots(root)).items
        assert item.trainable  # a missing build record is descriptive only
        assert item.build == {
            "state": "not_recorded", "built_at": None, "code_revision": None,
            "reason": ("no workspace_build.json: the workspace was built before build "
                       "times were recorded")}

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
    """A root with one workspace holding: a checkpoint trained on it, one
    recording another dataset, one with no validation figure, one torch's safe
    loader refuses, and one that is not a checkpoint at all."""
    import torch

    from src.utils.checkpoint_io import write_checkpoint
    from tests.fixtures.registered_cohort import write_fixture_checkpoint

    root = tmp_path_factory.mktemp("ckroot")
    ws = _copy(demo_workspace, root / "demo")
    digests = {}
    newest = _run(ws, "gat", "20261020-120000_new_aaaaaa")
    digests["trained"] = write_fixture_checkpoint(ws, newest / "best.pt")
    digests["other"] = write_checkpoint(
        {"checkpoint_kind": "training_state", "epoch": 7,
         "logs": {"val_loss": 0.1, "val_hits@10": 0.4},
         "training_input_digests": {"split_manifest": "9" * 64}}, newest / "other.pt")
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
            ws / "runs/gat/20261020-120000_new_aaaaaa/checkpoints/best.pt")
        assert (item.workspace, item.architecture, item.run) == (
            "demo", "gat", "20261020-120000_new_aaaaaa")
        assert item.validation == {"metric": "val_mrr", "value": 0.0}
        assert item.epoch == 1
        assert item.dataset_fingerprint == file_sha256(ws / "split_manifest.json")
        assert item.trained_on_this_dataset is True
        assert item.problem is None and item.size_bytes > 0

    def test_a_checkpoint_of_another_dataset_says_so(self, checkpoint_root):
        item = self._items(checkpoint_root)["other.pt"]
        assert item.dataset_fingerprint == "9" * 64
        assert item.trained_on_this_dataset is False
        assert item.validation == {"metric": "val_hits@10", "value": 0.4}
        assert item.checkpoint_kind == "training_state" and item.epoch == 7

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
        assert set(self._items(checkpoint_root, "gat")) == {"best.pt", "other.pt"}

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

    @pytest.mark.parametrize("workspace, architecture, fragment", [
        ("missing", None, "is not a workspace the server lists"),
        ("demo", "gatv9", "Unsupported conv_type"),
    ])
    def test_a_name_off_the_list(self, checkpoint_root, workspace, architecture, fragment):
        root, _, _ = checkpoint_root
        listing = sl.list_checkpoints(workspace, architecture, environ=_roots(root))
        assert listing.items == () and fragment in listing.problem

    def test_no_item_carries_a_path(self, checkpoint_root):
        root, _, _ = checkpoint_root
        listing = sl.list_checkpoints("demo", environ=_roots(root))
        assert str(root) not in json.dumps(listing.as_dict("checkpoints"))


class TestResolveCheckpoint:

    def test_a_listed_checkpoint_resolves(self, checkpoint_root):
        root, ws, _ = checkpoint_root
        path = sl.resolve_checkpoint("demo", "gat", "20261020-120000_new_aaaaaa", "best.pt",
                                     environ=_roots(root))
        assert path == ws / "runs/gat/20261020-120000_new_aaaaaa/checkpoints/best.pt"

    @pytest.mark.parametrize("architecture, run, filename, fragment", [
        ("gat", "20261020-120000_new_aaaaaa", "missing.pt", "is listed"),
        ("sage", "20261020-120000_new_aaaaaa", "best.pt", "is listed"),
        ("gat", "../sage/20261019-120000_old_bbbbbb", "weights.pt", "is listed"),
        ("", "20261020-120000_new_aaaaaa", "best.pt", "is not an architecture"),
        (None, "20261020-120000_new_aaaaaa", "best.pt", "is not an architecture"),
        ("gatv9", "20261020-120000_new_aaaaaa", "best.pt", "Unsupported conv_type"),
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
        (ws / "runs" / "gat").mkdir(parents=True)
        (ws / "runs" / "gat" / "20261020-000000_far_cccccc").symlink_to(outside.parent)
        with pytest.raises(sl.ServerListRefused, match="outside the workspace"):
            sl.resolve_checkpoint("demo", "gat", "20261020-000000_far_cccccc", "x.pt",
                                  environ=_roots(root))
        (item,) = sl.list_checkpoints("demo", environ=_roots(root)).items
        assert "outside the workspace" in item.problem


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

    def test_ordering_follows_the_setting(self, registered, tmp_path):
        cohort = _registry_copy(registered, tmp_path)
        role = cohort.version.manifest["role"]
        base = {ss.ENV_REGISTRY: str(cohort.registry)}
        (off,) = sl.list_cohort_versions(environ=base).items
        (on,) = sl.list_cohort_versions(environ={**base, ss.ENV_ORDERING_ROLES: role}).items
        assert (off.offers_ordering, on.offers_ordering) == (False, True)

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
@pytest.fixture
def client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.routes import lists

    app = FastAPI()
    app.include_router(lists.router, prefix="/api/v1")
    return TestClient(app)


class TestRoutes:

    def test_workspaces_without_a_root(self, client, monkeypatch):
        monkeypatch.delenv(ss.ENV_WORKSPACES_ROOT, raising=False)
        response = client.get("/api/v1/workspaces")
        assert response.status_code == 200
        assert response.json()["workspaces"] == []
        assert "SHEPHERD_WORKSPACES_ROOT is not set" in response.json()["problem"]

    def test_workspaces_and_their_checkpoints(self, client, checkpoint_root, monkeypatch):
        root, ws, digests = checkpoint_root
        monkeypatch.setenv(ss.ENV_WORKSPACES_ROOT, str(root))
        body = client.get("/api/v1/workspaces").json()
        assert body["problem"] is None
        assert [w["name"] for w in body["workspaces"]] == ["demo"]
        assert body["workspaces"][0]["dataset_fingerprint"] == file_sha256(
            ws / "split_manifest.json")

        listed = client.get("/api/v1/workspaces/demo/checkpoints",
                            params={"architecture": "gat"}).json()
        assert listed["problem"] is None
        assert {c["filename"]: c["sha256"] for c in listed["checkpoints"]} == {
            "best.pt": digests["trained"], "other.pt": digests["other"]}

    @pytest.mark.parametrize("name", ["nope", ".cache", "%2E%2E"])
    def test_a_request_names_only_a_listed_workspace(self, client, checkpoint_root,
                                                     monkeypatch, name):
        root, _, _ = checkpoint_root
        monkeypatch.setenv(ss.ENV_WORKSPACES_ROOT, str(root))
        response = client.get(f"/api/v1/workspaces/{name}/checkpoints")
        assert response.status_code == 200
        body = response.json()
        assert body["checkpoints"] == [] and "is not a workspace" in body["problem"]

    def test_a_path_is_not_a_name(self, client, checkpoint_root, monkeypatch):
        root, _, _ = checkpoint_root
        monkeypatch.setenv(ss.ENV_WORKSPACES_ROOT, str(root))
        assert client.get("/api/v1/workspaces/..%2Fdemo/checkpoints").status_code == 404

    def test_cohorts(self, client, registered, tmp_path, monkeypatch):
        cohort = _registry_copy(registered, tmp_path)
        monkeypatch.setenv(ss.ENV_REGISTRY, str(cohort.registry))
        body = client.get("/api/v1/cohorts").json()
        assert body["problem"] is None
        assert [v["label"] for v in body["versions"]] == [cohort.version.manifest["label"]]

    def test_the_lists_are_mounted_on_the_service(self):
        from src.api.main import app

        paths = {getattr(route, "path", None) for route in app.routes}
        assert {"/api/v1/workspaces", "/api/v1/cohorts",
                "/api/v1/workspaces/{name}/checkpoints"} <= paths
