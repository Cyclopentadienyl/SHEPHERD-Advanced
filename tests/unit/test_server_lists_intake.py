"""
The server's lists for the Test page (M5-2, UI-D): import sources, one version, one run.
=======================================================================================
`src/api/services/server_lists.py`: `list_intake_files`, `resolve_intake_file`,
`resolve_cohort_version`, `cohort_version_item` and `resolve_report`. What these
pin:

- **import sources from the server's intake directory only** (D7): its
  ``*.json`` files by name, with size and modification time and no content
  read; hidden files and other kinds are not offered; an entry that is not a
  file, or a link that leaves the directory, is listed with why and never
  offered; no intake directory, no list, and why;
- **a name resolves only when the listing offers it**: any other name, a path
  (relative or absolute, a ``..`` among them) or a link leaving the directory
  is refused, and the path returned is the listed entry, so the version
  manifest records the name the operator chose;
- **one cohort version by its label**, through the registry's label rule and
  its one reader; a label the rule refuses, an unknown one or no registry is
  refused; one version's item carries its runs and the states the given reader
  says;
- **one run's report by its run id**, found among the version's listed run
  directories: an unknown id or the store's own files beside the runs are
  refused.

Module: tests/unit/test_server_lists_intake.py
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.api.services import server_lists as sl
from src.config import server_settings as ss


def _intake(directory: Path) -> dict:
    return {ss.ENV_INTAKE_DIR: str(directory)}


@pytest.fixture
def intake(tmp_path):
    """An intake directory with two sources, files that are not offered, a
    directory named like a source and links in and out of it."""
    directory = tmp_path / "intake"
    directory.mkdir()
    (directory / "b-cases.json").write_text('{"format": "shepherd-cases/1", "cases": []}')
    (directory / "a-cases.JSON").write_text("{}")
    (directory / "notes.txt").write_text("not a source")
    (directory / ".partial.json").write_text("{}")
    (directory / "folder.json").mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (directory / "leaves.json").symlink_to(outside)
    (directory / "stays.json").symlink_to(directory / "b-cases.json")
    return directory


class TestIntakeListing:

    def test_no_intake_directory_no_list_and_why(self):
        listing = sl.list_intake_files(environ={})
        assert listing.items == () and "SHEPHERD_INTAKE_DIR is not set" in listing.problem

    def test_a_refused_setting_lists_nothing(self, tmp_path):
        listing = sl.list_intake_files(environ={ss.ENV_INTAKE_DIR: "relative/intake"})
        assert listing.items == () and "not an absolute path" in listing.problem
        listing = sl.list_intake_files(environ=_intake(tmp_path / "missing"))
        assert listing.items == () and "does not exist" in listing.problem

    def test_the_json_files_by_name_with_size_and_time(self, intake):
        items = {i.name: i for i in sl.list_intake_files(environ=_intake(intake)).items}
        assert sorted(items) == ["a-cases.JSON", "b-cases.json", "folder.json", "leaves.json",
                                 "stays.json"]
        offered = items["b-cases.json"]
        assert offered.problem is None
        assert offered.size_bytes == (intake / "b-cases.json").stat().st_size
        assert offered.modified_at.endswith("Z") and "T" in offered.modified_at
        assert items["a-cases.JSON"].problem is None
        assert items["stays.json"].problem is None, "a link inside the directory is offered"

    def test_an_entry_that_is_no_file_or_leaves_the_directory_is_never_offered(self, intake):
        items = {i.name: i for i in sl.list_intake_files(environ=_intake(intake)).items}
        leaves = items["leaves.json"]
        assert "outside the intake directory" in leaves.problem and leaves.size_bytes is None
        assert "is not a file" in items["folder.json"].problem

    def test_listing_reads_no_file(self, intake):
        from tests.fixtures.opens import count_opens

        with count_opens() as opens:
            sl.list_intake_files(environ=_intake(intake))
        assert opens.opens(intake / "b-cases.json") == 0


class TestResolveIntakeFile:

    def test_a_listed_name_resolves_to_its_listed_entry(self, intake):
        path = sl.resolve_intake_file("b-cases.json", environ=_intake(intake))
        assert path == intake.resolve() / "b-cases.json"
        linked = sl.resolve_intake_file("stays.json", environ=_intake(intake))
        assert linked.name == "stays.json", "the manifest records the name the operator chose"

    @pytest.mark.parametrize("name", ["missing.json", "notes.txt", ".partial.json",
                                      "../outside.json", "sub/b-cases.json", None, 3])
    def test_any_other_name_or_a_path_is_refused(self, intake, name):
        with pytest.raises(sl.ServerListRefused, match="is not an import source the server"):
            sl.resolve_intake_file(name, environ=_intake(intake))

    def test_an_absolute_path_is_refused_even_inside_the_directory(self, intake):
        with pytest.raises(sl.ServerListRefused):
            sl.resolve_intake_file(str(intake / "b-cases.json"), environ=_intake(intake))

    def test_a_link_that_leaves_the_directory_or_a_folder_is_refused(self, intake):
        with pytest.raises(sl.ServerListRefused, match="outside the intake directory"):
            sl.resolve_intake_file("leaves.json", environ=_intake(intake))
        with pytest.raises(sl.ServerListRefused, match="is not a file"):
            sl.resolve_intake_file("folder.json", environ=_intake(intake))

    def test_no_intake_directory_refuses(self):
        with pytest.raises(sl.ServerListRefused, match="SHEPHERD_INTAKE_DIR is not set"):
            sl.resolve_intake_file("b-cases.json", environ={})


# ---------------------------------------------------------------------------
# One version, one run
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def registered(tmp_path_factory):
    from tests.fixtures.registered_cohort import build_registered_cohort

    return build_registered_cohort(tmp_path_factory.mktemp("one-version") / "c")


@pytest.fixture
def store(registered, tmp_path):
    """A copy of the registry with one run directory holding a report, an
    interrupted one, and the store's usage history beside them."""
    from tests.fixtures.registered_cohort import copy_registered_cohort

    cohort = copy_registered_cohort(registered, tmp_path / "root")
    reports = cohort.version.directory / "reports"
    sha12 = cohort.checkpoint_sha256[:12]
    kept = reports / f"20261020T101500Z-{sha12}-0000000a"
    kept.mkdir(parents=True)
    (kept / "report.json").write_text("{}")
    (reports / f"20261020T111500Z-{sha12}-0000000b").mkdir()
    (reports / "usage.jsonl").write_text("")
    return cohort, {ss.ENV_REGISTRY: str(cohort.registry)}


class TestOneVersion:

    def test_a_label_resolves_through_the_registrys_reader(self, store):
        cohort, environ = store
        version = sl.resolve_cohort_version("demo-test-1", environ=environ)
        assert version.identity.sha256 == cohort.version.identity.sha256

    @pytest.mark.parametrize("label", ["../demo-test-1", "missing-label", "", None, "a/b"])
    def test_another_label_is_refused(self, store, label):
        _cohort, environ = store
        with pytest.raises(sl.ServerListRefused):
            sl.resolve_cohort_version(label, environ=environ)

    def test_no_registry_refuses(self):
        with pytest.raises(sl.ServerListRefused, match="SHEPHERD_REGISTRY is not set"):
            sl.resolve_cohort_version("demo-test-1", environ={})

    def test_one_versions_item_carries_its_runs_and_the_readers_states(self, store):
        cohort, environ = store
        asked = []

        def reader(run_directory):
            asked.append(Path(run_directory).name)
            return {"kept": "kept"}

        item = sl.cohort_version_item("demo-test-1", environ=environ, report_state=reader)
        assert item.label == "demo-test-1" and item.manifest_sha256 == cohort.version.identity.sha256
        runs = {r.run_id: r for r in item.reports}
        assert set(asked) == {name for name, r in runs.items() if r.problem is None}
        assert (cohort.version.directory / "reports" / "usage.jsonl").is_file()
        assert "usage.jsonl" not in runs, "the store's own file is not a run, so it is not listed"

    def test_a_listed_runs_report_path(self, store):
        cohort, environ = store
        sha12 = cohort.checkpoint_sha256[:12]
        run = f"20261020T101500Z-{sha12}-0000000a"
        path = sl.resolve_report("demo-test-1", run, environ=environ)
        assert path == cohort.version.directory / "reports" / run / "report.json"
        interrupted = f"20261020T111500Z-{sha12}-0000000b"
        assert sl.resolve_report("demo-test-1", interrupted, environ=environ).name == "report.json"

    @pytest.mark.parametrize("run", ["usage.jsonl", "20261020T999999Z-000000000000-00000000",
                                     "../reports", None])
    def test_another_run_is_refused(self, store, run):
        _cohort, environ = store
        with pytest.raises(sl.ServerListRefused):
            sl.resolve_report("demo-test-1", run, environ=environ)

    def test_a_run_directory_that_leaves_the_store_is_refused(self, store, tmp_path):
        cohort, environ = store
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        sha12 = cohort.checkpoint_sha256[:12]
        run = f"20261020T121500Z-{sha12}-0000000c"
        os.symlink(elsewhere, cohort.version.directory / "reports" / run)
        with pytest.raises(sl.ServerListRefused):
            sl.resolve_report("demo-test-1", run, environ=environ)
