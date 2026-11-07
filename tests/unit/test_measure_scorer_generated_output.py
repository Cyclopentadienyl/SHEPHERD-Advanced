"""
A generated split's report and ranks never replace an earlier run's (K8-6r.4).
=============================================================================
`scripts/measure_scorer.py --split ... --output FILE` writes the report at FILE
and the per-case ranks beside it, `<stem>_ranks.json`. Until K8-6r.4 both were
renamed over whatever was there: a second run on one `--output` replaced the
first, two runs at once could leave one run's report beside the other's ranks,
and a report that failed after its ranks left the earlier report beside the new
ranks. Now (plan document B rev 6 §3.10, row E; Decision 9):

- `main` refuses a used `--output` before the GPU lock and before any model is
  built (`_check_generated_targets`), naming an empty file there as a claimed
  place whose publication failed and ranks with no report as an interrupted
  publication, never as a result, and leaving them in place;
- each file is placed with `publish_bytes(..., replace=False)`, the ranks first
  and the report last, so a file that appears after the check is not
  overwritten either, by the hard link or, where `os.link` cannot link, by the
  claim-then-replace fallback (both exercised here);
- two processes publishing to one new `--output` at once leave exactly one
  run's pair, whole (real processes, `tests/fixtures/concurrent_writers.py`).

Every run measures the session's trained disease child
(`trained_disease_child`, tests/conftest.py) on its workspace's generated
`train` or `val` split, read in place; the two splits give two runs different
bytes in both files. Each test writes only under its own directory.

Module: tests/unit/test_measure_scorer_generated_output.py
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("torch")

REPORT = "measurement.json"
RANKS = "measurement_ranks.json"
FAILED_PUBLICATION = ("an empty file: a claimed place whose publication failed, or another "
                      "run's not yet finished, not a report")
INTERRUPTED = "an interrupted publication, or another run's not yet finished, not a result"
ANOTHER_OUTPUT = "it may also be a file another --output wrote under that name"
NOTHING = "Nothing was read or written."

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX links and error numbers")


@pytest.fixture
def cli():
    import scripts.measure_scorer as module

    return module


@pytest.fixture
def workspace(trained_disease_child):
    """``(data_dir, checkpoint)`` of the session's trained disease child, read in place."""
    return trained_disease_child["data_dir"], trained_disease_child["disease"]


@pytest.fixture(params=["link", "EPERM"])
def links(request, monkeypatch):
    """Both placements: the hard link, and the fallback a filesystem without
    hard links takes (`os.link` raising EPERM)."""
    if request.param == "EPERM":
        _no_hard_links(monkeypatch, errno.EPERM)
    return request.param


def _no_hard_links(monkeypatch, number):
    def refuse(src, dst, *args, **kwargs):
        raise OSError(number, os.strerror(number), str(src), None, str(dst))

    monkeypatch.setattr(os, "link", refuse)


def _generated(workspace, output, split="val"):
    data_dir, checkpoint = workspace
    return ["--mode", "disease", "--checkpoint", str(checkpoint), "--data-dir", str(data_dir),
            "--split", split, "--output", str(output), "--device", "cpu"]


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def _published(monkeypatch) -> dict:
    """Record the digest `publish_bytes` returned for each file it placed."""
    import src.utils.file_modes as file_modes

    real, placed = file_modes.publish_bytes, {}

    def publish(target, data, **kwargs):
        placed[Path(target).name] = digest = real(target, data, **kwargs)
        return digest

    monkeypatch.setattr(file_modes, "publish_bytes", publish)
    return placed


def _refused_before_anything(monkeypatch, cli):
    """A refusal must come before the GPU lock and before any model is built."""
    import src.models.bundle.record as bundle_record

    monkeypatch.setattr(bundle_record, "load_bundle",
                        lambda *a, **k: pytest.fail("a model was built after a refusal"))
    monkeypatch.setattr(cli, "take_gpu_lock",
                        lambda *a, **k: pytest.fail("the GPU lock was taken after a refusal"))


def _first_pair(cli, workspace, output, monkeypatch) -> dict:
    """One whole run's pair at `output` (`train`), with the digests it published."""
    with monkeypatch.context() as patch:
        placed = _published(patch)
        assert cli.main(_generated(workspace, output, "train")) == 0
    assert placed == {RANKS: _sha(output.with_name(RANKS)), REPORT: _sha(output)}
    return {RANKS: output.with_name(RANKS).read_bytes(), REPORT: output.read_bytes()}


def _pair(output: Path) -> dict:
    return {name: (output.parent / name).read_bytes() for name in _names(output.parent)}


def _fail_report_write(monkeypatch, output: Path, how: str) -> None:
    """The report's write fails with a full disk after the ranks were placed:
    at its flush (`fsync`, the hard-link mode) or at the fallback's replace onto
    its claimed empty file (`replace`)."""
    import src.utils.file_modes as file_modes

    def full_disk(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device", str(output))

    if how == "replace":
        real_replace = os.replace
        monkeypatch.setattr(os, "replace", lambda src, dst, *a, **k: (
            full_disk() if Path(dst) == output else real_replace(src, dst, *a, **k)))
        return
    real_stage, staged_for = file_modes.create_staging_file, {}

    def stage(target, **kwargs):
        handle, path = real_stage(target, **kwargs)
        staged_for[handle.fileno()] = Path(target)
        return handle, path

    real_fsync = os.fsync
    monkeypatch.setattr(file_modes, "create_staging_file", stage)
    monkeypatch.setattr(os, "fsync", lambda fd: (
        full_disk() if staged_for.get(fd) == output else real_fsync(fd)))


# ---------------------------------------------------------------------------
# E3, E9 (a): a used --output is refused by main, before anything is read
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("there", [
    "report and ranks", "report only", "ranks only", "empty report", "empty report and ranks",
    "dangling link at the report", "dangling link at the ranks",
])
def test_a_used_output_is_refused_before_anything_is_read_and_kept(
        there, cli, workspace, tmp_path, monkeypatch):
    """E3, E9 (a). What is there is named and left byte for byte: an empty
    file as a failed publication, ranks with no report as an interrupted one."""
    output = tmp_path / "run" / REPORT
    ranks = output.with_name(RANKS)
    output.parent.mkdir()
    if there.startswith("dangling"):
        (ranks if there.endswith("ranks") else output).symlink_to(tmp_path / "gone.json")
    else:
        if "report" in there:
            output.write_bytes(b"" if there.startswith("empty") else b'{"an earlier": "report"}')
        if "ranks" in there:
            ranks.write_bytes(b'[{"an earlier": "rank"}]')
    before = {p.name: (os.readlink(p) if p.is_symlink() else p.read_bytes())
              for p in output.parent.iterdir()}
    _refused_before_anything(monkeypatch, cli)

    with pytest.raises(ValueError) as refused:
        cli.main(_generated(workspace, output))

    message = str(refused.value)
    assert message.startswith(f"--output {output} cannot be used: ") and message.endswith(NOTHING)
    assert "it never replaces a file, nor deletes or renames one to make room" in message
    assert (FAILED_PUBLICATION in message) == there.startswith("empty")
    assert (INTERRUPTED in message) == (there in ("ranks only", "empty report and ranks",
                                                  "dangling link at the ranks"))
    after = {p.name: (os.readlink(p) if p.is_symlink() else p.read_bytes())
             for p in output.parent.iterdir()}
    assert after == before
    assert not (tmp_path / "gone.json").exists()


@pytest.mark.parametrize("earlier, given", [
    ("a.json", "a.txt"), ("a.json", "a"), ("b_ranks.json", "b.json"),
], ids=["same stem", "no suffix", "an earlier report at the ranks name"])
def test_a_file_another_output_wrote_at_the_ranks_name_is_not_called_an_interrupted_one(
        earlier, given, cli, tmp_path):
    """A whole earlier pair whose file has this --output's ranks name: two
    --output names with one stem share a ranks name, and `--output
    b_ranks.json` writes its report at `b.json`'s ranks name. The run is
    refused and the pair kept as it was; the message names only what was
    checked, no report at this --output, and says the file may be another
    --output's, never that the ranks have no report beside them."""
    earlier_output = tmp_path / earlier
    earlier_output.write_bytes(b'{"an earlier": "report"}')
    cli._ranks_path(earlier_output).write_bytes(b'[{"an earlier": "rank"}]')
    before = _pair(earlier_output)
    output = tmp_path / given
    ranks = cli._ranks_path(output)
    assert ranks.exists() and not output.exists()

    with pytest.raises(ValueError) as refused:
        cli._check_generated_targets(output)

    message = str(refused.value)
    assert f"{ranks} exists with no report at {output}: " in message
    assert ANOTHER_OUTPUT in message and message.endswith(NOTHING)
    assert "no report beside it" not in message
    assert _pair(earlier_output) == before


def test_an_output_that_is_a_directory_is_refused(cli, workspace, tmp_path, monkeypatch):
    output = tmp_path / "run" / REPORT
    output.mkdir(parents=True)
    _refused_before_anything(monkeypatch, cli)
    with pytest.raises(ValueError, match="is a directory") as refused:
        cli.main(_generated(workspace, output))
    assert str(refused.value).endswith(NOTHING)
    assert _names(output) == []


def test_the_check_creates_no_directory(cli, workspace, tmp_path, monkeypatch):
    """A run that passes the check and is then refused (here, by its GPU lock)
    has made no output directory: it is made at publication only (the probe
    asserts the same of a refused run). A guard: true before K8-6r.4 too."""
    output = tmp_path / "missing" / REPORT

    def busy(*args, **kwargs):
        raise OSError(errno.EBUSY, "the GPU lock is held")

    monkeypatch.setattr(cli, "take_gpu_lock", busy)
    with pytest.raises(OSError, match="GPU lock"):
        cli.main(_generated(workspace, output))
    assert not output.parent.exists()


# ---------------------------------------------------------------------------
# E1, E5, E6: the publication itself never replaces a file
# ---------------------------------------------------------------------------
def test_a_new_output_keeps_both_files_with_their_digests(links, cli, workspace, tmp_path,
                                                          monkeypatch):
    """E6: both files, the digests of the bytes placed, nothing else beside
    them, whichever way they were placed. A guard (a new output was published
    before K8-6r.4 too): the fallback must not lose the files revision 5's
    link-only primitive would have lost where there are no hard links."""
    output = tmp_path / "run" / REPORT
    placed = _published(monkeypatch)
    assert cli.main(_generated(workspace, output)) == 0
    assert _names(output.parent) == [REPORT, RANKS]
    assert placed == {RANKS: _sha(output.with_name(RANKS)), REPORT: _sha(output)}
    assert json.loads(output.read_text())["manifest"]["split"] == "val"


def test_a_reused_output_is_not_published_over_even_past_the_check(
        links, cli, workspace, tmp_path, monkeypatch):
    """E1, E6: with `main`'s check out of the way, the second publication
    still raises (FileExistsError, an OSError, at the ranks), and the first
    pair's bytes are unchanged."""
    output = tmp_path / "run" / REPORT
    first = _first_pair(cli, workspace, output, monkeypatch)
    monkeypatch.setattr(cli, "_check_generated_targets", lambda output: None)

    with pytest.raises(cli.ReportNotWritten) as failed:
        cli.main(_generated(workspace, output, "val"))

    assert isinstance(failed.value.__cause__, FileExistsError)
    assert failed.value.__cause__.filename == str(output.with_name(RANKS))
    assert {RANKS: output.with_name(RANKS).read_bytes(), REPORT: output.read_bytes()} == first
    assert _names(output.parent) == [REPORT, RANKS]


def test_a_second_run_on_a_used_output_is_refused_by_main(links, cli, workspace, tmp_path,
                                                         monkeypatch):
    """E6: the first run kept both; the second is refused by the check, the
    first pair as it was."""
    output = tmp_path / "run" / REPORT
    first = _first_pair(cli, workspace, output, monkeypatch)
    _refused_before_anything(monkeypatch, cli)
    with pytest.raises(ValueError, match="cannot be used"):
        cli.main(_generated(workspace, output, "val"))
    assert {RANKS: output.with_name(RANKS).read_bytes(), REPORT: output.read_bytes()} == first


@pytest.mark.parametrize("appears", ["ranks", "report"])
def test_a_file_that_appears_after_the_check_is_never_overwritten(
        appears, links, cli, workspace, tmp_path, monkeypatch, capsys):
    """E5, E6: another writer's file lands between `main`'s check and the
    publication. The run fails (`ReportNotWritten` caused by FileExistsError,
    from the link or from the fallback's claim) and that file is intact. A
    taken ranks file: this run wrote nothing. A taken report: this run's ranks
    stay, and the message says the report there is not this run's."""
    output = tmp_path / "run" / REPORT
    ranks = output.with_name(RANKS)
    foreign = ranks if appears == "ranks" else output
    real_check = cli._check_generated_targets

    def check_then_another_writer(path):
        real_check(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        foreign.write_bytes(b"another writer's file")

    monkeypatch.setattr(cli, "_check_generated_targets", check_then_another_writer)
    placed = _published(monkeypatch)

    assert cli._run_as_program(_generated(workspace, output)) == 1
    line = [x for x in capsys.readouterr().err.splitlines()
            if x.startswith("measure_scorer: measured, but the report was not written: ")]
    assert len(line) == 1
    assert f"a file is already there and is never replaced: '{foreign}'" in line[0]
    assert "No file was replaced" in line[0] and "Measure again to a new --output." in line[0]
    assert foreign.read_bytes() == b"another writer's file"
    if appears == "ranks":
        assert "this run wrote no file" in line[0]
        assert placed == {} and _names(output.parent) == [RANKS]
    else:
        assert f"{output} appeared after this run checked it and is not this run's report" \
            in line[0]
        assert placed == {RANKS: _sha(ranks)} and _names(output.parent) == [REPORT, RANKS]

    monkeypatch.setattr(cli, "_check_generated_targets", real_check)
    with pytest.raises(ValueError, match="cannot be used"):
        cli.main(_generated(workspace, output))


# ---------------------------------------------------------------------------
# E2, E8, E9 (b, c): a failed second file is a failed publication, never a result
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["link", "fallback"])
def test_a_report_that_fails_after_its_ranks_is_a_failed_publication(
        mode, cli, workspace, tmp_path, monkeypatch, capsys):
    """E2, E8, E9 (b, c). The ranks are placed, then the report's write fails
    (a full disk): with hard links at its flush, in the fallback at the
    replace onto its claim. No report exists; this run's ranks stay, and in
    the fallback the empty claimed file too. The next run on that path is
    refused, naming them as a failed publication, before anything is read;
    nothing is deleted. A retry to a new --output publishes both files."""
    output = tmp_path / "run" / REPORT
    ranks = output.with_name(RANKS)
    with monkeypatch.context() as patch:
        if mode == "fallback":
            _no_hard_links(patch, errno.EPERM)
        _fail_report_write(patch, output, "fsync" if mode == "link" else "replace")
        placed = _published(patch)
        assert cli._run_as_program(_generated(workspace, output)) == 1

    err = capsys.readouterr().err
    assert "No space left on device" in err and "Traceback" not in err
    assert (f"No file was replaced. No report was written at {output}; {ranks} may hold this "
            "run's ranks, or an empty file this run claimed, with no report beside it, an "
            f"interrupted write, and {output} may be an empty file this run claimed. Measure "
            "again to a new --output.") in err
    assert placed == {RANKS: _sha(ranks)}, "this run's own ranks, and no report"
    if mode == "link":
        assert _names(output.parent) == [RANKS]
    else:
        assert _names(output.parent) == [REPORT, RANKS] and output.read_bytes() == b""
    left = _pair(output)

    with monkeypatch.context() as patch:
        _refused_before_anything(patch, cli)
        with pytest.raises(ValueError) as refused:
            cli.main(_generated(workspace, output, "train"))
    assert INTERRUPTED in str(refused.value)
    assert (FAILED_PUBLICATION in str(refused.value)) == (mode == "fallback")
    assert _pair(output) == left

    retry = tmp_path / "retry" / REPORT
    with monkeypatch.context() as patch:
        if mode == "fallback":
            _no_hard_links(patch, errno.EPERM)
        placed = _published(patch)
        assert cli.main(_generated(workspace, retry, "train")) == 0
    assert placed == {RANKS: _sha(retry.with_name(RANKS)), REPORT: _sha(retry)}
    assert _pair(output) == left


def test_a_fallback_that_fails_at_the_ranks_names_the_empty_claim_there(
        cli, workspace, tmp_path, monkeypatch, capsys):
    """Where there are no hard links and the replace onto the ranks' claim
    fails (a full disk), the empty claimed file is at the ranks and --output
    was never touched. The message says the ranks may be an empty file this
    run claimed, and the next run on that path is refused by it, nothing
    deleted."""
    output = tmp_path / "run" / REPORT
    ranks = output.with_name(RANKS)
    with monkeypatch.context() as patch:
        _no_hard_links(patch, errno.EPERM)
        real_replace = os.replace

        def full_disk_at_the_ranks(src, dst, *args, **kwargs):
            if Path(dst) == ranks:
                raise OSError(errno.ENOSPC, "No space left on device", str(dst))
            return real_replace(src, dst, *args, **kwargs)

        patch.setattr(os, "replace", full_disk_at_the_ranks)
        assert cli._run_as_program(_generated(workspace, output)) == 1

    err = capsys.readouterr().err
    assert f"{ranks} may hold this run's ranks, or an empty file this run claimed," in err
    assert _names(output.parent) == [RANKS] and ranks.read_bytes() == b""
    with monkeypatch.context() as patch:
        _refused_before_anything(patch, cli)
        with pytest.raises(ValueError, match="cannot be used"):
            cli.main(_generated(workspace, output, "train"))
    assert _names(output.parent) == [RANKS] and ranks.read_bytes() == b""


def test_an_empty_claimed_report_is_never_read_as_a_report(tmp_path):
    """Decision 9: an empty claimed report is no JSON, so the shared report
    reader refuses it wherever a report is read. A guard: true before K8-6r.4."""
    from src.evaluation.report_store import read_report

    empty = tmp_path / REPORT
    empty.write_bytes(b"")
    with pytest.raises(ValueError):
        read_report(empty)


# ---------------------------------------------------------------------------
# E7: two processes publish to one new --output at once
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("no_hard_links", [False, True], ids=["link", "EPERM"])
def test_two_processes_on_one_new_output_leave_one_whole_pair(no_hard_links, workspace,
                                                              tmp_path):
    """E7. Two real processes (spawned, `sys.executable`) measure `train` and
    `val` to one new `--output`, each under a GPU lock of its own, held at a
    file-marker barrier once both have passed `main`'s check and again right
    before both publish. Exactly one run's pair is published whole, with the
    digests it placed; the other raises `ReportNotWritten` naming the
    collision at the ranks and placed nothing; no file holds a part of the
    other run, and nothing is left beside the pair."""
    from tests.fixtures.concurrent_writers import (
        CONTEXT,
        answers_from,
        measure_generated,
        processes,
    )

    output = tmp_path / "shared" / REPORT
    ranks = output.with_name(RANKS)
    markers = tmp_path / "markers"
    markers.mkdir()
    answers = CONTEXT.Queue()
    names = ("a", "b")
    specs = []
    for name, split in zip(names, ("train", "val"), strict=True):
        lock_root = tmp_path / f"workspaces-{name}"
        lock_root.mkdir()
        specs.append((measure_generated, (name, _generated(workspace, output, split),
                                          str(lock_root), str(markers), names, no_hard_links,
                                          answers)))
    with processes(*specs):
        got = {answer["name"]: answer for answer in answers_from(answers, 2)}

    assert all(got[n]["checked"] and got[n]["publishing"] for n in names), got
    winners = [n for n in names if got[n]["status"] == 0]
    assert len(winners) == 1, got
    (winner,) = winners
    (loser,) = [n for n in names if n != winner]
    split = {"a": "train", "b": "val"}

    assert got[loser]["error"] == "ReportNotWritten" and got[loser]["cause"] == "FileExistsError"
    assert f"a file is already there and is never replaced: '{ranks}'" in got[loser]["message"]
    assert "this run wrote no file" in got[loser]["message"]
    assert got[loser]["placed"] == {}, "the loser placed nothing, so it left no orphan"

    assert _names(output.parent) == [REPORT, RANKS]
    assert got[winner]["placed"] == {RANKS: _sha(ranks), REPORT: _sha(output)}
    assert got[loser]["tried"][RANKS] != _sha(ranks), "the ranks are not the loser's"
    assert json.loads(output.read_text())["manifest"]["split"] == split[winner]
    assert len(json.loads(ranks.read_text())) == \
        json.loads(output.read_text())["manifest"]["n_samples"]
