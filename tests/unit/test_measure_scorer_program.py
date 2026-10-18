"""
The measurement as a program: one message for a refusal, not a traceback.
=========================================================================
BACKLOG 26. `scripts/measure_scorer.py`'s `main` raises its refusals, typed, for
the tests and the deployment probe that call it in process; the script's entry
(`_run_as_program`) turns them into what the operator reads:

- a refusal (`ValueError` or `OSError`, raised before anything is written) is
  one line on stderr, `measure_scorer: refused: ...`, saying "Nothing was
  written." exactly once, and exit 1, for both kinds of cohort;
- a publication failure, after the first write may have happened, says what
  may be on disk and never that nothing was written; `create_report_dir`'s own
  refusal keeps its own text, which says what it created; a failure after the
  report was written names it;
- the command line (exit 2), the device check (now `--device cuda` without
  CUDA too, before anything is read) and a refused registration (exit 3) pass
  through as they were, and so does anything else: an out-of-memory error keeps
  its traceback;
- a checkpoint that does not load as a weights-only checkpoint is refused
  naming the file, as the graph readers name a tensor file; a failed allocation
  and the device's own errors are not converted.

One subprocess run pins the `__main__` wiring.

Module: tests/unit/test_measure_scorer_program.py
"""
from __future__ import annotations

import argparse
import ast
import errno
import inspect
import io
import re
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests.fixtures.registered_cohort import (  # noqa: E402
    build_registered_cohort,
    copy_registered_cohort,
)
from tests.fixtures.synthetic_workspace import build_workspace  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
REFUSED = "measure_scorer: refused: "
NOT_WRITTEN = "measure_scorer: measured, but the report was not written: "
LEDGER = "evaluations.json"


@pytest.fixture
def cli():
    import scripts.measure_scorer as module

    return module


@pytest.fixture
def workspace(tmp_path):
    return build_workspace(tmp_path / "ws")


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return build_registered_cohort(tmp_path_factory.mktemp("program_base"))


@pytest.fixture
def cohort(base, tmp_path):
    return copy_registered_cohort(base, tmp_path / "root")


def _generated(data_dir, checkpoint, output, device="cpu"):
    return ["--checkpoint", str(checkpoint), "--data-dir", str(data_dir),
            "--split", "val", "--output", str(output), "--device", device]


def _registered(cohort, *extra, label=None):
    return ["--checkpoint", str(cohort.checkpoint), "--data-dir", str(cohort.data_dir),
            "--cohort", label or cohort.version.manifest["label"],
            "--registry", str(cohort.registry), "--device", "cpu", *extra]


def _nothing_written(cohort):
    return (not (cohort.version.directory / "reports").exists()
            and not (cohort.checkpoint.parent / LEDGER).exists())


def _one_line(err, prefix):
    """The one stderr line starting with `prefix`; no traceback beside it."""
    lines = [line for line in err.splitlines() if line.startswith(prefix)]
    assert len(lines) == 1, err
    assert "Traceback" not in err, err
    return lines[0]


def _no_model(monkeypatch):
    import src.models.gnn.shepherd_gnn as shepherd_gnn

    monkeypatch.setattr(shepherd_gnn, "build_shepherd_model",
                        lambda *a, **k: pytest.fail("a model was built after a refusal"))


def _pre_contract(checkpoint, path):
    contents = torch.load(checkpoint, map_location="cpu", weights_only=True)
    del contents["training_input_digests"]
    torch.save(contents, path)
    return path


# ---------------------------------------------------------------------------
# A refusal is one message, for both kinds of cohort
# ---------------------------------------------------------------------------
def test_a_generated_split_refusal_is_one_message(cli, workspace, tmp_path, monkeypatch, capsys):
    """R3 on a generated split: its message does not say what was written, so
    the program entry says it, once, and exits 1 with nothing on disk."""
    data_dir, checkpoint = workspace
    old = _pre_contract(checkpoint, tmp_path / "old.pt")
    output = tmp_path / "out" / "measurement.json"
    _no_model(monkeypatch)

    assert cli._run_as_program(_generated(data_dir, old, output)) == 1

    err = capsys.readouterr().err
    line = _one_line(err, REFUSED)
    assert "records no training_input_digests" in line
    assert line.endswith("retrain it. Nothing was written.")
    assert err.count("Nothing was written") == 1
    assert not output.parent.exists()


def _samples_removed(cohort):
    cohort.version.samples_path.unlink()
    return {"argv": _registered(cohort), "says": "is missing, though"}


def _unknown_label(cohort):
    return {"argv": _registered(cohort, label="no-such-version"),
            "says": "holds no cohort version 'no-such-version'"}


def _evidence_missing(cohort):
    missing = cohort.data_dir.parent / "no-such-evidence"
    return {"argv": _registered(cohort, "--training-evidence", str(missing)),
            "says": "nothing was read or written"}


@pytest.mark.parametrize("case", [_samples_removed, _unknown_label, _evidence_missing],
                         ids=["says-it-already", "appended", "read-or-written"])
def test_a_registered_refusal_is_one_message_saying_nothing_was_written_once(
    cli, cohort, monkeypatch, capsys, case
):
    """A message that already says what was written, in either of this
    codebase's phrasings, gets no second sentence; one that does not gets
    "Nothing was written." once. No run directory, no ledger."""
    given = case(cohort)
    _no_model(monkeypatch)

    assert cli._run_as_program(given["argv"]) == 1

    err = capsys.readouterr().err
    line = _one_line(err, REFUSED)
    assert given["says"] in line
    said = len(re.findall(r"nothing was (?:read or )?written", err, re.IGNORECASE))
    assert said == 1, err
    assert _nothing_written(cohort)


def test_a_file_that_cannot_be_read_is_one_message(cli, workspace, tmp_path, capsys):
    """An `OSError` (here the checkpoint is missing, read after every workspace
    check) ends with a full stop before the sentence, as `import_cohort.py`'s."""
    data_dir, _ = workspace
    missing = tmp_path / "no-such.pt"
    output = tmp_path / "out" / "measurement.json"

    assert cli._run_as_program(_generated(data_dir, missing, output)) == 1

    line = _one_line(capsys.readouterr().err, REFUSED)
    assert "No such file or directory" in line and str(missing) in line
    assert line.endswith("'. Nothing was written.")
    assert not output.parent.exists()


# ---------------------------------------------------------------------------
# What passes through as it was
# ---------------------------------------------------------------------------
def test_a_rejected_command_line_still_exits_2(cli, capsys):
    with pytest.raises(SystemExit) as rejected:
        cli._run_as_program(["--checkpoint", "c.pt", "--data-dir", "d", "--split", "val"])

    assert rejected.value.code == 2
    assert "--split needs --output" in capsys.readouterr().err


@pytest.mark.parametrize("device", ["auto", "cuda"])
def test_a_cuda_request_without_cuda_is_refused_before_anything_is_read(
    cli, tmp_path, monkeypatch, device
):
    """`--device auto` was refused already; an explicit `cuda` now is too, in
    `_resolve_device`, instead of by torch when the checkpoint is loaded onto the
    device after every workspace check."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(cli, "read_measurement_inputs",
                        lambda *a, **k: pytest.fail("an input was read"))
    output = tmp_path / "out" / "measurement.json"

    with pytest.raises(SystemExit) as refused:
        cli._run_as_program(_generated(tmp_path / "ws", tmp_path / "c.pt", output, device))

    assert str(refused.value.code).startswith(f"--device {device} requires CUDA")
    assert "--device cpu" in str(refused.value.code)
    assert list(tmp_path.iterdir()) == []


def test_a_refused_registration_still_exits_3(cli, cohort, monkeypatch, capsys):
    """Measured and kept, but not registered: `main`'s own status, with its own
    message, and no refusal line."""
    import src.evaluation.sidecar as sidecar

    def read_only(*args, **kwargs):
        raise PermissionError(13, "Permission denied", str(cohort.checkpoint.parent))

    monkeypatch.setattr(sidecar, "create_staging_file", read_only)

    assert cli._run_as_program(_registered(cohort, "--record")) == cli.EXIT_NOT_REGISTERED

    err = capsys.readouterr().err
    assert "NOT REGISTERED" in err and REFUSED not in err and "Traceback" not in err
    assert len(list((cohort.version.directory / "reports").glob("*/report.json"))) == 1


# ---------------------------------------------------------------------------
# A publication failure says what may be on disk
# ---------------------------------------------------------------------------
def _fail_publishing(monkeypatch, is_report):
    import src.utils.file_modes as file_modes

    real = file_modes.publish_bytes

    def publish(target, data):
        if is_report(Path(target)):
            raise OSError(errno.ENOSPC, "No space left on device", str(target))
        return real(target, data)

    monkeypatch.setattr(file_modes, "publish_bytes", publish)


def test_a_generated_report_that_could_not_be_written_says_what_may_be_there(
    cli, workspace, tmp_path, monkeypatch, capsys
):
    """The ranks are published first: when the report then fails, the new ranks
    sit beside the earlier report. The message says so, and never that nothing
    was written; `main` raises it as `ReportNotWritten`, the cause chained."""
    data_dir, checkpoint = workspace
    output = tmp_path / "run" / "measurement.json"
    ranks = output.parent / "measurement_ranks.json"
    output.parent.mkdir()
    output.write_text("an earlier report")
    _fail_publishing(monkeypatch, lambda target: target == output)

    assert cli._run_as_program(_generated(data_dir, checkpoint, output)) == 1

    err = capsys.readouterr().err
    line = _one_line(err, NOT_WRITTEN)
    assert "No space left on device" in line
    assert f"{output} was left as it was" in line
    assert f"{ranks} may already hold this run's ranks" in line
    assert "nothing was written" not in err.lower() and REFUSED not in err
    assert output.read_text() == "an earlier report"
    assert ranks.is_file()

    with pytest.raises(cli.ReportNotWritten) as failed:
        cli.main(_generated(data_dir, checkpoint, output))
    assert isinstance(failed.value.__cause__, OSError)
    assert not isinstance(failed.value, (ValueError, OSError))


def test_a_registered_report_that_could_not_be_written_says_what_may_be_there(
    cli, cohort, monkeypatch, capsys
):
    """A run directory holding only the ranks, with no report to commit it."""
    _fail_publishing(monkeypatch, lambda target: target.name == "report.json")

    assert cli._run_as_program(_registered(cohort)) == 1

    err = capsys.readouterr().err
    line = _one_line(err, NOT_WRITTEN)
    reports = cohort.version.directory / "reports"
    assert "No report.json was published" in line
    assert f"A new run directory in {reports} may exist, empty or holding only ranks.json" in line
    assert "nothing was written" not in err.lower() and REFUSED not in err
    (run_dir,) = reports.iterdir()
    assert [p.name for p in run_dir.iterdir()] == ["ranks.json"]
    assert not (cohort.checkpoint.parent / LEDGER).exists()


def test_the_report_store_refusing_a_link_keeps_its_own_words(
    cli, cohort, tmp_path, monkeypatch, capsys
):
    """`create_report_dir` refuses a `reports` that is a link out of the version
    directory before creating anything; its message says "Nothing was written."
    itself, once, and the link's target stays empty."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (cohort.version.directory / "reports").symlink_to(elsewhere, target_is_directory=True)

    assert cli._run_as_program(_registered(cohort)) == 1

    err = capsys.readouterr().err
    line = _one_line(err, NOT_WRITTEN)
    assert "Replace the link with a directory. Nothing was written." in line
    assert err.count("Nothing was written") == 1 and REFUSED not in err
    assert list(elsewhere.iterdir()) == []


def test_the_report_store_refusing_its_run_directory_says_it_was_created(
    cli, cohort, monkeypatch, capsys
):
    """After the run directory is made, `create_report_dir`'s refusal says only
    the empty directory was created. That text is printed as it is: no "Nothing
    was written." is added, which would contradict it."""
    import src.evaluation.cohort_registry as cohort_registry

    real = cohort_registry._registry_directory

    def refuse_the_run_directory(path, *, within, done):
        if within is not None and within.name == "reports":
            raise cohort_registry.RegistryRefused(f"{path} resolves elsewhere (simulated). {done}")
        return real(path, within=within, done=done)

    monkeypatch.setattr(cohort_registry, "_registry_directory", refuse_the_run_directory)

    assert cli._run_as_program(_registered(cohort)) == 1

    err = capsys.readouterr().err
    line = _one_line(err, NOT_WRITTEN)
    (run_dir,) = (cohort.version.directory / "reports").iterdir()
    assert line.endswith(
        f"Only the empty run directory {run_dir} was created; nothing was written into it.")
    assert "Nothing was written" not in err and REFUSED not in err
    assert list(run_dir.iterdir()) == []


def test_a_failure_after_the_report_was_written_says_where_it_is(cli, cohort, monkeypatch, capsys):
    """The console closing under the run after publication (an `OSError` like any
    file's) is no refusal: the message names the report and the command that
    registers it, or says it already is, and never that nothing was written."""

    def console_closed(*args, **kwargs):
        raise BrokenPipeError(errno.EPIPE, "Broken pipe")

    monkeypatch.setattr(cli, "_print_registered_summary", console_closed)

    assert cli._run_as_program(_registered(cohort, "--record")) == 1

    err = capsys.readouterr().err
    line = _one_line(err, "measure_scorer: measured, but the run did not finish: ")
    (report,) = (cohort.version.directory / "reports").glob("*/report.json")
    assert f"The report was written: {report}." in line
    assert f'record_evaluation.py --report "{report}"' in line
    assert "nothing was written" not in err.lower() and REFUSED not in err

    with pytest.raises(cli.FailedAfterPublishing) as failed:
        cli.main(_registered(cohort))
    assert isinstance(failed.value.__cause__, BrokenPipeError)


# ---------------------------------------------------------------------------
# A checkpoint that does not parse is refused naming it
# ---------------------------------------------------------------------------
def _truncated(path):
    buffer = io.BytesIO()
    torch.save({"w": torch.zeros(3)}, buffer)
    return buffer.getvalue()[: len(buffer.getvalue()) // 2]


def _pickled_object(path):
    buffer = io.BytesIO()
    torch.save({"object": argparse.Namespace(arbitrary=True)}, buffer)
    return buffer.getvalue()


@pytest.mark.parametrize("junk, raised", [
    (lambda path: b"\x00\x01 not a checkpoint " * 8, "UnpicklingError"),
    (lambda path: b"hello, a text file\nline two\n", "KeyError"),
    (lambda path: b"", "EOFError"),
    (_truncated, "RuntimeError"),
    (_pickled_object, "UnpicklingError"),
], ids=["junk", "text", "empty", "truncated-zip", "not-weights-only"])
def test_a_checkpoint_that_does_not_parse_is_refused_naming_it(
    cli, workspace, tmp_path, monkeypatch, capsys, junk, raised
):
    """torch raises a different type for each kind of junk, naming no file; the
    measurement refuses with one `ValueError` that names it, torch's error
    chained, and the program prints it on one line."""
    data_dir, _ = workspace
    checkpoint = tmp_path / "given.pt"
    checkpoint.write_bytes(junk(checkpoint))
    output = tmp_path / "out" / "measurement.json"
    _no_model(monkeypatch)

    with pytest.raises(cli.UnparseableCheckpoint) as refused:
        cli.main(_generated(data_dir, checkpoint, output))
    assert isinstance(refused.value, ValueError)
    assert refused.value.path == checkpoint
    assert type(refused.value.__cause__).__name__ == raised

    assert cli._run_as_program(_generated(data_dir, checkpoint, output)) == 1
    line = _one_line(capsys.readouterr().err, REFUSED)
    assert line.startswith(f"{REFUSED}{checkpoint} could not be parsed as a weights-only "
                           f"checkpoint ({raised}")
    assert line.endswith("Nothing was written.")
    assert not output.parent.exists()


@pytest.mark.parametrize("failure", [
    MemoryError(),
    RuntimeError("DefaultCPUAllocator: can't allocate memory: you tried to allocate 8 bytes"),
    torch.OutOfMemoryError("CUDA out of memory (simulated)"),
    torch.AcceleratorError("CUDA error: an illegal memory access (simulated)"),
], ids=["MemoryError", "cpu-allocator", "OutOfMemoryError", "AcceleratorError"])
def test_a_failed_allocation_or_a_device_error_keeps_its_traceback(
    cli, workspace, tmp_path, monkeypatch, failure
):
    """Not properties of the file: neither the checkpoint read nor the program
    entry converts them, so they propagate as raised."""
    import src.utils.checkpoint_io as checkpoint_io

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(checkpoint_io, "read_checkpoint", fail)
    data_dir, checkpoint = workspace

    with pytest.raises(type(failure)) as raised:
        cli._run_as_program(_generated(data_dir, checkpoint, tmp_path / "out" / "m.json"))
    assert raised.value is failure


def test_a_cuda_load_on_a_host_without_cuda_is_not_called_unparseable(cli, workspace):
    """`main` refuses `--device cuda` first; a direct caller asking for CUDA here
    gets torch's own error, not a claim that a sound file does not parse."""
    if torch.cuda.is_available():
        pytest.skip("this host has CUDA; the case is a host without it")
    data_dir, checkpoint = workspace

    with pytest.raises(RuntimeError, match="CUDA") as raised:
        cli.read_measurement_inputs(checkpoint, data_dir, torch.device("cuda"), split="val")
    assert not isinstance(raised.value, ValueError)


def test_the_checkpoint_read_names_no_other_read(cli):
    """The helper reads through `read_checkpoint` only, like the helpers pinned
    in `test_measurement_reads_once.py`: no path read, no hash, no `torch.load`."""
    body = ast.parse(inspect.getsource(cli._read_checkpoint)).body[0]
    names = {node.id for node in ast.walk(body) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(body) if isinstance(node, ast.Attribute)}
    assert not names & {"read_once", "file_sha256", "hashlib", "open", "read_bytes",
                        "read_text"}
    assert "torch.load" not in ast.unparse(body)


# ---------------------------------------------------------------------------
# The script's entry, as the operator runs it
# ---------------------------------------------------------------------------
def test_run_as_a_program_a_refusal_prints_no_traceback(tmp_path):
    """`__main__` goes through `_run_as_program`: a split that is not the
    generator's, refused before anything is read, exits 1 with one message."""
    output = tmp_path / "out" / "measurement.json"
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "measure_scorer.py"),
         "--checkpoint", str(tmp_path / "c.pt"), "--data-dir", str(tmp_path / "ws"),
         "--split", "test", "--output", str(output), "--device", "cpu"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300,
    )

    assert completed.returncode == 1, completed.stderr
    line = _one_line(completed.stderr, REFUSED)
    assert "'test' is not one of the generator's splits" in line
    assert line.endswith("Nothing was written.")
    assert not output.parent.exists()
