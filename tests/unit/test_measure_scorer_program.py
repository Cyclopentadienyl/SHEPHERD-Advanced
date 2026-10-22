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
  report was written names it, also when the console closed under a buffered
  stdout (exit 1, not Python's exit 120 at shutdown);
- the command line (exit 2), the device check (now `--device cuda` without
  CUDA too, before anything is read) and a refused registration (exit 3) pass
  through as they were, and so does anything else: an out-of-memory error keeps
  its traceback;
- a checkpoint that does not load as a weights-only checkpoint is refused
  naming the file, as the graph readers name a tensor file; a failed allocation
  and the device's own errors are not converted;
- a cancel (SIGINT: the job runner's stop) is one line, no KeyboardInterrupt
  traceback, and the process still ends by SIGINT, which is how the runner
  tells a cancelled run from one that ended on its own; what stdout's buffer
  held is kept. Only the program's entry (`_exit_as_program`) ends the
  process: an in-process caller of `_run_as_program` (these tests) gets the
  KeyboardInterrupt back, so a Ctrl+C during them still lets pytest clean up.

Subprocess runs pin the `__main__` wiring, the closed console and the cancel.

Module: tests/unit/test_measure_scorer_program.py
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import errno
import inspect
import io
import os
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
NOT_FINISHED = "measure_scorer: measured, but the run did not finish: "
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
def cohort(base, tmp_path, monkeypatch):
    """A copy of the built cohort, its registry the deployment's
    (`SHEPHERD_REGISTRY`, inherited by a program started from here): a
    ``--cohort`` run measures only there (BACKLOG 37)."""
    copied = copy_registered_cohort(base, tmp_path / "root")
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(copied.registry))
    return copied


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
    line = _one_line(err, NOT_FINISHED)
    (report,) = (cohort.version.directory / "reports").glob("*/report.json")
    assert f"The report was written: {report}." in line
    assert f'record_evaluation.py --report "{report}"' in line
    assert "nothing was written" not in err.lower() and REFUSED not in err

    with pytest.raises(cli.FailedAfterPublishing) as failed:
        cli.main(_registered(cohort))
    assert isinstance(failed.value.__cause__, BrokenPipeError)


@pytest.mark.parametrize("reader", ["gone", "reading"])
def test_after_publishing_only_a_stdout_that_failed_is_redirected(cli, monkeypatch, reader):
    """stdout here is a real, buffered pipe. Its reader gone, it is pointed at
    the null device, so the interpreter's last flush (the write after the
    return) cannot fail and turn exit 1 into 120. Its reader still there, as
    when a registration raised with stdout on a file, it first gets what it
    buffered: an unconditional redirect would throw the summary away."""
    read_end, write_end = os.pipe()
    if reader == "gone":
        os.close(read_end)

    def summary_then_failure(argv):
        print("the summary")
        raise cli.FailedAfterPublishing("The report was written: r.json.")

    monkeypatch.setattr(cli, "main", summary_then_failure)
    stdout = open(write_end, "w", encoding="utf-8")  # buffered, as a pipe is
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        assert cli._run_as_program([]) == 1
    stdout.write("at exit\n")
    stdout.close()

    assert stderr.getvalue() == f"{NOT_FINISHED}The report was written: r.json.\n"
    if reader == "reading":
        assert os.read(read_end, 4096) == b"the summary\nat exit\n"
        os.close(read_end)


def test_with_record_a_console_gone_before_registering_registers_nothing(
    cli, cohort, monkeypatch
):
    """stdout a buffered pipe whose reader is gone, so the summary only fills
    the buffer. The console is found gone before the ledger is touched, as an
    unbuffered one is at its first print: nothing is registered, and the line
    that offers the registration is true. Found after `_register`, it used to
    follow a refusal ("nothing at that path is offered") with that offer."""
    import src.evaluation.registration as registration

    monkeypatch.setattr(registration, "register_report",
                        lambda *a, **k: pytest.fail("registered for a console that had gone"))
    read_end, write_end = os.pipe()
    os.close(read_end)
    stdout = open(write_end, "w", encoding="utf-8")  # buffered, as a pipe is
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        assert cli._run_as_program(_registered(cohort, "--record")) == 1
    stdout.close()  # the interpreter's last flush, which must not fail

    err = stderr.getvalue()
    line = _one_line(err, NOT_FINISHED)
    (report,) = (cohort.version.directory / "reports").glob("*/report.json")
    assert f"The report was written: {report}." in line
    assert f'record_evaluation.py --report "{report}"' in line
    assert "NOT REGISTERED" not in err
    assert not (cohort.checkpoint.parent / LEDGER).exists()


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


def test_run_as_a_program_a_closed_console_after_publication_names_the_report(
    workspace, tmp_path
):
    """stdout on a pipe whose reader is gone (a `tee` that died with its SSH
    session), buffered as an operator's is: no PYTHONUNBUFFERED, which this
    sandbox sets. The summary used to sit in the buffer until the interpreter's
    exit, which printed "Exception ignored ... BrokenPipeError" and exited 120,
    with no line naming the report."""
    data_dir, checkpoint = workspace
    output = tmp_path / "out" / "measurement.json"
    env = {name: value for name, value in os.environ.items() if name != "PYTHONUNBUFFERED"}
    read_end, write_end = os.pipe()
    os.close(read_end)
    try:
        completed = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "measure_scorer.py"),
             *_generated(data_dir, checkpoint, output)],
            cwd=str(REPO_ROOT), env=env, stdout=write_end, stderr=subprocess.PIPE,
            text=True, timeout=300,
        )
    finally:
        os.close(write_end)

    assert completed.returncode == 1, completed.stderr
    line = _one_line(completed.stderr, NOT_FINISHED)
    assert line.endswith(f"The report was written: {output}.")
    assert "Exception ignored" not in completed.stderr
    assert output.is_file()


@pytest.mark.skipif(sys.platform == "win32", reason="closes a POSIX descriptor in the child")
def test_run_as_a_program_with_no_stdout_measures_and_exits_0(workspace, tmp_path):
    """Started with descriptor 1 closed (`>&-`, a detached launch), Python sets
    `sys.stdout` to None and `print` writes nothing: the run measures and
    exits 0, with no traceback from flushing a stream that is not there."""
    data_dir, checkpoint = workspace
    output = tmp_path / "out" / "measurement.json"
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "measure_scorer.py"),
         *_generated(data_dir, checkpoint, output)],
        cwd=str(REPO_ROOT), stderr=subprocess.PIPE, text=True, timeout=300,
        preexec_fn=lambda: os.close(1),
    )

    assert completed.returncode == 0, completed.stderr
    assert "Traceback" not in completed.stderr
    assert output.is_file()


# ---------------------------------------------------------------------------
# A cancel: one line, no traceback, and still ended by SIGINT
# ---------------------------------------------------------------------------
def _interrupted(argv=None):
    raise KeyboardInterrupt


def test_in_process_a_cancel_is_one_line_and_comes_back_to_the_caller(cli, monkeypatch, capsys):
    """`_run_as_program` called in process (as these tests call it) and
    interrupted: `CANCELLED_LINE` once, then the KeyboardInterrupt back to the
    caller, whose cleanup runs. It never ends the caller's process: that is
    the program entry's to do. `_end_as_cancelled` is left as it is; only
    `os.kill` is recorded, so a self-kill fails this test instead of ending
    pytest."""
    import signal

    kills = []
    monkeypatch.setattr(cli, "main", _interrupted)
    monkeypatch.setattr(os, "kill", lambda pid, sig: kills.append((pid, sig)))
    previous = signal.getsignal(signal.SIGINT)
    try:
        with pytest.raises(KeyboardInterrupt):
            cli._run_as_program([])
    finally:
        handler_after = signal.signal(signal.SIGINT, previous)
    assert kills == [], "the in-process caller's own process was sent SIGINT"
    assert handler_after is previous, "the caller's SIGINT handler was changed"
    err = capsys.readouterr().err
    assert _one_line(err, "measure_scorer: cancelled") == cli.CANCELLED_LINE


def test_the_program_entry_ends_a_cancel_by_sigint_and_passes_a_status_on(cli, monkeypatch,
                                                                          capsys):
    """`__main__` is `_exit_as_program()`, the only place a cancel ends the
    process (`_end_as_cancelled`, which ends it by SIGINT), after the one line;
    any other outcome is `_run_as_program`'s status, unchanged."""
    tree = ast.parse(Path(cli.__file__).read_text())
    entry = [node for node in tree.body if isinstance(node, ast.If)
             and ast.unparse(node.test) == "__name__ == '__main__'"][-1]
    assert [ast.unparse(statement) for statement in entry.body] == ["_exit_as_program()"]

    ended = []
    monkeypatch.setattr(cli, "_end_as_cancelled", lambda: ended.append(True) or 130)
    monkeypatch.setattr(cli, "main", _interrupted)
    try:
        with pytest.raises(SystemExit) as cancelled:
            cli._exit_as_program([])
    except KeyboardInterrupt:
        pytest.fail("the cancel left the program entry as a KeyboardInterrupt (a traceback)")
    assert cancelled.value.code == 130 and ended == [True]
    err = capsys.readouterr().err
    assert _one_line(err, "measure_scorer: cancelled") == cli.CANCELLED_LINE

    monkeypatch.setattr(cli, "main", lambda argv=None: cli.EXIT_NOT_REGISTERED)
    with pytest.raises(SystemExit) as finished:
        cli._exit_as_program([])
    assert finished.value.code == cli.EXIT_NOT_REGISTERED and ended == [True]


_CANCELLED_CHILD = """
import sys, time
sys.path.insert(0, sys.argv[1])
import scripts.measure_scorer as cli

def measuring(args):
    print("PRINTED BEFORE THE CANCEL")  # not flushed: stdout is a pipe, block-buffered
    print("MEASURING", file=sys.stderr, flush=True)
    time.sleep(120)
    return 0

cli._run = measuring
cli._exit_as_program(sys.argv[2:])
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_run_as_a_program_a_cancel_prints_no_traceback_and_ends_by_sigint(tmp_path):
    """The job runner's stop is SIGINT, and it reads a status of "ended by
    SIGINT" as the cancel's doing (`TrainingManager._ended_by_stop`). The run,
    under the GPU lock and its execution settings, is interrupted mid-run: one
    line, no traceback, and the process ends by SIGINT. Stdout is buffered, as
    on the runner's pipe (no PYTHONUNBUFFERED, which this sandbox sets): what
    the run printed is still in the job's output, although that ending skips
    the interpreter's last flush."""
    import signal
    import time

    from src.utils.gpu_lock import LOCK_FILE_ENV, WORKSPACES_ROOT_ENV

    script = tmp_path / "child.py"
    script.write_text(_CANCELLED_CHILD)
    env = {k: v for k, v in os.environ.items()
           if k not in (LOCK_FILE_ENV, WORKSPACES_ROOT_ENV, "PYTHONUNBUFFERED")}
    child = subprocess.Popen(
        [sys.executable, str(script), str(REPO_ROOT),
         *_generated(tmp_path / "ws", tmp_path / "c.pt", tmp_path / "out" / "m.json")],
        cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        lines = []
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            line = child.stderr.readline()
            lines.append(line)
            if "MEASURING" in line or not line:
                break
        assert any("MEASURING" in line for line in lines), "".join(lines)
        child.send_signal(signal.SIGINT)
        out, rest = child.communicate(timeout=60)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    err = "".join(lines) + rest
    assert child.returncode == -signal.SIGINT, err
    assert _one_line(err, "measure_scorer: cancelled") == (
        __import__("scripts.measure_scorer", fromlist=["CANCELLED_LINE"]).CANCELLED_LINE)
    assert "KeyboardInterrupt" not in err
    assert out.splitlines() == ["PRINTED BEFORE THE CANCEL"], out


_LOADING_CHILD = """
import signal, sys, time
sys.path.insert(0, sys.argv[1])
import scripts.measure_scorer as cli

# What the script installs, run as a program, before it imports torch.
signal.signal(signal.SIGINT, cli._cancelled_while_loading)
print("LOADING", file=sys.stderr, flush=True)
time.sleep(120)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_a_cancel_while_the_modules_load_is_the_same_line_and_ending(tmp_path):
    """A Cancel right after Start lands while torch is still being imported,
    before `_run_as_program`: the same one line, no traceback, ended by
    SIGINT."""
    import signal
    import time

    script = tmp_path / "child.py"
    script.write_text(_LOADING_CHILD)
    child = subprocess.Popen([sys.executable, str(script), str(REPO_ROOT)], cwd=str(REPO_ROOT),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        lines = []
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            line = child.stderr.readline()
            lines.append(line)
            if "LOADING" in line or not line:
                break
        child.send_signal(signal.SIGINT)
        _out, rest = child.communicate(timeout=60)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    err = "".join(lines) + rest
    assert child.returncode == -signal.SIGINT, err
    assert "measure_scorer: cancelled" in err and "Traceback" not in err, err


def test_deterministic_turns_cudnn_benchmark_off_for_the_run_and_puts_it_back(cli, monkeypatch):
    """M5-3 review ARCH-10 (c): torch's default is already off, so a run
    asserting it off proves nothing; with benchmark left on by an in-process
    caller, the deterministic settings turn it off for the run only."""
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    before = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = True
    try:
        with cli._execution_settings(True):
            assert torch.backends.cudnn.benchmark is False
            assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is True
        with cli._execution_settings(False):
            assert torch.backends.cudnn.benchmark is True       # nothing changed without it
    finally:
        torch.backends.cudnn.benchmark = before


@pytest.mark.skipif(not Path("/proc/self/maps").exists(), reason="reads /proc (Linux)")
def test_a_cancel_while_the_real_program_imports_torch_prints_the_line_and_no_traceback():
    """M5-3 review ARCH-9: the test above installs the handler itself, so a
    broken guard in the program (``if __name__ == "main":``, say) passed every
    test while the real Cancel printed a traceback again. This runs the real
    program and sends SIGINT once torch's library is mapped into it, while
    torch is still being imported (before ``--help`` could print anything)."""
    import signal
    import time

    child = subprocess.Popen([sys.executable, str(REPO_ROOT / "scripts" / "measure_scorer.py"),
                              "--help"], cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    try:
        maps = Path(f"/proc/{child.pid}/maps")
        deadline = time.monotonic() + 120
        loading = False
        while child.poll() is None and time.monotonic() < deadline:
            try:
                text = maps.read_text()
            except OSError:
                break
            if "libtorch" in text or "torch/_C" in text:
                loading = True
                break
            time.sleep(0.002)
        assert loading, "torch was never seen loading in the program"
        child.send_signal(signal.SIGINT)
        out, err = child.communicate(timeout=120)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    assert "usage:" not in out, "the signal arrived after the imports (the help was printed)"
    assert child.returncode == -signal.SIGINT, err
    assert "measure_scorer: cancelled" in err and "Traceback" not in err, err


def test_the_loading_guard_comes_before_torch_and_the_run_unwinds_on_a_cancel(cli, monkeypatch):
    """Run as a program, the guard is installed before torch is imported; once
    `_run_as_program` starts, a cancel is a KeyboardInterrupt again, so the
    run unwinds (staged files removed, the GPU lock released) before it ends."""
    import signal

    tree = ast.parse(Path(cli.__file__).read_text())
    guard = next(i for i, node in enumerate(tree.body)
                 if isinstance(node, ast.If) and "_cancelled_while_loading" in ast.unparse(node))
    torch_import = next(i for i, node in enumerate(tree.body)
                        if isinstance(node, ast.Import) and node.names[0].name == "torch")
    assert guard < torch_import
    seen = []
    monkeypatch.setattr(cli, "main",
                        lambda argv=None: seen.append(signal.getsignal(signal.SIGINT)) or 0)
    previous = signal.signal(signal.SIGINT, cli._cancelled_while_loading)
    try:
        assert cli._run_as_program([]) == 0
    finally:
        signal.signal(signal.SIGINT, previous)
    assert seen == [signal.default_int_handler]
