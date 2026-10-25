"""
M4 end to end: import, measure, keep, register, read back -- through the real CLIs.
==================================================================================
Contract M4 (item 14, phase 1) acceptance, spec §6, on CPU. Every case runs the
production entry points in process -- `scripts/import_cohort.py`,
`scripts/measure_scorer.py`, `scripts/record_evaluation.py`, each through its
`main` -- on the demo workspace (real HPO and MONDO ids), a checkpoint over it,
and a registry in a tmp directory outside the repository, and asserts what they
leave on disk: the registry, the report store, and the ledger's bytes and
records.

The module imports one version through the CLI's preview and commit; each test
works on its own copy of that tree (`copy_registered_cohort`), so no test sees
another's reports or ledger.

> **Nothing here is an acceptance result about the scorer.** The checkpoint is
> untrained and the graph synthetic; what is accepted is the flow: what is
> refused, what is kept, what is recorded and what each label may claim.

Module: tests/integration/test_m4_flow.py
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tests.fixtures.opens import count_opens  # noqa: E402
from tests.fixtures.registered_cohort import (  # noqa: E402
    GRAPH_SOURCE_COUNTS,
    LABEL,
    ROLE,
    SOURCE_FILENAME,
    RegisteredCohort,
    build_registered_cohort,
    copy_registered_cohort,
    graph_source_cases,
    write_demo_workspace,
    write_fixture_checkpoint,
    write_source,
)
from tests.fixtures.replacement import HOWS, replace_after_read  # noqa: E402

LEDGER = "evaluations.json"
REPO = Path(__file__).resolve().parents[2]


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tree(root: Path):
    """Every file under `root` with its digest: what a step left on disk."""
    return {str(p.relative_to(root)): _sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def _import(registry, data_dir, source, label, *extra):
    """`import_cohort.main`, its stdout, and its return code."""
    import scripts.import_cohort as import_cohort

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = import_cohort.main(["--registry", str(registry), "--data-dir", str(data_dir),
                                   "--source", str(source), "--label", label, "--role", ROLE,
                                   *extra])
    return code, out.getvalue()


def _preview_and_commit(registry, data_dir, source, label):
    code, out = _import(registry, data_dir, source, label)
    assert code == 0, out
    preview_id = re.search(r"Preview id: ([0-9a-f]{16})", out).group(1)
    committed, committed_out = _import(registry, data_dir, source, label, "--commit", preview_id)
    assert committed == 0, committed_out
    return out, committed_out


@pytest.fixture(scope="module")
def flow(tmp_path_factory):
    """A workspace, a checkpoint and a source, and a version imported through
    the CLI: the preview first (its effect on the registry recorded), then the
    commit under the id the preview printed."""
    from src.evaluation.cohort_registry import resolve_version

    root = tmp_path_factory.mktemp("m4_flow").resolve() / "root"
    root.mkdir()
    data_dir = write_demo_workspace(root / "workspace")
    checkpoint = root / "checkpoints" / "best.pt"
    checkpoint_sha256 = write_fixture_checkpoint(data_dir, checkpoint)
    source = write_source(root / SOURCE_FILENAME, graph_source_cases(data_dir))
    registry = root / "registry"
    registry.mkdir()

    before = _tree(registry)
    code, preview_out = _import(registry, data_dir, source, LABEL)
    after_preview = _tree(registry)
    preview_id = re.search(r"Preview id: ([0-9a-f]{16})", preview_out).group(1)
    committed, commit_out = _import(registry, data_dir, source, LABEL, "--commit", preview_id)

    cohort = RegisteredCohort(data_dir, checkpoint, checkpoint_sha256, registry, source,
                              resolve_version(registry, LABEL))
    return SimpleNamespace(cohort=cohort, preview_code=code, preview_out=preview_out,
                           before=before, after_preview=after_preview,
                           commit_code=committed, commit_out=commit_out)


@pytest.fixture
def cohort(flow, tmp_path, monkeypatch):
    """A copy of the imported cohort, its registry the deployment's
    (`SHEPHERD_REGISTRY`, inherited by a command run as printed): a
    ``--cohort`` run measures, and a report is registered, only there
    (BACKLOG 37)."""
    copied = copy_registered_cohort(flow.cohort, tmp_path / "root")
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(copied.registry))
    return copied


def _measure(cohort, *extra, data_dir=None, checkpoint=None, label=LABEL):
    import scripts.measure_scorer as measure_scorer

    return measure_scorer.main([
        "--checkpoint", str(checkpoint or cohort.checkpoint),
        "--data-dir", str(data_dir or cohort.data_dir),
        "--cohort", label, "--registry", str(cohort.registry), "--device", "cpu", *extra,
    ])


def _record(report, checkpoint_dir):
    import scripts.record_evaluation as record_evaluation

    return record_evaluation.main(["--report", str(report), "--checkpoint-dir",
                                   str(checkpoint_dir)])


def _record_refused(report, checkpoint_dir) -> str:
    """A refused registration: `record_evaluation` exits non-zero with its one
    message."""
    with pytest.raises(SystemExit) as refused:
        _record(report, checkpoint_dir)
    assert refused.value.code not in (0, None)
    return str(refused.value.code)


def _reports(cohort, label=LABEL):
    return sorted((cohort.registry / "versions" / label / "reports").glob("*/report.json"),
                  key=lambda p: p.stat().st_mtime_ns)


def _only_report(cohort):
    (report,) = _reports(cohort)
    return report


def _ledger(cohort):
    return cohort.checkpoint.parent / LEDGER


def _records(cohort):
    return json.loads(_ledger(cohort).read_text())["records"]


def _r10(report):
    return json.loads(Path(report).read_text())["manifest"]["r10"]


def _printed_command(text: str) -> str:
    """The register-later line a run printed, as the operator would copy it."""
    lines = text.splitlines()
    (index,) = [i for i, line in enumerate(lines) if line.rstrip().endswith("Register it with:")
                or line.rstrip() == "Register it later with:"]
    command = lines[index + 1].strip()
    assert command.startswith("python scripts/record_evaluation.py "), command
    return command


def _run_as_printed(command: str) -> subprocess.CompletedProcess:
    """The printed line pasted into a POSIX shell at the repository root, with
    this interpreter for `python`."""
    line = shlex.quote(sys.executable) + command[len("python"):]
    return subprocess.run(line, shell=True, cwd=REPO, capture_output=True, text=True,
                          timeout=120)


def _served_view(monkeypatch, cohort, served: Path) -> dict:
    """`GET /api/v1/evaluations/loaded-model` as the server answers it, with a
    real pipeline loaded from `served` -- the path a deployment configures --
    published."""
    from fastapi.testclient import TestClient

    from src.api import main as api_main
    from src.inference.pipeline import DiagnosisPipeline
    from src.kg.graph import KnowledgeGraph

    for name in ("SHEPHERD_KG_PATH", "SHEPHERD_DATA_DIR", "SHEPHERD_CHECKPOINT_PATH"):
        monkeypatch.delenv(name, raising=False)
    pipeline = DiagnosisPipeline(graph=KnowledgeGraph.read_json(cohort.data_dir / "kg.json"),
                                 data_dir=str(cohort.data_dir), checkpoint_path=str(served),
                                 device="cpu")
    monkeypatch.setattr(api_main.app_state, "real_pipeline_requested", False, raising=False)
    monkeypatch.setattr(api_main.app_state, "pipeline", pipeline, raising=False)
    response = TestClient(api_main.app).get("/api/v1/evaluations/loaded-model")
    assert response.status_code == 200, response.text
    return response.json()


def _nothing_written(cohort, label=LABEL):
    return (not (cohort.registry / "versions" / label / "reports").exists()
            and not _ledger(cohort).exists())


# ---------------------------------------------------------------------------
# Import: preview, then commit
# ---------------------------------------------------------------------------
def test_the_import_preview_writes_nothing_and_its_printed_id_commits(flow):
    """The preview leaves the registry exactly as it was and prints an id; the
    commit under that id publishes the source copy, the samples and the manifest,
    whose counts are the graph source's."""
    assert flow.preview_code == 0
    assert flow.before == flow.after_preview == {}
    assert "Nothing was written" in flow.preview_out
    assert flow.commit_code == 0
    version = flow.cohort.version
    assert f"directory        {version.directory}" in flow.commit_out
    assert set(_tree(flow.cohort.registry)) == {
        f"sources/{_sha(flow.cohort.source)}.json",
        f"versions/{LABEL}/cohort.json", f"versions/{LABEL}/samples.json",
    }
    assert version.manifest["counts"] == GRAPH_SOURCE_COUNTS["keep"]


# ---------------------------------------------------------------------------
# Measure, keep, register
# ---------------------------------------------------------------------------
def test_measure_and_record_keeps_the_report_and_files_one_record(cohort):
    """The report is kept in the report store and registered beside the
    checkpoint: one record, keyed by the checkpoint's and the version manifest's
    digests, carrying the report's metrics, counts, relations and dataset record
    and naming the report by the digest of its bytes."""
    from src.evaluation.sidecar import REGISTERED_RELATIONS

    assert _measure(cohort, "--record") == 0

    report = _only_report(cohort)
    parsed = json.loads(report.read_text())
    (record,) = _records(cohort)
    assert record["checkpoint_digest"] == cohort.checkpoint_sha256
    assert record["cohort_role"] == ROLE
    assert record["cohort_digest"] == _sha(cohort.version.directory / "cohort.json")
    assert record["source_artifact_digest"] == _sha(report)
    assert record["metrics"] == parsed["authoritative_metrics"]
    assert record["n_ranked"] == GRAPH_SOURCE_COUNTS["keep"]["ranked_eligible"]
    assert record["cohort"]["counts"] == GRAPH_SOURCE_COUNTS["keep"]
    assert record["relations"] == REGISTERED_RELATIONS
    assert record["dataset_record"]["origin"] == "synthetic"
    assert record["corroborating_reports"] == []
    assert record["registration"]["sequence"] == 1


def test_registering_the_same_report_again_is_a_no_op(cohort):
    assert _measure(cohort, "--record") == 0
    before = _ledger(cohort).read_bytes()

    assert _record(_only_report(cohort), cohort.checkpoint.parent) == 0

    assert _ledger(cohort).read_bytes() == before


def test_a_remeasurement_with_other_report_bytes_corroborates(cohort, tmp_path, capsys):
    """The same checkpoint and version measured from a copy of the workspace: the
    report's paths differ, its key and every compared field agree, so its digest
    joins the first record's corroborating reports; no second record."""
    assert _measure(cohort, "--record") == 0
    copy = tmp_path / "workspace-copy"
    shutil.copytree(cohort.data_dir, copy)

    assert _measure(cohort, "--record", data_dir=copy) == 0

    first, second = _reports(cohort)
    assert first.read_bytes() != second.read_bytes()
    (record,) = _records(cohort)
    assert record["source_artifact_digest"] == _sha(first)
    assert record["corroborating_reports"] == [_sha(second)]
    assert "Agrees with record 1" in capsys.readouterr().out


def _beside_in_the_store(report: Path, data: bytes) -> Path:
    """`data` as the report of a new run directory beside `report`'s, named for
    the same checkpoint, as another run of it would be kept."""
    import secrets

    stamp, checkpoint, _ = report.parent.name.split("-")
    run_dir = report.parents[1] / f"{stamp}-{checkpoint}-{secrets.token_hex(4)}"
    run_dir.mkdir()
    (run_dir / "report.json").write_bytes(data)
    return run_dir / "report.json"


def test_a_contradicting_run_blocks_its_key_and_the_ledger_is_unchanged(cohort):
    """D5 (since M5-1): a second run under the same key with an edited metric,
    kept in the store. Registering it is refused ("runs disagree --
    investigate"), naming the metric, and so is registering the first report
    again; the ledger keeps its bytes."""
    assert _measure(cohort, "--record") == 0
    first = _only_report(cohort)
    before = _ledger(cohort).read_bytes()
    edited = json.loads(first.read_text())
    edited["authoritative_metrics"]["hits@1"] = 0.5 + edited["authoritative_metrics"]["hits@1"] / 2
    second = _beside_in_the_store(first, json.dumps(edited).encode("utf-8"))

    for report in (second, first):
        message = _record_refused(report, cohort.checkpoint.parent)
        assert "runs disagree — investigate" in message and "hits@1" in message
        assert "Nothing was written" in message
    assert _ledger(cohort).read_bytes() == before


def test_a_copy_outside_the_store_is_refused_and_the_original_registers(cohort, tmp_path):
    """Only the store's original is registered (D5, 2026-10-19): a copy is
    refused saying so, and the original the run printed registers."""
    assert _measure(cohort) == 0
    original = _only_report(cohort)
    copy = tmp_path / "shared" / "report.json"
    copy.parent.mkdir()
    copy.write_bytes(original.read_bytes())

    message = _record_refused(copy, cohort.checkpoint.parent)

    assert "is not a report the registry's report store holds" in message
    assert "Register the store's original instead" in message
    assert not _ledger(cohort).exists()
    assert _record(original, cohort.checkpoint.parent) == 0


@pytest.mark.parametrize("damage", ["truncated", "not_json", "not_a_report"])
def test_a_damaged_report_is_refused_and_nothing_is_written(cohort, damage):
    assert _measure(cohort) == 0
    original = _only_report(cohort)
    data = original.read_bytes()
    original.unlink()  # the store holds only the damaged report, so D5 has nothing to compare
    damaged = _beside_in_the_store(original, {
        "truncated": data[: len(data) // 2], "not_json": b"\xff\xfe",
        "not_a_report": b"[1, 2, 3]"}[damage])

    message = _record_refused(damaged, cohort.checkpoint.parent)

    assert str(damaged) in message and "Nothing was written" in message
    assert not _ledger(cohort).exists()


@pytest.mark.parametrize("how", HOWS)
def test_a_report_replaced_while_it_is_registered_cannot_be_paired(
    cohort, tmp_path, monkeypatch, how
):
    """Report A is read; B replaces it at once. The record carries A's numbers
    under A's digest -- never B's digest beside A's data (BACKLOG 18)."""
    assert _measure(cohort) == 0
    report = _only_report(cohort)
    a_bytes, a = report.read_bytes(), json.loads(report.read_text())
    b = json.loads(report.read_text())
    b["authoritative_metrics"]["hits@1"] = 0.5 + b["authoritative_metrics"]["hits@1"] / 2
    replace = replace_after_read(monkeypatch, after=report, target=report,
                                 data=json.dumps(b).encode("utf-8"), how=how)

    assert _record(report, cohort.checkpoint.parent) == 0

    assert replace.fired == 1
    (record,) = _records(cohort)
    assert record["source_artifact_digest"] == hashlib.sha256(a_bytes).hexdigest()
    assert record["metrics"] == a["authoritative_metrics"]
    assert _sha(report) != record["source_artifact_digest"]


def _other_checkpoint(cohort, root: Path) -> Path:
    """Another valid checkpoint over the same verified graph: the fixture's
    weights with one tensor and the epoch changed, written by the production
    writer, so its digest differs and measurement accepts it."""
    from src.utils.checkpoint_io import read_checkpoint, write_checkpoint

    payload = read_checkpoint(cohort.checkpoint, map_location="cpu",
                              weights_only=True).checkpoint
    payload["epoch"] = int(payload.get("epoch", 0)) + 1
    next(v for v in payload["state_dict"].values() if v.is_floating_point()).add_(0.125)
    path = root / "another-model" / "best.pt"
    digest = write_checkpoint(payload, path)
    assert digest != cohort.checkpoint_sha256
    return path


def _swap_after_publication(monkeypatch, data: bytes, how: str) -> list:
    """Replace the report `--record` is about to register, right after the run
    published it and before the registration reads it: by atomic rename (a new
    inode) or by rewriting the same file in place."""
    import scripts.measure_scorer as measure_scorer

    publish = measure_scorer._publish_to_store
    swapped = []

    def publish_then_replace(result, inputs):
        run_id, path, digest = publish(result, inputs)
        if how == "rename":
            staged = path.with_name(path.name + ".replacement")
            staged.write_bytes(data)
            os.replace(staged, path)
        else:
            with open(path, "r+b") as handle:
                handle.write(data)
                handle.truncate()
        swapped.append(path)
        return run_id, path, digest

    monkeypatch.setattr(measure_scorer, "_publish_to_store", publish_then_replace)
    return swapped


@pytest.mark.parametrize("how", HOWS)
@pytest.mark.parametrize("whose", ["another_checkpoint", "another_cohort_version"])
def test_record_registers_only_the_report_this_run_published(
    cohort, tmp_path, monkeypatch, capsys, whose, how
):
    """The reviewer's P2 (2026-10-17): the report `--record` reads back is
    replaced, after its publication and before the registration, by another
    valid report -- another checkpoint's on this version, or this checkpoint's
    on another version of the same source. The run measured, so it exits 3, and
    nothing is registered: no ledger beside the measured checkpoint or the
    other one, and no claim that the run's report is kept."""
    if whose == "another_checkpoint":
        other = _other_checkpoint(cohort, tmp_path)
        assert _measure(cohort, checkpoint=other) == 0
        replacement = _only_report(cohort)
    else:
        _preview_and_commit(cohort.registry, cohort.data_dir, cohort.source, "demo-test-2")
        assert _measure(cohort, label="demo-test-2") == 0
        (replacement,) = _reports(cohort, label="demo-test-2")
    data = replacement.read_bytes()
    capsys.readouterr()
    swapped = _swap_after_publication(monkeypatch, data, how)

    assert _measure(cohort, "--record") == 3

    (path,) = swapped
    assert path.read_bytes() == data, "the injected replacement is what was read"
    assert not _ledger(cohort).exists()
    assert not (tmp_path / "another-model" / LEDGER).exists()
    err = capsys.readouterr().err
    assert "is not the report this run published" in err
    assert "re-measure" in err
    assert "Register it later with" not in err, (
        "a replaced file is not offered for a later registration in the run's name")


def _lose_after_publication(monkeypatch, how: str) -> list:
    """Damage or remove the report `--record` is about to register, right after
    the run published it."""
    import scripts.measure_scorer as measure_scorer

    publish = measure_scorer._publish_to_store
    lost = []

    def publish_then_lose(result, inputs):
        run_id, path, digest = publish(result, inputs)
        if how == "missing":
            path.unlink()
        else:
            staged = path.with_name(path.name + ".replacement")
            staged.write_bytes(b"{broken")
            os.replace(staged, path)
        lost.append(path)
        return run_id, path, digest

    monkeypatch.setattr(measure_scorer, "_publish_to_store", publish_then_lose)
    return lost


@pytest.mark.parametrize("how", ["damaged_json", "missing"])
def test_record_never_calls_a_lost_report_kept(cohort, monkeypatch, capsys, how):
    """The reviewer's P3 (2026-10-17): a published report damaged or removed
    before `--record` reads it is refused (exit 3, no ledger), and the run
    neither says its report is kept nor prints a command to register it later."""
    lost = _lose_after_publication(monkeypatch, how)

    assert _measure(cohort, "--record") == 3

    assert len(lost) == 1
    assert not _ledger(cohort).exists()
    err = capsys.readouterr().err
    assert "re-measure" in err
    assert "is kept at" not in err
    assert "Register it later with" not in err


def test_a_ledger_refusal_of_the_published_report_keeps_it_for_later(cohort, capsys):
    """The control: when the report is the one published and the ledger is
    what refuses (here a schema 2 ledger beside the checkpoint), the report is
    kept and the run prints the command that registers it later."""
    ledger = _ledger(cohort)
    ledger.write_text(json.dumps({"schema_version": 2, "records": []}))
    before = ledger.read_bytes()

    assert _measure(cohort, "--record") == 3

    assert ledger.read_bytes() == before
    err = capsys.readouterr().err
    assert f"its report is kept at {_only_report(cohort)}" in err
    assert "Register it later with" in err


def test_record_after_the_swap_check_still_registers_an_untouched_report(cohort, monkeypatch):
    """The control for the test above: the same wrapper, replacing nothing,
    leaves the published bytes in place, and `--record` registers them."""
    import scripts.measure_scorer as measure_scorer

    publish = measure_scorer._publish_to_store
    digests = []

    def publish_and_note(result, inputs):
        run_id, path, digest = publish(result, inputs)
        digests.append((path, digest))
        return run_id, path, digest

    monkeypatch.setattr(measure_scorer, "_publish_to_store", publish_and_note)
    assert _measure(cohort, "--record") == 0
    ((path, digest),) = digests
    assert _sha(path) == digest, "publish_bytes returned the digest of the bytes written"
    (record,) = _records(cohort)
    assert record["source_artifact_digest"] == digest
    assert record["checkpoint_digest"] == cohort.checkpoint_sha256


def test_not_registering_keeps_the_report_for_a_later_registration(cohort):
    assert _measure(cohort) == 0

    report = _only_report(cohort)
    assert not _ledger(cohort).exists()
    assert _record(report, cohort.checkpoint.parent) == 0
    (record,) = _records(cohort)
    assert record["source_artifact_digest"] == _sha(report)


def test_a_generated_split_report_is_refused_by_record_evaluation(tmp_path):
    """BEHAVIOUR CHANGE (M4 decision 9): the ledger records results on registered
    test cohort versions only; a generated `val` report is selection data. Since
    M5-1 it is refused first for where it is: written to `--output`, never into a
    report store, it is not a report the store holds. (One placed in a store is
    refused by `build_record` as a generated split's,
    `tests/unit/test_registration.py`.)"""
    import scripts.measure_scorer as measure_scorer
    from tests.fixtures.synthetic_workspace import build_workspace

    data_dir, checkpoint = build_workspace(tmp_path / "ws")
    output = tmp_path / "out" / "measurement.json"
    assert measure_scorer.main(["--checkpoint", str(checkpoint), "--data-dir", str(data_dir),
                                "--split", "val", "--output", str(output),
                                "--device", "cpu"]) == 0

    message = _record_refused(output, checkpoint.parent)

    assert "is not a report the registry's report store holds" in message
    assert "Nothing was written" in message
    assert not (checkpoint.parent / LEDGER).exists()


def test_record_into_a_read_only_checkpoint_directory_keeps_the_report_and_exits_3(
    cohort, monkeypatch, capsys
):
    """The measurement succeeded and its report is kept; the registration could
    not write the ledger, so the run exits 3 naming the report and the command
    that registers it, which then works once the directory is writable."""
    import src.evaluation.sidecar as sidecar

    real = sidecar.create_staging_file

    def read_only(*args, **kwargs):
        raise PermissionError(13, "Permission denied", str(cohort.checkpoint.parent))

    monkeypatch.setattr(sidecar, "create_staging_file", read_only)

    assert _measure(cohort, "--record") == 3

    report = _only_report(cohort)
    err = capsys.readouterr().err
    assert "NOT REGISTERED" in err and "Permission denied" in err
    command = _printed_command(err)
    assert f'--report "{report}" --checkpoint "{cohort.checkpoint.resolve()}"' in command
    assert not _ledger(cohort).exists()
    monkeypatch.setattr(sidecar, "create_staging_file", real)
    done = _run_as_printed(command)
    assert done.returncode == 0, done.stderr
    assert len(_records(cohort)) == 1


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="root writes through directory permissions")
def test_record_into_a_directory_without_write_permission_exits_3(cohort):
    """The same, with the directory's real permissions (not as root)."""
    directory = cohort.checkpoint.parent
    os.chmod(directory, 0o555)
    try:
        assert _measure(cohort, "--record") == 3
    finally:
        os.chmod(directory, 0o755)
    assert _only_report(cohort).is_file()
    assert not _ledger(cohort).exists()


@pytest.mark.parametrize("extra, fragment", [
    (["--output", "out.json"], "--output does not apply to --cohort"),
    (["--record"], "--record apply to"),
    (["--training-evidence", "."], "--training-evidence apply to"),
    (["--registry", "."], "--registry apply to"),
], ids=["cohort+output", "split+record", "split+evidence", "split+registry"])
def test_arguments_of_the_other_kind_of_cohort_are_refused(
    cohort, tmp_path, monkeypatch, capsys, extra, fragment
):
    import scripts.measure_scorer as measure_scorer

    monkeypatch.chdir(tmp_path)
    common = ["--checkpoint", str(cohort.checkpoint), "--data-dir", str(cohort.data_dir),
              "--device", "cpu"]
    if extra[0] == "--output":
        argv = [*common, "--cohort", LABEL, "--registry", str(cohort.registry), *extra]
    else:
        argv = [*common, "--split", "val", "--output", "out.json", *extra]

    with pytest.raises(SystemExit) as refused:
        measure_scorer.main(argv)

    assert refused.value.code == 2
    assert fragment in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()
    assert _nothing_written(cohort)


@pytest.mark.parametrize("folder, name", [("My Models", "best.pt"), ("weights", "best.ckpt")],
                         ids=["space-in-path", "not-named-pt"])
def test_the_printed_command_run_as_printed_registers_the_measured_file(
    cohort, tmp_path, capsys, folder, name
):
    """Findings 1 and 14: the command a run without `--record` prints, pasted
    into a shell as printed, registers the report -- for a checkpoint whose
    path holds a space, and for one not named `*.pt`, which `--checkpoint`
    names directly -- in the ledger beside the measured file."""
    measured = tmp_path / folder / name
    measured.parent.mkdir()
    shutil.copy2(cohort.checkpoint, measured)
    assert _measure(cohort, checkpoint=measured) == 0
    command = _printed_command(capsys.readouterr().out)

    done = _run_as_printed(command)

    assert done.returncode == 0, done.stderr
    report = _only_report(cohort)
    (record,) = json.loads((measured.parent / LEDGER).read_text())["records"]
    assert record["source_artifact_digest"] == _sha(report)
    assert record["checkpoint_digest"] == cohort.checkpoint_sha256


@pytest.mark.parametrize("flag", ["--checkpoint-dir", "--checkpoint"])
def test_a_report_registered_through_a_link_is_shown_for_the_model_served_through_it(
    cohort, tmp_path, monkeypatch, flag
):
    """Findings 0/2/13/24/25: the deployment serves `served/best.pt`, a link to
    the run's checkpoint. Registering the kept report through the link's
    directory or the link itself files the ledger beside the file the link
    names -- where the service looks -- so the model status shows the record;
    no second ledger is made beside the link."""
    assert _measure(cohort) == 0
    served = tmp_path / "served"
    served.mkdir()
    (served / "best.pt").symlink_to(cohort.checkpoint)

    assert _record_through(_only_report(cohort), flag,
                           served if flag == "--checkpoint-dir" else served / "best.pt") == 0

    assert len(_records(cohort)) == 1
    assert not (served / LEDGER).exists()
    view = _served_view(monkeypatch, cohort, served / "best.pt")
    assert view["state"] == "records"
    assert view["checkpoint_sha256"] == cohort.checkpoint_sha256
    assert len(view["records"]) == 1


def _record_through(report, flag, named):
    import scripts.record_evaluation as record_evaluation

    return record_evaluation.main(["--report", str(report), flag, str(named)])


def test_a_corroborating_rerun_that_found_r10_evidence_says_the_record_keeps_its_own(
    cohort, tmp_path, capsys
):
    """Finding 3: measured with `--record` from a re-sampled workspace (R10
    unverifiable), then again with the training workspace as evidence: the
    second run corroborates, the record keeps its own unverifiable labels, and
    the console says so, naming the report that holds the labels this run
    found, rather than reporting plain agreement."""
    resampled = write_demo_workspace(tmp_path / "resampled", sample_seed=7)
    assert _measure(cohort, "--record", data_dir=resampled) == 0
    capsys.readouterr()

    assert _measure(cohort, "--record", "--training-evidence", str(cohort.data_dir),
                    data_dir=resampled) == 0

    out = capsys.readouterr().out
    second = _reports(cohort)[-1]
    assert "Agrees with record 1" in out
    assert re.search(
        r"This report's r10( and environment)? differ from record 1's; they are not "
        r"compared, so the record keeps its own and this report's are in "
        + re.escape(str(second)) + r"\.", out), out
    (record,) = _records(cohort)
    _unverifiable_everywhere(record["r10"])
    assert _r10(second)["training"]["evidence"]["state"] == "found"


# ---------------------------------------------------------------------------
# Refused before the checkpoint is read, with nothing written
# ---------------------------------------------------------------------------
def _refused_before_the_checkpoint(cohort, match, **kwargs):
    with count_opens() as opens:
        with pytest.raises(ValueError, match=match) as refused:
            _measure(cohort, **kwargs)
    assert opens.opens(cohort.checkpoint) == 0, "the checkpoint was read"
    return refused.value


def test_a_label_that_does_not_exist_is_refused(cohort):
    _refused_before_the_checkpoint(cohort, "holds no cohort version", label="never-imported")
    assert _nothing_written(cohort)


def test_a_version_mapped_against_another_graph_is_refused(cohort, tmp_path):
    """The same source imported against another export of the demo graph (other
    node features), then measured against this workspace: its indices name that
    export's nodes, so the binding refuses, naming the role."""
    other = write_demo_workspace(tmp_path / "other-export", feature_seed=7)
    _preview_and_commit(cohort.registry, other, cohort.source, "other-graph")

    refused = _refused_before_the_checkpoint(cohort, "mapped against another graph",
                                             label="other-graph")

    assert refused.roles == ("node_features",)
    assert _nothing_written(cohort, label="other-graph")


def test_a_dataset_record_mismatch_is_refused(cohort):
    """The workspace's dataset record replaced by another valid, bound record for
    the same graph: the version was mapped under the first, so it is refused."""
    from src.kg.artifacts import MANIFEST_FILENAME
    from src.kg.provenance import PROVENANCE_FILENAME, build_provenance, encode_provenance

    manifest_path = cohort.data_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    record = build_provenance(manifest["artifacts"]["kg"], counters={"rebuilt": 1},
                              origin="synthetic")
    data = encode_provenance(record).encode("utf-8")
    (cohort.data_dir / PROVENANCE_FILENAME).write_bytes(data)
    manifest["artifacts"]["provenance"] = hashlib.sha256(data).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    refused = _refused_before_the_checkpoint(cohort, "mapped against another graph")

    assert refused.roles == ("dataset_record",)
    assert _nothing_written(cohort)


def test_a_tampered_samples_file_is_refused_by_digest(cohort):
    samples = json.loads(cohort.version.samples_path.read_text())
    samples[0], samples[1] = samples[1], samples[0]
    cohort.version.samples_path.write_text(json.dumps(samples))

    _refused_before_the_checkpoint(cohort, "is not the samples")

    assert _nothing_written(cohort)


def test_a_checkpoint_moved_with_unchanged_bytes_keeps_its_records(cohort, tmp_path, capsys):
    """Records are found by digest, not file name: the checkpoint moved with its
    ledger is found by `--show`, and measuring it from the new place corroborates
    the record already there."""
    import scripts.record_evaluation as record_evaluation

    assert _measure(cohort, "--record") == 0
    moved = tmp_path / "elsewhere" / "checkpoints"
    shutil.move(str(cohort.checkpoint.parent), str(moved))
    capsys.readouterr()

    assert record_evaluation.main(["--checkpoint-dir", str(moved),
                                   "--show", cohort.checkpoint_sha256]) == 0
    (shown,) = json.loads(capsys.readouterr().out)
    assert shown["checkpoint_digest"] == cohort.checkpoint_sha256

    assert _measure(cohort, "--record", checkpoint=moved / "best.pt") == 0
    (record,) = json.loads((moved / LEDGER).read_text())["records"]
    assert len(record["corroborating_reports"]) == 1


# ---------------------------------------------------------------------------
# R10 (contract §5.4), through the CLI
# ---------------------------------------------------------------------------
def _states(scope_block):
    return {label: scope_block[label]["state"]
            for label in ("disease_overlap", "phenotype_set_overlap")}


def _unverifiable_everywhere(r10):
    for scope in ("training", "validation"):
        assert r10[scope]["evidence"]["state"] == "not_found"
        for label in ("disease_overlap", "phenotype_set_overlap"):
            assert r10[scope][label]["state"] == "unverifiable"
            assert r10[scope][label]["k"] is None, "an unverifiable label carries no count"


def test_r10_overlapping_diseases_with_disjoint_phenotype_sets(cohort):
    """The graph source shares diseases with training but none of its cases'
    phenotype sets: the two labels say different things, both from evidence."""
    assert _measure(cohort) == 0

    training = _r10(_only_report(cohort))["training"]
    assert training["evidence"]["state"] == "found"
    assert _states(training) == {"disease_overlap": "overlap", "phenotype_set_overlap": "none"}
    assert training["disease_overlap"]["k"] >= 1
    assert training["phenotype_set_overlap"]["k"] == 0


def test_r10_a_model_trained_on_another_split_is_unverifiable_until_its_evidence_is_given(
    cohort, tmp_path
):
    """The workspace re-sampled after training (same graph and dataset record,
    other `train` and `val`): its splits are not the bytes the checkpoint names,
    so R10 is unverifiable, never zero; the training workspace given as evidence
    gives the counts a run over it gives."""
    resampled = write_demo_workspace(tmp_path / "resampled", sample_seed=7)
    assert _measure(cohort, data_dir=resampled) == 0
    _unverifiable_everywhere(_r10(_reports(cohort)[-1]))

    assert _measure(cohort, "--training-evidence", str(cohort.data_dir),
                    data_dir=resampled) == 0
    with_evidence = _r10(_reports(cohort)[-1])
    assert _measure(cohort) == 0
    original = _r10(_reports(cohort)[-1])

    for scope in ("training", "validation"):
        assert with_evidence[scope]["evidence"]["state"] == "found"
        assert {k: v for k, v in with_evidence[scope].items() if k != "evidence"} == {
            k: v for k, v in original[scope].items() if k != "evidence"}


def test_r10_moved_evidence_is_found_through_training_evidence(cohort, tmp_path):
    """The bytes the checkpoint names, moved out of the workspace under other
    names: unverifiable without them, found by digest in the directory given."""
    evidence = tmp_path / "kept-inputs"
    evidence.mkdir()
    for split in ("train", "val"):
        shutil.move(str(cohort.data_dir / f"{split}_samples.json"),
                    str(evidence / f"run7_{split}_samples.json"))

    assert _measure(cohort) == 0
    _unverifiable_everywhere(_r10(_reports(cohort)[-1]))

    assert _measure(cohort, "--training-evidence", str(evidence)) == 0
    r10 = _r10(_reports(cohort)[-1])
    assert r10["training"]["evidence"]["path"] == str(evidence / "run7_train_samples.json")
    assert r10["validation"]["evidence"]["path"] == str(evidence / "run7_val_samples.json")


@pytest.mark.parametrize("case", ["missing", "altered"])
def test_r10_missing_or_altered_evidence_is_unverifiable(cohort, case):
    """`val` removed, or `train` with one byte changed: that scope's bytes are
    found nowhere, so its labels are unverifiable; the other scope is counted."""
    if case == "missing":
        (cohort.data_dir / "val_samples.json").unlink()
        lost, kept = "validation", "training"
    else:
        path = cohort.data_dir / "train_samples.json"
        path.write_bytes(path.read_bytes() + b" ")
        lost, kept = "training", "validation"

    assert _measure(cohort) == 0

    r10 = _r10(_only_report(cohort))
    assert r10[lost]["evidence"]["state"] == "not_found"
    assert set(_states(r10[lost]).values()) == {"unverifiable"}
    assert r10[lost]["disease_overlap"]["k"] is None
    assert r10[kept]["evidence"]["state"] == "found"


def test_r10_a_cohort_of_validation_cases_overlaps_validation_entirely(tmp_path, monkeypatch):
    """A source built from the workspace's own val cases: every test case's
    phenotype set is a validation case's (k = n), and the allocation keeps its
    diseases out of training."""
    cohort = build_registered_cohort(tmp_path / "root", source="val_cases")
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(cohort.registry))

    assert _measure(cohort) == 0

    r10 = _r10(_only_report(cohort))
    phenotype_sets = r10["validation"]["phenotype_set_overlap"]
    assert phenotype_sets["state"] == "overlap"
    assert phenotype_sets["k"] == phenotype_sets["n"] == 5
    assert r10["training"]["disease_overlap"]["state"] == "none"


def test_r10_counts_a_phenotype_set_whatever_its_order_and_repetition(tmp_path):
    """`[0, 1]` in training and `[0, 0, 1]` in the test cohort, one disease: the
    same set, so phenotype-set overlap counts it, though Mode C would score the
    two differently. No stored file can hold the repeat: the importer removes it
    from a test cohort, and the shared reader refuses a samples file with one
    (N2; the next test). So this case is shown on `compute_r10` directly, with
    the test cohort in memory."""
    from src.evaluation.r10 import compute_r10
    from src.kg.data_loader import DiagnosisSample

    train = tmp_path / "train_samples.json"
    train.write_text(json.dumps([{"patient_id": "t", "phenotype_ids": [0, 1], "disease_id": 2}]))
    test = [DiagnosisSample(patient_id="c", phenotype_ids=[0, 0, 1], disease_id=2)]

    r10 = compute_r10({"train_samples": _sha(train)}, test, [train])

    assert r10["training"]["phenotype_set_overlap"] == {"state": "overlap", "k": 1, "n": 1}
    assert r10["validation"]["disease_overlap"]["state"] == "no_validation_inputs_recorded"


def test_r10_evidence_not_in_phenotype_normal_form_is_unverifiable_and_the_run_completes(
    cohort, tmp_path
):
    """N2, through the CLI: the training file the checkpoint names lists a
    phenotype twice. The shared reader refuses it as `UnparseableFile`, so R10
    records that scope's evidence *unreadable* and its labels *unverifiable*, and
    the measurement completes rather than being refused. Training refuses such a
    file, so no run records one; the record is rewritten here by hand."""
    from src.utils.checkpoint_io import read_checkpoint, write_checkpoint

    rows = json.loads((cohort.data_dir / "train_samples.json").read_text())
    rows[0]["phenotype_ids"].append(rows[0]["phenotype_ids"][0])
    evidence = tmp_path / "kept-inputs" / "run7_train_samples.json"
    evidence.parent.mkdir()
    evidence.write_text(json.dumps(rows))
    payload = read_checkpoint(cohort.checkpoint, map_location="cpu",
                              weights_only=True).checkpoint
    payload["training_input_digests"]["train_samples"] = _sha(evidence)
    write_checkpoint(payload, cohort.checkpoint)

    assert _measure(cohort, "--training-evidence", str(evidence.parent)) == 0

    r10 = _r10(_only_report(cohort))
    assert r10["training"]["evidence"] == {
        "state": "unreadable",
        "path": str(evidence),
        "problem": "the recorded bytes could not be parsed as a samples file in phenotype "
                   "normal form",
    }
    assert set(_states(r10["training"]).values()) == {"unverifiable"}
    assert r10["training"]["disease_overlap"]["k"] is None
    assert r10["validation"]["evidence"]["state"] == "found", "the other scope carries on"
