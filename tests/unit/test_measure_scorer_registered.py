"""
Measuring a registered cohort version: order, refusals, report store, --record.
==============================================================================
Contract M4 §6 (decisions 8, 15, 16) at `scripts/measure_scorer.py`'s entry
point, on the registered-cohort fixture (`tests/fixtures/registered_cohort.py`:
the demo graph, a checkpoint over it, a version committed through the production
importer), built once per module and copied per test:

- the reads and checks run in the specified order, and **nothing is written --
  no run directory, no ledger -- until every one of them has passed**, R10's
  evidence-directory check included;
- each refusal the registered path adds is one `ValueError` naming the file and
  saying nothing was written: a version whose manifest lacks its source format,
  missing samples, an R10 candidate that cannot be read;
- the manifest records the version as an aggregate block (no case id), the
  relation states a registered run earns, and the R10 block `compute_r10` gives
  for the same inputs; R10's states for each kind of training record;
- the report store's commit order; `--record` registers through
  `register_report` into the ledger beside the checkpoint read, without hashing
  the directory again; a refused registration exits 3 and keeps the report
  unless it is no longer the one the run published.

The end-to-end acceptance list, through the real CLIs, is
`tests/integration/test_m4_flow.py`.

Module: tests/unit/test_measure_scorer_registered.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests.fixtures.opens import count_opens  # noqa: E402
from tests.fixtures.registered_cohort import (  # noqa: E402
    build_registered_cohort,
    copy_registered_cohort,
)

LEDGER = "evaluations.json"


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return build_registered_cohort(tmp_path_factory.mktemp("registered_base"))


@pytest.fixture
def cohort(base, tmp_path, monkeypatch):
    """A copy of the built cohort, its registry the deployment's
    (`SHEPHERD_REGISTRY`): a ``--cohort`` run measures, and a report is
    registered, only there (BACKLOG 37)."""
    copied = copy_registered_cohort(base, tmp_path / "root")
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(copied.registry))
    return copied


@pytest.fixture
def cli():
    import scripts.measure_scorer as module

    return module


def _argv(cohort, *extra, label=None):
    return ["--checkpoint", str(cohort.checkpoint), "--data-dir", str(cohort.data_dir),
            "--cohort", label or cohort.version.manifest["label"],
            "--registry", str(cohort.registry), "--device", "cpu", *extra]


def _reports(cohort):
    return sorted((cohort.version.directory / "reports").glob("*/report.json"))


def _report(cohort):
    (path,) = _reports(cohort)
    return path, json.loads(path.read_text())


def _nothing_written(cohort):
    return (not (cohort.version.directory / "reports").exists()
            and not (cohort.checkpoint.parent / LEDGER).exists())


# ---------------------------------------------------------------------------
# The order, and nothing written before every check passes
# ---------------------------------------------------------------------------
def test_the_reads_and_checks_run_in_the_specified_order_and_writing_comes_last(
    cli, cohort, monkeypatch
):
    """M4 §6: evidence directories (listed, nothing read), `cohort.json`, the
    workspace manifest and tensors, the version's samples, R1, the `kg.json`
    identification, R5, the cohort's binding, the checkpoint, R3, R10 -- and
    only then the run directory, the ranks and the report, in that order."""
    import src.evaluation.cohort_registry as cohort_registry
    import src.evaluation.r10 as r10
    import src.kg.artifacts as artifacts
    import src.kg.storage.file_storage as file_storage
    import src.utils.checkpoint_io as checkpoint_io
    import src.utils.file_modes as file_modes
    import src.utils.fingerprint as fingerprint

    calls = []

    def spy(module, name, label=None):
        real = getattr(module, name)

        def recorded(*args, **kwargs):
            calls.append(label(*args) if label else name)
            return real(*args, **kwargs)

        monkeypatch.setattr(module, name, recorded)

    spy(r10, "evidence_candidates")
    spy(cohort_registry, "resolve_version")
    spy(artifacts, "read_split_manifest")
    spy(file_storage, "read_graph_artifacts")
    spy(file_storage, "read_samples_file", lambda path, **k: f"read {Path(path).name}")
    spy(r10, "read_samples_file", lambda path, **k: f"R10 reads {Path(path).name}")
    spy(artifacts, "verify_graph_reads")
    spy(artifacts, "check_unparsed_kg_json")
    spy(artifacts, "verify_dataset_record")
    spy(cohort_registry, "verify_cohort_binding")
    spy(checkpoint_io, "read_checkpoint")
    spy(fingerprint, "verify_training_record")
    spy(r10, "compute_r10")
    spy(cohort_registry, "create_report_dir")
    spy(file_modes, "publish_bytes", lambda target, data: f"publish {Path(target).name}")

    assert cli.main(_argv(cohort)) == 0

    assert calls == [
        "evidence_candidates", "resolve_version", "read_split_manifest",
        "read_graph_artifacts", "read samples.json", "verify_graph_reads",
        "check_unparsed_kg_json", "verify_dataset_record", "verify_cohort_binding",
        "read_checkpoint", "verify_training_record", "compute_r10",
        "R10 reads train_samples.json", "R10 reads val_samples.json",
        "create_report_dir", "publish ranks.json", "publish report.json",
    ]


def _without_source_format(cohort):
    path = cohort.version.directory / "cohort.json"
    manifest = json.loads(path.read_text())
    del manifest["source"]["format"]
    path.write_text(json.dumps(manifest))
    return "records no source format"


def _samples_removed(cohort):
    cohort.version.samples_path.unlink()
    return "is missing, though"


def _samples_not_the_count(cohort):
    """Finding 15: one sample dropped and `cohort.json` re-pointed at the new
    bytes by digest, its counts left as they were -- a manifest consistent with
    itself whose denominators describe samples no run will read."""
    samples = json.loads(cohort.version.samples_path.read_text())[:-1]
    data = json.dumps(samples).encode("utf-8")
    cohort.version.samples_path.write_bytes(data)
    path = cohort.version.directory / "cohort.json"
    manifest = json.loads(path.read_text())
    manifest["samples"]["sha256"] = hashlib.sha256(data).hexdigest()
    path.write_text(json.dumps(manifest))
    n = manifest["samples"]["n"]
    return (f"holds {len(samples)} samples, and .* records samples.n {n} and "
            f"counts.ranked_eligible {manifest['counts']['ranked_eligible']}")


def _pre_contract_checkpoint(cohort):
    contents = torch.load(cohort.checkpoint, map_location="cpu", weights_only=True)
    del contents["training_input_digests"]
    torch.save(contents, cohort.checkpoint)
    return "records no training_input_digests"


def _unreadable_candidate(cohort, monkeypatch):
    import src.evaluation.r10 as r10

    real = r10.read_samples_file

    def refuse(path, **kwargs):
        if Path(path).name == "train_samples.json":
            raise PermissionError(13, "Permission denied", str(path))
        return real(path, **kwargs)

    monkeypatch.setattr(r10, "read_samples_file", refuse)
    return "R10 evidence candidate could not be read"


@pytest.mark.parametrize("case", ["no_source_format", "samples_removed",
                                  "samples_not_the_count", "pre_contract",
                                  "unreadable_candidate"])
def test_each_refusal_writes_nothing(cli, cohort, monkeypatch, case):
    """Refusals of the registered path's own and the shared checks it runs late:
    one `ValueError` each, naming what is wrong, and no run directory and no
    ledger, because nothing is written before every check passes -- R10, the last
    check, included."""
    import src.models.gnn.shepherd_gnn as shepherd_gnn

    if case == "unreadable_candidate":
        message = _unreadable_candidate(cohort, monkeypatch)
    else:
        message = {"no_source_format": _without_source_format,
                   "samples_removed": _samples_removed,
                   "samples_not_the_count": _samples_not_the_count,
                   "pre_contract": _pre_contract_checkpoint}[case](cohort)
    monkeypatch.setattr(shepherd_gnn, "build_shepherd_model",
                        lambda *a, **k: pytest.fail("a model was built after a refusal"))

    with pytest.raises(ValueError, match=message) as refused:
        cli.main(_argv(cohort))

    if case != "pre_contract":  # R3's message is the shared comparator's
        assert "Nothing was written" in str(refused.value)
    assert _nothing_written(cohort)


def test_the_internal_entry_refuses_a_request_that_is_neither_or_both_kinds(cli, cohort):
    """`read_measurement_inputs` is called by `main` and the Mode C tests; a call
    naming no cohort, both, or registry arguments for a split is a caller's
    error, refused before anything is read."""
    device = torch.device("cpu")
    with count_opens() as opens:
        with pytest.raises(ValueError, match="exactly one"):
            cli.read_measurement_inputs(cohort.checkpoint, cohort.data_dir, device)
        with pytest.raises(ValueError, match="exactly one"):
            cli.read_measurement_inputs(cohort.checkpoint, cohort.data_dir, device,
                                        split="val", cohort="demo-test-1")
        with pytest.raises(ValueError, match="registered cohort only"):
            cli.read_measurement_inputs(cohort.checkpoint, cohort.data_dir, device,
                                        split="val", registry=cohort.registry)
        with pytest.raises(ValueError, match="none was given"):
            cli.read_measurement_inputs(cohort.checkpoint, cohort.data_dir, device,
                                        cohort="demo-test-1")
    assert not [path for path, n in opens.counts.items() if n], "something was opened"


# ---------------------------------------------------------------------------
# What the manifest records
# ---------------------------------------------------------------------------
def test_the_manifest_records_the_version_as_an_aggregate_block(cli, cohort):
    """Exactly `COHORT_VERSION_FIELDS`, each from the one read of `cohort.json`;
    its manifest digest is the digest of those bytes. Exclusions, unmappable
    terms and case ids stay in the version, so no case id is in the report."""
    from src.evaluation.sidecar import COHORT_VERSION_FIELDS

    assert cli.main(_argv(cohort)) == 0

    path, report = _report(cohort)
    version = cohort.version.manifest
    block = report["manifest"]["cohort_version"]
    assert set(block) == set(COHORT_VERSION_FIELDS)
    assert block == {
        "label": version["label"], "role": version["role"],
        "manifest_sha256": _sha(cohort.version.directory / "cohort.json"),
        "source_sha256": version["source"]["sha256"],
        "source_format": version["source"]["format"],
        "samples_sha256": version["samples"]["sha256"],
        "graph": version["graph"],
        "dataset_record_sha256": version["dataset_record"]["sha256"],
        "mapping": version["mapping"], "counts": version["counts"],
    }
    assert report["manifest"]["split"] == version["role"]
    assert report["manifest"]["cohort_kind"] == "supplied"
    assert "case-" not in path.read_text(), "a case id reached the report"


def test_the_r10_block_is_compute_r10_over_the_checkpoints_record(cli, cohort):
    """The manifest's R10 is the pure function of the checkpoint's own record,
    the version's samples and the candidates; nothing else feeds it."""
    from src.evaluation.r10 import compute_r10, evidence_candidates
    from src.kg.storage.file_storage import read_samples_file

    assert cli.main(_argv(cohort)) == 0

    recorded = torch.load(cohort.checkpoint, weights_only=True)["training_input_digests"]
    samples = read_samples_file(cohort.version.samples_path).samples
    expected = compute_r10(recorded, samples, evidence_candidates(cohort.data_dir, []))
    assert _report(cohort)[1]["manifest"]["r10"] == expected


@pytest.mark.parametrize("record, training, validation, val_opens", [
    ("pipeline", "found", "found", 1),
    ("no_validation", "found", "not_recorded", 0),
    ("graph_only", "not_recorded", "not_recorded", 0),
])
def test_r10_states_follow_what_the_checkpoint_records(
    cli, tmp_path, monkeypatch, record, training, validation, val_opens
):
    """A pipeline record finds both scopes in the workspace; with no validation
    recorded, validation is `no_validation_inputs_recorded` and `val` is never
    read; a graph-only record is not the pipeline's, so both scopes are
    unverifiable and no candidate is read. Never a count without evidence."""
    cohort = build_registered_cohort(tmp_path / "root", training_record=record)
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(cohort.registry))

    with count_opens() as opens:
        assert cli.main(_argv(cohort)) == 0

    r10 = _report(cohort)[1]["manifest"]["r10"]
    assert r10["training"]["evidence"]["state"] == training
    assert r10["validation"]["evidence"]["state"] == validation
    assert opens.opens(cohort.data_dir / "val_samples.json") == val_opens
    if record == "no_validation":
        assert r10["validation"]["disease_overlap"]["state"] == "no_validation_inputs_recorded"
        assert r10["validation"]["disease_overlap"]["k"] is None
    if record == "graph_only":
        assert opens.opens(cohort.data_dir / "train_samples.json") == 0
        for scope in ("training", "validation"):
            assert r10[scope]["phenotype_set_overlap"] == {
                "state": "unverifiable", "k": None, "n": r10[scope]["phenotype_set_overlap"]["n"]}


def test_build_manifest_takes_the_kind_from_the_version_and_requires_r10(cli, cohort):
    """With a version block the manifest is a registered cohort's whatever
    `args.split` says; without R10 beside a version it is refused, so no report
    of a registered version can lack its overlap."""
    import argparse

    from src.evaluation.measurement import AutocastRegime

    args = argparse.Namespace(split=None, batch_size=4, checkpoint=cohort.checkpoint,
                              data_dir=cohort.data_dir)
    block = cli.cohort_version_block(cohort.version)
    common = {"digests": {"checkpoint": "a" * 64}, "dataset_record": {"origin": "synthetic"},
              "regime": AutocastRegime(enabled=False, dtype=None)}
    graph = {"num_nodes_dict": {"disease": 1}}

    manifest = cli.build_manifest(args, graph, 1, torch.device("cpu"), cohort_version=block,
                                  r10={"definition_version": 1}, **common)
    assert (manifest.split, manifest.cohort_kind) == (block["role"], "supplied")
    assert manifest.relations["cohort_graph"] == "verified"
    assert manifest.relations["R2"] == "not_applicable"
    with pytest.raises(ValueError, match="R10 is recorded for a registered cohort version"):
        cli.build_manifest(args, graph, 1, torch.device("cpu"), cohort_version=block, **common)


# ---------------------------------------------------------------------------
# --record
# ---------------------------------------------------------------------------
def test_record_registers_into_the_ledger_beside_the_checkpoint_without_rehashing_it(
    cli, cohort, monkeypatch, capsys
):
    """`register_report` with the measurement's own checkpoint read's directory:
    the report is read once by the registration, the checkpoint stays read once
    (by the measurement), and the directory's checkpoints are not hashed."""
    import src.evaluation.registration as registration

    for name in ("find_checkpoint", "checkpoint_matches"):
        monkeypatch.setattr(registration, name,
                            lambda *a, **k: pytest.fail("the checkpoints were hashed again"))

    with count_opens() as opens:
        assert cli.main(_argv(cohort, "--record")) == 0

    path, _ = _report(cohort)
    # Written through a staging file renamed into place, so the one open of the
    # report's own path is the registration's read.
    assert opens.opens(path) == 1
    assert opens.opens(cohort.checkpoint) == 1
    ledger = json.loads((cohort.checkpoint.parent / LEDGER).read_text())
    (record,) = ledger["records"]
    assert record["source_artifact_digest"] == _sha(path)
    assert record["checkpoint_digest"] == cohort.checkpoint_sha256
    assert f"Registered as record 1 in {cohort.checkpoint.parent / LEDGER}" in (
        capsys.readouterr().out)


def test_record_files_the_ledger_beside_the_checkpoint_the_link_names(cli, cohort, tmp_path):
    """The ledger is beside the checkpoint's resolved path, as the API reads it
    beside the served checkpoint; a link elsewhere does not move it."""
    elsewhere = tmp_path / "links"
    elsewhere.mkdir()
    link = elsewhere / "best.pt"
    link.symlink_to(cohort.checkpoint)
    argv = _argv(cohort, "--record")
    argv[argv.index(str(cohort.checkpoint))] = str(link)

    assert cli.main(argv) == 0

    assert (cohort.checkpoint.parent / LEDGER).is_file()
    assert not (elsewhere / LEDGER).exists()


def test_record_files_beside_the_bytes_read_though_the_link_is_retargeted_during_the_run(
    cli, cohort, tmp_path, monkeypatch
):
    """Finding 5: the checkpoint is read at its resolved path, so the ledger's
    directory is fixed when the weights are read. A link retargeted at another
    run's checkpoint once this run has read through it (here, after the report
    is kept) does not file this run's record beside those other weights."""
    run_b = tmp_path / "run-b" / "checkpoints"
    run_b.mkdir(parents=True)
    (run_b / "best.pt").write_bytes(b"run B's weights")
    link = tmp_path / "models" / "current.pt"
    link.parent.mkdir()
    link.symlink_to(cohort.checkpoint)
    publish = cli._publish_to_store

    def publish_then_retarget(result, inputs):
        published = publish(result, inputs)
        link.unlink()
        link.symlink_to(run_b / "best.pt")
        return published

    monkeypatch.setattr(cli, "_publish_to_store", publish_then_retarget)
    argv = _argv(cohort, "--record")
    argv[argv.index(str(cohort.checkpoint))] = str(link)

    assert cli.main(argv) == 0

    (record,) = json.loads((cohort.checkpoint.parent / LEDGER).read_text())["records"]
    assert record["checkpoint_digest"] == cohort.checkpoint_sha256
    assert not (run_b / LEDGER).exists()
    assert _report(cohort)[1]["manifest"]["checkpoint_path"] == str(link), (
        "the report keeps the path the operator gave")


def test_the_same_inputs_measured_again_are_already_registered(cli, cohort, capsys):
    """Two runs over the same inputs write byte-identical reports in two run
    directories; the second registration is a no-op and leaves the ledger's
    bytes as they were."""
    assert cli.main(_argv(cohort, "--record")) == 0
    ledger = cohort.checkpoint.parent / LEDGER
    before = ledger.read_bytes()

    assert cli.main(_argv(cohort, "--record")) == 0

    first, second = _reports(cohort)
    assert first.read_bytes() == second.read_bytes()
    assert ledger.read_bytes() == before
    assert "Already held as record 1; nothing written" in capsys.readouterr().out


@pytest.mark.parametrize("case", ["read_only", "schema_2_ledger"])
def test_a_refused_registration_keeps_the_report_and_exits_3(
    cli, cohort, monkeypatch, capsys, case
):
    """The measurement succeeded, so the report stays; the message names it and
    the command that registers it later; the ledger is as it was."""
    import src.evaluation.sidecar as sidecar

    ledger = cohort.checkpoint.parent / LEDGER
    if case == "read_only":
        monkeypatch.setattr(sidecar, "create_staging_file", lambda *a, **k: (_ for _ in ()).throw(
            PermissionError(13, "Permission denied", str(cohort.checkpoint.parent))))
        fragment = "Permission denied"
    else:
        ledger.write_text(json.dumps({"schema_version": 2, "records": []}))
        fragment = "evaluations.v2.json"
    before = ledger.read_bytes() if ledger.exists() else None

    assert cli.main(_argv(cohort, "--record")) == cli.EXIT_NOT_REGISTERED == 3

    path, _ = _report(cohort)
    err = capsys.readouterr().err
    assert fragment in err
    assert f"report is kept at {path}" in err
    assert (f'python scripts/record_evaluation.py --report "{path}" '
            f'--checkpoint "{cohort.checkpoint.resolve()}"') in err
    assert (ledger.read_bytes() if ledger.exists() else None) == before


def test_without_record_nothing_is_registered_and_the_command_is_printed(cli, cohort, capsys):
    """Not registering keeps the report and writes no ledger; the console says
    how to register it later."""
    assert cli.main(_argv(cohort)) == 0

    path, _ = _report(cohort)
    assert not (cohort.checkpoint.parent / LEDGER).exists()
    out = capsys.readouterr().out
    assert "Not registered (no --record)" in out
    assert (f'python scripts/record_evaluation.py --report "{path}" '
            f'--checkpoint "{cohort.checkpoint.resolve()}"') in out


def test_the_register_later_command_quotes_each_path_for_every_shell():
    """Finding 1: each path double-quoted, the form a POSIX shell, cmd and
    PowerShell all read as one argument for an ordinary path; a POSIX shell
    splits it into exactly the arguments `record_evaluation` takes."""
    import shlex

    import scripts.measure_scorer as cli

    report = Path("/srv/My Registry/versions/l/reports/r/report.json")
    checkpoint = Path("/home/op/My Models/best.ckpt")

    line = cli.register_later_command(report, checkpoint)

    assert line == ('python scripts/record_evaluation.py --report '
                    '"/srv/My Registry/versions/l/reports/r/report.json" '
                    '--checkpoint "/home/op/My Models/best.ckpt"')
    assert shlex.split(line)[2:] == ["--report", str(report), "--checkpoint", str(checkpoint)]


@pytest.mark.parametrize("record", ["pipeline", "graph_only"])
def test_the_console_gives_each_r10_label_in_the_caveats_words_and_the_definitions_once(
    cli, tmp_path, monkeypatch, capsys, record
):
    """Finding 9: each label's state is rendered through `R10_STATE_TEXT`, an
    unverifiable one through its scope's `R10_UNVERIFIABLE_REASON`, with no
    number where there is no count; the four definitions are printed once, so
    the console says that phenotype-set overlap ignores order and repetition and
    that "none" is not evidence about data outside the recorded inputs. The
    report keeps the raw states and no prose."""
    from src.evaluation.caveats import (
        R10_LABEL_TEXT,
        R10_STATE_TEXT,
        R10_UNVERIFIABLE_REASON,
    )

    cohort = build_registered_cohort(tmp_path / "root", training_record=record)
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(cohort.registry))

    assert cli.main(_argv(cohort)) == 0

    out = capsys.readouterr().out
    path, report = _report(cohort)
    r10 = report["manifest"]["r10"]
    lines = out.splitlines()
    for scope, labels in R10_LABEL_TEXT.items():
        evidence = r10[scope]["evidence"]["state"]
        for label, text in labels.items():
            value = r10[scope][label]
            words = (R10_UNVERIFIABLE_REASON[evidence] if value["state"] == "unverifiable"
                     else R10_STATE_TEXT[value["state"]])
            (line,) = [line for line in lines if line.strip().startswith(text["name"] + " ")]
            shown = line.strip()[len(text["name"]):].strip()
            assert shown.startswith(words), (line, words)
            if value["k"] is None:
                assert shown == words and not any(c.isdigit() for c in shown)
            else:
                assert f"{value['k']} of {value['n']}" in shown
            assert out.count(f"{text['name']}: {text['definition']}.") == 1
    assert "phenotype-set overlap ignores order and repetition, so the scoring inputs " \
           "may differ" in out
    assert out.count("not evidence about data outside them") >= 1
    kept = path.read_text()
    for prose in ("scoring inputs may differ", "not evidence about data", "there is no count",
                  "distinct test diseases"):
        assert prose not in kept, prose


def test_no_patient_level_data_reaches_the_ledger(cli, cohort):
    """The ledger can sit inside the repository tree beside a checkpoint; its
    record holds aggregates only: no case id, no phenotype list, no rank."""
    assert cli.main(_argv(cohort, "--record")) == 0

    text = (cohort.checkpoint.parent / LEDGER).read_text()
    assert "case-" not in text
    for forbidden in ("phenotype_ids", "patient_id", "sample_id", "exclusions",
                      "unmappable_terms"):
        assert forbidden not in text, forbidden


# ---------------------------------------------------------------------------
# M5-1: expected digests, the outcome file, the usage history, --deterministic
# ---------------------------------------------------------------------------
def _usage(cohort):
    from src.evaluation.report_store import read_usage_history

    entries, problems = read_usage_history(cohort.version)
    assert problems == []
    return entries


def test_the_expected_digests_let_the_chosen_run_through(cli, cohort):
    assert cli.main(_argv(cohort, "--expect-checkpoint-sha256", cohort.checkpoint_sha256,
                          "--expect-version-sha256", cohort.version.identity.sha256)) == 0
    assert len(_reports(cohort)) == 1


@pytest.mark.parametrize("which", ["version", "checkpoint"])
def test_a_read_that_is_not_the_chosen_one_is_refused_before_anything_is_measured(
    cli, cohort, monkeypatch, which
):
    """The version's `cohort.json` is compared right after it is read, before
    the workspace is; the checkpoint right after its read, before R3. Nothing
    is built or written, and the message names both digests."""
    import src.kg.artifacts as artifacts
    import src.models.gnn.shepherd_gnn as shepherd_gnn
    import src.utils.fingerprint as fingerprint

    other = "f" * 64
    monkeypatch.setattr(shepherd_gnn, "build_shepherd_model",
                        lambda *a, **k: pytest.fail("a model was built after a refusal"))
    if which == "version":
        monkeypatch.setattr(artifacts, "read_split_manifest",
                            lambda *a, **k: pytest.fail("the workspace was read"))
        flag, read = "--expect-version-sha256", cohort.version.identity.sha256
    else:
        monkeypatch.setattr(fingerprint, "verify_training_record",
                            lambda *a, **k: pytest.fail("R3 ran after the refusal"))
        flag, read = "--expect-checkpoint-sha256", cohort.checkpoint_sha256

    with pytest.raises(cli.ExpectedDigestMismatch) as refused:
        cli.main(_argv(cohort, flag, other, "--record"))

    message = str(refused.value)
    assert f"holds bytes {read[:12]}..., not the {other[:12]}..." in message
    assert "Nothing was written" in message
    assert _nothing_written(cohort) and _usage(cohort) == []


def _outcome(path):
    return json.loads(Path(path).read_text())


def test_the_outcome_file_says_what_the_run_did_and_report_state_agrees(
    cli, cohort, tmp_path
):
    from src.evaluation.registration import report_state

    out = tmp_path / "jobs" / "outcome.json"
    out.parent.mkdir()

    assert cli.main(_argv(cohort, "--outcome", str(out))) == 0

    path, _ = _report(cohort)
    outcome = _outcome(out)
    assert outcome["complete"] is True and outcome["schema_version"] == 1
    assert outcome["report"] == str(path) and outcome["run_id"] == path.parent.name
    assert outcome["report_sha256"] == _sha(path)
    assert outcome["checkpoint"] == str(cohort.checkpoint.resolve())
    assert outcome["checkpoint_sha256"] == cohort.checkpoint_sha256
    assert outcome["cohort"] == {"label": cohort.version.manifest["label"],
                                 "role": cohort.version.manifest["role"],
                                 "manifest_sha256": cohort.version.identity.sha256}
    assert outcome["deterministic"] is False
    assert outcome["registration"] == {
        "requested": False, "registered": False, "outcome": "not_requested", "sequence": None,
        "ledger": None,
        "why": "--record was not given; the report is kept for a later registration"}
    state = report_state(outcome["report"], checkpoint=outcome["checkpoint"],
                         expected_sha256=outcome["report_sha256"])
    assert (state.kept, state.registered) == ("kept", "not_registered")


def test_the_outcome_file_records_a_registration_and_a_refused_one(cli, cohort, tmp_path):
    from src.evaluation.registration import report_state

    out = tmp_path / "outcome.json"
    assert cli.main(_argv(cohort, "--record", "--outcome", str(out))) == 0
    registration = _outcome(out)["registration"]
    assert registration == {
        "requested": True, "registered": True, "outcome": "registered", "sequence": 1,
        "ledger": str(cohort.checkpoint.parent.resolve() / LEDGER),
        "why": "Registered as record 1"}

    ledger = cohort.checkpoint.parent / LEDGER
    ledger.write_text(json.dumps({"schema_version": 2, "records": []}))
    out = tmp_path / "outcome-2.json"  # a new file for each run
    assert cli.main(_argv(cohort, "--record", "--outcome", str(out))) == 3
    outcome = _outcome(out)
    assert outcome["complete"] is True
    assert outcome["registration"]["registered"] is False
    assert outcome["registration"]["outcome"] == "refused"
    assert "evaluations.v2.json" in outcome["registration"]["why"]
    state = report_state(outcome["report"], checkpoint=outcome["checkpoint"],
                         expected_sha256=outcome["report_sha256"])
    assert (state.kept, state.registered) == ("kept", "cannot_be_checked")


def test_no_outcome_file_is_written_when_the_run_is_refused_before_its_report(
    cli, cohort, tmp_path
):
    out = tmp_path / "outcome.json"
    _without_source_format(cohort)

    with pytest.raises(ValueError):
        cli.main(_argv(cohort, "--outcome", str(out)))
    assert not out.exists()


@pytest.mark.parametrize("where", ["missing_directory", "a_directory", "an_earlier_runs_file",
                                   "a_dangling_link"])
def test_an_outcome_file_that_cannot_be_written_is_refused_before_anything_is_read(
    cli, cohort, tmp_path, monkeypatch, where
):
    """Its directory must exist and it must not: an existing file -- an
    earlier run's outcome, or a file named by mistake -- is neither read as
    this run's nor replaced (the reviewers' P3, 2026-10-20)."""
    import src.evaluation.cohort_registry as cohort_registry

    monkeypatch.setattr(cohort_registry, "resolve_version",
                        lambda *a, **k: pytest.fail("something was read"))
    out = {"missing_directory": tmp_path / "absent" / "outcome.json",
           "a_directory": tmp_path}.get(where, tmp_path / "outcome.json")
    earlier = json.dumps({"complete": True, "registration": {"registered": True}})
    if where == "an_earlier_runs_file":
        out.write_text(earlier)
    elif where == "a_dangling_link":
        out.symlink_to(tmp_path / "nowhere.json")

    with pytest.raises(ValueError, match="--outcome .*Nothing was read or written") as refused:
        cli.main(_argv(cohort, "--outcome", str(out), "--record"))
    assert _nothing_written(cohort)
    if where == "an_earlier_runs_file":
        assert "already exists" in str(refused.value)
        assert out.read_text() == earlier
    elif where == "a_dangling_link":
        assert out.is_symlink() and not (tmp_path / "nowhere.json").exists()


def test_a_run_refused_before_its_report_never_leaves_an_earlier_outcome_as_its_own(
    cli, cohort, tmp_path
):
    """The reviewers' scenario: an outcome path reused by a run that is then
    refused (here by an expected digest) is refused first, so no file there
    reads as this run's "complete" outcome."""
    out = tmp_path / "outcome.json"
    assert cli.main(_argv(cohort, "--record", "--outcome", str(out))) == 0
    first = out.read_bytes()

    with pytest.raises(ValueError, match="already exists"):
        cli.main(_argv(cohort, "--record", "--outcome", str(out),
                       "--expect-checkpoint-sha256", "f" * 64))
    assert out.read_bytes() == first and len(_reports(cohort)) == 1


def test_a_run_that_stops_after_publishing_leaves_an_incomplete_outcome(
    cli, cohort, tmp_path, monkeypatch
):
    """The outcome is written once the report is published: a run that fails
    after that (here its console) leaves `complete` false naming the report,
    and says nothing it did not do."""
    out = tmp_path / "outcome.json"
    monkeypatch.setattr(cli, "_print_registered_summary",
                        lambda *a, **k: (_ for _ in ()).throw(OSError(5, "I/O error")))

    with pytest.raises(cli.FailedAfterPublishing):
        cli.main(_argv(cohort, "--record", "--outcome", str(out)))

    outcome = _outcome(out)
    path, _ = _report(cohort)
    assert outcome["complete"] is False and outcome["report"] == str(path)
    assert outcome["registration"]["registered"] is None
    assert "had not been attempted" in outcome["registration"]["why"]
    assert not (cohort.checkpoint.parent / LEDGER).exists()


def test_each_run_and_each_registration_is_in_the_versions_usage_history(cli, cohort):
    """D4: the measurement appends the run (its result, aggregates only) right
    after its report is published, and the registration function the
    registration, so command-line runs count."""
    assert cli.main(_argv(cohort)) == 0
    assert cli.main(_argv(cohort, "--record")) == 0

    first, second = _reports(cohort)
    entries = _usage(cohort)
    assert [e["kind"] for e in entries] == ["test_run", "test_run", "registered"]
    runs = {e["run_id"]: e for e in entries if e["kind"] == "test_run"}
    for path in (first, second):
        entry, report = runs[path.parent.name], json.loads(path.read_text())
        assert entry["report_sha256"] == _sha(path)
        assert entry["checkpoint_sha256"] == cohort.checkpoint_sha256
        assert entry["metrics"] == report["authoritative_metrics"]
        assert entry["n_ranked"] == report["n_ranked"]
        assert (entry["deterministic"], entry["device"]) == (False, "cpu")
    assert sorted(e["record_requested"] for e in runs.values()) == [False, True]
    assert entries[2]["outcome"] in ("registered", "already_registered", "corroborated")
    text = (cohort.version.directory / "reports" / "usage.jsonl").read_text()
    assert "case-" not in text and "phenotype_ids" not in text


def test_a_usage_history_that_cannot_be_appended_stops_the_run_after_its_report(
    cli, cohort, monkeypatch
):
    """The report is kept; the run says it is not in the usage history, and it
    does not go on to register."""
    import src.evaluation.report_store as report_store

    def fail(*args, **kwargs):
        raise PermissionError(13, "Permission denied", "usage.jsonl")

    monkeypatch.setattr(report_store, "record_test_run", fail)
    out = cohort.checkpoint.parent.parent / "outcome.json"

    with pytest.raises(cli.FailedAfterPublishing) as failed:
        cli.main(_argv(cohort, "--record", "--outcome", str(out)))

    assert "usage history could not be appended, so this run is not in it" in str(failed.value)
    assert "The report was written" in str(failed.value)
    assert len(_reports(cohort)) == 1
    assert not (cohort.checkpoint.parent / LEDGER).exists()
    # The outcome file is written all the same: the run published, then stopped.
    outcome = _outcome(out)
    assert outcome["complete"] is False and outcome["registration"]["registered"] is None


def test_a_failed_outcome_write_does_not_keep_a_published_run_out_of_the_history(
    cli, cohort, tmp_path, monkeypatch
):
    """The reviewers' P3 (2026-10-20): the usage history is the record and the
    outcome file the job runner's view, so the history is appended first and
    whatever the outcome write does; the failure then stops the run after
    publishing, saying the run is in the history."""
    def fail(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(cli, "_write_outcome", fail)

    with pytest.raises(cli.FailedAfterPublishing) as failed:
        cli.main(_argv(cohort, "--record", "--outcome", str(tmp_path / "outcome.json")))

    message = str(failed.value)
    assert "--outcome file" in message and "No space left on device" in message
    assert "this run is in its version's usage history" in message
    assert "The report was written" in message
    (path,) = _reports(cohort)
    assert [(e["kind"], e["run_id"]) for e in _usage(cohort)] == [
        ("test_run", path.parent.name)]
    assert not (cohort.checkpoint.parent / LEDGER).exists(), "it stops before registering"


def test_when_both_the_history_and_the_outcome_fail_the_message_says_both(
    cli, cohort, tmp_path, monkeypatch
):
    import src.evaluation.report_store as report_store

    monkeypatch.setattr(report_store, "record_test_run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError(5, "I/O error")))
    monkeypatch.setattr(cli, "_write_outcome",
                        lambda *a, **k: (_ for _ in ()).throw(OSError(28, "No space left")))

    with pytest.raises(cli.FailedAfterPublishing) as failed:
        cli.main(_argv(cohort, "--outcome", str(tmp_path / "outcome.json")))

    message = str(failed.value)
    assert "--outcome file" in message and "No space left" in message
    assert "usage history could not be appended, so this run is not in it" in message
    assert "this run is in its version's usage history" not in message


def test_a_disagreeing_run_is_kept_and_not_registered_without_a_later_command(
    cli, cohort, capsys
):
    """D5 through --record: a second run under a key whose stored reports
    disagree exits 3, keeps its report, and offers no register-later command,
    since none would be accepted while they disagree."""
    import secrets

    assert cli.main(_argv(cohort, "--record")) == 0
    (first,) = _reports(cohort)
    edited = json.loads(first.read_text())
    edited["authoritative_metrics"]["hits@1"] = 0.5 + edited["authoritative_metrics"]["hits@1"] / 2
    stamp, checkpoint, _ = first.parent.name.split("-")
    other = first.parents[1] / f"{stamp}-{checkpoint}-{secrets.token_hex(4)}"
    other.mkdir()
    (other / "report.json").write_text(json.dumps(edited))
    capsys.readouterr()

    assert cli.main(_argv(cohort, "--record")) == 3

    err = capsys.readouterr().err
    assert "runs disagree — investigate" in err and "hits@1" in err
    assert "where the investigation can read it" in err
    assert "Register it later with" not in err
    assert len(_reports(cohort)) == 3


def _disagreeing_report(cohort):
    """A second report under the key of the store's one report, its hits@1
    another number: the store now holds runs that disagree."""
    import secrets

    (first,) = _reports(cohort)
    edited = json.loads(first.read_text())
    edited["authoritative_metrics"]["hits@1"] = 0.5 + edited["authoritative_metrics"]["hits@1"] / 2
    stamp, checkpoint, _ = first.parent.name.split("-")
    other = first.parents[1] / f"{stamp}-{checkpoint}-{secrets.token_hex(4)}"
    other.mkdir()
    (other / "report.json").write_text(json.dumps(edited))
    return other / "report.json"


def test_without_record_a_disagreeing_run_offers_no_register_later_command(
    cli, cohort, capsys, tmp_path
):
    """The reviewer's P3 (2026-10-20): without --record the run checks the
    store as a registration would, so it never prints a command D5 would
    refuse: it says the runs disagree, and its outcome says so too."""
    assert cli.main(_argv(cohort)) == 0
    _disagreeing_report(cohort)
    capsys.readouterr()
    out = tmp_path / "outcome.json"

    assert cli.main(_argv(cohort, "--outcome", str(out))) == 0

    printed = capsys.readouterr().out
    assert "Not registered (no --record), and runs disagree — investigate" in printed
    assert "hits@1" in printed and "No register-later command is offered" in printed
    assert "Register it with" not in printed and "record_evaluation.py" not in printed
    registration = _outcome(out)["registration"]
    assert (registration["registered"], registration["outcome"]) == (False, "not_requested")
    assert "runs disagree — investigate" in registration["why"]
    assert not (cohort.checkpoint.parent / LEDGER).exists()


def test_without_record_a_key_that_cannot_be_checked_says_registering_waits_for_it(
    cli, cohort, capsys, tmp_path
):
    from tests.fixtures.report_store import store_bytes

    damaged = store_bytes(cohort.registry, b"{broken", version=cohort.version)
    out = tmp_path / "outcome.json"

    assert cli.main(_argv(cohort, "--outcome", str(out))) == 0

    printed = capsys.readouterr().out
    assert "cannot be checked" in printed and str(damaged) in printed
    assert "A later registration is refused until that can be checked" in printed
    assert "record_evaluation.py --report" in printed
    assert "cannot be checked" in _outcome(out)["registration"]["why"]


def test_a_runs_report_replaced_by_a_link_out_of_the_store_is_not_offered_later(
    cli, cohort, monkeypatch, capsys, tmp_path
):
    """The reviewer's P3 (2026-10-20): the run's `report.json` replaced, after
    its publication, by a link to a copy outside any store is not the run's
    report, so --record refuses it as such and offers no later registration;
    `report_state` agrees it is not kept."""
    from src.evaluation.registration import report_state

    real = cli._print_registered_summary
    outside = tmp_path / "outside.json"

    def summary_then_replace(inputs, result, run_id, report_path):
        real(inputs, result, run_id, report_path)
        outside.write_bytes(Path(report_path).read_bytes())
        Path(report_path).unlink()
        Path(report_path).symlink_to(outside)

    monkeypatch.setattr(cli, "_print_registered_summary", summary_then_replace)
    out = tmp_path / "outcome.json"

    assert cli.main(_argv(cohort, "--record", "--outcome", str(out))) == 3

    err = capsys.readouterr().err
    assert "is no longer a report the store holds" in err and "re-measure" in err
    assert "nothing at that path is offered for a later registration" in err
    assert "Register it later with" not in err and "kept at" not in err
    outcome = _outcome(out)
    state = report_state(outcome["report"], checkpoint=outcome["checkpoint"],
                         expected_sha256=outcome["report_sha256"])
    assert state.kept == "not_kept"
    assert not (cohort.checkpoint.parent / LEDGER).exists()


def test_a_report_in_the_store_too_deep_to_compare_is_a_refusal_not_a_traceback(
    cli, cohort, capsys, tmp_path
):
    """The reviewer's P3 (2026-10-20): a hand-edited report in the store that
    nests too deeply to be built leaves its key "cannot be checked"; --record
    exits 3 saying so, and the outcome file is complete."""
    assert cli.main(_argv(cohort)) == 0
    deep_path = _disagreeing_report(cohort)
    deep = json.loads(deep_path.read_text())
    node = deep["manifest"]["r10"]
    for _ in range(600):
        node["x"] = {}
        node = node["x"]
    deep_path.write_text(json.dumps(deep))
    capsys.readouterr()
    out = tmp_path / "outcome.json"

    assert cli.main(_argv(cohort, "--record", "--outcome", str(out))) == 3

    err = capsys.readouterr().err
    assert "cannot be checked" in err and "nests too deeply" in err and str(deep_path) in err
    outcome = _outcome(out)
    assert outcome["complete"] is True and outcome["registration"]["registered"] is False


def test_deterministic_runs_are_recorded_so_and_the_settings_are_put_back(
    cli, cohort, monkeypatch
):
    """--deterministic: deterministic algorithms on (not warn-only) and
    CUBLAS_WORKSPACE_CONFIG set while the run measures; the manifest records it,
    which makes it another key than a run without it; after the run, the
    process's settings are as they were."""
    import os

    import src.evaluation.measurement as measurement
    from src.evaluation.measurement import run_mode_c as real_run_mode_c

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    seen = []

    def observing(**kwargs):
        seen.append((torch.are_deterministic_algorithms_enabled(),
                     torch.is_deterministic_algorithms_warn_only_enabled(),
                     torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
                     os.environ.get("CUBLAS_WORKSPACE_CONFIG")))
        return real_run_mode_c(**kwargs)

    monkeypatch.setattr(measurement, "run_mode_c", observing)
    before = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic,
              torch.backends.cudnn.benchmark)

    assert cli.main(_argv(cohort, "--deterministic", "--record")) == 0
    assert cli.main(_argv(cohort, "--record")) == 0

    assert seen[0] == (True, False, True, False, ":4096:8")
    assert seen[1][0] is before[0] and seen[1][4] is None
    assert (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark) == before
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    manifests = [json.loads(p.read_text())["manifest"] for p in _reports(cohort)]
    assert sorted(m["deterministic_algorithms"] for m in manifests) == [False, True]
    records = json.loads((cohort.checkpoint.parent / LEDGER).read_text())["records"]
    assert len(records) == 2, "a deterministic run is another key"


def test_an_operation_without_a_deterministic_implementation_refuses_the_run(
    cli, cohort, monkeypatch
):
    """No fallback: torch's refusal becomes one refusal saying the run did not
    complete as requested, nothing written, and the settings are put back.
    Without --deterministic the same error is not reworded."""
    import src.evaluation.measurement as measurement

    error = RuntimeError("scatter_reduce_cuda does not have a deterministic implementation, "
                         "but you set 'torch.use_deterministic_algorithms(True)'.")

    def refuse(**kwargs):
        raise error

    monkeypatch.setattr(measurement, "run_mode_c", refuse)

    with pytest.raises(cli.DeterministicExecutionUnsupported) as refused:
        cli.main(_argv(cohort, "--deterministic", "--record"))

    message = str(refused.value)
    assert "no deterministic implementation" in message and "does not fall back" in message
    assert "scatter_reduce_cuda" in message and message.endswith("Nothing was written.")
    assert _nothing_written(cohort) and _usage(cohort) == []
    assert torch.are_deterministic_algorithms_enabled() is False
    with pytest.raises(RuntimeError) as raw:
        cli.main(_argv(cohort))
    assert raw.value is error


@pytest.mark.parametrize("state", ["bad_cublas_config", "cuda_initialised",
                                   "warn_only_inherited"])
def test_a_process_state_deterministic_execution_cannot_trust_is_refused(
    cli, cohort, monkeypatch, state
):
    import os

    import src.evaluation.cohort_registry as cohort_registry

    monkeypatch.setattr(cohort_registry, "resolve_version",
                        lambda *a, **k: pytest.fail("something was read"))
    if state == "bad_cublas_config":
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":0:0")
        argv, fragment = _argv(cohort, "--deterministic"), "not document as deterministic"
    elif state == "cuda_initialised":
        # CUDA started in this process before the variable could be set: cuBLAS
        # may already run without it (a CPU-only host never gets here alone).
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        argv, fragment = _argv(cohort, "--deterministic"), "CUDA was initialised in this process"
    else:
        torch.use_deterministic_algorithms(True, warn_only=True)
        argv, fragment = _argv(cohort), "warn-only mode"
    try:
        with pytest.raises(cli.DeterministicExecutionUnsupported, match=fragment) as refused:
            cli.main(argv)
    finally:
        torch.use_deterministic_algorithms(False)
    assert "Nothing was read or written" in str(refused.value)
    assert _nothing_written(cohort)
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ or state == "bad_cublas_config"


def _cancel_inside(monkeypatch, module, name):
    """`module.name` that sends this process a SIGINT (the runner's Cancel)
    after it ran, and waits long enough for it to be delivered."""
    import os
    import signal
    import time

    real = getattr(module, name)

    def cancelled(*args, **kwargs):
        answer = real(*args, **kwargs)
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.2)                       # a held SIGINT is not delivered here
        return answer

    monkeypatch.setattr(module, name, cancelled)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_a_cancel_right_after_publishing_still_leaves_the_run_in_its_history(
        cli, cohort, tmp_path, monkeypatch):
    """M5-3 review EVAL-1: a Cancel landing between publishing the report and
    appending it to the version's usage history kept the report out of the
    history (D4, D7), and nothing said so. The three steps are one against a
    cancel: the SIGINT is held until the history entry and the first outcome
    file are written, then ends the run."""
    _cancel_inside(monkeypatch, cli, "_publish_to_store")
    out = tmp_path / "outcome.json"
    with pytest.raises(KeyboardInterrupt):
        cli.main(_argv(cohort, "--outcome", str(out)))
    path, _ = _report(cohort)
    entries = [e for e in _usage(cohort) if e.get("kind") == "test_run"]
    assert [e.get("run_id") for e in entries] == [path.parent.name]
    outcome = _outcome(out)
    assert outcome["complete"] is False and outcome["run_id"] == path.parent.name


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_a_cancel_right_after_the_ledger_write_still_records_the_registration(
        cli, cohort, monkeypatch):
    """EVAL-1, the registration's half: the ledger record and its "registered"
    history entry are written together, a cancel between them held."""
    import src.evaluation.registration as registration

    _cancel_inside(monkeypatch, registration, "write_ledger")
    with pytest.raises(KeyboardInterrupt):
        cli.main(_argv(cohort, "--record"))
    ledger = json.loads((cohort.checkpoint.parent / LEDGER).read_text())
    assert [r["registration"]["sequence"] for r in ledger["records"]] == [1]
    kinds = [e.get("kind") for e in _usage(cohort)]
    assert kinds.count("registered") == 1 and kinds.count("test_run") == 1


# ---------------------------------------------------------------------------
# Only in the deployment's registry (BACKLOG 37; the M5 review's P2-7)
# ---------------------------------------------------------------------------
def _nothing_before_the_lock(cli, monkeypatch):
    """The lock and the device, each failing the test if reached."""
    monkeypatch.setattr(cli, "take_gpu_lock",
                        lambda *a, **k: pytest.fail("the GPU lock was taken"))
    monkeypatch.setattr(cli, "_resolve_device",
                        lambda *a, **k: pytest.fail("the device was resolved"))


def _tree(directory: Path):
    return {p: p.read_bytes() for p in sorted(directory.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("record", [False, True], ids=["measure", "record"])
def test_a_cohort_run_in_a_copy_of_the_registry_is_refused_before_the_lock_and_any_input(
    cli, cohort, tmp_path, monkeypatch, capsys, record
):
    """D7 and D4: every run on a cohort version is in the usage history the
    deployment reads. A run measuring a copy -- with or without `--record` --
    is refused before the GPU lock is taken, the device resolved or any input
    read, and the copy is left as it was (no run directory, no usage entry)."""
    copy = tmp_path / "registry-copy"
    shutil.copytree(cohort.registry, copy, symlinks=True)
    before = _tree(copy)
    _nothing_before_the_lock(cli, monkeypatch)
    argv = ["--checkpoint", str(cohort.checkpoint), "--data-dir", str(cohort.data_dir),
            "--cohort", cohort.version.manifest["label"], "--registry", str(copy),
            "--device", "cpu", *(["--record"] if record else [])]

    with count_opens() as opens:
        assert cli._run_as_program(argv) == 1

    err = capsys.readouterr().err
    assert err.startswith("measure_scorer: refused: --cohort measures a version of the "
                          f"deployment's registry only, and the registry {copy.resolve()} is "
                          "not the deployment's")
    assert f"SHEPHERD_REGISTRY ({cohort.registry}) resolves to {cohort.registry.resolve()}" in err
    assert "Nothing was read or written." in err and "cannot be registered" not in err
    assert not [p for p, n in opens.counts.items() if n and (
        str(copy) in p or str(cohort.data_dir) in p or str(cohort.checkpoint) in p)]
    assert _tree(copy) == before
    assert _nothing_written(cohort)


def test_with_the_setting_unset_a_cohort_run_is_refused_saying_how_to_set_it(
    cli, cohort, monkeypatch, capsys
):
    """A fresh SSH shell: which registry is the deployment's cannot be told, so
    the run is refused before anything, in words true there (no "features ...
    are off", no "cannot be registered later")."""
    monkeypatch.delenv("SHEPHERD_REGISTRY")
    _nothing_before_the_lock(cli, monkeypatch)

    assert cli._run_as_program(_argv(cohort)) == 1

    err = capsys.readouterr().err
    assert ("SHEPHERD_REGISTRY is not set here, so which registry is the deployment's cannot be "
            "told. Set SHEPHERD_REGISTRY in this environment to the value the service's "
            "environment gives it") in err
    assert "features" not in err and "cannot be registered" not in err
    assert _nothing_written(cohort)


def test_a_registry_named_through_a_link_to_the_deployments_is_measured(
    cli, cohort, tmp_path, capsys
):
    """Compared by resolved path: `--registry` through a link to the
    deployment's registry is that registry, and its report registers."""
    alias = tmp_path / "registry-link"
    alias.symlink_to(cohort.registry, target_is_directory=True)
    argv = ["--checkpoint", str(cohort.checkpoint), "--data-dir", str(cohort.data_dir),
            "--cohort", cohort.version.manifest["label"], "--registry", str(alias),
            "--device", "cpu", "--record"]

    assert cli.main(argv) == 0

    assert "Registered as record 1" in capsys.readouterr().out
    assert len(_reports(cohort)) == 1


@pytest.mark.parametrize("when", ["the lock is taken", "the device comes up"])
@pytest.mark.parametrize("record", [False, True], ids=["measure", "record"])
def test_a_registry_link_retargeted_after_the_check_does_not_move_the_run(
    cli, cohort, tmp_path, monkeypatch, capsys, when, record
):
    """EB-2: the root checked is the root the run reads and writes through.
    `--registry` names a link to the deployment's registry (not the link
    `SHEPHERD_REGISTRY` names), and the link is retargeted to a copy after the
    check: while the GPU lock is taken, or while the device is resolved. The
    run measures the deployment's version: its report and its usage-history
    entry are there, the copy is left as it was, and the register-later
    command names the deployment registry's report (with `--record`, the
    report registers). Catches `--registry` resolved again for the read."""
    import contextlib

    from src.evaluation.report_store import read_usage_history

    copy = tmp_path / "registry-copy"
    shutil.copytree(cohort.registry, copy, symlinks=True)
    before = _tree(copy)
    alias = tmp_path / "registry-link"
    alias.symlink_to(cohort.registry, target_is_directory=True)
    retargeted = []

    def retarget():
        alias.unlink()
        alias.symlink_to(copy, target_is_directory=True)
        retargeted.append(when)

    if when == "the lock is taken":
        take = cli.take_gpu_lock

        @contextlib.contextmanager
        def lock_then_retarget(*args, **kwargs):
            with take(*args, **kwargs):
                retarget()
                yield

        monkeypatch.setattr(cli, "take_gpu_lock", lock_then_retarget)
    else:
        resolve = cli._resolve_device

        def retarget_then_resolve(*args, **kwargs):
            retarget()
            return resolve(*args, **kwargs)

        monkeypatch.setattr(cli, "_resolve_device", retarget_then_resolve)
    argv = ["--checkpoint", str(cohort.checkpoint), "--data-dir", str(cohort.data_dir),
            "--cohort", cohort.version.manifest["label"], "--registry", str(alias),
            "--device", "cpu", *(["--record"] if record else [])]

    assert cli.main(argv) == 0

    out = capsys.readouterr().out
    assert retargeted == [when] and alias.resolve() == copy.resolve()
    (report,) = _reports(cohort)
    entries, problems = read_usage_history(cohort.version)
    runs = [e for e in entries if e["kind"] == "test_run"]
    assert problems == [] and [e["run_id"] for e in runs] == [report.parent.name]
    assert _tree(copy) == before
    if record:
        assert "Registered as record 1" in out
    else:
        assert f'record_evaluation.py --report "{report}"' in out
        assert str(copy) not in out


def test_a_registry_that_stops_being_the_deployments_during_the_run_offers_no_command(
    cli, cohort, tmp_path, monkeypatch, capsys
):
    """The critic's case: `SHEPHERD_REGISTRY` names a link retargeted after the
    run was allowed to start. The registration refuses it as not the
    deployment's registry (exit 3), the report is kept, and no register-later
    command is printed: it would be refused the same way."""
    from src.evaluation import registration

    link = tmp_path / "deployment-link"
    link.symlink_to(cohort.registry, target_is_directory=True)
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(link))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    real = registration.register_report

    def retargeted_then_registered(*args, **kwargs):
        link.unlink()
        link.symlink_to(elsewhere, target_is_directory=True)
        return real(*args, **kwargs)

    monkeypatch.setattr(registration, "register_report", retargeted_then_registered)
    outcome = tmp_path / "outcome.json"

    assert cli.main(_argv(cohort, "--record", "--outcome", str(outcome))) == 3

    path, _ = _report(cohort)
    err = capsys.readouterr().err
    assert "NOT REGISTERED." in err and "is not the deployment's" in err
    assert f"report is kept at {path}" in err
    assert "No register-later command is offered" in err
    assert "record_evaluation.py" not in err and "Register it later" not in err
    assert json.loads(outcome.read_text())["registration"]["outcome"] == "refused"
    assert not (cohort.checkpoint.parent / LEDGER).exists()
