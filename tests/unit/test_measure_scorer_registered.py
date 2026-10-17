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
  the directory again; a refused registration keeps the report and exits 3.

The end-to-end acceptance list, through the real CLIs, is
`tests/integration/test_m4_flow.py`.

Module: tests/unit/test_measure_scorer_registered.py
"""
from __future__ import annotations

import hashlib
import json
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
def cohort(base, tmp_path):
    return copy_registered_cohort(base, tmp_path / "root")


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
    cli, tmp_path, record, training, validation, val_opens
):
    """A pipeline record finds both scopes in the workspace; with no validation
    recorded, validation is `no_validation_inputs_recorded` and `val` is never
    read; a graph-only record is not the pipeline's, so both scopes are
    unverifiable and no candidate is read. Never a count without evidence."""
    cohort = build_registered_cohort(tmp_path / "root", training_record=record)

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
    cli, tmp_path, capsys, record
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
