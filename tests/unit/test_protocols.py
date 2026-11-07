"""
The evaluation protocol definition (`src/evaluation/protocols.py`).
===================================================================
**The product table pins** (document B rev 4 §5.5, row 1): the protocols this
build implements, each naming an evaluator the scorer dispatch has and a
model requirement the model spec can write, and a module the WebUI and the
job runner import without torch. The stages and tasks are checked against
`src.models.bundle.spec` here, in the test: the module itself imports nothing
outside the standard library.

**The consumers judge a new measurement by it, and only a new one** (§5.5,
the rows after the first; §5.6):

- under the test-only protocols p1 and p2 (the product table untouched),
  the command line measures the session's disease child as p2 and records
  that name, and the job runner launches p1: nothing downstream takes a
  protocol's name for the model's task (R1). The Test page's part is in
  tests/unit/test_webui_test_page.py
  (`test_a_listed_disease_child_offers_the_test_only_protocols_and_starts_one`);
- one reason, the definition's, from Start's reason, the Test page's start,
  the runner and the command line, before a job starts or a file is written:
  for a base asked as p1, and for a cohort version that lacks the genes the
  gene measurement needs (the command line then refuses before it opens the
  checkpoint);
- the base message, and a name that is no protocol answered by the ordinary
  sentence (the runner's cases are in tests/unit/test_job_runner.py,
  `test_inputs_of_the_wrong_shape_are_refused_before_launch`);
- the standing guard: reports whose modes are no protocol here are built,
  kept, registered where registration allows it at this revision, listed and
  shown, with the protocol table and every judgment of the definition
  made to fail if read.

Module: tests/unit/test_protocols.py
"""
from __future__ import annotations

import ast
import hashlib
import html
import inspect
import json
import re
import shutil
import sys
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluation import protocols

#: The version the cohort fixture imports (`tests/fixtures/registered_cohort.LABEL`).
LABEL = "demo-test-1"
#: A second version of the same graph source, whose cases list causal genes and
#: no candidate genes.
CAUSAL_ONLY = "demo-causal-1"
#: The products' sentence for a name that is not one of them.
NOT_IMPLEMENTED = "{!r} is not a measurement this build implements (disease, gene, patients_like_me)."


def test_the_product_protocols_are_disease_gene_and_patients_like_me():
    assert protocols.names() == ("disease", "gene", "patients_like_me")
    # Each row is keyed by its own name, the name `--mode` and `manifest.mode` carry.
    assert all(name == row.name for name, row in protocols.PROTOCOLS.items())


def test_each_protocol_names_an_evaluator_and_a_model_the_spec_can_write():
    from src.evaluation.task_measurement import TASK_RUNS
    from src.models.bundle.spec import LEARNING_STAGES, TASKS

    for row in protocols.PROTOCOLS.values():
        assert row.evaluator in TASK_RUNS, row
        assert row.learning_stage in LEARNING_STAGES, row
        assert row.task is None or row.task in TASKS, row


def test_a_protocol_that_needs_genes_reads_the_gene_fields():
    for row in protocols.PROTOCOLS.values():
        assert not row.needs or row.gene_fields, row


def test_the_module_imports_only_the_standard_library():
    tree = ast.parse(inspect.getsource(protocols))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, ast.dump(node)
            imported.add(node.module.split(".")[0])
    assert imported, "no import found: the check would pass vacuously"
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}, imported


# ---------------------------------------------------------------------------
# What the consumers share: test-only protocols, the site, the runner
# ---------------------------------------------------------------------------
@pytest.fixture
def test_only_protocols(monkeypatch):
    """§5.5's test-only protocols, the product table untouched: p1 and p2 both
    take a finetuned model with the disease head and run the disease
    evaluator, so neither name is a task. Every consumer reads the table
    through the module when it judges, so this reaches them all."""
    table = {name: protocols.Protocol(name, "finetune", "disease", "disease")
             for name in ("p1", "p2")}
    monkeypatch.setattr(protocols, "PROTOCOLS", table)
    return table


def _cli():
    """The measurement command line (`scripts/measure_scorer.py`)."""
    pytest.importorskip("torch")
    import scripts.measure_scorer as measure_scorer

    return measure_scorer


def _files(directory: Path) -> dict:
    """Every file under `directory`, with its size and modification time."""
    return {str(p.relative_to(directory)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(directory.rglob("*")) if p.is_file()}


class _Popens:
    """`subprocess.Popen` replaced (the minimal fake of
    tests/unit/test_job_runner.py): each call is recorded and takes the next
    fake process."""

    def __init__(self) -> None:
        self.queue: list = []
        self.calls: list = []

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        return self.queue.pop(0)


@pytest.fixture
def runner(monkeypatch):
    """A job runner whose spawns are `_Popens`' and whose job directories
    (`tempfile.mkdtemp` with the runner's prefix) are recorded; the
    deployment's device is unset, so the developer's environment never
    decides."""
    pytest.importorskip("fastapi")
    import src.api.services.training_manager as tm
    from tests.fixtures.job_processes import FakeProcess

    monkeypatch.delenv("SHEPHERD_DEVICE", raising=False)
    popens = _Popens()
    monkeypatch.setattr("subprocess.Popen", popens)
    job_dirs: list = []
    mkdtemp = tm.tempfile.mkdtemp

    def recorded(*args, **kwargs):
        made = mkdtemp(*args, **kwargs)
        if str(kwargs.get("prefix", "")).startswith("shepherd-test-job-"):
            job_dirs.append(made)
        return made

    monkeypatch.setattr(tm.tempfile, "mkdtemp", recorded)
    state = SimpleNamespace(manager=tm.TrainingManager(), popens=popens, job_dirs=job_dirs,
                            process=FakeProcess, TestRunInputs=tm.TestRunInputs)
    yield state
    for process in popens.queue:
        process.finish(0)
    job = state.manager._last.get("test")
    if job is not None and job.process is not None:
        job.process.finish(1)
        job.done.wait(5)


def _nothing_started(runner) -> bool:
    """No process spawned, no job directory made, no job recorded."""
    return (runner.popens.calls == [] and runner.job_dirs == []
            and runner.manager._last.get("test") is None)


@pytest.fixture(scope="module")
def site(trained_disease_child, tmp_path_factory):
    """A workspaces root whose workspace ``demo`` lists the session's trained
    base and disease child (`trained_disease_child`, copied: the session's
    files stay read-only) and a gene child's record (a synthetic payload as
    training writes it, never measured), and a registry with two versions of
    the workspace's graph source: `LABEL`, whose source lists no genes, and
    `CAUSAL_ONLY`, whose cases list a causal gene each and no candidate gene.
    Nothing here is changed by a test."""
    pytest.importorskip("torch")
    from tests.fixtures.registered_cohort import (
        graph_source_cases,
        import_version,
        registered_cohort_from_child,
        write_source,
    )
    from tests.fixtures.trained_children import BASE_RUN, CHILD_RUNS
    from tests.unit.test_task_first_selection import _put, _record_payload

    root = tmp_path_factory.mktemp("protocols")
    cohort = registered_cohort_from_child(trained_disease_child, root / "cohort")
    cases = [dict(case, causal_genes=["SCN1A"]) for case in graph_source_cases(cohort.data_dir)]
    import_version(cohort.registry, cohort.data_dir, write_source(root / "causal.json", cases),
                   label=CAUSAL_ONLY)
    demo = root / "ws" / "demo"
    shutil.copytree(cohort.data_dir, demo)
    listed = {}
    for run, trained in ((BASE_RUN, trained_disease_child["base"]),
                         (CHILD_RUNS["disease"], trained_disease_child["disease"])):
        path = demo / "runs" / "gatv2" / run / "checkpoints" / "last.pt"
        path.parent.mkdir(parents=True)
        shutil.copy2(trained, path)
        listed[run] = path
    gene_run = "20261003-000000_gene_cccccc"
    gene = _put(demo, "gatv2", gene_run, _record_payload("gene"), time.time())
    return SimpleNamespace(
        ws=root / "ws", demo=demo, registry=cohort.registry,
        base=listed[BASE_RUN], base_key=f"gatv2/{BASE_RUN}/last.pt",
        child=listed[CHILD_RUNS["disease"]], gene=gene, gene_key=f"gatv2/{gene_run}/last.pt")


@pytest.fixture
def listed(site, monkeypatch):
    """The Test page's lists of `site` for an operator's session: the page
    (the registry's versions), the workspace's checkpoints and its graph
    binding, as the page reads them; and the inputs the page would hand the
    runner for a listed checkpoint, version and protocol."""
    pytest.importorskip("gradio")
    from src.webui.components import served_status
    from src.webui.components import test_page as tp
    from tests.fixtures.operator_access import operator_request

    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(site.ws))
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(site.registry))
    monkeypatch.setenv("SHEPHERD_DEVICE", "cpu")
    monkeypatch.delenv("SHEPHERD_TRAINING_EVIDENCE_DIRS", raising=False)
    monkeypatch.setattr(served_status, "read_status", lambda: (None, "not read"))
    request = operator_request(monkeypatch)
    page = tp.read_page(request)
    models = tp.read_models("demo", request)
    binding = tp.workspace_binding(site.demo, tp.find(page["workspaces"], "name", "demo"))

    def inputs(runner, key, label, protocol):
        """`TestRunInputs` from the listing, as `start_test_run` builds them."""
        item, version = tp.find_model(models, key), tp.find(page["versions"], "label", label)
        return runner.TestRunInputs(
            registry=site.registry, version_label=label,
            version_sha256=version["manifest_sha256"], workspace=site.demo,
            checkpoint=site.demo / "runs" / key.replace("/last.pt", "/checkpoints/last.pt"),
            checkpoint_sha256=item["sha256"], protocol=protocol,
            learning_stage=item["learning_stage"], task=item["task"],
            version_counts=version["counts"])

    def start_state(key, label, protocol):
        return tp.start_state(page, models, "demo", label, key, binding, ("none", ""),
                              protocol=protocol)

    return SimpleNamespace(tp=tp, page=page, models=models, request=request, inputs=inputs,
                           start_state=start_state, site=site)


# ---------------------------------------------------------------------------
# Wiring with test-only protocols (§5.5, row 2)
# ---------------------------------------------------------------------------
def test_the_command_line_measures_by_a_protocol_whose_name_is_no_task(
        trained_disease_child, test_only_protocols, tmp_path, capsys):
    """R1 at the command line: `--mode p2` measures the session's disease
    child (p2 takes its stage and head), the report's mode is the name asked,
    and its candidates are the disease evaluator's; the run writes its
    `--output` (with the ranks beside it) and nothing else. "disease" is no
    choice under this table: argparse's ordinary rule."""
    cli = _cli()
    from src.evaluation.task_measurement import DISEASE_SEMANTICS

    workspace = Path(trained_disease_child["data_dir"])
    before = _files(workspace)
    out = tmp_path / "p2.json"
    assert cli.main(["--mode", "p2", "--checkpoint", str(trained_disease_child["disease"]),
                     "--data-dir", str(workspace), "--split", "val", "--output", str(out),
                     "--device", "cpu"]) == 0
    manifest = json.loads(out.read_text())["manifest"]
    assert manifest["mode"] == "p2"
    assert manifest["candidate_construction"] == DISEASE_SEMANTICS["candidate_construction"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["p2.json", "p2_ranks.json"]
    assert _files(workspace) == before
    capsys.readouterr()
    with pytest.raises(SystemExit) as refused:
        cli.parse_args(["--mode", "disease", "--checkpoint", "c.pt", "--data-dir", "d",
                        "--split", "val"])
    assert refused.value.code == 2
    assert "argument --mode: invalid choice: 'disease'" in capsys.readouterr().err


def test_the_runner_launches_a_protocol_whose_name_is_no_task(runner, test_only_protocols,
                                                              tmp_path):
    """R1 at the job runner: the protocol asked (p1) is judged on the listed
    record (finetune, disease) and becomes `--mode`; "disease" is no protocol
    under this table and is refused by the definition's ordinary sentence
    before anything is spawned."""
    registry = tmp_path / "registry"
    registry.mkdir()
    workspace = tmp_path / "workspaces" / "ws"
    asked = runner.TestRunInputs(
        registry=registry, version_label=LABEL, version_sha256="e" * 64, workspace=workspace,
        checkpoint=workspace / "runs" / "gatv2" / "r1" / "checkpoints" / "last.pt",
        checkpoint_sha256="c" * 64, protocol="disease", learning_stage="finetune",
        task="disease", version_counts={"source_cases": 3})

    refused = runner.manager.start_test(asked)
    assert refused["success"] is False
    assert refused["error"] == ("'disease' is not a measurement this build implements (p1, p2). "
                                "Nothing was started.")
    assert _nothing_started(runner)

    process = runner.process(pid=7)
    runner.popens.queue.append(process)
    assert runner.manager.start_test(replace(asked, protocol="p1"))["success"] is True
    (argv,) = runner.popens.calls
    assert argv[argv.index("--mode") + 1] == "p1" and argv.count("--mode") == 1
    assert "--task" not in argv and "--learning-stage" not in argv
    shown = runner.manager.get_test_status()["inputs"]
    assert (shown["protocol"], shown["learning_stage"], shown["task"]) == (
        "p1", "finetune", "disease")
    process.finish(1)
    assert runner.manager._last["test"].done.wait(5)


# ---------------------------------------------------------------------------
# One reason, before a job starts or a file is written (§5.5, rows 3 and 4)
# ---------------------------------------------------------------------------
def test_a_base_asked_as_p1_gets_one_reason_everywhere_and_nothing_starts(
        listed, runner, test_only_protocols, tmp_path):
    """The listed base of the session's trained pair, asked as p1: Start's
    reason, the Test page's start, the runner called directly with the
    listing's values and the command line on that file's bytes all say
    `protocols.refusal("p1", "base", None, counts)`, the base message. No
    process is spawned, no job directory made, no `--output` written."""
    cli = _cli()
    from src.models.bundle.record import ModelRecordError

    tp, site = listed.tp, listed.site
    item = tp.find_model(listed.models, site.base_key)
    assert (item["record"], item["learning_stage"], item["task"]) == ("model", "base", None)
    counts = tp.find(listed.page["versions"], "label", LABEL)["counts"]
    reason = protocols.refusal("p1", "base", None, counts)
    assert reason == protocols.BASE_MESSAGE

    assert listed.start_state(site.base_key, LABEL, "p1") == (False, f"Disabled: {reason}")
    assert tp.start_test_run(listed.page, listed.models, "demo", LABEL, site.base_key, False,
                             False, listed.request, protocol="p1", manager=runner.manager) == {
        "success": False, "error": f"{reason} Nothing was started."}
    result = runner.manager.start_test(listed.inputs(runner, site.base_key, LABEL, "p1"))
    assert (result["success"], result["error"]) == (False, f"{reason} Nothing was started.")

    out = tmp_path / "base.json"
    with pytest.raises(ModelRecordError) as refused:
        cli.main(["--mode", "p1", "--checkpoint", str(site.base), "--data-dir", str(site.demo),
                  "--split", "val", "--output", str(out), "--device", "cpu"])
    assert str(refused.value) == f"{reason} Nothing was written."
    assert _nothing_started(runner)
    assert not any(tmp_path.iterdir())


def test_a_version_without_the_genes_asked_for_is_refused_before_the_checkpoint(
        listed, runner):
    """The product table: the gene measurement asked of a listed gene child on
    a version whose source lists causal genes and no candidate genes. Start's
    reason, the runner and the command line say the definition's one sentence
    (`protocols.inputs_refusal`); the command line says it right after the
    version's `cohort.json` is read, before it opens the workspace or the
    checkpoint, and writes nothing."""
    cli = _cli()
    from tests.fixtures.opens import count_opens

    tp, site = listed.tp, listed.site
    item = tp.find_model(listed.models, site.gene_key)
    assert (item["record"], item["learning_stage"], item["task"]) == ("model", "finetune", "gene")
    counts = tp.find(listed.page["versions"], "label", CAUSAL_ONLY)["counts"]
    assert counts["genes"]["cases_with_causal_genes"] > 0
    assert counts["genes"]["cases_with_candidate_genes"] == 0
    reason = protocols.refusal("gene", "finetune", "gene", counts)
    assert reason == protocols.inputs_refusal("gene", counts) == (
        "this cohort version lists no candidate genes, which the gene measurement needs. "
        "Import a source that lists candidate_genes.")

    assert listed.start_state(site.gene_key, CAUSAL_ONLY, "gene") == (False, f"Disabled: {reason}")
    result = runner.manager.start_test(listed.inputs(runner, site.gene_key, CAUSAL_ONLY, "gene"))
    assert (result["success"], result["error"]) == (False, f"{reason} Nothing was started.")
    assert _nothing_started(runner)

    version = site.registry / "versions" / CAUSAL_ONLY
    with count_opens() as opens, pytest.raises(ValueError) as refused:
        cli.main(["--mode", "gene", "--checkpoint", str(site.gene), "--data-dir", str(site.demo),
                  "--cohort", CAUSAL_ONLY, "--registry", str(site.registry), "--device", "cpu"])
    assert str(refused.value) == f"{reason} Nothing was written."
    assert opens.opens(version / "cohort.json") == 1
    assert opens.opens(site.gene) == 0, "the checkpoint was opened"
    assert opens.opens(site.demo / "split_manifest.json") == 0, "the workspace was read"
    assert not (version / "reports").exists()


# ---------------------------------------------------------------------------
# The base message (§5.4) and names that are no protocol
# ---------------------------------------------------------------------------
def test_the_base_message_names_the_formal_test_as_not_implemented(monkeypatch):
    """§5.4: the reason for a base while no protocol takes one, whatever is
    asked of it. It names the formal test as not implemented and never says
    that a base has nothing to test; once a protocol takes a base (a
    test-only row here), a base is measured by it and the message is gone."""
    message = protocols.BASE_MESSAGE
    assert "formal test" in message and "not implemented" in message
    assert "no task to measure" not in message and "nothing to test" not in message
    assert protocols.available("base", None) == ((), message)
    for name in protocols.names():
        assert protocols.model_refusal(name, "base", None) == message
        assert protocols.refusal(name, "base", None, {}) == message

    takes_a_base = protocols.Protocol("p_base", "base", None, "disease")
    monkeypatch.setattr(protocols, "PROTOCOLS", {**protocols.PROTOCOLS, "p_base": takes_a_base})
    assert protocols.available("base", None) == (("p_base",), None)
    assert protocols.model_refusal("p_base", "base", None) is None
    assert protocols.model_refusal("disease", "base", None) == (
        "the disease measurement takes a finetune model with the disease head; this checkpoint "
        "is a base model with no task head. It can be measured by p_base.")


@pytest.mark.parametrize("name", ["C", "D", None, "Disease"])
def test_a_name_that_is_no_protocol_gets_the_ordinary_sentence(name, capsys):
    """A name this build does not implement, whatever it once meant, is
    refused by the definition's ordinary sentence naming the protocols it
    does implement, before any record or version is looked at: by every
    judgment the definition offers, and by the command line's internal entry
    before it reads anything; on the command line itself by argparse's own
    "invalid choice". The runner's cases are in tests/unit/test_job_runner.py."""
    cli = _cli()
    import torch

    from tests.fixtures.opens import count_opens

    said = NOT_IMPLEMENTED.format(name)
    for stage, task in (("finetune", "disease"), ("base", None), (None, None)):
        assert protocols.model_refusal(name, stage, task) == said
        assert protocols.refusal(name, stage, task, {}) == said
    assert protocols.inputs_refusal(name, {"genes": {"cases_with_causal_genes": 3}}) == said
    assert protocols.available("finetune", "disease") == (("disease",), None)

    def read():
        cli.read_measurement_inputs(Path("/nowhere/last.pt"), Path("/nowhere"),
                                    torch.device("cpu"), split="val", protocol=name)

    # A first call outside the counter does the entry's call-time imports, so
    # the counted call sees only what the refusal itself would open.
    with pytest.raises(ValueError):
        read()
    with count_opens() as opens, pytest.raises(ValueError) as refused:
        read()
    assert str(refused.value) == f"{said} Nothing was written."
    assert not [path for path, n in opens.counts.items() if n], "something was opened"
    if name is not None:
        capsys.readouterr()
        with pytest.raises(SystemExit) as stopped:
            cli.parse_args(["--mode", name, "--checkpoint", "c.pt", "--data-dir", "d",
                            "--split", "val"])
        assert stopped.value.code == 2
        assert f"argument --mode: invalid choice: {name!r}" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# It never gates reading (§5.5, the standing guard; §5.6)
# ---------------------------------------------------------------------------
class _NotForReaders(Mapping):
    """A protocol table that fails when it is read."""

    def __getitem__(self, name):
        raise AssertionError(f"a reader read the protocol definition ({name!r})")

    def __iter__(self):
        raise AssertionError("a reader read the protocol definition")

    def __len__(self):
        raise AssertionError("a reader read the protocol definition")


def _readers_never_judge(monkeypatch) -> None:
    """The protocol table and every judgment `src.evaluation.protocols`
    offers fail if read, so a reader that consulted the definition would fail
    the test, not pass by luck."""
    monkeypatch.setattr(protocols, "PROTOCOLS", _NotForReaders())
    for name in ("names", "available", "model_refusal", "inputs_refusal", "refusal"):
        monkeypatch.setattr(protocols, name, lambda *a, _name=name, **k: pytest.fail(
            f"a reader called protocols.{_name}"))


def test_reports_whose_modes_are_no_protocol_are_read_as_recorded(tmp_path, monkeypatch):
    """§5.6: the protocol definition judges whether a new measurement can run,
    and nothing else. Kept reports whose modes are not protocols here -- a
    Mode C and a Mode D report (`mode_c_report`, `mode_d_report`), the disease
    report with its mode set to "kg_link_prediction" (a reader-only label: no
    such test is implemented, N1) and one with mode "p1" -- are built into
    records, kept in a version's report store, listed by the store's scan and
    the Test page, registered, read from the ledger and shown by the API, the
    Models page and the Test page, each under its own mode, with the
    protocol table and every judgment of the definition made to fail if read.

    Since K8-6c no rule refuses a report by its mode (policy rev 15): the Mode
    D report registers by the general rules like the others, under its own
    mode."""
    pytest.importorskip("gradio")
    pytest.importorskip("fastapi")
    from src.api.routes.evaluations import _recorded_results
    from src.evaluation.caveats import mode_label
    from src.evaluation.cohort_registry import resolve_version
    from src.evaluation.registration import register_report
    from src.evaluation.report_store import scan_report_store
    from src.evaluation.sidecar import build_record, read_ledger, records_for
    from src.utils.fingerprint import ReadIdentity
    from src.webui.components import models_page
    from src.webui.components import test_page as tp
    from tests.fixtures.report_store import store_report
    from tests.unit.test_evaluation_sidecar import (
        mode_c_report,
        mode_d_report,
        registered_report,
    )

    weights = tmp_path / "runs" / "checkpoints" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"stand-in weights")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    registry = tmp_path / "registry"
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(registry))
    subjects = {"C": mode_c_report(), "D": mode_d_report(),
                "kg_link_prediction": registered_report(mode="kg_link_prediction"),
                "p1": registered_report(mode="p1")}
    assert not set(subjects) & set(protocols.names())
    _readers_never_judge(monkeypatch)

    records, paths = {}, {}
    for mode, report in subjects.items():
        report["manifest"]["artifact_digests"]["checkpoint"] = digest
        records[mode] = build_record(report, "f" * 64)
        assert records[mode]["scoring"]["mode"] == mode
        paths[mode] = store_report(registry, report)

    # Listed: the store's one scan builds each kept report into its record.
    version = resolve_version(registry, records["C"]["cohort"]["label"])
    scan = scan_report_store(version)
    for mode, path in paths.items():
        run_id = path.parent.name
        key, problem = scan.run_key(run_id, scan.digests[run_id])
        assert problem is None and key is not None, (mode, problem)
        assert scan.reports[key][run_id][1]["scoring"]["mode"] == mode
    assert scan.unplaced == () and scan.unbuilt == {}

    # Registered: every one, by the general rules, under its own mode.
    for mode, path in paths.items():
        outcome = register_report(path, checkpoint=weights,
                                  registered_at="2026-11-06T00:00:00Z").outcome
        assert outcome == "registered", (mode, outcome)
    ledger, _sha = read_ledger(weights.parent / "evaluations.json")
    registered = ["C", "D", "kg_link_prediction", "p1"]
    assert [r["scoring"]["mode"] for r in records_for(ledger, digest)] == registered

    # Shown: the API's view of the ledger beside the weights and the Models
    # page's view of each ledger record; every kept report, D's included, in
    # the Test page's report list and Result.
    api = _recorded_results(ReadIdentity(weights, digest))
    assert api.state == "records" and api.records_not_shown == 0
    assert [r.mode for r in api.records] == registered
    views = [models_page.record_view(record) for record in records_for(ledger, digest)]
    assert [view and view["mode"] for view in views] == registered
    reports = tp.read_reports(version.manifest["label"])
    assert reports["problem"] is None
    runs = {run["run_id"]: run for run in reports["runs"]}
    for mode, path in paths.items():
        run = runs[path.parent.name]
        assert run["record"]["scoring"]["mode"] == mode and run["unbuilt"] is None
        shown = html.unescape(re.sub(r"<[^>]+>", "", tp.result_html(reports, run["run_id"], [])))
        assert mode_label(mode, run["record"]["scoring"]["score_semantics"]) in shown
    assert len(re.findall(r"report kept", html.unescape(
        re.sub(r"<[^>]+>", "", tp.reports_html(reports, None))))) == len(paths)
