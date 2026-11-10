"""
Task measurements on generated-split data versions.
===================================================
Research comparison §5 Stage 5b (the owner's added item A, its measurement
half). A workspace's own generated split kept in the registry (Stage 4b) is
measured by ``measure_scorer --cohort <label>``, through the same evaluators
``--split`` uses, its data read through the registry. What is pinned here, on
the demo workspace and a base with a disease, a gene and a patients-like-me
child trained on it for real:

- **the gene fixture travels with the version**: when the workspace holds the
  functional gene fixture of its build, the import keeps its record and the
  role's rows byte for byte, bound by digest, and counts the rows; when it
  holds none, or one of another build or that does not describe the split,
  the version keeps none and the import says why, refusing nothing; a
  fixture whose files contradict its record is read by the one reader of a
  workspace's fixture, so ``--split`` refuses it and the import does not
  keep it, for the same reason;
- **the disease child** is measured on a generated train version and a
  generated val version: each report is kept in its version's report store,
  its manifest records cohort_kind ``generated``, the version's block (its
  role as recorded and its generation identity) and the role as the split,
  never test, with no R10;
- **the gene and patients-like-me children** are measured through the
  fixture the version keeps, on train and val; on a version without one they
  are refused before anything is read, with the stated data gap;
- **the same evaluator**: a version run scores exactly as the ``--split`` run
  of the same split does, whose path is unchanged (its report at --output,
  still unregistrable with today's sentence, the registry untouched);
- **refusals only where a run cannot be valid**, each before anything is
  written: the version's record contradicting its bytes (samples, the fixture
  record or rows, an edited copy or count), the version bound to another
  graph, a protocol the model or the data kind does not take, R10 evidence;
  a damaged kept fixture refuses only the measurements that read it;
- **kept, never overwritten**: two runs are two run directories; ``--record``
  keeps the report and says this revision's ledger does not record a
  generated split's run yet (Stage 6);
- **the reader and references**: an edited fixture block or count is refused
  by name; a kept fixture whose copy or bytes differ reads as ``differs``.

A generated ``test`` role cannot be imported today (the generator writes
train and val; tests/unit/test_generated_versions.py), so none is measured.

Module: tests/unit/test_generated_measurement.py
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, NamedTuple

import pytest

torch = pytest.importorskip("torch")

ROLES = ("train", "val")
TASKS = ("disease", "gene", "patients_like_me")
LEDGER = "evaluations.json"
BARE = "gen-val-bare"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tree(root: Path) -> dict:
    return {p: p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


class Site(NamedTuple):
    root: Path
    data_dir: Path
    bare: Path
    checkpoints: dict
    registry: Path


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """The demo workspace, a base and a child per task trained on it (the gene
    fixture derived on the way, `train_children`), a copy of the workspace
    without its fixture, and a registry holding the workspace's train and val
    splits (each keeping the fixture) and the copy's val split (keeping
    none). Nothing is measured here."""
    from src.evaluation.cohort_registry import commit_import, prepare_generated_import
    from tests.fixtures.registered_cohort import write_demo_workspace
    from tests.fixtures.trained_children import train_children

    root = (tmp_path_factory.mktemp("gen_measure") / "root").resolve()
    data_dir = write_demo_workspace(root / "workspace")
    trained = train_children(data_dir, TASKS, runs=root / "runs")
    checkpoints = {}
    for name in ("base", *TASKS):
        target = root / "models" / name / "last.pt"
        target.parent.mkdir(parents=True)
        shutil.copyfile(trained[name], target)
        checkpoints[name] = target
    shutil.rmtree(root / "runs")
    bare = Path(shutil.copytree(data_dir, root / "bare", ignore=shutil.ignore_patterns("derived")))
    registry = root / "registry"
    registry.mkdir()
    for role in ROLES:
        plan = prepare_generated_import(registry, data_dir, f"gen-{role}", role)
        commit_import(plan, registry, plan.preview_id)
    plan = prepare_generated_import(registry, bare, BARE, "val")
    commit_import(plan, registry, plan.preview_id)
    return Site(root, data_dir, bare, checkpoints, registry)


@pytest.fixture
def site(built, tmp_path, monkeypatch):
    """A copy of the built tree for one test, its registry the deployment's
    (`SHEPHERD_REGISTRY`): a ``--cohort`` run measures only there."""
    root = Path(shutil.copytree(built.root, tmp_path / "root"))

    def moved(path: Path) -> Path:
        return root / path.relative_to(built.root)

    copied = Site(root, moved(built.data_dir), moved(built.bare),
                  {name: moved(path) for name, path in built.checkpoints.items()},
                  moved(built.registry))
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(copied.registry))
    return copied


@pytest.fixture
def cli():
    import scripts.measure_scorer as module

    return module


def _argv(site: Site, label: str, task: str = "disease", *extra: str, mode: str = None):
    return ["--mode", mode or task, "--checkpoint", str(site.checkpoints[task]),
            "--data-dir", str(site.data_dir), "--cohort", label,
            "--registry", str(site.registry), "--device", "cpu", *extra]


def _version(site: Site, label: str):
    from src.evaluation.cohort_registry import resolve_version

    return resolve_version(site.registry, label)


def _runs(site: Site, label: str) -> list:
    from src.evaluation.cohort_registry import is_run_id

    reports = site.registry / "versions" / label / "reports"
    return sorted(p for p in reports.iterdir() if is_run_id(p.name)) if reports.is_dir() else []


def _report(run: Path) -> dict:
    return json.loads((run / "report.json").read_text())


def _fixture_file(site: Site, name: str) -> bytes:
    return (site.data_dir / "derived" / "gene_fixture" / name).read_bytes()


def _edit_manifest(site: Site, label: str, edit) -> None:
    """Rewrite a version's cohort.json with `edit` applied: an edit made after
    the import, which the version's readers must catch."""
    from src.evaluation.cohort_registry import encode_json

    path = site.registry / "versions" / label / "cohort.json"
    manifest = json.loads(path.read_text())
    edit(manifest)
    path.write_bytes(encode_json(manifest))


def _nothing_written(site: Site, before: dict) -> None:
    assert _tree(site.registry) == before
    for checkpoint in site.checkpoints.values():
        assert not (checkpoint.parent / LEDGER).exists()


# ---------------------------------------------------------------------------
# The import keeps the gene fixture, or says why it does not
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("role", ROLES)
def test_the_import_keeps_the_workspace_s_gene_fixture_byte_for_byte(site, role):
    from src.evaluation import cohort_registry as cr

    version = _version(site, f"gen-{role}")
    manifest = version.manifest
    record_bytes = _fixture_file(site, "task_fixture.json")
    rows_bytes = _fixture_file(site, f"{role}_samples.json")
    record = json.loads(record_bytes)
    assert (version.directory / "task_fixture.json").read_bytes() == record_bytes
    assert (version.directory / "task_fixture_samples.json").read_bytes() == rows_bytes
    fixture = manifest["task_fixture"]
    assert fixture == {
        "kind": "gene_fixture", "label": "functional fixture", "rule": record["rule"],
        "seed": 0, "distractors": 4, "derivation": record["counts"][role],
        "record": {"file": "task_fixture.json", "sha256": _sha(record_bytes),
                   "schema_version": 1},
        "samples": {"file": "task_fixture_samples.json", "sha256": _sha(rows_bytes),
                    "n": record["files"][role]["rows"]},
    }
    # The fixture was derived from the very samples the version keeps.
    assert record["workspace"]["samples"][role] == manifest["samples"]["sha256"]
    assert record["workspace"]["split_manifest"] == manifest["split_manifest"]["sha256"]
    rows = len(json.loads(rows_bytes))
    assert manifest["counts"] == {"samples": manifest["samples"]["n"],
                                  "diseases": manifest["counts"]["diseases"],
                                  "task_fixture_samples": rows}
    assert manifest["count_units"]["task_fixture_samples"] == cr.GENERATED_FIXTURE_COUNT_UNIT
    assert cr.cohort_version_block(version)["task_fixture"] == fixture
    states = {s.reference.name: s for s in cr.check_references(version,
                                                               workspace=site.data_dir)}
    for name in ("task_fixture", "task_fixture.samples", "samples", "split_manifest"):
        assert states[name].state == cr.VERIFIED, (name, states[name].reason)
    assert str(site.root) not in json.dumps(manifest)


def test_a_workspace_without_the_fixture_gives_a_version_that_keeps_none(site):
    from src.evaluation import cohort_registry as cr

    version = _version(site, BARE)
    assert "task_fixture" not in version.manifest
    assert set(version.manifest["counts"]) == set(cr.GENERATED_COUNT_UNITS)
    assert set(version.manifest["count_units"]) == set(cr.GENERATED_COUNT_UNITS)
    assert not (version.directory / "task_fixture.json").exists()
    assert not (version.directory / "task_fixture_samples.json").exists()
    assert cr.cohort_version_block(version)["task_fixture"] is None
    names = {ref.name for ref in cr.version_references(version)}
    assert not {"task_fixture", "task_fixture.samples"} & names
    # The same split as the fixture-keeping version, by content.
    kept = _version(site, "gen-val")
    assert version.manifest["samples"] == kept.manifest["samples"]


def test_the_preview_shows_the_fixture_kept_or_why_none_is(site, capsys):
    """The import's preview names the fixture kept (rule, seed, derivation and
    the two files by digest) or why none is, and what that leaves
    unmeasured; the commit says which. No row is printed."""
    import scripts.import_cohort as import_cohort

    base = ["--kind", "generated", "--registry", str(site.registry), "--role", "val"]
    assert import_cohort.main([*base, "--data-dir", str(site.data_dir), "--label", "v2"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"gene fixture +the workspace's functional fixture, rule "
                     r"one-causal-gene-from-the-disease-plus-distractors/v1, seed 0, 4 "
                     r"distractors \(cases 5, derived 5, ", out), out
    assert "task_fixture_samples.json, 5 rows, sha256 " in out
    command = out.strip().splitlines()[-1].split()
    assert import_cohort.main([*base, "--data-dir", str(site.data_dir), "--label", "v2",
                               "--commit", command[-1]]) == 0
    assert "with its functional gene fixture (5 rows)" in capsys.readouterr().out

    assert import_cohort.main([*base, "--data-dir", str(site.bare), "--label", "v3"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"gene fixture +none kept \(absent: No gene task fixture at ", out), out
    assert ("the gene and patients-like-me measurements are refused on this version for that "
            "data gap") in out
    assert "patient_id" not in out and "sim_" not in out


def _another_build(record):
    record["workspace"]["split_manifest"] = "0" * 64


def _other_samples(record):
    record["workspace"]["samples"]["val"] = "1" * 64


def _path_name(record):
    record["files"]["val"]["name"] = "../val_samples.json"


def _seed_as_text(record):
    record["seed"] = "zero"


@pytest.mark.parametrize("edit, said", [
    (_another_build, "another build: The gene fixture was derived from another build"),
    (_other_samples, "it was derived from other val samples than the version keeps"),
    (_path_name, "its val file is not named val_samples.json"),
    (_seed_as_text, "does not state what a version records of it: task_fixture.seed must be a "
                    "whole number"),
], ids=["another-build", "other-samples", "path-name", "seed-as-text"])
def test_a_fixture_that_does_not_describe_the_split_is_not_kept_and_refuses_nothing(
        site, edit, said):
    from src.evaluation.cohort_registry import commit_import, prepare_generated_import

    path = site.data_dir / "derived" / "gene_fixture" / "task_fixture.json"
    record = json.loads(path.read_text())
    edit(record)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    plan = prepare_generated_import(site.registry, site.data_dir, "gen-val-2", "val")
    assert "task_fixture" not in plan.manifest
    assert plan.task_fixture_bytes is plan.task_fixture_samples_bytes is None
    assert said in plan.task_fixture_note
    version = commit_import(plan, site.registry, plan.preview_id)
    assert "task_fixture" not in version.manifest


def _edit_record(edit):
    def change(data_dir: Path) -> None:
        path = data_dir / "derived" / "gene_fixture" / "task_fixture.json"
        record = json.loads(path.read_text())
        edit(record)
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return change


def _val_file_short_of_a_row(data_dir: Path) -> None:
    path = data_dir / "derived" / "gene_fixture" / "val_samples.json"
    path.write_text(json.dumps(json.loads(path.read_text())[:-1]))


def _val_rows_one_more(record):
    record["files"]["val"]["rows"] += 1


def _train_rows_one_fewer(record):
    record["files"]["train"]["rows"] -= 1


@pytest.mark.parametrize("change, said", [
    (_val_file_short_of_a_row, "The gene fixture's val file is not the one its record names"),
    (_edit_record(_val_rows_one_more), "The gene fixture's val file holds 5 rows, and its "
                                       "record names 6"),
    (_edit_record(_train_rows_one_fewer), "The gene fixture's train file holds "),
    (_edit_record(_path_name), "is not a readable record: its val file is not named "
                               "val_samples.json with a whole row count"),
], ids=["file-rows", "record-rows", "other-split-rows", "path-name"])
def test_a_fixture_contradicting_its_record_is_refused_by_split_and_not_kept_alike(
        cli, site, tmp_path, change, said):
    """One reader of a workspace's fixture (`read_gene_fixture`): a fixture
    whose files contradict its record (rows that are not the bytes named, a
    record naming another row count, of either split, or a file by a path) is
    refused by ``measure_scorer --split val --mode gene``, writing nothing,
    and is not kept by the import of that very split, for the same reason
    (the import still refuses nothing)."""
    from src.evaluation.cohort_registry import prepare_generated_import
    from src.kg.artifacts import read_split_manifest
    from src.kg.task_fixture import TaskFixtureError, read_gene_fixture

    change(site.data_dir)
    with pytest.raises(TaskFixtureError) as read:
        read_gene_fixture(site.data_dir,
                          manifest_sha256=read_split_manifest(site.data_dir).identity.sha256)
    reason = str(read.value)
    assert said in reason and read.value.state == "unreadable"
    output = tmp_path / "gene.json"
    with pytest.raises(TaskFixtureError, match=re.escape(reason)):
        cli.main(["--mode", "gene", "--checkpoint", str(site.checkpoints["gene"]),
                  "--data-dir", str(site.data_dir), "--split", "val", "--output", str(output),
                  "--device", "cpu"])
    assert not any(p.name.startswith("gene") for p in tmp_path.iterdir())
    plan = prepare_generated_import(site.registry, site.data_dir, "gen-val-2", "val")
    assert "task_fixture" not in plan.manifest
    assert plan.task_fixture_bytes is plan.task_fixture_samples_bytes is None
    assert plan.task_fixture_note == f"unreadable: {reason}"


def test_a_plan_whose_fixture_bytes_are_not_the_ones_it_binds_is_not_committed(site):
    from src.evaluation.cohort_registry import (
        CohortVersionRefused,
        commit_import,
        prepare_generated_import,
    )

    plan = prepare_generated_import(site.registry, site.data_dir, "gen-val-2", "val")
    before = _tree(site.registry)
    for changed in (plan._replace(task_fixture_samples_bytes=b"[]"),
                    plan._replace(task_fixture_bytes=None)):
        with pytest.raises(CohortVersionRefused, match="carries a functional gene fixture its "
                                                       "manifest does not bind"):
            commit_import(changed, site.registry, plan.preview_id)
    bare = prepare_generated_import(site.registry, site.bare, "gen-val-3", "val")
    with pytest.raises(CohortVersionRefused, match="carries a functional gene fixture"):
        commit_import(bare._replace(task_fixture_bytes=plan.task_fixture_bytes),
                      site.registry, bare.preview_id)
    assert _tree(site.registry) == before


# ---------------------------------------------------------------------------
# The disease child, on a generated train and a generated val version
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("role", ROLES)
def test_a_disease_child_is_measured_on_a_generated_version_and_kept(cli, site, role, capsys):
    from src.evaluation.cohort_registry import cohort_version_block, list_report_runs
    from src.evaluation.measurement import expected_relations

    label = f"gen-{role}"
    version = _version(site, label)
    assert cli.main(_argv(site, label)) == 0
    out = capsys.readouterr().out
    (run,) = _runs(site, label)
    assert sorted(p.name for p in run.iterdir()) == ["ranks.json", "report.json"]
    report = _report(run)
    manifest = report["manifest"]
    assert (manifest["mode"], manifest["cohort_kind"], manifest["split"]) == (
        "disease", "generated", role)
    block = manifest["cohort_version"]
    assert block == cohort_version_block(version)
    assert (block["kind"], block["role"], block["label"]) == ("generated", role, label)
    assert block["generation"] == version.manifest["generation"]
    assert block["split_manifest"] == version.manifest["split_manifest"]
    assert manifest["r10"] is None
    assert manifest["relations"] == expected_relations(registered=True)
    digests = manifest["artifact_digests"]
    assert digests["samples"] == version.manifest["samples"]["sha256"]
    assert "split_manifest" not in digests and "task_fixture" not in digests
    assert manifest["n_samples"] == version.manifest["counts"]["samples"] == report["n_ranked"]
    assert report["ranks"]["sha256"] == _sha((run / "ranks.json").read_bytes())
    assert "test" not in (manifest["split"], block["role"])
    # The role as it is, on the console; never presented as test or supplied.
    assert f"A generated split ({role}): {label}, manifest sha256 " in out
    assert "R10" not in out and "Cohort version" not in out
    assert (f"Not registered (no --record): this revision's ledger does not record a run on a "
            f"generated split ({role}) yet, so no registration is offered") in out
    assert not (site.checkpoints["disease"].parent / LEDGER).exists()
    assert [listed.run_id for listed in list_report_runs(version)] == [run.name]


def test_a_version_run_scores_as_the_split_run_does_and_the_split_path_is_unchanged(
        cli, site, tmp_path):
    """The same model on the same split, by `--split val` and by `--cohort` of
    its kept version: the same evaluator gives the same figures over the same
    bytes. The `--split` run writes its report and ranks at --output only (the
    registry untouched), records the workspace's split manifest and no
    version, and stays unregistrable with today's sentence."""
    from src.evaluation.sidecar import build_record

    for task in ("disease", "gene"):
        output = tmp_path / f"{task}.json"
        before = _tree(site.registry)
        assert cli.main(["--mode", task, "--checkpoint", str(site.checkpoints[task]),
                         "--data-dir", str(site.data_dir), "--split", "val", "--output",
                         str(output), "--device", "cpu"]) == 0
        assert _tree(site.registry) == before
        assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(task)) == [
            f"{task}.json", f"{task}_ranks.json"]
        split_report = json.loads(output.read_text())
        assert cli.main(_argv(site, "gen-val", task)) == 0
        (run,) = [r for r in _runs(site, "gen-val") if _report(r)["manifest"]["mode"] == task]
        version_report = _report(run)
        split_manifest, version_manifest = split_report["manifest"], version_report["manifest"]
        assert split_manifest["cohort_version"] is None and split_manifest["split"] == "val"
        assert "split_manifest" in split_manifest["artifact_digests"]
        for field in ("authoritative_metrics", "n_ranked", "n_ground_truth_absent"):
            assert version_report[field] == split_report[field], field
        for field in ("candidate_construction", "score_semantics", "metric_schema_version",
                      "n_samples", "cohort_kind"):
            assert version_manifest[field] == split_manifest[field], field
        for role in ("samples", "checkpoint", "task_fixture"):
            assert (version_manifest["artifact_digests"].get(role)
                    == split_manifest["artifact_digests"].get(role)), role
        assert json.loads((run / "ranks.json").read_text()) == json.loads(
            (tmp_path / f"{task}_ranks.json").read_text())
        if task == "gene":
            assert (version_report["sampler_evidence"]["task_fixture"]
                    == split_report["sampler_evidence"]["task_fixture"])
        with pytest.raises(ValueError, match=re.escape(
                "a generated split's report is not registered: the ledger does not yet support "
                "a generated split's data identity (it has no registered cohort version)")):
            build_record(split_report, _sha(output.read_bytes()))


# ---------------------------------------------------------------------------
# The gene and patients-like-me children, through the fixture the version keeps
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("task", ("gene", "patients_like_me"))
def test_a_gene_and_a_plm_child_are_measured_through_the_kept_fixture(cli, site, task, role,
                                                                      capsys):
    label = f"gen-{role}"
    version = _version(site, label)
    fixture = version.manifest["task_fixture"]
    assert cli.main(_argv(site, label, task)) == 0
    out = capsys.readouterr().out
    (run,) = _runs(site, label)
    report = _report(run)
    manifest = report["manifest"]
    assert (manifest["mode"], manifest["cohort_kind"], manifest["split"]) == (
        task, "generated", role)
    assert manifest["cohort_version"]["task_fixture"] == fixture
    digests = manifest["artifact_digests"]
    assert digests["task_fixture"] == fixture["record"]["sha256"]
    assert digests["samples"] == fixture["samples"]["sha256"]
    assert manifest["n_samples"] == fixture["samples"]["n"]
    evidence = report["sampler_evidence"]["task_fixture"]
    assert (evidence["record_sha256"], evidence["split"], evidence["label"]) == (
        fixture["record"]["sha256"], role, "functional fixture")
    assert evidence["derivation"] == fixture["derivation"]
    assert sorted(p.name for p in run.iterdir()) == ["ranks.json", "report.json"]
    assert (f"scored: its functional fixture ({fixture['rule']}, seed 0), "
            f"{report['n_ranked']} ranked of {fixture['samples']['n']} rows") in out
    assert f"A generated split ({role}): {label}" in out


@pytest.mark.parametrize("task", ("gene", "patients_like_me"))
def test_gene_and_plm_on_a_version_without_the_fixture_are_refused_as_a_data_gap(cli, site,
                                                                                 task, capsys):
    """Refused before anything is read (the GPU lock is taken, nothing else),
    with the stated data gap, never for the version's role; disease on the
    same version is measured."""
    from src.evaluation import protocols

    before = _tree(site.registry)
    gap = protocols.inputs_refusal(task, "generated", _version(site, BARE).manifest["counts"])
    assert gap.startswith("this generated split version keeps no functional gene fixture")
    assert cli._run_as_program(_argv(site, BARE, task)) == 1
    assert capsys.readouterr().err.strip() == f"measure_scorer: refused: {gap} Nothing was written."
    _nothing_written(site, before)
    assert cli.main(_argv(site, BARE)) == 0
    assert len(_runs(site, BARE)) == 1


def test_a_fixture_with_no_row_of_its_split_is_a_stated_gap(site):
    from src.evaluation import protocols

    counts = {**_version(site, "gen-val").manifest["counts"], "task_fixture_samples": 0}
    for task in ("gene", "patients_like_me"):
        said = protocols.inputs_refusal(task, "generated", counts)
        assert said.startswith("this generated split version's functional gene fixture holds no "
                               f"row of its split (none of its patients' diseases has a gene in "
                               f"the graph), so the {task} measurement has no case to score")
    assert protocols.inputs_refusal("disease", "generated", counts) is None


def test_the_runner_and_the_test_page_take_a_version_that_keeps_the_fixture(site, tmp_path):
    pytest.importorskip("gradio")
    from src.api.services.training_manager import TestRunInputs
    from src.evaluation import cohort_registry as cr
    from src.evaluation import protocols
    from src.webui.components import test_page as tp
    from tests.fixtures.capabilities import described

    assert protocols.GENERATED_FIXTURE_COUNT == cr.GENERATED_FIXTURE_COUNT
    version = _version(site, "gen-val")
    counts = version.manifest["counts"]
    row = {"label": "gen-val", "kind": "generated", "counts": counts}
    for task in TASKS:
        model = {"record": "model", "learning_stage": "finetune", "task": task,
                 "capabilities": described("finetune", task)}
        assert tp.measure_refusal(model, row, task) is None
        inputs = TestRunInputs(
            registry=site.registry, version_label="gen-val",
            version_sha256=version.identity.sha256, workspace=site.data_dir,
            checkpoint=site.checkpoints[task], checkpoint_sha256="c" * 64, protocol=task,
            learning_stage="finetune", task=task, capabilities=described("finetune", task),
            version_kind="generated", version_counts=counts)
        assert inputs.reports_dir() == version.directory / "reports"


# ---------------------------------------------------------------------------
# Refusals, each before anything is written
# ---------------------------------------------------------------------------
def _rewrite(site: Site, label: str, name: str, change) -> None:
    path = site.registry / "versions" / label / name
    path.write_bytes(change(path.read_bytes()))


def _drop_row(data: bytes) -> bytes:
    return json.dumps(json.loads(data)[:-1]).encode("utf-8")


def _edit_seed(manifest):
    manifest["task_fixture"]["seed"] = 7


def _edit_count(manifest):
    manifest["samples"]["n"] -= 1
    manifest["counts"]["samples"] -= 1


def _edit_graph(manifest):
    manifest["graph"]["edge_indices"] = "e" * 64


@pytest.mark.parametrize("task, change, said", [
    ("disease", lambda s: _rewrite(s, "gen-val", "samples.json", _drop_row),
     "is not the samples"),
    ("gene", lambda s: _rewrite(s, "gen-val", "task_fixture_samples.json", _drop_row),
     "is not the fixture rows"),
    ("patients_like_me", lambda s: _rewrite(s, "gen-val", "task_fixture.json",
                                            lambda d: d.replace(b'"seed": 0', b'"seed": 3')),
     "is not the fixture record"),
    ("gene", lambda s: _edit_manifest(s, "gen-val", _edit_seed),
     "does not record the functional gene fixture it keeps: its task_fixture.seed is not the "
     "kept record's"),
    ("disease", lambda s: _edit_manifest(s, "gen-val", _edit_count),
     "the registry was altered after the import"),
    ("disease", lambda s: _edit_manifest(s, "gen-val", _edit_graph),
     "was mapped against another graph than this run's: edge_indices"),
], ids=["samples", "fixture-rows", "fixture-record", "edited-copy", "edited-count",
        "another-graph"])
def test_a_version_whose_record_contradicts_its_bytes_is_refused_writing_nothing(
        cli, site, task, change, said):
    change(site)
    before = _tree(site.registry)
    with pytest.raises(ValueError, match=re.escape(said)) as refused:
        cli.main(_argv(site, "gen-val", task))
    assert str(refused.value).endswith("Nothing was written.") or "another graph" in said
    _nothing_written(site, before)


@pytest.mark.parametrize("name, change, said", [
    ("task_fixture_samples.json", lambda data: b"[]", "is not the fixture rows"),
    ("task_fixture.json", lambda data: data.replace(b'"seed": 0', b'"seed": 3'),
     "is not the fixture record"),
], ids=["fixture-rows", "fixture-record"])
def test_a_damaged_kept_fixture_refuses_only_the_measurements_that_read_it(
        cli, site, name, change, said):
    """A refusal is scoped to the protocol that reads the damaged file: the
    gene measurement of a version whose kept fixture is not the bytes it
    binds is refused, writing nothing; the disease measurement of the same
    version, which never reads the fixture, is measured and kept."""
    _rewrite(site, "gen-val", name, change)
    before = _tree(site.registry)
    with pytest.raises(ValueError, match=re.escape(said)):
        cli.main(_argv(site, "gen-val", "gene"))
    _nothing_written(site, before)
    assert cli.main(_argv(site, "gen-val")) == 0
    (run,) = _runs(site, "gen-val")
    assert _report(run)["manifest"]["mode"] == "disease"


@pytest.mark.parametrize("argv, said", [
    (lambda s: _argv(s, "gen-val", "disease", mode="gene"),
     "the gene measurement takes a finetune model with the gene head; this checkpoint is a "
     "finetune model with the disease head."),
    (lambda s: _argv(s, "gen-val", "base", mode="kg_link_prediction"),
     "the kg_link_prediction measurement does not measure a generated split; it measures a KG "
     "edge set."),
    (lambda s: _argv(s, "gen-val", "disease", "--training-evidence", str(s.data_dir)),
     "--training-evidence applies to a supplied cohort version only (R10: its cases' overlap "
     "with the training and validation inputs the checkpoint records); 'gen-val' is a generated "
     "split (val), whose relation to the model's training is the disease allocation that cut "
     "it, which the version records."),
], ids=["model-capabilities", "data-kind", "r10-evidence"])
def test_what_the_model_or_the_data_kind_does_not_take_is_refused_writing_nothing(
        cli, site, argv, said):
    before = _tree(site.registry)
    with pytest.raises(ValueError, match=re.escape(said)):
        cli.main(argv(site))
    _nothing_written(site, before)


# ---------------------------------------------------------------------------
# Kept, never overwritten; --record says Stage 6's gap
# ---------------------------------------------------------------------------
def test_two_runs_are_both_kept_and_record_keeps_the_report_unregistered(cli, site, tmp_path,
                                                                        capsys):
    assert cli.main(_argv(site, "gen-val")) == 0
    (first,) = _runs(site, "gen-val")
    kept = _tree(first)
    outcome = tmp_path / "outcome.json"
    assert cli.main(_argv(site, "gen-val", "disease", "--record", "--outcome",
                          str(outcome))) == cli.EXIT_NOT_REGISTERED
    err = capsys.readouterr().err
    assert ("this report measured a registered version of kind 'generated', and this "
            "revision's ledger records runs on a supplied cohort version only") in err
    assert ("This revision's ledger does not record a run on a generated split (val) yet, so no "
            "registration is offered.") in err
    assert "record_evaluation.py" not in err
    runs = _runs(site, "gen-val")
    assert len(runs) == 2 and _tree(first) == kept
    (second,) = set(runs) - {first}
    assert sorted(p.name for p in second.iterdir()) == ["ranks.json", "report.json"]
    assert _report(second)["authoritative_metrics"] == _report(first)["authoritative_metrics"]
    assert not (site.checkpoints["disease"].parent / LEDGER).exists()
    written = json.loads(outcome.read_text())
    assert (written["cohort"]["label"], written["cohort"]["role"]) == ("gen-val", "val")
    assert (written["registration"]["registered"], written["registration"]["outcome"]) == (
        False, "refused")


# ---------------------------------------------------------------------------
# The reader and the references
# ---------------------------------------------------------------------------
def _fixture_without_count(manifest):
    del manifest["counts"]["task_fixture_samples"]


def _count_without_fixture(manifest):
    del manifest["task_fixture"]


def _null_fixture(manifest):
    manifest["task_fixture"] = None
    del manifest["counts"]["task_fixture_samples"]
    del manifest["count_units"]["task_fixture_samples"]


def _count_not_rows(manifest):
    manifest["counts"]["task_fixture_samples"] += 1


def _rows_named_otherwise(manifest):
    manifest["task_fixture"]["samples"]["file"] = "../val_samples.json"


@pytest.mark.parametrize("edit, said", [
    (_fixture_without_count, "counts must hold a whole number of at least 0 for exactly "
                             "['samples', 'diseases', 'task_fixture_samples']"),
    (_count_without_fixture, "counts must hold a whole number of at least 0 for exactly "
                             "['samples', 'diseases']"),
    (_null_fixture, "task_fixture must be an object"),
    (_count_not_rows, "counts.task_fixture_samples (6) is not task_fixture.samples.n (5)"),
    (_rows_named_otherwise, "task_fixture.samples must hold the file task_fixture_samples.json"),
], ids=["no-count", "no-block", "null", "count", "file-name"])
def test_a_hand_edited_fixture_block_or_count_is_refused_naming_it(site, edit, said):
    from src.evaluation.cohort_registry import CohortVersionRefused

    _edit_manifest(site, "gen-val", edit)
    with pytest.raises(CohortVersionRefused, match=re.escape(said)):
        _version(site, "gen-val")


def test_a_kept_fixture_whose_copy_or_bytes_differ_reads_as_differs(site):
    from src.evaluation import cohort_registry as cr

    _edit_manifest(site, "gen-val", _edit_seed)
    states = {s.reference.name: s for s in cr.check_references(_version(site, "gen-val"))}
    record = states["task_fixture"]
    assert record.state == cr.DIFFERS
    assert record.reason == ("the version's task_fixture.json holds the bytes recorded, but its "
                             "task_fixture.seed is not the kept record's")
    assert states["task_fixture.samples"].state == cr.VERIFIED
    _rewrite(site, "gen-val", "task_fixture_samples.json", _drop_row)
    states = {s.reference.name: s for s in cr.check_references(_version(site, "gen-val"))}
    assert states["task_fixture.samples"].state == cr.DIFFERS
    assert str(site.root) not in states["task_fixture.samples"].reason
