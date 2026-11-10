"""
The KG link-prediction measurement, run end to end as an operator runs it.
==========================================================================
Research comparison §5 Stage 5. On the session's demo base and its disease
child (`trained_disease_child`, copied: the session's files stay read-only), a
KG edge set of the base's test part is imported through the importer's command
line (preview, then the printed commit), and `scripts/measure_scorer.py --mode
kg_link_prediction --cohort kg-test --registry DIR` is run as a subprocess on
the base and on the child. The assertions read the files the runs wrote: each
run's new directory in the version's report store holds its scores file and
its report, the report names the scores by digest and rows and carries its
exposure (held out for the base, after full-view exposure for the child), and
the console says the exposure's reading.

> **Nothing here is an acceptance result, and nothing here may be reported as
> one.** It shows the commands run and keep complete, self-describing outputs
> on a synthetic graph in this container, on CPU, with checkpoints trained for
> one epoch.

Module: tests/integration/test_kg_link_prediction_cli.py
"""
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[2]
TIMEOUT = 900
LABEL = "kg-test"


def _run(*argv: str, env=None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *argv], cwd=str(REPO_ROOT), capture_output=True,
                          text=True, timeout=TIMEOUT, env={**os.environ, **(env or {})})


@pytest.fixture(scope="module")
def measured(trained_disease_child, tmp_path_factory):
    """A KG edge set imported by the command line from the copied base, then
    measured by the command line on the base and on the disease child."""
    root = (tmp_path_factory.mktemp("kg_lp_cli") / "root").resolve()
    data_dir = Path(shutil.copytree(trained_disease_child["data_dir"], root / "workspace"))
    checkpoints = {}
    for name in ("base", "disease"):
        checkpoints[name] = root / name / "last.pt"
        checkpoints[name].parent.mkdir(parents=True)
        shutil.copyfile(trained_disease_child[name], checkpoints[name])
    registry = root / "registry"
    registry.mkdir()

    preview = _run("scripts/import_cohort.py", "--kind", "kg_edges", "--registry",
                   str(registry), "--data-dir", str(data_dir), "--from-checkpoint",
                   str(checkpoints["base"]), "--label", LABEL, "--role", "test")
    assert preview.returncode == 0, preview.stderr[-4000:]
    command = shlex.split(preview.stdout.strip().splitlines()[-1].strip())
    assert command[:2] == ["python", "scripts/import_cohort.py"]
    commit = _run(*command[1:])
    assert commit.returncode == 0, commit.stderr[-4000:]

    from src.evaluation.cohort_registry import is_run_id

    runs = {}
    reports = registry / "versions" / LABEL / "reports"

    def run_dirs():
        return {p for p in reports.iterdir() if is_run_id(p.name)} if reports.is_dir() else set()

    for name in ("base", "disease"):
        before = run_dirs()
        done = _run("scripts/measure_scorer.py", "--mode", "kg_link_prediction",
                    "--checkpoint", str(checkpoints[name]), "--data-dir", str(data_dir),
                    "--cohort", LABEL, "--registry", str(registry), "--batch-size", "3",
                    "--device", "cpu", env={"SHEPHERD_REGISTRY": str(registry)})
        assert done.returncode == 0, done.stderr[-4000:]
        (run,) = run_dirs() - before
        runs[name] = {"run": run, "stdout": done.stdout, "checkpoint": checkpoints[name],
                      "report": json.loads((run / "report.json").read_text())}
    return {"registry": registry, "runs": runs}


def test_each_run_keeps_its_scores_and_its_report_in_a_new_run_directory(measured):
    from src.evaluation.cohort_registry import (
        is_run_id,
        read_kg_edge_pairs,
        resolve_version,
    )
    from src.models.bundle.lp_head import lp_scores_from_table

    version = resolve_version(measured["registry"], LABEL)
    pairs = read_kg_edge_pairs(version).pairs
    reports = version.directory / "reports"
    names = sorted(p.name for p in reports.iterdir())
    assert sorted(n for n in names if is_run_id(n)) == sorted(
        r["run"].name for r in measured["runs"].values())
    for run in measured["runs"].values():
        directory, report = run["run"], run["report"]
        assert sorted(p.name for p in directory.iterdir()) == ["report.json", "scores.bin"]
        assert directory.name.split("-")[1] == hashlib.sha256(
            run["checkpoint"].read_bytes()).hexdigest()[:12]
        data = (directory / "scores.bin").read_bytes()
        assert hashlib.sha256(data).hexdigest() == report["scores"]["sha256"]
        table = lp_scores_from_table(data, report["scores"]["layout"])
        assert len(table) == report["scores"]["rows"] == int(pairs.head.numel())
        assert report["scores"]["keyed_to"]["lp_pairs_digest"] == pairs.digest
        assert report["manifest"]["mode"] == "kg_link_prediction"
        assert report["manifest"]["cohort_kind"] == "kg_edges"
        assert report["manifest"]["batch_size"] == 3
        assert report["n_ranked"] == int(pairs.head.numel())


def test_the_exposure_is_kept_with_each_result_and_said_on_the_console(measured):
    runs = measured["runs"]
    base, child = runs["base"]["report"]["exposure"], runs["disease"]["report"]["exposure"]
    assert (base["state"], base["reading"]) == (
        "held_out", "held out relative to the recorded training and validation graph use")
    assert (child["state"], child["reading"]) == ("after_full_view_exposure",
                                                  "after full-view exposure")
    for run, exposure in ((runs["base"], base), (runs["disease"], child)):
        out = run["stdout"]
        assert f"Exposure ({exposure['definition_version']}): {exposure['reading']}" in out
        assert re.search(r"^  lp_roc_auc\s+\d", out, re.MULTILINE), out
        # No register-later command: this revision's ledger does not record a
        # KG edge set's run yet, and the console says so instead.
        assert ("Not registered (no --record): this revision's ledger does not record a run "
                "on a KG edge set (test) yet") in out
        assert "record_evaluation.py" not in out
