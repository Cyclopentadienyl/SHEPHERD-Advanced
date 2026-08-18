"""
Synthetic legacy equivalence — NOT calibration.
==============================================
The frozen evaluator and the new harness are run against the same synthetic
workspace and compared on **the only things the frozen CLI writes to disk**:

  - the aggregate legacy MRR, from its report JSON;
  - the subgraph-local top-20 predictions, from its predictions JSON.

Everything else the harness computes — global candidate ids, sampled negatives,
candidate composition, mappings, batch membership — the oracle never emits, and
this test does not pretend otherwise.

> **This is not institutional calibration and must never be reported as such.**
> It shows the two agree on a synthetic graph in this container. Mode A is
> calibrated only by the institutional run, on real data and a real checkpoint,
> on CUDA hardware.

The oracle is invoked as a subprocess exactly once, in an isolated working
directory, because it has no importable entry point and writes its predictions
file relative to the current directory. One `subprocess.run` — no launcher, no
orchestration framework.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.evaluation.measurement import LEGACY_TRUNCATION_K  # noqa: E402
from tests.fixtures.synthetic_workspace import build_workspace  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BATCH_SIZE = 3


def _run_frozen_oracle(cwd: Path, data_dir: Path, checkpoint: Path) -> tuple[dict, list]:
    """Run `scripts/evaluate_model.py` and read back what it wrote.

    `cwd=tmp_path` with absolute input and report paths, because the frozen CLI
    exposes no predictions-path option and writes `predictions_{split}.json`
    relative to the working directory. Running it in place would pollute the
    repository and collide between parallel tests.
    """
    report_path = cwd / "legacy_report.json"
    completed = subprocess.run(
        [
            sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate_model.py"),
            "--checkpoint", str(checkpoint),
            "--data-dir", str(data_dir),
            "--split", "test",
            "--output", str(report_path),
            "--batch-size", str(BATCH_SIZE),
            "--save-predictions",
        ],
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, f"frozen oracle failed:\n{completed.stderr[-2000:]}"

    predictions_path = cwd / "predictions_test.json"
    assert predictions_path.exists(), "the frozen oracle wrote no predictions file"
    return json.loads(report_path.read_text()), json.loads(predictions_path.read_text())


def _run_harness(data_dir: Path, checkpoint: Path):
    import argparse

    from scripts.measure_scorer import build_manifest, build_model, load_graph_and_samples
    from src.evaluation.measurement import run_mode_a
    from src.kg.data_loader import DataLoaderConfig, create_diagnosis_dataloader

    device = torch.device("cpu")
    graph_data, samples = load_graph_and_samples(data_dir, "test")
    loader = create_diagnosis_dataloader(
        samples=samples, graph_data=graph_data,
        config=DataLoaderConfig(batch_size=BATCH_SIZE, num_workers=0, shuffle=False),
    )
    args = argparse.Namespace(checkpoint=checkpoint, data_dir=data_dir, split="test",
                              batch_size=BATCH_SIZE, num_workers=0, seed=None)
    return run_mode_a(
        model=build_model(checkpoint, device), dataloader=loader,
        manifest=build_manifest(args, graph_data, len(samples), device), device=device,
    )


@pytest.fixture(scope="module")
def both_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("equivalence")
    data_dir, checkpoint = build_workspace(root)
    oracle_cwd = root / "oracle_cwd"
    oracle_cwd.mkdir()

    report, predictions = _run_frozen_oracle(oracle_cwd, data_dir, checkpoint)
    return report, predictions, _run_harness(data_dir, checkpoint)


def test_aggregate_legacy_mrr_matches_the_oracle(both_runs):
    """The oracle's `mrr` is computed over its top-20 truncated lists, so it is
    the legacy truncated metric under a different name."""
    report, _, result = both_runs

    assert result.legacy_metrics[f"legacy_mrr_truncated_at_{LEGACY_TRUNCATION_K}"] == pytest.approx(
        report["metrics"]["mrr"], abs=1e-12
    )


def test_local_top_k_predictions_match_the_oracle(both_runs):
    """Compared in local space, because that is the only space the oracle's
    artifact exists in — it writes column indices and not the mapping."""
    _, predictions, result = both_runs

    oracle_rows = [[int(p) for p in row["predictions"]] for row in predictions]

    assert len(oracle_rows) == len(result.legacy_top_k_local)
    for oracle_row, harness_row in zip(oracle_rows, result.legacy_top_k_local):
        assert harness_row[: len(oracle_row)] == oracle_row


def test_the_untruncated_metric_is_not_claimed_to_match(both_runs):
    """They are different quantities and must not be conflated. Here the
    candidate set is smaller than 20 so they coincide numerically — which is
    exactly why the assertion is about the report's shape, not its value: the
    oracle has no untruncated metric to compare against at all."""
    report, _, result = both_runs

    assert "untruncated_mrr" in result.authoritative_metrics
    assert "untruncated_mrr" not in report["metrics"]
    assert report["metrics"]["mean_rank"] is None
    assert result.authoritative_metrics["mean_rank"] is not None


def test_this_run_is_not_calibration(both_runs):
    """A guard against the claim rather than the code. Nothing here touches real
    data, a real checkpoint or CUDA, so nothing here clears the acceptance gate.
    """
    _, _, result = both_runs

    assert result.manifest.device == "cpu"
    assert result.manifest.n_samples < 100
    assert "synthetic" not in result.manifest.score_semantics  # the manifest describes the scorer
