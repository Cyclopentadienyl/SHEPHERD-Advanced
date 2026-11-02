"""Task measurement of a finetuned disease bundle through the measurement entry
(`scripts/measure_scorer.py --mode disease`; PLAN_TEST_RESULTS §8, README
E1–E6, provenance §9.3 hazards 2 and 3).

A base and a disease child are trained on the demo workspace; the child is
measured on a generated split and on a registered cohort version, the latter
registered in the ledger beside the checkpoint under the task's own scoring
semantics. A base, and a checkpoint without a model spec, are refused before
anything is written. The ranks it writes keep the ranks rule
(`check_ranks_rows`, beside the writer): one row per case of the cohort, in its
order, an unrankable case kept with a null rank; the rule refuses rows that
repeat, drop, replace or move a case.

Module: tests/unit/test_disease_measurement.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from tests.fixtures.registered_cohort import (  # noqa: E402
    LABEL,
    graph_source_cases,
    import_version,
    write_demo_workspace,
    write_fixture_checkpoint,
    write_source,
)
from tests.fixtures.trained_children import train_children  # noqa: E402


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("measure")
    data_dir = write_demo_workspace(root / "workspace")
    registry = root / "registry"
    registry.mkdir()
    import_version(registry, data_dir, write_source(root / "cases.json",
                                                     graph_source_cases(data_dir)))
    trained = train_children(data_dir, ("disease",), runs=root)
    return {"root": root, "data_dir": data_dir, "registry": registry,
            "base": trained["base"], "child": trained["disease"]}


def test_generated_split_report(trained, tmp_path):
    import measure_scorer

    out = tmp_path / "report.json"
    code = measure_scorer.main(["--mode", "disease", "--checkpoint", str(trained["child"]),
                                "--data-dir", str(trained["data_dir"]), "--split", "val",
                                "--output", str(out), "--device", "cpu"])
    assert code == 0
    report = json.loads(out.read_text())
    manifest = report["manifest"]
    assert manifest["mode"] == "disease" and manifest["metric_schema_version"] == 2
    assert manifest["canonical_tie_policy_version"] == "average-rank/v1"
    assert manifest["model_construction"].startswith("model bundle")
    metrics = report["authoritative_metrics"]
    assert 0 < metrics["pooled_mrr"] <= 1 and "macro_recall@10" in metrics
    assert metrics["coverage"] <= 1
    ranks = json.loads((tmp_path / "report_ranks.json").read_text())
    assert len(ranks) == manifest["n_samples"]
    # The written file keeps the ranks rule against the split it measured.
    from src.evaluation.measurement import check_ranks_rows
    from src.kg.storage.file_storage import read_samples

    cohort = [s.patient_id for s in read_samples(trained["data_dir"], "val").samples]
    ranked = check_ranks_rows(ranks, cohort)
    assert len(ranked) == report["n_ranked"]


def test_registered_cohort_is_recorded_under_task_semantics(trained, monkeypatch):
    import measure_scorer
    from src.evaluation.sidecar import read_ledger

    monkeypatch.setenv("SHEPHERD_REGISTRY", str(trained["registry"]))
    code = measure_scorer.main(["--mode", "disease", "--checkpoint", str(trained["child"]),
                                "--data-dir", str(trained["data_dir"]), "--cohort", LABEL,
                                "--registry", str(trained["registry"]), "--device", "cpu",
                                "--record"])
    assert code == 0
    ledger, _ = read_ledger(trained["child"].parent / "evaluations.json")
    (record,) = ledger["records"]
    assert record["scoring"]["mode"] == "disease"
    assert record["scoring"]["candidate_construction"] == "every disease of the model's graph view"
    assert "pooled_mrr" in record["metrics"]


def test_a_base_is_refused_before_anything_is_written(trained, tmp_path):
    import measure_scorer

    out = tmp_path / "base.json"
    with pytest.raises(ValueError, match="not a finetune"):
        measure_scorer.main(["--mode", "disease", "--checkpoint", str(trained["base"]),
                             "--data-dir", str(trained["data_dir"]), "--split", "val",
                             "--output", str(out), "--device", "cpu"])
    assert not out.exists()


def test_a_checkpoint_without_a_spec_is_history(trained, tmp_path):
    import measure_scorer

    old = tmp_path / "old.pt"
    write_fixture_checkpoint(trained["data_dir"], old)
    out = tmp_path / "old.json"
    with pytest.raises(ValueError, match="written before the model spec"):
        measure_scorer.main(["--mode", "disease", "--checkpoint", str(old),
                             "--data-dir", str(trained["data_dir"]), "--split", "val",
                             "--output", str(out), "--device", "cpu"])
    assert not out.exists()


# ---------------------------------------------------------------- the ranks rule
COHORT = ["case-a", "case-b", "case-c", "case-d"]


def _rows(*ids, ranks=None):
    ranks = ranks or [1.0, None, 3.5, 2.0][:len(ids)]
    return [{"sample_id": i, "ground_truth": 7, "rank": r} for i, r in zip(ids, ranks)]


def test_rows_one_per_case_in_order_pass_and_name_the_ranked_cases():
    """A normal report: every case once, in the cohort's order, an unrankable
    case (case-b) kept with a null rank. The ranked cases come back in order."""
    from src.evaluation.measurement import check_ranks_rows

    assert check_ranks_rows(_rows(*COHORT), COHORT) == ["case-a", "case-c", "case-d"]


@pytest.mark.parametrize("ids, problem, position", [
    # The reviewer's P2-2 case: a case's id replaced by its neighbour's, the
    # row count kept.
    (["case-a", "case-a", "case-c", "case-d"], "duplicate", 1),
    (["case-a", "case-c", "case-d"], "missing", 1),
    (["case-a", "case-b", "case-x", "case-d"], "foreign", 2),
    (["case-b", "case-a", "case-c", "case-d"], "misordered", 0),
], ids=["duplicate", "missing", "foreign", "misordered"])
def test_rows_that_repeat_drop_replace_or_move_a_case_are_refused(ids, problem, position):
    from src.evaluation.measurement import RanksRowsRefused, check_ranks_rows

    with pytest.raises(RanksRowsRefused) as refused:
        check_ranks_rows(_rows(*ids), COHORT)
    assert (refused.value.problem, refused.value.position) == (problem, position)


@pytest.mark.parametrize("rows, position", [
    ({"rows": []}, 0),
    ([{"sample_id": "case-a"}], 0),                                  # no rank
    ([{"sample_id": "case-a", "rank": 0}], 0),                       # not a rank
    ([{"sample_id": "case-a", "rank": True}], 0),                    # a boolean
    ([{"sample_id": "case-a", "rank": 1}, ["case-b", None]], 1),     # not a row
], ids=["not-a-list", "no-rank", "rank-zero", "rank-boolean", "not-a-row"])
def test_rows_without_a_sample_id_and_a_rank_are_refused(rows, position):
    from src.evaluation.measurement import RanksRowsRefused, check_ranks_rows

    with pytest.raises(RanksRowsRefused) as refused:
        check_ranks_rows(rows, ["case-a", "case-b"])
    assert refused.value.problem in ("not_a_list", "malformed")
    assert refused.value.position == position


def test_the_writer_holds_what_it_writes_to_the_rule():
    """`ModeResult.to_ranks` refuses rows its own fields would write short: a
    rank list shorter than the cohort used to be cut silently by `zip`."""
    from src.evaluation.measurement import ModeResult

    result = ModeResult(manifest=None, authoritative_metrics={}, n_ranked=1,
                        n_ground_truth_absent=0, sampler_evidence={},
                        sample_ids=["case-a", "case-b"], truth_global_ids=[1, 2],
                        canonical_ranks=[1.0])
    # `RanksRowsRefused` is a ValueError, as every refusal of a row is.
    with pytest.raises(ValueError, match="case 1 of the cohort, 'case-b', has no ranks row"):
        result.to_ranks()
