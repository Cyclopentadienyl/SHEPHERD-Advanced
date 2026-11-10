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
    write_checkpoint_without_model_spec,
    write_demo_workspace,
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


def test_generated_split_report(trained, tmp_path, monkeypatch):
    import measure_scorer

    from src.evaluation.measurement import expected_relations
    from src.utils import run_environment

    # The environment comes from the one collector, for this run's device and this
    # repository; the code record is pinned so the revision check holds without .git.
    revision = {"state": "recorded", "revision": "0" * 39 + "1", "dirty": False,
                "modified_tracked": 0, "untracked": 0}
    facts = run_environment.collect_environment("cpu")
    roots, devices = [], []
    monkeypatch.setattr(run_environment, "code_revision",
                        lambda root: roots.append(root) or dict(revision))
    monkeypatch.setattr(run_environment, "collect_environment",
                        lambda device: devices.append(device) or facts)
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
    assert (roots, devices) == ([measure_scorer.PROJECT_ROOT], ["cpu"])
    assert manifest["environment"] == {"facts": facts, "code": revision}
    assert manifest["software_revision"] == manifest["environment"]["code"]["revision"]
    # Observed: no autocast is open, and the bundle that ran is no compile wrapper
    # (None would mean not observed).
    assert (manifest["amp_enabled"], manifest["amp_dtype"]) == (False, None)
    assert manifest["torch_compile_wrapped"] is False
    assert manifest["torch_version"] == str(torch.__version__) and manifest["graph_fingerprint"]
    assert manifest["relations"] == expected_relations(registered=False)
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
    # Observed, not a literal: run under autocast, the report says so.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert measure_scorer.main(["--mode", "disease", "--checkpoint", str(trained["child"]),
                                    "--data-dir", str(trained["data_dir"]), "--split", "val",
                                    "--output", str(tmp_path / "bf16.json"), "--device", "cpu"]) == 0
    bf16 = json.loads((tmp_path / "bf16.json").read_text())["manifest"]
    assert (bf16["amp_enabled"], bf16["amp_dtype"]) == (True, "torch.bfloat16")


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
    import re

    import measure_scorer

    out = tmp_path / "base.json"
    said = ("the disease measurement takes a finetune model with the disease head; this "
            "checkpoint is a base model with no task head. It can be measured by "
            "kg_link_prediction.")
    with pytest.raises(ValueError, match=re.escape(said)):
        measure_scorer.main(["--mode", "disease", "--checkpoint", str(trained["base"]),
                             "--data-dir", str(trained["data_dir"]), "--split", "val",
                             "--output", str(out), "--device", "cpu"])
    assert not out.exists()


def test_a_checkpoint_without_a_spec_is_history(trained, tmp_path):
    import measure_scorer

    old = tmp_path / "old.pt"
    write_checkpoint_without_model_spec(old, data_dir=trained["data_dir"])
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


def _result(sample_ids, truths, ranks):
    from src.evaluation.measurement import ModeResult

    return ModeResult(manifest=None, authoritative_metrics={}, n_ranked=1,
                      n_ground_truth_absent=0, sampler_evidence={}, sample_ids=sample_ids,
                      truth_global_ids=truths, canonical_ranks=ranks)


@pytest.mark.parametrize("truths, ranks, lengths", [
    ([1, 2], [1.0], "sample ids 2, ground truths 2, ranks 1"),
    ([1, 2], [1.0, None, 3.0], "sample ids 2, ground truths 2, ranks 3"),
    ([1, 2, 3], [1.0, None], "sample ids 2, ground truths 3, ranks 2"),
    ([1], [1.0, None], "sample ids 2, ground truths 1, ranks 2"),
], ids=["ranks-short", "ranks-long", "truths-long", "truths-short"])
def test_the_writer_refuses_lists_of_other_lengths(truths, ranks, lengths):
    """`ModeResult.to_ranks` builds no row unless every case has exactly one
    ground truth and one rank. The regression: a rank or truth list longer
    than the cohort was cut silently by `zip` and the rows passed the rule
    (65cb3dd caught only the shorter rank list, through the rule)."""
    with pytest.raises(ValueError, match=rf"the result's lists differ in length \({lengths}\); "
                                         "each case has exactly one of each"):
        _result(["case-a", "case-b"], truths, ranks).to_ranks()


def test_the_writer_holds_what_it_writes_to_the_rule():
    """Lists of one length are written one row per case, an unrankable case
    with a null rank, and held to the rule."""
    rows = _result(["case-a", "case-b"], [1, 2], [1.0, None]).to_ranks()
    assert rows == [{"sample_id": "case-a", "ground_truth": 1, "rank": 1.0},
                    {"sample_id": "case-b", "ground_truth": 2, "rank": None}]
