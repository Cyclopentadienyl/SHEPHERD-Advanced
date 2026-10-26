"""
Mode D: the served ranking, measured (`src/evaluation/mode_d.py`).
=================================================================
`docs/working/scorer-measurement/PLAN_B05.md` is the design. These tests are
Mode D's own, in one file, so retiring it later removes one module, one CLI
option and this file (§10).

They pin, in order:

- **the case sets** (§4): figures over the import-eligible cases are named
  post-import, and no figure claims to be end to end (the reviewer's
  counterexample: 100 source cases, 20 excluded at import, 80 ranked first);
- **one Mode C** (§5): the C side is `run_mode_c`'s own result on the same
  embeddings, and Mode D's module calls no scoring primitive itself;
- **the derived steps change one thing at a time** (§5): the two [9, 2] tie
  cases, the second with SP breaking a cosine tie;
- **the run through the real entry point** (§6): the report's blocks, the
  manifest's Mode D values with no field added, the record keyed and grouped
  apart from Mode C's and never ordered, the usage history's mode, each
  surface's label by the record's own mode (Test page, Models page, usage
  history), and the refusals before any case is ranked.

The integration tests use the registered-cohort fixture (CPU, deterministic,
an untrained model, so no figure here is evidence about the scorer).

Module: tests/unit/test_mode_d.py
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from src.evaluation.mode_d import (  # noqa: E402
    COMPARISON_STATEMENT,
    END_TO_END_ABSENT,
    METRIC_SCHEMA_VERSION,
    ModeDRefused,
    ServedCase,
    derived_ranks,
    import_counts_block,
    paired,
    post_import_metrics,
    reached_metrics,
    served_input,
    served_semantics,
)

REPO = Path(__file__).resolve().parents[2]


def _identity(mode="gnn_only", eta=1.0, build_id=None, device="cpu"):
    return {
        "scoring_mode": mode,
        "shortest_paths": {"state": "verified" if build_id else "absent",
                           "build_id": build_id, "max_hops": 5 if build_id else None},
        "settings": {"device": device, "device_requested": device, "eta": eta,
                     "max_path_length": 4, "path_length_penalty": 0.9,
                     "aggregation_method": "weighted_sum", "max_paths_per_candidate": 10,
                     "use_direct_path_optimization": True,
                     "phenotype_normalisation_version": 1},
    }


# ---------------------------------------------------------------------------
# What kind of scoring a Mode D record is
# ---------------------------------------------------------------------------
def test_semantics_name_the_served_scoring_and_refuse_the_fallback():
    """The scoring mode, η and the table's build id are in `score_semantics`; the
    discovery settings are in `candidate_construction` by digest, the device
    left out (a regime field). The fallback ranks without a model and is
    refused."""
    plain = served_semantics(_identity())
    mixed = served_semantics(_identity("gnn_plus_shortest_path", 0.7, "b" * 32))
    assert plain["mode"] == "D" and plain["metric_schema_version"] == METRIC_SCHEMA_VERSION == 2
    assert isinstance(plain["metric_schema_version"], int)
    assert plain["score_semantics"] == "served:gnn_only:eta=1.0:sp=none"
    assert mixed["score_semantics"] == f"served:gnn_plus_shortest_path:eta=0.7:sp={'b' * 32}"
    assert plain["batch_size"] == 1
    assert plain["canonical_tie_policy_version"] == "served-stable-discovery-order/v1"
    on_gpu = served_semantics(_identity(device="cuda:0"))
    assert on_gpu["candidate_construction"] == plain["candidate_construction"]
    other_settings = _identity()
    other_settings["settings"]["max_path_length"] = 3
    assert served_semantics(other_settings)["candidate_construction"] != \
        plain["candidate_construction"]
    with pytest.raises(ModeDRefused, match="path_reasoning_fallback"):
        served_semantics(_identity("path_reasoning_fallback", None))


# ---------------------------------------------------------------------------
# The case sets (P2-1)
# ---------------------------------------------------------------------------
def test_figures_over_the_import_eligible_cases_are_post_import_never_end_to_end():
    """The reviewer's counterexample: 100 source cases, 20 excluded at import as
    phenotype_unmappable, the other 80 ranked first. `post_import_mrr` is 1.0
    and says so; the import counts are beside it; no figure is named end to end,
    and the report says why."""
    metrics = post_import_metrics([1] * 80)
    assert metrics["post_import_mrr"] == 1.0 and metrics["coverage"] == 1.0
    counts = import_counts_block({"counts": {
        "source_cases": 100, "ranked_eligible": 80, "n_excluded": 20,
        "excluded": {"ground_truth_absent": 0, "phenotype_unmappable": 20}}})
    assert counts["source_cases"] == 100 and counts["n_excluded"] == 20
    assert counts["end_to_end"] == END_TO_END_ABSENT
    assert not any("end_to_end" in name or "end-to-end" in name for name in metrics)


def test_a_case_not_reached_or_refused_counts_zero_after_import_and_is_left_out_among_reached():
    ranks = [1, 2, None, 4, None]
    after = post_import_metrics(ranks)
    assert after["post_import_mrr"] == pytest.approx((1 + 1 / 2 + 1 / 4) / 5)
    assert after["post_import_hits@1"] == pytest.approx(1 / 5)
    assert after["coverage"] == pytest.approx(3 / 5)
    among = reached_metrics(ranks)
    assert among["reached_mrr"] == pytest.approx((1 + 1 / 2 + 1 / 4) / 3)
    assert among["reached_mean_rank"] == pytest.approx(7 / 3)
    assert reached_metrics([None, None]) == {}
    with pytest.raises(ModeDRefused):
        post_import_metrics([])


# ---------------------------------------------------------------------------
# The derived steps (P2-3)
# ---------------------------------------------------------------------------
def _case(served_ids, ordinals, served_rank, truth=9):
    return ServedCase("s", truth, "ranked" if served_rank else "not_reached", False,
                      served_rank, tuple(served_ids), tuple(ordinals), (0.0,) * len(served_ids))


def test_a_cosine_tie_kept_in_discovery_order_is_the_tie_step():
    """`gnn_only`, discovery [9, 2], cosine 0.5 each: both serve 0.75, so D keeps
    discovery order. G1 [2, 9] (lower id), G2 [9, 2] (discovery): the truth 9
    ranks 2, 1, 1; the whole change is the tie step's."""
    case = _case(served_ids=(9, 2), ordinals=(0, 1), served_rank=1)
    g1, g2 = derived_ranks(case, [0.5, 0.5])
    assert (g1, g2, case.served_rank) == (2, 1, 1)


def test_g2_takes_its_ties_from_the_discovery_ordinal_not_the_served_order():
    """The reviewer's counterexample: discovery [9, 2], cosine 0.5 each, SP 0.1
    for 9 and 0.9 for 2 at η 0.7, so D serves [2, 9]. G1 [2, 9], G2 [9, 2]: the
    truth 9 ranks 2, 1, 2. A G2 that re-sorted the served order would read
    [2, 9] and blame the score's effect on the tie step."""
    served_scores = {9: 0.7 * 0.75 + 0.3 * 0.1, 2: 0.7 * 0.75 + 0.3 * 0.9}
    assert served_scores[2] > served_scores[9]
    case = _case(served_ids=(2, 9), ordinals=(1, 0), served_rank=2)
    g1, g2 = derived_ranks(case, [0.5, 0.5])
    assert (g1, g2, case.served_rank) == (2, 1, 2)


def test_paired_counts_say_how_each_rank_moved():
    moved = paired([1, 2, 3, None, 5], [2, 2, 1, 4, None])
    assert (moved["better"], moved["equal"], moved["worse"]) == (1, 1, 1)
    assert moved["ranked_then_not"] == 1 and moved["n_both_ranked"] == 3
    assert moved["rank_difference"]["min"] == -2 and moved["rank_difference"]["max"] == 1
    # A refused case lost after C is counted apart from one the candidates lost.
    split = paired([1, 2, 3], [None, None, 3], refused=[True, False, False])
    assert (split["ranked_then_refused"], split["ranked_then_not"]) == (1, 1)


def test_rank_differences_are_nearest_rank_quantiles_and_say_their_sign():
    """The internal review's mode_d-6: the quantiles are nearest rank, as named
    (the even-count median never flips between lower and upper), and the block
    says what the difference is."""
    from src.evaluation.mode_d import RANK_DIFFERENCE, _quantiles

    assert _quantiles([-1, 0])["median"] == -1
    assert _quantiles([0, 0, 1, 1])["median"] == 0
    assert _quantiles([1, 2, 3, 4]) == {"min": 1, "q25": 1, "median": 2, "q75": 3, "max": 4}
    assert _quantiles([]) is None
    assert paired([3], [1])["rank_difference_is"] == RANK_DIFFERENCE
    assert "negative means the truth moved up" in RANK_DIFFERENCE


def test_a_sample_becomes_the_served_input_through_the_graphs_own_mapping():
    """Rows to HPO terms by the reverse mapping; a row the graph does not name,
    or one that is not an HPO term, refuses."""
    names = {0: "hpo:HP:0001250", 1: "hpo:HP:0001263", 2: "mondo:MONDO:0000001"}
    sample = SimpleNamespace(patient_id="c1", phenotype_ids=[1, 0], disease_id=0)
    assert served_input(sample, names).phenotypes == ["HP:0001263", "HP:0001250"]
    with pytest.raises(ModeDRefused, match="does not have"):
        served_input(SimpleNamespace(patient_id="c2", phenotype_ids=[7]), names)
    with pytest.raises(ModeDRefused, match="not an HPO term"):
        served_input(SimpleNamespace(patient_id="c3", phenotype_ids=[2]), names)


def test_each_kind_of_case_is_counted_as_designed():
    """§4's outcomes from `rank()`'s: a refused input is `refused` (never not
    reached), an empty candidate list `not_reached` with `no_candidates`, a truth
    among the candidates `ranked` at its served position, one outside them
    `not_reached`."""
    from src.core.types import PhenotypeInputAccount
    from src.evaluation.mode_d import rank_served_cases
    from src.inference.pipeline import ServedRanking

    diseases = {"mondo:MONDO:000000%d" % i: i for i in range(4)}
    # What the served validation used: the one term served_input gives.
    used = PhenotypeInputAccount(received=1, unknown=(), repeats_removed=0,
                                 used=("HP:0001250",))
    served = {
        "refused": ServedRanking("refused", ["over the input limit"], None),
        "empty": ServedRanking("no_candidates", ["no candidates"], None, used),
        "reached": ServedRanking("ranked", [], None, used, candidates=[
            SimpleNamespace(disease_id="mondo:MONDO:0000003", confidence_score=0.9),
            SimpleNamespace(disease_id="mondo:MONDO:0000001", confidence_score=0.8)],
            discovery_ordinals=[1, 0]),
        "missed": ServedRanking("ranked", [], None, used, candidates=[
            SimpleNamespace(disease_id="mondo:MONDO:0000002", confidence_score=0.5)],
            discovery_ordinals=[0]),
    }
    kg = SimpleNamespace(get_node_id_mapping=lambda: {"disease": diseases},
                         get_reverse_node_mapping=lambda: {"phenotype": {0: "hpo:HP:0001250"}})
    pipeline = SimpleNamespace(kg=kg, rank=lambda patient: served[patient.patient_id])
    samples = [SimpleNamespace(patient_id=name, phenotype_ids=[0], disease_id=1)
               for name in served]

    cases = rank_served_cases(pipeline, samples)

    assert [(c.outcome, c.no_candidates, c.served_rank) for c in cases] == [
        ("refused", False, None), ("not_reached", True, None), ("ranked", False, 2),
        ("not_reached", False, None)]
    assert cases[2].served_ids == (3, 1) and cases[2].discovery_ordinals == (1, 0)


# ---------------------------------------------------------------------------
# One Mode C (P2-2)
# ---------------------------------------------------------------------------
def test_mode_d_calls_no_scoring_primitive_itself():
    """Mode D's module imports none of the primitives a second evaluator would
    compose (pool, cosine, canonical ranking): its C side is `run_mode_c`."""
    tree = ast.parse((REPO / "src" / "evaluation" / "mode_d.py").read_text())
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                for alias in node.names}
    assert "run_mode_c" in imported
    assert not imported & {"masked_mean_pool", "cosine_score_matrix", "cosine_scores",
                           "canonical_ranking", "ranks_of_truth", "pool_patient_embeddings"}


# ---------------------------------------------------------------------------
# Through the real entry point
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def base(tmp_path_factory):
    from tests.fixtures.registered_cohort import build_registered_cohort

    return build_registered_cohort(tmp_path_factory.mktemp("mode_d_base"))


@pytest.fixture
def cohort(base, tmp_path, monkeypatch):
    from tests.fixtures.registered_cohort import copy_registered_cohort

    copied = copy_registered_cohort(base, tmp_path / "root")
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(copied.registry))
    return copied


def _argv(cohort, *extra, data_dir=None, checkpoint=None):
    return ["--checkpoint", str(checkpoint or cohort.checkpoint),
            "--data-dir", str(data_dir or cohort.data_dir),
            "--cohort", cohort.version.manifest["label"], "--registry", str(cohort.registry),
            "--device", "cpu", *extra]


def _reports(cohort):
    found = {}
    for path in sorted((cohort.version.directory / "reports").glob("*/report.json")):
        report = json.loads(path.read_text())
        found.setdefault(report["manifest"]["mode"], []).append((path, report))
    return found


def test_a_mode_d_run_keeps_one_report_with_its_blocks_and_registers_apart(cohort, tmp_path):
    """The report has every measurement report's keys and Mode D's blocks; the
    manifest has exactly Mode C's field set with Mode D's values; outcomes add up
    over the import-eligible cases; the ranks file has a row per case; the record
    is keyed and grouped apart from Mode C's; the usage history says D."""
    import scripts.measure_scorer as cli
    from src.evaluation.report_store import entry_mode, read_usage_history
    from src.evaluation.sidecar import build_record

    outcome = tmp_path / "outcome.json"
    assert cli.main(_argv(cohort)) == 0
    assert cli.main(["--mode", "D", *_argv(cohort, "--record", "--outcome", str(outcome))]) == 0
    assert json.loads(outcome.read_text())["mode"] == "D"          # cli-6
    reports = _reports(cohort)
    (c_path, c_report), = reports["C"]
    (d_path, d_report), = reports["D"]

    assert set(d_report) == {"manifest", "authoritative_metrics", "n_ranked", "outcomes",
                             "import_counts", "served", "comparison", "ranks"}
    assert set(d_report["manifest"]) == set(c_report["manifest"])
    manifest = d_report["manifest"]
    assert manifest["mode"] == "D" and manifest["metric_schema_version"] == 2
    assert manifest["batch_size"] == 1
    assert manifest["score_semantics"].startswith("served:gnn_only:eta=1.0:sp=none")
    counts = cohort.version.manifest["counts"]
    outcomes = d_report["outcomes"]
    assert outcomes["n_import_eligible"] == counts["ranked_eligible"]
    assert outcomes["ranked"] + outcomes["not_reached"] + outcomes["refused"] == \
        counts["ranked_eligible"]
    assert d_report["n_ranked"] == outcomes["ranked"]
    assert d_report["import_counts"]["source_cases"] == counts["source_cases"]
    assert d_report["comparison"]["statement"] == COMPARISON_STATEMENT
    assert d_report["comparison"]["mode_c"]["authoritative_metrics"] == \
        c_report["authoritative_metrics"]
    steps = d_report["comparison"]["steps"]
    assert "joint difference" in steps["what_changes"]["C->D"]        # mode_d-5
    assert "batch size" in steps["ties_are"]                          # mode_d-7
    assert set(steps["d_refused_while_c_ranked"]) == {"top_10", "top_100"}
    assert all("rank_difference_is" in block for block in steps["paired"].values())
    rows = json.loads((d_path.parent / "ranks.json").read_text())
    assert len(rows) == counts["ranked_eligible"]
    assert {row["outcome"] for row in rows} <= {"ranked", "not_reached", "refused"}

    c_record = build_record(c_report, "c" * 64)
    d_record = build_record(d_report, "d" * 64)
    assert d_record["scoring"]["mode"] == "D"
    assert d_record["scoring_semantics_digest"] != c_record["scoring_semantics_digest"]
    assert d_record["measurement_semantics_digest"] != c_record["measurement_semantics_digest"]
    ledger = json.loads((cohort.checkpoint.parent / "evaluations.json").read_text())
    assert [r["scoring"]["mode"] for r in ledger["records"]] == ["D"]

    history, _problems = read_usage_history(cohort.version)
    runs = [entry for entry in history if entry.get("kind") == "test_run"]
    assert [entry_mode(entry) for entry in runs] == ["C", "D"]


def test_the_c_side_is_run_mode_c_on_the_same_embeddings(cohort):
    """Its ranks equal a direct `run_mode_c` call on the pipeline's embeddings
    with the same samples, device and batch size, and the kept Mode C report's
    ranks, read by the digest it names, all agree (CPU, one environment)."""
    import scripts.measure_scorer as cli
    from src.evaluation.measurement import EncodedGraph, MeasurementManifest, run_mode_c
    from src.evaluation.mode_d import run_mode_d, served_semantics
    from src.inference.pipeline import build_served_pipeline

    assert cli.main(_argv(cohort)) == 0
    pipeline = build_served_pipeline(cohort.data_dir / "kg.json",
                                     checkpoint_path=str(cohort.checkpoint),
                                     data_dir=str(cohort.data_dir), device="cpu")
    samples = cli._read_version_samples(cohort.version).samples
    report = _reports(cohort)["C"][0][1]
    manifest_c = MeasurementManifest(**report["manifest"])
    manifest_d = MeasurementManifest(**{**report["manifest"],
                                        **served_semantics(pipeline.served_identity())})
    direct = run_mode_c(EncodedGraph(pipeline.node_embeddings, pipeline.embedding_regime),
                        samples, manifest_c, device="cpu", batch_size=manifest_c.batch_size)
    observed = []
    result, mode_c = run_mode_d(
        pipeline, samples, manifest_d, manifest_c, device="cpu",
        batch_size=manifest_c.batch_size, cohort_version=report["manifest"]["cohort_version"],
        kept_mode_c=lambda ranks: observed.append(dict(ranks)) or [],
    )
    assert mode_c.canonical_ranks == direct.canonical_ranks
    assert mode_c.authoritative_metrics == direct.authoritative_metrics
    kept = json.loads((_reports(cohort)["C"][0][0].parent / "ranks.json").read_text())
    assert {row["sample_id"]: row["rank"] for row in kept} == observed[0]
    assert [row["c_rank"] for row in result.to_ranks()] == direct.canonical_ranks


def test_the_observer_changes_nothing_mode_c_reports(cohort):
    """`run_mode_c` with an observer that reads every score gives the same result
    as without one."""
    import scripts.measure_scorer as cli
    from src.evaluation.measurement import EncodedGraph, MeasurementManifest, run_mode_c
    from src.inference.pipeline import build_served_pipeline

    assert cli.main(_argv(cohort)) == 0
    report = _reports(cohort)["C"][0][1]
    pipeline = build_served_pipeline(cohort.data_dir / "kg.json",
                                     checkpoint_path=str(cohort.checkpoint),
                                     data_dir=str(cohort.data_dir), device="cpu")
    samples = cli._read_version_samples(cohort.version).samples
    encoded = EncodedGraph(pipeline.node_embeddings, pipeline.embedding_regime)
    manifest = MeasurementManifest(**report["manifest"])
    seen = []
    plain = run_mode_c(encoded, samples, manifest, device="cpu", batch_size=4)
    watched = run_mode_c(encoded, samples, manifest, device="cpu", batch_size=4,
                         on_scores=lambda first, ids, scores: seen.append(
                             (first, list(ids), float(scores.sum()))))
    assert watched.to_dict() == plain.to_dict() and watched.to_ranks() == plain.to_ranks()
    assert [first for first, _, _ in seen] == list(range(0, len(samples), 4))


def _text(html: str) -> str:
    """Rendered HTML as a person reads it: tags dropped, entities decoded."""
    import html as html_lib
    import re

    return html_lib.unescape(re.sub(r"<[^>]+>", "", html))


def test_each_surface_labels_a_record_by_its_own_mode_and_mode_d_is_never_ordered(cohort):
    """Labels by record (§5, §6): the Test page's Result, the Models page's
    records and the usage history name each record's own mode and case set, with
    the pre-B-1 note beside Mode C records only; a Mode D scoring group is never
    offered for ordering, and the ordering itself refuses one."""
    pytest.importorskip("gradio")
    import scripts.measure_scorer as cli
    from src.evaluation.caveats import (
        MODE_C_RESULT_LABEL,
        MODE_D_NOTE,
        PRE_B1_NOTE,
        mode_label,
    )
    from src.evaluation.ordering import Candidate, order_by_test_score
    from src.evaluation.report_store import read_usage_history
    from src.webui.components import models_page as mp
    from src.webui.components import test_page as tp

    assert cli.main(_argv(cohort, "--record")) == 0
    assert cli.main(["--mode", "D", *_argv(cohort, "--record")]) == 0
    counts = cohort.version.manifest["counts"]
    reached_of = "of {} import-eligible cases reached".format(counts["ranked_eligible"])

    # The Test page's Result, read as the page reads the version's store.
    reports = tp.read_reports(cohort.version.manifest["label"])
    runs = {run["record"]["scoring"]["mode"]: run for run in reports["runs"] if run.get("record")}
    d_record = runs["D"]["record"]
    result = _text(tp.result_html(reports, runs["D"]["run_id"], []))
    assert mode_label("D", d_record["scoring"]["score_semantics"]) in result
    assert "scores them (gnn_only, η = 1.0); not the approved scorer policy" in result
    assert "Mode D MRR (import-eligible)" in result and "Mode D coverage" in result
    assert f"{d_record['n_ranked']} {reached_of} by the served pipeline" in result
    assert "Mode C" not in result and PRE_B1_NOTE not in result
    assert tp.TOPIC_HELP["test-mode-d"] == ("Mode D result", MODE_D_NOTE)
    assert "a Mode C record shown here is a separate run" in MODE_D_NOTE    # ui_api-6
    rows = _text(tp.reports_html(reports, None))
    assert "served ranking (Mode D): report kept" in rows         # ui_api-5
    assert "model ranking (Mode C): report kept" in rows
    c_result = _text(tp.result_html(reports, runs["C"]["run_id"], []))
    assert MODE_C_RESULT_LABEL in c_result and "Mode D" not in c_result

    # The Models page's records: each mode's label and note once, each row its mode.
    ledger = json.loads((cohort.checkpoint.parent / "evaluations.json").read_text())
    views = [mp.record_view(record) for record in ledger["records"]]
    assert [view["mode"] for view in views] == ["C", "D"]
    block = _text(mp.records_block_html(views, {}))
    assert block.count(PRE_B1_NOTE) == 1 and block.count(MODE_D_NOTE) == 1
    # One table per mode, each under its own label, its columns named in full
    # with the mode and the case set (ui_api-1); one served setting shown, so it
    # is in the label, as PLAN_B05 §5's fixed text has it (ui_api-3).
    raw = mp.records_block_html(views, {})
    assert 'aria-label="Recorded test results, model ranking (Mode C)"' in raw
    assert 'aria-label="Recorded test results, served ranking (Mode D)"' in raw
    d_label = mode_label("D", views[1]["score_semantics"])
    order = [MODE_C_RESULT_LABEL, "Mode C MRR (untruncated)", "Mode C Hits@10", d_label,
             "Mode D MRR (import-eligible)", "Mode D Hits@10 (import-eligible)"]
    assert [block.index(text) for text in order] == sorted(block.index(text) for text in order)
    assert "scores them (gnn_only, η = 1.0); not the approved scorer policy" in block
    assert "served scoring" not in block and reached_of in block
    # Two served settings: the label without them, each record's line its own.
    other = {**views[1], "score_semantics": "served:gnn_plus_shortest_path:eta=0.7:sp=abc"}
    two = _text(mp.records_block_html([views[1], other], {}))
    assert f"{mode_label('D')}." in two
    assert "served scoring gnn_only, η = 1.0" in two
    assert "served scoring gnn_plus_shortest_path, η = 0.7" in two
    d_only = _text(mp.records_block_html(views[1:], {}))
    assert "Mode D MRR (import-eligible)" in d_only and "Mode D Hits@10 (import-eligible)" in d_only
    assert MODE_C_RESULT_LABEL not in d_only and PRE_B1_NOTE not in d_only
    assert "Mode C MRR" not in d_only and "Mode C metrics" not in d_only

    # The usage history, an entry written before `mode` existed reading as Mode C.
    history, _problems = read_usage_history(cohort.version)
    c_entry, d_entry = [entry for entry in history if entry.get("kind") == "test_run"]
    c_usage, d_usage = _text(mp.usage_entry_html(c_entry)), _text(mp.usage_entry_html(d_entry))
    assert "Mode C MRR (untruncated)" in c_usage and "ranked cases" in c_usage
    assert "Mode D MRR (import-eligible)" in d_usage and "cases reached" in d_usage
    assert _text(mp.usage_entry_html({k: v for k, v in c_entry.items() if k != "mode"})) == c_usage

    # Ordering: only Mode C's group is offered; asked for Mode D's, nothing is ordered.
    c_record, d_record = ledger["records"]
    sha, version = c_record["checkpoint_digest"], c_record["cohort_digest"]
    listing = {"items": [{"architecture": "a", "run": "r", "filename": "m.pt", "sha256": sha}]}
    data = {"ledgers": {"a/r/m.pt": {"state": "records", "records": ledger["records"],
                                     "ledger": "evaluations.json"}}}
    assert mp.scoring_groups(listing, data, version, None) == [
        (c_record["scoring_semantics_digest"], 1)]

    def order(record):
        return order_by_test_score(
            [Candidate(sha, tuple(ledger["records"]), "evaluations.json")],
            cohort_digest=version, scoring_semantics_digest=record["scoring_semantics_digest"],
            metric="untruncated_mrr", role=record["cohort_role"],
            ordering_roles=(record["cohort_role"],), contested_state=lambda key: "uncontested")

    assert order(c_record).offered
    refused = order(d_record)
    assert not refused.offered and refused.rows == ()
    assert "mode D, not Mode C; only Mode C records are ordered" in refused.why


def test_a_refused_case_is_counted_apart_from_not_reached_and_progress_is_logged(
        cohort, monkeypatch, caplog):
    """The internal review's mode_d-1: a case the served validation refuses is
    `refused`, never counted in d_not_reached_while_c_ranked; and the long served
    pass logs where it is (a progress line naming the case count)."""
    import logging

    import scripts.measure_scorer as cli
    from src.inference.pipeline import DiagnosisPipeline, ServedRanking

    samples = cli._read_version_samples(cohort.version).samples
    first = str(samples[0].patient_id)
    original = DiagnosisPipeline.rank

    def rank(self, patient, *args, **kwargs):
        if patient.patient_id == first:
            return ServedRanking("refused", ["over the input limit"], patient)
        return original(self, patient, *args, **kwargs)

    monkeypatch.setattr(DiagnosisPipeline, "rank", rank)
    with caplog.at_level(logging.INFO, logger="src.evaluation.mode_d"):
        assert cli.main(["--mode", "D", *_argv(cohort)]) == 0
    (d_path, d_report), = _reports(cohort)["D"]
    rows = {row["sample_id"]: row for row in
            json.loads((d_path.parent / "ranks.json").read_text())}
    steps = d_report["comparison"]["steps"]
    assert d_report["outcomes"]["refused"] == 1 and rows[first]["outcome"] == "refused"
    for k in (10, 100):
        c_ranked = [row for row in rows.values()
                    if row["c_rank"] is not None and row["c_rank"] <= k]
        assert steps["d_refused_while_c_ranked"][f"top_{k}"] == sum(
            1 for row in c_ranked if row["outcome"] == "refused")
        assert steps["d_not_reached_while_c_ranked"][f"top_{k}"] == sum(
            1 for row in c_ranked if row["outcome"] == "not_reached")
    assert steps["d_refused_while_c_ranked"]["top_100"] == 1
    for pair in ("C->G1", "C->D"):          # never blamed on the candidate set
        assert steps["paired"][pair]["ranked_then_refused"] == 1
    # The last line comes after the last case: its tally is the whole pass.
    last = [line for line in caplog.text.splitlines() if "through the served path" in line][-1]
    assert f"Mode D: {len(samples)} of {len(samples)} cases through the served path" in last
    assert "'refused': 1" in last
    assert sum(d_report["outcomes"][name] for name in ("ranked", "not_reached", "refused")) == \
        len(samples)


def test_a_cohort_mapped_under_another_normalisation_is_refused_before_any_case_is_ranked(
        cohort, monkeypatch):
    """The internal review's gap finding: a version mapped under another
    phenotype normalisation than the served one is refused before the served
    pass, so the two sides never score different terms; nothing is written."""
    import scripts.measure_scorer as cli
    import src.evaluation.mode_d as mode_d
    import src.inference.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "PHENOTYPE_NORMALISATION_VERSION", 99)
    ranked = []
    monkeypatch.setattr(mode_d, "rank_served_cases", lambda *a, **k: ranked.append(1) or [])
    with pytest.raises(ModeDRefused, match="phenotype normalisation version 1 .* under 99"):
        cli.main(["--mode", "D", *_argv(cohort)])
    assert not ranked
    assert not list((cohort.version.directory / "reports").glob("*/report.json"))


def test_a_served_input_that_drops_a_mapped_term_is_refused(cohort, monkeypatch):
    """The same premise, case by case: the served pipeline must use exactly the
    terms the version mapped, or the run is refused and nothing is written."""
    from dataclasses import replace

    import scripts.measure_scorer as cli
    from src.inference.pipeline import DiagnosisPipeline

    original = DiagnosisPipeline.rank

    def rank(self, patient, *args, **kwargs):
        ranking = original(self, patient, *args, **kwargs)
        if ranking.phenotype_input is not None:
            ranking.phenotype_input = replace(ranking.phenotype_input,
                                              used=ranking.phenotype_input.used[:-1])
        return ranking

    monkeypatch.setattr(DiagnosisPipeline, "rank", rank)
    with pytest.raises(ModeDRefused, match="the served pipeline used"):
        cli.main(["--mode", "D", *_argv(cohort)])
    assert not list((cohort.version.directory / "reports").glob("*/report.json"))


def test_a_kept_report_whose_ranks_block_is_damaged_is_listed_unreadable(cohort):
    """The internal review's mode_d-2: a kept Mode C report whose `ranks` block
    is not an object is listed with the reason, never a traceback."""
    import scripts.measure_scorer as cli
    from src.evaluation.mode_d import kept_mode_c_reports

    assert cli.main(_argv(cohort)) == 0
    (c_path, c_report), = _reports(cohort)["C"]
    c_report["ranks"] = ["not", "an", "object"]
    c_path.write_text(json.dumps(c_report))
    checkpoint = c_report["manifest"]["artifact_digests"]["checkpoint"]
    (entry,) = kept_mode_c_reports(cohort.version, checkpoint, {})
    assert "ranks block is not an object" in entry["unreadable"]


def test_run_mode_d_checks_the_c_side_before_the_served_pass(cohort, monkeypatch):
    """The internal review's mode_d-4, for a caller other than the CLI: the C
    side's batch size, mode and regime are refused before any case is ranked."""
    import dataclasses

    import scripts.measure_scorer as cli
    import src.evaluation.mode_d as mode_d
    from src.evaluation.measurement import MeasurementManifest
    from src.inference.pipeline import build_served_pipeline

    assert cli.main(_argv(cohort)) == 0
    assert cli.main(["--mode", "D", *_argv(cohort)]) == 0
    reports = _reports(cohort)
    c_manifest = MeasurementManifest(**reports["C"][0][1]["manifest"])
    d_report = reports["D"][0][1]
    d_manifest = MeasurementManifest(**d_report["manifest"])
    pipeline = build_served_pipeline(cohort.data_dir / "kg.json",
                                     checkpoint_path=str(cohort.checkpoint),
                                     data_dir=str(cohort.data_dir), device="cpu")
    samples = cli._read_version_samples(cohort.version).samples
    ranked = []
    monkeypatch.setattr(mode_d, "rank_served_cases", lambda *a, **k: ranked.append(1) or [])

    def run(c_side=c_manifest, batch_size=4):
        return mode_d.run_mode_d(pipeline, samples, d_manifest, c_side,
                                 device=torch.device("cpu"), batch_size=batch_size,
                                 cohort_version=d_report["manifest"]["cohort_version"])

    for kwargs, match in (({"batch_size": 0}, "batch size"), ({"batch_size": True}, "batch size"),
                          ({"c_side": dataclasses.replace(c_manifest, mode="D")}, "not C")):
        with pytest.raises(ModeDRefused, match=match):
            run(**kwargs)
    # The C side's manifest claims autocast: Mode C's own regime check refuses it.
    with pytest.raises(RuntimeError, match="describes a run that did not happen"):
        run(c_side=dataclasses.replace(c_manifest, amp_enabled=True,
                                       amp_dtype="torch.bfloat16"))
    assert not ranked


@pytest.mark.parametrize("value", ["0", "-3", "two"])
def test_a_batch_size_that_is_not_a_positive_integer_is_refused_before_anything_runs(value):
    """The internal review's cli-5: refused at the command line (exit 2), not
    after Mode D's served pass has ranked every case."""
    import scripts.measure_scorer as cli

    with pytest.raises(SystemExit) as raised:
        cli.parse_args(["--mode", "D", "--checkpoint", "x.pt", "--data-dir", "d",
                        "--cohort", "label", "--registry", "/r", "--batch-size", value])
    assert raised.value.code == 2


def test_mode_d_is_refused_for_a_generated_split():
    import scripts.measure_scorer as cli

    with pytest.raises(SystemExit) as raised:
        cli.parse_args(["--mode", "D", "--checkpoint", "x.pt", "--data-dir", "d",
                        "--split", "val", "--output", "o.json"])
    assert raised.value.code == 2


def test_a_version_bound_to_another_graph_is_refused_before_any_case_is_ranked(
        cohort, tmp_path, monkeypatch):
    """The served pipeline over another workspace (its own checkpoint), the
    version bound to the first: refused by the cohort binding, nothing written."""
    import scripts.measure_scorer as cli
    import src.evaluation.mode_d as mode_d
    from src.evaluation.cohort_registry import CohortBindingMismatch
    from tests.fixtures.registered_cohort import write_demo_workspace, write_fixture_checkpoint

    other = write_demo_workspace(tmp_path / "other", feature_seed=7)
    other_checkpoint = tmp_path / "other_ckpt" / "best.pt"
    write_fixture_checkpoint(other, other_checkpoint)
    ranked = []
    monkeypatch.setattr(mode_d, "rank_served_cases",
                        lambda *a, **k: ranked.append(1) or [])
    with pytest.raises(CohortBindingMismatch):
        cli.main(["--mode", "D", *_argv(cohort, data_dir=other, checkpoint=other_checkpoint)])
    assert not ranked
    assert not (cohort.version.directory / "reports").exists()
