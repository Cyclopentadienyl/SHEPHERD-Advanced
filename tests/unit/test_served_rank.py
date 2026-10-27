"""
`DiagnosisPipeline.rank()`: the served ranking, whole, and `run` as its cut.
=========================================================================
Batch 3 split `run` at its truncation point (`docs/working/scorer-measurement/
PLAN_B05.md` §2): `rank()` does everything `run` did up to and including the
sort, and `run` is `rank()` followed by the cut at `top_k` and what it builds on
the rows it keeps. These tests pin that:

- **serving did not change**: `run` reproduces the output captured before the
  split (`tests/fixtures/served_rank_golden.json`), in all three scoring modes,
  exact ties included, over the fields promised not to change (no timestamps);
- **`run` is `rank()` cut at `top_k`**, for every case and several cuts;
- **each candidate's discovery ordinal** is its position before the sort, so
  ties keep discovery order and the order can be read back without inferring it
  from the served one;
- `bound_inputs()`, `embedding_regime` and `build_served_pipeline` give a
  measurement what the build read, with no second read.

The workspace and checkpoint are the registered-cohort fixture's (CPU,
deterministic); the shortest-path table is made by the production producer.

Module: tests/unit/test_served_rank.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
GOLDEN = REPO / "tests" / "fixtures" / "served_rank_golden.json"
MODES = {"fallback": "path_reasoning_fallback", "gnn_only": "gnn_only",
         "gnn_plus_sp": "gnn_plus_shortest_path"}
SCORE_TOLERANCE = 1e-5


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """The fixture's workspace and checkpoint, a copy given a shortest-path
    table by `compute_shortest_paths.py`, and one pipeline per scoring mode."""
    from src.inference.pipeline import DiagnosisPipeline
    from src.kg.graph import KnowledgeGraph
    from tests.fixtures.registered_cohort import build_registered_cohort, graph_source_cases

    root = tmp_path_factory.mktemp("served_rank")
    cohort = build_registered_cohort(root / "c")
    with_table = root / "ws_sp"
    shutil.copytree(cohort.data_dir, with_table)
    produced = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "compute_shortest_paths.py"),
         "--kg-path", str(with_table / "kg.json"), "--output-dir", str(with_table),
         "--max-hops", "5", "--workers", "1"],
        cwd=REPO, capture_output=True, text=True, timeout=600,
    )
    assert produced.returncode == 0, produced.stderr[-2000:]

    def pipeline(workspace: Path, with_model: bool) -> DiagnosisPipeline:
        return DiagnosisPipeline(
            graph=KnowledgeGraph.read_json(workspace / "kg.json"), data_dir=str(workspace),
            checkpoint_path=str(cohort.checkpoint) if with_model else None,
            device="cpu" if with_model else None,
        )

    pipelines = {"fallback": pipeline(cohort.data_dir, False),
                 "gnn_only": pipeline(cohort.data_dir, True),
                 "gnn_plus_sp": pipeline(with_table, True)}
    for name, built in pipelines.items():
        assert built.served_identity().scoring_mode == MODES[name], name
    cases = [case["phenotypes"] for case in graph_source_cases(cohort.data_dir)]
    patients = cases + [["HP:9999999"], ["HP:9999999"] + cases[0] + cases[0][:1],
                        [f"HP:{i:07d}" for i in range(150)]]
    return {"cohort": cohort, "with_table": with_table, "pipelines": pipelines,
            "patients": patients}


def _patient(phenotypes, patient_id="p"):
    from src.core.types import PatientPhenotypes

    return PatientPhenotypes(patient_id=patient_id, phenotypes=list(phenotypes))


def _row(candidate):
    """A candidate as the golden records it, each of its kind's fields in the
    slot of the value it replaced (Spec 1 §2.1): [disease, rank, the ranked
    score, the model term, the SP record, the path score]. A field the kind does
    not carry is None, so a slot can never hold another kind's number."""
    from src.core.types import ServedMixtureCandidate

    if isinstance(candidate, ServedMixtureCandidate):
        return [str(candidate.disease_id), candidate.rank, candidate.served_score,
                candidate.model_term, candidate.sp_term, None]
    return [str(candidate.disease_id), candidate.rank, candidate.path_score, None, None,
            candidate.path_score]


def _matches_capture(name, row, want):
    """Each kind's fields against the full-precision capture of 755643a, the code
    before the union. Strictly, what each field means: the fields the kind
    carries, the SP status (numeric where a table is served, NO_TABLE where sp
    read 0.0), no mixed_as. Within SCORE_TOLERANCE, the numbers: the served
    score and model term against confidence and gnn, the SP score against sp,
    the path score against reasoning. The fixture runs the model forward again,
    and its float32 differs by an ulp between supported CPUs (aarch64 against
    this x86 capture, the reviewer's run of ce1ce1d). Bit-for-bit equality
    holds between the versions in one environment, which is how it was shown."""
    conf, gnn, sp, reasoning = (float(v) for v in want[2:])
    score, model_term, record, path_score = row[2:]

    def close(value, captured):
        return value == pytest.approx(captured, abs=SCORE_TOLERANCE)

    if name == "fallback":
        return (close(path_score, reasoning) and score == path_score
                and model_term is None and record is None)
    if record is None or not close(score, conf) or not close(model_term, gnn):
        return False
    if name == "gnn_plus_sp":
        return record.numeric and close(record.score, sp) and record.mixed_as is None
    return (record.status == "NO_TABLE" and record.score is None and sp == 0.0
            and score == model_term)


def _ties(scores):
    """Which neighbours tie: exact equality, which a tolerance would blur."""
    return [a == b for a, b in zip(scores, scores[1:])]


def test_the_capture_comparison_allows_an_ulp_and_nothing_else():
    """The reviewer's run of ce1ce1d on aarch64 (gnn_only, case 0): the model's
    float32 came out an ulp from the x86 capture. That is the same ranking; a
    real change of value, a wrong status or a field of another kind is not."""
    from src.core.types import SPAnalysisRecord

    no_table = SPAnalysisRecord(status="NO_TABLE", phenotypes_submitted=1, phenotypes_mapped=1)
    want = ["mondo:MONDO:0010730", 2, "0.5461347103118896", "0.5461347103118896", "0.0", "1.0"]

    def row(score, record=no_table):
        return ["mondo:MONDO:0010730", 2, score, score, record, None]

    assert _matches_capture("gnn_only", row(0.5461346507072449), want)
    assert not _matches_capture("gnn_only", row(0.5461347103118896 + 1e-3), want)
    computed = SPAnalysisRecord(status="COMPUTED", phenotypes_submitted=1, phenotypes_mapped=1,
                                max_hops=5, mean_hop_distance=1.0, score=0.5,
                                unreachable_pair_count=0,
                                denominator_policy="MAPPED_ONLY_DENOMINATOR")
    assert not _matches_capture("gnn_only", row(0.5461346507072449, computed), want)
    assert not _matches_capture("fallback", row(0.5461346507072449), want)
    assert _ties([1.0, 1.0, 0.5]) == [True, False]


def _stable(value, key=None):
    """A served identity as the golden records it: each digest and build id
    reduced to whether it is there. The digests are checked against the
    fixture's own files beside it (`test_run_reproduces_...`); the shortest-path
    table's build id is random per build."""
    if isinstance(value, dict):
        return {k: _stable(v, k) for k, v in value.items()}
    if key is not None and (key.endswith("sha256") or key == "build_id"):
        return "<present>" if value else value
    return value


def test_run_reproduces_the_output_captured_before_the_split(served):
    """The order, ranks, ties, warnings and phenotype account exactly, the four
    scores within 1e-5 of the capture before the split (e61253e), and each kind's
    fields against the capture before the union (755643a, Batch 4): what they
    mean exactly, their numbers within 1e-5 (`_matches_capture`), in each mode
    and for every case, the fallback's exact ties at 1.0 included; and each
    mode's served identity, its digests reduced to whether they are there
    (PLAN_B05 §8's fields; the phenotype account and the identity added after
    the internal review, serving-2). Catches a split or a union that moved a
    value, reordered a tie or changed what an empty or refused input returns."""
    from dataclasses import asdict

    from src.kg.provenance import PROVENANCE_FILENAME

    fixture = json.loads(GOLDEN.read_text())
    golden = fixture["results"]
    exact = fixture["exact_scores_755643a"]
    kinds = {"fallback": "path_reasoning/v1", "gnn_only": "served_mixture/v1",
             "gnn_plus_sp": "served_mixture/v1"}
    cohort = served["cohort"]
    for name, pipeline in served["pipelines"].items():
        identity = pipeline.served_identity().to_dict()
        assert _stable(identity) == fixture["served_identity"][name], name
        # Each digest the identity names is the digest of the fixture's own file.
        workspace = served["with_table"] if name == "gnn_plus_sp" else cohort.data_dir
        files = {("checkpoint", "sha256"): cohort.checkpoint,
                 ("graph", "kg_sha256"): workspace / "kg.json",
                 ("graph", "split_manifest_sha256"): workspace / "split_manifest.json",
                 ("dataset_record", "sha256"): workspace / PROVENANCE_FILENAME}
        for (part, field), path in files.items():
            if identity[part].get(field) is not None:
                assert identity[part][field] == _sha(path), (name, part, field)
        for index, phenotypes in enumerate(served["patients"]):
            result = pipeline.run(_patient(phenotypes, f"p{index}"), top_k=1000,
                                  include_explanations=False)
            expected = golden[f"{name}|{index}"]
            assert result.warnings == expected["warnings"], (name, index)
            account = json.loads(json.dumps(asdict(result.phenotype_input)))
            assert account == expected["phenotype_input"], (name, index)
            assert result.result_kind == kinds[name], (name, index)
            rows = [_row(c) for c in result.candidates]
            assert [r[:2] for r in rows] == [e[:2] for e in expected["candidates"]], (name, index)
            for row, want in zip(rows, expected["candidates"]):
                assert row[2] == pytest.approx(want[2], abs=SCORE_TOLERANCE), (name, index)
            captured = exact[f"{name}|{index}"]
            for row, want in zip(rows, captured):
                assert want[:2] == row[:2] and _matches_capture(name, row, want), (
                    name, index, row, want)
            assert _ties([r[2] for r in rows]) == _ties([float(w[2]) for w in captured]), (
                name, index)


@pytest.mark.parametrize("top_k", [1, 2, 5, 1000])
def test_run_is_rank_cut_at_top_k(served, top_k):
    """Field for field, the candidates `run` returns are `rank()`'s first
    `top_k`; the warnings and the phenotype account are the same."""
    for name, pipeline in served["pipelines"].items():
        for phenotypes in served["patients"]:
            ranking = pipeline.rank(_patient(phenotypes))
            result = pipeline.run(_patient(phenotypes), top_k=top_k, include_explanations=False)
            assert [_row(c) for c in result.candidates] == \
                [_row(c) for c in ranking.candidates[:top_k]], name
            assert result.warnings == ranking.warnings
            assert result.phenotype_input == ranking.phenotype_input


def test_the_three_outcomes_are_the_three_returns_of_run(served):
    """`refused` is a refused validation (no candidates, its errors as the
    warnings); `no_candidates` is an input that maps but reaches nothing, or
    nothing at all; `ranked` holds every candidate, ranked 1..N."""
    pipeline = served["pipelines"]["gnn_plus_sp"]
    refused = pipeline.rank(_patient(["HP:9999999"]))
    assert refused.outcome == "refused" and refused.candidates == []
    assert refused.warnings == ["No valid phenotypes after validation"]

    ranked = pipeline.rank(_patient(served["patients"][0]))
    assert ranked.outcome == "ranked"
    assert [c.rank for c in ranked.candidates] == list(range(1, len(ranked.candidates) + 1))
    scores = [c.served_score for c in ranked.candidates]
    assert scores == sorted(scores, reverse=True)


def test_each_candidate_carries_its_discovery_ordinal(served):
    """The ordinals are a permutation of the positions before the sort: sorted
    by ordinal, the candidates are in `all_paths` order. Equal scores keep
    ordinal order (the stable sort). Catches an ordinal read from the served
    order, which equals discovery order only when nothing was reordered."""
    reordered = 0
    for name, pipeline in served["pipelines"].items():
        for phenotypes in served["patients"]:
            ranking = pipeline.rank(_patient(phenotypes))
            if ranking.outcome != "ranked":
                continue
            ordinals = ranking.discovery_ordinals
            assert sorted(ordinals) == list(range(len(ranking.candidates))), name
            by_ordinal = [c for _, c in sorted(zip(ordinals, ranking.candidates),
                                               key=lambda pair: pair[0])]
            discovered = [key for key, paths in ranking.all_paths.items() if paths]
            assert [str(c.disease_id) for c in by_ordinal] == discovered, name
            for (first, a), (second, b) in zip(zip(ordinals, ranking.candidates),
                                               zip(ordinals[1:], ranking.candidates[1:])):
                if _row(a)[2] == _row(b)[2]:
                    assert first < second, name
            reordered += ordinals != sorted(ordinals)
    assert reordered, "no case reordered its candidates, so nothing above was tested"


def test_bound_inputs_are_the_reads_the_build_took(served):
    """Against an independent account of the files: the four graph digests, the
    checkpoint's bytes, its recorded training inputs and the dataset record. A
    pipeline that serves no model verified no workspace and refuses."""
    from src.inference.pipeline import DiagnosisPipeline
    from src.kg.artifacts import GRAPH_TENSOR_ROLES
    from src.kg.graph import KnowledgeGraph
    from src.kg.provenance import PROVENANCE_FILENAME

    cohort = served["cohort"]
    bound = served["pipelines"]["gnn_only"].bound_inputs()
    assert bound.graph["kg"] == _sha(cohort.data_dir / "kg.json")
    for role in GRAPH_TENSOR_ROLES:
        files = [p for p in cohort.data_dir.iterdir() if p.stem == role]
        assert len(files) == 1 and bound.graph[role] == _sha(files[0]), role
    assert bound.checkpoint.sha256 == _sha(cohort.checkpoint) == cohort.checkpoint_sha256
    recorded = torch.load(cohort.checkpoint, weights_only=True)["training_input_digests"]
    assert bound.checkpoint_training_inputs == recorded
    assert bound.dataset_record.identity.sha256 == _sha(cohort.data_dir / PROVENANCE_FILENAME)
    assert bound.manifest_sha256 == _sha(cohort.data_dir / "split_manifest.json")
    assert bound.graph_data is not None
    with pytest.raises(ValueError, match="not built over a verified workspace"):
        served["pipelines"]["fallback"].bound_inputs()
    # A model supplied in memory over a verified workspace serves a GNN, and is
    # refused for what it is: no checkpoint names its weights (the internal
    # review's serving-1; the message used to say it served no model).
    in_memory = DiagnosisPipeline(
        graph=KnowledgeGraph.read_json(cohort.data_dir / "kg.json"),
        data_dir=str(cohort.data_dir), model=served["pipelines"]["gnn_only"].model,
        device="cpu")
    assert in_memory.served_identity().scoring_mode == "gnn_only"
    with pytest.raises(ValueError, match="supplied in memory, not read from a checkpoint"):
        in_memory.bound_inputs()


def test_the_embedding_regime_is_observed_at_the_forward(served):
    """Off on a plain CPU build; on, with its dtype, when the build runs inside an
    autocast context, which is what the embeddings were computed under."""
    from src.inference.pipeline import DiagnosisPipeline
    from src.kg.graph import KnowledgeGraph
    from src.utils.run_environment import AutocastRegime

    assert served["pipelines"]["gnn_only"].embedding_regime == AutocastRegime(False, None)
    cohort = served["cohort"]
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        under = DiagnosisPipeline(
            graph=KnowledgeGraph.read_json(cohort.data_dir / "kg.json"),
            data_dir=str(cohort.data_dir), checkpoint_path=str(cohort.checkpoint), device="cpu",
        )
    assert under.embedding_regime == AutocastRegime(True, "torch.bfloat16")


def test_the_service_and_build_served_pipeline_build_the_same_pipeline(served):
    """The API's `build_pipeline` builds through `build_served_pipeline`, so both
    serve the same identity from the same files."""
    from src.api import main as api
    from src.inference.pipeline import build_served_pipeline

    workspace = served["with_table"]
    checkpoint = str(served["cohort"].checkpoint)
    direct = build_served_pipeline(workspace / "kg.json", checkpoint_path=checkpoint,
                                   data_dir=str(workspace), device="cpu")
    bundle = api.build_pipeline(kg_path=str(workspace / "kg.json"), checkpoint_path=checkpoint,
                                data_dir=str(workspace), device="cpu")
    assert direct.served_identity() == bundle.pipeline.served_identity()
    assert direct.served_identity().scoring_mode == "gnn_plus_shortest_path"
