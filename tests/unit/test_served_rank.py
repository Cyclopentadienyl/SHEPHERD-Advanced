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
    return [str(candidate.disease_id), candidate.rank, candidate.confidence_score,
            candidate.gnn_score, candidate.sp_score, candidate.reasoning_score]


def test_run_reproduces_the_output_captured_before_the_split(served):
    """The order, ranks and warnings exactly, the four scores within 1e-5, in each
    mode and for every case, the fallback's exact ties at 1.0 included. Catches a
    split that moved a value, reordered a tie or changed what an empty or
    refused input returns."""
    golden = json.loads(GOLDEN.read_text())["results"]
    for name, pipeline in served["pipelines"].items():
        for index, phenotypes in enumerate(served["patients"]):
            result = pipeline.run(_patient(phenotypes, f"p{index}"), top_k=1000,
                                  include_explanations=False)
            expected = golden[f"{name}|{index}"]
            assert result.warnings == expected["warnings"], (name, index)
            rows = [_row(c) for c in result.candidates]
            assert [r[:2] for r in rows] == [e[:2] for e in expected["candidates"]], (name, index)
            for row, want in zip(rows, expected["candidates"]):
                assert row[2:] == pytest.approx(want[2:], abs=SCORE_TOLERANCE), (name, index)


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
    scores = [c.confidence_score for c in ranked.candidates]
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
                if a.confidence_score == b.confidence_score:
                    assert first < second, name
            reordered += ordinals != sorted(ordinals)
    assert reordered, "no case reordered its candidates, so nothing above was tested"


def test_bound_inputs_are_the_reads_the_build_took(served):
    """Against an independent account of the files: the four graph digests, the
    checkpoint's bytes, its recorded training inputs and the dataset record. A
    pipeline that serves no model verified no workspace and refuses."""
    from src.kg.artifacts import GRAPH_TENSOR_ROLES
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
