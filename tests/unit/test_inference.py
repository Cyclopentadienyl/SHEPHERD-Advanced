"""
Unit Tests for Inference Pipeline
=================================
Tests for DiagnosisPipeline and related components, on a real disease child
(`tests/fixtures/trained_children`, trained once per session): since K7 every
pipeline serves a model read from its checkpoint, and one with no model is
refused, never served by path reasoning.
"""
import inspect
from datetime import datetime

import pytest

torch = pytest.importorskip("torch")

from src.core.types import (  # noqa: E402
    DISEASE_TASK,
    DiseaseTaskCandidate,
    InferenceResult,
    PatientPhenotypes,
)
from src.inference import (  # noqa: E402
    DiagnosisPipeline,
    PipelineBuildError,
    PipelineConfig,
    create_diagnosis_pipeline,
)
from src.inference.pipeline import build_served_pipeline  # noqa: E402
from src.kg import KnowledgeGraph  # noqa: E402

#: Two phenotypes of the demo graph, in the disease child's graph view.
SEIZURE = "HP:0001250"
DEV_DELAY = "HP:0001263"
UNKNOWN = "HP:9999999"


def _built(trained, **kwargs):
    """A pipeline of its own over the trained child (for a test that changes
    the configuration)."""
    data_dir = trained["data_dir"]
    return DiagnosisPipeline(graph=KnowledgeGraph.read_json(data_dir / "kg.json"),
                             checkpoint_path=str(trained["disease"]), data_dir=str(data_dir),
                             device="cpu", **kwargs)


@pytest.fixture
def pipeline(disease_pipeline):
    return disease_pipeline


@pytest.fixture
def patient_phenotypes():
    """Sample patient phenotypes with valid KG IDs"""
    return PatientPhenotypes(
        patient_id="test_patient_001",
        phenotypes=[SEIZURE, DEV_DELAY],
    )


@pytest.fixture
def patient_phenotypes_with_invalid():
    """Patient phenotypes with some invalid IDs"""
    return PatientPhenotypes(
        patient_id="test_patient_002",
        phenotypes=[SEIZURE, UNKNOWN, DEV_DELAY],
    )


# =============================================================================
# Test DiagnosisPipeline
# =============================================================================
class TestDiagnosisPipeline:
    """Test DiagnosisPipeline"""

    def test_create_pipeline(self, pipeline):
        assert pipeline.kg is not None
        assert pipeline.config is not None
        assert pipeline.model is not None and pipeline._gnn_ready

    def test_create_pipeline_with_config(self, trained_disease_child):
        """Test creating pipeline with custom config"""
        config = PipelineConfig(
            max_path_length=3,
            include_explanations=False,
        )
        pipeline = _built(trained_disease_child, config=config)

        assert pipeline.config.max_path_length == 3
        assert pipeline.config.include_explanations is False
        assert pipeline.served_identity().settings.max_path_length == 3

    def test_validate_input(self, pipeline, patient_phenotypes):
        result = pipeline.validate_input(patient_phenotypes)

        assert result.is_valid
        assert len(result.validated_phenotypes) == 2
        assert result.rows is not None and len(result.rows) == 2

    def test_run_basic(self, pipeline, patient_phenotypes):
        result = pipeline.run(
            patient_input=patient_phenotypes,
            top_k=5,
            include_explanations=False,
        )

        assert isinstance(result, InferenceResult)
        assert result.patient_id == patient_phenotypes.patient_id
        assert isinstance(result.timestamp, datetime)
        assert result.inference_time_ms >= 0
        assert len(result.candidates) == 5

        # The model ranks: a disease task model's head scores every row.
        assert result.result_kind == DISEASE_TASK
        for candidate in result.candidates:
            assert isinstance(candidate, DiseaseTaskCandidate)
            assert candidate.rank > 0
            assert candidate.score <= 0
            assert candidate.disease_name is not None

    def test_run_with_explanations(self, pipeline, patient_phenotypes):
        result = pipeline.run(
            patient_input=patient_phenotypes,
            top_k=5,
            include_explanations=True,
        )

        assert len(result.candidates) > 0
        for candidate in result.candidates:
            assert candidate.explanation is not None
            assert len(candidate.explanation) > 0
        assert result.summary_explanation is not None

    def test_run_candidates_ranked(self, pipeline, patient_phenotypes):
        result = pipeline.run(
            patient_input=patient_phenotypes,
            top_k=10,
        )

        for i, candidate in enumerate(result.candidates):
            assert candidate.rank == i + 1
        scores = [c.score for c in result.candidates]
        assert scores == sorted(scores, reverse=True)

    def test_run_with_invalid_phenotypes(self, pipeline, patient_phenotypes_with_invalid):
        result = pipeline.run(
            patient_input=patient_phenotypes_with_invalid,
            top_k=5,
        )

        assert isinstance(result, InferenceResult)
        assert result.candidates
        assert result.phenotype_input.unknown == (UNKNOWN,)

    def test_run_empty_phenotypes(self, pipeline):
        patient = PatientPhenotypes(
            patient_id="test",
            phenotypes=[],
        )
        result = pipeline.run(patient_input=patient, top_k=5)

        assert len(result.candidates) == 0
        assert len(result.warnings) > 0

    def test_run_nonexistent_phenotypes(self, pipeline):
        patient = PatientPhenotypes(
            patient_id="test",
            phenotypes=["HP:9999999", "HP:8888888"],
        )
        result = pipeline.run(patient_input=patient, top_k=5)

        assert isinstance(result, InferenceResult)
        assert result.candidates == []

    def test_get_pipeline_config(self, pipeline):
        config_dict = pipeline.get_pipeline_config()

        # The class version string is gone (contract M3c); what is served is
        # stated part by part instead.
        assert "version" not in config_dict
        assert "max_path_length" in config_dict
        assert "aggregation_method" not in config_dict
        assert config_dict["has_model"] is True
        identity = config_dict["served_identity"]
        assert identity["scoring_mode"] == "disease_task"
        assert identity["checkpoint"]["state"] == "verified"
        assert identity["graph"]["state"] == "verified"
        assert identity["dataset_record"]["state"] == "verified"
        assert "aggregation_method" not in identity["settings"]


# =============================================================================
# No model, no pipeline (K7, PATH1 and G3)
# =============================================================================
class TestNoModelIsRefused:
    """A graph is never served without a trained model, and a model is never
    handed over in memory: the pipeline is built from a checkpoint over a
    verified workspace, or it is refused."""

    def test_no_checkpoint_is_refused(self, trained_disease_child):
        data_dir = trained_disease_child["data_dir"]
        graph = KnowledgeGraph.read_json(data_dir / "kg.json")
        for checkpoint in (None, ""):
            with pytest.raises(PipelineBuildError, match="No model is configured"):
                DiagnosisPipeline(graph=graph, checkpoint_path=checkpoint,
                                  data_dir=str(data_dir), device="cpu")

    def test_no_workspace_is_refused(self, trained_disease_child):
        data_dir = trained_disease_child["data_dir"]
        with pytest.raises(PipelineBuildError, match="no workspace"):
            DiagnosisPipeline(graph=KnowledgeGraph.read_json(data_dir / "kg.json"),
                              checkpoint_path=str(trained_disease_child["disease"]),
                              data_dir=None, device="cpu")

    def test_the_service_build_refuses_before_reading_the_graph(self, trained_disease_child,
                                                                monkeypatch):
        """`build_served_pipeline` with a graph and no checkpoint (a
        `SHEPHERD_KG_PATH`-only start) refuses before `kg.json` is parsed."""
        def no_read(*_args, **_kwargs):
            raise AssertionError("the graph was read for a build that has no model")

        monkeypatch.setattr(KnowledgeGraph, "read_json", staticmethod(no_read))
        with pytest.raises(PipelineBuildError, match="No model is configured"):
            build_served_pipeline(trained_disease_child["data_dir"] / "kg.json",
                                  checkpoint_path=None,
                                  data_dir=str(trained_disease_child["data_dir"]))

    @pytest.mark.parametrize("construct", [DiagnosisPipeline, create_diagnosis_pipeline])
    def test_there_is_no_in_memory_or_graph_only_seam(self, construct):
        """No `kg=` (a graph alone), `model=` or `graph_data=` (a model and
        tensors handed over in memory): no alias, no adapter."""
        parameters = inspect.signature(construct).parameters
        assert {"kg", "model", "graph_data"}.isdisjoint(parameters)
        assert {"graph", "checkpoint_path", "data_dir"} <= set(parameters)
        with pytest.raises(TypeError):
            construct(kg=KnowledgeGraph())


# =============================================================================
# Test PipelineConfig
# =============================================================================
class TestPipelineConfig:
    """Test PipelineConfig"""

    def test_default_config(self):
        """Test default configuration"""
        config = PipelineConfig()

        assert config.max_path_length == 4
        assert config.eta == 0.7  # Default 70% GNN, 30% shortest path
        assert config.sp_optional is True
        assert config.include_explanations is True
        assert config.include_ortholog_evidence is True

    def test_custom_config(self):
        """Test custom configuration"""
        config = PipelineConfig(
            max_path_length=3,
            eta=0.5,
            sp_optional=False,
            include_explanations=False,
            include_ortholog_evidence=False,
        )

        assert config.max_path_length == 3
        assert config.eta == 0.5
        assert config.sp_optional is False
        assert config.include_explanations is False
        assert config.include_ortholog_evidence is False

    def test_eta_extreme_values(self):
        """eta=1.0 = pure GNN; eta=0.0 = pure shortest path"""
        config_pure_gnn = PipelineConfig(eta=1.0)
        assert config_pure_gnn.eta == 1.0

        config_pure_sp = PipelineConfig(eta=0.0)
        assert config_pure_sp.eta == 0.0

    @pytest.mark.parametrize(
        "retired", ["ortholog_weight", "custom_scorers", "post_process_callbacks"]
    )
    def test_retired_no_op_fields_are_gone(self, retired):
        """K7 (S3): fields nothing read are not configuration. A caller that
        still passes one is refused, never silently ignored."""
        import dataclasses

        assert retired not in {f.name for f in dataclasses.fields(PipelineConfig)}
        with pytest.raises(TypeError):
            PipelineConfig(**{retired: None})

    def test_the_path_score_aggregation_is_gone(self):
        """K7 (PATH1, PATH2): nothing ranks by an aggregate of path scores, so
        there is no aggregation to configure."""
        import dataclasses

        from src.reasoning import PathReasoningConfig

        assert "aggregation_method" not in {f.name for f in dataclasses.fields(PipelineConfig)}
        assert "aggregation_method" not in {
            f.name for f in dataclasses.fields(PathReasoningConfig)}
        with pytest.raises(TypeError):
            PipelineConfig(aggregation_method="max")

    def test_retired_callback_aliases_are_gone(self):
        import src.inference.pipeline as pipeline_module

        assert not hasattr(pipeline_module, "ScoringCallback")
        assert not hasattr(pipeline_module, "PostProcessCallback")


# =============================================================================
# The model is wired through
# =============================================================================
class TestGNNInference:
    """The trained child's vectors are computed once and score every request;
    path reasoning only explains the rows shown."""

    def test_gnn_ready(self, pipeline):
        assert pipeline._gnn_ready is True
        assert pipeline._bundle_z is not None and pipeline._bundle_candidates is not None
        assert pipeline.served_identity().settings.device == "cpu"

    def test_scores_come_from_the_model(self, pipeline, patient_phenotypes):
        result = pipeline.run(
            patient_input=patient_phenotypes,
            top_k=10,
            include_explanations=False,
        )
        scores = [c.score for c in result.candidates]
        assert len(set(scores)) > 1, "every disease scored the same: the model is not wired"
        assert all(-4.0 <= s <= 0.0 for s in scores)

    def test_pipeline_config_reports_gnn(self, pipeline):
        config = pipeline.get_pipeline_config()
        assert config["gnn_ready"] is True
        assert config["has_model"] is True
        assert config["scoring_mode"] == "disease_task"

    def test_path_reasoner_still_runs(self, pipeline, patient_phenotypes):
        """Path reasoning provides explanation paths for the rows shown."""
        result = pipeline.run(
            patient_input=patient_phenotypes,
            top_k=10,
            include_explanations=False,
        )
        assert any(len(c.reasoning_paths) > 0 for c in result.candidates)


# =============================================================================
# The phenotype-list rule, through pipeline.run
# =============================================================================
# docs/working/PLAN_PHENOTYPE_NORMALISATION.md: repeats are removed by graph node
# after mapping, first occurrence kept, and nothing is truncated. These tests go
# through `run` (or `validate_input`, which `run` calls first) rather than
# restating the rule; the rule alone is tested in test_phenotype_normalisation.py.
def _patient(phenotypes, confidences=None):
    return PatientPhenotypes(
        patient_id="norm_patient",
        phenotypes=list(phenotypes),
        phenotype_confidences=confidences,
    )


def _ranking(result):
    return [(c.disease_id, c.rank, c.score) for c in result.candidates]


class TestPhenotypeListRule:
    """The served path applies the one phenotype-list rule."""

    def test_a_repeat_is_scored_once(self, pipeline):
        with_repeat = pipeline.run(_patient([SEIZURE, SEIZURE, DEV_DELAY]), top_k=5)
        without = pipeline.run(_patient([SEIZURE, DEV_DELAY]), top_k=5)

        assert _ranking(with_repeat) == _ranking(without)
        assert with_repeat.phenotype_input.used == (SEIZURE, DEV_DELAY)
        assert with_repeat.phenotype_input.repeats_removed == 1
        assert "Removed 1 repeated phenotype entry; each phenotype is scored once." in (
            with_repeat.warnings
        )

    def test_one_hundred_repeats_then_another_phenotype_keeps_both(self, pipeline):
        """The previous version truncated to the first 100 entries before
        removing anything, so this input scored the seizure alone."""
        result = pipeline.run(_patient([SEIZURE] * 100 + [DEV_DELAY]), top_k=5)
        both = pipeline.run(_patient([SEIZURE, DEV_DELAY]), top_k=5)

        assert _ranking(result) == _ranking(both)
        assert result.phenotype_input.received == 101
        assert result.phenotype_input.used == (SEIZURE, DEV_DELAY)
        assert "Removed 99 repeated phenotype entries; each phenotype is scored once." in (
            result.warnings
        )
        assert not any("using first" in w for w in result.warnings)

    def test_too_many_distinct_phenotypes_are_refused_not_truncated(self, trained_disease_child):
        pipeline = _built(trained_disease_child, config=PipelineConfig(max_phenotypes=1))

        result = pipeline.run(_patient([SEIZURE, DEV_DELAY]), top_k=5)

        assert result.candidates == []
        assert any(
            "2 distinct phenotypes found in the graph; at most 1 are accepted" in w
            and "not shortened" in w
            for w in result.warnings
        )
        # Nothing was scored, so nothing is reported as used.
        assert result.phenotype_input.used == ()
        assert result.phenotype_input.received == 2

    def test_the_count_is_taken_after_repeats_are_removed(self, trained_disease_child):
        pipeline = _built(trained_disease_child, config=PipelineConfig(max_phenotypes=1))

        result = pipeline.run(_patient([SEIZURE, SEIZURE, SEIZURE]), top_k=5)

        assert result.candidates
        assert result.phenotype_input.used == (SEIZURE,)

    def test_unknown_ids_keep_their_policy_and_are_accounted_for(self, pipeline):
        result = pipeline.run(_patient([UNKNOWN, SEIZURE, DEV_DELAY]), top_k=5)

        assert f"Unknown phenotype: {UNKNOWN}" in result.warnings
        assert result.phenotype_input.unknown == (UNKNOWN,)
        assert result.phenotype_input.used == (SEIZURE, DEV_DELAY)

    def test_everything_after_validation_reads_the_list_as_used(
        self, pipeline, monkeypatch
    ):
        """Scoring receives the normalised input, with each confidence still at
        the phenotype it was sent with: `[X, A, A, B]` keeps positions 1 and 3."""
        seen = {}
        original = pipeline._rank_by_disease_task

        def capture(patient_input, *args, **kwargs):
            seen["input"] = patient_input
            return original(patient_input, *args, **kwargs)

        monkeypatch.setattr(pipeline, "_rank_by_disease_task", capture)

        result = pipeline.run(
            _patient([UNKNOWN, SEIZURE, SEIZURE, DEV_DELAY], confidences=[0.1, 0.2, 0.3, 0.4]),
            top_k=5,
            include_explanations=True,
        )

        assert seen["input"].phenotypes == [SEIZURE, DEV_DELAY]
        assert seen["input"].phenotype_confidences == [0.2, 0.4]
        assert "Based on 2 input phenotypes" in result.summary_explanation

    @pytest.mark.parametrize("confidences", [[], [0.5], [0.5, 0.5, 0.5]])
    def test_a_confidence_list_of_the_wrong_length_is_refused(self, pipeline, confidences):
        result = pipeline.run(_patient([SEIZURE, DEV_DELAY], confidences=confidences), top_k=5)

        assert result.candidates == []
        assert result.phenotype_input is None
        assert any(
            f"phenotype_confidences has {len(confidences)} entries for 2 phenotypes" in w
            for w in result.warnings
        )

    def test_no_confidences_and_the_right_length_both_pass(self, pipeline):
        assert pipeline.validate_input(_patient([SEIZURE, DEV_DELAY])).is_valid
        assert pipeline.validate_input(_patient([SEIZURE, DEV_DELAY], [0.9, 0.8])).is_valid
