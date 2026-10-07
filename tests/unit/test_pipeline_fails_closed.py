"""A requested model is served, or nothing is.

`_load_model_from_checkpoint` returned None when the checkpoint was missing,
unreadable or did not build over the graph, and `_init_gnn_inference` returned
without the GNN when that happened or when there was no graph to compute its
embeddings from. The pipeline was then published as a success, and every
diagnosis was answered with path-reasoning scores in the model's place, over
HTTP 200, with nothing in the response saying so. `sp_optional=False` did the
same in its own way: it switched the GNN off, so a configuration requiring both
signals was served with neither.

Each of those now raises `PipelineBuildError`
(docs/working/PLAN_PROVENANCE_CONTRACT.md, M1). The API needed no change for the
refusal to land correctly, because it already builds before it publishes:

- **at startup**, no pipeline is published and `/diagnose` answers 503;
- **on a reload**, the candidate is refused and the pipeline already serving
  stays.

**Unchanged here, deliberately:** a pipeline built with no model source at all
still serves path reasoning. Whether that needs an explicit opt-in is B-2's open
policy question, not this change.

Module: tests/unit/test_pipeline_fails_closed.py
"""
from __future__ import annotations

import asyncio
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")

from src.inference import pipeline as pipeline_module  # noqa: E402
from src.inference.pipeline import (  # noqa: E402
    DiagnosisPipeline,
    PipelineBuildError,
    PipelineConfig,
)
from src.kg.graph import KnowledgeGraph  # noqa: E402
from tests.fixtures.synthetic_workspace import (  # noqa: E402
    FEATURE_DIM,
    HIDDEN_DIM,
    build_workspace,
)


def _graph_data(data_dir):
    return {
        "x_dict": torch.load(data_dir / "node_features.pt", weights_only=False),
        "edge_index_dict": torch.load(data_dir / "edge_indices.pt", weights_only=False),
        "num_nodes_dict": json.loads((data_dir / "num_nodes.json").read_text()),
    }


def _checkpoint_for_another_graph(data_dir, path):
    """A sound checkpoint, trained over a graph whose features are twice as wide.

    Everything about it is well formed: it loads, its configuration matches its
    weights, and it would serve its own graph. It does not build over this one.
    """
    from src.models.gnn.shepherd_gnn import ShepherdGNN, ShepherdGNNConfig

    graph = _graph_data(data_dir)
    metadata = (list(graph["num_nodes_dict"]), list(graph["edge_index_dict"]))
    in_channels = dict.fromkeys(graph["num_nodes_dict"], FEATURE_DIM * 2)
    model = ShepherdGNN(
        metadata=metadata,
        in_channels_dict=in_channels,
        config=ShepherdGNNConfig(hidden_dim=HIDDEN_DIM, num_layers=2, num_heads=2),
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {"hidden_dim": HIDDEN_DIM, "num_layers": 2, "num_heads": 2},
            "metadata": metadata,
            "in_channels_dict": in_channels,
            "epoch": 1,
        },
        path,
    )
    return path


@pytest.fixture
def workspace(tmp_path):
    return build_workspace(tmp_path)


# ==============================================================================
# The pipeline: a requested model is built, or the build raises
# ==============================================================================
class TestARequestedModelIsBuiltOrRefused:
    def test_a_sound_checkpoint_builds_its_model(self, workspace):
        """The control. Without it, every refusal below could be a harness that
        never builds anything."""
        data_dir, checkpoint = workspace

        pipeline = DiagnosisPipeline(
            kg=KnowledgeGraph(), graph_data=_graph_data(data_dir),
            checkpoint_path=str(checkpoint), device="cpu",
        )

        config = pipeline.get_pipeline_config()
        assert config["gnn_ready"] is True
        assert config["scoring_mode"] == "gnn_only"

    def test_a_missing_checkpoint_is_refused(self, workspace, tmp_path):
        data_dir, _ = workspace
        absent = tmp_path / "absent.pt"

        with pytest.raises(PipelineBuildError, match="does not exist"):
            DiagnosisPipeline(
                kg=KnowledgeGraph(), graph_data=_graph_data(data_dir),
                checkpoint_path=str(absent), device="cpu",
            )

    def test_an_unreadable_checkpoint_is_refused(self, workspace, tmp_path):
        data_dir, _ = workspace
        corrupt = tmp_path / "corrupt.pt"
        corrupt.write_bytes(b"not a checkpoint")

        with pytest.raises(PipelineBuildError, match="could not be read") as refused:
            DiagnosisPipeline(
                kg=KnowledgeGraph(), graph_data=_graph_data(data_dir),
                checkpoint_path=str(corrupt), device="cpu",
            )

        assert str(corrupt) in str(refused.value)
        assert refused.value.__cause__ is not None, "the reader's own error is kept"

    def test_a_checkpoint_for_another_graph_is_refused(self, workspace, tmp_path):
        data_dir, _ = workspace
        other = _checkpoint_for_another_graph(data_dir, tmp_path / "other.pt")

        with pytest.raises(PipelineBuildError, match="does not build a model"):
            DiagnosisPipeline(
                kg=KnowledgeGraph(), graph_data=_graph_data(data_dir),
                checkpoint_path=str(other), device="cpu",
            )

    def test_a_loader_that_returns_nothing_cannot_mark_the_gnn_ready(
        self, monkeypatch, workspace
    ):
        """The old contract, reintroduced by a future loader. With no model the
        embeddings are never computed, and `_gnn_ready` set anyway would score
        every candidate's GNN term as zero under a status that reports a GNN."""
        data_dir, checkpoint = workspace
        monkeypatch.setattr(
            DiagnosisPipeline, "_load_model_from_checkpoint",
            lambda self, checkpoint_path, device: None,
        )

        with pytest.raises(PipelineBuildError, match="no node embeddings"):
            DiagnosisPipeline(
                kg=KnowledgeGraph(), graph_data=_graph_data(data_dir),
                checkpoint_path=str(checkpoint), device="cpu",
            )

    def test_a_checkpoint_with_no_graph_is_refused(self, workspace):
        _, checkpoint = workspace

        with pytest.raises(PipelineBuildError, match="no graph"):
            DiagnosisPipeline(
                kg=KnowledgeGraph(), checkpoint_path=str(checkpoint), device="cpu",
            )

    def test_a_preloaded_model_with_no_graph_is_refused(self):
        """The other model source. A model handed in directly is as much a
        request for a GNN as a checkpoint path is."""
        with pytest.raises(PipelineBuildError, match="no graph"):
            DiagnosisPipeline(kg=KnowledgeGraph(), model=object(), device="cpu")

    def test_without_torch_a_requested_model_is_refused(self, monkeypatch, workspace):
        _, checkpoint = workspace
        monkeypatch.setattr(pipeline_module, "HAS_TORCH", False)

        with pytest.raises(PipelineBuildError, match="PyTorch is not available"):
            DiagnosisPipeline(kg=KnowledgeGraph(), checkpoint_path=str(checkpoint))

    def test_required_shortest_paths_that_are_absent_refuse_the_build(self, workspace):
        """`sp_optional=False` used to switch the GNN off and carry on, so a
        configuration that required both signals was served with neither."""
        data_dir, checkpoint = workspace
        assert not (data_dir / "shortest_paths.pt").exists()

        with pytest.raises(PipelineBuildError, match="sp_optional=False"):
            DiagnosisPipeline(
                kg=KnowledgeGraph(),
                config=PipelineConfig(sp_optional=False),
                checkpoint_path=str(checkpoint),
                data_dir=str(data_dir),
                kg_path=str(data_dir / "kg.json"),
                device="cpu",
            )

    def test_optional_shortest_paths_that_are_absent_still_build(self, workspace):
        """The default. Absent by configuration is the state
        `DISEASE_SCORER_POLICY.md` §2 blesses, and it stays a GNN pipeline."""
        data_dir, checkpoint = workspace

        pipeline = DiagnosisPipeline(
            kg=KnowledgeGraph(),
            checkpoint_path=str(checkpoint),
            data_dir=str(data_dir),
            kg_path=str(data_dir / "kg.json"),
            device="cpu",
        )

        assert pipeline.get_pipeline_config()["scoring_mode"] == "gnn_only"

    def test_no_model_source_still_serves_path_reasoning(self):
        """B-2's open policy question, left exactly where it was."""
        pipeline = DiagnosisPipeline(kg=KnowledgeGraph())

        config = pipeline.get_pipeline_config()
        assert config["gnn_ready"] is False
        assert config["scoring_mode"] == "path_reasoning_fallback"


# ==============================================================================
# The API: startup publishes nothing, a reload keeps what is serving
# ==============================================================================
@pytest.fixture
def api(monkeypatch):
    """App state as a fresh process has it, restored afterwards.

    `kg.json` in the synthetic workspace is a placeholder the manifest binds, so
    the graph object is stood in for. The digest check reads the file, not the
    object, and still runs.
    """
    import src.api.main as api_main

    for field in ("pipeline", "kg", "_current_data_dir", "_current_checkpoint_path"):
        monkeypatch.setattr(api_main.app_state, field, None, raising=False)
    monkeypatch.setattr(api_main.app_state, "model_version", "unknown", raising=False)
    monkeypatch.setattr(api_main.app_state, "is_ready", False, raising=False)
    monkeypatch.setattr(
        KnowledgeGraph, "load_json", staticmethod(lambda *a, **k: KnowledgeGraph())
    )
    for name in ("SHEPHERD_KG_PATH", "SHEPHERD_DATA_DIR", "SHEPHERD_CHECKPOINT_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SHEPHERD_DEVICE", "cpu")
    return api_main


def _diagnose(client):
    return client.post(
        "/api/v1/diagnose", json={"phenotypes": ["HP:0001250", "HP:0001263"]}
    )


def _reload(**kwargs):
    from src.api.routes.pipeline import PipelineReloadRequest, reload_pipeline

    return asyncio.run(reload_pipeline(PipelineReloadRequest(**kwargs)))


class TestStartupPublishesNothing:
    def _start(self, api, monkeypatch, data_dir, checkpoint):
        from fastapi.testclient import TestClient

        monkeypatch.setenv("SHEPHERD_KG_PATH", str(data_dir / "kg.json"))
        monkeypatch.setenv("SHEPHERD_DATA_DIR", str(data_dir))
        monkeypatch.setenv("SHEPHERD_CHECKPOINT_PATH", str(checkpoint))
        return TestClient(api.app)

    def test_a_sound_checkpoint_is_published_at_startup(
        self, api, monkeypatch, workspace
    ):
        """The control for the case below: the same startup, a good checkpoint."""
        data_dir, checkpoint = workspace

        with self._start(api, monkeypatch, data_dir, checkpoint):
            assert api.app_state.pipeline is not None
            assert api.app_state.pipeline.get_pipeline_config()["gnn_ready"] is True

    def test_a_checkpoint_that_cannot_load_leaves_diagnose_at_503(
        self, api, monkeypatch, workspace, tmp_path
    ):
        data_dir, _ = workspace
        corrupt = tmp_path / "corrupt.pt"
        corrupt.write_bytes(b"not a checkpoint")

        with self._start(api, monkeypatch, data_dir, corrupt) as client:
            assert api.app_state.pipeline is None, "startup published a pipeline"
            response = _diagnose(client)

        assert response.status_code == 503, response.text
        assert "could not be initialized" in response.text
        assert api.app_state.pipeline is None


class TestAReloadKeepsWhatIsServing:
    """The candidate is built first and published only once complete, so a
    refusal reaches nothing that was serving."""

    @staticmethod
    def _served(api, data_dir, checkpoint):
        result = _reload(data_dir=str(data_dir), checkpoint_path=str(checkpoint))
        assert result.success is True, result.message
        assert api.app_state.pipeline.get_pipeline_config()["gnn_ready"] is True
        return {
            field: getattr(api.app_state, field)
            for field in ("pipeline", "kg", "model_version",
                          "_current_data_dir", "_current_checkpoint_path")
        }

    @staticmethod
    def _assert_still_serving(api, before):
        for field, value in before.items():
            assert getattr(api.app_state, field) is value, (
                f"a refused reload changed app_state.{field}"
            )

    def test_a_checkpoint_that_cannot_be_read_is_refused(self, api, workspace, tmp_path):
        data_dir, checkpoint = workspace
        before = self._served(api, data_dir, checkpoint)
        corrupt = tmp_path / "corrupt.pt"
        corrupt.write_bytes(b"not a checkpoint")

        result = _reload(data_dir=str(data_dir), checkpoint_path=str(corrupt))

        assert result.success is False
        assert "could not be read" in result.message
        assert "still being served" in result.message
        self._assert_still_serving(api, before)

    def test_a_checkpoint_for_another_graph_is_refused(self, api, workspace, tmp_path):
        data_dir, checkpoint = workspace
        before = self._served(api, data_dir, checkpoint)
        other = _checkpoint_for_another_graph(data_dir, tmp_path / "other.pt")

        result = _reload(data_dir=str(data_dir), checkpoint_path=str(other))

        assert result.success is False
        assert "does not build a model" in result.message
        self._assert_still_serving(api, before)
