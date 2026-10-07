"""A requested model is served, or nothing is.

`_load_model_from_checkpoint` returned None when the checkpoint was missing or
did not build over the graph, and `_init_gnn_inference` returned without the GNN
when that happened or when there was no graph to compute its embeddings from.
The pipeline was then published as a success, and every diagnosis was answered
with path-reasoning scores in the model's place, over HTTP 200, with nothing in
the response saying so. `sp_optional=False` did the same in its own way: it
switched the GNN off, so a configuration requiring both signals was served with
neither. (An unreadable checkpoint already raised, as torch's own error; its
cases here hold it there.)

Each of those now raises `PipelineBuildError`
(docs/working/PLAN_PROVENANCE_CONTRACT.md, M1). The API's build-then-publish
order already put a raised build in the right place:

- **at startup**, no pipeline is published and `/diagnose` answers 503;
- **on a reload**, the candidate is refused and the pipeline already serving
  stays.

Three API changes close what that order did not: a checkpoint configured with
no knowledge graph is refused rather than treated as nothing configured; a blank
environment value is unset; and `/diagnose` no longer rebuilds a failed pipeline
on every request.

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

    def test_embeddings_without_a_node_type_scoring_reads_cannot_mark_it_ready(
        self, monkeypatch, workspace
    ):
        """An embedding table with no disease rows scores every GNN term as zero,
        as surely as no table at all."""
        data_dir, checkpoint = workspace
        real = DiagnosisPipeline._precompute_node_embeddings

        def _without_diseases(self, device=None):
            real(self, device)
            self._node_embeddings.pop("disease")

        monkeypatch.setattr(
            DiagnosisPipeline, "_precompute_node_embeddings", _without_diseases
        )

        with pytest.raises(PipelineBuildError, match="disease"):
            DiagnosisPipeline(
                kg=KnowledgeGraph(), graph_data=_graph_data(data_dir),
                checkpoint_path=str(checkpoint), device="cpu",
            )

    def test_a_file_that_is_not_a_training_checkpoint_is_refused(
        self, workspace, tmp_path
    ):
        data_dir, _ = workspace
        not_a_checkpoint = tmp_path / "tensor.pt"
        torch.save(torch.zeros(3), not_a_checkpoint)

        with pytest.raises(PipelineBuildError, match="not a training checkpoint"):
            DiagnosisPipeline(
                kg=KnowledgeGraph(), graph_data=_graph_data(data_dir),
                checkpoint_path=str(not_a_checkpoint), device="cpu",
            )

    def test_a_checkpoint_without_weights_is_named_as_such(self, workspace, tmp_path):
        """A format problem, not a graph mismatch, and the message says which."""
        data_dir, checkpoint = workspace
        stripped = torch.load(checkpoint, map_location="cpu", weights_only=False)
        del stripped["state_dict"]
        torch.save(stripped, tmp_path / "stripped.pt")

        with pytest.raises(PipelineBuildError, match="carries no model weights"):
            DiagnosisPipeline(
                kg=KnowledgeGraph(), graph_data=_graph_data(data_dir),
                checkpoint_path=str(tmp_path / "stripped.pt"), device="cpu",
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

    def test_required_shortest_paths_with_no_directory_say_so(self, workspace):
        """With the graph passed in memory there is no table to load and no load
        message to point at; the refusal names that instead."""
        data_dir, checkpoint = workspace

        with pytest.raises(PipelineBuildError, match="no data_dir was given"):
            DiagnosisPipeline(
                kg=KnowledgeGraph(),
                config=PipelineConfig(sp_optional=False),
                graph_data=_graph_data(data_dir),
                checkpoint_path=str(checkpoint),
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
    pytest.importorskip("fastapi")
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


def _bad_checkpoint(kind, data_dir, tmp_path):
    if kind == "missing":
        return tmp_path / "absent.pt"
    if kind == "built for another graph":
        return _checkpoint_for_another_graph(data_dir, tmp_path / "other.pt")
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a checkpoint")
    return corrupt


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

    @pytest.mark.parametrize("kind", ["missing", "built for another graph", "unreadable"])
    def test_a_checkpoint_that_cannot_load_leaves_diagnose_at_503(
        self, api, monkeypatch, workspace, tmp_path, kind
    ):
        """The first two were the old fallback: startup published a pipeline
        scoring by path reasoning, and `/diagnose` answered 200. An unreadable
        checkpoint already failed startup; it is here so it stays that way."""
        data_dir, _ = workspace
        checkpoint = _bad_checkpoint(kind, data_dir, tmp_path)

        with self._start(api, monkeypatch, data_dir, checkpoint) as client:
            assert api.app_state.pipeline is None, "startup published a pipeline"
            response = _diagnose(client)

        assert response.status_code == 503, response.text
        assert "could not be initialized" in response.text
        assert api.app_state.pipeline is None

    def test_a_failed_build_is_not_repeated_by_every_diagnosis(
        self, api, monkeypatch, workspace, tmp_path
    ):
        """The lazy retry in `/diagnose` rebuilt the whole pipeline on each
        request -- graph, digests, tensors -- inside an async route, and answered
        the same 503. The environment cannot change in a running process, so
        the retry could only repeat the failure."""
        data_dir, _ = workspace
        builds = []
        real_build = api.build_pipeline

        def _counted(*args, **kwargs):
            builds.append(kwargs)
            return real_build(*args, **kwargs)

        monkeypatch.setattr(api, "build_pipeline", _counted)

        with self._start(api, monkeypatch, data_dir, tmp_path / "absent.pt") as client:
            statuses = [_diagnose(client).status_code for _ in range(3)]

        assert statuses == [503, 503, 503]
        assert len(builds) == 1, f"{len(builds)} builds for one startup and 3 requests"

    def test_a_checkpoint_with_no_knowledge_graph_is_refused_not_mocked(
        self, api, monkeypatch, workspace
    ):
        """`build_pipeline` returned None with no KG path before recording the
        request, so a deployment that named a checkpoint got the demo answer:
        invented candidates over HTTP 200."""
        from fastapi.testclient import TestClient

        data_dir, checkpoint = workspace
        monkeypatch.setenv("SHEPHERD_DATA_DIR", str(data_dir))
        monkeypatch.setenv("SHEPHERD_CHECKPOINT_PATH", str(checkpoint))

        with TestClient(api.app) as client:
            assert api.app_state.pipeline is None
            assert api.app_state.is_ready is False, "startup did not try the build"
            response = _diagnose(client)

        assert response.status_code == 503, response.text
        assert "mock" not in response.text

    def test_a_blank_checkpoint_setting_is_no_checkpoint(self, api, monkeypatch, workspace):
        """An exported-but-empty variable is how a shell says nothing. Read as a
        path, it would refuse a deployment that asked for no model."""
        data_dir, _ = workspace
        monkeypatch.setenv("SHEPHERD_CHECKPOINT_PATH", "")

        bundle = api.build_pipeline(
            kg_path=str(data_dir / "kg.json"), data_dir=str(data_dir)
        )

        assert bundle.checkpoint_path is None
        assert bundle.config["scoring_mode"] == "path_reasoning_fallback"


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
        assert ".." not in result.message
        self._assert_still_serving(api, before)

    def test_a_checkpoint_for_another_graph_is_refused(self, api, workspace, tmp_path):
        data_dir, checkpoint = workspace
        before = self._served(api, data_dir, checkpoint)
        other = _checkpoint_for_another_graph(data_dir, tmp_path / "other.pt")

        result = _reload(data_dir=str(data_dir), checkpoint_path=str(other))

        assert result.success is False
        assert "does not build a model" in result.message
        self._assert_still_serving(api, before)
