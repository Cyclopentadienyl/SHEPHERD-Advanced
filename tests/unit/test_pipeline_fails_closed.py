"""A requested model is served, or nothing is.

`_load_model_from_checkpoint` returned None when the checkpoint was missing or
did not build over the graph, and `_init_gnn_inference` returned without the GNN
when that happened or when there was no graph to compute its embeddings from.
The pipeline was then published as a success, and every diagnosis was answered
with path-reasoning scores in the model's place, over HTTP 200, with nothing in
the response saying so. (An unreadable checkpoint already raised, as torch's own
error; its cases here hold it there.)

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

**No model source at all is the same refusal since K7** (PATH1, G3): a graph
is never served by path reasoning, and no model is handed over in memory.

**What is served is a trained task model** (K8-2a): every test here whose build
reaches the model serves the session's disease child (`child`), and "another
graph" is that child with its graph view record naming another view. The
synthetic workspace stays only where the build is refused before a model is
built. **A checkpoint without a model spec is refused at load, before any model
is built** (K8-4, SG1), by the build, the reload with or without a task, and
startup alike; a pre-contract file is refused for its missing record first.

Module: tests/unit/test_pipeline_fails_closed.py
"""
from __future__ import annotations

import asyncio
import hashlib

import pytest

torch = pytest.importorskip("torch")

from src.inference import pipeline as pipeline_module  # noqa: E402
from src.inference.pipeline import (  # noqa: E402
    DiagnosisPipeline,
    PipelineBuildError,
)
from src.kg.graph import KnowledgeGraph  # noqa: E402
from tests.fixtures.synthetic_workspace import build_workspace  # noqa: E402
from tests.unit import test_serving_reads_once as reads_once  # noqa: E402


def _checkpoint_for_another_graph(checkpoint, path):
    """A sound task model, built on another graph's view.

    `checkpoint` (the disease child) with its graph view record naming a view of
    the same shape over another graph. Everything about it is well formed: it
    loads, its model spec matches its weights, and it would serve the graph whose
    view it records. It does not build over this one. It records this
    workspace's graph roles, so the build is what refuses it, not the training
    record (M2.4, whose refusals `test_training_record.py` shows). Its record
    names that view the same way wherever it names it (the top level, the data
    roles and the split), as a model trained on it would, so it does not
    contradict itself (K8-4 review P2-2).
    """
    contents = torch.load(checkpoint, map_location="cpu", weights_only=False)
    other = hashlib.sha256(b"another graph").hexdigest()
    roles = contents["data_roles"]
    contents["graph_view"] = {**contents["graph_view"], "digest": other}
    roles["graph_view"] = {**roles["graph_view"], "digest": other}
    roles["split"] = {**roles["split"], "view_digest": other}
    torch.save(contents, path)
    return path


@pytest.fixture
def workspace(tmp_path):
    return build_workspace(tmp_path)


#: The session's trained disease child, ``(data_dir, checkpoint)``: the model
#: every build here that reaches one serves. Read-only; what a test writes goes
#: to `tmp_path`. Shared with `test_serving_reads_once.py` rather than copied.
child = reads_once.child


# ==============================================================================
# The pipeline: a requested model is built, or the build raises
# ==============================================================================
class TestARequestedModelIsBuiltOrRefused:
    def test_a_sound_checkpoint_builds_its_model(self, child):
        """The control. Without it, every refusal below could be a harness that
        never builds anything. A disease child scores as its task does."""
        data_dir, checkpoint = child

        pipeline = DiagnosisPipeline(
            graph=KnowledgeGraph.read_json(data_dir / "kg.json"), data_dir=str(data_dir),
            checkpoint_path=str(checkpoint), device="cpu",
        )

        config = pipeline.get_pipeline_config()
        assert config["gnn_ready"] is True
        assert config["scoring_mode"] == "disease_task"

    def test_a_missing_checkpoint_is_refused(self, workspace, tmp_path):
        data_dir, _ = workspace
        absent = tmp_path / "absent.pt"

        with pytest.raises(PipelineBuildError, match="does not exist"):
            DiagnosisPipeline(
                graph=KnowledgeGraph.read_json(data_dir / "kg.json"), data_dir=str(data_dir),
                checkpoint_path=str(absent), device="cpu",
            )

    def test_an_unreadable_checkpoint_is_refused(self, workspace, tmp_path):
        data_dir, _ = workspace
        corrupt = tmp_path / "corrupt.pt"
        corrupt.write_bytes(b"not a checkpoint")

        with pytest.raises(PipelineBuildError, match="could not be read") as refused:
            DiagnosisPipeline(
                graph=KnowledgeGraph.read_json(data_dir / "kg.json"), data_dir=str(data_dir),
                checkpoint_path=str(corrupt), device="cpu",
            )

        assert str(corrupt) in str(refused.value)
        assert refused.value.__cause__ is not None, "the reader's own error is kept"

    def test_a_checkpoint_for_another_graph_is_refused(self, child, tmp_path):
        data_dir, checkpoint = child
        other = _checkpoint_for_another_graph(checkpoint, tmp_path / "other.pt")

        with pytest.raises(PipelineBuildError, match="built on graph view"):
            DiagnosisPipeline(
                graph=KnowledgeGraph.read_json(data_dir / "kg.json"), data_dir=str(data_dir),
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
            lambda self, checkpoint_path, device, *, expected_sha256=None, within=None: None,
        )

        with pytest.raises(PipelineBuildError, match="no node embeddings"):
            DiagnosisPipeline(
                graph=KnowledgeGraph.read_json(data_dir / "kg.json"), data_dir=str(data_dir),
                checkpoint_path=str(checkpoint), device="cpu",
            )

    def test_embeddings_without_a_node_type_scoring_reads_cannot_mark_it_ready(
        self, monkeypatch, child
    ):
        """Vectors computed without what scoring reads (a task model's view
        vectors) leave no candidate scoreable, as surely as no vectors at all."""
        data_dir, checkpoint = child
        real = DiagnosisPipeline._precompute_node_embeddings

        def _without_the_view(self, device=None):
            real(self, device)
            self._bundle_z = None

        monkeypatch.setattr(
            DiagnosisPipeline, "_precompute_node_embeddings", _without_the_view
        )

        with pytest.raises(PipelineBuildError, match="no node embeddings for \\['the view'\\]"):
            DiagnosisPipeline(
                graph=KnowledgeGraph.read_json(data_dir / "kg.json"), data_dir=str(data_dir),
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
                graph=KnowledgeGraph.read_json(data_dir / "kg.json"), data_dir=str(data_dir),
                checkpoint_path=str(not_a_checkpoint), device="cpu",
            )

    def test_a_checkpoint_without_weights_is_named_as_such(self, child, tmp_path):
        """A format problem, not a graph mismatch, and the message says which."""
        data_dir, checkpoint = child
        stripped = torch.load(checkpoint, map_location="cpu", weights_only=False)
        del stripped["state_dict"]
        torch.save(stripped, tmp_path / "stripped.pt")

        with pytest.raises(PipelineBuildError, match="holds no model weights"):
            DiagnosisPipeline(
                graph=KnowledgeGraph.read_json(data_dir / "kg.json"), data_dir=str(data_dir),
                checkpoint_path=str(tmp_path / "stripped.pt"), device="cpu",
            )

    def test_a_checkpoint_with_no_workspace_is_refused(self, workspace):
        data_dir, checkpoint = workspace

        with pytest.raises(PipelineBuildError, match="no workspace"):
            DiagnosisPipeline(
                graph=KnowledgeGraph.read_json(data_dir / "kg.json"), data_dir=None,
                checkpoint_path=str(checkpoint), device="cpu",
            )

    def test_without_torch_a_requested_model_is_refused(self, monkeypatch, workspace):
        data_dir, checkpoint = workspace
        monkeypatch.setattr(pipeline_module, "HAS_TORCH", False)

        with pytest.raises(PipelineBuildError, match="PyTorch is not available"):
            DiagnosisPipeline(graph=KnowledgeGraph.read_json(data_dir / "kg.json"),
                              data_dir=str(data_dir), checkpoint_path=str(checkpoint))

    def test_no_model_source_is_refused(self, workspace):
        """K7 (PATH1): B-2's question is settled. A graph with no checkpoint is
        refused, never served by path reasoning."""
        data_dir, _ = workspace

        with pytest.raises(PipelineBuildError, match="No model is configured"):
            DiagnosisPipeline(graph=KnowledgeGraph.read_json(data_dir / "kg.json"),
                              data_dir=str(data_dir), checkpoint_path=None, device="cpu")


# ==============================================================================
# The API: startup publishes nothing, a reload keeps what is serving
# ==============================================================================
@pytest.fixture
def api(monkeypatch):
    """App state as a fresh process has it, restored afterwards.

    `kg.json` in the synthetic workspace is a placeholder the manifest binds. The
    real reader parses it to an empty graph, and the check compares the digest
    of the bytes it parsed (contract M2.1, S6).
    """
    pytest.importorskip("fastapi")
    import src.api.main as api_main

    for field in ("pipeline", "kg", "_current_data_dir", "_current_checkpoint_path"):
        monkeypatch.setattr(api_main.app_state, field, None, raising=False)
    monkeypatch.setattr(api_main.app_state, "is_ready", False, raising=False)
    for name in ("SHEPHERD_KG_PATH", "SHEPHERD_DATA_DIR", "SHEPHERD_CHECKPOINT_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SHEPHERD_DEVICE", "cpu")
    return api_main


def _bad_checkpoint(kind, checkpoint, tmp_path):
    if kind == "missing":
        return tmp_path / "absent.pt"
    if kind == "built for another graph":
        return _checkpoint_for_another_graph(checkpoint, tmp_path / "other.pt")
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
        self, api, monkeypatch, child
    ):
        """The control for the case below: the same startup, a good checkpoint."""
        data_dir, checkpoint = child

        with self._start(api, monkeypatch, data_dir, checkpoint):
            assert api.app_state.pipeline is not None
            assert api.app_state.pipeline.get_pipeline_config()["gnn_ready"] is True

    @pytest.mark.parametrize("kind", ["missing", "built for another graph", "unreadable"])
    def test_a_checkpoint_that_cannot_load_leaves_diagnose_at_503(
        self, api, monkeypatch, child, tmp_path, kind
    ):
        """The first two were the old fallback: startup published a pipeline
        scoring by path reasoning, and `/diagnose` answered 200. An unreadable
        checkpoint already failed startup; it is here so it stays that way."""
        data_dir, sound = child
        checkpoint = _bad_checkpoint(kind, sound, tmp_path)

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
        the same 503. Trying again is now an explicit act -- a reload, or a
        restart -- rather than expensive I/O on every diagnosis."""
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

    def test_a_checkpoint_with_no_knowledge_graph_is_refused(
        self, api, monkeypatch, workspace
    ):
        """`build_pipeline` returned None with no KG path before recording the
        request, so a deployment that named a checkpoint was treated as one
        that configured nothing (then answered by the retired demo)."""
        from fastapi.testclient import TestClient

        data_dir, checkpoint = workspace
        monkeypatch.setenv("SHEPHERD_DATA_DIR", str(data_dir))
        monkeypatch.setenv("SHEPHERD_CHECKPOINT_PATH", str(checkpoint))

        with TestClient(api.app) as client:
            assert api.app_state.pipeline is None
            assert api.app_state.is_ready is False, "startup did not try the build"
            response = _diagnose(client)

        assert response.status_code == 503, response.text
        assert response.json()["error"].startswith("A pipeline is configured")

    def test_a_blank_checkpoint_setting_is_no_checkpoint(self, api, monkeypatch, workspace):
        """An exported-but-empty variable is how a shell says nothing: read as
        no checkpoint, never as a path to look for. With no model the build is
        refused for that (K7), not for a file named ""."""
        data_dir, _ = workspace
        monkeypatch.setenv("SHEPHERD_CHECKPOINT_PATH", "")

        with pytest.raises(PipelineBuildError, match="No model is configured") as refused:
            api.build_pipeline(kg_path=str(data_dir / "kg.json"), data_dir=str(data_dir))

        assert "does not exist" not in str(refused.value)


class TestAReloadKeepsWhatIsServing:
    """The candidate is built first and published only once complete, so a
    refusal reaches nothing that was serving."""

    @staticmethod
    def _served(api, data_dir, checkpoint):
        result = _reload(data_dir=str(data_dir), checkpoint_path=str(checkpoint))
        assert result.success is True, result.message
        config = api.app_state.pipeline.get_pipeline_config()
        assert config["gnn_ready"] is True
        # **What is served is read from the live pipeline** (contract M3c): app
        # state keeps no copy of it, so the snapshot takes it from there.
        assert config["served_identity"]["checkpoint"]["state"] == "verified"
        return {
            "app_state": {
                field: getattr(api.app_state, field)
                for field in ("pipeline", "kg",
                              "_current_data_dir", "_current_checkpoint_path")
            },
            "served_identity": config["served_identity"],
        }

    @staticmethod
    def _assert_still_serving(api, before):
        for field, value in before["app_state"].items():
            assert getattr(api.app_state, field) is value, (
                f"a refused reload changed app_state.{field}"
            )
        assert (
            api.app_state.pipeline.get_pipeline_config()["served_identity"]
            == before["served_identity"]
        ), "a refused reload changed what is served"

    def test_a_checkpoint_that_cannot_be_read_is_refused(self, api, child, tmp_path):
        data_dir, checkpoint = child
        before = self._served(api, data_dir, checkpoint)
        corrupt = tmp_path / "corrupt.pt"
        corrupt.write_bytes(b"not a checkpoint")

        result = _reload(data_dir=str(data_dir), checkpoint_path=str(corrupt))

        assert result.success is False
        assert "could not be read" in result.message
        assert "still being served" in result.message
        assert ".." not in result.message
        self._assert_still_serving(api, before)

    def test_a_checkpoint_for_another_graph_is_refused(self, api, child, tmp_path):
        data_dir, checkpoint = child
        before = self._served(api, data_dir, checkpoint)
        other = _checkpoint_for_another_graph(checkpoint, tmp_path / "other.pt")

        result = _reload(data_dir=str(data_dir), checkpoint_path=str(other))

        assert result.success is False
        assert "built on graph view" in result.message
        self._assert_still_serving(api, before)


def _without_model_spec(checkpoint, path):
    """`checkpoint` (the disease child) with its model spec deleted and its
    training record kept: a file this build cannot run, which names this
    workspace's graph, so its missing spec is the only reason to refuse it."""
    contents = torch.load(checkpoint, map_location="cpu", weights_only=True)
    del contents["model_spec"]
    torch.save(contents, path)
    return path


class TestAFileWithoutAModelSpecIsRefusedAtLoad:
    """README §8, regression 7 (K8-4, SG1): a checkpoint without a model spec is
    refused at load, with `read_model_record`'s own reason (`HISTORY_ONLY`),
    before any graph view, model or vector is built, by every serving entrance:
    the service's build, a reload with an explicit path with and without a
    task, and startup. The training record is still checked first, so a
    pre-contract file keeps its own reason. Before K8-4 the stripped child
    reached the ShepherdGNN build, and with a task the bundle path ran its
    forward pass before the task check refused it."""

    @staticmethod
    def _observe_builds(monkeypatch):
        """Each build step a load can take, noted, then made as asked: the
        bundle (`load_bundle` imports `build_bundle` when it is called), the
        ShepherdGNN factory, the graph view and the view vectors."""
        import src.kg.graph_view as graph_view
        import src.models.bundle.builder as builder
        import src.models.gnn.shepherd_gnn as shepherd_gnn

        built = []
        for owner, name in ((builder, "build_bundle"), (shepherd_gnn, "build_shepherd_model"),
                            (graph_view, "build_graph_view"),
                            (DiagnosisPipeline, "_precompute_node_embeddings")):
            def noted(*args, _name=name, _real=getattr(owner, name), **kwargs):
                built.append(_name)
                return _real(*args, **kwargs)

            monkeypatch.setattr(owner, name, noted)
        return built

    def test_the_build_refuses_it_before_any_model_is_built(self, monkeypatch, child, tmp_path):
        from src.inference.pipeline import build_served_pipeline
        from src.models.bundle.record import HISTORY_ONLY

        data_dir, checkpoint = child
        stripped = _without_model_spec(checkpoint, tmp_path / "stripped.pt")
        built = self._observe_builds(monkeypatch)

        with pytest.raises(PipelineBuildError) as refused:
            build_served_pipeline(data_dir / "kg.json", checkpoint_path=str(stripped),
                                  data_dir=str(data_dir), device="cpu")

        assert HISTORY_ONLY in str(refused.value)
        assert built == [], f"built before the refusal: {built}"

    def test_a_pre_contract_file_is_refused_for_its_record_first(self, monkeypatch, child,
                                                                  tmp_path):
        """No model spec and no training record: the record is checked first,
        so the reason is the record's (the order the playbook quotes)."""
        from src.inference.pipeline import build_served_pipeline
        from src.models.bundle.record import HISTORY_ONLY
        from tests.fixtures.registered_cohort import write_checkpoint_without_model_spec

        data_dir, _ = child
        old = tmp_path / "old.pt"
        write_checkpoint_without_model_spec(old)
        built = self._observe_builds(monkeypatch)

        with pytest.raises(PipelineBuildError,
                           match="records no training_input_digests") as refused:
            build_served_pipeline(data_dir / "kg.json", checkpoint_path=str(old),
                                  data_dir=str(data_dir), device="cpu")

        assert HISTORY_ONLY not in str(refused.value)
        assert built == []

    @pytest.mark.parametrize("task", [None, "disease"])
    def test_a_reload_with_an_explicit_path_refuses_it_and_keeps_what_is_serving(
            self, api, monkeypatch, child, tmp_path, task):
        from src.models.bundle.record import HISTORY_ONLY

        data_dir, checkpoint = child
        before = TestAReloadKeepsWhatIsServing._served(api, data_dir, checkpoint)
        stripped = _without_model_spec(checkpoint, tmp_path / "stripped.pt")
        built = self._observe_builds(monkeypatch)

        result = _reload(data_dir=str(data_dir), checkpoint_path=str(stripped),
                         **({} if task is None else {"task": task}))

        assert result.success is False
        assert HISTORY_ONLY in result.message and "still being served" in result.message
        assert built == [], f"built before the refusal: {built}"
        TestAReloadKeepsWhatIsServing._assert_still_serving(api, before)

    def test_startup_refuses_it_and_publishes_nothing(self, api, monkeypatch, child, tmp_path):
        data_dir, checkpoint = child
        stripped = _without_model_spec(checkpoint, tmp_path / "stripped.pt")
        built = self._observe_builds(monkeypatch)

        with TestStartupPublishesNothing()._start(api, monkeypatch, data_dir,
                                                  stripped) as client:
            assert api.app_state.pipeline is None, "startup published a pipeline"
            response = _diagnose(client)

        assert response.status_code == 503, response.text
        assert built == [], f"built before the refusal: {built}"


class TestReadinessFollowsWhatIsServed:
    """`/ready`, `/pipeline/status` and `/diagnose` agree on whether a pipeline
    is being served, configured or not (K7: with nothing configured, nothing
    answers either; that case is shown last).

    A startup whose build fails clears readiness, and M1 makes that the outcome
    of the commonest misconfigurations. The 503 tells the operator to reload,
    so a reload that succeeds has to restore everything a failed startup took
    away: before this, it restored the pipeline and the status while `/ready`
    stayed 503 until a restart, and anything routing on the probe kept the
    service out of rotation while it answered diagnoses.

    Driven through the real lifespan and HTTP routes.
    """

    @staticmethod
    def _start(api, monkeypatch, data_dir, checkpoint):
        from fastapi.testclient import TestClient

        from tests.fixtures.operator_access import OPERATOR_BASE_URL, operator_mode

        monkeypatch.setenv("SHEPHERD_KG_PATH", str(data_dir / "kg.json"))
        monkeypatch.setenv("SHEPHERD_DATA_DIR", str(data_dir))
        monkeypatch.setenv("SHEPHERD_CHECKPOINT_PATH", str(checkpoint))
        # The reload is an operator control (D8): mode M1, loopback requests.
        operator_mode(monkeypatch)
        return TestClient(api.app, base_url=OPERATOR_BASE_URL)

    @staticmethod
    def _reload(client, data_dir, checkpoint):
        response = client.post(
            "/api/v1/pipeline/reload",
            json={"data_dir": str(data_dir), "checkpoint_path": str(checkpoint)},
        )
        assert response.status_code == 200, response.text
        return response.json()

    @staticmethod
    def _answers(client):
        """What each surface says, side by side."""
        status = client.get("/api/v1/pipeline/status").json()
        return {
            "ready": client.get("/ready").status_code,
            "diagnose": _diagnose(client).status_code,
            "initialized": status["initialized"],
            "gnn_ready": status["gnn_ready"],
        }

    @staticmethod
    def _served_identity(client):
        """What is served, as `/pipeline/status` reports it (contract M3c); None
        while no pipeline is."""
        return client.get("/api/v1/pipeline/status").json()["served_identity"]

    NOT_SERVING = {"ready": 503, "diagnose": 503, "initialized": False, "gnn_ready": False}
    SERVING = {"ready": 200, "diagnose": 200, "initialized": True, "gnn_ready": True}

    def test_a_failed_startup_is_not_ready(self, api, monkeypatch, workspace, tmp_path):
        data_dir, _ = workspace

        with self._start(api, monkeypatch, data_dir, tmp_path / "absent.pt") as client:
            assert self._answers(client) == self.NOT_SERVING
            assert "reload" in client.get("/ready").json()["error"], (
                "the commonest way into this state is not told how to recover"
            )

    def test_a_refused_reload_after_it_stays_not_ready(
        self, api, monkeypatch, child, tmp_path
    ):
        data_dir, checkpoint = child
        other = _checkpoint_for_another_graph(checkpoint, tmp_path / "other.pt")

        with self._start(api, monkeypatch, data_dir, tmp_path / "absent.pt") as client:
            result = self._reload(client, data_dir, other)

            assert result["success"] is False
            assert "built on graph view" in result["message"]
            assert self._answers(client) == self.NOT_SERVING

    def test_a_successful_reload_after_it_restores_all_of_them(
        self, api, monkeypatch, child, tmp_path
    ):
        """And only once the candidate is published: readiness is not set when
        the reload starts building."""
        data_dir, checkpoint = child
        ready_while_building = []
        real_build = api.build_pipeline

        def _observed(*args, **kwargs):
            ready_while_building.append(api.app_state.is_ready)
            return real_build(*args, **kwargs)

        with self._start(api, monkeypatch, data_dir, tmp_path / "absent.pt") as client:
            monkeypatch.setattr(api, "build_pipeline", _observed)
            result = self._reload(client, data_dir, checkpoint)

            assert result["success"] is True, result["message"]
            assert ready_while_building == [False]
            assert self._answers(client) == self.SERVING
            assert client.get("/ready").json()["pipeline_loaded"] is True

    def test_a_refused_reload_on_a_healthy_service_keeps_all_of_it(
        self, api, monkeypatch, child, tmp_path
    ):
        data_dir, checkpoint = child
        corrupt = tmp_path / "corrupt.pt"
        corrupt.write_bytes(b"not a checkpoint")
        fields = ("pipeline", "kg",
                  "_current_data_dir", "_current_checkpoint_path", "is_ready")

        with self._start(api, monkeypatch, data_dir, checkpoint) as client:
            assert self._answers(client) == self.SERVING
            before = {field: getattr(api.app_state, field) for field in fields}
            # What is served, as the status reports it (contract M3c).
            identity = self._served_identity(client)

            result = self._reload(client, data_dir, corrupt)

            assert result["success"] is False
            assert "still being served" in result["message"]
            for field, value in before.items():
                assert getattr(api.app_state, field) is value, (
                    f"a refused reload changed app_state.{field}"
                )
            assert self._served_identity(client) == identity, (
                "a refused reload changed what is served"
            )
            assert self._answers(client) == self.SERVING

    def test_the_whole_way_back_in_one_lifetime(self, api, monkeypatch, child, tmp_path):
        """The acceptance as one chain, in one service: a refusal must leave
        nothing behind that stops a later reload from restoring the service,
        and the restored service must survive a later refusal."""
        data_dir, checkpoint = child
        other = _checkpoint_for_another_graph(checkpoint, tmp_path / "other.pt")
        fields = ("pipeline", "kg",
                  "_current_data_dir", "_current_checkpoint_path", "is_ready")

        with self._start(api, monkeypatch, data_dir, tmp_path / "absent.pt") as client:
            assert self._answers(client) == self.NOT_SERVING
            assert self._served_identity(client) is None

            refused = self._reload(client, data_dir, other)
            assert refused["success"] is False and "built on graph view" in refused["message"]
            assert self._answers(client) == self.NOT_SERVING
            assert self._served_identity(client) is None

            assert self._reload(client, data_dir, checkpoint)["success"] is True
            assert self._answers(client) == self.SERVING
            restored = {field: getattr(api.app_state, field) for field in fields}
            identity = self._served_identity(client)
            assert identity["checkpoint"]["state"] == "verified"

            refused = self._reload(client, data_dir, other)
            assert refused["success"] is False and "built on graph view" in refused["message"]
            assert self._answers(client) == self.SERVING
            for field, value in restored.items():
                assert getattr(api.app_state, field) is value, (
                    f"a refusal after recovery changed app_state.{field}"
                )
            assert self._served_identity(client) == identity, (
                "a refusal after recovery changed what is served"
            )

    def test_a_configured_graph_that_is_missing_is_not_ready(
        self, api, monkeypatch, tmp_path
    ):
        """`build_pipeline` records the request and returns None here rather
        than raising, so startup completes and sets readiness. `/diagnose`
        refuses; `/ready` reads the same definition and refuses too."""
        from fastapi.testclient import TestClient

        monkeypatch.setenv("SHEPHERD_KG_PATH", str(tmp_path / "absent_kg.json"))

        with TestClient(api.app) as client:
            assert self._answers(client) == self.NOT_SERVING
            assert "reload" in client.get("/ready").json()["error"]

    def test_a_service_with_nothing_configured_is_not_ready_and_says_why(
        self, api, tmp_path, monkeypatch
    ):
        """K7 (DEMO1): nothing configured is not ready, and no diagnosis is
        answered; the reason says nothing is loaded. A reload that names a
        workspace is a request for a real pipeline; refused, the reason says a
        configured pipeline could not be initialized. (The reload is an
        operator's, which D8 allows: mode M1, loopback requests.)"""
        from fastapi.testclient import TestClient

        from tests.fixtures.operator_access import OPERATOR_BASE_URL, operator_mode

        empty = tmp_path / "not_a_workspace"
        empty.mkdir()
        operator_mode(monkeypatch)

        with TestClient(api.app, base_url=OPERATOR_BASE_URL) as client:
            assert self._answers(client) == self.NOT_SERVING
            assert client.get("/ready").json()["error"].startswith("No model is loaded")

            refused = client.post("/api/v1/pipeline/reload", json={"data_dir": str(empty)})

            assert refused.json()["success"] is False
            assert self._answers(client) == self.NOT_SERVING
            assert client.get("/ready").json()["error"].startswith("A pipeline is configured")
