"""Serving's device rule: `auto` means CUDA, and without CUDA it refuses.

Contract M3c (`docs/working/PLAN_PROVENANCE_CONTRACT.md` §5.2), the serving half
of what `test_training_device.py` pins for training. Serving used to turn an
unset device into the CPU silently, after the workspace had been verified, the
checkpoint read and the model built, so a service asked to run on the
deployment's hardware ran somewhere else and said so only in a log line. Now:

- `resolve_serving_device` refuses `auto` (or unset) and any CUDA request
  without CUDA, naming the explicit development setting `SHEPHERD_DEVICE=cpu`;
- the pipeline settles the device before any workspace input or the checkpoint
  is opened, and `build_pipeline` before `kg.json` is read;
- a model supplied in memory and a path-only pipeline resolve nothing;
- the served identity records the device asked for and the one in effect apart;
- a reload refused for its device leaves the served pipeline serving.

Module: tests/unit/test_serving_device.py
"""
from __future__ import annotations

import asyncio
import inspect

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")

from src.inference.pipeline import (  # noqa: E402
    DiagnosisPipeline,
    PipelineBuildError,
    ServingDeviceUnavailable,
    resolve_serving_device,
)
from src.kg.graph import KnowledgeGraph  # noqa: E402
from tests.fixtures.opens import count_opens  # noqa: E402
from tests.fixtures.phenotype_kg import build_two_phenotype_kg  # noqa: E402
from tests.unit import test_serving_reads_once as reads_once  # noqa: E402

#: Shared with `test_serving_reads_once.py` rather than copied: `api` is the app
#: state of a fresh process (with SHEPHERD_DEVICE=cpu, which
#: `no_device_configured` removes), `workspace` a synthetic data directory and a
#: checkpoint trained on its graph.
api = reads_once.api
workspace = reads_once.workspace

#: What every refusal tells an operator to set for development.
REMEDY = "SHEPHERD_DEVICE=cpu"


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def with_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


@pytest.fixture
def no_device_configured(api, monkeypatch):
    """The service as a deployment that names no device starts it."""
    monkeypatch.delenv("SHEPHERD_DEVICE", raising=False)
    return api


def _workspace_pipeline(data_dir, checkpoint, device):
    return DiagnosisPipeline(
        graph=KnowledgeGraph.read_json(data_dir / "kg.json"),
        data_dir=str(data_dir),
        checkpoint_path=str(checkpoint),
        device=device,
    )


def _opened_inputs(opens, data_dir, checkpoint):
    """Every workspace file and the checkpoint that was opened at least once."""
    return [path for path in [*sorted(data_dir.iterdir()), checkpoint]
            if path.is_file() and opens.opens(path)]


# ---------------------------------------------------------------------------
# 1. The rule
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("requested", [None, "", "  ", "auto", "cuda", "cuda:1"])
def test_a_cuda_request_refuses_without_cuda(no_cuda, requested):
    """Unset is `auto`, and `auto` is CUDA. Catches the silent CPU fallback
    coming back, and a refusal that does not say how to ask for the CPU."""
    with pytest.raises(ServingDeviceUnavailable, match="requires CUDA") as refused:
        resolve_serving_device(requested)

    assert REMEDY in str(refused.value)
    assert isinstance(refused.value, PipelineBuildError), (
        "the service reports a refused build through PipelineBuildError"
    )


def test_explicit_cpu_is_permitted(no_cuda):
    assert resolve_serving_device("cpu") == "cpu"
    assert resolve_serving_device(" cpu ") == "cpu"


@pytest.mark.parametrize("requested, resolved", [
    (None, "cuda"), ("", "cuda"), ("auto", "cuda"), ("cuda", "cuda"), ("cuda:1", "cuda:1"),
])
def test_auto_means_cuda_where_cuda_exists(with_cuda, requested, resolved):
    """A literal "auto" used to reach torch and fail there; it is the same
    request as unset."""
    assert resolve_serving_device(requested) == resolved


def test_another_device_string_is_handed_to_torch_as_given(no_cuda):
    """As training's rule does: the rule decides CUDA and CPU, nothing else."""
    assert resolve_serving_device("mps") == "mps"


# ---------------------------------------------------------------------------
# 2-5. The pipeline
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", [None, "auto", "cuda"])
def test_the_pipeline_refuses_before_any_input_is_opened(no_cuda, workspace, device):
    """The device is settled first, as training's is. Catches the refusal
    moving back behind the workspace, the checkpoint or the model build, where
    a multi-gigabyte deployment pays for all of them before being told no."""
    data_dir, checkpoint = workspace
    graph = KnowledgeGraph.read_json(data_dir / "kg.json")

    with count_opens() as opens:
        with pytest.raises(ServingDeviceUnavailable, match=REMEDY):
            DiagnosisPipeline(graph=graph, data_dir=str(data_dir),
                              checkpoint_path=str(checkpoint), device=device)

    assert _opened_inputs(opens, data_dir, checkpoint) == []


def test_a_model_supplied_in_memory_stays_on_its_own_device(no_cuda, workspace):
    """Nothing is resolved for it, so no CUDA is needed. The identity says no
    device was asked for and names the one its parameters are on: catches a
    refusal of a caller who never asked for CUDA, and an identity that invents
    a request."""
    from src.kg.storage.file_storage import read_graph_artifacts
    from src.models.gnn.shepherd_gnn import build_shepherd_model

    data_dir, checkpoint = workspace
    graph_data = read_graph_artifacts(data_dir, map_location="cpu").graph_data
    model = build_shepherd_model(
        torch.load(checkpoint, map_location="cpu", weights_only=True), graph_data
    )

    pipeline = DiagnosisPipeline(kg=KnowledgeGraph(), model=model, graph_data=graph_data)

    identity = pipeline.served_identity()
    assert identity.scoring_mode == "gnn_only"
    assert identity.checkpoint.state == "supplied_in_memory"
    assert identity.settings.device_requested is None
    assert identity.settings.device == str(next(model.parameters()).device) == "cpu"


@pytest.mark.parametrize("device", [None, "auto", "cuda"])
def test_a_path_only_pipeline_resolves_no_device(no_cuda, device):
    """No model, so nothing runs on a device and nothing is refused; the
    identity states no device either way."""
    pipeline = DiagnosisPipeline(kg=build_two_phenotype_kg(), device=device)

    settings = pipeline.served_identity().settings
    assert (settings.device, settings.device_requested, settings.eta) == (None, None, None)


def test_the_identity_keeps_the_request_apart_from_the_device_in_effect(
    no_cuda, workspace, monkeypatch
):
    """A checkpoint loaded with no device named is an `auto` request, and the
    device in effect is read from the model. With the rule allowing it here,
    the two differ: catches an identity that records one as the other."""
    import src.inference.pipeline as pipeline_module

    data_dir, checkpoint = workspace
    asked = []

    def allow_on_cpu(requested):
        asked.append(requested)
        return "cpu"

    monkeypatch.setattr(pipeline_module, "resolve_serving_device", allow_on_cpu)

    pipeline = _workspace_pipeline(data_dir, checkpoint, device=None)

    assert asked[0] is None, "the rule saw the request as given"
    settings = pipeline.served_identity().settings
    assert settings.device_requested == "auto"
    assert settings.device == "cpu"
    assert pipeline.get_pipeline_config()["checkpoint_meta"]["device"] == "cpu"


def test_an_explicit_cpu_request_is_recorded_as_asked(no_cuda, workspace):
    data_dir, checkpoint = workspace

    settings = _workspace_pipeline(data_dir, checkpoint, device="cpu").served_identity().settings

    assert (settings.device_requested, settings.device) == ("cpu", "cpu")


def test_the_pipeline_picks_no_device_on_its_own():
    """The silent fallback (`"cuda" if torch.cuda.is_available() else "cpu"`)
    lived in `_load_model_from_checkpoint`. Only `resolve_serving_device` asks
    torch about CUDA now; catches a method that chooses a device again."""
    import src.inference.pipeline as pipeline_module

    assert "torch.cuda.is_available" not in inspect.getsource(
        DiagnosisPipeline._load_model_from_checkpoint
    )
    assert "cuda.is_available" not in inspect.getsource(DiagnosisPipeline)
    assert "torch.cuda.is_available" in inspect.getsource(pipeline_module.resolve_serving_device)


# ---------------------------------------------------------------------------
# 6. The service's build
# ---------------------------------------------------------------------------
def _served_fields(api):
    state = api.app_state
    return {name: getattr(state, name)
            for name in ("pipeline", "kg", "_current_data_dir", "_current_checkpoint_path",
                         "is_ready")}


def test_the_service_build_refuses_before_the_graph_is_read(
    no_cuda, no_device_configured, workspace, monkeypatch
):
    """A deployment that names a checkpoint and no device asks for CUDA. The
    refusal comes before `kg.json` is parsed and changes nothing served:
    catches a check placed after the graph load, or one that publishes."""
    api = no_device_configured
    data_dir, checkpoint = workspace
    serving = DiagnosisPipeline(kg=build_two_phenotype_kg())
    monkeypatch.setattr(api.app_state, "pipeline", serving)
    before = _served_fields(api)

    def read_json(cls, path):
        raise AssertionError(f"{path} was read before the device was settled")

    monkeypatch.setattr(KnowledgeGraph, "read_json", classmethod(read_json))

    with count_opens() as opens:
        with pytest.raises(ServingDeviceUnavailable, match=REMEDY):
            api.build_pipeline(kg_path=str(data_dir / "kg.json"),
                               checkpoint_path=str(checkpoint), data_dir=str(data_dir))

    assert _opened_inputs(opens, data_dir, checkpoint) == []
    assert _served_fields(api) == before
    assert api.app_state.pipeline is serving


def test_the_service_build_on_the_cpu_is_asked_for_explicitly(
    no_cuda, no_device_configured, workspace, monkeypatch
):
    api = no_device_configured
    data_dir, checkpoint = workspace
    monkeypatch.setenv("SHEPHERD_DEVICE", "cpu")
    before = _served_fields(api)

    bundle = api.build_pipeline(kg_path=str(data_dir / "kg.json"),
                                checkpoint_path=str(checkpoint), data_dir=str(data_dir))

    settings = bundle.config["served_identity"]["settings"]
    assert (settings["device_requested"], settings["device"]) == ("cpu", "cpu")
    assert _served_fields(api) == before, "a build publishes nothing"


def test_a_service_with_no_checkpoint_needs_no_device(no_cuda, no_device_configured, workspace):
    """Path reasoning alone runs no model, so an unset device refuses nothing."""
    data_dir, _ = workspace

    bundle = no_device_configured.build_pipeline(kg_path=str(data_dir / "kg.json"),
                                                 data_dir=str(data_dir))

    assert bundle.config["served_identity"]["scoring_mode"] == "path_reasoning_fallback"


# ---------------------------------------------------------------------------
# 7. The reload route
# ---------------------------------------------------------------------------
def _reload(data_dir, checkpoint, device):
    from src.api.routes.pipeline import PipelineReloadRequest, reload_pipeline

    return asyncio.run(reload_pipeline(PipelineReloadRequest(
        data_dir=str(data_dir), checkpoint_path=str(checkpoint), device=device,
    )))


@pytest.fixture
def serving(no_device_configured, monkeypatch):
    """A pipeline already being served when the reload arrives."""
    pipeline = DiagnosisPipeline(kg=build_two_phenotype_kg())
    monkeypatch.setattr(no_device_configured.app_state, "pipeline", pipeline)
    return pipeline


@pytest.mark.parametrize("device", [None, "auto", "cuda"])
def test_a_reload_that_asks_for_cuda_without_it_is_refused_and_keeps_serving(
    no_cuda, no_device_configured, serving, workspace, device
):
    """The refusal reaches the operator with the remedy, and the pipeline that
    was serving still is: catches a device failure that tears down the served
    pipeline, or one reported as a crash."""
    api = no_device_configured
    data_dir, checkpoint = workspace

    result = _reload(data_dir, checkpoint, device)

    assert result.success is False
    assert REMEDY in result.message
    assert "still being served" in result.message
    assert api.app_state.pipeline is serving
    assert result.status.served_identity.model_dump() == serving.served_identity().to_dict()


def test_a_reload_on_the_cpu_succeeds_and_records_the_request(
    no_cuda, no_device_configured, serving, workspace
):
    api = no_device_configured
    data_dir, checkpoint = workspace

    result = _reload(data_dir, checkpoint, "cpu")

    assert result.success, result.message
    assert api.app_state.pipeline is not serving
    settings = result.status.served_identity.settings
    assert (settings.device_requested, settings.device) == ("cpu", "cpu")


def test_a_reload_with_no_device_falls_back_to_the_deployments_setting(
    no_cuda, no_device_configured, serving, workspace, monkeypatch
):
    """Null defers to SHEPHERD_DEVICE, as the request's field says."""
    data_dir, checkpoint = workspace
    monkeypatch.setenv("SHEPHERD_DEVICE", "cpu")

    result = _reload(data_dir, checkpoint, None)

    assert result.success, result.message
    assert result.status.served_identity.settings.device_requested == "cpu"


@pytest.mark.parametrize("device", [None, "auto"])
def test_a_reload_refused_for_its_device_reads_no_candidate_checkpoint(
    no_cuda, no_device_configured, serving, workspace, device
):
    """Automatic selection reads every candidate in the run layout to choose
    one. A device this host cannot provide refuses the reload whatever is
    chosen, so the refusal comes first: catches the check left inside
    `build_pipeline` alone, after the scan has read each checkpoint."""
    import shutil

    from src.api.routes.pipeline import PipelineReloadRequest, reload_pipeline

    api = no_device_configured
    data_dir, checkpoint = workspace
    run = data_dir / "runs" / "gat" / "20261015-000000_run_aaaaaa" / "checkpoints"
    run.mkdir(parents=True)
    candidate = run / "model.pt"
    shutil.copyfile(checkpoint, candidate)

    with count_opens() as opens:
        result = asyncio.run(reload_pipeline(PipelineReloadRequest(
            data_dir=str(data_dir), device=device,
        )))

    assert result.success is False
    assert REMEDY in result.message
    assert not opens.opens(candidate), "the scan read a candidate before the device refused"
    assert api.app_state.pipeline is serving
