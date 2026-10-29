"""Serving's device rule: `auto` means CUDA, and without CUDA it refuses.

Contract M3c (`docs/working/PLAN_PROVENANCE_CONTRACT.md` §5.2), the serving half
of what `test_training_device.py` pins for training. Serving used to turn an
unset device into the CPU silently, after the workspace had been verified, the
checkpoint read and the model built, so a service asked to run on the
deployment's hardware ran somewhere else and said so only in a log line. Now:

- `resolve_serving_device` refuses `auto` (or unset) and any CUDA request
  without CUDA, naming the explicit development setting `SHEPHERD_DEVICE=cpu`;
  without PyTorch it names PyTorch instead, whatever was asked for;
- the pipeline settles the device before any workspace input or the checkpoint
  is opened, and `build_pipeline` before `kg.json` is read;
- a pipeline with no model is refused for that, before any device is asked
  about (K7: no graph is served without a model, no model is handed over in
  memory);
- the served identity records the device asked for and the one in effect apart,
  the second read from the model, and the service passes the request on as
  given;
- a blank request defers to `SHEPHERD_DEVICE`, and a blank setting is unset;
- a startup refused for its device answers 503;
- a reload refused for its device leaves the served pipeline serving.

Module: tests/unit/test_serving_device.py
"""
from __future__ import annotations

import asyncio
import inspect
import logging

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
from tests.unit import test_pipeline_fails_closed as fails_closed  # noqa: E402
from tests.unit import test_serving_reads_once as reads_once  # noqa: E402

#: Shared with `test_serving_reads_once.py` rather than copied: `api` is the app
#: state of a fresh process (with SHEPHERD_DEVICE=cpu, which
#: `no_device_configured` removes), `workspace` a synthetic data directory and a
#: checkpoint trained on its graph.
api = reads_once.api
workspace = reads_once.workspace

#: How `test_pipeline_fails_closed.py` starts the app through its lifespan with
#: a workspace configured, and what each surface answers; reused, not copied.
Readiness = fails_closed.TestReadinessFollowsWhatIsServed

#: What every refusal tells an operator to set for development.
REMEDY = "SHEPHERD_DEVICE=cpu"

#: What the refusal names on a host without PyTorch.
NO_TORCH = "PyTorch is not available"


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def with_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


@pytest.fixture
def no_torch(monkeypatch):
    """A host without PyTorch (a path-only image), as the pipeline module sees it."""
    import src.inference.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HAS_TORCH", False)


@pytest.fixture
def cpu_allowed(monkeypatch):
    """The rule allowing the CPU for any request, as a CUDA host allows CUDA, so
    what each caller passes on can be seen here. Returns the requests the rule
    saw.

    Patched on the pipeline module, where every caller looks it up:
    `build_pipeline` and the reload route import it from there when they run.
    """
    import src.inference.pipeline as pipeline_module

    asked = []

    def allow_on_cpu(requested):
        asked.append(requested)
        return "cpu"

    monkeypatch.setattr(pipeline_module, "resolve_serving_device", allow_on_cpu)
    return asked


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


@pytest.mark.parametrize("requested", [None, "", "auto", "cuda", "cpu"])
def test_without_pytorch_the_refusal_names_pytorch_not_the_cpu(no_torch, requested):
    """**The cause before the remedy.** Without PyTorch no device can be had:
    a CUDA refusal prescribing `SHEPHERD_DEVICE=cpu` sent the operator to a
    second refusal for the real reason, and `cpu` passed the rule only to be
    refused later. Catches the PyTorch check placed after the CUDA check again,
    or dropped from the rule."""
    with pytest.raises(PipelineBuildError, match=NO_TORCH) as refused:
        resolve_serving_device(requested)

    assert not isinstance(refused.value, ServingDeviceUnavailable), (
        "a missing PyTorch reported as a missing CUDA"
    )
    assert REMEDY not in str(refused.value)


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


@pytest.mark.parametrize("device", [None, "auto", "cuda"])
def test_no_model_is_refused_for_that_before_any_device_is_asked_about(
        no_cuda, workspace, device):
    """K7: a pipeline with no checkpoint is not served by path reasoning, and is
    refused for the missing model, before anything is opened. The operator
    hears the real cause first, never a CUDA refusal for a model nobody
    named."""
    data_dir, checkpoint = workspace
    graph = KnowledgeGraph.read_json(data_dir / "kg.json")

    with count_opens() as opens:
        with pytest.raises(PipelineBuildError, match="No model is configured") as refused:
            DiagnosisPipeline(graph=graph, data_dir=str(data_dir), checkpoint_path=None,
                              device=device)

    assert not isinstance(refused.value, ServingDeviceUnavailable)
    assert _opened_inputs(opens, data_dir, checkpoint) == []


def test_the_identity_keeps_the_request_apart_from_the_device_in_effect(
    no_cuda, workspace, cpu_allowed
):
    """A checkpoint loaded with no device named is an `auto` request, and the
    device in effect is read from the model. With the rule allowing it here,
    the two differ: catches an identity that records one as the other."""
    data_dir, checkpoint = workspace

    pipeline = _workspace_pipeline(data_dir, checkpoint, device=None)

    assert cpu_allowed[0] is None, "the rule saw the request as given"
    settings = pipeline.served_identity().settings
    assert settings.device_requested == "auto"
    assert settings.device == "cpu"
    assert pipeline.get_pipeline_config()["checkpoint_meta"]["device"] == "cpu"


def test_the_device_in_effect_is_read_from_the_model_not_the_request(no_cuda, workspace):
    """Torch names where a model's parameters are itself: a `cuda` request
    serves on `cuda:0`. Here `cpu:0` splits the same way, a request the rule
    passes through and torch accepts, on parameters torch names `cpu`. Catches
    the resolved request recorded as the device in effect, which on a CUDA host
    says `cuda` for `cuda:0`, and claims CUDA for a model left where it was."""
    data_dir, checkpoint = workspace
    assert str(torch.zeros(1, device="cpu:0").device) == "cpu", (
        "the split this test relies on: torch names a CPU tensor without an index"
    )

    pipeline = _workspace_pipeline(data_dir, checkpoint, device="cpu:0")

    settings = pipeline.served_identity().settings
    assert settings.device_requested == "cpu:0"
    assert settings.device == str(next(pipeline.model.parameters()).device) == "cpu"


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
    no_cuda, no_device_configured, workspace, monkeypatch, disease_pipeline
):
    """A deployment that names a checkpoint and no device asks for CUDA. The
    refusal comes before `kg.json` is parsed and changes nothing served:
    catches a check placed after the graph load, or one that publishes. It
    comes after the request for a real pipeline is recorded, which is what
    makes `/diagnose` refuse it as a configured pipeline that could not be
    built, without trying the build again: catches the check moved ahead of
    that record."""
    api = no_device_configured
    data_dir, checkpoint = workspace
    serving = disease_pipeline
    monkeypatch.setattr(api.app_state, "pipeline", serving)
    monkeypatch.setattr(api.app_state, "real_pipeline_requested", False)
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
    assert api.app_state.real_pipeline_requested is True, (
        "a deployment refused for its device was recorded as asking for nothing"
    )


def test_a_startup_refused_for_its_device_answers_503(
    no_cuda, no_device_configured, workspace, monkeypatch, caplog
):
    """A deployment that names a checkpoint and no device, on a host without
    CUDA, is refused at startup, and every surface says no pipeline is served:
    `/diagnose` and `/ready` answer 503, as for any configured pipeline that
    could not be built. Catches the device check moved ahead of the record that
    a real pipeline was asked for (a natural "settle the device first"
    refactor): `/diagnose` then retries the build on every request and says
    nothing was configured. Driven through the real lifespan and HTTP routes,
    as `TestReadinessFollowsWhatIsServed` is."""
    api = no_device_configured
    data_dir, checkpoint = workspace
    monkeypatch.setattr(api.app_state, "real_pipeline_requested", False)

    with caplog.at_level(logging.ERROR, logger=api.__name__):
        with Readiness._start(api, monkeypatch, data_dir, checkpoint) as client:
            answers = Readiness._answers(client)
            diagnosis = fails_closed._diagnose(client)
            ready = client.get("/ready")

    assert any("requires CUDA" in record.getMessage() for record in caplog.records), (
        "startup failed for another reason than its device"
    )
    assert answers == Readiness.NOT_SERVING
    assert "candidates" not in diagnosis.json(), "a diagnosis was answered"
    assert diagnosis.json()["error"].startswith("A pipeline is configured")
    assert ready.json()["error"].startswith("A pipeline is configured")
    assert api.app_state.pipeline is None


@pytest.mark.parametrize("configured", [None, "cpu"])
def test_without_pytorch_the_service_build_refuses_naming_pytorch(
    no_torch, no_device_configured, workspace, monkeypatch, configured
):
    """Unset or `cpu`, the deployment is told what is missing, before the
    graph is read. Catches the CUDA refusal and its CPU remedy answering first,
    and a `cpu` deployment that parses `kg.json` only for the pipeline to
    refuse it."""
    api = no_device_configured
    data_dir, checkpoint = workspace
    if configured is not None:
        monkeypatch.setenv("SHEPHERD_DEVICE", configured)

    with count_opens() as opens:
        with pytest.raises(PipelineBuildError, match=NO_TORCH) as refused:
            api.build_pipeline(kg_path=str(data_dir / "kg.json"),
                               checkpoint_path=str(checkpoint), data_dir=str(data_dir))

    assert not isinstance(refused.value, ServingDeviceUnavailable)
    assert REMEDY not in str(refused.value)
    assert _opened_inputs(opens, data_dir, checkpoint) == []


@pytest.mark.parametrize("requested", [None, "", "  "])
@pytest.mark.parametrize("configured, expected", [
    (None, None), ("", None), ("  ", None), (" cpu ", "cpu"), ("cuda:1", "cuda:1"),
])
def test_a_blank_request_defers_to_the_deployments_setting(
    no_device_configured, monkeypatch, requested, configured, expected
):
    """Blank is unset at both steps, as the rule treats it. Catches the
    fallback chosen by truthiness again, which let `"  "` stand as a request
    for `auto` past `SHEPHERD_DEVICE`, and a blank `SHEPHERD_DEVICE` stand as
    one; and a padded value passed on padded."""
    if configured is not None:
        monkeypatch.setenv("SHEPHERD_DEVICE", configured)

    assert no_device_configured.requested_serving_device(requested) == expected


def test_a_named_request_wins_over_the_deployments_setting(no_device_configured, monkeypatch):
    """The request is what was asked for, stripped as the rule strips it:
    catches the setting taking precedence, or `" cuda "` passed on padded."""
    monkeypatch.setenv("SHEPHERD_DEVICE", "cpu")

    assert no_device_configured.requested_serving_device(" cuda ") == "cuda"


def test_the_service_build_passes_the_request_on_as_given(
    no_device_configured, cpu_allowed, workspace
):
    """`build_pipeline` settles the device only to refuse early; the pipeline is
    handed the request, so a deployment that names no device is recorded as
    `auto`. Catches the resolved device passed on instead, which on a CUDA host
    records `cuda` for a deployment that asked for `auto`."""
    data_dir, checkpoint = workspace

    bundle = no_device_configured.build_pipeline(kg_path=str(data_dir / "kg.json"),
                                                 checkpoint_path=str(checkpoint),
                                                 data_dir=str(data_dir))

    assert cpu_allowed[0] is None, "the rule saw the request as given"
    settings = bundle.config["served_identity"]["settings"]
    assert (settings["device_requested"], settings["device"]) == ("auto", "cpu")


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


def test_a_service_with_no_checkpoint_is_refused_for_that(no_cuda, no_device_configured,
                                                          workspace):
    """K7 (PATH1): a graph named with no checkpoint (a `SHEPHERD_KG_PATH`-only
    start) is refused for the missing model, not for its device, and is
    recorded as a request, so the service then answers 503."""
    api = no_device_configured
    data_dir, _ = workspace
    api.app_state.real_pipeline_requested = False

    with pytest.raises(PipelineBuildError, match="No model is configured") as refused:
        api.build_pipeline(kg_path=str(data_dir / "kg.json"), data_dir=str(data_dir))

    assert not isinstance(refused.value, ServingDeviceUnavailable)
    assert api.app_state.real_pipeline_requested is True


# ---------------------------------------------------------------------------
# 7. The reload route
# ---------------------------------------------------------------------------
def _reload(data_dir, checkpoint, device):
    from src.api.routes.pipeline import PipelineReloadRequest, reload_pipeline

    return asyncio.run(reload_pipeline(PipelineReloadRequest(
        data_dir=str(data_dir), checkpoint_path=str(checkpoint), device=device,
    )))


@pytest.fixture
def serving(no_device_configured, monkeypatch, disease_pipeline):
    """A pipeline already being served when the reload arrives."""
    monkeypatch.setattr(no_device_configured.app_state, "pipeline", disease_pipeline)
    return disease_pipeline


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


@pytest.mark.parametrize("device", [None, "", "  "])
def test_a_reload_with_no_device_falls_back_to_the_deployments_setting(
    no_cuda, no_device_configured, serving, workspace, monkeypatch, device
):
    """Null defers to SHEPHERD_DEVICE, as the request's field says, and a blank
    device is null: catches `"  "` read as a request for `auto`, refused here
    while `""` served on the CPU."""
    data_dir, checkpoint = workspace
    monkeypatch.setenv("SHEPHERD_DEVICE", "cpu")

    result = _reload(data_dir, checkpoint, device)

    assert result.success, result.message
    assert result.status.served_identity.settings.device_requested == "cpu"


def test_a_reload_passes_the_request_on_as_given(
    no_device_configured, cpu_allowed, serving, workspace
):
    """The route settles the device only to refuse before its scan, as
    `build_pipeline` does: a reload that names no device, on a deployment that
    names none, is recorded as `auto`. Catches the resolved device passed on by
    either, which on a CUDA host records `cuda` for an `auto` request."""
    data_dir, checkpoint = workspace

    result = _reload(data_dir, checkpoint, None)

    assert result.success, result.message
    settings = result.status.served_identity.settings
    assert (settings.device_requested, settings.device) == ("auto", "cpu")


def test_without_pytorch_a_reload_is_refused_naming_pytorch_and_keeps_serving(
    no_torch, no_device_configured, serving, workspace
):
    """The route's early check refuses with the cause, as a refused reload:
    catches that check catching the CUDA refusal alone, which let the PyTorch
    refusal escape the route as a server error, and the CUDA refusal answering
    first with its CPU remedy."""
    api = no_device_configured
    data_dir, checkpoint = workspace

    result = _reload(data_dir, checkpoint, None)

    assert result.success is False
    assert NO_TORCH in result.message
    assert REMEDY not in result.message
    assert "still being served" in result.message
    assert api.app_state.pipeline is serving


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
