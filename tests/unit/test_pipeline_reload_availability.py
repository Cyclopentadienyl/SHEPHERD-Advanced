"""A refused reload must not cost the clinic the pipeline it was already serving.

`POST /pipeline/reload` released the running pipeline first and built the
replacement second. Everything that can reject a workspace -- a crossed graph
source, a replaced artifact, an unsupported manifest schema, an unreadable
checkpoint, a configuration that fails to read back -- is discovered during that
second step, so each of those rejections landed on a service that had already
discarded a working pipeline. Pointing the reload at the wrong directory, which
is an operator typo, took the diagnosis service down until someone restarted it.

The rule is the same one startup already followed, and it is a rule about
*order*, not about recovery: build the candidate into local variables, and swap
references only once it is complete. Nothing is undone, so there is no rollback
path to get wrong.

Its reporting half matters too. Every failure used to answer `initialized=False`
-- true only because the teardown had already happened. With the teardown gone,
that answer would be a fresh lie, telling an operator the service is down while
it is still diagnosing patients.

Module: tests/unit/test_pipeline_reload_availability.py
"""
from __future__ import annotations

import asyncio
import copy

import pytest

pytest.importorskip("torch")

from src.api.routes.pipeline import (  # noqa: E402
    PipelineReloadRequest,
    _live_status,
    reload_pipeline,
)
from tests.unit.test_graph_artifact_binding import (  # noqa: E402
    TestTheClinicalPathIsAGraphConsumer,
    _workspace,
)


def _served_identity(digit: str, *, with_sp: bool) -> dict:
    """A coherent served identity (contract M3c) whose digests are `digit`s.

    Built through `src.core.types.ServedIdentity`, so its cross-part invariants
    hold, and given as the JSON form a pipeline's configuration reports. The two
    pipelines below differ in every digest, so a check that compares them cannot
    pass by accident.
    """
    from src.core.types import (
        ServedCheckpoint,
        ServedDatasetRecord,
        ServedGraph,
        ServedIdentity,
        ServedSettings,
        ServedShortestPaths,
    )

    return ServedIdentity(
        scoring_mode="gnn_plus_shortest_path" if with_sp else "gnn_only",
        checkpoint=ServedCheckpoint(state="verified", sha256=digit * 64),
        graph=ServedGraph(state="verified", kg_sha256=(digit + "1") * 32,
                          split_manifest_sha256=(digit + "2") * 32),
        shortest_paths=(
            ServedShortestPaths(state="verified", build_id=f"build-{digit}", max_hops=3)
            if with_sp else ServedShortestPaths(state="absent")
        ),
        dataset_record=ServedDatasetRecord(state="verified", origin="synthetic",
                                           sha256=(digit + "3") * 32),
        settings=ServedSettings(
            device="cpu", device_requested="cpu", eta=0.7 if with_sp else 1.0,
            max_path_length=4, path_length_penalty=0.9,
            aggregation_method="weighted_sum", max_paths_per_candidate=10,
            use_direct_path_optimization=True, phenotype_normalisation_version=1,
        ),
    ).to_dict()


# **What is served is identified by the pipeline, not by app state** (contract
# M3c): the version string these configurations carried, and the copy of it
# app state kept, are gone. Each carries a served identity instead.
SERVED_CONFIG = {
    "served_identity": _served_identity("a", with_sp=True),
    "scoring_mode": "served-mode",
    "gnn_ready": True,
    "sp_ready": True,
    "kg_nodes": 11,
    "kg_edges": 22,
    "has_model": True,
}
CANDIDATE_CONFIG = {
    "served_identity": _served_identity("b", with_sp=False),
    "scoring_mode": "candidate-mode",
    "gnn_ready": True,
    "sp_ready": False,
    "kg_nodes": 33,
    "kg_edges": 44,
    "has_model": True,
}


def _candidate_config(**changes):
    """`CANDIDATE_CONFIG` with `changes`, built anew on every call, as a real
    pipeline builds its configuration.

    **Never the constant itself.** Handed out through a shallow copy, its nested
    identity would be the constant's, so a reload that changed it in place would
    change what every later test compares with, and a check would then pass or
    fail by test order.
    """
    return {**copy.deepcopy(CANDIDATE_CONFIG), **changes}


class _ServedPipeline:
    """Stands in for the pipeline already answering clinical requests.

    **Its configuration is its own**, a deep copy of `SERVED_CONFIG` handed out
    shallowly, so a reload that changed what is served in place would change it
    here, where the status route reads it, and not in the constant a test
    compares with.
    """

    def __init__(self):
        self._config = copy.deepcopy(SERVED_CONFIG)

    def get_pipeline_config(self):
        return dict(self._config)


#: The app-state fields that describe the served pipeline, all of which
#: `publish_pipeline` writes.
_PUBLISHED_FIELDS = ("pipeline", "kg", "_current_data_dir", "_current_checkpoint_path")


def _serving(monkeypatch, data_dir="/served/workspace", checkpoint="/served/ckpt.pt"):
    """Put a healthy pipeline in app state and return a snapshot of every field
    that describes it, with a deep copy of the identity it serves."""
    import src.api.main as api

    served = _ServedPipeline()
    kg = object()
    monkeypatch.setattr(api.app_state, "pipeline", served, raising=False)
    monkeypatch.setattr(api.app_state, "kg", kg, raising=False)
    monkeypatch.setattr(api.app_state, "_current_data_dir", data_dir, raising=False)
    monkeypatch.setattr(
        api.app_state, "_current_checkpoint_path", checkpoint, raising=False
    )
    return api, {
        "pipeline": served,
        "kg": kg,
        "_current_data_dir": data_dir,
        "_current_checkpoint_path": checkpoint,
        "served_identity": copy.deepcopy(served.get_pipeline_config()["served_identity"]),
    }


def _assert_untouched(api, before):
    for field in _PUBLISHED_FIELDS:
        value = before[field]
        assert getattr(api.app_state, field) is value or getattr(
            api.app_state, field
        ) == value, f"reload changed app_state.{field} on a failed reload"
    # **What is served, as the status route reports it** (contract M3c): read
    # off the live pipeline through the strict response model, against the deep
    # copy taken when serving began. Compared with the module constant it was
    # shallow-copied from, the identity was the constant's own dictionary, so no
    # change a reload made to it could make the two differ.
    assert (
        _live_status().served_identity.model_dump() == before["served_identity"]
    ), "a failed reload changed what is served"


def _reload(**kwargs):
    return asyncio.run(reload_pipeline(PipelineReloadRequest(**kwargs)))


def _candidate_workspace(monkeypatch, tmp_path, name="candidate"):
    """A workspace the reload endpoint will accept as far as construction.

    `SHEPHERD_DEVICE=cpu`: a reload or startup that names no device asks for
    CUDA and, on a host without it, is refused before the workspace is read
    (contract M3c, §5.2). Every refusal these tests are about comes after that.
    """
    reached: list = []
    monkeypatch.setenv("SHEPHERD_DEVICE", "cpu")
    TestTheClinicalPathIsAGraphConsumer._pipeline_module(monkeypatch, reached)
    root = _workspace(tmp_path / name)
    (root / "ckpt.pt").write_bytes(b"weights")
    return root, reached


# --------------------------------------------------------------------------
# A rejected candidate leaves the served pipeline in place
# --------------------------------------------------------------------------
class TestARejectedCandidateCostsNothing:
    def test_a_crossed_graph_source_is_refused_and_changes_nothing(
        self, monkeypatch, tmp_path
    ):
        """This endpoint cannot cross two workspaces by composition -- it derives
        `kg_path` from `data_dir`, so the two always name the same directory.
        What it can be handed is a directory whose `kg.json` is another
        workspace's, which is what copying a graph file over a workspace
        produces, and which the manifest binding refuses on the bytes."""
        api, before = _serving(monkeypatch)
        other, _ = _candidate_workspace(monkeypatch, tmp_path, "other")
        root, _ = _candidate_workspace(monkeypatch, tmp_path, "candidate")
        (root / "kg.json").write_bytes((other / "kg.json").read_bytes())

        result = _reload(data_dir=str(root), checkpoint_path=str(root / "ckpt.pt"))

        assert result.success is False
        assert "is not the graph" in result.message
        _assert_untouched(api, before)

    def test_a_replaced_graph_artifact_is_refused_and_changes_nothing(
        self, monkeypatch, tmp_path
    ):
        api, before = _serving(monkeypatch)
        other, _ = _candidate_workspace(monkeypatch, tmp_path, "other")
        root, _ = _candidate_workspace(monkeypatch, tmp_path)
        (root / "node_features.pt").write_bytes((other / "node_features.pt").read_bytes())

        result = _reload(data_dir=str(root), checkpoint_path=str(root / "ckpt.pt"))

        assert result.success is False
        assert "is not the node_features artifact" in result.message
        _assert_untouched(api, before)

    def test_a_configuration_the_schema_rejects_changes_nothing(
        self, monkeypatch, tmp_path
    ):
        """`build_pipeline` obtains the configuration; it does not validate its
        shape. A pipeline that *returns* something unusable rather than raising
        constructs successfully, so the refusal happens later -- while the
        response is rendered -- and would land after publication if rendering
        came second."""
        import src.inference.pipeline as pipeline

        api, before = _serving(monkeypatch)
        root, _ = _candidate_workspace(monkeypatch, tmp_path)
        monkeypatch.setattr(
            pipeline.DiagnosisPipeline,
            "get_pipeline_config",
            lambda self: _candidate_config(kg_nodes="not an integer at all"),
        )

        result = _reload(data_dir=str(root), checkpoint_path=str(root / "ckpt.pt"))

        assert result.success is False
        assert "cannot be reported" in result.message
        assert result.status.initialized is True, "reported a down service"
        _assert_untouched(api, before)

    def test_a_configuration_no_encoder_can_serialise_changes_nothing(
        self, monkeypatch, tmp_path
    ):
        """Pydantic construction is not the last thing that can fail.

        `checkpoint_meta` is `Dict[str, Any]`, so an unserialisable value passes
        the model and is refused by the encoder FastAPI runs *after* the endpoint
        returns -- past the swap, where the request reports failure over state it
        has already replaced. Driving the coroutine directly, as these tests do,
        never reaches that boundary, so the endpoint has to reach it itself.
        """
        import src.inference.pipeline as pipeline

        api, before = _serving(monkeypatch)
        root, _ = _candidate_workspace(monkeypatch, tmp_path)
        monkeypatch.setattr(
            pipeline.DiagnosisPipeline,
            "get_pipeline_config",
            lambda self: _candidate_config(checkpoint_meta={"trained_on": object()}),
        )

        result = _reload(data_dir=str(root), checkpoint_path=str(root / "ckpt.pt"))

        assert result.success is False
        assert "cannot be reported" in result.message
        assert result.status.initialized is True, "reported a down service"
        _assert_untouched(api, before)

    def test_a_late_configuration_failure_is_refused_and_changes_nothing(
        self, monkeypatch, tmp_path
    ):
        """The last thing that can fail, and the one a caller is most tempted to
        do after publishing."""
        import src.inference.pipeline as pipeline

        api, before = _serving(monkeypatch)
        root, _ = _candidate_workspace(monkeypatch, tmp_path)
        monkeypatch.setattr(
            pipeline.DiagnosisPipeline,
            "get_pipeline_config",
            lambda self: (_ for _ in ()).throw(RuntimeError("late failure")),
        )

        result = _reload(data_dir=str(root), checkpoint_path=str(root / "ckpt.pt"))

        assert result.success is False
        assert "late failure" in result.message
        _assert_untouched(api, before)


# --------------------------------------------------------------------------
# ...and says so, without claiming the service is down
# --------------------------------------------------------------------------
class TestAFailedReloadReportsWhatIsStillServing:
    def test_the_failure_response_describes_the_live_pipeline(
        self, monkeypatch, tmp_path
    ):
        api, _ = _serving(monkeypatch)
        root, _ = _candidate_workspace(monkeypatch, tmp_path)
        other, _ = _candidate_workspace(monkeypatch, tmp_path, "other")
        (root / "edge_indices.pt").write_bytes((other / "edge_indices.pt").read_bytes())

        result = _reload(data_dir=str(root), checkpoint_path=str(root / "ckpt.pt"))

        assert result.success is False
        assert result.status.initialized is True, "reported a down service"
        assert result.status.scoring_mode == SERVED_CONFIG["scoring_mode"]
        assert result.status.served_identity.model_dump() == SERVED_CONFIG["served_identity"]
        assert result.status.current_data_dir == "/served/workspace"
        assert result.status.current_checkpoint_path == "/served/ckpt.pt"

    @pytest.mark.parametrize(
        "case,expect",
        [
            ("missing_files", "Missing required files"),
            ("bad_conv_type", "conv_type"),
            ("no_checkpoint", "No checkpoint found"),
        ],
    )
    def test_the_early_refusals_report_it_too(
        self, monkeypatch, tmp_path, case, expect
    ):
        """These paths never touch the pipeline at all, so `initialized=False`
        was wrong on every one of them -- it was simply invisible while the
        teardown above made it accidentally true."""
        api, before = _serving(monkeypatch)
        kwargs = {}
        if case == "missing_files":
            empty = tmp_path / "empty"
            empty.mkdir()
            kwargs["data_dir"] = str(empty)
        else:
            root, _ = _candidate_workspace(monkeypatch, tmp_path)
            kwargs["data_dir"] = str(root)
            kwargs["conv_type"] = "nonsense" if case == "bad_conv_type" else "auto"

        result = _reload(**kwargs)

        assert result.success is False
        assert expect in result.message
        assert result.status.initialized is True
        assert result.status.current_data_dir == "/served/workspace"
        _assert_untouched(api, before)


# --------------------------------------------------------------------------
# A healthy candidate does replace everything
# --------------------------------------------------------------------------
class TestAnAcceptedCandidateReplacesAllOfIt:
    def test_every_field_moves_to_the_new_pipeline_together(
        self, monkeypatch, tmp_path
    ):
        """Otherwise the fix above would be indistinguishable from a reload that
        never swaps, and the endpoint would keep serving the old workspace while
        reporting success."""
        import src.inference.pipeline as pipeline

        api, before = _serving(monkeypatch)
        root, _ = _candidate_workspace(monkeypatch, tmp_path)
        monkeypatch.setattr(
            pipeline.DiagnosisPipeline,
            "get_pipeline_config",
            lambda self: _candidate_config(),
        )

        result = _reload(data_dir=str(root), checkpoint_path=str(root / "ckpt.pt"))

        assert result.success is True
        assert api.app_state.pipeline is not before["pipeline"]
        assert api.app_state.kg is not before["kg"]
        # What is served moved with the pipeline (contract M3c): in the response
        # and in what the status route now reads off the live pipeline, both
        # through the strict response model, never from app state. The constant
        # is never handed to the code under test, so neither can match it by
        # sharing it.
        expected = CANDIDATE_CONFIG["served_identity"]
        assert result.status.served_identity.model_dump() == expected
        assert _live_status().served_identity.model_dump() == expected
        assert api.app_state._current_data_dir == str(root)
        assert api.app_state._current_checkpoint_path == str(root / "ckpt.pt")
        assert result.status.scoring_mode == CANDIDATE_CONFIG["scoring_mode"]
        assert result.status.current_data_dir == str(root)

    def test_the_model_is_only_built_after_the_workspace_is_accepted(
        self, monkeypatch, tmp_path
    ):
        """The swap is not the only ordering that matters: verification precedes
        construction, so a refused workspace never reaches the checkpoint or the
        model. `other` is built first because each `_candidate_workspace` call
        installs a fresh recorder, and only the last one installed is live."""
        api, _ = _serving(monkeypatch)
        other, _ = _candidate_workspace(monkeypatch, tmp_path, "other")
        root, reached = _candidate_workspace(monkeypatch, tmp_path)
        (root / "num_nodes.json").write_bytes((other / "num_nodes.json").read_bytes())

        result = _reload(data_dir=str(root), checkpoint_path=str(root / "ckpt.pt"))

        assert result.success is False
        assert reached == [], "the model was built out of a refused workspace"


# --------------------------------------------------------------------------
# Startup publishes the same set of fields
# --------------------------------------------------------------------------
def test_startup_records_which_workspace_it_published(monkeypatch, tmp_path):
    """`initialize_pipeline` set three of the five fields, leaving
    `/pipeline/status` unable to name the workspace it was serving until someone
    reloaded. One publisher writes all of them."""
    import src.api.main as api
    import src.inference.pipeline as pipeline

    root, _ = _candidate_workspace(monkeypatch, tmp_path)
    monkeypatch.setattr(api.app_state, "pipeline", None, raising=False)
    monkeypatch.setattr(api.app_state, "kg", None, raising=False)
    monkeypatch.setattr(api.app_state, "_current_data_dir", None, raising=False)
    monkeypatch.setattr(api.app_state, "_current_checkpoint_path", None, raising=False)
    monkeypatch.setattr(
        pipeline.DiagnosisPipeline,
        "get_pipeline_config",
        lambda self: _candidate_config(),
    )
    monkeypatch.setenv("SHEPHERD_KG_PATH", str(root / "kg.json"))
    monkeypatch.setenv("SHEPHERD_DATA_DIR", str(root))
    monkeypatch.setenv("SHEPHERD_CHECKPOINT_PATH", str(root / "ckpt.pt"))

    api.initialize_pipeline()

    assert api.app_state.pipeline is not None
    # What is served is the published pipeline's to say (contract M3c), as the
    # status route reads it, through the strict response model; app state keeps
    # no copy of it for a reload to leave behind.
    assert (
        _live_status().served_identity.model_dump()
        == CANDIDATE_CONFIG["served_identity"]
    )
    assert not hasattr(api.app_state, "model_version")
    assert api.app_state._current_data_dir == str(root)
    assert api.app_state._current_checkpoint_path == str(root / "ckpt.pt")


def test_a_service_with_no_pipeline_is_not_told_one_is_still_serving(
    monkeypatch, tmp_path
):
    """The reassurance is conditional on there being something to reassure about.
    A reload that fails on a service which never loaded a pipeline is a plain
    failure, and saying otherwise would be the same class of untruth as the
    `initialized=False` it replaces."""
    import src.api.main as api

    root, _ = _candidate_workspace(monkeypatch, tmp_path)
    other, _ = _candidate_workspace(monkeypatch, tmp_path, "other")
    (root / "node_features.pt").write_bytes((other / "node_features.pt").read_bytes())
    monkeypatch.setattr(api.app_state, "pipeline", None, raising=False)
    monkeypatch.setattr(api.app_state, "kg", None, raising=False)
    monkeypatch.setattr(api.app_state, "_current_data_dir", None, raising=False)
    monkeypatch.setattr(api.app_state, "_current_checkpoint_path", None, raising=False)

    result = _reload(data_dir=str(root), checkpoint_path=str(root / "ckpt.pt"))

    assert result.success is False
    assert "still being served" not in result.message
    assert result.status.initialized is False


def test_publication_formats_nothing_after_it_has_assigned(monkeypatch, tmp_path):
    """`publish_pipeline` logs an announcement interpolating configuration
    values, and `__str__` is the caller's code.

    **Tested through startup, not through reload.** Reload renders its response
    first, and the response schema types the three values the announcement
    interpolates — so by the time reload publishes, they are provably a `str`
    and two `bool`s and the formatting cannot fail. `initialize_pipeline`
    validates nothing, which is where the ordering inside `publish_pipeline` is
    the only thing standing between a hostile value and a pipeline that is live
    while its publisher raises.
    """
    import src.api.main as api
    import src.inference.pipeline as pipeline

    class _Hostile:
        def __str__(self):
            raise RuntimeError("formatting me is a mistake")

        __repr__ = __str__

    root, _ = _candidate_workspace(monkeypatch, tmp_path)
    monkeypatch.setattr(api.app_state, "pipeline", None, raising=False)
    monkeypatch.setattr(api.app_state, "kg", None, raising=False)
    monkeypatch.setattr(
        pipeline.DiagnosisPipeline,
        "get_pipeline_config",
        lambda self: _candidate_config(scoring_mode=_Hostile()),
    )
    monkeypatch.setenv("SHEPHERD_KG_PATH", str(root / "kg.json"))
    monkeypatch.setenv("SHEPHERD_DATA_DIR", str(root))
    monkeypatch.setenv("SHEPHERD_CHECKPOINT_PATH", str(root / "ckpt.pt"))

    with pytest.raises(RuntimeError, match="formatting me is a mistake"):
        api.initialize_pipeline()

    assert api.app_state.pipeline is None, "published, then failed to announce it"
    assert api.app_state.kg is None


# --------------------------------------------------------------------------
# The encoding boundary, on both paths
# --------------------------------------------------------------------------
class TestNothingLeavesThisEndpointUnencodable:
    """`checkpoint_meta` is `Dict[str, Any]`, and its values come from a
    checkpoint's `logs` -- whatever the trainer put there.

    The realistic hazard is not `object()`. It is a metric recorded as
    `loss.detach()` rather than `loss.item()`, which is a zero-dimensional
    tensor: a training detail that becomes a serving failure at the HTTP
    boundary, after the endpoint has returned.
    """

    def test_an_epoch_is_an_integer_or_it_is_nothing(self):
        """`True` is an `int` and `1.9` truncates: both would report metadata
        the checkpoint never contained. `inf` raises `OverflowError` from
        `int()`, which is the conversion failure that matters, because this runs
        after the model is built — raising here takes a working checkpoint out
        of service over a number in a status panel."""
        from src.inference.pipeline import _as_int

        torch = pytest.importorskip("torch")

        assert _as_int(3) == 3
        assert _as_int(3.0) == 3, "a genuinely integral value, not a truncation"
        assert _as_int(torch.tensor(7)) == 7
        assert _as_int(torch.tensor(7.0)) == 7
        for rejected in (True, False, 1.9, float("inf"), float("-inf"),
                         float("nan"), torch.tensor([1, 2]), "7", None,
                         complex(1, 2), 10 ** 400):
            assert _as_int(rejected) is None, f"{rejected!r} became an epoch"

    def test_a_metric_is_a_float_a_name_or_nothing(self):
        from src.inference.pipeline import _as_metric

        torch = pytest.importorskip("torch")

        assert _as_metric(torch.tensor(0.25)) == pytest.approx(0.25)
        assert isinstance(_as_metric(torch.tensor(0.25)), float)
        assert _as_metric(torch.tensor(3)) == pytest.approx(3.0)
        # Not valid JSON, but a fact about the run worth keeping.
        assert _as_metric(float("nan")) == "nan"
        assert _as_metric(torch.tensor(float("inf"))) == "inf"
        assert _as_metric(float("-inf")) == "-inf"
        # One number in any shape is still one number.
        assert _as_metric(torch.tensor([[1.0]])) == pytest.approx(1.0)
        for dropped in (torch.tensor([1.0, 2.0]), torch.tensor([[1.0, 2.0]]),
                        complex(1, 2), True, "0.5", None):
            assert _as_metric(dropped) is None, f"{dropped!r} became a metric"

    def test_neither_normaliser_can_take_a_checkpoint_out_of_service(self):
        """The contract that makes the two above safe to call where they are
        called: after the model is built, on values nobody validated."""
        from src.inference.pipeline import _as_int, _as_metric

        class _Hostile:
            def __float__(self):
                raise RuntimeError("not on this device")

            def __int__(self):
                raise RuntimeError("not on this device")

        assert _as_int(_Hostile()) is None
        assert _as_metric(_Hostile()) is None

    def test_the_normalised_values_encode(self):
        from fastapi.encoders import jsonable_encoder

        from src.api.routes.pipeline import PipelineStatusResponse
        from src.inference.pipeline import _as_int, _as_metric

        torch = pytest.importorskip("torch")

        jsonable_encoder(
            PipelineStatusResponse(
                initialized=True,
                checkpoint_meta={
                    "epoch": _as_int(torch.tensor(3)),
                    "val_loss": _as_metric(torch.tensor(0.25)),
                    "mrr": _as_metric(float("nan")),
                },
            )
        )

    def test_a_raw_tensor_that_reached_the_boundary_anyway_is_refused(
        self, monkeypatch, tmp_path
    ):
        """Normalising at the source is the fix; the encoder before publication
        is the guard that holds when a value gets past it by another route."""
        import src.inference.pipeline as pipeline

        torch = pytest.importorskip("torch")

        api, before = _serving(monkeypatch)
        root, _ = _candidate_workspace(monkeypatch, tmp_path)
        monkeypatch.setattr(
            pipeline.DiagnosisPipeline,
            "get_pipeline_config",
            lambda self: _candidate_config(
                checkpoint_meta={"val_loss": torch.tensor(0.25)}
            ),
        )

        result = _reload(data_dir=str(root), checkpoint_path=str(root / "ckpt.pt"))

        assert result.success is False
        assert "cannot be reported" in result.message
        _assert_untouched(api, before)

    @pytest.mark.parametrize(
        "case",
        ["missing_files", "bad_conv_type", "no_checkpoint", "bad_workspace"],
    )
    def test_every_refusal_crosses_the_encoder_before_it_is_returned(
        self, monkeypatch, tmp_path, case
    ):
        """A refusal that cannot be serialised becomes a 500 -- reporting a crash
        for a request that correctly declined to act."""
        from fastapi.encoders import jsonable_encoder

        _serving(monkeypatch)
        kwargs = {}
        if case == "missing_files":
            empty = tmp_path / "empty"
            empty.mkdir()
            kwargs["data_dir"] = str(empty)
        else:
            root, _ = _candidate_workspace(monkeypatch, tmp_path)
            kwargs["data_dir"] = str(root)
            if case == "bad_conv_type":
                kwargs["conv_type"] = "nonsense"
            elif case == "bad_workspace":
                other, _ = _candidate_workspace(monkeypatch, tmp_path, "other")
                (root / "node_features.pt").write_bytes((other / "node_features.pt").read_bytes())
                kwargs["checkpoint_path"] = str(root / "ckpt.pt")

        result = _reload(**kwargs)

        assert result.success is False
        jsonable_encoder(result)

    def test_a_live_pipeline_that_cannot_be_described_still_gets_a_refusal(
        self, monkeypatch, tmp_path
    ):
        """The fallback. `_live_status` reads the *running* pipeline's config,
        which is arbitrary too, so the refusal path can fail on state this
        request never touched. It still has to answer."""
        from fastapi.encoders import jsonable_encoder

        import src.api.main as api

        class _Undescribable:
            def get_pipeline_config(self):
                return {"checkpoint_meta": {"x": object()}}

        monkeypatch.setattr(api.app_state, "pipeline", _Undescribable(), raising=False)
        monkeypatch.setattr(api.app_state, "_current_data_dir", "/served", raising=False)
        empty = tmp_path / "empty"
        empty.mkdir()

        result = _reload(data_dir=str(empty))

        assert result.success is False
        assert "Missing required files" in result.message
        assert "could not be rendered" in result.message
        assert result.status.initialized is True, "a pipeline is serving; say so"
        jsonable_encoder(result)


def test_a_checkpoint_whose_metrics_are_tensors_still_loads_and_serves(tmp_path):
    """The point of normalising at the source is that such a checkpoint *works*.

    The encoder before publication would refuse it, which keeps the service
    safe and makes every reload of a perfectly good checkpoint fail. A metric
    written as `loss.detach()` is a training-side slip, not a reason to decline
    to serve a trained model.
    """
    torch = pytest.importorskip("torch")

    from src.inference.pipeline import DiagnosisPipeline
    from src.kg.graph import KnowledgeGraph
    from tests.fixtures.synthetic_workspace import build_workspace

    data_dir, checkpoint_path = build_workspace(tmp_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["epoch"] = torch.tensor(3)
    checkpoint["logs"] = {"val_loss": torch.tensor(0.25), "val_mrr": torch.tensor(0.5)}
    torch.save(checkpoint, checkpoint_path)

    # From the workspace: a checkpoint is loaded only over a graph verified from
    # one (contract M2.4).
    pipeline = DiagnosisPipeline(
        graph=KnowledgeGraph.read_json(data_dir / "kg.json"),
        data_dir=str(data_dir),
        checkpoint_path=str(checkpoint_path),
        device="cpu",
    )

    meta = pipeline.get_pipeline_config()["checkpoint_meta"]
    assert meta["epoch"] == 3 and isinstance(meta["epoch"], int)
    assert meta["val_loss"] == pytest.approx(0.25)
    assert isinstance(meta["val_loss"], float)
    assert meta["val_mrr"] == pytest.approx(0.5) and isinstance(meta["val_mrr"], float)
    assert isinstance(meta["params"], int)
    from fastapi.encoders import jsonable_encoder

    from src.api.routes.pipeline import _status_of

    jsonable_encoder(_status_of(pipeline.get_pipeline_config(), None, None))


def test_a_workspace_without_a_manifest_is_named_in_the_file_report(
    monkeypatch, tmp_path
):
    """It was always going to be refused -- reload builds a file-backed pipeline
    -- but the refusal arrived from inside the build, with a longer story. The
    file report is where an operator looks for what is missing."""
    api, before = _serving(monkeypatch)
    root, _ = _candidate_workspace(monkeypatch, tmp_path)
    (root / "split_manifest.json").unlink()

    result = _reload(data_dir=str(root), checkpoint_path=str(root / "ckpt.pt"))

    assert result.success is False
    assert "split_manifest.json" in result.message
    assert result.files_found["split_manifest.json"] is False
    _assert_untouched(api, before)
