"""The recovery path has to be reachable from the surface a service is started with.

The loader refuses to guess a hop bound and tells the operator to supply one.
That instruction was unusable: `build_pipeline` called the factory without a
config at all, so `PipelineConfig.sp_hop_bound` could only ever be set by
writing Python. The loader tests did not catch it because they assigned
`pipeline.config` directly — testing the function and not the wiring.

Module: tests/unit/test_hop_bound_reaches_the_service.py
"""
from __future__ import annotations

import pytest


class TestTheEnvironmentValueIsRead:
    """`SHEPHERD_SP_HOP_BOUND`, beside the other settings a deployment uses."""

    def test_absent_means_none(self, monkeypatch):
        from src.api.main import SP_HOP_BOUND_ENV, _configured_hop_bound

        monkeypatch.delenv(SP_HOP_BOUND_ENV, raising=False)

        assert _configured_hop_bound() is None

    def test_blank_means_none(self, monkeypatch):
        """An exported-but-empty variable is how a shell says nothing."""
        from src.api.main import SP_HOP_BOUND_ENV, _configured_hop_bound

        monkeypatch.setenv(SP_HOP_BOUND_ENV, "   ")

        assert _configured_hop_bound() is None

    def test_an_integer_is_read(self, monkeypatch):
        from src.api.main import SP_HOP_BOUND_ENV, _configured_hop_bound

        monkeypatch.setenv(SP_HOP_BOUND_ENV, " 3 ")

        assert _configured_hop_bound() == 3

    def test_an_unreadable_value_is_refused_not_ignored(self, monkeypatch):
        """Treating it as unset would put the service back in the state the
        loader refuses — serving without shortest paths while the operator
        believes they configured them."""
        from src.api.main import SP_HOP_BOUND_ENV, _configured_hop_bound

        monkeypatch.setenv(SP_HOP_BOUND_ENV, "five")

        with pytest.raises(ValueError, match="is not an integer"):
            _configured_hop_bound()


class TestItReachesTheFactory:
    """The gap itself: a config the service never passed.

    Intercepts `create_diagnosis_pipeline` at the point `build_pipeline` calls
    it, because that is exactly where the value was being dropped.
    """

    @staticmethod
    def _captured(monkeypatch, tmp_path):
        import json

        import src.inference.pipeline as pipeline_module
        from src.kg.graph import KnowledgeGraph

        seen = {}

        def _capture(**kwargs):
            seen.update(kwargs)
            raise RuntimeError("stop here; the config is what this test wants")

        monkeypatch.setattr(pipeline_module, "create_diagnosis_pipeline", _capture)
        monkeypatch.setattr(
            KnowledgeGraph, "load_json", staticmethod(lambda *a, **k: KnowledgeGraph())
        )

        kg_file = tmp_path / "kg.json"
        kg_file.write_text(json.dumps({"nodes": [], "edges": []}))
        return seen, kg_file

    def test_the_configured_bound_arrives_in_the_pipeline_config(
        self, monkeypatch, tmp_path
    ):
        from src.api.main import SP_HOP_BOUND_ENV, build_pipeline

        seen, kg_file = self._captured(monkeypatch, tmp_path)
        monkeypatch.setenv(SP_HOP_BOUND_ENV, "3")

        with pytest.raises(RuntimeError):
            build_pipeline(kg_path=str(kg_file))

        assert "config" in seen, "the factory was called without a config at all"
        assert seen["config"].sp_hop_bound == 3

    def test_no_setting_still_produces_a_config_with_none(
        self, monkeypatch, tmp_path
    ):
        """The unset case must not become a different code path — otherwise the
        wiring works only when someone happens to use it."""
        from src.api.main import SP_HOP_BOUND_ENV, build_pipeline

        seen, kg_file = self._captured(monkeypatch, tmp_path)
        monkeypatch.delenv(SP_HOP_BOUND_ENV, raising=False)

        with pytest.raises(RuntimeError):
            build_pipeline(kg_path=str(kg_file))

        assert seen["config"].sp_hop_bound is None


class TestTheLoaderAcceptsWhatTheServiceSupplies:
    """End to end in the sense that matters: a bound set the way a deployment
    sets it, reaching a table with no sidecar."""

    def test_a_service_supplied_bound_turns_shortest_paths_back_on(
        self, monkeypatch, tmp_path
    ):
        torch = pytest.importorskip("torch")

        from src.api.main import SP_HOP_BOUND_ENV, _configured_hop_bound
        from src.inference.pipeline import DiagnosisPipeline, PipelineConfig

        data_dir = tmp_path / "ws"
        data_dir.mkdir()
        torch.save(
            {
                "phenotype_idx": torch.arange(3, dtype=torch.int64),
                "target_idx": torch.arange(3, dtype=torch.int64),
                "target_type": torch.zeros(3, dtype=torch.int64),
                "distance": torch.tensor([1, 2, 3], dtype=torch.int8),
            },
            data_dir / "shortest_paths.pt",
        )
        # No sidecar beside it — the state the whole recovery path is for.

        monkeypatch.setenv(SP_HOP_BOUND_ENV, "3")
        pipeline = DiagnosisPipeline.__new__(DiagnosisPipeline)
        pipeline.config = PipelineConfig(sp_hop_bound=_configured_hop_bound())
        pipeline._sp_ready = False
        pipeline._sp_lookup = None
        pipeline._sp_max_hops = 5
        pipeline._sp_hop_bound_source = None

        pipeline._load_shortest_paths(data_dir)

        assert pipeline._sp_ready is True
        assert pipeline._sp_max_hops == 3
        assert pipeline._sp_hop_bound_source == "configured"
