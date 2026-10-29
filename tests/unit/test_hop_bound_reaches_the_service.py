"""The configured hop bound is retired (contract M2.5).

This file used to show that `SHEPHERD_SP_HOP_BOUND` reached the pipeline: the
loader refused to guess the hop bound of a shortest-path table with no sidecar
and told the operator to supply one, and `build_pipeline` once called the factory
without any config, so that instruction was unusable until the variable was
wired through.

**Since contract M2.5 such a table is refused before any bound is read**: with
no sidecar it records no binding to the graph. The setting could no longer reach
a served table, and a setting nothing uses can still fail a start-up -- an
unparsable value used to raise in `build_pipeline`. So it is retired: the
variable is not read, `PipelineConfig` has no `sp_hop_bound`, and the bound comes
from the sidecar alone (`test_sp_hop_bound.py`). What stays is the producer's
`--max-hops`, the sidecar's `max_hops` and `validate_hop_bound`.

Module: tests/unit/test_hop_bound_reaches_the_service.py
"""
from __future__ import annotations

import dataclasses

import pytest


def test_the_pipeline_config_has_no_configured_hop_bound():
    from src.inference.pipeline import PipelineConfig

    assert "sp_hop_bound" not in {field.name for field in dataclasses.fields(PipelineConfig)}


def test_the_service_neither_reads_nor_fails_on_the_retired_variable(monkeypatch, tmp_path):
    """Set to a value the old reader refused, it changes nothing: the graph is
    read and the factory is reached, with no hop-bound configuration."""
    import json

    import src.api.main as api_main
    import src.inference.pipeline as pipeline_module

    assert not hasattr(api_main, "SP_HOP_BOUND_ENV")
    assert not hasattr(api_main, "_configured_hop_bound")
    seen = {}

    def _capture(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here; the arguments are what this test wants")

    monkeypatch.setattr(pipeline_module, "create_diagnosis_pipeline", _capture)
    monkeypatch.setenv("SHEPHERD_SP_HOP_BOUND", "five")
    kg_file = tmp_path / "kg.json"
    kg_file.write_text(json.dumps({"nodes": [], "edges": []}))

    with pytest.raises(RuntimeError, match="stop here"):
        api_main.build_pipeline(kg_path=str(kg_file), checkpoint_path=str(tmp_path / "m.pt"),
                                data_dir=str(tmp_path), device="cpu")

    from src.kg.graph import GraphRead

    assert isinstance(seen["graph"], GraphRead)
    assert "config" not in seen
