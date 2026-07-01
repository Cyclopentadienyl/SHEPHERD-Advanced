"""
Unit tests for checkpoint architecture resolution (_resolve_arch_params).

Covers the precedence chain used to reconstruct a model at inference time:
  1. model_config sub-dict (self-describing, current trainers)
  2. legacy flat arch fields
  3. inference from weight names
  4. defaults
plus the train->inference round-trip contract and the "unsupported conv_type
fails loudly" rule.
"""
import dataclasses

import pytest

# pipeline.py / shepherd_gnn.py import torch at module load; skip if unavailable.
pytest.importorskip("torch")

from src.inference.pipeline import _resolve_arch_params  # noqa: E402
from src.models.gnn.shepherd_gnn import ShepherdGNNConfig  # noqa: E402

SUPPORTED = ("hgt", "gat", "sage")
VALID = {f.name for f in dataclasses.fields(ShepherdGNNConfig)}

HGT_KEYS = {
    "gnn_layers.0.conv.kqv_lin.lins.disease.weight",
    "gnn_layers.0.conv.p_rel.disease__is_a__disease",
    "gnn_layers.3.conv.skip.disease",
}
GAT_KEYS = {
    "gnn_layers.0.conv.convs.<disease___is_a___disease>.att_src",
    "gnn_layers.2.conv.convs.<disease___is_a___disease>.att_dst",
}


def _resolve(ckpt_config, state_keys=frozenset(), **kw):
    return _resolve_arch_params(
        ckpt_config,
        set(state_keys),
        valid_fields=VALID,
        supported_conv=SUPPORTED,
        **kw,
    )


# --------------------------------------------------------------- precedence
def test_tier1_model_config_wins():
    p = _resolve(
        {"model_config": {"conv_type": "hgt", "hidden_dim": 512, "num_layers": 3}},
        HGT_KEYS,
    )
    assert p["conv_type"] == "hgt"
    assert p["hidden_dim"] == 512
    assert p["num_layers"] == 3


def test_tier2_legacy_flat_fields():
    p = _resolve({"conv_type": "hgt", "hidden_dim": 256}, HGT_KEYS)
    assert p["conv_type"] == "hgt"
    assert p["hidden_dim"] == 256


def test_tier3_detect_from_weights():
    p = _resolve({}, HGT_KEYS)
    assert p["conv_type"] == "hgt"
    assert p["num_layers"] == 4  # indices 0..3


def test_tier4_absent_defaults_to_gat():
    # Nothing to go on -> GAT default (legacy behavior), NOT a raise.
    p = _resolve({}, set())
    assert p["conv_type"] == "gat"


# --------------------------------------------------------------- ground truth
def test_conflicting_conv_type_trusts_weights():
    p = _resolve({"model_config": {"conv_type": "gat"}}, HGT_KEYS)
    assert p["conv_type"] == "hgt"  # weights win


def test_unsupported_conv_type_raises():
    with pytest.raises(ValueError, match="unsupported conv_type"):
        _resolve({"model_config": {"conv_type": "newnet"}}, set())


def test_unknown_fields_filtered():
    p = _resolve({"model_config": {"conv_type": "gat", "bogus_field": 123}}, GAT_KEYS)
    assert "bogus_field" not in p
    assert p["conv_type"] == "gat"


# --------------------------------------------------------------- flags & dropout
def test_flags_inferred_from_keys_when_absent():
    p = _resolve({}, HGT_KEYS, has_pos_encoder=True, has_ortholog_gate=False)
    assert p["use_positional_encoding"] is True
    assert p["use_ortholog_gate"] is False


def test_model_config_flags_beat_key_inference():
    p = _resolve(
        {"model_config": {"conv_type": "hgt", "use_positional_encoding": False}},
        HGT_KEYS,
        has_pos_encoder=True,  # key says present, but the saved config says off
    )
    assert p["use_positional_encoding"] is False


def test_dropout_forced_to_zero_for_inference():
    p = _resolve({"model_config": {"conv_type": "gat", "dropout": 0.3}}, GAT_KEYS)
    assert p["dropout"] == 0.0


# --------------------------------------------------------------- round-trip
def test_roundtrip_full_model_config_reconstructs_config():
    """asdict(model_config) -> resolve -> ShepherdGNNConfig reproduces arch."""
    original = ShepherdGNNConfig(
        hidden_dim=384, num_layers=3, num_heads=6, conv_type="hgt", dropout=0.2
    )
    ckpt_config = {"model_config": dataclasses.asdict(original)}
    params = _resolve(ckpt_config, HGT_KEYS)
    rebuilt = ShepherdGNNConfig(**params)

    assert rebuilt.conv_type == original.conv_type
    assert rebuilt.hidden_dim == original.hidden_dim
    assert rebuilt.num_layers == original.num_layers
    assert rebuilt.num_heads == original.num_heads
    assert rebuilt.dropout == 0.0  # inference override
