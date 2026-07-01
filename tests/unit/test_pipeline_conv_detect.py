"""
Unit tests for checkpoint architecture inference in the diagnosis pipeline.

Regression guard for the bug where an HGT checkpoint whose config lacked
``conv_type`` was silently rebuilt as the GAT default, failing the state_dict
load and dropping the pipeline to path_reasoning_fallback.

The key sets below are taken verbatim from a real HGT checkpoint's load error
(HGT params) and the GAT model it was mistakenly rebuilt as (expected params).
"""
import pytest

# pipeline.py imports torch / PyG at module load; skip where unavailable.
pytest.importorskip("torch")

from src.inference.pipeline import (  # noqa: E402
    _infer_conv_type_from_keys,
    _infer_num_layers_from_keys,
)

# Real HGTConv parameter names (from the checkpoint that failed to load).
HGT_KEYS = {
    "gnn_layers.0.conv.kqv_lin.lins.disease.weight",
    "gnn_layers.0.conv.k_rel.weight",
    "gnn_layers.0.conv.v_rel.weight",
    "gnn_layers.0.conv.skip.disease",
    "gnn_layers.0.conv.p_rel.disease__is_a__disease",
    "gnn_layers.3.conv.out_lin.lins.gene.bias",
}

# Real GATConv (in HeteroConv) parameter names the model expected instead.
GAT_KEYS = {
    "gnn_layers.0.conv.convs.<disease___is_a___disease>.att_src",
    "gnn_layers.0.conv.convs.<disease___is_a___disease>.att_dst",
    "gnn_layers.0.conv.convs.<disease___is_a___disease>.lin.weight",
    "gnn_layers.3.conv.convs.<phenotype___rev_gene_has_phenotype___gene>.bias",
}

# SAGEConv (in HeteroConv) parameter names.
SAGE_KEYS = {
    "gnn_layers.0.conv.convs.<disease___is_a___disease>.lin_l.weight",
    "gnn_layers.0.conv.convs.<disease___is_a___disease>.lin_r.weight",
    "gnn_layers.1.conv.convs.<disease___is_a___disease>.lin_l.weight",
}


def test_detect_hgt():
    assert _infer_conv_type_from_keys(HGT_KEYS) == "hgt"


def test_detect_gat():
    assert _infer_conv_type_from_keys(GAT_KEYS) == "gat"


def test_detect_sage():
    assert _infer_conv_type_from_keys(SAGE_KEYS) == "sage"


def test_detect_unknown_returns_none():
    assert _infer_conv_type_from_keys(set()) is None
    assert _infer_conv_type_from_keys({"embedding.weight", "output.bias"}) is None


def test_detect_accepts_list_and_set():
    assert _infer_conv_type_from_keys(list(HGT_KEYS)) == "hgt"


def test_num_layers_from_keys():
    assert _infer_num_layers_from_keys(HGT_KEYS) == 4  # indices 0..3
    assert _infer_num_layers_from_keys(SAGE_KEYS) == 2  # indices 0..1
    assert _infer_num_layers_from_keys(set()) is None
    assert _infer_num_layers_from_keys({"pos_encoder.weight"}) is None
