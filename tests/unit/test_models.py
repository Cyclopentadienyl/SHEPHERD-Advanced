"""
# ==============================================================================
# Module: tests/unit/test_models.py
# ==============================================================================
# Purpose: Unit tests for src/models/ module
#
# Tests:
#   - GNN layer message passing
#   - Gradient flow through a two-layer stack of the layer
# ==============================================================================
"""
import pytest
from typing import Dict, List, Tuple

# Skip all tests if torch is not available
torch = pytest.importorskip("torch")
nn = torch.nn
Tensor = torch.Tensor

# Skip all tests if torch_geometric is not available
pytest.importorskip("torch_geometric")

# Import modules under test
from src.models.gnn.layers import HeteroGNNLayer


# ==============================================================================
# Fixtures
# ==============================================================================
@pytest.fixture
def hidden_dim():
    return 64


@pytest.fixture
def simple_metadata():
    """Simple metadata for testing (with reverse edges for bidirectional message passing)"""
    node_types = ["phenotype", "gene", "disease"]
    edge_types = [
        # Forward edges
        ("phenotype", "associated_with", "gene"),
        ("gene", "causes", "disease"),
        ("phenotype", "of", "disease"),
        # Reverse edges (for bidirectional message passing)
        ("gene", "rev_associated_with", "phenotype"),
        ("disease", "rev_causes", "gene"),
        ("disease", "rev_of", "phenotype"),
    ]
    return (node_types, edge_types)


@pytest.fixture
def sample_hetero_data(simple_metadata, hidden_dim):
    """Create sample heterogeneous graph data (with reverse edges)"""
    node_types, edge_types = simple_metadata

    x_dict = {
        "phenotype": torch.randn(10, hidden_dim),
        "gene": torch.randn(20, hidden_dim),
        "disease": torch.randn(5, hidden_dim),
    }

    # Forward edges
    phenotype_gene_edges = torch.stack([
        torch.randint(0, 10, (30,)),
        torch.randint(0, 20, (30,)),
    ])
    gene_disease_edges = torch.stack([
        torch.randint(0, 20, (15,)),
        torch.randint(0, 5, (15,)),
    ])
    phenotype_disease_edges = torch.stack([
        torch.randint(0, 10, (20,)),
        torch.randint(0, 5, (20,)),
    ])

    edge_index_dict = {
        # Forward edges
        ("phenotype", "associated_with", "gene"): phenotype_gene_edges,
        ("gene", "causes", "disease"): gene_disease_edges,
        ("phenotype", "of", "disease"): phenotype_disease_edges,
        # Reverse edges (for bidirectional message passing)
        ("gene", "rev_associated_with", "phenotype"): phenotype_gene_edges.flip(0),
        ("disease", "rev_causes", "gene"): gene_disease_edges.flip(0),
        ("disease", "rev_of", "phenotype"): phenotype_disease_edges.flip(0),
    }

    return x_dict, edge_index_dict


# ==============================================================================
# Test Layers
# ==============================================================================
class TestHeteroGNNLayer:
    """Tests for HeteroGNNLayer"""

    def test_forward_gat(self, simple_metadata, sample_hetero_data, hidden_dim):
        x_dict, edge_index_dict = sample_hetero_data

        layer = HeteroGNNLayer(
            hidden_dim=hidden_dim,
            metadata=simple_metadata,
            conv_type="gat",
            num_heads=4,
        )

        out_dict = layer(x_dict, edge_index_dict)

        assert "phenotype" in out_dict
        assert "gene" in out_dict
        assert "disease" in out_dict
        assert out_dict["phenotype"].shape == (10, hidden_dim)
        assert out_dict["gene"].shape == (20, hidden_dim)
        assert out_dict["disease"].shape == (5, hidden_dim)

    def test_forward_sage(self, simple_metadata, sample_hetero_data, hidden_dim):
        x_dict, edge_index_dict = sample_hetero_data

        layer = HeteroGNNLayer(
            hidden_dim=hidden_dim,
            metadata=simple_metadata,
            conv_type="sage",
        )

        out_dict = layer(x_dict, edge_index_dict)

        assert out_dict["phenotype"].shape == (10, hidden_dim)


# ==============================================================================
# Test Gradient Flow
# ==============================================================================
class TestGradientFlow:
    """Tests to ensure proper gradient flow"""

    def test_hetero_gnn_layer_stack_gradients(self, simple_metadata, sample_hetero_data,
                                              hidden_dim):
        """The stack the retired wrapper built by default (two gat layers, eight
        heads, residual, layer norm, dropout 0.1), now on the kept layer alone."""
        x_dict, edge_index_dict = sample_hetero_data
        x_dict = {k: v.clone().requires_grad_(True) for k, v in x_dict.items()}
        layers = nn.ModuleList([
            HeteroGNNLayer(hidden_dim=hidden_dim, metadata=simple_metadata, conv_type="gat")
            for _ in range(2)
        ])
        h_dict = x_dict
        for layer in layers:
            h_dict = layer(h_dict, edge_index_dict)
        loss = sum(v.sum() for v in h_dict.values())
        loss.backward()
        for k, v in x_dict.items():
            assert v.grad is not None, f"No gradient for {k}"
