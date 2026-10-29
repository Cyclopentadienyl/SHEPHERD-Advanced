"""
# ==============================================================================
# Module: tests/unit/test_models.py
# ==============================================================================
# Purpose: Unit tests for src/models/ module
#
# Tests:
#   - Encoder functionality (type, positional)
#   - GNN layer message passing
#   - OrthologGate behavior
#   - ShepherdGNN end-to-end
#   - torch.compile compatibility
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
from src.models.encoders import (
    NodeTypeEncoder,
    LaplacianPE,
    RandomWalkSE,
    PositionalEncoder,
    HeteroFeatureEncoder,
)
from src.models.gnn import (
    ShepherdGNN,
    ShepherdGNNConfig,
)
from src.models.gnn.layers import (
    HeteroGNNLayer,
    OrthologGate,
)


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
def metadata_with_ortholog():
    """Metadata including ortholog types (with reverse edges)"""
    node_types = ["phenotype", "gene", "disease", "mouse_gene", "mouse_phenotype"]
    edge_types = [
        # Forward edges
        ("phenotype", "associated_with", "gene"),
        ("gene", "causes", "disease"),
        ("phenotype", "of", "disease"),
        ("gene", "ortholog_of", "mouse_gene"),
        ("mouse_gene", "has_phenotype", "mouse_phenotype"),
        # Reverse edges
        ("gene", "rev_associated_with", "phenotype"),
        ("disease", "rev_causes", "gene"),
        ("disease", "rev_of", "phenotype"),
        ("mouse_gene", "rev_ortholog_of", "gene"),
        ("mouse_phenotype", "rev_has_phenotype", "mouse_gene"),
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


@pytest.fixture
def sample_hetero_data_with_ortholog(metadata_with_ortholog, hidden_dim):
    """Create sample data including ortholog nodes/edges (with reverse edges)"""
    node_types, edge_types = metadata_with_ortholog

    x_dict = {
        "phenotype": torch.randn(10, hidden_dim),
        "gene": torch.randn(20, hidden_dim),
        "disease": torch.randn(5, hidden_dim),
        "mouse_gene": torch.randn(15, hidden_dim),
        "mouse_phenotype": torch.randn(8, hidden_dim),
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
    gene_mouse_gene_edges = torch.stack([
        torch.randint(0, 20, (25,)),
        torch.randint(0, 15, (25,)),
    ])
    mouse_gene_phenotype_edges = torch.stack([
        torch.randint(0, 15, (20,)),
        torch.randint(0, 8, (20,)),
    ])

    edge_index_dict = {
        # Forward edges
        ("phenotype", "associated_with", "gene"): phenotype_gene_edges,
        ("gene", "causes", "disease"): gene_disease_edges,
        ("phenotype", "of", "disease"): phenotype_disease_edges,
        ("gene", "ortholog_of", "mouse_gene"): gene_mouse_gene_edges,
        ("mouse_gene", "has_phenotype", "mouse_phenotype"): mouse_gene_phenotype_edges,
        # Reverse edges
        ("gene", "rev_associated_with", "phenotype"): phenotype_gene_edges.flip(0),
        ("disease", "rev_causes", "gene"): gene_disease_edges.flip(0),
        ("disease", "rev_of", "phenotype"): phenotype_disease_edges.flip(0),
        ("mouse_gene", "rev_ortholog_of", "gene"): gene_mouse_gene_edges.flip(0),
        ("mouse_phenotype", "rev_has_phenotype", "mouse_gene"): mouse_gene_phenotype_edges.flip(0),
    }

    return x_dict, edge_index_dict


# ==============================================================================
# Test Encoders
# ==============================================================================
class TestNodeTypeEncoder:
    """Tests for NodeTypeEncoder"""

    def test_basic_encoding(self, hidden_dim):
        encoder = NodeTypeEncoder(num_types=5, hidden_dim=hidden_dim)
        type_indices = torch.tensor([0, 1, 2, 1, 0])

        output = encoder(type_indices)

        assert output.shape == (5, hidden_dim)

    def test_with_features_add(self, hidden_dim):
        encoder = NodeTypeEncoder(num_types=5, hidden_dim=hidden_dim, fusion_mode="add")
        type_indices = torch.tensor([0, 1, 2])
        features = torch.randn(3, hidden_dim)

        output = encoder(type_indices, features)

        assert output.shape == (3, hidden_dim)

    def test_with_features_gate(self, hidden_dim):
        encoder = NodeTypeEncoder(num_types=5, hidden_dim=hidden_dim, fusion_mode="gate")
        type_indices = torch.tensor([0, 1, 2])
        features = torch.randn(3, hidden_dim)

        output = encoder(type_indices, features)

        assert output.shape == (3, hidden_dim)


class TestPositionalEncoder:
    """Tests for PositionalEncoder"""

    def test_with_degree_only(self, hidden_dim):
        encoder = PositionalEncoder(
            hidden_dim=hidden_dim,
            use_lap_pe=False,
            use_rwse=False,
            use_degree=True,
        )
        degree = torch.randint(0, 100, (20,))

        output = encoder(degree=degree)

        assert output.shape == (20, hidden_dim)

    def test_with_all_encodings(self, hidden_dim):
        encoder = PositionalEncoder(
            hidden_dim=hidden_dim,
            use_lap_pe=True,
            use_rwse=True,
            use_degree=True,
            lap_pe_dim=16,
            rwse_walk_length=10,
        )

        lap_pe = torch.randn(20, 16)
        rwse = torch.randn(20, 10)
        degree = torch.randint(0, 100, (20,))

        output = encoder(lap_pe=lap_pe, rwse=rwse, degree=degree)

        assert output.shape == (20, hidden_dim)


class TestHeteroFeatureEncoder:
    """Tests for HeteroFeatureEncoder"""

    def test_different_input_dims(self, hidden_dim):
        in_channels_dict = {
            "phenotype": 128,
            "gene": 256,
            "disease": 64,
        }
        encoder = HeteroFeatureEncoder(
            in_channels_dict=in_channels_dict,
            hidden_dim=hidden_dim,
        )

        x_dict = {
            "phenotype": torch.randn(10, 128),
            "gene": torch.randn(20, 256),
            "disease": torch.randn(5, 64),
        }

        out_dict = encoder(x_dict)

        assert out_dict["phenotype"].shape == (10, hidden_dim)
        assert out_dict["gene"].shape == (20, hidden_dim)
        assert out_dict["disease"].shape == (5, hidden_dim)


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


class TestOrthologGate:
    """Tests for OrthologGate"""

    def test_without_ortholog_data(self, hidden_dim):
        gate = OrthologGate(hidden_dim=hidden_dim)

        h_core = {
            "gene": torch.randn(20, hidden_dim),
            "disease": torch.randn(5, hidden_dim),
        }

        # No ortholog data
        out_dict = gate(h_core, None)

        assert torch.equal(out_dict["gene"], h_core["gene"])
        assert gate.get_ortholog_contribution() is None

    def test_with_ortholog_data(self, hidden_dim):
        gate = OrthologGate(hidden_dim=hidden_dim, gate_type="learned")

        h_core = {
            "gene": torch.randn(20, hidden_dim),
        }
        h_ortholog = {
            "gene": torch.randn(20, hidden_dim),
        }

        out_dict = gate(h_core, h_ortholog)

        assert out_dict["gene"].shape == (20, hidden_dim)

        # Check contribution is computed
        contrib = gate.get_ortholog_contribution()
        assert contrib is not None
        assert "gene" in contrib
        assert 0 <= contrib["gene"] <= 1

    def test_is_ortholog_edge(self):
        assert OrthologGate.is_ortholog_edge(("gene", "ortholog_of", "mouse_gene"))
        assert OrthologGate.is_ortholog_edge(("gene", "human_mouse_ortholog", "mouse_gene"))
        assert not OrthologGate.is_ortholog_edge(("gene", "causes", "disease"))


# ==============================================================================
# Test ShepherdGNN
# ==============================================================================
class TestShepherdGNN:
    """Tests for ShepherdGNN model"""

    def test_basic_forward(self, simple_metadata, sample_hetero_data, hidden_dim):
        x_dict, edge_index_dict = sample_hetero_data

        model = ShepherdGNN(
            metadata=simple_metadata,
            config=ShepherdGNNConfig(
                hidden_dim=hidden_dim,
                num_layers=2,
                use_ortholog_gate=False,
            ),
        )

        out_dict = model(x_dict, edge_index_dict)

        assert "phenotype" in out_dict
        assert "gene" in out_dict
        assert "disease" in out_dict
        assert out_dict["disease"].shape == (5, hidden_dim)

    def test_with_ortholog_gate(
        self, metadata_with_ortholog, sample_hetero_data_with_ortholog, hidden_dim
    ):
        x_dict, edge_index_dict = sample_hetero_data_with_ortholog

        model = ShepherdGNN(
            metadata=metadata_with_ortholog,
            config=ShepherdGNNConfig(
                hidden_dim=hidden_dim,
                num_layers=2,
                use_ortholog_gate=True,
            ),
        )

        out_dict = model(x_dict, edge_index_dict)

        assert "gene" in out_dict
        assert "mouse_gene" in out_dict

        # Check ortholog contribution is available
        contrib = model.get_ortholog_contribution()
        # Note: contrib may be None if no ortholog edges processed in last layer

    def test_without_ortholog_data(
        self, metadata_with_ortholog, sample_hetero_data, hidden_dim
    ):
        """Test that model works with ortholog gate but without ortholog data"""
        x_dict, edge_index_dict = sample_hetero_data

        model = ShepherdGNN(
            metadata=metadata_with_ortholog,
            config=ShepherdGNNConfig(
                hidden_dim=hidden_dim,
                num_layers=2,
                use_ortholog_gate=True,
            ),
        )

        # Only pass core data (no ortholog nodes/edges)
        out_dict = model(x_dict, edge_index_dict)

        assert "phenotype" in out_dict
        assert "gene" in out_dict
        assert "disease" in out_dict


# ==============================================================================
# Test torch.compile Compatibility
# ==============================================================================
import shutil

# Check if C++ compiler is available (required for torch.compile on CPU)
_HAS_CPP_COMPILER = shutil.which("cl") is not None or shutil.which("g++") is not None


def _has_python_headers() -> bool:
    """Whether Python.h is present.

    torch.compile's inductor backend has Triton build a small C extension at runtime, which
    #includes <Python.h>. A machine can have a working compiler and CUDA yet still lack the
    development headers (python3-dev / python3.12-dev), in which case compilation fails with
    "fatal error: Python.h: No such file or directory" — a missing build dependency surfacing
    as a test failure rather than a skip.
    """
    import sysconfig
    from pathlib import Path

    include_dir = sysconfig.get_paths().get("include")
    return bool(include_dir) and Path(include_dir, "Python.h").exists()


# torch.compile needs a compiler AND the Python development headers. Gating on only the
# former lets an incomplete toolchain through, and the failure then appears deep inside
# Triton rather than as a clean skip.
_CAN_TORCH_COMPILE = (
    hasattr(torch, "compile")
    and _HAS_CPP_COMPILER
    and _has_python_headers()
)
_TORCH_COMPILE_SKIP_REASON = (
    "torch.compile requires a C compiler and the Python development headers "
    "(install python3-dev / python3.12-dev; on Windows, MSVC)"
)


class TestTorchCompile:
    """Tests for torch.compile compatibility"""

    @pytest.mark.skipif(not _CAN_TORCH_COMPILE, reason=_TORCH_COMPILE_SKIP_REASON)
    def test_compile_encoder(self, hidden_dim):
        encoder = NodeTypeEncoder(num_types=5, hidden_dim=hidden_dim)
        compiled = torch.compile(encoder, dynamic=True)

        type_indices = torch.tensor([0, 1, 2])
        output = compiled(type_indices)

        assert output.shape == (3, hidden_dim)

# ==============================================================================
# Test Gradient Flow
# ==============================================================================
class TestGradientFlow:
    """Tests to ensure proper gradient flow"""

    def test_shepherd_gnn_gradients(self, simple_metadata, sample_hetero_data, hidden_dim):
        x_dict, edge_index_dict = sample_hetero_data

        # Make inputs require gradients
        x_dict = {k: v.clone().requires_grad_(True) for k, v in x_dict.items()}

        model = ShepherdGNN(
            metadata=simple_metadata,
            config=ShepherdGNNConfig(
                hidden_dim=hidden_dim,
                num_layers=2,
            ),
        )

        out_dict = model(x_dict, edge_index_dict)

        # Compute loss and backprop
        loss = sum(v.sum() for v in out_dict.values())
        loss.backward()

        # Check gradients exist
        for k, v in x_dict.items():
            assert v.grad is not None, f"No gradient for {k}"
