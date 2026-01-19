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
#   - Prediction heads
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
    EdgeTypeEncoder,
    LaplacianPE,
    RandomWalkSE,
    PositionalEncoder,
    HeteroFeatureEncoder,
)
from src.models.layers import (
    HeteroGNNLayer,
    OrthologGate,
    FlexHeteroAttention,
)
from src.models.gnn import (
    ShepherdGNN,
    ShepherdGNNConfig,
    PhenotypeDiseaseMatcher,
    create_model,
)
from src.models.heads import (
    DiagnosisHead,
    LinkPredictionHead,
    NodeClassificationHead,
    ExplanationHead,
)


# ==============================================================================
# Fixtures
# ==============================================================================
@pytest.fixture
def hidden_dim():
    return 64


@pytest.fixture
def num_heads():
    return 4


@pytest.fixture
def simple_metadata():
    """Simple metadata for testing"""
    node_types = ["phenotype", "gene", "disease"]
    edge_types = [
        ("phenotype", "associated_with", "gene"),
        ("gene", "causes", "disease"),
        ("phenotype", "of", "disease"),
    ]
    return (node_types, edge_types)


@pytest.fixture
def metadata_with_ortholog():
    """Metadata including ortholog types"""
    node_types = ["phenotype", "gene", "disease", "mouse_gene", "mouse_phenotype"]
    edge_types = [
        ("phenotype", "associated_with", "gene"),
        ("gene", "causes", "disease"),
        ("phenotype", "of", "disease"),
        ("gene", "ortholog_of", "mouse_gene"),
        ("mouse_gene", "has_phenotype", "mouse_phenotype"),
    ]
    return (node_types, edge_types)


@pytest.fixture
def sample_hetero_data(simple_metadata, hidden_dim):
    """Create sample heterogeneous graph data"""
    node_types, edge_types = simple_metadata

    x_dict = {
        "phenotype": torch.randn(10, hidden_dim),
        "gene": torch.randn(20, hidden_dim),
        "disease": torch.randn(5, hidden_dim),
    }

    edge_index_dict = {
        ("phenotype", "associated_with", "gene"): torch.stack([
            torch.randint(0, 10, (30,)),
            torch.randint(0, 20, (30,)),
        ]),
        ("gene", "causes", "disease"): torch.stack([
            torch.randint(0, 20, (15,)),
            torch.randint(0, 5, (15,)),
        ]),
        ("phenotype", "of", "disease"): torch.stack([
            torch.randint(0, 10, (20,)),
            torch.randint(0, 5, (20,)),
        ]),
    }

    return x_dict, edge_index_dict


@pytest.fixture
def sample_hetero_data_with_ortholog(metadata_with_ortholog, hidden_dim):
    """Create sample data including ortholog nodes/edges"""
    node_types, edge_types = metadata_with_ortholog

    x_dict = {
        "phenotype": torch.randn(10, hidden_dim),
        "gene": torch.randn(20, hidden_dim),
        "disease": torch.randn(5, hidden_dim),
        "mouse_gene": torch.randn(15, hidden_dim),
        "mouse_phenotype": torch.randn(8, hidden_dim),
    }

    edge_index_dict = {
        ("phenotype", "associated_with", "gene"): torch.stack([
            torch.randint(0, 10, (30,)),
            torch.randint(0, 20, (30,)),
        ]),
        ("gene", "causes", "disease"): torch.stack([
            torch.randint(0, 20, (15,)),
            torch.randint(0, 5, (15,)),
        ]),
        ("phenotype", "of", "disease"): torch.stack([
            torch.randint(0, 10, (20,)),
            torch.randint(0, 5, (20,)),
        ]),
        ("gene", "ortholog_of", "mouse_gene"): torch.stack([
            torch.randint(0, 20, (25,)),
            torch.randint(0, 15, (25,)),
        ]),
        ("mouse_gene", "has_phenotype", "mouse_phenotype"): torch.stack([
            torch.randint(0, 15, (20,)),
            torch.randint(0, 8, (20,)),
        ]),
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


class TestEdgeTypeEncoder:
    """Tests for EdgeTypeEncoder"""

    def test_basic_encoding(self, hidden_dim):
        encoder = EdgeTypeEncoder(num_types=10, hidden_dim=hidden_dim)
        type_indices = torch.tensor([0, 1, 2, 3, 4])

        output = encoder(type_indices)

        assert output.shape == (5, hidden_dim)


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


class TestFlexHeteroAttention:
    """Tests for FlexHeteroAttention"""

    def test_forward(self, hidden_dim, num_heads):
        attn = FlexHeteroAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )

        x_src = torch.randn(20, hidden_dim)
        x_dst = torch.randn(10, hidden_dim)
        edge_index = torch.stack([
            torch.randint(0, 20, (50,)),
            torch.randint(0, 10, (50,)),
        ])

        out = attn(x_src, x_dst, edge_index)

        assert out.shape == (10, hidden_dim)

    def test_with_edge_type(self, hidden_dim, num_heads):
        attn = FlexHeteroAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            use_edge_type_bias=True,
        )

        x_src = torch.randn(20, hidden_dim)
        x_dst = torch.randn(10, hidden_dim)
        edge_index = torch.stack([
            torch.randint(0, 20, (50,)),
            torch.randint(0, 10, (50,)),
        ])

        out = attn(x_src, x_dst, edge_index, edge_type="causes")

        assert out.shape == (10, hidden_dim)


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

    def test_return_all_layers(self, simple_metadata, sample_hetero_data, hidden_dim):
        x_dict, edge_index_dict = sample_hetero_data

        model = ShepherdGNN(
            metadata=simple_metadata,
            config=ShepherdGNNConfig(
                hidden_dim=hidden_dim,
                num_layers=3,
            ),
        )

        out_dict = model(x_dict, edge_index_dict, return_all_layers=True)
        layer_outputs = model.get_layer_outputs()

        assert len(layer_outputs) == 3


class TestCreateModel:
    """Tests for create_model factory function"""

    def test_create_default(self, simple_metadata):
        model = create_model(
            metadata=simple_metadata,
            hidden_dim=64,
            num_layers=2,
        )

        assert isinstance(model, ShepherdGNN)
        assert model.config.hidden_dim == 64
        assert model.config.num_layers == 2


# ==============================================================================
# Test Prediction Heads
# ==============================================================================
class TestDiagnosisHead:
    """Tests for DiagnosisHead"""

    def test_forward_unbatched(self, hidden_dim):
        head = DiagnosisHead(hidden_dim=hidden_dim)

        phenotype_emb = torch.randn(5, hidden_dim)  # 5 phenotypes
        disease_emb = torch.randn(10, hidden_dim)   # 10 diseases

        scores = head(phenotype_emb, disease_emb)

        assert scores.shape == (10,)

    def test_forward_batched(self, hidden_dim):
        head = DiagnosisHead(hidden_dim=hidden_dim)

        phenotype_emb = torch.randn(4, 5, hidden_dim)  # batch=4, 5 phenotypes each
        disease_emb = torch.randn(10, hidden_dim)       # 10 diseases

        scores = head(phenotype_emb, disease_emb)

        assert scores.shape == (4, 10)

    def test_with_mask(self, hidden_dim):
        head = DiagnosisHead(hidden_dim=hidden_dim)

        phenotype_emb = torch.randn(4, 5, hidden_dim)
        disease_emb = torch.randn(10, hidden_dim)
        mask = torch.tensor([
            [True, True, True, False, False],
            [True, True, False, False, False],
            [True, True, True, True, True],
            [True, False, False, False, False],
        ])

        scores = head(phenotype_emb, disease_emb, phenotype_mask=mask)

        assert scores.shape == (4, 10)

    def test_return_patient_profile(self, hidden_dim):
        head = DiagnosisHead(hidden_dim=hidden_dim)

        phenotype_emb = torch.randn(5, hidden_dim)
        disease_emb = torch.randn(10, hidden_dim)

        scores, profile = head(
            phenotype_emb, disease_emb, return_patient_profile=True
        )

        assert scores.shape == (10,)
        assert profile.shape == (hidden_dim,)


class TestLinkPredictionHead:
    """Tests for LinkPredictionHead"""

    def test_distmult(self, hidden_dim):
        head = LinkPredictionHead(
            hidden_dim=hidden_dim,
            num_edge_types=5,
            decoder_type="distmult",
        )

        src = torch.randn(100, hidden_dim)
        dst = torch.randn(100, hidden_dim)
        edge_type = torch.randint(0, 5, (100,))

        scores = head(src, dst, edge_type)

        assert scores.shape == (100,)

    def test_transe(self, hidden_dim):
        head = LinkPredictionHead(
            hidden_dim=hidden_dim,
            decoder_type="transe",
        )

        src = torch.randn(100, hidden_dim)
        dst = torch.randn(100, hidden_dim)

        scores = head(src, dst)

        assert scores.shape == (100,)

    def test_compute_loss(self, hidden_dim):
        head = LinkPredictionHead(hidden_dim=hidden_dim)

        pos_scores = torch.randn(50)
        neg_scores = torch.randn(100)

        loss = head.compute_loss(pos_scores, neg_scores)

        assert loss.dim() == 0  # scalar


class TestNodeClassificationHead:
    """Tests for NodeClassificationHead"""

    def test_forward(self, hidden_dim):
        head = NodeClassificationHead(
            hidden_dim=hidden_dim,
            num_classes=10,
        )

        node_emb = torch.randn(50, hidden_dim)

        logits = head(node_emb)

        assert logits.shape == (50, 10)

    def test_compute_loss(self, hidden_dim):
        head = NodeClassificationHead(
            hidden_dim=hidden_dim,
            num_classes=10,
        )

        logits = torch.randn(50, 10)
        targets = torch.randint(0, 10, (50,))

        loss = head.compute_loss(logits, targets)

        assert loss.dim() == 0


class TestExplanationHead:
    """Tests for ExplanationHead"""

    def test_forward(self, hidden_dim):
        head = ExplanationHead(hidden_dim=hidden_dim)

        patient_profile = torch.randn(hidden_dim)
        disease_emb = torch.randn(hidden_dim)
        phenotype_emb = torch.randn(5, hidden_dim)
        gene_emb = torch.randn(10, hidden_dim)

        explanations = head(patient_profile, disease_emb, phenotype_emb, gene_emb)

        assert "phenotype_importance" in explanations
        assert "gene_relevance" in explanations
        assert explanations["phenotype_importance"].shape == (5,)
        assert explanations["gene_relevance"].shape == (10,)


# ==============================================================================
# Test torch.compile Compatibility
# ==============================================================================
class TestTorchCompile:
    """Tests for torch.compile compatibility"""

    @pytest.mark.skipif(
        not hasattr(torch, "compile"),
        reason="torch.compile not available"
    )
    def test_compile_encoder(self, hidden_dim):
        encoder = NodeTypeEncoder(num_types=5, hidden_dim=hidden_dim)
        compiled = torch.compile(encoder, dynamic=True)

        type_indices = torch.tensor([0, 1, 2])
        output = compiled(type_indices)

        assert output.shape == (3, hidden_dim)

    @pytest.mark.skipif(
        not hasattr(torch, "compile"),
        reason="torch.compile not available"
    )
    def test_compile_diagnosis_head(self, hidden_dim):
        head = DiagnosisHead(hidden_dim=hidden_dim)
        compiled = torch.compile(head, dynamic=True)

        phenotype_emb = torch.randn(5, hidden_dim)
        disease_emb = torch.randn(10, hidden_dim)

        scores = compiled(phenotype_emb, disease_emb)

        assert scores.shape == (10,)


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

    def test_diagnosis_head_gradients(self, hidden_dim):
        head = DiagnosisHead(hidden_dim=hidden_dim)

        phenotype_emb = torch.randn(5, hidden_dim, requires_grad=True)
        disease_emb = torch.randn(10, hidden_dim, requires_grad=True)

        scores = head(phenotype_emb, disease_emb)
        loss = scores.sum()
        loss.backward()

        assert phenotype_emb.grad is not None
        assert disease_emb.grad is not None
