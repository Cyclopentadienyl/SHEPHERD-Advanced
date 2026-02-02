"""
# ==============================================================================
# Module: tests/unit/test_retrieval.py
# ==============================================================================
# Purpose: Unit tests for src/retrieval/ module
#
# Tests:
#   - VectorIndexBase abstract interface
#   - VoyagerIndex HNSW implementation
#   - CuVSIndex IVF-PQ implementation (GPU)
#   - Factory function create_index()
#   - Backend resolution logic
#   - Index persistence (save/load)
# ==============================================================================
"""
import pytest
import numpy as np
from pathlib import Path
from typing import Dict
from unittest.mock import patch, MagicMock

# ==============================================================================
# Test: VectorIndexBase Interface
# ==============================================================================
class TestVectorIndexBase:
    """Tests for the abstract base class interface."""

    def test_base_class_is_abstract(self):
        """VectorIndexBase should not be instantiable directly."""
        from src.retrieval.backends.base import VectorIndexBase

        with pytest.raises(TypeError):
            VectorIndexBase(dim=768)

    def test_base_class_defines_required_methods(self):
        """VectorIndexBase should define all required abstract methods."""
        from src.retrieval.backends.base import VectorIndexBase
        import inspect

        abstract_methods = {
            name for name, method in inspect.getmembers(VectorIndexBase)
            if getattr(method, '__isabstractmethod__', False)
        }

        expected = {'backend_name', '_build_index_impl', '_search_impl',
                    '_batch_search_impl', '_save_impl', '_load_impl'}
        assert expected.issubset(abstract_methods)


# ==============================================================================
# Test: VoyagerIndex
# ==============================================================================
class TestVoyagerIndex:
    """Tests for the Voyager HNSW backend."""

    @pytest.fixture
    def voyager_available(self):
        """Check if voyager is available."""
        try:
            import voyager
            return True
        except ImportError:
            pytest.skip("voyager not installed")

    @pytest.fixture
    def sample_embeddings(self) -> Dict[str, np.ndarray]:
        """Generate sample embeddings for testing."""
        np.random.seed(42)
        dim = 128
        num_vectors = 100
        return {
            f"entity_{i}": np.random.randn(dim).astype(np.float32)
            for i in range(num_vectors)
        }

    def test_voyager_import(self, voyager_available):
        """VoyagerIndex should be importable."""
        from src.retrieval.backends.voyager_backend import VoyagerIndex
        assert VoyagerIndex is not None

    def test_voyager_init(self, voyager_available):
        """VoyagerIndex should initialize with default parameters."""
        from src.retrieval.backends.voyager_backend import VoyagerIndex

        index = VoyagerIndex(dim=128, metric="cosine")
        assert index.dim == 128
        assert index.metric == "cosine"
        assert index.backend_name == "voyager"

    def test_voyager_build_and_search(self, voyager_available, sample_embeddings):
        """VoyagerIndex should build index and perform search."""
        from src.retrieval.backends.voyager_backend import VoyagerIndex

        index = VoyagerIndex(dim=128, metric="cosine")
        index.build_index(sample_embeddings)

        # Search with a known vector
        query = sample_embeddings["entity_0"]
        results = index.search(query, top_k=5)

        assert len(results) == 5
        # First result should be the query itself (or very close)
        assert results[0][0] == "entity_0"

    def test_voyager_batch_search(self, voyager_available, sample_embeddings):
        """VoyagerIndex should support batch search."""
        from src.retrieval.backends.voyager_backend import VoyagerIndex

        index = VoyagerIndex(dim=128, metric="cosine")
        index.build_index(sample_embeddings)

        queries = [sample_embeddings["entity_0"], sample_embeddings["entity_1"]]
        results = index.batch_search(queries, top_k=3)

        assert len(results) == 2
        assert len(results[0]) == 3
        assert len(results[1]) == 3

    def test_voyager_save_load(self, voyager_available, sample_embeddings, tmp_path):
        """VoyagerIndex should persist and reload correctly."""
        from src.retrieval.backends.voyager_backend import VoyagerIndex

        # Build and save
        index = VoyagerIndex(dim=128, metric="cosine")
        index.build_index(sample_embeddings)
        save_path = tmp_path / "test_index"
        index.save(save_path)

        # Load into new instance
        index2 = VoyagerIndex(dim=128, metric="cosine")
        index2.load(save_path)

        # Verify search works
        query = sample_embeddings["entity_0"]
        results = index2.search(query, top_k=5)
        assert len(results) == 5
        assert results[0][0] == "entity_0"

    def test_voyager_empty_index_error(self, voyager_available):
        """VoyagerIndex should raise error on empty embeddings."""
        from src.retrieval.backends.voyager_backend import VoyagerIndex

        index = VoyagerIndex(dim=128)
        with pytest.raises(ValueError, match="empty"):
            index.build_index({})

    def test_voyager_search_before_build_error(self, voyager_available):
        """VoyagerIndex should raise error if searching before build."""
        from src.retrieval.backends.voyager_backend import VoyagerIndex

        index = VoyagerIndex(dim=128)
        query = np.random.randn(128).astype(np.float32)
        with pytest.raises(RuntimeError, match="not built"):
            index.search(query, top_k=5)


# ==============================================================================
# Test: CuVSIndex (GPU)
# ==============================================================================
class TestCuVSIndex:
    """Tests for the cuVS GPU backend."""

    @pytest.fixture
    def cuvs_available(self):
        """Check if cuVS is available."""
        try:
            import cuvs
            import torch
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")
            return True
        except ImportError:
            pytest.skip("cuVS not installed")

    @pytest.fixture
    def sample_embeddings(self) -> Dict[str, np.ndarray]:
        """Generate sample embeddings for testing."""
        np.random.seed(42)
        dim = 128
        num_vectors = 1000  # cuVS needs more vectors for IVF
        return {
            f"entity_{i}": np.random.randn(dim).astype(np.float32)
            for i in range(num_vectors)
        }

    def test_cuvs_import(self, cuvs_available):
        """CuVSIndex should be importable."""
        from src.retrieval.backends.cuvs_backend import CuVSIndex
        assert CuVSIndex is not None

    def test_cuvs_init(self, cuvs_available):
        """CuVSIndex should initialize with default parameters."""
        from src.retrieval.backends.cuvs_backend import CuVSIndex

        index = CuVSIndex(dim=128, metric="ip")
        assert index.dim == 128
        assert index.backend_name == "cuvs"

    def test_cuvs_build_and_search(self, cuvs_available, sample_embeddings):
        """CuVSIndex should build index and perform search."""
        from src.retrieval.backends.cuvs_backend import CuVSIndex

        index = CuVSIndex(dim=128, metric="ip", n_lists=10)
        index.build_index(sample_embeddings)

        query = sample_embeddings["entity_0"]
        results = index.search(query, top_k=5)

        assert len(results) == 5
        # First result should be close to the query


# ==============================================================================
# Test: Factory Functions
# ==============================================================================
class TestFactoryFunctions:
    """Tests for index creation factory functions."""

    def test_resolve_backend_auto(self):
        """resolve_backend('auto') should return available backend."""
        from src.retrieval.vector_index import resolve_backend

        backend = resolve_backend("auto")
        assert backend in ("cuvs", "voyager")

    def test_resolve_backend_explicit(self):
        """resolve_backend should accept explicit backend names."""
        from src.retrieval.vector_index import resolve_backend

        # Voyager should always be available after deployment
        try:
            import voyager
            backend = resolve_backend("voyager")
            assert backend == "voyager"
        except ImportError:
            pytest.skip("voyager not installed")

    def test_resolve_backend_invalid(self):
        """resolve_backend should raise error for invalid backend."""
        from src.retrieval.vector_index import resolve_backend

        with pytest.raises(ValueError, match="not available"):
            resolve_backend("invalid_backend")

    def test_create_index_voyager(self):
        """create_index should create VoyagerIndex."""
        try:
            import voyager
        except ImportError:
            pytest.skip("voyager not installed")

        from src.retrieval.vector_index import create_index

        index = create_index(backend="voyager", dim=128)
        assert index.backend_name == "voyager"

    def test_create_index_auto(self):
        """create_index('auto') should create best available backend."""
        from src.retrieval.vector_index import create_index

        try:
            index = create_index(backend="auto", dim=128)
            assert index.backend_name in ("cuvs", "voyager")
        except ImportError:
            pytest.skip("No vector backend available")


# ==============================================================================
# Test: Module Exports
# ==============================================================================
class TestModuleExports:
    """Tests for module-level exports."""

    def test_retrieval_init_exports(self):
        """src/retrieval/__init__.py should export key functions."""
        from src.retrieval import create_index, resolve_backend

        assert callable(create_index)
        assert callable(resolve_backend)

    def test_backends_init_exports(self):
        """src/retrieval/backends/__init__.py should export backends."""
        from src.retrieval.backends import VectorIndexBase

        assert VectorIndexBase is not None


# ==============================================================================
# Test: Edge Cases
# ==============================================================================
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def mock_index(self):
        """Create a mock index for testing."""
        try:
            import voyager
            from src.retrieval.backends.voyager_backend import VoyagerIndex
            return VoyagerIndex(dim=64)
        except ImportError:
            pytest.skip("voyager not installed")

    def test_dimension_mismatch(self, mock_index):
        """Index should raise error on dimension mismatch."""
        embeddings = {
            "entity_0": np.random.randn(128).astype(np.float32)  # Wrong dim
        }
        with pytest.raises(ValueError, match="dimension"):
            mock_index.build_index(embeddings)

    def test_top_k_clamp(self, mock_index):
        """Search should clamp top_k to available vectors."""
        embeddings = {
            f"entity_{i}": np.random.randn(64).astype(np.float32)
            for i in range(5)
        }
        mock_index.build_index(embeddings)

        # Request more than available
        results = mock_index.search(embeddings["entity_0"], top_k=100)
        assert len(results) <= 5

    def test_contains_operator(self, mock_index):
        """Index should support 'in' operator."""
        embeddings = {
            "entity_0": np.random.randn(64).astype(np.float32)
        }
        mock_index.build_index(embeddings)

        assert "entity_0" in mock_index
        assert "nonexistent" not in mock_index

    def test_len_operator(self, mock_index):
        """Index should support len() operator."""
        embeddings = {
            f"entity_{i}": np.random.randn(64).astype(np.float32)
            for i in range(10)
        }
        mock_index.build_index(embeddings)

        assert len(mock_index) == 10
