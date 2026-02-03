"""
# ==============================================================================
# Module: tests/integration/test_retrieval_integration.py
# ==============================================================================
# Purpose: Integration tests for vector retrieval module
#
# Tests the full workflow:
#   1. Create embeddings (simulating KG entity embeddings)
#   2. Build vector index
#   3. Save index to disk
#   4. Load index in new instance
#   5. Perform searches and validate results
#   6. Verify output format compatibility with downstream consumers
# ==============================================================================
"""
import pytest
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import tempfile

# ==============================================================================
# Fixtures
# ==============================================================================
@pytest.fixture
def realistic_embeddings() -> Dict[str, np.ndarray]:
    """
    Generate realistic embeddings simulating KG entities.

    Simulates:
    - Phenotypes (symptoms): HPO-like IDs
    - Genes: HGNC-like IDs
    - Diseases: OMIM-like IDs
    """
    np.random.seed(42)
    dim = 768  # Standard embedding dimension

    embeddings = {}

    # Phenotypes (100 entities)
    for i in range(100):
        entity_id = f"HP:{i:07d}"
        embeddings[entity_id] = np.random.randn(dim).astype(np.float32)

    # Genes (200 entities)
    for i in range(200):
        entity_id = f"HGNC:{i+1000}"
        embeddings[entity_id] = np.random.randn(dim).astype(np.float32)

    # Diseases (50 entities)
    for i in range(50):
        entity_id = f"OMIM:{i+100000}"
        embeddings[entity_id] = np.random.randn(dim).astype(np.float32)

    return embeddings


@pytest.fixture
def query_embeddings(realistic_embeddings) -> Dict[str, np.ndarray]:
    """Generate query embeddings (subset of known entities + noise)."""
    queries = {}

    # Use some known entity embeddings as queries
    known_ids = ["HP:0000001", "HGNC:1050", "OMIM:100010"]
    for eid in known_ids:
        if eid in realistic_embeddings:
            queries[eid] = realistic_embeddings[eid].copy()

    # Add a novel query (not in index)
    queries["novel_query"] = np.random.randn(768).astype(np.float32)

    return queries


# ==============================================================================
# Test: Full Integration Flow
# ==============================================================================
class TestRetrievalIntegration:
    """Integration tests for vector retrieval workflow."""

    @pytest.fixture
    def voyager_available(self):
        """Check if voyager is available."""
        try:
            import voyager
            return True
        except ImportError:
            pytest.skip("voyager not installed")

    def test_full_workflow(self, voyager_available, realistic_embeddings, query_embeddings, tmp_path):
        """
        Test complete workflow: build -> save -> load -> search.

        This validates:
        1. Index can be built from Dict[str, np.ndarray]
        2. Index can be saved to disk
        3. Index can be loaded into new instance
        4. Search returns correct format: List[Tuple[str, float]]
        5. Known queries return themselves as top result
        """
        from src.retrieval import create_index

        # Step 1: Create and build index
        index = create_index(backend="voyager", dim=768, metric="cosine")
        index.build_index(realistic_embeddings)

        assert len(index) == 350  # 100 + 200 + 50

        # Step 2: Save index
        save_path = tmp_path / "test_kg_index"
        index.save(save_path)

        # Verify files exist
        assert (save_path.with_suffix(".voyager")).exists()
        assert (save_path.with_suffix(".ids.json")).exists()

        # Step 3: Load into new instance
        index2 = create_index(backend="voyager", dim=768, metric="cosine")
        index2.load(save_path)

        assert len(index2) == 350

        # Step 4: Perform searches
        for query_id, query_vec in query_embeddings.items():
            results = index2.search(query_vec, top_k=10)

            # Validate output format
            assert isinstance(results, list)
            assert len(results) <= 10

            for entity_id, score in results:
                assert isinstance(entity_id, str)
                assert isinstance(score, float)

            # If query is a known entity, it should be top result
            if query_id in realistic_embeddings:
                assert results[0][0] == query_id, f"Expected {query_id} as top result"

    def test_batch_search_output_format(self, voyager_available, realistic_embeddings, tmp_path):
        """
        Test batch search returns correct format for downstream consumers.

        Expected format:
        [
            [(entity_id, score), (entity_id, score), ...],  # Query 1 results
            [(entity_id, score), (entity_id, score), ...],  # Query 2 results
            ...
        ]
        """
        from src.retrieval import create_index

        index = create_index(backend="voyager", dim=768, metric="cosine")
        index.build_index(realistic_embeddings)

        # Batch of 5 queries
        query_ids = ["HP:0000001", "HP:0000010", "HGNC:1000", "HGNC:1100", "OMIM:100000"]
        queries = [realistic_embeddings[qid] for qid in query_ids]

        results = index.batch_search(queries, top_k=5)

        # Validate structure
        assert isinstance(results, list)
        assert len(results) == 5

        for i, query_results in enumerate(results):
            assert isinstance(query_results, list)
            assert len(query_results) == 5

            # Each result is (entity_id, score)
            for entity_id, score in query_results:
                assert isinstance(entity_id, str)
                assert isinstance(score, float)

            # Top result should be the query itself
            assert query_results[0][0] == query_ids[i]

    def test_search_results_can_be_serialized(self, voyager_available, realistic_embeddings):
        """
        Test that search results can be JSON serialized.

        This is important for:
        - API responses
        - Caching
        - Logging

        Note: JSON doesn't have tuples, so tuples become lists after round-trip.
        """
        from src.retrieval import create_index

        index = create_index(backend="voyager", dim=768, metric="cosine")
        index.build_index(realistic_embeddings)

        query = realistic_embeddings["HP:0000001"]
        results = index.search(query, top_k=5)

        # Should be JSON serializable
        json_str = json.dumps(results)
        parsed = json.loads(json_str)

        # JSON converts tuples to lists, so compare values not types
        assert len(parsed) == len(results)
        for i, (entity_id, score) in enumerate(results):
            assert parsed[i][0] == entity_id
            assert parsed[i][1] == score

    def test_entity_id_preservation(self, voyager_available, realistic_embeddings, tmp_path):
        """
        Test that entity IDs are preserved exactly after save/load.

        Critical for:
        - Mapping back to KG nodes
        - Downstream entity linking
        """
        from src.retrieval import create_index

        # Build and save
        index = create_index(backend="voyager", dim=768)
        index.build_index(realistic_embeddings)
        index.save(tmp_path / "id_test")

        # Load
        index2 = create_index(backend="voyager", dim=768)
        index2.load(tmp_path / "id_test")

        # Check all entity IDs are preserved
        for entity_id in realistic_embeddings:
            assert entity_id in index2, f"Entity {entity_id} not found after load"

    def test_contains_operator(self, voyager_available, realistic_embeddings):
        """Test that 'in' operator works for entity lookup."""
        from src.retrieval import create_index

        index = create_index(backend="voyager", dim=768)
        index.build_index(realistic_embeddings)

        assert "HP:0000001" in index
        assert "HGNC:1000" in index
        assert "OMIM:100000" in index
        assert "nonexistent_entity" not in index


# ==============================================================================
# Test: Backend Resolution
# ==============================================================================
class TestBackendResolution:
    """Test automatic backend selection."""

    def test_resolve_auto_returns_valid_backend(self):
        """resolve_backend('auto') should return an available backend."""
        from src.retrieval import resolve_backend, list_available_backends

        backend = resolve_backend("auto")
        available = list_available_backends()

        assert backend in available

    def test_list_available_backends(self):
        """list_available_backends() should return non-empty list."""
        from src.retrieval import list_available_backends

        backends = list_available_backends()

        assert isinstance(backends, list)
        assert len(backends) >= 1  # At least voyager should be available
        assert "voyager" in backends  # Voyager is always available after deployment


# ==============================================================================
# Test: Error Handling
# ==============================================================================
class TestErrorHandling:
    """Test error handling for edge cases."""

    @pytest.fixture
    def voyager_available(self):
        try:
            import voyager
            return True
        except ImportError:
            pytest.skip("voyager not installed")

    def test_search_before_build_raises_error(self, voyager_available):
        """Searching before build should raise RuntimeError."""
        from src.retrieval import create_index

        index = create_index(backend="voyager", dim=768)
        query = np.random.randn(768).astype(np.float32)

        with pytest.raises(RuntimeError, match="not built"):
            index.search(query, top_k=5)

    def test_load_nonexistent_raises_error(self, voyager_available, tmp_path):
        """Loading non-existent index should raise FileNotFoundError."""
        from src.retrieval import create_index

        index = create_index(backend="voyager", dim=768)

        with pytest.raises(FileNotFoundError):
            index.load(tmp_path / "nonexistent_index")

    def test_dimension_mismatch_raises_error(self, voyager_available):
        """Building with wrong dimension should raise ValueError."""
        from src.retrieval import create_index

        index = create_index(backend="voyager", dim=768)

        # Embeddings with wrong dimension
        wrong_dim_embeddings = {
            "entity_1": np.random.randn(512).astype(np.float32),  # Wrong dim
        }

        with pytest.raises(ValueError, match="dimension"):
            index.build_index(wrong_dim_embeddings)


# ==============================================================================
# Test: Performance Smoke Test
# ==============================================================================
class TestPerformanceSmokeTest:
    """Basic performance smoke tests."""

    @pytest.fixture
    def voyager_available(self):
        try:
            import voyager
            return True
        except ImportError:
            pytest.skip("voyager not installed")

    def test_build_1k_vectors_completes(self, voyager_available):
        """Building index with 1K vectors should complete quickly."""
        from src.retrieval import create_index
        import time

        np.random.seed(42)
        embeddings = {
            f"entity_{i}": np.random.randn(768).astype(np.float32)
            for i in range(1000)
        }

        index = create_index(backend="voyager", dim=768)

        start = time.perf_counter()
        index.build_index(embeddings)
        elapsed = time.perf_counter() - start

        # Should complete in under 5 seconds
        assert elapsed < 5.0, f"Build took too long: {elapsed:.2f}s"

    def test_search_latency(self, voyager_available):
        """Single search should complete in < 10ms."""
        from src.retrieval import create_index
        import time

        np.random.seed(42)
        embeddings = {
            f"entity_{i}": np.random.randn(768).astype(np.float32)
            for i in range(1000)
        }

        index = create_index(backend="voyager", dim=768)
        index.build_index(embeddings)

        query = np.random.randn(768).astype(np.float32)

        # Warm up
        index.search(query, top_k=10)

        # Measure
        start = time.perf_counter()
        for _ in range(100):
            index.search(query, top_k=10)
        elapsed = time.perf_counter() - start

        avg_latency_ms = (elapsed / 100) * 1000

        # Should be under 10ms per query
        assert avg_latency_ms < 10.0, f"Search latency too high: {avg_latency_ms:.2f}ms"
