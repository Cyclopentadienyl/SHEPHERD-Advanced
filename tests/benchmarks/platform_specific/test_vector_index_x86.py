"""
# ==============================================================================
# Module: tests/benchmarks/platform_specific/test_vector_index_x86.py
# ==============================================================================
# Purpose: Benchmark tests for vector index backends on x86_64 platforms
#
# Tests:
#   - Voyager HNSW performance benchmarks
#   - cuVS IVF-PQ performance benchmarks (if CUDA available)
#   - Memory usage profiling
#   - Throughput measurements
#
# Skip Conditions:
#   - Skips cuVS tests if not on Linux or no CUDA
#   - Skips all tests if not on x86_64
# ==============================================================================
"""
import platform
import pytest
import numpy as np
import time
from typing import Dict

# Skip entire module if not x86_64
if platform.machine() not in ("x86_64", "AMD64"):
    pytest.skip("x86_64-specific tests", allow_module_level=True)


# ==============================================================================
# Fixtures
# ==============================================================================
@pytest.fixture(scope="module")
def large_embeddings() -> Dict[str, np.ndarray]:
    """Generate large embedding set for benchmarks (10K vectors)."""
    np.random.seed(42)
    dim = 768  # Common embedding dimension
    num_vectors = 10_000
    return {
        f"entity_{i}": np.random.randn(dim).astype(np.float32)
        for i in range(num_vectors)
    }


@pytest.fixture(scope="module")
def query_vectors(large_embeddings) -> np.ndarray:
    """Generate query vectors for benchmark."""
    np.random.seed(123)
    dim = 768
    num_queries = 100
    return np.random.randn(num_queries, dim).astype(np.float32)


# ==============================================================================
# Voyager Benchmarks
# ==============================================================================
class TestVoyagerBenchmark:
    """Performance benchmarks for Voyager HNSW backend."""

    @pytest.fixture
    def voyager_available(self):
        try:
            import voyager
            return True
        except ImportError:
            pytest.skip("voyager not installed")

    def test_voyager_build_time(self, voyager_available, large_embeddings):
        """Benchmark: Voyager index build time."""
        from src.retrieval.backends.voyager_backend import VoyagerIndex

        index = VoyagerIndex(dim=768, metric="cosine", ef_construction=200, M=16)

        start = time.perf_counter()
        index.build_index(large_embeddings)
        elapsed = time.perf_counter() - start

        print(f"\n[Voyager] Build time (10K vectors): {elapsed:.3f}s")
        assert elapsed < 60, "Build should complete within 60s"

    def test_voyager_search_throughput(self, voyager_available, large_embeddings, query_vectors):
        """Benchmark: Voyager search throughput (QPS)."""
        from src.retrieval.backends.voyager_backend import VoyagerIndex

        index = VoyagerIndex(dim=768, metric="cosine")
        index.build_index(large_embeddings)

        # Warm-up
        _ = index.search(query_vectors[0], top_k=10)

        # Benchmark single queries
        start = time.perf_counter()
        for q in query_vectors:
            _ = index.search(q, top_k=10)
        elapsed = time.perf_counter() - start

        qps = len(query_vectors) / elapsed
        print(f"\n[Voyager] Single query throughput: {qps:.1f} QPS")
        assert qps > 100, "Should achieve >100 QPS"

    def test_voyager_batch_throughput(self, voyager_available, large_embeddings, query_vectors):
        """Benchmark: Voyager batch search throughput."""
        from src.retrieval.backends.voyager_backend import VoyagerIndex

        index = VoyagerIndex(dim=768, metric="cosine")
        index.build_index(large_embeddings)

        queries = [query_vectors[i] for i in range(len(query_vectors))]

        start = time.perf_counter()
        _ = index.batch_search(queries, top_k=10)
        elapsed = time.perf_counter() - start

        qps = len(queries) / elapsed
        print(f"\n[Voyager] Batch query throughput: {qps:.1f} QPS")

    def test_voyager_recall_at_10(self, voyager_available, large_embeddings):
        """Benchmark: Voyager recall@10 accuracy."""
        from src.retrieval.backends.voyager_backend import VoyagerIndex

        index = VoyagerIndex(dim=768, metric="cosine")
        index.build_index(large_embeddings)

        # Test recall: query with known vectors
        correct = 0
        for i in range(100):
            key = f"entity_{i}"
            query = large_embeddings[key]
            results = index.search(query, top_k=10)
            if results[0][0] == key:
                correct += 1

        recall = correct / 100
        print(f"\n[Voyager] Recall@1: {recall:.2%}")
        assert recall > 0.95, "Recall@1 should be >95%"


# ==============================================================================
# cuVS Benchmarks (Linux + CUDA only)
# ==============================================================================
class TestCuVSBenchmark:
    """Performance benchmarks for cuVS GPU backend."""

    @pytest.fixture
    def cuvs_available(self):
        import sys
        if sys.platform == "win32":
            pytest.skip("cuVS not available on Windows")
        try:
            import cuvs
            import torch
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")
            return True
        except ImportError:
            pytest.skip("cuVS not installed")

    @pytest.fixture(scope="class")
    def large_embeddings_cuvs(self) -> Dict[str, np.ndarray]:
        """cuVS needs more vectors for IVF clustering."""
        np.random.seed(42)
        dim = 768
        num_vectors = 50_000
        return {
            f"entity_{i}": np.random.randn(dim).astype(np.float32)
            for i in range(num_vectors)
        }

    def test_cuvs_build_time(self, cuvs_available, large_embeddings_cuvs):
        """Benchmark: cuVS index build time."""
        from src.retrieval.backends.cuvs_backend import CuVSIndex

        index = CuVSIndex(dim=768, metric="ip", n_lists=100)

        start = time.perf_counter()
        index.build_index(large_embeddings_cuvs)
        elapsed = time.perf_counter() - start

        print(f"\n[cuVS] Build time (50K vectors): {elapsed:.3f}s")

    def test_cuvs_search_throughput(self, cuvs_available, large_embeddings_cuvs, query_vectors):
        """Benchmark: cuVS search throughput."""
        from src.retrieval.backends.cuvs_backend import CuVSIndex

        index = CuVSIndex(dim=768, metric="ip", n_lists=100)
        index.build_index(large_embeddings_cuvs)

        # Warm-up
        _ = index.search(query_vectors[0], top_k=10)

        start = time.perf_counter()
        for q in query_vectors:
            _ = index.search(q, top_k=10)
        elapsed = time.perf_counter() - start

        qps = len(query_vectors) / elapsed
        print(f"\n[cuVS] Single query throughput: {qps:.1f} QPS")


# ==============================================================================
# Comparison Benchmark
# ==============================================================================
class TestBackendComparison:
    """Compare performance between backends."""

    def test_backend_comparison(self, large_embeddings):
        """Compare available backends."""
        results = {}

        # Test Voyager
        try:
            from src.retrieval.backends.voyager_backend import VoyagerIndex
            index = VoyagerIndex(dim=768)
            start = time.perf_counter()
            index.build_index(large_embeddings)
            results["voyager_build"] = time.perf_counter() - start

            query = large_embeddings["entity_0"]
            start = time.perf_counter()
            for _ in range(100):
                _ = index.search(query, top_k=10)
            results["voyager_search_100"] = time.perf_counter() - start
        except ImportError:
            pass

        # Test cuVS (if available)
        try:
            import sys
            if sys.platform != "win32":
                import torch
                if torch.cuda.is_available():
                    from src.retrieval.backends.cuvs_backend import CuVSIndex
                    index = CuVSIndex(dim=768, n_lists=50)
                    start = time.perf_counter()
                    index.build_index(large_embeddings)
                    results["cuvs_build"] = time.perf_counter() - start

                    query = large_embeddings["entity_0"]
                    start = time.perf_counter()
                    for _ in range(100):
                        _ = index.search(query, top_k=10)
                    results["cuvs_search_100"] = time.perf_counter() - start
        except ImportError:
            pass

        print("\n=== Backend Comparison ===")
        for k, v in results.items():
            print(f"  {k}: {v:.3f}s")

        assert len(results) > 0, "At least one backend should be available"
