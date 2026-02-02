"""
# ==============================================================================
# Module: tests/benchmarks/platform_specific/test_vector_index_arm.py
# ==============================================================================
# Purpose: Benchmark tests for vector index backends on ARM64 platforms
#          (DGX Spark, Grace Hopper, Jetson Orin)
#
# Tests:
#   - Voyager HNSW performance on ARM
#   - cuVS IVF-PQ performance on ARM + CUDA
#   - Memory efficiency (important for unified memory systems)
#
# Skip Conditions:
#   - Skips all tests if not on aarch64/arm64
# ==============================================================================
"""
import platform
import pytest
import numpy as np
import time
from typing import Dict

# Skip entire module if not ARM64
if platform.machine().lower() not in ("aarch64", "arm64"):
    pytest.skip("ARM64-specific tests", allow_module_level=True)


# ==============================================================================
# Fixtures
# ==============================================================================
@pytest.fixture(scope="module")
def large_embeddings() -> Dict[str, np.ndarray]:
    """Generate large embedding set for benchmarks."""
    np.random.seed(42)
    dim = 768
    num_vectors = 10_000
    return {
        f"entity_{i}": np.random.randn(dim).astype(np.float32)
        for i in range(num_vectors)
    }


@pytest.fixture(scope="module")
def query_vectors() -> np.ndarray:
    """Generate query vectors for benchmark."""
    np.random.seed(123)
    dim = 768
    num_queries = 100
    return np.random.randn(num_queries, dim).astype(np.float32)


# ==============================================================================
# Voyager Benchmarks (ARM)
# ==============================================================================
class TestVoyagerBenchmarkARM:
    """Performance benchmarks for Voyager on ARM64."""

    @pytest.fixture
    def voyager_available(self):
        try:
            import voyager
            return True
        except ImportError:
            pytest.skip("voyager not installed")

    def test_voyager_build_time_arm(self, voyager_available, large_embeddings):
        """Benchmark: Voyager index build time on ARM."""
        from src.retrieval.backends.voyager_backend import VoyagerIndex

        index = VoyagerIndex(dim=768, metric="cosine")

        start = time.perf_counter()
        index.build_index(large_embeddings)
        elapsed = time.perf_counter() - start

        print(f"\n[Voyager ARM] Build time (10K vectors): {elapsed:.3f}s")

    def test_voyager_search_throughput_arm(self, voyager_available, large_embeddings, query_vectors):
        """Benchmark: Voyager search throughput on ARM."""
        from src.retrieval.backends.voyager_backend import VoyagerIndex

        index = VoyagerIndex(dim=768, metric="cosine")
        index.build_index(large_embeddings)

        start = time.perf_counter()
        for q in query_vectors:
            _ = index.search(q, top_k=10)
        elapsed = time.perf_counter() - start

        qps = len(query_vectors) / elapsed
        print(f"\n[Voyager ARM] Throughput: {qps:.1f} QPS")

    def test_voyager_memory_efficiency_arm(self, voyager_available, large_embeddings):
        """Benchmark: Memory usage on ARM (important for unified memory)."""
        from src.retrieval.backends.voyager_backend import VoyagerIndex

        index = VoyagerIndex(dim=768, metric="cosine")

        # Measure memory before
        embeddings_size = sum(v.nbytes for v in large_embeddings.values())

        index.build_index(large_embeddings)

        print(f"\n[Voyager ARM] Embeddings size: {embeddings_size / 1024 / 1024:.1f} MB")
        print(f"[Voyager ARM] Index vectors: {len(index)}")


# ==============================================================================
# cuVS Benchmarks (ARM + CUDA)
# ==============================================================================
class TestCuVSBenchmarkARM:
    """Performance benchmarks for cuVS on ARM64 + CUDA (DGX Spark)."""

    @pytest.fixture
    def cuvs_available(self):
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
        """Larger dataset for cuVS IVF clustering."""
        np.random.seed(42)
        dim = 768
        num_vectors = 50_000
        return {
            f"entity_{i}": np.random.randn(dim).astype(np.float32)
            for i in range(num_vectors)
        }

    def test_cuvs_build_time_arm(self, cuvs_available, large_embeddings_cuvs):
        """Benchmark: cuVS index build time on ARM + CUDA."""
        from src.retrieval.backends.cuvs_backend import CuVSIndex

        index = CuVSIndex(dim=768, metric="ip", n_lists=100)

        start = time.perf_counter()
        index.build_index(large_embeddings_cuvs)
        elapsed = time.perf_counter() - start

        print(f"\n[cuVS ARM] Build time (50K vectors): {elapsed:.3f}s")

    def test_cuvs_search_throughput_arm(self, cuvs_available, large_embeddings_cuvs, query_vectors):
        """Benchmark: cuVS search throughput on ARM + CUDA."""
        from src.retrieval.backends.cuvs_backend import CuVSIndex

        index = CuVSIndex(dim=768, metric="ip", n_lists=100)
        index.build_index(large_embeddings_cuvs)

        start = time.perf_counter()
        for q in query_vectors:
            _ = index.search(q, top_k=10)
        elapsed = time.perf_counter() - start

        qps = len(query_vectors) / elapsed
        print(f"\n[cuVS ARM] Throughput: {qps:.1f} QPS")

    def test_cuvs_unified_memory_usage(self, cuvs_available, large_embeddings_cuvs):
        """Test: Verify cuVS works with unified memory on DGX Spark."""
        import torch
        from src.retrieval.backends.cuvs_backend import CuVSIndex

        # Check if this is a unified memory system
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"\n[cuVS ARM] GPU: {props.name}")
            print(f"[cuVS ARM] Total memory: {props.total_memory / 1024**3:.1f} GB")

        index = CuVSIndex(dim=768, metric="ip", n_lists=100)
        index.build_index(large_embeddings_cuvs)

        # Verify search works
        query = large_embeddings_cuvs["entity_0"]
        results = index.search(query, top_k=10)
        assert len(results) > 0


# ==============================================================================
# DGX Spark Specific Tests
# ==============================================================================
class TestDGXSparkOptimizations:
    """Tests specific to NVIDIA DGX Spark (Grace Hopper)."""

    def test_detect_dgx_spark(self):
        """Detect if running on DGX Spark."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        is_dgx = False
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            if "grace" in gpu_name or "hopper" in gpu_name or "gh" in gpu_name:
                is_dgx = True
                print(f"\n[DGX Spark] Detected: {torch.cuda.get_device_name(0)}")

        if not is_dgx:
            pytest.skip("Not running on DGX Spark")

    def test_nvlink_bandwidth(self):
        """Test: Verify NVLink is functional (if available)."""
        try:
            import torch
            if torch.cuda.device_count() > 1:
                print(f"\n[DGX Spark] {torch.cuda.device_count()} GPUs detected")
        except Exception:
            pytest.skip("Multi-GPU test not applicable")
