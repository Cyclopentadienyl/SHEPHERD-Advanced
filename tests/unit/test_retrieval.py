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
        """Skip unless the cuVS backend can actually be constructed.

        The gate must check everything ``CuVSIndex._validate_cuvs`` imports, not just ``cuvs``:
        the backend also needs ``cupy``, which ``cuvs-cu13`` does not declare as a dependency.
        Checking only ``cuvs`` let a cuVS-without-cupy machine past the gate and turned a
        missing optional GPU dependency into a hard test error instead of a skip.
        """
        for module, hint in (
            ("cuvs", "cuVS not installed"),
            ("cupy", "cupy not installed — the cuVS backend imports it, but cuvs-cu13 does not "
                     "pull it in; install it to exercise the GPU vector index"),
        ):
            try:
                __import__(module)
            except ImportError:
                pytest.skip(hint)
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return True

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
        """resolve_backend('auto') returns a backend that is actually registered.

        Both backends are optional in the current deployment, so this asserts
        against the registry rather than against a hardcoded pair — and refuses to
        be satisfied by a name that is not in it, which is the defect that stood
        here. `TestBackendRegistrationGating` drives the same call with controlled
        observations instead of installed packages."""
        from src.retrieval.vector_index import list_available_backends, resolve_backend

        available = list_available_backends()
        if not available:
            pytest.skip("no vector index backend is installed")

        assert resolve_backend("auto") in available

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
        """create_index('auto') builds the backend `auto` resolved to.

        **No ImportError is caught here.** The previous version wrapped this call
        and skipped with "No vector backend available" — which on the deployment
        machine was false: voyager was installed and `test_create_index_voyager`
        passed in the same run. What had actually happened was that `auto`
        resolved to an unregistered cuVS. Swallowing the error turned that into
        green, and it is the reason the defect lived on the one platform where it
        mattered. An ImportError here is now a failure."""
        from src.retrieval.vector_index import list_available_backends, create_index

        available = list_available_backends()
        if not available:
            pytest.skip("no vector index backend is installed")

        index = create_index(backend="auto", dim=128)
        assert index.backend_name in available


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


# ==============================================================================
# Test: Backend registration is gated on discoverable dependencies
# ==============================================================================
# Found on the deployment machine: `test_create_index_auto` skipped there and
# passed on a CPU container — a test running backwards, because `auto` resolves to
# cuVS when CUDA is present and cuVS was registered without being installed.
#
# The cause was that `_register_backends` guarded on whether the backend *module*
# imports. Both backends import their dependency lazily inside a method, so that
# guard never fired for either of them and every backend was always registered.
# Registration feeds selection, the fallback chain and the operator report, so one
# wrong registration was three wrong answers.
#
# These drive registration through **controlled observations** — a fake
# discoverability answer, a fake platform, a fake GPU answer — rather than through
# what happens to be installed in CI. A test that depends on the environment is
# how this defect stayed invisible on every machine but one.
# ==============================================================================
class TestBackendRegistrationGating:
    """`_register_backends` must register only backends whose immediate required
    packages are discoverable.

    Deliberately narrower than "could actually be built": discovery does not prove
    a package imports, that a CUDA build matches the driver, or that an index
    constructs. Saying the wider thing here would misdescribe the contract the
    implementation was scoped to."""

    @staticmethod
    def _reregister(monkeypatch, *, present, platform="linux"):
        """Rebuild the registry against a stated set of discoverable packages."""
        from src.retrieval import vector_index as vi

        monkeypatch.setattr(vi.sys, "platform", platform)
        monkeypatch.setattr(vi, "_discoverable", lambda pkg: pkg in present)
        vi._register_backends()
        return vi

    @pytest.fixture(autouse=True)
    def _restore_registry(self):
        """The registry is module state, so every case here puts it back."""
        from src.retrieval import vector_index as vi

        saved = dict(vi._BACKEND_REGISTRY)
        yield
        vi._BACKEND_REGISTRY.clear()
        vi._BACKEND_REGISTRY.update(saved)

    def test_auto_selects_voyager_when_a_gpu_is_present_but_cuvs_is_not(self, monkeypatch):
        """The observed failure. CUDA presence routes `auto` to cuVS, so cuVS being
        absent must be visible at registration or the fallback chain never runs."""
        vi = self._reregister(monkeypatch, present={"voyager"})
        monkeypatch.setattr(vi, "is_gpu_available", lambda: True)

        assert vi.list_available_backends() == ["voyager"]
        assert vi.resolve_backend("auto") == "voyager"

    def test_cuvs_without_cupy_is_not_registered(self, monkeypatch):
        """`CuVSIndex._validate_cuvs` imports cuvs.neighbors **and** cupy, and
        `validate_installation.py` records cuvs-without-cupy as a state observed on
        a real deployment machine. Discovering cuvs alone would preserve exactly
        that failure."""
        vi = self._reregister(monkeypatch, present={"voyager", "cuvs"})
        monkeypatch.setattr(vi, "is_gpu_available", lambda: True)

        assert "cuvs" not in vi.list_available_backends()
        assert vi.resolve_backend("auto") == "voyager"

    def test_cuvs_is_registered_when_both_packages_are_discoverable(self, monkeypatch):
        """The positive case, so the gate is not merely refusing everything."""
        vi = self._reregister(monkeypatch, present={"voyager", "cuvs", "cupy"})
        monkeypatch.setattr(vi, "is_gpu_available", lambda: True)

        assert sorted(vi.list_available_backends()) == ["cuvs", "voyager"]
        assert vi.resolve_backend("auto") == "cuvs"

    def test_voyager_absent_is_not_listed(self, monkeypatch):
        """Voyager is gated by discovery on every platform. It is installed
        everywhere looked at so far, which is why its identical latent bug was
        invisible — not a reason to exempt it."""
        vi = self._reregister(monkeypatch, present={"cuvs", "cupy"})

        assert "voyager" not in vi.list_available_backends()

    def test_no_backend_at_all_raises_rather_than_returning_one(self, monkeypatch):
        """The existing no-backend RuntimeError, which was unreachable while every
        backend was registered unconditionally."""
        vi = self._reregister(monkeypatch, present=set())
        monkeypatch.setattr(vi, "is_gpu_available", lambda: False)

        assert vi.list_available_backends() == []
        with pytest.raises(RuntimeError, match="No available vector index backend"):
            vi.resolve_backend("auto")

    def test_an_explicitly_requested_absent_backend_fails_early(self, monkeypatch):
        """Previously this returned "cuvs" and failed later inside `create_index`
        on `import cuvs`. Failing at resolution names the problem where the caller
        asked the question, and lists what it could have asked for instead."""
        vi = self._reregister(monkeypatch, present={"voyager"})

        with pytest.raises(ValueError, match="not available") as caught:
            vi.resolve_backend("cuvs")
        assert "cuvs" not in str(caught.value).split("Valid options:")[1]

    def test_cuvs_stays_unregistered_on_windows_even_when_discoverable(self, monkeypatch):
        """A support statement, not a dependency one: cuVS does not run on Windows
        whatever pip reports."""
        vi = self._reregister(monkeypatch, present={"voyager", "cuvs", "cupy"},
                              platform="win32")

        assert vi.list_available_backends() == ["voyager"]

    def test_registration_drops_a_backend_whose_dependency_has_gone(self, monkeypatch):
        """Rebuilt, not accumulated. A backend registered by an earlier call must
        not survive as falsely available once discovery says it is gone."""
        vi = self._reregister(monkeypatch, present={"voyager", "cuvs", "cupy"})
        assert "cuvs" in vi.list_available_backends()

        registry_before = vi._BACKEND_REGISTRY
        self._reregister(monkeypatch, present={"voyager"})

        assert "cuvs" not in vi.list_available_backends()
        # Cleared in place rather than rebound, so a caller holding the dict is not
        # left reading a detached copy of the old answer.
        assert vi._BACKEND_REGISTRY is registry_before

    def test_a_discovery_error_fails_closed_without_breaking_registration(self, monkeypatch):
        """`find_spec` raises for ordinary conditions — ModuleNotFoundError when a
        parent package is absent, ValueError for an imported module with no
        __spec__. An optional-dependency probe must never be the reason
        `import src.retrieval` fails."""
        from src.retrieval import vector_index as vi

        def exploding(package):
            if package == "voyager":
                return True
            raise ModuleNotFoundError(f"No module named {package!r}")

        monkeypatch.setattr(vi.importlib.util, "find_spec", exploding)
        vi._register_backends()

        assert vi.list_available_backends() == ["voyager"]

    @pytest.mark.parametrize("raised", [ImportError("x"), ValueError("no __spec__")])
    def test_discoverable_swallows_the_discovery_exceptions(self, monkeypatch, raised):
        from src.retrieval import vector_index as vi

        def exploding(_package):
            raise raised

        monkeypatch.setattr(vi.importlib.util, "find_spec", exploding)
        assert vi._discoverable("anything") is False
