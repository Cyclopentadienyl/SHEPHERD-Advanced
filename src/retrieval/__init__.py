"""
SHEPHERD-Advanced Retrieval Module — IMPLEMENTED, detached from diagnosis.
==========================================================================
功能:
  - 向量索引工廠和後端管理
  - 自動偵測平台並選擇最佳後端
  - 支援 cuVS (Linux GPU) 和 Voyager (跨平台 CPU)

主要介面:
  - create_index(): 工廠函數，創建向量索引實例
  - resolve_backend(): 解析最佳可用後端
  - list_available_backends(): 列出所有可用後端

後端選擇策略:
  - Linux (x86/ARM): cuVS (GPU) → Voyager (CPU fallback)
  - Windows: Voyager (CPU only)

--------------------------------------------------------------------------
STATUS — read this before wiring anything to this package.

**Implemented and tested, but detached from the diagnosis pipeline.** This is
not an unimplemented placeholder: the backends work and have their own tests
(``tests/unit/test_retrieval.py``, ``tests/integration/test_retrieval_integration.py``,
``tests/integration/test_build_index.py``, and the platform benchmarks). What is
*reserved* here is the future natural-language / vector-mapping integration, not
the backend implementation.

Nothing under ``src/inference/`` depends on this package, and that is enforced:
``.import-linter.ini`` forbids ``src.inference``, ``src.api.routes.diagnose`` and
``src.webui`` from importing it, and ``tests/unit/test_vector_index_detachment.py``
pins the config, status and factory surfaces so it cannot be reconnected through
configuration either. "No diagnosis caller" therefore does not mean "unused".

Why it is kept: planned natural-language input and vector mapping.

PRECONDITION FOR REUSE — define the similarity contract first.
    ``VectorIndexBase.search()`` returns a raw backend *distance* with no defined
    direction, range, or normalisation (``backends/base.py``). Voyager (HNSW) and
    cuVS (IVF-Flat / IVF-PQ) need not agree on native semantics, so **no single
    caller-side conversion can be correct for both**. The diagnosis pipeline
    previously converted with ``(distance + 1) / 2`` as if the value were a cosine
    similarity; measurement showed Voyager's InnerProduct space returns ``1 - dot``,
    so the score fell as similarity rose. Any reuse must define the contract at the
    interface, not reinterpret backend-specific values at the call site.

    Relatedly: ``resolve_backend()`` selects a backend by import availability. That
    does not prove the backend can be constructed or searched — see
    ``scripts/validate_installation.py`` for the three states worth distinguishing.

WHAT IS REUSABLE, AND WHAT IS NOT.
    The backends and the factory here are general vector-index mechanics and are
    reusable as they stand. ``scripts/build_index.py`` is **not** a ready-made text
    pipeline — its CLI is ``--checkpoint / --data-dir / --embeddings / --node-types``,
    i.e. it builds an index from GNN node embeddings. A text-retrieval integration
    needs its own encoder, entity universe, metadata and compatibility contract.

Operator entry point: ``make vector-index ARGS="..."``.
Background and the decision record: ``docs/RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md``.

Module: src/retrieval/__init__.py
"""
from src.retrieval.vector_index import (
    create_index,
    create_index_from_config,
    get_index,
    list_available_backends,
    resolve_backend,
)
from src.retrieval.backends import VectorIndexBase, VoyagerIndex

# Re-export CuVSIndex only if available
try:
    from src.retrieval.backends import CuVSIndex
except ImportError:
    CuVSIndex = None  # type: ignore

__all__ = [
    # Factory functions
    "create_index",
    "create_index_from_config",
    "get_index",
    "resolve_backend",
    "list_available_backends",
    # Backend classes
    "VectorIndexBase",
    "VoyagerIndex",
    "CuVSIndex",
]
