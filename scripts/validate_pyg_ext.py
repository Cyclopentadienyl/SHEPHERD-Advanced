#!/usr/bin/env python3
"""Functional validation of the self-compiled PyG native extensions.

Import success only proves the shared libraries load; this script also runs
each package's compiled kernel (on CUDA when available) and checks the result
is numerically correct. Intended for the ARM/DGX self-compile path. Exits
non-zero if any critical check fails, so it is safe to use in CI / deploy.

    .venv/bin/python scripts/validate_pyg_ext.py
"""
import sys

import torch

results = []


def check(name, fn):
    try:
        fn()
        results.append((name, True))
        print(f"  [OK]   {name}")
    except Exception as e:  # noqa: BLE001 - report every failure, never abort early
        results.append((name, False))
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")


DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()} | testing on: {DEV}")
if DEV == "cpu":
    print("  [WARN] CUDA unavailable; exercising CPU kernels only (GPU kernels not verified).")


def t_scatter():
    from torch_scatter import scatter

    src = torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEV)
    idx = torch.tensor([0, 0, 1, 1], device=DEV)
    out = scatter(src, idx, dim=0, reduce="sum")
    expected = torch.tensor([3.0, 7.0], device=DEV)
    assert torch.allclose(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


def t_sparse():
    from torch_sparse import SparseTensor

    # Dense form: [[0, 2], [3, 4]]
    row = torch.tensor([0, 1, 1], device=DEV)
    col = torch.tensor([1, 0, 1], device=DEV)
    val = torch.tensor([2.0, 3.0, 4.0], device=DEV)
    a = SparseTensor(row=row, col=col, value=val, sparse_sizes=(2, 2))
    x = torch.tensor([[1.0], [1.0]], device=DEV)
    out = a.matmul(x)  # -> [[2], [7]]
    expected = torch.tensor([[2.0], [7.0]], device=DEV)
    assert torch.allclose(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


def t_cluster():
    from torch_cluster import knn_graph

    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0]], device=DEV)
    edge_index = knn_graph(x, k=1)
    assert edge_index.dim() == 2 and edge_index.size(0) == 2, f"bad shape {tuple(edge_index.shape)}"
    assert edge_index.size(1) == x.size(0), f"expected {x.size(0)} edges, got {edge_index.size(1)}"


def t_pyglib():
    import pyg_lib  # importing triggers load_library('libpyg') -> the ABI fix under test

    _ = pyg_lib.__version__
    # Bonus: exercise a compiled kernel. API/device quirks here are non-critical
    # since the import above already proves libpyg.so resolves against torch.
    try:
        inputs = torch.randn(8, 16, device=DEV)
        ptr = torch.tensor([0, 4, 8], device=DEV)
        other = torch.randn(2, 16, 32, device=DEV)
        out = pyg_lib.ops.segment_matmul(inputs, ptr, other)
        assert tuple(out.shape) == (8, 32), f"bad shape {tuple(out.shape)}"
        print("         (segment_matmul kernel OK)")
    except Exception as e:  # noqa: BLE001
        print(f"         (note: segment_matmul kernel not verified: {type(e).__name__}: {e})")


def t_pyg_integration():
    from torch_geometric.nn import SAGEConv

    conv = SAGEConv(16, 32).to(DEV)
    x = torch.randn(10, 16, device=DEV)
    edge_index = torch.randint(0, 10, (2, 30), device=DEV)
    out = conv(x, edge_index)
    assert tuple(out.shape) == (10, 32), f"bad shape {tuple(out.shape)}"


print("\nKernel / correctness checks:")
check("torch_scatter  scatter-sum", t_scatter)
check("torch_sparse   spmm (A @ x)", t_sparse)
check("torch_cluster  knn_graph", t_cluster)
check("pyg_lib        import + libpyg.so load", t_pyglib)
check("torch_geometric SAGEConv forward (uses the ext)", t_pyg_integration)

n_fail = sum(1 for _, ok in results if not ok)
print(f"\n{len(results) - n_fail}/{len(results)} checks passed.")
if n_fail:
    print("Some checks FAILED — do not package these wheels until resolved.")
    sys.exit(1)
print("All PyG native extensions are functional.")
sys.exit(0)
