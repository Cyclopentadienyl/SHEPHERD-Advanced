#!/usr/bin/env python3
"""
Validate Phase 1: the vectorized induced-subgraph construction in
SubgraphSampler._build_subgraph is BIT-IDENTICAL to the legacy Python loop, and
measure the speedup. Run on Spark (needs torch). Read-only; touches nothing.

    .venv/bin/python scripts/spikes/validate_fast_subgraph.py \
        --data-dir data/workspaces/hpo_2026-06

A PASS here means the optimization changes nothing the model sees — same nodes,
same edges, same local indices — it only computes them faster.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/workspaces/hpo_2026-06")
    ap.add_argument("--node-set-size", type=int, default=5000,
                    help="Total nodes in the test set (~ max_subgraph_nodes)")
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()

    import torch
    from src.kg.data_loader import SubgraphSampler, DataLoaderConfig

    d = Path(args.data_dir)
    edge_index_dict = torch.load(d / "edge_indices.pt", weights_only=True)
    with open(d / "num_nodes.json") as f:
        num_nodes = json.load(f)
    total_edges = sum(ei.size(1) for ei in edge_index_dict.values())
    print(f"edge types: {len(edge_index_dict)} | total directed edges: {total_edges}")

    print("building SubgraphSampler (one-time adjacency build; may take a few s)...")
    t = time.perf_counter()
    sampler = SubgraphSampler(edge_index_dict, num_nodes, DataLoaderConfig())
    print(f"  built in {time.perf_counter() - t:.1f}s")

    # A representative node set, sized per type proportionally to the KG.
    g = torch.Generator().manual_seed(0)
    sampled = {}
    total_nodes = sum(num_nodes.values())
    for nt, n in num_nodes.items():
        k = max(1, int(args.node_set_size * n / total_nodes))
        sampled[nt] = set(torch.randperm(n, generator=g)[:k].tolist())
    print("test node set sizes:", {nt: len(s) for nt, s in sampled.items()})

    # ---- correctness: vectorized (fast) vs legacy, must be bit-identical ----
    sampler.config.fast_subgraph_build = True
    nodes_f, edges_f, map_f = sampler._build_subgraph(sampled)
    sampler.config.fast_subgraph_build = False
    nodes_l, edges_l, map_l = sampler._build_subgraph(sampled)

    ok = True
    for nt in set(nodes_f) | set(nodes_l):
        if not torch.equal(nodes_f.get(nt, torch.tensor([])),
                           nodes_l.get(nt, torch.tensor([]))):
            ok = False; print("  NODES DIFFER:", nt)
    if set(edges_f) != set(edges_l):
        ok = False; print("  EDGE-TYPE KEYS DIFFER")
    for et in set(edges_f) & set(edges_l):
        if not torch.equal(edges_f[et], edges_l[et]):
            ok = False
            print(f"  EDGES DIFFER {et}: fast {tuple(edges_f[et].shape)} "
                  f"vs legacy {tuple(edges_l[et].shape)}")
    for nt in set(map_f) | set(map_l):
        a, b = map_f.get(nt), map_l.get(nt)
        if a is None or b is None or not torch.equal(a, b):
            ok = False; print("  MAPPING TENSORS DIFFER:", nt)

    n_edges = sum(e.size(1) for e in edges_f.values())
    print(f"\nBIT-IDENTICAL: {'YES (PASS)' if ok else 'NO (FAIL)'}   "
          f"| subgraph kept {n_edges} edges")

    # ---- speed ----
    def timeit(flag: bool) -> float:
        sampler.config.fast_subgraph_build = flag
        t0 = time.perf_counter()
        for _ in range(args.repeats):
            sampler._build_subgraph(sampled)
        return (time.perf_counter() - t0) / args.repeats * 1000

    fast_ms = timeit(True)
    legacy_ms = timeit(False)
    print(f"\n_build_subgraph timing (this is the per-batch hot path):")
    print(f"  legacy Python loop : {legacy_ms:8.1f} ms")
    print(f"  vectorized (fast)  : {fast_ms:8.1f} ms   -> {legacy_ms / max(fast_ms, 1e-9):.0f}x faster")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
