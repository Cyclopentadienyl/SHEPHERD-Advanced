#!/usr/bin/env python3
"""
FEASIBILITY SPIKE (temporary) — native subgraph sampling for the GNN pipeline.

Goal: empirically answer, on real hardware with torch available, whether a PyG
*native* (C++, GIL-releasing) sampler can replace the pure-Python per-batch
subgraph sampling in src/kg/data_loader.py without losing the heterogeneous
edge-type structure the trainer needs — and how much faster it is.

This script is standalone and READ-ONLY w.r.t. the training pipeline. It does
not import or modify the trainer/data_loader. Safe to run on Spark or Windows.
Delete after the spike concludes.

It checks three things:
  (A) torch_sparse.SparseTensor.sample_adj (the "surgical" path): does the C++
      sampler preserve enough info to recover per-edge-type subgraph edges, via
      an edge-id / edge-type value carried through the sample?  + speed.
  (B) torch_geometric.loader.NeighborLoader (the "canonical" path): does it
      produce a HeteroData mini-batch with per-edge-type edge_index_dict, and
      can it seed from multiple node types at once?
  (C) Python baseline mirroring _build_subgraph's hot loop, for a speed ratio.

Usage (on Spark, in the repo root):
    .venv/bin/python scripts/spikes/native_sampler_feasibility.py \
        --data-dir data/workspaces/hpo_2026-06
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data/workspaces/hpo_2026-06")
    ap.add_argument("--num-seeds", type=int, default=160,
                    help="Seed phenotype nodes (~ a batch of patients' symptoms)")
    ap.add_argument("--fanout", type=int, nargs="+", default=[15, 10])
    ap.add_argument("--repeats", type=int, default=20,
                    help="How many sampling ops to time for an average")
    args = ap.parse_args()

    import torch

    # --- environment report -------------------------------------------------
    _section("Environment")
    print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
    have = {}
    for name in ("torch_sparse", "torch_cluster", "pyg_lib", "torch_geometric"):
        try:
            m = __import__(name)
            have[name] = getattr(m, "__version__", "?")
            print(f"  {name}: {have[name]} OK")
        except Exception as e:  # noqa: BLE001
            have[name] = None
            print(f"  {name}: MISSING ({e})")

    # --- load the real graph ------------------------------------------------
    _section("Loading real workspace graph")
    data_dir = Path(args.data_dir)
    x_dict = torch.load(data_dir / "node_features.pt", weights_only=True)
    edge_index_dict = torch.load(data_dir / "edge_indices.pt", weights_only=True)
    with open(data_dir / "num_nodes.json") as f:
        num_nodes = json.load(f)
    node_types = list(x_dict.keys())
    print("node types:", {nt: int(x_dict[nt].size(0)) for nt in node_types})
    print("edge types:", len(edge_index_dict))
    total_edges = sum(ei.size(1) for ei in edge_index_dict.values())
    print("total directed edges:", total_edges)

    # Seeds: random phenotype nodes (the per-batch sampling starts from these).
    seed_type = "phenotype" if "phenotype" in node_types else node_types[0]
    n_seed_pool = int(x_dict[seed_type].size(0))
    g = torch.Generator().manual_seed(0)
    seeds = torch.randperm(n_seed_pool, generator=g)[: args.num_seeds]
    print(f"seeds: {len(seeds)} '{seed_type}' nodes")

    # =====================================================================
    # (C) Python baseline — mirror _build_subgraph's hot O(E) loop
    # =====================================================================
    _section("(C) Python baseline (mirrors current _build_subgraph hot loop)")
    # 1-hop neighbor set from seeds across all edge types, then the O(E) filter.
    try:
        t0 = time.perf_counter()
        for _ in range(args.repeats):
            # crude 1-hop expansion using python sets (representative of the
            # per-node dict lookups in _neighbor_sampling)
            sampled = {nt: set() for nt in node_types}
            sampled[seed_type].update(seeds.tolist())
            for etype, ei in edge_index_dict.items():
                s_t, _, d_t = etype
                src = ei[0].tolist()
                dst = ei[1].tolist()
                ss, ds = sampled[s_t], sampled[d_t]
                for a, b in zip(src, dst):  # the O(E) python loop
                    if a in ss:
                        ds.add(b)
            # (we stop at 1 hop for the baseline timing; real code does 2)
        dt = (time.perf_counter() - t0) / args.repeats
        py_nodes = sum(len(v) for v in sampled.values())
        print(f"  python 1-hop+filter: {dt*1000:.1f} ms/op, ~{py_nodes} nodes touched")
        baseline_ms = dt * 1000
    except Exception as e:  # noqa: BLE001
        print("  baseline FAILED:", e)
        baseline_ms = None

    # =====================================================================
    # (A) torch_sparse.SparseTensor.sample_adj — surgical native path
    # =====================================================================
    _section("(A) torch_sparse sample_adj (surgical C++ path)")
    sample_adj_ms = None
    if have.get("torch_sparse"):
        try:
            from torch_sparse import SparseTensor

            # Build ONE global homogeneous adjacency over all nodes. Offset each
            # node type into a single index space, and carry the ORIGINAL edge
            # id as the sparse 'value' so we can recover the per-edge-type
            # structure after sampling (this is the key feasibility question).
            offsets, off, order = {}, 0, []
            for nt in node_types:
                offsets[nt] = off
                off += int(x_dict[nt].size(0))
                order.append(nt)
            N = off

            rows, cols, eids, etype_of_edge = [], [], [], []
            eid = 0
            etype_list = list(edge_index_dict.keys())
            for ti, etype in enumerate(etype_list):
                s_t, _, d_t = etype
                ei = edge_index_dict[etype]
                r = ei[0] + offsets[s_t]
                c = ei[1] + offsets[d_t]
                rows.append(r); cols.append(c)
                n = ei.size(1)
                eids.append(torch.arange(eid, eid + n))
                etype_of_edge.append(torch.full((n,), ti, dtype=torch.long))
                eid += n
            row = torch.cat(rows); col = torch.cat(cols)
            global_eid = torch.cat(eids)
            etype_of_edge = torch.cat(etype_of_edge)

            # value = global edge id, so sample_adj can hand back which edges it kept
            adj = SparseTensor(row=row, col=col, value=global_eid.float(),
                               sparse_sizes=(N, N))

            seed_global = seeds + offsets[seed_type]
            t0 = time.perf_counter()
            kept_types = None
            for _ in range(args.repeats):
                sub_adj, n_id = adj.sample_adj(seed_global, num_neighbors=args.fanout[0], replace=False)
                kept_eid = sub_adj.storage.value().long()      # original edge ids kept
                kept_types = etype_of_edge[kept_eid]            # recover edge types!
            dt = (time.perf_counter() - t0) / args.repeats
            sample_adj_ms = dt * 1000

            uniq = torch.unique(kept_types) if kept_types is not None else torch.tensor([])
            print(f"  sample_adj 1-hop: {sample_adj_ms:.1f} ms/op, sampled {n_id.numel()} nodes")
            print(f"  EDGE-TYPE RECOVERY: kept edges map to {uniq.numel()} distinct edge types "
                  f"out of {len(etype_list)}  -> per-edge-type subgraph IS recoverable: "
                  f"{'YES' if uniq.numel() > 0 else 'NO'}")
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            print("  sample_adj FAILED:", e)
    else:
        print("  torch_sparse not available — skipping")

    # =====================================================================
    # (B) NeighborLoader — canonical hetero path
    # =====================================================================
    _section("(B) torch_geometric NeighborLoader (canonical hetero path)")
    nl_ms = None
    if have.get("torch_geometric"):
        try:
            from torch_geometric.data import HeteroData
            from torch_geometric.loader import NeighborLoader

            hd = HeteroData()
            for nt in node_types:
                hd[nt].x = x_dict[nt]
                hd[nt].num_nodes = int(x_dict[nt].size(0))
            for etype, ei in edge_index_dict.items():
                hd[etype].edge_index = ei

            num_neighbors = {etype: list(args.fanout) for etype in hd.edge_types}

            # Probe 1: can we seed from a single node type?
            loader = NeighborLoader(
                hd, num_neighbors=num_neighbors,
                batch_size=args.num_seeds,
                input_nodes=(seed_type, seeds),
                num_workers=0, shuffle=False,
            )
            t0 = time.perf_counter()
            batch = None
            it = iter(loader)
            for _ in range(args.repeats):
                batch = next(it) if True else None
                # NeighborLoader cycles; re-make iterator if exhausted
                if batch is None:
                    it = iter(loader); batch = next(it)
            nl_ms = (time.perf_counter() - t0) / args.repeats * 1000

            present_etypes = [et for et in batch.edge_types if batch[et].edge_index.numel() > 0]
            print(f"  NeighborLoader (single-type seed): {nl_ms:.1f} ms/op")
            print(f"  yielded HeteroData with x_dict keys: {list(batch.x_dict.keys())}")
            print(f"  edge types with edges in subgraph: {len(present_etypes)}/{len(hd.edge_types)} "
                  f"-> EDGE TYPES PRESERVED: {'YES' if present_etypes else 'NO'}")
            for nt in batch.node_types:
                n = batch[nt].num_nodes if 'num_nodes' in batch[nt] else batch[nt].x.size(0)
                print(f"    node '{nt}': {n} nodes in subgraph")

            # Probe 2: multi-type seeding (current code seeds pheno+disease+gene)
            try:
                multi = {nt: torch.randperm(int(x_dict[nt].size(0)))[:20] for nt in node_types}
                NeighborLoader(hd, num_neighbors=num_neighbors, batch_size=32,
                               input_nodes=multi, num_workers=0)
                print("  MULTI-TYPE SEEDING: accepted a {node_type: idx} dict -> YES")
            except Exception as e:  # noqa: BLE001
                print(f"  MULTI-TYPE SEEDING: NOT directly supported ({type(e).__name__}: {e})")
                print("    -> would need per-type loaders or seed-from-phenotype-only (design note)")
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            print("  NeighborLoader FAILED:", e)
    else:
        print("  torch_geometric not available — skipping")

    # =====================================================================
    # Verdict
    # =====================================================================
    _section("VERDICT (speed)")
    if baseline_ms:
        print(f"  python baseline (1-hop):     {baseline_ms:8.1f} ms/op")
    if sample_adj_ms:
        print(f"  torch_sparse sample_adj:     {sample_adj_ms:8.1f} ms/op", end="")
        if baseline_ms:
            print(f"   ->  {baseline_ms/sample_adj_ms:5.1f}x faster")
        else:
            print()
    if nl_ms:
        print(f"  NeighborLoader (2-hop full): {nl_ms:8.1f} ms/op (does more work: full 2-hop hetero)")
    print("\nKey questions answered above: edge-type recovery (A), edge-type "
          "preservation + multi-type seeding (B), and the speedup ratio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
