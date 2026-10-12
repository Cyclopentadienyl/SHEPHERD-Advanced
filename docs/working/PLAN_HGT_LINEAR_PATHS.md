# HGT's linear paths: two PyG problems, one minimal fix each

**Status (2026-10-11): proposal. Edit 1 approved by the owner on 2026-10-12 to land on its own and implemented the same day (§1.6–§1.8) as `8eae614` and code-reviewed with no P1 or P2 on 2026-10-12 (`1d0b5a6..06fd0f1`; CPU code review only, not a GB10 performance or CUDA numeric acceptance); Edit 2 waits for its GB10 comparison (§2.3, §4).** It splits BACKLOG item 20 into its two problems, confirms where each is reached on the formal model build path, and proposes one minimal fix for each, at the one place every entry point builds and runs HGT. Problem 1's fix needs only the owner's approval. Problem 2's fix also needs a GB10 run to choose its value. Nothing here is a workaround in an experiment script, a PyG rewrite or a second GNN backend.

**Reviewed (2026-10-12, `290c03a..1d0b5a6`): one P2, applied in this revision.** The kernel a run used must not be inferred from its compile setting ("compiled means loop"); Edit 2 now records the requested policy, the compile setting and the observed kernel separately, and is reduced to a bounded GB10 comparison first and the simplest sufficient policy after it. The reviewer accepted Edit 1's direction. The fix (`418b37e`) was code-reviewed with no P1 or P2 on 2026-10-12 (`1d0b5a6..06fd0f1`): the kernel is recorded as observed, not inferred, and the P2 can be closed.

**Evidence.** CPU only (4 vCPU), torch 2.10.0, torch_geometric 2.8.0.post1, pyg_lib 0.8.0. Three independent read-only investigations (two investigators and an adversarial verifier, 2026-10-11) ran probes against the formal build path; their scripts stay in the session scratch and are not evidence of record. GPU behaviour is stated as unknown wherever it is.

**Citations.** This repository at the commit that adds this file; torch_geometric as installed (`torch_geometric/nn/dense/linear.py`, `torch_geometric/nn/conv/hgt_conv.py`).

---

## 0. Where HGT is built and run

`HGTConv` is constructed in one place, `HeteroGNNLayer.__init__` (`src/models/gnn/layers.py:118-123`), and called in one place, `HeteroGNNLayer.forward` (`:182`). Every entry point reaches it through `ShepherdGNN`:

| Entry point | How it builds the model |
|---|---|
| Training | `scripts/train_model.py` `create_model_from_config` |
| Serving | `src/inference/pipeline.py` `_load_model_from_checkpoint` → `build_shepherd_model` |
| Measurement (Mode C) | `scripts/measure_scorer.py` → `build_shepherd_model` |
| Index build | `scripts/build_index.py` |
| Smoke test, fixtures | `scripts/test_gnn_inference.py`, `tests/fixtures/synthetic_workspace.py` |

`build_shepherd_model` is not shared by training or `build_index`, so it is not the shared location; `HeteroGNNLayer` is. Measurement's legacy Mode A, which built GAT only, was retired in the Mode A/B retirement's R2.

---

## 1. Problem 1 — the grouped path assigns outputs to the wrong node types (a correctness defect)

### 1.1 The defect

`HeteroDictLinear` (HGT's `kqv_lin` and `out_lin`) has a grouped branch through `pyg_lib.ops.grouped_matmul`. It builds its inputs in the order of its own layers and stores the outputs under `x_dict`'s keys (`linear.py:439-449`; the zip at `:447` runs over `x_dict.keys()`). When the two orders differ, each node type receives another type's projection. Upstream PyG's master branch still has the same zip.

### 1.2 When the grouped branch runs

- `torch_geometric.backend.use_segment_matmul` is `True`; or it is `None` (PyG's default) and `len(x_dict) >= 10` (`linear.py:433-438`);
- and pyg_lib's grouped matmul is available, and the code is not under `torch.compile` or TorchScript;
- `False` always takes the per-type loop.

Nothing in this repository sets the flag. The node-type count is the graph's: the loader keeps every node type, including ones with no rows in a subgraph.

### 1.3 Why the orders differ on the formal path

The layers' order is `metadata[0]`, which each builder takes from a different place: training from `num_nodes_dict`'s keys, serving and Modes B/C from `x_dict`'s keys, `build_index` from `sorted(x_dict)`. The input order comes from the caller:

- the training loader puts its seed types first (phenotype, disease, then gene) and the rest after;
- from the second layer on, `HeteroGNNLayer.forward` appends the node types that received no messages at the end (`layers.py:188-190`), so the next layer sees another order whenever some type is not a destination.

| Graph and path | Layers' order | Input to `kqv_lin` |
|---|---|---|
| Fixture, training, layer 0 | phenotype, gene, disease | phenotype, disease, gene — differs |
| Fixture, layer 1, training and serving (no `rev_*` edges) | phenotype, gene, disease | gene, disease, phenotype — differs |
| Production node-type set, training, layer 0 | disease, phenotype, gene | phenotype, disease, gene — differs |
| Production set, training layers ≥ 1; serving; Modes B and C | disease, phenotype, gene | same (`rev_*` edges make every type a destination) |
| Production set, `build_index` | disease, gene, phenotype | disease, phenotype, gene — differs |

The production order is inferred by running the builder's steps in the same sequence; it is not recorded anywhere.

### 1.4 Whether production reaches it today

**No.** The production subject has three node types (`scorer-measurement/PLAN_B04_PRODUCTIONISATION.md:826-827`), and the shipped build script adds three: diseases, phenotypes and genes (`scripts/build_knowledge_graph.py:268-305`, orthologs and literature off). The builder can name at most eight with today's code (disease, phenotype, pathway and mouse phenotype through `add_ontology`, `src/kg/builder.py:152-157`; gene, mouse gene and zebrafish gene, `:498-502`; publication, `:577`), so ten or more needs new build code, not a configuration change. With PyG's default, the grouped branch ran 0 times per forward at 3 and 9 node types and 4 times at 10 and 11 (formal training path, CPU).

### 1.5 What happens when it is reached

It is **silent more often than it raises**. Each wrong output is a correct projection of another type, stored under the wrong key. It raises only when an edge index runs past the end of the concatenated tensor (`IndexError`), or when `out_lin` permutes differently from `kqv_lin` so that row counts no longer match at the skip connection (`RuntimeError`).

| Case (grouped branch, per-type loop held for `HeteroLinear`) | Result |
|---|---|
| One layer, production metadata with `rev_*` edges, input in loader order, rows 7 / 5 / 3 | no error, wrong (forward 0.27, gradient 2.8) |
| Same, rows 40 / 30 / 20 | no error, wrong (forward 0.58) |
| Same, rows 5 / 7 / 3 | `IndexError` |
| Same, input already in the layers' order | exact |
| Ten node types, PyG's default flag, formal training path, eval mode | no error, wrong (forward 0.64) |

### 1.6 Proposed fix (Edit 1)

**Implemented (2026-10-12)** as proposed below, together with §1.7's edge order, which was checked and found equally neutral.

In `HeteroGNNLayer`, HGT branch only:

- `__init__`: record `self._hgt_node_order = tuple(self.conv.kqv_lin.lins)`, and raise unless it equals `tuple(self.conv.out_lin.lins)`. A later PyG change in how the two linears are ordered then fails at construction instead of silently.
- `forward`: build the conv's input in that order, `{nt: x_dict[nt].float() for nt in self._hgt_node_order if nt in x_dict}`. Keys the conv does not know are left out, as the loop branch already ignores them, and are put back by `:188-190` as today.

**What it changes.** Bitwise nothing on today's default path: forward outputs and every gradient were `torch.equal` before and after, in eval and train mode, for the fixture and a production-shaped graph, under training, serving and `build_index` metadata. No parameter, `state_dict` key, key order or initial weight changes, and there is no setting to record. In the ten-type case the fix turns a silently wrong result into the loop's result up to kernel rounding (forward 1.49e-7, gradients 9.5e-7, CPU).

**Rejected alternatives.** Building the layers in sorted node-type order changes fresh initialisation under the same seed and the parameter order, so an old optimizer state would resume onto the wrong parameters silently; it also still needs the reorder. Pinning the flag to `False` keeps the branch out of reach but does not make it correct, and it is Problem 2's decision.

### 1.7 A related order hazard (recorded, not triggered)

`k_rel` and `v_rel` are built with `is_sorted=True` (`hgt_conv.py:81-84`). If the edge dict passed to the forward is in a different order from `metadata[1]`, HGT gives wrong outputs silently on both paths (0.245 in a probe). The formal paths take their metadata edges from the same dict they forward with, so it is not triggered today. The implementation of Edit 1 checks whether reordering `edge_index_dict` to the conv's edge order is equally neutral, and adds it with a test if so; that is not yet verified.

**Checked and added with Edit 1 (2026-10-12).** Both kernel paths of `HeteroLinear` cut their input by `type_ptr`, so an edge order other than `metadata[1]`'s is wrong on either. The training loader builds its edge dict by iterating the graph's own `edge_index_dict`, keeping empty types (`src/kg/data_loader.py`, `_build_subgraph_edges_vectorized` and `_build_subgraph_edges_legacy`), and serving and `build_index` forward the graph's own dict, so on every formal path the reorder is the identity: forward outputs and every gradient were `torch.equal` before and after Edit 1 as implemented, in eval and train mode, for a loader batch and the full fixture graph (CPU). `HeteroGNNLayer` passes the conv its edges in the conv's edge-type order (`HGTConv.edge_types_map`); an edge type the conv does not know is kept, last, so the conv still refuses it with the `KeyError` it raised before.

### 1.8 Tests that would pin Edit 1 (`tests/unit/test_hgt_node_type_order.py`)

**As implemented (2026-10-12):** the five below, plus §1.7's edge order (the reverse of the conv's order gives `torch.equal` outputs and gradients under `False` and `True`) and an unknown edge type still refused. Removing the node reorder fails the grouped-branch, ten-type and formal-path tests; removing the edge reorder fails the edge-order test; dropping unknown edge types fails the refusal test.

Every test spies on `pyg_lib.ops.grouped_matmul`. The cases meant to exercise the grouped branch (flag `True`, or ten node types under PyG's default) require the spy to fire, so none of them passes without the branch running; the cases under `False`, or under PyG's default with fewer than ten node types, require it not to fire. All skip where grouped matmul is unavailable.

1. The conv's two linears and `metadata[0]` share one order (the construction check).
2. With production metadata and `rev_*` edges, the layer's output does not depend on the input dict's order, under PyG's default and under `False`.
3. Under `True`, output and gradients equal the `False` result (`assert_close`) for the row patterns above, including the ones that are silent and the one that raises today.
4. Ten node types under PyG's default: the spy fires, and the result equals the `False` result.
5. The formal path: a training-built HGT checkpoint rebuilt with `build_shepherd_model`, a real loader batch, and the same comparison through `build_index`.

---

## 2. Problem 2 — `HeteroLinear` picks its kernel by timing (a computation-path and reproducibility problem)

### 2.1 The mechanism

With the flag at `None`, `HeteroLinear` (HGT's `k_rel` and `v_rel`) times `pyg_lib.segment_matmul` against a per-type loop on its first forward for each instance and each order of magnitude of the row count, and keeps the faster (`linear.py:258-285`, `:301-313`):

- only the no-grad forward is timed, segment_matmul first and cold, three runs (one when `PYTEST_CURRENT_TEST` is set), no margin;
- the choice lives in a plain attribute (`_timing_cache`), per instance and per process; it is not in the `state_dict`, the checkpoint, `runtime.json` or the measurement manifest;
- without pyg_lib PyG takes the loop. Under compilation the choice depends on whether that forward is being captured by the compiler (`is_compiling()`, `linear.py:302`, `:314-315`), not on whether the model carries a compile wrapper: a wrapper whose graph falls back to eager, or an eager sub-region inside a compiled model, can still run segment_matmul. This project's `--compile` sets `suppress_errors` (`scripts/train_model.py:784`), which allows exactly that fallback, and `observe_torch_compile_wrapper` already says that "wrapped" does not mean a compiled graph ran (`src/evaluation/measurement.py:290-300`).

### 2.2 What it does on the formal path

- **The choice flips between fresh processes**, at fixture shapes and at production shapes. In four 4-thread production-shaped processes, 2, 4, 4 and 0 of the 8 instances took the loop; the timing margins were 15–25 %.
- **Training and serving never share a decision.** Training batches time at row-count magnitude 4–5 and the full-graph serving forward at 6, in separate processes.
- **Numbers.** On CPU at production width the two paths were bitwise equal; at fixture width they differed by up to 4.8e-7 (outputs) and 1.2e-7 (gradients), with rankings unchanged. On GPU this is unknown.
- **Speed.** The timing cannot see the backward pass, where the loop is much slower: about 35× on the `k_rel` operation alone at 150k rows, and a production-shaped CPU training step of 2.99 s on the loop against 1.30 s with segment_matmul. A noisy choice of the loop at process start slows that instance for the whole run. The heuristic also spends about 8 s timing on the first full-graph serving forward.

### 2.3 Proposed fix (Edit 2), in two steps

**Step 1 — a bounded GB10 comparison before any policy is fixed.** On the designated subject, compare today's heuristic, all loop (`use_segment_matmul=False`), all segment_matmul (`True`, with Edit 1 in place) and, only if those two differ materially, relation linears on segment_matmul with node linears on the loop. For each: per-step training time over a fixed number of batches, peak memory, the kernel each relation linear actually ran (spied), output and gradient differences against the loop, and ranking agreement on the full-graph serving forward. CPU numbers do not decide the GB10 default.

**Step 2 — the simplest sufficient shared policy.** If one fixed setting meets correctness and performance, it is applied at the one place HGT runs, `HeteroGNNLayer`, scoped to the conv call by a single save-and-restore in a `finally`, with no per-module hooks. Only if the relation and node linears genuinely need different paths is that split designed, inside the same layer. No PyG fork and no second backend.

**The requested policy is explicit and recorded.** It is a model-configuration field (in `ShepherdGNNConfig`), so it is saved with the checkpoint's `model_config` and read back by `build_shepherd_model`. The builders that copy the configuration by hand (`create_model_from_config` in `scripts/train_model.py`, and `scripts/build_index.py`) are fixed to carry it: that wiring gap is a defect to repair, not a reason to do without the field. Mode A was retired (R2) and is not accommodated.

**Three facts, recorded separately and never inferred from one another:** the requested policy (the configuration field); the compile setting and whether the model carries a compile wrapper; and the kernel each relation linear was observed to run in that execution. A compile wrapper does not mean the loop ran (§2.1); a control experiment in the review (torch 2.10.0, PyG 2.8.0.post1, a compiler-disabled `HeteroLinear.forward` under a `torch.compile` wrapper) observed segment_matmul. Where nothing was observed the kernel is recorded as unknown; where one execution ran different paths it is recorded per instance, not as a single value. They are written to training's `runtime.json`, the measurement manifest and the serving log at precompute, with the torch_geometric and pyg_lib versions. The observation is a spy on the two kernels, not a compiler-tracing framework.

Without pyg_lib PyG loops deterministically: warn as today and record the loop observed. Model construction is never blocked (the project's policy for optional native extensions).

**Limitation.** A scoped flag is process-global for the duration of the call. Nothing runs HGT forwards from several threads at once today (serving precomputes once; training and measurement are single-threaded); this is not a defect seen, and it is revisited if that changes.

### 2.4 Tests that would pin Edit 2 (`tests/unit/test_hgt_kernel_path.py`)

1. With `HeteroLinear._update_timing_cache` patched to raise, an HGT forward and backward still run: timing never happens.
2. Under the configured policy the relation linears call only the kernel it names, and `grouped_matmul` runs only if the policy says so.
3. The global flag is unchanged after a forward, whatever it was before (`True`, `None`, `False`), including after an exception raised inside the conv.
4. The policy is saved in `model_config`, read back by `build_shepherd_model` and carried by every builder; an unknown value is refused.
5. The recording classifies three cases correctly, from the kernels observed: eager; a normal compile; and an eager sub-region or fallback under a compile wrapper. With no observation it records unknown.

---

## 3. Acceptance

**Numerical.**
- Edit 1, PyG's default, before against after: `torch.equal` on outputs and all gradients for a training batch, the full graph through `build_shepherd_model`, and `build_index`. Done on CPU (all equal); repeat on the GB10 under `torch.use_deterministic_algorithms(True)`, since CUDA scatter is not bitwise run to run without it.
- Edit 1, flag `True` against `False`: `assert_close`, including a ten-type case where the spy shows the grouped branch ran.
- Edit 2: the tests of §2.4; on the GB10, Step 1's output, gradient and ranking comparisons, and two same-seed runs under the chosen policy bitwise equal under deterministic algorithms.

**Performance.** CPU, done: production-shaped training step 1.30 s (segment_matmul for the relation linears, loop for the node linears) against 2.99 s (all loop). GB10: Step 1's time and peak-memory measurements; the policy chosen from them is the simplest one within noise of the best measured.

**Then.** Edit 2 is an inference-path change, so the B04 readings are re-taken or assessed as `scorer-measurement/PLAN_B04_PRODUCTIONISATION.md:840` (§7.1.1, item 5) requires. Edit 1 changes no number on today's graphs and needs no re-take.

---

## 4. What the owner decides

1. Approve Edit 1 (correctness, bitwise neutral today, no GPU dependency). It can land on its own. **Decided by the owner (2026-10-12): Edit 1 first, on its own; implemented the same day.** Edit 2 waits for item 2's comparison.
2. Approve the bounded GB10 comparison of Edit 2's Step 1. The policy is chosen from its results, and Edit 2 lands after that.
3. Whether to report the zip at `linear.py:447` upstream.
