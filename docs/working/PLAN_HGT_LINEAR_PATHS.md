# HGT's linear paths: two PyG problems, one minimal fix each

**Status (2026-10-11): proposal, not implemented.** It splits BACKLOG item 20 into its two problems, confirms where each is reached on the formal model build path, and proposes one minimal fix for each, at the one place every entry point builds and runs HGT. Problem 1's fix needs only the owner's approval. Problem 2's fix also needs a GB10 run to choose its value. Nothing here is a workaround in an experiment script, a PyG rewrite or a second GNN backend.

**Evidence.** CPU only (4 vCPU), torch 2.10.0, torch_geometric 2.8.0.post1, pyg_lib 0.8.0. Three independent read-only investigations (two investigators and an adversarial verifier, 2026-10-11) ran probes against the formal build path; their scripts stay in the session scratch and are not evidence of record. GPU behaviour is stated as unknown wherever it is.

**Citations.** This repository at the commit that adds this file; torch_geometric as installed (`torch_geometric/nn/dense/linear.py`, `torch_geometric/nn/conv/hgt_conv.py`).

---

## 0. Where HGT is built and run

`HGTConv` is constructed in one place, `HeteroGNNLayer.__init__` (`src/models/gnn/layers.py:118-123`), and called in one place, `HeteroGNNLayer.forward` (`:182`). Every entry point reaches it through `ShepherdGNN`:

| Entry point | How it builds the model |
|---|---|
| Training | `scripts/train_model.py` `create_model_from_config` |
| Serving | `src/inference/pipeline.py` `_load_model_from_checkpoint` → `build_shepherd_model` |
| Measurement, Modes B and C | `scripts/measure_scorer.py` → `build_shepherd_model` |
| Index build | `scripts/build_index.py` |
| Smoke test, fixtures | `scripts/test_gnn_inference.py`, `tests/fixtures/synthetic_workspace.py` |

`build_shepherd_model` is not shared by training or `build_index`, so it is not the shared location; `HeteroGNNLayer` is. Legacy Mode A builds GAT only (it never sets `conv_type`) and is unaffected.

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

In `HeteroGNNLayer`, HGT branch only:

- `__init__`: record `self._hgt_node_order = tuple(self.conv.kqv_lin.lins)`, and raise unless it equals `tuple(self.conv.out_lin.lins)`. A later PyG change in how the two linears are ordered then fails at construction instead of silently.
- `forward`: build the conv's input in that order, `{nt: x_dict[nt].float() for nt in self._hgt_node_order if nt in x_dict}`. Keys the conv does not know are left out, as the loop branch already ignores them, and are put back by `:188-190` as today.

**What it changes.** Bitwise nothing on today's default path: forward outputs and every gradient were `torch.equal` before and after, in eval and train mode, for the fixture and a production-shaped graph, under training, serving and `build_index` metadata. No parameter, `state_dict` key, key order or initial weight changes, and there is no setting to record. In the ten-type case the fix turns a silently wrong result into the loop's result up to kernel rounding (forward 1.49e-7, gradients 9.5e-7, CPU).

**Rejected alternatives.** Building the layers in sorted node-type order changes fresh initialisation under the same seed and the parameter order, so an old optimizer state would resume onto the wrong parameters silently; it also still needs the reorder. Pinning the flag to `False` keeps the branch out of reach but does not make it correct, and it is Problem 2's decision.

### 1.7 A related order hazard (recorded, not triggered)

`k_rel` and `v_rel` are built with `is_sorted=True` (`hgt_conv.py:81-84`). If the edge dict passed to the forward is in a different order from `metadata[1]`, HGT gives wrong outputs silently on both paths (0.245 in a probe). The formal paths take their metadata edges from the same dict they forward with, so it is not triggered today. The implementation of Edit 1 checks whether reordering `edge_index_dict` to the conv's edge order is equally neutral, and adds it with a test if so; that is not yet verified.

### 1.8 Tests that would pin Edit 1 (`tests/unit/test_hgt_node_type_order.py`)

Each test spies on `pyg_lib.ops.grouped_matmul`, so none passes without the grouped branch actually running, and skips where it is unavailable.

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
- under `torch.compile`, or without pyg_lib, PyG takes the loop.

### 2.2 What it does on the formal path

- **The choice flips between fresh processes**, at fixture shapes and at production shapes. In four 4-thread production-shaped processes, 2, 4, 4 and 0 of the 8 instances took the loop; the timing margins were 15–25 %.
- **Training and serving never share a decision.** Training batches time at row-count magnitude 4–5 and the full-graph serving forward at 6, in separate processes.
- **Numbers.** On CPU at production width the two paths were bitwise equal; at fixture width they differed by up to 4.8e-7 (outputs) and 1.2e-7 (gradients), with rankings unchanged. On GPU this is unknown.
- **Speed.** The timing cannot see the backward pass, where the loop is much slower: about 35× on the `k_rel` operation alone at 150k rows, and a production-shaped CPU training step of 2.99 s on the loop against 1.30 s with segment_matmul. A noisy choice of the loop at process start slows that instance for the whole run. The heuristic also spends about 8 s timing on the first full-graph serving forward.

### 2.3 Proposed fix (Edit 2)

Pin the path inside `HeteroGNNLayer`, scoped to the HGT call, so it does not depend on timing or on the process-global flag outside that call:

- around the conv call: save the flag, set it to `False`, restore it in a `finally`. This also keeps Problem 1's grouped branch out of reach;
- when pyg_lib's segment_matmul is available, a forward pre-hook on `k_rel` and `v_rel` sets the flag to `True`, and a forward hook registered with `always_call=True` restores it, so an exception inside the call cannot leave it set;
- without pyg_lib, PyG already loops deterministically: warn as today and record `loop`. Model construction is never blocked (the project's policy for optional native extensions).

**No new configuration field.** `create_model_from_config`, `build_index` and Mode A build their configuration by hand, so a selectable field would be silently ignored by some builders. Instead the path **observed** is recorded: the relation and node matmul paths, whether the model is compiled (compiled means loop), and the torch_geometric and pyg_lib versions, in training's `runtime.json`, in the measurement manifest beside `torch_compile_wrapped`, and in the serving log at precompute.

**The value is chosen on the GB10.** On CPU, segment_matmul for `k_rel`/`v_rel` is the path the heuristic usually picks, is bitwise equal to the loop at production width, and avoids the loop's training cost. Before it is fixed, the GB10 shows which path today's default picks there (dump each instance's `_timing_cache` after a training step and after the serving forward) and whether segment_matmul runs and is fast there. If it fails or is slower, the hooks are dropped and the scoped `False` is the single pinned setting.

**Limitation.** The flag is process-global during each scoped call. Nothing runs HGT forwards from several threads at once today (serving precomputes once; training and measurement are single-threaded); if that changes, this is revisited.

### 2.4 Tests that would pin Edit 2 (`tests/unit/test_hgt_kernel_path.py`)

1. With `HeteroLinear._update_timing_cache` patched to raise, an HGT forward and backward still run: timing never happens.
2. `k_rel`/`v_rel` call only the pinned kernel, and `grouped_matmul` never runs.
3. The global flag is unchanged after a forward, whatever it was before (`True`, `None`, `False`), including after an exception injected inside `k_rel`.
4. The pinned model is bitwise equal to the same pin applied by hand (catches hooks that do not fire).
5. The observed path is recorded in `runtime.json` and in the measurement manifest (and is `None` for GAT).

---

## 3. Acceptance

**Numerical.**
- Edit 1, PyG's default, before against after: `torch.equal` on outputs and all gradients for a training batch, the full graph through `build_shepherd_model`, and `build_index`. Done on CPU (all equal); repeat on the GB10 under `torch.use_deterministic_algorithms(True)`, since CUDA scatter is not bitwise run to run without it.
- Edit 1, flag `True` against `False`: `assert_close`, including a ten-type case where the spy shows the grouped branch ran.
- Edit 2: the tests of §2.4; on the GB10, the subject's pinned result against today's default (max abs difference and top-k agreement), and two same-seed pinned runs bitwise equal under deterministic algorithms.

**Performance.** CPU, done: production-shaped training step 1.30 s (segment_matmul for the relation linears, loop for the node linears) against 2.99 s (all loop). GB10: per-step wall time over 30 batches for today's heuristic, the proposed pin and the all-loop pin, plus the full-graph serving forward; proposed acceptance, the pin within noise of the heuristic's median.

**Then.** Edit 2 is an inference-path change, so the B04 readings are re-taken or assessed as `scorer-measurement/PLAN_B04_PRODUCTIONISATION.md:840` (§7.1.1, item 5) requires. Edit 1 changes no number on today's graphs and needs no re-take.

---

## 4. What the owner decides

1. Approve Edit 1 (correctness, bitwise neutral today, no GPU dependency). It can land on its own.
2. Approve Edit 2's approach and the GB10 run that chooses its value; Edit 2 lands after that run.
3. Whether to report the zip at `linear.py:447` upstream.
