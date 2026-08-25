# Plan — configurability as a requirement, and the provenance gap that undermines it

**Status: draft for review. Nothing here is implemented.**

## 0. Why this document exists

An institutional statement of intent has been made explicit, and it changes how
two existing pieces of code should be judged:

> This system is **two things at once**. It is a clinical inference model that
> will go into service, and it is a research framework in which investigators
> iterate configurations, compare architectures, and study what shortens training
> without costing accuracy. **The data the framework produces is part of the
> deliverable, not a by-product.** Parameters that help one architecture may be
> neutral or harmful for another, so the pipeline must not be nailed to a single
> "optimal" path. Researchers and the clinical team configure, measure, and decide
> what is fit to deploy.

This is a **design position**, not a feature request, and it is recorded here so
that later work can be checked against it instead of re-deriving it.

**What this document is not.** It proposes no configuration framework, no
experiment-tracking subsystem, no run registry or provenance database, and no
schema hierarchy. Both proposals below are small and reuse machinery that already
exists. If either grows past what §4 describes, that growth is the finding.

---

## 1. The position, stated so it can be checked against

**P1. A knob is removed only when nothing could want it, and removing one is a
reviewable change rather than a tidy-up.** The reverse — a knob nobody turns —
is also a cost, so this is not "add options"; it is "do not close a door that
research needs open".

**P2. Where a setting affects results, the artifact records what was *observed*,
not what was *requested*.** This is already the project's stated rule, in
`BACKLOG.md` D2: *"the **observed** `torch.compile` execution state — an
execution fact, not a requested config value"*.

**P3. Refusing an input is a stronger act than recording it, and needs a stronger
reason.** A refusal protects one value; an honest record protects every value.
Refusal is right where a state would make an artifact lie and cannot be
described. It is wrong where the state can simply be written down.

**P4. Two runs are comparable only if the artifacts can prove they differed in
what the experimenter thinks they differed in.** A configuration comparison in
which the data could silently have changed is not a configuration comparison.

---

## 2. What is already right, established rather than assumed

Checked against the tree, not recalled:

| Surface | State |
|---|---|
| `TrainerConfig` | 24 fields |
| `src/config/training_fields.py` | 35 fields exposed to Runtime Settings |
| Architectures | `hgt`, `gat`, `sage` (`SUPPORTED_CONV_TYPES`), resolved by `resolve_arch_params` with a documented precedence so an HGT checkpoint is never rebuilt as the GAT default |
| AMP, `torch.compile`, `num_neighbors`, `num_negative_samples`, sampling strategy | all configurable |
| Checkpoint `config` key | `Trainer._serialize_config()` stores the **whole** `TrainerConfig` plus the entire model-architecture dataclass, so new architecture fields persist automatically |

**No work in the current stage removed a configuration option.** P1 is not
currently violated at the training layer. Two things below are, in narrower ways.

---

## 3. The two findings

### 3.1 The measurement harness refuses a state it could have recorded

`src/evaluation/measurement.py::assert_no_autocast` raises if either traversal is
entered inside an `autocast` block. It is called from `run_modes_ab` and
`run_mode_c`.

**The reason it was added is real.** `scripts/measure_scorer.py::build_manifest`
writes `amp_enabled=False` and `amp_dtype=None` as **literals**. Under autocast
every score shifts while the manifest goes on claiming fp32, so the artifact
would describe a run that did not happen.

**But the fix chosen was the stronger of the two available acts, and P3 says it
needed the stronger reason.** It does not have one. The state is trivially
describable: `torch.is_autocast_enabled(device_type)` and
`torch.get_autocast_dtype(device_type)` both answer on any host, including a
CPU-only one — verified by execution.

**The cost is a research question the harness can no longer answer.** *"What is
Mode A's MRR under the AMP setting the deployment actually uses?"* is legitimate,
and Modes A/B/C now refuse it. Note the asymmetry with `torch_compile_wrapped`,
added in the same work item: compile state is **observed and recorded**; AMP state
is **forbidden**. D2's own wording asked for the first treatment. The
inconsistency is the author's.

### 3.2 A checkpoint cannot say which data it was trained on

`scripts/train_model.py:658` sets `trainer.data_fingerprint =
compute_fingerprint(graph_data)`, and `src/training/callbacks.py:315-316` embeds
it in every saved checkpoint. So a fingerprint mechanism exists and is wired.

**What it records is the graph's shape, not its content.**
`src/utils/fingerprint.py::compute_fingerprint` returns `node_types`,
`node_counts`, `feature_dims` and `edge_types`.

**The project already documents this limit — on the measurement side.**
`scripts/measure_scorer.py::artifact_digests`:

> Paths are recorded too, but a path is not an identity — `checkpoints/best.pt`
> names a different file after every improvement — and the structural fingerprint
> is not one either, since two checkpoints trained on the same graph share it.

So the two sides of the project disagree about their own rule:

| Artifact | Raw content digests |
|---|---|
| Measurement manifest (`MeasurementManifest.artifact_digests`) | **yes** — checkpoint, samples, node_features, edge_indices, num_nodes |
| Training checkpoint | **no** — structural fingerprint only |

**The consequence lands exactly on P4.** Two investigators training on *different
sample files* produce checkpoints with **identical** `data_fingerprint` whenever
the disease and phenotype counts match. Comparing those two runs looks like a
configuration comparison and may be a dataset comparison, **and no artifact either
of them holds can detect the difference.** `verify_fingerprint` cannot: it
compares structure to structure.

This also touches work already queued. Item 6 asks which checkpoint is
authoritative and item 10 must produce M1–M3 checkpoint evidence; both are harder
to answer with checkpoints that cannot name their training data.

---

## 4. Proposals

Ordered by value. Each states what it does **not** do.

### 4.1 Proposal A — record the training checkpoint's input digests

**Change.**

1. Move `file_sha256` and `artifact_digests` from `scripts/measure_scorer.py`
   into `src/utils/fingerprint.py`, which already owns fingerprinting and sits in
   the bottom layer where every caller can reach it. Re-export from
   `measure_scorer` so no call site changes behaviour.
2. Extend `compute_fingerprint`'s **result** with a `digests` sub-dict, or attach
   digests alongside it at the `train_model.py:658` wiring point — §5 asks review
   which.
3. `verify_fingerprint` reports a digest mismatch as one more warning string in
   the list it already returns.

**Why this is reuse, not a new mechanism.** `file_sha256` already has three
callers outside its own module (`measure_scorer:88-92`,
`calibrate_mode_a:377,384`, `benchmark_sp_lookup:284`) while living in a script.
Lifting it gives one implementation to callers that already exist — the move is
overdue independently of this plan.

**Cost.** One SHA-256 pass over the graph artifacts and the sample files, once per
training run, at startup. Bounded and not on any hot path.

**Does not do.** No registry, no database, no comparison tooling, no
`verify_fingerprint` change beyond one warning string, no fatal error on mismatch —
it stays a warning, because whether a mismatch is disqualifying is a judgement
this code cannot make.

**Open risk to name.** A digest binds a checkpoint to exact bytes. If a workspace
is legitimately regenerated, every prior checkpoint reports a mismatch. That is
the correct signal, but it must arrive as information rather than as an
obstruction — hence warning, not error.

### 4.2 Proposal B — observe the AMP state instead of forbidding it

**Change.**

1. `build_manifest` **observes** `amp_enabled` and `amp_dtype` rather than
   writing literals.
2. `assert_no_autocast` becomes a **consistency** check: the traversal refuses if
   the state it observes differs from what its manifest recorded.
3. The `MeasurementManifest` docstrings drop "no traversal here opens an autocast
   context, therefore False" and say what the fields now mean.

**Why this is strictly stronger than the refusal it replaces.** The refusal
protects one value. The consistency check protects every value — including the
case the refusal never covered, where a manifest is built in one context and the
traversal runs in another.

**Does not do.** It adds no autocast **inside** the harness. The traversals still
open no autocast context of their own; the change is only that a caller's context
is described instead of rejected. Nor does it touch the differential calibration,
which already records the trainer's resolved AMP state, nor define any acceptance
criterion for AMP-on runs — that ownership was settled in BACKLOG §3.1.3 and this
plan does not reopen it.

**Boundary check.** Proposal B must not become "make everything observable".
Manifest fields sourced from `loader_config` are already the object the loader
uses, and inventing a probe for each would be the framework this plan refuses to
build.

---

## 5. Questions for review

1. **Proposal A's shape.** Digests inside `compute_fingerprint`'s result, or
   attached beside it? Inside keeps one call site; beside keeps a structural
   fingerprint structural. Author leans **beside**, for the second reason.
2. **Which files.** `artifact_digests` takes one `split`. Training consumes
   **train and val**. Does the training record digest both, or all
   `*_samples.json` present?
3. **`save_weights_only`.** That path writes `{"state_dict": ...}` only
   (`callbacks.py:296-297`). Do digests belong there too, or is a
   weights-only checkpoint deliberately not provenance-bearing?
4. **P1 against a decision already taken.** `DIAGNOSIS_SUBGRAPH_HOPS`
   (`src/kg/data_loader.py`) was made a constant with the note *"nothing varies
   it, and adding a knob nobody turns is a wider change than removing a
   duplicated literal"*. Under the position in §1, **hop count is exactly the kind
   of thing a researcher would vary.** Was that call wrong? The author does not
   think it should be reopened *in this plan* — it is a `DataLoaderConfig` change
   touching the training path and deserves its own item — but it should not stand
   unexamined either.
5. **Ordering.** Author proposes **A before B**: A closes a gap that makes
   comparisons unreliable and improves item 6's evidence; B restores a capability
   nothing is currently blocked on.

---

## 6. Acceptance

Both proposals are behaviour-preserving for every existing caller, so the bar is:

- every existing test passes unchanged;
- **A**: a checkpoint trained on one sample file and one trained on a *different*
  file of the same shape are distinguishable from their artifacts alone — the case
  §3.2 says is currently invisible, tested directly;
- **B**: a manifest produced inside an autocast block records the observed dtype,
  and a manifest whose recorded state disagrees with the traversal's is refused —
  both mutation-checked, including against the current refuse-absence code;
- `make lint-imports` reports 4 contracts kept.

**Neither proposal needs CUDA**, and neither may be used to argue for a CPU
substitute for the institutional runs in items 7a and 5a.
