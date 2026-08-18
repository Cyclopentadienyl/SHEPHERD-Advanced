# B-0.3 — Modes B and C

**Status:** proposal, revised after review. Three decisions, then implement.

Modes B and C need almost no new machinery. Nearly everything they run on was
either built in B-0.2 or already exists in production, and the point of this
document is to name the three places where a wrong choice would silently produce
**a control that is not a control** — which is the only failure mode this work
item cannot recover from.

---

## 1. What already exists

Verified in the tree at `0711441`, not assumed:

| Needed | Where it already is |
|---|---|
| Full-graph forward | `src/inference/pipeline.py:884-921` — `_precompute_node_embeddings` runs `self.model(x_dict, edge_index_dict)` over the whole graph once at load and caches the result |
| Scoring a whole candidate matrix | `src/inference/scoring.py:cosine_score_matrix` (B-0.2). The production per-candidate scorer's own comment says it: *"scoring the whole disease universe is the same call with a taller candidate matrix"* (`pipeline.py:1471-1473`) |
| Two ranking streams, rank extraction, cohort guards | `src/evaluation/measurement.py` (B-0.2) — mode-agnostic already |
| Manifest, artifact digests, CUDA gate, seeded launcher | `scripts/measure_scorer.py`, `scripts/calibrate_mode_a.py` (B-0.2) |
| Candidate universe for Mode C | `arange(num_nodes_dict["disease"])`. No construction to design |

**What production startup proves, and what it does not.** The production
full-graph forward is already known to fit on deployment hardware. **End-to-end
B/C peak memory remains an institutional CUDA acceptance measurement** — a
measurement run may hold graph inputs, full-graph embeddings, Mode A
intermediates, candidate score matrices and ranking temporaries at once, which
startup does not. No memory subsystem is designed unless that measurement fails:
no chunking, no streaming, no CPU offload, no memory manager.

---

## 2. Decision 1 — one production model-construction boundary

**The problem, in two parts.**

*Semantics.* Mode A uses `build_legacy_mode_a_model`, which mirrors the frozen
evaluator: architecture from `checkpoint["metadata"]`, hardcoded fallbacks.
Production does something deliberately different (`pipeline.py:780-830`): metadata
derived from `graph_data["edge_index_dict"]`, in-channels from the feature
tensors, and architecture recovered by a documented precedence
(`_resolve_arch_params`) so an HGT checkpoint is never silently rebuilt as GAT.
Modes B/C/D use production semantics; they may not import the legacy loader.

*Duplication.* Production construction currently lives inside
`DiagnosisPipeline._load_model_from_checkpoint`, mixed with fingerprint warnings,
`_checkpoint_meta` for the UI, logging and instance mutation. Writing B/C against
"production semantics" without a shared boundary means copying state-dict variant
selection, metadata derivation, in-channels resolution, arch resolution, config
construction, weight loading and device placement — and the two copies would then
drift, which invalidates B and C silently rather than loudly.

**Decided: one narrow shared builder, in `src/models/gnn/shepherd_gnn.py`.**

```python
def build_shepherd_model(checkpoint, graph_data, device=None) -> ShepherdGNN
```

Its whole responsibility is production-semantic construction and weight loading.
`_resolve_arch_params` moves down beside it. What stays in `DiagnosisPipeline`:
fingerprint verification and warnings, `_checkpoint_meta`, operator logging, and
assignment to instance fields.

*Why that module rather than a new one.* `src.models` sits below `src.inference`,
`src.training` and `src.evaluation` in the layers contract, so all three may
import it — and the trainer is a fourth caller that can adopt it later without a
layer change. `shepherd_gnn.py` already owns `ShepherdGNN` and
`ShepherdGNNConfig`; a builder returning one is at home there, and a new module
holding a single function is a home nobody finds.

*Why not a `ShepherdGNN.from_checkpoint` classmethod.* It would put knowledge of
trainer checkpoint layouts — which key holds the state dict, which legacy fields
carry architecture — onto the model class. The model should not know how it was
persisted.

*Known consequence, stated rather than discovered later:*
`tests/unit/test_pipeline_arch_resolve.py` imports `_resolve_arch_params` from
`src.inference.pipeline` and its import moves with the function. The pipeline
keeps importing the model module lazily, so the torch-free constraint on
`src/inference/pipeline.py` is unaffected.

*Not in scope:* `scripts/build_index.py`, `scripts/setup_demo.py` and
`scripts/test_gnn_inference.py` also construct models. They can adopt the builder
later; migrating them here would put B-0.3 behind an unrelated sweep.

**Architecture equality stays a reporting precondition.** The shared builder makes
B and C use production semantics; it does not make them equal to A's legacy
construction. Before A→B is interpreted as encoder scope, assert that the two
constructions produce a compatible architecture and identical loaded parameter
tensors for the checkpoint in hand, and fail the comparison naming the differing
fields if they do not. A difference is itself a finding worth having before any
number is quoted.

---

## 3. Decision 2 — A and B share one traversal

**Superseded by review.** The first draft proposed re-running the same seeded
dataloader for B and taking the candidate sets from the batches, on the grounds
that `calibrate_mode_a.py` has proven the batches reproduce across processes.
**That claim was stronger than the evidence.** What the calibration compares is
what the frozen oracle writes: the aggregate MRR and the per-sample local top-20.
Two runs can agree on both, on the input digests and on the aggregate sampler
evidence while their candidate sets differ **outside the top 20** — and Mode A
persists no per-sample candidate universe that would catch it. A→B would then be
described as isolating encoder scope while both sides had independently re-rolled
a stochastic candidate construction.

**Decided:** one traversal of the dataloader, both modes consuming the same batch
object.

```
for batch in dataloader:
    candidates = batch_data["original_indices"]["disease"]   # one tensor
    A: subgraph forward       -> score against candidates
    B: full-graph embeddings  -> score against the same candidates
```

Equality is then a property of the code rather than of a comparison, and it costs
no persisted candidate artifact and no extra RNG orchestration. **The identity is
also asserted at the point of use** — both scorers are handed the same tensor
object, and the driver checks it — because "shares a variable" is a structural
claim, and structural claims decay under edits.

*If A and B ever must be separate runs*, the fallback is a deterministic digest
of the complete ordered global candidate ids per batch, compared between runs.
Input-file digests, top-20 rows and aggregate sampler statistics are **not**
sufficient evidence and must not be presented as such.

**Mode C stays outside that traversal.** It needs the patient's phenotype ids and
full-graph embeddings, and nothing else: the candidate universe is every disease,
so the subgraph sampler contributes nothing and is not run. Phenotype ids come
from the samples file, which is where the dataloader gets them too. C asserts that
its cohort and ordering match A's rather than assuming it.

---

## 4. Decision 3 — where does the offline full-graph forward come from?

Two options, and this is the reuse-versus-duplication call:

1. **Construct a `DiagnosisPipeline`** and use its cached embeddings. Exact
   production behaviour by construction — but it drags in KG object loading, the
   shortest-path table, path-finding state and explanation machinery, none of
   which B and C use, and it makes the harness depend on the pipeline's whole
   initialisation contract.
2. **Call `model(x_dict, edge_index_dict)` directly** under `no_grad`, after
   moving tensors to the device. Three lines, and the same three lines production
   runs.

**Proposed: option 2, with the equivalence asserted rather than asserted-in-prose.**
The harness computes the embeddings itself and a test checks that the result
matches what a constructed pipeline caches for the same inputs. That keeps the
harness free of the pipeline's initialisation surface while making the "same
three lines" claim a checked one.

**Mode D is the exception and must use option 1**, because Mode D's whole content
is what production actually does. That is B-0.5's problem, and it is a real one:
the public entry point truncates to `top_k` (`pipeline.py:1268`) and runs path
construction and explanation, so obtaining an untruncated pre-`top_k` rank without
duplicating candidate construction is not a detail.

---

## 5. Not in this stage

| Excluded | Why |
|---|---|
| Mode D | B-0.5. Unresolved design problem above |
| Statistical protocol, institutional run | B-0.5 |
| Fixing candidate construction or negative sampling | The measurement exists to decide whether that is needed. Changing it first destroys the baseline |
| Any scorer change | Work item B proper |

**B-0.4 is smaller than its name.** "Vectorised SP lookup" reads like a rewrite,
but `sp_mean_distances` (`src/inference/scoring.py:240`) **already takes a
sequence of targets and returns a `(C,)` tensor**. The pipeline calls it with a
one-element list, once per candidate (`pipeline.py:1399`). The vectorisation is
therefore a caller change in production, not a new primitive — and the offline
harness can pass the whole candidate list from its first line. This is recorded
here so nobody plans a rewrite for it; it stays out of B-0.3 because it changes
production behaviour and belongs with a benchmark.

---

## 6. Acceptance

Same shape as B-0.2, and the same refusal to overclaim:

- Modes B and C run over the synthetic fixture and produce both metric families,
  with the manifest recording which mode, which candidate universe and which
  model construction;
- **A and B are shown to have scored the identical candidate ids**, by sharing the
  tensor and asserting it, not by comparing summaries;
- **production and B/C construct the model through the same builder**, so a change
  to production semantics cannot leave the measurement behind;
- the architecture-equality precondition (§2) is asserted, not assumed;
- the offline-encoder equivalence (§4) is asserted against a constructed pipeline;
- cohort integrity holds: in Mode C the ground truth is always in the universe, so
  absence is impossible and fatal, exactly as in Mode A;
- **no authoritative comparison between modes is claimed from a CPU synthetic
  run.** A→B→C on institutional CUDA hardware is a separate acceptance gate, as
  Mode A's parity is.
