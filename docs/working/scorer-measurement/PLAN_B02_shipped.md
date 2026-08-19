> **History, not authority.** This is the plan the shipped B-0.2 code was built
> from, kept for the reasoning behind it. Where it disagrees with the code, the
> code is what shipped — it went through several review rounds after this was
> written. The stage map is in [`README.md`](README.md).

# Review submission — B-0.2 work plan, rev 3

**B-0.2 — evaluator parity harness and rank-based metrics**

Self-contained; assumes no retained context. Rev 3 applies two blocking findings and eight major
corrections, all concerning **what the frozen oracle can actually prove**. Every factual claim was
verified against the source before acceptance.

---

## 1. Why this exists

The offline evaluator does not measure the deployed scorer. Verified from source:

| | Offline evaluator | Deployed pipeline |
|---|---|---|
| **Candidates** | per-batch **seeded and 2-hop-expanded subgraph** (`scripts/evaluate_model.py:257,260` → `src/kg/data_loader.py:836`); size varies per batch | per-patient **path-reachable diseases**, produced by path construction (`src/inference/pipeline.py:987-996, 1211`) |
| **Score** | pure cosine (`:290-292`); no `eta`, no SP anywhere in the file | `eta*emb + (1-eta)*sp` |
| **Output** | truncated to `max(top_k_values)` (`:299`); mean rank not computable | truncated to `top_k` (`pipeline.py:1271`) |

**Correction to rev 1, which claimed the pipeline "scores all loaded diseases". It does not.**
`_find_all_paths` runs first; when it returns nothing the pipeline returns an **empty result**
(`pipeline.py:989-996`), and `_score_and_rank_candidates` iterates only `all_paths.items()`. A
disease with no reasoning path is never scored. Rev 1 described the *target* design as though it were
current behaviour — the same error class as several earlier corrections.

**Mode C is the all-disease universe.** Nothing in the system does that today.

And the evaluator's candidate set is built from the answers (`src/kg/data_loader.py:916-926`):

```python
seed_nodes["disease"] = cat([disease_ids, negative_disease_ids, candidate_ids]).unique()
```

then 2-hop expanded. The ground truth is always a seed; the distractors are the batch's other ground
truths, its sampled negatives, and their neighbourhoods. **Correct for training** — the positive must
be in the subgraph to compute a loss — **not admissible as an evaluation candidate set.**

Reference method, for contrast (npj Digital Medicine 8:380, 2025, Methods, "Negative sampling"):
*"we randomly sample 1000 diseases from all diseases in the KG to serve as negative examples for each
batch. Then, we calculate a patient's similarity to all disease nodes in the KG at inference time."*

**A second rev-1 overstatement, withdrawn.** `(cos+1)/2` is strictly monotonic, so it changes no
ranking, no MRR and no Hits@K. It is a display-semantics mismatch, material only in Mode D where
`eta` mixes on a particular scale. Listing it beside the substantive differences overstated it.

---

## 2. Scope boundary

**In scope:** the harness, Mode A, both metric families, batched primitives, a manifest, synthetic
tests.

| Excluded | Why |
|---|---|
| Fixing negative sampling or candidate construction | **Mode A preserves the legacy behaviour as a control.** A sampler correction is separate work, *after* the baseline exists |
| Modes B, C, D — including **how** Mode D obtains ranks | B-0.3 and B-0.5 (§7) |
| Vectorised SP lookup | B-0.4 |
| Statistical protocol, institutional run | B-0.5 |
| Any scorer change | Work item B proper |

**B-0.2 does not correct the legacy candidate universe or sampling policy.** It adds a calibrated
harness and authoritative rank metrics alongside the preserved legacy control.

---

## 3. Module layout, and what happens to the legacy script

New work lands in one place:

```
src/evaluation/
    __init__.py
    measurement.py                # modes, cohort, per-patient scoring, ranking
scripts/measure_scorer.py         # thin CLI
```

**One module, not four.** A `modes` / `harness` / `metrics` / `manifest` split was considered and
rejected: B-0.2 needs Mode A only, and that structure would be predicted rather than earned.

### 3.1 The legacy script is not moved

**`scripts/evaluate_model.py` stays exactly where it is, behaviourally frozen, and is deleted after
institutional calibration. Its only edit is a docstring status and deletion notice** — it cannot be
both literally unmodified and carry that notice, and git history preserves the original artifact. No
second archive and no checksum mechanism.

Rev 1 proposed moving it to `scripts/legacy/`. Two reasons that is wrong:

1. `evaluate_model.py:61` computes `PROJECT_ROOT = Path(__file__).parent.parent` and inserts it on
   `sys.path`. Moving the file makes `PROJECT_ROOT` point at `scripts/` and breaks its imports.
2. **The one-line fix for that is itself the problem.** A frozen oracle's entire value is being the
   unmodified artefact that produces the reference numbers. Editing it — even trivially — makes it no
   longer the thing being compared against. *Frozen* and *moved* are in tension, because moving
   requires editing.

The second reason is the decisive one, and it is stronger than "moving creates work with no
measurement value": the work is one line; the loss of an unmodified reference is not recoverable.

**The institutional goal is still met.** The requirement was pipeline cleanliness — one capability
not spanning two modules, especially when one is destined for retirement. After B-0.2 the pipeline
has exactly one measurement module. The old script is not part of the pipeline; it is a comparison
artefact with a **deletion condition: institutional Mode A calibration succeeds.** It is deleted, not
archived — a `legacy/` directory whose only occupant is scheduled for deletion is worse than no
directory.

That docstring states: behaviourally frozen, non-authoritative, and when it goes.

### 3.2 Layering

`src.evaluation` is added to `.import-linter.ini` above `src.inference`:

```ini
layers =
    src.api : src.webui
    src.evaluation
    src.inference
    src.training
```

Evaluation may import lower layers; production inference may not import evaluation. **This is an
architectural check, not evidence that Mode A is calibrated.**

**Scope: the new package must introduce no violation.** Any unrelated pre-existing violation the
contract surfaces is not B-0.2's to fix.

---

## 4. Primitives — extend, do not unify

Added to `src/inference/scoring.py`:

```
masked_mean_pool(embeddings: (B,N,H), mask: (B,N)) -> (B,H)
cosine_score_matrix(patient_matrix: (B,H), candidate_matrix: (D,H)) -> (B,D)
```

`cosine_scores(patient_vector: (H,), candidate_matrix: (C,H)) -> (C,)` delegates to
`cosine_score_matrix` via unsqueeze/squeeze, with `B=1` equivalence tests.

**Pooling does *not* delegate.** The served path has an unpadded `(N,H)` tensor and naturally uses
`mean(dim=0)`; the evaluator has padded `(B,N,H)` and needs masked sum/count. Both keep their natural
implementation, bound by an **all-true-mask equivalence test**. Manufacturing a mask and a batch
dimension in the served path to force one implementation costs more clarity than sharing buys.

**`masked_mean_pool` empty-mask behaviour is a parity requirement, not a design choice.** An all-false
mask produces a **zero vector**, because the trainer clamps the denominator to one
(`src/training/trainer.py:744-751`). Preserve that unless upstream validation proves the case
impossible.

**Dtype follows the trainer's observed behaviour, not an assumed preservation rule.** The trainer
casts the mask with `.float()` (`trainer.py:746`), so under AMP or non-float32 embeddings the output
dtype is whatever promotion and autocast produce. The requirement is **parity with that**, plus **no
silent device transfer**. **Do not add casts to satisfy a preservation rule nobody stated.**

Tests: variable valid counts; all true; padding ignored; all false; shape mismatch; float32 and any
supported AMP dtype; device unchanged.

**PyTorch built-ins only** — `F.normalize`, `torch.mm`, tensor reductions. No custom cosine kernel,
no new dependency.

### 4.1 Metrics

Extend `src/utils/metrics.py`'s existing `RankingMetrics`. **No new metrics class.** Add one
rank-based entry point:

```
RankingMetrics.compute_from_ranks(ranks, k_values)
```

Its contract, small and explicit:

- `ranks` are **positive 1-based integers**; zero, negative and boolean values are rejected;
- **empty input raises** — it does not return `0.0`;
- `k_values` are positive integers;
- **all metrics share one denominator.**

**Ranking policy stays in the harness, not in the metrics module.** Score ordering, tie policy,
candidate identity and missing-ground-truth outcomes are measurement concerns; `src.utils.metrics`
must not become aware of Modes A–D. A ground truth outside the rank list stays outside it and is
handled by the measurement layer.

**An empty authoritative cohort fails.** It does not return `0.0` or a sentinel mean rank — that is
the defect already corrected once, when `generate_report` emitted a fabricated `mean_rank` of `0.0`
(commit `7dab728`). Untruncated ranks are what make that field computable again.

---

## 5. Two metric families

| Family | Purpose |
|---|---|
| `legacy_mrr_truncated_at_20` | reproduces the frozen CLI's behaviour; **the parity target** |
| `untruncated_mrr`, `mean_rank`, `hits_at_{1,5,10,20,50,100}` | authoritative, used across A/B/C/D |

> **`untruncated_mrr` is never asserted to reproduce `val_mrr`.** If the truth ranks 21st the legacy
> metric contributes 0 while untruncated MRR contributes 1/21.

**Two different K's, and conflating them was an error.**

| Metric | K | Role |
|---|---|---|
| `legacy_mrr_truncated_at_20` | **fixed at 20** | the **exact frozen-CLI parity target** |
| `legacy_mrr_truncated_at_K` | runtime | general compatibility metric the new harness supports |

`EvalConfig.top_k_values` is programmatically configurable, but **the frozen CLI exposes no
`--top-k-values` argument** — its seven flags are `--checkpoint --data-dir --split --output
--batch-size --device --save-predictions` — and `save_predictions` hardcodes `predictions[i][:20]`
(`:513`). So anything compared against the frozen CLI is compared at **K=20**. Institutional
calibration uses K=20. The manifest records the value used.

### 5.1 Two ranking streams

Exact legacy top-K parity and a new deterministic tie order **cannot both come from one ranking**.
They differ whenever scores tie. So there are two:

| Stream | Definition | Used for |
|---|---|---|
| `legacy_ranking` | preserves the frozen evaluator's candidate ordering and tie behaviour | `legacy_mrr_truncated_at_20` parity only |
| `canonical_ranking` | score descending, then canonical **global** disease index ascending | `untruncated_mrr`, `mean_rank`, Hits@K, across A/B/C/D |

`canonical_ranking` is produced with PyTorch's own stable sort, not by building Python tuples and not
by adding a lexsort dependency:

1. establish candidate order by **global disease index ascending**;
2. `torch.argsort(scores, descending=True, stable=True)`.

Stability then preserves global-ID order inside every equal-score group. Tested against: exact ties;
different input candidate orders; and the ground truth at several positions within one tie block.

Both are recorded: `legacy_tie_policy` and `canonical_tie_policy_version`.

*(The alternative — one stream, with legacy parity weakened to tie-block equivalence — is available
if the two-stream cost proves unjustified. It is not the default, because exact parity is the
stronger calibration signal.)*

---

## 6. What Mode A parity can and cannot check

**The frozen oracle cannot verify most of the structural contract, because it never emits it.**
`save_predictions` writes only `sample_id`, `ground_truth`, and `predictions[i][:20]` — subgraph-local
index strings (`evaluate_model.py:505-519`). No sampled negative IDs, no global candidate IDs, no
batch membership, no candidate count, no mapping. Rev 1 claimed a parity contract the oracle cannot
participate in.

Two separate activities, and the boundary is set by what the frozen CLI actually writes to disk.

**Oracle-observable — the only things a subprocess comparison can check:**

- aggregate legacy metrics, from the report JSON;
- local top-20 predictions, from the predictions JSON.

**Not oracle-observable:**

- **score vectors** — `all_scores` is only a return value of `evaluate_model()` (`:326`, bound at
  `:485`) and is never serialised. Rev 2 listed it under oracle parity; **withdrawn**;
- sampled negative IDs; global candidate IDs; candidate mapping; batch membership.

**Do not modify the frozen script to export scores.** Score behaviour is validated by the primitive
tests (§4) and by new-harness Mode A tests — not by the oracle.

**Harness structural validation** is a separate activity, checked against the dataloader and
fixtures: global candidate IDs; sampled negative IDs and duplicates; candidate composition;
local↔global mapping; batch membership; candidate count. **The frozen oracle observes none of these
and is not claimed to.**

### 6.1 Local → global

Use `original_indices` **directly**:

```python
local_to_global = batch_data["original_indices"]["disease"]
global_id = local_to_global[local_id]
```

`subgraph_nodes[node_type] = torch.tensor(sorted(nodes))` is indexed by local index and holds the
original index (`src/kg/data_loader.py:336-342`). `node_mapping` is the *global-to-local* dict;
inverting it would be unnecessary work and one more chance for a direction error. Rev 1 said "invert
`node_mapping`" — withdrawn.

Measurement output persists **global** disease IDs. Subgraph-local indices are not comparable across
batches.

### 6.2 Score and aggregate parity

Same checkpoint, device, dtype, AMP setting, model mode, subgraph and candidate ordering. On CPU
float32 with AMP off, `torch.testing.assert_close` at strict float32 tolerances; dtype-aware
tolerances otherwise. `legacy_mrr_truncated_at_20` matches exactly when ranks match, ~1e-12
aggregation tolerance.

**Sampler defects are preserved, not fixed**, and instrumented: total count, unique count,
duplicates, candidate composition.

### 6.3 Manifest

Repository revision; checkpoint and dataset fingerprints; split and sample ordering; **batch size**;
shuffle; worker count; Python / NumPy / Torch seeds; negative-sampling strategy and count; subgraph
strategy, hops, neighbour limits; candidate-construction mode; truncation K; both tie policies;
**PyTorch and CUDA versions; deterministic-algorithms setting; cuDNN deterministic and benchmark
flags; device; dtype; AMP and AMP dtype.**

**Batch size is Mode A semantics, not a performance knob** — the candidate universe depends on batch
composition.

Recording these values is the whole requirement. **No cross-device bitwise equality is required and
no reproducibility subsystem is built.**

---

## 7. Mode D — constraints only

Rev 1 decided how Mode D would work. That belongs to B-0.5, and there is a real design problem in it:
**the public pipeline truncates to `top_k` (`pipeline.py:1271`) and also runs path construction and
optional explanation stages**, so obtaining an untruncated pre-`top_k` rank without duplicating
candidate construction is not a detail.

B-0.2 records only the constraints that must survive:

- Mode D **reuses production candidate construction and scoring**; it does not re-derive them.
- Normal and SP-degraded states are **distinct**, never both labelled "served".
- Ground-truth absence from the candidate set is **recorded as its own outcome**, never silently
  treated as rank = infinity. It is possible in Mode D — candidate construction is path-based —
  and impossible in A and C.

**No pipeline measurement hook is added in B-0.2.**

---

## 8. Completion, and what the tests are called

No institutional data or checkpoint here (`data/` is 32K; no `.pt` files).

| Status | Contents |
|---|---|
| **B-0.2 implementation complete** | harness; Mode A; both metric families and both ranking streams; batched primitives with equivalence tests; manifest; synthetic fixtures; tie and missing-ground-truth tests; CLI tests; import-linter layer |
| **B-0.2 institutional calibration pending** | real Mode A parity; checkpoint compatibility; timing and memory; any authoritative number |

**Synthetic legacy-equivalence test** — the frozen script and Mode A run against the same synthetic
fixture, comparing **only what the oracle writes to disk** (§6): aggregate legacy metrics and local
top-20 predictions. The legacy CLI is invoked as a **subprocess in one bounded integration test**; no
subprocess orchestration framework.

**The two processes are not deterministic by default, so the fixture is built to make that
irrelevant.** The frozen evaluator sets no Python, NumPy or Torch seed, and sampling uses
`random.sample` and `random.choice` for neighbours (`src/kg/data_loader.py:220, 263`) and
`np.random.choice` / `random.randint` for negatives (`:534-597`). Two subprocesses can therefore build
different subgraphs and different candidate sets.

The fixture removes the dependence rather than controlling the RNG:

- a small graph, with the relevant node degrees **at or below the neighbour-sampling limit**, so
  `random.sample` has nothing to choose between;
- **every valid negative falls inside the same two-hop disease universe**, so which negatives are
  drawn cannot change the candidate set;
- fixed batch size and sample order; `num_workers=0`; CPU; AMP disabled.

*Fallback, only if such a fixture proves impractical:* a bounded test launcher that seeds Python,
NumPy and Torch before executing the legacy script via `runpy`. **Do not modify the production
dataloader and do not build an RNG orchestration framework.**

> **This is not calibration and is not called parity.** Only the institutional run clears Mode A.

**No scorer decision or production conclusion may rest on any number from this harness until
institutional Mode A calibration succeeds.**

---

## 9. Implementation order

1. Batched primitives + equivalence and empty-mask tests.
2. `RankingMetrics.compute_from_ranks`; both metric families.
3. `src/evaluation/measurement.py` — Mode A, both ranking streams; import-linter entry.
4. `scripts/measure_scorer.py` — thin CLI.
5. Manifest; synthetic fixtures; the synthetic legacy-equivalence test.
6. Mark `scripts/evaluate_model.py` frozen in its docstring — the only change it receives.

Then the institutional Mode A calibration run: an **acceptance gate, not another engineering
branch.**
