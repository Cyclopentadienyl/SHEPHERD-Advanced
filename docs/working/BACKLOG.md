# Programme backlog — what is open, in what order, and why

**Purpose.** One ordered list across every live phase, so the next action is
readable without reconstructing six review threads. Phase documents keep their
own detail; this file holds **ordering, dependencies and blockers only** and must
not restate their decisions.

**Status:** first revision, written after the first institutional measurements
came back and produced two findings that reach further than the stage they came
from. One direction it originally proposed was checked against source before
submission and withdrawn; §6 records that rather than hiding it.

---

## 1. The goal, unchanged

**Measure what the disease scorer actually does before changing it.** The offline
evaluator and the deployed pipeline score different candidate sets with different
formulas, so neither number describes the other. Work item B-0 builds a harness
and an A/B/C/D ladder that isolates **one difference per step**.

Everything below serves that, or is a consequence of discovering that part of it
does not currently work.

---

## 2. Established facts

M1-M7 were **measured on deployment hardware**; M8 was **read from source**. Both
are checkable; neither is recalled or inferred. These are what changed the
ordering.

| # | Fact | Where |
|---|---|---|
| M1 | **No checkpoint carries `metadata`.** All carry `config`, `data_fingerprint`, `epoch`, `logs` | checkpoint scan, both workspaces |
| M2 | **`in_channels` is 128** in every checkpoint; the frozen evaluator's hardcoded fallback is 256, which is the size mismatch it dies on | same |
| M3 | **The filename number is `val_mrr`** — `model-45-0.6975.pt` carries `logs["val_mrr"] = 0.69754…` | same |
| M4 | **100% of validation diseases appear in training** — 7,970 of 7,970. Train: 100,000 samples over 10,576 diseases; val: 15,000 over 7,970 | overlap audit |
| M5 | **SP reachability is dense**: a typical phenotype reaches 71.3% of diseases within 5 hops (16,845 of 23,640) | artifact scan |
| M6 | **The SP lookup misses the provisional latency budget by 1.7-2.5x** on the real artifact and a GB10 SPARK | `scorer-measurement/PLAN_B04.md` §9 |
| M7 | **Zero duplicate rows** on two independently built artifacts of different HPO vintages | same, §10 |
| M8 | **The trainer's own validation loop is Mode-A-shaped, not Mode-C-shaped** — per-batch subgraph forward, cosine against *the subgraph's* disease rows, top-20 truncation, MRR from the same function Mode A calls. See §2.1 | read from `trainer.py`, `metrics.py`, `measurement.py` |

### 2.1 M8 in detail — where `val_mrr` comes from

Read from source rather than recalled, because it overturns a direction this file
previously proposed.

| Step | `trainer._validate` | Mode A's legacy stream |
|---|---|---|
| Encoder input | `batch_data["subgraph_x_dict"]`, `subgraph_edge_index_dict` — the **per-batch subgraph** (`trainer.py:628-634`) | the same two keys off the same dataloader (`measurement.py:706-710`) |
| Padded phenotype ids | `phenotype_ids.clamp(min=0, max=…)` (`trainer.py:739`) | the same clamp, kept as oracle parity (`measurement.py:738`) |
| Patient vector | masked mean (`:744-751`) | `masked_mean_pool` (`:742`) |
| Candidates | `disease_emb = node_embeddings["disease"]` — **the subgraph's diseases**, not the KG's (`:762-763`) | `cosine_score_matrix(patients, disease_emb)`, same rows (`:743`) |
| Score | cosine, both sides `F.normalize`d (`:761-763`) | `cosine_score_matrix` |
| Ranking | `scores.sort(dim=-1, descending=True)` (`:651`) | `legacy_ranking`, which is that same call (`:151-157`) |
| Truncation | `pred_indices[:20]` (`:656`) | `LEGACY_TRUNCATION_K = 20` (`:243, :749`) |
| Metric | `RankingMetrics().compute_all(...)["mrr"]` (`:663-666`; `metrics.py:285`) | `RankingMetrics().mean_reciprocal_rank(...)` (`:809-810`) |

Every row is the same operation. **`val_mrr` is a Mode-A-shaped number, and M3
says every checkpoint carries it.**

**What is not established**, and would have to be closed before treating it as a
calibration target:

| # | Gap | Why it matters |
|---|---|---|
| D1 | The candidate set **is** the batch's subgraph, so batch size, shuffle and sampler policy change it | Two runs of the same checkpoint over the same split give different MRR if batching differs. The trainer's `DataLoaderConfig` values may be recoverable from the serialized `config` (M1); whether they are *sufficient* — sampler seed, neighbour policy — is unchecked |
| D2 | The trainer validates inside `autocast(..., dtype=self.amp_dtype, enabled=self.use_amp)` (`:633`); Mode A's traversal has no autocast | Changes score values and therefore tie order |
| D3 | The trainer compares `str(column_index)` against `str(disease_ids[i])` | Only meaningful if the dataloader emits subgraph-local disease ids. Mode A names the variable `disease_ids_local`; that both are the same space is an assumption until read |
| D4 | Cohort and split | `val_mrr` is over the val split at one epoch. Reproducing it needs the same split and the same sample count |

---

## 3. What these facts broke

### 3.1 Calibration cannot run as designed — this is the root blocker

Two loaders fail on the same missing key, in two different ways:

- `build_legacy_mode_a_model` indexes `checkpoint["metadata"]` and
  `checkpoint["in_channels_dict"]` directly (`measure_scorer.py:189-190`).
  Neither exists (M1) — **`KeyError`**.
- `create_model_from_checkpoint` uses `.get()` with defaults instead
  (`evaluate_model.py:163-173`): three node types, two edge types, 256 channels.
  It therefore builds a **structurally wrong model** and dies later in
  `load_state_dict` against a real 128 (M2).

**Mode A cannot execute on any checkpoint this project has produced**, and neither
can the frozen evaluator.

**Everything downstream inherits this.** B-0.2 and B-0.3 are implementation-
complete but their acceptance was always "the harness reproduces the oracle".
Without an oracle run, the harness is unvalidated — and B-0.5 would build a
statistical protocol on top of it.

#### What Mode A is actually carrying

Mode A has been doing **two** jobs under one name, and only one of them is broken:

- **(a) the bottom rung of the ladder** — per-batch subgraph encoder, subgraph
  candidates, cosine. A→B isolates encoder scope only if A holds this shape.
- **(b) bit-parity with the frozen evaluator** — the clamp, the truncation at 20,
  the local-index prediction artifact, and the oracle-mirroring model builder.

Job (b) is what requires `checkpoint["metadata"]`, and job (b) is unexecutable.
Job (a) needs no oracle at all.

#### Directions, corrected by M8

| Direction | Cost | Verdict |
|---|---|---|
| Repair the frozen evaluator to derive `in_channels` from features | Small | It is **frozen on purpose**. A repaired oracle is not the oracle whose number is being reproduced — and M8 removes the reason to want one |
| Accept that no oracle number exists and drop Mode A | Large | Discards the control the whole ladder is built around |
| ~~Calibrate **Mode C** against the trainer's `val_mrr`~~ | — | **Withdrawn — the premise is false.** The trainer's validation loop does not score the full disease matrix; it scores the batch's subgraph (M8, §2.1). Its `val_mrr` is not a Mode-C-shaped number |
| **Calibrate Mode A against the trainer's `val_mrr`** | Medium | Every step matches, verified line by line (§2.1). Requires closing D1-D4, and requires Mode A's model builder to stop mirroring the oracle's defaults and construct the way the **trainer** did — which `build_shepherd_model` already does by deriving in-channels from `x_dict` (`shepherd_gnn.py:569-573`) |

**The fourth is the recommendation, and it is not a workaround.** It replaces an
unexecutable target with an executable one that has the same shape, and M3 says
the value is already sitting in every checkpoint — no institutional oracle run
needed to obtain it.

**Its real cost, stated plainly:** it retires job (b). A Mode A that constructs
its model the way the trainer did is no longer bit-parity with
`scripts/evaluate_model.py`, and the legacy-removal checklist in
`scorer-measurement/README.md` is written around a parity that would then never
have been demonstrated. That checklist would need rereading, not just executing.
Whether losing (b) is acceptable is the decision item 1 asks for — **it is not
settled here.**

### 3.2 Validation measures no unseen-disease generalisation at all

M4 is the maximal case: **every** disease in val appears in train. The escalation
already sustained by review said the generator "permits" overlap; the measurement
says it is total.

This does not invalidate any engineering result. It **bounds what every number
may claim**, and the bound is tighter than the caveat currently drafted.

### 3.3 A policy inference is contradicted

`DISEASE_SCORER_POLICY.md` §3.5 records, explicitly as unmeasured, that most
candidates fall outside the 5-hop table so the SP term degenerates to a
reachability indicator. M5 says the opposite. The correction is factual and
touches no normative statement.

---

## 4. Ordered backlog

Ordered by **dependency**, not by size. An item may only start when everything it
depends on is resolved.

| # | Item | Depends on | Owner | Size |
|---|---|---|---|---|
| **1** | **Decide what calibration means now** (§3.1). The trainer's candidate universe is now **verified** (M8) and the recommendation is direction 4; what needs deciding is whether retiring oracle bit-parity is acceptable, and closing D1-D4 | — | needs review | design question |
| **2** | Update the contamination caveat to the measured 100% (§3.2), with both split file hashes | — | author | small |
| **3** | `DISEASE_SCORER_POLICY.md` §3.5 correction (§3.3) | — | author | ~5 lines |
| **4** | Reply to the sustained-with-narrowing contamination review | 2 | author | text only |
| **5** | **B-0.4 prototype phase** — prototype A and B, both caller shapes, per-subprocess memory | — **independent of 1** | author | the next real engineering |
| **6** | Confirm which checkpoint is authoritative. `select_checkpoint_in_dir` picks by logged ranking metric and would choose `model-22-0.7372.pt`, not the `model-45` supplied | — | institution | question |
| **7** | B-0.2 / B-0.3 institutional runs | **1** | both | blocked |
| **8** | B-0.5 — Mode D, the intermediate rung, statistical protocol | **1, 7** | — | not started |

**Parked deliberately, not forgotten:** `task-scope/` Q2–Q5 (settled, unscheduled)
and `scorer-retraining/` (scoping only, four gates uncleared). Neither blocks nor
is blocked by anything above.

---

## 5. Why item 5 does not wait for item 1

B-0.4 measures the shortest-path **lookup cost**. It consumes no checkpoint, no
sample split and no model — only `shortest_paths.pt`. Both its gates are cleared
(M6, M7), and its findings do not depend on whether the harness is calibrated.

So the root blocker at item 1 does not idle the engineering. It blocks the
**numbers**, not the **work**.

---

## 6. Contradictions found and resolved in this revision

Recorded because the request that produced this file was to stop them recurring.

| Contradiction | Resolution |
|---|---|
| `PLAN_B04` §5.3.2 required scanning "the deployed artifact" while §10.1 established there is no single one | §5.3.2 restated as evidence about the generator's invariant |
| `PLAN_B04` §9.4's estimated "+7 GB" stood beside §10.2's measured ~24 GB with no link | §9.4 marked as an estimate and pointed at the measurement |
| The `PLAN_B04` status header was spliced into a broken sentence by a scripted edit | Rebuilt as two named gates with their verdicts |
| `--split` defaulted to `test`, which the generator never writes | Made required on both entry points; the error names the splits that exist |
| This file's own first draft proposed calibrating **Mode C** against the trainer's `val_mrr`, on an unverified guess about the trainer's candidate universe | Verified before submission and **withdrawn**: the trainer scores the batch subgraph, so the number is Mode-A-shaped (M8). The direction table now records the withdrawal rather than deleting it |

**Authority above everything here:** `docs/DISEASE_SCORER_POLICY.md`.
