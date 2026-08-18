# Spec 1 — Results review: authority, SP policy, decomposition, surfaces

**Status:** draft, under review. **Normative.** Supersedes rev 6 §1–§8.
**Authority above this document:** `docs/DISEASE_SCORER_POLICY.md` — statements 4a/4b, conditions
C1–C10, the eager-SP / lazy-evidence schedule, and the three limits that replace `top_k`. Where a
choice here is constrained by a condition, the condition is cited; this document does not restate or
override them.

---

## 1. Requirements

| # | Requirement |
|---|---|
| R1 | Candidate list raised from 10 to **200+** |
| R2 | Paginated, clinician-selectable page size (10 / 20 / 50 / 100) |
| R3 | SP analysis surface supports **sorting** by SP |
| R4 | SP analysis surface supports **range filtering** by SP |
| R5 | SP filter **defaults off** (§3.6) |
| R6 | Per-candidate score decomposition, LIRICAL-style, is acceptable to these clinicians (§4) |
| R7 | Result storage — Spec 2 |
| R8 | Navigating to results and back must not feel like a chore (§5.1) |
| R9 | `selection_limit` stays an inference-time setting, so the clinician bounds their own wait |
| R10 | Multi-user is a possible future scenario, **not a current requirement** — Spec 4 §6 |

R6 borrows a presentation *idea* — each row explains itself. No LIRICAL code is used and the
mathematics differs entirely: LIRICAL composes likelihood ratios, §4 decomposes an inner product.

---

## 2. Data model — four separate types

A single envelope carrying canonical data, page state and lazily-populated fields would make C10 —
binding a view to an immutable result ID — **unenforceable**, not merely untidy.

| Type | Contents | Lifetime |
|---|---|---|
| `CanonicalDiagnosisResult` | candidates, canonical scores, canonical ranks, selected count, scorer provenance, KG and SP fingerprints, `result_id` | **immutable** once produced |
| `AnalysisViewState` | sort key and direction, active filters, page index, page size, hidden count, facet counts, actor and session, view-state version | transient, per view |
| `CandidateAnalysisRecord` | SP quantities (§3) and per-phenotype contributions (§4), with provenance | produced eagerly with the result; versioned separately |
| `EvidenceRequest` / `EvidenceResponse` | Spec 3 | lazy subresource |

- **Page size and offset never appear in the canonical result.** They describe a view, not an inference.
- **Evidence never mutates the canonical result.** A stale or failed evidence fetch degrades the
  evidence panel, never the ranking.
- **View state is never reapplied on load.** A reopened snapshot opens in the canonical view (C2).
  The view state accompanying a *clinician action* is retained as audit metadata on that action (C5,
  Spec 3 §5).

---

## 3. SP analysis contract

### 3.1 Which quantity

| Quantity | Definition | Role |
|---|---|---|
| **Mean hop distance** | mean over the patient's mapped phenotypes of `d(p, candidate)`, unreachable pairs contributing `max_hops + 1` | **the filter and sort key**, and the primary display |
| **SP score** | `1 / (1 + mean hop distance)` | shown alongside, for continuity with the existing score |

"Show me candidates within an average of 3 steps" is a statement a clinician can form and check;
"show me SP ≥ 0.25" is not. The `1/(1+d)` transform also compresses the far end so heavily that the
gap between 1 and 2 steps is seven times the gap between 5 and unreachable, so it actively obscures
the distinctions a filter would be built on.

> **The two quantities carry equivalent ordering information in opposite directions: ascending mean
> hop distance equals descending SP score. The UI labels the active direction explicitly.**

*Numerical footing:* the ordering equivalence holds under `max_hops = 5`, `N ≤ 100` (contractual —
`src/api/routes/diagnose.py:56` caps the phenotype list) and float64 accumulation. **The derivation
lives in `src/inference/scoring.py`'s `sp_mean_distances` docstring, next to the code it constrains,
and is not restated here** — a product specification is the wrong home for a floating-point proof,
and two copies would drift. Regression tests pin it (`tests/unit/test_scoring_primitives.py`).

### 3.2 Reference set

**No candidate-relative normalisation at this stage.** Both quantities are absolute, so C7 holds
trivially: nothing depends on which candidates are present, and values are stable under paging,
sorting and filtering.

Values are stable **within** a result. Comparing them **across** results is meaningful only when the
KG and SP fingerprints, `max_hops`, traversal semantics and mapping coverage all match; any surface
placing two results side by side checks compatibility and says so when it fails.

Candidate-relative normalisation — the form the reference paper uses — is deferred by institutional
decision. If adopted, C7 requires its reference set fixed to the canonical selected set at result
production, never to a page or a filtered subset.

### 3.3 Status

```
sp_status = COMPUTED            # every submitted phenotype mapped;   numeric
          | COMPUTED_PARTIAL    # some phenotypes unmapped;           numeric
          | NO_TABLE            # no shortest-path artifact loaded;   no number
          | TARGET_UNMAPPED     # target not in the node mapping;     no number
          | NO_PHENOTYPE_MAPPED # no phenotype mapped at all;         no number
```

> **Numeric set: `{COMPUTED, COMPUTED_PARTIAL}`. Non-numeric set: `{NO_TABLE, TARGET_UNMAPPED,
> NO_PHENOTYPE_MAPPED}`.** Predicates use these explicit sets. `sp_status != COMPUTED` is **not** a
> test for unavailability — it wrongly captures the numeric `COMPUTED_PARTIAL`.

The schema is a discriminated union: numeric fields **required** on the first two variants,
**forbidden** on the other three. An absent number and a zero must not be representable as the same
thing. (The legacy `_calculate_sp_score` returns `0.0` on four distinct failure paths, and `0.0` is
*below* the value a genuine "no path found" produces — so a numeric sort would place a mapping
failure beneath a real negative.)

**`TARGET_UNMAPPED` is defined against the canonical node mapping, not against the SP artifact.** The
artifact is sparse: absence of a `(phenotype, target, type)` row means *no path was found within the
hop limit*, which is a normal computed outcome. A target that resolves in the node index but appears
in no SP row is a valid **computed, all-unreachable** candidate.

### 3.4 Two "unreachable" notions, and the denominator

| | What it is | Representation |
|---|---|---|
| A **pair** with no path | normal and common | a number: contributes `max_hops + 1` to the mean, inside a `COMPUTED` or `COMPUTED_PARTIAL` result |
| A **candidate** that could not be evaluated | no table, target unmapped, no phenotype mapped | a non-numeric status |

An all-unreachable candidate is still computed, and **both extrema are named**: it carries the
**maximum** mean hop distance (`max_hops + 1`) and therefore the **minimum** SP score
(`1/(max_hops + 2)`). Unreachable pairs and all-unreachable candidates can occur under
`COMPUTED_PARTIAL` too.

Every analysis record carries what makes a value interpretable:

```
phenotypes_submitted, phenotypes_mapped, unmapped_ids[]
unreachable_pair_count, max_hops
denominator_policy = MAPPED_ONLY_DENOMINATOR | ALL_SUBMITTED_DENOMINATOR
```

**Current behaviour is `MAPPED_ONLY_DENOMINATOR`**, structurally rather than by choice: unmapped
phenotypes never enter `phenotype_indices`, so the mean divides by the mapped count. Mapped
phenotypes with no table entry are the *other* case and **are** imputed as `max_hops + 1`.

Consequence: a `COMPUTED_PARTIAL` value is on the same scale as a `COMPUTED` value but not on the
same footing — at equal true reachability it reads closer. Sorting them together is permitted;
hiding coverage is not. **Every partial row shows `phenotypes_mapped / phenotypes_submitted`.**

### 3.5 Presentation of unavailable candidates

They sort into a single group at one end, never interleaved as low scores; they occupy their **own
facet**, never a numeric bucket; and a numeric range filter never removes them — it selects among
computed values, and the unavailable group is included or excluded by its own explicit control.

### 3.6 Default state and facets

**The SP filter defaults off and stays off.** A default-on filter would grant SP candidate-admission
authority through the view layer, which statement 4a denies it in the pipeline, and a first-time
clinician would receive a list already narrowed by a signal they did not choose.

Instead, show the **distribution** of mean hop distance across the selected set — a compact histogram
or quantile summary — so the affordance is discoverable without being applied.

Per C8: facet counts are computed over the **whole selected set**, never the current page;
pagination never changes them; facets show the count a filter *would* leave before it is applied
(C6). The unavailable group is a facet in its own right.

---

## 4. Per-candidate score decomposition

### 4.1 The exact decomposition

For phenotype embeddings `p₁…p_N`, their mean `m`, and candidate embedding `d`:

```
cos(m, d) = Σᵢ (pᵢ · d) / (N ‖m‖ ‖d‖)
contributionᵢ(d) = (pᵢ · d) / (N ‖m‖ ‖d‖)
```

Contributions **sum exactly to the raw cosine**, which is what makes them displayable next to the
score. Cost: one `(N, H) × (H, C)` matmul, about 1M multiply-accumulates at 20 phenotypes, `H = 256`
and 200 candidates.

### 4.2 Required handling

| Case | Rule |
|---|---|
| Score semantics | Contributions decompose the **raw cosine**. Any transformed score displayed elsewhere has its relationship stated |
| Phenotype not mapped | Contributes nothing and is listed as **not analysable**, distinct from contributing zero |
| Duplicate entries | **The analysis does not decide.** It consumes exactly the sequence, multiplicity, mask and denominator `N` the canonical scorer used. Deciding independently — even "correctly" — makes contributions stop summing to the score they claim to decompose |
| Padding masks | Excluded from `N` and from the sum |
| Negative contributions | Real and clinically meaningful; displayed, never clipped |
| Tolerance | Displayed contributions sum to the displayed score within a declared tolerance, asserted in test |

### 4.3 The near-zero-norm boundary

The canonical scorer normalises with epsilon clamping (`F.normalize`, `eps = 1e-12`), so
`denominator = max(norm, eps)`. Exempting only *exact* zero is insufficient: a nonzero norm below
`eps` still diverges, because the scorer would use `eps` while the contribution formula uses the
smaller raw norm — the contributions then stop summing to the canonical score and can display as
enormous values.

> **When `‖m‖ ≤ eps` or `‖d‖ ≤ eps` — using exactly the scorer's epsilon, dtype and device semantics
> — the canonical score remains defined and is displayed. The decomposition is marked unavailable and
> is explicitly exempt from the exact-sum invariant.**

The invariant is asserted in test over the non-degenerate case only, and the exemption is named in
that test rather than left as an untested branch.

*Why not mirror the clamp instead.* Mirroring is mathematically exact but produces contributions of
order `10¹²` cancelling to zero — unreadable and actively misleading on a clinical surface. That
applies to a **near-zero patient mean produced by cancellation**; if the *disease* vector is exactly
zero, every dot product is zero and no large contributions arise. Both cases are covered by the rule
above.

### 4.4 What this explains — and what it does not

The decomposition explains a candidate's **absolute score**. It does **not** explain why that
candidate ranks above another: a phenotype contributing strongly to *every* candidate explains none
of the differences between them, yet would dominate the display by being the largest number on every
row.

Required wording, verbatim:

> **"Per-phenotype contribution to this candidate's raw cosine score."**

Forbidden: ~~"Why this candidate is ranked here."~~

The comparator analysis that *would* address ranking differences — each phenotype's contribution
relative to its mean contribution across the selected set — is nearly free to compute but is a
different analysis with different failure modes. **Deferred, and recorded so the gap is visible
rather than forgotten.**

### 4.5 Phenotype confidence is not a weight

Snapshots retain phenotype confidences, but pooling is an unweighted mean and the scorer does not use
confidence as a weight. Displaying a confidence next to a contribution invites the inference that one
influenced the other. The contribution surface records:

```
used_by_scorer = false
effective_aggregation = UNWEIGHTED_MEAN
```

Contribution provenance binds each output to the original ordered phenotype occurrence, duplicate
multiplicity, confidence, mapping outcome, and effective scorer index.

---

## 5. Two surfaces

| | Inference page | Analysis workspace |
|---|---|---|
| Question | "Did the run succeed, and is the answer obvious?" | "Work through the whole list systematically" |
| Shows | `summary_limit` rows (default 10), essential columns | full selected set, paginated, facets, sort, decomposition, SP panel |

The split is forced: 200+ rows cannot be usefully rendered on the inference page, and the only choice
is whether that page shows 0 rows or ~10 — and 0 turns it into a form with no feedback. The two tasks
are also different cognitive modes.

**Share the candidate view-model and the formatting authority; do not force both surfaces through one
heavily conditional component.** Two thin presentational components over one shared view-model. The
detailed reasoning display currently on the inference page moves to the workspace.

### 5.1 Navigation (R8)

The UI is already tabbed (`src/webui/app.py:191-201`) and per-session result state already exists
(`results_state = gr.State(None)`, `src/webui/components/diagnosis_panel.py:747`), so the mechanism
is present rather than new. Gradio renders tab content and hides it rather than unmounting, so
component state is expected to survive the round trip — **to be confirmed against the deployed
version rather than assumed**.

**The direction needing design attention is coming back, not going.**

- Within one result, the round trip preserves **page, sort and filters** — these live in
  `gr.State` and are normative.
- **Scroll position is best-effort, not normative.** It is browser DOM state, not component state,
  and whether it survives depends on the deployed Gradio version. Use Gradio's built-in `Tabs`,
  `Tab`, `State`, event outputs and component updates first; add custom JavaScript only if that is
  demonstrably insufficient, and do not build a separate client router or state store.
- Re-running inference produces a **new `result_id`**, so the previous view state does not apply.
  Required by C10 and correct, but it must be **explained on screen** or it reads as settings
  vanishing for no reason.

*Worth measuring after deployment:* how often clinicians open the workspace. If it is every time, the
signal is that `summary_limit` or the summary's columns are wrong — not that navigation needs work.

---

## 6. Limits and cost

| Parameter | Meaning | Authority | Default | Min | Max |
|---|---|---|---|---|---|
| `selection_limit` | candidates selected and eagerly enriched | clinician, within deployment bounds | **[OPEN]** | **[OPEN]** | **[OPEN]** |
| `summary_limit` | rows on the inference page | deployment | 10 | — | — |
| `page_size` | rows per workspace page | clinician | **[OPEN]** | 10 | 100 |

Also **[OPEN]**: an **interaction latency target** covering **inference + eager enrichment +
automatic snapshot publication**. The snapshot write is inside the target, not outside it: if
`snapshot_status` is final in the diagnosis response, then the write, the file `fsync`, read-back
validation, the rename, the **snapshot-directory `fsync`** and rotation all complete before the
response does (Spec 2 §7.3, §7.4). Two `fsync` calls, not one — the second is what the eviction is
allowed to depend on, and it is inside the target for the same reason the first is.

**Do not add a background-job subsystem to move it out of the target.** The payload is a few hundred
kilobytes; benchmark the synchronous path first and only then decide whether it needs hiding.

Without this number, Spec 3's budget design has nothing to aim at and §6.2's acceptance criterion has
nothing to pass or fail against.

### 6.1 What `selection_limit` bounds

| Stage | Bounded? |
|---|---|
| Scoring every disease in the KG universe | **No.** Statement 3 requires the full universe; the cost is a single matmul, independent of this parameter |
| Selecting candidates from those scores | Yes — this is its definition |
| Eager SP enrichment | **Yes** — where R9's cost control acts |
| Per-phenotype contributions | Yes |
| Evidence traversal | No — bounded per target instead (Spec 3 §2) |

> **`selection_limit` is a presentation and enrichment bound, never a scoring bound.** Using it to
> limit how many diseases are scored would be candidate gating under a new name.

### 6.2 The eager-SP performance dependency is unmet

`sp_mean_distances` has a batched-shaped interface but still loops candidate-by-phenotype, scanning a
slice per pair — about 4,000 slice scans at 200 candidates and 20 phenotypes, over 550,000 at
full-universe scale (`src/inference/scoring.py`).

> **Eager SP over 200 candidates is contingent on a genuinely vectorised SP lookup being delivered
> *and benchmarked*. It is not yet.**

This is a **B-1 acceptance criterion measured on institutional hardware**, against the declared
latency target, for a representative 20-phenotype, 200-candidate request. Sorted composite keys with `torch.searchsorted`, or a
sparse index, are **benchmark candidates, not a chosen implementation**. Measure the simplest
built-in PyTorch approach first; build a custom index only if it loses.

**That work completes the B-1 eager-SP dependency. It does not by itself complete B-0**, whose
remaining scope is: evaluator migration, modes A–D, full-universe candidate construction, untruncated
MRR and Hits@{1,5,10,20,50,100}, served-configuration manifests and fingerprints, negative-set count
and composition instrumentation, the paired-cohort/strata/bootstrap protocol, the measurement-output
schema, and the institutional measurement run.

---

## 7. Deferred, with reasons

| Item | Why |
|---|---|
| **Triage state** (reviewed / dismissed / flagged) | Persistent clinical state: actors, reasons, history, permissions, concurrency, retention, export. "Dismissed" must never silently remove a candidate — the principle behind C6 |
| **Multi-user and authentication** | Spec 4 §6 |
| **Candidate-relative SP normalisation** | §3.2 — institutional decision |
| **Comparator contribution analysis** | §4.4 — needs its own design and validation |
