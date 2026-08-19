# Task scope — what the supplied-short-list scenario changes

**Status:** rev 3. Scope decisions reviewed and settled; **implementation
uncommitted**. Rev 2 revised Q1, Q3 and Q4 after the reasoning or facts were
found wrong; rev 3 narrows F2's validation claim, replaces the proposed
per-candidate field vocabulary, and withdraws a sampling policy this document had
no standing to choose. The factual error it found in
`../../DISEASE_SCORER_POLICY.md` is corrected there under that record's rev 4.

The deploying institution described a second use case: clinicians who have
already narrowed a patient to **a list of suspected diseases or gene variants**
and want the system to work against that list. That resembles the reference
paper's causal-gene setting, which is the one task where the paper fuses a
shortest-path term — so the question "does this change the roadmap?" is real
rather than rhetorical.

**It does not change work item B-0**, and nothing here proposes adding to it.
The five questions below are about scope and sequencing outside B-0. Each
carries the author's position; the point of review is to contest them.

---

## 1. Context supplied by the institution

Recorded because it changes how two of the questions should be read:

- The project is **pre-alpha**. The current objective is to make the paper's
  design run at all; testing is organised around the basic architecture.
- `candidate_genes` was **requested as a reserved interface**, deliberately, so
  the capability can be switched on later without operating on the pipeline.
  The institution is not using it and knows it is inert.
- The institution expects to keep requesting feature changes and extensions.
  **Low coupling, maintainability and modular upgradeability are therefore
  first-order requirements**, not preferences — which is also why the legacy
  measurement path is being kept removable rather than merely tolerated.

---

## 2. Verified facts

Every row was read from the tree or from the paper before this document was
written. Nothing here is recalled.

| # | Fact | Source |
|---|---|---|
| F1 | The paper uses **three different similarity functions**, and only causal-gene discovery carries the SP term — the only task with a candidate short list (13.3 genes expert-curated, 244.3 variant-filtered). Disease ranking is `−‖z_d − z_P‖²`, over **all** diseases in the KG | `docs/SP_SCORE_GUIDE.md:341-349`; `docs/DISEASE_SCORER_POLICY.md:3.1` |
| F2 | `candidate_genes` is **structurally accepted as `Optional[List[str]]` and stored** on `PatientInput` — currently without length, non-blank or identifier-semantic validation — and **read by no scoring path**. Five occurrences repository-wide; none in `src/inference/pipeline.py` | `api/routes/diagnose.py:63,216`; `inference/input_validator.py:437,466`; `core/types.py:388` |
| F3 | `ShepherdGNN` constructs **no task head**, and the deployed checkpoint passes a strict `load_state_dict`, so it carries no task-head parameters | `DISEASE_SCORER_POLICY.md:3.2, 3.3` |
| F4 | **No gene-targeted ranking objective is active in the current training path.** The link-prediction and ortholog losses are gated on `positive_triples` / `negative_triples` / `ortholog_pairs`, and **no dataset, collate function or trainer in `src/` produces those keys** — the three names occur only inside `loss_functions.py`. `gene_ids` is not consumed by a loss either. Gene representations **may still receive indirect gradients** through heterogeneous message passing and the supervised disease objective; no current evidence establishes that their geometry is calibrated for patient-to-gene ranking | `training/loss_functions.py:513-584`; absence verified across `src/` |
| F5 | `DiagnosisSample.gene_ids` is collated into batches and read at `data_loader.py:929-930`, but **no loss consumes it** | `kg/data_loader.py:614,675,741-765,929` |
| F6 | A **dormant supplied-candidate mechanism already exists** in the training dataloader: `DiagnosisSample.candidate_disease_ids` plus `include_all_candidates`. `file_storage.read_samples` does not read the field, so it is dead on the measurement path | `kg/data_loader.py:613,656-669`; `kg/storage/file_storage.py:72-79` |
| F7 | `cosine_score_matrix(patient_matrix, candidate_matrix)` is **candidate-agnostic**, and `sp_mean_distances` already takes a sequence of targets | `inference/scoring.py:193,240` |
| F8 | The approved disease score is **raw cosine, explicitly interim**, because it reflects what the deployed checkpoint's training objective optimised — not the paper's Eq 18. Changing it requires the paper-parity retraining track, which is outside work item B | `DISEASE_SCORER_POLICY.md:1` statement 2 |

**F4 is the load-bearing one**, and an earlier revision overstated it as "gene
embeddings are unsupervised". That claim was not established: `ShepherdGNN` does
heterogeneous message passing, so gradients from the supervised disease and
contrastive objectives reach shared parameters and, through them, gene
representations. What is established is narrower and still sufficient to gate a
gene scorer — **nothing trains gene geometry for patient-to-gene ranking.**

**Why this cannot be settled from the source alone.** Which gradients actually
reach gene representations depends on which edge types connect genes to the
supervised node types, and the model derives its metadata from the built KG's
`edge_index_dict` keys rather than declaring them. The gradient-path audit
therefore has to read a real workspace artifact, not just the model code. That
audit is the gene work item's first step (Q5), not this document's.

---

## 3. Questions

### Q1 — Is a reserved-but-inert interface acceptable as it stands?

`candidate_genes` (F2) is a deliberate reservation, not an oversight. The
concern is narrower than "it does nothing": **nothing marks it as reserved.**
The field description reads *"Pre-selected candidate genes to consider"*, which
describes a working feature, and no test asserts the field is inert — so the day
the system leaves pre-alpha, nothing fails to point out that it still is.

**Settled: keep the interface, and make the caller see the inertness.** An
earlier revision proposed only a description change plus an inertness test.
Review rejected that as insufficient, correctly: **a test that proves the field
is ignored tells nobody but us.** The caller who sent it learns nothing.

Four bounded changes:

1. The description states the field is **reserved — accepted but currently
   ignored**, and does not affect disease candidates, scores or ranks.
2. When the field is non-null, the response carries a caller-visible warning:
   `candidate_genes was supplied but is not used by the current disease scorer.`
   **`DiagnoseResponse.warnings: List[str]` already exists**
   (`api/routes/diagnose.py:140`), so this adds no API surface.
3. A list-length bound and non-blank-string validation, matching the convention
   `phenotypes` already sets (`:53-56`). **No gene-identifier ontology** — that
   belongs to the interface's real design.
4. Tests covering both result invariance and the warning.

Rejected: returning 400 — the institution asked for the reservation, and
refusing the field breaks what it was reserved for. Also rejected: a feature
registry, a reservation framework, or implementing gene scoring now.

### Q2 — Does this scenario change the legacy-removal plan?

**Position: no, and the checklist in `../scorer-measurement/README.md` stands.**

The legacy candidate set is **answer-seeded** — a 2-hop subgraph grown from the
ground truth plus sampled negatives — so the answer is guaranteed present. A
clinician's short list carries no such guarantee and is known at inference time.
They are opposites in the property that matters.

Nor does legacy own the capability. Scoring a restricted candidate set lives in
`cosine_score_matrix` (F7), which the checklist already retains. A shorter
candidate list needs **less** machinery than Mode C's full universe, not more.

### Q3 — Supplied candidate universe: an input field, or a clinician-selectable scoring mode?

**Settled: an explicit request/result variant. Neither an unmarked nullable
field nor a post-result mode toggle.** At minimum
`OPEN_DISEASE_UNIVERSE` / `SUPPLIED_DISEASE_UNIVERSE`, where the supplied-universe
result records the universe kind, the supplied ids, the mapping outcomes and the
provenance. Its ranks are canonical **within that supplied universe**.

**The earlier argument for a bare optional field was wrong, and so was the rule
it cited.** It claimed statement 4a forbids a scoring-mode selector, and that an
input field escapes 4a because the clinician supplies clinical information rather
than an algorithm choice. Review rejected the second half as a false dichotomy —
an optional field that narrows the candidate universe changes canonical scoring
semantics whatever it is called — and the first half misreads the rule: **4a is
about SP, and about immutability *once produced*. It does not govern request
construction at all.**

**The rule that does govern this is statement 3**, which neither the original
argument nor the review cited: *"Every disease in the configured KG universe is
scored before selection."* A supplied universe scores a subset, so it is not a
loophole in 4a — it is a **declared exception to statement 3**, and that is
exactly why it must be a marked variant rather than a nullable field. An unmarked
field would make a statement-3 result and a non-statement-3 result share one
shape.

**One distinction survives the rejection, and it keeps the future change small.**
A supplied universe and a scoring-mode selector are not equivalent in what they
vary: the first changes *what is scored* under one fixed formula, the second
changes *the formula*. So a supplied universe needs a **universe discriminator**
with identical score semantics, whereas a mode selector would need §7's full
versioned discriminated union over **score semantics**. Both must be marked; only
one of them reaches into §7. Recording this so a later stage does not build the
larger mechanism when the smaller one is what the requirement asks for.

**The three supplied-input workflows are different tasks and must not share one
result shape:**

| Supplied | Task |
|---|---|
| Disease ids | The disease scorer over a restricted universe |
| Genes | Causal-gene prioritisation (Q5) |
| Variants | Annotation and variant→gene mapping, *then* causal-gene prioritisation |

In particular **`candidate_genes` is not a disease-universe field**, and using it
as one would merge two of these three.

This does close the gap `DISEASE_SCORER_POLICY.md` §5 identified — the current
top-k analysis set resembles the paper's short list *in size only*, because ours
is model-ranked while the paper's is externally supplied, and a clinician's list
is externally supplied. That remains true; it is a reason the scenario is worth
designing for, not a reason to skip the variant.

**One discriminator, not a parallel hierarchy.** If policy §7's versioned result
union already exists when this feature lands, the universe discriminator belongs
**inside the appropriate score-semantics variant** rather than beside it as a
second result hierarchy.

**The request-union or endpoint design is a later work item. Not B-0.**

### Q4 — Is an SP ablation a new measurement stage?

**Settled: no new run and no new mode — but it is a B-0.5 output contract, not
something already available.**

An earlier revision said the analysis could run "offline over what B-0 already
records", citing `DISEASE_SCORER_POLICY.md` §4 and §8 condition 4. **That is
false about the implemented harness.** `ModeResult` carries `sample_ids`,
`truth_global_ids` and `canonical_ranks`, and `to_ranks()` emits
`{sample_id, ground_truth, rank}` — **aggregate metrics and the truth's rank, no
per-candidate score components at all** (`evaluation/measurement.py`). The claim
was repeated from the policy record without being checked against the code.

**That makes the policy record wrong, not just this document.** §4 and §8
condition 4 both assert the components are recorded, and
`DISEASE_SCORER_POLICY.md` is a **living** document — by `docs/README.md`'s own
rule, a living document that disagrees with the code is a bug in the document.
Left uncorrected, the next reader forms the same wrong belief from the same
sentence. **Corrected under that record's own revision mechanism as rev 4**, a
factual amendment only: §4 and §8 condition 4 now state that the components are
not recorded and that avoiding a second institutional run is conditional on
B-0.5. No normative statement changed, so reapproval is not required — **but the
deploying institution should be told its authority document was corrected**, and
rev 4 says so.

**The contract B-0.5 must persist**, per candidate, for the analysis to be
possible without another inference run: global candidate id; candidate
universe/mode; raw cosine; SP availability status; raw or mean distance; this
repository's SP transform; the η score where applicable; ground-truth identity;
artifact fingerprints.

**The served API's field names must not be reused.** An earlier revision proposed
mirroring `DiagnosisCandidate`'s `confidence_score` / `gnn_score` / `sp_score`
(`api/routes/diagnose.py:118-121`) as the vocabulary. **That was a bad
suggestion, and this document had already cited the reasons.** Those fields carry
known semantics defects, not reusable canonical names:

| Field | Why it is not safe to reuse |
|---|---|
| `gnn_score` | The served **normalised** embedding score; B-0 records **raw cosine**. Same name, different coordinate system |
| `confidence_score` | An η mixture on one path and a reasoning score on the fallback path, with no per-result discriminator (policy §2) |
| `sp_score` | Returns `0.0` on four distinct failure paths, *below* the `1/7` a genuine "no path" produces — collapsing unavailable states into a number (policy §1.1, C9) |
| — | No raw mean distance and no typed SP status exist at all |

Explicit measurement names instead: `candidate_id`, `candidate_universe`,
`raw_cosine`, `mean_sp_distance`, `sp_status`, `repository_sp_score`,
`mixture_score`, `ground_truth`. A manifest may record how these map onto the
served API's fields. **The same name must never denote two coordinate systems.**

*One vocabulary, not two:* `sp_status` should be the typed status
`src/inference/scoring.py` already reserves — `COMPUTED` / `COMPUTED_PARTIAL` /
`NO_TABLE` / `TARGET_UNMAPPED` / `NO_PHENOTYPE_MAPPED` — which that module's
docstring assigns to "the B-1 analysis record". B-0.5 and B-1 should not invent
two status vocabularies for one quantity.

**The row count may be material and must be measured.** At Mode C scale it is
`n_samples × n_candidates`, and the disease universe is ~27,990 (policy §4); for
a UDN-sized cohort — 465 patients, the paper's figure, not a committed
institutional number — that is ~13 million logical rows for one mode. **That is
not by itself proof the artifact does not fit**: verbose JSON may be
unacceptable where typed columnar arrays are manageable, and a predeclared
analysis may be computed during the run without persisting every row at all.

**A top-N candidate window is invalid** for an analysis intended to test what
cosine selection excludes. Beyond that, whether B-0.5 retains the full candidate
axis for all patients, uses a statistically approved random or stratified patient
sample, or computes predeclared analyses during the run **is a B-0.5 statistical
and storage decision**, not this document's.

*An earlier revision picked one — a random patient subsample — on the grounds
that it is unbiased "in the axis that matters". That addressed candidate-axis
conditioning only: it introduces patient-sampling variance, can lose rare-disease
strata in a rare-disease cohort, and may underpower the very subgroup analyses
this section says require B-0.5's protocol. Choosing it here was premature.*

Two limits to state with any result:
- §8 condition 1: the comparison tests **this repository's** SP transform
  `1/(1+mean(d))`, not the paper's Eq 13. A result in either direction is a
  statement about this implementation.
- B-0.5's statistical protocol governs. Without it, "better in this scenario" is
  not separable from noise, and per-scenario subgroup analysis multiplies that
  risk.

### Q5 — Should causal-gene scoring become its own work item?

**Position: yes, as a named item — and its scope must open with F4.**

No work item currently covers it. The Phase-1 paper-parity list in
`docs/RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md` §6.B has four entries and
gene scoring is not among them; `DISEASE_SCORER_POLICY.md` §4 calls candidate-gene
scoring "unbuilt future work". Meanwhile the institution has reserved an
interface for it. That mismatch is the finding.

It covers **causal-gene and supplied-variant prioritisation**, and it is separate
rather than an extension of the disease path because it is the **only** task
where the paper's SP fusion is parity rather than deviation (F1). Building it
inside the disease scorer would re-import SP into disease ranking, which
statement 4a forbids.

**An earlier revision called the mechanics "nearly free" (F7). That was the wrong
frame.** The existing cosine and SP primitives make the final *arithmetic*
reusable; they do not make the clinical task or the training problem nearly free.
The work item's opening gates are where the real content is:

| Gate | Why |
|---|---|
| The corrected F4 finding | No active gene-targeted ranking objective |
| **Gradient-path audit** | What actually reaches gene representations — and it must read a built KG artifact, not just the model code |
| Gene-targeted supervision / retraining decision | Follows from the audit |
| Gene candidate and negative-set semantics | Undefined today |
| Variant→gene annotation and mapping | Required before any variant input can be scored |
| Gene-level ground truth and evaluation cohort | None exists in this repository |
| Paper Eq 13/14, η and normalisation verification | The paper's η is in the authors' repository, not the article |
| Task-specific score semantics and result type | Neither exists (F3) |
| Checkpoint compatibility | The deployed checkpoint has no task head |

**It must not begin by wiring the current gene embeddings into a clinical
scorer**, and it must not enter B-0 — B-0 measures the disease scorer. Whether to
schedule it at all is an institutional roadmap decision, not this document's.

---

## 4. Not asked here

- Whether to build the gene task **now** — roadmap and institutional priority.
- Whether to adopt the paper's Eq 18 for disease scoring — blocked on the
  retraining track (F8), not a scorer-selection question.
- Whether SP may return to disease ranking in a bounded setting — that is
  `DISEASE_SCORER_POLICY.md` §4's deferred cascade option, and it needs Q4's
  analysis first.

## 5. What this changes in B-0

**The implemented A/B/C work: nothing.** The legacy-removal checklist is
unamended — review accepted Q2 as written. B-0.4 proceeds as planned.

**B-0.5 gains one output-contract requirement**, from Q4: if the offline SP
analysis is to be claimed as available, the institutional run must persist the
per-candidate component rows listed there, under a bound that is not a top-N
window. **That is an output contract, not a new mode, not a new stage and not a
second institutional run.**

**Authority above everything here:** `docs/DISEASE_SCORER_POLICY.md`.
