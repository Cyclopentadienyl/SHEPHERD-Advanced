# Task scope — what the supplied-short-list scenario changes

**Status:** open questions, none decided. This document exists to be reviewed.

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
| F2 | `candidate_genes` is declared in the request model, validated, and stored on `PatientInput` — and **read by no scoring path**. Five occurrences repository-wide; none in `src/inference/pipeline.py` | `api/routes/diagnose.py:63,216`; `inference/input_validator.py:437,466`; `core/types.py:388` |
| F3 | `ShepherdGNN` constructs **no task head**, and the deployed checkpoint passes a strict `load_state_dict`, so it carries no task-head parameters | `DISEASE_SCORER_POLICY.md:3.2, 3.3` |
| F4 | **Gene embeddings are unsupervised.** The link-prediction and ortholog losses are gated on `positive_triples` / `negative_triples` / `ortholog_pairs`, and **no dataset, collate function or trainer in `src/` produces those keys** — the three names occur only inside `loss_functions.py`. Patient and disease embeddings *are* supervised, by the diagnosis and contrastive losses | `training/loss_functions.py:513-584`; absence verified across `src/` |
| F5 | `DiagnosisSample.gene_ids` is collated into batches and read at `data_loader.py:929-930`, but **no loss consumes it** | `kg/data_loader.py:614,675,741-765,929` |
| F6 | A **dormant supplied-candidate mechanism already exists** in the training dataloader: `DiagnosisSample.candidate_disease_ids` plus `include_all_candidates`. `file_storage.read_samples` does not read the field, so it is dead on the measurement path | `kg/data_loader.py:613,656-669`; `kg/storage/file_storage.py:72-79` |
| F7 | `cosine_score_matrix(patient_matrix, candidate_matrix)` is **candidate-agnostic**, and `sp_mean_distances` already takes a sequence of targets | `inference/scoring.py:193,240` |
| F8 | The approved disease score is **raw cosine, explicitly interim**, because it reflects what the deployed checkpoint's training objective optimised — not the paper's Eq 18. Changing it requires the paper-parity retraining track, which is outside work item B | `DISEASE_SCORER_POLICY.md:1` statement 2 |

**F4 is the load-bearing one.** It means a gene scorer built today would rank
genes by embeddings that no objective ever shaped for that task. That is the
same caveat statement 2 records for diseases, one degree worse.

---

## 3. Questions

### Q1 — Is a reserved-but-inert interface acceptable as it stands?

`candidate_genes` (F2) is a deliberate reservation, not an oversight. The
concern is narrower than "it does nothing": **nothing marks it as reserved.**
The field description reads *"Pre-selected candidate genes to consider"*, which
describes a working feature, and no test asserts the field is inert — so the day
the system leaves pre-alpha, nothing fails to point out that it still is.

**Position: keep the interface, mark it, and pin it with a test.** Amend the
description to say it is accepted and not yet acted on; add a test asserting
that two requests differing only in `candidate_genes` produce identical results.
That test is the honest form of a reserved slot: whoever implements the feature
must delete it deliberately, and until then the reservation cannot rot into a
silent no-op.

Rejected: returning 400. The institution asked for the reservation; refusing the
field would break the thing it was reserved for. Also rejected: implementing it
now — see Q5.

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

**Position: an optional input field. A mode switch should be refused.**

`DISEASE_SCORER_POLICY.md` §1 statement 4a makes the canonical candidate set and
rank order immutable once produced, and permits clinician control only as *view*
operations over that result. A scoring-mode selector would let a clinical action
change the canonical result, which is what 4a exists to prevent.

The input-field form avoids this entirely: the clinician supplies **clinical
information** (a candidate list), not an **algorithm choice**. One scorer, one
formula, a narrower candidate universe, and the result type records which
universe was used. It also matches the paper's shape — an externally supplied
list scored by the same similarity function.

Note this closes a gap `DISEASE_SCORER_POLICY.md` §5 identified explicitly: the
current top-k analysis set resembles the paper's short list *in size only*,
because ours is model-ranked while the paper's is externally supplied. A
clinician's list is externally supplied.

### Q4 — Is an SP ablation a new measurement stage?

**Position: no. It is offline analysis over what B-0 already records.**

`DISEASE_SCORER_POLICY.md` §4 states the cascade alternative "can be evaluated
offline from B-0's recorded per-candidate score components without a further
run", and §8 condition 4 repeats it. No new stage, no new mode.

Two limits to state with any result:
- §8 condition 1: the comparison tests **this repository's** SP transform
  `1/(1+mean(d))`, not the paper's Eq 13. A result in either direction is a
  statement about this implementation.
- Without B-0.5's statistical protocol, "better in this scenario" is not
  separable from noise, and per-scenario subgroup analysis multiplies that risk.

### Q5 — Should causal-gene scoring become its own work item?

**Position: yes, as a named item — and its scope must open with F4.**

No work item currently covers it. The Phase-1 paper-parity list in
`docs/RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md` §6.B has four entries and
gene scoring is not among them; `DISEASE_SCORER_POLICY.md` §4 calls candidate-gene
scoring "unbuilt future work". Meanwhile the institution has reserved an
interface for it. That mismatch is the finding.

Arguments for a **separate** item rather than an extension of the disease path:

- It is the **only** task where the paper's SP fusion is parity rather than
  deviation (F1). Building it inside the disease scorer would re-import SP into
  disease ranking, which statement 4a forbids.
- The mechanics are nearly free (F7): gene embeddings exist as a KG node type,
  `cosine_score_matrix` is candidate-agnostic, `sp_mean_distances` is already
  batched over targets.
- What is **not** free, and must be in the item's scope rather than discovered
  during it: F4 (nothing supervises gene embeddings), F3 (no task head), the
  paper's η for genes is published in the authors' repository rather than the
  article, and the result type and score semantics do not exist.

**It must not enter B-0.** B-0 measures the disease scorer. Sequencing relative
to B-1 is a roadmap decision for the institution, not for this document.

---

## 4. Not asked here

- Whether to build the gene task **now** — roadmap and institutional priority.
- Whether to adopt the paper's Eq 18 for disease scoring — blocked on the
  retraining track (F8), not a scorer-selection question.
- Whether SP may return to disease ranking in a bounded setting — that is
  `DISEASE_SCORER_POLICY.md` §4's deferred cascade option, and it needs Q4's
  analysis first.

## 5. What this changes in B-0

**Nothing.** B-0.4 and B-0.5 proceed as planned, and the legacy-removal
checklist is unamended. If review disagrees with Q2, that is the one answer that
would reach into B-0.

**Authority above everything here:** `docs/DISEASE_SCORER_POLICY.md`.
