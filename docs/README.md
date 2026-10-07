# Documentation index

Every Markdown document tracked in this repository — those under `docs/` and the ones deliberately
kept elsewhere — each with a status label. Two things are deliberately absent: this index itself,
and the images under `docs/images/`, which are embedded by the documents that use them.

Completeness is checked by `tests/unit/test_docs_index.py`, so a document added without an entry
here fails `make check` rather than quietly going missing.

**Status labels**

| Label | Meaning |
|---|---|
| **Living** | Kept current with the repository. If it disagrees with the code, that is a bug in the document. |
| **Dated snapshot** | Accurate as of its date; deliberately *not* updated. Read it as history, not as a description of the current tree. |
| **Archived** | Superseded. Kept for provenance under `docs/archive/`. |
| **Working** | Design and review material for work in progress, under `docs/working/`. Not written for clinicians or the engineering team, and not authoritative: when the phase lands, its decisions move into a living document. |

Snapshots are not corrected when the code moves on — the repository's rule is *correct living
documents, annotate dated snapshots*. That is also why nothing here has been relocated: several
snapshots cite each other by path, and moving a file would either break those links or require
editing history to match the present.

## Living documents

| Document | Contents |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Layered design, the η scoring model, and the design principles the system must satisfy |
| [`CONFIG_AUTHORITY.md`](CONFIG_AUTHORITY.md) | Decision record: which module owns which configuration, and why there is no central config manager |
| [`DIRECTORY_STRUCTURE.md`](DIRECTORY_STRUCTURE.md) | Where artifacts live and why KG-derived outputs are separated from KG-independent models |
| [`TRAINING_PIPELINE_PLAYBOOK.md`](TRAINING_PIPELINE_PLAYBOOK.md) | End-to-end build walkthrough: data sources, the four build steps, expected artifacts |
| [`GNN_ARCHITECTURE_NOTES.md`](GNN_ARCHITECTURE_NOTES.md) | Model design notes — conv types, fusion, head configuration |
| [`module_dependencies.md`](module_dependencies.md) | Inter-module dependency map. The layer rules it describes are enforced by `.import-linter.ini` (`make lint-imports`) |
| [`RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md`](RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md) | Open architecture findings under review: candidate discovery, retrieval, and the vector-index subsystem. Claims are individually labelled FACT / MEASURED / INFERENCE / OPEN |
| [`DISEASE_SCORER_POLICY.md`](DISEASE_SCORER_POLICY.md) | Decision record: what ranks disease candidates, what role the shortest-path signal may play, the evidence for it, and a table separating current behaviour from the approved target |
| [`SP_SCORE_GUIDE.md`](SP_SCORE_GUIDE.md) | The shortest-path score explained for clinicians (§1 meaning, §2 use) and for developers (§3 formulas, code, and where this implementation diverges from the reference paper) |

## Dated snapshots

Still referenced, and useful — but each describes the tree as it was on its date.

| Document | Date | Contents |
|---|---|---|
| [`MILESTONE_REPORT.md`](MILESTONE_REPORT.md) | rolling | Development progress and planned capabilities. Cited by `TRAINING_PIPELINE_PLAYBOOK.md` (SP memory) and by the retrieval findings (Patients-Like-Me status) |
| [`HANDOFF_SESSION_2026-02-23.md`](HANDOFF_SESSION_2026-02-23.md) | 2026-02-23 | Session handoff — dashboard/Gradio specifications |
| [`MODULE_SCAN_REPORT_2026-01-26.md`](MODULE_SCAN_REPORT_2026-01-26.md) | 2026-01-26 | Full module inventory (86 files at the time) |
| [`TRAINING_MODULE_AUDIT_2026-01-25.md`](TRAINING_MODULE_AUDIT_2026-01-25.md) | 2026-01-25 | Training module audit; identifies coupling issues, some since addressed |
| [`TORCH_COMPILE_EXPERIMENT_FINDINGS.md`](TORCH_COMPILE_EXPERIMENT_FINDINGS.md) | closed | Why `torch.compile` was evaluated and what was concluded. Self-marked 已封存 |
| [`Repair/REPAIR_CHECKLIST.md`](Repair/REPAIR_CHECKLIST.md) | rolling | Repair checklist; unchecked boxes are proposals, not commitments |
| [`Repair/SCAN_REPORT.md`](Repair/SCAN_REPORT.md) | 2026-07-22 | Repository scan that cross-checks the other snapshots against the tree |
| [`BRANCH_AUDIT_2026-10-02.md`](BRANCH_AUDIT_2026-10-02.md) | 2026-10-02 | Fifteen remote branches checked against `main` by at least two independent methods. Twelve were fully merged and only looked unmerged in a shallow clone; three hold commits outside `main`, each superseded or already present. All fifteen are marked pending deletion, and nothing is deleted |

## Working documents

Under `docs/working/`, one subfolder per work phase. These say things a finished document should
not — open questions, rejected alternatives, disagreements between author and reviewer — which is
why they are separated by folder from everything above. A phase folder that is still here after its
work shipped is stale, not authoritative.

| Document | Contents |
|---|---|
| [`working/README.md`](working/README.md) | The convention, and the list of live phases |
| [`working/BACKLOG.md`](working/BACKLOG.md) | The one ordered list across all live phases: what is open, in what order, what blocks what, and the measured facts that set that order. Ordering and dependencies only — the decisions themselves stay in the phase folders |
| [`working/PLAN_CONFIGURABILITY_AND_PROVENANCE.md`](working/PLAN_CONFIGURABILITY_AND_PROVENANCE.md) | **Approved for implementation.** Configurability as a stated requirement — the system is a clinical model *and* a research framework — and the two places existing code works against it: a measurement refusal that could be a record, and a checkpoint whose provenance metadata cannot identify its training inputs. Proposals A and B, with what each explicitly does not do |
| [`working/EVALUATION_COHORTS.md`](working/EVALUATION_COHORTS.md) | What the original team actually did about train/validation/test — read from their repository, not recalled — what of it is obtainable, and the three-cohort division of labour: MyGene2 for research comparison, a disease-disjoint synthetic split for unseen-disease generalisation, the institutional cohort as the acceptance benchmark. Includes what a fifteen-case cohort can and cannot decide |
| [`working/results-review/README.md`](working/results-review/README.md) | Results-review phase: what the specs cover, and how to run the closure audit |
| [`working/results-review/SPEC_0_INDEX.md`](working/results-review/SPEC_0_INDEX.md) | Index, requirements traceability, gates, open institutional values, backlog |
| [`working/results-review/SPEC_1_RESULTS_REVIEW.md`](working/results-review/SPEC_1_RESULTS_REVIEW.md) | Scorer/view authority, SP sort and filter policy, decomposition, two-surface UX, limits |
| [`working/results-review/SPEC_2_SNAPSHOT_REPOSITORY.md`](working/results-review/SPEC_2_SNAPSHOT_REPOSITORY.md) | Snapshot payload, rotation, retention, atomic publication, access decision. §11 carries the normative invariants and reopen triggers |
| [`working/results-review/SPEC_3_EVIDENCE_AND_AUDIT.md`](working/results-review/SPEC_3_EVIDENCE_AND_AUDIT.md) | Constraints any Gate 3 evidence design must satisfy |
| [`working/results-review/SPEC_4_DEPLOYMENT_SECURITY.md`](working/results-review/SPEC_4_DEPLOYMENT_SECURITY.md) | Route inventory, bind modes, authentication, CORS, risk acceptance |
| [`working/results-review/ARCHIVE_rev1_6.md`](working/results-review/ARCHIVE_rev1_6.md) | Revisions 1–6 of the above, with the withdrawn claims and what the scope audit removed. History, not authority |
| [`working/scorer-measurement/README.md`](working/scorer-measurement/README.md) | Scorer-measurement phase (work item B-0): the A/B/C/D mode ladder and the stage map |
| [`working/scorer-measurement/PLAN_B03.md`](working/scorer-measurement/PLAN_B03.md) | B-0.3 proposal — Modes B and C, and the three decisions that keep the ladder interpretable |
| [`working/scorer-measurement/PLAN_B04.md`](working/scorer-measurement/PLAN_B04.md) | B-0.4 proposal — vectorising the shortest-path primitive's body, why the production caller is deferred to B-1, and the correction to what B-0.4 was previously thought to be |
| [`working/PLAN_SP_ARTIFACT_INTEGRITY.md`](working/PLAN_SP_ARTIFACT_INTEGRITY.md) | The two open shortest-path producer defects — a tensor and a sidecar with nothing binding them, and an artifact that can be written into a workspace it was not computed from. **Approved and implemented** (backlog 5b) |
| [`working/PLAN_ONTOLOGY_PROVENANCE.md`](working/PLAN_ONTOLOGY_PROVENANCE.md) | Ontology provenance and selection — what the loader does not record, why selection belongs to the build rather than to configuration, and the proposed phasing. **Phases 0 and 1 implemented**; Phase 2 is planned separately below, Phase 3 unstarted |
| [`working/PLAN_ONTOLOGY_PHASE2.md`](working/PLAN_ONTOLOGY_PHASE2.md) | Ontology Phase 2 — a path per ontology on the build CLI, a resolver over configured roots, and a narrow imports policy that refuses rather than silently suppresses. Carries the parent plan's unverified imports measurement, now taken, and revises its §3.3 so the curated source list is operator-editable. **Approved and implemented**; the UI slot of §3.6.1 is recorded as owed and undelivered |
| [`working/PLAN_PROVENANCE_CONTRACT.md`](working/PLAN_PROVENANCE_CONTRACT.md) | Backlog item 15 — the provenance contract: every link from source files to the served model and its test results stated as a checkable relation, what is checked today (inside a workspace) and what is not (every link with the model on one side), the owner's decisions (a model whose training graph does not match is refused; old-pipeline artifacts unsupported), and milestones M1–M5 in dependency order, starting with B-2's fail-closed core. Revision 2 binds every recorded digest to the bytes actually parsed, checks the resume parent before anything is restored, and defines R10 (disease and phenotype-set overlap with the model's recorded training and validation inputs; *unverifiable* never shown as zero). Resume continues the same data in the first version (the owner's decision, §8 question 3). No accounts, keys, signing or chain. **Draft revision 2; not implemented** |
| [`working/PLAN_PHENOTYPE_NORMALISATION.md`](working/PLAN_PHENOTYPE_NORMALISATION.md) | Backlog item 16 — one rule for a case's phenotype list, from import to scoring. Traces what each entry point does today (generator, training and measurement readers, WebUI, API, direct pipeline): repeats are removed by string in the UI only, the pipeline truncates to 100 before dropping unknowns and with repeats counted, and `InputValidator` has no production caller. Corrects the evidence level of the research (the original SHEPHERD keeps repeats; Phenopacket does not require unique terms). Proposes removing repeats by graph node after mapping, refusing rather than truncating, producers normalising and readers checking, with the rule version recorded. Revision 2 moves the API's own summary onto the result, adds a length contract for per-position fields, and records the owner's decisions: the rule's principle adopted, `InputValidator` to be removed, and the WebUI no longer removing repeats, with a clear message built from the API's structured 422 when a list is over the limit. **Draft revision 2, amended; N1 implemented, code-reviewed and accepted on the homelab GPU; N2 not started** |
| [`working/PLAN_TEST_RESULTS.md`](working/PLAN_TEST_RESULTS.md) | Backlog item 14 — test results on MyGene2 and the institutional cohort, written by the pipeline beside the checkpoint and shown where a model is chosen. What exists, adjacent defects found while planning (F1–F4), and the decisions (D1–D8) before any code; D3 is settled by the scorer policy (the model's own disease ranking, Mode C), one path per concern is a stated rule, and three owner questions remain (the fourth, existing ledgers, is answered by the contract's decision 2). Revision 4 splits it into two phases, phase 1 a complete CLI flow resting on the provenance contract; its 2026-10-06 amendment records R10 as the contract defines it. **Draft revision 4; item 14 not implemented** |
| [`working/scorer-measurement/PLAN_B04_PRODUCTIONISATION.md`](working/scorer-measurement/PLAN_B04_PRODUCTIONISATION.md) | Backlog item 5a — how B-0.4's selected index reaches production, the decisions it asks a reviewer to make, and which of PLAN_B04 §13's readings this hardware can produce. **Revised to one served implementation and cleared to implement; not accepted for deployment** — §13's gate is unchanged: readings 1-4 are taken on the designated subject (§7.4), reading 5 is blocked, and no capacity decision is made |
| [`working/scorer-measurement/PLAN_B02_shipped.md`](working/scorer-measurement/PLAN_B02_shipped.md) | The plan the shipped B-0.2 harness was built from. History, not authority |
| [`working/task-scope/README.md`](working/task-scope/README.md) | Five scope questions raised by the institution's supplied-candidate-list use case: the reserved `candidate_genes` interface, legacy removal, the supplied-universe request/result variant, where the SP ablation belongs, and causal-gene scoring as its own work item. **Scope decisions reviewed; the reserved-interface item is implemented, the rest unscheduled** |
| [`working/scorer-retraining/README.md`](working/scorer-retraining/README.md) | Scoping for the scorer-retraining phase: the scorer-bundle unit of comparison, the experiment order, the versioned checkpoint scorer schema and its inference boundary, and the four kinds of legacy checkpoint. Nothing scheduled, no gate cleared |

## Archived

Superseded, kept for provenance. Links inside these files point at paths as they were written and
are deliberately left alone.

| Document | Contents |
|---|---|
| [`archive/ARCHITECTURE_REVIEW_2026-02-25.md`](archive/ARCHITECTURE_REVIEW_2026-02-25.md) | Systematic architecture review — superseded by `ARCHITECTURE.md` |
| [`archive/ENGINEERING_PROGRESS_REPORT_2026-02.md`](archive/ENGINEERING_PROGRESS_REPORT_2026-02.md) | Engineering progress report, 2026-02 |
| [`archive/HANDOFF_SESSION_2026-02-21.md`](archive/HANDOFF_SESSION_2026-02-21.md) | Session handoff, superseded two days later |
| [`archive/PROGRESS_2026-01-20.md`](archive/PROGRESS_2026-01-20.md) | Development progress summary, 2026-01-20 |
| [`archive/SESSION_HANDOFF.md`](archive/SESSION_HANDOFF.md) | Undated session handoff |
| [`archive/data_structure_and_validation_v3.md`](archive/data_structure_and_validation_v3.md) | Data-structure and validation design v3.0 |

## Documents kept outside `docs/`

Deliberate — each sits next to what it describes, and several are referenced by path from scripts
and other documents.

| Document | Why it lives there |
|---|---|
| [`../README.md`](../README.md) | Repository entry point |
| [`../medical-kg-blueprint.md`](../medical-kg-blueprint.md) | Project-level engineering blueprint; referenced from the repository root |
| [`../medical-kg-todo.md`](../medical-kg-todo.md) | Project-level task list |
| [`../deployment-guide.md`](../deployment-guide.md) | Deployment guide; referenced by `deploy.sh` |
| [`../deployment-guide.en.md`](../deployment-guide.en.md) | English edition of the deployment guide; the Chinese original is authoritative for `deploy.cmd`'s pointer |
| [`../data/external/README.md`](../data/external/README.md) | How to obtain and place external data sources — belongs with the data |
| [`../configs/deployment/README.md`](../configs/deployment/README.md) | Deployment config conventions |
| [`../models/pretrained/README.md`](../models/pretrained/README.md) | What belongs in the pretrained-model directory |

## Validation and diagnostic scripts

These are not part of `make check` and nothing runs them automatically. They are recorded here so
that "no automated caller" is not mistaken for "unused".

| Script | What it answers | When to run |
|---|---|---|
| `scripts/spikes/validate_fast_subgraph.py` | How much faster is `SubgraphSampler._build_subgraph`'s vectorized path than the legacy Python loop, on a real workspace? Also re-checks equivalence there. | When the speedup on real data matters. Correctness is covered by `tests/unit/test_subgraph_equivalence.py` in `make check`; this script measures, on data the test does not have. |
| `scripts/post_install_verify.py` | What CUDA driver and toolkit does the *host* have? Runs `nvidia-smi` and `nvcc --version`, prints JSON. | Diagnosing a mismatched or doubled CUDA install. `scripts/validate_installation.py` — the one `deploy.sh` runs — only reports the in-process view (`torch.version.cuda`), so it cannot see this. |
| `scripts/debug_voyager_windows.py` | Why does the Voyager backend misbehave on a given host? | Only while the vector-index subsystem is under review; see the retrieval findings document. |
