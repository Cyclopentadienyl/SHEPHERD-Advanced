# SHEPHERD-Advanced System Architecture

> **Last updated**: 2026-11-12, at `main` `30e8ae1` (the research-comparison round merged, #133;
> product code `e60f23e`).
> This file describes what runs on `main`.
> **How to read it**: each section says whether it describes **what runs** or an
> **approved target that is not implemented yet**. The disease scorer policy's authority is
> [`DISEASE_SCORER_POLICY.md`](DISEASE_SCORER_POLICY.md). §11 maps each backend capability to its formal
> entry point and WebUI page; §12 holds the rules that keep one implementation per function.
> **Anchor**: SHEPHERD (npj Digital Medicine 2025) by Harvard MIMS Lab.
> **What is next:** [`working/BACKLOG.md`](working/BACKLOG.md) §0. **What is open**, each with who
> decides: [`working/scorer-retraining/README.md`](working/scorer-retraining/README.md), "Who decided
> what", "Not decided here, and open"; the policy's own list is
> [`DISEASE_SCORER_POLICY.md`](DISEASE_SCORER_POLICY.md) §2.1, "Who decides what".
> **History.** The dated notes below are kept as written; K1–K8's records are
> `working/scorer-retraining/README.md` §8. Until 2026-11-09 this file was last synced at `main`
> `416a242` (2026-10-25, M5 merged, #129); until 2026-11-12 at `db8ee5a` (K0–K8, #131).
> **2026-10-27: the target formulation changed.** The paper-aligned two-stage design
> ([`working/scorer-retraining/README.md`](working/scorer-retraining/README.md)) replaces the cosine
> flow of §2.2 and the methods of §3.1–§3.3. What runs (§8, §11) is unchanged. B-1 as a raw-cosine
> ranking is not built first (policy rev 11).
> **Revision 2 of that design (2026-10-27):** one switch when the integration is delivered (the
> served disease mixture, the path-reasoning fallback ranking, Mode C's raw-cosine recipe and Mode D
> retire); the Mode C/D comparison is archived; the system has never been deployed at the hospital,
> so merging into `main` is an engineering version, not a clinical switch (policy rev 12).
> **K7 (2026-10-29, on `claude/paper-aligned-two-stage`):** the path-reasoning fallback, the demo
> answer, the in-memory model and graph seams and Mode D are removed; with no model loaded nothing
> is served (503). The served mixture, Mode C and the ShepherdGNN path stay only for checkpoints
> without a model spec (history files), until K8; the GB10 acceptance probe left them in K8-1.
> **K8-4 (2026-11-03):** the served mixture is removed and serving refuses a checkpoint without a
> model spec at load, before any model is built; Mode C, which still ran such a file, left the
> command line in K8-6a and its executor went in K8-6b. §8 describes what is served now.
> **K8 closeout (2026-11-08):** the ShepherdGNN training path is removed (K8-7: its loader, its
> `MultiTaskLoss`, its settings, `num_workers` and checkpoint record version 1), with the
> ShepherdGNN wrapper, its encoders and `OrthologGate` (K8-10), `RankingMetrics` (K8-8) and
> `src/inference/scoring.py` (K8-11). Every run names its learning stage; a checkpoint without a
> model spec is listed as a record this build cannot read, with the reason "written before the
> model spec: not readable by this build; start a new run", never as history (K8-7b).
> The §1 "Running" block and the rows marked K7, K8-4 or K8 below say what runs on the branch; the rest
> of this file still describes `416a242` where it says so.
> **2026-11-09:** K8's GB10 functional acceptance at `f2c2012` passed
> (`working/k8-acceptance-2026-11-09/`), and K0–K8 merged into `main` as #131 (`db8ee5a`). This
> file now describes `main`; the sentence above about the branch and `416a242` no longer holds.
> **2026-11-12:** the research-comparison round's GB10 acceptance at `e60f23e` passed on both deploy
> routes (`working/research-comparison-acceptance-2026-11-12/`), and the round merged into `main` as #133
> (`30e8ae1`); its plan and record is `working/research-comparison/README.md`.

---

## 1. System Identity

SHEPHERD-Advanced is a rare disease diagnosis system that:
- Uses a heterogeneous knowledge graph (PrimeKG-based) and GNN to generate node embeddings
- A KG link-prediction base model, then one model finetuned from it per task, each ranking with
  its own task head (designed 2026-10-27, built in K1–K8 and merged into `main` as #131;
  [`working/scorer-retraining/README.md`](working/scorer-retraining/README.md))
- Provides explainable evidence paths via PathReasoner (auxiliary)
- Supports three diagnostic tasks in one framework: they share the base, the infrastructure and the
  records, not a final vector space

**Running on `main` (since #131, `db8ee5a`, and #133, `30e8ae1`):** a base trained by KG link
prediction, then task models finetuned from it (disease, gene, patients-like-me), one served at a
time on the Diagnosis page and `POST /api/v1/diagnose`: a disease model ranks every disease of its
graph view by its head, a gene model the request's candidate genes, a patients-like-me model a
registered reference cohort's patients. Evidence paths explain disease rows after ranking; the
served identity is on every result; with no model loaded nothing is served (503). The research tools
around it: workspaces; base training and Fine-tune, with the frozen-backbone disease control
(`frozen_parts` `graph_side`) from the Fine-tune page, the command line and the job runner; data
versions by kind (supplied cohorts, KG edge sets, kept generated splits) with their roles; measurement by a protocol
this build implements that the model's capabilities admit, the common KG link-prediction protocol
among them; registration by data kind; one read-only results query with Compare and export (§11). A
checkpoint without a model spec (a ShepherdGNN history file) is listed as a record this build cannot
read (since K8-7b; "history" before), refused at load by serving (since K8-4), refused by
measurement (since K8-6a, when Mode C left the command line), and never trained from (K8-7 removed
the path that wrote such files).
*(History: `main` at `416a242`, 2026-10-25, ranked diseases by the interim served mixture of §8,
with the research tools above except the task models; the mixture retired in K8-4.)*

**Hardware**: CUDA GPUs (the GB10 is the accepted target; BACKLOG 24 for others). Library versions:
PyTorch, `torch_geometric` (PyG's core) and the rest of the locked set in `pyproject.toml` and
`uv.lock`; PyG's native extensions per platform (§9, BACKLOG 50).
**Graph layers**: a model bundle's encoder is its model spec's, behind the `GraphEncoder` contract
(`src/models/bundle/encoders.py`). The builder builds one kind today, `gatv2`, the reference: the
paper's homogeneous GATv2 over a learned node table (`working/scorer-retraining/README.md` §2–§3).
What a kind offers is its declaration (`src/models/bundle/spec.py` `ENCODER_DECLARATIONS`: `gatv2`
only) and the capability description derived from it (`src/models/bundle/capabilities.py`).
`hetero_gat`, `hgt` and `sage` are names the spec reads (`src/config/model_types.py` `ENCODER_KINDS`)
with no declaration: a run of one is refused at its start, and the builder refuses it
(`src/models/bundle/builder.py` `_build_encoder`). None of them is supported. A second architecture
is chosen by a fit assessment before any is ported (C0, then C1 for the one the owner chooses:
`working/ops-wiring-and-ui-ux/README.md` §10, where HGT is one candidate, not presumed the newest);
`HeteroGNNLayer` (`src/models/gnn/layers.py`, reached only by tests) is kept and reused where it fits.
*(The ShepherdGNN path's GAT, HGT and SAGE, `--conv-type`, left with its training path in K8-7 and
its wrapper in K8-10, 2026-11-08.)*
**Extension (not implemented)**: fuzzy input, free text to HPO/MONDO (`src/nlp/`).

---

## 2. Core Architectural Principle: GNN-Primary (running since K8-4)

> **Status (2026-11-09, `main` `db8ee5a`).** A disease task model ranks every disease of its graph
> view by its disease head, with no shortest-path term and no gate by path discovery; paths explain
> the rows shown, after ranking (§8). §2.1's rule holds on the research platform. The disease scorer
> policy — what ranks disease candidates, the role of the shortest-path signal, and the role of path
> reasoning — is owned by [`DISEASE_SCORER_POLICY.md`](DISEASE_SCORER_POLICY.md) (rev 18, a statement
> of fact), which keeps clinical adoption apart from the research platform (rev 12). Clinical
> adoption is open, owner first and then the institution; until a revision is approved, statement 2
> and gate 1 stand as written and bind the clinical switch (rev 11; the header's "What is open").
>
> **Status note (2026-08; still true at `416a242`).** The disease scorer policy — what ranks disease candidates, the role of
> the shortest-path signal, and the role of path reasoning — is owned by
> [`DISEASE_SCORER_POLICY.md`](DISEASE_SCORER_POLICY.md), which also records **which parts of this
> section the runtime does not yet implement**. In particular, candidate discovery is currently
> gated by path search and the ranking score currently includes a shortest-path term; §2.1 below
> describes the approved target, not present behaviour; §8 describes what runs. This section will be
> reconciled with the implementation when work item B-1 lands.
> **2026-10-27:** the approved target's ranking model becomes the paper-aligned disease head (work
> package A, [`working/scorer-retraining/README.md`](working/scorer-retraining/README.md)). §2.1's rule
> stands: the GNN ranks, paths explain. §2.2 is redrawn for it. Serving switches only through the
> policy's gates.
> *(2026-11-09: the two notes above are history; the status at the top of this section holds.)*

### 2.1 The Non-Negotiable Rule

> **GNN generates embeddings and drives all scoring/ranking.**
> **PathReasoner provides explainable evidence paths for clinician review.**
> **PathReasoner NEVER gates candidate discovery or drives scoring when GNN is available.**

This is the fundamental design principle derived from the original SHEPHERD paper. The GNN's key advantage over traditional path-based methods is **inductive generalization** — the ability to:
- Infer relationships for diseases never seen in training
- Handle missing edges in the knowledge graph via learned neighborhood structure
- Automatically learn which edge types and multi-hop signals are diagnostically relevant
- Generalize to novel phenotype combinations

If PathReasoner becomes the gatekeeper for candidate discovery, GNN's generalization ability is blocked.

### 2.2 Correct Flow (the paper-aligned disease task, running since K8-4)

```
Patient Phenotypes (HPO terms)
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Disease model (finetuned from a selected base)   │
│  GraphEncoder forward: full graph, layer-wise,    │
│  once at load; node vectors cached                │
└──────────────────┬───────────────────────────────┘
                   │
    ┌──────────────▼───────────────────────┐
    │  Patient encoder                      │
    │  Transformer over phenotype vectors   │
    │  → attention pooling → projection     │
    └──────────────┬───────────────────────┘
                   │
    ┌──────────────▼───────────────────────┐
    │  Disease head                         │
    │  −‖z_d − z_P‖² (normalised vectors)   │
    │  over EVERY KG disease, exact         │
    └──────────────┬───────────────────────┘
                   │
    ┌──────────────▼───────────────────────┐
    │  Top-K results                        │
    │  + PathReasoner evidence paths        │
    │  + attention weights                  │
    └──────────────────────────────────────┘
```

The gene task and patients-like-me are separate finetuned models with their own heads (§3). They do
not share this model's final vectors. The recipe, its sources and its deviations are
`working/scorer-retraining/README.md` §2. *(2026-11-09: the attention weights in the last box are
not built; attention views are later work, that README §3.3, "Later", and no result carries
them.)*

### 2.3 What PathReasoner Does and Does NOT Do (running)

**What runs:** a task model follows the table. Since K8-4 paths are found after ranking, for the
rows shown, and never decide which diseases are scored; since K7 there is no path-reasoning
fallback ranking (`path_reasoning_fallback` removed): with no model, nothing is served. *(Until
K8-4, as at `416a242`, path search decided which diseases the served mixture scored: a disease with
no path from the patient's phenotypes was never a candidate, and the search was capped in traversal
order.)*

| PathReasoner DOES | PathReasoner Does NOT |
|---|---|
| Find BFS paths between phenotypes and top-K diseases for explanation | Gate which diseases can be scored |
| Provide human-readable evidence chains for clinicians | Contribute to the ranked score when GNN is active |
| Supply shortest path length as optional supplementary feature | Act as first-pass candidate filter |
| Explain a ranked disease row after ranking (no scoring fallback since K7) | Override GNN ranking |

---

## 3. Three Diagnostic Tasks (three task models, running)

**The three tasks share a base and a framework, not a final embedding space** (2026-10-27). Each
task is finetuned from a selected base into its own checkpoint, so vectors from two task models are
not comparable (`working/scorer-retraining/README.md` §1). **All three run on `main`** as finetuned
task models (disease K3, gene and patients-like-me K4; the Diagnosis inputs K6), served one at a
time on the Diagnosis page (§8) and each measured by its own protocol (`disease`, `gene`,
`patients_like_me`; task measurement, §11).

**The reference paper's three tasks and this project** (checked against the paper's Methods,
npj Digital Medicine 8:380, 2025, and `docs/working/FINDINGS_PAPER_PROTOCOL_AND_AMP.md` §3.4 and
§3.6; added 2026-10-26). The paper's auxiliary scores belong to their tasks. They are not a set of
scores to add to any ranking.

| Task | Inputs it needs | The paper's formula and model head | Here today (`main`) | This project's difference | Where it landed; what is open |
|---|---|---|---|---|---|
| Causal gene discovery | patient phenotypes and a candidate gene list from variant filtering or curation (13.3 genes on average in one setting, 244.3 in the other; policy §3.1) | Eq 12–14: `η·EMBSIM + (1−η)·SPLSIM`, its own task head, and SP also in the training loss | the gene task model ranks the `candidate_genes` list (K4), which Diagnosis sends for a gene model (K6); any other model ignores it with a warning; the `gene` protocol measures it | — | work package B, done (`docs/working/scorer-retraining/README.md` §8; its SP specification §6); trained and measured on CUDA at demo size only (a one-epoch child on the GB10 at `e60f23e`, L7, merged as #133), and no real-data gene cohort exists yet (BACKLOG 25); the only task in which the paper mixes SP into a ranking |
| Patients-like-me | patient phenotypes and a reference cohort of patients with embeddings | Eq 16: `−‖z_Pi − z_Pj‖²` | the patients-like-me task model ranks a registered reference cohort version's patients (K4; Diagnosis, K6); the `patients_like_me` protocol measures it | — | work package B, done; trained and measured on CUDA at demo size only (a one-epoch child on the GB10 at `e60f23e`, L7, merged as #133); a formal reference set is later work (`docs/working/scorer-retraining/README.md` §8, "K8 record") |
| Novel disease characterisation (disease ranking) | patient phenotypes | Eq 18: `−‖z_d − z_P‖²` over every KG disease, on the patient encoder's output (a transformer encoder and an attention-weighted average) | the disease task model serves Diagnosis (K3; the legacy mixture of §8 retired in K8-4); the `disease` protocol measures it (task measurement, §11) | the served model is the paper-aligned recipe, with the deviations `docs/working/scorer-retraining/README.md` §2 lists. The policy's approved interim statement 2, raw cosine of an **unweighted** masked mean, is not the paper's encoder, projection or objective; it stands as written and has no served implementation on the research platform (rev 12) *(Mode C measured it until K8-6a and K8-6b; cosine and negative squared L2 order alike only on the same unit vectors, which does not make the encoders equal, and Mode C was not a reproduction of the paper)* | work package A, done; served on the research platform since K8-4 (the one switch, policy rev 12). **Open, clinical adoption** (the owner, then the institution; package C): rev 11's proposal for statement 2 and B-1 and rev 12's for gate 1's B-0 prerequisite are not approved; until a revision is approved, statement 2 and gate 1 stand as written and bind the clinical switch (rev 11; the header's "What is open"). The raw-cosine B-1 is not built first (policy rev 11) |

**SP for disease ranking** is this project's policy, not the paper's formula. The policy makes it
an optional analysis view, off by default, whose sorting and filtering change the view only and
never the canonical rank (policy statement 4b, C1–C10); that view is not built (policy §2), and no
disease result carries an SP term. **The evidence metapaths** of gate 3's design are
likewise this project's evidence policy, not a method the paper specifies
(`docs/working/results-review/SPEC_3_EVIDENCE_AND_AUDIT.md` §7.2).

**Each task keeps its own validation, and only the machinery is shared** (registered 2026-10-27).
- **Causal gene discovery** needs gene-level ground truth and an evaluation cohort, neither of which
  exists here (`docs/working/task-scope/README.md`). Its SP mixture (Eq 14) is that task's formula.
- **Patients-like-me** needs a labelled reference cohort, against which the patients it returns can
  be checked.
- **Disease ranking** is measured by the disease protocol on registered cohorts (task measurement,
  §11; Mode C until K8-6a), then by the institutional run (BACKLOG 8a, 8b).

They may share loading, identity, records and presentation. None of them is the disease ranking's
GNN/SP mixture. An adjustable-η analysis view (policy §1.4, a registered direction, not decided) is a
view over the disease result, not a model head for these tasks.

*(2026-11-09: since K4 and K6 the `gene` protocol measures causal gene discovery on registered
cohort versions that list causal and candidate genes (`src/evaluation/protocols.py`); a gene child
was measured and registered in the K6 smoke on CPU (§11), and no real-data gene cohort exists yet
(BACKLOG 25). The `patients_like_me` protocol measures patients-like-me on a cohort version that
lists causal genes; no such measurement was run in the K6 smoke (§11). The disease GNN/SP mixture
retired in K8-4 (§8). BACKLOG 7b, 8a and 8b still name the archived Mode D and wait on the decision
on rev 12's proposal, the owner and then the institution; gate 1 stands as written.)*

### 3.1 Causal Gene Discovery (the gene task model, running)

A gene task model (K4) ranks the `candidate_genes` request field, which Diagnosis sends for a gene
model (K6); with any other model it is ignored and answered with a warning
(`src/api/routes/diagnose.py`).
- **Input**: Patient phenotypes (Channel 1) + Candidate gene list (Channel 2)
- **Method (work package B)**: `η·EMBSIM + (1−η)·SPLSIM` over the candidate list, from the gene task's finetuned model, η from its model spec and SPLSIM from exact hop distances computed in process (§8; `working/scorer-retraining/README.md` §2, §6)
- **Output**: Ranked gene list with scores
- **Channel 2 source**: Upstream variant filtering pipeline (e.g., Exomiser) — SHEPHERD is the consumer, not producer

### 3.2 Similar Patient Retrieval (Patients-Like-Me) (the patients-like-me task model, running)
- **Input**: Patient phenotypes only (Channel 1)
- **Method (work package B)**: `−‖z_Pi − z_Pj‖²` on normalised vectors against a registered reference cohort version embedded by the same model; the query itself excluded
- **Output**: K most similar confirmed patients with shared phenotypes
- **Clinical value**: Diagnosis validation, especially before ordering genetic testing

### 3.3 Novel Disease Characterization (running as Diagnosis' disease ranking)
- **Input**: Patient phenotypes only (Channel 1)
- **Method (work package A)**: `−‖z_d − z_P‖²` on normalised vectors between the patient encoder's output and every disease of the model's graph view, from the disease task's finetuned model, served since K8-4 (§8)
- **Output**: "Most similar known diseases" ranking
- **Clinical value**: When the patient may have an undescribed condition

---

## 4. Two-Channel Input Design

| Channel | Content | Format | Required For |
|---------|---------|--------|-------------|
| **Channel 1** (Phenotypes) | Observed symptoms/features | HPO term IDs | ALL tasks (always required) |
| **Channel 2** (Candidate Genes) | Sequencing-derived suspects | Ensembl Gene IDs | Causal gene discovery only |

Channel 2 is read by the gene task model only (§3.1). As designed, the two channels are processed independently
inside the model and merged only at scoring time:
- Channel 1 determines the patient embedding direction
- Channel 2 determines comparison targets
- Channel 2 does NOT alter the patient embedding

---

## 5. Training

**What runs (since K8-7, 2026-11-08).** Every run names its learning stage: a base trained by KG
link prediction (`src/training/lp_task.py`) or a task model finetuned from a base
(`src/training/finetune_task.py`), each with its task's own sampler, loss and validation.
`scripts/train_model.py` (also started by the job runner) trains it on a generated workspace. The
Trainer selects by the task's selection metric (a base's `val_lp_accuracy`, a task model's
`val_pooled_mrr`) unless a monitor is stated; there is no top-20 measure. Each run writes its own
run directory; a run can be paused, resumed with its learning settings inherited, or extended at a
fixed rate (M3a). The checkpoint payload records the producer, the environment, the settings
requested and in effect, the progress, and the digests of the graph, the sample files and the split
manifest it trained on; the manifest binds the dataset record's digest, which the checkpoint does not
record itself (M2, M3a, M3b). A model bundle's checkpoint also carries its model record: the model
spec, the graph view, the stage, the task and the parent (`working/PLAN_PROVENANCE_CONTRACT.md` §9).
*(Until K8-7: one joint multi-task run with one loss, `MultiTaskLoss` in
`src/training/loss_functions.py`, of which only the diagnosis and contrastive terms were computed,
BACKLOG 57; it went with the ShepherdGNN training path and its loader.)*

- **Validation** is the Trainer's own, by the stage's selection metric, used for early stopping and
  checkpoint selection; it is not a test. A base validates on the view's validation edges
  (`src/training/lp_task.py`); a task model on the workspace's validation samples, disease-disjoint
  by construction: the workspace builder allocates diseases to train or val before it generates
  samples.
- **No held-out test split of patient samples** is generated (the generator writes train and val),
  and no institutional holdout protocol is defined (BACKLOG item 11). A base's held-out KG edges are
  measured as a KG edge set's `test` role, external cohorts as registered versions (task measurement,
  §11).
- **Accepted, and not** (`working/scorer-retraining/README.md` §8, "K8 record";
  `working/PLAN_PROVENANCE_CONTRACT.md` §9):
  - the KG base stage and a disease finetune passed K8's GB10 functional acceptance at `f2c2012`
    (2026-11-09, `working/k8-acceptance-2026-11-09/`); one-epoch demo-sized disease, gene and
    patients-like-me children were trained and measured on CUDA at the research-comparison round's
    GB10 acceptance (`e60f23e`, L7, merged as #133; `working/research-comparison-acceptance-2026-11-12/`);
    on the real data only the disease child was trained. The frozen-backbone control is unit-tested on
    the CPU and checked in fp32 by the acceptance probe (`L7.disease_frozen`, on the CPU in the
    tests), with no CUDA record until the round's candidate-slot run. Other encoder kinds: §1;
  - *branch* training, a new run from a parent's weights with its settings kept or changed on its
    stage's closed list, is started from the CLI, the job runner and the API (K5; the ops-wiring
    round's A1b; BACKLOG 23), the Training page's Branch mode following in that round's B1, and has
    no GB10 record;
  - the losses the new specs replaced are listed retired in that README's §7.

### Training → Inference Bridge
1. Training writes checkpoints into its run directory.
2. Serving loads a checkpoint (named by `SHEPHERD_CHECKPOINT_PATH` at startup; chosen from the
   server's list by a WebUI Load; or, through the reload API, given by path or selected from the
   workspace's latest run). It refuses the checkpoint unless its recorded graph digests (`kg` and the
   three tensors) match the workspace it is served over, and that workspace's dataset record is the
   one its manifest binds; then it runs the GNN forward pass once and caches every node embedding.
   **On CUDA it holds the deployment's GPU lock** from before the checkpoint is read until an
   operator stops inference (Diagnosis's Stop inference, or `POST /api/v1/pipeline/stop`), so a
   training or test job is refused, naming inference, until an operator stops it; on the CPU it
   takes none (the ops-wiring round's A3a and A3b, `src/api/main.py` `AppState`).
3. **No model, no pipeline** (K7): with a graph configured (`SHEPHERD_KG_PATH`) and no model
   source, with nothing configured, or with a requested model that cannot be built, no pipeline
   is published; `/diagnose` and `/ready` answer 503 with the reason and `/health` stays 200, so a
   model can be loaded. The path-reasoning fallback and the demo answer (`demo/v1`) are removed
   (BACKLOG 55).

---

## 6. Module Architecture

```
Layer 0 (Foundation):  src/core/          types, protocols, schema
Layer 1 (Config):      src/config/        server settings, allocator presets, serving device,
                                          training field spec, model-type constants
Layer 2 (Data):        src/ontology/      HPO/MONDO/GO via pronto
                       src/data_sources/  ortholog, pubmed (Phase 2)
Layer 3 (Knowledge):   src/kg/            graph, builder, data_loader (`DiagnosisSample` only since K8-7)
                       src/nlp/           fuzzy input (NOT YET IMPLEMENTED)
Layer 4 (Models):      src/models/        bundle (spec, builder, encoders), task heads and scorer,
                                          GNN layers (kept for the port), attention
                       src/retrieval/     vector index (cuVS/Voyager)
Layer 5 (Reasoning):   src/reasoning/     path_reasoning (auxiliary), explanation
Layer 6 (Training):    src/training/      trainer, the stages' task modules (lp_task, finetune_task)
Layer 7 (Inference):   src/inference/      pipeline (GNN-primary scoring)
Layer 7b (Evaluation): src/evaluation/    measurement by protocol, test cohort registry, R10, evaluation ledger
Layer 8 (Interface):   src/api/           FastAPI REST endpoints; src/api/services: job runner,
                                          server lists, record states, backend control
                       src/webui/         Gradio dashboard, mounted at /ui in the same process
```

**Import rules** (enforced by import-linter):
- Lower layers cannot import higher layers; `src/evaluation` sits above `src/inference` and below
  the interface (`.import-linter.ini`)
- `src/webui` cannot directly import `src/training`. It calls the services in `src/api/services` in
  process (no HTTP hop), and the job runner there starts the formal CLIs (`train_model.py`,
  `measure_scorer.py`) as subprocesses under the one GPU lock

---

## 7. Key Implementation Files

| Component | File | Key Responsibility |
|-----------|------|-------------------|
| Model bundle | `src/models/bundle/` (`spec.py`, `builder.py`, `encoders.py`, `capabilities.py`, `record.py`) | The model spec and its encoder declarations, the one builder, the input module and the `GraphEncoder` contract with its reference `GATv2Encoder`; a model's capability description, which the protocols read; the model record and the one way to read it (the ShepherdGNN wrapper, `src/models/gnn/shepherd_gnn.py`, was removed in K8-10) |
| GNN Layers | `src/models/gnn/layers.py` | `HeteroGNNLayer`, kept and not behind the `GraphEncoder` contract, reached only by tests; reused if a chosen second architecture fits it (§1) (`OrthologGate` was removed in K8-10) |
| Task heads | `src/models/tasks/heads.py` | The patient encoder and the disease, gene and patients-like-me heads of a model bundle (the unwired `src/models/decoders/` heads were deleted in K7) |
| Stage tasks | `src/training/lp_task.py`, `src/training/finetune_task.py` | Each stage's sampler, loss and validation (the multi-task loss, `src/training/loss_functions.py`, was removed in K8-7) |
| Trainer | `src/training/trainer.py` | Training loop with mixed precision, driven by its stage's task |
| Inference Pipeline | `src/inference/pipeline.py` | The served Diagnosis: the task model's ranking (§8), evidence and explanations |
| PathReasoner | `src/reasoning/path_reasoning.py` | BFS path enumeration (`find_paths`, `score_paths`): the evidence for a disease row (the mixture's candidate discovery went with it in K8-4, §2.3). Its path-score ranking (`aggregate_path_scores`, `get_top_candidates`) was removed in K7 |
| Knowledge Graph | `src/kg/graph.py` | KG data structure with PyG export |
| Vector Index | `src/retrieval/vector_index.py` | Standalone ANN index. **Detached from diagnosis** — retained for planned vector-mapping work, not on any inference path |
| Measurement (Mode C) | retired: no command since K8-6a, no executor since K8-6b | Every disease ranked by the model's raw cosine, on a registered test cohort version or a generated split: the formal Test's measurement until K6, then a command-line run (`scripts/measure_scorer.py` without `--mode`) on a checkpoint without a model spec until K8-6a, which made `--mode` required (row below). Its executor retired in K8-6b; `src/evaluation/measurement.py` keeps the report types and the ranks rule every measurement writes by. Kept Mode C reports and records are read by the shared readers (policy rev 15) |
| Task measurement | `src/evaluation/protocols.py` (the protocols) and `src/evaluation/task_measurement.py` (their evaluators); CLI `scripts/measure_scorer.py --mode <protocol>` | A model measured by a protocol this build implements that its capabilities admit (today disease, gene and patients_like_me, each taking a model that carries its task head, on a supplied cohort version or a generated split, the workspace's own (`--split`) or one kept as a version (research comparison Stage 5b, gene and patients_like_me through the functional gene fixture the version keeps); and kg_link_prediction, taking a base or any finetune by its LP head, on a KG edge set: research comparison Stage 5a, its run keeping each pair's raw scores and its exposure block beside its report); what the Test page runs, with the protocol chosen among those the model's listed capabilities admit (K8-6a). Since research comparison Stage 3 a protocol states what it needs of a model and the data kinds it measures, and every consumer (the command line, the job runner, the Test page, the probe) judges the one capability description the listing carries (`src/models/bundle/capabilities.py`, derived from the model spec and its encoder's declaration). The protocol definition judges only whether a new measurement can run; no reader uses it. Mode D, the served ranking's measurement, was removed in K7 (`src/evaluation/mode_d.py` deleted); its kept reports and records stay readable |
| Test cohort registry | `src/evaluation/cohort_registry.py`; CLI `scripts/import_cohort.py` | Immutable data versions bound to one graph's digests, kept outside the repository (provenance contract M4), each of one kind: a supplied cohort mapped to the graph (schema 1, as written since M4), a KG edge set (schema 2, `--kind kg_edges`, research comparison Stage 4): one part (`train`, `val` or `test`, as given) of a checkpoint's recorded edge split, rebuilt and checked, with its fixed LP pairs kept as a digest-bound file; or a generated split (schema 2, `--kind generated`, Stage 4b): one of a workspace's own generated splits, verified against its split manifest, its samples and that split manifest copied as written, identified by their digest, the split manifest's and split files' digests, the generation definition and its role, and with the workspace's functional gene fixture of that split, kept the same way, when it holds one of its build (Stage 5b). One reader for all, `resolve_version`; every version names its dependencies by content, never by path (`version_references`, `check_references`) |
| R10 | `src/evaluation/r10.py` | A test cohort's disease and phenotype-set overlap with the training and validation inputs a checkpoint records, found by digest. A candidate or evidence directory that cannot be read or checked refuses nothing and the search goes on: a scope found nowhere else reads *unverifiable*, naming each such place and its reason, never a count (definition version 2, K8-6r.3). Whether a named evidence directory is one is the explicit stat `server_settings.is_directory`, shared with `SHEPHERD_TRAINING_EVIDENCE_DIRS` |
| Evaluation ledger | `src/evaluation/sidecar.py`, `src/evaluation/registration.py`; CLI `scripts/record_evaluation.py` | Measurement results beside the checkpoint, keyed by its digest (ledger schema 4 since K8-6r.2, which reads schema 3 too: each record names its run, and a repeat is the same run's same report); one registration function for `--record` and `record_evaluation`; one registration at a time per ledger, across processes, through its lock file (K8-6r.1). Since research comparison Stage 6 a record is built per data kind through one function (`build_record`): a supplied cohort's as before, a KG edge set's with its own cohort block, the run's counts and metric statuses and its exposure block, a generated split's with its role and generation identity; one schema, one writer's rules, the record's `cohort.kind` telling them apart |
| Recorded results view | `src/api/routes/evaluations.py`; the one reader `src/api/services/record_views.py` | Read-only `GET /api/v1/evaluations/loaded-model`: the loaded checkpoint's records; shown by Models' read-only view (the Diagnosis card shows only a recorded-results state, since M5-2; worded "Recorded results" since research comparison Stage 6, a record being of any data kind and role). The one reader of recorded results (moved out of the route into `record_views` in research comparison Stage 8): `read_records` is every reader's one listing of a ledger's records for one set of weights (the route, Models through the shared model list, `record_evaluation.py --show`, the results service): the records that read as results, each as its data kind's view (`RecordView`, `KgEdgesRecordView` with its exposure view, read apart from the result so a block this build cannot read is "exposure not recorded" with its reason, `GeneratedRecordView`; research comparison Stage 6), and apart from them each one that does not, a kind this build does not read among them, by its index with a one-line reason naming fields, never values (`unreadable_records`, K8-6e.1); `result_view` reads a kept report's record by the same discrimination and parts (the results service's reading of a kept report, and the Test page's Result since Stage 8b); `checkpoint_ledgers` finds the ledger beside each listed checkpoint, `workspace_ledgers` every ledger of a workspace (each run's, whatever became of its weights) and `ledger_results` every record of one. The views build on the service layer's answer base (`src/api/services/response_models.py`, which the Diagnosis and evaluations routes import too), so the reader and the results service import no route |
| Results service and export | `src/api/services/results.py`; CLI `scripts/export_results.py` | The one read-only query over every kept report of every listed registry's stores and every readable ledger record of every ledger of the listed workspaces, whatever became of the weights it names (research comparison Stage 8): one row per result, a report and a record joined only on the run and its report bytes, a ledger-only record "report not available", a record merged before each run had its own record its own row, an unreadable item (a workspace entry outside its root and a record naming no weights among them) a row with its reason; each row with its model's identity from the server's lists, lineage, spec summary and structure and recorded training settings, its data identity and role, measurement settings, metrics as recorded, environment, the data's relation to training (a supplied cohort's R10, a KG edge set's exposure), states (a group's by `record_states.group_state`, the one turning of a scan into a state) and content references, `unknown` naming what has no source. One long-form serializer for CSV and JSON (one row per result and metric), its cells by `src/utils/csv_cells.py`, which the Diagnosis CSV shares. The command line names its sources explicitly and runs as the operator. Research › Models › Results (`src/webui/components/results_segment.py`, Stage 8b) lists the server's own rows by family or workspace (the service's filters; a result found in no listed workspace is listed under the workspace and counted in either scope), one table per protocol newest first, compares up to five side by side with no computed difference, and exports through the same serializer; each handler asks D8 first. A supplied cohort's R10 and a KG edge set's or generated split's lines are one rendering each (`renderers.r10_rows`, `renderers.data_kind_lines`) on Models, the Test page and Models › Results |
| Report store | `src/evaluation/report_store.py` | Every test run's kept report per cohort version, the scan that lists every run and describes how each condition group's runs compare (it never gates a registration, K8-6r.2), investigation notes, the usage history; one append at a time per file, through its lock file (K8-6r.1) |
| Workspace producer and readers | `src/kg/workspace.py`; `src/kg/artifacts.py` | One writer for every workspace (build, demo, probe); the manifest, dataset record and build record readers |
| Server lists | `src/api/services/server_lists.py`; `src/api/routes/lists.py` | Workspaces, checkpoints, cohort versions and intake files by name and digest, resolving only what they list |
| Job runner and GPU lock | `src/api/services/training_manager.py`; `src/utils/gpu_lock.py` | One managed GPU job at a time (training, test), its process group, cancel and stop; one lock file per deployment, also taken by the GPU CLIs (`train_model.py`, `measure_scorer.py`, `setup_demo.py --train-model`) and by the service while it serves on CUDA (`src/api/main.py`) |
| File lock | `src/utils/file_lock.py` | The one cooperative file lock (flock; Windows untested) under the GPU, ledger and report store locks; one lock per file, taken GPU, then ledger, then usage history where several are held (K8-6r.1) |
| Run layout and checkpoint reads | `src/utils/checkpoint_paths.py`; `src/utils/checkpoint_io.py`; `src/training/checkpoint_payload.py` | Run directories, one read per checkpoint with its digest, one payload builder for both writers |
| Operator access (D8) | `src/utils/operator_access.py`; `src/api/operator_access.py`; `src/webui/operator_access.py` | One check behind every operator control, in the API and the dashboard |
| Server settings | `src/config/server_settings.py`; `src/config/serving_device.py` | `SHEPHERD_DEPLOYMENT_MODE`, `_WORKSPACES_ROOT`, `_REGISTRY`, `_INTAKE_DIR`, `_TRAINING_EVIDENCE_DIRS` (server settings) and `_DEVICE` (serving device), read where used |
| Record states | `src/api/services/record_states.py` | Kept and registered states, and how a group's runs compare (descriptions only since K8-6r.2), for display (D4's ordering by test score retired in K8-6d, with `src/evaluation/ordering.py`) |
| Backend control | `src/api/services/backend_control.py`; `src/api/main.py` | Restart Backend (re-exec in place, after serving has stopped); shutdown that stops the active job, then serving |
| WebUI pages | `src/webui/components/` (`diagnosis_panel.py`, `diagnosis_card.py`, `model_list.py`, `datasets_page.py`, `training_console.py`, `finetune_page.py`, `test_page.py`, `models_page.py`, `runtime_settings.py`) | The pages; the shared help, renderers and model list |

---

## 8. The served Diagnosis score (running, since K8-4)

**What runs.** The Diagnosis page and `/api/v1/diagnose` serve one finetuned task model at a time,
and it ranks by its own task head (§2.2; `working/scorer-retraining/README.md` §3), each answer
naming its kind (`result_kind`):

| Task model | Ranks | Score | Kind |
|---|---|---|---|
| disease | every disease of the model's graph view | `−‖z_d − z_P‖²`, the disease head (uncalibrated) | `disease_task/v1` |
| gene | the request's candidate genes | `η × EMBSIM + (1 − η) × SPLSIM`, η from the model spec; SPLSIM from the exact hop distances the pipeline computes in process from the model's graph view (no table) | `gene_task/v1` |
| patients-like-me | a registered reference cohort's patients | the head's similarity (uncalibrated) | `patients_like_me/v1` |

The gene task is the one place a shortest-path term enters a score, as in the paper's candidate-gene
task; the disease task uses embedding distance alone (policy §3.1). The Test page measures a model by
a protocol this build implements that its capabilities admit (task measurement, §11).

**Retired in K8-4: the interim served mixture.** Until K8-4 a checkpoint without a model spec (a
ShepherdGNN history file) was served outside the WebUI by
`η × (cos + 1) / 2 + (1 − η) × 1 / (1 + mean hop distance)` (η = 0.7) over the diseases path search
reached, reading the shortest-path table (`served_mixture/v1`, scoring modes
`gnn_plus_shortest_path` and `gnn_only`). That serving is removed with its loader, scorer, result
kind and status fields; such a file is now refused at load (below). Mode C, a command-line
measurement of such files until K8-6a, is not the Diagnosis ranking. The table's code files (its
producer, pairing protocol and lookup) were retired in K8-5 (policy rev 14); a table left in a
workspace stays in place and nothing reads it.

### Refusals before anything is served
Every result states what produced it in its **served identity** (provenance contract M3c): the
checkpoint, graph, workspace manifest and dataset record by digest, the shortest-path term
(`computed` for a gene model, `absent` otherwise), the task's scoring mode and the settings in
effect, each part with its state. `get_pipeline_config()` and `GET /api/v1/pipeline/status` report
the same value for the loaded pipeline.
- **A model requested and not built** — a missing or unreadable checkpoint; one whose record names
  another graph or none (contract M2.4, checked first); **one without a model spec** (since K8-4,
  refused by `read_model_record` before any model is built, at startup and on every reload);
  **one whose record contradicts itself**, naming one input two ways, such as two KG digests
  (K8-4 review P2-2, refused by the same reader, provenance contract §9.1); a base; no graph data, no PyTorch, or a device the host cannot provide (contract M3c: `auto` or
  unset without CUDA): the build is refused (`PipelineBuildError`). Startup publishes nothing and
  `/diagnose` answers 503; a reload is refused and the served pipeline stays, as it does when a
  reload that names a task is given another task's model. Never replaced by path reasoning (provenance contract M1),
  and never moved to the CPU unless `cpu` was asked for. Since K7 no model is handed over in memory
- **No model source configured at all** (K7): no pipeline is built; `/diagnose` and `/ready` answer
  503 with the reason (B-2 is settled: a graph is never served without a model). The
  `path_reasoning/v1` result, its `path_score` and the identity states it used (checkpoint
  `absent`, graph `unverified`, `path_reasoning_fallback`) are removed

### Evidence after ranking (running)
After ranking, the **EvidencePanel** module (`src/reasoning/evidence_panel.py`) builds clinician-facing
evidence for each disease row shown, using PathReasoner as a building block. Paths never decide which
diseases are scored: a disease task model ranks every disease, so a row with no path within the limit
is reached and reads INSUFFICIENT. Gate 3's design (policy §6) is written: a target-restricted
traversal over four declared path kinds, with one fixed budget per target and a request ceiling at
admission (`docs/working/results-review/SPEC_3_EVIDENCE_AND_AUDIT.md` §7, awaiting approval).

#### Mode A — Direct Path Evidence
When the KG contains explicit paths from patient phenotypes to the candidate (≤ `weak_path_max_hops`, default 4), surface those paths verbatim:
- **STRONG_PATH** label if shortest path ≤ 2 hops
- **WEAK_PATH** label if shortest path is 3-4 hops

#### Mode INSUFFICIENT
When no direct path exists, the candidate is labeled **INSUFFICIENT** so clinicians know to validate
via independent clinical judgment; no evidence is assembled from anything else.

Mode B (analogy evidence by cosine over the ShepherdGNN embedding cache) was retired in K8-4 with
that cache, its only input (Q6); it was never reachable on the served path.

#### Critical invariant
EvidencePanel is run **after** scoring/ranking is complete. Toggling `include_explanations` does not change the candidate's score or candidate ordering — verified by `TestEvidencePanel.test_evidence_panel_does_not_affect_ranking` (on a trained disease child since K7).

### Two families called "Mode": they share letters, nothing else

The evidence modes above and the scorer-measurement modes are unrelated. Evidence Mode A has nothing to do with measurement Mode A, and there is no evidence Mode C or D.

| Family | Mode | What it is | Status |
|---|---|---|---|
| **Evidence**: the "why" in Diagnosis' candidate details, built after ranking (`src/reasoning/evidence_panel.py`) | A | Direct paths in the KG from the patient's phenotypes to the candidate | in use |
| | B | Analogy: the nearest known nodes in the GNN's embedding space, and their paths | retired in K8-4 with the ShepherdGNN embedding cache, its only input; never reachable on the served path |
| | INSUFFICIENT | No direct path found | in use: a disease row with no path within the limit |
| **Scorer measurement**: how a disease ranker is measured, work item B-0 (`src/evaluation/measurement.py`, `scripts/measure_scorer.py`) | A | Subgraph encoder; each batch's 2-hop subgraph seeded from the answers (the trainer's validation shape); the ladder's control | retired (R2, 2026-10-12) |
| | B | Full-graph encoder; the same candidates as A, so A→B isolates the encoder's scope | retired (R2, 2026-10-12) |
| | C | Full-graph encoder; every disease in the KG; raw cosine; ties go to the lower disease id. B→C isolates the candidate universe. The approved scorer, measured | the formal Test's measurement until K6, then a command-line run on a history file until K8-6a; since then no command runs it, and the Test page and the command line measure by a protocol (task measurement, §11). Its kept records read "approved scorer policy (Mode C)" until K8-6e.2 (2026-11-07); since then they are shown as recorded by the one display for every mode this build does not describe ("Mode C (a measurement mode this version does not describe)", every metric by its recorded name and value, the score semantics as recorded) |
| | D | Full-graph encoder; production's path-reachable candidates; the served score (the η/SP mixture, or the model term alone without a table). The ranking the Diagnosis tab served in October 2026 (the mixture retired in K8-4, above), measured | removed in K7 (implemented in B-0.5, Batch 3, 2026-10-26; `docs/working/scorer-measurement/PLAN_B05.md`; its GB10 run `docs/working/batch3-acceptance-2026-10-27/`). Kept records read "Served ranking of October 2026 (Mode D) … not the approved scorer policy" until K8-6e.2 (2026-11-07); since then they are shown as recorded by the one display for every mode this build does not describe (their metrics by recorded name and value, the score semantics such as `served:gnn_only:eta=1.0:sp=none` as recorded, never split) |

**How the measurement modes were decided.**
- The deploying institution decided (2026-08), before any measurement, that disease ranking is GNN-primary (every disease scored, raw cosine as the interim score) and that SP leaves the ranking for an optional, separate analysis (`docs/DISEASE_SCORER_POLICY.md` §1, §3.4). In the reference paper the disease task uses embedding distance alone; the SP fusion belongs to the candidate-gene task, which has a short candidate list. Mode C measures that scorer. Work item B-1 makes the Diagnosis tab rank by it; until then a Mode C result says the tab's ranking differs (`src/evaluation/caveats.py`). *(2026-11-07: B-1 was superseded by the task models, so the tab never ranked by Mode C; the note retired with the C and D labels in K8-6e.2, and a kept Mode C record is now shown as recorded under "Mode C (a measurement mode this version does not describe)", with no note.)*
- Modes A and B were retired on 2026-10-12: the formal Test's mode was settled as C (`docs/working/PLAN_TEST_RESULTS.md` D3), Mode A could not build a model from any checkpoint the current writer produces, and the institution had authorised retiring the old pipeline (`docs/working/scorer-measurement/ASSESSMENT_MODE_AB_RETIREMENT.md` §9).
- B-1 may not start before Policy Gate 1, a B-0 report comparing the served ranking (Mode D) with the planned GNN ranking (Mode C) (`docs/DISEASE_SCORER_POLICY.md` §6). C→D changes the candidates, the score and the tie rule at once, so Mode D also reports derived readings on the formal Mode C scores that change one of them at a time (`docs/working/scorer-measurement/PLAN_B05.md` §5).
  *(2026-11-09: policy rev 12 archived the C/D comparison, and its executors are removed (Mode D in
  K7, Mode C in K8-6b). Gates 1 and 3 gate B-1 and do not block the research implementation. Gate 1
  stands as written and is not cleared; until a revision is approved, statement 2 and gate 1 bind the
  clinical switch (rev 11). What replaces its B-0 prerequisite for task models is open: rev 12's
  proposal, the task models' own effect evaluation, not approved; the owner and then the institution
  decide it.)*

---

## 9. Platform Support

| Platform | GPU Attention | Vector Backend | Deploy Script |
|----------|--------------|----------------|--------------|
| Linux x86_64 | FlashAttention-2 / cuDNN SDPA | cuVS (GPU) → Voyager (CPU) | `deploy.sh` |
| Linux aarch64 (DGX Spark) | cuDNN SDPA (no FlashAttn) | cuVS (GPU) → Voyager (CPU) | `deploy.sh` |
| Windows x86_64 | FlashAttention-2 / torch SDPA | Voyager (CPU only) | `deploy.cmd` |

**Versions** live in `pyproject.toml` and `uv.lock` (since Batch 1, BACKLOG 50): PyTorch,
`torch_geometric` (PyG's pure-Python core, one wheel for every platform) and everything they
require. PyG's native extensions stay out of the lock because they come per platform: the deploy
scripts take them from the data.pyg.org index for the installed torch (derived, not written a
second time), or on ARM from the prebuilt release or a local build; A1 and A5 record their
versions. Since research-comparison Stage 2b (BACKLOG 71) the one run environment collector
(`src/utils/run_environment.py`), and so every checkpoint, every measurement's manifest, every ledger
record and A1, also records each extension as a named component: requested, installed, available,
selected and in effect told apart (what was not observed reads unknown), its native libraries' count
and aggregate SHA-256 (two builds of one version differ there; no file name is recorded), and the source receipt's
facts where the receipt matches what is installed. The record views show that identity in one wording,
on one line on Models and a component per line on the Test page, and carry every recorded fact of each
component (`components`: its states, fallback reason and source facts too), which the results query,
Compare and the export pass on. Recorded, never a gate. The launcher's optional accelerators stay out of the lock, and every install of one is
constrained to it (deployment guide §4.15). M5's GB10 acceptances (`b14c4de`, and the WebUI rounds of 2026-10-25) ran torch 2.10.0+cu130 with
`torch_geometric` 2.8.1, the current baseline, which `deploy.sh` installs from the lock since Batch 1
(its GB10 deploy and environment check at `0bc5178` passed, 2026-10-26; Windows not run); M4's ran
2.8.0.post1. K8's GB10 acceptance at `f2c2012` (2026-11-09) ran the same torch and `torch_geometric` with
`pyg-lib` 0.8.0 built locally, and the other three extensions byte-identical to the prebuilt release's, after the
prebuilt download failed
(`working/k8-acceptance-2026-11-09/`; `working/BACKLOG.md` 70, 71). The research-comparison round's GB10
acceptance at `e60f23e` (merged as #133; 2026-11-11/12, `working/research-comparison-acceptance-2026-11-12/`) ran the
same torch and `torch_geometric` on each ARM route in its own new environment: the prebuilt release (`pyg-lib` 0.6.0,
`torch-scatter` 2.1.2, `torch-sparse` 0.6.18, `torch-cluster` 1.6.3; A6 "prebuilt release"), and a local build of all
four (`pyg-lib` 0.8.0 at `26a2d1a`, the other three at the same versions; A6 "local build"), every native digest
differing from the release's. Upgrades (PyTorch, CUDA, PyG)
are allowed: they go through the same deployment, loading and training flow, are verified, and become
the new baseline; an older environment's evidence stays with that environment (BACKLOG 50).

The "GPU Attention" column names what can be enabled on each platform: SDPA comes with PyTorch, and
FlashAttention-2 (like xFormers and SageAttention) is installed only by the launcher's optional flags
(`launch_shepherd.sh --flash-attn`, `scripts/launch/shep_launch.py`), not by the deploy scripts. No running model path selects
an attention backend: the graph layers are PyG convolutions, and `src/models/attention/adaptive_backend.py`
is not wired in (§11, "reserved").

**Hardware acceleration is required; CUDA is currently the only supported backend** — the current deployment and acceptance scope, not a permanent architectural limit. Another accelerator is adapted, and accepted, on the actual target hardware once one is chosen, with its own explicit device-selection rule (never a silent fallback); hardware differences stay in the existing entry points (device resolution, allocator, AMP, capability checks, environment facts) rather than vendor branches across training, measurement, serving and the UI; and every backend shares the provenance contract and the scoring semantics, so a converted model is recorded as derived from its source checkpoint, never presented as it. No multi-backend framework is built in advance (`docs/working/BACKLOG.md`, item 24).

---

## 10. Design Constraints

1. **Zero real patient data for training**: All training uses simulated patients
2. **16GB VRAM support**: Subgraph sampling keeps memory bounded
3. **Offline-capable**: No internet required after initial data/model download
4. **Clinician-facing output**: Every prediction must include explainable evidence paths
5. **No false completeness**: Empty stub modules must not be presented as functional

---

## 11. Backend capabilities and their front-end entries (at `main` `30e8ae1`; product code `e60f23e`)

Each row is checked against the code: the formal entry really calls the implementation named, and
its output is what a front end would show. **Status**: *Connected* (a formal entry and a WebUI page
use it end to end); *Awaiting UI* (the formal entry works and its output is usable, but no page offers
it; what remains is the page and, where the row names one, a job type in the job runner that runs the
formal entry, whose rules stay in that entry); *CLI only* (a UI is not needed or was decided against); *Partial* (the named
part is missing); *Not implemented* (plans, reserved fields or unwired classes only). Evidence names
the GB10 acceptance and its tested commit where one exists (`docs/working/m*-acceptance-*/`); unit
and integration tests sit beside each module.

**Boundaries that are easy to confuse:**
- **A Mode C test score is not the Diagnosis page's ranking.** Mode C ranked every disease by raw
  cosine; Diagnosis serves a task model's own head (§8; the path-gated GNN/SP mixture is retired
  since K8-4).
- **Resume and extend are done; branch training is built for model bundles, from the CLI, the
  job runner and the API** (K5, the ops-wiring round's A1b; no GB10 record). A continued run
  inherits every learning setting. A branch is a new run from its parent's weights with a new
  optimizer; it changes only settings on its stage's list, or none (a same-settings branch), and
  records each change, `{}` when none (BACKLOG 23).
- **The GPU lock covers managed training and test jobs, the GPU CLIs (`train_model.py`,
  `measure_scorer.py`, `setup_demo.py --train-model`) and the service's CUDA inference** (the
  ops-wiring round's A3a): a model served on CUDA holds it until an operator stops inference, and a
  GPU job is refused meanwhile, naming inference; a job never unloads a model. CPU inference takes
  no lock and excludes nothing.
- **Trustworthy provenance and test records are not model quality or institutional acceptance.**
  Every GB10 run so far used synthetic cohorts or short trainings; no acceptance rule is declared.
- **M5's functional acceptance is not an acceptance of the WebUI's usability** (BACKLOG 51, 52).
- **A reserved field, an unwired class or an unreachable branch is not a feature** (the last rows).

| Capability and boundary | Formal entry | Shared implementation | WebUI | Status | Evidence | Remaining |
|---|---|---|---|---|---|---|
| **Build a workspace** from MONDO, HPO and the annotation files: graph, tensors, train/val samples (disease-disjoint), manifest, dataset record, build record. No test split | `scripts/build_knowledge_graph.py` | `src/kg/workspace.py` (the one writer), `src/kg/sample_generator.py`, `src/ontology/resolver.py` | Datasets shows the command only | Awaiting UI | demo build in the probe at `b14c4de`; the real build last ran 2026-09-29: its dataset record verifies (origin `files`, RW1 at `b14c4de`), but it has no build record, which M5 added | BACKLOG 29: a build job type in the job runner that runs this CLI behind D8 (the CLI stays the entry; budgets, ontology precedence and refusals stay in it); re-run a real build at current code |
| **Shortest-path table** (`shortest_paths.pt` and sidecar, bound to the graph); not part of the dataset fingerprint | none: `scripts/compute_shortest_paths.py` was removed in K8-5 | none: `src/inference/sp_artifact.py` and `sp_index.py` were removed in K8-5 | none; lists do not show its state | Retired in K8-5 | probe M4, R6, R7 at `b14c4de` | none: a table left in a workspace stays in place and nothing reads it |
| **Demo workspace**, optionally with a small trained model: since K7 `--train-model` trains a base and one disease child through the production trainer | `scripts/setup_demo.py` | the same writer; `train_model.train`; the GPU lock | none | CLI only | probe B1 at `b14c4de` writes the demo graph (`build_demo_kg`) through the shared writer; `setup_demo.py` itself and its `--train-model` lock have no GB10 record | none |
| **Workspace list, dataset and build records, Datasets page** (read-only, compare, "Use for training") | `GET /api/v1/workspaces`; `server_lists` | `src/api/services/server_lists.py`, `src/kg/artifacts.py`, `src/webui/components/workspaces.py` | Datasets; the Training run bar; Test's workspace | Connected | W1, RW1 at `b14c4de` | the open route returns problem texts with absolute paths (BACKLOG 53) |
| **Inventory before deployment** (M2.6) and dataset audits | `scripts/audit_checkpoint_family.py --inventory`, `scripts/audit_*.py` | the readers they audit | none | CLI only | M2's acceptance | two audits record a digest from a second read of what they parsed (BACKLOG 19.9; 19.12, optional) |
| **Start a training run** | `scripts/train_model.py`; `POST /api/v1/training/start` (D8; a strict start of a base, with every setting a new base reads, or of a continuation of either stage); the job runner | `src/training/trainer.py`, `src/utils/checkpoint_paths.py` (the run directory), `src/api/services/training_manager.py` | Training run bar, Start | Connected | W11, RW3 at `b14c4de`; manual Start at `b14c4de` | the API takes raw paths (BACKLOG 54; unknown keys are refused since K5); the two loss terms that never ran (BACKLOG 57) went with `MultiTaskLoss` in K8-7 (2026-11-08) |
| **Resume and extend** (inherited settings; extend at a fixed rate). Not branch | CLI flags; the API's `resume_from`; the job runner | `src/utils/run_plan.py`, the trainer, `checkpoint_paths.py` | Continue… drawer: Run → Checkpoint, Check, preview | Connected | M2, M9, M10, R3, R4 at `b14c4de` (the CLI path); Continue preview by hand at `f9202b6`; no GB10 record of a resume started from the drawer (unit tests cover it) | the API takes `resume_from` as a raw path (BACKLOG 54) and, since the ops-wiring round's A1a, `extend_lr`, with the preview's `zero_rate_groups` naming the groups whose rate ended at 0; the drawer sends no `extend_lr` yet (that round's B1), so from the pages such a parent is not extended. A model bundle run is resumed through the runner with its stage and task from its own record (K5); over the API a finetune child is continued too, its stage left out or stated as its own |
| **Pause after epoch N** | `train_model.py --pause-after-epoch` | the trainer | none, by decision | CLI only | probe M2 | the API refuses the field (422) since K5, instead of dropping it (BACKLOG 54) |
| **Stop training** | `POST /api/v1/training/stop` (D8); the runner | `training_manager.py` | Stop, with its question | Connected | W11, RW3; by hand at `b14c4de` | none |
| **Status, charts, CSV** of the run shown | `GET /api/v1/training/status`, `/training/metrics` | `training_manager.get_status`, `get_metrics_history` | run view: four charts, Export CSV | Partial | by hand at `b14c4de`, `f9202b6` | no page charts or exports an earlier run, though the backend reads any listed run (BACKLOG 52 H); BACKLOG 41 |
| **Run directories and checkpoint lists** (one read per file, with its digest) | `server_lists.list_checkpoints`; `GET /api/v1/workspaces/{name}/checkpoints` | `checkpoint_paths.read_run_checkpoints` | Continue drawer; Models; Change model; Test's model select | Connected | probes at `b14c4de` | three pickers over one list: move Test onto the shared model list (BACKLOG 52); `GET /training/checkpoints` is a legacy duplicate no page uses |
| **Training settings validation** | the job runner's start check for a new model bundle run and a Fine-tune start; the API's strict start and Fine-tune requests (`extra="forbid"`); the training process's own checks | `run_plan.new_run_settings` (what a new run takes, the CLI's rule) and `run_plan.new_run_problem` (the one check of a new run's settings, which the training process and the runner's starts make: `unread_problem`, the field spec's `value_problem` and `cross_field_problems`, `src/config/training_fields.py`: the rules across settings as data), `spec.hops_problem` and `spec.run_spec_problem`; the Fine-tune start's checks against its base (`_finetune_base_problem`) | the Training base form and the Fine-tune page, each asking the value check first (K6) | Partial | `test_training_fields.py`, `test_finetune_children.py`, `test_training_api.py`, `test_settings_projection.py` | `POST /api/v1/training/start` takes every setting a new base reads (ops-wiring round A1a), each value checked by `value_problem` and its bounds the field spec's; ShepherdGNN, finetune and unknown fields refused; one request is refused in one sentence by the CLI, the runner and the API; an explicit setting a new run would not read is refused, and a branch's change by its plan, while a resume or an extension, which changes nothing, never meets those rules; the pages offer only part of the settings until the round's B1 and B2 (BACKLOG 54) |
| **Run housekeeping** (delete, archive, rename, tags, compare) | — | — | — | Not implemented | — | BACKLOG 31 (compare, archive); tags are named in BACKLOG 23 as a prerequisite of the experiment phase and deferred (K5); delete and rename have no item yet |
| **Model families** (each checkpoint under the exact parent file it records; task and stage filters) | `GET /api/v1/workspaces/{name}/family` (D8); the lists' record fields | `src/api/services/server_lists.py` (`family_rows`, `list_family`) | Models: the Task select and the By family order (K6) | Connected on `main` (#131; K6) | unit tests; the K6 browser smoke on CPU (`working/scorer-retraining/README.md` §8, "K6 delivery"); no GB10 record | a fuller lineage view stays BACKLOG 52 (D) |
| **Fine-tune children** (one to three tasks from one base, in order, through the one job runner) | `POST /api/v1/training/finetune` (D8); `train_model.py --stage finetune` per task | `src/api/services/training_manager.py` (`start_finetune`) | Research › Prepare › Fine-tune: the form, the job card and its Stop (K6) | Connected on `main` (#131; K6) | unit tests; the K6 browser smoke on CPU (`working/scorer-retraining/README.md` §8, "K6 delivery"): three children in order, a Stop cancelling the queued one; no GB10 record | a stop cancels the queued children; a failed child does not stop the others |
| **Frozen-backbone disease control** (`frozen_parts` `graph_side`: the base's graph side kept as it is, only the patient encoder and the task head trained; each part's freeze and the downstream starting weights recorded; a rules-1 parent continues as `none`) | `train_model.py --frozen-parts`; `POST /api/v1/training/finetune` `settings` (D8; invalid settings 422); the job runner | `src/config/recipe.py` `frozen_parts_problem`, `src/models/bundle/builder.py` `ModelBundle.freeze`, `src/utils/run_plan.py` `decode_learning_settings` and `recorded_frozen_parts` | Fine-tune's Graph backbone (Trained or Frozen, the field spec's values) with a note on what trains, and the job card's row; the model lists, Models' Kind and the Results Model cell say "Disease · frozen graph backbone" (`renderers.model_kind_text`); Models › Results shows each part's freeze | Connected (research comparison Stage 9a, 9b; ops-wiring round A2) | `tests/unit/test_frozen_backbone_control.py`, `test_rules_1_continuation.py`, `test_webui_finetune_page.py`, on the CPU; the probe's fp32 check `L7.disease_frozen` (paused and resumed, the graph side's parameters and buffers unchanged), on the CPU in the tests, on CUDA at the candidate-slot run; AMP, its scaler and `torch.compile` with a freeze not exercised | none |
| **Branch training** (a new run from a model bundle parent's weights, its settings kept or changed on its stage's list, each change recorded, `{}` when none) | `scripts/train_model.py --branch-from`; the job runner (`TrainingManager.preview_branch`, `start_training` with `branch_from`); `POST /api/v1/training/branch` (D8) | `src/utils/run_plan.py` (`plan_branch`), `src/config/recipe.py` (`branch_settings`), the one start and spawn (`training_manager`), bound to the parent read (`--expect-parent-sha256`) | none to start one; Models shows what a branch changed (identity and By family) | Connected to the runner and the API (the ops-wiring round's A1b); the page is that round's B1 | unit tests (`test_branch_plan.py`, `test_branch_operation.py`, `test_job_runner.py`, `test_training_parent_expected.py`, `test_training_api.py`); the probe's L7.branch on the CPU; no GB10 record | the Training page's Branch mode (`working/ops-wiring-and-ui-ux/README.md`, B1); BACKLOG 23 |
| **One managed GPU job at a time**: the lock, its holder, refusals naming it | the job runner; `train_model.py`, `measure_scorer.py`, `setup_demo.py --train-model`; the service's CUDA load | `src/utils/gpu_lock.py`, `training_manager.py` | header chip; Training and Test GPU lines | Connected | W7, W11, RW3 at `b14c4de` | a job this backend did not start is shown and refused, and stopped by pid only (by design) |
| **Test run start and cancel**, process-group reaping | the runner's test job | `training_manager.start_test`, `cancel_test` | Test: Start test, Cancel | Connected | W8, W10 at `b14c4de` (a start run to the end, a training stop refused meanwhile; a start refused before measuring); Start test by hand at `773dc4a`; no GB10 record of a running test cancelled or of leftover group processes ended (unit and integration tests cover them) | none |
| **Backend stop stops its job; Restart Backend** (both stop serving too: accepted diagnoses finish, the model and the GPU lock are given back; Restart is refused while a load or a stop runs) | process signals, the systemd unit; Settings | `src/api/main.py` shutdown, `backend_control.py` | Settings: Restart Backend; stopping stays an operator action | Connected (Restart); CLI only (stop) | normal stops at `b14c4de`, `f9202b6`, `773dc4a` (no job running); no GB10 record of a Restart Backend (`tests/unit/test_backend_control.py` covers it, with serving: `test_restart_stops_serving_before_the_exec`) | no GB10 record of a stop while a job runs; a SIGKILLed backend leaves its job holding the lock (its own serving lock the kernel releases) |
| **CUDA inference and research jobs exclude each other** (minimal, BACKLOG 48): CUDA serving holds the deployment's one GPU lock, decided by the resolved device, from before a candidate is read until an operator stops inference; CPU serving takes no lock; one load or stop at a time, refused rather than queued; a stop refuses new diagnoses at once, lets the accepted ones finish, then releases | `POST /api/v1/pipeline/reload` and `POST /api/v1/pipeline/stop` (D8); start-up's environment config | `src/api/main.py` `AppState` (the lifecycle, admission, the holding, the one not-serving record), `routes/pipeline.py` `run_reload`, `run_stop`, `src/utils/gpu_lock.py`, `TrainingManager.serving_refusal` | the shared Load (Change model, Models' Load) loads; the Diagnosis card's Stop inference stops (D8, asked first and bound to the model shown, the one stop in process) and shows the lifecycle's state in its words; the GPU chip names this service's inference and the research starts point to its stop | Connected, minimal (the ops-wiring round's A3a, A3b) | `tests/unit/test_serving_lifecycle.py`, `tests/integration/test_serving_lock.py` (a CLI refused naming inference, across processes), on the CPU with CUDA faked; the page's stop in `tests/unit/test_diagnosis_panel.py`; the probe W12 on a service of its own, run on the CPU in the probe's tests; no GB10 record | W12 on CUDA at the GB10 candidate-slot acceptance (`working/ops-wiring-and-ui-ux/README.md` §8) |
| **Model list, preview and Load** (confirmation bound to the fingerprint shown) | `server_lists`; `POST /api/v1/pipeline/reload` (D8) | `src/webui/components/model_list.py` (one list), `routes/pipeline.py` `run_reload`, `src/api/main.py` `publish_pipeline` | Diagnosis: Change model; Models: Load | Connected | T5, RT3 at `b14c4de`; by hand at `f9202b6` | the API reload still takes a path. (BACKLOG 42, the SP table not bound to the confirmation, has no subject since K8-5: no table is read) |
| **Served identity and status** (with the lifecycle: serving, loading, in flight, the GPU lock held, the not-serving reason); Diagnosis card, badge and drawer | `GET /api/v1/pipeline/status` | `served_identity`, `AppState.snapshot`, `src/webui/components/served_status.py` | Diagnosis card, badge, identity drawer | Connected | M3c's acceptance at `52152e4`; probes at `b14c4de` | none |
| **Which model serves after a start or a restart**: the one the environment configures for start-up, loaded through the same lifecycle (lock, build, checks; a refusal reported as not ready with its reason, no other model chosen), else none; a page Load is not restored (the owner's authorisation of 2026-11-12) | environment variables at start-up | `src/api/main.py` lifespan, `initialize_pipeline` | Settings' Restart words say a loaded model is stopped and not loaded again unless the start-up configures one | Connected (the ops-wiring round's A3a, A3b; `/api/v1/pipeline/config`, whose file nothing read, is retired) | `tests/unit/test_backend_shutdown.py`, `test_diagnose_refuses_when_configured.py`, `test_backend_control.py` | no GB10 record of a restart while a model serves (BACKLOG 55) |
| **Import a test cohort** (preview, then commit a version) | `scripts/import_cohort.py`; Test's import drawer | `src/evaluation/cohort_registry.py` | Test: Import a new version… | Connected | T1 and T4 at `2c94a4e` (T2, T3, T5, T6 measure, register, view and audit the imported version); the import drawer by hand at `773dc4a` | MyGene2 and institutional adapters, OMIM/ORPHA (BACKLOG 25) |
| **Keep a generated split** (one of a workspace's own generated splits, as `train`, `val` or `test`, with its functional gene fixture when the workspace holds one; preview, then commit) | `scripts/import_cohort.py --kind generated` | `src/evaluation/cohort_registry.py` (`prepare_generated_import`, `version_references`, `check_references`) | none (`GET /cohorts` lists it by kind, role and counts; the Test page by label, kind and role ("gen-val · Generated split (val)", with its counts in their units; research comparison Stage 7); Diagnosis's Reference cohort select marked "generated split", never chosen by default) | CLI only (research comparison Stage 4b; the fixture since Stage 5b) | unit tests on CPU (`tests/unit/test_generated_versions.py`, with the registry move acceptance; `tests/unit/test_generated_measurement.py`); on the GB10 at `e60f23e` (research comparison, merged as #133; `working/research-comparison-acceptance-2026-11-12/`): L6 keeps a generated split and measures it | a generated test role cannot be imported (the generator writes train and val); reports measured with `--split` stay unregistrable |
| **Measure a kept generated split** (`--cohort <label>` of kind `generated`: disease over its samples, gene and patients-like-me over the fixture it keeps, through the evaluators `--split` uses; its report kept in the version's report store, under its role as recorded) | `measure_scorer.py --cohort` | `src/evaluation/cohort_registry.py` (`read_generated_task_fixture`), `src/evaluation/protocols.py` (the fixture's data gap) | the Test page's Start, through the job runner | CLI and Test page (research comparison Stage 5b) | unit tests on CPU (`tests/unit/test_generated_measurement.py`); a run through the job runner (`tests/integration/test_job_runner_measure.py`); on the GB10 at `e60f23e` (research comparison, merged as #133; `working/research-comparison-acceptance-2026-11-12/`): L6 measures a kept `val` split on CUDA | a version without the fixture is refused gene and patients-like-me as a stated data gap; its run is given no R10 evidence directory, by the job runner's one choice by kind (`evidence_choice`, Stage 7) |
| **Import a KG edge set** (a part of a checkpoint's recorded edge split, as `train`, `val` or `test`; preview, then commit) | `scripts/import_cohort.py --kind kg_edges --from-checkpoint` | `src/evaluation/cohort_registry.py` (`prepare_kg_edges_import`, `read_kg_edge_pairs`), the LP core's `fixed_lp_pairs` and `rebuild_split` | none (`GET /cohorts` lists it by kind, role and counts; the Test page by label, kind and role ("kg-test · KG edge set (test)", with its counts in their units, and the import command when the registry holds none; research comparison Stage 7); Diagnosis's Reference cohort select marked "KG edge set", never chosen by default) | CLI only (research comparison Stage 4) | unit tests on CPU (`tests/unit/test_kg_edge_versions.py`); on the GB10 at `e60f23e` (research comparison, merged as #133; `working/research-comparison-acceptance-2026-11-12/`): L1 imports one, and L4 its refusals before anything is written | measured by `kg_link_prediction` (research comparison Stage 5a); its runs registered with their exposure (Stage 6) |
| **Task measurement** of a registered version, by a protocol this build implements that the model's capabilities admit (today disease, gene, patients-like-me, and KG link prediction on a KG edge set, research comparison Stage 5a); every report kept | `scripts/measure_scorer.py --mode <protocol> --cohort`; the runner's test job, which passes the protocol chosen as `--mode` and refuses, before it starts, one the listed model's capabilities or the listed version do not admit (K8-6a; research comparison Stage 3) | `src/evaluation/protocols.py`, `src/evaluation/task_measurement.py`, `report_store.py` | Test: Measurement (K8-6a), Start test, Result (K6) | Connected on `main` (#131; K6); the GB10 functional acceptance of K8's agreed scope passed at `f2c2012` (2026-11-09): the disease protocol measured on CUDA, gene and patients-like-me on CPU only; since #133 (`e60f23e`'s GB10 acceptance) all four protocols measured on CUDA, gene and patients-like-me at demo size only | unit and integration tests; the K6 browser smoke on CPU (`working/scorer-retraining/README.md` §8, "K6 delivery"): a disease and a gene child measured and registered; the probe's W8, W9 and RW2 measure a disease child (`792861b`), and did on the GB10 at `f2c2012` (K8's acceptance, 2026-11-09, `working/k8-acceptance-2026-11-09/`; a locally built `pyg-lib`); on the GB10 at `e60f23e` (research comparison, merged as #133; `working/research-comparison-acceptance-2026-11-12/`), on both deploy routes, KG link prediction through `measure_scorer` and the job runner (L2, L3, L5), disease on a generated split (L6) and the three task protocols on demo-sized children (L7), on CUDA | a patients-like-me measurement was not run in the smoke |
| **Mode C measurement** of a registered version; every report kept | none since K8-6a (until then `scripts/measure_scorer.py --cohort` without `--mode`, on a history file, command line only) | none since K8-6b (the executor `measurement.run_mode_c` retired); `report_store.py` keeps its reports | none since K6: Test shows a kept Mode C report and can register it | Retired (the command in K8-6a, the executor in K8-6b); kept Mode C reports and records are read by the shared readers (policy rev 15) | W8 at `b14c4de`; by hand at `773dc4a` | none: retired (`working/scorer-retraining/README.md` §7). Not the Diagnosis ranking |
| **Measure a generated split** (`--split {train,val}`); its report and ranks are new files at `--output` that never replace one, and a used `--output` is refused (K8-6r.4) | `measure_scorer.py --split` | the same; `src/utils/file_modes.py` (`publish_bytes(..., replace=False)`) | none | CLI only | M3, R5 at `b14c4de` (`--split val`); its tests | not registered: it has no registered version, so no data identity the ledger records results by; the same split kept as a version (`import_cohort.py --kind generated`) is measured with `--cohort` and registered (research comparison Stage 6) |
| **Register a kept report** (with `--record`, or later by hand); report store and deployment registry only; by the verified run, reading no other report (K8-6r.2); a report of a registered version of any kind, its record built by kind (research comparison Stage 6) | `registration.register_report`; `scripts/record_evaluation.py`; Test | `src/evaluation/registration.py`, the ledger (schema 4) | Test: Register this report | Connected | W9, RW2 at `b14c4de`; by hand at `773dc4a`; by kind: unit tests on CPU (`tests/unit/test_ledger_by_data_kind.py`, the KG and generated measurement tests) and the command line (`tests/integration/test_kg_link_prediction_cli.py`); by kind on the GB10 at `e60f23e` (research comparison, merged as #133; `working/research-comparison-acceptance-2026-11-12/`): L2, L3 and L5–L7 | none |
| **Results query, Compare and export** (every kept report of the listed registries' stores and every readable ledger record of the listed workspaces, joined only on run and report identity; a ledger-only record reads "report not available"; an unreadable item is its own row; newest first, never capped or ordered by a metric) | `scripts/export_results.py`; Models › Results in process (no HTTP route) | `src/api/services/results.py`, `src/api/services/record_views.py`, `src/utils/csv_cells.py` | Research › Models › Results (`results_segment.py`): one table per protocol, Compare (the first five chosen, a UI cap only), Export | Connected (research comparison Stage 8, 8b) | `tests/unit/test_results_export.py`, `test_models_results_segment.py`; the round's browser check on the CPU (2026-11-10); L8 on the GB10 at `e60f23e`, on both deploy routes (merged as #133; `working/research-comparison-acceptance-2026-11-12/`) | the default order (BACKLOG 63); BACKLOG 72 |
| **D5 investigation notes** | Test page only (in process) | `report_store.add_investigation_note` | Test | Partial | — | not shown on Models or the loaded-model route (BACKLOG 40) |
| **Records read back, group description, usage history** | `GET /api/v1/evaluations/loaded-model` (records with their run, group description); the usage history: the Models page only (in process) | `record_states.py`, `report_store.py` | Models (records, usage, read-only view); Diagnosis card state | Connected (records, group description); Partial (usage: no formal entry); D4 ordering by test score retired in K8-6d | T5, RT3 at `b14c4de`; by hand at `773dc4a` (one tested model) | before a non-Gradio client: move the page-resident glue (ledger placement, usage entries) into `src/api/services` |
| **Compare the served ranking with Mode C** (Mode D, B-0.5) | removed in K7 (`--mode D` is refused) | — (`src/evaluation/mode_d.py` deleted with probe T7) | Test: Result and Models: kept Mode D records shown as recorded by the one display for every mode (since K8-6e.2; by their own label until then); a kept Mode D report registers by the general rules under its own mode (the same run's report again answers `already_registered`) | Removed (history readable) | implemented in BACKLOG §0 batch 3; the batch's short GB10 run passed at `be0ee1a` (2026-10-27, `working/batch3-acceptance-2026-10-27/`) | none: archived (owner decision, `working/scorer-retraining/README.md` §7) |
| **Model quality, institutional acceptance** | — | — | — | Not implemented (external) | — | data, terms and a declared protocol |
| **Diagnose** (the loaded task model's result kind, §8), result identity, CSV and report | `POST /api/v1/diagnose` | `src/inference/pipeline.py`, `src/models/tasks/scoring.py`, `served_identity` | Diagnosis page and exports | Connected | HTTP and the tab at `52152e4` (M3c); by hand at `f9202b6`; probes E7, R6 at `b14c4de` call the pipeline without explanations | none for the path fallback: since K7 nothing is served without a model (B-2 settled; `path_reasoning/v1` and the demo answer removed). Batch 4 (implemented and accepted, 2026-10-27) answers each kind with its own fields; an SP value with no number is a status, never 0.0 (policy C9; §11.1 step 7) |
| **Evidence and explanations** | the pipeline's explanations step | `evidence_panel.py`, `explanation_generator.py` | Evidence, Explanation, Paths | Connected (Mode A only) | by hand at `f9202b6` | evidence Mode B retired in K8-4 (it was never reachable); a disease row with no path reads INSUFFICIENT; the target-restricted traversal is gate 3's design (§11.1 step 6); a computed summary the route drops; inert flags (BACKLOG 56) |
| **B-1**: GNN-primary ranking over every disease; SP as an optional analysis surface; evidence for the visible page only | — | none built. As designed it reused Mode C's canonical ranking and `sp_index` for the SP surface (retired in K8-6b and K8-5); an SP surface would compute its distances as the gene task does (`src/kg/graph_view.py` `exact_distances`, as policy rev 14 says, `docs/DISEASE_SCORER_POLICY.md` §1.2). The disease scorer is the model bundle's head (`working/scorer-retraining/README.md` §3) | — | Superseded (not built) | — | **Superseded 2026-10-27:** the raw-cosine B-1 is not built. The paper-aligned disease model is served on the research platform when the A + B integration is delivered (one switch, policy rev 12); clinical adoption keeps its own conditions. The GNN-primary rule, SP as an optional surface and lazy evidence stay the target. *(2026-11-09: that model is served on the research platform since K8-4 and is on `main` since #131, where the GNN-primary rule holds. **Open, clinical adoption** (the owner, then the institution): rev 11's proposal for statement 2 and B-1 and rev 12's for gate 1's B-0 prerequisite are not approved; until a revision is approved, statement 2 and gate 1 bind the clinical switch (rev 11). **The switch's default** is the owner's decision, recorded as open (policy §7; §2.1, "Who decides what"); the header's "What is open".)* |
| **Paper-aligned two-stage training**: KG base, `finetune`, the disease, gene and patients-like-me tasks, one model builder | `train_model.py --stage`; `POST /api/v1/training/finetune` | `src/models/bundle/`, `scripts/train_model.py` `train` (the one training entry, `train(config, explicit)`, since K8-7 `e35b2a3`) | Training's base form, Fine-tune, Test by task, Models by task and family, Diagnosis by task (K6) | Built in K1–K8, on `main` (#131); the GB10 functional acceptance of K8's agreed scope passed at `f2c2012` (2026-11-09) for the GATv2 base and its disease child; the gene and patients-like-me heads on GPU at demo size only (`e60f23e`, L7); other architectures and the WebUI in a browser not accepted on the GB10 | unit tests; the K6 browser smoke on CPU (`working/scorer-retraining/README.md` §8, "K6 delivery"); K8's GB10 acceptance at `f2c2012` (2026-11-09, `working/k8-acceptance-2026-11-09/`): on the real data a GATv2 base trained (fp32), resumed and finetuned to a disease child, measured and served; on the demo, float16 with its scaler restored across a continuation (M2, M9); with a locally built `pyg-lib`; on the GB10 at `e60f23e` (research comparison, merged as #133; `working/research-comparison-acceptance-2026-11-12/`), one-epoch demo-sized gene and patients-like-me children trained and measured on CUDA (L7), on both deploy routes | designed 2026-10-27 (`working/scorer-retraining/README.md`, `working/PLAN_PROVENANCE_CONTRACT.md` §9, `working/PLAN_TEST_RESULTS.md` §8): work packages A–C |
| **HPO search, disease details, keeping a diagnosis** | mock routes (HPO search, disease details); `GET /api/v1/diagnose/{session_id}` answers 501 | — | — | Not implemented | — | BACKLOG 56. `candidate_genes` is no longer reserved: a gene task model ranks it (K4), and Diagnosis sends it for a gene model (K6); any other model ignores it with a warning |
| **Operator access (D8) and the read-only view** | `OPERATOR_ONLY` (`src/api/operator_access.py`) on every gated route; the dashboard's helper `src/webui/operator_access.py` in each operator handler | `src/utils/operator_access.py` (`check_operator_access`, the one check) | chips; greyed controls with the reason | Connected | W2–W6 at `b14c4de`; a read-only tab by hand at `773dc4a` | open read routes name server paths (BACKLOG 53) |
| **Server settings** | `SHEPHERD_*` environment, read where used | `src/config/server_settings.py` | shown, never edited | Connected | probes at `b14c4de` | none |
| **Launcher, bind guard, service unit** | `launch_shepherd.sh`, `scripts/launch/shep_launch.py`, the systemd unit | the bind rule in `src/config/server_settings.py` (`network_bind`, `startup_bind`), applied by the launcher, `src.api.main` and startup | none | CLI only | W2 at `b14c4de` starts the service with the launcher's uvicorn entry and arguments (not the bind guard's refusal or the unit, which their unit tests cover); `launch_shepherd.sh` by hand at `773dc4a` | the launcher's optional accelerators stay out of the lock; every pip install of one is constrained to `uv.lock` (exported by `uv`) and refused when that cannot be had, so none replaces a locked package (BACKLOG 50; deployment guide §4.15) |
| **Allocator preset, torch.compile, Restart Backend** | Settings | `src/config/runtime_presets.py` | Settings | Connected; compile Partial | its tests | compile reaches WebUI-started training only (BACKLOG 54) |
| **Device resolution** (`auto` is CUDA and refuses without it; `cpu` only when asked) | the training, measurement and serving resolvers | `resolve_serving_device` (`src/inference/pipeline.py`; the requested value read by `src/config/serving_device.py`, `SHEPHERD_DEVICE`), `resolve_training_device` (`scripts/train_model.py`), `_resolve_device` (`scripts/measure_scorer.py`) | device choices | Connected | probes at `b14c4de` | the API and CLI hand `mps` to torch unchecked and check `cuda:N` only for CUDA, not its ordinal (BACKLOG 24) |
| **WebUI usability** (density, layout, help texts) | — | the pages in `src/webui/components/` | every page | Partial: functionally accepted only | M5's manual rounds to `773dc4a` | BACKLOG 51, 52, in §0's UI batches; not a condition for reopening M5 |
| **Reserved, not features**: the attention backend selector and the launcher's accelerator flags (they install packages no model path uses); fuzzy input; the platform and config homes; several front ends on the LAN; non-CUDA hardware | — | — | — | Not implemented | — | BACKLOG 28, 24 |

### 11.1 The diagnosis and measurement chain (Batch 2 design, at `60feb6d`)

The same chain read end to end, from input to registration. For each step the table gives four
things:
- today's formal entry and its shared function;
- the gap;
- what changes, and in which batch of BACKLOG §0;
- which old entry goes.

The designs are:
- `docs/working/scorer-measurement/PLAN_B05.md` (Mode D);
- `docs/working/results-review/SPEC_1_RESULTS_REVIEW.md` §2.1 (the result union) and §2.2 (the
  shared scoring boundary);
- `docs/working/results-review/SPEC_3_EVIDENCE_AND_AUDIT.md` §7 (gate 3's traversal).

**What each piece does:**
- **Mode C** measures the approved policy: raw cosine over every disease.
- **Mode D** measured what the Diagnosis page served in October 2026, through serving's own method,
  and set it beside Mode C in the same run (removed in K7, below).
- **B-1** makes the Diagnosis page serve the approved policy, behind a switch.

Mode C changes no served value, and Mode D changed none. B-1 does.

**Superseded in part, 2026-10-27.** Batch 5's changes below (B-1 with raw cosine) are not built.
**Mode D (steps 2, 3 and 8) was removed in K7** with its executor; its rows stay as the design
record, and its kept reports stay readable.
- **Steps 2–5 and 7** become the paper-aligned disease model's steps in work package A:
  - one model builder;
  - the disease head over every disease;
  - `selection_limit` still applies (`canonical_ranking` retired with Mode C in K8-6b; the task
    models rank ties by average rank, E3);
  - a new result kind for that model, not `gnn_ranking/v1` as raw cosine.
- **The served switch** happens on the research platform when the A + B integration is delivered
  (policy rev 12); clinical adoption keeps its own conditions.
- **Step 6** (gate 3's evidence) keeps its design.
- **The rows** stay as the record of the Batch 2 design
  ([`working/scorer-retraining/README.md`](working/scorer-retraining/README.md) §8).

*(2026-11-09: Mode C retired as well, its command in K8-6a and its executor in K8-6b, its kept
reports readable; the served switch happened on the research platform in K8-4, and clinical adoption
is open (§2). Several entries the rows name are gone: the mixture and its scoring (K8-4),
`build_shepherd_model` (K8-10), `src/inference/scoring.py` (K8-11). "Today" in the table means
`60feb6d`; what runs is §8, and its status is §11.)*

| Step | Today: formal entry and shared function | Gap | Change (batch) | Old entry removed |
|---|---|---|---|---|
| 1. **Input** (phenotype terms) | `DiagnosisPipeline.validate_input` with `normalise_phenotypes` (`src/kg/phenotype_normalisation.py`); the API's `DiagnoseRequest`; for cohorts, the importer's `map_cases` | Serving maps a term three times, each handling a miss differently. A graph node with no tensor row is skipped silently. `validate_phenotypes=False` is a second behaviour | One mapping step that returns tensor rows. An unknown term is dropped and stated; a node without a row is refused (SPEC_1 §2.2). **Batch 4, implemented and accepted (2026-10-27)** | `_phenotypes_to_node_ids`; the per-score lookups; `PipelineConfig.validate_phenotypes` |
| 2. **Model forward** | Serving: `_precompute_node_embeddings` (float32, cached). Mode C: `encode_full_graph` (retired in K8-6b). Both build through `build_shepherd_model` | Serving does not expose its numerical regime | `embedding_regime` exposed and checked against Mode D's manifest (PLAN_B05 §3). **Batch 3, implemented** | none |
| 3. **Candidate set** | Serving: `_find_all_paths` (BFS and the direct finder, both with caps that depend on order). Mode C: every disease row | Statement 3 (every disease is scored) is not served; a truth discovery does not reach has no served rank | Batch 3 (implemented): Mode D measures today's gate and counts what it does not reach (`not_reached`). Batch 5: B-1 scores every disease row | Batch 5: discovery stops being the candidate gate; it stays only for the path-reasoning fallback |
| 4. **Scoring** | Serving: `_calculate_combined_score`, `_calculate_gnn_score` and `_calculate_sp_score`, each called once per candidate, over the primitives in `src/inference/scoring.py`. Mode C: `masked_mean_pool`, `cosine_score_matrix` | The pool clamps indices. An unavailable model term or SP value becomes `0.0`. `(cos+1)/2` is rounded in float32 | Batch 4 (implemented and accepted, 2026-10-27): the model term fails closed (`MODEL_TERM_UNAVAILABLE`); SP gets statuses from one caller, `sp_index.sp_analysis_records`; the per-candidate `_calculate_combined_score` and `_calculate_sp_score` are gone. Batch 5: `score_patients` (raw cosine) | Batch 5: `pool_patient_embeddings`. The mixture helpers go when `served_mixture/v1` retires |
| 5. **Ranking** (order, ties, cut) | Serving: a stable sort, then the `top_k` slice, inside `_score_and_rank_candidates`. Mode C: `canonical_ranking` (`src/evaluation/measurement.py`, retired in K8-6b) | No untruncated served rank exists. There are two tie rules, and serving cannot import Mode C's | Batch 3 (implemented): `DiagnosisPipeline.rank()`, which `run()` calls before its cut. Batch 5: `canonical_ranking` moves into `scoring.py`; `selection_limit` cuts after scoring (not done, 2026-11-07: `canonical_ranking` retired with Mode C in K8-6b instead of moving; the task evaluators rank by average rank, `average-rank/v1`, E3) | Batch 5: serving's own sort, for the GNN kind |
| 6. **Evidence** | `_add_explanations` calls `EvidencePanel.build_evidence`, always Mode A from discovery paths, eagerly, for the top K only | Not restricted to the target; no budget; Mode B and INSUFFICIENT unreachable; no record identity | `src/reasoning/target_evidence.py`: four declared path kinds, one fixed budget per target, computed lazily per page (SPEC_3 §7). **Batch 5, gate 3** | `EvidencePanel` and the eager step, together with the legacy result kinds |
| 7. **Result and report** | `InferenceResult` and `DiagnosisCandidate` (`src/core/types.py`); `POST /api/v1/diagnose`; the Diagnosis page, its CSV and its report | One flat record. `confidence_score` holds two quantities. An unavailable value is written as `0.0`. The demo answer carries invented scores | The result union, discriminated by `result_kind` (SPEC_1 §2.1). Batch 4 (implemented and accepted, 2026-10-27): `served_mixture/v1`, `path_reasoning/v1`, `demo/v1`, with no alias for the removed fields. Batch 5: `gnn_ranking/v1` | `confidence_score`, `gnn_score` and the bare `sp_score`; `scripts/test_gnn_inference.py`; `generate_evidence_summary` |
| 8. **Measurement** | Mode C: `scripts/measure_scorer.py --cohort` calls `run_mode_c` (the command retired in K8-6a, the executor in K8-6b). Mode D: none | Gate 1's comparison does not exist | `run_mode_d` and `--mode D`: post-import and among-reached metrics over named case sets; the C side is `run_mode_c` itself on the same embeddings; derived readings one change at a time (PLAN_B05). **Batch 3, implemented** | none |
| 9. **Registration and comparison** | `registration.register_report`; the ledger (D4 groups and the D5 key, both including `mode`); `report_store`'s usage history; the Test and Models pages | Labels assume Mode C; the usage history has no mode | Mode D's values go in the existing manifest fields, with no new field; labels come from each record; the usage history records a mode (PLAN_B05 §6). **Batch 3, implemented**; since K8-6e.2 (2026-11-07) the C and D labels are gone and every mode without words of its own is shown as recorded; the ordering refused a non-Mode-C scoring group itself (PLAN_B05 §11; the ordering retired in K8-6d) | none: no ledger rule changes |

**Proving correctness without rerunning old acceptances:**
- **Golden tests** pin what must not move: serving's output across batches 3 and 4, and Mode C's
  reports when batch 5 moves its ranking. They compare the promised fields (scores, order, ranks,
  cases, digests, semantic fields), never timestamps, revisions or run metadata
  (`docs/working/scorer-measurement/PLAN_B05.md` §8).
- **New tests** cover only what is new.
- **One short GB10 run per batch** where hardware matters: batch 3 runs Mode C, then Mode D, on a
  registered synthetic cohort (passed at `be0ee1a`, 2026-10-27).
- **Kept evidence still counts**, unchanged: M4's and M5's acceptances, B-0.4's SP benchmarks and the
  Mode C reports.

---

## 12. One implementation per function

Different uses may have different interfaces; one function never keeps two sets of rules. Every next
piece of work says first **which existing implementation it reuses, which gap it fills, and whether
it replaces an old entry**.

- **The front end never decides** model compatibility, continuation rules, registration
  eligibility, operator access or GPU use. It asks the shared implementation (`server_lists`,
  `run_plan`, `register_report` and `report_state`, `operator_access`, the job runner) and shows the
  answer. A kept or registered state, and how a group's runs compare, come from the report store's
  readers, never from a job's exit status.
- **New pages use the same lists, readers, checks and job runner** as the existing ones: one
  checkpoint reader, one model list, one reload, one served-status reader, one device resolver per
  role, allocator settings through `runtime_presets` only. No editor for `SHEPHERD_*` settings.
- **The CLIs stay formal operating and test entries.** CLI and API may parse their own arguments, but
  they share the business logic; a CLI does not go through HTTP for the sake of uniformity.
- **A new need changes the shared implementation** and moves its users with it; no second "new" copy.
  Logic that lives in a page today (the Test page's ledger placement, the Models page's ranking and
  usage entries) moves into `src/api/services` before a second client needs it.
- **No fallback for an old format nobody asked for.** The old pipeline's artifacts are unsupported and
  archived (contract decision 2); an old plan's compatibility passage is checked against it first.
- **No framework ahead of a user**: no plugin system, general job framework or free drag-and-drop
  dashboard as a prerequisite.
- **One disease scorer** (Batch 2 design, §11.1). The paper-aligned disease model scores a patient
  through one function in validation, measurement and serving (`patient_vectors` and the task
  head's scores, `src/models/tasks/scoring.py`); Mode C's raw-cosine ranking (`canonical_ranking`)
  retired in K8-6b. There is no copy and no test-only scorer. *(2026-11-08, superseded: "The
  trainer's validation scorer stays separately written, on purpose (the `scorer-independence`
  contract)" no longer holds. That scorer retired with the ShepherdGNN training path in K8-7, with
  the contract and the test it protected; a finetuned task's validation scores through the same
  `src/models/tasks/scoring.py`, and the oracle tests are hand-computed.)*
- **The served ranking was measured through serving's own method.** Mode D called
  `DiagnosisPipeline.rank()`, which `run()` itself calls (removed in K7 with Mode D). There was no
  second candidate construction, and Modes A and B are not revived.
- **Evidence never touches the ranking.** It takes an immutable result and returns records.
  The task scorers (`src.models.tasks`) never import `src.reasoning`: the layers contract keeps
  `src.models` below it. *(2026-11-08: `src.inference.scoring`, which this rule named, was removed
  in K8-11.)*
- **A value that could not be computed is a status, never a number**, in every result kind, export
  and measurement.
- **Dependency versions**: the formal deployment installs the currently verified set, its versions kept
  in one central place, not scattered across entry points or tests; an upgrade of PyTorch, CUDA or PyG
  goes through the same deployment, loading and training flow and becomes the new baseline once verified, with no second pipeline for old environments; a new
  environment records its own evidence and does not inherit an older acceptance; a model is never
  refused only because a package version string differs: loading follows the model contract and the
  compatibility checks it needs (BACKLOG 50).
  The pinned set is today's formal deployment set, not a permanent limit on PyG, torch or CUDA: the
  versions live in `pyproject.toml` and `uv.lock` only, and no training, inference or UI code checks
  a package version (deployment guide §4.15 lists what an upgrade changes). No general package
  manager and no matrix of historical combinations.
