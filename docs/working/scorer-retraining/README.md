# Paper-aligned two-stage training — reference recipe, model bundle and checkpoint contract

**Status: design revision 2 (2026-10-27), on `claude/paper-aligned-two-stage` from `main` `0e64666`;
the A + B integration is being built on this branch (§8).** Revision 1 (`5d0327a`) was reviewed by
the reviewer. That review's R1–R3, the owner's later decisions and the reviewer's decision record of
the same day are folded in below. Until 2026-10-27 this file was "Scorer retraining — scoping only"
(moved from `../task-scope/README.md` §6); what of that scoping still holds is §9, the rest is in git
(this file at `0e64666`). **One folder, one README** stays the rule for the design; the K0 entry
map beside it ([`ENTRY_MAP.md`](ENTRY_MAP.md), §10) is an inventory of the code, not design text.

**Who decided what.**
- **The owner:**
  - the direction: the paper's two stages and three tasks replace the disease cosine/SP mixture as
    the method; infrastructure is reused, not duplicated;
  - the branch name, the three-axis naming (§5) and the version policy (§0.2);
  - **parameters that need experiments start from sourced provisional defaults and never block the
    build** (§0.3);
  - **one switch:** when this work is delivered, active training, inference and evaluation use only
    the new contract (§4, §7);
  - the old Mode C/D comparison is **archived**, not run again and not a prerequisite;
  - the path-reasoning fallback ranking retires. PathReasoner stays as post-ranking evidence (§1);
  - patients-like-me is offered on the Diagnosis page in this round (§8);
  - OrthologGate is replaced by shared components, keeping cross-species reasoning as a required
    future capability (§3.3);
  - the system has never been deployed at the hospital: it is a research platform. Merging into
    `main` accepts an engineering version; it is not a hospital deployment, a clinical switch or a
    clinical performance approval.
- **The reviewer (Codex):** the answers RD1–RD11 to this project's verification, the review of
  `5d0327a` (R1–R3) and the decision record of 2026-10-27. They are adopted for this blueprint as
  **reviewer recommendations, not institutional approval**. Review decisions are cited "RDn"; the
  rows of §2 have their own letters, so a row number never names a review decision.
- **Not decided here:** clinical adoption. It needs its own effect evidence and the institution's
  procedure (`../../DISEASE_SCORER_POLICY.md`, rev 12).

**Why.** The owner and the reviewer found that the paper's SP is nothing like this project's SP, and a
wider check followed. This repository's model differs from the paper's at every layer:
- no KG pretraining stage, and the link-prediction loss is dead;
- a heterogeneous GAT (v1) with fixed random features instead of a homogeneous GATv2 over a learned
  node table;
- a masked mean of phenotype vectors instead of the patient encoder;
- a cosine/contrastive objective instead of the tasks' NCA and multi-similarity losses;
- no gene or patients-like-me task.

So this phase rebuilds the model and its objectives. It does not polish the interim scoring.

---

## 0. Sources, evidence levels, versions and parameters

### 0.1 Sources

| Level | Source | Cited as |
|---|---|---|
| Paper | Alsentzer et al., *npj Digital Medicine* 2025, Methods ([article](https://www.nature.com/articles/s41746-025-01749-1)) | "Eq n" |
| Fixed source | `mims-harvard/SHEPHERD` at `e95433aa550a65e1c4e398c2436ffd1c7cf8abc7` | `up:<path>:<line>`; paths under `shepherd/` unless they start with `data_prep/` or are at the root |
| Its dependencies, at its pins | torch 1.8.0 + cu102, torch-geometric 1.7.2 (`up:install_pyg.sh:1-5`); pytorch-metric-learning 0.9.98; allennlp 2.4.0 | package and file |
| This repository | `main` `0e64666` | `path:line` |

**Evidence boundary.** The released checkpoints, the original KG and the distance matrices were not
downloaded, and no published number was reproduced. So the recipe below is **a reference recipe
anchored on the fixed source**. It is not a proof that this commit is the complete recipe of every
published result (§2.7). The released scripts cannot run gene augmentation as committed (F10), which
is further evidence that the scripts are not a proven executed configuration.

### 0.2 Version policy (the owner, 2026-10-27)

- **The reference baseline runs on the current stack and hardware**: torch 2.10.0 (CUDA 13.0) and
  torch-geometric 2.8.1 (`pyproject.toml`, BACKLOG 50), on the GB10. It uses model settings as close
  to the paper's as possible.
- **Nothing is downgraded or emulated.** No old PyG, torch or CUDA is installed, and no translation
  layer reproduces an old version's numerics or parameter shapes. At the project's start the
  hospital was told that the upstream's packages cannot run on the new hardware, and that different
  versions give slightly different numbers.
- **Where an old default carries meaning, the spec states the meaning** and records the difference
  (§2.8, the norm). It does not chase the old implementation's numbers.
- **Consequence:** results are compared with the paper as a reference, never claimed as an exact
  reproduction.

### 0.3 Parameters (the owner, 2026-10-27)

- **A value that needs experiments is a sourced provisional default, never a blocker.** It starts
  from the paper or the fixed source, is a formal, configurable field, and goes into the model spec
  and the effective config with its source and "not calibrated on this data". Experiments change it
  later.
- **The paper's datasets differ from ours, and some cannot be obtained.** This round wires the three
  tasks end to end on synthetic functional fixtures (§2.6, §8). Formal data and effect studies are
  package C.
- **Definitions and boundaries are not parameters.** Exact distances, non-scorable states, tie
  rules, multi-positive metric formulas and the graph view are written and tested in this round. No
  undefined behaviour and no placeholder number stands in for a finished computation.

---

## 1. The design

**Two stages.**
1. **Base:** KG link-prediction pretraining over all relations of the reference graph view (G1),
   selected on validation (§2, `../PLAN_TEST_RESULTS.md` §8).
2. **Finetune:** from one exact, selected checkpoint, one to three tasks, each an independent child
   run:
   - **disease:** a patient's phenotypes → all diseases of the view;
   - **gene:** phenotypes plus a candidate gene list → the candidates ranked;
   - **patients-like-me:** phenotypes → a registered reference cohort.

**Children never chain.** Choosing three tasks starts three children from the same parent; no task
feeds another. The "finetuned base" of a task is simply that task's first finetune.

**Operations.** `new`, `resume`, `extend`, `branch` and `finetune` are distinct operations, never one
flag read in different ways.
- `resume` and `extend` keep their contracts (`../PLAN_PROVENANCE_CONTRACT.md`, M3a).
- `branch` keeps architecture and data (BACKLOG 23), starts a new optimizer and records every
  difference. Its closed list of changeable settings is built in this round, before any variable
  comparison. An extend never carries a seed or learning-setting change.
- `finetune` is its own operation and page. It never borrows resume or branch to get past a parent
  check.

**Parents and the family tree.** The original base and its qualified descendants may all be
finetune parents. The tree sits under the actual parent checkpoint. Run directories stay flat (no
nesting, no copies). One reader serves the tree and the tag filter; tags stay non-authoritative.

**Evidence paths stay; the fallback ranking goes.** PathReasoner is kept as the post-ranking
evidence for a ranked result: it never decides which candidates exist and never scores them. It is
this project's deliberate extension of the paper's shortest-path neighbourhoods (Fig. 4, drawn
offline there), required by the hospital so that a pattern the model generalised to still comes with
"a possibly similar pathogenic route" to inspect. The paper's own interpretability outputs are the
phenotype attention weights (which the paper says are not a clinical interpretability tool) and the
disease-category summary of the disease task. The path-reasoning **fallback ranking** retires: with
a graph and no model, diagnosis is refused, not answered by path scores.

**Settled; not to be asked again:**
- **Selection and lineage:**
  - base → validation selects an exact checkpoint → one to three independent finetunes;
  - the dataset fingerprint and the seed are comparison conditions, not a family id;
  - the same generation or parent does not by itself prove a single-variable difference.
- **Scope:**
  - modularity does not mean building the general plug-in framework now;
  - installed, selected and actually used stay distinct.
- **The old comparison:** Mode C/D is archived. It is not marked as passed, nor as scientifically
  meaningless; the owner no longer requires it. Mode D is not re-pointed at a new head. Its
  executors retire once the new task measurement and serving are wired (§7).
- **No raw-cosine B-1** is built.

---

## 2. The reference recipe

**Rules (RD1).**
- **The anchor:** the recipe is anchored on the fixed source and written in the paper's notation.
- **Conflicts:** where the paper's text and the source conflict, it follows the source, the only
  executable statement, and lists the paper's text. A source behaviour that is a defect, or a
  mechanism the paper states and the source does not implement, is the exception (§2.4).
- **Corrections:** obvious defects are corrected and listed. A configuration with corrections is
  never labelled identical to the source.
- **One recipe:** one recipe, not a switch per row. Every value is a provisional default (§0.3).

The rows are grouped:
- **§2.1:** the paper and the source agree (A);
- **§2.2:** confirmed conflicts (C);
- **§2.3:** the paper is silent or ambiguous; the source decides (U);
- **§2.4:** source defects corrected, and paper statements the source does not implement (F);
- **§2.5:** the graph view and data protocol (G);
- **§2.6:** the evaluation protocol (E);
- **§2.7:** not verified (N);
- **§2.8:** the norm semantics and the inference rule.

### 2.1 The paper and the source agree

| # | Paper | Fixed source | Adopted | Deviation | Small verification |
|---|---|---|---|---|---|
| A1 | Three-layer GAT, citing GATv2 (ref 106) | Three `GATv2Conv` layers (`up:node_embedder_model.py:6,81-85`) | Same | — | The built encoder has three GATv2 layers |
| A2 | DistMult decoder with tanh; hinge loss, margin 1 | `s = tanh(Σ h⊙r⊙t)` (`up:decoders.py`, `up:node_embedder_model.py:201-211`); loss `mean(max(0, 1 − (s⁺ − s⁻)))` (`:416-425`) | Same | — | Regression 1 (§8) |
| A3 | Edge split 80/10/10 | Reverse edges in the same part (`up:data_prep/construct_kg/prepare_graph.py:57-72,148-172`) | Same, on the view of G1 | — | Regression 3 |
| A4 | — | Base validation/test message passing over training edges only (`up:pretrain.py:68-70`). A training batch removes the edges between its source and target pairs, both directions (`up:samplers.py:177-193`) | Same for the base; finetune graphs in G4 | — | Regression 3 |
| A5 | Pretraining hyperparameters selected on validation micro-F1 | Checkpoint monitor `val/node_total_acc` (`up:pretrain.py:97`). Accuracy and `f1_score(average='micro')` come from the same thresholded predictions (`up:utils/pretrain_utils.py:118-127`), so they are **one number** | Select on LP accuracy (that micro-F1): `sigmoid(raw) > 0.5`, i.e. raw score > 0 (`up:node_embedder_model.py:246-249`, `up:hparams.py:27`). Also report ROC-AUC and AP (RD7) | — | `y=[1,1,0,0]`, `ŷ=[1,0,0,1]`: accuracy 0.5, micro-F1 0.5 |
| A6 | Patient encoder: Transformer (search range includes 0 layers) → attention pooling → projection | 3 layers, 8 heads (`up:train.py:67-68`); `TransformerEncoderLayer(d_model, nhead)` with torch defaults (`up:task_heads/patient_nca.py:51-52`, `gp_aligner.py:63-64`); a padding mask, no length cap | Same; the layer settings pinned (U3). Sequences are padded to the batch maximum with a mask, never truncated | — | Shape and mask test; a padded position gets zero attention |
| A7 | Disease: `−‖z_d − z_P‖²` with NCA | 1000 random candidate diseases plus the truths per training batch (`up:hparams.py:101`); inference over all diseases (`:102`) | Same for training. Validation: U6 | — | Regression 1's NCA case |
| A8 | Gene: `η·EMBSIM + (1−η)·SPLSIM`; multi-similarity loss; gene augmentation; upsampling of rare candidates | Same structure | See C8, C9, U12–U14, F9, F10 and §6 | — | Package B regressions |
| A9 | Patients-like-me: `−‖z_Pi − z_Pj‖²` with NCA; positives share a causal gene | Self excluded (`up:patient_nca_model.py:95`) | Same; self excluded by stable case identity (E5); batches per F12 | — | Self-exclusion test |
| A10 | Finetune: the GNN keeps training; new task head; new optimizer | The pretrained node model is loaded and the task head is new (`up:train.py:267`; `up:patient_nca_model.py:44-50`; `up:gene_prioritization_model.py:52-63`) | Same | — | Regression 2 |
| A11 | SP is a fixed KG-derived feature | Fixed bounds, not a per-candidate-list min-max (`up:data_prep/shortest_paths/add_spl_to_patients.py:18-19,104-109`); never updated by the optimizer | §6 | — | §6 |

### 2.2 Confirmed conflicts (the paper's text and the source differ)

| # | Paper | Fixed source | Adopted | Deviation | Small verification |
|---|---|---|---|---|---|
| C1 | The final layer averages its heads | The final layer concatenates. Relation vectors and the patient head are `output × heads` wide (`up:node_embedder_model.py:85,90`; `up:patient_nca_model.py:50`): 128 × 4 = 512 in the scripts | Concat (RD2). **The width comes from the model spec, never from a head.** | Follows the source: the scripts' widths depend on it. Released checkpoints not read (N1) | Spec → built shapes: encoder out = LP head width = patient head input |
| C2 | Norm and nonlinearity at every layer | After layers 1 and 2 only: conv → LayerNorm → leaky ReLU (slope 0.01) → BatchNorm → dropout 0.3. The last layer has nothing (`up:node_embedder_model.py:93-107,135-143`; `norm_method="batch_layer"`, `up:hparams.py:25`) | The source's placement; the norm semantics in §2.8 | Follows the source | Regression 4 |
| C3 | Eq 20: `λ·L_DX + (1−λ)·L_LP` | Computes `λ·L_LP + (1−λ)·L_task` (`up:gene_prioritization_model.py:184,207`; `up:patient_nca_model.py:147,179`). Script λ: 0.4 gene, 0.9 disease, 0.7 patients-like-me | **Two named weights, `task_loss_weight` and `lp_loss_weight` (RD9).** Starting values, the scripts' flipped (task/LP): gene 0.6/0.4, disease 0.1/0.9, patients-like-me 0.3/0.7. η is separate | Names remove the inverted λ | Hand-computed `0.1·L_task + 0.9·L_LP` |
| C4 | Eq 5 averages over edges | Each epoch every node is a source once (`up:samplers.py:167-173`). The sampling adjacency has a self-loop added to every node (`up:samplers.py:110-111`); a one-step random walk then draws among the d real neighbours and the self-loop, each with probability `1/(d+1)` (`:203`). The drawn pair becomes supervision only through the real edges between the two nodes in the partition (`up:utils/pretrain_utils.py:98-115`), one positive per such edge; the artificial self pair matches none | **The source's support set:** each source visited once per epoch, drawing among its real neighbours and the artificial self-loop. **The artificial self pair is never a positive; it is counted** (F6). Source visits, drawn pairs, LP positives and optimizer updates are counted separately | Listed difference from Eq 5; an epoch counts source visits as in the source, not the same number of positives | One source with one neighbour: enumerate both draws (real edge → one positive; self → none, counted); an isolated source: skipped and counted |
| C5 | Eq 7–9: two nested leaky ReLUs in a projection | One leaky ReLU (slope 0.1, `up:hparams.py:96`), no outer one (`up:task_heads/gp_aligner.py:46-56,81-86`; `up:task_heads/patient_nca.py:40-46,69-70`) | The source's single leaky ReLU | Follows the source | Module structure test |
| C6 | Training uses early stopping (the paper calls its training times upper bounds) | None in either stage. The full schedule runs and the best validation checkpoint is kept; `ReduceLROnPlateau` patience 1000 exceeds 200 epochs, so the rate never changes (`up:pretrain.py:97,109-117`; `up:train.py:283-292,311`; `up:hparams.py:38`) | **The source:** the full schedule (200 base, 100 finetune), the best validation checkpoint restored, early stopping **disabled** (not a large patience). The existing early-stopping callback stays an option. Constant rate, no scheduler | Follows the source; the paper's statement is listed | The run reaches max epochs; the restored checkpoint is the best validation epoch |
| C8 | Eq 15: multi-similarity without mining | α 1, β 20, γ 1, plus pair mining with margin 0.4 (`up:hparams.py:90-93`; `up:utils/loss_utils.py:123-134`) | The source. **Its effect is not assumed small** | Follows the source | Hand counterexample: positives [0.1, 0.9], negative [0.4]. Mining keeps 0.1, drops 0.9, keeps 0.4; the loss goes from 1.518369 to 1.241154 |
| C9 | EMBSIM `tanh(z_Pᵀ W z_g)` | On `F.normalize`d vectors, plus a bias (`up:task_heads/gp_aligner.py:95-101`) | The source | Follows the source | Hand-computed two-vector case |

C7 (gene augmentation) moved to F10: the source cannot run it as released.

### 2.3 The paper is silent or ambiguous; the source decides

| # | Paper | Fixed source | Adopted | Deviation | Small verification |
|---|---|---|---|---|---|
| U1 | — | Homogeneous message passing: node and relation types are ignored; relations appear only in the decoder (`up:node_embedder_model.py:121,126`) | A homogeneous view built from the same bound data as the heterogeneous one | — | The reference encoder receives no type or relation tensor |
| U2 | — | A learned table `nn.Embedding(N, 2048)`, default N(0, 1) initialisation, no external features (`up:node_embedder_model.py:62,117`); relation vectors `xavier_uniform_` with the leaky-ReLU gain (`:114`) | Same, width in the spec. **Transductive:** a new KG version needs a new base | — | Initialisation rule in the built module; table gradient in regression 2 |
| U3 | — | **Base:** `nfeat` 2048, `hidden` 256, `output` 128, 4 heads, dropout 0.3, lr 1e-4, wd 0 (`up:run_pretrain.sh:6-17`); widths 2048 → 2048 → 1024 → 512; 512 sources a batch; neighbour sizes [15, 10, 5]; 200 epochs; gradient clip 1.0; seed 1 (`up:hparams.py:18-44`). **Finetune:** batch 64 patients; 100 epochs; gradient clip 1.0 (`up:hparams.py:95-110`); seed 33 (`up:train.py:78`); first-hop size gene −1 (all), disease 15, patients-like-me 15, then 10, 5 (`up:hparams.py:74`; the `run_*.sh` scripts); `upsample_cand` gene 2, disease 3, patients-like-me 1; `sparse_sample` gene 100, disease 300, patients-like-me 300. **Transformer layer:** d_ff 2048, dropout 0.1, ReLU, post-norm, no final norm (torch defaults of `TransformerEncoderLayer`). **Slopes:** encoder leaky ReLU 0.01, projection 0.1, GATv2 attention 0.2 | All pinned as provisional defaults | Not checked against the released checkpoints (N1) | Spec values round-trip into the built module |
| U4 | — | `GATv2Conv` defaults at PyG 1.7.2: `concat=True`, attention dropout 0, `add_self_loops=True`, `share_weights=False`; no residual | Set explicitly in the spec; current PyG has the same parameters, and its newer `residual` option is pinned off | — | The built layers carry them |
| U5 | Negatives by shuffling same-type targets within the batch | Negatives by edge type: for each positive, a tail drawn with replacement from the same relation's positives in the batch (`up:node_embedder_model.py:150-170`; `up:utils/pretrain_utils.py:35-38`) | The same scheme, without the defect F1 | — | Regression 3 |
| U6 | — | Disease validation candidates: 1000 random diseases plus the kept truths (`up:train.py:173`), so not exactly 1000 per case | **All diseases of the view, computed exactly in chunks, no ANN (RD3).** Truths, ties and denominators per §2.6. Training negatives stay sampled | The same universe as the formal ranking | Chunked ranks equal one-shot ranks |
| U7 | Finetune LP "focuses on" gene, phenotype and disease relations | LP seeds are gene, phenotype and disease nodes, with neighbours of any type. So 18 of the upstream KG's 30 directed relations can be drawn, `gene→pathway` and `gene→GO` among them (`up:utils/pretrain_utils.py:98-115`; `up:samplers.py:365-387`) | **The source's seed-type rule. The relation list is derived from the view and recorded, never hard-coded (RD8).** Narrowing to the three types' induced subgraph is a separate research choice | — | The derived list on a toy KG |
| U8 | — | Finetune: Adam over the whole model (GNN, node table, relation vectors, head), one learning rate (1e-5 gene and disease, 5e-5 patients-like-me). No weight decay, no scheduler, clip 1.0. The upstream `--resume` restores weights only (`up:gene_prioritization_model.py:49-55,541-543`; `up:train.py:236-248,308-320`) | The starting values. A new optimizer for every finetune child | — | Regression 2 |
| U9 | Patient-driven sampling: m patients with their phenotypes and genes, plus n random gene, phenotype and disease nodes | Seeds and LP sources are every node of the batch patients (phenotypes, candidate genes including upsampled ones, causal genes, true diseases), the task candidates (disease: 1000 random diseases plus truths), patients-like-me's added same-gene patients, and `sparse_sample` random gene/phenotype/disease nodes (`up:train.py:156,167-168,181-184`; `up:samplers.py:348-414`) | The same seed set and sizes (U3) for training. Augmentation neighbours join the seeds only when augmentation is on (F10). A disease sample without candidate genes gets none invented. Evaluation embeddings follow §2.8 | — | Batch composition test |
| U10 | Eq 18–19: `−‖·‖²` on the projected vectors | pml `LpDistance(power=2)` normalises by default (pml 0.9.98 `distances/base_distance.py:7-8`). So for **non-zero** unit vectors `−‖a−b‖² = 2cos(a,b) − 2`. The softmax scale is 1 **in the selected configuration** (`up:utils/loss_utils.py:181`) | Normalisation stated in the spec, not left to a library default: L2 with ε 1e-12, so a zero vector stays zero. The cosine identity is not relied on for zero vectors | — | Hand-computed two-patient, three-disease NCA |
| U11 | Eq 6: attention pooling with one vector | allennlp 2.4.0 `BilinearAttention`: `aᵀWx_p + b`, then softmax | The source's form `aᵀWx_p`; `b` dropped. It cancels in the softmax and gets zero gradient, so forward and updates are unchanged. **The paper's and the source's parametrisations agree in the forward pass; their optimisation paths are not guaranteed equal.** | — | A three-phenotype example's weights |
| U12 | η searched over [0.1, 0.9] | η is `alpha`: default 0 (SP only), 0.5 in the gene script | Starting value 0.5. η is the gene score's mixing weight, unrelated to the loss weights | — | §6 |
| U13 | Eq 7–9 write one projection pair `W1, W2` for patients, genes and diseases (notation, not stated weight tying) | A separate two-layer tower per entity (`up:task_heads/gp_aligner.py:46-54`; `up:task_heads/patient_nca.py:40-44`) | The source's separate towers | — | Distinct parameter sets per tower |
| U14 | Add the k rarest candidate genes to each patient's list | Upsampling pool = all KG genes (each count starting at 1), `upsample_cand` genes per patient, excluding the patient's causal and existing candidates; no upsampling at validation or inference; the counter lives per worker per epoch | The source's pool and sizes; ties broken by gene id; **one counter in the main process, reset at each epoch start, saved in the resume state** | Determinism and strict resume need the counter's scope | Counter round-trips through a resume |
| U15 | — | Both stages train in fp32: `precision: 16` in `up:hparams.py:114` is never passed to the trainer (`up:pretrain.py:109-117`; `up:train.py:308-317`) | **fp32 for the reference** (AMP off), with dtype, AMP, TF32 and compile recorded in the effective config. AMP stays an option on the same Trainer, as a non-reference configuration; no new ban on HGT + AMP | — | Effective config records fp32; fp32 is not a bit-exact promise across hardware |

### 2.4 Source defects corrected, and paper statements the source does not implement

| # | Paper | Fixed source | Adopted | Deviation | Small verification |
|---|---|---|---|---|---|
| F1 | — | A negative may equal its positive (`up:node_embedder_model.py:165`, "currently possible"). A relation with one positive in a batch has hinge 1 and zero gradient | A negative never equals its own positive. A positive with no other same-relation tail in the batch adds no LP term and is counted. Other true edges are not filtered out, so a corrupted negative is not claimed to be a proven non-relation | Self-negatives train nothing | Regression 3 |
| F2 | — | The split is unseeded (`up:data_prep/construct_kg/prepare_graph.py:59`, `random.shuffle`). Regenerating the same split is not reproducible; a saved split file is | The split is made once on the view with a recorded seed and saved as an artifact with its digest. Every stage records the digest, and resume refuses a different one | Traceability | Same seed and view give the same digest |
| F3 | Validation edges that exist, plus negatives from the negative-sampling scheme | Validation positives (one random walk per source) and negatives are redrawn at every evaluation | **Every validation edge evaluated once,** in a seeded fixed order and fixed batches, against negatives drawn by U5's rule within each batch under a recorded seed, excluding the positive's own tail (RD7). The change from node-uniform to edge-uniform weighting (degree and relation mix) is recorded | Selection must not be noise | Two evaluations of the same weights give identical numbers |
| F4 | — | Metrics averaged per batch | Whole-set totals: correct over total; AUC and AP on the pooled scores; a class missing makes AUC/AP undefined, said so | Batch size must not weigh cases | Unequal batches, hand-computed |
| F5 | — | Finetune does not remove the predicted edges from message passing (`filter_edges=False`, `up:hparams.py:94`; `up:train.py:158`); pretraining does | **This project's leak rule, in both stages:** the step's target edges and their reverse and duplicate representations leave that step's message-passing graph (RD8) | Not a copy of the source: its finetune does not filter | Regression 3 |
| F6 | — | The artificial self-loop of C4 can be drawn by any source, with or without real neighbours; it matches no real edge | Never a positive, counted, in the base and the finetune samplers (R1). Removing **native** self-loop edges from the view (G1) is a different step; GATv2's own self-loops are a third use and stay | — | C4's enumeration test |
| F7 | — | The NCA drops a patient with no positive among the candidates silently (`up:utils/loss_utils.py:255`) | The truths are always among the candidates. A patient whose truth is not in the view is refused at data binding and counted | Nothing silently dropped | Data-binding refusal test |
| F8 | — | SP: unreachable stored as 0, bounds hard-coded, held-out edges included | §6 | §6 | §6 |
| F9 | Eq 15 over all of a patient's causal genes; patients-like-me positives share a causal gene | Each patient's causal genes are truncated to the first for gene labels, ranks and patients-like-me labels | **All causal genes are positives** (gene loss, metrics); a patients-like-me pair is positive when the causal-gene sets intersect. Multi-gene patients are counted | Follows the paper | Two-gene patient oracle |
| F10 | Gene augmentation after the projection: neighbours by shared phenotypes, normalised over the K neighbours (Eq 10), with a degree gate | Before the projection; needs a similarity table no released code produces and a degree file on the authors' machine; with the default weight 0.7 the scripts fail at the first batch; normalises over the candidate axis, not the K neighbours (`up:gene_prioritization_model.py:85-96,303-309`; `up:train.py:63-66`) | **The paper's Eq 10:** after the task projection; K = 3 and ω = 0.7 as provisional defaults; weights normalised over the K neighbours with padding outside the denominator; a gene with no valid shared-phenotype neighbour keeps its own vector and records that state (no division by zero, no ε neighbour); the same degree gate in training and inference. The neighbour table comes from the view's gene–phenotype relations, self excluded, ties stable, never from test truths; the graph and relations used for degree are recorded with it | Follows the paper; the released code is a defect, which does not prove the paper's experiments ran the broken code | Two-gene hand example; zero-neighbour gene |
| F11 | Patients-like-me batches guarantee each patient at least two positives | Not guaranteed | **The paper's rule:** a batch anchor has at least two other patients sharing a causal gene, never itself duplicated. A group too small for that does not anchor; it may serve as negatives; it is counted. No other loss is substituted | Follows the paper | Three same-gene patients plus negatives; a singleton group counted |

### 2.5 The graph view and data protocol

| # | Paper | This project | Adopted | Small verification |
|---|---|---|---|---|
| G1 | The KG has duplicate and self-loop edges removed and keeps only its largest connected component | The KG builder does neither | **A reference graph view**, an identified model input; the workspace bytes stay unchanged. Order: fix the included nodes and relations; drop native self-loops; deduplicate by relation semantics (different relations between the same nodes stay distinct; reverse mappings stay consistent); take the largest connected component of that full undirected view with a deterministic tie rule; only then cut the edge split. Recorded: the source graph digest, the view rule and result digests, the stable node mapping (shared by every consumer), and the excluded counts. A disease or gene outside the component cannot be ranked by this configuration; it keeps its id and reason and stays in coverage denominators. Other configurations may use several components | Dedup, self-loop and component counts on a toy graph; a stable mapping round-trips |
| G2 | Gene–phenotype edges added by closing disease–phenotype(+) with disease–gene (`up:data_prep/construct_kg/prepare_graph.py:121-132`) | Gene–phenotype edges from HPO `genes_to_phenotype` | A recorded data difference. HPO derives them in a similar way, but sources, versions and term granularity differ, so the edge sets are not claimed equal | — |
| G3 | A disease–phenotype(−) relation ("does not have") | HPO `NOT` annotations are skipped (`src/data_sources/hpo_annotations.py:176-178`) | A recorded data difference; no negative edge is ever traversed as support | — |
| G4 | — | — | **Which graph each stage passes messages over:** base training uses training edges with F5; base LP validation and test use training edges. Finetune training uses training edges with F5; **finetune validation, task tests and serving use the full view** (the source uses all edges, `up:train.py:156,167-168,183`; `up:predict.py:57`). LP figures computed after a finetune's full-view or SP exposure are never called held-out. Each stage's graphs go into `graph_exposure` | The exposure record per stage |
| G5 | — | Cross-species relations (orthologs, animal phenotypes) are optional data (`src/kg/builder.py`; off in `scripts/build_knowledge_graph.py:270`) | Carried by the same view (§3.3). Turning them off forms a view without them, before the model metadata is built; it never empties an edge list inside a built model | §3.3 tests |
| G6 | The paper's KG: 7 node types, 15 relation types | Built from this project's sources | Relations are derived from the view, never hard-coded; the KG content is a data difference | — |

### 2.6 The evaluation protocol (the reviewer's decision record, Q2)

| # | Rule |
|---|---|
| E1 | **Pooled positive-pair reading**, the source's (`up:utils/train_utils.py:17-24`): MRR = mean of 1/rank over every (query, positive) pair; top-k = share of those pairs with rank ≤ k. It is the validation selection metric of every task, named with its aggregation, candidate set and tie method. Disease: all diseases of the view. Gene: each case's own list, no upsampling, with the serving SP and augmentation. Patients-like-me: each validation patient against all other validation patients (a recorded deviation from the source's in-batch ranking) |
| E2 | **Macro recall@k per query**, the paper's patients-like-me reading: relevant patients found within the top-k threshold divided by all relevant patients in the reference set, averaged over eligible queries; reported beside E1 in its own column, likewise for genes. A count over k is precision@k, never called recall |
| E3 | **Ties by average rank** (scipy `rankdata(method='average')`, as the source). The screen may order ties by stable id; a tie straddling k uses the average-rank threshold and is not claimed equal to the k rows shown. Near-equal scores are never merged into ties |
| E4 | **States kept apart:** no other positive in the reference set; no reliable truth; input excluded; model run failed. The first two never fabricate a recall; every conditional metric names its denominator |
| E5 | **Coverage and a conservative reading:** for evaluation units fixed in advance with an assessable truth, a unit the model could not rank contributes 0. A query without positives is never silently a "recall 0". Post-import figures are never called end to end. Self-exclusion uses stable case identity, also across query and reference lists |
| E6 | **Contract errors still refuse:** no exception is swallowed to complete a whole-cohort figure |
| E7 | **Oracles:** positive ranks [1, 10] give pooled MRR 0.55 and best-positive MRR 1; 20 relevant patients with 3 in the top 5 give recall 3/20 and precision 3/5; two patients with different positive counts; an average tie straddling k; a query with no other positive |

### 2.7 Not verified

None of these blocks the build. Each is stated, not guessed.

- **N1.** The hyperparameters of the released checkpoints.
- **N2.** Which SP matrix the paper's experiments used. `shortest_paths.py` writes a `_noGO` file
  name, while `add_spl_to_patients.py` reads the other file by default, and its row order needs every
  node to appear on some edge.
- **N3.** Whether the original KG already holds duplicate reverse edges.
- **N4.** Whether this commit is the complete recipe of every published number (§0.1), and what
  augmentation setting the published runs actually used (F10).
- **N5.** Answered (R1): a source without a validation edge draws its artificial self-loop, which
  matches no edge, so it contributes nothing. F3 replaces that path either way.
- **N6.** How much the graph-mode norm (§2.8) differs between sampled training and full-view
  inference. The source has the same split.
- **N7.** Whether the source's gene training aligns SP columns with union embeddings: union
  embeddings follow batch order while SP columns follow gene id, which suggests a misalignment.
  Unconfirmed by a run; this project joins SP by global gene id (§6) and tests it.

### 2.8 Norm semantics and the inference rule

- **The old semantics.** PyG 1.7.2's `LayerNorm` without a batch vector normalises across all nodes
  and channels of the forward pass, not per node. Its weight and bias have shape [1]; PyG 2.8.1's
  have shape [in_channels]. Both no-batch paths compute `(x − mean)/(std + ε)`; the ε placement
  differs only on the batched branch.
- **Adopted:** the modern graph-mode `LayerNorm` (per-channel affine) and `BatchNorm1d` (running
  statistics in evaluation), in the source's order (C2). The parameter-shape difference from the
  fixed source is recorded; no scalar-affine or legacy norm class is added (version policy). Per-node
  norm is a different configuration (§5 axis 1), not the reference.
- **Consequence:** a node's vector depends on the node set of the forward pass. So two forward rules,
  both part of the model spec:
  - **training** uses sampled subgraphs, as the source does;
  - **evaluation and serving** use layer-wise full-view inference in node chunks. Each chunk keeps
    every node's full neighbour set and its attention softmax denominator, so chunks are never
    induced subgraphs. Each layer's norm statistics are taken over all of that layer's nodes before
    the norm and the next transform. Dropout is off and BatchNorm uses its running statistics. The
    same encoder modules run in both rules; there is no inference-only copy of the GNN math.
  - Sampled-batch evaluation is not assumed equivalent. Memory is a resource target to verify on
    the GB10, not a design guarantee.
- **Verification:** regression 4 (cross-chunk edges, a neighbour shared across chunks, blocks with
  different means and variances, two chunk sizes, against the one-shot forward on a small graph).

---

## 3. The model bundle, the builder and the module boundaries

**Shared components first (the owner).** The pipeline is built from shared functions and modules to
lower maintenance, keep upgrades flexible and stay compatible with acceleration kernels. A node is
duplicated, or a parallel pipeline built, only where it is necessary.

### 3.1 One bundle, one builder

```
model spec (versioned, self-describing)
  → graph view (G1, G5)
  → input module (learned node table; optional type embedding)
  → GraphEncoder
  → LP head (one diagonal relation vector per directed relation of the view)
  → finetune only: patient encoder → task head (disease | gene | patients-like-me)
```

- **One module, one builder.** Every trainable parameter (task head and relation vectors included)
  is a submodule of one `nn.Module`. One builder in `src.models` builds it from the spec.
- **Every consumer goes through it:** the Trainer's optimizer, the checkpoint writer and reader,
  serving, measurement, `scripts/setup_demo.py` and `scripts/build_index.py`.
- **What it replaces:** `build_shepherd_model` (`src/models/gnn/shepherd_gnn.py:506`),
  `create_model_from_config` (`scripts/train_model.py:928`) and `build_index.py`'s own construction
  (`:169`). After the switch there is no second builder and no structural guess from weight names.
- **Shared parts across tasks:** the disease and patients-like-me tasks share the patient encoder,
  attention pooling, projection towers and one NCA loss; all three share the exact chunked scorer
  and ranker and one metrics module (E1–E7); the patient Transformer uses torch's
  `scaled_dot_product_attention`, whose backend torch selects. `src/models/attention/adaptive_backend.py`
  is not wired without review.

**Lifecycle hooks.** The Trainer stays the only trainer, and `scripts/train_model.py`,
`run_plan`, `checkpoint_payload` and `checkpoint_io` the only training and saving system.
- **The task module supplies:** batch interpretation, the forward pass, named loss terms (`lp`,
  `task`) and metrics.
- **The Trainer owns:** the optimizer, AMP, accumulation, cancellation and saving.
- **Hooks:** `build(spec)`; `transfer(parent, spec)`, which returns the weight-transfer record; the
  stage's sampler; `step`, which returns the loss terms; `validate`, which returns a report with its
  digests. A batch with no valid LP term is handled explicitly: no empty mean, no invented loss, and
  no optimizer update in an LP-only base step.

### 3.2 Shared math and the scorer-independence contract (RD5)

- **Where it lives.** Encoder, head, loss and scorer math lives in `src.models`; training, inference
  and evaluation import it. No second NCA, multi-similarity loss or metrics module exists for another
  entry.
- **The contract still holds.** `scorer-independence` (`.import-linter.ini`) only forbids
  `src.inference.scoring` from importing `src.training`.
- **What changes is the oracle.** Tests use hand arithmetic or a short, independently written
  computation. **Expected and actual never both call the code under test.**
- **No new dependency.** pytorch-metric-learning and allennlp are not added. The small tensor losses
  (NCA, multi-similarity with mining, the normalised Lp distance) live in the one loss module with
  hand oracles; ranking uses scipy, already locked; AUC/AP are whole-set statistics.

### 3.3 The GraphEncoder contract, and cross-species reasoning

- **In:** global node ids and node types; the view's relation mapping; message-passing edges kept
  separate from supervision edges; the sampled subgraph or the full view.
- **Out:** node vectors of the width the spec declares.
- **Declared:** the width; whether node types or relations are used; the sampling needs; supported
  dtypes and AMP; the forward rules of §2.8; the actual capabilities (an unverified compile
  combination is not claimed).
- **Layers are not hops.** The three-layer reference GATv2 needs three hops; a full-graph or
  global-attention encoder has other execution strategies.
- **A swap is not automatically one variable.** Width, head capacity, type information, sampling and
  initialisation that change with the encoder are listed as differences.
- **Inheritance.** A finetune child uses its parent's encoder spec.
- **Existing encoders** (the heterogeneous GAT, HGT and SAGE) move behind the same contract as
  non-reference options with the same input module and LP head. They get the minimal build,
  forward, gradient and save/reload checks; no effect study. HGT carries BACKLOG 20 (Edit 2 waits
  for the GB10), and `segment_matmul` has no fp16 or bf16.

**Cross-species reasoning replaces OrthologGate** (the reviewer's review of the proposal, accepted).
The requirement stays; the faulty module goes.
- **S1, the graph view:** cross-species nodes and relations, with their reverse mappings, are part
  of the view (G5). The view records which graphs message passing, LP supervision, SP and evidence
  each read; switching relations off forms a different view and model metadata.
- **S2, the input module:** an optional node-type embedding, moved from the existing
  `HeteroFeatureEncoder.type_embeddings`; species, node kind and source ids stay in the data, never
  guessed from a name. Off in the reference. No separate species embedding.
- **S3, relation-aware message passing:** each encoder's standard mechanism. HGT and the
  heterogeneous GAT/SAGE carry relation-specific parameters; GATv2 may take a relation embedding as
  `edge_dim` (an identified, non-reference configuration; self-loop edge features are explicit
  zeros; COO edges only). SAGE is not called attention, and no attention is called a calibrated
  cross-species trust.
- **Explanations:** PathReasoner's cross-species paths are the main aid; a GATv2 attention reading is
  at most an optional internal number; HGT gets no hook. Without HP↔MP mappings no such evidence is
  produced, and a path is never claimed as a proven causal route.
- **Removed with it:** the gate, the second message pass, its settings and UI, the unfed ortholog
  loss and the "X% influenced" claim.
- **Not claimed:** an excluded relation does not keep other-species nodes out of graph-norm
  statistics; "no cross-species information at all" needs those nodes out of the view.
- **Later:** real cross-species KG data and HP↔MP mappings, effect studies, attention views.

### 3.4 Five boundaries and their candidate acceleration routes

Each boundary states its tensor shapes, node and relation mapping, mask, dtype and device, gradient
and batch semantics. The routes are candidates, not plug-ins. The plug-in system stays its own
project (BACKLOG 58); these boundaries are where it will attach, and no empty slots are added.

| Boundary | What crosses it | Candidate routes |
|---|---|---|
| Sampling and transfer | node ids, edges, relation ids, masks | pyg-lib samplers. GraphBolt is a possible route, not a verified PyG drop-in |
| Graph encoder | node vectors | pyg-lib or cuGraph fused kernels (possible routes), `torch.compile`, AMP. **GAT, GATv2 and HGT attention is sparse, along edges: FlashAttention, xFormers and SageAttention do not apply.** Dense attention kernels would apply only to a future global-attention encoder |
| Patient encoder | phenotype sequences padded to the batch maximum, with a mask | Dense attention through `scaled_dot_product_attention` (FlashAttention and other backends chosen by torch). Sequences are short, so any gain is measured |
| Task decoding and scoring | patient and candidate vectors, scores | **Exact** search (chunked matmul) keeps the scoring contract. **ANN** can drop candidates, which changes results and metrics: it needs its own declared approximation policy |
| Loss and update | losses, gradients | AMP (non-reference), `torch.compile`, gradient checkpointing |

---

## 4. The checkpoint contract (summary)

The full design is `../PLAN_PROVENANCE_CONTRACT.md` §9. Every new checkpoint records:

- **`model_spec`:** versioned, and enough to rebuild the bundle, including the graph view's identity.
- **`learning_stage`:** `base` | `finetune`; **`task`:** none for a base | `disease` | `gene` |
  `patients_like_me`; **`operation`:** `new` | `resume` | `extend` | `branch` | `finetune`.
- **Lineage:** the exact SHA-256 of the parent checkpoint file and the parent run; the family id
  (the root run's id, inherited).
- **The weight transfer:** loaded, initialised new, trainable.
- **Data role digests and exposure:** the KG, the view, the edge split artifact, the task data
  artifacts and reference cohort; which graph each stage saw (G4).
- **The effective loss and scoring identity:** term names, weights, versions, and every provisional
  default with its source (§0.3).
- **Selection evidence:** the digest of the validation report that selected the parent.

`progress.stage` keeps its meaning (the `original` / `extension` schedule). An early-stopped or
weights-only base **with a model spec** may be a finetune parent; strict resume is not loosened.

**One switch (R2).**
- When the work is delivered, active training, inference and evaluation accept only the new
  contract. A checkpoint without a model spec is listed as history; the new system never runs it,
  never guesses its spec and never re-registers its reports.
- History stays as it is: checkpoints, reports, ledgers, registries, workspaces and fixed Git
  revisions are not overwritten or migrated. A registry or ledger that holds historical records is
  never refused as a whole; new records are grouped by explicit model, task and metric semantics.
- Old code and environments are preserved by fixed revisions and existing evidence, not moved into a
  `legacy/` tree as a second live pipeline.
- Intermediate commits may still hold a path not yet retired; the delivered batch completes the
  retirement list (§7).

---

## 5. Naming: three axes that are never mixed

1. **Configuration identity (what the model is):** derived from the spec, e.g. "GATv2 · reference
   recipe r2" or "HGT · recipe r3".
2. **Relation to the paper (for researchers):** computed from the spec against the reference recipe.
   It reads "matches the reference recipe", or lists the differing items. The reference recipe itself
   reads "anchored on `e95433a`, with the corrections of §2.4", never "identical to upstream".
3. **Clinical evaluation status (for doctors): distinct evidence, not an auto-promotion line.**
   - **Validation selection:** a selection step, not clinical validity.
   - **Registered test reports:** each with its cohort and protocol. Registration is not a pass, and
     one model may have several reports with different conclusions.
   - **Institutional approval:** carries a scope, and is never generated from a score or a
     registration state.

**Baseline is a role.** "Baseline" is a role given to the reference configuration in a comparison.
Non-reference configurations need no umbrella name: "research variant" is not used. A factor changed
for an experiment is carried by branch lineage.

**What doctors see.** The model selector filters by task, required inputs, applicable data and
axis 3, and shows the model's identity. It does not show baseline or variant. Scores are ordered only
within a comparable task, cohort and protocol. A base offers no diagnosis.

**Where it comes from.** All of it is derived from authoritative records (spec, registered reports,
approval records). Manual tags stay non-authoritative.

---

## 6. The gene task's SP

**The source's chain** (`up:data_prep/shortest_paths/*`; `up:samplers.py:442`):
- **Distances:** an undirected, relation-blind, uncapped BFS on the KG's largest connected component
  gives gene-to-phenotype distances, including held-out LP edges.
- **The mean:** `d̄` is the mean over the patient's mapped phenotypes.
- **The transform:** `normalize(d̄) = 2(d̄−1)/(6.84−1) − 1` is negated in the sampler. So
  `SPLSIM = 1 − 2(d̄−1)/5.84`: closer is higher, with no clipping.
- **Unreachable** pairs are stored as 0.

**The paper's formula.** Eq 13's printed `NORM(x) = 2(x−max)/(max−min) − 1` is a typo: its range is
[−3, −1]. The intended `2(x−min)/(max−min) − 1`, applied to `−d̄` with bounds [−6.84, −1], gives the
source's value. The paper does not say where max and min come from.

**Worked example** (two phenotypes, three candidates):
- **Distances** g1 = (1, 3), g2 = (2, 4), g3 = (5, 8); `d̄` = 2, 3, 6.5.
- **Three readings:** the source +0.6575, +0.3151, −0.8836; the printed formula on this list −1,
  −1.444, −3; a per-list min-max 1, 0.556, −1.
- **Mixed with EMBSIM** (η = 0.5, EMBSIM g1 = −0.8, g3 = +0.9): fixed bounds rank g3 first, a per-list
  min-max ranks g1 first. **The normalisation is part of the task's definition, not a display
  choice.**

**The first executable rule** (the reviewer's review of `5d0327a`, §5, and the owner's parameter
policy):
1. **Formula:** `S = 1 − 2(mean_distance − lower)/(upper − lower)`, aggregation mean, never a min-max
   recomputed per candidate list.
2. **Provisional defaults:** `lower` 1, `upper` 6.84, η 0.5, from the fixed source. They are formal
   settings on one computation path, written into the model spec, the effective config and the
   scoring identity, recorded as "upstream fixed reference, not calibrated on this data". 6.84 is a
   data constant of the source's simulated patients; using it is allowed, claiming it calibrated is
   not. The source's upper bound depends on the aggregation (mean 6.84, max 9, min 6.5, median 7;
   `up:data_prep/shortest_paths/add_spl_to_patients.py:104-109`).
3. **Bounds checked:** finite and `upper > lower`, or the job is refused before it starts. No ε is
   added for a zero range and no other formula takes over.
4. **No clipping,** as in the source: a value outside the nominal range is kept, and validation, the
   UI and exports never cut it back to [−1, 1].
5. **Exact distances:** for every gene that can enter a batch (candidates and upsampled genes) and
   every phenotype of the bound patients, through one shared exact distance interface joined by
   global gene id. Not necessarily a dense all-genes × all-phenotypes table. Unreachable and
   incomplete stay apart: 0 and `max_hops+1` are never distances.
6. **Connectivity, stated narrowly:** SP on the same undirected reference view (G1) is connected
   when both ends are in its component. Direction or relation filters, training splits, missing
   mappings, a hop cap or an unfinished computation are not covered by that guarantee.
7. **Non-scorable, never fabricated:** a case whose needed distances are unreachable or incomplete
   is listed as non-scorable with its reason and stays in the denominator (E4, E5). In training, a
   union gene with an undefined mean distance leaves that patient's negatives, and a patient whose
   causal gene is undefined leaves that step's gene loss; both are counted. A truncation penalty or
   partial coverage would be a separate, declared scoring policy.
8. **Calibration later uses the same formula:** bounds fitted on the training role go into the same
   computation, with the fitting method and the training-role digest recorded. No second scorer and
   no calibration service are built now.
9. **Changed bounds or η are a different scoring identity.** Strict resume inherits the settings;
   research changes follow branch rules. No old checkpoint, report or recipe is overwritten.
10. **Graph and exposure:** the gene task's SP is computed on the full view; a model finetuned with it
    has seen held-out structure (G4).

**Today's SP is neither.** Today's producer runs an undirected BFS from phenotypes up to 5 hops,
and reads `max_hops+1` both beyond the cap and unreachable. Today's disease SP term `1/(1+mean hops)`
is neither the gene SPLSIM nor a per-list min-max (`../../SP_SCORE_GUIDE.md` §3). *Superseded at
K4:* the plan to extend the producer and reader for exact distances gave way to the in-process
computation below; there is no distance file of either kind.

**As built (K4, 2026-10-27).**
- **Distances:** one function, `src.kg.graph_view.exact_distances`, computes the exact hops from the
  digest-bound view in process; training, measurement and serving share it, and the table's digest
  goes into the record. There is no distance file. The view is one connected component, so the
  distances are a pure function of the view. The file reader lives in `src.inference`, which the
  training layer cannot import. The hop-capped producer, its sidecar and its reader now serve only
  the retiring disease mixture, and retire with it (§7).
- **Distance cost:** one unweighted search per phenotype source, at every training start and every
  measurement. About 0.11 s a source on a synthetic 100k-node, 8M-edge graph (review of K4), so a
  few thousand case phenotypes take minutes. The GB10 acceptance times it on the real view. If it is
  too slow, the computed table becomes a derived artifact bound by digest, like the fixture.
- **Augmentation:** the paper's Eq 10–11 (`src.models.tasks.gene`): neighbours by the raw count of
  shared phenotypes, and the degree gate `θ = ω·exp(−ω·|N_g|) + 0.2`, with `|N_g|` the gene's degree in
  the view. The source's form `+ (1 − ω − 0.1)` equals it only at ω = 0.7.
- **Task data:** `scripts/derive_task_fixture.py` writes the functional fixture to
  `derived/gene_fixture/`, bound to the workspace's split manifest by digest
  (`src.kg.task_fixture`).
- **Training candidates:** in training each patient is scored against the union of the batch's
  candidate lists, upsampled genes included, with its own causal genes as the positives. That is the
  source's default (`only_hard_distractors` False, `up:gene_prioritization_model.py:127-140`).
  Validation, measurement and serving rank each case's own list.
- **Denominators:** a case's assessable positives are the causal genes it lists, in the view or not
  (F9, E5). The importer keeps the declared causal genes that name no single gene of the graph
  as their normalised identifiers (`causal_genes_unmapped`: stripped, upper-case, each once), so
  such a case stays assessable. A case declaring no causal gene is no reliable truth (E4). A task
  measurement in which no case binds, or a patients-like-me cohort with no declared causal gene,
  is refused before anything is written (E6).
- **Patients-like-me:** batches seed every member's phenotypes, causal genes, candidate genes
  (anchors upsampled, `upsample_cand` 1, U14) and disease (U9). The batch's anchors are the NCA
  term's only queries (F11); a partner is a reference patient. A pair is positive when the declared
  causal-gene sets intersect: the same gene of the graph, or the same normalised identifier
  outside it; no alias is resolved. A patient excluded at binding keeps its E4 state (no reliable
  truth, no other positive, or input excluded with each same-gene patient as a positive), and a
  ranked patient's excluded partners are relevant items it could not rank.
- **Served gene identity:** the served identity names the model's η and shortest paths in state
  `computed`; a `gene_task/v1` result names its SP bounds.

---

## 7. Retirement list

An inventory carried out as each part's callers move, not a questionnaire and not a tail job (§10
maps every entry). No retired part keeps a UI switch, a no-op flag, a command alias or an import
fallback.

| Part | Where | Replaced by / why |
|---|---|---|
| The served disease mixture (`served_mixture/v1`, `_served_score`, path-gated candidate discovery as a ranking gate, SP mixed into the disease score) | `src/inference/pipeline.py`; `src/core/types.py` | The disease head over every disease of the view (K3). SP stays as the gene task's feature |
| The path-reasoning fallback ranking (`path_reasoning/v1`) | `src/inference/pipeline.py`; `src/core/types.py` | Refusal when a graph has no model (owner decision). PathReasoner stays as evidence |
| Mode D (`run_mode_d`, the `run_mode_c` observer, `--mode D`, its probes) | `src/evaluation/mode_d.py`; `scripts/measure_scorer.py`; `scripts/probe_deployment.py` | Archived (owner decision); history read-only |
| Mode C's masked-mean raw-cosine recipe | `src/evaluation/measurement.py` | Task measurement through the same entry, with task, metric and scoring semantics (K3, K4). Shared ranking, metrics, readers, registry, report store, ledger and GPU lock are kept |
| Structural inference and old-checkpoint execution (`build_shepherd_model`, `resolve_arch_params`, `create_model_from_config`) | `src/models/gnn/shepherd_gnn.py`; `scripts/train_model.py` | The one builder and the model spec |
| `OrthologGate`, the edge-type split, `use_ortholog_gate`, `ortholog_gate_type`, the Training page toggle | `src/models/gnn/layers.py:248`; `src/models/gnn/shepherd_gnn.py:162-168,215-228,287`; `src/webui/components/training_console.py` | S1–S3 (§3.3) |
| Positional encoder | `src/models/encoders/position_encoder.py:255`; built at `shepherd_gnn.py:136-137` | Applied only when a caller passes `pos_encoding_dict`; none does |
| Ortholog consistency loss and its weight slider | `src/training/loss_functions.py:353`; `training_console.py` | No data reaches it (BACKLOG 57) |
| The disease objective CE + hinge + InfoNCE on raw cosine over a masked mean | `loss_functions.py:88,279`; `src/training/trainer.py:744-751` | The disease head's NCA |
| `LinkPredictionLoss` | `loss_functions.py:196`; `trainer.py:659-673` | Dead (no triples, gene→disease only, zero relations): the LP head's hinge |
| `MultiTaskLoss` uncertainty weighting | `loss_functions.py:423-579` | The two named loss weights |
| `DiagnosisHead`, `LinkPredictionHead` | `src/models/decoders/heads.py:34,273` | Unwired, not paper parts |
| `PhenotypeDiseaseMatcher` | `shepherd_gnn.py:314` | Unwired; the paper's patient encoder answers its open question |
| Separate pooling and scoring copies (Trainer, measurement, pipeline) | `trainer.py:744-751`; `src/evaluation/measurement.py`; the pipeline | The bundle's shared math |
| The hop-capped SP file: producer, sidecar binding, index and reader (`compute_shortest_paths`, `sp_artifact`, `sp_index`, `_load_shortest_paths`, `sp_analysis_records`), with their benchmark, oracle and tests | `scripts/compute_shortest_paths.py`; `src/inference/sp_artifact.py`; `src/inference/sp_index.py`; `src/inference/pipeline.py`; `scripts/benchmark_sp_lookup.py`; `scripts/sp_scan_reference.py` | Exact distances computed in process from the view (§6, as built at K4); they serve only the mixture and leave with it |
| `read_samples`' `training_fields` and the generator's disease genes in `gene_ids` | `src/kg/storage/file_storage.py`; `src/kg/sample_generator.py` | Their one reader is ShepherdGNN's `DiagnosisDataset`. `gene_ids` now means a gene-task candidate list, read with `gene_fields` |

**Kept:** PathReasoner evidence; the KG builder's cross-species option;
build_index (retrieval stays for planned natural-language work) on the one builder; the
heterogeneous GAT/HGT/SAGE encoders behind the contract; every historical artifact.

---

## 8. Scope of this round and acceptance

**This round: the A + B integration** (the reviewer's workflow and decision record). Package A
(base, disease) and package B (gene, patients-like-me, family operations) are internal dependencies
and commit boundaries, not stops for external review. Package C (formal research, clinical adoption)
and site-wide UI/UX refinement stay outside.

**Order, by dependency, not by a clock:** K0 (documents and the entry map) → K1 (shared model, graph
and contracts) → K2 (base) → K3 (disease, first full vertical flow) → K4 (gene, patients-like-me) →
K5 (family operations, several children) → K6 (functional UI) → K7 (retirement check) → K8
(integration). Retirements happen as callers move; K7 is the final check. Each meaningful change is
a revertible commit; reverting relies on Git, never on a runtime compatibility flag.

**Checkpoint review (the owner, 2026-10-27).** Every K node closes only after these six checks,
recorded with the node's commit:
1. it follows this blueprint;
2. the pre-built extension points (encoders, cross-species view, acceleration boundaries, task
   heads) have their interfaces;
3. deviations from the paper are removed, except the listed exceptions and trade-offs;
4. the node links with the other parts into one pipeline;
5. no unnecessary reinvention, side branch or parallel pipeline;
6. every pipeline it makes retirable is fully removed, or listed with the node that removes it.

**In scope for the UI:** base training and Fine-tune separately; choosing a qualified parent and one
to three tasks; job cards with status, cancel and products; Models with the family and a task
filter; Test measuring and registering by task; Diagnosis offering the inputs of the loaded model's
task (one resident model; switching task means loading another model). **Patients-like-me on
Diagnosis:** a registered reference cohort version chosen from the server list, separate from the
query; candidates show research id, rank, similarity and the recorded HPO terms and diagnosis or
gene; a known causal gene is used as evaluation truth or reference context, never as input for an
unknown query; results, exports and caches carry the reference cohort version; no name or contact
field is added. Existing D8 and session isolation apply; no multi-tenant platform.

**Data in this round:** synthetic functional fixtures with real task truth roles. Gene and
patients-like-me samples come from a declared rule (one causal gene from the disease's genes plus
distractor candidates) as digest-bound derived artifacts of a workspace, labelled "functional
fixture", never "simulated patients" of the paper. Reference cohorts carry causal genes through an
additive optional import format; older formats stay readable.

**Hand-computable regressions** (the oracle is hand arithmetic or a few independent lines):
1. **LP score and loss:** three nodes, two relations: `s = tanh(Σ h⊙r⊙t)` and the hinge equal hand
   values; accuracy at raw score > 0 equals micro-F1; the disease NCA on two patients and three
   diseases.
2. **Gradient:** one base LP step reaches the node table, every encoder layer and every relation
   vector drawn (catches the dead-LP pattern); one joint finetune step updates task head, patient
   encoder, encoder and relation vectors with the hand-computed weighting; a batch without a valid LP
   term makes no update.
3. **Sampler and split:** C4's enumeration (one neighbour: real draw → one positive, self draw →
   none, counted; isolated source skipped and counted); no negative equals its positive; the target
   edge, its reverse and duplicates are absent from that step's message passing; base validation
   passes messages over training edges only, finetune validation over the full view (G4); same seed
   and view give the same split digest.
4. **Norm and inference rule:** graph-mode norm equals its formula and differs from per-node norm;
   chunked layer-wise inference equals the one-shot forward (§2.8's cases, two chunk sizes).
5. **Metrics:** whole-set aggregation against a per-batch mean; fixed negatives give identical
   numbers twice; chunked all-disease ranks equal one-shot ranks; E7's oracles.
6. **Save and reload** through the one builder: identical parameter names and outputs, strictly
   loaded.
7. **Refusals before the job starts:** missing or unknown spec version; spec not matching the
   weights; wrong task (the task filter runs first); wrong KG, view or data binding; wrong reference
   cohort; invalid SP bounds; a checkpoint without a model spec.
8. **Strict resume:** an interrupted base or finetune run resumes under M3a, the upsampling counter
   included; a finetune from an early-stopped or weights-only parent is accepted as a finetune and
   refused as a resume.
9. **Graph view:** dedup by relation, native self-loops, the component rule and coverage of excluded
   nodes; SP on the full view against a restricted view.
10. **Gene and patients-like-me:** §6's worked example; C8's counterexample; F10's augmentation and
    zero-neighbour case; F9's multi-gene patient; F11's batch rule; self-exclusion by case identity.
11. **Cross-species (§3.3):** a synthetic graph with human genes, animal genes and phenotypes and
    ortholog relations; each kept encoder builds, runs forward, gets gradients and reloads; turning
    relations off equals a view built without them with the same weights; no second mixing step.

**Delivery:** "functional integration done in the container, awaiting external review and a short
GB10 acceptance". The full suite, `lint-imports` and the diff check run once on the integrated
batch; later narrow fixes rerun what they touch. A browser smoke of the new UI runs on the normal
launcher with an explicit CPU setting; what cannot be checked is reported as not verified. A
functional pass, alignment with the paper's method, reproduction on the original data and clinical
validity are reported separately.

**BACKLOG items in scope, bounded:** 23 (branch's closed list and lineage), 54 (shared validation and
unknown-field refusal for the training API, CLI and UI touched here, without the unsupported HGT/AMP
ban), 57 (LP and task losses really wired; dead weights and replaced losses gone), 60 (stale
Training tests touched here), 61 (one score name in the new result contract), 58 (only real module
boundaries, the effective-implementation record and the native reference). Touched only where this
round touches them: 53, 40, 56, 52.

---

## 9. Carried over from the scoping phase (until 2026-10-27)

- **The unit is a bundle.** The scoping's `patient encoder + score family + objective + output
  semantics` is now concretely the model spec (§3).
- **Explicit metadata is authoritative. State-dict evidence may recover structure; it does not by
  itself establish the training objective or the score semantics.** Since revision 2 there is no
  structural-inference rule at all: the new system rebuilds from the model spec only (§4).
- **Rejected, and still rejected:** a runtime dropdown for the score family; inferring an objective
  from structure; moving this plan into `DISEASE_SCORER_POLICY.md`, the accepted authority record.
- **Superseded on 2026-10-27:** "nothing scheduled"; the open question on the authoritative patient
  encoder; "removal off the table" for the unwired classes; enumerating several bundles before
  choosing; the facts R1–R10 of the scoping (now §2 and §7, with today's lines).
- **The evaluation protocol** stays owned by `../BACKLOG.md` item 11. The task-aware parts this
  design needs are `../PLAN_TEST_RESULTS.md` §8.

---

## 10. Entry map: kept, changed or retired

[`ENTRY_MAP.md`](ENTRY_MAP.md) is the K0 inventory of the code at `0e64666`: 464 entries in six
areas, each with its verdict (148 keep, 212 modify, 42 replace, 47 retire, 15 uncertain), its role
after the switch, its callers and its tests, and the 118 hazards the passes found; a completeness
pass adds 39 rows and 15 hazards. It guides the nodes; this blueprint decides, and a row found
wrong is corrected at the node that touches it.

**Hazards that change how the nodes are built** (decided in K0):

| Hazard | Decision | Node |
|---|---|---|
| The API, WebUI and inventory must read a spec without torch; `src/models/gnn` imports torch eagerly | The spec and the model record are torch-free modules; only the builder loads torch | K1 |
| The run layout is keyed by conv type and refuses `gatv2` | Bundle runs are filed under their encoder kind; listing reads conv types and encoder kinds | K2 |
| Early stopping cannot be turned off; the best-k file name assumes `val_mrr` | A Trainer switch disables early stopping; files are named by the monitored metric | K2 |
| A batch without a valid loss term still makes an optimizer update | A task step without a loss makes no backward and no update, and is counted | K2 |
| A base reads and records patient samples it does not use | The base reads the graph alone and records only graph roles | K2 |
| `run_plan` knows only resume and extend, and refuses early-stopped and weights-only parents | Finetune gets its own parent check (§4); resume keeps every refusal | K3 |
| Measurement assumes one truth per case and id-ordered ties | Task measurement uses the one metrics module (E1–E5) | K3, K4 |
| Ledger and `build_record` refuse other schemas as a whole | §9.3 of the provenance contract: per-record manifest schema, ledger file schema 3 kept | K3 |
| Fixtures and the probe chain build specless checkpoints | Moved to the one builder as each consumer moves; the probe trains a base and a child | K3, K8 |
| Automatic serving selection is task-blind | Selection reads stage and task first (§9.3 hazard 3) | K3, K5 |
| Unknown training fields vanish silently (API, YAML, WebUI) | BACKLOG 54's shared validation; the bundle path already refuses ShepherdGNN-only settings | K2, K6 |
| Multi-child finetune versus the one-slot job runner | Children run in sequence in the existing manager (§9.3 hazard 6) | K5 |

**BACKLOG items this round touches** (§8): 23, 54, 57, 58, 60, 61; and 40, 52, 53, 56 only where a
node touches them.

**Authority above everything here:** `docs/DISEASE_SCORER_POLICY.md` for what may be offered to
doctors.
