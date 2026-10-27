# Paper-aligned two-stage training — reference recipe, model bundle and checkpoint contract

**Status: design (2026-10-27), on `claude/paper-aligned-two-stage` from `main` `0e64666`.** Nothing
here is implemented or trained. Work package A (§8) starts from this document. Until 2026-10-27 this
file was "Scorer retraining — scoping only" (moved from `../task-scope/README.md` §6); what of that
scoping still holds is §9, the rest is in git (this file at `0e64666`). **One folder, one README**
stays the rule.

**Who decided what.**
- **The owner:** the direction (the paper's two stages and three tasks replace the disease cosine/SP
  mixture as the method; infrastructure is reused, not duplicated); the branch name; the three-axis
  naming (§5); the version policy (§0.2).
- **The reviewer (Codex, 2026-10-27):** the answers D1–D11 to this project's verification of the
  paper against the fixed upstream source and this repository, with corrections. They are adopted for
  this blueprint as **reviewer recommendations, not institutional approval**. Each is cited where it
  applies.
- **Not decided here:** any clinical ranking switch (`../../DISEASE_SCORER_POLICY.md` keeps its
  gates; its rev 11 proposes, without approving, the revisions this direction needs).

**Why.** The owner and the reviewer found that the paper's SP is nothing like this project's SP, and a
wider check followed. This repository's model differs from the paper's at every layer:
- no KG pretraining stage, and the link-prediction loss is dead;
- a heterogeneous GAT (v1) with fixed random features instead of a homogeneous GATv2 over a learned
  node table;
- a masked mean of phenotype vectors instead of the patient encoder;
- a cosine/contrastive objective instead of the tasks' NCA and multi-similarity losses;
- no gene or patients-like-me task.

So the next phase rebuilds the model and its objectives. It does not polish the interim scoring.

---

## 0. Sources, evidence levels and the version policy

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
published result (§2.5).

### 0.2 Version policy (the owner, 2026-10-27)

- **The reference baseline runs on the current stack and hardware**: torch 2.10.0 (CUDA 13.0) and
  torch-geometric 2.8.1 (`pyproject.toml`, BACKLOG 50), on the GB10. It uses model settings as close
  to the paper's as possible.
- **Nothing is downgraded or emulated.** No old PyG, torch or CUDA is installed, and no translation
  layer reproduces an old version's numerics. At the project's start the hospital was told that the
  upstream's packages cannot run on the new hardware, and that different versions give slightly
  different numbers.
- **Where an old default carries meaning, the spec states the meaning.** Package A sets it
  explicitly and records it (§2.6, the norm). It does not chase the old implementation's numbers.
- **Consequence:** results are compared with the paper as a reference, never claimed as an exact
  reproduction.

---

## 1. The design

**Two stages.**
1. **Base:** KG link-prediction pretraining over all relations of the bound KG, selected on
   validation (§2, `../PLAN_TEST_RESULTS.md` §8).
2. **Finetune:** from one exact, selected checkpoint, one to three tasks, each an independent child
   run:
   - **disease:** a patient's phenotypes → all KG diseases;
   - **gene:** phenotypes plus a candidate gene list → the candidates ranked;
   - **patients-like-me:** phenotypes → a reference cohort.

**Children never chain.** Choosing three tasks starts three children from the same parent; no task
feeds another. The "finetuned base" of a task is simply that task's first finetune.

**Operations.** `new`, `resume`, `extend`, `branch` and `finetune` are distinct operations, never one
flag read in different ways.
- `resume` and `extend` keep their contracts (`../PLAN_PROVENANCE_CONTRACT.md`, M3a).
- `branch` keeps architecture and data (BACKLOG 23). It is formalised before any variable comparison
  starts: an extend never carries a seed or learning-setting change.
- `finetune` is its own operation and page. It never borrows resume or branch to get past a parent
  check.

**Parents and the family tree.** The original base and its qualified descendants may all be
finetune parents. The tree sits under the actual parent checkpoint. Run directories stay flat (no
nesting, no copies). One reader serves the tree and the tag filter; tags stay non-authoritative.

**Settled; not to be asked again** (the reviewer, 2026-10-27, §5):
- **Selection and lineage:**
  - base → validation selects an exact checkpoint → one to three independent finetunes;
  - the dataset fingerprint and the seed are comparison conditions, not a family id;
  - the same generation or parent does not by itself prove a single-variable difference.
- **Scope:**
  - modularity does not mean building the general plug-in framework now;
  - installed, selected and actually used stay distinct.
- **Mode D and the old Mode C:**
  - Mode D stays, temporary and removable; nothing new depends on it or its observer;
  - the old Mode C's raw-cosine history is never silently rewritten by a new head.
- **No raw-cosine B-1 first.** No raw-cosine B-1 is built ahead of this work. Research
  implementation and the clinical switch gates are separate.

---

## 2. The reference recipe

**Rules (D1).**
- **The anchor:** the recipe is anchored on the fixed source and written in the paper's notation.
- **Conflicts:** where the paper's text and the source conflict, it follows the source, the only
  executable statement, and lists the paper's text. A source behaviour that is a defect is the
  exception (§2.4).
- **Corrections:** obvious defects are corrected and listed. A configuration with corrections is
  never labelled identical to the source.
- **One recipe:** one recipe, not a switch per row.

The rows are grouped by evidence class:
- **§2.1:** the paper and the source agree;
- **§2.2:** confirmed conflicts;
- **§2.3:** the paper is silent;
- **§2.4:** source defects corrected;
- **§2.5:** not verified.

The packages (§8) set the scope:
- **package A:** the base and the disease task;
- **package B:** gene and patients-like-me.

### 2.1 The paper and the source agree

| # | Paper | Fixed source | Adopted | Deviation | Small verification |
|---|---|---|---|---|---|
| A1 | Three-layer GAT, citing GATv2 (ref 106) | Three `GATv2Conv` layers (`up:node_embedder_model.py:6,81-85`) | Same | — | The built encoder has three GATv2 layers |
| A2 | DistMult decoder with tanh; hinge loss, margin 1 | `s = tanh(Σ h⊙r⊙t)` (`up:decoders.py`, `up:node_embedder_model.py:201-211`); loss `mean(max(0, 1 − (s⁺ − s⁻)))` (`:416-425`) | Same | — | Regression 1 (§8) |
| A3 | Edge split 80/10/10 | Reverse edges in the same part (`up:data_prep/construct_kg/prepare_graph.py:57-72,148-172`) | Same | — | Regression 3 |
| A4 | — | Validation/test message passing over training edges only (`up:pretrain.py:68-70`). A training batch removes its target edges and their reverses from message passing (`up:samplers.py:177-193`) | Same, plus D5 | — | Regression 3 |
| A5 | Pretraining hyperparameters selected on validation micro-F1 | Checkpoint monitor `val/node_total_acc` (`up:pretrain.py:97`). Accuracy and `f1_score(average='micro')` come from the same thresholded predictions (`up:utils/pretrain_utils.py:118-127`), so they are **one number** | Select on LP accuracy (that micro-F1): `sigmoid(raw) > 0.5`, i.e. raw score > 0 (`up:node_embedder_model.py:246-249`, `up:hparams.py:27`). Also report ROC-AUC and AP (D7) | — | `y=[1,1,0,0]`, `ŷ=[1,0,0,1]`: accuracy 0.5, micro-F1 0.5 |
| A6 | Patient encoder: Transformer (search range includes 0 layers) → attention pooling → projection | The scripts use 3 layers, 8 heads; sequences padded to 100 with a mask | Same | — | Shape and mask test; a padded position gets zero attention |
| A7 | Disease: `−‖z_d − z_P‖²` with NCA | 1000 random candidate diseases plus the truths per training batch (`up:hparams.py:101`); inference over all diseases (`:102`) | Same for training. Validation: U6 | — | Regression 1's NCA case |
| A8 | Gene: `η·EMBSIM + (1−η)·SPLSIM`; multi-similarity loss; gene augmentation; upsampling of the k rarest candidates | Same structure | Package B; see C7–C9 and §6 | — | Package B |
| A9 | Patients-like-me: `−‖z_Pi − z_Pj‖²` with NCA; positives share the causal gene | Self excluded (`up:patient_nca_model.py:95`) | Package B | — | Self-exclusion test |
| A10 | Finetune: the GNN keeps training; new task head; new optimizer | The pretrained module is loaded whole and the head is new (`up:train.py:236-248`) | Same | — | Regression 2 |
| A11 | SP is a fixed KG-derived feature | Fixed bounds, not a per-candidate-list min-max (`up:data_prep/shortest_paths/add_spl_to_patients.py:18-19,104-109`); never updated by the optimizer | §6 | — | §6 |

### 2.2 Confirmed conflicts (the paper's text and the source differ)

| # | Paper | Fixed source | Adopted | Deviation | Small verification |
|---|---|---|---|---|---|
| C1 | The final layer averages its heads | The final layer concatenates. Relation vectors and the patient head are `output × heads` wide (`up:node_embedder_model.py:85,90`; `up:patient_nca_model.py:52`): 128 × 4 = 512 in the scripts | Concat (D2). **The width comes from the model spec, never from a head.** | Follows the source: the scripts' widths depend on it. Released checkpoints not read (N1) | Spec → built shapes: encoder out = LP head width = patient head input |
| C2 | Norm and nonlinearity at every layer | After layers 1 and 2 only: conv → LayerNorm → leaky ReLU → BatchNorm → dropout 0.3. The last layer has nothing (`up:node_embedder_model.py:93-107,135-143`; `norm_method="batch_layer"`, `up:hparams.py:25`) | The source's placement; its norm semantics in §2.6 | Follows the source | Regression 4 |
| C3 | Eq 20: `λ·L_DX + (1−λ)·L_LP` | Computes `λ·L_LP + (1−λ)·L_task` (`up:gene_prioritization_model.py:184,207`; `up:patient_nca_model.py:147,179`). Script λ: 0.4 gene, 0.9 disease, 0.7 patients-like-me | **Two named weights, `task_loss_weight` and `lp_loss_weight` (D9).** Starting values, the scripts' flipped (task/LP): gene 0.6/0.4, disease 0.1/0.9, patients-like-me 0.3/0.7. Not best values for a new KG. η is separate | Names remove the inverted λ | Hand-computed `0.1·L_task + 0.9·L_LP` |
| C4 | Eq 5 averages uniformly over edges | Each epoch every node is a source once, with one uniformly random training neighbour as its positive (`up:samplers.py:167-173,202-204`). So an edge weighs about `1/deg(source)` | **The source's node-based sampling.** An epoch means what it means in the source, which the 200-epoch schedule counts | Listed difference from Eq 5 | Toy graph: each node a source exactly once per epoch; positive frequency ≈ `1/deg` |
| C5 | Eq 7–9: two nested leaky ReLUs in a projection | One leaky ReLU, no outer one (`up:task_heads/gp_aligner.py:46-54,81-86`; `up:task_heads/patient_nca.py:40-44,69-70`) | The source's single leaky ReLU | Follows the source | Module structure test |
| C6 | Early stopping | None in either stage. `ReduceLROnPlateau` patience 1000 exceeds 200 epochs, so the rate never changes. Finetune saves every epoch (`up:pretrain.py:109-117`; `up:train.py:283-292,311`; `up:hparams.py:38`) | **The existing `EarlyStopping` callback** (`src/training/callbacks.py:134`) on the stage's validation metric. Patience is a recipe value. Constant rate, no scheduler (D7) | Follows the paper's statement with the existing callback | It stops at patience on a scripted metric sequence and monitors the named key |
| C7 | Gene augmentation after the projection | Before the projection (`up:gene_prioritization_model.py:85-96`). The gate is `ω·e^(−ω·deg) + (1−ω−0.1)`, equal to the paper's only at ω = 0.7. The degree gate is off by default (θ fixed 0.7), and prediction always uses the fixed gate (`up:train.py:64-66`; `up:gene_prioritization_model.py:92,303-309`) | The source (package B) | Follows the source | Hand-computed two-gene example |
| C8 | Eq 15: multi-similarity without mining | α 1, β 20, γ 1, plus pair mining with margin 0.4 (`up:hparams.py:90-93`; `up:utils/loss_utils.py:123-134`) | The source (package B). **Its effect is not assumed small** | Follows the source | Hand counterexample: positives [0.1, 0.9], negative [0.4]. Mining keeps 0.1, drops 0.9, keeps 0.4; the loss goes from 1.518369 to 1.241154 |
| C9 | EMBSIM `tanh(z_Pᵀ W z_g)` | On `F.normalize`d vectors, plus a bias (`up:task_heads/gp_aligner.py:95-101`) | The source (package B) | Follows the source | Hand-computed two-vector case |

### 2.3 The paper is silent; the source decides

| # | Paper | Fixed source | Adopted | Deviation | Small verification |
|---|---|---|---|---|---|
| U1 | — | Homogeneous message passing: node and relation types are ignored; relations appear only in the decoder (`up:node_embedder_model.py:121,126`) | A homogeneous view built from the same bound data as the heterogeneous one | — | The encoder receives no type or relation tensor |
| U2 | — | A learned table `nn.Embedding(N, 2048)`, default N(0, 1) initialisation, no external features (`up:node_embedder_model.py:62,117`) | A learned table, width in the spec. **Transductive:** a new KG version needs a new base | — | Table gradient in regression 2 |
| U3 | — | The scripts' "best hyperparameters from the paper" (`up:run_pretrain.sh:6-17`): `nfeat` 2048, `hidden` 256, `output` 128, 4 heads, dropout 0.3, lr 1e-4, wd 0. Layer widths 2048 → 2048 → 1024 → 512 (the first layer's hidden size is doubled). 512 sources a batch, neighbour sizes [15, 10, 5], 200 epochs, seed 1 (`up:hparams.py:18-44`) | The starting recipe values | Not checked against the released checkpoints (N1) | Spec values round-trip into the built module |
| U4 | — | `GATv2Conv` defaults at PyG 1.7.2: `concat=True`, attention dropout 0, `add_self_loops=True`, `share_weights=False` | Set explicitly in the spec; current PyG has the same parameters | — | The built layers carry them |
| U5 | — | Negatives by edge type: for each positive, a tail drawn with replacement from the same relation's positives in the batch (`up:node_embedder_model.py:150-170`; `up:utils/pretrain_utils.py:35-38`) | The same scheme, without the defect D1 | — | Regression 3 |
| U6 | — | Disease validation candidates: 1000 random diseases plus the kept truths (`up:train.py:173`), so not exactly 1000 per case | **All diseases of the bound KG, computed exactly in chunks, no ANN (D3).** Truths, masks, ties and denominators as `../PLAN_TEST_RESULTS.md` §8. Training negatives stay sampled | The same universe as the formal ranking | Chunked ranks equal one-shot ranks |
| U7 | Finetune LP "focuses on" gene, phenotype and disease relations | LP seeds are gene, phenotype and disease nodes, with neighbours of any type. So 18 of the upstream KG's 30 directed relations can be drawn, `gene→pathway` and `gene→GO` among them (`up:utils/pretrain_utils.py:98-115`; `up:samplers.py:365-387`) | **The source's seed-type rule. The relation list is derived from the bound KG and recorded, never hard-coded (D8).** Narrowing to the three types' induced subgraph is a separate research choice | — | The derived list on a toy KG |
| U8 | — | Finetune: Adam over the whole model (GNN, node table, relation vectors, head), one learning rate (1e-5 gene and disease, 5e-5 patients-like-me). No weight decay, no scheduler, clip 1.0. The upstream `--resume` restores weights only (`up:gene_prioritization_model.py:49-55,541-543`; `up:train.py:236-248,308-320`) | The starting values. A new optimizer for every finetune child | — | Regression 2 |
| U9 | — | A finetune batch: 64 patients plus random gene, phenotype and disease nodes (100 gene, 300 disease, 300 patients-like-me), sampled 3 hops (`up:train.py:156,167-168,181-184`) | The same for training. Evaluation embeddings follow §2.6 | — | Batch composition test |
| U10 | Eq 18–19: `−‖·‖²` on the projected vectors | pml `LpDistance(power=2)` normalises by default (pml 0.9.98 `distances/base_distance.py:7-8`). So for **non-zero** unit vectors `−‖a−b‖² = 2cos(a,b) − 2`. The softmax scale is 1 **in the selected configuration** (`up:utils/loss_utils.py:181`) | Normalisation stated in the spec, not left to a library default: L2 with ε 1e-12, so a zero vector stays zero. The cosine identity is not relied on for zero vectors | — | Hand-computed two-patient, three-disease NCA |
| U11 | Eq 6: attention pooling with one vector | allennlp 2.4.0 `BilinearAttention`: `aᵀWx_p + b`, then softmax | The source's form `aᵀWx_p`; `b` dropped. It cancels in the softmax and gets zero gradient, so forward and updates are unchanged. **The paper's and the source's parametrisations agree in the forward pass; their optimisation paths are not guaranteed equal.** | — | A three-phenotype example's weights |
| U12 | — | η is `alpha`: default 0 (SP only, `up:train.py`), 0.5 in the scripts | Package B; starting value 0.5. η is the gene score's mixing weight, unrelated to the loss weights and to today's disease η = 0.7 | — | §6 |

### 2.4 Source defects, corrected (deliberate deviations)

| # | Paper | Fixed source | Adopted | Deviation | Small verification |
|---|---|---|---|---|---|
| D1 | — | A negative may equal its positive (`up:node_embedder_model.py:165`, "currently possible"). A relation with one positive in a batch has hinge 1 and zero gradient | A negative never equals its own positive. A positive with no other same-relation tail in the batch adds no LP term and is counted. Other true edges are not filtered out (as in the source) | Self-negatives train nothing | Regression 3 |
| D2 | — | The split is unseeded (`up:data_prep/construct_kg/prepare_graph.py:59`, `random.shuffle`). Regenerating the same split is not reproducible; a saved split file is | The split is made once with a recorded seed and saved as an artifact with its digest. Every stage records the digest, and resume refuses a different one | Traceability | Same seed and KG give the same digest |
| D3 | — | Validation positives (a random walk per source) and negatives are redrawn at every evaluation, so the selection signal is noisy | **Every validation edge evaluated once, against fixed and traceable negatives (D7):** a fixed rule and order with a recorded seed, or a saved negatives artifact with its digest | Selection must not be noise | Two evaluations of the same weights give identical numbers |
| D4 | — | Metrics averaged per batch | Whole-set totals: correct over total; AUC and AP on the pooled scores. A recorded difference in aggregation (D7) | Batch size must not weigh cases | Unequal batches, hand-computed |
| D5 | — | Finetune does not remove the predicted edges from message passing (`filter_edges=False`, `up:hparams.py:94`; `up:train.py:158`); pretraining does | **In both stages,** the step's target edges and their reverse and duplicate representations leave that step's message-passing graph (D8). Which graph each stage saw is recorded | The target must not be visible | Regression 3 |
| D6 | — | A source with no edge in the partition gets itself as its "positive" (`up:samplers.py:203`, random walk) | Such a source is skipped and counted | A self-loop is no edge | Regression 3 |
| D7 | — | The NCA drops a patient with no positive among the candidates silently (`up:utils/loss_utils.py:255`) | The truths are always among the candidates. A patient whose truth is not a KG disease is refused at data binding and counted | Nothing silently dropped | Data-binding refusal test |
| D8 | — | SP: unreachable stored as 0, bounds hard-coded, held-out edges included | §6 (the gene task's gate) | §6 | §6 |

### 2.5 Not verified

None of these blocks package A. Each is stated, not guessed.

- **N1.** The hyperparameters of the released checkpoints. The scripts' "best hyperparameters" are
  not shown to be those of every published result.
- **N2.** Which SP matrix the paper's experiments used. `shortest_paths.py` writes a `_noGO` file
  name, while `add_spl_to_patients.py` reads the other file by default, and its row order needs every
  node to appear on some edge.
- **N3.** Whether the original KG already holds duplicate reverse edges.
- **N4.** Whether this commit is the complete recipe of every published number (§0.1).
- **N5.** How the source's validation treated a source node without a validation edge (the random
  walk returns the node itself). D3 replaces that path either way.
- **N6.** How much the graph-mode norm (§2.6) differs between sampled training and full-graph
  inference. The source has the same split: `predict.py` uses the full graph.

### 2.6 Norm semantics and the inference rule (decided here, before package A)

- **The old semantics.** PyG 1.7.2's `LayerNorm` without a batch vector normalises across all nodes
  and channels of the forward pass, not per node (the reviewer, from `layer_norm.py` at 1.7.2). Its
  affine shape and ε placement also differ from today's. Current PyG offers the same semantics as
  `LayerNorm(mode="graph")`; per node is `mode="node"`.
- **Adopted:** graph-mode `LayerNorm` and `BatchNorm1d` (running statistics in evaluation), in the
  source's order (C2). Numeric differences from the old implementation are accepted (§0.2). Per-node
  norm is a different configuration (§5 axis 1), not the reference.
- **Consequence:** a node's vector depends on the node set of the forward pass. So two forward rules,
  both part of the model spec:
  - **training** uses sampled subgraphs, as the source does;
  - **evaluation and serving** use layer-wise full-graph inference over the stage's message-passing
    graph, in node chunks. Each layer's norm statistics are taken over all of that layer's nodes,
    which gives the one-shot full-graph forward within float tolerance and fits in memory.
  - Sampled-batch evaluation is not assumed equivalent.
- **Verification:** regression 4.

---

## 3. The model bundle, the builder and the module boundaries

### 3.1 One bundle, one builder

```
model spec (versioned, self-describing)
  → input module (learned node table)
  → GraphEncoder
  → LP head (one diagonal relation vector per directed relation of the bound KG)
  → finetune only: patient encoder → task head (disease | gene | patients-like-me)
```

- **One module, one builder.** Every trainable parameter (task head and relation vectors included)
  is a submodule of one `nn.Module`. One builder in `src.models` builds it from the spec.
- **Every consumer goes through it:** the Trainer's optimizer, the checkpoint writer and reader,
  serving, measurement and `scripts/build_index.py`.
- **What it replaces for new specs:**
  - `build_shepherd_model` (`src/models/gnn/shepherd_gnn.py:506`), which today strict-loads a bare
    `ShepherdGNN`, so a checkpoint with task heads cannot be served or measured;
  - `create_model_from_config` (`scripts/train_model.py:928`);
  - `build_index.py`'s own construction (`:169`).
- **Old specs** keep their path only while history needs reading (§4).
- **Not overstated:** today's `MultiTaskLoss.log_vars` would be neither optimised nor saved, but
  they matter only with uncertainty weighting on, which is off by default. No default run has lost
  them.

**Lifecycle hooks.** The Trainer stays the only trainer, and `scripts/train_model.py`,
`run_plan`, `checkpoint_payload` and `checkpoint_io` the only training and saving system.
- **The task module supplies:** batch interpretation, the forward pass, named loss terms (`lp`,
  `task`) and metrics.
- **The Trainer owns:** the optimizer, AMP, accumulation, cancellation and saving.
- **Design-level hooks** (package A picks the code names):
  - `build(spec)`;
  - `transfer(parent, spec)`, which returns the weight-transfer record;
  - the stage's sampler;
  - `step`, which returns the loss terms;
  - `validate`, which returns a report with its digests.

### 3.2 Shared math and the scorer-independence contract (D5)

- **Where it lives.** Encoder, head and scorer math lives in `src.models`; training, inference and
  evaluation import it.
- **The contract still holds.** `scorer-independence` (`.import-linter.ini`) only forbids
  `src.inference.scoring` from importing `src.training`.
- **What changes is the oracle.** For new specs, tests that used two product implementations as each
  other's oracle (`tests/unit/test_trainer_validation_values.py`) move to hand arithmetic or a short,
  independently written computation. **Expected and actual never both call the head under test.**
- **When.** The contract's comment changes with package A. Mode C's primitives stay as they are for
  its history.

### 3.3 The GraphEncoder contract

- **In:**
  - global node ids and node types;
  - the relation mapping of the bound KG;
  - message-passing edges, kept separate from supervision edges;
  - the sampled subgraph, or the full graph.
- **Out:** node vectors of the width the spec declares.
- **Declared:**
  - the width;
  - whether node types or relations are used;
  - the sampling needs;
  - supported dtypes and AMP;
  - the forward rules of §2.6.
- **Layers are not hops.** The three-layer reference GATv2 needs three hops. That is not a universal
  rule: a full-graph or global-attention encoder has other execution strategies.
- **A swap is not automatically one variable.** Changing the encoder reuses the workflow, but the
  width, head capacity, type information, sampling and initialisation that change with it are listed
  as differences.
- **Inheritance.** A finetune child uses its parent's encoder spec. Homogeneous and heterogeneous
  views come from the same bound data.
- **Existing encoders.** The heterogeneous GAT, HGT and SAGE of `ShepherdGNN` move behind the same
  contract as non-reference encoders; package A need not validate them.
- **HGT** carries BACKLOG 20 (Edit 1 landed, reviewed on CPU only; Edit 2 waits for the GB10), and
  `segment_matmul` has no fp16 or bf16.

### 3.4 Five boundaries and their candidate acceleration routes

Each boundary states its tensor shapes, node and relation mapping, mask, dtype and device, gradient
and batch semantics. The routes are candidates, not plug-ins. The plug-in system stays its own
project (BACKLOG 58), and these boundaries are where it will attach; no empty slots are added now.

| Boundary | What crosses it | Candidate routes |
|---|---|---|
| Sampling and transfer | node ids, edges, relation ids, masks | pyg-lib samplers. GraphBolt is a possible route, not a verified PyG drop-in |
| Graph encoder | node vectors | pyg-lib or cuGraph fused kernels (possible routes), `torch.compile`, AMP. **GAT, GATv2 and HGT attention is sparse, along edges: FlashAttention, xFormers and SageAttention do not apply.** Dense attention kernels would apply only to a future global-attention encoder |
| Patient encoder | padded phenotype sequences (≤ 100) with a mask | Dense attention (SDPA, FlashAttention). `src/models/attention/adaptive_backend.py` is a candidate for review (mask, dropout, dtype, gradient and fallback semantics), not wired as is. Sequences are short, so any gain is measured |
| Task decoding and scoring | patient and candidate vectors, scores | **Exact** search (chunked matmul, cuVS exact) keeps the scoring contract. **ANN** can drop candidates, which changes results and metrics: it needs its own declared approximation policy |
| Loss and update | losses, gradients | AMP, `torch.compile`, gradient checkpointing |

---

## 4. The checkpoint contract (summary)

The full design is `../PLAN_PROVENANCE_CONTRACT.md` §9. Every new checkpoint records:

- **`model_spec`:** versioned, and enough to rebuild the bundle without guessing from key names.
- **`learning_stage`:** `base` | `finetune`.
- **`task`:** none for a base | `disease` | `gene` | `patients_like_me`.
- **`operation`:** `new` | `resume` | `extend` | `branch` | `finetune`.
- **Lineage:**
  - the exact SHA-256 of the parent checkpoint file, and the parent run;
  - the family id: the root run's id, inherited unchanged.
- **The weight transfer:** which parameter groups were loaded from the parent, which were initialised
  new, and which are trainable.
- **Data role digests:**
  - the KG, the edge split artifact and the finetune roles;
  - **which graph each stage saw:** message passing, supervision, SP.
- **The effective loss and scoring identity:** term names, weights and versions.
- **Selection evidence:** the digest of the validation report that selected the parent.

`progress.stage` keeps its meaning (the `original` / `extension` schedule). An early-stopped or
weights-only base **with a model spec** may be a finetune parent; strict resume is not loosened for
it.

**Migration.**
- **Only what is needed:** one switch boundary, explicit refusals, no compatibility framework.
- **History stays readable.** Not every old checkpoint stays resumable, and not every old report
  stays re-registrable.
- **No retroactive labels.** No existing checkpoint is relabelled base or finetune.
- **Task filter first.** It runs before serving's checkpoint selection, so a base or gene model is
  never loaded into disease serving.

---

## 5. Naming: three axes that are never mixed

1. **Configuration identity (what the model is):** derived from the spec, e.g. "GATv2 · reference
   recipe r1" or "HGT · recipe r3".
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
within a comparable task, cohort and protocol.

**Where it comes from.** All of it is derived from authoritative records (spec, registered reports,
approval records). Manual tags stay non-authoritative.

---

## 6. The gene task's SP (a gate for package B's gene task; it does not block package A)

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
- **Three readings:**
  - the source: +0.6575, +0.3151, −0.8836;
  - the printed formula on this list: −1, −1.444, −3;
  - a per-list min-max: 1, 0.556, −1.
- **Mixed with EMBSIM** (η = 0.5, EMBSIM g1 = −0.8, g3 = +0.9): fixed bounds rank g3 first, a per-list
  min-max ranks g1 first. **The normalisation is part of the task's definition, not a display
  choice.**

**Required before the gene task is built** (D4, as the reviewer refined it):
1. **Graph.** The full, specified KG version, as in the source. This is the gene task's protocol,
   separate from the base's held-out isolation. A model finetuned with that graph, or with SP derived
   from it, has seen held-out structure, so its LP figures on those edges are not "never seen".
2. **Distances.** Exact shortest distances for every needed (candidate gene, phenotype) pair. The
   existing producer and reader are extended: no second distance file outside the binding checks, no
   all-pairs matrix required.
3. **Unreachable and incomplete stay apart.** If a hop cap stays, the needed pairs are proven
   complete, or pairs beyond it are marked incomplete. `max_hops+1` is never presented as a true
   distance, and unreachable is never 0.
4. **Nothing fabricated.** The first reference recipe fabricates no SPLSIM for an unreachable or
   incomplete pair.
   - A case that cannot be fully scored is listed as non-scorable, with its reason, in the original
     denominator.
   - No gene or phenotype is dropped silently while claiming a complete ranking.
   - A truncation penalty or partial coverage would be a separate, declared scoring policy.
5. **Calibration.** The bounds are fitted on the training role only; validation selects the recipe,
   and test never takes part. Refitting on training plus validation is a new training procedure,
   after which that validation is no longer independent.
6. **Bounds.** The statistical unit, the degenerate range (max = min) and the out-of-bounds rule are
   written. The source's upper bound depends on the aggregation (mean 6.84, max 9, min 6.5,
   median 7; `up:data_prep/shortest_paths/add_spl_to_patients.py:104-109`). The source does not
   clip, and nothing clips silently for display.
7. **6.84 is a data constant** of the source's simulated patients. It is never copied to a new KG.

**Today's SP is neither.** Today's producer runs an undirected BFS from phenotypes up to 5 hops,
and reads `max_hops+1` both beyond the cap and unreachable. Today's disease SP term `1/(1+mean hops)`
is neither the gene SPLSIM nor a per-list min-max (`../../SP_SCORE_GUIDE.md` §3).

---

## 7. Retirement list (D10 and the replaced parts)

This list is an inventory the engineering carries out in the package that replaces each part, not a
questionnaire. No retired part keeps a UI switch.

| Part | Where | State | Retired |
|---|---|---|---|
| Positional encoder | `src/models/encoders/position_encoder.py:255`; built at `src/models/gnn/shepherd_gnn.py:136-137` | Applied only when a caller passes `pos_encoding_dict` (`:211-212`); none does | A, with the encoder's move behind the contract |
| Ortholog consistency loss | `src/training/loss_functions.py:353` | No data reaches it (BACKLOG 57) | A, with the replaced losses |
| The disease objective: CE + hinge + InfoNCE on raw cosine over a masked mean | `loss_functions.py:88,279`; `src/training/trainer.py:744-751` | Replaced by the disease head's NCA | A, for new specs; old runs keep their records |
| `LinkPredictionLoss` | `loss_functions.py:196`; `trainer.py:659-673` | Dead: nothing produces its triples, it scores gene→disease only, and zero relation vectors give loss 1 with zero gradient | A |
| `MultiTaskLoss` uncertainty weighting | `loss_functions.py:423-579` | Replaced by the two named loss weights | A |
| `DiagnosisHead`, `LinkPredictionHead` | `src/models/decoders/heads.py:34,273` | Unwired; not paper parts (MLP attention, LayerNorm/GELU projection; no tanh, all-pairs margin) | When their replacements land (A); the tests that build them move or go |
| `PhenotypeDiseaseMatcher` | `shepherd_gnn.py:314` | Unwired. It was one side of the open patient-encoder question (`../../RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md` §7 Q2), which this direction answers for the reference recipe: the paper's encoder | A |
| `OrthologGate` | `src/models/gnn/layers.py:248`; called at `shepherd_gnn.py:223-228` | Runs when ortholog edges exist. "The ortholog loss does not work" does not mean the gate is unused | **Stays** in the existing heterogeneous encoder, marked non-reference, when that encoder moves behind the contract. Retired cleanly if no kept configuration uses it |
| Separate pooling and scoring copies (Trainer, measurement, pipeline) | `trainer.py:744-751`, `src/evaluation/measurement.py`, the pipeline | Replaced by the bundle's shared math for new specs. Mode C keeps its raw-cosine meaning for its history | A |

---

## 8. Work packages and acceptance

Each package is complete in itself. It may take several commits; it goes to external review after
its wiring, self-check and related tests, never function by function. A formal large training run
is research, not a functional test. A functional pass, alignment with the paper's method,
reproduction on the original data and clinical validity are reported separately.

**Package A — one complete paper-aligned disease flow.**
- **The contract:** the shared model, task and checkpoint contract (§3, §4, the provenance contract
  §9).
- **The base:** KG base pretraining with the §2 recipe and its validation.
- **The disease task:** finetuned from an explicitly selected base, with the joint LP and task loss.
- **Measurement and serving:** the same model goes through both. Results and exports name the new
  semantics; BACKLOG 61's wording is settled here.
- **Lineage and the UI:**
  - the parent records;
  - choosing a parent in Models;
  - the separate Fine-tune operation;
  - no site-wide UI redesign, and the WebUI skill applies.
- **Branch:** its minimal formal semantics, before any comparison of variables.
- **Clean-up:** BACKLOG 60 fixed when Training is touched, and A's rows of §7.
- **Data:** synthetic fixtures with real task-specific truth roles (D6).

**Settled before package A starts** (this document):
- the input, encoder and LP head recipe: GATv2, norm, widths, negatives, the edge split (§2);
- the bundle, learning stage, operation and checkpoint contract (§3–§4);
- the single builder and its lifecycle hooks (§3.1);
- the retirement list (§7);
- the regressions below.

**Package A's hand-computable regressions.** The oracle is hand arithmetic or a few independently
written lines, never the head under test.
1. **LP score and loss.** Three nodes, two relations, hand-set vectors:
   - `s = tanh(Σ h⊙r⊙t)` and `mean(max(0, 1 − (s⁺ − s⁻)))` equal the hand values;
   - accuracy at raw score > 0 equals micro-F1;
   - the disease NCA on two patients and three diseases equals its hand value.
2. **Gradient.**
   - **One base LP step** on a toy KG gives a non-zero gradient to the node table, every encoder
     layer and every relation vector drawn. This catches the dead-LP pattern: a constant loss with
     zero gradient.
   - **One joint finetune step** updates the task head, the patient encoder, the encoder and the
     relation vectors. The two terms are weighted as hand-computed.
3. **Sampler and split.**
   - no negative equals its positive, and a positive without a valid negative is counted;
   - the target edge, its reverse and its duplicates are absent from that step's message passing;
   - validation message passing uses training edges only;
   - each node is a source once per epoch, and a source without an edge is skipped and counted;
   - the same seed and KG give the same split digest, with reverse edges in the same part.
4. **Norm and inference rule.**
   - graph-mode norm on a four-node toy equals the formula and differs from per-node norm;
   - chunked layer-wise full-graph inference equals the one-shot forward within tolerance.
5. **Metrics.**
   - whole-set aggregation against a per-batch mean on unequal batches;
   - fixed negatives give identical numbers in two evaluations;
   - chunked all-disease ranks equal one-shot ranks under the stated tie, mask and denominator rules.
6. **Save and reload.** `build(spec)` → save → load through the one builder: identical parameter
   names and outputs, task head and relation vectors included, strictly loaded.
7. **Refusals.** Each is refused before the job starts, with its reason and no key-name guessing:
   - a missing or unknown spec version;
   - a spec that does not match the weights;
   - the wrong task: a base or gene checkpoint is never selected for disease serving, since the task
     filter runs first;
   - the wrong KG or data binding;
   - an old checkpoint across the switch boundary.
8. **Strict resume.**
   - an interrupted base or finetune run resumes under the M3a contract;
   - a finetune from an early-stopped or weights-only parent is accepted as a finetune and refused as
     a resume.

**Package A's acceptance.**
- **On CPU:** the regressions; the affected contract checks (`lint-imports` among them); the related
  suites.
- **On the GB10:** one short acceptance of the new flow at the end of the package.
- **Not rerun:** historical acceptances.

**Package B — the other two tasks and the full family operations.**
- **Gene:** candidate lists and gene truths as data; the EMBSIM + SPLSIM head; the multi-similarity
  loss; augmentation and upsampling. §6 is settled first.
- **Patients-like-me:** the reference cohort contract, self-exclusion and same-gene positives.
- **Several children from one parent:**
  - sequenced through the existing job manager and the existing GPU-lock lifecycle, with no second
    queue;
  - each child with its own state, failure and cancellation;
  - the parent unchanged.
- **The family:** the family tree and tag filtering from the one reader; resume, extend and branch of
  finetune children; the table of comparable differences.
- **Fixtures:** per task, with real truth roles; disease fixtures never pose as gene or patient
  benchmarks.
- **Acceptance:** each task's hand regressions (§6's worked example, C8's counterexample,
  self-exclusion), plus package A's kinds of checks for the new paths.

**Package C — formal research and the serving switch.**
- **Research:** train the formal reference baseline, select on validation, and run the registered
  held-out or external tests.
- **The switch:** after the research report is checked, the clinical ranking switch goes through the
  existing gates (`../../DISEASE_SCORER_POLICY.md` §6–§7, with rev 11's proposed revisions approved or
  not).
- **Retirements:** decide whether Mode D's comparison evidence is enough. Then retire the temporary
  comparator and the legacy mixture separately, keeping historical reports.
- **Acceleration plug-ins:** their own project once the model contract is stable (BACKLOG 58).

**Outside the packages.**
- **Data and its use:** formal data and permission for the three tasks. It gates effect claims, not
  functional work (D6).
- **The clinical switch:** institutional scheduling.

---

## 9. Carried over from the scoping phase (until 2026-10-27)

- **The unit is a bundle.** The scoping's `patient encoder + score family + objective + output
  semantics` is now concretely the model spec (§3).
- **Explicit metadata is authoritative.** Structural inference from state-dict evidence is allowed
  only as a bounded legacy compatibility rule. **State-dict evidence may recover structure; it does
  not by itself establish the training objective or the score semantics.** The model spec is that
  metadata, and the new builder never guesses from key names. Since provenance contract M2 a
  checkpoint without `training_input_digests` is refused anyway (decision 2: no compatibility mode
  for the old pipeline).
- **Rejected, and still rejected:**
  - a runtime dropdown for the score family;
  - inferring an objective from structure;
  - moving this plan into `DISEASE_SCORER_POLICY.md`, the accepted authority record.
- **Superseded on 2026-10-27:**
  - "nothing scheduled";
  - the open question on the authoritative patient encoder;
  - "removal off the table" for the unwired classes (now §7);
  - enumerating several bundles before choosing (now one reference recipe; other configurations are
    branches);
  - the facts R1–R10 (now §2 and §7, with today's lines).
- **The evaluation protocol** stays owned by `../BACKLOG.md` item 11. The task-aware parts this
  design needs are `../PLAN_TEST_RESULTS.md` §8.

**Authority above everything here:** `docs/DISEASE_SCORER_POLICY.md` for anything served to doctors.
