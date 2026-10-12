# Validation versus Test under the paper's protocol, and an AMP-only training comparison

**Status: findings for discussion (2026-10-11). Nothing here is decided, and nothing in the code or
in another document is changed by it.** It answers two questions the project owner raised on
2026-10-10 and 2026-10-11, for the owner's discussion with the reviewer. §5 lists the points left
open.

**Reviewed (2026-10-11, `edf7a41..bd3ea00`): three P2, no P1; applied in this revision.** The
review's warning is kept here as a boundary: none of the research questions below becomes a
prerequisite of the provenance contract or of the Mode A/B retirement by being written down. In
particular, the chain "M2 one read per input → whole-model determinism → multi-seed retraining → a
full Test importer → full alignment with the paper" is not a dependency of the current work. This
document does not clear policy Gate 1, does not show AMP equivalent, and does not widen M2.

**Since this report (2026-10-11):** training's seed is now applied before the loaders and the
model are built, so it fixes the initial weights (M3a's seed ordering, implemented ahead of the
rest of M3a; plan M3a). Where the text below calls the initial weights not fixed, it describes
`edf7a41`. The other open points stand.

**Since this report (2026-10-12): the owner's clarification.** The owner reported on 2026-10-12
that the institution had already authorised retiring the old pipeline, Modes A and B included;
the new pipeline governs. Policy Gate 1 stays, and its comparison deliverable is kept, now
comparing the currently served ranking (Mode D, B-0.5's served behaviour) with the planned GNN
ranking (Mode C). On AMP, what the owner and the institution want to know is whether an
AMP-*trained* model is worse: any loss of accuracy or quality, against the large training-time
saving some GNN architectures get. That is the training-time question this report studies in §4
(§4.5, and §4.6's first experiment), recorded as bounded research in [BACKLOG item 21](BACKLOG.md)
(pilot first; the acceptable quality loss not yet given), not as a prerequisite of M2 or of the
Mode A/B retirement. It is not the same-checkpoint scoring comparison with autocast switched (the
differential's AMP report, item 7a's AMP leg, the retirement assessment's §8 item 7 and its
options), so that scoring evidence is no longer needed before the differential or Modes A and B
are deleted. The assessment's §8 items 4 and 7 are closed. Where the text below treats the
institution's B-0 scope, or which AMP question it wants answered, as unrecorded or open (§2,
§3.5, §4.6, §5 points 1 and 6 and the reviewer's recommendations 1 and 6), it describes the state
before this clarification.

**Citations.**
- This repository: `path:line` at `edf7a41`.
- The original SHEPHERD code: `upstream:path:line` at `e95433a` of
  https://github.com/mims-harvard/SHEPHERD, which equals its GitHub HEAD when checked. Paths are
  relative to that repository's root.
- The paper: Alsentzer et al., "Few shot learning for phenotype-driven diagnosis of patients with
  rare genetic diseases", *npj Digital Medicine* 8:380 (2025), full text PMC12181314, and its
  Supplementary PDF.
- Libraries: the installed torch 2.10.0, torch_geometric 2.8.0.post1 and pyg_lib 0.8.0. For
  PyTorch C++ sources, the commit the installed build reports (`449b176`).

**How it was made.** Two rounds of read-only research. Each report was re-checked by an independent
agent that tried to refute it. Where two reports disagreed on a probe result, the probe was re-run
by hand (§4.3). **No CUDA device was available.** Every probe ran on CPU. A statement about CUDA
behaviour rests on source code and documentation and is marked *source-level*.

**One caveat on the upstream code.** As committed, the released upstream code would not run. This
was found by reading the code; it was not executed:
- all three run scripts leave gene augmentation on without its similarity dictionary
  (`upstream:shepherd/hparams.py:81`, `upstream:shepherd/train.py:142-146`, `upstream:shepherd/samplers.py:590`);
- `predict.py` reads an `alpha` key its hyperparameters do not define (`upstream:shepherd/hparams.py:208-230`,
  `upstream:shepherd/samplers.py:296`);
- paths are hard-coded.

So a fact taken from that code describes the released design. Whether the paper's numbers came from
exactly that code and configuration is not shown.

---

## 1. The questions

1. **Is explaining "why the training-time validation score differs from the formal Test score" part
   of the institution's direction to follow the original paper's experimental method as closely as
   possible?** And does the decided data design still need that question answered?
2. **Can two training runs differ only in AMP?** That means the same dataset and every seed fixed and
   recorded, both for synthetic-patient generation and for GNN training. Each run is then judged by
   validation and by the formal Test. And is fixing every controllable variable enough by academic
   standards, given that PyG's GPU kernels are nondeterministic?

## 2. Summary

**Question 1: the paper does not require it; whether the institution separately agreed to it is not
recorded.**
- For the disease task, under the paper and the released code's default settings, validation and
  test differ in four ways (the 1,000-disease validation set is the code's default; the paper does
  not separate training from validation sampling; the gene and patients-like-me tasks use their data
  differently, §3.1):
  - population: simulated against real patients;
  - candidate set: about 1,000 sampled diseases against all 21,233;
  - GNN context: a per-batch sampled subgraph against either a cohort-wide sampled subgraph or a
    full-graph pass (the paper does not say which produced its numbers);
  - metric: exact-disease MRR against a category-level top-10 measure.
- The paper reports no validation numbers and never discusses the gap. As far as the paper shows,
  validation only selects hyperparameters and checkpoints (inference).
- The fidelity direction recorded in this repository governs:
  - the scorer policy: GNN-primary, SP out of the ranking;
  - the evaluation structure: disease-disjoint synthetic validation for selection, real cohorts for
    the final evaluation.
- No document ties the A/B/C/D ladder, or the reconciliation of validation and Test numbers, to
  paper fidelity. The decided data division mirrors the paper's structure, so the question does not
  arise from it either.
- Technical judgement: the ladder need not be kept to make validation and Test numbers agree.
- That does not settle what the institution agreed to. Policy Gate 1 still requires a "B-0
  measurement report" and refers to "the agreed B-0 scope", which is not recorded (§3.5). Whether
  the institution agreed to particular comparisons or deliverables is open, and has to be asked.

**Question 2: the data half can be done today; the training half cannot yet.**
- The dataset and the generation seeds reproduce byte for byte in a CPU probe on the demo KG, and
  are recorded and digest-checked.
- On the training side, the largest gap is that the model is built before the seed is applied, so
  the recorded seed does not govern the initial weights (M3a). Others remain (§4.2): effective AMP
  and the worker count are not recorded, there is no environment record and no deterministic
  option, and HGT's kernel path is chosen by timing.
- GPU nondeterminism is not beyond reach for this model. On a fixed GPU and library stack it is
  expected to be removable without modifying PyG (source-level, §4.3):
  - PyTorch's deterministic mode covers every atomic operation this model runs;
  - for HGT, one PyG backend flag must also be pinned.
- Academically, fixing and recording every controllable input is a necessary base, not the whole of
  it. Common research practice is to repeat runs and state which sources of variation the reported
  spread covers. Whether determinism is needed, and what spread is acceptable, depends on the
  study's purpose and the decision it supports (§4.4). The number of runs varies between sources,
  and none sets one for this project. Studies also find that nondeterminism can matter far more for
  subgroup and per-case results than for top-line metrics.
- The comparison the owner describes is a **training-time** AMP comparison. The retirement
  assessment's option (b) is an **evaluation-time** comparison: one model, scored with autocast on
  and off. The two are different experiments (§4.6).

---

## 3. Question 1 in detail

### 3.1 How the paper validated and tested

| | Validation | Test (the reported results) |
|---|---|---|
| **Patients** | Simulated. Train and validation are split by disease: "patients with the same disease are either entirely in the training or fully in the validation set … 36,224 and 6400 patients" (Methods) | Real: UDN (465) and MyGene2 (146), each the evaluation cohort of a task |
| **Use** | Hyperparameters by random search, chosen "to optimize the mean reciprocal rank … on the disease-split simulated validation set", which "is entirely independent of the evaluation dataset". Checkpoints by "early stopping based on validation performance", with the set not named. The paper's Results sections also call the real cohorts "validation", and a commented-out upstream configuration validates on MyGene2 (`upstream:project_config.py:18-19`); simulated validation is the most consistent reading, but which set selected the paper's checkpoints is not shown. The upstream code has no early stopping: it keeps every epoch's checkpoint, with validation MRR in the filename, and the best one is chosen by hand (`upstream:shepherd/train.py:283-292`, `:84`, `:202`, `:208`) | Reported |
| **GNN context** | A sampled computation graph per batch: the batch's own nodes, plus random sparse nodes, expanded by neighbour sampling (paper, *Additional training details*; `upstream:shepherd/samplers.py:365-387`, `:555-577`) | A cohort-wide sampled graph (`upstream:shepherd/train.py --do_inference`) or a full-graph pass (`upstream:shepherd/predict.py`). Which one produced the paper's numbers is not stated |
| **Disease candidates** | 1,000 random KG diseases plus the batch's correct diseases, pooled for the batch (`upstream:shepherd/hparams.py:101`; `upstream:shepherd/samplers.py:522-534`; `upstream:shepherd/utils/loss_utils.py:154-177`) | Every KG disease, 21,233 (`upstream:shepherd/hparams.py:102`; Table 1) |
| **Metric** | Exact-answer MRR and top-k (`upstream:shepherd/gene_prioritization_model.py:493-507`; `upstream:shepherd/patient_nca_model.py:455-467`), with ranks from `rankdata(method='average')` (`upstream:shepherd/patient_nca_model.py:85`, `:96`; `upstream:shepherd/task_heads/gp_aligner.py:150`) | Genes: recall@k and win rates (Methods also says MRR was measured). Patients-like-me: each UDN patient queries the whole MyGene2 cohort, recall@k (75 queries). Disease: the share of top-10 diseases in the patient's disease category, 45.7% against 16.4% at random, plus case studies. **No exact-disease rank or MRR** |

**What the paper says about the gap: nothing.**
- None of its 17 mentions of "validation" reports a validation metric.
- The Supplementary text has none either. Text inside figure images could be missed.
- The gap between validation and test is never compared, explained or treated as an obstacle.
- Seed practice: gene-task error bars are "standard deviation over models trained with five random
  seeds" (Fig. 3a, 3d).

### 3.2 What this repository records about paper fidelity

| Where | Wording | What it governs |
|---|---|---|
| `docs/DISEASE_SCORER_POLICY.md:199-204` (institutional decision, 2026-08) | "reproduce the original SHEPHERD design as the primary direction, deviating only where a design is demonstrably better and supported by citable evidence; … disease ranking is GNN-primary; … SP is removed from ranking" | Scorer and ranking policy |
| `docs/working/EVALUATION_COHORTS.md:180-182`, `:734-741` | The two gating answers, taken "on the institution's standing policy of staying aligned with the published method", confirmed by the institution as reported by the owner on 2026-10-02 | Evaluation design: no synthetic unseen-disease test; disease-disjoint validation without a refit |
| `docs/working/EVALUATION_COHORTS.md:856-859` | "This adopts the paper's central evaluation structure": disease-disjoint synthetic validation for selection, real cohorts for final evaluation, no synthetic test partition and no refit | The decided five-step path (`:848-854`) |

**What no document says.** None of them extends paper fidelity to measurement attribution, to the
A/B/C/D ladder, or to reconciling validation and Test numbers.

**What is recorded about the two numbers:**
- Numerical agreement between them is rejected: "Calibrate Mode C against the trainer's `val_mrr`" is
  withdrawn as "premise false" (`docs/working/BACKLOG.md:376`).
- What stays open is a different question: whether the disjoint validation metric separates
  candidate models well enough to choose between them, which is "not measured"
  (`EVALUATION_COHORTS.md:796-797`).

### 3.3 What B-0 and the ladder were for

1. **B-0 came first.** It was already an approved work item in the policy's first commit (`db0ecad`).
   Its named comparison was "C-vs-E′": the η mixture against raw cosine.
2. **E′ was made optional the same day.** It became "optional and non-gating under the agreed B-0
   scope", and remains a revisit condition if it is ever run (`54a473f`; now
   `DISEASE_SCORER_POLICY.md:316-318`).
3. **The A/B/C/D ladder was first written down a day later** (`3a255f3`), to tell apart what the
   offline evaluator measured and what the deployed scorer does
   (`docs/working/scorer-measurement/README.md:3-7`). B-0 already had lettered comparison modes at
   `db0ecad`, and Mode A's code landed earlier that day (`62b8103`).
4. **The training `val_mrr` came in later still.** It was proposed as a calibration target after the
   frozen evaluator proved unrunnable. Calibration against the `val_mrr` number was withdrawn
   (`docs/working/BACKLOG.md:376-377`; the first point under "What is recorded about the two numbers"
   in §3.2). What was adopted instead is a same-batch differential test against the trainer's own
   validation pass (`BACKLOG.md:378`): it compares two scoring paths on identical batches, not the
   validation and Test numbers.

### 3.4 The paper's auxiliary scores, by task

The owner reports that these were already presented to the institution. They are restated here only
because they bear on Question 1.

| Task | Score | Is the GNN the only ranking signal? |
|---|---|---|
| Causal gene discovery | Eq 14: `η·EMBSIM + (1−η)·SPLSIM`, with SPLSIM from Eq 13. The run script sets α = 0.5 (`upstream:shepherd/run_causal_gene_discovery.sh:18`); `train.py`'s default is 0 (`upstream:shepherd/train.py:76`). SP also enters the training loss (`upstream:shepherd/task_heads/gp_aligner.py:124-126`, the batch-wide branch used when `step_type == 'train'`, `upstream:shepherd/gene_prioritization_model.py:128`) | No: mixed with SP |
| Patients-like-me | Eq 16: `−‖z_Pi − z_Pj‖²` | Yes |
| Novel disease characterisation | Eq 18: `−‖z_d − z_P‖²` over every KG disease | Yes |

**Squared distance ranks like cosine here.** Upstream computes Eq 16 and Eq 18 on L2-normalised
embeddings: `LpDistance(power=2)` (`upstream:shepherd/utils/loss_utils.py:267-268`), and pytorch-metric-learning
v0.9.98 normalises by default. On unit vectors ‖a−b‖² = 2 − 2cos, so the rankings equal cosine
rankings of the projected embeddings. The policy removes SP from disease ranking, which matches what
the paper did for its disease task. That is not yet implemented: until B-1, the served disease score
is still `0.7·((cos+1)/2) + 0.3·SP` (`docs/DISEASE_SCORER_POLICY.md:123`).

### 3.5 What this means for the A/B retirement assessment

- **Paper fidelity gives no reason to keep A→B and B→C (inference).** They answer a question the
  paper itself did not ask.
- **The question to the institution is still needed.** "The agreed B-0 scope" remains unrecorded, so
  the assessment's question to the institution (its §8 item 4) still stands. *Superseded by the
  owner's clarification of 2026-10-12:* the owner reported that the institution had already
  authorised retiring the old pipeline, Modes A and B included, so §8 item 4 is closed. Gate 1
  stays; its deliverable now compares the served ranking (Mode D) with the planned GNN ranking
  (Mode C).
- **Nothing in the assessment changes** until the owner decides, and this document does not clear
  or change Gate 1.

### 3.6 Further fidelity gaps that bear on validation and Test (not a complete list; none of them part of the A/B question)

The repository does not implement the reference paper's method
(`docs/working/scorer-measurement/ASSESSMENT_MODE_AB_RETIREMENT.md:54`). The scorer deviations are
recorded in `docs/DISEASE_SCORER_POLICY.md` §2 and `docs/working/scorer-retraining/README.md`: the
served SP fusion and BFS-gated candidates until B-1, interim raw cosine, and a contrastive objective
in place of Eq 19. The table adds the gaps that bear on validation and Test.

| Gap | Paper | This repository |
|---|---|---|
| Validation candidate set | About 1,000 sampled diseases plus the batch's answers; untruncated ranks | The diseases of a sampled 2-hop subgraph, seeded by each answer and its negatives. Ranked and truncated at 20 (`src/training/trainer.py:689`). `val_mrr` is computed on that window, so the window shapes checkpoint selection |
| Negatives | 1,000 random diseases per batch | `num_negative_samples = 5` per sample (`src/kg/data_loader.py:96`). Validation never passes the training setting and always uses the default (`scripts/train_model.py:605-614`) |
| Patient encoder | "a transformer encoder and an attention-weighted average" | An unweighted mean (`src/inference/scoring.py:140-145`) |
| Synthetic patients | The authors' simulator | A degenerate generator, "below" the simulator's own degenerate baseline (`EVALUATION_COHORTS.md:366-370`) |
| Disease Test metric | Category-level share of the top 10 | The formal Test reports exact-disease MRR and hits@k (Mode C) |

The repository-side facts of the first four rows are recorded elsewhere; the negatives only as an
untracked finding (`ASSESSMENT_MODE_AB_RETIREMENT.md:684`). The paper side of the first row is not:
the repository describes upstream's validation pool as every KG disease (§3.7 item 3). The fifth is a
project choice. It is listed here because it is a departure from the paper's reported metric, and no
document found here records it as one.

### 3.7 Repository texts that describe the paper imprecisely (for a later documentation fix)

1. **`docs/SP_SCORE_GUIDE.md:420`** says SPLSIM is "min–max rescaled across the candidate set".
   - Upstream uses fixed constants on each gene's mean distance over the patient's phenotypes:
     `x_min = 1`, and `x_max = 6.84` for the mean (`upstream:data_prep/shortest_paths/add_spl_to_patients.py:18-19`, `:45`,
     `:105`).
   - The paper's printed NORM has a typo (`x − max` where the code has `x − x_min`); the code
     computes `2(x−x_min)/(x_max−x_min) − 1`.
2. **`docs/working/EVALUATION_COHORTS.md:893`** says early stopping runs on the unseen-disease
   validation "as it does upstream".
   - Upstream has no early stopping: it keeps every checkpoint, and the best is chosen by hand.
   - The paper names no validation set for early stopping.
   - A commented-out upstream configuration validates on MyGene2 (`upstream:project_config.py:18-19`).
3. **`docs/working/EVALUATION_COHORTS.md:541-543`** says a held-out disease "stays in the candidate
   pool, which is every disease in the KG". That holds for the source pool and for test. Upstream
   validation ranks against 1,000 sampled diseases plus the batch's answers (§3.1).
4. **`docs/DISEASE_SCORER_POLICY.md:155-161` and `docs/working/PLAN_TEST_RESULTS.md:383-385`** are
   right about the equations. They should add that upstream normalises embeddings first, so the
   ranking equals a cosine ranking (§3.4).
5. **`docs/SP_SCORE_GUIDE.md:347` and `docs/DISEASE_SCORER_POLICY.md:172`** say the supplementary
   materials were not examined. They now have been. They contain no SPLSIM ablation, only the
   rank-against-distance panels, Supp. Fig. 3b and 4d.
6. **`docs/working/task-scope/README.md:299`** lists η as unverified. The run script sets 0.5, with
   the caveat above that the script does not run as committed.
7. **No change needed for the list sizes.** The repository's list sizes (EXPERT-CURATED 13.3,
   VARIANT-FILTERED 244.3; `SP_SCORE_GUIDE.md:355`) agree with the paper's Results and figure ranges.
   The paper's Methods sentence states them the other way round, so the paper is inconsistent.

---

## 4. Question 2 in detail

### 4.1 What is fixed and recorded today

| Link | State |
|---|---|
| Synthetic patients, allocation and node features | **Fixed and recorded.**<br>• Every stream is `derive_stream(seed, name)`, which does not depend on `PYTHONHASHSEED` (`src/kg/disease_allocation.py:122-142`, used at `:346`; `src/kg/sample_generator.py:263-268`, `:695`). Node features use a recorded `feature_seed` (`src/kg/workspace.py:346-351`).<br>• The split manifest records both seeds and every file's SHA-256 (`disease_allocation.py:203-209`; `sample_generator.py:299-314`). Training and measurement refuse a workspace that does not match.<br>• CPU probe (`write_workspace` on the demo KG; the build CLI on real sources was not run): two builds with seed 42 gave all eight workspace files byte-identical, across two processes with different `PYTHONHASHSEED`.<br>• A real build reproduces only if the ontology and annotation source files are identical too; their digests are recorded in `kg.provenance.json` |
| Training seed: batch order, negatives, subgraphs | **Fixed, if the worker count is fixed.** The trainer seeds before the first iteration, and PyTorch seeds workers from the main generator.<br>CPU probe: the same seed and the same worker count gave identical epochs; a different worker count gave different negatives and subgraphs.<br>The worker count is not recorded in the checkpoint (M3a) |
| Training seed: initial weights | **Not fixed.** The model is built (`scripts/train_model.py:751`) before `Trainer.__init__` seeds (`src/training/trainer.py:255-257`), and torch seeds its default CPU generator randomly in every process.<br>CPU probe: two fresh processes with `--seed 42` gave different initial-weight digests.<br>The defect reaches every entry point, because the WebUI and API launch the same script. Its fix is M3a's (`docs/working/PLAN_PROVENANCE_CONTRACT.md:1575-1577`), not implemented |
| AMP state | The checkpoint records the *requested* `use_amp`, `amp_dtype`, `seed` and the resolved device (`src/training/callbacks.py:304-308`).<br>Effective AMP is `use_amp and device == cuda` (`trainer.py:380`); it is derivable but not stated (M3a, `PLAN_PROVENANCE_CONTRACT.md:1572`). `auto` falls back to CPU silently, taking AMP with it (M3a, `:1578`) |
| Judging both runs by the formal Test | Mode C scores any two checkpoints on one cohort file, records digests and per-patient ranks, and runs without autocast, as serving does. So both arms are judged in fp32.<br>On CUDA, Mode C is itself not bit-deterministic: its full-graph forward uses `scatter_add_`, so repeated runs of one checkpoint can differ (`docs/working/PLAN_TEST_RESULTS.md:473-476`), and measurement gets a deterministic switch only in phase 2 (`:514-515`).<br>The formal Test is a supplied cohort (`src/evaluation/cohort.py:70` generates only train and val), and its importer does not exist yet (item 14). Measuring on the validation split gives a number contaminated by selection |

Further facts for the comparison:
- **Selection is scored under the arm's precision.** With AMP on, the validation pass, and therefore
  the score that selects the checkpoint, runs under autocast (`trainer.py:666`).
- **AMP is partial for HGT.** HGT's convolution always runs in fp32 with autocast disabled
  (`src/models/gnn/layers.py:176-182`), so AMP changes only the other layers.
- **The WebUI cannot run an HGT AMP-on arm.** It forces AMP off for HGT
  (`src/webui/components/training_console.py:239`). That arm can run only from the CLI, YAML or API.
- **Selectors break ties differently.** Serving prefers the newest file (`src/utils/checkpoint_paths.py:135`),
  while EarlyStopping restores the earliest best epoch. The selection rule has to be fixed in
  advance.
- **Arms collide by default.** Default checkpoint, output and log paths are shared between runs, so
  each arm needs its own directories.

### 4.2 Gaps for "only AMP differs"

| Gap | Covered by |
|---|---|
| The seed is applied after the model is built | M3a |
| Effective AMP is not stated; `auto` falls back to CPU silently | M3a |
| The loader and sampling settings (worker count) are not in the checkpoint | M3a |
| No environment record: GPU, driver, cuDNN, determinism flags | M3a, through §5.3 of the contract |
| No deterministic option in training | None |
| No deterministic option in measurement; Mode C on CUDA is not bit-deterministic | `PLAN_TEST_RESULTS.md` phase 2 |
| The worker count and the AMP dtype can be set only through YAML; a missing `--config` file and unknown keys are ignored silently (`scripts/train_model.py:353`, `:357`) | None |
| HGT's kernel choice is made by timing (§4.3) | None |
| No importer for the formal Test cohort | Item 14 |

### 4.3 GPU nondeterminism: does PyG have to be modified?

**Expected no, for this model on a fixed GPU, driver and library stack.** This is source-level and
was not run on CUDA.

**What PyTorch's deterministic mode covers.** In torch 2.10, `torch.use_deterministic_algorithms(True)`
makes the atomic CUDA kernels this model reaches act deterministically (installed docstring,
`torch/__init__.py:1379-1406`); in the source they are rerouted to sort-based `index_put` paths:
- `scatter_add_`: PyG's sum aggregation and the softmax denominator
  (`torch_geometric/utils/_scatter.py:67-70`); deterministic on CUDA since torch 1.13 (PR #79466);
- `index_add_`, the backward of the message gather `index_select`; deterministic since torch 1.9;
- the backward of `gather`.

None of the ops this model dispatches is among the 47 sites in torch 2.10 that raise under the mode.

**Smaller items:**
- The softmax max uses `scatter_reduce` amax, whose value does not depend on order.
- `F.cross_entropy` on `[N, C]` logits does not raise, although the docstring lists NLLLoss.
- `CUBLAS_WORKSPACE_CONFIG` is no longer required in torch 2.10 (PyTorch PR #161749).
- The cuDNN flags have no effect: this model runs no convolution, RNN or batch norm.

**Within PyG:**
- The only PyG-side source is HGT's `HeteroLinear`. It chooses between `pyg_lib.segment_matmul` and a
  plain loop by timing both at run time (`torch_geometric/nn/dense/linear.py:258-285`, `:301-319`).
- `torch_scatter` and `torch_sparse` are not on this model's path.
- PyG's own determinism switch applies only when a CSR `ptr` exists, and this project passes plain
  `edge_index` tensors.

**CPU probe, re-run by hand** (one training step with the deterministic mode on; demo workspace with
four node types):

| Run | Result |
|---|---|
| GAT | No error. Repeat runs bitwise equal |
| HGT, PyG's default (timing decides) | No error. Repeat runs bitwise equal; gradient digest `503046ea…` |
| HGT, `torch_geometric.backend.use_segment_matmul = False` | No error. Repeat runs bitwise equal; gradient digest `0d5580c0…`, **different from the default** |
| HGT, `use_segment_matmul = True` | **Crashes** in this configuration: `IndexError` in `HGTConv.propagate`, because the node-type order of `x_dict` differs from the layers' order (below) |

**Two consequences:**
1. **The timing-based choice is a real source of run-to-run difference.** The two paths give
   different numbers, and the timing cache is not saved in the checkpoint. An evaluation or serving
   process re-times, so it may score through a different path than training used.
2. **Setting the flag to `True` breaks this model in this project's configuration.**
   - With the flag on, `HeteroDictLinear` takes `pyg_lib.grouped_matmul`. That path computes outputs
     in the order of its own layers but assigns them in `x_dict`'s order
     (`torch_geometric/nn/dense/linear.py:437-449`).
   - **The trigger is an order mismatch, not `True` as such.** The reviewer reproduced it
     independently on CPU with a two-layer HGT on the demo graph: when the two orders match, the
     default, `False` and `True` all complete forward and backward; with the layers ordered phenotype,
     gene, disease, mouse_gene and the inputs ordered phenotype, disease, gene, mouse_gene, `False`
     completes and `True` raises `IndexError` in message passing. This project's orders differ in
     exactly that way. Where row counts happened to fit, the result could be silently wrong
     instead of a crash.
   - The default takes that path only from ten node types up (`linear.py:434-435`).
     `src/kg/builder.py` names eight node types (seven with literature off, its default), so the
     current knowledge graph is expected to stay below ten. That is unverified for a production
     graph: no real data is in the repository, and `NodeType` has more values (`src/core/types.py:40-74`).
   - So it is a latent hazard: a graph with ten or more node types would send default HGT down the
     broken path.
   - Both this and the timing-based choice are recorded as BACKLOG item 20.

**A recipe for determinism, if wanted:**
1. Fix the seed before the model is built (M3a). Without this, nothing else matters.
2. Call `torch.use_deterministic_algorithms(True)` at process start.
3. HGT only: set `torch_geometric.backend.use_segment_matmul = False`. **Never `True`** while the
   order mismatch above exists. This is an existing library setting, not a second scoring path, but
   it selects different kernels: pinning it is its own change, with the setting recorded and its
   numerical and performance effect checked.
4. Keep `torch.set_float32_matmul_precision('highest')`. It is the default today, and the project
   never changes it.
5. Keep the worker count and `--compile` fixed.
6. Accept the result only if two same-seed runs on the target GPU give bitwise-equal weights after N
   steps. That verifies the specified path and environment only, not every shape, AMP dtype,
   compile setting or later version.

**Its cost has to be measured.** The literature reports about 12.6× at the op level and about 2.7×
end to end for GraphSAGE, on an older PyTorch (Shanmugavelu et al., SC24-W). Subgraph sampling in
Python is this project's per-batch bottleneck, so the slowdown is probably smaller.

**What stays uncontrolled even then:**
- a different GPU architecture or SM count: this affects cuBLAS, and dropout's launch grid depends on
  the SM count;
- different torch, CUDA, PyG or pyg_lib versions, and CPU against GPU (PyTorch's reproducibility note
  promises nothing across releases or platforms);
- AMP itself, which is the variable under study;
- a resumed run, because checkpoints store no generator state.

**History.** By source-level reading, the original SHEPHERD stack could not have been made
deterministic on GPU:
- it ran torch 1.8, PyG 1.7.2 and torch_scatter 2.0.8;
- under the flag, `scatter_add_` and the `index_select` backward raised in torch 1.8;
- Lightning 1.4.5's `deterministic=True` only sets the cuDNN flag.

Its five-seed error bars therefore probably include kernel nondeterminism (inference; it also
assumes the paper's runs used the released code). The PyG maintainers' statements that GPU scatter
is nondeterministic ("Scatter is a non-deterministic operation by design", #2788; similar wording in
#446, #3175 and discussion #8266) date from before torch 1.13 made `scatter_add_` deterministic, or
describe the default behaviour.

### 4.4 How research treats this

**A common practice is to repeat runs and report the spread, stating what it covers. Each practice
below has its own scope:** OGB's seed rule governs its leaderboard, and the NeurIPS checklist asks
that uncertainty and its sources be stated without fixing a number of runs. None of them is an
acceptance threshold for this project.

| Source | Practice |
|---|---|
| Open Graph Benchmark (Hu et al., NeurIPS 2020; leaderboard rules) | Mean and unbiased SD "over 10 different random seeds. You can either not fix random seeds at all, or use the random seeds from 0 to 9" |
| Errica et al., ICLR 2020 | Three runs per fold, "to smooth the effect of unfavorable random weight initialization" |
| Dwivedi et al., JMLR 2023 | "each experiment with 4 different seeds" |
| Shchur et al., NeurIPS 2018 workshop | 100 random splits × 20 initialisations |
| SHEPHERD (2025) | Five seeds, gene task |
| PyG paper (Fey & Lenssen, 2019) | Discloses: "training results can vary across the same manual seeds" |
| Ma et al., TMLR 2026 | Discloses that PyG scatter is nondeterministic on GPU, so results "might vary in an acceptable range" |

**How large the effect is:**
- On top-line metrics, implementation-level nondeterminism is about as large as seed variance:
  - Summers & Dinneen, ICML 2021: ResNet-14 accuracy SD 0.22% from cuDNN alone against 0.23% from
    initialisation alone;
  - Zhuang et al., MLSys 2022: "Top-line metrics do not differ substantially between algorithmic and
    implementation factors".
- It is much larger on per-class and subgroup metrics:
  - Pham et al., ASE 2020: up to 52.4% per-class difference;
  - Zhuang et al.: up to 3.3× subgroup SD.
- Per-patient rankings in a rare-disease cohort are a subgroup-sensitive setting.
- Zhuang et al. also find that controlling noise only partly barely reduces it: "Non-determinism must
  be controlled at all levels of the technical stack or is not worth controlling at all."

**Reporting standards:**
- The NeurIPS checklist (item 7) asks for error bars and a statement of "the factors of variability
  that the error bars are capturing".
- TRIPOD+AI (BMJ 2024; items 18f, 22 and 23a) asks for the analytical code, the full model and
  performance estimates with confidence intervals. It has no item on seeds or hardware.
- Heil et al. (Nature Methods 2021) name GPU nondeterminism as a reason the minimal standard requires
  publishing the trained model.

**Mixed precision:**
- Micikevicius et al. (ICLR 2018) and NVIDIA's guide report that AMP matches fp32 accuracy, from
  single runs.
- No source found prescribes a statistical equivalence protocol for AMP.
- One GNN study found fp16 aggregation unstable in another framework (Tarafder et al., HPDC 2025).
- This model's aggregation probably stays fp32 under autocast, through type promotion. This rests
  on the autocast docs, source reading and a CPU autocast probe of GAT and SAGE layers, and is
  unverified on GPU.

**Answer to the owner's academic question:**
- Fixing and recording every controllable input is the necessary base.
- Repeating runs, and stating what the spread covers, is common practice. The sources use 3 to 10
  runs, or splits × initialisations; none sets a number for this project. The 5–10 seeds in §4.5 are
  a proposal to be weighed.
- A spread across seeds that includes GPU nondeterminism measures the combined variation. It does
  not separate the two sources, and it does not show that the effect on individual patients is
  acceptable.
- The cited magnitudes (CNNs, GraphSAGE) show what can happen and how to measure it. They are not
  estimates for this project's HGT or GAT model.
- Whether GPU nondeterminism must be removed, or only measured and disclosed, depends on what the
  comparison is for and how sensitive the decision is (Zhuang et al. make the same point).
  Per-patient rankings are a sensitive setting.
- For this model it is also expected to be removable on a fixed stack (§4.3), if the cost is
  acceptable.

### 4.5 A design for the comparison (for discussion)

1. **Prerequisites:**
   - M3a's seed-before-build fix;
   - effective AMP and the worker count recorded;
   - for HGT, `use_segment_matmul = False` in both arms;
   - for a formal external Test comparison, a cohort imported through item 14 and scored by Mode
     C. The seed fix, the HGT path and M2.1 do not depend on that importer. Their tests may use small
     controlled data, which are not institutional acceptance results.
2. **Fixed across arms:**
   - one verified workspace, revision, GPU, worker count, batch size and AMP dtype;
   - compile off;
   - separate output, checkpoint and log directories;
   - the selection rule, including ties.
3. **Seeds.** The same list of 5–10 seeds in both arms, paired by seed.
4. **Noise floor.** For training, measure either:
   - same-seed fp32 repeats with deterministic mode off, which measure the run-to-run variation left
     under the controlled conditions (it is a pure GPU noise floor only if loader, CPU and
     execution-order effects have been excluded); or
   - deterministic mode in both arms, which is expected to make the within-seed noise zero
     (source-level; it must pass §4.3's bitwise check on the target GPU, and it is unverified for
     cuBLASLt fp16 paths under AMP). Its kernels give different but repeatable numbers, so both arms
     must use the same mode.

   For evaluation, repeat Mode C on one checkpoint on the target GPU. A CPU Mode C run (recorded
   `cuda_executed=false`) can be a clearly labelled diagnostic comparison. It does not replace the
   CUDA evaluation, and CPU–GPU closeness does not show that CUDA runs repeat.
5. **Purpose first.** Before running, state whether the study only describes the difference or must
   support a decision ("not worse than" or "equivalent").
   - **Descriptive.** Report, per arm, mean ± SD of the selected epoch, the selection `val_mrr`, and
     Mode C MRR and hits@k on the same Test cohort; the paired differences with 95% CIs; per-patient
     rank changes between arms, beside those between fp32 seeds; GradScaler skipped steps, training
     time and memory; the full environment; and the limits. **No equivalence claim.**
   - **Decision.** Specify in advance the metric, the direction of the difference, a practically
     acceptable margin δ, and the criterion. δ is set by whoever owns the decision; the fp32
     seed-to-seed SD describes noise and is not a margin. A simple conservative criterion: the claim
     is made only if the chosen CI lies entirely within the pre-specified range (±δ for
     equivalence; above −δ for "not worse than"). Otherwise the result is **inconclusive**. Not
     finding a difference does not show the difference is smaller than δ (Lakens 2017).
   - **Example.** Mean difference 0, 95% CI [−0.03, 0.03], δ = 0.01: "within ±0.01" cannot be
     claimed; the data cannot rule out a relevant difference.
6. **Scope of each interval.** A CI across training seeds on one fixed cohort covers training
   variation on that cohort. It is not a CI for generalisation to new patients, which is a
   patient-level interval of a different kind (TRIPOD+AI item 23a). State which sources each
   interval covers.
7. **Wording.** Descriptive: "difference X (95% CI …)". Decision: "equivalent within δ" or "not
   worse than δ" only when the criterion is met, otherwise "inconclusive". Never "identical".

### 4.6 Training-time AMP against evaluation-time autocast

**The two experiments:**
- **Training-time AMP** produces two trained models, each judged in fp32 by Mode C. It answers
  whether training with AMP gives a worse model, which is the owner's question.
- **Evaluation-time autocast** scores one model with autocast on and off. That is the retirement
  assessment's option (b) for the institution's AMP question (its §8 item 7).

**Where evaluation-time autocast matters.** Serving and the Mode C CLI run without autocast. So
evaluation-time autocast affects only the validation pass inside training, that is, which
checkpoint is selected. A training-time comparison includes that effect in its AMP arm, but mixed
with the effect of training under AMP. It does not isolate it, and it does not give option (b)'s
same-model, same-batch, per-sample quantities. Replacing option (b) with a training-time study
needs an explicit decision about which question has to be answered; a larger study does not
automatically satisfy the smaller one, and neither study is required merely for completeness.

**What the institution meant is not recorded.** The institution's recorded position ("the switch
stays a switch … run it both ways if they wish", `docs/working/BACKLOG.md:491-497`) sits in the
context of item 7a's evaluation-time leg. `TrainerConfig.use_amp` controls both training and
validation autocast, so whether the institution also meant training-time AMP is not recorded.
*Superseded by the owner's clarification of 2026-10-12:* what the owner and the institution want
to know is whether an AMP-trained model is worse, which is the training-time experiment above.
The evaluation-time evidence (the differential's AMP report, option (b)) is no longer a
precondition for deleting the differential or Modes A and B, and the assessment's §8 item 7 is
closed. The training-time study is bounded research (BACKLOG item 21), not a prerequisite of M2
or of the retirement.

**Upstream precedent.** The released upstream code trains in fp32: its hyperparameters contain
`'precision': 16`, but no trainer is passed it, and Lightning 1.4.5 defaults to 32. What precision
the paper's own runs used is not shown, since the released code does not run as committed. The
paper never mentions mixed precision, and there is no upstream evidence on what AMP does to results.
An AMP comparison is therefore this project's own addition.

---

## 5. Points for discussion (none decided)

1. Is the limited answer to Question 1 accepted: the paper does not require reconciling validation
   and Test numbers? The institution's agreed B-0 scope still has to be confirmed, and this document
   does not clear Gate 1. Should that question cite §3.1 and §3.4?
2. Should the fidelity gaps of §3.6 become tracked items? The validation candidate set (five
   negatives, a top-20 window) shapes checkpoint selection, and the paper's differs.
3. Should the texts of §3.7 be corrected in a documentation-only change?
4. Should M3a's seed-before-build fix come forward? It is a prerequisite for any paired comparison.
5. **Determinism.** Should the comparison use deterministic mode, whose cost has to be measured, or
   multi-seed only? Should the HGT kernel path be pinned (`False`) in training and evaluation alike,
   whatever is chosen?
6. Should the assessment's option (b) be replaced by the training-time comparison of §4.5? That is a
   proposal to the owner, and through the owner to the institution.
7. Which cohort serves as the formal Test for the comparison? It needs item 14's importer.

**The reviewer's recommendations on these points (2026-10-11), not yet the owner's decisions:**
1. Accept that the paper does not require reconciling validation and Test numbers. Do not answer the
   unknown B-0 scope as "no"; keep the limited question to the institution.
2. Link the deviations that actually affect product design to existing backlog items. Do not open a
   separate paper-parity programme, and do not change the scorer because of the literature review.
3. Correct the texts of §3.7 in a separate documentation change, kept apart from any change of
   product policy.
4. The seed-before-build fix is the known M3a item. It can come forward as a small change through the
   existing training entry and configuration, without a seed-wrapping entry point, scheduled with S8
   or M3a.
5. First a small GPU probe in the target environment, then decide whether a deterministic or
   multi-seed study is needed. The two are not exclusive: each seed's run can itself be
   deterministic. Handle the HGT defect separately from the study design, and fix it properly rather
   than working around it in an experiment script.
6. Do not assume a training-time study replaces the evaluation-time evidence. Decide which question
   needs answering and keep the smallest sufficient experiment; do not require both for
   completeness.
7. The formal Test uses item 14's import and Mode C when they are ready. Tests of the underlying
   fixes may use small controlled data, and are not institutional acceptance results.

**The reviewer's suggested order:** correct this report; continue M2.1 under its approved plan, not
held for an AMP study; keep the seed order and the record of effective settings in M3a; list the
HGT order defect as its own reproducible defect; if an AMP study is chosen, start with a small
feasibility check and size it to the decision, through the existing training, validation and Mode C
paths; keep in the Mode A/B retirement only the uses and acceptance requirements that are really
still open.

## 6. Sources

**Paper and code:**
- Alsentzer, Li, Kobren, Noori, Kohane, Zitnik. *npj Digit Med* 8:380 (2025).
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12181314/ (Supplementary from static-content.springer.com).
- Upstream code: https://github.com/mims-harvard/SHEPHERD at `e95433a`.

**Frameworks:**
- PyTorch reproducibility note: https://docs.pytorch.org/docs/2.14/notes/randomness.html.
- `torch.use_deterministic_algorithms`: installed docstring, torch 2.10.0.
- PyTorch PR #79466 (deterministic CUDA `scatter_add_`, in 1.13), PR #98060, PR #161749.
- PyG issues #446, #2788, #3175, #8866, discussion #8266, PR #9009:
  https://github.com/pyg-team/pytorch_geometric.
- Fey & Lenssen, "Fast Graph Representation Learning with PyTorch Geometric", ICLR 2019 workshop,
  https://arxiv.org/abs/1903.02428.

**GNN benchmarks:**
- Hu et al., "Open Graph Benchmark", NeurIPS 2020, https://arxiv.org/abs/2005.00687; rules at
  https://ogb.stanford.edu/docs/leader_overview/.
- Errica et al., ICLR 2020, https://arxiv.org/abs/1912.09893.
- Dwivedi et al., JMLR 24(43), 2023, https://jmlr.org/papers/v24/22-0567.html.
- Shchur et al., "Pitfalls of Graph Neural Network Evaluation", NeurIPS 2018 workshop,
  https://arxiv.org/abs/1811.05868.
- Ma et al., "Plain Transformers Can be Powerful Graph Learners", TMLR 2026,
  https://arxiv.org/abs/2504.12588.

**Nondeterminism studies:**
- Pham et al., ASE 2020, https://www.cs.purdue.edu/homes/lintan/publications/variance-ase20.pdf.
- Summers & Dinneen, ICML 2021, https://arxiv.org/abs/2103.04514.
- Zhuang et al., MLSys 2022,
  https://proceedings.mlsys.org/paper_files/paper/2022/file/427e0e886ebf87538afdf0badb805b7f-Paper.pdf.
- Bouthillier et al., MLSys 2021, https://arxiv.org/abs/2103.03098.
- Shanmugavelu et al., SC24-W 2024, doi:10.1109/SCW63240.2024.00028, https://arxiv.org/abs/2408.05148.

**Reporting standards:**
- NeurIPS Paper Checklist: https://neurips.cc/public/guides/PaperChecklist.
- Collins et al., TRIPOD+AI, BMJ 2024;385:e078378, https://doi.org/10.1136/bmj-2023-078378.
- Heil et al., Nature Methods 18:1132–1135 (2021), https://pmc.ncbi.nlm.nih.gov/articles/PMC9131851/.
- Lakens, "Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses",
  Social Psychological and Personality Science 8(4):355–362 (2017),
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5502906/.

**Mixed precision:**
- Micikevicius et al., "Mixed Precision Training", ICLR 2018, https://arxiv.org/abs/1710.03740.
- NVIDIA Mixed Precision Training guide:
  https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html.
- Tarafder et al., HPDC 2025, https://arxiv.org/abs/2411.01109.

## Appendix: the CPU probe of §4.3

The digests in §4.3 come from this configuration only; a different configuration gives different
digests, so they certify nothing beyond it. The probe script is kept outside the repository.
- **Workspace.** `write_workspace(build_demo_kg(), ws, feature_dim=16,
  samples=SampleBudget(num_train=40, num_val=10, val_disease_fraction=0.2, seed=42))`, with
  `build_demo_kg` from `scripts/setup_demo.py`. Four node types: phenotype 10, gene 8, disease 5,
  mouse_gene 1.
- **Process.** `torch.use_deterministic_algorithms(True)` before any model code is imported; the
  PyG flag, when pinned, set before construction.
- **Model.** Built as `scripts/train_model.py` did at `8421b8c` (`load_graph_data`, `create_dataloaders`,
  `create_model_from_config`), with `TrainConfig(device="cpu", seed=42, num_workers=0, hidden_dim=32,
  num_layers=2, num_heads=4, batch_size=8)` and `conv_type` GAT or HGT. **Unlike `train_model.py`,
  the probe calls `torch.manual_seed(0)` immediately before the model is built**, so two builds start
  from the same weights; this stands in for the M3a fix.
- **Step.** The first training batch, drawn after `torch.manual_seed(123)` and `random.seed(123)`.
  `torch.manual_seed(1)` before the forward pass, so the dropout masks match. One forward,
  `_compute_model_outputs`, loss and backward, as in `Trainer._train_epoch`.
- **Digest.** SHA-256 over the gradients of all parameters, sorted by name, first 16 hex digits.
  "Repeat runs bitwise equal" means a second, fresh build in the same process gave the same digest.
