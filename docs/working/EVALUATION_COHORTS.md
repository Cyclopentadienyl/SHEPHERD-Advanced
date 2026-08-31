# Evaluation cohorts — findings, and the division of labour

**Type:** findings report plus one institutional decision · **Date:** 2026-08 ·
**Status:** §1–§2 established; §3 decided; §5 open. **Revision 2** — §1.2 corrected: the original design's validation set is disease-disjoint and stays that way, so the deviation is this project's, not an addition to theirs; §5.2 answered as a condition

---

## 0. What this is, and is not

**Is:** a record of what the original SHEPHERD team actually did about train / validation /
test data — read from their repository, not recalled — what of it is obtainable, and how the
three candidate evaluation cohorts divide the work.

**Is not:** an implementation plan. Nothing here specifies code, file formats, or a schedule.
The generator change §5 implies has not been designed, and no code has been written on the
strength of this document.

**Why it exists:** the deploying institution asked how a test set should be built when its own
patient data cannot leave and arrives ten or twenty records at a time. Answering that needed the
original team's approach checked rather than assumed, and the check changed the answer.

Backlog item 11 (the evaluation-holdout protocol) is what this feeds.

---

## 1. What the original team did

Read from `mims-harvard/SHEPHERD` at commit `e95433a`, and from the external simulator it
depends on, `EmilyAlsentzer/rare-disease-simulation`. Every claim below was re-verified against
the file after the search that found it.

### 1.1 The split unit is the disease, not the patient

`data_prep/preprocess_patients_and_kg.py:274-289`:

```python
diseases = list(set([p['disease_id'] for p in filtered_patients]))
n_train = round(len(diseases) * frac_train)          # 0.70
n_val_test = round(len(diseases) * frac_val_test)    # 0.15
dx_train_patients = diseases[0:n_train]              # holds disease ids, despite the name
dx_val_patients   = diseases[n_train:n_val_test+n_train]
dx_test_patients  = diseases[n_val_test+n_train:]
dx_split_train_patients = [p for p in filtered_patients if p['disease_id'] in dx_train_patients]
```

Patients are assigned by their disease's membership, so the partitions are disease-disjoint by
construction. **This is the primitive worth taking.**

### 1.2 They folded the *third* partition back into train — validation stayed disjoint

Same file, `:294-297`, their own comment:

```python
#NOTE: we decided to merge the train & test sets into a single larger train set to be able to
# train on more diseases. We are posthoc merging to keep the code as was originally written.
dx_split_train_patient_ids = pd.concat([]dx_split_train_patient_ids, dx_split_test_patient_ids)
```

The function returns train and val only, and `main()` writes exactly two files (`:356-357`).
**The released design has no third, disease-disjoint partition.**

**What it does still have is a disease-disjoint validation set, and that distinction is the whole
point.** The slicing at `:281-283` cuts the disease list into three non-overlapping ranges, so
merging any two of them cannot intersect the third: train ends up with the first 70% plus the last
15%, validation with the middle 15%. This follows from the arithmetic and does not depend on the
code running — which, per §1.4, it does not. The shipped data files were not inspected:

| | train ∩ val, at the disease level |
|---|---|
| **Original team** | **empty** — disjoint by construction, and the merge does not touch it |
| **This project** | **7,970 of 7,970 — total overlap** ([`EVIDENCE_M4.json`](EVIDENCE_M4.json)) |

So their `val_mrr` measures generalisation to diseases the model has never seen, and ours measures
recognition of new phenotype subsets of diseases it has. Those are different quantities carrying
the same name.

**This is where the deviation actually is, and it runs the opposite way to the obvious reading.**
Nothing in this project added a separation the original design lacks. The original design has a
separation this project lost.

*Inference, not measurement.* A concern raised in discussion — that candidate models within one
training batch may fail to separate on validation — would follow naturally from a metric measured
on a totally-overlapping split, since such a metric rewards recognition rather than
generalisation and has less room to differ between models. **Neither the saturation nor the
failure to separate has been measured here**, and this document does not treat either as
established.

### 1.3 At inference, `test_data` points at everything

`shepherd/hparams.py:184`:

```python
'test_data': f'simulated_patients/disease_split_all_sim_patients_{project_config.CURR_KG}.txt',
```

No script in the repository writes a `disease_split_all_*` file; it exists only in their Harvard
Dataverse deposit. So the file their `--do_inference` path measures on includes the training
diseases.

### 1.4 Three facts that mean the split cannot be reproduced from their code

- `data_prep/preprocess_patients_and_kg.py` **does not parse.** `ast.parse` reports
  `SyntaxError line 296` — the `pd.concat([]dx_split_...` above is a typo. The module is
  unimportable, so the released split files were not produced by the committed code.
- **The split is unseeded.** `project_config.py:9` defines `SEED = 33`; it appears nowhere else
  in the repository. `:276` uses `list(set(...))` and never sorts or shuffles, so membership
  depends on Python's set iteration order. Reproduction relies on the shipped
  `_patient_ids.csv` lists, not on rerunning anything.
- **No unseen / zero-shot / holdout / disjoint machinery exists.** Those terms return zero hits
  across the repository. `README.md:65` claims only "Disease-split train and validation sets",
  which is consistent with the code and does not claim a held-out disease test set.

### 1.5 They wrote no patient simulator

SHEPHERD is a consumer. `data_prep/preprocess_patients_and_kg.py:28` imports
`read_simulated_patients` from `project_utils`, which defines no such function — it lives in the
external simulator. `README.md:68` names the source:

> More details about the simulated rare disease patients can be found
> [here](https://github.com/EmilyAlsentzer/rare-disease-simulation).

The recipe, verified in that repository's `config.py` and
`simulation_pipeline/modules/patient_simulator.py`:

| Stage | Mechanism | Parameters |
|---|---|---|
| Initialise | Per disease-associated HPO term, `np.random.binomial(1, freq)` on the **Orphanet frequency band**; success → positive phenotype, failure → *negative* phenotype (`:975-981`) | — |
| Dropout | Independent Bernoulli removal | `PROB_DROPOUT_POS = 0.2`, `PROB_DROPOUT_NEG = 0.9` |
| **Corruption** | Replace a phenotype with a **direct parent in the 2019 HPO DAG**, one generation, skipped when the parent is too unspecific (`:1025-1078`) | `PROB_CORRUPTION_POS = 0.25`, `PROB_CORRUPTION_NEG = 0.25` |
| Noise | Poisson-many HPO terms by age-conditioned population prevalence | `NOISY_POS_PHEN_SAMPLES_LAMBDA = 5`, `NOISY_NEG_PHEN_SAMPLES_LAMBDA = 3`, `PROB_NOISY_POSITIVE = 0.5` |
| **Distractor genes** | ~13 per patient from a 7-module taxonomy; 4 of the 7 inject phenotypes as a side effect | `N_DISTRACTORS_LAMBDA = 13`; `PATH_PHEN_PROB = 0.42`, `PHENO_DIST_PROB = 0.30`, `NON_SYNDROM_PHEN_PROB = 0.09`, `TISSUE_DIST_PROB = 0.08`, `INSUFF_EXPLAIN_PROB = 0.05`, `COMMON_FP_PROB = 0.03`, `UNIVERSAL_DIST_PROB = 0.03` |

`PATIENTS_PER_DISEASE = 1` in the repository, with the comment "We use 20 for the manuscript".

**Against this, `src/kg/sample_generator.py` is a degenerate case.** It picks a disease, keeps
`int(P × (1 − drop_rate))` of its phenotypes, and stops: no frequency weighting, no hierarchy
corruption, no noise phenotypes, no negative phenotypes, no distractor genes — `gene_ids` is the
disease's full gene list, identical for every sample of that disease. The simulator has a
`--random_genes` branch its authors built as the *degenerate baseline*; ours is below it.

One consequence is exact rather than statistical. Because `k` is fixed per disease, a disease
with `P` phenotypes admits exactly `C(P, k)` distinct samples: **1** for `P = 2`, 3 for `P = 3`,
6 for `P = 4`, 15 for `P = 6`. The audited workspace draws 115,000 samples over 10,576 diseases,
about 11 per disease, so by the pigeonhole principle low-phenotype diseases produce **byte-identical
duplicates**. A test split generated the same way would contain exact copies of training samples.

### 1.6 What this project's guards are, and are not

Two guards were added earlier in this phase and should not be confused with the boundary above:
`--split` is required with no default on both measurement entry points, and `read_samples` refuses
a missing split and lists what exists rather than substituting one.

Neither separates validation from test. They prevent a number being produced on a split the caller
did not ask for — which is a defect the original repository does have, at `hparams.py:184`, where
`test_data` silently resolves to the all-patients file. The guards are unrelated to
disease-disjointness and nothing here argues for removing them.

---

## 2. What is actually obtainable

| Cohort | Obtainable | Evidence |
|---|---|---|
| **UDN** | **No** | `README.md:68` "We are unfortunately unable to provide the UDN patients due to patient privacy concerns"; `data_prep/create_udn_cohort/README.md:3` withholds even the processing scripts |
| **MyGene2** | **Yes** | Listed in the Dataverse deposit (`README.md:66`, DOI `10.7910/DVN/TZTPFL`), and `data_prep/create_mygene2_cohort/retrieve_mygene2.{py,sh}` retrieve it from the public mygene2.org pages |
| **DDD** | **No**, not here | Appears only in commented-out paths and filename strings (`project_config.py:21`, `add_spl_to_patients.py:63-105`). No processing code. DDD itself is a controlled-access study |

**MyGene2 carries disease labels, in our namespace.** `preprocess_mygene2.py:96-99` writes
`positive_phenotypes`, `all_candidate_genes`, `true_genes` and `true_diseases` as MONDO ids, and
`README.md:70` states that the disease task is the one requiring `true_diseases`. Our own disease
nodes are MONDO (`src/kg/builder.py:147`), so the identifier space matches.

---

## 3. The division of labour — decided

Three cohorts, three different questions. None substitutes for another.

| Cohort | Answers | Standing |
|---|---|---|
| **MyGene2** | Does the model hold up against **real phenotype recording** — incomplete, coarse-grained, noisy — and how large is the gap between synthetic and real? | **Research comparison.** Not an acceptance gate |
| **Disease-disjoint synthetic split** | Does the model generalise to diseases it **never saw in training**? | Requires a generator change; two tiers, see below |
| **Institutional offline cohort** | Does the chosen model hold up on **this hospital's population**? | **Acceptance benchmark** |

**Confining MyGene2 to research comparison is the right call for two reasons beyond preference.**
Its disease distribution is that of a self-selected family-upload platform and has no relation to
the deploying hospital's case mix, so accepting a model on its evidence would be accepting on
evidence about a different population. And it keeps the dataset in the use it was published for:
using a research cohort as a clinical deployment gate is a different use, and would raise consent
and licensing questions that this division avoids entirely.

### 3.1 The synthetic split is two decisions, not one

§1.2 separates them, and they carry very different weight:

| | What it is | Cost | Standing |
|---|---|---|---|
| **Restore a disease-disjoint validation set** | Returning to the original design, which partitions the disease set and keeps validation disjoint | Validation diseases leave training — 15% in the original | Parity with the design being reimplemented. If the separation concern in §1.2 is real, this is what would address it |
| **Add a third disjoint test partition** | Going beyond the original design | A further slice of diseases leaves training | The original team built this and then removed it, in favour of training breadth. They had external real-patient cohorts to cover what they gave up; two of those three are unavailable here |

The first needs no justification beyond parity with the design being reimplemented. The second is
the one that needs the reserve-fraction judgement in §5.

---

## 4. What the institutional cohort can and cannot decide

The institutional cohort arrives in batches of ten to twenty records. That constraint does not
weaken an acceptance gate, but it does bound it, and the bound must be stated or the numbers will
be over-read.

Observed 80% top-10 accuracy, Wilson 95% interval:

| n | Interval | Width |
|---|---|---|
| 15 | [54.8%, 93.0%] | **38.1 points** |
| 50 | [67.0%, 88.8%] | 21.8 points |
| 100 | [71.1%, 86.7%] | 15.5 points |
| 200 | [73.9%, 85.0%] | 11.0 points |

Power to detect a real 15-point difference between two models (70% vs 85%, α = 0.05, two-sided):

| n per model | Power |
|---|---|
| 15 | **16.6%** |
| 60 | 50.3% |
| 120 | 79.5% |
| 250 | 98.0% |

**At fifteen cases the cohort cannot rank models.** A genuine fifteen-point difference would be
missed five times out of six. What it can do is act as a floor: confirm that the model already
chosen has not broken on this population.

Ranking therefore has to come from the cohorts that have the power for it, and the institutional
cohort accumulates — at roughly 120 cases the power to separate models returns. That is the
practical reason for the institution's own requirement that a test result record its **dataset
version**: only results against the same cohort version are comparable, and the cohort will not
stay the same size.

---

## 5. Open

Recorded rather than resolved. None of these is an engineering judgement.

1. **How many diseases to reserve for a third, disjoint test partition.** Applies to §3.1's
   second tier only — restoring a disjoint *validation* set is parity with the original design and
   carries the original's own 15%. Reserved diseases leave training, which is the trade-off the
   original team resolved in favour of training breadth; they had UDN, MyGene2 and DDD to cover
   what they gave up, and two of those three are unavailable here.
2. **Whether to apply for the larger institutional database — answered as a condition, not yet
   as a decision.** The ten-to-twenty figure is the single-batch extraction limit. A substantially
   larger database does exist in-hospital, behind an application and approval process.

   That makes this calculable rather than unknown. From §4: an acceptance cohort accumulating at
   ten to twenty per batch is adequate for a **floor check** on an already-chosen model and never
   becomes adequate for **ranking** until roughly 120 cases. Under the division of labour in §3,
   ranking belongs to MyGene2 and the synthetic split, so **the application is not required** —
   unless ranking *on this hospital's own population* is held to be a requirement that cannot be
   delegated to the other two cohorts. That is a value judgement for the institution, not an
   engineering one.
3. **Which criterion of record selects a model.** The built-in auto-selection reads the ranking
   metric from a checkpoint's own logs (`src/api/routes/pipeline.py:225`, priority
   `("val_mrr", "val_hits@10", "val_hits@1")` at `src/utils/checkpoint_paths.py:45`), which the
   M1–M3 audit found to be `val_mrr` in all fifteen checkpoints. That matches the institution's
   stated first stage — best validation model as the batch representative — so the two are a
   pipeline rather than competing rules. What is open is that the representative is chosen by a
   metric measured on a split with 100% disease overlap, which is the same saturation that makes
   models fail to separate on validation in the first place.
4. **Where a test result is recorded, and how it binds to a checkpoint.** A result cannot be
   written into the `.pt` without changing its SHA-256, and the M1–M5 evidence chain cites
   checkpoints by digest. A sidecar beside the checkpoint keeps the digest stable and makes
   "never tested" the absence of a file rather than a flag somebody has to maintain — the pattern
   `scripts/compute_shortest_paths.py` already uses for `<artifact>.meta.json`. Any such record
   must carry the checkpoint's digest, for the reason the shortest-path sidecar carries
   `num_pairs`: a digest identifies the sidecar, not what it describes.
5. **Whether more public academic patient cohorts should be importable.** Raised and
   **explicitly placed outside the current scope**; recorded because it constrains one near-term
   choice. If cohorts may later arrive from elsewhere — a settings field taking a URL, an import
   to a chosen location, a sampling range selected at training time — then a cohort has to be
   identified in any test record by **digest and version label**, never by filename or path, so
   that an imported cohort and a locally generated one are the same kind of thing to whatever
   reads the record. Deciding that now costs nothing; changing it after records exist does not.
   Separately, and not a reason to defer the idea: fetching a dataset by URL into a clinical
   system is a security surface, and would need its own review rather than being added to a
   settings page.
6. **Whether the generator's fidelity is addressed before or after the split question.** §1.5's
   gap — no frequency weighting, no hierarchy corruption, no noise, no distractors — inflates
   every synthetic metric, train, validation and test alike. A disease-disjoint test built on the
   current generator would be honest about *which diseases* it holds out and still optimistic
   about *how hard* they are.

---

## 6. Method

Repository claims were located by search and then re-read in the file before being written here;
file and line references are to `mims-harvard/SHEPHERD` at `e95433a` and to
`EmilyAlsentzer/rare-disease-simulation` at its default branch. The syntax error in §1.4 was
confirmed by running `ast.parse`. The intervals and power figures in §4 are Wilson intervals and
a two-proportion normal approximation, computed for this document; they describe sample sizes, not
any measurement of this system.

`mims-harvard/OptimusKG` was examined and is **not relevant** to either question: it is a
knowledge-graph construction pipeline with no model, no training loop, no split logic, and no
patient node type. Its `evals` module evaluates KG edges against the literature, which is a
different problem. It is noted only as a possible future KG substrate.
