# Evaluation cohorts — findings, and the division of labour

**Type:** findings report plus one institutional decision · **Date:** 2026-08

**Status:** §1–§2 established; §3 decided; §5 open and narrowed by §6.0's default path; §6 under
discussion except where a passage marks itself settled; §6.8 cleared to proceed with one item under
re-review. **One institutional question gates the rest — see §6.0.**

<details>
<summary><b>Revision history</b></summary>

- **6** — scope correction. §6 had grown a deployment-claim architecture in answer to a
  selection-quality defect. §6.0 records the short default path; `synthetic_test_unseen` is marked
  optional; everything conditional on a direct exact-checkpoint synthetic unseen claim is gated
  behind a requirement that does not currently exist. §1's statements are bound to intended design
  throughout; allocations are immutable, with new generations rather than edits; "no recorded
  evaluation is available" replaces "never tested"; §4.1 adds what makes a real cohort a valid
  test. §6.8's sensitivity curve is under re-review.
- **5** — eight internal-consistency corrections (upstream claims bounded to the committed
  source's intended design; the cohort-size threshold and the saturation claim removed; `C(P, k)`
  repetition distinguished from cross-split leakage), and the settled review conclusions recorded:
  §5.1's five decision dimensions, §6.4's sibling-evidence rules, §6.7's UI bounds, §6.8's scope.
- **4** — two precision corrections (the duplicate claim was about model-visible content, not
  bytes; the cohort-size threshold assumed an unpaired design where the comparison is paired), and
  §6 records the protocol shape under discussion.
- **3** — §3.1 states what holding a disease out actually removes, and that the first decision has
  a cost.
- **2** — §1.2 corrected: the upstream validation set is disease-disjoint and stays that way, so
  the deviation is this project's, not an addition to theirs; §5.2 answered as a condition.

</details>
---

## 0. What this is, and is not

**Is:** a record of how the original SHEPHERD repository handles train / validation / test data
— read from the committed source, not recalled — what of it is obtainable, and how the three
candidate evaluation cohorts divide the work.

**Is not:** an implementation plan. §6 records a protocol shape under discussion and §6.8 bounds
the scope of one audit script, but nothing here specifies file formats, interfaces or a schedule.
No code has been written on the strength of this document.

**Why it exists:** the deploying institution asked how a test set should be built when its own
patient data cannot leave and arrives ten or twenty records at a time. Answering that needed the
upstream approach checked rather than assumed, and the check changed the answer.

Backlog item 11 (the evaluation-holdout protocol) is what this feeds.

---

## 1. What the original team's code is designed to do

Read from `mims-harvard/SHEPHERD` at commit `e95433a`, and from the external simulator it
depends on, `EmilyAlsentzer/rare-disease-simulation`. Every claim below was re-verified against
the file after the search that found it.

**One scope rule governs the whole section.** These are statements about **committed source and
its intended design**, not about what the team executed. The split module does not parse (§1.4)
and the released artifacts were not inspected, so "the code partitions X" is establishable and
"they partitioned X" is not. Where the narrative below reads as execution, read intent.

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

### 1.2 The *third* partition is folded back into train — validation stays disjoint

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

So the `val_mrr` their code is **designed** to produce measures generalisation to diseases with
no labelled patient examples in training, while ours measures recognition of new phenotype subsets
of diseases that do have them. Different quantities under one name.

*Bounded claim.* Everything said here about the original arrangement is about the **committed
source's intended design**. That module does not run (§1.4) and the released split artifacts were
not inspected, so nothing here establishes what their shipped files actually contain.

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

### 1.5 The repository contains no patient simulator

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

One consequence is combinatorial. Because `k` is fixed per disease, a disease with `P`
phenotypes admits exactly `C(P, k)` distinct phenotype subsets: **1** for `P = 2`, 3 for `P = 3`,
6 for `P = 4`, 15 for `P = 6`. Any disease drawn more often than its own `C(P, k)` must therefore
repeat a subset, and a disease with `P = 2` repeats on the second draw.

**Three precisions, because earlier versions of this paragraph overstated the consequence.**
The repeats are identical in *model-visible content* — the phenotype set and the disease id — and
not in the record, which carries a distinct `patient_id` (`sample_generator.py:212`). And the
workspace's
aggregate of 115,000 samples over 10,576 diseases, about 11 per disease on average, does **not**
establish which diseases were actually drawn more often than their combination count. The bound is
exact; how many diseases cross it here is unmeasured.

Third: exceeding `C(P, k)` guarantees a repeated signature **somewhere in the generated
population**. It does not guarantee that any repeated signature straddles a train/test boundary —
that depends on where the repeats land, which the slicing decides. **Cross-split leakage is
unmeasured** and would need a canonical-signature audit over the generated cohorts to establish.

What follows is therefore conditional: to whatever extent diseases are drawn past that bound,
repeated model-visible content exists in the generated population, and whether any of it crosses
into a test split is an open question rather than a consequence.

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

Three cohorts, three different questions. **They are not interchangeable**, which is a weaker
and more defensible claim than "none substitutes for another": whether all three are *required*
depends on which questions the deployment actually has to answer, and §6 settles that the answer
is currently two of them.

| Cohort | Answers | Standing |
|---|---|---|
| **MyGene2** | Does the model hold up against **real phenotype recording** — incomplete, coarse-grained, noisy — and how large is the gap between synthetic and real? | **Research comparison.** Not an acceptance gate |
| **Disease-disjoint synthetic split** | Does the model generalise to diseases with **no labelled patient examples in training**? (Their KG nodes and edges are still present — §3.1) | Requires a generator change; two tiers, see below |
| **Institutional offline cohort** | Does the chosen model hold up on **this hospital's population**? | **Acceptance benchmark** |

**Confining MyGene2 to research comparison is the right call for two reasons beyond preference.**
Its disease distribution is that of a self-selected family-upload platform and has no relation to
the deploying hospital's case mix, so accepting a model on its evidence would be accepting on
evidence about a different population. And it keeps the dataset in the use it was published for:
using a research cohort as a clinical deployment gate is a different use, and would raise consent
and licensing questions that this division avoids entirely.

### 3.1 The synthetic split is two decisions, and the first is not free either

§1.2 separates them. Before the trade-off can be weighed, one word has to be unpacked.

#### What a disease being "held out" does and does not remove

Patients are not nodes in the knowledge graph. Its node types are `gene/protein`,
`effect/phenotype` and `disease` (`shepherd/preprocess.py:27-32`); a patient is an input to the
model, not part of the graph. Holding a disease out of the *patient* split therefore removes one
thing only:

| | A held-out disease's status |
|---|---|
| Its KG node, and its phenotype and gene edges | **Present**, and used |
| A labelled patient example of it | **Absent** |

Its embedding still comes from the graph — `patient_nca_model.py:70` indexes disease embeddings
out of the GNN's node outputs — and it stays in the candidate pool, which is every disease in the
KG (`dataset.py:194-197`). What it never gets is a supervised "this phenotype set → this disease".

One precision about "used". Patient training propagates over a **random 80%** of KG edges:
`train.py:156` passes `all_data.edge_index[:, train_mask]` as the sampler's graph, while
validation (`:167`) and inference (`:183`, `predict.py:116`) pass the full `edge_index`. That
80/10/10 mask is shuffled over all edges (`prepare_graph.py:57-62`) and is **independent of the
patient disease split**, so a held-out disease's neighbourhood is present during training under
the same random masking mechanism, and therefore at the same *expected* retention rate, as any
other disease's — not necessarily at the same realised rate in a given draw. It is not a second
holdout aimed at the same diseases.

**This is why a disease-disjoint validation set is a measuring instrument rather than a
mutilation.** The paper's claim is few-shot: that a disease with no patient examples can still be
ranked, from its graph structure. Withholding patient labels for 15% of diseases and then scoring
them is that claim's measurement. A model that does well has shown it does not merely recognise
what it was shown; a model that does badly has shown it does.

#### But the cost is real, and the upstream design's own shape says so

A model trained this way genuinely has no patient supervision for those diseases. That is not zero.

The evidence that it is not zero is the shape of the committed code itself: it builds three
partitions and then gives the third one back, with a comment saying why —
`preprocess_patients_and_kg.py:295`, "to be able to train on more diseases". Lost training
coverage is treated there as a cost worth a held-out test set, and **not** as a cost worth the
validation set.

No refit path exists in that repository: `hparams.py:181-187` maps to train and validation files
only, and no script retrains on the union. So the committed design carries that loss into the
model it produces. Whether the artifacts they released were produced that way is not establishable
from here: the module does not run, and the released files were not inspected.

#### The three options, and what each costs

| | Gets | Costs |
|---|---|---|
| **A — status quo (total overlap)** | Every disease keeps patient supervision | `val_mrr` measures recognition of new phenotype subsets of diseases that have labelled examples. That is within-disease generalisation, which is a real property — it is **not unseen-disease** generalisation, and it is the metric currently selecting the batch representative |
| **B — disjoint validation** | A metric that measures what the paper claims, and can separate models on it | ~15% of diseases lose patient supervision in the model that ships |
| **C — disjoint validation for selection, refit on all for deployment** | Both the measurement and full coverage | **The deployed model is not the measured one.** Standard practice in ML; for a clinical system, accepting a model on a different model's numbers may be a worse trade than either A or B |

The committed upstream design corresponds to **B**. **C** appears nowhere in that repository and
is recorded here as a real option rather than a recommendation — its cost lands in a place the
other two do not touch.

Under review, **B** is the option carried forward *for discussion*, and **C is explicitly not
prohibited**: refitting on all diseases for deployment is a legitimate engineering choice, and
what §6.4 requires of it is attribution discipline, not abandonment. §6.4 also records a variant
of **C** that keeps a direct unseen-disease measurement of the deployed weights.

The second decision, a third disjoint test partition, sits on top of whichever of these is chosen
and is the one needing the partition judgement in §5.1.

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

**Comparing two models is a different calculation, and no fixed threshold is stated here.**
Candidate models are scored on the *same* cases, so the comparison is paired: what carries the
signal is the cases where the two disagree, not their marginal accuracies. A paired test is
correspondingly more powerful than the independent-samples calculation an earlier revision of this
section used, and the required cohort size depends on the **discordance rate**, which nobody has
measured here.

The dependence is strong enough that quoting a single number would mislead. At 20% discordance the
power to detect the same 15-point difference reaches roughly 74% at 60 cases; at 30% discordance
the same 60 cases give roughly 56%. (Discordance is also bounded below by the difference itself —
two models whose accuracies differ by 15 points must disagree on at least 15% of cases — so the
rate cannot be arbitrarily small when the difference is large.)

**What does not depend on any of that is the interval.** At fifteen cases a single model's accuracy
is known to within about 38 points, so the cohort cannot report *how good* a model is at that size,
whatever it can say about which of two is better. Its dependable use is a floor: confirming the
already-chosen model has not broken on this population.

The cohort is append-only and grows, and only results against the same cohort version are
comparable — which is the practical content of the institution's own requirement that a test
result record its **dataset version**.

### 4.1 Being real does not by itself make a cohort a valid test

This applies to MyGene2 and to the institutional cohort alike, and it matters more once §6.0 makes
these two the whole of the final evaluation. Seven conditions, none of which follows from the data
being real:

| | Condition |
|---|---|
| 1 | Diagnosis ground truth is reliable |
| 2 | Phenotype inputs are in a compatible HPO representation |
| 3 | The cohort is frozen and versioned, and carries a digest |
| 4 | Metrics and acceptance rules are declared **before** the result is produced |
| 5 | The cohort is separated from training and from model selection |
| 6 | Access is controlled, so that repeated inspection does not silently convert it into a selection set |
| 7 | Disease overlap against the diseases that had synthetic patient supervision is reported, not assumed |

(6) is the burned-cohort hazard of §6.7 in its permanent home. It does not disappear when a
synthetic test partition is dropped — it is inherent to any acceptance gate, and it is an
**operating rule for the institution**, not a piece of software.

**MyGene2 and the institutional cohort are reported separately, never pooled.** Their populations
and their recording processes differ, so a single "real-world accuracy" over the two would be an
average across two different questions.

---

## 5. Open

Recorded rather than resolved. Most of these turn on institutional judgement; where a question
or sub-question belongs to engineering instead, the text says so — and in §5.1 engineering owes
two of the five answers before the institution can usefully give the other three.

1. **How the disease universe is partitioned, and what that costs.** Applies to §3.1's second
   tier — restoring a disjoint *validation* set is parity with the original design and carries the
   original's own 15%. Reserved diseases leave training, which is the trade-off the upstream code
   resolves in favour of training breadth. That repository references three real-patient
   cohorts — UDN, MyGene2 and DDD (§2) — as evaluation beyond the synthetic split; two of the
   three are unavailable here.

   Review split this into five questions, because they have different answerers and only the last
   two are engineering questions at all. **On the default path (§6.0) only (i), (ii) and (iv) are
   live** — they apply to the disease-disjoint *validation* set, which that path does need. (iii)
   is answered by refit-on-all, and (v) does not arise, because there is no permanent synthetic
   test partition to burn:

   | | Question | Whose |
   |---|---|---|
   | i | **How many** diseases are withheld from patient supervision | Institution |
   | ii | **Which strata** they are drawn from — prevalence band, phenotype count, gene count, KG degree — since a uniform draw and a stratified draw hold out different clinical content | Institution, informed by §6.8's audit |
   | iii | Whether a **permanent** loss of supervision for the withheld diseases is acceptable in a deployed model, or whether option **C** is required | Institution |
   | iv | Whether the generator is faithful enough (§1.5) for a disjoint split to mean what it appears to mean — item 6 below asks the sequencing question, this one asks the sufficiency question | Engineering, then institution |
   | v | What happens when a test cohort is **burned** — inspected during selection — and how the protocol regenerates from that point | Engineering, and it must be decided *before* the first cohort exists |

   (v) is the one that expires, *if it becomes live at all*. A cohort that has been looked at
   cannot be restored to uninspected by relabelling it, so the replacement path has to exist
   before anybody has a reason to want it. On the default path the same hazard applies to the
   institutional cohort instead, where §4.1(6) locates it.
2. **Whether to apply for the larger institutional database — answered as a condition, not yet
   as a decision.** The ten-to-twenty figure is the single-batch extraction limit. A substantially
   larger database does exist in-hospital, behind an application and approval process.

   §4 bounds what an accumulating cohort supports without fixing a size: a floor check is
   dependable at any size the interval permits, and the size needed to *rank* depends on a
   discordance rate nobody has measured. **No threshold is stated here, and no decision is derived
   from one.** Under the division of labour in §3 ranking belongs to MyGene2 and the synthetic
   split, so whether to apply turns on a different question — whether ranking *on this hospital's
   own population* is held to be non-delegable. A value judgement for the institution.
3. **Which criterion of record selects a model.** The built-in auto-selection reads the ranking
   metric from a checkpoint's own logs (`src/api/routes/pipeline.py:225`, priority
   `("val_mrr", "val_hits@10", "val_hits@1")` at `src/utils/checkpoint_paths.py:45`), which the
   M1–M3 audit found to be `val_mrr` in all fifteen checkpoints. That matches the institution's
   stated first stage — best validation model as the batch representative — so the two are a
   pipeline rather than competing rules. What is open is that the representative is chosen by a
   metric measured on a split with 100% disease overlap — a metric that, per §1.2, measures
   within-disease recognition rather than unseen-disease generalisation. Whether that costs it
   discriminating power between candidates is **not measured**, here or anywhere in this
   document.
4. **Where a test result is recorded, and how it binds to a checkpoint.** A result cannot be
   written into the `.pt` without changing its SHA-256, and the M1–M5 evidence chain cites
   checkpoints by digest. A sidecar beside the checkpoint keeps the digest stable and lets a
   missing file stand for **"no recorded evaluation is available"** rather than requiring a flag
   somebody maintains — the pattern `scripts/compute_shortest_paths.py` already uses for
   `<artifact>.meta.json`. Any such record
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

## 6. Recommended protocol, under discussion

The shape below came out of review of §1–§5. Most of it is **not decided**; each passage says
where it stands.

### 6.0 The default path, and what it makes unnecessary

An earlier revision of this section drifted, and the drift is worth naming because it is the kind
that recurs. The finding in §1.2 is that **model selection** runs on a saturated metric. That is a
selection-quality defect. What §6 grew in response was a **deployment-claim architecture** —
permanent test partitions, protocol state machinery, evidence-attribution surfaces — sized for a
claim the institution has not asked for: *that the exact deployed weights carry a direct synthetic
unseen-disease result*.

Absent that requirement, the path is short:

| | Step | Cohort |
|---|---|---|
| 1 | Train and select with a **disease-disjoint synthetic validation** set | `synthetic_train`, `synthetic_val_unseen` |
| 2 | Freeze the recipe | — |
| 3 | Refit the deployment checkpoint on **all** eligible diseases | — |
| 4 | Freeze the exact checkpoint digest | — |
| 5 | Evaluate that checkpoint as external research comparison | MyGene2 |
| 6 | Evaluate that checkpoint as the acceptance gate | institutional offline cohort |

**Two consequences.** A permanent `synthetic_test_unseen` partition is **not** required on this
path; disease-disjoint synthetic evaluation remains available as a recipe-level research
instrument. And step 3 removes the only cost §3.1 charged against option **B**: the diseases
withheld during selection get their supervision back in the model that ships, because the shipped
model is refit on everything. Under this path **B is close to free**, and it is the fix for the
defect that started §6.

What it costs: the deployed checkpoint has no direct synthetic unseen-disease number of its own.
§6.4 governs how that absence is stated. If the institution requires such a number for the deployed
weights, §6.3's trade-off returns in full and the conditional machinery below becomes necessary.
**That requirement does not currently exist**, and nothing conditional on it should be built.

### 6.1 Roles, named separately

One ambiguous val/test pair is what let two different quantities share the name `val_mrr` in §1.2.
Six roles, each answering one question. **This is a vocabulary, not a required set of six pipeline
components**. The default path in §6.0 does not use `synthetic_test_unseen` at all:

| Role | Contains | Used for |
|---|---|---|
| `synthetic_train` | — | Patient-supervised training |
| `synthetic_val_seen` | New presentations of **training** diseases | Training health, early stopping. Within-disease generalisation, which is a real thing and not nothing |
| `synthetic_val_unseen` | Diseases with **no** labelled patients in training; KG nodes and edges still present | Architecture, configuration and model selection |
| `synthetic_test_unseen` *(optional)* | Disease-disjoint, **not inspected** during selection | Required **only** for a direct exact-checkpoint synthetic unseen-disease claim (§6.0). Not on the default path |
| `MyGene2` | Real records | External research comparison. Not the acceptance gate |
| `institutional_acceptance` | In-hospital offline cohort | The exact checkpoint bytes proposed for deployment |

### 6.2 Partition before generation

The generator should consume a disease partition rather than own split policy — an interface shaped
like `generate_samples(disease_ids, config, seed, …)` rather than
`generate_training_samples(num_train, num_val, …)`.

A frozen disease-allocation artifact comes first, recording the disease-universe digest, per-
partition digests, allocation seed and schema version, KG/HPO/Orphanet versions, disease counts,
empty-intersection assertions and stratification summaries. **A seed alone is not enough**, because
the disease universe itself changes between KG vintages — the same lesson as
[`BACKLOG.md`](BACKLOG.md) §2.4,
where a figure was divided by a denominator from a different artifact.

**Frozen means immutable, and nothing is ever written back into it.** An allocation records how the
disease universe was cut, once. What each run then *did* with that cut — which partitions it
trained on, which it scored — belongs to the run and checkpoint provenance that already exists, not
to the allocation. When a partition is burned or the universe is recut, the answer is a **new
allocation generation** that links to its predecessor, never an edit to the old one. An allocation
that can be edited is not evidence of anything, because the thing it described has changed while
keeping its identity.

70/15/15 is acceptable as a *pilot* for characterisation. It is not a deployment default: it
withholds supervision for about 30% of diseases, more aggressive than the upstream arrangement of
85 train / 15 disjoint validation.

### 6.3 The trade-off cannot be engineered away

For one set of weights, these cannot both hold:

1. every disease received labelled patient supervision during training;
2. the checkpoint was evaluated on diseases with no labelled patient supervision.

No split scheme or provenance schema removes that. What a protocol can do is decide *which model
identity* each claim attaches to.

### 6.4 Two kinds of evidence, and they must not be merged

- **Recipe evidence** — disease-disjoint validation or cross-validation results. These describe a
  frozen training recipe: this architecture, this generator configuration, this schedule, measured
  on diseases it had no labels for.
- **Exact-checkpoint evidence** — results produced by a specific checkpoint digest on MyGene2 or
  the institutional cohort.

A deployment checkpoint refitted on all diseases is a **new model identity**. It shares a recipe
with the selection checkpoints and does not share weights: the training data differ, so the
gradient sequence differs, so the weights differ. Recipe evidence may be recorded against it *as
recipe evidence, naming the sibling checkpoint's digest*, and must never be displayed as a
measurement of its own weights.

**One structure avoids the dilemma for the unseen-disease claim.** If `synthetic_test_unseen` is
still held out after refitting on train plus `synthetic_val_unseen`, the refit checkpoint can be
measured on it **directly** — recovering the validation diseases' supervision while keeping a real
unseen-disease measurement of the exact deployed weights. Only if the test partition is also folded
back does the deployed model lose direct unseen-disease evidence entirely, and only then does the
recipe-attribution fallback become the whole of the story.

**When the fallback is the whole story, three rules constrain how it is presented.** They were
settled in review and are recorded as settled, not as proposals.

- **A shared seed is not a bridge.** An earlier draft reasoned that a sibling trained from the
  same seed is close enough for its numbers to describe the deployed weights. That is rejected:
  different training data produce a different gradient sequence and therefore different weights,
  and seed equality changes nothing about it. There is no partial credit here — the sibling's
  numbers are the sibling's.
- **Sibling performance is never written into the checkpoint's provenance fingerprint.** The
  fingerprint identifies *these* weights and their inputs. Admitting another checkpoint's metric
  into it makes the fingerprint claim something it did not measure, which is the exact failure the
  `data_fingerprint` / `training_input_digests` separation exists to prevent.
- **The absence is stated first, and the recipe evidence second.** Where a deployed checkpoint has
  no direct unseen-disease measurement, the surface says so in those terms —

  > No direct synthetic unseen-disease evaluation is recorded for these checkpoint weights.

  — and then presents recipe evidence as a separately labelled block naming the checkpoint digests
  it came from. That block may cite **several** fold checkpoints; a recipe measured across folds is
  stronger evidence about the recipe than any single fold, and nothing about citing more than one
  weakens the separation, because none of them is being claimed as these weights.

### 6.5 Evaluation records live beside the checkpoint, not inside it

A result written into the `.pt` changes its SHA-256, and the M1–M5 chain cites checkpoints by
digest. A sidecar or ledger keyed by checkpoint digest, cohort role, cohort version and digest,
metric schema version, allocation provenance and runtime revision keeps the digest stable and lets
a missing record stand for **"no recorded evaluation is available"** rather than requiring a flag
somebody maintains.

The wording matters and matches §6.4. A missing record does not establish that a checkpoint was
never evaluated — this provenance system is not closed, and an evaluation can happen outside it.
What the absence supports is the narrower and true statement that *this system holds no result for
these weights*.

### 6.6 Order, and what is explicitly last

**On the default path (§6.0):** correct this document; run the split feasibility audit (§6.8);
partition the disease universe before generation, so that validation is disease-disjoint;
characterise the current generator against the upstream simulator on one frozen allocation; record
the deployment checkpoint's evaluations beside it (§6.5).

**Conditional on the institution requiring a direct exact-checkpoint synthetic unseen claim, and
not before:** a permanent `synthetic_test_unseen` partition, allocation generations for burned
cohorts, and any UI that manipulates cohort roles.

**Last in either case:** UI (§6.7).

The upstream simulator should first be used as a **pinned external tool** — fixed commit,
recorded configuration, source digests, adapter into the current schema, and counts of identifiers
that failed to map — rather than reimplemented into `src`. Its code is MIT; the licences and
permitted uses of HPO, Orphanet and any patient-derived dataset are separate questions.

### 6.7 What a preprocessing or training control may and may not do

The institution asked for a preprocessing page and a stop-training checkbox with automatic
post-stop evaluation. Both are reasonable; neither should be built before the roles (§6.1) and the
sidecar contract (§6.5) are frozen, and both are bounded by two rules that came out of review.

Scope note, added in revision 6: this section describes **constraints on a control if one is
built**. It is not a request to build protocol state machinery, and on the default path (§6.0) the
sharpest case below — a burned synthetic test cohort — does not arise, because there is no
permanent synthetic test partition to burn.

**Reuse the provenance that exists.** A cohort's identity, and a model's relationship to it, are
already expressible through `training_input_digests` and the checkpoint provenance record. A page
that introduces its own parallel notion of "which dataset this run used" creates a second source of
truth that will drift from the first. Whatever a preprocessing page records, it records *into*
those structures.

**A control that changes a cohort's role is a protocol change, not a display setting.**
Moving a partition from held-out to trained-on, or evaluating on a partition that selection has
already touched, changes what every subsequent number means. Per §6.2 such a change does **not**
mutate the existing allocation: it produces a **new allocation generation** linked to the old one,
and the run provenance records which generation that run consumed. Otherwise "which cut was this
model trained under?" stops being answerable after the fact.

The sharp case is a **burned** cohort — one inspected during model selection. It cannot be
returned to untouched by a checkbox, a rename, or a new run label. Once inspected it is a selection
cohort permanently, and a fresh measurement needs a fresh cohort. A UI that permits the relabelling
silently is worse than no UI, because it produces evidence that looks clean and is not.

**This hazard is not specific to synthetic splits, and dropping the synthetic test partition does
not remove it.** It belongs to whatever cohort is the acceptance gate — on the default path, the
institutional one, whose operating conditions §4.1 now lists. Locating it in one place rather than
two is a simplification, not a relocation of the same machinery.

### 6.8 The approved next step: an aggregate-only split feasibility audit

This is the one concrete engineering step cleared to proceed, and it is deliberately narrow: it
tells the institution what partitioning the disease universe would actually cost, without choosing
a partition.

**It reports — all claim-independent, true whatever the institution decides:**

- the disease universe size and how many diseases are eligible for patient generation at all;
- the distribution of phenotype counts and of gene counts per disease;
- generator capacity bands derived from `C(P, k)` — how many diseases can support how many
  distinct samples, which is §1.5's bound turned into a histogram;
- counts of zero-degree and low-degree diseases in the KG;
- identifier-mapping success rates and the reasons for exclusions;
- the digests of its inputs and of the data version it ran against.

**Plus one derived figure, under re-review.** Review proposed deferring all split-ratio arithmetic
until the institution confirms it wants a synthetic unseen test. That is deferred too far, for a
reason independent of the test question: **the default path in §6.0 still needs a disease-disjoint
validation set**, and choosing one means choosing a withheld fraction and deciding whether the draw
is uniform or stratified. The cost of that choice is the same arithmetic.

What is dropped is the framing, not the computation. Rather than "balance under 85/10/5, 80/10/10,
70/15/15" — three named ratios that read as a menu of protocols — the audit reports a
**withheld-fraction sensitivity curve**: for `f` across a range, how many diseases leave patient
supervision, and how each stratum's share shifts, under a uniform draw and under a stratified one.
One axis instead of three candidate protocols. It is pure arithmetic over the disease universe,
commits to nothing, and is exactly what §5.1(i) and §5.1(ii) need in order to be answerable at all.

**It does not emit** patient identifiers, disease identifiers, per-disease listings, host or
operator names, or absolute paths — [`BACKLOG.md`](BACKLOG.md) §5.2 governs what an evidence
artifact may contain, and applies here unchanged.

**It does not decide.** It selects no fraction, modifies no generator behaviour, writes no
allocation, and builds no UI. Its output is an input to §5.1, not an answer to it.

---

## 7. Method

Repository claims were located by search and then re-read in the file before being written here;
file and line references are to `mims-harvard/SHEPHERD` at `e95433a` and to
`EmilyAlsentzer/rare-disease-simulation` at its default branch. The syntax error in §1.4 was
confirmed by running `ast.parse`.

§4's intervals are Wilson intervals. Its paired figures are a normal approximation to McNemar's
test, computed for this document at illustrative discordance rates of 20% and 30% — the rate itself
is unmeasured, and the illustration exists to show that the answer depends on it rather than to
supply a number. An earlier revision used an independent-samples two-proportion approximation,
which is the wrong model for candidates scored on the same cases; those figures and the threshold
derived from them have been removed. All of it describes sample sizes, not any measurement of this
system.

`mims-harvard/OptimusKG` was examined and is **not relevant** to either question: it is a
knowledge-graph construction pipeline with no model, no training loop, no split logic, and no
patient node type. Its `evals` module evaluates KG edges against the literature, which is a
different problem. It is noted only as a possible future KG substrate.
