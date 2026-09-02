# Evaluation cohorts — findings, and the division of labour

**Type:** findings report plus institutional decisions · **Date:** 2026-08

**Status:** §1–§2 established, including the published article (§1.6); §3 decided; §5's two gating
questions **answered**, its partition questions still open; §6.0 fixed; §6.8 cleared to proceed
with one item under re-review.

<details>
<summary><b>Revision history</b></summary>

- **14** — corrections from implementation review of the audit. §6.8 no longer promises
  **identifier-mapping success rates**, which a materialised `kg.json` cannot evidence — mapping
  happens during KG construction and leaves no trace — and no longer claims to report **KG degree**:
  the generator's profile builder propagates a gene's phenotypes onto its diseases, so the axis is a
  **propagated support size**, which is now what it is called. The audit also validates every numeric
  input before loading the graph or writing anything.
- **13** — the audit is implemented (`scripts/audit_split_feasibility.py`), and five bounded
  specification corrections land with it: the quota-versus-expectation gap has **one** source, not
  two, since both use the same `W` (the `f · N → W` rounding explains a different comparison);
  `X_s = 0` and `X_s = n_s` are named as **two different failures** — no validation representation
  and no training representation — with distinct JSON keys; the `samples_per_disease` rename is
  complete; and the missing bucket's position in the canonical order is fixed as last.
- **12** — one blocking probability error corrected. §6.8 gave
  P(stratum retains zero) as `C(n_s, W)/C(N, W)`, which is the probability that the *whole withheld
  set came from that stratum* — a different event. The correct form is
  `C(N − n_s, W − n_s)/C(N, W)`, zero when `W < n_s`. The two coincide only at `W = n_s`, which is
  precisely the case the previous revision's brute-force check used, so the test confirmed a
  coincidence rather than the formula; the audit's tests must now enumerate `W < n_s`, `W = n_s` and
  `W > n_s`. Also: the output contract is the **standard deviation**, not the variance; largest-
  remainder ties break on a canonical **bucket** key rather than a disease ordering; `k` is reserved
  for `C(P, k)` and the budget dimension is `samples_per_disease`; the audit's generator parameters
  are recorded as **assumptions** since no workspace records them yet; and the comparability claim
  is narrowed to sample-derived validation metrics.
- **11** — one real mathematical error corrected and one contract strengthened. §6.8 now defines an
  **integer withheld count `W`** and states every formula against it; the earlier hypergeometric
  mean `f · n_s` was wrong wherever `f · N` is not integral, and the quota/expectation gap has two
  sources, not one. §6.2's coverage contract becomes **constructive**: a sufficient budget does not
  guarantee coverage under replacement sampling, so one sample per allocated disease is emitted
  first, the realised set is derived from the emitted records rather than copied from the
  allocation, and the generation algorithm's version is recorded because that pass changes the
  sampling distribution. §6.0 and §6.2 no longer appear to contradict each other on allocation
  artifacts, and predecessor lineage is explicitly not required absent a consumer.
- **10** — corrections from implementation review. §6.2 records that **allocation is a separate
  step** and that **allocated is not realised**: generation draws with replacement, so an allocated
  disease is not guaranteed a sample, and "85% supervised" holds only where the budget reaches every
  allocated disease. §6.8's arithmetic is corrected — the largest-remainder quota is an integer and
  the hypergeometric expectation is fractional, so they are reported separately and never called
  identical; marginal quotas over four stratifications are **independent diagnostic targets**, not a
  jointly realisable allocation; `C(P, k)` uses the generator's own configured `k`; and eligibility
  comes from one shared exported helper rather than two implementations.
- **9** — factual corrections from independent verification of §1.6 against the article. **The
  claim that the paper reports no training/evaluation overlap was wrong**: it reports 109 of 319
  UDN diseases and 220 of 378 UDN causal genes as represented in the simulated cohort. What it does
  not report is how that divides between the training and validation partitions, which makes
  §4.1(7) a more specific requirement rather than a new one. The simulated cohort's disease count is
  inconsistent in the article (2,134 vs 2132) and the 42,624 discrepancy is recorded as unexplained.
  §4.2 revised: the UDN *source population* is difficulty-selected, so the ask is unselected cases
  from the intended deployment workflow, not from the hospital at large; Figure 3e's subset sizes
  (40, 78, 7, 7) are recorded, and the expert-labelled strata are the small ones. §6.0 no longer
  calls the paper a deployment protocol — the single-checkpoint rule is this project's own. §6.8's
  sensitivity curve is fully specified, and its earlier promise of "share shifts" is withdrawn as
  wrong under proportional allocation.
- **8** — the two gating questions are answered (§5): no direct synthetic unseen-disease result is
  required for the deployed weights, and option **B** is accepted, both on the institution's policy
  of staying aligned with the published method. §6.0 collapses to five steps with no refit and one
  model identity; §6.4 loses its subject and shrinks to a road-not-taken note; §6.1's default path
  uses four roles; §6.6 lists what is not built. §4.2 added: the reference cohort's inclusion
  criteria are completeness and diagnostic certainty, not difficulty, and its hard-case analysis is
  a post-hoc stratification of which two of four subsets are computed rather than curated — with
  what to ask the institution for.
- **7** — the published article is now a source. §1.6 records what it reports: two simulated
  partitions and no synthetic test set, disease-disjoint by design and stated as such, with the
  real cohorts as the evaluation and unseen-disease evidence stratified inside them. Its patient
  counts corroborate §1.2's reading of the committed code. Consequences: §6.0's shape is the
  published design rather than an engineering preference, its refit steps are a deviation the
  reference does not take, §5.1(iii) reopens, §3 notes that the paper trains on MyGene2 for one
  task, and §4.1(7) is marked as this project's own requirement.
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

*Bounded claim, now partly discharged.* Everything said here was read from **committed source**,
which does not run (§1.4) and whose released split artifacts were not inspected. The paper (§1.6)
corroborates both halves independently: it states the disease-level disjointness of train and
validation in words, and its patient counts — 36,224 train, 6,400 validation, 42,624 total —
are **85.0% / 15.0%**, exactly the proportions the merge above produces from a 70/15/15 slice.
The proportions alone do not prove the three-way slice happened, since a direct 85/15 split gives
the same numbers; together with the code's own comment about post-hoc merging, the reading is
coherent and the design and the release agree.

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

*A note on the simulated cohort's disease count.* The paper gives **two inconsistent figures** —
"2,134 unique rare diseases" in one passage and "2132 unique Mendelian disorders" in another — and
neither multiplies by the stated 20 patients per disease to the stated 42,624 total. The
discrepancy is recorded as **unexplained**; no reconciliation is offered here, and neither figure
should be cited as "the number of simulated *training* diseases", since both describe the complete
cohort before the split.

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

### 1.6 What the paper reports, and what it actually used as a test set

Source: Alsentzer et al., *Few shot learning for phenotype-driven diagnosis of patients with rare
genetic diseases*, **npj Digital Medicine** 8:380 (2025), DOI
[`10.1038/s41746-025-01749-1`](https://doi.org/10.1038/s41746-025-01749-1). This section is the
answer to an obvious question: if the third partition was folded back, how did a peer-reviewed
paper evaluate anything?

**It never claimed a synthetic test set.** The Methods describe two simulated partitions and no
more — "we split the list of diseases represented by the simulated patient cohort into training
and validation … patients with the same disease are either entirely in the training or fully in
the validation set", giving **36,224 train and 6,400 validation** patients out of 42,624 total.
The purpose is stated outright: the disease-stratified split exists "to enable SHEPHERD to
generalize to diseases unseen during training."

**The synthetic validation set selects hyperparameters.** "We select task-specific hyperparameters
to optimize the mean reciprocal rank of the correct genes, diseases, or patients on the
disease-split simulated validation set."

**The test set is three real cohorts.** UDN (N = 465), MyGene2 (N = 146) and DDD (N = 1,431), with
the separation stated explicitly: "the validation set containing simulated patients is entirely
independent of the evaluation dataset, which includes patients from the Undiagnosed Diseases
Network."

**The unseen-disease evidence is stratified inside the real cohort, not drawn from a synthetic
holdout.** They identify UDN patients whose causal genes have no known phenotype or disease
associations in the knowledge graph, and patients UDN experts flagged as having novel diseases or
novel disease genes, then report win rates on those subsets — up to 82–83% for the no-known-
association groups and 67–86% for the expert-flagged novel groups.

#### Three consequences for this project

| | |
|---|---|
| **The structure of §6.0 is the published reference design, not an engineering preference** | Disease-disjoint synthetic validation for selection, real cohorts for evaluation, no synthetic test partition. A peer-reviewed instance of exactly that path exists |
| **The statistical power does not transfer with the structure** | They evaluated on 2,042 real patients across three cohorts. This project has MyGene2 at 146 and an institutional cohort accumulating at ten to twenty per batch. §4's bounds are unchanged by this section |
| **They did not refit on all diseases** | No refit path exists in the repository (`hparams.py:181-187`), so the model they describe is option **B** of §3.1 and ships carrying the withheld diseases' loss. §5 records the institution's decision to do the same |

#### One caveat that cuts against using MyGene2 naively

For causal gene discovery they **also train** on MyGene2 and DDD patients — "these additional
cohorts constitute 3.6% of the training data" — while for patients-like-me retrieval "the
simulated cohort is used for training, and the UDN and MyGene2 cohorts are used for validation."
So MyGene2 is not a clean external cohort for every task in the paper. Using it purely as an
untouched research comparison (§3) is **stricter** than what the paper did, which is the right
direction, but it also means our MyGene2 numbers are not directly comparable to theirs on the gene
task.

#### What the paper reports about overlap, and what it does not

It **does** report cohort-level overlap, in the same Methods paragraph as the cohort size:

> Of the 378 unique causal genes and 319 unique MONDO diseases found in patients in the UDN
> cohort, 220 and 109 are represented in the simulated patient cohort, respectively. Furthermore,
> 81.8% of the phenotype terms found across UDN patients are also found in the simulated patient
> cohort […] but also emphasizes the need for developing models that can generalize to genes,
> diseases, and phenotype terms unseen at training time.

**What it does not report is how those overlaps divide between the training partition and the
disease-disjoint validation partition.** A disease counted as "represented in the simulated cohort"
may have been in either. §4.1(7) — overlap against the diseases that actually received patient
supervision — is therefore a **more specific** requirement than the paper's, not a substitute for
a missing one.

Separately, the paper's leakage argument is temporal and concerns the knowledge graph: recently
diagnosed UDN patients show no performance drop. That is a different question from partition
overlap.

### 1.7 What this project's guards are, and are not

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
| **Disease-disjoint synthetic split** | Does the model generalise to diseases with **no labelled patient examples in training**? (Their KG nodes and edges are still present — §3.1) | **Validation and model selection.** Requires a generator change (§6.2); no separate test tier is built (§5) |
| **Institutional offline cohort** | Does the chosen model hold up on **this hospital's population**? | **Acceptance benchmark** |

**Confining MyGene2 to research comparison is the right call for two reasons beyond preference.**
Its disease distribution is that of a self-selected family-upload platform and has no relation to
the deploying hospital's case mix, so accepting a model on its evidence would be accepting on
evidence about a different population. And it keeps the dataset in the use it was published for:
using a research cohort as a clinical deployment gate is a different use, and would raise consent
and licensing questions that this division avoids entirely.

Note from §1.6 that treating MyGene2 as untouched is **stricter than the paper**, which also
trains on MyGene2 and DDD for the causal gene discovery task. The direction is right, and it means
our MyGene2 numbers are not directly comparable to the published ones on that task.

### 3.1 What a disjoint split costs, and which option was taken

§1.2 raises two separable decisions — restore a disjoint validation set, and add a third disjoint
test partition. §5 has since closed the second (not built) and the first (option **B**). What
follows is the reasoning that produced those answers, kept because the cost it describes is real
and will be asked about again. Before the trade-off can be weighed, one word has to be unpacked.

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

**The institution has chosen B** (§5), on its policy of staying aligned with the published
method, which is itself B. **C** is not prohibited in principle — it is a legitimate engineering
choice with the attribution discipline §6.4 describes — but it is not the path being built, and no
surface for it should exist.

The second decision — a third disjoint test partition — is **not taken** (§5). What remains open
is not whether to cut the disease universe but how: the fraction and the stratification, which
§5.1 records and §6.8 measures.

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

(7) is a **more specific** requirement than the paper's, not a new one. §1.6 records that the
paper reports overlap between the *whole* simulated cohort and UDN (109 of 319 diseases, 220 of
378 causal genes) but not how that divides between the training and validation partitions. Only
the training-partition figure bounds what patient supervision could have contributed.

### 4.2 How the reference cohort was assembled — and what to ask the institution for

A natural reading of §1.6 is that experts hand-picked a list of never-before-seen disease
mechanisms to serve as the test set. **That is not what happened, and the difference changes the
ask.**

**The final evaluation cohort was not post-hoc selected for the four hard-case strata.** A
patient is included if they have (1) at least one phenotype term, (2) at least five candidate
genes, and (3) a diagnosis classified as *certain* or *highly likely* under the UDN's own
diagnostic-certainty annotations — completeness and ground-truth reliability, nothing about
novelty. Everyone meeting those bars is in, N = 465.

**But the source population is itself difficulty-selected, and this matters for the ask.**
Admission to the UDN requires objective findings *and* that prior clinical testing has failed to
produce a diagnosis. So the reference cohort is an unselected sample **of a hard-to-diagnose
referral population** — not an unselected sample of general clinical genetics. The correct analogue
is therefore consecutive cases from **the workflow the tool will actually be used in**, not
consecutive cases from the hospital at large.

**The hard-case analysis is a post-hoc stratification of that cohort into four subsets**, and only
two of the four need a clinician:

| | Subset | Who determines it |
|---|---|---|
| 1 | Causal gene has **no known phenotype association** in the knowledge graph | Computed from the KG |
| 2 | Causal gene has **no known disease association** in the knowledge graph | Computed from the KG |
| 3 | Patient has a **novel disease**, per UDN experts | Clinician |
| 4 | Patient has a **novel disease gene**, per UDN experts | Clinician |

Figure 3e gives their sizes in the UDN cohort: **N = 40** and **N = 78** for the two KG-derived
strata, **N = 7** and **N = 7** for the two expert-labelled ones. **The expert-labelled strata are
the small ones.** If 465 referral-population patients yielded seven of each, a cohort accumulating
at ten to twenty per batch will produce approximately none for a long time. The practical
conclusion is to spend little clinical labour there and to compute the two KG-derived strata, which
are five to eleven times larger, ourselves.

**Why this matters for the acceptance gate.** Selecting cases *for* difficulty and calling the
result the test set would break §4.1(5): the number would describe the selection, not the
hospital's population, and it could not answer "does this hold up on our patients". Selection on
difficulty is a legitimate way to build a *stress* cohort, but it is a different instrument from an
acceptance cohort and cannot stand in for one.

#### What to ask for

| Ask | Why |
|---|---|
| **Consecutive or otherwise unselected** cases **from the intended deployment workflow**, within whatever extraction limit applies | §4.1(5). Choosing cases on a criterion correlated with difficulty makes the number describe the selection. Drawing from a *different* population than the tool will serve makes it describe the wrong population — the reference cohort is unselected within a hard-to-diagnose referral stream, not within general genetics |
| A **diagnostic-certainty label** per case, and a stated bar | §4.1(1). The reference bar is "certain or highly likely"; a cohort with uncertain ground truth cannot be an acceptance gate |
| Phenotypes as **structured HPO codes**, with the HPO version recorded | §4.1(2) |
| One optional clinician field: **novel disease / novel disease gene / neither** | The only part of the stratification that needs expert judgement, and it is one field per case, not a curation exercise |

#### What this project computes itself, with no clinical labour

Subsets 1 and 2 above, from the knowledge graph. The disease-overlap report of §4.1(7), against the
diseases that had synthetic patient supervision. And the paper's distance stratum — it reports that
77.8% of patients whose phenotype terms sit **more than two hops** from their causal gene are still
resolved in the top five — which the existing shortest-path artifacts and
`scripts/compute_shortest_paths.py` already support without new machinery.

The practical consequence is that the ask on the institution is **small**: an unselected batch, a
certainty label, structured phenotypes, and one optional field. Not a curated novel-disease list.

**MyGene2 and the institutional cohort are reported separately, never pooled.** Their populations
and their recording processes differ, so a single "real-world accuracy" over the two would be an
average across two different questions.

---

## 5. Open

Recorded rather than resolved, except where marked **answered**. Most of these turn on
institutional judgement; where a question or sub-question belongs to engineering instead, the text
says so.

**Two questions have been answered**, on the institution's standing policy of staying aligned with
the published method (§1.6), and are to be confirmed at the next review meeting:

| Question | Answer |
|---|---|
| Must the deployed weights carry a **direct synthetic unseen-disease result**? | **No.** The paper's deployed model has none either. `synthetic_test_unseen` is not built |
| Is a permanent loss of patient supervision for the withheld diseases acceptable in the shipped model — option **B** — or is refitting on all diseases required? | **B is accepted**, matching the reference implementation, which has no refit path |

Together these fix §6.0's path and remove §6.4's subject. What remains open below is how the
disease universe is cut, not whether.

1. **How the disease universe is partitioned, and what that costs.** Applies to §3.1's second
   tier — restoring a disjoint *validation* set is parity with the original design and carries the
   original's own 15%. Reserved diseases leave training, which is the trade-off the upstream code
   resolves in favour of training breadth. That repository references three real-patient
   cohorts — UDN, MyGene2 and DDD (§2) — as evaluation beyond the synthetic split; two of the
   three are unavailable here.

   Review split this into five questions, because they have different answerers and only the last
   two are engineering questions at all. **On the default path (§6.0) only (i), (ii) and (iv)
   remain live** — they apply to the disease-disjoint *validation* set, which that path does need.
   (iii) is **answered: option B** (see the table above). (v) does not arise, because there is no
   permanent synthetic test partition to burn:

   | | Question | Whose |
   |---|---|---|
   | i | **How many** diseases are withheld from patient supervision | Institution |
   | ii | **Which strata** they are drawn from — prevalence band, phenotype count, gene count, KG degree — since a uniform draw and a stratified draw hold out different clinical content | Institution, informed by §6.8's audit |
   | iii | ~~Whether a **permanent** loss of supervision for the withheld diseases is acceptable, or whether refitting on all diseases is required~~ — **answered: option B**, matching §1.6 | Institution — done |
   | iv | Whether the generator is faithful enough (§1.5) for a disjoint split to mean what it appears to mean — item 6 below asks the sequencing question, this one asks the sufficiency question | Engineering, then institution |
   | v | What happens when a test cohort is **burned** — inspected during selection — and how the protocol regenerates from that point | Engineering, and it must be decided *before* the first cohort exists |

   (v) does not arise on the chosen path. The same hazard applies to the institutional cohort
   instead, where §4.1(6) locates it and §4.2 says what to ask for.
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

**Both gating questions are now answered** (§5), and the path is short:

| | Step | Cohort |
|---|---|---|
| 1 | Train with a **disease-disjoint synthetic validation** set | `synthetic_train`, `synthetic_val_unseen` |
| 2 | Select the checkpoint on that validation set | `synthetic_val_unseen` |
| 3 | Freeze the selected checkpoint's digest — **this is the model that ships** | — |
| 4 | Evaluate that checkpoint as external research comparison | MyGene2 |
| 5 | Evaluate that checkpoint as the acceptance gate | institutional offline cohort |

**This adopts the paper's central evaluation structure, and adds one rule of its own.** The
structure — disease-disjoint synthetic validation for selection, real cohorts for final evaluation,
no synthetic test partition and no refit — is §1.6's, and the institution's stated policy is to stay
aligned with it.

**The single-checkpoint deployment rule is this project's decision, not the paper's.** The article
reports task-specific research models, several cohort roles, and results averaged over multiple
random seeds; it does not describe a clinical deployment protocol and there is no "published
deployed model" to point at. Steps 3–5 below are an operational rule this project is choosing, and
they must be defended on their own terms rather than by citation.

**There is no refit, so there is one model identity.** The checkpoint selected in step 2 is the
checkpoint evaluated in steps 4–5 and the checkpoint that ships. Every number describes the weights
that are deployed. This is option **B** of §3.1, and its cost is real and accepted: roughly 15% of
diseases ship with no patient supervision, retaining only their knowledge-graph structure — which
is precisely the condition the paper's few-shot thesis is about.

**What this removes.** No permanent `synthetic_test_unseen` partition. No refit sibling, and so no
sibling-attribution rules (§6.4 is correspondingly short). No protocol state machinery, no burned
synthetic cohort, and **no rotation machinery for replacing one**. The remaining engineering is the
allocation step and one interface change in the generator (§6.2), plus the evaluation record (§6.5).

*Not to be confused with §6.2's immutability rule.* "No rotation machinery" means nothing is built
to retire and replace a burned test cohort, because there is no such cohort. It does not mean an
allocation may be edited: a genuinely new cut of the disease universe still writes a **new
artifact** rather than mutating the old one, which is a file-writing convention, not machinery.

### 6.1 Roles, named separately

One ambiguous val/test pair is what let two different quantities share the name `val_mrr` in §1.2.
Six roles, each answering one question. **This is a vocabulary, not a required set of six pipeline
components**. The default path in §6.0 uses **four** of them — the two marked optional are not on
it:

| Role | Contains | Used for |
|---|---|---|
| `synthetic_train` | — | Patient-supervised training |
| `synthetic_val_seen` *(optional)* | New presentations of **training** diseases | A within-disease training-health signal. The paper has no such partition and the default path does not use one — early stopping runs on `synthetic_val_unseen`, as it does upstream |
| `synthetic_val_unseen` | Diseases with **no** labelled patients in training; KG nodes and edges still present | Architecture, configuration, early stopping and checkpoint selection. **The deployed checkpoint is the one this selects** |
| `synthetic_test_unseen` *(optional)* | Disease-disjoint, **not inspected** during selection | Required **only** for a direct exact-checkpoint synthetic unseen-disease claim (§6.0). Not on the default path |
| `MyGene2` | Real records | External research comparison. Not the acceptance gate |
| `institutional_acceptance` | In-hospital offline cohort | The exact checkpoint bytes proposed for deployment |

### 6.2 Partition before generation

The generator should consume a disease partition rather than own split policy — an interface shaped
like `generate_samples(disease_ids, config, seed, …)` rather than
`generate_training_samples(num_train, num_val, …)`. Allocation is therefore a **separate step**
producing explicit train and validation disease collections, which the generator is handed. This is
the minimum seam that lets a later stratified allocation replace a uniform one without redesigning
generation; it is not a split framework.

**Allocation is not the same as realised supervision, and a budget alone does not close the gap.**
Sample generation draws diseases **with replacement** from whichever collection it is given
(`sample_generator.py:199`), so allocating a disease to the training partition does not guarantee it
receives any training sample — and **raising the budget does not guarantee it either**, because
replacement sampling can miss a disease at any budget. Two figures are therefore distinct
throughout: **allocated** and **realised**.

Coverage is made true by construction rather than by hoping:

1. **Refuse** when a partition's sample budget is smaller than its allocated disease count, naming
   both numbers.
2. **Emit one sample for every allocated disease first**, in a deterministic order.
3. **Distribute only the remainder** by the sampling rule.
4. **Derive the realised disease set from the emitted sample records**, never by copying the
   allocation.
5. **Assert** that the realised set equals the allocated set.
6. **Record both counts and both set digests independently** in the manifest.

Step 4 is what makes the manifest evidence rather than restatement: a realised field populated from
allocation metadata proves nothing, and a test for the coverage pass has to fail when that pass is
removed. Step 2 changes the sample distribution relative to pure replacement sampling — every
allocated disease now has at least one sample where before it might have had none — so the
**generation algorithm and its version are recorded** alongside the counts. What this protects is
narrow and worth stating precisely: two workspaces built under different generation rules do not
yield **sample-derived validation metrics that measure the same cohort regime**, so `val_mrr` from
one is not a like-for-like reading against `val_mrr` from the other. Checkpoints trained on them
remain perfectly comparable on a *shared external cohort*, or inside a controlled experiment where
the generator itself is the variable. The manifest is where the difference is visible.

A frozen disease-allocation artifact comes first, recording the disease-universe digest, per-
partition digests, allocation seed and schema version, KG/HPO/Orphanet versions, disease counts,
empty-intersection assertions and stratification summaries. **A seed alone is not enough**, because
the disease universe itself changes between KG vintages — the same lesson as
[`BACKLOG.md`](BACKLOG.md) §2.4,
where a figure was divided by a denominator from a different artifact.

**Frozen means immutable, and nothing is ever written back into it.** An allocation records how the
disease universe was cut, once. What each run then *did* with that cut — which partitions it
trained on, which it scored — belongs to the run and checkpoint provenance that already exists, not
to the allocation. When the universe is recut, the answer is a **new artifact**, never an edit to
the old one. An allocation that can be edited is not evidence of anything, because the thing it
described has changed while keeping its identity.

**Predecessor lineage is deliberately not required.** Recording a link to a previous allocation is
permitted where something actually reads it, and is not built on the chance that something might:
the digests already distinguish one cut from another, which is what provenance needs. This is a
naming and immutability convention, not a versioning system (§6.0).

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

**On the default path these coincide, and that is the point of taking it.** With no refit, the
selected checkpoint is the deployed checkpoint, so its `synthetic_val_unseen` result *is*
exact-checkpoint evidence and not an inheritance from a sibling. Nothing needs attributing across
model identities.

The distinction is kept because it stops being free the moment a refit is introduced. A checkpoint
refitted on all diseases would be a **new model identity**: same recipe, different training data,
therefore a different gradient sequence and different weights, which a shared seed does not change.
Were that ever adopted, its selection-time numbers could be recorded only as recipe evidence naming
the sibling's digest, never displayed as a measurement of its own weights, and never written into
its provenance fingerprint — for the reason the `data_fingerprint` / `training_input_digests`
separation exists. **The institution has decided against the refit** (§5), so this paragraph
describes a road not taken, and no surface should be built for it.

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

1. Correct this document.
2. Run the split feasibility audit (§6.8) — it is what makes §5.1(i) and (ii) answerable.
3. Partition the disease universe before generation, so validation is disease-disjoint (§6.2).
4. Characterise the current generator against the upstream simulator on one frozen allocation
   (§1.5 is the reason).
5. Record the deployed checkpoint's evaluations beside it (§6.5).
6. UI last (§6.7).

**Not built:** `synthetic_test_unseen`, refit siblings, allocation generations for burned synthetic
cohorts, and any surface that attributes one checkpoint's numbers to another. §5 closed the
questions those depended on.

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
- the **profile support size** distribution — a disease's phenotype count plus its gene count.
  Note carefully what this is *not*: the generator's profile builder propagates a gene's phenotypes
  onto the diseases that gene is associated with, so this is a **propagated support size, not the
  disease node's direct KG degree**. Direct degree is a different quantity and is not measured;
  what the audit can say about connectivity at the low end is that a disease with no incident edges
  at all has an empty profile and is therefore counted as excluded;
- the **exclusion count**, by the one reason a materialised `kg.json` can actually evidence:
  falling below `min_phenotypes`. An earlier draft of this list promised *identifier-mapping
  success rates*, which cannot be honoured from this input — mapping happens during KG construction
  and leaves no trace in the artifact. The promise is withdrawn rather than approximated;
- the digests of its inputs and of the data version it ran against;
- **the budget each coverage contract would require**: `|allocated| × samples_per_disease`, for a
  few values of `samples_per_disease`, against each partition's current budget. One column, from
  figures the audit already computes. (`k` is reserved throughout for the retained-phenotype count
  in `C(P, k)` and is never the budget dimension.)

That last line exists to settle a design question by measurement instead of argument. §6.2's
coverage contract is "one sample per allocated disease, then distribute the remainder", which needs
six mechanisms — a budget refusal, a coverage pass, a remainder rule, a realised-set derivation, an
assertion, and two manifest fields. The upstream simulator instead generates a **fixed number of
patients per disease** (§1.5: `PATIENTS_PER_DISEASE`, 20 in the manuscript), under which coverage
and balance are definitional and the first three mechanisms disappear.

**This is recorded as a question, not a proposal.** Whether the second shape is affordable depends
on `|allocated| × samples_per_disease` against the budgets actually in use, which nobody has
computed. If it is
affordable the contract simplifies; if it is not, §6.2 stands as written. The audit reports the
number either way, and no interface changes on the strength of this paragraph.

**Plus a withheld-fraction sensitivity curve, approved and now fully specified.** Option B still
requires choosing a disease-disjoint validation fraction, so this arithmetic is needed whatever else
is decided. Rather than "balance under 85/10/5, 80/10/10, 70/15/15" — three named ratios that read
as a menu of protocols — one axis: `f`, the fraction of eligible diseases withheld from patient
supervision.

**One correction to an earlier draft of this section, which was wrong.** It promised to show "how
each stratum's share shifts" between a uniform and a stratified draw. Under *proportional*
stratified allocation the expected shares are the same as under a uniform draw; what stratification
buys is lower **variance**, not a different mean. Promising a shift would have produced two nearly
identical curves and an implied difference that is not there.

**What is reported, per `f`, per stratum — all closed-form, with no sampling and no seeds:**

**Everything below is defined against an integer withheld count `W`, not against `f` directly.**
A withheld cohort has an integer size; `f · N` generally is not one. Fixing `W` first is what makes
the rest well defined:

```
N  = number of eligible diseases
W  = min(max(floor(f · N + 0.5), 1), N − 1)      # round half up, then clamp
```

The clamp is the §6.2 guard in arithmetic form: neither side may be empty. `N < 2` is refused
rather than clamped. **Every quantity below uses `W`, never `f · N`**, and the audit reports `W`
alongside `f` so the rounding is visible rather than implied.

| | Quantity | Form |
|---|---|---|
| 1 | Largest-remainder **quota** for the stratum | Integer, deterministic, summing to `W` |
| 2 | **Expected** diseases withheld under a uniform draw | `E[X_s] = W · n_s / N` — hypergeometric mean, **generally fractional** |
| 3 | **Standard deviation** of (2) | `sqrt( W · (n_s/N) · (1 − n_s/N) · (N − W)/(N − 1) )` |
| 4 | **P(stratum contributes zero withheld diseases)** — `X_s = 0` | `C(N − n_s, W) / C(N, W)` |
| 5 | **P(stratum retains zero diseases)** — `X_s = n_s` | `C(N − n_s, W − n_s) / C(N, W)`, zero when `W < n_s` |

**(3) is reported as a standard deviation, not a variance**, because it is then in disease-count
units and directly comparable to (2). The document, the JSON key and the tests all use the standard
deviation; reporting both would let them drift.

**(5) was wrong in an earlier revision and is worth naming, because of how it survived.** It was
given as `C(n_s, W) / C(N, W)`, which is the probability that the *entire withheld set came from
this stratum* — the event `X_s = W`, not `X_s = n_s`. The two coincide only when `W = n_s`, and the
brute-force check that was supposed to catch it used exactly that case. **The audit's tests must
therefore enumerate small universes across all three regimes — `W < n_s`, `W = n_s`, `W > n_s` —
and check both (4) and (5).** A single case is not a check when the failure mode is a degenerate
coincidence.

**(1) and (2) are reported separately and must not be conflated.** An earlier draft called them
identical and gave the mean as `f · n_s`; both were wrong. An integer quota and a fractional
expectation are different objects, and — a further correction — the gap between *them* has exactly
**one** source: the per-bucket integer rounding the quota performs. Both are computed from the same
`W`, so nothing else can separate them. The rounding of `f · N` to `W` explains a **different**
comparison: the gap between an ideal `f · n_s` and the realised-draw expectation `W · n_s / N`.

**The quotas are independent diagnostic targets, not an allocation.** Quotas computed marginally
over the phenotype-count, gene-count, KG-degree and capacity stratifications are in general **not
jointly realisable by any single disease subset** — one subset cannot generally hit four marginals
at once. Each column answers "what would balance on *this* axis cost?", and the audit describes them
as such. It does not describe, propose or implement a stratified allocation.

(4) and (5) are the decision-relevant pair, and they name **two different failures** that must not
be described in the same words:

| Event | What is lost | JSON key |
|---|---|---|
| `X_s = 0` — nothing from the stratum is withheld | **No validation representation.** The validation metric is silent about this stratum | `p_no_validation_representation` |
| `X_s = n_s` — the whole stratum is withheld | **No training representation.** The deployed model has no patient supervision anywhere in this stratum | `p_no_training_representation` |

This is why the choice between a uniform and a stratified draw is not merely stylistic: for a thin
stratum both failures carry real probability, and each is a number rather than a preference.

**Settled parameters, stated here so nothing is left to negotiate:**

- `f ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}`. 0.15 is the upstream value (§1.6) and the working
  default; the rest bound it on both sides.
- **Every numeric input is validated before the graph is loaded and before anything is written.**
  An evidence artifact that silently clamped a fraction of `−1`, or called a budget sufficient
  because `samples_per_disease` was zero, would be worse than no artifact: the numbers would look
  reportable. The domains are checked at the API, not only at the command line.
- **Strata, reported marginally and never crossed:** phenotype-count band, gene-count band,
  KG-degree band, and `C(P, k)` generator-capacity band. Crossed cells would be mostly empty at
  these sizes and would turn one table into a combinatorial one.
- **`k` is the generator's own configured rule**, not a nominal value: `C(P, k)` is computed with
  the same `k = min(max(min_phenotypes, int(P · (1 − drop_rate))), max_phenotypes, P)` the generator
  applies. A capacity band computed from a different `k` would describe a generator we do not run.
- **Eligibility comes from one shared helper**, exported from `src.kg`, which the generator and the
  audit both call. A second implementation of "which diseases are eligible" would be able to
  disagree with the first, which is the failure this whole document exists to prevent.
- **Missing values get their own explicit bucket.** Never imputed, never silently dropped.
- **Generator parameters are audit *inputs*, and are recorded as such.** `min_phenotypes`,
  `max_phenotypes`, `phenotype_drop_rate`, and the current train and validation budgets are supplied
  to the audit and echoed into its output under a name that marks them **assumptions**, not observed
  history. No generation manifest exists yet (§6.2 introduces the first one), so the configuration
  an existing workspace was built under is **not recoverable from that workspace** and must not be
  presented as if it were.
- **Rounding is largest-remainder over the stratum buckets**, whose quotas sum to `W` and none of
  which may exceed its own bucket size. Equal fractional remainders break on a **canonical bucket
  key**: numeric bands by ascending lower bound, the explicit **missing bucket last**, and the band
  label as the final tie-break so the order is total. The quota is assigned to a *bucket*, not to a
  disease. Reproducible without a seed.
- **No joint-imbalance scalar.** A single number over crossed strata does not help choose `f`, and
  a composite score invites exactly the menu-reading this section removed. Cheap to add later if it
  is ever wanted; it buys nothing now.

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

§1.6 is read from the published article — Alsentzer et al., npj Digital Medicine 8:380 (2025), DOI
`10.1038/s41746-025-01749-1` — and every quotation in it is verbatim from that text. The 85.0% /
15.0% figure is arithmetic on the patient counts the paper states.

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
