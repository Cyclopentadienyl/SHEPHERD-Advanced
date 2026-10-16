# SHEPHERD-Advanced Diagnosis Report

- **Patient ID**: m3c_smoke
- **Session ID**: sess_d6ac54f51afa
- **Timestamp**: 2026-10-16T20:54:21.026777
- **Inference time (ms)**: 302.2
- **Candidates**: 10
- **Query phenotypes**: HP:0001250, HP:0001263

## What produced this result

| Part | State | Value |
|------|-------|-------|
| Checkpoint | ✓ verified | `sha256:e94009526c80f97bcedf2d9a20a834d5e9bcf450f1036733599a36365d30fd66` |
| Knowledge graph (`kg.json`) | ✓ verified | `sha256:6cae2d1a58690eec9aa9c5e3ca9182c2fc942db0f468a5123983251bd4c51e43` |
| Workspace manifest (`split_manifest.json`) | ✓ verified | `sha256:131cdf81cfa20ab8b44609be78473d6dce3f3a18d4436fafd589c6221ddcf664` |
| Shortest-path table | ✓ verified | build\_id `fe1a7d148ae7c015dde0f3907347ee7b` · max\_hops 5 |
| Dataset record (`kg.provenance.json`) | ✓ verified · files | `sha256:2c7abdb6660cbbbbcb7579e74797a9bfbcd775114b7b9b93c8de640175e412ef` — origin files: built from identified source files |
| Scoring mode | in effect | `gnn_plus_shortest_path` — the GNN score combined with the shortest-path table |
| Device (in effect; requested) | in effect | `cuda:0` in effect; requested: `auto` |
| Mixing weight eta | in effect | 0.7 |
| Path settings | in effect | `max_path_length` 4 · `path_length_penalty` 0.9 · `aggregation_method` `weighted_sum` · `max_paths_per_candidate` 10 · `use_direct_path_optimization` true |
| Phenotype normalisation rule | in effect | version 1 |

Checked: these files are the ones their records name, and the records agree on the relations marked ✓. Not checked: who changed them, or anything a person able to edit both files and records could alter.

Not a certified medical device; decision support for a qualified clinician.

## Ranked candidates

| Rank | Disease | Confidence | GNN | SP | Evidence |
|:----:|---------|:----------:|:---:|:--:|----------|
| 1 | **hyperprolinemia type 1** | 0.837 | 0.982 | 0.500 | 🟢 Strong path support |
| 2 | **N-acetylaspartate deficiency** | 0.835 | 0.978 | 0.500 | 🟢 Strong path support |
| 3 | **cerebral palsy, spastic quadriplegic, 3** | 0.834 | 0.977 | 0.500 | 🟢 Strong path support |
| 4 | **branched-chain keto acid dehydrogenase kinase deficiency** | 0.833 | 0.976 | 0.500 | 🟢 Strong path support |
| 5 | **developmental and epileptic encephalopathy, 45** | 0.833 | 0.976 | 0.500 | 🟢 Strong path support |
| 6 | **neurodevelopmental disorder with or without early-onset generalized epilepsy** | 0.831 | 0.973 | 0.500 | 🟢 Strong path support |
| 7 | **intellectual disability, autosomal recessive 64** | 0.831 | 0.973 | 0.500 | 🟢 Strong path support |
| 8 | **LSM7-related leukodystrophy and cerebellar atrophy** | 0.827 | 0.967 | 0.500 | 🟢 Strong path support |
| 9 | **intellectual disability-hypotonia-spasticity-sleep disorder syndrome** | 0.823 | 0.962 | 0.500 | 🟢 Strong path support |
| 10 | **autosomal recessive ataxia due to ubiquinone deficiency** | 0.821 | 0.958 | 0.500 | 🟢 Strong path support |

## Per-candidate detail

### #1 — hyperprolinemia type 1
`MONDO:0009400` · confidence 0.8372

**Evidence**

### 🟢 Strong path support

> Direct knowledge-graph evidence found (shortest path = 1 hop).

**Mode A — Direct Path Evidence** (shortest: 1 hops)

1. `HP:0001250 → MONDO:0009400`

**Full explanation**

## Diagnosis Candidate: hyperprolinemia type 1
Confidence: HIGH (score: 0.837)

### Matching Phenotypes
- Seizure (HP:0001250)
- Global developmental delay (HP:0001263)

### Evidence Paths
**Direct Evidence:**
- Seizure --[phenotype of disease]--> hyperprolinemia type 1 (score: 0.616)

---
*Note: This is a computational prediction for clinical review. Final diagnosis should be made by qualified healthcare professionals based on comprehensive clinical evaluation.*

---

### #2 — N-acetylaspartate deficiency
`MONDO:0013549` · confidence 0.8349

**Evidence**

### 🟢 Strong path support

> Direct knowledge-graph evidence found (shortest path = 1 hop).

**Mode A — Direct Path Evidence** (shortest: 1 hops)

1. `HP:0001250 → MONDO:0013549`
2. `HP:0001263 → MONDO:0013549`

**Full explanation**

## Diagnosis Candidate: N-acetylaspartate deficiency
Confidence: HIGH (score: 0.835)

### Matching Phenotypes
- Seizure (HP:0001250)
- Global developmental delay (HP:0001263)

### Evidence Paths
**Direct Evidence:**
- Seizure --[phenotype of disease]--> N-acetylaspartate deficiency (score: 0.900)
- Global developmental delay --[phenotype of disease]--> N-acetylaspartate deficiency (score: 0.900)

---
*Note: This is a computational prediction for clinical review. Final diagnosis should be made by qualified healthcare professionals based on comprehensive clinical evaluation.*

---

### #3 — cerebral palsy, spastic quadriplegic, 3
`MONDO:0014862` · confidence 0.8336

**Evidence**

### 🟢 Strong path support

> Direct knowledge-graph evidence found (shortest path = 2 hops).

**Mode A — Direct Path Evidence** (shortest: 2 hops)

1. `HP:0001250 → ADD3 → MONDO:0014862`
2. `HP:0001263 → ADD3 → MONDO:0014862`

**Supporting genes**: ADD3

**Full explanation**

## Diagnosis Candidate: cerebral palsy, spastic quadriplegic, 3
Confidence: HIGH (score: 0.834)

### Matching Phenotypes
- Seizure (HP:0001250)
- Global developmental delay (HP:0001263)

### Key Associated Genes
- ADD3 (2 evidence paths)

### Evidence Paths
**Direct Evidence:**
- Seizure --[gene has phenotype]--> ADD3 --[gene associated with disease]--> cerebral palsy, spastic quadriplegic, 3 (score: 0.729)
- Global developmental delay --[gene has phenotype]--> ADD3 --[gene associated with disease]--> cerebral palsy, spastic quadriplegic, 3 (score: 0.729)

---
*Note: This is a computational prediction for clinical review. Final diagnosis should be made by qualified healthcare professionals based on comprehensive clinical evaluation.*

---

### #4 — branched-chain keto acid dehydrogenase kinase deficiency
`MONDO:0013970` · confidence 0.8333

**Evidence**

### 🟢 Strong path support

> Direct knowledge-graph evidence found (shortest path = 1 hop).

**Mode A — Direct Path Evidence** (shortest: 1 hops)

1. `HP:0001263 → MONDO:0013970`
2. `HP:0001250 → MONDO:0013970`

**Full explanation**

## Diagnosis Candidate: branched-chain keto acid dehydrogenase kinase deficiency
Confidence: HIGH (score: 0.833)

### Matching Phenotypes
- Seizure (HP:0001250)
- Global developmental delay (HP:0001263)

### Evidence Paths
**Direct Evidence:**
- Global developmental delay --[phenotype of disease]--> branched-chain keto acid dehydrogenase kinase deficiency (score: 0.900)
- Seizure --[phenotype of disease]--> branched-chain keto acid dehydrogenase kinase deficiency (score: 0.396)

---
*Note: This is a computational prediction for clinical review. Final diagnosis should be made by qualified healthcare professionals based on comprehensive clinical evaluation.*

---

### #5 — developmental and epileptic encephalopathy, 45
`MONDO:0014942` · confidence 0.8329

**Evidence**

### 🟢 Strong path support

> Direct knowledge-graph evidence found (shortest path = 1 hop).

**Mode A — Direct Path Evidence** (shortest: 1 hops)

1. `HP:0001250 → MONDO:0014942`
2. `HP:0001263 → MONDO:0014942`

**Full explanation**

## Diagnosis Candidate: developmental and epileptic encephalopathy, 45
Confidence: HIGH (score: 0.833)

### Matching Phenotypes
- Seizure (HP:0001250)
- Global developmental delay (HP:0001263)

### Evidence Paths
**Direct Evidence:**
- Seizure --[phenotype of disease]--> developmental and epileptic encephalopathy, 45 (score: 0.900)
- Global developmental delay --[phenotype of disease]--> developmental and epileptic encephalopathy, 45 (score: 0.900)

---
*Note: This is a computational prediction for clinical review. Final diagnosis should be made by qualified healthcare professionals based on comprehensive clinical evaluation.*

---

### #6 — neurodevelopmental disorder with or without early-onset generalized epilepsy
`MONDO:0030930` · confidence 0.8309

**Evidence**

### 🟢 Strong path support

> Direct knowledge-graph evidence found (shortest path = 1 hop).

**Mode A — Direct Path Evidence** (shortest: 1 hops)

1. `HP:0001250 → MONDO:0030930`

**Full explanation**

## Diagnosis Candidate: neurodevelopmental disorder with or without early-onset generalized epilepsy
Confidence: HIGH (score: 0.831)

### Matching Phenotypes
- Seizure (HP:0001250)
- Global developmental delay (HP:0001263)

### Evidence Paths
**Direct Evidence:**
- Seizure --[phenotype of disease]--> neurodevelopmental disorder with or without early-onset generalized epilepsy (score: 0.562)

---
*Note: This is a computational prediction for clinical review. Final diagnosis should be made by qualified healthcare professionals based on comprehensive clinical evaluation.*

---

### #7 — intellectual disability, autosomal recessive 64
`MONDO:0020846` · confidence 0.8308

**Evidence**

### 🟢 Strong path support

> Direct knowledge-graph evidence found (shortest path = 1 hop).

**Mode A — Direct Path Evidence** (shortest: 1 hops)

1. `HP:0001263 → MONDO:0020846`
2. `HP:0001250 → MONDO:0020846`

**Full explanation**

## Diagnosis Candidate: intellectual disability, autosomal recessive 64
Confidence: HIGH (score: 0.831)

### Matching Phenotypes
- Seizure (HP:0001250)
- Global developmental delay (HP:0001263)

### Evidence Paths
**Direct Evidence:**
- Global developmental delay --[phenotype of disease]--> intellectual disability, autosomal recessive 64 (score: 0.900)
- Seizure --[phenotype of disease]--> intellectual disability, autosomal recessive 64 (score: 0.180)

---
*Note: This is a computational prediction for clinical review. Final diagnosis should be made by qualified healthcare professionals based on comprehensive clinical evaluation.*

---

### #8 — LSM7-related leukodystrophy and cerebellar atrophy
`MONDO:0978294` · confidence 0.8266

**Evidence**

### 🟢 Strong path support

> Direct knowledge-graph evidence found (shortest path = 1 hop).

**Mode A — Direct Path Evidence** (shortest: 1 hops)

1. `HP:0001263 → MONDO:0978294`
2. `HP:0001250 → MONDO:0978294`

**Full explanation**

## Diagnosis Candidate: LSM7-related leukodystrophy and cerebellar atrophy
Confidence: HIGH (score: 0.827)

### Matching Phenotypes
- Seizure (HP:0001250)
- Global developmental delay (HP:0001263)

### Evidence Paths
**Direct Evidence:**
- Global developmental delay --[phenotype of disease]--> LSM7-related leukodystrophy and cerebellar atrophy (score: 0.900)
- Seizure --[phenotype of disease]--> LSM7-related leukodystrophy and cerebellar atrophy (score: 0.000)

---
*Note: This is a computational prediction for clinical review. Final diagnosis should be made by qualified healthcare professionals based on comprehensive clinical evaluation.*

---

### #9 — intellectual disability-hypotonia-spasticity-sleep disorder syndrome
`MONDO:0014210` · confidence 0.8234

**Evidence**

### 🟢 Strong path support

> Direct knowledge-graph evidence found (shortest path = 2 hops).

**Mode A — Direct Path Evidence** (shortest: 2 hops)

1. `HP:0001250 → ANK3 → MONDO:0014210`
2. `HP:0001263 → ANK3 → MONDO:0014210`

**Supporting genes**: ANK3

**Full explanation**

## Diagnosis Candidate: intellectual disability-hypotonia-spasticity-sleep disorder syndrome
Confidence: HIGH (score: 0.823)

### Matching Phenotypes
- Seizure (HP:0001250)
- Global developmental delay (HP:0001263)

### Key Associated Genes
- ANK3 (2 evidence paths)

### Evidence Paths
**Direct Evidence:**
- Seizure --[gene has phenotype]--> ANK3 --[gene associated with disease]--> intellectual disability-hypotonia-spasticity-sleep disorder syndrome (score: 0.729)
- Global developmental delay --[gene has phenotype]--> ANK3 --[gene associated with disease]--> intellectual disability-hypotonia-spasticity-sleep disorder syndrome (score: 0.729)

---
*Note: This is a computational prediction for clinical review. Final diagnosis should be made by qualified healthcare professionals based on comprehensive clinical evaluation.*

---

### #10 — autosomal recessive ataxia due to ubiquinone deficiency
`MONDO:0012784` · confidence 0.8206

**Evidence**

### 🟢 Strong path support

> Direct knowledge-graph evidence found (shortest path = 1 hop).

**Mode A — Direct Path Evidence** (shortest: 1 hops)

1. `HP:0001250 → MONDO:0012784`

**Full explanation**

## Diagnosis Candidate: autosomal recessive ataxia due to ubiquinone deficiency
Confidence: HIGH (score: 0.821)

### Matching Phenotypes
- Seizure (HP:0001250)
- Global developmental delay (HP:0001263)

### Evidence Paths
**Direct Evidence:**
- Seizure --[phenotype of disease]--> autosomal recessive ataxia due to ubiquinone deficiency (score: 0.900)

---
*Note: This is a computational prediction for clinical review. Final diagnosis should be made by qualified healthcare professionals based on comprehensive clinical evaluation.*

---