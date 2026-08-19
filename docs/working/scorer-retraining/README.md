# Scorer retraining — selecting a scorer bundle, and the checkpoint schema

**Status:** scoping only. Nothing here is scheduled, and no gate below is
cleared. This phase exists because the deployed scorer is **interim by
decision**, not because anything is broken: `DISEASE_SCORER_POLICY.md`
statement 2 records raw cosine as interim precisely because it reflects what the
deployed checkpoint's training objective optimised, and says a change requires
this track.

Moved here from `../task-scope/README.md` §6, where it was misfiled. It arrived
from a conversation about training-time similarity and has nothing to do with
the supplied-short-list scenario that document is named for. **One folder, one
README** — no schema, variants, legacy, experiment or history subdocuments.

**What this phase is not.** It is not a claim that the scoring mechanism was
omitted from the project. Work item B measures and preserves the current
deployed scorer faithfully; this phase compares bundles under a fixed protocol
and only then adds explicit checkpoint-schema support. **Replacing the scorer at
inference — with Euclidean, or a learned head — before the matching retraining
would create the score-semantics error, not fix one.**

---

## 1. Verified facts

Read from the tree, not recalled.

| # | Fact | Source |
|---|---|---|
| R1 | `PhenotypeDiseaseMatcher` exposes `similarity_type`: `bilinear` / `mlp` / `cosine` | `models/gnn/shepherd_gnn.py:332` |
| R2 | `DiagnosisHead` exposes `similarity_type`: `learned` / `cosine` / `euclidean`, branch at `:217` | `models/decoders/heads.py:52` |
| R3 | Neither class is constructed on any production path | no construction call in `src/` or `scripts/` |
| R4 | The live training similarity is **hardcoded cosine in three places** | `trainer.py:761-763`; `loss_functions.py:323-331`; `scoring.py:210-211` |
| R5 | `PhenotypeDiseaseMatcher`'s unwired status was **already recorded** before this phase existed | `RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md:433` |
| R6 | It is **not dead code** — it is one side of an open decision | same document §6.B item 2, §7 question 2 |
| R7 | All four `heads.py` classes are constructed **by tests** | `tests/unit/test_models.py:537-661` |
| R8 | Three different patient aggregations exist | masked mean `trainer.py:744-751`; masked mean then a learned aggregator `shepherd_gnn.py:346,397,422`; `phenotype_encoder` MLP plus attention `heads.py:69-76,100-106` |
| R9 | The checkpoint callback has **two formats**. Weights-only writes `state_dict` alone. The normal format writes `epoch`, `state_dict`, `optimizer_state_dict`, `logs`, a serialized trainer `config`, optionally `scheduler_state_dict`, and optionally a `data_fingerprint`. **Neither carries a producer identity or an explicit scorer schema**, and which format a given file used is not recorded in it | `training/callbacks.py:288-317` |
| R10 | The paper pairs its geometry with its objective: Eq 18's negative squared L2 is trained by Eq 19's NCA loss; this repository pairs cosine with a contrastive loss that L2-normalises internally | `SP_SCORE_GUIDE.md` §3; `loss_functions.py:323-331` |

**`DiagnosisHead`'s euclidean branch is not "the paper's Eq 18".** It shares a
distance family; the patient encoder, objective, candidate universe and
calibration have not been shown to match.

---

## 2. Decisions

### 2.1 The unit of comparison is a scorer bundle

```
ScorerVariant = patient_encoder + score_family + training_objective + output_semantics
```

All four are load-bearing, and R8 shows three patient encoders already exist in
the tree.

### 2.2 Order

1. Fix the task, candidate universe, cohort, split, and **evaluation protocol**.
2. Enumerate a small, **justified** set of scorer bundles.
3. Define the checkpoint scorer schema for each bundle.
4. Train the bundles.
5. Evaluate them under the fixed protocol.
6. Select one.
7. Add inference support for the selected explicit schema.

Fixing the protocol at step 1 — before enumeration — is what stops the protocol
being chosen to suit a bundle. **No Cartesian-product sweep.**

### 2.3 Checkpoint scorer schema

The checkpoint records a **versioned scorer schema** covering at least: patient
encoder/aggregation type and version; score family and direction; score
transform if any; training objective; and the architecture parameters needed to
reconstruct the scorer. The loader instantiates only a supported explicit schema
and **fails closed** on unknown or incompatible metadata.

**The inference boundary:**

> Explicit checkpoint metadata is authoritative. Structural inference from
> state-dict evidence is permitted only as a **bounded legacy compatibility
> rule**, where the mapping from observed keys or shapes to a supported
> architecture is sufficiently specific, validated by strict loading, and
> documented as a fallback rather than a general semantic detector.
>
> **State-dict evidence may recover structure. It does not, by itself, establish
> training objective or score semantics.**

`resolve_arch_params` sits inside that rule: conv architectures leave
distinguishing parameter names, explicit `model_config` stays authoritative, and
strict loading validates structural compatibility.

### 2.4 Legacy checkpoints — four kinds

| Kind | What may be said |
|---|---|
| Known legacy repository checkpoints, with provenance or as an accepted artifact family | May be classified **legacy headless cosine**; the historical objective stated **only where provenance supports it**. R9 gives that phrase a concrete hook: the normal format's optional `data_fingerprint` and its `logs` and `config` are what provenance could rest on — none is guaranteed present, and none is a scorer schema |
| Unknown schema-less checkpoints that strict-load into headless `ShepherdGNN` | Structurally compatible; **training objective unknown** |
| Explicit future scorer-schema checkpoints | Read the schema |
| Unsupported | Refused |

Serving a *known* legacy family with interim raw cosine may be an approved
compatibility policy.

### 2.5 The unwired switches

Severity **LOW**; nothing here blocks or precedes any B-0 stage. **Removal stays
off the table** until the authoritative patient encoder/scorer decision is made
(findings §7 question 2). The whole treatment is a short code-local annotation,
added when this phase or another legitimate edit touches those definitions —
**not a standalone work item or hotfix**. It says only: experimental and
unwired; not constructed by the current training or inference pipeline;
`similarity_type` is not a live runtime or checkpoint contract; the decision is
tracked in `RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md` §7.

---

## 3. Rejected alternatives

Kept as one line each. The reasoning that produced them is in git; what is worth
carrying forward is the conclusion and, where a factual claim was refuted, the
counter-example that refuted it.

| Rejected | Why |
|---|---|
| Choosing a score family before the experiments | Predetermines the answer the experiments exist to find |
| Comparing (score family, objective) pairs | Insufficient — the patient encoder is load-bearing too (R8) |
| A generic runtime dropdown for the score family | Breaks checkpoint/objective semantics; the schema must be explicit first |
| "Infer what leaves parameter evidence, record what does not" | Rested on "a score family has no parameters", which is false: `DiagnosisHead`'s `learned` branch builds a parametered `similarity_net` while `cosine` and `euclidean` build nothing — and those two are **identical** in the state dict |
| "A schema-less checkpoint that strict-loads is contrastive-cosine trained" | Infers an objective from structure, which §2.3 forbids; and its antecedent "produced by this repository's trainer" is not checkable, since neither callback format carries a producer identity (R9) — the serialized `config` is trainer configuration, not a scorer contract, and the weights-only format has none of it |
| Deleting the unwired classes | Would delete one side of an open architecture decision (R6) |
| A test asserting no constructor call exists | Breaks the day someone legitimately wires it |
| Moving this plan into `DISEASE_SCORER_POLICY.md` | That file is the **accepted authority record**; this is an unexecuted experimental plan |

---

## 4. Open questions and reopen conditions

- **Which patient encoder is authoritative** — mean-pooled cosine, or
  `PhenotypeDiseaseMatcher`'s learned aggregation. Tracked at
  `RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md` §7 question 2; this phase
  cannot start without it (§5).
- **Which bundles are worth training.** The set must be small and justified,
  never a product of the axes.
- **Whether the paper's bundle can be reproduced at all** — R10 gives the
  geometry and objective, but the paper's patient encoder and output semantics
  have not been established here.
- **Reopen this phase** if the authoritative-encoder decision changes, if the
  paper-parity track is scheduled, or if a checkpoint arrives whose scorer
  cannot be classified under §2.4.

---

## 5. Dependencies and gates

| Gate | Status |
|---|---|
| Authoritative patient encoder/scorer decision (findings §7 Q2) | **Not cleared** |
| A fixed evaluation protocol, cohort and split | **Not defined** |
| Gene-side work, if gene bundles are in scope — starts with the gradient-path audit against a real workspace artifact | **Not started** (`../task-scope/README.md` Q5) |
| Institutional scheduling | **Not requested** |

**Nothing in work item B depends on this phase**, and this phase does not block
B-0.4, B-0.5 or the legacy-removal checklist.

**Authority above everything here:** `docs/DISEASE_SCORER_POLICY.md`.
