# The shortest-path (SP) score — what it means, when to use it, how it is computed

**Audience.** §1 and §2 are written for clinicians and require no engineering or mathematical
background. §3 is for developers, bioinformaticians and reviewers who need the exact definition.

**What this document is not.** It explains what the SP score means and how to read it safely. **It
does not define ranking policy.** That authority belongs to
[`DISEASE_SCORER_POLICY.md`](DISEASE_SCORER_POLICY.md), which records what the disease scorer is,
what role SP may play, and why.

**Why this document exists.** The SP score is easy to misread, and one misreading is clinically
consequential: treating a low score as evidence *against* a diagnosis. It is not. This document
exists so that anyone who sees an SP number knows precisely what it does and does not say.

---

> ## Status — read this before anything else
>
> **Today, the SP score is part of the ranking.** How a candidate is scored depends on which
> artifacts are loaded:
>
> | Loaded | Score used for ranking |
> |---|---|
> | Trained model **and** shortest-path table | `0.7 × model score + 0.3 × SP` (`src/inference/pipeline.py:1310`) |
> | Trained model only | Model score alone |
> | No trained model | Path-reasoning score — **a different quantity written into the same field** |
>
> **Also today: candidates are discovered by path search before any scoring happens.** A disease
> with no qualifying path from the patient's phenotypes is never scored at all, whatever the model
> would have said about it.
>
> **Both of these are being changed.** The approved target is that every disease is scored by the
> model, that SP is removed from the ranking entirely, and that SP is re-exposed as a separate,
> opt-in analysis over candidates that have already been ranked. Until that work lands, §2's
> guidance describes how to *interpret* an SP value, not a switch that exists.
>
> The decision, its evidence and its current implementation status are in
> [`DISEASE_SCORER_POLICY.md`](DISEASE_SCORER_POLICY.md). Sections §1 and §3 describe the score
> itself and are unaffected by that change.

---

# §1. What the score means

## The short version

> **The SP score does not measure how likely the patient has the disease.**
>
> **It measures how closely the patient's recorded phenotypes and a candidate disease are connected
> in the knowledge graph this system currently has loaded, within a limit of five steps.**

Everything else in this section follows from that sentence — including, importantly, what the
knowledge graph is and is not.

## A picture, and where the picture ends

Imagine a wall of cards. There is a card for each symptom, each gene, each disease. Wherever the
system's source databases record a relationship — *this symptom occurs in this disease*, *this gene
causes this disease*, *this symptom is a subtype of that symptom* — a string is tied between two
cards.

**The wall is not "everything medicine knows".** It is what has been loaded into this system, from a
specific set of curated sources, on a specific date. The sources this system is built to ingest
include HPO, MONDO, Orphanet, OMIM, ClinVar, DisGeNET, GO, Reactome, PubMed/PubTator and the
model-organism databases MGI and ZFIN (`src/core/types.py:145-168`) — but **which of them a
particular deployment actually contains depends on how its graph was built**, and that is a question
for whoever built it, not something the score reveals. The wall is finite, assembled by other people
for other purposes, and certainly incomplete.

A patient arrives with several symptoms. For one candidate disease, the SP score asks:

> **Starting from each of the patient's symptom cards, how many strings must I follow to reach this
> disease card?**

| Strings to follow | What that means |
|---|---|
| **1** | The graph records a direct relationship between this symptom and this disease |
| **2** | No direct relationship, but the symptom connects to a gene and that gene to the disease |
| **3–5** | Connected only through a longer chain |
| **Cannot reach within 5** | **No path was found in the current knowledge graph within five steps** |

The system does this for each of the patient's symptoms, averages the number of strings, and
converts it so that **fewer strings gives a higher score**.

*Illustration:* lens dislocation → *FBN1* → Marfan syndrome is two strings. A well-recorded
association of this kind scores high.

## The score scale

| Score | Average strings | Plain reading |
|---|---|---|
| 0.50 | 1 | Directly connected in the graph |
| 0.33 | 2 | Connected through one intermediate |
| 0.25 | 3 | Connected, but indirectly |
| 0.17 | 5 | At the edge of the search limit |
| **0.14** | not reachable within 5 | **No path found in the current graph** |

The scale is compressed at the far end — not by design, but as a mathematical consequence of the
`1/(1+distance)` transform. The gap between 1 string and 2 strings (0.167) is **seven times** the
gap between 5 strings and no path found (0.024). Nearly all of the score's discriminating power sits
in the 1–3 string range.

## The single most important point

**A low SP score does not mean the patient does not have the disease.**

A low SP score means that the candidate is, on average, connected to the patient's recorded
phenotypes only through **long paths, missing paths within the five-step limit, or a mixture of
both**. The score averages across phenotypes, so it does not by itself tell you which.

**Only the minimum computable value — 0.143 (1/7) under the default five-step configuration — means
that no path was found within five steps for *any* phenotype included in the calculation.** A score
of 0.17 is equally consistent with every phenotype being genuinely connected at five steps.

Where a phenotype-candidate pair does have no path found, that single observation is consistent with
several very different situations:

- the disease is genuinely unrelated to this patient;
- the presentation is real but has not been reported;
- the relevant association exists in the literature but has not been curated into any of our source
  databases;
- it was curated, but our ingestion or identifier mapping missed it;
- our copy of the graph is out of date;
- the connection exists but requires more than five steps.

**The SP score cannot distinguish any of these from one another.**

Rare-disease diagnosis often involves atypical or incompletely represented presentations, so the
second possibility must not be dismissed.

A useful one-line summary:

> **The SP score measures how *unsurprising* a candidate would be, given what this system currently
> has loaded.**

## Five things the score cannot tell you

**1. It cannot tell you what kind of connection it found.** Every relationship in the graph counts
as one string, regardless of type. "Two strings" might be:

- symptom → *is a subtype of* → broader symptom → *occurs in* → disease — where the first step is
  **movement within a classification hierarchy** and only the second says anything clinical; or
- symptom → *associated with* → gene → *causes* → disease — **a mechanistic chain throughout**.

Both appear in the score as the number `2`. A high SP score can therefore rest largely on
classification proximity rather than on mechanism. **This is an important possible source of false
confidence in this number.** To see what the steps actually are, use the reasoning-path evidence,
which preserves relationship types; the SP score does not.

**2. It cannot tell you how strong the evidence is.** A single case report and a large cohort study
are both "one string", weighted identically.

**3. It cannot distinguish "never studied" from "studied and refuted".** The graph records that a
relationship was asserted, not whether it has held up.

**4. A high score is not proof either.** The limits above are symmetric. A short path shows that
these sources record a connection — not that the connection is causal, current, clinically
meaningful, or relevant to this patient.

**5. It is dragged down by the patient's least-connected symptom.** The score averages distances
across all of the patient's recorded symptoms, so one unconnected feature pulls the whole value
down — and **the fewer symptoms recorded, the more heavily a single unconnected one counts**.

Worked example, with every number shown so the effect can be checked rather than taken on trust.
Take a candidate two steps from each connected symptom:

| Patient | Distances | Average | Score |
|---|---|---|---|
| 3 symptoms, all connected | 2, 2, 2 | 2.00 | 0.333 |
| 3 symptoms, one with no path found | 2, 2, 6 | 3.33 | **0.231** |
| 10 symptoms, all connected | 2 × 10 | 2.00 | 0.333 |
| 10 symptoms, one with no path found | 2 × 9, 6 | 2.40 | **0.294** |

The size of the drop depends entirely on how close the other symptoms are, so no single percentage
describes it.

Clinically this cuts both ways. An atypical feature *should* sometimes argue against a diagnosis.
But rare diseases frequently present as "mostly consistent, plus one feature nobody can explain",
and that unexplained feature is often the one that eventually solves the case. **The SP score
penalises it.**

---

# §2. How to use the score

## It answers a different question from the model's score

The system produces two scores with **different intended interpretations**. They are not
independent — both are derived from the same knowledge graph, and the model is trained on it — but
they are asking different things, and confusing them is the main risk.

| | Question it addresses | Clinical analogue |
|---|---|---|
| **Model score** | *Does this patient's pattern resemble this disease?* | The physician who recognises a gestalt: "I have not seen this exact combination, but it looks like D" |
| **SP score** | *Is this connection already recorded in what we loaded?* | The physician checking a reference: "this combination appears in our sources" |

Both are legitimate. They are not the same question, and averaging them into one number — which the
system currently does — hides which one is speaking.

## Read the two scores together, never SP alone

|  | **High SP** (closely connected in the graph) | **Low SP** (weakly or not connected) |
|---|---|---|
| **High model score** | **Expected.** Model agreement plus a short recorded connection. Usually quick to check against known criteria. | **Model-supported, weakly connected in the current graph.** May warrant review for a novel presentation, a source-coverage gap, a mapping problem, or model error. |
| **Low model score** | Recorded connection, but the pattern does not fit this patient. | **Neither score provides positive support.** |

**The bottom-right cell is why a low SP score alone is useless.** Selecting on "low SP" returns an
enormous and overwhelmingly wrong set.

*Corrected against measurement.* This paragraph used to say most diseases have no path to any given
patient. They do: the median phenotype reaches 64.3% of diseases within 5 hops, and only 1.36% of
phenotypes reach none ([`EVIDENCE_M5.json`](working/EVIDENCE_M5.json)). The cell's point stands for a
different reason — a *reachable but distant* disease also scores near the floor, so a low SP score
does not separate "unrelated" from "connected at five steps".

## A warning that must not be softened

> **High model score with low SP means the two signals disagree. It does not mean the candidate is
> more likely to be correct.**

Disagreement is worth a clinician's attention because *something* is unusual — but that something
may be a genuinely novel presentation, a gap in what we loaded, an identifier-mapping error, or the
model being wrong. **The SP score cannot tell you which.**

A candidate that is both model-supported and well connected is **corroborated by two sources**; one
that is model-supported but weakly connected has support from one. Corroboration is not the same as
correctness, and this document makes no claim about how often either group turns out to be right —
that has not been measured here. But reading "the signals disagree" as "this is more likely to be
correct" would be a serious error.

## When to consult the SP score

**Consult it when you already have a short list of plausible candidates** — the model's top 10 or
20, or a set you are actively weighing. In that setting all candidates are already reasonable, and
"which of these is best supported by what we have loaded" is a genuine discriminator. This is the
setting the reference paper designed the signal for (§3).

Specifically, it helps with:

- **separating confirmation from discovery** — which high-ranked candidates would a conventional
  reference check also have surfaced, and which would it have missed;
- **triaging review effort** — a high-SP candidate can often be checked against known criteria
  quickly; a low-SP one needs primary reasoning;
- **framing a discussion** — "the model ranks this first and there is a recorded two-step
  connection" is a different statement from "the model ranks this first and our graph has no path".

## When **not** to consult it

| Situation | Why |
|---|---|
| **Ranking the full disease universe** | Candidates beyond the five-step limit all receive the same floor value, so the number stops discriminating there. **Now measured**: on the deployment artifact the median phenotype leaves ~35.7% of diseases beyond the limit, not the great majority ([`EVIDENCE_M5.json`](working/EVIDENCE_M5.json)). It is a large minority, not the expected condition — and it is not the whole of the concern, since a disease reachable at five steps scores `1/6` against a floor of `1/7`. How much of the *reachable* mass sits near that floor is a distance distribution no artifact records. |
| **Looking for undocumented or atypical presentations** | The score penalises exactly what you are looking for. |
| **As evidence that a diagnosis is wrong** | No path found is not evidence of no relationship. See §1. |
| **As an explanation of mechanism** | The score does not know what kinds of steps it counted. Use the reasoning paths instead. |

## On the question of rare diseases ranking too low

A question raised during clinical review: *the candidate list tends to put common diseases first;
how can rare diseases and hidden associations be surfaced?*

**Two mechanisms in the current system plausibly push some candidates upward:**

1. **Candidate discovery is gated by path search**, which stops after a fixed number of paths per
   symptom, in traversal order. Diseases with shorter or denser graph connections are reached first
   and reached more often.
2. **The SP term in the ranking score** directly rewards short graph connections, with a nominal
   coefficient of 0.3. That coefficient is not the term's effective contribution — see §3.

Both are being addressed: the first by scoring every disease whether or not a path exists, the
second by removing SP from the ranking
([`DISEASE_SCORER_POLICY.md`](DISEASE_SCORER_POLICY.md)).

**Two honest qualifications.** First, what the code establishes is a possible bias towards **shorter
or denser graph connections**. Whether denser connection tracks *commonness* of disease is a further
step that the code does not establish and that has not been measured here; a measurement is planned
as part of the candidate-discovery work, using graph connectivity as a proxy.

Second — **the natural-seeming fix, "rank by low SP to find the rare ones", does not work**, for the
reason given above: low SP selects mostly unrelated diseases. The usable instrument is the 2×2:
**among candidates the model already ranks highly, low SP marks those whose ranking is not explained
by a recorded connection.** That is an **annotation** on an already-fixed list, never a ranking
criterion — and never a filter.

**This distinction is normative, not stylistic.** Under
[`DISEASE_SCORER_POLICY.md`](DISEASE_SCORER_POLICY.md) statement 4, SP **cannot change which
candidates are returned or displayed, cannot hide any of them, and cannot alter their order or their
score**. A post-ranking filter would still be able to hide a candidate, which would recreate SP
gating by another route.

**SP may be shown as an annotation, colour, badge, separate column, or detail panel attached to the
already-fixed candidate list. It may not hide, reorder, regroup, rescore, filter, or remove
candidates.**

**A clinician may sort and filter the view by SP inside a dedicated SP analysis surface** — this was
permitted by amendment after the institution asked for it. What that permission does *not* include
is any change to the result itself: the candidate set, the scores and the canonical rank order are
produced by the disease scorer and are immutable. A view operation is a projection over them.

Ten conditions govern that permission — canonical rank stays visible, one action restores the
canonical view, the view says on screen that it is not canonical, exports default to the full
canonical result, the view state is recorded with any action taken from it, a filter permanently
shows how many candidates it is hiding, and the SP values themselves never change with paging,
sorting or filtering. They are set out in
[`DISEASE_SCORER_POLICY.md`](DISEASE_SCORER_POLICY.md) §1.1 and are binding.

The one asymmetry worth knowing as a reader: **a sort hides nothing, so it need not report an
exclusion; a filter does, so it must — always, and before you apply it.**

---

# §3. How the score is computed

## The reference paper's definition

Alsentzer E, Li MM, Kobren SN, Noori A, Kohane IS, Zitnik M. *Few shot learning for
phenotype-driven diagnosis of patients with rare genetic diseases.* **npj Digital Medicine** 8:380
(2025). DOI `10.1038/s41746-025-01749-1`. PMC `PMC12181314`.

For a patient's phenotype set `P` and a **candidate gene** `g` (Methods, *SHEPHERD: discovering
causal genes*):

```
SPLSIM(P, g) = NORM( AGG_{p∈P} ( − d(p, g) ) )                        (Eq 13)

SIM(P, g) = η · EMBSIM(P, g) + (1 − η) · SPLSIM(P, g)                 (Eq 14)
```

where `d(p, g)` is "the minimum number of hops between p and g in the KG", `AGG` is "some
aggregation function (e.g., mean)", `NORM` rescales to `[−1, 1]`, and `η ∈ [0, 1]` weights the two
terms. `EMBSIM(P, g) = tanh(z_Pᵀ W z_g)`, also in `[−1, 1]`.

The paper's stated rationale, quoted:

> "This approach is grounded in the observation that, while methods that learn global network
> topology yield higher overall performance than local methods considering only local network
> information, the latter tend to rank true candidate genes higher **when provided a short list of
> candidate genes**."

That observation is cited to a survey of gene-prioritisation tools (Zolotareva & Kleine,
*J. Integr. Bioinform.* 16, 20180069, 2019) rather than to an experiment within the paper. **No
ablation of SPLSIM appears in the main article as reviewed**; supplementary materials were not
examined. The selected value of η is published in the authors' repository rather than in the
article.

### Where the paper uses it — and where it does not

| SHEPHERD task | Scoring function | Uses SP? | Candidate set |
|---|---|---|---|
| Causal gene discovery | Eq 14 (above) | **yes** | a clinician- or pipeline-supplied gene list — **13.3 genes** on average (EXPERT-CURATED, SD 8.0) or **244.3** (VARIANT-FILTERED, SD 244.0) |
| Patients-like-me | `SIM(Pᵢ,Pⱼ) = −‖z_Pᵢ − z_Pⱼ‖²₂` (Eq 16) | no | all patients |
| Novel disease characterisation | `SIM(P,d) = −‖z_d − z_P‖²₂` (Eq 18) | **no** | **all diseases in the KG** |

**Of the paper's three tasks, only one uses the SP term — the only one with a candidate short
list.** Disease ranking, which is what this system performs, uses embedding similarity alone.

## What this implementation actually computes

Offline (`scripts/compute_shortest_paths.py`), for every phenotype node:

```python
# 1. Flatten the whole KG into an undirected adjacency list.
#    Edge direction and edge type are both discarded.
for edge in kg._edges:
    adj[str(edge.source_id)].append(str(edge.target_id))
    adj[str(edge.target_id)].append(str(edge.source_id))

# 2. Breadth-first search from the phenotype, up to max_hops (default 5).
#    BFS guarantees the first arrival at a node is its minimum distance.
distances = {source: 0}
queue = deque([(source, 0)])
while queue:
    current, depth = queue.popleft()
    if depth >= max_hops:
        continue
    for neighbor in adj.get(current, ()):
        if neighbor not in distances:
            distances[neighbor] = depth + 1
            queue.append((neighbor, depth + 1))

# 3. Keep only gene and disease targets; store (phenotype_idx, target_idx,
#    target_type, distance) as four parallel tensors in shortest_paths.pt.
```

Pairs absent from the table mean "no path found within `max_hops`".

Online (`src/inference/pipeline.py:_calculate_sp_score`), per candidate:

```python
UNREACHABLE = max_hops + 1                      # 6 with the default max_hops = 5

total = 0.0
for ph_idx in patient_phenotype_indices:
    d = lookup(ph_idx, target_idx, target_type)  # linear scan of this phenotype's slice
    total += d if d is not None else UNREACHABLE

avg_distance = total / len(patient_phenotype_indices)
return 1.0 / (1.0 + avg_distance)                # → [1/7, 1/2]
```

**Two distinct return regimes, currently indistinguishable to a caller.** The computation above
yields a value in `[1/7, 1/2]`. Separately, the function returns **`0.0`** when it cannot compute at
all — no shortest-path table loaded, no node mapping, the target absent from the mapping, or none of
the patient's phenotypes resolvable (`src/inference/pipeline.py:1333, 1337, 1347, 1358`). Because
`0.0 < 1/7`, **a lookup or mapping failure produces a lower value than a genuine "no path found"**,
and nothing currently distinguishes the two. Any surface that displays SP should treat `0.0` as
*unavailable*, not as a distance.

## Three ways this differs from the paper — all of them material

| | Paper | This implementation | Consequence |
|---|---|---|---|
| **Task** | candidate **gene** ranking, with a short list | **disease** ranking, over whatever the path search admits | The condition the paper's rationale depends on — a short candidate list — does not hold |
| **Transform** | `NORM(mean(−d))`, min–max rescaled **across the candidate set**, giving `[−1, 1]` | `1/(1 + mean(d))`, an **absolute** transform giving `[1/7, 1/2]` | See below |
| **η value** | hyperparameter-searched over `[0.1, 0.9]`, per task; chosen value not published in the article | fixed at `0.7` (`PipelineConfig.eta`) | The project default is a reasonable choice, not a value inherited from the paper. The in-code comment states this correctly. |

The transform difference has two consequences:

- **η is not the effective weight.** Comparing *maximum theoretical spans*: with `η = 0.7` the model
  term spans 0.7 while the SP term spans 0.107 — nominally 70/30, but 87/13 by span. This is a
  statement about the ranges the two terms can occupy, **not** a measurement of their actual
  contribution, which depends on how widely each varies in practice and has not been measured here.
- **Candidate-relative normalisation removes a patient-level additive offset only when that offset
  is constant across all candidates.** A phenotype contributes little or no discrimination when it
  is *equally* distant from every candidate — for example when it is unreachable from all of them,
  so every candidate receives the same `UNREACHABLE` value. It is **not** generally cancelled merely
  because all its distances are large: distances of 5 to one candidate and 6 to another still
  contribute to the spread. **The transform used here performs no explicit candidate-relative
  cancellation at all**, so §1's "least-connected symptom" effect is not attenuated as it can be in
  the paper's formulation.

**Unverified equivalence.** This implementation traverses an undirected, type-erased graph
(`build_undirected_adjacency`, above). The paper states only that `d(p, g)` is "the minimum number
of hops between p and g in the KG" and does not specify directionality or type handling. Whether the
paper's traversal is equivalent is **not established here**; §1's caveat about hop types is
demonstrated for this implementation and assumed, not verified, for the paper.

## Cost, and why it does not constrain the design

The lookup is a linear scan of one phenotype's slice per candidate per phenotype. With a few hundred
candidates this is unnoticeable. Over all ~27,990 diseases and ~10 phenotypes it becomes roughly
280,000 loop iterations, each launching several small tensor operations.

**This is a property of the current lookup structure, not of shortest-path scoring itself.** Sorting
each phenotype's slice by target and using binary search — or storing the table as a sparse matrix —
would make the whole computation a single vectorised operation, in the same cost class as the
model's own scoring step. The constraint is an implementation artifact and should not be cited as a
reason to prefer one scoring design over another.

## Related documents

- [`DISEASE_SCORER_POLICY.md`](DISEASE_SCORER_POLICY.md) — **the authority for scorer policy**: what
  ranks disease candidates, what role SP may play, the evidence, and implementation status
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — the layered design
- [`RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md`](RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md) —
  open findings on candidate discovery, including the path-gating described in §2
- [`TRAINING_PIPELINE_PLAYBOOK.md`](TRAINING_PIPELINE_PLAYBOOK.md) — how `shortest_paths.pt` is built
