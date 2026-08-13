# The shortest-path (SP) score — what it means, when to use it, how it is computed

**Audience.** §1 and §2 are written for clinicians and require no engineering or mathematical
background. §3 is for developers, bioinformaticians and reviewers who need the exact definition.

**Why this document exists.** The SP score is easy to misread, and one particular misreading is
clinically dangerous: treating a low score as evidence *against* a diagnosis. It is not. This
document exists so that anyone who sees an SP number knows precisely what it does and does not say.

---

> ## Status — read this before anything else
>
> **Today (as of this document's date), the SP score is part of the ranking.** The pipeline computes
> `confidence = 0.7 × GNN + 0.3 × SP` whenever a trained model and a shortest-path table are both
> loaded (`src/inference/pipeline.py:1310`). It therefore influences the order of the candidate list
> a clinician sees.
>
> **This is a deliberate divergence from the reference paper, and it is being changed.** The
> approved direction is to remove SP from the ranking entirely and expose it as a **separate,
> opt-in analysis over the already-ranked candidates**, with its own field or panel. Until that work
> lands, §2's usage guidance describes how to *interpret* the SP column, not a switch that exists.
>
> Sections §1 and §3 describe the score itself and are unaffected by that change.

---

# §1. What the score means

## The short version

> **The SP score does not measure how likely the patient has the disease.**
> **It measures how thoroughly medicine has already written down a connection between the patient's
> symptoms and that disease.**

Everything else in this section follows from that sentence.

## A picture

Imagine every fact medicine has established is written on a card:

- a card for each symptom
- a card for each gene
- a card for each disease

Whenever someone has published a relationship — *this symptom occurs in this disease*, *this gene
causes this disease*, *this symptom is a subtype of that symptom* — a string is tied between the two
cards. Our knowledge graph is that wall of cards and strings: roughly 27,990 disease cards, 19,389
symptom cards, plus genes and other entities.

A patient arrives with several symptoms. For one candidate disease, the SP score asks:

> **Starting from each of the patient's symptom cards, how many strings must I follow to reach this
> disease card?**

| Strings to follow | What that means in practice |
|---|---|
| **1** | Someone has written directly that this disease presents with this symptom. Textbook. |
| **2** | No one wrote it directly, but the symptom is linked to a gene, and that gene to the disease. |
| **3–5** | Connected only through a long chain of separate facts. Very indirect. |
| **Cannot reach within 5** | **Nobody has published anything connecting them, even indirectly.** |

The system does this for every one of the patient's symptoms, takes the average number of strings,
and converts it so that **fewer strings gives a higher score**.

*Illustration:* lens dislocation → *FBN1* → Marfan syndrome is two strings. A textbook association
of this kind scores high.

## The score scale

| Score | Average strings | Plain reading |
|---|---|---|
| 0.50 | 1 | "The textbook says this directly" |
| 0.33 | 2 | "Connected through one gene" |
| 0.25 | 3 | "Connected, but indirectly" |
| 0.17 | 5 | "Barely connected at all" |
| **0.14** | cannot reach | **"No published connection"** |

The scale is compressed at the far end on purpose and by consequence: the gap between 1 string and
2 strings (0.167) is **seven times** the gap between 5 strings and no connection at all (0.024).
Nearly all of the score's discriminating power sits in the 1–3 string range.

## The single most important point

**A low SP score does not mean the patient does not have the disease.**

A low score means only that **medicine has not written this connection down**. That can happen for
two completely different reasons:

- **(a)** the disease genuinely has nothing to do with this patient, **or**
- **(b)** the patient really does have it, and this presentation has simply never been reported.

**The SP score cannot tell these apart. To it, they look identical.**

Rare-disease diagnosis is largely the business of case **(b)**. A patient who has seen many
specialists without an answer very often has a presentation that is not in any textbook — which is
precisely the situation in which the SP score is least informative and most likely to mislead.

A useful one-line summary to keep in mind:

> **The SP score measures how *unsurprising* a diagnosis would be.**
> In rare disease, the correct answer is frequently the surprising one.

## Four things the score cannot tell you

**1. It cannot tell you what kind of connection it found.** Every relationship in the graph is
treated as one string, regardless of type. "Two strings" might be:

- symptom → *is a broader category of* → symptom → *is a broader category of* → symptom
  (**pure classification hierarchy — no biological link to the disease at all**), or
- symptom → *associated with* → gene → *causes* → disease (**a real causal chain**).

Both appear in the score as the number `2`. A high SP score can therefore come from mere
classification proximity rather than from any mechanism. **This is the main source of false
confidence in this number.** To see what the steps actually are, use the reasoning-path evidence,
which preserves the relationship types; the SP score does not.

**2. It cannot tell you how strong the evidence is.** A single 1987 case report and a
10,000-patient cohort study are both "one string", weighted identically.

**3. It cannot distinguish "never studied" from "studied and refuted".** The graph records that a
connection exists, not whether it has held up.

**4. It is dragged down by the patient's least-connected symptom.** The score averages the distances
across all of the patient's symptoms. So a patient with nine textbook features and one unexplained
feature scores lower than a patient with nine textbook features alone — and **the fewer symptoms
recorded, the more heavily a single unexplained one counts**. With three symptoms, one unreachable
symptom removes roughly 30% of the usable score range.

Clinically this cuts both ways. An atypical feature *should* sometimes argue against a diagnosis.
But rare diseases frequently present as "mostly consistent, plus one feature nobody can explain" —
and that unexplained feature is often the one that eventually solves the case. **The SP score
penalises it.**

---

# §2. How to use the score

## It answers a different question from the model's own score

The system produces two independent numbers, and confusing them is the main risk.

| | Question it answers | Clinical analogue |
|---|---|---|
| **GNN score** | *Does this patient's pattern resemble this disease?* | The physician who recognises a gestalt: "I have not seen this exact combination, but it looks like D" |
| **SP score** | *Has anyone connected these dots before?* | The well-read physician: "I recall this combination from the literature" |

Both are legitimate clinical faculties. They are not the same faculty, and averaging them into one
number — which the system currently does — hides which one is speaking.

## Read the two scores together, never the SP score alone

|  | **High SP** (well documented) | **Low SP** (no published link) |
|---|---|---|
| **High GNN** | **Textbook match.** Expected; confirmatory. Low review effort. | **Novel hypothesis.** The model sees a resemblance the literature has not recorded. **Highest review value.** |
| **Low GNN** | Documented disease that does not fit this patient. | **No signal.** This is the great majority of the 27,990 diseases. |

**The bottom-right cell is the reason a low SP score alone is useless.** Most diseases have no
published connection to any given patient, simply because they are unrelated. Selecting on "low SP"
returns an enormous and overwhelmingly wrong set.

## A warning that must not be softened

> **High GNN + low SP means the candidate is *novel*. It does not mean it is *more likely to be
> correct*.**

These are different properties. On average, textbook matches are probably *more* often correct —
they are textbook because they are common and well characterised.

The value of a high-GNN/low-SP candidate is different: **it is the kind of candidate that no
literature-based method could ever surface.** It is where this system contributes something a
literature search cannot. That makes it worth a clinician's attention — not because the odds are
better, but because nobody else will raise it.

Reading "novel" as "more likely" would be a serious error.

## When to consult the SP score

**Consult it when you already have a short list of plausible candidates** — the model's top 10 or
20, or a set of genes or diseases you are actively weighing. In that setting all candidates are
already reasonable, and "which of these is best supported by existing literature" is a genuine
discriminator. This is exactly the setting the reference paper designed it for (§3).

Concretely, it is useful for:

- **separating confirmation from discovery** — which of these high-ranked candidates would a
  literature search also have found, and which would it have missed;
- **triaging review effort** — a high-SP candidate can often be checked against known criteria
  quickly; a low-SP one needs primary reasoning;
- **explaining a ranking to a colleague** — "the model ranks this first, and there is a documented
  two-step link" is a different statement from "the model ranks this first and nothing in the
  literature connects them".

## When **not** to consult it

| Situation | Why |
|---|---|
| **Ranking the full disease universe** | Most candidates are unreachable and receive the same floor score. The number stops discriminating and becomes little more than a reachability flag. |
| **Looking specifically for undocumented or novel presentations** | The score systematically penalises exactly what you are looking for. |
| **As evidence that a diagnosis is wrong** | Absence of a published link is not evidence of absence. See §1. |
| **As an explanation of mechanism** | The score does not know what kind of steps it counted. Use the reasoning paths instead. |

## On the question of rare diseases ranking too low

A question raised during clinical review: *the candidate list tends to put common diseases first;
how can rare diseases and hidden associations be surfaced?*

**The concern is well founded.** Two mechanisms in the current system push well-studied — and
therefore usually more common — diseases upward:

1. **Candidate discovery is currently gated by graph paths**, and the path search stops after a
   fixed number of paths per symptom. Densely connected diseases are reached first and reached more
   often. Dense connection correlates with being well studied.
2. **The SP term in the ranking score** directly rewards documented proximity, at 30% weight.

Both are being addressed: the first by moving candidate discovery to the model (so every disease is
scored, whether or not a path exists), the second by removing SP from the ranking.

**But the natural-seeming fix — "rank by low SP to find the rare ones" — is wrong**, for the reason
given above: low SP selects mostly unrelated diseases. The correct instrument is the 2×2 above:
**among candidates the model already ranks highly, low SP marks those whose high ranking is not
explained by existing literature.** That is a filter applied *after* ranking, never a ranking
criterion itself.

**Honest limitation.** The two mechanisms above are derived from how the code works, not from
measurement — no study has yet been run on this deployment to quantify how strongly ranking favours
well-studied diseases. Such a measurement is planned as part of the candidate-discovery work, using
knowledge-graph connectivity as a proxy for how well studied a disease is.

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
*J. Integr. Bioinform.* 16, 20180069, 2019), not to an experiment within the paper. **No ablation of
SPLSIM appears in the paper**, and the selected value of η is published in the authors' repository
rather than in the article.

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

Pairs absent from the table mean "not reachable within `max_hops`".

Online (`src/inference/pipeline.py:_calculate_sp_score`), per candidate:

```python
UNREACHABLE = max_hops + 1                      # 6 with the default max_hops = 5

total = 0.0
for ph_idx in patient_phenotype_indices:
    d = lookup(ph_idx, target_idx, target_type)  # linear scan of this phenotype's slice
    total += d if d is not None else UNREACHABLE

avg_distance = total / len(patient_phenotype_indices)
return 1.0 / (1.0 + avg_distance)                # → [0.143, 0.5]
```

## Three ways this differs from the paper — all of them material

| | Paper | This implementation | Consequence |
|---|---|---|---|
| **Task** | candidate **gene** ranking, with a short list | **disease** ranking, over whatever the path search admits | The condition the paper's rationale depends on — a short candidate list — does not hold |
| **Transform** | `NORM(mean(−d))`, min–max rescaled **across the candidate set**, giving `[−1, 1]` | `1/(1 + mean(d))`, an **absolute** transform giving `[0.143, 0.5]` | Two effects. (a) Because the range is not `[−1,1]`, `η` is not the effective weight: with `η = 0.7` the GNN term spans 0.7 while the SP term spans only 0.107 — nominally 70/30, structurally closer to 87/13, and the true balance depends on how widely each term actually varies, which has not been measured here. (b) The paper's candidate-relative normalisation cancels a phenotype that is far from *every* candidate; the absolute transform does not, so §1's "least-connected symptom" effect is stronger here than in the paper. |
| **η value** | hyperparameter-searched over `[0.1, 0.9]`, per task; chosen value not published in the article | fixed at `0.7` (`PipelineConfig.eta`) | The project default is a reasonable choice, not a value inherited from the paper. The in-code comment states this correctly. |

A fourth difference is worth noting for interpretation rather than as a defect: the paper's
`d(p, g)` and this implementation's `d(p, d)` are both computed on an undirected, type-erased graph,
so §1's caveat about hop types applies to the paper as well.

## Cost, and why it currently constrains the design

The lookup is a linear scan of one phenotype's slice per candidate per phenotype. With a few hundred
candidates this is unnoticeable. Over all 27,990 diseases and ~10 phenotypes it becomes roughly
280,000 loop iterations, each launching several small tensor operations.

**This is a property of the current lookup structure, not of shortest-path scoring itself.** Sorting
each phenotype's slice by target and using a binary search — or storing the table as a sparse matrix
— would make the whole computation a single vectorised operation, in the same cost class as the
model's own scoring step. The constraint is therefore an implementation artifact and should not be
cited as a reason to prefer one scoring design over another.

## Related documents

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — the layered design and the scoring model
- [`RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md`](RETRIEVAL_AND_CANDIDATE_DISCOVERY_FINDINGS.md) —
  open findings on candidate discovery, including the path-gating described in §2
- [`TRAINING_PIPELINE_PLAYBOOK.md`](TRAINING_PIPELINE_PLAYBOOK.md) — how `shortest_paths.pt` is built
