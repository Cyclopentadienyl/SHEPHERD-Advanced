# PLAN — one rule for a case's phenotype list, from import to scoring

**Status: draft for review, revision 2. Nothing here is implemented, and nothing here authorises
implementation by itself.** The owner has adopted the rule, and the reviewer closed this
revision's corrections at `e9708cf`, so N1's scope is settled (§8). Facts about this repository
are cited at `15dfcb5`; the code is unchanged at `e9708cf`.

**Revision 2 (2026-10-07)** follows the review of `f7a5c8e`:
- **the API's own summary moves too** (§4.6, §5, §7). `/diagnose` builds its summary from the
  request, not from the result, so it would still count repeats after the pipeline removed them;
- **per-position fields get an input contract** (§4.5). A confidence list whose length differs
  from the phenotype list is refused before anything is dropped, at the API and in the pipeline;
- **the WebUI's own repeat removal is a choice for the owner**, with its consequences stated
  (§5, §8), instead of being called tidying;
- **smaller corrections:** "100 × A, then B" removes 99 repeats, not 100; the raw limit is a
  list length, not a request-size guard; a normalised version's sample bytes keep their digest
  when nothing changed; `measure_served_pipeline` is said not to check, without relying on the
  generator; the old `ValidationResult` is removed, not repointed;
- **the owner's decisions of 2026-10-07 are recorded** (§8): the rule's principle is adopted,
  and `InputValidator` is to be removed in the scope the reviewer listed.

**Amended 2026-10-07, after the review of `e9708cf`:**
- **decision W is made.** The WebUI stops removing repeats, and a list over the limit gets a
  clear message in the UI (§5, "The WebUI's message");
- the message's design, and N1's acceptance for it (§7), are added;
- N1's scope is settled (§8).

**Why this exists.** A case that lists one phenotype twice is scored differently from the same
case listing it once, and today different entry points treat the repeat differently. The
question arose from R10's case label (`PLAN_PROVENANCE_CONTRACT.md` §5.4), and the owner asked
whether removing repeats is what the original SHEPHERD or current practice does.

**How claims are marked:**
- **[Fact]** — confirmed in this repository's code, in the cited source, or in a cited document;
- **[Inference]** — follows from facts, not observed directly;
- **[Recommendation]** — what this plan proposes;
- **[Owner]** — the owner's to decide.

---

## 1. The research, at the evidence level it supports

This section corrects the summary given in discussion on 2026-10-06. Where that summary said more
than the sources do, the correction is noted.

### 1.1 The original SHEPHERD

Read from `mims-harvard/shepherd` at `e95433a`. The code was read and not run.
- **[Fact] The relevant paths keep repeats.**
  - The dataset maps `positive_phenotypes` to node indices as a list. It drops ids missing
    from its dictionary, and removes no repeats (`shepherd/dataset.py:118`).
  - Preprocessing maps old HPO ids to the node index of their current id
    (`data_prep/preprocess_patients_and_kg.py:146-147`). A profile listing both ids would carry
    that index twice after mapping.
  - The MyGene2 cohort takes a profile's phenotype rows as a list
    (`data_prep/create_mygene2_cohort/preprocess_mygene2.py:71`, written out at `:96`).
  - The patient vector is an attention-weighted sum over positions
    (`shepherd/task_heads/patient_nca.py:63-66`).
- **[Fact] No statement was found that gives repetition a meaning** — frequency, severity or
  confidence — in that code.
- **Not claimed:** whether keeping repeats was an oversight or a choice. The sources do not say.
- **Correction.** The summary said a repeat receives "double attention weight". The accurate
  statement: a repeated index enters the attention at each of its positions, so a repeat can
  change the weight distribution, the patient vector and the ranking.

### 1.2 GA4GH Phenopacket and phenopacket-tools

- **[Fact] The schema gives each `PhenotypicFeature` explicit fields:** `type` (1..1),
  `excluded`, `severity`, `onset` and `resolution` (each 0..1), and `modifiers` and `evidence`
  (each 0..*) (https://phenopacket-schema.readthedocs.io/en/2.0.0/phenotype.html).
- **[Fact] The schema does not state that a term may appear only once in a phenopacket.**
  - **Correction.** The summary said Phenopacket lists each feature once. That claim is
    withdrawn.
  - What the schema does support is narrower: observation details have explicit fields, rather
    than being carried by repetition.
- **[Fact] phenopacket-tools is a validation tool, not the schema** (Danis et al. 2023,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10191354/). It has:
  - an ancestry validator, "The phenopacket must not contain both term and its ancestor",
    except an observed ancestor with an excluded child;
  - an obsolete-id check.

  No duplicate-term validator is described in that paper. Neither rule is a requirement on this
  project. The ancestry rule is a different policy from removing repeats, and §4 does not adopt
  it.

### 1.3 Other tools

- **[Fact] LIRICAL computes a likelihood ratio for each observed term**
  (https://lirical.readthedocs.io/en/latest/explanations.html). Its documentation does not say
  how a repeated term is treated, so no claim is made about it.

### 1.4 This repository's training data

- **[Fact] The current generator produces no repeated id within a sample.**
  - Each disease profile is a sorted list of distinct phenotype indices
    (`src/kg/sample_generator.py:597-620, 647`).
  - Each sample draws from that list without replacement (`rng.sample`, `:709`), in draw order.
- **Not claimed:** anything about checkpoints trained on workspaces from earlier generators, or
  about files written outside this generator. Decision 2 of the contract retires old
  artifacts in any case.

### 1.5 Who calls `InputValidator`

- **Production: no caller.**
  - No module in `src/api`, `src/webui` or `src/inference/pipeline.py`, and no serving or
    measurement script, constructs it.
  - It is exported from `src/inference/__init__.py:40-56`.
  - `InputValidatorProtocol` names it as the implementation (`src/core/protocols.py:1132-1136`).
- **Tests:** `tests/unit/test_inference.py:148-250` and `:410-422`.
- **Tools:** `scripts/run_local_tests.py:155-157` imports it, and `:207-227, 407` runs a smoke
  test of the factory.
- **Its `ValidationResult` is exported too** (`src/inference/__init__.py:43, 55`), and
  `tests/unit/test_inference.py:26` imports it from there. The pipeline has its own
  `ValidationResult` (`src/inference/pipeline.py:208`).

## 2. What each entry point does today

**[Fact]** for every row. "Repeats" means the same phenotype listed more than once in one case.

| Entry point | Parsing and polarity | Id resolution and mapping | Unknown ids | Repeats | Order | Count checks | What reaches the scorer |
|---|---|---|---|---|---|---|---|
| **A. Sample generator** (`sample_generator.py:550-650, 690-715`) | Reads KG edges. Positive only, by construction | Already graph indices | Not possible: drawn from KG nodes | None: drawn without replacement from distinct indices | Draw order | `retained_phenotype_count` caps at `max_phenotypes` and the profile's size (`:63-80`), a property of generation | Writes `phenotype_ids` as integer indices |
| **B. Cohort import** | **Not implemented.** Planned as D6 (`PLAN_TEST_RESULTS.md` §4, D6). A supplied cohort today is an already-mapped `<split>_samples.json` made outside the repository | — | — | — | — | — | — |
| **C. Training reader** (`scripts/train_model.py:442-468` → `src/kg/data_loader.py:646-652` → collate `:696-712` → `src/training/trainer.py:795-806`) | Integer indices as stored. Positive only (`DiagnosisSample`, `data_loader.py:608-614`) | None needed | Out-of-range ids are **clamped** into range (`trainer.py:795`) | Kept, so each position enters the masked mean (`trainer.py:800-806`) | As stored | None | Every listed position |
| **D. Measurement reader, Mode C** (`src/kg/storage/file_storage.py:60-101` → `src/evaluation/measurement.py:1193-1223, 1285-1310`) | As stored. Positive only | None needed | Out-of-range ids and empty cases are **refused** (`measurement.py:1205-1223`) | Kept; masked mean over positions | As stored | None | Every listed position |
| **E. WebUI** (`src/webui/components/diagnosis_panel.py:95-113, 128-142, 588-606`) | A regex takes `HP` plus seven digits from free text and writes `HP:0000000`. Everything is positive | None: sends strings | Passed on | **Removed by string**, keeping first occurrences (`:106-112`) | First occurrence | None in the UI | Sends the list to `POST /api/v1/diagnose` |
| **F. API** (`src/api/routes/diagnose.py:71-76, 107-115, 133-141, 290-297`) | Strings as received. Positive only (`PatientPhenotypes`, `src/core/types.py:377-391`). A bad format is logged, not refused | None | Passed on | Kept | As received | **1 to 100 items, enforced by the request model**: a 101-item request is refused with 422 before anything runs | `PatientPhenotypes` to `pipeline.run` |
| **G. Direct pipeline** (`DiagnosisPipeline.run`, `src/inference/pipeline.py:1016-1135`) | Strings | Exact match only: `NodeID(HPO, id)` must be a KG node (`:1186-1202`). No alias or obsolete-id resolution | Dropped with a warning (`:1170-1174`), and filtered again at conversion (`:1195-1202`) | Kept | As received | Over `max_phenotypes` (default 100, `:186-187`): a warning, then **truncation to the first 100, before unknowns are dropped and with repeats counted** (`:1163-1170`) | Repeats reach path search (`:1204` onwards), the GNN mean (`pool_patient_embeddings`, `src/inference/scoring.py:135-155`, called at `pipeline.py:1531-1547`) and the SP mean distance (`sp_mean_distances`, `pipeline.py:1474-1489`) |
| **H. Served-pipeline measurement** (`scripts/measure_served_pipeline.py:495-520`) | Maps generated sample indices back to HPO strings | Refuses a non-HPO index | Refused | Inherits A's | Inherits A's | Refuses a count outside the API's limit | Sends to the API (F) |
| **I. `InputValidator`** (unused; `src/inference/input_validator.py`) | A flexible format regex `HP:\d{4,7}` (`:84-87`) | Optional ontology check: an obsolete term is kept with a warning (`:344-348`) | Warned and dropped (`:186-189, 335-342`) | **Removed by string, after truncation** (`:164-171`) | First occurrence | Truncates to `max_phenotypes` first (`:157-164`) | Not called |

**What is shared and what is not:**
- **Shared:** the served GNN mean and Mode C's masked mean are tied by an equivalence test
  (`scoring.py:158-165`); the training masked mean mirrors Mode C's (`scoring.py:167-176`).
- **Implemented separately, with different rules:**
  - **format parsing:** the UI regex, the API's warning-only check, and `InputValidator`'s
    regex;
  - **repeat removal:** the UI, by string, and `InputValidator`, by string and after
    truncation. Nobody removes repeats by graph node;
  - **count limits:** the API refuses, the pipeline truncates, `InputValidator` truncates;
  - **unknown ids:** serving drops them with a warning, measurement refuses, training clamps.

**[Inference] Consequences.**
- The same patient can be scored differently through the UI and through the API, whenever the
  input repeats a term.
- A model trained on generated data (A) has seen no repeated ids. A repeat at serving is an
  input shape its training data did not contain.

## 3. Counts and truncation

**[Fact] Where the limits are:**
- **The raw request size** is enforced at the API, by refusal (`diagnose.py:74`).
- **The pipeline's own limit** truncates (`pipeline.py:1163-1170`). It acts on the raw list,
  before unknown ids are dropped and with repeats counted.

**[Fact] "100 × A, then B", by entry point:**

| Entry point | Today |
|---|---|
| WebUI | Removes repeats → `[A, B]`; both are scored |
| API, called directly | 101 items → 422; nothing is scored. An explicit refusal |
| Direct pipeline call | Truncated to 100 × A, with a warning; **B is dropped**, and the GNN mean equals A alone |
| `InputValidator` (if it were wired) | Truncates first, so B is dropped |

**[Recommendation] Two limits, kept apart:**
- **The list-length limit stays at the network boundary**, and stays a refusal: at most 100
  items in `phenotypes` (`diagnose.py:74`). It bounds the list a request carries, not the model.
  It is not a limit on the HTTP body or on each string's length, and this plan does not claim
  request-size protection.
- **The model's limit is checked after mapping and repeat removal**, and refuses when it is
  exceeded. **Nothing truncates.** A truncated input scores a different patient, and the
  warning that announces it does not reach a reader of the scores.
- **Afterwards** (with decision W, §8):
  - WebUI: the 101 entries reach the API, which refuses them with 422. Nothing is scored, and
    the UI says why, with both numbers (§5);
  - API: still 422 for 101 items, an explicit refusal at the boundary, never a silent loss;
  - direct pipeline: `[A, B]` is scored, with 99 repeats removed. A direct call is not behind the
    network boundary, so its policy is not changed to match it;
  - 101 *distinct* known terms: refused by the API, and by the pipeline when called directly,
    where today the pipeline truncates them.
- **Unknown ids keep their current policies** in this work: dropped with a warning at serving,
  refused at measurement, clamped in training. Changing any of them is a separate behaviour
  change. Training's clamp is noted, not addressed here.

## 4. The rule, exactly

**Status: the principle is adopted by the owner (2026-10-07). The details below are settled with
the implementation scope, once this revision passes review (§8).**

1. **Unit.** The positive phenotypes of one case, within one observation scope.
   - **[Fact] The current formats can express nothing else.** `PatientPhenotypes` carries ids
     and optional confidences (`types.py:383-385`). `DiagnosisSample` carries positive indices
     (`data_loader.py:608-614`). Neither has an excluded flag, a time or an observation context.
   - So within what the pipeline consumes, every listed id is a positive observation of one case
     at one scope.
2. **Key.** The graph node an id maps to, after resolution and mapping.
   - Two source ids that map to one node are one phenotype.
   - **[Fact]** Serving resolves exact ids only (`pipeline.py:1186-1202`), so at serving "one
     node" means "one id". Collisions between different source ids arise at import (D6), where
     obsolete and alternative ids are mapped.
3. **Output order.** First occurrence, in input order. It is stable, and does not depend on how a
   set iterates.
   - **[Fact] The pooled scores do not depend on order, but other output does.**
     - The SP mean distance is a float64 mean of integer distances, so order cannot change
       it.
     - The GNN mean is a float32 mean (`scoring.py:155`), so it is order-independent only up to
       rounding.
     - Path search follows input order: direct paths are added once per source, with no repeat
       removal (`pipeline.py:1228-1232`), and a stable sort keeps the top paths
       (`:1270-1275`).
     - Candidate ties keep insertion order (`:1357`), which the code itself calls input order
       (`:1417-1423`).
     - The path-reasoning fallback score depends on path order (`:1380-1383`).
     - Explanations iterate the input list (`src/reasoning/explanation_generator.py:349`).
   - A fixed first-occurrence order is therefore what makes these parts deterministic, and it
     keeps the order a clinician entered for display.
4. **Not merged:**
   - **different patients.** That would change the case count and every denominator;
   - **ancestor and descendant terms.** phenopacket-tools treats them as redundant (§1.2), but
     removing them is another policy, not this rule;
   - **present and excluded of one id, or observations at different times.** The current
     formats cannot express these. An importer whose source can express them defines the
     observation scope and polarity it accepts, and reports what it cannot express.
     - **These are new rules, proposed here for D6 to adopt.** D6 today has three categories —
       malformed, unmappable term, truth absent (`PLAN_TEST_RESULTS.md` D6) — and none for
       polarity.
     - The rules: an excluded feature is never imported as a positive one, and a present and
       excluded pair for one id is reported, never merged silently;
   - **frequency, severity or time.** If they ever enter the model, they come as explicit fields
     with model semantics, never as repetition.
5. **Per-position fields follow.** The rule returns the positions it kept.
   - **Positions are counted in the request, not in the mapped list.** Today mapping drops
     unknown ids without recording which positions survived (`pipeline.py:1170-1174,
     1198-1201`). So the mapping step also returns the request positions it kept, and the rule
     works on those.
   - Any per-position field is reduced with the same positions. Today the only one is
     `phenotype_confidences`, which the API accepts (`diagnose.py:77-80, 133-141`) and nothing in
     scoring reads.
   - **The input contract comes first, because reducing by position needs it.**
     - **[Fact] Today nothing checks the length.** The API's validator checks only that each
       value is in [0, 1] (`diagnose.py:133-141`). `PatientPhenotypes` is a plain dataclass
       (`types.py:376-385`), so a direct pipeline call gets no check at all.
     - **[Inference] Without one, the reduction fails one of two ways.** `phenotypes = [A, A, B]`
       with `confidences = [0.8]` keeps positions `[0, 2]`. Indexing raises, and the API's
       catch-all turns it into a 500 (`diagnose.py:335-340`). Zipping truncates silently.
     - **[Recommendation] The contract:**
       - `None` means not provided;
       - a list, empty or not, must have exactly as many entries as `phenotypes` as received;
       - this is checked before unknown ids are dropped and before repeats are removed;
       - a mismatch is an input error: 422 from the API's request model, and an invalid input
         in `pipeline.run` through its existing invalid-input result (`pipeline.py:1053-1064`).
         Nothing is padded or truncated;
       - repeats are removed by the kept positions. The first occurrence's value is kept, and
         values are never combined.
   - This plan does not decide what confidences mean.
6. **The normalised list is what goes onward.** After the rule, nothing reads the raw request
   list. Three places read it today:
   - the explanations (`src/reasoning/explanation_generator.py:349`);
   - the pipeline's summary (`pipeline.py:1662`, "Based on N input phenotypes");
   - **the API's own summary**, which `/diagnose` builds from the request, not from the result:
     "Found N candidate diagnoses for `len(request.phenotypes)` phenotypes"
     (`diagnose.py:349`).

   Left so, they would still count repeats, unknown ids and entries past the limit. For
   `[A, A, B]` the API would answer "for 3 phenotypes" after the pipeline scored `[A, B]`.

   **[Recommendation] The result says what was used.**
   - `InferenceResult` carries the normalised list it scored, and the counts: received, unknown
     ids dropped, repeats removed.
   - The API's summary takes the used count from the result. Where it also shows the received
     count, it labels it as received.
   - The API does not normalise the request again or infer anything from it.
   - The mock path, which runs when no pipeline is configured (`diagnose.py:279-285`), has no
     result to read. Its summary labels the count as received.

**No complete clinical event model is built in advance.** The rule covers what the formats
express today.

## 5. Where the rule lives, and what it replaces

**[Recommendation]**

**One function and one version constant**, in a new module of `src.kg`.
- **Why there.** `src.kg` is the highest layer every caller can import: `src.kg` itself
  (generator, shared reader), `src.evaluation` (importer, measurement) and `src.inference`
  (pipeline) (`.import-linter.ini`). It is also the layer that defines the sample format
  (`DiagnosisSample`, `file_storage`). `src.utils` would be importable too, but it holds nothing
  about phenotypes.
- **It is pure:** no torch, no I/O.
- **Input:** a sequence of node identities. That is graph indices in files, and KG node ids at
  serving.
- **Output:**
  - the kept identities, in first-occurrence order;
  - the kept positions;
  - the number of repeats removed.
- **`PHENOTYPE_NORMALISATION_VERSION = 1`**, recorded wherever the rule's output is recorded
  (§6).

**Its callers:**

| Caller | Role |
|---|---|
| D6 importer (`PLAN_TEST_RESULTS.md` D6) | Producer: applies the rule after mapping, writes the normalised mapped version, and records the counts |
| Sample generator | Producer: already in normal form. It applies the same function, so the rule has one definition, and records the version |
| Shared reader `read_samples` (`file_storage.py:60-101`) | Consumer: **checks** that a stored case is in normal form, and refuses a file that is not, naming the entry point that rebuilds it. It never changes the data. Training and measurement reach it through contract M2.1 |
| Two readers that do not use it | **Named exceptions, and neither checks.** The frozen oracle, `scripts/evaluate_model.py:203-217`, reads samples itself and is byte-pinned (`tests/unit/test_frozen_evaluator.py`) for historical comparison; it is not changed for uniformity's sake, and it retires with the frozen evaluator. `scripts/measure_served_pipeline.py:487` reads `val_samples.json` with `json.loads` and does not check the file; that its producer writes no repeats is not a check. Its requests go through the API, so N1 applies to them. When it is next touched it moves onto the shared reader and its one check |
| `DiagnosisPipeline.run` | Checks the per-position contract (§4.5), then applies the rule to a request after mapping (`pipeline.py:1195-1202`), with the request positions mapping kept, before the model's count check. Everything after it — scoring, explanations, the summary — reads the normalised list, and the result carries that list and its counts (§4.6). The result's warnings state how many repeats were removed. The API and the UI reach it through here |
| API `/diagnose` | Enforces the list-length limit and, in its request model, the per-position contract. Builds its response from the result: the used count, and the received count only where labelled so (§4.6) |
| WebUI | **Text parsing only** (decision W): extraction and canonical formatting, keeping order and repeats. It presents the API's answers, including the 422 below. Mapping, repeat removal and the count decision stay in the service |

**`InputValidator`, function by function:**

| Function | Disposition |
|---|---|
| Truncation to `max_phenotypes` (`:157-164`) | Removed. It is the defect §3 describes |
| Repeat removal by string, after truncation (`:166-171`) | Superseded by the node-level rule in the pipeline |
| Flexible format regex (`:84-87`) | Not adopted. The API's handling of format is unchanged by this work |
| Unknown or obsolete warning via an ontology (`:316-358`) | Not adopted. Obsolete-id resolution belongs to D6's mapping; doing it at serving is a separate behaviour change |
| Confidence checks (`:361-398`) | Not adopted. Confidences are not scored |
| `patient_id` required | Not needed. The API supplies one |
| `ExtensibleInputValidator`, the factory, dict conversion (`:421-575`) | No caller |
| `ValidationResult` (`:50-60`), exported at `src/inference/__init__.py:43, 55` | **Removed with its export and the test import that uses it** (`tests/unit/test_inference.py:26`). It is **not** repointed to the pipeline's `ValidationResult`: the old one defines `__bool__` as `is_valid` (`input_validator.py:58-60`) and the pipeline's does not (`pipeline.py:208-214`), so a repointed name would behave differently while looking the same |
| **The module as a whole** | **Removed within N1** (the owner's decision, 2026-10-07): `InputValidator`, `ExtensibleInputValidator`, the factory and dict conversion, their exports, the tests that test only them, the smoke test in `run_local_tests.py`, and the protocol's stale "IMPLEMENTED" note. Every reference is confirmed gone. **Not removed:** `pipeline.validate_input` and the API's validation; their tests go on the entry points that run |

**It is not wired in whole.** Wiring the old validator would bring its truncation and its
string-level semantics with it.

**The WebUI's message when the list is too long** (decision W).
- **[Fact] What the user sees today.**
  - `_call_diagnose` turns an HTTP error into "API error: <status> — <raw body>"
    (`src/webui/components/diagnosis_panel.py:149-150`), and `_on_diagnose` shows it
    (`:610-619`).
  - For a 101-item list, the raw body is FastAPI's 422. Its one entry has `type: too_long`,
    `loc: [body, phenotypes]` and `ctx: {max_length: 100, actual_length: 101}`, and its `input`
    echoes all 101 submitted ids. This was checked with the installed pydantic 2.11.10 and
    fastapi 0.136.1, on a standalone model declaring the same field.
- **[Recommendation] The same call path, reading the structured error:**
  - **`_call_diagnose` returns a 422 as its status and `detail` entries**, not as raw text.
  - **For `phenotypes` with `too_long`, both numbers come from the server's error**:
    `ctx.actual_length` and `ctx.max_length`. The UI counts nothing and keeps no copy of the
    limit, so the request model stays the one place the limit is set. `measure_served_pipeline`
    already reads it from the model in the same way (`scripts/measure_served_pipeline.py:446-451`).
  - **The message counts entries, not lines.** The UI accepts several ids to a line, and
    commas and spaces between them. For example: "Recognised 101 HPO phenotype entries
    (repeats included); one submission accepts at most 100. No diagnosis was run, and the list
    was not shortened. Remove repeats or shorten the list, then submit again."
  - **Every other 422 entry is shown as one readable line**: the field and the validator's
    message, never the echoed input. A confidence-length error therefore reads as what it is,
    and never as "too many phenotypes".
  - **Other failures stay distinct.** No server address, unreachable, timeout, 500 and 503 keep
    their current messages. A body that cannot be parsed gets a general failure line. The
    formatting never raises.
- **[Fact] What the UI already does on failure, and keeps doing:**
  - the input box is not among the handler's outputs (`diagnosis_panel.py:950-958`), so the
    user's text stays for editing;
  - the message goes to the results area, which persists, not to a passing notice;
  - the error branch clears the candidates and the results state, and disables both exports
    (`:610-619`). A previous run's result is never shown as this one's.
- **A contract test pins the shape the UI reads**, on the real application route: `loc`,
  `type`, `ctx.max_length` and `ctx.actual_length`. A dependency upgrade that changes the shape
  then fails a test, instead of quietly degrading the message to the general line.
- **Not built:** a preview, live repeat removal, a validation endpoint, or a front-end
  validation system.
- **Considered and not recommended:** raising the API's list-length limit above the model's,
  so that repeats alone never cause a refusal and only more than 100 *distinct* terms do.
  - A list that exceeds 100 only because of repeats is an edge case.
  - The change would cost more than it saves. The pipeline's invalid-input result reaches the
    API as a 200 with no candidates (`pipeline.py:1053-1064`, then `diagnose.py:299-327`), and
    would have to become an HTTP error. `measure_served_pipeline` would have to read the
    model's limit rather than the request model's.

**Nothing else is added:** no validation framework, no registry, no second scoring path.

## 6. Provenance and result versions

**[Recommendation]**, against the provenance contract.
- **The source is kept as it is.** D6's source case set keeps its bytes and digest
  (`PLAN_TEST_RESULTS.md` D6).
- **The normalised samples are an artifact.**
  - The importer's mapped version, and the generator's sample files, are written in normal form
    and then digested.
  - Every digest comes from the same read that parses the file (contract M2.1).
- **The records say what was done.**
  - D6's mapped-version manifest and the split manifest carry `phenotype_normalisation`: the
    rule version, the number of cases affected and the number of repeats removed. The
    generator's are zero by construction.
  - The split manifest field lands with the contract's M3b, whose manifest change rebuilds
    workspaces anyway, so workspaces are rebuilt once, not twice.
- **Producers normalise; readers check.** Nothing normalises silently at run time and then
  records the input as though untouched.
  - The one run-time application is a served request, which has no file. Its result states the
    repeats removed.
  - Its served identity (contract M3c) carries the rule version, as one of the serving settings
    that change scores.
- **Two identities, not one.**
  - **The sample bytes.** If normalisation changes a case, the bytes change, the digest changes,
    and a ledger record keyed on the earlier digest (`sidecar.py:156-161`) does not attach.
  - **If normalisation changes no case,** the bytes and their digest stay the same. A record
    keyed on that digest then describes the same scored input, which is correct for its scores.
  - **The mapped version.** Its identity is the rule version and the manifest, which can differ
    while the sample bytes do not. Whether the ledger's cohort key carries that identity is
    settled with ledger v2 (`PLAN_TEST_RESULTS.md` §5). This plan does not assume that
    re-running normalisation always changes a digest.
- **R10 computes set relations only**, and never edits a scoring input (contract §5.4).

## 7. Acceptance

**Each test goes through the entry point it names.** None restates the rule beside the code and
compares the two.

- **The function itself:**
  - `[A, B]` and `[A, A, B]` give the same kept list, `[A, B]`, with one repeat removed;
  - order is first occurrence, whatever the input's order of repeats.
- **Mapping collisions.** Two source ids that map to one node are kept once by the importer, and
  counted in its manifest.
- **"100 × A, then B":**
  - through the WebUI handler with the API: 422, nothing scored, and the message of §5;
  - through the API: 422, nothing scored;
  - through `pipeline.run`: 101 entries, `[A, B]` is scored, and the warning states 99 repeats
    removed.

  B is never dropped silently.
- **101 distinct known terms** through `pipeline.run`: refused, not truncated.
- **One rule everywhere.** Within the API's list-length limit, for an input with a repeat, the
  WebUI, the API and `pipeline.run` produce the same scores as for the input without it. A
  measurement reading a file with a repeat is refused by the shared reader, naming the producer.
- **What the response says** (through the real API route, not `pipeline.run` alone):
  - `[A, A, B]`: the summary counts 2 used, and a warning states 1 repeat removed;
  - `[A, X, B]` with `X` unknown: the summary counts 2 used, and a warning names `X`;
  - the API's, the WebUI's and the pipeline's accounts of the input agree.
- **The WebUI, through its real handler and the API:**
  - **"100 × A, then B":**
    - the message shows 101 and 100;
    - the pipeline is not called;
    - the input text is kept;
    - there is no new result and no downloadable report;
  - **101 distinct valid ids:** the same refusal and message. Entries are counted, not lines:
    the test puts several ids on a line;
  - **within 100 entries, with repeats:**
    - the list sent keeps its repeats, and only the service removes them;
    - the API's used count and warning are right;
    - the UI does not remove repeats to make a test pass;
  - **corrected and submitted again:** the new result shows, with no stale error and no old
    result;
  - **other failures:**
    - a confidence-length 422 (the UI sends no confidences, so this is tested at the API's
      formatter) is not presented as "too many phenotypes";
    - a body that cannot be parsed, and a connection failure, raise nothing in the UI;
  - **the parser:** the test that pins repeat removal (`tests/unit/test_diagnosis_panel.py:42`)
    changes to pin that order and repeats are kept;
  - **the contract test** on the 422 shape (§5).
- **Per-position contract**, at the API and through `pipeline.run`:
  - `None` and a list of the right length pass;
  - an empty, a short and a long list are refused as input errors, and the API never answers
    500;
  - `[X, A, A, B]` with four confidences, `X` unknown: the surviving confidences are those at
    positions 1 and 3.
- **Nothing else changes:**
  - normalising never reduces the number of cases, and never merges patients;
  - an ancestor and its descendant both stay;
  - an input without repeats, of at most 100 entries, scores exactly as before.
- **Contradictions are not merged.** The importer reports an excluded feature, or a present and
  excluded pair for one id, under D6's rules. It never imports either as a plain positive.
- **Traceability.**
  - The source digest is unchanged.
  - The normalised artifact is digested from its own bytes, and its manifest names the rule
    version and the counts.
  - When normalisation changed a case, a ledger record made on the earlier version does not
    attach to the new one.
- **No cross-environment claim.** The rule's acceptance is about the inputs it produces, not
  about bit-identical scores across environments.

## 8. Decisions and where this lands

**Decision status:**
- **[Owner, decided 2026-10-06] Resume continues the same data in the first version.**
  Fine-tuning on other data is not supported yet. This is recorded in
  `PLAN_PROVENANCE_CONTRACT.md` M2.3 and §8, question 3.
- **[Owner, decided 2026-10-07] The rule's principle is adopted** (§4): repeats are removed by
  graph node after mapping, in first-occurrence order, and nothing truncates. As the reviewer
  advised, the implementation scope is settled only after this revision's two corrections
  (§4.5, §4.6) pass review. Nothing is implemented before that.
- **[Owner, decided 2026-10-07] `InputValidator` is removed**, in the scope §5 lists:
  - removed: the classes, the factory and dict conversion, their exports, the tests that test
    only them, the smoke test and the protocol's stale note;
  - the old `ValidationResult` is removed, not repointed;
  - kept: `pipeline.validate_input` and the API's validation.

  It is done within N1.
- **[Owner, decided 2026-10-07] Decision W: the WebUI stops removing repeats**, and a list over
  the limit gets a clear message in the UI.
  - One place removes repeats, and the count in the warning covers every repeat.
  - The cost, accepted: a pasted list of more than 100 entries that fits only after removing
    repeats is refused by the API with 422, where today it goes through. The message says so
    (§5).
  - The option not taken was to keep the UI's removal. Two places would then remove repeats,
    one by string and one by node, and the warning would count only what the service received.
- **N1's scope is settled.** The reviewer closed this revision's corrections at `e9708cf`, and
  decision W is made. N1 may be implemented and sent to code review, without a further approval
  for the same decisions. N2 keeps its dependencies.

**Behaviour changes, under the adopted rule:**

| Where | Today | Proposed | Visible effect |
|---|---|---|---|
| `pipeline.run`, a repeated id | Weighted in the GNN mean and the SP mean, and in path search | Removed after mapping, with a warning giving the count | Scores change for inputs with repeats only |
| `pipeline.run`, explanations and summary | Read the raw request list | Read the normalised list | Repeats, unknown ids and entries past the limit no longer appear there |
| `pipeline.run` and API, `phenotype_confidences` of the wrong length | Accepted | Refused as an input error, before anything is dropped | Only such requests; nothing scores confidences |
| API summary | Counts the request's items | Counts the phenotypes used, from the result | `[A, A, B]` reads "2 phenotypes" |
| `pipeline.run`, more than 100 entries | Truncated to the first 100, repeats included, with a warning | Repeats removed, then refused if more than 100 distinct remain | "100 × A, then B" scores A and B; 101 distinct terms are refused rather than truncated |
| API, more than 100 items | 422 | Unchanged | — |
| WebUI, parsing | Removes repeated strings | Keeps order and repeats (decision W) | A pasted list of more than 100 entries that fits only after removing repeats is refused (422) |
| WebUI, an API error | Shows the status and the raw response body | A list over the limit gets the message of §5, with both numbers from the server; any other 422 is one readable line per entry; other failures keep their messages | The echoed input no longer appears |
| Training and measurement readers | Accept repeats | Refuse a file not in normal form; the two named exceptions (§5) do not check | Only files with repeats; generated files have none |
| Importer | Does not exist | Normalises after mapping and records the counts | New |
| `InputValidator` | Unused | Removed (decided) | None at run time |

**Where it lands, once the implementation scope is settled:**
- **N1 — the serving part, a change of its own.**
  - The function, and its use in `pipeline.run`: request positions, the per-position contract,
    and the explanations and summary reading the normalised list.
  - The result carrying the list it used and its counts, and the API's response built from it.
  - The count rule.
  - The WebUI: parsing keeps repeats, the 422 message, and its contract test (§5).
  - `InputValidator`'s removal.

  It has its own complete acceptance (§7). It depends on no milestone. It changes served scores
  only for inputs with repeats or more than 100 entries, and refuses confidence lists of the
  wrong length.
- **N2 — the stored part, with the contract.**
  - The reader's check rides on M2.1, which makes `read_samples` the one reader for training
    and measurement.
  - The manifest fields ride on M3b's manifest change.
  - The generator applies the function in the same change.
- **The importer** would apply it from its first version (item 14 phase 1, step 1).

**Not a remedy anywhere here:** a compatibility mode for files with repeats, or a CPU fallback.
A file that fails the check is rebuilt by its producer.
