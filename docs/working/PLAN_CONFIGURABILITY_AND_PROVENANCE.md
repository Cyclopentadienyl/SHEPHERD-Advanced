# Plan — configurability as a requirement, and the provenance gap that undermines it

**Status: complete. Both proposals are implemented and approved** (A: `cf413d4`, B: `e226631`). Revision 3a of the design text.

The findings in §3 describe the state this plan was written against; both are now
addressed in code, and §4's proposals are the record of what was built.

Revisions 1 and 2 were reviewed and the direction accepted both times. Every
change comes from those reviews and is listed in §7 rather than folded in
silently. Revision 3 is document-only: no proposal was reopened and no scope was
added.

## 0. Why this document exists

An institutional statement of intent has been made explicit, and it changes how
two existing pieces of code should be judged:

> This system is **two things at once**. It is a clinical inference model that
> will go into service, and it is a research framework in which investigators
> iterate configurations, compare architectures, and study what shortens training
> without costing accuracy. **The data the framework produces is part of the
> deliverable, not a by-product.** Parameters that help one architecture may be
> neutral or harmful for another, so the pipeline must not be nailed to a single
> "optimal" path. Researchers and the clinical team configure, measure, and decide
> what is fit to deploy.

This is a **design position**, not a feature request, and it is recorded here so
that later work can be checked against it instead of re-deriving it.

**What this document is not.** It proposes no configuration framework, no
experiment-tracking subsystem, no run registry or provenance database, no schema
hierarchy, and no execution tracing. Both proposals below are small and reuse
machinery that already exists. If either grows past what §4 describes, that growth
is the finding.

---

## 1. The position, stated so it can be checked against

Revision 1 wrote these as absolutes. Absolutes are what a design position must
avoid, because they get applied where they do not fit — and P1 as first written
was unfalsifiable. All three are narrowed.

**P1. A knob that plausibly affects behaviour and has a concrete clinical or
research use must not be silently fixed or removed.** Removal or fixing needs a
reviewed reason — **not** proof that no imaginable caller could want it, which is
a bar nothing can clear. A clinical preset may legitimately constrain a knob the
research surface still exposes; those are different envelopes, not a conflict.

**P2. Where a setting affects results, the artifact must record the *effective*
value — what actually applied.** Recording the *requested* value as well is good
provenance where the two can meaningfully diverge, and the pair is often more
informative than either. What is forbidden is the requested value **standing in
for** the observed one. This is the project's existing rule in `BACKLOG.md` D2:
*"the **observed** `torch.compile` execution state — an execution fact, not a
requested config value"*.

**P3. Refusal needs a concrete, reviewable reason, and "the state is describable"
is not by itself a reason to accept it.** A meaningful state **inside the intended
clinical or research envelope** should normally be recorded rather than refused.

The reason must be nameable and checkable. Concrete invalidity, safety
constraints, unsupported semantics and an inability to produce an honest artifact
are examples — so are regulatory constraints, institutional policy, a declared
support envelope, and operational or resource limits. **This is not a closed
list, and it must not be read as one.** Revision 2 wrote exactly four grounds and
asked which of them a refusal belonged to, which replaced an unfalsifiable rule
with an over-specified one: a refusal resting on a legitimate ground outside the
list would have failed the test for the wrong reason.

The question to ask of any refusal is therefore **"what concrete ground does this
stand on, and is that ground still applicable?"** — not "could this have been
described?", and not "which of the listed grounds?"

**P4. Two runs are comparable only if the artifacts can prove they differed in
what the experimenter thinks they differed in.** A configuration comparison in
which the data could silently have changed is not a configuration comparison.

---

## 2. What is already right, established rather than assumed

Checked against the tree, not recalled:

| Surface | State |
|---|---|
| `TrainerConfig` | 24 fields |
| `src/config/training_fields.py` | 35 fields exposed to Runtime Settings |
| Architectures | `hgt`, `gat`, `sage` (`SUPPORTED_CONV_TYPES`), resolved by `resolve_arch_params` with a documented precedence so an HGT checkpoint is never rebuilt as the GAT default |
| AMP, `torch.compile`, `num_neighbors`, `num_negative_samples`, sampling strategy | all configurable |
| Checkpoint `config` key | `Trainer._serialize_config()` stores the **whole** `TrainerConfig` plus the entire model-architecture dataclass, so new architecture fields persist automatically |

**No work in the current stage removed a configuration option.** P1 is not
currently violated at the training layer. Two things below are, in narrower ways.

---

## 3. The two findings

### 3.1 A refusal whose concrete ground can be removed

`src/evaluation/measurement.py::assert_no_autocast` raises if either traversal is
entered inside an `autocast` block. It is called from `run_modes_ab` and
`run_mode_c`.

**The reason it was added is real.** `scripts/measure_scorer.py::build_manifest`
writes `amp_enabled=False` and `amp_dtype=None` as **literals**. Under autocast
every score shifts while the manifest goes on claiming fp32, so the artifact would
describe a run that did not happen — which is P3's fourth ground, *inability to
produce an honest artifact*.

**But that ground is removable, and removing it is cheaper than living with the
refusal.** The state is not merely describable; it is describable **at the point
that matters**, by `torch.is_autocast_enabled(device_type)` and
`torch.get_autocast_dtype(device_type)`, both of which answer on any host
including a CPU-only one — verified by execution. Once the manifest records what
applied, the artifact is honest and **that ground is gone**. No other ground is
claimed or, on inspection, available: an AMP-on measurement is not invalid, not
unsafe, not outside a declared support envelope, and not unsupported semantics —
it is the regime the deployment actually uses.

**The cost is a research question the harness can no longer answer.** *"What is
Mode A's MRR under the AMP setting the deployment actually uses?"* sits squarely
inside the research envelope, and Modes A/B/C now refuse it.

Note the asymmetry inside one work item: `torch_compile_wrapped` is **observed and
recorded**; AMP state is **forbidden**. D2's wording asked for the first
treatment. The inconsistency is the author's.

### 3.2 A checkpoint's provenance metadata cannot identify its training inputs

`scripts/train_model.py:658` sets `trainer.data_fingerprint =
compute_fingerprint(graph_data)`, and `src/training/callbacks.py:315-316` embeds
it in every checkpoint **that callback** writes, when the attribute is present. So
a fingerprint mechanism exists and is wired on the training pipeline's path.

**Not in "every saved checkpoint", which revision 2 claimed.** `Trainer.save_checkpoint`
(`trainer.py:961`) is a separate public writer: it takes `data_fingerprint` as an
explicit argument and **does not read `trainer.data_fingerprint`**, so a caller
who omits the argument gets a checkpoint with no provenance at all. The two
writers also disagree on schema — the direct writer stores `model_state_dict`
where the callback stores `state_dict`, which `trainer.py:996` already
accommodates by accepting both conventions. They are separate paths, and §4.1
says which one this plan covers.

**What it records is the graph's shape, not its content.**
`src/utils/fingerprint.py::compute_fingerprint` returns `node_types`,
`node_counts`, `feature_dims` and `edge_types`. Its signature is
`compute_fingerprint(graph_data, kg_total_nodes=None, kg_total_edges=None)` —
**samples are not a parameter at all**, so no property of the sample files, shape
included, participates in the fingerprint.

**The project already documents this limit — on the measurement side.**
`scripts/measure_scorer.py::artifact_digests`:

> Paths are recorded too, but a path is not an identity — `checkpoints/best.pt`
> names a different file after every improvement — and the structural fingerprint
> is not one either, since two checkpoints trained on the same graph share it.

So the two sides of the project disagree about their own rule:

| Artifact | Raw content digests |
|---|---|
| Measurement manifest (`MeasurementManifest.artifact_digests`) | **yes** — checkpoint, samples, node_features, edge_indices, num_nodes |
| Training checkpoint | **no** — structural fingerprint only |

**The defect, stated precisely.** Revision 1 said two such checkpoints are
"indistinguishable", which is false — their weights and their bytes differ. The
accurate statement is:

> **The checkpoints' provenance metadata cannot identify or distinguish their
> training inputs, so an observer cannot attribute an observed difference to data,
> configuration, randomness, or training behaviour.**

That is what P4 forbids. `verify_fingerprint` cannot close it either: it compares
structure to structure, and structure never saw the samples.

This also touches work already queued. Item 6 asks which checkpoint is
authoritative and item 10 must produce M1–M3 checkpoint evidence; both are harder
to answer with checkpoints that cannot name their training data.

---

## 4. Proposals

Ordered by value. Each states what it does **not** do.

### 4.1 Proposal A — record the training checkpoint's input digests

**Structure and content identity stay separate**, as two sibling checkpoint
fields. Revision 1 offered overloading `compute_fingerprint`'s result as an
option; review rejected it and the author agrees.

| Field | Answers |
|---|---|
| `data_fingerprint` (existing, unchanged) | is this checkpoint *structurally compatible* with the graph in front of me? |
| `training_input_digests` (new) | *which exact inputs* produced this checkpoint? |

**What is hashed: the semantic input roles the run actually consumed** — the
training samples, the validation samples actually used, node features, edge
indices, and `num_nodes` or its equivalent where graph construction consumes it.
**Not** every `*_samples.json` in the directory. Hashing whatever files happen to
be present records the directory, not the run, and would make a checkpoint's
provenance change when an unrelated split was added beside it.

**Change.**

1. Move `file_sha256` and `artifact_digests` from `scripts/measure_scorer.py`
   into `src/utils/fingerprint.py`, which already owns fingerprinting and sits in
   the bottom layer where every caller can reach it. Re-export from
   `measure_scorer` so no call site changes behaviour. The role-keyed shape may
   need a small generalisation, since the measurement caller passes one split and
   the training caller passes two.
2. Attach `training_input_digests` at the `train_model.py:658` wiring point,
   beside `data_fingerprint`, **and explicitly update
   `ModelCheckpoint._save_checkpoint` to copy it** into
   `checkpoint["training_input_digests"]` after the weights-only/full branch,
   next to where it copies the fingerprint today.

   Revision 2 said "let `callbacks.py` embed it the same way", which is wrong:
   the callback does not generalise over trainer attributes, it copies
   `trainer.data_fingerprint` by name (`callbacks.py:315-316`). Setting a new
   attribute on the trainer serialises nothing. This is a required edit, not a
   consequence.
3. **Capture is the deliverable; comparison is separable.** Recording the digest
   map is the primary provenance repair and lands on its own. Extending
   `verify_fingerprint` to compare against a current workspace requires deciding
   how it receives a current path for each semantic role, and how it reports
   missing inputs, legacy checkpoints and relocated workspaces. If those are not
   settled in the same item, the comparison helper is **explicitly deferred** and
   the capture still lands.

**Why this is reuse, not a new mechanism.** `file_sha256` already has three
callers outside its own module (`measure_scorer:88-92`,
`calibrate_mode_a:377,384`, `benchmark_sp_lookup:284`) while living in a script.
Lifting it gives one implementation to callers that already exist — the move is
overdue independently of this plan.

**Which checkpoints carry it — the writer boundary, decided on evidence.**

*In scope:* every checkpoint written by `ModelCheckpoint`, which is the designated
training pipeline's writer. That includes weights-only checkpoints. Revision 1
asked whether those should be excluded; the premise was wrong —
`callbacks.py:315-316` appends the fingerprint **after** the `save_weights_only`
branch closes, so they are already provenance-bearing, and digests follow the same
path.

*Out of scope:* `Trainer.save_checkpoint`. This is the bounded choice, and it is
made on a fact rather than a preference: **no invocation or call site exists in
`src/`, `scripts/` or `tests/`.** The remaining references are declarations or
comments about its schema — the method's own definition at `trainer.py:961`, the
protocol declaration at `src/core/protocols.py:1218`, a key-convention comment at
`trainer.py:996`, and a comment in `tests/integration/test_pipeline.py:290`.
Extending a writer nothing invokes would be completeness for its own sake, which
review explicitly warned against.

*The boundary is stated rather than left implicit*, so a future caller finds a
documented limit instead of a silent trap: a checkpoint written through
`Trainer.save_checkpoint` carries provenance **only** if the caller passes it.

Note that "uncalled" is not the same as "dead". The method satisfies a declared
interface (`src/core/protocols.py:1218`), which is a reason for it to exist
without a current invocation. Whether that interface should also require
provenance is a protocol question, not a provenance-capture question, and is
deliberately not this plan's to answer.

**Cost.** One SHA-256 pass over the graph artifacts and the sample files, once per
training run, at startup. Bounded and not on any hot path.

**Known limitation, documented rather than engineered around.** The digest is
taken at a different instant from the load, so a file changed in between would be
recorded incorrectly. **No file locking, transactional loading, content-addressed
store or provenance database is proposed for this item.** The limitation is stated
in the field's documentation.

**Does not do.** No registry, no database, no comparison tooling beyond §4.1 step 3, no
fatal error on mismatch — it stays a warning, because whether a mismatch is
disqualifying is a judgement this code cannot make. A legitimately regenerated
workspace will make every prior checkpoint report a mismatch; that is the correct
signal and it must arrive as information, not as an obstruction.

### 4.2 Proposal B — record the AMP regime instead of forbidding it

**Change.**

1. The manifest **records** `amp_enabled` and `amp_dtype` rather than writing
   literals.
2. `assert_no_autocast` becomes a **consistency** check: the traversal refuses if
   the regime it observes differs from what its manifest recorded.
3. The `MeasurementManifest` docstrings drop "no traversal here opens an autocast
   context, therefore False" and say what the fields now mean.

**Where the observation is taken, corrected.** Revision 1 said "observe in
`build_manifest`", and review found that is **too late for Modes B and C**. In
`scripts/measure_scorer.py`, `encode_full_graph` runs at line 497 while the B and
C manifests are constructed at lines 516 and 530. A caller who wrapped only the
embedding computation would get manifests describing the *later* context and
silently mislabelling those embeddings. The invariant is therefore:

> **The manifest describes the autocast regime at the computation that produced
> the recorded scores, or their source embeddings.**

Two boundaries, not one: Mode A's per-batch forward, and the production
embedding-generation call that Modes B and C consume. The observed value is small
and is passed into the manifest; nothing traces per-operation state.

**Named for what it can see.** If an autocast context opened *inside* a model is
not observable at these boundaries, the field is documented as the
**harness-boundary autocast regime** and claims nothing about nested contexts.
This is the same discipline `torch_compile_wrapped` was renamed under.

**Why this is stronger than the refusal it replaces.** The refusal protects one
value. The consistency check protects every value — including the case the
refusal never covered, where a manifest is built in one context and the
computation ran in another, which is exactly the B/C defect above.

**Does not do.** It adds no autocast **inside** the harness — the traversals still
open no autocast context of their own; a caller's ambient regime is described
rather than rejected. It adds **no AMP CLI or configuration surface**: the harness
honours and records the caller's ambient regime and does not gain a switch of its
own. It does not touch the differential calibration, which already records the
trainer's resolved AMP state, and it does not define any acceptance criterion for
AMP-on runs — that ownership was settled in BACKLOG §3.1.3 and this plan does not
reopen it.

**Boundary check.** Proposal B must not become "make everything observable".
Manifest fields sourced from `loader_config` are already the object the loader
uses, and inventing a probe for each would be the framework this plan refuses to
build.

---

## 5. Deferred, with the reason

**`DIAGNOSIS_SUBGRAPH_HOPS` is recorded as a candidate item, not opened here.**
It was made a constant earlier in this stage with the note *"nothing varies it,
and adding a knob nobody turns is a wider change than removing a duplicated
literal"*. Hop count is a plausible research knob, so that note does not settle
it — but under the **narrowed** P1 it does not immediately reopen it either:
P1 asks for a concrete clinical or research use, and none has been named. It
should therefore acquire an owner and a use case before anything is built, and if
opened it needs a default, bounds, config persistence and manifest provenance —
not merely a field. **It must not be added to satisfy a principle.**

---

## 6. Acceptance

Revision 1 called both proposals "behaviour-preserving for every existing caller".
That was wrong: A changes the checkpoint schema and its serialized bytes, and B
deliberately turns a refused state into an accepted one while adding a new
refusal. The accurate framing:

> **Both proposals preserve the existing default numerical path, while
> intentionally changing artifact schema and accepted execution states.**

**Proposal A:**

- default training numerics unchanged;
- structural-fingerprint semantics unchanged;
- digests cover the inputs actually consumed, and only those;
- a checkpoint trained on one sample file and one trained on a different file are
  **distinguishable from their provenance metadata** with no structural graph
  change — the case §3.2 says is currently unattributable, tested directly;
- digests present in weights-only callback checkpoints;
- explicit, tested behaviour for legacy checkpoints and missing digests;
- a stated boundary between capture and any current-workspace verification.

**Proposal B:**

- default no-autocast numerics unchanged;
- an AMP-on boundary regime is accepted and accurately recorded;
- a recorded regime that disagrees with the traversal's is refused;
- **Mode B/C attribution is correct**: a mutation test computes embeddings under
  one autocast context and constructs the manifest under another, and the
  implementation must not label those embeddings with the later context;
- no claim of observing model-internal nested autocast.

**Both:** the full pre-existing suite continues to pass. Existing tests may be
updated **only** where the intentionally changed artifact schema or execution-state
contract changes their asserted expectation; new acceptance and mutation tests
cover the new behaviour. This permits a legitimate schema or refusal-contract
update while ruling out broad fixture rewrites or the quiet deletion of old
coverage — "every existing test passes unchanged", as revision 2 put it, would
have either forbidden the first or invited the second. `make lint-imports` reports
4 contracts kept.

**Neither proposal needs CUDA**, and no CPU test here may be represented as a
substitute for the institutional CUDA evidence required by items 7a and 5a.

---

## 7. What review changed, and what it corrected

### Revision 2 — from the first review

| Revision 1 said | Corrected to |
|---|---|
| P1: remove a knob only if "nothing could want it" | Unfalsifiable. Now: a plausibly behaviour-affecting knob with a **concrete** use must not be silently fixed or removed; removal needs a reviewed reason. Clinical preset and research surface are different envelopes |
| P2: record observed **instead of** requested | Record the **effective** value; the requested value may also be recorded and is often useful. What is forbidden is requested **standing in for** observed |
| P3: refusal is stronger and needs a stronger reason; describability implies acceptance | Too absolute. Refusal stands on four concrete grounds; describability alone is not an argument for acceptance. §3.1 now names which ground applied and why it is removable |
| Two checkpoints are "indistinguishable" | **False** — weights and bytes differ. The defect is that **provenance metadata cannot identify or distinguish training inputs**, so a difference cannot be attributed |
| "sample files of the same shape" | Unnecessary qualifier: `compute_fingerprint` takes no samples parameter, so **no** property of the sample files participates |
| Digests inside `compute_fingerprint`'s result, or beside it — open question | Settled: **beside**, as a sibling field. Structure and content identity stay separate |
| Hash which files? — open question | Settled: the **semantic roles the run consumed**, not a directory scan |
| Are weights-only checkpoints deliberately not provenance-bearing? — open question | **Premise was wrong.** `data_fingerprint` is appended after the branch, so they already carry it. Digests follow the same path |
| Proposal B observes in `build_manifest` | **Too late for Modes B and C** — `encode_full_graph` at line 497, their manifests at 516 and 530. Observation moves to the computation boundaries, and a mutation test pins the mislabelling case |
| `DIAGNOSIS_SUBGRAPH_HOPS` — should it have been a field? | Recorded as a candidate needing an owner and a use case. Not opened here, and **not to be added to satisfy a principle** |
| "behaviour-preserving for every existing caller" | **False.** Both preserve the default numerical path while intentionally changing artifact schema and accepted execution states |
| Verification and capture treated as one change | Separated. Capture is the primary repair and lands alone; the comparison helper may be explicitly deferred |
| — | Added: the load-versus-hash race is documented as a limitation, with no locking, transactional loading or content-addressed store proposed |

### Revision 3 — from the second review

| Revision 2 said | Corrected to |
|---|---|
| P3: refusal stands on **four** grounds; ask "which of those four?" | **A closed taxonomy is the same failure as an absolute, one layer down.** Regulatory constraints, institutional policy, a declared support envelope and operational limits are equally legitimate grounds. P3 now requires a concrete, reviewable reason **with the list as examples**, and asks "what ground does this stand on, and is it still applicable?" §3.1 is reworded to match, and Proposal B is unaffected — its ground was dishonest metadata and observing at the computation boundary removes it |
| `data_fingerprint` is embedded in "every saved checkpoint" | **Overstated.** It is embedded in every checkpoint **`ModelCheckpoint` writes**. `Trainer.save_checkpoint` (`trainer.py:961`) is a separate public writer that takes the fingerprint as an argument and does not read `trainer.data_fingerprint`. The two also differ in schema — `model_state_dict` versus `state_dict`, which `trainer.py:996` already accommodates |
| Step 2: "let `callbacks.py` embed it the same way" | **Wrong: nothing propagates automatically.** The callback copies `trainer.data_fingerprint` by name, so a new attribute serialises nothing. Step 2 now names the required edit to `ModelCheckpoint._save_checkpoint` explicitly |
| Writer coverage left implicit | **Bounded on evidence, not symmetry.** In scope: the `ModelCheckpoint` path, weights-only included. Out of scope: `Trainer.save_checkpoint`, because it has **no callers anywhere** in `src/`, `scripts/` or `tests/`. The boundary is stated so a future caller meets a documented limit rather than a silent trap. Its having no callers at all is a separate question this plan does not act on |
| "every existing test passes unchanged" | **Ambiguous, and it cuts both ways** — it would either forbid the legitimate schema updates these proposals require, or invite loosening a test until it passes. Now: the full pre-existing suite continues to pass; existing tests may be updated **only** where the intentionally changed schema or execution-state contract changes their asserted expectation; new tests cover new behaviour |
| "§4.1.3" | `§4.1` has numbered steps, not a subsection. Now "§4.1 step 3" |

### Revision 3a — editorial, applied while implementation begins

| Revision 3 said | Corrected to |
|---|---|
| §3.1's title and body still named "P3's four grounds" and "P3's fourth ground" | P3 is no longer a closed or ordered list. The section is now "A refusal whose concrete ground can be removed", and the ground is named rather than numbered: *inability to produce an honest artifact* |
| "the only textual hit is a comment at `trainer.py:996`" | **Literally too strong** — it ignored the method's own declaration and the protocol declaration. Now: no **invocation or call site** exists in `src/`, `scripts/` or `tests/`; the remaining references are declarations or schema comments, listed with citations |
| `Trainer.save_checkpoint` framed as "dead public API" | **Uncalled is not dead.** It satisfies a declared interface at `src/core/protocols.py:1218`, which is a reason to exist without a caller. Whether that interface should require provenance is a protocol question, not a provenance-capture one, and is not this plan's to answer |
