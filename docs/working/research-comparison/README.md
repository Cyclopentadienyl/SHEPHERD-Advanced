# Research comparison: measuring, keeping, finding and exporting every result

**Status: being built on `claude/research-comparison`, from `main` `2da28cf` (2026-11-10).** This is the
round's plan and record. The owner authorised it on 2026-11-10 by adopting the reviewer's plan review and
its suggested authorisation reply, both rev 2 of 2026-11-10 (outside this repository, in
`SHEPHERD_research_comparison_review_2da28cf_20261110_rev2_evidence.zip`). They replace the author's first
plan (`PLAN_research_comparison_2da28cf.md`, outside the repository), which was not delivered as written.
The owner's terms:

- one authorisation, built continuously;
- internal checks that pass let the work continue, and only the stop conditions below pause the affected
  part;
- the whole batch goes to one external review at the end;
- the GB10 acceptance, a PR and a merge are handled separately, as before;
- quality comes first: there is no deadline and no widening of scope.

The stages below are a delivery index, not approval gates.

## 1. Goal

**The main goal.** One model is measured by every protocol its actual capabilities support. Every real
result is verifiable, kept, found, compared and exported:

- for different models, data roles, families and environments;
- through the one current pipeline, with no old executor kept for compatibility;
- through shared boundaries that can carry §6's extensions.

**Secondary goals:**

- the deployment's PyG source is identifiable;
- a frozen-backbone disease control is available;
- the defects on the main line's path are fixed;
- after the core, a minimal served resource and latency reading is re-attached to the existing probe (M1).

**Not pass conditions.** An accuracy gain, significance, equal experimental conditions and equal results are
not engineering pass conditions.

**What this round is.** The research-comparison core and the minimum functional interface needed to use it.
A full UI/UX pass comes later. This round does not empty the BACKLOG.

## 2. Natural retirement, and the one-pipeline rule (binding for this round)

- **What a current operation can do decides.** An artefact's own record and its actual bytes decide what the
  current implementation can do with it. Age, a version name, a mode or a directory name is never an extra
  block.
- **One parser may decode known shapes.** It may read known field shapes into one current internal
  representation. That is not licence to emulate an old algorithm. Where a meaning cannot be shown, nothing
  is guessed, no required field is ignored and no fingerprint is rewritten to make it fit.
- **What retires is one operation.** If running an artefact would need the old model construction,
  training, scoring or validation, or a historical environment, that operation retires naturally. Only that
  operation is refused, with its reason. Its other uses that can still be verified (reading, measurement,
  comparison, export) are untouched.
- **Dispatch is allowed; parallel pipelines are not.** Dispatching among current encoders, kernels, data
  adapters or evaluators is allowed. It goes through the one construction, validation, execution and
  provenance boundary. Two whole flows switched by an old version behind one outer function are not one
  pipeline.
- **Keeping an artefact is not freezing a format.** Old artefacts are never rewritten. That is different
  from keeping an old writer or schema forever. Known semantics stay readable. An unknown, required meaning
  may be refused, with a concrete reason.
- **Nothing is deleted or rewritten.** The user's data, models, ledgers and reports are never deleted or
  rewritten. No whole-file deletion is authorised in this round. Cleanup that can wait is left, and other
  work continues.

## 3. The owner's choices (2026-11-10)

- **O1:** the revised round, built in full. Stopping after Stage 7 and giving up export is not taken.
- **O2:** both the common KG link-prediction read-out and the frozen-backbone disease control are built. They
  are research capabilities, not a procedure every study must follow.
- **O3:** one common KG LP protocol. The formal KG test (N1) and a family's common read-out use the one
  evaluator. Their data role, exposure and research question are described apart. The disease control
  goes through the existing disease evaluator; no mode is copied for N1.
- **O4: rules-1 runs are not refused for their version number.**
  - The rule applies in one settings decode, used by preview and by training entry alike.
  - A rules-1 run continues when the current executor can faithfully run its released semantics. A new
    rules version is recorded where one is needed.
  - The old Trainer is not revived. Early stopping, validation-pair and optimizer protections stay.
  - An old operation that truly needs the old algorithm retires naturally, with its evidence. Usable
    operations are kept, and the freeze feature stays.
  - Only an undecidable compatibility, or an unexpected wide capability loss, pauses the affected part.
- **O4a: the freeze control's details.**
  - The setting is `none | graph_side`. Frozen parts run in eval.
  - The recipe's loss coefficients are kept and recorded as they are. A continuation keeps the setting.
  - The downstream initial weights are checked.
  - The CLI and the existing runner come first.
- **O5: a failed `pull` stops by default.**
  - A non-interactive fallback happens only when explicitly allowed.
  - The existing `*_SPEC` version and source options stay.
  - The installed wheels and native files are checked against what was meant to be installed.
  - A source receipt is kept. There is no version gate.
- **O6: Models › Results and one shared query service, with a thin CLI export.**
  - Five is only the UI's side-by-side cap; it never limits a query or an export.
  - Readable ledger-only records are included.
  - Learning settings are listed whether or not Stage 9 exists.
  - The CLI can join several authorised workspaces.
  - No statistical inference and no ordering by a metric for now.
- **O7:** a data version is created by CLI import. Roles are recorded as they are, train, val or test. A
  version is never refused for not being test.
- **O8:** Models may show results with their exposure. Automatic checkpoint selection still never reads test
  results, and no result is hidden to manage a researcher's choice.
- **O9: the GB10 acceptance comes after integration and the external review.**
  - It runs in an isolated new clone, and the new probes are required.
  - Small GPU regression cases are prepared for the three existing tasks; no large studies.
  - Stage 2 does not wait for the host.
  - Each of the two deploy routes is recorded as actually verified.

**Added to this round:**

- **(A)** A limited data identity for the existing generated splits. New results on them register in the
  same registry, report store and ledger. Archived reports are never rewritten, and no identity is made up
  for them.
- **(F4)** Every new LP run keeps its actual scores, mapped to its pairs, with counts and digests. Its
  exposure interpretation and sources are kept with the result.
- **(B)** BACKLOG 66: one invalid run or file name must not bring the new query down.
- **(C)** The test gaps of BACKLOG 65 on the code this round touches.
- **(F6)** `type_embedding` can be set, but base validation and the shared task full-view calls pass no
  `node_type`. Node types and relation order are wired through the shared graph input, sampled and full,
  with no second forward pipeline.
- **(D)** After the core: M1's minimal served resource and latency readings, back in the existing probe. A new
  performance platform is out.

## 4. Invariants

1. **Existing artefacts read and register exactly as before.** This covers kept reports, schema 3 and 4
   ledgers, schema-1 registry versions and existing checkpoints.
   - `build_record` output, store-scan keys and manifest digests stay byte-identical over the whole round
     (golden P0-1).
   - Read-side views change only by named additions, each in the commit that adds it.
2. **Refusals stay where a run cannot be valid.** A model that lacks the component a protocol actually needs
   is refused. So is data that does not match its own fingerprint, and so are changed bytes of an
   already-registered run's report. Different conditions, data roles, families, environments, exposure or
   results never refuse and never hide a result.
3. **The ledger stays schema 4.** The record's own kind discriminates. Its writer rules do not change. Key
   fields keep their names.
4. **One trainer, one ledger writer, one reader per record kind, one report store per data version, and one
   evaluator per protocol.**
5. **Different tasks' metrics are never one scale.** They may be queried and exported together, as raw values
   with explicit protocol, metric, unit, role and semantics. Nothing is ordered by a metric this round.
6. **Automatic checkpoint selection never reads test results** (`selection-ignores-test-results`).
7. **One operator check, D8,** governs every operator operation. No view or export carries a case id or a
   server path (D7).
8. **Model and data identity is by content and dependency reference; paths are locations only.** Moving or
   repackaging never reassigns a measurement run.
9. **A value that could not be computed is a status, never a number.** What is not observed reads unknown.

## 5. The stages (a delivery index)

Each stage ends green, with a commit and push to `claude/research-comparison`. Commits are grouped by reason.
The full suite runs at the checkpoints, not on every small commit.

### Checkpoint A: baselines, the shared LP computation, capabilities, deploy sources

**Stage 1: safety net, the shared LP core, node types (F6), the checked split rebuild.**

- **Goldens on the unchanged tree:**
  - P0-1: kept reports' records, store keys and views;
  - P0-2: protocol judgments and their sentences;
  - P0-3: base validation's output and pairs;
  - P0-4: the registry's schema-1 reads and the supplied importer's `cohort.json` bytes;
  - P0-5: a behavioural pin that a base's supervision and message passing stay on the train part and its
    validation pairs on the val part.
- **One LP core in `src/models/bundle`, beside `corrupt_tails`:**
  - `fixed_lp_pairs` gives the pairs, counts and pairs digest;
  - `score_lp_pairs` gives the scores and metrics;
  - base validation uses them, byte-identical, with training rules unchanged;
  - the reason is README §3.2: shared math lives in `src.models`.
- **Node types (F6).**
  - The shared graph input carries the true node ids, node types and relation order for the sampled and full
    paths.
  - Every formal caller passes them: base validation, the three finetune training steps, and the shared
    `full_view_vectors` used by validation, measurement and serving.
  - `type_embedding` on works end to end; the reference recipe (off) is unchanged.
- **One checked rebuild of a recorded split** in `src/kg/graph_view.py`, used by training's lineage check
  (its sentence unchanged) and by the new importer.

**Stage 2: deploy route and native build identity (items 70, 71; O5; E1).**

- **On a failed route, `deploy.sh` says so plainly and stops by default:**
  - it names what failed: fetch, checksum or install;
  - an explicit setting allows the build fallback, non-interactively too, decided once at start;
  - the batch scripts and guides are updated;
  - `*_SPEC` stays, and the fallback is tested.
- **The builder installs exactly what it built:**
  - it builds into a fresh directory with no pip cache;
  - it installs the wheels it built by path;
  - it checks each installed distribution's version and native-library digests against the wheel;
  - it writes a path-free source receipt: requested, resolved and installed identity; compiler; flags.
- **The one collector (`src/utils/run_environment.py`):**
  - Native identity: each extension's version, native file count and a stable aggregate digest. The ordering
    and aggregation are specified, and no file names are emitted.
  - Receipt facts, when they match the installed bytes; a mismatch shows its reason.
  - Separate facts for requested, installed, available, selected and observed in effect, each with a fallback
    reason. Anything not observed reads unknown.
  - Named, extensible component facts, so neither "only these four extensions" nor "CUDA is present" becomes
    a record rule.
  - Facts that cannot be read are a status; they never block a measurement.
- **Probe A1** keeps its keys and adds native identity. Record views show it.

**Stage 3: capabilities (E2).**

- **One capability description per model, derived from its validated spec and built components:**
  - the representations it outputs;
  - the heads and their scoring semantics (the LP head's DistMult semantics, never assumed from a name);
  - its node and relation needs;
  - its sampled or full computation needs.
- **Protocols declare** the capabilities and data kinds they need. The three existing rows are re-expressed
  with the same acceptance and sentences (P0-2). Capability judgment governs new runs only, never reading,
  comparing or exporting history.
- **Hop and width rules move to the encoder.** `hops_problem` and similar calls delegate to the encoder's
  declared sampling need. GATv2's heads×output and hops = layers rules stay in its own declaration and
  validation.
- **A test-only capability description** with another width and sampling need proves that no shared layer
  gates on `gatv2` or on layers = hops. An unknown implementation is still refused at build.

**Checkpoint A** is internal. Old validation pairs and math hold in a fixed environment, and reading old
results does not regress.

### Checkpoint B: data version → CLI / runner / Test → report → registration

**Stage 4: data versions (A; E3; E4).**

- **The registry manifest gains schema 2 with an explicit kind, through the same parser and validator:**
  - schema 1 reads as `supplied`, and supplied imports keep writing schema 1 byte-identically (P0-4);
  - the kind vocabulary is shared with the measurement manifest's `cohort_kind`.
- **A `kg_edges` version** has a role, recorded as it is: `train`, `val` or `test`. It binds:
  - the graph digests and the dataset record;
  - the view rule and digest;
  - the split rule, seed, digest and part counts;
  - its message-passing graph (the split's train part);
  - the pair rule, its batch boundaries and its seed;
  - a stored pairs file bound by the digest of the bytes parsed (M2.1).
- **Why pairs are stored.** torch's generator stream is not promised across releases, so rebuilding pairs
  could refuse the same data in another environment.
- **The importer** is the existing preview/commit flow with a `kg_edges` source:
  `scripts/import_cohort.py --kind kg_edges --from-checkpoint`. It takes the view and split from the named
  checkpoint's record, rebuilds and checks the split, and builds the pairs with `fixed_lp_pairs`.
- **Generated splits (A, limited).** A generated split's data identity has three parts: the workspace's
  split artefact digests, the generation definition and the role. New measurements on it can be kept and
  registered under that identity in the same registry, report store and ledger. They are never presented as
  `supplied` or as test. Archived reports keep exactly what they recorded.
- **Content identity and references (E3).**
  - New data and results carry content identities and dependency references: KG, naming and mapping, view,
    split, pairs, source, checkpoint, spec, parent, family.
  - A missing dependency is a status. Paths are locations only.
  - Store-root authorisation and content verification stay separate.
- **Source roles (E4).** A version can carry named, digested source roles beyond today's four KG source
  roles. The four are not copied into a permanent allow-list.
- **User-facing words:** "KG edge test set", or the version's role; never "cohort".
- **Older revisions** refuse a schema-2 version with its reason and never misread it.

**Stage 5: the LP measurement (O3; F4).**

- **One protocol, `kg_link_prediction`, with one evaluator:**
  - it accepts any model whose capabilities include the LP head's scoring semantics;
  - `BASE_MESSAGE` retires with its last caller, and its 13 test sites flip;
  - it reports metric schema 3: lp_accuracy (raw > 0), ROC-AUC (average ranks), AP, lp_loss and counts, each
    with its unit and denominator;
  - a metric that cannot be computed is a status;
  - message passing uses the version's train part for every model, with no full-view variant.
- **Refusals before anything is written:**
  - a version or checkpoint whose own record contradicts its bytes;
  - the version's view digest unequal to the model's;
  - a tampered pairs file;
  - a file without a model spec;
  - a protocol and data kind that do not match.
- **Not refusals.** A model of another split of the same view is measured, and its exposure says k of n.
  Measuring across views has no node mapping this round: it is a stated capability gap, not a research rule.
- **Each run's actual outputs are kept (F4).**
  - The actual positive and negative raw scores, keyed to the pairs file's rows, with exclusion states and
    counts.
  - They are streamed in chunks into the run's own artefact directory, with type, digest and row count.
  - The report commits last; a cancelled or failed run leaves its true state.
- **The exposure block** has a definition version and its sources: the checkpoint's model record and the
  version.
  - Its words are scoped to the recorded evidence. A base on its own split reads "held out relative to the
    recorded training and validation graph use". A finetune after full-view message passing reads "after
    full-view exposure". Another split reads k of n. A record that cannot be read reads unknown.
  - It is never a gate.
  - Negatives are rule-generated, never claimed as proven non-relations.
- **No new manifest field is forced,** but provenance is never dropped to avoid one. Counts keep explicit
  units and denominators.

**Stage 6: registration, the ledger and the one reader by data kind.**

- **`build_record` is per kind.** The per-kind rules cover the cohort block, `r10` (a mapping for supplied,
  absent for `kg_edges`), the counts and the generated split's identity. The field-set rule and supplied
  records stay unchanged.
- **Exposure travels with the record.** The exposure block and its definition version go into the ledger
  record, so a ledger-only reader still has it.
- **Mixed-ledger acceptance:**
  - an older reader lists an unknown kind's record as unreadable, record by record;
  - an older writer still appends other runs beside it.
- **One reader, discriminated by kind,** behind the API, Models, the loaded-model route and
  `record_evaluation --show`.

**Stage 7: job runner and Test page; BACKLOG 66.**

- **The R10 evidence choice is made once, by kind,** in the runner's inputs. The Test page's own Start check
  and the probe use it. The CLI refuses `--training-evidence` on a `kg_edges` version.
- **The Test page:**
  - lists versions by kind and role, with an empty state that names the CLI import;
  - offers measurements from the capability judgment, keeping the existing preselection rule;
  - has one Start per protocol;
  - gets updated help text and item 68's R10 row.
- **BACKLOG 66 at the shared list root.** A stable identity is kept apart from the display name, so an
  invalid name is listed, not a 500, and escaping never makes two names collide.

**Checkpoint B** is internal. It covers roles, exposure, the per-run artefacts, and two runs with equal and
with different results. File locks, never overwriting and atomic publication hold.

### Checkpoint C: query and export, settings, the freeze control, rules-1 continuation

**Stage 8: find, compare and export (F3; E3; E5).**

- **One read-only results service** in `src/api/services`.
  - It reads every kept report in every listed version's store, of every kind, and every readable ledger
    record.
  - A report and a ledger record are joined only on a reliable run and report identity. A ledger-only
    record is listed with the fields it has and the state "report not available". Historical merged records
    keep a null `run_id` and are never counted as separate runs.
  - One unreadable item never hides the others.
  - An unknown mode keeps its own mode and metrics.
- **Each row lists, where they can be reliably obtained:**
  - the recorded and effective training settings;
  - the model structure;
  - the parent and family;
  - the data and measurement settings;
  - the environment;
  - its content references.
  - What has no source says so; nothing is back-filled from today's recipe.
- **Identity comes from the server's own lists** and the approved source contexts, never from a path a
  report records.
- **D8:** every Results, Compare and Export handler calls the operator check first.
- **One serializer for CSV and JSON.** The long form has protocol, metric_name, value, unit, role, semantics
  and the identities. The Diagnosis CSV stays byte-identical.
- **Models › Results** shows one table per protocol, newest first. Compare puts up to five results side by
  side, differences-only, with no computed deltas.
- **The thin CLI export** (`scripts/export_results.py`) can join several authorised workspaces or registries
  through the same service. Query and export are never capped.

**Stage 9: the frozen-backbone control and rules-1 continuation (O2; O4; O4a; F1).**

- **The setting.** A learning setting `frozen_parts`, `none | graph_side`, where `graph_side` is the input,
  encoder and LP head.
  - Frozen parts run in eval, and a whole-model `train()` never switches them back.
  - It is recorded in `weight_transfer`: each part trainable or frozen, the forward mode, and the digest of
    the downstream initial weights.
  - Both arms keep the recipe's loss coefficients. The control's LP term is recorded as having no trainable
    parameter.
- **One settings decode, used by preview, resume, extend and branch alike.** It recognises the released
  rules-1 semantics as "no freeze": every graph part was trainable (`scripts/train_model.py:1612-1635`).
  - The new rules version is recorded on new checkpoints. Raw records are never rewritten. The effective
    settings and their sources are recorded.
  - A plain continuation never swaps the freeze setting; changing it is a recorded new run or a branch where
    a branch applies.
  - K5's protections stay: early stopping, the validation pairs, the optimizer groups, the scheduler and AMP.
- **Acceptance:**
  - rules-1 base and the three finetunes, through resume, extend and branch;
  - optimizer groups, scheduler, AMP, early stopping and validation pairs;
  - the new freeze state saved and restored;
  - frozen parameters and buffers unchanged, and the downstream parts learning;
  - both arms measured by the one disease evaluator;
  - each case shown compatible, or concretely incompatible.
- **What retirement can still mean.** An old continuation that would need another algorithm retires
  naturally, with its reason; nothing else does.

**Checkpoint C** is internal. There is no second trainer, ledger or reading flow.

### Close

**Stage 10: probes, M1, documents, the integrated check, the GB10 package, delivery.**

- **New required probes:**
  - native identity and route;
  - the LP protocol on the demo base through `measure_scorer` and the job runner, kept and registered;
  - the refusals;
  - a disease child measured by both of its protocols;
  - a generated-split registration;
  - a results export;
  - small GPU regression cases for disease, gene and patients-like-me.
- **M1 (D):** per-request time and the memory and device readings that can be obtained reliably. Cold start,
  warm-up and steady state are kept apart. GPU timing is synchronised. Absent fields say so. There is no
  performance gate.
- **Documents:** each stage updates what it makes false; this stage closes the record.
- **The integrated check, run once:** the full suite, the import contracts, the diff check and the necessary
  browser check.
- **The fixed-head GB10 acceptance package.**
- **Delivery for the one external review.** Fixes rerun only what they touch.

## 6. Extension boundaries (E1–E6)

An interface earns its place only with a current consumer, defined inputs and outputs and a small
acceptance. There are no empty plugins and no guessed settings for unknown architectures. Each delivery says
what is wired and tested, what has a central boundary but no implementation, and what is still a concrete
gap.

- **E1, acceleration:**
  - requested, installed, available, selected, in effect and fallback are recorded apart;
  - README §3.4's five computation boundaries are kept;
  - there is no permanent package version or hardware allow-list;
  - an algorithmic change to scoring (ANN truncation, quantisation) enters protocol or model semantics, not
    only the environment;
  - source URLs are stripped of credentials.
  - Accepted with test doubles: an extra component's facts flow to records and exports; installed but unused
    never reads as in effect; a fallback has its reason; two binaries of one version are told apart.
- **E2, other GNN architectures:**
  - F6 is fixed;
  - capabilities come from the spec and builder;
  - GATv2's rules stay local;
  - `chunk_size` is that encoder's execution parameter;
  - an LP head's scoring semantics are declared, never assumed from its name.
  - A future HGT still needs its own forward, gradient, save/load, resume and dtype tests.
- **E3, packaging:**
  - content identities and dependency references, with paths as locations;
  - a move acceptance for one data version, a checkpoint with its parent reference, a kept report and a
    ledger: identities and results hold under a new authorised root, and a missing dependency is a status;
  - a full archive importer and exporter, signing and import UI are later;
  - a KG subgraph or remapping is a new derived version with its own mapping and digest.
- **E4, cross-species:**
  - species, namespace, stable external ids, node and relation types and both mappings are kept;
  - no filtering of non-human nodes, relations or sources is added;
  - relation exclusion is decided before the view and carries its own identity;
  - no OrthologGate, no second message pass;
  - accepted on a small cross-species graph through view, split, LP pairs and a typed forward;
  - real HP↔MP mappings and effect studies are later and are stated as not tested.
- **E5, more tasks and data kinds:**
  - a model offers a capability set, with no "one task per model" rule;
  - data adapters say which inputs and labels they provide;
  - disease and MONDO requirements stay out of the shared reading layer;
  - an unknown kind is kept and listed with its reason.
- **E6, proportion:** current GATv2 and the current data and report types go through real consumers. Test
  doubles cover only dispatch to capabilities not implemented. There is no universal plugin manager, no
  untested backend, and no second registry or ledger.

## 7. Acceptance matrix

**Capabilities and the LP protocol**

- LP capability judged for a base and for disease, gene and patients-like-me finetunes; the three task
  protocols unchanged.
- A base and a finetune measured on the same pairs and message-passing graph.
- A model of another split accepted, with correct exposure; real file or record contradictions refused.

**Data roles**

- Train, val and test roles shown as they are.
- A new generated-split run registers.
- No identity made up for an old report.

**Registration and query**

- The same run with the same report is idempotent; the same run with a changed report is refused.
- Different runs are both kept, whether their bytes are equal or their numbers differ.
- Query and export cover these cases: report present or absent, ledger present or absent, a historical
  merged record, an unknown mode, and one unreadable item.
- CSV and JSON keep raw values, semantics, units, settings and environment.
- Several workspaces are read through the one authorisation and reading boundary.

**LP artefacts**

- The scores file, pairs and summary verify each other.
- A cancelled run, a failed second publication or a concurrent write never overwrites a successful
  result.

**Freeze control and rules-1**

- Frozen parameters and buffers unchanged; downstream parts learn.
- The whole-model `train()` does not flip frozen parts.
- Initialisation and training conditions are recorded and checkable.
- rules-1 continuation and new frozen runs give the same answer in preview and execution, shown case by
  case.

**Deploy**

- ARM pull success, an explicit stop, an explicit build fallback, a wrong wheel and a mismatched receipt.
- Software simulation and GB10 evidence are listed apart.

**Extension boundaries**

- E1–E6 as in §6.

## 8. Stop conditions

These pause only the affected part; independent work continues:

- the work would need an unauthorised whole-file deletion, or would rewrite user models, data, ledgers or
  reports;
- whether an artefact can be handled faithfully by the current implementation cannot be determined;
- an unexpected wide capability loss appears;
- a solution would need an unauthorised destructive migration or a new block on results;
- the work would need institutional data, a privacy or clinical decision, or an out-of-scope capability (HGT,
  joint training);
- evidence shows the core cannot be done without dropping data identity, provenance, retention or export.

An individual operation shown to need an old algorithm retires naturally; it never stops the round. A
golden difference stops that commit until its cause is found, not the round.

**Out of scope, with their interfaces kept where §6 says:**

- HGT's formal port;
- an acceleration plugin platform;
- joint and multi-head training;
- a full packaging tool;
- real cross-species data and effect studies;
- institutional formats;
- clinical and privacy decisions;
- cross-graph mapping algorithms;
- the full UI/UX pass;
- a large CI effort;
- TM27's change to existing data fingerprints.

## 9. Record

| Date | Stage | Commits | Note |
|---|---|---|---|
| 2026-11-10 | plan | `5e9a908` | This plan, integrating the reviewer's rev 2 review and the owner's authorisation. Baseline at `2da28cf` plus this commit: the full suite 6081 passed, 9 skipped; the 4 import contracts kept |
| 2026-11-10 | 1 | `edb52ba`..`084737c` | Goldens P0-1–P0-5 on the unchanged tree; strict goldens refuse regeneration. One LP core (`fixed_lp_pairs`, `score_lp_pairs`, `lp_pairs_digest`) with base validation byte-identical. One checked split rebuild. Node types through the shared graph input (F6), reproduced first through `train_model.train` with a typed base; a small cross-species graph through view, split and a typed forward (E4). Full suite: 6136 passed, 9 skipped |
| 2026-11-10 | 2 | `1837a3b`..`0cbe952` | A failed ARM pull stops by default, naming the failure (fetch, checksum, install or check); `PYG_ARM_ON_PULL_FAILURE=build` allows the build. Exact by-path installs with a version and native-digest check. A path-free source receipt in the venv (kept by `uv sync --inexact`). An install that was not verified stops (exits 16, 17). The one collector records named components (requested, installed, available, selected, in effect, fallback reason, version, native identity, source), with "unknown" where not observed; record views and probe A1 show the native identity. Software simulation only: neither route has run on a GB10 since |
| 2026-11-10 | 3 | `be4446d`..`6c7a66d` | Protocols judged by a capability description derived from the validated spec (`src/models/bundle/capabilities.py`), carried by the one listing to the CLI, runner, Test page and probe; P0-2 unchanged. GATv2's width and hop rules stay in its declaration; a test-only double passes the shared checks. An encoder kind this build cannot build is now refused at the training start instead of failing in the training process. Checkpoint A, the full suite: 6289 passed, 9 skipped; 4 import contracts kept |
| 2026-11-10 | 4 | `9772198`..`9aef682` | Registry schema 2 with an explicit kind, read by the same parser; supplied versions still write schema 1 byte-identically. `kg_edges` versions with roles train, val and test, bound to graph, view, split, message-passing graph and a stored `pairs.bin` read only through a digest-checking reader (val pairs equal base validation's). `generated` versions copy the role's samples and the split manifest (a generated test role cannot be imported: the generator writes train and val only). Content references and their checked states (E3); source roles without an allow-list (E4) |
| 2026-11-10 | 5 | `f2bc0b4`..`ada2644` | One protocol, `kg_link_prediction`, with one evaluator: metric schema 3, message passing over the version's train part, refusals before any write, a different split measured with k of n exposure, another view refused as a stated capability gap. Each run keeps its actual scores (`scores.bin`, keyed to pair rows, digested; the report commits last) and an exposure block with its definition version and sources. Task protocols measure registered generated versions through the same evaluators (the gene fixture is copied digest-bound when the workspace has one). The first attempt was interrupted by a container restart; its uncommitted work was kept and completed |
| 2026-11-10 | 6 | `140d919`..`847ad78` | Registration and the one reader by data kind: the ledger stays schema 4 with the writer's rules unchanged; KG and generated records carry their blocks and exposure; supplied records byte-identical. Checked against the `2da28cf` code: its reader lists the new kinds' records unreadable record by record and reads the supplied one; its writer still appends |
| 2026-11-10 | 7 | `5e36e83`..`f1525c2` | The R10 evidence choice made once by kind and used by the runner, the Test page and the probe; the Test page lists data versions by kind and role and offers what the capabilities and the kind allow; BACKLOG 66: a run or file name that is not valid text is listed with its reason, with a stable key apart from the shown name. Checkpoint B, the full suite: 6560 passed, 9 skipped; 4 import contracts kept |
| 2026-11-10 | 8 | `ccd44ce`..`cbd2e3f` | One read-only results service (`src/api/services/results.py`): every kept report of every kind in each listed version's store and every readable ledger record of the listed workspaces, joined only on run and report identity; a ledger-only record reads "report not available"; an unreadable item is its own row with its reason. The one reader's record views moved from the route into `src/api/services/record_views.py`; one CSV cell module (the Diagnosis CSV pinned byte-identical). `scripts/export_results.py` writes the long form (CSV or JSON, uncapped) for workspaces and registries passed explicitly. Models › Results with Compare (five, a UI cap only) and Export, each handler behind the D8 check; the Test page's result panel renders through the same row view |
| 2026-11-10 | 9 | `1ad2142`..`f49ede7` | Training rules 2 with the learning setting `frozen_parts` (`none`, `graph_side`); the released rules-1 entry is pinned unchanged. One settings decode for the preview and the training process: a rules-1 record continues with `frozen_parts: none`, decoded from its recorded weight transfer `all`; a missing, contradictory or unknown setting is refused with its reason. All 12 rules-1 cases (a base and three finetunes through resume, extend and branch) continue, also checked against genuine rules-1 checkpoints written by the `cbd2e3f` code. The frozen-backbone control: the graph side gets no gradient, is in no optimizer group and stays in eval mode through any `train()`; the model record lists each part, the forward mode and the downstream initial weights' SHA-256, shown in Models › Results and the export. CPU only: AMP, its scaler and `torch.compile` with a freeze are not exercised |
| 2026-11-10 | checkpoint C | `f49ede7` + fix | The full suite at `f49ede7`: 6665 passed, 9 skipped, 2 failed. The provenance pin found Stage 9b's in-memory parts digest unlisted; it is listed as an in-memory digest. `test_backend_shutdown`'s test-run case read "cancelled" only when the stop landed during the sleeper's start; its fixture now starts the run as the runner does (SIGINT held, ended by SIGINT on a cancel). Both files pass; 4 import contracts kept |
| 2026-11-10 | 10a, 10b | `4a7096f`..`7e1d09e` | Fourteen new required probes in the demo part (`A6`, `L1`–`L8`; the demo list 52 → 66, the real-data list unchanged at 15), each through its formal entry: the importer, `measure_scorer`, the job runner, the trainer, the fixture derivation and the export. They cover the KG link-prediction protocol kept and registered with its scores and exposure verified, the four refusals before any write, a disease child by both protocols, a generated split, the results export, and short disease, gene and patients-like-me children on the probe's device. The field contract and the closing check's next revision (TA-10) are in `GB10_ACCEPTANCE.md`. S1, M1's minimal served readings, is informational: in no required list and outside the exit status. CPU only: nothing ran on CUDA or the GB10 |
| 2026-11-10 | browser check | `7d7cfae`..`88e165d` | The round's screens checked in the real app at 1850 × 960 and 960 × 1080 (and narrow → wide), on the probe's kept demo flow with a frozen-backbone control added and a copy of genuine rules-1 checkpoints written by the `cbd2e3f` code: Research › Models › Results, Compare and Export; Research › Test by data kind; the Continue drawer for rules-1 and rules-2 parents; Fine-tune unchanged. Each finding was reproduced by a second checker. The round's own were fixed and looked at again in the app: result tables no longer squeezed, two runs with equal report bytes told apart, missing states in their own words, nested settings one row per entry under a sub-heading, numbers whole, the Test page's words for every data kind. All 8 rules-1 continuations preview as continuable, with the freeze listed once as decoded from rules 1. What it left open, mostly older than the round, is BACKLOG 72. The browser check ran on the CPU container, not on the GB10 |
| 2026-11-10 | close | `69a6daa` | The first final run at `97ea38b` had 1 failure: the static M2.1 pin found seven new hash or parse sites in the probe. Three were removed (a second parse of a kept report, a second read of a `cohort.json` and a JSON round trip of an in-memory spec); four are listed as the probe's own evidence or an HTTP answer, there being no product reader of `scores.bin` or of the export. The integrated check at `69a6daa`, run once: the full suite 6727 passed, 9 skipped; the 4 import contracts kept; the diff check from `2da28cf` clean. Delivered for the one external review; the GB10 run (`GB10_ACCEPTANCE.md` §4) follows that review. Nothing in this round ran on CUDA or the GB10 |
