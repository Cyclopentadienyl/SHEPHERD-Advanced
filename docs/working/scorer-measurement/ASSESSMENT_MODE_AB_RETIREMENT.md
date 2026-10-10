# Retiring measurement Modes A and B — assessment

**Status: inventory and plan only (2026-10-10).** Nothing here changes code, deletes a function,
adds a compatibility layer or a parallel pipeline, or cancels a decision. Every change to an
existing decision is a **proposal** for the owner; changes to the institution's documents are the
institution's. Sections 8 and 9 list what the owner has to decide, and the two questions only the
institution can answer (§8 items 4 and 7), on which the retirement is conditional.

**Base.** The inventory read `f581f9e`'s tree (M2.1 S5) through `git show f581f9e:<path>`. The
S1–S5 corrections from the internal review were then uncommitted in the working tree; they are not
part of the base and were committed afterwards as `6f16373` (§1.1). **All citations below are at
`f581f9e`.**

**How it was made.** Nine independent read-only inventories (the CLI surface, the measurement core,
the differential calibration, downstream consumers, historical gates, the Trainer's guarantees,
the tests, Mode C's acceptance, and M2.1), each with scratch probes and in-memory mutation runs
that changed no tracked file (test runs did write gitignored logs and coverage data); then a
completeness critic that re-checked about 60 citations
(13 corrected here), resolved eight contradictions between the inventories from the code, and
listed what they missed. The first committed draft (`a0b1ff4`) was then checked claim by claim by
independent verifiers; the 43 corrections they confirmed are applied in this revision, and the one
point on which two of them disagreed (whether the wide-cohort truncation test moves, §6.1) was
settled by a further in-memory mutation run. A last consistency and citation check of that
revision found 18 more, mostly places where the conditional framing had not been carried through
and bare line numbers that pointed at the wrong file; they are applied too.

---

## 0. Decided facts and the stance taken

**Decided, and not reopened here:**

- **The data division.** Train is synthetic training cases; Val and model selection is the
  disease-disjoint synthetic split; the research Test is MyGene2; the institutional acceptance
  Test is the hospital's offline cohort, reported apart from MyGene2
  (`docs/working/EVALUATION_COHORTS.md:498-509` for the three evaluation cohorts, `:722` for
  "reported separately, never pooled", `:848-854` for the decided path including the synthetic
  training split). Some proportions, data acquisition and acceptance details remain open
  (`:754-778`).
- **The formal Test is Mode C.** "Producing a test number | `measure_scorer.py`, Mode C"
  (`docs/working/PLAN_TEST_RESULTS.md:289`); "**This is Mode C**, and it is the primary score this
  item records" (`:349-350`). Item 14 depends on items 11, 11i and 15/M1–M3 only
  (`docs/working/BACKLOG.md:669`); neither Mode D nor B-1 is a prerequisite
  (`PLAN_TEST_RESULTS.md:910-911`).
- **The scorer policy.** SP does not enter the main disease ranking; it is post-ranking context
  (`docs/DISEASE_SCORER_POLICY.md:42-44`, `:48-49`). The record was accepted by the deploying
  institution (`:3`) and was taken **before** any B-0 measurement (`:206-209`).

**Not claimed.** The repository does not implement the reference paper's method. Raw cosine is
the interim disease score (`DISEASE_SCORER_POLICY.md:55-57`); the paper's disease scorer is a
squared distance (`PLAN_TEST_RESULTS.md:383-385`); B-1, which would implement the policy in serving,
is not implemented (`DISEASE_SCORER_POLICY.md:117-125`). `run_mode_c`'s "this is what the reference
method does" (`src/evaluation/measurement.py:1237-1240`) is true of its candidate universe only;
R2 narrows the sentence to say that (§2.10).

**Stance.**
- "Compare A, B and C, then choose the Test's mode" is **not** an open requirement: the choice is
  made. An old comparison is not a prerequisite of the new pipeline because a plan once said so.
- A dependency left by an old plan does not by itself show the work is still worth doing.
  "Previously planned", "might be useful later" and "a document requires it" are not reasons on
  their own. A component stays only for a current use: who uses its result, and which decision it
  changes.
- The technical difference A/B measured is real: ranking within a batch's subgraph and ranking
  against every disease are different examinations, even on the same cases. The question is not
  whether the difference exists, but whether the product needs it measured.
- Live training, validation and Mode C correctness are kept. They are not assumed to need Mode A.
- A name containing "legacy" is not evidence a thing can go, and a test calling a thing is not
  evidence it must stay. Several live, unrelated things share the names (§2.9).

---

## 1. Coordination with M2.1 (S1–S10)

### 1.1 The base, and what landed after it

- **Inventory base:** `f581f9e`, S1–S5 as first implemented (`eecd803`, `2faaaa4`, `350cfee`,
  `e4cf737`, `f581f9e`).
- **Uncommitted at inventory time, committed afterwards as `6f16373`:** the S1–S5 corrections
  from the internal review. They touch the shared readers and verifiers (`src/kg/artifacts.py`,
  `src/evaluation/cohort.py`, `src/kg/graph.py`, `src/kg/storage/__init__.py`), a `read_samples`
  docstring in `scripts/audit_split_overlap.py`, the test infrastructure, and docs: status lines,
  the plan's record of S1–S5 as implemented (its "As implemented" notes under S1 and S5),
  `task-scope/README.md` F6, a `PLAN_B04_PRODUCTIONISATION.md` citation, the `BACKLOG.md` row 15
  status and the plan's `docs/README.md` index entry. **None of them is A/B-specific.** Six files this
  assessment cites had lines moved: `src/evaluation/cohort.py` (after `:59`, by +1: an added
  `manifest_section` import, so the cited `:296` and `:343` are `:297` and `:344` at `6f16373`),
  `src/kg/storage/__init__.py` (after `:7`, by +2: the rewritten package summary, so the cited
  `:34-37` is `:36-39`), `scripts/audit_split_overlap.py` (within `:86-94`, by +1: the reworded
  `read_samples` docstring, so the cited `:98` is `:99`), `tests/unit/test_measurement_mode_a.py`
  (after `:837`, by −1: the measurement-roles fixture stopped writing placeholder tensors),
  `tests/unit/test_graph_artifact_binding.py` (after `:174`, by +1) and
  `docs/working/PLAN_PROVENANCE_CONTRACT.md` (from `:10` on). Citations here stay at
  `f581f9e`.

### 1.2 What in the committed S1–S5 serves only A/B

**No A/B-specific compatibility design was added.** S1–S5 touched `scripts/measure_scorer.py`
only to unpack reader results (`:226`, `:579-580`); `build_legacy_mode_a_model` (`:229-264`) is
unchanged. Three items touch A/B:

| Item | What it is | A/B-only? | If A/B retire |
|---|---|---|---|
| `read_samples`' three-field default (`src/kg/storage/file_storage.py:111`, reason `:126-130`; plan `PLAN_PROVENANCE_CONTRACT.md:698-702`) | Measurement reads three fields; `training_fields=True` adds `candidate_disease_ids` and `gene_ids` for training | The **written reason** is A/B ("would change what Modes A and B measure"). The interface is not: Mode C, both verifiers and both audits read nothing beyond the three default fields (Mode C `measurement.py:1287`, `:1290`, `:1314`, `:1316`; verifiers `cohort.py:296`, `:343`; audits `scripts/audit_split_overlap.py:98`, `scripts/audit_generator_fidelity.py:508-509`) | Keep the interface; reword the reason (§8, item 6). Training still needs both fields (S8) |
| The unpack in `load_legacy_mode_a_inputs` (`measure_scorer.py:226`) | Mechanical, as S4 required ("callers only unpack", plan `:952-954`) | Yes | Deleted with the function |
| `map_location="cpu"` for graph tensors (`file_storage.py:86`, `:95-97`) | Applies to every caller | No | No change |

### 1.3 What in the planned S6–S10 is A/B-specific: **deferred** pending this assessment

| Planned text (`PLAN_PROVENANCE_CONTRACT.md` at `f581f9e`) | Content | Status |
|---|---|---|
| S7 `:1022-1032` | Mode A uses the run's single read; delete `load_legacy_mode_a_inputs`; `build_legacy_mode_a_model` takes the checkpoint dict | **A-specific — deferred** |
| S7 `:1033-1036` | rewrite the C-only test to prove it never builds Mode A's model | **A-specific — deferred.** Its end-to-end Mode C assertions are kept either way |
| S7 `:1037-1039`, second sentence | "Modes A and B keep their shared-read regression tests on the executable fixtures" | **A/B-specific — deferred** |
| S0's hand-off to S7 (`:844-846`) | S7 deletes the loader and rewrites the test | A-specific |
| S10 `:1124-1127`, `:1131-1133` (second half) | Mode A/B "not applicable" rows; "until a runnable subject is named (BACKLOG 19.18)" | A/B-specific text; nothing to build |
| BACKLOG 19.18 (`BACKLOG.md:708`) and the clauses that cite it (`:674`, `:681-687`) | The legacy builder's deferred read | Entirely A/B-specific |

Everything else in S6–S10 is mode-agnostic: S6 (serving), S7's one read of manifest, tensors,
samples and checkpoint with verification and one digest map, S8 (training), S9 (path forms and
pins), S10's Mode C measurement row.

### 1.4 Effect of retiring A/B on M2.1's interfaces, and the smallest adjustment

| Interface or text | Effect | Smallest adjustment |
|---|---|---|
| `read_samples(training_fields=False)` | The written reason lapses; no consumer of the default behaves differently with five fields | Keep the flag; reword `file_storage.py:126-130` and plan `:698-702`, `:722-725` to "training passes its two extra fields explicitly" |
| S7 | Only Mode C's checkpoint read (`measure_scorer.py:601`) remains; the Mode A read (`:250`) used the same options | S7 loses its Mode A bullets and A/B regression tests; "one read … the checkpoint" stands |
| S9 (plan `:1057-1058`) | The `file_sha256` re-export pin in the Mode A test file moves with that test (§6.1) | Name its new home |
| S10 pre-flight (`:1108-1115`) | Its `metadata`/`in_channels_dict` expectation supports only Mode A's "not applicable" row | Keep the `weights_only=True` load check; drop the key expectation |
| S10's `--modes C` command (`:1120`) | Depends on whether a `--modes` option remains (§8, item 9) | Edit the command if the option goes |
| Decision 6's statements tied to A (`:1331-1332`, `:1356-1360`) | Superseded by a retirement decision | Add a note; `:1356-1360` forbids deletion only "because its name says legacy", which a no-current-use decision is not |
| `S7`'s test list | S7 names only one caller of `load_legacy_mode_a_inputs`; the others are `test_measurement_mode_a.py:46/51`, `:446/451`, `:483/488`, `:526/530`, `:927/933` and the `world` fixture in `test_measurement_modes_bc.py:53/62` | Holds whether or not A/B retire |

**Ordering.** If retirement lands **before S7**, S7's Mode A work is never built. If it lands
after, S7 builds Mode A plumbing (a new builder signature, A/B regression tests) that retirement
then deletes. **Recommendation: decide this assessment before S7 starts.** S6 and S8 do not depend
on it.

---

## 2. Components: what each is, who uses it, and the proposed disposition

Dispositions: **Retire** (goes with A/B); **Keep** (live, unchanged); **Keep in C form** (shared,
stays, with A-only wording or parameters removed); **Owner** (needs a decision in §8).

### 2.1 The CLI surface (`scripts/measure_scorer.py`)

| Component | Defined | Real callers | Live behaviour it protects | Deleting it affects | Proposed |
|---|---|---|---|---|---|
| `--modes`, default `"A"`, help "must stay the default" | `:438-446` | `main` `:564`; no caller of the script in `src/`, a Makefile, CI or a deploy script | **None.** Its original consumer, the calibration launcher, was deleted in S0 (`PLAN_B03.md:253-255`). Consequence today: a run naming no mode runs A, which raises `KeyError` on every checkpoint the current writer produces (§2.2) | `test_legacy_equivalence.py`'s `measured` fixture runs without `--modes` (`:60-77`) | **Retire** A and B as values; whether the option survives for `C` alone is §8 item 9 |
| `SUPPORTED_MODE_SETS`, `parse_modes` | `:450-493` | `main` `:564` | None: they make inter-mode comparisons attributable (`PLAN_B03.md:248-252`) | `test_legacy_equivalence.py:253-267`, `:337-358` | **Retire** |
| A/B dispatch | `:565`, `:568-573`, `:576-578`, `:582`, `:605-606`, `:615-640` | `main` | None live | — | **Retire**: the `if wants_legacy`/`else` collapses. Mode C's read stays: the import `:574` and the reads `:579-580`, unindented. `wants_production` (`:566`) becomes always true and goes with its `:598` guard, so the production build `:599-612` runs unconditionally |
| `_assert_same_cohort` | `:496-519`, called `:659-660` only when A and C both ran | `main` | None: Mode C's own integrity check is `_assert_cohort_is_intact` inside `run_mode_c` (`measurement.py:1324-1330`) | `test_legacy_equivalence.py:304-334` | **Retire** |
| `--predictions-output` and the legacy-shaped predictions file | flag `:407-411`; written `:683-691` only when A ran | none in code; its own docstring: "nothing diffs it against them now" (`measurement.py:707-708`) | None | `test_legacy_equivalence.py:96-121`, `:174`, `:235`, `:301`; `test_measurement_mode_a.py:548-581` | **Retire** |
| `subgraph_candidate_construction` | `:267-284` | `build_manifest` default `:351`; A/B labels `:628`, `:632` | None for C, which passes "every disease in the knowledge graph" (`:647`) | `test_measurement_mode_a.py:180-217` | **Retire** |
| `--num-workers` (as semantics) | `:415-426` | dataloader of A/B; recorded in every manifest | None for C, which has no dataloader (`measurement.py:1241-1245`; the script builds one only when A runs, `measure_scorer.py:615-620`), so the worker count does not change what it computes (`PLAN_TEST_RESULTS.md:487-488`) | ledger field (`sidecar.py:89`); `build_loader_config` reads it (`measure_scorer.py:300`) for every manifest, Mode C's included | **Retire** the option; the manifest field is §8 item 5 |
| `build_loader_config` | `:287-301` | `main` `:585` (every manifest, through `manifest_for` `:587-593`, which fills `build_manifest` `:355-363`; A/B's dataloader `:618-620`); the Mode C `world` fixture (`test_measurement_modes_bc.py:51`, `:67`) | Live only as the source of the loader and sampler fields in Mode C's manifest; reads `args.num_workers` | Removing `--num-workers` without changing it breaks every C run | **Owner, with §8 item 5**: if the loader and sampler fields go, delete it and record `batch_size` from `args`; if they stay, say what `num_workers` records once the option is gone. Lands in the same step as the option's removal (R2), not R3 |
| Multi-mode output naming and the per-mode console block | `:662`, `:672-677` (`primary = … else "A"`, `_mode{X}.json`), `:696` (the `legacy_metrics` getattr), `:699-701` ("candidate columns per batch") | `main` | None for a C-only run | — | **Retire** the A fallback and the multi-mode naming: one Mode C report and its `_ranks.json` at `--output`; the console prints Mode C's own fields |
| `--batch-size` | `:412-414` | A's candidate universe; C's chunk size (`measurement.py:1284`) | **Live for C**: moves C's numbers at floating-point level (`PLAN_TEST_RESULTS.md:488-490`) | — | **Keep**; its help text (reason given is A's) is rewritten |
| `--seed` and its validation | `:431-437`, `:526`, `:544-546` | every mode; recorded as `python_seed`/`numpy_seed`/`torch_seed` (`:392-394`); the ledger refuses a null seed (`src/evaluation/sidecar.py:280-288`) | **None for C's numbers**: C consumes no randomness (`measurement.py:1241-1250`). The ledger's null-seed refusal and the three seed entries in its semantics list (`sidecar.py:90-92`) exist for A/B's worker streams, negatives and candidate universe (`sidecar.py:83-86`, `:267-271`; `measure_scorer.py:69-75`, `:528-535`) | the ledger's null-seed refusal; seed tests `test_measurement_mode_a.py:679`, `:689`, `:700`, `:708`, `:805` | **Owner** (§8 item 5): retire it with the A/B manifest fields and the refusal, or keep it with a stated Mode C reason (the identity of any future measurement randomness), reviewed at item 14's ledger v2 |
| `_resolve_device` (the CUDA gate) | `:157-188` | every mode | **Live**: the formal Test runs on CUDA | — | **Keep** |
| Production build: `torch.load(weights_only=True)`, `build_shepherd_model`, `encode_full_graph` | `:598-612` | Modes B and C; serving builds through `build_shepherd_model` too (`src/inference/pipeline.py:1014`) | **Live (Mode C)** | — | **Keep** (the `:598` guard and the `"B" in modes` guard `:605-606` go) |
| `build_manifest` | `:304-395` | `main` `:589` (every mode, through `manifest_for` `:587-593`); tests `test_differential_calibration.py:75`, `:104`; `test_measurement_modes_bc.py:52`, `:76`; `test_measurement_mode_a.py:67`, `:202`, `:273`, `:466`, `:499`, `:534`, `:978` | **Live** for C's fields. Its defaults are Mode A's: `mode="A"`, `model_construction="frozen evaluator (legacy)"` (`:307-309`) and A's candidate text (`:350-354`) | callers that rely on the defaults: all seven `test_measurement_mode_a.py` calls and `test_differential_calibration.py:104` take `mode="A"` and the legacy `model_construction`. The `world` fixture (`test_measurement_modes_bc.py:76-85`) overrides only `mode`, `candidate_construction` and `n_samples`, so its Mode C manifests still record `model_construction="frozen evaluator (legacy)"`, unlike the CLI's Mode C (`measure_scorer.py:646-648`) | **Keep in C form**: no A defaults |
| `artifact_digests` | `:100-154` | every manifest; the ledger reads its digests (`sidecar.py:289-294`) | **Live (provenance)**. S7 replaces it with one digest map | — | **Keep** (until S7) |
| Printed ladder lines and the item-7a note | `:703-710`, `:714-716` (the note prints on **every** run, C-only included) | console | None. It ties formal-Test output to an A-only gate in text | — | **Retire** the ladder lines; rewrite the note |
| Module docstring, parser description "(Mode A)", run example naming no mode | `:2-33`, `:399`, `:19-23` | — | — | The example fails on real checkpoints (it runs A) | Rewrite |

### 2.2 `build_legacy_mode_a_model` (`measure_scorer.py:229-264`)

- **Callers:** `main` `:582`; tests `test_measurement_mode_a.py:43`, `:52`, `:253`, `:265`, `:480`,
  `:498`; `test_measurement_modes_bc.py:50`, `:93` (the `world` fixture, which every Mode C test
  there uses); `test_legacy_equivalence.py:287` (replaced by a function that fails if called).
- **Purpose:** rebuilds the model the way the deleted frozen evaluator did (`:230-242`).
- **Live behaviour protected: none.** It indexes `checkpoint["metadata"]` and
  `["in_channels_dict"]` (`:253-254`); the current writer stores neither
  (`src/training/callbacks.py:296-327`); no scanned checkpoint carries them (`BACKLOG.md:44`); only
  the test fixture writes them (`tests/fixtures/synthetic_workspace.py:160-161`). A scratch run on a
  writer-shaped checkpoint: `--modes A` raised `KeyError 'metadata'`, `--modes C` exited 0. The
  differential calibration does not use it (§2.5).
- **Its defect is not the case for retiring Mode A's core** (the trainer-shaped traversal); §3
  makes that case separately. Conversely, the core's possible value would not justify keeping this
  entry point.
- **Proposed: Retire.** No fallback, metadata or schema branch is added to make it run (reviewer's
  rule, plan `:617-618`).

### 2.3 `load_legacy_mode_a_inputs` (`measure_scorer.py:201-226`)

- **What it does:** the same two calls as Mode C's branch (`:579-580`):
  `read_graph_artifacts(...).graph_data` and `read_samples(...).samples`. Only the name and the
  planned lifetime differ.
- **Callers:** `main` `:577`; the test callers listed in §1.4; `test_legacy_equivalence.py:286`.
- **Live behaviour protected: none.**
- **Proposed: Retire** (S7 planned to delete it anyway; with retirement, before S7).

### 2.4 `run_mode_a`, `run_modes_ab` and their A/B-only helpers (`src/evaluation/measurement.py`)

| Component | Defined | Callers | Used by Mode C? | Proposed |
|---|---|---|---|---|
| `run_modes_ab` | `:931-1145` | CLI `:616`; `run_mode_a` `:927`; tests `test_measurement_modes_bc.py:108` (the `ab` fixture), `:143`, `:156`, `:514` | No | **Retire** |
| `run_mode_a` | `:920-928` | `differential.py:439` (the only caller outside tests); tests `test_measurement_mode_a.py:70` (the `_run` helper), `:283`, `:464`, `:497`, `:980`; `test_differential_calibration.py:527` | No | **Retire** with the differential (§2.5) |
| `legacy_ranking` | `:153-188` | `run_modes_ab` `:1056` | No | **Retire** |
| `LEGACY_TRUNCATION_K`, truncated top-k, legacy MRR family | `:245`; `:1057`; `:1110-1120` | A, the differential; the constant is also written on every manifest (`measure_scorer.py:366`) | No (constant only) | **Retire**; the manifest field is §8 item 5 |
| `ModeAResult` (`legacy_metrics`, `legacy_top_k_local`, `to_predictions`) | `:676-719` | `run_modes_ab`; `differential.py:102`, `:221`; CLI `:689` | No | **Retire** |
| `_SamplerEvidence` | `:722-793` | `run_modes_ab` | No (C builds its own dict, `:1337-1349`) | **Retire** |
| `_score_from_full_graph` and B's two clamps | `:839-879` (`:867`, `:873`) | `run_modes_ab` `:1077` | No | **Retire** |
| `to_global_ids` | `:65-102` | `_SamplerEvidence` `:762`, `_score_from_full_graph` `:868`, `run_modes_ab` `:1062`, `differential.py:340` | **No**: C's ids are global and its candidates are `arange(D)` (`:1274`) | **Retire.** Two kinds of text name it, and neither is a dependency: the scorer README's "must not need touching" list (`README.md:146-150`), and D3's malformed-truth contract, which names it as the third of three refusal boundaries (`BACKLOG.md:93`, `:174-181`, `:650`; `src/kg/data_loader.py:976-979`; `tests/unit/test_trainer_truth_invariant.py:9-17`; `tests/unit/test_data_pipeline.py:612-622`). Retiring it re-points that boundary rather than losing it: Mode C's truth refusal is `_assert_ids_in_range` (`measurement.py:1194-1225`, truth check `:1220-1225`, called at `:1286`). §4 has the row |
| `assert_constructions_agree` | `:1148-1191` | CLI `:606`, only when B runs | No | **Retire** |
| A's padding clamp | `:1046` | A only; it copies the Trainer's clamp (`src/training/trainer.py:795`), which stays | No: C validates and refuses (`:1286`, `:1304-1309`) | **Retire** |

**Shared with Mode C, kept in C form** (Mode C needs each; §7 covers their tests):
`canonical_ranking` with `_require_integer_ids` (`:54-62`, `:105-150`) and its tie-rule string
(`measure_scorer.py:368`); `ranks_of_truth` (`:191-229`); `_authoritative` (`:796-810`) and
`METRIC_SCHEMA_VERSION` (`:468`); `MeasurementManifest` (`:500-630`) for C's fields;
`ModeResult` with `to_dict` (the ledger's input) and `to_ranks` (`:633-673`); `AutocastRegime`,
`observe_autocast_regime`, `EncodedGraph`, `assert_manifest_describes_regime` (`:349-462`);
`observe_torch_compile_wrapper` (`:265-346`); `encode_full_graph` (`:813-836`), bound to serving
by `tests/integration/test_pipeline.py:381-429`; `_assert_cohort_is_intact` (`:882-917`, without
the `n_legacy_rows` name and the "In Mode A…" message, `:903-909`); `_assert_ids_in_range`
(`:1194-1225`); `run_mode_c` (`:1228-1353`); and the
served primitives `masked_mean_pool` and `cosine_score_matrix` (`src/inference/scoring.py:159-232`).
`src/evaluation/__init__.py` (docstring only, `:1-24`) describes the package around Mode A and is
rewritten. Several of these kept pieces give Mode A as their written reason; §2.10 lists them, and
only the wording changes. `validate_measurement_seed` (`:480-497`) is not in this list: Mode C
does not need it, and it follows the seed decision (§2.1 `--seed` row, §8 item 5), staying with
`--seed` and the ledger's refusal or retiring with them.

Mode C's `sampler_evidence` (`measurement.py:1337-1349`): `candidate_columns` stays (§7 (a); test
`test_measurement_modes_bc.py:293`). Its `max_subgraph_nodes: {}` and `negative_sampling` entries
exist only for uniformity with `_SamplerEvidence`; they are decided with §8 item 5, and test `:302`
is reclassified to match.

### 2.5 `compare_trainer_against_mode_a` (`src/evaluation/differential.py:373-546`)

- **What it compares.** One frozen batch list through the Trainer's validation pass
  (`trainer._run_evaluation_pass`, `differential.py:409-410`) and through `run_mode_a` on the same
  model (`:439`): per sample, the local top-20 row, the truth (both translated through the same
  gather, so the translation is not tested, `:47-54`) and the reciprocal rank, then the aggregate
  MRR, exactly (`:516-532`).
- **What it establishes:** that two maintained copies of one calculation have not diverged. Its
  own docstring: "It is not a correctness proof of either, and it cannot be" (`:30-37`).
- **What it does not establish:** anything about Mode C, the formal Test, the service scorer or
  SP; it never runs them.
- **What it needs:** `run_mode_a`, the A branch of `run_modes_ab` and its helpers, a full
  `MeasurementManifest`. Not the legacy builder or loader: it runs on `trainer.model`. A scratch run
  on a writer-shaped checkpoint agreed (n=6).
- **Callers:** its test file only. Item 7a's runner "will be new code" and does not exist
  (`BACKLOG.md:666`). `DifferentialResult.to_dict` has no writer.
- **Decisions its result could change today:** none of mode choice, SP policy, Val selection or
  the formal Test. Every gate it holds is about Mode A/B themselves (item 9, 7b, 19.18; §4).
- **The one live engineering protection, and the gap it would leave.** It is today the **only** test
  that sees the *values* of the Trainer's validation ranking, which drive `val_mrr`, early
  stopping, checkpoint selection and serving's automatic checkpoint choice
  (`src/utils/checkpoint_paths.py:45-60`). Reversing the sort (`trainer.py:684`), dropping the
  masked mean (`:801-805`) or dropping the disease normalisation (`:837`) is caught only in this
  file; the characterization tests check shapes, not values
  (`tests/unit/test_trainer_validation_characterization.py:57-77`, `:386-397`). A scratch probe
  that drives `_run_evaluation_pass` directly and computes the expected ranking independently with
  the served primitives caught a negated score and a dropped normalisation without Mode A; it had
  no padded batch, so it did not test the masked mean.
- **Proposed: Retire, after its Trainer-value coverage moves to the Trainer's own tests** (§5,
  §6) **and after the institution answers §8 item 7 on the AMP evidence it carries** (under (b),
  once the Trainer-only CUDA comparison has replaced it). With Mode A gone there is one copy, so
  "two copies agree" has nothing to compare; what remains necessary is that the one copy is
  right, which is a Trainer test.
- **Related:** `Trainer._EvaluationPass.predictions` and `.ground_truths` (`trainer.py:615-633`)
  exist for this comparison; the migrated Trainer test reads them, so they keep a reader. The
  Trainer's behaviour is not changed; only the `_EvaluationPass` docstring (`:623-626`) is
  reworded (§2.10). The `scorer-independence` import contract (`.import-linter.ini:72-94`: the
  reason in the comment `:72-87`, the contract `:88-94`) states the calibration as its reason. The
  migrated G1–G3 Trainer test (§5.1) takes its expected values from the served primitives; that
  comparison is meaningful only while `src.inference.scoring` does not import the training stack,
  so the contract is **kept, with its reason rewritten to name that test** (R2).

### 2.6 Downstream consumers of A/B outputs

- **No live reader consumes anything only A or B produces** — training, validation, serving, the
  WebUI and API, checkpoint selection, `measure_served_pipeline.py` and `probe_deployment.py`
  included.
- **The ledger** (`src/evaluation/sidecar.py:251-319`, via `scripts/record_evaluation.py:100-107`)
  reads `manifest`, `authoritative_metrics`, `n_ranked` and `n_ground_truth_absent`; it never reads
  `legacy_metrics`, predictions or ranks, and accepts A, B and C reports alike. Nothing reads the
  ledger except `record_evaluation.py --show`. No ledger or measurement output is committed.
- **The `_ranks.json` files** have no code reader. The institutional record names "per-sample
  ground-truth ranks" among the implemented B-0 artifacts (`DISEASE_SCORER_POLICY.md:6-8`); Mode C
  writes its own, so they stay with C.
- **Manifest fields that encode A/B** — `legacy_truncation_k`, `legacy_tie_policy`
  (`measure_scorer.py:366-367`), the sampler/loader fields Mode C does not use (`:355-360`,
  `:362-363`) and the three seed fields (`:392-394`), together with the ledger's null-seed refusal
  that exists only for A/B's worker streams (`sidecar.py:267-288`) — are written on every
  manifest, Mode C's included, and are in the ledger's semantics digest (`sidecar.py:76-97`).
  `batch_size` (`:361`) is not one of them: it records Mode C's chunk size
  (`measure_scorer.py:652`, `measurement.py:1284`), moves C's numbers and stays (§2.1).
  Removing them changes new records' keys, but `software_revision` is already in the digest
  (`sidecar.py:104`), so every commit already gives new keys (`PLAN_TEST_RESULTS.md:492-493`). The
  extra effect is on the ledger's tests. The field-partition test
  (`tests/unit/test_evaluation_sidecar.py:159-180`) needs `SEMANTIC_MANIFEST_FIELDS`,
  `NON_SEMANTIC_FIELDS` and `MeasurementManifest` changed together. The `num_workers` and
  `shuffle` rows (`:186-187`) of `test_loader_and_numerical_semantics_change_the_digest`
  (`:183-206`) fail once those fields leave `SEMANTIC_MANIFEST_FIELDS`, because the digest hashes
  only the named list (`sidecar.py:190-193`); its seed rows (`test_evaluation_sidecar.py:188-190`) follow the seed decision, and so do the ledger's seed tests `:267` and `:277` (§8 item 5).
  The rationales that cite Mode A or worker streams (`test_evaluation_sidecar.py:137-143`,
  `:199-201`, `:277-281`; `sidecar.py:60-64`, `:83-86`) are reworded (§2.10). Item 14's grouping
  already excludes the two `legacy_*` fields "when the oracle's surface retires"
  (`PLAN_TEST_RESULTS.md:414-416`). §8 item 5.

### 2.7 The shared test fixture (`tests/fixtures/synthetic_workspace.py`)

- **Stays:** `build_workspace` and `_graph` are used by live tests (`test_pipeline_fails_closed.py`,
  `test_pipeline_reload_availability.py`, `test_checkpoint_log_metrics.py`, `test_model_builder.py`,
  `test_train_model_run.py`). `build_workspace`'s size parameters stay too: after R1 their caller
  is the moved wide-cohort Trainer test (from `test_differential_calibration.py:510`, §6.1), which
  is the only test that catches a wrong top-20 window in the Trainer.
- **Goes with A/B:** `assert_candidate_universe_is_stable` (`:170-228`; callers
  `test_measurement_mode_a.py:79`, `test_differential_calibration.py:89`). The moved Trainer
  tests compute their expected rows from the batches they draw, so they do not need cross-seed
  stability. Also going: the `metadata`/`in_channels_dict` keys written only for the legacy
  builder (`:160-161`). Dropping the keys makes every fixture-based Mode C test run on a
  **writer-shaped** checkpoint, which no repository test does today. (`test_model_builder.py:64`
  overwrites the key, so it does not depend on the fixture writing it.)

### 2.8 Training-side code the retirement does not touch

The Trainer imports nothing from measurement (`trainer.py:70-83`). Its "legacy-shaped" pieces —
the sort (`:684`), the cut at 20 (`:689`), the clamp (`:795`) — are **live Trainer code**, not
Mode A, and stay (`docs/working/scorer-measurement/README.md:95-100`). Retirement changes Trainer
coverage, not Trainer behaviour.

### 2.9 Live names that collide and must not be touched

Evidence-panel "Mode A — Direct Path Evidence" / "Mode B — Analogy-Based Evidence"
(`src/webui/components/diagnosis_panel.py:13`, `:435`, `:445`; `src/inference/pipeline.py:452-453`;
`src/reasoning/evidence_panel.py:15`, `:20`; `src/core/types.py:422`); SP lookup "approach A"
(`PLAN_B04.md`); `checkpoint_legacy_flat` (`src/api/routes/pipeline.py:158`, `:375`); `legacy_flat`
(`src/config/model_types.py:111-177`); `_init_legacy_indexes` (`src/ontology/hierarchy.py:81-100`);
the loader's fallback to its legacy subgraph builder (`src/kg/data_loader.py:348-357`) and
`tests/unit/test_subgraph_equivalence.py`; "legacy" workspaces in `tests/unit/test_evidence_scripts.py`.

### 2.10 A/B wording in kept files (comments and docstrings only; rewritten in R2)

Behaviour unchanged in every row. Listed so R2 rewrites them and nothing kept goes on citing a
retired thing as its reason.

| Location (at `f581f9e`) | Text | After retirement |
|---|---|---|
| `src/kg/data_loader.py:976-979`; `tests/unit/test_trainer_truth_invariant.py:9-17`; `tests/unit/test_data_pipeline.py:612-622` | D3's "three boundaries", the third being `to_global_ids` | Re-point the third boundary to Mode C's `_assert_ids_in_range` (§4, D3 row) |
| `src/kg/storage/__init__.py:34-37` | "One wrapper stays on purpose: `load_legacy_mode_a_inputs` …" | False once the wrapper is deleted (§2.3) |
| `src/kg/storage/file_storage.py:10-12` | "Two of them are migrated here — … Mode C path and the legacy Mode A loader" | One caller remains |
| `src/kg/storage/file_storage.py:126-130` | `read_samples`' three-field default "would change what Modes A and B measure" | Reworded (§1.4, §8 item 6) |
| `src/kg/data_loader.py:771-781` | `DIAGNOSIS_SUBGRAPH_HOPS` justified by Mode A's candidate universe; "two readers" | Reword; one reader if §8 item 5 removes the manifest field |
| `src/evaluation/measurement.py:1-24` | module docstring: "two streams, deliberately … Mode A has to satisfy" | One stream once `legacy_ranking` goes |
| `src/evaluation/measurement.py:113-114`, `:508-510`, `:559-561`, `:609-610` | `canonical_ranking`, `MeasurementManifest`'s `batch_size`, `model_construction` and regime fields justified by A/B | Reword in Mode C's terms |
| `src/evaluation/measurement.py:1237-1240`, `:1243-1244`, `:1257-1260` | `run_mode_c`: "what the reference method does"; "the random draws that make A and B reproducible"; "Mode A clamps" | Narrow the first to the candidate universe (§0); drop the other two |
| `src/evaluation/__init__.py:1-24` | the package described around Mode A | Rewrite (§2.4) |
| `src/inference/scoring.py:168-177` | `masked_mean_pool` mirrors the Trainer "because Mode A's purpose is to reproduce …"; the all-false zero vector is "behaviour to preserve" | Reword: it is the Trainer value test's expected computation (§5.1 G2), so it keeps the Trainer's formula (§4 row) |
| `src/training/trainer.py:623-626` | `predictions`/`ground_truths` exist for "item 1d … Mode A's `legacy_top_k_local`" | Name the migrated Trainer test as their reader |
| `src/evaluation/sidecar.py:60-64`, `:83-86`, `:267-271` | `batch_size`'s key history cites Mode A's subgraph universe; `num_workers` cites "`measure_scorer`'s own --num-workers help"; the null-seed refusal cites worker streams and the candidate universe | Follow §8 item 5: removed with the fields, or reworded with a Mode C reason; the `batch_size` history may stay as history |
| `tests/unit/test_evaluation_sidecar.py:137-143`, `:199-201`, `:277-281` | test rationales citing Mode A's universe and worker streams | Same as the row above |
| `.import-linter.ini:72-87` | the calibration as the contract's reason | Name the migrated Trainer test (§2.5) |
| `scripts/record_evaluation.py:21` | usage example `--report artifacts/mode_a.json` | A Mode C report |
| `docs/working/EVALUATION_COHORTS.md:1097-1103` | `mode` in the key justified by "Mode A and Mode C" | Proposal: mark as history, or reword for Mode C and any future mode |

---

## 3. Does any use still need A or B?

Each candidate reason, asked the same questions: what open question it answers, who uses the
answer and which decision it changes, whether it is decided, research or history, whether it needs
a full external mode or a bounded test, and what is lost without it.

| Candidate reason | Open question it answers | Who uses it / decision changed | Standing | Full mode, or bounded test? | Lost if dropped |
|---|---|---|---|---|---|
| **A→B: encoder scope** (subgraph vs full-graph encoder, same candidates) | How much of the subgraph-vs-full gap comes from the encoder | No named user. It would have informed the choice of Test mode, which is made | **Research only** | A full mode on a checkpoint the current writer produces does not exist (Mode A cannot build one; B requires A) | An attribution number nobody has asked for. Re-derivable from history at a fixed revision if a research item ever names a user and a deliverable |
| **B→C: candidate scope** (subgraph candidates vs every disease, same encoder) | How much comes from the candidate universe | Same | **Research only** | Same: B needs A | Same |
| **The ladder as B-0's report** (README `:9-18`) | "Measure before changing the scorer" | Policy Gate 1 needs a "B-0 measurement report" before **B-1** (`DISEASE_SCORER_POLICY.md:285`, `:289`). The policy names no mode, never mentions A, B, calibration, 7a or item 9, and its revisit conditions name only C and E′ (`:316-318`). It does, however, call B-0 "approved for implementation" (`:289`) and refers to "the agreed B-0 scope" (`:318`). That scope is not recorded in the repository. The repository's only definition of B-0 includes Mode A (B-0.2) and Modes B and C (B-0.3) (README `:30-36`), and item 7b, the institutional measurement of B-0.2/B-0.3, is owned by "both" (`BACKLOG.md:667`) | Gate 1 is **institutional** and stays. Whether its agreed scope includes A/B is **unconfirmed; the institution must say**. Dropping A/B would narrow an approved, co-owned work item, which is a proposal to the institution, not an engineering decision | If the agreed scope includes A/B: full modes, and Mode A cannot build any checkpoint the current writer produces (19.18). Otherwise: none | Unknown until the institution confirms its agreed B-0 scope (§8 item 4) |
| **Differential calibration** (1d / 7a) | Do the Trainer's validation and Mode A's copy agree? | Its only consumers are gates on Mode A itself (§4) | Historical acceptance for a harness being retired | The live part (Trainer ranking values) is a bounded Trainer test (§5) | Nothing live for engineering, **once** the Trainer test exists; the AMP evidence it carries is the next row's question (§8 item 7). Without the Trainer test, the Trainer's ranking values are unprotected |
| **7a's AMP-on leg** (does fp16 reorder validation?) | Whether CUDA autocast reorders the Trainer's validation ranking | The institution's experimenters own the question ("run it both ways if they wish"; the criterion is theirs, `BACKLOG.md:491-505`); no recorded engineering decision consumes the number | The institution's to decide | If wanted, it is the Trainer with `use_amp` on vs off on CUDA — Mode A is fp32 and equal to the Trainer on CPU, so it adds nothing. The CPU Trainer tests of §5 cannot cover it: `use_amp` is off on CPU (`BACKLOG.md:443-446`, `:465-469`) | The per-sample AMP evidence that engineering recorded as already in place for the institution's experimenters (no runner has yet produced it on institutional hardware, §2.5): `DifferentialResult`'s resolved `amp_enabled`/`amp_dtype`, `bit_exact_contract`, the disagreeing-sample count and rate, the affected rows with both rankings, and the MRR gap (`BACKLOG.md:499-505`; `differential.py:79-82`). Whether it is still wanted is the institution's call (§8 item 7) |
| **Mode A as "the control"** (`measure_scorer.py:12-17`; `BACKLOG.md:375` rejected dropping it because it "discards the control the whole ladder is built around") | — | The ladder is no longer an open requirement | **Historical** | — | The reason fails with the ladder |
| **The predictions file**, **legacy MRR** | Diffing against the frozen evaluator | Nothing reads either; the evaluator is deleted | **Historical** | — | Nothing |
| **`--num-workers`/seed as measurement semantics** | A/B's candidate universe depends on worker streams | A/B only | **Historical** for measurement; the same PyTorch property holds in training (§5) | Training's property, if kept, is a data_loader test | Nothing for measurement. Whether the seed stays as run identity is §8 item 5 |

**The core and the entry point are separate questions.** Mode A's core — a traversal shaped like
Trainer validation — had one use left: a second copy against which the Trainer's copy could be
checked. That use ends when the second copy goes; the Trainer's own correctness still needs
checking, and that is done directly (§5). The old builder's defect is not used as evidence here,
and the core's former use is not a reason to keep the old entry point.

**Conclusion of this section.** No current engineering use needs Modes A or B. The comparisons they
made are research questions without a user, a decision or a deliverable. Two questions are not
engineering's to close: whether the institution's agreed B-0 scope still includes A/B (§8 item 4),
and whether it still wants the AMP evidence the differential supplies (§8 item 7). The proposal
below is conditional on those answers. The live protection A/B carry incidentally is listed in
§5–§7 and moves to the real use points before anything is deleted.
**Nothing is proposed for keeping as a "research tool"**; if a research item later names a user
and a deliverable, it starts from Mode C's code and the fixed-revision history.

---

## 4. Historical gates: disposition

Format: what it protected → still needed? → keep, cancel or rewrite → work affected. Changes to an
existing decision are **proposals**; institutional items are not engineering's to cancel.

| Gate or constraint | Where | Originally protected | Still needed? | Proposed | Affects |
|---|---|---|---|---|---|
| **Policy Gate 1: "B-0 measurement report" before B-1** | `DISEASE_SCORER_POLICY.md:285`, `:289` | The live clinical system against an unmeasured scorer change (`:265-266`) | **Yes — institutional** | **Keep.** Ask the institution to confirm "the agreed B-0 scope" (`:318`), which is not in the repository and may include A/B; propose that the report be defined without A/B, keeping the measurement of the deployed scorer (B-0.5 / Mode D) (§8 item 4) | B-1; whether R2 may proceed (§9) |
| Policy: SP code "demoted, not removed … required for B-0's comparison modes" | `:256`, `:262-264` | SP for the modes that consume it (D, E′) — not A/B, which are cosine-only | Yes, institutional | **Keep**; unaffected | SP subsystem |
| Institution: "the legacy measurement path is being kept removable" | `task-scope/README.md:40-42` | Maintainability | Yes | **Keep**; it supports this retirement | — |
| Institution: AMP-on criterion is theirs | `BACKLOG.md:491-505` | Who judges 7a's AMP leg | Yes, as stated | **Keep** the criterion. Retiring the differential withdraws the evidence engineering recorded as in place for it (`:499-505`), so that is put to the institution (§8 item 7) | §8 item 7 |
| **Item 7a** (engineering differential run, institutional hardware) | `BACKLOG.md:666`; `:476-479` | Mode A's agreement with the Trainer under CUDA/AMP | **No** as a certification of a harness being retired. (Its status line is also stale: "blocked on 1d", but 1d is done, `:653`; no checkpoint is designated for it) | **Cancel (proposal), conditional on §8 items 4 and 7**: if the institution keeps A/B in the agreed B-0 scope, 7b keeps its dependency on 7a and 7a stands. Otherwise its AMP-off leg (two copies agree) is superseded by the Trainer-value tests (§5). Its AMP-on leg cannot be covered by those CPU tests, because `use_amp` is off on CPU (`:443-446`, `:465-469`); if the institution wants it, it becomes a Trainer-only item: the same validation pass, `use_amp` on vs off, on CUDA (§8 item 7) | item 9, 7b; the printed note `measure_scorer.py:714-716` |
| **Item 7b** (institutional measurement B-0.2/B-0.3 ← 7a) | `BACKLOG.md:667`; README `:34` ("institutional run inherits B-0.2's acceptance") | A→B→C numbers not used before the harness was accepted | Not for any recorded engineering decision: the institutional Mode C run is item 14's (Test, Mode C), which does not depend on 7a/7b (`BACKLOG.md:669`). Its A/B part also has an unlisted blocker: CLI Mode A cannot build a real checkpoint (19.18). Whether Gate 1 needs it is open (§3, §8 item 4) | **Rewrite (proposal; 7b is co-owned with the institution, owner "both", `:667`, so the institution must agree, together with §8 item 4)**: drop A/B and the 7a dependency only if the institution accepts §8 item 4. 7b's other preconditions do not lapse: item 6 (the institution's choice of authoritative checkpoint), deployment CUDA verification and Mode C peak memory on institutional CUDA (PLAN_B03 row below) carry over to whatever takes 7b's remainder, a Mode C run under item 14 or a B-0.5 measurement. Items 2 and 3 are already done (`:656`, `:657`) | 8b |
| **Item 8b** (← 7b) | `BACKLOG.md:676` | B-0.5's institutional run | Depends on B-0.5, not on A/B | **Rewrite** its dependency on 7b (proposal; co-owned with the institution, owner "both", `:676`; conditional on §8 item 4, like 7b). 8a, 6, exact artifacts and the production-path prerequisites stay | B-0.5 |
| **Item 9** (rename ~70 refs, rewrite the checklist, delete the oracle-only surface; gated on "1d passed review incl. its institutional CUDA run") | `BACKLOG.md:677`; `:804-807`; §5.0 `:843-847`; README `:77-80` | That the harness is never left without an acceptance | **No**: the gate requires a test *of* Mode A before deleting Mode A. Once Mode A goes, there is no harness acceptance to protect; Mode C's acceptance is direct (§7) | **Replace** item 9 by the retirement change (§9); the deletion gate's precondition becomes "the migrated tests are in place" | item 9 steps 6–8 |
| README "What is actually oracle-only" table | README `:95-104` | Separating Trainer-shaped behaviour from oracle-only code | Its facts stay true; its scope ("only `build_legacy_mode_a_model` and the parity assertions") is overtaken | **Rewrite** for the retirement scope | — |
| README: `legacy_truncation_k`/`legacy_tie_policy` "renamed, not deleted" | README `:121-124` | Describing semantics the differential kept | **No** once the differential goes; the Trainer's `[:20]` is a literal in Trainer code | **Rewrite**; §8 item 5 | ledger fields |
| README removal order, steps 1–8; "What must not need touching" | README `:126-150` | An ordered, behaviour-neutral rename then deletion | Steps 1–5a done; 5b (7a) cancelled with 7a; the list protects live code except `to_global_ids` (see the D3 row) | **Rewrite** as the retirement sequence (§9); keep the list minus `to_global_ids` | — |
| **"Mode A stays the default"** | `measure_scorer.py:438-443` | The deleted launcher's path (`PLAN_B03.md:253-255`) | **No** | **Cancel** | §8 item 9 |
| A,C refused; B requires A | `measure_scorer.py:450`, `:479-490`; `PLAN_B03.md:248-252` | Attributable comparisons | No | **Cancel** with A/B | — |
| A/B architecture-equality precondition | `PLAN_B03.md:89-95`; `measurement.py:1148` | The A→B attribution | No | **Cancel** with B | — |
| A and B share one traversal | `PLAN_B03.md:112-113` | A and B scoring the same candidates | No | **Cancel** | — |
| **Modes B/C/D may not import the legacy loader or builder** | `PLAN_B03.md:46`; `measure_scorer.py:206-207`, `:241-242`; tested `test_legacy_equivalence.py:270-301` | Mode C against breakage from A's paths | Its purpose is met trivially once A is gone | **Keep the C-only CLI test**; drop its two monkeypatches (`:286-287`) | — |
| **Legacy builder: S0 left its build unchanged, and replacing or deleting it waits for item 9's verification; no new branch, metadata, schema or fallback for it** | `measure_scorer.py:238-239` (a past fact plus the item-9 gate; the earlier "may not" wording was removed in S0's review, plan `:612-616`); plan `:617-618`, `:726-727` | The no-fallback rule; no oracle revival | For the **production** builder (serving, Mode C): **yes**. For the legacy builder: moot once it is deleted | **Keep** the no-fallback rule and the stability of `build_shepherd_model` (README `:146-150`, "What must not need touching, at any step"; `:138` for the step-6 rename's "no … builder … behaviour change"). The item-9 gate on the legacy builder is replaced along with item 9; the rest lapses when the legacy builder is deleted | — |
| "No cross-mode conclusion may rest on the synthetic fixture" | README `:56-59`; tested `test_measurement_modes_bc.py:375` | Over-reading A=B=C on the fixture | No | **Cancel**; replaced by Mode C's own value test (§7) | — |
| BACKLOG §3.1: "drop Mode A — Rejected" | `BACKLOG.md:375`; `:400-402` | The ladder's control | No: rests on the ladder | **Rewrite** as superseded by this decision | — |
| **Item 1d** (done) | `BACKLOG.md:653` | Trainer ↔ Mode A agreement on CPU | Its live part moves (§5) | **Superseded**; keep as history | — |
| PLAN_B03 acceptance: A→B→C on institutional CUDA; end-to-end B/C peak memory as an institutional CUDA acceptance measurement | `PLAN_B03.md:209-211`, `:27-28` | Claims from CPU runs; memory | A→B→C: no. Mode C end-to-end peak memory on institutional CUDA: **yes, institutional**. S10's homelab reading does not meet it: S10 covers only its designated subject and excludes "any other deployment" (`PLAN_PROVENANCE_CONTRACT.md:1096-1101`; decision 7, `:1362-1365`) | **Rewrite**: drop A→B→C; keep Mode C peak memory as an institutional CUDA acceptance | 7b's remainder (the institutional Mode C run, item 14 or B-0.5; §8 items 3–4); S10 unchanged |
| **D3: malformed-truth REFUSE at three boundaries** — loader, loss, harness (`to_global_ids`) | `BACKLOG.md:93`; BACKLOG's §2.3 table `:174-181`; item 1a2 `:650`; restated in `src/kg/data_loader.py:976-979`, `tests/unit/test_trainer_truth_invariant.py:9-17`, `tests/unit/test_data_pipeline.py:612-622` | A malformed truth never scored as a rank miss in training, validation or measurement | The REFUSE contract: **yes**. The harness's local-id boundary: **no**. Mode C carries global ids and scores against `arange(D)` (`measurement.py:1274`); its truth refusal is `_assert_ids_in_range` (`:1220-1225`) | **Rewrite (proposal)**: the third boundary becomes Mode C's `_assert_ids_in_range`; drop the citation of `test_measurement_ranking.py:52-56`, which §6.2 deletes. D3 stays closed and the contract is unchanged | R2: BACKLOG's D3 text, its §2.3 table (`:174-181`) and item 1a2, and the three docstrings (§2.10) |
| `masked_mean_pool`: "mirrors `Trainer._compute_model_outputs` operation for operation … behaviour to preserve, not to improve" | `src/inference/scoring.py:168-177` | Mode A as a faithful control of the Trainer | The Mode A reason: **no**. Mode C refuses an empty phenotype list before pooling (`measurement.py:1207-1212`). A mirror-the-Trainer reason remains because the migrated Trainer test takes its expected values from this function (§5.1 G1–G3) | **Rewrite** the reason to name that test (proposal); behaviour unchanged | G1–G3; B-1's pool unification (`BACKLOG.md:669`) |
| PLAN_B02 "no conclusion until institutional Mode A calibration succeeds" | `PLAN_B02_shipped.md:109-110`, `:371-372` | B-0.2's parity acceptance | No; the file is "History, not authority" (`:1-4`) | **Leave as history** | — |
| `read_samples`' three-field default, "would change Modes A and B" | plan `:698-702`; `file_storage.py:126-130` | A/B semantics | The reason lapses; the interface is harmless | **Rewrite** the reason (§1.4) | S8 unchanged |
| S7's Mode A block; A/B regression tests | plan `:1022-1039` | Mode A on the shared read | No | **Cancel** (deferred now, §1.3) | S7 |
| S10's A/B "not applicable" rows | plan `:1124-1133` | Recording why A/B were not measured | No | **Rewrite** to one line | S10 |
| Item 14 phase 1: "`measure_scorer` in all its modes" moves onto the resolver | `PLAN_TEST_RESULTS.md:737-748` | Moving every mode | Only C remains | **Rewrite**: Mode C | item 14 (smaller) |
| BACKLOG 19.18 | `BACKLOG.md:708` | The deferred legacy builder | No | **Close** as "removed by …", as 19.19 was | — |
| PLAN_CONFIGURABILITY Proposal B: "Mode B/C attribution is correct" | `PLAN_CONFIGURABILITY_AND_PROVENANCE.md:374-376` | Regime provenance of records | Done; its Mode C half stays true | Leave as history | — |
| `Makefile:62-66`: `make check` includes integration tests, citing the `cuda_executed` assertion | `Makefile:62-66` | The CPU-run marker | Yes, but the assertion lives only in the Mode A `measured` fixture | **Move** the assertion to a Mode C CLI run (§6) | `make check` |

**Not affected:** EVALUATION_COHORTS' test-validity conditions (`:641-653`, institutional, no A/B
content); the institution's checkpoint choice (item 6); S10's Mode C capacity acceptance
(decision 7).

---

## 5. Training and validation guarantees that must be kept

Removing offline A/B does not change the Trainer's training or validation semantics; it removes
some of their **test coverage**. Each guarantee: live code → current protection → action.

### 5.1 Guarantees whose only protection runs through A/B — **move before anything is deleted**

| # | Guarantee | Live code | Only protection today | Move to |
|---|---|---|---|---|
| G1 | Validation ranks by **descending** score, and `val_mrr` is computed on the top-20 window of that ranking | `trainer.py:684`, `:689` | the differential file (12 tests fail on a reversed sort; a window that keeps the last 20 columns instead of the first, `pred_indices[-20:]`, fails only the wide-cohort test `:510`) | Trainer: drive `_run_evaluation_pass` on real loader batches; compute the expected top-20 rows with the served primitives `masked_mean_pool` and `cosine_score_matrix` (`src/inference/scoring.py:159-232`), not with Trainer code; assert `predictions` rows, truths and MRR, on the default workspace and on a wide one (more than 20 candidates per row) |
| G2 | Masked-mean pooling ignores padded phenotypes | `trainer.py:801-805` (also feeds the training loss) | `test_differential_calibration.py:698` only | the same Trainer test, with a padded batch and a model whose rows differ |
| G3 | Cosine normalises the disease embeddings | `trainer.py:836-838` | `test_differential_calibration.py:698` only | the same test, with unequal-norm embeddings |
| G4 | Trainer validation on real `DiagnosisDataLoader` batches with a `build_shepherd_model` model, in process | — | every differential test on the `cohort`/`wide_cohort` fixtures (`test_differential_calibration.py:58-146`): through `compare_trainer_against_mode_a` (`src/evaluation/differential.py:409-410`) at `:177`, `:510` and the other `run(cohort)` tests, and directly at `:208`, `:521` | covered by G1's test, i.e. the moved `:177` and `:510` (§6.1). Once they move, `:208` and `:521` add nothing and go with class 1 |
| G5 | Exactly `num_negative_samples` negatives per sample | `src/kg/data_loader.py:666` | `test_measurement_mode_a.py:587`; `test_legacy_equivalence.py:124` | data_loader test on `DiagnosisDataset` |
| G6 | Worker-drawn negatives follow the parent seed; the worker count is part of the stream | PyTorch worker seeding of `DiagnosisDataLoader` (`data_loader.py:832-840`); training uses 4 workers (`scripts/train_model.py:166`) | `test_measurement_mode_a.py:780`, `:792` (they already drive `DiagnosisDataset` directly, but import `scripts.measure_scorer`) | data_loader test, seeding the parent as training does, without the CLI |
| G7 | The live loader expands 2 hops | `data_loader.py:783`, `:853` | caught only incidentally (a Mode A precondition, `test_measurement_mode_a.py:903`, and the differential's wide fixture) | data_loader test on subgraph size or membership |
| G8 | The truth is a seed of its own subgraph | `data_loader.py:935-944`; enforced at run time at `:864` | **none that detects a break.** On the shared fixture every gene links every phenotype to every disease (`tests/fixtures/synthetic_workspace.py:16-18`), so the 2-hop expansion from the phenotype seeds reaches every disease. A truth dropped from the seeds (`data_loader.py:936`) is therefore still a candidate, and `test_measurement_mode_a.py:96` and `test_legacy_equivalence.py:166` both pass under that mutation. The only guard is the run-time refusal at `:864`, which no test drives through the loader (§5.3) | new data_loader test on a graph where the truth is reachable only as a seed: no path from the sample's phenotypes or genes to it within 2 hops, and no negative equal to it. Mutation check: emptying the truth from the seeds at `:936` must make it fail |

**Evidence that the move is enough** must be a mutation check on the moved tests, each mutation
caught by the Trainer test with no Mode A present: the reversed sort, the dropped masked mean and
the dropped normalisation (G1–G3), and a window that keeps 20 columns other than the top 20
(`pred_indices[-20:]` at `trainer.py:689`) on rows wider than 20. Row length alone is already
checked by `test_trainer_validation_characterization.py:386-397`, which checks shape only: an
in-memory run of the `[-20:]` mutation over the Trainer, characterization, truth-invariant,
checkpoint, rank-metric, training-run and differential tests failed exactly one test, the
differential's wide-cohort `:510`. A scratch probe of the G1 shape caught the reversed sort and
the dropped normalisation; the masked mean needs a padded batch, which that probe did not have.

### 5.2 Guarantees already protected directly — **unchanged**

Induced edges (`tests/unit/test_subgraph_equivalence.py`); the truth-range check as a unit
(`test_data_pipeline.py:639-680`); `val_` prefix, loss mean, best-metric bookkeeping, callback
order, empty traversal, malformed-truth refusal, AMP placement on CPU, top-20 local string rows
(`test_trainer_validation_characterization.py`); `DiagnosisLoss` refusals
(`test_trainer_truth_invariant.py:79`); checkpoint contents and provenance
(`test_training_provenance.py:370-397`, `test_checkpoint_log_metrics.py`); scheduler guard
(`test_scheduler_min_lr_guard.py`); training refusing a mixed workspace
(`test_graph_artifact_binding.py:127`, `:145`).

### 5.3 Gaps found that A/B never covered — **recorded, not part of retirement**

These are unprotected today and stay so whether A/B go or not: MRR and hits arithmetic in the
Trainer's path; tie order; what the Trainer's clamp (`:795`) does with a real negative id; whether
the loader actually calls its truth check; the remap round trip (every real-loader fixture has an
identity map); exclusion of the positive from negatives; neighbour limits; prefetch matching the
serial path; early-stopping patience and restore; top-k checkpoint mode; gradient accumulation,
clipping, scheduler skip and train-loss aggregation; the CUDA AMP/`GradScaler` path; a model that
yields no embeddings (validation returns `{'val_loss': 0.0}` and the run exits 0); negative global
phenotype ids admitted silently; the unbounded negative-draw loop. **Also found:** `--seed` does
not fix the initial weights, because the model is built (`train_model.py:751`) before the Trainer
seeds (`trainer.py:256`); and validation always draws the default five negatives, whatever the
training setting (`train_model.py:608-614`). These are for the owner to schedule separately.

---

## 6. Tests: keep, move, delete

Counts: 169 test functions (224 collected items, counted by `pytest --collect-only` at
`6f16373`) import or exercise an A/B component. Classes:

- **class 1** — its subject is a retired feature; delete with it. Where it also touches a live
  guarantee incidentally, the §5.1 replacement must land first: `test_measurement_mode_a.py:96`
  (G8), `:903` (G7), `test_legacy_equivalence.py:124` (G5), `test_differential_calibration.py:208`,
  `:521` (G4). **63.**
- **class 2** — live; stays, at most moved or trimmed (`test_legacy_equivalence.py:270` drops its
  two monkeypatches at `:286-287`; `test_measurement_ranking.py:251` drops its `legacy_ranking`
  assertion; `test_evaluation_sidecar.py:198` drops its `num_workers`/`shuffle` rows if §8 item 5
  removes those fields, and its seed rows follow the seed decision, as do `:267` and `:277`). **70.**
- **class 3** — live, reached through an A/B entry point; moves to the real use point. **36.**

| File | Functions | Class 1 | Class 2 | Class 3 |
|---|---|---|---|---|
| `tests/unit/test_measurement_mode_a.py` | 42 | 12 | 19 | 11 |
| `tests/unit/test_measurement_modes_bc.py` | 24 | 8 | 0 | 16 |
| `tests/unit/test_measurement_ranking.py` | 24 | 8 | 15 | 1 |
| `tests/unit/test_differential_calibration.py` | 31 | 27 | 0 | 4 |
| `tests/integration/test_legacy_equivalence.py` | 16 | 8 | 4 | 4 |
| `tests/unit/test_scoring_primitives.py` (the 18 tests that reach `masked_mean_pool`/`cosine_score_matrix`, directly or through `cosine_scores`: `:135`, `:149`, `:162`, `:290`, `:812`, `:851`–`:1018`) | 18 | 0 | 18 | 0 |
| `tests/unit/test_split_caveat.py` | 5 | 0 | 5 | 0 |
| others (`test_graph_artifact_binding.py:278`, `:290`; `test_evaluation_sidecar.py:159`, `:198`, `:267`, `:277`; `test_training_provenance.py:65`, `:470`; `integration/test_pipeline.py:381`) | 9 | 0 | 9 | 0 |
| **Total** | **169** | **63** | **70** | **36** |

The other 32 tests in `test_scoring_primitives.py` (served pooling, the SP index and transform,
the η mixture, the pipeline's SP and combined wrappers) touch no A/B component and are outside
this inventory; the first draft counted the whole file.

**Order matters.** All class-3 and 38 class-2 tests sit in files whose module-level imports name
A/B code (`test_measurement_mode_a.py:20-28`, `test_measurement_modes_bc.py:31-38`,
`test_measurement_ranking.py:20-25`, `test_differential_calibration.py:40-44`,
`test_legacy_equivalence.py:35`); deleting the names breaks collection of those files. Move
first, delete second.

### 6.1 Class 3 — where each moves (existing code only; no second Trainer, scorer or launcher)

| Use point | Tests (at `f581f9e`) | Note |
|---|---|---|
| **Trainer** (`_run_evaluation_pass`, beside `test_trainer_validation_characterization.py`) | `test_differential_calibration.py:177`, `:495` (folded into `:510` as an inline precondition), `:510`, `:698` | G1–G4; expected values from the served primitives, mutation-checked. `:510` keeps its wide workspace (`build_workspace` with more than 20 diseases and every `num_neighbors` hop above the disease count, as at `test_differential_calibration.py:124-146`) and asserts `:495`'s precondition inline: every candidate universe is larger than 20. It is the only test that catches a wrong top-20 window (§5.1) |
| **data_loader** | `test_measurement_mode_a.py:587` (negative count), `:780`, `:792` (worker streams); plus new tests for G7 (protected only incidentally, `test_measurement_mode_a.py:903` and the differential's wide fixture) and G8 (not protected by any test today, §5.1) | G5–G8; no `measure_scorer` import |
| **Shared ranking helpers** (`canonical_ranking`, `ranks_of_truth`) | `test_measurement_ranking.py:173` (identifier dtype refusal) | retarget its `float32`/`float64`/`bool` cases from `to_global_ids` to `canonical_ranking`'s `global_disease_ids` and to both arguments of `ranks_of_truth`; it is the only test of `_require_integer_ids`'s `torch.bool` branch (`measurement.py:61`), which §2.4 keeps |
| **Mode C** (`run_mode_c`, `encode_full_graph`, `_assert_cohort_is_intact`, `build_manifest(mode="C")`) | `test_measurement_mode_a.py:85` (authoritative half), `:107` (mean rank), `:145` (semantics fields), `:161` (numeric regime), `:471`, `:514` (cohort shrinkage), `:568` (report omits per-sample ids), `:805` (no null seed; follows the seed decision, §8 item 5) | each loses its Mode A half |
| **Mode C fixture** | all 16 class-3 tests in `test_measurement_modes_bc.py` (`:187`, `:210`, `:228`, `:246`, `:266`, `:293`, `:302` (follows §8 item 5, §2.4), `:311` (as "ids and truths in samples order"), `:321`, `:328` (C part), `:336`, `:412`, `:433`, `:447`, `:462`, `:475`) | only the `world` fixture changes: read through `read_graph_artifacts`/`read_samples`, build with `build_shepherd_model`, a Mode C manifest |
| **CLI, `--modes C` subprocess** | `test_legacy_equivalence.py:103` (authoritative family in the file), `:141` (digests are of the files given), `:166` (cohort whole), `:180` (`cuda_executed` false; `Makefile:62-66` cites it) | the `measured` fixture runs Mode C |

**Class 2 that must change file:** in `test_measurement_mode_a.py`, the compile-wrapper probes
(`:313`–`:405`, 8), `_resolve_device` (`:625`, `:636`), the `file_sha256` re-export (`:648`;
S9 names it), seed validation and application (`:679`, `:689`, `:700`, `:708`; these follow the seed decision, §8 item 5), artifact roles
(`:854`, `:863`, `:876`, `:885`) — a new CLI test file with no change to what they call. In
`test_legacy_equivalence.py`, `:270` (keep; drop its two monkeypatches), `:364`, `:378`, `:388`.

### 6.2 Class 1 — delete with A/B

`test_measurement_mode_a.py`: `:76`, `:96`, `:117`, `:132`, `:180`, `:220`, `:242`, `:422`,
`:435`, `:548`, `:607`, `:903`, and the legacy halves of the class-3 tests. Note `:903` tests Mode
A's copy of the clamp; the Trainer's clamp keeps only a shape test (§5.3).
`test_measurement_modes_bc.py`: `:118`, `:130`, `:139`, `:153`, `:349`, `:356`, `:375`, `:493`.
`:375` (A=B=C on the fixture) is today the only comparison of Mode C's rank **values** with
another path; it is replaced by Mode C's own value test (§7), not kept.
`test_measurement_ranking.py`: the `to_global_ids` tests (`:31`, `:41`, `:53`, `:59`) and the
`legacy_ranking` tests (`:124`, `:137`, `:150`, `:162`); `:173` moves (§6.1); `:251` keeps its
canonical half.
`test_differential_calibration.py`: all but `:177`, `:495`, `:510`, `:698` (the comparator's own
guards, its proofs that it can fail, its verdict fields and its AMP report; `:208` and `:521` also
run the Trainer on the real cohorts, but the moved `:177` and `:510` cover that, §5.1 G4).
`test_legacy_equivalence.py`: `:96`, `:115`, `:124`, `:193`, `:213`, `:253`, `:304`, `:343`.

---

## 7. The formal Test without A/B: direct acceptance of Mode C and the shared scorer

**Mode C's correctness never depended on Mode A**, but its direct tests are thinner than they
look: one Mode C rank-value check exists, and it is the A=B=C fixture comparison. What each
property has today, and the smallest direct test that closes the gap (a hand-built `EncodedGraph`
with axis-aligned vectors, so expected ranks follow from geometry, not from re-implementing the
scorer; no parallel evaluation engine):

| Property | Today (direct tests) | Gap | Minimal direct test |
|---|---|---|---|
| (a) Candidates are every disease | `test_measurement_modes_bc.py:293` (column count, 4 diseases); `:475` (last id accepted) | count only | ≥ 21 diseases, truth placed at rank D: assert rank D, `candidate_columns` D, every id 0…D−1 accepted as truth |
| (b) Pooling, cosine, ranking, tie rule | primitives: `test_scoring_primitives.py:851`–`:991`; `canonical_ranking`: `test_measurement_ranking.py:67`–`:197` | nothing end to end through `run_mode_c`; no tie and no padding inside a successful run (every fixture patient has two phenotypes) | two identical disease rows → lower id first, unchanged across `batch_size` and sample order; a variable-length batch whose padding row 0 is distinctive → no leak |
| (c) Counts, ground truth, MRR / hits / mean rank | `:321` (on an intact run `n_ground_truth_absent` is 0 and `n_ranked` equals `n_samples`; no absence is ever caused), `:336` (ranks line up) | **no Mode C metric value or key set is asserted**; truth ids checked only against A (`:311`); shrinkage and the absence refusal tested only in the Mode A file (`test_measurement_mode_a.py:514`, cases `:507-511`; its `n_absent` case `:507` is the only test that makes absence fatal) | geometry giving ranks such as [1, 2, 4]: exact `untruncated_mrr`, `mean_rank`, each hits@k and the key set; `n_samples` mismatch → refused; empty cohort → refused; `n_absent=1` passed to `_assert_cohort_is_intact` under a Mode C manifest → refused with "absent from the candidate set" (taking over mode_a `:507`'s case); `truth_global_ids` equal the samples' `disease_id` |
| (d) Illegal input and refusal | empty phenotypes, out-of-range ids, truth range, regime mismatch: `:408`–`:462`, `:187`–`:246` | missing embedding key is a bare `KeyError`; NaN only tested at the primitive; **a non-integer id (1.5, `True`) is accepted and truncated** (`measurement.py:1290`, `:1314`); `batch_size` ≤ 0 | tests for each; the non-integer and missing-key cases need a **code decision** first (§8 item 8) |
| (e) Same semantics as the service | `encode_full_graph` equals serving's cached embeddings (`tests/integration/test_pipeline.py:381-429`); the service's cosine delegates to `cosine_score_matrix` (`scoring.py:244`) | nothing shows `run_mode_c` routes through the shared primitives | patch `src.inference.scoring.cosine_score_matrix`/`masked_mean_pool` and assert `run_mode_c` follows (as `test_scoring_primitives.py:316` does for the service) |

**What is not, and cannot yet be, the same.** Until B-1, the service differs from Mode C by
design: BFS-discovered candidates, not every disease; `0.7·((cos+1)/2) + 0.3·SP`, not raw cosine
(`DISEASE_SCORER_POLICY.md:122-123`); a clamping pool (`scoring.py:155`), not refusal;
insertion-order ties, not the canonical rule (`src/inference/pipeline.py:1518`, a stable sort on
`confidence_score` alone; the code's note at `:1581-1582` says a tie "is then resolved by sort
stability — that is, by input order"). Tests can only pin those documented differences now. B-1
is to share Mode C's pool, cosine, candidate universe and tie rule, recorded as a constraint on
B-1 (`PLAN_TEST_RESULTS.md:365-380`), with raw cosine as the score target
(`DISEASE_SCORER_POLICY.md:123`).

**Not a substitute.** "Mode C equals the Trainer's `val_mrr`" is not an acceptance: candidate
sets, id spaces, truncation, metric sets and tie rules differ by design (`BACKLOG.md:376` withdrew
that comparison as a false premise).

---

## 8. Decisions for the owner

1. **Approve the retirement scope** in §9: CLI Modes A and B, the legacy builder and loader,
   `run_mode_a`/`run_modes_ab` and their A/B-only helpers, the differential calibration and its
   tests — after the moved tests are in place. **Recommended, conditional on items 4 and 7.** If
   the institution requires A/B content in the B-0 report (item 4), this scope is reopened; if it
   chooses item 7 (b), that comparison runs before the differential is deleted.
2. **The research comparisons (A→B, B→C)** are retired with no replacement as research tools. If a
   research item later names a user and a deliverable, it builds on Mode C's code and the
   fixed-revision history; nothing is kept as a standing research directory. **Recommended,
   subject to item 4**: if the institution's agreed B-0 scope includes A/B, these comparisons are
   B-0 measurements, not research, and are reopened with item 1.
3. **Backlog rows** (proposals; each lands only if the retirement goes ahead, i.e. after the
   institution's answers to items 4 and 7, §9): cancel 7a; rewrite 7b without A/B and without
   7a; rewrite 8b's dependency; replace item 9 with the retirement change; close 19.18; rewrite
   BACKLOG §3.1's "drop Mode A — Rejected" as superseded; re-point D3's third boundary (§4). 7b and 8b are co-owned with the
   institution (`BACKLOG.md:667`, `:676`): their rewrites go to the institution with items 4 and 7
   and depend on its answers, and 7b's item-6, deployment-CUDA and institutional Mode C peak-memory
   preconditions carry over to whatever replaces it.
4. **Policy Gate 1 (institutional).** Its "B-0 measurement report" names no mode, but the policy
   refers to "the agreed B-0 scope" (`DISEASE_SCORER_POLICY.md:318`; B-0 "approved for
   implementation", `:289`). That scope is not in the repository, and the repository's own B-0
   includes Modes A and B (README `:30-36`; item 7b, `BACKLOG.md:667`). **Ask the institution to
   confirm the scope**, and propose that Gate 1 be satisfied without Modes A/B, giving as the
   reason that Mode A cannot run on any checkpoint the current writer produces (19.18) and that
   the Test's mode is decided. B-0 exists to measure the served scorer before it changes (README
   `:3-7`), so the report would compare Mode C (the target scorer) with the served behaviour as
   B-0.5 measures it: Mode D, production's path-reachable candidates with the η/SP mixture, plus
   its intermediate cosine-on-reachable-candidates step (README `:20-28`, `:36`;
   `PLAN_TEST_RESULTS.md:189-191`). Neither needs A or B, and B-0.5 is not affected by this
   retirement. If formal Test-cohort results are also cited, they keep their roles
   (`EVALUATION_COHORTS.md:505-509`, MyGene2 "Not an acceptance gate"; test-validity conditions 5
   and 6, `:647-648`). The gate
   itself stays; engineering does not cancel it. **This one needs the institution.**
5. **Manifest fields and the seed.** `legacy_truncation_k`, `legacy_tie_policy`, the
   loader/sampler fields Mode C does not use (all of `measure_scorer.py:355-363` except
   `batch_size` at `:361`), `build_loader_config` (§2.1), Mode C's `sampler_evidence` entries
   `max_subgraph_nodes` and `negative_sampling` (§2.4), and the seed: `--seed`, the three seed
   fields (`:392-394`) and the ledger's null-seed refusal (`sidecar.py:267-288`). Either remove
   them with retirement (the ledger already rekeys on every commit; update `sidecar.py`'s field
   lists, its refusal and their tests: `test_evaluation_sidecar.py:159`, the `num_workers`,
   `shuffle` and seed rows of `:198`, and the seed tests `:267` and `:277`), or keep them with a
   stated Mode C reason, rewording `--seed`'s help (`measure_scorer.py:431-437`), `main`'s comment
   (`measure_scorer.py:528-536`) and the sidecar comments (`sidecar.py:83-86`, `:267-271`), which
   today give only A/B's reasons. `validate_measurement_seed` (`measurement.py:480-497`) follows
   the same choice. **Recommended: remove the
   `legacy_*` and loader/sampler fields with retirement**, so no Mode C record describes A's
   semantics. **The seed: owner's choice**; if kept, its review condition is item 14's ledger v2.
6. **`read_samples`' default.** Keep the three-field default and `training_fields=True`, rewording
   the reason, or always carry five fields. **Recommended: keep the interface** (no S1–S5 change);
   only the reason is reworded.
7. **AMP reordering of Trainer validation** (7a's AMP leg). The institution's experimenters own
   this question ("run it both ways if they wish", `BACKLOG.md:491-497`), and engineering recorded
   the per-sample evidence for it as already in place in `DifferentialResult` (`:499-505`).
   Retiring the differential removes that evidence, and the CPU Trainer tests of §5 cannot
   replace it (`use_amp` is off on CPU). Propose to the institution either (a) that they no
   longer need it, or (b) that a Trainer-only comparison, the same validation pass with `use_amp`
   on vs off on CUDA, replace it, reporting the same per-sample quantities; Mode A adds nothing to
   that comparison. **This one needs the institution**, like item 4. If they choose (b), that
   comparison runs before the differential is deleted.
8. **Mode C input decisions** found while inventorying: refuse non-integer and boolean ids (today
   silently truncated), and give a missing embedding key a named refusal. **Recommended: refuse**;
   each is a small Mode C change with its test.
9. **The `--modes` option.** Remove it (and edit S10's command line, plan `:1120`), or keep it
   accepting only `C`. **Recommended: remove**, rather than keep a compatibility flag; the
   measurement CLI then measures Mode C.
10. **Out of scope, surfaced for scheduling:** `--seed` does not fix initial weights; validation
    always uses five negatives; the Trainer gaps of §5.3.
11. **S7's timing if the institution's answer (items 4, 7) is slow:** hold S7, or run S7 for Mode C
    only and leave Mode A on its current reads until the answer. S7's Mode A items stay deferred
    either way (§1.3). **Owner decides.**

---

## 9. Proposed scope and sequence

**Scope (if approved):** §2's "Retire" rows; §2.4's shared core kept in C form; §6's moves before
§6.2's deletions; §7's Mode C tests; documentation and backlog rows per §4 and §8.

**Sequence** — each step one reviewable change; tests move before code goes:

| Step | Change | Production code? | Gate before the next |
|---|---|---|---|
| **R0** | This assessment; owner decisions (§8) | No | Owner decisions recorded (§8). R1 does not wait for the institution |
| **R1** | Move class-3 tests to their use points (§6.1): the Trainer value test (G1–G4, including the wide-cohort case), data_loader tests (G5–G8), the shared-ranking dtype test, the Mode C fixture rewrite, Mode C unit tests, the `--modes C` CLI subprocess test; move class-2 tests out of A-named files; add §7's Mode C tests that need no code decision | No (tests only) | Mutation checks: the reversed sort, the dropped masked mean, the dropped normalisation and a wrong top-20 window (`pred_indices[-20:]`) on rows wider than 20 are each caught by the Trainer test; emptying the truth from the seeds (`data_loader.py:936`) is caught by the G8 test; the Mode C value tests catch a broken tie rule and a wrong metric. **Before R2:** the institution's answers to §8 items 4 and 7 are recorded and, if item 7 is (b), the Trainer-only CUDA comparison has run |
| **R2** | Delete §2's "Retire" rows and §6.2's class-1 tests; drop the fixture's legacy keys so Mode C tests run on writer-shaped checkpoints; rewrite the CLI's texts, `build_manifest`'s defaults, `_assert_cohort_is_intact`'s wording, the import contract's reason, the scorer README and BACKLOG rows (§4, D3 included), plan S7/S10/19.18 (§1.3–§1.4), and the A/B wording left in kept files (§2.10). `build_loader_config` changes in this step with `--num-workers` (§2.1) | Yes (deletions, wording) | Full suites; import-linter; no Mode C manifest or ledger behaviour changes except §8 item 5 if chosen |
| **R3** | §8 item 5 (the parts not already landed in R2 with `--num-workers` and `build_loader_config`, §2.1), items 8 and 9, as decided | Yes (small) | Their own tests |
| — | **M2.1 S7** then runs on Mode C alone | — | — |

**What waits for the institution.** R1 is tests only and adds coverage whichever way the
institution answers, so it can go ahead once the owner approves. R2's A/B deletions and the
7a/7b/8b BACKLOG rewrites wait for the institution's answers to §8 item 4 (the agreed B-0 scope)
and item 7 (the AMP evidence). If item 7 is answered (b), the Trainer-only CUDA comparison runs
before the differential is deleted.

**Relation to M2.1:** R1–R2 before S7. S6 (serving) and S8 (training) proceed independently. S9's
pin list follows R1 (§1.4). If the institution's answer is slow, the owner chooses between holding
S7 and running S7 for Mode C only, with Mode A left on its current reads until the answer (§8 item
11). Either way S7's Mode A items stay unbuilt; building them in the meantime is the cost this
ordering avoids.

**Cost being avoided by deciding now:** S7's Mode A plumbing and A/B regression tests (plan
`:1022-1039`), item 9's ~70-reference rename (`BACKLOG.md:677`), item 7a's runner and its
institutional run, and the 4-worker Mode A subprocess every `make check` runs
(`test_legacy_equivalence.py:46-90`, the `measured` fixture, whose subprocess at `:60-77` passes
`--num-workers 4` at `:69`; `Makefile:66`).
