# Programme backlog — what is open, in what order, and why

**Purpose.** One ordered list across every live phase, so the next action is
readable without reconstructing six review threads. Phase documents keep their
own detail; this file holds **ordering, dependencies and blockers only** and must
not restate their decisions.

**Status:** sixth revision. The calibration decision at item 1 is **made and
reviewed** (§3.1.2); the legacy-removal checklist it invalidated is corrected and
suspended. (Both superseded on 2026-10-12: R2, implemented and code-reviewed with no P1 or P2 (`1d0b5a6..06fd0f1`, 2026-10-12), retired the
differential calibration with Modes A and B, and the scorer README carries the retirement's
record in place of the checklist.) B-0.4's measurement is **complete and reviewed**, and its
productionisation is a separate item (5a) behind its own gate. This revision adds
**M9 and item 11**: the standard generated workspace has no held-out split and no
protocol defines one — known in two passing mentions and owned by nothing. Item
11's scope is the **unit** of holdout, not a filename (§3.4).

**Next action (2026-10-21): M5-3, resumed after the pause the owner asked for at the end of M5-2 (weekly usage budget): the screenshots for acceptance and the delivery for review.** M5-2, the UI, is pushed (`badf546..da2a8b1`; full suite 4990 passed, 14 skipped): the two-level navigation with the operator-controls and GPU-jobs chips, the shared help control, the shared renderers and the one model list and Load; Datasets (read-only); Diagnosis with the "Model in use" card, the three-state results badge, the identity drawer and "Change model…", Run Diagnosis greyed out with a bubble while no model is loaded; Training with the run bar, the dataset fingerprint checked again at start, the Continue drawer and the uPlot charts (vendored 1.6.32); Test (intake import, runs through the job runner, keep and register, D5 notes); Models (the research list, ordering for the server's model-selection roles, usage history, Load). Each package was reviewed by two or three lenses and fixed. **M5-3 so far:** the shared follow-ups the package reviews found are fixed and reviewed (`4c7d12f`, `5994a46`): the shared Keep button's detached elements, a no-wrap run label, the label and checkbox contrast, the double refresh on first opening Research, the role code's wrap, measure_scorer's clean cancel, `list_report_runs` listing runs only, the Test page's note form, and the unused recorded-results block removed. The deployment probe carries M5's acceptance where it runs headless (`e3939f3`, `5a2d639`: A5, W1–W11, RW1–RW3; `PLAN_TEST_RESULTS.md` §6, Acceptance). The internal review of all of M5 (five lenses: architecture, security, evaluation, UI and long use, documents; 42 findings) is fixed on the branch after `5a2d639`, the documents read through by its documents lens; what it left for later is items 36–39 and item 32's restore after a reconnect. **M5-3 remains:** the screenshots for acceptance and the delivery for review. Before the pause the next action read: M5 on the development branch, reviewed once when all of it is done (the reviewer's instruction, relayed by the owner).** M5-0, the dashboard on Gradio 6.30.0 (`14176c3`, item 34), and M5-1, the backend foundations (`85af1fa..bfc0b7c`), are pushed: the server settings (`SHEPHERD_DEPLOYMENT_MODE`, the workspace root, the registry, the intake directory, the ordering roles) and the server's lists of workspaces, checkpoints and cohort versions by name and digest, with `workspace_build.json` recording a build time; D8's one check on every operator control, in the API and the dashboard alike; no cross-origin browser access by default; the loopback default bind; one GPU lock and the job runner for training and test runs; registration from the report store only, with D5's disagreement block, D4's ordering function and the usage history; and `measure_scorer`'s deterministic switch, expected digests and outcome file (`PLAN_TEST_RESULTS.md` D4, D5, D8 and §6, each "as implemented"). Next: M5-2, the UI (the two-level navigation, the Diagnosis tab's model card, the Training page's run bar and charts, Datasets, Test and Models), then M5-3 (the probe, the documents, the internal review and the screenshots) and the delivery for review. The owner's feedback from long use of the UI becomes items here, fixed in later rounds; M5 does not wait for it. Before M5-1 the next action read: M5. Batch B is merged into `main` (#128, `150105f`, the owner, 2026-10-19). The owner found the first M5 mockups clean and asked for a second round (navigation in two levels, clinical apart from pre-clinical; the loaded model visible on the Diagnosis tab before a diagnosis; the training page's dataset version, zoomable charts and control layout), and the reviewer's decisions, relayed by the owner, 2026-10-19 settled the open M5 questions: D8 gates every administrative operation, training start and stop included; the default bind is `127.0.0.1`; import sources come from a server intake directory; one GPU lock covers managed training and evaluation jobs, the formal CLI included; ordering by test score only for research-purpose cohorts; registration from the registry's report store only (`PLAN_TEST_RESULTS.md` D5, D7, D8, §4.9 question 1, §6). Then the layout direction on a thin Gradio prototype, and M5's delivery. Before the merge the next action read: the owner's decision on a pull request for batch B, then M5. Batch B (N2 without its manifest field, item 16; F5–F7, item 17; item 26; the owner's WebUI rules as item 27) was reviewed by the reviewer with no P1 or P2 (`1fe0ea9..2e099b6`, 2026-10-19): the reviewer's targeted run passed (722 tests), and `hpo_2026_0929_5a`'s train and val samples pass N2's reader on the homelab, so N2 needs no rebuild; a short manual check with a qualified model (the launcher waiting past 60 s, the two path fields, a diagnosis and its CSV) is suggested, not required; not merged. M5, item 14's phase 2, follows within the reviewer's boundaries relayed by the owner (`PLAN_TEST_RESULTS.md` §6, "M5's boundaries"): deployment mode M1 (§4.9 question 3, answered 2026-10-19), the job runner, the Test tab, Model Management, D4 and D5, the Diagnosis integration, with two checkpoints — the layout direction on a thin, operable Gradio prototype (a mockup before, screenshots at completion), then M5's delivery; M5 keeps several frontends on one inference backend possible without opening the LAN (item 28). The second mockup round (2026-10-19) adds to M5, at the owner's request, the two-level navigation, the Diagnosis tab's model card and change-model drawer, Run Diagnosis greyed out while no model is loaded, the dataset identity with its build time recorded from now on, a read-only Datasets page, the zoomable charts and the Training Console's run bar; dataset building in the WebUI, a release designation, model comparison, navigation follow-ups and model-with-dataset export and import are recorded as items 29 to 33. Before the review the next action read: the review of batch B, then M5. Batch B, chosen by the owner before M5 (2026-10-18), is built on this branch over `main` at `1fe0ea9`: N2 without its manifest field (item 16), F5–F7 (item 17) and item 26, with the owner's WebUI rules recorded as item 27; not merged. M5, item 14's phase 2 (`PLAN_TEST_RESULTS.md` §6, steps 5–7), follows and first needs the owner's answer to §4.9 question 3 (who operates, and which deployment mode); branch mode (item 23), the AMP research, HGT Edit 2 and the memory attribution (item 22) stay scheduled on their own, and real institutional testing waits for data, terms and a declared protocol. Before batch B the next action read: the owner's choice of the next work. M4 (item 14's phase 1) is merged into `main` (#126, `d2d31dd`, the owner, 2026-10-17). In the plan's order the next milestone is M5, item 14's phase 2 (`PLAN_TEST_RESULTS.md` §6, steps 5–7), which first needs the owner's answer to §4.9 question 3 (who operates, and which deployment mode); N2, branch mode (item 23), the AMP research, HGT Edit 2 and the memory attribution (item 22) stay scheduled on their own, and real institutional testing waits for data, terms and a declared protocol. Before the merge the next action read: the owner's decision on a pull request for item 14's phase 1, the contract's M4. Its GB10 functional acceptance, within the scope agreed, passed at `2c94a4e`: 84 of 84 probes, demo and real data, T1–T6 and RT1–RT3 among them, reviewed by the reviewer (the plan's §4, "M4's acceptance on the GB10"; evidence [`m4-acceptance-2026-10-17/`](m4-acceptance-2026-10-17/README.md)); functional only, not model quality, capacity or an institutional test; nothing merged. Before the run the next action read: the reviewer's code review of `59f519d..f3108c4` found one P2 (`--record` did not bind the report it read back to the run), fixed in `29c3d1f`; its incremental review closed it and found one P3 (a lost report called kept), fixed in `2c94a4e`; then the GB10 acceptance on that commit** (the plan's §4, "M4 as implemented"; `PLAN_TEST_RESULTS.md` §6, "Phase 1 as implemented"). Not accepted and not merged. Its internal review (ten dimensions, each finding put to independent skeptics) kept 29 findings, none P1, two P2 (the ledger split for a symlinked checkpoint, and links below the registry root), and refuted one; all 29 are fixed in `f3108c4` (the plan's §4, "M4 as implemented", "Internal review"). M4, built over `main` at `59f519d` (`6e37208`, `d3d6ce7`, then the probe's M4 probes and the documents; the full suite passed at `d3d6ce7`, 3751 passed, 14 skipped): test cohort versions imported from the CLI and bound to the graph, Mode C on a registered version with R10 and the environment recorded, every report kept, one registration function for `--record` and `record_evaluation`, ledger schema 3, and the loaded checkpoint's records on the Diagnosis tab's model status through a read-only route; items 18, 19.5 and 19.6 are done in it. The owner questions are asked once, each with the default M4 runs on (`PLAN_TEST_RESULTS.md` §4.9); real institutional testing waits for data, terms and a declared protocol. N2, branch mode (item 23), the AMP research, HGT Edit 2 and the memory attribution (item 22) stay scheduled on their own. Before M4 the next action read: M4, item 14's phase 1 (the plan's §4, "M4"; `PLAN_TEST_RESULTS.md` revision 4). M3c is merged into `main` (#124, `ed8f51d`, the owner, 2026-10-16). Before the merge the next action read: the owner's decision on M3c's PR and merge. Item 15's M3c (what is served is identified; the plan's §4, "M3c as implemented") was code-reviewed with no P1 or P2 at `1246f0c` (2026-10-16), its one P3 fixed after it, and its GB10 functional acceptance, within the scope agreed, is passed (2026-10-16): one run at `52152e4`, demo and real data, 75 of 75 probes, with a WebUI smoke and its exports, reviewed by the reviewer (the plan's §4, "M3c's acceptance on the GB10"; evidence [`m3c-acceptance-2026-10-16/`](m3c-acceptance-2026-10-16/README.md)); functional only, not capacity, AMP, model quality or a hospital test. M4 follows the merge. M3c, implemented on this branch over `cbba015`: one structured served identity on every result, the diagnose response and the pipeline status, replacing `model_version`; serving's `auto` refusing without CUDA, as training's does; every consumer moved in the same change (the API, the WebUI's tab, report, CSV and status, the probe, the measurement script, `setup_demo`); 19.1, 19.2 and 19.8's subject done in it. N2, branch mode (item 23), the AMP research, HGT Edit 2 and the memory attribution (item 22) stay scheduled on their own. M3b is merged into `main` (#123, `cbba015`, the owner, 2026-10-16). Before M3c the next action read: M3b's functional acceptance, within the scope agreed, is passed: one run on the GB10 at `c80c1e4`, the demo and the real workspace `hpo_2026_0929_5a`, 75 of 75 probes, reviewed by the reviewer (the plan's §4, "M3b's acceptance on the GB10"); not a capacity, AMP, model-quality or clinical reading; nothing merged; the homelab checkout restored to `main` at `9fe5113` (the operator, 2026-10-16). Open: the owner's decision on a pull request for M3b, then M3c in the plan's order; N2, branch mode (item 23), the AMP research, HGT Edit 2 and the memory attribution (item 22) stay scheduled on their own. Before the run the next action read: item 15's M3b GB10 functional acceptance, run as the plan's §4 "How M3b's acceptance is run" states (the deployment probe with `--m2-acceptance --device cuda`, demo and real data in one run, the same inputs as M2's and M3a's, a new evidence root, the checkout restored afterwards). M3b's code review is passed: the reviewer's read-only review of `2049b64..04ecf42` (2026-10-16) found no P1 or P2 and one P3 (N2's stale dependency on an M3b manifest change), synced in item 16 and its plan; the reviewer's read-only inventory of `hpo_2026_0929_5a` at that commit found its dataset record accepted, binding verified, origin files, so it needs no rebuild. Before that the next action read: the reviewer's code review of item 15's M3b (the dataset record is required; the plan's §4, "M3b as implemented"), then its GB10 acceptance as the plan's "How M3b's acceptance is run" prepares it. M3a is merged into `main` (#122, `2049b64`, the owner, 2026-10-15); M3b was built on that `main` as one batch (an internal review before delivery found no P1; its one P2, the deployment guides, and its P3s are fixed), with the record read once and refused by training, serving and measurement unless bound and naming the graph, its binding state and origin shown apart, the binding required at write, the writers' digests taken from the bytes they write (19.16, 19.17) and the builder stopping on an unidentified input. Not included: M3c, branch mode (item 23), the AMP research, HGT Edit 2 and the memory attribution (item 22), each scheduled on its own; M3a's acceptance is not rerun. Before M3b the next action read: M3a's functional acceptance, within the scope agreed, is passed, in two rounds the reviewer reviewed (round 01 at `4b1571e`: real data R1–R8; round 02 at `c4a21b5`: demo A–E and M; the product code the same; the plan's §4, "How M3a's acceptance is run"). Not a capacity, model-quality or clinical benchmark reading; M3b and M3c not included; two P3 tracked (the best-k list continued from a non-best best-k file; the API's range for an explicit value); nothing merged. The homelab checkout is restored to `main` at `9fe5113` (done, 2026-10-15; the reviewer checked it, tracked files clean, round 02's `checkout-restored.txt` kept; its earlier detached state at `c4a21b5` was before the restore). Open: the pull request of this branch, authorised by the owner on the reviewer's recommendation (2026-10-15), and its merge confirmed before further work; then M3b and M3c in the plan's order. Before round 02 the next action read: round 02 of item 15's M3a acceptance on the GB10, the demo part only (A–E and M) at the probe fix's commit, read as the plan's §4 "How M3a's acceptance is run" states (it ends NOT COMPLETE and exit 1 by design, the real data not requested; read by the report: no probe failed or errored, `m2_acceptance.demo.complete` true, every required demo probe passed). The reviewer's review of the probe fix (`4b1571e..c4a21b5`) closed its P2 with no new P1 or P2. Round 01 at `4b1571e` passed real data R1–R8 and failed demo M2 on a wrong probe assertion (successful updates assumed equal per epoch; AMP skipped one in the parent's epoch), with M3, M4, M9 and M10 failing as its dependants; the probe now counts the remaining epoch by attempts. Before round 01 the next action read: item 15's M3a acceptance on the GB10, after the reviewer's review of the probe's scaler-check fix (its review of `815d763..7e1e373` found one P2 in the acceptance tool: the scaler judged from its end values; fixed by observing the restore itself), run as the plan's §4 "How M3a's acceptance is run" states (the deployment probe with `--m2-acceptance`, the same real inputs as M2's, the saved allocator, the commit recorded). M3a's code review is passed: the re-review of the fixes (`094b90d..815d763`, 2026-10-15) closed all three P2 with no new P1 or P2; its functional acceptance is complete only with the GB10 evidence reviewed. Tracked, not blocking (P3): continued from a best-k file that is not early stopping's best epoch, the child's best-k list leaves out the parent's own file (the best weights stay named and restored); to be completed in the run plan and the best-k restore by the same pruning rule. Before the re-review the next action read: the reviewer's re-review of the fixes to item 15's M3a (`094b90d..` this branch's head). M3a was implemented on 2026-10-15 as one batch (`8876d2b..094b90d`; the plan's §4, "M3a as implemented"): run directories with the consumers moved and the one-time move tool, resume with inherited settings and a pause flag, extend at a fixed rate, stop reported as stopped, and the console's continue block. Its code review found three P2 and no P1 (a best file's weights lost after one generation, a hidden Extra epochs value blocking a resume, the API checking a continuation's rules on new-run defaults), so its functional acceptance is not passed; the fixes are recorded in the plan. After the re-review, one acceptance on the GB10; nothing here authorises a merge. Its design was revised after the reviewer's design review and its incremental review, with decisions 8–10 (the plan's §1 and §4, M3a; decision 8 confirmed by the owner on 2026-10-15). Before M3a: the design was not yet implemented (its seed ordering was done ahead, `290c03a`). The GNN experiment phase and the branch mode it needs are item 23. M2 is merged into `main` (#121, `9fe5113`, the owner, 2026-10-14); statements below that it is not merged, or that a pull request is to be finished first, are the record as it stood.** The owner decided on 2026-10-14, on the reviewer's recommendation, that M2's technical work is complete and its acceptance within the scope agreed is complete, not merged, and authorised a pull request (not its merge); no M3 change is added to this branch before the merge. M2's acceptance ran on the GB10 at `82d7573` and passed, demo and real data, 62 of 62 probes, as the reviewer found ([evidence](m2-acceptance-2026-10-14/README.md)): functional only, since the probe ran the trainer as a module and so under the native allocator, not the saved `cuda_async`; no capacity reading and no model-quality claim. The probe's allocator start-up was fixed (`b8a43c8`), reviewed with no P1 or P2, and confirmed on the GB10: its demo phases ran under `cudaMallocAsync` (42 of 42; the evidence's §6). The homelab checkout is back on `main`, `90a668f`. Statements below that M2 is not complete are the record as it stood. The acceptance was run through the deployment probe extended for it (`--m2-acceptance`, the owner's decision of 2026-10-13; the plan's "Acceptance for M2", "How it is run"), after its review's two P2 (the report could overwrite an input or earlier result; M3 measured the parent) were fixed. M2.2–M2.6 are implemented; the code review of `810d344..0aabfc3` found three P2 and no P1 (a checkpoint over an in-memory graph escaped M2.4, the resume parent's read was held through training, and the inventory reported a sidecar check as a servable table), fixed with the reviewer's recommendations to retire `SHEPHERD_SP_HOP_BOUND` and to refuse a checkpoint over an unverified graph, and the re-review of `0aabfc3..26fe84c` closed all three with no new P1 or P2 (the plan's §4, M2.2–M2.6, "As implemented"). M2 is not complete until the acceptance: fresh build → train → resume → serve with a retrained model, since no existing checkpoint records `kg`, the refusals of a wrong graph and wrong data leaving no run artifact, and the inventory run (M2.6). M2.1's technical work is complete: the owner accepted S10's capacity on its designated subject on 2026-10-13, with the remaining uncertainty explicitly accepted (the capacity README §6.2); not merged. **The record before that:** close M2.1 (S9, implemented and code-reviewed with no P1 or P2, `02a6954..f481882`, 2026-10-12; then S10 on its designated subject, run on 2026-10-12 with its re-reviewed tools at `90a668f` and the "after" commit `343e90e`, its evidence and explanation awaiting review, and the reviewer's recommended limited diagnostic of consecutive reloads chosen by the owner and run on 2026-10-13: each process grew by about 0.4–0.5 GB at every reload at both commits, before M2.1 as after it, the difference between the commits ambiguous, so acceptance 2 stays open; its second phase found every watched object of the earlier pipelines freed, and the owner ended the reload-memory investigation on 2026-10-13: the PyTorch wheel in use is confirmed to contain mimalloc purge defects that upstream has since fixed; they are a strong candidate cause of the observed memory retention; the attribution of all of the increase is not established (item 22); acceptance 2, with M2.1's ~30 MB difference recorded apart, is the owner's separate decision, closed on 2026-10-13 as above), then
M2.2–M2.6**, so that the contract is enforced with producer and consumer together, with a formal
way to produce acceptable new models and the deployment inventory (the reviewer's convergence
instruction of 2026-10-12, adopted by the owner and recorded in the plan's §4). Items 20, 21, 14
and M3 stay independent of it. Item 11, the evaluation-holdout protocol decision, remains open
and blocked on neither institutional evidence nor a designated checkpoint, but it is no longer
the only such item. Items 1 through 1e are complete. Since the owner's clarification of
2026-10-12, R2, the Mode A/B retirement change that replaces item 9, is not blocked on
institutional evidence either (item 9's row); it was implemented on 2026-10-12 and code-reviewed with no P1 or P2 the same day
(`1d0b5a6..06fd0f1`). Two directions this file
proposed were withdrawn along
the way, and several of its factual claims have been narrowed or re-cited under
review — §6 records all of them rather than hiding them.

---

## 1. The goal, unchanged

**Measure what the disease scorer actually does before changing it.** The offline
evaluator and the deployed pipeline score different candidate sets with different
formulas, so neither number describes the other. Work item B-0 builds a harness
and an A/B/C/D ladder that isolates **one difference per step**.

Everything below serves that, or is a consequence of discovering that part of it
does not currently work.

**The A and B rungs are retired; the goal is not.** The owner reported on 2026-10-12
that the institution had already authorised retiring the old pipeline's Mode A/B comparison
tools (§3.1; item 9's row), and R2 deleted them the same day. Policy Gate 1 stays, and its B-0 report is kept: it compares the
currently served ranking (Mode D, B-0.5's served behaviour) with the planned new GNN ranking
(Mode C; item 7b).

---

## 2. Established facts

M1-M7 were **measured on deployment hardware**; M8 and M9 were **read from
source**. Both kinds are checkable; neither is recalled or inferred. These are
what changed the ordering.

| # | Fact | Where |
|---|---|---|
| M1 | **No scanned checkpoint carries `metadata` or `in_channels_dict`.** All carry `config`, `data_fingerprint`, `epoch`, `logs`, `state_dict`. The claim is about the **current producers and the scanned family**, not about every checkpoint the project has ever written — no historical audit was run, and none is needed to reject the frozen evaluator as the acceptance oracle. **Confirmed by artifact** over 15 checkpoints in two families, `hgt` (10) and `gat` (5): both watched keys absent in all 15, the four expected keys present in all 15, nothing unreadable | [`EVIDENCE_M1_M3_hgt.json`](EVIDENCE_M1_M3_hgt.json) `39176ea3…`, [`EVIDENCE_M1_M3_gat.json`](EVIDENCE_M1_M3_gat.json) `5473fedb…` |
| M2 | **`in_channels` is 128** in every checkpoint; the frozen evaluator's hardcoded fallback was 256 (`scripts/evaluate_model.py` at `7dab728`, deleted 2026-10-10), which was the size mismatch it died on. **Confirmed by artifact**: read from `feature_encoder.projections.<node_type>.weight` rather than from a config field, 45 projections across 15 checkpoints, every one 128, none without a readable width | same |
| M3 | **The filename number is `val_mrr`** — `model-45-0.6975.pt` carries `logs["val_mrr"] = 0.69754…`. **Confirmed by artifact**: 13 agreements, **0 disagreements**, each compared at the precision its filename was written to; the 2 uncomparable are both `last.pt`, which carries no score. `val_mrr` is the ranking metric in all 15 | same |
| M4 | **100% of validation diseases appear in training** — 7,970 of 7,970. Train: 100,000 samples over 10,576 diseases; val: 15,000 over 7,970. **Confirmed by artifact, every digit**, with both split digests recorded | [`EVIDENCE_M4.json`](EVIDENCE_M4.json) `b3aed32a…` |
| M5 | **SP reachability is dense** — but the recorded figure was wrong and is superseded. Measured on `shortest_paths.pt` `7268900c…`: the **median** phenotype reaches **19,216.5 of 29,866** diseases (**64.3%**) within the configured 5 hops; the **mean** is 16,845.5 (**56.4%**); q1 51.2%, q3 68.9%, max 78.2%; **270 of 19,836 phenotypes reach none**. Dense is a property of the graph, not of one node — q1 is already above half. The recorded "71.3% (16,845 of 23,640)" divided a mean by a denominator from a **different artifact**; see §2.4 | [`EVIDENCE_M5.json`](EVIDENCE_M5.json) `58d79584…` |
| M6 | **The SP lookup breaches the provisional latency budget in 22 of 60 measurements**, worst 3,722 ms, on the real artifact and a GB10 SPARK. Approach A brings that to **0 of 60** at a cost of 3.44 GB permanent residence; the earlier "1.7-2.5x" figure came from a different artifact vintage and host and is superseded | `scorer-measurement/PLAN_B04.md` §12.3 |
| M7 | **Zero duplicate rows** on two independently built artifacts of different HPO vintages — evidence about the generator's invariant, not clearance for one file; every future artifact is protected by the load-time assertion | same, §10.1 |
| M8 | **The trainer's own validation loop is Mode-A-shaped, not Mode-C-shaped** — per-batch subgraph forward, cosine against *the subgraph's* disease rows, top-20 truncation, MRR from the same function Mode A calls. See §2.1 | read from `trainer.py`, `metrics.py`, `measurement.py` |
| M9 | **The standard generated workspace has no held-out test split.** `sample_generator.py:97-111` writes train and checkpoint-selection val only, and `val` is what `early_stopping_monitor=val_mrr` selects on. The measurement tools **can** consume a separately supplied `test` cohort — `--split` accepts it and `read_samples` would load it — but **no accepted protocol currently defines, creates, freezes or proves the independence of one** | read from `sample_generator.py`, `trainer.py:124`, `measure_scorer.py:275-277` |

### 2.1 M8 in detail — where `val_mrr` comes from

Read from source rather than recalled, because it overturns a direction this file
previously proposed.

| Step | `trainer._validate` | Mode A's legacy stream |
|---|---|---|
| Encoder input | `batch_data["subgraph_x_dict"]`, `subgraph_edge_index_dict` — the **per-batch subgraph** (`trainer.py:628-634`) | the same two keys off the same dataloader (`measurement.py:706-710`) |
| Padded phenotype ids | `phenotype_ids.clamp(min=0, max=…)` (`trainer.py:739`) | the same clamp, kept as trainer parity (`measurement.py:738`) |
| Patient vector | masked mean (`:744-751`) | `masked_mean_pool` (`:742`) |
| Candidates | `disease_emb = node_embeddings["disease"]` — **the subgraph's diseases**, not the KG's (`:762-763`) | `cosine_score_matrix(patients, disease_emb)`, same rows (`:743`) |
| Score | cosine, both sides `F.normalize`d (`:761-763`) | `cosine_score_matrix` |
| Ranking | `scores.sort(dim=-1, descending=True)` (`:651`) | `legacy_ranking`, which is that same call (`:151-157`) |
| Truncation | `pred_indices[:20]` (`:656`) | `LEGACY_TRUNCATION_K = 20` (`:243, :749`) |
| Metric | `RankingMetrics().compute_all(...)["mrr"]` (`:663-666`; `metrics.py:285`) | `RankingMetrics().mean_reciprocal_rank(...)` (`:809-810`) |

Every row is the same operation. **`val_mrr` is a Mode-A-shaped number, and M3
says every scanned checkpoint carries it.**

**Every row above holds only when identical model weights, batch/subgraph
objects, ID mappings and numerical execution context are supplied.** It is a
statement about algorithmic shape. It is **not** a claim of historical execution
parity — see §3.1.1.

### 2.2 The D-list: what still has to be recorded or closed

**Most of this is already recorded.** `MeasurementManifest`
(`measurement.py:259-340`) already carries `batch_size`, `shuffle`,
`num_workers`, `negative_sampling_strategy`, `num_negative_samples`,
`subgraph_strategy`, `subgraph_hops`, `num_neighbors`, `max_subgraph_nodes`, the
three seeds, `device`, `torch_version`, `cuda_version`, `dtype`, `amp_enabled`,
`deterministic_algorithms`, `cudnn_deterministic`, `cudnn_benchmark`, `split`,
`n_samples` and `artifact_digests`. Re-specifying those would duplicate fields
that exist. Only the rows below are open. (Since R2, 2026-10-12, the manifest no longer
carries `shuffle`, `num_workers`, the sampler fields or the three seeds, which Mode C does
not use; `batch_size` stays.)

| # | Open item |
|---|---|
| D1 | **Record the limitation, not a field.** Historical epoch RNG state was never saved, so the historical stochastic validation traversal is not exactly reproducible. Sample order in a *new* run is pinned by `shuffle=False` plus the samples digest. Since R2 (2026-10-12) measurement is Mode C alone, which takes patients in the samples file's order with no loader, so the samples digest pins it and no `shuffle` is recorded |
| D2 | **Closed by 1e.** `amp_dtype` and `torch_compile_wrapped` are on `MeasurementManifest`, and the latter was on `DifferentialResult` too — that was the artifact whose regime actually varied, until R2 retired the differential (2026-10-12). **The field is named for what its probe can see**: `isinstance` against the dynamo wrapper proves the object is wrapped, not that a compiled graph ran, since graph breaks and eager fallback leave a wrapped model running eagerly. Renamed from `torch_compiled` in review round 1 for the reason `cuda_executed` was renamed from `calibration_eligible`. Two corrections to this row's own wording: `amp_enabled` was not "recorded", it was **hardcoded `False`** in `build_manifest`; and a structural claim is not an observation, so 1e made `assert_no_autocast` **refuse** to run either traversal inside an autocast block rather than let the manifest describe a run that did not happen. **Superseded by Proposal B**, which removed that ground instead of living with it: the fields now record the regime observed at the computation that produced each mode's numbers, and `assert_manifest_describes_regime` refuses only the case a single field cannot describe — numbers produced under one regime and recorded under another. `observe_torch_compile_wrapper` is one tri-state function, not compile-metadata machinery. **`isinstance` against the imported wrapper class is the only evidence**, aggregated over two import paths (`torch._dynamo`, `torch._dynamo.eval_frame`) so a release that moves one still answers exactly. No class resolves -> `None`. **There is no attribute fallback, and three review rounds are why.** R1: the first version turned "not observable" into `False`, never reached its own documented fallback, and claimed "never raises" while `hasattr` propagates any `non-AttributeError`. R2: the repair over-corrected — `_orig_mod` alone became `True`, letting **any** object opt in by naming a field, and the test written for it froze that false positive as a specification. R3: pairing the attribute with a `torch._dynamo` module prefix did not rescue it either — `startswith` also matches `torch._dynamoevil`, and `__module__` is **assignable class metadata**, so it records what a class says about itself rather than how it was built. `_orig_mod` is a necessary marker, never sufficient evidence, and no arrangement of weak signals makes it so. Every mechanism confirmed by execution, and each superseded version fails cases the next one added |
| D3 | **Closed.** `_remap_indices` (`data_loader.py:956-964`) maps `batch["disease_ids"]` into subgraph-local space, the same space as the score-matrix columns, and trainer and Mode A read the same remapped tensor. The range invariant is enforced at **three independent boundaries** — loader, loss, harness — none of them in the CUDA hot path. See §2.3. **The legal-truth equality test is done (1e)**, and it is not the one 1d already had: both 1d cohorts are identity-mapped (`original_indices["disease"] == [0..n-1]`), under which a side reading the wrong id space produces the right numbers anyway. 1e adds a hand-built batch with a non-identity, non-sorted map so local != global. A trainer emitting global ids is refused there by **`DiagnosisLoss`**, not by the harness — the boundary §2.3 measured, confirmed by execution after the test's first draft predicted the wrong one. **The harness boundary moved when Mode A went:** R2 (item 9's row, 2026-10-12) re-pointed the third boundary to Mode C's `_assert_ids_in_range` (retirement assessment §4; `src/kg/data_loader.py`'s docstring and the two tests that name the boundaries). D3 stays closed and the contract is unchanged |
| D4 | No excluded-sample list is required **while the run is fail-fast**. Record that the run does not silently skip samples and that ranked plus ground-truth-absent account for the intact cohort. `n_ground_truth_absent` is a count, **not** an excluded- or failed-sample list; if skip-and-continue is ever introduced, skipped ids and reasons become required then |
| D5 | Exact artifact identity. Measurement already records the digests of the checkpoint, samples, node features, edge indices and num_nodes, since contract M2.1's S7 from the bytes it parsed (`measurement_input_digests`; A/B/C hashed them by path, `measure_scorer.py:79-92` at the time). Open: the **shortest-path artifact digest, added when Mode D consumes it**, and comparing recorded digests against the institution-approved artifact set at acceptance. No registry, no compatibility database |
| D6 | Aggregate `val_mrr` is insufficient for parity and is a historical sanity reference only (§3.1.1) |

### 2.3 D3 — where the malformed-truth invariant is enforced

The invariant at issue:

> every valid sample's seeded ground-truth disease must remap to
> `0 <= local_truth < number_of_subgraph_disease_rows`.

**How a violation could arise.** The mapping tensor has length
`max(sampled_global_id) + 1`, is initialised to `-1`, and holds local indices
only at sampled global-id positions (`data_loader.py:358-366`). A global id
inside the tensor range but absent from the subgraph therefore maps to `-1`; an
id beyond the tensor length is left unchanged by `_remap_indices`' own guard. In
Mode A the truth is one of the subgraph's seed nodes, so neither should happen —
which is exactly why it needs a guard rather than an assumption.

**Mode A enforced it, and the check was tested** (superseded: R2 retired Mode A and
`to_global_ids` on 2026-10-12; measurement's boundary is now Mode C's `_assert_ids_in_range`, in
global ids, in the table below). `to_global_ids`
raises on both failures *before* any indexing (`measurement.py:91-100`):

```
local id 4 is outside the subgraph's 4 disease nodes   # >= n_rows
local ids must be non-negative                          # the -1 hole
```

`tests/unit/test_measurement_ranking.py:52-56` parametrised exactly `[3]`, `[99]`
and `[-1]` (deleted with `to_global_ids` in R2). A `-1` never reached PyTorch's negative
indexing on this path, so it
cannot silently select the last candidate.

An earlier revision of this file credited `_assert_cohort_is_intact` with this.
That was the wrong citation: that function checks absent canonical truths and
cohort counts, not local-truth range. The conclusion held; the reference did not.

**The trainer refuses too — through the loss, not the metric.** An earlier
revision of this file claimed the trainer silently scored such a truth `0.0`
into `val_mrr`. That was wrong: it traced `_validate`'s metric fragment
(`trainer.py:648-657`) without checking what runs before it. `self.loss_fn(...)`
is called first (`:640`), `MultiTaskLoss` invokes `DiagnosisLoss` whenever
`diagnosis_scores` and `diagnosis_targets` are present
(`loss_functions.py:513-516`), and that reaches the malformed target through
`cross_entropy`, `gather` and `scatter_`. Measured on both branches:

```
label_smoothing=0.0  -1 hole  -> IndexError:   Target -1 is out of bounds.
label_smoothing=0.0  >= n     -> IndexError:   Target 5 is out of bounds.
label_smoothing=0.1  -1 hole  -> RuntimeError: index -1 is out of bounds ...
label_smoothing=0.1  >= n     -> RuntimeError: index 5 is out of bounds ...
```

`mean_reciprocal_rank` *is* permissive in isolation — it scores `"-1"` as `0.0` —
but the path never reaches it. **The primitive is permissive; the path is not.**

So there is no raise-versus-plausible-number asymmetry, only a difference in
where the error surfaces and how it reads.

**The contract is REFUSE, and it is not an open decision.** A malformed local
truth means the subgraph seeding, the id remap, the candidate alignment or the
batch wiring is wrong. Scoring it as a rank miss would convert a data-pipeline
failure into apparent model error and contaminate the loss, `val_mrr`, early
stopping and checkpoint selection.

**What was still worth fixing** — and is now done: the disease gather was
guarded by `disease_ids.clamp(min=0, max=...)`, a *silent correction* sitting
where the refusal belongs. It protected nothing, since the loss rejects the same
ids two statements later, and it made the refusal an accident of which tensor got
passed where — a later edit passing the clamped tensor as `diagnosis_targets`
would have removed the refusal without touching anything that looked
load-bearing. The clamp is gone and the gather uses the original ids.

**The explicit check lives at the loader, not the trainer, and the reason is
performance.** A first version put it in `_compute_model_outputs`, where
`disease_ids` has already been through `_move_to_device`; `bool(t.any())` on a
CUDA tensor forces a **host-device synchronisation on every valid training and
validation batch**. That is a permanent throughput cost to catch a condition that
cannot occur. It now sits in
`DiagnosisDataLoader._assert_disease_truth_in_range`, on the host, immediately
after the remap that creates the failure.

**Three boundaries, each small and local, none shared:**

| Boundary | Where | Covers |
|---|---|---|
| `_assert_disease_truth_in_range` | `data_loader.py`, CPU | the point the `-1` hole is created |
| `DiagnosisLoss` | `loss_functions.py` | any caller that bypasses that loader |
| `_assert_ids_in_range` | `src/evaluation/measurement.py`, Mode C | measurement's own boundary: a ground truth outside the graph's disease nodes is refused before ranking, in global ids, since Mode C has no subgraph (`tests/unit/test_measurement_mode_c.py`, `test_an_out_of_range_ground_truth_is_refused`). Until R2 (2026-10-12) this boundary was Mode A's `to_global_ids` (`measurement.py:91-100`), which checked local ids |

No cross-module validator: three checks of a few lines each keep the coupling
lower than one shared one would. A regression test asserts that **no `bool()` is
taken of the disease-id tensor in the hot path**, so the CUDA sync cannot be
reinstated by a later well-meaning edit.

The **phenotype** clamp at `trainer.py:739` is a different thing and stays:
`diagnosis_collate_fn` pads phenotype ids with `-1` by design and the mask
discards those positions. Disease truths are never padded.

Still to cover, and only reachable once 1c exists: **equality of the legal truth
id consumed by the trainer reference and by Mode A** on the same batch. (Done in 1e, D3's row; moot
since R2 retired Mode A, 2026-10-12, code-reviewed with no P1 or P2 in `1d0b5a6..06fd0f1`.)

---


### 2.4 What M5's recorded figure got wrong

Recorded as *"a typical phenotype reaches 71.3% of diseases within 5 hops (16,845
of 23,640)"*. The measurement that replaced it disagrees in two independent ways,
and the sentence above is retired rather than adjusted.

**The denominator came from a different artifact.** 23,640 appears in
`scorer-measurement/PLAN_B04.md` §9.1 as **"Disease targets"** for
`shortest_paths.pt` SHA-256 `9ada0c1a…` — 19,540 phenotypes, 429,971,678 rows.
The artifact audited here is `7268900c…` — 19,836 phenotypes, 430,585,772 rows —
and its graph holds **29,866** diseases, agreed between `num_nodes.json` and the
producer's own sidecar. Whatever "Disease targets" counted, it is not this
graph's disease count, so 71.3% is a ratio across two artifacts.

**What "Disease targets" counted is not established here, and nothing needs it.**
It may be an earlier vintage's disease count or the number of diseases actually
present in that table; this run does not say, and M5's schema deliberately does
not record a second denominator. Item 3 — the `DISEASE_SCORER_POLICY.md` §3.5
correction — needs only that reachability is *dense*, and that holds under either
reading: the median phenotype reaches 64.3% of the graph's diseases and q1
reaches 51.2%.

**"Typical" was a mean.** Mean 16,845.5 and median 19,216.5 differ by 2,371
diseases, 7.9 percentage points. The word chosen implied the second and the
number reported was the first.

**A prediction recorded in §5.2 of the previous revision was wrong, and is
corrected there.** It said the recorded figure was probably an overestimate
*because* phenotypes reaching no disease have no rows and would be dropped by a
count taken over the table. The mechanism is real and the direction was right,
but the magnitude is not: **270 of 19,836 phenotypes (1.36%)** reach nothing, far
too few to move a median by fifteen points. The denominator is the cause.

**Artifact vintages differ across the facts.** M6 and M7 were measured on
`9ada0c1a…`; M5 here is on `7268900c…`. Both are recorded by digest so the two
are never silently compared.

**One property re-confirmed in passing.** M7 recorded zero duplicate
`(phenotype, target)` pairs on two independently built artifacts. This is a
third, of a newer vintage: **0 duplicate pairs collapsed** out of 334,147,192
disease rows.

**The run's resource cost is recorded separately and more weakly.**
[`EVIDENCE_ITEM10_run_metrics.json`](EVIDENCE_ITEM10_run_metrics.json)
`305fb1a2…` holds the wall time, peak RSS, major page faults and bytes read
for the four runs. It is **transcribed by hand** from `/usr/bin/time -v` rather
than emitted by a committed script, so it is a weaker class of artifact than the
four beside it and says so in its own text. The console transcripts it was taken
from are not committed: §5.2 forbids absolute paths in evidence, and both the
`.out` and `.time` files carry them.

Two things in it are worth more than operational curiosity. The shortest-path scan
read **10.78 GB** from the filesystem against an artifact of **10.76 GB** — that
the whole table was read is established by disk traffic, independently of the
sidecar's `num_pairs` check, which concludes the same thing from tensor length.
And it took **21,846 major page faults** against 0-8 on the three runs that read
no large artifact, which is what a mapped read produces and a resident read does
not.


### 2.5 A defect that only the deployment hardware could show

Found while running the full suite on the institutional machine after item 10's
evidence run — not by looking for it.

`tests/unit/test_retrieval.py` reported **4 skips** there against **3** on a CPU
container. The extra one is `TestFactoryFunctions::test_create_index_auto`, and
the direction is backwards from the usual: a test that *passes* without CUDA
*skips* with it.

**The mechanism — and the seam this file first named wrongly.** The visible
symptom is in `resolve_backend`: it resolves to `cuvs` when `is_gpu_available()`
is true, and the block beneath it, commented *"Verify it can actually be
instantiated"*, validates only whether the platform is Windows. That description
is accurate and it is not where the defect was.

`_register_backends` guarded on whether a backend's **module** imports. Neither
backend imports its dependency at module level — `CuVSIndex` imports
`cuvs.neighbors` and `cupy` inside `_validate_cuvs`, `VoyagerIndex` imports
`voyager` inside a method — so that guard never fired for **either** backend and
every backend was registered on every host. `resolve_backend` then behaved
correctly given a registry that was wrong: `"cuvs"` was in it, so it was returned,
and the `["cuvs", "voyager"]` fallback beneath never ran because nothing raised.

Registration was the seam because selection, fallback *and* the operator report
all read the same registry: `scripts/build_index.py` printed
`Available backends: ['voyager', 'cuvs']` on a host with neither cuvs nor cupy.

**The test converted that into green.** It caught `ImportError` and skipped with
*"No vector backend available"* — false on that machine: voyager was installed,
and `test_create_index_voyager` passed in the same run.

**What this is and is not.** Not on the clinical path: the vector index was
detached from diagnosis by institutional decision and the
`diagnosis-retrieval-detachment` contract still enforces that. It is on the path
of the planned natural-language / vector-mapping work, and it was invisible on
every machine except one with CUDA and without cuVS — which describes the
deployment target and no CPU development box.

**It also contradicted a completed stage.** Stage 5 was *honest cuVS reporting and
orphan removal*. Its result lives in `scripts/validate_installation.py`, whose
comment forbids claiming availability on the strength of `import cuvs` alone —
*"a combination observed on a real deployment machine"*. That honesty was applied
there and not here, so two files answered the same question differently on the
same host. `_is_cuvs_available` was left behind by the same stage with zero
callers.

#### What was fixed, and what was deliberately left

Registration now asks whether each backend's **immediate required packages are
discoverable** — `voyager` for Voyager, `cuvs` **and** `cupy` for cuVS, since
`_validate_cuvs` imports both and cuvs-without-cupy is the state already observed
on a deployment machine. cuVS stays unregistered on Windows whatever is
discoverable; Voyager is gated on every platform. The registry is rebuilt rather
than accumulated, and cleared in place rather than rebound so a caller holding it
is not left reading a detached copy. Discovery fails closed: `find_spec` raises
for ordinary conditions, and an optional-dependency probe must never be the
reason `import src.retrieval` fails. `_is_cuvs_available` is deleted.
`resolve_backend` is unchanged.

**Discovery is the ceiling of the claim, on purpose.** It says the package is
findable. It does not say the package imports, that a CUDA build matches the
driver, that an index can be constructed, or that a search returns anything.
`validate_installation.py` remains the place where the deeper states are
distinguished, and this does not duplicate it into the import path.

Three limitations, recorded rather than discovered later:

  - **Discoverable is weaker than importable.** A package that is findable but
    raises on import still registers, and still fails at construction.
  - **Availability is inferred by the registry, not answered by the backend.** The
    structural fix is a `VectorIndexBase.is_available()` classmethod so each
    backend states its own contract. That is a design change to a detached
    subsystem and belongs with the natural-language retrieval integration, when
    its real support contract is known — not here.
  - **The probe logic still lives in two places**, here and in
    `validate_installation.py`. Consolidating them is worth doing and is not this
    fix.

## 3. What these facts broke

### 3.1 The calibration target did not exist — the root blocker, now decided

Two loaders fail on the same missing key, in two different ways:

- `build_legacy_mode_a_model` indexes `checkpoint["metadata"]` and
  `checkpoint["in_channels_dict"]` directly (`measure_scorer.py:189-190`).
  Neither exists (M1) — **`KeyError`**.
- `create_model_from_checkpoint` uses `.get()` with defaults instead
  (`evaluate_model.py:138-148` at `7dab728`): three node types, two edge types,
  256 channels. It therefore builds a **structurally wrong model** and dies later
  in `load_state_dict` against a real 128 (M2).

**Mode A cannot execute on any checkpoint in the scanned family**, and neither
could the frozen evaluator.

**Everything downstream inherits this.** B-0.2 and B-0.3 are implementation-
complete but their acceptance was always "the harness reproduces the oracle".
Without an oracle run, the harness is unvalidated — and B-0.5 would build a
statistical protocol on top of it.

#### What Mode A is actually carrying

Mode A has been doing **two** jobs under one name, and only one of them is broken:

- **(a) the bottom rung of the ladder** — per-batch subgraph encoder, subgraph
  candidates, cosine. A→B isolates encoder scope only if A holds this shape.
- **(b) bit-parity with the frozen evaluator** — the clamp, the truncation at 20,
  the local-index prediction artifact, and the oracle-mirroring model builder.

Job (b) is what requires `checkpoint["metadata"]`, and job (b) is unexecutable.
Job (a) needs no oracle at all.

#### Directions, corrected by M8

| Direction | Verdict |
|---|---|
| Repair the frozen evaluator to derive `in_channels` from features | **Rejected.** It is frozen on purpose, and a repaired evaluator is a *new* evaluator, not a reproduction of a historical executable artifact |
| Accept that no oracle number exists and drop Mode A | **Rejected** at the time: it discards the control the whole ladder is built around. **Superseded by the owner's clarification of 2026-10-12:** the owner reported that the institution had already authorised retiring the old pipeline's Mode A/B tools, so Mode A was retired with Mode B by R2 (item 9's row; 2026-10-12). The ground is current use, policy and the live tests R1 already moved, not a claim that Mode A never had scientific use. Policy Gate 1 stays (item 7b) |
| ~~Calibrate **Mode C** against the trainer's `val_mrr`~~ | **Withdrawn — premise false.** The trainer's validation loop scores the batch's subgraph, not the full disease matrix (M8, §2.1) |
| ~~Calibrate **Mode A** against the stored `val_mrr`~~ | **Withdrawn — not reproducible, and insufficient even if it were.** See §3.1.1 |
| **Same-batch differential calibration against the trainer's own validation pass** | **Adopted.** §3.1.2 |

#### 3.1.1 Why the stored `val_mrr` is not an oracle

Two independent reasons, the first confirmed from source:

- **The historical batching cannot be reconstructed.** Mode A's candidate set
  *is* the batch's subgraph, and validation subgraphs and negatives are drawn
  stochastically. `grep -n "rng\|get_rng_state" src/training/callbacks.py
  src/training/trainer.py` returns **no matches** — the checkpoint saves epoch,
  `state_dict`, `optimizer_state_dict`, `logs`, `config`, optionally
  `scheduler_state_dict` and optionally `data_fingerprint`, and no RNG state. The
  candidate universe at epoch 45 depended on the RNG stream position after 45
  epochs of training, which is unrecoverable.
- **One aggregate scalar cannot establish parity anyway.** Per-sample ranking,
  candidate-set, tie-order and ID-mapping disagreements can cancel in aggregate.

Stored `val_mrr` is retained as a **historical aggregate sanity reference** and
nothing more.

#### 3.1.2 The adopted direction

Retire frozen-evaluator bit parity; keep Mode A as the trainer-validation-shaped
bottom rung; calibrate it with a **same-batch differential test**: build the real
model through `build_shepherd_model`, hand the *same* batch and subgraph objects
to a trainer-path reference calculation and to the Mode A harness, and compare
per-sample local top-20 ids, truth ranks, reciprocal ranks, then aggregate MRR.

**The reference is the trainer's own code, not a third implementation.**
`Trainer._validate` (`trainer.py:615-681`) and `Trainer.evaluate`
(`:773-849`) already duplicate this calculation; the shared pass is extracted
once, privately, and all three callers use it. Writing a fourth expression of a
calculation that must agree would leave the newest copy the only untested one.

**Non-tautology is enforced by the layer contract, not by discipline.**
`.import-linter.ini` orders `src.evaluation` above `src.inference` above
`src.training`, and a lower layer may not import a higher one — so
`src/training/` **cannot** import `masked_mean_pool` or `cosine_score_matrix`
from `src/inference/scoring.py` even if someone wanted to. `make lint-imports`
fails if it is attempted. The trainer keeps its own inline `F.normalize` +
`torch.mm` (`trainer.py:761-763`), which is what makes it a genuinely
independent expression. Review permitted primitive sharing; the architecture
forbids it in this direction, and the stricter rule is the one that holds.

The differential harness itself sits in `src.evaluation`, which is above both,
so it may import the trainer's extracted pass and the Mode A traversal. That
direction is legal and is the only one that needs to be.

**Its cost, stated plainly:** job (b) is retired. The legacy-removal checklist in
`scorer-measurement/README.md` was written around a parity that will now never be
demonstrated, and most of what it lists for deletion is machinery the replacement
keeps. That checklist is **corrected and suspended**, not executed.

**Superseded in part by the owner's clarification of 2026-10-12.** The owner reported on
2026-10-12 that the institution had already authorised retiring the old pipeline's Mode A/B
comparison tools, and that the old pipeline may be modified or rewritten entirely. Mode A as
the bottom rung, and the differential calibration that checks it, were therefore retired
by R2 (item 9's row; 2026-10-12) rather than kept, and 7a is cancelled. The basis is
current use, policy and the live tests R1 already moved to their use points, not a claim that
Mode A never had scientific use. The direction above stays as the record of what was decided
and why.

#### 3.1.3 What 1d found: bit-exactness is a contract only when AMP is off

§3.1.2 specified the comparison but not the precision it runs at, and the two
paths do not agree about that.

- `Trainer._run_evaluation_pass` forwards inside
  `autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp)`.
- The Mode A traversal in `run_modes_ab` has **no autocast at all**.
- `trainer.py:380` resolves `use_amp = config.use_amp and device.type == "cuda"`.

So the answer depends on the device, and not by a little:

| Where | `use_amp` | What a comparison establishes |
|---|---|---|
| **CPU** — the bounded tests | always **False**, whatever the config says | **bit-exact agreement.** Both paths are fp32 and every disagreement is a real one |
| **CUDA** — item 7a's institutional run | **True** by default, `float16` | the trainer's scores differ from Mode A's in the last bits by construction. Anything near a tie may reorder |

**A CUDA disagreement is therefore a measurement, not a fault**, and the harness
must not be built to refuse it — refusing would make AMP's effect on the ranking
the one thing the calibration cannot observe. `DifferentialResult` records the
**resolved** `amp_enabled` / `amp_dtype` beside the verdict and exposes
`bit_exact_contract`, which says which of the two questions the run answered.

**This promotes item 1e.** `amp_dtype` was listed as a manifest tidiness item; it
is the field that says which of two comparisons a recorded number came from — an
exact comparison at **equal** precision, or an exact comparison of discrete ranking
artifacts computed at **unequal** precision. Both are exact; they are not the same
question. Neither is a tolerance comparison, and an earlier revision of this
sentence called the second one that, which was wrong twice over: no tolerance
exists anywhere in the harness, and the phrase invited a reader to expect a
threshold that is deliberately absent. A calibration artifact without `amp_dtype`
cannot say which of the two it is.

**What the bounded tests consequently cannot do**, stated so item 7a does not
inherit it silently: they cannot ask the CUDA question at all. The container this
work is done in has no CUDA device. The CPU result is a genuine and necessary
precondition — the two implementations agree when precision is held equal — and
it is not a substitute for the institutional run.

**And a gap item 7a must close before it runs, found on audit rather than in
review.** On CUDA the trainer's defaults are `use_amp=True, amp_dtype=float16`, so
a differential run left at its defaults compares an autocast trainer pass against
an fp32 Mode A pass. **Two legs, and 7a must say which one its acceptance turns
on rather than inheriting a default:**

| `TrainerConfig` | What the run answers | Who decides the criterion |
|---|---|---|
| `use_amp=False` | do the two implementations agree exactly | **engineering** — `agreed is True`, the same hard parity gate as on CPU |
| `use_amp=True` (the CUDA default) | does autocast reorder anything, on this model, cohort and device | **the institution's experimenters** — see below |

**A first draft of this section said an AMP-on run reports `agreed=False` "by
construction". That was wrong and is corrected here.** Nothing in the harness
compares score bits. The comparison is over discrete ranking artifacts — the
top-`K` rows, the truths, the reciprocal ranks and the aggregate — so a precision
difference registers only where it actually *reorders* something. A run in which
fp16 reorders nothing agrees exactly. There is likewise **no numerical tolerance
anywhere in the harness**, so an AMP-on result must not be described as a
"tolerance observation", which an earlier revision also called it. It is an exact
comparison of discrete artifacts computed under unequal precision regimes.

**The AMP-on criterion is not engineering's to set, and this is a decision, not a
deferral.** Whether autocast changes a diagnosis ranking enough to matter is an
empirical question about a particular model, cohort and device, and the acceptable
answer is a clinical judgement. Neither can be predicted from this side. The
institution's position is that **the switch stays a switch**: the hospital's
experimenters configure it, run it both ways if they wish, and record what they
observe.

So engineering's deliverable is the **evidence, not the verdict**, and it is
already in place — `DifferentialResult` carries the resolved `amp_enabled` and
`amp_dtype`, `bit_exact_contract` for which question was asked,
`n_samples_disagreeing` and its rate, the affected rows with their sample ids and
both rankings, and `mrr_absolute_difference`. Those are the quantities a
disagreement criterion would be written in terms of. **No threshold is hardcoded
and none should be.**

**Superseded by the owner's clarification of 2026-10-12.** The owner reported on 2026-10-12
that what the owner and the institution want to know about AMP is whether an AMP-*trained*
model is worse — the accuracy or quality lost against the large training-time saving some GNN
architectures get — which is a training study. It is not the same-checkpoint scoring comparison
with autocast switched that this section and 7a's AMP leg describe. That scoring evidence is
therefore no longer a precondition for deleting the differential or Modes A/B (retirement
assessment §8 item 7, closed), and 7a, its AMP leg included, is cancelled. The AMP training
study is recorded separately as bounded research (item 21), not as an M2 or R2 prerequisite.
This section stays as the record of what the differential established.

### 3.2 Validation measures no unseen-disease generalisation at all

M4 is the maximal case: **every** disease in val appears in train. The escalation
already sustained by review said the generator "permits" overlap; the measurement
says it is total.

This does not invalidate any engineering result. It **bounds what every number
may claim**, and the bound is tighter than the caveat currently drafted.

**Fixed under 11i, and `EVIDENCE_M4.json` is now a baseline rather than a
description of the present.** Generation consumes a disease allocation, so the
partitions are disjoint by construction. The M4 figure describes the regime that
produced it and is kept unchanged for that reason; the audit that produced it
gained a `--require-disjoint` gate and a cross-check of the sample files against
`split_manifest.json`, which is the one claim the generating process cannot make
about itself. Two consequences are **not** closed by the fix and are recorded
rather than assumed away: every checkpoint in the repository was trained under
the old regime, and a disease-level cut does not close the channel where one
phenotype set carries two different labels across it — `scripts/audit_generator_fidelity.py`
measures that.

### 3.3 A policy inference is contradicted — in its premise, not its conclusion

`DISEASE_SCORER_POLICY.md` §3.5 recorded, explicitly as unmeasured, that most
candidates fall outside the 5-hop table so the SP term degenerates to a
reachability indicator.

**The premise is refuted, per phenotype.** The median phenotype reaches 64.3% of
diseases within the configured 5 hops, q1 51.2%, max 78.16%; 270 of 19,836
phenotypes (1.36%) reach none at all.

**1.36% sizes phenotypes that reach nothing — not unreachable candidates.** Using
it for the second would understate them by orders of magnitude: the median
phenotype still leaves ~10,650 diseases unreachable and the best-connected leaves
~6,522. Unreachable pairs are common at every phenotype, and M5 does not size them
as a fraction of pairs.

**M5 does not reach the patient level at all.** The deployed scorer takes no union
over a patient's phenotypes and does not use the nearest reachable one: for each
candidate it iterates over every phenotype, gives each unreachable pair
`unreachable_distance`, and averages (`src/inference/scoring.py::sp_mean_distances`).
A per-phenotype distribution therefore cannot be carried across to what a patient
sees, in either direction.

**The conclusion is not refuted with the premise, and saying so would be the same
error in the other direction.** A candidate whose *mean* distance is 5 scores `1/6`
against a floor of `1/7` — one phenotype's five-hop path does not produce that
score by itself — so near-degeneracy could still arise from the distribution of
mean distances. Neither the reachable-distance distribution nor the patient-level
mean-distance distribution is measured, and no current artifact carries either.

The correction is factual and touches no normative statement. It also lands in
`SP_SCORE_GUIDE.md`, which restated the same claim for operators in two places —
one of them saying in as many words that the size of the group "has not been
measured on this deployment".

### 3.4 No accepted holdout protocol, and nothing owned that

M9. The mechanical guards landed and work: `--split` is required with no default
on the measurement entry point and `read_samples` lists what exists rather than
substituting. Those stop a number from being produced *silently* on the wrong
split.

**What is missing is a protocol, not a capability.** `--split` already accepts
`test` and `read_samples` would load a supplied `test_samples.json`. What no
document defines is which cohort that should be, who creates it, how it is
frozen, or what makes it independent. So the honest statement is not "the project
cannot measure held-out data" — it is that **nothing has decided what held-out
would mean here.**

**And the CLI help names only one of the two contamination kinds.** It states
checkpoint-selection contamination — `val` is what `early_stopping_monitor`
selects on — and says a test split exists only where an evaluation protocol
created one. It does **not** mention the measured train/val disease overlap; that
caveat is **item 2** and is blocked on item 10's M4 evidence. The figure must not
reach user-facing help ahead of its evidence.

Until M9 was written down this appeared only as a parenthetical in item 6 and one
line in `scorer-measurement/README.md`. **No item owned it**, which is how a
known problem becomes a forgotten one.

#### The unit of holdout has to be decided before any split is created

Item 11 is not "add `test_samples.json`". Three different things get called
held-out and they support different claims:

| Unit | What it can support | What it cannot |
|---|---|---|
| **Held-out sample views** over diseases already in training — a fresh post-training synthetic cohort | independence from *checkpoint selection*; robustness to phenotype dropout | **not** unseen-disease generalisation: it shares every disease with training |
| **Disease-disjoint** evaluation | unseen-disease generalisation | requires **retraining** — it cannot be produced for an existing checkpoint |
| **External clinical cohort** | independence of origin, and **patient-level external evaluation** on its own terms | **unseen-disease** claims, until its disease overlap with training is measured and reported. Overlap decides whether it *also* supports that claim — it does not decide whether the cohort supports any result at all |

**Item 11's first job is deciding which claim each phase needs**, then choosing
the unit that supports it. Choosing a split first and asking what it proves
afterwards is how a contaminated number acquires a clean-sounding name.

#### One decision, three consumers

| Waiting on it | How it appears there |
|---|---|
| item **8a** | B-0.5's protocol and output-contract design |
| `scorer-retraining/README.md` §5 | gate "a fixed evaluation protocol, cohort and split" — **not defined** |
| any future model comparison offered as generalisation evidence | — |

Decide it once. Deciding it three times is how they end up disagreeing. The
decision lives **here**; the other two carry a pointer, not a copy.

#### What item 11 does *not* block

It does **not** block trainer characterization (1b), the extraction (1c) or
differential-calibration *correctness* (1d). A contaminated `val` cohort remains
usable for **describing the deployed scorer**, which is what work item B-0 exists
to do — provided no held-out or generalisation claim is attached to the result.

### 3.5 What item 5a does and does not depend on

Recorded because "independent of the calibration line" was true of the
*implementation* and got read as true of the *acceptance*.

| | Depends on |
|---|---|
| **5a implementation** — wiring A into the loader and the primitive | nothing in B-0. Not calibration, not the split protocol, not the authoritative-checkpoint decision |
| **5a acceptance** — §13's integrated memory and reload gate | a **designated loadable checkpoint**, plus compatible graph and SP artifacts. A complete pipeline cold start with a resident model cannot be measured without a model |

**"Designated loadable" is not "authoritative".** 5a does not wait on item 6's
clinical decision; it needs *a* checkpoint that loads against the artifact set,
which is a far weaker requirement. If the finally deployed model has a materially
different architecture or memory footprint, compatibility is confirmed **for that
model** — that is a re-run of the gate, not a blocker on it now.

No checkpoint registry. One designated file, named in the 5a plan.

---

## 4. Ordered backlog

Ordered by **dependency**, not by size. An item may only start when everything it
depends on is resolved.

| # | Item | Depends on | Owner | Size |
|---|---|---|---|---|
| **1** | **The calibration decision** — §3.1.2, adopted and reviewed | — | **decided** | — |
| **1a** | Correct and **suspend** the legacy-removal checklist | 1 | author | **done** |
| **1a2** | **Malformed-truth invariant** (§2.3) — remove the silent disease clamp; enforce the range at the **loader** boundary on CPU, not in the trainer hot path where it would sync CUDA every batch. **Not a decision gate**: the contract is REFUSE and all three boundaries implement it (the third, measurement's, is Mode C's `_assert_ids_in_range` since R2; §2.3) | 1a | author | **done** |
| **1b** | Characterization tests freezing `Trainer._validate` / `Trainer.evaluate` observable behaviour — 32 tests. Shared-pass behaviour is driven through **both** entry points; caller-specific contracts stay separate. Each contract group mutation-checked against a representative defect (9 mutations). Found that the malformed-truth refusal has **two independent sources**, and which fires depends on the sign of the bad id — so the tests run both signs | 1a2 | author | **done** |
| **1c** | Extract the pass those two already duplicate — private, narrow. Behaviour-neutral: the 32 characterization tests pass **unchanged**, and each shared operation went 2 occurrences to 1 | 1b | author | **done** |
| **1d** | Same-batch differential calibration — `src/evaluation/differential.py`, 20 bounded tests. One materialised batch list to both paths; per-sample top-20, truth, reciprocal rank, then aggregate MRR. Five legs mutation-checked (scoring, pooling, truth, truncation, aggregate-only). Review round 1 upheld three findings — the aggregate did not gate the verdict, the supplied-`mode_a_result` seam let an acceptance gate accept evidence about itself, and the import contract was directional; two further sub-requests were declined with citations (§6). Found that **bit-exactness is a contract only when AMP is off** (§3.1.3), and that the shared synthetic cohort is too narrow to exercise the truncation at all — so the fixture gained size parameters and a second, wider cohort | 1c | author | **done**; superseded by the owner's clarification of 2026-10-12 — the differential it built was retired with Modes A/B by R2 (item 9's row; 2026-10-12), its live guarantees having moved to direct tests in R1. Kept as history |
| **1e** | D2 manifest additions and the legal-truth equality test (§2.3) — **done**. `amp_dtype` + `torch_compile_wrapped` on the manifest and on `DifferentialResult`; `assert_no_autocast` turned the manifest's `amp_enabled=False` from a structural claim into an enforced one, in both traversals — **later superseded by Proposal B**, which records the observed regime rather than forbidding a non-default one. Legal-truth equality tested under a **non-identity** id map, which is the case 1d's identity-mapped cohorts could not distinguish. Found that `build_manifest` had `amp_enabled` **hardcoded**, and that the wrong-space mutation is refused by the loss rather than the harness | 1c, 1d | author | **done** |
| **P** | **Configurability and provenance** — `PLAN_CONFIGURABILITY_AND_PROVENANCE.md`, both proposals **done and approved**. **A**: `training_input_digests` on the checkpoint, a sibling of `data_fingerprint`, covering the semantic roles a run consumed including a resume parent that was actually loaded; `file_sha256` moved to `src/utils/fingerprint.py`. **B**: the AMP regime is **recorded at the computation that produced each mode's numbers** rather than forbidden — `EncodedGraph` carries the embeddings with their regime into the B/C traversals, and the invariant enforced is `encoded.regime == manifest.regime == scoring regime`, checked per batch. Capture only: no current-workspace comparison, no registry, no AMP CLI, no threshold | — | author | **done** |
| **2** | Contamination caveat updated to the measured 100% (§3.2). Both measurement entry points' `--split` help now names **two** independent reasons `val` is not held-out — checkpoint selection, and that `sample_generator.py` slices one shuffled pool and never partitions by disease, so the overlap is structural rather than incidental to one workspace. The figure cites `EVIDENCE_M4.json`, which carries both split digests, rather than inlining hashes that are workspace-specific into help shown for any workspace | 10 — satisfied | author | **done** |
| **3** | `DISEASE_SCORER_POLICY.md` §3.5 corrected (§3.3), and the same claim where it was restated in `SP_SCORE_GUIDE.md`. **The premise is refuted per phenotype; the conclusion is not, and neither is anything at the patient level** — see §3.3 | 10 — satisfied | author | **done** |
| **4** | Reply to the sustained-with-narrowing contamination review | 2 | author | text only |
| **5** | **B-0.4 prototype phase** — both prototypes measured on the real artifact, twice; approach A selected for the primary GB10 platform | — **independent of 1 and of 10** | author | **measurement complete and reviewed** |
| **12** | **`evaluate([])` silently evaluated the val set** — `test_dataloader or self.val_dataloader` (`trainer.py:869`), so an explicit empty list was falsy and became a full validation pass: a caller asking for one cohort received numbers about a different set of patients. Selection is now `is None`. The other half was equally silent — `mean_loss` divides by `max(num_batches, 1)`, so an empty traversal returns `{"loss": 0.0}`, a statement about a model made without reading a patient — so an empty cohort is **refused**, matching `RankingMetrics`, `measurement.py` and `differential.py`, which already decline to invent a value for one. Checked after the traversal, so an empty generator is caught as surely as an empty list. The 1b test that froze this as *observed, not endorsed* said the fix had to be made by editing it rather than by an extraction nobody re-read; it was | 1c | author | **done** |
| **13** | **Backend registration did not check dependencies**, so `auto` could select an unusable backend, the fallback chain beneath it could never run, and `build_index.py` printed backends the host did not have. Affected **both** backends: neither imports its dependency at module level, so the module-import guard never fired. Not on the diagnosis path — retrieval is detached by decision and the contract holds. See §2.5 | — | author | **done** — registration gated on discoverable dependencies; three limitations recorded, not closed |
| **5a** | **B-0.4 productionisation** — A becomes **the** shortest-path primitive rather than one of two: one builder over raw columns, one sort, the scan retired from the served path and kept only as an independent test reference. The plan it needs exists and is **revised and cleared to implement**; the blanket "no `src/inference/` edit" gate it carried is removed and the reason is recorded in its status block. Its *implementation* depends on no calibration, split or checkpoint decision; its **acceptance does need a designated loadable checkpoint** plus compatible graph and SP artifacts — see §3.5 | 5; acceptance also needs a loadable checkpoint | author + institution | **implemented, not accepted** — §10 steps 0-7 done and reviewed; step 8: the supplementary SP-only figure is recorded at deployment scale (plan §6.2.1 — 9.00 bytes/row, 3.87 GB, on the real 429,971,678-row artifact, confirming §6.2's extrapolation; it yields **no serving peak**, because the benchmark does work before and around the build that the service does not), the measurement subject is **designated** (plan §7.1.1: an HGT checkpoint trained on the homelab against the artifact set, checked to load with the SP table bound, evidence in `scorer-measurement/subject-hgt-2026-09-29/`), and the procedure for readings 1, 2, 3b and 4 is reviewed (§7.3). **Readings 1, 2, 3b and 4 are taken and complete by §7.3's rule** (plan §7.4; three repeats on the designated subject by `scripts/measure_served_pipeline.py`, evidence `scorer-measurement/readings-hgt-2026-09-30/served_readings.json`, `5d509ba4…`); taking them is not passing a capacity gate, and §13 reading 5 is still blocked for want of the smallest supported target |
| **5b** | **SP artifact publication and binding** — the two defects `PLAN_B04_PRODUCTIONISATION.md` §8 left open: a tensor and a sidecar written with nothing recording that they belong together, and `--kg-path` / `--output-dir` independent so an artifact can land in a workspace it was not computed from. The consumer can now refuse a malformed table; neither check asks whether this is the *right* table. Plan: [`PLAN_SP_ARTIFACT_INTEGRITY.md`](PLAN_SP_ARTIFACT_INTEGRITY.md). **Needs no checkpoint and no institutional input**, which is why it runs ahead of 5a's remaining readings rather than behind them | — | author | **implemented** — producer, shared reader and all three consumers in one change; acceptance conditions tested and mutation-checked |
| **11** | **Decide the evaluation-holdout protocol** (M9, §3.4). **Institutional input received and recorded** in [`EVALUATION_COHORTS.md`](EVALUATION_COHORTS.md): three cohorts with three different jobs, MyGene2 for research comparison and the institutional offline cohort as the acceptance benchmark. §5's two gating answers — no direct synthetic unseen-disease result for the deployed weights, and option B with no refit — are **confirmed by the institution** (reported 2026-10-02). What remains open is the rest of that document's §5 — how many diseases to reserve and from which strata, whether to apply for the larger in-hospital database, and whether the generator is faithful enough. A **protocol decision**, not a code fix. Blocks 8a, `scorer-retraining` acceptance, and any held-out or generalisation claim. Does **not** block 1b/1c/1d | 2, 10 | needs review | design question, **partly answered** |
| **6** | Which checkpoint is authoritative. Engineering supplies hashes, logs, artifact-compatibility evidence and load results; the **institution decides**. The question must separate the *deployed* checkpoint from the one `select_checkpoint_in_dir` picks by the highest **contaminated** `val_mrr` — `model-22` winning that metric makes it neither clinically authoritative nor a held-out-generalisation winner | 2, 10 — **satisfied**, `EVIDENCE_M1_M3_hgt.json` and `EVIDENCE_M1_M3_gat.json` | institution | question, **unblocked** |
| **7a** | Engineering differential calibration run. **Its entry point will be new code.** By the owner's rule of 2026-10-09, contract M2.1's S0 (accepted at plan level in the re-review of `0708789` and, as revised, in the incremental review of `7e1af96`; extended since by the owner's decision of 2026-10-10 to remove `scripts/evaluate_model.py` and its pin test, contract M2.1 decision 6, accepted at plan level in the review of `45d9dea`; implemented 2026-10-10 and code-reviewed with no P1 or P2 at `3433bf8`) removed the superseded launcher `scripts/calibrate_mode_a.py` instead of rewriting it. 7a's runner is built on `compare_trainer_against_mode_a` (`src/evaluation/differential.py`) and records, beside its result, the digests of the bytes it read (through contract M2.1's readers), the software revision and the deployment host | 1d, 10 — **satisfied**, and every scanned checkpoint carries a `data_fingerprint`; D5 artifact set; a designated loadable checkpoint | author | **cancelled — superseded by the owner's clarification of 2026-10-12.** The owner reported that the institution had already authorised retiring the old pipeline's Mode A/B comparison tools, so the differential was retired with them (R2, item 9's row; 2026-10-12) and no run certifies it; and that the institution's AMP question is a training study (item 21), not this run's AMP leg (§3.1.3). Its runner was never written. Kept as history; was: blocked on 1d and the checkpoint designation only |
| **7b** | **Institutional measurement for Policy Gate 1's B-0 report** — rewritten by the owner's clarification of 2026-10-12, without Modes A/B and without 7a. Gate 1 stays and its committed comparison is kept: the currently served ranking (Mode D, B-0.5's served behaviour) against the planned new GNN ranking (Mode C). The served side is B-0.5's Mode D measurement (8b; retirement assessment §8 item 4). Retiring the A/B tools does not cancel this evaluation duty. Its other preconditions carry over: item 6, deployment CUDA verification, and Mode C's peak memory on institutional CUDA. Was: Institutional measurement (B-0.2 / B-0.3), depending on 7a, 2, 3, 6 and deployment CUDA verification | 8b, 6, deployment CUDA verification; 2, 3 — satisfied | both | blocked |
| **11i** | **Item 11's implementation arm** — the ordered work in [`EVALUATION_COHORTS.md`](EVALUATION_COHORTS.md) §6.6, tracked here so "where are we" is answerable from one table. **1** correct the document — done, revision 20. **2** split feasibility audit (`scripts/audit_split_feasibility.py`) — **done**, evidence committed as `EVIDENCE_split_feasibility_homelab.json` (schema 2, identical-sibling, KG `6889ed11…`). Worst validation-representation risk across four stratifications is a 69-disease band at 2.9% (f = 0.05) and 0.0013% (f = 0.15), so **stratified allocation is not built** and `disease_allocation.py` no longer calls that reading provisional. The figures belong to one MONDO vintage; a run on the institute's graph is a second vintage, not a second machine. **3** partition before generation (`src/kg/disease_allocation.py`, and `scripts/audit_split_overlap.py` reframed from a measurement into a gate) — done. **4** characterise the generator (`scripts/audit_generator_fidelity.py`) — done for the half needing no external tool; the upstream-simulator comparison is **deferred with its reason recorded** (§6.6). **5** evaluation records beside the checkpoint (`src/evaluation/sidecar.py`) — done. **6** UI — **out of this project's scope**, so the arm is complete at step 5. **Old paths removed, not kept beside the new ones**: a workspace cut before the allocation step is refused at every entry point, and the cohort kind (generated / supplied) is stated by the caller rather than inferred from whether a manifest exists — an inference that could not tell a legitimately manifest-free supplied cohort from a pre-allocation defect. Refusal means **verified**, not present: `verify_cohort_reads` (the read form that replaced `verify_generated_cohorts` in M2.1's S9) binds the manifest to the digests of the sample files a run read, to the disease sets recomputed from those records, to its own allocated digests and to disjointness. The one exception it named rather than glossed — `scripts/evaluate_model.py`, behaviourally frozen as Mode A's calibration reference with no preflight, its bytes pinned by `tests/unit/test_frozen_evaluator.py` — is closed: both were deleted by the owner's decision of 2026-10-10 (contract M2.1 decision 6). Note that 11i does **not** answer item 11: §5's (i), (ii) and (iv) are institutional questions, and this arm builds what makes them answerable | 11 | author | **done in engineering**; §5's (i), (ii) and (iv) remain institutional decisions |
| **14** | **Test results recorded by the pipeline, and shown where a model is chosen** — proposed by the project owner (2026-10-02). When a checkpoint selected on validation is tested on MyGene2 or the institutional cohort, the pipeline writes the result itself, bound to that checkpoint, so it cannot be omitted or mistyped and does not live in an editable file name; the result is then shown when the model is loaded and in the list of selectable models. Proposed shape: a separate Test tab; test-cohort management with version verification modelled on the training datasets'; and a choice between writing results automatically and, while a new test cohort is itself being evaluated before it becomes a baseline, running several rounds and writing one by hand. **What exists:** EVALUATION_COHORTS §6.5's ledger (`src/evaluation/sidecar.py`, `scripts/record_evaluation.py`) already binds a result to the checkpoint's SHA-256 and the cohort's digest beside the weights, refuses contradictions and survives a rename; it is written by a separate command and read by no API or UI. **To settle before design:** where the result lives (inside the `.pt` changes the digest every record cites — EVALUATION_COHORTS §5 item 4); whether ordering or choosing models by test score is allowed, since a cohort used to choose stops being an independent acceptance test (EVALUATION_COHORTS §4.1(6)); what tamper resistance is claimed beyond digest binding (since settled: binding only, no signing; D2, revision 4); and how a cohort is identified (digest and version label, EVALUATION_COHORTS §5 item 5). Taking it up reverses 11i's "UI out of scope" for this surface. **Plan drafted:** [`PLAN_TEST_RESULTS.md`](PLAN_TEST_RESULTS.md) sets out what exists, the decisions (D1–D8) and an order of work, and records adjacent defects found while planning (F1–F4). Revision 2 settles D3 by the scorer policy — a test result is the model's own disease ranking (Mode C, policy statements 1–3), with no SP term. Revision 3 states one path per concern (plan §4.0): ledger v2 (the old feature name for what M4 implemented as ledger schema 3) does not read v1 (the earlier schemas), one resolver locates each cohort's file so that the digest recorded is that of the samples scored (today `resolve_cohort` and `read_samples` locate it separately), and the copied checkpoint metrics share auto-selection's key list. Its 2026-10-05 amendment upgrades existing paths instead of working around them — the job runner is `training_manager` reshaped, with the Training Console moved onto it, and the existing checkpoint listing route is upgraded, its unsafe load converted — and records corrections from an independent verification: only the cosine is shared with the served scorer, not the pool, so unifying the pool is part of the constraint on B-1; D4 groups by seven named scoring-semantics fields and never promotes past a contested record; D5's block holds within one key, with a test run's inputs fixed; the checkpoint list follows SPEC_4's C3. F1–F3 are fixed (`d95965e`, `88569e4`, `fe44b8f`, `20f43d1`, `a57ed2d`, all on `main`). Revision 4 (2026-10-06) splits it into two phases: phase 1 is a complete CLI flow (import, run with the contract's checks, keep and record the report, show it on the existing model status), phase 2 the Test tab, job runner, D8 and ordering/arbitration; deterministic runs are no longer described as a cross-environment guarantee; D2 is settled (binding only, no signing). Its amendment after the review of `463a0df` removes the v1-ledger display condition, records R10 as the contract's §5.4 defines it, and moves training start's `checkpoint_dir` restriction back to phase 2 with the listing. **Phase 1 implemented on this branch (2026-10-16) as the contract's M4** (the contract's §4, "M4 as implemented"): `scripts/import_cohort.py` previews, then commits, an immutable test cohort version bound to the graph in a registry outside the repository; `scripts/measure_scorer.py --cohort LABEL --registry DIR` measures Mode C with R1, R3, R5 and the cohort↔graph binding checked on the bytes read, records R10, the environment and every denominator, and keeps every report; `--record` or later `scripts/record_evaluation.py --report` registers it through one function into ledger schema 3 beside the checkpoint; the Diagnosis tab's model status shows the loaded checkpoint's records through `GET /api/v1/evaluations/loaded-model`, labelled as the model's own Mode C ranking, not the tab's until B-1. The plan was corrected the same day: the repeat rule decided on 2026-10-07 (N1 delivered, now also in the importer; N2 not delivered), "ledger v2" named ledger schema 3, what M2 and M3 completed stated as done, and the owner questions asked once with M4's defaults (plan §4.9) | 11, 11i, **15** (M1–M3 before phase 1) | author + institution | **plan revision 4, corrected 2026-10-16; phase 1 (the contract's M4) implemented (2026-10-16), code-reviewed (one P2 and, in the incremental review, one P3, both fixed), its GB10 functional acceptance passed at `2c94a4e` (2026-10-17, 84 of 84), and merged into `main` by the owner (#126, `d2d31dd`, 2026-10-17); phase 2 (the contract's M5) in progress on the branch, not yet reviewed or merged (M5-0, M5-1 and M5-2 pushed; M5-3 under way, with the shared follow-ups, the probe's M5 acceptance and the internal review's fixes done), its boundaries recorded (2026-10-19: the reviewer's, relayed by the owner; plan §6, "M5's boundaries"; deployment mode M1, plan §4.9 question 3), and its open questions decided the same day (D8 without exceptions, the loopback default bind, the intake directory, one GPU lock, ordering by test score for research-purpose cohorts only, registration from the report store: plan D5, D7, D8, §4.9 question 1, §6); owner questions asked once (plan §4.9), phase 1 running on their defaults, the reviewer's literature check found reasonable by the owner; the institution's protocol and MyGene2's terms open** |
| **15** | **The provenance contract: every link in the pipeline checked where it is used** — the owner's goal (2026-10-06) is that a model always matches the exact setup that trained it, dataset version and every stage's settings included, with nothing misaligned between stages; the threat is misalignment, not an account holder, so no accounts, keys, signing or chain. **Decided:** a model whose recorded training graph does not match is refused (its SP and every graph-derived score cannot match it); old-pipeline artifacts are not supported (archived, not deleted, no compatibility mode); in the first version, resume continues the same data (fine-tuning on other data not yet supported). **Found:** links inside a workspace refuse today, but every link with the model on one side is checked by structure only, recorded and unread, or absent — model↔graph (structural warning only; the digest comparison was deferred, `fingerprint.py:226-234`), resume parent, training settings in effect (seed applied after the model is built; that ordering fixed ahead of the rest of M3a, `290c03a`), served identity (literal `model_version`), dataset record (no production reader); a failed model load fell back to path reasoning and was published as success (fixed by M1). **Plan:** [`PLAN_PROVENANCE_CONTRACT.md`](PLAN_PROVENANCE_CONTRACT.md) — M1 B-2's fail-closed core; M2 model↔graph enforced, producer and consumer together, on digests taken from the bytes actually parsed (before M2.1 each input was hashed by path at one moment and loaded at another; serving, training and measurement now record from one read, measurement's runs that included Mode A having been deleted by the Mode A/B retirement's R2, while item 19's sites do not), with the resume parent checked before anything is restored and a read-only rebuild inventory; M3 training/dataset/serving records and checks; M4–M5 are item 14's two phases. Existing records are upgraded; nothing new beside them except item 14's cohort version and report store | 14 depends on it; B-2 (`DISEASE_SCORER_POLICY.md`) | author | **plan revision 2; M1 implemented (2026-10-07), code-reviewed with no P1/P2 at `ed4872a`, and accepted on the homelab GPU at `7714459` (2026-10-08); M2.1's work plan revised after review with the owner's decisions of 2026-10-09; the plan-level re-review of `0708789` closed the earlier P2 and found one more (S10's resident-memory rule), since fixed, and the plan-level incremental review of `7e1af96` found no P1 or P2; the owner decided on 2026-10-10 that S0 also removes `scripts/evaluate_model.py` and its pin test, and that S10's capacity acceptance is limited to its designated subject (contract M2.1 decisions 6 and 7), and the plan-level review of that record (`7e1af96..45d9dea`) found no P1 or P2; its deferrals are items 18–19; M2.1's S0 implemented (2026-10-10) and code-reviewed with no P1 or P2 (`45d9dea..3433bf8`); S1–S5 implemented (2026-10-10) and code-reviewed with no P1/P2 (`e3eefee..6f16373`; the shared `read_samples` has changed since, in item 8's `06fd0f1`, code-reviewed with no P1 or P2 in `1d0b5a6..06fd0f1` on 2026-10-12); S7 implemented for Mode C only (2026-10-11) and code-reviewed with no P1/P2 at `7327798` (2026-10-12), as that declared scope and not as every mode migrated: a `--modes C` run reads each input once, checks the manifest, the graph tensors and (for a generated cohort) both sample files from those reads, and records what it read (M2.1 checks nothing about the checkpoint), while measurement runs that include Mode A (`A`, `A,B`, `A,B,C`) were not migrated — since the owner's clarification of 2026-10-12 they were deleted by the Mode A/B retirement's R2 (item 9's row; 2026-10-12, code-reviewed with no P1 or P2 in `1d0b5a6..06fd0f1`), so since R2 every measurement run takes S7's reads (implemented; not covered by S7's C-only review); M2.1 is still not complete: the S6/S8 batch (`8421b8c..cddad9d`) and, in `1d0b5a6..06fd0f1`, R2 (`e210b12`) and item 8 (`06fd0f1`) were code-reviewed with no P1 or P2 on 2026-10-12, and S9, implemented on 2026-10-12, was code-reviewed with no P1 or P2 the same day (`02a6954..f481882`), but S10 is outstanding: its plan amendments (`f481882..c1b8a12`) were reviewed with no P1 or P2, its tools (kept outside the repository) had three P2 in S9's review and, fixed, passed re-review with no P1 or P2 (draft 3, 2026-10-12), and it ran on 2026-10-12 (every acceptance 1 check passes at the after commit; comparability has 1 UNKNOWN, the service state, supplemented by the owner's statement, exit status 2 kept; acceptance 2 and 3 explained except the serving reload's resident rise, R3 +0.023 to +0.031 GB, source not located; the reviewer recommends a limited diagnostic of consecutive reloads before acceptance 2 is closed, chosen by the owner and run on 2026-10-13: each process grew by about 0.4–0.5 GB at every reload at both commits, before M2.1 as after it, and the difference between the commits is ambiguous; the owner ended the reload-memory investigation on 2026-10-13 with the wheel's bundled mimalloc purge defects as a strong candidate cause, attribution not established (item 22), and decides acceptance 2 separately), and the owner accepted its capacity on 2026-10-13 on its designated subject with the remaining uncertainty explicit, so **M2.1's technical work is complete** (not merged); M3a's seed ordering implemented ahead of the rest of M3a (2026-10-11) and code-reviewed with no P1/P2 at `290c03a` (2026-10-12): the seed now fixes the initial weights; S6 and S8 implemented (2026-10-11), their batch (`8421b8c..cddad9d`) code-reviewed as its own range with no P1 or P2 on 2026-10-12 (the later increment narrowing their claims, `cddad9d..3e8e559`, was reviewed with no P1/P2 on 2026-10-12, which is not a review of that batch) — they covered the workspace reads only; the resume parent's read and its checks moved to the top of the run, and a missing parent refuses, in M2.3; M2.2–M2.6 implemented (2026-10-13); their code review found three P2, closed by the re-review of `0aabfc3..26fe84c` with no new P1 or P2; M2's acceptance passed on the GB10 at `82d7573` (2026-10-14), functional only; the probe's allocator follow-up confirmed on the GB10 at `b8a43c8` (`cudaMallocAsync`); **M2's technical work complete and its acceptance within the scope agreed complete** (the owner, 2026-10-14); merged into `main` by the owner (#121, `9fe5113`, 2026-10-14); M3a's design revised on 2026-10-15 (decisions 8–10: resume with inherited settings and `extend`, one directory per run, serving's `weights_only=True`; all three the owner's, 2026-10-15); M3a implemented, code-reviewed, its functional acceptance passed on the GB10 in two rounds, and merged into `main` by the owner (#122, `2049b64`, 2026-10-15); M3b implemented (2026-10-15), code-reviewed with no P1 or P2 at `04ecf42` (2026-10-16), and its functional acceptance passed on the GB10 at `c80c1e4` (2026-10-16, 75 of 75, demo and real data), and merged into `main` by the owner (#123, `cbba015`, 2026-10-16); M3c implemented on this branch (2026-10-16: the served identity on every result, serving's `auto` refusing without CUDA, the consumers moved; the plan's §4, "M3c as implemented") and code-reviewed with no P1 or P2 at `1246f0c` (2026-10-16), its P3 fixed, and its functional acceptance passed on the GB10 at `52152e4` (2026-10-16, 75 of 75 with a WebUI smoke), and merged into `main` by the owner (#124, `ed8f51d`, 2026-10-16); M4 implemented (2026-10-16 over `59f519d`; the plan's §4, "M4 as implemented"), code-reviewed (2026-10-17: one P2 and, in the incremental review, one P3, both fixed), its functional acceptance passed on the GB10 at `2c94a4e` (2026-10-17, 84 of 84, demo and real data), and merged into `main` by the owner (#126, `d2d31dd`, 2026-10-17); M5 not implemented** |
| **16** | **One rule for a case's phenotype list, from import to scoring** — raised by R10's case label (contract §5.4) and the owner's question whether removing repeated phenotypes is what the original SHEPHERD or current practice does. **Found:** a repeated phenotype changes the GNN mean, the SP mean distance and path search; the WebUI removes repeats by string, the API keeps them, the direct pipeline keeps them and truncates to 100 before dropping unknown ids, so "100 × A, then B" loses B there; `InputValidator` removes repeats after truncating and has no production caller; the current generator writes no repeats; the original SHEPHERD keeps repeats and gives them no stated meaning; Phenopacket does not require unique terms. **Plan:** [`PLAN_PHENOTYPE_NORMALISATION.md`](PLAN_PHENOTYPE_NORMALISATION.md) — remove repeats by graph node after mapping, first-occurrence order, never merge patients, ancestors or present/excluded pairs; refuse instead of truncating; producers normalise and record the rule version and counts, readers check; N1 (serving) stands alone, N2 (stored data) was to ride on the contract's M2.1 and M3b, and the D6 importer uses it from its first version. **Since M3b (2026-10-16):** M3b kept split-manifest schema 3 and changed no manifest field, so N2's fields had no manifest change to ride on; N2 is its own change, unscheduled, with the reader check's M2.1 dependency met (the plan's §8, N2, "Since M3b"). **Since M4 (2026-10-16, on this branch):** the D6 importer applies N1's function after mapping, keeps the source bytes and records the rule version and counts (`map_cases`, `src/evaluation/cohort_registry.py`): N1 at a new entry point, not N2, and M4 has no N2 dependency. Noted for N2's design: once the reader's check lands, an R10 evidence file with repeats would fail to parse, and R10 would read that scope *unverifiable*. **Since N2 (2026-10-18)** it does (the plan's §8, N2, "As implemented"). | 15 (M2.1, done; M3c); 14 (D6) | author | **plan revision 2, amended; the owner adopted the rule, `InputValidator`'s removal and decision W (the WebUI stops removing repeats; a clear message for a list over the limit), 2026-10-07; N1 implemented, code-reviewed with no P1/P2, and accepted on the homelab GPU at `af49bd3` (2026-10-07); N1's function also applied by M4's importer (2026-10-16); N2 implemented without its manifest field (2026-10-18, batch B, on this branch; reviewed by the reviewer with no P1 or P2 (`1fe0ea9..2e099b6`, 2026-10-19); `hpo_2026_0929_5a`'s train and val samples pass its reader on the homelab, 2026-10-19; not merged): the shared reader refuses a stored samples file in which a case lists a phenotype twice (`UnparseableFile`, naming the file, the case, the rule version and the producer), the generator calls the rule (no byte changed), the split manifest's field waits for the next split-manifest schema change or the first change of the rule's version, and `measure_served_pipeline` stays the named exception (the plan's §6, §8)** |
| **17** | **Three defects found during N1's homelab acceptance** (2026-10-07; not caused by N1, present on `main` at `cffad34`). **F5 — the Diagnosis tab's model-configuration fields show the saved UI preferences, not the loaded pipeline.** `data_dir_input` takes its value from `.shepherd_ui_config.json` (`src/webui/components/diagnosis_panel.py:903-905`), and the status line names no path, so after a start with `SHEPHERD_*` set the tab showed `hpo_2026_0623_v2_orpha` while `hpo_2026_0929_5a` was served; pressing Load / Reload would have switched to the shown one. Likely fix: when a pipeline is loaded, show its data directory and checkpoint (the status route already returns `current_data_dir` and `current_checkpoint_path`). **F6 — `matching_phenotypes` is always empty on the real path.** The response model declares it (`src/api/routes/diagnose.py:183`), only the mock path fills it (`:453`), and the CSV and report exports carry the empty column. **F7 — the launcher gives up opening a browser after 60 s** (`scripts/launch/shep_launch.py:37`), and a startup that loads a large SP table eagerly took about 84 s. Separately recorded, not a defect: PyTorch warns that GB10's CUDA capability 12.1 is above the 12.0 it lists. **Fixed in batch B (2026-10-18):** **F5** — one status read per page load fills Data Directory and Checkpoint with what is served (a file chosen automatically included; a path-only start leaves Data Directory blank with a placeholder; nothing loaded keeps the saved values), a successful reload sets them from its result and a refused one leaves what was typed, the model status names the served workspace and checkpoint file on one line, and Save Config says it stores the paths shown; **F6** — `matching_phenotypes` holds the request's phenotypes as used that the graph annotates directly to the disease (an exact `phenotype_of_disease` term; not ancestors or descendants, not gene-mediated), filled whether or not explanations are asked for, from the one helper the explanation also uses (`find_matching_phenotypes`); the demo answer returns `[]`; the API schema and the CSV's column positions are unchanged; **F7** — the launcher polls `/health` until it answers, with no deadline, logs a still-starting line about every 30 s and opens the browser once; the opener stays a daemon thread, so it ends with the server | — | author | **done (2026-10-18, batch B, on this branch), reviewed by the reviewer with no P1 or P2 (`1fe0ea9..2e099b6`, 2026-10-19)** — not merged |
| **18** | **The evaluation ledger's stale-write guard compares a digest that may not describe the ledger it read** — found while planning M2.1 (contract §4, M2.1 decision 5). `record_evaluation.py` parsed the ledger (`read_ledger`) and then hashed the path again (`ledger_digest`); `write_ledger` compared that second digest with the file and replaced it. A write landing between the two reads makes the expected digest describe the other writer's file while the append is built on the older parse, so the comparison passes and the other writer's records are erased with no trace — reproduced against the real functions on temporary files. **Fix:** parse and hash the ledger from one read, so the expected digest is that of the records appended to; the same interleaving is then refused. The report the record is derived from has the same shape (parsed, then its path hashed, recorded as `source_artifact_digest`) and is fixed in the same change. **Not closed, and not claimed:** the check-then-replace window inside `write_ledger`, which its documentation states; the ledger stays single-writer by operational rule. No lock, job runner or new ledger is added. **Done in M4 (2026-10-16; merged as #126, `d2d31dd`):** `register_report` (`src/evaluation/registration.py`), the one registration function behind `measure_scorer.py --record` and `record_evaluation.py --report`, parses and identifies the report (`_read_report`) and the ledger (`read_ledger`) each from one `read_once`, and the write's expected digest is that of the ledger bytes parsed; the interleaving above is refused as stale, and a report replaced while it is registered cannot pair one file's numbers with the other's digest (`tests/unit/test_registration.py`). The window and the single-writer rule stay as stated | — | author | **done in M4** (2026-10-16; code-reviewed, its functional acceptance passed at `2c94a4e`, merged as #126, `d2d31dd`) |
| **19** | **Reads and writes deferred out of M2.1, each with its home** — M2.1 makes its four entry points parse and hash each recorded input from one read (contract §4, M2.1). The sites in the registry below the table read an input apart from its digest, or record a digest apart from their parse, and stay outside M2.1 by its decisions 4 and 5 (the owner, 2026-10-09), its "also unchanged" list, or, for 19.18, the reviewer's finding that Mode A cannot run on the designated checkpoint (its read stays its own: S7 was implemented for Mode C only, and runs that include Mode A were not migrated: since the owner's clarification of 2026-10-12 they were deleted by the Mode A/B retirement's R2 (item 9's row; 2026-10-12), and no Mode A builder that can run on a checkpoint the current writer produces was built, 19.18); 19.19 is instead removed by S0 (accepted at plan level in the re-review of `0708789` and, as revised by then, in the incremental review of `7e1af96`; its additions for contract M2.1 decision 6 accepted at plan level in the review of `45d9dea`; implemented 2026-10-10 and code-reviewed with no P1 or P2 at `3433bf8`). Each entry names the relation it does not guarantee and where it belongs, so none is lost when M2.1 closes. M2.1 restores nothing of the frozen evaluator, adds no checkpoint field and adds no fallback | 15 (M2.1) | author | **recorded** — each entry moves with its home |
| **20** | **HGT: two PyG problems on the formal build path, described separately in [`PLAN_HGT_LINEAR_PATHS.md`](PLAN_HGT_LINEAR_PATHS.md)** — found while researching an AMP-only comparison ([`FINDINGS_PAPER_PROTOCOL_AND_AMP.md`](FINDINGS_PAPER_PROTOCOL_AND_AMP.md) §4.3); torch_geometric 2.8.0.post1. **(a) A correctness defect.** `HeteroDictLinear`'s grouped branch (HGT's `kqv_lin`, `out_lin`) builds its inputs in its layers' order and stores the outputs under `x_dict`'s keys (`torch_geometric/nn/dense/linear.py:439-449`), so each node type can receive another's projection. On the formal path the two orders differ (training's loader, layers after the first, `build_index`). When the branch runs it is more often silently wrong than an `IndexError`. It runs with `torch_geometric.backend.use_segment_matmul = True`, or by PyG's default from ten node types up; production has three and the builder can name at most eight, so it is not reached today. Fix (Edit 1, approved by the owner on 2026-10-12 to land on its own; implemented 2026-10-12): `HeteroGNNLayer` passes the conv its node features in its layers' order, with a construction check, and its edges in the conv's edge-type order (the related hazard of the plan's §1.7); bitwise neutral today (CPU), pinned by `tests/unit/test_hgt_node_type_order.py`. **(b) A computation-path and reproducibility problem.** `HeteroLinear` (HGT's `k_rel`, `v_rel`) times `segment_matmul` against a loop per instance, row-count magnitude and process, and keeps the faster (`linear.py:258-285`, `:301-313`). The choice flips between processes, is recorded nowhere, differs between training and serving, and hides a loop backward about 35× slower. Proposed fix: a bounded GB10 comparison first, then the simplest sufficient policy applied in `HeteroGNNLayer`, explicit in the model configuration, with the requested policy, the compile setting and the observed kernel recorded separately (a compile wrapper does not mean the loop ran). Both fixes sit at the one place every entry point builds and runs HGT. GAT is unaffected | — | author | **(a) implemented (2026-10-12)**, code-reviewed with no P1 or P2 on 2026-10-12 (`1d0b5a6..06fd0f1`) at CPU code-review level only, not a GB10 performance or CUDA numeric acceptance; **(b) proposed (2026-10-11)**, waiting for its GB10 comparison. The plan was reviewed 2026-10-12 (`290c03a..1d0b5a6`) with one P2 (the kernel inferred from the compile setting), applied in `418b37e` (the kernel observed, not inferred), whose review (`1d0b5a6..06fd0f1`, 2026-10-12) found no new P1 or P2 and lets that P2 be closed; Edit 1 is `8eae614` |
| **21** | **AMP training study: is a model trained with AMP worse, and what does AMP save in training time and memory?** — bounded research. The owner reported on 2026-10-12 that what the owner and the institution want to know is whether an AMP-**trained** model loses accuracy or quality, weighed against the large training-time saving AMP gives some GNN architectures: a **training** study, **not** the same-checkpoint scoring comparison with autocast switched (the old differential's AMP report, 7a's AMP leg, the retirement assessment's §8 item 7 options (a)/(b)), whose wait that clarification closed. Background: [`FINDINGS_PAPER_PROTOCOL_AND_AMP.md`](FINDINGS_PAPER_PROTOCOL_AND_AMP.md) §4 — what is fixed and recorded today (§4.1), GPU nondeterminism and HGT's kernel path (§4.3, item 20), the two experiments told apart (§4.6). **Only the existing path:** `train_model` → `Trainer` → checkpoint → Mode C; no AMP-specific trainer, launcher or test engine. **Pilot first:** one pair on one CUDA GPU (on CPU `use_amp` has no effect, `trainer.py:380`), same verified workspace bytes (equal `training_input_digests` in both checkpoints), same architecture, hyper-parameters, training length, seed and worker count, so the two start from identical initial weights (possible since M3a's seed ordering, `290c03a`). The two runs' `config.yaml` (`scripts/train_model.py:739-741`) differ only in the AMP setting, with the AMP arm's dtype declared in advance; the dtype and the worker count are set only through YAML, where a missing file or an unknown key is ignored silently (findings §4.2), so the written `config.yaml` is what is compared. Separate output, checkpoint and log directories, since the defaults collide. It first confirms which ops AMP actually reaches for the architecture: HGT runs its convolution in fp32 with autocast off (`src/models/gnn/layers.py:200-218`), so for HGT AMP changes only the other layers. It records wall time, peak memory, GradScaler overflow (skipped) steps (float16 only; bfloat16 uses no scaler, `trainer.py:382-388`), and the training and validation curves. The curves and per-epoch time are already in the metrics log; skipped steps are detected (`trainer.py:561-568`) but not counted, and training reads no peak memory, so those two are recorded through the existing training path. **Formal comparison, only if the owner widens after the pilot:** several paired seeds; each arm picks its checkpoint by the same validation rule, declared before the run, ties included (with AMP on, the selecting validation pass runs under autocast, `trainer.py:666`, which is part of what the AMP arm is); both arms scored by the same Mode C, on the same test cohort, at the same scoring precision (Mode C runs without autocast); a seed, an epoch or the AMP setting is never chosen by test score. Report per-pair differences and their spread across seeds, beside the time and memory saved. Fixed seeds do not remove CUDA nondeterminism, so not every observed difference is attributable to AMP; whether runs use deterministic mode stays open (findings §5 point 5). **HGT:** an HGT pair needs item 20 (b)'s kernel path pinned identically in both arms through the formal build path, not a study script (item 20 (b)'s fix, Edit 2, waits for its own GB10 comparison and is not decided); otherwise the pair uses GAT, so the timing-based kernel choice cannot confound the comparison. The WebUI forces AMP off for HGT (`src/webui/components/training_console.py:239`), so an HGT AMP arm runs from the CLI, YAML or API. **Not decided:** the acceptable quality loss is a product/research tolerance not yet given, so no margin is assumed; the pilot is reported first, and whether to widen is decided on it. **Not a prerequisite of M2 or of R2**, and nothing else waits on it | Pilot: M3a's seed ordering (`290c03a`) — **satisfied** (implemented and code-reviewed, `7327798..290c03a`; shown on CPU by `tests/unit/test_training_seed.py`, not yet observed on the GB10). An HGT pair: item 20 (b)'s pinned path, else GAT. Formal comparison: the pilot's report and a decision to widen; a test cohort Mode C scores (which cohort is findings §5 point 7; the formal Test arrives through item 14's import) | author; the owner sets the tolerance and decides on widening | **recorded (2026-10-12)** — not started; pilot first |
| **22** | **Serving reload's resident-memory retention; attribution incomplete** — each reload left about 0.4–0.5 GB more resident memory in the serving process, at `90a668f` and `343e90e` alike (S10's reload diagnostic, [`reload-diag-2026-10-13/`](scorer-measurement/reload-diag-2026-10-13/README.md)); the earlier pipelines' watched objects were freed, and a later run did not repeat the steady growth. The PyTorch wheel in use is confirmed to contain mimalloc purge defects that upstream has since fixed (`libc10.so` allocates CPU memory through mimalloc 2.2.4; upstream `acd6f6c` and `4e50cec`); they are a strong candidate cause of the observed memory retention; the attribution of all of the increase is not established ([`reload-inproc-segments-2026-10-13/`](scorer-measurement/reload-inproc-segments-2026-10-13/README.md)). **Decided (the owner, 2026-10-13):** the investigation ends; no change to the product pipeline, no `MIMALLOC_*` setting, no profiler or new diagnostic tool, and no targeted patch for a dependency-level defect. **SHEPHERD's own references, checked within a bounded scope (2026-10-14, at `9fe5113`, the merge of #121; read-only, on this evidence and the existing tests, nothing rerun on the GB10):** no persistent holding was found in that scope. The earlier pipeline: only the service's state holds one, and publication rebinds it; phase 2 found every watched object freed at `343e90e`, and serving gained no long-lived attribute or cache in `343e90e..9fe5113`. A failed candidate: held by the reload's locals only, its error kept as text. Checkpoint reads: serving keeps the read's identity and metadata converted to primitives, not the parsed dictionary; measurement holds its read for its one run; the training console's listing loads each checkpoint in turn and returns only its epoch and logs, keeping none after it returns (two may be resident while the next one loads). The resume parent: released before training (`test_resume_parent.py`). Temporary buffers: the shared readers release the bytes they parsed (the release checks in `test_shared_readers.py` and `test_graph_read_json.py`); measurement drops the split it reads only to verify before the checkpoint is read (`test_measurement_reads_once.py`, after a collection); `build_sp_index`'s temporaries are its locals, the SP step's rise fell back after publication and `_precompute_node_embeddings` left none (the segments run). This does not rule out a leak, and it attributes none of the retention. The reviewer's independent check of `9fe5113..34275f7` (2026-10-14) agreed, with no P1 or P2 | a PyTorch wheel whose `libc10.so` is shown, in its machine code rather than a version string, to carry both upstream fixes | owner | **recorded (2026-10-13)** — no action until such a wheel exists; then a reload check with phase 1's procedure. SHEPHERD's own references checked within a bounded scope at `9fe5113` (2026-10-14): no persistent holding found there, the reviewer agreeing |
| **23** | **The GNN experiment phase, and the branch mode it needs** — the platform is also the GNN's experimental platform: before deployment the GNN is explored and evaluated in full at least once, and the owner and the institution start that work only once the platform is stable (the owner, 2026-10-15). Resume inherits every learning setting (contract decision 8), so a mode that starts from a parent and changes them, a *branch*, is a prerequisite of that phase. **Draft, not designed:** the same entry point and trainer; a closed list of fields that may change (the loss weights, temperature, margin, label smoothing, negative counts, and a new explicit learning-rate schedule with its own horizon and warmup); architecture and data unchanged (decision 7); whether the parent's optimizer moments carry over stated and recorded; the parent's SHA-256 and every change (parent value → new value) recorded; the run shown as a branch of its parent. Validation compares and chooses among experiments; the held-out test serves the agreed evaluation, and a test used again and again to choose models is no longer an independent acceptance. A comparison on validation before item 11 and M4 is provisional and says so. The phase's other prerequisites, among them M4/M5's model list with tags, filters and lineage, and item 11, are the owner's to settle | contract M3a (records, run directories, resume, `extend`) | owner (scope); author (design) | **recorded (2026-10-15)** — designed and built before the experiment phase |
| **24** | **Hardware beyond CUDA: four boundaries now, adaptation when a target exists** — the owner and the reviewer discussed on 2026-10-16 that the hospital may later add accelerators other than CUDA, and recorded four engineering principles instead of building for them now. **(1) CUDA is the current deployment and acceptance scope, not a permanent architectural limit.** The settled `auto` behaviour stays (CUDA, refused without it; the CPU only when asked for); a backend added later comes with its own explicit selection rule, and an unannounced CPU fallback stays forbidden. The policy reads, in the owner's wording of 2026-10-16: hardware acceleration is a hard requirement for deployment, and CUDA (NVIDIA) is currently the only supported backend; the refusals of training, measurement and serving say so. torch's ROCm builds also name their device `cuda` and have not been verified. **(2) Hardware differences are handled in one place.** Device resolution, the allocator, AMP, capability checks and environment collection keep to their existing entry points (the device resolvers of training, measurement and serving; the launcher's allocator preset; `src/utils/run_environment.py` for environment facts) and converge on them over time; no vendor branch is added separately in training, measurement, serving or the UI. **(3) Every backend shares the provenance contract and the scoring semantics.** A model converted by a tool such as OpenVINO is a new artifact: its record names the source checkpoint's digest and the conversion settings, and it is never presented as the original checkpoint (the served identity names what was served). **(4) Implemented and accepted on the actual target hardware, once there is one:** the operators needed, the training or inference capability, numeric differences and performance are verified there. No empty multi-backend framework is built in advance, and results across hardware are not required to be bit-identical | an actual non-CUDA target platform | owner (whether and which); author (the adaptation) | **principles recorded (2026-10-16); nothing built** — the work in hand continues as planned |
| **25** | **Test-cohort sources the M4 importer does not read yet** — found while building item 14's phase 1 (the contract's M4, 2026-10-16). The importer reads one source format, `shepherd-cases/1`, the project's shared import format (its name says nothing of whether cases are synthetic or real), with HPO phenotypes and MONDO diagnoses only. **Real file formats** (MyGene2's, the institution's) need an adapter that writes that format, built against de-identified samples once they exist (`PLAN_TEST_RESULTS.md` §4.9, M4 question 5). **OMIM or ORPHA ground truth** is refused as a whole today (question 3); translating it needs the MONDO file the dataset record verifies, with an ambiguous cross-reference excluded under its own category. The build's own maps are not a model for it: `build_omim_to_mondo_map` and `build_orpha_to_mondo_map` (`src/data_sources/hpo_annotations.py`) keep the last match silently when one OMIM or ORPHA id is a cross-reference of several MONDO terms (found while mapping M4); whether the graph build needs its own fix belongs here too. **Obsolete and alternative HPO ids** count as unmappable (question 4); following `replaced_by` needs the HPO file verified against the dataset record. **The reviewer's literature-based rules for this work (2026-10-17, found reasonable by the owner; `PLAN_TEST_RESULTS.md` §4.9, "Checked against the literature"):** an `alt_id` to its unique primary id and an obsolete id only through a unique, acyclic `replaced_by` chain ending at a graph node, `consider` and multiple targets left unresolved; OMIM and ORPHA accepted only through an exact, unique MONDO mapping to a disease in the graph; adapters keep positive and excluded phenotypes apart. No second importer, mapping rule or compatibility path: each adapter writes the one format, and the one importer maps it. **Upload through the browser** is not in M5 (the reviewer's decision, relayed by the owner, 2026-10-19): M5 takes sources from a server intake directory (`PLAN_TEST_RESULTS.md` D7); an upload would later feed the same importer once a controlled staging area and its cleanup exist | 14 (phase 1); de-identified samples; `PLAN_TEST_RESULTS.md` §4.9, M4 questions 3–5 and the literature check | author; institution (samples, formats) | **recorded (2026-10-16)** — not started; waits for real samples |
| **26** | **`measure_scorer.py`'s check refusals end in a Python traceback** — reported while building M4 (2026-10-16). A refusal raised by a check (a binding mismatch, a version's samples replaced, a dataset record that does not match, R3) is a `ValueError` from `main`, so the operator sees a traceback ending in the one-line message, exit 1. That is the measurement's behaviour before M4, kept so the generated path did not change with it; M4's other entry points (`import_cohort.py`, `record_evaluation.py`, `audit_split_overlap.py`) print one message saying nothing was written. **Fix:** where the script runs as a program, catch the expected refusals and print the message alone with a non-zero status, for both kinds of cohort, keeping exit 2 for arguments and 3 for a refused registration; tests and the probe that call `main` are unaffected. P3: nothing is written before such a refusal, and the message is correct. **As built (2026-10-18, batch B):** run as a program (`_run_as_program`), a `ValueError` or `OSError` refusal prints one line, `measure_scorer: refused: …`, ending in "Nothing was written." (added only when the message does not say it already), exit 1; `main` still raises its typed exceptions for the tests and the probe. A failure while publishing exits 1 saying what may be on disk (`ReportNotWritten`), and a failure after the report was written names the report (`FailedAfterPublishing`). Two adjacent tracebacks are gone as well: a checkpoint that does not parse is refused naming the file (out-of-memory and device errors keep their tracebacks), and `--device cuda` without CUDA is refused before anything is read, as `auto` already was | — | author | **done (2026-10-18, batch B, on this branch), reviewed by the reviewer with no P1 or P2 (`1fe0ea9..2e099b6`, 2026-10-19)** — not merged |
| **27** | **The WebUI's layout and interaction rules for the new pipeline's screens** — the owner's requirements (2026-10-18), from using the first WebUI, built for the old pipeline, and recorded before any screen is redesigned; the owner noted that much of this may lie outside the current construction scope. **Screens:** deployment displays are mostly landscape 16:9 or 16:10, from 1080p to 4K, and rarely portrait at the same ratios. **(1) One screen when nothing is expanded:** as far as possible, whatever is displayed, the whole page shows on one screen, clean and uncrowded, with no scrolling or paging. **(2) Expansion that keeps scrolling short:** as far as possible, an expanded collapsible section should not make the scroll too long; failing that, the collapsible tabs are laid out sideways, for example opening step by step to the right. **(3) Help behind an "i":** instead of help text under a label, a small "i" icon that shows the text in a bubble on hover. **(4) The served identity in one place:** the fingerprints now appear twice, in the pipeline status and in each result's "What produced this result", which duplicates a function and looks cluttered; the owner's idea is one separate block shown at all times; their first thought was a list, which they doubt would look good or read well, and they asked for suggestions. **Facts for the design:** there are 43 help texts at `1fe0ea9`, given as Gradio's `info=` (Diagnosis 6, Training Console 37); Gradio 5.29.1 has no tooltip parameter for inputs, so (3) needs the app's own CSS and HTML (`src/webui/app.py` already passes custom CSS). The guides' M3c rule is that a result shows its own identity, never the status's (deployment-guide.en.md, "The WebUI shows the result's own identity"); a design for (4) states how it keeps that, or the owner changes it. **Proposed by the author, not decided (2026-10-18, awaiting the owner):** a panel that holds more scrolls inside itself, never the page; the "i" bubble also on keyboard focus; for (4), one persistent block for the loaded pipeline and, on a result, a badge saying it was produced by what is loaded or listing the parts that differ, with the exports keeping the full identity; the UI foundation (layout, the "i" help, the identity block) as M5's first step, since M5 adds the Test tab and model management, with a static mockup the owner approves before it is built and screenshots checked at 1920×1080, 2560×1600, 3840×2160 and 1080×1920. Batch B's F5 adds one status line and no new help text under a label; one existing help text, the Checkpoint field's, gains the clause "shows the served file while a pipeline is loaded". **Since 2026-10-19 (the reviewer's M5 boundaries, relayed by the owner; `PLAN_TEST_RESULTS.md` §6, "M5's boundaries"):** the rules are carried out inside M5 — the main operations on one screen as far as possible in an ordinary window, never by shrinking text, clipping content or hiding warnings, and by the browser's usable size rather than the monitor's; the Diagnosis tab as input, candidate list and one fixed details area on a wide window, candidates and details switched on a narrow or portrait one; one shared "i" control for help (mouse, keyboard and click; dismissable; readable when the pointer enters the bubble), with refusals, job states, synthetic workspaces and unverified sources kept visible as short text; the served identity and each result's own identity kept apart in one shared block, with three states beside a result (equal, differs, cannot be checked). The layout direction is confirmed once on a thin, operable Gradio prototype; the owner asks for a mockup before UI work and screenshots at its completion (2026-10-19) | 14 (M5) | author; owner (the layout direction) | **recorded (2026-10-18); carried out in M5-2 (2026-10-20/21: the one-screen layout, the shared ⓘ help, the "Model in use" card and the results badge; deployment guide §4.14)** — the screenshots for acceptance remain (M5-3) |
| **28** | **Inference for several frontends on the hospital LAN — kept possible, not built** — the owner asked (2026-10-19) how to keep the option of several users on the hospital LAN, each with a frontend, using one backend for inference at the same time, without building a multi-user system now; the reviewer's advice, relayed by the owner. **Inference apart from management:** entering phenotypes, getting a ranking, and viewing and exporting one's own result may later be served over the LAN; training, importing cases, starting a test and registering a result stay where deployment mode M1 puts them (local or controlled SSH, `PLAN_TEST_RESULTS.md` §4.9 question 3), and so do switching the model, reloading and changing service settings; shared cases, history and roles are planned separately and are not a prerequisite. **What M5 keeps** (`PLAN_TEST_RESULTS.md` §6, "M5's boundaries"): every frontend through the shared API to the one pipeline; a case, its result and its exports per request or session, never global; administrative limits in the backend; one consistent pipeline and identity per diagnosis. **A candidate first version:** one served model shared by everyone, with no per-user model switch; a bounded queue that reports waiting, busy and timed out, covering the UI and direct API calls alike; one request on the GPU at a time at first. Not more server workers: each process loads its own model and data, multiplying memory (FastAPI's deployment notes, "memory per process"). Not a generic scheduler. **Before the LAN is opened**, a limited acceptance: two or three clients with different cases never see each other's results or exports; the queue is bounded, and a busy state, an error or an interruption never blocks the requests after it; a reload during inference behaves as defined (the first version may refuse a reload while busy); a LAN user cannot run an administrative operation by calling the backend directly. **The LAN is not authentication:** opening it connects the institution's existing access control or authentication entry (SPEC_4's M2), with no account system built here. An HTTP API existing is not evidence that several users are served safely. **Since 2026-10-19:** the default bind is `127.0.0.1` (SPEC_4 B1; another address only by explicit configuration, never lifting D8), and the multi-frontend audit for M5 found no request, result or export mixed today; what it leaves for this item: one request at a time comes from the blocked event loop rather than a bounded queue (nothing reports busy, `/health` stalls), a reload waits behind a diagnosis instead of being refused while busy, and the WebUI stops waiting for a reload after 180 s ("The page stopped waiting after 180 s; the server may still finish the reload.", with Read again) rather than the server refusing a reload while busy | 14 (M5's boundaries); SPEC_4 | author; institution (access control) | **recorded (2026-10-19)** — not started; M5 keeps it possible |
| **29** | **Dataset building and management in the WebUI** — the owner places dataset management and building under "Prepare" (2026-10-19). Today both are command-line only (`scripts/build_knowledge_graph.py --generate-samples`, and `scripts/setup_demo.py` for the demo; `scripts/preprocess_data.py` is an empty placeholder). A build page needs the job runner, the GPU lock and D8, and is a new product surface, so M5 gives only a read-only Datasets page over the server's workspace list (`PLAN_TEST_RESULTS.md` §6, "M5's boundaries") | 14 (M5) | author | **recorded (2026-10-19)** — not started |
| **30** | **A release designation for diagnosis** — an operator-set "released for diagnosis" mark on a checkpoint, with a reason, a link to an institutional acceptance record and the usage history as its audit trail; the clinician's change-model drawer would then list released models by default. It is never set from test scores (`PLAN_TEST_RESULTS.md` §4.9 question 1). It needs the owner's decision and the institution's acceptance protocol, and ties to `institutional_acceptance` and item 28. Related, for the owner: whether the fixed statement adopts research-use wording, and a short usability check of the Diagnosis tab's model card with three to five clinicians. Until then the UI never says "approved" or "validated" | 14 (M5); institution | author; owner; institution | **recorded (2026-10-19)** — not started |
| **31** | **Model comparison and housekeeping** — beyond M5's Models page: a side-by-side comparison of up to five checkpoints with "show only differences" over settings, architecture and dataset identity; archiving checkpoints from the lists while keeping them referenceable; search across workspaces ("which checkpoints were trained on dataset fingerprint X") | 14 (M5) | author | **recorded (2026-10-19)** — not started |
| **32** | **Navigation follow-ups** — after M5's two-level navigation: showing or hiding pages by role, which belongs with item 28 and the institution's access control (under deployment mode M1 every page is visible and the backend's D8 check is the protection, never hiding); per-page links (`?stage=` / `?sub=`, never `?page=`, which Gradio reserves); the training form's New and Continue merged into one mode with a single Start; and the page shown kept across a reconnect reload (a server restart, Settings' Restart Backend): today every open page lands on Diagnosis after one, and the chart's view comes back only once Training is opened again (M5-3 review UI-8; the outer and research page kept in `sessionStorage` or the address, selected on load through the cross-link path) | 14 (M5); 28 | author | **recorded (2026-10-19)** — not started |
| **33** | **Exporting and importing a model together with its dataset** — the owner asked (2026-10-19) whether a chosen model can be exported with its complete dataset version and every fingerprint needed to verify it, and imported on another machine; a model is never exported without the dataset it was trained on, because the two are bound. Nothing does this today. The contract makes it tractable: every checkpoint records the digests of the graph files, the split manifest, both sample files and the dataset record it was trained on (`training_input_digests`), and serving and measurement already check them. **Sketch, not designed:** export packs the chosen checkpoint, its workspace (graph files, `split_manifest.json`, samples, `kg.provenance.json`, `workspace_build.json`) and a package manifest listing each file's role and SHA-256, and refuses when the workspace's bytes do not match what the checkpoint records; the shortest-path table (about 11 GB for `hpo_2026_0929_5a`) is either included or rebuilt on import and checked against its recorded binding; the model's ledger records (aggregate figures) may go along, while patient-level files (test cohort versions' samples, per-patient ranks) never do (`PLAN_TEST_RESULTS.md` D7); import checks every digest and the checkpoint↔workspace binding with the existing checks before anything is placed, puts the workspace under the server's workspace root (reusing an identical one, refusing a same-named different one) and the checkpoint into a run directory, and is a D8-gated operation. "Signature" here means the recorded digests: the contract's decision is no keys, signing or chain (the threat is misalignment, not an adversary); authenticity between institutions would be a new decision. A new file format and a new product surface | 15; 14 (M5); 29 | author; owner (scope, format) | **recorded (2026-10-19)** — not started |
| **34** | **Upstream Gradio issues to recheck before release** — the dashboard moved to Gradio 6.30.0 as M5's first commit (2026-10-20; deployment guide §4.10). Before a release, each is rechecked against the Gradio then pinned. **(a) gradio#13895, SSE session loss** — open: a dropped `/queue/data` stream deletes the session's pending messages, and the browser shows "Connection Lost" and reloads the page; the draft fix (#13925) targets the next protocol only. M5 adds nothing for it: the only server setting is `GRADIO_HEARTBEAT_INTERVAL`, and a reverse proxy, outside M5, would need `proxy_buffering off`. **(b) `file=<URL>` serving remote HTML inline on our origin** — for a `.png`, `.txt` or `.json` URL whose server answers `text/html`, 6.30 serves it as `text/html` inline, a script on the dashboard's origin; the server-side fetch (since 6.20, #13593 / PR #13596) also ignores `HTTPS_PROXY`. Not reported upstream. M5 refuses every URL form of `file=` (and `process_recording`) in front of the mount (`src/api/middleware/gradio_routes.py`); the guard stays until Gradio offers a switch. **(c) gradio#13832, Dropdown `None`** — open (milestone "Gradio 7 Launch"); the maintainer reserves `None` for "no value selected". M5 uses `None` only that way (the diagnosis candidate selector, the training console's checkpoint list) and no choice carries a `None` value; a new dropdown that needs one uses a sentinel mapped to `None` in its handler. **(d) The Vega-Lite schema warning** — the native plots' spec names Vega-Lite v5.17.0 while the bundled library is v6.4.2, so each chart embed logs one console warning (and a chart with no rows logs Vega's "Infinite extent" warnings); harmless, not settable from Python, no upstream issue found. Moot since M5-2: the Training page's charts are the vendored uPlot 1.6.32, and no native plot remains in the dashboard | 14 (M5) | author | **recorded (2026-10-20)** — recheck before release |
| **35** | **Importing the dashboard before the API leaves the API without it** — found by M5-1's WP2 (2026-10-20), older than M5: `src/api/__init__.py` builds the application on import, so a program that imports `src.webui` before `src.api.main` gets a `src.api.main.app` without its `/ui` mount. The service never does this (it starts from `src.api.main`), so nothing deployed is affected; tests that need the mounted app build a fresh parent app instead. A fix would stop the package's `__init__` from building the app on import | — | author | **done (2026-10-21, the M5-3 review, EVAL-5)** — `src/api/__init__.py` builds nothing on import (its names are `src.api.main`'s, imported when first asked for), and `src/api/main.py` logs an import failure that is not a missing Gradio as an error; `tests/unit/test_api_import_order.py` imports in each order |
| **36** | **The read-only route's records carry no contested state.** `GET /api/v1/evaluations/loaded-model` returns the loaded checkpoint's records without D5's contested state, which the dashboard reads from the report store (`PLAN_TEST_RESULTS.md` D5). An API client cannot tell a contested record. A fix adds each record's state from `scan_report_store`, as Models' read-only view does (a change to the route's answer) | 14 (M5) | author | **recorded (2026-10-21)** — not started |
| **37** | **A copy of the whole registry registers as a second registry** (the M5-3 review, EVAL-4). A moved and a copied registry look the same, so `record_evaluation.py --report <copy>/versions/<label>/reports/<run>/report.json` registers into the ledger beside the checkpoint, while its "registered" entry goes to the copy's usage history and D5's block reads the copy's store; Models and the runner read only `SHEPHERD_REGISTRY`'s. `PLAN_TEST_RESULTS.md` D5 says a copy is not an entry point for registration. For the owner: refuse, when `SHEPHERD_REGISTRY` is set, a report whose registry is another one (in `register_report`'s command-line entry and `measure_scorer --registry`), or require `--registry` and check the report resolves under it. Until then the deployment guide (§4.9) says a copy is a second registry | 14 (M5) | owner | **recorded (2026-10-21)** — not started |
| **38** | **A run ended by SIGTERM, SIGKILL or a power loss between publishing and its usage history** (the M5-3 review, EVAL-1). A Cancel (SIGINT) is now held over publishing, the history entry and the first outcome file, and over a ledger record and its history entry (`src/utils/interrupts.py`); the job runner's SIGTERM/SIGKILL escalation and a power loss cannot be held. When the outcome file is missing or not complete, the runner could look for the run id and report digest in the version's `usage.jsonl` and say "this run is not in the version's usage history" when it is absent (a new runner statement) | 14 (M5) | author | **recorded (2026-10-21)** — not started |
| **39** | **The model list's first reading after a start, and the clinical preset reading runs it does not show** (the M5-3 review, UI-2). The server now reads each checkpoint once and again only when it changes (`checkpoint_paths.read_run_checkpoints`), so reopening a list, or the re-read after a Load, opens no unchanged file; the first listing after the service starts still reads every checkpoint of the workspace, and "Change model…" reads every run while it shows the latest four. A fix would list the runs read-free (`list_run_checkpoints`) and read only the runs shown, more on "Show older runs" (the listing's answer and the drawer's count would change) | 14 (M5) | author | **recorded (2026-10-21)** — not started |
| **8a** | B-0.5 protocol and output-contract **design**. **Consumes item 11's holdout decision and may not redefine it** | 1, **11** | author | **before** any expensive run |
| **8b** | B-0.5 institutional execution | 8a, 6, exact artifacts, production-path prerequisites. 7b dropped by the owner's clarification of 2026-10-12: it carried 7a and the A/B measurement, which B-0.5 does not need (retirement assessment §4); the rewritten 7b takes its served ranking from 8b instead. Was: 8a, 7b, 6, exact artifacts, production-path prerequisites | both | blocked |
| **9** | **Replaced by the Mode A/B retirement change, R2** ([`ASSESSMENT_MODE_AB_RETIREMENT.md`](scorer-measurement/ASSESSMENT_MODE_AB_RETIREMENT.md) §9) — superseded by the owner's clarification of 2026-10-12, which reported that the institution had already authorised retiring the old pipeline's Mode A/B tools. R2 deletes Modes A and B per the assessment's existing lists and rewrites the texts those lists name; it no longer waits for the institution (assessment §8 items 4 and 7 closed), and its deletion gate is that the migrated tests are in place (R1), not a test *of* Mode A. The owner decided on 2026-10-12 to start R2 now, together with the assessment's §8 item 5 (the measurement-only manifest fields and the measurement seed removed) and item 9 (`--modes` removed), with no compatibility flag or read branch for older records; item 8 (refuse non-integer and boolean ids) follows R2 as its own change, implemented the same day (`06fd0f1`; code-reviewed with no P1 or P2 on 2026-10-12, in `1d0b5a6..06fd0f1`). Was: Mechanical rename (~70 refs, 9 files), then rewrite the checklist, then delete the oracle-only surface. `scripts/evaluate_model.py`, its pin test and the oracle-parity tests in `tests/integration/test_legacy_equivalence.py` already went in contract M2.1's S0, the gate's one exception (§5; the owner's rule of 2026-10-09 and decision of 2026-10-10, contract M2.1 decisions 3 and 6) | R1 — implemented and code-reviewed with no P1/P2 at `3d53c9b` (2026-10-12). Was: **1d passed review incl. its institutional CUDA run** | author | **R2 implemented (2026-10-12) and code-reviewed with no P1 or P2 (`e210b12`, in `1d0b5a6..06fd0f1`, 2026-10-12)** — deletions and wording, with the assessment's §8 items 5 and 9 (retirement assessment §9.2). Was: behaviour-neutral |
| **10** | **Commit bounded evidence for M1-M5** — four JSON files and the three scripts that emit them. **Not raw console output** (§5.2). Unblocks 2, 3, 6 and 7a | — | institution + author | **done** — run on the deployment-sibling machine, evidence cited by digest in §2; M1-M4 confirmed, M5 corrected (§2.4) |


**Item 19's registry — reads and writes deferred out of M2.1.** Sites at `90a668f`. M2.1 refuses
none of them, and the relation each row names is left as it is. M2.1 touches two: S0 removes
19.19, and S3 fixes 19.21's recorded digest but not its placement. S7 was to pass 19.18's Mode A
builder (`build_legacy_mode_a_model`) the run's checkpoint read; it was implemented for Mode C only
(2026-10-11), so that builder kept loading its own path until it left with Modes A and B in R2
(2026-10-12), which replaces item 9 (superseded by the owner's clarification of 2026-10-12: it was
to leave with item 9's oracle-only surface, after 7a, now cancelled). No replacement that can run
on a checkpoint the current writer produces was built, and 19.18 is closed by R2 (implemented
2026-10-12; code-reviewed with no P1 or P2 in `1d0b5a6..06fd0f1`).
The others that M2.1's static pin meets are allowlisted with the
reason given. (As implemented in S9, 2026-10-12: the list is `ALLOWED` in
`tests/unit/test_provenance_pins.py`, each entry with its use and owner.)

| # | Site | Reads → records or compares | Does not guarantee | Belongs to |
|---|---|---|---|---|
| **19.1** | Reload route candidate scoring — `_checkpoint_score`, `src/api/routes/pipeline.py:325-333` (`torch.load(weights_only=False)`, `:328`), per candidate via `select_checkpoint_in_dir` / `select_auto_checkpoint` (`src/utils/checkpoint_paths.py:103-182`) | every `*.pt` in the architecture directory → compares the `logs` ranking score; records nothing | that the file served holds the bytes that were scored | **M3c** — contract M2.1 decision 4: "the build, and later M2.4, use the final `read_checkpoint`"; M3c takes the reported metric from the served build's own read. **Done in M3c (implemented 2026-10-16, merged as #124):** the scan reads each candidate with `read_checkpoint` and keeps its digest; when the chosen file's scored digest differs from the served identity's, the reload serves the build's read and `selection_reason` states the replacement with both digests, not a refusal (M2.1 decision 4) |
| **19.2** | Reload route second read — `src/api/routes/pipeline.py:361-368` (`:363`) | the chosen path again → `best {metric}={value}` in `selection_reason` | that the reported metric is of the bytes scored or served: the build reads a third time (`src/inference/pipeline.py:973`) and `checkpoint_meta` comes from that read — chose A, reported B, served C | **M3c** — same text. **Done in M3c (implemented 2026-10-16, merged as #124):** the second read is gone; the metric in `selection_reason` comes from the built pipeline's `checkpoint_meta`, the served read |
| **19.3** | `TrainingManager.get_checkpoints` — `src/api/services/training_manager.py:436-473` (`:464`, `weights_only=False`); also feeds the resume dropdown (`src/webui/components/training_console.py:499-524`) | each top-level `*.pt` → name, path, size, mtime, `epoch`, `logs`; no digest | that the epoch and metrics shown describe the file later resumed or loaded | **Item 14 phase 2 (contract M5)** — `PLAN_TEST_RESULTS.md` §6 step 5, one listing function with `weights_only=True`; added here: what the list shows, and any digest, comes from one read |
| **19.4** | `scripts/migrate_checkpoints.py` — `_conv_type_of` `:42-58` (`:50`, `weights_only=False`), then `shutil.move` by path (`:107`) | a legacy flat checkpoint's `config` or weight names → moves it into `checkpoints/<conv_type>/` | that the file moved is the file classified | **None** — old-pipeline artifacts (contract decision 2; §7). No existing checkpoint records `kg`, so M2 refuses every one (M2.6, "Consequence"). The reload refusal still recommends this script (`src/api/routes/pipeline.py:375-379`); that text changes with M2.4 |
| **19.5** | Ledger stale-write guard — `scripts/record_evaluation.py:84-85`, `:133`; `src/evaluation/sidecar.py:214-234`, `:399-451` | parses the ledger, hashes it again, compares that hash before replacing | that the expected digest describes the ledger appended to | **Item 18**. **Done in M4 (2026-10-16; merged as #126, `d2d31dd`):** `read_ledger` parses the ledger and returns the digest of those bytes from one `read_once`, and `register_report` passes that digest to `write_ledger` as the expected one; the guard's comparison by path just before the replacement, and its window, stay (single-writer, not compare-and-swap) |
| **19.6** | `source_artifact_digest` — `scripts/record_evaluation.py:100` parse, `:102` path hash; `sidecar.py:318` | the report → the record's source digest | that it is the digest of the report the record was derived from | **Item 18**; otherwise item 14's ledger schema 3 (M4; the ledger file's `schema_version` was already 2 since R2, `src/evaluation/sidecar.py`), which carries report digests. **Done in M4 (2026-10-16; merged as #126, `d2d31dd`):** `register_report` parses the report and takes its digest from one `read_once` (`_read_report`); a schema 3 record's `source_artifact_digest` is that read's digest, and an agreeing report's digest is added to `corroborating_reports` |
| **19.7** | `find_checkpoint` — `src/evaluation/sidecar.py:366-383` | hashes each `*.pt` → refuses unless one matches the report's checkpoint digest | that the matching file keeps those bytes afterwards; identification, not a recorded input | **None** — kept by M2.1 ("`file_sha256` stays only for identifying a file nobody is about to parse"). **Since M4's review fixes (merged as #126, `d2d31dd`):** the one hash is `checkpoint_matches` (`src/evaluation/sidecar.py`), which `find_checkpoint` calls for each `*.pt` (`record_evaluation.py --checkpoint-dir`) and `register_report` for the one file `record_evaluation.py --checkpoint` names, under any name; the pin moved with it (`tests/unit/test_provenance_pins.py`). Still identification only: the record carries the report's digest of the weights, never this one |
| **19.8** | `scripts/measure_served_pipeline.py` — subject `:1219-1223`; workload `:487-488`, `:523` | `kg.json`, checkpoint and SP table hashed by path before the service starts; the service reads them itself; `val_samples.json` parsed `:487`, hashed `:523`; `kg.json` parsed `:488`, unrecorded | that `served_readings.json` names the bytes the service loaded, or the samples the workload was drawn from | **M3c** for the subject — cite the served identity from the service's own read. The workload read: **a separate fix** on M2.1's `read_samples`, landing **after** S10, since S10 requires this script byte-identical at both commits. M2.1's S10 runs this script, and its evidence states the limitation. **Done in M3c for the subject (implemented 2026-10-16, merged as #124):** the subject is the served identity each repeat's service reports at R1, every part verified; nothing is hashed by path; a counted response must carry that identity and every repeat must report the same one. The workload read is still the separate fix; `PLAN_B04_PRODUCTIONISATION.md` §7.3 is amended in the same change |
| **19.9** | `scripts/audit_split_feasibility.py:456`, `:545` | parses `kg.json` → `inputs.kg = file_sha256(kg_path)` | that `inputs.kg` is the graph analysed | **A separate fix**, on M2.1's `KnowledgeGraph.read_json` (S3) |
| **19.10** | `scripts/audit_checkpoint_family.py` — `inspect_checkpoint` `:117-165` (`:138`), `build_report` `:290` | loads each `*.pt` → keys, widths, score; then hashes every file for `checkpoint_digests` | that each digest names the bytes inspected | **M2.6** — the inventory is "an option of an existing audit". It parses each checkpoint, so its digest comes from that read (M2.1, amended 2026-10-09) |
| **19.11** | `scripts/benchmark_sp_lookup.py` — `build_artifact_lookup` `:245-369` | loads the table (`:298`), checks the `build_id` pairing (`:305`) → `sha256 = file_sha256(path)` (`:352`) | that `sha256` names the table measured | **A separate fix, optional** — record the loaded tensor's `build_id`, as M2.1's SP exception does at serving, and label the file digest as identification of the path |
| **19.12** | `scripts/audit_sp_reachability.py` — `build_report` `:436-569` | memory-maps the table (`:427`) → `artifact_digest = file_sha256` (`:540`); `num_nodes.json` parsed `:482`, unrecorded; the sidecar digest from its parsed bytes (`:544`) | that `artifact_digest` names the table scanned; `num_nodes.json` is unidentified | **A separate fix, optional** — as 19.11, plus `num_nodes.json`'s digest from its read |
| **19.13** | `scripts/build_knowledge_graph.py` — annotation files parsed `:291`, `:298`, hashed `:342-343`; ontology digest `:336` | each source by role and digest → `kg.provenance.json` | that the annotation digests are of the bytes parsed; the ontology digest is the loader's handle digest (`src/ontology/loader.py:243-262`), which survives an atomic rename and not an in-place rewrite | **KG build record v2** — contract §4 "Not prerequisites", named by M3b |
| **19.14** | `_workspace_provenance_status` — `src/kg/artifacts.py:252-300` | parses the manifest again (`:278`), hashes `kg.json` (`:291`) → `provenance_status`; no production caller | that the declaration and graph digest it checks are the run's own reads | **M3b** — "read through M2.1, like every other input". **Done (2026-10-15, code-reviewed with no P1 or P2 at `04ecf42`):** removed; `verify_dataset_record` compares the record with the run's one manifest reading and `kg` digest |
| **19.15** | `_provenance_status` — `src/kg/provenance.py:357-418` | hashes the record (`:381`), compares with the declaration (`:389`), then parses it (`:398` → `:275`, `:283`) and compares its `kg_digest` (`:404`) | that the record matched against the declaration is the record parsed and matched against the graph | **M3b** — same text. **Done (2026-10-15, code-reviewed with no P1 or P2 at `04ecf42`):** removed; the record is hashed and parsed from one `read_once` buffer |
| **19.16** | `write_provenance` — `src/kg/provenance.py:211-233` | staged write → returns `file_sha256(target)` (`:233`), bound into the manifest (`src/kg/workspace.py:315`, `:342`) | that the bound digest is of the payload written | **M3b** ("required at write and read"). **Done (2026-10-15, code-reviewed with no P1 or P2 at `04ecf42`):** the digest of the bytes written, through `DigestingFile` |
| **19.17** | `write_workspace` graph digests — `src/kg/workspace.py:297-300` | hashes `kg.json` and the tensors by path after in-place writes → manifest and record | that the bound digests are of the bytes this writer serialised | **M3b** — as 19.16. **Done (2026-10-15, code-reviewed with no P1 or P2 at `04ecf42`):** `save_json` and `write_graph_data` return the digests of what they wrote |
| **19.18** | Measurement Modes A and B — `build_legacy_mode_a_model`, `scripts/measure_scorer.py:230-262`; `SUPPORTED_MODE_SETS` `:443` | indexes `checkpoint["metadata"]` and `["in_channels_dict"]` (`:251-252`); B ran only beside A. It loaded its own checkpoint path until R2: M2.1's S7 was implemented for Mode C only, and runs that included Mode A were never migrated | that it runs on a checkpoint the current writer produces: that writer (`src/training/callbacks.py:296-327`) writes neither key and no scanned checkpoint carries them (M1), so it raises `KeyError` on those; only fixtures reach it | **Removed by R2** (implemented 2026-10-12; code-reviewed with no P1 or P2 in `1d0b5a6..06fd0f1`), which replaces item 9: deleted with Modes A and B, as 19.19 was by S0 (retirement assessment §4). Superseded by the owner's clarification of 2026-10-12: it was "Item 9, after 7a", the oracle-only surface (`scorer-measurement/README.md`, "What is actually oracle-only"), and 7a is cancelled. M2.1 binds, measures and accepts Mode C only. Nothing replaces this builder: the CLI's Mode A was retired, not rebuilt |
| **19.19** | `scripts/calibrate_mode_a.py` — the digest bracket `:396`, `:403`; `calibration.json` `:433` | path hashes before the oracle and after the harness → `calibration.json` | that the recorded digests are of the bytes either subprocess parsed | **Removed in M2.1's S0** (2026-10-10; code-reviewed with no P1 or P2 at `3433bf8`). The owner's rule of 2026-10-09 is to remove it if nothing necessary uses it; S0's check found no necessary user, and the reviewer accepted S0 at plan level in the re-review of `0708789` and, as revised, in the incremental review of `7e1af96`; the owner has since extended S0 to remove `scripts/evaluate_model.py` and its pin test (contract M2.1 decision 6, 2026-10-10, accepted at plan level in the review of `45d9dea`). Item 7a was to write its own entry point; 7a is cancelled by the owner's clarification of 2026-10-12, so none is written (item 7a's row) |
| **19.20** | Path-reasoning-only serving — `src/inference/pipeline.py:531-532` (cited at M2.1 S6) | no model and no checkpoint: returns before the workspace is read and verified (`:574-578`); the graph was parsed by the API (`src/api/main.py:552`) | that the graph served matches any manifest | **B-2** — contract §8 question 2, not decided. Since M3c (implemented 2026-10-16, merged as #124) the served identity states such serving as it is: checkpoint *absent*, graph *unverified* (the digest of the `kg.json` read, no manifest), dataset record and table *absent*; it claims no verification and decides nothing for B-2 |
| **19.21** | The SP producer's workspace check — `scripts/compute_shortest_paths.py:418-425` | `--kg-path` and `--output-dir` independent; M2.1's S3 fixes the recorded digest, not the placement | that the table lands in the workspace whose graph it came from (serving refuses a mismatch, `src/inference/pipeline.py:787-790`) | **Contract §4 "Not prerequisites"** |

**Parked deliberately, not forgotten:** `task-scope/` Q2–Q5 (settled, unscheduled)
and `scorer-retraining/` (scoping only, four gates uncleared). Neither blocks nor
is blocked by anything above.

**Ontology provenance is unrecorded and its version is not pinnable — raised by
the maintainer, and the sharper half of the packaging problem below.** Verified
against the code rather than recalled:

| | Established |
|---|---|
| Cache location | `Path.home()/'.shepherd'/'ontologies'` — **outside the project**, in the operator's home |
| On a missing file | **Was** `urlretrieve('http://purl.obolibrary.org/obo/mondo.obo')` from a constant in `loader.py`. **Phase 2 changed how, not what:** the sources are configuration, every request and redirect passes the scheme and destination rules, and a fetched file is verified before it replaces anything. It is still **whatever is latest** at the source — Phase 2 lets a site name a file instead; it does not pin a download |
| The `version` argument | Used only as an in-memory cache key; the filename is `mondo.obo` regardless. **It reads as version selection and is not** — **closed by Phase 0**, which refuses any value but `"latest"` rather than ignoring it |
| Is the version knowable? | **Yes** — the OBO header carries `data-version`, and `hierarchy.py` already parses it |
| Is it recorded? | **Was no; closed by Phase 1.** `kg.provenance.json` carries all four source files by role, digest and raw `data-version`, bound to `kg.json`'s digest |
| Can a site select one? | **Was no; closed by Phase 2.** `--mondo-path` / `--hpo-path` name the file; without them, exactly one candidate across the configured roots is taken, and more than one refuses rather than ranking |

This is not hypothetical: the two machines used this session differ in MONDO
vintage (29,866 vs 32,109 disease nodes, different `kg_digest`) purely because
they were deployed on different dates. The differing digest *detects* it, which
is the manifest working — but it cannot say what differed, and a site cannot
deliberately reproduce another site's graph.

**The inert `version` argument is the part that should not wait**, whatever is
decided about pinning: an API that looks like it pins and does not is worse than
one that does not offer it. Recording `data-version` beside the build is the
cheap half and needs no new subsystem. Pinning — where the files live, whether a
mismatch refuses or warns — is a design question with real alternatives and
deserves its own plan. Both were out of scope for 5a.

**Where this stands now.** Phase 0 and Phase 1 of
[`PLAN_ONTOLOGY_PROVENANCE.md`](PLAN_ONTOLOGY_PROVENANCE.md) are implemented and
merged, which closes the two rows marked above. **Phase 2 is implemented** to
[`PLAN_ONTOLOGY_PHASE2.md`](PLAN_ONTOLOGY_PHASE2.md), whose design passed review:
`--mondo-path` / `--hpo-path` on the build CLI, a resolver over configured roots
that refuses rather than ranking, an imports policy that refuses every declared
import, a role check that stops a file being built into the wrong slot, and the
source list moved from `loader.py` into `configs/deployment.yaml` with the
scheme and destination rules enforced on every request. **One item is owed and
undelivered by design**: editing those URLs from the interface (§3.6.1) — the
backend is built for it, the front end is outside this programme.

A whole-branch adversarial review before the PR found the fetch path still
placing fallible steps after the state they should gate — a misfiled cache file
overwritten, a fresh copy published before it was checked — and those are
fixed (§3.8 there). **It also amended one approved default, flagged for
review:** a name this machine cannot resolve is left to a configured proxy
instead of being refused (§3.5.2), because proxy-only hospital networks could
otherwise never download.

**Measured on the homelab GB10 since**
([`EVIDENCE_probe_phase2_homelab.json`](EVIDENCE_probe_phase2_homelab.json)
`d01621f3…`, plan §4.4): the full deployment probe passed 48 of 48 — selection on
the June cache, two byte-identical real builds, and a real download of the
2026-09-01 releases through the production path, byte-identical to the
development sandbox's download-only run
([`EVIDENCE_probe_download_sandbox.json`](EVIDENCE_probe_download_sandbox.json)
`3b1a9beb…`). What routes the redirect hops took, and whether the real build
touched the cache, are not recorded (plan §4.4). That run's
first predecessor, in the sandbox, is what found the HPO regression fixed in
#107. **A hospital network has not been tried**: the files come from
`release-assets.githubusercontent.com`, not from a biomedical host, and only a
run there shows whether it is allowed.

Phase 3 (packaging) is unstarted and converges with the export/import idea
below.

**A workspace export/import pair — the maintainer's idea, recorded so it is not
lost.** Not a review finding and not scheduled. The hop-bound work exposed the
underlying problem: a workspace is seven files plus a shortest-path pair, the
`.pt` and its `.meta.json` belong to one build, and nothing stops a hand copy
from separating them. A packaging step that moves a workspace whole would remove
that class of accident at the source. **Whatever it would do, it replaces no
verification** — the manifest digests, the schema-3 recipe and the hop-bound
checks stay exactly where they are; an importer that trusted its own bundle
would reintroduce what they exist to catch.

---

## 5. Two orderings that are easy to get wrong

**Item 5 did not wait for item 1, and that held.** B-0.4 measures the
shortest-path **lookup cost**: it consumes no checkpoint, no sample split and no
model — only `shortest_paths.pt`. It ran to completion and was reviewed while
item 1 was still open, which is the independence claim discharged rather than
merely asserted. Item **5a** inherits it: productionising A still needs no
checkpoint and no calibration.

**Item 9 waits for everything.** The rename is behaviour-neutral and looks safe,
which is exactly why it must come last: the trainer helper's shape, the
per-sample result contract, manifest ownership and the calibration CLI are not
settled until 1d lands, so renaming earlier buys a second rename. And **nothing
oracle-only may be deleted until 1d has passed review including its institutional
CUDA run** — deleting the old acceptance path before the new one is accepted
would leave the harness with no acceptance at all. One exception, by the owner's
rule of 2026-10-09 and decision of 2026-10-10 (contract M2.1, decisions 3 and 6):
the oracle-parity tests in `tests/integration/test_legacy_equivalence.py`,
`scripts/evaluate_model.py` and its pin test went with the launcher
`scripts/calibrate_mode_a.py` before 1d's institutional run. The tests checked the
parity §3.1.2 retired as an acceptance, the evaluator was that parity's reference,
and its pin test kept the reference's bytes unchanged; the launcher could not run
that parity on the scanned family or on a checkpoint the current writer produces
(§3.1, M1, 19.18); and the differential calibration is untouched.

**Superseded by the owner's clarification of 2026-10-12.** Item 9 is replaced by the retirement
change R2, whose gate is that the migrated tests are in place (R1, implemented and reviewed),
not 1d's institutional CUDA run: 7a is cancelled, and Mode A is deleted rather than
accepted, so no harness acceptance is left for the gate to protect; Mode C's acceptance is direct
(retirement assessment §4, §7). The rule above held until then and is kept as history. R2 was
implemented on 2026-10-12 and code-reviewed with no P1 or P2 the same day (`1d0b5a6..06fd0f1`).

**8a comes before 7b for a reason:** designing B-0.5's output contract after an
expensive institutional run is how required evidence gets discovered too late to
collect.

### 5.0 Is any of this order forced? Checked at file level

Asked because reordering was being considered, and "these are independent" is a
structural claim that decays.

| Line | Files it touches |
|---|---|
| **B-0.4** (item 5) and **5a** | `src/inference/sp_index.py` — `SPLookup`, `build_sp_index`, `sp_mean_distances`; `src/inference/pipeline.py` — `_load_shortest_paths`, `_calculate_sp_score`; `src/inference/scoring.py` — `sp_scores_from_distances`, `validate_hop_bound`; `scripts/benchmark_sp_lookup.py`; `scripts/sp_scan_reference.py`; `tests/unit/test_sp_index.py`; `tests/unit/test_scoring_primitives.py`; `tests/integration/test_benchmark_sp_lookup.py`; `tests/integration/test_pipeline.py` |
| **Calibration** (1b-1e) | `src/training/trainer.py`; `src/evaluation/measurement.py`; `scripts/measure_scorer.py`; new trainer and D3 tests |

**Disjoint, and the layer contract keeps them that way.** The one file that could
have been shared is `src/inference/scoring.py` — it holds both the SP primitives
B-0.4 rewrites *and* the cosine primitives a naive extraction might have reached
for. `src.training` sits **below** `src.inference`, so the trainer cannot import
them; `trainer.py` today imports only `src.training.*` and `src.utils.*`, and
`make lint-imports` reports 4 contracts kept — 3 when this paragraph was
written, and `scorer-independence` was added afterwards to close the direction
the layers contract cannot express. A B-0.4 regression therefore has no
path to the calibration reference.

**Only two orderings were genuinely forced**, both in the table; the first is superseded by the
owner's clarification of 2026-10-12 (item 9 is replaced by R2, which follows R1's moved tests
rather than 1d's institutional acceptance):

- Item 9 (rename, then checklist rewrite, then deletion) after 1d's institutional
  acceptance — deleting the old acceptance path before the new one is accepted
  would leave the harness with none. §5's one exception went first: the
  oracle-parity tests in `test_legacy_equivalence.py`, `scripts/evaluate_model.py`
  and its pin test.
- 8a (protocol design) before 7b/8b — evidence requirements have to be known
  before an expensive run, not after.

**One soft coupling worth naming:** item 2's caveat quotes split file hashes, and
item 10's evidence file would carry hashes for the same claim. If they are
produced on different machines they will not match, and both must then say
**which workspace** they measured. That is the M7 situation and it is a strength,
not a conflict — two independent replications of a generator-level property —
but it must be written that way rather than discovered later.

### 5.1 An asymmetry in the evidence, recorded because it is load-bearing

M6 and M7 have committed artifacts — `EVIDENCE_B04_baseline_synthetic.json` and
`EVIDENCE_B04_artifact_spark.json` sit beside `PLAN_B04.md`, and anyone may
recompute from them.

**M1-M5 had no artifact.** They existed as text pasted into a review thread and
summarised here. That was the wrong way round: M1 and M2 are what established that
the calibration target does not exist — the largest decision this phase has made
— and M4 is what bounds every number the project will report. Those are precisely
the facts that most needed to be independently checkable, and they were the ones a
reviewer had to take on trust.

**Closed.** Four evidence files now sit beside this one, each emitted by a
committed script and cited by digest in §2. M1-M4 were confirmed against the
artifacts, every digit. M5 was not: the recorded figure combined a mean with a
denominator from a different artifact, which is exactly the class of error that
survives indefinitely in prose and does not survive one reproducible run. §2.4
records it.

### 5.2 What that evidence may and may not contain

**Not raw institutional console output.** These runs touch a clinical
deployment, and a console transcript carries filesystem paths, sample and patient
identifiers, and whatever else was on screen. Each item below is an **aggregate
JSON plus the script that emits it** — the script is what makes the number
reproducible, and the bounded schema is what keeps the artifact publishable.

| Fact | The file records | It must not record |
|---|---|---|
| M1-M3 | input digests, checkpoint count, key-presence summary, `in_channels` summary, and the filename-vs-`logs` metric comparison | checkpoint tensors, absolute paths, operator or host names |
| M4 | both split hashes, the two disease counts, and the size of their intersection | any patient id, sample id, or per-disease list |
| M5 | the SP artifact digest **and its sidecar's**, the disease denominator, the **configured** hop bound beside the observed one, the reachable count and percentage, **and the phenotype-selection rule** | per-phenotype rows |

**"A typical phenotype reaches 71.3%" was not reproducible as written**, and
putting it in JSON would not have made it so. The selection rule had to be
operational — which phenotype or phenotypes, chosen how, and whether 71.3% is one
phenotype's value, a median or a mean.

`scripts/audit_sp_reachability.py` states it by **removing the choice**: the
distribution is computed over every phenotype in the graph, and the report's
`selection_rule` says so. Two design points follow.

  - **Phenotypes reaching no disease are counted.** They have no rows in the
    artifact, so a count taken over the table drops exactly the zeroes and reports
    a distribution shifted upward.
  - **The hop bound is the configured one, not the largest distance present.** An
    artifact built to 5 hops whose longest path happens to be 4 would otherwise be
    reported as a 4-hop artifact, and every percentage in it read against a bound
    nobody chose. The configured value is read from the producer's
    `<artifact>.meta.json`; the observed maximum is recorded beside it.

**The run has happened, and it corrected this file's own prediction.** A previous
revision said here that the recorded 71.3% was probably an overestimate *because*
zero-reach phenotypes were being dropped. Measured: **270 of 19,836 phenotypes
(1.36%)** reach nothing — the mechanism is real, the direction was right, and the
magnitude is nowhere near enough to explain a fifteen-point gap. The cause is the
denominator, which came from a different artifact. §2.4 records what was measured
and what the recorded figure actually was. That 1.36% counts *phenotypes reaching
nothing*; it is not the prevalence of unreachable phenotype–candidate pairs, and
§3.3 records why the two must not be substituted for each other.

**No evidence database, registry or index.** Three files beside the plans they
support, exactly as `EVIDENCE_B04_*.json` already sit beside `PLAN_B04.md`.

---

## 6. Contradictions found and resolved in this revision

Recorded because the request that produced this file was to stop them recurring.

| Contradiction | Resolution |
|---|---|
| `PLAN_B04` §5.3.2 required scanning "the deployed artifact" while §10.1 established there is no single one | §5.3.2 restated as evidence about the generator's invariant |
| **This file's own §3.1.2 was silent on precision** while specifying an exact per-sample comparison, and the two paths run at different precisions on CUDA | §3.1.3 added; the verdict now carries the resolved AMP state and `bit_exact_contract` rather than implying one answer |
| **1d's first draft claimed the truth comparison "exercises the `original_indices` translation"** | **False, and self-corrected before review.** Both sides translate through the *same* gather, so it cancels. What it actually tests is that the trainer's `diagnosis_targets` and Mode A's `disease_ids_local` are the same ids — which holds exactly because the gather is injective. The docstring now says that, and cites where `to_global_ids` *is* tested |
| **1d's first draft froze the legacy metric key at import** while `run_modes_ab` rebuilds it per call — and the comment on it claimed the opposite | Resolved per call via `legacy_mrr_key()`. The stale constant would have raised `KeyError` the moment anything moved the truncation, while the comment promised it could not |
| **The shared synthetic cohort is 4 candidates wide**, so a top-20 truncation and no truncation are the same list, and every existing test on it certifies a truncation it never performed | `build_workspace` gained size parameters (defaults unchanged, so no existing caller moved) and 1d added a second cohort wider than `LEGACY_TRUNCATION_K`. `assert_candidate_universe_is_stable` now takes the loader config, because stability is a property of the graph **and** the sampling limits together — and it immediately rejected the first wide config, whose hop-1 limit was below the disease count |
| `PLAN_B04` §9.4's estimated "+7 GB" stood beside §10.2's measured ~24 GB with no link | §9.4 marked as an estimate and pointed at the measurement |
| The `PLAN_B04` status header was spliced into a broken sentence by a scripted edit | Rebuilt as two named gates with their verdicts |
| `--split` defaulted to `test`, which the generator never writes | Made required on both entry points; the error names the splits that exist |
| This file's own first draft proposed calibrating **Mode C** against the trainer's `val_mrr`, on an unverified guess about the trainer's candidate universe | Verified before submission and **withdrawn**: the trainer scores the batch subgraph, so the number is Mode-A-shaped (M8). The direction table now records the withdrawal rather than deleting it |
| Its second draft then proposed calibrating **Mode A** against the *stored* `val_mrr` — right shape, but unreproducible and insufficient | Withdrawn on review. §3.1.1 gives both reasons; the stored value is demoted to a historical sanity reference |
| The legacy-removal checklist instructed deletion of the clamp, local ranking, top-20 semantics, the A/B traversal and two manifest fields — all of which the adopted calibration **keeps** | Checklist **corrected and suspended**, with the oracle-only surface separated out and the removal order fixed at eight gated steps |
| An earlier reply claimed an out-of-range disease id "would be read as if it were already local" | Imprecise. The mapping tensor is `-1`-initialised, so an in-range unsampled id maps to `-1`; an out-of-range id is left unchanged and is not normally a valid local column. §2.3 states the invariant instead of the guard |
| §2.2 credited `_assert_cohort_is_intact` with enforcing the local-truth range invariant | **Wrong citation, right conclusion.** `to_global_ids` (`measurement.py:91-100`) is what enforces it, and `tests/unit/test_measurement_ranking.py:52-56` already covers `[3]`, `[99]` and `[-1]`. Corrected in §2.3 |
| Review held that a `-1` local truth reaches PyTorch negative indexing and silently selects the last candidate, leaving `n_absent` at zero | **Refuted empirically.** `to_global_ids` raises `"local ids must be non-negative"` before any indexing, and `tests/unit/test_measurement_ranking.py:52-56` already covers it. The trainer-side asymmetry proposed in the same reply was itself wrong — see the row below |
| This file claimed the trainer silently scored a malformed truth `0.0` into `val_mrr`, making it an open contract decision (item 1f) | **Wrong, and wrong the same way I had criticised: a fragment traced without its path.** `self.loss_fn` runs before prediction collection (`trainer.py:640`) and `DiagnosisLoss` raises on both label-smoothing branches. Verified by execution. Item 1f is dissolved; the contract is REFUSE and both paths already implement it |
| The first fix for the malformed-truth invariant put the check in `Trainer._compute_model_outputs`, after `_move_to_device` | `bool(cuda_tensor.any())` synchronises host and device **every valid batch**. Moved to `DiagnosisDataLoader._assert_disease_truth_in_range` — CPU, at the boundary that creates the hole. A test now asserts no `bool()` is taken of that tensor in the hot path |
| The trainer test file claimed to characterize "the complete trainer path" | It called `DiagnosisLoss` directly and rested the rest on source ordering. Claim narrowed to loss-level refusal; full orchestration coverage stays with item 1b |
| `benchmark_sp_lookup` chose shape order with `(cell_index + position) % 2`, intending to decorrelate it from the rotated implementation order | With two implementations the rotation moves `position` with `cell_index`, so the sum was **constant per implementation identity** — `current` singleton-first 60/60, prototypes batched-first 60/60. Keyed on the cell alone; regression test now runs the two-implementation configs. Impact on the collected data bounded in `PLAN_B04.md` §12.6 |
| M9 said the project "cannot produce a held-out number at all" | Overclaimed. `--split` accepts `test` and `read_samples` would load it, so the **tools** are not the blocker. Narrowed to: the standard generated workspace has none, and no accepted protocol defines, creates, freezes or proves the independence of one |
| §3.4 said the `--split` help "names both contamination kinds" | **False, and checked.** The help names checkpoint-selection contamination only; `grep -c "overlap\|100%"` over both entry points returns 0. The disease-overlap caveat is item 2 and is blocked on item 10's evidence — the figure must not reach user-facing help ahead of it |
| Item 11 read as "add `test_samples.json`" | Reduced a protocol question to a file. It must first decide **which claim each phase needs**, then the unit: held-out sample views over seen diseases, disease-disjoint (needs retraining), or an external cohort (overlap must be measured). §3.4 tabulates what each can and cannot support |
| §3.4's prose said three places share one decision, but item 8a still depended only on item 1 | Prose without a dependency edge is a wish. 8a now depends on 11 and may not redefine it; `scorer-retraining`'s gate carries a **pointer**, not a copy of the text |
| "No transient memory threat" from A's index build | Broader than the evidence. Narrowed to what was measured: **in the isolated cold benchmark** A added no peak above artifact loading. Integrated steady-state and reload peaks stay open under `PLAN_B04.md` §13 |
| Item 5a read as fully independent of the calibration line | True of its **implementation**, false of its **acceptance**: §13 measures a cold start with a resident model, which needs a designated loadable checkpoint plus compatible artifacts. §3.5 separates the two. Not the authoritative-checkpoint decision, and no registry |
| The absence of a held-out test split appeared only as a parenthetical in item 6 and one line in a phase README | Promoted to **M9** and **item 11**. The `--split` guards stop silent misuse but do not create the split; three places were already waiting on the same undecided protocol (§3.4) |
| M1 was written as "no checkpoint this project has produced" | Overclaimed: no historical audit was run. Narrowed to the current producers and the scanned family, which is sufficient to reject the frozen evaluator as the acceptance oracle |
| Item 10 asked for "the raw scan and audit outputs" | Raw institutional console output carries paths and sample identifiers. Replaced by three bounded aggregate JSONs plus their scripts (§5.2) |
| **1d shipped claiming aggregate MRR as the fourth agreement check while `agreed` ignored it** — review finding, upheld | The aggregate now gates the verdict via `aggregate_mrr_agreed`. On the path the function takes, per-sample agreement implies aggregate agreement, so this is a derived property — but a contract enforced only by a docstring is the defect this file keeps recording, and it survives a change to how either aggregate is obtained |
| **1d shipped an optional `mode_a_result` seam** — an acceptance gate accepting caller-supplied evidence about the thing it gates. Review finding, upheld | **Seam removed rather than field-checked.** Checking patient ids and counts does not repair it: a result from a different negative draw over the *same* patients matches on both. The one legitimate use — one pass, two artifacts — is served better by `DifferentialResult.mode_a_result` carrying the run the function performed, which gives the same saving with nothing to trust |
| **The commit message claimed non-tautology is "enforced by the build"; it was enforced in one direction only** | Verified by experiment, not argument: with a probe `from src.training.trainer import Trainer` in `src/inference/scoring.py`, all three contracts still reported KEPT. Layers are directional — `src.inference` may import `src.training`. A fourth contract, `scorer-independence`, closes that direction and was mutation-checked against the same probe. **Review round 2 narrowed the claim further, and correctly**: no contract can stop both sides delegating to a helper *below* both, and they already share `F.normalize` and `torch.mm` — `masked_mean_pool`'s own docstring says it mirrors the trainer "operation for operation". What the build forbids is a direct cross-stack import in either direction; what the calibration detects is divergence between two maintained copies, not correctness of either. Mode A is a control that preserves the legacy behaviour including its defects, so both being wrong the same way is the design |

**Authority above everything here:** `docs/DISEASE_SCORER_POLICY.md`.
