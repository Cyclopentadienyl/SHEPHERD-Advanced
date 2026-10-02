# Remote branch audit — 2026-10-02

**Scope:** the fifteen remote branches the project owner asked about. Thirteen looked unmerged,
and two (`claude/great-euler-329450`, `claude/fix-core-architecture-CElHS`) had already been
examined and marked for deletion. Every ref was read at `origin/main` = `657f108` (the merge of
PR #116).

**What this does:** it records, for each branch, whether anything on it is missing from `main`,
the evidence, and a mark. **It deletes nothing.** Deleting a remote branch is the owner's action.

---

## 1. A correction first: twelve of the fifteen were never unmerged

The session that started this audit worked in a **shallow clone**. `git rev-parse
--is-shallow-repository` printed `true`, and `.git/shallow` held five graft points. One of them was
`715058a` (`Merge pull request #53`), which showed up as a root of `main`. A real root is never a
PR merge. With `main`'s early history cut off, `git merge-base` found no common ancestor for any
branch created before the cut. Those branches therefore looked as if they carried their entire
history: 7 to 170 commits "ahead" of `main`. The earlier count in that session, fifteen branches
with commits `main` lacks, was an artefact of that cut.

After `git fetch --unshallow origin`, `main`'s roots are `ee5ffc4` and `043092a`, the same as
those branches'. The remote has 58 branch heads (`git ls-remote --heads origin`), one of them
`main`. **Of the 57 besides `main`, three have commits not reachable from `main`**, and they are
three of the fifteen. The other twelve of the fifteen are fully merged. These counts were taken
before this audit's own commit was pushed to `claude/dev-context-review-3wuh05`; that branch
holds one commit outside `657f108` from then on.

**For any later audit of this kind:** check `git rev-parse --is-shallow-repository` before making
any reachability claim.

## 2. Method

Each verdict rests on at least two independent checks.

| # | Check | What it establishes |
|---|---|---|
| 1 | `git rev-list --count origin/main..B` and `git cherry origin/main B` | Commits on `B` not reachable from `main`, and whether any of them has a patch-equivalent (same `git patch-id`) already in `main` |
| 2 | `git merge-base --is-ancestor B origin/main`, plus the merge commit on `main` whose parent is `B`'s tip | That the tip itself is in `main`'s history, and how it got there |
| 3 | GitHub pull-request records (`head.sha`, `merged_at`) | A source independent of the local clone: which tips were merged through a PR |
| 4 | For an unmerged branch: its diff, `git merge-tree --write-tree origin/main B`, and the files and plans that cover the same ground on `main` | Whether the content is already in `main`, superseded, or still unique |

Checks 1 and 2 were also run over all 57 branches besides `main`. Both find the same three
branches outside `main`, and no others. A separate agent re-derived all fifteen verdicts independently,
without seeing this document's results; its agreement is recorded in §5.

## 3. The fifteen branches

| Branch | Tip | Commits not in `main` | Tip is an ancestor of `main` | How the tip reached `main` | Verdict | Mark |
|---|---|---|---|---|---|---|
| `claude/build-gradio-frontend-Uu3nu` | `a10fc92` | 0 | yes | `55e6be3`, PR #41 (head `a10fc92`) | fully merged | pending deletion |
| `claude/fix-api-error-resume-ock9Q` | `c867d9a` | 0 | yes | `e162b72`, PR #6 (head `c867d9a`) | fully merged | pending deletion |
| `claude/fix-gnn-inference-bspgX` | `27fe909` | 0 | yes | `71960ed`, PR #30 (head `27fe909`) | fully merged | pending deletion |
| `claude/fix-thinking-block-error-F4Vbj` | `d98df46` | 0 | yes | `56a0122`, PR #17 (head `d98df46`) | fully merged | pending deletion |
| `claude/gradio-frontend-setup-eiKNV` | `26e60a0` | 0 | yes | `c523fdb`, PR #40 (head `26e60a0`) | fully merged | pending deletion |
| `claude/reload-repo-report-HZipv` | `ea769c7` | 0 | yes | `7306f1c`, a direct merge, not a PR. PR #25 merged the earlier head `ece773a`. Two direct merges on 2026-02-21 took the five commits after it: `e841bf0` up to `2790e08`, then `7306f1c` the tip | fully merged | pending deletion |
| `claude/resume-diagnosis-system-Z6TcA` | `6f4c752` | 0 | yes | `9b3e79e`, PR #47 (head `6f4c752`) | fully merged | pending deletion |
| `claude/review-handoff-docs-Z2At6` | `9a88080` | 0 | yes | `cde72ff`, PR #33 (head `9a88080`) | fully merged | pending deletion |
| `claude/review-project-architecture-fvnuF` | `8246b75` | 0 | yes | `970b742`, PR #48 (head `8246b75`) | fully merged | pending deletion |
| `claude/shepherd-development-continue-YGqtD` | `66ffdaf` | 0 | yes | `8cdaced`, PR #23 (head `66ffdaf`). `8cdaced` is not on `main`'s first-parent line, which reaches it through `e841bf0`, a direct merge of `claude/reload-repo-report-HZipv` (2026-02-21) | fully merged | pending deletion |
| `claude/shepherd-kg-reasoning-Cgxfk` | `750f0fd` | 0 | yes | `c44462a`, PR #7 (head `750f0fd`) | fully merged | pending deletion |
| `claude/step-b-shortest-path-Y8K2N` | `57debd0` | 0 | yes | `715058a`, PR #53 (head `57debd0`) | fully merged | pending deletion |
| `claude/fix-core-architecture-CElHS` | `841810c` | 1, patch-equivalent in `main` | no | its one commit's patch is `b9d5178`, merged by PR #65 (`claude/fix-kg-json-attributes`) | unmerged, equivalent content in `main` (§4.1) | pending deletion (owner, 2026-10-02) |
| `claude/great-euler-329450` | `8b41aeb` | 1 | no | never: no PR was opened | unmerged, the only copy of its file; approach superseded (§4.2). The independent pass classes it as still holding unique work (§5) | pending deletion (owner, 2026-10-02) |
| `claude/analyze-repo-structure-8JzqE` | `a669005` | 1 | no | PR #1 merged the earlier head `fc273ec`; `a669005` was committed 11 minutes later and never merged | unmerged, superseded (§4.3) | pending deletion |

**Deleting a fully merged branch loses no commit**: every commit on it stays in `main`'s
history. "Fully merged" here means every commit is reachable from `main`. It does not mean every
line those commits wrote still stands unchanged. Deleting the three unmerged ones discards `841810c`, `8b41aeb` and `a669005`. Their
contents are described below, so the record survives the branches.

## 4. The three branches with commits not in `main`

### 4.1 `claude/fix-core-architecture-CElHS`: equivalent content already in `main`

`841810c` (2026-05-12) "fix: preserve node attributes in KG save/load JSON roundtrip" adds three
lines to `src/kg/graph.py`, so `save_json` writes `Node.attributes` and `load_json` reads them back.

- `git patch-id --stable` gives `86b1d2d7…` for both `841810c` and `b9d5178`. `b9d5178` has the
  same message and date, and reached `main` through PR #65 (`claude/fix-kg-json-attributes`,
  merge `a5e3c8d`, 2026-05-13).
- `git cherry origin/main` marks `841810c` as `-`, already upstream.
- `git merge-tree --write-tree origin/main` gives a tree identical to `main`'s, so merging would
  change nothing.
- The lines are at `src/kg/graph.py:779-780` and `:848` on `main` today.

### 4.2 `claude/great-euler-329450`: not implemented elsewhere, approach superseded

`8b41aeb` (2026-06-17) adds `scripts/profile_training.py`, a 418-line observation-only
training-throughput profiler. It was written on the day the `torch.compile` experiment was
shelved. It aimed to attribute GB10 training time to Python lines and to compare fp16, bf16 and
fp32 throughput. Its own commit message says it had **not been run on GPU hardware**.

**Not implemented elsewhere.**
- `main` has no equivalent tool; `scripts/` has only `benchmark_attention.py` and
  `benchmark_sp_lookup.py`.
- No document or backlog item refers to it.
- No fp16/bf16/fp32 throughput comparison is recorded anywhere.
- No commit after 2026-06-17 takes the work up.

**The interfaces it calls still exist.** These were checked by importing current `main`:
`train_model.load_graph_data`, `create_dataloaders`, `create_model_from_config` and `TrainConfig`,
plus every `TrainerConfig`, `LossConfig` and `Trainer` member it uses. It would probably run.
**What it would measure is no longer production training:**

1. **It is a copy of the training step, and it differed from the original from the start.** It
   re-implements `Trainer._train_epoch` (its own words: "single source of truth, mirrors
   `Trainer._train_epoch`"). That is the parallel pipeline this project has since decided not to
   build. Production calls `loss.item()` on every step (`accumulated_loss += loss.item()`, present
   at the branch's base `c67ad87` and on `main`), which forces a GPU sync. The copy drops that
   call while saying it preserves production's syncs, and syncs are what it was written to
   attribute.
2. **It does not apply training's allocator rule.** `train_model.py` applies that rule only
   when run directly (`ALLOCATOR_SOURCE = _apply_allocator() if __name__ == "__main__"`, line 95).
   The profiler imports it as a module, so its allocator is whatever the shell provides. With
   nothing set there, that is torch's native allocator rather than the saved preset, which is
   `cudaMallocAsync` unless overridden and has applied to training since 2026-09-29 (`444c710`).
   A shell that exports the same setting would give the same allocator.
3. **It would bypass workspace verification.** `verify_generated_cohorts` is called in
   `train()` (line 717), not in the builders the profiler calls. Since 2026-09-08 every entry
   point refuses an unverified workspace (EVALUATION_COHORTS revisions 21–22); this would be one
   that does not.
4. **Its HGT comparison is narrower than it reads.** `HGTConv`'s `segment_matmul` supports only
   fp32, and `src/models/gnn/layers.py:177-180` has disabled autocast around the HGT
   convolution since 2026-05-14 (PR #76). For HGT, the three precisions differ only outside
   the convolution.

Its line references are also stale. It cites `trainer.py:214` and `:480-538`; the line it quotes
is now `:247`, and `_train_epoch` now spans `:506-634`. The independent pass (§5) ran it on CPU
against a scratch copy of `main`: `--help` worked, and `--phase-timing --profiler --trace-out`
completed on a synthetic workspace. To get that far it had to replace the fixture's
`train_samples.json`, because the fixture's samples point at node ids outside its tiny graph. No
GPU run of it is on record, and its commit reports it untested on GPU.

The question it was written to answer, why training is launch-bound and whether precision
changes throughput, **remains unanswered and is not on the backlog**. If it becomes a priority,
the measurement belongs around the real `train()` entry point (nsys, or `torch.profiler`
around the real `Trainer`), not in a copy of the step.

### 4.3 `claude/analyze-repo-structure-8JzqE`: superseded the same day

`a669005` (2026-01-13 13:57 UTC) "feat: implement ontology module with loader and constraints"
adds `src/ontology/{__init__,base,loader,constraints}.py` (1,651 lines). PR #1 had merged this
branch's earlier head `fc273ec` at 13:46 UTC. `a669005` was committed eleven minutes later (commit
timestamps; when it was pushed is not recorded) and never merged.

- **Superseded 45 minutes later.** `9885a67` (14:42 UTC) "feat(ontology): implement complete
  ontology module with OBO parser, hierarchy operations, and semantic similarity" covers the same
  ground: OBO loading for HPO, MONDO, GO and MP; ancestors and descendants; Resnik, Lin,
  Jiang-Conrath and Jaccard similarity; information content; search; constraint checking. It came
  with 43 unit tests and reached `main` through PR #2 (`claude/fix-api-error-resume-ock9Q`,
  merge `86aadf7`). `main`'s module has since been rebuilt by the ontology provenance phases, and
  `src/ontology/` now holds `download.py`, `resolver.py`, `roles.py` and `settings.py`. `base.py`
  never existed on `main`.
- **Its approach is one later work rejected.** Its loader downloads from hard-coded URLs on
  demand, `urlretrieve` at `loader.py:23,167`. The HPO URL is a `raw.githubusercontent.com` path
  (`:40`); `main` replaced such URLs with OBO PURLs in `cfa4b51`. On acquisition,
  `docs/working/PLAN_ONTOLOGY_PHASE2.md` takes the opposite rules:
  - an explicit path that is not readable refuses and never falls through to a cache (line 141);
  - more than one candidate refuses (line 143);
  - every fetch goes through one downloader at the request boundary (line 358).

  Its parser is a separate matter, and Phase 2 makes no rule about it. When `pronto` is
  missing, the branch switches to a limited built-in parser automatically, after logging a warning
(`:111-115`). `main` imports
  `pronto` unconditionally (`src/ontology/loader.py:28`) and keeps its own `OBOParser`
  (`:648`), which describes itself as legacy and for test fixtures; no production path calls
  it.
- **Merging would conflict.** `git merge-tree --write-tree origin/main` exits 1, with conflicts in
  `src/ontology/__init__.py`, `constraints.py` and `loader.py`.
- **Taking the branch's versions of those files would break `main`'s imports.** They do not
  define `OBOParser`, `OntologyConstraintChecker`, `OntologyFetchError`, `OntologyImportError`
  or `default_cache_dir`, which `main`'s scripts and tests import (found by the independent
  pass, §5). That is the risk of resolving the conflicts in the branch's favour, not of every
  possible resolution.
- **Seven helpers exist only here:** `get_specificity_score`, `rank_by_specificity`,
  `compute_phenotype_coverage`, `normalize_phenotypes`, `get_path_to_root`,
  `OntologyTerm.to_node` and `ConstraintConfig`. `git grep -w` finds none of them in `main`'s
  `src/`, `scripts/` or `tests/`. No plan or backlog item asks for them, and two have problems,
  confirmed by reading the code:
  - **`get_specificity_score`** computes a `max_ic` it never uses and normalises by a fixed
    `12.0`, which its own comment calls an approximation (`constraints.py:365-374`). That is a
    stated shortcut, not normalisation by the ontology's actual maximum.
  - **`compute_phenotype_coverage`** has its direction backwards. Its first loop takes a
    patient term whose *ancestors* include a disease term, so the patient's term is the more
    specific one, but files it as the patient being *more general*. Its third loop applies the
    same condition, so one term lands in both `ancestor_matches` and `descendant_matches`
    (weights 0.4 and 0.8), and the cap at 1.0 hides it (`constraints.py:398-483`).

  If any of them is wanted, it should be rewritten with tests on `main`, not merged.

## 5. Independent re-derivation

A separate agent was given only the fifteen branch names and an instruction to classify each,
read-only, without this document's results. It confirmed the clone was not shallow and used four
methods on every branch: `--is-ancestor`, `rev-list --count`, `git cherry` and `merge-tree`. For
the unmerged branches it also read their content against `main` and ran the profiler on CPU in a
scratch copy of `main`.

| Point | This audit | Independent pass | Resolution |
|---|---|---|---|
| The twelve fully merged branches | fully merged | fully merged; all four methods agree on each | agree |
| How `shepherd-development-continue` reached `main` | `8cdaced` (PR #23) | `e841bf0`, by first-parent line | both hold: `8cdaced` has the tip as a parent, and `main`'s first-parent line reaches `8cdaced` through `e841bf0`. §3 now gives both |
| `fix-core-architecture` | equivalent content in `main` | same, with the same patch-id match and PR #65 | agree |
| `analyze-repo-structure` | superseded | superseded; also found that taking the branch's files would break `main`'s imports, and seven branch-only helpers, two of them with problems | agree. The import breakage and the helpers were verified and added to §4.3. The pass also showed that this document had attributed the built-in-parser fallback to Phase 2's rules; §4.3 now separates the parser from acquisition |
| `great-euler` | unique file, approach superseded | unique file, still worth keeping if a training profiler is needed, after a GPU smoke run and an allocator fix | **differ in judgement, not in fact.** Both find it the only copy of a profiler with no GPU run on record, with the allocator and `loss.item()` drift. The owner marked it pending deletion on 2026-10-02. If its text should outlive the branch, a tag would keep the commit retrievable; that is the owner's call |

Every fact the independent pass reported that this audit had not checked was verified before it
was added, with one exception: its CPU run of the profiler (§4.2) was not repeated, and is
reported as its observation.

## 6. Outside the fifteen

- **The other 42 branches besides `main`** are all ancestors of `main` (check 2), and
  `git rev-list` finds no commit on any of them outside `main` (check 1). They are not marked here. The
  `archive/*` and `backup/*` branches among them look deliberately kept.
- **A finding in a document this audit read:** `docs/TORCH_COMPILE_EXPERIMENT_FINDINGS.md` says
  the `torch.compile` toggle was *not* merged into `main` ("未合入 `main`", "不合入 `main`").
  PR #83 merged it on 2026-06-17: `main` has `compile: bool = False` and `--compile` in
  `scripts/train_model.py` (lines 202, 320), and a toggle in the WebUI. It is off by default. The
  document is recorded as wrong here and is not edited in this commit.
