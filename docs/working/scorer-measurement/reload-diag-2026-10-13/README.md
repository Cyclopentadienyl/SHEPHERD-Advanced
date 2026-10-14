# S10 limited reload diagnostic: three reloads in one process, GB10, 2026-10-13

A diagnostic, not a reported reading. It answers the question S10's evidence left open
([`capacity-m2.1-2026-10-12/README.md`](../capacity-m2.1-2026-10-12/README.md), §4.5 and §6.1):
does the serving reload's resident difference between `343e90e` and `90a668f` keep growing over
consecutive successful reloads in one server process, and does the service stay healthy? S10's
own evidence, `SHA256SUMS` and `compare.txt` are unchanged by it.

**Result.** The procedure completed at both commits (exit status 0, every reload, readiness check
and diagnosis accepted, no OOM kill, no swap). Under the runbook's reading rules, fixed and
reviewed before the run:

- **Each process grows at every reload, at both commits** (rule 4): about +410 to +470 MB of
  resident memory per reload cycle, +1.2 GB after three reloads, at `90a668f` as at `343e90e`.
  It is present before M2.1 and cannot be attributed to it. A bounded cost is therefore **not**
  supported, and S10's acceptance 2 stays open.
- **The difference between the commits** reproduces S10's first-reload difference (rule 1:
  +32.6 MB), then falls and rises (D_k − D_1 = −5.0, +7.6 MB). It meets neither the plateau
  (rule 2) nor the growth rule (rule 3): **ambiguous** (rule 6).

Nothing here locates the source, and nothing is accepted: not S10's capacity, not M2.1, not M2.

## 1. What was run

| | |
|---|---|
| Commits | before `90a668f1334dfe6f73637dd8f11659ea906b9e1d`, after `343e90e993d0a5129cd547b95276b7e2b8b12fa9`, back to back in one boot (boot id the same in both documents) |
| Subject | `hpo_2026_0929_5a` / `model-02-0.1813.pt`: checkpoint, graph and SP table digests the same in both documents and the same as S10's |
| Tool | `s10_reload_diag.py`, SHA-256 `2d3cfc397b4867fe2cc876b6a58254967e64bec1f5771b695609ddfc963a41ff` (kept outside the repository, like S10's tools; its tests `16f523b5…`) |
| Runbook | `RUNBOOK_S10_RELOAD_DIAG.md`, SHA-256 `798abde883520caa3a1e5b5b32a3c30c897571de6eee5f7a26f385f6dccec99d`, its §4 reading rules revised after the reviewer's review and checked before the run |
| Procedure | per commit, one server process: S10's pre-read, launch, the same 200 requests as S10 (seed 20260930), then three reloads through `POST /api/v1/pipeline/reload`, each followed by a status check, a 10 s settle and 5 diagnoses; VmRSS, VmHWM, RssAnon, RssFile, RssShmem, Pss and the machine's system and swap in use at fixed points |
| Run by | the owner, on the GB10; the reviewer reviewed the results the same day and found them usable |

## 2. The readings (VmRSS, MB = 10^6 bytes)

| Point | before | after | after − before |
|---|---|---|---|
| after the 200 requests | 6445.7 | 6445.0 | −0.7 |
| reload 1's response | 6787.3 | 6819.9 | +32.6 |
| reload 2's response | 7247.5 | 7275.1 | +27.6 |
| reload 3's response | 7660.8 | 7701.0 | +40.2 |
| after reload 3's five checks | 7640.5 | 7676.4 | +35.9 |

Each process's own course (`compare.txt`, "Each commit's own rows"):

| | reload_own k = 1, 2, 3 | step k = 2, 3 | step at the checks k = 2, 3 | cumulative at k = 3 |
|---|---|---|---|---|
| before | +341.7, +543.1, +487.0 | +460.2, +413.3 | +469.3, +466.7 | +1215.2 |
| after | +375.0, +536.9, +499.2 | +455.2, +425.9 | +463.6, +474.6 | +1256.0 |

- The growth is anonymous memory: RssAnon moves with VmRSS to 0.1 MB, RssFile stays at
  411.5–415.1 MB, and Pss agrees.
- Nothing is given back by waiting: the 10 s settle changes nothing. The five checks after each
  reload release 20–83 MB, and each process stays well above its previous level.
- Each reload peaks at about 28.45–28.52 GB (VmHWM after a confirmed reset; about 22 GB above the
  level before it), at both commits, as in S10's R4.
- Reload wall times 78.7–81.6 s; launch 85.7–86.2 s.

## 3. The rules applied

| Rule (runbook §4) | Reading |
|---|---|
| 0 like for like | holds: designated commits, no tracked file modified, every like-for-like line "same", one boot, no GPU process and no server running before either run |
| 5 health | no failure: 6 reloads and 430 diagnoses accepted, 8 resets confirmed at the first try, OOM kills 0 → 0, swap 0 at every point, server exit −15 (the tool's SIGTERM). This covers the recorded counters at the recorded points, not continuous monitoring |
| 1 reproduced | holds: D at `after_requests` −0.7 MB (band −11 to +5), D_1 +32.6 MB (band +16 to +38) |
| 2 bounded cost | does not hold: D_3 − D_1 = +7.6 MB is above 6 MB, and neither process levels off |
| 3 the difference grows | does not hold: D_2 − D_1 = −5.0 MB and D_3 − D_2 = +12.6 MB are not two rises of 11 MB or more |
| 4 each process's own growth | **holds at both commits**: step_2 and step_3 are +460.2 / +413.3 MB (before) and +455.2 / +425.9 MB (after), all far above 11 MB; it comes with the reload (reload_own), and the checks between reloads release only part of it |
| 6 ambiguous | the course of the difference between the commits |

The reviewer's review of the results (2026-10-13) reached the same readings, recomputed every
increment from the points, and reproduced `compare.txt` byte for byte with the tool; so did the
author.

## 4. What can and cannot be said

Can: for this subject, these settings and three consecutive reloads, resident memory rose at
every reload at both commits, by about 0.4–0.5 GB each, while the service stayed healthy; the
difference between the commits was reproduced at the first reload and did not settle within the
bands.

Cannot: that the growth is unbounded or linear over more reloads, that the old pipeline is still
alive, that the allocator kept freed pages, that CUDA or any particular object is involved, or
that M2.1 caused any of it. RSS, RssAnon and Pss cannot make those attributions, and three
reloads say nothing about a longer service.

## 5. Next

**Later (2026-10-13):** the second phase and the investigation's conclusion are in
[`reload-inproc-segments-2026-10-13/`](../reload-inproc-segments-2026-10-13/README.md), and acceptance 2 was
closed with the remaining uncertainty explicit ([`capacity-m2.1-2026-10-12/` §6.2](../capacity-m2.1-2026-10-12/README.md#62-acceptance-2026-10-13)). What follows is the plan
as it stood.

The runbook's second phase: locate the per-reload growth common to both commits first, and treat
M2.1's difference separately. A read-only reading of the reload path (`src/api/routes/pipeline.py`
`reload_pipeline` builds a candidate, then `publish_pipeline` in `src/api/main.py` replaces the
two references `app_state.kg` and `app_state.pipeline`) finds no cache or registry that keeps
the old pipeline; whether its objects stay alive (reachable, or unreachable in cycles not yet
collected) or were freed and kept by the allocator needs a reading inside the process. Any such
diagnostic is proposed with its hypothesis, observations, number of runs and stopping rule, and
reviewed before it runs; no forced collection, trim or second loader goes into the product.

## 6. Files

The six files are as the owner sent them (archive `s10-reload-diag-2026-10-13.tar.gz`, SHA-256
`25132795506cbd3002bb2810b63e14a4d8ef3abde10519e190f123fcac835722`, whose own `SHA256SUMS`
passed); `SHA256SUMS` here lists them. The runbook and the reviewer's report from that archive
are not copied: the runbook is cited by its SHA-256 above, and the report's readings are in §3.
The server logs stay on the homelab and are not evidence.

| File | What |
|---|---|
| `before.json`, `after.json` | the diagnostic's documents, one per commit |
| `compare.txt` | `s10_reload_diag.py compare before.json after.json` |
| `before.exit.txt`, `after.exit.txt`, `compare.exit.txt` | the three exit statuses, all 0 |
