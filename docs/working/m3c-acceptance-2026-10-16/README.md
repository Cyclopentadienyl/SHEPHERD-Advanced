# Contract M3c's acceptance on the GB10, 2026-10-16

**Passed, within the scope agreed**, as the reviewer's final review of the evidence found
(2026-10-16): the served identity on every result, the pipeline status, the WebUI and its exports,
and serving's device rule, on the demo workspace and on the real workspace `hpo_2026_0929_5a`,
through the production entry points, at `52152e4`. It is a functional acceptance. It is not a
capacity reading, an AMP study, a measure of model quality or a hospital test, and not a merge.
The plan's record is [`PLAN_PROVENANCE_CONTRACT.md`](../PLAN_PROVENANCE_CONTRACT.md) §4, "M3c's
acceptance on the GB10"; the procedure is the section before it, "How M3c's acceptance is run".

## 1. The run

| | |
|---|---|
| Commit | `52152e458beea33d619069f0ded507dd6c1091a0`, no tracked file modified (the report's `tested_commit`). Its product code is the reviewed `1246f0c` plus the review's P3 fix in `src/api/routes/pipeline.py` |
| Probe | `scripts/probe_deployment.py --m2-acceptance --device cuda` with `--real-workspace data/workspaces/hpo_2026_0929_5a`, `--real-config` its `train_outputs/config.yaml`, `--real-reference-checkpoint` its `checkpoints/hgt/model-02-0.1813.pt`; work directory and report outside the repository |
| Inputs (SHA-256, before and after) | `config.yaml` `14bf8eea…970e`, `model-02-0.1813.pt` `33a7b39a…8b79`, `kg.provenance.json` `2c7abdb6…12ef`, `split_manifest.json` `131cdf81…f664`, as at M3b's acceptance; all four rechecked equal after the run, and again when the evidence was packaged |
| Device and allocator | NVIDIA GB10 (capability 12.1), torch `2.10.0+cu130`; the probe's device `cuda`; backend `cudaMallocAsync` from the saved preset, set before torch's first import |
| Outcome | 75 passed, 0 failed, 0 errors, 0 skipped; probe exit 0, `tee` exit 0. `m2_acceptance` demo, real data and overall complete |
| WebUI smoke | after the probe: the service started through the launcher over the real workspace with R4's child checkpoint and no `SHEPHERD_DEVICE` (so `auto`); one diagnosis in the Diagnosis tab, its CSV and Markdown exports downloaded, a reload from the tab with the same checkpoint path |
| Checkout | before: `main` at `9fe5113`; restored afterwards to the same, the two records byte-identical, tracked files clean |
| Run by | the owner, with the reviewer's runbook; the reviewer reviewed the evidence the same day |

## 2. What it shows

**The served identity beside the probe's own reads** (the report's facts):
- **E1** (demo, no table): checkpoint *verified* `70c44c83…9e8a` (the checkpoint D1 trained);
  graph *verified* `kg.json` `7980bf7d…fae1`, `split_manifest.json` `07d1a98a…e903`; table
  *absent*, `gnn_only`; record *verified*, `synthetic`, `9170959d…4396`; `device_requested` `cuda`,
  `device` `cuda:0`. **E2**: the status reports that identity unchanged. **E5**: a refused reload
  leaves it. **E7**: a diagnosis carries it.
- **M4** (demo, with a table): the demo child `f8fed154…99ee`, the same graph, manifest and record,
  table *verified* with `build_id` `e025285c…7dff` and `max_hops` 5, `gnn_plus_shortest_path`.
- **R6** (real): R4's child `e9400952…fd66`; graph *verified* `kg.json` `6cae2d1a…1e43`,
  `split_manifest.json` `131cdf81…f664`; table *verified* `fe1a7d14…ee7b`, `max_hops` 5; record
  *verified*, `files`, `2c7abdb6…12ef`; `gnn_plus_shortest_path`; `cuda` asked, `cuda:0` in
  effect. R5's measurement and R6's serving name the same checkpoint digest as R4.

**The rest of the contract, on the same run:** every probe M3b's acceptance required passed again
(M1–M11, C1, C5, B22, B23, R1–R8 and the writer, workspace, training and serving phases), within
the same limits as there: R8 compares the real workspace by listing, size and modification time,
not by a content hash of the workspace.

**The WebUI and its exports** (screenshots 01–03 here; 04 is kept with the evidence package):
- The model status's Serving line, the result's "Produced by" line and its full-identity block
  name R4's child, the real graph and manifest, table `fe1a7d14…`, the `files` record, `cuda:0`
  in effect with `auto` requested, and the §6 claim boundary.
- The diagnosis (`HP:0001250`, `HP:0001263`, top 10) gave 10 candidates. The CSV has the 15
  candidate columns and the 21 `served_*` columns, 10 rows, each `served_*` column the same on
  every row and equal to the identity above; the Markdown report's "What produced this result"
  carries the same parts.
- `/api/v1/pipeline/status` before and after a reload from the tab, and the `/diagnose` answer for
  the same phenotypes, carry one identical `served_identity` (`webui/diagnose.json` here; the
  status files name local paths and stay with the package). It differs from R6's only in
  `device_requested`: `auto` for a service started with no device, `cuda` for the probe's
  explicit one.

**Read with these notes:**
- Screenshot 01's Data Directory field shows a value saved earlier
  (`hpo_2026_0623_v2_orpha`), not the workspace loaded; screenshot 04, after it was corrected,
  shows the real one. That is BACKLOG 17's F5 (a form field apart from the loaded state); nothing
  shows the older graph was used: every identity names the real one.
- **The HTTP capture's order.** The first capture after the tab's reload ran without a "before"
  baseline (the script lacked its executable bit and was rerun with `bash`), so it is not a
  before/after comparison; it is kept as it is in the package (`webui/after-first-attempt/`). A
  new baseline was then taken, the same checkpoint reloaded again from the tab (21:17:37 local
  time in the server log), and the "after" capture taken; that second reload is the before/after
  comparison recorded, and both checks passed. The server stopped normally (exit 0).
- The real-data checkpoints come from a short training (one epoch, paused, resumed for the
  second). They are not a chosen serving model, and the diagnosis is not a clinical result.
- N2, M4 and M5 are not completed by this run.

## 3. Files here

| File | What it is |
|---|---|
| `m3c_report.json` | The probe's report, byte for byte |
| `exit.txt` | The probe's exit status |
| `webui/diagnose-request.json`, `webui/diagnose.json` | The `/diagnose` request and answer captured over HTTP during the smoke |
| `webui/diagnosis_m3c_smoke_sess_d6ac54f51afa.csv`, [`webui/diagnosis_m3c_smoke_sess_d6ac54f51afa_report.md`](webui/diagnosis_m3c_smoke_sess_d6ac54f51afa_report.md) | The tab's CSV and Markdown exports of that session, as downloaded |
| `screenshots/01-…`, `02-…`, `03-…` | The owner's screenshots of the model status, the candidates and the full identity, unmodified |
| `SHA256SUMS` | The digests of the files above |
| `.gitattributes` | Turns off the repository's line-ending normalisation here, so the files stay byte for byte (the CSV is written with CRLF line ends, as RFC 4180 has them) |

The rest stays on the homelab in `/home/user/m3c-acceptance-52152e4-01/` and in the reviewer's
handoff package (`m3c-acceptance-52152e4-01-handoff.tar.gz`, SHA-256
`c80d87cfbe613627df92c6705331a78ffb0ed45544bac3f56faac4fd8e5a4f23`, every file in it listed by
size and SHA-256): the console log (`8f1e48f5…a9c0`), the server log (`dbc4c791…0011`), the
status captures, screenshot 04 (`07ff5914…d137`), the run directories, measurements and
inventories, and the four real-training checkpoints, kept in place and listed by digest. Those
name local paths, which is why they are not copied here.
