# Batch 3's short GB10 run, Mode C then Mode D, 2026-10-27

**Passed, within the scope agreed, as the reviewer's review of the evidence found (2026-10-27).**
The run is the one short GB10 run Batch 3 called for (`../BACKLOG.md` §0, batch 3;
`../scorer-measurement/PLAN_B05.md` §8): Mode C, then Mode D, on a registered synthetic cohort,
through the formal measurement entry, recorded. It is a functional check on the hardware. It is not
a measure of model quality, the SP mixture path, a capacity reading, the Models page or an
institutional run. It clears no policy gate (gate 1 and gate 3 stand) and authorises no merge.

**The code tested is `be0ee1a662db8d16a85533c0394244ffcb64ae29`.** Only this record follows it: it
touches documentation alone, so `src`, `scripts`, `tests`, `configs`, `pyproject.toml` and `uv.lock`
are those tested.

## 1. The run

| | |
|---|---|
| Commit | `be0ee1a`, checked out detached in the deployment checkout `/home/user/Desktop/SHEPHERD-Advanced`, clean ([`tested-checkout.txt`](evidence/tested-checkout.txt), [`git-status-before.txt`](evidence/git-status-before.txt), empty); before and after it read `773dc4a`, restored ([`checkout-before.txt`](evidence/checkout-before.txt), [`checkout-restored.txt`](evidence/checkout-restored.txt)). Both reports record `software_revision` `be0ee1a…` and `environment.code.dirty` false |
| Inputs | workspace `m5_webui_demo_01`; checkpoint `runs/gat/20261024-173909_ui-live-stop-02_c1d9cf/checkpoints/last.pt`, SHA-256 `568b1098…56bf`; cohort version `m5-closeout-demo-01` (role `webui-synthetic`), SHA-256 `1c2c6050…96c3`, in the existing registry. Both pinned on the command line (`--expect-checkpoint-sha256`, `--expect-version-sha256`). Their files, the training record and the earlier kept report checked before the run and again after it, unchanged ([`inputs.sha256`](evidence/inputs.sha256), [`input-recheck.txt`](evidence/input-recheck.txt)) |
| Environment | NVIDIA GB10, driver 580.178.04 ([`gpu-before.txt`](evidence/gpu-before.txt)); torch 2.10.0+cu130; `pyproject.toml` and `uv.lock` the committed files ([`dependencies.sha256`](evidence/dependencies.sha256)); `PYTORCH_ALLOC_CONF=backend:cudaMallocAsync` set on the command line ([`allocator-setting.txt`](evidence/allocator-setting.txt)) |
| Commands | [`runbook/02-run.sh`](evidence/runbook/02-run.sh): `scripts/measure_scorer.py --mode C`, then `--mode D`, each with `--device cuda --batch-size 32 --record --outcome`; D only after C exited 0. No training, no service started. Exit codes, measurement and `tee`: C 0 0, D 0 0 ([`C-exit.txt`](evidence/C-exit.txt), [`D-exit.txt`](evidence/D-exit.txt)); the checks of [`runbook/03-check.py`](evidence/runbook/03-check.py) 0 0 ([`check-exit.txt`](evidence/check-exit.txt), [`acceptance-check.txt`](evidence/acceptance-check.txt), every check PASS) |
| Mode C | [`C/report.json`](evidence/C/report.json): `mode` C, `metric_schema_version` 1, `batch_size` 32, `cuda_executed` true; 10 cases ranked; untruncated MRR 0.4733, Hits@1 0.2, Hits@10 1.0, mean rank 2.9 ([`C.log`](evidence/C.log)) |
| Mode D | [`D/report.json`](evidence/D/report.json): `mode` D, `metric_schema_version` 2, `batch_size` 1 (one case per call through the served path; the C side ran at 32, as its regime records), `score_semantics` `served:gnn_only:eta=1.0:sp=none`. The workspace holds no shortest-path table, so the service serves `gnn_only` there too (the console: "No shortest_paths.pt found … pure GNN (eta=1.0 effective)") |
| Case sets | 12 source cases, 2 excluded at import (truth absent 1, phenotype unmappable 1), 10 import-eligible. Mode D: 7 ranked, 3 not reached, 0 refused, 10 in all; 2 of the 3 had no candidates at all (a subset of not reached, never added to it; the log's two "No paths found" warnings). Post-import MRR 0.3833 and coverage 0.7 over the 10; reached MRR 0.5476 over the 7, another denominator, never compared with Mode C's |
| Progress | [`D.log`](evidence/D.log): one line per case (10 cases, fewer than the 20 lines a long pass gets), the last "10 of 10 cases … {'ranked': 7, 'not_reached': 3, 'refused': 0}", equal to the report |
| The comparison | one change at a time over the 7 cases ranked at both ends: C→G1 (the candidate set) 2 moved up, 5 equal, 0 down, and the 3 not reached; G1→G2 (the tie rule) and G2→D (the score) 7 equal each. On these 10 cases the difference between Mode C and Mode D comes from the candidate set alone |
| Kept Mode C reports | `comparison.kept_mode_c_reports` lists this run's C report (`20261027T062536Z-568b10984843-f1ef4038`) and the earlier one kept on the version (`20261025T143321Z-568b10984843-c82e174d`), each `n_compared` 10, `n_equal` 10, none unreadable: Mode D's own C side ranks every case as the separate C runs did. That is the C side against C, not C against D |
| Ledger and usage | the checkpoint's ledger held 1 record and holds 3: the first unchanged, sequence 2 Mode C and 3 Mode D, each naming its own report, with different `scoring_semantics_digest` ([`ledger-before.json`](evidence/ledger-before.json), [`ledger-after.json`](evidence/ledger-after.json), [`ledger-new-C-D.json`](evidence/ledger-new-C-D.json)). The version's usage history keeps its earlier content as a prefix and adds a test run and a registration for each, the test runs naming mode C and D ([`usage-before.jsonl`](evidence/usage-before.jsonl), [`usage-after.jsonl`](evidence/usage-after.jsonl)) |
| Reports | C `20261027T062536Z-568b10984843-f1ef4038`, report SHA-256 `0330d725…9a66`; D `20261027T062539Z-568b10984843-61bfa57f`, report SHA-256 `effb476f…7bbc` ([`C-outcome.json`](evidence/C-outcome.json), [`D-outcome.json`](evidence/D-outcome.json)); the copies here equal the registry's files byte for byte, as the reviewer checked |
| Run by | the owner and the reviewer, on the homelab; reviewed by the reviewer the same day ([`CODEX_REVIEW.txt`](CODEX_REVIEW.txt), in Chinese, as written; the package's own [`README.txt`](README.txt)) |

## 2. What it shows

- **Mode D runs end to end on the GB10** through the formal entry: the served pipeline built by the
  service's own builder, each case ranked by `rank()`, the C side on the same embeddings, the report
  published, registered and recorded apart from Mode C's.
- **The case sets add up** as `PLAN_B05.md` §4 defines them, and the console's progress agrees with
  the report.
- **One Mode C.** Mode D's C side equals the independent Mode C runs case by case, on this revision
  and on the kept report of an earlier one.
- **Nothing outside the run changed:** the inputs read the same before and after, the earlier
  ledger record and usage history are kept, and the deployment checkout is back at `773dc4a`.

## 3. What it does not show

- **The SP mixture path.** This workspace has no shortest-path table, so Mode D ran in `gnn_only`
  (the probe's T7 on the demo cohort did too). The mixture rests on the unit tests and the golden
  test in all three scoring modes, run off the GB10.
- **The Models page.** No screenshot was taken (optional in the agreed steps); the page rests on its
  unit tests and the static render made for `1f51fb2`'s review fixes.
- **Every kernel.** torch warns that the GB10's capability 12.1 is above the 12.0 its build declares,
  as at Batch 1 (`../batch1-acceptance-2026-10-26/README.md` §3). The evidence is the two runs that
  completed, not every operation.
- **Anything about ranking quality.** Ten synthetic cases describe this run only. The figures are
  not evidence for or against either ranking policy.
- **The gates.** Gate 1 needs the institutional run and its protocol (BACKLOG 8a, 8b); gate 3 is
  untouched. Neither moves.

## 4. The files

The package as the owner handed it over, byte for byte: [`SHA256SUMS`](SHA256SUMS) covers its 33
files (all but itself and this README and `.gitattributes`); `sha256sum -c SHA256SUMS` from this
directory checks them. `inputs.sha256` and `dependencies.sha256` name homelab paths: those files
(the checkpoint, the workspace, the registry) are not here. The two logs were added with
`git add -f` past the root `.gitignore`'s `*.log`, and `.gitattributes` exempts their two
trailing-space lines from the whitespace check.
