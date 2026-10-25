# M5's WebUI manual acceptance on the GB10, 2026-10-24/25

**Functional: passed for what was planned, with one display fix after it; the WebUI's overall
usability is not accepted.** The manual acceptance ran on the deployment checkout in rounds, each
read by the reviewer from its screenshots, logs and exports. The last round (`773dc4a`, this
directory) supplied what had no evidence yet: a test cohort imported, measured, registered by hand
and read back, Models' record and ordering warning, the operator controls refused to a read-only
browser, and a normal service stop. It found one display contradiction, fixed after it (section 3).
The owner also said plainly that the pages are still overloaded and hard to read: a functional pass
is not a UX acceptance (section 4).

Everything here is **synthetic**: the demo workspace `m5_webui_demo_01`, its existing checkpoint
and a twelve-case cohort taken from the probe's own M4 demo cases. It is not an institutional test,
a model-quality, capacity, AMP or multi-user reading, and it authorises no merge.

## 1. The rounds

| Round | Commit tested | What it covered | Evidence |
|---|---|---|---|
| Functional acceptance of a clean deployment | `b14c4de` | 99 of 99 probes, M5's headless part included | [`../m5-acceptance-2026-10-24/`](../m5-acceptance-2026-10-24/README.md) |
| Training page, first rounds | `b14c4de`, `4764faa` | Training findings F1–F5, fixed in `4764faa` and `1f4eb05` | the design review, [`../WEBUI_UX_REVIEW_2026-10-25.md`](../WEBUI_UX_REVIEW_2026-10-25.md) §4 |
| Local UI recheck | `f9202b6` | F1 and F5 at the real window; a normal start and stop; the existing model's load, a diagnosis and its exports, their identity matching the files on disk; found the written explanation's "Confidence: HIGH", fixed in `ff664f6` | on the deployment machine, `m5-webui-recheck-f9202b6-01`; summarised in the design review §4 |
| Close-out | `773dc4a` | this directory | below |

## 2. The close-out round at `773dc4a`

| | |
|---|---|
| Service | `launch_shepherd.sh`, deployment mode M1, `127.0.0.1:8264`, the server's workspace root, registry and intake directory ([`evidence/service.env`](evidence/service.env)); the service log and the report both name `773dc4afb5629bad9ec463980ea49b878121492c`; the checkout clean ([`evidence/launch-b5rbkl/`](evidence/launch-b5rbkl/)) |
| Inputs | `m5_webui_demo_01` and its existing `last.pt` (`568b1098…56bf`), no training; the cohort source from the b14c4de probe's M4 demo cases, copied to the intake directory ([`evidence/prepared-inputs.json`](evidence/prepared-inputs.json), [`evidence/intake.sha256`](evidence/intake.sha256)) |
| Import | version `m5-closeout-demo-01`, role `webui-synthetic`: 12 source cases, 10 ranked-eligible, 2 excluded (ground truth absent 1, phenotype unmappable 1), as predicted before the run ([`registry/versions/m5-closeout-demo-01/`](registry/versions/m5-closeout-demo-01/)) |
| Measure | from the Test page without "Register the result": the real `measure_scorer.py` through the job runner, exit 0; Mode C, CUDA executed on the GB10, 10 ranked ([`report.json`](registry/versions/m5-closeout-demo-01/reports/20261025T143321Z-568b10984843-c82e174d/report.json)) |
| Register | by hand with Register this report: ledger schema 3, one record, sequence 1 ([`ledger/evaluations.json`](ledger/evaluations.json)); the usage history reads `test_run` (not requested) then `registered` ([`usage.jsonl`](registry/versions/m5-closeout-demo-01/reports/usage.jsonl)); the checkpoint, cohort, samples, ranks, report and ledger digests agree |
| Read back | Result and the report list read "Registered #1"; Models shows the same model and record: Mode C MRR 0.4733, Hits@1 0.2, Hits@10 1.0, over the 10 ranked cases, with the 12 source cases and the exclusions beside them |
| Ordering | By test score on that version: tested and untested models in their groups, the independence warning shown. One model has a result, so this shows the control, the groups, the reading and the warning, not an ordering among several scores |
| Read-only | a second tab at `http://localhost.:8264/ui/` read "Read-only: not addressed to localhost"; Test's Import, Start and Register, Models' list and Load, Settings' Apply and Restart, and Training's Start, Continue and Stop were refused with their reason. The page's refusals only: no request was sent past the page, so this is not a backend security test or a multi-user acceptance |
| Stop | server exit 0, `tee` exit 0; the log ends with "Application shutdown complete" and "Finished server process"; nothing listening on 8264 afterwards |
| Unchanged | the 26 workspace files recorded before the run (the earlier training runs' files among them) and the intake source re-hashed equal ([`verification.json`](verification.json)); the new ledger and registry files are this round's authorised outputs |
| Checkout | left detached at `773dc4a`, clean; not restored to `f9202b6` and `main` not touched |

**What is here and what is not.** The package's text evidence is copied here unchanged, with its
[`SHA256SUMS`](SHA256SUMS) (check with `sha256sum -c --ignore-missing SHA256SUMS`). Its nine
screenshots (about 7 MB) and the reviewer's review text stay in the package and on the deployment
machine (`m5-webui-closeout-773dc4a-01`), named by their digests in that file: `01-import-preview`,
`02-test-kept`, `03-test-registered`, `04-models-record`, `05-ordering-warning`, `06-readonly-test`,
`07-readonly-models`, `08-readonly-settings`, `09-weird`. The checkpoints are not copied; their
locations, sizes and digests are in [`large-artifacts-not-included.json`](large-artifacts-not-included.json).
The package's digests check its own bytes; they are not a signature.

## 3. The display fix after it

Screenshots 03 and 06: Result and the report list read "Registered #1" while the Test run card
beside them still read "Not registered" and said, in the present tense, that no ledger existed. The
card shows what the job runner read as the run ended (it ran without "Register the result"), and the
usage history confirms the order: no wrong, repeated or mixed registration. The card now heads that
reading "When the run ended:", the runner writes it in the past tense ("it was not registered (at
that time: …)"), and the card points to Result for the registration now; a regression runs a real
test, registers it from the page and checks that card and Result no longer disagree. Fixed after
`b326ee3`, reviewed separately; not re-run on the GB10 (it is seen with the next test run made
anyway).

## 4. What it does not show

- **The WebUI's usability.** The owner reports that Test and Models still repeat the model, cohort,
  source and fingerprints and give long paths and engineering detail the space the main actions
  need; with every section collapsed, the Training run bar grows into a large blank area
  (screenshot 09, root cause not located); and proposes bounded, draggable panel splits. These are
  later improvements (BACKLOG 52), not part of M5's close.
- A diagnosis in this round: no model was loaded and nothing was exported; the earlier diagnosis and
  export evidence (`f9202b6`) and the explanation fix's tests stand on their own.
- A backend security test, hospital-LAN access, authentication or several clinicians at once.
- R10 shows training and validation overlap, as this synthetic cohort was built to: not a measure of
  independent test quality.
- Any other dependency set: `torch_geometric` 2.8.1 here; the PyG version is to be pinned before a
  formal release (BACKLOG 50).
