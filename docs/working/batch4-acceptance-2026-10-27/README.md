# Batch 4's minimal GB10 check of `7fc417c`, 2026-10-27

**Passed, within the scope agreed, as the reviewer's review of the evidence found (2026-10-27).**
The check is the minimal GB10 confirmation the reviewer asked for after the review of `ce1ce1d`
and its two fixes (`../BACKLOG.md` §0, batch 4): the deployed service, on the existing checkpoint
and workspace with their shortest-path table, answering the result union through the API and the
Diagnosis page, with its exports, then stopped and restored. No training, no probe, no measurement.
It is a functional check on the hardware. It is not a measure of model quality, medical
performance, capacity or multi-user use, it clears no policy gate and it authorises no merge.

**The code tested is `7fc417c00602023a4bdce24afe5292f5c8ba5625`.** Only this record follows it: it
touches documentation alone, so `src`, `scripts`, `tests`, `configs`, `pyproject.toml` and `uv.lock`
are those tested.

## 1. The check

| | |
|---|---|
| Commit | `7fc417c`, checked out detached in the deployment checkout ([`tested-commit.txt`](evidence/tested-commit.txt)); before and after it read `773dc4a`, restored, the checkout clean ([`checkout-before.txt`](evidence/checkout-before.txt), [`checkout-restored.txt`](evidence/checkout-restored.txt)) |
| Inputs | the real workspace `hpo_2026_0929_5a` with its shortest-path table and sidecar, the checkpoint `m5-acceptance-b14c4de-01/work/real/resumed/checkpoints/last.pt` (model fingerprint `1e5c7273b377…`), the service environment and the runtime settings: eleven files, hashed before ([`inputs-before.sha256`](evidence/inputs-before.sha256)) and checked again after, 11 of 11 unchanged ([`input-recheck.txt`](evidence/input-recheck.txt)). The files themselves stay on the homelab |
| Environment | NVIDIA GB10, driver 580.178.04 ([`gpu-before.txt`](evidence/gpu-before.txt)); Linux aarch64, torch 2.10.0+cu130, CUDA 13.0; allocator preset `cuda_async`, `torch_compile` off ([`runtime-settings.json`](evidence/runtime-settings.json)); deployment mode M1, device `cuda` ([`service.env`](evidence/service.env)) |
| Service | started by the normal launch ([`service.log`](evidence/service.log)): the model on `cuda` (4,228,429 parameters) and its node embeddings computed there; the shortest-path table loaded and bound (431,902,937 pairs, `max_hops` 5 from its sidecar); `gnn_plus_shortest_path`, η 0.7 ([`pipeline-status.json`](evidence/pipeline-status.json), [`ready.json`](evidence/ready.json)). Stopped normally: server and `tee` exit 0 ([`service-exit.txt`](evidence/service-exit.txt)); the log holds no traceback, `ERROR` or HTTP 500 |
| API | one request, session `sess_7c26f2a10f43` ([`request.json`](evidence/request.json): `HP:0001250`, `HP:0001263`, top 10, explanations on), HTTP 200, curl exit 0 ([`diagnose-exit.txt`](evidence/diagnose-exit.txt)). [`diagnose.json`](evidence/diagnose.json): `result_schema_version` 2, `result_kind` `served_mixture/v1`, `eta` 0.7 equal to the served identity's, `candidates_reached` 475, 10 returned, ranks 1..10, `served_score` not increasing and equal to 0.7 × `model_term` + 0.3 × the SP score for each; every `sp_term` `COMPUTED`, 2 of 2 phenotypes mapped, mean hop distance 1, score 0.5, `mixed_as` null; none of the removed fields ([`api-check.txt`](evidence/api-check.txt)) |
| WebUI | a second, independent request with the same inputs, session `sess_6469db6034c4`. The [screenshot](evidence/screenshots/Pasted%20image.png): the kind's label with η = 0.7, the Score, Model term and SP columns, ten candidates; the first's Score 0.836, Model term 0.979 and SP 0.500 as in the API answer; the Explanation view with its Ranking line |
| CSV | [`exports/diagnosis_batch4_smoke_01_sess_6469db6034c4.csv`](evidence/exports/diagnosis_batch4_smoke_01_sess_6469db6034c4.csv), from the WebUI session, downloaded (HTTP 200 in the service log): 10 rows, 41 columns, `result_kind` on each row; the order, names, scores to four places, SP status and identity agree with the API answer, the disease ids without the API's internal `mondo:` prefix |
| Markdown report | **not in the run's exports.** [`review/diagnosis_batch4_smoke_01_sess_6469db6034c4_report.md`](review/diagnosis_batch4_smoke_01_sess_6469db6034c4_report.md) is the report the same WebUI session generated, copied byte for byte from Gradio's export cache by the reviewer: no diagnosis was run again and nothing was regenerated ([`review/RECOVERED_REPORT_SOURCE.txt`](review/RECOVERED_REPORT_SOURCE.txt)). The service log records no GET for it, so its download is the owner's report, not shown here; what was checked is its content: the session, the inputs, the result kind, the identity, ten candidates and their explanations, in agreement with the API answer and the CSV |
| Run by | the owner, on the homelab; reviewed by the reviewer the same day ([`review/CODEX_ACCEPTANCE_REVIEW.txt`](review/CODEX_ACCEPTANCE_REVIEW.txt), in Chinese, as written; the package's own [`README.txt`](README.txt)) |

## 2. What it shows

- **The served mixture with its table runs on the GB10** through the normal launch, the model on
  CUDA, and answers `served_mixture/v1` with η read from its identity and SP records with numbers.
- **The surfaces agree:** the API answer, the Diagnosis page, the CSV and the report give the same
  kind, order and values (two requests with the same inputs, compared candidate by candidate).
- **Nothing outside the check changed:** the eleven inputs read the same before and after, and the
  deployment checkout is back at `773dc4a`.

## 3. What it does not show

- **The other kinds and states on the GB10.** Only the mixture with a table ran. `gnn_only`, path
  reasoning, the demo answer, the refusals and the statuses with no number rest on the unit tests,
  the golden test and the local CPU checks of the delivery.
- **Every kernel.** torch warns that the GB10's capability 12.1 is above the 12.0 its build declares
  (BACKLOG 17); the evidence is the run that completed, not every operation.
- **cuVS.** The page's environment line says cupy is missing; the GPU vector backend is not shown
  to work, and nothing here depends on it.
- **The test suite.** No test was run on the GB10; the two Training tests of BACKLOG 60 stay
  failing and apart.
- **Anything about ranking quality.** One synthetic query describes this check only.
- **The gates.** Gate 1 and gate 3 stand.

## 4. The files

The package as the owner handed it over, byte for byte: [`SHA256SUMS`](SHA256SUMS) covers its 25
files (all but itself, this README and `.gitattributes`), and [`review/SHA256SUMS`](review/SHA256SUMS)
the review's four; `sha256sum -c SHA256SUMS` from this directory checks them. The input hashes and
[`review/ORIGINAL_EVIDENCE_SHA256SUMS.txt`](review/ORIGINAL_EVIDENCE_SHA256SUMS.txt) name homelab
paths, kept for tracing: those files are not here. The two logs were added with `git add -f` past the
root `.gitignore`'s `*.log`; `.gitattributes` keeps every file's bytes (the CSV ends its lines in
CRLF) and exempts the service log's trailing spaces from the whitespace check.
