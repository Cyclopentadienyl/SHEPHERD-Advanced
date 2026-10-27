Batch 3 GB10 C/D handoff — 2026-10-27 (Asia/Taipei)

Tested commit: be0ee1a662db8d16a85533c0394244ffcb64ae29
Verdict: PASS for the agreed synthetic-cohort CUDA C/D functional short test.
Read CODEX_REVIEW.txt for the independent review and limits.

Contents (copied without modifying the original evidence):
- evidence/C.log, D.log; C-exit.txt, D-exit.txt (measurement exit, tee exit).
- evidence/C/report.json, D/report.json and both ranks.json files.
- evidence/C-outcome.json, D-outcome.json identify the published registry reports.
- evidence/ledger-before.json, ledger-after.json, ledger-new-C-D.json.
- evidence/usage-before.jsonl, usage-after.jsonl.
- evidence/acceptance-check.txt, check-exit.txt, input hashes and recheck.
- checkout before/tested/restored records, GPU/dependency/allocator records, runbook.
- CODEX_REVIEW.txt: direct checks against the original registry and ledger.

The old ledger record remains; new C and D records are sequences 2 and 3.
D's kept_mode_c_reports points to this run's C report; all 10 compared ranks agree.
D outcomes: 7 ranked, 3 not_reached, 0 refused (10 import-eligible from 12 source cases).
This workspace has no SP table: served mode gnn_only, effective eta 1.0.
No Models screenshot was taken; it was optional. No model files are included.
The original checkout was restored to detached 773dc4afb5629bad9ec463980ea49b878121492c.

Verify the bundle after extraction, from this directory:
    sha256sum -c SHA256SUMS

SHA256SUMS is relative to this package and covers all included files except itself.
evidence/inputs.sha256 and dependencies.sha256 refer to original homelab paths;
those large or external files are intentionally not bundled. Their recorded checks
and Codex's independent recheck establish the reported preservation on homelab.
This package does not authorize changing any code, registry records or gates.
