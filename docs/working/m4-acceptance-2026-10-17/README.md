# Contract M4's acceptance on the GB10, 2026-10-17

**Passed, within the scope agreed**, as the reviewer's final review of the evidence found
(2026-10-17): contract M4's flow -- a test cohort version imported and bound to the graph,
measured by Mode C with every relation checked and R10 recorded, its report kept, registered, and
read back through the read-only view by the checkpoint's digest -- on the demo workspace and on
the real workspace `hpo_2026_0929_5a`, through the production entry points, at `2c94a4e`, together
with every probe M2's, M3a's, M3b's and M3c's acceptance requires. It is a functional acceptance
over synthetic cohorts. It is not a measure of model quality, a capacity reading, an AMP study or
an institutional test, and not a merge. The plan's record is
[`PLAN_PROVENANCE_CONTRACT.md`](../PLAN_PROVENANCE_CONTRACT.md) §4, "M4's acceptance on the GB10".

## 1. The run

| | |
|---|---|
| Commit | `2c94a4ebdaa418d0609041450c4836144b18ddcf`, no tracked file modified (the report's `tested_commit`). It holds M4 as code-reviewed (`59f519d..f3108c4`) with the review's P2 fixed (`29c3d1f`) and the incremental review's P3 fixed (`2c94a4e`) |
| Probe | `scripts/probe_deployment.py --m2-acceptance --device cuda` with `--real-workspace data/workspaces/hpo_2026_0929_5a`, `--real-config` its `train_outputs/config.yaml`, `--real-reference-checkpoint` its `checkpoints/hgt/model-02-0.1813.pt`; a new work directory and the report outside the repository, where the probe's registries live |
| Inputs (SHA-256, before and after) | `config.yaml` `14bf8eea…970e`, `model-02-0.1813.pt` `33a7b39a…8b79`, `kg.provenance.json` `2c7abdb6…12ef`, `split_manifest.json` `131cdf81…f664`, as at M3b's and M3c's acceptances; all four rechecked equal after the run, and again when the evidence was packaged |
| Device and allocator | NVIDIA GB10 (capability 12.1), torch `2.10.0+cu130`, PyG `2.8.0.post1`; the probe's device `cuda`; backend `cudaMallocAsync` from the saved preset, set before torch's first import |
| Outcome | **84 passed**, 0 failed, 0 errors, 0 skipped; probe exit 0, `tee` exit 0. `m2_acceptance` demo, real data and overall complete, T1–T6 and RT1–RT3 among the required probes |
| Checkout | before: `main` at `9fe5113`; restored afterwards to the same, the two records byte-identical, tracked and staged files clean, the original untracked items kept |
| Run by | the owner, with the reviewer's runbook; the reviewer reviewed the evidence the same day |

## 2. What it shows

**On the demo workspace (T1–T6):**
- **T1, import.** A source of 12 cases built from the demo graph's own HP and MONDO ids was
  previewed (the registry unchanged) and committed under the preview id `5a7100ec967fa7fd` as
  version `probe-demo-1` (role `probe-synthetic`), manifest `a1f6068d…1d16`, samples
  `91b77e24…a6e`. Counts: 12 source cases, 10 ranked, 2 excluded (1 ground truth absent, 1 with
  no mappable phenotype), 1 case with a lost term, 2 unmappable terms, 1 repeat removed; the
  version's graph digests are the workspace's.
- **T2, measure and record.** `measure_scorer.py --cohort --registry --record` on CUDA, exit 0:
  report `47f2b207…f30c` kept with its ranks (`9114d3ae…e9e3`, 10 rows); relations R1, R3, R5 and
  cohort↔graph *verified*, R2 not applicable; R10 with both scopes' evidence found in the
  workspace (training: disease overlap 3 of 5 distinct test diseases, 7 cases; phenotype-set
  overlap 2 of 10; validation: disease overlap 1 of 5, 2 cases; phenotype-set *none*, 0 of 10);
  one record (sequence 1) in the ledger beside the checkpoint `83ace3b2…257d`, naming the report
  by its digest.
- **T3.** The same report registered again: already held, the ledger untouched; a damaged copy
  refused, the ledger untouched.
- **T4.** A version mapped against M1's same-shaped other workspace refused before the checkpoint
  was read (0 checkpoint reads observed; differing roles `dataset_record`, `kg`,
  `node_features`); no report directory, the ledger untouched.
- **T5, the view.** With the measured checkpoint served, `GET /api/v1/evaluations/loaded-model`
  answered state `records` with that record, its `checkpoint_sha256` equal to the status's
  served checkpoint; with another checkpoint (`d4230564…e1b0`) served beside the same ledger, state
  `no_records`.
- **T6.** `audit_split_overlap.py --registry --cohort` on the version: report schema 6, its
  cohort version the version's, 3 of 5 diseases shared with `train`.

**On the real workspace (RT1–RT3, before R8):**
- **RT1.** A cohort of 20 of the real workspace's validation cases, written back as
  `shepherd-cases/1` through the real graph's ids, imported with no exclusion and no unmappable
  term (manifest `1fa612e3…13b0`), bound to the real graph.
- **RT2.** R4's checkpoint (`d1819b3b…41b1`, in the work directory) measured with `--record` on
  CUDA, exit 0: report `cc8f8f24…9620` kept, one record; R10 with both scopes' evidence found in
  the real workspace: training disease overlap *none* (0 of 20; the split is disease-disjoint by
  construction), validation phenotype-set overlap 20 of 20, as built.
- **RT3.** The view, with that checkpoint served, shows that record.

**Checked independently from the saved artifacts** (the reviewer at packaging, and again with
this repository's own readers when the evidence was recorded): in both parts the ledger holds one
schema 3 record whose checkpoint, cohort version, samples, report and ranks digests are the bytes
kept, the source kept under its digest, the metrics equal to the report's, the R10 block the
report's, and the read-only view accepting the record. The ledgers' SHA-256 are `8a2f3b7e…70ae`
(demo) and `11d64bad…12b8` (real), as the probe reported.

**Read with these notes:**
- The checkpoints are untrained (demo) or a short training (real: one epoch, paused, resumed for
  the second). The metrics in the reports describe nothing about model quality, and the real
  cohort is the validation cases by design: R10 discloses it, which is what was being checked.
- The run is functional: the reload handler and the read-only API were exercised, the browser
  was not. The same-key contradiction and the corroborating re-measurement are covered by the
  test suite, not by this run.
- R8 compares the real workspace by listing, size and modification time (19 entries), not by a
  content hash.
- The console holds the expected refusals' error lines and the existing GB10/PyTorch capability
  notice. The environment's `dirty` flag comes from an untracked file that was there before;
  no tracked file was modified.
- The institution's evaluation rules and the data's terms of use stay undecided
  (`PLAN_TEST_RESULTS.md` §4.9); a functional pass is not an institutional acceptance. M5 and N2
  are not part of this run.

## 3. Files here

| File | What it is |
|---|---|
| `m4_report.json` | The probe's report, byte for byte (it names no local path and no case id) |
| `exit.txt` | The probe's exit status |
| `SHA256SUMS` | The digests of the files above |
| `.gitattributes` | Turns off the repository's line-ending normalisation here, so the files stay byte for byte |

The rest stays on the homelab in `/home/user/m4-acceptance-2c94a4e-01/` and in the reviewer's
handoff package (`m4-acceptance-2c94a4e-01-handoff.tar.gz`, SHA-256
`e98663f9319a3691c5c26e7cb3a1f11f3df003c24cf9659ba7551837a88e18d8`, every source file listed by
size and SHA-256): the console log (`887e087a…d811`), the check logs, the cohort sources,
registries, reports and ranks, the two ledgers, the overlap audit, the run directories and
inventories. Those name local paths or carry per-case rows, which is why they are not copied
here. The four real-training checkpoints (about 192 MB) stay only on the homelab: the package
carries their SHA-256 and location, not their bytes.
