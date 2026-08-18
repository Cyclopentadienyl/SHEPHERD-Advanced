# B-0.3 — Modes B and C

**Status:** proposal. Three decisions, then implement.

Modes B and C need almost no new machinery. Nearly everything they run on was
either built in B-0.2 or already exists in production, and the point of this
document is to name the three places where a wrong choice would silently produce
**a control that is not a control** — which is the only failure mode this work
item cannot recover from.

---

## 1. What already exists

Verified in the tree at `0711441`, not assumed:

| Needed | Where it already is |
|---|---|
| Full-graph forward | `src/inference/pipeline.py:884-921` — `_precompute_node_embeddings` runs `self.model(x_dict, edge_index_dict)` over the whole graph once at load and caches the result |
| Scoring a whole candidate matrix | `src/inference/scoring.py:cosine_score_matrix` (B-0.2). The production per-candidate scorer's own comment says it: *"scoring the whole disease universe is the same call with a taller candidate matrix"* (`pipeline.py:1471-1473`) |
| Two ranking streams, rank extraction, cohort guards | `src/evaluation/measurement.py` (B-0.2) — mode-agnostic already |
| Manifest, artifact digests, CUDA gate, seeded launcher | `scripts/measure_scorer.py`, `scripts/calibrate_mode_a.py` (B-0.2) |
| Candidate universe for Mode C | `arange(num_nodes_dict["disease"])`. No construction to design |

**Memory is not an open question.** Production performs the full-graph forward at
every startup on the same hardware, so the encoder B and C need is already known
to fit.

---

## 2. Decision 1 — which model construction do B and C use?

**The problem.** Mode A uses `build_legacy_mode_a_model`, which mirrors the frozen
evaluator: architecture from `checkpoint["metadata"]` and `["in_channels_dict"]`,
with hardcoded fallbacks. Production does something deliberately different
(`pipeline.py:780-796`): it derives `metadata` from
`graph_data["edge_index_dict"]`, with a comment stating that using the KG's own
metadata instead would be wrong, because the loaded graph carries reverse edges
the KG object does not.

Modes B/C/D may not import the legacy loader — it retires with the frozen oracle.
So B uses production's resolution and A uses the oracle's. **If the two produce
different models, A→B is not "encoder scope" — it is encoder scope confounded
with architecture resolution**, and the ladder's first rung stops meaning
anything.

**Proposed:** treat the equality as a *precondition to be checked*, not a
property to be assumed. Before Mode B reports anything, assert that the two
constructions produce the same architecture and identical parameter tensors for
the checkpoint in hand; fail the run if they do not, naming the fields that
differ. If they do differ on institutional data, that is itself a finding worth
having before any B/C number is quoted, and the ladder gains an explicit extra
rung rather than a silent confound.

*Alternative considered and rejected:* have B use the legacy loader "just for
comparability". That keeps the confound and adds an import the reviewer has
already ruled out.

---

## 3. Decision 2 — how does Mode B obtain Mode A's candidate sets?

Mode B is defined as *the same candidates as A, scored with full-graph
embeddings*. `run_mode_a` does not persist per-sample candidate global ids: it
writes the legacy top-20 and aggregate sampler evidence, neither of which
reconstructs the universe.

**Proposed:** re-run the same seeded dataloader and take the candidate sets from
the batches, exactly as A did. The seeding machinery, worker-count discipline and
digest checks already exist in `calibrate_mode_a.py` and are proven to reproduce
the same batches across processes. A and B then differ only in what is done with
each batch.

*Alternative considered:* have Mode A emit a per-sample candidate-id artifact. It
grows Mode A's output by the candidate count for every sample and makes B depend
on an artifact rather than on a procedure that is already tested. Rejected unless
re-running proves too slow on institutional data, which is a measurement, not a
guess.

---

## 4. Decision 3 — where does the offline full-graph forward come from?

Two options, and this is the reuse-versus-duplication call:

1. **Construct a `DiagnosisPipeline`** and use its cached embeddings. Exact
   production behaviour by construction — but it drags in KG object loading, the
   shortest-path table, path-finding state and explanation machinery, none of
   which B and C use, and it makes the harness depend on the pipeline's whole
   initialisation contract.
2. **Call `model(x_dict, edge_index_dict)` directly** under `no_grad`, after
   moving tensors to the device. Three lines, and the same three lines production
   runs.

**Proposed: option 2, with the equivalence asserted rather than asserted-in-prose.**
The harness computes the embeddings itself and a test checks that the result
matches what a constructed pipeline caches for the same inputs. That keeps the
harness free of the pipeline's initialisation surface while making the "same
three lines" claim a checked one.

**Mode D is the exception and must use option 1**, because Mode D's whole content
is what production actually does. That is B-0.5's problem, and it is a real one:
the public entry point truncates to `top_k` (`pipeline.py:1268`) and runs path
construction and explanation, so obtaining an untruncated pre-`top_k` rank without
duplicating candidate construction is not a detail.

---

## 5. Not in this stage

| Excluded | Why |
|---|---|
| Mode D | B-0.5. Unresolved design problem above |
| Statistical protocol, institutional run | B-0.5 |
| Fixing candidate construction or negative sampling | The measurement exists to decide whether that is needed. Changing it first destroys the baseline |
| Any scorer change | Work item B proper |

**B-0.4 is smaller than its name.** "Vectorised SP lookup" reads like a rewrite,
but `sp_mean_distances` (`src/inference/scoring.py:240`) **already takes a
sequence of targets and returns a `(C,)` tensor**. The pipeline calls it with a
one-element list, once per candidate (`pipeline.py:1399`). The vectorisation is
therefore a caller change in production, not a new primitive — and the offline
harness can pass the whole candidate list from its first line. This is recorded
here so nobody plans a rewrite for it; it stays out of B-0.3 because it changes
production behaviour and belongs with a benchmark.

---

## 6. Acceptance

Same shape as B-0.2, and the same refusal to overclaim:

- Modes B and C run over the synthetic fixture and produce both metric families,
  with the manifest recording which mode, which candidate universe and which
  model construction;
- the architecture-equality precondition (§2) is asserted, not assumed;
- the offline-encoder equivalence (§4) is asserted against a constructed pipeline;
- cohort integrity holds: in Mode C the ground truth is always in the universe, so
  absence is impossible and fatal, exactly as in Mode A;
- **no authoritative comparison between modes is claimed from a CPU synthetic
  run.** A→B→C on institutional CUDA hardware is a separate acceptance gate, as
  Mode A's parity is.
