# PLAN — the two open SP producer defects: publication and binding

**Status: draft for review. No file is edited until this is approved.**

**Authority above everything here:** `docs/DISEASE_SCORER_POLICY.md`.
`PLAN_B04_PRODUCTIONISATION.md` §8 records the two defects this closes; this
document is only about how they are closed.

**What this is not.** It adds no scoring behaviour, no new artifact, no second
pipeline, and no input-selection feature. It does not complete any reading of
`PLAN_B04.md` §13. It does not touch `src/inference/sp_index.py`.

---

## 1. Why these two, and why before Ontology Phase 2

The consumer can now refuse a great deal: an unreadable table, missing columns,
a malformed empty one, duplicates, out-of-range ids, distances outside
`1 .. max_hops`, a bound it cannot establish. Every one of those checks asks
*is this table well formed*. None of them asks **is this the table that belongs
here** — that it was written by one finished run, and that it was computed from
the graph it is sitting next to.

That is a correctness question and it ranks above adding new ways to select
inputs. A well-formed shortest-path table computed from a different graph
produces ordinary-looking scores against the wrong node indices, and nothing in
the current path can tell.

---

## 2. The two defects, as they are today

### 2.1 The tensor is written before its sidecar, with nothing binding them

`scripts/compute_shortest_paths.py:348-360`:

```python
def save_shortest_paths(sp_data, output_path, metadata):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sp_data, output_path)                 # <- the multi-GB write

    metadata_path = output_path.with_suffix(".meta.json")
    with open(metadata_path, "w") as f:              # <- and then the ceiling
        json.dump(metadata, f, indent=2)
```

Two files, two writes, no relationship recorded between them. Three ways that
ends badly, and only the first is about ordering:

1. **A failure between the two** leaves a new tensor beside whatever sidecar was
   already there. If none was, the loader takes its "hop bound unknown" path and
   shortest-path scoring stays off — visible, and the weakest of the three.
2. **A new tensor beside a previous run's sidecar** is the dangerous one. A
   present sidecar is *binding* (that is defect 1's fix), so the new table is
   scored against the old ceiling. The floor check catches only the direction
   where the recorded distances exceed the declared bound; a 5-hop table read
   against a declared 7 passes every check and mis-scores every unreachable
   phenotype.
3. **Neither file is written whole.** `torch.save` and `json.dump` both write in
   place, so an interrupted run leaves a truncated file rather than no file.

### 2.2 `--kg-path` and `--output-dir` are independent

`scripts/compute_shortest_paths.py:396-420`: the KG is loaded from one argument
and the artifact is written under another, with no check that they have anything
to do with each other. The sidecar records `kg_total_nodes` and
`kg_total_edges`, which are **counts, not identity** — two different graphs with
the same totals are indistinguishable by them, and the loader reads neither.

Node indices are positions in that graph's mapping. An artifact computed from a
different KG therefore resolves to different nodes, silently.

---

## 3. Swapping the write order is not the fix

Write the sidecar first and the failure mirrors: an old tensor beside a new
sidecar, scored against a ceiling that describes a table that was never written.
Ordering moves which file is stale; it does not let a reader tell.

**What makes it detectable is a relationship recorded in both files**, and that
is what this plan adds. With one, the order genuinely stops mattering — a crash
at any point leaves two files that do not claim each other, and the loader
refuses rather than guessing which is current.

---

## 4. The design: say what this pairs with, then publish whole

Three fields, all in the sidecar except where noted. **Nothing here is new
machinery**: the digest helper, the whole-then-replace publication and the
declared-versus-present split all exist and are used by the workspace writer and
by `kg.provenance.json`.

| Field | Where | What it establishes |
|---|---|---|
| `build_id` | sidecar **and** inside the `.pt` dict | The two files came from one finished run |
| `kg_digest` | sidecar | Which graph the distances were computed from |
| `schema_version` | sidecar | A reader that does not recognise it says so rather than guessing which fields still mean what they say |

### 4.1 `build_id`, not a digest of the tensor — and why, so a reviewer can overrule

A SHA-256 of the `.pt` is the stronger claim: it says *which bytes*, and would
catch a tampered or truncated tensor as well as a mismatched one. It is not
proposed, for one reason: **the loader would have to re-read the whole tensor to
check it, on every cold start.** At the row counts `PLAN_B04.md` §10 records
that is a multi-gigabyte read added to the serving path, to detect a threat that
is not the one observed — nobody is editing the tensor, runs are being
interrupted and directories are being mixed up.

`build_id` costs nothing to check: the loader already has the `.pt` dict open,
so the comparison is a dict lookup. It detects exactly the failure that
happens. **If review prefers the digest, the change is one line in the producer
and one in the loader, and the cost belongs in §13's cold-start reading rather
than being hidden in it.**

`build_id` is a random 128-bit token rendered as hex. It is an identity, not a
recipe — it says two files were published together and claims nothing else.

### 4.2 `kg_digest` costs nothing either, because the pipeline already pays it

`file_sha256(kg.json)` is already computed during pipeline initialisation —
`verify_graph_artifacts` checks it against the manifest, and
`workspace_provenance_status` hashes it again for the provenance binding. The
SP loader compares the sidecar's `kg_digest` against the same value. No new
full-file read.

### 4.3 Publication

Both files are written to a temporary name in the destination directory,
`fsync`ed, and `os.replace`d into place — the shape `write_provenance` already
uses, for the same reason: *a value no encoder takes must not leave a truncated
file that parses as nothing and reads as a record that exists.* The sidecar is
serialised **whole, before the tensor is written**, so a sidecar that cannot be
built refuses before the expensive work rather than after it.

---

## 5. What the loader does with each state

The distinction this project keeps having to re-learn: **absent is not
mismatched, and unknown is not absent.**

| State | Meaning | Treatment |
|---|---|---|
| Sidecar has no `build_id` / `kg_digest` | Built before this change | **Unknown.** Logged once at WARNING, serves. Refusing would break every existing deployment for a claim they never made |
| `build_id` present in both and equal | One finished run | Proceed |
| `build_id` present in both and different | The pair was broken by an interrupted or mixed publication | **Refuse.** This is D1's "present but rejected", not "absent" |
| `build_id` in one file only | Same thing, seen from one side | **Refuse**, with the missing side named |
| `kg_digest` present and equal to the workspace's `kg.json` | Computed from this graph | Proceed |
| `kg_digest` present and different | Computed from another graph | **Refuse.** Node indices mean different nodes |
| `kg_digest` present, no `kg.json` in the directory | Cannot be checked | **Unknown**, logged. A graph-only consumer is a real shape and this plan does not make it a refusal |

A refusal keeps `_sp_ready` False and the existing publication discipline: on
reload the running pipeline keeps serving; at cold start the pipeline refuses
rather than reaching the mock generator. Both are already tested and neither is
changed.

---

## 6. Acceptance

The four the review named, plus three the failure modes above imply. Every one
of them goes **through the real producer and the real loader**; none is asserted
against a hand-built dict.

1. **An interrupted write leaves nothing acceptable.** The producer is failed
   part-way — after the sidecar, after the tensor, and during each — and the
   resulting directory is loaded. Every case either refuses or leaves
   shortest-path scoring off; none is accepted as sound.
2. **A new tensor beside a previous run's sidecar is refused**, and the message
   names the pairing rather than the ceiling.
3. **An artifact written from a different KG is refused** when it lands in a
   workspace whose `kg.json` differs, including the case where node and edge
   counts match — the counts the sidecar records today cannot tell them apart,
   and the test must contain a pair that proves it.
4. **A failed reload keeps the previous service.** Asserted on the real reload
   path, against the app state, not only on the loader.
5. **Old artifacts still load.** A sidecar with neither field reads as unknown,
   not as broken, and serves.
6. **The refusal is the artifact's, not the workspace's.** A rejected SP table
   does not stop the GNN from initialising — the lesson from the provenance
   round, where a check placed inside `verify_graph_artifacts` blocked model
   loading over a note that gates nothing.
7. **Mutation.** Each check has a mutant that fails exactly the test claiming
   it: `build_id` comparison removed, `kg_digest` comparison removed, the
   unknown/mismatched split collapsed, the temp-and-replace reverted to a direct
   write.

`make check` green. No new deployment probe: every item here is deterministic
and CPU-only, and probes are for claims ordinary CI cannot establish.

---

## 7. Sequence

1. This plan reviewed and approved.
2. Sidecar schema and the producer's publication — `build_id`, `kg_digest`,
   `schema_version`, whole-then-replace. Producer tests, including the
   interrupted-write cases.
3. The loader's side — the state table in §5, with the unknown/mismatched split
   and the refusal discipline unchanged.
4. Mutation, `make check`, and the reload/cold-start integration cases.
5. Documentation: `PLAN_B04_PRODUCTIONISATION.md` §8 closes defects 2 and 3;
   `BACKLOG.md` records what the sidecar now carries; the deployment guides
   gain the operator-facing part — what a refusal says and what to do about it.

---

## 8. What this deliberately does not build

No signature or tamper detection; the threat is interrupted runs and mixed
directories, not an adversary. No artifact registry, no manifest for SP files,
no migration tool for old sidecars — they read as unknown and that is the
designed outcome. No change to `sp_index.py`, to the served scoring, or to what
the producer computes. No new CLI surface beyond what recording these fields
requires.
