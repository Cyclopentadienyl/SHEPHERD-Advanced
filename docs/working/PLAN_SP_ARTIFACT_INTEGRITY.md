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
is what this plan adds. With one, the order genuinely stops mattering: whichever
file was replaced first, a pair that is half-updated does not claim itself, and
the reader refuses rather than guessing which half is current.

**"A crash at any point leaves two files that do not claim each other" would be
too strong**, and §6.1 is the precise version. Under temp-and-replace most crash
points leave a pair that is entirely intact — the old one before either replace,
the new one after both — and those are accepted. What is refused is the window
between the two replaces, which is the only state in which the files disagree.

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

`build_id` is a random 128-bit token rendered as hex, generated once per
publication. **What it establishes, stated so nothing wider is read into it:**
two files declare the same publication. It is **not** a checksum of the tensor's
bytes, **not** tamper detection, and **not** a freshness claim — two consistent
*old* files are consistent, and say nothing about whether a newer pair exists
elsewhere. Those limits do not weaken it for the failures this plan is about,
which are mixed and interrupted publications.

**Producer and reader ship together.** Emitting `build_id` while readers still
ignore it would leave the artifacts *labelled* and nothing protected, and the
labels would read as protection. The §7 sequence keeps them in one change.

### 4.2 `kg_digest` costs nothing either, because the pipeline already pays it

`verify_graph_source` already computes and returns the manifest-bound digests,
and the pipeline **discards its return value**. Keeping it and passing it to the
SP verifier costs one variable and no second hash.

**The caveat matters more than the saving.** That only holds on paths that
actually ran the verification. A caller supplying `graph_data` in memory did
not, and this must not be generalised into "every consumer has already paid for
a digest" — §5.1 is what that case falls to.

`file_sha256` is the existing shared helper and is reused; no second
implementation and no cache.

### 4.3 Publication

Both files are written to a temporary name in the destination directory,
`fsync`ed, and `os.replace`d into place — the shape `write_provenance` already
uses, for the same reason: *a value no encoder takes must not leave a truncated
file that parses as nothing and reads as a record that exists.* The sidecar is
serialised **whole, before the tensor is written**, so a sidecar that cannot be
built refuses before the expensive work rather than after it.

**The shape is reused; the function is not.** `write_provenance` is bound to
`kg.provenance.json`'s filename and its encoder, and bending it into a generic
writer would make one function answer to two schemas. What is copied is the
sequence — encode whole, temp in the destination directory, flush, fsync,
replace, clean up on failure — into a small writer beside the producer.

**One encoded payload, used twice.** The bytes proved to encode are the bytes
written; the pre-write check does not re-encode. This is not a hypothetical
tidiness rule — the workspace writer's pre-write gate and its writer were two
`json.dumps` calls with different arguments, they disagreed on a counter with
mixed-type keys, and the gate passed input that then failed the writer after
four files were on disk.

### 4.4 The producer records the graph it consumed

`kg_digest` is `file_sha256` of the **KG file this run loaded**, taken in
`main()` from the same path the graph was read from — never of a `kg.json`
found beside `--output-dir`. Hashing whatever is next to the destination would
record the claim the binding exists to check.

**It is required, and refused before any live file is touched.** A publication
with no digest would carry `schema_version` and `build_id` and no graph — a
declared pair under Rule 0, so never legacy, and permanently *unverifiable* at
the one consumer whose ranking depends on it. Nothing a new run writes needs to
be in that state: the producer knows which file it read. A caller with no source
file to hash has nothing to record and does not publish; artifacts written
before this protocol are unaffected, because they are legacy and the readers say
so. The shape rule — lowercase hex, 64 characters — lives once, in the shared
module, and the producer asks it rather than re-deriving a length check, so a
writer cannot publish a pair its own reader refuses.

**The assumption this makes, stated rather than left implicit:** the source file
is not modified while the BFS runs, which at deployment scale is hours. The
digest is taken from the path that was loaded and is recorded as *the input this
run was given*; a file swapped underneath a running job would be recorded
incorrectly, exactly as `file_sha256`'s own docstring already warns for the
general case.

The digest is taken **before the read and again after the job**, and the two are
compared — cheap for one file. Two things about that comparison, because it is
easy to read more into it than it carries:

- **The digest recorded is the one taken at load**, always. The second reading
  is a signal, never a replacement: overwriting the record with an end-of-job
  digest of the same path would record whatever is there *now* and call it the
  input, which is the failure this binding exists to prevent, reached from the
  other direction.
- **It is a comparison, not a lock.** It catches a file that differs at the end
  from the beginning. It cannot detect a change made and reverted during the
  BFS, and it holds nothing still. The snapshot assumption above is what
  correctness rests on; this only tells an operator when that assumption was
  visibly broken.

### 4.5 Every reader of the pair, not only the serving one

Three programs read `shortest_paths.pt` and its sidecar today:

| Reader | What it does today |
|---|---|
| `src/inference/pipeline.py` | the served path; gains the checks above |
| `scripts/benchmark_sp_lookup.py` | loads the tensor and resolves the bound from the sidecar |
| `scripts/audit_sp_reachability.py` | loads the tensor directly and reads `REQUIRED_SIDECAR_INTS` from the sidecar |

**If only the service learns to refuse a mismatched pair, the tools keep
reporting measurements over it**, and the two drift — a benchmark quoting
numbers for an artifact the service will not serve, and a reachability audit
describing a table paired with the wrong graph. The pairing and schema checks
therefore go in a **shared SP artifact reader** that all three use.

That does not mean dragging the pipeline into the tools: what is shared is
reading and validating the two files, not building an index or loading a model.
`audit_sp_reachability.py` was not in the first draft's inventory at all, which
is the sort of omission a reader list exists to prevent.

**The validation does not vary by caller; what the caller does with it may.**
The shared reader returns one result, and two of its states must stay distinct:

| Result | Every caller |
|---|---|
| The pair or the schema is **known invalid** — mismatched `build_id`, a partial declaration, an unrecognised schema, a `kg_digest` that differs | **Refuse.** No caller proceeds over this |
| **The KG binding is unverified** because no comparable graph was supplied | Not a verdict on the artifact. A ranking consumer refuses (§5.1); a tool measuring only the integer ids inside the tensor may proceed **and must label the limitation in its output** |

A reader that uses an *external* graph to interpret the node mapping is in the
first consumer's position, not the second's, whatever kind of program it is.
What varies is whether a caller needs the binding — never whether the files
were checked.

---

## 5. What the loader does with each state

The distinction this project keeps having to re-learn: **absent is not
mismatched, and unknown is not absent.**

**Legacy is a conclusion about *both* files, and it is evaluated first.** The
first draft's table put "sidecar has no fields" at the top and left a new tensor
beside an old sidecar matching two rows at once with no precedence — the exact
state this plan exists to catch could have been resolved as *legacy unknown*.

> **Rule 0.** A table is legacy only when **neither** the sidecar **nor** the
> `.pt` carries any declaration of this protocol. A declaration on either side
> makes the pair a new-format pair, and it is then validated as one: supported
> `schema_version`, every required field present and of the right type and
> shape, `build_id` equal on both sides, and the KG binding. **Nothing falls
> back to legacy from a partial state** — not a missing field, not an empty
> value, not an unknown schema, not a one-sided `build_id`, not a new tensor
> whose sidecar is gone.

Legacy tables still pass everything that already exists: structure, value
domains, and the hop-bound rules. Being legacy exempts an artifact from *this*
protocol, not from the checks it was already subject to. And a legacy artifact
is reported as **unrecorded provenance**, never as *verified*.

| State | Meaning | Treatment |
|---|---|---|
| Neither side declares the protocol | Built before this change | **Unknown.** Logged once at WARNING, serves, and is never described as source-verified |
| `build_id` present in both and equal | One finished run | Proceed to the KG binding |
| `build_id` present in both and different | The pair was broken by an interrupted or mixed publication | **Refuse.** This is D1's "present but rejected", not "absent" |
| `build_id` in one file only | Same thing, seen from one side | **Refuse**, with the missing side named. **Not** legacy |
| Declared but incomplete — a field missing, empty, or of the wrong type | A publication that did not finish, or a hand edit | **Refuse.** Naming the field |
| `schema_version` not recognised | Written by a newer producer | **Refuse.** Which of its fields still mean what they say is a guess — the same rule `read_provenance` already applies |

**Then, and only for a pair that reached "proceed" above**, the KG binding:

| State | Meaning | Treatment |
|---|---|---|
| `kg_digest` equal to the **consumed** graph's digest | Computed from this graph | Proceed |
| `kg_digest` different | Computed from another graph | **Refuse.** Node indices mean different nodes |
| The consumed graph has no trustworthy digest | Cannot be checked | **Not an exemption.** See §5.1 — this is the row the first draft got backwards |

`schema_version` is compared against an explicit list of accepted values, not
tested for truthiness: `0`, `""` and `False` are all falsy and none of them is
"absent".

### 5.1 The binding is to the graph that supplies the node mapping

**Not to whatever `kg.json` happens to sit in the directory**, and the
difference is a hole the first draft left open. `DiagnosisPipeline` verifies
file-backed graph artifacts only when it loads them from disk; a caller that
supplies `graph_data` **in memory** makes no claim about a persisted workspace,
so that verification is skipped by design — and the SP table is still loaded
from `data_dir`. Under the first draft's rule, an SP artifact computed from
graph A, dropped beside graph B's in-memory pipeline with no `kg.json` present,
would have read as *unknown* and been used. The binding this plan exists to add
would have been bypassed by removing a file.

So the question is **does the graph actually supplying the node mapping have a
digest we can trust**, not *is there a file next to the artifact*:

- **It does** — the file-backed path already establishes it.
  `verify_graph_source` returns the manifest-bound digests and the pipeline
  currently discards the return value; keeping it and handing it to the SP
  verifier costs one variable and no second hash.
- **It does not** — an in-memory graph, or any path that did not run that
  verification. Then an SP artifact *declaring* a `kg_digest` cannot be checked,
  and a **ranking consumer refuses**. "Unknown" is available to a reader that
  does not rank; it is not an exemption from a binding the artifact itself
  claims.
- **A consumer that never reads the table** does not reach this gate at all.

Two integration cases prove the binding is to the consumed graph: a new-format
SP artifact beside an in-memory graph with no trustworthy digest, and a
directory whose `kg.json` differs from the graph actually supplying the mapping.
The second is the one that would pass a naive implementation.

### 5.2 What a refusal does, which is unchanged

A refusal keeps `_sp_ready` False and the existing publication discipline: on
reload the running pipeline keeps serving; at cold start the pipeline refuses
rather than reaching the mock generator. Both are already tested and neither is
changed.

---

## 6. Acceptance

The four the review named, plus three the failure modes above imply. Every one
of them goes **through the real producer and the real loader**; none is asserted
against a hand-built dict.

1. **An interrupted write leaves no *mixed* pair usable — and does not condemn
   an intact one.** The first draft said every interruption must refuse or
   switch SP off, which is wrong under temp-and-replace: a failure while
   writing a temporary file has not touched the live pair at all, and that pair
   is still exactly as valid as it was a moment earlier. Refusing it would turn
   a producer crash into a service outage for artifacts nothing happened to.

   | Publication state | Correct outcome |
   |---|---|
   | First build; neither live file published yet | absent / SP off |
   | Update failed before either replace; both live files untouched | **accept the intact old pair** |
   | Exactly one replace completed | **refuse** — this is the mixed pair |
   | Both replaces completed, failure after | **accept the intact new pair** |

   **This is two independent whole-file replacements plus a reader that checks
   the pairing — not a two-file atomic transaction**, and the difference is
   worth stating because it bounds what is promised. The window between the two
   `os.replace` calls is real; what the design guarantees is that a reader
   landing inside it *refuses* rather than serving a mismatch. A pipeline
   already running keeps its in-memory state, so the cost of that window is a
   cold start or a reload attempted inside it, not a corrupted service. No disk
   rollback and no second serving path.

   The tests name the failure point rather than "an interruption": during the
   temp write, at the temp `fsync`, between the two replaces, and after both —
   each against both an existing old pair and an empty directory. One of them
   asserts that a failed temp write leaves the live files byte-identical.

   **Durability across power loss is explicitly not claimed.** That needs a
   directory `fsync` and a statement about the filesystem, and neither is in
   scope; the guarantee here covers write errors and process death.
2. **A new tensor beside a previous run's sidecar is refused**, and the message
   names the pairing rather than the ceiling.
3. **An artifact written from a different KG is refused** when it lands in a
   workspace whose `kg.json` differs, including the case where node and edge
   counts match — the counts the sidecar records today cannot tell them apart,
   and the test must contain a pair that proves it.
4. **A failed reload keeps the previous service.** Asserted on the real reload
   path, against the app state, not only on the loader.
5. **Old artifacts still load** — and "old" is Rule 0's definition, not the
   sidecar's alone. A pair in which **neither side** declares the protocol reads
   as unknown, not as broken, and serves. The test covers both sides, because a
   case written against the sidecar only would pass for a tensor that declares
   one while the sidecar does not — which Rule 0 refuses.
6. **The refusal is scoped to consumers of the ranking, not to every reader
   of the workspace.** Two halves, and the first draft of this item stated only
   one of them and would have been wrong for it.

   *What it must not do:* SP validation does not go inside
   `verify_graph_artifacts`. A graph or training consumer that never reads the
   shortest-path table is not blocked by one being broken. That is the
   provenance round's lesson — a check placed there blocked model loading over
   a note that gates nothing.

   *What it must do:* **a serving candidate configured with a broken SP table
   is refused, and does not publish.** Cold start therefore does not publish a
   pipeline and `/diagnose` answers 503; a reload reports failure and the
   previous pipeline keeps serving. Both behaviours already exist and are
   tested; this plan does not change them.

   **Why those are not in tension, and the rule that decides it.** SP today is
   *in the ranking*: `_calculate_combined_score` computes
   `0.7 x embedding + 0.3 x SP` whenever `_sp_ready`, so a table that is wrong
   reorders candidates. `DISEASE_SCORER_POLICY.md` §2 records exactly this, and
   records that its being a ranking term is **current behaviour**, not the
   target — the approved target is *"optional post-ranking contextual analysis
   only; separate field or panel; no effect on identity, order, rank or ranking
   score"*, and that row's implementation status is **"Not implemented — B-1"**.
   §3.5 also refutes the idea that the term is degenerate: on the deployment
   artifact the median phenotype reaches 64.3% of diseases within the
   configured 5 hops, so the SP contribution genuinely varies across candidates.

   So the contract is stated against the **role**, not against the artifact:

   > A pipeline whose **ranking** consumes the shortest-path table must refuse
   > to publish when that table cannot be trusted. A consumer that does not
   > read it is unaffected, and a consumer that reads it for context outside
   > the ranking is not a ranking consumer.

   Today the first sentence applies, because the ranking does consume it — so
   the rule reduces to the refusal above and nothing here is softened. **When
   B-1 moves SP out of the ranking the same rule yields a different answer by
   itself**, without this being re-argued: a broken contextual panel will not
   be a reason to refuse to serve. Writing it as "a broken SP artifact refuses
   the service" would have had to be found and rewritten then, and might not
   have been.

   This plan does **not** implement the target, add a switch for it, or make SP
   optional. That is B-1's, it is gated, and building it here would be the
   parallel pipeline this programme keeps refusing to grow.

   **One boundary recorded for B-1 now, while the reasoning is in front of
   someone.** When SP no longer affects candidates, order or the ranking score,
   a pipeline may serve GNN results while marking the SP contextual analysis
   **unavailable** — that is the decoupling the target exists for. What may
   *not* follow from it is a relaxation of validation: a broken or
   known-mismatched table must not be displayed as though its numbers were
   valid, and must not be replaced by zero or any other plausible-looking score.
   `unavailable` is already the policy's word for this, and it is the same
   distinction `SP_SCORE_GUIDE.md` draws between a genuine "no path found" and a
   lookup failure that collapses to `0.0`. Lower coupling of *availability*, not
   lower standards for *data*.
7. **Mutation.** Each check has a mutant that fails exactly the test claiming
   it: `build_id` comparison removed, `kg_digest` comparison removed, the
   legacy precedence rule inverted so a partial declaration falls back to
   unknown, the `schema_version` check made truthy rather than a list
   membership, the KG binding satisfied by a directory file instead of the
   consumed graph, and the temp-and-replace reverted to a direct write.

`make check` green. No new deployment probe: every item here is deterministic
and CPU-only, and probes are for claims ordinary CI cannot establish.

---

## 7. Sequence

1. This plan reviewed and approved.
2. Sidecar schema and the producer's publication — `build_id`, `kg_digest`,
   `schema_version`, whole-then-replace, the digest taken from the consumed KG
   and re-checked after the read. Producer tests, including each named failure
   point in §6.1.
3. The shared SP artifact reader (§4.5) and the state table in §5, with the
   legacy precedence rule, the KG binding of §5.1, and the refusal discipline
   unchanged. **Steps 2 and 3 land together**: a producer that labels artifacts
   while readers ignore the labels protects nothing and looks like it does.
4. `benchmark_sp_lookup.py` and `audit_sp_reachability.py` move onto the shared
   reader, so a pair the service refuses is not one the tools quietly measure.
5. Mutation, `make check`, and the reload/cold-start integration cases.
6. Documentation: `PLAN_B04_PRODUCTIONISATION.md` §8 closes defects 2 and 3;
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
