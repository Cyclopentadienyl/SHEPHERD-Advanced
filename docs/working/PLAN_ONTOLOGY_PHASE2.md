# Ontology Phase 2 — explicit selection, a resolver, and a stated imports policy

Phase 2 of [`PLAN_ONTOLOGY_PROVENANCE.md`](PLAN_ONTOLOGY_PROVENANCE.md), whose §4
table describes it in one line: *"a path per ontology on the build CLI, a
resolver over configured roots, and a stated imports policy (§4.2)."* This
document is what that line expands into, and it carries two things the parent
plan could not: **measurements it recorded as unverified**, and a **proposed
revision to one of its approved decisions**.

Phase 0 and Phase 1 are done and merged. Phase 1 records what a build consumed —
every source file by role, SHA-256 digest and raw `data-version`, bound to the
graph it describes. **What it cannot do is let anyone choose.** The build still
reads `<cache_dir>/<name>.obo` by convention, so the record answers "what was
used" for an input nobody selected.

---

## 1. What is actually there now

Verified against source rather than recalled. Line numbers are at `5870d43`.

### 1.1 The library can already open a file; the entry point has no route to it

`OntologyLoader.load(path)` (`src/ontology/loader.py:104`) takes a file and
parses it. `scripts/build_knowledge_graph.py` never calls it: it constructs
`OntologyLoader(cache_dir=ontology_cache_dir)` (`:153`) and calls `load_mondo()`
/ `load_hpo()`, which reach `_load_known_ontology` and open
`<cache_dir>/<name>.obo`, falling back to `<name>.owl`, falling back to a
download (`:167-180`).

**The gap is in the entry point, not the library.** The CLI exposes
`--ontology-cache-dir` (`:346`) and nothing finer. This is the parent plan's
§3.4 and it is the whole of Phase 2's first deliverable.

### 1.2 The curated URL list is a class constant

`ONTOLOGY_URLS` is defined at `src/ontology/loader.py:45` as a class attribute,
with `ONTOLOGY_OWL_URLS` beside it at `:53`. **A PURL that stops resolving is
therefore a code change**, which is the maintainer's objection and it is correct
against the current implementation.

There is already a manual-placement story, but only in the failure path:
`ONTOLOGY_MANUAL_INSTRUCTIONS` (`:193`) and the `RuntimeError` at `:238` tell an
operator where to download from and where to put the file. Phase 2 promotes that
from *what the error message says after everything failed* to *the first-class
path*.

### 1.3 The cache directory is keyed by filename alone

`_load_known_ontology` opens `<cache_dir>/<name>.obo` by convention. Nothing
checks that the file at that path is the ontology the name claims — an
`hpo.obo` holding MONDO loads without complaint, and Phase 1 records its digest
and `data-version` faithfully. **Provenance records what was opened; it does not
decide what should have been.**

`__init__` also calls `self.cache_dir.mkdir(parents=True, exist_ok=True)`
(`:64`) unconditionally, so merely constructing a loader creates a directory.
Phase 2's resolver reads roots and must not do this.

### 1.4 The OWL fallback is a second parse path, and it is not the measured one

`_download_ontology` tries OBO, then OWL (`:216-226`), and `_load_known_ontology`
will open a cached `.owl` if no `.obo` is present. **Every measurement in §2
below is of `.obo` products only.** The OWL products of these ontologies are a
different artifact and this plan makes no claim about them.

### 1.5 `pronto`'s import default is unbounded

`pronto 2.7.3`, `Ontology.__init__(..., import_depth: int = -1)`. The loader
passes no override, so a root file carrying `import:` lines resolves them
without bound, over the network, at parse time — and the root file's digest does
not cover what they contributed.

---

## 2. The measurement the parent plan asked for

§4.2 recorded the exposure as **conditional**: *"Unverified: whether the MONDO
and HPO artifacts these deployments use actually declare imports — no such file
exists in the environment this plan was written in."* It does now, by fetching
only each file's header with an HTTP range request.

| Artifact | `data-version` | `import:` in the header stanza |
|---|---|---|
| `hp.obo` | `hp/releases/2026-09-01` | none (header ends line 29) |
| `mondo.obo` | `releases/2026-09-01` | none (header ends line 115) |
| `go.obo` | `releases/2026-07-26` | none |
| `mp.obo` | `releases/2026-08-25/mp.obo` | none |

All four entries of `ONTOLOGY_URLS`, including the two no caller loads.
**The exposure §4.2 described is not realised by these artifacts as published
today.**

**Three limits on that sentence, because it is narrower than it sounds.** It is
one release of each, measured on one day; it says nothing about past or future
releases. It covers the `.obo` products only, and §1.4 shows the OWL path is
reachable. And it says nothing at all about **a file an operator supplies by
hand** — which Phase 2 is about to make the primary path.

**So the policy is needed more after this measurement, not less.** The four
curated artifacts stop being the only thing that gets parsed.

### 2.1 What `import_depth` actually does, measured

This decides the policy's implementation, so it was run rather than assumed. A
root file declaring one unreachable import:

| `import_depth` | Result |
|---|---|
| `0` | **Loads silently.** One term, no warning, `ontology.imports == {}` |
| `1` | Raises `URLError` |
| `-1` (the default) | Raises `URLError` |

And in every case, `ontology.metadata.imports` still carries the declared set —
`{'http://example.invalid/nonexistent.obo'}` — at depth 0 as well.

**`import_depth=0` alone would be the defect §4.2 forbids.** It does not make a
build self-contained; it makes an incomplete build look like a complete one, and
the graph that comes out is missing whatever the import contributed with nothing
saying so. The metadata is what makes an honest refusal possible.

---

## 3. The design

### 3.1 A path per ontology on the build CLI

`--mondo-path` and `--hpo-path`, aligned with the existing `--external-dir` that
the annotations already use, and **keyed by ontology name** per the parent plan's
§3.2 so that `go` and `mp` need a table entry rather than a schema change.

Precedence, stated because a silent fallback is what this programme keeps
removing:

| Given | Taken |
|---|---|
| An explicit path | That file. A path that does not exist **refuses**; it never falls through to a cache |
| No path, a resolver hit | The resolved file, with the root it came from logged |
| No path, no hit, download available | Download, as today |
| No path, no hit, no download | Refuse, naming the roots searched and the manual instructions already in `ONTOLOGY_MANUAL_INSTRUCTIONS` |

### 3.2 The resolver

A pure function over the filesystem: given roots, enumerate the ontology files
present with each one's identity — `data-version`, digest, size, term count.
No database, no install step, no state, and **no directory creation** (§1.3).

Roots come from `configs/deployment.yaml`'s `paths:` block (line 148), which
already holds `workspaces_root`, `cache_root` and friends — the parent plan's
§3.1 placed `ontology_roots` there and nothing here changes that.

### 3.3 The imports policy: **narrow**, and refused explicitly

**Require self-contained files. An input whose imports cannot be satisfied
locally is refused, by name.**

Implemented as measured in §2.1, and the order matters:

1. Parse with `import_depth=0`, so **nothing is fetched at parse time**.
2. Read `ontology.metadata.imports`.
3. If it is non-empty, **refuse**, naming each declared import and saying that a
   self-contained file is required.

Step 3 is the whole policy. Step 1 alone is silent suppression, which §4.2 rules
out and §2.1 measured. The wider option — support imports and record each
resolved dependency beside the root — stays available and is not proposed now:
it needs a resolution story, a digest per dependency and a provenance schema
change, and nothing measured says any artifact in use needs it.

**This is a refusal a real artifact might one day hit.** If a future MONDO
release declares an import, builds stop with a message naming it, and that is
the intended behaviour: the alternative is a graph silently missing what the
import carried.

### 3.4 Revising the parent plan's §3.3 — the curated list becomes operator-editable

**This changes an approved decision and is flagged rather than folded in.**

§3.3 reads: *"Recommendation: a curated source list, never a free-text URL
field"*, on the grounds that a user-supplied URL is fetched by `urlretrieve` and
handed to pronto, and arbitrary fetch-and-parse is a security surface in a
hospital.

The maintainer objected, and **two of the arguments behind §3.3 do not survive**:

- **"A wrong ontology silently corrupts the knowledge base" does not
  discriminate between the two designs.** A hand-placed wrong file has exactly
  the same effect and exactly the same record. It is a real risk of ontology
  selection in general; it is not an argument about where the URL comes from.
- **"Fetching unknown data is a risk" is largely the network's job.** A
  deployment that lets this server reach the internet has accepted that class
  already, and by that reasoning every program on every networked host in the
  building is the same exposure. The observation is correct.

What survives is narrower, and **all of it argues for constraining the field
rather than refusing it**:

| Residue | Answer |
|---|---|
| `urlretrieve` accepts `file://`, `ftp://` and `data:` — the opener's handlers include `FileHandler`, `FTPHandler`, `DataHandler` | Scheme allowlist: `http`/`https` only. A field named "download URL" should accept only URLs |
| Unbounded import chasing turns one named URL into unnamed fetches | §3.3 above. Owed regardless of where URLs come from |
| Who can supply one | Operator-facing configuration, not a field any UI user can type |

The last row is the only one that distinguishes the designs at all, and it
distinguishes *configuration* from *a form field*, not *editable* from *fixed*:
someone who can edit `deployment.yaml` already has filesystem access to the
deployment host and could place any file anywhere. **The list being editable
gives that person no capability they lack.**

**So: the curated list moves from `loader.py:45` into `configs/deployment.yaml`,
beside `ontology_roots`.** A PURL that rots becomes an operator edit rather than
a code change — the maintainer's actual requirement — while the field is not
free text typed at run time by an arbitrary user.

A reviewer who thinks §3.3 should stand as written should say so against this
section; the argument is recorded here precisely so it can be rebutted.

### 3.5 What this does not build

- **No UI.** The parent plan's §4.1 puts the front end outside this programme,
  and that is unchanged. Phase 2 ships the backend that makes a settings slot a
  small change later: one list, in configuration, keyed by ontology name, with
  the constraints enforced in the loader. Whoever builds that surface consumes
  it; they do not redesign it.
- **No registry, no version negotiation, no install-state database.** A path is
  enough.
- **No removal of `force_download`.**
- **No refusal on version mismatch.** Still §4.1's call: make it visible and
  comparable first.
- **No packaging.** That is Phase 3 and it comes last, for the reason the parent
  plan gives — done first it produces a bundle that cannot say what is inside it.
- **No change to the OWL path's behaviour** beyond applying the same imports
  policy to it. §1.4's gap is recorded, not closed.

---

## 4. Acceptance

Named before the work rather than argued after it. Ordinary unit tests.

1. **An explicit path is used, and a wrong one refuses rather than falling back.**
   `--mondo-path` pointing at a file that does not exist stops the build; it does
   not quietly open `<cache_dir>/mondo.obo`.
2. **The resolver reports identity, not just presence** — `data-version`, digest,
   size and term count for each file found under the configured roots.
3. **Constructing a resolver creates no directories.** §1.3's `mkdir` is the
   behaviour being kept out of the new path.
4. **A file declaring an import is refused, and the refusal names the import.**
   The measured trap: it must not load with the import silently dropped.
5. **A self-contained file loads with nothing fetched at parse time**, so the
   refusal above is not a reader that rejects everything.
6. **The source list is read from configuration**, and a deployment that edits a
   URL there changes what a download attempts without touching `loader.py`.
7. **A non-http scheme in that list is refused** when it is read, naming the
   entry — not at fetch time, and not by `urlretrieve` deciding what to do with
   it.
8. **Provenance is unchanged in shape and now describes a selected input**: a
   build given explicit paths records the same four roles with the same digests
   and declared versions Phase 1 records today.

---

## 5. Sequence

1. The resolver and its tests, as a pure function over roots. Nothing calls it yet.
2. The imports policy in `OntologyLoader.load`, with the refusal and its tests.
   Independent of everything else and the only item that changes existing
   behaviour.
3. The source list into `configs/deployment.yaml`, with the scheme allowlist.
4. `--mondo-path` / `--hpo-path` on the build CLI, wired to the resolver, with
   the precedence table of §3.1.
5. Acceptance matrix, documentation, and the operator-facing lines in the
   deployment guides.

Steps 1 and 2 are independent and could be reviewed separately from 3 and 4,
which is where §3.4's contested decision lives.

---

## 6. What this plan does not claim

- The §2 measurement is one release of four `.obo` artifacts on one day. It is
  not a guarantee about other releases, the `.owl` products, or operator-supplied
  files.
- Recording a digest and a `data-version` says *which file*; it does not say the
  file was the authentic artifact. Phase 1 deliberately records rather than
  refuses, and Phase 2 does not change that.
- The imports policy makes a build self-contained **with respect to declared
  imports**. It says nothing about what a file's own content contains.
- Nothing here touches `DISEASE_SCORER_POLICY.md`, the shortest-path artifacts,
  or any checkpoint.
