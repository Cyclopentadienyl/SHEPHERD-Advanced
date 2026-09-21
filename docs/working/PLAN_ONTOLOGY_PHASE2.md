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

### 3.1 A path per ontology, and what happens when more than one file fits

`--mondo-path` and `--hpo-path`, aligned with the existing `--external-dir` that
the annotations already use, and **keyed by ontology name** per the parent plan's
§3.2 so that `go` and `mp` need a table entry rather than a schema change.

**The first draft of this table said "a resolver hit" as though there could only
be one, which contradicted §3.2 two paragraphs later.** §3.2 enumerates *every*
file under *every* root with its version and digest — so two roots each holding a
MONDO, or one root holding two releases, is the case the parent plan's §3.1 says
multi-version coexistence is *for*, not an aberration. Recording which digest was
finally used does not decide who should have been used.

| Given | Taken |
|---|---|
| An explicit path | That file. A path that does not exist, or is not readable, **refuses** — it never falls through to a root or a cache |
| No path, **exactly one** valid candidate across all roots | That file, with its root and digest logged |
| No path, **more than one** candidate | **Refuse**, listing each candidate's path, `data-version` and digest, and asking for an explicit path |
| No path, no candidate, a source configured | Download (§3.5) |
| No path, no candidate, no source | Refuse, naming the roots searched and the manual instructions already in `ONTOLOGY_MANUAL_INSTRUCTIONS` |

**No implicit precedence.** Not root order, not modification time, and above all
not "the newest `data-version`" — a rule that picks silently is a rule nobody
reads, and ordering by a self-declared version string is a guess wearing a
comparison. Ambiguity is an operator's decision and the refusal hands them the
three facts they need to take it.

Two existing surfaces have to be placed rather than left to interact by accident:

- **`--ontology-cache-dir` becomes a root**, the last one searched, so an
  existing invocation keeps working and its directory is now enumerated rather
  than opened by convention. It is not special beyond its position.
- **`force_download` with an explicit path is a contradiction and refuses.**
  Naming a file and demanding a fresh download are two different instructions.
  With no explicit path it keeps its current meaning — bypass what is present
  and fetch. **What it must not become is a flag that survives in the signature
  while the resolver returns before anything reads it**; that failure has its own
  acceptance case.

### 3.2 The resolver

A pure function over the filesystem: given roots, enumerate the ontology files
present with each one's identity — `data-version`, digest, size, term count.
No database, no install step, no state, and **no directory creation** (§1.3).

**"Pure over the filesystem" has to hold for the term count too.** A count means
a parse, and a parse at pronto's default resolves imports over the network — so
enumeration would reach the internet before anyone has chosen a file. Listing
uses the same `import_depth=0` rule as §3.3, and a file that declares an import
is listed with its declaration visible rather than parsed further.

Roots come from `configs/deployment.yaml`'s `paths:` block (line 148), which
already holds `workspaces_root`, `cache_root` and friends — the parent plan's
§3.1 placed `ontology_roots` there and nothing here changes that.

**Where that configuration is read has to be named, because there is nowhere
obvious.** `src/config/config_validator.py` is an intentionally empty reserved
module — its docstring says so, and nothing on any runtime path imports it. So
Phase 2 adds one small reader with a stated contract: which file it reads, and
what relative paths in it are relative to. It does not revive a global
configuration manager, and it does not fill in the reserved validator.

### 3.3 The imports policy: refuse every declared import

**An input that declares any import is refused, by name.** Including imports
that point at a local file: resolving those is a dependency story with its own
digests and its own provenance schema, and nothing measured says an artifact in
use needs it.

> The first draft said "imports that cannot be satisfied locally", which is a
> different and weaker rule than the three steps below implement. The steps are
> the policy; that sentence was wrong.

Implemented as measured in §2.1, and the order matters:

1. Parse with `import_depth=0`, so **nothing is fetched at parse time**.
2. Read `ontology.metadata.imports`.
3. If it is non-empty, **refuse**, naming each declared import and saying that a
   self-contained file is required.

Step 3 is the whole policy. Step 1 alone is silent suppression, which the parent
plan's §4.2 rules out and §2.1 measured. The wider option — support imports and
record each resolved dependency beside the root — stays available and is not
proposed now.

**This is a refusal a real artifact might one day hit.** If a future MONDO
release declares an import, builds stop with a message naming it, and that is the
intended behaviour: the alternative is a graph silently missing what the import
carried.

### 3.4 The file has to be the ontology the slot asked for

**Recorded in §1.3 and then not designed against, which is the gap that matters
most here.** Reproduced on this tree at `407d46d`:

```
hpo.obo  — declares `ontology: mondo`, contains two MONDO terms
  OntologyLoader.load()                  -> LOADED, name='mondo', 2 terms
  builder.add_ontology(ont, PHENOTYPE)   -> 0 nodes, no exception
```

The build continues. `add_ontology` selects terms by the prefix belonging to the
node type, finds none, and returns zero; the provenance entry records role `hpo`
because **the caller passed that role**, together with a faithful digest and a
faithful `data-version`. The result is a workspace with a complete-looking
record and no phenotype nodes.

**So a role check runs before anything is written**, and it refuses when the
declared ontology contradicts the slot, or when the slot's namespace yields no
usable term. The refusal names the file, what it declares and what was expected.

Two things it must not do. It must not require every term to carry one prefix —
a legitimate ontology cross-references other namespaces. And it must not fall
back to a cached file when an explicitly named one fails the check; an explicit
path that is wrong is an error to report, not a reason to open something else.
`hp` and `hpo` are the same ontology under two spellings and the check has to
know that.

This asks for no version negotiation and refuses no old release. It answers
"is this the right *kind* of file", which is a different question from "is this
the right version" and from "is this authentic".

### 3.5 Where the scheme rule is enforced: the request, not the setting

The config-read allowlist in the first draft is necessary and **not sufficient**.
Measured on this machine's CPython 3.13.9:

```
HTTPRedirectHandler.http_error_302 permits redirect targets whose scheme is in
    ('http', 'https', 'ftp', '')
```

So an `https://` source that 302s to `ftp://` is followed, and a check performed
only when the list was read has already passed. This is not hypothetical for
this project specifically: **every configured source is a PURL, and a PURL is a
redirect.**

(An earlier reading of mine said `file://` was reachable the same way. It is
not — that came from calling `redirect_request` directly and bypassing the gate
above. The `ftp` hole is real; the `file` one was my error.)

**The rule belongs at the request boundary, in one downloader that every fetch
goes through:** the initial URL and *each* redirect target must satisfy the
scheme allowlist, and the OBO and OWL attempts use that same downloader and the
same configured sources. A fallback path that bypasses the check is the check
not existing. Tested with controlled responses; no request to any real host is
needed to cover it.

**`http`/`https` does not mean "external".** It admits `localhost`, link-local
addresses and anything on the hospital's own network. An in-house mirror is a
legitimate and probably desirable source, so the destination rule is a
**deployment setting** — an administrator states what this server may fetch from
— rather than either a blanket block or a blanket trust. Phase 2 provides the
enforcement point and a default; it does not decide a site's network policy.

### 3.6 Revising the parent plan's §3.3 — the source list becomes editable

**This changes an approved decision and is flagged rather than folded in.**

§3.3 reads: *"Recommendation: a curated source list, never a free-text URL
field"*, on the grounds that a user-supplied URL is fetched by `urlretrieve` and
handed to pronto, and arbitrary fetch-and-parse is a security surface in a
hospital.

The maintainer objected, and **two of the arguments behind §3.3 do not survive**:

- **"A wrong ontology silently corrupts the knowledge base" does not
  discriminate between the two designs.** A hand-placed wrong file has exactly
  the same effect and exactly the same record. It is a real risk of ontology
  selection in general — §3.4 is what actually addresses it — and it is not an
  argument about where a URL comes from.
- **"Fetching unknown data is a risk" is largely the network's job.** A
  deployment that lets this server reach the internet has accepted that class
  already. What does *not* follow from "the hospital has a firewall" is that the
  application may ignore where its requests go: a firewall's egress rules do not
  govern a server's requests to its own network, and §3.5 is where that is
  handled.

What survives argues for **constraining** the source list, not refusing to let
anyone edit it: the scheme rule of §3.5, the imports policy of §3.3, and the
question of who may edit.

**So the curated list moves out of `loader.py:45` into configuration.** A PURL
that rots becomes an edit rather than a code change.

#### 3.6.1 The requirement is a UI slot, and Phase 2 does not deliver it

**Stated plainly because the first draft overstated what it delivered.** It
called operator-editable configuration "the maintainer's actual requirement".
That is wrong: the requirement was *"把 URL 變成 UI 上顯現可修改的插槽"* — visible
and editable in the interface, the way a ComfyUI user installs from a URL.
Configuration editability is a **part** of that, not the whole of it.

So the contract is recorded as owed, not as forbidden:

| | |
|---|---|
| **The requirement** | An operator with data-management rights can see and edit the source URLs **in the interface** |
| **Phase 2 delivers** | One configuration model and one download-and-verify entry point, which a UI and the CLI both use |
| **Phase 2 does not deliver** | The interface itself. The parent plan's §4.1 puts the front end outside this programme |
| **Not the same thing** | Every diagnostic user controlling what the server downloads. The surface is for data management, and §3.5 governs the destinations |

**A phase boundary is not a prohibition**, and the parent plan's §3.3 must not be
read as one once this section is accepted. What stands from it is narrower: not a
free-text field exposed to every user of the application.

**Changing a source is not switching a graph.** Editing a URL, downloading a
file, and building a workspace from it are three steps, and only the third
changes what anything is served from. A running pipeline's graph is bound by the
manifest and this design must not give the settings surface a route to swap it;
a new ontology means a new build, which means the existing verification applies
unchanged.

### 3.7 What this does not build

- **No registry, no version negotiation, no install-state database.** A path is
  enough.
- **No removal of `force_download`** (§3.1 places it).
- **No refusal on version mismatch.** Still the parent plan's §4.1 call: make it
  visible and comparable first.
- **No packaging.** That is Phase 3 and it comes last.
- **No dependency resolver for imports** (§3.3).
- **No change to the OWL path's behaviour** beyond applying §3.3, §3.4 and §3.5
  to it equally. §1.4's measurement gap is recorded, not closed.

---

## 4. Acceptance

Named before the work rather than argued after it. Ordinary unit tests;
controlled responses where a fetch is involved, and no request to any real host.

**Selection**

1. An explicit path is used, and one that does not exist **refuses** rather than
   opening `<cache_dir>/<name>.obo`.
2. **Two candidates across two roots refuse**, and the message carries each one's
   path, `data-version` and digest.
3. **Two releases in one root refuse** the same way.
4. **Exactly one candidate is used**, so the refusals above are not a resolver
   that rejects everything.
5. **`--ontology-cache-dir` still works**, now as the last root.
6. **`force_download` with an explicit path refuses**; without one it still
   bypasses what is present. A mutant that leaves the flag in the signature while
   the resolver returns first must fail a test.

**The resolver**

7. Identity, not presence: `data-version`, digest, size and term count for each
   file found.
8. **Constructing a resolver creates no directories** (§1.3's `mkdir`).
9. **Enumeration issues no network request**, including for the term count, and
   including when a listed file declares an import.

**Imports**

10. A file declaring an import is **refused, naming it** — it must not load with
    the import dropped.
11. A file declaring an import that points at a **local** file is refused too
    (§3.3 as written, not as the first draft said).
12. A self-contained file loads with nothing fetched.

**Role**

13. A file declaring `ontology: mondo` in the HPO slot **refuses**, naming the
    file, what it declares and what was expected. The reproduction in §3.4 is the
    fixture.
14. A legitimate ontology carrying cross-references to other namespaces **still
    loads** — the check is not "every term has one prefix".
15. `hp` and `hpo` are accepted as the same ontology.
16. An explicit path failing the role check does **not** fall back to a cache.

**Downloads**

17. A source whose scheme is not allowed is refused **when the list is read**,
    naming the entry.
18. A permitted initial URL that **redirects to `ftp://` is refused at the
    redirect**, which is the measured gap the config-time check misses.
19. An ordinary `https` → `https` PURL redirect is still followed.
20. **The OWL fallback goes through the same downloader** and is refused by the
    same rules; a test must fail if it acquires its own fetch path.

**Provenance**

21. Unchanged in shape, and now describing a selected input: a build given
    explicit paths records the same four roles with the same digests and declared
    versions Phase 1 records today.

---

## 5. Sequence

Ordered so that the contested section (§3.6) is last and the rest can proceed
whatever a reviewer decides about it.

1. The resolver and its tests, as a pure function over roots — including the
   no-network enumeration of §3.2. Nothing calls it yet.
2. The imports policy in `OntologyLoader.load` (§3.3) and the role check (§3.4),
   with their refusals. These two change existing behaviour and nothing else here
   does.
3. The configuration reader, the source list, and the shared downloader with the
   scheme rule at the request boundary (§3.5).
4. `--mondo-path` / `--hpo-path` on the build CLI, wired to the resolver, with
   §3.1's precedence table.
5. Acceptance matrix, documentation, and the operator-facing lines in the
   deployment guides — including that the UI slot of §3.6.1 is **owed and not
   delivered**.

---

## 6. What this plan does not claim

- The §2 measurement is one release of four `.obo` artifacts on one day. It is
  not a guarantee about other releases, the `.owl` products, or operator-supplied
  files.
- **A digest is not a signature.** What exists is SHA-256 over bytes, artifact
  binding, and a recorded `data-version` that the file declares about itself.
  That identifies which bytes were used and whether a record belongs to a graph.
  It does not establish that a file came from its publisher, that its content is
  correct, or that it suits the role it was given — §3.4 addresses only the last
  of those, and only for gross mismatch.
- The imports policy makes a build self-contained **with respect to declared
  imports**. It says nothing about what a file's own content contains.
- §3.5 provides an enforcement point and a default. It does not decide a site's
  network policy, and `http`/`https` alone does not make a destination safe.
- Nothing here touches `DISEASE_SCORER_POLICY.md`, the shortest-path artifacts,
  or any checkpoint.
