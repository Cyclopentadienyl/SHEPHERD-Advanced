# PLAN — ontology provenance and selection

**Status: draft for review.** No source file is edited by this plan. It exists to
settle the phasing before any of it is built, the same gate `PLAN_B04.md` §13
applies to production code.

Covers two shipped documentation commits (`b2ecd0f`, `374d22c`) and the design
discussion they provoked.

---

## 1. What the two commits did

| | |
|---|---|
| `b2ecd0f` | Added §4.3 to `deployment-guide.md` — the shortest-path sidecar, the hop-bound recovery order, the process-level nature of `SHEPHERD_SP_HOP_BOUND`, and the status fields to check. Recorded the maintainer's workspace export/import idea in the backlog's parked list |
| `374d22c` | Added `deployment-guide.en.md` (English edition, indexed beside the original) and recorded the ontology-provenance finding in the backlog |

Neither touches `src/`. The English edition summarises the long PowerShell and
bash listings as the flow they perform rather than transcribing them: the guide
itself states those are v2.0 content kept for understanding the mechanics, with
`deploy.{sh,cmd}` as the real entry point, so a second verbatim copy would be
two things to keep in step for no reader. The shortest-path section is carried
in full, because it is operational.

---

## 2. What was verified, and what it means

All of the following was read from the code at `374d22c`, not recalled.

### 2.1 The ontologies are invisible inputs; the annotations are not

Three inputs feed a knowledge-graph build, and they are managed in two different
ways:

| Input | Source | Managed by |
|---|---|---|
| MONDO ontology `mondo.obo` | auto-downloaded to a home-directory cache | **nobody; invisible** |
| HPO **ontology** `hpo.obo` | the same mechanism | **nobody; invisible** |
| HPO **annotations** `phenotype.hpoa`, `genes_to_phenotype.txt` | `--external-dir` | **the operator, explicitly** |

So this is not a MONDO problem. The HPO *ontology* has it too, and the
distinction from the HPO *annotations* — which are already explicit — is the
asymmetry worth removing.

- Cache: `Path.home()/'.shepherd'/'ontologies'` (`loader.py:63`), **outside the
  project**.
- On a missing file: `urlretrieve('http://purl.obolibrary.org/obo/mondo.obo')` —
  **whatever is current**, with no pinning (`loader.py:138-140`).

### 2.2 The `version` argument reads as pinning and is not

`_load_known_ontology` uses `version` only as an in-memory cache key
(`cache_key = f"{ontology_name}_{version}"`) while the file it opens is always
`self.cache_dir / f"{ontology_name}.obo"`. So `load_mondo(version="2026-01-01")`
loads whatever `mondo.obo` happens to be.

**An API that looks like it pins a version and does not is worse than one that
does not offer the argument.** This is the one item that should not wait on any
design decision.

**`force_download` is a different case, and the first draft of this plan got it
wrong.** It was listed here as a second inert control. It is not: the parameter
is checked at all three branches of `_load_known_ontology` — the memory cache
(`:124`), the disk cache (`:130-135`) and the download decision (`:138`) — so
passing `True` bypasses both caches and re-fetches. What is true is that **no
caller anywhere passes it and no CLI exposes it**; that is a usability gap, not
a broken control, and the two claims are not the same. Removing it would delete
working behaviour.

The accurate statement about refreshing: there is no *command* for it, so an
operator's only route today is deleting the cached file by hand — while the
mechanism to do it properly already exists one layer down.

### 2.3 The version is knowable and is not recorded

The OBO header carries `data-version`, the loader parses the tag
(`loader.py:329`) and `hierarchy.py:127` already reads it. Nothing writes it
anywhere. The split manifest carries a recipe for `node_features.pt` and digests
for the tensors, and says nothing about which ontology produced `kg.json`.

This is the schema-3 principle applied to a different artifact: a digest says
*which bytes*, only a recipe says *how to make them again*. `kg.json` has the
first and not the second.

**Not hypothetical.** The two machines used this session differ in MONDO vintage
— 29,866 vs 32,109 disease nodes, different `kg_digest` — from deployment dates
alone. The differing digest *detects* the difference, which is the manifest
working; it cannot say *what* differed, and no site can deliberately reproduce
another's graph.

### 2.4 A silent-in-the-artifact data loss follows from it

`HPOAnnotationParser` maps each annotation's OMIM/ORPHA identifier onto MONDO.
An identifier absent from the loaded MONDO resolves to `None` and the annotation
is dropped (`hpo_annotations.py:186-188`). The count is kept as
`skipped_unmapped` and **logged** (`:207`) — but it reaches no artifact.

So updating `phenotype.hpoa` while the ontology stays at its install-day vintage
silently drops the diseases the new annotations added, and the workspace records
no trace of how many. **This is the concrete harm that makes the version
question load-bearing rather than tidiness.**

### 2.5 One control already exists, undocumented

`scripts/build_knowledge_graph.py` has `--ontology-cache-dir` (default
`~/.shepherd/ontologies/`). Separate cache directories per build already work
today — which is the seed of multi-version coexistence. Nothing documents it and
nothing records which directory a given build used.

---

## 3. The design question, and the constraint that shapes it

The maintainer's target is a ComfyUI-style experience: ontologies as manageable
resources with a default location, a user-specified location, selection of a
particular file, and possibly a URL to fetch from.

**The constraint that changes the shape of that.** ComfyUI's models and
extensions are interchangeable at runtime. Ontologies are not. Node insertion
order *is* index assignment — the contract `src/ontology/hierarchy.py` was made
to hold earlier on this branch — so `node_features.pt`, `edge_indices.pt`, the
cohort split and every checkpoint are bound to the integers a particular
ontology produced. **Swapping an ontology does not change a setting; it
invalidates the workspace.**

Therefore: **choosing an ontology is part of building a workspace, not part of
configuring a service.** A surface that offers it as a switchable setting would
promise something the data model cannot deliver. A library view that acquires
and inspects files is fine; *selection* belongs beside `--external-dir` on the
build path.

### 3.1 What a ComfyUI-shaped backend actually needs

ComfyUI works not because it has a registry but because **the filesystem is the
registry**. The same applies here, and it is small:

1. **A resolver.** Given roots (a default plus any the operator names),
   enumerate the ontology files present with each one's identity —
   `data-version`, digest, size, term count. A pure function over the
   filesystem: no database, no install step, no state.
2. **Explicit selection.** A path argument per ontology, aligned with
   `--external-dir`, which the annotations already use. A front end's "pick a
   version" is then literally "which path do I pass". Multi-version coexistence
   falls out of pointing at different files; no registry is required.
3. **Recorded provenance.** The manifest records what was used, which is what
   lets any surface later say "this workspace was built with MONDO 2026-06-11".

The cache becomes **one root with a default location** rather than a special
mechanism. `configs/deployment.yaml` already has a `paths:` block
(`workspaces_root`, `cache_root`, …), so `ontology_roots` belongs there rather
than in a new configuration file.

### 3.2 Keyed by ontology name from the start

`ONTOLOGY_URLS` already declares four ontologies (`hpo`, `mondo`, `go`, `mp`);
only `mondo` and `hpo` are loaded by any caller. Since more are expected, the
resolver and the provenance record should be **keyed by ontology name** rather
than carrying two hard-coded fields. The cost is near zero now and it avoids a
schema change per ontology later.

### 3.3 On fetching by URL — a narrower recommendation

The maintainer doubted this feature. The doubt is right; the reason is worth
stating precisely, because it is not the one assumed.

- **URL shape is not the obstacle.** All four entries in `ONTOLOGY_URLS` are OBO
  Foundry PURLs of one form (`http://purl.obolibrary.org/obo/<name>.obo`), and
  they are stable permanent redirects. A curated list is easy.
- **Free-text URLs are the obstacle.** This system is destined for a hospital.
  A user-supplied URL would be fetched by `urlretrieve` and handed to pronto to
  parse. Arbitrary fetch-and-parse is a security surface in that setting, not an
  inconvenience.

**Recommendation: a curated source list, never a free-text URL field.** Fetch a
known ontology from its PURL, then verify and record the `data-version` and
digest of what arrived.

### 3.4 The inversion worth making

Today auto-download is the primary path, and **the build CLI has no
per-ontology path argument** — a second correction to the first draft, which
said there was no way to point at a file at all. `OntologyLoader.load(path)`
exists (`loader.py:68`) and takes a file directly; what is missing is a route to
it from `scripts/build_knowledge_graph.py`, which only exposes
`--ontology-cache-dir`. The gap is in the entry point, not the library.

That ordering is still backwards for the target environment: a hospital network
may not reach the internet, and the offline path is the one that always works.

**"Point at a file you already have" should be the first-class path, and
download a convenience that may be absent.** This also meets the maintainer's
export/import idea from the other side — import a packaged workspace and its
ontologies arrive with it.

### 3.5 Where the record is written — the question Phase 1 must settle first

The first draft said "record it in the manifest" and called Phase 1 free of
design decisions. Two facts make that wrong.

**Graph-only builds write no manifest.** `write_workspace(samples=None)` is a
supported mode: it writes `kg.json` and the tensors and returns
(`workspace.py:251`) before any split manifest exists. Provenance placed only in
the split manifest would leave that path unrecorded — and it is the path whose
output most needs identifying, because a graph is what the ontologies produced.

**`kg.json` has nowhere to put it.** Its serialisation is `format_version`,
`nodes`, `edges` (`graph.py:799`). Adding a section would change every existing
`kg.json`'s bytes and therefore every recorded digest, invalidating manifests
that are otherwise sound.

**Proposal: a `kg.provenance.json` written beside `kg.json` on every build, and
bound by digest into the split manifest's `artifacts` map** exactly like the
four graph artifacts already are.

The obvious objection is that this project has just spent a long time on what
goes wrong when a sidecar can be separated from the file it describes — the
shortest-path `meta.json` is the whole reason for the hop-bound work. Binding it
by digest is the answer to precisely that: an unbound sidecar is separable and
undetectable, a digest-bound one is separable and *detected*, by machinery that
already exists and is already tested. For a graph-only build there is no
manifest to bind it into, and the honest statement is that such a workspace is
self-describing but not self-verifying — which is already true of everything
else in it.

An alternative — carrying the record inline in the split manifest as well — is
not proposed, because two copies that can disagree is a worse failure than one
copy that can go missing.

**Absent records read as unknown.** A workspace built before this exists has no
provenance, and nothing may fill that in from whatever is in the cache now: the
current file is not evidence about a past build.

### 3.6 All four inputs, or a narrower claim

Phase 1 as first drafted recorded the two ontologies and claimed it would
explain the two-site divergence. It would not. `--external-dir` names a
directory, and `phenotype.hpoa` and `genes_to_phenotype.txt` can both be
replaced in place under the same names. Same ontologies, same directory path,
even the same skipped count, and the graph can still differ because the
annotations did.

**So Phase 1 records all four source files** — two ontologies and two annotation
files — each with its role, its content digest, and its declared version where
one exists. The directory is a locator, not an identity.

This does not bring annotation selection or download management into scope. It
brings the annotation files into the *record*, which is where the reproducibility
claim actually lives.

### 3.7 Naming the numbers precisely

`skipped_unmapped` is a local counter in one parser
(`hpo_annotations.py:157-207`). It counts **annotation rows** that
`phenotype.hpoa` parsing could not resolve to a MONDO identifier, which includes
identifier types that are deliberately unsupported — DECIPHER among them — and
not only vintage mismatches. The builder drops further edges elsewhere when a
phenotype or disease node is absent (`builder.py:393`), and those are not in
this count.

**Record it as what it is**: rows skipped at one parsing stage, with the stage
named. It must not be presented as "diseases lost to a version mismatch" —
the first draft of this plan came close to doing exactly that.

Similarly, `Ontology.version` falls back to `format_version` and then to
`"Unknown"` (`hierarchy.py:124-128`), so it can return a *format* version where
a release is expected. **Provenance stores the raw `data-version`, null when the
file declares none.** The content digest is the identity; the declared version
is a label.

---

## 4. Proposed phasing

Ordered by what each phase needs from the one before, and by which decisions
each one forces.

| Phase | What | Needs a design decision? |
|---|---|---|
| **0** | Resolve the **`version`** promise — make it select a file or refuse a version it cannot honour. **`force_download` is left alone**; it works | **No.** Correcting a misleading API |
| **1** | `kg.provenance.json` on every build (§3.5), covering **all four source files** (§3.6), with the parsing counters named precisely (§3.7) | **Yes, one**: where the record is written. §3.5 proposes an answer |
| **2** | Explicit selection: a path per ontology on the build CLI, a resolver over configured roots, and a stated **imports policy** (§4.2) | Small: where roots are configured |
| **3** | Packaging — ontologies travel with a workspace; converges with export/import | **Yes**, and it should come last |

**Why 3 comes last.** Once 1 and 2 are done, packaging is moving things that are
already identified. Done first, it produces a bundle that cannot say what is
inside it.

**Why 0 does not wait.** It is the only item that is actively misleading rather
than merely missing, and it depends on nothing.

**Phase 1 is where the value is** — and only with §3.6's full four-file scope.
With the two ontologies alone it would leave a divergence caused by an updated
annotation file just as unexplainable as before, while sounding as though the
question were settled. It needs no front end and benefits CLI users immediately.

### 4.2 The imports policy Phase 2 owes

Pointing at a local file does not by itself make a build offline or
self-contained. The loader calls `pronto.Ontology(str(path))` with nothing else
(`loader.py:84`), and pronto 2.7.3's `import_depth` defaults to `-1` — resolve
every import, without bound. A root file carrying `import:` lines can therefore
pull further ontologies over the network at parse time, and the root file's
digest does not cover what they contributed.

**Verified: the default is unbounded and the loader passes no override.
Unverified: whether the MONDO and HPO artifacts these deployments use actually
declare imports** — no such file exists in the environment this plan was written
in, so the exposure is conditional and stated as conditional.

Phase 2 must therefore state a policy rather than inherit one. The narrow option
is to require self-contained files and refuse an input whose imports cannot be
satisfied locally; the wider one is to support imports and record each resolved
dependency alongside the root. **Silently suppressing imports is not an option**
— it would change the graph while appearing to make the build offline.

### 4.1 Explicitly not proposed

- No ontology registry, no version-negotiation layer, no install-state database.
  A path is enough.
- **No removal of `force_download`**, which works. Whether to expose a refresh
  command is a separate, later question.
- **No refusal on version mismatch, at least not in phase 1.** Make it visible
  and comparable first; whether a mismatch should block is a decision to take
  with evidence in hand, not before.
- No front end. UI is outside this programme's scope. Every phase above is
  backend work that stands on its own, and a front end added later consumes the
  resolver and passes paths.
- No change to `scripts/compute_shortest_paths.py` or the shortest-path sidecar;
  that is a separate recorded defect.

---

## 5. Where this sits against 5a

Nothing here blocks or is blocked by backlog item 5a. 5a's step 0 is complete
and approved; step 2 (moving approach A into `src/inference/sp_index.py`) is
next and touches none of this.

**Recommendation on order**, unchanged by this revision and endorsed by review.
Phase 0 is small, independent and corrects something misleading, so it can land
between 5a's steps without entangling them.
Phases 1-3 should wait until 5a reaches its stopping point — implemented, with
the integrated gate readings pending a designated measurement subject — so that
two multi-commit efforts are not open in the same files at once. 5a touches
`src/inference/`; phases 1-2 touch `src/ontology/`, `src/kg/` and the manifest
schema, so the overlap is small but the review attention is not.
