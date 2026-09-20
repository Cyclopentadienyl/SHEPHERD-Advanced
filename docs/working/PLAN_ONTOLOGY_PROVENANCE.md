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

`force_download` is a second inert control: it exists on the loader and **no
caller anywhere passes it**, and no CLI exposes it. Refreshing an ontology today
means deleting the cached file by hand.

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

Today, auto-download is the primary path and **there is no way to point at a
file at all**. That is backwards for the target environment: a hospital network
may not reach the internet, and the offline path is the one that always works.

**"Point at a file you already have" should be the first-class path, and
download a convenience that may be absent.** This also meets the maintainer's
export/import idea from the other side — import a packaged workspace and its
ontologies arrive with it.

---

## 4. Proposed phasing

Ordered by what each phase needs from the one before, and by which decisions
each one forces.

| Phase | What | Needs a design decision? |
|---|---|---|
| **0** | Resolve the inert `version` and `force_download` arguments — make them work or remove them | **No.** Correcting a misleading API |
| **1** | Record provenance: per-ontology `data-version`, digest and source in the manifest, plus `skipped_unmapped` | **No.** The values are already computed |
| **2** | Explicit selection: a path per ontology, resolver over configured roots | Small: where roots are configured |
| **3** | Packaging — ontologies travel with a workspace; converges with export/import | **Yes**, and it should come last |

**Why 3 comes last.** Once 1 and 2 are done, packaging is moving things that are
already identified. Done first, it produces a bundle that cannot say what is
inside it.

**Why 0 does not wait.** It is the only item that is actively misleading rather
than merely missing, and it depends on nothing.

**Phase 1 is where the value is.** It makes the two-site divergence explainable
instead of merely detectable, and it surfaces the annotation drop that no
artifact currently records. It needs no front end and benefits CLI users
immediately.

### 4.1 Explicitly not proposed

- No ontology registry, no version-negotiation layer, no install-state database.
  A path is enough.
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

**Recommendation on order.** Phase 0 is small, independent and correcting
something misleading, so it can land between 5a's steps without entangling them.
Phases 1-3 should wait until 5a reaches its stopping point — implemented, with
the integrated gate readings pending a designated measurement subject — so that
two multi-commit efforts are not open in the same files at once. 5a touches
`src/inference/`; phases 1-2 touch `src/ontology/`, `src/kg/` and the manifest
schema, so the overlap is small but the review attention is not.
