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

#### 3.1.1 What that fixes, and what it leaves free

**This is a default about ambiguity, not a hard-coded choice**, and the
difference is worth stating because "written into the rule" reads like "you are
stuck with one vendor, version, URL or directory". Nothing of the sort is fixed:

| | |
|---|---|
| **Fixed** | Only this: with no basis for choosing, do not choose |
| **Not fixed** | Which ontology, which release, which URL, which directory, how many coexist |
| **The cost** | Multi-candidate builds are not fully hands-off until a path is named |
| **Not the cost** | Automation. Naming a path automates completely, with no file deleted and no code changed |

**One selection entry point, shared.** The resolver lists candidates; a single
caller turns a list into a choice. The CLI passes a path to it and a later UI
shows the candidates and passes the chosen one — so ambiguity must never make
the listing surface unusable, which is the outcome a refusal buried inside the
loader would produce.

**If a persisted default is wanted later**, the smallest form that stays inside
"do not guess" is storing the path an operator already chose and replaying it as
an explicit choice through that same entry point. **Not delivered by Phase 2**,
recorded so it is not reinvented as a `first` / `latest` / strategy framework.

#### 3.1.2 `--ontology-cache-dir`: a deliberate behaviour change, not preserved behaviour

**The first draft claimed an existing invocation keeps working. That is not
true and the claim is withdrawn.** `_load_known_ontology` prefers `<name>.obo`
over `<name>.owl` in the same directory today. Under §3.1 both are valid
candidates, so a cache holding both — which is what the OBO→OWL download
fallback of §1.4 produces — **refuses** where it used to load. A second case:
the named cache holds one file and a configured root holds another.

Placing the cache last does not rescue either, precisely because the policy does
not select by root order. So:

- The flag is still accepted and still works **when there is exactly one
  candidate**, which is the common case.
- A deployment that was relying on implicit OBO-over-OWL precedence **needs to
  add `--mondo-path` / `--hpo-path`**, and the refusal message says so.
- The refusal lists each candidate's `data-version`, so when the two are the
  same release in two encodings the operator sees that immediately and the
  decision is trivial.

**Implicit OBO precedence was considered and is not reinstated.** It looks like
a format rule rather than a version rule, but `.obo` and `.owl` in one directory
are not guaranteed to be the same release — the fallback writes whichever
succeeded, whenever it ran — and §1.4 records that the OWL parse path is not the
one measured. Restoring it would be a stated rule, not a quiet one; this section
is where a reviewer who wants it should say so.

**`force_download` with an explicit path is a contradiction and refuses.**
Naming a file and demanding a fresh download are two different instructions.
With no explicit path it keeps its current meaning — bypass what is present and
fetch. **What it must not become is a flag that survives in the signature while
the resolver returns before anything reads it**; that failure has its own
acceptance case.

### 3.2 The resolver

A pure function over the filesystem: given roots, enumerate the ontology files
present with each one's identity — `data-version`, digest, size, term count.
No database, no install step, no state, and **no directory creation** (§1.3).

**"Pure over the filesystem" has to hold for the term count too.** A count means
a parse, and a parse at pronto's default resolves imports over the network — so
enumeration would reach the internet before anyone has chosen a file.

> **As implemented, listing does not call pronto at all**, which is stronger than
> this paragraph first said ("the same `import_depth=0` rule as §3.3"). An OBO
> file is read line by line for its header tags and `[Term]` stanzas; an OWL file
> is fed in 64 KiB blocks to an XML parser whose target keeps counts and never
> builds a tree, so memory stays flat whatever the file's size or line layout.
> The format is decided by the **content**, not the suffix — pronto sniffs the
> same way, so a `.owl` holding OBO text is listed as what pronto will parse.
> Declared imports are recorded wherever the document states them and shown in
> the listing and in an ambiguity refusal, so the operator sees that a
> candidate will be refused at load before choosing it.

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
plan's §4.2 rules out and §2.1 measured.

> **Step 2 is not enough on its own for OWL, found in the pre-PR review and
> measured.** pronto records only an `owl:imports` that sits directly under the
> first `owl:Ontology` element. One stated on an `rdf:Description` of the
> ontology IRI, or written as a nested `<owl:imports><owl:Ontology
> rdf:about="…"/></owl:imports>`, leaves `metadata.imports` empty and the file
> loads with the import dropped — the §2.1 defect by another route. So the
> refusal is on the **union** of what pronto records and what the §3.2 scanner
> finds, and an import element with no target is refused as one. The listing and
> the loader now agree on what a file imports.
>
> **A declaration with no readable target is still a declaration**, found in the
> review of that fix. pronto records `None` only for one directly under the first
> `owl:Ontology`, and the scanner recorded an import only when it could read a
> URL, so `<owl:imports/>`, `rdf:resource=""` or a blank-node `rdf:nodeID` inside
> an `rdf:Description` left no trace in either and the file loaded as
> self-contained. The scanner now records every declaration, as
> `(an import with no target)` when it names nothing, wherever it is written;
> an OBO `import:` with no value is listed the same way (fastobo refuses to
> parse it, so it never loads). The wider option — support imports and
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

> **So are a publisher's release products**, found in the pre-PR review. Measured
> from the headers of the 2026-09-01 releases (range requests, as in §2):
> `mondo-simple.obo` declares `ontology: mondo/mondo-simple`, `mondo-base.obo`
> `mondo/mondo-base`, `hp-base.obo` `hp/hp-base`, `hp-international.obo`
> `hp/hp-international`, and `hp-simple-non-classified.obo`
> `hp/hp-simple-non-classified`; none declares an import. The first
> implementation refused an explicitly named `mondo-simple.obo` in the MONDO
> slot as a role mismatch and silently dropped one from the resolver's listing.
> A fixed list of variant suffixes (`simple`, `base`, `basic`, `full`, `plus`,
> `edit`, `international`, `non-classified`) now reduces `<known
> ontology>-<variant>[-<variant>…]` to the ontology — stacked, because the last
> of those measured carries two — so such a file is a candidate and passes the
> check. `mondo-simple` in the HPO slot is still refused, and an unknown stem
> (`foo-base`) stays itself. An IRI's `#fragment` no longer changes the name
> either.

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
addresses and anything on the hospital's own network.

#### 3.5.1 The default, stated so it can be tested

**The first draft said "an enforcement point and a default" and never said what
the default was** — a sentence shaped like a contract that specifies nothing,
which is the failure this programme keeps removing. An implementer could ship
the scheme check alone and believe this section was delivered. So:

**Default: every address the destination resolves to must be globally routable.
A host on the operator's allow list is exempt.**

`ipaddress` decides it without a network call, and one predicate covers both
families and every case that matters — measured:

| Destination | `is_global` |
|---|---|
| `8.8.8.8`, `2606:4700::1111` | True → allowed |
| `127.0.0.1`, `::1` | False (loopback) |
| `10.0.0.5`, `192.168.1.10`, `172.16.0.1`, `fd00::1` | False (private) |
| `169.254.169.254` | False (link-local — the cloud metadata address) |
| `0.0.0.0` | False (unspecified) |

- **Applied to the initial URL and to every redirect target**, in the same
  downloader as the scheme rule of §3.5. One gate, one place.
- **The check is on the resolved addresses, not the hostname text.** A literal
  private IP in a URL has no hostname to match against a list, and a name that
  resolves into the hospital's network is inside it whatever it is called.
  Both are measured cases in §4.
- **An in-house mirror is a supported and probably desirable source**, reached
  by an administrator adding its host to the allow list. The default is not a
  blanket block on the hospital's network; it is "say so on purpose".

**The residual, bounded rather than implied away.** This is check-then-connect:
the resolution the check sees is not the one the socket uses, so a name whose
answer changes between them is not covered. Closing that means connecting to a
pinned address while preserving the `Host` header, which is a larger change than
Phase 2 and is **not claimed here**. What this default does cover is the
ordinary cases — a misconfigured URL, a redirect into the internal network, a
literal internal address — which is what a build tool fetching four known
artifacts is actually exposed to.

Phase 2 provides the enforcement point and this default. **It does not decide a
site's network policy**, and `http`/`https` plus `is_global` does not make a
destination trustworthy — only reachable-by-policy.

#### 3.5.2 Amendment: a name with no local answer is not a policy verdict

**This changes the approved default and is flagged rather than folded in.** It
was found in the pre-PR review, after §3.5.1 had been approved and implemented.

As approved, "every address the destination resolves to" had one unstated case:
**a name that does not resolve on this machine at all.** The implementation
refused it. On a network whose only route out is an HTTP proxy, which is common
in hospitals, external names often have no local DNS answer — resolving them is
the proxy's job — so that refusal meant **no download could ever succeed there**,
and it was reported as a policy refusal ("stop and fix the configuration") rather
than as what it was.

| A name that… | As approved and first implemented | Amended |
|---|---|---|
| resolves to global addresses only | allowed | allowed (unchanged) |
| resolves to any non-global address | refused, naming it | refused, naming it — **proxy or not** (unchanged) |
| is a literal address | judged as an address | judged as an address (unchanged) |
| is on `ontology.allowed_hosts` | allowed | allowed (unchanged) |
| **has no local answer, and the request goes to a configured proxy** | refused | **left to the proxy** |
| **has no local answer, no proxy** | refused as policy | **a transfer failure**, reported with the manual route |

"Goes to a configured proxy" means the standard `http_proxy` / `https_proxy`
environment settings, with `no_proxy` honoured: a host the proxy is bypassed for
gets no exemption, because that request would be resolved and connected here.

**"Goes to the proxy" is decided the way `urllib` decides it**, found in the
review of the amendment. `ProxyHandler` asks `proxy_bypass(req.host)` — the
request's host **with its port**, against `no_proxy` from the environment or the
operating system's settings — whatever proxy table it was built with. The first
implementation judged the bare host name, and for an explicit mapping read a
`"no"` entry the opener never consults, so with `no_proxy=host:8080` the rule
said "proxied" and the request to `http://host:8080/` went direct. The rule now
calls the same function on the same `Request`, and a test checks its answer
against the route the opener `download_ontology` builds actually takes.

**What this gives up, stated as a residual beside check-then-connect:** for a
proxied name with no local answer, the application does not judge the
destination; the proxy does. If the proxy resolves that name to an address inside
the hospital and is permitted to connect there, the request reaches it — that is
the proxy's egress policy, not something this check can see. What remains
covered, for the initial URL and every redirect target alike, is every name this
machine *can* resolve and every literal address, proxied or not. A reviewer who wants "refuse unless it resolves
here" restored should say so; the cost is that proxy-only deployments must
allow-list every source host.

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

**Changing a source is not switching a graph**, and the first draft's wording
was loose about the last step. Editing a URL, downloading a file, and building a
workspace are three things, and **none of them changes what is being served**:
the third produces a *new, verifiable workspace*. Putting that workspace into
service is a separate load-and-publish step that already exists and already
verifies. So a successful build must not be implemented as an automatic switch,
and the settings surface gets no route to swap a running pipeline's graph —
which is bound by its manifest, as it was before this plan.

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

### 3.8 The fetch path, as the pre-PR review left it

Found by a whole-branch adversarial review after the implementation was
approved. Each is the programme's recurring defect — a fallible step placed
after the state change it should have gated — and none changes a rule above;
they make the rules hold on the download path.

| Before | After |
|---|---|
| A download wrote straight to `<cache_dir>/<name>.<ext>`. **An `hpo.obo` that held MONDO was overwritten** by the HPO download, after the MONDO slot had already loaded it — and provenance then hashed the path, recording the new HPO bytes as the MONDO input | **The download is refused before anything is fetched** when the target path holds a file that is not a candidate for that ontology, naming it. The file is left alone. **The digest is of the bytes parsed:** `load` hashes, parses and scans imports through one open handle, and the `Ontology` carries that digest (`source_digest`) through publishing to provenance, which no longer re-reads the path. (The first version of this fix hashed the path right after each slot loaded and said nothing writes there in between; under two builds sharing a cache, one did.) An in-place write into the same file by another program while it is read is not covered; the loader itself only ever replaces files |
| `force_download` replaced the cached file, then parsed it; a fresh copy failing the role or imports check left the cache holding a file that would be refused next time | **Verify, then publish.** The fetch lands in a hidden staging file beside the target, is loaded with the slot's role and imports checks, and only then replaces the target (`os.replace`). A failure unlinks the staged file and the old one is untouched |
| The staging name was fixed per ontology, so two fetches into one cache at once shared it: one verified MONDO, the other's HPO landed under that name, and the first published it — found in the review of the row above | **Each fetch stages under a name of its own** (`mkstemp`, `.<name>.<ext>.<random>.staged`) and passes the final path explicitly; a failed attempt removes only its own file |
| A failed transfer fell back to whatever file sat at the cache path, with a log warning. A download is only attempted when selection found no candidate, so that file was one the resolver had **already passed over** | **No fallback.** A failed transfer is reported with the manual route, as a policy refusal already was. With nothing configured to fetch, the refusal names the roots searched and says nothing was attempted |
| The target's suffix was decided by `url.endswith(".owl")`, so an OWL source with a query string was written to `<name>.obo` | From the URL's path |
| A chunked response was judged against a `Content-Length` it does not carry; an interrupt mid-transfer left a `.part`; the published file was `0600` | `Transfer-Encoding` overrides `Content-Length` (RFC 9112 §6.1); any exit unlinks the partial file; the file is published with the process umask's mode |
| `build_knowledge_graph()` raised `SystemExit` for an ambiguity or a path-with-`--force-download` refusal, and let role, imports and fetch refusals escape as tracebacks. It is also called as a function (`scripts/probe_deployment.py`), and `SystemExit` is not an `Exception` | Typed exceptions a caller can catch; `main()` turns each into a message on stderr and exit status 2. The path-with-`--force-download` refusal now happens before the loader is constructed |

**Also closed:** `load_mondo()` / `load_hpo()` searched only the cache and skipped
the role check, where the build did not; they now search the configured roots
plus the cache and select through the same entry point. `ONTOLOGY_URLS` /
`ONTOLOGY_OWL_URLS` were left in `loader.py` after §3.6 moved the list to
configuration, read by nothing, and are removed. `ontology.sources.hp` was
accepted, scheme-checked and then never read; it is now the HPO source. An
`allowed_hosts` entry that can never match (`https://host`, `host/path`,
`host:443`) is refused when read, asking for the bare host name.

---

## 4. Acceptance

Named before the work rather than argued after it. Ordinary unit tests;
controlled responses where a fetch is involved, and no request to any real host.

**Selection**

1. An explicit path is used, and one that does not exist **refuses** rather than
   opening `<cache_dir>/<name>.obo`.
2. **An explicit path still succeeds while other valid candidates sit on disk**,
   and provenance records the named file's digest — not a rival's. This is what
   makes §3.1's refusal a default about ambiguity rather than a limit on
   coexistence, and the first draft's case 1 did not say the rivals were there.
3. **Two candidates across two roots refuse**, and the message carries each one's
   path, `data-version` and digest, and names `--mondo-path` / `--hpo-path` as
   the way to resolve it.
4. **Two releases in one root refuse** the same way.
5. **Exactly one candidate is used**, so the refusals above are not a resolver
   that rejects everything.
6. **`force_download` with an explicit path refuses**; without one it still
   bypasses what is present. A mutant that leaves the flag in the signature while
   the resolver returns first must fail a test.

**Migration from the cache convention** (§3.1.2 — these change behaviour, and
the tests exist so the change is deliberate rather than discovered)

7. **`--ontology-cache-dir` holding one file still works**, now as the last root.
8. **A cache holding both `<name>.obo` and `<name>.owl` refuses**, where today
   the OBO wins silently. The message shows both `data-version` values.
9. **A cache with one file and another root with one file refuses**, rather than
   the cache's position deciding it.
10. Adding `--mondo-path` to either of those makes the build succeed — the
    documented migration actually works.

**The resolver**

11. Identity, not presence: `data-version`, digest, size and term count for each
    file found.
12. **Constructing a resolver creates no directories** (§1.3's `mkdir`).
13. **Enumeration issues no network request**, including for the term count, and
    including when a listed file declares an import.

**Imports**

14. A file declaring an import is **refused, naming it** — it must not load with
    the import dropped.
15. A file declaring an import that points at a **local** file is refused too
    (§3.3 as written, not as the first draft said).
16. A self-contained file loads with nothing fetched.

**Role**

17. A file declaring `ontology: mondo` in the HPO slot **refuses**, naming the
    file, what it declares and what was expected. The reproduction in §3.4 is the
    fixture.
18. A legitimate ontology carrying cross-references to other namespaces **still
    loads** — the check is not "every term has one prefix".
19. `hp` and `hpo` are accepted as the same ontology.
20. An explicit path failing the role check does **not** fall back to a cache.

**Downloads — scheme**

21. A source whose scheme is not allowed is refused **when the list is read**,
    naming the entry.
22. A permitted initial URL that **redirects to `ftp://` is refused at the
    redirect**, which is the measured gap the config-time check misses.
23. An ordinary `https` → `https` PURL redirect is still followed.
24. **The OWL fallback goes through the same downloader** and is refused by the
    same rules; a test must fail if it acquires its own fetch path.

**Downloads — destination** (§3.5.1; a stub resolver supplies the addresses, so
no test touches DNS or any host)

25. A host resolving to a **private, loopback or link-local** address is refused,
    naming the address. `169.254.169.254` is one of the cases.
26. A **literal private IP** in the URL is refused — the check is on addresses,
    not on whether a hostname appears in a list.
27. A host on the **allow list resolving inside the network is permitted**, so an
    in-house mirror works. Without this the default is a blanket block.
28. A permitted public URL that **redirects to a private address is refused at
    the redirect**, the destination twin of case 22.
29. A host resolving to **several addresses, one of them private**, is refused —
    not admitted because one answer was acceptable.

**Provenance**

30. Unchanged in shape, and now describing a selected input: a build given
    explicit paths records the same four roles with the same digests and declared
    versions Phase 1 records today.

---

### 4.1 Where each condition is held

Filled in as the work landed, so "30 conditions to be met" became "30 conditions
with a named test" rather than a claim.

| # | Condition | Test |
|---|---|---|
| 1 | explicit path used; missing refuses | `test_ontology_resolver.py::TestSelection::test_an_explicit_path_that_does_not_exist_refuses` |
| 2 | explicit path wins with rivals on disk | `…::test_an_explicit_path_wins_while_other_versions_sit_on_disk` |
| 3 | two roots refuse, naming both | `…::test_two_roots_each_holding_one_refuse`, `…::test_more_than_one_candidate_refuses_and_names_them` |
| 4 | two releases in one root refuse | `…::test_more_than_one_candidate_refuses_and_names_them` |
| 5 | exactly one is used | `…::test_exactly_one_candidate_is_taken` |
| 6 | `force_download` contradiction; not decorative | `test_ontology_selection_cli.py::TestForceDownload` (3) |
| 7 | `--ontology-cache-dir` with one file works | `…::test_exactly_one_candidate_in_the_cache_still_works` |
| 8 | `.obo` + `.owl` in one cache refuses | `test_ontology_resolver.py::…::test_obo_is_not_preferred_over_owl`, `test_ontology_selection_cli.py::…::test_a_cache_holding_obo_and_owl_refuses` |
| 9 | cache + another root refuse | `test_ontology_selection_cli.py::…::test_a_root_and_the_cache_each_holding_one_refuses` |
| 10 | naming a path fixes it | `…::test_naming_a_path_fixes_that` |
| 11 | identity, not presence | `test_ontology_resolver.py::TestIdentityWithoutAParser` (7) |
| 12 | no directory creation | `…::test_it_creates_no_directories` |
| 13 | enumeration issues no request | `…::test_it_issues_no_network_request` |
| 14 | a declared import refuses, named | `test_ontology_imports_and_role.py::…::test_a_declared_import_is_refused_and_named` |
| 15 | a local import refuses too | `…::test_an_import_pointing_at_a_local_file_is_refused_too` |
| 16 | self-contained loads, nothing fetched | `…::test_a_self_contained_file_loads`, `…::test_nothing_is_fetched_while_parsing` |
| 17 | wrong slot refuses, named | `…::TestTheRoleCheck::test_the_reproduction_is_refused`, `test_ontology_selection_cli.py::…::test_a_wrong_slot_stops_the_build` |
| 18 | cross-namespace terms still load | `…::test_cross_namespace_terms_do_not_make_a_file_wrong` |
| 19 | `hp` and `hpo` are one | `…::test_hp_and_hpo_are_the_same_slot`, `test_ontology_resolver.py::TestTheAliases` |
| 20 | a failed role check does not fall back | `test_ontology_selection_cli.py::…::test_it_does_not_fall_back_to_the_cache` |
| 21 | bad scheme refused when read | `test_ontology_download_policy.py::…::test_the_settings_reader_asks_the_same_module` |
| 22 | redirect to `ftp` refused | `…::test_a_redirect_to_ftp_is_refused` (premise measured by `…::test_the_stdlib_would_follow_https_to_ftp`) |
| 23 | ordinary redirect still followed | `…::test_an_ordinary_redirect_is_still_followed` |
| 24 | the OWL fallback shares the downloader | `…::test_the_owl_fallback_has_no_fetch_of_its_own` |
| 25 | private / loopback / link-local refused | `…::test_an_address_inside_the_network_is_refused` (6) |
| 26 | a literal private IP refused | `…::test_a_literal_private_address_is_refused` |
| 27 | an allow-listed mirror permitted | `…::test_an_allowed_host_is_permitted_inside_the_network` |
| 28 | redirect into the network refused | `…::test_a_redirect_into_the_network_is_refused` |
| 29 | one private answer among several refuses | `…::test_one_private_answer_among_several_refuses` |
| 30 | provenance unchanged in shape | `test_ontology_selection_cli.py::TestProvenanceIsUnchangedInShape` (2) |

Cases 2, 30, 22, 23 and 28 are also held end to end: 2 and 30 through the real
build entry point with rival releases in a root and in the cache and every socket
entry point detonating
(`tests/integration/test_build_provenance.py::test_named_files_are_what_the_record_and_the_rebuild_command_name`),
and 22/23/28 through a real `urllib` opener rather than by calling the redirect
handler directly (`test_ontology_download_policy.py::TestTheRedirectGuardIsInThePath`).

### 4.2 Conditions added by the pre-PR review

Each is a confirmed finding of the whole-branch review (§3.2, §3.3, §3.4,
§3.5.2, §3.8), with the tests that hold it.

| # | Condition | Test |
|---|---|---|
| 31 | a misfiled cache file is never overwritten; nothing is fetched | `test_ontology_selection_cli.py::TestAMisfiledCacheFileIsNeverOverwritten` (2) |
| 32 | a fresh copy is verified before it replaces anything | `…::TestVerifyThenPublish` (5) |
| 33 | a failed transfer is reported, not replaced by a file on disk | `…::test_a_transfer_failure_is_reported_not_papered_over`, `…::test_a_fresh_copy_that_never_arrives_is_not_replaced_by_the_old_one` |
| 34 | nothing configured to fetch says so, naming the roots | `…::TestNothingToFetchIsSaidPlainly` |
| 35 | `load_mondo()` / `load_hpo()` search the configured roots | `…::TestTheNamedLoadersSearchTheConfiguredRoots` |
| 36 | refusals are exceptions and an exit status; the probe forwards paths | `…::TestTheBuildEntryPoint` (4) |
| 37 | release products are their ontology, stacked suffixes included | `test_ontology_resolver.py::…::test_a_release_product_is_its_ontology`, `test_ontology_imports_and_role.py::TestAReleaseProductIsItsOntology` (3) |
| 38 | the loader refuses every import the listing shows, and a targetless one | `test_ontology_imports_and_role.py::TestTheLoaderAndTheListingAgreeOnImports` (4) |
| 39 | the listing reads format by content, keeps memory flat, skips hidden files, warns on a misfiled file | `test_ontology_resolver.py::TestFindingsFromTheHolisticReview` |
| 40 | no local answer is not a policy verdict; a proxy is deferred to only when the request goes to it, and the request uses that proxy | `test_ontology_download_policy.py::TestNoLocalAnswerIsNotAPolicyVerdict` (5), `…::TestTheRequestGoesWhereTheRuleLooked` |
| 41 | chunked bodies, interrupts, file mode, malformed URLs | `…::TestTheTransferEdges` |
| 42 | the OWL fallback reaches the one downloader, behaviourally | `…::TestTheOwlFallbackGoesThroughTheSameGate` |
| 43 | `sources.hp`, malformed URLs and unmatchable allow-list entries in settings | `…::TestTheSettingsFindings` |
| 44 | each slot is hashed when it loads, not after both | `tests/integration/test_build_provenance.py::test_each_slot_is_hashed_when_it_loads_not_after_both` |
| 45 | two fetches at once: neither publishes the other's bytes, each carries its own digest | `test_ontology_selection_cli.py::TestTwoFetchesAtOnce` (2) |
| 46 | digest, parse and import scan are one reading of one file | `test_ontology_imports_and_role.py::TestTheDigestIsOfWhatWasParsed` |
| 47 | the proxy exemption holds only where the opener really goes to the proxy | `test_ontology_download_policy.py::TestTheRequestGoesWhereTheRuleLooked::test_the_rule_and_the_opener_agree` (18), `…::test_a_bypass_written_with_a_port_is_honoured` |
| 48 | a targetless import is refused wherever it is written | `test_ontology_imports_and_role.py::…::test_a_targetless_import_is_refused_wherever_it_is_written` (6), `…::test_an_obo_import_with_no_value_is_listed_as_one` |
| 49 | a real file whose encoding chardet misreads still loads, text and digest intact | `test_ontology_imports_and_role.py::TestAFileChardetMisreadsStillLoads` (3) |

### 4.3 What the first real download found

**Every condition above held and the merged code still could not load a real
HPO.** The deployment probe's download phase (added after this plan's PR)
fetched the 2026-09-01 releases through the production path: MONDO was
published; `hp.obo` failed to load with `seek of closed file`.

The cause was row 46's own fix. pronto guesses an encoding with chardet, and the
real `hp.obo` — valid UTF-8, 22 non-ASCII bytes in 10.9 MB, the first at byte
946 — is guessed ISO-8859-1. On that path pronto wraps the handle it is given in
an `EncodedFile` whose collection **closes it**, and `load`, which hashes,
parses and scans imports through one handle, then scanned a closed file. The
fixtures were ASCII and MONDO is guessed UTF-8, so no test, no mutant and no
review saw it. pronto now gets its own descriptor for the same open file
(`os.dup`): the same bytes, the same refusal to be replaced on Windows, and
pronto closing it closes only the duplicate. Measured on the real file after the
fix: 19,894 terms, `Folie à deux` intact, digest equal to the file's.

**Measured, so not assumed:** pronto's guessed decoding and a forced UTF-8
decoding give identical term names and synonyms for this release (zero
differences), so the misguess did not garble text; it only closed the handle.

This is the case the plan's §6 names in general — a named test is not a proof —
arriving in particular. It is why the download phase exists.

### 4.4 The full probe on a GB10 machine

Evidence: [`EVIDENCE_probe_phase2_homelab.json`](EVIDENCE_probe_phase2_homelab.json)
`d01621f3…` — the report `scripts/probe_deployment.py` wrote, committed byte for
byte. Run on the homelab (Linux aarch64, NVIDIA GB10, capability 12.1, torch
2.10.0+cu130) on 2026-09-29, with `--device cuda --external-dir data/external
--download`, at `43d47c5` as the operator reported it. **The report itself does
not record the commit**, so that attribution rests on the operator's word.

**48 probes, 48 passed.** What they settle for this plan — each row says where
its figures come from, because not all of them are in the report:

| | Measured | Source |
|---|---|---|
| Selection on an existing cache | The default cache held one MONDO (`releases/2026-06-02`) and one HPO (`hp/releases/2026-06-06`), downloaded by the old pipeline in June, and no ontology root is configured. The real build passed without a path, so it took them as the single candidates — **inferred**: the report records that it built, not which files it read | the operator's listing before the run (`enumerate_candidates`); F1 in the report |
| The §4.3 regression, on this machine | The June `hp.obo` is guessed ISO-8859-1 by chardet too, so before #107 this build would have failed with `seek of closed file`. It built | the console's `UnicodeWarning`; F1 |
| Reproducibility | Two builds from the same inputs, all eight canonical artifacts byte-identical (`kg_digest` `5ad83ab2…`); 54,912 nodes and 615,031 edges each | F2 in the report; the counts from the console |
| A real download, direct | 2026-09-01 releases over a home connection: both published, no proxy involved, every source name resolved locally. **MONDO `50c8367f…` and HPO `93dace95…` are the digests the development sandbox got for the same releases** — two networks, the same bytes | G1, G2 |
| Nothing outside the work directory | The two June files in the default cache untouched | G9 |
| The device | No native kernels for sm_121 (the build stops at sm_120 plus `compute_120` PTX); matmul and `scatter_add` run on the device and agree with the host (max error 4.6e-5) | A3, A4 |

**Still not measured.** A hospital network: whether `github.com` and
`release-assets.githubusercontent.com` — where the PURLs redirect for the file
itself — are allowed, whether port 80 is open for the first hop, whether HTTPS
is intercepted. The hand-off of an unresolvable name to a proxy (§3.5.2) was
not exercised, since every name resolved. And whether these results stand for
the hospital's deployment machine depends on it being of the same class.

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
- §3.5 provides an enforcement point and the default of §3.5.1. It does not
  decide a site's network policy, and neither `http`/`https` nor `is_global`
  makes a destination trustworthy — only reachable-by-policy.
- **The destination check is check-then-connect.** A name whose resolution
  changes between the check and the socket is not covered; pinning the connection
  to a checked address is outside Phase 2 and is not claimed.
- **A proxied name with no local answer is judged by the proxy, not here**
  (§3.5.2). That is an amendment to an approved default, flagged for review.
- **Every condition in §4.1 and §4.2 has a named test, and a named test is not a
  proof.** The tests were additionally run against deliberately broken copies of
  the code (mutation testing) to check that they fail when the rule they name is
  removed; that shows they detect those particular breakages, not that no other
  breakage exists. The last sweep was 73 mutants, one per contract in §4.1 and
  §4.2 including each of the three defects rows 45–48 were written against;
  every one was killed by the test of the rule it breaks. (In the sweep before
  it, five of 63 survived, each for want of a test, and those tests were
  added.) The harness is a working script and is not committed, so this is a
  report of what was run, not something a reviewer can re-run from the tree.
- **The UI slot of §3.6.1 is owed and not delivered.**
- Nothing here touches `DISEASE_SCORER_POLICY.md`, the shortest-path artifacts,
  or any checkpoint.
