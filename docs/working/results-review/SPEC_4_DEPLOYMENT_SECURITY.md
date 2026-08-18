# Spec 4 — Deployment security and threat model

**Status:** draft, under review. **Normative.** Supersedes rev 6 §10 and §12.
**Evidence:** `5c5c4b2` (main) plus branch commits `42e5d1e`, `337266f`, `dc9663b`, `7dab728`
(committed and pushed on `claude/dev-context-review-3wuh05`, **under review, not merged**).

---

## 1. Current state, verified

Every claim here was read from the source, not inferred.

| Fact | Location |
|---|---|
| Six routers mounted. **No mounted route or router applies an authentication or authorisation dependency.** An unused `Depends` import exists in `diagnose.py:35` and is not a control | `src/api/main.py:277-283` |
| The deployed launcher binds `0.0.0.0` | `scripts/launch/shep_launch.py:17` |
| API CLI entry point binds `0.0.0.0` | `src/api/main.py:466` |
| Supplied systemd unit binds `0.0.0.0` (and names a module path that does not exist: `app.main:app`) | `scripts/service/systemd/shepherd.service:9` |
| Generated demo commands and the package usage example bind `0.0.0.0` | `scripts/setup_demo.py:545,552`; `src/api/__init__.py:25` |
| **One contradictory instruction** omits `--host` and inherits Uvicorn's loopback default | `scripts/setup_demo.py:29` |
| The launcher **prints `127.0.0.1` while binding `0.0.0.0`** | `scripts/launch/shep_launch.py:323-333` |
| CORS is `allow_origins=["*"]` with `allow_credentials=True` | `src/api/main.py:161-166` |

**Institutional statement, reported and not verified here:** the hospital laboratories sit on a
protected local network; reaching the service across the internet requires SSH. This bounds the
audience to that network segment. It is not an authorisation mechanism and does not distinguish one
person on the segment from another.

### 1.1 The unauthenticated surface is not only diagnosis

This is the finding that most changes the picture, and earlier drafts of this design missed it
entirely by reasoning only about diagnosis and future snapshot history.

| Unauthenticated operation | Location | Effect |
|---|---|---|
| `POST /pipeline/reload` — accepts client-supplied `data_dir`, `checkpoint_path`, `device` | `pipeline.py:183-203` | Releases and reloads the shared singleton |
| `torch.load(..., weights_only=False)` on client-selected paths — **five sites, including the final model load** (§2.1) | `inference/pipeline.py:749` and four others | **Arbitrary deserialisation.** `weights_only=False` executes code during load |
| `POST /pipeline/config` | `pipeline.py:385-397` | Writes server configuration |
| `POST /training/start` — accepts `data_dir`, `output_dir`, `checkpoint_dir`, `log_dir`, `resume_from`, device and hyperparameters | `training.py:49-65, 174-191` | Launches a training subprocess |
| `POST /training/stop` | `training.py:194-200` | Controls the running training process |

So the surface currently permits model and knowledge-graph reload, configuration mutation, training
launch and stop, consumption of GPU, CPU, disk and process resources, and a code-executing
deserialisation path.

---

## 2. Two remedies, in order

**The bind is the primary remedy.** If the listener returns to loopback, everything in §1.1 becomes
locally reachable only, and the route-class work below has no audience to protect against. That is
one configuration change plus corrected instructions, and it is the cheapest correction available.

**Route classes matter when the bind stays open** — under mode M3 below. They are written down so
that choice is informed, not because a taxonomy is required first.

| Class | Members | Under M3 |
|---|---|---|
| **C1 Clinical inference and search** | `/diagnose`, `/search`, `/disease`, read-only `/system` | Already exposed today; the institution has in effect accepted this |
| **C2 Result history** | snapshot list, load, delete; manual snapshot creation | Requires the documented access decision (Spec 2 §9) |
| **C3 Pipeline and system administration** | `/pipeline/reload`, `/pipeline/config`, checkpoint inspection | **Disable or protect.** Not covered by clinical-exposure risk acceptance |
| **C4 Training control** | `/training/start`, `/training/stop` | **Disable or protect** |

> **C3 and C4 are different in kind from C1 and C2.** Accepting that a clinical tool is reachable on
> a lab segment is a decision about *who may use the tool*. C3 and C4 let a caller swap the model out
> from under a clinician mid-session, rewrite server configuration, and consume the GPU — and reach a
> code-executing deserialiser. `SHEPHERD_ACCEPT_UNAUTHENTICATED_NETWORK_BIND=1` (§3.1) does not
> unlock them.

### 2.1 Checkpoint deserialisation — the full site list

An earlier draft scoped this to two metadata reads and called it "one keyword argument". **That was
wrong, and it left the dangerous path open:** the *final model load* still deserialises with
`weights_only=False`, and it is the one a client-supplied `checkpoint_path` actually reaches.

| Site | Reachable from |
|---|---|
| `src/inference/pipeline.py:749` — **the final model load** | `POST /pipeline/reload` `checkpoint_path` |
| `src/api/routes/pipeline.py:228` — checkpoint scoring | same route |
| `src/api/routes/pipeline.py:267` — selected-checkpoint log read | same route |
| `src/api/services/training_manager.py:464` — checkpoint listing metadata | training routes, via a **client-influenced checkpoint directory** set through the training configuration. It globs `self.checkpoint_dir` and takes no path argument |
| `src/training/trainer.py:960` — resume | `POST /training/start` `resume_from` |

> **P0 — the five client-reachable loads above.** Audit each, verify real repository checkpoints
> against `weights_only=True`, and convert every compatible load. A format that is incompatible must
> be explicitly migrated, or restricted to a trusted source.

**"Trusted source" means selected by server-side configuration or an allowlisted repository.** A path
supplied in the request is client-selected and is *not* trusted merely because it resolves locally.

**Ordinary hardening, not P0** — same change, lower urgency, because they are not reachable over
HTTP: `scripts/migrate_checkpoints.py:48`, and `scripts/evaluate_model.py:111, 201, 205` which pass
no `weights_only` at all.

> Calls that omit `weights_only` inherit a **version-dependent default. Make the argument explicit.**

(`pyproject.toml:34` pins `torch==2.10.0`; the version present in any given working environment may
differ, which is precisely why the argument is stated rather than inherited.)

**Use PyTorch's built-in `weights_only` mechanism and the existing state-dict checkpoint format. Do
not write a custom safe unpickler.** The codebase already loads graph tensors and the shortest-path
artifact with `weights_only=True` (`pipeline.py:647`, `build_index.py`, `train_model.py`,
`setup_demo.py`), so this closes a gap in existing practice rather than introducing a new one.

**Compatibility must be tested against every checkpoint variant this repository produces**, or the
conversion breaks a path nobody exercised: `Trainer.save_checkpoint` output; callback checkpoint
output; a legacy migrated checkpoint; training resume; checkpoint metadata selection; the final
inference model load.

**Generate the fixtures by calling the repository's own save and migration code**, then exercise the
real consumers against what it produced. Do not hand-maintain large checkpoint dictionaries, and do
not build a checkpoint-fixture framework — a hand-written fixture tests the fixture, not the format.
Institutional checkpoints remain the final compatibility check. **Do not write a custom unpickler.**

This clone has no checkpoint, so the institutional half of that verification is part of the work
item, not of this document.

*On proportionality:* calling this remote code execution overstates it. A working chain also needs a
way to place a chosen file on that host, which is not obviously available through these routes. But a
network endpoint reaching a code-executing deserialiser is worth closing regardless.

---

## 3. Deployment modes

| Mode | Conditions |
|---|---|
| **M1 Local single workspace** | Loopback or SSH-forwarded access to a verified single-workspace deployment; one explicit workspace namespace |
| **M2 Authenticated** | Authenticated actor; authorisation; ownership; per-request checks |
| **M3 Accepted protected segment** | Documented network boundary; known reachable audience; C3/C4 disabled or protected; **recorded institutional acceptance of the risk** |

**All three are acceptable deployment postures.** An earlier draft permitted result history only
under M1 or M2, which silently replaced the institution's deployment decision with an architecture
requirement. Whether a protected laboratory segment is an acceptable perimeter for this tool is the
institution's judgement to make, not the designer's — and they have already made it for C1.

**M3 is not equivalent to authentication and is never described as though it were.** It is a recorded
decision that the audience the segment admits is acceptable. The design's job is to make that
audience legible.

**Loopback is not authorisation either.** It does not distinguish clinicians, local users, browser
sessions or SSH tunnels, and several people may reach the service through separate SSH forwards while
every request appears to originate from loopback. M1 therefore names a *verified single-workspace
deployment*, not merely a loopback bind.

### 3.1 The bind guard

- Unauthenticated launch defaults become `127.0.0.1`.
- A non-loopback bind without authentication is refused unless
  `SHEPHERD_ACCEPT_UNAUTHENTICATED_NETWORK_BIND=1` is set. The name states that this is **risk
  acceptance, not a control**; a self-declared flag authenticates nobody.
- Enforcement points: the launcher's default arguments, `src.api.main:main()`, and application
  startup reading the host the first two pass in.
- **Not covered, and not claimed to be:** a person typing `uvicorn ... --host 0.0.0.0` by hand. The
  application cannot read Uvicorn's CLI. That path is addressed by corrected instructions and by
  fixing the launcher's misleading URL display, not by an in-process guard.

---

## 4. P0 work

**Three kinds of work, separated because they have different owners and different urgency.**

**A — Immediate security and data fixes.** No decision needed; start now.

| # | Item |
|---|---|
| A1 | Audit and convert every client-reachable `torch.load` (§2.1) |
| A2 | Fix the launcher's misleading display (`shep_launch.py:323-333`) — see below |
| A3 | Fix the unbounded eager export accumulation (`diagnosis_panel.py:487-496`) |
| A4 | **Fix the systemd unit's invalid module path** (`app.main:app` → `src.api.main:app`) and any shipped command that cannot start as written |

**A2 in detail.** The launcher currently prints `127.0.0.1` while binding `0.0.0.0`. The fix is not
to print `0.0.0.0` as a URL — that is not a browser destination either. Report the bind and the
destinations as separate lines:

```
Listening on : 0.0.0.0:8000 (all interfaces)
Local UI     : http://127.0.0.1:8000/ui
Network UI   : use http://SERVER_HOST_OR_IP:8000/ui
```

The network line is deliberately **not a clickable URL** — an uppercase placeholder cannot be
mistaken for a working address, whereas `<hostname>` in angle brackets often is. Auto-open stays on
loopback. **Do not build network-interface discovery merely to print a URL.**

**B — Deployment decision.** Needs the LAN answer below.

| # | Item |
|---|---|
| B1 | Bind defaults to `127.0.0.1`, with the risk-acceptance opt-in (§3.1) |
| B2 | Under M3 only: unmount or protect C3 and C4 |

**C — Host-value documentation.** Follows B, no decision of its own.

*Packaging correctness is not all in this group.* A unit that names a module which does not exist is
broken under every deployment mode, so it is A4, not C1. Only the **host value** waits for the bind
decision. **Do not keep shipping a known-broken service unit while a bind decision is pending.**

| # | Item |
|---|---|
| C1 | Correct the **host values** in `setup_demo.py` (the `0.0.0.0` commands and the contradictory line 29), `src/api/__init__.py`, and the systemd unit |
| C2 | Document the actual deployment contract; withdraw the concurrency-safety claim wherever it appears |

**One operational answer decides group B:** *does anyone reach the WebUI by the server's LAN address
rather than through SSH?*

- **No** → the bind moves to loopback, the deployment is M1, and B2 has no audience.
- **Yes** → B1 would cut those users off. The institution chooses between bringing authentication
  forward (M2) and recording protected-segment acceptance (M3). Under M3, B2 becomes required and
  Spec 2 §9's access decision is path C.

Group A proceeds regardless and must not wait for that answer.

---

## 5. Concurrency

`initialize_pipeline` reassigns the shared `app_state.pipeline` (`src/api/main.py:424`) with no lock
or synchronisation contract in the module. **Concurrency safety is unverified.** A reload concurrent
with an in-flight diagnosis has no defined behaviour. Any multi-user deployment requires an explicit
concurrency model — locking or serialised execution — plus load testing.

---

## 6. Multi-user target state (P4) — not designed here

Recorded so the answer is not re-derived, not to be built.

The front/back boundary is already HTTP: the WebUI calls the API rather than importing the pipeline
(`src/webui/components/diagnosis_panel.py:18, 116-117`), which makes separate deployment tractable.
Genuinely small: making `API_BASE` configurable (hard-coded at `:44`); a standalone Gradio launch
path; CORS. Everything else is not — authorisation, TLS termination, bounded inference queueing, GPU
concurrency limits, reload-versus-inference synchronisation, request cancellation, per-actor quotas,
storage isolation and locking, operational monitoring.

**CORS** (`main.py:161-166`, `["*"]` with `allow_credentials=True`) is reviewed when this area is
touched. With no authentication present it is not the exposure; the absent authentication is. No
CORS design is written before then.
