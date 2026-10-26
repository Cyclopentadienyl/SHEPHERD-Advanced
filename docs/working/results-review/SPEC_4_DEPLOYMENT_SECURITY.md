# Spec 4 — Deployment security and threat model

**Status:** draft, under review. **Normative.** Supersedes rev 6 §10 and §12.
**Evidence:** `5c5c4b2` (main) plus branch commits `42e5d1e`, `337266f`, `dc9663b`, `7dab728`
(committed on `claude/dev-context-review-3wuh05`; merged as #100, `6265a81`).

---

## 1. State as verified at `5c5c4b2` (historical)

> **Superseded as a description of the running system (noted 2026-10-26).** This table is the state
> read from the source at `5c5c4b2`, kept as the record that motivated §2–§5. Since then:
> - **checkpoints** load with `weights_only=True` (M3a-1e); no `weights_only=False` remains in
>   `src` or `scripts` (checked at `7b0162f`);
> - **the bind** defaults to `127.0.0.1`, with the risk-acceptance guard for any other address (B1,
>   M5-1, §3.1 "as implemented");
> - **CORS** has an empty allow list;
> - **every operator route** (reload, configuration, training start and stop, the server lists) is
>   checked by D8 on every request (`OPERATOR_ONLY`; M5-1);
> - **the systemd unit** names the right module and binds `127.0.0.1` (A4, C1).
>
> The service runs under deployment mode M1: the owner or a designated technician, on the machine
> or through controlled SSH forwarding. This is settled and not asked again unless the deployment
> changes (`../PLAN_TEST_RESULTS.md` §4.9, question 3). M5's GB10 acceptance checked D8, CORS and the
> route guard on the running service (`../m5-acceptance-2026-10-24/README.md`, W2–W6).
>
> **Still open:** the export files' accumulation (A3; BACKLOG 56), and everything about snapshot
> storage and access (Spec 2 §9). Multi-user remote operation is separate work (§6). For what is
> implemented now, read §2–§4's "as implemented" notes and `../../ARCHITECTURE.md` §11.

Every claim in the table was read from the source at `5c5c4b2`, not inferred.

| Fact | Location |
|---|---|
| Six routers mounted. **No mounted route or router applies an authentication or authorisation dependency.** An unused `Depends` import exists in `diagnose.py:35` and is not a control. *Since contract M4 (2026-10-16), seven:* the seventh, `src/api/routes/evaluations.py`, serves one read-only route, `GET /api/v1/evaluations/loaded-model` (the loaded checkpoint's recorded test results; no parameters, no write), on its own router outside `pipeline.router`, and applies no authentication dependency either. The note adds no route class to §2 | `src/api/main.py:277-283`; since M4 `src/api/main.py:333-343` |
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

**Ordinary hardening, not P0** — same change, lower urgency, because it is not reachable over
HTTP: `scripts/migrate_checkpoints.py:50`. The three loads in `scripts/evaluate_model.py` that
passed no `weights_only` at all (`:111, 201, 205` at `7dab728`) went with that file, deleted on
2026-10-10 (`docs/working/PLAN_PROVENANCE_CONTRACT.md` M2.1, S0).

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

*As implemented (M5-1, 2026-10-20):* one rule, `network_bind` in `src/config/server_settings.py`:
a loopback bind (`127.0.0.0/8`, `::1`, `localhost`) is allowed; any other, the wildcards included,
is refused unless `SHEPHERD_ACCEPT_UNAUTHENTICATED_NETWORK_BIND` is exactly `1`, and then allowed
with a warning that names it risk acceptance, not a control, and says the operator controls stay
refused off loopback (D8, `PLAN_TEST_RESULTS.md`). Applied at the three enforcement points:
- the launcher (default `--host 127.0.0.1`), right after reading its arguments, before it installs
  or starts anything: it reads what uvicorn will listen on as uvicorn does, a socket it is handed
  (`--fd`) before a unix socket (`--uds`) before the last `--host`, from the command line or else
  `UVICORN_FD` / `UVICORN_UDS`; refuses with exit 2; and prints what it binds. An inherited
  socket's address is not read, so it counts as a network bind (`inherited_socket_bind`); a unix
  socket reaches this machine only and is allowed, with no URL shown (D8 refuses every operator
  control through it);
- `src.api.main:main()` (default `127.0.0.1`, refusing with `SystemExit`);
- application startup (`startup_bind`, first in the lifespan): both entry points pass the host
  they bind as `SHEPHERD_BIND_HOST`, set every time and removed when they bind no host, and
  startup applies the same rule to it, a refusal ending startup so uvicorn exits without serving.
  A hand-typed uvicorn passes no host and stays outside, as above.

The launcher also starts uvicorn with `--no-proxy-headers`, last on its command line. The service
has no authentication, so every network bind counts as one "without authentication". *Corrected
after review (2026-10-20):* the first version of this note said startup "cannot read the bind
uvicorn was given" and left it out; startup reads the host the other two pass in, as this section
asks. The launcher's check also ran after its accelerator installs, and it read `--host` only.

---

## 4. P0 work

**Three kinds of work, separated because they have different owners and different urgency.**

**A — Immediate security and data fixes.** No decision needed; start now.

| # | Item |
|---|---|
| A1 | Audit and convert every client-reachable `torch.load` (§2.1). *Done:* serving loads checkpoints with `weights_only=True` (M3a-1e), and no `weights_only=False` remains in `src` or `scripts` (checked 2026-10-26) |
| A2 | Fix the launcher's misleading display (`shep_launch.py:323-333`) — see below. *Done in M5-1 (2026-10-20):* the launcher prints `Listening on : <host>:<port> (<reach>)` before the URLs, which name the loopback address a browser on the machine can open. No "Network UI" line is printed: a network bind needs `SHEPHERD_ACCEPT_UNAUTHENTICATED_NETWORK_BIND=1`, and the operator controls stay refused off loopback (§3.1) |
| A3 | Fix the unbounded eager export accumulation (`diagnosis_panel.py:487-496` when written). *Still open* in its current form: export files are not cleaned up while the service runs (BACKLOG 56) |
| A4 | **Fix the systemd unit's invalid module path** (`app.main:app` → `src.api.main:app`) and any shipped command that cannot start as written. **Done for the unit:** module path, the project's interpreter in place of `/usr/bin/python`, `default.target` for a user unit, and a start through the launcher (`shep_launch.py --no-auto-install --no-browser -- --port 8264`), the path `launch_shepherd.sh` takes, so the service gets the saved allocator preset and the attention settings a manual start gets, after which the launcher execs into uvicorn (POSIX, `--no-browser`). A bare uvicorn start gets neither; `tests/unit/test_service_unit.py`. The bind address is the launcher's default and stays with C1. *Since M5-1 (2026-10-20):* the unit starts with `-- --host 127.0.0.1 --port 8264`, the bind stated where an operator would change it, and C1 is done (B1 below). **Still owed, on a deployment host:** a real systemd start, stop and crash restart, and a UI Restart Backend in the running service. Function-level tests and the container smoke do not stand in for them |

**A2 in detail** (the record before M5-1, when the launcher bound `0.0.0.0` by default). The launcher then printed `127.0.0.1` while binding `0.0.0.0`. The fix is not
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

**Decided 2026-10-19** (the reviewer's decisions, relayed by the owner, 2026-10-19; `PLAN_TEST_RESULTS.md` §4.9 question 3 and §6,
"M5's boundaries"): M5's first version runs under M1, and **B1 is taken: the bind defaults to
`127.0.0.1`**. Another address stays possible by explicit configuration under §3.1, never
lifting the operator controls' checks (D8); deployments that relied on the old default get
migration notes. LAN inference for several frontends is kept possible, not built (BACKLOG item
28). The question below is the record as it stood.

*As implemented (M5-1, 2026-10-20):*
- **B1:** the launcher (`UVICORN_DEFAULT_ARGS`), `src.api.main:main()` and the systemd unit
  (`-- --host 127.0.0.1 --port 8264`, stated where an operator would change it) bind `127.0.0.1`,
  with §3.1's guard ("as implemented" there). The launcher prints `Listening on : <host>:<port>`
  with what it reaches beside the URLs; migration notes for deployments that relied on `0.0.0.0`
  are in the deployment guide (§4.13). B2 has no audience under M1: C3 and C4 are not unmounted
  but checked by D8 on every request, under any bind, in the API (one FastAPI dependency) and in
  the dashboard (one Gradio helper), refused unless mode M1 and a loopback arrival with no
  forwarding header, a loopback Host, no foreign Origin and no `Sec-Fetch-Site` other than
  `same-origin` or `none` (a GET another site's page makes carries no Origin). CORS, reviewed
  with it (§6): the parent app's `CORSMiddleware` has an empty allow list, so no other origin may
  read a response, `/ui` included; Gradio's upload route, which the dashboard does not use, is
  refused in front of the mount, with its 1 MB per-file limit kept as a second line of defence.
  *M5-3 review (2026-10-21):* D8 cannot see framing (a dashboard loaded in another site's frame
  makes same-origin requests it allows), so every response refuses to be framed
  (`X-Frame-Options: DENY`, `Content-Security-Policy: frame-ancestors 'none'`,
  `src/api/middleware/frame_guard.py`); without it a framed dashboard would be open to
  clickjacking (a click turned into Restart Backend or a Load).
- **C1:** `src/api/__init__.py`'s usage example names `--host 127.0.0.1`, and the unit's comment
  no longer defers the bind. `scripts/setup_demo.py` already printed `--host 127.0.0.1` in both of
  its commands (since `0bc79d9`, 2026-09-09), and the instruction that omitted `--host` is gone,
  so it needed no change here. C2's deployment contract is the guide's §4.13.

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

*Observed 2026-10-19 (the multi-frontend audit for M5, BACKLOG item 28):* the routes are `async`
and call the pipeline on the event loop, so two diagnoses never overlap and a reload waits behind
a diagnosis; a running diagnosis keeps the pipeline and identity it started with. That is
serialisation by a blocked event loop, not a bounded queue: nothing reports busy or bounds the
wait, and `/health` stalls meanwhile. The bounded queue and a reload-while-busy rule are BACKLOG
item 28's, before any LAN opening. Managed training and evaluation jobs share one GPU lock
(`PLAN_TEST_RESULTS.md` §6); inference is not scheduled by it.

---

## 6. Multi-user target state (P4) — not designed here

Recorded so the answer is not re-derived, not to be built. *Since 2026-10-19:* BACKLOG item 28
records LAN inference for several frontends as kept possible (inference apart from management, a
candidate first version, the acceptance before opening), and `PLAN_TEST_RESULTS.md` §6 the
boundaries M5 keeps for it.

The front/back boundary is already HTTP: the WebUI calls the API rather than importing the pipeline
(`src/webui/components/diagnosis_panel.py:18, 116-117`), which makes separate deployment tractable.
Genuinely small: a configured API address for a WebUI in another process; a standalone Gradio launch
path; CORS. (The hard-coded `API_BASE` at `:44` was removed in October 2026, after this document's
evidence base: the in-process tab now calls an address the server accepted a request on,
`src/utils/server_address.py`.) Everything else is not — authorisation, TLS termination, bounded
inference queueing, GPU concurrency limits, reload-versus-inference synchronisation, request
cancellation, per-actor quotas, storage isolation and locking, operational monitoring.

**CORS** (`main.py:161-166`, `["*"]` with `allow_credentials=True`) is reviewed when this area is
touched. With no authentication present it is not the exposure; the absent authentication is. No
CORS design is written before then.
