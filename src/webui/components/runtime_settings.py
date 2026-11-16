"""
⚙ Settings: how computation runs on this server (M5-2; ops round B5)
====================================================================
Execution-level settings that change HOW computation runs on the hardware
(memory and compilation), as opposed to Research › Training, which configures
the model. One panel, its facts and fields before its actions (the WebUI
skill's §2), every value it shows escaped:

- **Running against saved** (`facts_html`): the memory allocator this backend
  runs now, read from its own environment by the rule that wrote it
  (`runtime_presets.running_allocator`: the saved preset at its start, an
  explicit setting in the launch environment, or none, PyTorch's default), is
  said apart from the preset saved for the next start, and torch.compile's
  saved value. An explicit setting in the launch environment is named as one,
  never a preset the launcher applied.
- **Fields**: the memory allocator, read once when the backend starts (a saved
  change applies after Restart Backend), and torch.compile, whose visible line
  leads with "Experimental" and says where it is read, once on the screen
  (`COMPILE_READ`: the Training page's starts, continues and branches;
  Fine-tune does not read it). Each is shown as saved each time the page is
  opened, from the same read as the facts. Their words say what is true today:
  GATv2 is the one architecture built; a measurement made on an earlier
  architecture is named as such.
- **Acceleration**: no toggle; the area says "Not implemented yet"
  (`ACCELERATION`): new acceleration work is deferred, and each run's record
  keeps what it ran with.
- **Actions**: Apply Settings saves the fields to
  ``.shepherd_runtime_settings.json`` (`src.config.runtime_presets`), which the
  launcher reads at start-up (`scripts/launch/shep_launch.py`) and Restart
  Backend applies again. Restart Backend relaunches the backend, and is on only
  while a restart would apply a saved allocator (`restart_applies`, the rule
  the facts' "Not running yet" reads too); Restart Backend and Stop service
  each ask once in the panel's one question row (`_ask`: what it does to every
  open page and to a loaded model, then "Restart now" or "Stop service now",
  or Cancel) before anything is asked of the backend. "Restart now" makes
  the question's check again (`_restart_refusal`): a question left open
  restarts nothing once Restart has turned off, and Apply and the slow timer
  close it then with the reason (`_restart_question_closed`).

Operator controls (D8, M5-1):
    Restart Backend and Apply Settings each check their own Gradio request
    first (`src/webui/operator_access.py`) and are refused, changing nothing,
    unless D8 allows it. The page follows the one gated-button rule
    (`chips.gated`, the WebUI skill's §3) when it is opened, and the buttons
    on the shell's slow timer: no lock while D8 allows the request; for a
    request D8 refuses, both buttons disabled with the lock, the two fields
    that need the operator (the allocator and torch.compile) not interactive,
    and the visible "Operator controls only: …" reason beside the buttons
    (`chips.operator_only`). Restart and Stop service are also off while a
    GPU job runs, and Restart while no saved allocator waits for it, each
    said beside them. A disabled control is a display, never the control.

Module: src/webui/components/runtime_settings.py
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Optional

import gradio as gr

from src.api.services.backend_control import is_training_active, restart_backend, stop_backend
from src.config.runtime_presets import (
    ALLOCATOR_PRESETS,
    load_runtime_settings,
    resolve_allocator,
    running_allocator,
    save_runtime_settings,
)
from src.webui.components import data_locations
from src.webui.components import renderers as r
from src.webui.components.chips import LOCK, gated, operator_only
from src.webui.operator_access import operator_access
from src.webui.shell import SETTINGS as SETTINGS_PAGE
from src.webui.shell import plain_html, tab_elem_id

#: Each allocator preset in plain words, its name beside them (the WebUI
#: skill's §5: an internal name only beside a plain label). One entry per
#: `ALLOCATOR_PRESETS` key.
ALLOCATOR_WORDS = {
    "cuda_async": "Driver memory pool",
    "expandable": "Expandable segments",
    "native_roundup": "Native, rounded",
    "native": "Native, no tuning",
}


def allocator_words(preset: str) -> str:
    """A preset as an option and a fact name it: its words, its name beside
    them ("Driver memory pool (cuda_async)")."""
    return f"{ALLOCATOR_WORDS.get(preset, preset)} ({preset})"


#: The allocator select's options: each preset's words, its name the value.
ALLOCATOR_CHOICES = [(allocator_words(name), name) for name in ALLOCATOR_PRESETS]

#: What a restart does to inference (the ops round's I7), said in the question
#: Restart Backend asks and when it starts: the backend stops serving before it
#: replaces its process, and the new one loads only a model its start-up
#: configures (the deployment guide names the setting).
RESTART_STOPS_SERVING = (
    "A loaded model is stopped first (the diagnoses already accepted finish) and is not "
    "loaded again unless the service's start-up configures one."
)
#: Restart Backend, asked once before it restarts (`_on_restart_request`, R-12).
RESTART_CONFIRM_LABEL = "Restart now"
RESTART_QUESTION = ("Restart the backend? Every open page reconnects in a few seconds. "
                    + RESTART_STOPS_SERVING)
#: Why Restart Backend is off while nothing waits for it (`restart_applies`).
RESTART_IDLE = "Restart Backend is off: a restart would run the same allocator."

#: Why Restart and Stop service are off while a GPU job runs (the backend
#: refuses both too: a re-exec or a stop would end the job's process), said
#: beside the buttons.
RESTART_BUSY = "Restart and Stop service are off while a GPU job (training or a test run) runs."

#: Stop service (UI-06e): a service started from its desktop entry has no
#: terminal to stop it from. Asked once before it stops
#: (`_on_stop_service_request`), then stopped through the backend's orderly
#: stop (`backend_control.stop_backend`).
STOP_LABEL = "Stop service"
STOP_CONFIRM_LABEL = "Stop service now"
STOP_QUESTION = (
    "Stop the service? The dashboard and the API stop on this computer, and every open page "
    "loses its connection. Start it again from its desktop entry or launch_shepherd.sh.")
STOP_DONE = ("Stopping the service. This page stops answering in a moment; start it again from "
             "its desktop entry or launch_shepherd.sh.")

#: The acceleration area: a fact, never a toggle that does nothing; said in
#: brackets after its "Not implemented yet" mark.
ACCELERATION = ("new acceleration work is deferred; each run’s record keeps the kernels and "
                "precision it ran with")

#: Where torch.compile's saved value is read: the Training page's base Start
#: and its Continue or Branch (`training_console`), the only readers; the
#: Fine-tune page does not read it. Said once on the screen, in the field's
#: visible line (`FIELD_INFO`), in these words.
COMPILE_READ = ("read when the Training page starts, continues or branches a run; Fine-tune "
                "does not read it")
#: torch.compile's warning, the lead of the field's visible line (R-13).
COMPILE_EXPERIMENTAL = "Experimental: keep it off unless you are benchmarking it."

# Browser-side poller chained after the restart click. Re-exec'ing the backend
# gives the new process a fresh Gradio session, so the already-loaded page keeps
# polling against a session the new server doesn't know — System Resources and
# dropdowns hang until a manual refresh. This watches /health and reloads the
# page (an automatic F5) only once the NEW process is up, detected by EITHER the
# uptime resetting (uptime_seconds drops below the value seen at click time) OR a
# down→up transition. If the restart was refused (e.g. training active), neither
# condition fires, the backend never goes down, and the poller just times out
# without reloading.
_RESTART_RELOAD_JS = """
async () => {
    const get = async () => {
        try {
            const r = await fetch('/health', {cache: 'no-store'});
            return r.ok ? await r.json() : null;
        } catch (e) { return null; }
    };
    const first = await get();
    const oldUptime = (first && typeof first.uptime_seconds === 'number')
        ? first.uptime_seconds : null;
    const deadline = Date.now() + 90000;
    let sawDown = false;
    const poll = async () => {
        const h = await get();
        if (h === null) {
            sawDown = true;
        } else if (h.status === 'healthy') {
            const restarted =
                (oldUptime !== null && typeof h.uptime_seconds === 'number'
                    && h.uptime_seconds < oldUptime)
                || sawDown;
            if (restarted) { window.location.reload(); return; }
        }
        if (Date.now() < deadline) { setTimeout(poll, 1000); }
    };
    setTimeout(poll, 2000);
}
"""


# ---------------------------------------------------------------------------
# What runs and what is saved
# ---------------------------------------------------------------------------
def _saved(settings: Mapping[str, Any]) -> tuple:
    """The saved allocator preset (`resolve_allocator`: an unknown or missing
    name gives the default) and torch.compile's saved value."""
    preset, _conf = resolve_allocator(settings.get("allocator_preset"))
    return preset, bool(settings.get("torch_compile", False))


def running_html(env: Mapping[str, str]) -> str:
    """The allocator this backend runs now (`running_allocator`), in words,
    with where it came from; an explicit setting in the launch environment
    with its variable and value, escaped."""
    run = running_allocator(env)
    if run["source"] is None:
        return ("PyTorch’s default (native) "
                '<span class="shep-muted">(no allocator was set when this backend started)</span>')
    said = r.e(allocator_words(run["preset"])) if run["preset"] else "A value no preset gives"
    if run["source"] == "preset":
        return f'{said} <span class="shep-muted">(the saved preset when this backend started)</span>'
    return (f'{said} <span class="shep-muted">(set in the launch environment:</span> '
            f'<code>{r.e(run["variable"])}={r.e(run["conf"])}</code><span class="shep-muted">)'
            "</span>")


def restart_applies(env: Mapping[str, str], settings: Mapping[str, Any]) -> bool:
    """**Whether Restart Backend would change what runs** (R-12): the saved
    preset is not the one this backend runs (`running_allocator`), and no
    explicit setting in the launch environment, which a restart keeps,
    overrides it. The one rule the facts' "Not running yet" and Restart's
    state read."""
    preset, _compile = _saved(settings)
    run = running_allocator(env)
    return run["source"] != "env" and run["preset"] != preset


def saved_html(env: Mapping[str, str], settings: Mapping[str, Any]) -> str:
    """The saved allocator preset and when it applies: said apart from what
    runs; a saved name that is no preset said with the default that applies;
    a preset the backend does not run yet with what applies it; one an
    explicit setting overrides with that."""
    requested = settings.get("allocator_preset")
    preset, _compile = _saved(settings)
    said = r.e(allocator_words(preset))
    if requested is None:
        said += ' <span class="shep-muted">(the default; none saved)</span>'
    elif requested != preset:
        said += (f' <span class="shep-muted">(the saved name {r.ident_html(requested)} is no '
                 "preset)</span>")
    if running_allocator(env)["source"] == "env":
        said += "<br>" + r.status_mark("warn", "The launch environment’s setting overrides it "
                                               "until the backend starts without it")
    elif restart_applies(env, settings):
        said += "<br>" + r.status_mark("warn", "Not running yet: Restart Backend applies it")
    return said


def facts_html(env: Optional[Mapping[str, str]] = None,
               settings: Optional[Mapping[str, Any]] = None) -> str:
    """The panel's facts (the module docstring), in the one summary list:
    the allocator running now against the one saved, torch.compile's saved
    value (where it is read is the field's line, `FIELD_INFO`), and the
    acceleration area."""
    env = os.environ if env is None else env
    settings = load_runtime_settings() if settings is None else settings
    _preset, compile_on = _saved(settings)
    return r.summary_list([
        ("Allocator now", running_html(env)),
        ("Allocator saved", saved_html(env, settings)),
        ("torch.compile saved", r.OFF if not compile_on else "On"),
        ("Acceleration", r.status_mark("neutral", "Not implemented yet")
         + f' <span class="shep-muted">({r.e(ACCELERATION)})</span>'),
    ])


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
#: The panel's question row's parts (`_ask`): the row, its question, its two
#: confirming buttons (one shown, the act asked for) and the status line.
RESTART = "restart"
STOP = "stop"


def _question_html(asked: Optional[str]) -> str:
    """The question row's question as drawn (`_ask`): `asked`'s words, or
    nothing. What the row shows is what an open Restart question is known by
    (`_restart_question_closed`)."""
    question = {RESTART: RESTART_QUESTION, STOP: STOP_QUESTION}.get(asked or "", "")
    return f'<p class="shep-sub">{r.e(question)}</p>' if question else ""


def _ask(asked: Optional[str] = None, said: str = "") -> tuple:
    """**The one question row** of Restart Backend and Stop service, as its
    outputs: shown with `asked`'s question and its own confirming button
    (`RESTART`, `STOP`), or hidden (None) with `said` in the status line."""
    return (gr.update(visible=asked is not None), gr.update(value=_question_html(asked)),
            gr.update(visible=asked == RESTART), gr.update(visible=asked == STOP),
            said)


def _restart_refusal(access: Any) -> Optional[str]:
    """**Why Restart Backend may not run now** (R-12), as the status line's
    mark, or None when it may: D8's answer for the event's own request
    (`access`, an operator control) first; off while a GPU job runs (the
    backend refuses then too, `RESTART_BUSY`) and while a restart would
    change nothing (`restart_applies`, `RESTART_IDLE`). The one check the
    question and the confirmed restart both make, so a question left open
    never restarts after the saved allocator, a job or D8 has turned Restart
    off (T06 review: "Restart now" re-exec'd the backend with the running
    allocator saved again)."""
    if not access.allowed:
        return r.status_mark("error", access.refusal)
    if is_training_active():
        return r.status_mark("neutral", RESTART_BUSY)
    if not restart_applies(os.environ, load_runtime_settings()):
        return r.status_mark("neutral", RESTART_IDLE)
    return None


def _on_restart_request(request: gr.Request):
    """Restart Backend, asked once (R-12): the question shown only while a
    restart may run (`_restart_refusal`), its reason in the status line
    otherwise. Returns `_ask`'s outputs."""
    refusal = _restart_refusal(operator_access(request))
    return _ask(said=refusal) if refusal else _ask(RESTART)


def _restart_question_closed(question: Any, request: gr.Request) -> tuple:
    """The question row once Restart has turned off (R-12's residual): a
    Restart question left open (`question`, the row's question as drawn) is
    closed with the reason, by the one check the question and "Restart now"
    make (`_restart_refusal`), when Apply saved the running allocator again or
    the slow timer finds Restart off (another tab's Apply, a GPU job); a Stop
    question, a closed row and a Restart still wanted are left as they are.
    Returns `_ask`'s outputs."""
    if question != _question_html(RESTART):
        return (gr.update(),) * 5
    refusal = _restart_refusal(operator_access(request))
    return _ask(said=refusal) if refusal else (gr.update(),) * 5


def _on_restart(request: gr.Request):
    """The confirmed Restart Backend: the question's check made again on
    this event's own request (`_restart_refusal`) before the backend is
    asked for anything, so the row closes with the reason instead when
    Restart has turned off since it was asked. The backend's own guard
    refuses while a GPU job runs. Returns `_ask`'s outputs, the row closed."""
    refusal = _restart_refusal(operator_access(request))
    if refusal:
        return _ask(said=refusal)
    result = restart_backend()
    if not result.get("success"):
        return _ask(said=r.status_mark(
            "error", f"Not restarted: {result.get('error') or 'cannot restart right now.'}"))
    return _ask(said=r.status_mark("running", "Restarting the backend…")
                + f" {RESTART_STOPS_SERVING} This page reloads by itself once the "
                  "backend is back (a few seconds).")


def _on_apply(alloc, compile_on, request: gr.Request):
    """Apply Settings: an operator control (D8), checked on this event's own
    request before the settings file is read or written. Returns what it did
    and the panel's facts read again."""
    access = operator_access(request)
    if not access.allowed:
        return r.status_mark("error", access.refusal), gr.update()
    data = load_runtime_settings()
    data["allocator_preset"] = alloc
    data["torch_compile"] = bool(compile_on)
    save_runtime_settings(data)
    said = f"Saved: {allocator_words(str(alloc))}; torch.compile {'on' if compile_on else 'off'}."
    return f'<span class="shep-muted">{r.e(said)}</span>', facts_html(settings=data)


def _on_change(alloc, compile_on):
    """The fields against the saved values: "Not saved" while they differ,
    nothing when they are the saved ones again."""
    preset, saved_compile = _saved(load_runtime_settings())
    if alloc == preset and bool(compile_on) == saved_compile:
        return ""
    return r.status_mark("warn", "Not saved: Apply Settings saves the fields above")


def _on_stop_service_request(request: gr.Request):
    """Stop service, asked once: an operator control (D8), checked on this
    event's own request before the question is shown; refused while a GPU
    job runs (the backend refuses then too). Returns `_ask`'s outputs."""
    access = operator_access(request)
    if not access.allowed:
        return _ask(said=r.status_mark("error", access.refusal))
    if is_training_active():
        return _ask(said=r.status_mark("neutral", RESTART_BUSY))
    return _ask(STOP)


def _on_stop_service(request: gr.Request):
    """The confirmed Stop service: an operator control (D8), checked on this
    event's own request before the backend is asked; the backend refuses
    while a GPU job or a model load runs (`stop_backend`) and logs who asked.
    Returns `_ask`'s outputs, the row closed."""
    access = operator_access(request)
    if not access.allowed:
        return _ask(said=r.status_mark("error", access.refusal))
    result = stop_backend()
    if not result.get("success"):
        return _ask(said=r.status_mark(
            "error", f"Not stopped: {result.get('error') or 'cannot stop right now.'}"))
    return _ask(said=r.status_mark("stopping", STOP_DONE))


def _on_confirm_cancel():
    """Cancel: the question closes; nothing was asked of the backend."""
    return _ask()


#: The operator controls' words, shown with no lock while D8 allows the
#: request (`chips.gated`).
APPLY_LABEL = "Apply Settings"
RESTART_LABEL = "Restart Backend"


def _refresh_restart_lock(request: gr.Request):
    """Apply, Restart and Stop service as this request may use them, by the
    one gated-button rule (`chips.gated`): for a refused request, all
    disabled with the lock and D8's short reason beside them
    (`chips.operator_only`; the header chip's ⓘ gives D8's whole reason);
    Restart and Stop service also off while a GPU job runs (`RESTART_BUSY`),
    and Restart while a restart would change nothing (`restart_applies`,
    `RESTART_IDLE`), the reason beside them. A display: each button's handler
    checks its own request."""
    locked = is_training_active()
    access = operator_access(request)
    idle = not restart_applies(os.environ, load_runtime_settings())
    reason = ("" if access.allowed else
              f'<div class="shep-reason"><span aria-hidden="true">{LOCK}</span>'
              f'<span>{r.e(operator_only(access.condition))}.</span></div>')
    why = RESTART_BUSY if locked else RESTART_IDLE if idle and access.allowed else None
    return (
        gr.update(**gated(APPLY_LABEL, access.allowed)),
        gr.update(**gated(RESTART_LABEL, access.allowed, interactive=not (locked or idle))),
        gr.update(visible=why is not None, value=(f'<div class="shep-reason">'
                                                  f'{r.status_mark("neutral", why)}</div>'
                                                  if why else "")),
        gr.update(visible=not access.allowed, value=reason),
        gr.update(**gated(STOP_LABEL, access.allowed, interactive=not locked)),
    )


def _on_open(request: gr.Request):
    """The page as it is opened: the buttons and their reasons
    (`_refresh_restart_lock`), the two fields that need the operator
    interactive only for a request D8 allows, and the fields and the facts
    from one read of the saved settings, so a field never shows a value the
    facts call saved apart from it (B5 review: a field kept the value read
    when the app was built, and Apply wrote it back over a newer save)."""
    allowed = operator_access(request).allowed
    settings = load_runtime_settings()
    preset, compile_on = _saved(settings)
    return (*_refresh_restart_lock(request), gr.update(interactive=allowed, value=preset),
            gr.update(interactive=allowed, value=compile_on), facts_html(settings=settings))


def _poll_restart_lock(location, question, request: gr.Request):
    """The shell's slow timer (``queue=False``): the buttons and their
    reasons read again only while Settings is shown, and a Restart question
    left open closed once Restart is off (`_restart_question_closed`); on
    any other page, no work and no change (M5-3 review ARCH-8: the page kept
    a queued 4 s timer of its own that ticked on every page once Settings
    had been opened). The fields are left as they are, so an open select is
    never redrawn."""
    if location != SETTINGS_PAGE:
        return (gr.update(),) * 10
    return (*_refresh_restart_lock(request), *_restart_question_closed(question, request))


def _after_apply(question, request: gr.Request):
    """Apply's last step: the buttons read again (Apply may make a restart
    wanted, or not) and a Restart question left open closed once it is off
    (`_restart_question_closed`)."""
    return (*_refresh_restart_lock(request), *_restart_question_closed(question, request))


#: When each field applies, visible under it (its ``info=``: a fact that
#: changes what happens, never only behind the ⓘ; the WebUI skill's §7).
FIELD_INFO = {
    "allocator_preset": "Read once when the backend starts: Restart Backend applies a saved "
                        "change.",
    "torch_compile": f"{COMPILE_EXPERIMENTAL} {COMPILE_READ[:1].upper()}{COMPILE_READ[1:]}.",
}

#: The fields' help, behind the shared ⓘ (M5-2): ``{elem_id: (title, text)}``.
FIELD_HELP = {
    "allocator_preset": (
        "Memory allocator",
        "How CUDA memory is pooled: it governs fragmentation and peak use.\n\n"
        "- The backend runs CUDA in its own process too (diagnosis), so it reads the allocator "
        "once at start-up: a saved change applies after Restart Backend.\n"
        "- An allocator set in the launch environment overrides the saved one.\n"
        "- A CPU run does not use it.",
    ),
    "torch_compile": (
        "Compile the model (torch.compile)",
        "Compiles supported model regions to cut overhead.\n\n"
        "- PyG workloads may graph-break, fall back to eager or run slower; GATv2, the one "
        "architecture built today, has not been measured with it.\n"
        "- It needs no restart.",
    ),
}

#: What each allocator preset does (static text, behind its accordion).
ALLOCATOR_DETAILS = (
    "- **Driver memory pool** (`cuda_async`, the default): NVIDIA's stream-ordered pool, "
    "driver-managed with no tuning; it may change with driver upgrades, and PyTorch's memory "
    "snapshot statistics are reduced under it.\n"
    "- **Expandable segments** (`expandable`): PyTorch's growable segments. They bound "
    "fragmentation for variable-size subgraphs and keep PyTorch's memory statistics.\n"
    "- **Native, rounded** (`native_roundup`): the native allocator with power-of-2 size "
    "rounding, to cut fragmentation at low cost.\n"
    "- **Native, no tuning** (`native`): can fragment badly on variable tensor sizes.\n\n"
    "Measured earlier on this project with an architecture no longer built (HGT, batch 256): "
    "the driver pool and expandable segments both used about 26 GB at a comparable speed; "
    "native fragmented (about 60 → 120 GB). Not yet measured with GATv2."
)
#: torch.compile's caveat (static text, behind its accordion; its lead,
#: `COMPILE_EXPERIMENTAL`, is the field's visible line).
COMPILE_DETAILS = (
    "Not yet measured with "
    "GATv2, the one architecture built today. On GB10 (DGX Spark) an earlier architecture "
    "(HGT) ran slower with it, and a known sm_121 Triton issue makes it fall back to eager. "
    "Compare MRR and Hits@K with an eager run before trusting a result."
)


def create_runtime_settings_tab(ctx: Any) -> None:
    """Build the Runtime Settings page under ⚙ Settings (call inside its gr.Tab).

    ``ctx`` is the page context (`src/webui/shell.py`). D8's answer, the
    saved fields, the facts and the restart lock are read each time the page
    is shown
    (``ctx.on_page_open``), and the buttons' lock on the shell's slow timer
    while it is shown (`_poll_restart_lock`, ``queue=False``, no work on
    another page); the page has no timer of its own. The backend refuses a
    restart while any GPU job runs either way (``restart_backend``); this
    keeps the button and its reason honest. The fields' help is behind the
    shared ⓘ (`FIELD_HELP`); Apply and Restart refresh the header's chips.
    The controls start disabled, as each page's gated controls do, until the
    page's open reads D8's answer for this request.
    """
    ctx.help.fields(FIELD_HELP)
    settings = load_runtime_settings()
    saved_alloc, saved_compile = _saved(settings)

    with gr.Row(elem_id="shep-settings-cols"):
        # Data locations (UI-06) on the left, Runtime settings on the right;
        # narrow, one above the other.
        data_locations.create_data_locations_panel(ctx)
        controls = _runtime_panel(ctx, settings, saved_alloc, saved_compile)
    _wire_runtime_panel(ctx, *controls)


def _runtime_panel(ctx: Any, settings: Mapping[str, Any], saved_alloc: str,
                   saved_compile: bool) -> tuple:
    """The Runtime settings panel's components (`create_runtime_settings_tab`)."""
    with gr.Column(elem_classes=["shep-panel", "shep-settings"], elem_id="shep-settings"):
        plain_html('<div class="shep-panel-hd"><h2>Runtime settings</h2>'
                   '<span class="shep-panel-right">this server’s hardware</span></div>'
                   '<p class="shep-intro">How computation runs on this server: memory and '
                   "compilation. Research › Training sets the model.</p>",
                   elem_id="shep-settings-hd")
        with gr.Column(elem_id="shep-settings-bd"):
            facts = plain_html(facts_html(settings=settings), elem_id="runtime_facts")
            allocator = gr.Dropdown(
                label="Memory allocator", choices=ALLOCATOR_CHOICES, value=saved_alloc,
                interactive=False, elem_id="allocator_preset",
                info=FIELD_INFO["allocator_preset"])
            with gr.Accordion("What each allocator does", open=False):
                gr.Markdown(ALLOCATOR_DETAILS)
            torch_compile = gr.Checkbox(
                label="Compile the model (torch.compile)", value=saved_compile,
                interactive=False, elem_id="torch_compile", info=FIELD_INFO["torch_compile"])
            with gr.Accordion("About torch.compile", open=False):
                gr.Markdown(COMPILE_DETAILS)
        with gr.Row(elem_classes=["shep-panel-ft"], elem_id="shep-settings-ft"):
            apply_btn = gr.Button(APPLY_LABEL, variant="primary", interactive=False, scale=0,
                                  elem_id="runtime_apply")
            restart_btn = gr.Button(RESTART_LABEL, variant="secondary", interactive=False,
                                    scale=0, elem_id="runtime_restart")
            stop_btn = gr.Button(STOP_LABEL, variant="secondary", interactive=False, scale=0,
                                 elem_id="runtime_stop")
            # Why Restart and Stop are off (a GPU job runs) and why all are
            # (D8), beside them.
            restart_lock_note = plain_html("", visible=False, elem_id="runtime_restart_lock")
            access_note = plain_html("", visible=False, elem_id="runtime_access_reason")
        # The one question row of Restart Backend and Stop service (`_ask`):
        # the question asked, its own confirming button, Cancel.
        with gr.Row(visible=False, elem_classes=["shep-panel-ft"],
                    elem_id="runtime_confirm") as confirm:
            question = plain_html("", elem_id="runtime_question")
            restart_now = gr.Button(RESTART_CONFIRM_LABEL, variant="primary", scale=0,
                                    visible=False, elem_id="runtime_restart_now")
            stop_now = gr.Button(STOP_CONFIRM_LABEL, variant="primary", scale=0,
                                 visible=False, elem_id="runtime_stop_now")
            cancel = gr.Button("Cancel", variant="secondary", scale=0,
                               elem_id="runtime_confirm_cancel")
        with gr.Column(elem_id="shep-settings-out"):
            status = plain_html("", elem_id="runtime_status")
            restart_status = plain_html("", elem_id="runtime_restart_status")
    return (facts, allocator, torch_compile, apply_btn, restart_btn, stop_btn,
            restart_lock_note, access_note, (confirm, question, restart_now, stop_now), cancel,
            status, restart_status)


def _wire_runtime_panel(ctx: Any, facts, allocator, torch_compile, apply_btn, restart_btn,
                        stop_btn, restart_lock_note, access_note, asking, cancel, status,
                        restart_status) -> None:
    """The Runtime settings panel's events (`create_runtime_settings_tab`)."""
    confirm, question, restart_now, stop_now = asking
    asked = [confirm, question, restart_now, stop_now, restart_status]
    lock_outputs = [apply_btn, restart_btn, restart_lock_note, access_note, stop_btn]
    ctx.timer_slow.tick(fn=_poll_restart_lock, inputs=[ctx.location, question],
                        outputs=[*lock_outputs, *asked], queue=False, show_progress="hidden")
    ctx.on_page_open(SETTINGS_PAGE, {"fn": _on_open,
                                     "outputs": [*lock_outputs, allocator, torch_compile, facts],
                                     "queue": False, "show_progress": "hidden"})
    # Restart Backend asks first; after the confirmed restart, a browser-side
    # poller reloads the page once the fresh backend is up (a new process means
    # a new Gradio session; without a reload the old page's pollers hang). The
    # poller no-ops if the restart was refused.
    restart_btn.click(fn=_on_restart_request, outputs=asked, show_progress="hidden")
    restart_now.click(fn=_on_restart, outputs=asked, show_progress="hidden").then(
        **ctx.chip_refresh).then(fn=None, inputs=None, outputs=None, js=_RESTART_RELOAD_JS)
    for field in (allocator, torch_compile):
        field.change(fn=_on_change, inputs=[allocator, torch_compile], outputs=[status],
                     show_progress="hidden")
    # Apply may make a restart wanted (or not): the buttons are read again,
    # and a Restart question left open closes once Restart is off.
    apply_btn.click(fn=_on_apply, inputs=[allocator, torch_compile],
                    outputs=[status, facts]).then(**ctx.chip_refresh).then(
        fn=_after_apply, inputs=[question], outputs=[*lock_outputs, *asked], queue=False,
        show_progress="hidden")
    stop_btn.click(fn=_on_stop_service_request, outputs=asked, show_progress="hidden")
    stop_now.click(fn=_on_stop_service, outputs=asked, show_progress="hidden")
    cancel.click(fn=_on_confirm_cancel, outputs=asked, show_progress="hidden")


_PAGE = "#" + tab_elem_id(SETTINGS_PAGE)

#: This page's style sheet (the page contract: listed in `src/webui/app.py`'s
#: ``PAGE_CSS``): one panel at a readable width, its body padded, its
#: actions one row with their reasons beside them.
PAGE_CSS = f"""
{_PAGE} #shep-settings-cols {{ gap: 16px !important; align-items: flex-start; flex-wrap: wrap; }}
{_PAGE} #shep-settings-cols > #shep-dl {{ flex: 55 1 640px !important; min-width: 0 !important; }}
{_PAGE} #shep-settings-cols > .shep-settings {{ flex: 45 1 480px !important; min-width: 0 !important; }}
{_PAGE} .shep-settings {{ max-width: 1040px; }}
{_PAGE} #shep-settings-bd {{ padding: 14px 16px !important; gap: 12px !important; }}
{_PAGE} #shep-settings-ft > button {{ flex: none !important; min-width: max-content !important; white-space: nowrap; }}
{_PAGE} #shep-settings-ft > .shep-html {{ flex: 1 1 240px !important; min-width: 0 !important; }}
{_PAGE} #shep-settings-out {{ padding: 0 16px 12px !important; gap: 6px !important; }}
{_PAGE} #runtime_status:has(.html-container:empty),
{_PAGE} #runtime_restart_status:has(.html-container:empty) {{ display: none !important; }}
{_PAGE} #runtime_confirm > .shep-html {{ flex: 1 1 240px !important; min-width: 0 !important; }}
{_PAGE} #runtime_confirm > button {{ flex: none !important; min-width: max-content !important; }}
{_PAGE} #runtime_confirm p {{ margin: 0; }}
""" + data_locations.PAGE_CSS


__all__ = ["ACCELERATION", "ALLOCATOR_CHOICES", "ALLOCATOR_WORDS", "COMPILE_EXPERIMENTAL",
           "COMPILE_READ", "FIELD_HELP", "FIELD_INFO", "PAGE_CSS", "RESTART_BUSY",
           "RESTART_CONFIRM_LABEL", "RESTART_IDLE", "RESTART_QUESTION", "RESTART_STOPS_SERVING",
           "STOP_CONFIRM_LABEL", "STOP_DONE", "STOP_LABEL", "STOP_QUESTION", "allocator_words",
           "create_runtime_settings_tab", "facts_html", "restart_applies", "running_html",
           "saved_html"]
