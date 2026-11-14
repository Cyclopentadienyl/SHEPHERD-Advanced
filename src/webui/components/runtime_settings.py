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
  said apart from the preset saved for the next start; torch.compile's saved
  value is said with where it is read (`COMPILE_READ`). An explicit setting in
  the launch environment is named as one, never a preset the launcher
  applied.
- **Fields**: the memory allocator, read once when the backend starts (a saved
  change applies after Restart Backend), and torch.compile, read when the
  Training page starts, continues or branches a run (Fine-tune does not read
  it). Each is shown as saved each time the page is opened, from the same read
  as the facts. Their words say what is true today: GATv2 is the one
  architecture built; a measurement made on an earlier architecture is named
  as such.
- **Acceleration**: no toggle; the area says "Not implemented yet"
  (`ACCELERATION`): new acceleration work is deferred, and each run's record
  keeps what it ran with.
- **Actions**: Apply Settings saves the fields to
  ``.shepherd_runtime_settings.json`` (`src.config.runtime_presets`), which the
  launcher reads at start-up (`scripts/launch/shep_launch.py`) and Restart
  Backend applies again; Restart Backend relaunches the backend.

Operator controls (D8, M5-1):
    Restart Backend and Apply Settings each check their own Gradio request
    first (`src/webui/operator_access.py`) and are refused, changing nothing,
    unless D8 allows it. The page follows the one gated-button rule
    (`chips.gated`, the WebUI skill's §3) when it is opened, and the buttons
    on the shell's slow timer: no lock while D8 allows the request; for a
    request D8 refuses, both buttons disabled with the lock, the two fields
    that need the operator (the allocator and torch.compile) not interactive,
    and the visible "Operator controls only: …" reason beside the buttons
    (`chips.operator_only`). Restart is also off while a GPU job runs, said
    beside it. A disabled control is a display, never the control.

Module: src/webui/components/runtime_settings.py
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Optional

import gradio as gr

from src.api.services.backend_control import is_training_active, restart_backend
from src.config.runtime_presets import (
    ALLOCATOR_PRESETS,
    load_runtime_settings,
    resolve_allocator,
    running_allocator,
    save_runtime_settings,
)
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

#: What a restart does to inference (the ops round's I7), said beside the
#: button and when it starts: the backend stops serving before it replaces its
#: process, and the new one loads only a model its start-up configures.
RESTART_STOPS_SERVING = (
    "A loaded model is stopped first (the diagnoses already accepted finish) and is not "
    "loaded again unless the service's start-up configures one (SHEPHERD_CHECKPOINT_PATH)."
)

#: Why Restart is off while a GPU job runs (the backend refuses it too: a
#: re-exec would orphan the job's process), said beside the button.
RESTART_BUSY = "Restart is off while a GPU job (training or a test run) runs."

#: The acceleration area: a fact, never a toggle that does nothing; said in
#: brackets after its "Not implemented yet" mark.
ACCELERATION = ("new acceleration work is deferred; each run’s record keeps the kernels and "
                "precision it ran with")

#: Where torch.compile's saved value is read: the Training page's base Start
#: and its Continue or Branch (`training_console`), the only readers; the
#: Fine-tune page does not read it. Said in the facts, under the field, in its
#: help and after Apply, in these words.
COMPILE_READ = ("read when the Training page starts, continues or branches a run; Fine-tune "
                "does not read it")

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
    run = running_allocator(env)
    if run["source"] == "env":
        said += "<br>" + r.status_mark("warn", "The launch environment’s setting overrides it "
                                               "until the backend starts without it")
    elif run["preset"] != preset:
        said += "<br>" + r.status_mark("warn", "Not running yet: Restart Backend applies it")
    return said


def facts_html(env: Optional[Mapping[str, str]] = None,
               settings: Optional[Mapping[str, Any]] = None) -> str:
    """The panel's facts (the module docstring), in the one summary list:
    the allocator running now against the one saved, torch.compile's saved
    value and where it is read (`COMPILE_READ`), and the acceleration area."""
    env = os.environ if env is None else env
    settings = load_runtime_settings() if settings is None else settings
    _preset, compile_on = _saved(settings)
    return r.summary_list([
        ("Allocator now", running_html(env)),
        ("Allocator saved", saved_html(env, settings)),
        ("torch.compile saved", (r.OFF if not compile_on else "On")
         + f' <span class="shep-muted">({r.e(COMPILE_READ)})</span>'),
        ("Acceleration", r.status_mark("neutral", "Not implemented yet")
         + f' <span class="shep-muted">({r.e(ACCELERATION)})</span>'),
    ])


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def _on_restart(request: gr.Request):
    """Restart Backend: an operator control (D8), checked on this event's own
    request before the backend is asked for anything. The backend's own guard
    refuses while a GPU job runs."""
    access = operator_access(request)
    if not access.allowed:
        return gr.update(value=r.status_mark("error", access.refusal))
    result = restart_backend()
    if not result.get("success"):
        return gr.update(value=r.status_mark(
            "error", f"Not restarted: {result.get('error') or 'cannot restart right now.'}"))
    return gr.update(value=r.status_mark("running", "Restarting the backend…")
                     + f" {RESTART_STOPS_SERVING} This page reloads by itself once the backend "
                       "is back (a few seconds).")


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
    said = (f"Saved: {allocator_words(str(alloc))}; torch.compile "
            f"{'on' if compile_on else 'off'}, {COMPILE_READ}.")
    return f'<span class="shep-muted">{r.e(said)}</span>', facts_html(settings=data)


def _on_change(alloc, compile_on):
    """The fields against the saved values: "Not saved" while they differ,
    nothing when they are the saved ones again."""
    preset, saved_compile = _saved(load_runtime_settings())
    if alloc == preset and bool(compile_on) == saved_compile:
        return ""
    return r.status_mark("warn", "Not saved: Apply Settings saves the fields above")


#: The two operator controls' words, shown with no lock while D8 allows the
#: request (`chips.gated`).
APPLY_LABEL = "Apply Settings"
RESTART_LABEL = "Restart Backend"


def _refresh_restart_lock(request: gr.Request):
    """Apply and Restart as this request may use them, by the one
    gated-button rule (`chips.gated`): for a refused request, both disabled
    with the lock and D8's short reason beside them (`chips.operator_only`;
    the header chip's ⓘ gives D8's whole reason); Restart also off while a
    GPU job runs, said beside it (`RESTART_BUSY`). A display: each button's
    handler checks its own request."""
    locked = is_training_active()
    access = operator_access(request)
    reason = ("" if access.allowed else
              f'<div class="shep-reason"><span aria-hidden="true">{LOCK}</span>'
              f'<span>{r.e(operator_only(access.condition))}.</span></div>')
    return (
        gr.update(**gated(APPLY_LABEL, access.allowed)),
        gr.update(**gated(RESTART_LABEL, access.allowed, interactive=not locked)),
        gr.update(visible=locked, value=(f'<div class="shep-reason">'
                                         f'{r.status_mark("neutral", RESTART_BUSY)}</div>'
                                         if locked else "")),
        gr.update(visible=not access.allowed, value=reason),
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


def _poll_restart_lock(location, request: gr.Request):
    """The shell's slow timer (``queue=False``): the buttons and their
    reasons read again only while Settings is shown; on any other page, no
    work and no change (M5-3 review ARCH-8: the page kept a queued 4 s timer
    of its own that ticked on every page once Settings had been opened). The
    fields are left as they are, so an open select is never redrawn."""
    if location != SETTINGS_PAGE:
        return (gr.update(),) * 4
    return _refresh_restart_lock(request)


#: When each field applies, visible under it (its ``info=``: a fact that
#: changes what happens, never only behind the ⓘ; the WebUI skill's §7).
FIELD_INFO = {
    "allocator_preset": "Read once when the backend starts: Restart Backend applies a saved "
                        "change.",
    "torch_compile": COMPILE_READ[:1].upper() + COMPILE_READ[1:] + ". Experimental.",
}

#: The fields' help, behind the shared ⓘ (M5-2): ``{elem_id: (title, text)}``.
FIELD_HELP = {
    "allocator_preset": (
        "Memory allocator",
        "How CUDA memory is pooled: it governs fragmentation and peak use. The backend runs "
        "CUDA in its own process too (diagnosis), so it reads the allocator once at start-up: "
        "a saved change applies after Restart Backend. An allocator set in the launch "
        "environment overrides the saved one. A CPU run does not use it.",
    ),
    "torch_compile": (
        "Compile the model (torch.compile)",
        "Compiles supported model regions to cut overhead. PyG workloads may graph-break, fall "
        "back to eager or run slower; GATv2, the one architecture built today, has not been "
        "measured with it. It needs no restart: it is " + COMPILE_READ + ".",
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
#: torch.compile's caveat (static text, behind its accordion).
COMPILE_DETAILS = (
    "**Experimental: keep it off unless you are benchmarking it.** Not yet measured with "
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
            # Why Restart is off (a GPU job runs) and why both are (D8),
            # beside them.
            restart_lock_note = plain_html("", visible=False, elem_id="runtime_restart_lock")
            access_note = plain_html("", visible=False, elem_id="runtime_access_reason")
        with gr.Column(elem_id="shep-settings-out"):
            status = plain_html("", elem_id="runtime_status")
            restart_status = plain_html("", elem_id="runtime_restart_status")
            plain_html(f'<p class="shep-note">Restart Backend relaunches the backend (the REST '
                       f"API and this page) so a newly saved memory allocator takes effect. "
                       f"{RESTART_STOPS_SERVING} The page reconnects by itself a few seconds "
                       "after a restart.</p>", elem_id="runtime_restart_note")

    lock_outputs = [apply_btn, restart_btn, restart_lock_note, access_note]
    ctx.timer_slow.tick(fn=_poll_restart_lock, inputs=[ctx.location], outputs=lock_outputs,
                        queue=False, show_progress="hidden")
    ctx.on_page_open(SETTINGS_PAGE, {"fn": _on_open,
                                     "outputs": [*lock_outputs, allocator, torch_compile, facts],
                                     "queue": False, "show_progress": "hidden"})
    # After the click handler runs, kick off a browser-side poller that reloads
    # the page once the fresh backend is up (a new process means a new Gradio
    # session; without a reload the old page's pollers hang). The poller no-ops
    # if the restart was refused.
    restart_btn.click(fn=_on_restart, outputs=[restart_status]).then(
        **ctx.chip_refresh).then(fn=None, inputs=None, outputs=None, js=_RESTART_RELOAD_JS)
    for field in (allocator, torch_compile):
        field.change(fn=_on_change, inputs=[allocator, torch_compile], outputs=[status],
                     show_progress="hidden")
    apply_btn.click(fn=_on_apply, inputs=[allocator, torch_compile],
                    outputs=[status, facts]).then(**ctx.chip_refresh)


_PAGE = "#" + tab_elem_id(SETTINGS_PAGE)

#: This page's style sheet (the page contract: listed in `src/webui/app.py`'s
#: ``PAGE_CSS``): one panel at a readable width, its body padded, its
#: actions one row with their reasons beside them.
PAGE_CSS = f"""
{_PAGE} .shep-settings {{ max-width: 1040px; }}
{_PAGE} #shep-settings-bd {{ padding: 14px 16px !important; gap: 12px !important; }}
{_PAGE} #shep-settings-ft > button {{ flex: none !important; min-width: max-content !important; white-space: nowrap; }}
{_PAGE} #shep-settings-ft > .shep-html {{ flex: 1 1 240px !important; min-width: 0 !important; }}
{_PAGE} #shep-settings-out {{ padding: 0 16px 12px !important; gap: 6px !important; }}
{_PAGE} #shep-settings-out .shep-note {{ margin: 0; }}
{_PAGE} #runtime_status:has(.html-container:empty),
{_PAGE} #runtime_restart_status:has(.html-container:empty) {{ display: none !important; }}
"""


__all__ = ["ACCELERATION", "ALLOCATOR_CHOICES", "ALLOCATOR_WORDS", "COMPILE_READ", "FIELD_HELP",
           "FIELD_INFO",
           "PAGE_CSS",
           "RESTART_BUSY", "RESTART_STOPS_SERVING", "allocator_words", "create_runtime_settings_tab",
           "facts_html", "running_html", "saved_html"]
