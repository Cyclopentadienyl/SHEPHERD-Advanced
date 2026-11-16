"""
The header's two chips: operator controls (D8) and GPU jobs (M5-2, UI-A).
=========================================================================
Both are read per request, never cached between sessions: on page load, on the
shell's timer (outside ``gr.Tabs``, ``queue=False``) and after every gated act
(`src/webui/shell.py`, `PageContext.chip_refresh`).

**The D8 chip** says what D8's one check (`src/utils/operator_access.py`) answers
for the request that asked: "Operator controls: on (this machine)", or grey
"🔒 Read-only: <the failed condition in short words>" (the lock, which means
refused, leads only the read-only chip). Its ⓘ lists every operator
control and how to reach them, and, when refused, the check's own reason (which
repeats no header value). It is a display: each control checks its own request.

**Gated buttons** follow one rule (`gated`; the WebUI skill's §3, a lock means
refused): no lock while D8 allows the request; disabled, with the lock and
the visible "Operator controls only: …" reason beside it (`operator_only`),
while D8 refuses it.

**The GPU chip** says what holds the one GPU slot, from the job runner
(`training_manager.active_job`) and, for a holder outside the runner, the GPU
lock's descriptive holder record (`src/utils/gpu_lock.py`): "GPU jobs: none",
"GPU job: training (job runner)", "GPU job: test run (job runner)", "GPU:
inference (this service)" (this process's CUDA inference, which holds the lock
while a model is loaded on CUDA, the ops round's I2; with where it is stopped),
"GPU: inference (pid …, since …)" (another process's), "GPU job:
train_model.py (pid …, since …)" (what the holder record says: its command, pid
and start; never "CLI", which the record does not say), or grey "GPU jobs: not
read". It never says "free": the lock is cooperative (a GPU process that does
not take it is not stopped by it), and inference on the CPU takes no lock,
which its ⓘ says. The research start controls repeat its words, with where to
stop this service's inference (`busy_words`); the job runner's refusal stays
the authority.

**Nothing here raises and no exception text is shown**: a read that fails is the
grey "not read" state with a fixed sentence. Every value from a file or a
request is escaped. A server path (the lock file's, in a lock problem) is named
only to a request D8 allows.

Module: src/webui/components/chips.py
"""
from __future__ import annotations

import html
import logging
from datetime import datetime
from typing import Any, Callable, Dict, Mapping, NamedTuple, Optional

import gradio as gr

from src.config.server_settings import DATA_LOCATIONS
from src.utils.gpu_lock import SERVING_JOB_TYPE, held_by_inference, lock_state
from src.utils.operator_access import CONDITIONS, OperatorAccess, reason_parts
from src.webui.components.help import REF_PREFIX, help_button

logger = logging.getLogger(__name__)

#: The failed condition in short words, by D8's condition code.
D8_SHORT = {
    "deployment_mode": "deployment mode not set to M1",
    "unreadable_request": "request not readable",
    "not_loopback": "request not local",
    "forwarding_header": "forwarded by a proxy",
    "host_not_loopback": "not addressed to localhost",
    "origin_mismatch": "sent from another origin",
    "cross_site": "sent from another site",
}
assert set(D8_SHORT) == set(CONDITIONS) - {"enabled"}

D8_ON = "Operator controls: on (this machine)"

#: **The lock means "refused"** (the WebUI skill's §3): a gated control shows
#: it only while D8 refuses the request it was drawn for (`gated`).
LOCK = "🔒"
#: The visible reason's lead beside a control D8 refuses (`operator_only`).
OPERATOR_ONLY = "Operator controls only:"


def operator_only(condition: Optional[str]) -> str:
    """**The visible reason beside a gated control D8 refuses** for this
    request, in D8's short words: "Operator controls only: request not
    local" (the header chip's ⓘ gives D8's whole reason). Every page says it
    in these words, beside the control it turns off."""
    return f"{OPERATOR_ONLY} {D8_SHORT.get(condition or '', D8_SHORT['unreadable_request'])}"


def gated(label: str, allowed: bool, *, interactive: bool = True) -> Dict[str, Any]:
    """**The one rule of a gated button** (the WebUI skill's §3), as the
    fields of its ``gr.update``: while D8 allows the request, its words as they
    are, with no lock, usable when `interactive` (the page's own condition: a
    selection, a running job); while D8 refuses it, disabled, with the lock
    before its words, and the page shows `operator_only` beside it. Each
    handler still checks its own request first: a greyed-out button is not a
    control."""
    return {"value": label if allowed else f"{LOCK} {label}",
            "interactive": bool(allowed and interactive)}

#: Every operator control, as the D8 chip's ⓘ lists them (no exception remains).
D8_SCOPE = (
    "- import, test start and cancel, registration, model load, stopping inference;\n"
    "- checkpoint list, training start and stop, pipeline reload, saving settings and data "
    "locations, restart, stopping the service."
)
#: How to reach them; the codes and the SSH command in the trailing Ref line
#: (each D8 chip's ⓘ ends with it, `d8_chip`).
D8_HOW = (
    "They need a service started here (desktop entry or launch_shepherd.sh) and a request "
    "that:\n"
    "- arrives on this machine's loopback, directly or through an SSH forward;\n"
    "- has no forwarding header (no reverse proxy) and comes from this dashboard.\n"
    "Ref: deployment mode M1 (SHEPHERD_DEPLOYMENT_MODE); ssh -L 8264:localhost:8264 <server>, "
    "then http://localhost:8264/ui"
)
#: Each D8 chip's lead (the bubble's first line, `help.help_lines`), which
#: the list of controls (`D8_SCOPE`) follows.
D8_ON_LEAD = "This request may use the operator controls:"
D8_READONLY_LEAD = "This request may not use the operator controls:"

#: What the GPU chip's ⓘ always says, after its state's words (`gpu_detail`).
GPU_SCOPE = (
    "One GPU lock covers:\n"
    "- training and test runs, from the job runner (here or through the API) or the formal "
    "command-line scripts;\n"
    "- this service's inference, which holds it while a model is loaded on CUDA.\n"
    "A second holder is refused, not queued. Inference on the CPU takes no lock."
)
#: The scripts `GPU_SCOPE` names in words, said in its trailing Ref line.
GPU_SCOPE_REF = "the formal scripts train_model.py, measure_scorer.py"
#: The GPU chip's lead when nothing holds the lock (its words: "none", never "free").
GPU_FREE_LEAD = "No managed job holds the GPU lock now."

GPU_NONE = "GPU jobs: none"
#: This service's CUDA inference holding the lock (`gpu_lock.held_by_inference`).
GPU_INFERENCE = "GPU: inference (this service)"
#: Where an operator frees the GPU from this service's inference: said beside
#: every research start it disables (`busy_words`) and in the chip's ⓘ.
STOP_INFERENCE_HERE = "stop it with Stop inference on Diagnosis"
GPU_NOT_READ = "GPU jobs: not read"
GPU_NOT_CONFIGURED = "GPU jobs: refused, no workspaces folder"
#: What the chip's ⓘ says with no workspaces root set (review of 8242b47,
#: P2-4), and its references; it names no server path, so every request may
#: see it.
GPU_NOT_CONFIGURED_DETAIL = (
    "No workspaces folder is set, so the GPU lock is not configured.\n"
    "The job runner refuses every training and test run until one is chosen in "
    f"{DATA_LOCATIONS}."
)
GPU_NOT_CONFIGURED_REF = "SHEPHERD_WORKSPACES_ROOT (the lock: .shepherd-gpu.lock in that folder)"
#: The chip's ⓘ before its first read (`CHIPS_PLACEHOLDER`).
NOT_READ_YET = "Not read yet; the page reads it once it has loaded."
#: Where a request that may not see the lock's problem is told to look.
GPU_LOCK_WHY_HIDDEN = f"An operator sees why in {DATA_LOCATIONS}."
_RUNNER_NOUN = {"training": "training", "test": "test run"}

_LOCK_SVG = ('<svg width="12" height="12" viewBox="0 0 16 16" aria-hidden="true" focusable="false">'
             '<rect x="3" y="7" width="10" height="7.5" rx="1.6" fill="currentColor"/>'
             '<path d="M5.2 7V5.2a2.8 2.8 0 0 1 5.6 0V7" fill="none" stroke="currentColor" '
             'stroke-width="1.6"/></svg>')

#: The ids of the hidden elements holding each chip's ⓘ text.
D8_TEXT_ID = "shep-chip-d8-text"
GPU_TEXT_ID = "shep-chip-gpu-text"

#: How much of a holder record's command the chip repeats.
_COMMAND_LIMIT = 40


class Chip(NamedTuple):
    """One chip: its look (`kind`, a CSS modifier), its words, its ⓘ text and,
    for a holder an operator here can end (this service's inference), where
    to end it (`remedy`, "" otherwise)."""

    kind: str
    text: str
    detail: str
    remedy: str = ""


# ---------------------------------------------------------------------------
# D8
# ---------------------------------------------------------------------------
def d8_chip(access: OperatorAccess) -> Chip:
    """The D8 chip for one decision. Allowed: the controls, then what they
    need (`D8_HOW`). Refused: the controls, then this request's own reason,
    D8's words (also the API's refusal): what failed and what to do, its
    codes in the bubble's one trailing Ref line (`reason_parts`), so the
    general conditions are not said again beside the one that failed."""
    if access.allowed:
        return Chip("on", D8_ON, f"{D8_ON_LEAD}\n{D8_SCOPE}\n\n{D8_HOW}")
    short = D8_SHORT.get(access.condition, "request not readable")
    text, ref = reason_parts(access.reason)
    return Chip("readonly", f"Read-only: {short}",
                f"{D8_READONLY_LEAD}\n{D8_SCOPE}\n\nFor this request: {text}\n"
                f"{REF_PREFIX} {ref or D8_HOW.rpartition(REF_PREFIX)[2].strip()}")


# ---------------------------------------------------------------------------
# GPU jobs
# ---------------------------------------------------------------------------
def gpu_detail(text: str, *refs: str) -> str:
    """**A GPU chip's ⓘ text, the one way every state composes it**: the
    state's own words, then what the lock covers (`GPU_SCOPE`), then one
    trailing Ref line, the state's references before the scope's
    (`GPU_SCOPE_REF`). The bubble shows a "Ref:" line as the muted reference
    line, the last (`help.help_lines`, the WebUI skill's §7); a state's own
    Ref line placed before the scope stood mid-bubble (T06 review)."""
    return f"{text}\n\n{GPU_SCOPE}\nRef: {'; '.join((*refs, GPU_SCOPE_REF))}"


def _since(started_at: Any) -> Optional[str]:
    """``HH:MM UTC`` from a holder record's ISO time, or None."""
    if not isinstance(started_at, str):
        return None
    try:
        moment = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        return None
    from src.webui.components.renderers import clock_utc

    return clock_utc(moment, seconds=False)


def _command(value: Any) -> str:
    text = " ".join(str(value or "unknown command").split())
    return text if len(text) <= _COMMAND_LIMIT else text[:_COMMAND_LIMIT - 1] + "…"


def gpu_chip_from(active: Optional[Mapping[str, Any]], lock: Mapping[str, Any],
                  *, operator: bool) -> Chip:
    """The GPU chip from the runner's active job and the lock as
    `src.utils.gpu_lock.lock_state` reads it now, the one place that turns the
    lock's location and probe into a status (root-cause review, P24-S3): its
    ``problem``, ``unset`` (the problem is that no workspaces root is set),
    ``held_by`` (who holds it at this instant, in words; None when free),
    ``holder`` (the holder's description record, or None) and
    ``holder_alive`` (True, False, or None when that cannot be told).
    `operator` says whether the asking request may see a server path.
    """
    if active:
        job_type = active.get("job_type")
        noun = _RUNNER_NOUN.get(job_type, "job")
        text = f"GPU job: {noun} (job runner)"
        if active.get("status") == "stopping":
            text += " · stopping"
        return Chip("busy", text, gpu_detail(f"Held by the job runner's {noun}."))
    if lock.get("unset"):
        return Chip("unread", GPU_NOT_CONFIGURED,
                    gpu_detail(GPU_NOT_CONFIGURED_DETAIL, GPU_NOT_CONFIGURED_REF))
    if lock.get("problem"):
        detail = ("The GPU lock cannot be used, so every managed job is refused until it is "
                  "corrected.")
        # An operator reads the problem itself (it says where to correct it);
        # anyone else, where an operator sees why, never the value's words.
        detail += f"\n{lock['problem']}" if operator else f"\n{GPU_LOCK_WHY_HIDDEN}"
        return Chip("unread", GPU_NOT_READ, gpu_detail(detail))
    description = str(lock.get("held_by") or "")
    if not description:
        return Chip("none", GPU_NONE, gpu_detail(GPU_FREE_LEAD))
    holder = lock.get("holder")
    alive = isinstance(holder, Mapping) and lock.get("holder_alive") is not False
    if held_by_inference(lock):
        # **This service's inference**, named as such, never by its holder
        # record's words (which name the host), by the one rule the job
        # runner's refusal names it by (`gpu_lock.held_by_inference`).
        return Chip("busy", GPU_INFERENCE,
                    gpu_detail("Held by this service's inference: a model is loaded on CUDA.\n"
                               f"To start a training or a test run, {STOP_INFERENCE_HERE} "
                               "(Model in use).", "POST /api/v1/pipeline/stop"),
                    remedy=STOP_INFERENCE_HERE)
    if alive:
        # What the holder record says, and only that: its command, pid and
        # start. A formal CLI run and another server's job runner write the
        # same record (the runner starts the same scripts), so the record does
        # not say which it is, and neither does the chip (M5-3: a job of a
        # second server on the same lock read "CLI").
        parts = []
        pid = holder.get("pid")
        if isinstance(pid, int) and not isinstance(pid, bool):
            parts.append(f"pid {pid}")
        since = _since(holder.get("started_at"))
        if since:
            parts.append(f"since {since}")
        what = ("GPU: inference" if holder.get("job_type") == SERVING_JOB_TYPE
                else f"GPU job: {_command(holder.get('command'))}")
        text = what + (f" ({', '.join(parts)})" if parts else "")
    else:
        text = "GPU job: lock held, holder not described"
    return Chip("busy", text, gpu_detail(f"Held by {description}."))


def busy_words(chip: Chip) -> str:
    """The GPU chip's words as a research start control repeats them beside the
    start it disables (the Training run bar, Fine-tune, Test): the chip's own,
    then, for this service's inference, where to stop it. A display: the job
    runner refuses a start itself, naming the holder."""
    return f"{chip.text}; {chip.remedy}" if chip.remedy else chip.text


def read_gpu_chip(manager: Any, *, operator: bool,
                  lock_reader: Callable[[], dict] = lock_state) -> Chip:
    """The GPU chip, read now; the grey "not read" state when anything fails."""
    try:
        return gpu_chip_from(manager.active_job(), lock_reader(), operator=operator)
    except Exception:  # noqa: BLE001 - a chip that raises would hide the page's state
        logger.exception("The GPU chip could not be read")
        return Chip("unread", GPU_NOT_READ,
                    gpu_detail("The job runner or the GPU lock could not be read just now."))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def chip_html(chip: Chip, *, which: str) -> str:
    """One chip as HTML: its words (with ``title``, since a long one may end in a
    titled ellipsis), its ⓘ and the ⓘ's hidden text, all escaped."""
    text_id = D8_TEXT_ID if which == "d8" else GPU_TEXT_ID
    title = "Operator controls" if which == "d8" else "GPU jobs"
    # A lock means "refused" (the WebUI skill's §3): only a read-only D8 chip
    # leads with it; the others with the dot.
    lead = (_LOCK_SVG if which == "d8" and chip.kind == "readonly"
            else '<span class="shep-dot" aria-hidden="true"></span>')
    words = html.escape(chip.text)
    return (
        f'<span class="shep-chip shep-chip-{which} shep-chip-{chip.kind}" '
        f'data-state="{chip.kind}">{lead}'
        f'<span class="shep-chip-text" title="{words}">{words}</span>'
        f'{help_button(f"chip-{which}", title, describedby=text_id, extra_class="shep-i-chip")}'
        f'<span id="{text_id}" hidden>{html.escape(chip.detail)}</span></span>'
    )


def chips_html(d8: Chip, gpu: Chip) -> str:
    """Both chips, as the header shows them."""
    return (f'<div class="shep-chips" role="status" aria-live="polite">'
            f'{chip_html(d8, which="d8")}{chip_html(gpu, which="gpu")}</div>')


#: Shown until the first read after a page load (the app is built before the
#: server serves, so nothing is read at build time).
CHIPS_PLACEHOLDER = chips_html(
    Chip("unread", "Operator controls: not read yet",
         f"{NOT_READ_YET} Operator controls:\n{D8_SCOPE}\n\n{D8_HOW}"),
    Chip("unread", GPU_NOT_READ, gpu_detail(NOT_READ_YET)))


def chips_on_change(seen: Any, now: Any, request: Any) -> Any:
    """The chips' update for a page's poll that read the GPU job state: both
    chips read again, for this request and on this tick, when what the poll
    read differs from what it read on its last tick (or it has not read it
    yet); nothing sent otherwise. So the header follows a job's start or end on
    the tick the page sees it (M5-3: "GPU jobs: none" stood beside Training's
    run bar showing a test run, the two read on different timers), with no
    timer and no request of its own."""
    if seen is not None and seen == now:
        return gr.update()
    return read_chips(request)


def read_chips(request: gr.Request) -> str:
    """Both chips for the request that asked: the shell's load, timer and
    after-gated-act step call this with ``queue=False``. Gradio passes the
    request through the positional ``gr.Request`` parameter."""
    from src.api.services.training_manager import training_manager
    from src.webui.operator_access import operator_access

    try:
        access = operator_access(request)
    except Exception:  # noqa: BLE001 - the helper does not raise; a chip never does
        logger.exception("The D8 chip could not be read")
        access = OperatorAccess(False, "unreadable_request", "This request could not be read.")
    return chips_html(d8_chip(access),
                      read_gpu_chip(training_manager, operator=access.allowed))


__all__ = [
    "CHIPS_PLACEHOLDER",
    "Chip",
    "D8_HOW",
    "D8_ON",
    "D8_SCOPE",
    "D8_SHORT",
    "GPU_INFERENCE",
    "GPU_NONE",
    "GPU_LOCK_WHY_HIDDEN",
    "GPU_NOT_READ",
    "GPU_NOT_CONFIGURED",
    "GPU_SCOPE",
    "GPU_SCOPE_REF",
    "LOCK",
    "OPERATOR_ONLY",
    "STOP_INFERENCE_HERE",
    "busy_words",
    "chip_html",
    "chips_html",
    "chips_on_change",
    "d8_chip",
    "gated",
    "gpu_chip_from",
    "gpu_detail",
    "operator_only",
    "read_chips",
    "read_gpu_chip",
]
