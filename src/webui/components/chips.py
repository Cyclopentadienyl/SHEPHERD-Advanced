"""
The header's two chips: operator controls (D8) and GPU jobs (M5-2, UI-A).
=========================================================================
Both are read per request, never cached between sessions: on page load, on the
shell's timer (outside ``gr.Tabs``, ``queue=False``) and after every gated act
(`src/webui/shell.py`, `PageContext.chip_refresh`).

**The D8 chip** says what D8's one check (`src/utils/operator_access.py`) answers
for the request that asked: "🔒 Operator controls: on (this machine)", or grey
"Read-only: <the failed condition in short words>". Its ⓘ lists every operator
control and how to reach them, and, when refused, the check's own reason (which
repeats no header value). It is a display: each control checks its own request.

**The GPU chip** says which managed GPU job holds the one GPU slot, from the job
runner (`training_manager.active_job`) and, for a job started elsewhere, the GPU
lock's descriptive holder record (`src/utils/gpu_lock.py`): "GPU jobs: none",
"GPU job: training (job runner)", "GPU job: test run (job runner)", "GPU job:
train_model.py (CLI, pid …, since …)", or grey "GPU jobs: not read". It never
says "free": the lock is cooperative, and the resident inference service is
outside it, which its ⓘ says.

**Nothing here raises and no exception text is shown**: a read that fails is the
grey "not read" state with a fixed sentence. Every value from a file or a
request is escaped. A server path (the lock file's, in a lock problem) is named
only to a request D8 allows.

Module: src/webui/components/chips.py
"""
from __future__ import annotations

import html
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, NamedTuple, Optional

import gradio as gr

from src.utils.operator_access import CONDITIONS, OperatorAccess
from src.webui.components.help import help_button

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

#: Every operator control, as the D8 chip's ⓘ lists them (no exception remains).
D8_SCOPE = (
    "Operator controls: the import (preview and commit), test start and cancel, "
    "registration, model load, the checkpoint list, training start and stop, pipeline "
    "reload, saving settings, and restart."
)
#: How to reach them.
D8_HOW = (
    "They work only when this deployment is set to M1 (SHEPHERD_DEPLOYMENT_MODE), the "
    "request arrives on this machine's loopback — on the machine itself, or through an SSH "
    "forward to localhost (ssh -L 8264:localhost:8264 <server>, then http://localhost:8264/ui) "
    "— with no forwarding header (no reverse proxy), and it comes from this dashboard's own "
    "page. Each control checks its own request; this chip only shows the answer."
)

#: What the GPU chip's ⓘ always says.
GPU_SCOPE = (
    "One GPU lock covers managed training and evaluation jobs: the ones started here or "
    "through the API (the job runner) and the formal training and evaluation CLIs "
    "(train_model.py, measure_scorer.py). A second job is refused, not queued. The resident "
    "inference service is outside the lock and may still hold GPU memory."
)

GPU_NONE = "GPU jobs: none"
GPU_NOT_READ = "GPU jobs: not read"
GPU_NO_LOCK = "GPU jobs: none in the job runner"
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
    """One chip: its look (`kind`, a CSS modifier), its words and its ⓘ text."""

    kind: str
    text: str
    detail: str


# ---------------------------------------------------------------------------
# D8
# ---------------------------------------------------------------------------
def d8_chip(access: OperatorAccess) -> Chip:
    """The D8 chip for one decision."""
    if access.allowed:
        return Chip("on", D8_ON, f"{D8_SCOPE}\n\n{D8_HOW}")
    short = D8_SHORT.get(access.condition, "request not readable")
    return Chip("readonly", f"Read-only: {short}",
                f"{D8_SCOPE}\n\nFor this request: {access.reason}\n\n{D8_HOW}")


# ---------------------------------------------------------------------------
# GPU jobs
# ---------------------------------------------------------------------------
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
    return moment.astimezone(timezone.utc).strftime("%H:%M UTC")


def _command(value: Any) -> str:
    text = " ".join(str(value or "unknown command").split())
    return text if len(text) <= _COMMAND_LIMIT else text[:_COMMAND_LIMIT - 1] + "…"


def gpu_chip_from(active: Optional[Mapping[str, Any]], lock: Mapping[str, Any],
                  *, operator: bool) -> Chip:
    """The GPU chip from the runner's active job and the lock as read now.

    `lock` holds ``configured``, ``problem``, ``held`` (whether someone holds
    it at this instant), ``holder`` (its description record, or None),
    ``holder_alive`` (True, False, or None when that cannot be told) and
    ``description`` (`describe_holder`'s words). `operator` says whether the
    asking request may see a server path.
    """
    if active:
        job_type = active.get("job_type")
        noun = _RUNNER_NOUN.get(job_type, "job")
        text = f"GPU job: {noun} (job runner)"
        if active.get("status") == "stopping":
            text += " · stopping"
        return Chip("busy", text, GPU_SCOPE)
    if lock.get("problem"):
        detail = ("The GPU lock cannot be used, so every managed job is refused until it is "
                  "corrected.")
        if operator:
            detail += f" {lock['problem']}"
        return Chip("unread", GPU_NOT_READ, f"{detail}\n\n{GPU_SCOPE}")
    if not lock.get("configured"):
        return Chip("none", GPU_NO_LOCK, (
            "The GPU lock is not configured (neither SHEPHERD_GPU_LOCK_FILE nor "
            "SHEPHERD_WORKSPACES_ROOT is set), so a job a CLI runs elsewhere is not seen "
            f"here and is not excluded.\n\n{GPU_SCOPE}"))
    if not lock.get("held"):
        return Chip("none", GPU_NONE, GPU_SCOPE)
    holder = lock.get("holder")
    description = str(lock.get("description") or "")
    if isinstance(holder, Mapping) and lock.get("holder_alive") is not False:
        parts = ["CLI"]
        pid = holder.get("pid")
        if isinstance(pid, int) and not isinstance(pid, bool):
            parts.append(f"pid {pid}")
        since = _since(holder.get("started_at"))
        if since:
            parts.append(f"since {since}")
        text = f"GPU job: {_command(holder.get('command'))} ({', '.join(parts)})"
    else:
        text = "GPU job: lock held, holder not described"
    return Chip("busy", text, f"Held by {description}.\n\n{GPU_SCOPE}" if description
                else GPU_SCOPE)


def read_lock(environ: Optional[Mapping[str, str]] = None) -> dict:
    """The GPU lock as the chip needs it, read now (`gpu_chip_from`)."""
    from src.utils.gpu_lock import (
        GpuLockMisconfigured,
        describe_holder,
        holder_alive,
        lock_location,
        probe,
        read_holder,
    )

    state = {"configured": False, "problem": None, "held": False, "holder": None,
             "holder_alive": None, "description": None}
    try:
        location = lock_location(environ)
    except (GpuLockMisconfigured, OSError) as exc:
        return {**state, "problem": str(exc)}
    if location.path is None:
        return state
    state["configured"] = True
    try:
        held_by = probe(location.path)
    except OSError as exc:
        return {**state, "problem": f"the GPU lock {location.path} cannot be checked ({exc})"}
    if held_by is None:
        return state
    holder = read_holder(location.path)
    return {**state, "held": True, "holder": holder,
            "holder_alive": holder_alive(holder) if holder is not None else None,
            "description": describe_holder(holder)}


def read_gpu_chip(manager: Any, *, operator: bool,
                  lock_reader: Callable[[], dict] = read_lock) -> Chip:
    """The GPU chip, read now; the grey "not read" state when anything fails."""
    try:
        return gpu_chip_from(manager.active_job(), lock_reader(), operator=operator)
    except Exception:  # noqa: BLE001 - a chip that raises would hide the page's state
        logger.exception("The GPU chip could not be read")
        return Chip("unread", GPU_NOT_READ,
                    f"The job runner or the GPU lock could not be read just now.\n\n{GPU_SCOPE}")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def chip_html(chip: Chip, *, which: str) -> str:
    """One chip as HTML: its words (with ``title``, since a long one may end in a
    titled ellipsis), its ⓘ and the ⓘ's hidden text, all escaped."""
    text_id = D8_TEXT_ID if which == "d8" else GPU_TEXT_ID
    title = "Operator controls" if which == "d8" else "GPU jobs"
    lead = (_LOCK_SVG if which == "d8" else '<span class="shep-dot" aria-hidden="true"></span>')
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
    Chip("unread", "Operator controls: not read yet", f"{D8_SCOPE}\n\n{D8_HOW}"),
    Chip("unread", GPU_NOT_READ, GPU_SCOPE))


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
    "GPU_NONE",
    "GPU_NOT_READ",
    "GPU_NO_LOCK",
    "GPU_SCOPE",
    "chip_html",
    "chips_html",
    "d8_chip",
    "gpu_chip_from",
    "read_chips",
    "read_gpu_chip",
    "read_lock",
]
