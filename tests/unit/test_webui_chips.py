"""
The header's chips (M5-2, UI-A): operator controls and GPU jobs.
================================================================
`src/webui/components/chips.py`. What these pin:

- **D8's chip says D8's answer for the request that asked**: on, or grey
  "Read-only: <the failed condition in short words>" for every condition D8
  can report; its ⓘ lists every operator control (no exception) and how to
  reach them, and, when refused, the check's own reason, never a header value;
- **the GPU chip** from the job runner and the lock's holder record: none,
  training or test run (job runner), a CLI holder with pid and start time, a
  holder that exited, a lock not configured, a lock that cannot be used, and
  grey "not read" when anything fails. Never "free"; no exception text; a
  server path only for a request D8 allows; a holder record's text escaped;
- **wired as the page contract says**: read on load, on the shell's slow
  timer and after a gated act, each with no queue, for the triggering request.

Module: tests/unit/test_webui_chips.py
"""
from __future__ import annotations

import os
import socket
from datetime import datetime, timezone

import pytest

gr = pytest.importorskip("gradio")

from src.utils.operator_access import CONDITIONS, ENABLED, OperatorAccess  # noqa: E402
from src.webui.components import chips as c  # noqa: E402
from tests.fixtures.operator_access import gradio_request, operator_mode  # noqa: E402

CANARY = "canary-header-value-7f3a"


# --------------------------------------------------------------------------- D8
def test_on_says_this_machine_and_lists_every_control():
    chip = c.d8_chip(ENABLED)
    assert (chip.kind, chip.text) == ("on", "Operator controls: on (this machine)")
    for control in ("import", "test start and cancel", "registration", "model load",
                    "stopping inference", "checkpoint list", "training start and stop",
                    "pipeline reload", "saving settings", "restart"):
        assert control in chip.detail, control
    assert "SSH forward" in chip.detail and "forwarding header" in chip.detail
    assert "except" not in chip.detail.lower()


@pytest.mark.parametrize("condition", [x for x in CONDITIONS if x != "enabled"])
def test_every_refused_condition_reads_read_only_in_short_words(condition):
    access = OperatorAccess(False, condition, f"The reason for {condition}.")
    chip = c.d8_chip(access)
    assert chip.kind == "readonly"
    assert chip.text == f"Read-only: {c.D8_SHORT[condition]}"
    assert f"For this request: The reason for {condition}." in chip.detail
    assert c.D8_SCOPE in chip.detail


def test_the_short_words_cover_every_condition():
    assert set(c.D8_SHORT) == set(CONDITIONS) - {"enabled"}
    assert c.D8_SHORT["not_loopback"] == "request not local"
    assert c.D8_SHORT["deployment_mode"] == "deployment mode not set to M1"


def test_the_chip_for_a_real_request_never_repeats_a_header_value(monkeypatch):
    operator_mode(monkeypatch)
    html = c.read_chips(gradio_request(headers=[("X-Forwarded-For", CANARY)]))
    assert "Read-only: forwarded by a proxy" in html
    assert CANARY not in html


def test_without_mode_m1_the_chip_says_so(monkeypatch):
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)
    html = c.read_chips(gradio_request())
    assert "Read-only: deployment mode not set to M1" in html


def test_without_a_request_it_fails_closed(monkeypatch):
    operator_mode(monkeypatch)
    assert "Read-only: request not readable" in c.read_chips(None)


# --------------------------------------------------------------------------- GPU jobs
LOCK_FREE = {"configured": True, "path": "/srv/ws/.shepherd-gpu.lock", "held_by": None,
             "holder": None, "holder_alive": None, "problem": None, "unset": False}


@pytest.mark.parametrize("job_type, status, text", [
    ("training", "running", "GPU job: training (job runner)"),
    ("test", "running", "GPU job: test run (job runner)"),
    ("training", "stopping", "GPU job: training (job runner) · stopping"),
])
def test_a_runner_job_is_named_by_its_type(job_type, status, text):
    chip = c.gpu_chip_from({"job_type": job_type, "status": status, "pid": 9}, LOCK_FREE,
                           operator=True)
    assert (chip.kind, chip.text) == ("busy", text)
    assert c.GPU_SCOPE in chip.detail


def test_nothing_running_is_none_never_free():
    """Never "free": the lock is cooperative. Since the ops round's A3a CUDA
    inference holds it, so the ⓘ says that and that the CPU takes none; it no
    longer says inference is outside the lock."""
    chip = c.gpu_chip_from(None, LOCK_FREE, operator=True)
    assert (chip.kind, chip.text) == ("none", "GPU jobs: none")
    assert "inference holds it while a model is loaded on CUDA" in chip.detail
    assert "Inference on the CPU takes no lock" in chip.detail
    assert "outside the lock" not in chip.detail and chip.remedy == ""


def _inference_holder(pid):
    from src.utils.gpu_lock import SERVING_COMMAND, SERVING_JOB_TYPE

    holder = {"job_type": SERVING_JOB_TYPE, "command": SERVING_COMMAND, "pid": pid,
              "host": "server-host-name", "started_at": "2026-11-12T09:58:12+00:00"}
    return {**LOCK_FREE, "holder": holder, "holder_alive": True,
            "held_by": f"an inference job ({SERVING_COMMAND}, pid {pid} on server-host-name, "
                       "as it described itself)"}


def test_this_services_inference_is_named_as_such_with_where_to_stop_it():
    """The ops round's A3b: a holder whose job type is inference and whose pid
    is this process's is this service's CUDA inference, "GPU: inference (this
    service)"; its ⓘ says where to stop it and names no host; the research
    start lines repeat the words with the pointer (`busy_words`)."""
    chip = c.gpu_chip_from(None, _inference_holder(os.getpid()), operator=False)
    assert (chip.kind, chip.text) == ("busy", "GPU: inference (this service)")
    assert chip.remedy == c.STOP_INFERENCE_HERE
    assert "Stop inference on Diagnosis" in chip.detail and "POST /api/v1/pipeline/stop" in chip.detail
    assert "server-host-name" not in chip.detail and c.GPU_SCOPE in chip.detail
    assert c.busy_words(chip) == ("GPU: inference (this service); stop it with Stop inference "
                                  "on Diagnosis")
    html = c.chip_html(chip, which="gpu")
    assert "GPU: inference (this service)" in html and "server-host-name" not in html


def test_another_processs_inference_is_inference_with_its_pid_never_this_service():
    chip = c.gpu_chip_from(None, _inference_holder(os.getpid() + 1), operator=True)
    assert chip.text == f"GPU: inference (pid {os.getpid() + 1}, since 09:58 UTC)"
    assert chip.remedy == "" and c.busy_words(chip) == chip.text
    assert "this service" not in chip.text


def test_a_runner_job_needs_no_pointer():
    chip = c.gpu_chip_from({"job_type": "training", "status": "running"}, LOCK_FREE,
                           operator=True)
    assert c.busy_words(chip) == "GPU job: training (job runner)"


def test_the_chip_and_the_research_starts_read_this_services_real_holding(monkeypatch,
                                                                          tmp_path):
    """This process holding the lock as its CUDA inference does
    (`take_gpu_lock(SERVING_JOB_TYPE, SERVING_COMMAND)`): the chip, the
    Training run bar's GPU line and the Test page's start reason name it and
    point to Stop inference."""
    from src.utils.gpu_lock import SERVING_COMMAND, SERVING_JOB_TYPE, lock_state, take_gpu_lock
    from src.webui.components import test_page as tp
    from src.webui.components import training_console as tc

    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(tmp_path))

    class Idle:
        def active_job(self):
            return None

    held = take_gpu_lock(SERVING_JOB_TYPE, SERVING_COMMAND)
    try:
        assert c.read_gpu_chip(Idle(), operator=False).text == c.GPU_INFERENCE
        line = tc._gpu_line(None, lock_state(), None, operator=True)
        assert line == {"text": f"{c.GPU_INFERENCE}; {c.STOP_INFERENCE_HERE}", "test": False}
        kind, words = tp.gpu_line(manager=Idle(), operator=True)
        assert (kind, words) == ("busy", f"{c.GPU_INFERENCE}; {c.STOP_INFERENCE_HERE}")
    finally:
        held.release()
    assert c.read_gpu_chip(Idle(), operator=True).text == "GPU jobs: none"


def test_a_holder_outside_this_runner_is_named_by_what_its_record_says():
    """Its command, pid and start time, from the holder record. Not "CLI": a
    formal CLI run and another server's job runner on the same lock write the
    same record, so the record does not say which it is (M5-3)."""
    holder = {"job_type": "training", "command": "train_model.py", "pid": 41822,
              "started_at": "2026-10-20T09:58:12+00:00"}
    lock = {**LOCK_FREE, "holder": holder, "holder_alive": True,
            "held_by": "a training job (train_model.py, pid 41822, as it described itself)"}
    chip = c.gpu_chip_from(None, lock, operator=False)
    assert chip.text == "GPU job: train_model.py (pid 41822, since 09:58 UTC)"
    assert "CLI" not in chip.text
    assert chip.detail.startswith("Held by a training job")
    bare = c.gpu_chip_from(None, {**lock, "holder": {"command": "measure_scorer.py"}},
                           operator=False)
    assert bare.text == "GPU job: measure_scorer.py"


def test_a_holder_that_exited_is_not_named_as_the_holder():
    holder = {"command": "train_model.py", "pid": 41822, "started_at": "2026-10-20T09:58:12Z"}
    lock = {**LOCK_FREE, "holder": holder, "holder_alive": False,
            "held_by": "a process other than the one its description names"}
    chip = c.gpu_chip_from(None, lock, operator=True)
    assert chip.text == "GPU job: lock held, holder not described"
    lock = {**LOCK_FREE, "holder": None, "held_by": "a process that has not described itself"}
    assert c.gpu_chip_from(None, lock, operator=True).text == (
        "GPU job: lock held, holder not described")


def test_a_lock_not_configured_says_so_and_that_jobs_are_refused(monkeypatch):
    """Review of 8242b47, P2-4: with no workspaces root set every job is
    refused, and the chip says so to every request (no path in it); it used to
    read "none in the job runner", as if jobs ran unlocked."""
    from src.utils.gpu_lock import lock_state

    monkeypatch.delenv("SHEPHERD_WORKSPACES_ROOT", raising=False)
    lock = lock_state()
    assert lock["unset"] is True and lock["configured"] is False and lock["problem"]
    for operator in (True, False):
        chip = c.gpu_chip_from(None, lock, operator=operator)
        assert (chip.kind, chip.text) == ("unread", "GPU jobs: refused, lock not configured")
        assert chip.detail.startswith("The GPU lock is not configured: "
                                      "SHEPHERD_WORKSPACES_ROOT, which names it, is not set")
        assert "refuses every training and test run" in chip.detail
        assert read_gpu_chip_now(operator) == chip


def read_gpu_chip_now(operator: bool) -> c.Chip:
    """The chip as the header reads it, through the lock's one status reader."""
    class Idle:
        def active_job(self):
            return None

    return c.read_gpu_chip(Idle(), operator=operator)


def test_the_chip_reads_the_lock_s_one_status_reader(monkeypatch, tmp_path):
    """Root-cause review, P24-S3: the chip turns no location or probe outcome
    into a state of its own; it reads `lock_state`, so a state the lock rule
    gives (here a workspaces root that is set but refused) reaches the chip
    as the runner sees it."""
    from src.utils import gpu_lock

    assert c.read_gpu_chip.__kwdefaults__["lock_reader"] is gpu_lock.lock_state
    assert not hasattr(c, "read_lock"), "a second reader of the lock's state"
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(tmp_path / "missing"))
    for operator in (True, False):
        chip = read_gpu_chip_now(operator)
        assert (chip.kind, chip.text) == ("unread", "GPU jobs: not read")
        assert ("The GPU lock under it cannot be used either" in chip.detail) is operator


def test_a_lock_that_cannot_be_used_names_its_path_only_to_an_operator():
    lock = {**LOCK_FREE, "configured": False,
            "problem": "SHEPHERD_WORKSPACES_ROOT='/srv/secret/x' is not usable"}
    for operator in (True, False):
        chip = c.gpu_chip_from(None, lock, operator=operator)
        assert (chip.kind, chip.text) == ("unread", "GPU jobs: not read")
        assert ("/srv/secret/x" in chip.detail) is operator


def test_the_lock_path_goes_only_to_a_request_d8_allows(monkeypatch):
    """`read_chips` tells the GPU chip whether the request that asked may see a
    server path: D8's own answer for that request, nothing else."""
    from tests.fixtures.operator_access import operator_request

    seen = []
    monkeypatch.setattr(c, "read_gpu_chip", lambda manager, *, operator: (
        seen.append(operator) or c.Chip("none", c.GPU_NONE, c.GPU_SCOPE)))
    c.read_chips(operator_request(monkeypatch))
    c.read_chips(gradio_request(server=("192.0.2.10", 8264)))
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)
    c.read_chips(gradio_request())
    assert seen == [True, False, False]


def test_a_runner_that_raises_reads_not_read_with_no_exception_text():
    class Broken:
        def active_job(self):
            raise RuntimeError("boom: /srv/secret")

    chip = c.read_gpu_chip(Broken(), operator=True, lock_reader=lambda: LOCK_FREE)
    assert (chip.kind, chip.text) == ("unread", "GPU jobs: not read")
    assert "boom" not in chip.detail and "RuntimeError" not in chip.detail


def test_a_holder_records_text_is_escaped_in_the_chip():
    holder = {"command": "<img src=x onerror=alert(1)>", "pid": 1, "started_at": None}
    lock = {**LOCK_FREE, "holder": holder, "holder_alive": None, "held_by": "<b>x</b>"}
    html = c.chip_html(c.gpu_chip_from(None, lock, operator=True), which="gpu")
    assert "<img" not in html and "&lt;img src=x" in html and "<b>x</b>" not in html


def test_the_chip_reads_a_real_lock_held_elsewhere(monkeypatch, tmp_path):
    """The lock file under the workspaces root, held through another open file
    (as a CLI's would be): the chip names the holder from its record."""
    from src.utils.gpu_lock import GpuLock

    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(tmp_path))

    class Idle:
        def active_job(self):
            return None

    assert c.read_gpu_chip(Idle(), operator=True).text == "GPU jobs: none"
    lock = GpuLock(tmp_path / ".shepherd-gpu.lock", job_type="training",
                   command="train_model.py").acquire()
    try:
        chip = c.read_gpu_chip(Idle(), operator=True)
        since = datetime.now(timezone.utc).strftime("%H:")
        assert chip.text.startswith(f"GPU job: train_model.py (pid {os.getpid()}, since {since}")
        assert socket.gethostname() in chip.detail
    finally:
        lock.release()
    assert c.read_gpu_chip(Idle(), operator=True).text == "GPU jobs: none"


def test_both_chips_render_with_their_i_and_its_hidden_text():
    html = c.chips_html(c.d8_chip(ENABLED), c.gpu_chip_from(None, LOCK_FREE, operator=True))
    assert html.count('class="shep-i shep-i-chip"') == 2
    assert 'aria-describedby="shep-chip-d8-text"' in html
    assert '<span id="shep-chip-d8-text" hidden>' in html
    assert 'role="status"' in html
    assert "free" not in html.lower()


# --------------------------------------------------------------------------- wiring
def test_the_chips_are_read_on_load_on_the_slow_timer_and_with_no_queue():
    from src.webui import shell
    from src.webui.app import PAGES

    with gr.Blocks() as demo:
        ctx = shell.build_shell(demo, PAGES)
    reads = [fn for fn in demo.fns.values() if fn.fn is c.read_chips]
    triggers = {t for fn in reads for t in fn.targets}
    assert (demo._id, "load") in triggers
    assert (ctx.timer_slow._id, "tick") in triggers
    assert all(fn.queue is False and fn.outputs == [ctx.chips] for fn in reads)
    # and after the gated acts already on the pages (training, reload, settings)
    followers = [fn for fn in reads if fn.trigger_after is not None]
    assert len(followers) >= 5


# --------------------------------------------------------------------------- the gated-button rule
@pytest.mark.parametrize("condition", sorted(c.D8_SHORT))
def test_a_gated_button_shows_no_lock_while_allowed_and_the_lock_and_reason_when_refused(
        condition):
    """The one gated-button rule (the WebUI skill §3, a lock means refused):
    while D8 allows the request, the button's words as they are, with no
    lock, usable when the page's own condition holds; while D8 refuses it,
    disabled whatever the page's condition, with the lock before its words,
    and the visible reason beside it in D8's short words."""
    for interactive in (True, False):
        assert c.gated("▸ Start", True, interactive=interactive) == {
            "value": "▸ Start", "interactive": interactive}
        assert c.gated("▸ Start", False, interactive=interactive) == {
            "value": "🔒 ▸ Start", "interactive": False}
    assert c.operator_only(condition) == f"Operator controls only: {c.D8_SHORT[condition]}"
    assert c.operator_only(None) == "Operator controls only: request not readable"


def test_every_gated_page_labels_its_buttons_by_the_one_rule():
    """Each page's gated buttons read the rule's words: Training's run bar,
    Fine-tune's Start and Stop, the Test page's, the Diagnosis card's Load or
    Change model… and Stop inference, the shared list's Load. No page keeps a
    lock in a label it draws for an operator."""
    from src.webui.components import diagnosis_card as card
    from src.webui.components import finetune_page as fp
    from src.webui.components import model_list as ml
    from src.webui.components import test_page as tp
    from src.webui.components import training_console as tc

    labels = [*tc.ACTION_LABELS.values(), fp.START_LABEL, fp.STOP_LABEL,
              *tp.GATED_LABELS.values(), ml.LOAD_LABEL]
    assert all("🔒" not in label for label in labels)
    for operator in (True, False):
        state = {"ok": True, "status": {"initialized": False}, "operator": operator,
                 "d8_short": None if operator else "request not local"}
        load = card.change_button(state)
        assert load["value"] == ("Load a model…" if operator else "🔒 Load a model…")
        reason = card.change_reason_html(state)
        assert ("Operator controls only:" in reason) is not operator
        listing = {"refused": not operator, "condition": "not_loopback", "items": []}
        assert ml.load_button(listing, None)["value"] == (
            "Load this model" if operator else "🔒 Load this model")
        beside = ml.load_reason_html(None, listing)
        assert ("🔒 Operator controls only: request not local." in beside) is not operator
