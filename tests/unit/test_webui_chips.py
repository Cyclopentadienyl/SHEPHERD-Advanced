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
                    "checkpoint list", "training start and stop", "pipeline reload",
                    "saving settings", "restart"):
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
LOCK_FREE = {"configured": True, "problem": None, "held": False, "holder": None,
             "holder_alive": None, "description": None}


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
    chip = c.gpu_chip_from(None, LOCK_FREE, operator=True)
    assert (chip.kind, chip.text) == ("none", "GPU jobs: none")
    assert "resident inference service is outside the lock" in chip.detail


def test_a_cli_holder_is_named_with_its_pid_and_start_time():
    holder = {"job_type": "training", "command": "train_model.py", "pid": 41822,
              "started_at": "2026-10-20T09:58:12+00:00"}
    lock = {**LOCK_FREE, "held": True, "holder": holder, "holder_alive": True,
            "description": "a training job (train_model.py, pid 41822, as it described itself)"}
    chip = c.gpu_chip_from(None, lock, operator=False)
    assert chip.text == "GPU job: train_model.py (CLI, pid 41822, since 09:58 UTC)"
    assert chip.detail.startswith("Held by a training job")


def test_a_holder_that_exited_is_not_named_as_the_holder():
    holder = {"command": "train_model.py", "pid": 41822, "started_at": "2026-10-20T09:58:12Z"}
    lock = {**LOCK_FREE, "held": True, "holder": holder, "holder_alive": False,
            "description": "a process other than the one its description names"}
    chip = c.gpu_chip_from(None, lock, operator=True)
    assert chip.text == "GPU job: lock held, holder not described"
    lock = {**LOCK_FREE, "held": True, "holder": None, "description": "a process that has not described itself"}
    assert c.gpu_chip_from(None, lock, operator=True).text == (
        "GPU job: lock held, holder not described")


def test_a_lock_not_configured_says_what_is_not_seen():
    chip = c.gpu_chip_from(None, {**LOCK_FREE, "configured": False}, operator=True)
    assert chip.text == "GPU jobs: none in the job runner"
    assert "not configured" in chip.detail and "not seen" in chip.detail


def test_a_lock_that_cannot_be_used_names_its_path_only_to_an_operator():
    lock = {**LOCK_FREE, "configured": False,
            "problem": "SHEPHERD_GPU_LOCK_FILE='/srv/secret/x' is not usable"}
    for operator in (True, False):
        chip = c.gpu_chip_from(None, lock, operator=operator)
        assert (chip.kind, chip.text) == ("unread", "GPU jobs: not read")
        assert ("/srv/secret/x" in chip.detail) is operator


def test_a_runner_that_raises_reads_not_read_with_no_exception_text():
    class Broken:
        def active_job(self):
            raise RuntimeError("boom: /srv/secret")

    chip = c.read_gpu_chip(Broken(), operator=True, lock_reader=lambda: LOCK_FREE)
    assert (chip.kind, chip.text) == ("unread", "GPU jobs: not read")
    assert "boom" not in chip.detail and "RuntimeError" not in chip.detail


def test_a_holder_records_text_is_escaped_in_the_chip():
    holder = {"command": "<img src=x onerror=alert(1)>", "pid": 1, "started_at": None}
    lock = {**LOCK_FREE, "held": True, "holder": holder, "holder_alive": None,
            "description": "<b>x</b>"}
    html = c.chip_html(c.gpu_chip_from(None, lock, operator=True), which="gpu")
    assert "<img" not in html and "&lt;img src=x" in html and "<b>x</b>" not in html


def test_the_chip_reads_a_real_lock_held_elsewhere(monkeypatch, tmp_path):
    """The lock file under the workspaces root, held through another open file
    (as a CLI's would be): the chip names the holder from its record."""
    from src.utils.gpu_lock import GpuLock

    monkeypatch.delenv("SHEPHERD_GPU_LOCK_FILE", raising=False)
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
        assert chip.text.startswith(f"GPU job: train_model.py (CLI, pid {os.getpid()}, since {since}")
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
