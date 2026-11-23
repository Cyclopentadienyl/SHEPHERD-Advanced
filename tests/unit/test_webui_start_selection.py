"""
Start uses the selection shown (OPS-C-R2)
=========================================
The external review of 6a2d52e, OPS-C-R2: a select's choice is read by later
events (the Fine-tune dataset's read, then its base list; a Run's
checkpoints), each a request of its own, and a ``gr.State`` is read when an
event runs. Start pressed while that reading was on its way, or after it
failed, read the state of the choice before: Fine-tune started on the dataset
and the base shown before the one chosen. The shared boundary
(`shell.unread_choice`): Start's own request carries the selects' values as
the reader sees them, and the server refuses beside Start, starting nothing,
while the session's state was read for another choice.

What these pin, in a real browser (skipped without Node and Playwright), the
listener that reads the choice held, failing or late, as the review did:

- **Fine-tune**: A → B then Start at once; A → B → A; the base list late; the
  dataset's read failing; a Run's checkpoints late. Each ends in exactly one
  start, with the dataset and the base shown at that Start, and a Start
  pressed before the reading landed is a stated refusal, never a start;
- **Training**: A → B in the run bar, then Start at once (the same window);

and, as handler tests, the Continue drawer's start and the Test page's Start
refuse a choice not read yet and start nothing.

Module: tests/unit/test_webui_start_selection.py
"""
from __future__ import annotations

import contextlib
import http.server
import json
import shutil
import threading
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

gr = pytest.importorskip("gradio")

from src.webui import shell  # noqa: E402
from src.webui.components import finetune_page as fp  # noqa: E402
from src.webui.components import model_list as ml  # noqa: E402
from src.webui.components import training_console as tc  # noqa: E402
from tests.fixtures.operator_access import operator_mode  # noqa: E402

BASE_RUN = "20261001-000000_base_aaaaaa"
LATE_RUN = "20261002-000000_late_bbbbbb"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from tests.fixtures.registered_cohort import write_demo_workspace

    return write_demo_workspace(tmp_path_factory.mktemp("selection") / "demo")


@pytest.fixture
def root(built, tmp_path, monkeypatch):
    """Two listed workspaces, ``alpha`` and ``beta``, each a base in
    `BASE_RUN` (the same key in both, as in the review); ``alpha`` also one in
    `LATE_RUN`, its latest run, so its Run select offers two runs."""
    from tests.unit.test_task_first_selection import _put, _record_payload

    root = tmp_path / "root"
    for name in ("alpha", "beta"):
        shutil.copytree(built, root / name)
        _put(root / name, "gatv2", BASE_RUN, _record_payload(logs={"val_lp_accuracy": 0.83}),
             300)
    _put(root / "alpha", "gatv2", LATE_RUN, _record_payload(logs={"val_lp_accuracy": 0.8}), 400)
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root))
    return root


class Gate:
    """The one read held (or failed) at a time: armed by the browser's script
    (``/arm``) for a listener and the choice it reads; ``/entered`` answers once
    the held read has begun, ``/release`` lets it go on."""

    def __init__(self) -> None:
        self.armed = None
        self.scenario = None
        self.entered, self.release = threading.Event(), threading.Event()

    def arm(self, scenario: str, what: str, match: str) -> None:
        self.entered.clear()
        self.release.clear()
        self.scenario, self.armed = scenario, (what, match)

    def meet(self, what: str, value) -> None:
        """A wrapped read: held, or failed, once, when armed for it."""
        if self.armed is None or self.armed[0] not in (what, f"fail-{what}") \
                or self.armed[1] not in str(value):
            return
        kind, self.armed = self.armed[0], None
        if kind.startswith("fail-"):
            raise RuntimeError("the read failed (test)")
        self.entered.set()
        assert self.release.wait(60), "never released"


@contextlib.contextmanager
def control(gate: Gate):
    """`gate` served to the browser's script on a loopback port of its own."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - the base class's name
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path == "/arm":
                gate.arm(query["scenario"], query["what"], query["match"])
                body = "armed"
            elif url.path == "/entered":
                body = "1" if gate.entered.wait(30) else "0"
            elif url.path == "/release":
                gate.release.set()
                body = "released"
            else:
                self.send_error(404)
                return
            data = body.encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        gate.release.set()
        server.shutdown()
        thread.join(10)


#: The scenarios, in a real browser at the owner's window. Each opens the page
#: in a session of its own, makes its choices with the pointer as a reader
#: does, presses Start while the read of the last choice is held (or after it
#: failed), then once it has landed; `startNow` reports what the page showed
#: at each Start: the dataset, the chosen base's summary and what is said
#: beside Start.
_SCENARIOS_JS = r"""
const { chromium } = require('playwright');
(async () => {
  const [url, ctl, ids, shots] = process.argv.slice(2);
  const fns = JSON.parse(ids);
  const call = async (path) => (await fetch(ctl + path)).text();
  const browser = await chromium.launch();
  const out = {};
  // Settled: nothing asked is unanswered, and still nothing after the next
  // frames (an answer's write starts a select's change, asked in its turn).
  const settle = async (page) => {
    for (let i = 0; i < 200; i++) {
      await page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 60000 });
      const still = await page.evaluate(() => new Promise((resolve) => {
        const asked = window.__shepQueue.joins.length;
        requestAnimationFrame(() => requestAnimationFrame(() => setTimeout(() => resolve(
          window.__shepQueue.idle() && window.__shepQueue.joins.length === asked), 0)));
      }));
      if (still) return;
    }
    throw new Error('never settled');
  };
  const open = async (where) => {
    const page = await browser.newPage({ viewport: { width: 1850, height: 960 } });
    await page.addInitScript(__QUEUE_JS__);
    await page.goto(url);
    await page.click('#shep-tab-research-button', { force: true });
    await page.waitForSelector('#shep-tab-research-button[aria-selected="true"]');
    if (where === 'finetune') {
      await page.click('#shep-page-finetune-button', { force: true });
      await page.waitForSelector('#shep-page-finetune-button[aria-selected="true"]');
    }
    const start = where === 'finetune' ? '#ft_start' : '#shep-train-start';
    await page.waitForFunction((start) => {
      const b = document.querySelector(start);
      return !!b && !b.disabled && window.__shepQueue.idle();
    }, start, { timeout: 60000 });
    if (where === 'finetune') {
      await page.locator('#ft_tasks input[type=checkbox]').first().check();
      await settle(page);
    }
    return page;
  };
  // An option chosen with the pointer: the select opened, then the option.
  const pick = async (page, id, name) => {
    const shown = await page.locator(`#${id} input`).inputValue();
    await page.locator(`#${id} input`).click();
    const option = name === null
      ? page.locator('[role=option]').filter({ hasNotText: shown }).first()
      : page.getByRole('option', { name, exact: true });
    await option.click();
    await page.locator('[role=option]').first().waitFor({ state: 'detached' });
  };
  // Start pressed; returns once its answer is drawn (the next step of its
  // chain is asked only after Start's outputs are written).
  const startNow = async (page, where) => {
    const ids = fns[where];
    const before = await page.evaluate((i) => window.__shepQueue.count(i), ids.after);
    const shown = await page.evaluate((where) => {
      const text = (sel) => { const el = document.querySelector(sel);
                              return el ? el.innerText.replace(/\s+/g, ' ').trim() : null; };
      const value = (sel) => { const el = document.querySelector(sel);
                               return el ? el.value : null; };
      return where === 'finetune'
        ? { dataset: value('#ft_dataset input'), base: text('#ft_base') }
        : { dataset: value('#shep-train-dataset input'), base: null };
    }, where);
    await page.locator(where === 'finetune' ? '#ft_start' : '#shep-train-start').click();
    await page.waitForFunction(([i, n]) => window.__shepQueue.count(i) > n, [ids.after, before],
                               { timeout: 30000 });
    await page.evaluate(() => new Promise((r) =>
      requestAnimationFrame(() => requestAnimationFrame(r))));
    shown.notice = await page.evaluate((sel) => document.querySelector(sel).innerText.trim(),
                                       where === 'finetune' ? '#ft_notice' : '#shep-train-notice');
    return shown;
  };
  const arm = (scenario, what, match) =>
    call(`/arm?scenario=${scenario}&what=${what}&match=${encodeURIComponent(match)}`);
  const entered = async () => {
    if ((await call('/entered')) !== '1') throw new Error('never held');
  };
  try {
    // A -> B, Start at once (the dataset's read held).
    await arm('a_to_b', 'dataset', 'beta');
    let page = await open('finetune');
    const before = await page.locator('#ft_dataset input').inputValue();
    await pick(page, 'ft_dataset', 'beta');
    await entered();
    const first = await startNow(page, 'finetune');
    await page.screenshot({ path: `${shots}/a_to_b-held-1850x960.png` });
    await call('/release');
    await settle(page);
    out.a_to_b = { before, first, last: await startNow(page, 'finetune') };
    await page.close();

    // A -> B -> A (B's read held while A is chosen again).
    await arm('a_b_a', 'dataset', 'beta');
    page = await open('finetune');
    await pick(page, 'ft_dataset', 'beta');
    await entered();
    await pick(page, 'ft_dataset', 'alpha');
    await call('/release');
    await settle(page);
    out.a_b_a = { last: await startNow(page, 'finetune') };
    await page.close();

    // The base list late (B's dataset read, its list held).
    await arm('list_late', 'bases', 'beta');
    page = await open('finetune');
    await pick(page, 'ft_dataset', 'beta');
    await entered();
    const late = await startNow(page, 'finetune');
    await call('/release');
    await settle(page);
    out.list_late = { first: late, last: await startNow(page, 'finetune') };
    await page.close();

    // The dataset's read failing, then the dataset chosen again.
    await arm('read_fails', 'fail-dataset', 'beta');
    page = await open('finetune');
    await pick(page, 'ft_dataset', 'beta');
    await settle(page);
    const failed = await startNow(page, 'finetune');
    await pick(page, 'ft_dataset', 'beta');
    await settle(page);
    out.read_fails = { first: failed, last: await startNow(page, 'finetune') };
    await page.close();

    // A Run's checkpoints late (the other run chosen, its checkpoints held).
    await arm('run_late', 'run', 'base_aaaaaa');
    page = await open('finetune');
    await pick(page, 'ft_run', null);
    await entered();
    const run = await startNow(page, 'finetune');
    await call('/release');
    await settle(page);
    out.run_late = { first: run, last: await startNow(page, 'finetune') };
    await page.close();

    // Training: A -> B in the run bar, Start at once.
    await arm('training', 'train-dataset', 'beta');
    page = await open('training');
    await pick(page, 'shep-train-dataset', 'beta');
    await entered();
    const training = await startNow(page, 'training');
    await call('/release');
    await settle(page);
    out.training = { first: training, last: await startNow(page, 'training') };
    await page.close();
  } finally {
    await call('/release').catch(() => {});
    await browser.close();
  }
  console.log(JSON.stringify(out));
})();
"""


def test_start_uses_the_selection_shown_or_refuses_one_not_read_yet(root, monkeypatch,
                                                                     tmp_path):
    """OPS-C-R2 (P2), in a real browser at 1850 x 960, the page served whole:
    each scenario's Start pressed while the last choice's read is held, late or
    failed is a stated refusal beside Start, and starts nothing; once the read
    has landed (or the choice is made again after a failure) one Start starts
    exactly one run, on the dataset and the base the page shows. Fails on
    6a2d52e: Fine-tune started on ``alpha`` with ``beta`` shown (A → B; the
    failed read), on ``beta`` with ``alpha`` shown once A → B → A settled (the
    second choice was dropped while the first one's read ran), on the other
    run's checkpoint while the Run select showed this one; and Training on
    ``alpha`` with ``beta`` in its run bar."""
    from src.webui.app import mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    operator_mode(monkeypatch)
    gate = Gate()
    starts = []

    def start_finetune(inputs, **kwargs):
        starts.append((gate.scenario, Path(inputs.data_dir).name,
                       Path(inputs.finetune_from).parts[-3]))
        return {"success": True}

    def start_training(config, **kwargs):
        starts.append((gate.scenario, Path(config["data_dir"]).name, None))
        return {"success": True}

    monkeypatch.setattr(fp.training_manager, "start_finetune", start_finetune)
    monkeypatch.setattr(tc.training_manager, "start_training", start_training)
    # The reads a choice starts, each wrapped to be held or failed by the gate;
    # what they read and write is unchanged.
    read_dataset, read_bases = fp._on_dataset, fp._on_finetune_bases
    read_run, read_training = ml.on_run_chosen, tc._on_dataset

    def held_dataset(name, chosen):
        gate.meet("dataset", name)
        return read_dataset(name, chosen)

    def held_bases(request: gr.Request, ws, current):
        gate.meet("bases", (ws or {}).get("name"))
        return read_bases(request, ws, current)

    def held_run(run, parents, current):
        gate.meet("run", run)
        return read_run(run, parents, current)

    def held_training(name, mode, request: gr.Request):
        gate.meet("train-dataset", name)
        return read_training(name, mode, request)

    monkeypatch.setattr(fp, "_on_dataset", held_dataset)
    monkeypatch.setattr(fp, "_on_finetune_bases", held_bases)
    monkeypatch.setattr(ml, "on_run_chosen", held_run)
    monkeypatch.setattr(tc, "_on_dataset", held_training)
    with gr.Blocks(analytics_enabled=False) as demo:
        shell.build_shell(demo, {**{page: (lambda ctx: None)
                                    for page in (*shell.RESEARCH_PAGES, shell.CLINICAL,
                                                 shell.SETTINGS)},
                                 shell.TRAINING: tc.create_training_tab,
                                 shell.FINETUNE: fp.create_finetune_page})
    fns = list(demo.fns.values())

    def chain(start_fn):
        (start,) = [fn for fn in fns if fn.fn is start_fn]
        (after,) = [fn for fn in fns if fn.trigger_after == start._id]
        return {"start": start._id, "after": after._id}

    ids = {"finetune": chain(fp._on_finetune_start), "training": chain(tc._on_start)}
    script = _SCENARIOS_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS))
    with control(gate) as ctl, served(demo, **mount_options()) as base:
        seen = run_node(script, tmp_path, base[:-len("gradio_api")], ctl, json.dumps(ids),
                        str(tmp_path), timeout=300)

    (tmp_path / "seen.json").write_text(json.dumps({"seen": seen, "starts": starts}, indent=2))

    def started(scenario):
        return [(ws, run) for name, ws, run in starts if name == scenario]

    assert seen["a_to_b"]["before"] == "alpha"
    for scenario, steps in seen.items():
        last, got = steps["last"], started(scenario)
        # Exactly one start, with the dataset and the base shown at that Start.
        assert len(got) == 1, (scenario, got, steps)
        assert got[0][0] == last["dataset"], (scenario, got, last)
        if got[0][1] is not None:
            assert got[0][1] in last["base"], (scenario, got, last)
        assert last["notice"] == "", (scenario, last)
        if "first" in steps:
            # Pressed before the read landed: a stated refusal, nothing started.
            # The mark's shape, its no-break space (`renderers.MARK_SPACE`), its words.
            assert steps["first"]["notice"].startswith("✕\u00a0Not started:"), (scenario, steps)
    assert seen["a_b_a"]["last"]["dataset"] == "alpha"
    # The words: the choice not read yet, by the select's label (the base list
    # late is the list's own refusal: the list shown is the other dataset's).
    unread = {"a_to_b": "Dataset", "read_fails": "Dataset", "run_late": "Run",
              "training": "Dataset"}
    for scenario, what in unread.items():
        assert seen[scenario]["first"]["notice"] == "✕\u00a0Not started: " + shell.UNREAD_CHOICE.format(
            what=what), scenario
    assert "the list shows alpha’s bases, not beta’s" in seen["list_late"]["first"]["notice"]


# --------------------------------------------------------------------------- the shared boundary
def test_the_boundary_names_the_first_choice_not_read_yet():
    """`shell.unread_choice`: each ``(what, shown, read)`` compared in order;
    the first that differ is said by its select's label, in sentences of at
    most 20 words (the skill's §5); none, or all alike, is None."""
    words = shell.UNREAD_CHOICE.format(what="Dataset")
    assert shell.unread_choice() is None
    assert shell.unread_choice(("Dataset", "alpha", "alpha"), ("Run", "r1", "r1")) is None
    assert shell.unread_choice(("Dataset", "beta", "alpha"), ("Run", "r2", "r1")) == words
    assert shell.unread_choice(("Dataset", "alpha", "alpha"), ("Run", "r2", "r1")) == \
        shell.UNREAD_CHOICE.format(what="Run")
    # A choice shown that the session has not read at all (a first choice
    # whose read failed) is not read yet either.
    assert shell.unread_choice(("Dataset", "beta", None)) == words
    assert all(len(sentence.split()) <= 20 for sentence in words.split(". "))


def test_every_start_carries_the_selects_it_uses_and_a_pick_keeps_every_choice():
    """The wiring: Start's own request carries the selects as the reader
    sees them (Fine-tune: Dataset and Run; Training: the run bar's Dataset;
    the Continue drawer's start: the run bar's Dataset and its Run; Test:
    its Run), and the Fine-tune dataset's pick keeps every choice
    ("multiple", as Training's and Test's do), so a choice made while the one
    before it is read is read after it, never dropped."""
    from src.webui.components import test_page as tp

    with gr.Blocks() as demo:
        ctx = shell.standalone_context(demo)
        fp.create_finetune_page(ctx)
        tc.create_training_tab(ctx)
        tp.create_test_page(ctx)
    fns = list(demo.fns.values())
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}

    def named(fn_name, module):
        (fn,) = [f for f in fns if getattr(f.fn, "__name__", None) == fn_name
                 and getattr(f.fn, "__module__", None) == module.__name__]
        return fn

    assert named("_on_finetune_start", fp).inputs[:2] == [by_id["ft_dataset"], by_id["ft_run"]]
    assert named("_on_start", tc).inputs[0] is by_id["shep-train-dataset"]
    assert named("_on_resume", tc).inputs[:2] == [by_id["shep-train-dataset"],
                                                  by_id["continue_run"]]
    test_start = named("_on_start", tp)
    assert test_start.inputs[4] is by_id["shep-test-model-run"]
    assert test_start.inputs[5] is by_id["shep-test-model"]
    pick = named("_on_dataset", fp)
    assert pick.targets == [(by_id["ft_dataset"]._id, "blur")]
    assert pick.trigger_mode == "multiple"


def test_the_continue_drawers_start_refuses_a_choice_not_read_yet(root, monkeypatch):
    """The drawer's start (Continue Training, Start branch) with the run
    bar's Dataset or the list's Run not yet read: refused beside it, in the
    boundary's words, the check never asked and nothing started; settled, it
    goes on to the check as before."""
    from tests.fixtures.operator_access import operator_request

    request = operator_request(monkeypatch)
    asked = []
    monkeypatch.setattr(tc.training_manager, "preview_continuation",
                        lambda *a, **k: asked.append("preview") or {"ok": False, "reason": "spy"})
    monkeypatch.setattr(tc.training_manager, "start_training",
                        lambda *a, **k: asked.append("start") or {"success": True})
    ws = tc.workspace_state("alpha")
    parents, _problem = tc._list_parents(ws)
    key = next(k for k in parents if LATE_RUN in k and k.endswith("/last.pt"))
    late = ml.item_run_key(parents[key])
    base = next(ml.item_run_key(item) for item in parents.values() if BASE_RUN in item["run"])
    values = [tc.form_value(tc._branch_field_value(name)) for name in tc.BRANCH_FIELDS]

    def start(dataset, run):
        return tc._on_resume(request, dataset, run, tc.CONTINUE, ws, "", "cpu",
                             tc.CONTINUE_FROM_LIST, key, "", None, None, parents, {}, [], *values)

    for dataset, run, what in (("beta", late, "Dataset"), ("alpha", base, "Run")):
        said, *rest = start(dataset, run)
        assert said == tc.notice_html("Not started: " + shell.UNREAD_CHOICE.format(what=what),
                                      mark="error")
        assert all(update == gr.update() for update in rest)
    assert asked == []
    start("alpha", late)
    assert asked == ["preview"]


def test_the_test_pages_start_refuses_a_run_not_read_yet(root, monkeypatch):
    """Research › Test's Start with the Run select on a run whose checkpoints
    have not landed (the Checkpoint select still on the run before's):
    refused in the boundary's words, `start_test_run` never asked; settled,
    Start goes on to its own checks (here: no data version listed). A
    read-only request meets D8's refusal first."""
    from src.webui.components import test_page as tp
    from tests.fixtures.operator_access import gradio_request, operator_request

    request = operator_request(monkeypatch)
    asked = []
    real = tp.start_test_run
    monkeypatch.setattr(tp, "start_test_run",
                        lambda *a, **k: asked.append(a[4]) or real(*a, **k))
    with gr.Blocks() as demo:
        tp.create_test_page(shell.standalone_context(demo))
    (start,) = [fn for fn in demo.fns.values() if getattr(fn.fn, "__name__", None) == "_on_start"
                and getattr(fn.fn, "__module__", None) == tp.__name__]
    models = tp.read_models("alpha", request)
    late = next(item for item in models["items"] if LATE_RUN in item["run"])
    key = ml.item_key(late)
    base = next(ml.item_run_key(item) for item in models["items"] if BASE_RUN in item["run"])

    said = start.fn({}, models, "alpha", None, base, key, None, False, False, request)
    assert shell.UNREAD_CHOICE.format(what="Run")[1:] in said and "Nothing was started." in said
    assert asked == []
    said = start.fn({}, models, "alpha", None, ml.item_run_key(late), key, None, False, False,
                    request)
    assert "Choose a data version the server lists" in said and asked == [key]
    forwarded = gradio_request(headers=[("X-Forwarded-For", "203.0.113.7")])
    said = start.fn({}, models, "alpha", None, base, key, None, False, False, forwarded)
    assert "Refused" in said and "still being read" not in said
