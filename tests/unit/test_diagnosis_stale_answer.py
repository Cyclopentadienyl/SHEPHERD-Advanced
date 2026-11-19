"""OPS-C-R3 (the external review of 6a2d52e): **an answer is never shown as
the valid result, nor offered for download, under inputs other than the ones
it answered** (UI-15, B73's residual).

Run Diagnosis' event writes the result, its state and the export files; only
the check chained after it (`answered_inputs`), a request of its own, compared
the inputs and cleared an answer to inputs edited while it ran. With that
request held, the genes box read MECP2 while the table showed the SCN1A/PTEN
answer, and its CSV downloaded; had the request failed, it stayed so.

These tests drive Gradio 6.30's own page in a real browser (Node and
Playwright, skipped without), the real panel, its wiring and the shell; a
task model's status, its answer and the card's facts are stand-ins. The
answer is held on the server until the script lets it go, the edit is made
while it is held, and the check chained after the run is held at the network
(or failed, or the edit's own request is answered after it): what the page
draws is sampled at every change of the page from the answer's arrival on,
never at the end alone.

REV-R3-a and REV-R3-b (the review of 961d901): the page's count of the edits
went up in a listener Gradio dispatches some time after the box's text has
changed (45 to 110 ms measured), so an answer arriving in between was drawn
and offered under the edited box, and a Run pressed in between carried the
new text with the count before the edit, its answer hidden then cleared. The
edit is now counted in its own event (`dp.INPUTS_HEAD`); the browser tests
make the edit, and press Run, with the page's clock paused (no animation
frame and no timeout runs), so nothing Gradio dispatches later can count it.

Module: tests/unit/test_diagnosis_stale_answer.py
"""
from __future__ import annotations

import copy
import json
import time

import gradio as gr
import pytest

from src.webui.components import diagnosis_panel as dp
from tests.unit.test_diagnosis_panel import (
    _build_tab,
    _cohorts,
    _gene_result,
    _named,
    _patients_result,
    _sent,
    _status_serving,
    _task_card,
    _task_identity,
    _version,
)


def test_every_edit_is_counted_in_its_own_event_and_each_request_carries_the_count(
        monkeypatch):
    """The page's inputs generation is the page head's (`INPUTS_HEAD`, in the
    dashboard's head): a count raised, and the page's gate written for it, in
    the edit's own event, by a capturing listener of the HPO box's and the
    genes box's input events, of the Reference cohort's blur and of a click
    on Clear. No listener of the page counts an edit any more (on 961d901 a
    listener of the boxes' ``.change`` and the cohort's blur did, which Gradio
    dispatches later: REV-R3-a, REV-R3-b), and nothing writes the hidden slot:
    each request that decides on an answer (Run Diagnosis, the check chained
    after it, each edit's own listener) carries the count in its last input,
    written by its own script (`INPUTS_GEN_JS`) in the step Gradio reads its
    inputs."""
    from src.webui.app import DASHBOARD_HEAD

    demo, _ctx = _build_tab(monkeypatch)
    blocks = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    gen = blocks["shep-dx-inputs-gen"]
    hpo, genes = blocks["dx_hpo_terms"], blocks["dx_candidate_genes"]
    cohort, clear = blocks["dx_reference_cohort"], blocks["shep-dx-clear"]
    assert isinstance(gen, gr.Number) and gen.visible is False and gen.value == 0
    head = dp.INPUTS_HEAD
    assert head.startswith("<script>") and head.endswith("</script>") and head in DASHBOARD_HEAD
    assert ("var EDITS = { input: '#dx_hpo_terms textarea, #dx_candidate_genes textarea',\n"
            "                focusout: '#dx_reference_cohort input', click: '#shep-dx-clear' };"
            ) in head
    assert "document.addEventListener(type, function (e) {" in head and "}, true);" in head
    assert json.dumps(dp.ANSWER_GATE_CSS) in head and "window.ShepDx = { gen:" in head
    assert not hasattr(dp, "INPUTS_EDITED_JS")
    # No listener counts an edit, and nothing writes the slot.
    assert not [fn for fn in demo.fns.values() if fn.fn is None and fn.js
                and {(hpo._id, "change"), (genes._id, "change"), (cohort._id, "blur")}
                & set(fn.targets)]
    assert [fn for fn in demo.fns.values() if gen in fn.outputs] == []
    carriers = {name: fn for name in ("_run_diagnosis", "_answered_inputs", "_phenotypes_typed",
                                      "_phenotypes_edited", "_cohort_picked")
                for fn in _named(demo, name)}
    assert len(carriers) == 5
    for name, fn in carriers.items():
        assert fn.js == dp.INPUTS_GEN_JS and fn.inputs[-1] is gen, name
    assert [fn.name for fn in demo.fns.values() if gen in fn.inputs] and all(
        fn.js == dp.INPUTS_GEN_JS for fn in demo.fns.values() if gen in fn.inputs)
    # The edits' own listeners are those of the edits the head counts.
    assert carriers["_phenotypes_typed"].targets == [(hpo._id, "change")]
    assert carriers["_phenotypes_edited"].targets == [(genes._id, "change")]
    assert carriers["_cohort_picked"].targets == [(cohort._id, "blur")]
    # Clear writes both boxes in the browser (its own script), with no input
    # event: its click is counted.
    assert isinstance(clear, gr.ClearButton) and {hpo, genes} <= {
        o for fn in demo.fns.values() if (clear._id, "click") in fn.targets for o in fn.outputs}


def test_an_edits_own_request_keeps_an_answer_of_the_count_it_carries(monkeypatch):
    """REV-R3-b: each request that decides on an answer clears it only when
    the generation it answered is not the one the request carries
    (`answered_inputs`): the check after the run and each edit's own
    listener alike, so an answer to a Run pressed just after an edit stays,
    whichever request comes first; a request from a page without the head's
    count (null) keeps none."""
    _sent(monkeypatch, _gene_result())
    status = _status_serving(_task_identity("gene_task"))
    monkeypatch.setattr(dp.card, "read_status_within", lambda *a, **k: (status, None))
    monkeypatch.setattr(dp.card, "served_facts", lambda s, **k: {
        "key": dp.card.served_key(s), "operator": False, "workspaces": [], "serving": None,
        "item": None, "item_problem": None, "recorded": None, "trained": None, "cohorts": None})
    out = dp.diagnose("HP:0001250", "", 10, True, _task_card("gene_task"), "idle", {}, None,
                      genes_text="SCN1A, PTEN Q", gen=12)
    result, state = out["results_state"], out["card_state"]
    assert dp.answered_inputs(result, state, 12) == {}
    assert dp.cohort_picked(result, state, "reference-1", 12).keys() == {"cohort"}
    for page in (11, 13, None):
        assert dp.answered_inputs(result, state, page) == dp.phenotypes_changed(result, state)
        assert dp.cohort_picked(result, state, "reference-1", page)["results_state"] is None
    unmarked = dp.diagnose("HP:0001250", "", 10, True, _task_card("gene_task"), "idle", {},
                           None, genes_text="SCN1A", gen=None)["results_state"]
    assert unmarked["_inputs_gen"] is None and dp.answer_mark(unmarked) == ""
    assert dp.answered_inputs(unmarked, state, None)["results_state"] is None


def test_an_answer_is_marked_with_the_generation_it_answered_and_gated_by_it(monkeypatch):
    """The results body that draws an answer marks the inputs generation it
    answered (`answer_mark`), however it is drawn again (a row chosen); a
    result no run of the page made carries none. The gate (`ANSWER_GATE_CSS`,
    written by `INPUTS_EDITED_JS` for the page's generation) hides every part
    drawn from an answer unless the body draws one of that generation, and
    the body itself when it draws one of another, showing the cleared state's
    words in its place; the page's own style keeps those words hidden."""
    _sent(monkeypatch, _gene_result())
    status = _status_serving(_task_identity("gene_task"))
    monkeypatch.setattr(dp.card, "read_status_within", lambda *a, **k: (status, None))
    monkeypatch.setattr(dp.card, "served_facts", lambda s, **k: {
        "key": dp.card.served_key(s), "operator": False, "workspaces": [], "serving": None,
        "item": None, "item_problem": None, "recorded": None, "trained": None, "cohorts": None})
    out = dp.diagnose("HP:0001250", "", 10, True, _task_card("gene_task"), "idle", {}, None,
                      genes_text="SCN1A, PTEN", gen=7)
    assert out["results_body"].startswith(
        '<span class="shep-dx-answered" data-shep-gen="7" hidden></span>')
    chosen = dp.select_candidate(out["results_state"], "evidence", 2)
    assert 'data-shep-gen="7"' in chosen["results_body"]
    assert "data-shep-gen" not in dp.results_body_html(_gene_result(), 1)
    gate = dp.ANSWER_GATE_CSS
    assert gate.count("{gen}") == 4 and gate.count("!important") == 4
    assert '#shep-dx:not(:has(#shep-dx-results-body [data-shep-gen="{gen}"]))' in gate
    assert ('#shep-dx:has(#shep-dx-results-body [data-shep-gen]:not([data-shep-gen="{gen}"])) '
            "#shep-dx-results-body { display: none !important; }") in gate
    for part in ("#shep-dx-results-hd .shep-panel-right", ".shep-dx-badge-row",
                 "#shep-dx-export-csv", "#shep-dx-export-md", "#shep-dx-details-hd",
                 "#shep-dx-seg", "#shep-dx-details-body", "#shep-dx-explanation",
                 "#shep-dx-details-ft", "#shep-dx-stale", "#shep-dx-details-stale-hd",
                 "#shep-dx-details-stale"):
        assert part in gate, part
    assert json.dumps(gate) in dp.INPUTS_HEAD
    assert ("#shep-tab-clinical #shep-dx-stale, #shep-tab-clinical #shep-dx-details-stale-hd, "
            "#shep-tab-clinical #shep-dx-details-stale { display: none !important; }") in dp.PAGE_CSS
    demo, _ctx = _build_tab(monkeypatch)
    blocks = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    assert dp.INPUTS_CHANGED in blocks["shep-dx-stale"].value
    assert blocks["shep-dx-details-stale"].value == dp.details_empty_html()
    assert blocks["shep-dx-details-stale-hd"].value == dp.details_header_html(None)
    # The cleared state says the same words.
    assert dp.INPUTS_CHANGED in dp.phenotypes_changed(out["results_state"], out["card_state"])[
        "results_body"]

#: **What the page draws of an answer** naming the regular expression `words`
#: (a page script, `window.__shepSample()`): its rows in the table, the
#: selected candidate's details, the header's count, the badge, a download
#: offered (an export button drawn and enabled), whether the page says the
#: inputs changed, and where it draws that notice, the details' heading and
#: their empty state; and, while ``window.__shepSampling`` is set, a sample at
#: every change of the page (``window.__shepSamples``).
_SAMPLER_JS = r"""(words) => {
  const q = (s) => document.querySelector(s);
  const seen = (el) => !!el && el.checkVisibility({ checkVisibilityCSS: true });
  const text = (s) => { const el = q(s); return seen(el) ? el.innerText : ''; };
  const answer = new RegExp(words);
  // Where the first drawn match of `sel` in `scope` starts, in CSS px from the top.
  const top = (scope, sel) => {
    const el = Array.from(document.querySelectorAll(`${scope} ${sel}`)).find(seen);
    return el ? Math.round(el.getBoundingClientRect().top) : null;
  };
  const offered = (s) => {
    const el = q(s);
    const control = el && (el.matches('button, a') ? el : el.querySelector('button, a'));
    return seen(el) && !!control && seen(control) && !control.disabled;
  };
  window.__shepSample = () => ({
    table: answer.test(text('#shep-dx-results-body')),
    details: answer.test(text('#shep-dx-details')),
    count: /candidate/.test(text('#shep-dx-results-hd')),
    // The badge's own parts (the help's described text is not the badge).
    badge: Array.from(document.querySelectorAll(
      '#shep-dx-badge :is(.shep-dx-badge-row, .shep-dx-badge-why)')).some(seen),
    download: offered('#shep-dx-export-csv') || offered('#shep-dx-export-md'),
    changed: /Inputs changed/.test(text('#shep-dx-results')),
    // The notice, the details' heading and their empty state, where they are drawn.
    where: [top('#shep-dx-results', '.shep-dx-notice-p'), top('#shep-dx-details', 'h2'),
            top('#shep-dx-details', '.shep-dx-empty')],
  });
  window.__shepSamples = [];
  window.__shepSampling = false;
  new MutationObserver(() => {
    if (window.__shepSampling) { window.__shepSamples.push(window.__shepSample()); }
  }).observe(document.body, { childList: true, subtree: true, attributes: true, characterData: true });
}"""

#: One page per served task (`mode`: the gene task model, whose answer names
#: SCN1A and PTEN; the patients-like-me model, whose answer names case-07 and
#: case-02). Each case fills the boxes, presses Run Diagnosis with its answer
#: held on the server, makes its edit while it is held (none for the control),
#: lets the answer go and samples, at every change of the page from then on,
#: whether the answer is drawn: its rows in the table, the selected
#: candidate's details, the header's count, the badge, a download offered (an
#: export button drawn and enabled), whether the page says the inputs
#: changed, and where it draws that notice, the details' heading and their
#: empty state. The run's check (`_answered_inputs`) is held at the network
#: ("hold") or fails ("fail"); `late` also holds the edit's own request until
#: the check has answered. The control (no edit) downloads the CSV while its
#: check is held.
_STALE_JS = r"""
const { chromium } = require('playwright');
const fs = require('fs');
const QUEUE_JS = __QUEUE_JS__;
// What the page draws of an answer naming `words`, sampled at every change of the page.
const SAMPLER = __SAMPLER__;
(async () => {
  const [base, gate, asked, modeFile, shots, ids] = process.argv.slice(2, 8);
  const fn = JSON.parse(ids);
  const HPO = '#dx_hpo_terms textarea', GENES = '#dx_candidate_genes textarea';
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1850, height: 960 }, deviceScaleFactor: 1, acceptDownloads: true,
    permissions: ['clipboard-read', 'clipboard-write'] });
  // Requests held at the network, by listener: let go by `release`, or failed.
  const holds = new Map(), ended = new Map();
  const hold = (f, fail) => {
    let release;
    const wait = new Promise((r) => { release = r; });
    const h = { fail: !!fail, wait, release, seen: 0 };
    holds.set(f, h);
    return h;
  };
  const listener = (request) => {
    try { return request.postDataJSON().fn_index; } catch (e) { return null; }
  };
  const intercept = async (route) => {
    const h = holds.get(listener(route.request()));
    if (!h) { return route.continue().catch(() => {}); }
    h.seen += 1;
    // A failed answer (the server's error), not a lost connection, which stops the page.
    if (h.fail) {
      return route.fulfill({ status: 500, contentType: 'application/json',
                             body: JSON.stringify({ error: 'failed by the test' }) }).catch(() => {});
    }
    await h.wait;
    return route.continue().catch(() => {});
  };
  // By path: the page asks with a query (its theme).
  await context.route((url) => /\/gradio_api\/(run\/predict|queue\/join)$/.test(url.pathname), intercept);
  const count = (request) => {
    const f = listener(request);
    if (f !== null && request.url().includes('/run/predict')) { ended.set(f, (ended.get(f) || 0) + 1); }
  };
  context.on('requestfinished', count);
  context.on('requestfailed', count);
  const frames = (page) => page.evaluate(() => new Promise((r) =>
    requestAnimationFrame(() => requestAnimationFrame(r))));
  const calls = () => (fs.existsSync(asked) ? Number(fs.readFileSync(asked, 'utf8') || 0) : 0);
  const until = async (check, what) => {
    for (let i = 0; i < 1500; i++) { if (check()) { return; } await new Promise((r) => setTimeout(r, 20)); }
    throw new Error('never: ' + what);
  };
  // Nothing asked is unanswered, and still nothing after the next frames.
  const settle = async (page) => {
    for (let i = 0; i < 200; i++) {
      await page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 60000 });
      const still = await page.evaluate(() => new Promise((resolve) => {
        const n = window.__shepQueue.joins.length;
        requestAnimationFrame(() => requestAnimationFrame(() => setTimeout(() => resolve(
          window.__shepQueue.idle() && window.__shepQueue.joins.length === n), 0)));
      }));
      if (still) { return; }
    }
    throw new Error('never settled');
  };
  const open = async (mode, box, words) => {
    fs.writeFileSync(modeFile, mode);
    const page = await context.newPage();
    await page.addInitScript(QUEUE_JS);
    // The shell's timers keep the network busy: wait for the box, not for idle.
    await page.goto(base, { waitUntil: 'load' });
    await page.waitForSelector(box, { state: 'visible', timeout: 60000 });
    await page.evaluate(SAMPLER, words);
    return page;
  };
  const sample = (page) => page.evaluate(() => window.__shepSample());
  const run = async (page, name, { fill, edit, value, check, late }) => {
    await fill();
    await settle(page);
    if (fs.existsSync(gate)) { fs.unlinkSync(gate); }
    const checked = hold(fn.check, check === 'fail');
    const asks = calls(), ends = ended.get(fn.check) || 0;
    const runs = await page.evaluate((f) => window.__shepQueue.answered(f), fn.run);
    await page.click('#shep-dx-run-btn');
    await until(() => calls() > asks, 'the run asked');
    const own = late === undefined ? null : hold(late, false);
    if (edit) { await edit(); }
    await frames(page);
    await page.evaluate(() => { window.__shepSamples = []; window.__shepSampling = true; });
    fs.writeFileSync(gate, '');
    await page.waitForFunction(([f, n]) => window.__shepQueue.answered(f) > n, [fn.run, runs],
                               { timeout: 30000 });
    // The run's outputs are written before what is chained after it is asked.
    await until(() => checked.seen > 0, 'the check asked');
    await frames(page);
    const out = { at_answer: await sample(page) };
    await page.screenshot({ path: `${shots}/${name}-at-answer-1850x960.png` });
    if (!edit) {
      const downloading = page.waitForEvent('download', { timeout: 15000 });
      await page.locator('#shep-dx-export-csv').click();
      const file = await (await downloading).path();
      out.csv = fs.readFileSync(file, 'utf8');
    }
    checked.release();
    holds.delete(fn.check);
    await until(() => (ended.get(fn.check) || 0) > ends, 'the check answered');
    await frames(page);
    if (own) {
      out.before_edit_answered = await sample(page);
      own.release();
      holds.delete(late);
    }
    await settle(page);
    out.end = await sample(page);
    await page.evaluate(() => { window.__shepSampling = false; });
    const samples = await page.evaluate(() => window.__shepSamples);
    out.samples = samples.length;
    out.ever = {};
    for (const k of ['table', 'details', 'count', 'badge', 'download', 'changed']) {
      out.ever[k] = samples.some((s) => s[k]);
    }
    out.value = await page.inputValue(value);
    return out;
  };
  const out = {};
  try {
    let page = await open('gene', GENES, 'SCN1A|PTEN');
    const cdp = await page.context().newCDPSession(page);
    const fill = async () => { await page.fill(HPO, 'HP:0001250'); await page.fill(GENES, 'SCN1A, PTEN'); };
    const key = async (sel, keys, text) => {
      await page.click(sel);
      for (const k of keys) { await page.keyboard.press(k); }
      if (text) { await page.keyboard.type(text); }
    };
    out.none = await run(page, 'none', { fill, value: GENES, check: 'hold', edit: null });
    out.typed = await run(page, 'typed', { fill, value: GENES, check: 'hold',
      edit: () => key(GENES, ['End'], ' X') });
    out.deleted = await run(page, 'deleted', { fill, value: GENES, check: 'hold',
      edit: () => key(GENES, ['ControlOrMeta+A', 'Delete']) });
    out.pasted = await run(page, 'pasted', { fill, value: GENES, check: 'hold', edit: async () => {
      await page.evaluate(() => navigator.clipboard.writeText('MECP2'));
      await key(GENES, ['ControlOrMeta+A', 'ControlOrMeta+V']);
    } });
    // An input method's composition, committed (Chromium's own, over CDP: no key event).
    out.composed = await run(page, 'composed', { fill, value: HPO, check: 'hold', edit: async () => {
      await key(HPO, ['ControlOrMeta+A']);
      for (const t of ['h', 'hp']) {
        await cdp.send('Input.imeSetComposition', { text: t, selectionStart: t.length, selectionEnd: t.length });
      }
      await cdp.send('Input.insertText', { text: 'HP:0001263' });
    } });
    out.check_failed = await run(page, 'check_failed', { fill, value: HPO, check: 'fail',
      edit: () => key(HPO, ['ControlOrMeta+A'], 'HP:0001263') });
    out.edit_late = await run(page, 'edit_late', { fill, value: GENES, check: 'hold', late: fn.edited,
      edit: () => key(GENES, ['End'], ' X') });
    await page.close();
    // A patients-like-me model: the Reference cohort switched during the run.
    page = await open('patients', '#dx_reference_cohort input', 'case-0');
    const pick = async (name) => {
      await page.locator('#dx_reference_cohort input').click();
      await page.getByRole('option', { name, exact: true }).click();
      await page.locator('[role=option]').first().waitFor({ state: 'detached' });
    };
    out.cohort = await run(page, 'cohort', { fill: () => page.fill(HPO, 'HP:0001250'),
      value: '#dx_reference_cohort input', check: 'hold', edit: () => pick('reference-2') });
    await page.close();
  } finally {
    for (const h of holds.values()) { h.release(); }
    await browser.close();
  }
  console.log(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
"""


def _served_page(monkeypatch, tmp_path):
    """The Diagnosis page in the real shell, served; a gene task model or a
    patients-like-me model (the script writes which in ``mode``) answers
    each run once ``gate`` exists. Returns the demo, the files and the
    runs asked."""
    statuses = {"gene": _status_serving(_task_identity("gene_task")),
                "patients": _status_serving(_task_identity("patients_like_me"))}
    cohorts = _cohorts(_version("reference-1"), _version("reference-2"))
    mode, gate, asked, calls = tmp_path / "mode", tmp_path / "gate", tmp_path / "asked", []
    mode.write_text("gene")

    def now():
        return mode.read_text() or "gene"

    monkeypatch.setattr(dp.card, "read_status_within", lambda *a, **k: (statuses[now()], None))
    monkeypatch.setattr(dp.card, "served_facts", lambda s, **k: {
        "key": dp.card.served_key(s), "operator": False, "workspaces": [], "serving": None,
        "item": None, "item_problem": None, "recorded": None, "trained": None,
        "cohorts": cohorts if now() == "patients" else None})

    def held_diagnose(**kwargs):
        calls.append(kwargs)
        asked.write_text(str(len(calls)))
        for _ in range(6000):
            if gate.exists():
                break
            time.sleep(0.01)
        return copy.deepcopy(_gene_result() if now() == "gene" else _patients_result())

    monkeypatch.setattr(dp, "_call_diagnose", held_diagnose)
    from src.webui import shell

    with gr.Blocks() as demo:
        shell.build_shell(demo, {**{page: (lambda ctx: None)
                                    for page in (*shell.RESEARCH_PAGES, shell.SETTINGS)},
                                 shell.CLINICAL: dp.create_diagnosis_tab})
    return demo, mode, gate, asked, calls


def test_an_answer_to_inputs_edited_during_its_run_is_never_shown_nor_downloadable(
        monkeypatch, tmp_path):
    """Each edit made while Run Diagnosis is in flight (typed, deleted, pasted,
    an input method's composition committed, the Reference cohort switched),
    with the run's check held, failed, or answered before the edit's own
    request: from the answer's arrival on, the page never draws the answer
    (its rows, the selected candidate's details, the header's count, the
    badge) and never offers its download; it says the inputs changed, and
    ends so, the box as edited. With no edit, the answer is drawn and its CSV
    downloads while the check is still held, as before. On 6a2d52e the answer
    was drawn and offered under the edited box until the check answered, and
    for good once it failed."""
    from src.webui.app import mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    demo, mode, gate, asked, calls = _served_page(monkeypatch, tmp_path)

    def named(name):
        (found,) = [fn._id for fn in demo.fns.values() if fn.name == name]
        return found

    ids = {"run": named("_run_diagnosis"), "check": named("_answered_inputs"),
           "edited": named("_phenotypes_edited")}
    shots = tmp_path / "shots"
    shots.mkdir()
    try:
        with served(demo, **mount_options()) as base:
            seen = run_node(_STALE_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS))
                            .replace("__SAMPLER__", _SAMPLER_JS), tmp_path,
                            base[:-len("gradio_api")], str(gate), str(asked), str(mode),
                            str(shots), json.dumps(ids))
    finally:
        gate.write_text("")
    edited = {"typed": "SCN1A, PTEN X", "deleted": "", "pasted": "MECP2",
              "composed": "HP:0001263", "check_failed": "HP:0001263",
              "edit_late": "SCN1A, PTEN X", "cohort": "reference-2"}
    for case, value in edited.items():
        out = seen[case]
        assert out["value"] == value, (case, out)
        # From the answer's arrival on: never drawn, never offered.
        assert out["samples"] > 0, (case, out)
        assert not any(out["ever"][k] for k in ("table", "details", "count", "badge",
                                                "download")), (case, out)
        # What the page says instead, while the check is held and at the end,
        # drawn where the cleared state draws it.
        assert out["at_answer"]["changed"], (case, out)
        assert out["end"]["changed"] and not out["end"]["table"], (case, out)
        assert None not in out["end"]["where"], (case, out)
        assert out["at_answer"]["where"] == out["end"]["where"], (case, out)
        if "before_edit_answered" in out:
            assert out["before_edit_answered"]["changed"], (case, out)
            assert not out["before_edit_answered"]["table"], (case, out)
    # No edit: drawn and downloadable at once, the check still held.
    none = seen["none"]
    assert none["at_answer"]["table"] and none["at_answer"]["download"], none
    assert none["at_answer"]["details"] and none["at_answer"]["count"], none
    assert not none["ever"]["changed"] and none["end"]["table"], none
    assert "SCN1A" in none["csv"] and "PTEN" in none["csv"], none
    # Every run was asked the inputs as they were when Run was pressed.
    genes = [c["candidate_genes"] for c in calls if c["candidate_genes"] is not None]
    assert genes == [["SCN1A", "PTEN"]] * 7, calls
    assert [c["reference_cohort"] for c in calls][-1] == "reference-1", calls


#: REV-R3-a and REV-R3-b in a browser, **the page's clock paused** for the edit
#: and the press (Playwright's: no animation frame and no timeout runs, so
#: nothing Gradio dispatches after the edit's own event can count it; a
#: click's request still leaves, as the reader's does): the gene task model's
#: answer held on the server until the script lets it go. Each case samples
#: what the page draws (`_SAMPLER_JS`) at every change of the page from just
#: before the answer is let go.
#:
#: - ``typed_then_run``: ``" Q"`` typed at the genes box's end and Run
#:   Diagnosis pressed at once; the clock runs again, then the answer goes.
#: - ``pasted_at_answer``: Run pressed, its answer held; MECP2 pasted over the
#:   genes box and the answer let go while the clock is still paused (the
#:   page writes the answer with no frame); what is drawn then, and once the
#:   clock runs again and every request is answered.
#: - ``cleared_at_answer``: the same with a click on Clear (which writes both
#:   boxes in the browser, with no input event).
_AT_ONCE_JS = r"""
const { chromium } = require('playwright');
const fs = require('fs');
const QUEUE_JS = __QUEUE_JS__;
const SAMPLER = __SAMPLER__;
(async () => {
  const [base, gate, asked, shots, run] = process.argv.slice(2, 7);
  const HPO = '#dx_hpo_terms textarea', GENES = '#dx_candidate_genes textarea';
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1850, height: 960 }, deviceScaleFactor: 1,
    permissions: ['clipboard-read', 'clipboard-write'] });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.clock.install();
  await page.addInitScript(QUEUE_JS);
  await page.goto(base, { waitUntil: 'load' });
  await page.waitForSelector(GENES, { state: 'visible', timeout: 60000 });
  await page.evaluate(SAMPLER, 'SCN1A|PTEN');
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  // Polled from here, never by a timer of the page's (its clock may be paused).
  const until = async (check, what) => {
    for (let i = 0; i < 1500; i++) { if (await check()) { return; } await wait(20); }
    throw new Error('never: ' + what);
  };
  const calls = () => (fs.existsSync(asked) ? Number(fs.readFileSync(asked, 'utf8') || 0) : 0);
  // Run Diagnosis' requests answered.
  const runsAnswered = () => page.evaluate((f) => window.__shepQueue.answered(f), Number(run));
  // Nothing asked is unanswered, and still nothing after the next frames (the clock running).
  const settle = async () => {
    for (let i = 0; i < 200; i++) {
      await page.waitForFunction(() => window.__shepQueue.idle(), null, { timeout: 60000 });
      const still = await page.evaluate(() => new Promise((resolve) => {
        const n = window.__shepQueue.joins.length;
        requestAnimationFrame(() => requestAnimationFrame(() => setTimeout(() => resolve(
          window.__shepQueue.idle() && window.__shepQueue.joins.length === n), 0)));
      }));
      if (still) { return; }
    }
    throw new Error('never settled');
  };
  // The page's clock paused, and seen to stand still.
  const pause = async () => {
    for (let i = 0; i < 5; i++) {
      await page.clock.pauseAt((await page.evaluate(() => Date.now())) + 60000);
      const at = await page.evaluate(() => Date.now());
      await wait(50);
      if (await page.evaluate(() => Date.now()) === at) { return; }
    }
    throw new Error('the page clock did not pause');
  };
  const centre = async (sel) => {
    const b = await page.locator(sel).boundingBox();
    return [b.x + b.width / 2, b.y + b.height / 2];
  };
  // A raw press and release of the mouse (no wait for the page's frames).
  const press = async ([x, y]) => { await page.mouse.move(x, y); await page.mouse.down(); await page.mouse.up(); };
  const sample = () => page.evaluate(() => window.__shepSample());
  const count = () => page.evaluate(() => (window.ShepDx ? window.ShepDx.gen() : null));
  const marks = () => page.evaluate(() => {
    const m = document.querySelector('#shep-dx-results-body [data-shep-gen]');
    const g = document.getElementById('shep-dx-answer-gate');
    return { mark: m ? m.getAttribute('data-shep-gen') : null,
             gate: g ? (g.textContent.match(/data-shep-gen="(\d+)"/) || [])[1] : null };
  });
  const fill = async () => {
    await page.fill(HPO, 'HP:0001250');
    await page.fill(GENES, 'SCN1A, PTEN');
    await settle();
  };
  const finish = async (out, name) => {
    await settle();
    out.end = await sample();
    await page.evaluate(() => { window.__shepSampling = false; });
    const samples = await page.evaluate(() => window.__shepSamples);
    out.samples = samples.length;
    out.ever = {};
    for (const k of ['table', 'details', 'count', 'badge', 'download', 'changed']) {
      out.ever[k] = samples.some((s) => s[k]);
    }
    // From the first sample that draws the whole answer on (Gradio writes its parts one by one).
    const whole = (s) => s.table && s.details && s.count && s.badge && s.download && !s.changed;
    const first = samples.findIndex(whole);
    out.after_drawn = first < 0 ? null : samples.slice(first).every(whole);
    out.raw = samples.map((s) => [s.table, s.details, s.count, s.badge, s.download, s.changed]);
    out.genes = await page.inputValue(GENES);
    out.hpo = await page.inputValue(HPO);
    await page.screenshot({ path: `${shots}/${name}-end-1850x960.png` });
    return out;
  };
  const out = {};
  try {
    // typed_then_run (REV-R3-b)
    await fill();
    if (fs.existsSync(gate)) { fs.unlinkSync(gate); }
    let asks = calls(), answered = await runsAnswered();
    let at = await centre('#shep-dx-run-btn');
    await page.focus(GENES);
    await page.keyboard.press('End');
    await pause();
    const before = await count();
    await page.keyboard.type(' Q');
    const typed = await count();
    await press(at);
    await until(() => calls() > asks, 'the run asked');
    let o = { counted: [before, typed] };
    await page.clock.resume();
    await page.evaluate(() => { window.__shepSamples = []; window.__shepSampling = true; });
    fs.writeFileSync(gate, '');
    await until(async () => (await runsAnswered()) > answered, 'the run answered');
    await settle();
    o.at_answer = await sample();
    o.marks = await marks();
    await page.screenshot({ path: `${shots}/typed_then_run-at-answer-1850x960.png` });
    out.typed_then_run = await finish(o, 'typed_then_run');
    // pasted_at_answer and cleared_at_answer (REV-R3-a)
    for (const name of ['pasted_at_answer', 'cleared_at_answer']) {
      await fill();
      fs.unlinkSync(gate);
      asks = calls();
      answered = await runsAnswered();
      await page.click('#shep-dx-run-btn');
      await until(() => calls() > asks, 'the run asked');
      at = await centre('#shep-dx-clear');
      await page.evaluate(() => navigator.clipboard.writeText('MECP2'));
      await page.focus(GENES);
      await page.keyboard.press('ControlOrMeta+A');
      await pause();
      o = { before: await count() };
      await page.evaluate(() => { window.__shepSamples = []; window.__shepSampling = true; });
      if (name === 'pasted_at_answer') { await page.keyboard.press('ControlOrMeta+V'); } else { await press(at); }
      o.counted = await count();
      fs.writeFileSync(gate, '');
      await until(async () => (await runsAnswered()) > answered, 'the run answered');
      o.at_answer = await sample();
      o.marks = await marks();
      await page.screenshot({ path: `${shots}/${name}-at-answer-1850x960.png` });
      await page.clock.resume();
      out[name] = await finish(o, name);
    }
  } finally {
    fs.writeFileSync(gate, '');
    await browser.close();
  }
  out.errors = errors;
  console.log(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_an_edit_counts_in_its_own_event_before_an_answer_or_a_run_that_follows_it(
        monkeypatch, tmp_path):
    """REV-R3-a, REV-R3-b (the review of 961d901), with the page's clock
    paused for the edit and the press, so only what runs in the edit's own
    event can count it: the page's count goes up at the edit; Run Diagnosis
    pressed just after typing carries the text and the count of that edit,
    and its answer is drawn and offered and stays so, the session keeping it;
    an answer let go just after a paste, or just after Clear, is never drawn
    nor offered under the edited boxes, at its arrival or after. On 961d901
    the count went up only once Gradio dispatched the box's change (none while
    the clock stood still): the typed-then-Run answer was hidden, then
    cleared ("Inputs changed"), and the pasted one was drawn, its download
    offered, under MECP2."""
    from src.webui.app import mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    demo, _mode, gate, asked, calls = _served_page(monkeypatch, tmp_path)
    (run,) = [fn._id for fn in demo.fns.values() if fn.name == "_run_diagnosis"]
    shots = tmp_path / "shots"
    shots.mkdir()
    try:
        with served(demo, **mount_options()) as base:
            seen = run_node(_AT_ONCE_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS))
                            .replace("__SAMPLER__", _SAMPLER_JS), tmp_path,
                            base[:-len("gradio_api")], str(gate), str(asked), str(shots),
                            str(run))
    finally:
        gate.write_text("")
    assert seen["errors"] == [], seen["errors"]
    # Typed, then Run at once: the run read the typed text, and its answer is
    # drawn and offered from its arrival on, and kept.
    typed = seen["typed_then_run"]
    assert calls[0]["candidate_genes"] == ["SCN1A", "PTEN", "Q"], calls
    assert typed["genes"] == "SCN1A, PTEN Q", typed
    for when in ("at_answer", "end"):
        drawn = typed[when]
        assert drawn["table"] and drawn["details"] and drawn["count"] and drawn["badge"], (
            when, typed)
        assert drawn["download"] and not drawn["changed"], (when, typed)
    assert typed["after_drawn"] is True, typed
    # An answer let go just after a paste or Clear: never drawn nor offered.
    for case, genes, hpo in (("pasted_at_answer", "MECP2", "HP:0001250"),
                             ("cleared_at_answer", "", "")):
        out = seen[case]
        assert (out["genes"], out["hpo"]) == (genes, hpo), (case, out)
        assert out["samples"] > 0, (case, out)
        assert not any(out["ever"][k] for k in ("table", "details", "count", "badge",
                                                "download")), (case, out)
        assert out["at_answer"]["changed"] and out["end"]["changed"], (case, out)
        assert not out["end"]["table"], (case, out)
    assert [c["candidate_genes"] for c in calls[1:]] == [["SCN1A", "PTEN"]] * 2, calls
    # Each edit was counted in its own event, the clock standing still: two
    # typed characters, a paste, a click on Clear; the typed-then-Run answer
    # is marked with the count of both characters, and the gate is its own.
    before, after = typed["counted"]
    assert after == before + 2, typed
    assert typed["marks"] == {"mark": str(after), "gate": str(after)}, typed
    for case in ("pasted_at_answer", "cleared_at_answer"):
        assert seen[case]["counted"] == seen[case]["before"] + 1, (case, seen[case])
