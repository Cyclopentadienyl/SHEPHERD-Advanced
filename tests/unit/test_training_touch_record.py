"""
A form's record of the fields the operator touched: every edit, an emptied field included (OPS-R1)
==================================================================================================
`src/webui/components/training_console.py` (`record_touches`, `TOUCH_HEAD`;
decision S2: explicit is what the operator touched). The external review of
46a5c2b, OPS-R1: Neighbours per hop emptied as the operator's first action on
it (select all, Backspace) was never recorded, so Start sent nothing for it
and the run trained with the reference while the field stood empty. Gradio
6.30's text box dispatches ``.input`` from a keypress alone, so an edit with
no keypress (Backspace, Delete, a cut, a paste) fired ``.change`` only, which
a page's own write fires too. What these pin:

- **every field whose touches a form records is marked for the page's touch
  script**, and the dashboard's head carries that script;
- **in a real browser** (Node with Playwright; skipped without), on a page
  of fields of each kind the record covers, wired by `record_touches` and
  served with the dashboard's head: emptying a text box (Backspace, Delete, a
  cut), a paste and a typed value each put the field's key in the record
  that a click on Start sent at once, before the field was left, and so does
  emptying a number box; the first fill, a page's own write of every value
  and a Tab through every field put nothing there;
- **on the Training page itself** (the dashboard as the service mounts it,
  a demo workspace, the job runner's start replaced by a recorder): Neighbours
  per hop emptied as the first action and Start clicked at once is refused in
  the shared check's words, beside Start and in the form's line, and nothing
  starts; so it is when the field is left first; Reset to reference empties
  the record and Start then sends no setting; a Tab through every field
  records nothing; the reference's own value typed is sent as the operator's.

The Fine-tune page and the Continue drawer's Branch fields record through the
same `record_touches` (their wiring: `test_webui_finetune_page.py`,
`test_training_console_resume.py`).

Module: tests/unit/test_training_touch_record.py
"""
from __future__ import annotations

import json
import subprocess

import pytest

gr = pytest.importorskip("gradio")

from src.webui.components import training_console as tc  # noqa: E402
from tests.fixtures.operator_access import operator_mode  # noqa: E402


def _browser():
    from tests.fixtures.browser import node_with_playwright

    found = node_with_playwright()
    if found is None:
        pytest.skip("Node with Playwright is not installed")
    return found


def _run(script: str, tmp_path, *args: str) -> dict:
    node, env = _browser()
    path = tmp_path / "touch.cjs"
    path.write_text(script, encoding="utf-8")
    result = subprocess.run([node, str(path), *args], env=env, capture_output=True, text=True,
                            timeout=400)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


# --------------------------------------------------------------------------- the wiring
def test_a_recorded_field_is_marked_for_the_touch_script_the_dashboard_loads():
    """`record_touches` marks the field it records (`TOUCH_CLASS`, kept beside
    the field's own classes), the touch script acts on that class alone and on
    trusted input events alone, and the dashboard's head carries it after the
    shell's script."""
    from src.webui.app import dashboard_head

    with gr.Blocks():
        record = tc.touched_record("rec")
        hops = gr.Textbox(value="15, 10, 5", max_lines=1, elem_id="hops")
        own = gr.Number(value=1, elem_id="own", elem_classes=["mine"])
        tc.record_touches(hops, "num_neighbors", record)
        tc.record_touches(own, "num_epochs", record)
    assert hops.elem_classes == [tc.TOUCH_CLASS]
    assert own.elem_classes == ["mine", tc.TOUCH_CLASS]
    assert f"closest('.{tc.TOUCH_CLASS}')" in tc.TOUCH_HEAD
    assert "e.isTrusted" in tc.TOUCH_HEAD and "key: 'Unidentified'" in tc.TOUCH_HEAD
    head = dashboard_head()
    assert head.count(tc.TOUCH_HEAD) == 1 and head.endswith(tc.TOUCH_HEAD)


# --------------------------------------------------------------------------- every edit, in a browser
#: A page of the field kinds the record covers, each wired by `record_touches`
#: as the forms wire theirs: a hop list and an "off" number (text boxes, as
#: `value_field` builds them), a whole number and a real one (number boxes).
#: Start writes what its own request carried; Reset is a page's own write of
#: the record and every value, as the forms' Reset is.
FIELDS = {"hops": "15, 10, 5", "off": "off", "whole": 200, "real": 0.0001}


def _edits_page():
    with gr.Blocks() as demo:
        record = tc.touched_record("rec")
        fields = {"hops": gr.Textbox(value=FIELDS["hops"], max_lines=1, elem_id="hops"),
                  "off": gr.Textbox(value=FIELDS["off"], max_lines=1, elem_id="off"),
                  "whole": gr.Number(value=FIELDS["whole"], precision=0, elem_id="whole"),
                  "real": gr.Number(value=FIELDS["real"], elem_id="real")}
        for key, field in fields.items():
            tc.record_touches(field, key, record)
        start = gr.Button("Start", elem_id="start")
        reset = gr.Button("Reset", elem_id="reset")
        sent = gr.Textbox(value="", elem_id="sent")
        start.click(lambda rec, *values: json.dumps({"record": rec, "values": values}),
                    inputs=[record, *fields.values()], outputs=[sent])
        reset.click(lambda: ([], "", *FIELDS.values()), outputs=[record, sent, *fields.values()])
    return demo


#: Each case on the one page: Reset (a page's own write), the gesture, then a
#: click on Start (Playwright's, the field not left unless the case says so),
#: and what Start's request carried. The first case is the first fill alone.
_EDITS_JS = r"""
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const context = await browser.newContext({ viewport: { width: 1850, height: 960 }, deviceScaleFactor: 1 });
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  const page = await context.newPage();
  await page.goto(process.argv[2], { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('#start', { timeout: 60000 });
  const sent = () => page.locator('#sent textarea, #sent input').first();
  const selectAll = () => page.keyboard.press('Control+A');
  const gestures = {
    'first fill': null,
    'Backspace': async () => { await selectAll(); await page.keyboard.press('Backspace'); },
    'Delete': async () => { await selectAll(); await page.keyboard.press('Delete'); },
    'cut': async () => { await selectAll(); await page.keyboard.press('Control+X'); },
    'paste': async () => {
      await page.evaluate(() => navigator.clipboard.writeText('3'));
      await selectAll(); await page.keyboard.press('Control+V');
    },
    'typed': async () => { await selectAll(); await page.keyboard.type('7'); },
  };
  const cases = [['first fill', null, false]];
  for (const field of ['hops', 'off']) {
    for (const name of ['Backspace', 'Delete', 'cut', 'paste', 'typed']) { cases.push([name, field, false]); }
  }
  cases.push(['Backspace', 'hops', true], ['Backspace', 'whole', false], ['Backspace', 'real', false]);
  cases.push(['tab through', null, false]);
  const out = [];
  for (const [name, field, leave] of cases) {
    if (out.length) {
      await page.click('#reset');
      await page.waitForFunction(() => {
        const box = document.querySelector('#sent textarea, #sent input');
        return box.value === '' && document.querySelector('#hops input').value === '15, 10, 5'
          && document.querySelector('#whole input').value === '200';
      });
    }
    if (name === 'tab through') {
      await page.locator('#hops input').focus();
      for (let i = 0; i < 6; i++) { await page.keyboard.press('Tab'); }
    } else if (field) {
      await page.locator(`#${field} input`).click();
      await gestures[name]();
      if (leave) { await page.keyboard.press('Tab'); }
    }
    await page.click('#start');
    await page.waitForFunction(() => document.querySelector('#sent textarea, #sent input').value !== '');
    out.push({ name, field, leave, sent: JSON.parse(await sent().inputValue()) });
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_every_edit_of_a_recorded_field_reaches_the_record_before_start_and_no_write_does(
        tmp_path):
    """OPS-R1, at the shared listener: on 46a5c2b the text boxes' Backspace,
    Delete, cut and paste cases sent an empty record (Start then sent nothing
    for the emptied field). Now each edit of each kind puts its field's key,
    and only it, in the record Start's own request carries, Start clicked at
    once (the field never left) or after a Tab; the values sent are the
    field's as left (empty, "3", "7", None for an emptied number box). The
    first fill, Reset's write of every value and a Tab through every field
    put nothing there."""
    from src.webui.app import mount_options
    from tests.fixtures.served_page import served

    _browser()
    with served(_edits_page(), **mount_options()) as base:
        seen = _run(_EDITS_JS, tmp_path, base[:-len("gradio_api")])
    order = list(FIELDS)
    left = {"Backspace": "", "Delete": "", "cut": "", "paste": "3", "typed": "7"}
    for case in seen:
        record, values = case["sent"]["record"], case["sent"]["values"]
        name, field = case["name"], case["field"]
        if field is None:
            assert record == [], case
            assert values == list(FIELDS.values()), case
            continue
        assert record == [field], case
        expected = dict(FIELDS)
        expected[field] = (None if field in ("whole", "real") else left[name])
        assert values == [expected[key] for key in order], case
    assert [case["name"] for case in seen if case["field"] is None] == ["first fill",
                                                                        "tab through"]
    assert len(seen) == 15


# --------------------------------------------------------------------------- the Training page
@pytest.fixture
def workspace_root(tmp_path_factory, monkeypatch):
    """A workspaces root holding one demo workspace (the service's list)."""
    from tests.fixtures.registered_cohort import write_demo_workspace

    root = tmp_path_factory.mktemp("touchroot")
    write_demo_workspace(root / "demo")
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root))
    return root


#: The Training page, Research › Training of the dashboard: the steps of the
#: cases, each Start's notice (its answer) and the form's line after it, and
#: the touch-recorded fields a Tab through the form reached.
_TRAINING_JS = r"""
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1850, height: 960 }, deviceScaleFactor: 1 })).newPage();
  await page.goto(process.argv[2], { waitUntil: 'domcontentloaded' });
  await page.click('#shep-tab-research-button', { timeout: 60000 });
  await page.waitForSelector('#shep-train-start:not([disabled])', { timeout: 60000 });
  const text = (id) => page.locator(`#${id}`).innerText();
  const hops = page.locator('#num_neighbors input');
  const startAndRead = async (expected) => {
    await page.click('#shep-train-start');
    await page.waitForFunction((words) => document.querySelector('#shep-train-notice').innerText.includes(words),
                               expected, { timeout: 30000 });
    return { notice: await text('shep-train-notice'), line: await text('shep-train-setby') };
  };
  const reset = async () => {
    await page.click('#shep-train-reset');
    await page.waitForFunction(() => document.querySelector('#num_neighbors input').value === '15, 10, 5'
      && !document.querySelector('#shep-train-setby').innerText.includes('Not valid'));
  };
  const refused = 'Neighbours per hop needs at least 1 item(s).';
  const out = {};
  await page.getByText('More training settings', { exact: true }).click();
  // (a) emptied as the first action on it, Start clicked at once.
  await hops.click();
  await page.keyboard.press('Control+A');
  await page.keyboard.press('Backspace');
  out.at_once = await startAndRead(refused);
  // (d) Reset to reference, then Start.
  await reset();
  out.after_reset = await startAndRead('Start 1 recorded');
  // (b) emptied, the field left, then Start.
  await hops.click();
  await page.keyboard.press('Control+A');
  await page.keyboard.press('Backspace');
  await page.keyboard.press('Tab');
  out.left_first = await startAndRead(refused);
  // (e) a Tab through every field of the form, the Model's too, then Start.
  await reset();
  await page.getByText('Model', { exact: true }).click();
  await page.locator('#num_epochs input').focus();
  const reached = [];
  for (let i = 0; i < 400; i++) {
    const at = await page.evaluate(() => {
      const el = document.activeElement;
      const field = el && el.closest ? el.closest('.shep-touch') : null;
      return { field: field ? field.id : null,
               inside: !!(el && el.closest && el.closest('#shep-train-params')),
               reset: !!(el && el.closest && el.closest('#shep-train-reset')) };
    });
    if (at.field && !reached.includes(at.field)) { reached.push(at.field); }
    if (at.reset || !at.inside) { break; }
    await page.keyboard.press('Tab');
  }
  out.reached = reached;
  out.tabbed = await startAndRead('Start 2 recorded');
  // (c) the reference's own value typed.
  await hops.click();
  await page.keyboard.press('Control+A');
  await page.keyboard.type('15, 10, 5');
  out.typed = await startAndRead('Start 3 recorded');
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_the_training_form_refuses_a_hop_list_emptied_first_and_records_no_page_write(
        tmp_path, workspace_root, monkeypatch):
    """OPS-R1 on the Training page as the service mounts it. On 46a5c2b the
    first case started a run with the reference's hops. Now: Neighbours per
    hop emptied as the operator's first action and Start clicked at once:
    Start's own request carries the key, the start is refused in the shared
    check's words beside Start and in the form's line, and the job runner is
    never asked; so it is when the field is left first. Reset to reference
    then Start: no key and no setting sent. A Tab through every field the
    form records (its More and Model fields included): no key. The
    reference's own value typed: the key, sent as the operator's."""
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.served_page import served

    _browser()
    operator_mode(monkeypatch)
    monkeypatch.setattr(tc, "load_runtime_settings", lambda *a, **k: {})
    asked = []

    def start_training(config, **kwargs):
        asked.append(config)
        return {"success": False,
                "error": f"Start {len(asked)} recorded by the test. Nothing was started."}

    monkeypatch.setattr(tc.training_manager, "start_training", start_training)
    carried = []
    original = tc._start

    def start(ws, run_name, device, values, touched):
        carried.append((list(touched or []),
                        dict(zip(tc.base_form_keys(), values, strict=True))))
        return original(ws, run_name, device, values, touched)

    monkeypatch.setattr(tc, "_start", start)
    demo = create_gradio_app()
    with served(demo, **mount_options()) as base:
        seen = _run(_TRAINING_JS, tmp_path, base[:-len("gradio_api")])

    refusal = "Not started: Neighbours per hop needs at least 1 item(s)."
    for case in ("at_once", "left_first"):
        assert seen[case]["notice"].strip().endswith(refusal), seen[case]
        assert "Not valid: Neighbours per hop needs at least 1 item(s)." in seen[case]["line"]
    assert [touched for touched, _values in carried] == [
        ["num_neighbors"], [], ["num_neighbors"], [], ["num_neighbors"]]
    assert carried[0][1]["num_neighbors"] == "" == carried[2][1]["num_neighbors"]
    assert carried[1][1]["num_neighbors"] == "15, 10, 5"
    # The job runner was asked only by the three starts the form let through:
    # after Reset and after the Tab, with no learning setting; the typed
    # reference, with it.
    run_keys = {"data_dir", "learning_stage", "device", "run_name", "compile"}
    assert len(asked) == 3
    assert set(asked[0]) == set(asked[1]) == run_keys
    assert set(asked[2]) == run_keys | {"num_neighbors"} and asked[2]["num_neighbors"] == [15, 10, 5]
    assert "Set by you: nothing" in seen["after_reset"]["line"]
    assert "Neighbours per hop 15, 10, 5 (reference 15, 10, 5)" in seen["typed"]["line"]
    # The Tab reached every field the form records and shows.
    shown = {block.elem_id for block in _base_fields() if block.visible is not False}
    assert set(seen["reached"]) == shown, sorted(shown ^ set(seen["reached"]))


def _base_fields():
    """The base form's fields as the page builds them, in `base_form_keys` order."""
    from src.webui import shell

    with gr.Blocks() as demo:
        tc.create_training_tab(shell.standalone_context(demo))
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    return [by_id[key] if key in by_id else by_id[f"spec-{key.replace('.', '-')}"]
            for key in tc.base_form_keys()]
