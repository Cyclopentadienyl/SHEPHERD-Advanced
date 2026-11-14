"""
A form's record of the fields the operator touched: every edit, an emptied field included, before any later click (OPS-R1)
==========================================================================================================================
`src/webui/components/training_console.py` (`record_touches`, `TOUCH_HEAD`;
decision S2: explicit is what the operator touched). The external review of
46a5c2b, OPS-R1: Neighbours per hop emptied as the operator's first action on
it (select all, Backspace) was never recorded, so Start sent nothing for it
and the run trained with the reference while the field stood empty. Gradio
6.30's text box dispatches ``.input`` from a keypress alone, so an edit with
no keypress (Backspace, Delete, a cut, a paste) fired ``.change`` only, which
a page's own write fires too. Its narrow re-review of 214b523 (OPS-R1's
residual): the edit's input reaches the record two of the text box's ticks
after the edit, and a click in between carried the record as it was, so
Start could still run the reference for a field the operator had just set.
What these pin:

- **every field whose touches a form records is marked for the page's touch
  script** (its class and its record's and key's mark), only the kinds whose
  every edit is one input are taken, and the dashboard's head carries that
  script;
- **the three forms that keep a record wire every field through
  `record_touches`**, as their pages build them: the Training base form, the
  Fine-tune page and the Continue drawer's Branch fields, each field's first
  input listener adding its own key to its own form's record, and the field
  marked; **each field's touch listener is followed by its acknowledgement**
  (`LANDED_JS`, naming its own record and key), which the click waits for;
  **it is the field's one input listener and what else the edit does is
  chained after it** (the line, the Branch check; the Fine-tune tasks'
  restart after their drop), so a deferred run ("always_last") never runs
  the touch listener again;
- **on the Training page, a line answered late never records the field
  again** (the review of 9a6895f): two keys with the line held on the server,
  Reset, then the line let go: two acknowledgements for two keys, the line
  says nothing is set and Start sends no setting (on 9a6895f a third
  acknowledgement put the key back and the reference was sent as the
  operator's);
- **in a real browser, with the edit's processing held** (the page's clock
  paused, so the text box's and the radio's ticks cannot pass): a click on
  Start, Reset or a task, or Enter on a task, made while an edit is pending
  sends nothing; once the clock runs, exactly one request leaves, after the
  edit has reached the record (the key and the new value; an emptied field,
  a new value, the reference retyped, a radio's choice), Reset leaves the
  record empty and a task's switch drops its keys after the edit, in the
  operator's order; a number box's edit is in the record within its own
  event, so nothing waits for it; on the Training page itself, the emptied
  Neighbours per hop is refused in the shared check's words, '9, 8, 7' and
  the reference retyped are sent as the operator's, and Reset clicked while
  the edit is pending leaves nothing set;
- **in a real browser** (Node with Playwright; skipped without), on a page
  of fields of each kind the record covers, wired by `record_touches` and
  served with the dashboard's head: emptying a text box (Backspace, Delete, a
  cut), a paste and a typed value each put the field's key in the record
  that a click on Start sent at once, before the field was left, and so does
  emptying a number box; the first fill, a page's own write of every value
  and a Tab through every field put nothing there; a switch toggled with
  Enter (Gradio's checkbox takes it as a toggle with no input event of its
  own) or Space puts its key there too, Enter while another edit is pending
  included;
- **on the Training page itself** (the dashboard as the service mounts it,
  a demo workspace, the job runner's start replaced by a recorder): Neighbours
  per hop emptied as the first action and Start clicked at once is refused in
  the shared check's words, beside Start and in the form's line, and nothing
  starts; so it is when the field is left first; Reset to reference empties
  the record and Start then sends no setting; a Tab through every field
  records nothing; the reference's own value typed is sent as the operator's.

What else a field's input does on each form is pinned with the form: the
"Set by you" line in `test_training_page.py` and
`test_webui_finetune_page.py`, the Branch fields' check that goes stale
(`_on_branch_edited`) in `test_training_console_resume.py`.

Module: tests/unit/test_training_touch_record.py
"""
from __future__ import annotations

import json
import subprocess
import time

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
    """`record_touches` marks the field it records (`TOUCH_CLASS` and its
    record's and key's `touch_mark`, kept beside the field's own classes),
    takes only the kinds whose every edit is one input of theirs (a dropdown
    is refused: its filter's keypresses are no input of it, so a click would
    wait for them for ever), the touch script acts on that class alone, gives
    trusted input events alone a keypress and holds trusted activations
    alone, and the dashboard's head carries it after the shell's script."""
    from src.webui.app import dashboard_head

    with gr.Blocks():
        record = tc.touched_record("rec")
        hops = gr.Textbox(value="15, 10, 5", max_lines=1, elem_id="hops")
        own = gr.Number(value=1, elem_id="own", elem_classes=["mine"])
        tc.record_touches(hops, "num_neighbors", record)
        tc.record_touches(own, "num_epochs", record)
        with pytest.raises(TypeError, match="Dropdown"):
            tc.record_touches(gr.Dropdown(choices=["a"]), "device", record)
    assert tc.touch_mark(record, "num_neighbors") == "shep-touch:rec:num_neighbors"
    assert hops.elem_classes == [tc.TOUCH_CLASS, "shep-touch:rec:num_neighbors"]
    assert own.elem_classes == ["mine", tc.TOUCH_CLASS, "shep-touch:rec:num_epochs"]
    assert f"FIELD = '.{tc.TOUCH_CLASS}', MARK = '{tc.TOUCH_CLASS}:'" in tc.TOUCH_HEAD
    assert f"HOLD_MS = {tc.TOUCH_HOLD_MS};" in tc.TOUCH_HEAD and "__" not in tc.TOUCH_HEAD
    assert "e.isTrusted" in tc.TOUCH_HEAD and "key: 'Unidentified'" in tc.TOUCH_HEAD
    head = dashboard_head()
    assert head.count(tc.TOUCH_HEAD) == 1 and head.endswith(tc.TOUCH_HEAD)


def _three_forms():
    """The three forms that keep a record, as their pages build them: the
    Blocks, and by record each recorded field by its key."""
    from src.webui import shell
    from src.webui.components import finetune_page as fp

    with gr.Blocks() as demo:
        ctx = shell.standalone_context(demo)
        tc.create_training_tab(ctx)
        fp.create_finetune_page(ctx)
    by_id = {getattr(block, "elem_id", None): block for block in demo.blocks.values()}
    expected = {
        "shep-train-touched": {key: by_id[key] if key in by_id
                               else by_id[f"spec-{key.replace('.', '-')}"]
                               for key in tc.base_form_keys()},
        "ft_touched": {name: by_id[f"ft_{name}"] for name in fp.FINETUNE_FIELDS},
        "shep-train-branch-touched": {name: by_id[f"branch_{name}"]
                                      for name in tc.BRANCH_FIELDS},
    }
    return demo, expected


def test_the_three_forms_record_every_field_through_the_one_listener_and_mark_it():
    """Decision S2 on the three forms that keep a record, as their pages
    build them: the Training base form (`shep-train-touched`, one field per
    `base_form_keys`), the Fine-tune page (`ft_touched`, `FINETUNE_FIELDS`)
    and the Continue drawer's Branch fields (`shep-train-branch-touched`,
    `BRANCH_FIELDS`). Each field's one input listener adds its own key to
    its own form's record in the browser (`TOUCH_JS`, no request), and the
    field carries `TOUCH_CLASS`, without which the touch script leaves an
    emptied text box unrecorded (OPS-R1); no other block carries the class.
    A field left unwired or wired by a listener of its own fails here: a
    start of that form would send nothing for an edit of it and use the
    reference's or the parent's value; so does a field with another input
    listener beside it (what else an edit does is chained after the touch
    listener: `test_what_else_an_edit_does_is_chained_after_the_record_write`)."""
    demo, expected = _three_forms()
    before, after = tc.TOUCH_JS.split("{key}")
    fns = list(demo.fns.values())
    recorded = {record: {} for record in expected}
    for fn in fns:
        if not (isinstance(fn.js, str) and fn.js.startswith(before) and fn.js.endswith(after)):
            continue
        (record,) = fn.outputs
        ((target, event),) = fn.targets
        assert fn.fn is None and fn.inputs == [record] and event == "input", fn.js
        assert fn.queue is False and fn.trigger_mode == "multiple", fn.js
        key = json.loads(fn.js[len(before):-len(after)])
        recorded[record.elem_id][key] = demo.blocks[target]
    for record, fields in expected.items():
        assert {key: field._id for key, field in recorded[record].items()} == {
            key: field._id for key, field in fields.items()}, record
        for key, field in fields.items():
            assert tc.TOUCH_CLASS in field.elem_classes, (record, key)
            (only,) = [fn for fn in fns if (field._id, "input") in fn.targets]
            assert only.js == tc.TOUCH_JS.replace("{key}", json.dumps(key)), (record, key)
    marked = {block._id for block in demo.blocks.values()
              if tc.TOUCH_CLASS in (getattr(block, "elem_classes", None) or [])}
    assert marked == {field._id for fields in expected.values() for field in fields.values()}


def test_every_recorded_field_of_the_three_forms_acknowledges_its_record():
    """OPS-R1's residual, the wiring the click waits on: on the three forms,
    every recorded field carries its own mark (`touch_mark`: its record and
    its key), and its touch listener is followed by exactly one
    acknowledgement (`LANDED_JS` naming that mark: js alone, no request, no
    input, no output), which Gradio dispatches after the record is written.
    A field whose acknowledgement is missing, names another record or key,
    or is chained after another listener, fails here: the page would hold a
    click until the bound and then drop it, or release it before that
    field's edit reached the record."""
    demo, expected = _three_forms()
    fns = list(demo.fns.values())
    before, after = tc.LANDED_JS.split("{mark}")
    landed = [fn for fn in fns
              if isinstance(fn.js, str) and fn.js.startswith(before) and fn.js.endswith(after)]
    assert len(landed) == sum(len(fields) for fields in expected.values())
    for record_id, fields in expected.items():
        record = next(block for block in demo.blocks.values()
                      if getattr(block, "elem_id", None) == record_id)
        for key, field in fields.items():
            mark = tc.touch_mark(record, key)
            assert field.elem_classes[-2:] == [tc.TOUCH_CLASS, mark], (record_id, key)
            touch = next(fn for fn in fns if (field._id, "input") in fn.targets)
            (ack,) = [fn for fn in fns if fn.trigger_after == touch._id
                      and isinstance(fn.js, str) and fn.js.startswith(before)]
            assert ack.js == tc.LANDED_JS.replace("{mark}", json.dumps(mark)), (record_id, key)
            assert ack.fn is None and ack.inputs == [] and ack.outputs == [], (record_id, key)
            assert ack.queue is False and not ack.trigger_only_on_success, (record_id, key)


def test_what_else_an_edit_does_is_chained_after_the_record_write():
    """The review of 9a6895f (P2): Gradio 6.30 runs a deferred listener
    ("always_last") again by dispatching its trigger again, so a second input
    listener of a recorded field (the line on what the operator set, the
    Branch check) ran every listener of the field's input once more when its
    answer came: the touch listener put the key back with no edit, after a
    task switch or Reset made since had dropped it, and a start sent it.
    Now, on the three forms, a recorded field's input has one listener, its
    touch listener, and what else the edit does is chained after it, each
    sending its last value ("always_last"; a deferred run of it runs it
    alone): the base form's and the Fine-tune page's line (the hop list's
    with its note), the Branch fields' check gone stale. So are the
    Fine-tune tasks: their input's one listener drops the task's own keys,
    and their restart is chained after it. A listener wired back beside the
    touch or the drop listener fails here."""
    from src.webui.components import finetune_page as fp

    demo, expected = _three_forms()
    fns = list(demo.fns.values())
    landed = tc.LANDED_JS.split("{mark}")[0]
    by_id = {getattr(block, "elem_id", None): block for block in demo.blocks.values()}
    said = {"shep-train-touched": {"shep-train-setby"}, "ft_touched": {"ft_setby"},
            "shep-train-branch-touched": {"continue_preview"}}
    for record_id, fields in expected.items():
        for key, field in fields.items():
            (touch,) = [fn for fn in fns if (field._id, "input") in fn.targets]
            (step,) = [fn for fn in fns if fn.trigger_after == touch._id
                       and not (isinstance(fn.js, str) and fn.js.startswith(landed))]
            assert step.fn is not None and step.trigger_mode == "always_last", (record_id, key)
            assert said[record_id] <= {block.elem_id for block in step.outputs}, (record_id, key)
    tasks = by_id["ft_tasks"]
    (drop,) = [fn for fn in fns if (tasks._id, "input") in fn.targets]
    assert drop.js == tc.DROP_JS.replace("{keys}", json.dumps(sorted(fp.TASK_OWN_FIELDS)))
    (restart,) = [fn for fn in fns if fn.trigger_after == drop._id]
    assert restart.fn is fp._on_tasks_input


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


# --------------------------------------------------------------------------- a click waits for the edits
#: A page of each kind of field a record covers (a text box, a number box, a
#: radio), each wired by `record_touches` as the forms wire theirs, a task
#: choice whose change drops a key from the record (`record_drops`, as the
#: Fine-tune tasks drop a task's own settings), Start, which writes what its
#: own request carried, and Reset, a page's write of the record, every value
#: and the tasks.
HELD_FIELDS = {"hops": "15, 10, 5", "whole": 200, "sched": "none"}


def _sent_record(record, *values):
    """What Start's own request carried."""
    return json.dumps({"record": record, "values": values})


def _held_page():
    with gr.Blocks() as demo:
        record = tc.touched_record("rec")
        fields = {"hops": gr.Textbox(value=HELD_FIELDS["hops"], max_lines=1, elem_id="hops"),
                  "whole": gr.Number(value=HELD_FIELDS["whole"], precision=0, elem_id="whole"),
                  "sched": gr.Radio(choices=["none", "cosine"], value=HELD_FIELDS["sched"],
                                    elem_id="sched")}
        for key, field in fields.items():
            tc.record_touches(field, key, record)
        tasks = gr.CheckboxGroup(choices=["a", "b"], value=["a"], interactive=True,
                                 elem_id="tasks")
        tc.record_drops(tasks, ["hops"], record)
        start = gr.Button("Start", elem_id="start")
        reset = gr.Button("Reset", elem_id="reset")
        sent = gr.Textbox(value="", elem_id="sent")
        start.click(_sent_record, inputs=[record, *fields.values()], outputs=[sent])
        reset.click(lambda: ([], "", *HELD_FIELDS.values(), ["a"]),
                    outputs=[record, sent, *fields.values(), tasks])
    return demo, record, start, reset


#: Each case on the one page, from Reset (the page's clock running): the
#: field selected, **the page's clock paused** (Playwright's: no timeout and
#: no animation frame runs, so a text box's and a radio's ticks cannot pass
#: and their input cannot reach the record), the edit, then the activation (a
#: raw press and release of the mouse, or Enter on a task); after 300 ms of
#: real time, the requests that left and the touch script's state; then the
#: clock runs again, and what Start's request carried (Reset's and the tasks'
#: cases click Start after their own write, the clock running).
_HELD_JS = r"""
const { chromium } = require('playwright');
(async () => {
  const fns = JSON.parse(process.argv[3]);
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1850, height: 960 }, deviceScaleFactor: 1 })).newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.clock.install();
  await page.goto(process.argv[2], { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('#start', { timeout: 60000 });
  const joins = [];
  page.on('request', (r) => { if (r.url().includes('/queue/join')) { joins.push(JSON.parse(r.postData()).fn_index); } });
  const count = (fn) => joins.filter((i) => i === fn).length;
  // Every request made so far seen (a request's event can reach this script after the page's answer to it).
  const until = async (check) => { for (let i = 0; i < 500 && !check(); i++) { await new Promise((r) => setTimeout(r, 20)); } };
  const expect = { start: 0, reset: 0 };
  const state = () => page.evaluate(() => (window.ShepTouch && window.ShepTouch.state ? window.ShepTouch.state() : null));
  const sent = () => page.locator('#sent textarea, #sent input').first().inputValue();
  const reset = async () => {
    await page.click('#reset');
    await page.waitForFunction(() => document.querySelector('#sent textarea, #sent input').value === ''
      && document.querySelector('#hops input').value === '15, 10, 5'
      && document.querySelector('#whole input').value === '200');
  };
  const centre = async (locator) => { const b = await locator.boundingBox(); return [b.x + b.width / 2, b.y + b.height / 2]; };
  const press = async ([x, y]) => { await page.mouse.move(x, y); await page.mouse.down(); await page.mouse.up(); };
  // The page's clock paused, and seen to stand still (a target already passed under load would not pause it).
  const pause = async () => {
    for (let i = 0; i < 5; i++) {
      await page.clock.pauseAt((await page.evaluate(() => Date.now())) + 60000);
      const at = await page.evaluate(() => Date.now());
      await new Promise((r) => setTimeout(r, 50));
      if (await page.evaluate(() => Date.now()) === at) { return; }
    }
    throw new Error('the page clock did not pause');
  };
  const startAndRead = async () => {
    await page.click('#start');
    await page.waitForFunction(() => document.querySelector('#sent textarea, #sent input').value !== '');
    return JSON.parse(await sent());
  };
  const enter = async () => {
    await page.evaluate(() => document.querySelector('#tasks input[name="b"]').focus());
    await page.keyboard.press('Enter');
  };
  const targets = {
    start: page.locator('#start'), reset: page.locator('#reset'),
    task: page.locator('#tasks label', { has: page.locator('input[name="b"]') }),
    radio: page.locator('#sched [data-testid="cosine-radio-label"]'),
  };
  const backspace = () => page.keyboard.press('Backspace');
  const cases = [
    ['emptied', '#hops input', backspace, 'start'],
    ['typed', '#hops input', () => page.keyboard.type('9, 8, 7'), 'start'],
    ['reference', '#hops input', () => page.keyboard.type('15, 10, 5'), 'start'],
    ['number', '#whole input', () => page.keyboard.type('5'), 'start'],
    ['radio', null, 'radio', 'start'],
    ['reset', '#hops input', backspace, 'reset'],
    ['task click', '#hops input', backspace, 'task'],
    ['task enter', '#hops input', backspace, null],
  ];
  const settled = () => {
    const s = window.ShepTouch && window.ShepTouch.state ? window.ShepTouch.state() : { held: 0, counted: {} };
    return !s.held && !Object.keys(s.counted).length;
  };
  const out = {};
  for (const [name, field, edit, act] of cases) {
    await reset();
    expect.reset += 1;
    await until(() => count(fns.start) === expect.start && count(fns.reset) === expect.reset);
    const at = { radio: await centre(targets.radio), act: act ? await centre(targets[act]) : null };
    if (field) { await page.locator(field).click(); await page.keyboard.press('Control+A'); }
    const before = { start: count(fns.start), reset: count(fns.reset) };
    await pause();
    if (edit === 'radio') { await press(at.radio); } else { await edit(); }
    if (act) { await press(at.act); } else { await enter(); }
    await new Promise((r) => setTimeout(r, 300));
    const held = { start: count(fns.start) - before.start, reset: count(fns.reset) - before.reset, state: await state() };
    await page.clock.resume();
    let record;
    if (name === 'reset' || name.startsWith('task')) {
      await page.waitForFunction((n) => (n === 'reset'
        ? document.querySelector('#hops input').value === '15, 10, 5'
        : document.querySelectorAll('#tasks input:checked').length === 2), name);
      await page.waitForFunction(settled);
      record = await startAndRead();
    } else {
      await page.waitForFunction(() => document.querySelector('#sent textarea, #sent input').value !== '', null, { timeout: 10000 });
      record = JSON.parse(await sent());
    }
    expect.start += 1;
    expect.reset += name === 'reset' ? 1 : 0;
    await until(() => count(fns.start) >= expect.start && count(fns.reset) >= expect.reset);
    out[name] = { held, sent: record, start: count(fns.start) - before.start, reset: count(fns.reset) - before.reset,
                  tasks: await page.evaluate(() => [...document.querySelectorAll('#tasks input:checked')].map((b) => b.name)) };
  }
  out.errors = errors;
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_a_click_made_while_an_edit_is_pending_waits_for_its_record(tmp_path):
    """OPS-R1's residual, with the order of events controlled: the page's
    clock is paused, so an edit of a text box or a radio is made but its
    input cannot reach the record (the box's ticks wait on a timeout or an
    animation frame), and the operator then clicks. On 214b523 Start's request
    left at once with the record as it was (``[]`` beside the emptied field):
    the run would take the reference. Now nothing leaves while the edit is
    pending (one click held, the edit counted under its field's mark); once
    the clock runs, exactly one request leaves, with the key and the new
    value: emptied, a new value, the reference's own value retyped, a radio's
    choice. A number box's edit is in the record within its own event, so
    Start goes at once. Reset clicked while an edit is pending goes after it,
    and the record is empty; a task chosen while an edit is pending (a click,
    or Enter on it) drops its key after the edit's, so the record is empty,
    in the operator's order (on 214b523 the drop came first and the edit's
    key stayed)."""
    from src.webui.app import mount_options
    from tests.fixtures.served_page import served

    _browser()
    demo, record, start, reset = _held_page()
    fns = {"start": next(fn._id for fn in demo.fns.values() if fn.name == "_sent_record"),
           "reset": next(fn._id for fn in demo.fns.values()
                         if fn.targets == [(reset._id, "click")])}
    with served(demo, **mount_options()) as base:
        seen = _run(_HELD_JS, tmp_path, base[:-len("gradio_api")], json.dumps(fns))
    # Nothing left while an edit was pending: Start and Reset were held.
    assert [(name, seen[name]["held"]["start"], seen[name]["held"]["reset"])
            for name in ("emptied", "typed", "reference", "radio", "reset")] == [
        ("emptied", 0, 0), ("typed", 0, 0), ("reference", 0, 0), ("radio", 0, 0), ("reset", 0, 0)]
    mark = {key: tc.touch_mark(record, key) for key in HELD_FIELDS}
    reference = list(HELD_FIELDS.values())
    for name, field, keys, typed in (("emptied", "hops", 1, ""), ("typed", "hops", 7, "9, 8, 7"),
                                     ("reference", "hops", 9, "15, 10, 5"),
                                     ("radio", "sched", 1, "cosine")):
        case = seen[name]
        # The click was held, the edit counted under its field's mark.
        assert case["held"]["state"] == {"counted": {mark[field]: keys}, "held": 1}, case
        # Then exactly one request, after the edit reached the record.
        assert case["start"] == 1, case
        values = dict(HELD_FIELDS)
        values[field] = typed
        assert case["sent"] == {"record": [field], "values": list(values.values())}, case
    number = seen["number"]
    assert number["held"] == {"start": 1, "reset": 0, "state": {"counted": {}, "held": 0}}, number
    assert number["sent"] == {"record": ["whole"], "values": ["15, 10, 5", 5, "none"]}, number
    after_reset = seen["reset"]
    assert after_reset["held"]["reset"] == 0 and after_reset["held"]["state"]["held"] == 1
    assert after_reset["reset"] == 1
    assert after_reset["sent"] == {"record": [], "values": reference}, after_reset
    for name in ("task click", "task enter"):
        case = seen[name]
        assert case["held"]["state"] == {"counted": {mark["hops"]: 1}, "held": 1}, case
        assert case["tasks"] == ["a", "b"], case
        assert case["sent"] == {"record": [], "values": ["", 200, "none"]}, case
    assert seen["errors"] == []


def _switch_page():
    """A recorded text box and a recorded switch (a checkbox, as `value_field`
    builds a bool), Start (what its own request carried) and Reset."""
    with gr.Blocks() as demo:
        record = tc.touched_record("rec")
        hops = gr.Textbox(value="15, 10, 5", max_lines=1, elem_id="hops")
        amp = gr.Checkbox(value=False, label="Mixed precision", elem_id="amp")
        tc.record_touches(hops, "hops", record)
        tc.record_touches(amp, "amp", record)
        start = gr.Button("Start", elem_id="start")
        reset = gr.Button("Reset", elem_id="reset")
        sent = gr.Textbox(value="", elem_id="sent")
        start.click(_sent_record, inputs=[record, hops, amp], outputs=[sent])
        reset.click(lambda: ([], "", "15, 10, 5", False), outputs=[record, sent, hops, amp])
    return demo


#: Each case from Reset: the switch focused, then Enter (Gradio's checkbox
#: takes it as a toggle) or Space (the browser's click); the last with an
#: edit of the text box pending (the page's clock paused) when Enter is
#: pressed. Then Start, once nothing is counted or held.
_SWITCH_JS = r"""
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1850, height: 960 }, deviceScaleFactor: 1 })).newPage();
  await page.clock.install();
  await page.goto(process.argv[2], { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('#start', { timeout: 60000 });
  const sent = () => page.locator('#sent textarea, #sent input').first().inputValue();
  const focus = () => page.evaluate(() => document.querySelector('#amp input').focus());
  const settled = () => {
    const s = window.ShepTouch && window.ShepTouch.state ? window.ShepTouch.state() : { held: 0, counted: {} };
    return !s.held && !Object.keys(s.counted).length;
  };
  const pause = async () => {
    for (let i = 0; i < 5; i++) {
      await page.clock.pauseAt((await page.evaluate(() => Date.now())) + 60000);
      const at = await page.evaluate(() => Date.now());
      await new Promise((r) => setTimeout(r, 50));
      if (await page.evaluate(() => Date.now()) === at) { return; }
    }
    throw new Error('the page clock did not pause');
  };
  const out = {};
  for (const name of ['enter', 'space', 'enter while pending']) {
    await page.click('#reset');
    await page.waitForFunction(() => document.querySelector('#sent textarea, #sent input').value === ''
      && document.querySelector('#hops input').value === '15, 10, 5' && !document.querySelector('#amp input').checked);
    if (name === 'enter while pending') {
      await page.locator('#hops input').click();
      await page.keyboard.press('Control+A');
      await pause();
      await page.keyboard.press('Backspace');
      await focus();
      await page.keyboard.press('Enter');
      out.held = await page.evaluate(() => (window.ShepTouch && window.ShepTouch.state ? window.ShepTouch.state() : null));
      await page.clock.resume();
    } else {
      await focus();
      await page.keyboard.press(name === 'enter' ? 'Enter' : ' ');
    }
    await page.waitForFunction(() => document.querySelector('#amp input').checked, null, { timeout: 10000 });
    await page.waitForFunction(settled);
    await new Promise((r) => setTimeout(r, 200));
    await page.click('#start');
    await page.waitForFunction(() => document.querySelector('#sent textarea, #sent input').value !== '');
    out[name] = JSON.parse(await sent());
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_a_switch_toggled_with_enter_is_the_operators_edit(tmp_path):
    """A recorded switch toggled from the keyboard is recorded as the
    operator's, as one clicked is. Gradio 6.30's checkbox takes Enter as a
    toggle in a keydown listener of its own, which writes the value and
    dispatches no input event and no ``.input``, so on 50c2979 the record
    stayed empty while the switch showed the new value, and Start sent
    nothing for it (the run took the reference). Space is the browser's
    click and was recorded. Now Enter on a recorded switch is made its click:
    Enter and Space each send the key with the value on; and Enter pressed
    while an edit of another field is pending is held like a click, then
    toggles and records, both keys sent."""
    from src.webui.app import mount_options
    from tests.fixtures.served_page import served

    _browser()
    with served(_switch_page(), **mount_options()) as base:
        seen = _run(_SWITCH_JS, tmp_path, base[:-len("gradio_api")])
    assert seen["enter"] == {"record": ["amp"], "values": ["15, 10, 5", True]}, seen
    assert seen["space"] == {"record": ["amp"], "values": ["15, 10, 5", True]}, seen
    assert seen["held"]["held"] == 1, seen
    assert seen["enter while pending"] == {"record": ["amp", "hops"], "values": ["", True]}, seen


# --------------------------------------------------------------------------- the Training page
@pytest.fixture
def workspace_root(tmp_path_factory, monkeypatch):
    """A workspaces root holding one demo workspace (the service's list)."""
    from tests.fixtures.registered_cohort import write_demo_workspace

    root = tmp_path_factory.mktemp("touchroot")
    write_demo_workspace(root / "demo")
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root))
    return root


def _recorded_starts(monkeypatch):
    """The Training page's Start as the operator runs it, the job runner's
    start replaced by a recorder: ``(asked, carried)``, the configs the job
    runner was asked to start (each answered "Start N recorded by the test.
    Nothing was started.") and what each Start's own request carried (the
    record and the base form's values by key)."""
    operator_mode(monkeypatch)
    monkeypatch.setattr(tc, "load_runtime_settings", lambda *a, **k: {})
    asked, carried = [], []

    def start_training(config, **kwargs):
        asked.append(config)
        return {"success": False,
                "error": f"Start {len(asked)} recorded by the test. Nothing was started."}

    monkeypatch.setattr(tc.training_manager, "start_training", start_training)
    original = tc._start

    def start(ws, run_name, device, values, touched):
        carried.append((list(touched or []),
                        dict(zip(tc.base_form_keys(), values, strict=True))))
        return original(ws, run_name, device, values, touched)

    monkeypatch.setattr(tc, "_start", start)
    return asked, carried


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
    asked, carried = _recorded_starts(monkeypatch)
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


#: The Training page with the edit's processing held (the page's clock
#: paused, as in `_HELD_JS`): Neighbours per hop selected, the clock paused,
#: the edit, then a raw press and release on Start (or Reset to reference);
#: after 300 ms of real time, Start's requests that left and the touch
#: script's state; then the clock runs and Start's answer is read (after
#: Reset, the form's line, then Start clicked with the clock running).
_HELD_TRAINING_JS = r"""
const { chromium } = require('playwright');
(async () => {
  const start = Number(process.argv[3]);
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1850, height: 960 }, deviceScaleFactor: 1 })).newPage();
  await page.clock.install();
  await page.goto(process.argv[2], { waitUntil: 'domcontentloaded' });
  await page.click('#shep-tab-research-button', { timeout: 60000 });
  await page.waitForSelector('#shep-train-start:not([disabled])', { timeout: 60000 });
  const joins = [];
  page.on('request', (r) => { if (r.url().includes('/queue/join')) { joins.push(JSON.parse(r.postData()).fn_index); } });
  const starts = () => joins.filter((i) => i === start).length;
  // Every Start request made so far seen (a request's event can reach this script after the page's answer).
  let expected = 0;
  const until = async (check) => { for (let i = 0; i < 500 && !check(); i++) { await new Promise((r) => setTimeout(r, 20)); } };
  const state = () => page.evaluate(() => (window.ShepTouch && window.ShepTouch.state ? window.ShepTouch.state() : null));
  const text = (id) => page.locator(`#${id}`).innerText();
  // Start's next answer, whatever it says (each says a new thing: a refusal, or "Start N recorded"),
  // then the form's line once it says `words` (it follows the field's input by a request of its own;
  // read as it is if it never does, for the test to say so).
  const soft = (waiting) => waiting.catch(() => null);
  let last = '';
  const answered = async (words) => {
    await page.waitForFunction((was) => {
      const now = document.querySelector('#shep-train-notice').innerText;
      return now !== was && (now.includes('Not started') || now.includes('recorded by the test'));
    }, last, { timeout: 30000 });
    last = await text('shep-train-notice');
    expected += 1;
    await soft(page.waitForFunction((w) => document.querySelector('#shep-train-setby').innerText.includes(w),
                                    words, { timeout: 15000 }));
    return { notice: last, line: await text('shep-train-setby') };
  };
  const reset = async () => {
    await page.click('#shep-train-reset');
    await page.waitForFunction(() => document.querySelector('#num_neighbors input').value === '15, 10, 5'
      && document.querySelector('#shep-train-setby').innerText.includes('Set by you: nothing'));
  };
  const centre = async (sel) => { const b = await page.locator(sel).boundingBox(); return [b.x + b.width / 2, b.y + b.height / 2]; };
  // The page's clock paused, and seen to stand still (a target already passed under load would not pause it).
  const pause = async () => {
    for (let i = 0; i < 5; i++) {
      await page.clock.pauseAt((await page.evaluate(() => Date.now())) + 60000);
      const at = await page.evaluate(() => Date.now());
      await new Promise((r) => setTimeout(r, 50));
      if (await page.evaluate(() => Date.now()) === at) { return; }
    }
    throw new Error('the page clock did not pause');
  };
  const held = async (edit, target) => {
    const at = await centre(target);
    await page.locator('#num_neighbors input').click();
    await page.keyboard.press('Control+A');
    await until(() => starts() === expected);
    const before = starts();
    await pause();
    await edit();
    await page.mouse.move(at[0], at[1]);
    await page.mouse.down();
    await page.mouse.up();
    await new Promise((r) => setTimeout(r, 300));
    const seen = { starts: starts() - before, state: await state() };
    await page.clock.resume();
    return seen;
  };
  await page.getByText('More training settings', { exact: true }).click();
  const out = {};
  out.emptied = await held(() => page.keyboard.press('Backspace'), '#shep-train-start');
  Object.assign(out.emptied, await answered('Not valid: Neighbours per hop'));
  await reset();
  out.typed = await held(() => page.keyboard.type('9, 8, 7'), '#shep-train-start');
  Object.assign(out.typed, await answered('Neighbours per hop 9, 8, 7'));
  await reset();
  out.reference = await held(() => page.keyboard.type('15, 10, 5'), '#shep-train-start');
  Object.assign(out.reference, await answered('Neighbours per hop 15, 10, 5'));
  out.reset = await held(() => page.keyboard.press('Backspace'), '#shep-train-reset');
  await page.waitForFunction(() => document.querySelector('#num_neighbors input').value === '15, 10, 5',
                             null, { timeout: 30000 });
  await soft(page.waitForFunction(() => document.querySelector('#shep-train-setby').innerText.includes('Set by you: nothing'),
                                  null, { timeout: 15000 }));
  out.reset.after = await state();
  await page.click('#shep-train-start');
  Object.assign(out.reset, await answered('Set by you: nothing'));
  await until(() => starts() >= expected);
  out.starts = starts();
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_a_start_clicked_while_the_hop_list_edit_is_pending_carries_it(
        tmp_path, workspace_root, monkeypatch):
    """OPS-R1's residual on the Training page as the service mounts it, the
    edit's processing held (the page's clock paused) while Start is clicked:
    on 214b523 Start's request left at once with the record as it was, so the
    emptied Neighbours per hop started a run with the reference [15, 10, 5].
    Now no Start request leaves while the edit is pending (the click held,
    the edit counted under the field's mark), and once the clock runs exactly
    one does, after the edit reached the record: the emptied field is refused
    in the shared check's words and nothing starts; '9, 8, 7' and the
    reference's own value retyped reach the job runner as the operator's.
    Reset to reference clicked while the edit is pending goes after it: the
    record is empty, the line says nothing is set, and Start then sends no
    setting."""
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.served_page import served

    _browser()
    asked, carried = _recorded_starts(monkeypatch)
    demo = create_gradio_app()
    start = next(fn._id for fn in demo.fns.values() if fn.name == "_on_start")
    with served(demo, **mount_options()) as base:
        seen = _run(_HELD_TRAINING_JS, tmp_path, base[:-len("gradio_api")], str(start))

    # Nothing left while the edit was pending: Start (and Reset) was held.
    assert [(case, seen[case]["starts"]) for case in ("emptied", "typed", "reference", "reset")] == [
        ("emptied", 0), ("typed", 0), ("reference", 0), ("reset", 0)], seen
    with gr.Blocks():
        mark = tc.touch_mark(tc.touched_record("shep-train-touched"), "num_neighbors")
    for case, keys in (("emptied", 1), ("typed", 7), ("reference", 9), ("reset", 1)):
        # The click was held, the edit counted under the field's mark.
        assert seen[case]["state"] == {"counted": {mark: keys}, "held": 1}, (case, seen[case])
    refusal = "Not started: Neighbours per hop needs at least 1 item(s)."
    assert seen["emptied"]["notice"].strip().endswith(refusal), seen["emptied"]
    assert "Not valid: Neighbours per hop needs at least 1 item(s)." in seen["emptied"]["line"]
    # Each held Start left once, after its edit's record: the key and the value.
    assert seen["starts"] == 4
    assert [(touched, values["num_neighbors"]) for touched, values in carried] == [
        (["num_neighbors"], ""), (["num_neighbors"], "9, 8, 7"),
        (["num_neighbors"], "15, 10, 5"), ([], "15, 10, 5")]
    assert [config.get("num_neighbors") for config in asked] == [[9, 8, 7], [15, 10, 5], None]
    assert "Neighbours per hop 9, 8, 7 (reference 15, 10, 5)" in seen["typed"]["line"]
    assert "Neighbours per hop 15, 10, 5 (reference 15, 10, 5)" in seen["reference"]["line"]
    assert seen["reset"]["after"] == {"counted": {}, "held": 0}
    assert "Set by you: nothing" in seen["reset"]["line"]


#: The Training page, one field's line held on the server (`gate`): two keys
#: in Neighbours per hop, so the first key's line is answered late and the
#: second's waits for it ("always_last"); once the touch script holds
#: nothing, Reset to reference; then the line is let go (the gate file
#: written), and once both of its answers are in, Start. The acknowledgements
#: counted, what Start's answer and the form's line say, and the line's
#: requests seen.
_REDISPATCH_JS = r"""
const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const [base, gate, line] = [process.argv[2], process.argv[3], Number(process.argv[4])];
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1850, height: 960 }, deviceScaleFactor: 1 })).newPage();
  await page.goto(base, { waitUntil: 'domcontentloaded' });
  await page.click('#shep-tab-research-button', { timeout: 60000 });
  await page.waitForSelector('#shep-train-start:not([disabled])', { timeout: 60000 });
  let asked = 0, answered = 0;
  const lineOf = (r) => { try { return JSON.parse(r.postData()).fn_index === line; } catch (e) { return false; } };
  page.on('request', (r) => { if (r.url().includes('/run/predict') && lineOf(r)) { asked += 1; } });
  page.on('requestfinished', (r) => { if (r.url().includes('/run/predict') && lineOf(r)) { answered += 1; } });
  await page.evaluate(() => {
    window.__landed = 0;
    const own = window.ShepTouch.landed;
    window.ShepTouch.landed = (mark) => { window.__landed += 1; return own(mark); };
  });
  const settled = () => { const s = window.ShepTouch.state(); return !s.held && !Object.keys(s.counted).length; };
  const until = async (check) => { for (let i = 0; i < 1500 && !check(); i++) { await new Promise((r) => setTimeout(r, 20)); } };
  await page.getByText('More training settings', { exact: true }).click();
  await page.locator('#num_neighbors input').click();
  await page.keyboard.press('Control+A');
  await page.keyboard.type('98');
  await page.waitForFunction(settled);
  await until(() => asked >= 1);
  const out = { keys: { asked, answered, landed: await page.evaluate(() => window.__landed) } };
  await page.click('#shep-train-reset');
  await page.waitForFunction(() => document.querySelector('#num_neighbors input').value === '15, 10, 5');
  fs.writeFileSync(gate, '');
  await until(() => answered >= 2);
  await page.waitForFunction(settled);
  await new Promise((r) => setTimeout(r, 500));
  out.after = { asked, answered, landed: await page.evaluate(() => window.__landed),
                line: await page.locator('#shep-train-setby').innerText() };
  await page.click('#shep-train-start');
  await page.waitForFunction(() => document.querySelector('#shep-train-notice').innerText.includes('recorded by the test'),
                             null, { timeout: 30000 });
  out.notice = await page.locator('#shep-train-notice').innerText();
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_a_late_line_answer_never_records_the_field_again_after_reset(
        tmp_path, workspace_root, monkeypatch):
    """The review of 9a6895f (P2), on the Training page as the service mounts
    it, the order of events controlled: the line of Neighbours per hop is
    held on the server, so two keys make its first request wait and its
    second wait for the first ("always_last"); Reset to reference then
    empties the record; the line is let go. On 9a6895f the line was a second
    input listener of the field, and Gradio runs a deferred listener again by
    dispatching its trigger again: every listener of the field's input ran
    once more, so the touch listener put the key back with no edit (a third
    acknowledgement for two keys), the line said the reference was set by
    the operator, and Start sent it as theirs. Now the line is chained after
    the touch listener and runs again alone: two acknowledgements for two
    keys, the line says nothing is set, and Start sends no setting."""
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.served_page import served

    _browser()
    asked, carried = _recorded_starts(monkeypatch)
    gate = tmp_path / "gate"
    own, held = tc._on_base_input, {}

    def held_line(touched, *values):
        if not held:
            held["first"] = True
            for _ in range(3000):
                if gate.exists():
                    break
                time.sleep(0.01)
        return own(touched, *values)

    monkeypatch.setattr(tc, "_on_base_input", held_line)
    demo = create_gradio_app()
    field = next(block for block in demo.blocks.values()
                 if getattr(block, "elem_id", None) == "num_neighbors")
    touch = next(fn for fn in demo.fns.values() if (field._id, "input") in fn.targets)
    line = next(fn._id for fn in demo.fns.values()
                if fn.fn is held_line and fn.trigger_after == touch._id)
    with served(demo, **mount_options()) as base:
        seen = _run(_REDISPATCH_JS, tmp_path, base[:-len("gradio_api")], str(gate), str(line))

    # Two keys: two acknowledgements, the first line request waiting.
    assert seen["keys"]["landed"] == 2 and seen["keys"]["asked"] == 1, seen
    # Its second run, once the first was answered, and nothing else.
    assert seen["after"]["asked"] == 2 == seen["after"]["answered"], seen
    assert seen["after"]["landed"] == 2, seen
    assert "Set by you: nothing" in seen["after"]["line"], seen
    assert [touched for touched, _values in carried] == [[]]
    assert len(asked) == 1 and "num_neighbors" not in asked[0], asked


def _base_fields():
    """The base form's fields as the page builds them, in `base_form_keys` order."""
    from src.webui import shell

    with gr.Blocks() as demo:
        tc.create_training_tab(shell.standalone_context(demo))
    by_id = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    return [by_id[key] if key in by_id else by_id[f"spec-{key.replace('.', '-')}"]
            for key in tc.base_form_keys()]
