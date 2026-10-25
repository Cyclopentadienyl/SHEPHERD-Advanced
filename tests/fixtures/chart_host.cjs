// Drives the real shep_charts.js through its public mount(), emulating the
// Gradio 6.30 semantics the module relies on (gradio 6.30.0's
// frontend/assets/HTML-*.js, core-*.js and index-*.js `component_server`):
// - `props` is a snapshot whose `value` is replaced BEFORE the watch callbacks
//   run, and they run in a microtask only when the value's JSON changed (an
//   identical tick calls nothing);
// - a server function takes the one argument as it is, has no AbortSignal,
//   and a failed call RESOLVES undefined (the client catches and warns).
// Timers are fake so the retry path can be stepped; the DOM is just enough for
// the module's cards, toolbar and message line; uPlot is a stub.
//
// usage: node tests/fixtures/chart_host.cjs <path to shep_charts.js> [scenario ...]
// prints {scenario: {ok, ...facts}} as JSON; tests/unit/test_webui_charts.py
// runs one scenario per test.
"use strict";
const vm = require("vm"), fs = require("fs");

function el(tag) {
  const e = {
    tag, attrs: {}, children: [], hidden: false, _text: "", _html: "", className: "",
    clientWidth: 400, clientHeight: 200, isConnected: true,
    classList: { set: new Set(), toggle(c, on) { on ? this.set.add(c) : this.set.delete(c); },
                 add(c) { this.set.add(c); }, remove(c) { this.set.delete(c); },
                 contains(c) { return this.set.has(c); } },
    setAttribute(k, v) { this.attrs[k] = String(v); }, getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
    addEventListener() {}, appendChild(c) { this.children.push(c); c.parent = this; return c; },
    remove() {}, contains() { return true; }, closest() { return null; },
    querySelectorAll() { return []; },
    get textContent() { return this._text; }, set textContent(v) { this._text = String(v); },
    get innerHTML() { return this._html; },
    set innerHTML(v) {
      this._html = String(v); this.children = [];
      if (this._html.indexOf("shep-chart-plot") >= 0) {     // a card: its parts
        const card = this, plot = el("div");
        plot.closest = (sel) => (sel === ".shep-chart-plot" ? plot : sel === ".shep-chart" ? card : null);
        this.parts = { ".shep-chart-plot": plot, ".shep-chart-tip": el("div"),
                       ".shep-chart-keys": el("span") };
      }
    },
    querySelector(sel) { return (this.parts && this.parts[sel]) || null; },
  };
  return e;
}

function makeHost(modulePath, initialToken) {
  // ---- fake timers
  let now = 0, nextId = 1; const timers = new Map();
  const fakeSetTimeout = (fn, ms) => { const id = nextId++; timers.set(id, { at: now + (ms || 0), fn }); return id; };
  const fakeClearTimeout = (id) => { timers.delete(id); };
  async function advance(ms) {
    const end = now + ms;
    for (;;) {
      let best = null;
      for (const [id, t] of timers) { if (t.at <= end && (!best || t.at < best[1].at)) { best = [id, t]; } }
      if (!best) { break; }
      timers.delete(best[0]); now = best[1].at; best[1].fn(); await settle();
    }
    now = end;
  }
  // ---- DOM: the template's parts the module reads
  const msg = el("div"); msg._text = "Charts loading…";
  const grid = el("div"), zoom = el("span"), cue = el("button");
  zoom.hidden = true; cue.hidden = true;
  const root = el("div"); const listeners = {};
  root.addEventListener = (type, fn) => { (listeners[type] = listeners[type] || []).push(fn); };
  Object.assign(root.attrs, { "data-series-fn": "series", "data-csv-fn": "csv", "data-empty": "No run to show yet." });
  root.querySelector = (s) => ({ ".shep-charts-msg": msg, ".shep-charts-grid": grid,
                                 ".shep-charts-zoom": zoom, ".shep-charts-new": cue })[s] || null;
  const element = el("div"); element.querySelector = (s) => (s === ".shep-charts" ? root : null);
  // ---- uPlot stub
  function UPlot(opts, data) {
    this.opts = opts; this.data = data; this.width = opts.width; this.height = opts.height;
    this.scales = { x: {}, y: {} }; this.over = el("div"); this.root = el("div");
  }
  UPlot.prototype.setData = function (d) { this.data = d; };
  UPlot.prototype.setScale = function (k, r) { this.scales[k] = { min: r.min, max: r.max }; };
  UPlot.prototype.batch = function (fn) { fn(); };
  UPlot.prototype.setSize = function (s) { this.width = s.width; this.height = s.height; };
  UPlot.prototype.destroy = function () {};
  const store = new Map();
  const win = { uPlot: UPlot, matchMedia: () => ({ matches: false }),
                sessionStorage: { getItem: (k) => (store.has(k) ? store.get(k) : null), setItem: (k, v) => store.set(k, String(v)) } };
  const ctx = { window: win, document: { createElement: (t) => el(t) }, console, JSON, Object, Array, Math, String,
                Promise, Error, setTimeout: fakeSetTimeout, clearTimeout: fakeClearTimeout,
                AbortController, requestAnimationFrame: () => 0, cancelAnimationFrame() {},
                ResizeObserver: class { observe() {} unobserve() {} disconnect() {} } };
  vm.createContext(ctx);
  vm.runInContext(fs.readFileSync(modulePath, "utf8"), ctx, { filename: modulePath });
  // ---- Gradio 6.30's props and watch
  const O = { value: initialToken };
  const clone = (v) => (v === undefined ? undefined : JSON.parse(JSON.stringify(v)));
  let applied = clone(O);
  const props = clone(O);
  const watchers = [];
  const watch = (keys, cb) => watchers.push({ keys: Array.isArray(keys) ? keys : [keys], cb });
  function setToken(v) {                                  // a tick's output for the figure
    O.value = v;
    if (JSON.stringify(applied) === JSON.stringify(O)) { return; }
    const changed = [];
    for (const k in O) { if (JSON.stringify(props[k]) !== JSON.stringify(O[k])) { changed.push(k); } props[k] = clone(O[k]); }
    applied = clone(O);
    if (changed.length) {
      queueMicrotask(() => { for (const w of watchers) { if (w.keys.some((k) => changed.includes(k))) { w.cb(); } } });
    }
  }
  // ---- server functions, 6.30's client: each call waits for the test to answer it
  const calls = [];
  const call = (fn) => (...args) => {
    const t = args.length === 1 ? args[0] : args;
    return new Promise((resolve, reject) => { calls.push({ fn, token: clone(t), resolve, reject }); });
  };
  const server = { series: call("series"), csv: call("csv") };
  win.ShepCharts.mount(element, props, watch, server, "training");
  const h = {
    key(k, chartIdx) {                                     // a key press on a chart's plot
      const inst = win.ShepCharts._inst.training;
      const st = inst.charts[inst.order[chartIdx || 0]];
      const e = { key: k, target: st.plot, preventDefault() {}, ctrlKey: false, metaKey: false, altKey: false };
      for (const fn of listeners.keydown || []) { fn(e); }
    },
    click(act) {                                          // a toolbar button
      const btn = el("button"); btn.attrs["data-chart-act"] = act;
      btn.closest = (sel) => (sel === "[data-chart-act]" ? btn : null);
      for (const fn of listeners.click || []) { fn({ target: btn, preventDefault() {} }); }
    },
    win, calls, msg, zoom, store, setToken, advance, root,
    series() { return calls.filter((c) => c.fn === "series"); },
    last() { const s = h.series(); return s[s.length - 1]; },
    exports() { return calls.filter((c) => c.fn === "csv").map((c) => c.token.run); },
    answer(c, data) { c.answered = true; c.resolve(data); return settle(); },
    fail(c) { c.answered = true; c.resolve(undefined); return settle(); },  // 6.30's client on any error
    reject(c) { c.answered = true; c.reject(new Error("rejected")); return settle(); },  // e.g. its check before the try
    async failEvery(n) {                                  // let each retry come, and fail it
      for (let i = 0; i < n; i++) { await advance(3000); const c = h.last(); if (!c.answered) { await h.fail(c); } }
    },
    pending() { return root.attrs["aria-busy"] === "true"; },
    view() { return win.ShepCharts.view("training"); },
    stored(run) { return win.ShepCharts._views.load({ key: "training" }, run); },
    state() {
      const v = win.ShepCharts.view("training");
      return { run: v ? v.run : null, rev: v ? v.rev : null, charts: v ? v.order.length : 0,
               epochs: v && v.order.length ? v.charts[v.order[0]].n : 0,
               message: msg.hidden ? "" : msg.textContent,
               zoom: zoom.hidden ? "" : zoom.textContent, pending: h.pending() };
    },
    timers() { return timers.size; },
  };
  return h;
}

function settle() { return new Promise((r) => setImmediate(r)); }

function payload(run, rev, epochs) {
  const x = Array.from({ length: epochs }, (_, i) => i + 1);
  return { run, rev, charts: [
    { id: "loss", title: "Loss", x, series: [{ label: "train", values: x.map((i) => 1 / i) }], yScale: "linear" },
    { id: "lr", title: "Learning rate", x, series: [{ label: "learning rate", values: x.map(() => 1e-3) }], yScale: "log" },
  ] };
}
const NOT_LISTED = { run: null, rev: 0, charts: [], empty: "This run is no longer listed." };
const tok = (run, rev, empty) => (run ? { run, rev } : { run: null, rev: 0, empty: empty || "No run to show yet." });
const TRYING = "The charts could not be loaded. Trying again…";
const GAVE_UP = "The charts could not be loaded.";

async function shown(run, rev, epochs) {               // a host with `run` drawn
  const h = makeHost(MODULE, tok(run, rev)); await settle();
  await h.answer(h.series()[0], payload(run, rev, epochs));
  return h;
}

let MODULE = null;
const S = {};
// ---------------------------------------------------------------- positive controls
S.control_new_epochs_are_drawn = async () => {         // the run drawn grows; an identical tick asks nothing
  const h = await shown("ws/A", 2, 2);
  const first = h.state();
  h.setToken(tok("ws/A", 3)); await settle();
  const during = h.state();
  await h.answer(h.series()[1], payload("ws/A", 3, 3));
  const after = h.state();
  h.setToken(tok("ws/A", 3)); await settle();
  return { ok: first.run === "ws/A" && first.epochs === 2 && !during.pending && during.epochs === 2 &&
               after.run === "ws/A" && after.epochs === 3 && after.message === "" && !after.pending &&
               h.series().length === 2, first, during, after, asked: h.series().length };
};
S.control_switch_runs = async () => {                  // A drawn, B asked: busy until B is drawn
  const h = await shown("ws/A", 2, 2);
  h.setToken(tok("ws/B", 5)); await settle();
  const during = h.state();
  await h.answer(h.series()[1], payload("ws/B", 5, 5));
  const s = h.state();
  return { ok: during.pending && s.run === "ws/B" && s.epochs === 5 && s.message === "" && !s.pending,
           during, s };
};
S.control_superseded_rev_is_dropped = async () => {    // rev 1 pending, rev 2 asked: rev 1's reply never drawn
  const h = makeHost(MODULE, tok("ws/A", 1)); await settle();
  h.setToken(tok("ws/A", 2)); await settle();
  await h.answer(h.series()[1], payload("ws/A", 2, 2));
  await h.answer(h.series()[0], payload("ws/A", 1, 1));
  const s = h.state();
  return { ok: s.run === "ws/A" && s.epochs === 2 && s.rev === 2 && !s.pending, s };
};
S.control_real_not_listed = async () => {              // the server's own answer: said, chart cleared
  const h = await shown("ws/A", 2, 2);
  h.setToken(tok("ws/gone", 4)); await settle();
  await h.answer(h.series()[1], NOT_LISTED);
  const s = h.state();
  return { ok: s.run === null && s.charts === 0 && s.message === "This run is no longer listed." && !s.pending, s };
};
S.control_zoom_survives_clear_and_return = async () => {   // zoom on A, clear, back to A: A's zoom comes back
  const h = await shown("ws/A", 9, 9);
  h.key("x"); h.key("ArrowLeft");
  const zoomed = h.view().x;
  await h.advance(1000);                                  // the view written (250 ms)
  h.setToken(tok(null, 0)); await settle();
  h.setToken(tok("ws/A", 9)); await settle();
  await h.answer(h.series()[1], payload("ws/A", 9, 9));
  const v = h.view();
  return { ok: v.run === "ws/A" && v.xManual === true && v.x[0] === zoomed[0] && v.x[1] === zoomed[1] &&
               h.state().zoom !== "", zoomed, back: v.x, note: h.state().zoom };
};
// ---------------------------------------------------------------- P2-5: a reply for what is no longer asked
S.reviewer_pending_then_cleared = async () => {        // A pending -> cleared -> A's reply
  const h = makeHost(MODULE, tok("ws-A/run-A", 1)); await settle();
  h.setToken(tok(null, 0, "Choose a run from workspace B")); await settle();
  await h.answer(h.series()[0], payload("ws-A/run-A", 1, 1));
  const s = h.state();
  return { ok: s.run === null && s.charts === 0 && s.message === "Choose a run from workspace B" && !s.pending, s };
};
S.shown_A_pending_B_back_to_A = async () => {          // A drawn -> B pending -> A -> B's reply
  const h = await shown("ws/A", 2, 2);
  h.setToken(tok("ws/B", 7)); await settle();
  h.setToken(tok("ws/A", 2)); await settle();
  const back = h.state();
  await h.answer(h.series()[1], payload("ws/B", 7, 7));
  const s = h.state();
  return { ok: !back.pending && s.run === "ws/A" && s.epochs === 2 && !s.pending && h.series().length === 2,
           back, s, asked: h.series().length };
};
S.shown_A_pending_B_then_cleared = async () => {
  const h = await shown("ws/A", 2, 2);
  h.setToken(tok("ws/B", 7)); await settle();
  h.setToken(tok(null, 0, "No run to show yet.")); await settle();
  await h.answer(h.series()[1], payload("ws/B", 7, 7));
  const s = h.state();
  return { ok: s.run === null && s.charts === 0 && s.message === "No run to show yet." && !s.pending, s };
};
S.cleared_A_pending_B_pending_A_reply_late = async () => {  // A then B asked; A's reply after B's
  const h = makeHost(MODULE, tok(null, 0)); await settle();
  h.setToken(tok("ws/A", 1)); await settle();
  h.setToken(tok("ws/B", 3)); await settle();
  await h.answer(h.series()[1], payload("ws/B", 3, 3));
  await h.answer(h.series()[0], payload("ws/A", 1, 1));
  const s = h.state();
  return { ok: s.run === "ws/B" && s.epochs === 3 && !s.pending, s };
};
S.pending_cleared_then_same_run_again = async () => {  // A pending -> cleared -> A again: the newer ask is drawn
  const h = makeHost(MODULE, tok("ws/A", 1)); await settle();
  h.setToken(tok(null, 0)); await settle();
  h.setToken(tok("ws/A", 1)); await settle();
  const asked = h.series().length;
  await h.answer(h.series()[0], payload("ws/A", 1, 1));  // the older call: dropped by the sequence number
  const mid = h.state();
  await h.answer(h.series()[1], payload("ws/A", 1, 1));
  const s = h.state();
  return { ok: asked === 2 && mid.pending && s.run === "ws/A" && s.epochs === 1 && s.message === "" && !s.pending,
           asked, mid, s };
};
// Two calls outstanding for the identical token: the token held now cannot
// tell them apart (`stillAsked` passes both), and the server's answer can
// change between them (training_chart_series answers "no longer listed" once
// the run is not listed, and its rev is the rows read when it answers). Only
// the sequence number keeps the older answer from being drawn last.
S.same_token_newer_answer_not_listed = async () => {   // A pending, cleared, A again; newer: not listed, older: A late
  const h = makeHost(MODULE, tok("ws/A", 1)); await settle();
  h.setToken(tok(null, 0)); await settle();
  h.setToken(tok("ws/A", 1)); await settle();
  const asked = h.series().length;
  await h.answer(h.series()[1], NOT_LISTED);
  const mid = h.state();
  await h.answer(h.series()[0], payload("ws/A", 1, 1));
  const s = h.state();
  return { ok: asked === 2 && mid.message === "This run is no longer listed." && s.run === null && s.charts === 0 &&
               s.message === "This run is no longer listed." && !s.pending, asked, mid, s };
};
S.same_token_older_read_answers_last = async () => {   // {A, rev 3} asked twice; newer read 4 rows, older 3 rows late
  const h = makeHost(MODULE, tok("ws/A", 3)); await settle();
  h.setToken(tok(null, 0)); await settle();
  h.setToken(tok("ws/A", 3)); await settle();
  const asked = h.series().length;
  await h.answer(h.series()[1], payload("ws/A", 4, 4));  // the log grew between the two reads
  const mid = h.state();
  await h.answer(h.series()[0], payload("ws/A", 3, 3));
  const s = h.state();
  return { ok: asked === 2 && mid.epochs === 4 && s.run === "ws/A" && s.epochs === 4 && s.rev === 4 && !s.pending,
           asked, mid, s };
};
S.reply_queued_before_the_watch_callback = async () => {   // B's reply and the switch back to A in one task
  const h = await shown("ws/A", 2, 2);
  h.setToken(tok("ws/B", 7)); await settle();
  h.series()[1].resolve(payload("ws/B", 7, 7));          // the reply's callback is queued first ...
  h.setToken(tok("ws/A", 2));                            // ... then Gradio's watch microtask
  for (let i = 0; i < 4; i++) { await Promise.resolve(); }
  const transient = h.state();
  await settle();
  const s = h.state();
  return { ok: transient.run === "ws/A" && s.run === "ws/A" && s.epochs === 2 && !s.pending &&
               h.series().length === 2, transient, s, asked: h.series().length };
};
S.superseded_failure_after_clear_is_ignored = async () => {  // A pending fails after the clear: no retry, no message
  const h = makeHost(MODULE, tok("ws/A", 1)); await settle();
  h.setToken(tok(null, 0, "Pick a run")); await settle();
  await h.fail(h.series()[0]);
  const right = h.state(), timers = h.timers();          // nothing drawn, so a timer would be a retry
  await h.advance(10000);
  const s = h.state();
  return { ok: right.message === "Pick a run" && timers === 0 && s.message === "Pick a run" &&
               h.series().length === 1 && !s.pending, right, timers, s, asked: h.series().length };
};
S.superseded_failure_after_switch_back_is_ignored = async () => {
  const h = await shown("ws/A", 2, 2);
  h.setToken(tok("ws/B", 7)); await settle();
  h.setToken(tok("ws/A", 2)); await settle();
  await h.fail(h.series()[1]);
  await h.advance(10000);
  const s = h.state();
  return { ok: s.run === "ws/A" && s.message === "" && h.series().length === 2 && !s.pending, s,
           asked: h.series().length };
};
S.export_names_the_run_shown = async () => {           // settled after A -> B pending -> A: chart, token, export agree
  const h = await shown("ws/A", 2, 2);
  h.setToken(tok("ws/B", 7)); await settle();
  h.setToken(tok("ws/A", 2)); await settle();
  await h.answer(h.series()[1], payload("ws/B", 7, 7));
  h.click("csv");
  const s = h.state();
  return { ok: s.run === "ws/A" && !s.pending && JSON.stringify(h.exports()) === JSON.stringify(["ws/A"]),
           s, exported: h.exports() };
};
// ---------------------------------------------------------------- the pending cue
S.pending_while_another_run_loads = async () => {      // A drawn, B asked: busy, and Export CSV asks for B
  const h = await shown("ws/A", 2, 2);
  h.setToken(tok("ws/B", 7)); await settle();
  h.click("csv");
  const s = h.state();
  return { ok: s.pending && JSON.stringify(h.exports()) === JSON.stringify(["ws/B"]), s, exported: h.exports() };
};
S.pending_stays_while_no_reply_comes = async () => {   // no timeout on the call: still busy, never A under B
  const h = await shown("ws/A", 2, 2);
  h.setToken(tok("ws/B", 7)); await settle();
  await h.advance(10 * 60 * 1000);
  const s = h.state();
  return { ok: s.pending && h.series().length === 2, s, asked: h.series().length };
};
S.pending_replaces_the_cleared_line = async () => {    // cleared, then A asked: busy, not "No run to show yet."
  const h = makeHost(MODULE, tok(null, 0, "No run to show yet.")); await settle();
  const cleared = h.state();
  h.setToken(tok("ws/A", 2)); await settle();
  const s = h.state();
  return { ok: !cleared.pending && cleared.message === "No run to show yet." && s.pending, cleared, s };
};
S.pending_on_first_load = async () => {                // the first ask is busy too; drawn, it is not
  const h = makeHost(MODULE, tok("ws/A", 2)); await settle();
  const first = h.state();
  await h.answer(h.series()[0], payload("ws/A", 2, 2));
  const s = h.state();
  return { ok: first.pending && !s.pending && s.run === "ws/A", first, s };
};
S.pending_view_buttons_do_nothing = async () => {      // while busy, the hidden run's view is not changed
  const h = await shown("ws/A", 9, 9);
  h.key("x");                                           // zoom A, written after 250 ms
  await h.advance(1000);
  const before = JSON.stringify(h.stored("ws/A"));
  h.setToken(tok("ws/B", 4)); await settle();
  h.click("zoom-in"); h.click("zoom-out"); h.click("reset"); h.key("x"); h.key("0");
  await h.advance(1000);
  const during = JSON.stringify(h.stored("ws/A"));
  await h.answer(h.series()[1], payload("ws/B", 4, 4));
  h.click("zoom-in");                                   // drawn again: the buttons work
  const after = h.state();
  return { ok: before !== "null" && during === before && /zoomed/.test(after.zoom) && !after.pending,
           before, during, after };
};
S.new_epochs_never_mark_pending = async () => {        // no flicker while a run trains
  const h = await shown("ws/A", 2, 2);
  h.setToken(tok("ws/A", 3)); await settle();
  const during = h.pending();
  await h.answer(h.series()[1], payload("ws/A", 3, 3));
  return { ok: !during && h.state().epochs === 3, during };
};
S.retry_keeps_trying_again = async () => {             // a retry is not busy: "Trying again…" stays readable
  const h = makeHost(MODULE, tok("ws/A", 2)); await settle();
  await h.fail(h.series()[0]);
  await h.advance(3000);
  const s = h.state();
  return { ok: h.series().length === 2 && !s.pending && s.message === TRYING, s };
};
// ---------------------------------------------------------------- failures
S.failed_call_is_not_not_listed = async () => {        // 6.30 answers undefined on error: retried, said
  const h = makeHost(MODULE, tok("ws/A", 2)); await settle();
  await h.fail(h.series()[0]);
  const s = h.state();
  await h.advance(3000);
  const asked = h.series().length;
  await h.answer(h.series()[1], payload("ws/A", 2, 2));
  const after = h.state();
  return { ok: s.message === TRYING && !s.pending && asked === 2 && after.run === "ws/A" && after.message === "",
           s, asked, after };
};
S.rejected_call_is_a_failure = async () => {           // a call that rejects (6.30's check before its try)
  const h = makeHost(MODULE, tok("ws/A", 2)); await settle();
  await h.reject(h.series()[0]);
  const s = h.state();
  await h.advance(3000);
  return { ok: s.message === TRYING && !s.pending && h.series().length === 2, s, asked: h.series().length };
};
S.rejected_call_after_clear_is_ignored = async () => { // A pending rejects after the clear: no retry, no message
  const h = makeHost(MODULE, tok("ws/A", 1)); await settle();
  h.setToken(tok(null, 0, "Pick a run")); await settle();
  await h.reject(h.series()[0]);
  const right = h.state(), timers = h.timers();          // nothing drawn, so a timer would be a retry
  await h.advance(10000);
  const s = h.state();
  return { ok: right.message === "Pick a run" && timers === 0 && s.message === "Pick a run" &&
               h.series().length === 1 && !s.pending, right, timers, s, asked: h.series().length };
};
S.rejected_call_after_switch_back_is_ignored = async () => {  // A drawn, B pending, A again, B rejects
  const h = await shown("ws/A", 2, 2);
  h.setToken(tok("ws/B", 7)); await settle();
  h.setToken(tok("ws/A", 2)); await settle();
  await h.reject(h.series()[1]);
  await h.advance(10000);
  const s = h.state();
  return { ok: s.run === "ws/A" && s.epochs === 2 && s.message === "" && h.series().length === 2 && !s.pending,
           s, asked: h.series().length };
};
S.failed_switch_does_not_keep_old_run = async () => {  // A drawn, B's call fails: A is not left under B
  const h = await shown("ws/A", 2, 2);
  h.setToken(tok("ws/B", 5)); await settle();
  await h.fail(h.series()[1]);
  const s = h.state();
  await h.failEvery(25);
  const out = h.state();
  return { ok: s.run === null && s.charts === 0 && s.message === TRYING && !s.pending &&
               out.run === null && out.message === GAVE_UP && h.timers() === 0 && h.series().length === 22,
           s, out, asked: h.series().length, timers: h.timers() };
};
S.retry_budget_is_per_token = async () => {            // the budget spent on B; C still gets its retries
  const h = makeHost(MODULE, tok("ws/B", 5)); await settle();
  await h.fail(h.series()[0]);
  await h.failEvery(25);
  const before = h.series().length;
  h.setToken(tok("ws/C", 1)); await settle();
  await h.fail(h.last());
  const s = h.state();
  await h.advance(3000);
  return { ok: h.series().length - before === 2 && s.message === TRYING, retried: h.series().length - before, s };
};
S.failed_new_epochs_keep_the_run = async () => {       // the run drawn, newer rows fail: its curves stay, retried
  const h = await shown("ws/A", 2, 2);
  h.setToken(tok("ws/A", 3)); await settle();
  await h.fail(h.series()[1]);
  const s = h.state();
  await h.advance(3000);
  await h.answer(h.series()[2], payload("ws/A", 3, 3));
  const after = h.state();
  return { ok: s.run === "ws/A" && s.epochs === 2 && s.message === "" && !s.pending && after.epochs === 3,
           s, after };
};
S.retry_cancelled_by_switch = async () => {            // B failed (retry waiting), A asked: no B call later
  const h = await shown("ws/A", 2, 2);
  h.setToken(tok("ws/B", 7)); await settle();
  await h.fail(h.series()[1]);
  h.setToken(tok("ws/A", 2)); await settle();
  await h.answer(h.series()[2], payload("ws/A", 2, 2));
  await h.advance(10000);
  const s = h.state();
  const bAfter = h.series().slice(2).filter((c) => c.token.run === "ws/B").length;
  return { ok: s.run === "ws/A" && s.message === "" && bAfter === 0 && !s.pending, s, bAfter };
};
// ---------------------------------------------------------------- nothing drawn
S.cleared_hides_zoom_note = async () => {              // zoomed on A, then cleared: no "zoomed" note left
  const h = await shown("ws/A", 9, 9);
  h.key("x");
  const zoomedNote = h.state().zoom;
  h.setToken(tok(null, 0, "No run to show yet.")); await settle();
  const s = h.state();
  return { ok: zoomedNote !== "" && s.zoom === "" && s.message === "No run to show yet.", zoomedNote, s };
};
S.cleared_keeps_last_zoom_of_run = async () => {       // a zoom 100 ms before the clear is kept for the run
  const h = await shown("ws/A", 9, 9);
  await h.advance(1000);                                  // the first view written
  h.key("x");                                             // saveView in 250 ms
  const zoomed = h.view().x;
  await h.advance(100);
  h.setToken(tok(null, 0)); await settle();
  await h.advance(1000);
  const kept = h.stored("ws/A");
  return { ok: !!(kept && kept.x && kept.x[0] === zoomed[0] && kept.x[1] === zoomed[1]), zoomed, kept };
};

(async () => {
  const [modulePath, ...names] = process.argv.slice(2);
  MODULE = modulePath;
  const run = names.length ? names : Object.keys(S);
  const out = {};
  for (const n of run) {
    if (!S[n]) { out[n] = { ok: false, error: "no such scenario" }; continue; }
    try { out[n] = await S[n](); } catch (e) { out[n] = { ok: false, error: String((e && e.stack) || e) }; }
  }
  console.log(JSON.stringify(out, null, 1));
  process.exitCode = Object.values(out).every((r) => r.ok) ? 0 : 1;
})();
