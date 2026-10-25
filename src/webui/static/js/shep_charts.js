/*
 * The dashboard's one chart component (M5-2): window.ShepCharts.
 * ==============================================================
 * Draws a figure of charts with uPlot 1.6.32 (vendored, never edited:
 * /static/vendor/uplot-1.6.32/) from curve groups sent as data. The Python
 * side is src/webui/components/charts.py (`chart_figure`), which builds one
 * gr.HTML with a static template and loads this file through that
 * component's head= only.
 *
 * Data, from the component's server function (one parameter, the token):
 *   {run, rev, charts: [{id, title, x: [...], xLabel,
 *                        series: [{label, values: [...]}], yScale: "linear"|"log"}]}
 * Colours are assigned by label (never by position); the second colour is
 * also dashed. No chart has zoom code of its own: the figure holds one X view
 * (shared, so the epoch axis is one) and one Y view per chart, and
 * cursor.sync puts the cursor's epoch line on every chart.
 *
 * Lifecycle (Gradio 6.30, verified in the lifecycle lab):
 *   mount(element, props, watch, server, key) destroys any previous instance
 *   for the key (a re-mount: visible=False/True, gr.render, a reconnect
 *   reload), then: one AbortController for every listener, all on the
 *   component's root (no bare document listener); one ResizeObserver (sizes
 *   from contentRect, a 0-width report skipped while hidden, a detached
 *   element destroys the instance); uPlot.destroy() on teardown; uPlot's own
 *   mousedown/mouseup binding disabled (1.6.32 leaks a document listener when
 *   a chart is destroyed mid-press); watch('value', refresh) plus one
 *   refresh() now (updates that came while the head was loading are not
 *   replayed). A chart is created once per chart and series set, then updated
 *   by setData(data, false) with the stored view applied.
 *
 * Replies (Gradio 6.30's server functions cannot be cancelled, and a failed
 * request resolves undefined instead of rejecting): a reply is drawn only if
 * both conditions hold, each with its own job. `stillAsked`: it answers the
 * token the component holds now, so a reply for a token no longer asked is
 * dropped and a late reply never brings back a run after the selection was
 * cleared or changed, even back to the run drawn (a change may send no
 * request, and Gradio replaces the token before the watch callback runs).
 * The sequence number: it answers the newest request, so of two calls for the
 * same token (asked again after a clear or another run), the newer answer
 * wins; the token cannot tell them apart, and the server's answer can change
 * between them (the run no longer listed, more rows read). A failed call is
 * retried (each token has its own retries) and never read as "no longer
 * listed"; another run's curves never stay under it. While the
 * reply for a run other than the one drawn is outstanding, the root is
 * aria-busy="true" and CHART_CSS hides the figure with `visibility` (the
 * charts are kept, and the ResizeObserver still measures them) and says
 * "Charts loading…"; a retry and new epochs of the run drawn never set it.
 * The toolbar's view buttons do nothing meanwhile (they would change the
 * hidden run's view).
 *
 * Two uPlot 1.6.32 defects are worked around here: log axes get their own
 * tick splits (`logSplits`: 1.6.32's logAxisSplits hangs or throws for a
 * minimum that is not a power of ten, upstream #1098), and the locale guard
 * (shep_charts_locale.js, loaded before uPlot) keeps uPlot loading when
 * navigator.language is a locale Intl refuses (en-US@posix, #1053).
 *
 * Interaction: wheel zoom centred on the cursor (over the X axis X only, over
 * the Y axis Y only, inside both), drag pan, double-click and "Reset view",
 * keyboard (+ - x X y Y arrows 0 End), zoom limits (a minimum span; at most
 * 1000x the data extent, data decades + 6 on a log axis; the epoch axis never
 * starts below 0, `normX`, wherever a view is set), log axes computed
 * in log10 space with values <= 0 or missing as gaps, the view kept per run
 * in sessionStorage (the last 20 runs shown in the tab; a run not shown
 * before fits its data, its epochs and every chart's values), whole epochs
 * on the epoch axis and in the zoom note, and while zoomed during
 * a live run an overlay button "N new epochs ->". The wheel zooms in every
 * layout, narrow included; over anything but a chart's plot or axes it
 * scrolls the page, and a touch still scrolls the page in the narrow layout.
 */
(function () {
  "use strict";
  if (window.ShepCharts) { return; }

  var PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#b07d00", "#8a5cd0", "#d04a8a"];
  var DASHES = [null, [7, 4], [2, 3], [9, 3, 2, 3], [4, 4], [1, 2]];
  var WHEEL_STEP = 0.0015;
  var KEY_ZOOM = 0.8;
  var KEY_PAN = 0.1;
  var STORE_PREFIX = "shep-charts-view-v2:", STORE_RUNS = 20;
  var FONT = "14px 'IBM Plex Sans', system-ui, sans-serif";
  var INK = "#434a58", GRID = "#e6e8ee";
  var RETRY_MS = 3000, RETRY_MAX = 20;

  var INST = {};      // key -> the mounted instance
  var COLORS = {};    // key -> chart id -> {label: colour slot}; kept across re-mounts
  var created = 0, destroyed = 0;

  // ------------------------------------------------------------ formatting
  function fmt(v) {
    if (v == null || !isFinite(v)) { return ""; }
    var a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e5)) { return v.toExponential(2).replace(/\.?0+e/, "e"); }
    return String(+v.toPrecision(4));
  }
  function fmtEpoch(v) { return v == null || !isFinite(v) ? "" : String(+v.toFixed(2)); }
  // Epochs are whole: the axis steps by whole numbers (EPOCH_INCRS) and the
  // zoom note names whole epochs.
  function wholeEpoch(v) { return v == null || !isFinite(v) ? "" : String(Math.round(v)); }
  var EPOCH_INCRS = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000,
                     10000, 20000, 25000, 50000, 100000];
  function epochAxis(axis) {
    return Object.assign({}, axis, { size: 30, incrs: EPOCH_INCRS,
                                     values: function (u, ticks) { return ticks.map(wholeEpoch); } });
  }

  // ------------------------------------------------------------ log ticks (#1098)
  function logSplits(u, axisIdx, min, max) {
    if (!(min > 0) || !(max > min) || !isFinite(max)) { return []; }
    var e0 = Math.floor(Math.log10(min)), e1 = Math.ceil(Math.log10(max));
    var dec = Math.log10(max) - Math.log10(min);
    var mult = dec > 6 ? [1] : dec > 3 ? [1, 3] : dec > 1.2 ? [1, 2, 5] : [1, 2, 3, 4, 5, 6, 7, 8, 9];
    var out = [];
    for (var e = e0; e <= e1 && out.length < 40; e++) {
      for (var i = 0; i < mult.length; i++) {
        var v = +(mult[i] * Math.pow(10, e)).toPrecision(12);
        if (v >= min && v <= max) { out.push(v); }
      }
    }
    if (out.length >= 2) { return out; }
    // less than a decade with fewer than two round values in view: linear steps
    var raw = (max - min) / 4, mag = Math.pow(10, Math.floor(Math.log10(raw))), r = raw / mag;
    var step = (r < 1.5 ? 1 : r < 3.5 ? 2 : r < 7.5 ? 5 : 10) * mag;
    out = [];
    for (var t = Math.ceil(min / step) * step; t <= max && out.length < 40; t += step) {
      out.push(+t.toPrecision(12));
    }
    return out;
  }

  // ------------------------------------------------------------ axis transforms
  function fwd(st, axis, v) { return axis === "y" && st.yLog ? Math.log10(v) : v; }
  function inv(st, axis, t) { return axis === "y" && st.yLog ? Math.pow(10, t) : t; }

  // ------------------------------------------------------------ extents
  function chartXExtent(st) {
    var xs = st.data[0];
    return xs.length ? [xs[0], xs[xs.length - 1]] : null;
  }
  function figureXExtent(inst) {
    var lo = Infinity, hi = -Infinity;
    eachChart(inst, function (st) {
      var e = chartXExtent(st);
      if (e) { lo = Math.min(lo, e[0]); hi = Math.max(hi, e[1]); }
    });
    if (lo === Infinity) { return [0, 1]; }
    if (lo === hi) { lo -= 0.5; hi += 0.5; }
    return normX([lo, hi]);
  }
  // The one rule for an epoch view: it never starts below epoch 0. A view that
  // would is moved right with its width kept, so a zoom, a pan, the keys,
  // follow, the auto range and a view kept from before (a stored negative one
  // included) all end at the same bound, and no view is reset for it.
  function normX(r) {
    return r[0] < 0 ? [0, r[1] - r[0]] : r;
  }
  // The values in view (inside `xr` when given; all of them when none is),
  // finite and, on a log axis, above zero; padded 5 % in transformed space.
  function chartYExtent(st, xr) {
    var lo = Infinity, hi = -Infinity, xs = st.data[0];
    for (var pass = 0; pass < 2 && lo === Infinity; pass++) {
      for (var s = 1; s < st.data.length; s++) {
        var ys = st.data[s];
        for (var i = 0; i < ys.length; i++) {
          var v = ys[i];
          if (v == null || !isFinite(v) || (st.yLog && !(v > 0))) { continue; }
          if (pass === 0 && xr && (xs[i] < xr[0] || xs[i] > xr[1])) { continue; }
          if (v < lo) { lo = v; }
          if (v > hi) { hi = v; }
        }
      }
    }
    if (lo === Infinity) { return st.yLog ? [1e-6, 1] : [0, 1]; }
    if (st.yLog) {
      var a = Math.log10(lo), b = Math.log10(hi);
      if (a === b) { a -= 0.2; b += 0.2; }
      var p = (b - a) * 0.05;
      return [Math.pow(10, a - p), Math.pow(10, b + p)];
    }
    if (lo === hi) { var d = Math.abs(lo) * 0.05 || 0.5; lo -= d; hi += d; }
    var pad = (hi - lo) * 0.05;
    return [lo - pad, hi + pad];
  }
  function allYExtent(st) { return chartYExtent(st, null); }

  function eachChart(inst, fn) {
    for (var i = 0; i < inst.order.length; i++) {
      var st = inst.charts[inst.order[i]];
      if (st) { fn(st); }
    }
  }

  // ------------------------------------------------------------ the view
  function xView(inst) { return inst.xManual && inst.x ? normX(inst.x) : figureXExtent(inst); }

  function applyView(inst) {
    var xr = xView(inst);
    inst.x = xr.slice();
    eachChart(inst, function (st) {
      var yr = st.yManual && st.y ? st.y : chartYExtent(st, xr);
      st.y = yr.slice();
      if (!st.u) { return; }
      st.u.batch(function () {
        st.u.setScale("x", { min: xr[0], max: xr[1] });
        st.u.setScale("y", { min: yr[0], max: yr[1] });
      });
      updateAria(inst, st);
    });
    updateToolbar(inst);
  }

  function minSpan(inst, st, axis, c) {
    if (axis === "y" && st.yLog) { return 1e-4; }
    if (axis === "x") { return 1; }
    var e = allYExtent(st);
    return Math.max(Math.abs(c) * 1e-9, (e[1] - e[0]) * 1e-6, 1e-12);
  }
  function maxSpan(inst, st, axis, ms) {
    if (axis === "x") {
      var ex = figureXExtent(inst);
      return Math.max((ex[1] - ex[0]) * 1000, ms * 10);
    }
    var e = allYExtent(st);
    var span = fwd(st, "y", e[1]) - fwd(st, "y", e[0]);
    return st.yLog ? span + 6 : Math.max(span * 1000, ms * 10);
  }

  // Zoom one axis of one chart (X is the figure's) by `factor` around
  // `center` (a data value; the middle of the view when none).
  function zoomAxis(inst, st, axis, factor, center) {
    var r = axis === "x" ? xView(inst) : (st.yManual && st.y ? st.y : chartYExtent(st, xView(inst)));
    var t0 = fwd(st, axis, r[0]), t1 = fwd(st, axis, r[1]);
    var c = center == null || !isFinite(center) || (axis === "y" && st.yLog && !(center > 0))
      ? (t0 + t1) / 2 : fwd(st, axis, center);
    var n0 = c - (c - t0) * factor, n1 = c + (t1 - c) * factor;
    var ms = minSpan(inst, st, axis, c);
    if (n1 - n0 < ms) { var k = ms / (n1 - n0 || ms); n0 = c - (c - n0) * k; n1 = c + (n1 - c) * k; }
    var Ms = maxSpan(inst, st, axis, ms);
    if (n1 - n0 > Ms) { var k2 = Ms / (n1 - n0); n0 = c - (c - n0) * k2; n1 = c + (n1 - c) * k2; }
    setAxis(inst, st, axis, inv(st, axis, n0), inv(st, axis, n1));
  }

  // Pan one axis by `frac` of its span (+ moves the view to higher values).
  function panAxis(inst, st, axis, frac, from) {
    var r = from || (axis === "x" ? xView(inst) : (st.yManual && st.y ? st.y : st.y || allYExtent(st)));
    var t0 = fwd(st, axis, r[0]), t1 = fwd(st, axis, r[1]);
    var d = (t1 - t0) * frac;
    setAxis(inst, st, axis, inv(st, axis, t0 + d), inv(st, axis, t1 + d));
  }

  function setAxis(inst, st, axis, a, b) {
    if (!isFinite(a) || !isFinite(b) || !(b > a) || (axis === "y" && st.yLog && !(a > 0))) { return; }
    if (axis === "x") {
      if (!inst.xManual) { inst.frozenLast = lastEpoch(inst); }
      inst.x = normX([a, b]);
      inst.xManual = true;
    } else {
      st.y = [a, b];
      st.yManual = true;
    }
    saveView(inst);
  }

  function lastEpoch(inst) {
    var last = null;
    eachChart(inst, function (st) {
      var xs = st.data[0];
      if (xs.length && (last === null || xs[xs.length - 1] > last)) { last = xs[xs.length - 1]; }
    });
    return last;
  }

  function reset(inst) {
    inst.xManual = false;
    inst.frozenLast = null;
    eachChart(inst, function (st) { st.yManual = false; });
    applyView(inst);
    saveView(inst);
  }

  // Jump to the newest epoch, keeping the zoom width.
  function follow(inst) {
    var ext = figureXExtent(inst), x = xView(inst), w = x[1] - x[0];
    inst.x = normX([ext[1] - w, ext[1]]);
    inst.xManual = true;
    inst.frozenLast = ext[1];
    applyView(inst);
    saveView(inst);
  }

  // Epochs that arrived while X was zoomed and lie right of the view; epochs
  // already there when the zoom began are not new.
  function newEpochsBeyond(inst) {
    if (!inst.xManual || !inst.x) { return 0; }
    var edge = Math.max(inst.x[1], inst.frozenLast == null ? -Infinity : inst.frozenLast);
    var longest = null;
    eachChart(inst, function (st) {
      if (!longest || st.data[0].length > longest.length) { longest = st.data[0]; }
    });
    var n = 0;
    if (longest) { for (var i = longest.length - 1; i >= 0 && longest[i] > edge; i--) { n++; } }
    return n;
  }

  // ------------------------------------------------------------ persistence (per run)
  // One entry per run (the most recent STORE_RUNS runs), so going back to a
  // run shown earlier in this tab brings its view back too.
  function storeKey(inst) { return STORE_PREFIX + inst.key; }
  function readStore(inst) {
    var raw = window.sessionStorage.getItem(storeKey(inst));
    var all = raw ? JSON.parse(raw) : null;
    return all && typeof all === "object" && all.runs && typeof all.runs === "object" ? all : { runs: {}, order: [] };
  }
  function writeView(inst) {
    clearTimeout(inst.saveTimer);
    inst.saveTimer = null;
    if (!inst.run) { return; }
    try {
      var ys = {};
      eachChart(inst, function (st) { if (st.yManual && st.y) { ys[st.id] = st.y; } });
      var all = readStore(inst);
      var order = (Array.isArray(all.order) ? all.order : []).filter(function (r) { return r !== inst.run; });
      order.push(inst.run);
      all.runs[inst.run] = { x: inst.xManual ? inst.x : null, frozenLast: inst.frozenLast, ys: ys };
      while (order.length > STORE_RUNS) { delete all.runs[order.shift()]; }
      all.order = order;
      window.sessionStorage.setItem(storeKey(inst), JSON.stringify(all));
    } catch (err) { /* storage blocked: the view is not kept across a reload */ }
  }
  function saveView(inst) {
    if (inst.saveTimer) { return; }
    inst.saveTimer = setTimeout(function () { inst.saveTimer = null; if (!inst.destroyed) { writeView(inst); } }, 250);
  }
  function loadView(inst, run) {
    try {
      var v = readStore(inst).runs[run];
      return v && typeof v === "object" ? v : null;
    } catch (err) { return null; }
  }

  // A run other than the one shown: its own stored view, or auto-fit. Nothing
  // of the run shown until now is carried over, neither the epoch view nor any
  // chart's value view (the charts are kept when their ids are the same).
  function enterRun(inst, run) {
    if (inst.saveTimer) { writeView(inst); }            // the run shown until now keeps its last view
    var kept = loadView(inst, run);
    inst.xManual = !!(kept && kept.x);
    inst.x = kept && kept.x ? kept.x : null;
    inst.frozenLast = kept ? kept.frozenLast : null;
    eachChart(inst, function (st) { st.yManual = false; st.y = null; });
    inst.pendingYs = kept ? kept.ys || {} : {};
    inst.run = run;
    if (!kept) { saveView(inst); }
  }

  // ------------------------------------------------------------ hit testing
  function zoneOf(st, clientX, clientY) {
    if (!st || !st.u) { return null; }
    var o = st.u.over.getBoundingClientRect();
    var w = st.u.root.querySelector(".u-wrap").getBoundingClientRect();
    var inX = clientX >= o.left && clientX <= o.right;
    var inY = clientY >= o.top && clientY <= o.bottom;
    if (inX && inY) { return "xy"; }
    if (inX && clientY > o.bottom && clientY <= w.bottom) { return "x"; }
    if (inY && clientX < o.left && clientX >= w.left) { return "y"; }
    return null;
  }
  function wheelDelta(e) {
    var d = e.deltaY;
    if (e.deltaMode === 1) { d *= 33; } else if (e.deltaMode === 2) { d *= 400; }
    return Math.max(-400, Math.min(400, d));
  }
  function chartOf(inst, target) {
    var card = target && target.closest ? target.closest(".shep-chart") : null;
    return card && inst.root.contains(card) ? inst.charts[card.getAttribute("data-chart")] : null;
  }

  // ------------------------------------------------------------ rendering
  function colourSlot(inst, chartId, label) {
    var byKey = COLORS[inst.key] || (COLORS[inst.key] = {});
    var m = byKey[chartId] || (byKey[chartId] = {});
    if (!(label in m)) { m[label] = Object.keys(m).length; }
    return m[label];
  }
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  // A series' swatch in the legend and the tooltip: a short line in its colour
  // and its own dash pattern (the slot's DASHES entry, as its line is drawn),
  // so series that overlap are told apart without colour.
  function swatchHtml(slot) {
    var dash = DASHES[slot % DASHES.length];
    return '<svg class="shep-sw" width="24" height="8" viewBox="0 0 24 8" aria-hidden="true" focusable="false">' +
      '<line x1="0" y1="4" x2="24" y2="4" stroke="' + PALETTE[slot % PALETTE.length] + '" stroke-width="2"' +
      (dash ? ' stroke-dasharray="' + dash.join(" ") + '"' : "") + "/></svg>";
  }

  function keysHtml(inst, spec) {
    return (spec.series || []).map(function (s) {
      var slot = colourSlot(inst, spec.id, String(s.label));
      return '<span class="shep-chart-key">' + swatchHtml(slot) + esc(s.label) + "</span>";
    }).join("");
  }

  function cardHtml(inst, spec) {
    return '<div class="shep-chart-hd"><span class="shep-chart-title">' + esc(spec.title) + "</span>" +
      '<span class="shep-chart-keys">' + keysHtml(inst, spec) + "</span></div>" +
      '<div class="shep-chart-plot" tabindex="0" role="img"></div>' +
      '<div class="shep-chart-tip" hidden></div>' +
      (spec.xLabel ? '<div class="shep-chart-xl">' + esc(spec.xLabel) + "</div>" : "");
  }

  function buildCards(inst, specs) {
    destroyCharts(inst);
    inst.grid.innerHTML = "";
    inst.order = [];
    inst.charts = {};
    specs.forEach(function (spec) {
      var card = document.createElement("div");
      card.className = "shep-chart";
      card.setAttribute("data-chart", spec.id);
      card.innerHTML = cardHtml(inst, spec);
      inst.grid.appendChild(card);
      var st = { id: spec.id, card: card, plot: card.querySelector(".shep-chart-plot"),
                 tip: card.querySelector(".shep-chart-tip"), u: null, y: null, yManual: false,
                 data: [[]], labels: [], yLog: false, title: spec.title };
      inst.charts[spec.id] = st;
      inst.order.push(spec.id);
      inst.ro.observe(st.plot);
    });
    inst.root.classList.toggle("shep-charts-n1", specs.length === 1);
  }

  function sanitize(st, values) {
    var out = new Array(values.length), dropped = 0;
    for (var i = 0; i < values.length; i++) {
      var v = values[i];
      if (v == null || typeof v !== "number" || !isFinite(v) || (st.yLog && !(v > 0))) {
        out[i] = null;
        if (v != null) { dropped++; }
      } else { out[i] = v; }
    }
    st.dropped += dropped;
    return out;
  }

  function create(inst, st, width, height) {
    var series = [{ label: "epoch", value: function (u, v) { return fmtEpoch(v); } }];
    st.labels.forEach(function (label) {
      var slot = colourSlot(inst, st.id, label);
      var dash = DASHES[slot % DASHES.length];
      var s = { label: label, stroke: PALETTE[slot % PALETTE.length], width: 2, spanGaps: true,
                points: { size: 5 }, value: function (u, v) { return fmt(v); } };
      if (dash) { s.dash = dash; }
      series.push(s);
    });
    var axis = { stroke: INK, font: FONT, grid: { stroke: GRID, width: 1 },
                 ticks: { stroke: GRID, width: 1, size: 4 } };
    var opts = {
      width: Math.max(120, Math.floor(width)),
      height: Math.max(80, Math.floor(height)),
      pxAlign: true,
      scales: { x: { time: false, auto: false },
                y: st.yLog ? { auto: false, distr: 3, log: 10 } : { auto: false } },
      axes: [
        epochAxis(axis),
        // Short plots (a laptop's 760 px) keep at least two or three labels.
        Object.assign({}, axis, { size: 64, space: function (u, i, min, max, dim) { return dim < 140 ? 22 : 32; },
                                  values: function (u, ticks) {
          return ticks.map(function (v) { return v == null ? "" : fmt(v); });
        } }, st.yLog ? { splits: logSplits } : {}),
      ],
      series: series,
      legend: { show: false },
      cursor: {
        drag: { x: false, y: false, setScale: false },
        // uPlot's own mousedown/mouseup/dblclick are off: the pan and reset
        // are ours, and 1.6.32 leaks a document mouseup listener when a chart
        // is destroyed during a press.
        bind: { mousedown: function () { return null; }, mouseup: function () { return null; },
                dblclick: function () { return null; } },
        sync: { key: "shep-" + inst.key, scales: ["x", "y"],
                match: [function (a, b) { return a === b; }, function () { return false; }],
                setSeries: false },
        points: { size: 7 },
      },
      hooks: { setCursor: [function (u) { showTip(inst, st, u); }] },
    };
    st.plot.innerHTML = "";
    st.u = new window.uPlot(opts, st.data, st.plot);
    created++;
  }

  function showTip(inst, st, u) {
    var idx = u.cursor.idx;
    if (idx == null || inst.hover !== st.id || u.cursor.left == null || u.cursor.left < 0) {
      if (!st.tip.hidden) { st.tip.hidden = true; }
      return;
    }
    var rows = ['<div class="shep-tip-hd">epoch ' + esc(fmtEpoch(st.data[0][idx])) + "</div>"];
    for (var s = 1; s < st.data.length; s++) {
      var v = st.data[s][idx];
      var slot = colourSlot(inst, st.id, st.labels[s - 1]);
      rows.push("<div>" + swatchHtml(slot) + esc(st.labels[s - 1]) + " " + (v == null ? "—" : esc(fmt(v))) + "</div>");
    }
    st.tip.innerHTML = rows.join("");
    st.tip.hidden = false;
    var plotBox = st.plot.getBoundingClientRect(), cardBox = st.card.getBoundingClientRect();
    var over = u.over.getBoundingClientRect();
    var left = over.left - cardBox.left + u.cursor.left + 12;
    var maxLeft = cardBox.width - st.tip.offsetWidth - 4;
    if (left > maxLeft) { left = over.left - cardBox.left + u.cursor.left - st.tip.offsetWidth - 12; }
    st.tip.style.left = Math.max(4, left) + "px";
    st.tip.style.top = Math.max(0, plotBox.top - cardBox.top + 8) + "px";
  }

  function updateAria(inst, st) {
    var xs = st.data[0], parts = [];
    for (var s = 1; s < st.data.length; s++) {
      var ys = st.data[s];
      for (var i = ys.length - 1; i >= 0; i--) {
        if (ys[i] != null) { parts.push(st.labels[s - 1] + " " + fmt(ys[i]) + " at epoch " + fmtEpoch(xs[i])); break; }
      }
    }
    var view = inst.xManual ? "Zoomed: epochs " + wholeEpoch(inst.x[0]) + " to " + wholeEpoch(inst.x[1]) + "."
      : "Showing all epochs.";
    var beyond = newEpochsBeyond(inst);
    if (beyond) { view += " " + beyond + " newer epochs are to the right of the view."; }
    if (st.dropped) { view += " " + st.dropped + " value(s) at or below zero are not shown on the log scale."; }
    var label = st.title + " chart. " + (parts.length ? "Latest: " + parts.join("; ") + ". " : "No data yet. ") +
      view + " Keys: + or - zoom, x or y one axis (Shift zooms out), arrows pan, End the latest epochs, 0 resets.";
    if (st.plot.getAttribute("aria-label") !== label) { st.plot.setAttribute("aria-label", label); }
  }

  function updateToolbar(inst) {
    var note = inst.root.querySelector(".shep-charts-zoom");
    if (note) {
      var any = inst.xManual;
      eachChart(inst, function (st) { any = any || st.yManual; });
      var text = inst.xManual ? "zoomed: epochs " + wholeEpoch(inst.x[0]) + "–" + wholeEpoch(inst.x[1])
        : any ? "zoomed: values" : "";
      if (note.textContent !== text) { note.textContent = text; }
      note.hidden = !text;
    }
    var cue = inst.root.querySelector(".shep-charts-new");
    if (cue) {
      var n = newEpochsBeyond(inst);
      cue.hidden = !n;
      if (n) {
        var txt = n + " new epoch" + (n === 1 ? "" : "s") + " →";
        if (cue.textContent !== txt) {
          cue.textContent = txt;
          cue.setAttribute("aria-label", n + " new epochs to the right of the view: show the latest epochs at this width (End key)");
        }
      }
    }
  }

  function setMessage(inst, text) {
    var msg = inst.root.querySelector(".shep-charts-msg");
    if (!msg) { return; }
    if (msg.textContent !== text) { msg.textContent = text; }
    msg.hidden = !text;
  }

  // ------------------------------------------------------------ data
  function sameSpecs(inst, specs) {
    if (specs.length !== inst.order.length) { return false; }
    for (var i = 0; i < specs.length; i++) { if (specs[i].id !== inst.order[i]) { return false; } }
    return true;
  }

  function apply(inst, d) {
    var specs = (d && d.charts) || [];
    if (!sameSpecs(inst, specs)) { buildCards(inst, specs); }
    var empty = true;
    specs.forEach(function (spec) {
      var st = inst.charts[spec.id];
      var log = spec.yScale === "log";
      var labels = (spec.series || []).map(function (s) { return String(s.label); });
      var recreate = st.u && (st.yLog !== log || labels.join("\u0001") !== st.labels.join("\u0001"));
      st.yLog = log;
      st.dropped = 0;
      st.title = spec.title;
      st.data = [Array.prototype.slice.call(spec.x || [])].concat((spec.series || []).map(function (s) {
        return sanitize(st, Array.prototype.slice.call(s.values || []));
      }));
      if (labels.join("\u0001") !== st.labels.join("\u0001")) {
        st.labels = labels;
        st.card.querySelector(".shep-chart-keys").innerHTML = keysHtml(inst, spec);
      }
      if (st.data[0].length) { empty = false; }
      st.card.classList.toggle("shep-chart-empty", !st.data[0].length);
      if (recreate) { st.u.destroy(); destroyed++; st.u = null; }
      if (st.u) { st.u.setData(st.data, false); }
      else { sizeOrCreate(inst, st); }
    });
    applyView(inst);
    setMessage(inst, specs.length ? (empty ? (d.empty || "No epochs logged yet.") : "") : (d.empty || ""));
  }

  function sizeOrCreate(inst, st, box) {
    var w = box ? box.width : st.plot.clientWidth, h = box ? box.height : st.plot.clientHeight;
    if (!(w > 0) || !(h > 0) || !st.labels.length) { return; }
    if (!st.u) { create(inst, st, w, h); applyView(inst); return; }
    if (Math.abs(st.u.width - w) > 1 || Math.abs(st.u.height - h) > 1) {
      st.u.setSize({ width: Math.floor(w), height: Math.floor(h) });
    }
  }

  function destroyCharts(inst) {
    eachChart(inst, function (st) {
      if (st.u) { try { st.u.destroy(); } catch (err) { /* already gone */ } st.u = null; destroyed++; }
      if (inst.ro) { inst.ro.unobserve(st.plot); }
    });
  }

  // ------------------------------------------------------------ input
  function bind(inst) {
    var root = inst.root, sig = { signal: inst.ac.signal };
    var on = function (type, fn, extra) { root.addEventListener(type, fn, Object.assign({}, sig, extra || {})); };
    var drag = null;

    on("wheel", function (e) {
      var st = chartOf(inst, e.target), zone = zoneOf(st, e.clientX, e.clientY);
      if (!zone) { return; }                  // not over a plot or its axes: the page scrolls
      e.preventDefault();
      var f = Math.exp(wheelDelta(e) * WHEEL_STEP), o = st.u.over.getBoundingClientRect();
      if (zone.indexOf("x") >= 0) { zoomAxis(inst, st, "x", f, st.u.posToVal(e.clientX - o.left, "x")); }
      if (zone.indexOf("y") >= 0) { zoomAxis(inst, st, "y", f, st.u.posToVal(e.clientY - o.top, "y")); }
      applyView(inst);
    }, { passive: false });

    on("pointerdown", function (e) {
      if (e.button !== 0) { return; }
      var st = chartOf(inst, e.target), zone = zoneOf(st, e.clientX, e.clientY);
      if (!zone) { return; }
      var o = st.u.over.getBoundingClientRect(), wrap = st.u.root.querySelector(".u-wrap");
      drag = { st: st, zone: zone, cx: e.clientX, cy: e.clientY, x: xView(inst).slice(),
               y: (st.y || allYExtent(st)).slice(), w: o.width, h: o.height, id: e.pointerId, wrap: wrap };
      try { wrap.setPointerCapture(e.pointerId); } catch (err) { /* a synthetic event */ }
      wrap.classList.add("shep-panning");
      e.preventDefault();
    });
    on("pointermove", function (e) {
      var st = chartOf(inst, e.target);
      inst.hover = st ? st.id : null;
      if (!drag) { return; }
      var dx = e.clientX - drag.cx, dy = e.clientY - drag.cy;
      if (!drag.moved && Math.abs(dx) + Math.abs(dy) < 3) { return; }
      drag.moved = true;
      if (drag.zone.indexOf("x") >= 0) { panAxis(inst, drag.st, "x", -dx / drag.w, drag.x); }
      if (drag.zone.indexOf("y") >= 0) { panAxis(inst, drag.st, "y", dy / drag.h, drag.y); }
      applyView(inst);
    });
    var endDrag = function () {
      if (!drag) { return; }
      try { drag.wrap.releasePointerCapture(drag.id); } catch (err) { /* a synthetic event */ }
      drag.wrap.classList.remove("shep-panning");
      drag = null;
    };
    on("pointerup", endDrag);
    on("pointercancel", endDrag);
    on("pointerleave", function (e) {
      if (e.target === root) { inst.hover = null; eachChart(inst, function (st) { st.tip.hidden = true; }); }
    });
    on("dblclick", function (e) {
      var st = chartOf(inst, e.target);
      if (zoneOf(st, e.clientX, e.clientY)) { reset(inst); }
    });
    on("keydown", function (e) {
      var plot = e.target && e.target.closest ? e.target.closest(".shep-chart-plot") : null;
      if (!plot || e.ctrlKey || e.metaKey || e.altKey || pending(inst)) { return; }
      var st = chartOf(inst, plot);
      if (!st || !st.u) { return; }
      var k = e.key, handled = true;
      if (k === "+" || k === "=") { zoomAxis(inst, st, "x", KEY_ZOOM); zoomAxis(inst, st, "y", KEY_ZOOM); }
      else if (k === "-" || k === "_") { zoomAxis(inst, st, "x", 1 / KEY_ZOOM); zoomAxis(inst, st, "y", 1 / KEY_ZOOM); }
      else if (k === "x") { zoomAxis(inst, st, "x", KEY_ZOOM); }
      else if (k === "X") { zoomAxis(inst, st, "x", 1 / KEY_ZOOM); }
      else if (k === "y") { zoomAxis(inst, st, "y", KEY_ZOOM); }
      else if (k === "Y") { zoomAxis(inst, st, "y", 1 / KEY_ZOOM); }
      else if (k === "ArrowLeft") { panAxis(inst, st, "x", -KEY_PAN); }
      else if (k === "ArrowRight") { panAxis(inst, st, "x", KEY_PAN); }
      else if (k === "ArrowUp") { panAxis(inst, st, "y", KEY_PAN); }
      else if (k === "ArrowDown") { panAxis(inst, st, "y", -KEY_PAN); }
      else if (k === "End") { e.preventDefault(); follow(inst); return; }
      else if (k === "0" || k === "Home") { e.preventDefault(); reset(inst); return; }
      else { handled = false; }
      if (handled) { e.preventDefault(); applyView(inst); }
    });
    on("click", function (e) {
      var btn = e.target && e.target.closest ? e.target.closest("[data-chart-act]") : null;
      if (!btn || !root.contains(btn)) { return; }
      var act = btn.getAttribute("data-chart-act");
      if (act === "csv") { exportCsv(inst, btn); return; }
      if (pending(inst)) { return; }                     // the view drawn is hidden: not another run's
      if (act === "reset") { reset(inst); return; }
      if (act === "follow") { follow(inst); return; }
      var f = act === "zoom-in" ? KEY_ZOOM : 1 / KEY_ZOOM;
      var first = inst.charts[inst.order[0]];
      if (!first) { return; }
      zoomAxis(inst, first, "x", f);
      eachChart(inst, function (st) { zoomAxis(inst, st, "y", f); });
      applyView(inst);
    });
  }

  function exportCsv(inst, btn) {
    var fn = inst.root.getAttribute("data-csv-fn");
    var t = token(inst);
    if (!fn || !inst.server || typeof inst.server[fn] !== "function" || !t.run) { return; }
    btn.disabled = true;
    Promise.resolve(inst.server[fn](t)).then(function (out) {
      if (inst.destroyed || !out || !out.csv) { return; }
      var blob = new Blob([out.csv], { type: "text/csv;charset=utf-8" });
      var url = URL.createObjectURL(blob), a = document.createElement("a");
      a.href = url;
      a.download = out.filename || "metrics.csv";
      inst.root.appendChild(a);
      a.click();
      a.remove();
      setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
    }).catch(function () { /* the button comes back; nothing was saved */ })
      .then(function () { btn.disabled = false; });
  }

  // ------------------------------------------------------------ the token
  function token(inst) {
    var t = inst.props.value;
    if (typeof t === "string") { try { t = JSON.parse(t); } catch (err) { t = null; } }
    return t && typeof t === "object" ? t : {};
  }

  // Whether the component still asks for what request `t` was sent for: its
  // token now, not the one saved with the request. Gradio replaces
  // props.value before it calls the watch callbacks, so a reply that comes
  // after the selection changed (another run, the run drawn again, or none)
  // is never drawn, whatever the refresh for that change did.
  function stillAsked(inst, t) {
    var now = token(inst);
    return now.run === t.run && now.rev === t.rev;
  }

  // No run drawn: the run drawn until now keeps its last view (written now,
  // as when another run is entered) and nothing of it stays on screen, its
  // zoom note and new-epochs button included.
  function leaveRun(inst) {
    if (inst.saveTimer) { writeView(inst); }
    inst.run = null; inst.rev = null;
    inst.x = null; inst.xManual = false; inst.frozenLast = null;
    buildCards(inst, []);
    updateToolbar(inst);
  }

  // The pending cue: the run drawn never stands for another run asked.
  function setPending(inst, on) {
    var v = on ? "true" : "false";
    if (inst.root.getAttribute("aria-busy") !== v) { inst.root.setAttribute("aria-busy", v); }
  }
  function pending(inst) { return inst.root.getAttribute("aria-busy") === "true"; }

  function refresh(inst, retrying) {
    if (inst.destroyed) { return; }
    clearTimeout(inst.retryTimer);                          // this refresh replaces a retry
    if (!retrying) { inst.retries = 0; }                    // each token has its own retries
    var t = token(inst);
    if (!t.run) {
      setPending(inst, false);
      if (inst.run !== null) { leaveRun(inst); }
      setMessage(inst, t.empty || inst.root.getAttribute("data-empty") || "");
      return;
    }
    if (t.run === inst.run && t.rev === inst.rev && !inst.failed) { setPending(inst, false); return; }
    var my = ++inst.seq;
    var fn = inst.root.getAttribute("data-series-fn");
    if (!fn || !inst.server || typeof inst.server[fn] !== "function") { return; }
    setPending(inst, t.run !== inst.run && !retrying);      // a retry keeps its "Trying again…"
    Promise.resolve(inst.server[fn](t)).then(function (d) {
      // unmounted, a newer request sent, or no longer what the token asks for
      if (inst.destroyed || my !== inst.seq || !stillAsked(inst, t)) { return; }
      setPending(inst, false);
      if (d == null) { throw new Error("no reply"); }       // a failed call: never "no longer listed"
      inst.failed = false;
      inst.retries = 0;
      if (d.run !== t.run) {                                // the run is no longer the one listed
        leaveRun(inst);
        setMessage(inst, d.empty || "This run is no longer listed.");
        return;
      }
      if (d.run !== inst.run) { enterRun(inst, d.run); }   // a new run: fit it, or bring back its view
      inst.rev = d.rev;
      apply(inst, d);
      if (inst.pendingYs) {
        var ys = inst.pendingYs;
        inst.pendingYs = null;
        eachChart(inst, function (st) { if (ys[st.id]) { st.y = ys[st.id]; st.yManual = true; } });
        applyView(inst);
      }
    }).catch(function () {
      if (inst.destroyed || my !== inst.seq || !stillAsked(inst, t)) { return; }
      setPending(inst, false);
      if (inst.run !== null && inst.run !== t.run) { leaveRun(inst); }  // never another run's curves
      inst.failed = true;
      var again = (inst.retries = (inst.retries || 0) + 1) <= RETRY_MAX;
      if (again) { inst.retryTimer = setTimeout(function () { refresh(inst, true); }, RETRY_MS); }
      if (inst.run === null) {
        setMessage(inst, again ? "The charts could not be loaded. Trying again…" : "The charts could not be loaded.");
      }
    });
  }

  // ------------------------------------------------------------ mount
  function mount(element, props, watch, server, key) {
    if (INST[key]) { INST[key].destroy(); }
    var root = element.querySelector(".shep-charts");
    if (!root || !window.uPlot) {
      var msg = element.querySelector(".shep-charts-msg");
      if (msg) { msg.textContent = "The chart library did not load."; msg.hidden = false; }
      return null;
    }
    var inst = { key: key, element: element, root: root, grid: root.querySelector(".shep-charts-grid"),
                 props: props, server: server, charts: {}, order: [], x: null, xManual: false,
                 frozenLast: null, run: null, rev: null, seq: 0, destroyed: false, hover: null,
                 ac: new AbortController(), raf: 0 };
    INST[key] = inst;
    inst.ro = new ResizeObserver(function (entries) {
      if (!element.isConnected) { inst.destroy(); return; }   // removed: it reports 0 x 0 once
      var boxes = [];
      for (var i = 0; i < entries.length; i++) {
        var r = entries[i].contentRect;
        if (r.width > 0 && r.height > 0) { boxes.push([entries[i].target, r]); }  // hidden: skipped
      }
      if (!boxes.length) { return; }
      cancelAnimationFrame(inst.raf);
      inst.raf = requestAnimationFrame(function () {
        if (inst.destroyed) { return; }
        boxes.forEach(function (b) {
          eachChart(inst, function (st) { if (st.plot === b[0]) { sizeOrCreate(inst, st, b[1]); } });
        });
      });
    });
    inst.destroy = function () {
      if (inst.destroyed) { return; }
      if (inst.saveTimer) { writeView(inst); }              // a pending view is kept, not dropped
      inst.destroyed = true;
      inst.ac.abort();
      inst.ro.disconnect();
      cancelAnimationFrame(inst.raf);
      clearTimeout(inst.retryTimer);
      destroyCharts(inst);
      if (INST[key] === inst) { delete INST[key]; }
    };
    bind(inst);
    root.setAttribute("data-state", "ready");
    setMessage(inst, "");
    watch("value", function () { refresh(inst); });
    refresh(inst);
    return inst;
  }

  // What tests read: instance counts and the view, never used by the page.
  function stats() {
    var live = 0, charts = 0, canvases = 0;
    Object.keys(INST).forEach(function (k) {
      live++;
      eachChart(INST[k], function (st) { if (st.u) { charts++; } });
      canvases += INST[k].root.querySelectorAll("canvas").length;
    });
    return { instances: live, charts: charts, canvases: canvases, created: created, destroyed: destroyed };
  }
  function view(key) {
    var inst = INST[key];
    if (!inst) { return null; }
    var ys = {};
    eachChart(inst, function (st) {
      ys[st.id] = { y: st.y && st.y.slice(), manual: st.yManual, n: st.data[0].length,
                    scale: st.u ? [st.u.scales.y.min, st.u.scales.y.max] : null,
                    scaleX: st.u ? [st.u.scales.x.min, st.u.scales.x.max] : null,
                    log: st.yLog, labels: st.labels.slice() };
    });
    return { run: inst.run, rev: inst.rev, x: inst.x && inst.x.slice(), xManual: inst.xManual,
             beyond: newEpochsBeyond(inst), charts: ys, order: inst.order.slice() };
  }

  window.ShepCharts = { mount: mount, logSplits: logSplits, stats: stats, view: view,
                        _inst: INST, _u: function (key, id) { var i = INST[key]; return i && i.charts[id] && i.charts[id].u; },
                        _views: { write: writeView, load: loadView, enter: enterRun },
                        _x: { norm: normX, view: xView, zoom: zoomAxis, pan: panAxis },
                        _epochs: { axis: epochAxis, label: wholeEpoch },
                        _keys: { html: keysHtml, swatch: swatchHtml, dashes: DASHES } };
})();
