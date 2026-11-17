"""
A real headless browser for tests: Node with Playwright, where installed
========================================================================
The WebUI tests that measure what a browser draws (the shared ⓘ's bubble,
the header's layout) run a small Node script that drives Playwright's
Chromium. `node_with_playwright` finds Node and a ``NODE_PATH`` under which it
resolves Playwright (npm's global root, where it is pre-installed); a test
skips without it. Nothing is installed.

Moved here from the help tests, so the shell's tests use the same.

**What a page asks of the server** (`QUEUE_JS`, a page init script): every
queue join the page sends (its listener's ``fn_index``, ``trigger_id`` and
``event_id``) and every event's completion on the page's event stream, as
``window.__shepQueue``: ``joins``, ``done`` (event ids answered),
``inflight`` (joined, not yet answered), ``direct`` (a ``queue=False``
listener's requests not yet answered) and ``idle()`` (nothing unanswered). A
test counts the requests one act sent and waits for what it asked to be
answered, never for a fixed time. `run_node` runs a script and reads the JSON
it prints.

**What a page draws, measured where the browser draws it** (`LOOK_JS`, a page
init script; T04, the WebUI's shared tokens): ``window.__shepLook`` reads
Gradio's effective styles, never the style sheet's text:

- ``typeScale(root)``: each visible text's size and weight (``"15/400"`` with
  a count and examples) and the visible texts drawn upper case; a text is a
  node with a letter or a digit (a glyph alone, ×, ⓘ or ↻, is an icon);
- ``fields(root)``: every visible text box, number box, text area and select
  wrap (a select's border is its wrap's), with its border's width and its
  contrast against what is behind it, and whether it is disabled;
- ``labels(root)``: every visible field label, Gradio's (``block-info``) and
  the pages' own (`renderers.FIELD_ROWS` keys and the classes listed in
  ``LABELS``), with its text and style (size, weight, colour, case) and
  whether the field it names cuts it (it runs past that block's box);
- ``pseudo(el, which)``: a ``::before`` or ``::after`` box from its computed
  position, in page coordinates, with its content and type;
- ``topAt(x, y)``: what ``document.elementsFromPoint`` finds first there;
- ``ratio(a, b)`` and ``behind(el)``: WCAG contrast of two computed colours,
  and the first opaque background behind an element.

``typeScale`` also lists every visible text under 14 px, a glyph included.

**The shared UI measurement check** (D08, INT-AUDIT): `LOOK_JS` also measures
one window of the page shown, ``check()`` (no sideways page scroll, its scroll
regions with their panel and the regions they are inside, ``typeScale``,
``fields``, ``labels``, the research tabs' group captions against their tabs'
line and hit area, the words of its radios and segments) and
``focusWalk(scope)``: each control drawn (in ``scope`` when given), focused as
a keyboard focuses it, what is over its centre (another control's own
overlay, the help bubble or an open list it leaves behind, aside) and its
focus cue's contrast. `walk_ui` drives one browser through a list of stops
(`UI_WALK_JS`: a page at a window size, with a drawer of it opened where a
stop asks; the page's requests answered and its drawing still before it
measures; the selects' options read where a stop asks; a drawer opened, and
a row of it chosen, where a stop asks), and `ui_problems`
turns what it saw into one line per broken rule. The representative windows
are `UI_SIZES`, the pages `UI_PAGES`.

Module: tests/fixtures/browser.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


def node_with_playwright() -> tuple[str, dict[str, str]] | None:
    """Node and a NODE_PATH under which it resolves Playwright, or None."""
    node = shutil.which("node")
    if node is None:
        return None
    env = dict(os.environ)
    npm = shutil.which("npm")
    if npm:
        found = subprocess.run([npm, "root", "-g"], capture_output=True, text=True, timeout=60)
        if found.returncode == 0 and found.stdout.strip():
            env["NODE_PATH"] = os.pathsep.join(
                p for p in (env.get("NODE_PATH"), found.stdout.strip()) if p)
    probe = subprocess.run([node, "-e", "require.resolve('playwright')"], env=env,
                           capture_output=True, timeout=60)
    return (node, env) if probe.returncode == 0 else None


#: The page's queue as the page itself sees it (Gradio 6.30's client joins with
#: ``fetch`` and reads its event stream with ``fetch`` too): installed with
#: ``page.addInitScript`` before the page loads.
QUEUE_JS = r"""
(() => {
  const q = window.__shepQueue = {joins: [], done: {}, inflight: 0, direct: 0};
  /* The listener `fnIndex`'s joins, and how many of them were answered. */
  q.count = (fnIndex) => q.joins.filter((j) => j.fn_index === fnIndex).length;
  q.answered = (fnIndex) => q.joins.filter((j) => j.fn_index === fnIndex && j.event_id !== null
                                            && j.event_id in q.done).length;
  /* Nothing asked is unanswered: every join, and every listener that skips the
     queue (``queue=False``, answered by its own request). */
  q.idle = () => q.inflight === 0 && q.direct === 0 && q.joins.every((j) => j.event_id !== null);
  const original = window.fetch;
  window.fetch = function (input, init) {
    const url = typeof input === 'string' ? input : (input && input.url) || String(input);
    const sent = original.apply(this, arguments);
    if (url.includes('/run/predict')) {
      q.direct += 1;
      sent.finally(() => { q.direct -= 1; }).catch(() => {});
      return sent;
    }
    if (url.includes('/queue/join')) {
      /* Counted when it is asked, before any answer: a second request an act
         sends is counted before the first one's answer is read. */
      let body = {};
      try { body = JSON.parse((init && init.body) || '{}'); } catch (e) { body = {}; }
      const entry = {fn_index: body.fn_index, trigger_id: body.trigger_id, event_id: null};
      q.joins.push(entry);
      q.inflight += 1;
      sent.then((res) => res.clone().json()).then((j) => { entry.event_id = j.event_id; })
        .catch(() => {});
      return sent;
    }
    if (!url.includes('/queue/data')) { return sent; }
    return sent.then((res) => {
      if (!res.body) { return res; }
      const [mine, theirs] = res.body.tee();
      (async () => {
        const reader = theirs.getReader(), dec = new TextDecoder();
        let buf = '';
        for (;;) {
          const {done, value} = await reader.read();
          if (done) { break; }
          buf += dec.decode(value, {stream: true});
          let i;
          while ((i = buf.indexOf('\n\n')) >= 0) {
            const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
            try {
              const m = JSON.parse(chunk.replace(/^data: /, ''));
              if (m.msg === 'process_completed' && !(m.event_id in q.done)) {
                q.done[m.event_id] = m.success !== false;
                q.inflight -= 1;
              }
            } catch (e) { /* a heartbeat */ }
          }
        }
      })().catch(() => {});
      return new Response(mine, {status: res.status, statusText: res.statusText,
                                 headers: res.headers});
    });
  };
})();
"""


#: The page's own measure of what it draws (module docstring): installed with
#: ``page.addInitScript`` before the page loads.
LOOK_JS = r"""
(() => {
  const parse = (c) => {
    const m = c && c.match(/rgba?\(([^)]+)\)/);
    if (!m) { return null; }
    const p = m[1].split(',').map((x) => parseFloat(x));
    return { r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1 };
  };
  const lum = (c) => {
    const f = (v) => { v /= 255; return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
    return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b);
  };
  const ratio = (a, b) => {
    const x = typeof a === 'string' ? parse(a) : a, y = typeof b === 'string' ? parse(b) : b;
    const la = lum(x), lb = lum(y);
    return +(((Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05)).toFixed(2));
  };
  const behind = (el) => {
    for (let e = el; e; e = e.parentElement) {
      const c = parse(getComputedStyle(e).backgroundColor);
      if (c && c.a > 0.5) { return c; }
    }
    return { r: 255, g: 255, b: 255, a: 1 };
  };
  const shown = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) { return false; }
    const s = getComputedStyle(el);
    return s.visibility !== 'hidden' && s.display !== 'none' && +s.opacity > 0.05;
  };
  const typeScale = (root) => {
    const pairs = {}, upper = [], small = [];
    const walk = document.createTreeWalker(root || document.body, NodeFilter.SHOW_TEXT);
    for (let n = walk.nextNode(); n; n = walk.nextNode()) {
      const t = n.textContent.trim();
      if (!t) { continue; }
      const el = n.parentElement;
      if (!el || !shown(el) || el.closest('.sr-only, .shep-vh, script, style, [hidden]')) { continue; }
      const s = getComputedStyle(el);
      if (parseFloat(s.fontSize) < 14) { small.push(s.fontSize + ' ' + t.slice(0, 30)); }
      if (!/[A-Za-z0-9]/.test(t)) { continue; }
      const k = parseFloat(s.fontSize) + '/' + s.fontWeight;
      const p = pairs[k] || (pairs[k] = { n: 0, ex: [] });
      p.n += 1;
      if (p.ex.length < 4 && !p.ex.includes(t.slice(0, 30))) { p.ex.push(t.slice(0, 30)); }
      if (s.textTransform === 'uppercase') { upper.push(t.slice(0, 30)); }
    }
    return { pairs, upper, small };
  };
  const FIELDS = 'input:not([type=checkbox]):not([type=radio]):not([type=range]):not([type=file])' +
                 ':not([role=combobox]), textarea, .wrap:has(> .wrap-inner)';
  const fields = (root) => [...(root || document).querySelectorAll(FIELDS)]
    .filter((e) => shown(e) && !e.closest('#shep-help-texts'))
    .map((e) => {
      const s = getComputedStyle(e), w = parseFloat(s.borderTopWidth), c = parse(s.borderTopColor);
      return { id: (e.closest('[id]') || {}).id || '', tag: e.tagName, width: w, color: s.borderTopColor,
               background: s.backgroundColor,
               ratio: c && w > 0 && c.a > 0.5 ? ratio(c, behind(e.parentElement)) : 0,
               disabled: !!(e.disabled || (e.querySelector && e.querySelector('input:disabled'))) };
    });
  const LABELS = 'span[data-testid="block-info"], .shep-kv-field > .shep-k, .shep-rb-k, .shep-ft-k, ' +
                 '.shep-dx-card-title, .shep-dx-opt';
  /* A label is cut when it runs past the field it names: the nearest block
     drawn as a box (a block drawn with display: contents is its parent's). */
  const cutBy = (e) => {
    let b = e.parentElement && e.parentElement.closest('.block');
    while (b && getComputedStyle(b).display === 'contents') { b = b.parentElement && b.parentElement.closest('.block'); }
    if (!b) { return false; }
    const r = e.getBoundingClientRect(), k = b.getBoundingClientRect();
    return r.right > k.right + 1 || r.left < k.left - 1;
  };
  const labels = (root) => [...(root || document).querySelectorAll(LABELS)]
    .filter((e) => { const r = e.getBoundingClientRect(); return r.width > 4 && r.height > 4 && shown(e); })
    .map((e) => { const s = getComputedStyle(e);
      return { text: e.textContent.trim(), style: [s.fontSize, s.fontWeight, s.color, s.textTransform],
               cut: cutBy(e) }; });
  const pseudo = (el, which) => {
    const s = getComputedStyle(el, which), r = el.getBoundingClientRect();
    const left = parseFloat(s.left), top = parseFloat(s.top);
    return { content: s.content, display: s.display, fontSize: s.fontSize, fontWeight: s.fontWeight,
             textTransform: s.textTransform, pointerEvents: s.pointerEvents, color: s.color,
             left: r.left + left, top: r.top + top, width: parseFloat(s.width), height: parseFloat(s.height) };
  };
  const topAt = (x, y) => {
    const t = document.elementsFromPoint(x, y)[0];
    return t ? { tag: t.tagName, id: t.id, role: t.getAttribute('role'),
                 tab: !!t.closest('[role="tab"]'), cls: String(t.className).slice(0, 60) } : null;
  };
  /* The shared UI measurement check (D08): `check`, `focusWalk`. */
  const drawn = (e) => e.getClientRects().length > 0;
  const frame = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  const named = (e) => {
    if (e === document.scrollingElement) { return 'page'; }
    const holder = e.id ? null : e.closest('[id]'), cls = typeof e.className === 'string' ? e.className.trim() : '';
    return (e.id ? '#' + e.id : (holder ? '#' + holder.id + ' ' : '') + e.tagName.toLowerCase()
            + (cls ? '.' + cls.split(/\s+/).filter((c) => !c.startsWith('svelte-')).slice(0, 2).join('.') : ''));
  };
  /* The page shown (level 1, or the research tab within it) and its panels:
     the children of its innermost one-child box. */
  const shownPage = () => {
    const outer = document.querySelector('#shep-nav > .tab-wrapper [role="tab"][aria-selected="true"]');
    if (!outer) { return { tab: null, box: null, panels: [] }; }
    let tab = document.getElementById('shep-tab-' + outer.getAttribute('data-tab-id'));
    if (outer.getAttribute('data-tab-id') === 'research') {
      const inner = document.querySelector('#shep-research-nav > .tab-wrapper [role="tab"][aria-selected="true"]');
      tab = inner && document.getElementById('shep-page-' + inner.getAttribute('data-tab-id'));
    }
    let box = tab && [...tab.querySelectorAll('.shep-page')].find(drawn), kids = [];
    while (box) {
      kids = [...box.children].filter(drawn);
      if (kids.length === 1 && kids[0].children.length) { box = kids[0]; } else { break; }
    }
    return { tab, box, panels: kids };
  };
  /* An overlay a control opens and closes on its own: the help bubble (when
     its ⓘ is not `c`) and a select's open list (when not `c`'s). */
  const passing = (e, c) => (e.closest('#shep-help-bubble')
                             && !(c.matches('.shep-i') && c.getAttribute('aria-expanded') === 'true'))
    || (!!e.closest('ul.option-list, .options:has(> ul.option-list)') && !(c.closest('.block') || c).contains(e));
  /* Every region that scrolls vertically (the page itself when it does),
     with the area it is in (its panel, or an area the page marks inside one,
     `.shep-area`: the Diagnosis load drawer's list and its chosen model's
     details, side by side) and the regions it is inside. */
  const regions = () => {
    const se = document.scrollingElement, found = [];
    if (se.scrollHeight > se.clientHeight + 1) { found.push(se); }
    for (const e of document.body.querySelectorAll('*')) {
      if (e.scrollHeight <= e.clientHeight + 1 || !drawn(e)) { continue; }
      if (e.matches('textarea, select') || e.closest('#shep-help-bubble, ul.option-list')) { continue; }
      if (/(auto|scroll)/.test(getComputedStyle(e).overflowY) && shown(e)) { found.push(e); }
    }
    const { panels } = shownPage();
    return found.map((e) => {
      const k = e === se ? -1 : panels.findIndex((p) => p.contains(e));
      const fixed = e !== se && getComputedStyle(e).position === 'fixed';
      const own = k < 0 ? null : e.closest('.shep-area');
      const sub = own && own !== panels[k] && panels[k].contains(own) ? ' ' + named(own) : '';
      return { name: named(e), area: k < 0 ? 'page' : 'panel ' + (k + 1) + sub,
               inside: found.filter((o) => o !== e && (o === se ? !fixed : o.contains(e))).map(named) };
    });
  };
  /* The research tabs' group captions (a tab's ::before with content): where
     they sit against their tabs' line and what is under their centre. */
  const captions = () => {
    const tabs = [...document.querySelectorAll('#shep-research-nav > .tab-wrapper [role="tab"]')].filter(drawn);
    const boxes = tabs.map((t) => t.getBoundingClientRect());
    return tabs.map((t, i) => ({ t, r: boxes[i], c: pseudo(t, '::before') }))
      .filter(({ c }) => c.display !== 'none' && !['none', 'normal', '""', ''].includes(c.content) && c.width > 0)
      .map(({ t, r, c }) => ({
        tab: t.id, text: c.content, size: parseFloat(c.fontSize), pointer: c.pointerEvents,
        bottom: c.top + c.height, line: Math.min(...boxes.filter((b) => Math.abs(b.top - r.top) < 2).map((b) => b.top)),
        onTab: boxes.some((b) => c.left < b.right && c.left + c.width > b.left && c.top < b.bottom && c.top + c.height > b.top),
        hit: topAt(c.left + c.width / 2, c.top + c.height / 2) }));
  };
  /* What a choice drawn in the page says: a radio's or a segment's words. */
  const choices = () => [...document.querySelectorAll('label:has(> input[type=radio]), .shep-seg label, [role=radio]')]
    .filter((e) => drawn(e) && shown(e)).map((e) => e.textContent.trim());
  /* What one window of the page shows: no sideways page scroll, its scroll
     regions, its text, its fields, its labels, its captions, its choices. */
  const check = () => {
    const se = document.scrollingElement;
    return { overX: se.scrollWidth - se.clientWidth, regions: regions(), type: typeScale(), fields: fields(),
             labels: labels(), captions: captions(), choices: choices() };
  };
  /* Each control drawn, focused as a keyboard focuses it (the caller pressed
     a key first, so a script's focus is :focus-visible): is anything but an
     overlay of another control over its centre, and is its focus cue (what
     its focus changes: outline, shadow, border or fill, its own or its
     select's box) at 3:1 against what is behind it? An `auto` outline is
     the browser's own ring, not a token: 0. What its focus changes is read
     from it unfocused: the caller's key press (or an earlier step) may have
     focused it already, and a look read while focused changes nothing (the
     T06 review: Training's Continue drawer's Close, its first control, read
     "none" though it draws a 3 px ring). Blurred, a script's focus stays
     :focus-visible (the keyboard's modality is kept). */
  const look = (e) => { const s = getComputedStyle(e);
    return { outline: s.outlineStyle !== 'none' && parseFloat(s.outlineWidth) > 0
               ? [s.outlineStyle, s.outlineColor].join('|') : '',
             shadow: s.boxShadow, border: parseFloat(s.borderTopWidth) > 0 ? s.borderTopColor : '', fill: s.backgroundColor }; };
  const cue = (before, after, e) => {
    const bg = behind(e.parentElement), out = [];
    if (after.outline && after.outline !== before.outline) {
      const [style, colour] = after.outline.split('|');
      out.push(style === 'auto' ? 0 : ratio(colour, bg));
    }
    if (after.shadow !== before.shadow && after.shadow !== 'none') {
      const c = parse(after.shadow); if (c && c.a > 0.5) { out.push(ratio(c, bg)); }
    }
    if (after.border && after.border !== before.border) { out.push(ratio(after.border, bg)); }
    if (after.fill !== before.fill) { const c = parse(after.fill); if (c && c.a > 0.5) { out.push(ratio(c, bg)); } }
    return out;
  };
  const FOCUSABLE = 'a[href], button, input, textarea, select, summary, [tabindex]';
  const focusWalk = async (scope) => {
    const root = (scope && document.querySelector(scope)) || document;
    const controls = [...root.querySelectorAll(FOCUSABLE)].filter((c) => c.tabIndex >= 0 && !c.disabled
      && drawn(c) && c.offsetWidth > 2 && c.offsetHeight > 2 && shown(c) && !c.closest('#shep-help-texts'));
    const covered = [], cues = [];
    const settled = (c, box) => Promise.all([...c.getAnimations(), ...(box === c ? [] : box.getAnimations())]
      .map((a) => a.finished.catch(() => null)));
    for (const c of controls) {
      const box = c.closest('.wrap:has(> .wrap-inner)') || c;
      if (c.contains(document.activeElement)) { document.activeElement.blur(); await settled(c, box); }
      const before = [look(c), look(box)];
      c.focus();
      if (document.activeElement !== c) { continue; }
      if (c.matches('.shep-i')) { await frame(); }        // its bubble placed
      // Its focus cue drawn to the end (Gradio's buttons move every property).
      await settled(c, box);
      const r = c.getBoundingClientRect(), what = named(c) + ' "' +
        (c.getAttribute('aria-label') || c.textContent || c.value || '').trim().slice(0, 24) + '"';
      if (r.bottom <= 0 || r.top >= innerHeight || r.right <= 0 || r.left >= innerWidth) {
        covered.push(what + ' out of view'); continue;
      }
      const top = document.elementsFromPoint(r.left + r.width / 2, r.top + r.height / 2).find((e) => !passing(e, c));
      if (top && !c.contains(top) && !top.contains(c)) { covered.push(what + ' under ' + named(top)); }
      const ratios = [...cue(before[0], look(c), c), ...(box === c ? [] : cue(before[1], look(box), box))];
      if (!ratios.length || Math.max(...ratios) < 3) { cues.push(what + ' ' + (ratios.length ? Math.max(...ratios) : 'none')); }
    }
    if (document.activeElement && document.activeElement !== document.body) { document.activeElement.blur(); }
    // The overlays the walk opened close on their own (the bubble's hide, a list's fade).
    for (let i = 0; i < 60; i++) {
      const bubble = document.getElementById('shep-help-bubble');
      if ((!bubble || bubble.hidden) && !document.querySelector('ul.option-list')) { break; }
      await frame();
    }
    return { n: controls.length, covered, cues };
  };
  /* The top of each drawn element `sel` matches, rounded: values that should
     stand on one line (a row of result tiles) compared by their tops. */
  const tops = (sel) => [...document.querySelectorAll(sel)].filter(drawn)
    .map((e) => Math.round(e.getBoundingClientRect().top));
  /* An element's own text as drawn: how many lines it takes and where its
     last line ends (a field label's words against the control after it). */
  const textBox = (el) => {
    if (!el || !drawn(el)) { return null; }
    const range = document.createRange();
    range.selectNodeContents(el);
    const rects = [...range.getClientRects()].filter((r) => r.width > 0);
    if (!rects.length) { return null; }
    const lines = new Set(rects.map((r) => Math.round(r.top)));
    const last = rects.reduce((a, b) => (b.top > a.top + 1 || (Math.abs(b.top - a.top) <= 1 && b.right > a.right) ? b : a));
    return { lines: lines.size, right: Math.round(last.right), top: Math.round(rects[0].top) };
  };
  window.__shepLook = { parse, ratio, behind, shown, typeScale, fields, labels, pseudo, topAt,
                        shownPage, regions, captions, choices, check, focusWalk, tops, textBox };
})();
"""


#: The representative windows (D08): the owner's, 1280 wide, short and wide,
#: a short laptop's (1366 x 768 less the browser's own bars), portrait, a
#: narrow column and a phone.
UI_SIZES = ((1850, 960), (1280, 900), (1920, 700), (1366, 620), (1440, 1600), (960, 1080), (390, 844))

#: Every page, by the id its tab button carries (``shep-tab-<id>-button`` for
#: level 1, ``shep-page-<id>-button`` under Research).
UI_PAGES = ("clinical", "datasets", "training", "finetune", "test", "models", "settings")

#: The walk (`walk_ui`): one browser, one page, the stops in order. A stop
#: sets the window, opens its page if another is shown (a click, the tab
#: selected, the page's requests answered), waits until what the page draws
#: holds still for two frames and records the caller's `extra`; unless it
#: says ``"check": false`` it also records ``check()``, a screenshot (with a
#: directory for them) and ``focusWalk()`` (after one Tab press, so focus is
#: a keyboard's); a stop with ``options`` then opens each select drawn and
#: reads its options' words. A stop with ``open`` (``{"click": opener,
#: "shown": drawer, "close": its close}``, and ``"pick"``/``"picked"``: a
#: row to choose in it and what shows it was chosen) opens that drawer
#: before it measures, chooses that row, walks focus inside the drawer only
#: (the panels it covers are not its measure) and closes it again after.
UI_WALK_JS = r"""
const { chromium } = require('playwright');
const fs = require('fs');
const QUEUE_JS = __QUEUE_JS__, LOOK_JS = __LOOK_JS__, EXTRA = __EXTRA_JS__;
(async () => {
  const stops = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: stops[0].size[0], height: stops[0].size[1] } })).newPage();
  await page.addInitScript(QUEUE_JS);
  await page.addInitScript(LOOK_JS);
  const idle = () => page.waitForFunction(() => window.__shepQueue && window.__shepQueue.idle(),
                                          null, { timeout: 60000 });
  const settle = () => page.evaluate(async () => {
    const sign = () => { const { panels } = window.__shepLook.shownPage(), se = document.scrollingElement;
      return [document.documentElement.getAttribute('data-shep-layout'), se.scrollWidth, se.scrollHeight,
              ...panels.map((p) => { const r = p.getBoundingClientRect(); return [r.left, r.top, r.width, r.height].join(','); })].join('|'); };
    let last = null;
    for (let i = 0; i < 60; i++) {
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      const now = sign();
      if (now === last) { return i; }
      last = now;
    }
    return -1;
  });
  const button = (id) => (id === 'clinical' || id === 'settings' ? `#shep-tab-${id}-button` : `#shep-page-${id}-button`);
  // A stop's drawer (`stop.open`: its opener, the drawer and its close): opened
  // once the page is shown, its requests answered; closed again after the stop.
  const drawer = async (spec, open) => {
    await page.click(open ? spec.click : spec.close, { force: true });
    await page.waitForSelector(spec.shown, { state: open ? 'visible' : 'hidden' });
    await idle();
    if (open && spec.pick) {
      await page.waitForSelector(spec.pick, { state: 'visible' });
      await page.click(spec.pick);
      await page.waitForSelector(spec.picked, { state: 'attached' });
      await idle();
      await page.evaluate(() => window.scrollTo(0, 0));
    }
  };
  const open = async (id) => {
    if (!(id === 'clinical' || id === 'settings')
        && !(await page.$('#shep-tab-research-button[aria-selected="true"]'))) {
      await page.click('#shep-tab-research-button', { force: true });
      await page.waitForSelector('#shep-tab-research-button[aria-selected="true"]');
    }
    await page.click(button(id), { force: true });
    await page.waitForSelector(button(id) + '[aria-selected="true"]').catch(async (e) => {
      throw new Error(`${id} not opened: ` + JSON.stringify(await page.evaluate((s) => {
        const b = document.querySelector(s), r = b.getBoundingClientRect();
        return { at: [r.left, r.top, r.width, r.height], scrollY,
                 top: document.elementsFromPoint(r.left + r.width / 2, r.top + r.height / 2).slice(0, 3)
                   .map((x) => x.id || String(x.className).slice(0, 40)) }; }, button(id))) + ' ' + e.message);
    });
    await idle();
  };
  const options = async () => {
    // Each select by a locator, found again at each act: a page's answer can
    // draw a select anew between two of them.
    const words = [], selects = page.locator('input[role="combobox"]');
    for (let i = 0, n = await selects.count(); i < n; i++) {
      const input = selects.nth(i);
      if (!(await input.isVisible()) || !(await input.isEnabled())) { continue; }
      const list = await input.getAttribute('aria-controls');
      await input.click({ force: true });
      const opened = await page.waitForSelector(`#${list} [role=option]`, { timeout: 5000 }).then(() => true, () => false);
      if (opened) {
        words.push(...await page.$$eval(`#${list} [role=option]`,
                                        (os) => os.map((o) => o.getAttribute('aria-label') || o.textContent.trim())));
      }
      await page.keyboard.press('Escape');
      await page.evaluate(() => document.activeElement && document.activeElement.blur());
      await page.waitForSelector(`#${list}`, { state: 'detached', timeout: 5000 }).catch(() => {});
      await idle();
    }
    return words;
  };
  await page.goto(process.argv[2], { waitUntil: 'load' });
  await page.waitForSelector('#shep-chips .shep-chip', { timeout: 60000 });
  await idle();
  const shots = process.argv[4] || null;
  const out = [];
  let shown = null;
  for (const stop of stops) {
    const [w, h] = stop.size;
    await page.setViewportSize({ width: w, height: h });
    if (stop.page !== shown) { await open(stop.page); shown = stop.page; }
    await page.evaluate(() => window.scrollTo(0, 0));
    if (stop.open) { await drawer(stop.open, true); }
    const seen = { size: `${w}x${h}`, page: stop.page, frames: await settle() };
    if (stop.open) { seen.open = stop.open.shown; }
    if (EXTRA) { seen.extra = await page.evaluate(EXTRA); }
    if (stop.check !== false) {
      seen.check = await page.evaluate(() => window.__shepLook.check());
      const name = `${stop.page}${stop.open ? '_' + stop.open.shown.replace(/[^\w-]/g, '') : ''}_${w}x${h}`;
      if (shots) { await page.screenshot({ path: `${shots}/${name}.png` }); }
      await page.keyboard.press('Tab');
      seen.focus = await page.evaluate((scope) => window.__shepLook.focusWalk(scope), stop.open ? stop.open.shown : null);
      await page.evaluate(() => window.scrollTo(0, 0));
    }
    if (stop.options) { seen.options = await options(); }
    if (stop.open) { await drawer(stop.open, false); }
    out.push(seen);
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def walk_ui(page_url: str, tmp_path: Path, stops: list[dict[str, Any]], *,
            extra_js: str | None = None, shots: Path | None = None,
            timeout: int = 1500) -> list[dict[str, Any]]:
    """Walk the dashboard at `page_url` through `stops` (each ``{"size":
    [w, h], "page": id}``, with ``"check": false``, ``"options": true`` or
    ``"open": {...}`` where wanted; in order, `UI_WALK_JS`) and return what
    each stop recorded.
    `extra_js` is a function's source the page runs at each stop (its value is
    the stop's ``extra``); `shots`, a directory for a screenshot of each
    checked stop (``<page>_<w>x<h>.png``)."""
    script = (UI_WALK_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS))
              .replace("__LOOK_JS__", json.dumps(LOOK_JS))
              .replace("__EXTRA_JS__", extra_js or "null"))
    plan = tmp_path / "stops.json"
    plan.write_text(json.dumps(stops), encoding="utf-8")
    args = [page_url, str(plan)] + ([str(shots)] if shots else [])
    return run_node(script, tmp_path, *args, timeout=timeout)


#: The type scale (the WebUI skill §2): region title, body, keys; two weights.
TYPE_SIZES, TYPE_WEIGHTS = {"14", "15", "20"}, {"400", "600"}


def ui_problems(seen: list[dict[str, Any]]) -> list[str]:
    """What the shared UI measurement check finds in a walk (`walk_ui`), one
    line per problem, ``"<size> <page>: <what>"``; empty when every stop
    passes:

    - the page scrolls sideways;
    - a scroll region inside another one (the page included), or two in one
      area (a panel, or an area a page marks inside one, ``shep-area``): one
      scroll region per area;
    - a focused control out of view, or with anything but another control's
      own overlay over its centre;
    - text under 14 px, a size or weight outside the type scale, text drawn
      upper case;
    - an enabled field's border, or a control's focus cue, under 3:1;
    - field labels drawn in more than one style on a page, or a label its
      field cuts (a box squeezed under its label's width);
    - a group caption on its tabs' line, over a tab, hit-tested as a tab,
      taking the pointer, or under 14 px;
    - a choice or a select's option with more than one " · "."""
    problems: list[str] = []
    for stop in (s for s in seen if "check" in s):
        where, check = f"{stop['size']} {stop['page']}", stop["check"]

        def found(what: str, where: str = where) -> None:
            problems.append(f"{where}: {what}")

        if check["overX"] > 0:
            found(f"the page scrolls sideways by {check['overX']} px")
        areas: dict[str, list[str]] = {}
        for region in check["regions"]:
            areas.setdefault(region["area"], []).append(region["name"])
            if region["inside"]:
                found(f"scroll region {region['name']} inside {region['inside']}")
        for area, names in areas.items():
            if len(names) > 1:
                found(f"{len(names)} scroll regions in {area}: {names}")
        for what in stop["focus"]["covered"]:
            found(f"focused {what}")
        for what in stop["focus"]["cues"]:
            found(f"focus cue under 3:1: {what}")
        type_ = check["type"]
        for text in type_["small"]:
            found(f"text under 14 px: {text}")
        for pair, facts in type_["pairs"].items():
            size, weight = pair.split("/")
            if size not in TYPE_SIZES or weight not in TYPE_WEIGHTS:
                found(f"{pair} outside the type scale: {facts['ex']}")
        if type_["upper"]:
            found(f"text drawn upper case: {type_['upper'][:4]}")
        for field in check["fields"]:
            if not field["disabled"] and field["ratio"] < 3:
                found(f"field border {field['ratio']}:1 ({field['id']})")
        styles = {tuple(label["style"]) for label in check["labels"]}
        if len(styles) > 1:
            found(f"field labels in {len(styles)} styles: {sorted(styles)}")
        for label in check["labels"]:
            if label.get("cut"):
                found(f"field label {label['text']!r} cut by its field")
        for cap in check["captions"]:
            if (cap["bottom"] > cap["line"] + 0.5 or cap["onTab"] or (cap["hit"] or {}).get("tab")
                    or cap["pointer"] != "none" or cap["size"] < 14):
                found(f"group caption {cap['text']} is not apart from the tabs: {cap}")
        for words in (*check["choices"], *stop.get("options", ())):
            if words.count(" · ") > 1:
                found(f"more than one ' · ' in {words!r}")
    return problems


#: **What the shared model list's tree shows, read where the browser draws it**
#: (D04, `model_list._tree_html`; a page init script): ``window.__shepTree(root)``
#: gives its groups (each title, whether unfolded, its runs and files drawn),
#: its runs (words, whether unfolded) and its files (key, words, selected),
#: each drawn or folded away by what the page shows (`hidden`, a box), the
#: count line beside it, and a selection the list says it does not show.
TREE_JS = r"""
(() => {
  window.__shepTree = (root) => {
    const el = document.querySelector(root);
    if (!el) { return null; }
    const shown = (t) => !t.hidden && t.getBoundingClientRect().height > 0;
    const text = (t) => (t ? t.innerText.replace(/\s+/g, ' ').trim() : null);
    const opened = (b) => !!b && b.getAttribute('aria-expanded') === 'true';
    return {
      groups: [...el.querySelectorAll('tbody[data-node]')].map((body) => ({
        node: body.getAttribute('data-node'),
        title: text(body.querySelector('.shep-grp-t')),
        open: opened(body.querySelector('tr.shep-grp button.shep-fold')),
        runs: [...body.querySelectorAll('tr.shep-tree-run')].filter(shown).length,
        files: [...body.querySelectorAll('tr[data-key]')].filter(shown).length,
      })),
      runs: [...el.querySelectorAll('tr.shep-tree-run')].map((tr) => ({
        node: tr.getAttribute('data-fold'), words: text(tr.querySelector('td')),
        shown: shown(tr), open: opened(tr.querySelector('button.shep-fold')),
        context: tr.classList.contains('shep-ctx'),
      })),
      files: [...el.querySelectorAll('tr[data-key]')].map((tr) => ({
        key: tr.getAttribute('data-key'), words: text(tr.querySelector('td')),
        shown: shown(tr), selected: tr.getAttribute('aria-current') === 'true',
      })),
      outside: text(el.querySelector('.shep-list-outside')),
      fit: fit(el.querySelector('table.shep-mtree')),
    };
  };
  // Whether the tree fits its frame (its width against its wrapper's), and every
  // word a shown row splits over two lines (D8: a narrow frame never cuts a word).
  const fit = (table) => {
    if (!table) { return null; }
    const wrap = table.closest('.shep-tbl-wrap');
    const split = [];
    const walk = document.createTreeWalker(table, NodeFilter.SHOW_TEXT);
    for (let node = walk.nextNode(); node; node = walk.nextNode()) {
      for (const m of node.data.matchAll(/\S+/g)) {
        const range = document.createRange();
        range.setStart(node, m.index);
        range.setEnd(node, m.index + m[0].length);
        if (range.getClientRects().length > 1) { split.push(m[0]); }
      }
    }
    return { over: Math.round(table.getBoundingClientRect().width - wrap.clientWidth), split };
  };
})();
"""


def run_node(script: str, tmp_path: Path, *args: str, timeout: int = 400) -> Any:
    """Run a Node script (Playwright resolvable, `node_with_playwright`; the
    caller skips without it) and return the JSON it prints."""
    found = node_with_playwright()
    assert found is not None, "Node with Playwright is not installed"
    node, env = found
    path = tmp_path / "page.cjs"
    path.write_text(script, encoding="utf-8")
    result = subprocess.run([node, str(path), *args], env=env, capture_output=True, text=True,
                            timeout=timeout)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


__all__ = ["LOOK_JS", "QUEUE_JS", "TREE_JS", "TYPE_SIZES", "TYPE_WEIGHTS", "UI_PAGES", "UI_SIZES",
           "UI_WALK_JS",
           "node_with_playwright", "run_node", "ui_problems", "walk_ui"]
