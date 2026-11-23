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
  and the first opaque background behind an element;
- ``words(el)``: a block's words as drawn (its ``innerText``, trimmed), never
  the words of a Gradio status tracker that is not drawn (a drawn one's are
  read like any other): the one way a walk's fact reader reads words.

``typeScale`` also lists every visible text under 14 px, a glyph included.

**The shared UI measurement check** (D08, INT-AUDIT): `LOOK_JS` also measures
one window of the page shown, ``check()`` (no sideways page scroll, its scroll
regions with their panel and the regions they are inside, ``typeScale``,
``fields``, ``labels``, the stage hint of the research pages' row against
the row and its tabs, the words of its radios and segments) and
``focusWalk(scope)``: each control drawn (in ``scope`` when given), focused as
a keyboard focuses it, what is over its centre (another control's own
overlay, the help bubble or an open list it leaves behind, aside) and its
focus cue's contrast. `walk_ui` drives one browser through a list of stops
(`UI_WALK_JS`: a page at a window size, with a drawer of it opened where a
stop asks; the page's requests answered and its drawing still before it
measures; the selects' options read where a stop asks; a drawer opened, and
a row of it chosen, where a stop asks), and `ui_problems`
turns what it saw into one line per broken rule. The representative windows
are `UI_SIZES`, the pages `UI_PAGES`. The walk also counts each fingerprint
drawn in the page's visible text (``repeats``; `repeat_problems`), can run a
diagnosis at a stop (``diagnose``), and can walk as a read-only viewer
(`READ_ONLY_HEADERS`) or on a high-DPI screen (``device_scale``);
`PTR_SIZES` are PLAN_TEST_RESULTS §6's acceptance windows the representative
ones leave out (T06-D: INT-AUDIT, PTR-LAYOUT).

**The layout rules** (the UI sweep of 2026-11-22): `LOOK_JS`'s ``layout``
(recorded in every checked stop's ``check``) measures what the sweep's
findings were (a message beside fields, a squeezed box, a choice group
leaving one option alone, two left edges, a notice said twice, the tab
groups' separators) and what the L6 after screenshots saw that those missed (a
page cut at its foot with no scroll cue, words cut by a box, a wrapped group
pushed to the line's end, a disabled button with nothing said, a token or a
fact broken where it fits, labels of one row off one line); `layout_problems`
turns it into lines. A walk can serve the dashboard at first use or with an
empty workspaces folder (`walk_state`, `WALK_STATES`) and measure each page at
the layout windows (`layout_stops`, `LAYOUT_SIZES`); a stop measures once the
page's requests have been answered for ten frames in a row (a page's opening
chain is idle for an instant between two of its events). **The display
floor** (`floor_size`, `at_floor`; the owner's decision of 2026-11-23): the
walk's windows are at or above it but one, which is drawn as at the floor
(``check``'s ``floor``: the layout pass's marks, the frame, the document's
scroll), and ``in-view`` and ``sideways-scroll`` hold at or above it.

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
     queue (``queue=False``, answered by its own request); and the page's fonts
     are in (`served` serves them; drawn in a fallback first, a text is drawn
     again when its font arrives). */
  q.idle = () => q.inflight === 0 && q.direct === 0 && q.joins.every((j) => j.event_id !== null)
    && document.fonts.status === 'loaded';
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
  /* The stage hint (the shell's, after the research pages' tablist), when
     drawn: its words, whether it lies inside the row and off every tab, is
     cut, sits in the tablist, or could be taken for a control (a role, a
     focus stop, a link or button, a live region, the pointer's hand). */
  const stageHint = () => {
    const row = document.querySelector('#shep-research-nav > .tab-wrapper');
    const hint = row && row.querySelector(':scope > .shep-stage');
    if (!hint || !drawn(hint) || !shown(hint)) { return null; }
    const r = hint.getBoundingClientRect(), w = row.getBoundingClientRect();
    const tabs = [...row.querySelectorAll('[role="tab"]')].filter(drawn).map((t) => t.getBoundingClientRect());
    const CONTROL = 'a, button, input, select, textarea, [role], [tabindex], [aria-live], [contenteditable]';
    return { words: words(hint).replace(/\s+/g, ' '), size: parseFloat(getComputedStyle(hint).fontSize),
             inRow: r.left >= w.left - 0.5 && r.right <= w.right + 0.5 && r.top >= w.top - 0.5 && r.bottom <= w.bottom + 0.5,
             overTab: tabs.some((b) => r.left < b.right - 0.5 && r.right > b.left + 0.5 && r.top < b.bottom - 0.5 && r.bottom > b.top + 0.5),
             cut: hint.scrollWidth > hint.clientWidth + 1 || hint.scrollHeight > hint.clientHeight + 1,
             inTablist: !!hint.closest('[role="tablist"]'),
             control: hint.matches(CONTROL) || !!hint.querySelector(CONTROL) || getComputedStyle(hint).cursor === 'pointer' };
  };
  /* What a choice drawn in the page says: a radio's or a segment's words. */
  const choices = () => [...document.querySelectorAll('label:has(> input[type=radio]), .shep-seg label, [role=radio]')]
    .filter((e) => drawn(e) && shown(e)).map((e) => e.textContent.trim());

  /* **The layout rules** (the UI sweep of 2026-11-22; `layout_problems`): what the page shown
     draws, each measured relative to what is drawn, never against a window size or a pixel
     figure of one screen:
     - beside: a text block (no field in it) drawn on more than one line that stands beside a
       field or a button of its region (its panel; outside every panel, its bar: Training's run
       bar), overlapping its line by half the control's height: only a one-line relation sits
       beside the control it explains (the skill's §2); the words of one area a page marks
       (`.shep-area`: a list and its chosen item's details) beside another area's button aside;
     - narrow: a text box or a select whose inner width holds fewer than 10 of its own "0";
     - overlap: two controls that overlap by more than 2 x 2 px (a ⓘ's cover over its disabled
       button aside, and a control of an open drawer where the drawer is drawn over the other: a
       layer, absolute or fixed and no control, that holds one and not the other and is the top
       element where they meet);
     - orphan: a choice group on two or more lines whose last line holds one option and whose
       first holds more;
     - centred: a field or a button centred in a row more than two and a half times its own
       height, at least half its height below the row's top (relations to what is drawn, no px
       figure; for the pages' 39-40 px buttons no looser than the twice-plus-24-px it replaced:
       Fine-tune's Start in its 105 px row and Check in its 115 px one are found);
     - split: the controls of one group (`.shep-grp`) not on one line (a ⓘ, a Copy and a
       "Details ›" link aside);
     - edge: in a panel's left part, a heading, paragraph, reason, notice, empty state, field
       label or field box whose left edge is 1 px to 2em off the panel heading's (lists, tables,
       term rows and the words after a status mark aside);
     - clipped: a control cut by an ancestor that hides its overflow and is no scroll region (a
       box of ``display: contents`` draws no box, so it clips nothing);
     - empty: a panel with no visible words or control under its header;
     - help: a registered help text drawn on the page outside the bubble;
     - notices: each notice key (`data-notice`, `data-notice-also`) and how often it is drawn;
     - tabs: the research tabs' lines, their order, and each separator drawn while the tabs wrap
       or at a line's start, and anything in the tablist that is not a tab. */
  const INTERACTIVE = 'button, input:not([type=hidden]), textarea, select, a[href], [role=button], '
    + '[role=tab], [role=radio], [role=checkbox], [role=switch], summary';
  const SKIP = 'script, style, noscript, .sr-only, .shep-vh, [hidden], #shep-help-texts, '
    + '#shep-help-bubble, ul.option-list';
  const BOXES = 'input[type=text], input[type=number], input[type=search], input:not([type]), '
    + 'input[role=combobox], textarea';
  const visible = (e) => {
    if (!shown(e)) { return false; }
    for (let a = e; a && a !== document.body; a = a.parentElement) {
      if (+getComputedStyle(a).opacity < 0.05) { return false; }
    }
    return true;
  };
  const blockOf = (el) => {
    let e = el;
    while (e && e !== document.body && /^(inline|contents)/.test(getComputedStyle(e).display)) { e = e.parentElement; }
    return e;
  };
  const zeroOf = (e) => {
    const s = getComputedStyle(e);
    const c = (zeroOf.c || (zeroOf.c = document.createElement('canvas'))).getContext('2d');
    c.font = `${s.fontStyle} ${s.fontWeight} ${s.fontSize} ${s.fontFamily}`;
    return c.measureText('0').width || parseFloat(s.fontSize) * 0.55;
  };
  const textNodes = (root) => {
    const out = [], w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let n = w.nextNode(); n; n = w.nextNode()) {
      const t = n.textContent.trim(), el = n.parentElement;
      if (!t || !el || el.closest(SKIP) || !visible(el)) { continue; }
      const range = document.createRange();
      range.selectNodeContents(n);
      const rects = [...range.getClientRects()].filter((r) => r.width > 0.5 && r.height > 0.5);
      if (rects.length) { out.push({ node: n, el, t, rects }); }
    }
    return out;
  };
  const clip = (s, n = 60) => s.replace(/\s+/g, ' ').trim().slice(0, n);
  /* How many lines an element's words are drawn on: its text's boxes in order, a box that
     overlaps the line before it by half its own height on that line (a status mark and the
     words after it share one). */
  const linesOf = (el) => {
    const range = document.createRange();
    range.selectNodeContents(el);
    const rects = [...range.getClientRects()].filter((q) => q.width > 0.5 && q.height > 0.5)
      .sort((a, b) => a.top - b.top);
    let n = 0, bottom = -Infinity;
    for (const q of rects) {
      if (q.top > bottom - q.height / 2) { n += 1; bottom = q.bottom; } else { bottom = Math.max(bottom, q.bottom); }
    }
    return n;
  };
  /* A ⓘ's cover over the disabled button it explains (the gated-button rule) is meant to lie
     over it. */
  const coverOf = (a, b) => a.matches('.shep-i-cover') || b.matches('.shep-i-cover');
  /* The part of an element its ancestors that clip or scroll leave in view. */
  const seenRect = (e) => {
    const r = e.getBoundingClientRect();
    let L = r.left, T = r.top, R = r.right, B = r.bottom;
    for (let a = e.parentElement; a && a !== document.documentElement; a = a.parentElement) {
      const s = getComputedStyle(a);
      if (s.display === 'contents') { continue; }
      if (s.overflowX !== 'visible' || s.overflowY !== 'visible') {
        const q = a.getBoundingClientRect();
        if (s.overflowX !== 'visible') { L = Math.max(L, q.left); R = Math.min(R, q.right); }
        if (s.overflowY !== 'visible') { T = Math.max(T, q.top); B = Math.min(B, q.bottom); }
      }
      if (s.position === 'fixed') { break; }
    }
    return { left: L, top: T, right: R, bottom: B, width: Math.max(0, R - L), height: Math.max(0, B - T),
             whole: r.width * r.height };
  };
  /* **A control drawn in an open drawer, over another** (`overlap`): `a` lies in a layer that
     does not hold `b` (a box drawn absolute or fixed that is no control, below the first box
     they share), and at the point (x, y) that layer, not `b`, is drawn. A control drawn over
     another in their own box (a Clear over a switch) is an overlap. */
  const drawnOver = (a, b, x, y) => {
    let layer = null;
    for (let p = a.parentElement; p && p !== document.body && !p.contains(b); p = p.parentElement) {
      if (/^(absolute|fixed)$/.test(getComputedStyle(p).position) && !p.matches(INTERACTIVE)) { layer = p; break; }
    }
    const top = layer && document.elementFromPoint(x, y);
    return !!top && layer.contains(top);
  };
  /* Where an element's own words or box start: a box (a field, a select's wrap) at its border,
     text at its content edge (after its padding and border). */
  const startOf = (e) => {
    const r = e.getBoundingClientRect();
    if (e.matches('input, textarea, .wrap')) { return r.left; }
    const s = getComputedStyle(e);
    return r.left + (parseFloat(s.paddingLeft) || 0) + (parseFloat(s.borderLeftWidth) || 0);
  };
  const layout = () => {
    const shownNow = shownPage(), tab = shownNow.tab;
    /* The page's panels: its outermost `.shep-panel`s (a run bar, a drawer included), else the
       children of its innermost one-child box. */
    const outer = tab ? [...tab.querySelectorAll('.shep-panel')]
      .filter((p) => visible(p) && !p.parentElement.closest('.shep-panel')) : [];
    const panels = outer.length ? outer
      : shownNow.panels.filter((p) => !p.matches('[data-testid="status-tracker"]'));
    const scope = tab || document.body, out = { beside: [], narrow: [], overlap: [], orphan: [], centred: [],
      split: [], edge: [], clipped: [], empty: [], help: [], notices: {}, tabs: null, inview: [], cue: [],
      wrapEnd: [], disabled: [], token: [], sepEnd: [], chain: [], rowTop: [], rowHeight: [], sideways: [] };
    const panelOf = (e) => panels.findIndex((p) => p.contains(e));
    /* A text's region for `beside`: its panel, else the outermost bar it is in (Training's run
       bar is no panel), else none (-1 on a page with no panels). */
    const regionOf = (e) => {
      const k = panelOf(e);
      if (k >= 0) { return k; }
      let bar = null;
      for (let b = e.closest('.shep-bar'); b; b = b.parentElement && b.parentElement.closest('.shep-bar')) { bar = b; }
      return bar || (panels.length ? null : -1);
    };
    const ctl = [...scope.querySelectorAll(INTERACTIVE)].filter((e) => !e.closest(SKIP) && visible(e));
    const fieldsAndButtons = ctl.filter((c) => c.matches('input, textarea, button, [role=combobox]')
      && !c.matches('.shep-i, .shep-copy, .shep-nb-go, .shep-link'));
    const texts = textNodes(scope);
    // beside
    const blocks = new Set(texts.map((x) => blockOf(x.el)));
    for (const b of blocks) {
      if (!b || b === scope || b.querySelector('input, textarea, select')) { continue; }
      // A table's cells, a summary list's rows and a control's own words are their own forms,
      // not a message beside a control.
      if (b.closest('table, dl, label, [role=tablist]') || b.closest(INTERACTIVE)) { continue; }
      // A closed disclosure's body is laid out but not drawn.
      if (b.closest('details:not([open])') && !b.closest('summary')) { continue; }
      const words = (b.innerText || '').trim();
      if (linesOf(b) < 2) { continue; }
      const r = b.getBoundingClientRect(), k = regionOf(b);
      if (k === null) { continue; }
      const near = fieldsAndButtons.filter((c) => {
        if (regionOf(c) !== k || b.contains(c)) { return false; }
        const q = c.getBoundingClientRect(), over = Math.min(q.bottom, r.bottom) - Math.max(q.top, r.top);
        if (!(over > 0.5 * q.height && (q.right <= r.left + 2 || q.left >= r.right - 2))) { return false; }
        // Beside it in one row, not a column of a form beside another: the branch that holds the
        // words, under the box that holds both, has no field of its own.
        let branch = b;
        while (branch.parentElement && !branch.parentElement.contains(c)) { branch = branch.parentElement; }
        // Two areas a page marks side by side (`.shep-area`: a list and its chosen item's
        // details) are two regions: the details' words are not a message beside the list.
        const area = b.closest('.shep-area');
        if (area && !area.contains(c)) { return false; }
        return !branch.querySelector('input:not([type=hidden]), textarea, select');
      });
      if (near.length) { out.beside.push(`${named(b)} "${clip(words, 50)}" beside ${named(near[0])}`); }
    }
    // narrow
    for (const f of [...scope.querySelectorAll(BOXES)].filter((e) => !e.closest(SKIP) && visible(e))) {
      const s = getComputedStyle(f), r = f.getBoundingClientRect();
      const inner = r.width - parseFloat(s.paddingLeft) - parseFloat(s.paddingRight), zeros = inner / zeroOf(f);
      if (zeros < 10) { out.narrow.push(`${named(f.closest('.block[id], [id]') || f)} holds ${zeros.toFixed(1)} "0"`); }
    }
    // overlap and clipped
    const boxes = ctl.map((e) => [e, seenRect(e)]).filter(([, r]) => r.width > 2 && r.height > 2
                                                         && r.width * r.height > 0.5 * r.whole);
    for (let i = 0; i < boxes.length; i++) {
      for (let j = i + 1; j < boxes.length; j++) {
        const [a, ra] = boxes[i], [b, rb] = boxes[j];
        if (a.contains(b) || b.contains(a) || coverOf(a, b) || coverOf(b, a)) { continue; }
        const la = a.closest('label');
        if (la && la === b.closest('label')) { continue; }
        const w = Math.min(ra.right, rb.right) - Math.max(ra.left, rb.left);
        const h = Math.min(ra.bottom, rb.bottom) - Math.max(ra.top, rb.top);
        if (w <= 2 || h <= 2) { continue; }
        // Where they meet, an open drawer holding one is drawn over the other: no overlap.
        const x = Math.max(ra.left, rb.left) + w / 2, y = Math.max(ra.top, rb.top) + h / 2;
        if (drawnOver(a, b, x, y) || drawnOver(b, a, x, y)) { continue; }
        out.overlap.push(`${named(a)} and ${named(b)} by ${Math.round(w)} x ${Math.round(h)} px`);
      }
    }
    // clipped: cut by an ancestor that hides overflow, before any scroll region (what a scroll
    // region holds beyond its box is scrolled out of view, not cut)
    for (const e of ctl) {
      const r = e.getBoundingClientRect();
      if (r.width <= 2 || r.height <= 2) { continue; }
      for (let a = e.parentElement; a && a !== document.documentElement; a = a.parentElement) {
        const s = getComputedStyle(a);
        if (s.display === 'contents') { continue; }
        if (s.position === 'fixed' || /(auto|scroll)/.test(s.overflowX + ' ' + s.overflowY)) { break; }
        if (!/(hidden|clip)/.test(s.overflowX + ' ' + s.overflowY)) { continue; }
        const q = a.getBoundingClientRect();
        const cut = (/(hidden|clip)/.test(s.overflowX) && (r.left < q.left - 1 || r.right > q.right + 1))
          || (/(hidden|clip)/.test(s.overflowY) && (r.top < q.top - 1 || r.bottom > q.bottom + 1));
        if (cut && getComputedStyle(e).textOverflow !== 'ellipsis') { out.clipped.push(`${named(e)} cut by ${named(a)}`); break; }
      }
    }
    // centred
    const rows = new Set();
    for (const c of fieldsAndButtons) {
      const q = c.getBoundingClientRect();
      for (let a = c.parentElement; a && a !== scope; a = a.parentElement) {
        const s = getComputedStyle(a), ar = a.getBoundingClientRect();
        if ((s.display.includes('flex') && s.flexDirection.startsWith('row')) || s.display.includes('grid')) {
          // A grid's row is its row track: of a grid of several rows, the tracks the control's box
          // spans (Models' foot stacks its details, Load, Load's reason and its note in four
          // tracks at 390 px: Load stood near the foot's middle in a track of its own height).
          let top = ar.top, bottom = ar.bottom;
          const tracks = s.display.includes('grid')
            ? s.gridTemplateRows.split(' ').map(parseFloat).filter((n) => !isNaN(n)) : [];
          if (tracks.length > 1) {
            let y = ar.top + parseFloat(s.borderTopWidth) + parseFloat(s.paddingTop), first = null, last = null;
            for (const t of tracks) {
              if (y + t > q.top + 0.5 && y < q.bottom - 0.5) { first = first === null ? y : first; last = y + t; }
              y += t + (parseFloat(s.rowGap) || 0);
            }
            if (first !== null) { top = first; bottom = last; }
          }
          const h = bottom - top;
          if (h > 2.5 * q.height && q.top - top >= q.height / 2
              && Math.abs((q.top + q.bottom) / 2 - (top + bottom) / 2) < 0.2 * h && !rows.has(a)) {
            rows.add(a);
            out.centred.push(`${named(c)} in ${named(a)} (${Math.round(h)} px tall)`);
          }
          break;
        }
        if (ar.height > 1.2 * q.height) { break; }
      }
    }
    // orphan
    const groups = new Set([...scope.querySelectorAll('input[type=radio], input[type=checkbox]')]
      .filter((i) => visible(i.closest('label') || i)).map((i) => (i.closest('label') || i).parentElement));
    for (const g of groups) {
      const items = [...g.children].filter((c) => visible(c) && c.querySelector('input[type=radio], input[type=checkbox]'));
      if (items.length < 2) { continue; }
      const lines = {};
      for (const it of items) { const k = Math.round(it.getBoundingClientRect().top / 6); lines[k] = (lines[k] || 0) + 1; }
      const counts = Object.keys(lines).sort((a, b) => a - b).map((k) => lines[k]);
      if (counts.length > 1 && counts[counts.length - 1] === 1 && counts[0] > 1) {
        out.orphan.push(`${named(g.closest('.block[id]') || g)}: ${counts.join(' + ')} options a line`);
      }
    }
    // split
    for (const g of scope.querySelectorAll('.shep-grp')) {
      const parts = [...g.querySelectorAll('input:not([type=radio]):not([type=checkbox]), textarea, button, .wrap:has(> .wrap-inner)')]
        .filter((c) => visible(c) && !c.matches('.shep-i, .shep-copy, .shep-link') && !c.closest('.wrap:has(> .wrap-inner) input') )
        .map((c) => c.getBoundingClientRect());
      if (parts.length > 1 && Math.max(...parts.map((r) => r.top)) >= Math.min(...parts.map((r) => r.bottom))) {
        out.split.push(`${named(g)} on ${new Set(parts.map((r) => Math.round(r.top / 8))).size} lines`);
      }
    }
    // edge
    const EDGE = 'h2, h3, p, .shep-reason, .shep-nb, .shep-empty, span[data-testid="block-info"], '
      + 'input[type=text], input[type=number], textarea, .wrap:has(> .wrap-inner)';
    const HANG = 'li, td, th, dd, dt, .shep-nb-text, .shep-nb-next, .shep-nb-steps, .shep-nb-more, '
      + '.shep-kv, table, .shep-seg, [role=tablist], .shep-bubble, .shep-panel-hd .shep-panel-right';
    for (const p of panels) {
      const head = p.querySelector('.shep-panel-hd h2, .shep-panel-hd h3, h2');
      if (!head || !visible(head)) { continue; }
      const pr = p.getBoundingClientRect(), edge = startOf(head);
      const em = parseFloat(getComputedStyle(head).fontSize) * 2;
      const off = [...p.querySelectorAll(EDGE)].filter((e) => visible(e) && !e.closest(HANG) && !e.closest(SKIP))
        .filter((e) => { const r = e.getBoundingClientRect(); return r.left < pr.left + 0.4 * pr.width && getComputedStyle(e).textAlign !== 'center'; })
        .map((e) => [e, Math.round(startOf(e) - edge)])
        .filter(([, dx]) => Math.abs(dx) > 1 && Math.abs(dx) <= em);
      for (const [e, dx] of off.slice(0, 6)) { out.edge.push(`${named(e)} ${dx > 0 ? '+' : ''}${dx} px off ${named(head)}`); }
    }
    // empty
    for (const p of panels) {
      const said = textNodes(p).some((x) => !x.el.closest('.shep-panel-hd') && /[A-Za-z0-9]/.test(x.t));
      const acts = [...p.querySelectorAll(INTERACTIVE)].some((e) => !e.closest(SKIP) && visible(e));
      if (!said && !acts) { out.empty.push(named(p)); }
    }
    // help texts drawn on the page
    const helps = [...document.querySelectorAll('#shep-help-texts [id^="shep-help-text-"]')]
      .map((h) => clip(h.textContent, 40)).filter((h) => h.length >= 20);
    for (const x of textNodes(document.body)) {
      const t = clip(x.t, 40);
      if (t.length >= 20 && helps.some((h) => h.startsWith(t) || t.startsWith(h))) { out.help.push(`${named(x.el)} "${t}"`); }
    }
    // notices
    for (const e of document.querySelectorAll('[data-notice], [data-notice-also]')) {
      if (!visible(e)) { continue; }
      for (const key of `${e.getAttribute('data-notice') || ''} ${e.getAttribute('data-notice-also') || ''}`.split(/\s+/)) {
        if (key) { out.notices[key] = (out.notices[key] || 0) + 1; }
      }
    }
    // the research tabs
    const list = document.querySelector('#shep-research-nav > .tab-wrapper > .tab-container');
    if (list && drawn(list)) {
      const kids = [...list.children].filter((k) => drawn(k));
      const tabs = kids.filter((k) => k.getAttribute('role') === 'tab');
      const lines = [], seps = [];
      let lastTop = null;
      for (const t of tabs) {
        const top = Math.round(t.offsetTop / 4), first = lastTop === null || top !== lastTop;
        if (first) { lines.push([]); }
        lines[lines.length - 1].push(t.getAttribute('data-tab-id'));
        const before = getComputedStyle(t, '::before');
        if (before.content !== 'none' && before.content !== 'normal' && before.display !== 'none'
            && parseFloat(before.width) > 0) { seps.push({ tab: t.getAttribute('data-tab-id'), lineStart: first }); }
        lastTop = top;
      }
      out.tabs = { lines, seps, others: kids.filter((k) => k.getAttribute('role') !== 'tab').map(named),
                   focusable: [...list.querySelectorAll('[tabindex]:not([role=tab])')].map(named) };
    }
    seeAll(scope, panels, texts, ctl, regionOf, panelOf, out);
    return out;
  };
  /* **What the after screenshots saw** (the sweep's L7: SW-LK1..10; `layout_problems`' in-view,
     scroll-cue, clipped text, wrap-end, disabled-reason, token-break, sep-end, row-top and
     row-height), each a relation to what is drawn, never a pixel figure of one screen:
     - inview: a control or a text line drawn past the window's bottom or right edge that no
       scroll region in view holds (`layout_problems` holds it at every window at or above the
       display floor, `at_floor`, where a page's task fits or one region scrolls inside, never
       the page; below the floor the document scrolls to it);
     - cue: a scroll region (the page's own tab item included, never the document) whose drawn
       content runs past its edge, below it with no scroll cue (the shared R-73 cue,
       ``shep-scroll-cue``: a mask drawn while more is below), sideways with no sideways cue (a
       mask, or the drawn cue just before it, Compare's ``shep-rs-cue``); what runs past a
       nested region is that region's own;
     - clipped (text): a text line cut by an ancestor that hides its overflow in that axis (no
       scroll region of that axis between them), a cut to an ellipsis included unless the whole
       words are its ``title``;
     - wrapEnd: in a bar (`.shep-bar`), a later line of groups that does not start on the bar's
       start edge (a group pushed to the end leaves the start empty);
     - disabled: a disabled button drawn with no shared reason (a reason line ``shep-reason``, a
       relation ``data-notice-rel``, its bar's note ``shep-bar-note``) in its group or bar, on its
       line or just under it (within its own height), unless it is no primary action and its
       region draws the notice or a relation; words a page drew there in a form of its own are
       named as such (C11: a disabled control is said one way on every page);
     - token: a literal token (``code``, ``shep-mono``, a fingerprint, ``shep-full``) whose line
       ends at a hyphen inside a word (it reads hyphenated), or that breaks a word that fits its
       box on a line of its own (a flag cut in two);
     - sepEnd: a separator (" · ") that ends a line, its next fact on the next line;
     - chain: a fact of a " · " chain (the words between two separators) drawn over two lines
       though it fits its block on a line of its own;
     - rowTop: two labelled fields side by side in one row of fields (their nearest shared box
       a row; each the one labelled field of its branch) whose labels do not share a top line
       (a quarter of the label's height apart or more);
     - rowHeight: a table row inside a sideways scroll box whose height is set by a cell out of
       the box's view (its words over one and a half times as tall as every cell in view, the
       row at least that tall);
     - sideways: a region whose drawn content runs past its right edge (it scrolls sideways),
       masked or not, unless a drawn ``shep-rs-cue`` just before it says so in words. */
  const seeAll = (scope, panels, allTexts, allCtl, regionOf, panelOf, out) => {
    // A closed disclosure's body is laid out (a geometry query lays it out) but not drawn.
    const folded = (e) => { const d = e.closest('details:not([open])'), s = e.closest('summary');
      return !!d && !(s && s.parentElement === d); };
    const texts = allTexts.filter((x) => !folded(x.el)), ctl = allCtl.filter((c) => !folded(c));
    const flow = (e, axis) => getComputedStyle(e)[axis === 'y' ? 'overflowY' : 'overflowX'];
    const scrolls = (e, axis) => /(auto|scroll)/.test(flow(e, axis));
    const hides = (e, axis) => /(hidden|clip)/.test(flow(e, axis));
    // An element's padding box: where its overflow is cut or scrolled.
    const pad = (e) => { const r = e.getBoundingClientRect(), left = r.left + e.clientLeft, top = r.top + e.clientTop;
      return { left, top, right: left + e.clientWidth, bottom: top + e.clientHeight }; };
    const masked = (e) => { const s = getComputedStyle(e), m = s.maskImage || s.webkitMaskImage;
      return !!m && m !== 'none'; };
    const say = (e) => clip((e.innerText || e.getAttribute('aria-label') || e.value || e.getAttribute('placeholder')
      || e.id || '').trim(), 30);
    // A box drawn as no box (``display: contents``: Gradio's blocks and forms around a field)
    // clips nothing and scrolls nothing.
    const boxed = (e) => getComputedStyle(e).display !== 'contents';
    // Words hidden on purpose and kept for a screen reader (a 1 px box that clips them, or a
    // ``clip`` rectangle: the icon Copy's label): never drawn, so never cut.
    const quiet = new Map();
    const unseen = (e) => {
      if (quiet.has(e)) { return quiet.get(e); }
      const s = getComputedStyle(e), r = e.getBoundingClientRect();
      const found = (s.display !== 'contents'
          && ((s.clip !== 'auto' && s.clip !== '' && s.position === 'absolute')
              || (/(hidden|clip)/.test(s.overflowX + s.overflowY) && (r.width <= 1 || r.height <= 1))))
        || (!!e.parentElement && e.parentElement !== document.body && unseen(e.parentElement));
      quiet.set(e, found);
      return found;
    };
    // What a page draws, each line of text and each control, with the element it belongs to.
    const items = [...texts.flatMap((x) => x.rects.map((r) => ({ el: x.el, r, t: x.t }))),
                   ...ctl.map((c) => ({ el: c, r: c.getBoundingClientRect(), t: say(c) }))]
      .filter((it) => it.r.width > 0.5 && it.r.height > 0.5 && !unseen(it.el));
    // How far past an edge, downwards, a line or a control runs before something of it is cut:
    // a quarter of its own height (a line's box holds its half-leading, a ⓘ's box its target's
    // room round the drawn mark); sideways a glyph run ends at its last glyph, so 1 px.
    const past = (r) => Math.max(1, 0.25 * r.height);
    // The scroll region of an axis an element is in (itself, when it is one), below the document.
    const nearest = new Map();
    const scroller = (e, axis) => {
      const cache = nearest.get(e) || {};
      if (axis in cache) { return cache[axis]; }
      let found = null;
      for (let a = e; a && a !== document.body && a !== document.documentElement; a = a.parentElement) {
        if (scrolls(a, axis) && boxed(a)) { found = a; break; }
      }
      cache[axis] = found; nearest.set(e, cache);
      return found;
    };
    // Under an open layer (a drawer: a box drawn absolute or fixed, no control, that does not
    // hold it) where its middle is drawn: not seen at this stop, so no word or control of its is
    // judged there (a ⓘ's cover over its disabled button is no layer).
    const layered = (e) => {
      const r = e.getBoundingClientRect(), x = r.left + r.width / 2, y = r.top + r.height / 2;
      if (r.width < 1 || r.height < 1 || x < 0 || y < 0 || x > innerWidth || y > innerHeight) { return false; }
      const t = document.elementFromPoint(x, y);
      if (!t || e.contains(t) || t.contains(e) || t.closest('.shep-i-cover')) { return false; }
      for (let p = t; p && p !== document.body; p = p.parentElement) {
        if (p.contains(e)) { return false; }
        if (/^(absolute|fixed)$/.test(getComputedStyle(p).position) && !p.matches(INTERACTIVE)) { return true; }
      }
      return false;
    };
    // inview
    const outs = new Set();
    for (const it of items) {
      const outY = it.r.bottom > innerHeight + past(it.r), outX = it.r.right > innerWidth + 1;
      if ((!outY && !outX) || outs.has(it.el)) { continue; }
      let reach = false;
      for (let a = it.el; a && a !== document.body && a !== document.documentElement; a = a.parentElement) {
        if (((outY && scrolls(a, 'y')) || (outX && scrolls(a, 'x'))) && boxed(a)) {
          const q = a.getBoundingClientRect();
          if (q.bottom <= innerHeight + 1 && q.right <= innerWidth + 1) { reach = true; break; }
        }
      }
      if (reach) { continue; }
      outs.add(it.el);
      if (outs.size <= 8) {
        out.inview.push(`${named(it.el)} "${clip(it.t, 30)}" drawn ${outY ? Math.round(it.r.bottom - innerHeight) + ' px under the window' : Math.round(it.r.right - innerWidth) + ' px right of the window'}, in no scroll region in view`);
      }
    }
    if (outs.size > 8) { out.inview.push(`and ${outs.size - 8} more out of view`); }
    // cue
    const boxes = [scope, ...scope.querySelectorAll('*')].filter((e) => drawn(e) && visible(e)
      && !e.matches('textarea, select, input') && !e.closest('#shep-help-bubble, ul.option-list, #shep-help-texts'));
    for (const g of boxes) {
      const sy = scrolls(g, 'y') && g.scrollHeight > g.clientHeight + 1;
      const sx = scrolls(g, 'x') && g.scrollWidth > g.clientWidth + 1;
      if (!sy && !sx) { continue; }
      const box = pad(g), mine = items.filter((it) => g.contains(it.el));
      if (sy && !masked(g)) {
        const below = mine.filter((it) => scroller(it.el, 'y') === g && it.r.bottom > box.bottom + past(it.r))
          .sort((a, b) => a.r.top - b.r.top);
        if (below.length) {
          out.cue.push(`${named(g)} holds ${Math.round(Math.max(...below.map((it) => it.r.bottom)) - box.bottom)} px `
            + `of content under its edge with no scroll cue (first: "${clip(below[0].t, 30)}")`);
        }
      }
      const before = g.previousElementSibling;
      if (sx && !masked(g) && !(before && before.matches('.shep-rs-cue') && visible(before) && drawn(before))) {
        const right = mine.filter((it) => scroller(it.el, 'x') === g && it.r.right > box.right + 1)
          .sort((a, b) => a.r.left - b.r.left);
        if (right.length) {
          out.cue.push(`${named(g)} holds ${Math.round(Math.max(...right.map((it) => it.r.right)) - box.right)} px `
            + `of content past its right edge with no sideways cue (cut: "${clip(right[0].t, 30)}")`);
        }
      }
    }
    // sideways: a region whose drawn content runs past its right edge (it scrolls sideways), a
    // mask or not (the shell's automatic sideways mask hid a table wider than its frame: the
    // sweep's L7, fixer A's item 6); a region that says in words that it scrolls sideways (a
    // drawn ``shep-rs-cue`` just before it: Compare's columns, one per result) is meant to.
    for (const g of boxes) {
      if (!(scrolls(g, 'x') && g.scrollWidth > g.clientWidth + 1)) { continue; }
      const before = g.previousElementSibling;
      if (before && before.matches('.shep-rs-cue') && visible(before) && drawn(before)) { continue; }
      const box = pad(g), right = items.filter((it) => g.contains(it.el) && scroller(it.el, 'x') === g
        && it.r.right > box.right + 1).sort((a, b) => a.r.left - b.r.left);
      if (right.length) {
        out.sideways.push(`${named(g)} scrolls sideways: ${Math.round(Math.max(...right.map((it) => it.r.right)) - box.right)} px `
          + `of its content past its right edge (first: "${clip(right[0].t, 30)}")`);
      }
    }
    // clipped text
    const cutAt = new Set();
    for (const x of texts) {
      if (unseen(x.el)) { continue; }
      for (const r of x.rects) {
        let doneX = false, doneY = false, cut = null;
        for (let a = x.el; a && a !== document.body && a !== document.documentElement; a = a.parentElement) {
          const s = getComputedStyle(a);
          if (s.display === 'contents') { continue; }
          const q = pad(a);
          if (!doneX && hides(a, 'x') && (r.left < q.left - 1 || r.right > q.right + 1)) { cut = [a, 'x', Math.max(q.left - r.left, r.right - q.right)]; break; }
          if (!doneY && hides(a, 'y') && (r.top < q.top - past(r) || r.bottom > q.bottom + past(r))) { cut = [a, 'y', Math.max(q.top - r.top, r.bottom - q.bottom)]; break; }
          doneX = doneX || scrolls(a, 'x'); doneY = doneY || scrolls(a, 'y');
          if ((doneX && doneY) || s.position === 'fixed') { break; }
        }
        if (!cut || cutAt.has(x.node)) { continue; }
        const [a, axis, by] = cut;
        // An ellipsis says the words go on; its whole words are then the element's title.
        if (axis === 'x' && getComputedStyle(a).textOverflow === 'ellipsis') {
          const titled = x.el.closest('[title]');
          if (titled && clip(titled.getAttribute('title') || '', 400).includes(clip(x.t, 200))) { continue; }
        }
        cutAt.add(x.node);
        out.clipped.push(`text "${clip(x.t, 30)}" (${named(x.el)}) cut by ${named(a)} by ${Math.round(by)} px `
          + `${axis === 'x' ? 'sideways' : 'at its foot'}`);
      }
    }
    // wrapEnd
    const flexItems = (box) => {
      const found = [];
      for (const k of box.children) {
        const s = getComputedStyle(k);
        if (s.display === 'contents') { found.push(...flexItems(k)); continue; }
        if (s.display === 'none' || /^(absolute|fixed)$/.test(s.position) || !drawn(k)) { continue; }
        const q = k.getBoundingClientRect();
        if (q.width >= 1 && q.height >= 1) { found.push([k, q]); }
      }
      return found;
    };
    for (const bar of scope.querySelectorAll('.shep-bar')) {
      if (!visible(bar) || bar.closest(SKIP) || layered(bar)) { continue; }
      const lines = [];
      for (const [k, q] of flexItems(bar)) {
        const line = lines[lines.length - 1];
        if (line && q.top < line.bottom - 1 && q.bottom > line.top + 1) {
          line.items.push([k, q]); line.top = Math.min(line.top, q.top); line.bottom = Math.max(line.bottom, q.bottom);
        } else { lines.push({ items: [[k, q]], top: q.top, bottom: q.bottom }); }
      }
      const s = getComputedStyle(bar), start = bar.getBoundingClientRect().left + bar.clientLeft + parseFloat(s.paddingLeft);
      lines.slice(1).forEach((line, i) => {
        const first = line.items.reduce((a, b) => (b[1].left < a[1].left ? b : a));
        if (first[1].left - start > 1) {
          out.wrapEnd.push(`${named(bar)} line ${i + 2} starts ${Math.round(first[1].left - start)} px in from the bar's `
            + `start edge (${named(first[0])}), the start of its line left empty`);
        }
      });
    }
    // disabled
    // The shared reason forms: a reason line, a relation to the notice, a bar's note under its
    // group (the skill's §2), each drawn with words.
    const reasons = [...scope.querySelectorAll('.shep-reason, [data-notice-rel], .shep-bar-note')]
      .filter((e) => visible(e) && drawn(e) && !e.closest(SKIP) && /[A-Za-z0-9]/.test(e.innerText || ''));
    const outermost = (e, sel) => { let found = null;
      for (let b = e.closest(sel); b; b = b.parentElement && b.parentElement.closest(sel)) { found = b; } return found; };
    // On the button's line, or under it within its own height and across its width.
    const besideOrUnder = (p, r) => (p.top < r.bottom && p.bottom > r.top)
      || (p.top >= r.bottom - 1 && p.top <= r.bottom + r.height && p.left < r.right && p.right > r.left);
    for (const b of scope.querySelectorAll('button, [role=button]')) {
      if (!(b.disabled || b.getAttribute('aria-disabled') === 'true') || b.closest(SKIP) || !visible(b)
          || b.matches('.shep-i, .shep-copy, .shep-link, [role=tab]') || layered(b)) { continue; }
      const r = b.getBoundingClientRect();
      if (r.width < 2 || r.height < 2) { continue; }
      const k = regionOf(b), grp = b.closest('.shep-grp'), bar = outermost(b, '.shep-bar');
      const near = reasons.some((q) => ((grp && grp.contains(q)) || (bar && bar.contains(q)))
        || (regionOf(q) === k && besideOrUnder(q.getBoundingClientRect(), r)));
      if (near) { continue; }
      const region = typeof k === 'number' ? (k >= 0 ? panels[k] : null) : k;
      const said = !!region && [...region.querySelectorAll('[data-notice], [data-notice-rel]')].some((n) => visible(n) && drawn(n));
      if (said && !b.matches('.primary')) { continue; }
      // Words a page drew beside it in a form of its own (not the shared reason line, C11).
      const own = texts.find((x) => regionOf(x.el) === k && !b.contains(x.el) && !x.el.closest(INTERACTIVE)
        && /[A-Za-z0-9]/.test(x.t) && x.rects.some((q) => besideOrUnder(q, r)));
      out.disabled.push(`${named(b)} "${say(b)}" disabled with `
        + (own ? `its reason in a page's own form, no shared reason line ("${clip(own.t, 30)}", ${named(own.el)})`
               : 'nothing said in its group, beside or under it')
        + (said ? '; its region\'s notice does not cover a primary action' : ''));
    }
    // token
    const TOKEN = 'code, kbd, samp, .shep-mono, .shep-fpv, .shep-full';
    const tokens = new Set(texts.map((x) => x.el.closest(TOKEN)).filter((t) => t && scope.contains(t)
      && !(t.parentElement && t.parentElement.closest(TOKEN))));
    for (const t of tokens) {
      if (linesOf(t) < 2 || unseen(t) || layered(t)) { continue; }
      const chars = [];
      const w = document.createTreeWalker(t, NodeFilter.SHOW_TEXT);
      for (let n = w.nextNode(); n; n = w.nextNode()) {
        if (!visible(n.parentElement)) { continue; }
        for (let i = 0; i < n.data.length; i++) {
          const rg = document.createRange(); rg.setStart(n, i); rg.setEnd(n, i + 1);
          chars.push({ c: n.data[i], r: [...rg.getClientRects()].find((q) => q.width > 0.1) || null });
        }
      }
      let line = -1, bottom = -Infinity;
      for (const ch of chars) {
        if (!ch.r || /\s/.test(ch.c)) { ch.line = null; continue; }
        if (ch.r.top > bottom - ch.r.height / 2) { line += 1; bottom = ch.r.bottom; } else { bottom = Math.max(bottom, ch.r.bottom); }
        ch.line = line;
      }
      const box = blockOf(t), bs = getComputedStyle(box);
      const room = box.clientWidth - parseFloat(bs.paddingLeft) - parseFloat(bs.paddingRight);
      let word = [];
      const judge = () => {
        const drawnChars = word.filter((ch) => ch.line !== null);
        for (let i = 0; i + 1 < drawnChars.length; i++) {
          if (drawnChars[i].line === drawnChars[i + 1].line) { continue; }
          const text = word.map((ch) => ch.c).join(''), width = drawnChars.reduce((a, ch) => a + ch.r.width, 0);
          if (drawnChars[i].c === '-') {
            out.token.push(`${named(t)} "${clip(text, 40)}" ends a line at a hyphen inside a word (reads hyphenated)`);
          } else if (width <= room - 1) {
            out.token.push(`${named(t)} "${clip(text, 40)}" broken over two lines though it fits its box on a line of `
              + `its own (${Math.round(width)} of ${Math.round(room)} px)`);
          } else { continue; }
          break;
        }
        word = [];
      };
      for (const ch of chars) { if (/\s/.test(ch.c)) { judge(); } else { word.push(ch); } }
      judge();
    }
    // sepEnd
    // The first character drawn after `from` in a text node, and the words from it.
    const firstChar = (node, from) => {
      for (let j = from; j < node.data.length; j++) {
        if (/\s/.test(node.data[j])) { continue; }
        const q = document.createRange(); q.setStart(node, j); q.setEnd(node, j + 1);
        const r = [...q.getClientRects()].find((x) => x.width > 0.1);
        if (r) { return [r, node.data.slice(j)]; }
      }
      return null;
    };
    // ... and after the text `k`'s character `at`, in its own node or the next ones of its block.
    const firstAfter = (k, at, block) => {
      const here = firstChar(texts[k].node, at + 1);
      if (here) { return here; }
      for (let i = k + 1; i < texts.length && block.contains(texts[i].node); i++) {
        const found = firstChar(texts[i].node, 0);
        if (found) { return found; }
      }
      return null;
    };
    texts.forEach((x, k) => {
      for (let at = x.node.data.indexOf('·'); at >= 0; at = x.node.data.indexOf('·', at + 1)) {
        const rg = document.createRange(); rg.setStart(x.node, at); rg.setEnd(x.node, at + 1);
        const dot = [...rg.getClientRects()].find((q) => q.width > 0.1);
        const next = dot && !unseen(x.el) && !layered(x.el) && firstAfter(k, at, blockOf(x.el));
        if (next && next[0].top >= dot.bottom - dot.height / 2) {
          out.sepEnd.push(`${named(blockOf(x.el))} ends a line with " · " before "${clip(next[1], 24)}"`);
        }
      }
    });
    // chain: each fact of a " · " chain (its block's words between two separators) drawn over
    // two lines though it fits the block on a line of its own
    const chains = new Set(texts.filter((x) => x.node.data.includes('·')).map((x) => blockOf(x.el)));
    for (const blk of chains) {
      if (!blk || linesOf(blk) < 2 || layered(blk)) { continue; }
      const bs = getComputedStyle(blk), room = blk.clientWidth - parseFloat(bs.paddingLeft) - parseFloat(bs.paddingRight);
      // A separator not drawn (the shell hides one its line breaks after) still parts two facts.
      const chars = [], byNode = new Map(texts.map((x) => [x.node, x]));
      const wk = document.createTreeWalker(blk, NodeFilter.SHOW_TEXT);
      for (let n = wk.nextNode(); n; n = wk.nextNode()) {
        const x = byNode.get(n);
        if (!x) {
          if (n.data.includes('·') && n.parentElement && n.parentElement.closest('.shep-sep')) { chars.push({ c: '·', r: null }); }
          continue;
        }
        if (unseen(x.el)) { continue; }
        for (let i = 0; i < x.node.data.length; i++) {
          const rg = document.createRange(); rg.setStart(x.node, i); rg.setEnd(x.node, i + 1);
          chars.push({ c: x.node.data[i], r: [...rg.getClientRects()].find((q) => q.width > 0.1) || null });
        }
      }
      let line = -1, bottom = -Infinity, fact = [];
      for (const ch of chars) {
        if (!ch.r || /\s/.test(ch.c)) { continue; }
        if (ch.r.top > bottom - ch.r.height / 2) { line += 1; bottom = ch.r.bottom; } else { bottom = Math.max(bottom, ch.r.bottom); }
        ch.line = line;
      }
      const judge = () => {
        const inked = fact.filter((ch) => ch.r && !/\s/.test(ch.c));
        const width = fact.reduce((a, ch) => a + (ch.r ? ch.r.width : 0), 0);
        if (new Set(inked.map((ch) => ch.line)).size > 1 && width <= room - 1) {
          out.chain.push(`${named(blk)} draws the fact "${clip(fact.map((ch) => ch.c).join(''), 40)}" over two lines `
            + `though it fits on a line of its own (${Math.round(width)} of ${Math.round(room)} px)`);
        }
        fact = [];
      };
      for (const ch of chars) { if (ch.c === '·') { judge(); } else { fact.push(ch); } }
      judge();
    }
    // rowTop
    const fieldsDrawn = [...scope.querySelectorAll('.block')].filter((f) => visible(f) && drawn(f) && !f.closest(SKIP))
      .map((f) => [f, [...f.querySelectorAll('span[data-testid="block-info"]')]
        .find((l) => l.closest('.block') === f && visible(l) && l.getBoundingClientRect().height > 4)])
      .filter(([, l]) => l);
    const layer = (c) => { let e = c; while (e && getComputedStyle(e).display === 'contents') { e = e.parentElement; } return e; };
    for (let i = 0; i < fieldsDrawn.length; i++) {
      for (let j = i + 1; j < fieldsDrawn.length; j++) {
        const [a, la] = fieldsDrawn[i], [b, lb] = fieldsDrawn[j];
        if (panelOf(a) !== panelOf(b) || layered(la) || layered(lb)) { continue; }
        const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
        if (Math.min(ra.bottom, rb.bottom) - Math.max(ra.top, rb.top) < 0.5 * Math.min(ra.height, rb.height)
            || !(ra.right <= rb.left + 1 || rb.right <= ra.left + 1)) { continue; }
        let c = a.parentElement;
        while (c && !c.contains(b)) { c = c.parentElement; }
        const cs = c && getComputedStyle(layer(c));
        if (!cs || !((cs.display.includes('flex') && cs.flexDirection.startsWith('row')) || cs.display.includes('grid'))) { continue; }
        const branch = (e) => { let x = e; while (x.parentElement !== c) { x = x.parentElement; } return x; };
        const one = (x) => fieldsDrawn.filter(([f]) => x.contains(f)).length === 1;
        if (!one(branch(a)) || !one(branch(b))) { continue; }
        const pa = la.getBoundingClientRect(), pb = lb.getBoundingClientRect(), dy = Math.abs(pa.top - pb.top);
        if (dy >= 0.25 * Math.min(pa.height, pb.height)) {
          out.rowTop.push(`${named(a)} "${clip(la.textContent, 20)}" and ${named(b)} "${clip(lb.textContent, 20)}": `
            + `labels ${Math.round(dy)} px apart in one row (${named(c)})`);
        }
      }
    }
    // rowHeight
    const inks = (cell) => { const r = document.createRange(); r.selectNodeContents(cell);
      const q = [...r.getClientRects()].filter((x) => x.width > 0.5 && x.height > 0.5);
      return q.length ? Math.max(...q.map((x) => x.bottom)) - Math.min(...q.map((x) => x.top)) : 0; };
    for (const tr of scope.querySelectorAll('tr')) {
      if (!visible(tr) || tr.closest(SKIP) || layered(tr)) { continue; }
      const sc = scroller(tr, 'x');
      if (!sc || !(sc.scrollWidth > sc.clientWidth + 1)) { continue; }
      const box = pad(sc), cells = [...tr.children].filter((c) => visible(c) && drawn(c));
      const seen = cells.filter((c) => { const q = c.getBoundingClientRect(); return q.left < box.right - 1 && q.right > box.left + 1; });
      const away = cells.filter((c) => !seen.includes(c));
      if (!seen.length || !away.length) { continue; }
      const inView = Math.max(...seen.map(inks)), tall = Math.max(...away.map(inks)), h = tr.getBoundingClientRect().height;
      if (inView > 0 && tall > 1.5 * inView && h >= 1.5 * inView) {
        out.rowHeight.push(`${named(tr)} ${Math.round(h)} px tall, set by a cell out of ${named(sc)}'s view `
          + `(${Math.round(tall)} px of words against ${Math.round(inView)} px in view)`);
      }
    }
  };
  /* What one window of the page shows: no sideways page scroll, its scroll
     regions, its text, its fields, its labels, its stage hint, its choices, and
     the layout rules' findings (`layout`). */
  /* **What the display floor decides** (`style.FLOOR_WIDTH` x `FLOOR_HEIGHT`; `layout_problems`'
     ``floor``): the layout pass's marks (the root's side or stack and short, the header's
     classes, every box marked narrow, folded or alone), the app's frame as drawn and how far the
     document scrolls each way. Below the floor they are compared with the floor's own. */
  const MARKS = ['data-shep-narrow', 'data-shep-folded', 'data-shep-alone'];
  const floorFacts = () => {
    const root = document.documentElement, se = document.scrollingElement;
    const app = document.querySelector('gradio-app'), shell = document.getElementById('shep-shell');
    const r = app ? app.getBoundingClientRect() : { width: 0, height: 0 };
    return { layout: root.getAttribute('data-shep-layout'), short: root.hasAttribute('data-shep-short'),
             header: shell ? [...shell.classList].filter((c) => c.startsWith('shep-')).sort() : [],
             marked: [...document.querySelectorAll(MARKS.map((m) => `[${m}]`).join(', '))].filter(drawn)
               .map((e) => `${named(e)} ${MARKS.filter((m) => e.hasAttribute(m)).join(' ')}`).sort(),
             frame: [Math.round(r.width), Math.round(r.height)],
             scroll: [se.scrollWidth - se.clientWidth, se.scrollHeight - se.clientHeight] };
  };
  const check = () => {
    const se = document.scrollingElement;
    return { overX: se.scrollWidth - se.clientWidth, regions: regions(), type: typeScale(), fields: fields(),
             labels: labels(), stage: stageHint(), choices: choices(), layout: layout(), floor: floorFacts() };
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
  /* A repeated fact (the skill's §12, INT-AUDIT): each fingerprint drawn on
     the page shown (the shared digest value's 12 hex digits,
     `renderers.fingerprint_value`) and how many times its digits are in the
     page's visible text, a full 64-digit value or a plain mention included. */
  const repeats = () => {
    const { tab } = shownPage(), root = tab || document.body, text = root.innerText;
    const facts = new Set([...root.querySelectorAll('code.shep-fpv')].filter((e) => drawn(e) && shown(e))
      .map((e) => e.textContent.trim()).filter((t) => /^[0-9a-f]{12}$/.test(t)));
    return Object.fromEntries([...facts].map((f) => [f, text.split(f).length - 1]).filter(([, n]) => n > 0));
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
  /* A block's words as drawn, the one way a walk's fact reader reads words:
     its innerText, trimmed, never the words of a Gradio status tracker that
     is not drawn. While an event that writes a block is in flight, Gradio
     keeps its tracker among the block's children even when the event hides
     its progress (the class `hide`, opacity 0, a 0 px box), and the
     tracker's timer ("0.2s") is in the block's innerText though nothing of
     it is seen (V-24). A tracker Gradio draws (a shown timer, a failed
     event's red "Error") is read like any other words (the review of
     6a14d6b, R3-2: every tracker was skipped, so a check of such a block
     passed on nothing). With an undrawn tracker inside, the words are those
     of the block's other children, each as drawn (one not drawn says
     nothing, as in innerText; one drawn as `display: contents` by its own). */
  const TRACKER = '[data-testid="status-tracker"]';
  const undrawn = (t) => {
    const box = t.getBoundingClientRect();
    return t.classList.contains('hide') || +getComputedStyle(t).opacity === 0
      || box.width < 1 || box.height < 1;
  };
  const words = (el) => {
    if (!el || (el.matches(TRACKER) && undrawn(el))) { return ''; }
    if (![...el.querySelectorAll(TRACKER)].some(undrawn)) { return el.innerText.trim(); }
    return [...el.childNodes].map((n) => {
      if (n.nodeType === Node.TEXT_NODE) { return n.data.trim(); }
      if (n.nodeType !== Node.ELEMENT_NODE) { return ''; }
      return drawn(n) || getComputedStyle(n).display === 'contents' ? words(n) : '';
    }).filter((t) => t).join('\n');
  };
  window.__shepLook = { parse, ratio, behind, shown, typeScale, fields, labels, pseudo, topAt,
                        shownPage, regions, stageHint, choices, check, focusWalk, repeats, tops, textBox,
                        words, layout, floorFacts };
})();
"""


#: The representative windows (D08), all at or above the display floor
#: (`style.FLOOR_WIDTH` x `FLOOR_HEIGHT`, the owner's decision of 2026-11-23;
#: a window below it is drawn as at the floor, the layout walk's ``floor``):
#: the owner's, 1280 wide, short and wide, a laptop's (1366 x 768), a tall
#: one (stacked: no wider than tall), and the floor itself.
UI_SIZES = ((1850, 960), (1280, 900), (1920, 720), (1366, 768), (1440, 1600), (1280, 720))

#: PLAN_TEST_RESULTS §6's layout acceptance windows the representative ones
#: leave out (PTR-LAYOUT), at or above the floor: 1440 x 900, 1707 x 870 (a
#: 2560 x 1440 screen at 150 %), 1920 x 1080, 2560 x 1600 and 3840 x 2160
#: (zoom changes the CSS viewport; a high-DPI screen only the device pixels
#: per CSS pixel: `walk_ui`'s ``device_scale``). A 1080 x 1920 portrait
#: screen and a 1920 x 1080 window at 200 % (960 x 540) are below the floor.
PTR_SIZES = ((1440, 900), (1707, 870), (1920, 1080), (2560, 1600), (3840, 2160))

#: A request D8 refuses as forwarded by a proxy (`walk_ui`'s ``headers``): the
#: walk of a read-only viewer on a service in mode M1, on this machine.
READ_ONLY_HEADERS = {"X-Forwarded-For": "203.0.113.7"}

#: Every page, by the id its tab button carries (``shep-tab-<id>-button`` for
#: level 1, ``shep-page-<id>-button`` under Research).
UI_PAGES = ("clinical", "datasets", "training", "finetune", "test", "models", "settings")

#: The walk (`walk_ui`): one browser, one page, the stops in order (a stop with
#: ``"focus": false`` records no focus walk: a layout measure's own walks). A stop
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
#: (the panels it covers are not its measure) and closes it again after. A
#: stop with ``diagnose`` (``{"hpo": terms, "top_k": n, "rows": n}``) runs a
#: diagnosis on Diagnosis first and waits for its rows to be drawn; the
#: result stays for the stops after it. A checked stop also records
#: ``repeats`` (each fingerprint drawn and how many times the page's text
#: says it, `repeat_problems`); ``tag`` names its screenshot apart. The
#: walk's browser sends `walk_ui`'s ``headers`` with every request
#: (`READ_ONLY_HEADERS`: a read-only viewer), at its ``device_scale``.
UI_WALK_JS = r"""
const { chromium } = require('playwright');
const fs = require('fs');
const QUEUE_JS = __QUEUE_JS__, LOOK_JS = __LOOK_JS__, EXTRA = __EXTRA_JS__, HEADERS = __HEADERS__,
      SCALE = __SCALE__;
(async () => {
  const stops = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: stops[0].size[0], height: stops[0].size[1] },
                                                 deviceScaleFactor: SCALE,
                                                 ...(HEADERS ? { extraHTTPHeaders: HEADERS } : {}) })).newPage();
  await page.addInitScript(QUEUE_JS);
  await page.addInitScript(LOOK_JS);
  // Nothing asked is unanswered over ten frames in a row: a page's opening chain (`.then` after
  // `.then`, each joined a moment after the one before is answered) is idle for an instant
  // between two of its events, and a stop measured there missed what its last event drew (the
  // L7 rules: Fine-tune's reason beside its disabled Start, drawn late in the chain).
  const idle = async () => {
    await page.evaluate(() => { window.__shepQuiet = 0; });
    await page.waitForFunction(() => {
      if (!(window.__shepQueue && window.__shepQueue.idle())) { window.__shepQuiet = 0; return false; }
      window.__shepQuiet += 1;
      return window.__shepQuiet >= 10;
    }, null, { timeout: 60000, polling: 'raf' });
  };
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
    // The opener is clicked where it is drawn once the page has settled at the
    // stop's size: a forced click right after a resize lands where the layout
    // pass is about to move it from (Diagnosis' card folds at 1280 x 600 and
    // its Details › moves), and that click missed its button.
    if (open) { await idle(); await settle(); }
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
  // A stop's diagnosis (`stop.diagnose`): its terms and Top-K entered, Run
  // Diagnosis asked once the page's own requests are answered, and its answer
  // drawn (the results' rows) before the stop measures.
  const diagnose = async (spec) => {
    await page.fill('#dx_hpo_terms textarea', spec.hpo);
    await page.fill('#dx_top_k input', String(spec.top_k));
    await idle();
    await page.waitForSelector('#shep-dx-run-btn:not([disabled])', { timeout: 60000 });
    await page.click('#shep-dx-run-btn', { force: true });
    await page.waitForFunction((n) => document.querySelectorAll('#shep-dx-results-body tbody tr[data-key]').length >= n,
                               spec.rows, { timeout: 120000 });
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
    if (stop.diagnose) { await diagnose(stop.diagnose); }
    if (stop.open) { await drawer(stop.open, true); }
    const seen = { size: `${w}x${h}`, page: stop.page, frames: await settle() };
    if (stop.open) { seen.open = stop.open.shown; }
    // A drawer opened with a row chosen (`stop.open.pick`): the chosen model whose facts it drew,
    // so a walk can show that its row-chosen stops ran with their choice (the sweep's L7, FX-2).
    if (stop.open && stop.open.pick) {
      seen.picked = await page.$eval(stop.open.picked, (el) => el.getAttribute('data-facts-key')
        || el.textContent.trim()).catch(() => '');
    }
    if (EXTRA) { seen.extra = await page.evaluate(EXTRA); }
    if (stop.check !== false) {
      seen.check = await page.evaluate(() => window.__shepLook.check());
      seen.repeats = await page.evaluate(() => window.__shepLook.repeats());
      if (stop.tag) { seen.tag = stop.tag; }
      const name = `${stop.page}${stop.tag ? '_' + stop.tag : ''}${stop.open ? '_' + stop.open.shown.replace(/[^\w-]/g, '') : ''}_${w}x${h}`;
      if (shots) { await page.screenshot({ path: `${shots}/${name}.png` }); }
      if (stop.focus !== false) {
        await page.keyboard.press('Tab');
        seen.focus = await page.evaluate((scope) => window.__shepLook.focusWalk(scope), stop.open ? stop.open.shown : null);
      } else {
        seen.focus = { n: 0, covered: [], cues: [] };
      }
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
            timeout: int = 1500, headers: dict[str, str] | None = None,
            device_scale: float = 1) -> list[dict[str, Any]]:
    """Walk the dashboard at `page_url` through `stops` (each ``{"size":
    [w, h], "page": id}``, with ``"check": false``, ``"options": true`` or
    ``"open": {...}`` where wanted; in order, `UI_WALK_JS`) and return what
    each stop recorded.
    `extra_js` is a function's source the page runs at each stop (its value is
    the stop's ``extra``); `shots`, a directory for a screenshot of each
    checked stop (``<page>_<w>x<h>.png``); `headers`, sent with every
    request the walk's browser makes (`READ_ONLY_HEADERS`); `device_scale`,
    the device pixels per CSS pixel (2: a high-DPI screen)."""
    script = (UI_WALK_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS))
              .replace("__LOOK_JS__", json.dumps(LOOK_JS))
              .replace("__EXTRA_JS__", extra_js or "null")
              .replace("__HEADERS__", json.dumps(headers) if headers else "null")
              .replace("__SCALE__", json.dumps(device_scale)))
    plan = tmp_path / "stops.json"
    plan.write_text(json.dumps(stops), encoding="utf-8")
    args = [page_url, str(plan)] + ([str(shots)] if shots else [])
    return run_node(script, tmp_path, *args, timeout=timeout)


#: **The stage hint each research page shows** (`shell.STAGES`, pinned here
#: in its words): Diagnosis and Settings show none.
STAGE_HINTS = {"datasets": "Stage: Prepare", "training": "Stage: Prepare", "finetune": "Stage: Prepare",
               "test": "Stage: Evaluate", "models": "Stage: Evaluate"}

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
    - on a research page, no stage hint, or one whose words are not its
      page's stage (`STAGE_HINTS`), that runs out of the research pages'
      row, lies over a tab, is cut, sits inside the tablist, is drawn under
      14 px or could be taken for a control; on Diagnosis or Settings, a
      stage hint drawn;
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
        hint, want = check["stage"], STAGE_HINTS.get(stop["page"])
        if want is None and hint is not None:
            found(f"a stage hint drawn off the research pages: {hint}")
        elif want is not None and hint is None:
            found("no stage hint drawn")
        elif hint is not None and (hint["words"] != want or not hint["inRow"] or hint["overTab"]
                                   or hint["cut"] or hint["inTablist"] or hint["size"] < 14
                                   or hint["control"]):
            found(f"stage hint not plain text inside the row, off the tabs, saying {want!r}: {hint}")
        for words in (*check["choices"], *stop.get("options", ())):
            if words.count(" · ") > 1:
                found(f"more than one ' · ' in {words!r}")
    return problems


#: **The layout rules** (the UI sweep of 2026-11-22; `LOOK_JS`'s ``layout``, each
#: relative to what is drawn): the key of each, as `layout_problems` names it.
LAYOUT_RULES = ("beside", "narrow-box", "overlap", "orphan", "centred", "split-group", "edge",
                "clipped", "empty-region", "help-on-page", "notice-repeat", "tab-groups", "in-view",
                "scroll-cue", "wrap-end", "disabled-reason", "token-break", "sep-end", "chain-break",
                "row-top", "row-height", "sideways-scroll", "floor")
_LAYOUT_KEYS = {"beside": "beside", "narrow-box": "narrow", "overlap": "overlap", "orphan": "orphan",
                "centred": "centred", "split-group": "split", "edge": "edge", "clipped": "clipped",
                "empty-region": "empty", "help-on-page": "help", "scroll-cue": "cue",
                "wrap-end": "wrapEnd", "disabled-reason": "disabled", "token-break": "token",
                "sep-end": "sepEnd", "chain-break": "chain", "row-top": "rowTop",
                "row-height": "rowHeight"}
#: The rules held only at or above the display floor (`at_floor`): below it the layout is the
#: floor's and the document scrolls, so what lies past the window is reached by that scroll.
_AT_FLOOR_KEYS = {"in-view": "inview", "sideways-scroll": "sideways"}


def floor_size() -> tuple[int, int]:
    """**The display floor** (`style.FLOOR_WIDTH` x `FLOOR_HEIGHT`, the owner's decision of
    2026-11-23), as a window size."""
    from src.webui.style import FLOOR_HEIGHT, FLOOR_WIDTH

    return FLOOR_WIDTH, FLOOR_HEIGHT


def at_floor(size: str) -> bool:
    """Whether a stop's window (``"<w>x<h>"``) is at or above the display floor in both axes."""
    w, h = (int(n) for n in size.split("x"))
    fw, fh = floor_size()
    return w >= fw and h >= fh


def layout_problems(seen: list[dict[str, Any]], rules: tuple[str, ...] = LAYOUT_RULES,
                    pages: tuple[str, ...] | None = None) -> list[str]:
    """**What the layout rules find in a walk** (`walk_ui`; `LOOK_JS`'s
    ``layout``), one line per finding, ``"<size> <page>: <rule>: <what>"``,
    for the `rules` asked (all, `LAYOUT_RULES`, by default) and the `pages`
    asked (every page walked by default). Not part of `ui_problems` yet: the
    shared walk holds these at zero once every page is laid out again (the
    sweep's integration); a page's lane runs them on its own pages:

    - ``beside``: a text block drawn on two or more lines beside a field or button of its panel
      (or of its bar, outside every panel);
    - ``narrow-box``: a text box or select holding fewer than 10 of its own "0";
    - ``overlap``: two controls overlapping by more than 2 x 2 px (one an open drawer lies over
      aside);
    - ``orphan``: a choice group whose last line holds one option, its first more;
    - ``centred``: a control centred in a row over two and a half times its own height;
    - ``split-group``: the controls of one ``.shep-grp`` on more than one line (a ⓘ, a Copy and a
      "Details ›" link aside);
    - ``edge``: a heading, paragraph, notice, label or field 1 px to 2em off its panel
      heading's left edge;
    - ``clipped``: a control cut by an ancestor that hides overflow and does not scroll (a box of
      ``display: contents`` clips nothing);
    - ``empty-region``: a panel with nothing under its header;
    - ``help-on-page``: a help text drawn outside the bubble;
    - ``notice-repeat``: a notice key drawn more than once on one screen;
    - ``tab-groups``: a separator drawn at a line's start or while the research tabs wrap,
      anything but a tab in the tablist, a focus stop that is not a tab, or the tabs out of
      `RESEARCH_PAGES`' order (a group whose tabs the row's width wraps apart is the decided
      design: the tabs wrap by width alone, no group is broken to a line of its own, the
      sweep's decision 3);
    - ``in-view`` (at every window at or above the display floor, `at_floor`): a control or a
      text line of the page drawn past the window's bottom or right edge that no scroll region in
      view holds;
    - ``scroll-cue``: a scroll region whose drawn content runs past its edge with no cue: below
      it, no R-73 scroll cue (``shep-scroll-cue``'s mask); sideways, no mask and no drawn cue
      just before it (Compare's ``shep-rs-cue``): a table wider than its panel, its cells cut;
    - ``clipped`` also holds a text line cut by an ancestor that hides its overflow in that axis,
      a cut to an ellipsis included unless its whole words are its ``title``;
    - ``wrap-end``: a later line of a bar's groups that does not start on the bar's start edge;
    - ``disabled-reason``: a disabled button with no shared reason (``shep-reason``, a relation,
      its bar's ``shep-bar-note``) in its group or bar, on its line or within its own height under
      it, unless it is no primary action and its region draws the notice or a relation (words a
      page drew there in a form of its own are named: C11 says it one way on every page);
    - ``token-break``: a literal token whose line ends at a hyphen inside a word, or that breaks a
      word fitting its box on a line of its own;
    - ``sep-end``: a separator (" · ") ending a line, its next fact on the next line;
    - ``chain-break``: a fact of a " · " chain drawn over two lines though it fits its block on a
      line of its own (a key split from its value, two chips of one fact apart);
    - ``row-top``: two labelled fields side by side in one row whose labels do not share a top
      line (a quarter of the label's height apart or more);
    - ``row-height``: a table row in a sideways scroll box whose height a cell out of the box's
      view sets (an empty band across the cells in view);
    - ``sideways-scroll`` (at or above the floor): a region whose content runs past its right
      edge, so it scrolls sideways, a mask or not (the shell's automatic sideways cue hid a
      table wider than its frame), unless it says so in words just before it (a drawn
      ``shep-rs-cue``: Compare, a column per result);
    - ``floor`` (the display floor, `floor_size`): at or above it, the document never scrolls
      sideways and, side by side, never at all (the frame never scrolls); below it in either
      axis, the layout pass's marks (side or stack, short, the header's classes, each box marked
      narrow, folded or alone) and the frame are those of the stop at the floor of the same page
      and drawer in the walk, and the document scrolls in each axis the window is under the
      floor (drawn as at the floor, never shrunk)."""
    from src.webui.shell import RESEARCH_PAGES

    problems: list[str] = []
    measured = [s for s in seen if "check" in s and "layout" in s["check"]]
    fw, fh = floor_size()

    def same(stop: dict[str, Any]) -> tuple:
        return stop["page"], stop.get("tag"), stop.get("open"), bool(stop.get("picked"))

    at_the_floor = {same(s): s["check"].get("floor") for s in measured if s["size"] == f"{fw}x{fh}"}
    for stop in measured:
        if pages is not None and stop["page"] not in pages:
            continue
        where = " ".join(p for p in (stop["size"], stop["page"], stop.get("tag"), stop.get("open")) if p)
        found = stop["check"]["layout"]
        for rule in rules:
            if rule in _LAYOUT_KEYS:
                for what in found.get(_LAYOUT_KEYS[rule], ()):
                    problems.append(f"{where}: {rule}: {what}")
            if rule in _AT_FLOOR_KEYS and at_floor(stop["size"]):
                for what in found.get(_AT_FLOOR_KEYS[rule], ()):
                    problems.append(f"{where}: {rule}: {what}")
        facts = stop["check"].get("floor")
        if "floor" in rules and facts is not None:
            w, h = (int(n) for n in stop["size"].split("x"))
            if at_floor(stop["size"]):
                if facts["scroll"][0] > 0:
                    problems.append(f"{where}: floor: the document scrolls sideways by {facts['scroll'][0]} px")
                if facts["layout"] == "side" and facts["scroll"][1] > 0:
                    problems.append(f"{where}: floor: side by side, the document scrolls by "
                                    f"{facts['scroll'][1]} px (the frame scrolls)")
            else:
                ref = at_the_floor.get(same(stop))
                if ref is None:
                    problems.append(f"{where}: floor: below the floor, and no stop at the floor "
                                    f"({fw}x{fh}) of the same page and drawer to compare")
                else:
                    for key in ("layout", "short", "header", "marked"):
                        if facts[key] != ref[key]:
                            problems.append(f"{where}: floor: {key} {facts[key]} where the floor's is {ref[key]}")
                    if any(abs(a - b) > 1 for a, b in zip(facts["frame"], ref["frame"])):
                        problems.append(f"{where}: floor: the frame drawn {facts['frame']} where the floor's "
                                        f"is {ref['frame']}")
                for axis, short, room in ((0, w < fw, "narrower"), (1, h < fh, "shorter")):
                    if short and facts["scroll"][axis] <= 0:
                        problems.append(f"{where}: floor: {room} than the floor and the document does not "
                                        "scroll that way (the layout shrank to the window)")
        if "notice-repeat" in rules:
            for key, times in found.get("notices", {}).items():
                if times > 1:
                    problems.append(f"{where}: notice-repeat: {key} drawn {times} times")
        tabs = found.get("tabs")
        if "tab-groups" in rules and tabs:
            order = [page for line in tabs["lines"] for page in line]
            if order != list(RESEARCH_PAGES):
                problems.append(f"{where}: tab-groups: tabs in the order {order}")
            for sep in tabs["seps"]:
                if len(tabs["lines"]) > 1 or sep["lineStart"]:
                    problems.append(f"{where}: tab-groups: a separator before {sep['tab']} "
                                    f"drawn while the tabs take {len(tabs['lines'])} lines")
            for other in (*tabs["others"], *tabs["focusable"]):
                problems.append(f"{where}: tab-groups: {other} in the tablist")
    return problems


#: **The states a walk can serve** (`state_settings`): ``first_use`` (no
#: workspaces folder, no results registry: every first start), ``empty`` (an
#: empty workspaces folder and an empty registry); a configured world is the
#: caller's own (the shared walk's `ui_walk`).
WALK_STATES = ("first_use", "empty")


def state_settings(state: str, root: Path) -> dict[str, str]:
    """The service's settings for a walk's `state` (`WALK_STATES`), its folders
    made under `root`: the launcher's local start (operator controls on for a
    request on this machine's loopback, `SHEPHERD_LOCAL_ENTRY`), the CPU, and
    for ``empty`` an empty workspaces folder and an empty registry.
    `served_page.served_service` gives the service a home, configuration,
    state and settings file of its own, so nothing of the account is read."""
    settings = {"SHEPHERD_LOCAL_ENTRY": "1", "SHEPHERD_BIND_HOST": "127.0.0.1",
                "SHEPHERD_DEVICE": "cpu"}
    if state == "first_use":
        return settings
    if state == "empty":
        for name in ("workspaces", "registry"):
            (root / name).mkdir(parents=True, exist_ok=True)
        return {**settings, "SHEPHERD_WORKSPACES_ROOT": str(root / "workspaces"),
                "SHEPHERD_REGISTRY": str(root / "registry")}
    raise ValueError(f"no walk state {state!r}")


#: The layout measure's windows (the owner's decision of 2026-11-23): the owner's two windows
#: (1850 x 960, 1550 x 830), a laptop (1366 x 768), the display floor (1280 x 720), a large
#: screen (2560 x 1440: nothing stretched past readability), one window below the floor
#: (1024 x 640: drawn as at the floor, the document scrolls), then the owner's again (back from
#: below the floor).
LAYOUT_SIZES = ((1850, 960), (1550, 830), (1366, 768), (1280, 720), (2560, 1440), (1024, 640),
                (1850, 960))


def layout_stops(pages: tuple[str, ...] = UI_PAGES, sizes: tuple[tuple[int, int], ...] = LAYOUT_SIZES,
                 *, focus: bool = False) -> list[dict[str, Any]]:
    """The stops of a layout measure: each page at each size, checked (the
    layout rules included), with no focus walk unless `focus`."""
    return [{"size": [w, h], "page": page, **({} if focus else {"focus": False})}
            for page in pages for w, h in sizes]


def walk_state(state: str, tmp_path: Path, stops: list[dict[str, Any]], *,
               shots: Path | None = None, extra_js: str | None = None,
               timeout: int = 1500) -> list[dict[str, Any]]:
    """**Serve the dashboard in a walk state and walk it** (`WALK_STATES`,
    `state_settings`; `served_page.served_service`, `walk_ui`): what each stop
    recorded. A page's lane measures its own pages at first use with
    ``layout_problems(walk_state("first_use", tmp, layout_stops(("training",))))``."""
    from tests.fixtures.served_page import served_service

    root = Path(tmp_path)
    settings = state_settings(state, root / "data")
    with served_service(root / "service", settings) as page:
        return walk_ui(page, root, stops, shots=shots, extra_js=extra_js, timeout=timeout)


#: **What the shared model list's tree shows, read where the browser draws it**
#: (D04, `model_list._tree_html`; a page init script): ``window.__shepTree(root)``
#: gives its groups (each its runs and files drawn), its runs (words, whether
#: unfolded) and its files (key, words, selected),
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
        runs: [...body.querySelectorAll('tr.shep-tree-run')].filter(shown).length,
        files: [...body.querySelectorAll('tr[data-key]')].filter(shown).length,
      })),
      runs: [...el.querySelectorAll('tr.shep-tree-run')].map((tr) => ({
        node: tr.getAttribute('data-fold'), words: text(tr.querySelector('td')),
        shown: shown(tr), open: opened(tr.querySelector('button.shep-fold')),
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


def repeat_problems(seen: list[dict[str, Any]], allowed: dict[str, int] | None = None) -> list[str]:
    """**A fact said more than once on one screen** (the WebUI skill's §1.2
    and §12; INT-AUDIT, B27-5), from a walk (`walk_ui`): each fingerprint a
    checked stop drew (``repeats``) said more times than once, or than
    `allowed` gives its page (a page whose second place has a reason the
    caller states), one line each, ``"<size> <page>: fingerprint <hex> said
    <n> times"``; empty when every screen says each once."""
    allowed = allowed or {}
    problems: list[str] = []
    for stop in (s for s in seen if "repeats" in s):
        where = " ".join(p for p in (stop["size"], stop["page"], stop.get("tag"), stop.get("open")) if p)
        for fact, times in stop["repeats"].items():
            if times > allowed.get(stop["page"], 1):
                problems.append(f"{where}: fingerprint {fact} said {times} times")
    return problems


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


__all__ = ["LAYOUT_RULES", "LAYOUT_SIZES", "LOOK_JS", "PTR_SIZES", "QUEUE_JS", "READ_ONLY_HEADERS",
           "STAGE_HINTS", "TREE_JS", "TYPE_SIZES", "TYPE_WEIGHTS", "UI_PAGES", "UI_SIZES", "UI_WALK_JS",
           "WALK_STATES", "at_floor", "floor_size", "layout_problems", "layout_stops", "node_with_playwright", "repeat_problems",
           "run_node", "state_settings", "ui_problems", "walk_state", "walk_ui"]
