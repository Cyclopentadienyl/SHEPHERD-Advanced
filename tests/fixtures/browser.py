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
``window.__shepQueue``: ``joins``, ``done`` (event ids answered) and
``inflight`` (joined, not yet answered). A test counts the requests one act
sent and waits for what it asked to be answered, never for a fixed time.
`run_node` runs a script and reads the JSON it prints.

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
  ``LABELS``), with its text and style (size, weight, colour, case);
- ``pseudo(el, which)``: a ``::before`` or ``::after`` box from its computed
  position, in page coordinates, with its content and type;
- ``topAt(x, y)``: what ``document.elementsFromPoint`` finds first there;
- ``ratio(a, b)`` and ``behind(el)``: WCAG contrast of two computed colours,
  and the first opaque background behind an element.

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
  const q = window.__shepQueue = {joins: [], done: {}, inflight: 0};
  /* The listener `fnIndex`'s joins, and how many of them were answered. */
  q.count = (fnIndex) => q.joins.filter((j) => j.fn_index === fnIndex).length;
  q.answered = (fnIndex) => q.joins.filter((j) => j.fn_index === fnIndex && j.event_id !== null
                                            && j.event_id in q.done).length;
  const original = window.fetch;
  window.fetch = function (input, init) {
    const url = typeof input === 'string' ? input : (input && input.url) || String(input);
    const sent = original.apply(this, arguments);
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
    const pairs = {}, upper = [];
    const walk = document.createTreeWalker(root || document.body, NodeFilter.SHOW_TEXT);
    for (let n = walk.nextNode(); n; n = walk.nextNode()) {
      const t = n.textContent.trim();
      if (!t || !/[A-Za-z0-9]/.test(t)) { continue; }
      const el = n.parentElement;
      if (!el || !shown(el) || el.closest('.sr-only, .shep-vh, script, style, [hidden]')) { continue; }
      const s = getComputedStyle(el), k = parseFloat(s.fontSize) + '/' + s.fontWeight;
      const p = pairs[k] || (pairs[k] = { n: 0, ex: [] });
      p.n += 1;
      if (p.ex.length < 4 && !p.ex.includes(t.slice(0, 30))) { p.ex.push(t.slice(0, 30)); }
      if (s.textTransform === 'uppercase') { upper.push(t.slice(0, 30)); }
    }
    return { pairs, upper };
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
  const labels = (root) => [...(root || document).querySelectorAll(LABELS)]
    .filter((e) => { const r = e.getBoundingClientRect(); return r.width > 4 && r.height > 4 && shown(e); })
    .map((e) => { const s = getComputedStyle(e);
      return { text: e.textContent.trim(), style: [s.fontSize, s.fontWeight, s.color, s.textTransform] }; });
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
  window.__shepLook = { parse, ratio, behind, shown, typeScale, fields, labels, pseudo, topAt };
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


__all__ = ["LOOK_JS", "QUEUE_JS", "node_with_playwright", "run_node"]
