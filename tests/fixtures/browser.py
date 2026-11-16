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


__all__ = ["QUEUE_JS", "node_with_playwright", "run_node"]
