"""
A real headless browser for tests: Node with Playwright, where installed
========================================================================
The WebUI tests that measure what a browser draws (the shared ⓘ's bubble,
the header's layout) run a small Node script that drives Playwright's
Chromium. `node_with_playwright` finds Node and a ``NODE_PATH`` under which it
resolves Playwright (npm's global root, where it is pre-installed); a test
skips without it. Nothing is installed.

Moved here from the help tests, so the shell's tests use the same.

Module: tests/fixtures/browser.py
"""
from __future__ import annotations

import os
import shutil
import subprocess


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


__all__ = ["node_with_playwright"]
