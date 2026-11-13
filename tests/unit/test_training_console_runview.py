"""
Research › Prepare › Training: the run view (the ops-wiring round's B1b).
=========================================================================
`src/webui/components/training_console.py`. What these pin:

- **the run bar keeps its own height** (BACKLOG 52 F): its column's scale is
  0, so Gradio's inline flex-grow never gives it a share of the page's room.

Module: tests/unit/test_training_console_runview.py
"""
from __future__ import annotations

import pytest

gr = pytest.importorskip("gradio")

from src.webui.components import training_console as tc  # noqa: E402
from tests.unit.test_training_page import _page  # noqa: E402


def test_the_run_bar_keeps_its_own_height_and_the_run_view_takes_the_rest():
    """BACKLOG 52 F, found at its root: Gradio writes a column's scale inline
    as its flex-grow, which outranks the page's ``flex: none``; at the
    default 1 the run bar took half the room the run view left and drew it
    blank under its two rows (189 px for 97 px of rows at 1850 x 960, more
    with every section collapsed). At 0 it is its rows' height."""
    _demo, _ctx, by_id = _page()
    assert by_id["shep-train-runbar"].scale == 0
    assert by_id["shep-train-runview"].scale == 22 and by_id["shep-train-params"].scale == 10
    assert "#shep-train-runbar {" in tc.PAGE_CSS and "flex: none;" in tc.PAGE_CSS.split(
        "#shep-train-runbar {", 1)[1].split("}", 1)[0]
