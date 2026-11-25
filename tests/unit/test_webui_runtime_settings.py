"""
⚙ Settings as a read-only viewer sees it (the owner's D6 of 2026-11-24)
=======================================================================
D6: the full lock reason is said once, in one place (the header's D8 chip,
which each "Operator controls only" phrase opens); beside each locked action
its short phrase; never words that point at a reason drawn elsewhere ("see
above", "see below", "the reason is with": it may have scrolled away or be in
another region). Settings said it three ways: Data locations' notice
repeated who can act ("Folder paths are shown to operator controls only"),
and Runtime settings drew "Read-only for this request: the reason is with the
actions below." between its facts and its fields.

Module: tests/unit/test_webui_runtime_settings.py
"""
from __future__ import annotations

import re

import pytest

pytest.importorskip("gradio")

from tests.fixtures.operator_access import gradio_request  # noqa: E402

#: Words that point at a reason drawn elsewhere (R-WHY; the walk's ``pointer-words``).
POINTER = re.compile(r"\bsee (above|below)\b|\bthe reason is with\b", re.I)


def _html(outputs) -> str:
    """Every value a handler draws, its HTML as drawn."""
    return " ".join(str(o.get("value", "")) for o in outputs if isinstance(o, dict))


def _text(outputs) -> str:
    """Every value a handler draws, as its words."""
    return re.sub(r"<[^>]+>", " ", _html(outputs))


def test_read_only_settings_say_each_lock_where_it_is_and_point_nowhere(monkeypatch, tmp_path):
    """For a request D8 refuses, what Settings draws (both panels, as the page's open draws
    them) holds no words that point at a reason drawn elsewhere; Runtime settings' one phrase,
    beside Apply, Restart and Stop, also names the two fields the request dims, so each is
    described by it (R-WHY's ``data-for``); Data locations' notice says its consequence (the
    paths are hidden), not who can act, which the chip and the phrases say. Fails on 9adb3c4:
    Runtime's "the reason is with the actions below" line, and the notice's "shown to operator
    controls only"."""
    import src.config.runtime_presets as presets
    import src.webui.components.data_locations as dl
    import src.webui.components.runtime_settings as rs

    monkeypatch.setattr(presets, "RUNTIME_SETTINGS_FILE", tmp_path / "runtime.json")
    monkeypatch.setattr(rs, "is_training_active", lambda: False)
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)
    refused = gradio_request()
    runtime, locations = rs._on_open(refused), dl._on_locations_open(refused)
    for name, outputs in (("runtime", runtime), ("data locations", locations)):
        found = POINTER.search(_text(outputs))
        assert found is None, (name, found and found.group(0))
    phrases = re.findall(r'<button[^>]*class="shep-why"[^>]*data-for="([^"]+)"', _html(runtime))
    assert len(phrases) == 1, phrases
    assert {"runtime_apply", "runtime_restart", "runtime_stop", "allocator_preset",
            "torch_compile"} <= set(phrases[0].split()), phrases
    notice = _text([locations[len(dl.KEYS) * len(dl.ROW_PARTS)]])
    assert "hidden" in notice and "operator controls" not in notice.lower(), notice
