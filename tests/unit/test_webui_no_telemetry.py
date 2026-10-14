"""
The dashboard sends no telemetry and checks no version.
======================================================
Gradio's default starts a version check against api.gradio.app and sends usage
analytics through huggingface_hub whenever a `gr.Blocks` is built, and the API
builds the dashboard at import (`src/api/main.py`), so every start of the
service tried to reach both. The owner decided on 2026-10-12 that it does not:
the dashboard is built with `analytics_enabled=False`, whatever the environment
says.

This pins Gradio's own switch and the two requests it gates; it does not claim
that nothing else in the process can reach Hugging Face.

Module: tests/unit/test_webui_no_telemetry.py
"""
import pytest

gr = pytest.importorskip("gradio")


def test_the_dashboard_is_built_with_analytics_off_whatever_the_environment(monkeypatch):
    import threading

    import gradio.analytics as analytics

    requests = []
    monkeypatch.setattr(analytics, "version_check", lambda *a, **k: requests.append("version"))
    monkeypatch.setattr(analytics, "_do_normal_analytics_request",
                        lambda *a, **k: requests.append("analytics"))
    # Gradio's own default; an operator's environment usually leaves it unset.
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "True")
    # With analytics off Gradio sets this for the whole process; restored after.
    monkeypatch.delenv("HF_HUB_DISABLE_TELEMETRY", raising=False)
    started = []
    real_start = threading.Thread.start

    def start(self):
        started.append(getattr(self, "_target", None))
        return real_start(self)

    monkeypatch.setattr(threading.Thread, "start", start)

    from src.webui.app import create_gradio_app

    app = create_gradio_app()

    assert app.analytics_enabled is False
    assert analytics.version_check not in started
    assert requests == []
