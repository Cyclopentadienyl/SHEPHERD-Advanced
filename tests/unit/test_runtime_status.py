"""
Runtime status reporting must survive the failures it exists to report.
=======================================================================
A deployment can lose its GNN and keep serving: the pipeline falls back to path
reasoning, returns a full ranked result, and answers 200. That happened on a real
machine for a whole session. The signals existed — a tick turned into a cross,
one INFO line in the log — but neither was proportionate to the consequence.

Two surfaces were added for that, and both are tested here:

  - `src/utils/version_checker.py` and `GET /system/runtime`, which say which
    torch / CUDA / PyG stack the process is actually on;
  - the diagnosis panel's banner and load-time toast, which say when a ranking
    was produced without the GNN.

The load-bearing test is `test_probe_survives_an_unimportable_torch`. A reporter
that raises when torch is broken reports nothing at exactly the moment it is
needed, which is worse than having no reporter — the page would fail to render
rather than render the bad news.
"""
import importlib

import pytest

from src.utils import version_checker
from src.utils.version_checker import (
    DEGRADED,
    NOTICE,
    OK,
    format_runtime_line,
    probe_runtime,
)

LIBNCCL_ERROR = "libnccl.so.2: cannot open shared object file: No such file or directory"


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    """probe_runtime caches for the life of the process; tests must not inherit it."""
    version_checker._cached = None
    yield
    version_checker._cached = None


def _block_imports(monkeypatch, blocked, error=LIBNCCL_ERROR):
    """Make the named top-level packages fail to import, as a broken install does."""
    real = importlib.import_module

    def fake(name, *args, **kwargs):
        if name.split(".")[0] in blocked:
            raise ImportError(error)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(version_checker.importlib, "import_module", fake)


# ---------------------------------------------------------------------------
# The reporter itself
# ---------------------------------------------------------------------------
def test_probe_reports_a_status_and_never_raises():
    report = probe_runtime(force=True)
    assert report["status"] in (OK, NOTICE, DEGRADED)
    assert isinstance(report["issues"], list)
    for key in ("python", "platform", "torch", "torch_geometric", "pyg_native", "retrieval"):
        assert key in report


def test_probe_survives_an_unimportable_torch(monkeypatch):
    """The case this module exists for, using the error a real machine produced."""
    _block_imports(monkeypatch, {"torch", "torch_geometric"})

    report = probe_runtime(force=True)  # must not raise

    assert report["status"] == DEGRADED
    assert report["torch"]["available"] is False
    assert LIBNCCL_ERROR in report["torch"]["error"]
    assert any("torch is not importable" in i for i in report["issues"])


def test_probe_reports_its_own_failure_rather_than_raising(monkeypatch):
    """Defence in depth for the unforeseen.

    Each individual probe already catches, so this covers the outer guard: if
    anything in the report builder fails in a way not anticipated, the caller
    still receives a report saying so instead of an exception.
    """
    def exploding(*a, **k):
        raise RuntimeError("unforeseen")

    monkeypatch.setattr(version_checker, "_build_report", exploding)

    report = probe_runtime(force=True)  # must not raise

    assert report["status"] == DEGRADED
    assert any("unforeseen" in i for i in report["issues"])


def test_missing_torch_geometric_alone_is_degraded(monkeypatch):
    """torch alone is not enough — GNN scoring needs PyG."""
    pytest.importorskip("torch")
    _block_imports(monkeypatch, {"torch_geometric"}, error="No module named 'torch_geometric'")

    report = probe_runtime(force=True)

    assert report["status"] == DEGRADED
    assert any("torch_geometric" in i for i in report["issues"])


def _fake_probes(monkeypatch, overrides):
    """Replace specific probe targets, leaving the rest real.

    Keyed by the *import target*, not the report key — which is the distinction
    the cuVS probe got wrong: `import cuvs` succeeding says nothing about
    `cuvs.neighbors`, and it is the latter the backend needs.
    """
    real = version_checker._probe_module
    monkeypatch.setattr(
        version_checker, "_probe_module", lambda name: overrides.get(name) or real(name)
    )


_PRESENT = {"available": True, "version": "25.2", "error": None}


def _absent(msg):
    return {"available": False, "version": None, "error": msg}


def test_cuvs_without_cupy_is_a_notice_not_a_failure(monkeypatch):
    """The state observed on a real deployment: importable, unusable, not fatal."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    _fake_probes(monkeypatch, {
        "cuvs": _PRESENT,
        "cuvs.neighbors": _PRESENT,
        "cupy": _absent("No module named 'cupy'"),
    })

    report = probe_runtime(force=True)

    assert report["status"] == NOTICE
    assert any("cupy is missing" in i for i in report["issues"])


def test_cuvs_installed_but_neighbors_broken_is_not_silent(monkeypatch):
    """`import cuvs` succeeding is not evidence the backend can be constructed.

    CuVSIndex needs `from cuvs.neighbors import ivf_flat, ivf_pq`
    (cuvs_backend.py:126). Probing the top-level package and reporting the deeper
    capability is precisely the defect this reporter exists to expose, so it must
    not commit it.
    """
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    _fake_probes(monkeypatch, {
        "cuvs": _PRESENT,
        "cuvs.neighbors": _absent("libcuvs.so: cannot open shared object file"),
        "cupy": _PRESENT,
    })

    report = probe_runtime(force=True)

    assert report["retrieval"]["cuvs"]["available"] is False
    assert report["status"] in (NOTICE, DEGRADED)
    assert any("cuvs.neighbors is not importable" in i for i in report["issues"])
    # And it must not be presented as a working backend.
    line = format_runtime_line(report)
    assert "cuVS" not in line


def test_missing_voyager_is_named_not_omitted(monkeypatch):
    """Voyager is a hard dependency; validate_installation treats its absence as
    an error. It is detached from diagnosis, so this is a notice rather than
    degraded — but silence would break this module's own rule that a missing
    piece is named rather than left out."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    _fake_probes(monkeypatch, {"voyager": _absent("No module named 'voyager'")})

    report = probe_runtime(force=True)

    assert report["status"] == NOTICE, "retrieval is detached — it must not mark diagnosis degraded"
    assert any("Voyager is not importable" in i for i in report["issues"])
    assert "Voyager MISSING" in format_runtime_line(report)


def test_line_names_what_is_missing_rather_than_omitting_it(monkeypatch):
    """An absent entry reads as "not shown"; a named one reads as "not there"."""
    _block_imports(monkeypatch, {"torch", "torch_geometric"})

    line = format_runtime_line(probe_runtime(force=True))

    assert "torch MISSING" in line
    assert "PyG MISSING" in line


# ---------------------------------------------------------------------------
# The API surface
# ---------------------------------------------------------------------------
def test_runtime_endpoint_returns_the_report():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.api.main import app

    response = TestClient(app).get("/api/v1/system/runtime")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in (OK, NOTICE, DEGRADED)
    assert "torch" in body and "pyg_native" in body


# ---------------------------------------------------------------------------
# The WebUI surfaces
# ---------------------------------------------------------------------------
def _status(gnn_ready: bool) -> dict:
    checkpoint = ({"state": "verified", "sha256": "a" * 64} if gnn_ready
                  else {"state": "absent", "sha256": None})
    return {
        "initialized": True,
        "gnn_ready": gnn_ready,
        "sp_ready": gnn_ready,
        "scoring_mode": "gnn_plus_shortest_path" if gnn_ready else "path_reasoning_fallback",
        "kg_nodes": 100,
        "kg_edges": 200,
        "served_identity": {"checkpoint": checkpoint,
                            "graph": {"state": "verified", "kg_sha256": "b" * 64,
                                      "split_manifest_sha256": "c" * 64},
                            "shortest_paths": {"state": "verified" if gnn_ready else "absent"},
                            "dataset_record": {"state": "verified", "origin": "files"}},
    }


def _card(status):
    from datetime import datetime

    from src.webui.components import diagnosis_card as card

    return card.advance(None, status, None, now=datetime(2026, 10, 20), operator=True,
                        d8_short=None, open_checkpoints=False, facts=lambda s, **k: {
                            "key": card.served_key(s), "operator": True, "workspaces": [],
                            "serving": None, "item": None, "item_problem": None,
                            "recorded": None, "trained": None})


def test_the_card_says_path_reasoning_for_as_long_as_it_is_true():
    """No GNN, no trained model: the Diagnosis card says so in red, every time
    it is read (it is not a toast that is seen once), and not otherwise."""
    pytest.importorskip("gradio")
    from src.webui.components import diagnosis_card as card

    without = card.body_html(_card(_status(False)))
    with_gnn = card.body_html(_card(_status(True)))

    assert "No trained model" in without and "path reasoning only" in without
    assert "shep-tag-red" in without
    assert "No trained model" not in with_gnn and "shep-tag-red" not in with_gnn
    # A model reported without its GNN (defensive): red too.
    broken = _status(True)
    broken["gnn_ready"] = False
    assert "GNN not loaded" in card.body_html(_card(broken))


def test_reading_the_card_raises_no_toast(monkeypatch):
    """The card is read on load, on the timer and after each diagnosis; a
    warning toast there would repeat, and a toast that repeats is one people
    learn to dismiss unread."""
    pytest.importorskip("gradio")
    from src.webui.components import diagnosis_card as card

    warnings = []
    monkeypatch.setattr(card, "logger", card.logger)
    import gradio as gr

    monkeypatch.setattr(gr, "Warning", lambda msg, **kw: warnings.append(msg))
    card.body_html(_card(_status(False)))
    card.served_facts_html(_card(_status(False)))

    assert warnings == []


def test_footer_renders_bad_news_instead_of_raising(monkeypatch):
    """The footer reports a broken environment, so it must not break in one."""
    pytest.importorskip("gradio")
    from src.webui import app as webui_app

    def exploding_probe(*a, **k):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(webui_app, "probe_runtime", exploding_probe)

    footer = webui_app._runtime_footer()  # must not raise

    assert "Runtime status unavailable" in footer
    assert "probe exploded" in footer


def test_footer_is_red_when_a_capability_is_gone(monkeypatch):
    pytest.importorskip("gradio")
    from src.webui import app as webui_app

    monkeypatch.setattr(
        webui_app,
        "probe_runtime",
        lambda *a, **k: {"status": DEGRADED, "issues": ["torch is not importable: boom"]},
    )
    monkeypatch.setattr(webui_app, "format_runtime_line", lambda r: "torch MISSING")

    footer = webui_app._runtime_footer()

    assert "#d32f2f" in footer  # red
    assert "Degraded runtime" in footer
    assert "torch is not importable" in footer
