"""
Research › Prepare › Datasets (M5-2, UI-A): the server's workspaces, read-only.
================================================================================
`src/webui/components/datasets_page.py`. What these pin:

- **the server's list, through WP1's one reader** (`list_workspaces`), drawn by
  the shared renderer: 12 hex in the table, the full 64 on Copy, more digits
  when two listed fingerprints share 12;
- **the markers**: "serving reads this workspace" from the status's
  ``current_data_dir`` (by resolved directory, through the server's list),
  "same dataset as served" for another workspace with the same **full**
  fingerprint, "in your Training run bar" from this session's selection, and
  "not trainable: <why>" rows still listed;
- **no absolute path**: a reason that names a server path is cut to the
  workspaces root; the details give the path relative to it;
- **the details**: the full fingerprint with Copy, what the manifest binds
  with short digests, the seeds and split counts;
- **Use for training ›** sets this session's run-bar workspace and opens
  Research › Training; nothing on the server changes; a workspace that is not
  trainable cannot be used;
- **everything escaped** (a workspace may be named with markup characters).

Module: tests/unit/test_webui_datasets.py
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

from src.utils.fingerprint import file_sha256  # noqa: E402
from src.webui.components import datasets_page as dsp  # noqa: E402

ODD = "<b>odd&name"


@pytest.fixture(scope="module")
def built_workspaces(tmp_path_factory):
    from tests.fixtures.registered_cohort import write_demo_workspace

    base = tmp_path_factory.mktemp("dsbuild")
    return (write_demo_workspace(base / "demo"),
            write_demo_workspace(base / "resampled", sample_seed=7))


@pytest.fixture
def root(built_workspaces, tmp_path, monkeypatch):
    demo, resampled = built_workspaces
    root = tmp_path / "root"
    root.mkdir()
    shutil.copytree(demo, root / "demo")
    shutil.copytree(demo, root / "demo_copy")
    shutil.copytree(resampled, root / "resampled")
    shutil.copytree(demo, root / ODD)
    (root / "scratch_ws").mkdir()
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root))
    return root


def _serving(monkeypatch, data_dir):
    monkeypatch.setattr(dsp, "read_status",
                        lambda: ({"initialized": True, "current_data_dir": data_dir}, None))


def _row(html: str, name: str) -> str:
    import html as htmllib

    key = htmllib.escape(name, quote=True)
    match = re.search(rf'<tr[^>]*data-key="{re.escape(key)}".*?</tr>', html, re.S)
    assert match, name
    return match.group(0)


def test_the_list_is_the_servers_with_the_markers(root, monkeypatch):
    _serving(monkeypatch, str(root / "demo"))
    data = dsp.read_datasets()
    assert [r["name"] for r in data["rows"]] == sorted([ODD, "demo", "demo_copy", "resampled",
                                                         "scratch_ws"])
    assert data["serving"] == "demo"
    html = dsp.table_html(data, run_bar="resampled", selected="demo")
    assert "serving reads this workspace" in _row(html, "demo")
    assert "same dataset as served" in _row(html, "demo_copy")
    assert "same dataset as served" in _row(html, ODD)
    assert "in your Training run bar" in _row(html, "resampled")
    assert "same dataset" not in _row(html, "resampled")
    assert "not trainable: no readable split manifest" in _row(html, "scratch_ws")
    assert 'aria-selected="true"' in _row(html, "demo")
    fingerprint = file_sha256(root / "demo" / "split_manifest.json")
    assert f'title="{fingerprint}">{fingerprint[:12]}</code>' in _row(html, "demo")
    assert f'data-copy="{fingerprint}"' in _row(html, "demo")


def test_serving_is_matched_through_the_list_by_resolved_directory(root, monkeypatch):
    (root / "alias").symlink_to(root / "resampled", target_is_directory=True)
    _serving(monkeypatch, str(root / "resampled"))
    data = dsp.read_datasets()
    # A listed link inside the root resolves to the same directory: the first
    # listed name that resolves there is the one marked.
    assert data["serving"] in ("alias", "resampled")
    _serving(monkeypatch, "/somewhere/else")
    assert dsp.read_datasets()["serving"] is None
    monkeypatch.setattr(dsp, "read_status", lambda: (None, "the pipeline status could not be read"))
    data = dsp.read_datasets()
    assert data["serving"] is None and data["status_read"] is False
    assert "serving reads" not in dsp.table_html(data, None, None)


def test_no_absolute_path_is_shown(root, monkeypatch):
    _serving(monkeypatch, str(root / "demo"))
    data = dsp.read_datasets()
    scratch = next(r for r in data["rows"] if r["name"] == "scratch_ws")
    assert str(root) not in (scratch["not_trainable"] or "")
    page = (dsp.header_html(data) + dsp.table_html(data, None, None)
            + "".join(dsp.details_html(data, r["name"]) for r in data["rows"]))
    assert str(root) not in page and str(Path(root).resolve()) not in page
    assert dsp.redact(f"{root}/scratch_ws/split_manifest.json is missing", root) == (
        "scratch_ws/split_manifest.json is missing")
    assert dsp.redact(f"cannot list {root}", root) == "cannot list the workspaces root"


def test_the_details_show_what_the_manifest_binds(root, monkeypatch):
    _serving(monkeypatch, None)
    data = dsp.read_datasets()
    html = dsp.details_html(data, "demo")
    fingerprint = file_sha256(root / "demo" / "split_manifest.json")
    assert fingerprint in html and f'data-copy="{fingerprint}"' in html
    for label in ("kg", "node_features", "edge_indices", "num_nodes", "train_samples",
                  "val_samples", "record"):
        assert f'<span class="shep-k">{label}</span><code>' in html, label
    assert "allocation seed 42" in html and "feature seed" in html
    assert "train 20 · val 5" in html
    assert "<code>demo</code>" in html and "in the server’s workspace root" in html
    assert "Select a dataset" in dsp.details_html(data, None)


def test_two_fingerprints_sharing_twelve_hex_get_more_digits(root, monkeypatch):
    _serving(monkeypatch, None)
    data = dsp.read_datasets()
    first = data["rows"][1]["dataset_fingerprint"]
    data["rows"][2]["dataset_fingerprint"] = first[:12] + "f" * 52
    html = dsp.table_html(data, None, None)
    assert f">{first[:16]}</code>" in html


def test_a_markup_name_is_escaped_everywhere(root, monkeypatch):
    _serving(monkeypatch, str(root / ODD))
    data = dsp.read_datasets()
    page = dsp.table_html(data, ODD, ODD) + dsp.details_html(data, ODD)
    assert "<b>odd" not in page and "&lt;b&gt;odd&amp;name" in page


def _page():
    from src.webui.shell import standalone_context

    with gr.Blocks() as demo:
        ctx = standalone_context(demo)
        dsp.create_datasets_page(ctx)
    fns = {fn.name: fn for fn in demo.fns.values()}
    return demo, ctx, fns


def test_use_for_training_selects_the_workspace_in_this_session_and_opens_training(
        root, monkeypatch):
    _serving(monkeypatch, None)
    _demo, ctx, fns = _page()
    use = fns["_use"]
    assert use.outputs == [ctx.training_workspace, ctx.nav, ctx.research_nav]
    data = dsp.read_datasets()
    name, outer, inner = use.fn(data, "resampled")
    assert name == "resampled"
    assert outer.selected == "research" and inner.selected == "training"
    # a workspace that is not trainable is not put in the run bar
    unchanged, outer, inner = use.fn(data, "scratch_ws")
    assert unchanged == gr.update() and outer == gr.update() and inner == gr.update()


def test_the_page_refreshes_when_opened_and_rows_select_their_details(root, monkeypatch):
    _serving(monkeypatch, str(root / "demo"))
    demo, ctx, fns = _page()
    refresh = fns["_refresh"]
    data, chosen, header, table, details, use = refresh.fn(None, None)
    assert chosen == data["rows"][0]["name"] and "server workspace list · 5 · read-only" in header
    assert use["interactive"] is True
    select = fns["_select"]

    class Event:
        _data = {"key": "scratch_ws"}

    key, table, details, use = select.fn(data, None, Event())
    assert key == "scratch_ws" and use["interactive"] is False
    assert "Not trainable" in details

    class Forged:
        _data = {"key": "../../etc"}

    assert select.fn(data, None, Forged()) == (gr.update(),) * 4


def test_nothing_listed_says_why(monkeypatch):
    monkeypatch.delenv("SHEPHERD_WORKSPACES_ROOT", raising=False)
    monkeypatch.setattr(dsp, "read_status", lambda: (None, "x"))
    data = dsp.read_datasets()
    assert data["rows"] == [] and "SHEPHERD_WORKSPACES_ROOT is not set" in data["problem"]
    assert "SHEPHERD_WORKSPACES_ROOT is not set" in dsp.table_html(data, None, None)


def test_the_training_form_takes_the_workspace_until_the_run_bar_reads_it(root, monkeypatch):
    """Today's Training form (before UI-C's run bar): "Use for training ›" fills
    this session's Data Directory with the listed workspace and opens Data Paths;
    a name the server does not list changes nothing."""
    from src.api.services import training_manager as tm
    from src.webui import shell
    from src.webui.app import PAGES
    from src.webui.components import training_console as tc

    with gr.Blocks() as demo:
        ctx = shell.build_shell(demo, PAGES)
    (bridge,) = [fn for fn in demo.fns.values() if fn.fn is tc._from_datasets]
    assert bridge.inputs == [ctx.training_workspace]
    assert [getattr(b, "elem_id", None) for b in bridge.outputs][0] == "data_dir"
    assert isinstance(bridge.outputs[1], gr.Accordion)

    value, accordion = tc._from_datasets("resampled")
    assert value["value"] == str((root / "resampled").resolve()) and accordion.open is True
    monkeypatch.setattr(tm, "PROJECT_ROOT", root.parent)
    value, _ = tc._from_datasets("resampled")
    assert value["value"] == "SHEPHERD-Advanced/root/resampled"
    assert tc._from_datasets("../etc") == (gr.update(), gr.update())
    assert tc._from_datasets(None) == (gr.update(), gr.update())
