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
- **no absolute path, for any request**: a path under the root is cut to the
  part inside it, a link's target outside the root and any other absolute path
  read "a path outside the workspaces root", a refused root setting is said
  without its path, and no sentence reads "the workspaces root the workspaces
  root"; the details give the path relative to the root;
- **the details**: the full fingerprint with Copy (named with the workspace),
  what the manifest binds with short digests, the seeds and split counts, and
  how the workspace compares with the others ("Same graph build as …, another
  cut", "Same dataset as …"); the manifests are read once per refresh;
- **"no split manifest"** when the file is absent, "no readable split
  manifest" when it is there but not read;
- **long names wrap** at every size, so the "In use" column stays in view;
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
    assert "not trainable: no split manifest" in _row(html, "scratch_ws")
    assert 'aria-selected="true"' in _row(html, "demo")
    fingerprint = file_sha256(root / "demo" / "split_manifest.json")
    assert f'title="{fingerprint}">{fingerprint[:12]}</code>' in _row(html, "demo")
    assert f'data-copy="{fingerprint}"' in _row(html, "demo")
    assert 'aria-label="Copy the full dataset fingerprint of demo"' in _row(html, "demo")


def test_an_unreadable_manifest_is_told_apart_from_an_absent_one(root, monkeypatch):
    _serving(monkeypatch, None)
    (root / "broken").mkdir()
    (root / "broken" / "split_manifest.json").write_text("{not json")
    data = dsp.read_datasets()
    states = {r["name"]: r["manifest"] for r in data["rows"]}
    assert states["scratch_ws"] == "absent" and states["broken"] == "unreadable"
    assert states["demo"] == "read"
    html = dsp.table_html(data, None, None)
    assert "not trainable: no readable split manifest" in _row(html, "broken")
    assert "not trainable: no split manifest" in _row(html, "scratch_ws")


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
    assert dsp.redact(f"the workspaces root {root} cannot be listed", root) == (
        "the workspaces root cannot be listed")
    assert dsp.redact(f"{root}2/x and /etc/passwd and C:\\data\\ws", root) == (
        f"{dsp.OUTSIDE_PATH} and {dsp.OUTSIDE_PATH} and {dsp.OUTSIDE_PATH}")
    assert dsp.redact("train / val and and/or", root) == "train / val and and/or"


def test_a_link_outside_the_root_shows_no_absolute_path(root, tmp_path, monkeypatch):
    """A workspace that is a link to a directory outside the root: WP1 names
    its target; the page, to any request, does not."""
    away = tmp_path / "outside-mount" / "secret-place" / "bigws"
    away.mkdir(parents=True)
    (root / "big").symlink_to(away, target_is_directory=True)
    _serving(monkeypatch, None)
    data = dsp.read_datasets()
    big = next(r for r in data["rows"] if r["name"] == "big")
    assert big["not_trainable"].startswith("big resolves outside the workspaces root, so it is")
    page = (dsp.table_html(data, None, "big") + dsp.details_html(data, "big"))
    assert str(away) not in page and "secret-place" not in page and str(tmp_path) not in page
    assert "the workspaces root the workspaces root" not in page


def test_a_refused_root_setting_is_said_without_its_path(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(tmp_path / "secret-place" / "missing"))
    monkeypatch.setattr(dsp, "read_status", lambda: (None, "x"))
    data = dsp.read_datasets()
    assert data["rows"] == [] and data["problem"] == dsp.ROOT_REFUSED
    assert "secret-place" not in dsp.table_html(data, None, None)


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
    assert 'aria-label="Copy the full dataset fingerprint of demo"' in html


def test_the_details_say_how_a_workspace_compares_with_the_others(root, monkeypatch):
    """"resampled" is the same graph build as "demo" cut with another seed;
    "demo_copy" is the same dataset as "demo" (the mockup's sentence)."""
    _serving(monkeypatch, None)
    data = dsp.read_datasets()
    html = dsp.details_html(data, "demo")
    assert ("Same dataset as <b>&lt;b&gt;odd&amp;name</b>, <b>demo_copy</b>: the same full "
            "fingerprint, so the same recorded build.") in html
    assert ("Same graph build as <b>resampled</b> (kg, tensors and record identical), another "
            "cut: the allocation seed differs, so the samples and the dataset fingerprints "
            "differ.") in html
    assert "Same graph build as" not in dsp.details_html(data, "scratch_ws")


def test_the_manifests_are_read_once_per_refresh_not_on_each_selection(root, monkeypatch):
    import src.kg.artifacts as artifacts

    _serving(monkeypatch, None)
    data = dsp.read_datasets()
    monkeypatch.setattr(artifacts, "read_split_manifest",
                        lambda *a, **k: pytest.fail("a selection read a manifest"))
    assert "allocation seed 42" in dsp.details_html(data, "demo")


def test_long_names_wrap_at_every_size_so_in_use_stays_in_view():
    css = dsp.PAGE_CSS
    rule = next(line for line in css.splitlines() if "td.shep-ds-name" in line)
    assert "overflow-wrap: anywhere" in rule and "td.shep-ds-src" in rule
    before_media = css.split("@media")[0]
    assert "td.shep-ds-name" in before_media                # not only when narrow
    assert all(line.strip().startswith(("#shep-page-datasets", "@media", "}", ""))
               for line in css.splitlines())


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
    assert use.outputs == [ctx.training_workspace, *ctx.nav_outputs]
    data = dsp.read_datasets()
    name, outer, inner, shown = use.fn(data, "resampled")
    assert name == "resampled" and shown == "training"
    assert outer.selected == "research" and inner.selected == "training"
    # a workspace that is not trainable is not put in the run bar
    unchanged, *nav = use.fn(data, "scratch_ws")
    assert unchanged == gr.update() and nav == [gr.update()] * 3


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


def test_the_model_lists_workspace_select_reads_the_same_list_the_same_way(root, monkeypatch):
    """One reader for every page that shows a workspace: the model list's
    select says "no split manifest" as Datasets does, and no reason holds an
    absolute path there either."""
    from src.webui.components import model_list as ml
    from tests.fixtures.operator_access import operator_request

    monkeypatch.setattr(ml, "read_status", lambda: (None, "x"))
    listing = ml.list_models(None, operator_request(monkeypatch))
    states = {w["name"]: w["manifest"] for w in listing["workspaces"]}
    assert states["scratch_ws"] == "absent" and states["demo"] == "read"
    from src.webui.components.renderers import workspace_choices

    labels = {name: label for label, name in workspace_choices(listing["workspaces"])}
    assert labels["scratch_ws"] == "scratch_ws · no split manifest"
    assert str(root) not in repr(listing["workspaces"])
