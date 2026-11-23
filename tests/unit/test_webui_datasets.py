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
- **the details**: titled "Selected workspace" (the row beside it names it),
  its origin and record in the run bar's form; the fingerprint as 12 hex with
  Copy (named with the workspace), one fact per row, the manifest's files by
  plain names behind "Manifest contents (7 files) ›", no Path row; how the
  workspace compares with the others ("Same graph build as …, another cut",
  "Same dataset as …"); the manifests are read once per refresh;
- **"no split manifest"** when the file is absent, "no readable split
  manifest" when it is there but not read;
- **long names wrap** at every size, so the "In use" column stays in view;
- **Use for training ›** sets this session's run-bar workspace and opens
  Research › Training; nothing on the server changes; a workspace that is not
  trainable cannot be used;
- **everything escaped** (a workspace may be named with markup characters);
- **the header in plain words**: "Workspaces (N)", never the tab's name again
  and never "read-only" (D8's word for a refused request); its intro defines
  nothing, the fingerprint's ⓘ is on its column and the markers' on In use;
- **one line a row** (R-36): Built (UTC), the fingerprint, Sources ("none
  (synthetic)", never "—"), Train / val on one line, In use with no input-like
  box; in a browser each row is at most 40 px at 1850 x 960;
- **its data versions** (ops round B5): the registry's versions mapped against
  the chosen workspace's graph, read by Test's reader and compared by Test's
  comparison, each by its kind, role and the measurements its data admits
  (the data side of Test's one rule), in the selected row's details; read
  once per refresh; no row carries ``aria-selected``.

Module: tests/unit/test_webui_datasets.py
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

from src.utils.fingerprint import file_sha256  # noqa: E402
from src.webui import style  # noqa: E402
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
    # A plain table's selected row is `aria-current`, never `aria-selected` (skill §8).
    assert 'aria-current="true"' in _row(html, "demo") and "aria-selected" not in html
    assert 'aria-current' not in _row(html, "resampled")
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
    # The fingerprint and sources cells say each missing state in its own
    # words (the WebUI skill §6), never one "—" or "not verified"; one word
    # for the one cause (B0a review: the record cell said "Not read"): no
    # manifest is "Not recorded", a manifest not read "Unknown".
    from src.webui.components import renderers as r

    for name, word in (("scratch_ws", r.NOT_RECORDED), ("broken", r.UNKNOWN)):
        row = _row(html, name)
        assert row.count(r.status_mark("neutral", word)) == 2, name    # fingerprint, sources
        assert r.NO_VALUE not in row.split("shep-ds-use")[0], name
        assert "not verified" not in row and "Not read" not in row, name


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
    """R-35: on ba6e85e the details led with the full 64-hex fingerprint
    under "SHA-256 of split_manifest.json", then a bordered card inside the
    panel of seven digests under snake_case keys, then two rows of facts
    joined by " · ", and a Path row that repeated the name. Now the
    fingerprint is 12 hex with Copy (its whole value in its title and on
    Copy), one fact per row, the manifest's files by plain names behind
    "Manifest contents (7 files) ›" with no inner card, and no Path row."""
    _serving(monkeypatch, None)
    data = dsp.read_datasets()
    html = dsp.details_html(data, "demo")
    fingerprint = file_sha256(root / "demo" / "split_manifest.json")
    assert (f'<dt class="shep-k">Dataset fingerprint</dt><dd class="shep-v"><code class="shep-fpv" '
            f'title="{fingerprint}">{fingerprint[:12]}</code>') in html
    assert f'data-copy="{fingerprint}"' in html and f">{fingerprint}<" not in html
    assert 'aria-label="Copy the full dataset fingerprint of demo"' in html
    assert "shep-card" not in html and " · " not in html
    assert "SHA-256" not in html and "split_manifest" not in html and "node_features" not in html
    manifest = re.search(r'<details class="shep-more shep-ds-manifest"><summary>Manifest contents '
                         r'\(7 files\) ›</summary>(.*?)</details>', html, re.S)
    assert manifest, html
    for label in ("Graph", "Node features", "Edge index", "Node counts", "Train samples",
                  "Val samples", "Dataset record"):
        assert re.search(rf'<dt class="shep-k">{label}</dt><dd class="shep-v"><code class="shep-fpv" '
                         rf'title="[0-9a-f]{{64}}">[0-9a-f]{{12}}</code></dd>', manifest.group(1)), label
    assert '<dt class="shep-k">Allocation seed</dt><dd class="shep-v">42</dd>' in html
    assert re.search(r'<dt class="shep-k">Diseases \(train / val\)</dt><dd class="shep-v">'
                     r'\d+ / \d+</dd>', html)
    assert '<dt class="shep-k">Feature seed</dt>' in html
    # Beside its row: no Path row (it is the name), nor what the selected row
    # shows (built, sources, samples; the skill's §1.2). Built and Sources,
    # the columns a narrow table drops, are drawn here only while it drops
    # them (`optional_html`, `PAGE_CSS`).
    optional = dsp.optional_html(next(x for x in data["rows"] if x["name"] == "demo"))
    assert optional in html
    rest = html.replace(optional, "")
    for key in ("Path", "Built (UTC)", "Sources", "Samples (train / val)"):
        assert f'<dt class="shep-k">{key}</dt>' not in rest, key
    assert re.findall(r'<dt class="shep-k">([^<]+)</dt>', optional) == ["Built (UTC)", "Sources"]
    assert "workspace root" not in html
    assert "Select a dataset" in dsp.details_html(data, None)


def test_the_details_say_how_a_workspace_compares_with_the_others(root, monkeypatch):
    """"resampled" is the same graph build as "demo" cut with another seed;
    "demo_copy" is the same dataset as "demo" (the mockup's sentence)."""
    _serving(monkeypatch, None)
    data = dsp.read_datasets()
    html = dsp.details_html(data, "demo")
    assert ("Same dataset as <b>&lt;b&gt;odd&amp;name</b>, <b>demo_copy</b>: the same full "
            "fingerprint, so the same recorded build.") in html
    assert ("Same graph build as <b>resampled</b> (graph, tensors and record identical), "
            "another cut: the allocation seed differs, so the samples and the dataset "
            "fingerprints differ.") in html
    assert "Same graph build as" not in dsp.details_html(data, "scratch_ws")


def test_the_manifests_are_read_once_per_refresh_not_on_each_selection(root, monkeypatch):
    import src.kg.artifacts as artifacts

    _serving(monkeypatch, None)
    data = dsp.read_datasets()
    monkeypatch.setattr(artifacts, "read_split_manifest",
                        lambda *a, **k: pytest.fail("a selection read a manifest"))
    assert '<dt class="shep-k">Allocation seed</dt><dd class="shep-v">42</dd>' in (
        dsp.details_html(data, "demo"))


def test_the_first_panel_is_titled_by_its_content_and_defines_nothing(root, monkeypatch):
    """R-30 and R-34: on ba6e85e the first panel was titled "Datasets" right
    under the selected "Datasets" tab, its intro defined the fingerprint by a
    file name ("the SHA-256 of each workspace's split manifest
    (split_manifest.json)"), a three-line footnote under the table explained
    the markers and named current_data_dir and workspace_build.json, and the
    note beside Use for training ended "Start is the gated act". Now the title
    is "Workspaces (5)", the intro one plain line, the fingerprint's ⓘ on its
    column, the markers' ⓘ on In use (no file or field name), and the note
    says what the button does."""
    from src.webui.components import help as h

    _serving(monkeypatch, None)
    data = dsp.read_datasets()
    header = dsp.header_html(data)
    assert '<h2>Workspaces <span class="shep-muted">(5)</span></h2>' in header
    assert "Datasets" not in header and "read-only" not in header
    intro = re.search(r'<p class="shep-intro">(.*?)</p>', header).group(1)
    assert intro == dsp.INTRO and "<" not in intro
    assert "SHA-256" not in intro and "split_manifest" not in intro and "fingerprint" not in intro
    table = dsp.table_html(data, None, "demo")
    head = re.search(r"<thead>(.*?)</thead>", table).group(1)
    assert re.findall(r'data-help="([^"]+)"', head) == ["dataset-fingerprint", "datasets-in-use"]
    assert 'data-help' not in header
    # The footnote is gone; its words are the In use ⓘ's, without a file or
    # field name, within the help budget (the skill's §7).
    assert not hasattr(dsp, "FOOTNOTE")
    text = dsp.PAGE_HELP["datasets-in-use"][1]
    assert h.budget_problems(text) == ()
    for internal in ("current_data_dir", "workspace_build", ".json", "Ref:"):
        assert internal not in text, internal
    assert "gated" not in dsp.USE_NOTE and "Start" not in dsp.USE_NOTE


def test_a_row_is_one_line_and_in_use_is_no_input_box(root, monkeypatch):
    """R-36: on ba6e85e a row was two or three lines (75 px at 1850 x 960):
    "UTC" wrapped under the date, train and val were stacked, the record's
    mark and word split, and "in your Training run bar" was a white box with
    a grey border, the look of a text field. Now the columns are Name, Built
    (UTC), Dataset fingerprint, Sources, Train / val and In use, each cell one
    line; the run-bar marker is the shared OK mark (shape and word, no box);
    "not trainable" is the run bar's grey tag; Origin and Record moved to the
    details' header. The browser test below measures the rows."""
    from src.webui.components import renderers as r

    _serving(monkeypatch, str(root / "demo"))
    data = dsp.read_datasets()
    html = dsp.table_html(data, "resampled", "demo")
    heads = [re.sub(r"<[^>]+>", "", th) for th in re.findall(r"<th[^>]*>(.*?)</th>", html)]
    assert heads == ["Name", "Built (UTC)", "Dataset fingerprint", "Sources", "Train / val", "In use"]
    row = next(x for x in data["rows"] if x["name"] == "resampled")
    built = r.built_date(row["build"])
    assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d", built)
    cells = _row(html, "resampled")
    assert (f'<td class="shep-nowrap shep-ds-date shep-col-opt">{built}</td>' in cells
            and "UTC" not in cells)
    assert '<td class="shep-num shep-nowrap">20 / 5</td>' in cells
    assert r.status_mark("ok", "in your Training run bar") in cells
    for retired in ("shep-mark-outline", "shep-mark-grey", "shep-mark-serve-same", "shep-ds-wide",
                    "shep-ds-narrow"):
        assert retired not in html, retired
    for retired in ("shep-ds-wide", "shep-ds-narrow"):      # nor a rule of the page's own sheet
        assert retired not in dsp.PAGE_CSS, retired
    for retired in ("shep-mark-outline", "shep-mark-grey", "shep-mark-serve-same"):  # nor a shared one
        assert retired not in style.SHELL_CSS, retired
    assert "<div" not in html.split("<tbody>")[1]                     # no second line in a cell
    assert '<span class="shep-sub">same dataset as served</span>' in _row(html, "demo_copy")
    scratch = next(x for x in data["rows"] if x["name"] == "scratch_ws")
    assert r.not_trainable_chip(scratch) in _row(html, "scratch_ws")
    assert r.not_trainable_chip(scratch) in r.dataset_identity_html(scratch)   # the run bar's form


def test_origin_record_and_sources_have_one_form_and_the_details_name_no_row(
        root, monkeypatch):
    """R-53 and B51-h: on ba6e85e "synthetic" was an amber chip in the table
    and muted text in the details header, the record "● Verified" in the
    table and a green "● record verified" chip in the header and the run
    bar, Sources "—" in the table and "none (synthetic)" in the details, and
    the details header repeated the selected row's name, record and origin
    beside it. Now origin and record show once, in the details' header, by
    the run bar's own renderer; Sources is one renderer and "none
    (synthetic)" is never "—"; the header names the row by its relation."""
    from src.webui.components import renderers as r

    _serving(monkeypatch, None)
    data = dsp.read_datasets()
    row = next(x for x in data["rows"] if x["name"] == "demo")
    table, details = dsp.table_html(data, None, "demo"), dsp.details_html(data, "demo")
    marks = r.dataset_marks_html(row)
    assert marks == r.chip("synthetic", "amber") + r.record_chip(row)
    assert marks in r.dataset_identity_html(row)                             # the run bar
    head = re.search(r'<div class="shep-panel-hd">(.*?)</div>', details).group(1)
    assert head == f"<h2>{dsp.DETAILS_TITLE}</h2>{marks}" and "demo" not in head
    assert "synthetic" not in table.replace(r.SYNTHETIC_SOURCES, "")
    for record in ("record verified", "Verified", r.record_chip(row)):
        assert record not in table, record
    for name in ("demo", "resampled", ODD):
        assert (f'<td class="shep-ds-src shep-col-opt">{r.SYNTHETIC_SOURCES}</td>'
                in _row(table, name)), name
    # The row beside it says it; the details only while the table drops the column.
    assert r.SYNTHETIC_SOURCES not in details.replace(dsp.optional_html(row), "")


def test_what_the_run_bar_marker_means_is_said_once(root, monkeypatch):
    """B51-h: on ba6e85e the page said three times what "in your Training run
    bar" means (the marker, a footnote, and the note beside Use for
    training). The note beside the button says it now; the In use ⓘ names
    the button instead."""
    _serving(monkeypatch, None)
    data = dsp.read_datasets()
    shown = (dsp.header_html(data) + dsp.table_html(data, "demo", "demo")
             + dsp.details_html(data, "demo") + dsp.use_note_html(data, "demo"))
    assert shown.count("browser tab") == 1 and dsp.USE_NOTE in shown
    assert "nothing on the server" not in shown.lower()
    table = dsp.table_html(data, "demo", "demo")
    assert table.count("in your Training run bar") == 1                        # the marker
    bubble = dsp.PAGE_HELP["datasets-in-use"][1]
    assert "browser tab" not in bubble and "Use for training ›" in bubble


def test_long_names_wrap_at_every_size_so_in_use_stays_in_view(root, monkeypatch):
    """A long name or source list wraps in its cell at every size, so In use
    stays in view. The sweep's L7 (SW-LK3): an 11.5em floor on a Name that
    could break anywhere held 173 px for "demo" and wrapped In use ("bar"
    alone) and the headers at 1850 x 960. A name now breaks only before its
    own separators (its narrowest is its longest piece, it never reads
    hyphenated), with no floor; a source list still wraps anywhere."""
    from src.webui.components import renderers as r

    css = dsp.PAGE_CSS
    rule = next(line for line in css.splitlines() if "td.shep-ds-src" in line)
    assert "overflow-wrap: anywhere" in rule
    assert not [line for line in css.splitlines() if "td.shep-ds-name" in line and "{" in line]
    _serving(monkeypatch, None)
    table = dsp.table_html(dsp.read_datasets(), None, None)
    assert '<td class="shep-ds-name shep-w600"><span class="shep-brk"><span>demo</span></span></td>' in table
    assert r.name_pieces_html("rare-cohort_2026") == (
        '<span class="shep-brk"><span>rare</span><wbr><span>-cohort</span><wbr>'
        '<span>_2026</span></span>')
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
    data, chosen, header, table, details, use, note = refresh.fn(None, None)
    assert chosen == data["rows"][0]["name"]
    assert '<h2>Workspaces <span class="shep-muted">(5)</span></h2>' in header
    assert "read-only" not in header and "Datasets" not in header
    assert use["interactive"] is True and dsp.USE_NOTE in note
    select = fns["_select"]

    class Event:
        _data = {"key": "scratch_ws"}

    key, table, details, use, note = select.fn(data, None, Event())
    assert key == "scratch_ws" and use["interactive"] is False
    # Its reason once in the details: the line that says the manifest was not
    # read, not again as a "Not trainable" row.
    why = next(x for x in data["rows"] if x["name"] == "scratch_ws")["not_trainable"]
    assert "The manifest was not read" in details and details.count(why) == 1
    # Disabled, with the reason beside it as visible text (M5-3).
    assert "Use for training disabled: scratch_ws is not trainable: no split manifest." in note
    assert dsp.USE_NOTE not in note

    class Forged:
        _data = {"key": "../../etc"}

    assert select.fn(data, None, Forged()) == (gr.update(),) * 5


def test_the_reason_beside_use_for_training_is_escaped_and_names_no_path(root, monkeypatch):
    _serving(monkeypatch, None)
    data = {"rows": [{"name": "<b>x", "trainable": False,
                      "not_trainable": "files <i>its</i> manifest binds are missing",
                      "manifest": "read", "dataset_fingerprint": "a" * 64}]}
    note = dsp.use_note_html(data, "<b>x")
    assert "<b>" not in note and "<i>" not in note and "&lt;b&gt;x" in note
    assert "Select a dataset" in dsp.use_note_html(data, None)


def test_the_build_line_is_one_quiet_paragraph_drawn_only_under_a_listed_table():
    """M5-3: the reason row was a flex row, so the build line's text and code
    chips became four columns at 1536 x 760. The UI sweep of 2026-11-22
    (B-07): the line sat in body text as a reason under an empty list, 80 px
    below the list's own reason, with its own left edge. Now it is one
    paragraph, each command a literal token (never broken inside a word), on
    the panel's content edge, and drawn only under a listed table: an empty
    list says how in its own place (`empty_html`)."""
    import inspect

    from src.webui.components import renderers as r

    line = dsp.build_line_html()
    assert line == f'<p class="shep-ds-build">{r.e(dsp.BUILD_LINE)}</p>{dsp.commands_html()}'
    assert "shep-reason" not in line and line.count('<code class="shep-tok">') == 2
    assert "build_line_html()" in inspect.getsource(dsp.create_datasets_page)
    css = " ".join(dsp.PAGE_CSS.split())
    assert (f"{dsp._PAGE} .shep-ds-main:not(:has(table.shep-dstbl)) #shep-ds-notes "
            "{ display: none !important; }") in css
    assert f"{dsp._PAGE} #shep-ds-notes {{ padding: 0 var(--shep-inset) " in css


def test_the_build_commands_are_rows_with_copy_never_inside_a_sentence():
    """The independent check of the UI sweep (C2, the reviewer's GB10-UI-02:
    the build command between ordinary words): the first-use step and the
    line under a listed table said both commands in one sentence (51 and 21
    characters, and a parenthesis), and at 1550 x 830 the step, at the
    measure, broke "scripts/build_knowledge_graph" from ".py --generate-
    samples". Each sentence now says the act in words, with no token; each
    command is a key-value row of its own (the text brief's §8), its token
    whole and its Copy holding the full command."""
    import html as html_lib

    from src.webui.components import renderers as r

    for sentence in (dsp.BUILD_STEP, dsp.BUILD_LINE):
        assert "shep-tok" not in r.literal_html(sentence), sentence
        assert ";" not in sentence and "(" not in sentence
    rows = dsp.commands_html()
    assert rows.startswith('<dl class="shep-kv') and "shep-ds-cmds" in rows
    keys = re.findall(r'<dt class="shep-k">(.*?)</dt>', rows)
    assert keys == [key for key, _command, _what in dsp.COMMANDS]
    for _key, command, _what in dsp.COMMANDS:
        assert f'data-copy="{html_lib.escape(command, quote=True)}"' in rows
        assert r.token_html(command) in rows


def test_nothing_listed_says_why(monkeypatch):
    monkeypatch.delenv("SHEPHERD_WORKSPACES_ROOT", raising=False)
    monkeypatch.setattr(dsp, "read_status", lambda: (None, "x"))
    data = dsp.read_datasets()
    assert data["rows"] == [] and "No workspaces folder is set" in data["problem"]
    assert "No workspaces folder is set" in dsp.table_html(data, None, None)


def test_at_first_use_the_list_says_its_cause_once_as_numbered_steps(monkeypatch):
    """The UI sweep of 2026-11-22 (B-07, GB10-UI-02, the owner's screenshot
    03): with no workspaces folder the list's place drew the whole server
    sentence (296 characters, the variable inside it) as a reason with its
    own left edge, 16 px off the title's, and the build command 80 px below
    it in body text. Now it is the folder's one notice (`notice_block`,
    from the setting's parts, never from its words) on the panel's content
    edge: its state, then numbered steps, its link to Settings › Data
    locations first, the build second (its commands literal tokens); the
    variable only under Details ›. The details and Use for training say a
    relation, never the notice again; with a folder that lists nothing, the
    list's own words and the same build step."""
    from src.webui.components import renderers as r
    from src.webui.components.workspaces import notice_view
    from src.config.server_settings import WORKSPACES_ROOT_NOTICE

    monkeypatch.delenv("SHEPHERD_WORKSPACES_ROOT", raising=False)
    monkeypatch.setattr(dsp, "read_status", lambda: (None, "x"))
    data = dsp.read_datasets()
    assert data["notice"] == notice_view(WORKSPACES_ROOT_NOTICE)
    table = dsp.table_html(data, None, None)
    assert table == (f'<div class="shep-ds-empty">'
                     f'{r.notice_block(data["notice"], steps=(dsp.BUILD_STEP,))}'
                     f'{dsp.commands_html()}</div>')
    assert table.count('data-notice="workspaces_root"') == 1
    steps = re.search(r'<ol class="shep-nb-steps">(.*?)</ol>', table).group(1)
    first, second = re.findall(r"<li>(.*?)</li>", steps)
    assert 'data-shep-go="settings" data-shep-focus="shep-dl"' in first
    assert second == r.e(dsp.BUILD_STEP) and "shep-tok" not in second
    lead = table.split('<ol class="shep-nb-steps">')[0]
    assert "SHEPHERD_" not in re.sub(r"<[^>]+>", "", lead)                 # only under Details
    # Said once: the details and the button relate to it.
    screen = (dsp.header_html(data) + table + dsp.details_html(data, None)
              + dsp.use_note_html(data, None))
    assert screen.count("data-notice=") == 1 and screen.count("No workspaces folder is set.") == 1
    note = dsp.use_note_html(data, None)
    assert note == r.see_above(dsp.USE_FOR_TRAINING, data["notice"])
    assert 'data-notice-rel="workspaces_root"' in note
    assert dsp.DETAILS_NONE in dsp.details_html(data, None)
    assert dsp.DETAILS_SELECT not in dsp.details_html(data, None)
    # A folder set that lists no workspace: its own words and the build step, no notice.
    empty = {"rows": [], "notice": None, "problem": None}
    said = dsp.table_html(empty, None, None)
    assert dsp.NO_WORKSPACE_YET in said and r.e(dsp.BUILD_STEP) in said
    assert dsp.commands_html() in said
    assert "data-notice" not in said and said.startswith('<div class="shep-ds-empty">')
    assert "No workspace is listed" in dsp.use_note_html(empty, None)
    # A folder set and not usable: its notice alone (the folder comes first, no build step).
    refused = {**data["notice"], "kind": "error", "summary": "The workspaces folder cannot be used."}
    assert "shep-nb-steps" not in dsp.table_html({"rows": [], "notice": refused}, None, None)
    # On the panel's content edge, as the title and the intro are.
    css = " ".join(dsp.PAGE_CSS.split())
    assert f"{dsp._PAGE} .shep-ds-empty {{ padding: var(--shep-space-2) var(--shep-inset) " in css


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


#: A script left as a docstring only: it does nothing and exits 0.
_PLANNED_STUB = "PLANNED — NOT YET IMPLEMENTED"


def test_the_build_line_names_only_a_command_that_builds_a_dataset():
    """M5-3 review ARCH-1: the page named ``scripts/preprocess_data.py``, an
    empty placeholder that prints nothing and exits 0, as a dataset build tool.
    Every script the line names exists and has a command-line entry; the
    command that writes a trainable workspace is the one named."""
    repo = Path(__file__).resolve().parents[2]
    commands = " ".join(command for _key, command, _what in dsp.COMMANDS)
    named = re.findall(r"scripts/[\w./-]+\.py", commands)
    assert "scripts/build_knowledge_graph.py" in named
    assert "--generate-samples" in dsp.BUILD_COMMAND
    for script in named:
        text = (repo / script).read_text(encoding="utf-8")
        assert _PLANNED_STUB not in text, script
        assert "def main(" in text and "argparse" in text, script


def test_the_placeholder_script_is_retired_and_no_shipped_text_sends_an_operator_to_it():
    """K7 (X2): the empty ``scripts/preprocess_data.py`` placeholder is deleted.
    The guard stays: the dashboard, the scripts and both deployment guides never
    tell anyone to run it."""
    repo = Path(__file__).resolve().parents[2]
    assert not (repo / "scripts" / "preprocess_data.py").exists()
    places = [*(repo / "src").rglob("*.py"), *(repo / "scripts").rglob("*.py"),
              repo / "deployment-guide.md", repo / "deployment-guide.en.md"]
    naming = [str(p.relative_to(repo)) for p in places
              if "preprocess_data.py" in p.read_text(encoding="utf-8")]
    assert naming == []


def test_a_spaced_path_outside_the_root_is_replaced_whole(root):
    """SECURITY-4, on the Datasets wording: nothing of the path's tail."""
    out = dsp.redact("cannot read /mnt/clinic share/MRN 0012345/split_manifest.json now", root)
    assert out == f"cannot read {dsp.OUTSIDE_PATH} now"
    assert dsp.redact(f"{root}/my ws/split_manifest.json is missing", root) == (
        "my ws/split_manifest.json is missing")


def test_the_training_popover_and_datasets_show_one_details_body(root, monkeypatch):
    """M5-3 review ARCH-4: Training kept its own copy of the details markup
    (and imported the Datasets page for the manifest's facts); the copies
    already differed (Datasets said the disease counts, Training did not).
    Both now show the one body (`renderers.dataset_details_html`)."""
    from src.webui.components import renderers as r
    from src.webui.components import training_console as tc

    _serving(monkeypatch, None)
    data = dsp.read_datasets()
    row = next(x for x in data["rows"] if x["name"] == "demo")
    body = r.dataset_details_html(row, row["facts"], problem=row["facts_problem"])
    assert "Diseases (train / val)" in body and "Manifest contents (7 files)" in body
    assert body in dsp.details_html(data, "demo")
    state = tc.workspace_state("demo")
    assert state["facts"] == row["facts"]
    popover = r.dataset_details_html(state["row"], state["facts"], popover=True)
    assert popover in tc.details_html(state)
    assert body == r.dataset_details_html(state["row"], state["facts"])
    # The one body, by where it stands: a popover, which stands alone, says
    # what Datasets' row shows beside its details, and lists the manifest's
    # files open (no disclosure inside a popover, the skill's §1.5) in the
    # same list.
    for key in ("Built (UTC)", "Sources", "Samples (train / val)"):
        assert f'<dt class="shep-k">{key}</dt>' in popover and key not in body, key
    assert "<details" not in popover and "<details" in body
    files = re.search(r'<dl class="shep-kv shep-kv-2">.*?</dl>', body, re.S).group(0)
    assert files in popover


def test_no_page_reaches_into_another_page_or_the_renderers_private_names():
    """The page contract (`src/webui/shell.py`): a page reaches nothing of
    another page; shared pieces live in renderers.py and workspaces.py. **The
    one exception, recorded in the contract**: Fine-tune is Training's second
    form (README §1) and uses training_console's public training-form pieces
    (as ``tc``), and no other page (K6 closing review: the guard did not check
    finetune_page at all)."""
    import ast

    from src.webui import shell

    pages = {"datasets_page", "diagnosis_panel", "training_console", "finetune_page",
             "test_page", "models_page", "runtime_settings"}
    allowed = {"finetune_page": {"training_console"}}
    assert "Fine-tune is Training's second form" in " ".join(shell.__doc__.split())
    folder = Path(dsp.__file__).parent
    for page in sorted(pages):
        source = (folder / f"{page}.py").read_text(encoding="utf-8")
        reached = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and node.module:
                names = {node.module.rsplit(".", 1)[-1], *(a.name for a in node.names)}
                reached |= names & (pages - {page})
            if isinstance(node, ast.Attribute) and node.attr.startswith("_") \
                    and isinstance(node.value, ast.Name) and node.value.id in ("r", "tc"):
                raise AssertionError(f"{page} uses {node.value.id}.{node.attr}")
        assert reached <= allowed.get(page, set()), (page, reached)
    assert "from src.webui.components import training_console as tc" in (
        folder / "finetune_page.py").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- data versions
@pytest.fixture
def registry(root, tmp_path, monkeypatch):
    """A registry holding a supplied cohort mapped against "demo"'s graph (and
    so against every listed workspace with the same graph and record) and the
    demo's generated validation split, through the importer's two steps."""
    from src.evaluation.cohort_registry import commit_import, prepare_generated_import
    from tests.fixtures.registered_cohort import graph_source_cases, import_version, write_source

    registry = tmp_path / "registry"
    registry.mkdir()
    import_version(registry, root / "demo",
                   write_source(tmp_path / "cases.json", graph_source_cases(root / "demo")),
                   label="ref-demo", role="demo")
    plan = prepare_generated_import(registry, root / "demo", "gen-val", "val")
    commit_import(plan, registry, plan.preview_id)
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(registry))
    return registry


def test_the_details_list_the_versions_test_reads_by_the_same_comparison(
        root, registry, monkeypatch):
    """The versions under a workspace are the registry's, as Test lists them
    (`server_lists.list_cohort_versions`), bound to its graph by the one
    comparison Test and Diagnosis use (`workspace_binding`, `binding_state`),
    never a second registry or reader."""
    from src.api.services.server_lists import list_cohort_versions
    from src.webui.components.workspaces import (
        binding_state,
        read_workspaces,
        version_row,
        workspace_binding,
    )

    _serving(monkeypatch, None)
    data = dsp.read_datasets()
    listed = read_workspaces()
    versions = [version_row(item) for item in list_cohort_versions(contents=True).items]
    assert [v["label"] for v in data["versions"]] == [v["label"] for v in versions]
    for row in listed["rows"]:
        mine = next(x for x in data["rows"] if x["name"] == row["name"])["versions"]
        if row["manifest"] != "read":
            assert mine["bound"] == [] and mine["problem"], row["name"]
            continue
        binding = workspace_binding(listed["directories"][row["name"]], row)
        assert mine["bound"] == [v["label"] for v in versions
                                 if binding_state(v, binding)[0] == "bound"], row["name"]
    html = dsp.details_html(data, "demo")
    assert "Data versions <span class=\"shep-muted\">(2)</span>" in html
    assert "aria-selected" not in html + dsp.table_html(data, None, "demo")
    # The version's label is kept whole (`renderers.whole_words`): its line breaks
    # only at a space, never at its own hyphen (the L6 check: "demo-" / "test-1").
    supplied = re.search(r'<tr><td[^>]*><span class="shep-nowrap">ref-demo</span></td>.*?</tr>',
                         html, re.S).group(0)
    assert "Supplied cohort version" in supplied and ">demo</code>" in supplied
    generated = re.search(r'<tr><td[^>]*><span class="shep-nowrap">gen-val</span></td>.*?</tr>',
                          html, re.S).group(0)
    assert "Generated split" in generated and ">val</code>" in generated
    # Not mapped against its graph's record: "scratch_ws" has no manifest.
    assert "Not checked against the registry" in dsp.details_html(data, "scratch_ws")


def test_a_row_outside_the_root_is_not_checked_for_its_own_reason(
        root, registry, tmp_path, monkeypatch):
    """B5 review: a listed name that resolves outside the root has no
    directory, and its versions block said Test's words for an empty select,
    "no workspace is chosen", on the row that is chosen. It says the row's
    own reason, without a path."""
    away = tmp_path / "outside-mount" / "linked"
    shutil.copytree(root / "demo", away)
    (root / "linked_out").symlink_to(away, target_is_directory=True)
    _serving(monkeypatch, None)
    data = dsp.read_datasets()
    row = next(x for x in data["rows"] if x["name"] == "linked_out")
    assert row["manifest"] == "not_checked"
    assert row["versions"]["problem"] == row["not_trainable"].rstrip(". ")
    html = dsp.details_html(data, "linked_out")
    assert "no workspace is chosen" not in html
    assert ("Not checked against the registry: linked_out resolves outside the workspaces "
            "root, so it is") in html
    assert str(away) not in html and "outside-mount" not in html


def test_the_versions_not_listed_under_a_workspace_are_counted_by_why(
        root, registry, tmp_path, monkeypatch):
    """The comparison's other answers are counted under the table: a version
    mapped against another graph (another graph export of the same graph),
    one whose kept files changed after its import (one byte of its
    samples.json: UI-12, B72-T3) and one that cannot be read (an interrupted
    import), never listed as bound and never dropped unsaid."""
    from tests.fixtures.registered_cohort import (
        DEFAULT_FEATURE_SEED,
        graph_source_cases,
        import_version,
        write_demo_workspace,
        write_source,
    )

    other = write_demo_workspace(tmp_path / "other-export",
                                 feature_seed=DEFAULT_FEATURE_SEED + 1)
    import_version(registry, other,
                   write_source(tmp_path / "other.json", graph_source_cases(other)),
                   label="ref-other", role="demo")
    import_version(registry, root / "demo",
                   write_source(tmp_path / "changed.json", graph_source_cases(root / "demo")),
                   label="ref-changed", role="demo")
    samples = registry / "versions" / "ref-changed" / "samples.json"
    data = bytearray(samples.read_bytes())
    data[len(data) // 2] ^= 0x01
    samples.write_bytes(bytes(data))
    (registry / "versions" / "broken-import").mkdir()
    _serving(monkeypatch, None)
    data = dsp.read_datasets()
    assert {v["label"] for v in data["versions"]} == {
        "broken-import", "gen-val", "ref-changed", "ref-demo", "ref-other"}
    mine = next(x for x in data["rows"] if x["name"] == "demo")["versions"]
    assert mine == {"bound": ["gen-val", "ref-demo"], "another": 1, "changed": 1,
                    "unknown": 1, "problem": None}
    html = dsp.details_html(data, "demo")
    assert 'Data versions <span class="shep-muted">(2)</span>' in html
    assert ('<p class="shep-sub">Not listed here: 1 version mapped against another graph, '
            "1 whose files changed after the import, 1 that cannot be read or checked "
            "now.</p>") in html
    assert ">ref-other<" not in html and ">broken-import<" not in html
    assert ">ref-changed<" not in html


def test_each_version_says_the_measurements_its_data_admits_by_tests_rule(
        root, registry, monkeypatch):
    """What a version admits is the data side of Test's one rule
    (`protocols.inputs_refusal`): a measurement offered with no reason, and
    one that measures its kind but not its data with that reason, one click
    away; one that measures another kind is not named."""
    from src.evaluation import protocols
    from src.evaluation.caveats import MEASUREMENT_NAMES

    _serving(monkeypatch, None)
    data = dsp.read_datasets()
    for version in data["versions"]:
        admitted, refused = dsp.admitted_measurements(version)
        kind, counts = version["kind"], version["counts"]
        assert admitted == [MEASUREMENT_NAMES[n] for n in protocols.names()
                            if protocols.inputs_refusal(n, kind, counts) is None]
        assert [why for _name, why in refused] == [
            protocols.inputs_refusal(n, kind, counts) for n in protocols.names()
            if protocols.inputs_refusal(n, kind, counts) is not None
            and kind in protocols.PROTOCOLS[n].data_kinds]
        assert "KG link prediction" not in admitted + [name for name, _ in refused]
    html = dsp.details_html(data, "demo")
    supplied = re.search(r'<tr><td[^>]*><span class="shep-nowrap">ref-demo</span></td>.*?</tr>',
                         html, re.S).group(0)
    assert "<td>Disease<details" in supplied
    label = re.search(r"<summary>(.*?)</summary>", supplied).group(1)
    assert re.sub(r"<[^>]+>", "", label) == "Not admitted: Gene, Patients-like-me\u00a0›"
    # A name never breaks at its own hyphen; the chevron never stands alone (SW-LK3).
    assert '<span class="shep-nowrap">Patients-like-me</span>\u00a0›' in label
    assert "lists no causal genes" in supplied


def test_the_registry_is_read_once_per_refresh_and_its_absence_is_said(
        root, registry, monkeypatch):
    from src.api.services import server_lists

    _serving(monkeypatch, None)
    data = dsp.read_datasets()
    monkeypatch.setattr(server_lists, "list_cohort_versions",
                        lambda *a, **k: pytest.fail("a selection read the registry"))
    assert "ref-demo" in dsp.details_html(data, "demo")
    monkeypatch.undo()
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root))
    monkeypatch.delenv("SHEPHERD_REGISTRY", raising=False)
    _serving(monkeypatch, None)
    data = dsp.read_datasets()
    html = dsp.details_html(data, "demo")
    # The registry's notice (`notice_block`), said once, here: the workspaces
    # folder is set, so its link is this screen's one link to Data locations.
    from src.webui.components import renderers as r

    assert data["registry_notice"]["key"] == "registry" and data["notice"] is None
    versions = html.split("Data versions", 1)[1]
    assert r.notice_block(data["registry_notice"]) in versions
    assert 'data-notice="registry"' in versions and "No results registry is set." in versions
    assert "<table" not in versions and "No registry is listed" not in versions
    # With the workspaces folder's notice on the screen too, no second link.
    both = {**data, "notice": {"key": "workspaces_root"}}
    assert "data-shep-go" not in dsp.versions_html(both, data["rows"][0])


def test_a_workspace_no_version_was_mapped_against_says_where_versions_come_from(
        root, tmp_path, monkeypatch):
    registry = tmp_path / "empty-registry"
    registry.mkdir()
    monkeypatch.setenv("SHEPHERD_REGISTRY", str(registry))
    _serving(monkeypatch, None)
    data = dsp.read_datasets()
    html = dsp.details_html(data, "demo")
    assert "No data version is mapped against this workspace’s graph yet." in html
    assert dsp.IMPORT_LINE in html


# --------------------------------------------------------------------------- in a browser
#: Research › Datasets in a browser at the owner's window (1850 x 960), then
#: narrow (960 x 1080): a row chosen and put in the Training run bar ("Use for
#: training ›", then back), each step waited for by what it draws (the row's
#: `aria-current`, the Training tab selected, the marker drawn) and the page's
#: queue idle (`QUEUE_JS`), never by a fixed time.
_DATASETS_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
(async () => {
  const browser = await chromium.launch();
  const out = {};
  for (const [w, h] of [[1850, 960], [960, 1080]]) {
    const page = await (await browser.newContext({ viewport: { width: w, height: h } })).newPage();
    await page.addInitScript(QUEUE_JS);
    const idle = () => page.waitForFunction(() => window.__shepQueue && window.__shepQueue.inflight === 0
      && window.__shepQueue.joins.every((j) => j.event_id !== null), null, { timeout: 60000 });
    await page.goto(process.argv[2], { waitUntil: 'load' });
    await page.waitForSelector('#shep-tab-research-button', { timeout: 60000 });
    await idle();
    await page.click('#shep-tab-research-button');
    await page.click('#shep-page-datasets-button');
    await page.waitForSelector('#shep-ds-table tr[data-key="demo"]', { timeout: 60000 });
    await idle();
    // The name cell: a click on the row's Copy copies, it does not select.
    await page.click('#shep-ds-table tr[data-key="demo"] td:first-child');
    await page.waitForSelector('#shep-ds-table tr[data-key="demo"][aria-current="true"]');
    await idle();
    await page.waitForFunction(() => !document.querySelector('#shep-ds-use').disabled);
    await page.click('#shep-ds-use');
    await page.waitForSelector('#shep-page-training-button[aria-selected="true"]');
    await idle();
    await page.click('#shep-page-datasets-button');
    await page.waitForFunction(() => {
      const cell = document.querySelector('#shep-ds-table tr[data-key="demo"] td.shep-ds-use');
      return !!cell && cell.textContent.includes('in your Training run bar');
    }, null, { timeout: 60000 });
    await idle();
    out[w] = await page.evaluate(() => {
      const drawn = (e) => e.getClientRects().length > 0;
      const text = document.getElementById('shep-page-datasets').innerText;
      const boxes = [...document.querySelectorAll('#shep-ds-table td.shep-ds-use *')].filter((e) => {
        const cs = getComputedStyle(e);
        return drawn(e) && cs.backgroundColor === 'rgb(255, 255, 255)' && parseFloat(cs.borderTopWidth) > 0;
      }).map((e) => e.textContent);
      const head = document.querySelector('#shep-ds-details .shep-panel-hd');
      const sx = document.scrollingElement;
      return {
        rows: [...document.querySelectorAll('#shep-ds-table tr[data-key]')].map((tr) =>
          [tr.getAttribute('data-key'), tr.getBoundingClientRect().height]),
        boxes,
        title: document.querySelector('#shep-ds-header h2').innerText,
        tab: document.querySelector('#shep-page-datasets-button').innerText.trim(),
        head: head.innerText,
        tabMeaning: text.split('browser tab').length - 1,
        hex64: (text.match(/\b[0-9a-f]{64}\b/g) || []).length,
        cards: [...document.querySelectorAll('#shep-ds-details .shep-card')].filter(drawn).length,
        sideways: sx.scrollWidth - sx.clientWidth,
      };
    });
    await page.close();
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_in_a_browser_rows_are_one_line_and_each_fact_is_said_once(root, tmp_path, monkeypatch):
    """R-36, R-30, R-35 and B51-h as the dashboard draws them (Node and
    Playwright, skipped without). On ba6e85e at 1850 x 960 a row was 62 to
    75 px, "in your Training run bar" a white box with a grey border, the
    first panel titled "Datasets" under the "Datasets" tab, the details
    header "demo ● record verified synthetic" beside the selected row, the
    full 64-hex fingerprint in a bordered card's panel, and what the run-bar
    marker means said three times. Now every row is at most 40 px, no In use
    mark is a white bordered box, the title is "Workspaces (5)", the details
    header names no row, no 64 hex and no inner card are drawn, the marker's
    meaning is said once, and the page never scrolls sideways (narrow
    too)."""
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    _serving(monkeypatch, None)
    with served(create_gradio_app(), **mount_options()) as base:
        seen = run_node(_DATASETS_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS)), tmp_path,
                        base[:-len("gradio_api")])
    wide = seen["1850"]
    assert len(wide["rows"]) == 5 and all(h <= 40 for _key, h in wide["rows"]), wide["rows"]
    for width, facts in seen.items():
        assert facts["boxes"] == [], (width, facts["boxes"])
        assert facts["title"] == "Workspaces (5)" and facts["tab"] not in facts["title"], width
        assert facts["head"].startswith(dsp.DETAILS_TITLE) and "demo" not in facts["head"], width
        assert facts["hex64"] == 0 and facts["cards"] == 0, width
        assert facts["tabMeaning"] == 1, width
        assert facts["sideways"] == 0, width


_SWITCH_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
(async () => {
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1850, height: 960 } })).newPage();
  await page.addInitScript(QUEUE_JS);
  const idle = () => page.waitForFunction(() => window.__shepQueue && window.__shepQueue.inflight === 0
    && window.__shepQueue.joins.every((j) => j.event_id !== null), null, { timeout: 60000 });
  await page.goto(process.argv[2], { waitUntil: 'load' });
  await page.waitForSelector('#shep-tab-research-button', { timeout: 60000 });
  await idle();
  await page.click('#shep-tab-research-button');
  await page.click('#shep-page-datasets-button');
  await page.waitForSelector('#shep-ds-table tr[data-key="demo"]', { timeout: 60000 });
  await idle();
  const frames = () => page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
  const facts = (w) => page.evaluate((w) => {
    const drawn = (e) => !!e && e.getClientRects().length > 0;
    const wrap = document.querySelector('#shep-ds-table .shep-tbl-wrap');
    const main = document.querySelector('.shep-ds-main').getBoundingClientRect();
    const det = document.querySelector('.shep-ds-details').getBoundingClientRect();
    return { w, layout: document.documentElement.getAttribute('data-shep-layout'),
             over: wrap.scrollWidth - wrap.clientWidth, narrow: wrap.hasAttribute('data-shep-narrow'),
             optional: [...wrap.querySelectorAll('.shep-col-opt')].filter(drawn).length,
             optkv: drawn(document.querySelector('#shep-ds-details .shep-ds-optkv')),
             beside: Math.abs(main.top - det.top) < 1 && det.left >= main.right };
  }, w);
  const out = { sweep: [] };
  for (let w = 1000; w <= 1700; w += 20) {
    await page.setViewportSize({ width: w, height: 960 });
    // The shell lays the page out in the frame after a resize (`update`): two frames on, it has.
    await frames();
    out.sweep.push(await facts(w));
  }
  // At the narrowest window that sets the page side by side, a row still opens its details.
  const first = out.sweep.find((s) => s.layout === 'side');
  if (first) {
    await page.setViewportSize({ width: first.w, height: 960 });
    await frames();
    await page.click('#shep-ds-table tr[data-key="resampled"] td:first-child');
    await page.waitForSelector('#shep-ds-table tr[data-key="resampled"][aria-current="true"]');
    await idle();
    await frames();
    out.picked = { ...(await facts(first.w)),
      head: await page.evaluate(() => document.querySelector('#shep-ds-details .shep-panel-hd h2').innerText),
      fingerprint: await page.evaluate(() => !!document.querySelector('#shep-ds-details .shep-fpv')) };
  }
  console.log(JSON.stringify(out));
  await browser.close();
})();
"""


def test_in_a_browser_the_page_sits_side_by_side_from_the_narrow_tables_minimum(
        root, tmp_path, monkeypatch):
    """The layout rule puts Datasets side by side when its panels' minimum
    widths fit (`--shep-panel-min`). Before the UI sweep of 2026-11-22 the
    list's minimum was its whole six-column table's (912 px, whatever the
    list held: an empty list too), so the page stacked below about 1460 px of
    window (the owner's 1366 and 1280 laptops), and above it the table never
    scrolled sideways only because the minimum reserved all of it. Now the
    list's panel keeps the table with its optional columns dropped (Built,
    Sources; the shell's fit drops them while the whole table does not fit,
    and the selected row's details say them then). Swept across the switch:
    stacked below it and side by side above it, the two panels beside each
    other; side by side the table never scrolls sideways; while it drops its
    optional columns the details say them and while it draws them they do
    not; the page is side by side at windows where the whole table does not
    fit (so from the narrow table's minimum); and there a row still opens
    its details."""
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    _serving(monkeypatch, None)
    with served(create_gradio_app(), **mount_options()) as base:
        seen = run_node(_SWITCH_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS)), tmp_path,
                        base[:-len("gradio_api")])
    sweep = seen["sweep"]
    layouts = [s["layout"] for s in sweep]
    assert layouts[0] == "stack" and layouts[-1] == "side", sweep
    side = [s for s in sweep if s["layout"] == "side"]
    assert all(s["beside"] for s in side), side
    assert [s for s in side if s["over"] > 0] == [], side
    for s in sweep:
        assert (s["optional"] == 0) == s["narrow"] and s["optkv"] == s["narrow"], s
    assert any(s["narrow"] for s in side), side                 # side by side from the narrow table
    assert any(not s["narrow"] for s in side), side             # and whole where it fits
    picked = seen["picked"]
    assert picked["head"].startswith(dsp.DETAILS_TITLE) and picked["fingerprint"], picked
    assert picked["layout"] == "side" and picked["over"] == 0, picked
    assert picked["optkv"] == picked["narrow"], picked


_VERSIONS_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
(async () => {
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1850, height: 960 } })).newPage();
  await page.addInitScript(QUEUE_JS);
  const idle = () => page.waitForFunction(() => window.__shepQueue && window.__shepQueue.inflight === 0
    && window.__shepQueue.joins.every((j) => j.event_id !== null), null, { timeout: 60000 });
  await page.goto(process.argv[2], { waitUntil: 'load' });
  await page.waitForSelector('#shep-tab-research-button', { timeout: 60000 });
  await idle();
  await page.click('#shep-tab-research-button');
  await page.click('#shep-page-datasets-button');
  await page.waitForSelector('#shep-ds-table tr[data-key="demo"]', { timeout: 60000 });
  await idle();
  await page.click('#shep-ds-table tr[data-key="demo"] td:first-child');
  await page.waitForSelector('#shep-ds-details .shep-dsvtbl summary', { timeout: 60000 });
  await idle();
  const frames = () => page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
  const out = [];
  for (const [w, h] of [[1850, 960], [1550, 830], [1366, 768], [1280, 720]]) {
    await page.setViewportSize({ width: w, height: h });
    await frames(); await frames();
    out.push(await page.evaluate((size) => {
      const table = document.querySelector('#shep-ds-details .shep-dsvtbl');
      const wrap = table.closest('.shep-tbl-wrap'), frame = wrap.getBoundingClientRect();
      const label = table.querySelector('summary');
      const lines = [...label.getClientRects()];
      return { size, layout: document.documentElement.getAttribute('data-shep-layout'),
               over: Math.round(table.getBoundingClientRect().right - frame.right),
               labelOver: Math.round(Math.max(...lines.map((q) => q.right)) - frame.right),
               lastLineEnds: label.textContent.trim().endsWith('\u203a') };
    }, `${w}x${h}`));
  }
  console.log(JSON.stringify(out));
  await browser.close();
})();
"""


def test_in_a_browser_the_versions_table_and_its_disclosure_fit_the_details(
        root, registry, tmp_path, monkeypatch):
    """The sweep's L7 (SW-LK3): side by side at 1366 x 768 the Data versions
    table was 12 px wider than its frame and scrolled sideways, its "Not
    admitted: Gene, Patients-like-me ›" cut at the chevron: a disclosure's
    label was never allowed to wrap (``width: max-content``), so a label built
    from data set its column's width. Side by side at the owner's windows and
    two laptops, the table and the label's every line end inside the table's
    frame, and the chevron ends the label."""
    from src.webui.app import create_gradio_app, mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    _serving(monkeypatch, None)
    with served(create_gradio_app(), **mount_options()) as base:
        seen = run_node(_VERSIONS_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS)), tmp_path,
                        base[:-len("gradio_api")])
    side = [s for s in seen if s["layout"] == "side"]
    assert {s["size"] for s in side} >= {"1850x960", "1366x768"}, seen
    for s in side:
        assert s["over"] <= 1 and s["labelOver"] <= 1 and s["lastLineEnds"], s


def test_the_panels_are_sized_by_what_they_hold_and_the_table_drops_optional_columns(
        root, monkeypatch):
    """B-04 and R4 (the UI sweep of 2026-11-22): the two panels were given
    fixed shares (``scale=`` 17:10) and the list a minimum of its whole
    table's width in px, 912 px with nothing listed too. Now each panel
    grows from its own basis (the shared `shep-sized`, in rem), the list's
    minimum is the narrow table's while a table is drawn and only its words'
    measure otherwise (no px), and the table is the shell's fit box with
    Built and Sources its optional columns."""
    from src.webui.style import FIT, FIT_WIDE

    css = " ".join(dsp.PAGE_CSS.split())
    assert "912px" not in css and "495px" not in css and "min-height: 120px" not in css
    mins = dict(re.findall(r"(\.shep-panel\.shep-ds-main(?::has\(table\.shep-dstbl\))?) \{[^}]*"
                           r"--shep-panel-min: ([\d.]+)rem", css))
    empty, table = float(mins[".shep-panel.shep-ds-main"]), float(
        mins[".shep-panel.shep-ds-main:has(table.shep-dstbl)"])
    assert empty < table, mins
    assert re.search(r"\.shep-panel\.shep-ds-details \{[^}]*--shep-panel-basis: [\d.]+rem; "
                     r"--shep-panel-grow: 1; --shep-panel-min: [\d.]+rem", css)
    demo, _ctx, _fns = _page()
    columns = [b for b in demo.blocks.values()
               if {"shep-ds-main", "shep-ds-details"} & set(getattr(b, "elem_classes", None) or [])]
    assert len(columns) == 2 and all("shep-sized" in b.elem_classes for b in columns)
    assert len({b.scale for b in columns}) == 1               # Gradio's default: no share of a window
    _serving(monkeypatch, None)
    html = dsp.table_html(dsp.read_datasets(), None, "demo")
    assert f'<div class="shep-tbl-wrap {FIT}">' in html and f'shep-dstbl {FIT_WIDE}"' in html
    heads = re.findall(r"<th([^>]*)>(.*?)</th>", html)
    optional = [re.sub(r"<[^>]+>", "", text) for attrs, text in heads if "shep-col-opt" in attrs]
    assert optional == ["Built (UTC)", "Sources"]
    for row in re.findall(r"<tr[^>]*data-key=.*?</tr>", html, re.S):
        assert row.count("shep-col-opt") == 2


#: At first use, as drawn: the notice's and the title's left edges, how many
#: notices the page draws, whether the build line is drawn, and the gap
#: between the Use for training relation's mark and its words.
_FIRST_USE_JS = r"""() => {
  const q = (s) => document.querySelector(s);
  const drawn = (e) => !!e && e.getClientRects().length > 0;
  const page = q('#shep-page-datasets'), head = q('#shep-ds-header h2'), nb = q('#shep-ds-table .shep-nb');
  const mark = q('#shep-ds-use-note .shep-nb-mark'), words = q('#shep-ds-use-note .shep-nb-text');
  const left = (e) => e.getBoundingClientRect().left + parseFloat(getComputedStyle(e).paddingLeft);
  // Where each is drawn: the glyphs, not their boxes (a grid column stretches the mark's box).
  const ink = (e) => { const r = document.createRange(); r.selectNodeContents(e);
    return [...r.getClientRects()].filter((x) => x.width > 0); };
  const shape = mark ? ink(mark)[0] : null, text = words ? ink(words)[0] : null;
  return { notices: [...page.querySelectorAll('[data-notice]')].filter(drawn).length,
           edge: nb && head ? Math.round(left(nb) - left(head)) : null,
           build: drawn(q('#shep-ds-notes')),
           markGap: shape && text ? Math.round(text.left - shape.right) : null };
}"""


def test_in_a_browser_at_first_use_the_cause_is_said_once_on_the_panels_edge(tmp_path):
    """B-07 and the owner's screenshot 03 as the dashboard draws them with no
    workspaces folder (the first-use walk's service): at every walked window
    the list's place holds the folder's one notice, starting on the title's
    edge (it started 16 px left of it), the build line is not drawn under an
    empty list, the page's layout rules find nothing, and the relation beside
    Use for training keeps its mark beside its words (it grew to the
    footer's width, its mark 56 px from them)."""
    from tests.fixtures.browser import layout_problems, node_with_playwright, walk_state

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    sizes = ([1850, 960], [1366, 768], [800, 900], [390, 844])
    stops = [{"size": size, "page": "datasets", "focus": False} for size in sizes]
    seen = walk_state("first_use", tmp_path, stops, extra_js=_FIRST_USE_JS)
    assert layout_problems(seen, pages=("datasets",)) == []
    for stop in seen:
        got = stop["extra"]
        assert got["notices"] == 1 and got["edge"] is not None and abs(got["edge"]) <= 1, stop
        assert got["build"] is False, stop
        assert got["markGap"] is not None and got["markGap"] <= 12, stop
