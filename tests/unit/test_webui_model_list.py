"""
The shared model list and the one Load (M5-2, UI-A).
====================================================
`src/webui/components/model_list.py`. What these pin:

- **listing opens checkpoints, so it is an operator control**: D8's helper on
  the triggering request first; refused, nothing is listed or opened, and the
  reason is said; a direct call through Gradio's HTTP API meets the same check;
- **the rows**: "In use" pinned on top by the **full** model fingerprint the
  status serves, marking the checkpoint only; the clinical preset shows models
  finetuned for a task only, the latest 4 runs then "Show older runs"; the
  research preset every run; an unreadable checkpoint greyed; each row's
  recorded tests a state read from the ledger beside the checkpoint and, for
  how each record's group compares (a description), the report store;
- **the presets and the page's view**: the research preset is a table with
  "In use (Diagnosis)" on its own row, not pinned, and its listing carries the
  family view of the same one listing; a page's view lists one filter of the
  one vocabulary (all, base, a task, task models) or its explicit
  rows, draws its groups in its order (the rest after them), fills one extra
  column, and puts a note beside Load; the list redraws when the view changes;
  each filter's empty state and count say it in its own words;
- **the Load**: D8 on the "Load" request; a base or a refused record (a file
  without a model spec among them) refused before anything of the file is read (the research Load off
  with its reason); the item found in the server's list again and its bytes
  hashed again (a list read earlier is caught), one confirmation (from → to:
  the task and scoring mode from the one table, the SP row a state of the
  task, the changed rows marked, what it replaces); then D8 again on "Reload
  with …", **both columns read again**: a change of what is served (any part
  of the served identity or the served workspace, the From column rebuilt
  from the plan's own listing) or of what would be loaded (the file's bytes,
  the workspace's graph; a file rewritten into a base is refused) is said with
  the new values and nothing is reloaded; a workspace SP table never changes a
  task model's To side; WP2's reload with the workspace, the checkpoint the
  server resolves (never a client path) and the task its record names, bound
  to the model fingerprint shown: through the real `run_reload` on a disease
  child trained in the test, bytes that replace the file after the Load hashed
  it are refused on the reload's own read and never served;
  "served model changed" set when any part of what is served changed; a Load
  writes no usage history entry (that writer retired with By test score,
  K8-6d);
- **the wiring**: Load and Reload refresh the chips, Reload sets the "served
  model changed" trigger and the phase and re-reads the list, Keep closes the
  confirmation; the 180 s count runs in the list's own confirmation, so both
  presets can sit in one page;
- **escaped**: a workspace name with markup in the confirmation.

Module: tests/unit/test_webui_model_list.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

from src.config.model_types import WRITTEN_BEFORE_SPEC  # noqa: E402
from src.utils.fingerprint import file_sha256  # noqa: E402
from src.webui.components import model_list as ml  # noqa: E402
from tests.fixtures.operator_access import (  # noqa: E402
    OPERATOR_BASE_URL,
    gradio_request,
    operator_mode,
    operator_request,
)

#: The base every task model here is finetuned from (its family's root run).
BASE_RUN = "20261001-000000_base_aaaaaa"
#: (architecture, run, files, kind) newest run first, as the listing orders
#: them: model bundles filed under ``gatv2`` (a base and its disease, gene and
#: patients-like-me children), a file without a model spec under ``hgt``
#: (an earlier build's directory, listed like any other) and one that cannot
#: be read under ``sage``.
RUNS = [
    ("gatv2", "20261018-091200_disease_8c2e7a", ("model-12.pt", "model-08.pt"), "disease"),
    ("gatv2", "20261017-091500_disease_3fa9c1", ("model-05.pt",), "disease"),
    ("sage", "20261016-083000_run_e4c2a9", ("broken.pt",), None),
    ("gatv2", "20261015-080000_gene_aaaaaa", ("model-03.pt",), "gene"),
    ("gatv2", "20261014-080000_patients_like_me_bbbbbb", ("model-02.pt",), "patients_like_me"),
    ("gatv2", "20261013-080000_disease_cccccc", ("model-01.pt",), "disease"),
    ("hgt", "20261012-080000_run_dddddd", ("model-07.pt",), "specless"),
    ("gatv2", BASE_RUN, ("base.pt",), "base"),
]


def _key(name):
    """A file's listed key (architecture/run/file) by its file name."""
    for arch, run, files, _kind in RUNS:
        if name in files:
            return f"{arch}/{run}/{name}"
    raise KeyError(name)


def payload(kind, epoch, fingerprint, parent=None):
    """A checkpoint's payload as the trainer writes it: a model bundle's record
    (a base, or a child of `parent` for a task; `test_task_first_selection`'s
    writer) or, for ``specless``, a file written before the model spec."""
    meta = {"checkpoint_kind": "weights", "epoch": epoch,
            "training_input_digests": {"split_manifest": fingerprint}}
    if kind == "specless":
        return {**meta, "logs": {"val_mrr": 0.3 + epoch / 100}}
    from tests.unit import test_task_first_selection as bundles

    assert bundles.BASE_RUN == BASE_RUN
    task = None if kind == "base" else kind
    metric = "val_lp_accuracy" if task is None else "val_pooled_mrr"
    return {**bundles._record_payload(task, logs={metric: 0.3 + epoch / 100}, parent=parent),
            **meta}


@pytest.fixture(scope="module")
def demo_workspace(tmp_path_factory):
    from tests.fixtures.registered_cohort import write_demo_workspace

    return write_demo_workspace(tmp_path_factory.mktemp("mlbuild") / "demo")


@pytest.fixture
def root(demo_workspace, tmp_path, monkeypatch):
    """A workspaces root with one workspace and eight runs of checkpoints."""
    from src.utils.checkpoint_io import write_checkpoint

    root = tmp_path / "root"
    workspace = shutil.copytree(demo_workspace, root / "demo")
    fingerprint = file_sha256(workspace / "split_manifest.json")
    now = time.time()
    shas = {}
    base = workspace / "runs" / "gatv2" / BASE_RUN / "checkpoints" / "base.pt"
    base.parent.mkdir(parents=True)
    shas["base.pt"] = write_checkpoint(payload("base", 1, fingerprint), base)
    parent = {"sha256": shas["base.pt"], "run": BASE_RUN}
    for index, (arch, run, files, kind) in enumerate(RUNS):
        directory = workspace / "runs" / arch / run / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        for offset, name in enumerate(files):
            path = directory / name
            if kind is None:
                path.write_bytes(b"not a checkpoint")
            elif kind != "base":
                epoch = int(name.split("-")[1].split(".")[0])
                shas[name] = write_checkpoint(payload(kind, epoch, fingerprint, parent), path)
            stamp = now - index * 3600 - offset * 60
            os.utime(path, (stamp, stamp))
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root))
    monkeypatch.delenv("SHEPHERD_REGISTRY", raising=False)
    return {"root": root, "workspace": workspace, "fingerprint": fingerprint, "shas": shas,
            "parent": parent}


def _status(root, served="model-05.pt", data_dir=None):
    """What the pipeline serves: a disease task model, as a bundle is served
    (`disease_task`, no shortest-path table, no mixing weight)."""
    sha = root["shas"].get(served)
    kg = json.loads((root["workspace"] / "split_manifest.json").read_text())["artifacts"]["kg"]
    return {
        "initialized": True, "current_data_dir": str(data_dir or root["workspace"]),
        "checkpoint_meta": {"epoch": 5},
        "served_identity": {
            "scoring_mode": "disease_task",
            "checkpoint": {"state": "verified" if sha else "absent", "sha256": sha},
            "graph": {"state": "verified", "kg_sha256": kg,
                      "split_manifest_sha256": root["fingerprint"]},
            "shortest_paths": {"state": "absent"},
            "dataset_record": {"state": "verified", "origin": "synthetic", "sha256": "e" * 64},
            "settings": {"device": "cpu", "device_requested": "cpu", "eta": None,
                         "max_path_length": 4, "path_length_penalty": 0.5,
                         "max_paths_per_candidate": 10,
                         "use_direct_path_optimization": True,
                         "phenotype_normalisation_version": 1},
        },
    }


@pytest.fixture
def serving(root, monkeypatch):
    status = {"value": _status(root)}
    monkeypatch.setattr(ml, "read_status", lambda: (status["value"], None))
    return status


# --------------------------------------------------------------------------- listing
def test_a_read_only_request_lists_nothing_and_opens_nothing(root, serving, monkeypatch):
    from src.api.services import server_lists

    monkeypatch.setattr(server_lists, "list_checkpoints",
                        lambda *a, **k: pytest.fail("a refused request listed checkpoints"))
    monkeypatch.delenv("SHEPHERD_DEPLOYMENT_MODE", raising=False)
    listing = ml.list_models("demo", gradio_request())
    assert listing["refused"] is True and listing["items"] == []
    assert listing["problem"].startswith("Refused: ")
    operator_mode(monkeypatch)
    listing = ml.list_models("demo", gradio_request(server=("192.0.2.10", 8264)))
    assert listing["refused"] is True and "did not arrive on this machine" in listing["problem"]
    # A listing is no act (M5-3 review UI-6): the short line every page uses.
    shown = ml.list_html(listing, None, ml.PRESETS["clinical"])
    assert "Operator controls only: request not local." in shown
    assert "Refused" not in shown and "Nothing was done" not in shown


def test_an_operator_lists_the_serving_workspace_with_the_one_in_use_located(
        root, serving, monkeypatch):
    """The drawer (the clinical preset) draws the one tree Models draws (D04:
    the two may differ in their defaults, never in how a model is drawn):
    each run a summary row in the one run words, its files folded under it,
    each by what it is; the model in use marked once, in its own row, its run
    unfolded and the row located (RC-B: it was pinned in an "In use now" group
    and named again in its run, "— in use, pinned above"); the latest four
    runs, then "Show older runs"; models finetuned for a task only."""
    listing = ml.list_models(None, operator_request(monkeypatch))
    assert listing["workspace"] == "demo" and listing["serving"] == "demo"
    assert listing["served_sha"] == root["shas"]["model-05.pt"]
    assert len(listing["items"]) == 9
    html = ml.list_html(listing, None, ml.PRESETS["clinical"])
    assert html.count('<table class="shep-tbl shep-mtbl shep-mtree ') == 1
    assert "In use now" not in html and "pinned above" not in html
    served = _row_of(html, "model-05.pt")
    assert html.count('shep-tag-accent">In use<') == 1 and 'In use<' in served
    assert " hidden" not in served and f'data-locate="{_key("model-05.pt")}"' in html
    assert " hidden" in _row_of(html, "model-12.pt")                     # another run: folded
    # Each run is named once, by its summary row in the one run words (B4,
    # BACKLOG 72's record of B3); its files say what each is.
    from src.webui.components import renderers as r

    for arch, run, files, _kind in RUNS[:2]:
        words = r.run_words(arch, run, fit=False)
        assert words in _visible(_run_row(html, f"{arch}/{run}")), run
        for name in files:
            assert words not in _visible(_row_of(html, name)), name
    assert [_cells(_row_of(html, name))[0] for name in ("model-12.pt", "model-08.pt")] == [
        "model-12.pt", "model-08.pt"]                       # no recorded selection here
    # the clinical preset: models finetuned for a task only (a base offers no
    # diagnosis; refused and unreadable files are never run), the latest four
    # runs, then the rest on request
    for other in ("base_aaaaaa", "run_dddddd", "run_e4c2a9", "metadata not readable"):
        assert other not in html, other
    assert "Show older runs (1)" in html and "disease_cccccc" not in html
    older = ml.list_html(listing, None, ml.PRESETS["clinical"], show_older=True)
    assert "disease_cccccc" in older and "Show older runs" not in older
    research = ml.list_html(listing, None, ml.PRESETS["research"])
    assert "disease_cccccc" in research and 'In use (Diagnosis)<' in research
    # One row group, its own tbody; the table's rows take no `aria-selected`.
    assert html.count("<tbody") == 1 and "aria-selected" not in html + research
    # The drawer's own rule is its scope, not a filter it names: no "· Task
    # models only, of 9 listed" (the 9 would count files it never shows); no
    # order word beside the order it is drawn in, no " · " (R-27, R-77).
    assert ml.count_line(listing, ml.PRESETS["clinical"], False) == (
        "Latest 4 of 5 runs, 5 saved files")


def test_the_drawer_shows_the_run_in_use_beyond_its_latest_runs(root, serving, monkeypatch):
    """BACKLOG 39, D04: a model in use in a run older than the latest four is
    drawn with them, its run unfolded and its row located, and the count line
    says so; a selection in a run left out is said, its details kept, never
    dropped (the rows' pinned "In use now" group is gone with the rows)."""
    serving["value"] = _status(root, served="model-01.pt")          # the oldest task run
    listing = ml.list_models("demo", operator_request(monkeypatch))
    two = ml.Preset("two", 2, "In use", rows="task_models")
    html = ml.list_html(listing, None, two)
    served = _row_of(html, "model-01.pt")
    assert 'In use<' in served and " hidden" not in served
    assert f'data-locate="{_key("model-01.pt")}"' in html
    assert "gene_aaaaaa" not in html and "Show older runs (2)" in html
    assert ml.count_line(listing, two, False) == (
        "Latest 2 of 5 runs and the one in use, 4 saved files")
    # The drawer reads on until the model in use is read (`_latest_checkpoints`):
    # here every task run, so every one is drawn and counted.
    clinical = ml.PRESETS["clinical"]
    listing = ml.list_models("demo", operator_request(monkeypatch),
                             latest_runs=clinical.latest_runs)
    html = ml.list_html(listing, None, clinical)
    assert 'In use<' in _row_of(html, "model-01.pt") and "Show older runs" not in html
    assert ml.count_line(listing, clinical, False) == "5 runs, 6 saved files"
    # A selection in a run the drawer leaves out: said, kept.
    serving["value"] = _status(root)
    listing = ml.list_models("demo", operator_request(monkeypatch))
    one = ml.Preset("one", 1, "In use", rows="task_models")
    picked = _key("model-03.pt")
    html = ml.list_html(listing, picked, one)
    assert f'data-key="{picked}"' not in html
    assert ("Selected model not shown among the latest runs: Gene, gene · GATv2 "
            "2026-10-15 08:00; its details are kept") in _visible(html)
    assert f'data-facts-key="{picked}"' in ml.facts_html(listing, picked)
    assert f'data-key="{picked}"' in ml.list_html(listing, picked, one, show_older=True)


def test_the_drawers_rows_and_facts_name_the_workspace_dataset_by_its_relation(
        root, serving, monkeypatch):
    """R-7+B51a, RC-B, B27-5 (the drawer's part), NEW-diagnosis-3: each row of
    the Load for inference drawer read "model fingerprint … · trained on demo
    · dataset fingerprint 50bef1bf5f9f · synthetic" (three " · "), repeating
    the dataset fingerprint and "synthetic" the workspace line under the
    select says, ten times on one screen. Now a row trained on the chosen
    workspace's current dataset says nothing of it (the dataset said once,
    under the select), every row has at most one " · ", and the chosen
    model's facts name that dataset by its relation ("the selected
    workspace's current dataset"), with no synthetic banner for a file that
    is listed, not served; counted on the drawer's one screen."""
    from src.webui.components import renderers as r

    listing = _listing(root, monkeypatch)
    clinical = ml.PRESETS["clinical"]
    picked = _key("model-12.pt")
    rows = ml.list_html(listing, picked, clinical, show_older=True)
    facts = ml.facts_html(listing, picked)
    line = ml.workspace_line_html(listing)
    workspace = next(w for w in listing["workspaces"] if w["name"] == "demo")
    digits = r.distinguishing_digits(w.get("dataset_fingerprint") for w in listing["workspaces"])
    dataset = root["fingerprint"][:digits]
    assert all(item.get("trained_on_this_dataset") is True for item in ml.shown_items(
        listing, None, clinical))
    seen = _visible(rows)
    assert dataset not in seen and "trained on" not in seen and "synthetic" not in seen
    for row in re.findall(r"<tr[ >].*?</tr>", rows, re.S):
        assert _visible(row).count(" · ") <= 1, _visible(row)
    assert r.CURRENT_DATASET in _visible(facts) and dataset not in _visible(facts)
    assert "SYNTHETIC WORKSPACE" not in facts
    screen = _visible(line + rows + facts)
    assert screen.count(dataset) == 1, screen                     # the workspace line
    if r.origin_of(workspace) == "synthetic":
        assert screen.count("synthetic") == 1, screen


def _diagnosis_page(root, monkeypatch):
    """The Diagnosis page alone (the shell, other pages empty), for an
    operator, the demo workspace's model served: its Load for inference
    drawer, as the browser tests below open it."""
    from src.webui import shell
    from src.webui.components import diagnosis_panel as dp

    operator_mode(monkeypatch)
    status = _status(root)
    monkeypatch.setattr(dp.card, "read_status_within", lambda *a, **k: (status, None))
    monkeypatch.setattr(dp.card, "served_facts", lambda s, **k: {
        "key": dp.card.served_key(s), "operator": True, "workspaces": [], "serving": None,
        "item": None, "item_problem": None, "recorded": None, "trained": None,
        "cohorts": None})
    with gr.Blocks() as demo:
        shell.build_shell(demo, {**{page: (lambda ctx: None)
                                    for page in (*shell.RESEARCH_PAGES, shell.SETTINGS)},
                                 shell.CLINICAL: dp.create_diagnosis_tab})
    return demo


_DRAWER_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
const ROWS = '#shep-models-clinical-rows', FACTS = '#shep-models-clinical-facts';
(async () => {
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1850, height: 960 } })).newPage();
  await page.addInitScript(QUEUE_JS);
  const idle = () => page.waitForFunction(() => window.__shepQueue && window.__shepQueue.idle(), null, { timeout: 30000 });
  const frames = () => page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
  const geometry = () => page.evaluate(([R, F]) => {
    const box = (e) => { const b = e.getBoundingClientRect(); return [Math.round(b.top), Math.round(b.bottom)]; };
    const rows = document.querySelector(R), facts = document.querySelector(F);
    const load = document.querySelector('#shep-models-clinical-load'), lb = load.getBoundingClientRect();
    const top = document.elementsFromPoint(lb.left + lb.width / 2, lb.top + lb.height / 2)[0];
    return { layout: document.documentElement.getAttribute('data-shep-layout'),
             rows: [...rows.querySelectorAll('[data-key], tr.shep-tree-run')].filter((t) => !t.hidden)
               .slice(0, 4).map((t) => Math.round(t.getBoundingClientRect().top)),
             list: box(rows), facts: box(facts), load: box(load), loadOnTop: !!top && load.contains(top),
             factsScroll: facts.scrollTop, factsScrolls: facts.scrollHeight > facts.clientHeight + 1,
             innerH: innerHeight };
  }, [ROWS, FACTS]);
  await page.goto(process.argv[2], { waitUntil: 'load' });
  await page.waitForSelector('#shep-dx-change-btn:not([disabled])', { timeout: 60000 });
  await idle();
  await page.click('#shep-dx-change-btn');
  await page.waitForSelector(`${ROWS} [data-key]`, { state: 'attached', timeout: 30000 });
  await idle();
  const out = {};
  // Side by side: choose the first file shown, then read its details with the wheel.
  const first = await page.$eval(`${ROWS} [data-key]:not([hidden])`, (r) => r.getAttribute('data-key'));
  await page.click(`${ROWS} [data-key="${first}"]`);
  await page.waitForSelector(`${FACTS} .shep-facts-title`, { timeout: 30000 });
  await idle(); await frames();
  out.before = await geometry();
  const at = await page.$eval(FACTS, (e) => { const b = e.getBoundingClientRect();
    return [b.left + b.width / 2, b.top + Math.min(b.height, innerHeight - b.top) / 2]; });
  await page.mouse.move(at[0], at[1]);
  await page.mouse.wheel(0, 400);
  await page.waitForFunction((F) => { const f = document.querySelector(F);
    return f.scrollTop > 0 || f.closest('.shep-model-body').scrollTop > 0; }, FACTS, { timeout: 10000 });
  await frames();
  out.after = await geometry();
  // Side by side at 1280 x 900, the narrowest pair asked for: the tree whole in its region.
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.waitForFunction(() => document.documentElement.getAttribute('data-shep-layout') === 'side');
  await frames();
  const fit = () => page.evaluate(([R, F]) => { const rows = document.querySelector(R), facts = document.querySelector(F);
    return { rows: [rows.clientWidth, rows.scrollWidth], facts: [facts.clientWidth, facts.scrollWidth],
             body: [rows.parentElement.clientWidth, rows.parentElement.scrollWidth] }; }, [ROWS, FACTS]);
  out.fit = await fit();
  // Another host's text metrics, standing in: the tree's words a little wider (WIDER_TREE).
  await page.addStyleTag({ content: process.argv[3] });
  await frames();
  out.wider = await fit();
  out.widerFacts = await page.$eval(FACTS, (f) => f.getBoundingClientRect().width);
  await page.addStyleTag({ content: process.argv[3].replace(/letter-spacing:[^;}]*/, 'letter-spacing: normal') });
  await frames();
  // Measured in the dashboard's own type: its fonts' faces, as the page loaded them.
  out.font = await page.evaluate(() => [...document.fonts]
    .filter((f) => f.family.replace(/["']/g, '') === 'IBM Plex Sans').map((f) => f.status));
  // Stacked: the list, the details, Load, in that order; a file chosen brings its
  // details into view, "Back to the list" the row.
  await page.setViewportSize({ width: 760, height: 1000 });
  await page.waitForFunction(() => document.documentElement.getAttribute('data-shep-layout') === 'stack');
  await frames();
  out.stacked = await geometry();
  await page.click(`${ROWS} tr.shep-tree-run button.shep-fold[aria-expanded="false"]`);
  const second = await page.$eval(`${ROWS} tr[data-key]:not([hidden])`, (r) => r.getAttribute('data-key'));
  await page.click(`${ROWS} tr[data-key="${second}"]`);
  await page.waitForFunction(([F, k]) => { const t = document.querySelector(F + ' [data-facts-key]');
    if (!t || t.getAttribute('data-facts-key') !== k) { return false; }
    const b = t.getBoundingClientRect(); return b.top >= -1 && b.top < innerHeight / 2; },
    [FACTS, second], { timeout: 30000 });
  await idle();
  out.picked = await page.evaluate(() => !!document.activeElement.closest('[data-facts-key]'));
  await page.click(`${FACTS} button.shep-back-list`);
  await page.waitForFunction(([R, k]) => { const b = document.querySelector(`${R} tr[data-key="${k}"]`).getBoundingClientRect();
    return b.top >= 0 && b.bottom <= innerHeight; }, [ROWS, second], { timeout: 10000 });
  out.back = await page.evaluate(([R, k]) => document.activeElement === document.querySelector(`${R} tr[data-key="${k}"]`), [ROWS, second]);
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


#: Another host's text metrics, standing in: the model tree's words a little
#: wider than this host draws them (OPS-FIX-R2; the reviewer's host drew the
#: tree 620 px where this one draws it 599 px).
WIDER_TREE = "#shep-models-clinical-rows table { letter-spacing: 0.04em; }"


def test_the_drawers_list_and_details_each_scroll_on_their_own_in_a_browser(
        root, serving, monkeypatch, tmp_path):
    """UI-07 (RC-C) as drawn (Node and Playwright, skipped without), the
    Diagnosis page with its Load for inference drawer open for an operator:
    side by side at 1850 x 960, the details of the chosen model longer than
    their region scroll on their own under the wheel and the list's rows stay
    where they were (they moved with the details: one region held both, 216
    px at ba6e85e), Load in view under both, nothing over it; stacked at 760
    x 1000, the list, the details and Load in that order, a file chosen
    brings its details into view with the focus on their title, and "↑ Back
    to the list" brings the row back into view and focuses it. At 1280 x 900,
    still side by side, the tree of every task's models (614 px) is whole in
    its region and the details in theirs: the list was 524 px there, its tree
    cut at the Fingerprint column, before the shared tree took 7 px a side
    and the regions their measured minimums. Measured in the dashboard's own
    type (IBM Plex Sans, served beside the page as the API serves it): the
    test server served no fonts, so the page was drawn in the host's fallback
    and the tree, 612.9 px at its narrowest in this host's Inter, was 617 px
    in its 616 px region on the reviewer's host (the review of 6a2d52e) and
    655 px where no Inter is installed (DejaVu Sans); in IBM Plex Sans it is
    599.2 px. Fails on 6a2d52e (the fonts did not load).

    OPS-FIX-R2 (the recheck of d7ebfc3): with IBM Plex Sans loaded the tree
    was still 620 px in its 616 px region on the reviewer's host: the region's
    minimum was a number fitted to this host's text. It now takes the tree's
    own narrowest, as far as the details' minimum leaves room, so with the
    tree's words a little wider (`WIDER_TREE`, a stand-in for another host's
    metrics) the tree is still whole in its region and the details keep their
    minimum. Fails on d7ebfc3 (the tree 640 px in 616)."""
    from src.webui.app import mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    with served(_diagnosis_page(root, monkeypatch), **mount_options()) as base:
        seen = run_node(_DRAWER_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS)), tmp_path,
                        base[:-len("gradio_api")], WIDER_TREE)
    before, after = seen["before"], seen["after"]
    assert before["layout"] == "side"
    assert after["rows"] == before["rows"], seen                  # the list stays
    assert before["factsScrolls"] and after["factsScroll"] > 0, seen
    assert after["load"] == before["load"] and after["loadOnTop"], seen
    assert after["load"][1] <= after["innerH"], seen
    assert after["list"][0] == after["facts"][0], seen            # side by side
    assert seen["font"] and all(state == "loaded" for state in seen["font"]), seen["font"]
    fit = seen["fit"]                         # at 1280 x 900, nothing cut or scrolled sideways
    assert all(scroll <= client for client, scroll in fit.values()), fit
    # The same with the tree's words wider, the details at least their minimum.
    wider = seen["wider"]
    assert all(scroll <= client for client, scroll in wider.values()), wider
    assert wider["rows"][0] > fit["rows"][0], (fit, wider)        # the region took the tree's
    assert seen["widerFacts"] >= 222, seen["widerFacts"]
    stacked = seen["stacked"]
    assert stacked["layout"] == "stack"
    assert stacked["list"][1] <= stacked["facts"][0] < stacked["load"][0], stacked
    assert seen["picked"] is True and seen["back"] is True, seen


_DRAWER_HEAD_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
const SIZES = [[1850, 960], [1280, 900], [960, 1080], [390, 844]];
(async () => {
  const browser = await chromium.launch();
  const out = {};
  for (const [w, h] of SIZES) {
    const page = await (await browser.newContext({ viewport: { width: w, height: h },
                                                   deviceScaleFactor: 1 })).newPage();
    await page.addInitScript(QUEUE_JS);
    const idle = () => page.waitForFunction(() => window.__shepQueue && window.__shepQueue.idle(), null, { timeout: 30000 });
    await page.goto(process.argv[2], { waitUntil: 'load' });
    await page.waitForSelector('#shep-dx-change-btn:not([disabled])', { timeout: 60000 });
    await idle();
    await page.click('#shep-dx-change-btn');
    await page.waitForSelector('#shep-models-clinical-rows [data-key]', { state: 'attached', timeout: 30000 });
    await idle();
    await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
    out[`${w}x${h}`] = await page.evaluate(() => {
      const drawer = document.querySelector('#shep-dx-change');
      const edge = drawer.getBoundingClientRect().right - (parseFloat(getComputedStyle(drawer).borderRightWidth) || 0);
      // Each word drawn in the title and the task segment, where the browser draws it.
      const words = [];
      for (const root of [document.querySelector('#shep-dx-change-title'), document.querySelector('#shep-dx-change-task')]) {
        const walk = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
        for (let n = walk.nextNode(); n; n = walk.nextNode()) {
          const re = /\S+/g;
          for (let m = re.exec(n.textContent); m; m = re.exec(n.textContent)) {
            const range = document.createRange();
            range.setStart(n, m.index); range.setEnd(n, m.index + m[0].length);
            for (const b of range.getClientRects()) { words.push({ word: m[0], right: b.right }); }
          }
        }
      }
      const labels = [...document.querySelectorAll('#shep-dx-change-task label')].map((l) => {
        const span = l.querySelector('span') || l;
        return { text: l.innerText.trim(), whole: span.scrollWidth <= span.clientWidth + 1 };
      });
      return { edge, past: words.filter((x) => x.right > edge + 0.5), labels,
               drawer: [drawer.scrollWidth, drawer.clientWidth],
               page: [document.scrollingElement.scrollWidth, innerWidth] };
    });
    await page.close();
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_the_drawers_title_and_task_segment_stay_inside_it_at_every_window(
        root, serving, monkeypatch, tmp_path):
    """V-19 (FL-UI-03's T04 review line, the WebUI skill §2): stacked at 390 x
    844 the Load for inference drawer's title note ran 217 px and its task
    segment 49 px past the drawer's right edge, cut ("operator controls · one
    mo", "Patients-like-"): both took their one line's width and never gave
    way. In a real browser (skipped without Node and Playwright), the drawer
    open for an operator at 1850 x 960, 1280 x 900, 960 x 1080 and 390 x 844:
    every word of its title and of its task segment is drawn inside the
    drawer, each segment label whole (no ellipsis), and nothing scrolls
    sideways, the drawer or the page. Fails on 78444a6 at 390 x 844."""
    from src.webui.app import mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    with served(_diagnosis_page(root, monkeypatch), **mount_options()) as base:
        seen = run_node(_DRAWER_HEAD_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS)), tmp_path,
                        base[:-len("gradio_api")])
    assert set(seen) == {"1850x960", "1280x900", "960x1080", "390x844"}
    for size, got in seen.items():
        assert got["past"] == [], (size, got)
        assert [label["text"] for label in got["labels"]] == [
            "All tasks", "Disease", "Gene", "Patients-like-me"], (size, got)
        assert all(label["whole"] for label in got["labels"]), (size, got)
        assert got["drawer"][0] <= got["drawer"][1] and got["page"][0] <= got["page"][1], (size, got)


#: The Load drawer open, a file chosen, at each window: its facts' title, with
#: "↑ Back to the list", as the browser draws it.
_FACTS_TITLE_JS = r"""
const { chromium } = require('playwright');
const QUEUE_JS = __QUEUE_JS__;
const ROWS = '#shep-models-clinical-rows', FACTS = '#shep-models-clinical-facts';
const SIZES = [[1850, 960], [760, 1000], [390, 844]];
(async () => {
  const browser = await chromium.launch();
  const out = {};
  for (const [w, h] of SIZES) {
    const page = await (await browser.newContext({ viewport: { width: w, height: h },
                                                   deviceScaleFactor: 1 })).newPage();
    await page.addInitScript(QUEUE_JS);
    const idle = () => page.waitForFunction(() => window.__shepQueue && window.__shepQueue.idle(), null, { timeout: 30000 });
    await page.goto(process.argv[2], { waitUntil: 'load' });
    await page.waitForSelector('#shep-dx-change-btn:not([disabled])', { timeout: 60000 });
    await idle();
    await page.click('#shep-dx-change-btn');
    await page.waitForSelector(`${ROWS} [data-key]`, { state: 'attached', timeout: 30000 });
    await idle();
    const key = await page.$eval(`${ROWS} tr[data-key]:not([hidden])`, (r) => r.getAttribute('data-key'));
    await page.click(`${ROWS} tr[data-key="${key}"]`);
    await page.waitForFunction(([F, k]) => { const t = document.querySelector(F + ' [data-facts-key]');
      return !!t && t.getAttribute('data-facts-key') === k; }, [FACTS, key], { timeout: 30000 });
    await idle();
    await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
    out[`${w}x${h}`] = await page.evaluate((F) => {
      const title = document.querySelector(F + ' [data-facts-key]');
      const back = title.querySelector('button.shep-back-list');
      const tb = title.getBoundingClientRect(), bb = back.getBoundingClientRect();
      const lines = (e) => { const range = document.createRange(); range.selectNodeContents(e);
        return new Set([...range.getClientRects()].map((x) => Math.round(x.top))).size; };
      const mid = (e) => { const b = e.getBoundingClientRect(); return b.top + b.height / 2; };
      // Each word drawn in the title, where the browser draws it.
      const past = [];
      const walk = document.createTreeWalker(title, NodeFilter.SHOW_TEXT);
      for (let n = walk.nextNode(); n; n = walk.nextNode()) {
        const re = /\S+/g;
        for (let m = re.exec(n.textContent); m; m = re.exec(n.textContent)) {
          const range = document.createRange();
          range.setStart(n, m.index); range.setEnd(n, m.index + m[0].length);
          for (const b of range.getClientRects()) {
            if (b.width && (b.left < tb.left - 0.5 || b.right > tb.right + 0.5)) { past.push(m[0]); }
          }
        }
      }
      return { layout: document.documentElement.getAttribute('data-shep-layout'),
               shown: getComputedStyle(back).display !== 'none',
               backLines: back.offsetParent ? lines(back) : 0,
               backWhole: back.scrollWidth <= back.clientWidth && back.scrollHeight <= back.clientHeight,
               backIn: bb.left >= tb.left - 0.5 && bb.right <= tb.right + 0.5,
               backHeight: Math.round(bb.height), past,
               chipBesideName: Math.abs(mid(title.querySelector('.shep-tag')) - mid(title.querySelector('b'))) < 2,
               page: [document.scrollingElement.scrollWidth, innerWidth] };
    }, FACTS);
    await page.close();
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_back_to_the_list_keeps_its_words_on_one_line_in_a_stacked_drawer(
        root, serving, monkeypatch, tmp_path):
    """V-23 (the verification of 78444a6, R-28's look): stacked, the Load
    drawer's chosen model's title kept "↑ Back to the list" beside the name
    and the "selected" chip, and the button got what they left: 57 px at 390
    x 844, its words on three lines inside its 32 px box; 117 px and two
    lines at 760 x 1000. In a real browser (skipped without Node and
    Playwright), the drawer open for an operator and a file chosen: stacked
    (760 x 1000, 390 x 844) the button keeps its words on one line, whole,
    inside its box and the title, taking a line of its own when the name and
    chip leave it no room; every word of the title inside it; nothing scrolls
    sideways. Side by side at 1850 x 960, unchanged: no button, the chip
    beside the name. Fails on 78444a6 at 760 x 1000 and 390 x 844."""
    from src.webui.app import mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    with served(_diagnosis_page(root, monkeypatch), **mount_options()) as base:
        seen = run_node(_FACTS_TITLE_JS.replace("__QUEUE_JS__", json.dumps(QUEUE_JS)), tmp_path,
                        base[:-len("gradio_api")])
    side = seen.pop("1850x960")
    assert side["layout"] == "side" and not side["shown"], side
    assert side["chipBesideName"] and side["past"] == [], side
    assert set(seen) == {"760x1000", "390x844"}
    for size, got in seen.items():
        assert got["layout"] == "stack" and got["shown"], (size, got)
        assert got["backLines"] == 1 and got["backWhole"] and got["backIn"], (size, got)
        assert got["backHeight"] == 32 and got["past"] == [], (size, got)
        assert got["page"][0] <= got["page"][1], (size, got)


def test_in_use_is_matched_by_the_full_fingerprint_only(root, serving, monkeypatch):
    sha = root["shas"]["model-05.pt"]
    serving["value"]["served_identity"]["checkpoint"]["sha256"] = sha[:12] + "f" * 52
    listing = ml.list_models("demo", operator_request(monkeypatch))
    html = ml.list_html(listing, None, ml.PRESETS["clinical"])
    assert "In use now" not in html and "In use<" not in html


def test_each_rows_recorded_tests_come_from_the_ledger_beside_it(root, serving, monkeypatch):
    from src.evaluation.sidecar import empty_ledger, write_ledger

    checkpoints = root["workspace"] / "runs" / "gatv2"
    write_ledger(checkpoints / RUNS[0][1] / "checkpoints" / "evaluations.json",
                 empty_ledger(), None)
    (checkpoints / RUNS[1][1] / "checkpoints" / "evaluations.json").write_text("{not json")
    listing = ml.list_models("demo", operator_request(monkeypatch))
    recorded = listing["recorded"]
    assert recorded[_key("model-12.pt")]["state"] == "none"
    assert recorded[_key("model-05.pt")]["state"] == "not_read"
    assert recorded[_key("model-03.pt")]["state"] == "no_ledger"
    assert _key("broken.pt") not in recorded                      # unreadable: no state


def test_contested_and_unchecked_are_counted_apart_from_the_report_store(root, monkeypatch):
    """A record whose group's runs differ in its version's report store
    (``contested``, a description since K8-6r.2) is counted; one whose
    comparison is incomplete (its store, or a version the registry does not
    list) is counted apart and never as agreeing, and never hides a known
    difference (M5-3 review EVAL-3: one uncheckable record turned
    "2 · 1 contested" into "3 · contested state not checked"). The row keeps
    one " · " (the skill's §5). The four records are real ones, registered in
    order, since the listing reads every record through the readers' one
    listing (K8-6e.1): only records that read as results are counted."""
    import src.evaluation.report_store as report_store
    import src.evaluation.sidecar as sidecar
    from src.webui.components import renderers as r
    from tests.unit.test_evaluation_sidecar import cohort_version, registered_report

    sha = root["shas"]["model-12.pt"]
    ledger = sidecar.empty_ledger()
    for n, digest in enumerate(("1" * 64, "1" * 64, "2" * 64, "3" * 64), start=1):
        report = registered_report(
            artifact_digests={"checkpoint": sha, "samples": "5" * 64, "node_features": "2" * 64,
                              "edge_indices": "3" * 64, "num_nodes": "4" * 64,
                              "kg": "1" * 64},
            split="research_selection",
            cohort_version=cohort_version(role="research_selection", manifest_sha256=digest))
        record = sidecar.build_record(report, f"{n:x}" * 64)
        _, ledger = sidecar.append_record(ledger, record,
                                          run_id=f"20261016T120000Z-{sha[:12]}-0000000{n}",
                                          registered_at="2026-10-16T12:00:00Z")
    monkeypatch.setattr(sidecar, "read_ledger", lambda path: (ledger, "x" * 64))

    class Version:
        def __init__(self, digest):
            self.identity = type("I", (), {"sha256": digest})()

    class Versions:
        def by_digest(self, digest):
            if digest == "3" * 64:
                return None, "its cohort version is not one the server's registry lists"
            return Version(digest), None

    class Status:
        def __init__(self, state):
            self.state, self.differing, self.disagreeing_runs, self.problems = state, (), (), ()

    class Scan:
        reports: dict = {}

        def __init__(self, state):
            self._state = state

        def status(self, key):
            return Status(self._state)

    from src.api.services.record_states import CohortVersions

    monkeypatch.setattr(CohortVersions, "listed",
                        classmethod(lambda cls, environ=None: Versions()))
    states = {"1" * 64: "contested", "2" * 64: "uncontested"}
    monkeypatch.setattr(report_store, "scan_report_store",
                        lambda version: Scan(states[version.identity.sha256]))
    item = {"architecture": "gatv2", "run": RUNS[0][1], "filename": "model-12.pt",
            "sha256": sha, "problem": None}
    result = ml.recorded_tests_for(root["workspace"], [item])[ml.item_key(item)]
    assert (result.state, result.count, result.contested, result.unchecked) == (
        "records", 4, 2, 1)
    assert r.recorded_tests_text(result) == (
        "Recorded results: 4 · runs differ for 2; comparison incomplete for 1")
    states["2" * 64] = "cannot_be_checked"
    result = ml.recorded_tests_for(root["workspace"], [item])[ml.item_key(item)]
    assert (result.count, result.contested, result.unchecked) == (4, 2, 2)
    states["1" * 64] = "uncontested"
    states["2" * 64] = "uncontested"
    result = ml.recorded_tests_for(root["workspace"], [item])[ml.item_key(item)]
    assert r.recorded_tests_text(result) == (
        "Recorded results: 4 · comparison incomplete for 1")
    # K8-6e.1: a record that loses a field is counted apart, never as a result,
    # and never in a group's count; the others still read.
    del ledger["records"][0]["n_ranked"]
    states["1" * 64] = "contested"
    result = ml.recorded_tests_for(root["workspace"], [item])[ml.item_key(item)]
    assert (result.count, result.contested, result.unchecked, result.unreadable) == (3, 1, 1, 1)
    assert r.recorded_tests_text(result) == (
        "Recorded results: 3 · 1 not readable; runs differ for 1; comparison incomplete for 1")


# --------------------------------------------------------------------------- the Load
def _listing(root, monkeypatch):
    return ml.list_models("demo", operator_request(monkeypatch))


def _drawn(demo, handle, listing, selected=None, older=False, view=None, search="", fold=None,
           said=None):
    """The list's one draw (`_draw`, on its token's and its view's change) on
    a session's state and the search box: what it draws, by component."""
    (draw,) = [fn for fn in demo.fns.values() if fn.name == "_draw" and handle.rows in fn.outputs]
    out = draw.fn(*[listing, selected, older, view, search, fold or {}, said][:len(draw.inputs)])
    return dict(zip(map(id, draw.outputs), out, strict=True))


def test_load_needs_d8_on_its_own_request_and_resolves_nothing_when_refused(
        root, serving, monkeypatch):
    from src.api.services import server_lists

    listing = _listing(root, monkeypatch)
    key = _key("model-12.pt")
    monkeypatch.setattr(server_lists, "resolve_checkpoint",
                        lambda *a, **k: pytest.fail("a refused Load resolved a checkpoint"))
    plan, html = ml.request_load(listing, key, gradio_request(server=("192.0.2.10", 8264)))
    assert plan is None and "Refused: " in html


def _changed_rows(html):
    rows = [row for row in html.split("<tr") if 'class="shep-changed"' in row]
    return [row.split('<th scope="row">')[1].split("<")[0] for row in rows]


def _from_cells(html):
    """The confirmation's "From — in use now" column as a person reads it: each
    row's label and its first cell, tags dropped and entities decoded."""
    import html as html_lib

    cells = {}
    for row in html.split("<tr")[1:]:
        if '<th scope="row">' not in row:
            continue
        label = row.split('<th scope="row">')[1].split("<")[0]
        cell = row.split("<td>")[1].split("</td>")[0]
        cells[label] = html_lib.unescape(re.sub(r"<[^>]+>", "", cell)).strip()
    return cells


def test_load_shows_one_confirmation_from_to_with_the_changed_rows_marked(
        root, serving, monkeypatch):
    listing = _listing(root, monkeypatch)
    key = _key("model-12.pt")
    plan, html = ml.request_load(listing, key, operator_request(monkeypatch))
    new, old = root["shas"]["model-12.pt"][:12], root["shas"]["model-05.pt"][:12]
    assert plan["to"]["sha256"] == root["shas"]["model-12.pt"]
    assert plan["from"]["sha256"] == root["shas"]["model-05.pt"]
    assert f"Reload the pipeline with <code>{new}</code> for every session?" in html
    assert ml.REPLACES_NOTE in html
    # Review of B4 (finding 4): the page's wait says only its own wait, the
    # load's state is the server's (the Diagnosis card's Loading), and no
    # word claims diagnoses wait: the model in use keeps answering every
    # session until the new one is published (I5).
    wait = html[html.index('<div class="shep-load-wait"'):]
    wait = wait[:wait.index("</div>")]
    assert "Reload sent; waiting for the server&#x27;s answer…" in wait and "Loading" not in wait
    assert "wait" not in ml.REPLACES_NOTE and "until then nothing changes" in ml.REPLACES_NOTE
    assert "Diagnoses wait" not in html
    assert _changed_rows(html) == ["Model", "Model fingerprint"]
    assert "disease · GATv2 2026-10-17 09:15, recorded epoch 5" in html and old in html
    assert "same dataset" in html and ">same<" in html
    # The task and its scoring mode from the one table, on both sides; the
    # disease task uses no shortest-path table, said as such on both sides.
    assert html.count("Disease (<code>disease_task</code>)") == 2
    assert html.count("not used by this task") == 2 and ">absent<" not in html
    assert "eta" not in html                         # no mixing weight is predicted
    # not a client path: the plan names the item, never a file
    assert all("/" not in str(plan[k]) or k == "run" for k in ("workspace", "filename"))
    # A gene model's To side: its task and the term it computes from its view.
    plan, html = ml.request_load(listing, _key("model-03.pt"),
                                 operator_request(monkeypatch))
    assert plan["to"]["mode"] == "gene_task" and plan["to"]["sp"] == "computed"
    assert _changed_rows(html) == ["Model", "Model fingerprint", "Task and scoring", "SP table"]
    assert "Gene (<code>gene_task</code>)" in html
    assert "computed from the model's graph view" in html
    # Nothing served (no model loaded, the first screen since K7): the From
    # rows that describe what is served say "not applicable", never words that
    # describe something served (unlisted, absent, files, and the fingerprint
    # "not recorded", which read as a model whose fingerprint is missing: the
    # round's closing check).
    serving["value"] = {"initialized": False}
    plan, html = ml.request_load(_listing(root, monkeypatch), key,
                                 operator_request(monkeypatch))
    frm = _from_cells(html)
    assert plan["from"]["served"] is False and frm["Model"] == "no model loaded"
    for row in ("Model fingerprint", "Trained on", "Serving workspace", "SP table", "Data"):
        assert frm[row] == "not applicable", row
    shown = " ".join(frm.values())
    for word in ("not in this workspace's list", "not a listed workspace", "absent", "files",
                 "not recorded"):
        assert word not in shown, word
    assert ">same<" not in html and "same dataset" not in html


def test_an_unloadable_item_cannot_be_loaded(root, serving, monkeypatch):
    listing = _listing(root, monkeypatch)
    plan, html = ml.request_load(listing, _key("broken.pt"),
                                 operator_request(monkeypatch))
    assert plan is None and "Select a model that can be loaded" in html
    plan, html = ml.request_load(listing, "../../../etc/passwd",
                                 operator_request(monkeypatch))
    assert plan is None


def _refused_record(root):
    """A file whose model record this build refuses (another record version),
    in a run of its own; its key."""
    from src.utils.checkpoint_io import write_checkpoint

    run = "20261019-080000_disease_ffffff"
    path = root["workspace"] / "runs" / "gatv2" / run / "checkpoints" / "model-09.pt"
    path.parent.mkdir(parents=True)
    write_checkpoint({**payload("disease", 9, root["fingerprint"], root["parent"]),
                      "record_version": 999}, path)
    return f"gatv2/{run}/model-09.pt"


def test_a_base_a_file_without_a_model_spec_and_a_refused_record_are_refused_before_any_read(
        root, serving, monkeypatch):
    """README §5 ("a base offers no diagnosis"), §4 and K6 decision 1: a base,
    a file without a model spec (its record refused by the one reader, in its
    one phrase) and any other refused record are refused from the listed
    record, before the file is resolved, hashed or the status read (the
    pipeline would refuse a base only after reading the graph)."""
    from src.api.services import server_lists
    from src.evaluation import sidecar

    refused = _refused_record(root)
    listing = _listing(root, monkeypatch)
    assert ml.find_item(listing, refused)["record"] == "refused"
    for target, name in ((server_lists, "resolve_checkpoint"), (sidecar, "checkpoint_matches"),
                         (ml, "read_status")):
        monkeypatch.setattr(target, name, lambda *a, **k: pytest.fail("a refused row was read"))
    for key, why in ((_key("base.pt"), ml.BASE_REFUSAL),
                     (_key("model-07.pt"), f"Record not readable: {WRITTEN_BEFORE_SPEC}."),
                     (refused, "Record not readable: ")):
        plan, html = ml.request_load(listing, key, operator_request(monkeypatch))
        assert plan is None and why in html and "Nothing was done." in html, key
        assert not ml.can_load(ml.find_item(listing, key))
    assert "record version 999" in ml.load_refusal(ml.find_item(listing, refused))
    assert ml.can_load(ml.find_item(listing, _key("model-12.pt")))
    # The reason beside Load is escaped (a record's problem is the server's words).
    evil = "<img src=x onerror=alert(1)>"
    shown = ml.load_reason_html({"record": "refused", "record_problem": evil,
                                 "sha256": "a" * 64, "problem": None})
    assert evil not in shown and "Record not readable: &lt;img src=x" in shown


def _plan(root, monkeypatch):
    listing = _listing(root, monkeypatch)
    plan, _html = ml.request_load(listing, _key("model-12.pt"), operator_request(monkeypatch))
    return plan


def test_confirm_needs_d8_again_and_a_refusal_reloads_nothing(root, serving, monkeypatch):
    plan = _plan(root, monkeypatch)
    calls = []
    step = ml.finish_load(plan, gradio_request(headers=[("Origin", "https://evil.example")]),
                          reload=lambda request: calls.append(request))
    assert calls == [] and "Refused: " in step.reason_html and step.plan == plan
    assert step.changed is False


def test_a_model_in_use_that_changed_since_is_said_never_reloaded_over(root, serving, monkeypatch):
    plan = _plan(root, monkeypatch)
    serving["value"] = _status(root, served="model-08.pt")
    calls = []
    step = ml.finish_load(plan, operator_request(monkeypatch),
                          reload=lambda request: calls.append(request))
    assert calls == [] and step.plan is not None and step.changed is False
    assert ml.FROM_CHANGED in step.confirm_html and ml.NOTHING_RELOADED in step.confirm_html
    assert step.plan["from"]["sha256"] == root["shas"]["model-08.pt"]
    # The From column is read as the confirmation reads it, from the plan's
    # own listing: the served workspace is named, the model and what it was
    # trained on are known, and an unchanged row is not marked.
    assert step.plan["from"]["model"] == "disease · GATv2 2026-10-18 09:12, recorded epoch 8"
    assert "not a listed workspace" not in step.confirm_html
    assert "not in this workspace's list" not in step.confirm_html
    assert _changed_rows(step.confirm_html) == ["Model", "Model fingerprint"]
    # Confirmed again as shown, it reloads.
    sent = []
    from src.api.routes.pipeline import PipelineReloadResponse, PipelineStatusResponse

    again = ml.finish_load(step.plan, operator_request(monkeypatch), reload=lambda request: (
        sent.append(request) or PipelineReloadResponse(success=True, message="ok",
                                                       status=PipelineStatusResponse(**_status(
                                                           root, served="model-12.pt")))))
    assert len(sent) == 1 and "Reloaded." in again.confirm_html


@pytest.mark.parametrize("change", ["workspace", "eta"])
def test_any_part_of_what_is_served_that_changed_is_said(root, serving, monkeypatch, change,
                                                       tmp_path):
    """The same checkpoint, served by another session with another workspace
    or eta: the Load says so rather than reloading over it. (The shortest-path
    state follows the model's task since K8-4, so the same checkpoint cannot be
    served with another one.)"""
    plan = _plan(root, monkeypatch)
    status = _status(root)
    if change == "workspace":
        other = tmp_path / "elsewhere"
        other.mkdir()
        status["current_data_dir"] = str(other)
    else:
        status["served_identity"]["settings"]["eta"] = 0.5
    serving["value"] = status
    calls = []
    step = ml.finish_load(plan, operator_request(monkeypatch),
                          reload=lambda request: calls.append(request))
    assert calls == [] and step.plan is not None and ml.FROM_CHANGED in step.confirm_html


def test_a_reload_of_the_same_checkpoint_with_another_workspace_is_a_change(
        root, serving, monkeypatch, tmp_path):
    """"served model changed" follows the whole served identity: the same
    checkpoint served from another directory is a change for the card."""
    from src.api.routes.pipeline import PipelineReloadResponse, PipelineStatusResponse

    other = tmp_path / "elsewhere"
    other.mkdir()
    serving["value"] = _status(root, served="model-05.pt", data_dir=other)
    listing = _listing(root, monkeypatch)
    plan, _html = ml.request_load(listing, _key("model-05.pt"),
                                  operator_request(monkeypatch))
    step = ml.finish_load(plan, operator_request(monkeypatch), reload=lambda request: (
        PipelineReloadResponse(success=True, message="ok", status=PipelineStatusResponse(
            **_status(root, served="model-05.pt")))))
    assert "Reloaded." in step.confirm_html and step.served == root["shas"]["model-05.pt"]
    assert step.changed is True
    unchanged = ml.finish_load(plan, operator_request(monkeypatch), reload=lambda request: (
        PipelineReloadResponse(success=False, message="refused", status=PipelineStatusResponse(
            **_status(root, served="model-05.pt", data_dir=other)))))
    assert unchanged.changed is False


def test_a_refused_reload_is_shown_without_a_server_path(root, serving, monkeypatch):
    """Review of B4 (finding 7): the reload route answers its operator with
    the build's own words, which name the checkpoint's absolute path (a
    listed child whose workspace graph changed after training is refused at
    build so); the page shows them through the dashboard's one redaction
    (`workspaces.redact`, as the list's problems), never the path."""
    from src.api.routes.pipeline import PipelineReloadResponse, PipelineStatusResponse

    plan = _plan(root, monkeypatch)
    checkpoint = (root["workspace"] / "runs" / "gatv2" / RUNS[0][1] / "checkpoints"
                  / "model-12.pt")
    elsewhere = "/srv/other/place/last.pt"
    message = (f"Pipeline initialization failed: Checkpoint {checkpoint} does not match this "
               f"workspace's graph: kg differs. Not {elsewhere}.")
    step = ml.finish_load(plan, operator_request(monkeypatch), reload=lambda request: (
        PipelineReloadResponse(success=False, message=message, status=PipelineStatusResponse(
            **_status(root)))))
    assert "Not reloaded; the model in use is unchanged." in step.confirm_html
    assert str(root["root"]) not in step.confirm_html and elsewhere not in step.confirm_html
    inside = checkpoint.relative_to(root["root"])
    assert f"Checkpoint {inside} does not match" in step.confirm_html
    assert "a path outside the workspaces root" in step.confirm_html


def _rewrite(root, name, epoch, kind="disease"):
    """Write other bytes into a listed checkpoint, as a training run that
    overwrites ``last.pt`` each epoch does (by default another epoch of the
    same disease model)."""
    from src.utils.checkpoint_io import write_checkpoint

    path = root["workspace"] / "runs" / "gatv2" / RUNS[0][1] / "checkpoints" / name
    return write_checkpoint(payload(kind, epoch, root["fingerprint"], root["parent"]),
                            path), path


def test_bytes_that_changed_after_the_confirmation_are_shown_never_loaded(
        root, serving, monkeypatch):
    """The trainer rewrites the file between "Load" and "Reload with …": the
    confirmation is shown again with what the file holds now; nothing is
    reloaded until that is confirmed."""
    from src.api.routes.pipeline import PipelineReloadResponse, PipelineStatusResponse

    plan = _plan(root, monkeypatch)
    new_sha, path = _rewrite(root, "model-12.pt", 13)
    calls = []
    step = ml.finish_load(plan, operator_request(monkeypatch),
                          reload=lambda request: calls.append(request))
    assert calls == [] and step.plan is not None and step.changed is False
    assert ml.TO_CHANGED in step.confirm_html
    assert step.plan["to"]["sha256"] == new_sha and step.plan["to"]["model"].endswith("epoch 13")
    assert f"Reload the pipeline with <code>{new_sha[:12]}</code>" in step.confirm_html
    sent = []
    again = ml.finish_load(step.plan, operator_request(monkeypatch), reload=lambda request: (
        sent.append(request) or PipelineReloadResponse(
            success=True, message="ok", status=PipelineStatusResponse(**{
                **_status(root), "served_identity": {
                    **_status(root)["served_identity"],
                    "checkpoint": {"state": "verified", "sha256": new_sha}}}))))
    assert [Path(r.checkpoint_path) for r in sent] == [path]
    assert "Reloaded." in again.confirm_html and again.served == new_sha


def test_a_file_rewritten_into_a_base_after_the_confirmation_is_never_reloaded(
        root, serving, monkeypatch):
    """The refusal is checked again on what the file holds at "Reload with
    …": a base written over the confirmed file is refused, never shown to
    confirm and never reloaded."""
    plan = _plan(root, monkeypatch)
    _rewrite(root, "model-12.pt", 3, kind="base")
    calls = []
    step = ml.finish_load(plan, operator_request(monkeypatch),
                          reload=lambda request: calls.append(request))
    assert calls == [] and step.plan is None and step.changed is False
    assert "Not reloaded." in step.confirm_html and ml.BASE_REFUSAL in step.confirm_html
    # Asked for again from the list read before: refused from what it holds now.
    plan, html = ml.request_load(_listing(root, monkeypatch), _key("model-12.pt"),
                                 operator_request(monkeypatch))
    assert plan is None and ml.BASE_REFUSAL in html


def test_a_workspace_sp_table_never_changes_a_task_models_to_side(root, serving, monkeypatch):
    """The To side was guessed from the workspace (an SP table there read as
    "gnn_plus_shortest_path" with the server's eta), which a task model's
    serving never reads: the mode is the task's, from the one table, so a
    table appearing after the confirmation changes nothing and the Load goes
    on with the task its record names."""
    from src.api.routes.pipeline import PipelineReloadResponse, PipelineStatusResponse

    plan = _plan(root, monkeypatch)
    assert (plan["to"]["mode"], plan["to"]["sp"]) == ("disease_task", "absent")
    (root["workspace"] / "shortest_paths.pt").write_bytes(b"a table")
    sent = []
    step = ml.finish_load(plan, operator_request(monkeypatch), reload=lambda request: (
        sent.append(request) or PipelineReloadResponse(success=True, message="ok",
                                                       status=PipelineStatusResponse(**_status(
                                                           root, served="model-12.pt")))))
    assert len(sent) == 1 and ml.TO_CHANGED not in step.confirm_html
    assert "Reloaded." in step.confirm_html and sent[0].task == "disease"


def test_the_to_sides_sp_state_is_the_served_identitys_one_rule(root, serving, monkeypatch):
    """The SP state the To column says for a task model is the one its scoring
    mode is served with (`core.types.SCORING_MODE_SP_STATES`, the rule
    `ServedIdentity` checks and the pipeline builds from), read from that one
    table and never from a copy: a change of the rule changes what the
    confirmation says, so reloading the same model never marks its SP row
    changed."""
    from src.core import types

    listing = _listing(root, monkeypatch)
    item = ml.find_item(listing, _key("model-12.pt"))
    for task, mode in types.TASK_SCORING_MODES.items():
        side = ml._to_side({**item, "task": task}, "demo", root["workspace"], [])
        assert (side["mode"], side["sp"]) == (mode, types.SCORING_MODE_SP_STATES[mode])
    monkeypatch.setitem(types.SCORING_MODE_SP_STATES, "patients_like_me", "computed")
    side = ml._to_side({**item, "task": "patients_like_me"}, "demo", root["workspace"], [])
    assert side["sp"] == "computed"


def test_a_list_read_earlier_is_caught_when_load_is_asked(root, serving, monkeypatch):
    listing = _listing(root, monkeypatch)
    new_sha, _path = _rewrite(root, "model-12.pt", 14)
    plan, html = ml.request_load(listing, _key("model-12.pt"),
                                 operator_request(monkeypatch))
    assert plan["to"]["sha256"] == new_sha and plan["to"]["model"].endswith("epoch 14")
    assert "The file changed since the list was read" in html and new_sha[:12] in html


def test_load_finds_the_item_in_the_servers_list_again(root, serving, monkeypatch):
    """A listed item whose file is gone since the list was read is refused,
    and nothing is opened from the listing's word for it."""
    listing = _listing(root, monkeypatch)
    (root["workspace"] / "runs" / "gatv2" / RUNS[0][1] / "checkpoints" / "model-12.pt").unlink()
    plan, html = ml.request_load(listing, _key("model-12.pt"),
                                 operator_request(monkeypatch))
    assert plan is None and "Refused: " in html and "is listed in the workspace" in html


def test_the_reload_gets_what_the_server_resolves_and_the_fingerprint_shown(
        root, serving, monkeypatch):
    from src.api.routes.pipeline import PipelineReloadResponse, PipelineStatusResponse

    plan = _plan(root, monkeypatch)
    sent = []

    def reload(request):
        sent.append(request)
        status = _status(root, served="model-12.pt")
        return PipelineReloadResponse(success=True, message="ok",
                                      status=PipelineStatusResponse(**status))

    step = ml.finish_load(plan, operator_request(monkeypatch), reload=reload)
    (request,) = sent
    expected = (root["workspace"] / "runs" / "gatv2" / RUNS[0][1] / "checkpoints" / "model-12.pt")
    assert Path(request.data_dir) == root["workspace"].resolve()
    assert Path(request.checkpoint_path) == expected
    assert request.expected_checkpoint_sha256 == root["shas"]["model-12.pt"]
    # The task the confirmed model's record names: the reload refuses a build
    # that serves another (switching task means loading another model).
    assert request.task == "disease"
    assert step.changed is True and step.served == root["shas"]["model-12.pt"]
    assert "Reloaded." in step.confirm_html and step.plan is None


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A complete workspace with a base and a disease child trained on it for
    real (`tests/fixtures/trained_children`, 1 epoch on CPU), once per module."""
    from tests.fixtures.registered_cohort import write_demo_workspace
    from tests.fixtures.trained_children import train_children

    return train_children(write_demo_workspace(tmp_path_factory.mktemp("mltrained") / "data"),
                          ("disease",))


@pytest.fixture
def buildable(trained, tmp_path, monkeypatch):
    """A workspaces root whose one workspace (``data``) the real reload builds:
    a graph, and a disease model finetuned on it as ``last.pt`` of its run, the
    file a running training job rewrites every epoch. Nothing is served yet;
    the device is the CPU."""
    import src.api.main as api_main
    from tests.fixtures.trained_children import CHILD_RUNS

    for field in ("pipeline", "kg", "_current_data_dir", "_current_checkpoint_path"):
        monkeypatch.setattr(api_main.app_state, field, None, raising=False)
    monkeypatch.setattr(api_main.app_state, "is_ready", False, raising=False)
    for name in ("SHEPHERD_KG_PATH", "SHEPHERD_DATA_DIR", "SHEPHERD_CHECKPOINT_PATH",
                 "SHEPHERD_REGISTRY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SHEPHERD_DEVICE", "cpu")
    root = tmp_path / "root"
    data_dir = shutil.copytree(trained["data_dir"], root / "data")
    run = CHILD_RUNS["disease"]
    last = data_dir / "runs" / "gatv2" / run / "checkpoints" / "last.pt"
    source = tmp_path / "source.pt"
    shutil.copyfile(last, source)
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(root))
    return {"api": api_main, "data_dir": data_dir, "last": last, "source": source,
            "key": f"gatv2/{run}/last.pt"}


def _shifted(source, shift):
    """The bytes of `source`'s model with every weight moved by `shift`: another
    model the build accepts (the same architecture, trained on the same graph)."""
    import io

    import torch

    contents = torch.load(source, map_location="cpu", weights_only=False)
    contents["state_dict"] = {name: value + shift if value.is_floating_point() else value
                              for name, value in contents["state_dict"].items()}
    buffer = io.BytesIO()
    torch.save(contents, buffer)
    return buffer.getvalue()


@pytest.mark.parametrize("how", ["rename", "rewrite"])
def test_bytes_replaced_before_the_reloads_own_read_are_refused_never_served(
        buildable, monkeypatch, how):
    """EB-1, through the real `run_reload` and `build_pipeline`. A model loaded
    as shown is served (the positive control). Then another model is listed and
    confirmed, and the file is replaced after the Load hashed it, while the build
    reads the workspace's graph and before it reads the checkpoint, as a
    training run's next epoch renames a new ``last.pt`` into place (and once in
    place). The reload compares its own read with the fingerprint shown: nothing
    is published, the model in use stays, and the page says not reloaded and
    why. Catches the reload given no fingerprint (or not comparing it) and a
    check after publication."""
    from tests.fixtures.replacement import replace_after_read, replace_file

    api, last = buildable["api"], buildable["last"]
    published = []
    publish = api.publish_pipeline

    def spy(bundle):
        published.append(bundle.config["served_identity"]["checkpoint"]["sha256"])
        return publish(bundle)

    monkeypatch.setattr(api, "publish_pipeline", spy)

    def plan_for_last():
        listing = ml.list_models("data", operator_request(monkeypatch))
        plan, html = ml.request_load(listing, buildable["key"], operator_request(monkeypatch))
        assert plan is not None, html
        return plan

    first = file_sha256(last)
    loaded = ml.finish_load(plan_for_last(), operator_request(monkeypatch))
    assert "Reloaded." in loaded.confirm_html, loaded.confirm_html
    assert published == [first] and loaded.served == first
    serving = api.app_state.pipeline
    assert serving.served_identity().checkpoint.sha256 == first

    replace_file(last, _shifted(buildable["source"], 1.0), how)
    shown = file_sha256(last)
    plan = plan_for_last()
    assert plan["to"]["sha256"] == shown
    other = _shifted(buildable["source"], 2.0)
    swap = replace_after_read(monkeypatch, after=buildable["data_dir"] / "kg.json",
                              target=last, data=other, how=how)
    step = ml.finish_load(plan, operator_request(monkeypatch))

    assert swap.fired == 1
    held = file_sha256(last)
    assert held not in (first, shown)
    assert published == [first]
    assert api.app_state.pipeline is serving
    assert serving.served_identity().checkpoint.sha256 == first
    assert "Not reloaded; the model in use is unchanged." in step.confirm_html
    assert "is not the model that was shown" in step.confirm_html
    assert f"it holds {held[:12]} now, not {shown[:12]}" in step.confirm_html
    assert "Reloaded" not in step.confirm_html.replace("Not reloaded", "")
    assert step.served == first and step.changed is False and step.plan is None


@pytest.mark.parametrize("bytes_now", ["torn", "another graph", "not a dict", "gone"])
def test_any_other_bytes_at_the_shown_file_are_refused_as_not_the_model_shown(
        buildable, monkeypatch, bytes_now):
    """Root-cause round 3 (INT-R1), through the real `run_reload` and
    `build_pipeline`: whatever replaced the confirmed file after the Load
    hashed it -- torn bytes, a checkpoint recorded for another graph, a pickle
    that is not a checkpoint, or nothing -- is refused on the build's one
    checkpoint read before torch parses it, as "not the model that was shown":
    never as unreadable, another graph's or not a checkpoint. No model is
    built, nothing is published, and the model in use stays."""
    from tests.unit.test_served_identity import (
        change_after_the_graph_read,
        never_parsed_or_built,
        not_the_shown_bytes,
    )

    api, last = buildable["api"], buildable["last"]

    def plan_for_last():
        listing = ml.list_models("data", operator_request(monkeypatch))
        plan, html = ml.request_load(listing, buildable["key"], operator_request(monkeypatch))
        assert plan is not None, html
        return plan

    shown = file_sha256(last)
    loaded = ml.finish_load(plan_for_last(), operator_request(monkeypatch))
    assert "Reloaded." in loaded.confirm_html, loaded.confirm_html
    serving = api.app_state.pipeline
    plan = plan_for_last()
    data = not_the_shown_bytes(last, bytes_now)
    fired = change_after_the_graph_read(monkeypatch, buildable["data_dir"], last, data)
    never_parsed_or_built(monkeypatch)

    step = ml.finish_load(plan, operator_request(monkeypatch))

    assert fired, "the build never read the graph"
    held = (f"it is gone; {shown[:12]} was shown" if data is None
            else f"it holds {file_sha256(last)[:12]} now, not {shown[:12]}")
    assert "Not reloaded; the model in use is unchanged." in step.confirm_html
    assert f"is not the model that was shown: {held}. Nothing was published" in \
        step.confirm_html, step.confirm_html
    assert api.app_state.pipeline is serving
    assert step.served == shown and step.changed is False and step.plan is None


def test_a_refused_reload_leaves_the_model_in_use_and_says_so(root, serving, monkeypatch):
    from src.api.routes.pipeline import PipelineReloadResponse, PipelineStatusResponse

    plan = _plan(root, monkeypatch)
    step = ml.finish_load(plan, operator_request(monkeypatch), reload=lambda request: (
        PipelineReloadResponse(success=False, message="No checkpoint found.",
                               status=PipelineStatusResponse(**_status(root)))))
    assert "Not reloaded; the model in use is unchanged." in step.confirm_html
    assert "No checkpoint found." in step.confirm_html and step.changed is False
    raising = ml.finish_load(plan, operator_request(monkeypatch),
                             reload=lambda request: (_ for _ in ()).throw(RuntimeError("x")))
    assert "Not reloaded" in raising.confirm_html and "RuntimeError" in raising.confirm_html


def test_a_markup_workspace_name_is_escaped_in_the_confirmation(root, serving, monkeypatch):
    listing = _listing(root, monkeypatch)
    plan, _html = ml.request_load(listing, _key("model-12.pt"),
                                  operator_request(monkeypatch))
    plan["to"]["workspace"] = "<b>x</b>"
    html = ml.confirmation_html(plan)
    assert "<b>x</b>" not in html and "&lt;b&gt;x&lt;/b&gt;" in html


# --------------------------------------------------------------------------- presets and views
def test_the_research_preset_is_a_tree_with_in_use_on_its_own_row(root, serving, monkeypatch):
    """D04: the research preset draws runs as summary rows, their files
    folded under them, each still a row of its own; no score column (a
    file's figures are its details, and validations of different tasks are
    never put on one scale); "In use (Diagnosis)" on the served file, whose
    run is unfolded and located."""
    listing = _listing(root, monkeypatch)
    research = ml.list_html(listing, None, ml.PRESETS["research"])
    assert '<table class="shep-tbl shep-mtbl shep-mtree ' in research
    assert "In use now" not in research
    heads = re.findall(r'<th scope="col"[^>]*>([^<]*)</th>', research)
    assert heads == list(ml.TREE_COLUMNS) == ["Model", "Task", "Operation", "Epochs", "Files",
                                              "Fingerprint"]
    assert "Validation" not in research and "0.3500" not in research
    served_row = _row_of(research, "model-05.pt")
    assert "In use (Diagnosis)" in served_row and " hidden" not in served_row
    assert research.count("In use (Diagnosis)") == 1
    assert f'data-locate="{_key("model-05.pt")}"' in research
    assert "Metadata not readable: the file cannot be read as a checkpoint; cannot be loaded" in (
        research)
    # Every file a row (nothing dropped by folding), the others' runs folded.
    assert research.count(" data-key=") == 9
    assert ' hidden><td><div class="shep-tree-name shep-tree-leaf' in _row_of(
        research, "model-12.pt")


def test_a_pages_view_orders_and_filters_its_groups(root, serving, monkeypatch):
    """A page's view: its filter, its groups (each its own row group, B72-C5,
    its runs in the order it gives, the rest after them), its context rows
    counted apart, and its note beside Load. The table draws no group header
    and no depth (the stage-lineage round: By family's families are the family
    graph's, `_graph_html`, and a group's title is no header row of its own)."""
    listing = _listing(root, monkeypatch)
    keys = [ml.item_key(i) for i in listing["items"]]
    first, second = _key("model-01.pt"), _key("model-12.pt")
    run_a = "gatv2/20261013-080000_disease_cccccc"
    view = {"task": "disease",
            "groups": [{"title": "Group A", "keys": [first, second, "gatv2/nope/x.pt"],
                        "runs": [{"run": run_a, "depth": 0},
                                 {"run": "gatv2/20261018-091200_disease_8c2e7a", "depth": 1}]},
                       {"title": "Group B", "keys": []}],
            "rest_title": "The rest",
            "note": "A note from the page <b>."}
    html = ml.list_html(listing, None, ml.PRESETS["research"], view=view)
    assert html.index(f'data-key="{first}"') < html.index(
        f'data-key="{second}"') < html.index(f'data-key="{_key("model-05.pt")}"')
    for other in ("run_e4c2a9", "gene_aaaaaa", "base_aaaaaa", "run_dddddd"):
        assert other not in html, other                    # not disease models: filtered out
    bodies = re.findall(r"<tbody[^>]*>(.*?)</tbody>", html)
    assert len(bodies) == 2 and 'scope="rowgroup"' not in html     # Group B has no row
    for retired in ("Group A", "Group B", "The rest"):
        assert retired not in _visible(html), retired              # no group header row
    for retired in ("shep-grp", "shep-d1", "shep-d0", "rowgroup"):
        assert retired not in html, retired
    assert ml.count_line(listing, ml.PRESETS["research"], False, view) == "Disease: 3 of 8 runs"
    # The page's note beside Load only for a selected row Load can serve
    # (R-78): nothing before a selection, nor for a base, whose reason is said.
    assert ml.note_html(view) == ""
    servable = next(i for i in listing["items"] if ml.can_load(i))
    assert ml.note_html(view, servable) == (
        '<span class="shep-note">A note from the page &lt;b&gt;.</span>')
    base = next(i for i in listing["items"] if i.get("learning_stage") == "base")
    assert ml.loadable(base) and ml.note_html(view, base) == ""
    assert len(keys) == 9
    # A view's explicit rows (a family view's, its context rows included).
    rows = ml.shown_items(listing, {"task": "disease", "keys": [_key("base.pt"), first]})
    assert [ml.item_key(i) for i in rows] == [first, _key("base.pt")]
    # Its count says the filter's runs and the context runs apart: the base
    # shown as the parent of the disease models is never counted as one.
    family = [_key(name) for name in ("base.pt", "model-12.pt", "model-08.pt", "model-05.pt",
                                      "model-01.pt")]
    view = {"task": "disease", "keys": family, "context": [_key("base.pt")],
            "groups": [{"title": "Family from base", "keys": family}]}
    assert ml.count_line(listing, ml.PRESETS["research"], False, view) == (
        "Disease: 3 of 8 runs, with 1 run they come from")


def test_one_filter_vocabulary_and_the_drawer_lists_task_models_only(root, serving, monkeypatch):
    """The list's filters (`TASK_FILTERS`) by the server's one predicate; the
    clinical preset shows models finetuned for a task whatever the view asks,
    so a direct call with no view (or one asking for bases) lists no base or
    refused file; each filter's empty state is said in its own words. No
    filter sets a file without a model spec apart: it is a row of All, as
    any refused record is."""
    listing = _listing(root, monkeypatch)
    research, clinical = ml.PRESETS["research"], ml.PRESETS["clinical"]

    def shown(view, preset=research):
        return sorted(i["filename"] for i in ml.shown_items(listing, view, preset))

    assert shown({"task": "base"}) == ["base.pt"]
    assert shown({"task": "disease"}) == ["model-01.pt", "model-05.pt", "model-08.pt",
                                          "model-12.pt"]
    assert shown({"task": "gene"}) == ["model-03.pt"]
    assert shown({"task": "patients_like_me"}) == ["model-02.pt"]
    assert "history" not in ml.TASK_FILTERS
    assert shown({"task": "history"}) == shown({"task": "all"})
    assert shown({"task": "task_models"}) == shown(None, clinical) == shown({"task": "all"},
                                                                            clinical)
    assert len(shown({"task": "all"})) == len(shown(None)) == len(shown({"task": "x"})) == 9
    assert "base.pt" not in shown(None, clinical) and "model-07.pt" not in shown(None, clinical)
    assert shown({"task": "base"}, clinical) == []
    assert ml.list_html(listing, None, clinical, view={"task": "base"}) == (
        '<p class="shep-empty">No base is listed in this workspace.</p>')
    no_gene = {**listing, "items": [i for i in listing["items"] if i.get("task") != "gene"]}
    assert "No gene model is listed in this workspace." in ml.list_html(
        no_gene, None, research, view={"task": "gene"})
    assert ml.count_line(no_gene, research, False, {"task": "gene"}) == (
        "Gene: no checkpoint shown")
    assert ml.count_line(listing, research, False, {"task": "patients_like_me"}) == (
        "Patients-like-me: 1 of 8 runs")
    # Each count in its own number (K6 closing smoke: "1 checkpoints in 1 runs").
    assert ml.count_line(listing, research, False) == "8 runs, 9 saved files"
    assert ml.count_line(listing, research, False, {"task": "gene"}) == "Gene: 1 of 8 runs"
    # The drawer's task filter is counted against what the drawer lists (its
    # five runs of task models), never against the bases and refused files it
    # hides; asking for every row it shows names no filter at all.
    assert ml.count_line(listing, clinical, False, {"task": "disease"}) == (
        "Disease: 3 of 5 runs")
    assert ml.count_line(listing, clinical, False, {"task": "task_models"}) == ml.count_line(
        listing, clinical, False, {"task": "all"}) == ml.count_line(listing, clinical, False)


def test_the_list_redraws_when_the_page_changes_its_view(root, serving, monkeypatch):
    from src.webui.shell import standalone_context

    with gr.Blocks() as demo:
        ctx = standalone_context(demo)
        handle = ctx.create_model_list("research", head=lambda: gr.Button("Full records"),
                                       foot=lambda: gr.Button("Compare"))
    (redraw,) = [fn for fn in demo.fns.values() if (handle.view._id, "change") in fn.targets]
    assert redraw.name == "_draw" and (handle.drawn._id, "change") in redraw.targets
    assert redraw.outputs == [handle.rows, handle.count, handle.facts, handle.load_button,
                              handle.reason, handle.note]
    listing = _listing(root, monkeypatch)
    # D04: a filter never changes what is selected; the list says it does not
    # show the selection, whose details stay.
    disease = _key("model-12.pt")
    rows, count, facts, load, reason, note = redraw.fn(listing, disease, False,
                                                       {"task": "gene"}, "", {}, None)
    assert "Selected model not shown by this filter (Task: Gene)" in rows
    assert disease not in re.findall(r'data-key="([^"]+)"', rows)
    assert "Select a model" not in facts and load["interactive"] is True and reason == ""
    _rows, _count, _facts, load, reason, _note = redraw.fn(listing, disease, False,
                                                           {"task": "disease"}, "", {}, None)
    assert load["interactive"] is True and reason == ""
    assert "not shown by this filter" not in _rows
    # A selection the listing does not hold is drawn as none (never described).
    gone = redraw.fn(listing, "gatv2/x/gone.pt", False, {"task": "disease"}, "", {}, None)
    assert "Select a model" in gone[2] and gone[3]["interactive"] is False
    assert handle.head_extra.value == "Full records" and handle.foot_extra.value == "Compare"
    assert handle.refresh["inputs"] == [handle.workspace, handle.show_older, handle.view,
                                        handle.listing, handle.selected, handle.search,
                                        handle.fold, handle.clock]


def test_the_list_is_drawn_in_one_place_from_the_state_as_it_is_then(
        root, serving, monkeypatch):
    """G07 (the review of 00258f4, SL-R1: a pick answered after a newer search
    drew the list for no search): what the list shows (its rows, its count,
    the facts) is drawn by one listener per list, on its draw token's and its
    view's change, from States (Gradio reads them when the draw runs) and the
    search box (the page reads it when it asks for the draw; the review of
    5171661: a search answered after a Clear drew its text again under the
    empty box), one draw of a list at a time and the last one asked for kept
    ("always_last"); every event that changes what the list shows writes the
    token and none draws the rows or the count itself. The draw is pure:
    nothing is read from disk or listed again."""
    import builtins

    demo, _ctx, handle, other = _component("clinical", both=True)
    for each in (handle, other):
        (draw,) = [fn for fn in demo.fns.values()
                   if fn.name == "_draw" and each.rows in fn.outputs]
        assert set(draw.targets) == {(each.drawn._id, "change"), (each.view._id, "change")}
        assert [i for i in draw.inputs if not isinstance(i, gr.State)] == [each.search]
        assert draw.trigger_mode == "always_last" and draw.concurrency_limit == 1
        assert draw.concurrency_id == f"{each.elem_id}-draw"
        for part in (each.rows, each.count, each.facts):
            assert [fn for fn in demo.fns.values() if part in fn.outputs] == [draw]
        for name in ("_refresh", "_select", "_searched"):
            changers = [fn for fn in demo.fns.values() if fn.name == name
                        and (each.drawn in fn.outputs or each.listing in fn.inputs)]
            assert changers and all(each.drawn in fn.outputs for fn in changers), name
    listing = _listing(root, monkeypatch)
    picked = _key("model-12.pt")
    with monkeypatch.context() as patched:
        patched.setattr(ml, "list_models", lambda *a, **k: pytest.fail("the draw listed"))
        patched.setattr(ml, "read_status", lambda *a, **k: pytest.fail("the draw read the status"))
        patched.setattr(builtins, "open", lambda *a, **k: pytest.fail("the draw read a file"))
        drawn = _drawn(demo, handle, listing, picked, False, {"task": "disease"}, "disease")
    assert 'aria-current="true"' in _row_of(drawn[id(handle.rows)], "model-12.pt")
    assert drawn[id(handle.count)].startswith("Disease: ") and "“disease”" in drawn[
        id(handle.count)]
    assert drawn[id(handle.load_button)]["interactive"] is True
    # The pure function the draw calls, the same from the same state.
    again = ml.list_drawing(listing, picked, ml.PRESETS["clinical"], view={"task": "disease"},
                            search="disease", fold={})
    assert (again.rows, again.count, again.facts) == (
        drawn[id(handle.rows)], drawn[id(handle.count)], drawn[id(handle.facts)])


class _Page:
    """One session of a list's page, as Gradio 6.30 runs its events
    (`served_page.ServedSession` for the real queue): a component's value is
    the page's when the event is asked, a ``gr.State`` the session's when it
    is processed, and an answer's outputs are applied before the changes they
    make fire their ``.change`` (the list's draw, on its token's change; the
    search, on the box's). `ask` gathers an event's component inputs; `run`
    processes it; `type` and `clear` change the box in the page, as a key
    typed, a delete or a paste, and Clear, do."""

    def __init__(self, demo, handle):
        self.demo = demo
        self.handle = handle
        # A session's own copy of each State's first value, as Gradio gives
        # each session (the list's clock a new one, its folds a new dict).
        self.values = {id(b): (copy.deepcopy(b.value) if isinstance(b, gr.State)
                               else getattr(b, "value", None)) for b in demo.blocks.values()}
        (self.draw,) = [fn for fn in demo.fns.values()
                        if fn.name == "_draw" and handle.rows in fn.outputs]
        # The box's change listeners: what the page asks when its text changes.
        self.on_search = [fn for fn in demo.fns.values()
                          if (handle.search._id, "change") in fn.targets]
        self.fns = {fn.name: fn for fn in demo.fns.values()
                    if set(map(id, fn.inputs + fn.outputs)) & {
                        id(handle.listing), id(handle.fold), id(handle.search),
                        id(handle.selected), id(handle.plan), id(handle.drawn)}}

    def __getitem__(self, block):
        return self.values[id(block)]

    def __setitem__(self, block, value):
        self.values[id(block)] = value

    def ask(self, name):
        fn = self.fns[name] if isinstance(name, str) else name
        return fn, {id(b): self.values[id(b)] for b in fn.inputs if not isinstance(b, gr.State)}

    def run(self, asked, *, evt=None, request=None):
        import inspect

        fn, gathered = asked
        inputs = iter([gathered.get(id(b), self.values[id(b)]) for b in fn.inputs])
        args = []
        # Gradio's special arguments by their parameter, the inputs in order.
        for param in inspect.signature(fn.fn).parameters.values():
            if param.name == "request":
                args.append(request if request is not None else gradio_request())
            elif param.name == "evt":
                args.append(gr.EventData(None, evt or {}))
            else:
                value = next(inputs, inspect.Parameter.empty)
                if value is inspect.Parameter.empty:
                    break
                args.append(value)
        out = fn.fn(*args)
        out = out if isinstance(out, tuple) else (out,)
        changed = set()
        for block, value in zip(fn.outputs, out, strict=True):
            if isinstance(value, dict) and value.get("__type__") == "update":
                if "value" not in value:
                    continue
                value = value["value"]
            if (isinstance(block, gr.State) or block is self.handle.search) \
                    and self.values[id(block)] != value:
                changed.add(id(block))
            self.values[id(block)] = value
        if id(self.handle.drawn) in changed or id(self.handle.view) in changed:
            self.run(self.ask(self.draw))
        if id(self.handle.search) in changed:
            for each in [self.ask(fn) for fn in self.on_search]:
                self.run(each)

    def act(self, name, **kw):
        self.run(self.ask(name), **kw)

    def type(self, text, *, answered=True):
        """The box's text changed in the page (a key typed, Backspace, a
        paste): its change listeners asked with the page as it is now, and
        answered unless `answered` is False (then returned, to run later)."""
        moved = self[self.handle.search] != text
        self[self.handle.search] = text
        asked = [self.ask(fn) for fn in self.on_search] if moved else []
        if answered:
            for each in asked:
                self.run(each)
        return asked

    def clear(self, *, answered=True):
        """Clear: a front-end function (no request of its own) whose return
        is the box's text, then the box's change, as `type`."""
        (fn,) = [fn for fn in self.demo.fns.values()
                 if (self.handle.search_clear._id, "click") in fn.targets]
        assert fn.fn is None and fn.outputs == [self.handle.search], fn
        assert fn.js == ml.CLEAR_SEARCH_JS == "() => ''"
        return self.type("", answered=answered)


def test_a_search_answered_after_its_clear_draws_the_box_as_it_is(root, serving, monkeypatch):
    """G07 (the review of 5171661, P2-1: "be" typed, Clear clicked, the
    search answered after Clear's: the list drew "be" under the empty box):
    the draw reads the search box as it is when it is asked, never the text a
    search was typed with, so a search answered once Clear has emptied the
    box draws the box as it is (Clear's own change, asked behind it, draws it
    again). The search kept is said alike by the count and the rows."""
    demo, _ctx, handle, _ = _component("research")
    page = _Page(demo, handle)
    page[handle.listing] = listing = _listing(root, monkeypatch)
    page.run(page.ask(page.draw))
    full = page[handle.count]
    late = page.type("disease", answered=False)  # its search asked, not answered yet
    behind = page.clear(answered=False)          # the box emptied in the page
    assert late and behind and page[handle.search] == ""
    for asked in late:                           # answered after Clear
        page.run(asked)
    assert page[handle.search] == ""
    assert page[handle.count] == full and "“disease”" not in page[handle.count]
    assert page[handle.rows] == ml.list_drawing(listing, None, ml.PRESETS["research"]).rows
    for asked in behind:
        page.run(asked)
    assert page[handle.count] == full
    # Typed and answered in order: the box's search drawn.
    page.type("disease")
    assert "“disease”" in page[handle.count]


def test_every_change_of_the_search_box_is_searched_by_one_path(root, serving, monkeypatch):
    """The adversarial check of 37912a9 (F2): Gradio 6.30's Textbox sends
    ``input`` from its keypress handler alone, so Backspace, Delete, a cut, a
    paste and an input method's composition never searched (the list kept
    drawing the old search under the box). The search is the box's
    ``change``, which Gradio sends for each new value, whatever made it, the
    last one asked kept ("always_last"), and its one listener: a page follows
    the search on the search token it writes. Clear is a front-end function
    that
    empties the box, with no request of its own, and the list's "Clear
    search" clicks Clear; no answer writes the box (a change it made would be
    searched once more). So each way the box changes drops the search's folds
    (`search_node`), keeps the operator's own and draws the box's text."""
    demo, _ctx, handle, other = _component("clinical", both=True)
    for each in (handle, other):
        on_box = [fn for fn in demo.fns.values()
                  if {(each.search._id, "input"), (each.search._id, "change")} & set(fn.targets)]
        (searched,) = [fn for fn in on_box if fn.name == "_searched"]
        assert searched.targets == [(each.search._id, "change")]
        assert searched.trigger_mode == "always_last"
        # The folds changed in place (`ListClock`), never an output.
        assert searched.inputs == [each.fold, each.clock]
        assert searched.outputs == [each.drawn, each.searched]
        # The box's one listener: two "always_last" listeners of one event set
        # each other off without end in Gradio 6.30 (a page follows the search
        # on `searched`; Models' wiring test checks it there).
        assert [fn.name for fn in demo.fns.values()
                if (each.search._id, "change") in fn.targets] == ["_searched"]
        (clear,) = [fn for fn in demo.fns.values()
                    if (each.search_clear._id, "click") in fn.targets]
        assert clear.fn is None and clear.outputs == [each.search]
        assert clear.js == ml.CLEAR_SEARCH_JS == "() => ''"
        assert [fn.name for fn in demo.fns.values() if fn.fn and each.search in fn.outputs] == []
    # "Clear search" is clicked as the list's Clear, in the page; the server
    # knows no such act (a page sending it changes nothing).
    assert "key === '__clear_search__'" in ml.MODEL_LIST_JS
    assert "list.id + '-search-clear'" in ml.MODEL_LIST_JS
    page = _Page(demo, handle)
    page[handle.listing] = listing = _listing(root, monkeypatch)
    page.run(page.ask(page.draw))
    full = page[handle.count]
    own, searched_fold = {"g:other": False}, ml.search_node(
        ml.run_node("gatv2/20261015-080000_gene_aaaaaa"))
    page[handle.fold] = {**own, searched_fold: False}
    page.act("_select", evt={"key": "__clear_search__"})
    assert page[handle.fold] == {**own, searched_fold: False} and page[handle.count] == full

    def drawn_for(query):
        return ml.list_drawing(listing, None, ml.PRESETS["clinical"], search=query,
                               fold=own).count

    # Typed; Backspace; select all and Delete; pasted; Clear: each drops the
    # search's folds (one folded while the search before it was shown) and
    # draws the box's text.
    for text, act in (("disease", page.type), ("diseas", page.type), ("", page.type),
                      ("disease", page.type), ("", lambda _text: page.clear())):
        page[handle.fold] = {**page[handle.fold], searched_fold: False}
        act(text)
        assert page[handle.search] == text
        assert page[handle.fold] == own, text
        assert page[handle.count] == drawn_for(text), text
    assert "“diseas”" in drawn_for("diseas")


def test_what_was_said_beside_load_stays_while_its_selection_is_drawn(
        root, serving, monkeypatch):
    """The review of 5171661 (P3-1 of the interaction review, Nit-2 of the
    correctness review): a refusal said beside Load when it was clicked is
    the session's (``said``), drawn by the list's one draw while the
    selection it was said for is drawn: a search, its Clear and "Show older
    runs" keep it (each draws the list again); a new pick, a refresh or a
    Load that opens its confirmation says the selection's own reason."""
    demo, _ctx, handle, _ = _component("clinical")
    page = _Page(demo, handle)
    operator_mode(monkeypatch)
    page[handle.workspace] = "demo"
    page.act("_refresh")
    listing = page[handle.listing]
    picked = _key("model-12.pt")
    page.act("_select", evt={"key": picked})
    assert page[handle.selected] == picked and page[handle.reason] == ""
    # Load asked from a host D8 refuses: the refusal is said beside it.
    page.act("_request", request=gradio_request(server=("192.0.2.10", 8264)))
    refusal = page[handle.reason]
    assert "shep-reason" in refusal and page[handle.plan] is None
    page.type("disease")
    assert "“disease”" in page[handle.count] and page[handle.reason] == refusal
    page.clear()
    assert page[handle.search] == "" and page[handle.reason] == refusal
    page.act("_select", evt={"key": "__older__"})
    assert page[handle.show_older] is True and page[handle.reason] == refusal
    # Another selection: its own reason; picked again: the refusal is not its.
    page.act("_select", evt={"key": _key("model-05.pt")})
    assert page[handle.reason] == ""
    page.act("_select", evt={"key": picked})
    assert page[handle.selected] == picked and page[handle.reason] == ""
    # Refused again, then the list read again: what was said was the list before.
    page.act("_request", request=gradio_request(server=("192.0.2.10", 8264)))
    assert page[handle.reason] == refusal
    page[handle.workspace] = listing["workspace"]
    page.act("_refresh")
    assert page[handle.selected] == picked and page[handle.reason] == ""
    assert page[handle.listing]["workspace"] == listing["workspace"]


def test_a_draw_token_is_new_every_time_whatever_the_clock_reads(root, serving, monkeypatch):
    """The review of 5171661 (P3-1 of the correctness review): Gradio fires a
    state's change only when its value differs at the end of an event, and a
    clock can read the same twice (Python 3.12 on Windows ticks
    ``time.monotonic`` every 15.6 ms), so a token from the clock could leave
    a change undrawn. The tokens are one count: with the clock held, two
    changes write two tokens."""
    demo, _ctx, handle, _ = _component("research")
    searched = next(fn for fn in demo.fns.values() if fn.name == "_searched")
    assert searched.outputs[0] is handle.drawn
    monkeypatch.setattr(time, "monotonic_ns", lambda: 1)
    monkeypatch.setattr(time, "monotonic", lambda: 1.0)
    first, second = searched.fn({})[0], searched.fn({})[0]
    assert first != second and 0 not in (first, second)
    assert ml.draw_token() != ml.draw_token()


def test_a_draw_that_fails_changes_nothing_drawn(root, serving, monkeypatch, caplog):
    """Nit-1 of the review of 5171661: Gradio 6.30 drops a deferred call
    after a failed one, so a draw that raised would leave a draw asked for
    meanwhile undone; the draw says the failure in the log and changes
    nothing (the list keeps its last drawing until the next change)."""
    demo, _ctx, handle, _ = _component("research")
    listing = _listing(root, monkeypatch)

    def broken(*_args, **_kwargs):
        raise RuntimeError("broken drawing")

    monkeypatch.setattr(ml, "list_drawing", broken)
    with caplog.at_level("ERROR", logger=ml.logger.name):
        drawn = _drawn(demo, handle, listing)
    assert all(value == gr.update() for value in drawn.values())
    assert "could not be drawn" in caplog.text


def test_the_research_load_is_off_for_a_base_or_refused_row_with_its_reason(
        root, serving, monkeypatch):
    """Research › Models lists every row; a base or a refused record selected
    there turns Load off, its reason beside it (in the selection's, the
    redraw's and the refresh's answers), before anything is asked."""
    from src.webui.shell import standalone_context

    with gr.Blocks() as demo:
        ctx = standalone_context(demo)
        handle = ctx.create_model_list("research")
    select = next(fn for fn in demo.fns.values() if fn.name == "_select")
    assert select.outputs[0] is handle.selected and handle.drawn in select.outputs
    listing = _listing(root, monkeypatch)

    def pick(name):
        out = select.fn(listing, None, False, None, "", {},
                        gr.EventData(None, {"key": _key(name)}), gradio_request())
        assert out[0] == _key(name)
        return _drawn(demo, handle, listing, out[0])

    for name, why in (("base.pt", ml.BASE_REFUSAL),
                      ("model-07.pt", f"Record not readable: {WRITTEN_BEFORE_SPEC}.")):
        drawn = pick(name)
        assert drawn[id(handle.load_button)]["interactive"] is False
        assert drawn[id(handle.reason)] == f'<div class="shep-reason shep-reason-grey">{why}</div>'
    drawn = pick("model-05.pt")
    assert drawn[id(handle.load_button)]["interactive"] is True
    assert drawn[id(handle.reason)] == ""
    operator_mode(monkeypatch)
    refreshed = handle.refresh["fn"]("demo", False, None, listing, _key("base.pt"),
                                     gradio_request())
    by = dict(zip(map(id, handle.refresh["outputs"]), refreshed, strict=True))
    drawn = _drawn(demo, handle, by[id(handle.listing)], by[id(handle.selected)])
    assert drawn[id(handle.load_button)]["interactive"] is False
    assert ml.BASE_REFUSAL in drawn[id(handle.reason)]


def test_a_view_shaped_for_another_workspace_never_hides_this_workspaces_rows(
        root, serving, monkeypatch):
    """Choosing another workspace draws the list (the shared refresh, then
    the one draw) with the page's view before the page shapes a new one. What
    a view names by key (a family view's rows, its groups and cells) was read
    from one workspace's listing (the view's ``workspace``; else the listing
    before the refresh), so on another workspace's listing it is left out and
    the view's filter, column and note apply: never "No checkpoint is listed
    in this workspace." over a workspace that holds checkpoints, nor another
    workspace's family titles. The same workspace's listing (the refresh after
    a Load) keeps the view whole."""
    from src.webui.shell import standalone_context

    with gr.Blocks() as demo:
        ctx = standalone_context(demo)
        handle = ctx.create_model_list("research")
    elsewhere = "gatv2/20260901-000000_base_ffffff/base.pt"
    family = {"task": "disease", "keys": [elsewhere],
              "groups": [{"title": "Family from run 2026-09-01 00:00 · 1 file",
                          "keys": [elsewhere]}],
              "column": "Lineage", "cells": {elsewhere: "New run"}, "note": "Reloads.",
              "workspace": "other"}
    operator_mode(monkeypatch)

    def refresh(view, previous):
        fold = {}
        out = handle.refresh["fn"]("demo", False, view, previous, None, gradio_request(), "",
                                   fold)
        by = dict(zip(map(id, handle.refresh["outputs"]), out, strict=True))
        return by, _drawn(demo, handle, by[id(handle.listing)], by[id(handle.selected)],
                          by[id(handle.show_older)], view, "", fold)

    by, drawn = refresh(family, {**ml.empty_listing(), "workspace": "other"})
    rows = drawn[id(handle.rows)]
    for name in ("model-12.pt", "model-08.pt", "model-05.pt", "model-01.pt"):
        assert f'data-key="{_key(name)}"' in rows, name
    assert "Family from run" not in rows and "Group" not in rows
    assert drawn[id(handle.count)] == "Disease: 3 of 8 runs"
    # Nothing is selected: no note beside Load (R-78).
    assert drawn[id(handle.note)] == ""
    # A view that names no workspace was read from the listing before the refresh.
    listing = by[id(handle.listing)]
    unnamed = {k: v for k, v in family.items() if k != "workspace"}
    assert "keys" not in ml.view_for_listing(unnamed, {"workspace": "other"}, listing)
    assert ml.view_for_listing(unnamed, listing, listing) is unnamed
    # The same workspace: the view's own rows and groups.
    mine = {**family, "workspace": "demo", "keys": [_key("model-05.pt")],
            "groups": [{"title": "Family from run 2026-10-01 00:00 · 1 file",
                        "keys": [_key("model-05.pt")]}]}
    rows = refresh(mine, listing)[1][id(handle.rows)]
    assert rows.count(" data-key=") == 1 and "Family from run 2026-10-01" in rows
    assert "Group" not in rows


def _row_of(html, name):
    """The row (a drawer's button, or a table's row) of the file `name`."""
    at = html.index(f'data-key="{_key(name)}"')
    start = max(html.rfind("<button", 0, at), html.rfind("<tr", 0, at))
    return html[start:html.index("</button>" if html[start:].startswith("<button") else "</tr>",
                                 at)]


def _run_row(html, run):
    """The tree's summary row of the run directory `run`."""
    at = html.index(f'data-fold="r:{run}"')
    start = html.rfind("<tr", 0, at)
    return html[start:html.index("</tr>", at)]


def _cells(row):
    """A table row's cells as the words they show (tags dropped)."""
    return [re.sub(r"<[^>]+>", "", cell) for cell in re.findall(r"<td[^>]*>(.*?)</td>", row)]


def _visible(html):
    """HTML as the words it shows (tags dropped, entities read)."""
    import html as html_

    return html_.unescape(re.sub(r"<[^>]+>", "", html))


def test_every_row_says_what_its_file_is_in_both_presets(root, serving, monkeypatch):
    """README §5 (map B7): what a file is is said once where a person reads
    its row: in its run's Task column (a run's files share their record,
    `renderers.record_kind_text`), in the drawer, which lists every task's
    models together, as in the research tree, its refused and unreadable
    files included, under any group."""
    listing = _listing(root, monkeypatch)
    rows = ml.list_html(listing, None, ml.PRESETS["clinical"], show_older=True)
    for run, kind in (("20261018-091200_disease_8c2e7a", "Disease"),
                      ("20261017-091500_disease_3fa9c1", "Disease"),
                      ("20261015-080000_gene_aaaaaa", "Gene"),
                      ("20261014-080000_patients_like_me_bbbbbb", "Patients-like-me")):
        assert _cells(_run_row(rows, f"gatv2/{run}"))[1] == kind, run
    view = {"groups": [{"title": "Group A", "keys": [_key("model-07.pt")]}]}
    table = ml.list_html(listing, None, ml.PRESETS["research"], view=view)
    for run, kind, operation in (
            ("gatv2/20261001-000000_base_aaaaaa", "Base", "New run"),
            ("hgt/20261012-080000_run_dddddd", "Record not readable", "Unknown"),
            ("sage/20261016-083000_run_e4c2a9", "Record unknown", "Unknown"),
            ("gatv2/20261015-080000_gene_aaaaaa", "Gene", "Fine-tune")):
        row = _run_row(table, run)
        assert _cells(row)[1] == kind and _cells(row)[2].startswith(operation), (run, row)


def test_a_frozen_backbone_row_says_so_in_both_presets(root, serving, monkeypatch):
    """Ops-wiring round, decision F1; D04's three freeze words: the
    frozen-backbone control's drawer row leads with "Disease · frozen graph
    backbone" (`renderers.model_kind_text`, from its record's weight
    transfer), its run in the research tree reads "Fine-tune, frozen"
    (`renderers.operation_words`); a fine-tune recording every part
    trainable "Fine-tune, full", other frozen parts "Fine-tune, partly
    frozen"."""
    listing = _listing(root, monkeypatch)
    frozen = {"trainable": ["patient_encoder", "task_head"],
              "parts": {"input": "frozen", "encoder": "frozen", "lp_head": "frozen",
                        "patient_encoder": "trainable", "task_head": "trainable"}}
    partly = {"parts": {"input": "frozen", "encoder": "trainable", "lp_head": "trainable",
                        "patient_encoder": "trainable", "task_head": "trainable"}}
    for item in listing["items"]:
        if item.get("filename") in ("model-12.pt", "model-08.pt"):
            item["weight_transfer"] = frozen
        elif item.get("filename") == "model-05.pt":
            item["weight_transfer"] = {"trainable": "all"}
        elif item.get("filename") == "model-01.pt":
            item["weight_transfer"] = partly
    rows = ml.list_html(listing, None, ml.PRESETS["clinical"], show_older=True)
    table = ml.list_html(listing, None, ml.PRESETS["research"])
    for run, words in (("20261018-091200_disease_8c2e7a", "Fine-tune, frozen"),
                       ("20261017-091500_disease_3fa9c1", "Fine-tune, full"),
                       ("20261013-080000_disease_cccccc", "Fine-tune, partly frozen")):
        for html in (rows, table):
            assert _cells(_run_row(html, f"gatv2/{run}"))[1:3] == ["Disease", words], run
    # Each word whole: a narrow frame may break the line at a space, never at a
    # word's own hyphen (D8).
    assert ('<span class="shep-op"><span class="shep-nowrap">Fine-tune,</span> '
            'partly frozen</span>') in table


def test_the_research_listing_carries_the_family_view_from_its_one_listing(
        root, serving, monkeypatch):
    """`listing["family"]`: per filter, the server's family view
    (`list_family(listing=)`) of the same listing the rows came from, so the
    files are listed once; parent first, the base kept as a context row of a
    task's view; a parent that is not available says why with no server path
    (a file's problem may name one)."""
    from src.api.services import server_lists
    from src.utils.checkpoint_io import write_checkpoint

    # A child whose recorded parent run holds a file that cannot be read, in a
    # family of its own (its record names its own run as the family's root).
    run = "20261012-070000_disease_eeeeee"
    orphan = root["workspace"] / "runs" / "gatv2" / run / "checkpoints" / "model-04.pt"
    orphan.parent.mkdir(parents=True)
    write_checkpoint({**payload("disease", 4, root["fingerprint"],
                                {"sha256": "9" * 64, "run": RUNS[2][1]}), "family_id": run},
                     orphan)
    os.utime(orphan, (time.time() - 10 * 3600,) * 2)
    calls = []
    real = server_lists.list_checkpoints
    monkeypatch.setattr(server_lists, "list_checkpoints",
                        lambda *a, **k: calls.append(a) or real(*a, **k))
    listing = ml.list_models("demo", operator_request(monkeypatch), family=True)
    assert len(calls) == 1
    family = listing["family"]
    assert set(family) == {"all", "base", "disease", "gene", "patients_like_me"}
    disease = family["disease"]
    assert [(row["key"], row["depth"], row["context"]) for row in disease] == [
        (_key("base.pt"), 0, True), *((_key(name), 1, False) for name in (
            "model-12.pt", "model-08.pt", "model-05.pt", "model-01.pt")),
        (f"gatv2/{run}/model-04.pt", 0, False)]
    assert all(row["parent"] == _key("base.pt") for row in disease[1:5])
    assert [(row["key"], row["group"]) for row in family["all"]][-2:] == [
        (_key("broken.pt"), "refused"), (_key("model-07.pt"), "refused")]
    assert [row["key"] for row in family["base"]] == [_key("base.pt")]
    assert [row["key"] for row in family["gene"]] == [_key("base.pt"), _key("model-03.pt")]
    lost = disease[-1]
    assert lost["parent_state"] == "unavailable" and "could not be read" in lost[
        "parent_problem"]
    assert str(root["root"]) not in json.dumps(family)
    # The drawer's listing reads no family view.
    assert "family" not in ml.list_models("demo", operator_request(monkeypatch))
    # A parent's problem that names a server path (an OSError's words, say) is
    # redacted as the list's own problems are.
    import dataclasses

    named = f"[Errno 13] Permission denied: '{root['workspace']}/runs/gatv2/x/last.pt'"
    real_family = server_lists.list_family
    monkeypatch.setattr(server_lists, "list_family", lambda *a, **k: server_lists.Listing(tuple(
        dataclasses.replace(row, parent_problem=named) for row in real_family(*a, **k).items)))
    listing = ml.list_models("demo", operator_request(monkeypatch), family=True)
    assert str(root["root"]) not in json.dumps(listing["family"])
    assert "Permission denied" in listing["family"]["disease"][-1]["parent_problem"]


# --------------------------------------------------------------------------- the wiring
def _component(preset="clinical", both=False):
    from src.webui.shell import standalone_context

    with gr.Blocks() as demo:
        ctx = standalone_context(demo)
        handle = ctx.create_model_list(preset)
        other = ctx.create_model_list("research") if both else None
    return demo, ctx, handle, other


def _named(demo, name, handle=None):
    found = [fn for fn in demo.fns.values() if fn.name == name
             and (handle is None or set(map(id, fn.outputs)) & {id(handle.reason),
                                                                 id(handle.phase)})]
    assert len(found) == 1, (name, len(found))
    return found[0]


def _after(demo, fn):
    nxt = [d for d in demo.fns.values() if d.trigger_after == fn._id]
    assert len(nxt) == 1, fn.name
    return nxt[0]


def test_load_and_reload_refresh_the_chips_and_reload_sets_the_trigger(root, serving, monkeypatch):
    from src.webui.components.chips import read_chips

    demo, ctx, handle, _ = _component()
    request = _named(demo, "_request")
    assert handle.phase in request.outputs and handle.plan in request.outputs
    chips = _after(demo, request)
    assert chips.fn is read_chips and chips.outputs == [ctx.chips]
    loading = _named(demo, "_loading")
    assert loading.outputs == [handle.phase, handle.reload_button, handle.keep_button]
    assert loading.queue is False
    assert '"shep-models-clinical-confirm"' in loading.js
    assert '"shep-models-clinical-reload"' in loading.js        # Reload off at once, in the page
    confirm = _after(demo, loading)
    assert confirm.name == "_confirm" and handle.reload_done is not None
    assert ctx.served_model_changed in confirm.outputs and handle.phase in confirm.outputs
    answered = _after(demo, confirm)                    # the page's step (`load_answered_js`)
    assert answered.fn is None and answered.js == ml.load_answered_js(handle.elem_id)
    chips = _after(demo, answered)
    assert chips.fn is read_chips
    assert _after(demo, chips).fn is handle.refresh["fn"]


def test_confirm_returns_the_phase_and_the_served_model_changed_token(root, serving, monkeypatch):
    """The confirmed Load is a plain ``def`` handler: Gradio runs it, and the
    reload's build, in a worker thread, never on the event loop (BACKLOG 28,
    checkpoint A's verification), as the API route runs its reload."""
    import inspect

    from src.api.routes import pipeline as pipeline_routes
    from src.api.routes.pipeline import PipelineReloadResponse, PipelineStatusResponse

    demo, ctx, handle, _ = _component()
    confirm = _named(demo, "_confirm")
    assert not inspect.iscoroutinefunction(confirm.fn)
    phase_at = confirm.outputs.index(handle.phase)
    token_at = confirm.outputs.index(ctx.served_model_changed)
    plan = _plan(root, monkeypatch)
    monkeypatch.setattr(pipeline_routes, "run_reload", lambda request: PipelineReloadResponse(
        success=True, message="ok", status=PipelineStatusResponse(
            **_status(root, served="model-12.pt"))))
    out = confirm.fn(plan, operator_request(monkeypatch))
    assert out[0] is None and out[phase_at] == "idle"
    assert out[token_at].startswith(root["shas"]["model-12.pt"] + "@")
    # shown again (what is served changed): still confirming, no token
    serving["value"] = _status(root)
    shown = _plan(root, monkeypatch)
    serving["value"] = _status(root, served="model-08.pt")
    out = confirm.fn(shown, operator_request(monkeypatch))
    assert out[phase_at] == "confirming" and out[token_at] == gr.update()
    # an unexpected error never leaves the page "loading"
    monkeypatch.setattr(ml, "finish_load", lambda *a, **k: (_ for _ in ()).throw(KeyError("x")))
    out = confirm.fn(plan, operator_request(monkeypatch))
    assert out[phase_at] == "idle" and "The Load stopped with an error." in out[2]


def test_while_a_reload_runs_the_confirmation_shows_only_its_status_and_close():
    """M5-3 (the mockup's footer after "Reload with …"): Reload and Keep were
    still drawn while the reload ran, and a second click could ask for a second
    reload. The step before the reload hides Reload and makes Keep "Close"
    (closing does not stop it); the page's script disables Reload at once and
    counts in this list's confirmation only: a page says the load from the
    server's status (B4: Diagnosis' card), never from this count."""
    phase, reload_, keep = ml.loading_buttons({"to": {"sha256": "a" * 64}})
    assert phase == "loading" and reload_["visible"] is False and keep["value"] == "Close"
    assert ml.loading_buttons(None) == ("idle", gr.update(), gr.update())
    script = ml.start_wait_js("x-confirm", "x-reload")
    assert 'getElementById("x-reload")' in script and "again.disabled = true" in script
    assert "data-shep-load-elapsed" not in script and "data-shep-load-card" not in script
    # A page's own outputs ride on the same step, before the reload holds the server.
    from src.webui.shell import standalone_context

    with gr.Blocks() as demo:
        ctx = standalone_context(demo)
        extra_in, extra_out = gr.State("s"), gr.HTML("")
        seen = []
        handle = ctx.create_model_list(
            "clinical", on_loading={"fn": lambda plan, s: seen.append((plan, s)) or ("line",),
                                    "inputs": [extra_in], "outputs": [extra_out]})
    loading = _named(demo, "_loading")
    assert loading.inputs == [handle.plan, extra_in]
    assert loading.outputs == [handle.phase, handle.reload_button, handle.keep_button, extra_out]
    plan = {"to": {"sha256": "b" * 64}}
    out = loading.fn(plan, "s")
    assert out[0] == "loading" and out[2]["value"] == "Close" and out[3] == "line"
    assert seen == [(plan, "s")]
    assert loading.fn(None, "s")[0] == "idle" and seen[-1] == (None, "s")


def test_a_close_while_the_reload_runs_is_done_in_the_page_and_sent_after_its_answer():
    """M5-3 review: while the reload ran, Close (and the drawer's ×) did
    nothing until the reload answered; then its close and
    the answer came back in no fixed order, and the box could be left open,
    empty, with "Reload with <the model now served>". While a reload runs the
    page closes it at once (`load_guard`) and clicks Close again only after
    the answer (`load_answered_js`), so the server handles it in order; the
    page's own hiding goes when that close is done (`load_closed_js`)."""
    assert ml.load_guard("x") == {"list": "x", "closers": ["#x-keep"], "hide": "#x-confirm-box",
                                  "then": "#x-keep", "openers": ["#x-load"]}
    drawer = {"hide": "#d", "then": "#d-x button", "also": ["#d-x button", "#d-cancel"],
              "openers": ["#open"]}
    assert ml.load_guard("x", drawer) == {"list": "x", "closers": ["#x-keep", "#d-x button",
                                                                   "#d-cancel"],
                                          "hide": "#d", "then": "#d-x button",
                                          "openers": ["#open"]}
    demo, _ctx, handle, _ = _component()
    assert handle.confirm.elem_id == "shep-models-clinical-confirm-box"
    loading = _named(demo, "_loading")
    from src.webui.components.help import as_json_for_script

    assert as_json_for_script(ml.load_guard("shep-models-clinical")) in loading.js
    for part in ("document.addEventListener('click'", "e.stopImmediatePropagation()",
                 "'shep-load-closed-' + id", "{ display: none !important; }",
                 ":not(#shep-load-x1):not(#shep-load-x2):not(#shep-load-x3)",
                 "g.openers.some(hit)"):
        assert part in loading.js, part
    # A close and an opener are both held from the page's own handlers.
    assert loading.js.count("e.preventDefault(); e.stopImmediatePropagation();") == 2
    answered = _after(demo, _after(demo, loading))
    assert answered.fn is None and "st.busy = false" in answered.js
    assert "document.querySelector(st.guard.then)" in answered.js and "el.click()" in answered.js
    assert '"shep-models-clinical"' in answered.js
    # Close's own chain ends with the page's step that drops its hiding.
    done = _after(demo, _after(demo, _named(demo, "_keep")))
    assert done.fn is None and done.js == ml.load_closed_js(handle.elem_id) == handle.after_close_js
    assert "!st.busy" in done.js and "css.remove()" in done.js
    # A page that puts the list in a drawer names it, and its own closes.
    from src.webui.shell import standalone_context

    with gr.Blocks() as page:
        ctx = standalone_context(page)
        listed = ctx.create_model_list("clinical", close_while_loading=drawer)
    script = _named(page, "_loading").js
    assert as_json_for_script(ml.load_guard(listed.elem_id, drawer)) in script


def test_keep_closes_the_confirmation(root, serving, monkeypatch):
    demo, _ctx, handle, _ = _component()
    keep = _named(demo, "_keep")
    out = dict(zip(map(id, keep.outputs), keep.fn(), strict=True))
    assert out[id(handle.plan)] is None and out[id(handle.confirm)]["visible"] is False
    assert out[id(handle.phase)] == "idle"


def test_keep_changes_nothing_inside_the_box_it_hides_and_resets_it_next():
    """A response that hides a column and updates a component inside it
    leaves detached elements and listeners in the browser on every cycle
    (Gradio 6.30, measured: about 113 elements and 2 listeners per
    Load-then-Keep). Keep hides the box; the step after it empties the box and
    puts Reload back, unless a Load has opened a confirmation meanwhile."""
    demo, _ctx, handle, _ = _component()
    keep = _named(demo, "_keep")
    inside = {id(handle.confirm_html), id(handle.reload_button), id(handle.keep_button)}
    assert id(handle.confirm) in set(map(id, keep.outputs))
    assert not inside & set(map(id, keep.outputs))
    after = _after(demo, keep)
    assert after.name == "_after_keep" and after.inputs == [handle.plan]
    assert after.outputs == [handle.confirm_html, handle.reload_button]
    html_, reload_ = after.fn(None)
    assert html_ == "" and reload_["visible"] is True
    assert after.fn({"to": {"sha256": "a" * 64}}) == (gr.update(), gr.update())


def test_both_presets_sit_in_one_page_each_counting_in_its_own_confirmation():
    demo, _ctx, clinical, research = _component(both=True)
    assert (clinical.elem_id, research.elem_id) == ("shep-models-clinical", "shep-models-research")
    ids = [getattr(b, "elem_id", None) for b in demo.blocks.values() if getattr(b, "elem_id", None)]
    assert len(ids) == len(set(ids))
    scripts = sorted(fn.js for fn in demo.fns.values() if fn.name == "_loading")
    assert len(scripts) == 2
    assert '"shep-models-clinical-confirm"' in scripts[0] + scripts[1]
    # Each names only its own list's confirmation and Reload button.
    import re

    for script in scripts:
        named = re.findall(r'"(shep-models-[a-z]+)-(confirm|reload)"', script)
        assert sorted(kind for _p, kind in named) == ["confirm", "reload"]
        assert len({prefix for prefix, _k in named}) == 1
    assert "document.querySelector('.shep-model-confirm')" not in "".join(scripts)


# --------------------------------------------------------------------------- through Gradio
def test_a_direct_call_through_gradios_api_meets_the_same_check(root, serving, monkeypatch):
    """The list's events, called as `gradio_client` would: refused off
    loopback, listed for an operator. No workspace is posted (the dropdown has
    no choices until the first read), so the server picks the serving one."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.webui.shell import standalone_context

    with gr.Blocks() as demo:
        ctx = standalone_context(demo)
        handle = ctx.create_model_list("clinical")
    app = FastAPI()
    gr.mount_gradio_app(app, demo, path="/ui")
    import uuid

    refresh = next(d for d in demo.fns.values() if d.name == "_refresh")
    draw = next(d for d in demo.fns.values() if d.name == "_draw")
    operator_mode(monkeypatch)

    def call(base_url):
        # One session: the refresh, then the draw it asks for (its token's change).
        session = uuid.uuid4().hex
        answers = []
        with TestClient(app, base_url=base_url) as client:
            for fn, data in ((refresh, [None, False, None, None, None, "", {}, None]),
                             (draw, [None] * len(draw.inputs))):
                started = client.post(f"/ui/gradio_api/call/{fn.api_name}",
                                      json={"data": data, "session_hash": session})
                assert started.status_code == 200, started.text
                stream = client.get(f"/ui/gradio_api/call/{fn.api_name}/"
                                    f"{started.json()['event_id']}")
                lines = stream.text.splitlines()
                answers.append(json.loads(lines[lines.index("event: complete") + 1]
                                          [len("data:"):]))
        return answers

    # The outputs a client sees: the listing itself is session state (not
    # returned: a State's place holds nothing), so the refresh answers the
    # workspace choice; the draw, the rows, the count, the facts, from that
    # session's listing.
    refused = call("http://192.0.2.10")
    assert "Operator controls only: request not local." in json.dumps(refused)
    assert "disease_8c2e7a" not in json.dumps(refused)
    (_listing, _selected, workspace, *_states), (rows, count, *_rest) = call(OPERATOR_BASE_URL)
    assert workspace["value"] == "demo"
    # The drawer opened the latest runs only (BACKLOG 39), so how many older
    # runs hold a model is not said.
    assert count == "Latest 4 runs, 5 saved files" and "listed" not in count
    # The run named once, by its row in the tree (B4), its file by what it is.
    assert "disease · GATv2 2026-10-18 09:12" in _visible(_run_row(
        rows, "gatv2/20261018-091200_disease_8c2e7a"))
    assert _cells(_row_of(rows, "model-12.pt"))[:4] == ["model-12.pt", "", "",
                                                         "recorded epoch 12"]
    assert handle.load_button is not None and handle.refresh["fn"] is refresh.fn


def test_the_list_shows_no_absolute_server_path(root, serving, monkeypatch, tmp_path):
    """M5-3 review SECURITY-3: a workspace that is a link out of the root was
    refused with its absolute target and the root's path, which the list
    printed escaped but not redacted (Datasets redacts the same reason)."""
    outside = tmp_path / "outside" / "elsewhere_ws"
    outside.mkdir(parents=True)
    (root["root"] / "leaving_ws").symlink_to(outside)
    listing = ml.list_models("leaving_ws", operator_request(monkeypatch))
    assert listing["workspace"] == "leaving_ws" and listing["items"] == []
    shown = (ml.list_html(listing, None, ml.PRESETS["clinical"])
             + ml.count_line(listing, ml.PRESETS["clinical"], False) + repr(listing["problem"]))
    assert listing["problem"] and "outside the workspaces root" in listing["problem"]
    assert str(tmp_path) not in shown and str(root["root"]) not in shown, shown


def test_a_selection_made_after_a_load_survives_the_lists_re_read(root, serving, monkeypatch):
    """M5-3 review UI-3: the re-read chained after every Load reset the list,
    dropping a row picked once the outcome was closed (its facts gone, Load
    disabled, nothing said). A selection still listed in the same workspace
    stays; one of another workspace, or no longer listed, does not."""
    from src.webui.shell import standalone_context

    with gr.Blocks() as demo:
        ctx = standalone_context(demo)
        handle = ctx.create_model_list("clinical")
    refresh = handle.refresh
    assert refresh["inputs"][3:5] == [handle.listing, handle.selected]
    operator_mode(monkeypatch)
    request = gradio_request()
    before = ml.list_models("demo", request)
    picked = _key("model-12.pt")
    out = refresh["fn"]("demo", False, None, before, picked, request)
    by = dict(zip(map(id, refresh["outputs"]), out, strict=True))
    assert by[id(handle.selected)] == picked
    drawn = _drawn(demo, handle, by[id(handle.listing)], picked)
    assert drawn[id(handle.load_button)]["interactive"] is True
    assert 'aria-current="true"' in _row_of(drawn[id(handle.rows)], "model-12.pt")
    assert "Select a model" not in drawn[id(handle.facts)]
    elsewhere = ({**before, "workspace": "other"}, picked)
    gone = (before, "gatv2/x/gone.pt")
    for previous, key in (elsewhere, gone):
        out = refresh["fn"]("demo", False, None, previous, key, request)
        by = dict(zip(map(id, refresh["outputs"]), out, strict=True))
        assert by[id(handle.selected)] is None
        drawn = _drawn(demo, handle, by[id(handle.listing)], None)
        assert drawn[id(handle.load_button)]["interactive"] is False


# --------------------------------------------------------------------------- the tree (D04)
def _rows(html):
    """Every row of the tree: its keys and whether it is folded away."""
    return [(m.group(1), m.group(2), " hidden" in m.group(0))
            for m in re.finditer(r'<tr class="([^"]*)"(?: data-key="([^"]*)")?[^>]*>', html)]


def test_the_tree_folds_runs_and_unfolds_the_selected_and_the_served(root, serving, monkeypatch):
    """D04: every run a summary row, its files folded; the run of the model in
    use and of the selection unfolded (each file still its own row); what the
    operator folds or unfolds is kept (the fold state) and wins over the
    default; a group has no toggle of its own (its title was By family's
    header row, retired with the family table: the stage-lineage round), so a
    group entry left in the fold state hides nothing; folding never changes
    what is selected, and nothing is dropped from the table."""
    listing = _listing(root, monkeypatch)
    research = ml.PRESETS["research"]
    html = ml.list_html(listing, None, research)
    files = {key: hidden for cls, key, hidden in _rows(html) if key}
    assert len(files) == 9                                     # nothing dropped
    served_run = {_key("model-05.pt")}
    assert {k for k, hidden in files.items() if not hidden} == served_run
    # A selection unfolds its run too, and is the row located.
    picked = _key("model-03.pt")
    html = ml.list_html(listing, picked, research)
    files = {key: hidden for cls, key, hidden in _rows(html) if key}
    assert files[picked] is False and files[_key("model-05.pt")] is False
    assert f'data-locate="{picked}"' in html
    assert 'aria-current="true"' in _row_of_key(html, picked)
    # The operator's own folds win: the selection's run folded by hand stays
    # folded, another run unfolded stays open; the selection is unchanged.
    fold = {ml.run_node("gatv2/20261015-080000_gene_aaaaaa"): False,
            ml.run_node("gatv2/20261014-080000_patients_like_me_bbbbbb"): True}
    html = ml.list_html(listing, picked, research, fold=fold)
    files = {key: hidden for cls, key, hidden in _rows(html) if key}
    assert files[picked] is True and files[_key("model-02.pt")] is False
    assert 'aria-current="true"' in _row_of_key(html, picked)
    assert 'aria-expanded="false"' in html and "▸" in html
    # A group draws no toggle; an old group entry hides none of its runs.
    view = {"groups": [{"node": "g:fam", "title": "Family A",
                        "keys": [_key("model-12.pt"), _key("model-08.pt")]}]}
    html = ml.list_html(listing, None, research, view=view, fold={"g:fam": False})
    body = re.search(r'<tbody data-node="g:fam">(.*?)</tbody>', html, re.S).group(1)
    assert 'data-fold="g:fam"' not in html and "Family A" not in html
    assert [hidden for cls, _key_, hidden in _rows(body) if "shep-tree-run" in cls] == [False]
    assert ml.count_line(listing, research, False, view, fold={"g:fam": False}) == (
        ml.count_line(listing, research, False, view))           # folding hides, counts nothing


def test_a_search_finds_names_and_fingerprints_and_keeps_where_they_come_from(
        root, serving, monkeypatch):
    """D04's search: plain text, ignoring case, in a run's name or words or a
    file's name; a model fingerprint by its first hex digits (four or more) or
    whole; a match keeps the runs it comes from (the view's exact parent
    files) as context, which the family graph draws quieter and says so; a file
    found by its own name or fingerprint has its run unfolded; nothing matched
    says so; the selection never changes."""
    listing = _listing(root, monkeypatch)
    research = ml.PRESETS["research"]
    sha = root["shas"]["model-03.pt"]
    family = [_key(n) for n in ("base.pt", "model-12.pt", "model-08.pt", "model-05.pt",
                                "model-01.pt", "model-03.pt", "model-02.pt")]
    parents = {key: _key("base.pt") for key in family[1:]}
    view = {"keys": family, "parents": parents,
            "groups": [{"node": "g:base", "title": "Family from base", "keys": family,
                        "family": True}]}
    by_name = ml.tree_view(listing, view, research, search="GENE")
    assert by_name.matched == (_key("model-03.pt"),) and by_name.context == (_key("base.pt"),)
    assert ml.count_line(listing, research, False, view, search="GENE") == (
        "1 run matches “GENE”, with 1 run they come from")
    for query in (sha[:4].upper(), sha[:12], sha):
        tree = ml.tree_view(listing, view, research, search=query)
        assert tree.matched == (_key("model-03.pt"),), query
        assert ml.run_node("gatv2/20261015-080000_gene_aaaaaa") in tree.open_nodes
    assert ml.tree_view(listing, view, research, search=sha[:3]).matched == ()   # too short
    assert ml.search_hit({"filename": "model-03.pt", "run": "x"}, "MODEL-03") == "file"
    html = ml.list_html(listing, _key("model-12.pt"), research, view=view, search="gene")
    assert "Selected model not shown by this filter (search “gene”)" in html
    assert 'data-key="__clear_search__"' in html
    # The family graph (the same view with its edges) draws the run they come
    # from as context: quieter, in words.
    gene = "gatv2/20261015-080000_gene_aaaaaa"
    graph = ml.list_html(listing, None, research, search="gene",
                         view={**view, "edges": [{"from": _key("base.pt"), "to": gene}]})
    base = graph[graph.index('data-run="gatv2/20261001-000000_base_aaaaaa"'):]
    base = base[:base.index("</button>")]
    assert "shep-fam-node shep-ctx" in base and ml.r.status_mark(
        "neutral", "Parent of a shown run") in base
    assert f'data-run="{gene}"' in graph and "shep-ctx" not in graph[
        graph.index(f'<li class="shep-fam-li" data-run="{gene}"'):].split("</button>")[0]
    nothing = ml.list_html(listing, None, research, view=view, search="zzz")
    assert "No checkpoint matches “zzz” in this workspace." in nothing
    assert ml.count_line(listing, research, False, view, search="zzz") == "No run matches “zzz”"


def test_the_refresh_keeps_the_selection_and_the_folds_and_a_lone_row_is_chosen(
        root, serving, monkeypatch):
    """D04: a refresh keeps a selection still listed in the same workspace,
    whatever the filter or search show, and what was folded; another
    workspace starts afresh. R-27: a lone candidate is chosen, so Load is no
    dead end."""
    from src.webui.shell import standalone_context

    with gr.Blocks() as demo:
        ctx = standalone_context(demo)
        handle = ctx.create_model_list("research")
    operator_mode(monkeypatch)
    request = gradio_request()
    before = ml.list_models("demo", request)
    picked = _key("model-03.pt")
    fold = {ml.run_node("gatv2/20261013-080000_disease_cccccc"): True}
    # The session's one fold dict: a refresh changes it in place, never returns it.
    session_fold = dict(fold)
    assert handle.fold not in handle.refresh["outputs"]
    out = handle.refresh["fn"]("demo", False, {"task": "disease"}, before, picked, request,
                               "nothing matches this", session_fold)
    by = dict(zip(map(id, handle.refresh["outputs"]), out, strict=True))
    assert by[id(handle.selected)] == picked and session_fold == fold
    drawn = _drawn(demo, handle, by[id(handle.listing)], picked, False, {"task": "disease"},
                   "nothing matches this", session_fold)
    assert "Selected model not shown by this filter (Task: Disease" in drawn[id(handle.rows)]
    out = handle.refresh["fn"]("demo", False, None, {**before, "workspace": "other"}, picked,
                               request, "", session_fold)
    by = dict(zip(map(id, handle.refresh["outputs"]), out, strict=True))
    assert by[id(handle.selected)] is None and session_fold == {}
    # A lone candidate (the drawer's one task model) is chosen.
    lone = {**before, "items": [i for i in before["items"] if i["filename"] == "model-03.pt"]}
    assert ml.kept_selection(None, lone, None, None, ml.PRESETS["clinical"]) == picked
    assert ml.kept_selection(None, before, None, None, ml.PRESETS["clinical"]) is None
    assert ml.kept_selection(None, {**before, "items": [ml.find_item(before, _key(
        "base.pt"))]}, None, None, ml.PRESETS["research"]) is None    # a base: never loaded


def test_a_fold_toggle_is_kept_and_draws_nothing_again(root, serving, monkeypatch):
    """A fold toggle is folded in the page at once (`MODEL_LIST_JS`); the
    server only keeps it, for the next drawing."""
    from src.webui.shell import standalone_context

    with gr.Blocks() as demo:
        ctx = standalone_context(demo)
        handle = ctx.create_model_list("research")
    select = next(fn for fn in demo.fns.values() if fn.name == "_select")
    assert select.trigger_mode == "multiple"
    listing = _listing(root, monkeypatch)
    node = ml.run_node("gatv2/20261015-080000_gene_aaaaaa")
    fold = {}
    out = select.fn(listing, None, False, None, "", fold, gr.EventData(None, {"fold": node,
                                                                            "open": True}),
                    gradio_request())
    by = dict(zip(map(id, select.outputs), out, strict=True))
    # Kept in the session's one fold dict, in place (`ListClock`), never an output.
    assert fold == {node: True} and handle.fold not in select.outputs
    # Nothing else changed, and no draw asked for (the draw token unchanged).
    assert handle.drawn in select.outputs and handle.rows not in select.outputs
    assert all(v == gr.update() for v in by.values())
    assert "foldRows(node, open)" in ml.MODEL_LIST_JS and "scrollIntoView" in ml.MODEL_LIST_JS


def test_the_tree_escapes_and_names_entries_it_could_not_list(root, serving, monkeypatch):
    """Text from a run's name is escaped in its row; a plain table's selected
    row is `aria-current`, never `aria-selected` (the skill's §8); a file's
    Fingerprint cell is its 12 hex by the one digest value, with its dataset
    only where it is not the workspace's current one (BACKLOG 69 L1:) an
    entry the listing could not list reads its own name and where it sits,
    never an upper-cased directory as an architecture."""
    listing = _listing(root, monkeypatch)
    research = ml.PRESETS["research"]
    item = ml.find_item(listing, _key("model-03.pt"))
    marked = {**item, "run": "20261015-080000_<b>x_aaaaaa", "key": "gatv2/m/model-03.pt",
              "run_key": "gatv2/m", "trained_on_this_dataset": False,
              "dataset_fingerprint": "f" * 64}
    unlisted = {"workspace": "demo", "architecture": "runs", "run": "bad\\xff",
                "filename": "bad\\xff", "key": "/626164ff", "run_key": "/626164ff",
                "sha256": None, "problem": "runs/bad\\xff cannot be listed (…)", "record": None}
    data = {**listing, "items": [marked, unlisted, *listing["items"]]}
    html = ml.list_html(data, "gatv2/m/model-03.pt", research)
    assert "<b>x" not in html and "&lt;b&gt;x" in html and "aria-selected" not in html
    assert "Another dataset" in _row_of_key(html, "gatv2/m/model-03.pt")
    cells = re.findall(r"<td(?: class=\"[^\"]*\")?>(.*?)</td>", _row_of_key(html, _key("model-03.pt")),
                       re.S)
    assert cells[-1] == r_fingerprint(item["sha256"], listing)
    (entry,) = _entry_rows(html)
    assert "bad\\xff" in entry and "a folder in runs/" in entry and "RUNS" not in entry
    assert "runs/bad\\xff cannot be listed (…)" in entry          # why, on the page
    # Counted apart, never as a run or a saved file (B69 L3), a filter's
    # "of N runs" too (it read "Disease: 3 of 10 runs"), by its state (V-22).
    assert ml.count_line(data, research, False) == "9 runs, 10 saved files, 1 entry not read"
    assert ml.count_line(data, research, False, {"task": "disease"}) == "Disease: 3 of 9 runs"
    # A file whose metadata cannot be read: named once, in its Model cell; its
    # state and reason in words on the page, the loader's reference left to
    # its details; at most one " · ", nothing said only on hover (the skill's
    # §2, §5, §6). It read "broken.pt | broken.pt · metadata not readable ·
    # cannot be loaded", its reason in a title only.
    broken = _row_of_key(html, _key("broken.pt"))
    words = _visible(broken)
    assert words.count("broken.pt") == 1, words
    assert ("Metadata not readable: the file cannot be read as a checkpoint; cannot be loaded"
            in words), words
    assert words.count(" · ") <= 1 and "title=" not in broken, broken


def _entry_rows(html):
    """The rows of entries the listing could not list (no key: no choice)."""
    return re.findall(r'<tr class="shep-tree-entry[^"]*"[^>]*>.*?</tr>', html, re.S)


def test_the_count_line_says_an_unlisted_entry_in_its_rows_words(root, serving, monkeypatch):
    """V-22 (the verification of fcc3dea, part RB2): beside a run folder whose
    name is not valid text, Models' count line read "3 runs, 7 saved files, 1
    entry not listed" while that entry's row was drawn in the list, saying it
    "cannot be listed …, so no file in it is read". The count says the
    entry's state as its row does: never opened, "not read" (the one word
    for that state, `renderers.unreadable_words`), counted once, apart from
    the runs and files; the drawer, which lists task models only, draws no
    such row and counts none. Fails on 78444a6.

    The review of 6a14d6b (R3-1): an alias (runs/alias -> gatv2, B69-L4) is
    such an entry too, and its row says the opposite of "not read": its runs
    were read and are listed under gatv2 once, "so it is not listed again".
    Each entry is counted by the state its own row says, in that state's own
    words (the skill's §6: missing states keep their own words), never
    merged: it read "2 entries not read". Fails on 6a14d6b."""
    from src.webui.components import renderers as r

    runs = root["workspace"] / "runs" / "gatv2"
    (runs / os.fsdecode(b"bad\xff")).mkdir()
    listing = ml.list_models("demo", operator_request(monkeypatch))
    research, clinical = ml.PRESETS["research"], ml.PRESETS["clinical"]
    count = ml.count_line(listing, research, False)
    assert count == "8 runs, 9 saved files, 1 entry not read", count
    (item,) = [i for i in listing["items"] if r.entry_place(i)]
    assert r.unreadable_words(item) == "not read"
    (row,) = _entry_rows(ml.list_html(listing, None, research))
    assert "so no file in it is read" in _visible(row)
    assert _entry_rows(ml.list_html(listing, None, clinical, show_older=True)) == []
    assert "entry" not in ml.count_line(listing, clinical, True)
    # An alias beside it: its runs are the runs counted on the same line.
    os.symlink("gatv2", runs.parent / "alias")
    listing = ml.list_models("demo", operator_request(monkeypatch))
    count = ml.count_line(listing, research, False)
    assert count == "8 runs, 9 saved files, 1 entry not read, 1 entry not listed again", count
    rows = [_visible(row) for row in _entry_rows(ml.list_html(listing, None, research))]
    states = {i["run"]: r.unreadable_words(i) for i in listing["items"] if r.entry_place(i)}
    assert states == {"bad\\xff": "not read", "alias": "not listed again"}, states
    for name, state in states.items():
        (words,) = [w for w in rows if w.startswith(name)]
        # Its row and the count say its state alike.
        assert f"1 entry {state}" in count and (
            "so no file in it is read" if state == "not read" else f"so it is {state}") in words
    assert "entry" not in ml.count_line(listing, clinical, True)


def _row_of_key(html, key):
    at = html.index(f'data-key="{key}"')
    start = html.rfind("<tr", 0, at)
    return html[start:html.index("</tr>", at)]


def r_fingerprint(sha, listing):
    from src.webui.components import renderers as r

    return r.fingerprint_value(sha, what="model fingerprint", copy=False,
                               digits=r.distinguishing_digits(
                                   i.get("sha256") for i in listing["items"]))


def test_the_drawer_opens_only_the_runs_it_shows_and_the_rest_on_show_older_runs(
        root, serving, monkeypatch):
    """BACKLOG 39 (M5-3 review UI-2): the drawer read every checkpoint of the
    workspace to show four runs. Its first listing opens the latest runs only
    (a base run read on the way is not shown), the model in use's included,
    and "Show older runs" opens the rest."""
    import src.utils.checkpoint_io as checkpoint_io
    from src.webui.shell import standalone_context

    with gr.Blocks() as demo:
        ctx = standalone_context(demo)
        handle = ctx.create_model_list("clinical")
    operator_mode(monkeypatch)
    opened = []
    original = checkpoint_io.read_checkpoint
    monkeypatch.setattr(checkpoint_io, "read_checkpoint",
                        lambda path, **kw: opened.append(Path(path).name) or original(path, **kw))
    out = handle.refresh["fn"]("demo", False, None, None, None, gradio_request())
    by = dict(zip(map(id, handle.refresh["outputs"]), out, strict=True))
    first = sorted(opened)
    assert first == ["broken.pt", "model-02.pt", "model-03.pt", "model-05.pt", "model-08.pt",
                     "model-12.pt"], first
    listing = by[id(handle.listing)]
    assert listing["runs_unread"] == 3
    drawn = _drawn(demo, handle, listing)
    assert "Show older runs (up to 3 more)" in drawn[id(handle.rows)]
    assert drawn[id(handle.count)] == "Latest 4 runs, 5 saved files"
    select = next(fn for fn in demo.fns.values() if fn.name == "_select")
    opened.clear()
    out = select.fn(listing, None, False, None, "", {}, gr.EventData(None, {"key": "__older__"}),
                    gradio_request())
    shown = dict(zip(map(id, select.outputs), out, strict=True))
    assert {"model-01.pt", "base.pt", "model-07.pt"} <= set(opened)
    older = shown[id(handle.listing)]
    assert older["runs_unread"] == 0 and shown[id(handle.show_older)] is True
    assert "disease_cccccc" in _drawn(demo, handle, older, None, True)[id(handle.rows)]
    # Another workspace picked after "Show older runs" lists its latest runs
    # only (its every file was opened: "Show older runs" was kept for the
    # session); the same workspace read again keeps what was asked.
    shutil.copytree(root["workspace"], root["root"] / "other")
    opened.clear()
    out = handle.refresh["fn"]("other", True, None, older, None, gradio_request())
    by = dict(zip(map(id, handle.refresh["outputs"]), out, strict=True))
    assert sorted(opened) == first, sorted(opened)
    assert by[id(handle.listing)]["workspace"] == "other" and by[id(handle.show_older)] is False
    assert "Show older runs (up to 3 more)" in _drawn(demo, handle, by[id(handle.listing)])[
        id(handle.rows)]
    out = handle.refresh["fn"]("demo", True, None, older, None, gradio_request())
    by = dict(zip(map(id, handle.refresh["outputs"]), out, strict=True))
    assert by[id(handle.show_older)] is True and by[id(handle.listing)]["runs_unread"] == 0


def test_a_drawer_search_says_the_older_runs_it_has_not_searched(root, serving, monkeypatch):
    """The drawer's first listing reads its latest runs only (BACKLOG 39), so
    its search and its Task filter see those runs only: a query that matches
    an older run not read yet is never "no match in this workspace" (it was,
    with no way to the run). The count and the empty list say the latest runs
    were searched, "Show older runs" stays under them, and once clicked the
    search finds the older run."""
    from src.webui.shell import standalone_context

    with gr.Blocks() as demo:
        ctx = standalone_context(demo)
        handle = ctx.create_model_list("clinical")
    operator_mode(monkeypatch)
    clinical = ml.PRESETS["clinical"]
    listing = ml.list_models("demo", gradio_request(), latest_runs=clinical.latest_runs)
    assert listing["runs_unread"] == 3
    html = ml.list_html(listing, None, clinical, search="cccccc")
    assert "in this workspace" not in html
    assert ("No model finetuned for a task matches “cccccc” in the latest runs. Show older runs "
            "searches the rest.") in _visible(html)
    assert "Show older runs (up to 3 more)" in html
    assert ml.count_line(listing, clinical, False, search="cccccc") == (
        "No run matches “cccccc” in the latest runs")
    assert ml.count_line(listing, clinical, False, search="disease") == (
        "2 runs match “disease” in the latest runs")
    assert ml.count_line(listing, clinical, False, {"task": "disease"}) == (
        "Disease: 2 of the latest 4 runs")
    no_plm = {**listing, "items": [i for i in listing["items"] if "patients" not in i["run"]]}
    empty = ml.list_html(no_plm, None, clinical, view={"task": "patients_like_me"})
    assert ("No patients-like-me model is listed in the latest runs. Show older runs reads the "
            "rest.") in _visible(empty) and "Show older runs (up to 3 more)" in empty
    # Show older runs reads the rest, and the search finds the older run.
    select = next(fn for fn in demo.fns.values() if fn.name == "_select")
    out = select.fn(listing, None, False, None, "cccccc", {},
                    gr.EventData(None, {"key": "__older__"}), gradio_request())
    shown = dict(zip(map(id, select.outputs), out, strict=True))
    drawn = _drawn(demo, handle, shown[id(handle.listing)], None, shown[id(handle.show_older)],
                   None, "cccccc")
    assert "disease_cccccc" in drawn[id(handle.rows)]
    assert drawn[id(handle.count)] == "1 run matches “cccccc”"
    # Every run read: the whole workspace's words.
    full = ml.list_models("demo", gradio_request())
    assert "No model finetuned for a task matches “zzz” in this workspace." in _visible(
        ml.list_html(full, None, clinical, search="zzz"))


def test_a_search_reveals_what_it_found_and_clearing_it_brings_the_folds_back(
        root, serving, monkeypatch):
    """G7 (the stage-lineage round; it was "a search unfolds what it found",
    which dropped the operator's own folds for good): a match is never folded
    away while the search shows the list, a run holding a file found by its
    own name or fingerprint revealed as the list is drawn, and the fold state
    is not changed by it; what the operator folds while the search is shown
    is that search's (``q:`` entries, `search_node`), so clearing the search
    (or another search) drops it and brings the operator's own folds back.
    The drawer's tree and the family graph read the one rule (`fold_open`)."""
    from src.webui.shell import standalone_context

    with gr.Blocks() as demo:
        ctx = standalone_context(demo)
        handle = ctx.create_model_list("research")
    fns = {fn.name: fn for fn in demo.fns.values()}
    searched, select = fns["_searched"], fns["_select"]
    listing = _listing(root, monkeypatch)
    run = ml.run_node("gatv2/20261015-080000_gene_aaaaaa")
    manual = {run: False, "g:other": False}

    def hidden(rows):
        return {k: h for _c, k, h in _rows(rows) if k}[_key("model-03.pt")]

    query = root["shas"]["model-03.pt"][:8]
    fold = {**manual, ml.search_node(run): False}
    searched.fn(fold)
    assert fold == manual                      # the operator's folds kept; an old search's dropped
    assert searched.outputs == [handle.drawn, handle.searched]
    drawn = _drawn(demo, handle, listing, None, False, None, query, fold)
    assert not hidden(drawn[id(handle.rows)])  # revealed while the search shows the list
    assert drawn[id(handle.count)].startswith("1 run matches")
    # Folded again while the search is shown: that search's, the manual entry untouched.
    select.fn(listing, None, False, None, query, fold,
              gr.EventData(None, {"fold": run, "open": False, "searched": True}),
              gradio_request())
    during = dict(fold)
    assert during == {**manual, ml.search_node(run): False}
    assert hidden(ml.list_html(listing, None, ml.PRESETS["research"], search=query, fold=during))
    # Cleared (Clear empties the box in the page, and its change is the
    # search's): the operator's own folds are back, the search's are gone.
    (clear,) = [fn for fn in demo.fns.values() if (handle.search_clear._id, "click") in fn.targets]
    assert clear.fn is None and clear.outputs == [handle.search]
    assert searched.targets == [(handle.search._id, "change")]
    fold = dict(during)
    searched.fn(fold)
    assert fold == manual
    assert hidden(_drawn(demo, handle, listing, None, False, None, "", fold)[id(handle.rows)])
    assert ml.fold_open(manual, run, "", False, True) is False
    assert ml.fold_open(manual, run, query, True, False) is True


def _selected_item(run, filename, completed, figure, top):
    """A listed task-model file as the server lists it (the T08 flow's
    b-ft-full-disease and b-ft-frozen-disease runs): its progress, whether a
    record of its run names it among the best-k, its Trainer validation
    figure and the run's recorded selection."""
    digest = hashlib.sha256(f"{run}/{filename}".encode()).hexdigest()
    return {"workspace": "demo", "architecture": "gatv2", "run": run, "filename": filename,
            "sha256": digest, "checkpoint_kind": "training_state", "epoch": completed - 1,
            "completed_epochs": completed, "in_best_k": top,
            "validation": {"metric": "val_pooled_mrr", "value": figure},
            "selection": {"monitor": "val_pooled_mrr", "mode": "max"}, "problem": None,
            "record": "model", "learning_stage": "finetune", "task": "disease",
            "operation": "finetune", "key": f"gatv2/{run}/{filename}",
            "run_key": f"gatv2/{run}"}


def test_the_tree_names_tied_best_and_top_k_saves_as_its_selects_do():
    """T08-flow-4 (the WebUI skill §8): the tree's rows say what each file is
    by the words its Run → Checkpoint option leads with, one rule
    (`_checkpoint_entries`): the two files tied at the run's best recorded
    pooled MRR both read "Best pooled MRR" and two top-k saves "Top-k save",
    their Epochs telling them apart; a file's name only where two files would
    read exactly alike (what and how far they trained). The T08 flow saw
    "model-01-0.8400.pt" and "model-00-0.8400.pt" where Test's and Fine-tune's
    selects said "Best pooled MRR". A search that shows one file of a run
    never makes it its run's best (it was named among the files shown)."""
    full, frozen, twins = ("20261117-071255_b-ft-full-disease_263055",
                           "20261117-071330_b-ft-frozen-disease_cf68f3",
                           "20261117-071400_b-ft-twins-disease_aaaaaa")
    items = [_selected_item(full, "last.pt", 3, 0.39, False),
             _selected_item(full, "model-02-0.3900.pt", 3, 0.39, True),
             _selected_item(full, "model-01-0.8400.pt", 2, 0.84, True),
             _selected_item(full, "model-00-0.8400.pt", 1, 0.84, True),
             _selected_item(frozen, "last.pt", 3, 0.27, False),
             _selected_item(frozen, "model-02-0.2700.pt", 3, 0.27, True),
             _selected_item(frozen, "model-01-0.3333.pt", 2, 0.3333, True),
             _selected_item(frozen, "model-00-0.4467.pt", 1, 0.4467, True),
             _selected_item(twins, "last.pt", 4, 0.9, False),
             _selected_item(twins, "model-03-0.1.pt", 4, 0.1, True),
             _selected_item(twins, "model-03-0.2.pt", 4, 0.2, True)]
    listing = {"items": items, "workspaces": []}
    expected = {
        (full, "last.pt"): ["Latest save", "3"],
        (full, "model-02-0.3900.pt"): ["Top-k save", "3"],
        (full, "model-01-0.8400.pt"): ["Best pooled MRR", "2"],
        (full, "model-00-0.8400.pt"): ["Best pooled MRR", "1"],
        (frozen, "last.pt"): ["Latest save", "3"],
        (frozen, "model-02-0.2700.pt"): ["Top-k save", "3"],
        (frozen, "model-01-0.3333.pt"): ["Top-k save", "2"],
        (frozen, "model-00-0.4467.pt"): ["Best pooled MRR", "1"],
        (twins, "last.pt"): ["Latest save", "4"],
        # Alike in what they are and how far they trained: their names.
        (twins, "model-03-0.1.pt"): ["model-03-0.1.pt", "4"],
        (twins, "model-03-0.2.pt"): ["model-03-0.2.pt", "4"]}
    for preset in ("research", "clinical"):
        html = ml.list_html(listing, None, ml.PRESETS[preset], show_older=True)
        for (run, name), (what, epochs) in expected.items():
            cells = [_visible(cell).strip()
                     for cell in _cells(_row_of_key(html, f"gatv2/{run}/{name}"))]
            assert [cells[0], cells[3]] == [what, epochs], (preset, run, name, cells)
        # The selects' options lead with the same words (one rule).
        for run in (full, frozen, twins):
            parents = {item["key"]: item for item in items if item["run"] == run}
            for label, key in ml.checkpoint_choices(parents, list(parents)):
                run_, name = key.split("/")[1:]
                assert label.startswith(expected[(run_, name)][0] + " · "), (label, key)
        # A search shows one file of a run: still called what it is.
        found = ml.list_html(listing, None, ml.PRESETS[preset], show_older=True,
                             search="model-02-0.3900")
        assert [_visible(c).strip() for c in _cells(
            _row_of_key(found, f"gatv2/{full}/model-02-0.3900.pt"))][0] == "Top-k save"
        assert found.count('class="shep-tree-file') == 1


# --------------------------------------------------------------------------- the list's clock
# The review of dcaed68 (SL-R3): a handler wrote session state computed from
# what it read as it began, over what another handler committed meanwhile. A
# handler that waits inside another is held by patching what it reads
# (`list_models`), so what commits meanwhile is run in order, no thread, no
# timed wait.
def _session(preset="research"):
    demo, _ctx, handle, _ = _component(preset)
    return _Page(demo, handle), handle


def test_the_lists_clock_is_one_object_per_session_and_no_handler_returns_it(
        root, serving, monkeypatch):
    """Gradio 6.30 gives each session a deep copy of a State's first value and
    passes that object itself to every handler that reads the State; one no
    handler returns is the session's one object. The clock deep-copies into
    a new clock (its own lock, nothing committed), and neither it nor the
    folds (changed in place under it) is any handler's output; every list
    handler that reads the list's state reads the clock."""
    demo, _ctx, handle, other = _component("clinical", both=True)
    for each in (handle, other):
        assert isinstance(each.clock, gr.State) and isinstance(each.clock.value, ml.ListClock)
        for fn in demo.fns.values():
            assert each.clock not in fn.outputs and each.fold not in fn.outputs, fn.name
        for name in ("_refresh", "_select", "_searched", "_draw"):
            fns = [f for f in demo.fns.values() if f.name == name and each.fold in f.inputs]
            assert fns and all(f.inputs[-1] is each.clock for f in fns), name
    assert handle.clock is not other.clock
    clock = handle.clock.value
    clock.generation, clock.selected = 3, _key("model-12.pt")
    copied = copy.deepcopy(clock)
    assert isinstance(copied, ml.ListClock) and copied is not clock
    assert copied.lock is not clock.lock
    assert (copied.generation, copied.selected, copied.listed) == (0, None, False)
    # Two sessions of one page: two clocks.
    one, two = _Page(demo, handle), _Page(demo, handle)
    assert one[handle.clock] is not two[handle.clock] and one[handle.fold] is not two[handle.fold]


def test_a_re_read_keeps_a_pick_a_fold_and_show_older_runs_committed_while_it_read(
        root, serving, monkeypatch):
    """SL-R3b: the list read again (Models' opening, the re-read after a
    Load) while a newer pick, a fold and "Show older runs" are committed: its
    answer writes the listing and keeps them (the selection committed last,
    judged by `kept_selection`), never the selection, folds or "Show older
    runs" it began with."""
    page, handle = _session("research")
    operator_mode(monkeypatch)
    page[handle.workspace] = "demo"
    page.act("_refresh")
    first, newer = _key("model-12.pt"), _key("model-05.pt")
    page.act("_select", evt={"key": first, "gen": "1"})
    node = ml.run_node("gatv2/20261015-080000_gene_aaaaaa")
    original = ml.list_models

    def meanwhile(*args, **kwargs):
        monkeypatch.setattr(ml, "list_models", original)
        page.act("_select", evt={"key": newer, "gen": "1"})
        page.act("_select", evt={"fold": node, "open": False, "gen": "1"})
        page.act("_select", evt={"key": "__older__", "gen": "1"})
        assert page[handle.selected] == newer and page[handle.show_older] is True
        return original(*args, **kwargs)

    monkeypatch.setattr(ml, "list_models", meanwhile)
    page.act("_refresh")
    assert page[handle.selected] == newer
    assert page[handle.fold] == {node: False} and page[handle.show_older] is True
    # The drawing agrees: the newer pick is the one current.
    assert 'aria-current="true"' in _row_of(page[handle.rows], "model-05.pt")
    assert page[handle.listing][ml.GENERATION] == 1               # the same workspace
    assert handle.clock.value is not page[handle.clock]          # the session's own
    assert page[handle.clock].selected == newer


def test_a_re_read_clears_a_newer_pick_no_longer_listed_and_never_sets_back_its_own(
        root, serving, monkeypatch):
    """The latest selection judged by the shared rule (`kept_selection`): a
    pick made while the list was read, whose file the read no longer lists,
    is cleared, never replaced by the selection the re-read began with."""
    page, handle = _session("research")
    operator_mode(monkeypatch)
    page[handle.workspace] = "demo"
    page.act("_refresh")
    first, newer = _key("model-12.pt"), _key("model-05.pt")
    page.act("_select", evt={"key": first, "gen": "1"})
    original = ml.list_models

    def meanwhile(*args, **kwargs):
        monkeypatch.setattr(ml, "list_models", original)
        page.act("_select", evt={"key": newer, "gen": "1"})
        read = original(*args, **kwargs)
        return {**read, "items": [i for i in read["items"] if ml.item_key(i) != newer]}

    monkeypatch.setattr(ml, "list_models", meanwhile)
    page.act("_refresh")
    assert page[handle.selected] is None
    assert "Select a model" in page[handle.facts]


def test_a_selection_gone_from_a_re_read_is_cleared_and_said_never_replaced(
        root, serving, monkeypatch):
    """The owner's selection rule (the recheck of bd5e678), in the shared
    rule (`kept_selection`, `ListClock.commit_listing`): (a) a first listing,
    or another workspace's, with nothing selected preselects its lone
    loadable row (R-27; a preselection loads nothing); (c) a search or a
    filter that hides the selection keeps it, said as not shown; (b) the same
    workspace read again without the checkpoint selected clears it and never
    selects another file, not even the lone loadable row left, and says so
    beside Load (the list's `said`), Load off and the facts empty; a re-read
    with nothing selected chooses nothing, and a pick is the selection."""
    shutil.copytree(root["workspace"], root["root"] / "other")
    page, handle = _session("research")
    operator_mode(monkeypatch)
    said = handle.refresh["outputs"][4]
    preset = ml.PRESETS["research"]
    chosen, lone = _key("model-12.pt"), _key("model-05.pt")
    original = ml.list_models

    def only(*keys):
        def read(*args, **kwargs):
            listing = original(*args, **kwargs)
            return {**listing, "items": [i for i in listing["items"] if ml.item_key(i) in keys]}
        return read

    # (a) The first listing: its lone loadable row preselected, nothing said.
    monkeypatch.setattr(ml, "list_models", only(lone))
    page[handle.workspace] = "demo"
    page.act("_refresh")
    assert page[handle.selected] == lone and page[said] is None
    # (c) Read again whole (the preselection kept), another row picked, a
    # search hiding it, read again: kept, said as not shown.
    monkeypatch.setattr(ml, "list_models", original)
    page.act("_refresh")
    assert page[handle.selected] == lone and page[said] is None
    page.act("_select", evt={"key": chosen, "gen": "1"})
    assert page[handle.selected] == chosen
    page.type("gene")
    page.act("_refresh")
    assert page[handle.selected] == chosen and page[said] is None
    assert "Selected model not shown" in page[handle.rows]
    page.clear()
    # (b) Read again without it, one loadable row left: cleared and said, never
    # that row; Load off, no facts.
    monkeypatch.setattr(ml, "list_models", only(lone))
    page.act("_refresh")
    assert page[handle.selected] is None and page[handle.clock].selected is None
    assert page[said] == {"key": None, "html": f'<div class="shep-reason shep-reason-grey">'
                                               f'{ml.SELECTION_GONE}</div>'}
    assert ml.SELECTION_GONE in page[handle.reason]
    assert "Select a model" in page[handle.facts]
    assert 'aria-current="true"' not in page[handle.rows]
    drawing = ml.list_drawing(page[handle.listing], page[handle.selected], preset,
                              said=page[said])
    assert drawing.load["interactive"] is False and ml.SELECTION_GONE in drawing.reason
    # A search keeps what was said (`said` is the list's state, drawn with no selection).
    page.type("disease")
    assert ml.SELECTION_GONE in page[handle.reason]
    page.clear()
    # Read again with nothing selected: still nothing chosen, nothing said.
    page.act("_refresh")
    assert page[handle.selected] is None and page[said] is None
    assert ml.SELECTION_GONE not in page[handle.reason]
    # A pick is the selection, and nothing is said beside Load.
    page.act("_select", evt={"key": lone, "gen": "1"})
    assert page[handle.selected] == lone and page[said] is None
    # (a) Another workspace, one loadable row: preselected, nothing said.
    page[handle.workspace] = "other"
    page.act("_refresh")
    assert page[handle.listing][ml.GENERATION] == 2
    assert page[handle.selected] == lone and page[said] is None
    # The rule alone: of the same workspace never a lone candidate.
    listing = only(lone)("demo", gradio_request())
    same = {"workspace": "demo"}
    assert ml.kept_selection(same, listing, chosen, None, preset) is None
    assert ml.kept_selection(same, listing, None, None, preset) is None
    assert ml.kept_selection(same, listing, lone, None, preset) == lone
    assert ml.kept_selection(None, listing, None, None, preset) == lone
    assert ml.kept_selection({"workspace": "other"}, listing, chosen, None, preset) == lone


def test_a_re_read_answered_after_another_workspace_was_chosen_changes_nothing(
        root, serving, monkeypatch):
    """SL-R3 (the opening's re-read held while another workspace is chosen and
    drawn): its answer is of a workspace the list no longer shows, so it
    changes nothing, the workspace chosen, its listing and its selection
    kept. And "Show older runs" asked while the drawer read its latest runs
    only drops that read (what it holds is less than what is shown)."""
    shutil.copytree(root["workspace"], root["root"] / "other")
    page, handle = _session("research")
    operator_mode(monkeypatch)
    page[handle.workspace] = "demo"
    page.act("_refresh")
    original = ml.list_models

    def meanwhile(*args, **kwargs):
        monkeypatch.setattr(ml, "list_models", original)
        page[handle.workspace] = "other"
        page.act("_refresh")
        page.act("_select", evt={"key": _key("model-05.pt"), "gen": "2"})
        return original(*args, **kwargs)

    monkeypatch.setattr(ml, "list_models", meanwhile)
    page[handle.workspace] = "demo"
    page.act("_refresh")
    assert page[handle.listing]["workspace"] == "other"
    assert page[handle.listing][ml.GENERATION] == 2
    assert page[handle.selected] == _key("model-05.pt")
    assert page[handle.clock].workspace == "other"

    drawer, handle = _session("clinical")
    drawer[handle.workspace] = "demo"
    drawer.act("_refresh")
    assert drawer[handle.listing]["runs_unread"] == 3

    def older_meanwhile(*args, **kwargs):
        monkeypatch.setattr(ml, "list_models", original)
        drawer.act("_select", evt={"key": "__older__", "gen": "1"})
        return original(*args, **kwargs)

    monkeypatch.setattr(ml, "list_models", older_meanwhile)
    drawer.act("_refresh")
    assert drawer[handle.show_older] is True and drawer[handle.listing]["runs_unread"] == 0


def test_a_pick_for_another_generation_is_refused_even_back_in_the_same_workspace(
        root, serving, monkeypatch):
    """SL-R3a: a pick is committed only for the listing the list shows now,
    the one it was judged on and the rows it was made in (the generation the
    page sends, `generation_marker`). A pick answered after another workspace
    was chosen, one sent from the rows before the switch, and one from before
    a switch away and back (A, B, then A again: the same relative key, the
    same workspace) select nothing; a fold sent from them changes no fold. A
    pick of the rows drawn now is the selection."""
    shutil.copytree(root["workspace"], root["root"] / "other")
    page, handle = _session("research")
    operator_mode(monkeypatch)
    key = _key("model-05.pt")
    page[handle.workspace] = "demo"
    page.act("_refresh")
    before = page[handle.listing]
    # Asked from demo's rows, processed once other's listing is committed
    # (Gradio reads a State as the event is processed: other's listing).
    queued = page.ask("_select")
    page[handle.workspace] = "other"
    page.act("_refresh")
    assert ml.find_item(page[handle.listing], key) is not None   # the same key there
    page.run(queued, evt={"key": key, "gen": "1"})
    assert page[handle.selected] is None
    assert 'data-gen="2"' in page[handle.rows]                   # what the rows say now
    page.act("_select", evt={"key": key, "gen": "1"})            # sent from demo's rows
    assert page[handle.selected] is None
    page.act("_select", evt={"fold": ml.run_node("gatv2/20261015-080000_gene_aaaaaa"),
                             "open": False, "gen": "1"})
    assert page[handle.fold] == {}
    # Back to demo: a third generation; an act of the first is not a new one.
    page[handle.workspace] = "demo"
    page.act("_refresh")
    page.act("_select", evt={"key": key, "gen": "1"})
    assert page[handle.selected] is None
    assert page[handle.listing][ml.GENERATION] == 3 and 'data-gen="3"' in page[handle.rows]
    held = page.ask("_select")
    page[handle.listing] = before                # judged on the first listing, held inside
    page.run(held, evt={"key": key})
    assert page[handle.selected] is None
    page[handle.listing] = {**before, ml.GENERATION: 3}
    for sent in ("x", "", "1.0"):                                # never a generation
        page.act("_select", evt={"key": key, "gen": sent})
        assert page[handle.selected] is None, sent
    page.act("_select", evt={"key": key, "gen": "3"})
    assert page[handle.selected] == key
    assert ml.list_generation("3") == 3 and ml.list_generation({ml.GENERATION: 3}) == 3
    assert ml.list_generation(True) is None and ml.list_generation("x") is None


def test_a_newer_pick_stays_when_an_older_pick_answers_after_it(root, serving, monkeypatch):
    """SL-R3a (the review of dcaed68, acceptance 3): a run's pick in ``demo``
    held where it chooses the run's file; meanwhile ``other`` is chosen and a
    file of it picked (the same relative keys there). The held pick answers
    after that: it selects nothing, and the newer pick stays."""
    shutil.copytree(root["workspace"], root["root"] / "other")
    page, handle = _session("research")
    operator_mode(monkeypatch)
    page[handle.workspace] = "demo"
    page.act("_refresh")
    held_run, newer = _key("model-12.pt").rsplit("/", 1)[0], _key("model-05.pt")
    original = ml.run_choice

    def meanwhile(*args, **kwargs):
        monkeypatch.setattr(ml, "run_choice", original)
        page[handle.workspace] = "other"
        page.act("_refresh")
        page.act("_select", evt={"key": newer, "gen": "2"})
        assert page[handle.selected] == newer
        return original(*args, **kwargs)

    monkeypatch.setattr(ml, "run_choice", meanwhile)
    page.act("_select", evt={"run": held_run, "gen": "1"})
    assert page[handle.selected] == newer and page[handle.listing]["workspace"] == "other"
    assert 'aria-current="true"' in _row_of(page[handle.rows], "model-05.pt")
    assert 'aria-current="true"' not in _row_of(page[handle.rows], "model-12.pt")


def test_the_folds_are_changed_in_place_and_never_written_back(root, serving, monkeypatch):
    """The fold-dict overwrite (OPS-SL-R3): the folds are the session's one
    dict, changed in place under the clock (a fold kept, a search's dropped,
    a pick's unfolds, another workspace's cleared) and returned by no handler,
    so an answer that began before a fold cannot write an older copy back."""
    page, handle = _session("research")
    operator_mode(monkeypatch)
    page[handle.workspace] = "demo"
    page.act("_refresh")
    folds = page[handle.fold]
    node = ml.run_node("gatv2/20261015-080000_gene_aaaaaa")
    page.act("_select", evt={"fold": node, "open": False, "gen": "1"})
    page.type("disease")
    page.act("_select", evt={"fold": node, "open": True, "searched": True, "gen": "1"})
    assert page[handle.fold] is folds and folds == {node: False, ml.search_node(node): True}
    page.clear()
    assert page[handle.fold] is folds and folds == {node: False}
    # A search's change held while a fold is kept: the fold stays.
    late = page.type("gene", answered=False)
    page.act("_select", evt={"fold": ml.run_node("gatv2/20261013-080000_disease_cccccc"),
                             "open": True, "gen": "1"})
    for asked in late:
        page.run(asked)
    assert folds == {node: False, ml.run_node("gatv2/20261013-080000_disease_cccccc"): True}
    shutil.copytree(root["workspace"], root["root"] / "other")
    page[handle.workspace] = "other"
    page.act("_refresh")
    assert page[handle.fold] is folds and folds == {}


# --------------------------------------------------------------------------- the UI sweep of 2026-11-22
def test_a_pages_tools_share_one_bar_with_the_search_and_the_drawer_keeps_its_row():
    """GB10-UI-03, B-03 (the sweep's M2): a page's own tools (`tools`) are
    built inside the list's tools row right after the search group, so a
    page's group (Models' Order) and the search share one bar
    (``shep-bar``): the search group (``shep-grp shep-grp-grow``) grows and
    keeps its box's floor, a group that does not fit moves under it whole, and
    the count line is the bar's note (a row of its own). The page's column is
    on the panel's one content edge (``shep-edge``). The Diagnosis drawer
    passes no tools and keeps its field row, its search row and its count
    beside the search, as before."""
    from src.webui.shell import standalone_context

    def tools_of(preset, tools=None):
        with gr.Blocks() as demo:
            ctx = standalone_context(demo)
            ctx.create_model_list(preset, tools=tools)
        config = demo.get_config_file()
        known = {c["id"]: (c.get("props") or {}) for c in config["components"]}
        props = type("Props", (dict,), {"__missing__": lambda self, key: {}})(known)

        def walk(node):
            yield node
            for child in node.get("children", []):
                yield from walk(child)

        row = next(n for n in walk(config["layout"])
                   if "shep-model-tools" in (props[n["id"]].get("elem_classes") or []))
        kids = [(props[c["id"]].get("elem_id"), props[c["id"]].get("elem_classes") or [])
                for c in row.get("children", [])]
        main = [props[n["id"]].get("elem_classes") or [] for n in walk(config["layout"])
                if "shep-model-main" in (props[n["id"]].get("elem_classes") or [])]
        return props[row["id"]].get("elem_classes") or [], kids, main

    def tools():
        with gr.Row(elem_id="page-order", elem_classes=["shep-grp"]):
            gr.Radio(choices=["a", "b"], show_label=False, container=False)

    classes, kids, main = tools_of("research", tools)
    assert "shep-bar" in classes and "shep-field-row" not in classes
    assert [k for k, _c in kids] == ["shep-models-research-searchrow", "page-order",
                                     "shep-models-research-count"]
    assert {"shep-search-row", "shep-grp", "shep-grp-grow"} <= set(kids[0][1])
    assert "shep-bar-note" in kids[2][1]
    assert main and "shep-edge" in main[0]
    # The drawer: no tools, its field row, its search row and count as they were.
    classes, kids, main = tools_of("clinical")
    assert "shep-field-row" in classes and "shep-bar" not in classes and main == []
    assert kids == [("shep-models-clinical-searchrow", ["shep-search-row"]),
                    ("shep-models-clinical-count", ["shep-html", "shep-field-aside"])]


def test_the_trees_optional_columns_are_marked_and_its_frame_is_a_fit_box(root, monkeypatch):
    """B-13 and the sweep's M6: the tree's Epochs, Files and Fingerprint are
    optional columns (``shep-col-opt`` on their header and on every cell of
    theirs, a run's and a file's), so the table can drop them while it does not
    fit, before its frame scrolls sideways; Model, Task and Operation never
    carry the mark. The frame is a fit box (``style.FIT``) whose wide form is
    the table (``style.FIT_WIDE``): the shell's layout pass measures it and
    marks it narrow (`style.NARROW_ATTR`). A row that spans the columns (an
    unreadable file, an entry not listed) still spans them."""
    from src.webui.style import FIT, FIT_WIDE

    listing = _listing(root, monkeypatch)
    research = ml.PRESETS["research"]
    view = {"keys": None}
    tree = ml.tree_view(listing, None, research, show_older=True,
                        fold={ml.run_node(ml.item_run_key(i)): True for i in listing["items"]})
    html = ml._tree_html(listing, None, research, view, tree)
    assert f'class="shep-tbl-wrap shep-mtbl-wrap {FIT}"' in html
    assert f'class="shep-tbl shep-mtbl shep-mtree {FIT_WIDE}"' in html
    heads = re.findall(r'<th scope="col"( class="([^"]*)")?>([^<]+)</th>', html)
    marked = {name: "shep-col-opt" in (classes or "") for _a, classes, name in heads}
    assert marked == {"Model": False, "Task": False, "Operation": False, "Epochs": True,
                      "Files": True, "Fingerprint": True}, heads
    assert ml.OPTIONAL_COLUMNS == ("Epochs", "Files", "Fingerprint")
    for row in re.findall(r"<tr class=\"shep-tree-(?:run|file)[^>]*>(.*?)</tr>", html):
        cells = re.findall(r"<td([^>]*)>", row)
        if len(cells) == len(ml.TREE_COLUMNS):
            assert ["shep-col-opt" in c for c in cells] == [False, False, False, True, True, True], row
        else:                                     # a row spanning the columns after Model
            assert len(cells) == 2 and f'colspan="{len(ml.TREE_COLUMNS) - 1}"' in cells[1], row


def test_the_lists_empty_state_says_each_cause_once_in_the_flow(monkeypatch, tmp_path):
    """The skill's §9 and the sweep's first use: the list with no workspaces
    folder set drew the folder's whole problem (296 characters, its variable in
    the sentence) as a bare reason; it is now the folder's one notice
    (`renderers.notice_block`: its state, the way to Settings › Data
    locations, the rest under Details ›), which the listing carries
    (``notice``, from `workspaces.read_workspaces`). An empty workspaces folder
    says there is no workspace yet and the next step; a workspace with no
    checkpoint says so. Each is in the flow (``.shep-empty``), never a bare
    reason, and names no variable in its words."""
    research = ml.PRESETS["research"]
    monkeypatch.delenv("SHEPHERD_WORKSPACES_ROOT", raising=False)
    first = ml.list_models(None, operator_request(monkeypatch))
    assert first["notice"]["key"] == "workspaces_root" and first["workspace"] is None
    html = ml.list_html(first, None, research)
    assert 'data-notice="workspaces_root"' in html and 'data-shep-go="settings"' in html
    assert "SHEPHERD_WORKSPACES_ROOT" not in _visible(html).split("Details")[0]
    assert "shep-reason" not in html
    empty_root = tmp_path / "workspaces"
    empty_root.mkdir()
    monkeypatch.setenv("SHEPHERD_WORKSPACES_ROOT", str(empty_root))
    empty = ml.list_models(None, operator_request(monkeypatch))
    assert empty["notice"] is None and empty["workspaces"] == []
    assert ml.list_html(empty, None, research).startswith(f'<p class="shep-empty">{ml.NO_WORKSPACE}</p>')
    (empty_root / "demo").mkdir()
    none_listed = {**empty, "workspace": "demo", "workspaces": [{"name": "demo"}]}
    assert ml.list_html(none_listed, None, research).startswith(
        f'<p class="shep-empty">{ml.NO_CHECKPOINT}</p>')
