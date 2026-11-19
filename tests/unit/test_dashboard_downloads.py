"""OPS-C-R4 (the external review of 6a2d52e; the WebUI ledger's B72-E1): **a
download link names no server path.**

Every file the dashboard offers to download (Diagnosis' CSV and report,
Models › Results' Export) was handed to Gradio's ``gr.DownloadButton`` or
``gr.File`` as a path; Gradio copied it into its cache and answered the event
with that copy's absolute path, and the browser downloaded it from
``/ui/gradio_api/file=<that path>``: the server's temporary directory (or the
directory Gradio was configured to use) was in the event's answer and in the
link.

Now the shared download boundary (`src/webui/exports.py`) gives each file an
opaque, unguessable id, and each export's button (a ``gr.Button`` with a
``link``) the link to the dashboard's one download route
(``<mount>/download/<id>``), which serves that file while it lives; an
unknown, malformed or expired id answers 404, and only a file the download
directory holds can be offered. The files' content, names and lifetime are as
before; Gradio keeps no copy.

- the id mapping, unit by unit;
- each export the dashboard offers, through Gradio's own routes and the real
  handlers: the event's answer and the link name no server path, the file
  downloads with its bytes and its name unchanged, and nothing else does;
- a linked button in Gradio 6.30's own page, in a real browser (Node and
  Playwright, skipped without): a click saves the file through the link.

Module: tests/unit/test_dashboard_downloads.py
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit

import gradio as gr
import pytest

pytest.importorskip("fastapi")

from tests.unit.test_results_export import world  # noqa: E402,F401  (the fixture)

#: A link as the browser gets it: relative to the page (Gradio's page sets
#: ``<base href="./">``, the mount), the route and one opaque id.
LINK = re.compile(r"download/[A-Za-z0-9_-]{43}")


@pytest.fixture
def downloads(monkeypatch, tmp_path):
    """The download directory of this process, new, under `tmp_path`; the ids
    offered so far forgotten."""
    from src.webui import exports

    monkeypatch.setattr(exports.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(exports, "_EXPORT_DIR", None)
    monkeypatch.setattr(exports, "_OFFERED", {})
    return exports.export_dir()


# ---------------------------------------------------------------------------
# The id mapping
# ---------------------------------------------------------------------------
def test_an_offered_file_is_named_by_an_opaque_id_that_serves_it_while_it_lives(
        downloads, tmp_path):
    """`offer`: what the browser gets for a file of the download directory is
    a link with a fresh, unguessable id, and nothing that names where the
    file is; `offered`: the id gives back that file while it lives, and
    nothing else ever: an unknown id, a malformed or path-like one, an
    expired or deleted file."""
    from src.webui import exports

    path = exports.write_new("diagnosis_pt_01_sess_abc", "csv", "rank,gene\n1,SCN1A\n")
    link = exports.offer(path)
    assert LINK.fullmatch(link), link
    assert str(tmp_path) not in link and path.name not in link
    file_id = link.split("/", 1)[1]
    assert exports.offered(file_id) == path
    # Each offer is its own id: an id says nothing of another.
    again = exports.offer(str(path))
    assert again != link and exports.offered(again[len("download/"):]) == path
    # An unknown id, a malformed one, anything path-like: nothing.
    for asked in ("A" * 43, file_id[:-1], file_id + "A", "", str(path), path.name,
                  f"../{path.name}", "..%2F" + path.name, "/" + file_id, file_id + "/",
                  f"{file_id}/../{path.name}", "%2F".join(str(path).split("/")), None, 7):
        assert exports.offered(asked) is None, asked
    # Expired (its lifetime, from when it was written): nothing, and the id is
    # forgotten, even should the file be touched again.
    old = time.time() - exports.LIFETIME_SECONDS - 60
    os.utime(path, (old, old))
    assert exports.offered(file_id) is None
    os.utime(path, None)
    assert exports.offered(file_id) is None
    # A file deleted (pruned, or the directory removed): nothing.
    other = exports.write_new("results_x", "json", "[]")
    other_id = exports.offer(other)[len("download/"):]
    other.unlink()
    assert exports.offered(other_id) is None


def test_only_a_file_of_the_download_directory_can_be_offered(downloads, tmp_path):
    """No submitted path becomes a download target: `offer` refuses a path
    outside the download directory, a link in it, a directory and a file
    that is not there, and forgets the ids of files that expired when it
    offers the next."""
    from src.webui import exports

    elsewhere = tmp_path / "elsewhere.csv"
    elsewhere.write_text("secret")
    link = downloads / "link.csv"
    link.symlink_to(elsewhere)
    (downloads / "sub").mkdir()
    for path in (elsewhere, link, downloads / "sub", downloads / "missing.csv",
                 downloads / ".." / "elsewhere.csv", "relative.csv"):
        with pytest.raises(ValueError):
            exports.offer(path)
    assert exports._OFFERED == {}
    expired = exports.write_new("old", "csv", "x")
    exports.offer(expired)
    old = time.time() - exports.LIFETIME_SECONDS - 60
    os.utime(expired, (old, old))
    exports.offer(exports.write_new("new", "csv", "y"))
    assert [path.name for path in exports._OFFERED.values()] == ["new.csv"]


def test_every_download_of_the_dashboard_is_a_linked_button_and_the_route_is_the_mounts():
    """The dashboard builds no ``gr.DownloadButton`` and no ``gr.File``
    (Gradio's file controls, which hand it a path and link its copy by that
    path); each export is a ``gr.Button`` its handler gives the link
    `exports.offer` answers (none before a file is written); the route is the
    mounted app's own; Gradio's cache holds no download file, so its sweep is
    retired with the copies."""
    from src.webui.app import create_gradio_app, mount_options

    app = create_gradio_app()
    assert not [b for b in app.blocks.values() if isinstance(b, (gr.DownloadButton, gr.File))]
    exports_ = {b.elem_id: b for b in app.blocks.values()
                if getattr(b, "elem_id", None) in ("shep-dx-export-csv", "shep-dx-export-md",
                                                   "shep-rs-file")}
    assert set(exports_) == {"shep-dx-export-csv", "shep-dx-export-md", "shep-rs-file"}
    for button in exports_.values():
        assert type(button) is gr.Button and button.link is None, button
    (route,) = mount_options()["app_kwargs"]["routes"]
    assert route.path == "/download/{file_id}" and route.methods >= {"GET", "HEAD"}
    assert app.delete_cache is None
    # Every link to the route is followed as a download (REV-R4-a), by one
    # script in the dashboard's head.
    from src.webui import exports

    head = mount_options()["head"]
    assert head.count(exports.DOWNLOAD_HEAD) == 1
    assert f"var LINK = {json.dumps(exports.LINK)};" in exports.DOWNLOAD_HEAD
    assert "new URL(LINK, document.baseURI).href" in exports.DOWNLOAD_HEAD
    assert "a.setAttribute('download', '')" in exports.DOWNLOAD_HEAD
    assert "}, true);" in exports.DOWNLOAD_HEAD


# ---------------------------------------------------------------------------
# Each export, served
# ---------------------------------------------------------------------------
def _diagnosis_page(monkeypatch):
    """The Diagnosis page in the real shell; a gene task model answers each
    run (its status, its facts and its answer are stand-ins, as in
    `test_diagnosis_panel`)."""
    import copy

    from src.webui import shell
    from src.webui.components import diagnosis_panel as dp
    from tests.unit.test_diagnosis_panel import _gene_result, _status_serving, _task_identity

    status = _status_serving(_task_identity("gene_task"))
    monkeypatch.setattr(dp.card, "read_status_within", lambda *a, **k: (status, None))
    monkeypatch.setattr(dp.card, "served_facts", lambda s, **k: {
        "key": dp.card.served_key(s), "operator": False, "workspaces": [], "serving": None,
        "item": None, "item_problem": None, "recorded": None, "trained": None, "cohorts": None})
    monkeypatch.setattr(dp, "_call_diagnose", lambda **kwargs: copy.deepcopy(_gene_result()))
    with gr.Blocks() as demo:
        shell.build_shell(demo, {**{page: (lambda ctx: None)
                                    for page in (*shell.RESEARCH_PAGES, shell.SETTINGS)},
                                 shell.CLINICAL: dp.create_diagnosis_tab})
    return demo


def _completed(session, fn, data):
    """Ask `fn` with `data` on `session`; Gradio's answer as the browser
    reads it from the event stream (the ``process_completed`` message)."""
    event = session.join(fn, data)
    assert session.wait(event)
    return session.done[event]


def _link_of(message, fn, block):
    """The link the export button `block` holds after the answer `message`
    to `fn`: its update's ``link`` (on 6a2d52e a download control's
    ``value``, Gradio's file data, whose ``url`` was the link)."""
    update = message["output"]["data"][fn.outputs.index(block)]
    if "link" in update:
        return update["link"]
    return (update.get("value") or {}).get("url")


def _no_server_path(text, roots, what):
    for root in roots:
        assert str(root) not in text, f"{what} names a server path ({root}): {text[:600]}"
    assert "file=" not in text, f"{what} names Gradio's file route: {text[:600]}"


def _downloaded(session, page, link, directory, roots, what):
    """`link`, as the browser resolves it against the page, gets a file of
    the download `directory`, whole (by HEAD, and a range of it too), as an
    attachment under the file's own name; no part of the answer names a
    server path. Returns the link resolved, the name and the bytes."""
    assert LINK.fullmatch(str(link)), f"{what}: the link names more than an id: {link}"
    url = urljoin(page, link)
    assert urlsplit(url).path.startswith("/ui/download/"), url
    answer = session.client.get(url)
    assert answer.status_code == 200, (what, answer.status_code, answer.text[:300])
    name = re.fullmatch(r'attachment; filename="([A-Za-z0-9_.-]+)"',
                        answer.headers["content-disposition"]).group(1)
    written = (directory / name).read_bytes()
    assert answer.content == written, what
    assert answer.headers["cache-control"] == "no-store"
    _no_server_path(json.dumps(dict(answer.headers)), roots, f"{what}'s headers")
    head = session.client.head(url)
    assert head.status_code == 200 and head.content == b""
    assert int(head.headers["content-length"]) == len(written)
    part = session.client.get(url, headers={"Range": "bytes=0-9"})
    assert part.status_code == 206 and part.content == written[:10], what
    return url, name, written


def test_each_export_downloads_by_a_link_that_names_no_server_path(
        world, monkeypatch, tmp_path):  # noqa: F811 (the fixture)
    """Diagnosis' CSV and report (the real page in the shell, Run Diagnosis
    asked through Gradio's queue) and Models › Results' Export, CSV and JSON
    (the real segment, Export asked through the queue): Gradio's answer and
    each link name no server path (neither the download directory nor
    Gradio's cache, nor its file route); each link downloads the file the
    handler wrote, its bytes and its name unchanged (the CSV is the page's
    own, the export the service's serializer), with HEAD and a range; Gradio
    keeps no copy; an unknown, path-like or expired id answers 404 naming
    nothing, and Gradio's file route does not serve the file's own path.
    On 6a2d52e the answer held ``/tmp/gradio/<hash>/diagnosis_pt_g_sess_g.csv``
    and the link ``…/ui/gradio_api/file=`` that path."""
    from src.api.services import results as service
    from src.webui import exports
    from src.webui.app import mount_options
    from src.webui.components import diagnosis_panel as dp
    from src.webui.components import results_segment as rs
    from tests.fixtures.served_page import ServedSession, served
    from tests.unit.test_models_results_segment import _listing, _operator, _shown, _state

    monkeypatch.setattr(exports.tempfile, "tempdir", str(tmp_path / "temp"))
    (tmp_path / "temp").mkdir()
    monkeypatch.setattr(exports, "_EXPORT_DIR", None)
    if hasattr(exports, "_OFFERED"):
        monkeypatch.setattr(exports, "_OFFERED", {})

    # --- Diagnosis: the page in the real shell, a gene task model's answer.
    demo = _diagnosis_page(monkeypatch)
    blocks = {getattr(b, "elem_id", None): b for b in demo.blocks.values()}
    (run,) = [fn for fn in demo.fns.values() if fn.name == "_run_diagnosis"]
    data = [None if isinstance(b, gr.State) else b.value for b in run.inputs]
    data[run.inputs.index(blocks["dx_hpo_terms"])] = "HP:0001250"
    data[run.inputs.index(blocks["dx_candidate_genes"])] = "SCN1A, PTEN"
    seen = {}
    with served(demo, **mount_options()) as base:
        page = base[:-len("gradio_api")]
        session = ServedSession(base, demo)
        try:
            message = _completed(session, run, data)
            directory = exports.export_dir()
            roots = (directory, Path(demo.GRADIO_CACHE))
            for elem_id, what in (("shep-dx-export-csv", "Diagnosis' CSV"),
                                  ("shep-dx-export-md", "Diagnosis' report")):
                link = _link_of(message, run, blocks[elem_id])
                seen[what] = _downloaded(session, page, link, directory, roots, what)
            _no_server_path(json.dumps(message), roots, "Run Diagnosis' answer")
            csv_url, csv_name, _bytes = seen["Diagnosis' CSV"]
            # The names as before (the result's patient and session).
            assert csv_name == "diagnosis_pt_g_sess_g.csv"
            assert seen["Diagnosis' report"][1] == "diagnosis_pt_g_sess_g_report.md"
            # The page's own CSV of the result the session holds.
            (result,) = [value for value in (session.state(b) for b in run.outputs
                                             if isinstance(b, gr.State))
                         if isinstance(value, dict) and "candidates" in value]
            text = session.client.get(csv_url).text
            assert "SCN1A" in text and "PTEN" in text
            assert text == dp._build_results_csv(result)
            # Nothing else downloads: an unknown id, a path-like one, the
            # file's own path, the file's own path through Gradio's route.
            path = directory / csv_name
            for asked in ("download/" + "A" * 43, "download/..%2F..%2Fetc%2Fpasswd",
                          "download/" + quote(str(path), safe=""), "download/" + path.name):
                answer = session.client.get(urljoin(page, asked))
                assert answer.status_code == 404, (asked, answer.status_code)
                assert path.read_bytes() not in answer.content
                _no_server_path(answer.text, roots, f"the answer to {asked}")
            assert session.client.get(urljoin(page, "download/" + "A" * 43)).text == \
                exports.NOT_OFFERED
            # Gradio's own route refuses the file's path (its refusal repeats
            # the path it was sent, nothing more).
            answer = session.client.get(urljoin(page, "gradio_api/file=" + str(path)))
            assert answer.status_code == 403 and path.read_bytes() not in answer.content
            # Expired (a day after it was written): 404, naming nothing.
            old = time.time() - exports.LIFETIME_SECONDS - 60
            os.utime(path, (old, old))
            answer = session.client.get(csv_url)
            assert answer.status_code == 404, answer.status_code
            _no_server_path(answer.text, roots, "an expired link's answer")
            # Gradio keeps no copy of any.
            names = [name for _url, name, _bytes in seen.values()]
            assert not [p for name in names for p in Path(demo.GRADIO_CACHE).rglob(name)]
        finally:
            session.close()

    # --- Models › Results: the segment; Export of the rows shown.
    _operator(monkeypatch)
    state, _scope, _rows = _shown(_state(world, monkeypatch))
    listing, selected = _listing()
    shown = [state["rows"][i] for i in state["shown"]]
    with gr.Blocks() as segment:
        seg = gr.Radio(choices=[rs.RESULTS], value=rs.RESULTS, visible=False)
        listing_state, selected_state = gr.State(), gr.State()
        parts = rs.create_results_segment()
        rs.wire_results_segment(parts, seg=seg, listing=listing_state, selected=selected_state)
    (export,) = [fn for fn in segment.fns.values() if fn.name == "_on_results_export"]
    (compare,) = [fn for fn in segment.fns.values() if fn.name == "_on_results_compare"]
    with served(segment, **mount_options()) as base:
        page = base[:-len("gradio_api")]
        session = ServedSession(base, segment)
        try:
            # The session's first event (Gradio holds its states from then on).
            _completed(session, compare, [None, [], True])
            session.set_state(parts["state"], state)
            session.set_state(listing_state, listing)
            session.set_state(selected_state, selected)
            for fmt, text in (("csv", service.to_csv(shown)), ("json", service.to_json(shown))):
                message = _completed(session, export, [None, None, None, rs.WORKSPACE, [],
                                                       rs.SHOWN, fmt])
                directory = exports.export_dir()
                roots = (directory, Path(segment.GRADIO_CACHE))
                link = _link_of(message, export, parts["file"])
                what = f"Models › Results' {fmt.upper()} export"
                _url, name, written = _downloaded(session, page, link, directory, roots, what)
                # The serializer's bytes, under the name as before, which the
                # note says.
                assert written == text.encode("utf-8"), what
                assert re.fullmatch(rf"results_workspace-lab_\d{{8}}T\d{{6}}Z(-\d+)?\.{fmt}", name)
                note = message["output"]["data"][export.outputs.index(parts["export_note"])]
                assert name in json.dumps(note), note
                _no_server_path(json.dumps(message), roots, f"Export's answer ({fmt})")
                assert not list(Path(segment.GRADIO_CACHE).rglob(name))
        finally:
            session.close()


# ---------------------------------------------------------------------------
# In a browser
# ---------------------------------------------------------------------------
#: Two export buttons as the pages build them (Diagnosis': kept mounted but
#: hidden, disabled; Results': not shown) in Gradio's own page: Make writes a
#: file for each and gives each its link, as the handlers do; then each is
#: clicked as a reader clicks it. What the browser saved, from which address,
#: under which name; what each button is drawn as; whether the page asked
#: Gradio's file route for anything. Then Expire puts both files past their
#: lifetime (REV-R4-a) and the CSV's export is clicked again: whether the
#: click was a download (and whether it failed), whether the page's own frame
#: went anywhere, and whether the page is still there.
_SAVE_JS = r"""
const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const [base] = process.argv.slice(2, 3);
  const browser = await chromium.launch();
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 },
                                             acceptDownloads: true });
  const page = await context.newPage();
  const asked = [];
  page.on('request', (request) => asked.push(request.url()));
  await page.goto(base, { waitUntil: 'networkidle' });
  const before = await page.evaluate(() => {
    const el = document.querySelector('#save-csv');
    return { tag: el.tagName, hidden: el.classList.contains('hidden'), disabled: el.disabled };
  });
  await page.locator('#make').click();
  await page.waitForFunction(() => document.querySelector('a#save-csv[href]') &&
                             document.querySelector('a#save-json[href]'), null, { timeout: 30000 });
  const save = async (selector) => {
    const waiting = page.waitForEvent('download', { timeout: 15000 });
    await page.locator(selector).click();
    const download = await waiting;
    return { url: download.url(), name: download.suggestedFilename(),
             text: fs.readFileSync(await download.path(), 'utf8'),
             href: await page.locator(selector).getAttribute('href'),
             label: (await page.locator(selector).innerText()).trim() };
  };
  const out = { before, csv: await save('#save-csv'), json: await save('#save-json'),
                still: page.url() };
  await page.locator('#expire').click();
  await page.waitForFunction(() => /expired/.test((document.querySelector('#expired') || {}).textContent || ''),
                             null, { timeout: 30000 });
  const went = [];
  page.on('framenavigated', (frame) => { if (frame === page.mainFrame()) { went.push(frame.url()); } });
  const clicked = Promise.race([
    page.waitForEvent('download', { timeout: 15000 }).then(async (d) => (
      { download: d.url(), failure: await d.failure() })),
    page.waitForEvent('framenavigated', { timeout: 15000 }).then((f) => ({ navigated: f.url() })),
  ]);
  const href = await page.locator('#save-csv').getAttribute('href');
  await page.locator('#save-csv').click();
  out.expired = { href, ...(await clicked) };
  await page.waitForLoadState('load');
  out.expired.went = went;
  out.expired.url = page.url();
  out.expired.page = await page.locator('#make').count();
  out.expired.download_attr = await page.evaluate(() => {
    const a = document.querySelector('a#save-csv');
    return a ? a.getAttribute('download') : null;
  });
  out.asked = asked.filter((url) => url.includes('/download/') || url.includes('file='));
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((err) => { console.error(err); process.exit(1); });
"""


def test_a_linked_export_button_saves_its_file_in_a_browser(downloads, tmp_path):
    """An export button as the pages build it, in Gradio 6.30's own page:
    before a file it is a disabled button (hidden or not shown); once its
    handler gives it `exports.offer`'s link it is drawn as that link, and a
    click saves the file written, under its own name, from
    ``/ui/download/<id>``, leaving the page where it was; nothing the page
    asks names Gradio's file route. Once the file is past its lifetime
    (REV-R4-a), a click on its link is a download that fails and the page
    stays (`exports.DOWNLOAD_HEAD`); on 47bd59a the tab went to the route's
    404 text and the page was gone."""
    from src.webui import exports
    from src.webui.app import mount_options
    from tests.fixtures.browser import node_with_playwright, run_node
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    texts = {"csv": "rank,gene\n1,SCN1A\n2,PTEN\n", "json": '[{"metric": "mrr"}]'}

    written = []

    def make():
        csv = exports.write_new("diagnosis_pt_01_sess_abc", "csv", texts["csv"])
        data = exports.write_new("results_workspace-lab", "json", texts["json"])
        written.extend([csv, data])
        return (gr.update(link=exports.offer(csv), interactive=True, visible=True),
                gr.update(value="⬇ Download JSON", link=exports.offer(data), visible=True))

    def expire():
        # Written past the files' lifetime (a day) ago, as the clock would leave them.
        then = time.time() - exports.LIFETIME_SECONDS - 60
        for path in written:
            os.utime(path, (then, then))
        return "expired"

    with gr.Blocks() as demo:
        made = gr.Button("Make", elem_id="make")
        csv_button = gr.Button("⬇ Export CSV", size="sm", interactive=False, visible="hidden",
                               elem_id="save-csv")
        json_button = gr.Button("⬇ Download", size="sm", visible=False, elem_id="save-json")
        made.click(make, outputs=[csv_button, json_button])
        expired = gr.HTML("", elem_id="expired")
        gr.Button("Expire", elem_id="expire").click(expire, outputs=[expired])
    with served(demo, **mount_options()) as base:
        page = base[:-len("gradio_api")]
        out = run_node(_SAVE_JS, tmp_path, page)
    assert out["before"] == {"tag": "BUTTON", "hidden": True, "disabled": True}, out
    for kind, name, label in (("csv", "diagnosis_pt_01_sess_abc.csv", "⬇ Export CSV"),
                              ("json", "results_workspace-lab.json", "⬇ Download JSON")):
        saved = out[kind]
        assert re.fullmatch(r"http://127\.0\.0\.1:\d+/ui/download/[A-Za-z0-9_-]{43}",
                            saved["url"]), saved
        assert LINK.fullmatch(saved["href"]), saved
        assert saved["name"] == name and saved["text"] == texts[kind], saved
        assert saved["label"] == label, saved
    assert out["still"].split("?")[0] == page, out
    # Past its lifetime: the click is a download that fails, the page stays.
    expired = out["expired"]
    assert "navigated" not in expired and expired["went"] == [], expired
    assert expired["url"] == out["still"] and expired["page"] == 1, expired
    assert expired["download"] == urljoin(page, expired["href"]), expired
    assert expired["failure"], expired
    assert expired["download_attr"] == "", expired
    assert not [url for url in out["asked"] if "file=" in url], out
    assert str(downloads) not in json.dumps(out)


# ---------------------------------------------------------------------------
# A control with no file to offer (OPS-FIX-R1, the recheck of d7ebfc3)
# ---------------------------------------------------------------------------
#: Each step of `plan` in the real page at the owner's window: its acts
#: (fill, click), then each export control as a reader meets it: what it is
#: drawn as (``href``, disabled, shown), a mouse click at its centre, the
#: keyboard's Tab from `plan.from` until it (then Enter), and focus asked by
#: script (then Enter); for each, the file the browser saved (name and text)
#: or none. A wait that does not end is recorded, never thrown, so a page that
#: answers differently is measured, not lost.
_NO_FILE_JS = r"""
const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const [base, planText, queue] = process.argv.slice(2, 5);
  const plan = JSON.parse(planText);
  const browser = await chromium.launch();
  const context = await browser.newContext({ viewport: { width: 1850, height: 960 },
                                             acceptDownloads: true });
  const page = await context.newPage();
  await page.addInitScript(queue);
  await page.goto(base);
  await page.waitForSelector(plan.from, { timeout: 30000 });
  const joins = () => page.evaluate(() => window.__shepQueue.joins.length);
  const idle = () => page.waitForFunction(() => window.__shepQueue.idle(), null,
                                          { timeout: 30000 });
  const saved = async (act) => {
    const waiting = page.waitForEvent('download', { timeout: 2500 })
      .then(async (d) => ({ name: d.suggestedFilename(),
                            text: fs.readFileSync(await d.path(), 'utf8') }))
      .catch(() => null);
    await act();
    return await waiting;
  };
  const probe = async (id) => {
    const sel = '#' + id;
    if (!(await page.$(sel))) {
      return { state: null, mouse: null, reached: false, keyboard: null, focused: false,
               scripted: null };
    }
    const state = await page.$eval(sel, (e) => ({
      tag: e.tagName, href: e.getAttribute('href'), disabled: e.disabled === true,
      shown: e.checkVisibility({ checkVisibilityCSS: true }) }));
    if (state.shown) { await page.locator(sel).scrollIntoViewIfNeeded(); }
    const box = state.shown ? await page.locator(sel).boundingBox() : null;
    const mouse = box ? await saved(() => page.mouse.click(box.x + box.width / 2,
                                                         box.y + box.height / 2)) : null;
    await page.focus(plan.from);
    const tabs = [];
    for (let i = 0; i < 40; i += 1) {
      await page.keyboard.press('Tab');
      const at = await page.evaluate(() => (document.activeElement || {}).id || '');
      tabs.push(at);
      if (at === id) { break; }
    }
    const reached = tabs[tabs.length - 1] === id;
    const keyboard = reached ? await saved(() => page.keyboard.press('Enter')) : null;
    const focused = await page.evaluate((s) => {
      const e = document.querySelector(s); e.focus(); return document.activeElement === e;
    }, sel);
    const scripted = focused ? await saved(() => page.keyboard.press('Enter')) : null;
    await page.evaluate(() => document.activeElement && document.activeElement.blur());
    return { state, mouse, reached, keyboard, focused, scripted };
  };
  const out = [];
  for (const step of plan.steps) {
    for (const act of step.acts) {
      const before = await joins();
      if (act.fill) { await page.fill(act.fill, act.text); }
      else { await page.locator(act.click, act.has ? { hasText: act.has } : {}).first().click(); }
      if (act.answered) {
        await page.waitForFunction((n) => window.__shepQueue.joins.length > n, before,
                                   { timeout: 30000 }).catch(() => null);
      }
      await page.waitForTimeout(150);
      await idle();
    }
    const said = await page.waitForFunction(([s, t]) => (
      (document.querySelector(s) || {}).innerText || '').includes(t), step.marker,
      { timeout: 15000 }).then(() => true).catch(() => false);
    await idle();
    const controls = {};
    for (const id of plan.controls) { controls[id] = await probe(id); }
    out.push({ name: step.name, said, controls });
  }
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((err) => { console.error(err); process.exit(1); });
"""


def _no_file_walk(demo, plan, tmp_path):
    """`_NO_FILE_JS` over `demo`, served as the dashboard mounts it; each
    step's measures by name."""
    from src.webui.app import mount_options
    from tests.fixtures.browser import QUEUE_JS, node_with_playwright, run_node
    from tests.fixtures.served_page import served

    if node_with_playwright() is None:
        pytest.skip("Node with Playwright is not installed")
    with served(demo, **mount_options()) as base:
        steps = run_node(_NO_FILE_JS, tmp_path, base[:-len("gradio_api")], json.dumps(plan),
                         QUEUE_JS)
    return {step["name"]: step for step in steps}


def _offers(step, control, name, holds, lacks):
    """At `step`, `control` offers only the file `name`: drawn as a link, and
    the mouse, Tab then Enter, and focus then Enter each save that file,
    which holds `holds` and none of `lacks`."""
    seen = step["controls"][control]
    assert step["said"], step["name"]
    assert seen["state"]["tag"] == "A" and LINK.fullmatch(seen["state"]["href"] or ""), seen
    for way in ("mouse", "keyboard", "scripted"):
        got = seen[way]
        assert got is not None, (step["name"], control, way, seen)
        assert got["name"] == name, (step["name"], control, way, got["name"])
        assert holds in got["text"], (step["name"], control, way)
        assert not [text for text in lacks if text in got["text"]], (step["name"], control, way)


def _offers_nothing(step, control):
    """At `step`, `control` points at nothing and cannot be used: no link;
    the mouse, Tab and focus by script reach no file (a disabled button
    takes no focus; a hidden one has no box; one not shown may not be drawn
    at all)."""
    seen = step["controls"][control]
    assert step["said"], step["name"]
    if seen["state"] is not None:  # drawn (a control not shown may not be in the page)
        assert seen["state"]["href"] is None, (step["name"], control, seen["state"])
        assert seen["state"]["tag"] == "BUTTON" and seen["state"]["disabled"], \
            (step["name"], seen)
    assert not seen["reached"] and not seen["focused"], (step["name"], control, seen)
    assert seen["mouse"] is None and seen["keyboard"] is None and seen["scripted"] is None, \
        (step["name"], control, seen)


def test_a_diagnosis_export_offers_only_the_answer_on_screen_or_nothing(
        downloads, monkeypatch, tmp_path):
    """OPS-FIX-R1 (the recheck of d7ebfc3): the real Diagnosis page in a
    browser, through its own events: an answer whose exports were written
    (A: SCN1A), then one whose writing failed (B: MECP2), then one with no
    candidate (B: KCNQ2), then a written one again (C: PTEN), then its
    inputs edited (the reset, after all those events). At each step the CSV
    and the report offer the answer on screen's own file, or nothing: by the
    mouse, by Tab then Enter, by focus then Enter. On d7ebfc3 an update's
    ``link=None`` was dropped by Gradio (it removes a None-valued key from an
    update, in place in the page's shared update), so the failed export's
    disabled control was still the earlier answer's ``<a href>``, reachable
    by Tab, and Enter saved SCN1A's CSV under MECP2's answer."""
    import copy

    from src.webui.components import diagnosis_panel as dp
    from tests.unit.test_diagnosis_panel import _gene_result

    demo = _diagnosis_page(monkeypatch)
    original = dp._write_exports

    def answer(**kwargs):
        (gene,) = kwargs["candidate_genes"]
        result = copy.deepcopy(_gene_result())
        result.update(patient_id=f"pt_{gene}", session_id=f"sess_{gene}", warnings=[])
        one = {**result["candidates"][0], "gene_name": gene, "gene_id": f"disgenet:{gene}"}
        result["candidates"] = [] if gene == "KCNQ2" else [one]
        return result

    def write(result):
        if result["session_id"] == "sess_MECP2":
            raise OSError("simulated: the download directory cannot be written")
        return original(result)

    monkeypatch.setattr(dp, "_call_diagnose", answer)
    monkeypatch.setattr(dp, "_write_exports", write)
    genes = "#dx_candidate_genes textarea"
    run = {"click": "#shep-dx-run-btn", "answered": True}
    body = "#shep-dx-results-body"
    plan = {"from": "#shep-dx-run-btn", "controls": ["shep-dx-export-csv", "shep-dx-export-md"],
            "steps": [
                {"name": "A", "marker": [body, "SCN1A"],
                 "acts": [{"fill": "#dx_hpo_terms textarea", "text": "HP:0001250"},
                          {"fill": genes, "text": "SCN1A"}, run]},
                {"name": "B failed", "marker": [body, "could not be written"],
                 "acts": [{"fill": genes, "text": "MECP2"}, run]},
                {"name": "B none", "marker": [body, "No candidates found"],
                 "acts": [{"fill": genes, "text": "KCNQ2"}, run]},
                {"name": "C", "marker": [body, "PTEN"],
                 "acts": [{"fill": genes, "text": "PTEN"}, run]},
                {"name": "reset", "marker": [body, "Inputs changed"],
                 "acts": [{"fill": "#dx_hpo_terms textarea", "text": "HP:0001263"}]}]}
    seen = _no_file_walk(demo, plan, tmp_path)
    others = ["SCN1A", "MECP2", "KCNQ2", "PTEN"]
    for step, gene in (("A", "SCN1A"), ("C", "PTEN")):
        lacks = [other for other in others if other != gene]
        _offers(seen[step], "shep-dx-export-csv", f"diagnosis_pt_{gene}_sess_{gene}.csv",
                f"disgenet:{gene}", lacks)
        _offers(seen[step], "shep-dx-export-md", f"diagnosis_pt_{gene}_sess_{gene}_report.md",
                f"pt_{gene}", lacks)
    for step in ("B failed", "B none", "reset"):
        for control in ("shep-dx-export-csv", "shep-dx-export-md"):
            _offers_nothing(seen[step], control)
    # Shown, disabled, beside their reason while an answer is on screen;
    # hidden once there is none.
    for step, shown in (("B failed", True), ("B none", True), ("reset", False)):
        assert seen[step]["controls"]["shep-dx-export-csv"]["state"]["shown"] is shown, step


def test_a_results_export_offers_only_the_last_export_or_nothing(
        world, downloads, monkeypatch, tmp_path):  # noqa: F811 (the fixture)
    """OPS-FIX-R1: Models › Results' Export in a browser, through the
    segment's own events (a read, the scope, then Export): CSV of the rows
    shown (A), then Selected with nothing chosen (B: nothing to export), then
    JSON whose writing fails (B), then JSON written (C). The download offers
    the last export's own file, or nothing: by the mouse, by Tab then Enter,
    by focus then Enter. On d7ebfc3 nothing-to-export kept the earlier link
    (hidden), and a failed write raised, leaving the earlier export offered."""
    from src.api.services import results as service
    from src.webui.components import results_segment as rs
    from tests.unit.test_models_results_segment import _listing, _operator, _shown, _state

    _operator(monkeypatch)
    state, _scope, _rows = _shown(_state(world, monkeypatch))
    shown = [state["rows"][i] for i in state["shown"]]
    listing, selected = _listing()
    calls = []
    original = rs.export_file

    def write(rows, fmt, name):
        calls.append(fmt)
        if len(calls) == 2:
            raise OSError("simulated: the download directory cannot be written")
        return original(rows, fmt, name)

    monkeypatch.setattr(rs, "export_file", write)
    with gr.Blocks() as segment:
        seg = gr.Radio(choices=[rs.RESULTS], value=rs.RESULTS, visible=False)
        parts = rs.create_results_segment()
        rs.wire_results_segment(parts, seg=seg, listing=gr.State(listing),
                                selected=gr.State(selected))
    parts["column"].visible = True
    export = {"click": "#shep-rs-export", "answered": True}
    note = "#shep-rs-export-note"
    plan = {"from": "#shep-rs-export", "controls": ["shep-rs-file"],
            "steps": [
                {"name": "A", "marker": [note, "as CSV"],
                 "acts": [{"click": "#shep-rs-again", "answered": True},
                          {"click": "#shep-rs-scope label", "has": "This workspace",
                           "answered": True}, export]},
                {"name": "B none", "marker": [note, "Choose results above"],
                 "acts": [{"click": "#shep-rs-which label", "has": "Selected"}, export]},
                {"name": "B failed", "marker": [note, "Export file not written"],
                 "acts": [{"click": "#shep-rs-which label", "has": "All shown"},
                          {"click": "#shep-rs-format label", "has": "JSON"}, export]},
                {"name": "C", "marker": [note, "as JSON"], "acts": [export]}]}
    seen = _no_file_walk(segment, plan, tmp_path)
    assert calls == ["csv", "json", "json"], calls
    for step, text in (("A", service.to_csv(shown)), ("C", service.to_json(shown))):
        got = seen[step]["controls"]["shep-rs-file"]
        name = (got["mouse"] or {}).get("name", "")
        assert name.endswith(".csv" if step == "A" else ".json"), (step, got)
        _offers(seen[step], "shep-rs-file", name, text, [])
        assert got["mouse"]["text"] == text, step
    for step in ("B none", "B failed"):
        _offers_nothing(seen[step], "shep-rs-file")
        drawn = seen[step]["controls"]["shep-rs-file"]["state"]
        assert drawn is None or not drawn["shown"], (step, drawn)
