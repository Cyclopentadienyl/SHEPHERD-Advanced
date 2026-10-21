"""
The dashboard's one chart component (M5-2, UI-C).
=================================================
`src/webui/components/charts.py` and `src/webui/static/js/shep_charts.js`.
What these pin (the browser behaviour itself is checked with Playwright, in
the package's scratch browser checks):

- **the vendored library**: uPlot 1.6.32's files byte-identical to the npm
  tarball (their SHA-256), its licence, a README naming the version, the
  tarball's integrity and each file's digest, and every asset the component
  loads served by the parent app's ``/static`` (200);
- **one component, one way to load it**: a ``gr.HTML`` with a static template
  (never ``${value}``), no prose CSS, no padding or container, the value a
  tiny token, the data from its server functions; the scripts in its own
  ``head=`` (the locale guard, uPlot, the module, in that order) and never in
  the mount's;
- **the data**: values rounded to 6 significant digits, gaps for anything not
  finite, a chart's scale "linear" or "log" only;
- **the module's lifecycle and the two 1.6.32 workarounds** (static, and in
  Node where it is installed: the log ticks for a minimum that is not a power
  of ten, the locale guard for ``en-US@posix``).

Module: tests/unit/test_webui_charts.py
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

from src.webui.components import charts  # noqa: E402

STATIC = charts.STATIC_DIR
MODULE_JS = (STATIC / charts.MODULE).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- the vendored library
def test_the_vendored_files_are_the_tarballs_byte_for_byte():
    assert charts.UPLOT_VERSION == "1.6.32"
    for relative, digest in charts.VENDORED.items():
        assert hashlib.sha256((STATIC / relative).read_bytes()).hexdigest() == digest, relative
    assert set(p.name for p in (STATIC / charts.UPLOT_DIR).iterdir()) == {
        "uPlot.iife.min.js", "uPlot.min.css", "LICENSE", "README.txt"}


def test_the_vendored_readme_names_the_version_the_integrity_and_each_digest():
    readme = (STATIC / charts.UPLOT_DIR / "README.txt").read_text(encoding="utf-8")
    assert "1.6.32" in readme and "MIT" in readme
    assert ("sha512-KIMVnG68zvu5XXUbC4LQEPnhwOxBuLyW1AHtpm6IKTXImkbLgkMy+jabjLgSLMasNuGGzQm"
            "/ep3tOkyTxpiQIw==") in readme
    assert "c800a63b432bad692d6d746f44f0882aa73a49ae" in readme
    for digest in charts.VENDORED.values():
        assert digest in readme
    licence = (STATIC / charts.UPLOT_DIR / "LICENSE").read_text(encoding="utf-8")
    assert licence.startswith("The MIT License (MIT)") and "Leon Sorokin" in licence


def _static_client():
    """The parent app's /static, as `src/api/main.py` mounts it."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from starlette.staticfiles import StaticFiles

    import src.api.main as main

    assert Path(main._static_dir).resolve() == STATIC.resolve()
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=str(main._static_dir)), name="static")
    return TestClient(app)


def test_every_asset_the_component_loads_is_served():
    client = _static_client()
    urls = re.findall(r'(?:src|href)="(/static/[^"]+)"', charts.CHART_HEAD)
    assert len(urls) == 4
    for url in urls:
        response = client.get(url)
        assert response.status_code == 200, url
    for relative in charts.VENDORED:
        assert client.get(f"/static/{relative}").status_code == 200, relative


def test_the_head_loads_the_guard_then_uplot_then_the_module_and_versions_our_scripts():
    head = charts.CHART_HEAD
    guard = head.index(charts.LOCALE_GUARD)
    library = head.index("uPlot.iife.min.js")
    module = head.index(charts.MODULE)
    assert guard < library < module
    for relative in (charts.LOCALE_GUARD, charts.MODULE):
        digest = hashlib.sha256((STATIC / relative).read_bytes()).hexdigest()[:12]
        assert f"/static/{relative}?v={digest}" in head
    assert "cdn" not in head.lower() and "http" not in head


def test_the_scripts_are_never_in_the_mounts_head():
    """Gradio's dedupe skips, and does not await, a script already in the
    document: one in the mount's ``head=`` would race the component."""
    from src.webui.app import mount_options

    head = mount_options()["head"]
    for name in ("uPlot.iife.min.js", "shep_charts.js", "shep_charts_locale.js"):
        assert name not in head


# --------------------------------------------------------------------------- the component
def _figure():
    def series_fn(token):
        return token

    def csv_fn(token):
        return {"csv": ""}

    with gr.Blocks():
        figure = charts.chart_figure(key="k", elem_id="fig", series_fn=series_fn, csv_fn=csv_fn,
                                     help_key="training-charts")
    return figure


def test_the_figure_is_one_html_with_a_static_template_and_a_token_value():
    figure = _figure()
    assert isinstance(figure, gr.HTML)
    assert figure.apply_default_css is False and figure.padding is False
    assert figure.container is False
    assert "${" not in figure.html_template
    assert figure.value == {"run": None, "rev": 0}
    assert figure.head == charts.CHART_HEAD
    assert 'data-series-fn="series_fn"' in figure.html_template
    assert 'data-csv-fn="csv_fn"' in figure.html_template
    assert "window.ShepCharts.mount(element, props, watch, server, 'k')" in figure.js_on_load
    # The data comes through the component's server functions.
    assert getattr(figure, "series_fn")._is_server_fn is True
    assert getattr(figure, "csv_fn")._is_server_fn is True


def test_the_toolbar_offers_reset_export_zoom_and_the_gestures():
    template = _figure().html_template
    for act in ("reset", "csv", "zoom-in", "zoom-out", "follow"):
        assert f'data-chart-act="{act}"' in template
    assert charts.GESTURES in template and "The wheel scrolls the page here" in template
    assert 'aria-describedby="shep-help-text-training-charts"' in template


# --------------------------------------------------------------------------- the data
def test_values_are_rounded_and_what_is_not_a_finite_number_is_a_gap():
    assert charts.rounded(0.123456789) == 0.123457
    assert charts.rounded(1234567.89) == 1234570.0
    for gap in (float("nan"), float("inf"), None, True, "1.0"):
        assert charts.rounded(gap) is None
    spec = charts.chart_spec("lr", "Learning rate (log)", [1, 2, 3],
                             [("learning rate", [1e-4, 0.0, float("nan")])], y_scale="log")
    assert spec == {"id": "lr", "title": "Learning rate (log)", "x": [1.0, 2.0, 3.0],
                    "series": [{"label": "learning rate", "values": [0.0001, 0.0, None]}],
                    "yScale": "log"}
    json.dumps(spec)                 # plain lists, as the server function returns them
    with pytest.raises(ValueError):
        charts.chart_spec("x", "X", [], [], y_scale="symlog")


def test_the_token_and_the_payload():
    assert charts.token("ws/gat/r", 12) == {"run": "ws/gat/r", "rev": 12}
    assert charts.token(None, 0, empty="none") == {"run": None, "rev": 0, "empty": "none"}
    assert charts.token_from([{"run": "a"}]) == {"run": "a"}
    assert charts.token_from({"run": "a"}) == {"run": "a"}
    assert charts.token_from("ws/../x") == {}
    assert charts.figure_payload("r", 3, [], empty="e") == {"run": "r", "rev": 3, "charts": [],
                                                            "empty": "e"}


def test_no_chart_text_is_under_14px():
    sizes = re.findall(r"font-size:\s*([0-9.]+)px", charts.CHART_CSS)
    assert sizes and all(float(s) >= 14 for s in sizes)
    assert "FONT = \"14px" in MODULE_JS                      # the axes, drawn on the canvas


# --------------------------------------------------------------------------- the module
def test_the_module_cleans_up_after_itself():
    """One AbortController for every listener, all on the component's root;
    one ResizeObserver that destroys on detachment and skips a hidden (0-width)
    report; uPlot.destroy() on teardown; a re-mount destroys the instance it
    replaces; stale replies dropped; the value watched and read once now."""
    js = MODULE_JS
    assert "new AbortController()" in js and "signal: inst.ac.signal" in js
    assert "document.addEventListener" not in js and "window.addEventListener" not in js
    assert "new ResizeObserver" in js and "if (!element.isConnected) { inst.destroy(); return; }" in js
    assert "r.width > 0 && r.height > 0" in js
    assert "if (INST[key]) { INST[key].destroy(); }" in js
    assert "st.u.destroy()" in js and "inst.ac.abort()" in js and "inst.ro.disconnect()" in js
    assert "my !== inst.seq" in js
    assert 'watch("value", function () { refresh(inst); });' in js
    assert js.index('watch("value"') < js.index("    refresh(inst);\n    return inst;")
    assert "st.u.setData(st.data, false)" in js


def test_the_module_works_around_the_two_uplot_1632_defects_and_its_mousedown_leak():
    js = MODULE_JS
    assert "splits: logSplits" in js                                    # #1098
    assert "mousedown: function () { return null; }" in js            # destroy mid-press leak
    assert "mouseup: function () { return null; }" in js
    guard = (STATIC / charts.LOCALE_GUARD).read_text(encoding="utf-8")
    assert "new Intl.NumberFormat(navigator.language)" in guard        # #1053
    assert 'value: "en-US"' in guard


def test_the_module_shares_the_epoch_axis_and_keeps_the_view_per_run():
    js = MODULE_JS
    assert 'sync: { key: "shep-" + inst.key, scales: ["x", "y"]' in js
    assert "setSeries: false" in js
    assert "window.sessionStorage.setItem(storeKey(inst)" in js and "catch (err)" in js
    # a pending view is written for the run it belongs to before another run is shown
    assert "if (inst.saveTimer) { writeView(inst); }            // the run shown until now" in js
    # zoom limits: a minimum span, at most 1000x the data, data decades + 6 on log
    assert "if (axis === \"x\") { return 1; }" in js
    assert "(ex[1] - ex[0]) * 1000" in js and "span + 6" in js


def test_colours_are_assigned_by_label_and_the_second_is_dashed():
    js = MODULE_JS
    assert "function colourSlot(inst, chartId, label)" in js
    assert "if (!(label in m)) { m[label] = Object.keys(m).length; }" in js
    assert re.search(r"var DASHES = \[null, \[", js)


NODE = shutil.which("node")


def _node(script: str) -> str:
    result = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.skipif(NODE is None, reason="Node is not installed")
def test_log_ticks_return_for_minimums_that_hang_uplot_1632():
    """1.6.32's logAxisSplits loops forever (or throws) for these minimums
    (upstream #1098); ours returns finite, increasing ticks inside the range."""
    out = _node(
        "const vm=require('vm');const fs=require('fs');"
        "const ctx={window:{},console};vm.createContext(ctx);"
        f"vm.runInContext(fs.readFileSync({json.dumps(str(STATIC / charts.MODULE))},'utf8'),ctx);"
        "const f=ctx.window.ShepCharts.logSplits;const res={};"
        "for (const [lo,hi] of [[9.7e-5,3e-3],[9.99e-5,3e-3],[9.5e-6,3e-3],[0.99e-3,3e-3],"
        "[9.6255e-5,1e-3],[1.2e-7,1.4e-4],[3.1e-5,3.3e-5],[1e-12,1]]) {"
        "  const t=f(null,1,lo,hi);"
        "  res[lo+','+hi]=t.length>=2&&t.length<=40&&t.every((v,i)=>v>=lo&&v<=hi&&(i===0||v>t[i-1]));"
        "}"
        "res.empty=[f(null,1,0,1).length,f(null,1,-1,1).length,f(null,1,1,1).length];"
        "console.log(JSON.stringify(res));")
    res = json.loads(out)
    assert res.pop("empty") == [0, 0, 0]
    assert all(res.values()), res


@pytest.mark.skipif(NODE is None, reason="Node is not installed")
def test_the_locale_guard_lets_a_posix_locale_load_uplot():
    out = _node(
        "const vm=require('vm');const fs=require('fs');"
        "const nav={language:'en-US@posix'};"
        "const ctx={navigator:nav,Intl,Object,console};vm.createContext(ctx);"
        "let before='ok';try{new Intl.NumberFormat(nav.language)}catch(e){before='throws'}"
        f"vm.runInContext(fs.readFileSync({json.dumps(str(STATIC / charts.LOCALE_GUARD))},'utf8'),ctx);"
        "let after='ok';try{new Intl.NumberFormat(ctx.navigator.language)}catch(e){after='throws'}"
        "console.log(JSON.stringify([before, after, ctx.navigator.language]));")
    before, after, language = json.loads(out)
    assert after == "ok" and language in ("en-US", "en-US@posix")
    if before == "throws":
        assert language == "en-US"


@pytest.mark.skipif(NODE is None, reason="Node is not installed")
def test_each_run_keeps_its_own_view_in_the_tab():
    """Going back to a run shown earlier in the tab brings its view back (the
    last 20 runs are kept); a run never shown has none, so it fits its data;
    unreadable or blocked storage never throws."""
    out = _node(
        "const vm=require('vm');const fs=require('fs');"
        "const m=new Map();const store={getItem:k=>m.has(k)?m.get(k):null,setItem:(k,v)=>m.set(k,String(v))};"
        "const ctx={window:{sessionStorage:store},console,JSON,Array,Object,setTimeout,clearTimeout};vm.createContext(ctx);"
        f"vm.runInContext(fs.readFileSync({json.dumps(str(STATIC / charts.MODULE))},'utf8'),ctx);"
        "const V=ctx.window.ShepCharts._views;"
        "const inst=(run,x)=>({key:'training',run,xManual:!!x,x,frozenLast:null,charts:{},order:[]});"
        "V.write(inst('ws/gat/a',[3,9]));V.write(inst('ws/gat/b',null));"
        "const res={a:V.load(inst('ws/gat/a'),'ws/gat/a'),b:V.load(inst('ws/gat/b'),'ws/gat/b'),"
        "           never:V.load(inst('x'),'ws/gat/never')};"
        "for(let i=0;i<20;i++){V.write(inst('ws/gat/r'+i,[i,i+5]));}"
        "res.aAfter20=V.load(inst('x'),'ws/gat/a');res.r19=V.load(inst('x'),'ws/gat/r19');"
        "m.set('shep-charts-view-v2:training','{not json');res.corrupt=V.load(inst('x'),'ws/gat/r19');"
        "store.setItem=()=>{throw new Error('blocked')};store.getItem=()=>{throw new Error('blocked')};"
        "let threw=false;try{V.write(inst('ws/gat/c',[1,2]));res.blocked=V.load(inst('x'),'ws/gat/c');}catch(e){threw=true}"
        "res.threw=threw;console.log(JSON.stringify(res));")
    res = json.loads(out)
    assert res["a"]["x"] == [3, 9] and res["b"]["x"] is None and res["never"] is None
    assert res["aAfter20"] is None and res["r19"]["x"] == [19, 24]
    assert res["corrupt"] is None and res["blocked"] is None and res["threw"] is False


@pytest.mark.skipif(NODE is None, reason="Node is not installed")
def test_a_run_shown_next_fits_every_chart_unless_it_has_its_own_view():
    """A value-axis zoom on run A is never applied to run B (same chart ids):
    entering B clears every chart's value view and the epoch view; going back
    to A brings A's stored views back."""
    out = _node(
        "const vm=require('vm');const fs=require('fs');"
        "const m=new Map();const store={getItem:k=>m.has(k)?m.get(k):null,setItem:(k,v)=>m.set(k,String(v))};"
        "const ctx={window:{sessionStorage:store},console,JSON,Array,Object,setTimeout,clearTimeout};vm.createContext(ctx);"
        f"vm.runInContext(fs.readFileSync({json.dumps(str(STATIC / charts.MODULE))},'utf8'),ctx);"
        "const V=ctx.window.ShepCharts._views;"
        "const st=(id)=>({id,yManual:true,y:[0.6,1.2],data:[[]]});"
        "const inst={key:'training',run:'ws/gat/a',xManual:true,x:[3,9],frozenLast:9,"
        "            charts:{loss:st('loss'),lr:st('lr')},order:['loss','lr'],saveTimer:null};"
        "V.write(inst);"
        "V.enter(inst,'ws/gat/b');"
        "const b={run:inst.run,x:inst.x,xManual:inst.xManual,pending:inst.pendingYs,"
        "         loss:[inst.charts.loss.yManual,inst.charts.loss.y],lr:[inst.charts.lr.yManual,inst.charts.lr.y]};"
        "V.enter(inst,'ws/gat/a');"
        "const a={x:inst.x,xManual:inst.xManual,pending:inst.pendingYs};"
        "clearTimeout(inst.saveTimer);"
        "console.log(JSON.stringify({b,a}));")
    res = json.loads(out)
    assert res["b"] == {"run": "ws/gat/b", "x": None, "xManual": False, "pending": {},
                        "loss": [False, None], "lr": [False, None]}
    assert res["a"] == {"x": [3, 9], "xManual": True,
                        "pending": {"loss": [0.6, 1.2], "lr": [0.6, 1.2]}}


@pytest.mark.skipif(NODE is None, reason="Node is not installed")
def test_the_epoch_axis_and_zoom_note_name_whole_epochs():
    out = _node(
        "const vm=require('vm');const fs=require('fs');"
        "const ctx={window:{},console};vm.createContext(ctx);"
        f"vm.runInContext(fs.readFileSync({json.dumps(str(STATIC / charts.MODULE))},'utf8'),ctx);"
        "const E=ctx.window.ShepCharts._epochs;"
        "const a=E.axis({font:'f'});"
        "console.log(JSON.stringify({incrs:a.incrs||null,font:a.font,ticks:a.values(null,[1,2,10]),"
        "                            labels:[1.3,2.5,59.999,null].map(E.label)}));")
    res = json.loads(out)
    assert res["incrs"] and res["incrs"][0] == 1 and all(float(v).is_integer() for v in res["incrs"])
    assert res["font"] == "f" and res["ticks"] == ["1", "2", "10"]
    assert res["labels"] == ["1", "3", "60", ""]
    assert "epochAxis(axis)," in MODULE_JS                    # the chart's epoch axis
    assert '"zoomed: epochs " + wholeEpoch(inst.x[0])' in MODULE_JS


def test_the_figures_buttons_keep_their_padding_under_gradios_button_rule():
    """Gradio's ``.gradio-container-<version> button { padding: 0 }`` (a
    class and an element) outranks a one-class rule: the figure's button
    rules name the figure too."""
    for cls in ("shep-charts-btn", "shep-charts-icon", "shep-charts-new"):
        assert re.search(rf"^\.shep-charts \.{cls} \{{", charts.CHART_CSS, re.M), cls
        assert not re.search(rf"^\.{cls} \{{", charts.CHART_CSS, re.M), cls
