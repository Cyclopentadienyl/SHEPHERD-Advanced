"""
The dashboard's one chart component (M5-2, UI-C): a figure of uPlot charts.
==========================================================================
`docs/working/PLAN_TEST_RESULTS.md` §6, "M5's boundaries" (the charts: uPlot,
vendored). One Python builder (`chart_figure`, used by Training and by any
later chart) and one browser module (``window.ShepCharts``,
`src/webui/static/js/shep_charts.js`) that takes curve groups as data plus
settings: per chart a title, ``{x, series: [{label, values}], yScale:
"linear" | "log"}`` (`chart_spec`). Colours are assigned by label, never by
position, and the second colour is also dashed; zoom and pan live in the
module, never in a chart; the figure shares one X view and keeps one Y view
per chart.

**The library.** uPlot **1.6.32**, the minified IIFE and its CSS
byte-identical to the npm tarball, with its MIT licence and a README naming the
version, the tarball's integrity and each file's SHA-256
(`src/webui/static/vendor/uplot-1.6.32/`; `VENDORED`), served by the parent
app's ``/static``. Nothing comes from a CDN and the vendored files are never
edited: the two 1.6.32 defects that matter are worked around in our code (log
ticks, `shep_charts.js`; the locale guard, `shep_charts_locale.js`).

**Gradio wiring** (6.30's lifecycle, verified in a browser): one ``gr.HTML``
with a static ``html_template`` (never ``${value}``, never changed at runtime:
Gradio leaves the charts' DOM alone while the template string is unchanged),
``apply_default_css=False`` (Gradio's prose CSS would reach the buttons), no
padding and no container, its height in the page's CSS. The scripts load only
through this component's ``head=`` (`CHART_HEAD`, in order: the locale guard,
uPlot, the module), never through the mount's ``head=``. The component's
value is a tiny token, ``{"run": <run id>, "rev": <rows logged>}``; the data
is fetched through the component's ``server_functions`` (one parameter, the
token), so an unchanged token sends nothing and a tick never carries the
curves. The chart is never hidden with ``visible=False`` (that re-mounts it;
the module still cleans up if it happens).

**Server functions are the page's**: each takes the token, validates the run
id against what the server lists (never a path), and returns `figure_payload`
(plain lists, values rounded to `SIGNIFICANT` significant digits, non-finite
values as gaps) or, for Export CSV, ``{"filename", "csv"}``.

**Replies and the pending cue** (the module): a reply is drawn only if it
answers the newest request and the token the component holds now, so a late
reply never brings back a run after the selection was cleared or changed. A
failed call (6.30's client answers it with ``undefined``) is retried, never
read as "no longer listed". While the reply for a run other than the one
drawn is outstanding, the module marks the figure ``aria-busy="true"`` and
`CHART_CSS` hides the figure with ``visibility`` (never ``display``, which
would give the ResizeObserver 0 x 0 and leave a new run's charts uncreated)
and says `LOADING`, the template's own first line; a retry and new epochs of
the run drawn never mark it.

Module: src/webui/components/charts.py
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import gradio as gr

from src.webui.components.help import help_button

#: The parent app serves this directory at ``/static`` (`src/api/main.py`).
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
UPLOT_VERSION = "1.6.32"
UPLOT_DIR = f"vendor/uplot-{UPLOT_VERSION}"

#: The vendored files and their SHA-256, as in the npm tarball
#: ``uplot-1.6.32.tgz`` (integrity sha512-KIMVnG68…, shasum c800a63b…; the
#: directory's README): a test checks the files have not changed.
VENDORED = {
    f"{UPLOT_DIR}/uPlot.iife.min.js": "19c8d4c6ad88929a79f4ae49d6f7161566dfd0ba3d15cc495e974f787eb78f1f",
    f"{UPLOT_DIR}/uPlot.min.css": "df630c6a8d6f8eeaff264b50f73ce5b114f646ffd9a0bb74f049b0a00135fa04",
    f"{UPLOT_DIR}/LICENSE": "8f989229699b4fe2f1a0432d0e9edc338a8a911e250e2d1b01ecd770a5f5b1bd",
}

#: Our own scripts, versioned by their content in the URL so a browser never
#: runs an old copy after an update.
MODULE = "js/shep_charts.js"
LOCALE_GUARD = "js/shep_charts_locale.js"

#: Values are sent rounded to this many significant digits.
SIGNIFICANT = 6

#: What the figure says while it loads: the template's line until the module
#: mounts, and over the hidden figure while the module marks it busy (`CHART_CSS`).
LOADING = "Charts loading…"


def _versioned(relative: str) -> str:
    digest = hashlib.sha256((STATIC_DIR / relative).read_bytes()).hexdigest()[:12]
    return f"/static/{relative}?v={digest}"


def chart_head() -> str:
    """The component's ``head=``: the locale guard, uPlot's CSS and script,
    then the module, in that order (Gradio runs them in insertion order and
    loads each ``src`` once however many components name it)."""
    return (f'<script src="{_versioned(LOCALE_GUARD)}"></script>'
            f'<link rel="stylesheet" href="/static/{UPLOT_DIR}/uPlot.min.css">'
            f'<script src="/static/{UPLOT_DIR}/uPlot.iife.min.js"></script>'
            f'<script src="{_versioned(MODULE)}"></script>')


CHART_HEAD = chart_head()


# ---------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------
def rounded(value: Any) -> Optional[float]:
    """A finite number rounded to `SIGNIFICANT` significant digits, else None
    (a gap: NaN, infinity, a missing value, a bool or a non-number)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return float(f"{value:.{SIGNIFICANT}g}")


def chart_spec(chart_id: str, title: str, x: Sequence[Any],
               series: Iterable[tuple], *, y_scale: str = "linear",
               x_label: Optional[str] = None) -> Dict[str, Any]:
    """One chart: ``series`` is ``(label, values)`` pairs aligned with `x`."""
    if y_scale not in ("linear", "log"):
        raise ValueError(f"y_scale is 'linear' or 'log', not {y_scale!r}")
    spec: Dict[str, Any] = {
        "id": chart_id, "title": title, "x": [rounded(v) for v in x],
        "series": [{"label": str(label), "values": [rounded(v) for v in values]}
                   for label, values in series],
        "yScale": y_scale,
    }
    if x_label:
        spec["xLabel"] = x_label
    return spec


def figure_payload(run: Optional[str], rev: Any, charts: List[Dict[str, Any]],
                   *, empty: Optional[str] = None) -> Dict[str, Any]:
    """What a series server function returns: the run and rev it answers for,
    and the charts in figure order. `empty` is said when no chart has data."""
    payload: Dict[str, Any] = {"run": run, "rev": rev, "charts": charts}
    if empty:
        payload["empty"] = empty
    return payload


def token(run: Optional[str], rev: Any = 0, *, empty: Optional[str] = None) -> Dict[str, Any]:
    """The component's value: which run, and how many rows are logged."""
    value: Dict[str, Any] = {"run": run, "rev": rev}
    if empty:
        value["empty"] = empty
    return value


def token_from(data: Any) -> Mapping[str, Any]:
    """The token a server function was called with (the browser passes the
    one argument as it is)."""
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
    return data if isinstance(data, Mapping) else {}


# ---------------------------------------------------------------------------
# The component
# ---------------------------------------------------------------------------
_ZOOM_IN = ('<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" focusable="false">'
            '<circle cx="7" cy="7" r="4.6" fill="none" stroke="currentColor" stroke-width="1.6"/>'
            '<path d="M10.4 10.4l3.4 3.4M5 7h4M7 5v4" stroke="currentColor" stroke-width="1.6" '
            'stroke-linecap="round"/></svg>')
_ZOOM_OUT = ('<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" focusable="false">'
             '<circle cx="7" cy="7" r="4.6" fill="none" stroke="currentColor" stroke-width="1.6"/>'
             '<path d="M10.4 10.4l3.4 3.4M5 7h4" stroke="currentColor" stroke-width="1.6" '
             'stroke-linecap="round"/></svg>')
_HOME = ('<svg width="15" height="15" viewBox="0 0 16 16" aria-hidden="true" focusable="false">'
         '<path d="M2.5 7.5L8 3l5.5 4.5V13H9.6V9.8H6.4V13H2.5z" fill="none" stroke="currentColor" '
         'stroke-width="1.5" stroke-linejoin="round"/></svg>')
_DOWNLOAD = ('<svg width="15" height="15" viewBox="0 0 16 16" aria-hidden="true" focusable="false">'
             '<path d="M8 2.5v7.5M4.8 7l3.2 3.2L11.2 7M3 13.2h10" fill="none" stroke="currentColor" '
             'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>')

#: The gestures in one line: the lead of the figure's ⓘ (the page's help
#: topic), not text kept on the toolbar (the GB10 WebUI recheck: one line less
#: over the plots).
GESTURES = "Wheel: zoom · drag: pan · double-click: reset"


def figure_template(*, series_fn: str, csv_fn: Optional[str], help_key: Optional[str],
                    help_title: str, empty: str) -> str:
    """The component's static template: the toolbar, the grid the module
    fills, the "new epochs" overlay and a message line (`LOADING` at first)."""
    from html import escape

    help_html = help_button(help_key, help_title) if help_key else ""
    csv = (f'<button type="button" class="shep-charts-btn" data-chart-act="csv">{_DOWNLOAD}'
           '<span>Export CSV</span></button>' if csv_fn else "")
    return (
        f'<div class="shep-charts" data-state="loading" data-series-fn="{escape(series_fn)}" '
        f'data-csv-fn="{escape(csv_fn or "")}" data-empty="{escape(empty)}">'
        '<div class="shep-charts-bar">'
        '<span class="shep-charts-zoom" hidden></span>'
        '<span class="shep-charts-tools">'
        f'{help_html}'
        f'<button type="button" class="shep-charts-btn" data-chart-act="reset">{_HOME}'
        '<span>Reset view</span></button>'
        f'{csv}'
        '<span class="shep-charts-modebar" role="group" aria-label="Zoom">'
        f'<button type="button" class="shep-charts-icon" data-chart-act="zoom-in" '
        f'aria-label="Zoom in" title="Zoom in">{_ZOOM_IN}</button>'
        f'<button type="button" class="shep-charts-icon" data-chart-act="zoom-out" '
        f'aria-label="Zoom out" title="Zoom out">{_ZOOM_OUT}</button>'
        '</span></span></div>'
        '<div class="shep-charts-body">'
        '<div class="shep-charts-grid"></div>'
        '<button type="button" class="shep-charts-new" data-chart-act="follow" hidden></button>'
        f'<div class="shep-charts-msg">{escape(LOADING)}</div>'
        '</div></div>')


def chart_figure(*, key: str, elem_id: str, series_fn: Callable[[Any], Any],
                 csv_fn: Optional[Callable[[Any], Any]] = None, help_key: Optional[str] = None,
                 help_title: str = "Chart gestures", empty: str = "No data yet.",
                 elem_classes: Optional[List[str]] = None) -> Any:
    """The figure: one ``gr.HTML`` whose value is the token (`token`), whose
    data comes from `series_fn` and whose Export CSV calls `csv_fn` (both
    server functions taking the token). `key` names the figure to the module
    (its views are kept per key and run)."""
    fns = [series_fn] + ([csv_fn] if csv_fn else [])
    return gr.HTML(
        value=token(None, 0),
        html_template=figure_template(
            series_fn=series_fn.__name__, csv_fn=csv_fn.__name__ if csv_fn else None,
            help_key=help_key, help_title=help_title, empty=empty),
        js_on_load=(f"if (window.ShepCharts) {{ window.ShepCharts.mount(element, props, watch, "
                    f"server, {key!r}); }} else {{ var m = element.querySelector('.shep-charts-msg'); "
                    f"if (m) {{ m.textContent = 'The chart library did not load.'; }} }}"),
        apply_default_css=False, padding=False, container=False,
        head=CHART_HEAD, server_functions=fns, elem_id=elem_id,
        elem_classes=["shep-html", "shep-figure", *(elem_classes or [])])


#: The figure's look, for the head's ``<style>`` (a page that draws a figure
#: puts it in its ``PAGE_CSS``): text never under 14 px, the axes included
#: (they are drawn at 14 px by the module). The buttons' rules name the
#: figure too (two classes): Gradio's own ``.gradio-container-<version>
#: button { padding: 0; font-size: 100% }`` outranks a single class. While
#: the module marks the figure busy (``aria-busy="true"``) its grid, overlay,
#: zoom note and message are hidden with ``visibility`` and the body says
#: `LOADING` (filled in for ``@LOADING@``); in the narrow layout the body keeps
#: room for that line. **A plot box is sized by the layout alone**
#: (``contain: size``): the canvas drawn into it never counts toward its size,
#: so a canvas drawn for an earlier layout (narrow, or before a banner came or
#: went) cannot hold the box at that size; the module's one ResizeObserver then
#: fits the canvas to the box, in both directions.
CHART_CSS = """
.shep-figure > .html-container { padding: 0 !important; }
.shep-charts { position: relative; display: flex; flex-direction: column; min-width: 0; min-height: 0; height: 100%; }
.shep-charts-bar { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; padding: 0 0 8px; min-height: 40px; }
.shep-charts-zoom { display: inline-flex; align-items: center; height: 28px; padding: 0 8px; border-radius: 6px; background: var(--shep-accent-50); border: 1px solid var(--shep-accent-100); color: var(--shep-accent-text); font-size: 14px; white-space: nowrap; }
.shep-charts-zoom[hidden] { display: none; }
.shep-charts-tools { margin-left: auto; display: inline-flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.shep-charts .shep-charts-btn { display: inline-flex; align-items: center; gap: 6px; height: 34px; padding: 0 12px; border-radius: 8px; border: 1px solid var(--shep-border-2); background: #fff; color: var(--shep-text); font-size: 14px; font-weight: 500; cursor: pointer; white-space: nowrap; }
.shep-charts .shep-charts-btn:hover, .shep-charts .shep-charts-icon:hover { background: var(--shep-panel-2); }
.shep-charts .shep-charts-btn:disabled { opacity: .6; cursor: progress; }
.shep-charts-modebar { display: inline-flex; border: 1px solid var(--shep-border-2); border-radius: 8px; overflow: hidden; }
.shep-charts .shep-charts-icon { display: inline-flex; align-items: center; justify-content: center; width: 34px; height: 32px; border: 0; background: #fff; color: var(--shep-text-2); cursor: pointer; }
.shep-charts-icon + .shep-charts-icon { border-left: 1px solid var(--shep-border-2); }
.shep-charts-btn:focus-visible, .shep-charts-icon:focus-visible, .shep-chart-plot:focus-visible, .shep-charts-new:focus-visible { outline: 2px solid var(--shep-accent); outline-offset: 2px; }
.shep-charts-body { position: relative; flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column; }
.shep-charts-grid { flex: 1 1 auto; min-height: 0; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); grid-auto-rows: minmax(0, 1fr); gap: 10px 22px; }
.shep-charts-n1 .shep-charts-grid { grid-template-columns: minmax(0, 1fr); }
.shep-chart { position: relative; display: flex; flex-direction: column; min-width: 0; min-height: 0; }
.shep-chart-hd { display: flex; align-items: center; gap: 4px 14px; flex-wrap: wrap; min-height: 24px; }
.shep-chart-title { font-weight: 600; font-size: 15px; color: var(--shep-text); }
.shep-chart-keys { display: inline-flex; gap: 4px 14px; flex-wrap: wrap; }
.shep-chart-key { display: inline-flex; align-items: center; gap: 6px; font-size: 14px; color: var(--shep-text-2); white-space: nowrap; }
.shep-sw { display: inline-block; flex: none; vertical-align: middle; }
.shep-chart-plot { flex: 1 1 auto; min-height: 0; min-width: 0; position: relative; border-radius: 4px; contain: size; }
.shep-chart-plot .u-wrap { cursor: crosshair; touch-action: none; }
.shep-chart-plot .u-wrap.shep-panning { cursor: grabbing; }
.shep-chart-xl { text-align: center; font-size: 14px; color: var(--shep-text-2); line-height: 1.3; }
.shep-chart-empty .shep-chart-plot { visibility: hidden; }
.shep-chart-tip { position: absolute; z-index: 3; pointer-events: none; background: #fff; border: 1px solid var(--shep-border-2); border-radius: 6px; box-shadow: 0 4px 14px rgba(20,24,40,.14); padding: 6px 8px; font-size: 14px; line-height: 1.35; color: var(--shep-text); white-space: nowrap; }
.shep-chart-tip[hidden] { display: none; }
.shep-chart-tip .shep-tip-hd { font-weight: 600; }
.shep-chart-tip .shep-sw { margin-right: 6px; }
.shep-charts .shep-charts-new { position: absolute; top: 2px; right: 2px; z-index: 4; height: 30px; padding: 0 10px; border-radius: 8px; border: 1px solid var(--shep-accent-100); background: #fff; color: var(--shep-accent-text); font-size: 14px; font-weight: 600; cursor: pointer; box-shadow: 0 2px 8px rgba(20,24,40,.12); }
.shep-charts-new[hidden] { display: none; }
.shep-charts-msg { position: absolute; inset: 40% 0 auto 0; text-align: center; font-size: 15px; color: var(--shep-muted); pointer-events: none; }
.shep-charts-msg[hidden] { display: none; }
.shep-charts[aria-busy="true"] .shep-charts-grid, .shep-charts[aria-busy="true"] .shep-charts-new, .shep-charts[aria-busy="true"] .shep-charts-zoom, .shep-charts[aria-busy="true"] .shep-charts-msg { visibility: hidden; }
.shep-charts[aria-busy="true"] .shep-charts-body::after { content: "@LOADING@"; position: absolute; inset: 40% 0 auto 0; text-align: center; font-size: 15px; color: var(--shep-muted); pointer-events: none; }
@media (max-width: 1399px), (max-height: 759px) {
  .shep-charts { height: auto; }
  .shep-charts-grid { grid-template-columns: minmax(0, 1fr); grid-auto-rows: auto; }
  .shep-chart-plot { flex: none; height: 260px; touch-action: pan-y; }
  .shep-chart-plot .u-wrap { touch-action: pan-y; }
  .shep-charts-msg { position: static; padding: 24px 0; }
  .shep-charts[aria-busy="true"] .shep-charts-body { min-height: 72px; }
}
""".replace("@LOADING@", LOADING)


__all__ = [
    "CHART_CSS",
    "CHART_HEAD",
    "GESTURES",
    "LOADING",
    "SIGNIFICANT",
    "STATIC_DIR",
    "UPLOT_VERSION",
    "VENDORED",
    "chart_figure",
    "chart_head",
    "chart_spec",
    "figure_payload",
    "figure_template",
    "rounded",
    "token",
    "token_from",
]
