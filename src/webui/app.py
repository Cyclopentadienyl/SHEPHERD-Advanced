"""
SHEPHERD-Advanced Gradio Dashboard
====================================
Main Gradio application with tabbed interface.

Module: src/webui/app.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/webui/app.py

Purpose:
    Build and export the Gradio Blocks application for the SHEPHERD-Advanced
    dashboard, and mount it into the FastAPI app at /ui (`mount_dashboard`).

Layout (M5-2, UI-A): the shell (`src/webui/shell.py`) lays out two navigation
levels, "Diagnosis (clinical)" (the first screen), "Research (pre-clinical)"
with Datasets · Training · Fine-tune · Test · Models, and "⚙ Settings"; the
header's chips;
the shared timers; and gives each page builder the page context (the page
contract is the shell's docstring). The runtime line stays at the bottom.

Dependencies:
    - gradio==6.30.0 (pinned in pyproject.toml). From Gradio 6 a mounted app
      takes its theme, CSS, head and JS from gr.mount_gradio_app(), not from
      gr.Blocks(); `mount_dashboard` is the one place that passes them
      (`mount_options`: the theme, the CSS, and the head: every style sheet and
      the scripts of the shared ⓘ, the shell and the touch record,
      `training_console.TOUCH_HEAD`).

Called by:
    - src.api.main (`mount_dashboard`, at /ui)

Version: 1.0.0
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gradio as gr

from src.utils.version_checker import DEGRADED, NOTICE, format_runtime_line, probe_runtime
from src.webui import shell
from src.webui.components import (
    datasets_page,
    diagnosis_panel,
    finetune_page,
    models_page,
    runtime_settings,
    test_page,
    training_console,
)
from src.webui.components import renderers as r
from src.webui.components.datasets_page import create_datasets_page
from src.webui.components.diagnosis_panel import create_diagnosis_tab
from src.webui.components.help import HELP_CSS, HELP_HEAD, HelpRegistry
from src.webui.components.runtime_settings import create_runtime_settings_tab
from src.webui.components.training_console import create_training_tab
from src.webui.style import SHELL_CSS

if TYPE_CHECKING:
    from fastapi import FastAPI


#: The runtime line's words when every checked module imports (a normal state:
#: quiet, no mark; the WebUI skill's §6).
RUNTIME_OK = "Runtime: every checked module imports."


def _runtime_footer() -> str:
    """The always-visible runtime line, as HTML every value of which is
    escaped (the probe's words carry exception text).

    **The consequence first** (the WebUI skill's §6): a degraded runtime's
    issues (a capability is gone: torch or PyG will not import, so GNN
    scoring cannot run) lead with the shared mark of an error, a notice's (a
    performance-only or optional gap, no CUDA device visible) with the mark of
    a condition to watch, each issue in the probe's own consequence-first
    words (`version_checker`); a runtime with no issue reads quietly. The
    versions (`format_runtime_line`) are one click away ("Versions ›"). The
    colours are the status tokens, never a raw hex.

    Wrapped: this footer exists to report a broken environment, so it must not
    itself break in one. Its own failure is rendered rather than raised.
    """
    try:
        report = probe_runtime()
        line = format_runtime_line(report)
        status = report.get("status")
        issues = [str(issue) for issue in report.get("issues", [])]
    except Exception as exc:  # noqa: BLE001 — a footer that raises reports nothing
        return ('<div class="shep-rt">' + r.status_mark(
            "error", f"Runtime status unavailable: could not probe the environment ({exc}).")
            + "</div>")
    if status in (DEGRADED, NOTICE) and issues:
        lead = r.status_mark("error" if status == DEGRADED else "warn", " ".join(issues))
    else:
        lead = f"<span>{r.e(RUNTIME_OK)}</span>"
    return (f'<div class="shep-rt">{lead}<details class="shep-more shep-rt-v">'
            f"<summary>Versions ›</summary><span>{r.e(line)}</span></details></div>")


#: The dashboard's CSS: the IBM Plex font files (served by the API at /static,
#: so the page needs no network) and the polled panels' loading flash. Passed
#: to the mount (`mount_dashboard`).
DASHBOARD_CSS = """
    /* IBM Plex Sans — local font files (offline-safe) */
    @font-face {
        font-family: 'IBM Plex Sans';
        src: url('/static/fonts/ibm-plex-sans-400.woff2') format('woff2');
        font-weight: 400;
        font-style: normal;
        font-display: swap;
    }
    @font-face {
        font-family: 'IBM Plex Sans';
        src: url('/static/fonts/ibm-plex-sans-500.woff2') format('woff2');
        font-weight: 500;
        font-style: normal;
        font-display: swap;
    }
    @font-face {
        font-family: 'IBM Plex Sans';
        src: url('/static/fonts/ibm-plex-sans-600.woff2') format('woff2');
        font-weight: 600;
        font-style: normal;
        font-display: swap;
    }
    @font-face {
        font-family: 'IBM Plex Sans';
        src: url('/static/fonts/ibm-plex-sans-700.woff2') format('woff2');
        font-weight: 700;
        font-style: normal;
        font-display: swap;
    }

    /* CSS fallback stack (not in theme font= to avoid Gradio trying to load them as files) */
    .gradio-container, .gradio-container * {
        font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
    }

    /* Suppress Gradio's opacity/loading flash on the polled resources line */
    #resource_display,
    #resource_display *,
    div:has(> #resource_display) {
        animation: none !important;
        transition: none !important;
        opacity: 1 !important;
    }
    #resource_display.pending,
    #resource_display.generating,
    div:has(> #resource_display).pending,
    div:has(> #resource_display).generating {
        opacity: 1 !important;
        pointer-events: auto !important;
    }
    #resource_display .eta-bar,
    div:has(> #resource_display) .eta-bar {
        display: none !important;
    }
"""

#: Each page's own style sheet (the page contract, `src/webui/shell.py`): a page
#: module's ``PAGE_CSS``, listed here beside its builder in `PAGES`.
PAGE_CSS = (
    datasets_page.PAGE_CSS,
    diagnosis_panel.PAGE_CSS,
    test_page.PAGE_CSS,
    models_page.PAGE_CSS,
    training_console.PAGE_CSS,
    finetune_page.PAGE_CSS,
    runtime_settings.PAGE_CSS,
)


def dashboard_head(page_css: tuple = PAGE_CSS) -> str:
    """The page's head (M5-2), given to the mount (`mount_options`):

    - **one ``<style>`` element: the shared ⓘ's, the shell's and every page's
      style sheet** (`PAGE_CSS`). Gradio 6.30 rewrites the mount's ``css=``:
      inside an ``@media`` rule it keeps only a copy of each selector prefixed
      with ``.gradio-container … .contain``, so a media query there cannot
      reach the page frame above ``.contain`` (the one-screen layout needs
      ``body`` and Gradio's containers), and it drops ``@supports`` and
      ``@container`` rules. A style element in the head is applied as written;
    - **the head scripts**: the shared ⓘ (and Copy), the shell (always
      light, title, header row) and the touch record's (every edit of a
      field whose touches a form records is that field's input, an emptied
      text box included: `training_console.TOUCH_HEAD`). They run once per
      page load, before any component exists, so each works by
      document-level delegation.
    """
    return ("<style>" + HELP_CSS + SHELL_CSS + "".join(page_css) + "</style>"
            + HELP_HEAD + shell.SHELL_HEAD + training_console.TOUCH_HEAD)


DASHBOARD_HEAD = dashboard_head()

#: Text sizes of the theme: nothing under 14 px, body 15 px (the approved look).
TEXT_SIZE = gr.themes.Size(name="shepherd", xxs="14px", xs="14px", sm="14px", md="15px",
                           lg="16px", xl="20px", xxl="24px")


def _dashboard_theme() -> gr.themes.Base:
    """The Soft theme with IBM Plex (the font files are declared in DASHBOARD_CSS)."""
    return gr.themes.Soft(
        font=["IBM Plex Sans"],
        font_mono=["IBM Plex Mono"],
        text_size=TEXT_SIZE,
    )


#: Each page's builder, by the shell's page ids (the page contract; a page's
#: style sheet goes in `PAGE_CSS`).
PAGES = {
    shell.CLINICAL: create_diagnosis_tab,
    shell.DATASETS: create_datasets_page,
    shell.TRAINING: create_training_tab,
    shell.FINETUNE: finetune_page.create_finetune_page,
    shell.TEST: test_page.create_test_page,
    shell.MODELS: models_page.create_models_page,
    shell.SETTINGS: create_runtime_settings_tab,
}


def create_gradio_app() -> gr.Blocks:
    """
    Create the SHEPHERD-Advanced Gradio dashboard.

    Returns:
        gr.Blocks application instance. Mount it with `mount_dashboard`, which
        passes the theme, CSS, head script and switches (`mount_options`).
    """
    with gr.Blocks(
        # The first screen's title; the head script follows the location.
        title=shell.TITLES[shell.CLINICAL],
        # No usage analytics and no version check (owner, 2026-10-12). Left to
        # Gradio's default, building this app started a version check against
        # api.gradio.app and an analytics request through huggingface_hub, and
        # the API builds it at import, so every start of the service tried to
        # reach both. This is the one place the dashboard is built. It switches
        # off only Gradio's own requests, not every Hugging Face request.
        analytics_enabled=False,
        # The layout follows the browser's usable width, not a fixed cap.
        fill_width=True,
        # Theme and CSS are not set here: Gradio 6 drops them from a mounted
        # Blocks, with only a warning. `mount_dashboard` passes them.
    ) as app:
        registry = HelpRegistry()
        shell.build_shell(app, PAGES, help_registry=registry)
        # Every help text, once, escaped: after every page has registered its own.
        shell.help_texts(registry)

        # Runtime stack, always visible. Two jobs: say which torch/CUDA/PyG
        # combination is actually running, and make a degraded environment
        # impossible to miss. A deployment can lose its GNN without any error
        # appearing here, so absence has to be shown, not merely not-shown.
        shell.plain_html(_runtime_footer(), elem_id="runtime_footer")

    return app


#: The largest file Gradio's upload route accepts, per file (M5-1): a second
#: line of defence only. The dashboard has no upload control: its files go the
#: other way (exports), and import sources come from the server's intake
#: directory, never a browser (D7). Gradio still registers the route, and CORS
#: does not stop a cross-origin multipart POST from reaching it (a "simple"
#: request needs no preflight), so the service refuses the route in front of
#: the mount before its body is read (`RefuseUnusedGradioRoutes`). This limit
#: bounds each file only, never how many a request carries (Gradio takes up to
#: 1000) or how many requests come, so it is not what stops such writes; it
#: holds should the route be reached, say on a mount without that guard. A
#: larger file is refused with 413 while it streams.
MAX_UPLOAD_SIZE = "1mb"


def mount_options() -> dict[str, Any]:
    """Everything `gr.mount_gradio_app` is given besides the app, the Blocks and the path.

    The look and the page's scripts: theme, CSS and ``head``
    (`DASHBOARD_HEAD`: every style sheet, and the scripts of the shared ⓘ,
    the shell and the touch record, `training_console.TOUCH_HEAD`); there is
    no ``js``. All of them belong here: Gradio 6 drops them from
    ``gr.Blocks``.

    The switches, each off whatever the environment says, because left to
    Gradio a ``GRADIO_*`` variable can turn it on:

    - ``run_history``: Gradio 6 records each event's inputs and outputs in the
      browser by default and offers to save them to a Hugging Face bucket.
      Diagnosis inputs are a patient's phenotypes; they are not kept in the
      browser nor offered to a cloud store. Off, the run-history page answers
      404 and its other routes are not registered.
    - ``enable_monitoring``: only ``False`` refuses ``/monitoring/summary``
      (left at None, it serves the queue's event analytics).
    - ``mcp_server``: would publish every event handler as an MCP tool.
    - ``ssr_mode``: server-side rendering starts a Node.js server process.
    - ``max_file_size``: `MAX_UPLOAD_SIZE` (below).
    - ``footer_links``: none. Gradio's footer offers its API page, "Built with
      Gradio" and its own "Settings" (a theme switch with a dark mode the
      dashboard's colours do not have, and screen recording), beside the
      dashboard's ⚙ Settings; it also takes height from the one-screen layout.

    Two routes Gradio registers with no switch at all are refused in front of
    the mount: ``src/api/middleware/gradio_routes.py``.
    """
    return {
        "theme": _dashboard_theme(),
        "css": DASHBOARD_CSS,
        "head": DASHBOARD_HEAD,
        "run_history": False,
        "enable_monitoring": False,
        "mcp_server": False,
        "ssr_mode": False,
        "max_file_size": MAX_UPLOAD_SIZE,
        "footer_links": [],
    }


def mount_dashboard(app: FastAPI, path: str = "/ui") -> gr.Blocks:
    """Build the dashboard and mount it on ``app`` at ``path``; return the Blocks."""
    blocks = create_gradio_app()
    gr.mount_gradio_app(app, blocks, path=path, **mount_options())
    return blocks


__all__ = ["DASHBOARD_CSS", "DASHBOARD_HEAD", "MAX_UPLOAD_SIZE", "PAGES", "PAGE_CSS",
           "create_gradio_app", "dashboard_head", "mount_dashboard", "mount_options"]
