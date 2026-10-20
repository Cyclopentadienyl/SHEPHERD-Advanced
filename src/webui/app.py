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
with Datasets · Training · Test · Models, and "⚙ Settings"; the header's chips;
the shared timers; and gives each page builder the page context (the page
contract is the shell's docstring). The runtime line stays at the bottom.

Dependencies:
    - gradio==6.30.0 (pinned in pyproject.toml). From Gradio 6 a mounted app
      takes its theme, CSS, head and JS from gr.mount_gradio_app(), not from
      gr.Blocks(); `mount_dashboard` is the one place that passes them
      (`mount_options`: the theme, the CSS and the one head script of the
      shared ⓘ and the shell).

Called by:
    - src.api.main (`mount_dashboard`, at /ui)

Version: 1.0.0
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gradio as gr

from src.utils.version_checker import DEGRADED, NOTICE, format_runtime_line, probe_runtime
from src.webui import shell
from src.webui.components.datasets_page import create_datasets_page
from src.webui.components.diagnosis_panel import create_diagnosis_tab
from src.webui.components.help import HELP_CSS, HELP_HEAD, HelpRegistry
from src.webui.components.runtime_settings import create_runtime_settings_tab
from src.webui.components.training_console import create_training_tab
from src.webui.style import SHELL_CSS

if TYPE_CHECKING:
    from fastapi import FastAPI


def _runtime_footer() -> str:
    """Markdown for the always-visible runtime line.

    Colour carries the severity so it reads at a glance: red when a capability
    is gone (torch or PyG will not import — GNN scoring cannot run), amber for
    performance-only or optional gaps, plain grey when everything is present.

    Wrapped: this footer exists to report a broken environment, so it must not
    itself break in one. Its own failure is rendered rather than raised.
    """
    try:
        report = probe_runtime()
        line = format_runtime_line(report)
        status = report.get("status")
        issues = report.get("issues", [])
    except Exception as exc:  # noqa: BLE001 — a footer that raises reports nothing
        return (
            '<span style="color:#d32f2f"><b>Runtime status unavailable</b> — '
            f"could not probe the environment: {exc}</span>"
        )

    if status == DEGRADED:
        detail = " ".join(issues)
        return (
            f'<span style="color:#d32f2f"><b>⛔ Degraded runtime</b> — {line}<br>'
            f"{detail}</span>"
        )
    if status == NOTICE:
        detail = " ".join(issues)
        # One line, at the page's size: text is never under 14 px (M5-2).
        return (
            f'<span style="color:#b45309"><b>⚠️ {line}</b> — {detail}</span>'
        )
    return f'<span style="color:#888">{line}</span>'


#: The dashboard's CSS: the IBM Plex font files (served by the API at /static,
#: so the page needs no network), the polled panels' loading flash, and the
#: Runtime Settings badges. Passed to the mount (`mount_dashboard`).
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

    /* Suppress Gradio's opacity/loading flash on polled components */
    #resource_display,
    #resource_display *,
    #status_display,
    #status_display *,
    div:has(> #resource_display),
    div:has(> #status_display) {
        animation: none !important;
        transition: none !important;
        opacity: 1 !important;
    }
    #resource_display.pending,
    #resource_display.generating,
    #status_display.pending,
    #status_display.generating,
    div:has(> #resource_display).pending,
    div:has(> #resource_display).generating,
    div:has(> #status_display).pending,
    div:has(> #status_display).generating {
        opacity: 1 !important;
        pointer-events: auto !important;
    }
    #resource_display .eta-bar,
    #status_display .eta-bar,
    div:has(> #resource_display) .eta-bar,
    div:has(> #status_display) .eta-bar {
        display: none !important;
    }

    /* Runtime Settings — coloured impact/scope pill badges */
    .shep-badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin: 2px 0 8px;
    }
    .shep-badge {
        display: inline-flex;
        align-items: center;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 14px;
        font-weight: 600;
        line-height: 1.55;
        letter-spacing: 0.01em;
        white-space: nowrap;
        border: 1px solid transparent;
    }
    .shep-badge-mem     { background: rgba(37,99,235,0.12);  color: #1d4ed8; border-color: rgba(37,99,235,0.30); }
    .shep-badge-speed   { background: rgba(22,163,74,0.12);  color: #15803d; border-color: rgba(22,163,74,0.30); }
    .shep-badge-prec    { background: rgba(217,119,6,0.14);  color: #b45309; border-color: rgba(217,119,6,0.34); }
    .shep-badge-exp     { background: rgba(147,51,234,0.12); color: #7e22ce; border-color: rgba(147,51,234,0.30); }
    .shep-badge-restart { background: rgba(220,38,38,0.12);  color: #b91c1c; border-color: rgba(220,38,38,0.30); }
    .shep-badge-nextrun { background: rgba(100,116,139,0.14);color: #475569; border-color: rgba(100,116,139,0.32); }

    /* Restart-locked warning (training in progress) */
    .shep-restart-locked {
        color: #dc2626;
        font-weight: 600;
        font-size: 14px;
    }
"""

#: The page's head (M5-2), given to the mount (`mount_options`):
#:
#: - **the shell's and the shared ⓘ's style sheet, as a ``<style>`` element**.
#:   Gradio 6.30 rewrites the mount's ``css=``: inside an ``@media`` rule it
#:   keeps only a copy of each selector prefixed with ``.gradio-container …
#:   .contain``, so a media query there cannot reach the page frame above
#:   ``.contain`` (the one-screen layout needs ``body`` and Gradio's
#:   containers), and it drops ``@supports`` and ``@container`` rules. A style
#:   element in the head is applied as written;
#: - **the one head script**: the shared ⓘ (and Copy) and the shell (title,
#:   header row). It runs once per page load, before any component exists, so
#:   both work by document-level delegation.
DASHBOARD_HEAD = ("<style>" + HELP_CSS + SHELL_CSS + "</style>"
                  + HELP_HEAD + shell.SHELL_HEAD)

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


#: Each page's builder, by the shell's page ids (the page contract). Test and
#: Models are added by UI-D; until then the shell shows their temporary body.
PAGES = {
    shell.CLINICAL: create_diagnosis_tab,
    shell.DATASETS: create_datasets_page,
    shell.TRAINING: create_training_tab,
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
        gr.Markdown(_runtime_footer(), elem_id="runtime_footer")

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

    The look: theme, CSS and ``head`` (the shared ⓘ and the shell's script,
    `DASHBOARD_HEAD`); there is no ``js``. All of them belong here: Gradio 6
    drops them from ``gr.Blocks``.

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
    }


def mount_dashboard(app: FastAPI, path: str = "/ui") -> gr.Blocks:
    """Build the dashboard and mount it on ``app`` at ``path``; return the Blocks."""
    blocks = create_gradio_app()
    gr.mount_gradio_app(app, blocks, path=path, **mount_options())
    return blocks


__all__ = ["DASHBOARD_CSS", "DASHBOARD_HEAD", "MAX_UPLOAD_SIZE", "PAGES", "create_gradio_app",
           "mount_dashboard", "mount_options"]
