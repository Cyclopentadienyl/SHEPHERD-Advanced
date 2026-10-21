"""
The dashboard's look: tokens, the shell, the pages UI-A draws (M5-2).
=====================================================================
The approved look of the M5 mockups (round 2): IBM Plex Sans, an indigo
accent, 15 px body text and **nothing under 14 px**; two navigation levels with
two selection cues each; a third style for in-page segmented controls
(``.shep-seg``); the layout rule (wide: one screen, regions scroll inside;
narrow: everything flows, nothing scrolls inside).

This style sheet goes into the page's head as a ``<style>`` element, with the
shared ⓘ's and each page's own ``PAGE_CSS`` (`src/webui/app.py`,
`dashboard_head`): Gradio 6 drops CSS given to ``gr.Blocks`` and rewrites the
mount's ``css=`` inside ``@media``. What is here is the shell and the pieces
every page shares (panels, tags, tables, the model list); a rule for one page
lives in that page's module. Gradio's own markup is reached only through the
``elem_id``\\ s and classes the shell sets, the tab roles (``[role="tab"]``,
``aria-selected``) and the wrapper classes Gradio documents for styling
(``.tab-wrapper``, ``.tab-container``, ``.block``).

The page is always light (the shell's head script keeps Gradio's dark mode
off): the tokens below have no dark values.

Module: src/webui/style.py
"""
from __future__ import annotations

#: Width (CSS px) from which the brand is drawn over the level-1 tab row.
BRAND_WIDTH = 190

SHELL_CSS = """
:root {
  --shep-bg: #f3f4f7; --shep-panel: #ffffff; --shep-panel-2: #f8f9fb;
  --shep-border: #e1e4ea; --shep-border-2: #cdd2da;
  --shep-text: #1d2230; --shep-text-2: #434a58; --shep-muted: #5b6270;
  --shep-accent: #5558e3; --shep-accent-600: #4447cf; --shep-accent-50: #eef0ff;
  --shep-accent-100: #dfe2fe; --shep-accent-text: #3a3dbd;
  --shep-amber-50: #fff7e3; --shep-amber-100: #fdeab8; --shep-amber-300: #f0c252;
  --shep-amber-600: #c27c00; --shep-amber-800: #7a4600;
  --shep-green-50: #edf9f1; --shep-green-200: #a9dfbb; --shep-green-800: #126133;
  --shep-grey-50: #f2f3f5; --shep-grey-200: #d9dce2; --shep-grey-600: #535a66; --shep-grey-800: #2f3540;
  --shep-red-50: #fdf0f0; --shep-red-200: #f1b9b9; --shep-red-700: #a52222;
  --shep-mono: 'IBM Plex Mono', 'DejaVu Sans Mono', 'Liberation Mono', monospace;
  --shep-row1-h: 54px; --shep-brand-w: __BRAND__px;
}

/* ---------------------------------------------------------------- page frame */
:root { color-scheme: light; }
body, gradio-app, .gradio-container { background: var(--shep-bg) !important; }
body { color: var(--shep-text); }
.gradio-container { font-size: 15px; color: var(--shep-text); }
/* The header runs edge to edge; each page pads itself. */
.gradio-container > .main { padding: 0 !important; }
.gradio-container > .main > .wrap, .gradio-container .contain { padding: 0 !important; max-width: none !important; }
.gradio-container sub, .gradio-container small { font-size: 14px; vertical-align: baseline; }
.gradio-container code { font-family: var(--shep-mono) !important; font-size: 14px; }
#shep-shell { position: relative; gap: 0 !important; background: var(--shep-bg); }

/* ---------------------------------------------------------------- row 1: brand, stage switch, chips, Settings */
#shep-brand {
  position: absolute; top: 0; left: 16px; height: var(--shep-row1-h); z-index: 4;
  display: flex; align-items: center; pointer-events: none; padding: 0 !important;
  width: auto !important; min-width: 0 !important; border: 0 !important; background: none !important;
}
.shep-brand { font-size: 17px; font-weight: 700; letter-spacing: -0.01em; white-space: nowrap; color: var(--shep-text); }
#shep-chips {
  position: absolute; top: 0; right: calc(var(--shep-settings-w, 120px) + 32px); z-index: 4;
  height: var(--shep-row1-h); display: flex; align-items: center; justify-content: flex-end;
  padding: 0 !important; width: auto !important; min-width: 0 !important; border: 0 !important;
  background: none !important; max-width: calc(100% - var(--shep-settings-w, 120px) - 48px);
}
.shep-chips { display: flex; flex-wrap: wrap; justify-content: flex-end; align-items: center; gap: 6px 10px; }
.shep-chip {
  display: inline-flex; align-items: center; gap: 5px; height: 28px; padding: 0 4px 0 10px; border-radius: 999px;
  font-size: 14px; line-height: 1; white-space: nowrap; border: 1px solid var(--shep-border-2);
  background: #fff; color: var(--shep-text-2); max-width: 100%;
}
.shep-chip-text { overflow: hidden; text-overflow: ellipsis; max-width: 360px; }
.shep-chip button.shep-i { width: 18px; height: 18px; margin: 0 0 0 2px; border-color: currentColor; color: inherit; background: transparent; }
.shep-chip-on { background: var(--shep-accent-50); border-color: var(--shep-accent-100); color: var(--shep-accent-text); }
.shep-chip-readonly, .shep-chip-unread { background: var(--shep-grey-50); border-color: var(--shep-grey-200); color: var(--shep-grey-800); }
.shep-chip-busy { background: var(--shep-accent-50); border-color: var(--shep-accent-100); color: var(--shep-accent-text); }
.shep-dot { width: 8px; height: 8px; border-radius: 50%; background: #8a909b; flex: none; }
.shep-chip-busy .shep-dot { background: var(--shep-accent); box-shadow: 0 0 0 3px var(--shep-accent-100); }
.shep-chip-unread .shep-dot { background: var(--shep-grey-600); }

#shep-nav > .tab-wrapper {
  height: auto !important; padding: 0 !important; margin: 0 !important;
  background: var(--shep-panel); border-bottom: 1px solid var(--shep-border);
}
#shep-nav > .tab-wrapper > .tab-container {
  height: auto !important; min-height: var(--shep-row1-h); background: none !important;
  padding: 8px 16px 8px calc(16px + var(--shep-brand-w) + 16px); align-items: center; overflow: visible;
}
#shep-nav > .tab-wrapper > .tab-container::after { display: none; }
#shep-nav > .tab-wrapper > .tab-container > button[role="tab"] {
  height: 36px; padding: 0 16px; border: 1px solid var(--shep-border-2); background: #fff;
  font-size: 15px; font-weight: 500; color: var(--shep-text-2); border-radius: 0; overflow: visible;
}
#shep-nav > .tab-wrapper > .tab-container > button[role="tab"]:first-of-type { border-radius: 9px 0 0 9px; }
#shep-nav > .tab-wrapper > .tab-container > button[role="tab"] + button[role="tab"] { border-left: 0; border-radius: 0 9px 9px 0; }
#shep-nav > .tab-wrapper > .tab-container > button[role="tab"][aria-selected="true"] {
  background: var(--shep-accent); border-color: var(--shep-accent); color: #fff; font-weight: 700;
}
#shep-nav > .tab-wrapper > .tab-container > button[role="tab"]::after { display: none; }
#shep-nav > .tab-wrapper > .tab-container > button[role="tab"]:hover:not([aria-selected="true"]) { background: var(--shep-panel-2); }
#shep-nav > .tab-wrapper .right-tab-group { flex-wrap: nowrap; }
#shep-nav > .tab-wrapper .right-tab-group > button[role="tab"] {
  height: 34px; padding: 0 12px; border: 1px solid transparent; border-radius: 8px;
  font-size: 15px; font-weight: 500; color: var(--shep-text-2); overflow: visible;
}
#shep-nav > .tab-wrapper .right-tab-group > button[role="tab"]::before,
#shep-nav > .tab-wrapper .right-tab-group > button[role="tab"]::after { display: none !important; }
#shep-nav > .tab-wrapper .right-tab-group > button[role="tab"][aria-selected="true"] {
  background: var(--shep-accent-50); border-color: var(--shep-accent-100); color: var(--shep-accent-text);
  font-weight: 700; box-shadow: inset 0 -3px 0 var(--shep-accent);
}
/* Not enough room for one row: the chips and Settings go onto a second line. */
#shep-shell.shep-nav-wrapped #shep-nav > .tab-wrapper > .tab-container { row-gap: 10px; padding-bottom: 0; }
#shep-shell.shep-nav-wrapped #shep-nav > .tab-wrapper .right-tab-group {
  flex-basis: 100%; justify-content: flex-end; min-height: max(44px, calc(var(--shep-chips-h, 28px) + 12px));
}
#shep-shell.shep-nav-wrapped #shep-chips {
  top: var(--shep-row1-h); height: max(44px, calc(var(--shep-chips-h, 28px) + 12px));
}

/* ---------------------------------------------------------------- row 2: the research pages */
#shep-research-nav > .tab-wrapper {
  height: auto !important; padding: 0 !important; margin: 0 !important;
  background: var(--shep-panel); border-bottom: 1px solid var(--shep-border);
}
#shep-research-nav > .tab-wrapper > .tab-container {
  height: auto !important; min-height: 44px; background: none !important; padding: 0 16px; overflow: visible;
}
#shep-research-nav > .tab-wrapper > .tab-container::after { display: none; }
#shep-research-nav > .tab-wrapper > .tab-container > button[role="tab"] {
  height: 44px; padding: 0 12px; font-size: 15px; font-weight: 500; color: var(--shep-text-2);
  border-top: 3px solid transparent; border-bottom: 3px solid transparent; overflow: visible;
}
#shep-research-nav > .tab-wrapper > .tab-container > button[role="tab"][aria-selected="true"] {
  color: var(--shep-accent-text); border-bottom-color: var(--shep-accent); font-weight: 700;
}
#shep-research-nav > .tab-wrapper > .tab-container > button[role="tab"]::after { display: none; }
/* Group labels: generated content with empty alternative text, not clickable. */
#shep-page-datasets-button { margin-left: 84px; }
#shep-page-test-button { margin-left: 116px; }
#shep-page-datasets-button::before, #shep-page-test-button::before {
  position: absolute; right: calc(100% + 6px); top: 0; bottom: 0; display: flex; align-items: center;
  font-size: 14px; font-weight: 600; letter-spacing: .07em; text-transform: uppercase;
  color: var(--shep-muted); pointer-events: none; white-space: nowrap;
}
#shep-page-datasets-button::before { content: "Prepare"; content: "Prepare" / ""; }
#shep-page-test-button::before {
  content: "Evaluate"; content: "Evaluate" / ""; border-left: 1px solid var(--shep-border-2);
  padding-left: 14px; margin: 10px 0; top: 0; bottom: 0;
}

/* ---------------------------------------------------------------- tab panels */
#shep-nav > .tabitem, #shep-research-nav > .tabitem { padding: 0 !important; border: 0 !important; background: none !important; }
#shep-nav > .tabitem > .column, #shep-research-nav > .tabitem > .column { gap: var(--layout-gap); }
#shep-tab-clinical > .column, #shep-tab-settings > .column, #shep-research-nav > .tabitem > .column { padding: 16px; }
.shep-temporary { padding: 16px; color: var(--shep-muted); }

/* ---------------------------------------------------------------- in-page segmented control (third style) */
.shep-seg .wrap { display: inline-flex !important; gap: 2px !important; padding: 3px; border: 1px solid var(--shep-grey-200); border-radius: 8px; background: var(--shep-grey-50); }
.shep-seg .wrap label { border: 1px solid transparent !important; border-radius: 6px !important; background: none !important; box-shadow: none !important; padding: 5px 12px !important; font-size: 15px; font-weight: 500; }
.shep-seg .wrap label { color: var(--shep-text-2) !important; }
.shep-seg .wrap label.selected { background: #fff !important; border-color: var(--shep-grey-200) !important; box-shadow: 0 1px 2px rgba(20,24,40,.12) !important; font-weight: 600; color: var(--shep-text) !important; }
.shep-seg .wrap label span { color: inherit !important; }
.shep-seg .wrap label input { position: absolute; opacity: 0; width: 1px; height: 1px; }

/* ---------------------------------------------------------------- primary buttons */
/* White text on every primary button at a contrast of at least 4.5:1, at rest
   and on hover (the Soft theme's indigo-500 is 4.47:1 and its hover indigo-400
   2.98:1). A disabled primary button is grey with grey text, never a faded
   accent: it reads as off, and its words stay readable. */
.gradio-container {
  --button-primary-background-fill: var(--shep-accent);
  --button-primary-background-fill-hover: var(--shep-accent-600);
  --button-primary-border-color: var(--shep-accent);
  --button-primary-border-color-hover: var(--shep-accent-600);
  --button-primary-text-color: #ffffff;
  --button-primary-text-color-hover: #ffffff;
}
.gradio-container button.primary[disabled] {
  background: var(--shep-grey-50) !important; border-color: var(--shep-grey-200) !important;
  color: var(--shep-grey-600) !important; opacity: 1 !important; filter: none !important; box-shadow: none !important;
}

/* ---------------------------------------------------------------- shared pieces */
.shep-muted { color: var(--shep-muted); }
.shep-note { font-size: 14px; color: var(--shep-muted); }
.shep-note p { font-size: 14px; }
.shep-sep { color: var(--shep-border-2); }
.shep-nowrap { white-space: nowrap; }
.shep-w600 { font-weight: 600; }
.shep-sub { font-size: 14px; color: var(--shep-text-2); }
.shep-num { text-align: right !important; font-variant-numeric: tabular-nums; }
.shep-vh { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
.shep-fp { white-space: nowrap; }
.shep-fpk { color: var(--shep-text-2); }
.shep-fpv { font-family: var(--shep-mono) !important; font-weight: 700; color: var(--shep-text); font-size: 14px; }
.shep-full { font-family: var(--shep-mono) !important; font-size: 14px; overflow-wrap: anywhere; word-break: break-all; }
.shep-tag {
  display: inline-flex; align-items: center; gap: 4px; height: 24px; padding: 0 8px; border-radius: 999px;
  font-size: 14px; white-space: nowrap; border: 1px solid var(--shep-border-2); background: #fff; color: var(--shep-text-2);
  vertical-align: middle;
}
.shep-tag-amber { background: var(--shep-amber-50); border-color: var(--shep-amber-300); color: var(--shep-amber-800); }
.shep-tag-ok { background: var(--shep-green-50); border-color: var(--shep-green-200); color: var(--shep-green-800); }
.shep-tag-grey { background: var(--shep-grey-50); border-color: var(--shep-grey-200); color: var(--shep-grey-800); }
.shep-tag-accent { background: var(--shep-accent-50); border-color: var(--shep-accent-100); color: var(--shep-accent-text); }
.shep-tag-red { background: var(--shep-red-50); border-color: var(--shep-red-200); color: var(--shep-red-700); }
.shep-ok-t { color: var(--shep-green-800); display: inline-flex; align-items: center; gap: 3px; }
.shep-amber-t { color: var(--shep-amber-800); }
.shep-reason { display: flex; gap: 8px; align-items: flex-start; font-size: 14px; color: var(--shep-text-2); line-height: 1.45; }
.shep-reason-grey { border: 1px solid var(--shep-grey-200); background: var(--shep-grey-50); border-radius: 8px; padding: 10px 12px; color: var(--shep-grey-800); }
.shep-notice { border-radius: 6px; padding: 8px 10px; font-size: 14px; }
.shep-notice-amber { border: 1px solid var(--shep-amber-300); border-left: 4px solid var(--shep-amber-600); background: var(--shep-amber-50); color: var(--shep-amber-800); }
.shep-warn { color: var(--shep-amber-800); margin: 0 0 6px; }
.shep-mark { display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 14px; line-height: 1.4; margin: 1px 4px 1px 0; }
.shep-mark-serve { background: var(--shep-accent-50); border: 1px solid var(--shep-accent-100); color: var(--shep-accent-text); font-weight: 500; }
.shep-mark-serve-same { background: #fff; border: 1px dashed var(--shep-accent-100); color: var(--shep-accent-text); }
.shep-mark-outline { background: #fff; border: 1px solid var(--shep-border-2); color: var(--shep-text-2); }
.shep-mark-grey { background: var(--shep-grey-50); border: 1px solid var(--shep-grey-200); color: var(--shep-grey-800); }
.shep-btn { height: 32px; padding: 0 12px; border-radius: 8px; border: 1px solid var(--shep-border-2); background: #fff; font-size: 14px; cursor: pointer; }

/* panels */
.shep-panel { background: var(--shep-panel) !important; border: 1px solid var(--shep-border) !important; border-radius: 10px !important; padding: 0 !important; gap: 0 !important; overflow: hidden; flex-wrap: nowrap !important; }
.shep-panel > .block, .shep-panel > .form { border: 0 !important; box-shadow: none !important; }
.shep-panel-hd { display: flex; align-items: center; gap: 10px; padding: 11px 16px; flex-wrap: wrap; border-bottom: 1px solid var(--shep-border); min-height: 48px; }
.shep-panel-hd h2 { font-size: 16px; font-weight: 600; margin: 0; }
.shep-panel-right { margin-left: auto; color: var(--shep-text-2); font-size: 14px; }
.shep-panel-bd { padding: 14px 16px; display: flex; flex-direction: column; gap: 12px; }
.shep-panel-ft { padding: 12px 16px !important; border-top: 1px solid var(--shep-border); align-items: center !important; gap: 12px !important; flex-wrap: wrap !important; }
.shep-intro { padding: 12px 16px 0; margin: 0; font-size: 14px; color: var(--shep-text-2); }
.shep-card { border: 1px solid var(--shep-border); border-radius: 8px; background: #fff; }
.shep-card-hd .shep-muted { font-weight: 400; font-size: 14px; }
.shep-card-hd { padding: 9px 12px; border-bottom: 1px solid var(--shep-border); font-weight: 600; font-size: 15px; }
.shep-kv { display: grid; grid-template-columns: max-content 1fr; gap: 6px 14px; font-size: 15px; align-items: baseline; }
.shep-card .shep-kv { padding: 10px 12px; }
.shep-kv-2 { grid-template-columns: max-content max-content max-content max-content; }
.shep-k { color: var(--shep-text-2); }
.shep-digest { display: grid; grid-template-columns: 1fr auto; gap: 4px 12px; align-items: center; }
.shep-digest .shep-dl { grid-column: 1 / -1; font-size: 14px; font-weight: 600; color: var(--shep-text-2); }

/* HTML the dashboard renders itself (`shell.plain_html`): no Gradio padding. */
.shep-html > .html-container { padding: 0 !important; }

/* tables */
.shep-tbl-wrap { border: 1px solid var(--shep-border); border-radius: 8px; overflow: auto; margin: 12px 16px; }
.shep-tbl { border-collapse: separate !important; border-spacing: 0; width: 100%; margin: 0 !important; padding: 0 !important; }
.shep-tbl th { position: sticky; top: 0; z-index: 1; background: var(--shep-panel-2); font-size: 14px; font-weight: 600; color: var(--shep-text-2); text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--shep-border); }
.shep-tbl td { padding: 8px 10px; border-bottom: 1px solid #edf0f3; font-size: 15px; vertical-align: top; line-height: 1.35; }
.shep-tbl tr:last-child td { border-bottom: 0; }
.shep-tbl tbody tr[data-key] { cursor: pointer; }
.shep-tbl tbody tr[data-key]:hover td { background: var(--shep-panel-2); }
.shep-tbl tbody tr.shep-on td { background: var(--shep-accent-50); }
.shep-tbl tbody tr.shep-on td:first-child { box-shadow: inset 4px 0 0 var(--shep-accent); }
.shep-tbl tbody tr[data-key]:focus-visible { outline: 2px solid var(--shep-accent); outline-offset: -2px; }
.shep-tbl tr.shep-grp th { position: static; background: var(--shep-panel-2); font-size: 14px; font-weight: 600; color: var(--shep-text-2); padding: 6px 10px; border-bottom: 1px solid var(--shep-border); }
.shep-tbl tbody tr.shep-row-dis td { background: var(--shep-grey-50); color: var(--shep-muted); cursor: default; }

/* the shared model list */
.shep-list { display: flex; flex-direction: column; border: 1px solid var(--shep-border); border-radius: 8px; overflow: hidden; }
.shep-list-hd { padding: 6px 12px; background: var(--shep-panel-2); border-bottom: 1px solid var(--shep-border); font-size: 14px; color: var(--shep-text-2); }
.shep-list-note { padding: 6px 12px; font-size: 14px; color: var(--shep-muted); border-bottom: 1px solid #edf0f3; }
button.shep-row {
  display: flex; flex-direction: column; align-items: flex-start; gap: 2px; text-align: left; width: 100%;
  padding: 8px 12px; border: 0; border-bottom: 1px solid #edf0f3; background: #fff; font: inherit; color: inherit; cursor: pointer;
}
.shep-row:hover { background: var(--shep-panel-2); }
.shep-row-on { background: var(--shep-accent-50) !important; box-shadow: inset 4px 0 0 var(--shep-accent); }
.shep-row-dis { background: var(--shep-grey-50); color: var(--shep-muted); cursor: default; }
.shep-row-main { font-weight: 600; }
.shep-row-sub { font-size: 14px; color: var(--shep-text-2); }
.shep-row-more { align-items: center; color: var(--shep-accent-text); font-weight: 500; }
.shep-mtbl-wrap { margin: 0; }
.shep-mtbl td { overflow-wrap: anywhere; }
.shep-mtbl-wrap + .shep-row-more { border: 1px solid var(--shep-border); border-top: 0; border-radius: 0 0 8px 8px; }
.shep-facts-title { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; font-size: 16px; }
.shep-facts { display: flex; flex-direction: column; gap: 8px; }
.shep-facts-sec { border: 1px solid var(--shep-border); border-radius: 8px; background: #fff; }
.shep-facts-sec > summary, .shep-facts-sec > h4 { padding: 8px 12px; font-weight: 600; font-size: 15px; cursor: pointer; margin: 0; }
.shep-facts-sec > h4 { cursor: default; }
.shep-facts-n { color: var(--shep-muted); margin-right: 4px; }
.shep-facts-sum { font-weight: 400; color: var(--shep-text-2); font-size: 14px; }
.shep-facts-bd { padding: 4px 12px 10px; }
.shep-facts-warn { border-color: var(--shep-amber-300); background: var(--shep-amber-50); }
.shep-confirm { border: 1px solid var(--shep-accent-100); background: #fbfbff; border-radius: 10px; padding: 12px 14px; }
.shep-confirm-q { font-size: 16px; font-weight: 600; margin: 4px 0 10px; }
.shep-ftbl { border-collapse: separate; border-spacing: 0; width: 100%; max-width: 1080px; border: 1px solid var(--shep-accent-100); border-radius: 8px; background: #fff; }
.shep-ftbl th, .shep-ftbl td { padding: 7px 10px; border-bottom: 1px solid #edf0f3; font-size: 15px; text-align: left; vertical-align: top; }
.shep-ftbl thead th { font-size: 14px; color: var(--shep-text-2); background: var(--shep-panel-2); }
.shep-ftbl tr.shep-changed th { box-shadow: inset 4px 0 0 var(--shep-accent); }
.shep-ftbl tr.shep-changed td:last-child { font-weight: 600; }
.shep-confirm-note { font-size: 14px; color: var(--shep-text-2); margin: 10px 0 4px; }
.shep-outcome { border-radius: 8px; padding: 10px 12px; font-size: 15px; }
.shep-outcome-ok { background: var(--shep-green-50); border: 1px solid var(--shep-green-200); color: var(--shep-green-800); }
.shep-outcome-amber { background: var(--shep-amber-50); border: 1px solid var(--shep-amber-300); color: var(--shep-amber-800); }
.shep-outcome-grey { background: var(--shep-grey-50); border: 1px solid var(--shep-grey-200); color: var(--shep-grey-800); }
.shep-if-waited { display: none; }
.shep-waited .shep-if-waited { display: inline; }

/* ---------------------------------------------------------------- the runtime line */
#runtime_footer { padding: 4px 16px !important; font-size: 14px; }
#runtime_footer p { margin: 0; }

/* ---------------------------------------------------------------- the layout rule */
/* Wide (>= 1400 x 760 CSS px): one screen. The selected page fills the space
   under the header; a page taller than that scrolls inside its own panel. */
@media (min-width: 1400px) and (min-height: 760px) {
  html, body { height: 100%; overflow: hidden; }
  gradio-app { height: 100vh; display: flex; flex-direction: column; }
  .gradio-container { height: 100vh; min-height: 0 !important; display: flex !important; flex-direction: column; }
  .gradio-container > .main { flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column; }
  .gradio-container > .main > .wrap { flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column; }
  .gradio-container .contain { flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column; }
  .gradio-container .contain > .column { flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column; gap: 0 !important; }
  #shep-shell { flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column; }
  #shep-nav { flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column; }
  #shep-nav > .tabitem { flex: 1 1 auto; min-height: 0; overflow: auto; }
  #shep-tab-research { overflow: hidden !important; }
  #shep-tab-research > .column { flex: 1 1 auto; min-height: 0; padding: 0 !important; display: flex; flex-direction: column; }
  #shep-research-nav { flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column; }
  #shep-research-nav > .tabitem { flex: 1 1 auto; min-height: 0; overflow: auto; }
  #shep-research-nav > .tabitem > .column { min-height: 100%; }
  .shep-page { min-height: 0; height: 100%; align-items: stretch !important; flex-wrap: nowrap !important; }
  .shep-page > .shep-panel { min-height: 0; max-height: 100%; display: flex; flex-direction: column; }
  .shep-page > .shep-panel > .block { flex: none; }
  footer, #runtime_footer { flex: none; }
}
/* Narrow: the regions stack, the page scrolls, nothing scrolls inside. */
@media (max-width: 1399px), (max-height: 759px) {
  .shep-page { flex-direction: column !important; }
  .shep-page > .shep-panel { width: 100% !important; }
  .shep-tbl-wrap { overflow: visible; }
  .shep-tbl th { position: static; }
  .shep-tbl td { overflow-wrap: anywhere; }
}
""".replace("__BRAND__", str(BRAND_WIDTH))

__all__ = ["BRAND_WIDTH", "SHELL_CSS"]
