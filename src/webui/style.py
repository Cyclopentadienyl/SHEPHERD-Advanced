"""
The dashboard's look: tokens, the shell, the pages UI-A draws (M5-2).
=====================================================================
The approved look of the M5 mockups (round 2): IBM Plex Sans, an indigo
accent, 15 px body text and **nothing under 14 px**; two navigation levels with
two selection cues each; a third style for in-page segmented controls
(``.shep-seg``); the layout rule (side by side: one screen, regions scroll
inside; stacked: everything flows, nothing scrolls inside but a table wider
than its panel, sideways).

**The layout rule is one decision** (D02, the WebUI ledger's UI-02): the
shell's head script (`src/webui/shell.py`, its layout pass) writes
``data-shep-layout="side"`` or ``"stack"`` on the root element, and every
style sheet keys its side-by-side and stacked rules off it, through `SIDE`
and `STACK` (``SIDE + " { … }"``, the rules nested inside), never a media
query on the window's size. The browser's usable area (the window's inner
size) decides it: no wider than it is tall, stacked; wider, side by side when
the page shown fits by its panels' own minimum widths (each panel's
``--shep-panel-min``, set in its page's style sheet; side by side, a panel is
never narrower), else stacked. A short window (`SHORT`, under
`SHORT_BELOW` px tall) is density only: it never stacks, shrinks text or
squeezes a chart. Before the first pass nothing is written, which reads as
stacked.

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

**Status colours** (the WebUI skill §6; K6 decision 2): ``--shep-status-ok``,
``-neutral``, ``-warn``, ``-error`` and ``-info`` are the colours of a status
mark and its word (the one indicator, `renderers.status_mark`, its
``.shep-st-*`` classes below), aliases of the existing tag colours (``.shep-tag-ok``,
``-grey``, ``-amber``, ``-red``, ``-accent``), so no new hex enters; a mark
always carries its shape and word too. `tests/unit/test_webui_button_contrast.py`
holds them to WCAG's 4.5:1 for text against every surface a mark sits on.

Module: src/webui/style.py
"""
from __future__ import annotations

#: Width (CSS px) the brand takes over the level-1 tab row: its name in the
#: body size and the strong weight (15 px, 600; the type scale's 20 px is a
#: region's title, and at 20 px the stage switch no longer fits beside it at
#: 420 px) is 151 px in IBM Plex Sans.
BRAND_WIDTH = 156

#: The root element's attribute the layout pass writes (``side`` or ``stack``,
#: module docstring), and the selectors every style sheet nests its rules in:
#: a rule inside `SIDE` holds side by side, inside `STACK` stacked (also
#: before the first pass). ``:where()`` adds no specificity, as a media query
#: added none, so a rule moved inside keeps its weight against the rules
#: around it. Nested rules start with a selector (``#id``, ``.class``) or
#: ``&`` (the root element itself; ``& body`` for an element name).
LAYOUT_ATTR = "data-shep-layout"
SIDE = ':where(html[data-shep-layout="side"])'
STACK = ':where(html:not([data-shep-layout="side"]))'
#: Set while the window's inner height is under `SHORT_BELOW` px: denser
#: headers, never stacking (the density rules of Training's run view and the
#: Diagnosis load confirmation).
SHORT_ATTR = "data-shep-short"
SHORT = ':where(html[data-shep-short])'
#: Side by side and short at once (a dense header beside the plots, the load
#: confirmation's room).
SIDE_SHORT = ':where(html[data-shep-layout="side"][data-shep-short])'
SHORT_BELOW = 900

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
  --shep-grey-50: #f2f3f5; --shep-grey-200: #d9dce2; --shep-grey-500: #737a87; --shep-grey-600: #535a66; --shep-grey-800: #2f3540;
  --shep-red-50: #fdf0f0; --shep-red-200: #f1b9b9; --shep-red-700: #a52222;
  --shep-status-ok: var(--shep-green-800); --shep-status-neutral: var(--shep-grey-800);
  --shep-status-warn: var(--shep-amber-800); --shep-status-error: var(--shep-red-700);
  --shep-status-info: var(--shep-accent-text);
  --shep-mono: 'IBM Plex Mono', 'DejaVu Sans Mono', 'Liberation Mono', monospace;
  --shep-row1-h: 46px; --shep-brand-w: __BRAND__px;
  /* The cues every navigation level and control reads (D01, UI-10): aliases
     of the colours above, no new hex. */
  --shep-tab-h: 32px; --shep-caption-h: 22px; --shep-group-gap: 32px;
  --shep-tab-hover-bg: var(--shep-grey-50); --shep-tab-hover-line: var(--shep-grey-500);
  --shep-tab-selected-line: var(--shep-accent); --shep-tab-selected-text: var(--shep-accent-text);
  --shep-focus-ring: var(--shep-accent-600); --shep-group-caption: var(--shep-muted);
  --shep-control-border: var(--shep-grey-500); --shep-control-border-hover: var(--shep-grey-600);
  --shep-disabled-bg: var(--shep-grey-50); --shep-disabled-border: var(--shep-grey-200);
  --shep-disabled-text: var(--shep-grey-600);
  /* The one type scale (the WebUI skill's §2): a region's title, body and
     values, keys and meta; two weights; one field-label style. */
  --shep-title-size: 20px; --shep-body-size: 15px; --shep-key-size: 14px;
  --shep-weight-strong: 600;
  --shep-label-size: 14px; --shep-label-weight: 600; --shep-label-color: var(--shep-text);
  /* A field row's two measures (D03, `.shep-field-row`): a select's or text box's box, 39 px and
     its 1 px border each side, and the room Gradio's padded block leaves under it. */
  --shep-field-box-h: 41px; --shep-field-pad-b: 10px;
  /* The room every scroll region keeps under what it brings into view (R-25): a line of a text
     box (21 px) and a little more. */
  --shep-caret-room: 32px;
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
/* Two weights (the type scale): bold text in rendered HTML and Markdown is the
   strong weight, never the browser's 700. */
.gradio-container b, .gradio-container strong, .gradio-container th { font-weight: var(--shep-weight-strong); }
#shep-shell { position: relative; gap: 0 !important; background: var(--shep-bg); }

/* ---------------------------------------------------------------- row 1: brand, stage switch, chips, Settings */
#shep-brand {
  position: absolute; top: 0; left: 16px; height: var(--shep-row1-h); z-index: 4;
  display: flex; align-items: center; pointer-events: none; padding: 0 !important;
  width: auto !important; min-width: 0 !important; border: 0 !important; background: none !important;
}
.shep-brand { font-size: var(--shep-body-size); font-weight: 600; letter-spacing: -0.01em; white-space: nowrap; color: var(--shep-text); }
#shep-chips {
  position: absolute; top: 0; right: calc(var(--shep-settings-w, 120px) + 32px); z-index: 4;
  height: var(--shep-row1-h); display: flex; align-items: center; justify-content: flex-end;
  padding: 0 !important; width: auto !important; min-width: 0 !important; border: 0 !important;
  background: none !important; max-width: calc(100% - var(--shep-settings-w, 120px) - 48px);
}
.shep-chips { display: flex; flex-wrap: wrap; justify-content: flex-end; align-items: center; gap: 6px 10px; }
/* The header's chips (R-21; the WebUI skill's §6, normal states are quiet): a
   normal state ("Operator controls: on", "GPU jobs: none") is plain text with
   its mark, no box; only a state to watch is a lozenge: read-only amber with
   the lock, a GPU job running in the info colour, a state not read grey. The
   marks are the status tokens: on a filled green dot, none and not read a
   hollow grey ring, a job the info colour. */
.shep-chip {
  display: inline-flex; align-items: center; gap: 5px; height: 28px; padding: 0 2px; border-radius: 999px;
  font-size: 14px; line-height: 1; white-space: nowrap; border: 1px solid transparent;
  background: none; color: var(--shep-text-2); max-width: 100%;
}
/* A chip never runs off the window (at 360 px the D8 chip's left edge was at -41): its words end
   in an ellipsis inside the room the header leaves, their whole text as their title. */
.shep-chip-text { overflow: hidden; text-overflow: ellipsis; max-width: 360px; min-width: 0; }
#shep-chips > div, #shep-chips > div > div { min-width: 0; max-width: 100%; }
.shep-chip button.shep-i { margin: 0 -3px 0 -1px; color: inherit; }
.shep-chip button.shep-i::before { inset: 3px; border-color: currentColor; background: transparent; }
.shep-chip-readonly, .shep-chip-unread, .shep-chip-busy { padding: 0 4px 0 10px; }
.shep-chip-readonly { background: var(--shep-amber-50); border-color: var(--shep-amber-300); color: var(--shep-amber-800); }
.shep-chip-unread { background: var(--shep-grey-50); border-color: var(--shep-grey-200); color: var(--shep-grey-800); }
.shep-chip-busy { background: var(--shep-accent-50); border-color: var(--shep-accent-100); color: var(--shep-accent-text); }
.shep-dot {
  width: 8px; height: 8px; border-radius: 50%; box-sizing: border-box; flex: none;
  border: 1.5px solid var(--shep-status-neutral); background: none;
}
.shep-chip-on .shep-dot { background: var(--shep-status-ok); border-color: var(--shep-status-ok); }
.shep-chip-busy .shep-dot { background: var(--shep-status-info); border-color: var(--shep-status-info); box-shadow: 0 0 0 3px var(--shep-accent-100); }

#shep-nav > .tab-wrapper {
  height: auto !important; padding: 0 !important; margin: 0 !important;
  background: var(--shep-panel); border-bottom: 1px solid var(--shep-border);
}
#shep-nav > .tab-wrapper > .tab-container {
  height: auto !important; min-height: var(--shep-row1-h); background: none !important;
  padding: 5px 16px 5px calc(16px + var(--shep-brand-w) + 16px); align-items: center; overflow: visible;
}
#shep-nav > .tab-wrapper > .tab-container::after { display: none; }
#shep-nav > .tab-wrapper > .tab-container > button[role="tab"] {
  height: 36px; padding: 0 16px; border: 1px solid var(--shep-border-2); background: #fff;
  font-size: 15px; font-weight: 400; color: var(--shep-text-2); border-radius: 0; overflow: visible;
}
#shep-nav > .tab-wrapper > .tab-container > button[role="tab"]:first-of-type { border-radius: 9px 0 0 9px; }
#shep-nav > .tab-wrapper > .tab-container > button[role="tab"] + button[role="tab"] { border-left: 0; border-radius: 0 9px 9px 0; }
#shep-nav > .tab-wrapper > .tab-container > button[role="tab"]::after { display: none; }
#shep-nav > .tab-wrapper .right-tab-group { flex-wrap: nowrap; }
#shep-nav > .tab-wrapper .right-tab-group > button[role="tab"] {
  height: 34px; padding: 0 12px; border: 1px solid transparent; border-radius: 8px;
  font-size: 15px; font-weight: 400; color: var(--shep-text-2); overflow: visible;
}
#shep-nav > .tab-wrapper .right-tab-group > button[role="tab"]::before,
#shep-nav > .tab-wrapper .right-tab-group > button[role="tab"]::after { display: none !important; }
/* Level 1's one selected look (R-20): the stage switch's segment and Settings
   alike are filled in the accent with white text, a look no other level has
   (level 2 is an underline, level 3 a raised white segment, a form's radio a
   tinted option with its dot). */
#shep-nav > .tab-wrapper > .tab-container > button[role="tab"][aria-selected="true"],
#shep-nav > .tab-wrapper .right-tab-group > button[role="tab"][aria-selected="true"] {
  background: var(--shep-accent); border-color: var(--shep-accent); color: #fff; font-weight: 600; box-shadow: none;
}
#shep-nav > .tab-wrapper button[role="tab"]:hover:not([aria-selected="true"]) {
  background: var(--shep-tab-hover-bg); color: var(--shep-text);
}
/* Keyboard focus on every tab: a 2 px ring from the focus token (level 1
   outside the tab, level 2 inside it, so its row never clips it). */
#shep-nav > .tab-wrapper button[role="tab"]:focus-visible,
#shep-research-nav > .tab-wrapper button[role="tab"]:focus-visible {
  outline: 2px solid var(--shep-focus-ring) !important; outline-offset: 2px; box-shadow: none;
}
#shep-research-nav > .tab-wrapper button[role="tab"]:focus-visible { outline-offset: -2px; }
/* Not enough room for one row: the chips and Settings go onto a second line. */
#shep-shell.shep-nav-wrapped #shep-nav > .tab-wrapper > .tab-container { row-gap: 10px; padding-bottom: 0; }
#shep-shell.shep-nav-wrapped #shep-nav > .tab-wrapper .right-tab-group {
  flex-basis: 100%; justify-content: flex-end; min-height: max(44px, calc(var(--shep-chips-h, 28px) + 12px));
}
#shep-shell.shep-nav-wrapped #shep-chips {
  top: var(--shep-chips-top, var(--shep-row1-h)); height: max(44px, calc(var(--shep-chips-h, 28px) + 12px));
}

/* ---------------------------------------------------------------- row 2: the research pages */
#shep-research-nav > .tab-wrapper {
  height: auto !important; padding: 0 !important; margin: 0 !important;
  background: var(--shep-panel); border-bottom: 1px solid var(--shep-border);
}
/* D01: each group's caption sits on a tier of its own above its tabs (the
   container's top padding), the tabs are 32 px, the groups are a group gap
   apart with a divider between them. */
#shep-research-nav > .tab-wrapper > .tab-container {
  height: auto !important; min-height: calc(var(--shep-caption-h) + var(--shep-tab-h)); background: none !important;
  padding: var(--shep-caption-h) 16px 0; row-gap: var(--shep-caption-h); align-items: flex-end; overflow: visible;
}
#shep-research-nav > .tab-wrapper > .tab-container::after { display: none; }
#shep-research-nav > .tab-wrapper > .tab-container > button[role="tab"] {
  height: var(--shep-tab-h); padding: 0 12px; font-size: 15px; font-weight: 400; color: var(--shep-text-2);
  border-top: 0; border-bottom: 3px solid transparent; border-radius: 6px 6px 0 0; overflow: visible;
}
#shep-research-nav > .tab-wrapper > .tab-container > button[role="tab"]:hover:not([aria-selected="true"]) {
  background: var(--shep-tab-hover-bg); border-bottom-color: var(--shep-tab-hover-line); color: var(--shep-text);
}
#shep-research-nav > .tab-wrapper > .tab-container > button[role="tab"][aria-selected="true"] {
  color: var(--shep-tab-selected-text); border-bottom-color: var(--shep-tab-selected-line); font-weight: 600;
}
#shep-research-nav > .tab-wrapper > .tab-container > button[role="tab"]::after { display: none; }
/* Group captions (R-22): generated content with empty alternative text, above
   the first tab of each group and aligned with its text, outside every tab's
   box: never clickable (pointer-events none), never focusable, no hover. */
#shep-page-test-button { margin-left: var(--shep-group-gap); }
#shep-page-datasets-button::before, #shep-page-test-button::before {
  position: absolute; left: 12px; bottom: calc(100% + 2px); height: 18px; line-height: 18px;
  font-size: 14px; font-weight: 600; color: var(--shep-group-caption); pointer-events: none; white-space: nowrap;
}
#shep-page-datasets-button::before { content: "Prepare"; content: "Prepare" / ""; }
#shep-page-test-button::before { content: "Evaluate"; content: "Evaluate" / ""; }
/* The divider between the groups, in the middle of the group gap, through the
   caption tier. */
#shep-research-nav > .tab-wrapper > .tab-container > #shep-page-test-button::after {
  display: block !important; content: ""; position: absolute; left: calc(var(--shep-group-gap) / -2); width: 1px;
  top: calc(2px - var(--shep-caption-h)); bottom: 6px; height: auto; right: auto;
  background: var(--shep-border-2); transform: none; animation: none;
}
/* Not enough room for both groups on one line (the shell's layout pass,
   ``shep-rnav-wrapped``): the Evaluate group goes whole onto a line of its
   own, under a caption tier of its own, with no divider at its line's start. */
#shep-shell.shep-rnav-wrapped #shep-research-nav > .tab-wrapper > .tab-container { row-gap: 0; }
#shep-shell.shep-rnav-wrapped #shep-research-nav > .tab-wrapper > .tab-container::after {
  display: block !important; content: ""; flex-basis: 100%; height: var(--shep-caption-h); order: 1;
  position: static; background: none; border: 0;
}
#shep-shell.shep-rnav-wrapped #shep-page-test-button, #shep-shell.shep-rnav-wrapped #shep-page-models-button { order: 2; }
#shep-shell.shep-rnav-wrapped #shep-page-test-button { margin-left: 0; }
#shep-shell.shep-rnav-wrapped #shep-research-nav > .tab-wrapper > .tab-container > #shep-page-test-button::after { display: none !important; }

/* ---------------------------------------------------------------- tab panels */
#shep-nav > .tabitem, #shep-research-nav > .tabitem { padding: 0 !important; border: 0 !important; background: none !important; }
#shep-nav > .tabitem > .column, #shep-research-nav > .tabitem > .column { gap: var(--layout-gap); }
#shep-tab-clinical > .column, #shep-tab-settings > .column, #shep-research-nav > .tabitem > .column { padding: 16px; }

/* ---------------------------------------------------------------- in-page segmented control (third style) */
.shep-seg .wrap { display: inline-flex !important; gap: 2px !important; padding: 3px; border: 1px solid var(--shep-grey-200); border-radius: 8px; background: var(--shep-grey-50); }
.shep-seg .wrap label { border: 1px solid transparent !important; border-radius: 6px !important; background: none !important; box-shadow: none !important; padding: 5px 12px !important; font-size: 15px; font-weight: 400; }
.shep-seg .wrap label { color: var(--shep-text-2) !important; }
.shep-seg .wrap label.selected { background: #fff !important; border-color: var(--shep-grey-200) !important; box-shadow: 0 1px 2px rgba(20,24,40,.12) !important; font-weight: 600; color: var(--shep-text) !important; }
.shep-seg .wrap label span { color: inherit !important; }
.shep-seg .wrap label input { position: absolute; opacity: 0; width: 1px; height: 1px; }
/* One row, always (M5-3: the candidate details' "Paths 5" wrapped onto a second
   row at 1536 x 760). The segments are grid columns sized up to their text: when
   the row is short of room, the short labels stay whole and the longest gives
   way, ending in an ellipsis with its whole text as its title (the shell's head
   script). */
.shep-seg .wrap {
  display: inline-grid !important; grid-auto-flow: column; grid-auto-columns: minmax(0, max-content);
  max-width: 100%; box-sizing: border-box;
}
.shep-seg .wrap label { min-width: 0 !important; }
.shep-seg .wrap label span { margin-left: 0 !important; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

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
/* One disabled look for every button, primary, secondary or stop (M5-3: Gradio
   fades a disabled secondary button to half opacity, so beside a grey primary
   one a second style read as "almost enabled"). */
.gradio-container button.primary:disabled, .gradio-container button.secondary:disabled, .gradio-container button.stop:disabled {
  background: var(--shep-grey-50) !important; border-color: var(--shep-grey-200) !important;
  color: var(--shep-grey-600) !important; opacity: 1 !important; filter: none !important; box-shadow: none !important;
  cursor: not-allowed;
}

/* ---------------------------------------------------------------- field labels and checkboxes */
/* A field's label (HPO terms, Patient ID, Workspace) is plain 14 px / 600 text,
   never a chip (the WebUI skill's §3: the Soft theme drew it as a tinted pill,
   indigo-500 on indigo-100 at 3.63:1, which reads as a tag or a button): no
   fill, no padding, the body text's colour, at least 4.5:1 on every surface a
   field sits on. A component's corner label (``block-label``, a plot's or a
   file's) stays the theme's chip, its text at least 4.5:1 on it. An
   unchecked checkbox's border is what shows it is there: at least 3:1 against
   its fill and what is around it (WCAG 1.4.11; the theme's grey-100 is 1.1:1,
   and its hover grey-300 1.4:1). A field's info= line (a state set at runtime,
   such as the Reference cohort select's "…: cannot be read: …", travels there)
   and a placeholder are text too: the theme's subdued grey (#9ca3af) is 2.54:1
   on white; the secondary text and the muted token are 8.9:1 and 6.1:1.

   **A chosen radio** (Training's Device, Fine-tune's Graph backbone) is a form
   option, not navigation (R-20): its label is tinted with an accent border,
   its text the body colour and its dot filled in the accent, never the
   level-1 switch's solid accent fill with white text (the theme's chosen fill
   was 4.47:1, and the accent made it the switch's exact look).

   **A field's box** (R-8): text, number and text-area inputs and every select
   have a 1 px border in the control token, 4.3:1 on white (the theme's was
   0 px, #f9fafb, 1.05:1), darker on hover, the focus token when focused; no
   shadow. A select's border is on its wrap (`.wrap` around `.wrap-inner`:
   Gradio 6.30 draws it in its own grey, #e5e7eb, by a rule as specific as a
   two-class one, so these name ``body`` too), and a text box drawn without a
   container (Training's Run name) gets the same border Gradio gives one in a
   container. */
.gradio-container {
  --block-title-text-color: var(--shep-label-color);
  --block-title-background-fill: transparent;
  --block-title-padding: 0;
  --block-title-radius: 0;
  --block-title-text-size: var(--shep-label-size);
  --block-title-text-weight: var(--shep-label-weight);
  --block-label-text-color: var(--shep-accent-text);
  --checkbox-border-color: var(--shep-grey-500);
  --checkbox-border-color-hover: var(--shep-accent);
  --checkbox-border-color-focus: var(--shep-focus-ring);
  --checkbox-border-color-selected: var(--shep-accent);
  --checkbox-background-color-selected: var(--shep-accent);
  --checkbox-label-border-width: 1px;
  --checkbox-label-border-color: var(--shep-grey-200);
  --checkbox-label-border-color-selected: var(--shep-accent);
  --checkbox-label-background-fill-selected: var(--shep-accent-50);
  --checkbox-label-text-color-selected: var(--shep-text);
  --block-info-text-color: var(--shep-text-2);
  --input-placeholder-color: var(--shep-muted);
  --input-border-width: 1px;
  --input-border-color: var(--shep-control-border);
  --input-border-color-hover: var(--shep-control-border-hover);
  --input-border-color-focus: var(--shep-focus-ring);
  --input-shadow: none;
  --input-shadow-focus: 0 0 0 1px var(--shep-focus-ring);
}
body .gradio-container .wrap:has(> .wrap-inner),
body .gradio-container :is(input[type=text]:not([role=combobox]), input[type=number], input[type=password], textarea) {
  border: var(--input-border-width) solid var(--input-border-color); box-shadow: none;
}
body .gradio-container .wrap:has(> .wrap-inner):hover,
body .gradio-container :is(input[type=text]:not([role=combobox]), input[type=number], input[type=password], textarea):hover {
  border-color: var(--input-border-color-hover);
}
body .gradio-container .wrap:has(> .wrap-inner):focus-within,
body .gradio-container :is(input[type=text]:not([role=combobox]), input[type=number], input[type=password], textarea):focus {
  border-color: var(--input-border-color-focus); box-shadow: var(--input-shadow-focus);
}
/* **A disabled field looks off** (a read-only viewer's Settings,
   NEW-settings-datasets-footer-3): a disabled text box, number box or select
   is grey with grey text and a light border, a disabled checkbox or radio
   grey (a checked one grey filled, its mark kept), and its words grey, so it
   never reads as one that can be changed (WCAG 1.4.11 exempts an inactive
   control's border; its text stays at 4.5:1). Hover changes nothing: Gradio
   6.30's own hover rule (.gradio-style input[type=checkbox]:hover, and
   radio's) ignores disabled, so it would turn the accent like an enabled one. */
.gradio-container input:not([type=checkbox]):not([type=radio]):disabled, .gradio-container textarea:disabled {
  background-color: var(--shep-disabled-bg) !important; border-color: var(--shep-disabled-border) !important;
  color: var(--shep-disabled-text) !important; -webkit-text-fill-color: var(--shep-disabled-text) !important;
  cursor: not-allowed;
}
body .gradio-container .wrap:has(> .wrap-inner input:disabled),
body .gradio-container .wrap:has(> .wrap-inner input:disabled):hover {
  background-color: var(--shep-disabled-bg); border-color: var(--shep-disabled-border);
}
.gradio-container input[type=checkbox]:disabled:not(:checked), .gradio-container input[type=radio]:disabled:not(:checked),
.gradio-container input[type=checkbox]:disabled:not(:checked):hover,
.gradio-container input[type=radio]:disabled:not(:checked):hover {
  border-color: var(--shep-disabled-border); background-color: var(--shep-disabled-bg); cursor: not-allowed;
}
.gradio-container input[type=checkbox]:disabled:checked, .gradio-container input[type=radio]:disabled:checked,
.gradio-container input[type=checkbox]:disabled:checked:hover, .gradio-container input[type=radio]:disabled:checked:hover {
  border-color: var(--shep-disabled-text); background-color: var(--shep-disabled-text);
}
.gradio-container label:has(> input[type=checkbox]:disabled), .gradio-container label:has(> input[type=radio]:disabled) {
  color: var(--shep-disabled-text); cursor: not-allowed;
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
.shep-fpv { font-family: var(--shep-mono) !important; font-weight: 600; color: var(--shep-text); font-size: 14px; }
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
.shep-amber-t { color: var(--shep-amber-800); }
/* The one status indicator (`renderers.status_mark`, the WebUI skill's §6):
   the state's colour, its shape (hidden from a screen reader) and its word,
   in the weight of the text around it. */
.shep-st-ok { color: var(--shep-status-ok); }
.shep-st-neutral { color: var(--shep-status-neutral); }
.shep-st-warn { color: var(--shep-status-warn); }
.shep-st-error { color: var(--shep-status-error); }
.shep-st-info { color: var(--shep-status-info); }
.shep-st-shape { display: inline-block; margin-right: 4px; font-size: 15px; line-height: 1; }
.shep-st > .shep-spinner { margin-right: 6px; }
.shep-st-chip .shep-st-shape { margin-right: 0; }
.shep-reason { display: flex; gap: 8px; align-items: flex-start; font-size: 14px; color: var(--shep-text-2); line-height: 1.45; }
.shep-reason-grey { border: 1px solid var(--shep-grey-200); background: var(--shep-grey-50); border-radius: 8px; padding: 10px 12px; color: var(--shep-grey-800); }
.shep-notice { border-radius: 6px; padding: 8px 10px; font-size: 14px; }
.shep-notice-amber { border: 1px solid var(--shep-amber-300); border-left: 4px solid var(--shep-amber-600); background: var(--shep-amber-50); color: var(--shep-amber-800); }
.shep-notice-red { border: 1px solid var(--shep-red-200); border-left: 4px solid var(--shep-red-700); background: var(--shep-red-50); color: var(--shep-red-700); }
.shep-touch-lost { margin-top: 4px; }
.shep-warn { color: var(--shep-amber-800); margin: 0 0 6px; }
.shep-mark { display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 14px; line-height: 1.4; margin: 1px 4px 1px 0; }
.shep-mark-serve { background: var(--shep-accent-50); border: 1px solid var(--shep-accent-100); color: var(--shep-accent-text); font-weight: 400; }
.shep-mark-serve-same { background: #fff; border: 1px dashed var(--shep-accent-100); color: var(--shep-accent-text); }
.shep-mark-outline { background: #fff; border: 1px solid var(--shep-border-2); color: var(--shep-text-2); }
.shep-mark-grey { background: var(--shep-grey-50); border: 1px solid var(--shep-grey-200); color: var(--shep-grey-800); }
.shep-btn { height: 32px; padding: 0 12px; border-radius: 8px; border: 1px solid var(--shep-border-2); background: #fff; font-size: 14px; cursor: pointer; }
/* An identifier (a role code, a run, a checkpoint file) is never broken inside
   its name: it stays whole, and in a place too narrow for it ends in an
   ellipsis, its whole value in its title (`renderers.ident_html`). */
.gradio-container code.shep-ident {
  display: inline-block; max-width: 100%; vertical-align: bottom; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; overflow-wrap: normal; word-break: normal;
}
/* Words kept in groups: a line breaks between groups, never leaving a word or
   two of a group alone on a line (`renderers.word_groups`); a group wider than
   its line still wraps inside itself. */
.shep-wg { display: inline-block; max-width: 100%; }
/* A select's value longer than its box ends in an ellipsis, and the shell's
   head script gives it its whole value as its title (the layout rule's
   titled ellipsis), on every page. */
.gradio-container .wrap-inner input[role="combobox"] { text-overflow: ellipsis; }
/* An accordion's ▼ turns in a square box, so turned it stays inside its header
   (M5-3: a turned glyph, taller than wide, spilled 3 px past the button). */
.gradio-container button.label-wrap > .icon {
  flex: none; display: inline-flex !important; align-items: center; justify-content: center;
  width: 22px; height: 22px; line-height: 1;
}
/* Its direction (R-31): Gradio 6.30 turns its ▼ 90° clockwise when closed, so
   it pointed left (◀). Closed it points right (▸), open down (▾), on every
   accordion: the turn is set here over Gradio's inline one. */
.gradio-container button.label-wrap > .icon { transform: rotate(-90deg) !important; }
.gradio-container button.label-wrap.open > .icon { transform: none !important; }

/* **A field row** (D03, the WebUI ledger's UI-03): labelled fields with an action beside them
   (Fine-tune's base picker, Training's Continue drawer, Test's Run, the Diagnosis load drawer's
   head). Gradio 6.30 draws a field as a padded block whose last line is its box, its label and any
   info line above it, so every item lines up on the boxes: each block draws its box at its
   bottom (two fields with different labels or info lines still line up; their form wrapper keeps
   its stretch), an action button is the box's height and stands on the box's bottom line (an
   icon-only one, ``shep-icon-btn``, a square of it), and words or another control beside a field
   (``shep-field-aside``) end on the box's bottom, a line of words centred on it. A wrapped row
   keeps the button beside the last box. A region that draws its blocks denser sets
   ``--shep-field-pad-b`` with its padding, so the two never drift apart. */
.shep-field-row { align-items: flex-end !important; flex-wrap: wrap !important; gap: 8px 12px !important; }
.shep-field-row > .form > .block, .shep-field-row > .block:not(.shep-field-aside) {
  display: flex !important; flex-direction: column !important; justify-content: flex-end !important;
}
.shep-field-row > .form > .block > .container { flex: none !important; }
.shep-field-row > button {
  align-self: flex-end !important; flex: none !important; width: auto !important; min-width: max-content !important;
  height: var(--shep-field-box-h) !important; min-height: 0 !important; margin: 0 0 var(--shep-field-pad-b) !important;
  padding: 0 14px !important;
}
.shep-field-row > button.shep-icon-btn { min-width: var(--shep-field-box-h) !important; width: var(--shep-field-box-h) !important; padding: 0 !important; }
.shep-field-row > .shep-field-aside {
  align-self: flex-end !important; flex: none !important; width: auto !important; margin: 0 0 var(--shep-field-pad-b) !important;
  min-height: var(--shep-field-box-h); display: flex !important; align-items: center;
}
.shep-field-row > .shep-field-aside > .html-container { display: flex; align-items: center; }

/* panels */
.shep-panel { background: var(--shep-panel) !important; border: 1px solid var(--shep-border) !important; border-radius: 10px !important; padding: 0 !important; gap: 0 !important; overflow: hidden; flex-wrap: nowrap !important; }
.shep-panel > .block, .shep-panel > .form { border: 0 !important; box-shadow: none !important; }
.shep-panel-hd { display: flex; align-items: center; gap: 10px; padding: 11px 16px; flex-wrap: wrap; border-bottom: 1px solid var(--shep-border); min-height: 48px; }
.shep-panel-hd h2 { font-size: var(--shep-title-size); font-weight: 600; margin: 0; line-height: 1.3; }
.shep-panel-right { margin-left: auto; color: var(--shep-text-2); font-size: 14px; }
.shep-panel-bd { padding: 14px 16px; display: flex; flex-direction: column; gap: 12px; }
.shep-panel-ft { padding: 12px 16px !important; border-top: 1px solid var(--shep-border); align-items: center !important; gap: 12px !important; flex-wrap: wrap !important; }
.shep-intro { padding: 12px 16px 0; margin: 0; font-size: 14px; color: var(--shep-text-2); }
.shep-card { border: 1px solid var(--shep-border); border-radius: 8px; background: #fff; }
.shep-card-hd .shep-muted { font-weight: 400; font-size: 14px; }
.shep-card-hd { padding: 9px 12px; border-bottom: 1px solid var(--shep-border); font-weight: 600; font-size: 15px; }
/* The one key-value summary list (`renderers.summary_list`, the skill's §4): a
   dl, its key column 8 to 12em (a longer key wraps), keys at 14 px, values at
   15 px (a region that sets its own text size, the Continue drawer's dense
   check at 14 px, sets its lists' values with it); no border of its own (one
   container level per region). */
.shep-kv { display: grid; grid-template-columns: fit-content(12em) minmax(0, 1fr); gap: 6px 14px; font-size: 15px; align-items: baseline; margin: 0; }
.shep-kv > dt, .shep-kv > dd { margin: 0; }
.shep-kv:not(.shep-kv-2) > .shep-k { min-width: 8em; }
.shep-kv > .shep-k { font-size: 14px; overflow-wrap: anywhere; }
.shep-kv > .shep-v { min-width: 0; overflow-wrap: anywhere; }
.shep-card .shep-kv { padding: 10px 12px; }
.shep-kv-2 { grid-template-columns: max-content max-content max-content max-content; }
/* A summary list's "Details ›" (`renderers.model_spec_html`): one level, its
   body under it, never a disclosure inside another nor a bordered box. */
.shep-more > summary { cursor: pointer; color: var(--shep-accent-text); width: max-content; min-height: 24px; line-height: 24px; list-style: none; }
.shep-more > summary::-webkit-details-marker { display: none; }
.shep-more > summary:hover { text-decoration: underline; }
/* Keyboard focus on every button, link and disclosure (Gradio's buttons, an
   accordion's head, a link-styled button, an icon Copy, a "Details ›"): the
   focus token's 2 px ring, never the browser's own (`outline: auto`, in no
   token: a faint grey ring on the accent buttons). At no specificity, so a
   control with a ring of its own (a tab, a table row, the ⓘ) keeps it (R-52, D08). */
:where(button, a[href], summary):focus-visible { outline: 2px solid var(--shep-focus-ring); outline-offset: 2px; }
.shep-more > .shep-kv { gap: 2px 12px; }
/* A form's one line on what the operator set (`training_console.set_by_line_html`: the
   Training base form's and the Fine-tune page's), and a value Start would refuse. */
.shep-setby p { margin: 0; font-size: 14px; color: var(--shep-text-2); overflow-wrap: anywhere; }
.shep-setby p.shep-amber-t { margin-top: 4px; color: var(--shep-amber-800); }
.shep-setby .shep-more { margin-top: 2px; font-size: 14px; }
.shep-setby-kv { font-size: 14px; }
/* A digest row (`renderers.summary_list`'s wide rows): its label on a line of
   its own over the whole list, its full value with an icon-only Copy on the
   line under it. */
.shep-kv > .shep-kv-wide { grid-column: 1 / -1; min-width: 0; }
.shep-kv > .shep-k.shep-kv-wide { font-size: 14px; font-weight: 600; margin-bottom: -4px; }
.shep-kv > .shep-v.shep-kv-wide { display: flex; flex-wrap: wrap; align-items: center; gap: 2px 8px; }
.shep-kv > .shep-v.shep-kv-wide > code.shep-full { flex: 0 1 auto; min-width: 0; }
.shep-k { color: var(--shep-text-2); }
/* A summary list among a form's fields (`renderers.FIELD_ROWS`): its keys are
   the form's field labels (R-16, one label style per page). */
.shep-kv.shep-kv-field > .shep-k { font-size: var(--shep-label-size); font-weight: var(--shep-label-weight); color: var(--shep-label-color); }
/* A record's relations (`renderers.relations_html`): a list of its own whose
   key column takes a relation's whole name on one line, up to 32em. */
.shep-kv.shep-kv-rel { grid-template-columns: fit-content(32em) minmax(0, 1fr); gap: 2px 14px; }
.shep-kv.shep-kv-rel > .shep-k { overflow-wrap: normal; }
/* R10's overlap matrix (`renderers.r10_table_html`, the skill's §4): a row
   per scope, a column per label, row lines only (no box of its own inside
   its region's). `table.` outranks the research tables' `.shep-tbl th/td`
   where it sits inside one (a Models record, a Results row): never their
   sticky, filled header. */
table.shep-r10 { border-collapse: collapse; margin: 0; font-size: 15px; line-height: 1.35; }
table.shep-r10 th, table.shep-r10 td { position: static; background: none; text-align: left; vertical-align: top; padding: 4px 14px 4px 0; border-bottom: 1px solid var(--shep-border); font-size: 15px; }
table.shep-r10 thead th { font-size: 14px; font-weight: 600; color: var(--shep-text-2); }
table.shep-r10 tbody th { font-size: 14px; font-weight: 600; min-width: 9em; }
table.shep-r10 tbody tr:last-child > * { border-bottom: 0; }
table.shep-r10 .shep-r10-n { font-size: 14px; font-weight: 400; color: var(--shep-text-2); }
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
.shep-tbl tbody tr[data-key] { cursor: pointer; scroll-margin-top: var(--shep-tbl-head-h, 0px); }
.shep-tbl tbody tr[data-key]:hover td { background: var(--shep-panel-2); }
.shep-tbl tbody tr.shep-on td { background: var(--shep-accent-50); }
.shep-tbl tbody tr.shep-on td:first-child { box-shadow: inset 4px 0 0 var(--shep-accent); }
.shep-tbl tbody tr[data-key]:focus-visible { outline: 2px solid var(--shep-focus-ring); outline-offset: -2px; }
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
/* The shared list's line under its workspace select (`model_list.workspace_line_html`). */
.shep-ws-line { display: flex; flex-wrap: wrap; gap: 4px 8px; align-items: center; font-size: 14px; color: var(--shep-text-2); }
.shep-row-sub { font-size: 14px; color: var(--shep-text-2); }
.shep-row-more { align-items: center; color: var(--shep-accent-text); font-weight: 400; }
.shep-mtbl-wrap { margin: 0; }
.shep-mtbl td { overflow-wrap: anywhere; }
.shep-mtbl-wrap + .shep-row-more { border: 1px solid var(--shep-border); border-top: 0; border-radius: 0 0 8px 8px; }
.shep-facts-title { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; font-size: 15px; font-weight: 600; }
.shep-facts { display: flex; flex-direction: column; gap: 8px; }
.shep-facts-sec { border: 1px solid var(--shep-border); border-radius: 8px; background: #fff; }
.shep-facts-sec > summary, .shep-facts-sec > h4 { padding: 8px 12px; font-weight: 600; font-size: 15px; cursor: pointer; margin: 0; }
.shep-facts-sec > h4 { cursor: default; }
.shep-facts-n { color: var(--shep-muted); margin-right: 4px; }
.shep-facts-sum { font-weight: 400; color: var(--shep-text-2); font-size: 14px; }
.shep-facts-bd { padding: 4px 12px 10px; }
.shep-facts-warn { border-color: var(--shep-amber-300); background: var(--shep-amber-50); }
.shep-confirm { border: 1px solid var(--shep-accent-100); background: #fbfbff; border-radius: 10px; padding: 12px 14px; }
.shep-confirm-q { font-size: 15px; font-weight: 600; margin: 4px 0 10px; }
.shep-ftbl { border-collapse: separate; border-spacing: 0; width: 100%; max-width: 1080px; border: 1px solid var(--shep-accent-100); border-radius: 8px; background: #fff; }
.shep-ftbl th, .shep-ftbl td { padding: 7px 10px; border-bottom: 1px solid #edf0f3; font-size: 15px; text-align: left; vertical-align: top; }
.shep-ftbl thead th { font-size: 14px; color: var(--shep-text-2); background: var(--shep-panel-2); }
.shep-ftbl tr.shep-changed th { box-shadow: inset 4px 0 0 var(--shep-accent); }
.shep-ftbl tr.shep-changed td:last-child { font-weight: 600; }
.shep-confirm-note { font-size: 14px; color: var(--shep-text-2); margin: 10px 0 4px; }
/* A reload in progress (the confirmation's status, Diagnosis' card): a small turning ring. */
.shep-spinner { display: inline-block; width: 12px; height: 12px; vertical-align: -1px; border-radius: 50%; border: 2px solid var(--shep-accent-100); border-top-color: var(--shep-accent); animation: shep-spin 0.9s linear infinite; }
@keyframes shep-spin { to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: reduce) { .shep-spinner { animation: none; } }
.shep-outcome { border-radius: 8px; padding: 10px 12px; font-size: 15px; }
.shep-outcome-ok { background: var(--shep-green-50); border: 1px solid var(--shep-green-200); color: var(--shep-green-800); }
.shep-outcome-amber { background: var(--shep-amber-50); border: 1px solid var(--shep-amber-300); color: var(--shep-amber-800); }
.shep-outcome-grey { background: var(--shep-grey-50); border: 1px solid var(--shep-grey-200); color: var(--shep-grey-800); }
.shep-if-waited { display: none; }
.shep-waited .shep-if-waited { display: inline; }

/* ---------------------------------------------------------------- the runtime line */
#runtime_footer { padding: 4px 16px !important; font-size: 14px; }
/* The consequence first, then "Versions ›", whose line opens beside it. */
.shep-rt { display: flex; flex-wrap: wrap; align-items: baseline; gap: 2px 12px; }
.shep-rt-v { display: inline-flex; flex-wrap: wrap; gap: 0 8px; align-items: baseline; }
.shep-rt-v > span { color: var(--shep-text-2); }

/* **A text box that grows as it is typed in keeps its caret in view** (R-25: at 1536 x 760 the
   Diagnosis HPO box's caret went out of view from its sixth line on). The browser brings the caret
   into its scroll region as a key is typed, then Gradio grows the box a line under it; every
   scroll region (the page's, a panel's) keeps a line's room under what it brings into view, so
   the caret stays in view while the box grows. */
html, .shep-panel * { scroll-padding-bottom: var(--shep-caret-room); }
/* **A scroll cue** (R-73): a region with ``shep-scroll-cue`` that has more below fades its last
   lines out, and draws them whole again at its end (the shell's head script keeps
   ``data-shep-more`` on it as it scrolls and as its content changes), whatever the system's
   scrollbars (overlay ones draw nothing at rest). */
.shep-scroll-cue[data-shep-more] {
  -webkit-mask-image: linear-gradient(to bottom, #000 calc(100% - 28px), transparent);
  mask-image: linear-gradient(to bottom, #000 calc(100% - 28px), transparent);
}

/* ---------------------------------------------------------------- the layout rule */
/* Side by side (`SIDE`, the shell's one decision): one screen. The selected
   page fills the space under the header; a page taller than that scrolls
   inside its own panel; a panel is never narrower than its page's
   ``--shep-panel-min`` (the decision's own measure). */
.shep-page { --shep-panel-min: 0px; }
__SIDE__ {
  &, & body { height: 100%; overflow: hidden; }
  & gradio-app { height: 100vh; display: flex; flex-direction: column; }
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
  .shep-page > .shep-panel { min-height: 0; max-height: 100%; display: flex; flex-direction: column; min-width: var(--shep-panel-min) !important; }
  .shep-page > .shep-panel > .block { flex: none; }
  & footer, #runtime_footer { flex: none; }
  /* A panel's header stays where it is when its body scrolls (M5-3: the Test
     Result panel's title scrolled away with the report): a header drawn inside
     the region that scrolls sticks to that region's top. */
  .shep-panel .shep-panel-hd { position: sticky; top: 0; z-index: 6; background: var(--shep-panel); }
}
/* Stacked: the regions stack, the page scrolls, nothing scrolls inside; a
   table wider than its panel scrolls sideways in its own box (it was cut by
   the panel, out of reach: Datasets at 760 x 1000). */
__STACK__ {
  .shep-page { flex-direction: column !important; }
  .shep-page > .shep-panel { width: 100% !important; }
  .shep-tbl-wrap { overflow-x: auto; overflow-y: hidden; }
  .shep-tbl th { position: static; }
  .shep-tbl td { overflow-wrap: anywhere; }
}
""".replace("__BRAND__", str(BRAND_WIDTH)).replace("__SIDE__", SIDE).replace("__STACK__", STACK)


__all__ = ["BRAND_WIDTH", "LAYOUT_ATTR", "SHELL_CSS", "SHORT", "SHORT_ATTR", "SHORT_BELOW", "SIDE", "SIDE_SHORT",
           "STACK"]
