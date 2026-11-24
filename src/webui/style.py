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
size, raised to the floor of the arrangement the window takes: no wider than
tall, `STACK_FLOOR_WIDTH`; wider, the display floor, `FLOOR_WIDTH` x `FLOOR_HEIGHT`)
decides it: no wider than it is tall, stacked; wider, side by side when
the page shown fits by its panels' own minimum widths (each panel's
``--shep-panel-min``, set in its page's style sheet; side by side, a panel is
never narrower), else stacked. A short window (`SHORT`, under
`SHORT_BELOW` px tall) is density only: it never stacks, shrinks text or
squeezes a chart. Before the first pass nothing is written, which reads as
stacked. Below the floor the frame keeps the floor's size and the document
scrolls, so every mark is the floor's.

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
#: region's title, and at 20 px the area switch no longer fits beside it at
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
#: **The display floor** (the owner's decision of 2026-11-23): the smallest usable browser area
#: the dashboard is drawn for, in CSS px (a 1080p screen at 150 % gives about 1280 wide). At or
#: above it in both axes the interface is whole: side by side the frame never scrolls (the
#: area's one region scrolls inside, with its cue), nothing scrolls sideways and nothing is
#: compressed to fit. Below it in either axis the layout is drawn as at the floor and shrinks no
#: further: the app's frame keeps the floor's size in that axis (`SHELL_CSS`: its ``min-width``,
#: and side by side its height) and the document scrolls as a whole. The shell's layout pass
#: decides every mark on the window's size raised to the floor in each axis (`shell.py`'s
#: ``area()``), so below the floor each mark is the floor's. One pair, read by those two only:
#: no media query on the window's size, no page's own copy.
FLOOR_WIDTH, FLOOR_HEIGHT = 1280, 720
#: **The stacked arrangement's floor** (the owner's decision of 2026-11-23 on portrait windows): a
#: window no wider than tall takes the stacked arrangement, and this is its floor, a 1080p screen
#: turned upright at 125 % (1080 at 100 %). Measured stacked, every page in both states with every
#: drawer holds every shared rule with nothing compressed at 1080, 1024, 960 and 864 px wide; at
#: 720 (the same screen at 150 %) Models' tree in its narrow form runs past its panel, and at 800
#: it has a few px to spare. The display floor above is the side-by-side arrangement's; the layout
#: pass raises the window to the floor of the arrangement the window takes (`shell.py`'s
#: ``area()``: no wider than tall, this width in both axes, the narrowest window no wider than
#: tall; wider, `FLOOR_WIDTH` x `FLOOR_HEIGHT`) and writes that floor's width (`FLOOR_VAR`) for the
#: frame's ``min-width`` (`SHELL_CSS`), so a portrait window at or above it never scrolls sideways,
#: and one below it is drawn as at it, the document scrolling.
STACK_FLOOR_WIDTH = 864
#: The width of the floor the window's arrangement takes, in px (a custom property on the root
#: element the layout pass writes): the frame's ``min-width``.
FLOOR_VAR = "--shep-floor-w"
#: **A box drawn in its wide form while that form fits** (OPS-FIX-R2; the
#: layout pass's fit, `src/webui/shell.py`): a box of class `FIT` shows its
#: wide form (class `FIT_WIDE`, a table) while the form's own narrowest, its
#: min-content width in the type it is drawn in, fits the box; when it does
#: not, the pass sets `NARROW_ATTR` on the box and its page's style sheet
#: draws its narrow form instead. No fixed width decides it, so another
#: host's text metrics cannot make the wide form overflow its box.
FIT = "shep-fit"
FIT_WIDE = "shep-fit-wide"
#: **A segmented control whose labels stay whole** (with ``shep-seg``): one row while the row
#: holds them, else one label a line (the layout pass's narrow mark, `NARROW_ATTR`, measured as
#: a choice group's), never a label ended in an ellipsis nor one left alone on a line.
SEG_WHOLE = "shep-seg-whole"
#: **A column a narrow table drops** (the shared ``[data-shep-narrow]`` rule below): a cell of a
#: column whose facts the row's details also show carries it, so a `FIT_WIDE` table drops that
#: column before its box scrolls sideways.
COL_OPT = "shep-col-opt"
#: **An optional column's fact said in its row while the column is dropped** (the shared
#: ``[data-shep-narrow]`` rule below): drawn only while its FIT box is narrow, so the fact is
#: said once in either form (the model tree's file rows: how far each file trained).
COL_TWIN = "shep-col-twin"
#: **A table cell whose value reads as one** (``white-space: nowrap``): a value of a closed set (a
#: model's task, its operation), so its column takes the value's width and a column of names or
#: prose wraps instead (the L6 check: the Models table broke "Fine-tune, full" and "Record not
#: readable" over two lines while its Model column had room). A table that cannot then fit drops
#: its optional columns, then scrolls in its frame (`FIT`).
CELL_WHOLE = "shep-cell-whole"
NARROW_ATTR = "data-shep-narrow"
#: **A table cell whose words read as one while their table has the room** (a count's words in
#: its place: "recorded epoch 9", `renderers.epoch_words`): the layout pass measures a `FIT` box's
#: wide form once more with these cells whole and, when that still fits, marks the box
#: `ROOMY_ATTR`, and the cells keep their words on one line; otherwise they wrap as words do, so
#: binding them never makes a table drop its optional columns or scroll where it fitted (the
#: sweep's L7 check, D13: "recorded" / "epoch 9" doubled every older run's row at 1850 x 960
#: while the Model column had room; bound always, the tree dropped its Epochs at 1550 x 830).
CELL_ROOMY = "shep-cell-roomy"
ROOMY_ATTR = "data-shep-roomy"
#: **The width a `FIT` box's narrow form needs** (a custom property, in px): the layout pass
#: measures that form at its narrowest, adds what the panel holding the box draws around it (the
#: nearest ancestor with a ``--shep-panel-min``) and writes the sum on that panel's row, so a
#: page whose panel shares move (Models with a row chosen) leaves the panel at least that wide
#: where it can, from the type drawn and the names listed, never a width fitted to one world (the
#: sweep's L7 check, K2: the chosen details left the list its fixed minimum, 2.6 px under the
#: narrow form beside a 52-character run name, and the rows scrolled sideways at 1366 x 768).
FIT_LEAST = "--shep-fit-least"
#: **The width a `FIT` box's wide form needs, given to its panel** (a custom property, in px, on
#: the panel holding the box, the nearest ancestor with a ``--shep-panel-min``): the layout pass
#: measures the wide form at its narrowest and adds what the panel draws around it, and writes
#: that where the panels beside it keep their own minimums, else 0px, before it judges the box,
#: so a page whose panels share the row (Datasets) can give the box's panel that width from the
#: panel beside, never a width fitted to one world (the sweep's L7 check, K3: bound whole, a mark
#: made the workspaces table 5 px too wide for its share at 1366 x 768; its narrow form then moved
#: two facts into the details beside, which scrolled, where the details could spare the 5 px).
FIT_WANT = "--shep-fit-want"
#: **A box drawn whole while its column has the room** (OPS-FIX-N1; the layout
#: pass's fold, `src/webui/shell.py`): side by side, a box of class `FOLD`
#: shares a column with a region of class `FOLD_ROOM` (the column: the box's
#: nearest ancestor that holds such a region as a child), and the region names
#: its floor, a height in px in its custom property `FOLD_FLOOR` (a region without it
#: has a floor of 0, so its box folds only when it alone overflows). While the column
#: holds everything in it with the box whole and the region at its floor (or
#: at its whole content, when that is less), the box is drawn whole; when it
#: does not, the pass sets `FOLDED_ATTR` on the box, whose parts of class
#: `FOLD_PART` are then not drawn (this sheet; they stay one click away in the
#: box's own details), and its page's style sheet may draw the rest denser.
#: The floor decides the fold and is not a CSS minimum: where the box, folded
#: or with nothing to fold, still leaves the region less, the region takes
#: what is left and stays its column's one scroll region (a minimum there
#: overflowed the column, a second scroll region around the region, the
#: column's foot out of view; the review of the fold's first version). The box
#: is measured drawn whole, so the measure never depends on the state it sets,
#: and no window size decides it: the box's own height, the column's and the
#: region's floor do.
#: **The bottom scroll cue's fade length** (a custom property, px): the shell's layout pass sets
#: it on a region with more below so the fade reaches back over the nearest thing the region
#: draws at its edge (28 px at least; the sweep's L7 check, C1: a fade over the gap between two
#: fields reads as nothing).
FADE_VAR = "--shep-more-fade"
FOLD = "shep-fold"
FOLD_ROOM = "shep-fold-room"
FOLD_FLOOR = "--shep-fold-floor"
FOLD_PART = "shep-fold-part"
FOLDED_ATTR = "data-shep-folded"

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
  /* Code, paths and variables (M7, the sweep's decision 2): the system's monospace faces, scaled
     to Plex Sans' x-height (`font-size-adjust` on every element drawn in it), so a token sits in
     a sentence at the body's size. No monospace font file ships. */
  --shep-mono: ui-monospace, 'SFMono-Regular', Menlo, Consolas, 'DejaVu Sans Mono', 'Liberation Mono', monospace;
  --shep-mono-adjust: 0.516;
  --shep-row1-h: 46px; --shep-brand-w: __BRAND__px;
  /* The cues every navigation level and control reads (D01, UI-10): aliases
     of the colours above, no new hex. */
  --shep-tab-h: 32px; --shep-tab-hover-bg: var(--shep-grey-50);
  --shep-tab-selected-line: var(--shep-accent); --shep-tab-selected-text: var(--shep-accent-text);
  --shep-focus-ring: var(--shep-accent-600);
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
  /* **The spacing scale and the text layout** (the WebUI skill's §2; the text brief's §6): one
     scale; a panel's one content edge; the gap inside a group and between groups; the measure
     of prose (56 characters of its own size: about 73 a line); unitless line heights for text
     that wraps and for one or two lines; the order item < paragraph < above a heading, and a
     heading closer to its content than to what is above it. */
  --shep-space-1: 4px; --shep-space-2: 8px; --shep-space-3: 12px; --shep-space-4: 16px;
  --shep-space-5: 24px; --shep-space-6: 32px;
  --shep-inset: 16px;
  --shep-gap-in: var(--shep-space-2); --shep-gap-between: var(--shep-space-4);
  --shep-page-gap: var(--shep-space-4);
  --shep-measure: 56ch;
  --shep-lh-prose: 1.45; --shep-lh-compact: 1.3; --shep-lh-title: 1.3;
  --shep-space-para: 8px; --shep-space-item: 4px;
  --shep-space-head-above: 16px; --shep-space-head-below: 4px;
  /* The narrowest a text box is drawn (its group wraps first) and the room a select takes
     when it has it (M2). */
  --shep-input-floor: 12ch; --shep-select-floor: 14rem;
  /* The gap before the first tab of each later tab group, its separator in the middle (M8). */
  --shep-tabgroup-gap: 20px;
}
/* A panel's size by what it holds (M4): registered, so every reader (the layout pass's measure
   included) reads a length in px whatever unit a page writes it in (rem, em). */
@property --shep-panel-min { syntax: '<length>'; inherits: false; initial-value: 0px; }
@property --shep-panel-basis { syntax: '<length>'; inherits: false; initial-value: 0px; }
@property --shep-panel-grow { syntax: '<number>'; inherits: false; initial-value: 1; }

/* ---------------------------------------------------------------- page frame */
:root { color-scheme: light; }
body, gradio-app, .gradio-container { background: var(--shep-bg) !important; }
body { color: var(--shep-text); }
.gradio-container { font-size: 15px; color: var(--shep-text); }
/* **Every text that wraps breaks prettily** (the text brief's §4; the sweep's L7 check, D13): the
   wrap style is inherited, so one rule at the dashboard's root reaches every paragraph, reason,
   table cell, summary value and link, and the browser may rebalance a text's last lines so one
   short word is less often left alone ("only", "server", "better" stood alone on a last line in
   cells and values the page rules did not reach). It is a preference, not a guarantee: Chromium
   leaves some two-line texts with a word alone (the sweep's L7 check, K7: "Supplied cohort" /
   "version" in a narrow cell), and Firefox ignores ``pretty`` and keeps its greedy breaks, so
   nothing may depend on it. A ``white-space: nowrap`` text keeps its one line; a heading
   or a lead keeps its own ``balance``. */
.gradio-container { text-wrap-style: pretty; }
/* The header runs edge to edge; each page pads itself. */
.gradio-container > .main { padding: 0 !important; }
.gradio-container > .main > .wrap, .gradio-container .contain { padding: 0 !important; max-width: none !important; }
.gradio-container sub, .gradio-container small { font-size: 14px; vertical-align: baseline; }
.gradio-container code { font-family: var(--shep-mono) !important; font-size: 14px; font-size-adjust: var(--shep-mono-adjust); }
/* **Monospace where a page asks for it** (M7): a literal token, a log, an identifier. Two
   classes, so the mount's rule that draws everything in Plex Sans (`app.DASHBOARD_CSS`,
   ``.gradio-container *``) does not win over it. */
.gradio-container .shep-mono, .gradio-container .shep-fpv, .gradio-container .shep-full,
.gradio-container :is(code, .shep-mono) * {
  font-family: var(--shep-mono) !important; font-size-adjust: var(--shep-mono-adjust);
}
/* Two weights (the type scale): bold text in rendered HTML and Markdown is the
   strong weight, never the browser's 700. */
.gradio-container b, .gradio-container strong, .gradio-container th { font-weight: var(--shep-weight-strong); }
#shep-shell { position: relative; gap: 0 !important; background: var(--shep-bg); }
/* **The display floor** (`FLOOR_WIDTH`, `FLOOR_HEIGHT`; a window no wider than tall, the stacked
   arrangement's, `STACK_FLOOR_WIDTH`): the app's frame is never narrower than the floor of the
   arrangement the window takes (`FLOOR_VAR`, the layout pass's; the display floor's before its first
   pass), less the room the window's own scrollbar takes (``100vw - 100%``: a stacked page's), so a
   window at the floor never scrolls sideways and one below it scrolls as a whole, its layout drawn
   as at the floor. Side by side its height is the window's and never under the floor's (`SIDE`,
   below). */
gradio-app { min-width: calc(var(__FLOOR_VAR__, __FLOOR_W__px) - (100vw - 100%)); }

/* ---------------------------------------------------------------- row 1: brand, area switch, chips, Settings */
#shep-brand {
  position: absolute; top: 0; left: 16px; height: var(--shep-row1-h); z-index: 4;
  display: flex; align-items: center; pointer-events: none; padding: 0 !important;
  width: auto !important; min-width: 0 !important; border: 0 !important; background: none !important;
}
.shep-brand { font-size: var(--shep-body-size); font-weight: 600; letter-spacing: -0.01em; white-space: nowrap; color: var(--shep-text); }
/* The brand is the page's h1: no heading size or margin of its own. */
#shep-brand h1.shep-brand { margin: 0; padding: 0; line-height: inherit; font-size: var(--shep-body-size); font-weight: 600; letter-spacing: -0.01em; color: var(--shep-text); }
#shep-chips {
  position: absolute; top: 0; right: calc(var(--shep-settings-w, 120px) + 32px); z-index: 4;
  height: var(--shep-row1-h); display: flex; align-items: center; justify-content: flex-end;
  padding: 0 !important; width: auto !important; min-width: 0 !important; border: 0 !important;
  background: none !important; max-width: calc(100% - var(--shep-settings-w, 120px) - 48px);
}
/* The chips on one line in a box that hugs its right edge (``#shep-chips``): at the display floor's
   width the widest two chips, each capped below (``.shep-chip-text``), leave the box room to spare. */
.shep-chips { display: flex; align-items: center; gap: 10px; }
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
/* A chip naming a long holder (the GPU lock's command, pid and start) ends its words in an
   ellipsis at its cap, their whole text as their title (the shell's titled ellipsis). */
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
/* Level 1's one selected look (R-20): the area switch's segment and Settings
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

/* ---------------------------------------------------------------- row 2: the research pages, the stage hint */
/* The research pages' tabs in one row, 32 px, and after them the stage hint
   (the shell's head script, `shell.STAGES`): the row wraps by itself, so when
   it has no room for both the hint goes onto a line of its own, still at the
   right, and the tabs wrap among themselves only when they alone do not fit
   (no measure, no window size). The tablist sizes to its tabs and takes the
   rest of its line. */
#shep-research-nav > .tab-wrapper {
  height: auto !important; padding: 0 !important; margin: 0 !important; flex-wrap: wrap; align-items: center;
  background: var(--shep-panel); border-bottom: 1px solid var(--shep-border);
}
#shep-research-nav > .tab-wrapper > .tab-container {
  width: auto; flex: 1 1 auto; min-width: 0;
  height: auto !important; min-height: var(--shep-tab-h); background: none !important;
  padding: 4px 16px 0; align-items: flex-end; overflow: visible;
}
#shep-research-nav > .tab-wrapper > .tab-container::after { display: none; }
#shep-research-nav > .tab-wrapper > .tab-container > button[role="tab"] {
  height: var(--shep-tab-h); padding: 0 12px; font-size: 15px; font-weight: 400; color: var(--shep-text-2);
  border-top: 0; border-bottom: 3px solid transparent; border-radius: 6px 6px 0 0; overflow: visible;
}
/* A tab under the pointer: a fill and darker text, never a line under it, so
   the selected tab is the only one that reads as selected (N7). */
#shep-research-nav > .tab-wrapper > .tab-container > button[role="tab"]:hover:not([aria-selected="true"]) {
  background: var(--shep-tab-hover-bg); color: var(--shep-text);
}
#shep-research-nav > .tab-wrapper > .tab-container > button[role="tab"][aria-selected="true"] {
  color: var(--shep-tab-selected-text); border-bottom-color: var(--shep-tab-selected-line); font-weight: 600;
}
#shep-research-nav > .tab-wrapper > .tab-container > button[role="tab"]::after { display: none; }
/* The stage hint: plain text, "Stage:" in the muted colour and the stage in
   the body text's strong weight; no control's look (no box, pointer or hover).
   Its words break between them before they would ever run out of the row. */
#shep-research-nav > .tab-wrapper > .shep-stage {
  flex: 0 1 auto; min-width: 0; margin: 0 16px 0 auto; padding: 6px 0 4px; line-height: 1.4;
  font-size: var(--shep-key-size); font-weight: 400; color: var(--shep-muted); text-align: right;
  overflow-wrap: anywhere; cursor: default;
}
#shep-research-nav > .tab-wrapper > .shep-stage > .shep-stage-v {
  font-size: var(--shep-body-size); font-weight: var(--shep-weight-strong); color: var(--shep-text);
}

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
/* A segment whose labels stay whole (`SEG_WHOLE`): one row while it fits, one label a line when
   the layout pass marks it narrow (as a choice group), never one alone under the others. */
.shep-seg.__SEG_WHOLE__ .wrap { display: inline-flex !important; flex-wrap: nowrap !important; }
.shep-seg.__SEG_WHOLE__[__NARROW__] .wrap { flex-direction: column !important; align-items: stretch !important; }

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
  --body-text-color-subdued: var(--shep-text-2);
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
.shep-note { font-size: 14px; color: var(--shep-muted); max-width: var(--shep-measure); }
/* A note in a span (beside a button) keeps the measure too: as an inline box it took none. */
span.shep-note { display: inline-block; }
.shep-note p { font-size: 14px; }
.shep-sep { color: var(--shep-border-2); }
/* A fact of a " · " chain (`renderers.chain_html`): one unit, moved to the next line whole where
   its line cannot hold it, wrapped inside only when it is wider than a line (L7 SW-LK7). */
.shep-fact { display: inline-block; max-width: 100%; }
/* What a line never parts (`renderers.bound_html`: a count and its unit, a date and its time, a
   short parenthesis): one unit the same way, so a narrow box never cuts or overflows it. */
.shep-unit { display: inline-block; max-width: 100%; }
/* A unit draws the line its text is decorated with: no decoration is drawn into an inline block,
   so a link's underline stopped where its place began ("Choose one in" underlined, "Settings ›
   Data locations" not; the sweep's L7 check, K1). It takes the decoration of the element it
   stands in, a link's hover thickness included. */
.shep-fact, .shep-unit { text-decoration: inherit; }
/* A separator the line breaks after is not drawn (the shell's layout pass marks it; its box stays,
   so nothing moves): a line neither starts nor ends with " · " (the sweep's L7, SW-LK2, SW-LK7). */
.shep-sep[data-shep-end] { visibility: hidden; }
/* A workspace's marks ("synthetic", the record's state; `renderers.dataset_marks_html`) move as one
   group, wrapping inside only where a line cannot hold them, on every page (Test's chips parted on
   two lines, Fine-tune's did not, Training kept its own copy of this rule). */
.shep-ds-marks { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 4px 6px; vertical-align: middle; }
.shep-nowrap { white-space: nowrap; }
/* A name drawn in pieces (`renderers.name_pieces_html`): each piece on one line, a line breaking
   only before the name's own separators, a piece longer than its box broken as the last resort. */
.shep-brk { overflow-wrap: break-word; }
.shep-brk > span { white-space: nowrap; }
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
/* The shape is inline, never an atomic box: a line may break after an atomic inline whatever
   follows it, so a line parted "●" from "verified" (the sweep's L7, SW-LK7); inline, with the no-break
   space after it (`renderers.MARK_SPACE`), its word starts on its line. */
.shep-st-shape { display: inline; margin-right: 4px; font-size: 15px; line-height: 1; }
.shep-st > .shep-spinner { margin-right: 6px; }
/* The space between the shape and the word (`renderers.MARK_SPACE`): in the text, so a copy reads
   "○ Saved", of no width, so the shape's margin draws the gap as before. In a chip the shape, space
   and word are one inline run (`shep-st-in`), so the tag's flex gap is the shape's margin. */
.shep-st-sp { font-size: 0; }
.shep-st-chip .shep-spinner { margin-right: 4px; }
/* **A reason** (why a control is off, what a list cannot show): text in one flow, never a row of
   columns (M5: as a flex box its text and each element in it were columns of their own), at the
   measure, start-aligned. Led by a mark hidden from a screen reader (a lock, a status shape), the
   mark keeps a column of its own and every line of the words starts on one edge: the mark and
   one element of words after it (`data_locations.reason_line`, `renderers.phrase_line`), as every
   producer draws it. Both columns are as wide as what they hold (the words' at most the line): an
   ``auto`` mark column took the room the words left, so on a line wider than them the words
   started far from their mark (74 px on Fine-tune, 209 px on Test), and a page's own
   ``justify-content`` still places the pair. */
.shep-reason {
  display: block; font-size: 14px; color: var(--shep-text-2); line-height: var(--shep-lh-prose);
  max-width: var(--shep-measure); text-align: start;
}
.shep-reason:has(> :first-child[aria-hidden="true"]) {
  display: grid; grid-template-columns: max-content minmax(0, max-content); column-gap: 6px; align-items: baseline;
}
.shep-reason:has(> :first-child[aria-hidden="true"]) > :not(:first-child) { grid-column: 2; }
/* A neutral reason (a refusal, a state not read) is a reason line like every other, in the
   neutral text colour: no filled box of its own (one container per region; it was the one boxed
   reason, beside Models' and Diagnosis' Load). */
.shep-reason-grey { color: var(--shep-grey-800); }
.shep-notice { border-radius: 6px; padding: 8px 10px; font-size: 14px; max-width: var(--shep-measure); }
.shep-notice-amber { border: 1px solid var(--shep-amber-300); border-left: 4px solid var(--shep-amber-600); background: var(--shep-amber-50); color: var(--shep-amber-800); }
.shep-notice-red { border: 1px solid var(--shep-red-200); border-left: 4px solid var(--shep-red-700); background: var(--shep-red-50); color: var(--shep-red-700); }
.shep-touch-lost { margin-top: 4px; }
.shep-warn { color: var(--shep-amber-800); margin: 0 0 6px; }
/* A mark is one unit in its border: never drawn over two lines inside it (the sweep's L7 check, D5:
   Datasets' "serving reads this workspace" took two lines in one border at 1366 x 768); its cell
   wraps round it. */
.shep-mark { display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 14px; line-height: 1.4; margin: 1px 4px 1px 0; white-space: nowrap; }
.shep-mark-serve { background: var(--shep-accent-50); border: 1px solid var(--shep-accent-100); color: var(--shep-accent-text); font-weight: 400; }
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
.shep-field-row > .form > .block:not(fieldset), .shep-field-row > .block:not(.shep-field-aside, fieldset) {
  display: flex !important; flex-direction: column !important; justify-content: flex-end !important;
}
.shep-field-row > .form > .block > .container { flex: none !important; }
/* A choice group is no column: its title and its ⓘ are its own children, on one line (a column put
   them on two); it stands on the boxes' bottom line as a whole. */
.shep-field-row > .form > fieldset.block, .shep-field-row > fieldset.block { align-self: flex-end !important; }
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
/* A panel's header keeps its own height: in a panel's column it never shrinks under its words (a
   drawer's header was squeezed at 1366 x 768). */
.shep-panel-hd { display: flex; align-items: center; gap: 10px; padding: 11px 16px; flex-wrap: wrap; border-bottom: 1px solid var(--shep-border); min-height: 48px; flex: none !important; }
.shep-panel-hd h2 { font-size: var(--shep-title-size); font-weight: 600; margin: 0; line-height: 1.3; }
.shep-panel-right { margin-left: auto; color: var(--shep-text-2); font-size: 14px; }
.shep-panel-bd { padding: 14px 16px; display: flex; flex-direction: column; gap: 12px; }
.shep-panel-ft { padding: 12px 16px !important; border-top: 1px solid var(--shep-border); align-items: center !important; gap: 12px !important; flex-wrap: wrap !important; }
/* A panel's footer keeps its own height: its body takes the room the panel has left (a footer
   drawn as a column grew with it, Fine-tune's Start row to 111 px). */
.shep-panel > .shep-panel-ft { flex: none !important; }
/* **Words beside a control only while they are one line** (the skill's §2: only a one-line
   relation sits beside the control it explains; anything longer takes a line of its own under
   it): in a wrapping row, the words (``shep-beside``) are as wide as their one line, so a row
   that cannot hold them and the control on one line puts them on the next line, from the row's
   start, never a column of two or three lines beside the button. ``shep-beside-row`` is such a
   row (a form's "Set by you" line and Reset to reference). */
.shep-beside-row { flex-wrap: wrap !important; align-items: baseline !important; gap: var(--shep-space-2) var(--shep-space-3) !important; }
.shep-beside-row > button { flex: none !important; width: auto !important; min-width: max-content !important; }
.shep-beside { flex: 1 1 max-content !important; min-width: 0 !important; }
/* Words that would stand beside the control on two lines or more (their sentence longer than the
   measure that caps it, in a row wide enough to hold the measure and the control: 2560 px) take
   the row's next line: the shell's layout pass marks them (``data-shep-own-line``). */
.shep-beside-row > .shep-beside[data-shep-own-line] { flex-basis: 100% !important; }
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
   body under it, never a disclosure inside another nor a bordered box. Its label is as wide
   as its words and wraps only where its box is narrower: a label built from data never sets
   its column's width (the sweep's L7, SW-LK3: Datasets' "Not admitted: Gene,
   Patients-like-me ›" made the Measurements column 280 px, the versions table scrolled
   sideways and the chevron was cut). */
.shep-more > summary { cursor: pointer; color: var(--shep-accent-text); width: fit-content; max-width: 100%; min-height: 24px; line-height: 24px; list-style: none; }
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
/* One recorded result (`renderers.record_html`, R-82): the Test page's
   result and each of Models' records alike. Its answer first (a task's
   tiles), then its rows, then its metrics as recorded one click away, the
   disclosure's summary a 24 px target that shows the keyboard focus. */
.shep-rec-label { font-weight: 600; font-size: 15px; display: flex; align-items: center; gap: 4px; flex-wrap: wrap; }
.shep-rec-body { display: flex; flex-direction: column; gap: 10px; min-width: 0; }
.shep-rec-tiles { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); grid-auto-rows: auto; gap: 8px; }
__STACK__ .shep-rec-tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); }
/* Each tile spans three of the tiles' rows (name, value, what it is over) and
   takes them as its own (a subgrid), so every value of a row of tiles stands
   on one line however a name wraps (NEW-test-4: 19-20 px apart at 1536 and
   1440 wide). */
.shep-rec-tile { border: 1px solid var(--shep-border); border-radius: 8px; padding: 8px 10px; background: #fff; min-width: 0; display: grid; grid-row: span 3; grid-template-rows: subgrid; row-gap: 0; align-content: start; }
.shep-rec-tile-l, .shep-rec-tile-n { font-size: 14px; color: var(--shep-text-2); }
.shep-rec-tile-v { font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums; }
.shep-rec-fold { font-size: 15px; overflow-wrap: anywhere; }
.shep-rec-fold > summary { cursor: pointer; min-height: 24px; line-height: 24px; width: max-content; max-width: 100%; color: var(--shep-text-2); }
.shep-rec-fold[open] > summary { margin-bottom: 4px; }
.shep-rec-ln { line-height: 1.45; }
.shep-rec-notes ul { margin: 4px 0 0; padding-left: 18px; font-size: 15px; }
.shep-digest { display: grid; grid-template-columns: 1fr auto; gap: 4px 12px; align-items: center; }
.shep-digest .shep-dl { grid-column: 1 / -1; font-size: 14px; font-weight: 600; color: var(--shep-text-2); }

/* ---------------------------------------------------------------- the notice (M1) */
/* **The one notice** (`renderers.notice_block`, `notices_block`, `phrase_line`; the text brief's
   §1): one line at the body size, its status mark in a column of its own so a wrapped line
   starts under the words; the page's consequence and the next step, a link to its place, in the
   same flow; the why and how under "Details ›" at 14 px, its references muted. Prose at the
   measure, start-aligned, in the secondary text colour; no box of its own (one container per
   region). Steps, Details and a second line hang on the words' edge, never under the mark. */
.shep-nb {
  --shep-nb-mark-w: 1em; --shep-nb-mark-gap: 6px;
  display: flex; flex-direction: column; gap: var(--shep-space-item); max-width: var(--shep-measure);
  font-size: var(--shep-body-size); line-height: var(--shep-lh-prose); color: var(--shep-text-2);
  text-align: start; text-wrap: pretty;
}
.shep-nb-line, .shep-nb-rel {
  display: grid !important; grid-template-columns: var(--shep-nb-mark-w, 1em) minmax(0, 1fr);
  column-gap: var(--shep-nb-mark-gap, 6px); align-items: baseline; margin: 0;
}
.shep-nb-rel { --shep-nb-mark-w: 1em; --shep-nb-mark-gap: 6px; }
.shep-nb-mark { font-size: 15px; line-height: inherit; text-align: center; }
.shep-nb-text { min-width: 0; }
/* Two classes: Gradio's reset (``.gradio-container-<version> p, ol``: margin and padding 0) is
   one class and an element. */
.shep-nb > :is(.shep-nb-next, .shep-nb-steps, .shep-nb-more) {
  margin: 0 0 0 calc(var(--shep-nb-mark-w) + var(--shep-nb-mark-gap));
}
.shep-nb > .shep-nb-steps { padding: 0 0 0 1.25em; list-style: decimal outside; }
.shep-nb-steps > li + li { margin-top: var(--shep-space-item); }
.shep-nb-steps > li::marker { color: var(--shep-muted); }
.shep-nb-more { font-size: 14px; line-height: var(--shep-lh-prose); }
.shep-nb-more[open] > summary { margin-bottom: var(--shep-space-item); }
.shep-nb .shep-nb-kv { gap: var(--shep-space-item) 12px; font-size: 14px; }
.shep-nb .shep-nb-kv > .shep-v { font-size: 14px; overflow-wrap: break-word; }
.shep-nb .shep-nb-ref { margin: var(--shep-space-item) 0 0; font-size: 14px; color: var(--shep-muted); }
/* Several notices' Details (`renderers.notices_block`): each notice's rows under its setting's name. */
.shep-nb .shep-nb-sub { margin: 0 0 var(--shep-space-item); font-size: 14px; font-weight: 600; }
.shep-nb .shep-nb-kv + .shep-nb-sub { margin-top: var(--shep-space-para); }
/* The next step: a link in the sentence (underlined, the accent text), a button to the keyboard. */
.gradio-container button.shep-nb-go {
  display: inline; margin: 0; padding: 0; border: 0; background: none; font: inherit;
  color: var(--shep-accent-text); text-align: start; cursor: pointer;
  text-decoration: underline; text-decoration-thickness: 1px; text-underline-offset: 3px;
}
.gradio-container button.shep-nb-go:hover { text-decoration-thickness: 2px; }
/* **R-WHY: a dimmed control's phrase** (`renderers.reason_phrase`): five words at most beside the
   control, a link in the accent text (4.5:1 on the page's ground), focusable, one unit on its
   line; activated, its whole reason comes into view (a notice) or opens in the one help bubble
   (a header chip's). The words say the need or who can act, never "disabled". */
.gradio-container button.shep-why {
  display: inline; margin: 0; padding: 0; border: 0; background: none; font: inherit;
  color: var(--shep-accent-text); text-align: start; cursor: pointer; white-space: nowrap;
  text-decoration: underline; text-decoration-thickness: 1px; text-underline-offset: 3px;
}
.gradio-container button.shep-why:hover, .gradio-container button.shep-why[aria-expanded="true"] {
  text-decoration-thickness: 2px;
}
/* A literal token in a sentence (`renderers.token_html`): breaks only at its <wbr>, no hyphen. */
.gradio-container code.shep-tok { hyphens: none; white-space: normal; overflow-wrap: normal; }
.gradio-container code.shep-tok > span { white-space: nowrap; }
/* **An empty region's words** (the WebUI skill's §9; M9): in the flow, never an overlay. */
.shep-empty {
  margin: 0; padding: var(--shep-space-2) 0; max-width: var(--shep-measure); font-size: var(--shep-body-size);
  line-height: var(--shep-lh-prose); color: var(--shep-text-2); text-align: start; text-wrap: pretty;
}
/* **Paragraphs** (M5, the text brief's §2): Gradio's reset draws every paragraph with no margin;
   a region spaces its paragraphs by its own gap (``--shep-space-para``), never by a margin of a
   page's own or a <br><br>. Prose in a panel keeps the measure; at no specificity, so a page's
   own class (a table's, a summary list's) keeps its width. */
:where(.shep-panel p, .shep-edge p) { max-width: var(--shep-measure); text-wrap: pretty; }
/* The same measure for a field's line under its label (Gradio's ``info=``) and a muted sentence
   on a line of its own after a break: Settings' torch.compile line ran to about 88 characters at
   1850 and 106 at 2560, its Acceleration sentence to 67 (the sweep's L7 check, C12). The muted
   line is a box only there, so a muted word inside a line still wraps with it. */
:where(.gradio-container .info-text) { max-width: var(--shep-measure); }
:where(br + .shep-muted) { display: inline-block; max-width: var(--shep-measure); }
:where(.shep-panel h2, .shep-panel h3, .shep-edge h2, .shep-edge h3) { text-wrap: balance; }

/* ---------------------------------------------------------------- bars and groups (M2) */
/* **A bar is a wrapping row of groups** (Training's run bar, a list's tools, a run's header): a
   group never splits, it moves to the next line whole; a message is never a group, it is the
   bar's note, a row of its own the bar's whole width (hidden while it says nothing: a ``gr.HTML``
   always draws its container and the element inside it, so what is tested is that element's
   content, never the container's). Give a
   ``gr.Row`` the class ``shep-bar`` and its parts ``shep-grp`` (``-grow``: takes the line's
   room, from ``--shep-grp-basis``; ``-end``: at the line's end; ``-wrap``: its aside may go under
   its field); the note ``shep-bar-note``. A text box in a group keeps its floor
   (``--shep-input-floor``): the group wraps before the box would shrink under it. */
.shep-bar {
  display: flex !important; flex-wrap: wrap !important; align-items: flex-end !important;
  gap: var(--shep-space-2) var(--shep-gap-between) !important;
}
.shep-bar > .shep-grp {
  flex: 0 1 auto !important; flex-wrap: nowrap !important; align-items: flex-end !important;
  gap: var(--shep-gap-in) !important; width: auto !important;
  min-width: min(var(--shep-grp-min, 0px), 100%) !important;
}
.shep-bar > .shep-grp-grow { flex: 1 1 var(--shep-grp-basis, 20rem) !important; }
.shep-bar > .shep-grp-end { margin-inline-start: auto !important; }
/* An end group that the bar wrapped onto a line of its own starts that line, on the bar's edge
   (the shell's layout pass marks it; an auto margin takes no part in line breaking, so the mark
   never moves it to another line): pushed to the end alone, it left the line's start empty (the
   sweep's L7, SW-LK5: Training's actions 643 px in from the run bar's edge at 1550 x 830). */
.shep-bar > .shep-grp-end[data-shep-alone] { margin-inline-start: 0 !important; }
.shep-bar > .shep-grp-wrap { flex-wrap: wrap !important; }
.shep-bar > .shep-bar-note { flex: 1 1 100% !important; min-width: 0 !important; }
.shep-bar > .shep-bar-note:not(:has(> .html-container > :not(:empty), .prose > *)) { display: none !important; }
.shep-grp :is(input[type=text], input[type=search], textarea):not([role=combobox]) { min-width: var(--shep-input-floor); }
.shep-grp > :is(.block, .form):has(input[type=text]:not([role=combobox]), textarea) {
  flex: 1 1 18ch !important; min-width: calc(var(--shep-input-floor) + 24px) !important;
}
.shep-grp > button { flex: none !important; width: auto !important; min-width: max-content !important; white-space: nowrap; }
/* A column of groups (a form's sections): the gap between groups, a smaller one inside each. */
.shep-stack { display: flex !important; flex-direction: column !important; gap: var(--shep-gap-between) !important; }
.shep-stack > .shep-grp-v { display: flex; flex-direction: column; gap: var(--shep-gap-in); }

/* ---------------------------------------------------------------- fields and choice groups (M3) */
/* **A row of fields** (``shep-fields``): each field takes its basis (``--shep-field-basis``) and
   never goes under its floor (``--shep-field-min``); the row wraps whole fields. A grid of fields
   (``shep-field-grid``): as many columns of at least ``--shep-field-min`` as fit. Gradio draws a
   row's adjacent fields in one ``.form``, so the rules reach the fields through it: the form
   wraps them as the row would, or lays them in the grid's columns (it drew them as one line of
   equal shares, every floor and basis ignored). */
.shep-fields {
  display: flex !important; flex-wrap: wrap !important; align-items: flex-start !important;
  gap: var(--shep-space-2) var(--shep-space-4) !important;
}
.shep-fields > :is(.block, .form), .shep-fields > .form > .block:not(fieldset) {
  flex: 1 1 var(--shep-field-basis, 12rem) !important; min-width: min(var(--shep-field-min, 10rem), 100%) !important;
}
/* A row's fields share their labels' top line (the sweep's L7, SW-LK2: aligned by their bottoms,
   a field with an info line under its label lifted that label 23 px above its neighbour's). */
.shep-fields > .form {
  flex-wrap: wrap !important; align-items: flex-start !important; gap: var(--shep-space-2) var(--shep-space-4) !important;
}
.shep-field-grid, .shep-field-grid > .form {
  display: grid !important; gap: var(--shep-space-2) var(--shep-space-4) !important;
  grid-template-columns: repeat(auto-fill, minmax(min(var(--shep-field-min, 15rem), 100%), 1fr));
}
.shep-field-grid > .form { grid-column: 1 / -1; background: none !important; border: 0 !important; box-shadow: none !important; }
.shep-field-grid > fieldset.block:has(input[type=radio], input[type=checkbox]),
.shep-field-grid > .form > fieldset.block:has(input[type=radio], input[type=checkbox]) { grid-column: 1 / -1; }
/* **A choice group is whole or one per line** (M3; Gradio draws a radio or checkbox group as a
   fieldset block): in a row of fields it takes its one-line width and moves to the next line
   whole; in a column it keeps the column's width, on its edge (Gradio centres an auto-margin
   block, and a form's grey ground showed beside it); its options never wrap inside their words.
   When even a whole line cannot hold them, the layout pass marks it (`NARROW_ATTR`, its fit
   measured with the mark off) and they stand one per line: never one option alone under the
   others. */
.gradio-container fieldset.block:has(input[type=radio], input[type=checkbox]):not(.shep-seg) {
  max-width: 100% !important; margin-inline: 0 !important;
}
.gradio-container :is(.row, .shep-fields, .shep-bar, .shep-grp, .shep-field-row) > fieldset.block:has(input[type=radio], input[type=checkbox]):not(.shep-seg),
.gradio-container :is(.row, .shep-fields, .shep-bar, .shep-grp, .shep-field-row) > .form > fieldset.block:has(input[type=radio], input[type=checkbox]):not(.shep-seg) {
  flex: 0 1 auto !important; width: max-content !important;
}
.gradio-container fieldset.block:not(.shep-seg) > .wrap:has(> label) > label > span { white-space: nowrap; }
/* A choice group's line under its label (Gradio's ``info=``) wraps inside the options' width and
   never sets the group's width (Fine-tune's Device was the width of its note on one line, 388 px
   for options of 254, and pushed Run name off its row; the sweep's L7, SW-LK2); in a row of
   fields a group with such a line grows into the row's free room, so the line stays one line
   where there is room. */
.gradio-container fieldset.block:has(input[type=radio], input[type=checkbox]):not(.shep-seg) > .info-text {
  inline-size: 0; min-inline-size: 100%;
}
.gradio-container :is(.row, .shep-fields, .shep-field-row) > fieldset.block:has(> .info-text):has(input[type=radio], input[type=checkbox]):not(.shep-seg),
.gradio-container :is(.row, .shep-fields, .shep-field-row) > .form > fieldset.block:has(> .info-text):has(input[type=radio], input[type=checkbox]):not(.shep-seg) {
  flex-grow: 1 !important;
}
.gradio-container fieldset.block[__NARROW__]:not(.shep-seg) > .wrap:has(> label) {
  flex-direction: column !important; align-items: flex-start !important;
}
/* A select's box never runs past its container (M9; Training's Viewing was 520 px in 358). */
.gradio-container .wrap:has(> .wrap-inner) { max-width: 100%; }

/* ---------------------------------------------------------------- one content edge (M5) */
/* **Inside a panel every heading, text, label and field starts on one edge** (the panel's inset,
   ``--shep-inset``): a Gradio block's padding is vertical only there (it added 12 px beside its
   label and box, so fields stood right of the text above them), its vertical padding the field
   row's measure; a block placed straight in a panel, with no padded column around it, takes the
   inset itself. */
.shep-panel, .shep-edge { --block-padding: var(--shep-field-pad-b) 0; }
.shep-panel > .block.padded, .shep-panel > .form > .block.padded { padding-inline: var(--shep-inset); }
/* With no padding beside its blocks, a form's row of fields keeps the gap between fields (its
   boxes touched); a form's ground (Gradio draws it in the border colour, its gaps 0) showed grey
   wherever its blocks left part of a line empty: it is the panel's. */
.shep-panel .form, .shep-edge .form { column-gap: var(--shep-gap-between) !important; background: none !important; }
/* A region drawn as a column of a panel without its class (the shared model list's columns) takes
   the same edge with ``shep-edge``. A notice's link focuses its target's heading (the shell's go
   link): a heading is no control, so it draws no focus ring. */
.shep-panel-hd :is(h2, h3)[tabindex="-1"]:focus { outline: none; }

/* ---------------------------------------------------------------- tables that fit (M6) */
/* **A table drops its optional columns before it scrolls**: a wide table is a FIT box's wide form
   (``shep-fit`` on its wrapper, ``shep-fit-wide`` on the table); a cell of a column its row's
   details also show carries ``shep-col-opt``. While the whole table does not fit, the layout
   pass marks the wrapper narrow and those columns are not drawn; when even that does not fit,
   the wrapper scrolls sideways (its own region). A word breaks in a cell only when it alone is
   wider than its column. */
[__NARROW__] :is(th, td, col).__COL_OPT__ { display: none !important; }
/* **A dropped column's fact said in its row while its column is not drawn** (``shep-col-twin``):
   a row whose only facts beside its name are in optional columns (the model tree's saved files:
   how far each trained) says the one a reader compares rows by under its name, only while the
   wrapper is narrow, so each fact is drawn once in either form (the sweep's L7, SW-LK4: "Latest
   save" and "Best pooled MRR" rows showed a name and empty cells). */
.__FIT_BOX__:not([__NARROW__]) .__COL_TWIN__ { display: none !important; }
.shep-tbl td { overflow-wrap: break-word; }
.shep-tbl td.__CELL_WHOLE__ { white-space: nowrap; }

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
.shep-tbl tbody tr.shep-row-dis td { background: var(--shep-grey-50); color: var(--shep-muted); cursor: default; }
/* A greyed row that can still be chosen (an unreadable file: its details say why); a greyed row
   with no key (an entry the listing could not list) is no choice and keeps the default cursor. */
.shep-tbl tbody tr.shep-row-dis[data-key] td { cursor: pointer; }

/* the shared model list: a note under a list (the Test page's), "Show older runs" under the
   tree (`model_list.older_html`) */
.shep-list-note { padding: 6px 12px; font-size: 14px; color: var(--shep-muted); border-bottom: 1px solid #edf0f3; }
button.shep-row {
  display: flex; flex-direction: column; align-items: flex-start; gap: 2px; text-align: left; width: 100%;
  padding: 8px 12px; border: 0; border-bottom: 1px solid #edf0f3; background: #fff; font: inherit; color: inherit; cursor: pointer;
}
.shep-row:hover { background: var(--shep-panel-2); }
.shep-row-dis { background: var(--shep-grey-50); color: var(--shep-muted); cursor: default; }
/* The shared list's line under its workspace select (`model_list.workspace_line_html`). */
.shep-ws-line { display: flex; flex-wrap: wrap; gap: 4px 8px; align-items: center; font-size: 14px; color: var(--shep-text-2); }
.shep-row-more { align-items: center; color: var(--shep-accent-text); font-weight: 400; }
.shep-mtbl-wrap { margin: 0; }
/* Words in the list's table break only where they must, in either layout (a column is
   never narrower than its longest word, so "Disease" stays one word); a table wider
   than its frame scrolls in it rather than squeezing a word. */
.shep-tbl.shep-mtbl td { overflow-wrap: break-word; }
.shep-mtbl-wrap + .shep-row-more { border: 1px solid var(--shep-border); border-top: 0; border-radius: 0 0 8px 8px; }
/* The shared list's tree (D04, `model_list._tree_html`): each group its own row group,
   each run's saved files under it, folded by a 24 x 24 toggle (the skill's §10 target)
   whose glyph is its cue (aria-expanded says its state). Every row is drawn; a folded
   one is `hidden`. */
.shep-mtree tr[hidden] { display: none; }
.shep-mtree .shep-fold {
  display: inline-flex; align-items: center; justify-content: center; flex: none;
  width: 24px; height: 24px; margin: 0 4px 0 -2px; padding: 0; border: 1px solid transparent;
  border-radius: 4px; background: none; color: var(--shep-text); font-size: 14px; cursor: pointer;
}
.shep-mtree .shep-fold:hover { border-color: var(--shep-grey-500); }
.shep-mtree .shep-fold:focus-visible { outline: 2px solid var(--shep-focus-ring); outline-offset: -2px; }
/* A toggle focused under the table's sticky header scrolls clear of it, as a row does
   (`model_list.MODEL_LIST_JS` keeps the header's height in `--shep-tbl-head-h`). */
.shep-mtree .shep-fold { scroll-margin-top: calc(var(--shep-tbl-head-h, 0px) + 4px); }
/* The fold toggle stands on a wrapped name's first line, not the middle of its lines: its glyph
   on that line's baseline, as it stands beside a name of one line. */
.shep-mtree .shep-tree-name { display: flex; align-items: baseline; min-height: 24px; }
.shep-mtree .shep-tree-label { min-width: 0; overflow-wrap: break-word; }
/* A saved file's name stands in from its run's (past the fold toggle's column), and the line of
   its dropped columns' facts under it (``__COL_TWIN__``) on the same edge (the sweep's L7 check,
   C10: it started 26 px left of its row's name at 1366 and 1280). */
.shep-mtree .shep-tree-leaf, .shep-mtree .shep-tree-leaf + .__COL_TWIN__ { margin-left: 26px; }
.shep-mtree tr.shep-tree-run { cursor: pointer; }
.shep-mtree tr.shep-tree-run:hover td { background: var(--shep-panel-2); }
.shep-mtree td.shep-num { white-space: nowrap; }
/* A count is one figure; words in its place ("epochs not recorded") wrap as words do, and read
   as one line while their table has the room (``__CELL_ROOMY__``). */
.shep-mtree td.shep-num .shep-muted { white-space: normal; }
.__FIT_BOX__[__ROOMY__] .shep-tbl :is(td, th) .__CELL_ROOMY__ { white-space: nowrap; }
/* A saved file's name, then its In-use chip, under it when the column is narrow. */
.shep-mtree .shep-tree-leaf { flex-wrap: wrap; column-gap: 6px; }
/* **The family graph** (Models' By family, `model_list._graph_html`; G1-G9, I1-I10): each
   family's runs as nested lists, a node a box as large as its text and padding (I1), the
   lines drawn from the lists' boxes, each meeting its node's box at its name line
   (`--shep-fam-anchor` from the node's top), so a node that grows down or a fold moves
   no line off a box. Top to bottom by default; left to right where the region holds it
   (`[data-fam-orient="h"]` on the region, set by `model_list.FAMILY_JS` from the graph's
   measured width; a subtree unfolded past it is panned to in the region). A node with
   child runs has its arrow area beside its body: hidden (focusable, of no size, I1) until
   the pointer is over the node or the keyboard's focus is in it (one box, one fill, so the
   pointer crosses no gap, I2/I4; focus a pointer's press leaves on an arrow keeps nothing
   open, or the area closing as the press moves on would shift what is under the pointer
   between its press and release, I7), then below the body top to bottom (the node grows
   down, the arrow centred, what follows moves down) and in the connector lane to its right
   left to right, over the line (no other box moves, the arrow level with the name line, so
   a line said under the body moves it nowhere, I2/I5); the whole area is the arrow's
   button; always shown where a pointer is coarse or none hovers (I4); no transition under
   reduced motion (I7). One look says "selected" (the run holding the selected model: the
   accent fill and bar; a node's body selects its run). Lines and node borders meet 3:1
   (the skill's §10). */
.shep-fam {
  --shep-fam-gap: 10px; --shep-fam-pad: 0.45em 0.7em; --shep-fam-anchor: calc(1px + 0.45em + 0.675em);
  --shep-fam-act: 2em; --shep-fam-lane: calc(var(--shep-fam-act) + 0.6em); --shep-fam-stub: 1.2em;
  --shep-fam-indent: 1.1em; --shep-fam-line: var(--shep-grey-500);
  display: flex; flex-direction: column; gap: 12px; min-width: 0; font-size: 15px;
}
.shep-fam [hidden] { display: none !important; }
.shep-fam ul { list-style: none; margin: 0; padding: 0; }
.shep-fam-forest { display: flex; flex-direction: column; gap: 12px; }
/* Families told apart by a rule between them, not only by space. */
.shep-fam-tree + .shep-fam-tree { border-top: 1px solid var(--shep-border); padding-top: 12px; }
.shep-fam-tree { display: flex; flex-direction: column; gap: var(--shep-fam-gap); }
.shep-fam-li { position: relative; }
.shep-fam-node {
  position: relative; box-sizing: border-box; width: max-content; max-width: min(100%, 30em);
  border: 1px solid var(--shep-control-border); border-radius: 8px; background: #fff;
}
.shep-fam .shep-fam-body {
  display: flex; flex-direction: column; gap: 2px; box-sizing: border-box; width: 100%; margin: 0;
  padding: var(--shep-fam-pad); border: 0; border-radius: 7px; background: none; font: inherit;
  color: var(--shep-text); text-align: left; line-height: 1.35; cursor: pointer; overflow-wrap: break-word;
}
.shep-fam-node:hover { background: var(--shep-panel-2); }
.shep-fam .shep-fam-body:focus-visible { outline: 2px solid var(--shep-focus-ring); outline-offset: -3px; }
.shep-fam-name { font-size: 15px; overflow-wrap: anywhere; }
.shep-fam-ln { font-size: 14px; color: var(--shep-text-2); }
/* The run holding the selected model: the selected look, the accent fill and the bar the
   tree's selected row has; hover keeps it. The folded ancestor it is in (I10): the bar and
   its words (`.shep-fam-held`), in the colour of the node's own lines: they act on nothing,
   so they never read as a link ("Show it" above the graph and the arrow show the way). */
.shep-fam-node.shep-sel, .shep-fam-node.shep-sel:hover { border-color: var(--shep-accent); background: var(--shep-accent-50); }
.shep-fam-node.shep-sel::before, .shep-fam-node.shep-fam-holds::before {
  content: ""; position: absolute; left: -1px; top: -1px; bottom: -1px; width: 4px;
  border-radius: 8px 0 0 8px; background: var(--shep-accent); pointer-events: none;
}
.shep-fam-held { display: block; padding: 0 0.7em 0.45em; font-size: 14px; line-height: 1.35; color: var(--shep-text-2); }
/* A run kept only for a shown run (a task filter's base, a search's ancestor): quieter. */
.shep-fam-node.shep-ctx { border-style: dashed; }
.shep-fam-node.shep-ctx .shep-fam-name .shep-w600 { font-weight: 400; }
/* A folded subtree: the box drawn as a stack, its arrow saying how many runs it hides. */
.shep-fam-node.shep-fam-shut { box-shadow: 3px 3px 0 -1px #fff, 3px 3px 0 0 var(--shep-control-border); }
/* A boundary: where a parent not listed here is, no choice (G5). */
.shep-fam-bound {
  display: flex; flex-direction: column; gap: 2px; padding: var(--shep-fam-pad); line-height: 1.35;
  border-style: dashed; background: var(--shep-panel-2); color: var(--shep-text-2);
}
/* The arrow area: of no size until shown, its button still a tab stop. */
.shep-fam-act { display: flex; justify-content: center; height: 0; overflow: hidden; opacity: 0; }
.shep-fam-node:is(:hover, :has(:focus-visible)) > .shep-fam-act { height: auto; overflow: visible; opacity: 1; }
.shep-fam .shep-fam-tog {
  display: inline-flex; align-items: center; justify-content: center; flex: none; width: 100%; height: 30px;
  padding: 0; border: 0; border-radius: 4px; background: none; font: inherit;
  color: var(--shep-accent-text); cursor: pointer;
}
.shep-fam .shep-fam-tog:hover { background: var(--shep-grey-50); }
.shep-fam .shep-fam-tog:focus-visible { outline: 2px solid var(--shep-focus-ring); outline-offset: -2px; }
.shep-fam .shep-fam-tog svg { transform: rotate(90deg); }
.shep-fam .shep-fam-tog[aria-expanded="true"] svg { transform: rotate(-90deg); }
/* Top to bottom: each child run in from its parent's left edge, a line down from the parent
   and across to the child's name line. */
.shep-fam-kids { margin-left: var(--shep-fam-indent); }
.shep-fam-kids > .shep-fam-li { padding: var(--shep-fam-gap) 0 0 var(--shep-fam-stub); }
.shep-fam-kids > .shep-fam-li::before {
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; border-left: 1px solid var(--shep-fam-line);
}
.shep-fam-kids > .shep-fam-li:last-child::before { bottom: auto; height: calc(var(--shep-fam-gap) + var(--shep-fam-anchor)); }
.shep-fam-kids > .shep-fam-li::after {
  content: ""; position: absolute; left: 0; width: var(--shep-fam-stub);
  top: calc(var(--shep-fam-gap) + var(--shep-fam-anchor)); border-top: 1px solid var(--shep-fam-line);
}
/* Left to right: each run's children in a column after its connector lane, which a node
   with child runs keeps whether they are shown or folded, so the arrow area shown in it
   never pushes anything or widens the region. */
[data-fam-orient="h"] .shep-fam-forest { width: max-content; }
[data-fam-orient="h"] .shep-fam-tree { gap: 0; }
[data-fam-orient="h"] .shep-fam-li { display: flex; align-items: flex-start; }
[data-fam-orient="h"] .shep-fam-node { flex: none; max-width: 18em; margin: calc(var(--shep-fam-gap) / 2) 0; }
[data-fam-orient="h"] .shep-fam-node.shep-fam-parent, [data-fam-orient="h"] .shep-fam-bound { margin-right: var(--shep-fam-lane); }
[data-fam-orient="h"] .shep-fam-kids { position: relative; margin-left: 0; }
[data-fam-orient="h"] .shep-fam-kids::before {
  content: ""; position: absolute; left: calc(-1 * var(--shep-fam-lane)); width: var(--shep-fam-lane);
  top: calc(var(--shep-fam-gap) / 2 + var(--shep-fam-anchor)); border-top: 1px solid var(--shep-fam-line);
}
[data-fam-orient="h"] .shep-fam-kids > .shep-fam-li { padding: 0 0 0 var(--shep-fam-stub); }
[data-fam-orient="h"] .shep-fam-kids > .shep-fam-li::before { top: 0; bottom: 0; height: auto; }
[data-fam-orient="h"] .shep-fam-kids > .shep-fam-li:first-child::before { top: calc(var(--shep-fam-gap) / 2 + var(--shep-fam-anchor)); }
[data-fam-orient="h"] .shep-fam-kids > .shep-fam-li:last-child::before { bottom: auto; height: calc(var(--shep-fam-gap) / 2 + var(--shep-fam-anchor)); }
[data-fam-orient="h"] .shep-fam-kids > .shep-fam-li:only-child::before { display: none; }
[data-fam-orient="h"] .shep-fam-kids > .shep-fam-li::after { top: calc(var(--shep-fam-gap) / 2 + var(--shep-fam-anchor)); }
[data-fam-orient="h"] .shep-fam-act {
  position: absolute; z-index: 1; left: calc(100% - 1px); top: -1px; bottom: -1px; box-sizing: border-box;
  width: var(--shep-fam-act); height: auto; align-items: stretch; overflow: visible; padding: 0;
  opacity: 0; pointer-events: none; border: 1px solid var(--shep-control-border); border-left: 0;
  border-radius: 0 8px 8px 0; background: inherit;
}
[data-fam-orient="h"] .shep-fam-node:is(:hover, :has(:focus-visible)) > .shep-fam-act { opacity: 1; pointer-events: auto; padding: 0; }
[data-fam-orient="h"] .shep-fam-node.shep-fam-parent:is(:hover, :has(:focus-visible)) { border-top-right-radius: 0; border-bottom-right-radius: 0; border-right-color: transparent; }
[data-fam-orient="h"] .shep-fam-node.shep-sel > .shep-fam-act { border-color: var(--shep-accent); }
[data-fam-orient="h"] .shep-fam .shep-fam-tog {
  height: auto; align-items: flex-start; padding-top: max(0px, calc(var(--shep-fam-anchor) - 9px));
  border-radius: 0 7px 7px 0;
}
[data-fam-orient="h"] .shep-fam .shep-fam-tog svg { transform: none; }
[data-fam-orient="h"] .shep-fam .shep-fam-tog[aria-expanded="true"] svg { transform: rotate(180deg); }
@media (prefers-reduced-motion: no-preference) {
  [data-fam-orient="h"] .shep-fam-act { transition: opacity 0.12s ease; }
}
/* A coarse pointer, or none that hovers: the arrow area always there (I4). */
@media (any-pointer: coarse), (hover: none) {
  .shep-fam { --shep-fam-act: max(2em, 36px); }
  .shep-fam-act { height: auto; overflow: visible; opacity: 1; }
  .shep-fam .shep-fam-tog { height: 36px; }
  [data-fam-orient="h"] .shep-fam-act { opacity: 1; pointer-events: auto; padding: 0; }
  [data-fam-orient="h"] .shep-fam-node.shep-fam-parent { border-top-right-radius: 0; border-bottom-right-radius: 0; border-right-color: transparent; }
}
.shep-fam .shep-fam-show { padding: 0 12px; }
.shep-fam .shep-fam-h { margin: 4px 0 0; font-size: 15px; font-weight: 600; }
/* The shared list's search line (D04): "Search" beside its box, as a field row says it, and
   its Clear after it. */
.shep-search-row { flex-wrap: nowrap !important; align-items: center !important; gap: 8px !important; }
.shep-search-row > :first-child, .shep-search-row > button { flex: none !important; width: auto !important; min-width: max-content !important; }
/* The box is a group's text box (M2): it takes the line's room and never goes under its floor;
   where the row has not that room its parent wraps it whole (a bar) or it scrolls in its region,
   never a 22 px box with Clear over it (B-03: every child here was allowed to shrink to 0). */
.shep-search-row > :not(:first-child):not(button) {
  flex: 1 1 18ch !important; min-width: calc(var(--shep-input-floor) + 24px) !important;
}
/* In a bar (whose groups stand on one bottom line) "Search" and Clear stay on the box's middle
   line: a label beside its box is no label above it. */
.shep-bar > .shep-grp.shep-search-row { align-items: center !important; }
.shep-search-lbl { font-size: 15px; color: var(--shep-text-2); white-space: nowrap; }
.shep-list-outside { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 8px; margin: 0 0 8px; font-size: 14px; color: var(--shep-text-2); }
.shep-facts-title { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; font-size: 15px; font-weight: 600; }
/* Its "↑ Back to the list" (`model_list.BACK_TO_LIST`), shown stacked only, at the title's end. */
.shep-facts-title > .shep-back-list { display: none; margin-left: auto; font-weight: 400; }
/* Model Facts' sections are groups of their column, not boxes in it (the skill's §2: one
   container level per region, one content edge per panel): each title and its rows start on
   the column's edge, a section set apart by the heading's space above it. A warning keeps its
   own mark and words (`renderers.warning_html`: ▲, amber). */
.shep-facts { display: flex; flex-direction: column; gap: var(--shep-space-head-above); }
.shep-facts-sec > h3 { font-weight: 600; font-size: 15px; margin: 0 0 var(--shep-space-head-below); }
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
   scroll region (the page's, a page's own region side by side, a panel's) keeps a line's room
   under what it brings into view, so the caret stays in view while the box grows. A page's own
   region side by side is its tab's (``#shep-nav`` and ``#shep-research-nav``'s tab items, the
   layout rule below): without the room there, a box partly under that region's foot was focused
   by bringing its first line just into view, its centre left under the runtime line below
   (Settings' evidence folders box at 1366 x 620). */
html, .shep-panel *, #shep-nav > .tabitem, #shep-research-nav > .tabitem {
  scroll-padding-bottom: var(--shep-caret-room);
}
/* **A scroll cue** (R-73): every region that scrolls (any element whose own overflow scrolls;
   no page opts in, the sweep's L7) and has more below fades its last lines out, and draws them
   whole again at its end; one with more past its right edge fades its right edge the same way
   (a table wider than its frame). The shell's head script keeps ``data-shep-more`` and
   ``data-shep-more-x`` on it as it scrolls and as its content changes, whatever the system's
   scrollbars (overlay ones draw nothing at rest). */
[data-shep-more] {
  -webkit-mask-image: linear-gradient(to bottom, #000 calc(100% - var(--shep-more-fade, 28px)), transparent);
  mask-image: linear-gradient(to bottom, #000 calc(100% - var(--shep-more-fade, 28px)), transparent);
}
[data-shep-more-x] {
  -webkit-mask-image: linear-gradient(to right, #000 calc(100% - 28px), transparent);
  mask-image: linear-gradient(to right, #000 calc(100% - 28px), transparent);
}
[data-shep-more][data-shep-more-x] {
  -webkit-mask-image: linear-gradient(to bottom, #000 calc(100% - var(--shep-more-fade, 28px)), transparent),
    linear-gradient(to right, #000 calc(100% - 28px), transparent);
  mask-image: linear-gradient(to bottom, #000 calc(100% - var(--shep-more-fade, 28px)), transparent),
    linear-gradient(to right, #000 calc(100% - 28px), transparent);
  -webkit-mask-composite: source-in;
  mask-composite: intersect;
}
/* **A box folded** (`FOLD`, OPS-FIX-N1): its column has not the room for it whole beside the region
   at its floor (`FOLD_FLOOR`), so its long parts are not drawn; the box's own details hold them. */
[__FOLDED__] .__FOLD_PART__ { display: none !important; }

/* ---------------------------------------------------------------- the layout rule */
/* Side by side (`SIDE`, the shell's one decision): one screen. The selected
   page fills the space under the header; a page taller than that scrolls
   inside its own panel; a panel is never narrower than its page's
   ``--shep-panel-min`` (the decision's own measure). The frame is the window's height, never
   under the floor's (`FLOOR_HEIGHT`): at or above the floor nothing scrolls but the regions
   inside; below it the document scrolls as a whole, the frame drawn as at the floor. */
.shep-page { --shep-panel-min: 0px; }
__SIDE__ {
  & { height: 100%; overflow: auto; }
  & body { height: 100%; overflow: visible; }
  & gradio-app { height: max(100vh, __FLOOR_H__px); display: flex; flex-direction: column; }
  .gradio-container { height: max(100vh, __FLOOR_H__px); min-height: 0 !important; display: flex !important; flex-direction: column; }
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
  /* A level-1 page laid out as a row of panels (`.shep-page`: Diagnosis, Settings) fills its tab,
     so its panels take the page's height and scroll inside, never the tab. */
  #shep-nav > .tabitem > .column:has(> .shep-page) { height: 100%; box-sizing: border-box; }
  .shep-page { min-height: 0; height: 100%; align-items: stretch !important; flex-wrap: nowrap !important; }
  .shep-page > .shep-panel { min-height: 0; max-height: 100%; display: flex; flex-direction: column; min-width: var(--shep-panel-min) !important; }
  .shep-page > .shep-panel > .block { flex: none; }
  /* **A panel sized by what it holds** (M4): a panel of class ``shep-sized`` grows from its
     basis by its share (``--shep-panel-basis``, ``--shep-panel-grow``, in rem or em) in place of
     Gradio's ``scale=``, and never under its ``--shep-panel-min``. */
  .shep-page > .shep-panel.shep-sized { flex: var(--shep-panel-grow) 1 var(--shep-panel-basis) !important; }
  & footer, #runtime_footer { flex: none; }
  /* A panel's header stays where it is when its body scrolls (M5-3: the Test
     Result panel's title scrolled away with the report): a header drawn inside
     the region that scrolls sticks to that region's top. */
  .shep-panel .shep-panel-hd { position: sticky; top: 0; z-index: 6; background: var(--shep-panel); }
  /* **A form drawn dense** (``shep-form-dense``: Training's parameters and its Continue drawer,
     Fine-tune's form; B27-1, NEW-training-finetune-2): each field's padded block, its label and
     its box as tight as drawn, with the field row's measure (``--shep-field-pad-b``) set to the
     same padding, so an action beside the fields still stands on their boxes' line. Stacked,
     nothing scrolls inside and the fields keep Gradio's room. */
  .shep-form-dense { --shep-field-pad-b: 4px; }
  .shep-form-dense .block.padded { padding-top: var(--shep-field-pad-b) !important; padding-bottom: var(--shep-field-pad-b) !important; }
  .shep-form-dense [data-testid="block-info"] { margin-bottom: 4px !important; padding-top: 2px !important; padding-bottom: 2px !important; }
  .shep-form-dense .wrap label { padding-top: 4px !important; padding-bottom: 4px !important; }
  .shep-form-dense input[type="number"], .shep-form-dense input[type="text"] { padding-top: 6px !important; padding-bottom: 6px !important; }
  /* The shared model list's tree (`model_list._tree_html`, on Models and in the Diagnosis load
     drawer) 7 px a side, so it fits the region it is given whole: at 10 px its columns' longest
     words and a 12-hex digest made it 602 px in a 589 px frame (Models at 1440 x 900, M5-3), and
     601 px in the drawer's 524 px list at 1280 x 900, cut at its Fingerprint column. Two classes:
     Gradio's own `.gradio-container-… th` (10 px) comes later and outranks one. */
  .shep-tbl.shep-mtbl th, .shep-tbl.shep-mtbl td { padding-left: 7px; padding-right: 7px; }
}
/* Stacked: the regions stack, the page scrolls, nothing scrolls inside; a
   table wider than its panel scrolls sideways in its own box (it was cut by
   the panel, out of reach: Datasets at 760 x 1000). */
__STACK__ {
  .shep-page { flex-direction: column !important; }
  .shep-page > .shep-panel { width: 100% !important; }
  .shep-tbl-wrap { overflow-x: auto; overflow-y: hidden; }
  .shep-tbl th { position: static; }
  .shep-tbl td { overflow-wrap: break-word; }
  /* "↑ Back to the list" keeps its words' width, on one line; the title wraps its items, so the
     button takes a line of its own, at its end, when the name and its chip leave it no room (it got
     what they left: 57 px at 390 x 844, its words on three lines in its 32 px box; V-23). */
  .shep-facts-title { flex-wrap: wrap; }
  .shep-facts-title > .shep-back-list { display: inline-flex; align-items: center; flex: none; white-space: nowrap; }
}
""".replace("__BRAND__", str(BRAND_WIDTH)).replace("__SIDE__", SIDE).replace("__STACK__", STACK) \
    .replace("__FOLDED__", FOLDED_ATTR).replace("__FOLD_PART__", FOLD_PART) \
    .replace("__NARROW__", NARROW_ATTR).replace("__COL_OPT__", COL_OPT).replace("__COL_TWIN__", COL_TWIN).replace("__FIT_BOX__", FIT).replace("__SEG_WHOLE__", SEG_WHOLE) \
    .replace("__CELL_WHOLE__", CELL_WHOLE).replace("__CELL_ROOMY__", CELL_ROOMY).replace("__ROOMY__", ROOMY_ATTR) \
    .replace("__FLOOR_W__", str(FLOOR_WIDTH)).replace("__FLOOR_H__", str(FLOOR_HEIGHT)) \
    .replace("__FLOOR_VAR__", FLOOR_VAR)


__all__ = ["BRAND_WIDTH", "CELL_ROOMY", "CELL_WHOLE", "COL_OPT", "COL_TWIN", "FADE_VAR", "FIT", "FIT_LEAST", "FIT_WANT", "FIT_WIDE", "SEG_WHOLE", "FOLD", "FOLD_FLOOR", "FOLD_PART", "FOLD_ROOM",
           "FLOOR_HEIGHT", "FLOOR_VAR", "FLOOR_WIDTH", "FOLDED_ATTR", "LAYOUT_ATTR", "NARROW_ATTR", "SHELL_CSS", "SHORT",
           "SHORT_ATTR", "SHORT_BELOW", "SIDE", "SIDE_SHORT", "STACK", "STACK_FLOOR_WIDTH", "ROOMY_ATTR"]
