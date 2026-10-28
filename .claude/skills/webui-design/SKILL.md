---
name: webui-design
description: Design rules and the review workflow for the SHEPHERD-Advanced Gradio WebUI. Use whenever adding, changing or reviewing a WebUI screen, label, select, table, help text, status indicator or chart layout (src/webui/**), and before delivering any UI change for review.
---

# WebUI design rules

The rules every WebUI change follows, the author's and the reviewer's alike. They come from the owner's
requirements (BACKLOG 27), the GB10 WebUI acceptance and recheck (2026-10-24/25), and a research pass over design
systems, Gradio, ML tools, clinical decision-support guidance and agent review workflows
(`docs/working/WEBUI_UX_REVIEW_2026-10-25.md`; sources in `references/sources.md`). When a screen cannot meet a rule,
say why in the delivery.

**How this skill applies.** `AGENTS.md`'s rules on the scale of a fix and when verification stops come first. The
skill says how to do the UI work a task asks for; it never widens a task's scope or adds a gate to it, and a finding
outside the task is recorded for later, not fixed in passing. Its numbers are guidance, except the owner's 14 px text
floor and the WCAG contrast ratios (4.5:1 text, 3:1 non-text).

## 1. The owner's principles

1. **One screen, nothing expanded.** At the deployment window (about 1850 × 960 CSS px; landscape 1080p to 4K) a page's
   main task fits without scrolling the page. When it cannot, one region scrolls inside, never the frame.
2. **Each fact once on a screen.** A fingerprint, a name, a state or a number appears in one place, where it reads
   naturally and stands out. Elsewhere it is referred to by a short relation ("same dataset as the run bar"). Repeat
   it only where the two places can differ (a result's own identity against what is selected).
3. **Pick the form by the content's shape, not plain text by default** (section 4).
4. **Status is colour, shape and word together** (section 6): the owner's traffic lights, readable without colour.
5. **Low-priority detail is one click away** behind a control with a specific label ("Details ›"), never two levels
   deep (no popover inside a popover, no accordion inside a drawer).

## 2. Layout, hierarchy and spacing

- **Three text sizes, two weights.** Region title 20 px, body and values 15 px, keys and meta 14 px (the floor: nothing
  under 14 px); weights 400 and 600. At most two large elements per page, the primary one largest. Squint test: blur
  a screenshot 5–10 px; the main region (results, charts, table) must still dominate.
- **One spacing scale** (2/4/8/12/16/24/32 px): 8 px or less inside a group, 16–24 px between groups; a heading sits
  closer to its content than to the group above. One container level per region: no bordered box inside a box.
- **One scroll region per area.** Header and status rows are in flow, the content row takes `minmax(0, 1fr)` and is
  the only thing that scrolls. Nothing sticky or fixed may cover a chart's plot or axes, a table row or a focused
  control; a status line is a row of its own (the run view: header, charts, resources).
- **Charts are never squeezed.** Each plot keeps a floor (today 140 px plot area; about 200 px is a later target),
  2 × 2 when the chart area is about 1000 px wide or more, one column below; when the floor does not fit, the chart
  region scrolls. Lines meet 3:1 against the plot; series differ by more than colour, and the legend and tooltip
  swatches draw each series' own dash pattern.

## 3. Navigation and what looks clickable

- **Each navigation level looks different.** Level 1 is the header switch (Diagnosis / Research), level 2 the page
  tabs (underline, at least two cues for the selected one), level 3 in-page segmented controls. No two levels share a
  treatment, and in-page radios do not copy the level-1 switch.
- **Group labels are not tabs.** PREPARE and EVALUATE carry no tab box, padding, hover, underline or pointer, sit
  outside the tab row's hit area (set apart by position, spacing and weight at 14 px), and are not focusable.
- **Labels are text, not chips.** Field labels are plain 14 px/600 text; a tinted pill reads as a tag or a button.
  Static headings get no background.
- **A lock means "refused".** A gated button shows no lock while the operator may use it; when D8 refuses, the button
  is disabled, shows the lock and the visible "Operator controls only: …" reason beside it.

## 4. Choosing the form

| Content | Form | Example |
|---|---|---|
| One thing, several attributes | Key-value summary list (`dl`: key column 8–12em, value, optional Copy) | Run details, the chosen checkpoint, a model's identity |
| Many things, the same fields, compared | Table (rows × columns), 32–40 px rows, 1–2 word headers | Models, reports, the R10 overlap matrix (training / validation × disease / phenotype set) |
| Choose one of many | Short options; two cascading selects for a hierarchy; a single-select table when options need columns | Run → Checkpoint |
| How to do something | Numbered steps | Continue: Run → Checkpoint → Check → Continue Training |
| Two things that differ | Comparison table, differences only | the parent checkpoint against the run shown |
| A quantity or proportion | A bar or progress line with the number | 195 / 1000 epochs, 10 of 13 cases ranked |
| A state | Status indicator (section 6) | Operator controls, GPU, a model loaded |
| An explanation | One-sentence answer, then facts as key-value rows, the narrative collapsed | Evidence for a candidate |

Text that runs past two lines gets structure: an answer-first lead, then bullets or key-value rows.

## 5. Words

- **Front-load.** The first two words (about 11 characters) carry the meaning of a label, option, row or help lead.
- Sentence case and full words ("Learning rate", "Graph layer type"); internal codes (Mode C, D3, R10, SHEPHERD_*)
  only beside a plain label, or in a trailing "Ref:" line.
- Sentences up to 20 words (25 at most), paragraphs up to 3 sentences.
- **No dot-joined chains.** At most one " · " in a label, option or row; facts that need more go into a key-value list.
- **Identity in three layers:** a short human label in titles, lists and options (name · architecture · time); the
  12-hex fingerprint with an icon Copy in that object's details; the full provenance one click away. Option values keep
  the full key (Gradio `(label, value)` tuples), so shortening a label never changes what is chosen.
- An uncalibrated number is named for what it is ("Score", never "Confidence" or "Probability").
- Numbers are right-aligned with tabular figures; units are said once, in the header or key.

## 6. Status indicators (traffic lights)

The `--shep-status-*` tokens exist since K6 (`src/webui/style.py`), as aliases of the tag colours with no new hex; the
existing `shep-tag-ok/-grey/-amber/-red` classes carry the same colours.

| State | Colour token | Shape | Word |
|---|---|---|---|
| OK, on, current | `--shep-status-ok` (green) | ● filled | On / Loaded / Current |
| Off, not loaded, not applicable, unknown, cannot be verified, absent | `--shep-status-neutral` (grey) | ○ hollow | the state's own word (below) |
| Needs attention, does not block | `--shep-status-warn` (amber) | ▲ | the condition, short |
| Error, refused, blocks the task | `--shep-status-error` (red) | ✕ | the condition, short |
| Running | `--shep-status-info` (blue) | spinner (only here) | Running |

- **Missing states keep their own words.** Off, Not loaded, Not applicable, Unknown, Cannot be verified (unverifiable)
  and Absent (not recorded) say different things and are never merged into one "Off" or "—". A dash means only "no
  value here" where nothing else is meant; an unverifiable result never reads 0, — or "none".
- Colour never carries meaning alone (WCAG 1.4.1): the shape and the word do. Show the word beside the mark; a mark
  without its word is allowed only where space forces it, and then it gets a bubble on hover **and** keyboard focus
  plus an `aria-label`. Hover-only text is not enough (keyboard, touch, screen readers).
- Red is for errors and refusals only; "disabled" or "off" is grey. Normal states are quiet (plain text or neutral);
  only states to watch get a coloured lozenge.
- No raw status hex outside the tokens. Environment warnings lead with the consequence ("No CUDA device visible: GPU
  training is unavailable"); versions sit behind a disclosure.

## 7. Help bubbles (ⓘ)

- A bubble is a **toggletip**: click, Enter or Space opens and pins it; Esc and an outside click close it; hover and
  focus preview it (hover after a short delay). Pinned, it is a dialog; a hover preview is a tooltip.
- **Budget (guidance):** a lead of about 90 characters that answers the question; then a few short bullets or rows;
  about 60–80 words in all; a width of about 320 px.
- **Never the only place for must-know facts.** Formats, units, refusals and states that change what happens are
  visible text (`info=` under the field, or the reason line). A bubble explains; it does not warn.
- **Placement never covers the field it explains**: try beside, then above, then below (Floating UI's flip order).
- **One bubble, extended only when needed.** Today's bubble (`src/webui/components/help.py`) shows plain text with
  paragraphs. When a text needs more structure (a lead, a list, key-value rows), extend that one shared component; do
  not add a second tooltip mechanism, and do not impose a grammar on every text in advance. Whatever it renders is
  built with `createElement` and `textContent`, never parsed as HTML; a test feeds `<img src=x onerror=…>` and must see
  literal text.

## 8. Selects, lists and tables

- Options are 40 characters or fewer, unique, with the distinguishing token first and at most one separator. Over
  about 15 options: filterable; under 5: a radio or segmented control.
- The chosen item's details show in a summary list beside or under the select, never in its option text.
- Checkpoint options say what the file is ("Latest save", "Top-k save") and how far it trained (`completed_epochs`;
  an older file without it says "recorded epoch"). If "best" is used, it means the highest Trainer val MRR recorded
  among this run's saved files, read from the list's metadata, never from file names; ties may share it; a missing
  score, weights only, early stop and extend limits stay visible. A word on an option never changes its value, digest
  or permission. A validation figure is named as the Trainer's, never as a test result.
- Tables: a title with its count, a one-line description, the name column first (pinned when wide), selection opens
  details beside the list without leaving it. No ARIA `aria-selected` on rows of a plain `table`.

## 9. Empty, waiting and refused states

Every empty region says, in its own words per cause, what will appear and the next step, with at most one action:
first use, waiting for a job, filtered to nothing, not recorded, refused. A refusal says who can act ("Operator
controls only: request not local"). Refusals stay visible inline next to what they disable; nothing auto-dismisses.

## 10. Accessibility floor

Text contrast 4.5:1 (placeholders and `info=` included; never use a placeholder for information); borders, focus rings,
selected indicators and chart lines 3:1; every interactive element shows focus; targets at least 24 × 24 px (the ⓘ and
Copy buttons included); correct roles (tablist/tab, listbox/option, dialog for a pinned bubble).

## 11. Gradio 6.30 notes

- `gr.Dropdown` options are plain text: no per-option description, no grouping, no two-line options. Use
  `(label, value)` tuples, cascade with `.change` (or `.select`, which carries the index), and put the chosen item's
  facts in `info=` (markdown; HTML escaped) or a summary list. The options list fades out for 200 ms on blur and
  reopens on focus: screenshots and tests wait for it to detach, and Print Screen catches it half-transparent.
- Gradio writes inline styles on blocks (e.g. `overflow: visible`): a scroll region needs `!important`.
- Theme variables reach Gradio's own pieces: `--block-title-background-fill` (label chips), `--input-border-color`,
  `--block-info-text-color`, `--input-placeholder-color`.
- `gr.Sidebar` is a fixed full-height overlay: not for page navigation. Keep page navigation in `gr.Tabs`
  (`overflow_behavior="wrap"`, Settings with `alignment="right"`).
- A third-party `gradio_*` component is judged on its merits: it must work offline (no CDN, no Google Fonts), be
  pinned, be maintained and render safely (escape what it shows). Otherwise build the small widget as `gr.HTML` with
  `js_on_load`, event delegation on `element`, and every value escaped in Python.
- Target only `elem_id`, `elem_classes`, the documented `<elem_id>-button` tab ids and ARIA attributes; keep a tested
  inventory of any Gradio-internal class used. If such overrides keep growing across milestones, record the decision
  between `gr.HTML` islands and a custom frontend on `gr.Server`.

## 12. Before delivering UI work

Choose the checks by the change's risk, run them once, and say in the delivery what was checked and how. Do not rerun
checks that already passed for the same code.

| Change | Checks |
|---|---|
| Wording, a label, an option or help text | the unit test that pins it (options: unique, short, one separator; help: the lead answers the question) |
| A colour or theme token | the contrast test of the tokens it touches |
| A widget or renderer (HTML, JS) | a unit or Node test of what it renders, escaping included; one look in the real app |
| Layout or scrolling | geometry measured in the browser at the owner's window (about 1850 × 960) and the narrow layout: no horizontal page scroll, one scroll region, nothing sticky over a chart axis or a focused control (`document.elementsFromPoint`), the same after narrow → wide; a screenshot of each |
| A broad layout change across pages | the above at more sizes (1920 × 1080, 1536 × 760, 960 × 1080, 1080 × 1920), a read-only browser, and axe-core offline (e.g. `axe-playwright-python`) |
| A repeated fact (rule 1.2) | count the fingerprint, run id or name in one screen's visible text; more than one needs a reason |

## 13. Decided

- **Decided (2026-10-25):** the 14 px floor stays; the PREPARE / EVALUATE group labels are set apart by position,
  spacing and weight, not a smaller size.
- **Decided (2026-10-25):** governed texts (the caveats, the D3 label, the §4.9 note, R10) may change their
  presentation within fixed semantics without a line-by-line sign-off: R10 keeps each cell's state, k/n, units, cases
  and unverifiable reasons, and an unverifiable cell never reads 0, — or "no overlap"; Mode C never implies an online
  total; the key warnings (ordering by test score uses up the cohort's independence) stay visible where the choice is
  made; denominators, thresholds and the ordering and registration policy do not change. A change to the semantics is
  the owner's.
