---
name: webui-design
description: Design rules and the review workflow for the SHEPHERD-Advanced Gradio WebUI. Use whenever adding, changing or reviewing a WebUI screen, label, select, table, help text, status indicator or chart layout (src/webui/**), and before delivering any UI change for review.
---

# WebUI design rules

The rules every WebUI change follows, the author's and the reviewer's alike. They come from the owner's
requirements (BACKLOG 27), the GB10 WebUI acceptance and recheck (2026-10-24/25), and a research pass over design
systems, Gradio, ML tools, clinical decision-support guidance and agent review workflows
(`docs/working/WEBUI_UX_REVIEW_2026-10-25.md`; sources in `references/sources.md`). A rule here is checkable; when a
screen cannot meet one, say why in the delivery.

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
- **Charts are never squeezed.** Each plot keeps a floor (today 140 px plot area; the target is 200–220 px), 2 × 2 when
  the chart area is about 1000 px wide or more, one column below; when the floor does not fit, the chart region
  scrolls. Lines meet 3:1 against the plot; series differ by more than colour (dash pattern shown in the legend too).

## 3. Navigation and what looks clickable

- **Each navigation level looks different.** Level 1 is the header switch (Diagnosis / Research), level 2 the page
  tabs (underline, at least two cues for the selected one), level 3 in-page segmented controls. No two levels share a
  treatment, and in-page radios do not copy the level-1 switch.
- **Group labels are not tabs.** PREPARE and EVALUATE carry no tab box, padding, hover, underline or pointer, sit
  outside the tab row's hit area (an overline above the group, or a smaller label set apart), and are not focusable.
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

The `--shep-status-*` tokens are to be introduced with the first change that touches status marks; until then the
existing `shep-tag-ok/-grey/-amber/-red` classes carry these colours.

| State | Colour token | Shape | Word |
|---|---|---|---|
| OK, on, current | `--shep-status-ok` (green) | ● filled | On / Loaded / Current |
| Off, not loaded, not applicable, unknown | `--shep-status-neutral` (grey) | ○ hollow | Off / Not loaded / — |
| Needs attention, does not block | `--shep-status-warn` (amber) | ▲ | the condition, short |
| Error, refused, blocks the task | `--shep-status-error` (red) | ✕ | the condition, short |
| Running | `--shep-status-info` (blue) | spinner (only here) | Running |

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
- **Budget:** a lead of 90 characters or fewer that answers the question; then at most 3 short bullets or rows; about
  60–80 words in all; at most 2 bold terms; width up to 320 px.
- **Never the only place for must-know facts.** Formats, units, refusals and states that change what happens are
  visible text (`info=` under the field, or the reason line). A bubble explains; it does not warn.
- **Placement never covers the field it explains**: try beside, then above, then below (Floating UI's flip order).
- **Structured, but safe.** The bubble body is built only with `createElement` and `textContent` from a fixed line
  grammar, never parsed as HTML: line 1 is the lead (bold); `- ` lines are a list; `Term: text` lines are a two-column
  key-value table; `**word**` is bold; a blank line starts a paragraph. A test feeds `<img src=x onerror=…>` and must
  see literal text. (This renderer is the planned replacement for today's plain `textContent` bubble.)

## 8. Selects, lists and tables

- Options are 40 characters or fewer, unique, with the distinguishing token first and at most one separator. Over
  about 15 options: filterable; under 5: a radio or segmented control.
- The chosen item's details show in a summary list beside or under the select, never in its option text.
- Per run, at most one option is "latest" and one is "best <metric>", and the details state the rule ("best = highest
  Trainer val MRR among saved files"). A validation figure is named as the Trainer's, never as a test result.
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
- No third-party `gradio_*` components (offline, version drift): build small widgets as `gr.HTML` with `js_on_load`,
  event delegation on `element`, and escape every value in Python. No CDN, no Google Fonts.
- Target only `elem_id`, `elem_classes`, the documented `<elem_id>-button` tab ids and ARIA attributes; keep a tested
  inventory of any Gradio-internal class used. If such overrides keep growing across milestones, record the decision
  between `gr.HTML` islands and a custom frontend on `gr.Server`.

## 12. Before delivering UI work

1. **Screenshots** of every changed page and state at 1850 × 960 (the primary), 1920 × 1080, 1536 × 760, 960 × 1080
   (narrow) and 1080 × 1920 (portrait): drawers open, a bubble open, a read-only browser.
2. **Geometry**, measured in the browser, not judged by eye: no horizontal page scroll; one scroll region; no sticky or
   fixed element over a chart axis or a focused control (`document.elementsFromPoint`); each chart at its floor;
   the same size after a reload and after narrow → wide.
3. **Repeats:** count each 12-hex fingerprint, run id and file name in the visible text of one screen; more than one
   needs a reason.
4. **Words:** a unit test over option builders (≤ 40 characters, ≤ 1 separator) and the help registry (lead ≤ 90
   characters, word budget).
5. **Accessibility:** axe-core run offline (e.g. `axe-playwright-python`, which bundles it) on each page and state;
   contrast of the colour tokens.
6. Attach the screenshots and the measured results to the delivery. A UI claim without that evidence is not done.

## 13. Open decisions (the owner's)

- The 14 px floor against an 11–12 px overline for the PREPARE / EVALUATE group labels.
- Governed texts (the caveats, the D3 label, the §4.9 note) are reworded or re-laid out only with the owner's sign-off.
