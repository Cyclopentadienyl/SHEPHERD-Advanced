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
The design brief and the text brief are `references/design-brief.md` and `references/text-brief.md`; this skill wins.

**How this skill applies.** `AGENTS.md`'s rules on the scale of a fix and when verification stops come first. The
skill says how to do the UI work a task asks for; it never widens a task's scope or adds a gate to it, and a finding
outside the task is recorded for later, not fixed in passing. Its numbers are guidance, except the owner's 14 px text
floor and the WCAG contrast ratios (4.5:1 text, 3:1 non-text).

## 1. The owner's principles

1. **One screen, nothing expanded.** At the deployment window (about 1850 × 960 CSS px; landscape 1080p to 4K) a page's
   main task fits without scrolling the page. When it cannot, one region scrolls inside, never the frame. **The
   arrangement is chosen by width** (R-ARR, the owner's D11 of 2026-11-24, as Apple's and Google's size classes are;
   `shell.area()`): a window wider than tall tries side by side first, which needs the display floor's width and the
   page's panels to fit (`sideFits()`); every other window, a portrait one or a landscape one too narrow for side
   by side (1100 × 1050, 1233 × 640), stacks. Height never changes the arrangement. **The display floor** (the owner,
   2026-11-23) is side by side's: a usable browser area of 1280 × 720 CSS px (`style.FLOOR_WIDTH`,
   `FLOOR_HEIGHT`). At or above it the interface is whole and tidy, the frame never scrolls (the area's one region
   scrolls inside, with its cue), nothing scrolls sideways and nothing is compressed to fit; a side-by-side window
   under 720 px tall is drawn as at the floor, shrinks no further, and the browser scrolls the page as a whole. The stacked arrangement's floor is
   `style.STACK_FLOOR_WIDTH` (864 px, a 1080p screen turned upright at 125 %), with no height floor: at or above it
   every page holds every rule as side by side does, nothing overlapping, covering or cut; only a window under it is
   drawn at it and scrolls sideways. Nothing is built for a window under a floor.
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
- **A region that scrolls ends in a cue over words** (R-CUE, the owner's D4 of 2026-11-24; `shell.py` `fadeAt()`,
  `recut`): its fade, at most about one and a half of the region's lines, is drawn only where it draws the last text
  or value line out (the edge cuts the line, or the line's foot lies in the fade's lower half), a line shown by half
  its height or more above the edge (`shows`: never a sliver of a line's tops in the fade's faintest part); anywhere
  else, the edge in a gap, the region draws the one shared edge cue (a hairline and an inset shadow,
  `data-shep-more-edge`), never a fade over blank space, and a field's box or a control's that starts in that gap goes
  under the edge whole, so its ground never hides the cue. Words kept for a screen reader are no line. Where its last line in view would be a title (a
  heading or a field's label) with every control it heads under the edge, the region ends in the gap above that
  title, with the edge cue. A group title is drawn as a heading so that rule can see it. That cut is the region's
  top state only (H01, the owner's direction of 2026-11-28): a band that holds the chevron, its height with a step
  of the spacing scale clear above and below it (`--shep-more-gap`; 14 px, a least of the mark's own, never a
  figure of its own), shows a small downward chevron in its middle, under the region, while more lies below (one
  mark token at 3:1, no plate, no motion, nothing to press or focus; drawn on the band's own ground by the shared
  pass, never a node); scrolled down by any means, the chevron goes and the region takes the band's room back in
  the same pass (taller by its cut, the block under it unmoved), content passing its edge with the fade or the edge
  cue; back at its top the cut is measured again from what it holds now, read as at its top (a scrollTop of 1 px
  sets the same cut as 0, once). The new look, and what a band under the least should show, await the owner.
- **A message never shares a row with fields or buttons.** Beside a dimmed control stands only its reason's short
  phrase or one line in the page's words (R-WHY, §9: "Needs a workspaces folder", "Choose a dataset to start.");
  anything longer is the full-width note under its group (`.shep-bar-note`), from the group's left edge, at the
  measure.
- **Side-by-side groups share their first lines and their first controls' tops** (R-ALIGN, the owner's A of
  2026-11-24): a note one field alone has (its `info=`) follows its control, never between its label and its control,
  and a title with a note keeps a plain label's room; bottoms may be ragged, with no spacer, `min-height` or stretched
  box to even them.
- **Bars wrap groups whole.** A bar (`.shep-bar`) is a wrapping row of groups (`.shep-grp`, `-grow`, `-end`); a
  group moves to the next line whole, and a text box keeps its floor (`--shep-input-floor`): its group wraps
  first, never a squeezed box. Choice groups are whole too: one line, or one option per line when a whole line
  cannot hold them (the layout pass's mark), never one option alone. In a row of fields a choice group takes
  its one-line width; in a column it keeps the column's width, on its edge. A segmented control whose
  labels must stay whole (`style.SEG_WHOLE`) is a choice group too: one row, or one label a line.
- **One content edge per panel.** Every heading, text, label and field of a panel column starts on one edge
  (`--shep-inset`; Gradio blocks in a panel pad vertically only, so a form's row keeps
  `--shep-gap-between` between its fields and draws no ground of its own). Indent only for subordination:
  list text, a Details body, the words after a status mark (which keeps a column of its own).
- **Panels sized by what they hold** (`.shep-sized`: `--shep-panel-basis`, `--shep-panel-grow`,
  `--shep-panel-min`, in rem or em), not by `scale=` shares fitted to one window.
- **A field's width comes from its role** (R-WIDTH, the owner's D12 of 2026-11-24): a number `--shep-w-num` (10ch), a
  name `--shep-w-name` (28ch), a path or a search growing along its row from `--shep-w-grow-min` (24ch), a choice its
  longest option (`shep-w-num`, `-name`, `-path`, `-search`); white space beside a short field is fine. An accordion's
  mark sits by its title and the whole title row toggles. A button is its words' width (the least specific default);
  it spans only as the one drawn action of a fieldless area no wider than its panel's widest field, with nothing
  beside it (`shell.spans()`), and a column that only wraps one action is that action's width.
- **Charts are never squeezed.** Each plot keeps a floor (`PLOT_MIN_HEIGHT`, 170 px of plot area; about 200 px
  at the owner's window, where the four plots measure 198 px each, the shared epoch axis labelled once under them),
  judged by the chart's container, never the window: stacked, 2 × 2 from a figure of `charts.TWO_COLUMNS_FROM`,
  one column below; side by side, always 2 × 2, each plot at its floor, or the chart region scrolls. Lines
  meet 3:1 against the plot; series differ by more than colour, and the legend and tooltip swatches draw each
  series' own dash pattern.

### Text layout

- **Measure and rhythm.** Prose takes `max-width: var(--shep-measure)` (56ch: about 73 characters a line);
  tables, key-value rows, options and code rows are exempt. Spacing comes from the tokens, never from
  `<br><br>` or a first-line indent: item 4 px < paragraph 8 px < above a heading 16 px, below a heading
  4 px; line height 1.45 for text that wraps, 1.3 for one or two lines. Text that can wrap is start-aligned.
- **Message shape** (reasons, refusals, notices, empty states): one lead of 20 words or fewer with the state
  and its consequence, then the next step or who can act, as a sentence or one link; why and how as term rows
  under `Details ›`, which closes the lead's last line where the line has room and opens its rows under it, on the
  words' edge (R-DETAILS, `renderers.notice_block`); environment variables, lock files, guide sections and codes only
  there or in Ref. No semicolons, "please", "sorry", "oops", "invalid", "an error occurred" or "we".
- **Literal tokens** (paths, variables, flags) in the monospace token (`code`, `.shep-mono`), at least 14 px,
  breaking only before their own separators (`renderers.token_html`), never `break-all` on words, never U+200B.

## 3. Navigation and what looks clickable

- **Each navigation level looks different.** Level 1 is the header switch (Diagnosis / Research; its accessible name
  is "Area"), level 2 the page tabs (underline, at least two cues for the selected one; a tab under the pointer gets a
  fill, never the underline), level 3 in-page segmented controls. No two levels share a treatment, and in-page radios
  do not copy the level-1 switch.
- **The research tabs are drawn in groups** (decided by the owner, 2026-11-22): "Datasets │ Training Fine-tune
  │ Test Models" (`shell.TAB_GROUPS`). The groups are not stages (Datasets is still Prepare): a gap before each
  later group's first tab and a decorative line in it (an empty `::before`: no node, role, focus stop or text),
  drawn only while the tabs sit on one line; no tab moves and the tab order is unchanged.
- **The research stage is a hint, not a tier of labels.** The research page tabs sit in one row with no group captions
  above them; the stage of the page shown is plain text at the right of that row ("Stage: Prepare", "Stage: Evaluate"),
  from the one mapping `shell.STAGES`. It is no control: no tab box, hover, underline, pointer, focus stop, role or live
  region. It follows the page shown however it was opened, never a running job, and none shows off the research pages.
  When the row has no room it takes a right-aligned line of its own; nothing is shrunk to fit.
- **Labels are text, not chips.** Field labels are plain 14 px/600 text; a tinted pill reads as a tag or a button.
  Static headings get no background.
- **A lock means "refused".** A gated button shows no lock while the operator may use it; when D8 refuses, the button
  is dimmed, shows the lock and has R-WHY's phrase "Operator controls only" beside it (§9), which opens the D8 chip's
  answer for this request.

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
| A lineage (what was made from what) | Node-link tree drawn as DOM: one node per run, each edge from the exact parent file; nested lists, per node a body button (selects) and, with children, a sibling arrow button (folds only); an idle node is as large as its text, the arrow area shows on hover or keyboard focus and always on touch; details stay in the Details column | Models › By family |

Text that runs past two lines gets structure: an answer-first lead, then bullets or key-value rows. **Information
load** (R-LOAD, the owner's B of 2026-11-24): adjacent text with no control or heading between stays within about
three rendered lines, or it becomes key-value rows or goes behind `Details ›`; a summary shows the facts its stage
needs, declared as data (Fine-tune's `PARENT_STAGE_FACTS`), the rest behind its one Details. A reviewer judges each
cluster; no walk rule counts lines.

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
- **A header chip that names no holder has four states, never one** (R-STATE, the owner's "unread" of 2026-11-24;
  `chips.py`): not set up (a missing setting, "GPU jobs: not set up": grey ○), not read yet (before the first read,
  "… not read yet": grey, no mark), could not be read (cannot be verified: grey ○) and lock not usable (it refuses
  every managed job: red ✕). Each is its own kind, `data-state`, class and words; a page reads them by kind
  (`chips.LOCK_UNUSABLE`), never by their words.

## 7. Help bubbles (ⓘ)

- A bubble is a **toggletip**: click, Enter or Space opens and pins it; Esc and an outside click close it; hover and
  focus preview it (hover after a short delay). Pinned, it is a dialog; a hover preview is a tooltip.
- **Budget (guidance):** a lead of about 90 characters that answers the question (aim at 78: two lines of a
  320 px bubble); then none or 2–4 items or term rows of one form, each 76 characters at most; about 60–80
  words in all; Ref last. `help.budget_problems(text, help.TEXT_BUDGET)` names what a text runs over. Every
  registered text is held to it (`test_webui_help.py`); the one exemption is words kept verbatim from their
  source: a lead that is its field's `FieldSpec.description` (API wording) and a governed note.
- **A bubble is judged by four things** (R-HELP, the owner's C9 of 2026-11-24): it covers no control, it reads at
  4.5:1, every trimmed fact stays reachable (Details or the Ref line), and its facts are complete: up to four items
  when needed, different facts never merged to reach three, never padded to two. The two-line count in the 320 px
  bubble is a note, not a gate (fonts and zoom change it); the budget above stays the gate. In a setup topic the GUI
  path and the desktop start come before environment variables and SSH.
- **Never the only place for must-know facts.** Formats, units, refusals and states that change what happens are
  visible text (`info=` under the field, or a dimmed control's phrase or need line). A bubble explains; it does not
  warn.
- **Placement never covers the field it explains**, nor, for a reason's phrase, the controls the phrase names
  (`data-for`). `help.py`'s `place()` tries beside, above and below, sliding along each side, and takes a place that
  covers no other control; where none holds the whole bubble, it opens shorter in the tallest place that covers none
  and still shows its head, its lead and two lines, its text scrolling inside under the shared cue, and where no place
  of its width is clear, the same at a narrower width (220 px at the least); only where no such place exists does it
  take, of the places tried, the one over the fewest controls. That least (its head, its lead and two lines) is the
  floor, kept by the owner's F4 of 2026-11-25: a bubble is not compressed further, so that it stays readable; one
  bubble is open at a time and the scroll cue says there is more to read, so the few places where it still lies over
  another control are accepted. A reason's phrase opens this same bubble, anchored at
  the phrase and reading its source's hidden text by id.
- **One bubble, extended only when needed.** Today's bubble (`src/webui/components/help.py`) reads every text by one
  small grammar, defined there once (`BULLET`, `TERM_PATTERN`, `REF_PREFIX`, read in Python by `help_lines`) and given
  to its script as written: the first line is the lead, shown bold; "- " lines form one list; a line led by a short
  capitalised "Term: " shows the term bold; a "Ref:" line, last, holds codes and source references, muted; any other
  line is a paragraph. When a text needs more structure than that, extend that one shared component and its grammar;
  do not add a second tooltip mechanism, and do not make every text take on more structure in advance. Whatever it
  renders is built with `createElement` and `textContent`, never parsed as HTML; a test feeds `<img src=x onerror=…>`
  and must see literal text.

## 8. Selects, lists and tables

- Options are 40 characters or fewer, unique, with the distinguishing token first and at most one separator. Over
  about 15 options: filterable; under 5: a radio or segmented control.
- The chosen item's details show in a summary list beside or under the select, never in its option text.
- Checkpoint options say what the file is ("Latest save", "Top-k save") and how far it trained (`completed_epochs`;
  an older file without it says "recorded epoch"). If "best" is used, it means the best value of the run's own
  recorded selection metric, in its recorded direction, among this run's saved files (the checkpoint's
  `callbacks.model_checkpoint` `monitor` and `mode`: a base selects on `val_lp_accuracy`, a task model on
  `val_pooled_mrr`, both higher is better; `checkpoint_paths.ranking_score_detail` reads the value), named by that
  metric and read from the list's metadata, never from file names; ties may share it; an LP figure is never called
  MRR; a missing score, weights only, early stop and extend limits stay visible. A word on an option never changes its
  value, digest or permission. A validation figure is named as the Trainer's, never as a test result.
- Tables: a title with its count, a one-line description, the name column first (pinned when wide), selection opens
  details beside the list without leaving it. No ARIA `aria-selected` on rows of a plain `table`.

## 9. Empty, waiting and refused states

Every empty region says, in its own words per cause, what will appear and the next step, with at most one action:
first use, waiting for a job, filtered to nothing, not recorded, refused. **First use** (no workspaces folder, no
results registry): each cause once per screen, where it empties a region, through the one notice
(`renderers.notice_block` / `notices_block`, from `server_settings.Notice` via `workspaces.setting_notice` and
`lock_notice`), its GUI path as the link ("Choose one in Settings › Data locations"), the rest under Details ›;
every control it dims has the notice's phrase beside it (R-WHY, below), never the notice again. An empty state is in
the flow (`.shep-empty`), never an overlay. A refusal says who can act ("Operator controls only", D8's reason one
activation away). Refusals stay visible inline next to what they disable; nothing auto-dismisses.

**Why a control is dimmed** (R-WHY, the owner's D6 of 2026-11-24, and SW-C11): one reason source per cause
(`renderers.REASONS`: a setting's notice, or a header chip's hidden text), its whole reason drawn once per screen.
Beside each dimmed control stands either its cause's phrase, five words at most, a focusable link-styled button keyed
to that source (`reason_phrase`: a notice's is brought into view with the focus on its link, a chip's opens the help
bubble at the phrase), or, for a cause of the page's own, one line in the page's words led by the need
(`need_words`: "Choose a dataset to start.", "Load a model first."). Where the cause's own notice is drawn in the same
bar as the controls it dims, nothing is added beside them: the notice names them (`notice_block(describes=…)`). The
control's `aria-describedby` names its phrase or notice. Two causes are never one phrase, and the words never say
"disabled", "see above", "see below" or "(see <place>)": the dimmed look says the control cannot be used.

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
  inventory of any Gradio-internal class used (since the 2026-11-22 sweep: `fieldset.block` with its options'
  `.wrap`, `.wrap:has(> .wrap-inner)`, `.block.padded` and `--block-padding`, a panel's `.form` ground and
  gap, `.row` as a row of fields, a `gr.HTML`'s `.html-container` and the element inside it, empty while it
  says nothing).
- Gradio's reset (`.gradio-container-<version> p, ol, ul { margin: 0; padding: 0 }`) is one class and an
  element: a rule that spaces a paragraph or a list needs two classes. If such overrides keep growing across milestones, record the decision
  between `gr.HTML` islands and a custom frontend on `gr.Server`.

## 12. Before delivering UI work

Choose the checks by the change's risk, run them once, and say in the delivery what was checked and how. Do not rerun
checks that already passed for the same code.

| Change | Checks |
|---|---|
| A new screen or a changed layout (before it is built) | a static mockup first, its small sources committed with a README that says who checked it (self-checked or owner-approved; e.g. `docs/working/ops-wiring-and-ui-ux/mockups/`); without the owner's approval it is never called accepted (B27-P) |
| Any visible change (when it is delivered) | before and after screenshots of what changed, at the owner's window (about 1850 × 960) and the display floor (1280 × 720), named in the delivery and in the ledger row it closes (B27-P) |
| Wording, a label, an option or help text | the unit test that pins it (options: unique, short, one separator; help: the lead answers the question) |
| A colour or theme token | the contrast test of the tokens it touches |
| A widget or renderer (HTML, JS) | a unit or Node test of what it renders, escaping included; one look in the real app |
| Layout or scrolling | geometry measured in the browser at the owner's window (about 1850 × 960) and the display floor (1280 × 720): no horizontal page scroll, one scroll region, nothing sticky over a chart axis or a focused control (`document.elementsFromPoint`), the same after floor → wide; a screenshot of each. Pass conditions are relative to what is drawn (inside its row, off its neighbours, the same control under the pointer), never a named window or a fixed pixel figure |
| A broad layout change across pages | the above at the check sizes (1850 × 960, 1550 × 830, 1366 × 768, 1280 × 720, 2560 × 1440, one window below the floor, drawn as at the floor, a landscape window too narrow for side by side (1100 × 1050) and the same window turned (1050 × 1100), stacked, and the portrait windows 1080 × 1920 and 1440 × 2560, stacked), a read-only browser, and axe-core offline (e.g. `axe-playwright-python`) |
| A repeated fact (rule 1.2) | count the fingerprint, run id or name in one screen's visible text; more than one needs a reason |

The layout, scrolling, type, border and focus rows are measured on every page at the representative sizes by one
shared check, which a layout, token or navigation change runs rather than a script of its own:
`tests/unit/test_webui_shell.py::test_every_page_at_every_size_passes_the_ui_measurement_check` (`walk_ui` and
`ui_problems` in `tests/fixtures/browser.py`; its screenshots in the test's temporary `ui-walk*/shots`). **The
layout rules** (beside, narrow-box, overlap, orphan, centred, split-group, edge, clipped, empty-region,
help-on-page, notice-repeat, tab-groups, in-view and sideways-scroll at every check size at or above the floor,
scroll-cue, edge-heading, wrap-end, disabled-reason, dimmed-words, reason-repeat, pointer-words, token-break, sep-end,
chain-break, row-top, row-control-top, row-height, floor: below it the marks are the floor's and the page scrolls,
and settle: a stop is measured once three frames in a row are the same, read at every frame, and one that never
holds still is a finding; `layout_problems`) are measured in the same walk and, with the
**first-use and empty-folder walks** (`walk_state("first_use" | "empty", …)`, `layout_stops`), held at zero on
every page at `LAYOUT_SIZES` (1850 × 960, 1550 × 830, 1366 × 768, 1280 × 720, 2560 × 1440, 1280 × 640 under the
floor, 1100 × 1050 a landscape window that stacks and 1050 × 1100 the same window turned, 1080 × 1920, 1440 × 2560
and 1850 × 960 again) in all three
states, first use, empty and configured, and configured as a read-only
viewer (`test_every_page_of_every_state_has_no_layout_finding`); a page's change runs them on its own pages at first
use and configured. The same
walk counts each fingerprint drawn in a screen's visible text (`repeat_problems`), and walks PLAN_TEST_RESULTS §6's
other sizes with Diagnosis' largest Top-K answered on long disease names, and every page as a read-only viewer
(`acceptance_walk`). axe-core is not installed here: those rows' accessibility checks are the walk's own
measurements until it is.

## 13. Decided

- **Decided (2026-10-25):** the 14 px floor stays; the PREPARE / EVALUATE group labels are set apart by position,
  spacing and weight, not a smaller size.
- **Decided (2026-10-25):** governed texts (the caveats, the D3 label, the §4.9 note, R10) may change their
  presentation within fixed semantics without a line-by-line sign-off: R10 keeps each cell's state, k/n, units, cases
  and unverifiable reasons, and an unverifiable cell never reads 0, — or "no overlap"; Mode C never implies an online
  total; the key warnings (ordering by test score uses up the cohort's independence) stay visible where the choice is
  made; denominators, thresholds and the ordering and registration policy do not change. A change to the semantics is
  the owner's. *(2026-11-12: ordering by test score retired in K8-6d and Mode C's measurement in K8-6a and K8-6b, so
  no page offers that choice; kept Mode C records are still read and shown by the shared readers.)*
- **Decided (2026-11-21, supersedes the first 2026-10-25 line's group labels):** the PREPARE / EVALUATE captions above
  the research tabs are retired; the stage is a plain-text hint at the right of the tab row (§3). The owner and the
  external reviewer found the captions read as a second tab tier and the hovered tab's underline as a second selection.
- **Decided (2026-11-22, the owner):** the research tabs are grouped "Datasets │ Training Fine-tune │ Test
  Models" with decorative separators drawn only while the tabs sit on one line (§3); the Stage hint is unchanged.
  First use is said once per screen by the one notice, its GUI path as the link (§9); text follows the measure,
  the spacing tokens and the message shape (§2).
- **Decided (2026-11-24, the owner, after the L7 package):** the arrangement is chosen by width (§1.1); a lock's
  whole reason is said once, with a short phrase beside each control it dims (§9); not set up, not read yet, could
  not be read and lock not usable never share one state or one vague word (§6); a help bubble is judged by covering no
  control, readability and a reachable full reference, its two lines a reminder (§7); a region that must scroll has a
  recognisable cue, never a lone title at its edge or a fade over blank space (§2); a path's source keeps a line of
  its own; Firefox and Gradio are not patched for a single word's wrap. A screen that scrolls is judged by three
  questions: can the reader see the next step, find all of the content and see the focus.
- **Decided (2026-11-21):** Models › By family is a node-link family view drawn as DOM, with no graph library, because
  DOM nodes give this view real buttons, keyboard focus and a hierarchy that assistive technology reads, with no new
  dependency: a choice for this view, not a rule against a graph library later. The view data stays runs
  plus edges keyed by the exact parent file, so a layout engine could place the same nodes if lineage ever has several
  parents. Its orientation follows the measured room (left to right when it fits, else top to bottom), decided only when
  the graph's content, the region's width or the fonts change, never by a hover, a fold or a selection. A body pick
  never scrolls or re-orients the graph; the room of a line gone above it is kept until the pointer and focus leave,
  then given back as scroll as far as the region has scrolled.
  Details' "Saved files" offers only the files the list draws. No run panel and nothing sticky: a panel under a tall
  graph is out of view after a click. A graph wider than its region pans inside the region, never the page.
- **Decided (2026-11-25, the owner, revising D9 the same day):** a choice that changes only the viewer's own page is a
  **display preference**, kept in that browser (⚙ Settings › Display, `components/display.py`: `localStorage`, every
  access wrapped, a missing, unreadable or unknown value the default), with no Apply, no server setting and no
  operator lock, so a read-only viewer may change it too. It is read in the head before anything is painted (a mark
  on the root element that the style sheets and the page's scripts read), so no frame shows another state, and a
  change applies at once to the open page unless its reader acted on that control there. The first one: stacked,
  Training's Parameters start Shown (the default, the page scrolling), Folded, or Automatic (the fold by the room,
  the run view kept invisible in its place until the fold has decided, a few seconds at most, so it never jumps);
  the reader's own Hide or Show wins for the page session. Side by side nothing changes.
