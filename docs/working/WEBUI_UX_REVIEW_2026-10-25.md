# WebUI design review, 2026-10-25

The owner asked, after the WebUI recheck at `4764faa`, for a look at UI/UX references, guidance, community
discussion and reusable skills; an audit of the implemented pages against them; and a judgement on whether the fixes
are light enough to finish within M5 or need a phase of their own, or a redesign. The rules that came out of it are in
the project skill `.claude/skills/webui-design/SKILL.md` (sources in its `references/sources.md`). This is a review
record: it changes no code, and what is done next is the owner's decision, with the reviewer, for M5's close.

## 1. Method

- **Research**, five independent passes: design guidance and design systems (Nielsen Norman Group, IBM Carbon,
  GOV.UK and NHS, PatternFly, WCAG 2.2); Gradio 6.30's own sources, docs, issues and well-made Spaces (Hugging Face's
  Trackio among them); ML experiment-tracking and model-registry UIs (W&B, MLflow, TensorBoard, ClearML, Trackio);
  clinical decision-support guidance and phenotype tools (FDA CDS guidance, AHRQ/ONC, LIRICAL, Exomiser, Isabel);
  agent skills and review tooling (Anthropic's skills format, Vercel's web interface guidelines, a Playwright
  design-review agent, axe-core). About 250 sources were opened; 79 checkable principles and 71 implementation
  references came back.
- **Audit**, three page groups (the shell, Diagnosis and Settings; Training and Datasets; Test and Models), from 15
  screenshots at 1847 × 964 CSS px (the owner's window) and 960 px wide, and from the source. The fixtures are the
  scratch demo tree, not the GB10; Diagnosis was seen with no model loaded.
- Each finding was rated by severity (high: misleads, hides data or blocks a task; medium: slows or confuses; low:
  polish), effort (S under 2 h, M under a day, L more) and scope (light adjustment: wording, CSS, markup within the
  layout; structural: regrouping within a page; redesign: a new page layout or navigation). The high findings' evidence
  was checked against the code before this record was written.

## 2. Verdict: adjust, do not redesign

All three audits found the structure sound and **no finding needs a redesign**: 83 findings, 70 light adjustments and
13 structural; 8 high, 50 medium and 25 low (the counts of `review.json`). The skeleton already follows what the
research recommends: two navigation levels with four Research pages (a side navigation pays off only above about
five), list-with-details pages (as ClearML and MLflow do), Diagnosis reading left to right from input to ranked list
to details, a run view whose charts scroll in their own row, one help registry, fingerprints with Copy, refusals kept
visible, everything served offline. What makes the pages feel overloaded sits in a few shared layers, so each fix
spreads to every page:

1. **Theme tokens.** Field labels are tinted chips that read as tags or buttons; input borders are almost invisible
   (about 1.05:1 to 1.5:1); placeholders and `info=` text, a refusal among them, use Gradio's subdued grey at 2.54:1;
   four text sizes and four weights too close to form a hierarchy; status colours as raw hex.
2. **Text habits.** Facts joined with " · " into run-on labels, options and rows; help texts of 90–190 words in plain
   paragraphs; internal codes (Mode C, D3, R10) where a plain label belongs; the same fact repeated on one screen
   ("No model loaded" four times on an empty Diagnosis page; a dataset fingerprint four times on Models).
3. **The help bubble.** Plain text only (no lead, emphasis, list or table), 360 px wide, and placed over the field it
   explains.
4. **Navigation and affordance.** PREPARE and EVALUATE sit on the tab row at a tab's weight and read as tabs; the
   level-1 switch and the in-page radios share one filled style; a lock is drawn on buttons the operator may use.

## 3. The high findings

"M5" marks what the close-out batch of 2026-10-25 changed (section 4); "Later" is a later improvement (section 4).

| # | Page | Finding | Fix | Status |
|---|---|---|---|---|
| 1 | Training | The Device help said `auto` falls back to the CPU; the trainer refuses `auto` without CUDA, so the default could start a refused run with nothing on the page saying why | Both help texts; regression test | Fixed in `e54a5f1` |
| 2 | Shell, all pages | Placeholders and `info=` text (a refusal among them) at 2.54:1 | Theme variables `--block-info-text-color`, `--input-placeholder-color` from the existing tokens | M5 |
| 3 | Training | The Hits@K legend tells dashed series apart by colour alone (every dashed key has one style), three of five series overlap, and two palette colours miss 3:1 | Legend and tooltip swatches drawn with each series' own dash pattern | M5; the palette colours Later |
| 4 | Diagnosis | An uncalibrated score is labelled "Confidence", and the scores repeat in the details header | "Score" on the page and in the report, with an ⓘ and a report line saying it is a ranking score, not a calibrated disease probability; API field, scoring and CSV columns unchanged | M5; no phenotype column; the repeat Later |
| 5 | Test | The Model select is a dot-joined provenance chain per option | The Training pair (Run → Checkpoint), through shared renderer helpers | Later |
| 6 | Test | Two report rows read identically (the time cut to the minute) while showing different registration states | Rows that would read the same get the seconds, then the run id's last part; selection and registration keep the full run id | M5 |
| 7 | Test | R10 overlap (leakage) is two undifferentiated paragraphs; its help is a 155-word run-on | A table is allowed within the fixed semantics (section 5, decision 3) | Later |
| 8 | Test | The Result shows the report's own identity (run id, cohort label and version, checkpoint digest, deterministic, device) but no readable workspace or training-run name | A short name, or a hint when the selection differs from what was measured; a historical report's identity is never replaced by the current selection; no new reader or data format | Later (UX) |

Item 7, and any re-layout of the caveats, the D3 label or the §4.9 note, keeps the governed semantics (section 5,
decision 3).

## 4. Size, and M5's close-out scope

The audits estimate about 2–3 days for the shell's light adjustments plus 3–5 for its structural items, a few days
for Training and Datasets, and 4–6 for Test and Models: about 2½ to 3½ working weeks for all 83, none of it a new
layout or navigation. A two-part split (about a week within M5, then a design pass) was proposed; the owner and the
reviewer chose a **narrow close-out** instead (2026-10-25).

**Must-fix within M5** (done in the close-out batch `f9202b6`, each with a regression test that fails on the code before
it; the reviewer's incremental review of `e6ab1c3..f9202b6` found no P1 or P2):

- the score named for what it is: "Score" in the table, the details header and the human-readable report, with an ⓘ
  and a report line saying it ranks the candidates and is not a calibrated disease probability (appendix 1, in part);
- readable `info=` lines (refusals among them) and placeholders through the existing tokens, and the Settings
  restart-locked note in the red token (appendix 2 and 31, in part);
- the Hits@K legend and tooltip swatches drawn with each series' own dash pattern (appendix 33, in part);
- report rows that would read the same told apart by seconds, then the run id's last part (appendix 62).

**Still to accept within M5:** the owner's local UI recheck of the Training round (F1, F5 and its help) on the runs
already made (wide, narrow, back to wide, the Continue preview; no retraining), then the remaining manual acceptance
(section 6). A finding there blocks the close only when it is a concrete defect: a wrong target, mixed state, a
needed warning that cannot be seen, or a control that cannot be operated.

**Later improvements, not in M5** (the rest of the appendix, including the unfinished parts of the items above):

- site-wide: field labels as plain text, input borders at 3:1, a lock only on refused buttons, status marks with the
  traffic-light vocabulary, group labels set apart from the tabs, the environment line that leads with its
  consequence, the research tabs inside the header row when there is room;
- the help bubbles: all 78 texts rewritten to the budget; the structured renderer and placement that never covers its
  field, if and when the one shared bubble is extended;
- Diagnosis: the scores said once, the input echo (which HPO ids were used), evidence as key-value rows, terms and
  paths shown name first, the identity drawer as summary lists;
- Settings: one panel, fields before actions, running against saved allocator, read-only locks;
- Test: the Run → Checkpoint pair (shared renderer helpers), the reports list beside the Result, one record renderer
  for Test and Models, R10 as a table, tiles without the repeated "Mode C", a readable workspace and run name in the
  Result;
- Models and Datasets: details as summary lists, an empty state with its next step;
- Training: a one-row run bar, Continue's buttons in the drawer's footer, a higher chart floor (about 200 px), the
  chart palette's two colours below 3:1, checkpoint words "best" (decision 4);
- a UI audit in the repository (screenshots, geometry, repeats, axe offline).

## 5. Decisions (2026-10-25, owner and reviewer)

1. **Scope:** the narrow close-out of section 4, not the A/B split.
2. **Text floor:** 14 px stays; the PREPARE / EVALUATE group labels are set apart by position, spacing and weight.
3. **Governed texts:** their presentation may change within fixed semantics without a line-by-line sign-off. An R10
   table keeps each cell's state, k/n, units, cases and unverifiable reasons; an unverifiable cell never reads 0, —
   or "no overlap"; Mode C never implies an online total; the key warnings (ordering by test score uses up the
   cohort's independence) stay visible where the choice is made; denominators, thresholds and the ordering and
   registration policy do not change.
4. **Checkpoint words:** "Latest save" and "Top-k save" stay. If "best" is used later it means the highest Trainer
   val MRR recorded among this run's saved files, read from the list's metadata, never from file names; ties may share
   the word; a missing score, weights only, early stop and extend limits stay visible; the epoch is
   `completed_epochs`, older files keep "recorded epoch"; a word never changes a value, digest or permission.
5. **The skill:** revised to the reviewer's notes (checks chosen by risk, numbers as guidance, one shared bubble
   extended only when needed, third-party components judged on their merits, distinct missing-state words, the
   AGENTS rules first, no scope of its own), then referenced from `AGENTS.md`.

## 6. For the reviewer, on M5's close

M5's functional acceptance passed (99 of 99 probes); that evidence stays tied to `b14c4de`, and the UI increments
since carry their own tests. The close needs: the review of the close-out batch (passed, 2026-10-25); the owner's
local UI recheck at `f9202b6` (section 4), which the code review does not replace; and the remaining manual
acceptance, compared against the notes already taken and adding only what is missing (models the probe made may be
reused):

- launcher and service settings;
- Datasets selection;
- Training, the items this round changed: Start, live zoom, Keep training, Stop, the Continue preview, Export CSV,
  session isolation;
- Diagnosis: load, inference, results, CSV and report identity;
- Test: preview import → version → measure → register → read back;
- Models: display and the ordering warning;
- read-only refusals and a normal service stop.

M1's scope is unchanged and BACKLOG 50 (pin PyG) stays open. The later improvements need not be cleared to close.

## Appendix: every finding

The audits' own words, abridged; the full records (evidence with file and line, the principle each cites, the
research's principles and implementation references) are in
[`webui-ux-review-2026-10-25/review.json`](webui-ux-review-2026-10-25/review.json). Finding 32's fix there predates
`e54a5f1`, which made it.

| # | Page | Finding | Severity | Effort | Scope | Fix (abridged) |
|---|---|---|---|---|---|---|
| 1 | Diagnosis (results and details) | An uncalibrated score is labelled 'Confidence', and the scores appear twice for the selected candidate | high | M | structural | 1) Rename the column to 'Score' and give it an ⓘ (plain text is fine for this): 'Relative to the other candidates in this list. Not a probability that the patient has the disease.' Ref: the Isabel Pro likelihood-bar disclaimer. 2) Use… |
| 2 | Shell (global theme tokens), affects Diagnosis, Settings and Training | Placeholders and info= text, including a refusal, still use Gradio's subdued grey at 2.54:1 | high | S | light-adjustment | Add to the token block in style.py: .gradio-container { --block-info-text-color: var(--shep-text-2); --block-info-text-size: 14px; --input-placeholder-color: #6b7280; } (#434a58 is 8.9:1 and #6b7280 is about 4.8:1 on white). Do this… |
| 3 | Diagnosis (candidate details) | Evidence uses internal jargon, and phenotypes and paths show bare ids | medium | M | structural | 1) Replace the Mode headings with a shep-kv list: 'Evidence type: Direct links in the knowledge graph / Shortest link: 2 steps / Direct paths: 4 / Supporting genes: …' ('Similar genes' for Mode B). Put the narrative in native… |
| 4 | Diagnosis (empty state) | 'No model loaded' appears four times and 'no result yet' twice | medium | S | light-adjustment | Make the card the one place that says it: 'No model loaded' plus its one action. Shorten the run reason to 'Disabled until a model is loaded.' (the D8 reason stays visible). For no_model, the results empty state becomes title 'Load a… |
| 5 | Diagnosis (input and results) | No echo of which HPO ids were recognised or used | medium | M | structural | Under the textarea, add a plain_html line updated on .input (local, no API, show_progress='hidden'). It renders _parse_hpo_ids(text) as '3 ids recognised · not ids: "Seizure"' (escaped). After a run, show one 'As used' chip row at the… |
| 6 | Diagnosis (input column) | The input format rule is only in a placeholder and a bubble, and the example implies names are read | medium | S | light-adjustment | Use gr.Textbox(info='Only HP:####### ids are read; names and punctuation are ignored.'). This needs the --block-info-text-color fix first. Note that 6.30 escapes HTML in info, while **bold** and `code` work. Shorten the placeholder to… |
| 7 | Diagnosis, Change model drawer (and Models, Test, Training workspace selects) | Option labels and model rows are dot-joined chains that repeat facts | medium | M | light-adjustment | Keep the (label, value) tuples and change only the labels. workspace_choices returns (name, name), or 'name · synthetic' when that is needed to tell entries apart, at most 40 characters. The fingerprint appears once, in the line under… |
| 8 | Diagnosis, Settings (all text inputs and dropdowns) | Input boundaries are nearly invisible | medium | S | light-adjustment | .gradio-container { --input-border-color: #8d94a1; --input-border-color-focus: var(--shep-accent); --input-shadow: none; --input-shadow-focus: 0 0 0 2px var(--shep-accent-100); } (#8d94a1 is 3.05:1 on white, the grey the ⓘ already uses). |
| 9 | Help bubble (all pages) | Help bubbles are long run-on paragraphs with no lead line and no emphasis | medium | M | light-adjustment | (a) Renderer: replace help.py:268 with a function of about 30 lines that builds the body only with createElement and textContent. Line 1 becomes <p><strong>; '- ' lines become <ul><li>; 'Term: text' becomes <strong>Term</strong> plus a… |
| 10 | Help bubble (all pages) | The bubble covers the element it explains and is wider than the budget | medium | S | light-adjustment | In place(): find the explained region with var reg = anchor.closest('.shep-dx-card, .shep-panel-hd, .block, tr, .shep-kv') // anchor, and its panel with anchor.closest('.shep-panel'). Try the placements in a fixed order and take the… |
| 11 | Settings | A wall of text, actions above the settings they apply, and controls stretched to 1800px | medium | M | structural | Wrap the page in a shep-panel with max-width about 760px. Title at 20px. Cut the intro to one line: 'Hardware settings: speed, memory and precision. Model settings are in Research › Training.' Show the env-override sentence only when it… |
| 12 | Settings | Restart backend is one click for every session, and the page never shows whether a restart is pending | medium | M | structural | At import, capture the process's PYTORCH_ALLOC_CONF (which shep_launch.apply_allocator sets) and map it back to a preset through ALLOCATOR_PRESETS. Show persistent inline text under the allocator: 'Running: cuda_async · Saved:… |
| 13 | Settings | The torch.compile warning hides in a bare 'Details' accordion, and the badges use six hues with no shared meaning | medium | S | light-adjustment | gr.Checkbox(info='Experimental: often slower for HGT on GB10; compare MRR with an eager run before trusting results.'). Rename the accordions 'Details: what each allocator does' and 'Details: GB10 and HGT caveats'. Replace the badge… |
| 14 | Settings (read-only browsers, D8) | Read-only viewers can edit the fields and are told to click a disabled button | medium | S | light-adjustment | Add allocator and torch_compile to lock_outputs and return gr.update(interactive=access.allowed) for them. Give _on_change a gr.Request: for a refused request, return 'Operator controls only: <D8 short reason>' instead of the unsaved… |
| 15 | Shell (all Research pages) | Two navigation rows plus the footer take 131px of a 964px window | medium | M | structural | (1) Move the footer into a header chip (see the footer finding): saves 32px. (2) When Research is selected and the shell's layout() pass finds room, draw the research tab row inside row 1 to the right of the switch, in the underline… |
| 16 | Shell (all pages: field labels) | Field labels are indigo chips, and each page mixes 3 or 4 label styles | medium | S | light-adjustment | Use one label style site-wide via theme variables in style.py (the names are verified in the 6.30 bundle): .gradio-container { --block-title-background-fill: none; --block-title-padding: 0; --block-title-text-color: var(--shep-text);… |
| 17 | Shell (all pages: type scale) | Four sizes and four weights, with 14, 15 and 16px too close to form a hierarchy | medium | M | light-adjustment | Define tokens --shep-fs-title: 20px (at most 2 per page; on Diagnosis only 'Diagnosis results', on Settings the page title), --shep-fs-body: 15px, --shep-fs-meta: 14px (keeps the owner's floor). Use weights 400 and 600 only. Selected… |
| 18 | Shell (operator controls on Diagnosis, Training, Test, Models) | A padlock on enabled buttons carries no state | medium | S | light-adjustment | Add one helper, renderers.gated_label(text, allowed), and use it at every gated button. Allowed: plain label ('Load a model…'). Refused: '🔒 ' + label, disabled, with the visible 'Operator controls only: <reason>' line beside it (D8).… |
| 19 | Shell footer (all pages) | The environment warning leads with a version list, buries the problem and never states the consequence | medium | S | structural | Lead with the consequence in one bold line: 'CUDA device not visible: GPU training and GPU tests are unavailable.' Put the versions behind a native <details><summary>Environment details</summary> or an ⓘ topic. Better, render… |
| 20 | Shell header (level-1 switch and Settings) | The Diagnosis/Research switch looks like a selected form radio, and Settings has a fourth navigation look | medium | S | light-adjustment | Keep the accent fill for primary actions and selected form values. Restyle level 1: selected segment = background var(--shep-grey-800), border var(--shep-grey-800), white text, weight 600 (two cues, distinct from the indigo radios and… |
| 21 | Shell header chips | The read-only state is quieter than the normal state, and the chips look like buttons | medium | S | light-adjustment | Use one status vocabulary with --shep-status-{error,serious,warn,ok,info,neutral} tokens, each state shown as icon or shape plus text. Normal states are plain text with no pill border: lock glyph + 'Operator: on' in text-2, and 'GPU:… |
| 22 | Shell, Research page tabs (04, 05, 08, 10, 13, 14 screenshots) | The PREPARE/EVALUATE group labels sit in the tab row and read as two more tabs | medium | S | light-adjustment | Keep the CSS generated content (content: 'Prepare' / ''), pointer-events:none and RESEARCH_TABLIST_LABEL, and insert no nodes into the tablist. Option A (preferred; the owner must exempt overlines from the 14px floor in the… |
| 23 | Diagnosis | Developer details in the clinician's view: inference milliseconds and a ticking clock | low | S | light-adjustment | Drop the ms from the header and keep it in the exports and report. Show 'checked …' only when the card is not current (the grey states already say 'last known HH:MM:SS'), or show a static '✓ current' with the time in its title attribute. |
| 24 | Diagnosis (input column) | Left edges do not line up, and Options is a box inside the panel | low | S | light-adjustment | Set #shep-dx-fields .block.padded { padding: 0 0 8px !important }, so every block starts at the column's 16px padding. Drop the Options border and background. Lay it out as a 2-column grid (labels / controls), 'Candidates (Top-K)' [10]… |
| 25 | Diagnosis (input column); panel headers on all pages | Sticky regions can hide focused controls on shorter windows | low | S | light-adjustment | Set #shep-dx-input-bd { scroll-padding-bottom: calc(var(--dx-sticky-h, 220px) + 8px) }, with --dx-sticky-h taken from .shep-dx-sticky.offsetHeight in the shell's layout pass. Add .shep-dx-scroll, .shep-tbl-wrap { scroll-padding-top:… |
| 26 | Diagnosis (results, identity drawer) | The device notice shows twice with the drawer open, banners are all caps, and 'synthetic' shows three times | low | S | light-adjustment | Remove the device line from both drawer renderers; keep it in the footer and in the exports. Write the banner headlines in sentence case with <b> ('Synthetic workspace: not a clinical result'). While a result from the same served… |
| 27 | Diagnosis, Change model drawer | Misaligned meta line, a dead end with one candidate, and cryptic footer and title | low | S | light-adjustment | Put the meta on its own left-aligned line under the select, with a plural helper ('1 checkpoint in 1 run'). Preselect the row when the list holds exactly one (or the 'In use now'/latest row), so facts and Load are ready. Footer:… |
| 28 | Diagnosis, identity drawer (Details ›) | Accordion sections inside a drawer: two levels of disclosure | low | M | structural | Render the drawer as always-open summary-list sections (dl.shep-sl, grid-template-columns: minmax(8em,12em) 1fr auto) headed Identity / Training / Provenance / Warnings, Warnings first when non-empty. Render 'This result' with separate… |
| 29 | Help bubble (all pages) | Tooltip semantics on a toggletip, instant hover-open, an 'Esc to close' footer on every bubble, and small ⓘ targets | low | S | light-adjustment | When pinned, set role='dialog' and aria-label=title (or announce through a role=status live region; ref: the Inclusive Components toggletip). Keep role=tooltip for the hover preview. Add a 300ms open delay on mouseover only (none on… |
| 30 | Research pages (shell consistency) | The first panel's title repeats the selected tab name | low | S | light-adjustment | Title panels by their content and key count: 'Workspaces (5)', 'Checkpoints (1 in 1 run)'. Move the right-hand meta ('server workspace list · 5 · read-only') into that title, with no dot chain. |
| 31 | Settings | Restart-locked warning fails contrast, and the collapsed accordion chevron points left | low | S | light-adjustment | Render the lock as a .shep-notice with --shep-red-700 (#a52222, ≥7:1) beside the Restart control. Rotate the Gradio label-wrap icon so a closed accordion shows ▸ and an open one ▾ (one rule next to the existing button.label-wrap > .icon… |
| 32 | Training | Device 'auto' help is wrong, and the default is refused on this host | high | S | light-adjustment | 1) Rewrite both texts: "auto: uses CUDA; Start is refused when no CUDA device is visible. Choose cpu for development." 2) Show the state as info= on the Device radio, using the pattern this file already has for AMP_HGT_REFUSAL… |
| 33 | Training | Hits@K chart hides three of its five series; its legend separates them by colour only | high | S | light-adjustment | 1) Draw each legend swatch as inline SVG with the same pattern as its line, e.g. <svg width=22 height=8><line x2=22 y1=4 y2=4 stroke=COLOR stroke-width=2 stroke-dasharray='7 4'/></svg>, built from DASHES[slot]. 2) Replace #1baf7a with… |
| 34 | Datasets | Intro and footnote prose expose internals; 'read-only' collides with the D8 role | medium | S | light-adjustment | Header right: '5 workspaces · built on the server'. Drop the intro definition (keep it in the existing 'dataset-fingerprint' ⓘ). Move FOOTNOTE into an ⓘ on the 'In use' column header, without file or field names. Shorten USE_NOTE to… |
| 35 | Datasets | Details panel puts verification-level hashes first and packs several facts per row | medium | M | light-adjustment | In dataset_details_html: 1) Fingerprint row: 12-hex plus 'Copy full', with the 64-hex in the title. 2) Rows: Samples train 20 / Samples val 5 / Diseases 3 train, 1 val / Allocation seed 42 / Feature seed 20260909 / Sources / Built. Drop… |
| 36 | Datasets | Tall table rows, and an input-like 'In use' box | medium | S | light-adjustment | Header 'Built (UTC)' with a nowrap date only. Train/val on one line, '20 / 5', with tabular-nums. Render marks as .shep-tag pills, or plain text with an icon (e.g. '◉ Training run bar'); no white box. Merge Origin and Record into one… |
| 37 | Research navigation (Datasets, Training) | PREPARE and EVALUATE group labels still read as tabs | medium | S | light-adjustment | Use the overline option. On #shep-page-datasets-button::before and #shep-page-test-button::before set top:auto; bottom:calc(100% - 4px); left:12px; right:auto; font-size:12px (or 14px with weight 600 if the 14 px floor holds);… |
| 38 | Training | Charts are at the 140 px floor because the run view gives three rows to small text | medium | M | structural | Reclaim about 120-150 px, then raise the floor. 1) Merge run_snapshot_html into the run-view header line as muted text after the name ('… 17:39 · 195/1000 epochs · can be continued'); about −36 px. 2) Move Run name into the top of the… |
| 39 | Training | 'No training is running' is said three times, once in a button-shaped pill | medium | S | light-adjustment | 1) While idle, hide Stop rather than disabling it: add a 'shep-rb-idle' class on #shep-train-runbar, which already receives classes via view_outputs, plus CSS .shep-rb-idle #shep-train-stop{display:none}. Return reason "" for the idle… |
| 40 | Training | A lock icon sits on every gated button even when operator controls are on | medium | S | light-adjustment | Draw the lock only when D8 refuses. render() already knows whether the request is from the operator: add 'shep-rb-readonly' to the run bar's classes when it is not, and change the selector to '.shep-rb-readonly .shep-gated::before'. Do… |
| 41 | Training | Run details popover repeats facts, lacks the model fingerprint and nests a help bubble | medium | M | light-adjustment | Rows in ML-RD-1 order, through one shared kv renderer (P10 dl.shep-sl, grid minmax(8em,12em) 1fr auto; GOV.UK summary list ref): Run folder (relative path plus copy); Checkpoint 'checkpoints/last.pt'; Model fingerprint 12-hex plus copy… |
| 42 | Training | Help texts mix two styles and hide constraints the operator must know | medium | M | light-adjustment | 1) Rewrite each text to P07 grammar: a lead of 90 characters or fewer saying what the setting is, then 1-3 '- ' bullets. Examples: 'Passes over the training samples.\n- Smoke test: 2–5\n- Quick: 10–50\n- Full: 100–500'. 'Graph layer… |
| 43 | Training | Help bubble covers the field it explains | medium | S | light-adjustment | In the bubble placement code in help.py, measure the trigger's field (btn.closest('.block').getBoundingClientRect()). If the bubble would overlap that field, place it beside the field: to the right when there is room (the params column… |
| 44 | Training | Choosing HGT changes a setting inside a collapsed section | medium | S | light-adjustment | When HGT is chosen, also return gr.update(info='HGT: Mixed precision set to Off (Expert); keep batch size ≤16 on a 16 GB GPU') for conv_type, and an info line on batch_size when it is above 16. Clear both ("") for gat and sage, as the… |
| 45 | Training | Parameters column is single-file and long at wide widths, but dense when narrow | medium | S | light-adjustment | Apply the existing grid at all widths in #shep-train-params-bd .gr-group .styler > .form (grid-template-columns: repeat(auto-fill, minmax(220px, 1fr))). That gives 2 columns in the 560 px column, so Basic fits in about 4 rows and… |
| 46 | Training | In-page radios and field labels borrow the level-1 and link styles | medium | S | light-adjustment | Add elem_classes=['shep-seg'] to conv_type, device, amp_mode, continue_source and continue_device (white segment on grey track). Inside #shep-train set the label tokens to plain text: --block-title-background-fill: transparent;… |
| 47 | Training (Continue drawer) | Checkpoint options are still dot-joined jargon, and the details repeat the Run | medium | S | light-adjustment | 1) Labels: 'latest · epoch 195', 'best val MRR 0.287 · epoch 21', 'epoch 14'. Compute 'best' per run as the readable file with the highest recorded validation metric, ties going to the later epoch, and give each alias to exactly one… |
| 48 | Training (Continue drawer) | The drawer's instructions contradict its state | medium | S | light-adjustment | Pick one model and say it. Option A (smallest): set the prompt to 'Optional: Check checkpoint reads the file and shows the plan and the settings it keeps.' Option B: make the check part of Continue. Continue Training runs _checked and… |
| 49 | Training (Continue drawer) | Continue mode splits one task across two regions, with duplicate controls | medium | M | structural | 1) Put Continue Training and Cancel in the drawer footer. #shep-train-drawer-why (:3014, already a .shep-panel-ft) can hold the reason and both buttons. 2) While the drawer is open, hide Start/Continue…/Stop and the 'RUN a new one…'… |
| 50 | Training (all pages) | Environment footer leads with versions and never states the consequence | medium | S | light-adjustment | Lead with the consequence: 'No CUDA device visible: GPU training is unavailable. Check the driver.', followed by a 'Versions ›' disclosure (native <details> in the gr.Markdown/HTML) holding the package list. Use an inline SVG warning… |
| 51 | Training / shared | Field boundaries fall below 3:1 contrast, and the 37 'i' targets are 20 px | medium | S | light-adjustment | Add a --shep-field-border token at 3:1 or more (e.g. #8d94a1, 3.05:1, already the ⓘ ring colour) and apply it to Gradio inputs inside .shep-panel through --input-border-color, and to the three select wraps. Keep the 20 px glyph but give… |
| 52 | Datasets | Selectable rows carry invalid ARIA, and links lack a focus style | low | S | light-adjustment | Either give the table role='grid' with role='row' and gridcell semantics, or drop aria-selected and expose the selection as a visually hidden 'selected' text plus aria-current='true' on the row. Add a shared… |
| 53 | Datasets / Training | The same status is drawn three different ways | low | S | light-adjustment | Have the table, the details header and the run bar all call r.chip and r.record_chip. Choose one empty form for Sources ('none, synthetic') and use it in both places. |
| 54 | Training | Chart titles use internal jargon, and one-series legends repeat their title | low | S | light-adjustment | Titles: 'Loss', 'Validation MRR (top 20)', 'Validation Hits@K', 'Learning rate (log scale)'. In keysHtml, skip the legend when a chart has a single series. Put 'Epoch' once in the shared toolbar ('x: epoch, shared') instead of under two… |
| 55 | Training | A sticky chart toolbar remains inside the chart scroll row | low | S | structural | Take the toolbar out of the scroller. Either render it in #shep-train-rv-top, merged with the header row from the chart-budget finding, or give #shep-train-charts scroll-padding-top equal to the bar's height (WCAG technique C43). Add… |
| 56 | Training | Type scale is too flat, and one uppercase style serves four roles | low | S | light-adjustment | Adopt the tokens 20px for region titles (run name, page title), 15px for body and values, 14px for keys and meta (the owner's floor), and weights 400/600 only. Keep uppercase for the nav group overline alone. Render DATASET and RUN NAME… |
| 57 | Training | Run-view header tags are ambiguous and look clickable | low | S | light-adjustment | Write 'latest run' or drop the word; the Run select already orders latest first. Show the continued-from relation as plain text with an icon ('↳ parent of the new run'), in grey, not accent. Keep colour tags for states that need… |
| 58 | Training | Labels mix casing, abbreviations and an emoji | low | S | light-adjustment | Use sentence case and full words: 'Learning rate', 'Batch size', 'Graph layer type', 'Hidden size', 'Layers', 'Neighbours per layer'. Use choices as (label, value) tuples: [('GAT','gat'),('HGT','hgt'),('SAGE','sage')]. Drop the emoji.… |
| 59 | Training | Gradio progress timers flash on several handlers | low | S | light-adjustment | Add show_progress='hidden' to those seven listeners. Check whether this also removes the blank band in the Continue-mode run bar. |
| 60 | Training (narrow 960) | At narrow width, Parameters push the run view below the fold | low | S | light-adjustment | In the max-width:1399px media query, set order on #shep-train-runview before #shep-train-params (flex order), or start the Parameters accordion 'Basic' collapsed in the narrow layout. The run bar stays first. |
| 61 | Test | Model select still uses dot-joined provenance chains, unlike Training's Run → Checkpoint pair | high | M | structural | Move run_words, checkpoint_words, _choices, _run_choices, _checkpoint_choices and parent_details_html out of training_console.py into renderers.py and reuse them on Test. Use two gr.Dropdowns, Run and Checkpoint. Run.change returns… |
| 62 | Test | Report rows read identically yet show different registration states | high | M | light-adjustment | Label each row with clock_utc(seconds=True), e.g. '11:10:47 UTC', or apply Training's _choices de-dup so the run suffix (1885e8bd) appears only when two labels collide. Lead with the model's run label taken from the report record's own… |
| 63 | Test | R10 train/test overlap (leakage) is buried in two undifferentiated paragraphs | high | M | light-adjustment | Render R10 as a 2x2 mini table inside the Result kv grid. Rows are Training and Validation; columns are 'Disease overlap' and 'Same disease + phenotype set'. Example cells: '⚠ 3 of 5 diseases · 7 cases' (amber with an icon when k>0),… |
| 64 | Test | Result does not say which workspace or run produced the report; the only clue is a clipped path | high | M | light-adjustment | Open the Result body with a 'Measured' summary list. Rows: Workspace (demo), Model ('GAT · run 10-19 09:00 · epoch 5' plus 12-hex and copy, from the record's checkpoint fields), Cohort version (label only), Regime (CPU · deterministic… |
| 65 | Models | The amber independence warning shows even when the list is 'Newest first' | medium | S | light-adjustment | Keep the §4.9 rule but tier it. With Newest first, show one muted line under the Order control ('Ordering by test score uses up the cohort as an independent test.' plus the existing ⓘ). With 'By test score', show the amber notice, bold… |
| 66 | Models | The dataset identity appears 4 times; the fingerprints column repeats the workspace's own dataset | medium | S | light-adjustment | Change the table column to 'Fingerprint' and show the model 12-hex only. Show the dataset in a row only as an exception chip, 'other dataset' (amber), when trained_on_this_dataset is False. In the details, use a 'Dataset' row: 'this… |
| 67 | Models | Details identity is a mix of keys, paths and repeated labels instead of a summary list | medium | M | light-adjustment | Use one key-value list in fixed order. Identity: Run, Epoch, File (code, full path on hover), Model fingerprint (12-hex, copy, full value in title). Training: Dataset (this workspace ✓ · synthetic). Validation: 'Val MRR (Trainer,… |
| 68 | Models | The empty recorded-results state is a dead end that names a file; the segments show '· 0' twice | medium | M | light-adjustment | Title: 'No recorded tests'. One sentence: 'Test results are kept beside the checkpoint once a run is registered.' One action: 'Test this model ›'. It returns gr.Tabs(selected='test') and sets the Test page's Run/Checkpoint to this… |
| 69 | Test | R10 help bubble is a 155-word single paragraph that repeats itself | medium | M | light-adjustment | Write one hand-made text of about 55 words in place of the generated join: lead line 'How much of the test set the model already saw.' then 'Disease overlap: test diseases that are also a training (or validation) case's disease.' then… |
| 70 | Test | The same identity facts repeat across the three panels | medium | M | light-adjustment | (1) The workspace option is the name only: names are unique directories, and the identity line under the select already shows the fingerprint and synthetic chip. Change it in the shared renderers.workspace_choices so Training, Datasets,… |
| 71 | Test | 'Mode C' prefixes every metric; the first tile wraps and misaligns the values | medium | S | light-adjustment | Make the MODE_C_RESULT_LABEL heading the legend of the tile group (role=group aria-labelledby, or fieldset/legend), so any crop of the group still carries the D3 label. Tile labels become 'MRR', 'Hits@1', 'Hits@5', 'Hits@10', with… |
| 72 | Test | Cohort version details and Relations are run-on dot strings that lead with codes | medium | M | light-adjustment | Render the version as a key-value list (renderers.kv_cells or a new dl.shep-sl, grid minmax(8em,12em) 1fr auto). Rows: Role, Cases ('10 of 13 eligible · 3 excluded'), Version fingerprint (12-hex, copy, binding chip), Imported, Format,… |
| 73 | Test | The Reports list sits in the Cohort column, away from the Result it drives, and the Result clips without a cue | medium | M | structural | Move reports_list into the top of the Result column as a compact single-select table (When · Model · State, radio-row semantics, 'Show all N'); the Result body below it scrolls. Add a scroll cue when scrollHeight > clientHeight: a… |
| 74 | Test, Models | Help texts exceed the reading budget and lean on internal codes | medium | M | light-adjustment | Rewrite every entry as a 1-sentence answer plus at most 3 bullets, about 60 words or fewer, with no sentence over 25 words. Move setting names into a trailing 'Server setting: SHEPHERD_…' line and drop decision codes, or put them in a… |
| 75 | Test, Models | A lock icon on enabled operator buttons and an operator-only header read as 'locked' for the operator | medium | S | light-adjustment | When access.allowed, update the labels without the lock (gr.update(value='Register this report')). Keep the lock only on refused controls, next to the visible 'Operator controls only: <reason>' text, which stays for read-only viewers… |
| 76 | Test, Models (site-wide) | Field labels are tinted chips that look like tags; sibling keys are plain text | medium | S | light-adjustment | Override the block label for .shep-page: .gradio-container .shep-page [data-testid=block-info] {background:none; padding:0; color:var(--shep-text-2); font-size:14px; font-weight:600}, or set the theme vars --block-label-background-fill… |
| 77 | Models | Count line has a plural bug and repeats the order; ordering mode uses dot-joined headers and options | low | S | light-adjustment | Count line: '1 checkpoint · 1 run', with the order word dropped. Ordered column: header 'Test MRR', with the version and regime in the group title that already names them. Scoring options: 'Semantics A (3 checkpoints)', with the digest… |
| 78 | Models | Load note is jargon and shows before anything is selected | low | S | light-adjustment | Suggested wording: 'Loads this model for Diagnosis after a From → To check. Earlier results keep their own model identity.' Render the note only when a loadable row is selected; note_html returns '' otherwise. |
| 79 | Test | Run options: the detached ⓘ, a wrapping checkbox label, a shared ⓘ, and two similar 'Register' controls | low | S | light-adjustment | Stack the two checkboxes one per line, or set .shep-test-checks > .block {flex:0 0 auto} so the ⓘ stays inline after the text. Rename the checkbox 'Register when it completes', which matches the running row's 'registers when it… |
| 80 | Test | 'Other metrics' sort lexically: Hits@100 before Hits@20 | low | S | light-adjustment | Move _metric_order to renderers.py and use sorted(..., key=lambda kv: _metric_order(kv[0])) in _record_html. Add a regression test that Hits@20 comes before Hits@100. |
| 81 | Test | Low-information idle and 'kept' states take space and colour | low | S | light-adjustment | While idle, collapse the job card to one muted line, 'No test run since the server started', and hide Cancel until something is running. Show kept or registration chips only for the states that matter ('Not registered', 'Registered #n',… |
| 82 | Test, Models | The same report record is rendered by two different code paths | low | M | light-adjustment | Add one renderers.record_summary(record) that returns ordered (label, html) rows: Measured (workspace, model, cohort), Metrics, Coverage, Checks, R10 table, Environment, Dataset, Registration. Test renders it as a key-value list; Models… |
| 83 | Test, Models | Accessibility details: role=list without list items, a likely clipped focus ring, 20px ⓘ targets | low | S | light-adjustment | Use role=listbox and role=option with aria-selected, plus arrow-key handling in _REPORTS_JS, or drop role=list. Add .shep-test-rep:focus-visible {outline:2px solid var(--shep-accent); outline-offset:-2px}, copied from style.py:321. Give… |
