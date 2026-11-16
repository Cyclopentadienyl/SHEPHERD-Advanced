# WebUI ledger: every recorded WebUI problem and its state

**The one state source for the WebUI's known problems** (rev 2 completion plan, 2026-11-16, T00). BACKLOG items and the round README point here by ID; a plan names the IDs it takes. Base: `ba6e85e`. Built from the audit of 2026-11-16, which checked every recorded WebUI problem against `ba6e85e` in the running app (a CPU demo workspace in mode M1) or in the code, then had a second reviewer look for missing items and overturn verdicts; the audit's evidence files were not kept in the repository, its findings are restated here.

Sources read: `docs/working/WEBUI_UX_REVIEW_2026-10-25.md` (UX) and its `review.json` (RJ), `docs/working/BACKLOG.md` (BL), the round README `docs/working/ops-wiring-and-ui-ux/README.md` (OPS), the skill `.claude/skills/webui-design/SKILL.md` (SK), the M5 WebUI acceptance (M5W), `docs/working/PLAN_TEST_RESULTS.md` (PTR), `docs/working/scorer-retraining/README.md` (SR), and the owner's fix list rev 1 (2026-11-16).

## States

| State | Meaning |
|---|---|
| to do | open, not started in this round |
| in progress | being changed in this round |
| partly done | some of it is done; `Done when` says what is left |
| code-verified | done, with the commit and the tests or browser checks that show it |
| code-verified at ba6e85e (audit) | already done before this round; the audit saw it |
| awaiting hardware | needs a CUDA device or the GB10 to check |
| awaiting owner acceptance | done in code; the owner's own check of the screen is pending |
| not applicable | its subject is retired or gone; the reason names the source |
| outside this round's scope (owner to decide) | a new product surface, not a defect of an existing screen; kept here, not dropped |

A row is closed only with its scope, its tests and the screen it was seen on; a fix on one page is never read as a site-wide fix.

## The counts, reconciled

- **218 rows**: 183 recorded problems (the audit's inventory of 163 and 20 the second reviewer found missing from it) and 35 new findings of the 2026-11-16 audit.
- **The 2026-10-25 review's 83 findings are 77 rows** (`R-` IDs): finding 1 is split into 1a (the label), 1b (the repeat), 1c (the evidence marks) and 1d (the phenotype count); nine findings are merged into another (37 into 22; 40 and 75 into 18; 43 into 10; 46 into 20; 50 into 19; 51 into 8; 56 into 17; 76 into 16). At `ba6e85e`: 16 fixed, 34 partly, 25 not done (14 of them kept open by no record at all), 2 not applicable. The not-applicable two (R-44 and R-65, retired features) are why 16 + 34 + 25 reported to the owner made 75, not 77.
- **The owner's own reports and rechecks are 33 rows**: 10 fixed, 10 partly, 7 not done, 5 new in the fix list rev 1, 1 awaiting hardware. Several are also review rows (R-22+R-37, R-38, R-42, R-47, R-1b), so these counts overlap the review's and are not added to them.
- **The 35 new findings** (`NEW-` IDs): 18 restate a recorded row (`Duplicate of` names it), 17 are new problems. They overlap the counts above and are not added to them.
- **Overstated records**: eight rows a record called done that were only partly done (three verdicts overturned by the second reviewer: R-18+R-40+R-75+B68+B72, R-66+B51f, R-72+B72; and R-25+K6, RC-B, B72-N7, R-16+R-76+B72, R-53).
- **By state at the base**: 97 to do, 49 partly done, 59 already done, 1 awaiting hardware, 5 not applicable, 7 outside this round's scope.

## Rows by task

### UI-01 Navigation hierarchy

4 rows: 4 to do.

#### NEW-shell-nav-theme-1 · Chosen in-page radio fill is exactly the level-1 switch's fill

- **State**: to do · found by the 2026-11-16 audit · page: shell
- **Duplicate of**: R-20+R-46+B72 (closed with it)
- **At ba6e85e**: Close fix 3d85531 made the chosen in-page radio's fill exactly the level-1 switch's fill: --checkbox-label-background-fill-selected is var(--shep-accent) (src/webui/style.py:240), the same as the selected level-1 tab (style.py:113-114, both rgb(85,88,227) in theme-1850x960.json and shell1-1850x960.json). The likeness BACKLOG 72 'left open' became an exact match, and the close record (OPS:364) mentions only the 5.4:1 contrast.
- **Done when**: The selected level-1 tab and a selected in-page radio have different computed styles (fill or shape), checked in a browser at 1850x960.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### R-15 · Two navigation rows plus footer take 131px of a 964px window

- **State**: to do · found by the 2026-10-25 review · severity medium · page: shell
- **Problem**: Two navigation rows plus the footer took 131 px of 964, out of the chart grid (evidence cites the owner's 'charts squeezed' complaint).
- **At ba6e85e**: At 1850x960 the header row is 0-55 px, the research tab row 55-100 px and the runtime footer 928-960 px: 132 of 960 px (the review measured 131 of 964). The research tabs are still a second row. The footer is still its own 32 px row, now leading with the consequence. Training's four plots are 230/217 px. At 960 px wide the header wraps to 99 px, with a 45 px tab row under it.
- **Done when**: At 1850x960 the research tabs sit in the header row (and/or the footer is folded into it), so navigation and footer together take clearly less than 132 px and Training's chart grid gains that height.
- **Sources**: UX:217 (appendix 15); RJ audits[0].findings[16]; UX:119-120 (section 4: research tabs inside the header row when there is room); RJ research[1].principles[1] (G2)
- **Closed at**: —

#### R-20+R-46+B72 · Level-1 switch and in-page radios share one filled accent style

- **State**: to do · found by the 2026-10-25 review · severity medium · page: shell
- **Problem**: The Diagnosis/Research switch looked like a selected form radio, in-page radios copied its filled style, and Settings had a fourth navigation look.
- **At ba6e85e**: The level-1 switch's selected 'Research (pre-clinical)' is a #5558e3 fill with white 700 text. Training's chosen Device 'auto' and Fine-tune's chosen Graph backbone 'Trained' have exactly the same #5558e3 fill with white text, and neither uses the .shep-seg third style. Close fix 3d85531 caused the exact match: it set --checkbox-label-background-fill-selected to var(--shep-accent) to fix contrast. When Settings is selected it has a fourth look (an accent-50 box with an inset underline), and the level-1 switch then shows neither stage as selected. The level-1 tablist is still named 'Stage'. R-46's label half is fixed (b33cb14).
- **Done when**: The selected level-1 switch option looks different from a selected in-page radio (Training Device, Fine-tune Graph backbone use .shep-seg or another style), and Settings' selected look fits the navigation levels; a computed-style test shows the fills differ.
- **Sources**: UX:222 (appendix 20); RJ audits[0].findings[7]; UX:248 (appendix 46); RJ audits[1].findings[13]; UX:54 (section 2 layer 4); BL:1030 (item 72 Shell); SK:48-50
- **Closed at**: —

#### R-22+R-37 · PREPARE/EVALUATE group labels sit on the tab row and read as tabs

- **State**: to do · found by the owner · severity medium · page: shell
- **Problem**: The PREPARE and EVALUATE group labels are ::before content on the first tab of each group, on the tabs' 44 px row at about the same size and weight, so they read as two more (clickable) tabs.
- **At ba6e85e**: Unchanged. 'Prepare' and 'Evaluate' are ::before content of the first tab in each group. They are absolutely positioned left of that tab, inside the same 44 px row and centred on the tabs' line, at 14 px/600 uppercase with .07em spacing in #5b6270 (6.13:1). The tabs beside them are 15 px/500 #434a58, so the labels are heavier than the unselected tabs. The gap from PREPARE to Datasets is about 18 px, against 24 px between tabs, and the offsets are still hard-coded margins (84 px and 116 px). The accessibility half is met: the labels sit outside the tabs' hit area (elementsFromPoint returns the tablist; the cursor is auto), have pointer-events none, are not focusable, and the tablist's aria-label names the groups. The visual half the owner reported, that they read as tabs, is not. At 480 px the EVALUATE group wraps onto its own line, with its divider at the line's start. The CSS predates the shallow history's root 9a96a72 (2026-10-30), no commit in 30e8ae1..ba6e85e touched it, and no test pins its look (tests/unit/test_webui_shell.py:74 checks only the aria-label). The shell docstring (shell.py:17-23) still describes it as is.
- **Done when**: PREPARE and EVALUATE no longer sit on the tabs' line: they are placed apart (e.g. as overlines or with a clear gap and divider) at 14 px, distinguished by position, spacing and weight, without hard-coded pixel margins; a browser test checks their vertical offset from the tabs.
- **Sources**: UX:224 (appendix 22); RJ audits[0].findings[2]; UX:239 (appendix 37); RJ audits[1].findings[19]; UX:53-54 (section 2 layer 4); UX:119 (section 4 site-wide); UX:152 (section 5 decision 2); RJ audits[0].overall (point 4); RJ research[1].principles[0] (G1); RJ research[4].principles[7] (S8); RJ audits[2].findings[15] (R-76 principle); SK:51-52; SK:187-188
- **Closed at**: —

### UI-02 Orientation and scrolling

16 rows: 4 to do, 5 partly done, 1 outside this round's scope (owner to decide), 6 code-verified at ba6e85e (audit).

#### FL-UI-02 · Layout picks side-by-side vs stacked from fixed 1400x760 sizes, not the window's shape

- **State**: to do · found by the owner · severity unrated (fix list) · page: shell
- **Problem**: The panels sit side by side only at 1400 px wide or more and 760 px tall or more; otherwise they stack (style.py:443-478). The fix list's counterexamples follow from this: 1280 x 900 and 1920 x 700 stack, while 1440 x 1600 (portrait) is side by side. A short-height window is forced into the stacked layout although it has the width.
- **At ba6e85e**: Recorded only in the fix list rev 1 (2026-11-16), confirmed there from the code. Earlier records state the rule (by the usable size), but none records the fixed threshold as a problem. Not fixed.
- **Done when**: At 1920x700 and 1280x900-wide-landscape windows the panels sit side by side and at 1440x1600 portrait they stack; short height is handled by region scrolling with no control covered, checked by a browser test over these sizes.
- **Sources**: the owner's fix list rev 1 (2026-11-16); src/webui/style.py:443-478; docs/working/BACKLOG.md:985 (item 27: 'by the browser's usable size rather than the monitor's'); docs/working/PLAN_TEST_RESULTS.md:2390-2394
- **Closed at**: —

#### NEW-diagnosis-3 · Records claim sticky-cover and drawer-repeat fixes that are not in the code

- **State**: to do · found by the 2026-11-16 audit · page: diagnosis
- **Duplicate of**: R-25+K6 (closed with it)
- **At ba6e85e**: Two records say something is fixed that is not: - scorer-retraining/README.md:963-966 says no focused Diagnosis control is covered at 1536 x 760 with each model loaded. At ba6e85e, with the disease model served, the HPO box's caret line is under the sticky card from 8 lines on (14-sticky-textarea-1536x760.png). - BACKLOG 72's B4 note says the load drawer 'says the chosen workspace once'. Every row still repeats 'trained on demo · dataset fingerprint 50bef1bf5f9f · synthetic' under the workspace line.
- **Done when**: At 1536x760 with the disease model served, the HPO box caret stays visible above the sticky card while typing 12 lines, and the load drawer rows no longer repeat the workspace facts; or the README and BACKLOG 72 notes are corrected to say open.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### NEW-shell-nav-theme-2 · At 480 px the wrapped EVALUATE group starts its line with a divider

- **State**: to do · found by the 2026-11-16 audit · page: shell
- **At ba6e85e**: At 480 px wide the EVALUATE group wraps onto its own line, and its divider (the ::before border-left) is drawn at that line's start, so the line reads '| EVALUATE Test Models' ((audit evidence, not kept in the repository) src/webui/style.py:164-167).
- **Done when**: At 480 px wide, the EVALUATE line has no divider drawn at its line start (no '| EVALUATE'), checked by screenshot.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### R-60 · At narrow width Parameters push the run view below the fold

- **State**: to do · found by the 2026-10-25 review · severity low · page: training
- **Problem**: At 960 px the Parameters pushed the run view below the fold.
- **At ba6e85e**: At 960x960 the header wraps to two rows, the run bar is 205 px, and the full Parameters panel comes next. The run view's header starts at y=883 and its first plot at y=1059, so no chart is visible without scrolling the page (page scroll 1386 px). The narrow media rules do not reorder the run view or collapse the parameters (training_console.py:5088-5101).
- **Done when**: At 960x960 the Training run view's first plot is visible without page scrolling (run view placed before Parameters, or Parameters start folded).
- **Sources**: UX:262 (appendix 60); RJ audits[1].findings[26]
- **Closed at**: —

#### B72-C3 · Tall wide Compare: sideways scrollbar unreachable mid-table

- **State**: partly done · found by an internal review · severity minor · page: results
- **Problem**: Sideways scroll with the bar only at the foot.
- **At ba6e85e**: The B3 mechanism works. I forced an overflow on the client (table min-width 1600px). The cue 'Wider than the panel: scroll sideways; the Field column stays.' then appears above the table (cue top 165, table top 190). Its bar's scroll range equals the table's (663 = 663), and scrolling the top bar to its end moved the table to 663. With the 4 real results, Compare fits its panel at 1850x960 (937/937) and at 960x1080 (892/892), so a natural five-column overflow could not be produced (only 4 results). The Compare table is 12,342 px tall with 'Differences only' and 21,825 px with all rows. Between its top and its foot, no horizontal bar is in view.
- **Done when**: With a Compare wider and taller than its panel, a sideways scrollbar is reachable from anywhere in the table (sticky or bounded region) without scrolling to its top or foot.
- **Sources**: BL:1030
- **Closed at**: —

#### PTR-LAYOUT · Layout acceptance matrix (portrait, zoom, high DPI, long names) never run

- **State**: partly done · found by an internal review · page: shell
- **Problem**: PLAN_TEST_RESULTS §6's acceptance requires the layout to be checked landscape and portrait, at high DPI and zoom, with long disease names and many results and warnings. Its M5-3 check reads 'layout: not yet recorded'. The matrix (1920x1080, 1707x870, 1536x760, 1440x900, 1280x720, 1080x1920, 200 % zoom; M1 and read-only) 'was not run and is not an M5 gate'. The ops close covered 1850x960, 1536x760, 1920x1080, 960x1080, 1080x1920 and 480/760 wide (OPS:363). It did not cover 200 % zoom, high DPI, 2560x1600 or 3840x2160, 1440x900, 1280x720, long names or many results.
- **At ba6e85e**: 'Not yet recorded' (PTR). Not an M5 gate; partly covered by the ops close. Nothing schedules the rest.
- **Done when**: Screenshots at 1920x1080, 2560x1600, 3840x2160, 1080x1920 and 200% zoom, M1 and read-only, with long disease names and many results, are recorded with no clipped or covered controls and no page-level horizontal scroll.
- **Sources**: docs/working/PLAN_TEST_RESULTS.md:2450-2452; docs/working/PLAN_TEST_RESULTS.md:2484-2490; docs/working/BACKLOG.md:985 (item 27 'Screens' and the proposed screenshot sizes); docs/working/ops-wiring-and-ui-ux/README.md:363
- **Closed at**: —

#### R-25+K6 · HPO box's lower lines and caret go under the sticky card at 1536x760

- **State**: partly done · found by the 2026-10-25 review · severity low · page: diagnosis
- **Problem**: Sticky regions (Diagnosis card and Run, panel headers) could hide focused controls; the K6 smoke saw the sticky card over a focused control at 1536 x 760.
- **At ba6e85e**: The part that was fixed: the card and Run sit in a sticky region, and the input column's scroll padding equals that region's height (402 px at 1536 x 760 with the disease model served). Tabbing through Patient ID, Top-K, the checkbox, the info buttons and the card buttons, no focused control was covered at 1536 x 760 or 1850 x 960. The part that remains, at 1536 x 760 with the model served: - the scroll area is 591 px and the sticky region 402 px, leaving 189 px for the fields; - once the HPO box holds 8 or more lines (it grows to 12), the line with the caret is under the card: elementFromPoint at the caret line returns the sticky card at 8 and 12 lines; - at 12 lines the box's lower 127 px are hidden; - 3-6 lines are fine. At 1850 x 960 a 12-line entry is not covered. The K6 record says no focused Diagnosis control is covered at 1536 x 760 with each model loaded. B4 has since made the served card taller: state, lead, name, fingerprint, trained-on, chip, recorded results and two rows of buttons.
- **Done when**: At 1536x760 with a model served, typing multiple lines into the HPO box never puts its caret or visible lines under the sticky card/Run region, checked by a browser test.
- **Sources**: UX:227 (appendix 25); RJ audits[0].findings[29]; SR:963-966 (K6 smoke P2)
- **Closed at**: —

#### R-38+owner · Training plots squeezed near the minimum height; run bar still two rows

- **State**: partly done · found by the owner · severity medium · page: training
- **Problem**: The run view gave three rows to small text and the footer ~150 px, leaving the four plots at the 140 px floor.
- **At ba6e85e**: PLOT_MIN_HEIGHT is 170 (training_console.py:4797-4806, 443cfc1). Measured at 1850x960: plot areas are 183/183 px in the top row and 170/170 px in the bottom row, up from about 130-155 px at the audit. The state line now sits in the header and the run bar's blank area is gone (c215031). The run bar is still two rows, 119 px tall (y 116-235: the Dataset row is 41 px, the Run name / Start row 40 px). The run-view header wraps to 62 px, the toolbar is its own 34 px row, resources take 38 px, and the page footer about 30 px. The code comment says the chart body holds 519 px and a 200 px floor needs 566. The skill still says 'today 140 px plot area' (SKILL.md:41).
- **Done when**: At 1850x960 the Training run bar is one row and each of the four plot areas measures about 200 px tall without page scroll.
- **Sources**: UX:240 (appendix 38); RJ audits[1].findings[2]; RJ audits[1].overall; RJ research[1].principles[1] (G2); RJ research[2].principles[6] (ML-CH-1); RJ research[4].principles[4] (S5); UX:130 (section 4 Training); BL:1010 (52); SK:41-42
- **Closed at**: —

#### R-73 · Test Result scroll region has no scroll cue

- **State**: partly done · found by the 2026-10-25 review · severity medium · page: test
- **Problem**: Reports list away from the Result; the Result clipped without a cue.
- **At ba6e85e**: The reports are now a compact single-select table (Report · Model · Measurement · State) at the top of the Result panel, with the result under it in the same scroll region. Narrow, the order Data, Run, Result keeps the table beside its result. There is still no scroll cue: at 1850x960 the Result body has scrollHeight 1536 against clientHeight 681, overflow auto, mask none. No gradient or mask exists anywhere in the Test page CSS, and the last line is cut mid-text above 'Register this report'. Whether a scrollbar shows depends on the OS: headless Chromium uses overlay scrollbars (0 px).
- **Done when**: When the Result body's scrollHeight exceeds its clientHeight, a visible scroll cue (mask or line, as Compare's SCROLL_CUE) shows and disappears at the end.
- **Sources**: UX:275 (appendix 73); RJ audits[2].findings[9]; UX:126-127; BL:1010 (52)
- **Closed at**: —

#### B52-G · Draggable splits between the main panels (owner's idea)

- **State**: outside this round's scope (owner to decide) · found by the owner · severity idea · page: shell
- **Problem**: The owner proposed bounded, draggable panel splits.
- **At ba6e85e**: There are no draggable splits: the panels are fixed flex columns, and no splitter or drag code exists in src/webui.
- **Why this state**: Draggable splits are a new product surface listed as out of scope for this round (tasks_def.md BACKLOG list, plan §1).
- **Sources**: BL:1010 (item 52 (G)); M5W:66-67
- **Closed at**: —

#### B52-F · Collapsed Training run bar grew into a blank area

- **State**: code-verified at ba6e85e (audit) · found by the owner · page: training
- **Problem**: With every section collapsed the run bar grew into a large blank area (screenshot 09).
- **At ba6e85e**: The run bar is built at scale=0 (training_console.py:4382-4387), so it no longer takes a share of the page's height. It measured 119 px for two rows with no blank space, both idle and with the Continue drawer open.
- **Sources**: BL:1010 (item 52 (F)); M5W:64-66; PTR:2488-2489
- **Closed at**: —

#### OPS-B5a · Header chips covered the Research tab at a narrow width

- **State**: code-verified at ba6e85e (audit) · found by an internal review · page: shell
- **Problem**: At 480 px the D8 chip covered the Research tab.
- **At ba6e85e**: At 480 and 760 px wide, the centre of every tab (level 1, Settings and the five research tabs) hits that tab: no header chip covers one. The chips sit on their own line under the stage switch.
- **Sources**: OPS:361; commit 25e6af9
- **Closed at**: —

#### OPS-C10 · Test's Start below the fold at 1850x960

- **State**: code-verified at ba6e85e (audit) · found by an internal review · page: test
- **Problem**: After b639a0c Start was below the fold.
- **At ba6e85e**: The Run panel's body scrolls over a fixed foot that holds Start, its reason, the job's state and Cancel. Measured with no page scroll and no sideways scroll, before and after a job: Start at y 790–830 at 1850x960, 705–745 at 1440x900 and 669–709 at 1536x864, with Cancel and Register in view each time.
- **Sources**: OPS:363-364
- **Closed at**: —

#### OPS-C9 · Models had more than one scroll region at some heights

- **State**: code-verified at ba6e85e (audit) · found by an internal review · page: models
- **Problem**: Two scroll regions.
- **At ba6e85e**: Measured at 1850x960, 1850x760, 1536x760, 1920x1080 and 1850x1300, on a 15-row workspace with a long detail: the page itself does not scroll. Exactly one region scrolls in each panel: shep-models-research-rows for the list (e.g. 501/1227 px at 1850x960) and shep-mp-detail-bd for the detail (682/1967). On the shared 4-row demo, the list region scrolls only where it must (1850x760, 1536x760). At 960 px wide the page is the one scroll region.
- **Sources**: OPS:363-364
- **Closed at**: —

#### R-55 · Sticky chart toolbar covered chart titles when scrolled

- **State**: code-verified at ba6e85e (audit) · found by the 2026-10-25 review · severity low · page: training
- **Problem**: A sticky toolbar inside the chart scroller covered the first row's titles when scrolled ('the same class of problem as the status line the owner reported').
- **At ba6e85e**: The chart toolbar is a static row of its own, outside the scrolling chart body (training_console.py:5051-5076). At 1536x760 the body scrolls (527 px of content in 293 px); after scrolling it by 120 px, document.elementFromPoint at every plot corner still hits the plot. At 1850x960 nothing scrolls.
- **Sources**: UX:257 (appendix 55); RJ audits[1].findings[21]
- **Closed at**: —

#### TR-F1+owner · Plot size stuck after narrow; sticky status line covered axes

- **State**: code-verified at ba6e85e (audit) · found by the owner · page: training
- **Problem**: At b14c4de a narrow canvas height stuck in the wide plots and the resources line left the view; then the sticky resources line hid the bottom charts' ticks and axis label at the owner's window.
- **At ba6e85e**: Resources is a row of its own below the charts (#resource_display flex none, training_console.py:5068-5069). Going wide, then narrow (960), then wide again, the plot areas measured 183/170 px, then 213, then 182/170, and the canvases 229/217 px (not stuck at the narrow 260). No axis corner is covered (elementFromPoint). The bottom plots end at y=851 and resources start at y=873.
- **Sources**: M5W:21; commit 4764faa (F1); commit 1f4eb05 (F1); UX:100-101; RJ research[1].principles[2] (G3); RJ research[2].principles[9] (ML-CH-4); RJ research[4].principles[3] (S4)
- **Closed at**: —

### UI-03 Field-row alignment

2 rows: 2 to do.

#### FL-UI-03 · Fine-tune's Refresh sits 15 px below the Run and Checkpoint selects

- **State**: to do · found by the owner · severity unrated (fix list) · page: finetune
- **Problem**: In the base-picker row the Refresh button is size='sm' and the row is bottom-aligned (finetune_page.py:1192-1197, CSS :1389 'align-items: flex-end', :1391). At 1850 x 960 the two selects' boxes span y 309-348 (centre 329) and Refresh spans y 329-358 (centre 344). The button sits 15 px lower, its top level with the selects' centre line (seen).
- **At ba6e85e**: Recorded only in the fix list rev 1. The ops close fix gave Refresh the gated-button rule only. Open at ba6e85e.
- **Done when**: At 1850x960, 1536x760 and 200% zoom the Refresh button's vertical centre equals the selects' box centre within 2 px, through the shared row style.
- **Sources**: the owner's fix list rev 1 (2026-11-16); src/webui/components/finetune_page.py:1192-1197, 1389-1391; docs/working/ops-wiring-and-ui-ux/README.md:364 (close fix 7220957: Refresh by the gated-button rule, not its alignment)
- **Closed at**: —

#### R-24 · Diagnosis input column has three left edges; Options is a box inside the panel

- **State**: to do · found by the 2026-10-25 review · severity low · page: diagnosis
- **Problem**: In the Diagnosis input column left edges did not line up and Options was a box inside the panel.
- **At ba6e85e**: Measured at 1850 x 960: - the HPO terms and Patient ID labels and boxes start at x=41; - the Options box spans x=33-369, with a 1 px #e1e4ea border and a #f8f9fb background; - inside it 'Options' starts at x=44 and the Top-K label at x=108, and the Explanations checkbox wraps to x=44, under 'Options' rather than under Top-K; - the 'MODEL IN USE' title and Run start at x=33. That makes three left edges (33, 41, 44), and Options is a box inside the panel next to the card's own grey surface. At 960 px wide the Options box (x=489) sits 8 px left of Patient ID (x=497).
- **Done when**: At 1850x960 every label and control in the Diagnosis input column starts at one left edge, Options has no border or background, and Options' controls sit in a two-column label/control grid.
- **Sources**: UX:226 (appendix 24); RJ audits[0].findings[22]
- **Closed at**: —

### UI-04 The shared model list: folding, search, lineage

14 rows: 5 to do, 6 partly done, 1 outside this round's scope (owner to decide), 2 code-verified at ba6e85e (audit).

#### B39-UI2 · Model list reads every checkpoint though it shows only four runs

- **State**: to do · found by an internal review · page: models
- **Problem**: Reads more than it shows (M5-3 UI-2).
- **At ba6e85e**: model_list.list_models calls server_lists.list_checkpoints(chosen) for the whole workspace (model_list.py:555). That reads every run's checkpoints (server_lists.py:696, read_run_checkpoints over all runs, cached per file after the first read). The clinical preset ('Change model…') shows only the latest 4 runs (model_list.py:248, latest_runs=4; :761) from that full read. list_run_checkpoints is not used for a read-free listing. The demo has 2 runs, so the cost cannot show (inferred from code).
- **Done when**: A unit test shows the first listing and Change model read only the runs shown, and more only after 'Show older runs'.
- **Sources**: BL:997 (item 39)
- **Closed at**: —

#### B69-L1 · Unlistable runs/ entry reads 'RUNS · <name>'; name line upper-cases the directory

- **State**: to do · found by an internal review · severity cosmetic · page: models
- **Problem**: Such names read 'RUNS · <name>'; the 'architecture' field names a location.
- **At ba6e85e**: Seen in the real app (my private copy with a workspace copy holding runs/bad\xff, runs/gatv2/run\xff and an alias link). The rows read 'bad\xff · RUNS', 'run\xff · GATv2', 'demo-disease · ALIAS 2026-11-16 09:13' and 'demo-base · ALIAS …'. The name line upper-cases the directory name (renderers.architecture_label :719-725, 'A location, never the model's architecture'). The API's architecture field is 'runs' or 'alias'. The visible reason reads 'metadata not readable · cannot be loaded' even for a directory that was never read.
- **Done when**: An unlistable directory's row reads its plain name and location in words (no upper-cased directory), and the API 'architecture' field names the encoder, not a directory.
- **Sources**: BL:1027 (item 69)
- **Closed at**: —

#### B69-L3 · Broken link or link loop under runs/ skipped without a word

- **State**: to do · found by an internal review · severity cosmetic · page: models
- **Problem**: Such an entry vanishes silently.
- **At ba6e85e**: I put a broken link (runs/gatv2/broken-link -> /nonexistent) and a link loop (runs/gatv2/loop-link -> itself) in the workspace copy. Neither shows in the list or in the route's answer. The count reads 'newest first · 13 checkpoints in 7 runs', with no word about the two entries.
- **Done when**: With a broken link and a link loop under runs/, the model list and route say each was skipped and why.
- **Sources**: BL:1027
- **Closed at**: —

#### B69-L4 · Alias link inside the workspace lists each checkpoint twice

- **State**: to do · found by an internal review · severity cosmetic · page: models
- **Problem**: Each file listed twice through an alias link.
- **At ba6e85e**: I put runs/alias -> gatv2 in the workspace copy. Every file is listed twice: 63d8975e5d6c, 51d6b00e443d, 48b5931a3e75 and b67a242cce9c each appear under both GATv2 and ALIAS rows, plus duplicate corrupt.pt, odd\xff.pt and run\xff entries (15 rows for 8 real entries). The ledger counts repeat with them.
- **Done when**: With runs/alias -> gatv2, each checkpoint is listed once (or alias entries are marked as aliases) and ledger counts are not doubled.
- **Sources**: BL:1027
- **Closed at**: —

#### FL-UI-04 · Models list has no folding or search, and files of one run read alike

- **State**: to do · found by the owner · severity unrated (fix list) · page: models
- **Problem**: Every saved file is its own tall row, and families and runs cannot be folded. No search by name or fingerprint exists. Two files of one run read identically: on the shared demo both demo-disease rows read 'Disease demo-disease · GATv2 2026-11-16 09:13 1 epoch completed', and only the fingerprint and validation cells differ. Under By family all four files are listed flat under one family title (c3-models-byfamily.json). The backend already has exact parent links (server_lists family_rows), as the fix list notes.
- **At ba6e85e**: Fix list rev 1. It overlaps RC-B, whose fold was asked of the shared model list and not built (RC-B verdict partly), and RC-D and BACKLOG 31 (not started). No search exists.
- **Done when**: Families and runs show as foldable summary rows (kind, task, file count); each file row shows save kind and epoch so two files of one run read differently; a name/fingerprint search filters while keeping the family title; the selection survives fold, filter and refresh.
- **Sources**: the owner's fix list rev 1 (2026-11-16); docs/working/WEBUI_UX_REVIEW_2026-10-25.md:135-137 (recheck (B): 'fold each run's checkpoints ... in the shared model list'); docs/working/WEBUI_UX_REVIEW_2026-10-25.md:139-140 (recheck (D)); docs/working/BACKLOG.md:989 (item 31: search, archiving)
- **Closed at**: —

#### B72-C5 · Models checkpoint list groups not in their own tbody

- **State**: partly done · found by an internal review · severity minor · page: models
- **Problem**: <th scope=colgroup> without <colgroup>; rowgroup headers in one <tbody>.
- **At ba6e85e**: Compare is done: 2 <colgroup> and 8 <tbody> row groups, each headed by th scope=rowgroup (results_segment.py:1290-1318). The Models checkpoint list is not: under By family, its table has 1 <tbody>, the group row 'Family from run demo-base · 2026-11-16 09:13 (4 files)' is a <tr><th scope=rowgroup> inside it, and there is no colgroup (model_list.py:857-868).
- **Done when**: Under By family (and the clinical preset's 'In use now'), each group is its own <tbody> headed by th scope=rowgroup, checked by a unit test on the rendered table.
- **Sources**: BL:1030
- **Closed at**: —

#### R-27 · Load drawer: count line wraps beside the select, lone row not preselected, cryptic footer

- **State**: partly done · found by the 2026-10-25 review · severity low · page: diagnosis
- **Problem**: The Change model drawer had a misaligned meta line, a dead end with one candidate, and a cryptic footer and title.
- **At ba6e85e**: Fixed: - the drawer is titled 'Load for inference' for both 'Load a model…' and 'Change model…', so the title no longer contradicts its button (50c21a1); - the count line uses correct plurals ('2 checkpoints in 1 run', model_list._counted, in place before 2026-10-30). Still open: - the count line still sits to the right of the select and wraps ('newest first · 2 checkpoints / in 1 run' at x≈1017, y 187-208), not on its own line under it; - nothing is preselected: kept_selection returns None for a fresh listing, so the facts say 'Select a model to see its facts.' and Load stays disabled until a click. With exactly one row that is the dead end the audit described; the demo's run has two files, so that case is inferred from code; - the footer is unchanged: 'Load reloads the pipeline and asks once, from → to. Replaces what every session is served; …'; - the rows say 'Recorded results: no ledger', an internal word.
- **Done when**: In the Load for inference drawer the count line sits on its own line under the select without wrapping, a single candidate row is preselected, and the footer wording and 'no ledger' are replaced by plain next-step text.
- **Sources**: UX:229 (appendix 27); RJ audits[0].findings[24]
- **Closed at**: —

#### R-7+B51a · Load-for-inference drawer rows are dot-joined chains repeating dataset fingerprint/synthetic

- **State**: partly done · found by the 2026-10-25 review · severity medium · page: diagnosis
- **Problem**: Option labels (workspace selects on Diagnosis, Models, Test, Training; Change model rows) were dot-joined chains; the workspace option repeated the fingerprint and 'synthetic' the identity line under it shows.
- **At ba6e85e**: Workspace options are the name only (only 'demo' exists here). Run options carry one ' · ' ('demo-disease · GATv2 2026-11-16 09:13') on Training, Fine-tune and Models and in the drawer heading, and the drawer names each run once, in its heading. But each row of the Diagnosis 'Load for inference' drawer still reads 'model fingerprint 63d8975e5d6c · trained on demo · dataset fingerprint 50bef1bf5f9f · synthetic'. That is three ' · ' in one row, repeating the dataset fingerprint and 'synthetic' from the identity line directly above, so the dataset fingerprint appears three times on that screen.
- **Done when**: In the Diagnosis Load for inference drawer, a row trained on the chosen workspace shows no trained-on, dataset fingerprint or 'synthetic', and each row sub-line has at most one ' · '; a unit test on the shared row renderer pins it.
- **Sources**: UX:209 (appendix 7); RJ audits[0].findings[15]; BL:1009 (item 51 (a)); BL:1030 (item 72 Test: options carry up to five ' · '); UX:48-49 (section 2 layer 2); RJ research[4].principles[6] (S7: 'Long dot-joined option strings were the owner's complaint')
- **Closed at**: —

#### R-77+B51f · Models count line repeats the order word and has two dot separators

- **State**: partly done · found by the 2026-10-25 review · severity low · page: models
- **Problem**: Models' count line read '1 checkpoints in 1 runs' and repeated the Order toggle; ordering mode used dot-joined headers.
- **At ba6e85e**: The plural is fixed: 'newest first · 4 checkpoints in 2 runs', and with Task = Base 'newest first · 2 checkpoints in 1 run · Base only, of 4 listed' (model_list._counted :871-873; count_line :876-906). The line still opens with the order word ('newest first', or 'in the groups below'; model_list.py:898, 905) directly under the 'Newest first' toggle. The filtered line has two ' · '. The ordering-mode headers and options no longer exist, because By test score retired in 3ed3938.
- **Done when**: The count line under the order toggle does not start with 'newest first'/'in the groups below' and contains at most one ' · ', also when a Task filter is set.
- **Sources**: UX:279 (appendix 77); RJ audits[2].findings[19]; BL:1009 (item 51 (f), second half)
- **Closed at**: —

#### RC-B · Diagnosis model drawer does not fold run files; rows repeat fingerprint, chip and workspace

- **State**: partly done · found by the owner's recheck · page: diagnosis
- **Problem**: The model drawer laid every run's checkpoints out flat; runs hard to tell apart, facts repeated (screenshots 06_1, 06_2).
- **At ba6e85e**: Done: - the drawer groups files by run under a heading ('Run demo-disease · GATv2 2026-11-16 09:13'); - files have short names under it ('Latest save, 1 epoch completed', 'Best pooled MRR, 1 epoch completed'); - the chosen file's facts show beside the list, and the model in use is pinned. Not done: - runs are not folded: every file of each of the latest 4 runs is listed open, and older runs sit behind 'Show older runs (n)'; - facts still repeat. The workspace line already says 'dataset fingerprint 50bef1bf5f9f · synthetic', yet each row says again 'model fingerprint … · trained on demo · dataset fingerprint 50bef1bf5f9f · synthetic' (three ' · '; skill §5 allows one), with a 'Disease' kind line on top; - the model in use is named twice: in the pinned row and again in its run ('— in use, pinned above').
- **Done when**: In the Diagnosis model drawer each run heading folds and unfolds its files, and file rows no longer repeat the dataset fingerprint, synthetic chip or workspace name (a relation such as 'this dataset' instead).
- **Sources**: UX:135-137; BL:1010 (item 52 (B); 'two-level selects elsewhere'); BL:182
- **Closed at**: —

#### RC-D · No training lineage view beyond By family's Lineage column

- **State**: partly done · found by the owner's recheck · page: models
- **Problem**: No lineage view of how runs derive.
- **At ba6e85e**: 'By family' groups a family under 'Family from run demo-base · 2026-11-16 09:13 (4 files)'. Its Lineage column gives each file's recorded operation and exact parent ('New run', 'Finetune of 48b5931a3e75'; models_page.py:1143-1179). Two files of one run both read 'New run', so epochs are not treated as branches. The Results tab's Compare shows Family and Parent run as raw run ids. There is no lineage view of how runs derive, such as a run graph or tree across runs.
- **Done when**: By family shows each file's operation and parent run in run words (not raw ids) on Models and in Compare's Family/Parent run, and the record states whether this meets BACKLOG 52 (D).
- **Sources**: UX:139-140; BL:1010 (item 52 (D)); BL:989 (item 31)
- **Closed at**: —

#### B31 · No archiving from the lists and no search across workspaces

- **State**: outside this round's scope (owner to decide) · found by an internal review · page: models
- **Problem**: BACKLOG 31 records three things beyond M5's Models page: a side-by-side comparison (built since as Results › Compare, research-comparison Stage 8b), archiving checkpoints from the lists while keeping them referenceable, and search across workspaces ('which checkpoints were trained on dataset fingerprint X'). Archiving and search do not exist at ba6e85e. OPS U3 excluded 'archive or delete tools' from the round.
- **At ba6e85e**: Recorded 2026-10-19. Compare is built; archive and search are not started. Only cited as a source of RC-D in the inventory.
- **Why this state**: Compare is built; what remains, archiving checkpoints and search across workspaces, are deferred new product surfaces in tasks_def.md. In-workspace name/fingerprint search is covered by FL-UI-04 (UI-04).
- **Sources**: docs/working/BACKLOG.md:989 (item 31); docs/working/BACKLOG.md:183 (§0 'Later': '30, 31'); docs/working/ops-wiring-and-ui-ux/README.md:172 (U3)
- **Closed at**: —

#### B66 · Non-text run or file name failed both checkpoint lists

- **State**: code-verified at ba6e85e (audit) · found by an internal review · page: models
- **Problem**: HTTP 500 hid every other file.
- **At ba6e85e**: In the workspace copy, runs/bad\xff, runs/gatv2/run\xff and a checkpoint named odd\xff.pt each show as one row with its reason ('… is not listed (its name is not valid text, shown here with backslash escapes; it cannot be opened by name), so it is not read'). Every other file is listed. GET /api/v1/workspaces/linkcases/checkpoints answers 200 with 15 entries.
- **Sources**: BL:1024 (item 66)
- **Closed at**: —

#### B72-N6 · Diagnosis drawer rows repeated their run under a run heading

- **State**: code-verified at ba6e85e (audit) · found by an internal review · page: diagnosis
- **Problem**: Each row repeated its run.
- **At ba6e85e**: Under a run heading the rows say only what each file is and how far it trained. The run is named once, in its heading.
- **Sources**: BL:1030 (B3 note)
- **Closed at**: —

### UI-05 The ordering switch's events

1 rows: 1 code-verified.

#### FL-UI-05 · By family is selected while the table and count line read newest first

- **State**: code-verified · found by the owner · severity unrated (fix list) · page: models
- **Problem**: In the owner's screenshot, Order shows 'By family' selected, yet the count line reads 'newest first · 16 checkpoints in 8 runs'. The table has no family titles and no Lineage column, and the Base row comes after the fine-tunes. Not reproduced here: on the shared demo (one family, 4 files), within 400 ms of choosing By family the table showed 'Family from run demo-base · 2026-11-16 09:13 (4 files)' and a Lineage column, and it stayed so at 4 s (c3-models-byfamily.json). The cause is unknown. Inferred candidates: more runs or families, a slow answer, or a redraw lost in transit.
- **At ba6e85e**: Fix list rev 1 only, marked 'cause to be reproduced'. Not reproduced on the shared demo.
- **Done when**: After switching Order (including repeated and fast switches), the control, count line wording, family titles/Lineage column and row order all match the chosen order; while pending or failed the page says so.
- **Cause (reproduced in T02)**: the page's one listener (`_sync`, "always_last") reshaped the list only when the event that fired was a control, and Gradio 6.30 runs a deferred call again with the values gathered then but the first trigger's id. On a CPU demo copy at ba6e85e, with the selection's `_sync` join delayed by `page.route`: a row chosen, then By family, ended with the switch on By family, the count line "newest first", no family title, no Lineage. A fast By family then Newest first ended on By family: every answer wrote the controls back, so the late answer set the switch back and the deferred call read it. A Task chosen while a selection answered was dropped the same way.
- **Fix**: every answer shapes the view from the controls as they read and writes it when the list does not hold it (`same_view`, the view an input), whichever trigger asked; a control's value is written only where it cannot apply as it reads; a failed answer changes nothing and says under the options what was not applied (the error's kind only). D05: "Applying By family…" / "Applying Task: Disease…" beside the switch until the rows drawn say they answer the controls (`model_list.view_marker`, the view's `shaped_for`); no page lock.
- **Sources**: the owner's fix list rev 1 (2026-11-16)
- **Closed at**: 4e419fb (with d2d02f6 for the Task select's one request). Tests: `test_webui_models_page.py::test_a_control_changed_while_the_page_answers_is_applied_and_one_pick_is_one_request` (browser, the dashboard as mounted, one `_sync` answer held: a row then By family; By family then Newest first, the second answer held until the first one's rows are drawn; a row then Task Disease; each ends with the switch, the count line, the groups and Lineage agreeing with the last change and the line cleared, "Applying …" said while pending; on ba6e85e all fail: By family dropped, the switch set back, Disease dropped), and the unit cases `test_an_answer_writes_a_control_back_only_where_it_cannot_apply`, `test_an_answer_that_fails_says_what_was_not_applied_and_changes_nothing`, `test_the_rows_say_the_view_they_were_drawn_for_and_another_workspace_drops_it`. Seen on a CPU demo workspace at 1850×960 and 960×1080: the line under the switch at 14 px, gone once applied, no sideways scroll. The other "who fired" listeners were checked: no other always_last listener reads its trigger (the Diagnosis drawer's Task is a Radio whose one listener writes the view only).

### UI-06 Launch and configuration in the GUI

10 rows: 2 to do, 3 partly done, 1 awaiting owner acceptance, 4 code-verified at ba6e85e (audit).

#### FL-SVC · Service stopped during manual acceptance with no recorded reason

- **State**: partly done · found by other · page: shell
- **Problem**: The fix list's section 4 records that the manual-acceptance service log shows a normal shutdown at 16:09:14, with no crash trace and no known trigger. The reviewer restarted it with the acceptance launcher and handed it to a user-level background service. It is explicitly not a reproduced product crash, and the restart is not a fix for UI-06.
- **At ba6e85e**: Cause unknown, not reproduced. Recorded only in the fix list.
- **Progress (2026-11-16, T03 launcher)**: The stop asked for from Settings is named. `src/api/services/backend_control.stop_backend` (the Stop service backend path; the button is not built yet) logs `Stopping the service (pid N): asked for by Stop service in Settings.` before it sends the orderly stop's SIGTERM. Checked by test_backend_control (`test_stop_service_stops_a_real_server_through_its_orderly_stop`, a real uvicorn server): the line comes before the application's `Shutting down`. A stop by a signal from outside is still logged only as uvicorn's `Shutting down`, without the signal or its sender, and a crash writes no line, so the 16:09:14 shutdown would still read the same. A start without a terminal now appends the server's output to a launch log in the state directory, so such a log is kept at all (FL-UI-06).
- **Progress (2026-11-16, T03 gui-docs: b13e262)**: Stop service is in Settings beside Restart Backend: asked once ("Stop the service? …", Stop service now or Cancel), off while a GPU job runs (said beside it), locked with D8's reason for a read-only request, and an operator control checked on its own request (test_webui_data_locations, test_operator_access). In Chromium on the real launcher it stopped the server with the reason line in its log before uvicorn's `Shutting down`, and the instance file was removed. A stop by a signal from outside and a crash are still not named, as before.
- **Done when**: Every service stop writes a log line naming its trigger (signal, request, crash); a unit test sends SIGTERM and a stop request and checks the reason line.
- **Sources**: the owner's fix list rev 1 (2026-11-16)
- **Closed at**: —

#### FL-UI-06 · Data locations and registry need environment variables before start; no GUI setup

- **State**: awaiting owner acceptance · found by the owner · severity unrated (fix list) · page: settings
- **Problem**: launch_shepherd.sh starts scripts/launch/shep_launch.py without loading any deployment file. src/config/server_settings.py reads M1, the workspaces root and the registry only from the environment and is read-only without M1. Settings saves compute settings only. A clinician therefore cannot open the dashboard by double click and set data locations in the GUI. The automated acceptance passed only because its script pre-set them.
- **At ba6e85e**: Fix list rev 1 only. Not built. Earlier records treat deployment settings as environment-only (M5-1 rule cited in the fix list).
- **Progress (2026-11-16, T03 core: dd5f162, f92676b, 3d27037)**: The owner withdrew the M5-1 environment-only rule. The service side is built: `server_settings` resolves every data location at its one read point (`_resolve`): an explicit environment variable first, then the applied values of a saved settings file outside the repository (`src/config/deployment_settings.py`: `SHEPHERD_SETTINGS_FILE`, else the user config dir), then nothing. Each `Setting` carries its `source`; a saved value is checked by the same directory rules at each read; an unreadable file is a problem naming it for every setting it would give, never read as empty. Writes go through `src/utils/settings_store.py` (the file's lock, one atomic writer shared with `runtime_presets.save_runtime_settings`, 0600/0700, an unreadable file renamed only by the explicit repair). Apply (`src/api/services/data_locations.py`) runs only with the job runner's slot idle and every GPU lock the old values name free; otherwise the change stays pending with its reason. The backend's startup tries it; `take_gpu_lock` re-reads the location once it holds the lock. The deployment mode gets the local start's default (`SHEPHERD_LOCAL_ENTRY=1` with a loopback bind host), and an explicit mode is never upgraded. The problem words point to Settings › Data locations, with the variable as "Ref:". Tests, the probe and its servers each name a settings file of their own. Checked by unit tests (test_server_settings, test_backend_control, test_operator_access, test_runtime_presets, test_probe_m5_acceptance, test_results_export). The real launcher was also started with `HOME` and `XDG_*` in scratch directories and no `SHEPHERD_*` set, after a save and apply from a plain shell. The Datasets page listed the demo workspace from the saved file at 1850x960, read-only. Started again with `SHEPHERD_LOCAL_ENTRY=1` set by hand (the launcher's `--local-entry` is not built), the same values were in effect and the header read 'Operator controls: on (this machine)'. A gated route answered 200 on loopback and 403 with a foreign Host or a forwarding header. Not built yet: the launcher's `--local-entry` and `--settings-file` and its any-cwd start, the Settings › Data locations panel, the HTTP view and CLI, and the docs. The double-click acceptance is not checked.
- **Progress (2026-11-16, T03 launcher)**: The launcher half is built. `launch_shepherd.sh` finds its own checkout (links followed) and runs that checkout's venv and launcher with `--local-entry`. It no longer looks for `.venv` in the directory it was started from. `launch_shepherd.cmd` does the same with `cd /d` to its folder (not run on Windows here). `shep_launch.py` changes to the repository before it starts the server, after it has read a relative `--settings-file` against the start directory. It removes its own variables from the environment it was given. It sets `SHEPHERD_LOCAL_ENTRY=1` only for `--local-entry` with a loopback bind, and `--settings-file` becomes the service's `SHEPHERD_SETTINGS_FILE`. One backend per configuration (`src/utils/launch_instance.py`): the launcher takes a lock named after the repository and the settings file in the user state directory and hands it through its exec to the server. It writes a 0600 info file (pid, port, URL, a random instance id, repository, settings file, log). `/health` answers that id, and the server removes the file at shutdown. A second start of a configuration whose lock is held and whose `/health` answers with the file's id opens that page and exits 0; a 200 without the id is never taken for it. Port: with no `--port`, 8000, else the first free one of 8001…8019, said in the log; a `--port` in use is refused with the reason. Without a terminal, `--local-entry` appends the launcher's and the server's output to `logs/launch.log` in the state directory, moved aside past 10 MB. Output already sent to a file or a pipe (`nohup … > my.log`, a script that reads it) stays there. A failure before the server is up (no venv, a refused bind, a refused port, a server that ends before it answers) opens a local error page saying what failed, what to do and where the log is. `launch_shepherd.sh --install-desktop-entry [--settings-file PATH]` writes `~/.local/share/applications/shepherd-advanced.desktop` (Exec runs this checkout's absolute `launch_shepherd.sh` by bash, `Terminal=false`, the repository's icon; a slot's entry has its own name). The Stop service backend path is in `backend_control.stop_backend`: refused while a GPU job or a model load or stop runs; after its delay it closes the idle slot and sends the orderly stop's SIGTERM. Its Settings button is not built. The backend's startup Apply was already wired by T03 core. Checked by unit tests (test_service_unit, test_launcher_browser_wait, test_launch_instance, test_backend_control) and the related files. With the real launcher, `HOME`/`XDG_CONFIG_HOME`/`XDG_STATE_HOME`/`XDG_DATA_HOME` in empty scratch directories and no `SHEPHERD_*` set: `launch_shepherd.sh` started with no terminal from another directory served from the repository. `/health` named the info file's id. With stdout at /dev/null the output was in the launch log, and with stdout in a file it stayed in that file. tests/integration/test_backend_stops_its_job.py passed (17), its `script` case included; that case first failed on the redirect, which was then narrowed. The operator controls were on (local start), and the demo workspace was listed from the saved file. A second start, and a start through the installed desktop entry's Exec line from `/`, opened the running page and exited 0. A slot with its own settings file on the taken port was refused with the error page. After a SIGTERM by pid the info file was gone and the lock free. The desktop Exec line started again on another port with the same saved values in effect (source saved). An explicit `SHEPHERD_DEPLOYMENT_MODE=read-only` start stayed read-only. A server that exited before answering opened the error page. Not done: the Settings › Data locations panel, the Stop service button, the HTTP view and CLI, and the docs. Not checked: a real desktop session's double click (no desktop here; the Exec line was run as a desktop runs it), Windows and macOS, and Restart Backend on a launcher-started server (the lock descriptor was seen without close-on-exec in the running server, and a unit test shows the lock goes through an exec).
- **Progress (2026-11-16, T03 gui-docs: e45d972, b13e262, c42a088, a81e83c, bcbf9ac)**: The GUI half is built. Settings has a Data locations panel beside Runtime settings (wide 55:45, narrow above it; `src/webui/components/data_locations.py`), drawn at each opening from one view of the data locations (`src/utils/deployment_view.py`, through `server_settings`' read point, no reader of its own). Each row (Workspaces folder, Results registry, Import folder, Training evidence folders) says what needs it, what is in effect and its source in plain words, the saved value when it differs, and a path box with Check (the same rule a job's read applies, the resolved path and what the server's lists find: "1 workspace") and Save (saved and applied at once). A required folder that is not set offers Create suggested folder (`~/SHEPHERD-data/<name>`, created only on that click, 0700 for the registry and the import folder). A row the environment sets keeps its box and Save and says the saved value applies once the start-up setting is removed. A pending change has one notice with its reason and Apply; an unreadable file one notice with Replace (renamed, kept), Save off and Create hidden. A read-only request sees "Set (path hidden)", every control locked and D8's reason once. Every act is an operator control, Check included. `GET /api/v1/settings/deployment` (open, read-only, paths hidden for a request D8 refuses) and `scripts/deployment_settings.py show` give the same view. The words that told a clinician to set `SHEPHERD_*` variables (the GPU chip and its bubble, the D8 chip's bubble, Datasets' refused root, Diagnosis' missing registry, the Test page's help) now point to Settings › Data locations, the variable only as "Ref:". Guides en and zh (§4.9, §4.11–§4.14), README, the playbook, PLAN_TEST_RESULTS' current rules and BACKLOG 45/17 say how it is set up now. Checked by unit tests (test_webui_data_locations, test_operator_access's gated handlers and route, test_server_settings' view and CLI, test_webui_gradio6, test_webui_chips, test_entry_documents, test_backend_control) and in Chromium on the real launcher. `launch_shepherd.sh` was started from another directory with `HOME` and `XDG_*` in empty scratch directories and no `SHEPHERD_*` set. The dashboard opened read-write (local start), with Settings showing "Nothing is set yet" and the GPU chip "refused, no workspaces folder". Create suggested folder made `~/SHEPHERD-data/registry` 0700 and saved it. Check on a folder holding the setup_demo.py workspace read "Usable … (1 workspace)", and Save read "Saved and in effect", the GPU chip turning to "none". Datasets then listed `demo`. The route answered the same view as `deployment_settings.py show` run from a plain shell (they differ only in the mode: the shell is not the local start). Stop service asked once, stopped the server with "Stopping the service (pid N): asked for by Stop service in Settings." in its log, and removed its instance file. Started again, the same values were in effect (Saved in Settings). A start with `SHEPHERD_DEPLOYMENT_MODE=read-only` showed "Set (path hidden)", every control locked with the reason, and no scratch path in the panel or the route. A start with `SHEPHERD_REGISTRY` set showed the override; a corrupted file showed the unreadable notice, and Replace kept the copy. At 1850x960 the page did not scroll and no text was under 14 px; at 760x1000 the panels stacked with no horizontal scroll. Screenshots outside the repository (scratchpad t03/gui-docs). With the scratch paths of about 120 characters, the overridden and unreadable states make the panel 850–870 px tall, so the Settings region scrolls by a few pixels at 1850x960; with paths like the mockup's they fit. Not checked: a double click in a real desktop session (no desktop here; the launcher part ran the desktop entry's Exec line), Windows and macOS, and the mockups' visual acceptance by the owner (not approved yet).
- **Progress (2026-11-16, T03 review fixes: 0f11217, 4e6fb0e)**: Two review findings fixed. (1) A read-only request saw a refused value's own words, which quote it as given: the redaction hides only absolute paths, so `SHEPHERD_REGISTRY='clinic/MRN 0012345/registry'` or `~/intake/Jane Doe` was shown by `GET /api/v1/settings/deployment` and the panel's Problem row. Hidden, a refused value, a location from an unusable settings file, a refused `SHEPHERD_SETTINGS_FILE` and the GPU lock under a refused folder now read in fixed words ("Results registry set when the service started is refused: an operator sees why in Settings › Data locations."); an operator still reads the refusal. (2) Saving and applying a folder that the service's own start-up setting overrides moved every other process's value: with `SHEPHERD_WORKSPACES_ROOT=A` on the service and B saved, a command run by hand with no variable took B's GPU lock while the service's job held A's. Apply now holds back such a key (`settings_store.held_back`): its applied value is kept and it applies at a start without the setting; a saved value equal to the setting applies as before. The panel offers no Apply for it, and its row's "Saved, applies when the start-up setting is removed" is now true. Checked by regression tests that fail on the old code (test_server_settings, test_webui_data_locations) and the related files (729 passed). The real launcher was then started with the override, a pending saved root and relative registry and import values, with `HOME`/`XDG_*` in scratch directories. Its startup Apply logged the hold-back and left the file's applied part empty. The read-only route showed no value, `held_back: ["workspaces_root"]` and the fixed words. A plain shell's `deployment_settings.py show` read "GPU lock: not configured". Not checked in a browser: the panel's held-back row (unit tests only).
- **Done when**: Starting by double click with no environment variables opens the dashboard; Settings lets the operator set workspaces root and registry, saves them, and reuses them after restart; GUI, CLI and jobs read the same settings source; a missing location disables only the operations that need it.
- **Sources**: the owner's fix list rev 1 (2026-11-16); docs/working/BACKLOG.md:975 (item 17 F7, the launcher); docs/working/BACKLOG.md:1003 (item 45: commands refuse in a fresh shell); docs/working/BACKLOG.md:1013 (item 55: what serves after a start); docs/working/BACKLOG.md:987 (item 29)
- **Closed at**: —

#### NEW-settings-datasets-footer-1 · Settings says where torch.compile is read twice in the same words

- **State**: to do · found by the 2026-11-16 audit · page: settings
- **At ba6e85e**: Settings says where torch.compile is read twice in the same words on one screen. The facts row reads 'torch.compile saved: Off (read when the Training page starts, continues or branches a run; Fine-tune does not read it)' and the checkbox's info line repeats it (runtime_settings.py:210-211 and :319, from COMPILE_READ :104-109, whose comment says it is said in the facts, under the field, in its help and after Apply). This is against skill §1.2. Screenshot (audit evidence, not kept in the repository)
- **Done when**: On the Settings screen the 'read when the Training page starts…' sentence appears once.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### NEW-settings-datasets-footer-2 · Restart note names SHEPHERD_CHECKPOINT_PATH and repeats the allocator info

- **State**: to do · found by the 2026-11-16 audit · page: settings
- **At ba6e85e**: Settings' restart note under the action row is three sentences of body text that names the internal variable SHEPHERD_CHECKPOINT_PATH (runtime_settings.py:90-93 and :412-415), against skill §5, which allows internal codes only beside a plain label or in a trailing 'Ref:'. It also repeats the allocator field's info line ('Restart Backend applies a saved change'). Visible at 1850x960 (same screenshot).
- **Done when**: The restart note is one plain sentence without internal variable names in body text (any code only in a trailing 'Ref:') and does not repeat the allocator field's info line.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### R-12 · Restart Backend is one click with no confirmation and not tied to a pending restart

- **State**: partly done · found by the 2026-10-25 review · severity medium · page: settings
- **Problem**: Restart Backend was one click for every session; the page never showed running vs saved allocator.
- **At ba6e85e**: Running against saved is fixed. 'Allocator now' reads the allocator this backend runs from its own environment (running_allocator), and 'Allocator saved' is shown apart from it. When the two differ, the saved row shows a ▲ 'Not running yet: Restart Backend applies it', and this survives a reload because it is computed on every page open (test_webui_gradio6.py:156). In the demo both rows read 'Driver memory pool (cuda_async)'. The restart half is not fixed. An operator's click on Restart Backend still calls restart_backend() at once, with no confirmation step. The button is enabled whether or not a restart is pending, and it sits in the same row and at the same height (40 px) as Apply; only its variant (secondary) differs. One click still reconnects every open page. Read-only sessions are now locked out by D8.
- **Done when**: Clicking Restart Backend shows a confirmation row ('every open page reconnects' with Restart and Cancel), and Restart is enabled or marked as wanted only when the saved allocator differs from the running one.
- **Sources**: UX:214 (appendix 12); RJ audits[0].findings[19]; UX:125 ('running against saved allocator')
- **Closed at**: —

#### R-13 · torch.compile warning substance hidden behind accordion; 'Experimental' at line end

- **State**: partly done · found by the 2026-10-25 review · severity medium · page: settings
- **Problem**: The torch.compile warning hid in a bare 'Details' accordion; badges used six hues with no shared meaning.
- **At ba6e85e**: The six-hue badges are gone (0ee56f9 retired the coloured badges, their raw hex and the red 'Restart required'). The only mark left on the page is the shared neutral ○ 'Not implemented yet'. The two accordions now have specific labels, 'What each allocator does' and 'About torch.compile'. The torch.compile warning itself is still inside the closed 'About torch.compile' accordion: 'Experimental: keep it off unless you are benchmarking it ... compare MRR and Hits@K with an eager run before trusting a result'. The only visible trace is the last word of the checkbox's info line: 'Read when the Training page starts, continues or branches a run; Fine-tune does not read it. Experimental.'
- **Done when**: The torch.compile field shows a visible info line or warn mark that leads with 'Experimental' and says to keep it off unless benchmarking, without opening the accordion.
- **Sources**: UX:215 (appendix 13); RJ audits[0].findings[20]
- **Closed at**: —

#### B17-F7 · Launcher gave up opening a browser after 60 s

- **State**: code-verified at ba6e85e (audit) · found by other · page: launch
- **Problem**: Found at N1's homelab acceptance: shep_launch.py stopped waiting after 60 s, while a start that loaded a large SP table took about 84 s. Fixed in batch B (2026-10-18): it polls /health with no deadline, logs about every 30 s and opens the browser once.
- **At ba6e85e**: Fixed 2026-10-18 (batch B), reviewed, merged #128. The inventory carries F5 and F6 of item 17 but not F7.
- **Sources**: docs/working/BACKLOG.md:975 (item 17 F7)
- **Closed at**: —

#### OPS-B5b · Settings offered acceleration toggles and HGT claims not built

- **State**: code-verified at ba6e85e (audit) · found by an internal review · page: settings
- **Problem**: Acceleration area and help claimed behaviour not built (inferred from B5's wording and 0ee56f9).
- **At ba6e85e**: The Acceleration row in the facts list reads '○ Not implemented yet (new acceleration work is deferred; each run's record keeps the kernels and precision it ran with)', with no toggle. The help and accordion texts no longer state HGT behaviour as current. Before: 'dynamic PyG/HGT workloads may graph-break' and 'HGT has been observed to run slower'. Now: 'GATv2, the one architecture built today, has not been measured with it', and the HGT figures are 'measured earlier on this project with an architecture no longer built'. No acceleration toggle existed in src/webui before B5 (git grep at 0ee56f9^ finds none), so that half of B5 was a guard against adding one, not a removal.
- **Sources**: OPS:218-219 (B5); commit 0ee56f9
- **Closed at**: —

#### R-11 · Settings was a wall of text with actions above settings, stretched controls

- **State**: code-verified at ba6e85e (audit) · found by the 2026-10-25 review · severity medium · page: settings
- **Problem**: Settings was a wall of text with actions above the settings they apply, controls stretched to 1800 px.
- **At ba6e85e**: Settings is one bordered panel (max-width 1040 px, left-aligned) with a one-line intro, then a four-row summary list of facts (Allocator now / Allocator saved / torch.compile saved / Acceleration), then the two fields with their info lines and two specifically labelled accordions, then the action row (Apply Settings primary, Restart Backend secondary) in the panel foot, then a three-sentence restart note. Measured at 1850x960: the panel is 1040x669 px, the allocator block is 1006 px wide (its input about 982 px), the action row is at y=604 below the fields (fields at y=282-523), and the page does not scroll (docScroll 0). At 960x1080 the panel is 928 px with no horizontal scroll. The env-override sentence appears only when an explicit launch setting is active (saved_html).
- **Sources**: UX:213 (appendix 11); RJ audits[0].findings[18]; UX:125 (section 4 Settings)
- **Closed at**: —

#### R-14 · Read-only viewers could edit Settings fields

- **State**: code-verified at ba6e85e (audit) · found by the 2026-10-25 review · severity medium · page: settings
- **Problem**: Read-only viewers could edit Settings fields and were told to click a disabled button.
- **At ba6e85e**: For a request D8 refuses (a forwarded request), the page opens with the allocator select and the torch.compile checkbox disabled. Measured: input.disabled=true for both. Both buttons read '🔒 Apply Settings' and '🔒 Restart Backend' and are disabled. The reason 'Operator controls only: forwarded by a proxy.' is visible beside them (14 px, rgb(67,74,88) on white, 8.9:1). Because the fields cannot be changed, the 'Not saved: Apply Settings saves the fields above' prompt cannot appear for that viewer, and the open shows no click-this prompt. For the operator, fields and buttons are enabled with no lock.
- **Sources**: UX:216 (appendix 14); RJ audits[0].findings[21]; UX:125 ('read-only locks')
- **Closed at**: —

### UI-07 The model drawer's scrolling

1 rows: 1 to do.

#### RC-C · Diagnosis model drawer list and details share one scroll

- **State**: to do · found by the owner's recheck · page: diagnosis
- **Problem**: In the Diagnosis model drawer reading the details moved the list.
- **At ba6e85e**: The list and the facts share one scroll container (.shep-model-body, overflow:auto). At 1850 x 960, with the first row chosen, the facts are 782 px tall inside a 566 px body. Scrolling the body 216 px to read the facts moved the list rows by the same 216 px (y 266 → 50). Load stays in reach: it is in the foot at y 859, outside the scroller.
- **Done when**: At 1850x960 with a row chosen, scrolling the details pane leaves the list rows at the same y positions (each has its own bounded scroll) and Load stays visible.
- **Sources**: UX:137-139; BL:1010 (item 52 (C)); BL:182
- **Closed at**: —

### UI-08 Form density after expanding

16 rows: 3 to do, 3 partly done, 1 outside this round's scope (owner to decide), 1 not applicable, 8 code-verified at ba6e85e (audit).

#### B27-2 · Expanding More training settings / More settings makes scrolling long

- **State**: to do · found by the owner · severity requirement · page: training
- **Problem**: An expanded collapsible section made the scroll too long.
- **At ba6e85e**: Expanding 'More training settings' on Training grows the Parameters region from 615 to 988 px of content (373 px to scroll). Fine-tune's 'More settings' grows its form to 1242 px in a 681 px region (561 px to scroll). Accordions still open downward in one column; nothing opens sideways or step by step.
- **Done when**: At 1850x960, expanding Training's 'More training settings' and Fine-tune's 'More settings' adds little scroll (e.g. expanded fields laid out in two columns or a sideways step layout), measured by region scrollHeight, or an explicit owner decision is recorded.
- **Sources**: BL:985 (item 27 (2)); SK:28-29
- **Closed at**: —

#### NEW-training-finetune-2 · Continue drawer opens with step 3 Check cut off while Start is enabled

- **State**: to do · found by the 2026-11-16 audit · page: training
- **At ba6e85e**: At 1850x960, opening Continue… shows step '3. Check' cut at the drawer body's bottom edge, with its prompt below the fold, while '4. Start' (Continue Training) is enabled in the footer: drawer-1850x960-1-open.png in the same folder
- **Done when**: At 1850x960, opening Continue… shows step 3 Check and its prompt fully inside the drawer body, or Start is not presented as the next step before Check is visible.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### NEW-training-finetune-5 · Fine-tune's Device and Run name sit below the form's visible body at 1850x960

- **State**: to do · found by the 2026-11-16 audit · page: finetune
- **Duplicate of**: B27-1 (closed with it)
- **At ba6e85e**: Fine-tune at 1850x960: Device (default auto, refused on this host) and Run name sit at y 913-1004, below the visible part of the form's scrolling body (which ends at 846; scrollHeight 851, clientHeight 681), and the 'More settings' header is cut at the edge (finetune-1850x960.png, finetune-1850x960.json)
- **Done when**: At 1850x960 with nothing expanded, Fine-tune's Device, Run name and the 'More settings' header are fully inside the form's visible area without inner scrolling.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### B27-1 · Fine-tune form scrolls with nothing expanded at the owner's window

- **State**: partly done · found by the owner · severity requirement · page: finetune
- **Problem**: The first WebUI's pages scrolled; deployment displays are landscape 16:9/16:10, 1080p-4K, rarely portrait.
- **At ba6e85e**: The wide layout rule holds: at 1850x960 no page frame scrolls on any of the 7 pages. Diagnosis, Datasets, Training, Test, Models and Settings show their main task with no region scrolling (on tiny demo data). Fine-tune's form already scrolls 170 px inside its region with nothing expanded (681 of 851 px shown; 'More settings' is half under the footer row, and Start stays in that footer). At 960 px wide the page flows and scrolls by design (Training is 2346 px tall).
- **Done when**: At 1850x960 with nothing expanded, Fine-tune's form fits its region without scrolling and 'More settings' is fully visible above the footer row.
- **Sources**: BL:985 (item 27 'Screens' and (1)); SK:21-22; PTR:2390-2394
- **Closed at**: —

#### R-48 · Continue drawer prompt says to Check while a checkpoint is preselected and Start enabled

- **State**: partly done · found by the 2026-10-25 review · severity medium · page: training
- **Problem**: The Continue drawer's instructions contradicted its state.
- **At ba6e85e**: The drawer now shows numbered steps: 1. Run, 2. Checkpoint, 3. Check, 4. Start (B1c 7649fa0). The contradiction remains. The prompt is unchanged, 'Select a checkpoint, then click Check checkpoint.' (training_console.py:3284), while a checkpoint is already preselected and 'Continue Training' is enabled before any check (seen: resume button enabled; the start runs the check itself, :4093-4111). At 1850x960 the drawer opens with step 3's row cut at the body's bottom edge and the prompt below it. 'Check checkpoint' is still a size='sm' button (:4538). Before any check, the dataset relation is a filled green box ('● Trained on the dataset selected in the run bar').
- **Done when**: The Continue drawer's prompt matches its state (says the check is optional, or Start performs it), Check is a normal secondary button, and the dataset relation is one line of text, not a filled box.
- **Sources**: UX:250 (appendix 48); RJ audits[1].findings[6]
- **Closed at**: —

#### R-49 · Continue task split across drawer and run bar with duplicate controls

- **State**: partly done · found by the 2026-10-25 review · severity medium · page: training
- **Problem**: Continue split one task across the drawer and run bar, with duplicate controls.
- **At ba6e85e**: Continue Training (or Start branch) and Cancel are now in the drawer's footer beside their reason (7649fa0; seen), and Continue… and Stop are hidden in the run bar while the drawer is open. Still split across two regions: the run bar keeps a disabled Start with 'Start disabled: close Continue… to start a new run.' and the line 'RUN a new one, continued from the checkpoint chosen in Continue or branch a run'. The drawer's × Close and the footer's Cancel do the same thing (training_console.py:4715-4717). The 'Checkpoint source: From run list / Manual path' radio still takes a row near the top.
- **Done when**: While the Continue drawer is open, the run bar shows no disabled Start, no Start reason and no 'RUN a new one' line; the drawer has one closer and 'Manual path' sits behind a link.
- **Sources**: UX:251 (appendix 49); RJ audits[1].findings[7]; UX:130 (section 4: Continue's buttons in the drawer's footer)
- **Closed at**: —

#### B32-start · Training's New and Continue not merged into one mode with a single Start

- **State**: outside this round's scope (owner to decide) · found by an internal review · page: training
- **Problem**: BACKLOG 32 records 'the training form's New and Continue merged into one mode with a single Start'. At ba6e85e Training still has Start for a new run plus a Continue… drawer with its own Continue Training or Start branch. While the drawer is open, the run bar keeps a disabled Start with 'close Continue… to start a new run' (see R-49's verdict).
- **At ba6e85e**: Recorded 2026-10-19, not started. B1c reorganised the drawer (Continue or Branch as numbered steps) without merging the modes.
- **Why this state**: Merging New and Continue into one Start is listed as a deferred new product surface in tasks_def.md (plan §1 backlog).
- **Sources**: docs/working/BACKLOG.md:990 (item 32); src/webui/components/training_console.py:24, 70, 89 (module docstring: Start, Continue…)
- **Closed at**: —

#### R-44 · Choosing HGT silently changed a setting in a collapsed section

- **State**: not applicable · found by the 2026-10-25 review · severity medium · page: training
- **Problem**: Choosing HGT silently changed Mixed precision inside a collapsed Expert section.
- **At ba6e85e**: Only GATv2 is built: ENCODER_DECLARATIONS is {'gatv2': GATv2Declaration()} (spec.py:439). architecture_choices (training_console.py:976-984) therefore returns one choice, so no architecture radio is drawn (:4451); the Model accordion states 'Architecture GATv2' as a fact. No conv_type handler or HGT mixed-precision refusal remains in src/webui. Mixed precision is a field in 'More training settings'.
- **Why this state**: Only GATv2 is built (ENCODER_DECLARATIONS has only gatv2, spec.py:439), so no architecture choice or HGT handler exists that could change a hidden setting.
- **Sources**: UX:246 (appendix 44); RJ audits[1].findings[11]
- **Closed at**: —

#### B72-CD1 · Continue drawer printed inherited settings as Python repr

- **State**: code-verified at ba6e85e (audit) · found by an internal review · page: training
- **Problem**: One None stood for three different meanings.
- **At ba6e85e**: Every inherited setting is said in words by renderers.setting_words, for example 'Task = not taken by a base', 'Early stopping patience = off', 'Graph backbone = Trained'. No Python repr appears.
- **Sources**: BL:1030 (item 72)
- **Closed at**: —

#### B72-CD2 · Continue drawer's model_spec row about 3,000 px tall

- **State**: code-verified at ba6e85e (audit) · found by an internal review · page: training
- **Problem**: The whole spec with its sources in one row.
- **At ba6e85e**: The Model spec row reads 'Differs from the reference GATv2 spec: Input › Dimension 8 (2048), … and 1 more' and puts the whole spec behind a closed 'Details ›'.
- **Sources**: BL:1030; OPS:354
- **Closed at**: —

#### B72-CD3 · Continue drawer result nested three bordered boxes

- **State**: code-verified at ba6e85e (audit) · found by an internal review · page: training
- **Problem**: Three nested bordered boxes.
- **At ba6e85e**: After a check, the only bordered elements inside the drawer body are Gradio's radio wraps, at depth 0; the result has no bordered box inside another. The check result and the dataset relation are tinted fills (green or red), not nested borders.
- **Sources**: BL:1030
- **Closed at**: —

#### B72-CD4 · Finished run's first Check named internal extend_epochs field

- **State**: code-verified at ba6e85e (audit) · found by an internal review · page: training
- **Problem**: 'extend_epochs' beside the on-screen 'Extra epochs'.
- **At ba6e85e**: A finished run's first Check now reads 'The parent last.pt finished its original training (1/1 epochs), so there is nothing left to resume. To train further, extend it at a fixed rate with Extra epochs. Enter Extra epochs above, then click Check checkpoint again.' page_words maps the CLI field names to the drawer's labels (training_console.py:3304-3314). run_plan.py:627-628 still says extend_epochs for the CLI.
- **Sources**: BL:1030
- **Closed at**: —

#### B72-CD5 · Run options started with 'GATv2 · ' shared prefix

- **State**: code-verified at ba6e85e (audit) · found by an internal review · page: training
- **Problem**: The distinguishing name came late.
- **At ba6e85e**: Run options lead with the run's name: 'demo-disease · GATv2 2026-11-16 09:13' and 'demo-base · GATv2 2026-11-16 09:13'.
- **Sources**: BL:1030
- **Closed at**: —

#### OPS-B2 · Fine-tune Epochs field sat below its region

- **State**: code-verified at ba6e85e (audit) · found by an internal review · page: finetune
- **Problem**: Epochs at 883 px under a region ending at 834 px.
- **At ba6e85e**: At 1850x960 Fine-tune's Epochs field sits at y 728-818, inside the form body, which ends at 846.
- **Sources**: OPS:358 (B2 row)
- **Closed at**: —

#### R-45 · Training Parameters single-file and long at wide widths

- **State**: code-verified at ba6e85e (audit) · found by the 2026-10-25 review · severity medium · page: training
- **Problem**: Training Parameters were single-file and long wide, dense narrow.
- **At ba6e85e**: The base form uses a grid of repeat(auto-fill, minmax(240px, 1fr)) at every width (training_console.py:5004-5006). At 1850x960 the common fields sit two per row (Epochs | Learning rate, Batch size | Seed, Split seed), Device has its own row, and the 'More training settings' and 'Model' accordions are in view at y 610 and 662. At 960 px the fields fall into three columns. 'Parameters' appears once.
- **Sources**: UX:247 (appendix 45); RJ audits[1].findings[12]
- **Closed at**: —

#### RC-A · Mismatched side padding under 'Settings taken from this checkpoint'

- **State**: code-verified at ba6e85e (audit) · found by the owner's recheck · page: training
- **Problem**: The note and the summary table under it had different side padding (owner's screenshot 04_1).
- **At ba6e85e**: 'Settings taken from this checkpoint' is now a disclosure with no box of its own. Its summary, its note ('They cannot be changed when continuing; to train with other settings, choose Branch above.') and its key-value list all span x=45..551 inside the drawer (measured), so they share one side padding.
- **Sources**: UX:133-135; BL:1010 (item 52 (A))
- **Closed at**: —

### UI-09 R10 train/test overlap presentation

2 rows: 1 to do, 1 code-verified at ba6e85e (audit).

#### R-63 · R10 train/test overlap buried in two run-on dot-joined paragraphs

- **State**: to do · found by the 2026-10-25 review · severity high · page: test
- **Problem**: The R10 overlap was two neutral paragraphs.
- **At ba6e85e**: R10 is still two key-value rows. Each holds one run-on sentence joined with ' · ' and keeps the stutter. For example: 'inputs (evidence found): Training disease overlap: overlap — 3 of 5 distinct test diseases; 7 test cases with one of them · Training phenotype-set overlap: overlap — 2 of 10 test cases'. It is plain dark text (computed colour rgb(29,34,48)) with no mark or colour, although 3 of 5 test diseases were seen in training. There is no table. Models' recorded results show the same paragraphs.
- **Done when**: Test and Models show R10 as a table (training/validation x disease/phenotype set) with a status mark on each overlap cell and no repeated 'overlap: overlap' stutter.
- **Sources**: UX:265 (appendix 63); RJ audits[2].findings[2]; UX:68 (section 3 high #7); UX:71-72; UX:127; UX:153-157 (decision 3); BL:1010 (52); SK:63; SK:189-193
- **Closed at**: —

#### B68-R10 · Refused R10 evidence setting shown amber with no mark

- **State**: code-verified at ba6e85e (audit) · found by an internal review · page: test
- **Problem**: A refused R10 evidence setting, which turns Start off, read amber with no mark.
- **At ba6e85e**: A refused evidence setting renders as a red '✕ Refused: …' (class shep-st-error, rgb(165,34,34)). A directory that cannot be checked renders as an amber ▲. A KG edge set renders as '○ Not applicable · …' in grey; I saw that live on my copy with the KG edge set chosen. The shared server sets no evidence directory, so I rendered the refused state from ba6e85e's evidence_html into the live page's CSS.
- **Sources**: BL:1026 (item 68)
- **Closed at**: —

### UI-10 Contrast, type scale, focus and targets from shared tokens

17 rows: 5 to do, 7 partly done, 5 code-verified at ba6e85e (audit).

#### NEW-settings-datasets-footer-3 · Disabled Settings fields look like enabled ones for a read-only viewer

- **State**: to do · found by the 2026-11-16 audit · page: settings
- **At ba6e85e**: For a read-only viewer, the disabled allocator select and the disabled torch.compile checkbox look almost the same as enabled ones; only the dropdown arrow disappears. The 'Operator controls only' reason sits beside the buttons, not the fields. Screenshot (audit evidence, not kept in the repository) Low severity.
- **Done when**: For a read-only viewer, the allocator select and torch.compile checkbox have a visibly distinct disabled style (computed colour/border differs from enabled) and the operator-only reason is shown next to the fields.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### NEW-shell-nav-theme-4 · Diagnosis identity banners and warnings line draw their own ⚠ glyph

- **State**: to do · found by the 2026-11-16 audit · page: diagnosis
- **Duplicate of**: B72-N7 (closed with it)
- **At ba6e85e**: The Diagnosis result's identity banners and warnings line still draw their own ⚠ glyph (src/webui/components/diagnosis_panel.py:1698, 1811), although BACKLOG 72's B5-review note says no page draws a status glyph of its own. Inferred from the code; it needs a diagnosis to see.
- **Done when**: The Diagnosis result banners and warnings line use the shared status mark; grep of diagnosis_panel.py finds no literal ⚠ in rendered HTML.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### R-17+R-56 · Type scale too flat: four sizes, four weights; uppercase style serves several roles

- **State**: to do · found by the 2026-10-25 review · severity medium · page: shell
- **Problem**: Text used 14, 15 and 16 px and four weights, too close to form a hierarchy; one uppercase style served four roles.
- **At ba6e85e**: Every page still uses 14, 15, 16 and 17 px at weights 400, 500, 600 and 700. Training, for example, measures 14/400, 15/400, 14/600, 15/600, 15/500, 15/700, 16/600, 14/500, 17/700 and 14/700. No 20 px region title exists: panel titles are 16/600, the brand 17/700, tabs 15/500 (700 when selected). The 14 px/600 uppercase letter-spaced style still serves three roles: the nav group labels, Training's field labels and Diagnosis' card title (that one at 700).
- **Done when**: Computed styles on every page use only 20/15/14 px and weights 400/600, with a 20 px region title, and the uppercase letter-spaced style serves at most one role; a test checks the set of sizes/weights.
- **Sources**: UX:219 (appendix 17); RJ audits[0].findings[17]; UX:258 (appendix 56); RJ audits[1].findings[22]; UX:47; SK:33-35
- **Closed at**: —

#### R-1c · Diagnosis evidence marks are raw-hex dots of one shape; report uses colour emoji

- **State**: to do · found by the 2026-10-25 review · severity high (finding 1 as a whole); this part low · page: diagnosis
- **Problem**: Review finding 1 recorded that the Evidence column's dots use raw hex outside the tokens, including #d69e00 at 2.4:1. Fix step 4 asked to move them onto the --shep-status-* tokens. At ba6e85e the table and the details still draw one filled dot shape in raw hex for every state (.shep-dx-dot-strong #1f9d55, -weak #d69e00 at 2.40:1 on white, -none #8a909b; diagnosis_panel.py:3558-3561; markup at :1726-1727 and :1882-1883). They are not the shared ●/▲/○ marks. The Markdown report's table and explanation headings use 🟢/🟡/⚪ (LABEL_STYLES, diagnosis_panel.py:220-224), as seen in the diagnosis checker's export (11-export-report.md:40-42). The word sits beside each dot, so colour is not the only cue, but the states are told apart by hue alone.
- **At ba6e85e**: Not in M5's must-fix list (UX:84-85 names only the label) and not in section 4's Later list. The B5 review's 'no page draws a status glyph of its own' (BACKLOG 72 note of 2026-11-14) covered the results badge, not these dots. Unchanged at ba6e85e.
- **Done when**: Evidence marks in the table, details and report use --shep-status-* tokens with distinct shapes per state, no raw hex (e.g. #d69e00) and no colour emoji.
- **Sources**: docs/working/webui-ux-review-2026-10-25/review.json audits[0].findings[0] (evidence: 'Evidence column's coloured dots use raw hex ... #d69e00 at 2.4:1'; fix step 4); docs/working/WEBUI_UX_REVIEW_2026-10-25.md:47 (section 2 layer 1: 'status colours as raw hex'); src/webui/components/diagnosis_panel.py:220-224, 1726-1727, 1882-1883, 3558-3561
- **Closed at**: —

#### R-8+R-51 · Input and select borders nearly invisible (1.0-1.52:1, below 3:1)

- **State**: to do · found by the 2026-10-25 review · severity medium · page: shell
- **Problem**: Text inputs and dropdowns had borders at about 1.05:1 to 1.52:1, below the 3:1 non-text floor, site-wide.
- **At ba6e85e**: No input border token is set (grep 'input-border-color' in src/webui finds nothing), so the theme's #f9fafb applies (1.05:1 on white). Text and number inputs have 0 px borders plus the theme's 0 1px 4px rgba(0,0,0,.1) shadow: Epochs, Learning rate, Seed, HPO terms and Patient ID are white on white (1.0:1). Gradio dropdowns have 0 px borders, a white fill on a #e5e7eb frame (1.24:1). Training's own select wraps and Run name have a 1 px #cdd2da border (1.52:1). All are below the 3:1 floor.
- **Done when**: Every text/number input and dropdown on all pages has a border of at least 3:1 against its background (e.g. #8d94a1), set through --input-border-color and the select wraps, and a contrast test pins it.
- **Sources**: UX:210 (appendix 8); RJ audits[0].findings[10]; UX:253 (appendix 51); RJ audits[1].findings[14]; UX:46 (section 2 layer 1); UX:118 (section 4: 'input borders at 3:1'); BL:1010 (52: 'site-wide label, border, lock and status rework'); SK:150
- **Closed at**: —

#### B72-N7 · Diagnosis identity banners and warnings line draw their own ⚠ glyph

- **State**: partly done · found by an internal review · page: diagnosis
- **Problem**: Datasets, the Diagnosis card, Training lines and the results badge drew their own marks.
- **At ba6e85e**: Datasets, the Diagnosis card, Training's lines and the results badge use the shared mark (renderers.status_mark; renderers.CHECK retired). But the Diagnosis result's identity banners still draw their own ⚠ before each headline, and its warnings line reads '⚠ Warnings: …', where the shared warn shape is ▲. Inferred from the code, not seen: the banners need a diagnosis, which only the diagnosis checker may run.
- **Done when**: The Diagnosis result's identity banners and warnings line use the shared status mark (▲ via renderers.status_mark) instead of ⚠; a test finds no ⚠ in the rendered result HTML.
- **Sources**: BL:1030 (B3, B4, B5 notes)
- **Closed at**: —

#### OWN-1 · Pages still overloaded and hard to read (theme layer unchanged)

- **State**: partly done · found by the owner · severity overall · page: shell
- **Problem**: After M5's functional pass the owner said the pages are still overloaded and hard to read.
- **At ba6e85e**: The round improved the shared pieces: status marks, gated buttons, short options, summary lists, plain field labels and a footer that leads with the consequence. The theme layer that the review (UX §2) blamed for the overload remains: a flat type scale (four sizes and four weights, no 20 px title), invisible input borders (1.52:1 at most), PREPARE/EVALUATE on the tab row, a level-1 switch identical to the in-page radios, and inverted chip emphasis. Pages still carry engineering detail, for example Datasets' footnotes naming scripts/build_knowledge_graph.py, current_data_dir and workspace_build.json. No UX acceptance by the owner has happened since 2026-10-25.
- **Done when**: The theme-layer items R-8, R-15, R-17, R-20, R-21 and R-22 are closed and the owner records UX acceptance of the pages.
- **Sources**: M5W:9-10; M5W:62-67; UX:190-192; PTR:16; BL:1010 (item 52)
- **Closed at**: —

#### R-16+R-76+B72 · Pages still mix field-label styles (uppercase micro-labels beside sentence case)

- **State**: partly done · found by the 2026-10-25 review · severity medium · page: shell
- **Problem**: Field labels were indigo/tinted chips that look like tags, buttons or links; each page mixed 3-4 label styles (R-76: 'the same affordance confusion the owner raised for PREPARE/EVALUATE'; R-46's label half).
- **At ba6e85e**: Gradio field labels are plain 14 px/600 #1d2230 with no fill and no padding on every page; the chips are gone. But each page still mixes label styles. Training's DATASET, RUN NAME and VIEWING are uppercase, letter-spaced 14/600 #434a58, beside sentence-case 'Epochs' at 14/600 #1d2230. Diagnosis has 'MODEL IN USE' (uppercase 14/700) and 'Options' (14/500) beside 'Top-K' (14/600). On Test, 'R10 evidence', 'Batch size' and 'Device' are plain grey keys beside dark 600 labels.
- **Done when**: Each page uses one field-label style: Training's DATASET/RUN NAME/VIEWING and Diagnosis' 'MODEL IN USE' render in the same sentence-case 14-15px/600 as other labels, verified by a computed-style check.
- **Sources**: UX:218 (appendix 16); RJ audits[0].findings[9]; UX:278 (appendix 76); RJ audits[2].findings[15]; RJ audits[1].findings[13] (R-46 label half); UX:45-46; UX:118 (section 4); BL:1030 (item 72 Shell); SK:53-54
- **Closed at**: —

#### R-21 · Read-only chip is quieter than the normal state; header chips look like buttons

- **State**: partly done · found by the 2026-10-25 review · severity medium · page: shell
- **Problem**: In the header the read-only state was quieter than normal and status chips looked like buttons; status colours were raw hex.
- **At ba6e85e**: Done: the lock now appears only on the read-only chip, which leads with 'Read-only:'. Remaining: the normal 'Operator controls: on (this machine)' is still an accent-tinted pill (#eef0ff fill, #3a3dbd text), while 'Read-only: forwarded by a proxy' is a grey pill (#f2f3f5, #2f3540). The state viewers must notice is still quieter than the normal one. Both chips and 'GPU jobs: none' are 28 px bordered pills with a 999 px radius, so they look like buttons while showing normal states. The dot is raw #8a909b.
- **Done when**: In the header, normal states ('Operator controls: on', 'GPU jobs: none') render as plain text without pill borders, while the read-only state is an amber lozenge with the lock; status colours come from tokens, not raw hex.
- **Sources**: UX:223 (appendix 21); RJ audits[0].findings[6]; UX:47; UX:118 ('status marks with the traffic-light vocabulary')
- **Closed at**: —

#### R-31 · Collapsed accordion chevron points left instead of right/down

- **State**: partly done · found by the 2026-10-25 review · severity low · page: settings
- **Problem**: The restart-locked warning failed contrast; a collapsed accordion's chevron pointed left.
- **At ba6e85e**: The contrast half is gone. f9202b6 fixed it, and 0ee56f9 then retired the red restart-locked note. While a GPU job runs, Restart's reason is now the shared neutral mark 'Restart is off while a GPU job (training or a test run) runs.'. I could not see it, because nothing was training and starting a job is not allowed here; this is from the code. The chevron half is unchanged. A collapsed accordion's icon is Gradio's ▼ with inline 'transform: rotate(90deg)', computed matrix(0,1,-1,0), so it points left (◀). Opened, it is rotate(0deg), pointing down. The project's only rule on that icon boxes it in 22x22 px and does not turn it. Seen on both Settings accordions at 1850 and 960 wide. Inferred: the same on every gr.Accordion in the app (e.g. Training's), since the only rule is the shared one.
- **Done when**: Closed Gradio accordions show a right-pointing chevron and open ones a down-pointing chevron on every page.
- **Sources**: UX:233 (appendix 31); RJ audits[0].findings[28]; UX:11; UX:86-87
- **Closed at**: —

#### R-33 · Hits@K chart palette has two colours under 3:1 and overlapping series cannot be isolated

- **State**: partly done · found by the 2026-10-25 review · severity high · page: training
- **Problem**: The Hits@K legend told dashed series apart by colour alone, three of five series overlapped, two palette colours missed 3:1.
- **At ba6e85e**: Legend and tooltip swatches are now inline SVG lines drawn with each series' own dash (shep_charts.js:391-404). This is seen in the Pooled top-k legend: top-1 solid, top-3 '7 4', top-5 '2 3', top-10 '9 3 2 3'. The palette is unchanged (shep_charts.js:74). Against the plot's white background (measured rgb(255,255,255)), #1baf7a is 2.82:1 and #eb6834 is 3.20:1, and top-5 is drawn in #1baf7a with a dotted line. Series that coincide still hide each other: at epoch 1, top-5 at 1.0 sits under top-10, so only three of the four points show. The legend keys are plain spans (no button, role or aria-pressed, and no uPlot setSeries), so a viewer cannot isolate an overlapped series. The tooltip lists every value.
- **Done when**: Every chart series colour measures at least 3:1 against the plot background, and clicking a legend key toggles that series so overlapping lines can be isolated.
- **Sources**: UX:235 (appendix 33); RJ audits[1].findings[1]; UX:64 (section 3 high #3); UX:88; UX:131 (section 4); BL:1010 (52)
- **Closed at**: —

#### R-57 · Run-view 'continued from' relation is an accent pill that looks clickable

- **State**: partly done · found by the 2026-10-25 review · severity low · page: training
- **Problem**: Run-view header tags were ambiguous and looked clickable.
- **At ba6e85e**: The ambiguous 'latest' now reads 'the latest run' or 'a past run' (RELATION_LATEST, training_console.py:2125; 443cfc1). While the Continue drawer is open, the relation is an accent-tinted pill, 'the run continued from' or 'the run branched from' (shep-tag-accent, :2284-2289), in the same tint as links and selected controls. The ordinary 'same dataset as the run bar' is a green chip (status_mark as_chip=True, :2258), although the skill's §6 says normal states are quiet. Neither has a pointer cursor or focus.
- **Done when**: In the run-view header the continued/branched-from relation is plain grey text with an icon, and coloured tags appear only for states needing attention.
- **Sources**: UX:259 (appendix 57); RJ audits[1].findings[23]
- **Closed at**: —

#### OPS-C6 · A chosen radio's label was 4.46:1

- **State**: code-verified at ba6e85e (audit) · found by an internal review · severity P2 · page: shell
- **Problem**: Below 4.5:1 (P2).
- **At ba6e85e**: The chosen radio's label is white on #5558e3 (the dashboard's accent), 5.4:1, measured in the page (rgb(85,88,227)) and computed.
- **Sources**: OPS:363-364
- **Closed at**: —

#### R-2 · Placeholders and info= text used Gradio's subdued grey at 2.54:1

- **State**: code-verified at ba6e85e (audit) · found by the 2026-10-25 review · severity high · page: shell
- **Problem**: Placeholders and info= lines, a refusal among them, used Gradio's subdued grey at 2.54:1.
- **At ba6e85e**: Placeholders and info= lines use the dashboard tokens: --block-info-text-color: var(--shep-text-2) (#434a58) and --input-placeholder-color: var(--shep-muted) (#5b6270). Measured at 1850x960: the Diagnosis placeholders ('HP:0001250 — Seizure…', '(auto-generated if blank)') are 6.13:1 on white, and the Settings info= lines ('Read once when the backend starts…') are 14 px #434a58 at 8.9:1. A contrast test pins both.
- **Sources**: UX:204 (appendix 2); RJ audits[0].findings[1]; UX:63 (section 3, high #2); UX:46-47 (section 2 layer 1); UX:86-87 (must-fix); BL:1010
- **Closed at**: —

#### R-52+B52L · Plain tables used aria-selected; links lacked a focus style

- **State**: code-verified at ba6e85e (audit) · found by the 2026-10-25 review · severity low · page: shell
- **Problem**: Datasets' rows (and the Diagnosis disease table and model table, per 52 L) carried aria-selected on a plain table; links had no focus style.
- **At ba6e85e**: Datasets' selected row carries aria-current="true" and no aria-selected, and on no page does any element outside tab or option roles carry aria-selected. With keyboard focus, 'Details ›' (.shep-link), the icon Copy buttons and the model-list rows show Chromium's default focus ring (outline auto, dark on white), clearly visible; table rows show the 2 px accent outline. There is no explicit :focus-visible rule for .shep-link or button.shep-copy, so their ring is the browser default. Inferred: a browser with a weaker default would show less.
- **Sources**: UX:254 (appendix 52); RJ audits[1].findings[28]; BL:1010 (item 52 (L), K6d2 review)
- **Closed at**: —

#### R-58 · Training labels mixed casing, abbreviations and an emoji

- **State**: code-verified at ba6e85e (audit) · found by the 2026-10-25 review · severity low · page: training
- **Problem**: Training labels mixed casing, abbreviations and an emoji.
- **At ba6e85e**: Form labels are sentence case and full words, from renderers.SETTING_LABELS (renderers.py:1501-1526): Epochs, Learning rate, Batch size, Seed, Split seed, Neighbours per hop, Gradient clip, Mixed precision, Early stopping patience, Epochs per validation. Model fields read 'Input › Dimension' and 'Encoder › Layers'. There is no emoji control (no Randomize seed), and the architecture reads 'GATv2'.
- **Sources**: UX:260 (appendix 58); RJ audits[1].findings[24]
- **Closed at**: —

#### R-83 · role=list without list items and a likely clipped focus ring

- **State**: code-verified at ba6e85e (audit) · found by the 2026-10-25 review · severity low · page: shell
- **Problem**: Reports list used role=list without items; focus ring likely clipped (the 20 px ⓘ half is in R-29+R-83+B72).
- **At ba6e85e**: Every role="list" now holds role="listitem" children, with the heading outside the list (seen in the Diagnosis drawer: two listitem children). The Test reports are now a .shep-tbl table whose rows take a 2 px accent outline at -2 px offset, so the ring is not clipped. The reports table itself could not be seen (the registry is empty); its focus state is read from the code. The 20 px ⓘ half belongs to R-29.
- **Sources**: UX:285 (appendix 83); RJ audits[2].findings[20]
- **Closed at**: —

### UI-11 One action, one operation

4 rows: 1 to do, 3 code-verified.

#### B46-races · Brief older displays on Training/Test pill, banner, Continue drawer and job card

- **State**: to do · found by an internal review · severity non-blocking · page: training
- **Problem**: A tick crossing a render can show an older pill; banner dismissal; Continue drawer can show the previous workspace's list/check; Test job card briefly older.
- **At ba6e85e**: The demo has one workspace and no running job, so the brief older displays cannot be produced (a tick crossing a render on the pill, a banner dismissal, the Continue drawer's previous-workspace list or check, the Test job card). Per the record, the pill is still written by both the fast poll and the renders. The record keeps these as an open, non-blocking follow-up.
- **Done when**: With two workspaces and a running job, a test that crosses a poll tick with a render or switches workspace never shows an older pill, a re-shown dismissed banner, the previous workspace's Continue list/check, or an older Test job card.
- **T02**: not attempted. The pill's two writers (the fast poll and the renders, `POLL_KEYS`) are the Training page's documented render/poll contract; ending the race needs one writer or an ordering stamp there, and the reproduction needs two workspaces and a running job. The select-event fix of RJ-G6 does not change it.
- **Sources**: BL:1004 (item 46)
- **Closed at**: —

#### RJ-G6 · One dropdown pick sends its .input event twice

- **State**: code-verified · found by the 2026-10-25 review · severity unrated (research principle) · page: shell
- **Problem**: Research G6 recorded from Gradio 6.30's source, not yet confirmed in a browser, that an option pick dispatches 'input' and then blur dispatches it again. It named the repo's .input wiring on dropdowns, and R-61's fix asked to 'switch model.input to .change or .select (G6)'. Reproduced now on the shared server by counting queue joins. One pick on Test's Run select sent shep-test-model-run.input twice, and one pick on Training's Viewing select sent shep-train-view-run.input twice (c2-dropdown-events.json). The wiring is unchanged: test_page.py:3289, 3313, 3342, 3344, 3362; training_console.py:4651, 4657; finetune_page.py:1316; model_list.py:1781.
- **At ba6e85e**: Recorded only in review.json's research. No later record or test names it. Reproduced at ba6e85e.
- **Done when**: A browser test counting queue joins shows exactly one request per option pick on Test's Run select and every other dropdown wired with .input.
- **Cause (measured in T02 on a bare Gradio 6.30 page)**: a single-select Dropdown sends `input` on every blur and once more for an option clicked (2 per pointer pick, 1 per keyboard pick, 1 for a focus that chose nothing). Its `select` is not sent for a pick made with the arrow keys and Enter, and `change` follows the server's own writes too (Test writes its selects back on purpose), so neither is the choice's event; its `blur` comes once per option chosen and never from a write. A multiselect and a Radio send `input` once per change.
- **Fix**: `shell.picked(control)`, the one place that names a choice's event (a single-select Dropdown's blur, any other control's input), wired at every select choice: the shared list's workspace, Models' Task, Test's workspace, version, checkpoint and run, Training's dataset and Viewing, Fine-tune's dataset, Diagnosis' reference cohort, the Results segment's filters. No debounce; two real picks are two requests.
- **Sources**: docs/working/webui-ux-review-2026-10-25/review.json research[1].principles[5] (G6-dropdown-events); docs/working/webui-ux-review-2026-10-25/review.json audits[2].findings[0] (fix: 'Also switch model.input to .change or .select (G6)')
- **Closed at**: d2d02f6. Tests: `test_webui_shell.py::test_every_select_choice_in_the_dashboard_is_its_one_event_per_pick` (the whole dashboard: no single-select Dropdown choice on `input`; the named selects on blur), `test_webui_shell.py::test_one_pick_of_a_select_is_one_request_and_the_pages_write_is_none` (browser, Gradio's own page: pointer 1, keys 1, the server's write 0, a Radio pick 1), and the Models browser test above (one Task pick by pointer and by keys: 1 request each; 2 for the pointer pick on ba6e85e). Seen in the dashboard on a CPU demo workspace, counting queue joins: Test's Run select and Training's Viewing select, 1 request per pointer pick and 1 per keyboard pick (2 and 1 on ba6e85e). The training touch record (Textbox, Number, Checkbox, Radio) is unchanged.

#### OPS-R1 · 'Set by you' can name what Reset cleared in a key burst; pressed dropdown pick not held

- **State**: code-verified · found by the 2026-10-25 review · severity P2 · page: training
- **Problem**: A field emptied as the first edit was not recorded, so Start sent the reference; then a click could beat the edit's record.
- **At ba6e85e**: Emptying Neighbours per hop as the first edit (Ctrl+A, then Backspace) is recorded: the line reads 'Set by you: Epochs 3 (reference 200) / Not valid: Neighbours per hop needs at least 1 item(s).', and Reset clears it (seen). The held-click synchronisation (TOUCH_HEAD, LANDED_JS) was not exercised.
- **Done when**: A unit/browser test that types two keys and presses Reset in one burst ends with a 'Set by you' line naming nothing Reset cleared, and a dropdown option chosen by a press is recorded before an immediate Start click sends the form.
- **T02**: no code change; both cases are browser tests of the R1 fix that the audit did not run. The forms' choice fields are Radios (the touch record holds no Dropdown), so "an option chosen by a press" is the radio case.
- **Sources**: OPS:367; OPS:368; OPS:371; OPS:372
- **Closed at**: the R1 fix before ba6e85e (146cfca, 2810853, 30ba727), rerun at e72d4a0: `test_training_touch_record.py::test_a_late_line_answer_never_records_the_field_again_after_reset` (the Training page as mounted: two keys with the line held, Reset, the line let go: "Set by you: nothing", Start sends no setting) and `::test_a_click_made_while_an_edit_is_pending_waits_for_its_record` (a radio's option pressed and Start pressed at once: Start held until the edit is recorded, then one request with the key and the value); the file's 11 tests pass, and the OPS-R2 null tests (`test_settings_projection.py`, `test_training_fields.py`, `test_webui_renderers.py`, `test_webui_finetune_page.py`, `test_training_console_resume.py`) pass.

#### R-59 · Five Training listeners still flash Gradio progress timers

- **State**: code-verified · found by the 2026-10-25 review · severity low · page: training
- **Problem**: Progress timers flashed on several Training handlers.
- **At ba6e85e**: Two of the seven listeners left with their controls (randomize_seed, conv_type). Five still have no show_progress='hidden': continue_source.change twice (training_console.py:4752, :4760), checkpoint_dropdown.change (:4762), manual_checkpoint.change (:4764) and continue_device.change (:4766). In the browser, visible Gradio progress text ('processing | 0.1/0.1s', 'queue: 1/1 | 0.1/0.1s') appeared on continue_run, checkpoint_dropdown, the parent details, manual_checkpoint and continue_preview when switching the checkpoint source, changing the drawer's Device or choosing a checkpoint.
- **Done when**: continue_source (both), checkpoint_dropdown, manual_checkpoint and continue_device .change listeners have show_progress='hidden', checked by a unit test over the listener config, and no timer flashes on those changes.
- **Sources**: UX:261 (appendix 59); RJ audits[1].findings[25]
- **Closed at**: e72d4a0. Test: `test_training_page.py::test_no_listener_of_the_continue_drawers_controls_shows_a_progress_timer` (every listener of the drawer's source, run, checkpoint, path, Device and operation; on ba6e85e it names the five). Seen in the browser on a CPU demo workspace, a mutation observer counting visible Gradio progress text in the drawer while the source was switched, a path typed, a listed checkpoint chosen and Device changed twice: 54 visible timer texts on ba6e85e (on continue_preview, continue_run, checkpoint_dropdown, the parent's details, manual_checkpoint), none now.

### UI-12 What bound and verified mean

3 rows: 2 to do, 1 code-verified at ba6e85e (audit).

#### B72-T3 · Tampered pairs.bin or split record still listed as 'Bound', refused only at Start

- **State**: to do · found by an internal review · page: test
- **Problem**: list_versions reads cohort.json alone.
- **At ba6e85e**: list_versions still reads cohort.json and nothing else. Reproduced on my isolated copy: I changed one byte of demo-kg-1's pairs.bin. The version still read '● Bound to this graph' and Start was enabled with no reason. The run then failed at start: 'measure_scorer: refused: …pairs.bin is not the pairs file … cohort.json binds (its bytes are c35c7d4a53b1..., the version binds 28e2b12861ba...)'. The table got a grey 'Failed' row. (I restored the byte afterwards.)
- **Done when**: After changing one byte of a version's pairs.bin (or contradicting its split record), the version list no longer reads 'Bound to this graph' and Start is disabled with a reason naming the mismatch.
- **Sources**: BL:1030
- **Closed at**: —

#### NEW-models-results-5 · 'metadata not readable' shown for entries that were never read

- **State**: to do · found by the 2026-11-16 audit · page: models
- **At ba6e85e**: The visible words 'metadata not readable · cannot be loaded' also appear for entries that were never read, such as a directory or run whose name is not text ('bad\xff · RUNS'), so the words claim a read that did not happen (renderers.py:1241). Evidence: (audit evidence, not kept in the repository)
- **Done when**: An entry whose name is not valid text (e.g. 'bad\xff') reads a wording that does not claim a read happened; 'metadata not readable' appears only after a failed read, checked by unit test.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### B72-T2 · Version mapped against another graph shown as amber chip without mark

- **State**: code-verified at ba6e85e (audit) · found by an internal review · page: test
- **Problem**: Not shown as a refusal with its mark.
- **At ba6e85e**: A version mapped against another graph shows, in its Graph row, an amber '▲ Mapped against another graph (kg differ)' (rgb(122,70,0)). Under the version comes a red '✕ Not measured on this workspace: the version was mapped against another graph, so a run would refuse it. …'. Start's reason points there: 'Disabled: the data version is mapped against another graph (see Data).' I rendered this from ba6e85e's version_html into the live page's CSS; the demo has one graph.
- **Sources**: BL:1030
- **Closed at**: —

### UI-13 Path redaction and path exposure

5 rows: 4 to do, 1 code-verified at ba6e85e (audit).

#### B53 · Open read routes name server paths the pages hide

- **State**: to do · found by an internal review · page: api
- **Problem**: pipeline/status and training/status return absolute paths, so the pages' redaction can be read around (API, not a page).
- **At ba6e85e**: Reproduced for a read-only request (X-Forwarded-For): GET /api/v1/training/status answers gpu.lock.path as the absolute server path (…/scratchpad/audit-ui/ws/.shepherd-gpu.lock). GET /api/v1/pipeline/status carries no path while no model is loaded; its current_data_dir and current_checkpoint_path need a served model to show.
- **Done when**: Per the owner's decision, GET /api/v1/training/status and /api/v1/pipeline/status for a read-only request return no absolute server path (gpu.lock.path, current_data_dir, current_checkpoint_path redacted at the service boundary); an API test pins it.
- **Sources**: BL:1011 (item 53); OPS:364
- **Closed at**: —

#### B72-E1 · Export download link exposes Gradio's /tmp cache path

- **State**: to do · found by an internal review · page: results
- **Problem**: The download link shows /ui/gradio_api/file=/tmp/gradio/….
- **At ba6e85e**: On my private copy, Export (All shown, CSV) gave a link with href http://127.0.0.1:8871/ui/gradio_api/file=/tmp/gradio/bbcde121…/results_workspace-demo_20261116T103647Z.csv. The visible name and the download name carry no path, but the link does. The file is written by exports.write_new into a mkdtemp directory and served by gr.File (results_segment.py:1334-1345, 1506).
- **Done when**: After the owner's D7 decision, the Export link's href carries no server temp path (or the decision to keep it is recorded).
- **Sources**: BL:1030; OPS:172 (U3)
- **Closed at**: —

#### B72-N5 · Path outside every root with a bracketed spaced part is only partly redacted

- **State**: to do · found by an internal review · severity minor · page: shell
- **Problem**: '/data/Jane (MRN 1)/x.json' shows its bracket on.
- **At ba6e85e**: Reproduced: workspaces.redact('Cannot read /data/Jane (MRN 1)/x.json: denied', root=/srv/workspaces) returns 'Cannot read a path outside the workspaces root (MRN 1)a path outside the workspaces root denied'. The bracketed part of the name is still shown.
- **Done when**: workspaces.redact('Cannot read /data/Jane (MRN 1)/x.json: denied', root=...) returns text with no part of the path, including '(MRN 1)'; a unit test pins it.
- **Sources**: BL:1030 (B0c review note)
- **Closed at**: —

#### NEW-shell-nav-theme-3 · Path redaction outside the root swallows the following ': '

- **State**: to do · found by the 2026-11-16 audit · page: shell
- **At ba6e85e**: Path redaction swallows the ': ' after a path outside the workspaces root: workspaces.redact('Cannot read /data/other/x.json: denied', /srv/workspaces) gives 'Cannot read a path outside the workspaces root denied', a run-on sentence. Inside the root the colon stays ('Cannot read demo/x.json: denied'). Seen through src/webui/components/workspaces.py:53 and src/utils/path_redaction.py:70; separate from B72-N5's bracket case.
- **Done when**: A unit test shows redact('Cannot read /data/other/x.json: denied', root) keeps the colon, e.g. 'Cannot read a path outside the workspaces root: denied'.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### OPS-C8 · A Training/Fine-tune printed line could name a server path

- **State**: code-verified at ba6e85e (audit) · found by an internal review · page: training
- **Problem**: Training/Fine-tune printed lines redacted only for some requests.
- **At ba6e85e**: No absolute server path appears in the Training page's text or title attributes (run details opened, Continue checked on cpu) or on Fine-tune. Paths are shown relative: 'demo/runs/gatv2/…/checkpoints/last.pt' and 'gatv2/…/last.pt'.
- **Sources**: OPS:363-364
- **Closed at**: —

### UI-14 Reload restores the page and valid selections

4 rows: 3 to do, 1 outside this round's scope (owner to decide).

#### B32-UI8 · Every open page lands on Diagnosis after a reconnect reload

- **State**: to do · found by an internal review · page: shell
- **Problem**: After a restart every page lands on Diagnosis; the chart view returns only when Training is opened (M5-3 review UI-8); also per-page links, pages by role, New/Continue merged recorded.
- **At ba6e85e**: Reproduced in a reload (the page-level part of a reconnect reload; the service was not restarted): with Research › Models open, the page came back on Diagnosis (clinical). The shell builds level 1 with selected=CLINICAL, and no page is kept in sessionStorage or in the address; the head script writes only __theme=light.
- **Done when**: With Research › Models open, a page reload (and a reconnect reload) comes back on Research › Models, with the outer and research page kept in sessionStorage or the address.
- **Sources**: BL:990 (item 32)
- **Closed at**: —

#### B34-a · Gradio SSE session loss shows 'Connection Lost' and reloads (upstream)

- **State**: to do · found by other · severity upstream · page: shell
- **Problem**: gradio#13895: a dropped stream reloads the page.
- **At ba6e85e**: Gradio is still pinned at 6.30.0, and the dashboard adds nothing for a dropped /queue/data stream. Not reproduced: dropping the shared server's SSE stream would disturb the other checkers.
- **Done when**: Against the pinned Gradio 6.30.0, dropping the /queue/data stream in an isolated instance is rechecked before release and the result (page reloads or recovers, and on which page) is recorded.
- **Sources**: BL:992 (item 34 (a))
- **Closed at**: —

#### B46-mount · Diagnosis tab mount writes the session location whatever page is shown

- **State**: to do · found by an internal review · page: shell
- **Problem**: About 1.5 s after load the location becomes Diagnosis, so a page opened before polls nothing.
- **At ba6e85e**: The code is unchanged: Diagnosis' tab select writes the session's location with no guard. Observed: on load, with no click, the page sends _shown_clinical (dependency 474) at about 1.4 s, along with Diagnosis' card refresh (dependency 16). The race itself was not reproduced in two tries: when Research and then Models were clicked at about 1.37 s, before that select, the Diagnosis select did not fire at all. Inferred to remain possible within a narrower timing window.
- **Done when**: A Diagnosis tab select fired at mount does not overwrite the session location when another page is already shown; a test opening Models before that select sees Models' polling continue.
- **Sources**: BL:1004 (item 46, end)
- **Closed at**: —

#### B32-links · No per-page links (?stage=/?sub=) and no pages by role

- **State**: outside this round's scope (owner to decide) · found by an internal review · page: shell
- **Problem**: BACKLOG 32 records, besides the reconnect landing (B32-UI8), two more follow-ups. Per-page links ('?stage=' / '?sub=', never '?page=') are not built. Showing or hiding pages by role is tied to item 28 and the institution's access control. At ba6e85e the shell reads no address parameter and keeps no page in sessionStorage (grep of shell.py, app.py and style.py finds no location.search, URLSearchParams or sessionStorage).
- **At ba6e85e**: Recorded 2026-10-19, not started.
- **Why this state**: Per-page links and role-based pages are listed as deferred new product surfaces in tasks_def.md (plan §1 backlog); the reconnect landing part (B32-UI8) is handled under UI-14.
- **Sources**: docs/working/BACKLOG.md:990 (item 32); docs/working/PLAN_TEST_RESULTS.md:2387-2389
- **Closed at**: —

### UI-15 Diagnosis input changes and result identity

3 rows: 3 to do.

#### B73 · Genes box edited without a keypress keeps the old diagnosis result

- **State**: to do · found by an internal review · page: diagnosis
- **Problem**: Select-all+Delete, cut or paste changes the genes while the old result stays (read from Gradio's code; not reproduced).
- **At ba6e85e**: Not reproduced; this is from the code. The genes box still clears the result only on .input (genes_input.input(**edited)), so an edit with no keypress keeps the old result. The HPO box uses .change. The demo serves no gene task model, so the genes box never appears here.
- **Done when**: A unit/UI test shows that changing the genes box by paste, cut or select-all+Delete (a .change event) clears the shown result, as the HPO box already does.
- **Sources**: BL:1031 (item 73); OPS:371
- **Closed at**: —

#### NEW-diagnosis-5 · A typed HPO name is dropped silently; report lists unused ids as used

- **State**: to do · found by the 2026-11-16 audit · page: diagnosis
- **Duplicate of**: R-5 (closed with it)
- **At ba6e85e**: A name typed in the HPO box ('Hypotonia') is dropped without a word on the page, in the warnings or in the report. The report's 'Query phenotypes' lists the duplicate and the unknown id as if they were used (11-export-report.md line 10). This adds to R-5 and R-6.
- **Done when**: Entering 'Hypotonia', a duplicate and an unknown id shows each under 'Not used' with a reason on the page, and the exported report's query phenotypes lists only ids actually used.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### R-5 · No echo of which HPO ids were recognised or used

- **State**: to do · found by the 2026-10-25 review · severity medium · page: diagnosis
- **Problem**: Nothing showed which HPO ids were recognised from the input or used by the run.
- **At ba6e85e**: I typed 'HP:0001250 Seizure, HP:0001263, Hypotonia, HP:9999999, HP:0001250'. Before Run, nothing under the box says which ids were recognised. After Run, the only echo is the server's warnings line: 'Unknown phenotype: HP:9999999; Removed 1 repeated phenotype entry; each phenotype is scored once.' That line already existed at f9202b6. The name 'Hypotonia' was dropped without a word. There is no 'As used' list with names. The report's 'Query phenotypes' lists the ids as parsed, the duplicate and the unknown id included, with no used / not-used split. The CSV has no such column.
- **Done when**: Under the HPO box a live line says how many ids were recognised and which tokens are not ids; after a run an 'As used' row lists terms with names and a 'Not used (n)' list gives a reason per id; the CSV and report carry the same split.
- **Sources**: UX:207 (appendix 5); RJ audits[0].findings[12]; UX:123 (section 4 Diagnosis: 'the input echo')
- **Closed at**: —

### UI-16 Every other recorded problem, by page

116 rows: 53 to do, 24 partly done, 1 awaiting hardware, 3 outside this round's scope (owner to decide), 4 not applicable, 31 code-verified at ba6e85e (audit).

**UI-16, datasets**

#### B51-h · Datasets details header repeats the selected row; run-bar meaning said three times

- **State**: to do · found by an internal review · page: datasets
- **Problem**: The selected row's name, record and origin repeated in the details header; the footnote repeated 'in your Training run bar'.
- **At ba6e85e**: Both repeats are still on screen at 1850x960. The details header reads 'demo ● record verified synthetic' right beside the selected table row 'demo | synthetic | ● Verified'. The meaning of 'in your Training run bar' is given three times: the In use mark, the footnote ('“In your Training run bar” is this browser tab's selection; nothing on the server changes.') and the note beside 'Use for training ›' ('Selects it in your Training run bar (this browser tab) … Nothing changes on the server'). 'Training run bar' was counted 3 times in the page's visible text.
- **Done when**: The selected dataset's name/record/origin appear once (the header refers to it by relation or the row does not repeat it), and the meaning of 'in your Training run bar' is stated once on the page.
- **Sources**: BL:1009 (item 51 (h))
- **Closed at**: —

#### R-30 · First panel title repeats the selected tab name (Datasets, Models)

- **State**: to do · found by the 2026-10-25 review · severity low · page: datasets
- **Problem**: Each Research page's first panel title repeated the selected tab's name.
- **At ba6e85e**: Datasets' first panel is titled 'Datasets', directly under the selected 'Datasets' tab, and Models' is titled 'Models' under 'Models'. Training ('Parameters') and Test ('Data') do not repeat; Fine-tune ('Fine-tune a base') half-repeats.
- **Done when**: The first panel on Datasets and Models is titled by its content and count (e.g. 'Workspaces (1)', 'Checkpoints (4 in 2 runs)') and no longer repeats the selected tab name.
- **Sources**: UX:232 (appendix 30); RJ audits[0].findings[30]
- **Closed at**: —

#### R-35 · Datasets details lead with full hashes and pack several facts per row in a nested box

- **State**: to do · found by the 2026-10-25 review · severity medium · page: datasets
- **Problem**: Datasets' details put manifest hashes first and packed several facts per row.
- **At ba6e85e**: The details panel is as the finding described. The full 64-hex fingerprint comes first ('Dataset fingerprint · SHA-256 of split_manifest.json' plus 64 hex plus Copy). Under it is a bordered card ('.shep-card', 1px solid rgb(225,228,234)) inside the panel, 'What the manifest binds sha256, first 12 · full values in the manifest', with seven 12-hex digests under snake_case keys (kg, node_features, edge_indices, num_nodes, train_samples, val_samples, record). Then two rows join several facts with dots: 'Split: allocation seed 42 · feature seed 20260909' and 'Samples: train 200 · val 50 · diseases (train / val) 3 / 1' (two separators, where the skill §5 allows one). 'Path: demo (in the server's workspace root)' repeats the name. The comparisons are still long sentences in code; they cannot show here with one workspace.
- **Done when**: The details panel shows the fingerprint as 12 hex with Copy, one fact per row, manifest contents behind a 'Manifest contents (N files)' disclosure with plain labels and no inner card, and no Path row when it equals the name.
- **Sources**: UX:237 (appendix 35); RJ audits[1].findings[17]; UX:129 (section 4)
- **Closed at**: —

#### R-36 · Datasets table rows are tall and 'In use' looks like an input box

- **State**: to do · found by the 2026-10-25 review · severity medium · page: datasets
- **Problem**: Datasets' rows were tall and the 'In use' mark an input-like white box.
- **At ba6e85e**: Rows are still tall. At 1850x960 the demo row is 75 px (the skill asks for 32-40), and at 960x1080 it is 95 px. 'Built' still wraps 'UTC' onto a second line ('2026-11-16 09:13' / 'UTC'), because the nowrap span covers only the date. Train / val is still stacked ('200' over '50'). The Record cell splits its mark from its word ('●' / 'Verified'). The 'In use' mark 'in your Training run bar' is still a white box with a 1 px rgb(205,210,218) border and 6 px radius, the look of a text field (its border is 1.52:1 against white). It is stacked under 'serving reads this workspace'. At 1850 px or more, Origin and Record are still separate columns; below 1800 they fold into the name cell, as before.
- **Done when**: At 1850x960 a dataset row is at most 40 px tall: 'Built (UTC)' date on one line, train/val on one line with tabular figures, 'In use' as a tag or icon text (no white box), Origin/Record merged or moved to details.
- **Sources**: UX:238 (appendix 36); RJ audits[1].findings[18]
- **Closed at**: —

#### R-34 · Datasets intro defines fingerprint with file names; footnote and USE_NOTE expose internals

- **State**: partly done · found by the 2026-10-25 review · severity medium · page: datasets
- **Problem**: Datasets' intro and footnote exposed internals; its 'read-only' wording collided with the D8 role.
- **At ba6e85e**: The 'read-only' collision is fixed: the header now reads '1 workspace on the server' (b1678f8). The footnote's internal names are now after a trailing 'Ref:'. The rest is as before. The intro still defines the fingerprint with a file name: 'Dataset fingerprint: the SHA-256 of each workspace's split manifest (split_manifest.json)'. The details panel says it again ('Dataset fingerprint · SHA-256 of split_manifest.json'), so SHA-256 and split_manifest each appear twice on screen. The footnote is still a three-line visible paragraph under the table, not an ⓘ. It explains markers and says 'A build time appears once a build records one, outside the fingerprint. Ref: the pipeline status's current_data_dir; workspace_build.json.' USE_NOTE still ends 'Nothing changes on the server; Start is the gated act.'
- **Done when**: The Datasets intro has no fingerprint definition or file names, the footnote is an info bubble on the 'In use' header without file/field names, and USE_NOTE no longer says 'Start is the gated act'.
- **Sources**: UX:236 (appendix 34); RJ audits[1].findings[16]
- **Closed at**: —

#### R-53 · Datasets draws origin, record and Sources status in different forms on one screen

- **State**: partly done · found by the 2026-10-25 review · severity low · page: datasets
- **Problem**: The same status was drawn three ways on Datasets and Training.
- **At ba6e85e**: Done: the status glyphs are one shared mark (● ○ ▲ ✕ through renderers.status_mark), and renderers.CHECK has retired. Remaining: Datasets still draws the same states in different forms on one screen. 'synthetic' is an amber chip in the table and plain muted text in the details header. The record is a plain '● Verified' mark in the table, and a '● record verified' green chip in the details header and on Training, Models and Fine-tune. Sources reads '—' in the table and 'none (synthetic)' in the details.
- **Done when**: On Datasets, origin, record and Sources each render in one form across the table, the details header and the run bar, and a known 'none (synthetic)' Sources is never shown as '—'.
- **Sources**: UX:255 (appendix 53); RJ audits[1].findings[27]
- **Closed at**: —

#### B29 · Dataset building and management not available in the WebUI

- **State**: outside this round's scope (owner to decide) · found by the owner · page: datasets
- **Problem**: The owner placed dataset management and building under Prepare (2026-10-19). Building is still command-line only: the Datasets page is read-only and its BUILD_LINE points to scripts/build_knowledge_graph.py --generate-samples and scripts/setup_demo.py (datasets_page.py:79-83). It needs the job runner, the GPU lock and D8, so it is a new product surface.
- **At ba6e85e**: Recorded 2026-10-19, not started, scheduled 'Later'.
- **Why this state**: Dataset building in the WebUI is listed as a deferred new product surface in tasks_def.md (needs job runner, GPU lock and D8).
- **Sources**: docs/working/BACKLOG.md:987 (item 29); docs/working/BACKLOG.md:183 (§0 'Later, on their own schedule'); docs/working/PLAN_TEST_RESULTS.md:2387-2389; docs/working/ops-wiring-and-ui-ux/README.md §3 point 8 ('Not in this round: ... a data builder'); src/webui/components/datasets_page.py:79-83
- **Closed at**: —

**UI-16, diagnosis**

#### B56 · Diagnosis keeps mock HPO search/disease routes and never cleans export temp files

- **State**: to do · found by an internal review · page: diagnosis
- **Problem**: Ranked summary dropped; mock-only HPO search and disease detail routes; keeping a diagnosis not built; export temp files not cleaned.
- **At ba6e85e**: Still in place: - the HPO search route answers mock data (src/api/routes/search.py:118-195); - the disease detail routes answer mock data (src/api/routes/disease.py:137-204, mock tables :208-350); - export files go to a mkdtemp folder that is never cleaned ('periodic cleanup is future work', src/webui/exports.py:11, :30); - keeping a diagnosis is not built (inferred: no route). K7 removed the no-op pipeline fields.
- **Done when**: The HPO search and disease detail routes either answer real data or are removed/refuse with a clear message (no mock tables served), and export temp folders are deleted after download or on a cleanup schedule, verified by a unit test.
- **Sources**: BL:1014 (item 56)
- **Closed at**: —

#### B62 · Patients-like-me rows show reference patient details against D7's aggregates-only rule

- **State**: to do · found by an internal review · severity decision · page: diagnosis
- **Problem**: Patients-like-me rows show reference patients' ids, terms, diagnoses; D7 says aggregates only for an institutional cohort.
- **At ba6e85e**: From the code: a patients-like-me result's rows show Research ID, Similarity, Diagnosis and Causal genes, and its details list every recorded HPO term. D7 is still undecided. The demo cannot show it: there is no patients-like-me model and the registry is empty.
- **Done when**: The owner's D7 decision is recorded, and the patients-like-me result follows it (e.g. only aggregates for an institutional cohort), verified by a renderer unit test on a sample result.
- **Sources**: BL:1020 (item 62)
- **Closed at**: —

#### NEW-diagnosis-1 · Explanation view adds a third wording of the device notice

- **State**: to do · found by the 2026-11-16 audit · page: diagnosis
- **Duplicate of**: R-26 (closed with it)
- **At ba6e85e**: The Explanation view ends with its own disclaimer ('Note: This is a computational prediction for clinical review. Final diagnosis should be made by qualified healthcare professionals…') while the results footer's device line is on screen. That is a third wording of the device notice, which R-26 does not name ((audit evidence, not kept in the repository)
- **Done when**: With a result's Explanation view open, the device notice appears only once on screen (the footer line); the Explanation text has no own disclaimer.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### NEW-diagnosis-2 · Evidence view states the shortest path twice

- **State**: to do · found by the 2026-11-16 audit · page: diagnosis
- **At ba6e85e**: The Evidence view says the shortest path twice: once in the summary quote ('Direct knowledge-graph evidence found (shortest path = 1 hop).') and again in 'Shortest: 1 hop · 5 direct paths' (diagnosis_panel.py:1880-1892; screenshot 09-details-evidence-1850.png in the same verify-diagnosis directory).
- **Done when**: The Evidence view states the shortest path length once (either the summary or the 'Shortest' line).
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### NEW-diagnosis-4 · BACKLOG 51 still lists (g) Diagnosis repeats as open after eed93d6 fixed it

- **State**: to do · found by the 2026-11-16 audit · page: diagnosis
- **At ba6e85e**: BACKLOG 51 (BL:1009) still lists (g), the Diagnosis repeats, as open. The fix in eed93d6 has no note there, so the record lags the code.
- **Done when**: BACKLOG 51's (g) item carries a dated note citing eed93d6 as the fix.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### NEW-help-2 · 'Model in use' help grew to 152 words and its bubble covers Run Diagnosis

- **State**: to do · found by the 2026-11-16 audit · page: diagnosis
- **Duplicate of**: R-9 (closed with it)
- **At ba6e85e**: The ops round made the 'Model in use' help longer instead of shorter. A3b 7be6fe1 raised CARD_HELP (diagnosis_card.py:150-162) from 128 to 153 words (152 now) as one paragraph, with its repeated clause kept. That goes against the skill's §7 budget and PLAN_TEST_RESULTS's 'Long explanations go into the details area'. Its pinned bubble is 320 x 551 px and covers the card and Run Diagnosis. Evidence: (audit evidence, not kept in the repository) No record mentions the growth.
- **Done when**: CARD_HELP is at most about 80 words with a lead line and no repeated clause, and its pinned bubble does not cover the card's Run Diagnosis button at 1850x960.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### NEW-shell-nav-theme-5 · Load drawer rows chain three ' · ' and repeat the identity line

- **State**: to do · found by the 2026-11-16 audit · page: diagnosis
- **Duplicate of**: R-7+B51a (closed with it)
- **At ba6e85e**: The Diagnosis 'Load for inference' drawer's rows have three ' · ' each ('model fingerprint … · trained on demo · dataset fingerprint … · synthetic', renderers.model_row_html at renderers.py:1112-1124), against skill §5's 'at most one ' · ' in a row', and repeat the identity line that B4 placed above them (diagnosis-load-drawer-1850x960.png).
- **Done when**: In the Diagnosis Load drawer, rows trained on the chosen workspace show at most one ' · ' and do not repeat trained-on, dataset fingerprint or synthetic stated above them.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### R-1d · Diagnosis table has no phenotype-match count (k/N)

- **State**: to do · found by the 2026-10-25 review · severity high (finding 1 as a whole; this part was rated with it) · page: diagnosis
- **Problem**: Review finding 1 recorded that the candidate table 'still has no phenotype match count'. Its fix step 2 asked for Rank | Disease | Score | Phenotypes k/N | Evidence, with k/N computed in the renderer from the query and matching_phenotypes and no API change. The Diagnosis audit's overall verdict lists 'a k/N phenotype column' among the structural changes, and research DX-01 asks the same. Section 3 #4 records it as excluded from M5 ('M5; no phenotype column; the repeat Later'), but section 4's Later list for Diagnosis does not carry it and neither does BACKLOG 52, so it fell out of every plan. At ba6e85e the disease table is still Rank, Disease, Score, Evidence (diagnosis_panel.py:1731-1733). The matching phenotypes appear only in the selected candidate's details segment.
- **At ba6e85e**: Left out of M5 on purpose (UX:65) and never put into section 4's Later list, BACKLOG 52 or any ops-round stage. Nothing since. Still absent at ba6e85e.
- **Done when**: The Diagnosis candidate table has a Phenotypes k/N column computed in the renderer from the query and matching_phenotypes, pinned by a unit test.
- **Sources**: docs/working/webui-ux-review-2026-10-25/review.json audits[0].findings[0] (evidence: 'still no phenotype match count'; fix step 2); docs/working/webui-ux-review-2026-10-25/review.json audits[0].overall ('Diagnosis table: ... a k/N phenotype column'); docs/working/webui-ux-review-2026-10-25/review.json research[3].principles[0] (DX-01); docs/working/WEBUI_UX_REVIEW_2026-10-25.md:65 (section 3 #4: 'M5; no phenotype column'); docs/working/WEBUI_UX_REVIEW_2026-10-25.md:123 (section 4 Diagnosis Later list, which omits it); src/webui/components/diagnosis_panel.py:1731-1733
- **Closed at**: —

#### R-23 · Inference milliseconds and a ticking 'checked' clock in the clinician view

- **State**: to do · found by the 2026-10-25 review · severity low · page: diagnosis
- **Problem**: The clinician's view showed inference milliseconds and a ticking 'checked …' clock.
- **At ba6e85e**: The results header still shows the milliseconds: '5 candidates · 63.1 ms · produced 09:50:54 UTC', which also has two ' · '. The card shows 'checked HH:MM:SS UTC' while it is current, and it changes on every poll. I read 'checked 09:48:12 UTC' and, 12 s later, 'checked 09:48:27 UTC'.
- **Done when**: The results header shows no ms (kept in exports/report), and the card shows 'checked …' only when not current (or a static mark with the time in its title), so the visible text does not change on each poll.
- **Sources**: UX:225 (appendix 23); RJ audits[0].findings[27]
- **Closed at**: —

#### R-26 · Device notice twice with drawer open, all-caps banners, 'synthetic' many times

- **State**: to do · found by the 2026-10-25 review · severity low · page: diagnosis
- **Problem**: With the identity drawer open the device notice showed twice, banners were all caps, 'synthetic' three times.
- **At ba6e85e**: With the identity drawer open, the device line is on screen twice: in the results footer, and in the drawer ('2 Uses · Not a certified medical device…' in Model in use, or a bold line in This result). The banner is still all caps, 'SYNTHETIC WORKSPACE: not a clinical result.', both on the results and in the drawer's Warnings. 'synthetic' is still said in many places at once: the card chip 'synthetic data', the results banner, the drawer's Warnings banner, 'origin synthetic' and 'Synthetic workspaces are for testing only'. The page text has 6 matches with Model in use open and 4 with This result open.
- **Done when**: With the identity drawer open the device line appears only in the footer (and exports), banner headlines are sentence case, and the card's 'synthetic data' chip is hidden while a result banner from the same served identity is shown.
- **Sources**: UX:228 (appendix 26); RJ audits[0].findings[23]
- **Closed at**: —

#### R-3 · Evidence uses 'Mode A' jargon; phenotypes and paths show bare ids

- **State**: to do · found by the 2026-10-25 review · severity medium · page: diagnosis
- **Problem**: Candidate evidence was headed with internal Mode A/B jargon; phenotypes and paths showed bare ids before names.
- **At ba6e85e**: Evidence view: 'Strong path support', the summary in a quote, then the heading 'Mode A — Direct path evidence' and the line 'Shortest: 1 hop · 5 direct paths, listed under Paths ›', then supporting-gene chips. 'Mode A' is the name of an old-pipeline mode retired in R2 (BACKLOG 659). Matching phenotypes view: a bare 'HP:0001263' and the 34-word MATCHING_NOTE. Paths view: bare ids only, e.g. 'HP:0001250 → CDKL5 → MONDO:0012091'. The names are available: the same candidate's Explanation view says 'Global developmental delay (HP:0001263)', and kg.json holds every node's name.
- **Done when**: Candidate evidence shows key-value rows without 'Mode' wording with the narrative folded; phenotypes and path steps show term names first with ids muted; matching phenotypes are split annotated/not annotated with counts; paths show the 3 shortest as named chips with the rest behind a disclosure.
- **Sources**: UX:205 (appendix 3); RJ audits[0].findings[14]; UX:123-124 (section 4 Diagnosis: evidence as key-value rows, terms and paths name first)
- **Closed at**: —

#### R-6 · HPO input rule only in placeholder/bubble; placeholder implies names are read

- **State**: to do · found by the 2026-10-25 review · severity medium · page: diagnosis
- **Problem**: The rule that only HP:####### ids are read lived only in a placeholder and a bubble; the placeholder implied names are read.
- **At ba6e85e**: The placeholder is unchanged: 'HP:0001250 — Seizure / HP:0001263, HP:0002376 (commas / spaces are fine too)'. It still suggests that names are read, and it disappears once typing starts. There is no visible info line under the field. The rule 'Names and stray punctuation are ignored; only HP:####### ids are used' appears only in the info bubble. Patient ID still uses the placeholder '(auto-generated if blank)' and has no info line.
- **Done when**: A visible info line under HPO terms says 'Only HP:####### ids are read; names and punctuation are ignored.', the placeholder reads 'HP:0001250, HP:0001263', and Patient ID has a visible info line.
- **Sources**: UX:208 (appendix 6); RJ audits[0].findings[11]
- **Closed at**: —

#### OWN-SP · Gene table shows component scores by default; no opt-in score facets

- **State**: partly done · found by the owner · page: diagnosis
- **Problem**: At the recheck the owner asked why the table still showed an SP column.
- **At ba6e85e**: The disease table reads Rank, Disease, Score, Evidence: there is no SP column. That is because a disease task model serves no SP (K8-4, f4a8c0f). No facet list exists, so no other score can be added as an opt-in column or filter. Inferred from code (the demo has no gene model): a gene task model's table shows 'Model similarity' and 'SP similarity' columns by default beside Score.
- **Done when**: The gene task table shows Rank, Gene, Score, Evidence by default, with Model similarity and SP similarity available only as opt-in columns/filters from a facet list.
- **Sources**: UX:142-147; BL:1010 (item 52, 'Not item 52's')
- **Closed at**: —

#### R-1b · Selected candidate's score repeats in the details header

- **State**: partly done · found by the owner · severity high (finding 1 as a whole) · page: diagnosis
- **Problem**: The details header repeated the row's scores ('Confidence · GNN · SP'); the same numbers twice for one candidate.
- **At ba6e85e**: The details header still repeats the selected row's score. Row 1 shows -2.844 in the table, and the header beside it reads 'MONDO:0012091 Score -2.844'. GNN and SP left the header only because the mixture model and its columns were retired in K8-4 (f4a8c0f). That was a scoring change, not a UI fix. At f9202b6 the header read 'Score <b>…</b> · GNN … · SP …'.
- **Done when**: details_header_html shows only '#rank name ID' (also for gene and patients-like-me headers), and a test checks each score string occurs once outside the drawer and exports.
- **Sources**: UX:203 (appendix 1); RJ audits[0].findings[0]; UX:65 ('the repeat Later'); UX:123 (section 4 Diagnosis: 'the scores said once'); RJ research[3].principles[4] (DX-05)
- **Closed at**: —

#### R-28 · 'Model in use' drawer still nests collapsible sections (two levels of disclosure)

- **State**: partly done · found by the 2026-10-25 review · severity low · page: diagnosis
- **Problem**: The identity drawer held accordion sections inside a drawer.
- **At ba6e85e**: Fixed: 'This result' is one summary list (What produced this result: Checkpoint, Knowledge graph, …, each with its mark and value). Still open: 'Model in use' still shows Model Facts as numbered collapsible sections. '1 Identity' is open; '2 Uses', '3 Data' and '4 Evidence' are collapsed; '5 Warnings' follows. That is still an accordion inside a drawer, so most facts are a second click away. The Load drawer's facts use the same renderer.
- **Done when**: In the identity drawer and the Load drawer, Model Facts sections are always open (no collapsible sections), with Warnings first when present.
- **Sources**: UX:230 (appendix 28); RJ audits[0].findings[25]; UX:124 (section 4)
- **Closed at**: —

#### B48 · Training and inference exclusion on the GPU not verified

- **State**: awaiting hardware · found by the owner · severity requirement · page: diagnosis
- **Problem**: Owner's provisional requirement: while training runs, Diagnosis and inference are locked; while a pipeline is loaded, Training is locked.
- **At ba6e85e**: Seen here: the card has '■ Stop inference' with a two-step question, and the header chip reads 'GPU jobs: none'. CPU serving takes no lock: /api/v1/pipeline/status returned serving:'serving' and holds_gpu_lock:false. That matches the owner's 2026-11-12 choice to lock the GPU, not the pages. Not seen, from the code: on CUDA, serving holds the one GPU lock and GPU jobs are refused with a message naming inference (training_manager.py:1072 serving_refusal; chips.py:126).
- **Why this state**: Serving holds the GPU lock only on CUDA; checking that training and inference exclude each other needs a CUDA device (GB10 acceptance W12 not run).
- **Done when**: On a CUDA device, while serving, a GPU job is refused with a message naming inference and the header chip shows it, and while training, loading for inference is refused likewise.
- **Sources**: BL:1006 (item 48); OPS:62-64 (section 3 point 1)
- **Closed at**: —

#### B30-usability · No clinician usability check of the Diagnosis model card

- **State**: outside this round's scope (owner to decide) · found by an internal review · page: diagnosis
- **Problem**: BACKLOG 30 records, for the owner, 'a short usability check of the Diagnosis tab's model card with three to five clinicians', together with the release designation. No record shows it was done. The WebUI acceptances so far are functional, and usability is explicitly not accepted (PTR head, M5W §4).
- **At ba6e85e**: Recorded 2026-10-19, not started (§0 'Later').
- **Why this state**: The clinician usability study is listed as a deferred item in tasks_def.md (plan §1 backlog).
- **Sources**: docs/working/BACKLOG.md:988 (item 30); docs/working/m5-webui-acceptance-2026-10-25/README.md §4
- **Closed at**: —

#### B17-F5 · Diagnosis fields showed saved UI preferences, not the loaded pipeline

- **State**: not applicable · found by other · severity defect · page: diagnosis
- **Problem**: The tab showed one data directory while another was served.
- **At ba6e85e**: The Data Directory and Checkpoint fields no longer exist. The card shows what the server's status reports, and a load is chosen from the server's listing. src/webui no longer reads or writes .shepherd_ui_config.json or /pipeline/config (I10).
- **Why this state**: The Data Directory and Checkpoint fields were removed; the card shows the server's status and src/webui no longer reads .shepherd_ui_config.json or /pipeline/config (I10).
- **Sources**: BL:975 (item 17 F5, N1 homelab acceptance)
- **Closed at**: —

#### B17-F6 · matching_phenotypes always empty in exports

- **State**: code-verified at ba6e85e (audit) · found by other · severity defect · page: diagnosis
- **Problem**: Empty column in CSV/report on the real path.
- **At ba6e85e**: Exports now carry the matching phenotypes: the CSV's matching_phenotypes is 'HP:0001263' for rank 1. The report has a 'Matching Phenotypes' section per candidate, and the page shows 'Matching phenotypes 1'.
- **Sources**: BL:975 (item 17 F6)
- **Closed at**: —

#### B27-4 · The served identity in one place

- **State**: code-verified at ba6e85e (audit) · found by the owner · severity requirement · page: diagnosis
- **Problem**: Fingerprints appeared twice, in the pipeline status and each result's 'What produced this result'.
- **At ba6e85e**: The served identity is one block, always shown: the 'Model in use' card with state, name, model fingerprint, trained-on and dataset fingerprint, and the synthetic chip. Each result carries a badge with the shared status mark ('● Produced by the model in use ›'; ▲ for an earlier model, ○ for not compared). The badge opens 'This result', which marks the parts that differ 'In use now'. The CSV's served_* columns and the report's 'What produced this result' keep the full identity.
- **Sources**: BL:985 (item 27 (4)); PTR:2422-2436
- **Closed at**: —

#### B61 · One quantity named 'model term' and 'GNN score'

- **State**: code-verified at ba6e85e (audit) · found by an internal review · page: diagnosis
- **Problem**: Two names for one figure.
- **At ba6e85e**: src/webui names neither 'GNN score' nor 'model term' (grep), apart from the refusal code MODEL_TERM_UNAVAILABLE. The table says 'Score'.
- **Sources**: BL:1019 (item 61)
- **Closed at**: —

#### OPS-B4 · Diagnosis card showed the whole not-serving reason

- **State**: code-verified at ba6e85e (audit) · found by an internal review · page: diagnosis
- **Problem**: Since A3b the card showed the whole reason (environment settings, a route) and its own 'Loading <12 hex> for every session…' line.
- **At ba6e85e**: With no model, the card shows '○ Not loaded', the lead 'Nothing has been loaded since the service started', and 'Details ›' for the full reason of about 50 words. While serving it shows '● Serving'. The page's own 'Loading <12 hex> for every session…' line is gone.
- **Sources**: OPS:213-217 (B4, review of A3b)
- **Closed at**: —

#### OPS-C12 · A refused reload was not written to the service log

- **State**: code-verified at ba6e85e (audit) · found by an internal review · severity P2 · page: diagnosis
- **Problem**: One of the two P2 findings of the ops close browser acceptance (15 findings): a refused reload left no line in the service log. It was fixed in the close ('a refused reload logged once with why'). The inventory carries the other close fixes (OPS-C1 to C10, and Refresh inside R-18) but not this one.
- **At ba6e85e**: Fixed in the close (18518a6..500a8d6, OPS:364). Not rechecked here (it needs a refused Load, which only the diagnosis checker may do).
- **Sources**: docs/working/ops-wiring-and-ui-ux/README.md:363 (close, browser acceptance: 'a refused reload was not written to the service log'); docs/working/ops-wiring-and-ui-ux/README.md:364 (close, fixes)
- **Closed at**: —

#### OPS-C5+K7 · Load confirmation From side read as if a model were served

- **State**: code-verified at ba6e85e (audit) · found by an internal review · page: diagnosis
- **Problem**: With nothing served the load plan's From side read as if something were (K7 review, again at the ops close).
- **At ba6e85e**: With nothing served, the confirmation's From column reads 'no model loaded' for Model and 'not applicable' for the fingerprint, trained-on, workspace, task, SP table and data rows.
- **Sources**: OPS:363-364; SR:1108-1112 (K7 P3)
- **Closed at**: —

#### R-1a · Uncalibrated score was labelled 'Confidence'

- **State**: code-verified at ba6e85e (audit) · found by the 2026-10-25 review · severity high · page: diagnosis
- **Problem**: The Diagnosis table, details header and report called an uncalibrated ranking score 'Confidence'.
- **At ba6e85e**: Nothing on the page or in the report says 'Confidence' any more. The table header reads 'Score' with an info bubble: 'Score ranks the candidates of this result. It is not a calibrated disease probability.' The kind line above the table says 'ranked by its Score (uncalibrated)'. The details header and the exported report also say 'Score', and the report carries the same sentence.
- **Sources**: UX:203 (appendix 1); RJ audits[0].findings[0]; UX:65 (section 3, high #4); UX:84-85 (section 4 must-fix); BL:1010 (item 52); SK:83
- **Closed at**: —

#### R-4+B51g · 'No model loaded' appeared four times and 'no result yet' twice

- **State**: code-verified at ba6e85e (audit) · found by the 2026-10-25 review · severity medium · page: diagnosis
- **Problem**: On an empty Diagnosis page 'No model loaded' appeared four times and 'no result yet' twice; Run Diagnosis' reason repeated the card's 'no model loaded'.
- **At ba6e85e**: Empty page at 1850 x 960: - the card says '○ Not loaded', then 'Nothing has been loaded since the service started' and 'Details ›'; - Run Diagnosis' reason is 'Disabled until a model is loaded.', the audit's own wording; - the right side of the results header is empty; - the body says 'No result yet' once. Counts in the page text: 'No model loaded' 0, 'Not loaded' 1, 'No result yet' 1.
- **Sources**: UX:206 (appendix 4); RJ audits[0].findings[13]; UX:50 (section 2 layer 2); BL:1009 (item 51 (g))
- **Closed at**: —

#### RC-E · Explanation said 'Confidence: HIGH (score ...)'

- **State**: code-verified at ba6e85e (audit) · found by the owner's recheck · severity high (misleads) · page: diagnosis
- **Problem**: The explanation and summary headed each candidate 'Confidence: HIGH / MEDIUM / LOW (score …)'.
- **At ba6e85e**: The Explanation view says 'Score: -2.844 (a ranking score, not a calibrated disease probability)'. There is no 'Confidence' and no HIGH/MEDIUM/LOW band. The report says the same.
- **Sources**: UX:90-94; BL:1010 (item 52); M5W:22
- **Closed at**: —

**UI-16, finetune**

#### NEW-training-finetune-6 · Emptied task setting reads 'must be set, not null'

- **State**: to do · found by the 2026-11-16 audit · page: finetune
- **Duplicate of**: OPS-R2 (closed with it)
- **At ba6e85e**: Fine-tune: an emptied task setting reads 'Not valid: Task loss weight must be set, not null.' ('null' is a programming word on the page; finetune-form-checks.json)
- **Done when**: Emptying a Fine-tune task setting shows a refusal without the word 'null' (e.g. 'Task loss weight must be set.'), and the reason also shows beside Start.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### NEW-training-finetune-7 · Idle Fine-tune panel shows a disabled Stop with no reason

- **State**: to do · found by the 2026-11-16 audit · page: finetune
- **At ba6e85e**: Fine-tune job panel while idle: a visible, disabled '■ Stop' with no reason (ft_stop disabled, ft_stop_reason empty) beside 'No Fine-tune job since the server started…', the same pattern as R-39 (paths-and-ft-stop.json, finetune-1850x960.png)
- **Done when**: With no Fine-tune job, the job panel shows no Stop button, or the disabled Stop has a visible reason.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### OPS-R2 · Explicit null task setting passed and page said reference would be used

- **State**: code-verified at ba6e85e (audit) · found by the 2026-10-25 review · severity P2 · page: finetune
- **Problem**: Null loss weight or sample size passed; Fine-tune said the reference would be used; the child failed.
- **At ba6e85e**: With Disease ticked, emptying Task loss weight gives 'Set by you: nothing valid. / Not valid: Task loss weight must be set, not null.'. The page never says the task's reference would be used (seen). It passed the narrow re-review (OPS:371).
- **Sources**: OPS:367; OPS:369
- **Closed at**: —

**UI-16, help**

#### R-69 · R10 help bubble is a 155-word single paragraph that repeats itself

- **State**: to do · found by the 2026-10-25 review · severity medium · page: help
- **Problem**: The R10 help text was a 155-word paragraph.
- **At ba6e85e**: test-r10 (test_page.py:384-388) and models-r10 (models_page.py:241-245) are still the generated `" · ".join(f"{name}: {definition}")` of the four R10_LABEL_TEXT definitions. The served text is 155 words in one paragraph with no sentence break, and the training and validation definitions repeat each other. I added my own ⓘ to my browser's copy of the Test page, pointing at the served text. The bubble rendered 320 x 616 px, at 1850 x 960 and at 960 x 1080. The real ⓘs are on a result's 'R10, training' row (test_page.py:2225) and on Models' recorded results (models_page.py:1038). Neither shows in the demo, whose registry is empty.
- **Done when**: test-r10 and models-r10 are a hand-written text of about 55 words: a lead line and one definition per line, keeping each label's meaning; a test checks the word count.
- **Sources**: UX:271 (appendix 69); RJ audits[2].findings[4]; UX:68; UX:121
- **Closed at**: —

#### R-10+R-43+B52K · Heading ⓘ bubbles can cover the region they explain

- **State**: partly done · found by the 2026-10-25 review · severity medium · page: help
- **Problem**: The bubble opened over the element it explains (HPO terms, Epochs) and was 360 px wide against a ~320 px budget.
- **At ba6e85e**: Field help is fixed. A field's ⓘ takes the field's rectangle from its elem_id (help.py:263-266; d675d17) and places the bubble beside, then above, then below the field (help.py:311-317; f2bf458). Measured at 1850 x 960: the HPO terms bubble sits at x 379-699 beside its field (x 33-369), the Epochs bubble at x 300-620 beside its field (x 33-290), and Device beside its field, all with 0 px² overlap. At 960 x 1080 HPO terms and Epochs also have 0 overlap. Width is fixed: every bubble measured 320 px (BUBBLE_MAX_WIDTH, help.py:80). A topic ⓘ in a region's heading is not fixed. Its protected area is only the line it sits in (help.py:277, `closest('… label, div')`). For 'Model in use' that is the 138 x 22 px title (33,442 to 171,464). The pinned bubble (181,401 to 501,952) covers 55% of the card it explains ('Not loaded', 'Nothing has been loaded', Details ›) and 39% of Run Diagnosis and its 'Disabled until…' reason. At 960 x 1080 it covers 35% of the card and 40% of Run Diagnosis. This is the same case the review cited (02-diagnosis-help-bubble.png). The R10 ⓘ sits in a result row and needs a report, so it cannot be seen in the demo. By the code its key-value row is protected (help.py:268-276).
- **Done when**: A ⓘ in a region heading (the Diagnosis card first) places its bubble outside the region it explains (closest region container), with 0 overlap measured in a browser test.
- **Sources**: UX:212 (appendix 10); RJ audits[0].findings[4]; UX:245 (appendix 43); RJ audits[1].findings[10]; UX:51-52; UX:121-122; BL:1010 (item 52 (K), K6 closing smoke); SK:118
- **Closed at**: —

#### R-29+R-83+B72 · 'Esc to close' footer on hover previews; card ↻ below 24x24

- **State**: partly done · found by the 2026-10-25 review · severity low · page: help
- **Problem**: Pinned bubble kept tooltip semantics, hover opened instantly, every bubble had an 'Esc to close' footer, ⓘ targets were 20 px (also R-51, R-83, BACKLOG 72).
- **At ba6e85e**: Done: a pinned bubble is role=dialog with aria-labelledby on its title, and a hover or focus preview is role=tooltip (help.py:346-351; measured on HPO terms, Epochs, Device and dx-card). A hover opens after 300 ms (HOVER_OPEN_MS, help.py:81, :368-373): hidden at 120 ms, shown at 500 ms. Keyboard focus opens at once. Every visible ⓘ on Diagnosis and Training measured 24 x 24 (HIT_TARGET, help.py:82, :539-547), the header chips' ⓘ included. Remaining: every bubble, hover previews included, still has the 'Esc to close' footer (help.py:247-249, CSS :580). It measured 32 px tall on each bubble I opened. The card's ↻ (`.shep-dx-reread`, diagnosis_panel.py:3022-3024, CSS :3483 `padding: 0 4px`) measures 23.3 x 16 px, under the 24 x 24 floor.
- **Done when**: The 'Esc to close' footer appears only on a pinned bubble (or is dropped), and the Diagnosis card's ↻ measures at least 24x24 px.
- **Sources**: UX:231 (appendix 29); RJ audits[0].findings[26]; RJ audits[1].findings[14] (R-51 ⓘ half); RJ audits[2].findings[20] (R-83 ⓘ half); BL:1030 (item 72 Shell)
- **Closed at**: —

#### R-74 · Test and Models help texts exceed the budget and cite internal codes (D5)

- **State**: partly done · found by the 2026-10-25 review · severity medium · page: help
- **Problem**: Test and Models help texts exceeded the budget and leaned on internal codes.
- **At ba6e85e**: Some of what the finding named left with the features it described. Ordering by test score retired (3ed3938, K8-6d), so models-ordering is 26 words (it was 101). models-regime, SHEPHERD_ORDERING_ROLES, '(D4)' and PRE_B1_NOTE are gone (K8-6, e.g. 6992597). The rest is as found or longer. test-preview-id is unchanged: 57 words with a 34-word sentence (test_page.py:375). shep-test-record ends with an unexplained '(D5)' and has a 29-word sentence (:325). test-fixed also cites '(D5)' (:389). test-evidence is 73 words with 'R10' and SHEPHERD_TRAINING_EVIDENCE_DIRS (:396). shep-test-workspace opens with a 30-word sentence naming SHEPHERD_WORKSPACES_ROOT (:283), and shep-test-version and shep-test-intake name SHEPHERD_REGISTRY and SHEPHERD_INTAKE_DIR. models-results is 88 words (results_segment.py:160), models-contested 70 words in one paragraph (models_page.py:225), shep-test-deterministic 61 words in one paragraph (test_page.py:332), and shep-test-model and shep-test-measurement 79 words each. Across all texts, 16 still carry internal codes.
- **Done when**: Remaining Test and Models help texts are one answer sentence plus at most 3 bullets, about 60 words, no sentence over 25 words, with decision codes and setting names only in a trailing 'Ref:' line; a test over these texts pins it.
- **Sources**: UX:276 (appendix 74); RJ audits[2].findings[5]; UX:121
- **Closed at**: —

#### R-9 · Help bubbles are long plain-text paragraphs with no bold lead or lists

- **State**: partly done · found by the 2026-10-25 review · severity medium · page: help
- **Problem**: Help bubbles were 90-190-word run-on paragraphs with no lead or emphasis; the bubble renders plain text only.
- **At ba6e85e**: The one bubble still fills its body with plain text: `bd.textContent = textOf(btn)` (help.py:360) under `white-space: pre-line` (help.py:578). Nothing can be bold, and there is no lead or list markup. Every bubble I opened measured bodyChildren 0 and font-weight 400. I read all 131 registered texts from the live page's hidden container at ba6e85e (the review counted 78). 90 have line breaks, 18 use a typed '•' as plain text, and the Training settings share one generated shape (setting_help, training_console.py:365-385). Against the budget: 41 texts are still one paragraph (21 of them over 40 words), 8 run over 80 words and 21 have a sentence over 25 words. The review's own examples are unchanged or longer. 'Model in use' (CARD_HELP, diagnosis_card.py:150-162) is one 152-word paragraph. It was 128 words at the round base 30e8ae1, and A3b 7be6fe1 made it 153. It still says 'after each diagnosis, after a model change and with ↻' twice, and the pinned bubble is 320 x 551 px. The D8 chip text is 93 words (bubble 320 x 507 px) and R10 is 155 words. No test checks the budget across the registry; only a few single texts have their own test (test_diagnosis_panel.py:1621, :2313; test_webui_test_page.py:4145-4159).
- **Done when**: The shared bubble renders a bold lead line, '-'/'•' lines as a real list and bold 'Term:' using createElement/textContent; the longest texts are cut to about 60-80 words; a test over every registered text checks the budget.
- **Sources**: UX:211 (appendix 9); RJ audits[0].findings[3]; UX:48-52 (section 2 layers 2, 3); UX:121-122 (section 4: all 78 texts rewritten; the structured renderer); BL:1010 (52: 'the 78 help texts'); SK:114-123
- **Closed at**: —

#### B27-3 · Help behind an 'i' instead of text under labels

- **State**: code-verified at ba6e85e (audit) · found by the owner · severity requirement · page: help
- **Problem**: Help texts sat under labels (43 info= texts at 1fe0ea9).
- **At ba6e85e**: Every ordinary help text sits behind one shared ⓘ (help.py: one registry, one hidden container, one head script; 131 texts). It opens on hover after 300 ms, on focus and on click. info= keeps only short format and must-know lines: on Diagnosis, Datasets, Training, Fine-tune, Test and Models no info= line shows at load, and Settings shows two short ones (12 and 17 words). A test guards it: test_no_component_keeps_ordinary_help_in_info.
- **Sources**: BL:985 (item 27 (3)); PTR:2407-2421
- **Closed at**: —

**UI-16, models**

#### B40 · Investigation notes shown only on Test, not on Models or loaded-model route

- **State**: to do · found by an internal review · page: models
- **Problem**: Contested records on Models lack their notes.
- **At ba6e85e**: Investigation notes are read only by the Test page (test_page.py:670-684, read_investigation_notes). Models' contest_html (models_page.py:736-778) shows 'runs under these conditions differ — investigate…' with no note. Nothing under src/api reads notes for /api/v1/evaluations/loaded-model (grep finds no caller). The demo has no contested group, so this was not seen on screen.
- **Done when**: A contested record on Models and in /api/v1/evaluations/loaded-model shows its investigation notes, or 'notes cannot be read'.
- **Sources**: BL:998 (item 40); BL:1382
- **Closed at**: —

#### B69-L2 · Automatic selection reason counts directories as 'file(s)'

- **State**: to do · found by an internal review · severity cosmetic · page: models
- **Problem**: Counts 'file(s)' when directories are among the unread items.
- **At ba6e85e**: Reproduced with the shared function that the reload route uses (pipeline.py:745-751). With five unlistable directories under runs/, select_auto_run_checkpoint's reason ends '…; and 2 more file(s) not read', although the two are directories (checkpoint_paths.py:1443, with_not_read).
- **Done when**: With unlistable directories under runs/, select_auto_run_checkpoint's reason counts 'items' (or directories and files apart), pinned by a unit test.
- **Sources**: BL:1027
- **Closed at**: —

#### NEW-models-results-1 · Recorded results join metrics and environment facts in long ' · ' chains

- **State**: to do · found by the 2026-11-16 audit · page: models
- **At ba6e85e**: Models › Recorded results joins facts with ' · ' instead of rows (models_page.py:913-918, 950-955); skill §5 allows at most one. Two chains: about 24 metrics ('Disease coverage 0.8000 · Disease conservative pooled MRR 0.3250 · … · Disease macro precision@100 0.0100') and 'Environment: CPU … · Dataset: synthetic · Registered #2 at … from run …, report …'. Seen on my private ba6e85e copy: (audit evidence, not kept in the repository) BACKLOG 72 recorded only the 'Relations:' chain, which B3 fixed.
- **Done when**: Models › Recorded results shows metrics as rows/table and environment, dataset and registration as separate rows, with at most one ' · ' per line.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### NEW-models-results-3 · Each recorded result says its dataset origin three times

- **State**: to do · found by the 2026-11-16 audit · page: models
- **At ba6e85e**: Each record in the recorded-results block says its dataset origin three times: 'Dataset of every record below: synthetic …', a 'synthetic' chip in the cohort cell, and 'Dataset: synthetic' in the environment line. With two records, 'synthetic' appears 6 times on one Models screen, against 1 when no records are shown (rule 1.2). Evidence: (audit evidence, not kept in the repository)
- **Done when**: With two recorded results on Models, 'synthetic' appears once for the block (no per-record chip and environment repeat).
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### NEW-models-results-4 · Unreadable checkpoint's reason exists only as a hover title

- **State**: to do · found by the 2026-11-16 audit · page: models
- **At ba6e85e**: On Models, an unreadable checkpoint's reason ('the file cannot be read as a checkpoint; Ref: …', or the not-valid-text reason) exists only as a hover title on a row marked aria-disabled (renderers.py:1239-1242). The row cannot be selected, so the detail branch that would show the reason (models_page.py:1395-1399) is never reached. Skill §6/§7 say hover-only text is not enough for keyboard, touch or screen-reader users. Evidence: (audit evidence, not kept in the repository)
- **Done when**: An unreadable checkpoint row shows its reason as visible text (or the row is focusable and its detail shows the reason), reachable by keyboard and touch.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### NEW-test-2 · Models' recorded results chain ~24 metrics and environment facts with ' · '

- **State**: to do · found by the 2026-11-16 audit · page: models
- **Duplicate of**: NEW-models-results-1 (closed with it)
- **At ba6e85e**: Models' recorded results put the other metrics in one ' · ' chain of about 24 facts, each starting with 'Disease' ('Disease coverage 0.8000 · Disease conservative pooled MRR 0.3250 · … · Disease macro precision@100 0.0100'; models_page.py:947). The next line chains 'Environment: … · Dataset: … · Registered #2 at … from run …, report …' (models_page.py:950-951). The skill (§5) allows at most one ' · '. No record names either. Seen: (audit evidence, not kept in the repository)
- **Done when**: Recorded results show metrics and environment/dataset/registration facts as rows or a table with at most one ' · ' per line.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### R-68 · Empty recorded-results state is a dead end naming a file; segments show '· 0' twice

- **State**: to do · found by the 2026-10-25 review · severity medium · page: models
- **Problem**: Models' empty recorded-results state was a dead end naming a file.
- **At ba6e85e**: Selecting a checkpoint with no ledger shows 'No ledger (evaluations.json) is beside this checkpoint, so no recorded result can be shown; that does not mean these weights were never evaluated.' (models_page.py:1437-1440). With a ledger that holds nothing for the file it shows 'This ledger holds no result for these weights; …' (RECORD_ABSENCE). Neither has a title, a next step or a 'Test this model ›' action. The segments 'Recorded results · 0 | Usage history · 0 | Results' still show with both counts at 0 (models_page.py:1413-1419). The table cell says 'no ledger' or 'none in ledger' (renderers.py:694-700).
- **Done when**: Each empty cause shows a titled state with one next step ('Test this model ›' or the read-only refusal), segments are hidden when both counts are 0, and no text says 'ledger'.
- **Sources**: UX:270 (appendix 68); RJ audits[2].findings[13]; UX:129
- **Closed at**: —

#### R-78 · Models Load note is jargon and shows before anything is selected

- **State**: to do · found by the 2026-10-25 review · severity low · page: models
- **Problem**: Models' Load note was jargon and showed before a selection.
- **At ba6e85e**: The Load footer reads 'Reloads the pipeline and asks once, from → to, as Change model does. Results already produced keep their own identity.' It shows before any row is selected, and after a selection the text is unchanged. The note is a constant (models_page.py:194-195 LOAD_NOTE), passed as the view's note in every state (:1250, :1256).
- **Done when**: With no row selected the Load footer shows no note; with a loadable row selected it shows a plain-words note without 'pipeline' or 'from → to'.
- **Sources**: UX:280 (appendix 78); RJ audits[2].findings[22]
- **Closed at**: —

#### B72-N3 · Some digests still build their own code element outside fingerprint_value

- **State**: partly done · found by an internal review · page: models
- **Problem**: Digests drawn several ways.
- **At ba6e85e**: The cells the item names use the one digest value, renderers.fingerprint_value (:1461-1485), with class shep-fpv and the full value as title. These are: the Models Fingerprint column (renderers.py:1248), the Results and Compare cells (results_segment._cell_html :1132), the By family Lineage (models_page.py:1159) and the Datasets table (datasets_page.py:246). Other digests still build their own <code>: Datasets details' 'What the manifest binds' and the dataset fingerprint block (renderers.py:521-529), data_kind_lines' 'Split manifest' (renderers.py:1932, shown on Models and Test records), the load plan From/To cells (model_list.py:1154, 1170), and Diagnosis (diagnosis_panel.py:1803, 2774).
- **Done when**: A grep/unit test finds every digest in src/webui rendered through renderers.fingerprint_value (class shep-fpv with full title).
- **Sources**: BL:1030 (B0c note)
- **Closed at**: —

#### R-18+R-40+R-75+B68+B72 · Models list header still says 'operator controls only' to the operator

- **State**: partly done · found by the 2026-10-25 review · severity medium · page: models
- **Problem**: A lock was drawn on every gated button (Diagnosis, Training, Test, Models) even when the operator may use it.
- **At ba6e85e**: The button half holds: there is no 🔒 on any enabled operator button, and the lock appears only on refused ones. Finding 75 has a second half, 'an operator-only header read as locked for the operator'. Its fix: drop the lock and the phrase, or say 'Listing opens checkpoint files'. Only the lock went. For an operator on loopback, whose header chip reads 'Operator controls: on (this machine)', Models' list header still reads 'listed by the server (operator controls only)' (models_page.py:1278-1279). Seen at 1850 x 960: (audit evidence, not kept in the repository) and c1.json (modelsHeader). The read-only form still shows the code line '🔒 Checkpoint list: operator controls only (D8 · SPEC_4 C3)' (models_page.py:206).
- **Done when**: For an operator on loopback, the Models list header no longer reads 'listed by the server (operator controls only)' (drop the phrase or say 'Listing opens checkpoint files'); locks remain only on refused buttons.
- **Sources**: UX:220 (appendix 18); RJ audits[0].findings[8]; UX:242 (appendix 40); RJ audits[1].findings[4]; UX:277 (appendix 75); RJ audits[2].findings[14]; UX:54; UX:118 ('a lock only on refused buttons'); BL:1026 (item 68); BL:1030 (item 72 Shell); OPS:364 (close fix 7220957)
- **Closed at**: —

#### R-66+B51f · Models records repeat the workspace dataset's origin per record

- **State**: partly done · found by the 2026-10-25 review · severity medium · page: models
- **Problem**: On Models the dataset identity appeared four times; rows repeated names and fingerprint.
- **At ba6e85e**: The verdict's counts (the dataset fingerprint once, 'synthetic' once) were taken on the shared demo, whose registry is empty, so no recorded result was on screen. The table column (model 12 hex only) and the details' Dataset row are fixed. But with recorded results the Models details repeat the workspace dataset's origin again. The records block opens 'Dataset of every record below: synthetic — a generated stand-in…', and then each record's line repeats '… · Dataset: synthetic · Registered #n …'. With two records, the workspace dataset's 'synthetic' therefore appears four times on one screen, counting the workspace line's chip. That is the same count finding 66 made. Source: the models-results checker's private ba6e85e copy ((audit evidence, not kept in the repository) lines 9-10, 31, 60, 87) and the test checker's copy (.../out/verify-test/own-models-row0.txt). The records' header already says every record shares the dataset, so the per-record repeat is not a place where the two can differ (rule 27(5)).
- **Done when**: With two recorded results, Models details say the shared dataset ('synthetic') once in the records header and no record line repeats 'Dataset: …' when it matches.
- **Sources**: UX:268 (appendix 66); RJ audits[2].findings[11]; UX:50; BL:1009 (item 51 (f), first half)
- **Closed at**: —

#### R-67 · Models details repeat the 'in use' header when nothing is served

- **State**: partly done · found by the 2026-10-25 review · severity medium · page: models
- **Problem**: Models' details mixed keys, paths and repeated labels.
- **At ba6e85e**: The detail is now one summary list in a fixed order: Fingerprint with Copy, File, Dataset, Kind, Operation, Parent, Family, Encoder, Relation to the paper, then Validation. Validation is a plain row ('pooled MRR (Trainer) 0.3133 — selection metric on val, not an independent test'), not a bordered card. The run directory and the whole fingerprint sit behind 'Details ›' (models_page.py:1307-1369). The header still says one fact twice when no model is served: a grey 'not in use' chip and, on the right, 'nothing is in use for diagnosis'. This is because detail_parts adds _served_line whenever the row is not in use (models_page.py:1388-1390, 1292-1305).
- **Done when**: When no model is served, the Models details show no right-hand 'in use' text; it appears only when another model is in use.
- **Sources**: UX:269 (appendix 67); RJ audits[2].findings[12]; UX:129
- **Closed at**: —

#### R-82 · Same report record rendered by two code paths on Test and Models

- **State**: partly done · found by the 2026-10-25 review · severity low · page: models
- **Problem**: Test and Models rendered the same record two ways.
- **At ba6e85e**: Some pieces are shared now: the one reader (record_views.result_view), relation rows (renderers.relation_rows :1810) and the R10 text (renderers.r10_rows :1867). The record itself is still rendered by two code paths: Test's _record_html (test_page.py:2094-2230) and Models' _mode_table (models_page.py:888-966). They show the same record in different forms and orders. Test gives task rows ('Pooled top-k k=3 0.3750, …', 'Macro recall@k …'), a Coverage row, an 'R10, training' key, and Environment, Native build, Dataset and Registration as separate rows. Models gives a table row, then one line chaining 24 metrics with ' · ' ('Disease coverage 0.8000 · Disease conservative pooled MRR 0.3250 · … · Disease macro precision@100 0.0100'), a bold 'R10, training inputs (evidence found): …' line, and 'Environment: … · Dataset: synthetic · Registered #2 at … from run …, report …'.
- **Done when**: One shared renderer produces the record's rows, and a unit test shows Test and Models emit the same rows in the same order, with Models' metric and Environment chains as rows.
- **Sources**: UX:284 (appendix 82); RJ audits[2].findings[18]; UX:127
- **Closed at**: —

#### K7-P3b · contest_html's default assumed Mode C

- **State**: not applicable · found by an internal review · severity P3 · page: models
- **Problem**: A confirmed P3 of the K7 review round: Models' contested-state renderer (contest_html) defaulted to Mode C. The inventory's K7-P3 carries only the round's Register-for-Mode-D and no-model-refusal P3s, and OPS-C5 the Load confirmation's From side.
- **At ba6e85e**: Fixed in 56c8508 (not in this shallow clone). Mode C's measurement retired since (K8-6a/6b), so the subject is likely gone (inferred).
- **Why this state**: Mode C online-total measurement is retired (K8-6a/6b; retired features list in tasks_def.md), so contest_html's Mode C default has no subject.
- **Sources**: docs/working/scorer-retraining/README.md:1108-1112 (K7 review round)
- **Closed at**: —

#### R-65 · Amber independence warning shown under 'Newest first'

- **State**: not applicable · found by the 2026-10-25 review · severity medium · page: models
- **Problem**: Models showed the amber independence warning even with 'Newest first'.
- **At ba6e85e**: Models offers only 'Newest first' and 'By family' (models_page.py:158-159 ORDER_CHOICES). No independence notice exists anywhere in src/webui (grep for independence_html, 'no longer an independent', 'uses up the cohort' finds nothing). The page at 1850x960 shows no amber notice.
- **Why this state**: Retired K8-6d: ordering by test score and its independence warning were removed; Models offers only Newest first and By family, and no independence notice exists in src/webui.
- **Sources**: UX:267 (appendix 65); RJ audits[2].findings[10]
- **Closed at**: —

#### B52-J · Checkpoint epoch wording differed across pages

- **State**: code-verified at ba6e85e (audit) · found by an internal review · page: models
- **Problem**: Recorded 0-based epoch vs completed epochs across pages.
- **At ba6e85e**: Each checkpoint says how far it trained in the same words on every page seen: '1 epoch completed' on Models, in Fine-tune's Checkpoint option ('Latest save · 1 epoch completed') and in the Diagnosis drawer rows, and '1 of 1 epochs completed' in Training's run view.
- **Sources**: BL:1010 (item 52 (J), K6 smoke)
- **Closed at**: —

#### OPS-C7 · Unreadable checkpoint not said in plain words

- **State**: code-verified at ba6e85e (audit) · found by an internal review · page: models
- **Problem**: Shown by its exception.
- **At ba6e85e**: I put a corrupt.pt (random bytes) in the workspace copy. Its row shows 'corrupt.pt · metadata not readable · cannot be loaded', and its reason reads 'the file cannot be read as a checkpoint; Ref: UnpicklingError: Weights only load failed…'. The route's problem field leads with the same words (checkpoint_paths.refusal_line, CHECKPOINT_NOT_READ).
- **Sources**: OPS:363-364
- **Closed at**: —

**UI-16, results**

#### B72-C4 · Compare code tokens break mid-token in JSON/prose cells

- **State**: to do · found by an internal review · severity minor · page: results
- **Problem**: Identifiers break mid-token; hyphen-break question for the owner.
- **At ba6e85e**: Compare's prose and JSON cells use 'overflow-wrap: anywhere' (models_page.py:1915). I measured 72 line breaks inside a token at 1850x960 and 49 at 960x1080. Examples: 'patient_represent|ation', 'distmult_raw_scor|e/tanh_hinge', 'patient_disease/n|egative_normalised', and the reference recipe sources 'mims-harvard/SHEPHERD@e95433a …' in the Learning rate, Scheduler, Weight decay and Graph backbone rows. Run ids break only at hyphens.
- **Done when**: At 1850x960 and 960x1080 no code token in Compare cells breaks except at its own separators (per the owner's hyphen answer).
- **Sources**: BL:1030
- **Closed at**: —

#### NEW-models-results-2 · Compare of 4 results is 12,342 px tall even with Differences only

- **State**: to do · found by the 2026-11-16 audit · page: results
- **At ba6e85e**: Compare of 4 results is 12,342 px tall with 'Differences only' (168 alike rows already hidden) and 21,825 px with all rows. The detail body scrolls 14,710 px against 682 visible at 1850x960. Most of it is per-stage recipe sources and raw JSON, far from one screen. Evidence: (audit evidence, not kept in the repository) no record names it.
- **Done when**: Compare of 4 results at 1850x960 with Differences only fits its key rows in about one screen, with per-stage recipe sources and raw JSON folded behind disclosures.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### B72-C1 · Compare says digests and counts twice and names the family by id

- **State**: partly done · found by an internal review · severity minor · page: results
- **Problem**: Repeated digests/LP head/counts; family by id; every row under 'Differences only' with one Unknown.
- **At ba6e85e**: 'Differences only' works: '168 rows recorded alike in every column are hidden, and 11 that differ only by an Unknown explained above'. Compare still says facts twice: - Graph-view digest 304112006a74 appears under Data identity · Graph view, as the Split's View digest, and under Content references · graph_view. - Split 268b187ce189, pairs 28e2b12861ba, samples 91b77e24e5fc and source cd305fa92e20 each appear under Data identity and again under Content references. - Pairs '6' appears under Data counts and as Rows under Data identity · Pairs. 'Edges without negative 2' appears under both Data counts and Run counts. - 'LP loss weight' is a row under both Learning setting and Loss. The Family row shows the raw id '20261116-091306_demo-base_ebaa08', while the Results table says 'family from run demo-base · 2026-11-16 09:13'.
- **Done when**: In Compare each digest and count appears once (by relation elsewhere) and Family reads in run words, not a run id.
- **Sources**: BL:1030
- **Closed at**: —

#### B63-xws · Results cannot be compared across workspaces in the page

- **State**: outside this round's scope (owner to decide) · found by the owner · page: results
- **Problem**: The owner's direction (2026-11-09) is comparing, querying and exporting results 'for any models, datasets or environments'. Item 63's note of 2026-11-10 records that Results › Compare lists a family's or a workspace's results and that 'comparing across workspaces in the page is not built, the command line's export joins them'.
- **At ba6e85e**: Direction set; the page-level cross-workspace compare not built and not scheduled.
- **Why this state**: Cross-workspace compare in the page is listed as a deferred new product surface in tasks_def.md; the CLI export joins workspaces.
- **Sources**: docs/working/BACKLOG.md:1021 (item 63, direction 2026-11-09 and notes of 2026-11-10)
- **Closed at**: —

#### OPS-C1 · Compare's 'Settings changed' applied to a non-branch

- **State**: code-verified at ba6e85e (audit) · found by an internal review · page: results
- **Problem**: Wrong row for a non-branch.
- **At ba6e85e**: Compare's 'Settings changed' row reads '○ Not applicable' for all four results, none of which is a branch: a new base and three fine-tunes. results_segment.py:810-822 gives a same-settings branch 'Same settings as its parent' and gives any other operation NOT_APPLICABLE.
- **Sources**: OPS:363-364
- **Closed at**: —

**UI-16, shell**

#### B27-P · No mockup before UI work and no completion screenshots for the owner

- **State**: to do · found by the owner · severity requirement · page: shell
- **Problem**: Process requirement: see a static mockup before UI work and screenshots when done.
- **At ba6e85e**: No mockup for the ops round's UI work exists in the repository: grep -ril mockup docs/ finds only PLAN_TEST_RESULTS.md and BACKLOG.md, both about M5, and no commit in 30e8ae1..ba6e85e mentions one. The round's screenshots were its own browser acceptance (OPS:363); no record says they were given to the owner. The skill's §12 checklist, the working rule, has no mockup step.
- **Done when**: The round's UI changes have a static mockup committed before the screen is built and screenshots delivered at completion, or the owner's explicit decision replacing this is recorded; the skill's checklist includes the step.
- **Sources**: BL:985 (item 27, 2026-10-19)
- **Closed at**: —

#### B72-N4 · Option middle cut can split a multi-code-point emoji

- **State**: to do · found by an internal review · severity minor · page: shell
- **Problem**: renderers.middle_cut can cut a multi-code-point emoji.
- **At ba6e85e**: middle_cut still slices by code point. Reproduced: a name holding the ZWJ family emoji (MAN, ZWJ, WOMAN, ZWJ, GIRL) cut to 20 characters keeps a lone GIRL after the ellipsis, and fit_option gives 'run-👨‍👩…👩‍👧-end · …', which splits the sequence.
- **Done when**: renderers.middle_cut and fit_option cut at grapheme-cluster boundaries; a unit test with a ZWJ family emoji name keeps the sequence whole or drops it entirely.
- **Sources**: BL:1030 (B0c review note)
- **Closed at**: —

#### INT-AUDIT · No UI audit script in the repository

- **State**: to do · found by an internal review · page: shell
- **Problem**: The owner's measurable complaints (squeezed charts, covered axes) were caught by the owner, not a script.
- **At ba6e85e**: The repository has no UI audit script. It has targeted pieces: token contrast tests (labels, info lines, placeholders, radios, checkboxes, status tokens) and a few browser tests that pin geometry, such as the header chips at several widths. Nothing checks input borders, the type scale, a repeated fact on screen, skill §3's group-label rule or the navigation levels, so the PREPARE/EVALUATE problem the owner raised again was not caught by any check. axe-core ran once offline at the close (OPS:363) and is not in the repository.
- **Done when**: A repository UI audit runs screenshots at the owner's sizes and checks geometry, repeat counts, axe offline, input borders at 3:1, the type scale, group-label placement and distinct navigation styles, and fails on the current PREPARE/EVALUATE problem.
- **Sources**: UX:132 (section 4); BL:1010 (52); RJ research[4].principles[1] (S2)
- **Closed at**: —

#### B27-5 · Same fact still repeated on one screen (drawer fingerprints, Models count line)

- **State**: partly done · found by the owner · severity requirement · page: shell
- **Problem**: The same fingerprint, name or state appeared in several places on one screen.
- **At ba6e85e**: Many instances are fixed: Training's run view says 'same dataset as the run bar'; Test and Models give the dataset once, under the select; Models' fingerprint column is the model's 12 hex only; Diagnosis says 'Not loaded' once. Still repeated on the screens seen: the Diagnosis load drawer shows dataset fingerprint 50bef1bf5f9f and 'synthetic' three times (the identity line plus each row). Models' count line 'newest first · 4 checkpoints in 2 runs' repeats the Order toggle 'Newest first' directly above it (B51-f). Datasets' details header 'demo ● record verified synthetic' repeats the selected row beside it (B51-h), and its footnote repeats what 'In your Training run bar' means next to USE_NOTE.
- **Done when**: The Diagnosis load drawer shows the dataset fingerprint and 'synthetic' once, and Models' count line no longer repeats the Order toggle ('newest first'); a repeat-count check on those screens pins it.
- **Sources**: BL:985 (item 27 (5)); BL:1009 (item 51); SK:23-25; RJ research[3].principles[13] (DX-14); RJ research[4].principles[5] (S6); commit 1f4eb05
- **Closed at**: —

#### B72-N1 · One run read two ways across pages

- **State**: code-verified at ba6e85e (audit) · found by an internal review · page: shell
- **Problem**: run_label and models_page.run_words dropped the run's name.
- **At ba6e85e**: A run reads one way across pages: 'demo-disease · GATv2 2026-11-16 09:13' in Training's Viewing select, the Models rows and the drawer heading, and 'demo-base · GATv2 2026-11-16 09:13' in Fine-tune's Run select. Details rows give the run's directory in full, by design (B0c review).
- **Sources**: BL:1030 (note 2026-11-13, review of B0b)
- **Closed at**: —

#### OPS-C2 · Run names that cut alike lost where they differ

- **State**: code-verified at ba6e85e (audit) · found by an internal review · page: shell
- **Problem**: Shortened run options could read the same.
- **At ba6e85e**: Run names that would cut alike keep their start through where they differ. fit_options on 'cpb-disease-full-finetune-2026-11-16-a' and 'cpb-disease-frozen-finetune-2026-11-16-a' gives 'cpb-disease-fu… · GATv2 …' and 'cpb-disease-fr… · GATv2 …', where a plain fit_option cuts both to 'cpb-dis…11-16-a'.
- **Sources**: OPS:363-364
- **Closed at**: —

#### R-19+R-50+B72 · Environment footer led with versions and never stated the consequence

- **State**: code-verified at ba6e85e (audit) · found by the 2026-10-25 review · severity medium · page: shell
- **Problem**: The runtime footer led with a version list and never said what a missing CUDA device means.
- **At ba6e85e**: The footer now leads with the consequence, in the shared warn mark: '▲ No CUDA device visible: GPU training is unavailable. torch is built for CUDA 13.0; check the driver and CUDA libraries.' The colour is rgb(122,70,0) on rgb(243,244,247), 7.06:1, from the token; there is no emoji and no raw hex. The versions are behind a closed native 'Versions ›' disclosure (7.39:1). Opened, it shows 'torch 2.10.0+cu130 · CUDA 13.0 (no device) · PyG 2.8.1 · native ext OK · Voyager'. A healthy runtime reads 'Runtime: every checked module imports.' with no mark. The row is 32 px high at 1850 and 960 wide.
- **Sources**: UX:221 (appendix 19); RJ audits[0].findings[5]; UX:252 (appendix 50); RJ audits[1].findings[15]; UX:120; BL:1030 (item 72 Shell); SK:107-108
- **Closed at**: —

**UI-16, test**

#### B72-C2 · Recorded LP metrics show raw names, unrounded values, no units

- **State**: to do · found by an internal review · severity minor · page: test
- **Problem**: Test's and Models' recorded LP metrics shown raw.
- **At ba6e85e**: A KG link-prediction report shows 'Recorded metrics: lp_accuracy 0.3333333333333333 / lp_average_precision 0.4771524771524771 / lp_loss 1.0628392696380615 / lp_roc_auc 0.3888888888888889'. These are the raw names, unrounded, with no units, on both Test and Models. Word labels (LP_METRIC_WORDS, 'ROC-AUC' and so on) exist in caveats but this display does not use them.
- **Done when**: On Test and Models a KG link-prediction report shows plain metric names (e.g. 'ROC-AUC'), values rounded to 4 digits, and units stated once.
- **Sources**: BL:1030
- **Closed at**: —

#### B72-T1 · Measurement refusals name the protocol by its code

- **State**: to do · found by an internal review · page: test
- **Problem**: protocols.inputs_refusal names the protocol by code (pinned by the P0-2 golden).
- **At ba6e85e**: The measurement refusals still name the protocol by its code. Rendered at ba6e85e: '…which the patients_like_me measurement needs. Import a source that lists causal_genes.'; '…which the kg_link_prediction measurement scores…'; 'the kg_link_prediction measurement does not measure a supplied cohort version'. model_refusal uses the same '{protocol.name}' form. A helper _named already gives plain words with the code beside, but only available()'s reason uses it. Start shows the sentence verbatim. Not seen live: the demo's models and versions do not trigger it.
- **Done when**: Rendered measurement and model refusals name the measurement by its plain label (code only beside it), with the P0-2 golden updated to match.
- **Sources**: BL:1030
- **Closed at**: —

#### NEW-test-1 · Test Start's 'Disabled: choose a model.' names a control that no longer exists

- **State**: to do · found by the 2026-11-16 audit · page: test
- **Duplicate of**: RJ-61b (closed with it)
- **At ba6e85e**: Test Start's reason with no checkpoint is 'Disabled: choose a model.' (src/webui/components/test_page.py:1697). Since B3 (e26fba9) the page has no Model control, only Run and Checkpoint, and the line under Checkpoint already says 'Choose the run, then the checkpoint to test.' (:1485). Seen: (audit evidence, not kept in the repository)
- **Done when**: With no checkpoint chosen, Test's Start reason names the Run/Checkpoint choice or is omitted, and the instruction appears once.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### NEW-test-3 · After registering, Test job card still shows 'Not registered … run record_evaluation.py'

- **State**: to do · found by the 2026-11-16 audit · page: test
- **Duplicate of**: B52-E (closed with it)
- **At ba6e85e**: After a report is registered from the page, the Test job card keeps its unfolded 12-line command-line output, ending 'Not registered (no --record). Register it with: python scripts/record_evaluation.py --report "<registry>/…" --checkpoint "<workspaces>/…"'. That is stale advice and a path, shown beside the Result's 'Registered record 2' (test_page.py:1789-1801). Seen: (audit evidence, not kept in the repository)
- **Done when**: After a report is registered from the page, the job card's output is folded behind Details and no visible line says 'Not registered' or gives the record_evaluation command.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### NEW-test-4 · Result tile values misalign when labels wrap at 1536 and 1440 widths

- **State**: to do · found by the 2026-11-16 audit · page: test
- **Duplicate of**: R-71 (closed with it)
- **At ba6e85e**: The Result's tile values fall out of line at common laptop windows: at 1536x864, the 'Disease pooled top-1/top-10' labels wrap and their values sit 19 px lower; at 1440x900, the gap is 20 px. There is no nowrap and no shared baseline (test_page.py CSS :3701-3705). Measured: (audit evidence, not kept in the repository)
- **Done when**: At 1536x864 and 1440x900 all Result tile values share one baseline within 2 px.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### NEW-test-5 · Result calls a report 'the selected workspace' on equal dataset digest only

- **State**: to do · found by the 2026-11-16 audit · page: test
- **Duplicate of**: R-64 (closed with it)
- **At ba6e85e**: Inferred, not seen: the Result says 'the selected workspace' whenever the report's dataset record digest equals the selected workspace's (test_page.py:1909-1911). So a report measured in another workspace built on the same dataset record would be attributed to the selected one.
- **Done when**: A unit test with a report measured in another workspace on the same dataset record shows the Result naming that other workspace, not 'the selected workspace'.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### NEW-test-6 · Relation keys are full sentences that wrap to 3 lines in a 12em column

- **State**: to do · found by the 2026-11-16 audit · page: test
- **Duplicate of**: R-72+B72 (closed with it)
- **At ba6e85e**: The Result's relation keys are full sentences in a 12em key column ('R2 generated split's samples match the workspace manifest' wraps to 3 lines), so each relation row is 2–3 lines tall. Seen: (audit evidence, not kept in the repository)
- **Done when**: At 1850x960 each Result relation row is one line tall (short keys, detail in the value).
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### R-79 · Test run options: detached info icon, wrapping checkbox, shared info, two 'Register' names

- **State**: to do · found by the 2026-10-25 review · severity low · page: test
- **Problem**: Test's run options had a detached ⓘ, a wrapping checkbox, a shared ⓘ and two similar Register controls.
- **At ba6e85e**: All four parts remain at 1850x960. (1) The 'Deterministic execution' label wraps (height 42 px), and its ⓘ sits at x=950 on the block's right edge (block spans 746–985), away from the text. (2) The checkbox 'Register the result' and the button 'Register this report' still have near-identical names. (3) The Batch size ⓘ ('Fixed by the server') explains Device too, and Device has none. (4) The CSS that causes it, .shep-test-checks > .block flex 0 1 auto, is unchanged.
- **Done when**: At 1850x960 the Deterministic execution label fits one line with its info icon beside the text, Device has its own help, and the Register checkbox and button have clearly different names.
- **Sources**: UX:281 (appendix 79); RJ audits[2].findings[16]
- **Closed at**: —

#### R-81 · Test idle job card, idle Cancel and 'Report kept' chip take space and colour

- **State**: to do · found by the 2026-10-25 review · severity low · page: test
- **Problem**: Test's idle job card and 'kept' chips took space and colour.
- **At ba6e85e**: While idle, the Run panel shows a bordered card 'Latest test run / No test run since the server started.' (479 × 95 px). The foot also shows 'Test run Idle' with a disabled Cancel (160 × 29) that nothing can use. The Result's chip row shows a green '● Report kept' for every kept report, and the job card's outcome repeats it. The report table rows no longer carry 'kept'; only that part is gone.
- **Done when**: While idle the Run panel shows one muted line instead of the bordered card, Cancel is hidden until a run starts, and the Result's chip row has no 'Report kept' chip.
- **Sources**: UX:283 (appendix 81); RJ audits[2].findings[21]
- **Closed at**: —

#### RJ-61b · Test's model choice gives its instruction twice and preselects nothing

- **State**: to do · found by the 2026-10-25 review · severity high (finding 61 as a whole); this part low · page: test
- **Problem**: Review finding 61 also recorded that 'Choose the model to test.' and 'Disabled: choose a model.' give the same instruction twice, and that the Model select had nothing preselected. Its fix: drop the first line, preselect 'best' (ML-CK-3). At ba6e85e model_html still says 'Choose the run, then the checkpoint to test.' (test_page.py:1485) while start_state says 'Disabled: choose a model.' (:1697) once a version is chosen. That is inferred from the code: the demo has no data version, so Start's reason read 'Disabled: choose a data version.' (c1.json). Nothing is preselected (Run empty, c1.json). That is deliberate: model_list.py:399-402 says the Test page 'never measures a model it picked itself'.
- **At ba6e85e**: Preselection declined by design in code. The doubled instruction is not named in any later record.
- **Done when**: With no checkpoint chosen, the Test page shows the choose instruction once (no 'Disabled: choose a model.' repeat naming a control that does not exist) and a valid checkpoint (e.g. best) is preselected when one exists.
- **Sources**: docs/working/webui-ux-review-2026-10-25/review.json audits[2].findings[0] (evidence and fix); docs/working/webui-ux-review-2026-10-25/review.json research[2].principles[4] (ML-CK-3); src/webui/components/test_page.py:1485, 1697; src/webui/components/model_list.py:395-414
- **Closed at**: —

#### B52-E · Test job output, paths and engineering lines take the main space

- **State**: partly done · found by the owner · page: test
- **Problem**: Test and Models repeat model, cohort, source and fingerprints and give long paths and engineering detail the space the main actions need.
- **At ba6e85e**: Identity repeats are mostly gone (see R-64 and R-70), and the regions are titled: 'Latest test run', 'Reports — 3 of the selected version', 'Result'. Engineering detail still takes the main space. After a run, the Run panel's scroll region shows a paragraph with redacted paths ('<registry>/versions/…/reports/…', '<workspaces>/demo/runs/…/evaluations.json'). Under it are 12 unfolded lines of command-line output: R10 definitions and 'python scripts/record_evaluation.py --report … --checkpoint …'. Both stay after the report has been registered from the page. The Result carries the score-semantics formula, four native-build lines with digests and the environment class. Models' record has a 24-fact metrics chain.
- **Done when**: After a run, the job's output lines, redacted paths and the result's engineering lines are folded behind 'Details ›' and the visible Run/Result panels show only the main states.
- **Sources**: BL:1010 (item 52 (E)); M5W:64-66; BL:182 (section 0 UI batches); OPS:211 (B3)
- **Closed at**: —

#### R-70+B51bcd · Test still repeats version role, counts/exclusions and fingerprint in run id

- **State**: partly done · found by the 2026-10-25 review · severity medium · page: test
- **Problem**: Test repeated identity across panels: version option and line both gave role and counts; Run header repeated workspace and version; Result repeated version, role, fingerprint; report id repeated fingerprint.
- **At ba6e85e**: Done: the workspace option is the name only ('demo'); the Run header repeats nothing (run_header_html:1458); the Result names workspace, model and version by relation; the tiles carry no count. Still repeated on one screen at 1850x960 (visible-text count): (1) the version's role is in its option ('demo-cohort-1 · role mygene2') and again in the Role row under it, where 51(b) asked for the option to carry the role and the line the counts; (2) the exclusion breakdown and '12 source cases' are in Data's Cases row and again in the Result's Coverage row, though the version is 'the selected version'; (3) the model's 12 hex is in Run's details and again inside the Result's Report run id (51(d)).
- **Done when**: At 1850x960 the version's role appears once (option or Role row, not both), counts and exclusions appear once when the result's version is the selected one, and the report id does not restate the fingerprint.
- **Sources**: UX:272 (appendix 70); RJ audits[2].findings[6]; BL:1009 (item 51 (b), (c), (d))
- **Closed at**: —

#### R-71 · Test result tiles repeat the task prefix and misalign when labels wrap

- **State**: partly done · found by the 2026-10-25 review · severity medium · page: test
- **Problem**: 'Mode C' prefixed every tile; the first tile wrapped.
- **At ba6e85e**: 'Mode C' is gone: the Mode C measurement was retired (K8-6a/6b), and kept Mode C records get no tiles. But the current task's four tiles each start with the same word: 'Disease pooled MRR', 'Disease pooled top-1', 'Disease pooled top-10', 'Disease coverage'. They sit under a heading that already says 'Disease task (model bundle): …', while the rows below drop the prefix. At 1850x960 the labels fit and the values align (y=652). At 1536x864, 'Disease pooled top-1' and 'top-10' wrap, and their values sit 19 px lower (671 vs 652). At 1440x900, 'Disease coverage' sits 20 px off (725 vs 745). The tile CSS has no nowrap and no shared baseline.
- **Done when**: The task's tiles drop the 'Disease' prefix already in their heading, and at 1440 and 1536 widths all tile values sit on one baseline.
- **Sources**: UX:273 (appendix 71); RJ audits[2].findings[7]; UX:127
- **Closed at**: —

#### R-72+B72 · Relation rows still lead with their code (R1, R2...) on Test and Models

- **State**: partly done · found by the 2026-10-25 review · severity medium · page: test
- **Problem**: Test's version details and Relations, and Models' 'Relations:' line (~30 ' · ' facts), were run-on dot strings.
- **At ba6e85e**: The run-on chains are gone: the cohort version is a summary list, and each relation is its own row with a shared mark. The finding's title is '…run-on dot strings that lead with codes', and its fix put the meaning first with the code after ('Graph files match manifest (R1) ✓'). The relation rows still open with their codes: 'R1 graph files match the workspace manifest', 'R2 generated split's samples…', 'R3 model trained on this graph', 'R5 dataset record bound to this graph'. This is by design in the one renderer (renderers.py:1828-1829: label = f"{name} {display}"), on Test and on Models. See own-r1-unreg-after.txt lines 135-141 and p-models-row0.txt in the checkers' out directories. That breaks skill §5's front-loading: the first two words carry the code. The same Models record block still runs a 24-metric ' · ' chain and an 'Environment: … · Dataset: … · Registered …' chain (p-models-row0.txt lines 59-60). That part belongs to R-82's partly verdict. This is a minor residual, but it is the substance the title names.
- **Done when**: Each relation row on Test and Models starts with its meaning and shows the code after it (e.g. 'Graph files match manifest (R1)'), and its key does not wrap to 3 lines.
- **Sources**: UX:274 (appendix 72); RJ audits[2].findings[8]; BL:1030 (item 72 Models)
- **Closed at**: —

#### K7-P3 · Register offered for kept Mode D report; no-model refusal named wrong control

- **State**: not applicable · found by an internal review · severity P3 · page: test
- **Problem**: K7 review UI P3s.
- **At ba6e85e**: Commit 56c8508 is not in this clone: history before 9a96a72 is grafted, and git cat-file fails for it. Part 1 (Register offered for a kept Mode D report) was reversed on purpose. Under K8-6c (policy rev 15) no rule refuses a report by its mode, and a test pins Register as enabled for a kept Mode D report. Part 2 (the no-model refusal) concerned Diagnosis in the K7 smoke. Diagnosis now says 'Disabled until a model is loaded.' and its card's action reads 'Load a model…'; I read this in code and did not look at it (that page belongs to the diagnosis checker).
- **Why this state**: Obsolete: Register for a kept Mode D report was reversed on purpose under K8-6c (policy rev 15, pinned by a test), and the no-model refusal concerned Diagnosis, which has since changed. The new 'choose a model' wording on Test is tracked separately (RJ-61b / NEW-test-1).
- **Sources**: SR:1108-1112
- **Closed at**: —

#### ACC-773 · Test run card 'Not registered' beside Result's 'Registered #1'

- **State**: code-verified at ba6e85e (audit) · found by the 2026-10-25 review · severity display contradiction · page: test
- **Problem**: Result and the report list read 'Registered #1' while the run card still read 'Not registered' in the present tense (screenshots 03, 06).
- **At ba6e85e**: The job card heads its reading 'When the run ended:' and adds 'The report's registration now is shown in Result.' Seen after I registered a report from the page: the card read 'When the run ended: ● Report kept · Not registered', beside the Result's '● Registered record 2' and the row's 'Registered #2'. The two now read as different moments, not as contradicting present states.
- **Sources**: M5W:50-61; UX:184-185; PTR:52
- **Closed at**: —

#### B51-e · Each Test report row said 'not registered' twice

- **State**: code-verified at ba6e85e (audit) · found by an internal review · page: test
- **Problem**: Each report row said 'not registered' in its text and its tag.
- **At ba6e85e**: Each report row has the time, the model by relation, the measurement and one state chip. No sub-line restates the chip ('not registered' appears once per row).
- **Sources**: BL:1009 (item 51 (e))
- **Closed at**: —

#### R-61+B52I+B72 · Test's model select was a dot-joined provenance chain per option

- **State**: code-verified at ba6e85e (audit) · found by the 2026-10-25 review · severity high · page: test
- **Problem**: Each Test model option was a ~100-character chain wrapping to two lines, unlike Training's Run -> Checkpoint pair.
- **At ba6e85e**: The Test page picks the model with two cascading selects, Run then Checkpoint, using the shared model_list.parents_update. Each option has at most 40 characters and one ' · '. Seen on the shared server: Run options 'demo-disease · GATv2 2026-11-16 09:13' (37 characters) and 'demo-base · GATv2 2026-11-16 09:13'; Checkpoint options 'Latest save · 1 epoch completed' (33) and 'Best pooled MRR · 1 epoch completed' (35). The chosen file's details are a summary list: Model, File, Validation, Model fingerprint (12 hex with Copy), Dataset.
- **Sources**: UX:263 (appendix 61); RJ audits[2].findings[0]; UX:66 (section 3 high #5); UX:126; BL:1010 (item 52 (I), K6 smoke; 'two-level selects elsewhere'); BL:1030 (item 72 Test)
- **Closed at**: —

#### R-62 · Report rows read identically with different registration states

- **State**: code-verified at ba6e85e (audit) · found by the 2026-10-25 review · severity high · page: test
- **Problem**: Two report rows read identically while showing different registration states.
- **At ba6e85e**: Report rows that would read alike get the seconds, then the run id's last part. Seen on my isolated copy: two runs in one minute read '2026-11-16 10:13:56 UTC' and '10:13:46 UTC', and an older one reads '10:12 UTC'. Each row has one state chip. For an operator every row is refined: 'Registered #1', 'Registered #2', 'not registered'. No two rows read the same with different states.
- **Sources**: UX:264 (appendix 62); RJ audits[2].findings[1]; UX:67 (section 3 high #6); UX:89; UX:108-109
- **Closed at**: —

#### R-64 · Result did not say which workspace or run produced the report

- **State**: code-verified at ba6e85e (audit) · found by the 2026-10-25 review · severity high · page: test
- **Problem**: The Result did not say which workspace or run produced the report.
- **At ba6e85e**: The Result opens with a summary list: Report (run id), Workspace, Model, Version, Deterministic algorithms, Device. Each fact is said by its relation when it matches the selection ('the selected workspace / model / version') and in full when it differs. Seen with another checkpoint selected: the report's Model row read 'Disease / demo-disease · GATv2 2026-11-16 09:13 / Latest save · 1 epoch completed / 63d8975e5d6c'. The Registration row names the ledger by a short relative path.
- **Sources**: UX:266 (appendix 64); RJ audits[2].findings[3]; UX:69 (section 3 high #8); UX:128; BL:1010
- **Closed at**: —

#### R-80 · 'Other metrics' sorted lexically (Hits@100 before Hits@20)

- **State**: code-verified at ba6e85e (audit) · found by the 2026-10-25 review · severity low · page: test
- **Problem**: Other metrics sorted lexically.
- **At ba6e85e**: A task record's cut-off families are sorted by k as numbers. Seen: 'Pooled top-k: k=3, k=5, k=20, k=50, k=100'; 'Macro recall@k: k=1 … k=100'. Models sorts its other metrics with _metric_order, by k. The Mode C measurement that produced 'Hits@100 before Hits@20' is retired.
- **Sources**: UX:282 (appendix 80); RJ audits[2].findings[17]
- **Closed at**: —

**UI-16, training**

#### B41 · Chart is silent when a drawn run cannot get its newer epochs

- **State**: to do · found by an internal review · page: training
- **Problem**: After 20 retries an ended run stays at older epochs with no message.
- **At ba6e85e**: From the code; the demo cannot make the series server function fail. After RETRY_MAX (20) failed retries, setMessage runs only when inst.run === null (shep_charts.js:819-827). A drawn run that cannot get its newer epochs therefore stays at its older epochs with no message.
- **Done when**: When the series function fails RETRY_MAX times for a drawn run, the chart shows a line over the curves such as 'Newer epochs could not be loaded.' (covered by a JS unit test).
- **Sources**: BL:999 (item 41); PTR:2380-2381
- **Closed at**: —

#### B69-T2 · Run with only best.pt reads 'dataset not recorded (no checkpoint)'

- **State**: to do · found by an internal review · severity cosmetic · page: training
- **Problem**: The header reads last.pt only.
- **At ba6e85e**: From the code; not seen, because the demo has no run with only best.pt. _snapshot_for reads checkpoints/last.pt only (training_console.py:2179-2195). With none, the header says 'dataset not recorded (no checkpoint)' (:2241-2243), and the state line also says 'No checkpoint saved, so nothing can be continued' (:2361-2362), even though best.pt exists and the drawer lists it.
- **Done when**: For a run with only best.pt, the run-view header names the dataset and the state line does not say no checkpoint was saved, verified by a unit test of _snapshot_for on such a run.
- **Sources**: BL:1027
- **Closed at**: —

#### B72-RV1 · Run view state line carries two ' · ' and says 'use Continue…' with drawer open

- **State**: to do · found by an internal review · page: training
- **Problem**: Two separators in the state line.
- **At ba6e85e**: The run view's state line, now inside the header, still carries two ' · ': 'Finished · 1 of 1 epochs completed · to train further, use Continue…' (training_console.py:2367). The stopped-early form does the same (:2370). The line also says 'use Continue…' while the Continue drawer is open and the Continue… button is hidden.
- **Done when**: The run view's finished and stopped-early state lines contain at most one ' · ', and do not say 'use Continue…' while the Continue drawer is open.
- **Sources**: BL:1030
- **Closed at**: —

#### NEW-help-1 · 17 model-spec fields in Training's Model accordion have no help

- **State**: to do · found by the 2026-11-16 audit · page: training
- **At ba6e85e**: Training's Model accordion: 17 editable model-spec fields have no ⓘ and no help text. The number fields are Dimension, Layers, Hidden, Output, Heads, First hidden multiplier, Dropout, Attention dropout, Attention slope, Activation slope, Relation edge dimension (it has only its format line) and Margin. The switches are Type embedding, Add self loops, Share weights, Concatenate and Residual. B1a added them: training_console.py:4461-4469 builds them with value_field and registers no help, and FIELD_HELP (:4230-4319) covers only BASE_FIELDS and the branch fields. Evidence: (audit evidence, not kept in the repository) and training-info-lines.json ('fieldsWithoutHelp'). I found no record of it.
- **Done when**: Every editable field in the Model accordion has an ⓘ or info line; the training-info-lines check reports zero fieldsWithoutHelp.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### NEW-training-finetune-1 · Continue Check with Device auto shows the CLI's --device sentence

- **State**: to do · found by the 2026-11-16 audit · page: training
- **Duplicate of**: R-32 (closed with it)
- **At ba6e85e**: Continue drawer, Check with the default Device auto on this no-CUDA host: the result box shows the CLI's sentence '--device auto requires CUDA … Pass --device cpu explicitly for development; the run records the device it was given. Nothing was started.' (flag words, about 50 words). Source: training_manager.py:619-632; evidence: (audit evidence, not kept in the repository)
- **Done when**: On a no-CUDA host, Check in the Continue drawer with Device auto shows a short page-words refusal (e.g. 'No CUDA device: choose cpu') with no '--device' flag words.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### NEW-training-finetune-3 · State line says 'use Continue…' while the Continue drawer is open

- **State**: to do · found by the 2026-11-16 audit · page: training
- **Duplicate of**: B72-RV1 (closed with it)
- **At ba6e85e**: The run view's state line says 'to train further, use Continue…' while the Continue drawer is open and the Continue… button is hidden (training_console.py:2367; drawer-1850x960-1-open.png)
- **Done when**: With the Continue drawer open, the run view's state line does not mention 'use Continue…'.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### NEW-training-finetune-4 · Run view reads '1 of 1 epochs completed'

- **State**: to do · found by the 2026-11-16 audit · page: training
- **At ba6e85e**: '1 of 1 epochs completed' in the run view's state line: _epochs_done does not apply the singular (training_console.py:2316-2319)
- **Done when**: A unit test of _epochs_done gives '1 of 1 epoch completed' for one epoch and 'epochs' for more.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### NEW-training-finetune-8 · Skill says 140 px chart floor while code uses 170

- **State**: to do · found by the 2026-11-16 audit · page: training
- **Duplicate of**: R-38+owner (closed with it)
- **At ba6e85e**: Training page footer and the skill disagree on the chart floor: SKILL.md:41 says 'today 140 px plot area', the code's PLOT_MIN_HEIGHT is 170 (training_console.py:4806)
- **Done when**: SKILL.md's chart floor value matches PLOT_MIN_HEIGHT in training_console.py.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### NEW-training-finetune-9 · Record gaps: UX:130, B72-CD4 and unnamed appendix items lack notes

- **State**: to do · found by the 2026-11-16 audit · page: training
- **At ba6e85e**: Record gaps: UX:130 still lists 'Continue's buttons in the drawer's footer' as Later, although 7649fa0 put them there; B72-CD4 (extend_epochs) was fixed by 7649fa0's page_words, but no dated BACKLOG 72 note says so. The appendix items with no named record (R-39, R-41, R-59, R-60, B72-RV1) are kept open only by UX:116's blanket 'the rest of the appendix' and BACKLOG 52's 'its appendix holds all 83 findings'; no stage or schedule names them
- **Done when**: UX:130 no longer lists Continue's footer buttons as Later, BACKLOG 72 has a dated note for B72-CD4 citing 7649fa0, and R-39, R-41, R-59, R-60, B72-RV1 are each named in a scheduled record.
- **Sources**: found by the 2026-11-16 audit at ba6e85e
- **Closed at**: —

#### R-39 · 'No training is running' said three times, once in a button-shaped pill

- **State**: to do · found by the 2026-10-25 review · severity medium · page: training
- **Problem**: While idle the same sentence appeared three times, once in a button-shaped pill.
- **At ba6e85e**: When idle, the same fact appears three times. The header chip says 'GPU jobs: none'. The run-bar pill says '○ Idle · no training job since the server started' (training_console.py:1515-1516), still drawn as a bordered, fully rounded lozenge (CSS :4866: 1px border, radius 999px, 30 px tall) beside the 40 px buttons. The reason line says 'Stop disabled: no training is running.' (:1700-1702) next to a visible, disabled '■ Stop'.
- **Done when**: While idle the Training page states the idle fact once, as plain muted text (no bordered rounded pill), and Stop is hidden or has no reason line.
- **Sources**: UX:241 (appendix 39); RJ audits[1].findings[3]; RJ audits[1].overall
- **Closed at**: —

#### R-41 · Run details popover repeats folder/fingerprint, lacks model fingerprint, nests help bubble

- **State**: to do · found by the 2026-10-25 review · severity medium · page: training
- **Problem**: The Run details popover repeated facts, lacked the model fingerprint and nested a help bubble.
- **At ba6e85e**: The Run details popover now uses the shared summary list (training_console.py:2290-2311, renderers.summary_list), but its content has the same three problems. The run folder appears twice: as Run 'gatv2/20261116-091309_demo-disease_269dc9' and inside Checkpoint 'demo/runs/gatv2/20261116-091309_demo-disease_269dc9/checkpoints/last.pt'. Logs is a sentence ('the run directory's logs/'), not a value. 'Dataset used' shows 50bef1bf5f9f with Copy, which repeats the run bar's fingerprint while the header already says 'same dataset as the run bar', and it carries an ⓘ (help_button), so a help bubble opens from inside the popover. There is no model fingerprint row, although the drawer names the same file by 63d8975e5d6c.
- **Done when**: The Run details popover names the run folder and dataset fingerprint once (or 'same as the run bar'), shows the model's 12-hex fingerprint with Copy, gives Logs a value and contains no nested info bubble.
- **Sources**: UX:243 (appendix 41); RJ audits[1].findings[8]; RJ audits[1].overall
- **Closed at**: —

#### B69-T1 · val_loss-only run's middle charts show the generic empty message

- **State**: partly done · found by an internal review · severity cosmetic · page: training
- **Problem**: Generic 'No other figure charted'; blank charts at full height with an orphan axis label narrow.
- **At ba6e85e**: From the code only: the demo has no run whose log holds only val_loss. An empty chart now hides its plot and axis label (shep_charts.js:606-609 toggles shep-chart-empty; 30f7249), so blank charts no longer keep full height or leave an orphan axis label. A plain val_loss log is still the kind 'other' (training_console.py:2412-2421). Its two middle charts are still titled with the generic 'No other figure charted: Export CSV has every logged figure' and 'No further figure charted' (:2406-2407), with no line of their own (empty is None for 'other', :2497-2498).
- **Done when**: For a run whose log holds only val_loss, the empty charts say the run logs only its loss (not the generic 'No other figure charted'), checked by a renderer unit test with such a log.
- **Sources**: BL:1027
- **Closed at**: —

#### R-32 · Device default 'auto' is refused on a CUDA-less host with no warning before Start

- **State**: partly done · found by the 2026-10-25 review · severity high · page: training
- **Problem**: The Device help said 'auto' falls back to CPU; the trainer refuses 'auto' without CUDA.
- **At ba6e85e**: The help text is fixed. Both Device help texts say auto is CUDA and is refused without it (training_console.py:4248-4254 for the base form, :4298-4305 for the Continue drawer; finetune_page.py:1131 the same). Since AF1 dc08792 the job runner resolves the device before it spawns anything (training_manager.py:619-632 and :1264; training_device.py:25-49), so Start with auto on this host is refused and creates no run (inferred from the code: Start was not pressed, per the audit rules). The B5 footer (6a9a7c3) leads with 'No CUDA device visible: GPU training is unavailable…'. What remains: Device still defaults to auto on Training (:4439-4440), in the Continue drawer (:4518-4519) and on Fine-tune (finetune_page.py:1249-1250). Start and Continue Training stay enabled, and nothing at the Device field or beside Start says auto will be refused here. Beside Start the reason line reads 'Stop disabled: no training is running.' In the Continue drawer, Check with the default auto answers in the CLI's words: '--device auto requires CUDA, which is not available here. … Pass --device cpu explicitly for development; the run records the device it was given. Nothing was started.' (seen)
- **Done when**: On a host with no CUDA device, the Device radios on Training, Continue and Fine-tune show an info line that auto and cuda will be refused (choose cpu), or a reason beside Start, and the refusal text uses page words.
- **Sources**: UX:234 (appendix 32); RJ audits[1].findings[0]; UX:62 (section 3 high #1); UX:198-199
- **Closed at**: —

#### R-42+owner · Training help texts lack bold leads, exceed budget, and Model-spec fields lack help

- **State**: partly done · found by the owner · severity medium · page: training
- **Problem**: Training help texts were pipe-joined fragments in two styles, with must-know constraints inside bubbles and no bold.
- **At ba6e85e**: Done: no help text uses pipes any more (0 of 131). Each Training setting's help is built in one shape from its field spec (setting_help, training_console.py:365-385; FIELD_HELP :4230-4319): a one-line lead, then 'Reference:', 'Source:' and a closing line. Formats and must-know rules are now visible info= lines under their fields, built from their roots (FIELD_FORMATS, training_console.py:335-352; 12d0c8e). In the browser I saw Neighbours per hop 'Whole numbers separated by commas, one per encoder layer; -1 takes all.', Early stopping patience 'Epochs, or "off"…', Gradient accumulation, Epochs per validation and Relation edge dimension. The HGT batch limit and the 'heads must divide' rule went with the features they belonged to: only GATv2 is offered (shown as a fact, 'Architecture GATv2'), whose width is output x heads (spec.py:294), and spec.validate refuses bad combinations. Remaining: (1) No bold or emphasis anywhere, because the renderer is plain text. (2) The drawer and chart texts are still long prose: checkpoint_dropdown 101 words, training-charts 110, continue_source 75, continue_operation 66, continue_device 64. (3) Each setting's bubble carries engineering references in the middle of its text ('Source: mims-harvard/SHEPHERD@e95433a hparams.py:18-44 (200 epochs)', 'README C6', 'README U15'). (4) Two bullet styles: 18 texts use '•', Device uses '- '. (5) The Model accordion's spec fields have no help at all (see the first new problem).
- **Done when**: Training help bubbles show bold leads via the shared renderer, drawer and chart texts fit the budget, Source/README references sit in a trailing 'Ref:' line, one bullet style is used, and the Model-spec fields have help.
- **Sources**: UX:244 (appendix 42); RJ audits[1].findings[9]; RJ audits[1].overall; UX:121; commit 1f4eb05
- **Closed at**: —

#### R-54 · One-series chart legends repeat their title; '(log)' and x label repeated

- **State**: partly done · found by the 2026-10-25 review · severity low · page: training
- **Problem**: Chart titles used jargon; one-series legends repeated their title.
- **At ba6e85e**: Chart titles follow the run's kind (30f7249; training_console.py:2385-2410). A base reads 'LP accuracy (validation, Trainer)' and 'ROC-AUC and AP (validation, Trainer)'; a task model reads 'Pooled MRR (validation, Trainer)' and 'Pooled top-k (validation, Trainer)'; the fourth is 'Learning rate (log)'. 'Trainer top-20' and 'Hits@K' are gone, and naming the Trainer is now a skill rule (§8). Single-series legends still repeat their titles ('pooled MRR', 'LP accuracy', 'learning rate'), because keysHtml draws a key for every series (shep_charts.js:400-405). 'epoch (shared axis)' still sits under both bottom charts.
- **Done when**: A chart with one series shows no legend, the learning-rate title says '(log scale)', and the epoch x label appears once per chart grid.
- **Sources**: UX:256 (appendix 54); RJ audits[1].findings[20]
- **Closed at**: —

#### B52-H · Run view could chart only the current run

- **State**: code-verified at ba6e85e (audit) · found by an internal review · page: training
- **Problem**: An earlier run's curves or CSV could not be chosen.
- **At ba6e85e**: The Viewing select lists the workspace's runs (demo-disease and demo-base). Choosing demo-base redrew the charts with the base's titles and the header 'a past run', while the run bar and its pill stay the training job's.
- **Sources**: BL:1010 (item 52 (H)); OPS:208
- **Closed at**: —

#### K6-patience · Base form's patience field left the form invalid

- **State**: code-verified at ba6e85e (audit) · found by an internal review · severity P2 · page: training
- **Problem**: K6 smoke P2: patience field and an Epochs edit left the form invalid.
- **At ba6e85e**: Early stopping patience reads 'off' once the form mounts. Changing Epochs to 3 gives 'Set by you: Epochs 3 (reference 200)' with no 'Not valid' (seen).
- **Sources**: SR:963-966
- **Closed at**: —

#### OPS-C3 · Run bar's Run name did not name a continued or branched run

- **State**: code-verified at ba6e85e (audit) · found by an internal review · page: training
- **Problem**: Run name wrong for continued/branched runs.
- **At ba6e85e**: Judged from the code; not exercised, because it would start a run. The drawer's start takes the run bar's Run name (_on_resume inputs, training_console.py:4743-4746), and the Run name help says 'It names a continued or branched run too' (:4239-4243). The Run name field stays visible in Continue mode (seen).
- **Sources**: OPS:363-364
- **Closed at**: —

#### OPS-C4 · '1 epochs' in the continuation plan's words

- **State**: code-verified at ba6e85e (audit) · found by an internal review · page: training
- **Problem**: Singular not followed.
- **At ba6e85e**: The continuation plan's words now read 'Extend training by 1 epoch at a fixed rate of 1e-05 (epoch 2, new target 2)', and the options read '1 epoch completed'.
- **Sources**: OPS:363-364
- **Closed at**: —

#### R-47+TR-F5+owner · Checkpoint options were a wall of dot-joined jargon

- **State**: code-verified at ba6e85e (audit) · found by the owner · severity medium · page: training
- **Problem**: At b14c4de a checkpoint entry wrapped to four lines (run, epoch, file, kind, fingerprint); after 1f4eb05's Run -> Checkpoint pair the audit still found 'Latest save · … · Trainer val MRR 0.2800' and details repeating the Run.
- **At ba6e85e**: Checkpoint options read 'Latest save · 1 epoch completed' and 'Best pooled MRR · 1 epoch completed' (base: Latest save / Best LP accuracy), with the distinguishing word first, one ' · ' and at most 40 characters. Run options read 'demo-disease · GATv2 2026-11-16 09:13'. 'Best' comes from the recorded selection metric. The chosen file's details form a summary list: File, Kind, Holds, Continuing, Validation 'pooled MRR (Trainer) 0.3133 — selection metric on val, not an independent test', Selection, Model fingerprint 63d8975e5d6c with Copy, and Run (its directory).
- **Sources**: UX:249 (appendix 47); RJ audits[1].findings[5]; M5W:21; commit 4764faa (F5); commit 1f4eb05 (F5); UX:101-102; UX:131; UX:158-161 (decision 4); RJ audits[1].overall; RJ research[4].principles[6] (S7); RJ research[2].principles[0] (ML-ID-1)
- **Closed at**: —

#### TR-F2 · Chart wheel did not zoom in a narrow window

- **State**: code-verified at ba6e85e (audit) · found by the owner · page: training
- **Problem**: Wheel zoom was skipped below 1400 x 760.
- **At ba6e85e**: At 960x960 the mouse wheel over the first plot zoomed it ('zoomed: epochs 0–2') and the page did not scroll (scrollY 686 before and after). The chart module's header says the wheel zooms in every layout, narrow included (shep_charts.js:65-68).
- **Sources**: M5W:21; commit 4764faa (F2)
- **Closed at**: —

#### TR-F3 · Chart epoch axis could run below 0

- **State**: code-verified at ba6e85e (audit) · found by the owner · page: training
- **Problem**: A chart view could start below epoch 0.
- **At ba6e85e**: normX clamps the view's start at 0 (shep_charts.js:153-155). After zooming, 30 ArrowLeft pans ended at 'zoomed: epochs 0–1', never below 0.
- **Sources**: M5W:21; commit 4764faa (F3)
- **Closed at**: —

#### TR-F4 · Stop's question named a stale epoch

- **State**: code-verified at ba6e85e (audit) · found by the owner · page: training
- **Problem**: Stop's question named the last checkpoint's epoch read when asked, stale by the answer.
- **At ba6e85e**: Judged from the code; not seen, because the question appears only while a job runs and the audit may not start one. STOP_QUESTION names no epoch and says what can be continued is shown once the job has stopped (training_console.py:3150-3171).
- **Sources**: M5W:21; commit 4764faa (F4)
- **Closed at**: —

