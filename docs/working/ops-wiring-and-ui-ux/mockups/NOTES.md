# T01 mockups: navigation, model list, Settings › Data locations

Base: `ba6e85e` (branch `claude/ops-wiring-and-ui-ux`). Date: 2026-11-16.

**The owner has not approved these mockups.** They apply the defaults D01, D04, D05, D06 and D07 from the rev 2
package as I read them, and I checked them against the WebUI skill myself. Nobody else has reviewed them: no owner
sign-off, no Codex review. Implementation can go ahead (plan §2: the mockups do not wait for approval), but none of
this counts as visual acceptance.

## How they were made

- **Real app, real data.** `git archive ba6e85e` into a scratch snapshot. `scripts/setup_demo.py --train-model
  --epochs 1 --device cpu` built the demo workspace. The earlier fixture script `train_families.py` then added a
  frozen fine-tune, a gene fine-tune, a branch, an extension, a second base and its disease model: 8 runs, 16 saved
  files, 2 families. The app ran from the snapshot at three ports:
  - 8871: M1, workspaces root and registry set. Navigation, Models, drawer and most Settings states.
  - 8872: M1 only, nothing configured. The real first start.
  - 8873: not M1, root and registry set. The real read-only state.
- **Injected, not rebuilt.** Every mockup is CSS or DOM injected into the running page, so fonts, tokens, Gradio
  widgets and the real details panels are the app's own. The Data locations panel is new HTML inside the real
  Settings page, and its buttons reuse Gradio's own button classes. Every name, task, operation, epoch count,
  fingerprint and parent in the model tree comes from the server's family listing (`server_lists.list_family`,
  dumped to `work/family.json`).
- **Not real:**
  - The folder paths on the Settings panel are illustrative (`/home/ops/SHEPHERD-data/...`). The servers actually
    ran on scratch paths about 120 characters long.
  - The GPU chip in the "pending" state was rewritten to the chip's own wording for a runner job.
  - In two Models states (selection outside the filter, and D05) the real details panel of the selection was put
    back after the real page dropped it (see finding F1).
- Screenshots: Chromium (Playwright), deviceScaleFactor 1, PNG, viewport only. Wide is 1850×960; narrow is 760×1000,
  plus 420×900 for navigation.
- Scripts are in `scripts/`: `common.cjs` (the D01 CSS and tokens, the self-check), `nav.cjs`, `tree.cjs` (the shared
  tree list), `models.cjs`, `drawer.cjs`, `settings.cjs`, `focus_check.cjs`. Measurements are in `shots/*.json`.
  "Before" screenshots of today's pages are in `work/base_*.png`.

All files below are in `shots/`.

## D01 Navigation (option (e), refined)

Files:
- `D01_nav_1850x960_normal.png`
- `D01_nav_1850x960_hover-datasets.png`
- `D01_nav_1850x960_focus-finetune.png`
- `D01_nav_760x1000_narrow.png`
- `D01_nav_420x900_narrow.png`

Choices:
- **Captions above each group.** "Prepare" and "Evaluate" sit above their tab groups at 14 px / 600 in the muted
  token, which is 6.13:1 on white. They are set apart by:
  - position: 2 px above the first tab of the group, left-aligned with its text;
  - spacing: 33 px between the groups, plus a 1 px divider;
  - weight: 600 against the tabs' 500, and colour.

  They are generated content with empty alternative text and `pointer-events: none`. A hit-test at a caption lands
  on the tablist, not a tab, with cursor `auto`. They cannot be focused and have no hover. **Sentence case** follows
  the D01 text and skill §5. Option (e) as first rendered was uppercase with tracking; going back is one CSS line
  (`text-transform`).
- **Tab size.** Tabs are 32 px tall with rounded top corners. Row 1 is 47 px (switch padding 5 px) and row 2 is
  55 px (22 px caption tier plus 32 px tabs), so header plus tabs is **102 px** at 1850×960. Target: about 100.
- **Shared tokens on `:root`:** `--shep-tab-hover-bg`, `--shep-tab-hover-line`, `--shep-tab-selected-line`,
  `--shep-tab-selected-text`, `--shep-focus-ring`, `--shep-group-caption` and `--shep-control-border`. All are aliases
  of existing colours; no new hex.
  - **Hover:** grey-50 fill and a 3 px grey-500 line. Before this, the hover was #f9fafb on white and could not be
    seen.
  - **Keyboard focus:** a 2 px accent-600 ring drawn inside the tab (offset −2 px), so the row never clips it. The
    same token is used on the level-1 switch. Today the browser's default 1 px auto outline is drawn.
  - **Selected:** keeps its three cues: a 3 px accent line, accent text, weight 700. It reads clearer because
    nothing else now looks like it: hover is grey and focus is a ring.
- **Rejected for the selected tab: a tinted fill.** The level-1 Settings tab already uses tint plus underline, and
  skill §3 says no two navigation levels share a treatment. A 4 px line is a one-token change if the owner wants
  more.
- **Narrow.** The groups wrap as whole groups, using the same measure-and-toggle pass as row 1's `shep-nav-wrapped`.
  - 760: one row; header plus tabs is 151 px because the chips wrap in row 1.
  - 420: Prepare on one line and Evaluate on the next; 281 px.
  - Going back from narrow to wide returns to one row at 102 px.
- **Superseded (2026-11-21):** the captions, the divider, the group gap, the grey hover line and the narrow group wrap
  are retired by the external reviewer's package of 2026-11-21. The stage of the page shown is a plain-text "Stage:
  Prepare / Evaluate" hint at the right of the research tab row, a hovered tab has a fill and no line, and header plus
  tabs is about 84 px at 1850×960. See the round README §12, 2026-11-21 ("stage hint (N)"), and UI_LEDGER OPS-SL-N.
  The text above is kept as T01's record.

## D04 Model list (Models page)

Files (each at `_1850x960` and `_760x1000`):
- `D04_models_default`: by family, every run's files folded, nothing selected.
- `D04_models_run-expanded-file-selected`: a real selection (the extension run's "Best pooled MRR" file). Its
  family and run are unfolded and located, and the other family is folded by hand.
- `D04_models_search-name`: "disease".
- `D04_models_search-fingerprint`: "8f67".
- `D04_models_selected-outside-filter`: Task set to Gene, with the Disease selection kept.

Choices:
- **The tree.** Family rows sit over run rows (indented by lineage from the exact parent file), which sit over
  saved-file rows. Each family and run has its own 24×24 fold toggle.
  - Columns: Model, Task, Operation, Epochs, Files, Fingerprint. Rows are 36 px; numbers are right-aligned.
  - The Validation column is gone from the list. Scores stay in the details panel, so validation figures from
    different tasks are not shown as one scale (plan UI-04).
- **Operation words.** "New run", "Fine-tune, full", "Fine-tune, frozen", "Branch", "Extend".
  - D04's "Base" is drawn as Task "Base" (the Task filter already treats Base as a value) plus Operation "New run",
    so the same word does not appear twice in a row.
  - Frozen comes from the listing's own kind ("frozen graph backbone").
- **File words.** "Latest save" and "Best pooled MRR" (or "Best LP accuracy" for a base), named by the run's
  recorded selection metric. These are the app's own `checkpoint_words`.
- **Family heading.** "Family from demo-base", with "6 runs, 12 saved files" on the right. It names its root as a
  relation; the root's own row carries the full label.
- **Order control kept.** By family is the default; Newest first lists run summaries flat (see D05).
- **Search** sits before the Order control, with a plain "Search" label and a 24×24 clear button. Intended rule
  (D04):
  - Case-insensitive substring of the name.
  - Or a fingerprint prefix: at least 4 hex digits, or the full 64.
  - A query that could be either should try both; the mockup draws only the two clear-cut cases.
  - Matches keep their ancestors. Ancestor-only rows are muted (text-2, weight 500).
  - The matched text gets an amber-100 mark with a 2 px amber-800 underline, so the mark has a shape and not only
    a colour.
  - The count line says what is shown, for example "5 runs match "disease", shown with the runs they come from".
- **Selection outside the filter.** A grey ○ line under the count: "Selected model outside this filter: a Disease
  model, its details kept on the right", with one action, "Show all tasks". The selection's details stay on the
  right, and the list does not pretend the model matches.
- **Narrow.** The list's own width decides, through a container query, not the window (D02's direction). Below
  880 px a run's architecture and start time move under its name; the name never breaks.
- **Superseded (2026-11-21), By family only:** the family heading rows and the indented run rows are retired under By
  family. Each family is a node-link view: one node per run under the exact parent file it records; a node's body
  selects its run, its arrow only folds the runs under it, and the run's saved files are in Details. Newest first and
  the Diagnosis drawer keep the tree above. See the round README §12, 2026-11-21 ("By family, round 1" and "round 2"),
  and UI_LEDGER OPS-SL-GI. The text above is kept as T01's record.

## D04 The Diagnosis "Change model" drawer (Load for inference)

Files:
- `D04_drawer_1850x960.png` and `D04_drawer_1850x960_details-scrolled.png`
- `D04_drawer_760x1000.png` and `D04_drawer_760x1000_details-scrolled.png`

Choices:
- **Same renderer, clinical preset.** The drawer uses the same tree renderer as Models. Base runs are not
  loadable, so they are not rows; the family heading names them.
- **Head row reorganised.** Workspace, Task segmented control and the Compare link come first, then the workspace
  line, then Search with the count line. Today the count line is squeezed into a slot 195 px wide; at 760 px it is
  54 px wide and 168 px tall.
- **Wide (1850×960).**
  - The list takes 60% and the details 40%, each its own scroll region (`overflow: auto`,
    `overscroll-behavior: contain`).
  - The footer with Load stays under both. Measured: Load at y 861–901.
  - Scrolling the details to the end leaves the list top (301.6) and Load (861) where they were.
- **Narrow (760×1000).**
  - When opened, the drawer is one screen tall and scrolled into view. The list region (274 px, content 495 px)
    sits over the details region (252 px, content 649 px). Each scrolls on its own, and Load stays in view at
    y 890–930.
  - This is a deliberate exception to style.py's narrow rule ("nothing scrolls inside"), because D04 and UI-07 ask
    for it. The two regions sit one above the other, not nested; the page scrolls only outside the drawer.
- **Sticky list header.** The list's header row sticks inside its region, with `scroll-padding-top`. A test that
  scrolled each file row under the header and focused it found the row itself on top every time
  (`D04_drawer_focus_check.json`).
- **Today, for comparison** (measured on the real drawer):
  - At 1850 the list and the details share one scroll region: 566 px tall for 1028 px of content.
  - At 760 the drawer is 2064 px tall and Load is about 2.0k px below the top of the list.

## D05 Ordering pending

Files: `D05_models_order-pending_1850x960.png`, `D05_models_order-pending_760x1000.png`.

- The count line's place shows "Applying By family…" with the app's spinner, in the info colour (the skill keeps
  the spinner for running states only), as `role="status"`.
- The Order control already shows the new choice. The list keeps the previous order (Newest first: run summaries,
  files folded) until the new one is drawn. Nothing on the page is locked.

## D06 / D07 Settings › Data locations

Files (each at `_1850x960` and `_760x1000`):
- `D06_settings_first_start`, from the real unconfigured server (8872).
- `D06_settings_first_start_models-page`: the Models page in the same state. This is the companion that shows the
  dashboard works and how a waiting page points to the panel.
- `D06_settings_configured`
- `D06_settings_overridden`
- `D06_settings_pending`
- `D06_settings_unreadable`
- `D06_settings_readonly`, from the real not-M1 server (8873). Its chip and its Runtime settings panel are the
  app's own.

Placement: a new panel beside the Runtime settings panel, on its left at wide (55 : 45) and above it at narrow.
Its intro line: "These are folders on the server machine: this computer when SHEPHERD was started from its desktop."

Each row:
- **Left:** the plain name (Workspaces folder, Results registry, Import folder, Training evidence folders) and what
  it is for and who needs it. This is how the panel "says which features wait for which folder".
- **Right**, as a key-value list:
  - **In effect:** the value, or ○ Not set. The source is on the same line: Saved in Settings, or Set when the
    service started (environment).
  - **Saved:** only when it differs from what is in effect, or is pending.
  - **Change to** (or **Set to**): a 39 px path box with Check and Save, the same height as the buttons (D03).
  - **Create suggested folder:** only for an unset required folder. The button names `~/SHEPHERD-data/<name>`;
    the full path is in its title.
  - **Advanced: source ›:** one level deep. It shows the `SHEPHERD_*` start-up setting and whether it is set, the
    D07 file `~/.config/shepherd-advanced/deployment.json`, and the order "start-up setting, then the saved
    value". The variable names appear nowhere else.
- The evidence row is a list, with Remove beside each folder and "Add folder" for the box.
- The optional row's empty state says what that means: "None set: the check reads the measured workspace only".

The states:
- **First start (nothing set).** One neutral line: "○ Nothing is set yet. Each folder is needed only by what its row
  names; Settings and every other page open as usual." There is no wizard and no page lock: the header and every
  tab work. The Models page says in plain words "○ No workspaces folder is set, so no workspace or model is listed
  here yet", with one action, "Set one in Settings › Data locations". Today the same place names
  SHEPHERD_WORKSPACES_ROOT.
- **Configured.** Everything is saved in Settings. One new path has been checked: "● Usable: an existing folder
  outside the app folder; this service can read it".
- **Overridden at start.** The registry is "Set when the service started (environment)". Its Saved line reads
  "Saved, applies when the start-up setting is removed". The box and Save stay usable because they change only the
  saved value. Advanced is open and shows SHEPHERD_REGISTRY.
  - I read "input disabled for effect" as "the box cannot change what is in effect", and said it on the Saved line
    rather than greying the box, because saving must stay allowed.
- **Pending while a job runs.** The GPU chip reads "GPU job: training (job runner)". The workspaces row's Saved line
  reads "▲ Saved, applies when the running job ends", and Apply is off.
  - Semantics I chose (plan §4.4): Apply applies now through the existing restart path when nothing runs. While a
    job runs, Apply is off and the saved value takes effect when the job ends.
  - The no-job case ("Saved, not in effect yet", with Apply on) is not drawn.
- **Unreadable settings file.** One red ✕ notice, because it blocks saving: "Settings file cannot be read: the
  folders saved in it are unknown and not used; Save is off until the file is replaced. Folders set when the
  service started still apply."
  - It has one action, "Replace the unreadable file (a copy is kept)", and names the file.
  - The rows show only what is in effect. Save is off, and Create is hidden because it would save.
  - The fact that Save is off is stated once, in the notice, not on every row.
- **Read-only.**
  - Values read "Set (path hidden)"; the sources stay visible.
  - Every box and button is disabled, and every button shows 🔒.
  - One reason line sits at the top of the panel, in D8's words, the same as the Runtime footer: "🔒 Operator
    controls only: deployment mode not set to M1. Folder paths are shown to operator controls only."

## Self-check against the WebUI skill

These were measured in the page by `common.cjs` `CHECK` and by each script; the numbers are in `shots/*.json`.

- **14 px floor:** the smallest visible text is 14 px in every screenshot. The check walks every visible text node.
- **Text 4.5:1:**
  - The lowest measured value is 5.4:1 (white on the accent primary button).
  - Other low values: disabled button text 6.26:1, captions 6.13:1, context rows 8.9:1, the check result word
    7.54:1, ▲ 7.77:1, ✕ 6.63:1 on red-50.
  - Excluded: the app's existing " · " separator (#cdd2da, 1.52:1). It is `aria-hidden` decoration, not text that
    informs.
- **Non-text 3:1 for the cues that identify a control:**

  | Cue | Measured |
  |---|---|
  | Tab hover line | 3.89:1 on its fill, 4.32:1 on the bar |
  | Selected tab line | 5.4:1 |
  | Focus ring | 6.89:1 (6.08:1 on accent-50) |
  | Path and search box border | 4.83:1 |
  | Selected tree row: 4 px bar plus fill | 4.77:1 on accent-50 |
  | Tree toggle: the glyph is the cue | 8.9:1; hover border 4.32:1 |
  | Match underline (amber-800) | 7.77:1 |

  - The box border's computed colour is #6b7280, not the grey-500 token (#737a87, 4.32:1). Both pass. Which
    variable wins should be pinned in T04.
  - Not held to 3:1: disabled read-only boxes (1.37:1, inactive components are exempt under WCAG 1.4.11 and they
    carry the lock and the reason), row dividers (decoration), and hover-versus-rest differences (not required,
    plan §6).
- **One scroll region per area:**
  - Navigation: none.
  - Models at 1850: none is needed, because the rows region (the designated one) fits.
  - Drawer: the list and details regions only, side by side or stacked, never nested.
  - Settings at 1850: nothing scrolls in any state, after compacting; the panel is 626–737 px in the 881 px
    available.
  - Narrow pages flow, apart from the drawer exception explained above.
  - No state scrolls horizontally.
- **Nothing covers a focused control:**
  - `elementsFromPoint` at the centre and two corners of Load, Cancel, the search box and the path boxes returns
    the control itself.
  - Captions never overlap the tabs.
  - The drawer's sticky list header does not cover a focused row (see the drawer section).
- **Each fact once:**
  - The selected file's 12-hex fingerprint appears in its list row and in the details. Reason: the list is where a
    fingerprint search matches it; the details carry the full value and Copy.
  - The family heading names its root by relation only.
  - The read-only reason appears once per panel, beside what it disables.
  - "Save is off" is said once, in the unreadable notice.
  - The override's precedence is said on the Saved line, and the general order once, under Advanced.
- **Labels, not chips:** field keys (In effect, Change to, Search, Workspace) and the captions are plain text with
  no background. States use the one status mark (shape plus word). No new chip or tag was added; the header chips
  and the "synthetic" / "record verified" tags are today's.
- **Targets of at least 24 px:** tree toggles and the clear button are 24×24, Remove is 28 px tall, row buttons are
  39 px, and Advanced is a 24 px line.

## Findings and open points, recorded and not fixed here

- **F1:** today's Models page drops the selection when the Task filter excludes it (observed: the details reset to
  "Select a model…"). D04 wants it kept and noted (T05).
- **F2:** today's drawer:
  - The list and the details share one scroll region.
  - At 760 the Load button sits about 2.0k px below the list.
  - The head row squeezes the count line (UI-07, T05).
- **F3:**
  - The first-start GPU chip still reads "refused, lock not configured".
  - Its ⓘ names SHEPHERD_WORKSPACES_ROOT.
  - Plain-words follow-up for T03/T06; not drawn here.
- **F4:** the input border token: see the 3:1 note above (T04).
- **Not drawn:**
  - The "● In use" mark for a served file (written in `tree.cjs`; the demo serves nothing).
  - A failed Check ("✕ Not usable: …").
  - The no-job pending state.
  - The 1920×1080, 1536×760 and other sizes; those belong to T04/T08's measurements.
