# Text layout brief

How WebUI text is arranged, help bubbles included, written for the 2026-11-22 UI sweep (the round README's §12) and
kept for every agent that writes, places or reviews WebUI text. **The skill (`../SKILL.md`) wins on any conflict**;
read this with it and with `design-brief.md`. It governs how text is arranged, not what it says: every fact of an old
text lands in the lead, a row, Details or the Ref line; governed texts (skill §13) keep their semantics. Numbers are
guidance, except the 14 px floor and the WCAG values.

## 0. Measured (IBM Plex Sans as shipped, Chromium, 2026-11-22)
- 320 px bubble (292 px text box, 15 px): about 39 characters per line (34–44), bold lead the same, list items 38.
  `60ch` at 15 px = 540 px, about 78 per line (max 82); `56ch` = 504 px, about 73 (max 78). Plex digits are equal width.
- 64 literal help texts: 22 leads over 78 characters (3 bubble lines), 2 over 90; 58 of 116 items over 76 characters
  (3+ lines). `help.budget_problems` then allowed a 100-character lead and 4 items; the sweep held every registered
  text to §11's budget, and since 2026-11-23 every drawn part of a bubble to two lines (§14, check 9); since
  2026-11-24 that count is a note, not a gate (skill §7, R-HELP).
- The owner's screenshots of 2026-11-22: 9–12-line refusals right-aligned in a narrow column beside Start; an env var,
  a lock file and a guide section inside sentences; one cause in 3 places; `.shepherd-gpu.lock` broken at its hyphen;
  text left of its label.

## 1. Message shape (reasons, refusals, notices, warnings, first use)
- **Lead:** one sentence of 20 words or fewer with the consequence (what cannot happen, or what is true now), then the
  next step or who can act, as a second short sentence or one button. Nothing else stays in the flow.
- **Why and how** go in term rows or behind `Details ›`; env vars, lock files, guide sections and codes only there or in Ref.
- **Errors and refusals:** what happened, why (only if known and short), what to do; never invent a cause. No "please",
  "sorry", "oops", "invalid", "an error occurred", "we", "Note:", "Warning:" (the status mark says it).
- **Build each message whole:** never glue a sentence from fragments around a variable; a variable fills a row value.
- **Placement:** a refusal stays inline beside what it dims (skill §3, §9), but beside the control stands only its
  cause's phrase (five words at most, opening the one place its whole reason is said) or one line in the page's words
  led by the need (R-WHY); anything longer takes the full width under the group it explains, from the group's left
  edge, capped by the measure. Never a narrow text column between fields.

## 2. Paragraphs and sentences
- One idea per paragraph, at most 3 sentences (skill §5). Running prose past 2 rendered lines at its measure gets
  structure (lead, then rows; skill §4), not a longer paragraph. Split where the actor, condition or consequence changes.
- Sentences of 20 words, 25 at most. No semicolons, in any text the WebUI writes, module constants and the texts its
  functions build alike (the owner, 2026-11-23): split into sentences or a list. One colon per sentence at most: a
  second colon is a second fact, so a second row. No parenthesis holding more than about 4 words.
- Paragraphs are separated by space (`--shep-space-para`), never by `<br><br>` or a first-line indent.

## 3. Lists
- 3–7 parallel items make a vertical list; 2 stay in a sentence (term rows excepted); numbers only for steps or ranks.
- A lead-in introduces it (in a bubble, the lead); never nest. Each item's first two words carry it; one grammatical
  form, lengths within about 2×. All fragments without a full stop or all sentences with one; capital first letter.
- Hanging indent: `list-style-position: outside`, the bubble's `padding-left: 18px`; wrapped lines align with the item's
  first word. Marker: the small default `disc` in `--shep-muted` (`::marker`); never ● ○ ▲ ✕ or `circle`, the status
  shapes (skill §6). An en-dash marker is the owner's call; do not switch.
- Definitions use term rows ("Term: definition", bold term, `TERM_PATTERN`); an object's facts use `.shep-kv` (skill §4).

## 4. One left edge, indents, alignment
- **One left edge per column of a region:** a field's label, its field, the text under it, reasons, notes and empty text
  start at one x (±1 px). Text about a field sits in that field's block (`info=` or the same padding), not in a sibling
  block with its own padding.
- **Indent only for subordination** (list text, a Details body, text after a status mark): one level in a bubble, two on
  a page. A status mark keeps its own column (`.shep-reason` flex: mark, then text), so wrapped lines align with the text.
- **Alignment:** `text-align: start` for anything that can wrap; never `justify`; right only for numbers in columns and
  one-line meta (stage hint). Centre a box, not lines (an empty plot's message box). `text-indent: 0`; `hyphens: manual`;
  `text-wrap: pretty` on paragraphs and bubble bodies, `balance` on leads and headings (progressive enhancement).

## 5. Measure
- Prose takes `max-width: var(--shep-measure)` = `56ch` (about 73 characters a line, 78 at most): `.shep-reason`,
  `.shep-note`, `.shep-notice`, empty and first-use text, Details prose, panel `p`, and (since 2026-11-23) a field's
  `info=` line and a muted line of its own. Exempt: tables, key-value rows, option lists, charts, header chips, code
  rows. `ch` follows each element's own size, so one token serves 14 and 15 px. Use `max-width`, never a fixed or px
  width. The bubble stays 320 px: count its text in lines of about 39 characters.

## 6. Tokens, defined once in `style.py` `:root` (the bubble's CSS in `help.py` reads the same ones)
```css
--shep-measure: 56ch;           /* prose blocks */
--shep-lh-prose: 1.45;          /* text that can wrap to 3+ lines (the bubble and .shep-reason) */
--shep-lh-compact: 1.3;         /* labels, one-line values, table cells: 2 lines or fewer */
--shep-lh-title: 1.3;           /* 20 px region titles */
--shep-space-para: 8px;         /* between paragraphs, and between a lead and its body */
--shep-space-item: 4px;         /* between list items and between term rows */
--shep-space-head-above: 16px;  /* above an in-panel heading */
--shep-space-head-below: 4px;   /* below it: a heading sits with its content */
```
- Unitless line heights. Order: item < paragraph < above heading, below < above (skill §2's proximity rule, on its
  scale). The bubble uses these tokens (6 px and 2 px before the sweep). No fixed `height` on a text box: text survives
  the WCAG 1.4.12 overrides (line height 1.5, paragraph spacing 2×, letters 0.12em, words 0.16em) without clipping or
  overlap.

## 7. Emphasis, capitals, headings in small panels
- Weight 600 is the only emphasis, for structure only: a lead, a term, a label, a heading. Never bold mid-sentence. No
  italics, no underline except links, no colour as emphasis (colour means status). Sentence case; capitals only in codes.
- A heading inside a panel is body size, 15 px/600 (or the 14 px/600 label style); only region titles are 20 px. At most
  two levels per region; no colon or period after it; no eyebrow above it; no background; front-loaded.

## 8. Code, paths, environment variables
- Monospace (`--shep-mono`) only for literal tokens (paths, env vars, file names, config keys, flags, digests), never for
  labels or prose, always with an explicit `font-size` ≥ 14 px (`code` and `.shep-full` already do).
- Never a token in a lead. In a body sentence at most one, of 30 characters or fewer; longer or more go to a key-value row
  with Copy, the Ref line or Details. The sentence names the thing in words ("the workspaces folder").
- Long tokens break only at their own separators and never look hyphenated: `renderers.token_html` escapes the token
  and breaks it only before a `/ . _ -` that follows a non-separator, each piece kept on one line with a `<wbr>` between
  pieces; a command-line option (a word that starts with "-") is one piece, so a command breaks at the space before it
  (since 2026-11-23). Never `word-break: break-all` on text with words (`.shep-full` on a lone hex string is fine);
  never U+200B (copied with the path). Bubbles use `textContent`: keep Ref tokens short. In a narrow column prefer a
  middle ellipsis plus Copy.

## 9. Numbers
- `font-variant-numeric: tabular-nums` on numeric cells, value tiles, live counters; default figures in sentences.
  Numerals, not words; units once; right-aligned columns (skill §5). Keep number formats: separators are out of scope.

## 10. Empty states and first use
- Every empty region says what will appear and the next step in 1–2 sentences, own words per cause, at most one action
  (skill §9); never blank. First use: each cause once per screen through the one notice, where it empties a region, its
  GUI path as the link ("Choose one in Settings › Data locations."); elsewhere on that screen each control it dims has
  the notice's phrase ("Needs a workspaces folder", `renderers.reason_phrase`), or nothing where the notice stands in
  the same bar and names them; a dimmed control's own cause is one line led by the need ("Choose a dataset to start.",
  `renderers.need_words`), never "disabled" (skill §9); no env vars, file names or defaults.

## 11. Help bubbles (ⓘ)
Grammar as skill §7 (`help_lines`): line 1 the lead (bold); `- ` items; `Term: ` rows; other lines paragraphs; `Ref:` last.

| Part | Rule |
|---|---|
| Lead | Answers the question at the ⓘ (what it is, what it changes); starts with the answer or a verb; no repeat of title or label. ≤ 90 characters (skill ceiling); aim ≤ 78 = 2 lines at 320 px. |
| Body | 2–4 parallel, front-loaded items or term rows, each ≤ 2 lines (76 characters), ideally 1 (38); or one paragraph of ≤ 2 sentences instead. Four items where the facts need them: never merge two facts to reach three, never pad to two. |
| Ref | Last, muted, 14 px: only codes, file names, env vars, guide sections, sources. |
| Total | 60–80 words without Ref; never truncated; no nesting; nothing must-know (formats, units, refusals, states stay visible). |

Restructuring a long text, in order:
1. Name the reader's question at that ⓘ; rewrite line 1 as its answer in 78 characters or fewer.
2. Sort the other facts: must-know to visible text (`info=`, reason line); explanation to items; identifiers and sources
   to Ref; mechanism the reader cannot act on to Details or the guide (say so in the delivery).
3. One fact per item: ";", two colons, a parenthesis or "and" between two facts makes two items or a term row. One
   form for all items (all term rows, all fragments, or all sentences).
4. Run `budget_problems`, then look at the bubble at 320 px: lead and every item 2 lines or fewer, as a reminder. It is
   accepted when it covers no control, reads at 4.5:1 and keeps every fact reachable (skill §7, R-HELP).

Example `finetune-best-save` (illustrative; check each fact against the code). Before: 91-character lead, items of 2–3
lines with colons and a parenthesis:
> The child's saved file with the best value of its recorded selection metric; ties share it.
> - Its metric and direction are named under the file: a task model selects on validation pooled MRR, higher is better.
> - It is the Trainer's figure on the val samples, not a test result: test the model on Research › Test.
> - Shown to the operator once the child has ended (reading it opens the run's checkpoints).

After: 63-character lead, one form (term rows), 43 words; only the parenthesis (mechanism) is dropped:
> The child's saved file with the best recorded selection metric.
> - Metric: validation pooled MRR, higher is better
> - Source: the Trainer's figure, not a Research › Test result
> - Ties: several saved files can share it
> - Shown: to the operator once the child has ended

## 12. Worked example: the refusal beside Start (the owner's screenshot, Training and Fine-tune)
Before: 64 words beside the button; the cause also under Dataset and in the header chip.
> Start disabled — the GPU lock cannot be used (No workspaces folder is set, so the GPU lock is not configured and every
> managed GPU job is refused rather than run without excluding the others. Choose the folder in Settings › Data locations:
> the lock is .shepherd-gpu.lock in it, the one file every process that uses the folder takes (deployment guide §4.11).
> Ref: SHEPHERD_WORKSPACES_ROOT).

After (since 2026-11-24, R-WHY and R-DETAILS): the screen's one first-use statement, in Training's run bar under
Dataset and Run name, in the notice's form: "○ No workspaces folder is set. Training needs one. Choose one in
Settings › Data locations" with `Details ›` closing the line. Its Details holds the folder's rows and, merged, the GPU
lock's ("GPU lock: `.shepherd-gpu.lock` in the workspaces folder", "Taken by: every process that uses the folder",
"GPU jobs: refused while it is unset, never run without the lock"), then one Ref line (`SHEPHERD_WORKSPACES_ROOT` ·
deployment guide §4.11). Beside Start, nothing: the notice is in the same bar and names Start and Continue…, so it is
their description. On Fine-tune the notice stands at the top of the form, away from Start, so Start has the phrase
"Needs a workspaces folder", which brings the notice into view with the focus on its link.

## 13. Do not
- First-line indents, `<br><br>` spacing, justified text, centred or right-aligned text that wraps, fixed text heights;
  all caps beyond codes, italics, bold mid-sentence, underlined non-links, colour as emphasis.
- Walls of text (over 3 sentences, or prose past 2 lines); a message over one line beside a button; the same message
  in several columns or panels of one screen (skill §1.2); env vars, paths or codes in sentences; U+200B; `break-all`.
- Semicolons, two colons in a sentence, clause-long parentheses, " · " chains, a dash joining two facts in a lead, a
  colon after a heading, ● ○ ▲ ✕ as list markers, nested lists, 2-item bullets, numbers that are not steps.

## 14. Checks (judgement first, then the scan; folded into `ui_problems`, `layout_problems`, `repeat_problems`, `budget_problems`)
Reviewer, before any script: blur a screenshot, then read only the leads of each panel and bubble; if the leads alone do
not say what is true and what to do next, the text is not done. Then the scan: 1–5, 7, 12 pass or fail for text the
change touches; 6, 8's "> 78" note, 9, 10, 11 list findings for the delivery. Findings outside the task are recorded, not
fixed. No new gate.

| # | Check | Measured by | Pass |
|---|---|---|---|
| 1 | Measure | computed `max-width` of prose, or characters per rendered line (`Range.getClientRects`) | ≤ 56ch, or average ≤ 75 and max ≤ 80 |
| 2 | Left edges | per region column: `left` of each label, help, reason, note, empty text (markers, mark columns excluded) | one value ±1 px |
| 3 | Alignment | computed `text-align`, `text-indent`, line count | no `justify`; `center`/`right`/`end` only on one-line text and numeric cells; indent 0 |
| 4 | Line height | `line-height / font-size` of elements of 3+ lines | ≥ 1.4; no px line height on wrapping text |
| 5 | Spacing order | gaps of items, paragraphs, headings | item < paragraph < above heading; below < above |
| 6 | Paragraph size | sentences and rendered lines per paragraph or reason | ≤ 3 sentences; prose ≤ 2 lines, else structured (skill §4); sentences ≤ 25 words |
| 7 | Tokens in prose | prose outside Ref, Details, `code`, key-value rows: `SHEPHERD_\w+`, `§\d`, `[\w.-]+\.(lock\|json\|pt)\b`, `/\w+/`; mono size; `word-break` | none; mono ≥ 14 px; no `break-all` on mixed text |
| 8 | Bubble strings | `help_lines` over every registered text (`budget_problems`) | lead ≤ 90 (note > 78); 0 or 2–4 items, each ≤ 76; ≤ 80 words; one item form; Ref last |
| 9 | Bubble render | every registered text and every header chip's bubble at 320 px (`test_webui_help.py`, since 2026-11-23; a note since 2026-11-24, R-HELP) | listed, not asserted: the lead, each paragraph, the Ref line and each item ≤ 2 lines, parts kept verbatim from a source or governed named as exempt; nothing clipped |
| 10 | Repeats | normalised sentences of 6+ words in one screen's visible text | each once |
| 11 | Words | regex over UI strings | no `please\|sorry\|oops\|invalid\|an error occurred\|\bwe\b`; no `;` |
| 12 | Figures | computed `font-variant-numeric` of numeric cells, tiles, counters | includes `tabular-nums` |
| 13 | Text spacing (broad layout changes only) | WCAG 1.4.12 style sheet injected | no `scrollHeight > clientHeight` outside scroll regions; no overlap |

## Sources (each opened during the sweep, 2026-11-22)
- Apple HIG: https://developer.apple.com/design/human-interface-guidelines/writing · https://developer.apple.com/design/human-interface-guidelines/offering-help · https://developer.apple.com/design/human-interface-guidelines/layout · https://developer.apple.com/design/human-interface-guidelines/typography · https://developer.apple.com/design/human-interface-guidelines/design-principles — Ive: https://www.apple.com/newsroom/2013/06/10Apple-Unveils-iOS-7/ · https://stripe.com/sessions/2025/a-conversation-with-sir-jony-ive — Rams: https://www.vitsoe.com/us/about/good-design
- Butterick: https://practicaltypography.com/summary-of-key-rules.html · https://practicaltypography.com/line-length.html · https://practicaltypography.com/line-spacing.html · https://practicaltypography.com/space-between-paragraphs.html · https://practicaltypography.com/first-line-indents.html · https://practicaltypography.com/bulleted-and-numbered-lists.html · https://practicaltypography.com/headings.html — Butterick (cont.): https://practicaltypography.com/bold-or-italic.html · https://practicaltypography.com/all-caps.html · https://practicaltypography.com/monospaced-fonts.html · https://practicaltypography.com/alternate-figures.html · https://practicaltypography.com/centered-text.html · https://practicaltypography.com/justified-text.html
- Bringhurst via Rutter: http://webtypography.net/2.1.2 · http://webtypography.net/2.1.3 · http://webtypography.net/2.2.1 — WCAG: https://www.w3.org/WAI/WCAG22/Understanding/visual-presentation.html · https://www.w3.org/WAI/WCAG22/Understanding/text-spacing.html — Chrome: https://developer.chrome.com/docs/modern-web-guidance/guides/visual-design/typography.md.txt
- MDN: https://developer.mozilla.org/en-US/docs/Web/CSS/overflow-wrap · https://developer.mozilla.org/en-US/docs/Web/CSS/word-break · https://developer.mozilla.org/en-US/docs/Web/CSS/text-wrap · https://developer.mozilla.org/en-US/docs/Web/CSS/font-variant-numeric · https://developer.mozilla.org/en-US/docs/Web/CSS/list-style-position · https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/wbr — Meyer: https://meyerweb.com/eric/thoughts/2010/02/12/fixed-monospace-sizing/ — Carbon: https://carbondesignsystem.com/elements/typography/type-sets/ · https://carbondesignsystem.com/patterns/forms-pattern/ — Atlassian: https://atlassian.design/foundations/content/designing-messages/error-messages · https://atlassian.design/foundations/content/designing-messages/empty-state · https://atlassian.design/components/tooltip/usage
- USWDS: https://designsystem.digital.gov/components/typography/ — GitLab: https://design.gitlab.com/components/popover — Shopify: https://shopify.dev/docs/apps/design/content/grammar-and-mechanics.md — Microsoft Style Guide: https://learn.microsoft.com/en-us/style-guide/scannable-content/lists · https://learn.microsoft.com/en-us/style-guide/punctuation/semicolons · https://learn.microsoft.com/en-us/style-guide/developer-content/formatting-developer-text-elements — GOV.UK: https://guidance.publishing.service.gov.uk/writing-to-gov-uk-standards/style-guides/a-to-z-style-guide/ · https://design-system.service.gov.uk/components/error-message/
- NN/g: https://www.nngroup.com/articles/inverted-pyramid/ · https://www.nngroup.com/articles/chunking/ · https://www.nngroup.com/articles/presenting-bulleted-lists/ · https://www.nngroup.com/articles/error-message-guidelines/ · https://www.nngroup.com/articles/empty-state-interface-design/ — Agent skills: https://raw.githubusercontent.com/pbakaus/impeccable/main/.claude/skills/impeccable/SKILL.md · https://raw.githubusercontent.com/jakubkrehel/skills/main/skills/better-typography/SKILL.md · https://raw.githubusercontent.com/jakubkrehel/skills/main/skills/better-writing/SKILL.md
