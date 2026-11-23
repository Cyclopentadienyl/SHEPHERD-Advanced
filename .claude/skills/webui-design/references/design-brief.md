# Design brief

The shared taste behind the WebUI's rules, written for the 2026-11-22 UI sweep (the round README's §12) and kept for
every later change and its review. **The skill (`../SKILL.md`) wins on any conflict**; read this with it and with
`sources.md`. The owner asked that the work follow the logic of the great product designers (Jony Ive's work at Apple,
Dieter Rams' "less, but better", Apple's Human Interface Guidelines: clarity, deference, depth), and on 2026-11-23
that the aesthetic standard match Jony Ive's and the UI skills' standards in every arrangement, a stacked portrait
window included. Apply it as judgement, not decoration.

## The stance

1. **Simplicity is clarity, not emptiness.** Every element earns its place by helping the operator do the task in
   front of them. Remove what does not; never add ornament to make a screen look busy or "designed".
2. **The content leads; the interface defers.** Data, fields and the one next action are the subject. Chrome
   (borders, boxes, tints, icons) stays quiet so the content reads first. One container level per region.
3. **Honest and calm.** Say what is true now, in plain words, once. A refusal says the consequence and the next
   step; the engineering detail is one click away (Details ›), never a wall of text in the flow.
4. **Care in the details nobody asks for.** Edges align to one grid, spacing follows one scale, text sits on one
   left edge per panel, controls of one kind look and behave alike everywhere. Inconsistency reads as broken.
5. **Hierarchy by position, size and weight, not by volume.** One primary action per area, clearly first; secondary
   actions quieter; the eye should land on the right thing within a second (squint test).
6. **Built to be resized.** A layout is a set of relationships, not a picture of one window. Groups keep together
   and wrap as a whole; content defines size (intrinsic sizing, flex-basis, grid auto-fit/minmax, container
   queries), not fixed pixels tuned to one full-screen landscape window. At any window at or above the display
   floor (skill §1.1: 1280 × 720 CSS px; 864 px for a window no wider than tall, which stacks) nothing overlaps or
   covers, nothing is squeezed to a sliver and nothing is clipped. Below the floor the layout is drawn as at the
   floor and the page scrolls; it is never compressed further, and nothing is built for such a window.

## In practice (this project)

- A label sits with its field; a field group sits apart from the actions. A long message never shares a row with
  fields or buttons: only a one-line relation sits beside the control it explains, and anything longer takes the full
  width under the group it explains (skill §2).
- Status: colour, shape and word together (skill §6). Quiet when normal.
- First use (nothing configured): each cause is said once per screen by the one notice, its GUI path as the link
  ("Choose one in Settings › Data locations"), the rest under Details ›; a control that depends on it says a short
  relation (skill §9). Never the same paragraph repeated in every column.
- Words: front-loaded, sentence case, short sentences; technical names (environment variables, lock files, guide
  sections) only under Details or in a Ref line.
- Typography: three sizes and two weights (skill §2); one family; code in the monospace token; no text under 14 px.
- Motion and colour are functional only.

## How to judge your result

Look at your own screenshots at the skill's check sizes (§12) as a designer would: does each screen look calm,
aligned and obviously usable? If a screen needs explanation to look right, it is not done.
