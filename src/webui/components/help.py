"""
The shared ⓘ: one help control for every ordinary help text (M5-2, UI-A).
=========================================================================
`docs/working/PLAN_TEST_RESULTS.md` §6, "M5's boundaries" (layout and help):
an ordinary field's help sits behind one shared "i" control that works with the
mouse, the keyboard and a click, can be dismissed, and stays readable when the
pointer moves into the bubble. A refusal's reason, a job's state, the synthetic
marker, an unverified source and the facts of Diagnosis' Reference cohort
select stay visible short text, never behind it.

**Which design, and why: H-B, a registry by ``elem_id`` and one shared control**
(the M5 maps offered H1, an overlay over Gradio's ``info=``, and H-B). Checked
in Gradio 6.30's DOM before choosing:

- 6.30 renders ``info=`` as a ``div.info-text`` inside the field's ``<label>``
  (or ``<fieldset>``), so the help text is part of the field's accessible name
  and its look is Gradio's. H1 would hide that div and rebuild it from
  Gradio's internal markup and sibling order;
- a state-bearing text set at runtime travels in ``info=`` (Diagnosis' Reference
  cohort select: the chosen version's fingerprint and whether it was mapped
  against the served graph, or why it cannot be read), and H1 would have to
  except it;
- with H-B the help texts live in Python, escaped once and tested like any other
  value; ``info=`` is left only to the texts that must stay visible; and the one
  page assumption is the component's ``elem_id`` and its label's title element.

**How it works.** Each page builder registers its help through the page
context (`HelpRegistry.field` for a component, `HelpRegistry.topic` for a ⓘ
inside rendered HTML). The shell renders every registered text once, escaped,
into one hidden container (`HelpRegistry.texts_html`), and the one head script
(`HELP_HEAD`, given to ``gr.mount_gradio_app``):

- adds a ``<button type="button">`` after the title of each registered field as
  the field mounts (a ``MutationObserver``: lazily mounted tabs and accordions
  included), with ``aria-expanded`` and ``aria-describedby`` naming the help
  text; the field's own input is labelled by its title alone and described by
  its visible ``info=`` line, when it has one, then the help text (`describe`,
  kept on every pass, so an ``info=`` set or drawn again at runtime stays
  part of it), and the ``<label>`` names its input by id, so a click on the
  title still focuses the field (a button inside a label would otherwise
  become its control);
- shows one shared bubble, a toggletip (the WebUI skill's §7), by
  document-level delegation, so a ⓘ inside an HTML value that is replaced
  later keeps working: a hover previews it after a short delay
  (`HOVER_OPEN_MS`) and keyboard focus at once, both as a ``tooltip``; a click
  (or Enter or Space) pins it open as a ``dialog`` named by its title; Esc and
  a click outside close it; it stays open while the reader's pointer moves
  in the bubble (a bubble that only came under a resting pointer, as it
  follows its ⓘ on a scroll, keeps no preview open);
- lets the pointer through a preview the pointer did not open (focus opened
  it): the reader's pointer moving over it turns its pointer events off, so
  a click goes to the control under it, and only its focus keeps or ends it
  (a hover preview stays the pointer's, to be read on, WCAG 1.4.13; a pinned
  bubble is a dialog); the wheel over it still scrolls its text where its
  height is capped, then the page past the text's end;
- does not draw a bubble whose ⓘ is out of view (outside the window, or
  outside the visible part of a region it scrolls in: as Floating UI's
  ``hide`` reads its reference's clipping ancestors) and lets clicks reach
  what is under it; it stays open and is drawn again where it belongs when
  its ⓘ is back in view. Focus inside it then moves to its ⓘ without
  scrolling; Esc closes it and still reaches the page. Closing never scrolls
  the page: focus returns to a ⓘ with ``preventScroll``;
- places it so it **never covers the field it explains** (the field's
  component, by its ``elem_id``; else the table row, the key-value row's key
  and value, or the line the ⓘ sits in): beside that field, then above it, then below it; of the sides
  that fit the window without overlapping it, the one that lies over the fewest drawn controls (R-HELP:
  a bubble covers no control), the first in that order among equals; where each of those lies over a
  control, the tallest place that lies over none with its height capped there (never shorter than its
  head, its lead and two lines after it; Floating UI's ``size``), then the same at a narrower width
  (220 px at the least) where none of its width is clear, and where none is clear at any size, of
  every place tried the one over the fewest controls; failing all, narrower beside it, or on
  the side with more room with its height capped; a capped text scrolls
  inside under the scroll cue (``data-shep-more``); at most 320 px wide (`BUBBLE_MAX_WIDTH`), the ⓘ's hit target at
  least 24 × 24 px;
- fills the bubble by the help text's grammar (`help_lines`, the WebUI
  skill's §7): its first line is the lead, shown bold; a line that starts
  with "- " is an item of one list; a line led by a short term and ": "
  shows the term bold ("Default: 0"); a trailing "Ref: …" line holds the
  codes and source references, set apart; any other line is a paragraph.
  Every part is an element made with ``createElement`` and filled with
  ``textContent``: no help text is ever parsed as HTML in the browser;
- a ⓘ that is not a field's, a table row's or a key-value row's and sits
  inside a region marked `REGION_CLASS` (a card it heads) places its bubble
  outside that whole region when one of the four sides fits, else outside
  the line it sits in, as any other ⓘ;
- a field in a container marked `KEY_FIRST_CLASS` is drawn as its key, its ⓘ
  and its control, in that order (Diagnosis' Options): its control is kept
  after its ⓘ in the page too (Gradio puts a checkbox's box before its
  words), so focus and a screen reader meet them in the order drawn (WCAG
  2.4.3, 1.3.2);
- closes by one control at the right end of its head, beside its title: the
  ×, and on a pinned bubble (a dialog) the framed Esc keycap beside it, both
  one button (``aria-keyshortcuts="Escape"``); a preview shows the × alone.
  No row under the text says it: that room is the text's.

The same script copies a full value from a ``button.shep-copy`` (``data-copy``)
to the clipboard: the shared renderers show 12 hex digits and copy all 64.

Module: src/webui/components/help.py
"""
from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

#: What an entry's key may hold: it becomes part of an HTML id.
_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")

#: The prefix of a help text's element id (``aria-describedby`` names it).
TEXT_ID_PREFIX = "shep-help-text-"

#: **One description, several writers** (R-WHY beside the field help): a control's
#: ``aria-describedby`` is written by this script (a field's info line and its help
#: text, ``describe``) and by the shell's (a dimmed control's phrase or notice,
#: ``describeDimmed``). Each writer records the ids it put there in an attribute of
#: its own (`mark`) and adds or takes back only those, every other id kept in its
#: place: a writer that replaced the whole attribute dropped the other's ids (the
#: last one to run won). Both scripts include this one function.
DESCRIBED_BY_JS = r"""
  function describedBy(el, mark, want) {
    var had = (el.getAttribute(mark) || '').split(/\s+/).filter(Boolean);
    var next = (el.getAttribute('aria-describedby') || '').split(/\s+/).filter(function (x) {
      return x && (had.indexOf(x) < 0 || want.indexOf(x) >= 0);
    });
    for (var i = 0; i < want.length; i++) { if (next.indexOf(want[i]) < 0) { next.push(want[i]); } }
    var ids = next.join(' '), mine = want.join(' ');
    if (!ids) { el.removeAttribute('aria-describedby'); }
    else if (el.getAttribute('aria-describedby') !== ids) { el.setAttribute('aria-describedby', ids); }
    if (!mine) { el.removeAttribute(mark); } else if (el.getAttribute(mark) !== mine) { el.setAttribute(mark, mine); }
  }
"""

#: The bubble's widest (CSS px; the WebUI skill's §7 budget), the delay before
#: a hover opens its preview (ms), and the ⓘ's smallest hit target (CSS px,
#: the skill's §10 floor).
BUBBLE_MAX_WIDTH = 320
HOVER_OPEN_MS = 300
HIT_TARGET = 24

#: The class of a region whose heading's ⓘ explains the whole region (a
#: card): its bubble is placed outside the region, not only outside the line
#: the ⓘ sits in (`fieldRect`'s ``regionRect``).
REGION_CLASS = "shep-help-region"

#: The class of a container whose fields are drawn as rows of their key, their
#: ⓘ and their control, in that order (Diagnosis' Options): the help keeps
#: each field's control after its ⓘ in the page (`keyFirst`).
KEY_FIRST_CLASS = "shep-key-first"

#: **A help text's grammar**, read the same way by `help_lines` and by the
#: bubble's script (the patterns are given to it as written here): the first
#: line is the lead; a line starting with `BULLET` is a list item; a line (or
#: an item) led by a short capitalised term, then ": ", shows the term bold;
#: a line starting with `REF_PREFIX` holds the codes and source references.
#: A blank line only separates.
BULLET = "- "
TERM_PATTERN = r"^([A-Z][^:.!?]{0,31}):\s+(\S.*)$"
REF_PREFIX = "Ref:"
_TERM = re.compile(TERM_PATTERN)


@dataclass(frozen=True)
class HelpLine:
    """One line of a help text as the bubble shows it: its `kind` ("lead",
    "item", "para" or "ref"), its `term` (shown bold, without its ": "), and
    the rest of its text."""

    kind: str
    text: str
    term: Optional[str] = None


def help_lines(text: str) -> Tuple[HelpLine, ...]:
    """A help text's lines by the grammar the bubble shows (`TERM_PATTERN`)."""
    out = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if not out:
            out.append(HelpLine("lead", line))
            continue
        kind = "para"
        if line.startswith(BULLET):
            kind, line = "item", line[len(BULLET):].strip()
        elif line.startswith(REF_PREFIX):
            kind = "ref"
        found = _TERM.match(line)
        out.append(HelpLine(kind, found.group(2), found.group(1)) if found
                   else HelpLine(kind, line))
    return tuple(out)


def sentence_items(text: Optional[str]) -> Optional[str]:
    """**A governed note laid out for the bubble, its words unchanged**
    (MODE_RESULT_NOTES; the owner of 2026-11-24: keep its meaning and limits,
    adjust its layout first): a text written as one paragraph (a note shown
    elsewhere as a line), its first sentence the lead; two to four further
    sentences the bubble's one list (`BULLET` items, the budget's 2-4), one
    further sentence a paragraph under the lead, and more than the list holds
    a sentence a line. Each sentence keeps its words and its full stop. None
    stays None."""
    if text is None:
        return None
    lead, *rest = [part for part in _SENTENCE_END.split(text.strip()) if part] or [""]
    if 2 <= len(rest) <= TEXT_BUDGET.items:
        rest = [f"{BULLET}{part}" for part in rest]
    return "\n".join([lead, *rest])


@dataclass(frozen=True)
class HelpBudget:
    """The WebUI skill's §7 budget, as the test over every registered text
    reads it: a lead of about 90 characters (`lead_chars` the most), a few
    items, about 60-80 words in all without the Ref line, no sentence over
    25 words, and internal codes (`CODE`) only in the trailing Ref line.

    **The text layout's budget** (`TEXT_BUDGET`; the UI sweep of 2026-11-22,
    its text brief's §11 and §14 #8) adds what a 320 px bubble draws in
    lines of about 39 characters: a lead of two lines (`lead_note`: a
    longer one is noted, `budget_notes`), none or 2-4 items (`items_min`),
    each of two lines at most (`item_chars`), all of one form
    (`one_item_form`: all term rows or none, all sentences with a full stop
    or all fragments without), and plain words (`plain_words`: no semicolon,
    no "please", "sorry", "oops", "invalid", "an error occurred" or "we")."""

    lead_chars: int = 100
    words: int = 80
    sentence_words: int = 25
    items: int = 4
    lead_note: Optional[int] = None
    item_chars: Optional[int] = None
    items_min: int = 0
    one_item_form: bool = False
    plain_words: bool = False


HELP_BUDGET = HelpBudget()
#: The text layout's budget (`HelpBudget`): every registered text is to meet
#: it once its page is laid out again (the sweep's lanes; enforced at their
#: integration, not before).
TEXT_BUDGET = HelpBudget(lead_chars=90, lead_note=78, item_chars=76, items_min=2,
                         one_item_form=True, plain_words=True)

#: Internal codes a help text names only in its trailing Ref line (the WebUI
#: skill's §5): setting names, decision, review and README codes, a source
#: commit or file, and a deployment mode's code.
CODE = re.compile(r"SHEPHERD_[A-Z_]+|\b[DR]\d{1,2}\b|\bREADME\b|@[0-9a-f]{7,}|"
                  r"\b\w+\.py\b|\bMode [A-Z]\b|\bM\d\b")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
#: Words a message never says (the text brief's §1 and §14 #11).
NOT_SAID = re.compile(r"\b(?:please|sorry|oops|invalid|an error occurred|we)\b", re.IGNORECASE)


def budget_problems(text: str, budget: HelpBudget = HELP_BUDGET) -> Tuple[str, ...]:
    """What in `text` is over the budget (`HelpBudget`), in words; none for
    a text within it."""
    lines = help_lines(text)
    problems = []
    if not lines:
        return ("no text",)
    if len(lines[0].text) > budget.lead_chars:
        problems.append(f"lead of {len(lines[0].text)} characters (at most {budget.lead_chars})")
    body = [line for line in lines if line.kind != "ref"]
    words = sum(len(f"{line.term or ''} {line.text}".split()) for line in body)
    if words > budget.words:
        problems.append(f"{words} words (at most {budget.words})")
    items = sum(line.kind == "item" for line in lines)
    if items > budget.items:
        problems.append(f"{items} list items (at most {budget.items})")
    for line in body:
        whole = f"{line.term}: {line.text}" if line.term else line.text
        for sentence in _SENTENCE_END.split(whole):
            if len(sentence.split()) > budget.sentence_words:
                problems.append(f"a sentence of {len(sentence.split())} words: {sentence[:40]}…")
        if CODE.search(whole):
            problems.append(f"a code outside the Ref line: {CODE.search(whole).group(0)}")
        if whole.startswith("•"):
            problems.append("a '•' bullet (items start with '- ')")
    refs = [i for i, line in enumerate(lines) if line.kind == "ref"]
    if refs and refs != [len(lines) - 1]:
        problems.append("a Ref line that is not the last line")
    listed = [line for line in lines if line.kind == "item"]
    if budget.items_min and 0 < len(listed) < budget.items_min:
        problems.append(f"{len(listed)} list item (none, or {budget.items_min} at least)")
    for line in listed:
        whole = f"{line.term}: {line.text}" if line.term else line.text
        if budget.item_chars and len(whole) > budget.item_chars:
            problems.append(f"an item of {len(whole)} characters (at most {budget.item_chars}): "
                            f"{whole[:40]}…")
    if budget.one_item_form and len(listed) > 1:
        if len({line.term is None for line in listed}) > 1:
            problems.append("items of two forms (term rows and plain items)")
        if len({line.text.rstrip().endswith((".", "!", "?")) for line in listed}) > 1:
            problems.append("items of two forms (sentences and fragments)")
    if budget.plain_words:
        for line in body:
            whole = f"{line.term}: {line.text}" if line.term else line.text
            if ";" in whole:
                problems.append(f"a semicolon: {whole[:40]}…")
            found = NOT_SAID.search(whole)
            if found:
                problems.append(f"a word a message never says: {found.group(0)!r}")
    return tuple(problems)


def budget_notes(text: str, budget: HelpBudget = TEXT_BUDGET) -> Tuple[str, ...]:
    """What in `text` is within the budget but worth a look (`HelpBudget`'s
    `lead_note`): a lead of three bubble lines."""
    lines = help_lines(text)
    if lines and budget.lead_note and budget.lead_note < len(lines[0].text) <= budget.lead_chars:
        return (f"a lead of {len(lines[0].text)} characters (three lines over "
                f"{budget.lead_note})",)
    return ()

#: The ⓘ glyph: an inline SVG, so no font or network is needed.
_I_ICON = ('<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true" '
           'focusable="false"><circle cx="6" cy="2.3" r="1.25" fill="currentColor"/>'
           '<rect x="5" y="4.5" width="2" height="6.2" rx="1" fill="currentColor"/></svg>')
_COPY_ICON = ('<svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true" '
              'focusable="false"><rect x="5" y="5" width="8.5" height="9" rx="1.6" '
              'fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M3.5 10.5V3.6'
              'c0-.6.5-1.1 1.1-1.1H10" fill="none" stroke="currentColor" stroke-width="1.5" '
              'stroke-linecap="round"/></svg>')


def text_id(key: str) -> str:
    """The id of the element holding `key`'s help text."""
    return f"{TEXT_ID_PREFIX}{key}"


def _check_key(key: str) -> str:
    if not isinstance(key, str) or not _KEY.match(key):
        raise ValueError(f"a help key is a letter then letters, digits, '_' or '-': {key!r}")
    return key


@dataclass(frozen=True)
class HelpEntry:
    """One help text. `target` is the ``elem_id`` of the component whose title
    gets the ⓘ (a field's help), or `None` for a topic that rendered HTML
    references by key (`help_button`)."""

    key: str
    title: str
    text: str
    target: Optional[str] = None


class HelpRegistry:
    """The help texts of one built dashboard, by key. One text per key: a key
    registered again with other words is refused (two texts for one field would
    make the one shown depend on build order)."""

    def __init__(self) -> None:
        self._entries: Dict[str, HelpEntry] = {}

    def _add(self, entry: HelpEntry) -> str:
        if not entry.title or not entry.text:
            raise ValueError(f"help {entry.key!r} needs a title and a text")
        known = self._entries.get(entry.key)
        if known is not None and known != entry:
            raise ValueError(f"help {entry.key!r} is registered twice with different texts")
        self._entries[entry.key] = entry
        return entry.key

    def field(self, elem_id: str, title: str, text: str) -> str:
        """The help of the component with this ``elem_id``: its ⓘ goes after
        the component's title. Returns the key (the ``elem_id``)."""
        return self._add(HelpEntry(_check_key(elem_id), title, text, target=elem_id))

    def fields(self, texts: Mapping[str, Tuple[str, str]]) -> None:
        """Several fields at once: ``{elem_id: (title, text)}``."""
        for elem_id, (title, text) in texts.items():
            self.field(elem_id, title, text)

    def topic(self, key: str, title: str, text: str) -> str:
        """A help text that rendered HTML shows behind `help_button(key, …)`."""
        return self._add(HelpEntry(_check_key(key), title, text))

    def topics(self, texts: Mapping[str, Tuple[str, str]]) -> None:
        for key, (title, text) in texts.items():
            self.topic(key, title, text)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def entries(self) -> Tuple[HelpEntry, ...]:
        return tuple(self._entries.values())

    def targets(self) -> Tuple[str, ...]:
        """The ``elem_id`` of every component that has field help."""
        return tuple(e.target for e in self._entries.values() if e.target is not None)

    def texts_html(self) -> str:
        """Every help text, escaped, in one hidden container: the elements
        ``aria-describedby`` names and the bubble reads (``textContent``)."""
        rows = []
        for entry in self._entries.values():
            target = (f' data-help-target="{html.escape(entry.target, quote=True)}"'
                      if entry.target else "")
            rows.append(
                f'<div id="{text_id(entry.key)}" data-help-key="{entry.key}" '
                f'data-help-title="{html.escape(entry.title, quote=True)}"{target}>'
                f"{html.escape(entry.text)}</div>")
        return f'<div class="shep-help-texts" hidden>{"".join(rows)}</div>'


def help_button(key: str, title: str, *, describedby: Optional[str] = None,
                extra_class: str = "") -> str:
    """A ⓘ for rendered HTML. It names the registered text of `key`, or, with
    `describedby`, the id of a hidden element the same HTML carries (a text that
    changes per request, such as the D8 chip's reason)."""
    target = describedby or text_id(_check_key(key))
    classes = "shep-i" + (f" {extra_class}" if extra_class else "")
    return (f'<button type="button" class="{classes}" data-help="{html.escape(key, quote=True)}" '
            f'data-help-title="{html.escape(title, quote=True)}" '
            f'aria-label="Help: {html.escape(title, quote=True)}" aria-expanded="false" '
            f'aria-describedby="{html.escape(target, quote=True)}">{_I_ICON}</button>')


def copy_button(value: str, what: str, *, icon_only: bool = False) -> str:
    """A Copy button that puts the full `value` on the clipboard (the shell's
    head script). `what` names it for a screen reader ("the full dataset
    fingerprint"); `icon_only` keeps the word for screen readers only."""
    classes = "shep-copy shep-copy-icon" if icon_only else "shep-copy"
    return (f'<button type="button" class="{classes}" data-copy="{html.escape(value, quote=True)}" '
            f'aria-label="Copy {html.escape(what, quote=True)}" title="Copy {html.escape(what, quote=True)}">'
            f'{_COPY_ICON}<span class="shep-copy-label">Copy</span></button>')


def registered_keys_in(rendered: str) -> Tuple[str, ...]:
    """The keys of every registered-text ⓘ a rendered HTML string references
    (for the test that every one exists)."""
    return tuple(re.findall(rf'aria-describedby="{TEXT_ID_PREFIX}([A-Za-z0-9_-]+)"', rendered))


# ---------------------------------------------------------------------------
# The browser side: one script, document-level delegation
# ---------------------------------------------------------------------------
_HELP_JS = r"""
(function () {
  if (window.ShepHelp) { return; }
  // The bubble's anchors: every ⓘ, and a dimmed control's phrase whose reason a
  // header chip holds (`renderers.reason_phrase`, R-WHY: the one bubble, its text
  // read by the chip text's document id, never a copy and never a second tooltip).
  var SEL = 'button.shep-i, button.shep-why[data-help]';
  function anchorByKey(key) {
    var all = document.querySelectorAll(SEL);
    for (var i = 0; i < all.length; i++) { if (all[i].getAttribute('data-help') === key) { return all[i]; } }
    return null;
  }
  // The bubble's width: at most MAX_W, at least LEAST_W, narrowed by WIDTH_STEP (`place`).
  var MAX_W = __MAX_W__, LEAST_W = 220, WIDTH_STEP = 20, HOVER_MS = __HOVER_MS__, GAP = 10, EDGE = 8;
  var REGION = '__REGION__', KEY_FIRST = '__KEY_FIRST__';
  var BULLET = __BULLET__, REF = __REF__, TERM = new RegExp(__TERM__);
  var bubble = null, anchor = null, anchorKey = null, pinned = false;
  var hideTimer = null, openTimer = null, previewFor = null;
  // `byPointer`: the pointer opened the preview (or came to its ⓘ); `yielded`:
  // a preview focus opened let the reader's pointer through (`claim`).
  var byPointer = false, yielded = false;
  // `anchorFocused`: the reader's focus is on the shown bubble's ⓘ, kept
  // while a drawing removes that ⓘ (`place` gives it to the new one).
  var anchorFocused = false;

  function closestHelp(node) {
    return node && node.closest ? node.closest(SEL) : null;
  }
  function textOf(btn) {
    var ids = (btn.getAttribute('aria-describedby') || '').split(/\s+/);
    for (var i = 0; i < ids.length; i++) {
      var el = ids[i] && document.getElementById(ids[i]);
      if (el) { return el.textContent; }
    }
    return '';
  }
  // A help text's grammar (help.py's help_lines): the first line is the lead,
  // "- " lines one list, a short "Term:" shown bold, a trailing "Ref:" line
  // set apart. Every part is made here and filled with textContent.
  function termInto(el, line) {
    var m = TERM.exec(line);
    if (!m) { el.textContent = line; return; }
    var t = document.createElement('strong');
    t.className = 'shep-bubble-term'; t.textContent = m[1] + ':';
    el.appendChild(t); el.appendChild(document.createTextNode(' ' + m[2]));
  }
  function fill(bd, text) {
    while (bd.firstChild) { bd.removeChild(bd.firstChild); }
    var lines = String(text || '').split('\n'), list = null, lead = true;
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i].trim();
      if (!line) { continue; }
      if (lead) {
        var p0 = document.createElement('p'); p0.className = 'shep-bubble-lead';
        p0.textContent = line; bd.appendChild(p0); lead = false; continue;
      }
      if (line.indexOf(BULLET) === 0) {
        if (!list) { list = document.createElement('ul'); list.className = 'shep-bubble-list'; bd.appendChild(list); }
        var li = document.createElement('li');
        termInto(li, line.slice(BULLET.length).trim()); list.appendChild(li); continue;
      }
      list = null;
      var p = document.createElement('p');
      if (line.indexOf(REF) === 0) { p.className = 'shep-bubble-ref'; }
      termInto(p, line); bd.appendChild(p);
    }
  }
  function titleOf(btn) {
    return btn.getAttribute('data-help-title') ||
      (btn.getAttribute('aria-label') || '').replace(/^Help: /, '');
  }
  function ensureBubble() {
    if (bubble) { return bubble; }
    bubble = document.createElement('div');
    bubble.className = 'shep-bubble';
    bubble.id = 'shep-help-bubble';
    bubble.setAttribute('role', 'tooltip');
    bubble.hidden = true;
    var hd = document.createElement('div'); hd.className = 'shep-bubble-hd';
    var title = document.createElement('span'); title.className = 'shep-bubble-title';
    title.id = 'shep-help-bubble-title';
    // One close control at the head's right end (the owner, 2026-11-25): the framed Esc
    // keycap (a pinned bubble's, `setRole`) beside the ×, both in one button, so a click on
    // either part closes. The keycap is drawn, not read: the button's name says what it
    // does and aria-keyshortcuts names its key.
    var x = document.createElement('button');
    x.type = 'button'; x.className = 'shep-bubble-x'; x.tabIndex = -1;
    x.setAttribute('aria-label', 'Close help'); x.setAttribute('aria-keyshortcuts', 'Escape');
    var kbd = document.createElement('span'); kbd.className = 'shep-kbd'; kbd.textContent = 'Esc';
    kbd.setAttribute('aria-hidden', 'true');
    var mark = document.createElement('span'); mark.className = 'shep-bubble-x-mark'; mark.textContent = '×';
    x.appendChild(kbd); x.appendChild(mark);
    hd.appendChild(title); hd.appendChild(x);
    var bd = document.createElement('div'); bd.className = 'shep-bubble-bd';
    bubble.appendChild(hd); bubble.appendChild(bd);
    // The reader's pointer moving in the bubble keeps it (to read on). A
    // bubble that came under a resting pointer (it follows its ⓘ when the
    // page scrolls) gets a "mouseenter" from the browser but no "mousemove":
    // that keeps nothing, so a preview never outlives its focus or hover
    // and covers what the pointer rests on. A preview focus opened is not
    // the pointer's: the pointer moving over it passes through it (the
    // control under it gets the click), and its leaving ends nothing.
    bubble.addEventListener('mousemove', function () {
      if (pinned || byPointer) { clearTimeout(hideTimer); } else if (!yielded) { yielded = true; claim(); }
    });
    bubble.addEventListener('mouseleave', function () { if (!pinned && byPointer) { scheduleHide(); } });
    x.addEventListener('click', function (e) { e.stopPropagation(); close(true); });
    document.body.appendChild(bubble);
    return bubble;
  }
  // The field a ⓘ explains, which the bubble never covers: a registered
  // field's component (its key is the component's elem_id; Gradio 6 also
  // gives the field's <label> a "block" class, so the nearest ".block" is
  // only its label), else its table row, else its key-value row (the key
  // cell and the value beside it), else the line the ⓘ sits in.
  function box(r) { return {left: r.left, top: r.top, right: r.right, bottom: r.bottom}; }
  // An element drawn with "display: contents" (a field whose label's cells
  // sit in a grid row, as Diagnosis' Options) has no box of its own: its
  // rectangle is the one its drawn children take together.
  function rectOf(el) {
    if (getComputedStyle(el).display !== 'contents') { return box(el.getBoundingClientRect()); }
    var u = null;
    for (var c = el.firstElementChild; c; c = c.nextElementSibling) {
      var r = rectOf(c);
      if (r.right <= r.left && r.bottom <= r.top) { continue; }
      u = u ? {left: Math.min(u.left, r.left), top: Math.min(u.top, r.top),
               right: Math.max(u.right, r.right), bottom: Math.max(u.bottom, r.bottom)} : r;
    }
    return u || box(el.getBoundingClientRect());
  }
  function ownField(btn) {
    var el = btn.classList.contains('shep-i-field') &&
      document.getElementById(btn.getAttribute('data-help') || '');
    return (el && el.contains(btn)) ? el : null;
  }
  // A ⓘ that is no field's, table row's or key-value row's, inside a region
  // marked REGION (a card it heads): the region, kept uncovered when a side fits.
  function regionRect(btn) {
    if (ownField(btn) || btn.closest('tr') || btn.closest('.shep-k')) { return null; }
    var region = btn.closest('.' + REGION);
    return region ? box(region.getBoundingClientRect()) : null;
  }
  function join(a, b) {
    return {left: Math.min(a.left, b.left), top: Math.min(a.top, b.top),
            right: Math.max(a.right, b.right), bottom: Math.max(a.bottom, b.bottom)};
  }
  // A dimmed control's phrase (R-WHY) explains the controls it names (data-for): they are
  // kept uncovered with the phrase's own line, as a field is for its ⓘ (the phrase's line
  // alone let the bubble fall on the button it explains).
  function explained(btn, r) {
    var ids = (btn.getAttribute('data-for') || '').split(/\s+/).filter(Boolean);
    for (var i = 0; i < ids.length; i++) {
      var c = document.getElementById(ids[i]), q = c && rectOf(c);
      if (q && q.right > q.left && q.bottom > q.top) { r = join(r, q); }
    }
    return r;
  }
  function fieldRect(btn) {
    var el = ownField(btn) || btn.closest('tr');
    if (!el) {
      var key = btn.closest('.shep-k');
      if (key) {
        var k = box(key.getBoundingClientRect()), next = key.nextElementSibling;
        return next ? join(k, next.getBoundingClientRect()) : k;
      }
      el = btn.closest('.shep-chip, li, p, dt, dd, h1, h2, h3, h4, label, div') || btn;
    }
    return explained(btn, rectOf(el));
  }
  function overlaps(a, b) {
    return a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
  }
  // **How many drawn controls each placement would lie over** (R-HELP: a bubble covers no
  // control): every button, field, select, link and choice in view (not inside the bubble,
  // not its ⓘ, not hidden, not scrolled out of its region), by its box, a pixel's slack at
  // each edge so a placement that only meets a control's edge is no cover.
  var CONTROLS = 'button, input:not([type=hidden]), select, textarea, a[href], [role=radio], [role=checkbox]';
  var lastSide = null, lastWidth = null;
  // The drawn controls' boxes, read once a placement (`seen` read when a place meets one).
  function controlBoxes() {
    var out = [], all = document.querySelectorAll(CONTROLS);
    for (var i = 0; i < all.length; i++) {
      var c = all[i];
      if (c === anchor || (bubble && bubble.contains(c))) { continue; }
      var r = c.getBoundingClientRect();
      if (r.width < 1 || r.height < 1) { continue; }
      out.push({el: c, q: {left: r.left + 1, top: r.top + 1, right: r.right - 1, bottom: r.bottom - 1}, seen: null});
    }
    return out;
  }
  function coveredCounts(places, boxes) {
    var counts = places.map(function () { return 0; });
    for (var i = 0; i < boxes.length; i++) {
      var b = boxes[i], hit = [];
      for (var k = 0; k < places.length; k++) {
        var p = places[k];
        if (overlaps(b.q, {left: p.left, top: p.top, right: p.left + p.w, bottom: p.top + p.h})) { hit.push(k); }
      }
      if (!hit.length) { continue; }
      if (b.seen === null) { b.seen = getComputedStyle(b.el).visibility !== 'hidden' && !outOfView(b.el); }
      if (!b.seen) { continue; }
      for (var h = 0; h < hit.length; h++) { counts[hit[h]] += 1; }
    }
    return counts;
  }
  // Whether a ⓘ is out of view: not drawn, or wholly outside the window or
  // the visible part of a region it scrolls in (an ancestor that clips its
  // overflow; one that does not hold an absolutely or fixed positioned box
  // does not clip it), as Floating UI's `hide` reads its reference.
  var CLIPS = /auto|scroll|hidden|clip|overlay/;
  function holdsFixed(s) {
    return s.transform !== 'none' || s.perspective !== 'none' || s.filter !== 'none' ||
      /paint|layout|strict|content/.test(s.contain || '') || /transform|perspective|filter/.test(s.willChange || '');
  }
  function outOfView(el) {
    var r = el.getBoundingClientRect();
    if (!r.width && !r.height) { return true; }
    var c = {left: 0, top: 0, right: document.documentElement.clientWidth,
             bottom: document.documentElement.clientHeight};
    var pos = getComputedStyle(el).position;
    for (var n = el.parentElement; n && n !== document.body && n !== document.documentElement; n = n.parentElement) {
      var s = getComputedStyle(n);
      if (pos === 'fixed' ? !holdsFixed(s) : pos === 'absolute' && s.position === 'static' && !holdsFixed(s)) {
        continue;
      }
      if (s.display !== 'contents' && s.display !== 'inline') {
        var b = n.getBoundingClientRect(), left = b.left + n.clientLeft, top = b.top + n.clientTop;
        if (CLIPS.test(s.overflowX)) { c.left = Math.max(c.left, left); c.right = Math.min(c.right, left + n.clientWidth); }
        if (CLIPS.test(s.overflowY)) { c.top = Math.max(c.top, top); c.bottom = Math.min(c.bottom, top + n.clientHeight); }
      }
      pos = s.position === 'fixed' || s.position === 'absolute' ? s.position : 'static';
    }
    return r.bottom <= c.top || r.top >= c.bottom || r.right <= c.left || r.left >= c.right;
  }
  function place() {
    if (!bubble || bubble.hidden || !anchor) { return; }
    if (!anchor.isConnected) {
      var again = anchorKey && anchorByKey(anchorKey);
      if (!again) { close(false); return; }
      // A drawing replaced the ⓘ the reader's focus was on: focus goes to
      // the new one, unscrolled, as the shared disclosure keeper does
      // (`renderers.KEEP_OPEN_JS`), so Escape still closes the bubble.
      var refocus = anchorFocused && (!document.activeElement || document.activeElement === document.body);
      anchor = again; anchor.setAttribute('aria-expanded', 'true');
      if (refocus) { returning = again; again.focus({preventScroll: true}); returning = null; }
    }
    // Its ⓘ out of view: not drawn and no click's target, still open; focus
    // inside it goes to its ⓘ, the page unmoved.
    var away = outOfView(anchor);
    bubble.style.visibility = away ? 'hidden' : '';
    if (away) {
      if (bubble.contains(document.activeElement)) { anchor.focus({preventScroll: true}); }
      return;
    }
    var a = anchor.getBoundingClientRect(), f = fieldRect(anchor), region = regionRect(anchor);
    var vw = document.documentElement.clientWidth, vh = document.documentElement.clientHeight;
    var width = Math.max(LEAST_W, Math.min(MAX_W, vw - 2 * EDGE));
    bubble.style.maxWidth = width + 'px';
    // Its whole height read with its cap kept (`whole`): lifting the cap to measure it scrolled a
    // capped text back to its top on every scroll of it.
    var text = bubble.querySelector('.shep-bubble-bd');
    function whole() {
      return Math.min(bubble.offsetHeight - text.clientHeight + text.scrollHeight, vh - 2 * EDGE);
    }
    var bw = bubble.offsetWidth, bh = whole();
    var cx = a.left + a.width / 2, cy = a.top + a.height / 2;
    function x(left, w) { return Math.min(Math.max(EDGE, left), vw - w - EDGE); }
    function y(top, h) { return Math.min(Math.max(EDGE, top), vh - h - EDGE); }
    // A side, its bubble aligned from the ⓘ (its start) or ending at it (`end`, Floating UI's
    // "-end" alignment): beside, it runs down from the ⓘ or ends under it; above or below, it
    // runs right from the ⓘ or ends past it. `slide` (0 to 1) places it between the two along
    // its side (Floating UI's `shift`), the ⓘ still within its edge: 0 its start, 1 its end.
    function at(side, w, h, g, end, slide) {
      g = g || f;
      var t = slide === undefined ? (end ? 1 : 0) : slide;
      var top = y(cy - 24 + t * (48 - h), h), left = x(cx - 24 + t * (48 - w), w);
      if (side === 'right') { return {side: side, left: g.right + GAP, top: top, w: w, h: h}; }
      if (side === 'left') { return {side: side, left: g.left - GAP - w, top: top, w: w, h: h}; }
      if (side === 'above') { return {side: side, left: left, top: g.top - GAP - h, w: w, h: h}; }
      return {side: side, left: left, top: g.bottom + GAP, w: w, h: h};
    }
    function fits(p, g) {
      return p.left >= EDGE && p.top >= EDGE && p.left + p.w <= vw - EDGE &&
        p.top + p.h <= vh - EDGE &&
        !overlaps({left: p.left, top: p.top, right: p.left + p.w, bottom: p.top + p.h}, g);
    }
    // Beside the field, then above it, then below it, each aligned from the
    // ⓘ, then ending at it, then slid between the two along its side (SLIDES
    // steps, the ⓘ still within the bubble's edge): of the places that fit the
    // window and leave the field uncovered, the one that lies over the fewest
    // drawn controls (R-HELP: a bubble covers no control), the first in that
    // order among equals, the place it stands in now kept while no other
    // covers fewer (no flip while the page scrolls); a region's ⓘ tries the
    // whole region first. The L8 close: in a 720 px window the two alignments
    // of every side lay over a control while a place between them was clear.
    var sides = ['right', 'left', 'above', 'below'], chosen = null, SLIDES = 12;
    var guards = region ? [region, f] : [f];
    function places(h) {
      var fitting = [];
      for (var k = 0; k < guards.length; k++) {
        for (var i = 0; i < sides.length; i++) {
          for (var e = 0; e < 2; e++) {
            var p = at(sides[i], bw, h, guards[k], e === 1);
            p.place = sides[i] + (e ? '-end' : '');
            if (fits(p, guards[k])) { fitting.push(p); }
          }
        }
        for (var si = 0; si < sides.length; si++) {
          for (var n = 1; n < SLIDES; n++) {
            var q = at(sides[si], bw, h, guards[k], false, n / SLIDES);
            q.place = sides[si] + '~' + n;
            if (fits(q, guards[k])) { fitting.push(q); }
          }
        }
      }
      return fitting;
    }
    // Of `fitting`, the one over the fewest controls, the place it stands in now kept among equals.
    function fewest(fitting, counts) {
      var best = 0;
      for (var j = 1; j < fitting.length; j++) {
        if (counts[j] < counts[best] || (counts[j] === counts[best] && fitting[j].place === lastSide &&
            fitting[best].place !== lastSide)) { best = j; }
      }
      return best;
    }
    var boxes = controlBoxes(), fitting = places(bh), fewestSeen = null;
    // Of the places tried, the one over the fewest controls, with the width it was tried at: among
    // equals the place and width it stands in now (no flip while the page scrolls: a focus
    // preview re-placed on each scroll took 220 and 320 px by turns, and each change of its size
    // laid the page out again under a scrolled region), else the first tried.
    function consider(p, count) {
      var w = bubble.style.maxWidth, now = p.place === lastSide && w === lastWidth;
      if (!fewestSeen || count < fewestSeen.count || (count === fewestSeen.count && now && !fewestSeen.now)) {
        fewestSeen = {place: p, count: count, width: w, now: now};
      }
    }
    if (fitting.length) {
      var counts = coveredCounts(fitting, boxes), best = fewest(fitting, counts);
      chosen = fitting[best];
      if (counts[best] > 0) {
        consider(chosen, counts[best]);
        chosen = shorter(boxes) || narrower(boxes);
        if (!chosen) { bubble.style.maxWidth = fewestSeen.width; chosen = fewestSeen.place; }
      }
    }
    // **Shorter where no place of its height is clear** (R-HELP, the owner's C9 of 2026-11-24:
    // "covers no control, readable, the full reference reachable"; the L8 close, CK8-1: a
    // read-only phrase's answer, about 300 px tall, lay over a control in 17 of 56 openings from
    // 1850 x 960 to 1280 x 720, Fine-tune's whole Device field at 1550 x 830, while a shorter
    // place beside the phrase was clear). Where every place of its whole height lies over a
    // control, the bubble takes the tallest place that lies over none, its height capped there
    // (Floating UI's `size`), its text scrolling inside under the scroll cue; never shorter than
    // its head, its lead and two lines after it, so the answer reads whole and more is seen to
    // follow.
    function shorter(boxes) {
      var lead = text.firstElementChild;
      if (!lead || !lead.nextElementSibling) { return null; }
      var lh = parseFloat(getComputedStyle(text).lineHeight) || 22;
      var least = Math.ceil(bh - text.scrollHeight + lead.offsetTop + lead.offsetHeight - text.offsetTop + 2 * lh);
      if (least >= bh) { return null; }
      for (var h = bh - lh; h > least - lh; h -= lh) {
        var cap = Math.max(least, Math.floor(h)), shown = places(cap);
        if (!shown.length) { continue; }
        var counts = coveredCounts(shown, boxes), best = fewest(shown, counts);
        shown[best].cap = cap;
        if (counts[best] === 0) { return shown[best]; }
        consider(shown[best], counts[best]);
      }
      return null;
    }
    // **Then narrower** (the L8 close's final check, F4: configured at 1850 x 960 the pinned
    // Dataset bubble lay over the run bar's fingerprint facts, Batch size and the Viewing select:
    // beside a field in a dense bar every place of the bubble's width met a control at any height
    // down to its least, while one narrower beside the field was clear). Where no place of its
    // width is clear, the same search at a narrower width, LEAST_W at the least (Floating UI's
    // `size` reads the width available too): the widest that has a place over no control, its
    // whole height first, then shorter. None at any width or height: of every place tried, the
    // one over the fewest controls (`fewestSeen`; at 1850 x 960 the Dataset bubble then lies over
    // one ⓘ's top, not six controls), the first tried among equals.
    function narrower(boxes) {
      var full = bw, tall = bh;
      for (var w = full - WIDTH_STEP; w > LEAST_W - WIDTH_STEP; w -= WIDTH_STEP) {
        bubble.style.maxWidth = Math.max(w, LEAST_W) + 'px';
        bw = bubble.offsetWidth; bh = whole();
        var at = places(bh), counts = coveredCounts(at, boxes), best = at.length ? fewest(at, counts) : -1;
        if (best >= 0 && counts[best] === 0) { return at[best]; }
        if (best >= 0) { consider(at[best], counts[best]); }
        var found = shorter(boxes);
        if (found) { return found; }
      }
      bubble.style.maxWidth = width + 'px'; bw = full; bh = tall;
      return null;
    }
    if (!chosen) {
      // Narrower beside it, on the side with room for LEAST_W px or more.
      var roomR = vw - EDGE - f.right - GAP, roomL = f.left - GAP - EDGE;
      var room = Math.max(roomR, roomL);
      if (room >= LEAST_W) {
        bubble.style.maxWidth = Math.min(width, room) + 'px';
        chosen = at(roomR >= roomL ? 'right' : 'left', bubble.offsetWidth, whole());
      }
    }
    if (!chosen) {
      // Above or below, on the side with more room, its height capped to
      // that room (its text then scrolls inside).
      var roomAbove = f.top - GAP - EDGE, roomBelow = vh - EDGE - f.bottom - GAP;
      var up = roomAbove > roomBelow, room2 = Math.max(48, up ? roomAbove : roomBelow), tall = whole();
      chosen = at(up ? 'above' : 'below', bubble.offsetWidth, Math.min(tall, room2));
      if (tall > room2) { chosen.cap = room2; }
    }
    // Its cap set only when it changes (bubble.style.maxHeight), so a text scrolled inside it stays.
    var cap = chosen.cap ? chosen.cap + 'px' : '';
    if (bubble.style.maxHeight !== cap) { bubble.style.maxHeight = cap; }
    lastSide = chosen.place || chosen.side;
    lastWidth = bubble.style.maxWidth;
    bubble.style.left = chosen.left + 'px';
    bubble.style.top = Math.max(EDGE, chosen.top) + 'px';
    bubble.classList.toggle('shep-up', chosen.side === 'above');
    bubble.classList.toggle('shep-at-right', chosen.side === 'right');
    bubble.classList.toggle('shep-at-left', chosen.side === 'left');
    bubble.style.setProperty('--shep-ax', Math.max(10, Math.min(chosen.w - 22, cx - chosen.left - 6)) + 'px');
    bubble.style.setProperty('--shep-ay', Math.max(10, Math.min(chosen.h - 22, cy - chosen.top - 6)) + 'px');
    // Its text capped (above, or by the room it has), the scroll cue every region that scrolls
    // has (R-73, the shell's MORE, drawn by the style sheet) while more lies below.
    var more = text.scrollHeight - text.scrollTop - text.clientHeight > 1;
    if (text.hasAttribute('data-shep-more') !== more) { text.toggleAttribute('data-shep-more', more); }
  }
  // Pinned, the bubble is a dialog named by its title; a preview (hover or
  // focus) is a tooltip.
  function setRole() {
    if (!bubble) { return; }
    bubble.setAttribute('role', pinned ? 'dialog' : 'tooltip');
    // The Esc keycap beside the × is a pinned bubble's: a preview closes as the pointer leaves.
    bubble.querySelector('.shep-bubble-x > .shep-kbd').hidden = !pinned;
    if (pinned) { bubble.setAttribute('aria-labelledby', 'shep-help-bubble-title'); }
    else { bubble.removeAttribute('aria-labelledby'); }
  }
  // Who the bubble answers to: pinned, or a preview the pointer opened, it
  // takes the pointer's events (the pointer can be moved onto it, WCAG
  // 1.4.13); a preview focus opened lets the reader's pointer through once
  // it moves over it (`yielded`), its own wheel aside (`wheelText`).
  function claim() {
    if (!bubble) { return; }
    if (pinned || byPointer) { yielded = false; }
    bubble.style.pointerEvents = yielded ? 'none' : '';
    if (yielded) { document.addEventListener('wheel', wheelText, {passive: false}); }
    else { document.removeEventListener('wheel', wheelText); }
  }
  // A preview that lets the pointer through still scrolls its own text under
  // the wheel (its height capped, its text scrolling inside), as one that
  // takes the pointer does: the wheel over it moves its text while the text
  // can move, and past its end, or with none to move, the page as before.
  function wheelText(e) {
    if (!yielded || !bubble || bubble.hidden || bubble.style.visibility === 'hidden' ||
        e.ctrlKey || !e.deltaY) { return; }
    var r = bubble.getBoundingClientRect();
    if (e.clientX < r.left || e.clientX >= r.right || e.clientY < r.top || e.clientY >= r.bottom) {
      return;
    }
    var bd = bubble.querySelector('.shep-bubble-bd');
    var dy = e.deltaY * (e.deltaMode === 1 ? (parseFloat(getComputedStyle(bd).lineHeight) || 20)
                         : e.deltaMode === 2 ? bd.clientHeight : 1);
    if ((dy > 0 ? bd.scrollHeight - bd.clientHeight - bd.scrollTop : bd.scrollTop) < 1) { return; }
    e.preventDefault();
    bd.scrollTop += dy;
  }
  // `viaPointer`: a hover opened it. Focus on the ⓘ of a preview already
  // shown leaves it whoever's it was.
  function open(btn, pin, viaPointer) {
    clearTimeout(hideTimer);
    clearTimeout(openTimer); previewFor = null;
    if (anchor !== btn || !bubble || bubble.hidden) { byPointer = false; yielded = false; }
    if (anchor && anchor !== btn) { anchor.setAttribute('aria-expanded', 'false'); pinned = false; }
    if (anchor !== btn) { lastSide = null; }
    anchor = btn; anchorKey = btn.getAttribute('data-help');
    pinned = pin ? true : pinned;
    byPointer = byPointer || !!viaPointer;
    var b = ensureBubble();
    b.querySelector('.shep-bubble-title').textContent = titleOf(btn);
    fill(b.querySelector('.shep-bubble-bd'), textOf(btn));
    setRole();
    claim();
    b.hidden = false;
    btn.setAttribute('aria-expanded', 'true');
    place();
  }
  // A hover previews after a short delay, so a pointer passing over a ⓘ
  // opens nothing.
  function openSoon(btn) {
    if (previewFor === btn) { return; }
    clearTimeout(openTimer);
    previewFor = btn;
    openTimer = setTimeout(function () { previewFor = null; if (btn.isConnected) { open(btn, false, true); } }, HOVER_MS);
  }
  // Focus inside the bubble goes back to its ⓘ once it is closed, without
  // scrolling the page to it; that focus opens nothing (`returning`): the
  // reader closed it (before, it opened a preview with no ⓘ to follow,
  // which stayed drawn where it was when the page scrolled).
  var returning = null;
  function close(returnFocus) {
    clearTimeout(hideTimer);
    clearTimeout(openTimer); previewFor = null;
    var back = returnFocus && anchor && anchor.isConnected && bubble &&
      bubble.contains(document.activeElement) ? anchor : null;
    if (bubble) { bubble.hidden = true; bubble.style.visibility = ''; }
    if (anchor) { anchor.setAttribute('aria-expanded', 'false'); }
    anchor = null; anchorKey = null; pinned = false; byPointer = false; yielded = false; lastSide = null;
    setRole();
    claim();
    if (back) { returning = back; back.focus({preventScroll: true}); returning = null; }
  }
  // A hide the last ⓘ's pointer left behind never cancels another ⓘ's
  // preview that is waiting to open: that one replaces the bubble instead.
  function scheduleHide() {
    clearTimeout(hideTimer);
    hideTimer = setTimeout(function () { if (!pinned && !previewFor) { close(false); } }, 250);
  }
  // The pointer on the ⓘ of the preview shown makes it the pointer's too.
  document.addEventListener('mouseover', function (e) {
    var btn = closestHelp(e.target);
    if (!btn) { return; }
    if (anchor !== btn || !bubble || bubble.hidden) { openSoon(btn); }
    else { clearTimeout(hideTimer); byPointer = true; claim(); }
  });
  document.addEventListener('mouseout', function (e) {
    var btn = closestHelp(e.target);
    if (!btn) { return; }
    var to = e.relatedTarget;
    if (to && (btn.contains(to) || (bubble && bubble.contains(to)))) { return; }
    if (previewFor === btn) { clearTimeout(openTimer); previewFor = null; }
    if (!pinned) { scheduleHide(); }
  });
  document.addEventListener('focusin', function (e) {
    var btn = closestHelp(e.target);
    if (btn && btn !== returning) { open(btn, false); }
    anchorFocused = !!anchor && closestHelp(e.target) === anchor;
  });
  // Chromium's focusout for a drawing that removes the focused ⓘ is the one
  // for a click on nothing focusable (no relatedTarget, the ⓘ still in the
  // page): told apart once the script running now is done.
  document.addEventListener('focusout', function (e) {
    if (anchor && closestHelp(e.target) === anchor) {
      var left = anchor;
      if (e.relatedTarget) { anchorFocused = false; }
      else {
        Promise.resolve().then(function () {
          if (left.isConnected && document.activeElement !== left) { anchorFocused = false; }
        });
      }
    }
  });
  document.addEventListener('focusout', function (e) {
    var btn = closestHelp(e.target);
    if (!btn || pinned) { return; }
    var to = e.relatedTarget;
    if (to && bubble && bubble.contains(to)) { return; }
    scheduleHide();
  });
  document.addEventListener('click', function (e) {
    var btn = closestHelp(e.target);
    if (btn) {
      // A ⓘ inside a field's <label>: the click opens the help, never the field.
      e.preventDefault();
      if (anchor === btn && pinned) { close(false); } else { open(btn, true); }
      return;
    }
    if (bubble && !bubble.hidden && !bubble.contains(e.target)) { close(false); }
  }, true);
  // Esc closes the bubble; while its ⓘ is out of view (the bubble not drawn)
  // the key also goes on to what the reader sees.
  document.addEventListener('keydown', function (e) {
    if ((e.key === 'Escape' || e.key === 'Esc') && bubble && !bubble.hidden) {
      var drawn = bubble.style.visibility !== 'hidden';
      close(true);
      if (drawn) { e.stopPropagation(); }
    }
  }, true);
  window.addEventListener('resize', place);
  window.addEventListener('scroll', place, true);

  // Field help: a ⓘ after the title of each registered component, added as
  // the component mounts (lazily mounted tabs and accordions included).
  function titleIn(el) {
    return el.querySelector('[data-testid="block-info"]') ||
      el.querySelector('label .label-text') || el.querySelector('label > span');
  }
  // The ⓘ sits inside the field's <label>; a <button> there would become the
  // label's control (a click on the field's title would open the help, not
  // focus the field), so the label names its own field by id.
  function keepLabel(target, key) {
    var input = target.querySelector('textarea, input:not([type=radio]):not([type=hidden])');
    var label = input && input.closest ? input.closest('label') : null;
    if (!label || !target.contains(label)) { return; }
    if (!input.id) { input.id = 'shep-in-' + key; }
    if (label.htmlFor !== input.id) { label.htmlFor = input.id; }
  }
  // A field's description: its visible info= line first (Gradio's
  // .info-text, set when it is built or at runtime, which Gradio may draw
  // again), then its help text, so a screen reader reads what the page shows
  // under the field as well as the bubble's text. Kept on every pass, merged
  // with what the shell describes the same control by (`DESCRIBED_BY_JS`).
__DESCRIBED_BY__
  function describe(target, key, textId) {
    var input = target.querySelector('textarea, input:not([type=radio]):not([type=hidden])');
    if (!input) { return; }
    var ids = [], info = target.querySelector('.info-text');
    if (info && (info.textContent || '').trim()) {
      if (!info.id) { info.id = 'shep-info-' + key; }
      ids.push(info.id);
    }
    ids.push(textId);
    describedBy(input, 'data-shep-help-by', ids);
  }
  // A field drawn as its key, its ⓘ and its control (in a container marked
  // KEY_FIRST) keeps its control after its ⓘ in the page as well, so focus
  // and a screen reader meet them in the order drawn: Gradio puts a
  // checkbox's box before its words, where the ⓘ follows.
  function keyFirst(target, btn) {
    if (!target.closest('.' + KEY_FIRST)) { return; }
    var input = target.querySelector('input:not([type=hidden])');
    if (input && btn && input.parentNode === btn.parentNode &&
        (input.compareDocumentPosition(btn) & Node.DOCUMENT_POSITION_FOLLOWING)) {
      btn.insertAdjacentElement('afterend', input);
    }
  }
  function enhance() {
    var texts = document.querySelectorAll('.shep-help-texts [data-help-target]');
    for (var i = 0; i < texts.length; i++) {
      var t = texts[i], target = document.getElementById(t.getAttribute('data-help-target'));
      if (!target) { continue; }
      var key = t.getAttribute('data-help-key'), known = target.querySelector('button.shep-i[data-help="' + key + '"]');
      if (known) {
        keyFirst(target, known); keepLabel(target, key); describe(target, key, t.id); continue;
      }
      var title = titleIn(target);
      if (!title) { continue; }
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'shep-i shep-i-field';
      btn.setAttribute('data-help', key);
      btn.setAttribute('data-help-title', t.getAttribute('data-help-title') || '');
      btn.setAttribute('aria-label', 'Help: ' + (t.getAttribute('data-help-title') || ''));
      btn.setAttribute('aria-expanded', 'false');
      btn.setAttribute('aria-describedby', t.id);
      btn.innerHTML = '__I_ICON__';
      title.insertAdjacentElement('afterend', btn);
      keyFirst(target, btn);
      if (!title.id) { title.id = 'shep-lbl-' + key; }
      var input = target.querySelector('textarea, input:not([type=radio])');
      if (input && !input.hasAttribute('aria-label')) { input.setAttribute('aria-labelledby', title.id); }
      describe(target, key, t.id);
      keepLabel(target, key);
    }
  }
  var pending = false;
  function schedule() {
    if (pending) { return; }
    pending = true;
    requestAnimationFrame(function () { pending = false; enhance(); place(); });
  }
  new MutationObserver(schedule).observe(document.documentElement, {childList: true, subtree: true});

  // Copy a full value (a 64-hex fingerprint) from a button.shep-copy.
  document.addEventListener('click', function (e) {
    var btn = e.target && e.target.closest ? e.target.closest('button.shep-copy') : null;
    if (!btn) { return; }
    var value = btn.getAttribute('data-copy') || '';
    var done = function () {
      btn.setAttribute('data-copied', '1');
      var label = btn.querySelector('.shep-copy-label');
      if (label) { label.textContent = 'Copied'; }
      setTimeout(function () { btn.removeAttribute('data-copied'); if (label) { label.textContent = 'Copy'; } }, 1500);
    };
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(value).then(done, function () {});
    } else {
      var area = document.createElement('textarea');
      area.value = value; area.setAttribute('readonly', ''); area.style.position = 'fixed'; area.style.left = '-9999px';
      document.body.appendChild(area); area.select();
      try { document.execCommand('copy'); done(); } catch (err) { /* nothing copied */ }
      document.body.removeChild(area);
    }
  });

  // `title`: a field's title as the ⓘ finds it, which the touch script names a field by.
  window.ShepHelp = {open: open, close: close, enhance: enhance, title: titleIn};
})();
"""

#: The help control's script, for the mount's ``head=`` (`src/webui/app.py`).
HELP_HEAD = "<script>" + (_HELP_JS.replace("__I_ICON__", _I_ICON.replace("'", "\\'"))
                          .replace("__MAX_W__", str(BUBBLE_MAX_WIDTH))
                          .replace("__HOVER_MS__", str(HOVER_OPEN_MS))
                          .replace("__REGION__", REGION_CLASS)
                          .replace("__KEY_FIRST__", KEY_FIRST_CLASS)
                          .replace("__BULLET__", json.dumps(BULLET))
                          .replace("__REF__", json.dumps(REF_PREFIX))
                          .replace("__TERM__", json.dumps(TERM_PATTERN))
                          .replace("__DESCRIBED_BY__", DESCRIBED_BY_JS)) + "</script>"


#: The look of the ⓘ and its bubble (part of the dashboard's CSS). The ⓘ's
#: button is the hit target, `HIT_TARGET` square; the 20 px ring it draws
#: inside (``::before``) keeps the look, and its negative margins keep the
#: line's height.
HELP_CSS = """
button.shep-i {
  position: relative; display: inline-flex; align-items: center; justify-content: center; flex: none;
  width: __HIT__px; height: __HIT__px; padding: 0; margin: -2px 0 -2px 4px; border: 0; border-radius: 50%;
  background: transparent; color: #4b5260; cursor: pointer; vertical-align: middle; line-height: 1;
}
button.shep-i::before {
  content: ""; position: absolute; inset: 2px; border-radius: 50%;
  border: 1.5px solid #8d94a1; background: #fff;
}
button.shep-i > svg { position: relative; }
button.shep-i:hover, button.shep-i[aria-expanded="true"] { color: var(--shep-accent-text); }
button.shep-i:hover::before, button.shep-i[aria-expanded="true"]::before { border-color: var(--shep-accent); background: var(--shep-accent-50); }
button.shep-i:focus-visible { outline: 2px solid var(--shep-accent); outline-offset: 0; }
.shep-i-field { top: -1px; }
.shep-help-texts { display: none !important; }
.shep-bubble {
  position: fixed; z-index: 1000; width: max-content; max-width: __MAX_W__px; box-sizing: border-box;
  max-height: calc(100vh - 16px); display: flex; flex-direction: column;
  background: #fff; border: 1px solid var(--shep-border-2); border-radius: 10px;
  box-shadow: 0 10px 28px rgba(20, 24, 40, .16), 0 2px 6px rgba(20, 24, 40, .08);
  padding: 12px 14px 10px; font-size: 15px; line-height: var(--shep-lh-prose, 1.45); color: var(--shep-text);
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
  text-align: left; white-space: normal; font-weight: 400;
}
.shep-bubble * { font-family: inherit; }
.shep-bubble[hidden] { display: none; }
.shep-bubble::before {
  content: ""; position: absolute; width: 12px; height: 12px; background: #fff;
  border-left: 1px solid var(--shep-border-2); border-top: 1px solid var(--shep-border-2);
  top: -7px; left: var(--shep-ax, 18px); transform: rotate(45deg);
}
.shep-bubble.shep-up::before { top: auto; bottom: -7px; transform: rotate(225deg); }
.shep-bubble.shep-at-right::before { top: var(--shep-ay, 18px); left: -7px; transform: rotate(-45deg); }
.shep-bubble.shep-at-left::before { top: var(--shep-ay, 18px); left: auto; right: -7px; transform: rotate(135deg); }
.shep-bubble-hd { display: flex; align-items: flex-start; gap: 8px; font-weight: 600; margin-bottom: 4px; flex: none; }
/* The title wraps beside the close control, never under or behind it. */
.shep-bubble-title { flex: 1 1 auto; min-width: 0; overflow-wrap: anywhere; }
/* The one close control (the owner, 2026-11-25): the Esc keycap, pinned only, beside the ×,
   one button. Its 26 px box is centred on the title's first line (a 15 px line of 1.45: its
   margins give back the 2 px it stands over the line at each edge); the × keeps its 26 px
   square, so a preview's control is the × alone as before. */
.shep-bubble-x {
  flex: none; display: inline-flex; align-items: center; height: 26px; min-width: 26px; margin: -2px 0;
  padding: 0; border: 0; border-radius: 6px; background: none; color: var(--shep-muted);
  font-size: 20px; font-weight: 400; cursor: pointer;
}
.shep-bubble-x-mark { width: 26px; text-align: center; }
.shep-bubble-x > .shep-kbd { margin-left: 4px; }
.shep-bubble-x > .shep-kbd[hidden] { display: none; }
.shep-bubble-x:hover { background: var(--shep-panel-2); color: var(--shep-text); }
.shep-bubble-x:focus-visible { outline: 2px solid var(--shep-accent); outline-offset: 0; }
.shep-bubble-bd { overflow-wrap: anywhere; overflow-y: auto; min-height: 0; flex: 1 1 auto; }
/* Every part starts at no margin (one weight with the rule below, which comes later, so the
   space between parts is the text tokens' and nothing else's). */
.shep-bubble-bd > * { margin: 0; }
/* The text layout's tokens (style.py; the text brief's §6): a paragraph's space between the
   bubble's parts, an item's between list items; prose wraps evenly, the lead balanced. */
.shep-bubble-bd { text-wrap: pretty; }
.shep-bubble-bd > * + * { margin-top: var(--shep-space-para, 8px); }
.shep-bubble-lead, .shep-bubble-term { font-weight: 600; }
.shep-bubble-lead { text-wrap: balance; }
.shep-bubble-list { padding: 0 0 0 18px; list-style: disc outside; }
.shep-bubble-list > li { margin: 0; padding: 0; }
.shep-bubble-list > li::marker { color: var(--shep-muted); }
.shep-bubble-list > li + li { margin-top: var(--shep-space-item, 4px); }
.shep-bubble-bd > .shep-bubble-ref { font-size: 14px; color: var(--shep-muted); }
/* A keyboard key, framed (the close control's Esc). */
.shep-kbd {
  display: inline-block; font-size: 14px; padding: 0 6px; border: 1px solid var(--shep-border-2);
  border-bottom-width: 2px; border-radius: 5px; background: var(--shep-panel-2); color: var(--shep-text-2); line-height: 20px;
}
button.shep-copy {
  display: inline-flex; align-items: center; gap: 4px; height: 28px; padding: 0 8px; border-radius: 6px;
  border: 1px solid var(--shep-border-2); background: #fff; color: var(--shep-text-2); font-size: 14px; cursor: pointer;
  vertical-align: middle; white-space: nowrap;
}
button.shep-copy:hover { background: var(--shep-panel-2); }
.shep-copy[data-copied] { border-color: var(--shep-green-200); color: var(--shep-green-800); }
.shep-copy.shep-copy-icon .shep-copy-label { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
""".replace("__HIT__", str(HIT_TARGET)).replace("__MAX_W__", str(BUBBLE_MAX_WIDTH))


def as_json_for_script(value: object) -> str:
    """JSON that is safe inside a ``<script>`` element (no ``</script>``)."""
    return (json.dumps(value, ensure_ascii=True)
            .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"))


__all__ = [
    "DESCRIBED_BY_JS",
    "HELP_CSS",
    "HELP_HEAD",
    "BULLET",
    "CODE",
    "HELP_BUDGET",
    "NOT_SAID",
    "TEXT_BUDGET",
    "HelpBudget",
    "HelpEntry",
    "HelpLine",
    "HelpRegistry",
    "KEY_FIRST_CLASS",
    "REF_PREFIX",
    "REGION_CLASS",
    "TERM_PATTERN",
    "TEXT_ID_PREFIX",
    "as_json_for_script",
    "budget_notes",
    "budget_problems",
    "copy_button",
    "help_button",
    "help_lines",
    "sentence_items",
    "registered_keys_in",
    "text_id",
]
