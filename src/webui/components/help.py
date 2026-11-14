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
  a click outside close it; it stays open while the pointer is in the bubble;
- places it so it **never covers the field it explains** (the field's
  component, by its ``elem_id``; else the table row, the key-value row's key
  and value, or the line the ⓘ sits in): beside that field, then above it, then below it, the first that fits
  the window without overlapping it; failing all, narrower beside it, or on
  the side with more room with its height capped (a long text scrolls
  inside); at most 320 px wide (`BUBBLE_MAX_WIDTH`), the ⓘ's hit target at
  least 24 × 24 px;
- fills the bubble with ``textContent`` only: no help text is ever parsed as
  HTML in the browser.

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

#: The bubble's widest (CSS px; the WebUI skill's §7 budget), the delay before
#: a hover opens its preview (ms), and the ⓘ's smallest hit target (CSS px,
#: the skill's §10 floor).
BUBBLE_MAX_WIDTH = 320
HOVER_OPEN_MS = 300
HIT_TARGET = 24

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
  var SEL = 'button.shep-i';
  var MAX_W = __MAX_W__, HOVER_MS = __HOVER_MS__, GAP = 10, EDGE = 8;
  var bubble = null, anchor = null, anchorKey = null, pinned = false;
  var hideTimer = null, openTimer = null, previewFor = null;

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
    var x = document.createElement('button');
    x.type = 'button'; x.className = 'shep-bubble-x'; x.tabIndex = -1;
    x.setAttribute('aria-label', 'Close help'); x.textContent = '×';
    hd.appendChild(title); hd.appendChild(x);
    var bd = document.createElement('div'); bd.className = 'shep-bubble-bd';
    var ft = document.createElement('div'); ft.className = 'shep-bubble-ft';
    var kbd = document.createElement('span'); kbd.className = 'shep-kbd'; kbd.textContent = 'Esc';
    ft.appendChild(kbd); ft.appendChild(document.createTextNode(' to close'));
    bubble.appendChild(hd); bubble.appendChild(bd); bubble.appendChild(ft);
    bubble.addEventListener('mouseenter', function () { clearTimeout(hideTimer); });
    bubble.addEventListener('mouseleave', function () { if (!pinned) { scheduleHide(); } });
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
  function fieldRect(btn) {
    var el = btn.classList.contains('shep-i-field') &&
      document.getElementById(btn.getAttribute('data-help') || '');
    el = (el && el.contains(btn)) ? el : null;
    el = el || btn.closest('tr');
    if (!el) {
      var key = btn.closest('.shep-k');
      if (key) {
        var k = box(key.getBoundingClientRect()), next = key.nextElementSibling;
        if (!next) { return k; }
        var v = next.getBoundingClientRect();
        return {left: Math.min(k.left, v.left), top: Math.min(k.top, v.top),
                right: Math.max(k.right, v.right), bottom: Math.max(k.bottom, v.bottom)};
      }
      el = btn.closest('.shep-chip, li, p, dt, dd, h1, h2, h3, h4, label, div') || btn;
    }
    return box(el.getBoundingClientRect());
  }
  function overlaps(a, b) {
    return a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
  }
  function place() {
    if (!bubble || bubble.hidden || !anchor) { return; }
    if (!anchor.isConnected) {
      var again = anchorKey && document.querySelector(SEL + '[data-help="' + anchorKey + '"]');
      if (again) { anchor = again; anchor.setAttribute('aria-expanded', 'true'); }
      else { close(false); return; }
    }
    var a = anchor.getBoundingClientRect(), f = fieldRect(anchor);
    var vw = document.documentElement.clientWidth, vh = document.documentElement.clientHeight;
    var width = Math.max(220, Math.min(MAX_W, vw - 2 * EDGE));
    bubble.style.maxWidth = width + 'px';
    bubble.style.maxHeight = '';
    var bw = bubble.offsetWidth, bh = bubble.offsetHeight;
    var cx = a.left + a.width / 2, cy = a.top + a.height / 2;
    function x(left, w) { return Math.min(Math.max(EDGE, left), vw - w - EDGE); }
    function y(top, h) { return Math.min(Math.max(EDGE, top), vh - h - EDGE); }
    function at(side, w, h) {
      if (side === 'right') { return {side: side, left: f.right + GAP, top: y(cy - 24, h), w: w, h: h}; }
      if (side === 'left') { return {side: side, left: f.left - GAP - w, top: y(cy - 24, h), w: w, h: h}; }
      if (side === 'above') { return {side: side, left: x(cx - 24, w), top: f.top - GAP - h, w: w, h: h}; }
      return {side: side, left: x(cx - 24, w), top: f.bottom + GAP, w: w, h: h};
    }
    function fits(p) {
      return p.left >= EDGE && p.top >= EDGE && p.left + p.w <= vw - EDGE &&
        p.top + p.h <= vh - EDGE &&
        !overlaps({left: p.left, top: p.top, right: p.left + p.w, bottom: p.top + p.h}, f);
    }
    // Beside the field, then above it, then below it: the first that fits
    // the window and leaves the field uncovered.
    var sides = ['right', 'left', 'above', 'below'], chosen = null;
    for (var i = 0; i < sides.length && !chosen; i++) {
      var p = at(sides[i], bw, bh);
      if (fits(p)) { chosen = p; }
    }
    if (!chosen) {
      // Narrower beside it, on the side with room for 220 px or more.
      var roomR = vw - EDGE - f.right - GAP, roomL = f.left - GAP - EDGE;
      var room = Math.max(roomR, roomL);
      if (room >= 220) {
        bubble.style.maxWidth = Math.min(width, room) + 'px';
        if (bubble.offsetHeight > vh - 2 * EDGE) { bubble.style.maxHeight = (vh - 2 * EDGE) + 'px'; }
        chosen = at(roomR >= roomL ? 'right' : 'left', bubble.offsetWidth, bubble.offsetHeight);
      }
    }
    if (!chosen) {
      // Above or below, on the side with more room, its height capped to
      // that room (its text then scrolls inside).
      var roomAbove = f.top - GAP - EDGE, roomBelow = vh - EDGE - f.bottom - GAP;
      var up = roomAbove > roomBelow;
      bubble.style.maxHeight = Math.max(48, up ? roomAbove : roomBelow) + 'px';
      chosen = at(up ? 'above' : 'below', bubble.offsetWidth, bubble.offsetHeight);
    }
    bubble.style.left = chosen.left + 'px';
    bubble.style.top = Math.max(EDGE, chosen.top) + 'px';
    bubble.classList.toggle('shep-up', chosen.side === 'above');
    bubble.classList.toggle('shep-at-right', chosen.side === 'right');
    bubble.classList.toggle('shep-at-left', chosen.side === 'left');
    bubble.style.setProperty('--shep-ax', Math.max(10, Math.min(chosen.w - 22, cx - chosen.left - 6)) + 'px');
    bubble.style.setProperty('--shep-ay', Math.max(10, Math.min(chosen.h - 22, cy - chosen.top - 6)) + 'px');
  }
  // Pinned, the bubble is a dialog named by its title; a preview (hover or
  // focus) is a tooltip.
  function setRole() {
    if (!bubble) { return; }
    bubble.setAttribute('role', pinned ? 'dialog' : 'tooltip');
    if (pinned) { bubble.setAttribute('aria-labelledby', 'shep-help-bubble-title'); }
    else { bubble.removeAttribute('aria-labelledby'); }
  }
  function open(btn, pin) {
    clearTimeout(hideTimer);
    clearTimeout(openTimer); previewFor = null;
    if (anchor && anchor !== btn) { anchor.setAttribute('aria-expanded', 'false'); pinned = false; }
    anchor = btn; anchorKey = btn.getAttribute('data-help');
    pinned = pin ? true : pinned;
    var b = ensureBubble();
    b.querySelector('.shep-bubble-title').textContent = titleOf(btn);
    b.querySelector('.shep-bubble-bd').textContent = textOf(btn);
    setRole();
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
    openTimer = setTimeout(function () { previewFor = null; if (btn.isConnected) { open(btn, false); } }, HOVER_MS);
  }
  function close(returnFocus) {
    clearTimeout(hideTimer);
    clearTimeout(openTimer); previewFor = null;
    if (bubble) { bubble.hidden = true; }
    if (anchor) {
      anchor.setAttribute('aria-expanded', 'false');
      if (returnFocus && anchor.isConnected && bubble && bubble.contains(document.activeElement)) {
        anchor.focus();
      }
    }
    anchor = null; anchorKey = null; pinned = false;
    setRole();
  }
  function scheduleHide() {
    clearTimeout(hideTimer);
    hideTimer = setTimeout(function () { if (!pinned) { close(false); } }, 250);
  }
  document.addEventListener('mouseover', function (e) {
    var btn = closestHelp(e.target);
    if (btn) { if (anchor !== btn || !bubble || bubble.hidden) { openSoon(btn); } else { clearTimeout(hideTimer); } }
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
    if (btn) { open(btn, false); }
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
  document.addEventListener('keydown', function (e) {
    if ((e.key === 'Escape' || e.key === 'Esc') && bubble && !bubble.hidden) {
      close(true);
      e.stopPropagation();
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
  // under the field as well as the bubble's text. Kept on every pass.
  function describe(target, key, textId) {
    var input = target.querySelector('textarea, input:not([type=radio]):not([type=hidden])');
    if (!input) { return; }
    var ids = [], info = target.querySelector('.info-text');
    if (info && (info.textContent || '').trim()) {
      if (!info.id) { info.id = 'shep-info-' + key; }
      ids.push(info.id);
    }
    ids.push(textId);
    var want = ids.join(' ');
    if (input.getAttribute('aria-describedby') !== want) { input.setAttribute('aria-describedby', want); }
  }
  function enhance() {
    var texts = document.querySelectorAll('.shep-help-texts [data-help-target]');
    for (var i = 0; i < texts.length; i++) {
      var t = texts[i], target = document.getElementById(t.getAttribute('data-help-target'));
      if (!target) { continue; }
      var key = t.getAttribute('data-help-key');
      if (target.querySelector(SEL + '[data-help="' + key + '"]')) {
        keepLabel(target, key); describe(target, key, t.id); continue;
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
                          .replace("__HOVER_MS__", str(HOVER_OPEN_MS))) + "</script>"


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
  padding: 12px 14px 10px; font-size: 15px; line-height: 1.45; color: var(--shep-text);
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
.shep-bubble-hd { display: flex; align-items: center; gap: 8px; font-weight: 600; margin-bottom: 4px; }
.shep-bubble-x {
  margin-left: auto; width: 26px; height: 26px; border: 0; background: none; color: var(--shep-muted);
  display: inline-flex; align-items: center; justify-content: center; border-radius: 6px; font-size: 18px; cursor: pointer;
}
.shep-bubble-bd { white-space: pre-line; overflow-wrap: anywhere; overflow-y: auto; min-height: 0; flex: 1 1 auto; }
.shep-bubble-hd, .shep-bubble-ft { flex: none; }
.shep-bubble-ft { margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--shep-border); font-size: 14px; color: var(--shep-muted); }
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
    "HELP_CSS",
    "HELP_HEAD",
    "HelpEntry",
    "HelpRegistry",
    "TEXT_ID_PREFIX",
    "as_json_for_script",
    "copy_button",
    "help_button",
    "registered_keys_in",
    "text_id",
]
