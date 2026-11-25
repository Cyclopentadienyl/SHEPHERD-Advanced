"""
Display preferences: this browser's own layout choices (the owner, 2026-11-25)
=============================================================================
What a reader of the dashboard chooses for their own page, saved in their
browser (``localStorage``), never on the server: no settings file, no API, no
operator lock. A choice changes only how the page is drawn for that reader, so
a read-only viewer may make it too. ⚙ Settings' "Display" panel offers it
(`runtime_settings`); the page it shapes reads it (`training_console`); this
module is the one place both import, as the page contract asks of a shared
piece (`src/webui/shell.py`).

**Training's Parameters, stacked** (the owner's D9 of 2026-11-25, revised the
same day: keep the automatic fold, and let each user choose the default): how
the Parameters body starts where Training stacks its two panels (a portrait
window, or a landscape one too narrow for them side by side). Side by side the
body is always shown and nothing here applies.

- **Shown** (`SHOWN`, the default: the owner's D9): built shown; the page never
  folds or opens it.
- **Folded** (`FOLDED`): built folded under its title; the page never opens it.
- **Automatic** (`AUTO`): the fold by the room of c14e5b8 (`training_console.FOLD_JS`):
  shown unless folding it brings more of the run view's charts into the first
  screen, decided after the page's first answers and again only when the
  window changes (its size or the arrangement), never on a change of the
  page's content; until that first decision the run view is kept invisible in
  its place, then shown where it stays (F2's option (c)), or shown anyway after
  a few seconds, the body as it stands (then the first decision: answers that
  come later change nothing).

The reader's own Hide or Show on the page wins for that page session, whatever
the choice. A missing, unreadable or unknown value is Shown.

**Read before the first paint.** The head script (`DISPLAY_HEAD`, the mount's
``head``: it runs once per page load, before any component exists) reads the
saved choice and writes it on the root element (`PARAMS_ATTR`), which the
style sheets and the page's scripts read; so Training's first painted frame is
already in the choice's starting state. ``window.ShepDisplay.setParams`` saves
a new choice, writes the attribute again and tells the open page
(`PARAMS_EVENT`), which applies it at once unless its reader used the fold
there. Every storage access is wrapped: with storage blocked the choice holds
for the page session only.

Module: src/webui/components/display.py
"""
from __future__ import annotations

from src.webui.components.help import as_json_for_script

#: The ``localStorage`` key of the Parameters' starting state.
PARAMS_KEY = "shep-training-parameters"
#: The root element's attribute that holds the choice in effect (the head script's).
PARAMS_ATTR = "data-shep-params"
#: The event (on ``document``) a new choice is announced by, its ``detail`` the choice.
PARAMS_EVENT = "shep-params"
#: The three choices (`SHOWN` the default) and their options in ⚙ Settings.
SHOWN, FOLDED, AUTO = "shown", "folded", "auto"
PARAMS_MODES = (SHOWN, FOLDED, AUTO)
PARAMS_DEFAULT = SHOWN
PARAMS_CHOICES = [("Shown", SHOWN), ("Folded", FOLDED), ("Automatic", AUTO)]


_DISPLAY_JS = r"""
(function () {
  if (window.ShepDisplay) { return; }
  var KEY = __KEY__, ATTR = __ATTR__, EVENT = __EVENT__, MODES = __MODES__, DEFAULT = __DEFAULT__;
  function valid(v) { return MODES.indexOf(v) >= 0 ? v : DEFAULT; }
  var saved = null;
  try { saved = window.localStorage.getItem(KEY); } catch (err) { saved = null; }
  var root = document.documentElement;
  root.setAttribute(ATTR, valid(saved));
  window.ShepDisplay = {
    params: function () { return valid(root.getAttribute(ATTR)); },
    setParams: function (v) {
      v = valid(v);
      try { window.localStorage.setItem(KEY, v); } catch (err) { /* this page session only */ }
      root.setAttribute(ATTR, v);
      try { document.dispatchEvent(new CustomEvent(EVENT, {detail: v})); } catch (err) { /* no page listens */ }
      return v;
    }
  };
})();
"""

#: The early read, for the mount's ``head=`` (`app.dashboard_head`).
DISPLAY_HEAD = ("<script>" + _DISPLAY_JS
                .replace("__KEY__", as_json_for_script(PARAMS_KEY))
                .replace("__ATTR__", as_json_for_script(PARAMS_ATTR))
                .replace("__EVENT__", as_json_for_script(PARAMS_EVENT))
                .replace("__MODES__", as_json_for_script(list(PARAMS_MODES)))
                .replace("__DEFAULT__", as_json_for_script(PARAMS_DEFAULT))
                + "</script>")

#: The choice in effect, for a Gradio event's ``js`` (Settings' radio, on load).
READ_PARAMS_JS = (f"() => (window.ShepDisplay ? window.ShepDisplay.params() : "
                  f"{as_json_for_script(PARAMS_DEFAULT)})")
#: A choice the reader made, saved and applied (Settings' radio, on input): browser only.
WRITE_PARAMS_JS = "(v) => { if (window.ShepDisplay) { window.ShepDisplay.setParams(v); } }"


__all__ = ["AUTO", "DISPLAY_HEAD", "FOLDED", "PARAMS_ATTR", "PARAMS_CHOICES", "PARAMS_DEFAULT",
           "PARAMS_EVENT", "PARAMS_KEY", "PARAMS_MODES", "READ_PARAMS_JS", "SHOWN",
           "WRITE_PARAMS_JS"]
