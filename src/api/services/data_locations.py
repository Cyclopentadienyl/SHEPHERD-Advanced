"""
Data locations: save, apply and repair the saved deployment settings
=====================================================================
UI-06 (the owner's decision of 2026-11-16). The service-side entry points the
Settings page, an HTTP route and the backend's startup call, over the file's
locked writes (`src/utils/settings_store.py`) and the job runner's slot:

- `save_and_apply`: save the chosen values (each checked by the rule its
  setting is read by), then try to apply them at once;
- `apply_saved`: make the saved values the ones in use, **only while nothing
  could be using the old locations**: no job in the runner's slot (checked
  under the runner's lock, as Restart Backend's `close_if_idle` does, so a
  start cannot slip in), and no holder of a GPU lock the old values name (a
  command started by hand, this service's CUDA inference;
  `settings_store.apply_saved`). Otherwise the change stays pending, with the
  reason, and applies when Apply is pressed again or at the next start;
- `replace_unreadable`: rename a saved file whose content cannot be read,
  keeping it, so Settings can save again;
- `view`: the one view of the data locations (`src.utils.deployment_view`)
  as this service reads it, its pending reason from the same check `apply_saved`
  makes (the job runner's slot, then the old values' GPU locks), so Settings
  and the route say the reason Apply would give.

The service reads its settings at each use (`src.config.server_settings`), so
an applied change needs no restart; a restart reads the file again, since
nothing writes a resolved value into the environment. Every caller is an
operator control: it applies D8's check (`check_operator_access`) before it
calls any of these.

Module: src/api/services/data_locations.py
"""
from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Optional

from src.api.services.training_manager import training_manager
from src.config import deployment_settings
from src.utils import deployment_view, settings_store
from src.utils.settings_store import ApplyOutcome, RepairOutcome, SaveOutcome

__all__ = ["SaveAndApply", "apply_saved", "replace_unreadable", "save_and_apply", "view"]


class SaveAndApply(NamedTuple):
    """What a save from Settings did: the save, and the apply tried after it
    (None when nothing was saved)."""

    save: SaveOutcome
    apply: Optional[ApplyOutcome]


def apply_saved(environ: Optional[Mapping[str, str]] = None) -> ApplyOutcome:
    """Apply the saved data locations when nothing could be using the old ones
    (module docstring); else leave them pending and say why."""
    # Always under the runner's lock: the file is read again under its own lock
    # by the apply, so a save made meanwhile is never applied past this check.
    busy, outcome = training_manager.run_while_idle(lambda: settings_store.apply_saved(environ))
    if busy is None:
        return outcome
    current = deployment_settings.read_saved(environ)
    if current.problem is not None or not current.pending:
        return ApplyOutcome(False, False, "Nothing was applied: no saved change waits.")
    return ApplyOutcome(False, True, (
        f"Saved, not applied yet: {busy}, so {settings_store.APPLIES_LATER}."))


def view(environ: Optional[Mapping[str, str]] = None) -> deployment_view.DeploymentView:
    """The data locations as this service reads them, with why a pending
    change waits from the check `apply_saved` makes: this runner's slot
    first (`busy_reason`, its words), then the GPU locks (module docstring)."""
    return deployment_view.deployment_view(environ, runner_busy=training_manager.busy_reason())


def save_and_apply(values: Mapping[str, Any],
                   environ: Optional[Mapping[str, str]] = None) -> SaveAndApply:
    """Save `values` (key of the saved file → value; None or empty clears it),
    then try to apply them at once (`apply_saved`)."""
    saved = settings_store.save(values, environ)
    return SaveAndApply(saved, apply_saved(environ) if saved.saved else None)


def replace_unreadable(environ: Optional[Mapping[str, str]] = None) -> RepairOutcome:
    """Rename an unreadable saved settings file out of the way, keeping it
    (`settings_store.replace_unreadable`)."""
    return settings_store.replace_unreadable(environ)
