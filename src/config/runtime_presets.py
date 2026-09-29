"""
Runtime allocator presets — shared, dependency-light source of truth.
=====================================================================
Imported by BOTH the WebUI Runtime Settings tab
(``src/webui/components/runtime_settings.py``) and the launcher
(``scripts/launch/shep_launch.py``).

This module deliberately has **no** heavy imports (no gradio / torch), so the
launcher can read it before starting the server, and unit tests can exercise it
without the WebUI stack.

Single source of truth here avoids the UI and launcher drifting apart, and
defines one absolute settings-file path so both read/write the same file
regardless of current working directory.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

# Repo root: src/config/runtime_presets.py -> parents[2] == repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_SETTINGS_FILE = REPO_ROOT / ".shepherd_runtime_settings.json"

# Preset name -> PYTORCH_ALLOC_CONF value. Native presets state ``backend:native``
# explicitly for clarity (it is the default backend, but being explicit avoids
# ambiguity about which backend the tuning options apply to).
ALLOCATOR_PRESETS: dict[str, str] = {
    "cuda_async": "backend:cudaMallocAsync",
    "expandable": "backend:native,expandable_segments:True",
    "native_roundup": "backend:native,roundup_power2_divisions:4,max_non_split_rounding_mb:512",
    "native": "backend:native",
}
DEFAULT_ALLOCATOR = "cuda_async"

# Env marker recording where PYTORCH_ALLOC_CONF came from:
#   "preset" -> resolved from the saved preset; re-resolve it on every start.
#   "env"    -> an explicit override the launcher saw; preserve it.
ALLOC_SOURCE_ENV = "SHEPHERD_ALLOC_SOURCE"


def load_runtime_settings(path: Path | None = None) -> dict:
    """Load persisted runtime settings.

    Returns an empty dict if the file is absent, unreadable, or malformed —
    a user-specific UI settings file must never block startup.
    """
    p = path or RUNTIME_SETTINGS_FILE
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            # Valid JSON of the wrong shape (list/str/number) must not reach
            # downstream .get(...) calls — only a JSON object is usable.
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError, ValueError):
            return {}
    return {}


def save_runtime_settings(data: dict, path: Path | None = None) -> None:
    p = path or RUNTIME_SETTINGS_FILE
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def resolve_allocator(preset: str | None) -> tuple[str, str]:
    """Resolve a preset name to ``(resolved_preset, PYTORCH_ALLOC_CONF)``.

    Unknown or missing preset names fall back to ``DEFAULT_ALLOCATOR`` rather
    than silently producing a framework-default / empty configuration.
    """
    if preset not in ALLOCATOR_PRESETS:
        preset = DEFAULT_ALLOCATOR
    return preset, ALLOCATOR_PRESETS[preset]


def allocator_env(env: Mapping[str, str], settings: dict) -> dict[str, str]:
    """The environment a backend process should start CUDA under.

    **One rule for starting and restarting.** The launcher applies it when it
    starts the backend -- and the service unit starts the backend through the
    launcher -- and Restart Backend (``backend_control.resolve_restart_env``)
    applies it again, so a start, a crash restart, a reboot and a UI restart
    resolve the same saved preset.

    An explicit override is an allocator variable that is present and not
    marked ``"preset"``; it is returned untouched, marked ``"env"`` or not.
    Anything else gets the saved preset: no allocator variable at all -- a
    marker without a value overrides nothing -- or a value this rule set
    earlier, marked ``"preset"``, which is re-resolved so a newly saved preset
    takes effect.
    """
    new_env = dict(env)
    marker = new_env.get(ALLOC_SOURCE_ENV)
    has_env_alloc = (
        "PYTORCH_ALLOC_CONF" in new_env or "PYTORCH_CUDA_ALLOC_CONF" in new_env
    )
    if not has_env_alloc or marker == "preset":
        _preset, conf = resolve_allocator(settings.get("allocator_preset"))
        new_env["PYTORCH_ALLOC_CONF"] = conf
        new_env[ALLOC_SOURCE_ENV] = "preset"
    return new_env


__all__ = [
    "REPO_ROOT",
    "RUNTIME_SETTINGS_FILE",
    "ALLOCATOR_PRESETS",
    "DEFAULT_ALLOCATOR",
    "load_runtime_settings",
    "save_runtime_settings",
    "resolve_allocator",
    "ALLOC_SOURCE_ENV",
    "allocator_env",
]
