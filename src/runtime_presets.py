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
from pathlib import Path

# Repo root: src/runtime_presets.py -> parents[1] == repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
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


def effective_allocator(env: dict, settings: dict) -> tuple[str | None, str | None]:
    """Decide the allocator to apply, honouring explicit env overrides.

    If ``PYTORCH_ALLOC_CONF`` / ``PYTORCH_CUDA_ALLOC_CONF`` is already set in
    ``env``, returns ``(None, None)`` — the explicit override wins and nothing
    should be changed. Otherwise resolves the persisted preset (with fallback).
    """
    if "PYTORCH_ALLOC_CONF" in env or "PYTORCH_CUDA_ALLOC_CONF" in env:
        return None, None
    return resolve_allocator(settings.get("allocator_preset"))


__all__ = [
    "REPO_ROOT",
    "RUNTIME_SETTINGS_FILE",
    "ALLOCATOR_PRESETS",
    "DEFAULT_ALLOCATOR",
    "load_runtime_settings",
    "save_runtime_settings",
    "resolve_allocator",
    "effective_allocator",
]
