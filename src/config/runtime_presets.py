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

#: The variables CUDA's allocator is configured by, in the order PyTorch
#: reads them: the first one present wins, an empty value included.
ALLOCATOR_VARIABLES = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")


def allocator_variable(env: Mapping[str, str]) -> str | None:
    """**Whether `env` sets CUDA's allocator, and by which variable**: the
    first of `ALLOCATOR_VARIABLES` that is present, its value empty or not
    (an empty value is still a setting PyTorch reads), or None. The one test
    `allocator_env` writes by and `running_allocator` reads by."""
    return next((name for name in ALLOCATOR_VARIABLES if name in env), None)


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


def allocator_env(
    env: Mapping[str, str], settings: dict, *, inherit: bool = False
) -> dict[str, str]:
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

    ``inherit=True`` is for a process the backend starts, such as a training
    run. It keeps any allocator already in its environment, a ``"preset"`` one
    included: the contract is that such a process runs under the backend's
    allocator, which changes when the backend restarts (Runtime Settings). It
    applies the saved preset only when nothing was inherited -- a training run
    started from a shell.
    """
    new_env = dict(env)
    marker = new_env.get(ALLOC_SOURCE_ENV)
    has_env_alloc = allocator_variable(new_env) is not None
    if not has_env_alloc or (marker == "preset" and not inherit):
        _preset, conf = resolve_allocator(settings.get("allocator_preset"))
        new_env["PYTORCH_ALLOC_CONF"] = conf
        new_env[ALLOC_SOURCE_ENV] = "preset"
    return new_env


def running_allocator(env: Mapping[str, str]) -> dict:
    """**What a process started under `env` runs CUDA's allocator as**: the
    read side of `allocator_env`, which wrote `env` when the process started.

    Returns ``{"variable", "conf", "preset", "source"}``: the allocator
    variable that is set (`allocator_variable`, the test `allocator_env`
    writes by: ``PYTORCH_ALLOC_CONF`` first, as PyTorch reads it, present
    even when empty) and its value; the preset whose value it is (None for a
    value no preset gives, an empty one included); and where it came from, by
    `allocator_env`'s own rule: ``"preset"`` (marked so: the saved preset at
    the process's start), ``"env"`` (any other allocator variable: an explicit
    setting in the launch environment, which a restart keeps) or None (no
    allocator variable: PyTorch's default backend). A saved preset changed
    since then is not what runs until the process starts again.
    """
    variable = allocator_variable(env)
    conf = env[variable] if variable else None
    preset = next((name for name, value in ALLOCATOR_PRESETS.items() if value == conf), None)
    source = None if conf is None else (
        "preset" if env.get(ALLOC_SOURCE_ENV) == "preset" else "env")
    return {"variable": variable, "conf": conf, "preset": preset, "source": source}


__all__ = [
    "REPO_ROOT",
    "RUNTIME_SETTINGS_FILE",
    "ALLOCATOR_PRESETS",
    "DEFAULT_ALLOCATOR",
    "load_runtime_settings",
    "save_runtime_settings",
    "resolve_allocator",
    "ALLOC_SOURCE_ENV",
    "ALLOCATOR_VARIABLES",
    "allocator_env",
    "allocator_variable",
    "running_allocator",
]
