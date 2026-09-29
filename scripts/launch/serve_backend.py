#!/usr/bin/env python3
"""
Start the backend as the service unit does, under the saved allocator preset.
=============================================================================
    python scripts/launch/serve_backend.py --host 127.0.0.1 --port 8264

Run from the repository root. Arguments go to uvicorn unchanged, after the app.

**Why this exists.** CUDA reads ``PYTORCH_ALLOC_CONF`` when it first allocates,
so the allocator has to be in the environment before anything imports torch.
The launcher (``shep_launch.py``) puts it there. A bare
``python -m uvicorn src.api.main:app`` puts nothing there, so it ran the native
allocator -- which fragments under HGT training -- until someone pressed
Restart Backend. This applies the rule Restart Backend applies
(``allocator_env``) and then replaces itself with exactly that bare command. The
process left running is the one ``backend_control`` knows how to restart, and a
systemd start, a crash restart, a reboot and a UI restart all resolve the same
saved preset.

**Nothing here may import torch or ``src.api``** before ``os.execve``;
importing ``src.api`` builds the whole application. Only
``src.config.runtime_presets`` is imported, which is dependency-light by
design. The launcher's accelerator installs and browser are deliberately not
here.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.runtime_presets import allocator_env, load_runtime_settings  # noqa: E402

UVICORN_APP = "src.api.main:app"


def exec_plan(
    env: Mapping[str, str],
    settings: dict,
    args: Sequence[str],
    executable: str = sys.executable,
) -> tuple[str, list[str], dict]:
    """The program, argv and environment the backend is started with."""
    command = [executable, "-m", "uvicorn", UVICORN_APP, *args]
    return executable, command, allocator_env(env, settings)


def main(argv: Sequence[str] | None = None) -> None:
    executable, command, env = exec_plan(
        os.environ,
        load_runtime_settings(),
        sys.argv[1:] if argv is None else argv,
    )
    os.execve(executable, command, env)


if __name__ == "__main__":
    main()
