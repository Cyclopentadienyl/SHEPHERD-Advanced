#!/usr/bin/env python3
"""
The deployment's data locations as this shell reads them
=========================================================
UI-06 (the owner's decision of 2026-11-16). The data locations -- the
workspaces folder, the results registry, the import folder and the training
evidence folders -- are chosen in the dashboard's Settings › Data locations
and kept in one small file outside the repository; an environment variable
set when a process starts still wins. This command prints what **this shell**
reads, by the service's one resolution (`src.config.server_settings`), in the
same view the dashboard and ``GET /api/v1/settings/deployment`` give
(`src.utils.deployment_view`), so a shell, the service and a job it starts
can be compared::

    python scripts/deployment_settings.py show          # one fact per line
    python scripts/deployment_settings.py show --json   # the route's fields

For each location: the value in effect and its source (an environment
variable of this shell, which wins; else the saved file's applied value;
else not set), the saved value and whether it is pending, whether the
environment overrides a saved value, and the problem that turns its feature
off. Then the saved settings file (``SHEPHERD_SETTINGS_FILE`` when set, else
the account's user configuration file), the deployment mode and the GPU lock.

It reads only, and shows the paths: it runs as the account whose files they
are. Changing a location is the dashboard's (an operator control, D8); a
command that takes a location (``--registry``, ``--training-evidence``) is
never refused because the dashboard was not set up, when it is given one.

Exit status: 0 when the view was printed, whatever it says (a location not
set is a fact of the view, not a failure of the command); 2 for a usage
error.

Module: scripts/deployment_settings.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import deployment_view  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print the deployment's data locations as this shell reads them "
                    "(Settings › Data locations; an environment variable set here wins).")
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser("show", help="print each location in effect, its source, the "
                                            "saved and pending values and its problem")
    show.add_argument("--json", action="store_true",
                      help="print the view as JSON, the fields of GET /api/v1/settings/deployment")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    view = deployment_view.deployment_view()
    if args.json:
        print(json.dumps(deployment_view.as_dict(view, paths_shown=True), indent=2,
                         ensure_ascii=False))
    else:
        print("\n".join(deployment_view.text_lines(view)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
