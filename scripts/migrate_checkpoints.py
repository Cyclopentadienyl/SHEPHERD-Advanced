#!/usr/bin/env python3
"""
One-time move of a workspace's old checkpoints into the run layout.
====================================================================
Contract M3a-2, decision 9 (`docs/working/PLAN_PROVENANCE_CONTRACT.md`). Every
training run now writes into a directory of its own,
``{workspace}/runs/{conv_type}/{date}_{name}_{run_id}/``, and nothing reads the
older ``{workspace}/checkpoints/`` layouts at run time: there is no fallback.
This tool moves those older files once, explicitly, by the owner.

**It lists a plan first and moves only with ``--apply``.** For each file under
``{workspace}/checkpoints/`` (flat or per architecture), from one read with the
safe loader:

- **M2's record checks pass** (serving's comparison of the recorded graph
  roles, ``verify_training_record``): moved into an *imported run* per source
  directory and architecture, labelled "M2 record checks pass; no M3 resume
  state; servability not judged". The comparison reads the recorded roles,
  not the weights; whether the file serves is decided by serving's own entry
  point, never by this move. The imported run's id is assigned here and is not
  in the file's bytes; settings the file does not record stay unknown, and
  nothing is written into a checkpoint.
- **M2 refuses it** (a recorded role that does not match, or no record):
  moved to ``{workspace}/archive/``, keeping its path under ``checkpoints/``.
- **The check cannot be completed** (the safe loader cannot open the file,
  the workspace's own graph is refused, the architecture cannot be told):
  *unverifiable*, left where it is and listed. That is not a finding that the
  model is invalid; such a file is still reached by its explicit path.

**A move keeps the bytes**: each file is renamed within the workspace
(``os.rename``, the same filesystem) and its SHA-256 checked again at the new
place, so every digest recorded about it still names it. **A destination that
already exists stops the whole move** before anything is moved, and the
sources stay; there is no ``--overwrite``. Nothing is deleted; emptied
directories are left. Saved settings that name a moved path are listed for the
owner to change.

    python scripts/migrate_checkpoints.py data/workspaces/<ws>
    python scripts/migrate_checkpoints.py data/workspaces/<ws> --apply
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Imported at start, not inside the per-file handler: a broken import must fail
# once, here, not be reported as an unreadable checkpoint for every file.
from src.config.model_types import SUPPORTED_CONV_TYPES, _infer_conv_type_from_keys  # noqa: E402

#: The label every imported run carries (decision 9).
IMPORTED_LABEL = "M2 record checks pass; no M3 resume state; servability not judged"
#: The label file in an imported run.
IMPORTED_RECORD = "imported.json"
#: Saved console settings that may name a checkpoint path.
SETTINGS_FILES = (".shepherd_ui_config.json", ".shepherd_training_config.json")


def _architecture(record: Any) -> Optional[str]:
    """The architecture a checkpoint records (model_config, then the flat
    config), or the one its weight names show; `None` when neither tells."""
    if not isinstance(record, dict):
        return None
    config = record.get("config") if isinstance(record.get("config"), dict) else {}
    model_config = config.get("model_config")
    if isinstance(model_config, dict) and model_config.get("conv_type"):
        return str(model_config["conv_type"]).strip().lower()
    if config.get("conv_type"):
        return str(config["conv_type"]).strip().lower()
    state_dict = record.get("state_dict") or record.get("model_state_dict") or {}
    return _infer_conv_type_from_keys(set(state_dict)) if isinstance(state_dict, dict) else None


def classify(workspace: Path) -> List[Dict[str, Any]]:
    """Every file under `workspace/checkpoints/`, read once with the safe
    loader and judged by M2's serving comparison."""
    import scripts.audit_checkpoint_family as audit
    from src.kg.artifacts import GRAPH_ARTIFACTS
    from src.utils.checkpoint_io import read_checkpoint
    from src.utils.fingerprint import verify_training_record

    checkpoints = workspace / "checkpoints"
    files = sorted(p for p in checkpoints.rglob("*.pt") if p.is_file())
    try:
        bound = audit._graph_check(workspace)
        graph_refused = None
    except Exception as exc:  # noqa: BLE001 - shown, never moves anything
        bound, graph_refused = None, f"{type(exc).__name__}: {exc}"

    entries = []
    for path in files:
        entry: Dict[str, Any] = {"source": path, "group": path.parent.relative_to(checkpoints)}
        try:
            read = read_checkpoint(path, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 - unverifiable, not invalid
            entry.update(kind="unverifiable",
                         reason=f"the safe loader cannot open it ({type(exc).__name__})")
            entries.append(entry)
            continue
        entry["sha256"] = read.identity.sha256
        if graph_refused is not None:
            entry.update(kind="unverifiable", reason=f"the workspace's graph is refused: "
                                                      f"{graph_refused}")
            entries.append(entry)
            continue
        verdict = audit._judge(lambda: verify_training_record(
            read.checkpoint, bound, tuple(GRAPH_ARTIFACTS),
            recorded_by=f"checkpoint {path.name}", compared_with="this workspace's graph",
        ), workspace)
        if verdict["verdict"] == "refused":
            entry.update(kind="archive", reason=verdict["reason"])
        else:
            architecture = _architecture(read.checkpoint)
            if architecture not in SUPPORTED_CONV_TYPES:
                entry.update(kind="unverifiable",
                             reason=f"its architecture cannot be told ({architecture!r})")
            else:
                entry.update(kind="import", architecture=architecture)
        entries.append(entry)
    return entries


def plan_moves(workspace: Path, entries: List[Dict[str, Any]],
               now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Destinations: one imported run per source directory and architecture,
    and `archive/` keeping the path under `checkpoints/`."""
    from src.utils.checkpoint_paths import propose_run_dir, run_checkpoint_dir

    runs: Dict[tuple, Path] = {}
    for entry in entries:
        source = entry["source"]
        if entry["kind"] == "archive":
            entry["destination"] = workspace / "archive" / source.relative_to(workspace)
        elif entry["kind"] == "import":
            key = (entry["architecture"], entry["group"])
            if key not in runs:
                group = "-".join(entry["group"].parts) or "flat"
                runs[key] = propose_run_dir(workspace, entry["architecture"],
                                            f"imported-{group}", now=now)
            entry["run_dir"] = runs[key]
            entry["destination"] = run_checkpoint_dir(runs[key]) / source.name
    return entries


def settings_naming(entries: List[Dict[str, Any]]) -> List[str]:
    """Saved console settings whose values name a file this move relocates."""
    moved = {e["source"].resolve(): e["destination"] for e in entries if "destination" in e}
    found = []
    for name in SETTINGS_FILES:
        path = REPO_ROOT / name
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for key, value in (values.items() if isinstance(values, dict) else []):
            if not isinstance(value, str) or not value.endswith(".pt"):
                continue
            candidate = Path(value) if Path(value).is_absolute() else REPO_ROOT / value
            target = moved.get(candidate.resolve())
            if target is not None:
                found.append(f"{name}: {key} = {value}  ->  now {target}")
    return found


def apply_moves(workspace: Path, entries: List[Dict[str, Any]]) -> None:
    """Move every planned file, or none: a destination that exists stops the
    move before anything moves. Each rename is checked by digest."""
    from src.utils.checkpoint_paths import RUN_CHECKPOINTS_DIRNAME
    from src.utils.fingerprint import file_sha256

    planned = [e for e in entries if "destination" in e]
    taken = [e["destination"] for e in planned if e["destination"].exists()]
    taken += sorted({e["run_dir"] for e in planned if "run_dir" in e and e["run_dir"].exists()})
    if taken:
        raise SystemExit("Stopped before moving anything: these destinations already exist, "
                         "and a move never replaces a file:\n  "
                         + "\n  ".join(str(t) for t in taken))

    for entry in planned:
        destination = entry["destination"]
        if "run_dir" in entry and not (entry["run_dir"] / RUN_CHECKPOINTS_DIRNAME).exists():
            (entry["run_dir"] / RUN_CHECKPOINTS_DIRNAME).mkdir(parents=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.rename(entry["source"], destination)
        if entry.get("sha256") and file_sha256(destination) != entry["sha256"]:
            raise SystemExit(f"{destination} does not have the bytes read before the move "
                             f"(sha256 {entry['sha256']}); stopped.")

    for run_dir in sorted({e["run_dir"] for e in planned if "run_dir" in e}):
        members = [e for e in planned if e.get("run_dir") == run_dir]
        (run_dir / IMPORTED_RECORD).write_text(json.dumps({
            "label": IMPORTED_LABEL,
            "imported_at": datetime.now().isoformat(timespec="seconds"),
            "files": [{"file": e["destination"].name,
                       "moved_from": str(e["source"].relative_to(workspace)),
                       "sha256": e["sha256"]} for e in members],
        }, indent=2) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("workspace", help="Workspace directory holding checkpoints/")
    parser.add_argument("--apply", action="store_true", help="Move the files (default: plan only)")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)
    if not (workspace / "checkpoints").is_dir():
        print(f"No checkpoints directory under {workspace}; nothing to move.")
        return 0
    entries = plan_moves(workspace, classify(workspace))
    if not entries:
        print(f"No checkpoints under {workspace / 'checkpoints'}; nothing to move.")
        return 0

    titles = {"import": f"Into imported runs ({IMPORTED_LABEL})",
              "archive": "To archive/ (M2 refuses them)",
              "unverifiable": "Left in place: unverifiable (not a finding that they are invalid)"}
    print(f"{'MOVING' if args.apply else 'PLAN (nothing is moved without --apply)'}: "
          f"{len(entries)} file(s) under {workspace / 'checkpoints'}")
    for kind, title in titles.items():
        chosen = [e for e in entries if e["kind"] == kind]
        if not chosen:
            continue
        print(f"\n{title}:")
        for e in chosen:
            source = e["source"].relative_to(workspace)
            if "destination" in e:
                print(f"  {source}  ->  {e['destination'].relative_to(workspace)}")
            else:
                print(f"  {source}")
            if e.get("reason"):
                print(f"      {e['reason']}")

    named = settings_naming(entries)
    if named:
        print("\nSaved settings that name a moved file (change them after the move):")
        for line in named:
            print(f"  {line}")

    if args.apply:
        apply_moves(workspace, entries)
        print("\nMoved. Every digest still names the same bytes.")
    else:
        print("\nRe-run with --apply to move. Imported run ids are drawn again then.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
