#!/usr/bin/env python3
"""
One-shot migration: move legacy flat checkpoints into architecture subdirs.
===========================================================================
Before the architecture-scoped layout, every conv type wrote to
``{workspace}/checkpoints/*.pt``, so different architectures overwrote each
other's ``last.pt`` and intermixed their best files. This moves each legacy flat
checkpoint into ``{workspace}/checkpoints/<conv_type>/``, determining the
conv_type from the checkpoint's own ``model_config`` (or, for pre-model_config
checkpoints, from its weight names).

Run ONCE per workspace. Idempotent — only top-level ``checkpoints/*.pt`` are
considered, so files already inside an architecture subdir are left alone.
Dry-run by default; pass ``--apply`` to actually move.

    python scripts/migrate_checkpoints.py data/workspaces/<ws>
    python scripts/migrate_checkpoints.py data/workspaces/<ws> --apply

This exists so the runtime code does NOT need a permanent legacy fallback: the
legacy flat set is finite and frozen (no new flat checkpoints are produced), so
it is retired once here rather than carried forever in the selection logic.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.model_types import DEFAULT_CONV_TYPE, SUPPORTED_CONV_TYPES


def _conv_type_of(ckpt_path: Path) -> str:
    """Determine a checkpoint's conv type from model_config, flat config, or weights."""
    import torch

    from src.inference.pipeline import _infer_conv_type_from_keys

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model_config = config.get("model_config") if isinstance(config, dict) else None
    if isinstance(model_config, dict) and model_config.get("conv_type"):
        return str(model_config["conv_type"]).strip().lower()
    if isinstance(config, dict) and config.get("conv_type"):
        return str(config["conv_type"]).strip().lower()
    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or {}
    return (_infer_conv_type_from_keys(set(state_dict.keys())) or DEFAULT_CONV_TYPE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("workspace", help="Workspace dir containing checkpoints/")
    parser.add_argument("--apply", action="store_true", help="Actually move (default: dry run)")
    args = parser.parse_args()

    ckpt_dir = Path(args.workspace) / "checkpoints"
    if not ckpt_dir.is_dir():
        print(f"No checkpoints directory: {ckpt_dir}")
        return 1

    flat = sorted(p for p in ckpt_dir.glob("*.pt") if p.is_file())
    if not flat:
        print(f"No legacy flat checkpoints to migrate under {ckpt_dir}.")
        return 0

    print(f"{'MOVING' if args.apply else 'DRY RUN'} — {len(flat)} flat checkpoint(s):")
    for p in flat:
        try:
            conv_type = _conv_type_of(p)
        except Exception as exc:  # noqa: BLE001 — report and skip a bad file
            print(f"  ! {p.name}: could not read ({exc}); skipping")
            continue
        if conv_type not in SUPPORTED_CONV_TYPES:
            print(f"  ! {p.name}: unrecognised conv_type '{conv_type}', using '{DEFAULT_CONV_TYPE}'")
            conv_type = DEFAULT_CONV_TYPE
        dest_dir = ckpt_dir / conv_type
        print(f"  {p.name}  ->  {conv_type}/{p.name}")
        if args.apply:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dest_dir / p.name))

    print("\nDone." if args.apply else "\nDry run — re-run with --apply to move the files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
