"""
Checkpoint path derivation & selection — dependency-light.
==========================================================
Single source of truth for the on-disk checkpoint layout, shared by the trainer
(write path), the training service (listing), and the inference reload route
(read path) so all three agree:

    {data_dir}/checkpoints/{conv_type}/     e.g.  .../checkpoints/hgt/

Splitting by architecture stops different conv types (HGT/GAT/SAGE) from
overwriting each other's ``last.pt`` and intermixing their ``model-*.pt`` best
checkpoints in one flat directory.

No torch / PyG imports — safe to import from training scripts and API services.

Legacy note:
    Flat checkpoints written before this layout (``{data_dir}/checkpoints/*.pt``)
    are intentionally NOT handled by a runtime fallback here. That set is finite
    and frozen — no new flat checkpoints are ever produced — so rather than carry
    permanent fallback code + tests, legacy checkpoints are loaded via an
    explicit checkpoint path (already supported by the reload API) or retired
    once via ``scripts/migrate_checkpoints.py``.

Module: src/utils/checkpoint_paths.py
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

from src.config.model_types import DEFAULT_CONV_TYPE, SUPPORTED_CONV_TYPES

PathLike = Union[str, Path]

# best-validation checkpoint filename written by ModelCheckpoint:
# ``model-<epoch>-<val_loss>.pt`` (e.g. model-11-8.6543.pt).
_MODEL_CKPT_RE = re.compile(r"^model-\d+-(?P<val>[0-9]+(?:\.[0-9]+)?)\.pt$")


def normalize_conv_type(conv_type: Optional[str]) -> str:
    """Lower-case a conv type.

    Empty/missing falls back to the default, but an explicit unsupported value
    (e.g. ``"gatv2"``, ``"bad"``) raises rather than silently degrading to GAT —
    a mislabelled checkpoint dir or a wrong-architecture load is worse than a
    loud error.
    """
    ct = str(conv_type or "").strip().lower()
    if not ct:
        return DEFAULT_CONV_TYPE
    if ct not in SUPPORTED_CONV_TYPES:
        raise ValueError(
            f"Unsupported conv_type {conv_type!r}; supported: {SUPPORTED_CONV_TYPES}"
        )
    return ct


def resolve_checkpoint_dir(
    data_dir: PathLike,
    conv_type: Optional[str],
    explicit_dir: Optional[PathLike] = None,
) -> Path:
    """Where a training run's checkpoints live.

    An explicit dir is honoured verbatim (the user chose it — no conv_type is
    appended). Otherwise the architecture-scoped default
    ``{data_dir}/checkpoints/{conv_type}`` is used.
    """
    if explicit_dir:
        return Path(explicit_dir)
    return Path(data_dir) / "checkpoints" / normalize_conv_type(conv_type)


def select_checkpoint_in_dir(ckpt_dir: PathLike) -> Optional[Path]:
    """Pick the checkpoint to serve from one directory.

    Preference: best-validation ``model-*.pt`` (lowest val_loss) → ``last.pt`` →
    newest ``*.pt`` by mtime. Best-val is preferred for inference over the
    final-epoch ``last.pt``.
    """
    d = Path(ckpt_dir)
    if not d.is_dir():
        return None
    pts = list(d.glob("*.pt"))
    if not pts:
        return None

    best: Optional[Path] = None
    best_val: Optional[float] = None
    for p in pts:
        m = _MODEL_CKPT_RE.match(p.name)
        if not m:
            continue
        try:
            val = float(m.group("val"))
        except ValueError:
            continue
        if best_val is None or val < best_val:
            best_val, best = val, p
    if best is not None:
        return best

    last = d / "last.pt"
    if last.exists():
        return last
    return max(pts, key=lambda p: p.stat().st_mtime)


def _architecture_dirs(base: Path) -> List[Path]:
    """Valid architecture subdirs under a checkpoints/ dir (ignores stray names)."""
    if not base.is_dir():
        return []
    return [d for d in base.iterdir() if d.is_dir() and d.name in SUPPORTED_CONV_TYPES]


def select_auto_checkpoint(
    base_checkpoints_dir: PathLike,
) -> Tuple[Optional[Path], Optional[str], str]:
    """Auto selection: the most-recently-trained architecture, serving its best.

    Returns ``(checkpoint_path, conv_type, reason)``. Only architecture subdirs
    are considered; legacy flat checkpoints are handled out-of-band (explicit
    path / migration), not by a runtime fallback here.
    """
    base = Path(base_checkpoints_dir)
    ranked: List[Tuple[float, Path]] = []
    for d in _architecture_dirs(base):
        pts = list(d.glob("*.pt"))
        if pts:
            ranked.append((max(p.stat().st_mtime for p in pts), d))
    if not ranked:
        return None, None, "no architecture-subdir checkpoints found"
    ranked.sort(key=lambda t: t[0], reverse=True)
    arch_dir = ranked[0][1]
    selected = select_checkpoint_in_dir(arch_dir)
    return selected, arch_dir.name, f"auto: latest-trained architecture '{arch_dir.name}'"


__all__ = [
    "normalize_conv_type",
    "resolve_checkpoint_dir",
    "select_checkpoint_in_dir",
    "select_auto_checkpoint",
]
