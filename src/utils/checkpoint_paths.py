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

import math
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

ScoreFn = Callable[[Path], Optional[float]]

from src.config.model_types import DEFAULT_CONV_TYPE, SUPPORTED_CONV_TYPES

PathLike = Union[str, Path]

# Ranking metrics used to pick the best checkpoint, in priority order, ALL
# "higher is better". The best checkpoint is decided from a checkpoint's own
# ``logs`` metadata — NOT from the filename. The number in ``model-<epoch>-
# <value>.pt`` has no stable meaning (it is whatever ModelCheckpoint's
# monitor/filename is configured to, e.g. val_mrr here, val_loss by default), so
# it must never be used to choose a model. val_loss is deliberately excluded — it
# is a poor quality signal for this task; ranking metrics are what matter.
RANKING_SCORE_KEYS = ("val_mrr", "val_hits@10", "val_hits@1")


def ranking_score_detail(logs) -> Optional[Tuple[str, float]]:
    """Return ``(metric_name, value)`` from a checkpoint's logs, or None.

    Picks the first available ranking metric (higher is better); ignores
    missing, non-numeric, and non-finite values.
    """
    if not isinstance(logs, dict):
        return None
    for key in RANKING_SCORE_KEYS:
        value = logs.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return key, float(value)
    return None


def ranking_score_from_logs(logs) -> Optional[float]:
    """Ranking score (higher is better) from a checkpoint's logs, or None."""
    detail = ranking_score_detail(logs)
    return detail[1] if detail else None


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


def select_checkpoint_in_dir(
    ckpt_dir: PathLike, score_fn: Optional[ScoreFn] = None
) -> Optional[Path]:
    """Pick the checkpoint to serve from one directory.

    When ``score_fn`` is given, every candidate is scored by it (``score_fn(path)
    -> Optional[float]``, **higher is better**) and the best-scoring one wins;
    ties break on newer mtime, then filename. Scoring exceptions and non-finite /
    ``None`` scores are ignored, so one unreadable checkpoint can't block the rest.

    When no candidate yields a valid score (or ``score_fn`` is None), fall back to
    ``last.pt``, then the newest ``*.pt``. Selection NEVER parses a metric from the
    filename — that number's meaning is config-dependent and unreliable.
    """
    d = Path(ckpt_dir)
    if not d.is_dir():
        return None
    pts = list(d.glob("*.pt"))
    if not pts:
        return None

    if score_fn is not None:
        scored: List[Tuple[float, float, str, Path]] = []
        for p in pts:
            try:
                score = score_fn(p)
            except Exception:  # a bad checkpoint must not block the others
                score = None
            if isinstance(score, (int, float)) and not isinstance(score, bool) \
                    and math.isfinite(score):
                scored.append((float(score), p.stat().st_mtime, p.name, p))
        if scored:
            best = max(scored, key=lambda t: (t[0], t[1], t[2]))
            return best[3]

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
    base_checkpoints_dir: PathLike, score_fn: Optional[ScoreFn] = None
) -> Tuple[Optional[Path], Optional[str], str]:
    """Auto selection: the most-recently-trained architecture, serving its best.

    The architecture is chosen by mtime (latest trained); within it the best
    checkpoint is chosen by ``score_fn`` (see ``select_checkpoint_in_dir``).
    Metrics from different architectures are not compared directly.

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
    selected = select_checkpoint_in_dir(arch_dir, score_fn=score_fn)
    return selected, arch_dir.name, f"auto: latest-trained architecture '{arch_dir.name}'"


__all__ = [
    "RANKING_SCORE_KEYS",
    "ranking_score_detail",
    "ranking_score_from_logs",
    "normalize_conv_type",
    "resolve_checkpoint_dir",
    "select_checkpoint_in_dir",
    "select_auto_checkpoint",
]
