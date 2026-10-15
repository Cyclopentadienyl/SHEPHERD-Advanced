"""
Checkpoint path derivation & selection — dependency-light.
==========================================================
Single source of truth for the on-disk run layout, shared by the training entry
point (write path), the training service and console (listing, resume source),
and the inference reload route (read path), so all of them agree:

    {data_dir}/runs/{conv_type}/{date}_{name}_{run_id}/
        config.yaml  runtime.json  final_metrics.json
        checkpoints/   last.pt, model-*.pt
        logs/          train_*.json, progress.json

**One directory per training run** (contract decision 9, M3a-2). A run, new or
resumed, writes only into a directory no run has used, so no run overwrites or
prunes another's files. Runs sit side by side under their architecture, never
nested by lineage: a resumed run records its parent by SHA-256, not by where it
sits.

No torch / PyG imports — safe to import from training scripts and API services.

Old layouts:
    Checkpoints written before this layout (``{data_dir}/checkpoints/{conv}/``
    and the older flat ``{data_dir}/checkpoints/*.pt``) are intentionally NOT
    found by a runtime fallback here. Nothing writes there any more; such a file
    is reached by an explicit checkpoint path, which every reader accepts and
    checks by its bytes, or moved once, by the owner, with
    ``scripts/migrate_checkpoints.py`` (decision 9). Not being listed does not
    make a file invalid.

Module: src/utils/checkpoint_paths.py
"""
from __future__ import annotations

import math
import re
import secrets
from datetime import datetime
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


# =============================================================================
# Run directories (contract M3a-2)
# =============================================================================
#: Every training run, new or resumed, writes into a directory of its own:
#: ``{data_dir}/runs/{conv_type}/{date}_{name}_{run_id}/``, holding the run's
#: ``config.yaml``, ``runtime.json`` and ``final_metrics.json``, its
#: ``checkpoints/`` and its ``logs/``. Directories are flat, never nested by
#: lineage: a resumed run records its parent by digest, not by where it sits.
#: The name is for people; a run is its run id, a checkpoint its SHA-256.
RUNS_DIRNAME = "runs"
RUN_CHECKPOINTS_DIRNAME = "checkpoints"
RUN_LOGS_DIRNAME = "logs"
#: Six hex digits. A collision is drawn again, never reused.
_RUN_ID_BYTES = 3
_RUN_NAME_LIMIT = 40


def slugify_run_name(name: Optional[str]) -> str:
    """A run name as it appears in a directory name: lower case ASCII letters,
    digits and single hyphens, at most 40 characters; empty for no name."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(name or "").strip().lower()).strip("-")
    return slug[:_RUN_NAME_LIMIT].strip("-")


def runs_root(data_dir: PathLike, conv_type: Optional[str]) -> Path:
    """``{data_dir}/runs/{conv_type}``: where an architecture's runs live."""
    return Path(data_dir) / RUNS_DIRNAME / normalize_conv_type(conv_type)


#: Output locations retired by the run layout. A configuration that still sets
#: one is refused rather than ignored: the run would otherwise write somewhere
#: other than where its author looks.
RETIRED_LOCATION_FIELDS = ("output_dir", "checkpoint_dir", "log_dir")


def refuse_retired_locations(values, *, source: str) -> None:
    """Refuse a configuration (a mapping) that sets an output location the run
    layout retired. An empty value is the old default and says nothing.

    Raises:
        ValueError: naming the fields and what replaced them.
    """
    named = sorted(k for k in RETIRED_LOCATION_FIELDS
                   if isinstance(values, dict) and values.get(k) not in (None, ""))
    if named:
        raise ValueError(
            f"{source} sets {', '.join(named)}, which no longer exist: every run writes "
            "into a directory of its own, {data_dir}/runs/{conv_type}/..., with its "
            "checkpoints/ and logs/ inside. Remove them; give run_name to name the run, "
            "or run_dir for an explicit directory that is absent or empty."
        )


class RunDirectoryTaken(FileExistsError):
    """A run directory that is not free: it holds files, or another run claimed
    it first. A run never writes into another run's directory."""


def propose_run_dir(
    data_dir: PathLike,
    conv_type: Optional[str],
    name: Optional[str] = None,
    *,
    now: Optional[datetime] = None,
) -> Path:
    """A new run directory's path, not yet created:
    ``{data_dir}/runs/{conv_type}/{YYYYMMDD-HHMMSS}_{name or 'run'}_{6 hex}``.

    A name already on disk is drawn again. Proposing claims nothing:
    `claim_run_dir` does, so a launcher (the training service) can name the
    directory before the run that writes it has passed its refusals.
    """
    root = runs_root(data_dir, conv_type)
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    label = slugify_run_name(name) or "run"
    for _ in range(100):
        candidate = root / f"{stamp}_{label}_{secrets.token_hex(_RUN_ID_BYTES)}"
        if not candidate.exists():
            return candidate
    raise RunDirectoryTaken(f"no unused run directory under {root} after 100 tries")


def check_run_dir_free(run_dir: PathLike) -> Path:
    """Refuse a run directory that is not free: an existing file, or a
    directory holding anything. Absent or empty is free. Creates nothing, so an
    explicitly given location is refused before any run artifact exists;
    `claim_run_dir` decides again when it creates."""
    path = Path(run_dir)
    if path.exists():
        if not path.is_dir():
            raise RunDirectoryTaken(f"the run directory {path} exists and is not a directory")
        if any(path.iterdir()):
            raise RunDirectoryTaken(
                f"the run directory {path} already holds files. A run writes only into a "
                "directory no run has used; name a new one, or leave it empty to have "
                "one allocated.")
    return path


def claim_run_dir(run_dir: PathLike) -> Path:
    """Create the run directory and its ``checkpoints/`` and ``logs/``, or refuse.

    **Claimed by creating ``checkpoints/``** (`mkdir` without `exist_ok`): of two
    runs handed the same free directory, one creates it and the other is refused,
    so neither writes into the other's files. The caller claims only after every
    refusal, since this is the run's first artifact.
    """
    path = check_run_dir_free(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    try:
        run_checkpoint_dir(path).mkdir()
    except FileExistsError:
        raise RunDirectoryTaken(f"the run directory {path} was claimed by another run") from None
    run_log_dir(path).mkdir(exist_ok=True)
    return path


def allocate_run_dir(
    data_dir: PathLike,
    conv_type: Optional[str],
    name: Optional[str] = None,
    *,
    now: Optional[datetime] = None,
) -> Path:
    """Propose and claim a new run directory; a directory claimed in between
    by another run is drawn again, never shared."""
    for _ in range(100):
        try:
            return claim_run_dir(propose_run_dir(data_dir, conv_type, name, now=now))
        except RunDirectoryTaken:
            continue
    raise RunDirectoryTaken(
        f"no unused run directory under {runs_root(data_dir, conv_type)} after 100 tries")


def run_checkpoint_dir(run_dir: PathLike) -> Path:
    """A run's ``checkpoints/``."""
    return Path(run_dir) / RUN_CHECKPOINTS_DIRNAME


def run_log_dir(run_dir: PathLike) -> Path:
    """A run's ``logs/``."""
    return Path(run_dir) / RUN_LOGS_DIRNAME


def run_checkpoints(run_dir: PathLike) -> List[Path]:
    """A run's checkpoint files, by name."""
    directory = run_checkpoint_dir(run_dir)
    return sorted(directory.glob("*.pt")) if directory.is_dir() else []


def _run_recency(run_dir: Path) -> float:
    """When the run last wrote a checkpoint (0 for none)."""
    return max((p.stat().st_mtime for p in run_checkpoints(run_dir)), default=0.0)


def list_runs(data_dir: PathLike, conv_type: Optional[str] = None) -> List[Path]:
    """A workspace's run directories, most recently trained first.

    Every architecture's, or one's. Ordered by the newest checkpoint each holds,
    then by name, latest first (the name starts with the run's start time), so
    the order never depends on the filesystem's listing order.
    """
    base = Path(data_dir) / RUNS_DIRNAME
    architectures = ([normalize_conv_type(conv_type)] if conv_type
                     else list(SUPPORTED_CONV_TYPES))
    runs: List[Path] = []
    for architecture in architectures:
        directory = base / architecture
        if directory.is_dir():
            runs.extend(p for p in directory.iterdir() if p.is_dir())
    return sorted(runs, key=lambda r: (_run_recency(r), r.name), reverse=True)


def run_architecture(run_dir: PathLike) -> str:
    """The architecture a run directory belongs to (its parent's name)."""
    return Path(run_dir).parent.name


def list_run_checkpoints(
    data_dir: PathLike, conv_type: Optional[str] = None
) -> List[Tuple[Path, Path]]:
    """Every checkpoint in a workspace's runs, as ``(run_dir, checkpoint)``:
    runs most recently trained first (`list_runs`), and within a run the newest
    file first, then by name. The one listing the console, the training service
    and the API show; what a file is comes from its own record, read by the
    caller, never from where it sits."""
    entries: List[Tuple[Path, Path]] = []
    for run in list_runs(data_dir, conv_type):
        files = sorted(run_checkpoints(run),
                       key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
        entries.extend((run, path) for path in files)
    return entries


def select_run_checkpoint(
    data_dir: PathLike, conv_type: Optional[str], score_fn: Optional[ScoreFn] = None
) -> Tuple[Optional[Path], str]:
    """One architecture's selection: its latest run, then the existing in-run
    ordering (`select_checkpoint_in_dir`). Runs are not ranked against each
    other by their metrics: they may differ in every setting."""
    architecture = normalize_conv_type(conv_type)
    runs = [r for r in list_runs(data_dir, architecture) if run_checkpoints(r)]
    if not runs:
        return None, f"no run of '{architecture}' has a checkpoint"
    run = runs[0]
    return (select_checkpoint_in_dir(run_checkpoint_dir(run), score_fn=score_fn),
            f"latest run '{run.name}' of '{architecture}'")


def select_auto_run_checkpoint(
    data_dir: PathLike, score_fn: Optional[ScoreFn] = None
) -> Tuple[Optional[Path], Optional[str], str]:
    """Auto selection over runs: the most recently trained run of any
    architecture, then the existing in-run ordering. Returns
    ``(checkpoint_path, conv_type, reason)``. Only the run layout is searched;
    checkpoints elsewhere are reached by an explicit path."""
    runs = [r for r in list_runs(data_dir) if run_checkpoints(r)]
    if not runs:
        return None, None, "no run has a checkpoint"
    run = runs[0]
    architecture = run_architecture(run)
    return (select_checkpoint_in_dir(run_checkpoint_dir(run), score_fn=score_fn),
            architecture, f"auto: latest run '{run.name}' ({architecture})")


__all__ = [
    "RANKING_SCORE_KEYS",
    "RETIRED_LOCATION_FIELDS",
    "RUNS_DIRNAME",
    "RUN_CHECKPOINTS_DIRNAME",
    "RUN_LOGS_DIRNAME",
    "RunDirectoryTaken",
    "allocate_run_dir",
    "check_run_dir_free",
    "claim_run_dir",
    "list_run_checkpoints",
    "list_runs",
    "propose_run_dir",
    "refuse_retired_locations",
    "run_architecture",
    "run_checkpoint_dir",
    "run_checkpoints",
    "run_log_dir",
    "runs_root",
    "select_auto_run_checkpoint",
    "select_run_checkpoint",
    "slugify_run_name",
    "ranking_score_detail",
    "ranking_score_from_logs",
    "normalize_conv_type",
    "select_checkpoint_in_dir",
]
