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

import copy
import math
import re
import secrets
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

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


def _mtime(path: Path) -> float:
    """A listed file's modification time, for ordering only; 0 when it cannot
    be read: a link whose target is gone, or a checkpoint training removed
    (beyond `save_top_k`) between the listing and this call. Such a file sorts
    last and stays listed; a reader says what became of it."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _run_recency(run_dir: Path) -> float:
    """When the run last wrote a checkpoint (0 for none)."""
    return max((_mtime(p) for p in run_checkpoints(run_dir)), default=0.0)


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
        files = sorted(run_checkpoints(run), key=lambda p: (_mtime(p), p.name), reverse=True)
        entries.extend((run, path) for path in files)
    return entries


class ListedCheckpoint(NamedTuple):
    """One checkpoint of the listing, as one read of its file found it.

    `sha256` is the digest of the bytes that read returned: the model
    fingerprint, which a ledger record and a served identity name. The rest is
    the file's own record, parsed from those same bytes: its kind, its epoch,
    its `logs` (`{}` when it records none), the Trainer's validation figure
    (`ranking_score_detail` of those logs: the trainer's top-20 selection
    metric, never a test result) and the dataset fingerprint it was trained on
    (`training_input_digests["split_manifest"]`, the SHA-256 of that
    workspace's `split_manifest.json`).

    **A file the read refuses is listed, never dropped:** `problem` says why
    (a format `weights_only=True` refuses, a file gone since it was listed),
    and the digest and record are `None`. A file that loads as something other
    than a checkpoint record keeps its digest and says so in `problem`. A file
    that resolves outside the directory the caller bounds the listing by is
    listed with `problem` and is not read at all (`read_run_checkpoints`'
    `within`).

    `size_bytes` is the file's size as listed (a `stat` beside the read,
    descriptive only); `None` for a file not read because it is outside, or one
    that cannot be found.
    """

    run_dir: Path
    path: Path
    sha256: Optional[str] = None
    checkpoint_kind: Optional[str] = None
    epoch: Any = None
    logs: Optional[Dict[str, Any]] = None
    validation: Optional[Tuple[str, float]] = None
    dataset_fingerprint: Optional[str] = None
    problem: Optional[str] = None
    size_bytes: Optional[int] = None


#: How much of a refusal's text a listing keeps: its first line, bounded, so a
#: multi-paragraph loader message stays one readable reason.
_PROBLEM_LIMIT = 300


def _first_line(exc: BaseException) -> str:
    text = next((line.strip() for line in str(exc).splitlines() if line.strip()), "")
    if len(text) > _PROBLEM_LIMIT:
        text = text[:_PROBLEM_LIMIT - 3] + "..."
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _outside(path: Path, within: Path) -> Optional[str]:
    """Why `path` is not read under `within` (resolved), or `None` when it
    resolves inside it."""
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError) as exc:  # a link loop: RuntimeError before 3.13
        return f"cannot be resolved ({type(exc).__name__}), so it is not read"
    if resolved == within or within not in resolved.parents:
        return "resolves outside the workspace, so it is not read"
    return None


def _size(path: Path) -> Optional[int]:
    try:
        return path.stat().st_size
    except OSError:
        return None


#: Each listed file's last read in this process, by its path: the file's
#: identity as `os.stat` gave it around that read, and what the read found
#: (`ListedCheckpoint` without its two paths). Least recently listed first.
_READS: "OrderedDict[str, Tuple[Tuple[int, ...], Dict[str, Any]]]" = OrderedDict()
_READS_LOCK = threading.Lock()
#: How many files' reads are kept (a few kilobytes each).
READS_KEPT = 4096
#: A file whose content or metadata changed less than this long before its
#: read is read again at every listing: a rewrite within one tick of the file
#: system's clock could leave its `stat` unchanged.
SETTLED_NS = 2_000_000_000


def _identity(path: Path) -> Optional[Tuple[int, ...]]:
    """What tells one state of a file from another without reading it: its
    device, inode, size, and modification and change times (the change time
    moves with any write or metadata change and cannot be set back)."""
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _recalled(path: Path, identity: Optional[Tuple[int, ...]]) -> Optional[Dict[str, Any]]:
    if identity is None:
        return None
    with _READS_LOCK:
        kept = _READS.get(str(path))
        if kept is None or kept[0] != identity:
            return None
        _READS.move_to_end(str(path))
        fields = kept[1]
    return {**fields, "logs": copy.deepcopy(fields["logs"])}


def _remember(path: Path, before: Optional[Tuple[int, ...]], started_ns: int,
              entry: ListedCheckpoint) -> None:
    """Keep `entry` for `path` when the file did not change during the read
    and had settled before it (`SETTLED_NS`)."""
    if before is None or _identity(path) != before:
        return
    if max(before[3], before[4]) > started_ns - SETTLED_NS:
        return
    fields = entry._asdict()
    del fields["run_dir"], fields["path"]
    fields["logs"] = copy.deepcopy(fields["logs"])
    with _READS_LOCK:
        _READS[str(path)] = (before, fields)
        _READS.move_to_end(str(path))
        while len(_READS) > READS_KEPT:
            _READS.popitem(last=False)


def _read_one(run: Path, path: Path) -> ListedCheckpoint:
    """One file's entry, from one read of its bytes."""
    from src.utils.checkpoint_io import read_checkpoint

    size = _size(path)
    try:
        read = read_checkpoint(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 — listed with its reason, never dropped
        return ListedCheckpoint(run, path, problem=_first_line(exc), size_bytes=size)
    meta = read.checkpoint
    if not isinstance(meta, dict):
        return ListedCheckpoint(
            run, path, sha256=read.identity.sha256,
            problem=(f"it loads as {type(meta).__name__}, not a checkpoint record, so it "
                     "carries no metadata"),
            size_bytes=size)
    logs = meta.get("logs") if isinstance(meta.get("logs"), dict) else {}
    digests = meta.get("training_input_digests")
    fingerprint = digests.get("split_manifest") if isinstance(digests, dict) else None
    return ListedCheckpoint(
        run, path, sha256=read.identity.sha256,
        checkpoint_kind=meta.get("checkpoint_kind"),
        epoch=meta.get("epoch"),
        logs=logs,
        validation=ranking_score_detail(logs),
        dataset_fingerprint=fingerprint if isinstance(fingerprint, str) else None,
        size_bytes=size,
    )


def read_run_checkpoints(
    data_dir: PathLike, conv_type: Optional[str] = None, *,
    within: Optional[PathLike] = None,
) -> List[ListedCheckpoint]:
    """The listing (`list_run_checkpoints`, same order), each file read once,
    and again only when it changes.

    Each file is loaded through `read_checkpoint` with `weights_only=True`
    (tensors and plain data only, contract decision 10) on the CPU, and what
    the entry shows and its digest come from that one read (BACKLOG 19.3), so
    a file rewritten while it is listed is never shown with another file's
    digest. Reading opens every checkpoint, which SPEC_4 classes as checkpoint
    inspection (C3): a caller that offers this list over the network is gated
    as D8 says. `list_run_checkpoints` stays the read-free form, for a caller
    that only needs to know which files exist.

    **A file is not read again while it is unchanged** (M5-3 review, UI-2:
    every opening of a model list read every checkpoint of the workspace, so
    the lists slowed with every run). What a read found is kept in this
    process by the file's path and its identity (`_identity`: device, inode,
    size, modification and change times, taken before and after the read), and
    a later listing that finds the same identity shows it without opening the
    file; any write, replacement or metadata change reads it again, as does a
    file that changed in the `SETTLED_NS` before its read. The bytes are still
    checked where they are used: a load, a start or a measurement re-reads them
    against the digest shown. The kept reads are shown only through the
    callers that read them, which D8 gates as before.

    `within`, the workspace directory, bounds what is read: a file that
    resolves outside it (a run or a checkpoint that is a link out of the
    workspace) is listed with that reason and **never opened**, so nothing of
    a file elsewhere is read or shown. The check runs before the read (and
    before a kept read is used); the server's list passes its workspace
    (`server_lists.list_checkpoints`).

    torch is imported only here, when a file is read, so this module stays
    importable without it.
    """
    bound = Path(within).resolve() if within is not None else None
    entries: List[ListedCheckpoint] = []
    for run, path in list_run_checkpoints(data_dir, conv_type):
        outside = _outside(path, bound) if bound is not None else None
        if outside is not None:
            entries.append(ListedCheckpoint(
                run, path, problem=f"{run.name}/{path.name} {outside}"))
            continue
        identity = _identity(path)
        kept = _recalled(path, identity)
        if kept is not None:
            entries.append(ListedCheckpoint(run, path, **kept))
            continue
        started_ns = time.time_ns()
        entry = _read_one(run, path)
        _remember(path, identity, started_ns, entry)
        entries.append(entry)
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
    "ListedCheckpoint",
    "RunDirectoryTaken",
    "allocate_run_dir",
    "check_run_dir_free",
    "claim_run_dir",
    "list_run_checkpoints",
    "list_runs",
    "propose_run_dir",
    "read_run_checkpoints",
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
