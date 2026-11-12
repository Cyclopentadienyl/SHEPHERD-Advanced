"""
Checkpoint path derivation & selection — dependency-light.
==========================================================
Single source of truth for the on-disk run layout, shared by the training entry
point (write path), the training service and console (listing, resume source),
and the inference reload route (read path), so all of them agree:

    {data_dir}/runs/{layout directory}/{date}_{name}_{run_id}/
        config.yaml  runtime.json  final_metrics.json
        checkpoints/   last.pt, model-*.pt
        logs/          train_*.json, progress.json

**One directory per training run** (contract decision 9, M3a-2). A run, new or
resumed, writes only into a directory no run has used, so no run overwrites or
prunes another's files. A new run is filed under its model spec's encoder kind
(`RUN_LAYOUT_KEYS`), never nested by lineage: a resumed run records its parent
by SHA-256, not by where it sits. **The listing reads every directory under
``runs/``**, whatever its name (`list_runs`): what a file is comes from its own
record, never from the directory it sits in. One it cannot list, a name that is
not valid text among them, is one entry saying why, and hides nothing else; so
is a run directory or a checkpoint file whose name is not valid text (BACKLOG
66). **An entry's identity is apart from the name it is shown by**
(`listing_key`): a name that is not text is shown with backslash escapes and
identified by its bytes, so two different names never share an identity.

No torch / PyG imports — safe to import from training scripts and API services.

Old layouts:
    Checkpoints written before this layout (``{data_dir}/checkpoints/{conv}/``
    and the older flat ``{data_dir}/checkpoints/*.pt``) are intentionally NOT
    found by a runtime fallback here (decision 9). Nothing writes there any more;
    such a file is reached by an explicit checkpoint path, which every reader
    accepts and checks by its bytes. Not being listed does not make a file
    invalid.

Module: src/utils/checkpoint_paths.py
"""
from __future__ import annotations

import copy
import math
import os
import re
import secrets
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Tuple, Union

ScoreFn = Callable[[Path], Optional[float]]
#: Whether a candidate file may be selected (automatic selection's record rule).
AcceptFn = Callable[[Path], bool]

from src.config.model_types import (
    ENCODER_KINDS,
    MODEL_RECORD_READ_KEYS,
)
from src.utils.fingerprint import ReadOutside, lies_within, stat_identity

PathLike = Union[str, Path]

# Ranking metrics used to pick the best checkpoint, in priority order, ALL
# "higher is better". The best checkpoint is decided from a checkpoint's own
# ``logs`` metadata — NOT from the filename. The number in ``model-<epoch>-
# <value>.pt`` has no stable meaning (it is whatever ModelCheckpoint's
# monitor/filename is configured to), so it must never be used to choose a
# model. A loss is not a selection figure here; ranking metrics are what matter.
# A model bundle run monitors its stage's metric: a base val_lp_accuracy, a
# finetune val_pooled_mrr (src.training.lp_task.BASE_SELECTION_METRIC and
# finetune_task.FINETUNE_SELECTION_METRIC, which this layer cannot import; a
# test pins them equal). One run has one stage and task, so the first one
# available is that run's own.
RANKING_SCORE_KEYS = ("val_pooled_mrr", "val_lp_accuracy")


def ranking_score_detail(logs) -> Optional[Tuple[str, float]]:
    """Return ``(metric_name, value)`` from a checkpoint's logs, or None.

    Picks the first available ranking metric (higher is better); ignores
    missing, non-numeric, and non-finite values (an integer past a float's
    range among them, which `math.isfinite` would raise on).
    """
    if not isinstance(logs, dict):
        return None
    for key in RANKING_SCORE_KEYS:
        value = logs.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        try:
            score = float(value)
        except OverflowError:  # an integer too large for a float: no finite figure
            continue
        if math.isfinite(score):
            return key, score
    return None


def _accepted(accept: Optional[AcceptFn], path: Path) -> bool:
    if accept is None:
        return True
    try:
        return bool(accept(path))
    except Exception:  # noqa: BLE001 - a file that cannot be judged is not chosen
        return False


def select_checkpoint_in_dir(
    ckpt_dir: PathLike, score_fn: Optional[ScoreFn] = None,
    accept: Optional[AcceptFn] = None,
) -> Optional[Path]:
    """Pick the checkpoint to serve from one directory.

    With ``accept``, only the files it takes are candidates, the fallback
    included (automatic selection's record rule: a checkpoint's learning stage
    and task, provenance §9.3 hazard 3).

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
    return _select_among([p for p in d.glob("*.pt") if _accepted(accept, p)], d, score_fn)


def _select_among(pts: List[Path], d: Path, score_fn: Optional[ScoreFn]) -> Optional[Path]:
    """The in-run ordering `select_checkpoint_in_dir` states, over `pts`, the
    candidates of the checkpoint directory `d`."""
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
                # `_mtime`: a file removed since its read (training prunes
                # its top-k files) sorts last, never raising out of here.
                scored.append((float(score), _mtime(p), p.name, p))
        if scored:
            best = max(scored, key=lambda t: (t[0], t[1], t[2]))
            return best[3]

    last = d / "last.pt"
    if last in pts:
        return last
    return max(pts, key=lambda p: (_mtime(p), p.name))


# =============================================================================
# Run directories (contract M3a-2)
# =============================================================================
#: Every training run, new or resumed, writes into a directory of its own:
#: ``{data_dir}/runs/{layout key}/{date}_{name}_{run_id}/``, the layout key a
#: new run's model spec's encoder kind (``gatv2``), holding the run's
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


#: The keys new runs are filed under: the model bundle's encoder kinds (README
#: §3.3), each its own directory. The writer's closed set only: a reader lists
#: every directory under ``runs/`` (`list_runs`), whatever its name.
RUN_LAYOUT_KEYS = ENCODER_KINDS


def normalize_run_layout_key(key: Optional[str]) -> str:
    """A new run's layout key, lower case: an encoder kind (``gatv2`` among
    them); anything else raises, an empty key too: a run is filed under its
    model spec's encoder kind, never under a default. The writer's check only
    (`runs_root`, `propose_run_dir`); no reader asks it."""
    k = str(key or "").strip().lower()
    if not k:
        raise ValueError("A run layout key is required: a run is filed under its model "
                         "spec's encoder kind, and none was given.")
    if k not in RUN_LAYOUT_KEYS:
        raise ValueError(f"Unsupported run layout key {key!r}; supported: {RUN_LAYOUT_KEYS}")
    return k


def runs_root(data_dir: PathLike, conv_type: Optional[str]) -> Path:
    """``{data_dir}/runs/{layout key}``: where a new run of a model spec's
    encoder kind is filed."""
    return Path(data_dir) / RUNS_DIRNAME / normalize_run_layout_key(conv_type)


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
            "into a directory of its own, {data_dir}/runs/{encoder kind}/..., with its "
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
    ``{data_dir}/runs/{conv_type}/{YYYYMMDD-HHMMSS}_{name or 'run'}_{6 hex}``,
    `conv_type` the run's layout key (its model spec's encoder kind), required:
    an empty one raises (`normalize_run_layout_key`).

    A name already on disk is drawn again. Proposing claims nothing:
    `claim_run_dir` does, so a launcher (the training service) can name the
    directory before the run that writes it has passed its refusals. A layout
    whose directory resolves outside the workspace is refused here already
    (`run_dir_outside`), so a launcher refuses before it starts anything.

    Raises:
        ValueError: the proposed directory resolves outside the workspace.
    """
    root = runs_root(data_dir, conv_type)
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    label = slugify_run_name(name) or "run"
    for _ in range(100):
        candidate = root / f"{stamp}_{label}_{secrets.token_hex(_RUN_ID_BYTES)}"
        if not candidate.exists():
            _refuse_outside(candidate, data_dir)
            return candidate
    raise RunDirectoryTaken(f"no unused run directory under {root} after 100 tries")


def run_dir_outside(run_dir: PathLike, data_dir: PathLike) -> Optional[str]:
    """Why the run directory `run_dir` may not be written for the workspace
    `data_dir`, or `None`.

    **Writing keeps the boundary reading keeps** (`checkpoint_not_read`): a
    run directory placed in the workspace (every one the layout names,
    ``{data_dir}/runs/...``, and an explicit `run_dir` given under the
    workspace) must resolve inside it. Behind a link out of the workspace
    (``runs/``, a layout directory, or the run directory itself) the run
    would land where the listing and automatic selection never read, so a
    finetune would write a run the reload never chooses (the reviewer's K8-1
    recheck). A run directory given outside the workspace is an explicit
    location the layout does not hold, as before: nothing selects from it
    automatically, and its checkpoints are served by an explicit
    checkpoint_path."""
    path, base = Path(os.path.abspath(run_dir)), Path(os.path.abspath(data_dir))
    try:
        within = base.resolve()
    except (OSError, RuntimeError) as exc:  # a link loop: RuntimeError before 3.13
        return (None if base not in path.parents else
                f"the workspace {base} cannot be resolved ({type(exc).__name__}), so no run "
                "is written in it")
    if base not in path.parents and within not in path.parents:
        return None
    outside = _outside(path, within)
    if outside is None:
        return None
    if not outside.startswith("resolves outside"):
        return (f"the run directory {path} {outside}, so it is not written: a link on its "
                "path loops or is broken; fix that link, or give another run_dir.")
    return (f"the run directory {path} {outside}, so it is not written: the listing and "
            "automatic selection read only runs inside the workspace and would never choose "
            "it. Keep the workspace's runs/ and its layout directories inside the workspace "
            "(not links out of it); to keep a run elsewhere, give an explicit run_dir outside "
            "the workspace and serve its checkpoint by an explicit checkpoint_path.")


def _refuse_outside(run_dir: PathLike, data_dir: PathLike) -> None:
    problem = run_dir_outside(run_dir, data_dir)
    if problem is not None:
        raise ValueError(problem)


def check_run_dir_free(run_dir: PathLike, *, data_dir: PathLike) -> Path:
    """Refuse a run directory that is not free: an existing file, or a
    directory holding anything. Absent or empty is free. Creates nothing, so an
    explicitly given location is refused before any run artifact exists;
    `claim_run_dir` decides again when it creates.

    A run directory placed in the workspace `data_dir` that resolves outside
    it is refused first (`run_dir_outside`): the one boundary of the run
    layout holds for writing as for reading.

    Raises:
        ValueError: the directory resolves outside the workspace.
        RunDirectoryTaken: the directory is not free.
    """
    _refuse_outside(run_dir, data_dir)
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


def claim_run_dir(run_dir: PathLike, *, data_dir: PathLike) -> Path:
    """Create the run directory and its ``checkpoints/`` and ``logs/``, or refuse.

    **Claimed by creating ``checkpoints/``** (`mkdir` without `exist_ok`): of two
    runs handed the same free directory, one creates it and the other is refused,
    so neither writes into the other's files. The caller claims only after every
    refusal, since this is the run's first artifact. A directory placed in the
    workspace `data_dir` that resolves outside it is refused before anything is
    created (`check_run_dir_free`, `run_dir_outside`).
    """
    path = check_run_dir_free(run_dir, data_dir=data_dir)
    path.mkdir(parents=True, exist_ok=True)
    try:
        run_checkpoint_dir(path).mkdir()
    except FileExistsError:
        raise RunDirectoryTaken(f"the run directory {path} was claimed by another run") from None
    run_log_dir(path).mkdir(exist_ok=True)
    return path


def run_checkpoint_dir(run_dir: PathLike) -> Path:
    """A run's ``checkpoints/``."""
    return Path(run_dir) / RUN_CHECKPOINTS_DIRNAME


def run_log_dir(run_dir: PathLike) -> Path:
    """A run's ``logs/``."""
    return Path(run_dir) / RUN_LOGS_DIRNAME


def _run_files(run_dir: PathLike) -> Tuple[List[Path], List[Path]]:
    """A run's checkpoint files, by name, in two lists: those whose name is
    text, and those whose name is not (`text_problem`), which no answer can
    carry and no request can name."""
    directory = run_checkpoint_dir(run_dir)
    files = sorted(directory.glob("*.pt")) if directory.is_dir() else []
    return ([p for p in files if text_problem(p.name) is None],
            [p for p in files if text_problem(p.name) is not None])


def run_checkpoints(run_dir: PathLike) -> List[Path]:
    """A run's checkpoint files, by name: those whose name is text, the only
    ones a list shows by name, a request names or a selection chooses. One
    whose name is not text is the listing's entry with its reason
    (`list_run_checkpoints`' `unlisted`), never read."""
    return _run_files(run_dir)[0]


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


#: Why a listing cannot list a directory under ``runs/``, a run directory or a
#: checkpoint file whose name is not valid text (`unlisted_problem` puts it in
#: its sentence).
_NAME_NOT_TEXT = ("its name is not valid text, shown here with backslash escapes; "
                  "it cannot be opened by name")


def _name_bytes(name: str) -> bytes:
    """The bytes the file system holds for a listed name (`os.fsencode`)."""
    try:
        return os.fsencode(name)
    except UnicodeEncodeError:  # a surrogate the system's decoding never gives
        return name.encode("utf-8", "backslashreplace")


def _shown_name(name: str) -> str:
    """A name the file system listed, as text every answer can carry: the
    name itself when it is text (`text_problem`), otherwise its bytes with
    each one that is not UTF-8 written as a backslash escape (``bad\\xff``).
    A display only: a name that is text may read the same (BACKLOG 66), so
    an entry is identified by `listing_key`, never by this."""
    if text_problem(name) is None:
        return name
    return _name_bytes(name).decode("utf-8", "backslashreplace")


def name_key(name: str) -> str:
    """A listed name's identity, apart from how it is shown: the name itself
    when it is text; otherwise ``/`` and its bytes in lowercase hexadecimal.
    No name the file system lists holds ``/``, so no text name has this form,
    and two different names never share one (BACKLOG 66)."""
    if text_problem(name) is None:
        return name
    return "/" + _name_bytes(name).hex()


def listing_key(names: Tuple[str, ...]) -> str:
    """**The one identity of a listed entry** (BACKLOG 66): its own names
    under ``runs/``, each by `name_key`, joined by ``/``: a checkpoint file's
    three (the directory under ``runs/``, the run, the file), a run
    directory's two, a directory directly under ``runs/`` its one
    (`ListedCheckpoint.key`, `ListedCheckpoint.run_key`). For names that are
    text a file's is ``architecture/run/file``, as the lists always keyed an
    entry. Two different name tuples never share a key, whatever their
    lengths: a name that is text is never empty and holds no ``/``, and one
    that is not text is keyed ``/`` and its bytes, so the empty part before
    its hexadecimal marks it. Selecting one entry never picks another."""
    return "/".join(name_key(name) for name in names)


def listed_names(run_dir: Path, path: Path) -> Tuple[str, str, str]:
    """An entry's names as every answer shows them (`_shown_name`): the
    directory its run sits in under ``runs/`` (`run_architecture`), the run's
    and the file's. An entry for a directory the listing could not list has
    that directory as its run and its file."""
    return (_shown_name(Path(run_dir).parent.name), _shown_name(Path(run_dir).name),
            _shown_name(Path(path).name))


def shown_path(path: PathLike) -> str:
    """A listed path as text every answer can carry: as it is when it is
    text, otherwise with backslash escapes (`_shown_name`'s rule, over the
    whole path)."""
    return _shown_name(str(path))


def list_runs(data_dir: PathLike, conv_type: Optional[str] = None, *,
              unlisted: Optional[List[Tuple[Path, Path, str]]] = None) -> List[Path]:
    """A workspace's run directories, most recently trained first.

    **Every directory under ``runs/``, found by listing it**, whatever its name
    (an encoder kind this build files new runs under, a name an earlier build
    used, one an operator gave): what a file is comes from its own record,
    never from where it sits. `conv_type` keeps the runs of the one directory
    whose name equals it; a name is compared with the listed names, never
    joined to a path, so ``.``, ``..``, ``a/b`` or ``""`` names none.

    Ordered by the newest checkpoint each holds, then by the run's name, then
    by its directory's, latest first (the name starts with the run's start
    time), so the order never depends on the filesystem's listing order.

    **Only ``runs/`` itself failing to list fails the list** (its `OSError`
    propagates). A directory under it that cannot be listed hides nothing
    else: its runs are not read, and it is appended to `unlisted`, when the
    caller gives one, as ``(directory, directory, problem)``: its path, and
    why in the listing's sentence (`unlisted_problem`: the error's kind and
    the system's words, never a path), by its shown name, so their order never
    depends on the filesystem's either. **A directory under it, or a run
    directory, whose name is not valid text** (a byte that is not UTF-8, which
    Python reads as a lone surrogate) is one this listing cannot list: no
    answer can carry its name and no request can give it, so it is appended to
    `unlisted` the same way, shown with backslash escapes (`_shown_name`,
    ``bad\\xff``) and identified by its bytes (`listing_key`), and nothing in
    it is read (BACKLOG 66).
    """
    base = Path(data_dir) / RUNS_DIRNAME
    runs: List[Path] = []
    if not base.is_dir():
        return runs
    refused: List[Tuple[Path, Path, str]] = []
    for directory in base.iterdir():
        if conv_type is not None and directory.name != conv_type:
            continue
        try:
            if not directory.is_dir():
                continue
            if text_problem(directory.name) is not None:
                refused.append((directory, directory, unlisted_problem(
                    directory.relative_to(base), _NAME_NOT_TEXT)))
                continue
            found = [p for p in directory.iterdir() if p.is_dir()]
        except OSError as exc:  # a directory this account may not enter
            # Its kind and the system's words, never the path it names.
            refused.append((directory, directory, unlisted_problem(
                directory.relative_to(base), f"{type(exc).__name__}: {exc.strerror}"
                if exc.strerror else refusal_line(exc))))
            continue
        for run in found:
            if text_problem(run.name) is None:
                runs.append(run)
            else:
                refused.append((run, run, unlisted_problem(run.relative_to(base),
                                                           _NAME_NOT_TEXT)))
    if unlisted is not None:
        unlisted.extend(sorted(refused, key=lambda entry: listed_names(entry[0], entry[1])))
    return sorted(runs, key=lambda r: (_run_recency(r), r.name, r.parent.name), reverse=True)


def unlisted_problem(under: Path, why: str) -> str:
    """What every reader of the runs says of an entry it cannot list
    (`list_runs`' and `list_run_checkpoints`' `unlisted`): the listing's entry
    and automatic selection's reason, in one sentence, naming it by `under`,
    its path relative to ``runs/``, as shown (`_shown_name`). A directory
    directly under ``runs/`` holds runs, a run directory its files; anything
    deeper is a checkpoint file."""
    parts = Path(under).parts
    shown = "/".join(_shown_name(part) for part in (RUNS_DIRNAME, *parts))
    if len(parts) == 1:
        return f"{shown} cannot be listed ({why}), so no run in it is read"
    if len(parts) == 2:
        return f"{shown} cannot be listed ({why}), so no file in it is read"
    return f"{shown} is not listed ({why}), so it is not read"


def run_architecture(run_dir: PathLike) -> str:
    """The layout directory a run directory sits in (its parent's name): a
    location, never the model's architecture, which its record names."""
    return Path(run_dir).parent.name


def list_run_checkpoints(
    data_dir: PathLike, conv_type: Optional[str] = None, *, run: Optional[str] = None,
    unlisted: Optional[List[Tuple[Path, Path, str]]] = None,
) -> List[Tuple[Path, Path]]:
    """Every checkpoint in a workspace's runs, as ``(run_dir, checkpoint)``:
    runs most recently trained first (`list_runs`, `conv_type` and `unlisted`
    as there), and within a run the newest file first, then by name. The one
    listing the console, the training service and the API show; what a file
    is comes from its own record, read by the caller, never from where it
    sits. `run`, a run directory's name, lists that run alone (a parent looked
    up by the run its child records); a directory under ``runs/`` that could
    not be listed may hold it, so it stays in `unlisted`, and a run directory
    whose name is not text is not it.

    **A checkpoint file whose name is not valid text** (BACKLOG 66) is not
    listed by name: it is appended to `unlisted` as ``(run_dir, file,
    problem)`` (`unlisted_problem`, shown with backslash escapes) and never
    read, chosen or resolved; its run's other files are listed."""
    base = Path(data_dir) / RUNS_DIRNAME
    refused: List[Tuple[Path, Path, str]] = []
    entries: List[Tuple[Path, Path]] = []
    for run_dir in list_runs(data_dir, conv_type, unlisted=refused):
        if run is not None and run_dir.name != run:
            continue
        named, unnamed = _run_files(run_dir)
        files = sorted(named, key=lambda p: (_mtime(p), p.name), reverse=True)
        entries.extend((run_dir, path) for path in files)
        refused.extend((run_dir, path, unlisted_problem(path.relative_to(base), _NAME_NOT_TEXT))
                       for path in sorted(unnamed, key=lambda p: _shown_name(p.name)))
    if unlisted is not None:
        unlisted.extend(entry for entry in refused
                        if run is None or entry[0] != entry[1] or entry[0].parent == base)
    return entries


class ListedCheckpoint(NamedTuple):
    """One checkpoint of the listing, as one read of its file found it.

    `sha256` is the digest of the bytes that read returned: the model
    fingerprint, which a ledger record and a served identity name. The rest is
    the file's own record, parsed from those same bytes: its kind, its epoch,
    its `logs` (`{}` when it records none), the Trainer's validation figure
    (`ranking_score_detail` of those logs: the selection metric of the run's
    stage, never a test result) and the dataset fingerprint it was trained on
    (`training_input_digests["split_manifest"]`, the SHA-256 of that
    workspace's `split_manifest.json`).

    **A file the read refuses is listed, never dropped:** `problem` says why
    (a format `weights_only=True` refuses, a file gone since it was listed),
    and the digest and record are `None`. A file that loads as something other
    than a checkpoint record keeps its digest and says so in `problem`, as
    does one whose metadata a list shows is not of the kind its field says
    (`_unshown`: the kind a text, the epochs whole numbers, the logs text
    names of figures, each string text, `text_problem`), with no metadata. A file
    that resolves outside the directory the caller bounds the listing by, or
    whose open lands outside it, is listed with `problem` and is not read at
    all (`read_run_checkpoints`' `within`). A directory under ``runs/`` that
    cannot be listed, or a run directory whose name is not text, is one entry,
    its `run_dir` and `path` that directory, with `problem` only; a checkpoint
    file whose name is not text is one entry, its `run_dir` its run, with
    `problem` only (`list_runs`, `list_run_checkpoints`: BACKLOG 66).

    `names` and `key` are its shown names and its identity (BACKLOG 66): a
    name that is not text is shown with backslash escapes and identified by
    its bytes, so two entries never share a key; `run_key` is its run's
    identity, by which a list groups its entries. `under` is set for an
    entry that is a directory (`run_dir` and `path` that directory): its own
    names under ``runs/``, one for a directory directly under it, two for a
    run directory, which its identity is made of (`listing_key`), so it never
    shares one with a run or a file whose names read alike; `None` for a
    file, identified by its run's names and its own.

    `size_bytes` is the file's size as listed (a `stat` beside the read,
    descriptive only); `None` for a file not read because it is outside, or one
    that cannot be found.

    `completed_epochs` and `end` are the file's progress record, as a training
    state writes it (`progress.completed_epochs`, `progress.end`): how many
    epochs it completed and how its stage ended (`None` while it runs, or
    "paused", "finished", "stopped_early"). A file that records no progress (a
    weights-only file, an older format) has `None` for both; nothing is derived
    from `epoch` in their place.

    `model_record` is a model bundle checkpoint's record as the same read found
    it, `None` for a file without a model spec: `record_version` and every
    other key the one reader of records reads (`MODEL_RECORD_READ_KEYS`: the
    §9.1 keys and the `training_input_digests` it compares them with), without
    the weights. It is not checked here: a caller reads it through
    `src.models.bundle.record`, the one reader of records, which this layer
    cannot import, and judges the file as a load of it does.

    `training` is the same read's training record, as the file holds it
    (research comparison Stage 8): ``rules``, the training rules it was
    written under (`training_rules`), and its configuration's ``requested``
    and ``effective`` settings (`config.requested`, `config.effective`), each
    `None` where the file records none (a weights-only file has no
    configuration). `None` for a file whose metadata was not read. A caller
    reads it through `run_plan.recorded_training_settings`.
    """

    run_dir: Path
    path: Path
    sha256: Optional[str] = None
    checkpoint_kind: Optional[str] = None
    epoch: Optional[int] = None
    logs: Optional[Dict[str, Any]] = None
    validation: Optional[Tuple[str, float]] = None
    dataset_fingerprint: Optional[str] = None
    problem: Optional[str] = None
    size_bytes: Optional[int] = None
    completed_epochs: Optional[int] = None
    end: Optional[str] = None
    model_record: Optional[Dict[str, Any]] = None
    under: Optional[Tuple[str, ...]] = None
    training: Optional[Dict[str, Any]] = None

    @property
    def names(self) -> Tuple[str, str, str]:
        """The entry's names as every answer shows them, text whatever the
        file system holds (`listed_names`): the directory its run sits in
        under ``runs/``, the run's and the file's."""
        return listed_names(self.run_dir, self.path)

    @property
    def key(self) -> str:
        """The entry's identity in its workspace's listing, apart from its
        shown names (`listing_key`, BACKLOG 66): ``architecture/run/file``
        for a file whose names are text; a directory's own names under
        ``runs/`` (`under`) for a directory."""
        if self.under is not None:
            return listing_key(self.under)
        return listing_key((Path(self.run_dir).parent.name, Path(self.run_dir).name,
                            Path(self.path).name))

    @property
    def run_key(self) -> str:
        """The identity of the run the entry belongs to, by which a list
        groups its entries (`listing_key` of the run directory's names:
        ``architecture/run`` for names that are text); a directory directly
        under ``runs/`` is its own group."""
        if self.under is not None:
            return listing_key(self.under[:2])
        return listing_key((Path(self.run_dir).parent.name, Path(self.run_dir).name))


#: How much of a problem's text a reason keeps (`problem_line`): its first line,
#: bounded, so a multi-paragraph loader message stays one readable reason.
_PROBLEM_LIMIT = 300


def problem_line(text: Any) -> str:
    """A problem's text as one reason line: its first non-empty line, each
    character that is not printable written as its escape (``\\udc80``,
    ``\\x00``), then bounded to `_PROBLEM_LIMIT` characters. The text can
    carry what a file itself holds (torch's legacy loader names the protocol
    version the file holds; a ledger record's field names), and a lone
    surrogate there made every list that showed the reason answer HTTP 500;
    escaped, it is text any answer carries. "" for a text with no line."""
    line = next((part.strip() for part in str(text).splitlines() if part.strip()), "")
    if not line.isprintable():
        line = "".join(c if c.isprintable() else repr(c)[1:-1] for c in line)
    if len(line) > _PROBLEM_LIMIT:
        line = line[:_PROBLEM_LIMIT - 3] + "..."
    return line


def refusal_line(exc: BaseException) -> str:
    """A refused read of a checkpoint as a reason shows it, the listing's
    (`_read_one`) and the training status's (`TrainingManager.latest_snapshot`):
    the exception's kind and the first line of its message (`problem_line`:
    escaped and bounded)."""
    text = problem_line(str(exc))
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def text_problem(value: Any) -> Optional[str]:
    """**The one check of a string read from a checkpoint before an answer
    shows it**, the record reader's (`src.models.bundle.record`), the
    listing's (`_read_one`) and the training status's: why `value` is not
    text, as the words a reason goes on from, or None for a string every
    character of which encodes as UTF-8. The safe loader reads any string a
    pickle holds, a lone surrogate ("\\udc80") among them, which no answer
    can encode: one file holding one in a field a list shows made that whole
    list answer HTTP 500. The caller names the field; the value itself is
    never echoed."""
    if not isinstance(value, str):
        return f"is a {type(value).__name__}, not text"
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        return f"is not text: character {exc.start} is a lone surrogate"
    return None


#: How deep a value read from a checkpoint may nest, and how many values it may
#: hold in all, for `plain_data` to walk it: the deepest part a model record
#: compares, a graph view rule, is a mapping holding a list of names. Past
#: either, the value is refused naming where the walk stopped, so the walk is
#: bounded however the file was written (the safe loader reads any depth and
#: shares values freely: one list held many times over at each level is a few
#: objects in the file and more values than any walk can read). **The safe
#: loader's bounds only**: the checkpoint readers pass them (the model record's,
#: `src.models.bundle.record`; the listing's training settings,
#: `run_plan.recorded_training_settings`); a value parsed from JSON is never
#: held to them (review fix RC-R1).
PLAIN_DEPTH = 16
PLAIN_VALUES = 10_000

#: How deep a measuring run's report may nest, counting each object and list as
#: a level. The measurement writes 7 or 8; anything near this is a hand edit.
#: Counted, not recursed, so a refusal never depends on the interpreter's
#: recursion limit or on how deep the caller's stack already is. Stated once
#: here, the lowest layer that needs it: the registration refuses a report
#: deeper than it (`src.evaluation.sidecar`), a record view holds a record's
#: components to it (`src.api.services.record_views`), and the collector holds
#: a component's facts to what such a report can carry
#: (`run_environment.COMPONENT_DEPTH`).
MAX_REPORT_DEPTH = 64


class NotPlainData(ValueError):
    """Why a value is not plain data an answer can carry (`plain_data`):
    `steps`, the path from the value's root to the part refused (a mapping's
    keys and a list's indices), `where`, that path as words (`name`, then
    ``.key`` and ``[index]``), and `why`, the words a reason goes on from.
    Never the value itself."""

    def __init__(self, name: str, steps: Tuple[Union[str, int], ...], why: str) -> None:
        self.name, self.steps, self.why = name, tuple(steps), why
        where = self.where
        super().__init__(f"{where} {why}" if where else why)

    @property
    def where(self) -> str:
        text = self.name
        for step in self.steps:
            text += f"[{step}]" if isinstance(step, int) else (f".{step}" if text else step)
        return text


def plain_data(value: Any, name: str = "", *, depth: Optional[int] = None,
               values: Optional[int] = None, finite: bool = True) -> Any:
    """**The one rule for a value read from a file that an answer must carry**
    (review fix RC-R1): `value` (named `name`) as plain data that states its
    kind beside its content, or a `NotPlainData` naming the path to the first
    part that is not, and why, never the value.

    Plain data is what every answer can carry and compare: null, a boolean, a
    finite number, a string that is text (`text_problem`: a lone surrogate is
    not, and no answer can encode one; valid Unicode of every script is), a
    list (or tuple), and a mapping keyed by such strings. Anything else, a
    number that is not finite among it (no JSON answer holds one), is refused.
    Two values are the same exactly when their returned forms are: by kind as
    well as content, so a boolean is never a number (Python's ``True == 1``),
    and a value with no one answer to ``==`` (a tensor the safe loader also
    accepts) is refused, never compared.

    **The walk ends for every value it is given, at any depth** (the
    reviewer's K8-4r2 recheck: a list holding itself made a recursive walk run
    until `RecursionError`). It keeps its own path, never the interpreter's
    stack, so a value nested deeper than the interpreter can recurse (JSON
    parses one thousands of levels deep) is read to its end or refused, never
    a `RecursionError`. A scalar is judged where it stands; a list or mapping
    already on the path walked (a value holding itself) is refused, for every
    value. **Bounds only where the safe loader reads**: `depth` (how many lists
    and mappings may lie above a part) and `values` (how many values the walk
    may read in all, one held many times over counting each time) refuse what
    lies past them, naming where the walk stopped. A checkpoint's readers pass
    the loader's (`PLAIN_DEPTH`, `PLAIN_VALUES`: the safe loader reads any
    depth and shares values freely, so a few objects in a file can be more
    values than any walk can read): the model record's
    (`src.models.bundle.record`) and the listing's training settings
    (`run_plan.recorded_training_settings`). A value parsed from JSON (a
    report, a ledger record, a manifest) is read with none, the default
    (review fix RC-R1, the reviewer's case: a record holding 10,001 legitimate
    build facts): it cannot hold itself or share a part, so the walk is as
    long as the file; where what a reader shows of it must be bounded, its
    format's own bound does it (a record's components by a report's
    `MAX_REPORT_DEPTH`, `src.api.services.record_views`; a collected
    component by the depth its report can hold, `run_environment`).
    ``finite=False`` only where the reader itself gives each number that is
    not finite a form every answer carries before any answer gets it (a
    checkpoint's training settings, which may record one, are shown with it as
    its text: `run_plan.recorded_training_settings`); every other part is
    judged as above.

    Every reader that shows what a file holds reads it by this rule: a model
    record's compared and shown parts (`src.models.bundle.record`), a
    component's facts when collected and when read
    (`src.utils.run_environment.component_facts`, the record views), each
    record view and measurement setting (`src.api.services.record_views`), a
    checkpoint's training settings (`run_plan.recorded_training_settings`) and
    each row of the results service (`src.api.services.results`)."""
    root, opened_now = object(), object()
    walked = 0
    # The lists and mappings open on the path walked, outermost first, each as
    # [itself, the step it was reached by, its parts not yet read, the forms of
    # those read], and their ids, which find a value holding itself.
    opened: List[List[Any]] = []
    on_path: set = set()

    def where(step: Any) -> Tuple[Union[str, int], ...]:
        steps = tuple(frame[1] for frame in opened[1:])
        return steps if step is root else steps + (step,)

    def read(part: Any, step: Any) -> Any:
        """`part`'s form when it is a scalar; `opened_now` once a list or a
        mapping is opened for its parts to be read; or its refusal."""
        nonlocal walked
        walked += 1
        if values is not None and walked > values:
            raise NotPlainData(name, (), f"holds more than {values} values")
        if part is None:
            return ("null",)
        if isinstance(part, bool):
            return ("boolean", part)
        if isinstance(part, (int, float)):
            if finite and isinstance(part, float) and not math.isfinite(part):
                raise NotPlainData(name, where(step), "is not a finite number")
            return ("number", part)
        if isinstance(part, str):
            why = text_problem(part)
            if why is not None:
                raise NotPlainData(name, where(step), why)
            return ("string", part)
        if not isinstance(part, (list, tuple, Mapping)):
            raise NotPlainData(name, where(step), f"is a {type(part).__name__}, not plain data")
        if id(part) in on_path:
            raise NotPlainData(name, where(step),
                               f"is a {type(part).__name__} that contains itself")
        if depth is not None and len(opened) >= depth:
            raise NotPlainData(name, where(step), f"nests deeper than {depth} levels")
        if isinstance(part, Mapping):
            for key in part:
                if not isinstance(key, str):
                    raise NotPlainData(name, where(step), f"has a key that is not a string "
                                                          f"({type(key).__name__})")
                why = text_problem(key)
                if why is not None:
                    raise NotPlainData(name, where(step), f"has a key that {why}")
            opened.append([part, step, iter(part.items()), {}])
        else:
            opened.append([part, step, iter(enumerate(part)), []])
        on_path.add(id(part))
        return opened_now

    def keep(forms: Any, step: Any, form: Any) -> None:
        if isinstance(forms, dict):
            forms[step] = form
        else:
            forms.append(form)

    form = read(value, root)
    while opened:
        part, step, parts, forms = opened[-1]
        for inner, child in parts:
            read_form = read(child, inner)
            if read_form is opened_now:
                break  # its parts are read first, then this one's others
            keep(forms, inner, read_form)
        else:
            opened.pop()
            on_path.discard(id(part))
            form = ("mapping", forms) if isinstance(forms, dict) else ("list", tuple(forms))
            if opened:
                keep(opened[-1][3], step, form)
    return form


def plain_data_problem(value: Any, name: str = "", **bounds: Any
                       ) -> Optional[NotPlainData]:
    """Why `value` is not plain data an answer can carry (`plain_data`, its
    `bounds` and ``finite`` as there), or None: for a reader that only asks."""
    try:
        plain_data(value, name, **bounds)
    except NotPlainData as exc:
        return exc
    return None


def whole_number_problem(value: Any) -> Optional[str]:
    """Why `value`, an epoch count read from a checkpoint, is not a whole
    number (a boolean is not one), as the words a reason goes on from, or
    None; the listing's check (`_unshown`) and the training status's."""
    if isinstance(value, int) and not isinstance(value, bool):
        return None
    return f"is a {type(value).__name__}, not a whole number"


def _figure_problem(value: Any) -> Optional[str]:
    """A logged figure, as the trainer writes `logs` (`{name: figure}`): a
    number (a non-finite one included), a boolean, null or text."""
    if value is None or isinstance(value, (bool, int, float)):
        return None
    if isinstance(value, str):
        return text_problem(value)
    return f"is a {type(value).__name__}, not a number"


def _unshown(meta: Dict[str, Any], logs: Dict[str, Any], progress: Dict[str, Any],
             fingerprint: Any) -> Optional[str]:
    """Why the metadata a listing shows of a read file cannot be shown as its
    fields' kinds say (`ListedCheckpoint`), naming the first field that is not
    (`text_problem`, a whole number, a logged figure), or None. A list puts
    these values in its answer as they are, so one that is not of its kind,
    or not text, made the whole list fail (a lone surrogate, an epoch of 1.5,
    a tensor among the logs); now that file alone is listed with the reason."""
    for name, value, problem in (
            ("checkpoint_kind", meta.get("checkpoint_kind"), text_problem),
            ("epoch", meta.get("epoch"), whole_number_problem),
            ("progress.completed_epochs", progress.get("completed_epochs"),
             whole_number_problem),
            ("progress.end", progress.get("end"), text_problem),
            ("training_input_digests.split_manifest", fingerprint, text_problem)):
        why = problem(value) if value is not None else None
        if why is not None:
            return f"its {name} {why}"
    for key, value in logs.items():
        why = text_problem(key)
        if why is not None:
            return f"its logs hold a name that {why}"
        why = _figure_problem(value)
        if why is not None:
            shown = repr(key) if len(key) <= 60 else repr(key[:57]) + "..."
            return f"its logs entry {shown} {why}"
    return None


def _outside(path: Path, within: Path) -> Optional[str]:
    """What keeps `path` from being inside `within` (resolved), or `None` when
    it resolves inside it; the words a reason goes on from."""
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError) as exc:  # a link loop: RuntimeError before 3.13
        return f"cannot be resolved ({type(exc).__name__})"
    if not lies_within(resolved, within):
        return "resolves outside the workspace"
    return None


def checkpoint_not_read(run_dir: Path, path: Path, within: Path) -> Optional[str]:
    """The reason a run's checkpoint `path` is not read under `within` (the
    workspace, resolved), as a listing shows it, or `None` for a file that
    resolves inside it.

    **The one workspace boundary of the run layout**: the listing
    (`read_run_checkpoints`' `within`), every filter over it, automatic
    selection (`select_auto_run_checkpoint`) and the server's resolution of a listed checkpoint (`server_lists.resolve_checkpoint`,
    `locate_checkpoint`) ask it of each file before anything reads the file's
    content; the writer keeps it too (`run_dir_outside`). The file's resolved
    path decides, so a run directory or a layout directory that is a link out
    of the workspace leaves every file under it outside.

    **A name is judged here, not the file a later open finds.** A name can
    point elsewhere at every lookup, so the reads that follow (the listing's,
    and automatic selection's scan and build) bind the same boundary
    (`lies_within`) to the file each one opens, by its descriptor
    (`read_once(within=)`): a file swapped out of the workspace after this
    check is refused before a byte of it is read."""
    outside = _outside(path, within)
    return None if outside is None else f"{run_dir.name}/{path.name} {outside}, so it is not read"


def _size(path: Path) -> Optional[int]:
    try:
        return path.stat().st_size
    except OSError:
        return None


#: Each listed file's last read in this process, by its path and the boundary
#: the read was bound to (`read_run_checkpoints`' `within`, resolved, or None
#: for an unbounded read): the file's identity (the one the read's open
#: descriptor gave, which the name gave before and after the read too), and
#: what the read found (`ListedCheckpoint` without its two paths). A
#: listing recalls only a read made under its own boundary, so what an
#: unbounded read (or one bound to another workspace) found of a file outside
#: is never shown by a listing bound to this one. Least recently listed first.
_READS: "OrderedDict[Tuple[str, Optional[str]], Tuple[Tuple[int, ...], Dict[str, Any]]]" = (
    OrderedDict())
_READS_LOCK = threading.Lock()
#: How many files' reads are kept (a few kilobytes each).
READS_KEPT = 4096
#: A file whose content or metadata changed less than this long before its
#: read is read again at every listing: a rewrite within one tick of the file
#: system's clock could leave its `stat` unchanged.
SETTLED_NS = 2_000_000_000
#: How deep the values a read found may nest for the read to be kept: a kept
#: read is copied in and out (`copy.deepcopy`, which recurses at least once per
#: level), while the safe loader reads any depth (its unpickler keeps its own
#: stack). A file holding a value nested deeper is read again at every listing
#: instead; a record nests a few levels.
KEPT_DEPTH = 32


def _identity(path: Path) -> Optional[Tuple[int, ...]]:
    """The file `path` names now, as `stat_identity` tells one state of a file
    from another without reading it, or None where it cannot be had."""
    try:
        return stat_identity(path.stat())
    except OSError:
        return None


def _nests_within(value: Any, limit: int) -> bool:
    """Whether `value` nests at most `limit` levels as `copy.deepcopy` walks
    it: into lists, tuples, sets, mappings (their keys too) and an object's
    attributes (a tensor's), each looked into once and in the copy's order, so
    a value held twice or holding itself costs one look. Walked without
    recursion, so any depth is measured."""
    seen = set()
    stack = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        if item is None or isinstance(item, (str, bytes, bytearray, int, float, complex)):
            continue
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, dict):
            held = [part for pair in item.items() for part in pair]
        elif isinstance(item, (list, tuple, set, frozenset)):
            held = list(item)
        else:
            attributes = getattr(item, "__dict__", None)
            held = list(attributes.values()) if isinstance(attributes, dict) else []
        if held and depth >= limit:
            return False
        stack.extend((part, depth + 1) for part in reversed(held))
    return True


def _kept_as(path: Path, within: Optional[Path]) -> Tuple[str, Optional[str]]:
    """The key of `path`'s read bound to `within` (`_READS`)."""
    return str(path), (str(within) if within is not None else None)


def _recalled(path: Path, identity: Optional[Tuple[int, ...]],
              within: Optional[Path]) -> Optional[Dict[str, Any]]:
    if identity is None:
        return None
    key = _kept_as(path, within)
    with _READS_LOCK:
        kept = _READS.get(key)
        if kept is None or kept[0] != identity:
            return None
        _READS.move_to_end(key)
        fields = kept[1]
    return {**fields, "logs": copy.deepcopy(fields["logs"]),
            "model_record": copy.deepcopy(fields["model_record"]),
            "training": copy.deepcopy(fields["training"])}


def _remember(path: Path, within: Optional[Path], before: Optional[Tuple[int, ...]],
              opened: Optional[Tuple[int, ...]], started_ns: int,
              entry: ListedCheckpoint) -> None:
    """Keep `entry` for `path`, read bound to `within`, only when the file the
    read opened (`opened`, its open descriptor's identity) is the file the name
    gave before the read (`before`) and still gives after it, and it had
    settled before the read (`SETTLED_NS`).

    So nothing is kept under a file's identity that a read of another file
    found (the reviewer's K8-4r2 recheck, P3: a name switched to another file
    for the open and back gave the same identity before and after, and that
    file's digest was then shown for the first). A refusal no read made (a
    file opened outside the workspace or not placed, a read that failed:
    `opened` is None) is never kept, so the next listing reads the file again
    rather than repeat a reason the name may no longer earn. Nor is a read
    whose values nest past `KEPT_DEPTH`, which could not be copied."""
    if before is None or opened != before or _identity(path) != before:
        return
    if max(before[3], before[4]) > started_ns - SETTLED_NS:
        return
    fields = entry._asdict()
    del fields["run_dir"], fields["path"]
    if not _nests_within((fields["logs"], fields["model_record"], fields["training"]),
                         KEPT_DEPTH):
        return
    fields["logs"] = copy.deepcopy(fields["logs"])
    fields["model_record"] = copy.deepcopy(fields["model_record"])
    fields["training"] = copy.deepcopy(fields["training"])
    key = _kept_as(path, within)
    with _READS_LOCK:
        _READS[key] = (before, fields)
        _READS.move_to_end(key)
        while len(_READS) > READS_KEPT:
            _READS.popitem(last=False)


def _opened_outside(exc: ReadOutside) -> str:
    """A bounded read's refusal, as the words a reason goes on from: the file
    its open found lies outside the workspace, or cannot be placed."""
    if exc.opened is not None:
        return "was opened outside the workspace"
    return f"cannot be placed inside the workspace ({exc.why})"


def _read_one(run: Path, path: Path, within: Optional[Path]
              ) -> Tuple[ListedCheckpoint, Optional[Tuple[int, ...]]]:
    """One file's entry, from one read of its bytes, and the identity of the
    file that read opened (`ReadIdentity.opened`; None where nothing was
    read); a read bounded to `within` (resolved) when the listing is, so a
    file its open finds outside is listed as not read, with the reason, before
    a byte of it is read. The entry holds the values this read loaded, which
    nothing else holds (`_remember` keeps copies)."""
    from src.utils.checkpoint_io import read_checkpoint

    size = _size(path)
    try:
        read = read_checkpoint(path, map_location="cpu", weights_only=True, within=within)
    except ReadOutside as exc:
        return ListedCheckpoint(
            run, path, problem=f"{run.name}/{path.name} {_opened_outside(exc)}, so it is not read"
        ), None
    except Exception as exc:  # noqa: BLE001 — listed with its reason, never dropped
        return ListedCheckpoint(run, path, problem=refusal_line(exc), size_bytes=size), None
    meta, opened = read.checkpoint, read.identity.opened
    if not isinstance(meta, dict):
        return ListedCheckpoint(
            run, path, sha256=read.identity.sha256,
            problem=(f"it loads as {type(meta).__name__}, not a checkpoint record, so it "
                     "carries no metadata"),
            size_bytes=size), opened
    logs = meta.get("logs") if isinstance(meta.get("logs"), dict) else {}
    digests = meta.get("training_input_digests")
    fingerprint = digests.get("split_manifest") if isinstance(digests, dict) else None
    progress = meta.get("progress") if isinstance(meta.get("progress"), dict) else {}
    unshown = _unshown(meta, logs, progress, fingerprint)
    if unshown is not None:
        return ListedCheckpoint(
            run, path, sha256=read.identity.sha256,
            problem=f"{unshown}, so none of its metadata is shown", size_bytes=size), opened
    record = ({"record_version": meta.get("record_version"),
               **{k: meta[k] for k in MODEL_RECORD_READ_KEYS if k in meta}}
              if meta.get("model_spec") is not None else None)
    config = meta.get("config") if isinstance(meta.get("config"), dict) else {}
    training = {"rules": meta.get("training_rules"), "requested": config.get("requested"),
                "effective": config.get("effective")}
    return ListedCheckpoint(
        run, path, sha256=read.identity.sha256,
        checkpoint_kind=meta.get("checkpoint_kind"),
        epoch=meta.get("epoch"),
        logs=logs,
        validation=ranking_score_detail(logs),
        dataset_fingerprint=fingerprint,
        size_bytes=size,
        completed_epochs=progress.get("completed_epochs"),
        end=progress.get("end"),
        model_record=record,
        training=training,
    ), opened


def read_run_checkpoints(
    data_dir: PathLike, conv_type: Optional[str] = None, *,
    within: Optional[PathLike] = None, run: Optional[str] = None,
) -> List[ListedCheckpoint]:
    """The listing (`list_run_checkpoints`, same order, `run` as there), each
    file read once, and again only when it changes.

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
    process by the file's path, the boundary the read was bound to (`within`)
    and the file's identity (`stat_identity`: device, inode, size, modification
    and change times), kept only when the read's open descriptor gave the
    identity the name gave before and after the read (`_remember`: a read of
    another file, or a refusal, is never kept), and a later listing under
    the same boundary that finds the same identity shows it without opening
    the file; any write, replacement or metadata change reads it again, as does a
    file that changed in the `SETTLED_NS` before its read. The bytes are still
    checked where they are used: a load, a start or a measurement re-reads them
    against the digest shown. The kept reads are shown only through the
    callers that read them, which D8 gates as before.

    `within`, the workspace directory, bounds what is read: a file that
    resolves outside it (a run or a checkpoint that is a link out of the
    workspace) is listed with that reason and **never opened**, so nothing of
    a file elsewhere is read or shown. The check runs before the read (and
    before a kept read is used), and **the read itself is bound to it** by the
    file it opens (`read_checkpoint(within=)`), as automatic selection's reads
    are: a name swapped for a link out of the workspace after the check is
    listed as not read, with the reason, and not a byte of the file it now
    names is read or shown. A kept read is recalled only by a listing bound as
    that read was, so what an unbounded listing read of an outside file is
    never shown by a bounded one. The server's list passes its workspace
    (`server_lists.list_checkpoints`); a caller that passes none reads every
    listed file unbounded, as before. It is the boundary automatic selection
    applies too (`checkpoint_not_read`), so a file the list refuses is never chosen.

    **A directory under ``runs/`` that cannot be listed is listed, after the
    files, as one entry** whose `run_dir` and `path` are that directory and
    whose `problem` says why (`list_runs`' `unlisted`, which names one whose
    name is not text by its name as shown): what it holds is not read, and
    nothing else is hidden. So is a run directory or a checkpoint file whose
    name is not text (BACKLOG 66), each shown with backslash escapes
    (`ListedCheckpoint.names`) and identified by its own names under
    ``runs/``, a name that is not text by its bytes (`ListedCheckpoint.key`,
    `under`). Only ``runs/`` itself failing to list raises.

    torch is imported only here, when a file is read, so this module stays
    importable without it.
    """
    bound = Path(within).resolve() if within is not None else None
    base = Path(data_dir) / RUNS_DIRNAME
    entries: List[ListedCheckpoint] = []
    unlisted: List[Tuple[Path, Path, str]] = []
    for run_dir, path in list_run_checkpoints(data_dir, conv_type, run=run, unlisted=unlisted):
        problem = checkpoint_not_read(run_dir, path, bound) if bound is not None else None
        if problem is not None:
            entries.append(ListedCheckpoint(run_dir, path, problem=problem))
            continue
        identity = _identity(path)
        kept = _recalled(path, identity, bound)
        if kept is not None:
            entries.append(ListedCheckpoint(run_dir, path, **kept))
            continue
        started_ns = time.time_ns()
        entry, opened = _read_one(run_dir, path, bound)
        _remember(path, bound, identity, opened, started_ns, entry)
        entries.append(entry)
    # A directory entry is identified by its own names under runs/ (one or
    # two), a file by its run's and its own (`ListedCheckpoint.key`).
    entries.extend(ListedCheckpoint(run_dir, path, problem=problem,
                                    under=path.relative_to(base).parts if path == run_dir
                                    else None)
                   for run_dir, path, problem in unlisted)
    return entries


def _first_run_holding(runs: List[Path], accept: Optional[AcceptFn], within: Path
                       ) -> Tuple[Optional[Path], List[Path], List[str]]:
    """The first of `runs` holding a candidate: a checkpoint inside the
    workspace `within` (`checkpoint_not_read`) that `accept` takes (any, without one).
    Returns that run, its candidates, and the listing's reason for each file
    of the runs judged that resolves outside. **The boundary comes before
    `accept`**, which may read every file it is asked about, so a file outside
    the workspace is never opened; the runs after the one found are never
    judged."""
    outside: List[str] = []
    for run in runs:
        inside = []
        named, unnamed = _run_files(run)
        for path in named:
            problem = checkpoint_not_read(run, path, within)
            if problem is None:
                inside.append(path)
            else:
                outside.append(problem)
        # A file whose name is not text is no candidate (BACKLOG 66): named
        # as the listing names it, never silently passed over.
        outside.extend(unlisted_problem(Path(run.parent.name, run.name, path.parent.name,
                                             path.name), _NAME_NOT_TEXT)
                       for path in sorted(unnamed, key=lambda p: _shown_name(p.name)))
        candidates = [path for path in inside if _accepted(accept, path)]
        if candidates:
            return run, candidates, outside
    return None, [], outside


#: How many of the files a selection did not read for the boundary its reason
#: names; the rest are counted.
NOT_READ_NAMED = 3


def with_not_read(reason: str, outside: List[str]) -> str:
    """A selection's reason, with the files it did not read for the boundary,
    in the listing's words: the first `NOT_READ_NAMED`, then how many more, so
    a runs directory linked out of the workspace gives a reason of bounded
    length. The reload adds the files its scan's bounded reads refused the
    same way."""
    if not outside:
        return reason
    more = len(outside) - NOT_READ_NAMED
    return (f"{reason}; not read: {'; '.join(outside[:NOT_READ_NAMED])}"
            + (f"; and {more} more file(s) not read" if more > 0 else ""))


def select_auto_run_checkpoint(
    data_dir: PathLike, score_fn: Optional[ScoreFn] = None,
    accept: Optional[AcceptFn] = None, not_read: Optional[List[str]] = None,
) -> Tuple[Optional[Path], Optional[str], str]:
    """Auto selection over runs: the most recently trained run, in any
    directory under ``runs/`` (`list_runs`), holding a candidate, then the
    existing in-run ordering. Returns
    ``(checkpoint_path, run layout directory, reason)``. Only the run layout is
    searched; checkpoints elsewhere are reached by an explicit path.

    **Inside the workspace only** (the reviewer's K8-1 recheck): a file that
    resolves outside `data_dir`, a run or layout directory linked out of it
    included, is no candidate and is never read, as the listing refuses it
    (`checkpoint_not_read`); the reason names it in the listing's words
    (the first `NOT_READ_NAMED`, then a count), and a list given as
    `not_read` is extended with every one. **A directory under ``runs/`` it
    cannot list is named the same way, first** (`unlisted_problem`, the
    listing's entry): the run chosen is then the latest of the runs read,
    never silently the latest of all.

    **With ``accept``, the record decides, never the directory** (provenance
    §9.3, hazard 3): a run counts only when it holds a file ``accept`` takes
    (the reload's rule: a model bundle finetuned for the task asked for)."""
    unlisted: List[Tuple[Path, Path, str]] = []
    runs = list_runs(data_dir, unlisted=unlisted)
    run, candidates, outside = _first_run_holding(runs, accept, Path(data_dir).resolve())
    outside = [problem for _run, _path, problem in unlisted] + outside
    if not_read is not None:
        not_read.extend(outside)
    if run is None:
        return None, None, with_not_read("no run has a checkpoint", outside)
    key = run_architecture(run)
    # The reason names the run by its location (`<layout key>/<run>`, the
    # lists' label), never as an architecture: what runs is the loaded spec's.
    return (_select_among(candidates, run_checkpoint_dir(run), score_fn), key,
            with_not_read(f"auto: latest run '{key}/{run.name}'", outside))


__all__ = [
    "RANKING_SCORE_KEYS",
    "RETIRED_LOCATION_FIELDS",
    "RUN_LAYOUT_KEYS",
    "RUNS_DIRNAME",
    "RUN_CHECKPOINTS_DIRNAME",
    "RUN_LOGS_DIRNAME",
    "ListedCheckpoint",
    "MAX_REPORT_DEPTH",
    "NOT_READ_NAMED",
    "NotPlainData",
    "PLAIN_DEPTH",
    "PLAIN_VALUES",
    "RunDirectoryTaken",
    "check_run_dir_free",
    "checkpoint_not_read",
    "claim_run_dir",
    "list_run_checkpoints",
    "list_runs",
    "listed_names",
    "listing_key",
    "name_key",
    "plain_data",
    "plain_data_problem",
    "propose_run_dir",
    "read_run_checkpoints",
    "problem_line",
    "refusal_line",
    "refuse_retired_locations",
    "run_architecture",
    "run_checkpoint_dir",
    "run_checkpoints",
    "run_dir_outside",
    "run_log_dir",
    "runs_root",
    "select_auto_run_checkpoint",
    "shown_path",
    "slugify_run_name",
    "ranking_score_detail",
    "text_problem",
    "unlisted_problem",
    "whole_number_problem",
    "with_not_read",
    "normalize_run_layout_key",
    "select_checkpoint_in_dir",
]
