"""
The code revision and the environment a run was produced under.
================================================================
Contract M3a (`docs/working/PLAN_PROVENANCE_CONTRACT.md`, M3a-1, §5.3, §8
question 1). Training records both in every checkpoint it writes, collected once
at the start of the run. They are **recorded, never enforced** (§5.2): an
environment difference never refuses, and matching fields are a precondition for
comparing two results, not a guarantee that they agree.

**What is not here, on purpose.** No file names, no paths, no hostnames, no user
names: a checkpoint travels, and these values describe the software and the
hardware, not the machine's owner or its layout. Counts stand in for lists.
Error text is not recorded either, because an import error can carry a path; a
failed probe records `None`, which reads as "not known", never as "absent".

Every value is a string, a number, a boolean, `None`, or a dict or list of those,
so a checkpoint carrying them loads with `torch.load(..., weights_only=True)`.

Today's other collectors (the measurement manifest's fields, the deployment
probe's commit check, `src/utils/version_checker.py`) stay as they are until
their entry points are next changed; this is the one later entry points use
(the plan's "Not prerequisites").

Module: src/utils/run_environment.py
"""
from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Untracked paths that are a run's own products, not code, so they do not make
#: a checkout dirty. **Untracked only:** a change to a tracked file counts
#: wherever it is, `data/` included. Paths are repository-relative, as git
#: reports them. Ignored files are never counted (git leaves them out).
GENERATED_UNTRACKED_PREFIXES = (
    "data/workspaces/",
    "data/cache/",
    "outputs/",
    "logs/",
    "reports/",
    "profiles/",
)

_PROBE_TIMEOUT_SECONDS = 15


def _git(root: Path, *args: str) -> Optional[bytes]:
    """Git's stdout as bytes, or `None` for any failure: no git, no repository,
    a repository git refuses (ownership), a timeout or a non-zero exit.

    **Bytes, never decoded here.** Paths in `git status -z` are raw bytes, which
    need not decode in the locale's encoding (a file name in another encoding,
    or a Windows code page): decoding them used to raise `UnicodeDecodeError`,
    which nothing caught, and stopped the training run whose revision was only
    being recorded.
    """
    try:
        done = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True, timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return done.stdout if done.returncode == 0 else None


_GENERATED_PREFIXES_BYTES = tuple(p.encode("ascii") for p in GENERATED_UNTRACKED_PREFIXES)


def _count_status(porcelain: bytes) -> Dict[str, int]:
    """Counts from `git status --porcelain=v1 -z`: tracked files with any change,
    staged or not, and untracked files outside `GENERATED_UNTRACKED_PREFIXES`.
    Works on the raw bytes: a path is compared by prefix, never decoded."""
    modified = untracked = 0
    entries = porcelain.split(b"\0")
    i = 0
    while i < len(entries):
        entry = entries[i]
        i += 1
        if len(entry) < 4:
            continue
        code, path = entry[:2], entry[3:]
        if code == b"??":
            if not path.startswith(_GENERATED_PREFIXES_BYTES):
                untracked += 1
            continue
        modified += 1
        # A rename or copy is followed by its source path, which is not a
        # second change.
        if b"R" in code or b"C" in code:
            i += 1
    return {"modified_tracked": modified, "untracked": untracked}


def _commit(raw: Optional[bytes]) -> Optional[str]:
    """A commit id from `git rev-parse HEAD`'s output, or `None`. A commit id
    is hexadecimal; anything else is not taken for one."""
    if not raw:
        return None
    text = raw.strip().decode("ascii", errors="replace")
    return text if text and all(c in "0123456789abcdef" for c in text) else None


def code_revision(root: Path) -> Dict[str, Any]:
    """The revision `root`'s checkout is at, and whether it is dirty.

    - **`state`** is `recorded` when git named a commit, `unrecorded` otherwise:
      no `.git`, no git, or a git that refuses the repository. An unrecorded
      revision is never guessed (§8 question 1), and its dirty state is `None`,
      never "clean".
    - **`dirty`** is true when any tracked file differs from the commit, staged or
      not, or an untracked file exists outside the run's own products; `None`
      when git named the commit but could not report its status.
    """
    head = _commit(_git(root, "rev-parse", "HEAD"))
    if head is None:
        return {"state": "unrecorded", "revision": None, "dirty": None,
                "modified_tracked": None, "untracked": None}
    status = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if status is None:
        return {"state": "recorded", "revision": head, "dirty": None,
                "modified_tracked": None, "untracked": None}
    counts = _count_status(status)
    return {"state": "recorded", "revision": head,
            "dirty": bool(counts["modified_tracked"] or counts["untracked"]),
            **counts}


def _driver_version(index: int) -> Optional[str]:
    try:
        done = subprocess.run(
            ["nvidia-smi", f"--id={index}", "--query-gpu=driver_version",
             "--format=csv,noheader"],
            capture_output=True, timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    value = done.stdout.decode("ascii", errors="replace").strip() if done.returncode == 0 else ""
    return value.splitlines()[0].strip() if value else None


def _plain(value: Any) -> Any:
    """A str subclass as a plain str; anything else as it is."""
    return str(value) if isinstance(value, str) else value


def _optional(read: Any) -> Any:
    try:
        return read()
    except Exception:  # noqa: BLE001 — a fact that cannot be read is not known
        return None


def collect_environment(device: str) -> Dict[str, Any]:
    """The §5.3 facts for a run on `device` (a torch device string).

    The GPU facts describe the device the run uses, and are `None` for a run off
    CUDA. TF32 is read through both of torch's switches where they exist: the
    `allow_tf32` flags and the newer `fp32_precision` settings.
    """
    import torch

    env: Dict[str, Any] = {
        "python": platform.python_version(),
        # `torch.__version__` is a `TorchVersion`, a str subclass the safe
        # loader refuses; recorded as a plain str.
        "torch": str(torch.__version__),
        "torch_git_version": _optional(lambda: _plain(torch.version.git_version)),
        "cuda_build": _plain(torch.version.cuda),
        "cudnn": _optional(lambda: torch.backends.cudnn.version()),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": _optional(
            torch.is_deterministic_algorithms_warn_only_enabled),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "tf32": {
            "matmul_allow_tf32": _optional(lambda: torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": _optional(lambda: torch.backends.cudnn.allow_tf32),
            "matmul_fp32_precision": _optional(
                lambda: str(torch.backends.cuda.matmul.fp32_precision)),
            "cudnn_fp32_precision": _optional(
                lambda: str(torch.backends.cudnn.fp32_precision)),
        },
        "gpu": None,
        "pyg": _pyg_facts(),
    }
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        capability = _optional(lambda: torch.cuda.get_device_capability(index))
        env["gpu"] = {
            "name": _optional(lambda: _plain(torch.cuda.get_device_name(index))),
            "capability": list(capability) if capability is not None else None,
            "driver": _driver_version(index),
        }
    return env


def _pyg_facts() -> Dict[str, Any]:
    """PyG's version and which native extensions import, by name and version."""
    facts: Dict[str, Any] = {"version": None, "native": {}}
    try:
        import torch_geometric

        facts["version"] = _plain(getattr(torch_geometric, "__version__", None))
    except Exception:  # noqa: BLE001 — absent or broken: not known
        return facts
    try:
        from src.utils.pyg_native_check import check_pyg_native_extensions

        facts["native"] = {
            str(name): {"available": bool(status.available), "version": _plain(status.version)}
            for name, status in check_pyg_native_extensions().items()
        }
    except Exception:  # noqa: BLE001 — the probe itself failed: not known
        facts["native"] = None
    return facts


def assert_primitive(value: Any, where: str = "value") -> None:
    """Raise `TypeError` unless `value` is built only of the types a
    `weights_only=True` load accepts as plain data: str, int, float, bool,
    None, and dicts with string keys or lists of those."""
    # Exact types: a str or int subclass (torch's `TorchVersion`, an enum) is
    # pickled as its class, which the safe loader refuses.
    if value is None or type(value) in (str, bool, int, float):
        return
    if type(value) is dict:
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{where} has a non-string key {key!r}")
            assert_primitive(item, f"{where}.{key}")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            assert_primitive(item, f"{where}[{index}]")
        return
    raise TypeError(f"{where} is a {type(value).__name__}, not plain data")


__all__: List[str] = [
    "GENERATED_UNTRACKED_PREFIXES",
    "assert_primitive",
    "code_revision",
    "collect_environment",
]
