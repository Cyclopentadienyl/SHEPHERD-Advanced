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

**Named components** (research-comparison README §6, E1; BACKLOG 71, since
Stage 2b). Under ``components`` each named component of the stack is recorded in
one shape, whatever it is (`COMPONENT_FIELDS`, `component_facts`): its kind;
whether it is requested, installed, available, selected and observed in effect,
each true, false or ``"unknown"``; why it fell back, where that is known; its
distribution version; its native identity; and its source. ``requested``,
``selected`` and ``in_effect`` carry a value only where something observed it
and read ``"unknown"`` otherwise: an installed or importable component is never
read as one in effect.
Today's components are PyG's native extensions (`COMPONENT_READERS`); nothing
here or in a reader of records takes that list, or a CUDA device, for a rule,
and a component another part of the stack adds is one more reader. A component
whose use changes what is computed (candidate truncation, quantisation) is
stated in the protocol's or the model's semantics as well; this record alone
never carries such a change. The facts are read once per process
(`collect_components`), never raise (a fact that cannot be read is a status),
and never enter a semantic digest or a condition group: the environment is
recorded, never compared or enforced. Records written before Stage 2b carry
``pyg.native`` (each extension's import state and module version) instead and
are read as they are.

The deployment probe's A1 reads the components from here (`collect_components`).
A record's components are read by two projections, each taking only what the
record holds: `native_identity`, the display summary every page shows, and
`recorded_components`, every recorded fact, which the record views, the
results query, Compare and the export carry (review fix RC-P2-1).
Today's other collectors (the measurement manifest's fields, the deployment
probe's commit check, `src/utils/version_checker.py`) stay as they are until
their entry points are next changed; this is the one later entry points use
(the plan's "Not prerequisites").

Module: src/utils/run_environment.py
"""
from __future__ import annotations

import copy
import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from src.utils import pyg_native_check as _native
from src.utils.checkpoint_paths import MAX_REPORT_DEPTH, plain_data

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
    `allow_tf32` flags and the newer `fp32_precision` settings. ``pyg`` is
    `torch_geometric`'s version; ``components`` is `collect_components`, read
    once per process, while every other fact is read at each call (a setting
    can change within a process).
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
        "components": collect_components(),
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
    """`torch_geometric`'s version (its native extensions are components)."""
    try:
        import torch_geometric

        return {"version": _plain(getattr(torch_geometric, "__version__", None))}
    except Exception:  # noqa: BLE001 — absent or broken: not known
        return {"version": None}


# =============================================================================
# Named components (research-comparison README §6, E1; BACKLOG 71)
# =============================================================================
#: What a component's state reads when nothing observed it.
UNKNOWN = "unknown"

#: Every component's facts, in this order:
#:
#: - ``kind``: what sort of component it is (``"PyG native extension"``);
#: - ``requested``: a setting of the run asked for it, as recorded; true, false
#:   or ``"unknown"``. Apart from ``selected``: a setting can ask for one
#:   backend while a dispatcher chooses another;
#: - ``installed``: a distribution of it is installed (its package metadata is
#:   found on the interpreter's path); true, false or ``"unknown"`` (it imports,
#:   but no distribution's metadata names it);
#: - ``available``: this process can use it (it imports); true, false or
#:   ``"unknown"``;
#: - ``selected``: a dispatcher chose it for the run, as observed; true, false
#:   or ``"unknown"``;
#: - ``in_effect``: observed to be what ran; true, false or ``"unknown"``. Never
#:   true for a component that is not available or fell back;
#: - ``fallback_reason``: why it is not what runs, as observed; null otherwise;
#: - ``version``: its distribution's version, from the package metadata; null
#:   when none is installed or it cannot be read;
#: - ``native``: its native identity, ``{"state", "count", "aggregate_sha256"}``
#:   and a ``reason`` where the state is not ``recorded`` (`NATIVE_STATES`):
#:   the count is a whole number or null, and the aggregate a SHA-256 given
#:   exactly when the state is ``recorded``, which gives its count too;
#: - ``source``: where it came from, ``{"state": "recorded", "route", ...}``
#:   (the route is text; PyG's native extensions add the receipt's
#:   ``requested``, ``resolved``, ``wheel_sha256``, ``compiled_against``,
#:   ``toolchain`` and ``flags``) or ``{"state": "not recorded", "reason"}``.
#:
#: `component_facts` holds every component to this shape, which is what the
#: record view reads (`src.api.services.record_views.RecordComponent`, the
#: identity, and `RecordComponentFacts`, every fact): a fact the view would
#: refuse is refused when it is collected, and becomes the component's
#: status, so a record is never hidden for its environment.
COMPONENT_FIELDS: Tuple[str, ...] = (
    "kind", "requested", "installed", "available", "selected", "in_effect",
    "fallback_reason", "version", "native", "source",
)
_STATE_FIELDS = ("requested", "installed", "available", "selected", "in_effect")

#: How deep a component's facts may nest when collected (`component_facts`;
#: lists and mappings, the facts' own object the first): as deep as its
#: measuring run's report can hold them and still be registered. A
#: registration refuses a report nesting deeper than
#: `checkpoint_paths.MAX_REPORT_DEPTH` levels (which a measurement never comes
#: near), and a component's facts lie five levels
#: down in a report (``manifest.environment.facts.components.<name>``), their
#: values one level below their deepest list or mapping. Deeper, the component
#: is recorded as "its facts could not be read", so its run's report is still
#: registered and the collector's own copies (`collect_components`) stay far
#: from the interpreter's recursion limit. No count bound: a reader may observe
#: any number of build facts (review fix RC-R1). Pinned against the
#: registration by a test.
COMPONENT_DEPTH = MAX_REPORT_DEPTH - 5 - 1

#: A native identity's states: its libraries' count and aggregate digest
#: recorded; no distribution installed; installed, and its file list or a
#: library cannot be read (the count is set where the list was read); or no
#: native identity read for the component.
NATIVE_RECORDED = "recorded"
NATIVE_NOT_INSTALLED = "not installed"
NATIVE_UNREADABLE = "cannot be read"
NATIVE_NOT_RECORDED = "not recorded"
NATIVE_STATES = (NATIVE_RECORDED, NATIVE_NOT_INSTALLED, NATIVE_UNREADABLE, NATIVE_NOT_RECORDED)

#: A source's states: the deploy layer's source receipt matched what is
#: installed and its facts are recorded, or not, with why.
SOURCE_RECORDED = "recorded"
SOURCE_NOT_RECORDED = "not recorded"

#: The kind of PyG's native extensions.
PYG_NATIVE_KIND = "PyG native extension"


def _native_block(state: str, count: Optional[int] = None, aggregate: Optional[str] = None,
                  reason: Optional[str] = None) -> Dict[str, Any]:
    block: Dict[str, Any] = {"state": state, "count": count, "aggregate_sha256": aggregate}
    if reason is not None:
        block["reason"] = reason
    return block


def _not_recorded(reason: str) -> Dict[str, Any]:
    return {"state": SOURCE_NOT_RECORDED, "reason": reason}


def component_facts(kind: str, **observed: Any) -> Dict[str, Any]:
    """One component's facts in the one shape (`COMPONENT_FIELDS`), from what
    its reader observed; every fact it does not give reads as not observed
    (its states ``"unknown"``, no fallback known, no version, its native
    identity and source not recorded).

    Refuses (`ValueError`, `TypeError`) what no observation can say, and so
    everything the record view would refuse (`COMPONENT_FIELDS`): a fact
    outside the shape; a state that is neither a boolean nor ``"unknown"``; a
    component in effect that is not available or fell back; a fallback reason
    or version that is not text; a native identity whose state, count or
    aggregate is not one of the shape's (`_check_native`); a source whose state
    is not ``recorded`` or ``not recorded``, or with a route that is not text
    exactly when it is recorded (`_check_source`); a value that is not plain
    data (`assert_primitive`); or a value no answer can carry
    (`checkpoint_paths.plain_data`, the one rule every reader of the record
    holds it to again: a string or a key that is not text, such as a lone
    surrogate, or a number that is not finite, which the records' JSON
    encoding refuses), or nesting deeper than its report can hold it
    (`COMPONENT_DEPTH`). Valid Unicode of every script is text, and any
    number of facts is read. `collect_components` records such a refusal as
    the component's status and never raises it.
    """
    if not isinstance(kind, str) or not kind:
        raise ValueError("a component's kind is text")
    outside = sorted(set(observed) - set(COMPONENT_FIELDS[1:]))
    if outside:
        raise ValueError(f"facts outside a component's shape: {outside}")
    facts: Dict[str, Any] = {
        "kind": kind, "requested": UNKNOWN, "installed": UNKNOWN, "available": UNKNOWN,
        "selected": UNKNOWN, "in_effect": UNKNOWN, "fallback_reason": None, "version": None,
        "native": _native_block(NATIVE_NOT_RECORDED),
        "source": _not_recorded("no source is recorded for this component"),
    }
    facts.update(observed)
    for field in _STATE_FIELDS:
        value = facts[field]
        if type(value) is not bool and value != UNKNOWN:
            raise ValueError(f"a component's {field} is true, false or {UNKNOWN!r}")
    reason = facts["fallback_reason"]
    if reason is not None and not _is_text(reason):
        raise ValueError("a fallback reason is text")
    if facts["in_effect"] is True and (facts["available"] is False or reason is not None):
        raise ValueError("a component in effect is available and has not fallen back")
    if facts["version"] is not None and type(facts["version"]) is not str:
        raise ValueError("a component's version is text")
    _check_native(facts["native"])
    _check_source(facts["source"])
    assert_primitive(facts, "component")
    # What every reader of the record carries (review fix RC-R1): JSON's
    # escapes write a lone surrogate, and every answer that shows it, a CSV
    # or JSON export, an API response, would then fail on it; the encoding
    # every record is written with refuses NaN and infinity. Held to the depth
    # its report can be registered with, and to no count (`COMPONENT_DEPTH`).
    plain_data(facts, "component", depth=COMPONENT_DEPTH)
    return facts


_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def _is_text(value: Any) -> bool:
    return type(value) is str and bool(value)


def _check_reason(block: Mapping[str, Any], part: str) -> None:
    if "reason" in block and not _is_text(block["reason"]):
        raise ValueError(f"a component's {part} reason is text")


def _check_native(block: Any) -> None:
    """A native identity in the shape (`COMPONENT_FIELDS`): a state of
    `NATIVE_STATES`; its ``count`` and ``aggregate_sha256`` both given, the
    count a whole number or null, the aggregate 64 lowercase hex digits given
    exactly when the state is recorded, which gives its count too; a reason,
    where there is one, is text."""
    if type(block) is not dict or block.get("state") not in NATIVE_STATES:
        raise ValueError(f"a component's native identity has one of the states {NATIVE_STATES}")
    if "count" not in block or "aggregate_sha256" not in block:
        raise ValueError("a component's native identity gives its count and aggregate_sha256")
    count, aggregate = block["count"], block["aggregate_sha256"]
    if count is not None and (type(count) is not int or count < 0):
        raise ValueError("a native library count is a whole number of at least 0")
    if aggregate is not None and (type(aggregate) is not str
                                  or not _SHA256_HEX.fullmatch(aggregate)):
        raise ValueError("a native aggregate is a SHA-256 in 64 lowercase hex digits")
    recorded = block["state"] == NATIVE_RECORDED
    if recorded != (aggregate is not None) or (recorded and count is None):
        raise ValueError("a native identity's aggregate is given exactly when it is recorded, "
                         "with its count")
    _check_reason(block, "native identity")


def _check_source(block: Any) -> None:
    """A source in the shape (`COMPONENT_FIELDS`): ``recorded`` with its route
    as text, or ``not recorded`` with no route; a reason, where there is one,
    is text."""
    if type(block) is not dict or block.get("state") not in (SOURCE_RECORDED,
                                                             SOURCE_NOT_RECORDED):
        raise ValueError(f"a component's source is {SOURCE_RECORDED!r} or "
                         f"{SOURCE_NOT_RECORDED!r}")
    route = block.get("route")
    if (block["state"] == SOURCE_RECORDED) != (route is not None) or (
            route is not None and not _is_text(route)):
        raise ValueError("a component's source gives its route, as text, exactly when it is "
                         "recorded")
    _check_reason(block, "source")


@dataclass(frozen=True)
class ComponentReader:
    """A family of named components and how their facts are read (E1).

    `read` takes `names` and gives, for each, the facts it observed (without
    the kind, which `component_facts` adds before filling in what was not
    observed). A name it leaves out, a reader that raises, and facts no
    observation can say (`component_facts` refuses them) are recorded as that
    component's status: every state unknown, its native identity and source
    not read, and why.

    **What a reader gives is recorded, shown and exported.** Every fact it
    observes, its source's own build facts among them (a commit, a compiler,
    flags), goes into each checkpoint and manifest and every ledger record,
    and from there to the record views, the results query, Compare and the
    export (`recorded_components`). So a reader records nothing secret: no
    credential, token or private value, and no file name or path. The one D7
    redaction each text passes on its way out (`redact_paths`) replaces an
    absolute path that starts its own word (``build dir=/home/…``), its other
    words shown as they are; a path joined to other text, a compiler flag's
    (``-I/opt/…``, ``-L/srv/…``) or a ``file:`` URL's, is not caught and is
    shown as written, so a reader never records one.
    """

    kind: str
    names: Tuple[str, ...]
    read: Callable[[Tuple[str, ...]], Mapping[str, Mapping[str, Any]]]


_FACTS_NOT_READ = "its facts could not be read"


def _not_read(kind: str) -> Dict[str, Any]:
    return component_facts(kind, native=_native_block(NATIVE_UNREADABLE, reason=_FACTS_NOT_READ),
                           source=_not_recorded(_FACTS_NOT_READ))


def _read_receipt() -> Tuple[Optional[Dict[str, Any]], str]:
    """The deploy layer's source receipt in this interpreter's environment
    (`pyg_native_check.read_receipt`), or None and why there is none to read,
    in fixed words: an error's own text can carry a path."""
    try:
        receipt = _native.read_receipt()
    except ValueError:
        return None, "the file in the source receipt's place is not a receipt this build reads"
    except Exception:  # noqa: BLE001 — an OSError names its path
        return None, "the source receipt in this environment cannot be read"
    if receipt is None:
        return None, "no source receipt is kept in this environment"
    return receipt, ""


def _native_identity(installed: "_native.InstalledIdentity", available: Any) -> Dict[str, Any]:
    """An installed distribution's native identity: the count of its native
    libraries and their aggregate digest (`pyg_native_check.installed_native_facts`;
    the method is `read_pyg_native`'s)."""
    if installed.copies == 0:
        if available is True:
            return _native_block(NATIVE_UNREADABLE, reason="it imports, but no installed "
                                 "distribution lists its files")
        return _native_block(NATIVE_NOT_INSTALLED)
    if installed.native is None:
        return _native_block(NATIVE_UNREADABLE,
                             reason="its installed file list (RECORD) cannot be read")
    facts = _native.installed_native_facts(installed.native)
    if facts["native_aggregate_sha256"] is None:
        return _native_block(NATIVE_UNREADABLE, count=facts["native_count"],
                             reason="an installed native library cannot be read")
    return _native_block(NATIVE_RECORDED, facts["native_count"], facts["native_aggregate_sha256"])


#: The fields of a receipt's requested source that a record leaves out: the
#: prebuilt release's asset file name, which its ``asset_sha256`` names, as the
#: wheel's SHA-256 names the wheel.
_REQUESTED_LEFT_OUT = ("asset",)


def _requested_source(requested: Any) -> Any:
    """A receipt entry's requested source as a record keeps it: no file name
    (`_REQUESTED_LEFT_OUT`), and its URL without user information, query or
    fragment (`pyg_native_check.strip_credentials`, as the deploy layer wrote
    it), whatever a hand-edited receipt holds. None when the receipt names
    none; a value that is not an object raises `ValueError`."""
    if requested is None:
        return None
    if not isinstance(requested, Mapping):
        raise ValueError("the requested source is not an object")
    kept = {key: value for key, value in requested.items() if key not in _REQUESTED_LEFT_OUT}
    if isinstance(kept.get("url"), str):
        kept["url"] = _native.strip_credentials(kept["url"])
    return kept


def _receipt_source(name: str, installed: "_native.InstalledIdentity", available: Any,
                    receipt: Optional[Mapping[str, Any]], problem: str) -> Dict[str, Any]:
    """The receipt's facts for one installed distribution, only when the
    receipt's entry for it matches what is installed (`receipt_entry_state`,
    the deploy check's own comparison); otherwise why not. No file name is
    kept: the wheel's SHA-256 names the wheel, and the requested source is
    `_requested_source`'s."""
    if installed.copies == 0:
        return _not_recorded("no installed distribution names it" if available is True
                             else "it is not installed")
    if receipt is None:
        return _not_recorded(problem)
    wanted = _native.normalised_name(name)
    entries = [entry for entry in receipt.get("wheels") or []
               if isinstance(entry, Mapping)
               and _native.normalised_name(str(entry.get("distribution"))) == wanted]
    if not entries:
        return _not_recorded("the source receipt does not name it")
    if len(entries) > 1:
        return _not_recorded("the source receipt names it more than once")
    entry = entries[0]
    state = _native.receipt_entry_state(entry, installed)
    if state["state"] != _native.RECEIPT_MATCHES:
        return _not_recorded(
            f"the source receipt does not describe what is installed: {state['reason']}")
    route = receipt.get("route")
    if type(route) is not str or not route:
        return _not_recorded("the source receipt names no route")
    try:
        requested = _requested_source(entry.get("requested"))
    except ValueError:
        return _not_recorded("the source receipt's requested source is not one this build reads")
    wheel = entry.get("wheel") if isinstance(entry.get("wheel"), Mapping) else {}
    return {"state": SOURCE_RECORDED, "route": route,
            "requested": requested, "resolved": entry.get("resolved"),
            "wheel_sha256": wheel.get("sha256"),
            "compiled_against": receipt.get("compiled_against"),
            "toolchain": receipt.get("toolchain"), "flags": receipt.get("flags")}


def read_pyg_native(names: Tuple[str, ...]) -> Dict[str, Dict[str, Any]]:
    """PyG's native extensions, as components (`ComponentReader.read`).

    For each extension, by its import name (its distribution's name under
    PEP 503):

    - ``installed`` and ``version``: the first distribution of the name on the
      interpreter's path, as an import finds it, and its version from its
      package metadata (`pyg_native_check.installed_identity`); ``"unknown"``
      when it imports but no distribution's metadata names it;
    - ``available``: whether it imports in this process (the import probe,
      `check_pyg_native_extensions`, cached). Where it does not, ``in_effect``
      is false, which is observed (it cannot run here), and the fallback reason
      says only what was observed: it is not installed, or it is installed but
      does not import. What PyG does instead is not observed here (it differs
      by operation and by PyG release) and is not said. Where it imports,
      ``selected`` and ``in_effect`` stay ``"unknown"``: PyG chooses per
      operation, and nothing here observes it. ``requested`` stays
      ``"unknown"``: nothing here reads a setting that asks for one;
    - ``native``: **the native identity.** The distribution's file list
      (``RECORD``) names its files; those inside site-packages whose names are
      a native library's (`is_native_library`) are each hashed with SHA-256
      (`stream_sha256`). Recorded: their count, and one aggregate digest, the
      SHA-256 of the UTF-8 text with one line ``<name>\\t<sha256>\\n`` per
      library, ``<name>`` its POSIX name relative to site-packages, the lines
      in code-point order of the names (`aggregate_sha256`). The names are
      hashed, never recorded; two builds of one version differ here. The
      source receipt keys on the same aggregate;
    - ``source``: the deploy layer's source receipt (`pyg_native_check`), its
      route, requested source (no file name, its URL without credentials:
      `_requested_source`) and resolved source, wheel SHA-256, the torch and
      CUDA build compiled against, toolchain and flags, only when its entry's
      version and native aggregate equal the installed ones; otherwise ``not
      recorded``, with why (no receipt, it does not name the extension, or
      what differs).

    An extension whose facts cannot be read is left out, so the collector
    records it as such (`ComponentReader`).
    """
    imports = _native.check_pyg_native_extensions()
    receipt, problem = _read_receipt()
    observed: Dict[str, Dict[str, Any]] = {}
    for name in names:
        try:
            installed = _native.installed_identity(name)
            status = imports.get(name)
            available: Any = bool(status.available) if status is not None else UNKNOWN
            facts: Dict[str, Any] = {
                # Something imports under the name, but no distribution's
                # metadata names it: whether it is installed is not known.
                "installed": (True if installed.copies
                              else UNKNOWN if available is True else False),
                "available": available,
                "version": _plain(installed.version) if installed.copies else None,
                "native": _native_identity(installed, available),
                "source": _receipt_source(name, installed, available, receipt, problem),
            }
            if available is False:
                facts["in_effect"] = False
                facts["fallback_reason"] = ("it is installed but does not import"
                                            if installed.copies else "it is not installed")
            observed[name] = facts
        except Exception:  # noqa: BLE001 — this extension's facts are not known
            continue
    return observed


#: The readers of every named component, in record order. Today's one family
#: is PyG's native extensions; this is the boundary another component joins
#: at, never a list a record is checked against.
COMPONENT_READERS: Tuple[ComponentReader, ...] = (
    ComponentReader(PYG_NATIVE_KIND, _native.NATIVE_EXTENSIONS, read_pyg_native),
)

#: `collect_components`' one reading in this process.
_components_cache: Optional[Dict[str, Dict[str, Any]]] = None


def _read_components() -> Dict[str, Dict[str, Any]]:
    components: Dict[str, Dict[str, Any]] = {}
    for reader in COMPONENT_READERS:
        try:
            observed = reader.read(tuple(reader.names))
        except Exception:  # noqa: BLE001 — the reader failed: each of its names says so
            observed = {}
        for name in reader.names:
            try:
                components[name] = component_facts(reader.kind, **dict(observed[name]))
            except Exception:  # noqa: BLE001 — not given, or not a fact: a status
                components[name] = _not_read(reader.kind)
    return components


def collect_components() -> Dict[str, Dict[str, Any]]:
    """Every named component's facts (`COMPONENT_READERS`, `component_facts`),
    by name: read once per process, since hashing native libraries costs a
    read of each, and given as a copy, so a caller's change never reaches
    another. Never raises: a fact that cannot be read is a status."""
    global _components_cache
    if _components_cache is None:
        _components_cache = _read_components()
    return copy.deepcopy(_components_cache)


def native_identity(components: Any) -> Any:
    """Each named component's native identity and route, as the record views
    show them (`src.api.services.record_views.RecordView.native_identity`, the
    Test page, the deployment probe's verdict): its ``kind`` and ``version``,
    its ``native`` ``state``, ``count`` and ``aggregate_sha256``, and its
    ``source`` ``state``, with the ``route`` only where the source receipt
    matched what was installed.

    It picks fields and validates nothing, so a reader decides what to refuse:
    a value that is not an object is returned as it is, and a field an entry
    lacks is left out. ``components`` is `collect_components`' output as an
    environment record carries it.
    """
    if not isinstance(components, Mapping):
        return components
    return {name: _identity_of(entry) for name, entry in components.items()}


def _identity_of(entry: Any) -> Any:
    if not isinstance(entry, Mapping):
        return entry
    picked = {field: entry[field] for field in ("kind", "version", "native", "source")
              if field in entry}
    for part, fields in (("native", ("state", "count", "aggregate_sha256")),
                         ("source", ("state", "route"))):
        value = picked.get(part)
        if isinstance(value, Mapping):
            picked[part] = {field: value[field] for field in fields if field in value}
    return picked


#: A native identity's facts a record holds (`COMPONENT_FIELDS`), its reason
#: among them.
_NATIVE_FACTS = ("state", "count", "aggregate_sha256", "reason")
#: What a path in a recorded text is shown as (`recorded_components`): the
#: words every surface uses for a server path it does not name.
_SERVER_PATH = "a server path"


def recorded_components(components: Any) -> Any:
    """**Every recorded fact of each named component**, as every reader
    passes them on: the record views' ``components``
    (`src.api.services.record_views.RecordComponentFacts`) and through them
    the results service's rows, Compare and the export (research comparison
    §6 E1; review fix RC-P2-1). `native_identity` stays the display summary.

    By name, as `native_identity` keys them, each component's facts in the
    shape's order (`COMPONENT_FIELDS`): its ``kind``; its ``requested``,
    ``installed``, ``available``, ``selected`` and ``in_effect`` states as
    recorded; its ``fallback_reason`` and ``version``; its ``native``
    identity's ``state``, ``count``, ``aggregate_sha256`` and ``reason``;
    and its ``source``, every fact the record holds (its ``state``,
    ``route`` and ``reason``, the source receipt's ``requested``,
    ``resolved``, ``wheel_sha256``, ``compiled_against``, ``toolchain`` and
    ``flags``, or another reader's own build facts, such as a commit or a
    compiler), the same for every component and every reader. A requested
    source is shown as a record keeps one (`_requested_source`: no file
    name, its URL without credentials), whatever an older record holds.

    Taken only from ``components``, the environment record a report or a
    ledger record already read holds (``environment.facts.components``):
    no path is opened and nothing is collected. **A field an entry lacks is
    left out**, never filled in (a reader reads it as unknown: a record
    written before ``requested`` was recorded has none). **Every text**, a
    key inside a fact or a component's name included, **passes the one D7
    redaction** (`path_redaction.redact_paths`): an absolute path that
    starts its own word is shown as "a server path"; one joined to other
    text (a flag's ``-I/…``, a ``file:`` URL) is not caught, so a reader
    records none (`ComponentReader`). Nothing else is checked here, so the
    reader decides what to refuse, as for `native_identity`: a value that is
    not an object is returned as it is. The record views hold what this
    gives to the collector's own rule (`checkpoint_paths.plain_data`: text,
    finite numbers; review fix RC-R1), so what they pass on is plain data an
    answer can carry; a record whose projection holds anything else is
    listed as unreadable, naming the field by this projection's keys and
    never its value, and what this leaves out or redacts never hides it."""
    if not isinstance(components, Mapping):
        return components
    return {_redacted(name): _facts_of(entry) for name, entry in components.items()}


def _facts_of(entry: Any) -> Any:
    if not isinstance(entry, Mapping):
        return entry
    picked = {field: entry[field] for field in COMPONENT_FIELDS if field in entry}
    native = picked.get("native")
    if isinstance(native, Mapping):
        picked["native"] = {field: native[field] for field in _NATIVE_FACTS if field in native}
    source = picked.get("source")
    if isinstance(source, Mapping):
        source = dict(source)
        requested = source.get("requested")
        if isinstance(requested, Mapping):
            try:
                source["requested"] = _requested_source(requested)
            except ValueError:  # a URL that does not parse: shown without it
                source["requested"] = {key: value for key, value in requested.items()
                                       if key not in (*_REQUESTED_LEFT_OUT, "url")}
        picked["source"] = source
    return _redacted(picked)


def _redacted(value: Any) -> Any:
    """`value` with every text in it, an object's keys included, through the
    one D7 redaction (`redact_paths`, no root named: any absolute path is
    replaced); every other value as it is."""
    from src.utils.path_redaction import redact_paths

    if isinstance(value, str):
        return redact_paths(value, (), other=_SERVER_PATH)
    if isinstance(value, Mapping):
        return {_redacted(key): _redacted(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redacted(item) for item in value]
    return value


def _identity_items(identity: Any) -> Tuple[Optional[str], List[Tuple[str, Optional[str]]]]:
    """The words of `native_identity`, for its line and its lines: the state
    that stands in for the whole identity (``"not recorded"`` when there is
    none, a record written before research-comparison Stage 2b), or None and
    each component's words with its route's (None where it has no route: it
    is not installed, or its entry cannot be read). By name, in code-point
    order, whatever order the record keeps: a record read from the ledger,
    whose encoding orders keys, and the same record read from its report give
    the same words. Never raises: a part that cannot be read says so."""
    if identity is None:
        return "not recorded", []
    if not isinstance(identity, Mapping):
        return "cannot be read", []
    if not identity:
        return "no component recorded", []
    items: List[Tuple[str, Optional[str]]] = []
    for name, entry in sorted(identity.items(), key=lambda item: str(item[0])):
        if not isinstance(entry, Mapping):
            items.append((f"{name} cannot be read", None))
            continue
        native = entry.get("native") if isinstance(entry.get("native"), Mapping) else {}
        state = native.get("state")
        if state == NATIVE_NOT_INSTALLED:
            items.append((f"{name} not installed", None))
            continue
        version = entry.get("version")
        words = [str(name), str(version) if version else "version not read"]
        digest = native.get("aggregate_sha256")
        if state == NATIVE_RECORDED and isinstance(digest, str):
            words.append(digest[:12] if native.get("count") else "no native library")
        else:
            words.append(f"libraries {state}" if isinstance(state, str) and state
                         else "libraries not recorded")
        source = entry.get("source") if isinstance(entry.get("source"), Mapping) else {}
        route = source.get("route") if source.get("state") == SOURCE_RECORDED else None
        items.append((" ".join(words), str(route) if route else "route not recorded"))
    return None, items


def native_identity_text(identity: Any) -> str:
    """`native_identity` as one line of plain text (Models, the probe's
    console): each component by name, its version and the first 12 hex
    digits of its native libraries' aggregate digest (or the state that
    stands in for it), and its source's route, said once in front when every
    installed component shares it (`_identity_items`). ``"not recorded"``
    when there is no identity. Never raises."""
    whole, items = _identity_items(identity)
    if whole is not None:
        return whole
    routes = {route for _text, route in items if route is not None}
    if len(routes) == 1:
        return f"{routes.pop()} — " + ", ".join(text for text, _route in items)
    return ", ".join(text if route is None else f"{text} ({route})" for text, route in items)


def native_identity_lines(identity: Any) -> List[str]:
    """`native_identity` in the line's words (`native_identity_text`), as a
    lead and one line per component, for a surface whose column is too narrow
    for the one line (the Test page's result). The lead says the route once
    when every installed component shares it, and how many components there
    are (``"local build · 4 components"``); where the routes differ, it says
    so and each component's line names its own. The state that stands in for
    the whole identity is the only line when there is no component. Never
    raises."""
    whole, items = _identity_items(identity)
    if whole is not None:
        return [whole]
    routes = {route for _text, route in items if route is not None}
    count = f"{len(items)} component{'' if len(items) == 1 else 's'}"
    if len(routes) == 1:
        return [f"{routes.pop()} · {count}", *(text for text, _route in items)]
    lead = f"{count} · routes differ" if routes else count
    return [lead, *(text if route is None else f"{text} ({route})" for text, route in items)]


@dataclass(frozen=True)
class AutocastRegime:
    """The autocast state at a **harness boundary**, and nothing wider.

    Named for what it can see. This is read at the point the harness performs or
    calls a computation; an autocast context opened *inside* a model is not
    observable from here and is not claimed. The same discipline as
    `torch_compile_wrapped`, and for the same reason: a field whose name promises
    more than its probe can see gets read for the promise.

    `dtype` is `None` exactly when `enabled` is false, so a reader never has to
    decide whether a dtype beside a disabled regime means anything.

    **One home for its readers**: the measurement command observes the run's
    regime with `observe_autocast_regime`, and the task evaluators check it.
    """

    enabled: bool
    dtype: Optional[str]


def observe_autocast_regime(device: Any) -> AutocastRegime:
    """Read the ambient autocast state for `device`'s type.

    Device-independent: `torch.is_autocast_enabled("cuda")` answers on a CPU-only
    host, and `torch.device("cuda:1").type` resolves to `"cuda"`, so an indexed
    device is handled.
    """
    import torch as _torch

    device_type = _torch.device(device).type if device is not None else "cpu"
    if not _torch.is_autocast_enabled(device_type):
        return AutocastRegime(enabled=False, dtype=None)
    return AutocastRegime(enabled=True, dtype=str(_torch.get_autocast_dtype(device_type)))


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
    "AutocastRegime",
    "COMPONENT_DEPTH",
    "COMPONENT_FIELDS",
    "COMPONENT_READERS",
    "ComponentReader",
    "GENERATED_UNTRACKED_PREFIXES",
    "NATIVE_STATES",
    "PYG_NATIVE_KIND",
    "UNKNOWN",
    "assert_primitive",
    "code_revision",
    "collect_components",
    "collect_environment",
    "component_facts",
    "native_identity",
    "native_identity_lines",
    "native_identity_text",
    "observe_autocast_regime",
    "read_pyg_native",
    "recorded_components",
]
