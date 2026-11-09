"""
# ==============================================================================
# Module: src/utils/pyg_native_check.py
# ==============================================================================
# Purpose: Runtime verification that PyG native extensions are actually
#          importable (and therefore being used), rather than silently falling
#          back to slower pure-Python / torch.scatter_reduce kernels.
#
# Why this exists:
#   PyG degrades *gracefully* when pyg-lib / torch-scatter / torch-sparse /
#   torch-cluster are missing — it raises no exception and prints no warning, it
#   just routes ops through slower fallbacks. So "training runs" does NOT prove
#   the self-compiled native kernels are active. scripts/validate_pyg_ext.py
#   answers this at deploy time; this module makes the same fact observable at
#   *runtime* (a one-time log line, plus a UI-facing warning string).
#
# Scope:
#   Two parts, kept apart because they cost differently.
#   - The import probe (`check_pyg_native_extensions` and its summaries): pure
#     import probing, no torch tensors created, safe to call anywhere (including
#     processes without a GPU), cached after the first probe. It is what a page
#     load may call (src/utils/version_checker.py), and it hashes nothing.
#   - The install identity (BACKLOG 70, 71; research-comparison Stage 2): what an
#     installed distribution and its native libraries are, byte for byte, against
#     the wheel that was meant to be installed, and the path-free source receipt
#     the deploy layer keeps in the virtual environment. It reads and hashes
#     files, so it runs at deploy time (scripts/build_pyg_arm.sh, deploy.sh's
#     prebuilt pull, through scripts/validate_pyg_ext.py), never on a page load.
#     `stream_sha256` is its one digest. Never a gate on a model or a result: a
#     difference is recorded, and only an install that is not what was meant to
#     be installed fails the deploy step that made it.
#   Standard library only, so the deploy layer can run it before anything else.
# ==============================================================================
"""
from __future__ import annotations

import email.parser
import hashlib
import importlib
import importlib.metadata
import io
import json
import logging
import os
import re
import sys
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Native extensions PyG can offload kernels to. Tuple order = display order.
_NATIVE_EXTENSIONS = ("pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster")


class ExtensionStatus(NamedTuple):
    """Probe result for a single native extension."""

    name: str
    available: bool
    version: Optional[str]
    error: Optional[str]  # short reason when unavailable (import error text)


def _probe(name: str) -> ExtensionStatus:
    """Try to import one extension. A broken build may raise at import time
    (not only ImportError), so we catch broadly and record the reason."""
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - any failure means "not usable"
        return ExtensionStatus(name, False, None, str(exc))
    version = getattr(module, "__version__", None)
    return ExtensionStatus(name, True, version, None)


_cached: Optional[Dict[str, ExtensionStatus]] = None


def check_pyg_native_extensions(force: bool = False) -> Dict[str, ExtensionStatus]:
    """Probe each native extension once and cache the structured result.

    Python already caches the imports themselves; we additionally cache the
    parsed result so repeated UI polls cost nothing. Pass ``force=True`` to
    re-probe (e.g. after a deliberate reinstall).
    """
    global _cached
    if _cached is not None and not force:
        return _cached
    _cached = {name: _probe(name) for name in _NATIVE_EXTENSIONS}
    return _cached


def get_missing_extensions(
    statuses: Optional[Dict[str, ExtensionStatus]] = None,
) -> List[str]:
    """Names of extensions that failed to import (i.e. running on fallback)."""
    statuses = statuses or check_pyg_native_extensions()
    return [name for name in _NATIVE_EXTENSIONS if not statuses[name].available]


def format_status_line(
    statuses: Optional[Dict[str, ExtensionStatus]] = None,
) -> str:
    """One-line human-readable summary for logs."""
    statuses = statuses or check_pyg_native_extensions()
    parts = []
    for name in _NATIVE_EXTENSIONS:
        st = statuses[name]
        if st.available:
            parts.append(f"{name} {st.version or '?'} OK")
        else:
            parts.append(f"{name} MISSING")
    return "PyG native extensions: " + " | ".join(parts)


def get_fallback_warning(
    statuses: Optional[Dict[str, ExtensionStatus]] = None,
) -> Optional[str]:
    """User-facing warning string if any extension is missing, else ``None``.

    Intended for the WebUI error/warning banner. This is a *warning*, not an
    error: the system still runs, just slower for the affected ops.
    """
    missing = get_missing_extensions(statuses)
    if not missing:
        return None
    return (
        "PyG native extensions missing: "
        + ", ".join(missing)
        + ". Training and inference still run, but the affected GNN ops "
        "(notably HGTConv segment_matmul and neighbor sampling) fall back to "
        "slower pure-Python / torch.scatter_reduce kernels. "
        "Run scripts/validate_pyg_ext.py to diagnose, or re-run the deploy "
        "PyG-extension step."
    )


def log_pyg_native_status(
    statuses: Optional[Dict[str, ExtensionStatus]] = None,
) -> Dict[str, ExtensionStatus]:
    """Emit the status to the logger once: INFO when all present, WARNING when
    any are missing. Returns the statuses so callers can reuse them."""
    statuses = statuses or check_pyg_native_extensions()
    line = format_status_line(statuses)
    if get_missing_extensions(statuses):
        logger.warning("%s  -> running on fallback kernels for the missing ones", line)
    else:
        logger.info(line)
    return statuses


# ==============================================================================
# The install identity (BACKLOG 70, 71; research-comparison Stage 2, O5)
# ==============================================================================
# Deploy-time only: everything below reads and hashes files. Nothing on a page
# load calls it (src/utils/version_checker.py uses the import probe above).

_CHUNK_BYTES = 1 << 20

#: A native library inside a distribution, by its name: a shared object, a
#: Windows extension module or library, or a macOS library, versioned or not.
_NATIVE_LIBRARY = re.compile(r"\.(?:so|pyd|dll|dylib)(?:\.[0-9]+)*$", re.IGNORECASE)


def stream_sha256(stream: BinaryIO) -> str:
    """SHA-256 of everything `stream` yields from where it stands, read in chunks.

    **The one digest of the install identity.** A wheel file, its members, an
    installed native library, a local source and the aggregates below all go
    through it, so the deploy check, the source receipt and the run
    environment's native identity hash the same way. It takes an open binary
    stream, never a path: the caller opens what it means.

    Not `src.utils.fingerprint.file_sha256`: that one identifies a run's
    provenance inputs (contract M2.1) and is where the path-form poison of the
    provenance pins sits. These bytes describe the environment; they are never
    a run's input, and a record carrying them is never compared as one.
    """
    digest = hashlib.sha256()
    while True:
        block = stream.read(_CHUNK_BYTES)
        if not block:
            return digest.hexdigest()
        digest.update(block)


def aggregate_sha256(named: Mapping[str, str]) -> str:
    """One digest for a set of named digests, without emitting the names.

    The SHA-256 of the UTF-8 text made of one line ``<name>\\t<sha256>\\n`` per
    entry, the entries in code-point order of their names. A name is a
    distribution-relative POSIX name (``torch_scatter/_scatter_cuda.so``) or a
    tree-relative one; it is hashed, never recorded. A name holding a tab or a
    line break is refused, since the text would no longer say which line is
    which.
    """
    lines = []
    for name in sorted(named):
        if "\t" in name or "\n" in name or "\r" in name:
            raise ValueError("a name with a tab or a line break cannot be aggregated")
        lines.append(f"{name}\t{named[name]}\n")
    return stream_sha256(io.BytesIO("".join(lines).encode("utf-8")))


def is_native_library(name: str) -> bool:
    """Whether a file name inside a distribution is a native library."""
    return bool(_NATIVE_LIBRARY.search(name.rsplit("/", 1)[-1]))


def normalised_name(name: str) -> str:
    """A distribution name as PEP 503 compares it."""
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True)
class WheelIdentity:
    """What a wheel file holds, as an install of it must reproduce.

    `native` maps each native library's site-packages-relative name to its
    SHA-256. A member under ``<name>.data/purelib/`` or ``.../platlib/`` is
    named where it is installed; members of the other ``.data`` schemes are
    installed outside site-packages and are not compared.
    """

    name: str
    version: str
    file: str  # the wheel's file name, no directory
    sha256: str
    native: Dict[str, str]


def _installed_name(member: str) -> Optional[str]:
    """Where a wheel member lands relative to site-packages, or None."""
    if member.endswith("/"):
        return None
    head, _, rest = member.partition("/")
    if head.endswith(".data"):
        scheme, _, inner = rest.partition("/")
        return inner if scheme in ("purelib", "platlib") and inner else None
    return member


def wheel_identity(wheel: os.PathLike) -> WheelIdentity:
    """Read a wheel's name, version, file digest and native library digests.

    A wheel that cannot be read raises `OSError` (the file) or `ValueError`
    (its content: not a zip archive, a damaged member, no single METADATA
    naming a Name and a Version)."""
    wheel = Path(wheel)
    with wheel.open("rb") as handle:
        file_digest = stream_sha256(handle)
    try:
        with zipfile.ZipFile(wheel) as archive:
            members = archive.namelist()
            metadata = [m for m in members
                        if m.count("/") == 1 and m.endswith(".dist-info/METADATA")]
            if len(metadata) != 1:
                raise ValueError(f"{wheel.name}: expected one .dist-info/METADATA, "
                                 f"found {len(metadata)}")
            headers = email.parser.BytesHeaderParser().parsebytes(archive.read(metadata[0]))
            name, version = headers.get("Name"), headers.get("Version")
            if not name or not version:
                raise ValueError(f"{wheel.name}: its METADATA names no Name or Version")
            native = {}
            for member in members:
                target = _installed_name(member)
                if target is not None and is_native_library(target):
                    with archive.open(member) as stream:
                        native[target] = stream_sha256(stream)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{wheel.name} is not a readable wheel: {exc}") from exc
    return WheelIdentity(name=str(name).strip(), version=str(version).strip(),
                         file=wheel.name, sha256=file_digest, native=native)


@dataclass(frozen=True)
class InstalledIdentity:
    """An installed distribution, as the interpreter's path finds it.

    `copies` is how many distributions of the name are in the path's distinct
    locations (`distinct_locations`); the other fields describe the first, the
    one an import uses. `native` is None when
    its file list (``RECORD``) cannot be read; a listed native library that
    cannot be read maps to None. (One that is listed but gone from disk is left
    out of the file list by `importlib.metadata` itself, so it reads as not
    installed.)
    """

    name: str
    copies: int
    version: Optional[str]
    native: Optional[Dict[str, Optional[str]]]


def distinct_locations(path: Sequence[str]) -> List[str]:
    """`path` with each location once, in order: an entry that resolves to a
    place an earlier entry already names is left out.

    A virtual environment's ``lib64`` is a link to its ``lib``, and where
    ``sys.platlibdir`` is ``lib64`` (the RHEL, Fedora and SUSE Pythons) `site`
    puts both names on ``sys.path``, since it compares absolute paths, not real
    ones. Searched as given, one installed distribution would be found twice.
    """
    seen = set()
    distinct = []
    for entry in path:
        real = os.path.realpath(entry)
        if real not in seen:
            seen.add(real)
            distinct.append(entry)
    return distinct


def installed_identity(name: str, path: Optional[Sequence[str]] = None) -> InstalledIdentity:
    """The installed distribution called `name` (PEP 503 comparison), looked up
    in the distinct locations of `path` (default: ``sys.path``), with its native
    libraries' digests. Native libraries installed outside site-packages (a
    ``RECORD`` entry that starts with ``..``) are not compared, as on the wheel
    side."""
    locations = distinct_locations(sys.path if path is None else path)
    found = list(importlib.metadata.distributions(name=name, path=locations))
    if not found:
        return InstalledIdentity(name=name, copies=0, version=None, native=None)
    first = found[0]
    files = first.files
    native: Optional[Dict[str, Optional[str]]] = None
    if files is not None:
        native = {}
        for entry in files:
            relative = entry.as_posix()
            if relative.startswith("..") or not is_native_library(relative):
                continue
            try:
                with open(first.locate_file(entry), "rb") as handle:
                    native[relative] = stream_sha256(handle)
            except OSError:
                native[relative] = None
    return InstalledIdentity(name=name, copies=len(found), version=first.version,
                             native=native)


@dataclass(frozen=True)
class WheelCheck:
    """One wheel against what is installed under its name: `problems` is empty
    exactly when the installed distribution's version and every native
    library's SHA-256 equal the wheel's."""

    wheel: WheelIdentity
    installed: InstalledIdentity
    problems: Tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


def _problems(wheel: WheelIdentity, installed: InstalledIdentity) -> List[str]:
    if installed.copies == 0:
        return ["not installed"]
    problems = []
    if installed.copies > 1:
        problems.append(f"installed {installed.copies} times on the path")
    if installed.version != wheel.version:
        problems.append(f"installed version {installed.version}, the wheel's {wheel.version}")
    if not wheel.native:
        problems.append("the wheel holds no native library")
    if installed.native is None:
        problems.append("its installed file list (RECORD) cannot be read")
        return problems
    for name in sorted(set(wheel.native) | set(installed.native)):
        if name not in installed.native:
            problems.append(f"native library {name} of the wheel is not installed")
        elif name not in wheel.native:
            problems.append(f"installed native library {name} is not in the wheel")
        elif installed.native[name] is None:
            problems.append(f"native library {name} is listed as installed but cannot be read")
        elif installed.native[name] != wheel.native[name]:
            problems.append(f"native library {name} differs from the wheel's")
    return problems


def check_installed_against_wheels(
    wheels: Sequence[os.PathLike], path: Optional[Sequence[str]] = None,
) -> List[WheelCheck]:
    """The one install check: each wheel against the distribution installed
    under its name. Two wheels naming one distribution are both refused, since
    only one of them can be what is installed."""
    identities = [wheel_identity(wheel) for wheel in wheels]
    named: Dict[str, int] = {}
    for identity in identities:
        key = normalised_name(identity.name)
        named[key] = named.get(key, 0) + 1
    checks = []
    for identity in identities:
        installed = installed_identity(identity.name, path)
        problems = _problems(identity, installed)
        if named[normalised_name(identity.name)] > 1:
            problems.insert(0, "more than one of the given wheels is this distribution")
        checks.append(WheelCheck(wheel=identity, installed=installed, problems=tuple(problems)))
    return checks


# --- What was asked for: a source, described without a path ---------------

_VCS_SCHEMES = ("git", "hg", "svn", "bzr")
_ARCHIVE_SUFFIXES = (".whl", ".tar.gz", ".tgz", ".tar.bz2", ".tbz", ".tar.xz", ".txz",
                     ".tar", ".zip")
#: Directories a local source tree's digest leaves out: version control's own
#: data and Python's bytecode cache, neither of which a build reads as source.
_TREE_SKIPPED = frozenset({".git", ".hg", ".svn", "__pycache__"})
_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*):")
_DIRECT_REFERENCE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*@\s*(\S.*)$")
_REQUIREMENT = re.compile(
    r"^([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)\s*(?:\[[^\]]*\])?\s*([^;]*?)\s*(?:;.*)?$")
#: A version specifier set (PEP 440 operators and version characters), or none.
_SPECIFIER = re.compile(r"^(?:(?:===|==|!=|~=|<=|>=|<|>)[A-Za-z0-9.*+!_-]+"
                        r"(?:,(?:===|==|!=|~=|<=|>=|<|>)[A-Za-z0-9.*+!_-]+)*)?$")


def strip_credentials(url: str) -> str:
    """`url` with no user information (user name or password), query or
    fragment: what is left names the place, not who fetched from it or with
    which token."""
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError:
        port = None
    netloc = host + (f":{port}" if port else "")
    return urllib.parse.urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def tree_sha256(root: os.PathLike) -> str:
    """A local source tree's digest: `aggregate_sha256` over each regular
    file's tree-relative POSIX name and SHA-256, version control data and
    bytecode caches left out, symbolic links to directories not followed."""
    root = Path(root)
    named = {}
    for directory, subdirectories, files in os.walk(root):
        subdirectories[:] = sorted(d for d in subdirectories if d not in _TREE_SKIPPED)
        for file_name in files:
            full = Path(directory) / file_name
            if full.is_file():
                with full.open("rb") as handle:
                    named[full.relative_to(root).as_posix()] = stream_sha256(handle)
    return aggregate_sha256(named)


def _local_source(path: Path) -> Dict[str, Any]:
    if path.is_file():
        with path.open("rb") as handle:
            digest = stream_sha256(handle)
        form = "wheel" if path.name.lower().endswith(".whl") else "sdist"
        return {"kind": "local source", "form": form, "sha256": digest}
    if path.is_dir():
        return {"kind": "local source", "form": "tree", "sha256": tree_sha256(path)}
    raise FileNotFoundError(f"the local source {path} does not exist")


def _looks_local(text: str, base: Path) -> bool:
    """pip's reading of a requirement as a path: it holds a path separator or
    starts with a dot, or it names an existing archive file."""
    if os.sep in text or (os.altsep and os.altsep in text) or text.startswith("."):
        return True
    return text.lower().endswith(_ARCHIVE_SUFFIXES) and (base / text).is_file()


def describe_source(spec: str, *, cwd: Optional[os.PathLike] = None) -> Dict[str, Any]:
    """A pip requirement, as the source receipt records what was requested.

    Never a filesystem path, a user name or a credential:

    - a VCS URL (``git+https://...@ref``): ``{"kind": "vcs", "vcs", "url", "ref"}``,
      the URL without user information, query or fragment (a ``subdirectory``
      fragment is kept as its own field); the commit it resolved to is added by
      the build, which is the one that knows it. A repository on this machine
      (``git+file://...``) is recorded as ``"url": None, "repository": "local"``,
      its location being a path;
    - a registry requirement (``torch-scatter``, ``torch-scatter==2.1.2``):
      ``{"kind": "registry", "name", "specifier"}``; the version it resolved to
      is the built wheel's;
    - a local file or directory (a path, or a ``file:`` URL): ``{"kind": "local
      source", "form": "sdist" | "wheel" | "tree", "sha256"}``, the file's or
      the tree's SHA-256 (`tree_sha256`), taken before anything is built from it;
    - any other URL (an archive on a web server): ``{"kind": "url", "url"}``,
      stripped like a VCS URL.

    A ``name @ url`` direct reference keeps its ``name``. A local source that
    does not exist raises `FileNotFoundError`; a string pip could not read as
    a requirement raises `ValueError`.
    """
    text = spec.strip()
    base = Path(cwd) if cwd is not None else Path.cwd()
    name = None
    direct = _DIRECT_REFERENCE.match(text)
    if direct and _SCHEME.match(direct.group(2)):
        name, text = direct.group(1), direct.group(2).strip()
    scheme_match = _SCHEME.match(text)
    scheme = scheme_match.group(1).lower() if scheme_match and len(scheme_match.group(1)) > 1 else ""
    vcs, _, transport = scheme.partition("+")
    if transport and vcs in _VCS_SCHEMES:
        parts = urllib.parse.urlsplit(text)
        location, ref = parts.path, None
        if "@" in location:
            location, _, ref = location.rpartition("@")
        described: Dict[str, Any] = {"kind": "vcs", "vcs": vcs}
        if transport == "file":
            # A repository on this machine: its location is a path, so only
            # that it is local is said; the commit the build resolved names it.
            described.update(url=None, repository="local")
        else:
            bare = urllib.parse.urlunsplit((transport, parts.netloc, location, "", ""))
            described["url"] = strip_credentials(bare)
        described["ref"] = ref or None
        fragment = urllib.parse.parse_qs(parts.fragment)
        if fragment.get("subdirectory"):
            described["subdirectory"] = fragment["subdirectory"][0]
    elif scheme == "file":
        local = Path(urllib.request.url2pathname(urllib.parse.urlsplit(text).path))
        described = _local_source(local)
    elif scheme:
        described = {"kind": "url", "url": strip_credentials(text)}
    elif _looks_local(text, base):
        described = _local_source(base / Path(text).expanduser())
    else:
        requirement = _REQUIREMENT.match(text)
        specifier = re.sub(r"\s+", "", requirement.group(2)) if requirement else ""
        if not requirement or not _SPECIFIER.match(specifier):
            raise ValueError(f"not a requirement pip reads: {spec!r}")
        described = {"kind": "registry", "name": requirement.group(1), "specifier": specifier}
    if name:
        described["name"] = name
    return described


# --- The source receipt -----------------------------------------------------
# One file in the virtual environment, outside every distribution, so that
# `uv sync --inexact` leaves it where it is (tests/unit/test_deploy_pyg_route.py
# runs uv to show it). Each install of the native set by the deploy layer
# replaces it whole. It is keyed by the installed native digests: a later
# install from anywhere else does not match it, and the mismatch is reported,
# never attributed to it.

RECEIPT_KIND = "shepherd-pyg-native-receipt"
RECEIPT_SCHEMA = 1
RECEIPT_LOCATION = ("share", "shepherd-advanced", "pyg-native-receipt.json")
ROUTE_LOCAL_BUILD = "local build"
ROUTE_PREBUILT_RELEASE = "prebuilt release"


class NotAVirtualEnvironment(RuntimeError):
    """The receipt is kept in a virtual environment only, never in a system
    Python's prefix."""


def receipt_path(prefix: Optional[os.PathLike] = None) -> Path:
    """Where the receipt of the environment at `prefix` (default: this
    interpreter's ``sys.prefix``) is kept."""
    return Path(prefix if prefix is not None else sys.prefix).joinpath(*RECEIPT_LOCATION)


def _installed_facts(native: Mapping[str, Optional[str]]) -> Dict[str, Any]:
    """A set of native libraries, recorded without their names."""
    known = {name: digest for name, digest in native.items() if digest is not None}
    return {"native_count": len(native),
            "native_aggregate_sha256": aggregate_sha256(known) if len(known) == len(native)
            else None,
            "native_sha256": sorted(known.values())}


def build_receipt(
    *,
    route: str,
    checks: Sequence[WheelCheck],
    requested: Mapping[str, Mapping[str, Any]],
    commits: Optional[Mapping[str, str]] = None,
    compiled_against: Mapping[str, Any],
    toolchain: Optional[Mapping[str, Any]],
    flags: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """The receipt of an install that passed `check_installed_against_wheels`.

    `requested` and `commits` are keyed by the wheel's file name. `requested`
    is `describe_source`'s description for a build, or the release's for the
    prebuilt pull; `commits` holds the commit a VCS source resolved to, where
    the build named it. `compiled_against`, `toolchain` and `flags` are the
    facts the route knows (None where it knows none: the prebuilt release
    carries no compiler facts). No version is judged here: what was resolved
    and what is installed are recorded as they are.
    """
    if not checks or not all(check.ok for check in checks):
        raise ValueError("a receipt is written only for an install that passed its check")
    commits = commits or {}
    wheels = []
    for check in checks:
        wheel = check.wheel
        if wheel.file not in requested:
            raise ValueError(f"no requested source is given for {wheel.file}")
        source = dict(requested[wheel.file])
        resolved: Dict[str, Any] = {"version": wheel.version}
        if source.get("kind") == "vcs":
            resolved["commit"] = commits.get(wheel.file)
        wheels.append({
            "distribution": wheel.name,
            "requested": source,
            "resolved": resolved,
            "installed": {"version": check.installed.version,
                          **_installed_facts(check.installed.native or {})},
            "wheel": {"file": wheel.file, "sha256": wheel.sha256},
        })
    return {
        "receipt": RECEIPT_KIND,
        "schema": RECEIPT_SCHEMA,
        "route": route,
        "compiled_against": dict(compiled_against),
        "toolchain": dict(toolchain) if toolchain is not None else None,
        "flags": dict(flags) if flags is not None else None,
        "wheels": sorted(wheels, key=lambda entry: normalised_name(entry["distribution"])),
    }


def write_receipt(receipt: Mapping[str, Any], prefix: Optional[os.PathLike] = None) -> Path:
    """Write `receipt` whole into the virtual environment at `prefix` (default:
    this interpreter's, which must be a virtual environment), replacing the
    one there."""
    from src.utils.file_modes import publish_bytes

    if prefix is None:
        if sys.prefix == sys.base_prefix:
            raise NotAVirtualEnvironment(
                "this interpreter is not a virtual environment; the receipt is kept in one only")
        prefix = sys.prefix
    target = receipt_path(prefix)
    target.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes(target, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return target


def read_receipt(prefix: Optional[os.PathLike] = None) -> Optional[Dict[str, Any]]:
    """The receipt kept at `prefix` (default: this interpreter's), or None when
    there is none. A file that is not a receipt of a known schema raises
    `ValueError`, with what it is not."""
    target = receipt_path(prefix)
    try:
        with target.open("rb") as handle:
            data = handle.read()
    except FileNotFoundError:
        return None
    receipt = json.loads(data.decode("utf-8"))
    if not isinstance(receipt, dict) or receipt.get("receipt") != RECEIPT_KIND:
        raise ValueError("the file is not a PyG native source receipt")
    if receipt.get("schema") != RECEIPT_SCHEMA or not isinstance(receipt.get("wheels"), list):
        raise ValueError(f"the receipt's schema {receipt.get('schema')!r} is not one this "
                         f"build reads")
    return receipt


RECEIPT_MATCHES = "matches"
RECEIPT_DIFFERS = "differs"


def receipt_against_installed(
    receipt: Mapping[str, Any], path: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Each distribution the receipt names, against what is installed now.

    Keyed by the installed native digests: an entry ``matches`` only when the
    installed version and the aggregate of the installed native libraries
    equal the receipt's; otherwise it ``differs``, with the reason. Information
    only: a difference never refuses anything.
    """
    states: Dict[str, Dict[str, Any]] = {}
    for entry in receipt.get("wheels", []):
        name = str(entry.get("distribution"))
        recorded = entry.get("installed") or {}
        installed = installed_identity(name, path)
        if installed.copies == 0:
            reason = "it is not installed now"
        elif installed.native is None:
            reason = "its installed file list (RECORD) cannot be read"
        else:
            facts = _installed_facts(installed.native)
            if installed.version != recorded.get("version"):
                reason = (f"installed version {installed.version}, the receipt's "
                          f"{recorded.get('version')}")
            elif facts["native_aggregate_sha256"] is None:
                reason = "an installed native library cannot be read"
            elif facts["native_aggregate_sha256"] != recorded.get("native_aggregate_sha256"):
                reason = "its installed native libraries are not the ones the receipt records"
            else:
                reason = None
        states[name] = ({"state": RECEIPT_MATCHES} if reason is None
                        else {"state": RECEIPT_DIFFERS, "reason": reason})
    return states


__all__ = [
    "ExtensionStatus",
    "check_pyg_native_extensions",
    "get_missing_extensions",
    "format_status_line",
    "get_fallback_warning",
    "log_pyg_native_status",
    "stream_sha256",
    "aggregate_sha256",
    "is_native_library",
    "normalised_name",
    "WheelIdentity",
    "wheel_identity",
    "distinct_locations",
    "InstalledIdentity",
    "installed_identity",
    "WheelCheck",
    "check_installed_against_wheels",
    "strip_credentials",
    "tree_sha256",
    "describe_source",
    "RECEIPT_KIND",
    "RECEIPT_SCHEMA",
    "RECEIPT_LOCATION",
    "ROUTE_LOCAL_BUILD",
    "ROUTE_PREBUILT_RELEASE",
    "NotAVirtualEnvironment",
    "receipt_path",
    "build_receipt",
    "write_receipt",
    "read_receipt",
    "RECEIPT_MATCHES",
    "RECEIPT_DIFFERS",
    "receipt_against_installed",
]
