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
#   Pure import probing — no torch tensors created, safe to call anywhere
#   (including processes without a GPU). Results are cached after first probe.
# ==============================================================================
"""
from __future__ import annotations

import importlib
import logging
from typing import Dict, List, NamedTuple, Optional

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


__all__ = [
    "ExtensionStatus",
    "check_pyg_native_extensions",
    "get_missing_extensions",
    "format_status_line",
    "get_fallback_warning",
    "log_pyg_native_status",
]
