"""
Runtime stack report — what is actually installed and usable.
=============================================================
Answers one question for the operator: *what is this process running on?*
Interpreter, platform, torch and its CUDA line, PyTorch Geometric, the PyG native
extensions, and the retrieval backends — with a severity so a caller can colour
the answer without re-deriving the rules.

Why it exists. A deployment can degrade in ways that produce no error and no
missing output: the GNN silently unavailable, native kernels silently replaced by
slower fallbacks, cuVS installed but unusable. Each is visible somewhere in a log,
and each has in practice been missed, because the signal was not proportionate to
the consequence. This turns those into one structured answer that the UI can show
continuously.

**This module must never raise.** It is the thing that reports "torch is broken",
so a broken torch has to come back as data, not as a traceback — otherwise it
disappears exactly when it is needed. Every probe is wrapped, and so is the
public entry point.

Severity levels:
  ok        everything present
  notice    optional or performance-only: native extensions on fallback kernels,
            cuVS present without cupy, a CUDA build with no visible device
  degraded  a capability is gone: torch or torch_geometric will not import, so
            GNN scoring cannot run at all

Scope, stated so the module is not mistaken for more than it is. This reports
**what is installed**. It does not yet check installed versions against what the
project requires, which is the other half of what this file was reserved for, and
the seven sites that inspect versions their own way — among them
``src/utils/pyg_native_check.py``, ``src/retrieval/backends/voyager_backend.py``,
``scripts/validate_installation.py`` and ``scripts/validate_pyg_ext.py`` — are
untouched. Consolidating them is separate work.

Deliberate boundary with ``pyg_native_check``: that module answers "do the
compiled extensions load and produce correct results here", a functional check.
This one reports presence and version, and delegates to it rather than
duplicating it.

Module: src/utils/version_checker.py
"""
from __future__ import annotations

import importlib
import platform
import sys
from typing import Any, Dict, List, Optional

from src.utils.pyg_native_check import check_pyg_native_extensions

# Optional retrieval-side packages. cuVS additionally needs cupy to construct a
# backend — importing cuvs alone has, in this project's history, been mistaken
# for "GPU vector search works".
_RETRIEVAL_PACKAGES = ("voyager", "cuvs", "cupy")

OK = "ok"
NOTICE = "notice"
DEGRADED = "degraded"

_cached: Optional[Dict[str, Any]] = None


def _probe_module(name: str) -> Dict[str, Any]:
    """Import a module and report the outcome. A broken build can raise things
    other than ImportError, so this catches broadly — any failure means the
    module is not usable, which is what the caller needs to know."""
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 — any failure means "not usable"
        return {"available": False, "version": None, "error": str(exc)}
    return {
        "available": True,
        "version": getattr(module, "__version__", None),
        "error": None,
    }


def _probe_torch() -> Dict[str, Any]:
    """torch, its CUDA build, and whether a device is actually visible.

    A CUDA-built torch that reports no device is worth surfacing: it usually
    means a driver or library problem rather than a deliberate CPU deployment.
    """
    info = _probe_module("torch")
    info.update({"cuda_build": None, "cuda_available": None, "device_name": None})
    if not info["available"]:
        return info
    try:
        import torch

        info["cuda_build"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # noqa: BLE001 — reporting must survive a broken CUDA stack
        info["error"] = f"torch imported but CUDA probe failed: {exc}"
    return info


def probe_runtime(force: bool = False) -> Dict[str, Any]:
    """Structured report of the runtime stack. Never raises.

    Cached after the first call — repeated UI polls should cost nothing.
    ``force=True`` re-probes, e.g. after a deliberate reinstall.
    """
    global _cached
    if _cached is not None and not force:
        return _cached

    try:
        report = _build_report(force=force)
    except Exception as exc:  # noqa: BLE001 — a reporter that dies reports nothing
        report = {
            "status": DEGRADED,
            "issues": [f"runtime probe failed: {exc}"],
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.machine()}",
            "torch": {"available": False, "version": None, "error": str(exc)},
            "torch_geometric": {"available": False, "version": None, "error": None},
            "pyg_native": {},
            "retrieval": {},
        }
    _cached = report
    return report


def _build_report(force: bool = False) -> Dict[str, Any]:
    torch_info = _probe_torch()
    pyg_info = _probe_module("torch_geometric")

    native = {
        name: {
            "available": status.available,
            "version": status.version,
            "error": status.error,
        }
        for name, status in check_pyg_native_extensions(force=force).items()
    }
    retrieval = {name: _probe_module(name) for name in _RETRIEVAL_PACKAGES}

    issues: List[str] = []
    status = OK

    if not torch_info["available"]:
        status = DEGRADED
        issues.append(f"torch is not importable: {torch_info['error']}")
    elif not pyg_info["available"]:
        status = DEGRADED
        issues.append("torch_geometric is not importable — GNN scoring cannot run")
    else:
        missing = [n for n, s in native.items() if not s["available"]]
        if missing:
            status = NOTICE
            issues.append(
                "PyG native extensions missing (" + ", ".join(missing) + ") — "
                "affected kernels fall back to slower implementations"
            )
        if torch_info["cuda_build"] and not torch_info["cuda_available"]:
            status = NOTICE
            issues.append(
                f"torch is built for CUDA {torch_info['cuda_build']} but no device is "
                "visible — check the driver and CUDA libraries"
            )
        if retrieval["cuvs"]["available"] and not retrieval["cupy"]["available"]:
            status = NOTICE
            issues.append(
                "cuVS is installed but cupy is missing — the GPU vector backend "
                "cannot be constructed (Voyager/CPU is unaffected)"
            )

    return {
        "status": status,
        "issues": issues,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "torch": torch_info,
        "torch_geometric": pyg_info,
        "pyg_native": native,
        "retrieval": retrieval,
    }


def format_runtime_line(report: Optional[Dict[str, Any]] = None) -> str:
    """One-line summary for a status bar, e.g.

        torch 2.10.0+cu130 · CUDA 13.0 · PyG 2.8.0 · pyg_lib 0.6.0 · Voyager

    Missing pieces are named rather than omitted — an absent entry reads as
    "not shown", a named one reads as "not there", and only the second is true.
    """
    report = report or probe_runtime()
    parts: List[str] = []

    torch_info = report.get("torch", {})
    if torch_info.get("available"):
        parts.append(f"torch {torch_info.get('version') or '?'}")
        if torch_info.get("cuda_available"):
            parts.append(f"CUDA {torch_info.get('cuda_build') or '?'}")
        elif torch_info.get("cuda_build"):
            parts.append(f"CUDA {torch_info['cuda_build']} (no device)")
        else:
            parts.append("CPU")
    else:
        parts.append("torch MISSING")

    pyg = report.get("torch_geometric", {})
    parts.append(f"PyG {pyg.get('version') or '?'}" if pyg.get("available") else "PyG MISSING")

    native = report.get("pyg_native", {})
    missing = [n for n, s in native.items() if not s.get("available")]
    if native and not missing:
        parts.append("native ext OK")
    elif missing:
        parts.append("native ext missing: " + ", ".join(missing))

    retrieval = report.get("retrieval", {})
    if retrieval.get("voyager", {}).get("available"):
        parts.append("Voyager")
    if retrieval.get("cuvs", {}).get("available"):
        parts.append("cuVS" if retrieval.get("cupy", {}).get("available") else "cuVS (no cupy)")

    return " · ".join(parts)


__all__ = ["probe_runtime", "format_runtime_line", "OK", "NOTICE", "DEGRADED"]
