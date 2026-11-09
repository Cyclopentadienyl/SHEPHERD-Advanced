#!/usr/bin/env python3
"""Validation of the PyG native extensions: what is installed, and whether it works.

    .venv/bin/python scripts/validate_pyg_ext.py

Functional validation, as before: import success only proves the shared
libraries load, so this also runs each package's compiled kernel (on CUDA when
available) and checks the result is numerically correct. Exits non-zero if any
critical check fails, so it is safe to use in CI / deploy. It ends with the
source receipt's state (below), which is information and never changes the exit
status.

    .venv/bin/python scripts/validate_pyg_ext.py --check-wheels W.whl [W.whl ...]

The install check (BACKLOG 70; research-comparison Stage 2): each named wheel
against the distribution installed under its name. The installed version and
every installed native library's SHA-256 must equal the wheel's. It imports no
torch. The ARM source build (`scripts/build_pyg_arm.sh`) and `deploy.sh`'s
prebuilt pull run it on exactly the wheels they installed, and with
``--write-receipt`` it then writes the path-free source receipt into the
virtual environment (BACKLOG 71). The Linux index route and `deploy.cmd` do not
run it. Its exit status, so that a caller can say which happened: 0 installed
exactly as the wheels hold them (and the receipt written, when asked); 1 the
check failed, or the wheels cannot be read; 2 a usage error; 3 the check
passed, but the receipt could not be written; 4 the check passed, and no
receipt is kept, this interpreter not being a virtual environment.

    .venv/bin/python scripts/validate_pyg_ext.py --receipt

The source receipt against what is installed now: per distribution, whether the
installed version and native libraries are the ones it records. Information
only: exit 0 whatever it says.

    .venv/bin/python scripts/validate_pyg_ext.py --describe-source SPEC

The builder's use: prints the path-free description of a pip requirement, taken
before anything is built from it.

The digest, the check, the source description and the receipt are
`src/utils/pyg_native_check.py`'s; nothing here hashes or compares on its own.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.pyg_native_check import (  # noqa: E402 - after the path is set
    RECEIPT_LOCATION,
    RECEIPT_MATCHES,
    ROUTE_LOCAL_BUILD,
    ROUTE_PREBUILT_RELEASE,
    NotAVirtualEnvironment,
    build_receipt,
    check_installed_against_wheels,
    describe_source,
    read_receipt,
    receipt_against_installed,
    write_receipt,
)

#: The build flags a receipt of a local build records, as exported to the build.
BUILD_FLAGS = ("TORCH_CUDA_ARCH_LIST", "FORCE_CUDA", "MAX_JOBS")
RECEIPT_WHERE = "/".join(RECEIPT_LOCATION)

#: --check-wheels' exit statuses (argparse's usage error is 2); see the module
#: docstring. The deploy scripts name each outcome by them.
EXIT_CHECK_FAILED = 1
EXIT_RECEIPT_NOT_WRITTEN = 3
EXIT_NO_RECEIPT_KEPT = 4

#: A version number, as a compiler states it on its own (``13.3.0``).
_VERSION_NUMBER = re.compile(r"\d+(?:\.\d+)*")
#: nvcc's statement of its version: ``release 13.0, V13.0.88``.
_NVCC_RELEASE = re.compile(r"\brelease (\d+(?:\.\d+)+)(?:, V(\d+(?:\.\d+)+))?")


# --- The functional kernel checks --------------------------------------------

def _kernel_checks() -> int:
    import torch

    results = []

    def check(name, fn):
        try:
            fn()
            results.append((name, True))
            print(f"  [OK]   {name}")
        except Exception as e:  # noqa: BLE001 - report every failure, never abort early
            results.append((name, False))
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()} | testing on: {dev}")
    if dev == "cpu":
        print("  [WARN] CUDA unavailable; exercising CPU kernels only (GPU kernels not verified).")

    def t_scatter():
        from torch_scatter import scatter

        src = torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)
        idx = torch.tensor([0, 0, 1, 1], device=dev)
        out = scatter(src, idx, dim=0, reduce="sum")
        expected = torch.tensor([3.0, 7.0], device=dev)
        assert torch.allclose(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"

    def t_sparse():
        from torch_sparse import SparseTensor

        # Dense form: [[0, 2], [3, 4]]
        row = torch.tensor([0, 1, 1], device=dev)
        col = torch.tensor([1, 0, 1], device=dev)
        val = torch.tensor([2.0, 3.0, 4.0], device=dev)
        a = SparseTensor(row=row, col=col, value=val, sparse_sizes=(2, 2))
        x = torch.tensor([[1.0], [1.0]], device=dev)
        out = a.matmul(x)  # -> [[2], [7]]
        expected = torch.tensor([[2.0], [7.0]], device=dev)
        assert torch.allclose(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"

    def t_cluster():
        from torch_cluster import knn_graph

        x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0]], device=dev)
        edge_index = knn_graph(x, k=1)
        assert edge_index.dim() == 2 and edge_index.size(0) == 2, f"bad shape {tuple(edge_index.shape)}"
        assert edge_index.size(1) == x.size(0), f"expected {x.size(0)} edges, got {edge_index.size(1)}"

    def t_pyglib():
        import pyg_lib  # importing triggers load_library('libpyg') -> the ABI fix under test

        _ = pyg_lib.__version__
        # Bonus: exercise a compiled kernel. API/device quirks here are non-critical
        # since the import above already proves libpyg.so resolves against torch.
        try:
            inputs = torch.randn(8, 16, device=dev)
            ptr = torch.tensor([0, 4, 8], device=dev)
            other = torch.randn(2, 16, 32, device=dev)
            out = pyg_lib.ops.segment_matmul(inputs, ptr, other)
            assert tuple(out.shape) == (8, 32), f"bad shape {tuple(out.shape)}"
            print("         (segment_matmul kernel OK)")
        except Exception as e:  # noqa: BLE001
            print(f"         (note: segment_matmul kernel not verified: {type(e).__name__}: {e})")

    def t_pyg_integration():
        from torch_geometric.nn import SAGEConv

        conv = SAGEConv(16, 32).to(dev)
        x = torch.randn(10, 16, device=dev)
        edge_index = torch.randint(0, 10, (2, 30), device=dev)
        out = conv(x, edge_index)
        assert tuple(out.shape) == (10, 32), f"bad shape {tuple(out.shape)}"

    print("\nKernel / correctness checks:")
    check("torch_scatter  scatter-sum", t_scatter)
    check("torch_sparse   spmm (A @ x)", t_sparse)
    check("torch_cluster  knn_graph", t_cluster)
    check("pyg_lib        import + libpyg.so load", t_pyglib)
    check("torch_geometric SAGEConv forward (uses the ext)", t_pyg_integration)

    n_fail = sum(1 for _, ok in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed.")
    _show_receipt()
    if n_fail:
        print("Some checks FAILED — do not package these wheels until resolved.")
        return 1
    print("All PyG native extensions are functional.")
    return 0


# --- The source receipt, shown -----------------------------------------------

def _show_receipt() -> None:
    """The receipt's state against what is installed now. Never fails."""
    print(f"\nSource receipt ({RECEIPT_WHERE} in this environment):")
    try:
        receipt = read_receipt()
    except (OSError, ValueError) as exc:
        print(f"  [WARN] the receipt cannot be read: {exc}")
        return
    if receipt is None:
        print("  none: this environment records no install by the deploy layer's checked routes")
        print("  (the ARM source build and the prebuilt pull, which keep receipts since BACKLOG 70);")
        print("  the source of what is installed is not known.")
        return
    print(f"  route: {receipt.get('route')}")
    for name, state in receipt_against_installed(receipt).items():
        entry = next((w for w in receipt["wheels"] if w.get("distribution") == name), {})
        version = (entry.get("installed") or {}).get("version")
        if state["state"] == RECEIPT_MATCHES:
            print(f"  [OK]     {name} {version}: installed as the receipt records "
                  f"({_requested_words(entry.get('requested') or {})})")
        else:
            print(f"  [DIFFERS] {name}: {state['reason']}; the receipt does not describe "
                  f"what is installed")


def _requested_words(requested: Dict[str, Any]) -> str:
    kind = requested.get("kind")
    if kind == "vcs":
        return f"{requested.get('url') or 'a local repository'} at {requested.get('ref') or 'its default branch'}"
    if kind == "registry":
        return f"{requested.get('name')}{requested.get('specifier') or ''} from a package index"
    if kind == "local source":
        return f"a local {requested.get('form')} {str(requested.get('sha256'))[:12]}"
    if kind == ROUTE_PREBUILT_RELEASE:
        return f"release {requested.get('tag')}"
    return str(kind)


# --- The install check and the receipt ----------------------------------------

def _tool_output(argv: List[str]) -> Optional[str]:
    """What a command prints on success, or None."""
    try:
        done = subprocess.run(argv, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.decode("utf-8", "replace") if done.returncode == 0 else None


def _tool_version(command: str) -> Optional[str]:
    """A compiler's version number, or None when it states none.

    Only the number is recorded, never a banner: a banner is free text (gcc's
    configured package version, a vendor's or a build host's line) that can
    carry a path or a person's name. gcc and g++ state it alone with
    ``-dumpfullversion`` (``-dumpversion`` before gcc 7); nvcc's ``--version``
    states ``release X.Y, VX.Y.Z``, of which the full ``X.Y.Z`` is kept."""
    if command == "nvcc":
        stated = _NVCC_RELEASE.search(_tool_output([command, "--version"]) or "")
        return (stated.group(2) or stated.group(1)) if stated else None
    for flag in ("-dumpfullversion", "-dumpversion"):
        number = (_tool_output([command, flag]) or "").strip()
        if _VERSION_NUMBER.fullmatch(number):
            return number
    return None


def _pairs(values: Optional[List[List[str]]]) -> Dict[str, str]:
    return {Path(wheel).name: value for wheel, value in (values or [])}


def _check_wheels(args: argparse.Namespace) -> int:
    wheels = [Path(w) for w in args.check_wheels]
    try:
        checks = check_installed_against_wheels(wheels)
    except (OSError, ValueError) as exc:
        print(f"  [FAIL] the wheels cannot be read: {exc}")
        return EXIT_CHECK_FAILED
    print("Install check (each installed distribution against the wheel meant to be installed):")
    for check in checks:
        wheel = check.wheel
        if check.ok:
            print(f"  [OK]   {wheel.name} {wheel.version}: the installed version and its "
                  f"{len(wheel.native)} native librar{'y' if len(wheel.native) == 1 else 'ies'} "
                  f"equal the wheel's")
        else:
            print(f"  [FAIL] {wheel.name} {wheel.version}: " + "; ".join(check.problems))
    failed = [c for c in checks if not c.ok]
    print(f"{len(checks) - len(failed)} of {len(checks)} installed exactly as the wheels hold them.")
    if failed:
        print("The installed native extensions are NOT the ones this step meant to install.")
        return EXIT_CHECK_FAILED
    if args.write_receipt is None:
        return 0
    return _write_receipt(args, checks)


def _write_receipt(args: argparse.Namespace, checks) -> int:
    if args.write_receipt == ROUTE_LOCAL_BUILD:
        requested = {name: json.loads(text) for name, text in _pairs(args.source).items()}
        toolchain = {tool: _tool_version(tool) for tool in ("nvcc", "gcc", "g++")}
        flags = {flag: os.environ.get(flag) for flag in BUILD_FLAGS}
        basis = "the build's target environment"
    else:
        release = {"kind": ROUTE_PREBUILT_RELEASE, "tag": args.release_tag,
                   "asset": args.release_asset, "asset_sha256": args.release_asset_sha256}
        requested = {check.wheel.file: release for check in checks}
        toolchain = flags = None
        basis = "the release's declared target"
    try:
        receipt = build_receipt(
            route=args.write_receipt, checks=checks, requested=requested,
            commits=_pairs(args.resolved_commit),
            compiled_against={"torch": args.compiled_against_torch,
                              "cuda": args.compiled_against_cuda, "basis": basis},
            toolchain=toolchain, flags=flags)
        write_receipt(receipt)
    except NotAVirtualEnvironment as exc:
        print(f"  [WARN] no source receipt written: {exc}")
        return EXIT_NO_RECEIPT_KEPT
    except (OSError, ValueError) as exc:
        print(f"  [FAIL] the source receipt could not be written: {exc}")
        return EXIT_RECEIPT_NOT_WRITTEN
    print(f"Source receipt written into this virtual environment ({RECEIPT_WHERE}).")
    return 0


def _describe(spec: str) -> int:
    try:
        described = describe_source(spec)
    except (OSError, ValueError) as exc:
        print(f"the source {spec!r} cannot be described: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(described, sort_keys=True))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-wheels", nargs="+", metavar="WHEEL",
                      help="check the installed distributions against these wheels")
    mode.add_argument("--receipt", action="store_true",
                      help="show the source receipt against what is installed now")
    mode.add_argument("--describe-source", metavar="SPEC",
                      help="print a pip requirement's path-free description")
    parser.add_argument("--write-receipt", choices=(ROUTE_LOCAL_BUILD, ROUTE_PREBUILT_RELEASE),
                        help="after a passing --check-wheels, write the source receipt")
    parser.add_argument("--source", nargs=2, action="append", metavar=("WHEEL", "JSON"),
                        help="a local build's requested source for WHEEL (--describe-source)")
    parser.add_argument("--resolved-commit", nargs=2, action="append", metavar=("WHEEL", "SHA"),
                        help="the commit a VCS source of WHEEL resolved to")
    parser.add_argument("--compiled-against-torch")
    parser.add_argument("--compiled-against-cuda")
    parser.add_argument("--release-tag")
    parser.add_argument("--release-asset")
    parser.add_argument("--release-asset-sha256")
    args = parser.parse_args(argv)
    if args.write_receipt and not args.check_wheels:
        parser.error("--write-receipt goes with --check-wheels")
    if args.write_receipt == ROUTE_PREBUILT_RELEASE and not (
            args.release_tag and args.release_asset and args.release_asset_sha256):
        parser.error("a prebuilt release's receipt needs its tag, asset and asset SHA-256")
    if args.check_wheels:
        return _check_wheels(args)
    if args.receipt:
        _show_receipt()
        return 0
    if args.describe_source is not None:
        return _describe(args.describe_source)
    return _kernel_checks()


if __name__ == "__main__":
    sys.exit(main())
