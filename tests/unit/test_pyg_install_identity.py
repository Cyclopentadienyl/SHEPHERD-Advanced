"""
The PyG install identity: one digest, one install check, one source receipt.
============================================================================
`src/utils/pyg_native_check.py`'s deploy-time half (BACKLOG 70, 71;
research-comparison Stage 2a). The scripts that use it are run in
`test_deploy_pyg_route.py`; this pins the functions themselves:

- `stream_sha256` is SHA-256, chunked, from where the stream stands, and the
  aggregate is the specified text's digest, independent of the order given;
- the install check compares an installed distribution's version and native
  libraries with a wheel's, and says what differs; one directory reached by
  two path entries (a venv's ``lib64`` link) is one install, and a file that
  is not a wheel is said to be unreadable;
- `scripts/validate_pyg_ext.py` gives a receipt that cannot be written, and
  one not kept outside a virtual environment, statuses of their own, and
  records compiler version numbers, never their banners;
- a requested source is described with no path, no user name and no
  credential;
- the receipt is written into the virtual environment, read back, and keyed by
  the installed native digests;
- the module stays standard-library only, and the page-load reporter
  (`src/utils/version_checker.py`) never reaches the hashing half.

Module: tests/unit/test_pyg_install_identity.py
"""
from __future__ import annotations

import ast
import base64
import getpass
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from src.utils import pyg_native_check as native
from src.utils.pyg_native_check import (
    RECEIPT_DIFFERS,
    RECEIPT_MATCHES,
    NotAVirtualEnvironment,
    aggregate_sha256,
    build_receipt,
    check_installed_against_wheels,
    describe_source,
    installed_identity,
    is_native_library,
    read_receipt,
    receipt_against_installed,
    receipt_entry_state,
    receipt_path,
    stream_sha256,
    strip_credentials,
    tree_sha256,
    wheel_identity,
    write_receipt,
)

REPO = Path(__file__).resolve().parents[2]


def _wheel(directory: Path, name: str, version: str, natives: dict, *, extra=None) -> Path:
    """A wheel holding `natives` (member name -> bytes) and a dist-info."""
    files = {f"{name}/__init__.py": b"", **natives, **(extra or {})}
    dist = f"{name}-{version}.dist-info"
    files[f"{dist}/METADATA"] = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n".encode()
    files[f"{dist}/WHEEL"] = b"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: py3-none-any\n"

    def entry(path, data):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        return f"{path},sha256={digest},{len(data)}\n"

    files[f"{dist}/RECORD"] = ("".join(entry(p, d) for p, d in files.items())
                               + f"{dist}/RECORD,,\n").encode()
    path = directory / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for member, data in files.items():
            archive.writestr(member, data)
    return path


def _install(wheel: Path, site: Path) -> None:
    """Unpack a wheel into `site` as an installer would (purelib and platlib
    members moved to the root; RECORD rewritten to the installed names)."""
    site.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as archive:
        written = []
        for member in archive.namelist():
            head, _, rest = member.partition("/")
            if head.endswith(".data"):
                scheme, _, inner = rest.partition("/")
                if scheme not in ("purelib", "platlib"):
                    continue
                member_target = inner
            else:
                member_target = member
            target = site / member_target
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
            written.append(member_target)
    dist = next(p for p in written if p.endswith(".dist-info/METADATA")).split("/")[0]
    (site / dist / "RECORD").write_text("".join(f"{p},,\n" for p in written))


NATIVES = {"pkg_a/_ops.so": b"ops", "pkg_a/lib/libcore.so.1": b"core"}


# --- The digest -----------------------------------------------------------------

def test_the_digest_is_sha256_in_chunks_from_where_the_stream_stands():
    data = bytes(range(256)) * (8192 + 7)                       # over two chunks
    assert stream_sha256(io.BytesIO(data)) == hashlib.sha256(data).hexdigest()
    assert stream_sha256(io.BytesIO(b"")) == hashlib.sha256(b"").hexdigest()
    stream = io.BytesIO(b"headtail")
    stream.read(4)
    assert stream_sha256(stream) == hashlib.sha256(b"tail").hexdigest()


def test_the_aggregate_is_the_digest_of_its_specified_text():
    named = {"b/two.so": "2" * 64, "a/one.so": "1" * 64}
    text = f"a/one.so\t{'1' * 64}\nb/two.so\t{'2' * 64}\n"
    assert aggregate_sha256(named) == hashlib.sha256(text.encode()).hexdigest()
    assert aggregate_sha256(dict(reversed(list(named.items())))) == aggregate_sha256(named)
    assert aggregate_sha256({}) == hashlib.sha256(b"").hexdigest()
    with pytest.raises(ValueError):
        aggregate_sha256({"a\tb.so": "0" * 64})


@pytest.mark.parametrize("name, native", [
    ("torch_scatter/_scatter_cuda.so", True), ("pyg_lib/lib/libpyg.so", True),
    ("x/libgomp.so.1", True), ("x/libfoo.so.1.2", True), ("x/_ext.pyd", True),
    ("x/lib.dylib", True), ("x/sources.py", False), ("x/so_what.txt", False),
    ("x/lib.so.txt", False),
])
def test_a_native_library_is_known_by_its_name(name, native):
    assert is_native_library(name) is native


# --- The wheel and the install check ----------------------------------------------

def test_a_wheel_is_read_with_its_native_libraries_where_they_install(tmp_path):
    wheel = _wheel(tmp_path, "pkg_a", "1.0", NATIVES, extra={
        "pkg_a-1.0.data/platlib/pkg_a/_plat.so": b"plat",
        "pkg_a-1.0.data/scripts/helper.so": b"elsewhere",       # installed outside site
    })
    identity = wheel_identity(wheel)
    assert (identity.name, identity.version, identity.file) == ("pkg_a", "1.0", wheel.name)
    assert identity.sha256 == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert identity.native == {
        "pkg_a/_ops.so": hashlib.sha256(b"ops").hexdigest(),
        "pkg_a/lib/libcore.so.1": hashlib.sha256(b"core").hexdigest(),
        "pkg_a/_plat.so": hashlib.sha256(b"plat").hexdigest(),
    }


def test_the_install_check_passes_only_for_what_the_wheel_holds(tmp_path):
    (tmp_path / "built").mkdir()
    built = _wheel(tmp_path / "built", "pkg_a", "1.0", NATIVES)
    site = tmp_path / "site"
    _install(built, site)
    (check,) = check_installed_against_wheels([built], path=[str(site)])
    assert check.ok and check.problems == ()
    assert check.installed.copies == 1 and check.installed.version == "1.0"

    # Another build of the same version: one native library's bytes differ.
    (tmp_path / "other").mkdir()
    other = _wheel(tmp_path / "other", "pkg_a", "1.0", {**NATIVES, "pkg_a/_ops.so": b"ops2"})
    (check,) = check_installed_against_wheels([other], path=[str(site)])
    assert check.problems == ("native library pkg_a/_ops.so differs from the wheel's",)

    # Another version, and a native library the wheel does not hold.
    (tmp_path / "newer").mkdir()
    newer = _wheel(tmp_path / "newer", "pkg_a", "1.1", {"pkg_a/_ops.so": b"ops"})
    (check,) = check_installed_against_wheels([newer], path=[str(site)])
    assert check.problems == ("installed version 1.0, the wheel's 1.1",
                              "installed native library pkg_a/lib/libcore.so.1 is not in the wheel")


def test_the_install_check_names_a_missing_or_doubled_install(tmp_path):
    (tmp_path / "w").mkdir()
    wheel = _wheel(tmp_path / "w", "pkg_a", "1.0", NATIVES)
    (check,) = check_installed_against_wheels([wheel], path=[str(tmp_path / "empty")])
    assert check.problems == ("not installed",)

    _install(wheel, tmp_path / "one")
    _install(wheel, tmp_path / "two")
    (check,) = check_installed_against_wheels(
        [wheel], path=[str(tmp_path / "one"), str(tmp_path / "two")])
    assert check.problems == ("installed 2 times on the path",)

    # A listed library gone from disk: importlib.metadata leaves it out of the
    # distribution's files, so it reads as not installed.
    (library,) = [p for p in (tmp_path / "one").rglob("_ops.so")]
    library.unlink()
    (check,) = check_installed_against_wheels([wheel], path=[str(tmp_path / "one")])
    assert check.problems == ("native library pkg_a/_ops.so of the wheel is not installed",)

    checks = check_installed_against_wheels([wheel, wheel], path=[str(tmp_path / "two")])
    assert all(c.problems[0] == "more than one of the given wheels is this distribution"
               for c in checks)


def test_a_wheel_without_a_native_library_is_not_a_native_install(tmp_path):
    (tmp_path / "w").mkdir()
    wheel = _wheel(tmp_path / "w", "pkg_a", "1.0", {})
    _install(wheel, tmp_path / "site")
    (check,) = check_installed_against_wheels([wheel], path=[str(tmp_path / "site")])
    assert check.problems == ("the wheel holds no native library",)


def test_one_directory_reached_by_two_path_entries_is_one_install(tmp_path):
    """A virtual environment's ``lib64`` is a link to ``lib``, and where
    ``sys.platlibdir`` is ``lib64`` (the RHEL, Fedora and SUSE Pythons) `site`
    puts both names on ``sys.path``: one install, never "installed 2 times"."""
    (tmp_path / "w").mkdir()
    wheel = _wheel(tmp_path / "w", "pkg_a", "1.0", NATIVES)
    venv = tmp_path / "venv"
    site = venv / "lib" / "python3.12" / "site-packages"
    _install(wheel, site)
    (venv / "lib64").symlink_to("lib")
    both = [str(venv / "lib64" / "python3.12" / "site-packages"), str(site)]

    (check,) = check_installed_against_wheels([wheel], path=both)
    assert check.ok, check.problems
    assert check.installed.copies == 1
    receipt = build_receipt(route="local build", checks=[check],
                            requested={wheel.name: {"kind": "registry", "name": "pkg_a",
                                                    "specifier": ""}},
                            compiled_against={}, toolchain=None, flags=None)
    assert receipt_against_installed(receipt, path=both) == {"pkg_a": {"state": RECEIPT_MATCHES}}
    assert installed_identity("pkg_a", path=[str(site), str(site)]).copies == 1


def test_a_file_that_is_not_a_wheel_cannot_be_read_and_says_so(tmp_path):
    broken = tmp_path / "pkg_a-1.0-py3-none-any.whl"
    broken.write_bytes(b"not a zip archive")
    with pytest.raises(ValueError, match="pkg_a-1.0-py3-none-any.whl"):
        wheel_identity(broken)

    done = subprocess.run([sys.executable, str(REPO / "scripts" / "validate_pyg_ext.py"),
                           "--check-wheels", str(broken)],
                          capture_output=True, text=True, timeout=120, cwd=tmp_path)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "[FAIL] the wheels cannot be read" in done.stdout
    assert "Traceback" not in done.stderr


# --- The requested source, described --------------------------------------------

def test_a_registry_requirement_keeps_its_name_and_specifier():
    assert describe_source("torch-scatter") == {"kind": "registry", "name": "torch-scatter",
                                                "specifier": ""}
    assert describe_source("torch_scatter >= 2.1, <3 ; python_version > '3'") == {
        "kind": "registry", "name": "torch_scatter", "specifier": ">=2.1,<3"}
    with pytest.raises(ValueError):
        describe_source("torch-scatter at the newest")


def test_a_vcs_url_keeps_its_place_and_ref_without_who_fetched_it():
    assert describe_source("git+https://reader:t0ken@github.com/pyg-team/pyg-lib.git@0.6.0") == {
        "kind": "vcs", "vcs": "git", "url": "https://github.com/pyg-team/pyg-lib.git",
        "ref": "0.6.0"}
    assert describe_source(
        "pyg-lib @ git+ssh://git@github.com/pyg-team/pyg-lib.git@main"
        "#subdirectory=src&egg=pyg_lib") == {
        "kind": "vcs", "vcs": "git", "url": "ssh://github.com/pyg-team/pyg-lib.git",
        "ref": "main", "subdirectory": "src", "name": "pyg-lib"}
    assert describe_source("git+https://github.com/pyg-team/pyg-lib.git")["ref"] is None
    local = describe_source("git+file:///home/someone/src/pyg-lib@0.6.0")
    assert local == {"kind": "vcs", "vcs": "git", "url": None, "repository": "local",
                     "ref": "0.6.0"}


def test_other_urls_lose_their_credentials_query_and_fragment():
    assert describe_source("https://user:pw@files.example.org:8443/x/pkg.tar.gz?sig=abc#sha256=1") \
        == {"kind": "url", "url": "https://files.example.org:8443/x/pkg.tar.gz"}
    assert strip_credentials("https://u@[::1]:8080/p?q#f") == "https://[::1]:8080/p"


def test_a_local_source_is_its_digest_and_never_its_path(tmp_path):
    tree = tmp_path / "someone" / "torch_cluster-src"
    (tree / "csrc").mkdir(parents=True)
    (tree / "setup.py").write_text("setup()\n")
    (tree / "csrc" / "op.cpp").write_text("int f();\n")
    (tree / ".git").mkdir()
    (tree / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    expected = aggregate_sha256({
        "csrc/op.cpp": hashlib.sha256(b"int f();\n").hexdigest(),
        "setup.py": hashlib.sha256(b"setup()\n").hexdigest(),
    })
    assert tree_sha256(tree) == expected                        # .git left out
    described = describe_source(str(tree))
    assert described == {"kind": "local source", "form": "tree", "sha256": expected}
    assert describe_source("./someone/torch_cluster-src", cwd=tmp_path) == described

    sdist = tmp_path / "torch_scatter-2.1.2.tar.gz"
    sdist.write_bytes(b"an sdist")
    assert describe_source("torch_scatter-2.1.2.tar.gz", cwd=tmp_path) == {
        "kind": "local source", "form": "sdist", "sha256": hashlib.sha256(b"an sdist").hexdigest()}
    assert describe_source(sdist.as_uri())["sha256"] == hashlib.sha256(b"an sdist").hexdigest()
    with pytest.raises(FileNotFoundError):
        describe_source(str(tmp_path / "missing-tree"))


# --- The receipt -------------------------------------------------------------------

def _receipt_for(tmp_path, *, version="1.0", natives=NATIVES):
    wheels = tmp_path / "wheels"
    wheels.mkdir(exist_ok=True)
    wheel = _wheel(wheels, "pkg_a", version, natives)
    site = tmp_path / "venv" / "site"
    _install(wheel, site)
    checks = check_installed_against_wheels([wheel], path=[str(site)])
    receipt = build_receipt(
        route="local build", checks=checks,
        requested={wheel.name: describe_source(
            "git+https://reader:t0ken@github.com/org/pkg_a.git@v1")},
        commits={wheel.name: "a" * 40},
        compiled_against={"torch": "2.10.0+cu130", "cuda": "13.0", "basis": "test"},
        toolchain={"nvcc": "Cuda compilation tools, release 13.0, V13.0.88", "gcc": "gcc 13",
                   "g++": "g++ 13"},
        flags={"TORCH_CUDA_ARCH_LIST": "12.1", "FORCE_CUDA": "1", "MAX_JOBS": "8"})
    return receipt, wheel, site


def test_the_receipt_records_identity_without_a_path_a_user_or_a_credential(tmp_path):
    receipt, wheel, _ = _receipt_for(tmp_path)
    (entry,) = receipt["wheels"]
    assert entry["requested"]["url"] == "https://github.com/org/pkg_a.git"
    assert entry["resolved"] == {"version": "1.0", "commit": "a" * 40}
    assert entry["wheel"] == {"file": wheel.name,
                              "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}
    native_digests = {name: hashlib.sha256(data).hexdigest() for name, data in NATIVES.items()}
    assert entry["installed"] == {"version": "1.0", "native_count": 2,
                                  "native_aggregate_sha256": aggregate_sha256(native_digests),
                                  "native_sha256": sorted(native_digests.values())}
    text = json.dumps(receipt)
    for private in ("t0ken", "reader", str(tmp_path), str(Path.home()), "_ops.so", "libcore"):
        assert private not in text
    user = getpass.getuser()
    if len(user) > 3:
        assert user not in text


def test_a_receipt_is_written_only_for_a_checked_install_with_its_source(tmp_path):
    (tmp_path / "w").mkdir()
    wheel = _wheel(tmp_path / "w", "pkg_a", "1.0", NATIVES)
    failed = check_installed_against_wheels([wheel], path=[str(tmp_path / "nothing")])
    with pytest.raises(ValueError, match="passed its check"):
        build_receipt(route="local build", checks=failed, requested={wheel.name: {}},
                      compiled_against={}, toolchain=None, flags=None)
    _install(wheel, tmp_path / "site")
    passed = check_installed_against_wheels([wheel], path=[str(tmp_path / "site")])
    with pytest.raises(ValueError, match="no requested source"):
        build_receipt(route="local build", checks=passed, requested={}, compiled_against={},
                      toolchain=None, flags=None)


def test_the_receipt_is_kept_in_the_virtual_environment_and_read_back(tmp_path, monkeypatch):
    receipt, _, _ = _receipt_for(tmp_path)
    venv = tmp_path / "venv"
    assert read_receipt(venv) is None
    written = write_receipt(receipt, venv)
    assert written == receipt_path(venv) == venv / "share" / "shepherd-advanced" / \
        "pyg-native-receipt.json"
    assert read_receipt(venv) == receipt

    written.write_text(json.dumps({"receipt": "something else"}))
    with pytest.raises(ValueError, match="not a PyG native source receipt"):
        read_receipt(venv)
    written.write_text(json.dumps({**receipt, "schema": 99}))
    with pytest.raises(ValueError, match="schema 99"):
        read_receipt(venv)
    # NaN or infinity is no value a receipt records, and a run environment
    # that copied one would stop its record's write (research-comparison 2b):
    # written as such, or as a number too large for a float.
    for constant in ("NaN", "Infinity", "-Infinity", "1e400", "-1.5e999"):
        written.write_text(json.dumps(receipt)[:-1] + f', "x": {constant}}}')
        with pytest.raises(ValueError, match=f"holds {constant}"):
            read_receipt(venv)
    written.write_text(json.dumps(receipt)[:-1] + ', "x": 1.5e300, "y": 12}')
    assert read_receipt(venv) == {**receipt, "x": 1.5e300, "y": 12}

    monkeypatch.setattr(sys, "base_prefix", sys.prefix)
    with pytest.raises(NotAVirtualEnvironment):
        write_receipt(receipt)


def test_the_receipt_is_keyed_by_the_installed_native_digests(tmp_path):
    receipt, _, site = _receipt_for(tmp_path)
    assert receipt_against_installed(receipt, path=[str(site)]) == {
        "pkg_a": {"state": RECEIPT_MATCHES}}

    (site / "pkg_a" / "_ops.so").write_bytes(b"another build")
    assert receipt_against_installed(receipt, path=[str(site)]) == {
        "pkg_a": {"state": RECEIPT_DIFFERS, "reason": "its installed native libraries are not "
                                                      "the ones the receipt records"}}
    assert receipt_against_installed(receipt, path=[str(tmp_path / "nothing")]) == {
        "pkg_a": {"state": RECEIPT_DIFFERS, "reason": "it is not installed now"}}


def test_one_entry_is_compared_by_the_one_comparison_with_an_identity_already_read(tmp_path):
    """`receipt_entry_state` is what `receipt_against_installed` does per
    entry; the run environment collector gives it the identity it has read,
    so nothing is hashed twice."""
    receipt, _, site = _receipt_for(tmp_path)
    (entry,) = receipt["wheels"]
    installed = installed_identity("pkg_a", path=[str(site)])

    assert receipt_entry_state(entry, installed) == {"state": RECEIPT_MATCHES}
    assert receipt_entry_state({**entry, "installed": {**entry["installed"], "version": "0.9"}},
                               installed) == {
        "state": RECEIPT_DIFFERS, "reason": "installed version 1.0, the receipt's 0.9"}
    assert receipt_entry_state({**entry, "installed": "not an object"}, installed)[
        "state"] == RECEIPT_DIFFERS


def test_the_installed_identity_reads_the_record_of_the_first_copy(tmp_path):
    (tmp_path / "w").mkdir()
    _install(_wheel(tmp_path / "w", "pkg_a", "1.0", NATIVES), tmp_path / "site")
    identity = installed_identity("PKG-A", path=[str(tmp_path / "site")])   # PEP 503 name
    assert identity.copies == 1 and identity.version == "1.0"
    assert identity.native == {name: hashlib.sha256(data).hexdigest()
                               for name, data in NATIVES.items()}


# --- What may reach it --------------------------------------------------------------

def test_the_module_imports_the_standard_library_only():
    tree = ast.parse(Path(native.__file__).read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            roots.add(node.module.split(".")[0])
    # src.utils.file_modes, for the receipt's whole-file write, is itself stdlib only.
    assert roots - {"src"} <= set(sys.stdlib_module_names), roots
    assert all(name == "src.utils.file_modes" for name in _src_imports(tree))


def _src_imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src"):
            yield node.module


def test_the_page_load_reporter_never_reaches_the_hashing_half():
    """version_checker runs on page loads: it uses the import probe only."""
    hashing = {"stream_sha256", "aggregate_sha256", "wheel_identity", "installed_identity",
               "check_installed_against_wheels", "tree_sha256", "describe_source",
               "read_receipt", "receipt_against_installed", "build_receipt", "write_receipt",
               "installed_native_facts", "receipt_entry_state"}
    tree = ast.parse((REPO / "src" / "utils" / "version_checker.py").read_text(encoding="utf-8"))
    named = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    named |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    named |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
              for alias in node.names}
    assert not named & hashing
    assert "hashlib" not in named


@pytest.fixture
def validator():
    """`scripts/validate_pyg_ext.py`, as a module."""
    spec = importlib.util.spec_from_file_location(
        "validate_pyg_ext_under_test", REPO / "scripts" / "validate_pyg_ext.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tools(directory: Path, scripts: dict) -> None:
    directory.mkdir()
    for name, body in scripts.items():
        tool = directory / name
        tool.write_text("#!/bin/sh\n" + body)
        tool.chmod(0o755)


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in compilers are sh scripts")
def test_the_receipt_records_compiler_version_numbers_never_their_banners(
        tmp_path, monkeypatch, validator):
    """A banner is free text: gcc's configured package version or a vendor's
    line can carry a path or a person's name, so only the number is kept."""
    _tools(tmp_path / "bin", {
        "gcc": 'case "$1" in -dumpfullversion) echo 13.3.0 ;;\n'
               '  *) echo "gcc (built by jdoe in /home/jdoe/gcc) 13.3.0" ;; esac\n',
        # A gcc before 7: no -dumpfullversion, a full -dumpversion.
        "g++": 'case "$1" in -dumpfullversion) exit 1 ;; -dumpversion) echo 6.5.0 ;;\n'
               '  *) echo "g++ (jdoe) 6.5.0" ;; esac\n',
        "nvcc": 'echo "nvcc: NVIDIA (R) Cuda compiler driver"\n'
                'echo "Built on Mon_Jan_01 by jdoe at /home/jdoe/cuda"\n'
                'echo "Cuda compilation tools, release 12.8, V12.8.93"\n',
        "cc": 'echo "jdoe special 13"\n',
    })
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    assert validator._tool_version("gcc") == "13.3.0"
    assert validator._tool_version("g++") == "6.5.0"
    assert validator._tool_version("nvcc") == "12.8.93"
    assert validator._tool_version("cc") is None             # states no bare number
    assert validator._tool_version("clang-not-here") is None


def _checked_install(tmp_path, monkeypatch):
    (tmp_path / "w").mkdir()
    wheel = _wheel(tmp_path / "w", "pkg_a", "1.0", NATIVES)
    _install(wheel, tmp_path / "site")
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    return ["--check-wheels", str(wheel), "--write-receipt", "prebuilt release",
            "--release-tag", "t", "--release-asset", "a.tar.gz", "--release-asset-sha256", "0" * 64]


def test_a_receipt_that_cannot_be_written_has_its_own_status(tmp_path, monkeypatch, capsys,
                                                             validator):
    """Not the check's status: the install passed its check, and the caller
    must say that only the receipt is missing."""
    argv = _checked_install(tmp_path, monkeypatch)
    (tmp_path / "venv" / "share").mkdir(parents=True)
    (tmp_path / "venv" / "share" / "shepherd-advanced").write_text("a file in the way")

    assert validator.main(argv) == validator.EXIT_RECEIPT_NOT_WRITTEN == 3
    out = capsys.readouterr().out
    assert "1 of 1 installed exactly as the wheels hold them" in out
    assert "[FAIL] the source receipt could not be written" in out


def test_no_receipt_outside_a_virtual_environment_has_its_own_status(tmp_path, monkeypatch,
                                                                     capsys, validator):
    argv = _checked_install(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "base_prefix", sys.prefix)

    assert validator.main(argv) == validator.EXIT_NO_RECEIPT_KEPT == 4
    assert "no source receipt written" in capsys.readouterr().out
    assert read_receipt(tmp_path / "venv") is None
    monkeypatch.setattr(sys, "base_prefix", "/somewhere/else")
    assert validator.main(argv) == 0
    assert read_receipt(tmp_path / "venv")["route"] == "prebuilt release"


def test_the_install_check_imports_no_torch(tmp_path):
    (tmp_path / "w").mkdir()
    wheel = _wheel(tmp_path / "w", "pkg_a", "1.0", NATIVES)
    probe = (
        "import runpy, sys\n"
        f"sys.argv = ['validate_pyg_ext.py', '--check-wheels', {str(wheel)!r}]\n"
        "try:\n"
        f"    runpy.run_path({str(REPO / 'scripts' / 'validate_pyg_ext.py')!r}, run_name='__main__')\n"
        "except SystemExit as exit:\n"
        "    print('exit', exit.code)\n"
        "print('torch imported' if 'torch' in sys.modules else 'no torch')\n")
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                          timeout=120, cwd=tmp_path)
    assert "[FAIL] pkg_a 1.0: not installed" in done.stdout, done.stdout + done.stderr
    assert done.stdout.strip().splitlines()[-2:] == ["exit 1", "no torch"]
