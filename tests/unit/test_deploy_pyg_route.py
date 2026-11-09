"""
The ARM + CUDA PyG route of `deploy.sh` and `scripts/build_pyg_arm.sh`, run.
============================================================================
BACKLOG 70 and the deploy half of 71 (research-comparison Stage 2a; the owner's
O5 (ii)). The scripts run for real, through `subprocess`, in a temporary
directory, with stand-ins on `PATH` for what this machine cannot be or must not
touch:

- `uname -m` answers `aarch64`;
- `uv` installs a wheel by unpacking it into the temporary virtual
  environment (and can be told to fail, to install a wrong byte, or to install
  a package that does not import);
- `curl` serves a release tarball made here, or fails like a timed-out GitHub;
- the target virtual environment is a real one (`python -m venv`), holding a
  fake `torch` (2.10.0+cu130 with one CUDA device, or another version for the
  host that does not match the release), a fake `torch_geometric`, and a fake
  `pip` whose `pip wheel` writes a small wheel with native libraries and logs
  the arguments it was given;
- `nvcc`, `gcc`, `g++`, `cmake` and `ninja` answer `--version`.

What is real: both scripts, `sha256sum`, `tar`, the install check and the
receipt (`scripts/validate_pyg_ext.py` over `src/utils/pyg_native_check.py`),
and, in the last test, `uv` itself. **This is software simulation.** Nothing
here runs on an ARM host or a GB10: the real route (download from GitHub, the
CUDA build, the kernels) is verified only by the GB10 acceptance, and is
listed apart from these tests.

Module: tests/unit/test_deploy_pyg_route.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pytest

REPO = Path(__file__).resolve().parents[2]
PY_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"
#: The builder's own check for Python's headers (scripts/build_pyg_arm.sh).
PYTHON_HEADER = Path(f"/usr/include/python{sys.version_info.major}.{sys.version_info.minor}/Python.h")
needs_header = pytest.mark.skipif(
    not PYTHON_HEADER.is_file(),
    reason=f"the builder checks {PYTHON_HEADER}, which this machine lacks")
pytestmark = pytest.mark.skipif(os.name != "posix" or shutil.which("bash") is None,
                                reason="the deploy scripts are bash")

NATIVE = ("pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster")
RELEASE_VERSIONS = {"pyg_lib": "0.6.0", "torch_scatter": "2.1.2", "torch_sparse": "0.6.18",
                    "torch_cluster": "1.6.3"}
FAKE_COMMIT = "c0ffee" * 6 + "c0ff"

# --- The rig's files ----------------------------------------------------------

WHEEL_MAKER = r'''
import base64, hashlib, os, zipfile

def make_wheel(out, name, version, salt="", tag="cp312-cp312-linux_aarch64"):
    """A wheel of `name` with two native libraries whose bytes depend on salt."""
    files = {
        f"{name}/__init__.py": f"__version__ = {version!r}\n".encode(),
        f"{name}/_native.so": f"native {name} {version} {salt}".encode(),
        f"{name}/lib/libextra.so.1": f"extra {name} {version} {salt}".encode(),
    }
    dist = f"{name}-{version}.dist-info"
    files[f"{dist}/METADATA"] = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n".encode()
    files[f"{dist}/WHEEL"] = (b"Wheel-Version: 1.0\nGenerator: rig\nRoot-Is-Purelib: false\n"
                              + f"Tag: {tag}\n".encode())
    def entry(path, data):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        return f"{path},sha256={digest},{len(data)}\n"
    record = "".join(entry(p, d) for p, d in files.items()) + f"{dist}/RECORD,,\n"
    files[f"{dist}/RECORD"] = record.encode()
    path = os.path.join(out, f"{name}-{version}-{tag}.whl")
    with zipfile.ZipFile(path, "w") as archive:
        for member, data in files.items():
            archive.writestr(member, data)
    return path
'''

FAKE_PIP = r'''
import json, os, re, sys
sys.path.insert(0, os.environ["RIG_ROOT"])
from rig_wheels import make_wheel

args = sys.argv[1:]
with open(os.environ["RIG_LOG"], "a") as log:
    log.write("pip " + json.dumps(args) + "\n")
if args[:1] != ["wheel"]:
    sys.exit(2)
spec, out = args[1], args[args.index("-w") + 1]
tail = spec.rstrip("/").rsplit("/", 1)[-1]
name = next(v for k, v in (("scatter", "torch_scatter"), ("sparse", "torch_sparse"),
                           ("cluster", "torch_cluster"), ("pyg", "pyg_lib")) if k in tail)
defaults = {"torch_scatter": "2.1.2", "torch_sparse": "0.6.18", "torch_cluster": "1.6.3",
            "pyg_lib": "0.6.0"}
found = re.search(r"==([0-9][A-Za-z0-9.+]*)", spec) or re.search(r"@([0-9][0-9.]*)$", spec)
version = found.group(1) if found else defaults[name]
if spec.startswith("git+"):
    print(f"  Resolved {spec[4:].rpartition('@')[0]} to commit " + "c0ffee" * 6 + "c0ff")
if name in os.environ.get("RIG_PIP_FAIL", "").split(","):
    print("rig pip: the build failed")
    sys.exit(1)
print("Saved " + make_wheel(out, name, version, os.environ.get("RIG_BUILD_SALT", "")))
'''

FAKE_TORCH = r'''
import os
__version__ = os.environ.get("RIG_TORCH_VERSION", "2.10.0+cu130")

class _Version:
    cuda = "13.0"
    git_version = "rig"

version = _Version()

class _Tensor:
    device = "cuda:0"

def zeros(*shape, **kwargs):
    return _Tensor()

class cuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 1

    @staticmethod
    def get_device_capability(index=0):
        return (12, 1)

    @staticmethod
    def get_device_name(index=0):
        return "rig GPU"
'''

RIG_INSTALL = r'''
import os, pathlib, shutil, sys, sysconfig, zipfile

reinstall = sys.argv[1] == "1"
purelib = pathlib.Path(sysconfig.get_paths()["purelib"])
for wheel in sys.argv[2:]:
    with zipfile.ZipFile(wheel) as archive:
        dist = next(m.split("/")[0] for m in archive.namelist()
                    if m.endswith(".dist-info/METADATA"))
        name = dist.split("-")[0]
        existing = list(purelib.glob(f"{name}-*.dist-info"))
        if existing and not reinstall:
            print(f"rig uv: {name} already installed; kept")
            continue
        for old in existing:
            for line in (old / "RECORD").read_text().splitlines():
                target = purelib / line.split(",")[0]
                if target.is_file():
                    target.unlink()
            shutil.rmtree(old)
        archive.extractall(purelib)
    if name == os.environ.get("RIG_UV_TAMPER"):
        library = purelib / name / "_native.so"
        library.write_bytes(library.read_bytes() + b" and something else")
    if name == os.environ.get("RIG_UV_BREAK_IMPORT"):
        # Not a native library: the byte check passes, the import does not.
        with open(purelib / name / "__init__.py", "a") as init:
            init.write("raise ImportError('rig: undefined symbol')\n")
    print(f"rig uv: installed {pathlib.Path(wheel).name}")
'''

STUB_UV = r'''#!/bin/bash
echo "uv $*" >> "$RIG_LOG"
case "$1" in
    --version) echo "uv 0.0.0 (rig)"; exit 0 ;;
    sync) exit 0 ;;
    pip) ;;
    *) exit 1 ;;
esac
shift
[ "${1:-}" = install ] || exit 2
shift
py=""; reinstall=0; wheels=()
while [ $# -gt 0 ]; do
    case "$1" in
        --python) py="$2"; shift 2 ;;
        --reinstall) reinstall=1; shift ;;
        --no-deps) shift ;;
        *.whl) wheels+=("$1"); shift ;;
        *) echo "rig uv: not served here: $1" >&2; exit 2 ;;
    esac
done
[ -n "$py" ] && [ "${#wheels[@]}" -gt 0 ] || exit 2
[ -z "${RIG_UV_FAIL:-}" ] || { echo "rig uv: install failed" >&2; exit 1; }
exec "$py" "$RIG_ROOT/rig_install.py" "$reinstall" "${wheels[@]}"
'''

STUB_CURL = r'''#!/bin/bash
echo "curl $*" >> "$RIG_LOG"
out=""; url=""
while [ $# -gt 0 ]; do
    case "$1" in
        -o) out="$2"; shift 2 ;;
        --max-time) shift 2 ;;
        -*) shift ;;
        *) url="$1"; shift ;;
    esac
done
case "$url" in
    *data.pyg.org*) echo '<a href="x">pyg_lib-0.6.0+pt210cu130-cp312-cp312-linux_x86_64.whl</a>' ;;
    *github.com*)
        if [ "${RIG_CURL:-ok}" != ok ]; then
            echo "curl: (22) The requested URL returned error: 504" >&2; exit 22
        fi
        cp "$RIG_TARBALL" "$out" ;;
    *) exit 6 ;;
esac
'''

STUBS = {
    "uname": '#!/bin/bash\nif [ "${1:-}" = -m ]; then echo aarch64; else echo Linux; fi\n',
    "uv": STUB_UV,
    "curl": STUB_CURL,
    "nvcc": ('#!/bin/bash\necho "nvcc: NVIDIA (R) Cuda compiler driver"\n'
             'echo "Cuda compilation tools, release 13.0, V13.0.88"\n'),
    "gcc": ('#!/bin/bash\ncase "${1:-}" in -dumpfullversion) echo 13.3.0 ;;\n'
            '  *) echo "gcc (Rig 13.3.0) 13.3.0" ;; esac\n'),
    "g++": ('#!/bin/bash\ncase "${1:-}" in -dumpfullversion) echo 13.3.0 ;;\n'
            '  *) echo "g++ (Rig 13.3.0) 13.3.0" ;; esac\n'),
    "cmake": '#!/bin/bash\necho "cmake version 3.28.3"\n',
    "ninja": '#!/bin/bash\necho "1.11.1"\n',
}


@dataclass
class Rig:
    root: Path
    repo: Path
    venv: Path
    purelib: Path
    log: Path
    env: Dict[str, str] = field(default_factory=dict)

    def run(self, script: str, *args: str, timeout: int = 180, **env: str):
        merged = {**self.env, **env}
        return subprocess.run(["bash", script, *args], cwd=self.repo, env=merged,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              timeout=timeout)

    def python(self, *args: str, **env: str):
        return subprocess.run([str(self.venv / "bin" / "python"), *args], cwd=self.repo,
                              env={**self.env, **env}, capture_output=True, text=True,
                              timeout=120)

    def calls(self, tool: str) -> List[str]:
        if not self.log.exists():
            return []
        return [line[len(tool) + 1:] for line in self.log.read_text().splitlines()
                if line.startswith(tool + " ")]

    def pip_wheel_calls(self) -> List[List[str]]:
        return [json.loads(line) for line in self.calls("pip")]

    def installs(self) -> List[str]:
        return [line for line in self.calls("uv") if line.startswith("pip install --python")]

    def receipt(self) -> Optional[dict]:
        path = self.venv / "share" / "shepherd-advanced" / "pyg-native-receipt.json"
        return json.loads(path.read_text()) if path.exists() else None

    def installed_native(self, name: str) -> bytes:
        return (self.purelib / name / "_native.so").read_bytes()


def _wheel_maker(root: Path):
    spec = importlib.util.spec_from_file_location("rig_wheels", root / "rig_wheels.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_rig(tmp_path: Path, *, pin_release: bool = True, release_salt: str = "release") -> Rig:
    root = tmp_path / "rig"
    repo, bindir, home, cuda = root / "repo", root / "bin", root / "home", root / "cuda"
    for directory in (repo / "scripts", bindir, home, cuda / "bin", root / "tmp"):
        directory.mkdir(parents=True)
    (root / "rig_wheels.py").write_text(WHEEL_MAKER)
    (root / "rig_install.py").write_text(RIG_INSTALL)
    for name, text in STUBS.items():
        stub = bindir / name
        stub.write_text(text)
        stub.chmod(0o755)

    venv = repo / ".venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    purelib = Path(subprocess.run(
        [str(venv / "bin" / "python"), "-c",
         "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        check=True, capture_output=True, text=True).stdout.strip())
    for package, text in (("torch", FAKE_TORCH), ("torch_geometric", "__version__ = '2.8.1'\n"),
                          ("pip", ""), ("setuptools", ""), ("wheel", "")):
        (purelib / package).mkdir()
        (purelib / package / "__init__.py").write_text(text)
    (purelib / "pip" / "__main__.py").write_text(FAKE_PIP)

    # The scripts under test, as they are; deploy.sh is copied so that a test
    # can pin the rig's release tarball in place of the real asset's digest.
    for name in ("build_pyg_arm.sh", "validate_pyg_ext.py"):
        (repo / "scripts" / name).symlink_to(REPO / "scripts" / name)
    (repo / "src").symlink_to(REPO / "src")
    deploy = (REPO / "deploy.sh").read_text(encoding="utf-8")

    # The release tarball: the four wheels at its top level, as the real one.
    wheels_dir = root / "release-wheels"
    wheels_dir.mkdir()
    maker = _wheel_maker(root)
    tarball = root / f"pyg-ext-torch2.10.0-cu130-{PY_TAG}-linux_aarch64.tar.gz"
    with tarfile.open(tarball, "w:gz") as archive:
        for name, version in RELEASE_VERSIONS.items():
            wheel = Path(maker.make_wheel(str(wheels_dir), name, version, release_salt))
            archive.add(wheel, arcname=wheel.name)
    if pin_release:
        digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
        line = next(l for l in deploy.splitlines() if l.startswith("PYG_PREBUILT_SHA256_cp312="))
        deploy = deploy.replace(line, f'PYG_PREBUILT_SHA256_{PY_TAG}="{digest}"')
    (repo / "deploy.sh").write_text(deploy, encoding="utf-8")

    log = root / "calls.log"
    env = {
        "PATH": f"{bindir}:/usr/bin:/bin",
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "CUDA_HOME": str(cuda),
        "RIG_ROOT": str(root),
        "RIG_LOG": str(log),
        "RIG_TARBALL": str(tarball),
        # The builder's logs and the pull's download (mktemp) stay in the rig.
        "TMPDIR": str(root / "tmp"),
    }
    return Rig(root=root, repo=repo, venv=venv, purelib=purelib, log=log, env=env)


def _assert_stopped_before_completion(done) -> None:
    assert "Deployment Complete!" not in done.stdout
    assert "[STOP] Prebuilt PyG pull failed" in done.stdout


# --- The prebuilt pull ---------------------------------------------------------

def test_a_pull_installs_the_release_wheels_by_path_checks_them_and_keeps_a_receipt(tmp_path):
    rig = make_rig(tmp_path)
    done = rig.run("deploy.sh", PYG_ARM_STRATEGY="pull")

    assert done.returncode == 0, done.stdout + done.stderr
    assert "[OK] SHA-256 verified" in done.stdout
    assert "4 of 4 installed exactly as the wheels hold them" in done.stdout
    assert "installed and checked; source receipt written" in done.stdout
    assert "PyG native route: prebuilt release pyg-arm-cu130-torch2.10.0" in done.stdout
    assert "If the pull fails: stop" in done.stdout
    (install,) = rig.installs()
    assert "--reinstall --no-deps" in install
    assert sorted(Path(word).name.split("-")[0] for word in install.split()
                  if word.endswith(".whl")) == sorted(NATIVE)
    assert rig.pip_wheel_calls() == []                       # nothing was built

    receipt = rig.receipt()
    assert receipt["route"] == "prebuilt release"
    assert receipt["toolchain"] is None and receipt["flags"] is None
    # The release's declared target, in the notation a source build records
    # (torch.__version__, torch.version.cuda), so one target reads alike.
    assert receipt["compiled_against"] == {"torch": "2.10.0+cu130", "cuda": "13.0",
                                           "basis": "the release's declared target"}
    assert {w["distribution"]: w["resolved"]["version"] for w in receipt["wheels"]} == \
        RELEASE_VERSIONS
    tarball_digest = hashlib.sha256(Path(rig.env["RIG_TARBALL"]).read_bytes()).hexdigest()
    for wheel in receipt["wheels"]:
        assert wheel["requested"] == {"kind": "prebuilt release",
                                      "tag": "pyg-arm-cu130-torch2.10.0",
                                      "asset": Path(rig.env["RIG_TARBALL"]).name,
                                      "asset_sha256": tarball_digest}
        assert wheel["installed"]["native_count"] == 2


def test_a_failed_download_stops_by_default_and_builds_nothing(tmp_path):
    rig = make_rig(tmp_path)
    done = rig.run("deploy.sh", PYG_ARM_STRATEGY="pull", RIG_CURL="fail")

    assert done.returncode == 12, done.stdout + done.stderr
    _assert_stopped_before_completion(done)
    assert "Download failed" in done.stdout and "the download failed" in done.stdout
    assert "SHA-256 verified" not in done.stdout
    assert rig.pip_wheel_calls() == [] and rig.installs() == []
    assert rig.receipt() is None


def test_a_checksum_mismatch_stops_and_is_never_reported_as_verified(tmp_path):
    rig = make_rig(tmp_path, pin_release=False)         # the real asset's pin stays
    done = rig.run("deploy.sh", PYG_ARM_STRATEGY="pull")

    assert done.returncode == 13, done.stdout + done.stderr
    _assert_stopped_before_completion(done)
    assert "SHA-256 mismatch" in done.stdout and "NOT verified" in done.stdout
    assert "SHA-256 verified" not in done.stdout
    assert rig.installs() == [] and rig.pip_wheel_calls() == []


@needs_header
def test_an_explicit_build_fallback_builds_after_a_failed_pull(tmp_path):
    rig = make_rig(tmp_path)
    done = rig.run("deploy.sh", PYG_ARM_STRATEGY="pull", RIG_CURL="fail",
                   PYG_ARM_ON_PULL_FAILURE="build")

    assert done.returncode == 0, done.stdout + done.stderr
    assert "Building from source instead" in done.stdout
    assert "PyG native route: local build, after the prebuilt pull failed (fetch)" in done.stdout
    assert len(rig.pip_wheel_calls()) == 4
    assert rig.receipt()["route"] == "local build"


def test_the_host_that_does_not_match_the_release_stops_a_pull_too(tmp_path):
    """Case B: the release cannot be used on this host, which is a failed pull;
    it builds only where a build was allowed."""
    rig = make_rig(tmp_path)
    done = rig.run("deploy.sh", PYG_ARM_STRATEGY="pull", RIG_TORCH_VERSION="2.11.0+cu130")

    assert done.returncode == 11, done.stdout + done.stderr
    _assert_stopped_before_completion(done)
    assert "the prebuilt release is not usable on this host" in done.stdout
    assert rig.calls("curl") == [] and rig.pip_wheel_calls() == []


@needs_header
def test_the_host_that_does_not_match_the_release_builds_where_allowed(tmp_path):
    rig = make_rig(tmp_path)
    done = rig.run("deploy.sh", PYG_ARM_STRATEGY="pull", RIG_TORCH_VERSION="2.11.0+cu130",
                   PYG_ARM_ON_PULL_FAILURE="build")

    assert done.returncode == 0, done.stdout + done.stderr
    assert "after the prebuilt pull failed (target)" in done.stdout
    assert rig.calls("curl") == [
        "-fsSL --max-time 15 https://data.pyg.org/whl/torch-2.11.0+cu130.html"]
    assert rig.receipt()["compiled_against"]["torch"] == "2.11.0+cu130"


def test_a_wrong_wheel_installed_fails_the_pull_check(tmp_path):
    rig = make_rig(tmp_path)
    done = rig.run("deploy.sh", PYG_ARM_STRATEGY="pull", RIG_UV_TAMPER="torch_scatter")

    assert done.returncode == 15, done.stdout + done.stderr
    _assert_stopped_before_completion(done)
    assert ("[FAIL] torch_scatter 2.1.2: native library torch_scatter/_native.so differs "
            "from the wheel's") in done.stdout
    assert "the installed extensions are not shown to be the release's wheels" in done.stdout
    assert rig.receipt() is None


def _block_the_receipt(rig: Rig) -> None:
    """A file where the receipt's directory goes: the receipt cannot be written."""
    (rig.venv / "share").mkdir(exist_ok=True)
    (rig.venv / "share" / "shepherd-advanced").write_text("a file in the way")


def test_a_pull_whose_receipt_cannot_be_written_says_so_and_builds_nothing(tmp_path):
    """The release's wheels passed their check: the receipt, not the check,
    failed. A build would keep its receipt in the same place, so the build
    fallback does not apply."""
    rig = make_rig(tmp_path)
    _block_the_receipt(rig)
    done = rig.run("deploy.sh", PYG_ARM_STRATEGY="pull", PYG_ARM_ON_PULL_FAILURE="build")

    assert done.returncode == 17, done.stdout + done.stderr
    assert "Deployment Complete!" not in done.stdout
    assert "4 of 4 installed exactly as the wheels hold them" in done.stdout
    assert "installed and checked, but the source receipt could not be written" in done.stdout
    assert "not the release's wheels" not in done.stdout
    assert rig.pip_wheel_calls() == []


def test_a_failed_install_of_the_pull_stops(tmp_path):
    rig = make_rig(tmp_path)
    done = rig.run("deploy.sh", PYG_ARM_STRATEGY="pull", RIG_UV_FAIL="1")

    assert done.returncode == 14, done.stdout + done.stderr
    _assert_stopped_before_completion(done)
    assert "the install failed" in done.stdout


def test_an_unknown_failure_setting_is_refused_before_anything_runs(tmp_path):
    rig = make_rig(tmp_path)
    done = rig.run("deploy.sh", PYG_ARM_STRATEGY="pull", PYG_ARM_ON_PULL_FAILURE="maybe")

    assert done.returncode == 4, done.stdout + done.stderr
    assert "is not 'stop' or 'build'" in done.stdout
    assert rig.calls("uv") == []


def test_with_no_terminal_and_no_strategy_the_default_stays_skip(tmp_path):
    rig = make_rig(tmp_path)
    done = rig.run("deploy.sh")

    assert done.returncode == 0, done.stdout + done.stderr
    assert "No TTY; defaulting to 'skip'" in done.stdout
    assert "PyG native route: none: skipped" in done.stdout
    assert rig.calls("curl") == [] and rig.installs() == []


@needs_header
def test_the_menu_asks_what_a_failed_pull_does_once_at_the_start(tmp_path):
    """On a terminal, choosing the pull asks the failure question there and
    then; the failed download later builds without asking anything."""
    import pty

    rig = make_rig(tmp_path)
    master, slave = pty.openpty()
    try:
        process = subprocess.Popen(["bash", "deploy.sh"], cwd=rig.repo,
                                   env={**rig.env, "RIG_CURL": "fail"}, stdin=slave,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        os.close(slave)
        os.write(master, b"1\n2\n")
        output, _ = process.communicate(timeout=180)
    finally:
        os.close(master)

    assert process.returncode == 0, output
    assert output.count("If the download, its checksum, its install or its check fails") == 1
    assert "If the pull fails: build" in output
    assert "Building from source instead" in output
    assert len(rig.pip_wheel_calls()) == 4


# --- The source build -----------------------------------------------------------

@needs_header
def test_the_spec_overrides_reach_the_builder_and_its_receipt(tmp_path):
    rig = make_rig(tmp_path)
    local = rig.root / "sources" / "torch_cluster-src"
    local.mkdir(parents=True)
    (local / "setup.py").write_text("# a local source tree\n")
    from src.utils.pyg_native_check import tree_sha256
    tree = tree_sha256(local)
    specs = {
        "SCATTER_SPEC": "torch-scatter==2.1.1",
        "SPARSE_SPEC": "git+https://reader:s3cret@github.com/rusty1s/pytorch_sparse.git@0.6.17",
        "CLUSTER_SPEC": str(local),
        "PYGLIB_SPEC": "git+https://github.com/pyg-team/pyg-lib.git@0.5.0",
    }
    done = rig.run("deploy.sh", PYG_ARM_STRATEGY="compile", **specs)

    assert done.returncode == 0, done.stdout + done.stderr
    calls = rig.pip_wheel_calls()
    assert [call[1] for call in calls] == [specs["SCATTER_SPEC"], specs["SPARSE_SPEC"],
                                           specs["CLUSTER_SPEC"], specs["PYGLIB_SPEC"]]
    assert all("--no-cache-dir" in call for call in calls)
    assert rig.calls("curl") == []                  # PYGLIB_SPEC given: nothing resolved

    receipt = rig.receipt()
    by_name = {w["distribution"]: w for w in receipt["wheels"]}
    assert by_name["torch_scatter"]["requested"] == {"kind": "registry", "name": "torch-scatter",
                                                     "specifier": "==2.1.1"}
    assert by_name["torch_scatter"]["resolved"] == {"version": "2.1.1"}
    assert by_name["torch_sparse"]["requested"] == {
        "kind": "vcs", "vcs": "git", "url": "https://github.com/rusty1s/pytorch_sparse.git",
        "ref": "0.6.17"}
    assert by_name["torch_sparse"]["resolved"] == {"version": "0.6.17", "commit": FAKE_COMMIT}
    assert by_name["torch_cluster"]["requested"] == {"kind": "local source", "form": "tree",
                                                     "sha256": tree}
    assert by_name["pyg_lib"]["resolved"]["version"] == "0.5.0"
    assert receipt["toolchain"] == {"nvcc": "13.0.88", "gcc": "13.3.0", "g++": "13.3.0"}
    assert receipt["flags"]["FORCE_CUDA"] == "1"
    assert receipt["flags"]["TORCH_CUDA_ARCH_LIST"] == "12.1"
    assert receipt["flags"]["MAX_JOBS"]
    assert receipt["compiled_against"] == {"torch": "2.10.0+cu130", "cuda": "13.0",
                                           "basis": "the build's target environment"}
    text = json.dumps(receipt)
    for private in ("s3cret", "reader", str(rig.root), str(tmp_path), "/home", "/tmp"):
        assert private not in text


@needs_header
def test_each_build_run_uses_a_fresh_directory_and_installs_only_its_own_wheels(tmp_path):
    rig = make_rig(tmp_path)
    out = rig.root / "wheel-out"
    out.mkdir()
    maker = _wheel_maker(rig.root)
    stale = Path(maker.make_wheel(str(out), "torch_scatter", "9.9.9", "stale"))

    first = rig.run("scripts/build_pyg_arm.sh", PYG_WHEEL_OUT=str(out), RIG_BUILD_SALT="one",
                    PYGLIB_SPEC="git+https://github.com/pyg-team/pyg-lib.git@0.6.0")
    second = rig.run("scripts/build_pyg_arm.sh", PYG_WHEEL_OUT=str(out), RIG_BUILD_SALT="two",
                     PYGLIB_SPEC="git+https://github.com/pyg-team/pyg-lib.git@0.6.0")

    assert first.returncode == 0 and second.returncode == 0, second.stdout + second.stderr
    runs = sorted(p for p in out.iterdir() if p.is_dir())
    assert len(runs) == 2 and all(p.name.startswith("run-") for p in runs)
    assert stale.exists() and all(len(list(run.glob("*.whl"))) == 4 for run in runs)
    first_install, second_install = rig.installs()
    second_wheels = [w for w in second_install.split() if w.endswith(".whl")]
    assert len(second_wheels) == 4
    run_dirs = {Path(w).parent for w in second_wheels}
    assert len(run_dirs) == 1 and run_dirs != {Path(w).parent for w in first_install.split()
                                               if w.endswith(".whl")}
    assert str(stale) not in first_install + second_install
    assert rig.installed_native("torch_scatter") == b"native torch_scatter 2.1.2 two"


@needs_header
def test_a_failed_build_installs_nothing_and_exits_non_zero(tmp_path):
    rig = make_rig(tmp_path)
    done = rig.run("scripts/build_pyg_arm.sh", RIG_PIP_FAIL="torch_sparse",
                   PYGLIB_SPEC="git+https://github.com/pyg-team/pyg-lib.git@0.6.0")

    assert done.returncode == 2, done.stdout + done.stderr
    assert "did not build; nothing was installed" in done.stdout
    assert rig.installs() == [] and rig.receipt() is None


@needs_header
def test_a_failed_install_and_a_wrong_wheel_exit_non_zero(tmp_path):
    rig = make_rig(tmp_path)
    spec = "git+https://github.com/pyg-team/pyg-lib.git@0.6.0"
    failed = rig.run("scripts/build_pyg_arm.sh", RIG_UV_FAIL="1", PYGLIB_SPEC=spec)
    assert failed.returncode == 3, failed.stdout + failed.stderr
    assert "The install of this run's wheels failed" in failed.stdout

    wrong = rig.run("scripts/build_pyg_arm.sh", RIG_UV_TAMPER="torch_cluster", PYGLIB_SPEC=spec)
    assert wrong.returncode == 4, wrong.stdout + wrong.stderr
    assert "[FAIL] torch_cluster 1.6.3: native library torch_cluster/_native.so differs" in \
        wrong.stdout
    assert rig.receipt() is None


@needs_header
def test_an_extension_that_does_not_import_fails_the_build_before_its_receipt(tmp_path):
    rig = make_rig(tmp_path)
    done = rig.run("scripts/build_pyg_arm.sh", RIG_UV_BREAK_IMPORT="pyg_lib",
                   PYGLIB_SPEC="git+https://github.com/pyg-team/pyg-lib.git@0.6.0")

    assert done.returncode == 4, done.stdout + done.stderr
    assert "[--] pyg_lib: rig: undefined symbol" in done.stdout
    assert "An installed extension does not import" in done.stdout
    assert rig.receipt() is None


@needs_header
def test_a_build_whose_receipt_cannot_be_written_has_its_own_exit(tmp_path):
    rig = make_rig(tmp_path)
    _block_the_receipt(rig)
    done = rig.run("scripts/build_pyg_arm.sh",
                   PYGLIB_SPEC="git+https://github.com/pyg-team/pyg-lib.git@0.6.0")

    assert done.returncode == 5, done.stdout + done.stderr
    assert "4 of 4 installed exactly as the wheels hold them" in done.stdout
    assert "its source receipt could not be written" in done.stdout
    assert "source receipt written into the virtual environment" not in done.stdout


@needs_header
def test_a_build_for_an_interpreter_that_is_not_a_virtual_environment_keeps_no_receipt(tmp_path):
    """The receipt is kept in a virtual environment only; the builder says
    that none was written, rather than that one was."""
    rig = make_rig(tmp_path)
    # A .pth import line runs at every start-up (a distribution's own
    # sitecustomize, as Ubuntu's, would shadow one in the venv).
    (rig.purelib / "rig_not_a_venv.pth").write_text(
        "import os, sys; sys.base_prefix = sys.prefix if os.environ.get('RIG_NOT_A_VENV') "
        "else sys.base_prefix\n")
    done = rig.run("scripts/build_pyg_arm.sh", RIG_NOT_A_VENV="1",
                   PYGLIB_SPEC="git+https://github.com/pyg-team/pyg-lib.git@0.6.0")

    assert done.returncode == 0, done.stdout + done.stderr
    assert "no source receipt: the target python is not a virtual environment" in done.stdout
    assert "source receipt written into the virtual environment" not in done.stdout
    assert rig.receipt() is None


@needs_header
def test_a_failed_source_build_is_said_and_the_deployment_continues(tmp_path):
    """The compile route keeps its meaning: a build that did not complete is
    named, nothing from it is installed, and the deployment goes on."""
    rig = make_rig(tmp_path)
    done = rig.run("deploy.sh", PYG_ARM_STRATEGY="compile", RIG_PIP_FAIL="pyg_lib")

    assert done.returncode == 0, done.stdout + done.stderr
    assert "PyG native route: none: the source build did not complete" in done.stdout
    assert "This deployment did not install the PyG native extensions" in done.stdout
    assert rig.installs() == []
    # Its logs stay in the rig's temporary directory.
    assert any((rig.root / "tmp").glob("*/pyg-lib.log"))


def _assert_stopped_unverified(done, rig: Rig) -> None:
    assert done.returncode == 16, done.stdout + done.stderr
    assert "Deployment Complete!" not in done.stdout
    assert "NOT verified" in done.stdout
    assert "did not install" not in done.stdout and "PyG native route: none" not in done.stdout
    assert rig.receipt() is None


@needs_header
@pytest.mark.parametrize("fault", [{"RIG_UV_TAMPER": "torch_cluster"},
                                   {"RIG_UV_BREAK_IMPORT": "torch_cluster"},
                                   {"RIG_UV_FAIL": "1"}])
def test_a_source_build_that_installed_but_is_not_verified_stops_the_deployment(tmp_path, fault):
    """Once the build has installed its wheels, a failed install, check or
    import leaves bytes nobody has verified: never "nothing installed"."""
    rig = make_rig(tmp_path)
    done = rig.run("deploy.sh", PYG_ARM_STRATEGY="compile", **fault)

    _assert_stopped_unverified(done, rig)
    assert len(rig.installs()) == 1


@needs_header
def test_the_build_after_a_failed_pull_stops_when_its_install_is_not_verified(tmp_path):
    rig = make_rig(tmp_path)
    done = rig.run("deploy.sh", PYG_ARM_STRATEGY="pull", RIG_CURL="fail",
                   PYG_ARM_ON_PULL_FAILURE="build", RIG_UV_TAMPER="torch_cluster")

    _assert_stopped_unverified(done, rig)
    assert "Building from source instead" in done.stdout
    assert ("[FAIL] torch_cluster 1.6.3: native library torch_cluster/_native.so differs "
            "from the wheel's") in done.stdout


@needs_header
def test_a_pull_that_installed_unverified_wheels_stops_when_the_build_does_not_complete(tmp_path):
    """The pull's check failed after its install, and the build allowed in its
    place stopped before installing anything: the release's unverified wheels
    are what is installed, and the deployment says so and stops."""
    rig = make_rig(tmp_path)
    done = rig.run("deploy.sh", PYG_ARM_STRATEGY="pull", PYG_ARM_ON_PULL_FAILURE="build",
                   RIG_UV_TAMPER="torch_scatter", RIG_PIP_FAIL="pyg_lib")

    _assert_stopped_unverified(done, rig)
    assert "Building from source instead" in done.stdout
    assert len(rig.installs()) == 1                   # the pull's; the build installed nothing
    assert rig.installed_native("torch_scatter").endswith(b" and something else")


@needs_header
def test_a_source_build_whose_receipt_cannot_be_written_stops_the_deployment(tmp_path):
    rig = make_rig(tmp_path)
    _block_the_receipt(rig)
    done = rig.run("deploy.sh", PYG_ARM_STRATEGY="compile")

    assert done.returncode == 17, done.stdout + done.stderr
    assert "Deployment Complete!" not in done.stdout
    assert "installed and checked, but the source receipt could not be written" in done.stdout


# --- The receipt, later ----------------------------------------------------------

@needs_header
def test_a_receipt_that_no_longer_matches_the_installed_bytes_says_so(tmp_path):
    rig = make_rig(tmp_path)
    built = rig.run("scripts/build_pyg_arm.sh",
                    PYGLIB_SPEC="git+https://github.com/pyg-team/pyg-lib.git@0.6.0")
    assert built.returncode == 0, built.stdout + built.stderr

    shown = rig.python("scripts/validate_pyg_ext.py", "--receipt")
    assert shown.returncode == 0 and shown.stdout.count("[OK]") == 4, shown.stdout

    # Another install of torch_sparse, from elsewhere, after the build.
    library = rig.purelib / "torch_sparse" / "_native.so"
    library.write_bytes(b"another build of the same version")
    shown = rig.python("scripts/validate_pyg_ext.py", "--receipt")
    assert shown.returncode == 0, shown.stdout                 # information, never a gate
    assert ("[DIFFERS] torch_sparse: its installed native libraries are not the ones the "
            "receipt records") in shown.stdout
    assert shown.stdout.count("[OK]") == 3


# --- uv and the receipt's place ---------------------------------------------------

@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed here")
def test_uv_sync_inexact_leaves_the_receipt_and_the_extensions(tmp_path):
    """Where the receipt is kept: one file in the virtual environment, outside
    every distribution. Run against uv itself, offline: `uv sync --inexact`
    removes neither it nor a distribution outside the lock; an exact `uv sync`
    removes that distribution and still leaves the receipt, which then no
    longer matches anything installed."""
    from src.utils.pyg_native_check import (
        check_installed_against_wheels,
        build_receipt,
        read_receipt,
        receipt_against_installed,
        write_receipt,
    )

    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname = "receipt-place"\nversion = "0"\nrequires-python = ">=3.10"\n'
        'dependencies = []\n\n[tool.uv]\npackage = false\n')
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("UV_", "VIRTUAL_ENV", "CONDA"))}
    env.update(UV_CACHE_DIR=str(tmp_path / "uv-cache"), UV_PYTHON_DOWNLOADS="never")

    def uv(*args):
        done = subprocess.run(["uv", *args], cwd=project, env=env, capture_output=True,
                              text=True, timeout=120)
        assert done.returncode == 0, done.stdout + done.stderr

    base = getattr(sys, "_base_executable", sys.executable)
    uv("sync", "--offline", "--python", base)
    venv = project / ".venv"
    maker_root = tmp_path / "maker"
    maker_root.mkdir()
    (maker_root / "rig_wheels.py").write_text(WHEEL_MAKER)
    # A tag this machine installs: uv refuses the rig's aarch64 tag on another host.
    wheel = Path(_wheel_maker(maker_root).make_wheel(str(tmp_path), "torch_scatter", "2.1.2",
                                                     tag="py3-none-any"))
    uv("pip", "install", "--offline", "--python", str(venv / "bin" / "python"), "--no-deps",
       str(wheel))

    site = subprocess.run([str(venv / "bin" / "python"), "-c",
                           "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                          capture_output=True, text=True, check=True).stdout.strip()
    checks = check_installed_against_wheels([wheel], path=[site])
    assert checks[0].ok, checks[0].problems
    receipt = build_receipt(route="local build", checks=checks,
                            requested={wheel.name: {"kind": "registry", "name": "torch-scatter",
                                                    "specifier": ""}},
                            compiled_against={"torch": None, "cuda": None, "basis": "test"},
                            toolchain=None, flags=None)
    write_receipt(receipt, venv)

    uv("sync", "--inexact", "--offline")
    assert read_receipt(venv) == receipt
    assert receipt_against_installed(receipt, path=[site]) == {"torch_scatter": {"state": "matches"}}

    uv("sync", "--offline")
    assert read_receipt(venv) == receipt
    assert receipt_against_installed(receipt, path=[site]) == {
        "torch_scatter": {"state": "differs", "reason": "it is not installed now"}}
