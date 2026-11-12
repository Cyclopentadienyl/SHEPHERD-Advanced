"""
Files published by rename get the mode a plain open() would give them.

The publishers write a staging file and rename it into place. With `tempfile`
that staging file was 0600, so the split manifest, provenance and the
evaluation ledger all published 0600 -- including over files that had been
0644. `create_staging_file` restores open()'s behaviour: a new file gets
the directory's policy, applied by the kernel; a rewritten file keeps its mode.

**The reference for "what open() would do" is open() itself**, run in the same
directory under the same umask, so the tests hold under whatever policy the
host applies -- umask or default ACL -- rather than restating a formula.
"""
import errno
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from src.utils import file_modes
from src.utils.file_modes import create_staging_file

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX permission bits; Windows has ACLs"
)


@pytest.fixture(params=[0o022, 0o077], ids=["umask022", "umask077"])
def umask(request):
    previous = os.umask(request.param)
    try:
        yield request.param
    finally:
        os.umask(previous)


@pytest.fixture
def umask022():
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


def _mode(path) -> int:
    return stat.S_IMODE(Path(path).stat().st_mode)


def _open_would_give(directory: Path) -> int:
    """The mode a plain open() gives a new file in `directory`, measured."""
    reference = directory / ".reference-open"
    with open(reference, "w"):
        pass
    mode = _mode(reference)
    reference.unlink()
    return mode


def _stage(target: Path) -> Path:
    handle, name = create_staging_file(target)
    handle.close()
    return Path(name)


# ------------------------------------------------------------------- the helper
def test_a_new_target_is_staged_as_open_would_create_it(umask, tmp_path):
    assert _mode(_stage(tmp_path / "new.bin")) == _open_would_give(tmp_path)


def test_a_new_target_leaves_the_mode_to_the_kernel(tmp_path, monkeypatch):
    # 0o666 is what open() asks for; the kernel then applies the umask or the
    # directory's default ACL. Asking for less would override an ACL.
    requested = []
    real_open = os.open
    monkeypatch.setattr(file_modes.os, "open", lambda path, flags, mode=0o777: (
        requested.append(mode) or real_open(path, flags, mode)))
    _stage(tmp_path / "new.bin")
    assert requested == [0o666]


@pytest.mark.parametrize("existing", [0o600, 0o640, 0o664, 0o644])
def test_a_rewritten_target_keeps_its_mode(existing, umask022, tmp_path):
    # 0o664 under umask 022 is the case where os.open's mode alone would lose
    # a bit; the helper sets the old mode exactly.
    target = tmp_path / "old.bin"
    target.write_bytes(b"old")
    os.chmod(target, existing)
    assert _mode(_stage(target)) == existing


def test_special_bits_are_not_carried_over(umask022, tmp_path):
    target = tmp_path / "old.bin"
    target.write_bytes(b"old")
    os.chmod(target, 0o2755)
    assert _mode(_stage(target)) == _mode(target) & 0o777


def test_a_name_that_already_exists_is_never_opened(tmp_path, monkeypatch):
    # O_EXCL: a symlink planted at the chosen name is not written through.
    decoy = tmp_path / "decoy"
    (tmp_path / f"t.bin{'a' * 16}.tmp").symlink_to(decoy)
    tokens = iter(["a" * 16, "b" * 16])
    monkeypatch.setattr(file_modes.secrets, "token_hex", lambda n: next(tokens))
    staged = _stage(tmp_path / "t.bin")
    assert staged.name == f"t.bin{'b' * 16}.tmp"
    assert not decoy.exists()


def test_windows_flags_are_requested_where_they_exist(tmp_path, monkeypatch):
    # O_BINARY keeps Windows from translating newlines in a binary artifact.
    monkeypatch.setattr(os, "O_BINARY", 0x8000, raising=False)
    seen = []
    real_open = os.open
    monkeypatch.setattr(file_modes.os, "open", lambda path, flags, mode=0o777: (
        seen.append(flags) or real_open(path, flags & ~0x8000, mode)))
    _stage(tmp_path / "t.bin")
    assert seen[0] & 0x8000


def test_the_mode_is_set_on_the_file_that_was_opened_not_on_its_name(umask022, tmp_path, monkeypatch):
    # Between the O_EXCL create and the mode change, another account that can
    # rename files here swaps the staging name for a symlink to a victim. A
    # chmod by name would follow it; the mode must land on the opened file.
    victim = tmp_path / "victim"
    victim.write_bytes(b"not yours")
    os.chmod(victim, 0o600)
    target = tmp_path / "old.bin"
    target.write_bytes(b"old")
    os.chmod(target, 0o664)
    real_open = os.open

    def _open_then_swap(path, flags, mode=0o777):
        fd = real_open(path, flags, mode)
        os.rename(path, f"{path}.moved")
        os.symlink(victim, path)
        return fd

    monkeypatch.setattr(file_modes.os, "open", _open_then_swap)
    handle, _name = create_staging_file(target)
    try:
        assert _mode(victim) == 0o600
        assert stat.S_IMODE(os.fstat(handle.fileno()).st_mode) == 0o664
    finally:
        handle.close()


def test_a_failed_chmod_leaves_nothing_behind(umask022, tmp_path, monkeypatch):
    target = tmp_path / "old.bin"
    target.write_bytes(b"old")

    def _refuse(*args):
        raise PermissionError("chmod refused")

    monkeypatch.setattr(file_modes.os, "fchmod", _refuse)
    with pytest.raises(PermissionError):
        create_staging_file(target)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["old.bin"]


def test_a_failed_open_of_the_stream_leaves_nothing_behind(tmp_path):
    with pytest.raises(LookupError):
        create_staging_file(tmp_path / "t.txt", binary=False, encoding="no-such-codec")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(shutil.which("setfacl") is None, reason="setfacl not installed")
def test_a_default_acl_decides_as_it_would_for_open(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    result = subprocess.run(["setfacl", "-d", "-m", "u::rw,g::rw,o::r", str(shared)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"default ACLs unsupported here: {result.stderr.strip()}")
    assert _mode(_stage(shared / "new.bin")) == _open_would_give(shared)


# ------------------------------------------------------------------- publishers
def _split_manifest(directory):
    from scripts.setup_demo import build_demo_kg
    from src.kg.artifacts import GRAPH_ARTIFACTS
    from src.kg.disease_allocation import allocate_diseases
    from src.kg.sample_generator import build_eligible_disease_profiles, generate_training_samples
    from src.utils.fingerprint import file_sha256
    from tests.fixtures.generated_workspace import default_graph_export

    # The graph export and the dataset record a manifest binds (contract M3b),
    # as stand-ins: the generator binds their digests and reads nothing in them.
    bound = {**GRAPH_ARTIFACTS, "provenance": "kg.provenance.json"}
    for role, filename in bound.items():
        if not (directory / filename).exists():
            (directory / filename).write_bytes(role.encode())
    kg = build_demo_kg()
    generate_training_samples(
        kg=kg,
        allocation=allocate_diseases(build_eligible_disease_profiles(kg, 2), 0.2, seed=42),
        num_train=20, num_val=5, output_dir=directory,
        graph_digests={role: file_sha256(directory / filename)
                       for role, filename in bound.items()},
        graph_export=default_graph_export(),
    )
    return [directory / "split_manifest.json"]


def _provenance(directory):
    from src.kg.provenance import PROVENANCE_FILENAME, build_provenance, write_provenance

    write_provenance(directory, build_provenance("0" * 64, origin="synthetic"))
    return [directory / PROVENANCE_FILENAME]


def _evaluation_ledger(directory):
    from src.evaluation.sidecar import empty_ledger, ledger_digest, write_ledger

    target = directory / "ledger.json"
    write_ledger(target, empty_ledger(), ledger_digest(target))
    return [target]


PUBLISHERS = {
    "split_manifest": _split_manifest,
    "provenance": _provenance,
    "evaluation_ledger": _evaluation_ledger,
}


@pytest.mark.parametrize("publisher", sorted(PUBLISHERS))
def test_a_new_artifact_gets_what_open_would_give_it(publisher, umask, tmp_path):
    expected = _open_would_give(tmp_path)
    for path in PUBLISHERS[publisher](tmp_path):
        assert _mode(path) == expected, (publisher, path.name, oct(_mode(path)))


@pytest.mark.parametrize("existing", [0o600, 0o640])
@pytest.mark.parametrize("publisher", sorted(PUBLISHERS))
def test_a_rewritten_artifact_keeps_its_mode(publisher, existing, umask022, tmp_path):
    # First publication, then the operator restricts or shares the files, then
    # the same publisher writes them again: the operator's choice stays.
    paths = PUBLISHERS[publisher](tmp_path)
    for path in paths:
        os.chmod(path, existing)
    for path in PUBLISHERS[publisher](tmp_path):
        assert _mode(path) == existing, (publisher, path.name, oct(_mode(path)))


# ------------------------------------------------------------------ publish_bytes
# The one publish sequence for bytes already in memory (M4): stage, write, flush,
# fsync, rename, unlink the staged file on failure, and the digest of exactly the
# bytes written.
def test_publish_bytes_writes_the_bytes_and_returns_their_digest(tmp_path):
    import hashlib

    from src.utils.file_modes import publish_bytes

    target = tmp_path / "out.json"
    data = b'{"a": 1}'
    assert publish_bytes(target, data) == hashlib.sha256(data).hexdigest()
    assert target.read_bytes() == data
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.json"]


def test_publish_bytes_replaces_a_target_and_keeps_its_mode(umask022, tmp_path):
    import hashlib

    from src.utils.file_modes import publish_bytes

    target = tmp_path / "out.json"
    target.write_bytes(b"old")
    os.chmod(target, 0o640)
    assert publish_bytes(target, b"new") == hashlib.sha256(b"new").hexdigest()
    assert target.read_bytes() == b"new"
    assert _mode(target) == 0o640


def test_publish_bytes_gives_a_new_target_what_open_would(umask, tmp_path):
    from src.utils.file_modes import publish_bytes

    expected = _open_would_give(tmp_path)
    publish_bytes(tmp_path / "new.bin", b"x")
    assert _mode(tmp_path / "new.bin") == expected


def test_publish_bytes_creates_no_directories(tmp_path):
    from src.utils.file_modes import publish_bytes

    with pytest.raises(FileNotFoundError):
        publish_bytes(tmp_path / "missing" / "out.json", b"x")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("fail_at", ["fsync", "replace"])
def test_a_failed_publish_leaves_the_target_and_no_staged_file(fail_at, tmp_path, monkeypatch):
    from src.utils.file_modes import publish_bytes

    target = tmp_path / "out.json"
    target.write_bytes(b"old")

    def _fail(*args, **kwargs):
        raise OSError(f"{fail_at} failed")

    monkeypatch.setattr(file_modes.os, fail_at, _fail)
    with pytest.raises(OSError, match=fail_at):
        publish_bytes(target, b"new")
    assert target.read_bytes() == b"old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.json"]


@pytest.mark.parametrize("data", ["text", bytearray(b"mutable")], ids=["str", "bytearray"])
def test_publish_bytes_refuses_anything_but_bytes_before_staging(data, tmp_path):
    # A bytearray could change between the hash and the write, so the digest
    # would not be of the bytes written; nothing is created either way.
    from src.utils.file_modes import publish_bytes

    with pytest.raises(TypeError, match="nothing was written"):
        publish_bytes(tmp_path / "out.json", data)
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------- publish_bytes(replace=False)
# K8-6r.4: a target that is taken is never overwritten. The hard link places the
# file and fails when anything is there; where hard links are not supported, an
# O_EXCL claim of the target and a replace of that claim do the same. Only
# FileExistsError is a collision.
def _sha(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


#: The errors `os.link` gives on Linux where it cannot make the link at all.
NO_HARD_LINKS = {"EPERM": errno.EPERM, "ENOTSUP": errno.ENOTSUP, "EXDEV": errno.EXDEV}


def _no_hard_links(monkeypatch, number):
    """`os.link` as a filesystem without hard links answers it."""
    def refuse(src, dst, *args, **kwargs):
        raise OSError(number, os.strerror(number), str(src), None, str(dst))

    assert number in (errno.EPERM, errno.ENOTSUP, errno.EXDEV)
    monkeypatch.setattr(file_modes.os, "link", refuse)


@pytest.fixture(params=["link", *NO_HARD_LINKS])
def placement(request, monkeypatch):
    """Both ways a file is placed without replacing: the hard link, and the
    claim-then-replace fallback for each error that sends `os.link` there."""
    if request.param != "link":
        _no_hard_links(monkeypatch, NO_HARD_LINKS[request.param])
    return request.param


def test_a_new_target_gets_the_bytes_and_their_digest_and_no_staged_file(placement, tmp_path):
    """E4, E6: published whole, the digest of the bytes written, nothing left beside it."""
    from src.utils.file_modes import publish_bytes

    target = tmp_path / "report.json"
    assert publish_bytes(target, b'{"run": 1}', replace=False) == _sha(b'{"run": 1}')
    assert target.read_bytes() == b'{"run": 1}'
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.json"]


def test_an_existing_target_is_refused_and_kept(placement, tmp_path):
    """E1, E4, E6: the second publication raises FileExistsError naming the
    target only, and the first file's bytes and mode are as they were."""
    from src.utils.file_modes import publish_bytes

    target = tmp_path / "report.json"
    publish_bytes(target, b"first", replace=False)
    os.chmod(target, 0o640)
    with pytest.raises(FileExistsError) as taken:
        publish_bytes(target, b"second", replace=False)
    assert taken.value.filename == str(target) and taken.value.filename2 is None
    assert "never replaced" in str(taken.value)
    assert target.read_bytes() == b"first" and _mode(target) == 0o640
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.json"]


def test_a_dangling_link_at_the_target_is_refused_and_left_as_it_was(placement, tmp_path):
    """E4, E6: a link to nothing is something there; neither the link nor the
    claim follows it to create the file it points at."""
    from src.utils.file_modes import publish_bytes

    target = tmp_path / "report.json"
    target.symlink_to(tmp_path / "elsewhere.json")
    with pytest.raises(FileExistsError):
        publish_bytes(target, b"new", replace=False)
    assert target.is_symlink() and os.readlink(target) == str(tmp_path / "elsewhere.json")
    assert not (tmp_path / "elsewhere.json").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.json"]


def test_an_empty_claimed_file_is_never_replaced_either(placement, tmp_path):
    """E9: another run's claim, an empty file, is a taken place like any other."""
    from src.utils.file_modes import publish_bytes

    target = tmp_path / "report.json"
    target.write_bytes(b"")
    with pytest.raises(FileExistsError):
        publish_bytes(target, b"new", replace=False)
    assert target.read_bytes() == b""
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.json"]


def test_a_collision_at_the_link_does_not_fall_back_to_a_claim(tmp_path, monkeypatch):
    """Only FileExistsError is a collision: a taken target is never claimed,
    so the fallback's replace never runs against it."""
    from src.utils.file_modes import publish_bytes

    target = tmp_path / "report.json"
    target.write_bytes(b"first")
    real_open, opened = os.open, []
    monkeypatch.setattr(file_modes.os, "open", lambda path, *a, **k: (
        opened.append(Path(path)) or real_open(path, *a, **k)))
    monkeypatch.setattr(file_modes.os, "replace",
                        lambda *a, **k: pytest.fail("a taken target was replaced"))
    with pytest.raises(FileExistsError):
        publish_bytes(target, b"second", replace=False)
    assert target not in opened, "the taken target was opened to be claimed"
    assert target.read_bytes() == b"first"


@pytest.mark.parametrize("error", sorted(NO_HARD_LINKS))
def test_a_failed_replace_of_the_claim_leaves_only_the_empty_claim(error, tmp_path, monkeypatch):
    """E6, E8, E9 (b): the claim was made, the replace onto it failed (a full
    disk): the target is this call's empty file, never removed by name, and the
    staged file is gone. A later publication there is refused by that claim."""
    from src.utils.file_modes import publish_bytes

    _no_hard_links(monkeypatch, NO_HARD_LINKS[error])
    real_replace = file_modes.os.replace

    def full_disk(src, dst, *args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device", str(dst))

    target = tmp_path / "report.json"
    monkeypatch.setattr(file_modes.os, "replace", full_disk)
    with pytest.raises(OSError, match="No space left"):
        publish_bytes(target, b"new", replace=False)
    assert target.is_file() and not target.is_symlink() and target.read_bytes() == b""
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.json"]

    monkeypatch.setattr(file_modes.os, "replace", real_replace)
    with pytest.raises(FileExistsError):
        publish_bytes(target, b"again", replace=False)
    assert target.read_bytes() == b""
    retry = tmp_path / "retry.json"
    assert publish_bytes(retry, b"again", replace=False) == _sha(b"again")
    assert retry.read_bytes() == b"again"


def test_the_link_is_the_commit_point_so_a_failed_tidy_is_not_a_failed_publication(
        tmp_path, monkeypatch):
    """Once the hard link has placed the file, `target` holds the whole bytes.
    An error removing the staging name after that (an I/O error) must not
    report the whole file as failed: the call returns the digest, and the
    staging name, a second name for the same bytes, may be left beside it."""
    from src.utils.file_modes import publish_bytes

    real_unlink = Path.unlink

    def io_error_at_the_staging_name(self, missing_ok=False):
        if self.name.endswith(".tmp"):
            raise OSError(errno.EIO, "Input/output error", str(self))
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", io_error_at_the_staging_name)
    target = tmp_path / "report.json"
    assert publish_bytes(target, b'{"whole": true}', replace=False) == _sha(b'{"whole": true}')
    assert target.read_bytes() == b'{"whole": true}'
    left = [p for p in tmp_path.iterdir() if p != target]
    assert all(p.name.endswith(".tmp") and p.read_bytes() == b'{"whole": true}' for p in left)


@pytest.mark.parametrize("fail_at", ["fsync", "link"])
def test_a_failure_before_placement_leaves_no_file(fail_at, tmp_path, monkeypatch):
    """A write that fails before the file is placed (a full disk at the flush,
    or at the link itself on a filesystem that then also refuses the claim)
    leaves no target and no staged file."""
    from src.utils.file_modes import publish_bytes

    def full_disk(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(file_modes.os, fail_at, full_disk)
    if fail_at == "link":
        real_open = os.open
        monkeypatch.setattr(file_modes.os, "open", lambda path, flags, mode=0o777: (
            full_disk() if Path(path).name == "report.json" else real_open(path, flags, mode)))
    target = tmp_path / "report.json"
    with pytest.raises(OSError, match="No space left"):
        publish_bytes(target, b"new", replace=False)
    assert list(tmp_path.iterdir()) == []


def test_the_default_still_replaces(tmp_path):
    """Every other writer keeps `os.replace` (the default is unchanged)."""
    from src.utils.file_modes import publish_bytes

    target = tmp_path / "out.json"
    target.write_bytes(b"old")
    publish_bytes(target, b"new")
    assert target.read_bytes() == b"new"
