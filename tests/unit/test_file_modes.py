"""
Files published by rename get the mode a plain open() would give them.

The publishers write a staging file and rename it into place. With `tempfile`
that staging file was 0600, so the SP pair, the split manifest, provenance,
the evaluation ledger and the UI config all published 0600 -- including over
files that had been 0644. `create_staging_file` restores open()'s behaviour: a
new file gets the directory's policy, applied by the kernel; a rewritten file
keeps its mode.

**The reference for "what open() would do" is open() itself**, run in the same
directory under the same umask, so the tests hold under whatever policy the
host applies -- umask or default ACL -- rather than restating a formula.
"""
import asyncio
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
def _sp_pair(directory):
    import torch

    from src.inference.sp_artifact import publish_sp_artifact, sidecar_path

    target = directory / "shortest_paths.pt"
    publish_sp_artifact({"distance": torch.zeros(1, dtype=torch.int8)}, target, {"max_hops": 5})
    return [target, sidecar_path(target)]


def _split_manifest(directory):
    from scripts.setup_demo import build_demo_kg
    from src.kg.artifacts import GRAPH_ARTIFACTS
    from src.kg.disease_allocation import allocate_diseases
    from src.kg.sample_generator import build_eligible_disease_profiles, generate_training_samples
    from src.utils.fingerprint import file_sha256
    from tests.fixtures.generated_workspace import default_graph_export

    for role, filename in GRAPH_ARTIFACTS.items():
        if not (directory / filename).exists():
            (directory / filename).write_bytes(role.encode())
    kg = build_demo_kg()
    generate_training_samples(
        kg=kg,
        allocation=allocate_diseases(build_eligible_disease_profiles(kg, 2), 0.2, seed=42),
        num_train=20, num_val=5, output_dir=directory,
        graph_digests={role: file_sha256(directory / filename)
                       for role, filename in GRAPH_ARTIFACTS.items()},
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


def _ui_config(directory):
    pytest.importorskip("fastapi")
    from src.api.routes import pipeline

    target = directory / ".shepherd_ui_config.json"
    real = pipeline.CONFIG_FILE
    pipeline.CONFIG_FILE = target
    try:
        asyncio.run(pipeline.save_ui_config(pipeline.UIConfigResponse()))
    finally:
        pipeline.CONFIG_FILE = real
    return [target]


PUBLISHERS = {
    "sp_pair": _sp_pair,
    "split_manifest": _split_manifest,
    "provenance": _provenance,
    "evaluation_ledger": _evaluation_ledger,
    "ui_config": _ui_config,
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
