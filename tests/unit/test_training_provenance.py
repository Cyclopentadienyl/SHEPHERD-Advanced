"""Proposal A — a checkpoint must be able to say which inputs produced it.

`compute_fingerprint` records the graph's **shape**: node types, node counts,
feature dims, edge types. Its signature takes no samples argument, so no property
of the sample files enters it. A checkpoint therefore carries a structural
compatibility answer and nothing that identifies its training inputs, and an
observer comparing two runs cannot attribute a difference to data rather than to
configuration, randomness or training behaviour.

`training_input_digests` answers the second question and is kept **beside** the
fingerprint rather than folded into it: a structural identity that also changed
whenever a byte moved would stop being a compatibility check.

**Scope, and what these tests deliberately do not cover.** Capture only. There is
no comparison against a current workspace, no registry, no locking and no
transactional loading — `verify_fingerprint` still checks structure alone and says
so. Since contract M2.1's S8 every recorded digest is that of the bytes a reader
parsed (`read_training_inputs`, `resolve_resume_checkpoint`), so the record and
the inputs trained on come from the same reads; the entry-point acceptance for
that is `tests/unit/test_training_reads.py`.

Module: tests/unit/test_training_provenance.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.training.callbacks import ModelCheckpoint, ModelCheckpointConfig  # noqa: E402
from src.utils.fingerprint import (  # noqa: E402
    compute_fingerprint,
    compute_input_digests,
    file_sha256,
    verify_fingerprint,
)


# ---------------------------------------------------------------------------
# The digest primitives
# ---------------------------------------------------------------------------
def test_file_sha256_returns_none_for_an_absent_file(tmp_path):
    """`file_sha256` identifies a file nobody is about to parse, and an absent
    file answers `None` rather than raising. Training does not record through it
    since contract M2.1's S8: its record is built from the reads it parsed."""
    assert file_sha256(tmp_path / "never_written.json") is None


def test_digests_are_keyed_by_role_and_follow_content_not_name(tmp_path):
    """Two files with different names and identical bytes share a digest; the same
    name with different bytes does not. That is the whole claim: same digest means
    the same bytes were consumed."""
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text('{"x": 1}')
    second.write_text('{"x": 1}')

    same = compute_input_digests({"one": first, "two": second})
    assert same["one"] == same["two"]

    second.write_text('{"x": 2}')
    assert compute_input_digests({"two": second})["two"] != same["two"]


def test_the_shared_contract_lives_below_the_scripts_that_use_it():
    """`file_sha256` was defined in `scripts/measure_scorer.py` and imported from
    there by two other scripts and a test — an entry point serving as a library,
    and the SP benchmark in particular had no business depending on the scorer
    measurement to get a hash function.

    The re-export is asserted too, because those callers must keep working.
    """
    from scripts.measure_scorer import file_sha256 as reexported

    assert reexported is file_sha256


# ---------------------------------------------------------------------------
# What a training run records, and what it must not
# ---------------------------------------------------------------------------
def _workspace(tmp_path, *, train_ids=(0, 1), val_ids=(2,), extra_split=None,
               manifest=True):
    """A workspace as the generator leaves one.

    ``manifest`` is a parameter because its **absence** is now a refusal rather
    than a variant: a workspace without one was built before the disease
    allocation step, and training on it is what the refusal exists to stop.

    **Everything is written before the manifest**, because the manifest binds the
    bytes on disk — the graph export as well as the cohorts. Writing a tensor
    afterwards produces exactly the mixed workspace `verify_graph_artifacts`
    exists to refuse, which is the right behaviour and the wrong fixture.
    """
    from tests.fixtures.generated_workspace import (
        one_sample_per_disease,
        profiles_for,
        write_generated_workspace,
    )

    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "kg.json").write_text(json.dumps({"synthetic": tmp_path.name}))
    (tmp_path / "num_nodes.json").write_text(json.dumps({"disease": 2}))
    torch.save({"disease": torch.zeros(2, 4)}, tmp_path / "node_features.pt")
    torch.save({("disease", "x", "disease"): torch.zeros(2, 0, dtype=torch.long)},
               tmp_path / "edge_indices.pt")
    if extra_split:
        (tmp_path / f"{extra_split}_samples.json").write_text("[]")

    if manifest:
        write_generated_workspace(
            tmp_path, train_ids=list(train_ids), val_ids=list(val_ids)
        )
    else:
        profiles = profiles_for(list(train_ids) + list(val_ids))
        for split, ids in (("train", train_ids), ("val", val_ids)):
            (tmp_path / f"{split}_samples.json").write_text(json.dumps(
                one_sample_per_disease(split, list(ids), profiles)
            ))
    return tmp_path


def _training_record(data_dir: Path, *, parent=None):
    """**The production functions, not a copy of them.**

    A first version of this file rebuilt the role map here. That proves the test
    agrees with the test, and it is how the caller's use of an out-of-scope name
    survived a full green run: every assertion below passed while
    `scripts/train_model.py` would have raised `NameError`. The mirror is gone:
    this reads the workspace as `train` does and builds the record as `train`
    does.
    """
    from scripts.train_model import read_training_inputs, training_input_digests

    return training_input_digests(read_training_inputs(data_dir), parent=parent)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_the_recorded_roles_are_the_semantic_inputs_a_run_consumes(tmp_path):
    data_dir = _workspace(tmp_path / "ws")

    digests = _training_record(data_dir)

    assert set(digests) == {
        "train_samples", "val_samples", "node_features", "edge_indices", "num_nodes",
        "split_manifest",
    }
    files = {"train_samples": "train_samples.json", "val_samples": "val_samples.json",
             "node_features": "node_features.pt", "edge_indices": "edge_indices.pt",
             "num_nodes": "num_nodes.json", "split_manifest": "split_manifest.json"}
    for role, name in files.items():
        assert digests[role] == _sha256(data_dir / name), role


def test_the_manifest_is_recorded_as_a_training_input(tmp_path):
    """Its digest belongs in the record, and it is the digest of the one reading
    the checks compared against. Whether it *describes* this workspace is checked
    at the top of `train`, ahead of the run directories and the config file --
    see tests/unit/test_graph_artifact_binding.py, which owns that ordering."""
    from scripts.train_model import read_training_inputs, training_input_digests

    data_dir = _workspace(tmp_path / "ws")

    inputs = read_training_inputs(data_dir)

    assert training_input_digests(inputs)["split_manifest"] == inputs.manifest.identity.sha256
    assert inputs.manifest.identity.sha256 == _sha256(data_dir / "split_manifest.json")


def test_an_unrelated_split_beside_the_inputs_is_not_recorded_or_opened(tmp_path):
    """Nothing globs the data directory. A `test_samples.json` appearing next to
    these files must not change the record of a run that never opened it —
    otherwise the artifact describes the directory rather than the run."""
    from tests.fixtures.opens import count_opens

    data_dir = _workspace(tmp_path / "ws", extra_split="test")

    with count_opens() as opens:
        digests = _training_record(data_dir)

    assert "test_samples" not in digests
    assert opens.opens(data_dir / "test_samples.json") == 0
    assert (data_dir / "test_samples.json").exists(), "the file must really be there"


def test_a_run_without_validation_does_not_claim_a_validation_input(tmp_path):
    """A workspace whose validation split is empty trains without validation, so
    recording the role would claim an input the run's results do not rest on.

    Worded carefully: the empty `val_samples.json` **is** read, verified and
    parsed. The claim is that its samples were not used by a validation pass,
    not that the file was never touched."""
    from scripts.train_model import read_training_inputs

    data_dir = _workspace(tmp_path / "ws", val_ids=())

    inputs = read_training_inputs(data_dir)

    assert inputs.with_validation is False
    assert "val_samples" not in _training_record(data_dir)


def test_the_record_refuses_a_role_with_no_read(tmp_path):
    """A digest that was never taken cannot be recorded as `None`: the record
    names only bytes a reader parsed."""
    from scripts.train_model import read_training_inputs, training_input_digests

    data_dir = _workspace(tmp_path / "ws")
    inputs = read_training_inputs(data_dir)
    reads = dict(inputs.graph.reads)
    del reads["num_nodes"]
    unread = inputs._replace(graph=inputs.graph._replace(reads=reads))

    with pytest.raises(ValueError, match="no read of num_nodes"):
        training_input_digests(unread)


# ---------------------------------------------------------------------------
# The resume parent, which is an input too
# ---------------------------------------------------------------------------
class _RecordingTrainer:
    """Records what `restore_checkpoint` was handed, and nothing else.

    A real `Trainer` would bring a model, an optimizer and a config along for a
    decision that touches none of them."""

    device = torch.device("cpu")

    def __init__(self):
        self.restored = []

    def restore_checkpoint(self, checkpoint):
        self.restored.append(checkpoint)


def test_no_resume_requested_reads_nothing_and_records_nothing():
    """The production decision, not a helper's response to an argument.

    These tests exist because an earlier version built the role map from a path
    or `None` and assumed `train()` chose correctly. That assumption is what the
    earlier mirrored-role-map defect was made of.
    """
    from scripts.train_model import resolve_resume_checkpoint

    trainer = _RecordingTrainer()

    assert resolve_resume_checkpoint(trainer, None) is None
    assert resolve_resume_checkpoint(trainer, "") is None
    assert trainer.restored == []


def test_a_requested_but_missing_parent_warns_restores_nothing_and_records_nothing(tmp_path):
    """`train()` continues after the warning (refusing is M2.3's), so the run did
    not consume the parent, and the record omits it rather than holding a role
    with no digest."""
    from scripts.train_model import resolve_resume_checkpoint

    trainer = _RecordingTrainer()
    absent = tmp_path / "not_here.pt"

    assert resolve_resume_checkpoint(trainer, str(absent)) is None
    assert trainer.restored == [], "a missing parent must not be restored"


def test_the_parent_restored_and_the_identity_returned_come_from_one_read(tmp_path):
    """The returned identity **is** that of the bytes the trainer restored from,
    and the parent is opened once, so recording a parent the run did not restore
    cannot be introduced without editing the function that does both."""
    from scripts.train_model import resolve_resume_checkpoint
    from tests.fixtures.opens import count_opens

    trainer = _RecordingTrainer()
    parent = tmp_path / "parent.pt"
    saved = {"state_dict": {"w": torch.arange(4.0)}, "epoch": 7}
    torch.save(saved, parent)

    with count_opens() as opens:
        returned = resolve_resume_checkpoint(trainer, str(parent))

    assert opens.opens(parent) == 1
    assert returned.sha256 == _sha256(parent)
    assert len(trainer.restored) == 1
    assert trainer.restored[0]["epoch"] == 7
    assert torch.equal(trainer.restored[0]["state_dict"]["w"], saved["state_dict"]["w"])


def test_the_decision_feeds_the_record_end_to_end(tmp_path):
    """The two halves joined: a real decision, then the record built from it."""
    from scripts.train_model import resolve_resume_checkpoint

    data_dir = _workspace(tmp_path / "ws")
    parent = tmp_path / "parent.pt"
    torch.save({"state_dict": {}}, parent)
    trainer = _RecordingTrainer()

    restored = resolve_resume_checkpoint(trainer, str(parent))
    assert _training_record(data_dir, parent=restored)["resume_checkpoint"] == _sha256(parent)

    unrestored = resolve_resume_checkpoint(trainer, str(tmp_path / "gone.pt"))
    assert "resume_checkpoint" not in _training_record(data_dir, parent=unrestored)


def test_two_parents_with_different_bytes_are_distinguishable(tmp_path):
    """The case the field exists for: identical workspace files, different
    parents. Without the parent role these two runs carried the same record."""
    from scripts.train_model import resolve_resume_checkpoint

    data_dir = _workspace(tmp_path / "ws")
    first, second = tmp_path / "a.pt", tmp_path / "b.pt"
    torch.save({"state_dict": {"w": torch.zeros(1)}}, first)
    torch.save({"state_dict": {"w": torch.ones(1)}}, second)

    def record_for(parent):
        restored = resolve_resume_checkpoint(_RecordingTrainer(), str(parent))
        return _training_record(data_dir, parent=restored)

    a, b = record_for(first), record_for(second)

    assert a["resume_checkpoint"] != b["resume_checkpoint"]
    assert a["train_samples"] == b["train_samples"], "only the parent differs"


# ---------------------------------------------------------------------------
# The acceptance case: attribution
# ---------------------------------------------------------------------------
def test_two_runs_over_different_samples_are_distinguishable_from_provenance(tmp_path):
    """**The defect this proposal exists to close.**

    Both workspaces have the same graph, so their structural fingerprints agree —
    that is correct and unchanged, since the graph really is the same. What was
    missing is any record able to say the sample files differed. Without it an
    observed difference between two checkpoints could be data, configuration,
    randomness or training behaviour, and nothing could separate them.
    """
    graph = {
        "x_dict": {"disease": torch.zeros(2, 4)},
        "edge_index_dict": {("disease", "x", "disease"): torch.zeros(2, 0, dtype=torch.long)},
        "num_nodes_dict": {"disease": 2},
    }
    first = _workspace(tmp_path / "a")
    second = _workspace(tmp_path / "b", train_ids=(5, 6), val_ids=(7,))

    # Two independently constructed but equivalent graphs. A first version
    # compared `compute_fingerprint(graph)` with itself, which is true of any
    # function and is evidence of nothing.
    equivalent = {
        "x_dict": {"disease": torch.zeros(2, 4)},
        "edge_index_dict": {("disease", "x", "disease"): torch.zeros(2, 0, dtype=torch.long)},
        "num_nodes_dict": {"disease": 2},
    }
    assert compute_fingerprint(graph) == compute_fingerprint(equivalent), (
        "the structural fingerprint must still see these two graphs as the same"
    )

    a = _training_record(first)
    b = _training_record(second)

    assert a["train_samples"] != b["train_samples"]
    assert a["node_features"] == b["node_features"], (
        "only the samples differ; the graph inputs must still match"
    )


# ---------------------------------------------------------------------------
# The checkpoint writer
# ---------------------------------------------------------------------------
class _StubModel:
    def state_dict(self):
        return {"w": torch.zeros(1)}


class _StubTrainer:
    """Only what `_save_checkpoint` reads. A real `Trainer` would drag an
    optimizer, a scheduler and a config in for no benefit to this contract."""

    def __init__(self, **attrs):
        self.model = _StubModel()
        self.optimizer = None
        self.scheduler = None
        for name, value in attrs.items():
            setattr(self, name, value)

    def _serialize_config(self):
        return {}


def _write(tmp_path, *, weights_only, **trainer_attrs):
    callback = ModelCheckpoint(ModelCheckpointConfig(
        dirpath=str(tmp_path), save_weights_only=weights_only,
    ))
    trainer = _StubTrainer(**trainer_attrs)
    trainer.optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    path = tmp_path / "ckpt.pt"
    callback._save_checkpoint(trainer, path, epoch=1, logs={})
    return torch.load(path, map_location="cpu", weights_only=False)


@pytest.mark.parametrize("weights_only", [False, True])
def test_both_checkpoint_shapes_carry_the_digests(tmp_path, weights_only):
    """Placed after the weights-only branch on purpose. A weights-only checkpoint
    is still an artifact somebody must later identify, and the questions it needs
    answered do not stop mattering because the optimizer state was left out."""
    loaded = _write(
        tmp_path, weights_only=weights_only,
        data_fingerprint={"node_types": ["disease"]},
        training_input_digests={"train_samples": "abc"},
    )

    assert loaded["training_input_digests"] == {"train_samples": "abc"}
    assert loaded["data_fingerprint"] == {"node_types": ["disease"]}


def test_the_digests_are_a_sibling_field_not_folded_into_the_fingerprint(tmp_path):
    """Structure and content identity stay separate. A fingerprint that also
    changed whenever a byte moved would stop being a compatibility check."""
    loaded = _write(
        tmp_path, weights_only=False,
        data_fingerprint={"node_types": ["disease"]},
        training_input_digests={"train_samples": "abc"},
    )

    assert "training_input_digests" not in loaded["data_fingerprint"]
    assert loaded["data_fingerprint"] == {"node_types": ["disease"]}


def test_a_trainer_carrying_no_digests_writes_a_checkpoint_without_the_key(tmp_path):
    """Legacy behaviour, and the reason the callback tests for the attribute. A
    trainer built outside `scripts/train_model.py` has never set it, and that must
    produce a checkpoint without the key rather than a crash or a null claim."""
    loaded = _write(tmp_path, weights_only=False, data_fingerprint={"node_types": []})

    assert "training_input_digests" not in loaded


# ---------------------------------------------------------------------------
# What was deliberately not built
# ---------------------------------------------------------------------------
def test_the_structural_verifier_does_not_pretend_to_check_digests():
    """An empty warning list means "structurally compatible", never "trained on
    these files". Comparison against a current workspace is deferred, and the
    docstring says so — this asserts the behaviour rather than trusting the prose.
    """
    graph = {
        "x_dict": {"disease": torch.zeros(2, 4)},
        "edge_index_dict": {("disease", "x", "disease"): torch.zeros(2, 0, dtype=torch.long)},
        "num_nodes_dict": {"disease": 2},
    }
    checkpoint = {
        "data_fingerprint": compute_fingerprint(graph),
        "training_input_digests": {"train_samples": "a digest from another machine"},
    }

    assert verify_fingerprint(checkpoint, graph) == []


def test_a_checkpoint_predating_the_field_still_verifies_structurally():
    """Absence of a digest map means "not recorded" and nothing more."""
    graph = {
        "x_dict": {"disease": torch.zeros(2, 4)},
        "edge_index_dict": {("disease", "x", "disease"): torch.zeros(2, 0, dtype=torch.long)},
        "num_nodes_dict": {"disease": 2},
    }

    assert verify_fingerprint({"data_fingerprint": compute_fingerprint(graph)}, graph) == []


#: Files this gate covers, **repo-relative**; the test resolves them against the
#: repository root rather than the caller's cwd.
#: `src/training/callbacks.py` is excluded: it carries 33 pre-existing
#: `Undefined name 'Trainer'` findings from quoted forward references without a
#: `TYPE_CHECKING` import, none of them from this change and none of them this
#: item's to fix.
#:
#: **The SP files are here for a hazard, not for tidiness.** Backlog item 5a
#: moved the shortest-path primitive into `src/inference/sp_index.py` and left an
#: independent reference in `scripts/sp_scan_reference.py`. Both carry
#: `from __future__ import annotations`, which makes a dataclass field annotated
#: with a name that is no longer in scope construct *and instantiate* without
#: error — measured, on the file this replaced: the module imported and all 28 of
#: its tests passed with the annotation dangling. `make check` runs
#: `lint-imports` and the tests rather than `make lint`, so F821 — the one check
#: in the default gate that sees it — was never reached. Listing them costs one
#: ruff invocation.
_F821_CLEAN = (
    "scripts/train_model.py",
    "scripts/measure_scorer.py",
    "scripts/benchmark_sp_lookup.py",
    "scripts/sp_scan_reference.py",
    "scripts/audit_sp_reachability.py",
    "src/inference/sp_index.py",
    "src/inference/sp_artifact.py",
    "src/utils/fingerprint.py",
    # Contract M2.1, S8: the resume read and its restore moved between these.
    "src/training/trainer.py",
    "src/core/protocols.py",
    "tests/unit/test_training_provenance.py",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_the_touched_files_reference_no_undefined_names():
    """The failure the mirrored helper hid, checked by the tool that already does
    this correctly.

    `val_samples` is bound in `create_dataloaders` while the provenance wiring
    lives in `train`. Reading it there is a `NameError` at runtime — and every
    test in this file passed anyway, because the first version rebuilt the role
    map beside the source instead of calling it.

    **This does not hand-roll scope analysis.** A first attempt did, and got it
    wrong: `ruff` is configured with the `F` family in `pyproject.toml`, F821 is
    in it, and it reports exactly this defect at exactly the right line. What was
    missing is that `make check` runs `lint-imports` and the unit tests, **not**
    `make lint`, so nothing in the default gate ran it. This closes that gap for
    the files this work touched, and nothing wider.

    **Fail-closed, which the first version was not.** Ruff warns on a target it
    cannot find and still exits 0 — verified: `ruff check /definitely/not/here`
    prints "All checks passed!" and returns 0. Passing repo-relative paths without
    a cwd therefore meant a run from anywhere else linted nothing and passed. So
    the paths are resolved against the repository root, each is asserted to exist
    **before** ruff is invoked, and only a zero return code passes.

    Ruff's absence is detected as absence, not inferred from a return code: an
    exit status outside {0, 1} is a configuration, CLI or I/O failure and must
    fail the test rather than quietly skip it.
    """
    import importlib.util
    import subprocess
    import sys

    if importlib.util.find_spec("ruff") is None:
        pytest.skip("ruff is not installed in this environment")

    targets = [_REPO_ROOT / name for name in _F821_CLEAN]
    missing = [str(path) for path in targets if not path.exists()]
    assert not missing, f"lint targets do not exist, so linting them proves nothing: {missing}"

    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select", "F821",
         "--output-format", "concise", *[str(path) for path in targets]],
        capture_output=True, text=True, cwd=_REPO_ROOT,
    )

    assert result.returncode == 0, (
        f"ruff exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
