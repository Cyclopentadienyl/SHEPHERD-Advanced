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
so. The digest is taken at a different instant from any load, which is documented
as a limitation rather than engineered around.

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
def test_a_missing_file_is_recorded_as_absent_rather_than_raised(tmp_path):
    """A role a run did not consume and a file that is genuinely gone read the
    same way, and neither stops a training run to report itself."""
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
def _workspace(tmp_path, *, samples_payload, extra_split=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "train_samples.json").write_text(json.dumps(samples_payload))
    (tmp_path / "val_samples.json").write_text(json.dumps(samples_payload))
    (tmp_path / "num_nodes.json").write_text(json.dumps({"disease": 2}))
    torch.save({"disease": torch.zeros(2, 4)}, tmp_path / "node_features.pt")
    torch.save({("disease", "x", "disease"): torch.zeros(2, 0, dtype=torch.long)},
               tmp_path / "edge_indices.pt")
    if extra_split:
        (tmp_path / f"{extra_split}_samples.json").write_text("[]")
    return tmp_path


def _training_roles(data_dir: Path, *, with_val: bool):
    """**The production function, not a copy of it.**

    A first version of this file rebuilt the same dict here. That proves the test
    agrees with the test, and it is how the caller's use of an out-of-scope name
    survived a full green run: every assertion below passed while
    `scripts/train_model.py` would have raised `NameError`. The mirror is gone.
    """
    from scripts.train_model import training_input_roles

    return training_input_roles(data_dir, with_validation=with_val)


def test_the_recorded_roles_are_the_semantic_inputs_a_run_consumes(tmp_path):
    data_dir = _workspace(tmp_path / "ws", samples_payload=[{"patient_id": "p0"}])

    digests = compute_input_digests(_training_roles(data_dir, with_val=True))

    assert set(digests) == {
        "train_samples", "val_samples", "node_features", "edge_indices", "num_nodes",
    }
    assert all(value is not None for value in digests.values())


def test_an_unrelated_split_beside_the_inputs_is_not_recorded(tmp_path):
    """Nothing globs the data directory. A `test_samples.json` appearing next to
    these files must not change the record of a run that never opened it —
    otherwise the artifact describes the directory rather than the run."""
    data_dir = _workspace(
        tmp_path / "ws", samples_payload=[{"patient_id": "p0"}], extra_split="test",
    )

    digests = compute_input_digests(_training_roles(data_dir, with_val=True))

    assert "test_samples" not in digests
    assert (data_dir / "test_samples.json").exists(), "the file must really be there"


def test_a_run_without_validation_does_not_claim_a_validation_input(tmp_path):
    """`scripts/train_model.py` loads val conditionally and trains without it when
    absent, so recording the role unconditionally would claim an input the run's
    results do not rest on.

    Worded carefully: an existing-but-empty `val_samples.json` **is** opened and
    parsed before the loader becomes `None`. The claim is that the samples were
    not used by a validation pass, not that the file was never touched — and the
    file is not added to the role map merely because it was probed."""
    data_dir = _workspace(tmp_path / "ws", samples_payload=[])

    digests = compute_input_digests(_training_roles(data_dir, with_val=False))

    assert "val_samples" not in digests


# ---------------------------------------------------------------------------
# The resume parent, which is an input too
# ---------------------------------------------------------------------------
def test_a_run_that_resumes_nothing_records_no_parent(tmp_path):
    data_dir = _workspace(tmp_path / "ws", samples_payload=[])

    assert "resume_checkpoint" not in _training_roles(data_dir, with_val=False)


def test_a_loaded_parent_checkpoint_is_recorded(tmp_path):
    """A resumed run restores weights, optimizer, scheduler, scaler and training
    state from its parent, so the parent is an input its results rest on."""
    from scripts.train_model import training_input_roles

    data_dir = _workspace(tmp_path / "ws", samples_payload=[])
    parent = tmp_path / "parent.pt"
    torch.save({"state_dict": {}}, parent)

    roles = training_input_roles(data_dir, with_validation=False, resumed_from=parent)

    assert roles["resume_checkpoint"] == parent
    assert compute_input_digests(roles)["resume_checkpoint"] is not None


def test_a_requested_but_missing_parent_is_omitted_rather_than_recorded_as_none(tmp_path):
    """`train()` warns and continues when the resume path does not exist, so the
    run did not consume it.

    Omission and a present role with a `None` digest are different statements:
    `None` already means "this role's file was absent" for a role the run *did*
    consume, and overloading it would lose that distinction.
    """
    from scripts.train_model import training_input_roles

    data_dir = _workspace(tmp_path / "ws", samples_payload=[])

    # This is what `train()` passes when `resume_path.exists()` was False.
    roles = training_input_roles(data_dir, with_validation=False, resumed_from=None)

    assert "resume_checkpoint" not in roles


def test_two_parents_with_different_bytes_are_distinguishable(tmp_path):
    """The case the field exists for: identical workspace files, different
    parents. Without the parent role these two runs carried the same digest map."""
    from scripts.train_model import training_input_roles

    data_dir = _workspace(tmp_path / "ws", samples_payload=[{"patient_id": "p0"}])
    first, second = tmp_path / "a.pt", tmp_path / "b.pt"
    torch.save({"state_dict": {"w": torch.zeros(1)}}, first)
    torch.save({"state_dict": {"w": torch.ones(1)}}, second)

    a = compute_input_digests(
        training_input_roles(data_dir, with_validation=False, resumed_from=first))
    b = compute_input_digests(
        training_input_roles(data_dir, with_validation=False, resumed_from=second))

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
    first = _workspace(tmp_path / "a", samples_payload=[{"patient_id": "p0"}])
    second = _workspace(tmp_path / "b", samples_payload=[{"patient_id": "p1"}])

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

    a = compute_input_digests(_training_roles(first, with_val=True))
    b = compute_input_digests(_training_roles(second, with_val=True))

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


#: Files this work touched that are F821-clean, **repo-relative**; the test
#: resolves them against the repository root rather than the caller's cwd.
#: `src/training/callbacks.py` is excluded: it carries 33 pre-existing
#: `Undefined name 'Trainer'` findings from quoted forward references without a
#: `TYPE_CHECKING` import, none of them from this change and none of them this
#: item's to fix.
_F821_CLEAN = (
    "scripts/train_model.py",
    "scripts/measure_scorer.py",
    "src/utils/fingerprint.py",
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
