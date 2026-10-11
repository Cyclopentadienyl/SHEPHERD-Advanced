"""
The measurement CLI's own guards, and the facts it records about a run.
=======================================================================
`scripts/measure_scorer.py` parses arguments, identifies its artifacts, gates
the device and assembles the manifest around one call into
`src.evaluation.measurement`. A number is evidence only with that record beside
it, so each recorded fact is tested where it is produced, independent of which
measurement mode then runs:

- whether the model that ran was a `torch.compile` wrapper, as a tri-state
  that never guesses (`observe_torch_compile_wrapper`);
- the device gate: `auto` refuses to fall back to CPU, and an explicit `cpu`
  run is permitted but recorded as not CUDA;
- artifact identity by content, and which files a measurement is bound to;
- the seed, validated at the CLI and applied before anything consumes it;
- `--split`, which has no default, and the missing-split error.

Module: tests/unit/test_measure_scorer_cli.py
"""
import json
import sys

import pytest

torch = pytest.importorskip("torch")

# ---------------------------------------------------------------------------
# Observing the compile wrapper — isinstance, and the tri-state around it
# ---------------------------------------------------------------------------
#
# `isinstance` against the imported wrapper class is the only evidence used. Two
# earlier versions accepted `_orig_mod` as a fallback signal and both were wrong:
# any object may carry that attribute name, and pairing it with `__module__` did
# not fix it, since `__module__` is assignable class metadata rather than
# provenance. When the class cannot be imported, nothing was observed and the
# answer is `None`.


def _block_the_exact_probe(monkeypatch):
    """Make both of the observer's import paths fail.

    `sys.modules[name] = None` is how the import system is told a module is
    absent; verified by execution to raise `ModuleNotFoundError`. This stands in
    for the real risk, which is torch moving these names between versions.

    Both, because the observer aggregates every path that resolves — blocking one
    leaves the observation exact, which is the point of having two and is tested
    on its own below.
    """
    monkeypatch.setitem(sys.modules, "torch._dynamo", None)
    monkeypatch.setitem(sys.modules, "torch._dynamo.eval_frame", None)


def test_an_exact_positive_is_observed():
    """Against a real `torch.compile` wrapper, not a stand-in carrying the right
    attribute — a stand-in would test the test."""
    import torch.nn as nn

    from src.evaluation.measurement import observe_torch_compile_wrapper

    assert observe_torch_compile_wrapper(torch.compile(nn.Linear(4, 4))) is True


def test_an_exact_negative_is_observed():
    import torch.nn as nn

    from src.evaluation.measurement import observe_torch_compile_wrapper

    assert observe_torch_compile_wrapper(nn.Linear(4, 4)) is False


def test_no_model_is_not_observed_rather_than_not_compiled():
    """`None` and `False` are different claims and must not be flattened."""
    from src.evaluation.measurement import observe_torch_compile_wrapper

    assert observe_torch_compile_wrapper(None) is None


def test_an_object_carrying_only_the_wrapper_attribute_is_not_a_wrapper():
    """`_orig_mod` is a **necessary** marker of the known wrapper, never sufficient
    evidence of wrapper identity: any object may legally carry that attribute name.

    An earlier version promoted the attribute to `True`, and the test written for
    it asserted this very object *was* a wrapper — freezing a false positive as
    though it were a specification.
    """
    from src.evaluation.measurement import observe_torch_compile_wrapper

    class NotAWrapper:
        _orig_mod = object()

    assert observe_torch_compile_wrapper(NotAWrapper()) is False


def test_the_package_level_path_keeps_the_probe_exact_on_its_own(monkeypatch):
    """Two import paths, so a torch release that moves one still leaves the
    observation exact rather than dropping it to `None`."""
    import torch.nn as nn

    from src.evaluation.measurement import observe_torch_compile_wrapper

    monkeypatch.setitem(sys.modules, "torch._dynamo.eval_frame", None)

    assert observe_torch_compile_wrapper(nn.Linear(4, 4)) is False


def test_no_resolvable_wrapper_class_is_not_observed(monkeypatch):
    """The subject is a **real** compiled wrapper on purpose.

    Even a genuine true positive becomes `None` once the classes cannot be
    imported, because there is then no evidence — only guesses that an object
    could arrange to satisfy. That is the whole reason the field is
    `Optional[bool]` and not a bool with a default.
    """
    import torch.nn as nn

    from src.evaluation.measurement import observe_torch_compile_wrapper

    wrapped = torch.compile(nn.Linear(4, 4))
    _block_the_exact_probe(monkeypatch)

    assert observe_torch_compile_wrapper(wrapped) is None


def test_an_object_that_declares_torch_provenance_is_still_not_observed(monkeypatch):
    """The counterexample that removed the fallback, pinned so it cannot return.

    A second version of this observer accepted `_orig_mod` **plus** a class whose
    `__module__` began with `torch._dynamo`. It failed twice: `startswith` also
    matches `torch._dynamoevil`, and `__module__` is assignable class metadata —
    any object can declare it, so it describes what a class says about itself, not
    how it was built. This object satisfies both of those weak signals and was
    never touched by `torch.compile`.
    """
    from src.evaluation.measurement import observe_torch_compile_wrapper

    class Impostor:
        __module__ = "torch._dynamo.fake"
        _orig_mod = object()

    _block_the_exact_probe(monkeypatch)

    assert observe_torch_compile_wrapper(Impostor()) is None


def test_hostile_attribute_access_does_not_raise():
    """The observer says what a run was, so it must return data rather than a
    traceback — a manifest field is not worth killing a measurement for.

    With the attribute fallback gone the observer reads no attributes at all, so
    this now pins the contract end to end rather than one probe inside it: an
    object whose `__getattr__` raises comes back as an answer.
    """
    from src.evaluation.measurement import observe_torch_compile_wrapper

    class Hostile:
        def __getattr__(self, name):
            raise RuntimeError("attribute access is hostile")

    assert observe_torch_compile_wrapper(Hostile()) is False


# ---------------------------------------------------------------------------
# The CLI's own guards — argument-level, not driver-level
# ---------------------------------------------------------------------------
def test_auto_device_refuses_to_fall_back_to_cpu(monkeypatch):
    """CUDA is a hard project requirement. A silent CPU fallback would produce a
    number that reads as institutional and is not."""
    from scripts import measure_scorer

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(SystemExit, match="requires CUDA"):
        measure_scorer._resolve_device("auto")


def test_explicit_cpu_is_permitted_but_ineligible(monkeypatch):
    """Development in a container without a GPU is real work; quoting its number
    as an acceptance result is not."""
    from scripts import measure_scorer

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device, eligible = measure_scorer._resolve_device("cpu")

    assert device.type == "cpu"
    assert eligible is False


def test_artifact_digests_identify_content_not_paths(tmp_path):
    """Two files with identical content hash the same however they are named; a
    one-byte change does not."""
    import hashlib

    from scripts.measure_scorer import file_sha256

    original = tmp_path / "a.bin"
    copy = tmp_path / "b.bin"
    altered = tmp_path / "c.bin"
    original.write_bytes(b"shepherd")
    copy.write_bytes(b"shepherd")
    altered.write_bytes(b"shepherE")

    assert file_sha256(original) == hashlib.sha256(b"shepherd").hexdigest()
    assert file_sha256(original) == file_sha256(copy)
    assert file_sha256(original) != file_sha256(altered)
    assert file_sha256(tmp_path / "absent.bin") is None


class TestTheRandomStreamHasAnIdentity:
    """An unseeded measurement is not reproducible, so it cannot be evidence.

    `--seed` defaulted to `None` and the RNGs were seeded only when a value was
    supplied, so a default run wrote three nulls into the manifest. Two such runs
    consumed different worker streams -- PyTorch seeds each worker as
    `base_seed + worker_id` -- different negatives and a different candidate
    universe, while hashing to the same measurement semantics. The ledger then saw
    one measurement with two answers and refused the second as a contradiction.
    """

    def test_the_default_run_is_seeded_and_records_the_applied_value(self):
        from scripts.measure_scorer import DEFAULT_MEASUREMENT_SEED, parse_args

        args = parse_args(["--checkpoint", "c.pt", "--data-dir", "d",
                           "--split", "val", "--output", "o.json"])

        assert args.seed == DEFAULT_MEASUREMENT_SEED
        assert isinstance(args.seed, int) and not isinstance(args.seed, bool)

    @pytest.mark.parametrize("bad", [-1, 2 ** 32])
    def test_the_cli_refuses_a_seed_outside_its_domain(self, bad):
        """Checked at the CLI as well as at the ledger: neither boundary can rely
        on the other having run, and a value NumPy's seeder rejects would seed
        nothing usable here. A non-integer never reaches this check -- argparse's
        `type=int` is the right layer for that, and the next case pins it."""
        from scripts.measure_scorer import main

        with pytest.raises(ValueError, match="must be an integer in"):
            main(["--checkpoint", "c.pt", "--data-dir", "d", "--split", "val",
                  "--output", "o.json", "--seed", str(bad)])

    def test_a_non_integer_seed_is_refused_by_argparse(self):
        from scripts.measure_scorer import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--checkpoint", "c.pt", "--data-dir", "d", "--split", "val",
                        "--output", "o.json", "--seed", "not-a-number"])

    @pytest.mark.parametrize("good", [0, 2 ** 32 - 1])
    def test_both_ends_of_the_seed_domain_are_accepted(self, monkeypatch, good):
        """The domain check must not reject the ends of the domain it enforces,
        and the value it accepts is the one applied. The run is stopped at device
        resolution, which comes after validation and seeding; the three generators'
        state there must be the state that seed gives."""
        import random

        import numpy as np

        import scripts.measure_scorer as cli
        from src.evaluation.measurement import MAX_MEASUREMENT_SEED

        assert good in (0, MAX_MEASUREMENT_SEED)

        class ReachedDeviceResolutionError(Exception):
            pass

        seen = {}

        def stop(*args, **kwargs):
            seen["torch"] = torch.initial_seed()
            seen["numpy"] = int(np.random.get_state()[1][0])
            seen["python"] = random.getstate() == random.Random(good).getstate()
            raise ReachedDeviceResolutionError

        monkeypatch.setattr(cli, "_resolve_device", stop)

        with pytest.raises(ReachedDeviceResolutionError):
            cli.main(["--checkpoint", "c.pt", "--data-dir", "d", "--split", "val",
                      "--output", "o.json", "--seed", str(good)])

        assert seen["torch"] == good
        assert seen["numpy"] == good
        assert seen["python"] is True


class TestTheRolesAMeasurementRecords:
    """Which files a number is bound to, and the one that says what `val` means.

    A `val_mrr` under a disease-disjoint cut measures generalisation to diseases
    with no labelled examples. Under a sample-level slice it measures recognition
    of new phenotype subsets of diseases that have them. Two different quantities
    under one name, and nothing in `val_samples.json`'s digest separates them --
    only `split_manifest.json` does.
    """

    @staticmethod
    def _workspace(root, *, manifest: bool = True, supplied: bool = False):
        from tests.fixtures.generated_workspace import (
            one_sample_per_disease,
            profiles_for,
            write_generated_workspace,
        )

        root.mkdir(parents=True, exist_ok=True)
        # The graph files are the shared fixture's real export, which the
        # manifest binds. A workspace with no manifest is refused before any of
        # them is read, so it gets none.
        profiles = profiles_for([0, 1, 2])
        if manifest:
            write_generated_workspace(root, train_ids=[0, 1], val_ids=[2],
                                      profiles=profiles)
        else:
            for split, ids in (("train", [0, 1]), ("val", [2])):
                (root / f"{split}_samples.json").write_text(json.dumps(
                    one_sample_per_disease(split, ids, profiles)
                ))
        if supplied:
            (root / "test_samples.json").write_text(json.dumps(
                one_sample_per_disease("test", [0], profiles)
            ))
        (root / "ckpt.pt").write_bytes(b"weights")
        return root

    def test_a_generated_cohort_records_the_cut_it_came_from(self, tmp_path):
        from scripts.measure_scorer import artifact_digests

        root = self._workspace(tmp_path / "ws")
        digests = artifact_digests(root / "ckpt.pt", root, "val", "generated")

        assert digests["split_manifest"] is not None
        assert digests["split_manifest"] != digests["samples"]

    def test_a_supplied_cohort_records_no_allocation(self, tmp_path):
        """Not a missing file: nobody cut it from this disease universe, so there
        is no allocation for it to name."""
        from scripts.measure_scorer import artifact_digests

        root = self._workspace(tmp_path / "ws", supplied=True)
        digests = artifact_digests(root / "ckpt.pt", root, "test", "supplied")

        assert "split_manifest" not in digests
        assert (root / "split_manifest.json").exists(), (
            "the workspace's own manifest is there and still does not apply"
        )

    def test_a_pre_allocation_workspace_cannot_be_measured(self, tmp_path):
        """Its cohorts overlap, so a val number from it measures something else.
        Refused rather than recorded with a null."""
        from scripts.measure_scorer import artifact_digests

        root = self._workspace(tmp_path / "ws", manifest=False)
        with pytest.raises(ValueError, match="generated before the disease allocation"):
            artifact_digests(root / "ckpt.pt", root, "val", "generated")

    def test_nothing_else_in_the_directory_becomes_a_role(self, tmp_path):
        """A record of the directory is not a record of the run."""
        from scripts.measure_scorer import artifact_digests

        root = self._workspace(tmp_path / "ws")
        (root / "notes.txt").write_bytes(b"scratch")
        (root / "mygene2_samples.json").write_bytes(b"[]")
        digests = artifact_digests(root / "ckpt.pt", root, "val", "generated")

        assert set(digests) == {
            "checkpoint", "samples", "node_features", "edge_indices", "num_nodes",
            "split_manifest",
        }


# ==============================================================================
# --split is required, and a missing split says what the workspace has
# ==============================================================================
def test_split_has_no_default_on_the_measurement_entry_point():
    """It defaulted to `test`, which the generator never writes.

    `src/kg/sample_generator.py` produces train and val only, so the default named
    a file no ordinary workspace contains. Requiring the flag is deliberate rather
    than switching the default to `val`: `val` is the checkpoint-selection split,
    and a default would let a caller measure on it without ever deciding to.
    """
    import scripts.measure_scorer as measure

    with pytest.raises(SystemExit):
        measure.parse_args(["--checkpoint", "c.pt", "--data-dir", "d", "--output", "o.json"])


def test_missing_split_error_lists_what_the_workspace_actually_has(tmp_path):
    from src.kg.storage.file_storage import read_samples

    (tmp_path / "train_samples.json").write_text("[]")
    (tmp_path / "val_samples.json").write_text("[]")

    with pytest.raises(FileNotFoundError, match=r"this workspace has: train, val"):
        read_samples(tmp_path, "test")


def test_missing_split_error_when_nothing_is_there(tmp_path):
    from src.kg.storage.file_storage import read_samples

    with pytest.raises(FileNotFoundError, match="no \\*_samples.json files at all"):
        read_samples(tmp_path, "val")
