"""
The measurement CLI's own guards, and the facts it records about a run.
=======================================================================
`scripts/measure_scorer.py` parses arguments, identifies its artifacts, gates
the device and assembles the manifest around one call into
`src.evaluation.measurement`. A number is evidence only with that record beside
it, so each recorded fact is tested where it is produced, apart from the
measurement itself:

- whether the model that ran was a `torch.compile` wrapper, as a tri-state
  that never guesses (`observe_torch_compile_wrapper`);
- the device gate: `auto` refuses to fall back to CPU, and an explicit `cpu`
  run is permitted but recorded as not CUDA;
- that the `file_sha256` re-export is gone (contract M2.1, S9);
- `--split`, which has no default, and the missing-split error;
- which arguments go with which kind of cohort (contract M4, decision 8):
  exactly one of `--split` and `--cohort`; `--output` with `--split` only;
  `--registry` required with `--cohort`; `--registry`, `--training-evidence`
  and `--record` refused with `--split`. Each refusal exits 2 with one message
  before the device is resolved or anything is read.

Which files a measurement records, and that each is the digest of the bytes
read, is pinned at the entry point in `test_measurement_reads_once.py`.

Module: tests/unit/test_measure_scorer_cli.py
"""
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


def test_the_file_sha256_reexport_is_gone():
    """The re-export existed so callers importing a hashing primitive from this
    script kept working; none did but tests. Since contract M2.1's S9 a
    measurement records only the digests of the bytes it read, so the entry point
    binds no path hash at all (the primitive's own tests are in
    `test_training_provenance.py`)."""
    import scripts.measure_scorer as measure_scorer

    assert not hasattr(measure_scorer, "file_sha256")
    assert not hasattr(measure_scorer, "_file_sha256")


@pytest.mark.parametrize("retired", [
    ["--modes", "C"],
    ["--seed", "0"],
    ["--num-workers", "0"],
    ["--predictions-output", "p.json"],
], ids=lambda argv: argv[0])
def test_the_cli_has_no_option_mode_c_would_ignore(retired):
    """Mode C is the only mode, has no dataloader, draws nothing and writes no
    predictions rows. An option kept for any of those would be accepted and
    change nothing, which reads as a setting that took effect."""
    from scripts.measure_scorer import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--checkpoint", "c.pt", "--data-dir", "d", "--split", "val",
                    "--output", "o.json", *retired])


# ==============================================================================
# --split is required, and a missing split says what the workspace has (at the
# reader; a measurement refuses it earlier, at `resolve_cohort`)
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


# ==============================================================================
# Which arguments go with which kind of cohort (contract M4, decision 8)
# ==============================================================================
_BASE = ["--checkpoint", "c.pt", "--data-dir", "d", "--device", "cpu"]


@pytest.mark.parametrize("extra, fragment", [
    (["--split", "val", "--output", "o.json", "--registry", "/r"], "--registry apply to"),
    (["--split", "val", "--output", "o.json", "--training-evidence", "/e"],
     "--training-evidence apply to"),
    (["--split", "val", "--output", "o.json", "--record"], "--record apply to"),
    (["--split", "val", "--output", "o.json", "--record", "--registry", "/r"],
     "--registry, --record apply to"),
    (["--split", "val", "--output", "o.json", "--expect-version-sha256", "a" * 64],
     "--expect-version-sha256 apply to"),
    (["--split", "val", "--output", "o.json", "--outcome", "out.json"],
     "--outcome apply to"),
    (["--cohort", "demo-test-1", "--registry", "/r", "--expect-checkpoint-sha256", "A" * 64],
     "is not a SHA-256"),
    (["--cohort", "demo-test-1", "--registry", "/r", "--expect-version-sha256", "a" * 63],
     "is not a SHA-256"),
    (["--split", "val"], "--split needs --output"),
    (["--cohort", "demo-test-1"], "--cohort needs --registry"),
    (["--cohort", "demo-test-1", "--registry", "/r", "--output", "o.json"],
     "--output does not apply to --cohort"),
    (["--split", "val", "--cohort", "demo-test-1", "--output", "o.json"], "not allowed with"),
    ([], "one of the arguments --split --cohort is required"),
], ids=["split+registry", "split+evidence", "split+record", "split+two",
        "split+expect-version", "split+outcome", "checkpoint-digest-not-hex",
        "version-digest-short", "split-no-output", "cohort-no-registry", "cohort+output",
        "both", "neither"])
def test_an_argument_of_the_other_kind_of_cohort_is_refused_before_anything_runs(
    extra, fragment, capsys, monkeypatch, tmp_path
):
    """Exit 2 with one message, from `main`, before the device is resolved:
    nothing is read and nothing is written. A generated split's report is not
    registered and gets no R10, and a registered version's report goes to the
    report store, not to `--output`."""
    import scripts.measure_scorer as measure_scorer

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(measure_scorer, "_resolve_device",
                        lambda *a: pytest.fail("the device was resolved"))

    with pytest.raises(SystemExit) as refused:
        measure_scorer.main([*_BASE, *extra])

    assert refused.value.code == 2
    assert fragment in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == [], "something was written"


@pytest.mark.parametrize("extra", [
    ["--split", "val", "--output", "o.json"],
    ["--cohort", "demo-test-1", "--registry", "/r"],
    ["--cohort", "demo-test-1", "--registry", "/r", "--record",
     "--training-evidence", "/e1", "--training-evidence", "/e2"],
], ids=["split", "cohort", "cohort-full"])
def test_each_kind_of_cohort_parses_with_its_own_arguments(extra):
    """The counterpart of the refusals: each kind's own arguments parse, and
    `--training-evidence` repeats in the order given."""
    from scripts.measure_scorer import parse_args

    args = parse_args([*_BASE, *extra])

    assert (args.split is None) != (args.cohort is None)
    if args.cohort is not None:
        assert args.output is None
        assert [str(p) for p in args.training_evidence] == (
            ["/e1", "/e2"] if "--training-evidence" in extra else [])
    else:
        assert args.training_evidence == [] and args.record is False


@pytest.mark.parametrize("extra", [
    ["--split", "val", "--output", "o.json"],
    ["--cohort", "demo-test-1", "--registry", "/r", "--expect-version-sha256", "b" * 64,
     "--outcome", "out.json"],
], ids=["split", "cohort"])
def test_the_m5_options_parse_with_the_cohort_they_belong_to(extra):
    """M5-1: `--deterministic` and `--expect-checkpoint-sha256` apply to both
    kinds; `--expect-version-sha256` and `--outcome` to a registered version.
    None has a default that does anything."""
    from scripts.measure_scorer import parse_args

    args = parse_args([*_BASE, *extra, "--deterministic", "--expect-checkpoint-sha256",
                       "a" * 64])

    assert args.deterministic is True and args.expect_checkpoint_sha256 == "a" * 64
    defaults = parse_args([*_BASE, *extra[:4]] if args.cohort else [*_BASE, *extra])
    assert (defaults.deterministic, defaults.expect_checkpoint_sha256,
            defaults.expect_version_sha256, defaults.outcome) == (False, None, None, None)
