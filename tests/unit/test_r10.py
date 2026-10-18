"""
R10 as contract §5.4 defines it, at the function level (M4, spec §0.16, §3).
============================================================================
`evidence_candidates` lists where to look; `compute_r10` finds the bytes the
checkpoint names among them, by digest, and counts two labels per scope. This
file covers every row of §5.4's acceptance list that a function can show:

- the current split is disjoint, but the model was trained on another one: the
  current file's digest differs, so it is not used; with the real evidence
  absent the labels are *unverifiable*, with it supplied the overlap is counted;
- evidence moved with its bytes intact is found through an evidence directory;
- evidence missing or altered is *unverifiable*, never zero;
- disjoint phenotype sets with overlapping diseases: phenotype-set *none*,
  disease *overlap* with its counts;
- a training case `[0, 1]` and a test case `[0, 0, 1]` of one disease are a
  phenotype-set *overlap*; evidence that itself lists a phenotype twice is
  refused by the shared reader (N2), so its scope is *unverifiable*, not counted.

(The resume row, "a resume over other sample files is refused at training", is
M2.3's, at training; it has no R10 function to test.)

And the states and the reader: the four states, `k`/`cases` `None` whenever a
label is not *overlap*/*none*; no `val_samples` entry means *no validation
inputs recorded*, and one that is there and is not a digest leaves validation
*unverifiable*; neither file recorded leaves both scopes *unverifiable*;
matching evidence that does not parse is *unverifiable* with a problem that
holds no row data; a candidate that is not even JSON is skipped without a parse
error; each candidate is opened once even while both scopes are unmatched; the
raw buffers and the matching samples are released before the next read; an
evidence directory that is not a directory, or cannot be listed, is refused.

Module: tests/unit/test_r10.py
"""
from __future__ import annotations

import hashlib
import json
import os
import weakref

import pytest

from src.evaluation import r10
from src.evaluation.r10 import (
    EVIDENCE_STATES,
    R10_DEFINITION_VERSION,
    R10_LABELS,
    R10_STATES,
    compute_r10,
    evidence_candidates,
)
from src.kg.data_loader import DiagnosisSample
from tests.fixtures.opens import count_opens
from tests.fixtures.release import check_release


def _s(patient, phenotypes, disease):
    return DiagnosisSample(patient_id=patient, phenotype_ids=list(phenotypes), disease_id=disease)


def _write(path, samples, **dumps):
    """Write `samples` as a samples file (a JSON list of rows); return its digest."""
    rows = [
        {"patient_id": s.patient_id, "phenotype_ids": s.phenotype_ids, "disease_id": s.disease_id}
        for s in samples
    ]
    data = json.dumps(rows, **dumps).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _assert_well_formed(block):
    """The §3.2 shape, and the rule every block keeps: a count only where the
    state is *overlap* or *none*, `None` (never 0) otherwise."""
    assert set(block) == {"definition_version", "candidates_examined", "training", "validation"}
    assert block["definition_version"] == R10_DEFINITION_VERSION
    for scope in ("training", "validation"):
        part = block[scope]
        assert set(part) == {"recorded_sha256", "evidence", *R10_LABELS}
        assert set(part["evidence"]) == {"state", "path", "problem"}
        assert part["evidence"]["state"] in EVIDENCE_STATES
        assert set(part["disease_overlap"]) == {"state", "k", "n", "cases"}
        assert set(part["phenotype_set_overlap"]) == {"state", "k", "n"}
        for label in R10_LABELS:
            entry = part[label]
            assert entry["state"] in R10_STATES
            counted = entry["state"] in ("overlap", "none")
            assert isinstance(entry["n"], int) and entry["n"] > 0
            for field in ("k", "cases"):
                if field in entry:
                    if counted:
                        assert isinstance(entry[field], int)
                    else:
                        assert entry[field] is None, (scope, label, field, entry)
            if counted:
                assert (entry["k"] > 0) == (entry["state"] == "overlap")
                assert part["evidence"]["state"] == "found"
            else:
                assert part["evidence"]["state"] != "found"
    # It goes into a JSON report and ledger as written.
    json.dumps(block, allow_nan=False)


#: The test cohort: three distinct diseases (1, 2, 3) over four cases.
TEST = [
    _s("t0", [10, 11], 1),
    _s("t1", [12], 1),
    _s("t2", [20, 21], 2),
    _s("t3", [30], 3),
]


@pytest.fixture
def workspace(tmp_path):
    """A measurement workspace whose own splits share nothing with TEST."""
    root = tmp_path / "workspace"
    train = _write(root / "train_samples.json", [_s("w0", [90], 9), _s("w1", [91], 8)])
    val = _write(root / "val_samples.json", [_s("w2", [92], 7)])
    return root, train, val


# ---------------------------------------------------------------------------
# evidence_candidates
# ---------------------------------------------------------------------------
class TestEvidenceCandidates:
    def test_workspace_first_then_each_directory_sorted_and_not_recursive(self, tmp_path,
                                                                           workspace):
        root, _, _ = workspace
        first, second = tmp_path / "e1", tmp_path / "e2"
        for name in ("b_samples.json", "a_samples.json", "notes.json"):
            _write(first / name, TEST)
        _write(first / "nested" / "c_samples.json", TEST)
        _write(second / "z_samples.json", TEST)

        candidates = evidence_candidates(root, [second, first])

        assert candidates == [
            (root / "train_samples.json").resolve(), (root / "val_samples.json").resolve(),
            (second / "z_samples.json").resolve(),
            (first / "a_samples.json").resolve(), (first / "b_samples.json").resolve(),
        ]

    def test_missing_workspace_files_and_directories_named_like_samples_are_skipped(
        self, tmp_path
    ):
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "val_samples.json").mkdir()
        evidence = tmp_path / "evidence"
        (evidence / "dir_samples.json").mkdir(parents=True)
        _write(evidence / "x_samples.json", TEST)

        assert evidence_candidates(root, [evidence]) == [(evidence / "x_samples.json").resolve()]
        assert evidence_candidates(root, []) == []

    def test_deduplicated_by_resolved_path(self, tmp_path, workspace):
        """The workspace named again as evidence, the same directory twice, and a
        link to a listed file each add nothing: the same bytes are not read twice."""
        root, _, _ = workspace
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        os.symlink(root / "train_samples.json", evidence / "linked_samples.json")
        _write(evidence / "own_samples.json", TEST)

        candidates = evidence_candidates(root, [root, evidence, evidence])

        assert candidates == [
            (root / "train_samples.json").resolve(), (root / "val_samples.json").resolve(),
            (evidence / "own_samples.json").resolve(),
        ]
        assert len(set(candidates)) == len(candidates)

    @pytest.mark.parametrize("kind", ["missing", "file"])
    def test_an_evidence_directory_that_is_not_a_directory_is_refused(self, tmp_path,
                                                                      workspace, kind):
        root, _, _ = workspace
        named = tmp_path / "not_a_dir"
        if kind == "file":
            _write(named, TEST)
        good = tmp_path / "good"
        _write(good / "a_samples.json", TEST)

        with pytest.raises(ValueError, match="is not a directory; nothing was read") as refused:
            evidence_candidates(root, [good, named])

        assert str(named) in str(refused.value)

    @pytest.mark.parametrize("error, says", [
        (PermissionError(13, "Permission denied"), "PermissionError: Permission denied"),
        (OSError(5, "Input/output error"), "OSError: Input/output error"),
    ], ids=["permission", "io-error"])
    def test_an_evidence_directory_that_cannot_be_listed_is_refused(
        self, tmp_path, workspace, monkeypatch, error, says
    ):
        """A mode-0700 archive of another account, or a stale network mount:
        `is_dir()` passes and the listing fails. `Path.glob` swallows that error
        and yields nothing, so the checkpoint's inputs, which the archive holds,
        would be recorded as not found although it was never looked in. Catches a
        listing that does not refuse."""
        root, _, _ = workspace
        archive = tmp_path / "archive"
        _write(archive / "run7_train_samples.json", TEST)
        good = tmp_path / "good"
        _write(good / "a_samples.json", TEST)
        real_scandir = os.scandir

        def scandir(path="."):
            if os.fspath(path) == os.fspath(archive):
                raise error
            return real_scandir(path)

        monkeypatch.setattr(os, "scandir", scandir)

        with count_opens() as opens, pytest.raises(ValueError) as refused:
            evidence_candidates(root, [good, archive])

        message = str(refused.value)
        assert message.startswith(
            f"Training evidence directory {archive} could not be listed ({says}); "
            "nothing was read or written.")
        assert sum(opens.counts.values()) == 0, "no candidate is read"


# ---------------------------------------------------------------------------
# Units: disease overlap and phenotype-set overlap
# ---------------------------------------------------------------------------
def test_both_labels_are_counted_in_their_own_units(tmp_path):
    """Disease overlap: diseases 1 and 2 of TEST's three occur in training, and
    three test cases have one of them. Phenotype-set overlap: only t0's set
    {10, 11} under disease 1 occurs, in another order, under another patient id."""
    train = tmp_path / "train_samples.json"
    digest = _write(train, [
        _s("other-patient", [11, 10], 1),   # t0's set, reordered: counted
        _s("x", [20], 2),                   # disease 2, a subset of t2's set: not counted
        _s("y", [30], 4),                   # t3's set under another disease: not counted
    ])

    block = compute_r10({"train_samples": digest}, TEST, [train])

    _assert_well_formed(block)
    training = block["training"]
    assert training["recorded_sha256"] == digest
    assert training["evidence"] == {"state": "found", "path": str(train), "problem": None}
    assert training["disease_overlap"] == {"state": "overlap", "k": 2, "n": 3, "cases": 3}
    assert training["phenotype_set_overlap"] == {"state": "overlap", "k": 1, "n": 4}


def test_disjoint_phenotype_sets_with_overlapping_diseases(tmp_path):
    """§5.4: phenotype-set *none*, disease *overlap* with its counts."""
    train = tmp_path / "train_samples.json"
    digest = _write(train, [_s("a", [10, 11, 12], 1), _s("b", [99], 3), _s("c", [12], 2)])

    block = compute_r10({"train_samples": digest}, TEST, [train])

    _assert_well_formed(block)
    assert block["training"]["disease_overlap"] == {"state": "overlap", "k": 3, "n": 3,
                                                    "cases": 4}
    assert block["training"]["phenotype_set_overlap"] == {"state": "none", "k": 0, "n": 4}


@pytest.mark.parametrize("train_set, test_set", [
    ([0, 1], [0, 0, 1]),        # §5.4's case: repetition in the test cohort
    ([1, 0], [0, 1, 1, 0]),     # repetition in the test cohort, order reversed
])
def test_a_training_case_0_1_and_a_test_case_0_0_1_of_one_disease_overlap(
    tmp_path, train_set, test_set
):
    """§5.4: order and repetition are ignored, so these are one phenotype set.
    The repetition is on the test side, which R10 is handed as samples; evidence
    with one does not parse (N2, below)."""
    train = tmp_path / "train_samples.json"
    digest = _write(train, [_s("train-0", train_set, 5)])
    test = [_s("test-0", test_set, 5)]

    block = compute_r10({"train_samples": digest}, test, [train])

    _assert_well_formed(block)
    assert block["training"]["phenotype_set_overlap"] == {"state": "overlap", "k": 1, "n": 1}
    assert block["training"]["disease_overlap"] == {"state": "overlap", "k": 1, "n": 1,
                                                    "cases": 1}


@pytest.mark.parametrize("train_set, test_set", [
    ([0, 0, 1], [1, 0]),        # repetition in the evidence, order reversed
    ([1, 1, 0], [0, 1, 1, 0]),  # both
])
def test_evidence_that_lists_a_phenotype_twice_is_unreadable_not_counted(
    tmp_path, train_set, test_set
):
    """N2: the shared reader refuses a stored case that is not in phenotype
    normal form, so the file holding the recorded bytes does not parse. Its scope
    is *unverifiable* with no count, not the overlap a set comparison would find,
    and the problem names the rule and nothing of the row."""
    train = tmp_path / "train_samples.json"
    digest = _write(train, [_s("PATIENT-SECRET-7", train_set, 5)])
    test = [_s("test-0", test_set, 5)]

    block = compute_r10({"train_samples": digest}, test, [train])

    _assert_well_formed(block)
    training = block["training"]
    assert training["evidence"] == {
        "state": "unreadable",
        "path": str(train),
        "problem": "the recorded bytes could not be parsed as a samples file in phenotype "
                   "normal form",
    }
    for label in R10_LABELS:
        assert training[label]["state"] == "unverifiable"
        assert training[label]["k"] is None
    assert training["disease_overlap"]["cases"] is None


def test_the_same_set_under_another_disease_is_not_a_phenotype_set_overlap(tmp_path):
    train = tmp_path / "train_samples.json"
    digest = _write(train, [_s("a", [0, 1], 6)])

    block = compute_r10({"train_samples": digest}, [_s("t", [1, 0], 5)], [train])

    assert block["training"]["phenotype_set_overlap"]["state"] == "none"
    assert block["training"]["disease_overlap"]["state"] == "none"


def test_none_in_both_labels_when_the_evidence_shares_nothing(tmp_path, workspace):
    root, train, val = workspace

    block = compute_r10({"train_samples": train, "val_samples": val}, TEST,
                        evidence_candidates(root, []))

    _assert_well_formed(block)
    for scope in ("training", "validation"):
        assert block[scope]["evidence"]["state"] == "found"
        assert block[scope]["disease_overlap"] == {"state": "none", "k": 0, "n": 3, "cases": 0}
        assert block[scope]["phenotype_set_overlap"] == {"state": "none", "k": 0, "n": 4}


# ---------------------------------------------------------------------------
# Evidence: only the bytes the checkpoint names
# ---------------------------------------------------------------------------
def test_trained_on_another_split_unverifiable_without_its_evidence_and_counted_with_it(
    tmp_path, workspace
):
    """§5.4: the current workspace's split is disjoint from the test cohort, but
    the model was trained on another one. The current file's digest differs, so
    it is not used, and without the real file the labels are *unverifiable*
    (not *none*). Once the operator names the directory holding it, the overlap
    is counted."""
    root, _, _ = workspace
    real = tmp_path / "elsewhere" / "train_samples.json"
    digest = _write(real, [_s("r", [20, 21], 2)])
    recorded = {"train_samples": digest}

    without = compute_r10(recorded, TEST, evidence_candidates(root, []))
    with_it = compute_r10(recorded, TEST, evidence_candidates(root, [real.parent]))

    _assert_well_formed(without)
    assert without["candidates_examined"] == 2, "both workspace files were read and compared"
    assert without["training"]["evidence"] == {"state": "not_found", "path": None,
                                               "problem": None}
    assert without["training"]["disease_overlap"] == {"state": "unverifiable", "k": None,
                                                      "n": 3, "cases": None}
    assert without["training"]["phenotype_set_overlap"] == {"state": "unverifiable",
                                                            "k": None, "n": 4}
    _assert_well_formed(with_it)
    assert with_it["training"]["evidence"]["path"] == str(real.resolve())
    assert with_it["training"]["disease_overlap"] == {"state": "overlap", "k": 1, "n": 3,
                                                      "cases": 1}
    assert with_it["training"]["phenotype_set_overlap"] == {"state": "overlap", "k": 1, "n": 4}


def test_evidence_moved_with_its_bytes_intact_is_found_through_an_evidence_directory(
    tmp_path
):
    """§5.4: renamed and moved out of any workspace, it is still the file the
    checkpoint names, because the digest decides, not the name or the place."""
    original = tmp_path / "old_workspace" / "train_samples.json"
    digest = _write(original, [_s("a", [10, 11], 1)])
    moved = tmp_path / "archive" / "2026_run_train_samples.json"
    moved.parent.mkdir()
    original.rename(moved)
    empty_workspace = tmp_path / "new_workspace"
    empty_workspace.mkdir()

    block = compute_r10({"train_samples": digest}, TEST,
                        evidence_candidates(empty_workspace, [moved.parent]))

    _assert_well_formed(block)
    assert block["training"]["evidence"] == {"state": "found", "path": str(moved.resolve()),
                                             "problem": None}
    assert block["training"]["phenotype_set_overlap"] == {"state": "overlap", "k": 1, "n": 4}


@pytest.mark.parametrize("how", ["missing", "one_byte", "reserialised"])
def test_evidence_missing_or_altered_is_unverifiable_never_zero(tmp_path, how):
    """§5.4. Re-serialising the same rows changes the bytes, so even a file
    that would parse to the same samples is not the evidence."""
    path = tmp_path / "evidence" / "train_samples.json"
    samples = [_s("a", [10, 11], 1)]
    digest = _write(path, samples)
    if how == "missing":
        path.unlink()
    elif how == "one_byte":
        path.write_bytes(path.read_bytes().replace(b"10", b"13"))
    else:
        _write(path, samples, indent=2)
    assert not path.exists() or _sha(path.read_bytes()) != digest

    block = compute_r10({"train_samples": digest}, TEST,
                        evidence_candidates(tmp_path / "ws", [path.parent]))

    _assert_well_formed(block)
    assert block["training"]["evidence"]["state"] == "not_found"
    for label in R10_LABELS:
        assert block["training"][label]["state"] == "unverifiable"
        assert block["training"][label]["k"] is None


def test_a_candidate_that_vanished_after_listing_is_skipped(tmp_path):
    gone = tmp_path / "gone_samples.json"
    real = tmp_path / "train_samples.json"
    digest = _write(real, [_s("a", [10, 11], 1)])

    block = compute_r10({"train_samples": digest}, TEST, [gone, real])

    _assert_well_formed(block)
    assert block["candidates_examined"] == 1
    assert block["training"]["evidence"]["state"] == "found"


def test_a_stranger_that_is_not_even_json_is_skipped_without_a_parse_error(tmp_path):
    """Its digest is compared before its bytes are decoded, so it cannot end the
    search: the evidence after it is still found. Without the comparison first,
    this raises (or records a problem against a file that was never evidence)."""
    stranger = tmp_path / "a_samples.json"
    stranger.write_bytes(b"\xff\xfe{ not utf-8, not json")
    float_ids = tmp_path / "b_samples.json"
    float_ids.write_text(json.dumps([{"patient_id": "p", "phenotype_ids": [1.5],
                                      "disease_id": 2}]))
    real = tmp_path / "c_samples.json"
    digest = _write(real, [_s("a", [10, 11], 1)])

    block = compute_r10({"train_samples": digest}, TEST, [stranger, float_ids, real])

    _assert_well_formed(block)
    assert block["candidates_examined"] == 3
    assert block["training"]["evidence"] == {"state": "found", "path": str(real),
                                             "problem": None}


@pytest.mark.parametrize("junk", [
    b"\xff\xfe{ not utf-8",
    b'[{"patient_id": ',
    b'{"samples": []}',
    json.dumps([{"patient_id": "PATIENT-SECRET-7", "phenotype_ids": [1.5],
                 "disease_id": 2}]).encode("utf-8"),
])
def test_matching_evidence_that_does_not_parse_is_unverifiable_with_the_problem(
    tmp_path, junk
):
    """The checkpoint names these bytes, so this is the evidence, and it cannot
    be read: *unverifiable*, with the problem recorded. The problem names what
    the bytes failed to parse as and nothing of their content, since the block
    is copied into the ledger beside the checkpoint."""
    path = tmp_path / "val_samples.json"
    path.write_bytes(junk)
    train = tmp_path / "train_samples.json"
    train_digest = _write(train, [_s("a", [10, 11], 1)])

    block = compute_r10({"train_samples": train_digest, "val_samples": _sha(junk)}, TEST,
                        [path, train])

    _assert_well_formed(block)
    validation = block["validation"]
    assert validation["evidence"]["state"] == "unreadable"
    assert validation["evidence"]["path"] == str(path)
    problem = validation["evidence"]["problem"]
    assert problem.startswith("the recorded bytes could not be parsed as ")
    assert "PATIENT" not in problem and "1.5" not in problem and str(path) not in problem
    for label in R10_LABELS:
        assert validation[label]["state"] == "unverifiable"
        assert validation[label]["k"] is None
    assert validation["disease_overlap"]["cases"] is None
    assert block["training"]["evidence"]["state"] == "found", "the other scope carries on"


def test_unreadable_evidence_is_not_searched_for_again(tmp_path):
    """Another file with the same digest holds the same bytes and would fail the
    same way, so it is not read."""
    junk = b"[not json"
    first, second = tmp_path / "a_samples.json", tmp_path / "b_samples.json"
    first.write_bytes(junk)
    second.write_bytes(junk)

    with count_opens() as opens:
        block = compute_r10({"train_samples": _sha(junk)}, TEST, [first, second])

    assert block["training"]["evidence"]["state"] == "unreadable"
    assert (opens.opens(first), opens.opens(second)) == (1, 0)


# ---------------------------------------------------------------------------
# What the record names
# ---------------------------------------------------------------------------
def test_no_recorded_validation_file_is_no_validation_inputs_recorded(tmp_path, workspace):
    """A pipeline record (it names `train_samples`) with no `val_samples`:
    training ran no validation pass. Shown as such, with no count, and no
    candidate is read on validation's behalf."""
    root, train, _ = workspace

    with count_opens() as opens:
        block = compute_r10({"train_samples": train}, TEST, evidence_candidates(root, []))

    _assert_well_formed(block)
    validation = block["validation"]
    assert validation["recorded_sha256"] is None
    assert validation["evidence"] == {"state": "not_recorded", "path": None, "problem": None}
    assert validation["disease_overlap"] == {"state": "no_validation_inputs_recorded",
                                             "k": None, "n": 3, "cases": None}
    assert validation["phenotype_set_overlap"] == {"state": "no_validation_inputs_recorded",
                                                   "k": None, "n": 4}
    assert block["training"]["evidence"]["state"] == "found"
    assert opens.opens(root / "val_samples.json") == 0, "training matched first; nothing left"


@pytest.mark.parametrize("recorded", [
    None,
    {},
    {"kg": "f" * 64, "node_features": "e" * 64, "edge_indices": "d" * 64, "num_nodes": "c" * 64},
    "not a mapping",
], ids=["none", "empty", "graph-roles-only", "not-a-mapping"])
def test_neither_file_recorded_leaves_both_scopes_unverifiable_and_reads_nothing(
    workspace, recorded
):
    """A record that names no `train_samples` is not one the pipeline wrote, so
    it cannot show that training skipped validation either: both scopes are
    *unverifiable*, not *no validation inputs recorded*."""
    root, _, _ = workspace

    with count_opens() as opens:
        block = compute_r10(recorded, TEST, evidence_candidates(root, []))

    _assert_well_formed(block)
    assert block["candidates_examined"] == 0
    assert sum(opens.counts.values()) == 0
    for scope in ("training", "validation"):
        assert block[scope]["recorded_sha256"] is None
        assert block[scope]["evidence"]["state"] == "not_recorded"
        for label in R10_LABELS:
            assert block[scope][label]["state"] == "unverifiable"


def test_a_recorded_validation_file_without_a_training_file(workspace):
    """Training *unverifiable* (no record of it); validation counted from its evidence."""
    root, _, val = workspace

    block = compute_r10({"val_samples": val}, TEST, evidence_candidates(root, []))

    _assert_well_formed(block)
    assert block["training"]["disease_overlap"]["state"] == "unverifiable"
    assert block["validation"]["evidence"]["state"] == "found"
    assert block["validation"]["disease_overlap"]["state"] == "none"


NOT_DIGESTS = [
    ("", "an empty string"),
    (None, "null"),
    (123, "a value of type int"),
    (["a" * 64], "a value of type list"),
    ({"sha256": "a" * 64}, "a value of type dict"),
]


@pytest.mark.parametrize("value, kind", NOT_DIGESTS)
def test_a_val_samples_entry_that_is_not_a_digest_is_unverifiable_not_unrecorded(
    workspace, value, kind
):
    """§5.4 allows *no validation inputs recorded* only when the checkpoint
    records no `val_samples`, because training ran no validation pass. An entry
    that is there and is not a digest names validation inputs (perhaps the ones
    early stopping used) that no file can be matched against: *unverifiable*,
    with a problem naming the role and the value's kind, never the value.
    Catches the earlier rule, which counted it as not recorded."""
    root, train, _ = workspace

    block = compute_r10({"train_samples": train, "val_samples": value}, TEST,
                        evidence_candidates(root, []))

    _assert_well_formed(block)
    validation = block["validation"]
    assert validation["recorded_sha256"] is None
    assert validation["evidence"] == {
        "state": "not_recorded", "path": None,
        "problem": f"the checkpoint records val_samples as {kind}, not a SHA-256 string, "
                   "so no file can be looked for by it; it is not a record the pipeline "
                   "wrote",
    }
    for label in R10_LABELS:
        assert validation[label]["state"] == "unverifiable"
    assert block["training"]["evidence"]["state"] == "found"


@pytest.mark.parametrize("value, kind", NOT_DIGESTS[:2])
def test_a_train_samples_entry_that_is_not_a_digest_names_its_problem(workspace, value, kind):
    """Training was already *unverifiable* for such an entry; its evidence now
    says why. With `val_samples` absent, validation stays *unverifiable* too:
    the record is not the pipeline's, so it cannot show validation was skipped."""
    root, _, _ = workspace

    block = compute_r10({"train_samples": value}, TEST, evidence_candidates(root, []))

    _assert_well_formed(block)
    assert block["training"]["evidence"]["problem"].startswith(
        f"the checkpoint records train_samples as {kind}, not a SHA-256 string")
    assert block["validation"]["evidence"]["problem"] is None, "absent, not malformed"
    for scope in ("training", "validation"):
        for label in R10_LABELS:
            assert block[scope][label]["state"] == "unverifiable"


def test_training_and_validation_recording_the_same_bytes_are_both_found_from_one_read(
    tmp_path
):
    path = tmp_path / "shared_samples.json"
    digest = _write(path, [_s("a", [10, 11], 1)])

    with count_opens() as opens:
        block = compute_r10({"train_samples": digest, "val_samples": digest}, TEST, [path])

    _assert_well_formed(block)
    assert opens.opens(path) == 1
    assert block["candidates_examined"] == 1
    for scope in ("training", "validation"):
        assert block[scope]["evidence"]["path"] == str(path)
        assert block[scope]["phenotype_set_overlap"]["k"] == 1


def test_the_four_states_are_the_ones_named():
    assert R10_STATES == ("overlap", "none", "unverifiable", "no_validation_inputs_recorded")
    assert R10_DEFINITION_VERSION == 1


def test_an_empty_test_cohort_is_refused_before_any_read(workspace):
    root, train, _ = workspace

    with count_opens() as opens:
        with pytest.raises(ValueError, match="empty cohort"):
            compute_r10({"train_samples": train}, [], evidence_candidates(root, []))

    assert sum(opens.counts.values()) == 0


# ---------------------------------------------------------------------------
# Reads: once each, compared against both digests, released before the next
# ---------------------------------------------------------------------------
def test_each_candidate_is_opened_once_while_both_scopes_are_unmatched(tmp_path, workspace):
    """Neither recorded file is among the candidates, so every one is read and
    compared against both digests at once: one open each, not one per scope."""
    root, _, _ = workspace
    evidence = tmp_path / "evidence"
    _write(evidence / "a_samples.json", TEST)
    (evidence / "b_samples.json").write_bytes(b"\xff not json")
    candidates = evidence_candidates(root, [evidence])
    assert len(candidates) == 4

    with count_opens() as opens:
        block = compute_r10({"train_samples": "a" * 64, "val_samples": "b" * 64}, TEST,
                            candidates)

    assert {str(path): opens.opens(path) for path in candidates} == {
        str(path): 1 for path in candidates
    }
    assert block["candidates_examined"] == 4
    for scope in ("training", "validation"):
        assert block[scope]["evidence"]["state"] == "not_found"


def test_reading_stops_once_every_recorded_file_is_found(tmp_path, workspace):
    root, train, val = workspace
    evidence = tmp_path / "evidence"
    _write(evidence / "later_samples.json", TEST)
    candidates = evidence_candidates(root, [evidence])

    with count_opens() as opens:
        block = compute_r10({"train_samples": train, "val_samples": val}, TEST, candidates)

    assert block["candidates_examined"] == 2
    assert opens.opens(evidence / "later_samples.json") == 0


def test_every_raw_buffer_is_released_before_the_next_candidate_is_read(monkeypatch,
                                                                         tmp_path):
    """The release check looks at each `read_once` and after the return: no
    candidate's bytes are alive then, whether it mismatched, matched and did not
    parse, or matched and was counted."""
    stranger = tmp_path / "a_samples.json"
    stranger.write_bytes(b"\xff not json" * 512)
    junk = b"[not json" * 512
    unreadable = tmp_path / "b_samples.json"
    unreadable.write_bytes(junk)
    train = tmp_path / "c_samples.json"
    digest = _write(train, [_s(f"p{i}", [10, 11], 1) for i in range(200)])
    after = tmp_path / "d_samples.json"
    _write(after, TEST)

    block, reads = check_release(
        monkeypatch,
        lambda: compute_r10({"train_samples": digest, "val_samples": _sha(junk)}, TEST,
                            [stranger, unreadable, train, after]),
    )

    assert reads == (stranger, unreadable, train)
    assert block["training"]["evidence"]["state"] == "found"
    assert block["validation"]["evidence"]["state"] == "unreadable"


def test_matching_samples_are_released_before_the_next_read(monkeypatch, tmp_path):
    """Matching evidence is reduced to the test keys it holds; its samples are
    gone by the time the next candidate is read."""
    train = tmp_path / "a_samples.json"
    train_digest = _write(train, [_s(f"p{i}", [10, 11], 1) for i in range(50)])
    val = tmp_path / "b_samples.json"
    val_digest = _write(val, [_s("v", [20, 21], 2)])
    alive = []
    real = r10.read_samples_file

    def watched(path, **kwargs):
        still = [ref for ref in alive if ref() is not None]
        assert not still, f"{len(still)} samples from an earlier read are alive at {path}"
        result = real(path, **kwargs)
        alive.extend(weakref.ref(sample) for sample in result.samples)
        return result

    monkeypatch.setattr(r10, "read_samples_file", watched)

    block = compute_r10({"train_samples": train_digest, "val_samples": val_digest}, TEST,
                        [train, val])

    assert len(alive) == 51
    assert all(ref() is None for ref in alive)
    assert block["validation"]["phenotype_set_overlap"]["k"] == 1


def test_candidates_are_read_with_the_digests_still_wanted(monkeypatch, tmp_path, workspace):
    """The expectation passed to the reader is every recorded digest still
    unmatched: both at first, then only validation's once training is found."""
    root, train, val = workspace
    seen = []
    real = r10.read_samples_file

    def spy(path, **kwargs):
        seen.append((path.name, kwargs["expected_sha256"]))
        return real(path, **kwargs)

    monkeypatch.setattr(r10, "read_samples_file", spy)

    compute_r10({"train_samples": train, "val_samples": val}, TEST,
                evidence_candidates(root, []))

    assert seen == [("train_samples.json", {train, val}), ("val_samples.json", {val})]
