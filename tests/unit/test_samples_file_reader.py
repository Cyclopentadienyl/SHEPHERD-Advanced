"""
The one samples parser compares before it parses (M4, spec §1.2).
=================================================================
`read_samples_file` is the parser behind `read_samples`, the registered cohort
version's `samples.json` and R10's evidence candidates. Given
`expected_sha256`, it compares the digest of its one `read_once` with the
digests given and refuses other bytes before decoding them. This file shows:

- the comparison comes before UTF-8 decoding and JSON parsing: bytes that are
  neither UTF-8 nor JSON, under another digest, are refused as a mismatch, never
  as a parse error;
- the comparison and the parse are of one buffer: a replacement after the read
  is neither compared nor parsed, and each file is opened once, match or not;
- neither the mismatch nor its traceback keeps the bytes;
- matching bytes are parsed and refused exactly as `read_samples` always has,
  and every refusal of a read file carries that read's identity, with a `what`
  that holds none of the file's content;
- a case that lists a phenotype twice is refused (N2) by the shared rule, after
  the integer check, naming the producer that rebuilds the file; a file in
  normal form is read as written;
- a missing file names its path; an empty or mistyped expectation is refused
  before anything is read;
- `read_samples` keeps its message for a missing split and delegates.

Module: tests/unit/test_samples_file_reader.py
"""
from __future__ import annotations

import hashlib
import json

import pytest

from src.kg.data_loader import DiagnosisSample
from src.kg.phenotype_normalisation import PHENOTYPE_NORMALISATION_VERSION, normalise_phenotypes
from src.kg.storage import file_storage
from src.kg.storage.file_storage import (
    SamplesDigestMismatch,
    SamplesRead,
    UnparseableFile,
    read_samples,
    read_samples_file,
)
from src.utils.fingerprint import ReadIdentity
from tests.fixtures.opens import count_opens
from tests.fixtures.release import check_release
from tests.fixtures.replacement import replace_after_read

#: Neither UTF-8 nor JSON: decoding it raises before json ever sees it.
JUNK = b"\xff\xfe{ not utf-8, not json"


def _write(path, samples):
    """Write `samples` as the JSON list `read_samples_file` parses; return its digest."""
    rows = [
        {"patient_id": s.patient_id, "phenotype_ids": s.phenotype_ids, "disease_id": s.disease_id}
        for s in samples
    ]
    data = json.dumps(rows).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


SAMPLES = [
    DiagnosisSample(patient_id="p0", phenotype_ids=[0, 1], disease_id=2),
    DiagnosisSample(patient_id="p1", phenotype_ids=[3], disease_id=4),
]


@pytest.fixture
def samples_file(tmp_path):
    path = tmp_path / "cohort_samples.json"
    return path, _write(path, SAMPLES)


# ---------------------------------------------------------------------------
# What it returns
# ---------------------------------------------------------------------------
def test_with_no_expectation_it_parses_and_returns_the_reads_identity(samples_file):
    path, digest = samples_file

    result = read_samples_file(path)

    assert isinstance(result, SamplesRead)
    assert result.identity == ReadIdentity(path=path, sha256=digest)
    assert result.samples == SAMPLES


@pytest.mark.parametrize("form", ["str", "set", "frozenset"])
def test_matching_bytes_are_parsed_whatever_form_the_expectation_takes(samples_file, form):
    path, digest = samples_file
    expected = {"str": digest, "set": {digest, "0" * 64},
                "frozenset": frozenset({digest})}[form]

    result = read_samples_file(path, expected_sha256=expected)

    assert result.identity.sha256 == digest
    assert result.samples == SAMPLES


def test_training_fields_are_read_as_read_samples_reads_them(tmp_path):
    path = tmp_path / "x_samples.json"
    path.write_text(json.dumps([{"patient_id": "p", "phenotype_ids": [1], "disease_id": 2,
                                 "candidate_disease_ids": [2, 3], "gene_ids": [7]}]))

    plain = read_samples_file(path).samples[0]
    full = read_samples_file(path, training_fields=True).samples[0]

    assert (plain.candidate_disease_ids, plain.gene_ids) == (None, None)
    assert (full.candidate_disease_ids, full.gene_ids) == ([2, 3], [7])


# ---------------------------------------------------------------------------
# Compare before parse, from the one read
# ---------------------------------------------------------------------------
def test_a_non_matching_file_that_is_not_even_json_is_a_mismatch_not_a_parse_error(tmp_path):
    """Decoding these bytes raises; the mismatch is raised instead, so the
    comparison came first. Without it, or with it after the parse, this is an
    `UnparseableFile` ("could not be parsed as UTF-8 JSON")."""
    path = tmp_path / "other_samples.json"
    path.write_bytes(JUNK)
    wanted = frozenset({"a" * 64, "b" * 64})

    with pytest.raises(SamplesDigestMismatch) as refused:
        read_samples_file(path, expected_sha256=wanted)

    assert not isinstance(refused.value, UnparseableFile)
    assert refused.value.path == path
    assert refused.value.sha256 == hashlib.sha256(JUNK).hexdigest()
    assert refused.value.expected == wanted
    assert isinstance(refused.value.expected, frozenset)
    assert str(path) in str(refused.value)
    assert refused.value.__cause__ is None, "nothing was decoded, so nothing failed to decode"


def test_a_mismatch_is_a_value_error(samples_file):
    """Existing callers that catch `ValueError` for a bad file still do."""
    path, _ = samples_file

    with pytest.raises(ValueError):
        read_samples_file(path, expected_sha256="0" * 64)


@pytest.mark.parametrize("matching", [True, False])
def test_the_file_is_opened_once_match_or_not(samples_file, matching):
    path, digest = samples_file

    with count_opens() as opens:
        if matching:
            read_samples_file(path, expected_sha256=digest)
        else:
            with pytest.raises(SamplesDigestMismatch):
                read_samples_file(path, expected_sha256="0" * 64)

    assert opens.opens(path) == 1


@pytest.mark.parametrize("how", ["rename", "rewrite"])
def test_the_comparison_is_of_the_bytes_parsed_not_of_the_path(monkeypatch, tmp_path, how):
    """The file is replaced right after its read. Expecting the original passes
    and parses the original; expecting the replacement, which is on disk by the
    time anything could look again, is refused: neither the comparison nor the
    parse goes back to the path."""
    path = tmp_path / "train_samples.json"
    original = _write(path, SAMPLES)
    other = [DiagnosisSample(patient_id="q", phenotype_ids=[9], disease_id=9)]
    replacement = json.dumps([{"patient_id": "q", "phenotype_ids": [9],
                               "disease_id": 9}]).encode("utf-8")
    replaced = hashlib.sha256(replacement).hexdigest()

    record = replace_after_read(monkeypatch, after=path, target=path, data=replacement, how=how)
    result = read_samples_file(path, expected_sha256=original)
    assert record.fired == 1
    assert result.identity.sha256 == original != _sha(path)
    assert result.samples == SAMPLES

    _write(path, SAMPLES)
    record = replace_after_read(monkeypatch, after=path, target=path, data=replacement, how=how)
    with pytest.raises(SamplesDigestMismatch) as refused:
        read_samples_file(path, expected_sha256=replaced)
    assert record.fired == 1
    assert refused.value.sha256 == original
    assert _sha(path) == replaced
    assert read_samples_file(path, expected_sha256=replaced).samples == other


def test_a_kept_mismatch_keeps_no_bytes(monkeypatch, tmp_path):
    """The refusal and its traceback are returned and held while the release
    check looks: the buffer is released before the raise, so neither holds it.
    Without the reader's `del read` before raising, the traceback's frame does."""
    path = tmp_path / "other_samples.json"
    path.write_bytes(JUNK * 64)

    def refusal():
        try:
            read_samples_file(path, expected_sha256="0" * 64)
        except SamplesDigestMismatch as exc:
            return exc
        raise AssertionError("no mismatch")

    kept, reads = check_release(monkeypatch, refusal)

    assert isinstance(kept, SamplesDigestMismatch) and kept.__traceback__ is not None
    assert reads == (path,)


def test_a_matching_read_releases_its_buffer(monkeypatch, samples_file):
    path, digest = samples_file

    result, reads = check_release(monkeypatch,
                                  lambda: read_samples_file(path, expected_sha256=digest))

    assert reads == (path,)
    assert result.samples == SAMPLES


# ---------------------------------------------------------------------------
# Matching bytes: today's parse and refusals, with the read's identity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("junk, message", [
    (JUNK, "could not be parsed as UTF-8 JSON"),
    (b'[{"patient_id": ', "could not be parsed as UTF-8 JSON"),
    (b'{"samples": []}', "could not be parsed as a samples file"),
    (b'[{"tampered": true}]', "could not be parsed as a samples file"),
])
def test_matching_bytes_that_do_not_parse_are_refused_with_the_reads_identity(
    tmp_path, junk, message
):
    path = tmp_path / "train_samples.json"
    path.write_bytes(junk)
    digest = hashlib.sha256(junk).hexdigest()

    with pytest.raises(UnparseableFile, match=message) as refused:
        read_samples_file(path, expected_sha256={digest, "0" * 64})

    assert str(path) in str(refused.value)
    assert refused.value.identity == ReadIdentity(path=path, sha256=digest)
    assert refused.value.what in str(refused.value)
    assert refused.value.__cause__ is not None, "the parser's own error stays chained"


@pytest.mark.parametrize("field, value", [("disease_id", 2.0), ("disease_id", True),
                                          ("phenotype_ids", [0, 1.5])])
def test_a_non_integer_id_is_refused_and_what_names_no_patient(tmp_path, field, value):
    """The message names the patient, as `read_samples` always has; `what` is
    the rule alone, for a caller that records the failure where row data must
    not go."""
    row = {"patient_id": "PATIENT-SECRET-7", "phenotype_ids": [0, 1], "disease_id": 2}
    row[field] = value
    path = tmp_path / "train_samples.json"
    path.write_text(json.dumps([row]))
    digest = _sha(path)

    with pytest.raises(UnparseableFile, match="refused rather than converted") as refused:
        read_samples_file(path, expected_sha256=digest)

    assert "PATIENT-SECRET-7" in str(refused.value)
    assert refused.value.identity.sha256 == digest
    assert refused.value.what == "a samples file with integer ids"
    assert "PATIENT-SECRET-7" not in refused.value.what and repr(value) not in refused.value.what


# ---------------------------------------------------------------------------
# Phenotype normal form (N2): the shared rule, checked after the integer check
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("phenotypes, counted", [
    ([0, 1, 0], "1 entry of its phenotype_ids repeats"),
    # Two ids listed twice each: two repeated entries, not one id listed thrice.
    ([1, 1, 2, 2], "2 entries of its phenotype_ids repeat"),
    ([5, 5, 5, 7, 7], "3 entries of its phenotype_ids repeat"),
])
def test_a_case_that_lists_a_phenotype_twice_is_refused_naming_its_producers(
    tmp_path, phenotypes, counted
):
    """The message names the file, the sample, the patient, the number of
    repeated entries (every one after an id's first, whichever id), the rule's
    version and the producer of each kind of samples file; `what` names the rule
    alone, since R10 copies it into the report and the ledger."""
    rows = [{"patient_id": "p0", "phenotype_ids": [0, 1], "disease_id": 2},
            {"patient_id": "PATIENT-SECRET-7", "phenotype_ids": phenotypes, "disease_id": 2}]
    path = tmp_path / "train_samples.json"
    path.write_text(json.dumps(rows))
    digest = _sha(path)

    with pytest.raises(UnparseableFile, match="not in phenotype normal form") as refused:
        read_samples_file(path, expected_sha256=digest)

    message = str(refused.value)
    assert str(path) in message
    assert "sample 1 (patient 'PATIENT-SECRET-7')" in message
    assert f"{counted} an id listed earlier in the case" in message
    assert f"rule version {PHENOTYPE_NORMALISATION_VERSION}" in message
    assert "scripts/build_knowledge_graph.py --generate-samples for a workspace split" in message
    assert "scripts/import_cohort.py for a test cohort version" in message
    assert refused.value.identity == ReadIdentity(path=path, sha256=digest)
    assert refused.value.what == "a samples file in phenotype normal form"
    assert "PATIENT-SECRET-7" not in refused.value.what


def test_the_check_is_the_shared_rule_on_phenotype_ids_only(monkeypatch, tmp_path):
    """The reader calls `normalise_phenotypes` and names its version constant, so
    the rule has one definition (plan §7), and gives it each sample's
    `phenotype_ids` and nothing else (§4.1): a repeated gene or candidate id is
    not a repeated phenotype."""
    given = []

    def rule(identities):
        given.append(list(identities))
        return normalise_phenotypes(identities)

    monkeypatch.setattr(file_storage, "normalise_phenotypes", rule)
    monkeypatch.setattr(file_storage, "PHENOTYPE_NORMALISATION_VERSION", 7)
    path = tmp_path / "train_samples.json"
    path.write_text(json.dumps([
        {"patient_id": "p", "phenotype_ids": [3, 4], "disease_id": 1,
         "candidate_disease_ids": [1, 1], "gene_ids": [9, 9]},
        {"patient_id": "q", "phenotype_ids": [4, 4], "disease_id": 1},
    ]))

    with pytest.raises(UnparseableFile, match="rule version 7"):
        read_samples_file(path, training_fields=True)

    assert given == [[3, 4], [4, 4]]


@pytest.mark.parametrize("phenotypes", [[1, True], [True, 1], [1, 1.0]])
def test_an_id_that_is_not_an_integer_is_reported_as_such_not_as_a_repeat(tmp_path, phenotypes):
    """`True == 1` and `1.0 == 1`, so the rule alone would count either pair as one
    phenotype listed twice. The integer check runs first and names the id."""
    path = tmp_path / "train_samples.json"
    path.write_text(json.dumps([{"patient_id": "p", "phenotype_ids": phenotypes,
                                 "disease_id": 2}]))

    with pytest.raises(UnparseableFile, match="refused rather than converted") as refused:
        read_samples_file(path)

    assert refused.value.what == "a samples file with integer ids"
    assert "normal form" not in str(refused.value)


def test_a_file_in_normal_form_is_read_as_written(tmp_path):
    """Checked, never normalised: each case keeps the file's ids in the file's
    order, which is not sorted."""
    samples = [DiagnosisSample(patient_id="p0", phenotype_ids=[9, 3, 5], disease_id=2),
               DiagnosisSample(patient_id="p1", phenotype_ids=[3], disease_id=2)]
    path = tmp_path / "train_samples.json"
    digest = _write(path, samples)

    result = read_samples_file(path, expected_sha256=digest)

    assert result.samples == samples
    assert result.identity == ReadIdentity(path=path, sha256=digest)


def test_read_samples_refusals_carry_the_identity_too(tmp_path):
    """The same parser: `read_samples`' refusal of a bad split is an
    `UnparseableFile` with the read's identity, and still a `ValueError`."""
    path = tmp_path / "train_samples.json"
    path.write_bytes(b"null")

    with pytest.raises(ValueError, match="could not be parsed as a samples file") as refused:
        read_samples(tmp_path, "train")

    assert isinstance(refused.value, UnparseableFile)
    assert refused.value.identity == ReadIdentity(path=path, sha256=_sha(path))


@pytest.mark.parametrize("name, junk", [("num_nodes.json", b"{not json"),
                                        ("node_features.pt", b"not a tensor file")])
def test_a_graph_file_that_does_not_parse_carries_its_identity(tmp_path, name, junk):
    """`_read_json` and `_read_tensors` refuse through the same helper."""
    pytest.importorskip("torch")
    from tests.fixtures.generated_workspace import write_generated_workspace

    root, _ = write_generated_workspace(tmp_path / "ws", train_ids=[0, 1], val_ids=[2])
    (root / name).write_bytes(junk)

    with pytest.raises(UnparseableFile, match="could not be parsed") as refused:
        file_storage.read_graph_artifacts(root)

    assert refused.value.identity == ReadIdentity(path=root / name,
                                                  sha256=hashlib.sha256(junk).hexdigest())


# ---------------------------------------------------------------------------
# Refusals before the read
# ---------------------------------------------------------------------------
def test_a_missing_file_is_refused_naming_its_path(tmp_path):
    path = tmp_path / "gone_samples.json"

    with pytest.raises(FileNotFoundError, match="Samples file not found") as refused:
        read_samples_file(path, expected_sha256="0" * 64)

    assert str(path) in str(refused.value)


@pytest.mark.parametrize("expected", [set(), frozenset(), "", {""}, {"a" * 64, ""}])
def test_an_expectation_nothing_could_match_is_refused_before_the_read(samples_file, expected):
    """Otherwise the read would be refused whatever the file held, and an empty
    set would make every candidate look like a stranger."""
    path, _ = samples_file

    with count_opens() as opens:
        with pytest.raises(ValueError, match="names no usable digest.*nothing was read"):
            read_samples_file(path, expected_sha256=expected)

    assert opens.opens(path) == 0


@pytest.mark.parametrize("expected", [["a" * 64], ("a" * 64,), 7])
def test_an_expectation_of_another_type_is_refused_before_the_read(samples_file, expected):
    """A list or tuple is not silently taken as a set of digests."""
    path, _ = samples_file

    with count_opens() as opens:
        with pytest.raises(TypeError, match="must be a digest string or a set"):
            read_samples_file(path, expected_sha256=expected)

    assert opens.opens(path) == 0


# ---------------------------------------------------------------------------
# read_samples delegates and keeps its own message
# ---------------------------------------------------------------------------
def test_read_samples_delegates_to_the_one_parser(monkeypatch, tmp_path):
    path = tmp_path / "val_samples.json"
    _write(path, SAMPLES)
    calls = []
    real = file_storage.read_samples_file

    def spy(p, **kwargs):
        calls.append((p, kwargs))
        return real(p, **kwargs)

    monkeypatch.setattr(file_storage, "read_samples_file", spy)

    result = read_samples(tmp_path, "val", training_fields=True)

    assert calls == [(path, {"training_fields": True})]
    assert result.identity.path == path


def test_read_samples_still_names_the_splits_a_workspace_has(tmp_path):
    _write(tmp_path / "train_samples.json", SAMPLES)
    (tmp_path / "dir_samples.json").mkdir()

    with pytest.raises(FileNotFoundError, match="this workspace has: train$"):
        read_samples(tmp_path, "test")
