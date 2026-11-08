"""
R10: how a test cohort relates to what the model was trained and validated on.
===============================================================================
Contract §5.4 defines it and this module computes it, for a registered test
cohort only (M4 decision 16). It is recorded, never enforced: nothing here
refuses a measurement because a cohort overlaps. Its rules exist to keep a label
from saying more than its evidence.

**Evidence is only the bytes the checkpoint names.** A checkpoint records the
SHA-256 of the `train_samples` and `val_samples` files training read
(`training_input_digests`) and no path to them, so the files are found by
digest. The measurement workspace's own `train_samples.json` and
`val_samples.json`, then every `*_samples.json` directly inside each directory
the operator names, are candidates: places to look, never defaults. A file with
another digest (a rebuilt workspace, another seed, another split) says nothing
about what the model saw, in either direction, and is skipped.

**Compared before parsed, each candidate read once.** Every candidate goes
through `read_samples_file` with the digests still wanted, so its one read is
compared before a byte of it is decoded: a stranger that is not even JSON is
skipped as `SamplesDigestMismatch`, never a parse error that would end the
search. A candidate is read once even while both scopes are still unmatched,
because one read is compared against both digests. A file that holds the
recorded bytes and does not parse leaves its scope *unverifiable*, with the
problem recorded.

**A place that cannot be looked in is recorded, and the search goes on**
(K8-6r.3). A candidate that cannot be read (permission, a directory put in its
place, an I/O error), a listed path whose stat or resolve fails, an evidence
directory that exists and cannot be listed, and a named evidence directory
whose own state cannot be learned are each an `UnreadCandidate`: its bytes were
never compared, so it may have held the evidence. The remaining candidates are
still read, and a scope found in one of them is found. A scope still not found
once the search ends, while some place could not be looked in, is
`search_incomplete`: *unverifiable*, with a problem naming each such place and
its reason, never "not found" and never a count. Running out of memory while
reading a candidate is a resource failure, not a finding: it is re-raised as a
`MemoryError` naming the candidate, and no state is recorded for it. Only a
named evidence directory known not to be one is refused, as an argument error.

**Four states per label, and a number only where there is evidence.** *overlap*
and *none* need the evidence found and parsed, and carry `k`. Two states carry
`k = None` and `cases = None`, never 0, so no reader can take either for "no
leakage":
- *unverifiable*: the recorded bytes were not found, or could not be parsed, or
  a place that may hold them could not be looked in, or the record names no
  training file at all, so it is not one the pipeline wrote;
- *no_validation_inputs_recorded*: the record names a training file and has no
  `val_samples` entry at all, because training ran no validation pass. With
  neither file recorded, validation is *unverifiable* instead: such a record
  cannot show that training skipped validation. So is a `val_samples` entry
  that is there and is not a digest: it names validation inputs, which may be
  the ones early stopping used, and no file can be looked for by it.

`n` is the test cohort's own size and is always given.

**Units** (§5.4). Disease overlap: the distinct test diseases that occur as a
disease in the scope's samples, `k` of `n` distinct test diseases, and `cases`,
the test cases whose disease is among them. Phenotype-set overlap: the test
cases whose `(disease_id, frozenset(phenotype_ids))` occurs in the scope, `k` of
`n` test cases. Order and repetition are ignored, so a training case `[0, 1]`
and a test case `[0, 0, 1]` of one disease overlap although a scorer that
averages every listed position may score them differently; that is the case
this label exists to show. Patient ids are ignored, near-duplicates (subsets,
supersets) are not counted, and no normalisation is applied: the frozenset is
already free of order and repetition, and `normalise_phenotypes` keeps the same
set, so applying it would change nothing and would be a second rule.

**Since N2, no repeat reaches the comparison from a file.** The shared reader
refuses a case that lists a phenotype twice, so an evidence file holding one
does not parse: its scope is *unverifiable*, not counted as the overlap a set
would find. A registered test cohort holds none either, since the importer
removes them. Ignoring repetition still matters for samples handed in directly,
which is how the `[0, 0, 1]` case is tested.

**Memory.** A training file can hold hundreds of thousands of rows. The test
cohort's keys are built first, and each matching evidence file is reduced to
the test keys it contains and released before the next candidate is read.

**What the block holds.** Digests, counts, states, the path of each evidence
file and, for an unreadable one, a problem that names what its bytes failed to
parse as, never a row of them; for a search left incomplete, a problem naming
each place that could not be looked in and its error, never a byte of it; for a
role recorded with a value that is not a digest, a problem naming the role and
the value's kind, never the value. The block is copied into the report and into
the ledger beside the checkpoint, and a checkpoint can sit inside the repository
tree.

Module: src/evaluation/r10.py
"""
from __future__ import annotations

import errno
import fnmatch
import os
import stat
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Tuple, Union

from src.config.server_settings import is_directory
from src.kg.storage.file_storage import (
    SamplesDigestMismatch,
    UnparseableFile,
    read_samples_file,
)

if TYPE_CHECKING:
    from src.kg.data_loader import DiagnosisSample

__all__ = [
    "EVIDENCE_STATES",
    "R10_DEFINITION_VERSION",
    "R10_LABELS",
    "R10_SCOPES",
    "R10_STATES",
    "UnreadCandidate",
    "compute_r10",
    "evidence_candidates",
]

#: Bumped when a unit, a state or the evidence rule changes, so two blocks are
#: compared only under one definition.
#:
#: Version 2 (K8-6r.3) adds the evidence state "search_incomplete": a place that
#: could not be looked in no longer ends the run. A version-1 block is read as
#: written, with its original meaning, by the same readers: it has no
#: "search_incomplete", and its "not_found" meant every candidate was read and
#: compared, since under version 1 a candidate that could not be read refused
#: the run. "not_found" means the same under version 2, so no block is migrated.
#:
#: Still 2 since a named evidence directory whose state cannot be learned is
#: one more such place (K8-6 completion review, 2026-11-08): no state's meaning
#: changed, and no earlier block can hold one, because that run was refused.
R10_DEFINITION_VERSION = 2

#: Every state a label can take. Only the first two carry a count.
R10_STATES: Tuple[str, ...] = ("overlap", "none", "unverifiable", "no_validation_inputs_recorded")

#: What became of a scope's evidence: found and parsed; recorded but held by no
#: candidate, every one read and compared; held by a candidate that does not
#: parse; not recorded at all; recorded and held by no candidate that was read,
#: while at least one place could not be looked in (version 2).
EVIDENCE_STATES: Tuple[str, ...] = (
    "found", "not_found", "unreadable", "not_recorded", "search_incomplete",
)

#: Scope -> the `training_input_digests` role that names its file.
R10_SCOPES: Tuple[Tuple[str, str], ...] = (
    ("training", "train_samples"),
    ("validation", "val_samples"),
)

#: The two labels of each scope, in the block's order.
R10_LABELS: Tuple[str, ...] = ("disease_overlap", "phenotype_set_overlap")

#: The role whose absence, beside a recorded training file, means training ran
#: no validation pass.
_VALIDATION_ROLE = dict(R10_SCOPES)["validation"]

#: The measurement workspace's own candidates, looked at first.
_WORKSPACE_CANDIDATES: Tuple[str, ...] = ("train_samples.json", "val_samples.json")
#: What counts as a candidate inside an evidence directory (non-recursive).
_EVIDENCE_PATTERN = "*_samples.json"

#: The stat errors that say a path is not there, as Python 3.12's `Path.is_file`
#: treats them; any other is a place that could not be looked at.
_ABSENT_ERRNOS = frozenset({errno.ENOENT, errno.ENOTDIR, errno.EBADF, errno.ELOOP})

_CaseKey = Tuple[int, FrozenSet[int]]


@dataclass(frozen=True)
class UnreadCandidate:
    """A place R10 could not look in: a candidate that could not be read, a
    listed path whose stat or resolve failed, an evidence directory that exists
    and could not be listed, or a named one whose own state could not be
    learned. Its bytes were never compared, so it may hold the evidence.
    `reason` is "<ErrorType>: <strerror>"; nothing of the content."""

    path: Path
    reason: str


#: What `evidence_candidates` lists and `compute_r10` takes: a file to read, or
#: a place already known to be unreadable, in its place in the order.
EvidenceCandidate = Union[Path, UnreadCandidate]


def _reason(exc: OSError) -> str:
    """"<ErrorType>: <strerror>", as every place R10 could not look at is named."""
    reason = exc.strerror or (f"errno {exc.errno}" if exc.errno is not None
                              else "no reason given")
    return f"{type(exc).__name__}: {reason}"


def _unread(path: Path, exc: OSError) -> UnreadCandidate:
    return UnreadCandidate(Path(path), _reason(exc))


def _mode(path: Path) -> Optional[int]:
    """`path`'s `st_mode`, links followed; None when it is not there (an errno
    in `_ABSENT_ERRNOS`, or a path the platform cannot encode).

    Any other stat error is raised. Explicit, because `Path.is_file` and
    `Path.is_dir` do not keep one rule: from Python 3.14 they answer False for
    every `OSError`, so a place that could not be looked at would read as one
    that is not there.
    """
    try:
        return path.stat().st_mode
    except OSError as exc:
        if exc.errno in _ABSENT_ERRNOS:
            return None
        raise
    except ValueError:
        return None


def evidence_candidates(data_dir: Path,
                        evidence_dirs: Sequence[Path]) -> List[EvidenceCandidate]:
    """The files R10 looks at, in order, each once.

    The measurement workspace's `train_samples.json` and `val_samples.json`
    (skipped when absent: the workspace is a candidate like any other, not a
    requirement), then every `*_samples.json` directly inside each evidence
    directory, sorted by name, not recursive. Only regular files: a directory
    with a samples name can hold no bytes to compare.

    **De-duplicated by resolved path, and returned resolved,** so an evidence
    directory that is the workspace itself, or a link to a file already listed,
    does not read the same bytes twice, and the report names each file by where
    its bytes are.

    **A named evidence directory known not to be one is refused,** before any
    directory is listed: one that does not exist, is not a directory, is a
    link loop, has a name too long to exist or cannot be encoded. The operator
    named it as a place to look, and skipping it would turn their mistake into
    "not found". Whether it is one is `server_settings.is_directory`, the
    explicit stat `SHEPHERD_TRAINING_EVIDENCE_DIRS` is checked by too, so the
    command line and the Test page read a directory alike.

    **A place that cannot be looked at is returned in its place, as an
    `UnreadCandidate`, and nothing is refused for it** (K8-6r.3): an evidence
    directory that exists and cannot be listed (permission, an I/O error on a
    network mount), and a listed path whose stat or resolve fails for a reason
    other than its absence. **So is a named evidence directory whose own state
    cannot be learned** (K8-6 completion review, 2026-11-08: its stat fails
    otherwise, behind a parent this account may not enter, say): it is neither
    shown to exist nor shown absent, so it is not listed and not refused.
    `compute_r10` records each, so a scope it might have held reads
    *unverifiable* with that reason, never "not found", and the remaining
    candidates are still read. Each directory is listed with `os.scandir`
    rather than `Path.glob`, which swallows the error and yields nothing: the
    archive's files would become "not found" although none of them was looked
    at. A listed path keeps R10's own rule (`_mode`): one whose name is too
    long to stat was listed, so it is a place not looked in, not an absent
    one. With no such place the list holds paths only.
    """
    named: list[EvidenceCandidate] = []
    for given in evidence_dirs:
        directory = Path(given)
        try:
            known = is_directory(directory)
        except OSError as exc:
            named.append(_unread(directory, exc))
            continue
        if not known:
            raise ValueError(
                f"Training evidence directory {directory} does not exist or is not a "
                "directory; nothing was read or written. Name a directory that holds "
                "the *_samples.json files the checkpoint was trained on."
            )
        named.append(directory)
    listed: List[EvidenceCandidate] = [Path(data_dir) / name for name in _WORKSPACE_CANDIDATES]
    for directory in named:
        listed.extend([directory] if isinstance(directory, UnreadCandidate)
                      else _evidence_names(directory))
    candidates: List[EvidenceCandidate] = []
    seen = set()
    for item in listed:
        if isinstance(item, Path):
            try:
                mode = _mode(item)
                if mode is None or not stat.S_ISREG(mode):
                    continue
                item = item.resolve()
            except OSError as exc:
                item = _unread(item, exc)
        if item not in seen:
            seen.add(item)
            candidates.append(item)
    return candidates


def _evidence_names(directory: Path) -> List[EvidenceCandidate]:
    """The paths in `directory` whose names match `_EVIDENCE_PATTERN`, sorted
    by name; or, when it cannot be listed, the directory itself as one
    `UnreadCandidate` naming the error."""
    try:
        with os.scandir(directory) as entries:
            names = [entry.name for entry in entries
                     if fnmatch.fnmatchcase(entry.name, _EVIDENCE_PATTERN)]
    except OSError as exc:
        return [_unread(directory, exc)]
    return [directory / name for name in sorted(names)]


def compute_r10(
    recorded: Optional[Mapping[str, Any]],
    test_samples: Sequence[DiagnosisSample],
    candidates: Sequence[EvidenceCandidate],
) -> Dict[str, Any]:
    """The R10 block for one test cohort against one checkpoint's record.

    Args:
        recorded: the checkpoint's `training_input_digests`, as read (`None` when
            it has none). A role whose value is not a non-empty string names no
            digest to look for: its evidence is `not_recorded`, with a problem
            naming the role when the role is there at all. Only an absent
            `val_samples` beside a recorded `train_samples` is *no validation
            inputs recorded*; a malformed one leaves validation *unverifiable*.
        test_samples: the test cohort's samples, as the measurement reads them.
        candidates: the files to look at, from `evidence_candidates`, with any
            place it could not look at in its place (`UnreadCandidate`).

    Pure apart from the candidate reads. A candidate is read only while some
    recorded digest is still unmatched, against all of them at once: a
    mismatch or a file that has vanished since it was listed is skipped; a
    match that does not parse makes its scope's evidence `unreadable`. **Any
    other failure to read a candidate** (permission, a directory put in its
    place, an I/O error) **is recorded, and the search goes on** (K8-6r.3): the
    candidate may have been the evidence, so a scope that no candidate read
    holds, while such a place exists, is `search_incomplete`, with path None
    and a problem naming each place and its reason, never `not_found`. A scope
    found in another candidate stays `found`.

    `candidates_examined` counts the candidates whose bytes were read and
    compared; a place that could not be read is not among them.

    Raises:
        ValueError: `test_samples` is empty. "none" over nobody would read as a
            finding, and an empty cohort has no result for it to describe (Mode
            C's metrics refuse one too).
        MemoryError: memory ran out reading a candidate. A resource failure,
            not a finding about the file or about overlap, so it is never a
            state: re-raised naming the candidate, chained from the original.
    """
    if not test_samples:
        raise ValueError(
            "R10 needs the test cohort's samples; an empty cohort has nothing to "
            "compare, and 'none' over nobody would read as a finding"
        )
    record: Mapping[str, Any] = recorded if isinstance(recorded, Mapping) else {}
    wanted = {scope: _recorded_digest(record.get(role)) for scope, role in R10_SCOPES}

    test_cases: List[_CaseKey] = [
        (sample.disease_id, frozenset(sample.phenotype_ids)) for sample in test_samples
    ]
    test_diseases = frozenset(disease for disease, _ in test_cases)
    test_case_set = frozenset(test_cases)

    evidence: Dict[str, Dict[str, Any]] = {
        scope: (_evidence("not_found") if wanted[scope] is not None
                else _evidence("not_recorded", problem=_not_a_digest(record, role)))
        for scope, role in R10_SCOPES
    }
    found: Dict[str, Tuple[FrozenSet[int], FrozenSet[_CaseKey]]] = {}
    unread: List[UnreadCandidate] = []
    examined = 0
    for candidate in candidates:
        unmatched = {
            digest for scope, digest in wanted.items()
            if digest is not None and evidence[scope]["state"] == "not_found"
        }
        if not unmatched:
            break
        if isinstance(candidate, UnreadCandidate):
            unread.append(candidate)
            continue
        try:
            read = read_samples_file(Path(candidate), expected_sha256=unmatched)
        except FileNotFoundError:
            continue
        except MemoryError as exc:
            raise MemoryError(
                f"memory ran out reading the R10 evidence candidate {candidate}: a resource "
                "failure, not a finding about the file or about overlap"
            ) from exc
        except SamplesDigestMismatch:
            examined += 1
            continue
        except UnparseableFile as refused:
            examined += 1
            problem = f"the recorded bytes could not be parsed as {refused.what}"
            for scope in _scopes_wanting(wanted, evidence, refused.identity.sha256):
                evidence[scope] = _evidence(
                    "unreadable", path=str(refused.identity.path), problem=problem
                )
            continue
        except OSError as exc:
            # Never compared: it may hold the evidence. Recorded, and the
            # remaining candidates are still read.
            unread.append(_unread(Path(candidate), exc))
            continue
        examined += 1
        identity = read.identity
        keys = _keys_in(read.samples, test_diseases, test_case_set)
        # Reduced to the test keys it holds; the samples go before the next read.
        del read
        for scope in _scopes_wanting(wanted, evidence, identity.sha256):
            evidence[scope] = _evidence("found", path=str(identity.path))
            found[scope] = keys
    if unread:
        # "not_found" would say every candidate was read and compared.
        problem = (
            "no candidate that was read holds the recorded bytes, and these places "
            "could not be looked in: "
            + "; ".join(f"{place.path} ({place.reason})" for place in unread)
        )
        for scope in evidence:
            if evidence[scope]["state"] == "not_found":
                evidence[scope] = _evidence("search_incomplete", problem=problem)

    # Only an absent entry says training ran no validation pass. One that is
    # there and is not a digest names validation inputs that cannot be looked
    # for, and "none recorded" would say more than the record does.
    no_validation_pass = (
        wanted["training"] is not None and _VALIDATION_ROLE not in record
    )
    block: Dict[str, Any] = {
        "definition_version": R10_DEFINITION_VERSION,
        "candidates_examined": examined,
    }
    for scope, _ in R10_SCOPES:
        if scope in found:
            state = None  # each label's state is counted from the evidence
        elif scope == "validation" and no_validation_pass:
            state = "no_validation_inputs_recorded"
        else:
            state = "unverifiable"
        block[scope] = {
            "recorded_sha256": wanted[scope],
            "evidence": evidence[scope],
            **_labels(found.get(scope), state, test_cases, test_diseases),
        }
    return block


def _recorded_digest(value: Any) -> Optional[str]:
    """The recorded digest, or `None` when the role is absent or not a non-empty
    string (the same test as `verify_training_record`'s `if not recorded`)."""
    return value if isinstance(value, str) and value else None


def _not_a_digest(record: Mapping[str, Any], role: str) -> Optional[str]:
    """The problem for a role that is recorded with a value that is not a
    digest, or `None` when the role is absent. Names the role and the value's
    kind, never the value: it is not one the pipeline writes."""
    if role not in record:
        return None
    value = record[role]
    if isinstance(value, str):
        kind = "an empty string"
    elif value is None:
        kind = "null"
    else:
        kind = f"a value of type {type(value).__name__}"
    return (
        f"the checkpoint records {role} as {kind}, not a SHA-256 string, so no file "
        "can be looked for by it; it is not a record the pipeline wrote"
    )


def _evidence(state: str, *, path: Optional[str] = None,
              problem: Optional[str] = None) -> Dict[str, Any]:
    return {"state": state, "path": path, "problem": problem}


def _scopes_wanting(
    wanted: Mapping[str, Optional[str]], evidence: Mapping[str, Mapping[str, Any]], sha256: str
) -> List[str]:
    """The scopes still unmatched whose recorded digest is `sha256`; both, when
    training and validation record the same bytes."""
    return [
        scope for scope, digest in wanted.items()
        if digest == sha256 and evidence[scope]["state"] == "not_found"
    ]


def _keys_in(
    samples: Iterable[Any], test_diseases: FrozenSet[int], test_case_set: FrozenSet[_CaseKey]
) -> Tuple[FrozenSet[int], FrozenSet[_CaseKey]]:
    """The test diseases and test cases that occur in one evidence file.

    Only rows whose disease is a test disease build a phenotype set, so a large
    training file costs one membership test per row.
    """
    diseases = set()
    cases = set()
    for sample in samples:
        if sample.disease_id in test_diseases:
            diseases.add(sample.disease_id)
            key = (sample.disease_id, frozenset(sample.phenotype_ids))
            if key in test_case_set:
                cases.add(key)
    return frozenset(diseases), frozenset(cases)


def _labels(
    keys: Optional[Tuple[FrozenSet[int], FrozenSet[_CaseKey]]],
    state: Optional[str],
    test_cases: Sequence[_CaseKey],
    test_diseases: FrozenSet[int],
) -> Dict[str, Dict[str, Any]]:
    """The scope's two labels: counted from `keys` when there is evidence, else
    `state` with `k` and `cases` left `None`."""
    n_diseases, n_cases = len(test_diseases), len(test_cases)
    if keys is None:
        return {
            "disease_overlap": {"state": state, "k": None, "n": n_diseases, "cases": None},
            "phenotype_set_overlap": {"state": state, "k": None, "n": n_cases},
        }
    diseases, cases = keys
    disease_k = len(diseases)
    disease_cases = sum(1 for disease, _ in test_cases if disease in diseases)
    phenotype_k = sum(1 for key in test_cases if key in cases)
    return {
        "disease_overlap": {
            "state": "overlap" if disease_k else "none",
            "k": disease_k, "n": n_diseases, "cases": disease_cases,
        },
        "phenotype_set_overlap": {
            "state": "overlap" if phenotype_k else "none",
            "k": phenotype_k, "n": n_cases,
        },
    }
