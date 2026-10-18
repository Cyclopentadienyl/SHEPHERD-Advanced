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

**Four states per label, and a number only where there is evidence.** *overlap*
and *none* need the evidence found and parsed, and carry `k`. Two states carry
`k = None` and `cases = None`, never 0, so no reader can take either for "no
leakage":
- *unverifiable*: the recorded bytes were not found, or could not be parsed, or
  the record names no training file at all, so it is not one the pipeline wrote;
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
and a test case `[0, 0, 1]` of one disease overlap although Mode C, which
averages every listed position, may score them differently; that is the case
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
parse as, never a row of them; for a role recorded with a value that is not a
digest, a problem naming the role and the value's kind, never the value. The block is copied into the report and into the
ledger beside the checkpoint, and a checkpoint can sit inside the repository
tree.

Module: src/evaluation/r10.py
"""
from __future__ import annotations

import fnmatch
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Tuple

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
    "compute_r10",
    "evidence_candidates",
]

#: Bumped when a unit, a state or the evidence rule changes, so two blocks are
#: compared only under one definition.
R10_DEFINITION_VERSION = 1

#: Every state a label can take. Only the first two carry a count.
R10_STATES: Tuple[str, ...] = ("overlap", "none", "unverifiable", "no_validation_inputs_recorded")

#: What became of a scope's evidence: found and parsed; recorded but held by no
#: candidate; held by a candidate that does not parse; not recorded at all.
EVIDENCE_STATES: Tuple[str, ...] = ("found", "not_found", "unreadable", "not_recorded")

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

_CaseKey = Tuple[int, FrozenSet[int]]


def evidence_candidates(data_dir: Path, evidence_dirs: Sequence[Path]) -> List[Path]:
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

    **An evidence directory that does not exist or is not a directory is
    refused,** before any directory is listed: the operator named it as a place
    to look, and skipping it would turn their mistake into "not found".

    **So is one that cannot be listed** (permission, an I/O error on a network
    mount), before any candidate is read. Each directory is listed with
    `os.scandir` rather than `Path.glob`, which swallows the error and yields
    nothing: the archive's files would become "not found" although none of them
    was looked at, while one unreadable file inside a listable directory refuses
    the run (`compute_r10`).
    """
    directories = [Path(directory) for directory in evidence_dirs]
    for directory in directories:
        if not directory.is_dir():
            raise ValueError(
                f"Training evidence directory {directory} does not exist or is not a "
                "directory; nothing was read or written. Name a directory that holds "
                "the *_samples.json files the checkpoint was trained on."
            )
    listed = [Path(data_dir) / name for name in _WORKSPACE_CANDIDATES]
    for directory in directories:
        listed.extend(directory / name for name in _evidence_names(directory))
    candidates: List[Path] = []
    seen = set()
    for path in listed:
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(resolved)
    return candidates


def _evidence_names(directory: Path) -> List[str]:
    """The names in `directory` that match `_EVIDENCE_PATTERN`, sorted, or a
    refusal naming the directory when it cannot be listed."""
    try:
        with os.scandir(directory) as entries:
            names = [entry.name for entry in entries
                     if fnmatch.fnmatchcase(entry.name, _EVIDENCE_PATTERN)]
    except OSError as exc:
        reason = exc.strerror or (f"errno {exc.errno}" if exc.errno is not None
                                  else "no reason given")
        raise ValueError(
            f"Training evidence directory {directory} could not be listed "
            f"({type(exc).__name__}: {reason}); nothing was read or written. A "
            "directory that is not looked in would report the checkpoint's inputs as "
            "not found; name one this process can list."
        ) from exc
    return sorted(names)


def compute_r10(
    recorded: Optional[Mapping[str, Any]],
    test_samples: Sequence[DiagnosisSample],
    candidates: Sequence[Path],
) -> Dict[str, Any]:
    """The R10 block for one test cohort against one checkpoint's record.

    Args:
        recorded: the checkpoint's `training_input_digests`, as read (`None` when
            it has none). A role whose value is not a non-empty string names no
            digest to look for: its evidence is `not_recorded`, with a problem
            naming the role when the role is there at all. Only an absent
            `val_samples` beside a recorded `train_samples` is *no validation
            inputs recorded*; a malformed one leaves validation *unverifiable*.
        test_samples: the test cohort's samples, as Mode C reads them.
        candidates: the files to look at, from `evidence_candidates`.

    Pure apart from the candidate reads. A candidate is read only while some
    recorded digest is still unmatched, against all of them at once: a
    mismatch or a file that has vanished since it was listed is skipped; a
    match that does not parse makes its scope's evidence `unreadable`. Any other
    failure to read a candidate (permission, a directory put in its place)
    propagates: it may have been the evidence, and "not found" would say more
    than is known.

    `candidates_examined` counts the candidates whose bytes were read and
    compared.

    Raises:
        ValueError: `test_samples` is empty. "none" over nobody would read as a
            finding, and an empty cohort has no result for it to describe (Mode
            C's metrics refuse one too).
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
    examined = 0
    for candidate in candidates:
        unmatched = {
            digest for scope, digest in wanted.items()
            if digest is not None and evidence[scope]["state"] == "not_found"
        }
        if not unmatched:
            break
        try:
            read = read_samples_file(Path(candidate), expected_sha256=unmatched)
        except FileNotFoundError:
            continue
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
        examined += 1
        identity = read.identity
        keys = _keys_in(read.samples, test_diseases, test_case_set)
        # Reduced to the test keys it holds; the samples go before the next read.
        del read
        for scope in _scopes_wanting(wanted, evidence, identity.sha256):
            evidence[scope] = _evidence("found", path=str(identity.path))
            found[scope] = keys

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
