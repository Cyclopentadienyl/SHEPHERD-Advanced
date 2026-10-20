"""
Test cohort versions: a source case set mapped to one graph, kept outside the repository.
========================================================================================
Contract M4 (item 14, phase 1; `docs/working/PLAN_TEST_RESULTS.md` D6, D7). A
supplied cohort used to be a loose `<split>_samples.json` dropped into a
workspace: integer indices with nothing saying which graph's node order they
index, which source file they came from, or which cases were left out on the
way. An in-range index from another graph passed every check.

A **cohort version** closes that. The importer reads the operator's source file
once, maps it against the graph the workspace manifest binds, and publishes
three things into a registry the operator keeps outside the repository:

- ``sources/<sha256>.json`` -- the source bytes, verbatim, named by their digest;
- ``versions/<label>/samples.json`` -- the mapped samples, in the format
  `read_samples` parses;
- ``versions/<label>/cohort.json`` -- the version manifest, **written last**: it
  binds the samples by digest, names the source by digest, records the four
  graph digests and the dataset record it was mapped against, the mapping
  policy, and every count a reader needs to see what was excluded and why. Its
  presence is what makes a version exist.

**Immutable.** A label is taken by ``mkdir(exist_ok=False)``, so two imports
cannot both claim one. A version directory without ``cohort.json`` is an
interrupted import: the resolver refuses it and the importer refuses its label,
naming the directory for the operator to remove. Nothing here deletes anything.

**Patient-level data never enters the repository tree** (D7). The registry is a
path the operator gives on the command line, or to the service as
``SHEPHERD_REGISTRY`` (M5-1, `src.config.server_settings`, one rule for both); it
must be absolute, must already exist, and is refused when it resolves inside the
repository. Nothing creates a registry root, and no request ever names one. Each
directory below the root is resolved again right before content is written into
it, or a version is read from it, and refused when a link takes it outside the
registry or into the repository.

**One mapping rule.** A phenotype maps when ``hpo:<term>`` is a phenotype node
and a disease when ``mondo:<term>`` is a disease node -- serving's lookup -- and
the mapped list then goes through `normalise_phenotypes` (N1), the one rule for
repeats. An obsolete or alternative HPO id is not a node, so it is unmappable
(OWNER Q4). A disease named in OMIM or ORPHA refuses the import as a whole
(OWNER Q3): the graph's diseases are MONDO terms, and translating needs a MONDO
file verified against the dataset record, which is later work.

**How the counts are taken** (each recorded in the manifest):

- an excluded case has one primary ``category``, the first of
  `EXCLUSION_CATEGORIES` that applies, and lists every reason that applied;
- per case, ``terms_total`` and ``terms_unmappable`` count the entries the
  source listed, so a repeated unmappable term counts each time it is listed;
- ``unmappable_terms`` maps each HPO term that did not map to the number of
  source cases listing it, excluded cases included; ``terms_unmappable`` in the
  counts is the number of such distinct terms;
- ``cases_with_lost_terms``, ``repeats_removed`` and ``cases_with_repeats``
  describe the **ranked** cases, the samples written: how many are scored on
  fewer terms than the source listed, and what normalisation removed.

Module: src/evaluation/cohort_registry.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from src.config.server_settings import DirectoryRefused, check_registry_root
from src.evaluation.cohort import GENERATED_SPLITS, validate_split_name
from src.kg.artifacts import GRAPH_ARTIFACTS
from src.kg.phenotype_normalisation import PHENOTYPE_NORMALISATION_VERSION, normalise_phenotypes
from src.utils.fingerprint import ReadIdentity

if TYPE_CHECKING:
    from src.kg.artifacts import DatasetRecord

#: The repository this module is part of. A registry resolving inside it is
#: refused (D7), and so is a directory below a registry that resolves inside it.
#: Computed from this file, as `src/config/runtime_presets.py` does, and
#: compared after `resolve()`, so a symlink into the tree is caught.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The one source format this revision reads: the project's shared import
#: format. Its name says how a file is laid out, not whether its cases are
#: synthetic or real; a real format gets an adapter to it when samples exist
#: (OWNER Q5).
SOURCE_FORMAT = "shepherd-cases/1"

#: Bumped when the version manifest's shape changes in a way an old reader
#: cannot read. A reader refuses any other value rather than guessing.
VERSION_SCHEMA_VERSION = 1

VERSION_MANIFEST = "cohort.json"
VERSION_SAMPLES = "samples.json"
SOURCES_DIR = "sources"
VERSIONS_DIR = "versions"
REPORTS_DIR = "reports"

#: What a case that lost some, but not all, of its terms in mapping becomes:
#: ranked on the terms that mapped, or excluded. Recorded per version (OWNER Q1).
PARTIAL_PHENOTYPE_POLICIES: Tuple[str, ...] = ("keep", "exclude")
DEFAULT_PARTIAL_PHENOTYPES = "keep"

#: Why a source case is not ranked, in precedence order: an excluded case's
#: category is the first that applies, and its entry lists every one that did.
EXCLUSION_CATEGORIES: Tuple[str, ...] = ("ground_truth_absent", "phenotype_unmappable")

WHAT_THIS_IS = "a test cohort version: a source case set mapped to one graph's node indices"

#: Disease namespaces a source may use that this revision does not map (OWNER
#: Q3). A case naming one is not malformed; the import is refused as a whole.
UNSUPPORTED_DISEASE_NAMESPACES: Tuple[str, ...] = ("OMIM", "ORPHA", "Orphanet")

_SOURCE_KEYS = frozenset({"format", "cases"})
_CASE_KEYS = frozenset({"case_id", "phenotypes", "disease"})
# ASCII digits, not `\d`, which also matches digits of other scripts.
_HPO_TERM = re.compile(r"HP:[0-9]{7}")
_MONDO_TERM = re.compile(r"MONDO:[0-9]{7}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PREVIEW_ID = re.compile(r"[0-9a-f]{16}")

#: Every field `commit_import` writes. A manifest missing one is not a version
#: this revision reads.
REQUIRED_VERSION_FIELDS: Tuple[str, ...] = (
    "schema_version", "what_this_is", "label", "role", "source", "graph",
    "dataset_record", "mapping", "counts", "samples", "unmappable_terms",
    "exclusions", "imported",
)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
class RegistryRefused(ValueError):
    """The registry path is not one this project keeps patient-level data in,
    or the registry's contents contradict themselves."""


class MalformedSource(ValueError):
    """The source does not fit `SOURCE_FORMAT`. The import is refused as a whole,
    and `problems` lists every problem found, by case index and case id (D6)."""

    def __init__(self, source: str, problems: Sequence[str], also: str = "") -> None:
        self.problems: Tuple[str, ...] = tuple(problems)
        listed = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(
            f"{source} is not a {SOURCE_FORMAT} source ({len(self.problems)} "
            f"problem{'s' if len(self.problems) != 1 else ''}); the import is refused "
            f"as a whole:\n{listed}{also}"
        )


class UnsupportedDiseaseNamespace(ValueError):
    """Cases name their ground truth in a namespace this revision does not map.
    `namespaces` maps each namespace to the number of cases using it."""

    def __init__(self, source: str, namespaces: Mapping[str, int]) -> None:
        self.namespaces: Dict[str, int] = dict(namespaces)
        super().__init__(
            f"{source} names the ground truth of {sum(self.namespaces.values())} "
            f"case(s) outside MONDO ({_namespace_counts(self.namespaces)}). The "
            "graph's diseases are MONDO terms and this revision maps MONDO ids only; "
            "translating needs a MONDO file verified against the dataset record, "
            "which is not built. The import is refused as a whole, so no case is "
            "dropped silently."
        )


class CohortVersionRefused(ValueError):
    """A cohort version cannot be created, or one that exists cannot be read."""


class CohortBindingMismatch(ValueError):
    """A cohort version was mapped against another graph or dataset record than
    the run's. `roles` names every role that differs."""

    def __init__(self, message: str, roles: Sequence[str]) -> None:
        super().__init__(message)
        self.roles: Tuple[str, ...] = tuple(roles)


def _namespace_counts(namespaces: Mapping[str, int]) -> str:
    return ", ".join(f"{name}: {count}" for name, count in sorted(namespaces.items()))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
class SourceCase(NamedTuple):
    """One case as the source states it, after the format checks."""

    case_id: str
    phenotypes: Tuple[str, ...]
    disease: str


class WorkspaceGraph(NamedTuple):
    """The graph a workspace manifest binds, read and verified once each.

    `bound` is the four graph digests the manifest binds (the same dict R3
    compares a checkpoint with); `node_mapping` is the parsed `kg.json`'s
    per-type node index, the graph object itself is not kept."""

    bound: Dict[str, str]
    dataset_record: "DatasetRecord"
    node_mapping: Dict[str, Dict[str, int]]


class MappedCohort(NamedTuple):
    """What mapping made of the source: the samples to rank and everything left out."""

    samples: List[Dict[str, Any]]
    exclusions: List[Dict[str, Any]]
    counts: Dict[str, Any]
    unmappable_terms: Dict[str, int]


class ImportPlan(NamedTuple):
    """Everything a commit writes, computed from one read of each input.

    `manifest` is the version manifest without its ``imported`` block, which is
    what `preview_id` covers. `source_bytes` are the bytes parsed, kept so the
    copy in the registry is the file that was mapped, not a second read of it.
    """

    label: str
    role: str
    manifest: Dict[str, Any]
    preview_id: str
    source_sha256: str
    source_bytes: bytes
    samples_bytes: bytes


class CohortVersionRead(NamedTuple):
    """A committed version as resolved: its manifest, the identity of the bytes
    that manifest was parsed from, its directory, and where its samples are."""

    manifest: Dict[str, Any]
    identity: ReadIdentity
    directory: Path
    samples_path: Path


# ---------------------------------------------------------------------------
# Registry, labels and roles
# ---------------------------------------------------------------------------
def validate_registry(path: Any) -> Path:
    """The registry root, resolved, or a refusal (§0.1).

    **Absolute,** because a relative path means whatever the working directory
    makes of it, and the project's settings resolve relative paths against the
    repository root -- the one place the registry must not be. **Outside the
    repository after `resolve()`,** so a symlink into the tree is caught.
    **Existing,** because the operator creates the registry; a mistyped path
    must not become a new registry nobody chose.

    Returns the resolved path, which every later step joins onto, so the root
    link swapped after this check cannot move what is written.

    **This checks the root only.** Each directory below it that content is
    written into (``sources/``, ``versions/``, ``versions/<label>/``, its
    ``reports/`` and a run directory), and the version directory a reader
    returns, is resolved again right before that use by `_registry_directory`,
    which refuses one that a link takes outside the registry or into the
    repository. Those checks run just before each write, not atomically with
    it: a link swapped in the moment between is not caught. The registry is the
    operator's, on the machine, and one writer at a time (D7).

    **One rule for `--registry` and the service's `SHEPHERD_REGISTRY`**
    (`src.config.server_settings.check_registry_root`, M5-1): it lives below
    this layer because the service's settings are read there too.
    """
    try:
        return check_registry_root(path)
    except DirectoryRefused as exc:
        raise RegistryRefused(str(exc)) from None


def _registry_directory(path: Path, *, within: Optional[Path], done: str) -> Path:
    """`path` resolved, or a refusal: a directory below the registry, checked
    right before content is written into it or a version is read from it.

    `validate_registry` checks the root, and joining plain names onto it does
    not stay below it: a ``sources``, ``versions``, ``versions/<label>`` or
    ``reports`` that is a link (made by hand, or kept by a copy of an older
    registry) would carry the source bytes, the mapped samples and the
    per-case ranks wherever it points, the repository included, while every
    root check passed. So each such directory is resolved again and refused
    when it resolves inside the repository (D7) or outside `within`, the
    resolved directory it belongs under (the registry root, or a version
    directory that was itself checked against the root); `within` is `None`
    only for a directory whose own place is all there is to check.

    Returns the resolved path, which the caller then writes through. `done`
    says what was written before this refusal, so the operator's message is
    true at every step.
    """
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError) as exc:  # a link loop: RuntimeError before Python 3.13
        raise RegistryRefused(
            f"{path} cannot be resolved ({type(exc).__name__}: {exc}). {done}"
        ) from None
    repo = REPO_ROOT.resolve()
    if resolved == repo or repo in resolved.parents:
        where = (f"inside the repository ({repo}). Patient-level data is never kept in the "
                 "repository tree (D7)")
    elif within is not None and resolved != within and within not in resolved.parents:
        where = f"outside {within}, the directory it belongs under"
    else:
        return resolved
    raise RegistryRefused(
        f"{path} resolves to {resolved}, {where}: a link inside the registry would carry "
        f"patient-level files there. Replace the link with a directory. {done}"
    )


def validate_label(label: Any) -> str:
    """A version label: the identifier rule split names follow (one rule, not a
    second regex), because it becomes a directory name and a ledger value."""
    try:
        return validate_split_name(label)
    except ValueError as exc:
        raise CohortVersionRefused(f"the cohort label is refused: {exc}") from None


def validate_role(role: Any) -> str:
    """A cohort role: the identifier rule, and never a generator split name.

    `train` and `val` name the generator's cohorts; a test cohort carrying one
    would read as a generated cohort in every artifact downstream, the hazard
    `src/evaluation/cohort.py` refuses for a supplied split.
    """
    try:
        validate_split_name(role)
    except ValueError as exc:
        raise CohortVersionRefused(f"the cohort role is refused: {exc}") from None
    if role in GENERATED_SPLITS:
        raise CohortVersionRefused(
            f"the cohort role {role!r} is one of the generator's own split names "
            f"({', '.join(GENERATED_SPLITS)}); a test cohort carrying it would read as "
            "a generated cohort. Give it its own role, such as mygene2."
        )
    return role


def validate_partial_policy(policy: Any) -> str:
    if policy not in PARTIAL_PHENOTYPE_POLICIES:
        raise CohortVersionRefused(
            f"the partial-phenotype policy must be one of "
            f"{', '.join(PARTIAL_PHENOTYPE_POLICIES)}, got {policy!r}"
        )
    return policy


def _require_label_free(root: Path, label: str) -> None:
    """Refuse a label that is taken, naming what is there.

    A check before writing, for a clear message; `commit_import`'s
    ``mkdir(exist_ok=False)`` is what makes the claim atomic."""
    directory = root / VERSIONS_DIR / label
    if not os.path.lexists(directory):
        return
    if (directory / VERSION_MANIFEST).is_file():
        raise CohortVersionRefused(
            f"the cohort version {label!r} already exists ({directory}). Versions are "
            "immutable; import under a new label."
        )
    raise CohortVersionRefused(
        f"{directory} exists without {VERSION_MANIFEST}: an interrupted import, not a "
        "version. Nothing removes it automatically; remove the directory to reuse the "
        "label, or import under a new one."
    )


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------
def _refuse_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """`json` keeps the last of two equal keys silently; a case stating its
    disease twice is ambiguous, so it is refused instead."""
    seen: Dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"a JSON object states the key {key!r} twice")
        seen[key] = value
    return seen


def _unsupported_namespace(disease: Any) -> Optional[str]:
    """The namespace of a disease id in `UNSUPPORTED_DISEASE_NAMESPACES`, else `None`."""
    if not isinstance(disease, str):
        return None
    namespace, colon, _ = disease.partition(":")
    return namespace if colon and namespace in UNSUPPORTED_DISEASE_NAMESPACES else None


def _json_kind(value: Any) -> str:
    """What a JSON value is, without the value itself."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, str):
        return "a string" if value else "an empty string"
    if isinstance(value, list):
        return "a list" if value else "an empty list"
    if isinstance(value, dict):
        return "an object"
    return type(value).__name__


def _not_an_id(value: Any, rule: str) -> str:
    """How a refusal describes a phenotype term or disease that is not an id:
    by its type, or the rule a string broke -- never by the value, which is a
    patient's data (D7)."""
    if isinstance(value, str) and value:
        return f"is not {rule}"
    return f"is {_json_kind(value)}, not {rule}"


def _printable_text(value: str) -> bool:
    """Printable and encodable as UTF-8: no control or format character, no
    separator but the space, and no lone surrogate (which JSON's ``\\ud800``
    escape makes). A case id is printed in the preview and kept as text in
    the version; a newline in one could forge a line of the preview, and a lone
    surrogate cannot be printed at all."""
    if not value.isprintable():
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def parse_source(data: bytes, *, source: Any = "the source") -> List[SourceCase]:
    """Parse a `SOURCE_FORMAT` source strictly (§0.4).

    ``{"format": "shepherd-cases/1", "cases": [{"case_id", "phenotypes",
    "disease"}, ...]}`` with exactly these keys at both levels: `case_id` a
    non-empty string of printable text, unique; `phenotypes` a non-empty list
    of ``HP:`` ids; `disease` a ``MONDO:`` id. **Every problem is collected
    before refusing,** so an operator fixes a file in one pass, not one error
    at a time. A disease in `UNSUPPORTED_DISEASE_NAMESPACES` is not malformed
    and is counted apart.

    **A problem names, never echoes, a case's data** (D7): each is located by
    case index, case id, field and position, and says what type the value was
    or which rule it broke. A refusal is printed, and may be kept in a console
    log, so it never ties a phenotype list or a diagnosis to a case id. A case
    id that is not printable text is shown escaped (`ascii`).

    `source` names the file in messages; `data` is the bytes of its one read.

    Raises:
        MalformedSource: listing every problem by case index and case id.
        UnsupportedDiseaseNamespace: the source is well formed and some cases
            name their disease in a namespace this revision does not map.
    """
    try:
        text = data.decode("utf-8")
        parsed = json.loads(text, object_pairs_hook=_refuse_duplicate_keys)
    except (ValueError, RecursionError) as exc:
        problem = f"not UTF-8 JSON ({type(exc).__name__}: {exc})"
        raise MalformedSource(str(source), [problem]) from None

    problems: List[str] = []
    if not isinstance(parsed, dict):
        problem = f"the top level is not a JSON object ({type(parsed).__name__})"
        raise MalformedSource(str(source), [problem])
    extra = sorted(set(parsed) - _SOURCE_KEYS)
    if extra:
        problems.append(f"the top level has keys this format does not define: {extra}")
    if parsed.get("format") != SOURCE_FORMAT:
        problems.append(
            f"format is {parsed.get('format')!r}; this importer reads {SOURCE_FORMAT!r} only"
        )
        raise MalformedSource(str(source), problems)
    cases = parsed.get("cases")
    if not isinstance(cases, list):
        problems.append(f"cases is not a list ({type(cases).__name__})")
        raise MalformedSource(str(source), problems)
    if not cases:
        problems.append("cases is empty; a source with no cases is not a cohort")

    result: List[SourceCase] = []
    first_index: Dict[str, int] = {}
    foreign: Counter = Counter()
    for index, case in enumerate(cases):
        where = f"cases[{index}]"
        if not isinstance(case, dict):
            problems.append(f"{where} is not a JSON object ({type(case).__name__})")
            continue
        case_id = case.get("case_id")
        case_id_ok = isinstance(case_id, str) and bool(case_id) and _printable_text(case_id)
        if case_id_ok:
            where = f"cases[{index}] (case_id {case_id!r})"
            if case_id in first_index:
                problems.append(f"{where}: case_id repeats cases[{first_index[case_id]}]")
            else:
                first_index[case_id] = index
        elif isinstance(case_id, str) and case_id:
            problems.append(
                f"{where}: case_id {ascii(case_id)} is not printable text (it holds a control "
                "or format character, a separator other than the space, or a lone "
                "surrogate); a case id is printed in the preview and kept as text"
            )
        else:
            problems.append(f"{where}: case_id must be a non-empty string, got "
                            f"{_json_kind(case_id)}")
        missing = sorted(_CASE_KEYS - set(case))
        extra = sorted(set(case) - _CASE_KEYS)
        if missing:
            problems.append(f"{where}: missing {missing}")
        if extra:
            problems.append(f"{where}: keys this format does not define: {extra}")

        # A key that is absent was reported above; its value is not judged again.
        phenotypes = case.get("phenotypes")
        phenotypes_ok = False
        if "phenotypes" in case:
            if isinstance(phenotypes, list) and phenotypes:
                bad = [(position, term) for position, term in enumerate(phenotypes)
                       if not (isinstance(term, str) and _HPO_TERM.fullmatch(term))]
                for position, term in bad:
                    problems.append(f"{where}: phenotypes[{position}] "
                                    f"{_not_an_id(term, 'an HPO id (HP: and seven digits)')}")
                phenotypes_ok = not bad
            else:
                problems.append(f"{where}: phenotypes must be a non-empty list, got "
                                f"{_json_kind(phenotypes)}")

        disease = case.get("disease")
        disease_ok = isinstance(disease, str) and bool(_MONDO_TERM.fullmatch(disease))
        namespace = _unsupported_namespace(disease)
        if namespace is not None:
            foreign[namespace] += 1
        elif "disease" in case and not disease_ok:
            problems.append(
                f"{where}: disease {_not_an_id(disease, 'a MONDO id (MONDO: and seven digits)')}"
            )

        if phenotypes_ok and disease_ok and case_id_ok:
            result.append(SourceCase(case_id, tuple(phenotypes), disease))

    if problems:
        also = (f"\nAlso, the ground truth of {sum(foreign.values())} case(s) is outside "
                f"MONDO ({_namespace_counts(foreign)}), which this revision does not map."
                if foreign else "")
        raise MalformedSource(str(source), problems, also)
    if foreign:
        raise UnsupportedDiseaseNamespace(str(source), foreign)
    return result


# ---------------------------------------------------------------------------
# The graph, and the mapping
# ---------------------------------------------------------------------------
def read_workspace_graph(data_dir: Path) -> WorkspaceGraph:
    """Read the graph a workspace manifest binds through the existing chain (§0.5).

    `read_split_manifest` → `KnowledgeGraph.read_json` → `read_graph_artifacts`
    → `verify_graph_source_read` → `verify_dataset_record`, one read each, the
    serving composition. **The tensors are read although only `kg.json` is
    used:** `verify_graph_reads` refuses a role that was not read, because the
    manifest binds all four files of one export, and a version records all four
    digests so measurement can compare them with the graph it scores against.
    The tensors are released as soon as they are verified.

    Raises:
        ValueError (and `DatasetRecordRefused`): from the readers and verifiers,
            naming the file.
        FileNotFoundError: `kg.json` is absent.
    """
    from src.kg.artifacts import (
        read_split_manifest,
        verify_dataset_record,
        verify_graph_source_read,
    )
    from src.kg.graph import KnowledgeGraph
    from src.kg.storage.file_storage import read_graph_artifacts

    data_dir = Path(data_dir)
    manifest = read_split_manifest(data_dir)
    graph = KnowledgeGraph.read_json(data_dir / GRAPH_ARTIFACTS["kg"])
    reads = read_graph_artifacts(data_dir, map_location="cpu").reads
    bound = verify_graph_source_read(manifest, graph.identity, reads)
    record = verify_dataset_record(manifest, bound["kg"])
    return WorkspaceGraph(dict(bound), record, graph.kg.get_node_id_mapping())


def map_cases(
    cases: Sequence[SourceCase],
    node_mapping: Mapping[str, Mapping[str, int]],
    partial_policy: str,
) -> MappedCohort:
    """Map each case to per-type node indices, then normalise its phenotypes (§0.5).

    A phenotype key is ``str(NodeID(HPO, term))`` in the graph's ``phenotype``
    index and a disease key ``str(NodeID(MONDO, term))`` in its ``disease``
    index, serving's lookup. The mapped indices and their source positions go
    through `normalise_phenotypes`, so repeats are removed by node, first
    occurrence kept, exactly as serving does.

    A case is excluded when its disease is not a disease node
    (`ground_truth_absent`), when no term maps, or -- under ``exclude`` -- when
    any term does not map (`phenotype_unmappable`). A case with no mapped term
    is always excluded: Mode C refuses an empty phenotype list.
    """
    from src.core.types import DataSource, NodeID

    validate_partial_policy(partial_policy)
    # A graph without a node type maps nothing of that type; every case then
    # lands in an exclusion category, which says so.
    phenotype_index = node_mapping.get("phenotype") or {}
    disease_index = node_mapping.get("disease") or {}

    samples: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    excluded = dict.fromkeys(EXCLUSION_CATEGORIES, 0)
    unmappable_terms: Counter = Counter()
    lost = repeats_removed = cases_with_repeats = 0

    for case in cases:
        disease_id = disease_index.get(str(NodeID(DataSource.MONDO, case.disease)))
        mapped: List[int] = []
        positions: List[int] = []
        unmapped: List[str] = []
        for position, term in enumerate(case.phenotypes):
            index = phenotype_index.get(str(NodeID(DataSource.HPO, term)))
            if index is None:
                unmapped.append(term)
            else:
                mapped.append(index)
                positions.append(position)
        unmappable_terms.update(set(unmapped))

        reasons: List[str] = []
        if disease_id is None:
            reasons.append("ground_truth_absent")
        if not mapped or (unmapped and partial_policy == "exclude"):
            reasons.append("phenotype_unmappable")
        if reasons:
            excluded[reasons[0]] += 1
            exclusions.append({
                "case_id": case.case_id,
                "category": reasons[0],
                "reasons": reasons,
                "terms_total": len(case.phenotypes),
                "terms_unmappable": len(unmapped),
            })
            continue

        normalised = normalise_phenotypes(mapped, positions=positions)
        if unmapped:
            lost += 1
        if normalised.repeats_removed:
            cases_with_repeats += 1
            repeats_removed += normalised.repeats_removed
        samples.append({
            "patient_id": case.case_id,
            "phenotype_ids": list(normalised.kept),
            "disease_id": disease_id,
        })

    counts = {
        "source_cases": len(cases),
        "ranked_eligible": len(samples),
        "excluded": excluded,
        "n_excluded": len(exclusions),
        "cases_with_lost_terms": lost,
        "terms_unmappable": len(unmappable_terms),
        "repeats_removed": repeats_removed,
        "cases_with_repeats": cases_with_repeats,
    }
    return MappedCohort(samples, exclusions, counts, dict(sorted(unmappable_terms.items())))


# ---------------------------------------------------------------------------
# The version manifest
# ---------------------------------------------------------------------------
def _sha256(data: bytes) -> str:
    """The digest of bytes in memory: the samples about to be written, and the
    canonical form of a would-be manifest."""
    return hashlib.sha256(data).hexdigest()


def encode_json(value: Any) -> bytes:
    """How every registry file is written: ASCII, sorted keys, two-space
    indent, and no NaN or infinity (which no JSON reader is obliged to take)."""
    return json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True,
                      allow_nan=False).encode("ascii")


def _source_entry(record: "DatasetRecord", role: str) -> Optional[Dict[str, Any]]:
    """The dataset record's entry for one ontology, or `None` when it names none
    (a synthetic record names no sources at all)."""
    for entry in record.record["sources"]:
        if entry["role"] == role:
            return {"digest": entry["digest"], "declared_version": entry["declared_version"]}
    return None


def build_version_manifest(
    *,
    label: str,
    role: str,
    source_sha256: str,
    source_filename: str,
    workspace: WorkspaceGraph,
    partial_policy: str,
    mapped: MappedCohort,
    samples_sha256: str,
) -> Dict[str, Any]:
    """The version manifest (§2.3) without its ``imported`` block.

    Built only from what was read and computed: the source's digest and
    basename, the four digests the workspace manifest binds, the dataset record
    the run verified (its digest, origin and HPO/MONDO entries, or `None` where
    it names none), the mapping policy, the counts and the samples' digest.
    """
    counts = mapped.counts
    return {
        "schema_version": VERSION_SCHEMA_VERSION,
        "what_this_is": WHAT_THIS_IS,
        "label": label,
        "role": role,
        "source": {
            "sha256": source_sha256,
            "filename": source_filename,
            "format": SOURCE_FORMAT,
            "n_cases": counts["source_cases"],
        },
        "graph": {role_name: workspace.bound[role_name] for role_name in GRAPH_ARTIFACTS},
        "dataset_record": {
            "sha256": workspace.dataset_record.identity.sha256,
            "origin": workspace.dataset_record.origin,
            "hpo": _source_entry(workspace.dataset_record, "hpo"),
            "mondo": _source_entry(workspace.dataset_record, "mondo"),
        },
        "mapping": {
            "phenotype_normalisation_version": PHENOTYPE_NORMALISATION_VERSION,
            "phenotype_namespace": "HP",
            "disease_namespace": "MONDO",
            "partial_phenotypes": partial_policy,
        },
        "counts": counts,
        "samples": {"sha256": samples_sha256, "n": len(mapped.samples)},
        "unmappable_terms": dict(mapped.unmappable_terms),
        "exclusions": list(mapped.exclusions),
    }


def preview_id(manifest: Mapping[str, Any]) -> str:
    """The first 16 hex digits of the SHA-256 of the manifest's canonical JSON
    (sorted keys, no whitespace), its ``imported`` block left out (§0.6).

    The commit recomputes the manifest from fresh reads and publishes only
    under the same id, so what is committed is what the operator previewed:
    the same source bytes, graph, dataset record, policy and outcome.
    """
    content = {key: value for key, value in manifest.items() if key != "imported"}
    canonical = json.dumps(content, ensure_ascii=True, sort_keys=True,
                           separators=(",", ":"), allow_nan=False)
    return _sha256(canonical.encode("ascii"))[:16]


# ---------------------------------------------------------------------------
# Preview and commit
# ---------------------------------------------------------------------------
def prepare_import(
    registry: Any,
    data_dir: Path,
    source_path: Path,
    label: str,
    role: str,
    partial_policy: str,
) -> ImportPlan:
    """Read, map, and build the version and the bytes to write; write nothing.

    The source is read once: its digest and its parse come from the same bytes,
    and those bytes are what a commit keeps. The workspace graph is read once
    through `read_workspace_graph`. The label is checked against the registry
    here as well as at commit, so a preview of a taken label says so.

    Raises:
        RegistryRefused, CohortVersionRefused: the registry, label, role or
            policy is refused.
        MalformedSource, UnsupportedDiseaseNamespace: from `parse_source`.
        ValueError: from the workspace readers and verifiers.
        OSError: a file could not be read.
    """
    from src.utils.fingerprint import read_once

    root = validate_registry(registry)
    label = validate_label(label)
    role = validate_role(role)
    partial_policy = validate_partial_policy(partial_policy)
    _require_label_free(root, label)

    source_path = Path(source_path)
    read = read_once(source_path)
    cases = parse_source(read.data, source=source_path)
    workspace = read_workspace_graph(Path(data_dir))
    mapped = map_cases(cases, workspace.node_mapping, partial_policy)

    samples_bytes = encode_json(mapped.samples)
    manifest = build_version_manifest(
        label=label, role=role, source_sha256=read.sha256,
        source_filename=source_path.name, workspace=workspace,
        partial_policy=partial_policy, mapped=mapped,
        samples_sha256=_sha256(samples_bytes),
    )
    return ImportPlan(label, role, manifest, preview_id(manifest), read.sha256, read.data,
                      samples_bytes)


def _utc(now: datetime, pattern: str) -> str:
    if now.tzinfo is None:
        raise ValueError(f"{now!r} has no time zone; a recorded time must say which it is in")
    return now.astimezone(timezone.utc).strftime(pattern)


def _keep_source(root: Path, plan: ImportPlan) -> Path:
    """Keep the source bytes as ``sources/<sha256>.json``, never overwriting,
    and return the resolved ``sources/`` directory.

    ``sources/`` is resolved before anything is read from or written into it
    (`_registry_directory`), then created when absent. A file already there
    must hash to its name -- the same source imported before -- and is then
    left as it is. Anything else means the registry was altered, and nothing is
    written over it.
    """
    from src.utils.file_modes import publish_bytes
    from src.utils.fingerprint import read_once

    sources = _registry_directory(root / SOURCES_DIR, within=root, done="Nothing was written.")
    sources.mkdir(exist_ok=True)
    target = sources / f"{plan.source_sha256}.json"
    try:
        existing = read_once(target)
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if existing.sha256 != plan.source_sha256:
            raise RegistryRefused(
                f"registry damaged: {target} does not hash to its name (its bytes are "
                f"{existing.sha256[:12]}...). A kept source is never overwritten; "
                "nothing was written. Investigate the registry before importing again."
            )
        return sources
    publish_bytes(target, plan.source_bytes)
    return sources


def commit_import(
    plan: ImportPlan,
    registry: Any,
    expected_preview_id: str,
    *,
    now: Optional[datetime] = None,
) -> CohortVersionRead:
    """Publish the version `plan` describes, if it is the one previewed (§0.6).

    Writes nothing until every check passes: the registry, the preview id
    (recomputed from `plan.manifest`, not taken from `plan`), the plan's label
    and bytes against what its manifest binds, a non-empty sample set, and a
    free label. Then, in order: the source is kept (no-clobber), the version
    directory is claimed with ``mkdir(exist_ok=False)``, ``samples.json`` is
    published, and ``cohort.json`` -- the commit point -- last, with the
    ``imported`` block (when, and `code_revision(REPO_ROOT)`, both taken before
    the first write). ``sources/``, ``versions/`` and, after the claim, the
    version directory are each resolved again before anything is written into
    them, and refused when a link takes one outside the registry or into the
    repository (`_registry_directory`); the refusal says what was already
    written. A failure after the claim leaves a directory without
    ``cohort.json``, which every reader refuses as incomplete.

    `now` is the time recorded as ``imported.at``; the current time when omitted.

    Returns the version as `resolve_version` reads it back.
    """
    from src.utils.file_modes import publish_bytes
    from src.utils.run_environment import code_revision

    try:
        root = validate_registry(registry)
        if not (isinstance(expected_preview_id, str)
                and _PREVIEW_ID.fullmatch(expected_preview_id)):
            raise CohortVersionRefused(
                f"{expected_preview_id!r} is not a preview id (16 lowercase hex digits, "
                "as the preview prints it)."
            )
        computed = preview_id(plan.manifest)
        if computed != expected_preview_id:
            raise CohortVersionRefused(
                f"the inputs changed since the preview: the files and arguments now give "
                f"preview id {computed}, not {expected_preview_id}. Run the preview again "
                "and check what changed before committing."
            )
        # The id covers the manifest, not the bytes beside it; those must be the
        # ones it binds, or the registry would keep a file under another's name.
        label = validate_label(plan.manifest["label"])
        source_sha256 = plan.manifest["source"]["sha256"]
        if (plan.label != label or plan.source_sha256 != source_sha256
                or _sha256(plan.source_bytes) != source_sha256
                or _sha256(plan.samples_bytes) != plan.manifest["samples"]["sha256"]):
            raise CohortVersionRefused(
                f"the plan for {label!r} carries a label, source or samples its manifest "
                "does not bind, so it is not one prepare_import made from one read of each "
                "input."
            )
        if not plan.manifest["samples"]["n"]:
            raise CohortVersionRefused(
                f"no case of the source is eligible for ranking against this graph, so the "
                f"version {label!r} would measure nobody. See the preview's exclusions."
            )
        _require_label_free(root, label)
    except (RegistryRefused, CohortVersionRefused) as exc:
        raise type(exc)(f"{exc} Nothing was written.") from None
    manifest = {
        **plan.manifest,
        "imported": {
            "at": _utc(now or datetime.now(timezone.utc), "%Y-%m-%dT%H:%M:%SZ"),
            "code_revision": code_revision(REPO_ROOT),
        },
    }
    manifest_bytes = encode_json(manifest)

    # Each directory is resolved again before it is written into: a link below
    # the root would otherwise carry the files wherever it points (D7).
    sources = _keep_source(root, plan)
    kept = (f"Only the source copy was written (under {sources}, named by its digest, which "
            "is harmless to keep).")

    versions = _registry_directory(root / VERSIONS_DIR, within=root, done=kept)
    versions.mkdir(exist_ok=True)
    directory = versions / label
    try:
        directory.mkdir(exist_ok=False)
    except FileExistsError:
        raise CohortVersionRefused(
            f"{directory} was created by another import while this one ran, so the label "
            f"{label!r} is taken. {kept}"
        ) from None
    directory = _registry_directory(
        directory, within=root,
        done=(f"Only the source copy (under {sources}) and the empty directory {directory}, "
              "which claimed the label, were written; nothing was written into it. Every "
              "reader refuses it as an interrupted import; remove it to reuse the label."),
    )

    publish_bytes(directory / VERSION_SAMPLES, plan.samples_bytes)
    publish_bytes(directory / VERSION_MANIFEST, manifest_bytes)
    return resolve_version(root, label)


# ---------------------------------------------------------------------------
# Reading a version back
# ---------------------------------------------------------------------------
def _is_count(value: Any) -> bool:
    """A whole number of at least 0, and not a boolean (`True` is an `int`)."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _counts_problems(manifest: Dict[str, Any], samples_n: Any) -> List[str]:
    """The counts a report and the ledger record as the version's denominators
    (§2.3), present and consistent with each other and with the version.

    A measurement copies ``counts`` into its report and the ledger as they
    are, and prints coverage from them; so the keys must be there, and the
    equalities the importer writes must still hold: ``ranked_eligible ==
    samples.n``; ``source_cases == ranked_eligible + n_excluded ==
    source.n_cases``; ``n_excluded == sum(excluded) == len(exclusions)``. A
    manifest edited or damaged after the import is refused here, before
    anything is measured or written, rather than recorded as coverage that
    does not describe the cohort.
    """
    counts = manifest["counts"]
    if not isinstance(counts, dict):
        return ["counts must be an object"]
    problems = [f"counts.{key} must be a whole number of at least 0"
                for key in ("source_cases", "ranked_eligible", "n_excluded")
                if not _is_count(counts.get(key))]
    excluded = counts.get("excluded")
    if not (isinstance(excluded, dict) and set(excluded) == set(EXCLUSION_CATEGORIES)
            and all(_is_count(n) for n in excluded.values())):
        problems.append(f"counts.excluded must hold a whole number of at least 0 for exactly "
                        f"{list(EXCLUSION_CATEGORIES)}")
    if problems:
        return problems

    source_cases, ranked = counts["source_cases"], counts["ranked_eligible"]
    n_excluded = counts["n_excluded"]
    if _is_count(samples_n) and ranked != samples_n:
        problems.append(f"counts.ranked_eligible ({ranked}) is not samples.n ({samples_n})")
    if source_cases != ranked + n_excluded:
        problems.append(f"counts.source_cases ({source_cases}) is not ranked_eligible plus "
                        f"n_excluded ({ranked} + {n_excluded})")
    source = manifest["source"]
    n_cases = source.get("n_cases") if isinstance(source, dict) else None
    if not _is_count(n_cases):
        problems.append("source must hold n_cases, a whole number of at least 0")
    elif source_cases != n_cases:
        problems.append(f"counts.source_cases ({source_cases}) is not source.n_cases "
                        f"({n_cases})")
    if n_excluded != sum(excluded.values()):
        problems.append(f"counts.n_excluded ({n_excluded}) is not the sum of counts.excluded "
                        f"({sum(excluded.values())})")
    exclusions = manifest["exclusions"]
    if isinstance(exclusions, list) and n_excluded != len(exclusions):
        problems.append(f"counts.n_excluded ({n_excluded}) is not the number of exclusions "
                        f"listed ({len(exclusions)})")
    return problems


def _version_shape_problems(manifest: Dict[str, Any]) -> List[str]:
    """What a reader relies on, beyond the field list: the four graph digests,
    the dataset record's digest, the samples' digest and count, the source's
    digest, counts that are present and consistent (`_counts_problems`), and a
    role that is still a role."""
    problems = [f"missing {name}" for name in REQUIRED_VERSION_FIELDS if name not in manifest]
    if problems:
        return problems

    def is_sha(value: Any) -> bool:
        return isinstance(value, str) and bool(_SHA256.fullmatch(value))

    graph = manifest["graph"]
    if not isinstance(graph, dict) or not all(is_sha(graph.get(r)) for r in GRAPH_ARTIFACTS):
        problems.append(f"graph must hold a SHA-256 for each of {list(GRAPH_ARTIFACTS)}")
    for block in ("dataset_record", "source"):
        if not (isinstance(manifest[block], dict) and is_sha(manifest[block].get("sha256"))):
            problems.append(f"{block} must hold a sha256")
    samples = manifest["samples"]
    n = samples.get("n") if isinstance(samples, dict) else None
    if not (isinstance(samples, dict) and is_sha(samples.get("sha256")) and _is_count(n)):
        problems.append("samples must hold a sha256 and a count n")
    for block in ("mapping", "unmappable_terms", "imported"):
        if not isinstance(manifest[block], dict):
            problems.append(f"{block} must be an object")
    if not isinstance(manifest["exclusions"], list):
        problems.append("exclusions must be a list")
    problems += _counts_problems(manifest, n)
    try:
        validate_role(manifest["role"])
    except CohortVersionRefused as exc:
        problems.append(str(exc))
    return problems


def resolve_version(registry: Any, label: str) -> CohortVersionRead:
    """Resolve a committed version: one read of its ``cohort.json``, parsed from
    those bytes, and the identity of those bytes.

    Does not read ``samples.json``: its path is returned, and the caller reads
    it once, comparing the read's digest with ``manifest["samples"]["sha256"]``.

    The version directory is resolved before anything is read from it, and the
    resolved path is the one returned, so a caller writing reports under it
    writes where this check looked (`_registry_directory`).

    Raises:
        RegistryRefused: the registry path is refused (`validate_registry`), or
            the version directory resolves outside it or inside the repository.
        CohortVersionRefused: the label is invalid; there is no such version;
            the directory has no ``cohort.json`` (an interrupted import); the
            manifest cannot be read or parsed, is not an object, is not schema
            `VERSION_SCHEMA_VERSION`, names another label than its directory,
            lacks a required field, or holds counts that are missing or
            contradict each other or the version (`_counts_problems`).
    """
    from src.utils.fingerprint import read_once

    root = validate_registry(registry)
    label = validate_label(label)
    directory = _registry_directory(root / VERSIONS_DIR / label, within=root,
                                    done="No version is read from there.")
    path = directory / VERSION_MANIFEST
    if not directory.is_dir():
        raise CohortVersionRefused(
            f"the registry {root} holds no cohort version {label!r} (no {directory})."
        )
    try:
        read = read_once(path)
    except FileNotFoundError:
        raise CohortVersionRefused(
            f"{directory} has no {VERSION_MANIFEST}: an interrupted import, not a version. "
            "Remove the directory and import again."
        ) from None
    except OSError as exc:
        raise CohortVersionRefused(
            f"{path} could not be read ({type(exc).__name__}: {exc})"
        ) from exc
    identity = read.identity
    try:
        text = read.data.decode("utf-8")
        del read
        manifest = json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise CohortVersionRefused(
            f"{path} is not UTF-8 JSON ({type(exc).__name__}: {exc}), so it describes no "
            "version this revision can read."
        ) from None
    if not isinstance(manifest, dict):
        raise CohortVersionRefused(f"{path} is not a JSON object.")
    schema = manifest.get("schema_version")
    if isinstance(schema, bool) or schema != VERSION_SCHEMA_VERSION:
        raise CohortVersionRefused(
            f"{path} is schema {schema!r}; this revision reads schema "
            f"{VERSION_SCHEMA_VERSION} only."
        )
    if manifest.get("label") != label:
        raise CohortVersionRefused(
            f"{path} names the label {manifest.get('label')!r}, not its directory's "
            f"{label!r}: the manifest was moved or edited, so it does not identify this "
            "version."
        )
    problems = _version_shape_problems(manifest)
    if problems:
        raise CohortVersionRefused(
            f"{path} is not a complete, consistent version manifest: {'; '.join(problems)}."
        )
    return CohortVersionRead(manifest, identity, directory, directory / VERSION_SAMPLES)


# ---------------------------------------------------------------------------
# Listing the registry (M5-1)
# ---------------------------------------------------------------------------
class ListedVersion(NamedTuple):
    """One entry of the registry's ``versions/``: its name on disk, the version
    as `resolve_version` reads it, or why it cannot be read (an interrupted
    import, a manifest this revision does not read, a link that leaves the
    registry)."""

    label: str
    version: Optional[CohortVersionRead]
    problem: Optional[str]


class ListedRun(NamedTuple):
    """One entry of a version's ``reports/``: its run id, its directory resolved
    inside ``reports/``, or why it is not a run directory. What the run kept and
    whether its report is registered is the registration's to say, not this."""

    run_id: str
    directory: Optional[Path]
    problem: Optional[str]


#: A run id as `create_report_dir` makes it: UTC time, checkpoint digest (12),
#: random part (8).
_RUN_ID = re.compile(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}-[0-9a-f]{8}")


def list_versions(registry: Any) -> List[ListedVersion]:
    """Every entry under the registry's ``versions/``, by name, each read by
    `resolve_version` (the one reader) and kept as a value when it refuses, so
    an interrupted import is listed with its reason rather than hidden.

    Reads each ``cohort.json`` once, and nothing else; never a samples file.

    Raises:
        RegistryRefused: the registry root is refused (`validate_registry`), or
            ``versions/`` resolves outside it or into the repository.
    """
    root = validate_registry(registry)
    versions = _registry_directory(root / VERSIONS_DIR, within=root,
                                   done="No version was read.")
    if not versions.is_dir():
        return []
    listed: List[ListedVersion] = []
    for entry in sorted(versions.iterdir(), key=lambda p: p.name):
        try:
            listed.append(ListedVersion(entry.name, resolve_version(root, entry.name), None))
        except (RegistryRefused, CohortVersionRefused) as exc:
            listed.append(ListedVersion(entry.name, None, str(exc)))
    return listed


def list_report_runs(version: CohortVersionRead) -> List[ListedRun]:
    """The run directories in a version's ``reports/``, newest run id first.

    ``reports/`` is resolved inside the version directory, and each run
    directory inside ``reports/``, as `create_report_dir` checks them before it
    writes: one a link takes elsewhere, or an entry that is not a directory
    named as a run id, is listed with its reason. Reads no file.

    Raises:
        RegistryRefused: ``reports/`` resolves outside the version directory, or
            the version directory into the repository.
    """
    directory = _registry_directory(version.directory, within=None, done="Nothing was read.")
    reports = _registry_directory(directory / REPORTS_DIR, within=directory,
                                  done="Nothing was read.")
    if not reports.is_dir():
        return []
    runs: List[ListedRun] = []
    for entry in sorted(reports.iterdir(), key=lambda p: p.name, reverse=True):
        if not _RUN_ID.fullmatch(entry.name):
            runs.append(ListedRun(entry.name, None, (
                f"{entry.name!r} in {reports} is not named as a run id, so it is not a "
                "run directory")))
            continue
        try:
            resolved = _registry_directory(entry, within=reports, done="Nothing was read.")
        except RegistryRefused as exc:
            runs.append(ListedRun(entry.name, None, str(exc)))
            continue
        if not resolved.is_dir():
            runs.append(ListedRun(entry.name, None, f"{entry} is not a directory"))
            continue
        runs.append(ListedRun(entry.name, resolved, None))
    return runs


def verify_cohort_binding(
    version_manifest: Mapping[str, Any],
    bound: Mapping[str, str],
    dataset_record_sha256: str,
) -> None:
    """Refuse a version mapped against another graph or dataset record (§0.7).

    **The same comparison as R3.** The version's four recorded graph digests
    must equal the run's `bound`, and its dataset record digest the run's
    verified record. A version's indices mean nodes of the graph it was mapped
    against; scored against another, an in-range index names a different node
    and nothing downstream can tell. `split_manifest` is not compared: it
    records a sampling event, and two workspaces can share every graph digest.
    Reads nothing.

    Raises:
        CohortBindingMismatch: naming every role that differs (`roles`).
    """
    graph = version_manifest.get("graph")
    graph = graph if isinstance(graph, Mapping) else {}
    record = version_manifest.get("dataset_record")
    record = record if isinstance(record, Mapping) else {}
    label = version_manifest.get("label")

    roles: List[str] = []
    details: List[str] = []
    for role in GRAPH_ARTIFACTS:
        recorded, held = graph.get(role), bound.get(role)
        if not recorded or recorded != held:
            roles.append(role)
            details.append(f"{role} (version {str(recorded)[:12]}..., run {str(held)[:12]}...)")
    recorded = record.get("sha256")
    if not recorded or recorded != dataset_record_sha256:
        roles.append("dataset_record")
        details.append(f"dataset_record (version {str(recorded)[:12]}..., run "
                       f"{str(dataset_record_sha256)[:12]}...)")
    if roles:
        raise CohortBindingMismatch(
            f"the cohort version {label!r} was mapped against another graph than this "
            f"run's: {'; '.join(details)}. Its indices name nodes of that graph, so "
            "scored here they would name different nodes. Import the source again "
            "against this workspace under a new label.",
            roles,
        )


def create_report_dir(
    version: CohortVersionRead, checkpoint_sha256: str, now: datetime
) -> Tuple[str, Path]:
    """A new directory for one run's report under the version (§2.1).

    The run id is ``<UTC time>-<checkpoint digest, 12>-<8 random hex>``: sortable
    by time, readable as which weights, and unique without a lock. ``reports/``
    is created when absent; the run directory is created with
    ``exist_ok=False``, so two runs never share one.

    **Every directory is resolved before it is written into**
    (`_registry_directory`): the version directory must not resolve inside the
    repository, ``reports/`` must resolve inside the version directory, and the
    run directory inside ``reports/``. `resolve_version` checked the version
    directory against its registry, so the report lands inside the registry
    and never in the repository, whatever links are below it. The resolved run
    directory is returned, and the caller writes through it.
    """
    if not (isinstance(checkpoint_sha256, str) and _SHA256.fullmatch(checkpoint_sha256)):
        raise ValueError(f"{checkpoint_sha256!r} is not a SHA-256 hexdigest")
    run_id = (f"{_utc(now, '%Y%m%dT%H%M%SZ')}-{checkpoint_sha256[:12]}-"
              f"{secrets.token_hex(4)}")
    # The resolved version directory is what reports/ must stay inside; one in
    # the repository is refused here, naming it rather than its reports/.
    directory = _registry_directory(version.directory, within=None,
                                    done="Nothing was written.")
    reports = _registry_directory(directory / REPORTS_DIR, within=directory,
                                  done="Nothing was written.")
    reports.mkdir(parents=True, exist_ok=True)
    run_dir = reports / run_id
    run_dir.mkdir(exist_ok=False)
    run_dir = _registry_directory(
        run_dir, within=reports,
        done=f"Only the empty run directory {run_dir} was created; nothing was written into it.",
    )
    return run_id, run_dir


__all__ = [
    "CohortBindingMismatch",
    "CohortVersionRead",
    "CohortVersionRefused",
    "DEFAULT_PARTIAL_PHENOTYPES",
    "EXCLUSION_CATEGORIES",
    "ImportPlan",
    "ListedRun",
    "ListedVersion",
    "MalformedSource",
    "MappedCohort",
    "PARTIAL_PHENOTYPE_POLICIES",
    "REPO_ROOT",
    "REPORTS_DIR",
    "REQUIRED_VERSION_FIELDS",
    "RegistryRefused",
    "SOURCES_DIR",
    "SOURCE_FORMAT",
    "SourceCase",
    "UNSUPPORTED_DISEASE_NAMESPACES",
    "UnsupportedDiseaseNamespace",
    "VERSIONS_DIR",
    "VERSION_MANIFEST",
    "VERSION_SAMPLES",
    "VERSION_SCHEMA_VERSION",
    "WHAT_THIS_IS",
    "WorkspaceGraph",
    "build_version_manifest",
    "commit_import",
    "create_report_dir",
    "encode_json",
    "list_report_runs",
    "list_versions",
    "map_cases",
    "parse_source",
    "prepare_import",
    "preview_id",
    "read_workspace_graph",
    "resolve_version",
    "validate_label",
    "validate_partial_policy",
    "validate_registry",
    "validate_role",
    "verify_cohort_binding",
]
