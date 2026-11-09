"""
Data versions: a source case set mapped to one graph, or a KG edge set, kept outside the repository.
====================================================================================================
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

**The report store** is the version's ``reports/`` directory, the one part of a
version that grows: a run directory per measurement (`create_report_dir`,
named by `RUN_ID_PATTERN`), and beside them two append-only files, the usage
history and the investigation notes (`src/evaluation/report_store.py`, M5's D4
and D5). Every reader takes a run directory by its name, so neither file is
read as a run.

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

**Genes, optional per case** (README §8: reference cohorts carry causal genes;
PLAN_TEST_RESULTS §8.3–8.4). A case may list ``causal_genes`` and
``candidate_genes``, each a non-empty list of distinct gene identifiers: a gene
node's id (``disgenet:SCN1A``) or a symbol only one gene node carries
(`KnowledgeGraph.gene_lookup`). Mapped, they are the sample's
``causal_gene_ids`` and ``gene_ids``; an identifier that maps to no gene is
left out and counted, and the case is not excluded for it (a task binds what
it needs). The ``genes`` counts appear only when the source lists genes.

**Kinds and schemas** (research comparison §5 Stage 4, owner's O7). A version
is of one data kind (`VERSION_KINDS`, the words of
`src.evaluation.cohort.COHORT_KINDS`), read and checked by the one reader,
`resolve_version`, in either schema it reads (`VERSION_SCHEMAS`):

- **schema 1** records no kind and is a ``supplied`` cohort, as every version
  before schema 2 is; a supplied import still writes schema 1, byte for byte
  as before (golden P0-4);
- **schema 2** states its ``kind``, and is read only for a kind written in
  it (`KIND_SCHEMAS`: a KG edge set, a generated split); a schema-2 manifest
  of kind ``supplied`` is a shape no revision writes, and is refused rather
  than read by the schema-1 rules. A revision that reads schema 1 only
  refuses a schema-2 version by its schema check ("is schema 2; this revision
  reads schema 1 only"), so it never reads one as a supplied cohort, and a
  revision that reads schema 2 but not a kind refuses that kind by name
  ("this revision reads kinds ... only").

A **KG edge set** (``kg_edges``) is one part of a recorded edge split of one
graph view, kept as fixed link-prediction pairs. `prepare_kg_edges_import`
takes the view's rule and the split from a named checkpoint's model record
(``data_roles``), rebuilds both from the workspace's graph and checks them
(`src.kg.graph_view.rebuild_split`), then builds the pairs of the chosen part
by base validation's rule (`src.models.bundle.lp_head.fixed_lp_pairs`, the
split's seed, `BASE_FIXED`'s batch size). Its role is the part, recorded as
given: ``train``, ``val`` or ``test``, none refused for not being test. Its
manifest binds the four graph digests and the dataset record (compared by
`verify_cohort_binding`, as a supplied version's are), the view's rule and
digest, the split's rule, seed, digest and part counts, the message-passing
graph (the split's train part, README A4; for role ``train`` the measured
edges are in it, and the manifest says so), the pair rule with its seed, its
batch size and the pairs each batch kept (the batch boundaries in the stored
rows, and each batch's edges left out), and ``pairs.bin``: the pairs as
compact little-endian int64 rows (head, relation, tail, negative tail) in the
order made, bound by the digest of its bytes and read only through
`read_kg_edge_pairs`. torch's generator
stream is not promised across releases, so the pairs are kept, never rebuilt.
Its counts carry their units (``count_units``). The checkpoint it was taken
from is named by its digest only, never by its file name, so a copy or a link
of the same checkpoint makes the same version. Nothing about it holds a case.

A **generated split** (``generated``; research comparison §3 (A), a limited
data identity) is one of a workspace's generated patient splits, the
``<role>_samples.json`` that `src/kg/sample_generator.py` wrote and the split
manifest binds, which ``measure_scorer --split`` reads in place.
`prepare_generated_import` reads and verifies the workspace as that
measurement does (the graph, the dataset record, and both generated splits
against the one split manifest reading: their digests, disease sets and
disjointness, `verify_cohort_reads`), then keeps the role's samples. Its data
identity is the samples' content digest, the workspace's split artefact
digests (the split manifest and every samples file it binds), the generation
definition (the generator's algorithm, version and recorded settings, and the
allocation it cut), and its role, recorded as given: ``train``, ``val`` or
``test`` (`GENERATED_ROLES`). It is listed and named as a generated split of
its role, never as a supplied cohort, and as test only when its role is test;
no role is refused. The generator writes, and `verify_cohort_reads` checks,
``train`` and ``val`` only, so a ``test`` import is refused for what the
workspace lacks (a generated test split this revision can verify), never for
the role. **The samples are
copied, not bound in place**: a workspace is rebuilt or re-sampled in its own
directory, and a version that only named the workspace's files by digest
would stop being readable, and verifiable as data, the day that happened. The
copy is the bytes the workspace held, bound by the digest its split manifest
records, so a registered generated version is as immutable and verifiable as
a supplied one, and the workspace's split is still named by content: whether
the workspace still holds it is a reference's state (`check_references`), not
a condition of reading the version. Archived reports measured with
``--split`` are untouched: no identity is made up for them, and registering
them stays refused (`src.evaluation.sidecar.build_record`). Measuring a
registered generated split and registering that result are the research
comparison's Stages 5 and 6.

**Content references** (E3). Every version records what it depends on by
content identity, never by path: `version_references` names each reference
of a version by its kind's own fields (the four graph digests: ``kg.json``,
which carries the node naming, node types, species, namespaces and external
ids, and the tensors; the dataset record; each source the record names, by
role; and per kind the view, the split, the pairs and the checkpoint the view
and split were taken from, by its SHA-256; the split manifest, the split's
files and the samples; or the case source and the samples). A path is only a
location: `check_references` compares the bytes found at a location the
caller gives (a version's own directory, a workspace, a checkpoint file) with
the recorded digests, and reports each reference's state (verified, differs,
missing, unreadable, or not checked); a missing dependency is a state, never
a crash and never a pass. Which roots may be read is decided apart from this:
a registry root by `validate_registry` (and the service's
``SHEPHERD_REGISTRY``), a workspace by its caller. A registry copied to
another authorised root resolves to the same versions, by the same digests.

**Source roles** (E4). A KG edge set's and a generated split's
``dataset_record.sources`` list each named source as its record names it:
role, digest and declared version, with no list of allowed roles (any role
that is an identifier, each once, each with a SHA-256). Species, namespace and
external-id distinctions are kept by reference to the graph (``kg.json``) and,
for a KG edge set, the view, never derived from symbols. Today's KG builder
names four roles (`src.kg.provenance.SOURCE_ROLES`); a real new source needs
the builder's own provenance extension first, which this module does not
make.

Module: src/evaluation/cohort_registry.py
"""
from __future__ import annotations

import copy
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

#: The schema a supplied import writes: schema 1, which records no kind, so a
#: supplied version's bytes are what they always were (golden P0-4).
VERSION_SCHEMA_VERSION = 1
#: The schema that states its kind, which a KG edge set is written in.
KINDED_SCHEMA_VERSION = 2
#: The schemas `resolve_version` reads, each bumped when the manifest's shape
#: changes in a way an old reader cannot read. A reader refuses any other value
#: rather than guessing.
VERSION_SCHEMAS: Tuple[int, ...] = (VERSION_SCHEMA_VERSION, KINDED_SCHEMA_VERSION)

#: The kinds a registered version is, one name each, the words of
#: `src.evaluation.cohort.COHORT_KINDS` (the measurement manifest's
#: ``cohort_kind``): a supplied cohort, a KG edge set, and a generated split
#: kept from a workspace.
SUPPLIED = "supplied"
KG_EDGES = "kg_edges"
GENERATED = "generated"
VERSION_KINDS: Tuple[str, ...] = (SUPPLIED, KG_EDGES, GENERATED)
#: The schema each kind is written in. A manifest stating a kind in another
#: schema is a shape no revision writes, so it is refused, never read by the
#: rules of the shape it resembles (research comparison §2: nothing guessed).
KIND_SCHEMAS: Dict[str, int] = {SUPPLIED: VERSION_SCHEMA_VERSION, KG_EDGES: KINDED_SCHEMA_VERSION,
                                GENERATED: KINDED_SCHEMA_VERSION}

#: A KG edge set's roles: the parts of an edge split
#: (`src.kg.graph_view.SPLIT_NAMES`, pinned equal by
#: tests/unit/test_kg_edge_versions.py; this module imports no torch).
KG_EDGES_ROLES: Tuple[str, ...] = ("train", "val", "test")

#: A generated split's roles, recorded as given (owner's O7): the split it is,
#: whose samples are the workspace's ``<role>_samples.json``. The generator
#: writes ``train`` and ``val`` (`src.evaluation.cohort.GENERATED_SPLITS`), the
#: splits `verify_cohort_reads` checks; a ``test`` version needs a workspace
#: holding a generated test split this revision verifies, which none does
#: today, so its import is refused for that absence, never for the role.
GENERATED_ROLES: Tuple[str, ...] = ("train", "val", "test")

VERSION_MANIFEST = "cohort.json"
VERSION_SAMPLES = "samples.json"
#: A KG edge set's pairs (`lp_pairs_table`'s bytes), bound by its manifest.
VERSION_PAIRS = "pairs.bin"
SOURCES_DIR = "sources"
VERSIONS_DIR = "versions"
REPORTS_DIR = "reports"

#: A run's two files in the report store, in the order the measurement publishes
#: them: the per-case ranks, then the report that names them by digest (the
#: run's commit point; a run directory without it is an interrupted write).
RANKS_FILE = "ranks.json"
REPORT_FILE = "report.json"

#: The name `create_report_dir` gives a run directory: ``<UTC time>-<checkpoint
#: digest, 12 hex>-<8 random hex>``. Every reader of the report store takes a
#: run directory by this name and nothing else, so a file kept beside the run
#: directories (the usage history, the investigation notes and their lock files)
#: is never read as one.
RUN_ID_PATTERN = re.compile(r"[0-9]{8}T[0-9]{6}Z-([0-9a-f]{12})-[0-9a-f]{8}")

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
_GENE_KEYS = ("causal_genes", "candidate_genes")
# ASCII digits, not `\d`, which also matches digits of other scripts.
_HPO_TERM = re.compile(r"HP:[0-9]{7}")
_MONDO_TERM = re.compile(r"MONDO:[0-9]{7}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PREVIEW_ID = re.compile(r"[0-9a-f]{16}")

#: Every field `commit_import` writes for a supplied version. A manifest
#: missing one is not a version this revision reads.
REQUIRED_VERSION_FIELDS: Tuple[str, ...] = (
    "schema_version", "what_this_is", "label", "role", "source", "graph",
    "dataset_record", "mapping", "counts", "samples", "unmappable_terms",
    "exclusions", "imported",
)

#: Every field `commit_import` writes for a KG edge set (`build_kg_edges_manifest`).
KG_EDGES_REQUIRED_FIELDS: Tuple[str, ...] = (
    "schema_version", "kind", "what_this_is", "label", "role", "graph", "dataset_record",
    "graph_view", "split", "message_passing", "pairs", "counts", "count_units",
    "taken_from", "imported",
)

KG_EDGES_WHAT_THIS_IS = ("a KG edge set: one part of a recorded edge split of one graph view, "
                         "kept as fixed link-prediction pairs")

#: A KG edge set's counts, each with its unit. An edge is a directed edge of
#: the graph view: an edge and its reverse twin are one fact and two edges, and
#: always in the same part.
KG_EDGES_COUNT_UNITS: Dict[str, str] = {
    "positive_edges": "directed graph-view edges of the role's part, each a positive",
    "pairs": "(positive, negative) pairs: a positive edge with its one negative tail",
    "edges_without_negative": ("positive edges left out: no other tail of their relation in "
                               "their batch to draw a negative from"),
    "message_passing_edges": "directed graph-view edges of the split's train part",
}

#: The model-record fields a KG edge set's view and split are taken from.
KG_EDGES_TAKEN_FROM = ("data_roles.graph_view", "data_roles.split")

#: What `cohort_version_block` gives for a KG edge set, exactly.
KG_EDGES_BLOCK_FIELDS: Tuple[str, ...] = (
    "label", "role", "kind", "manifest_sha256", "graph", "dataset_record_sha256",
    "graph_view", "split", "message_passing", "pairs", "counts", "count_units",
)

#: Every field `commit_import` writes for a generated split
#: (`build_generated_manifest`).
GENERATED_REQUIRED_FIELDS: Tuple[str, ...] = (
    "schema_version", "kind", "what_this_is", "label", "role", "graph", "dataset_record",
    "split_manifest", "split_artifacts", "generation", "allocation", "realised", "verified",
    "samples", "counts", "count_units", "imported",
)

GENERATED_WHAT_THIS_IS = ("a generated split: one of a workspace's generated patient splits, "
                          "its samples kept as the generator wrote them")

#: A generated split's counts, each with its unit.
GENERATED_COUNT_UNITS: Dict[str, str] = {
    "samples": "generated patients: the sample rows of the role's split, as written",
    "diseases": "distinct diseases among those patients (the split's realised disease set)",
}

#: What `cohort_version_block` gives for a generated split, exactly:
#: aggregates and digests, never a sample.
GENERATED_BLOCK_FIELDS: Tuple[str, ...] = (
    "label", "role", "kind", "manifest_sha256", "graph", "dataset_record_sha256",
    "samples_sha256", "split_manifest", "split_artifacts", "generation", "allocation",
    "counts", "count_units",
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


class CohortVersionChanged(ValueError):
    """The version is not the one a caller chose: `resolve_version` was given
    the SHA-256 of its ``cohort.json`` (``expected_sha256``) and its one read
    has other bytes (``found``), or finds no version there (``found`` is
    None); raised before the bytes are decoded. A ``ValueError``, like the
    reader's other refusals, and not a `CohortVersionRefused`: it says only
    that the bytes are not the ones chosen, never what is wrong with them.

    It carries the facts (``path``, the ``cohort.json`` read or looked for;
    ``expected``; ``found``) and its message states only them. What the
    difference means depends on where the expectation came from (a run started
    for a version, a version listed earlier), so each caller words its own
    refusal from these fields."""

    def __init__(self, path: str | Path, expected: str, found: Optional[str]) -> None:
        self.path = Path(path)
        self.expected = expected
        self.found = found
        held = (f"holds {found[:12]}..., not the {expected[:12]}... expected"
                if found is not None else f"is gone; {expected[:12]}... was expected")
        super().__init__(f"{self.path} {held}.")


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
    #: Gene identifiers, None when the source does not list them.
    causal_genes: Optional[Tuple[str, ...]] = None
    candidate_genes: Optional[Tuple[str, ...]] = None


class WorkspaceGraph(NamedTuple):
    """The graph a workspace manifest binds, read and verified once each.

    `bound` is the four graph digests the manifest binds (the same dict R3
    compares a checkpoint with); `node_mapping` is the parsed `kg.json`'s
    per-type node index, the graph object itself is not kept."""

    bound: Dict[str, str]
    dataset_record: "DatasetRecord"
    node_mapping: Dict[str, Dict[str, int]]
    #: Gene identifiers -> gene index (`KnowledgeGraph.gene_lookup`).
    gene_lookup: Optional[Dict[str, int]] = None


class MappedCohort(NamedTuple):
    """What mapping made of the source: the samples to rank and everything left out."""

    samples: List[Dict[str, Any]]
    exclusions: List[Dict[str, Any]]
    counts: Dict[str, Any]
    unmappable_terms: Dict[str, int]


class ImportPlan(NamedTuple):
    """Everything a commit writes, computed from one read of each input.

    `manifest` is the version manifest without its ``imported`` block, which is
    what `preview_id` covers. For a supplied version, `source_bytes` are the
    bytes parsed, kept so the copy in the registry is the file that was
    mapped, not a second read of it, and `samples_bytes` the mapped samples.
    A KG edge set keeps no source file (its graph is the workspace's, bound by
    digest) and no samples: `pairs_bytes` are its pairs (`VERSION_PAIRS`). A
    generated split keeps no source file either: `samples_bytes` are the
    workspace's samples file, the bytes its split manifest binds.
    """

    label: str
    role: str
    manifest: Dict[str, Any]
    preview_id: str
    source_sha256: Optional[str]
    source_bytes: Optional[bytes]
    samples_bytes: Optional[bytes]
    pairs_bytes: Optional[bytes] = None

    @property
    def kind(self) -> Optional[str]:
        """The kind of the version this plan writes (`version_kind`)."""
        return version_kind(self.manifest)


class CohortVersionRead(NamedTuple):
    """A committed version as resolved: its manifest, the identity of the bytes
    that manifest was parsed from, its directory, and where its samples are
    (a supplied cohort's and a generated split's ``samples.json``; None for a
    kind that keeps none: a KG edge set's pairs are read by
    `read_kg_edge_pairs`)."""

    manifest: Dict[str, Any]
    identity: ReadIdentity
    directory: Path
    samples_path: Optional[Path]

    @property
    def kind(self) -> str:
        """The version's data kind, in the words a measurement manifest's
        `cohort_kind` uses (`src.evaluation.cohort.COHORT_KINDS`), which a
        new measurement is judged by (`src.evaluation.protocols`): a schema-1
        version records no kind and is a supplied cohort; a schema-2 version
        states its own (`version_kind`)."""
        return version_kind(self.manifest)


class KgEdgePairsRead(NamedTuple):
    """A KG edge set's pairs as `read_kg_edge_pairs` parsed them from one read
    of ``pairs.bin`` (`src.models.bundle.lp_head.LPPairs`, with the version's
    counts and the pairs' digest), and the identity of those bytes."""

    pairs: Any
    identity: ReadIdentity


def version_kind(manifest: Mapping[str, Any]) -> Optional[str]:
    """The kind of a version manifest: ``supplied`` for schema 1, which records
    none, and the stated ``kind`` for schema 2 (None when it states none, which
    `resolve_version` refuses). The one rule the reader, a plan and every
    listing read a kind by."""
    if manifest.get("schema_version") == VERSION_SCHEMA_VERSION:
        return SUPPLIED
    kind = manifest.get("kind")
    return kind if isinstance(kind, str) else None


def not_a_supplied_cohort(kind: Any, role: Any) -> Optional[str]:
    """Why a version of `kind` (its role `role`) is not a supplied cohort
    version, or None when it is one: ``a KG edge set (test), not a supplied
    cohort version``.

    The one sentence a reader of a supplied cohort's cases gives a version of
    another kind, which it lists or is named but cannot read as cases: the
    Diagnosis route and its card (a reference cohort's patients) and the
    overlap audit (`scripts/audit_split_overlap.py`, a cohort's diseases).
    Each caller says what it needed the cases for. A KG edge set holds edges,
    not cases, and is named by its role, never as a cohort (research
    comparison §5 Stage 4)."""
    if kind == SUPPLIED:
        return None
    return f"{version_words(kind, role)}, not a supplied cohort version"


def version_words(kind: Any, role: Any) -> str:
    """A version of another kind than a supplied cohort as a sentence names
    it, by its kind and role: ``a KG edge set (test)``, ``a generated split
    (val)`` (`not_a_supplied_cohort`'s words). Never "cohort": a KG edge set
    holds edges, and a generated split is the workspace generator's, never a
    supplied cohort (research comparison §5 Stage 4)."""
    if kind == KG_EDGES:
        return f"a KG edge set ({role})"
    if kind == GENERATED:
        return f"a generated split ({role})"
    return f"a version of kind {kind!r}"


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
        raise CohortVersionRefused(f"the version label is refused: {exc}") from None


def validate_role(role: Any) -> str:
    """A supplied cohort's role: the identifier rule, and never a generator
    split name.

    `train` and `val` name the generator's cohorts; a test cohort carrying one
    would read as a generated cohort in every artifact downstream, the hazard
    `src/evaluation/cohort.py` refuses for a supplied split. A KG edge set's
    role is another rule (`validate_kg_edges_role`).
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


def validate_kg_edges_role(role: Any) -> str:
    """A KG edge set's role: the part of the edge split it holds, recorded as
    given (owner's O7). ``train`` and ``val`` are roles like ``test``: measuring
    the train part is a fitting diagnostic and the val part a development
    measurement, and neither is ever relabelled as test."""
    if not (isinstance(role, str) and role in KG_EDGES_ROLES):
        raise CohortVersionRefused(
            f"a KG edge set's role is the part of the edge split it holds, one of "
            f"{', '.join(KG_EDGES_ROLES)}; got {role!r}"
        )
    return role


def validate_generated_role(role: Any) -> str:
    """A generated split's role: the split it is, recorded as given (owner's
    O7), one of `GENERATED_ROLES`. ``train`` is a fitting diagnostic, ``val``
    the checkpoint-selection split, ``test`` a held-out split where a workspace
    has one; none is refused for what it is, and none is ever relabelled."""
    if not (isinstance(role, str) and role in GENERATED_ROLES):
        raise CohortVersionRefused(
            f"a generated split's role is the split it is, one of "
            f"{', '.join(GENERATED_ROLES)}; got {role!r}"
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
            f"the version {label!r} already exists ({directory}). Versions are "
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
        extra = sorted(set(case) - _CASE_KEYS - set(_GENE_KEYS))
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

        genes: Dict[str, Optional[Tuple[str, ...]]] = {}
        genes_ok = True
        for key in _GENE_KEYS:
            if key not in case:
                genes[key] = None
                continue
            value = case[key]
            if not (isinstance(value, list) and value
                    and all(isinstance(g, str) and g.strip() and _printable_text(g)
                            for g in value)):
                problems.append(f"{where}: {key} must be a non-empty list of gene identifiers "
                                f"(non-empty printable strings), got {_json_kind(value)}")
                genes_ok = False
            elif len(set(value)) != len(value):
                problems.append(f"{where}: {key} lists a gene identifier twice")
                genes_ok = False
            else:
                genes[key] = tuple(value)

        if phenotypes_ok and disease_ok and case_id_ok and genes_ok:
            result.append(SourceCase(case_id, tuple(phenotypes), disease,
                                     genes["causal_genes"], genes["candidate_genes"]))

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
    return WorkspaceGraph(dict(bound), record, graph.kg.get_node_id_mapping(),
                          graph.kg.gene_lookup())


def map_cases(
    cases: Sequence[SourceCase],
    node_mapping: Mapping[str, Mapping[str, int]],
    partial_policy: str,
    gene_lookup: Optional[Mapping[str, int]] = None,
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
    is always excluded: a case with no phenotype cannot be scored.
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
    genes_listed = any(case.causal_genes or case.candidate_genes for case in cases)
    if genes_listed and gene_lookup is None:
        raise ValueError("the source lists genes and no gene lookup was given to map them")
    gene_counts = {"cases_with_causal_genes": 0, "cases_with_candidate_genes": 0,
                   "identifiers_listed": 0, "identifiers_unmappable": 0,
                   "identifiers_repeated": 0, "cases_with_unmapped_causal_genes": 0}

    def map_genes(identifiers: Sequence[str]) -> Tuple[List[int], List[str]]:
        """The genes the identifiers name, each once, and the identifiers that
        name no single gene, normalised by the lookup's case rule (stripped,
        upper-case), each once, in source order."""
        mapped_genes: List[int] = []
        unmapped: List[str] = []
        for identifier in identifiers:
            index = gene_lookup.get(identifier, gene_lookup.get(identifier.strip().upper()))
            gene_counts["identifiers_listed"] += 1
            if index is None:
                gene_counts["identifiers_unmappable"] += 1
                if identifier.strip().upper() not in unmapped:
                    unmapped.append(identifier.strip().upper())
            elif index in mapped_genes:
                gene_counts["identifiers_repeated"] += 1
            else:
                mapped_genes.append(index)
        return mapped_genes, unmapped

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
        sample = {
            "patient_id": case.case_id,
            "phenotype_ids": list(normalised.kept),
            "disease_id": disease_id,
        }
        if case.causal_genes:
            causal, unmapped_causal = map_genes(case.causal_genes)
            sample["causal_gene_ids"] = causal
            gene_counts["cases_with_causal_genes"] += 1
            if unmapped_causal:
                # The declared truth is the graph's genes and these: a gene the
                # graph lacks stays a positive the case could not rank (E5,
                # F9), and its identifier says which patients share it; never
                # "no causal gene".
                sample["causal_genes_unmapped"] = unmapped_causal
                gene_counts["cases_with_unmapped_causal_genes"] += 1
        if case.candidate_genes:
            sample["gene_ids"] = map_genes(case.candidate_genes)[0]
            gene_counts["cases_with_candidate_genes"] += 1
        samples.append(sample)

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
    if genes_listed:
        counts["genes"] = gene_counts
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
    mapped = map_cases(cases, workspace.node_mapping, partial_policy, workspace.gene_lookup)

    samples_bytes = encode_json(mapped.samples)
    manifest = build_version_manifest(
        label=label, role=role, source_sha256=read.sha256,
        source_filename=source_path.name, workspace=workspace,
        partial_policy=partial_policy, mapped=mapped,
        samples_sha256=_sha256(samples_bytes),
    )
    return ImportPlan(label, role, manifest, preview_id(manifest), read.sha256, read.data,
                      samples_bytes)


# ---------------------------------------------------------------------------
# A KG edge set (research comparison §5 Stage 4)
# ---------------------------------------------------------------------------
def _dataset_record_block(dataset_record: "DatasetRecord") -> Dict[str, Any]:
    """The dataset record as a KG edge set and a generated split record it:
    its digest, its origin, and every source it names, by role, as it names
    them (role, digest, declared version), with no list of allowed roles
    (E4). Species, namespaces and external ids are not derived here: they are
    the graph's, referenced by its digest."""
    return {
        "sha256": dataset_record.identity.sha256,
        "origin": dataset_record.origin,
        "sources": [{"role": entry["role"], "digest": entry["digest"],
                     "declared_version": entry.get("declared_version")}
                    for entry in dataset_record.record["sources"]],
    }


def build_kg_edges_manifest(
    *,
    label: str,
    role: str,
    bound: Mapping[str, str],
    dataset_record: "DatasetRecord",
    checkpoint_sha256: str,
    view: Any,
    split: Any,
    pairs: Any,
    pairs_sha256: str,
    batch_size: int,
) -> Dict[str, Any]:
    """A KG edge set's manifest (schema 2) without its ``imported`` block.

    Built only from what was read and computed: the four digests the workspace
    manifest binds and the dataset record the run verified (its digest, origin
    and every source it names, by role, with no list of allowed roles); the
    graph view (`GraphView`: its rule and digest) and the edge split
    (`EdgeSplit.record`: rule, seed, view digest, digest, edges per part) the
    checkpoint's record named and this workspace rebuilt; the message-passing
    graph, the split's train part; the pairs (`LPPairs`) of the role's part,
    with the rule, seed and batch size that made them, the pairs each batch
    kept, the digest of the file's bytes and the LP core's pairs digest; the
    counts with their units; and the checkpoint the view and split were taken
    from, by its digest (a dependency reference: its file name is a location
    and is not recorded). Aggregates only: no edge is listed here.
    """
    from src.models.bundle.lp_head import LP_PAIR_RULE, LP_PAIRS_LAYOUT

    record = split.record()
    edges = record["edges"]
    positive = int(pairs.counts["edges"])
    return {
        "schema_version": KINDED_SCHEMA_VERSION,
        "kind": KG_EDGES,
        "what_this_is": KG_EDGES_WHAT_THIS_IS,
        "label": label,
        "role": role,
        "graph": {role_name: bound[role_name] for role_name in GRAPH_ARTIFACTS},
        "dataset_record": _dataset_record_block(dataset_record),
        "graph_view": {"rule": copy.deepcopy(view.rule), "digest": view.digest},
        "split": record,
        # README A4: base validation and test message passing are the training
        # edges only. For role train the measured edges are among them, and
        # the version says so rather than removing anything.
        "message_passing": {"part": "train", "edges": edges["train"],
                            "includes_measured_edges": role == "train"},
        "pairs": {
            "file": VERSION_PAIRS,
            "layout": copy.deepcopy(LP_PAIRS_LAYOUT),
            "rows": int(pairs.counts["pairs"]),
            "sha256": pairs_sha256,
            "lp_pairs_digest": pairs.digest,
            "rule": LP_PAIR_RULE,
            "part": role,
            "seed": int(split.seed),
            "batch_size": int(batch_size),
            "batches": -(-positive // int(batch_size)),
            # The batch boundaries in the stored rows: batch k's pairs are the
            # batch_pairs[k] rows after the earlier batches'; its edges less
            # that count are the ones it left out (plan review §5 item 7).
            "batch_pairs": [int(n) for n in pairs.batch_pairs],
        },
        "counts": {
            "positive_edges": positive,
            "pairs": int(pairs.counts["pairs"]),
            "edges_without_negative": int(pairs.counts["without_negative"]),
            "message_passing_edges": edges["train"],
        },
        "count_units": dict(KG_EDGES_COUNT_UNITS),
        "taken_from": {"checkpoint_sha256": checkpoint_sha256,
                       "record_fields": list(KG_EDGES_TAKEN_FROM)},
    }


def _read_checkpoint_record(checkpoint: Path) -> Tuple[Any, Any]:
    """The checkpoint at `checkpoint`, one read at its resolved path, and its
    model record (`read_model_record`), or a refusal naming the file.

    A file torch cannot parse as a weights-only checkpoint is named, with
    torch's error; a file that cannot be read keeps the read's own `OSError`.
    A checkpoint without a model record this build reads is refused for that
    (`ModelRecordError`, a `ValueError`): its view and split are not known.
    """
    from src.models.bundle.record import ModelRecordError, read_model_record
    from src.utils.checkpoint_io import read_checkpoint

    try:
        read = read_checkpoint(Path(checkpoint).resolve(), map_location="cpu",
                               weights_only=True)
    except (OSError, MemoryError):
        raise
    except Exception as exc:  # noqa: BLE001 - torch raises a different type for each kind of junk
        raise CohortVersionRefused(
            f"{checkpoint} could not be read as a checkpoint ({type(exc).__name__}: {exc})"
        ) from exc
    try:
        record = read_model_record(read.checkpoint)
    except ModelRecordError as exc:
        raise CohortVersionRefused(
            f"the checkpoint {checkpoint} has no model record this build reads ({exc}), so "
            "the graph view and edge split it was trained on are not known"
        ) from None
    return read, record


def prepare_kg_edges_import(
    registry: Any,
    data_dir: Path,
    checkpoint: Path,
    label: str,
    role: str,
) -> ImportPlan:
    """Read the workspace and the checkpoint, rebuild the checkpoint's view and
    split, build the role's pairs and the version; write nothing.

    In order, each refusal before anything is written:

    1. the registry, the label (free, as at commit) and the role
       (`validate_kg_edges_role`);
    2. the workspace graph through the readers measurement uses: the split
       manifest, the tensors against it, ``kg.json`` identified against it and
       the dataset record (`verify_graph_reads`, `check_unparsed_kg_json`,
       `verify_dataset_record`);
    3. the checkpoint, one read, trained on this graph (`verify_training_record`,
       contract M2.4), with a model record this build reads (which refuses a
       record naming one input two ways);
    4. the view its record names (``data_roles.graph_view``), rebuilt from this
       graph by its rule, must have the recorded digest; its split
       (``data_roles.split``) is rebuilt and checked field by field
       (`rebuild_split`), the first field that differs named;
    5. the pairs of the role's part by base validation's rule
       (`fixed_lp_pairs` with the split's seed and `BASE_FIXED`'s
       ``validation_batch_size``), so a ``val`` set's pairs are the ones base
       validation scores.

    Raises:
        RegistryRefused, CohortVersionRefused: the registry, label, role,
            checkpoint, view or split is refused.
        ValueError: from the workspace readers and verifiers, naming the file.
        OSError: a file could not be read.
    """
    from src.config.recipe import BASE_FIXED
    from src.kg.artifacts import (
        check_unparsed_kg_json,
        read_split_manifest,
        verify_dataset_record,
        verify_graph_reads,
    )
    from src.kg.graph_view import (
        GraphViewError,
        SplitMismatchError,
        build_graph_view,
        rebuild_split,
    )
    from src.kg.storage.file_storage import read_graph_artifacts
    from src.models.bundle.lp_head import fixed_lp_pairs, lp_pairs_table
    from src.utils.fingerprint import verify_training_record

    root = validate_registry(registry)
    label = validate_label(label)
    role = validate_kg_edges_role(role)
    _require_label_free(root, label)

    data_dir, checkpoint = Path(data_dir), Path(checkpoint)
    manifest = read_split_manifest(data_dir)
    graph = read_graph_artifacts(data_dir, map_location="cpu")
    bound = verify_graph_reads(manifest, graph.reads)
    check_unparsed_kg_json(data_dir, manifest)
    dataset_record = verify_dataset_record(manifest, bound["kg"])

    read, record = _read_checkpoint_record(checkpoint)
    verify_training_record(read.checkpoint, bound, tuple(GRAPH_ARTIFACTS),
                           recorded_by=f"checkpoint {checkpoint}",
                           compared_with=f"the workspace {data_dir}")
    checkpoint_sha256 = read.identity.sha256
    del read

    roles = record.data_roles
    view_record = roles.get("graph_view")
    split_record = roles.get("split")
    if not (isinstance(view_record, Mapping) and isinstance(view_record.get("rule"), Mapping)
            and isinstance(split_record, Mapping)):
        raise CohortVersionRefused(
            f"the checkpoint {checkpoint}'s model record names no graph view rule or edge "
            f"split ({', '.join(KG_EDGES_TAKEN_FROM)}), so there is no split to take a part of"
        )
    try:
        view = build_graph_view(graph.graph_data, rule=copy.deepcopy(dict(view_record["rule"])))
    except GraphViewError as exc:
        raise CohortVersionRefused(
            f"the graph view rule the checkpoint {checkpoint} records cannot be applied to "
            f"this workspace's graph: {exc}"
        ) from None
    del graph
    recorded_digest = view_record.get("digest")
    if view.digest != recorded_digest:
        raise CohortVersionRefused(
            f"the checkpoint {checkpoint} records graph view {str(recorded_digest)[:12]}..., "
            f"but its rule builds view {view.digest[:12]}... from this workspace's graph: the "
            "record does not describe this graph's view"
        )
    try:
        split = rebuild_split(view, split_record)
    except SplitMismatchError as exc:
        raise CohortVersionRefused(
            f"the checkpoint {checkpoint}'s recorded edge split does not rebuild from this "
            f"workspace's view: {exc}"
        ) from None

    part = split.mask(role)
    batch_size = int(BASE_FIXED["validation_batch_size"])
    pairs = fixed_lp_pairs(view.edge_index(part), view.rel[part], seed=split.seed,
                           batch_size=batch_size)
    pairs_bytes = lp_pairs_table(pairs.head, pairs.relation, pairs.tail, pairs.negative_tail)
    version = build_kg_edges_manifest(
        label=label, role=role, bound=bound, dataset_record=dataset_record,
        checkpoint_sha256=checkpoint_sha256, view=view, split=split, pairs=pairs,
        pairs_sha256=_sha256(pairs_bytes), batch_size=batch_size,
    )
    return ImportPlan(label, role, version, preview_id(version), None, None, None, pairs_bytes)


# ---------------------------------------------------------------------------
# A generated split (research comparison §3 (A), §5 Stage 4)
# ---------------------------------------------------------------------------
#: The split manifest's blocks a generated split's version copies, as they are:
#: the generation definition (the generator's algorithm, its version and the
#: settings it recorded) and the disease allocation it cut, both counts and
#: digests only, and what each split realised.
_SPLIT_MANIFEST_BLOCKS = ("generation", "allocation", "realised")


def _split_files(artifacts: Mapping[str, Any]) -> Dict[str, Any]:
    """The samples files a split manifest's ``artifacts`` binds, by name
    (``train_samples``, ``val_samples``, ...), each with its digest: the
    workspace's split artefacts beside the manifest itself."""
    return {name: digest for name, digest in sorted(artifacts.items())
            if name.endswith("_samples")}


def build_generated_manifest(
    *,
    label: str,
    role: str,
    bound: Mapping[str, str],
    dataset_record: "DatasetRecord",
    split_manifest: Mapping[str, Any],
    split_manifest_sha256: str,
    verified: Any,
    samples_sha256: str,
    n_samples: int,
) -> Dict[str, Any]:
    """A generated split's manifest (schema 2) without its ``imported`` block.

    Built only from what was read and verified: the four digests the
    workspace manifest binds and the dataset record the import verified
    (`_dataset_record_block`); the split manifest, by the digest of the bytes
    read, and its blocks copied as they are (`_SPLIT_MANIFEST_BLOCKS`: the
    generation definition, the allocation, what each split realised); the
    split's files by the digests that manifest binds (`_split_files`); what
    was verified (`GeneratedCohorts`: the splits in scope, disjointness
    measured); the samples, the role's file kept as ``samples.json``, by
    digest and row count; and the counts with their units (the role's
    patients, and the diseases its realised set holds). Aggregates and
    digests only: no sample, and no path.
    """
    realised = split_manifest.get("realised")
    realised = realised if isinstance(realised, Mapping) else {}
    return {
        "schema_version": KINDED_SCHEMA_VERSION,
        "kind": GENERATED,
        "what_this_is": GENERATED_WHAT_THIS_IS,
        "label": label,
        "role": role,
        "graph": {role_name: bound[role_name] for role_name in GRAPH_ARTIFACTS},
        "dataset_record": _dataset_record_block(dataset_record),
        "split_manifest": {"sha256": split_manifest_sha256,
                           "schema_version": split_manifest.get("schema_version")},
        "split_artifacts": _split_files(split_manifest.get("artifacts") or {}),
        **{block: copy.deepcopy(split_manifest.get(block)) for block in _SPLIT_MANIFEST_BLOCKS},
        "verified": {"splits": list(verified.verified),
                     "disjointness_measured": bool(verified.disjointness_measured)},
        "samples": {"file": VERSION_SAMPLES, "sha256": samples_sha256, "n": int(n_samples)},
        "counts": {"samples": int(n_samples), "diseases": realised.get(f"{role}_diseases")},
        "count_units": dict(GENERATED_COUNT_UNITS),
    }


def prepare_generated_import(
    registry: Any,
    data_dir: Path,
    label: str,
    role: str,
) -> ImportPlan:
    """Read and verify a workspace's generated splits, keep the role's; write
    nothing.

    In order, each refusal before anything is written:

    1. the registry, the label (free, as at commit) and the role
       (`validate_generated_role`);
    2. the workspace through the readers ``measure_scorer --split`` uses: the
       split manifest, the tensors against it, ``kg.json`` identified against
       it and the dataset record (`verify_graph_reads`,
       `check_unparsed_kg_json`, `verify_dataset_record`);
    3. the role's split must be one the split manifest binds: the generator
       writes ``train`` and ``val`` only, so ``test`` is refused here, for
       what the workspace lacks, naming the splits it binds;
    4. the generator's splits, one read each (`read_samples`), against that
       one manifest reading (`verify_cohort_reads`: their digests, disease
       sets and disjointness);
    5. the role's file read once more for its bytes, which must hash to the
       digest just verified, so the bytes kept are the bytes verified.

    **Why the samples are copied** (the module's note): a registered version
    is immutable, and a workspace is not; the copy is bound by the digest the
    workspace's split manifest records, so it is verifiable after the
    workspace changes, and the workspace's split stays named by content.

    Raises:
        RegistryRefused, CohortVersionRefused: the registry, label or role is
            refused, the workspace binds no split of that role, or its file
            changed while it was read.
        ValueError: from the workspace readers and verifiers, naming the file.
        OSError: a file could not be read.
    """
    from src.evaluation.cohort import GENERATED_SPLITS, verify_cohort_reads
    from src.kg.artifacts import (
        MANIFEST_FILENAME,
        check_unparsed_kg_json,
        manifest_section,
        read_split_manifest,
        verify_dataset_record,
        verify_graph_reads,
    )
    from src.kg.storage.file_storage import read_graph_artifacts, read_samples
    from src.utils.fingerprint import read_once

    root = validate_registry(registry)
    label = validate_label(label)
    role = validate_generated_role(role)
    _require_label_free(root, label)

    data_dir = Path(data_dir)
    manifest = read_split_manifest(data_dir)
    graph = read_graph_artifacts(data_dir, map_location="cpu")
    bound = verify_graph_reads(manifest, graph.reads)
    del graph
    check_unparsed_kg_json(data_dir, manifest)
    dataset_record = verify_dataset_record(manifest, bound["kg"])

    split_files = _split_files(manifest_section(manifest.manifest, manifest.identity.path,
                                                "artifacts"))
    if f"{role}_samples" not in split_files or role not in GENERATED_SPLITS:
        held = ", ".join(name[: -len("_samples")] for name in split_files) or "none"
        raise CohortVersionRefused(
            f"the workspace {data_dir} holds no generated {role} split this revision can "
            f"verify: its {MANIFEST_FILENAME} binds the splits {held}, and the generator "
            f"writes and verifies {' and '.join(GENERATED_SPLITS)} only, so there is no "
            f"{role} split to keep."
        )
    reads = {name: read_samples(data_dir, name) for name in GENERATED_SPLITS}
    verified = verify_cohort_reads(manifest, reads)
    verified_sha256 = reads[role].identity.sha256
    n_samples = len(reads[role].samples)
    del reads

    kept = read_once(data_dir / f"{role}_samples.json")
    if kept.sha256 != verified_sha256:
        raise CohortVersionRefused(
            f"{kept.identity.path} changed while it was read (its bytes are {kept.sha256[:12]}"
            f"..., the {verified_sha256[:12]}... verified a moment before against "
            f"{MANIFEST_FILENAME}); nothing is kept. Import again once the workspace is not "
            "being written."
        )
    version = build_generated_manifest(
        label=label, role=role, bound=bound, dataset_record=dataset_record,
        split_manifest=manifest.manifest, split_manifest_sha256=manifest.identity.sha256,
        verified=verified, samples_sha256=kept.sha256, n_samples=n_samples,
    )
    # The reader's rules, before anything is written: a split manifest the
    # verifiers passed but which does not state the generation definition a
    # version records (an algorithm and its version) is refused here, not
    # published as a version every reader refuses.
    problems = _generated_shape_problems({**version, "imported": {}})
    if problems:
        raise CohortVersionRefused(
            f"the workspace {data_dir}'s {MANIFEST_FILENAME} does not describe its {role} split "
            f"as a generated split version records it: {'; '.join(problems)}."
        )
    return ImportPlan(label, role, version, preview_id(version), None, None, kept.data)


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
    and bytes against what its manifest binds, something to measure (a
    non-empty sample set; a KG edge set's pairs), and a free label. Then, in
    order: a supplied version's source is kept (no-clobber), the version
    directory is claimed with ``mkdir(exist_ok=False)``, its data file is
    published (``samples.json``, a supplied cohort's mapped samples or a
    generated split's samples as the workspace held them; a KG edge set's
    ``pairs.bin``), and
    ``cohort.json`` -- the commit point -- last, with the ``imported`` block
    (when, and `code_revision(REPO_ROOT)`, both taken before the first write).
    ``sources/``, ``versions/`` and, after the claim, the version directory are
    each resolved again before anything is written into them, and refused when
    a link takes one outside the registry or into the repository
    (`_registry_directory`); the refusal says what was already written. A
    failure after the claim leaves a directory without ``cohort.json``, which
    every reader refuses as incomplete.

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
        data_file = _plan_data_file(plan, label)
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
    if plan.kind == SUPPLIED:
        sources = _keep_source(root, plan)
        kept = (f"Only the source copy was written (under {sources}, named by its digest, "
                "which is harmless to keep).")
        claimed = f"Only the source copy (under {sources}) and the empty directory"
    else:
        kept = "Nothing was written."
        claimed = "Only the empty directory"

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
        done=(f"{claimed} {directory}, which claimed the label, "
              f"{'were' if plan.kind == SUPPLIED else 'was'} written; nothing was written "
              "into it. Every reader refuses it as an interrupted import; remove it to reuse "
              "the label."),
    )

    name, data = data_file
    publish_bytes(directory / name, data)
    publish_bytes(directory / VERSION_MANIFEST, manifest_bytes)
    return resolve_version(root, label)


def _plan_data_file(plan: ImportPlan, label: str) -> Tuple[str, bytes]:
    """The data file a commit publishes before the manifest, by the plan's
    kind, once the plan's label and bytes are those its manifest binds and it
    holds something to measure; else a refusal.

    A supplied version: its label, source and samples, and at least one case
    eligible for ranking. A KG edge set: its label and pairs, and at least one
    pair (a part none of whose edges has a negative would measure nothing). A
    generated split: its label and samples, and at least one sample."""
    manifest = plan.manifest
    kind = plan.kind
    if kind == SUPPLIED:
        source_sha256 = manifest["source"]["sha256"]
        if (plan.label != label or plan.source_sha256 != source_sha256
                or plan.source_bytes is None or plan.samples_bytes is None
                or _sha256(plan.source_bytes) != source_sha256
                or _sha256(plan.samples_bytes) != manifest["samples"]["sha256"]):
            raise CohortVersionRefused(
                f"the plan for {label!r} carries a label, source or samples its manifest "
                "does not bind, so it is not one prepare_import made from one read of each "
                "input."
            )
        if not manifest["samples"]["n"]:
            raise CohortVersionRefused(
                f"no case of the source is eligible for ranking against this graph, so the "
                f"version {label!r} would measure nobody. See the preview's exclusions."
            )
        return VERSION_SAMPLES, plan.samples_bytes
    if kind == KG_EDGES:
        if (plan.label != label or plan.pairs_bytes is None
                or _sha256(plan.pairs_bytes) != manifest["pairs"]["sha256"]):
            raise CohortVersionRefused(
                f"the plan for {label!r} carries a label or pairs its manifest does not "
                "bind, so it is not one prepare_kg_edges_import made from one read of each "
                "input."
            )
        if not manifest["counts"]["pairs"]:
            raise CohortVersionRefused(
                f"none of the {manifest['counts']['positive_edges']} edges of the "
                f"{manifest['role']} part has a negative tail (no other tail of its relation "
                f"in its batch), so the KG edge set {label!r} would measure nothing."
            )
        return VERSION_PAIRS, plan.pairs_bytes
    if kind == GENERATED:
        if (plan.label != label or plan.samples_bytes is None
                or plan.source_bytes is not None or plan.pairs_bytes is not None
                or _sha256(plan.samples_bytes) != manifest["samples"]["sha256"]):
            raise CohortVersionRefused(
                f"the plan for {label!r} carries a label or samples its manifest does not "
                "bind, so it is not one prepare_generated_import made from one read of each "
                "input."
            )
        if not manifest["samples"]["n"]:
            raise CohortVersionRefused(
                f"the workspace's generated {manifest['role']} split holds no sample, so the "
                f"version {label!r} would measure nobody."
            )
        return VERSION_SAMPLES, plan.samples_bytes
    raise CohortVersionRefused(f"the plan for {label!r} is of kind {kind!r}, which this "
                               f"revision does not write ({', '.join(VERSION_KINDS)}).")


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


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value))


def _is_whole(value: Any) -> bool:
    """A whole number, and not a boolean (a seed may be any integer)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _graph_problems(manifest: Dict[str, Any]) -> List[str]:
    """The four graph digests, which `verify_cohort_binding` compares, of
    either kind."""
    graph = manifest["graph"]
    if not isinstance(graph, dict) or not all(_is_sha(graph.get(r)) for r in GRAPH_ARTIFACTS):
        return [f"graph must hold a SHA-256 for each of {list(GRAPH_ARTIFACTS)}"]
    return []


def _kind_problem(manifest: Dict[str, Any]) -> Optional[str]:
    """Why a manifest of a schema this revision reads does not say its kind as
    that schema does, or None: schema 1 states none (it is a supplied cohort),
    schema 2 states one of `VERSION_KINDS` that is written in schema 2
    (`KIND_SCHEMAS`). A kind stated in a schema it is not written in is a
    shape no revision writes; reading it by the rules of the shape it
    resembles would guess at what its fields mean."""
    schema = manifest.get("schema_version")
    if schema == VERSION_SCHEMA_VERSION:
        if "kind" in manifest:
            return (f"is schema {VERSION_SCHEMA_VERSION}, which records no kind, yet states kind "
                    f"{manifest['kind']!r}: it was edited, so it does not say what it is.")
        return None
    if "kind" not in manifest:
        return f"is schema {schema!r}, which must state its kind (missing kind)."
    kind = version_kind(manifest)
    if kind not in VERSION_KINDS:
        return (f"is of kind {manifest['kind']!r}; this revision reads kinds "
                f"{', '.join(VERSION_KINDS[:-1])} and {VERSION_KINDS[-1]} only.")
    if KIND_SCHEMAS[kind] != schema:
        return (f"is schema {schema!r} of kind {kind!r}, which no revision writes: a {kind} "
                f"version is schema {KIND_SCHEMAS[kind]}.")
    return None


def _version_shape_problems(manifest: Dict[str, Any]) -> List[str]:
    """What a reader relies on, beyond the field list, by the manifest's kind
    (`version_kind`): a supplied cohort's (`_supplied_shape_problems`), a KG
    edge set's (`_kg_edges_shape_problems`) or a generated split's
    (`_generated_shape_problems`)."""
    kind = version_kind(manifest)
    if kind == KG_EDGES:
        return _kg_edges_shape_problems(manifest)
    if kind == GENERATED:
        return _generated_shape_problems(manifest)
    return _supplied_shape_problems(manifest)


def _supplied_shape_problems(manifest: Dict[str, Any]) -> List[str]:
    """What a reader relies on, beyond the field list: the four graph digests,
    the dataset record's digest, the samples' digest and count, the source's
    digest, counts that are present and consistent (`_counts_problems`), and a
    role that is still a role."""
    problems = [f"missing {name}" for name in REQUIRED_VERSION_FIELDS if name not in manifest]
    if problems:
        return problems

    problems += _graph_problems(manifest)
    for block in ("dataset_record", "source"):
        if not (isinstance(manifest[block], dict) and _is_sha(manifest[block].get("sha256"))):
            problems.append(f"{block} must hold a sha256")
    samples = manifest["samples"]
    n = samples.get("n") if isinstance(samples, dict) else None
    if not (isinstance(samples, dict) and _is_sha(samples.get("sha256")) and _is_count(n)):
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


def _kg_edges_counts(manifest: Dict[str, Any], problems: List[str]) -> Optional[Dict[str, int]]:
    """A KG edge set's counts when they are exactly `KG_EDGES_COUNT_UNITS`' keys,
    each a whole number of at least 0, with a unit each; else None, with the
    problems named."""
    keys = list(KG_EDGES_COUNT_UNITS)
    units = manifest["count_units"]
    if not (isinstance(units, dict) and set(units) == set(keys)
            and all(isinstance(u, str) and u for u in units.values())):
        problems.append(f"count_units must give a unit for each of {keys}")
    counts = manifest["counts"]
    if not (isinstance(counts, dict) and set(counts) == set(keys)):
        problems.append(f"counts must hold exactly {keys}")
        return None
    bad = [f"counts.{key} must be a whole number of at least 0"
           for key in keys if not _is_count(counts[key])]
    problems += bad
    return None if bad else counts


def _kg_edges_shape_problems(manifest: Dict[str, Any]) -> List[str]:
    """What a reader relies on in a KG edge set, beyond the field list, each
    problem naming its field: the four graph digests and the dataset record's
    digest (`verify_cohort_binding` compares them); a role that is a part of
    the split; the view's rule and digest; the split's rule, seed, digest and
    part counts, recorded for this view; counts with their units, equal to
    what the split and the pairs say (the pairs plus the edges without a
    negative are the part's edges, the message-passing edges the train part's);
    the message-passing graph, the split's train part, saying whether the
    measured edges are in it (exactly when the role is ``train``); the pairs
    file's name, digests, rule, layout, part, seed, batch size, batch count,
    the pairs each batch kept (`_batch_pairs_problems`) and its row count; and
    the checkpoint the view and split were taken from."""
    problems = [f"missing {name}" for name in KG_EDGES_REQUIRED_FIELDS if name not in manifest]
    if problems:
        return problems

    problems += _graph_problems(manifest)
    problems += _dataset_record_problems(manifest["dataset_record"])
    role = manifest["role"]
    try:
        validate_kg_edges_role(role)
    except CohortVersionRefused as exc:
        problems.append(str(exc))
        role = None

    view = manifest["graph_view"]
    if not (isinstance(view, dict) and isinstance(view.get("rule"), dict)
            and _is_sha(view.get("digest"))):
        problems.append("graph_view must hold a rule (an object) and a digest (a SHA-256)")
        view = {}
    split = manifest["split"]
    parts: Optional[Dict[str, Any]] = None
    if not isinstance(split, dict):
        problems.append("split must be an object")
        split = {}
    else:
        if not (isinstance(split.get("rule"), str) and split["rule"]):
            problems.append("split.rule must be a non-empty string")
        if not _is_whole(split.get("seed")):
            problems.append("split.seed must be a whole number")
        if not _is_sha(split.get("digest")):
            problems.append("split.digest must be a SHA-256")
        if split.get("view_digest") != view.get("digest"):
            problems.append("split.view_digest is not graph_view.digest")
        parts = split.get("edges")
        if not (isinstance(parts, dict) and set(parts) == set(KG_EDGES_ROLES)
                and all(_is_count(n) for n in parts.values())):
            problems.append(f"split.edges must hold a whole number of at least 0 for exactly "
                            f"{list(KG_EDGES_ROLES)}")
            parts = None

    counts = _kg_edges_counts(manifest, problems)
    if counts is not None:
        positive, paired = counts["positive_edges"], counts["pairs"]
        without = counts["edges_without_negative"]
        if paired + without != positive:
            problems.append(f"counts.pairs ({paired}) plus counts.edges_without_negative "
                            f"({without}) is not counts.positive_edges ({positive})")
        if parts is not None and role is not None and positive != parts[role]:
            problems.append(f"counts.positive_edges ({positive}) is not split.edges.{role} "
                            f"({parts[role]})")
        if parts is not None and counts["message_passing_edges"] != parts["train"]:
            problems.append(f"counts.message_passing_edges ({counts['message_passing_edges']}) "
                            f"is not split.edges.train ({parts['train']})")

    passing = manifest["message_passing"]
    if not (isinstance(passing, dict) and passing.get("part") == "train"):
        problems.append("message_passing must be the split's train part (part train, README A4)")
    else:
        if counts is not None and not (_is_count(passing.get("edges"))
                                       and passing["edges"] == counts["message_passing_edges"]):
            problems.append(f"message_passing.edges ({passing.get('edges')!r}) is not "
                            f"counts.message_passing_edges ({counts['message_passing_edges']})")
        if role is not None and passing.get("includes_measured_edges") is not (role == "train"):
            problems.append(f"message_passing.includes_measured_edges must be "
                            f"{str(role == 'train').lower()} for role {role}: the train part "
                            "is the message-passing graph")

    pairs = manifest["pairs"]
    if not isinstance(pairs, dict):
        problems.append("pairs must be an object")
    else:
        if pairs.get("file") != VERSION_PAIRS:
            problems.append(f"pairs.file must be {VERSION_PAIRS}")
        problems += [f"pairs.{key} must be a SHA-256" for key in ("sha256", "lp_pairs_digest")
                     if not _is_sha(pairs.get(key))]
        if not (isinstance(pairs.get("rule"), str) and pairs["rule"]):
            problems.append("pairs.rule must be a non-empty string")
        if not isinstance(pairs.get("layout"), dict):
            problems.append("pairs.layout must be an object")
        if role is not None and pairs.get("part") != role:
            problems.append(f"pairs.part ({pairs.get('part')!r}) is not the role ({role})")
        if not (_is_whole(pairs.get("seed")) and pairs.get("seed") == split.get("seed")):
            problems.append(f"pairs.seed ({pairs.get('seed')!r}) is not split.seed "
                            f"({split.get('seed')!r})")
        batch = pairs.get("batch_size")
        if not (_is_count(batch) and batch > 0):
            problems.append("pairs.batch_size must be a whole number above 0")
        elif counts is not None:
            batches = -(-counts["positive_edges"] // batch)
            if not (_is_count(pairs.get("batches")) and pairs["batches"] == batches):
                problems.append(f"pairs.batches ({pairs.get('batches')!r}) is not "
                                f"counts.positive_edges in batches of {batch} ({batches})")
            else:
                problems += _batch_pairs_problems(pairs.get("batch_pairs"), batch,
                                                  counts["positive_edges"], counts["pairs"])
        if counts is not None and not (_is_count(pairs.get("rows"))
                                       and pairs["rows"] == counts["pairs"]):
            problems.append(f"pairs.rows ({pairs.get('rows')!r}) is not counts.pairs "
                            f"({counts['pairs']})")

    taken = manifest["taken_from"]
    if not (isinstance(taken, dict) and _is_sha(taken.get("checkpoint_sha256"))
            and isinstance(taken.get("record_fields"), list)):
        problems.append("taken_from must hold a checkpoint_sha256 and the record_fields read")
    if not isinstance(manifest["imported"], dict):
        problems.append("imported must be an object")
    return problems


def _batch_pairs_problems(batch_pairs: Any, batch_size: int, positive: int,
                          rows: int) -> List[str]:
    """Why ``pairs.batch_pairs`` is not the pairs each batch kept, or none: one
    whole number per batch of `batch_size` over the `positive` edges, none
    above its batch's edge count (the last batch holds the remainder), summing
    to the `rows` the pairs file holds."""
    batches = -(-positive // batch_size)
    sizes = [min(batch_size, positive - k * batch_size) for k in range(batches)]
    if not (isinstance(batch_pairs, list) and len(batch_pairs) == batches
            and all(_is_count(n) for n in batch_pairs)):
        return [f"pairs.batch_pairs must hold a whole number of at least 0 for each of the "
                f"{batches} batches"]
    problems = [f"pairs.batch_pairs[{k}] ({n}) is above its batch's {size} edges"
                for k, (n, size) in enumerate(zip(batch_pairs, sizes, strict=True)) if n > size]
    if sum(batch_pairs) != rows:
        problems.append(f"pairs.batch_pairs sum to {sum(batch_pairs)}, not counts.pairs ({rows})")
    return problems


def _dataset_record_problems(record: Any) -> List[str]:
    """Why a KG edge set's or a generated split's ``dataset_record`` is not
    one a reader can name its sources by, or none: a SHA-256, and its sources
    as `_source_roles_problems` reads them."""
    if not (isinstance(record, dict) and _is_sha(record.get("sha256"))
            and isinstance(record.get("sources"), list)):
        return ["dataset_record must hold a sha256 and its sources"]
    return _source_roles_problems(record["sources"])


def _source_roles_problems(sources: List[Any]) -> List[str]:
    """Why a version's named sources (``dataset_record.sources``) are not each
    a named, digested source role, or none (E4): each an object with a
    ``role`` that is an identifier (`validate_split_name`, the label's rule),
    a ``digest`` that is a SHA-256 and a ``declared_version`` that is text or
    null, no role twice. **No list of allowed roles**: the KG builder's four
    (`src.kg.provenance.SOURCE_ROLES`) are what today's records name, not what
    a version may carry."""
    problems: List[str] = []
    seen: List[str] = []
    for index, entry in enumerate(sources):
        where = f"dataset_record.sources[{index}]"
        if not (isinstance(entry, dict) and set(entry) == {"role", "digest", "declared_version"}):
            problems.append(f"{where} must hold exactly role, digest and declared_version")
            continue
        try:
            role = validate_split_name(entry["role"])
        except ValueError:
            problems.append(f"{where}.role ({entry['role']!r}) is not an identifier")
            continue
        if role in seen:
            problems.append(f"{where} names the role {role!r} a second time")
        seen.append(role)
        if not _is_sha(entry["digest"]):
            problems.append(f"{where}.digest (role {role}) must be a SHA-256")
        if not (entry["declared_version"] is None or isinstance(entry["declared_version"], str)):
            problems.append(f"{where}.declared_version (role {role}) must be text or null")
    return problems


def _generated_shape_problems(manifest: Dict[str, Any]) -> List[str]:
    """What a reader relies on in a generated split, beyond the field list,
    each problem naming its field: the four graph digests and the dataset
    record (its digest and named sources); a role of `GENERATED_ROLES`; the
    split manifest's digest; the split's files, each by a SHA-256, among them
    the role's, which is the samples' digest (the samples are the file the
    workspace's split manifest binds); the generation definition (an
    algorithm and its version) and the allocation, each an object; what each
    split realised, holding the role's disease count, which ``counts.diseases``
    is; what was verified; the samples (``samples.json``, a SHA-256 and a
    count, which ``counts.samples`` is); and counts with their units."""
    problems = [f"missing {name}" for name in GENERATED_REQUIRED_FIELDS
                if name not in manifest]
    if problems:
        return problems

    problems += _graph_problems(manifest)
    problems += _dataset_record_problems(manifest["dataset_record"])
    role = manifest["role"]
    try:
        validate_generated_role(role)
    except CohortVersionRefused as exc:
        problems.append(str(exc))
        role = None

    split_manifest = manifest["split_manifest"]
    if not (isinstance(split_manifest, dict) and _is_sha(split_manifest.get("sha256"))):
        problems.append("split_manifest must hold a sha256")
    files = manifest["split_artifacts"]
    if not (isinstance(files, dict) and files
            and all(isinstance(name, str) and name.endswith("_samples") and _is_sha(digest)
                    for name, digest in files.items())):
        problems.append("split_artifacts must name each split's samples file "
                        "(<split>_samples) with a SHA-256")
        files = {}

    generation = manifest["generation"]
    if not (isinstance(generation, dict) and isinstance(generation.get("algorithm"), str)
            and generation["algorithm"] and _is_whole(generation.get("algorithm_version"))):
        problems.append("generation must state the generator's algorithm and algorithm_version")
    if not isinstance(manifest["allocation"], dict):
        problems.append("allocation must be an object")
    realised = manifest["realised"]
    if not isinstance(realised, dict):
        problems.append("realised must be an object")
        realised = {}
    verified = manifest["verified"]
    if not (isinstance(verified, dict) and isinstance(verified.get("splits"), list)
            and (role is None or role in verified["splits"])
            and isinstance(verified.get("disjointness_measured"), bool)):
        problems.append("verified must list the splits verified, the role's among them, and "
                        "whether disjointness was measured")

    samples = manifest["samples"]
    n = samples.get("n") if isinstance(samples, dict) else None
    if not (isinstance(samples, dict) and samples.get("file") == VERSION_SAMPLES
            and _is_sha(samples.get("sha256")) and _is_count(n)):
        problems.append(f"samples must hold the file {VERSION_SAMPLES}, a sha256 and a count n")
        samples = {}
    elif role is not None and files and files.get(f"{role}_samples") != samples["sha256"]:
        problems.append(f"samples.sha256 is not split_artifacts.{role}_samples: the samples are "
                        "the file the workspace's split manifest binds")

    keys = list(GENERATED_COUNT_UNITS)
    units = manifest["count_units"]
    if not (isinstance(units, dict) and set(units) == set(keys)
            and all(isinstance(u, str) and u for u in units.values())):
        problems.append(f"count_units must give a unit for each of {keys}")
    counts = manifest["counts"]
    if not (isinstance(counts, dict) and set(counts) == set(keys)
            and all(_is_count(counts[key]) for key in keys)):
        problems.append(f"counts must hold a whole number of at least 0 for exactly {keys}")
    else:
        if _is_count(n) and counts["samples"] != n:
            problems.append(f"counts.samples ({counts['samples']}) is not samples.n ({n})")
        if role is not None and counts["diseases"] != realised.get(f"{role}_diseases"):
            problems.append(f"counts.diseases ({counts['diseases']}) is not "
                            f"realised.{role}_diseases ({realised.get(f'{role}_diseases')!r})")
    if not isinstance(manifest["imported"], dict):
        problems.append("imported must be an object")
    return problems


def resolve_version(registry: Any, label: str, *,
                    expected_sha256: Optional[str] = None) -> CohortVersionRead:
    """Resolve a committed version: one read of its ``cohort.json``, parsed from
    those bytes, and the identity of those bytes.

    Does not read ``samples.json`` (a supplied cohort's or a generated
    split's): its path is returned, and the caller reads it once, comparing
    the read's digest with ``manifest["samples"]["sha256"]``. Nor a KG edge
    set's ``pairs.bin``, which `read_kg_edge_pairs` reads. Nor anything a
    version references outside its directory (`check_references` does).

    **Either schema, by kind** (`VERSION_SCHEMAS`): schema 1 is a supplied
    cohort and states no kind; schema 2 states its kind (`VERSION_KINDS`).
    Each kind's fields are checked by its own rules (`_version_shape_problems`).

    The version directory is resolved before anything is read from it, and the
    resolved path is the one returned, so a caller writing reports under it
    writes where this check looked (`_registry_directory`).

    **`expected_sha256` compares before it decodes** (the version a caller
    chose by the digest of its ``cohort.json``): other bytes there -- torn,
    another schema, another valid version -- raise `CohortVersionChanged` and
    are never decoded; a version directory or ``cohort.json`` that is gone
    raises the same, since the bytes chosen are no longer there, never the
    advice to remove an interrupted import. As `read_checkpoint` and
    `read_report` do. The registry and the label are still checked first:
    they are arguments, not bytes.

    Raises:
        RegistryRefused: the registry path is refused (`validate_registry`), or
            the version directory resolves outside it or inside the repository.
        CohortVersionChanged: ``expected_sha256`` is given and the read is not
            it, or there is nothing to read.
        CohortVersionRefused: the label is invalid; there is no such version;
            the directory has no ``cohort.json`` (an interrupted import); the
            manifest cannot be read or parsed, is not an object, is of a schema
            not in `VERSION_SCHEMAS`, does not state its kind as its schema
            does, states one this revision does not read or one in a schema
            that kind is not written in (`KIND_SCHEMAS`), names another label
            than its directory, lacks a required field of its kind, or holds
            counts that are missing or contradict each other or the version.
    """
    from src.utils.fingerprint import read_once

    root = validate_registry(registry)
    label = validate_label(label)
    directory = _registry_directory(root / VERSIONS_DIR / label, within=root,
                                    done="No version is read from there.")
    path = directory / VERSION_MANIFEST
    if not directory.is_dir():
        if expected_sha256 is not None:
            raise CohortVersionChanged(path, expected_sha256, None)
        raise CohortVersionRefused(
            f"the registry {root} holds no version {label!r} (no {directory})."
        )
    try:
        read = read_once(path)
    except FileNotFoundError:
        if expected_sha256 is not None:
            raise CohortVersionChanged(path, expected_sha256, None) from None
        raise CohortVersionRefused(
            f"{directory} has no {VERSION_MANIFEST}: an interrupted import, not a version. "
            "Remove the directory and import again."
        ) from None
    except OSError as exc:
        raise CohortVersionRefused(
            f"{path} could not be read ({type(exc).__name__}: {exc})"
        ) from exc
    identity = read.identity
    if expected_sha256 is not None and identity.sha256 != expected_sha256:
        del read
        raise CohortVersionChanged(path, expected_sha256, identity.sha256)
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
    if isinstance(schema, bool) or schema not in VERSION_SCHEMAS:
        raise CohortVersionRefused(
            f"{path} is schema {schema!r}; this revision reads schemas "
            f"{' and '.join(str(s) for s in VERSION_SCHEMAS)} only."
        )
    kind_problem = _kind_problem(manifest)
    if kind_problem is not None:
        raise CohortVersionRefused(f"{path} {kind_problem}")
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
    samples = (directory / VERSION_SAMPLES if version_kind(manifest) in (SUPPLIED, GENERATED)
               else None)
    return CohortVersionRead(manifest, identity, directory, samples)


def read_kg_edge_pairs(version: CohortVersionRead) -> KgEdgePairsRead:
    """A KG edge set's pairs, from one read of its ``pairs.bin`` (contract M2.1).

    **The digest is of the bytes parsed, and compared before they are**: the
    file is read once (`read_once`), its digest compared with the one the
    version binds (``pairs.sha256``), and only then are the same bytes parsed
    (`lp_pairs_from_table`, by the layout the version records, which must be
    the LP core's `LP_PAIRS_LAYOUT`). The parsed pairs must then be the rows
    and the pairs digest the version records (``pairs.rows``,
    ``pairs.lp_pairs_digest``). A file changed after the import, gone, or of
    another layout is refused by name; nothing is ever rebuilt in its place.

    The version directory is resolved before anything is read from it
    (`_registry_directory`), as every read of a registry file is.

    Returns:
        The pairs as `LPPairs` (head, relation, tail, negative tail, in the
        order made; the version's counts as `fixed_lp_pairs` names them; the
        pairs digest; the pairs each batch kept), and the identity of the
        bytes read.

    Raises:
        CohortVersionRefused: the version is not a KG edge set, or its pairs
            file is missing, is not the bytes it binds, or does not parse to
            the pairs it records.
        RegistryRefused: the version directory resolves into the repository.
    """
    from src.models.bundle.lp_head import (
        LP_PAIRS_LAYOUT,
        LPPairs,
        lp_pairs_digest,
        lp_pairs_from_table,
    )
    from src.utils.fingerprint import read_once

    manifest = version.manifest
    if version.kind != KG_EDGES:
        raise CohortVersionRefused(
            f"{version.identity.path} is a version of kind {version.kind!r}, which keeps no "
            "pairs file."
        )
    block = manifest["pairs"]
    if block["layout"] != LP_PAIRS_LAYOUT:
        raise CohortVersionRefused(
            f"{version.identity.path} lays its pairs out as {block['layout']!r}; this revision "
            f"reads {LP_PAIRS_LAYOUT!r} only. Nothing was read."
        )
    directory = _registry_directory(version.directory, within=None, done="Nothing was read.")
    path = directory / VERSION_PAIRS
    try:
        read = read_once(path)
    except FileNotFoundError:
        raise CohortVersionRefused(
            f"{path} is missing, though {version.identity.path} binds it: the KG edge set "
            "cannot be read."
        ) from None
    identity = read.identity
    if identity.sha256 != block["sha256"]:
        del read
        raise CohortVersionRefused(
            f"{path} is not the pairs file {version.identity.path} binds (its bytes are "
            f"{identity.sha256[:12]}..., the version binds {block['sha256'][:12]}...): it was "
            "changed after the import, so it is not read."
        )
    try:
        head, relation, tail, negative_tail = lp_pairs_from_table(read.data)
    except ValueError as exc:
        raise CohortVersionRefused(f"{path} is not a pairs table: {exc}") from None
    finally:
        del read
    counts = manifest["counts"]
    if head.numel() != block["rows"]:
        raise CohortVersionRefused(
            f"{path} holds {head.numel()} pairs, and {version.identity.path} records "
            f"{block['rows']}."
        )
    digest = lp_pairs_digest(head, relation, tail, negative_tail)
    if digest != block["lp_pairs_digest"]:
        raise CohortVersionRefused(
            f"{path} parses to pairs {digest[:12]}..., not the {block['lp_pairs_digest'][:12]}"
            f"... {version.identity.path} records."
        )
    pairs = LPPairs(head, relation, tail, negative_tail,
                    {"edges": counts["positive_edges"],
                     "without_negative": counts["edges_without_negative"],
                     "pairs": counts["pairs"]}, digest, tuple(block["batch_pairs"]))
    return KgEdgePairsRead(pairs, identity)


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


def _directory_entries(directory: Path, *, done: str) -> List[Path]:
    """The entries of a registry directory a listing reads, none when it does
    not exist, or a refusal when it cannot be listed: `Path.is_dir` and
    `iterdir` raise on a directory this account may not enter (one the other
    account made with ``mkdir -m 700``), and a list must say so rather than
    fail with a bare error."""
    try:
        if not directory.is_dir():
            return []
        return list(directory.iterdir())
    except OSError as exc:
        raise RegistryRefused(
            f"{directory} cannot be listed ({type(exc).__name__}: {exc.strerror or exc}). "
            f"The account that runs this must be able to read it. {done}"
        ) from None


def list_versions(registry: Any) -> List[ListedVersion]:
    """Every entry under the registry's ``versions/``, by name, each read by
    `resolve_version` (the one reader) and kept as a value when it refuses, so
    an interrupted import is listed with its reason rather than hidden.

    Reads each ``cohort.json`` once, and nothing else; never a samples file.
    An entry that cannot even be checked (a directory another account made
    unreadable to this one) is listed with that reason too.

    Raises:
        RegistryRefused: the registry root is refused (`validate_registry`);
            ``versions/`` resolves outside it or into the repository, or cannot
            be listed.
    """
    root = validate_registry(registry)
    versions = _registry_directory(root / VERSIONS_DIR, within=root,
                                   done="No version was read.")
    entries = _directory_entries(versions, done="No version was read.")
    listed: List[ListedVersion] = []
    for entry in sorted(entries, key=lambda p: p.name):
        try:
            listed.append(ListedVersion(entry.name, resolve_version(root, entry.name), None))
        except (RegistryRefused, CohortVersionRefused) as exc:
            listed.append(ListedVersion(entry.name, None, str(exc)))
        except OSError as exc:
            listed.append(ListedVersion(entry.name, None, (
                f"{entry} cannot be read ({type(exc).__name__}: {exc.strerror or exc}).")))
    return listed


def list_report_runs(version: CohortVersionRead) -> List[ListedRun]:
    """The run directories in a version's ``reports/``, newest run id first.

    ``reports/`` is resolved inside the version directory, and each run
    directory inside ``reports/``, as `create_report_dir` checks them before it
    writes: one a link takes elsewhere, or an entry that is not a directory
    named as a run id, is listed with its reason. **The store's own files**
    beside the runs, the usage history and the investigation notes
    (`report_store.STORE_FILES`) and their lock files
    (`report_store.STORE_LOCK_FILES`, K8-6r.1), are not runs and are not
    listed, so no caller has to know the store's layout to skip them. Reads no
    file.

    Raises:
        RegistryRefused: ``reports/`` resolves outside the version directory, or
            the version directory into the repository; or ``reports/`` cannot
            be listed (made unreadable to this account by another).
    """
    directory = _registry_directory(version.directory, within=None, done="Nothing was read.")
    reports = _registry_directory(directory / REPORTS_DIR, within=directory,
                                  done="Nothing was read.")
    from src.evaluation.report_store import STORE_FILES, STORE_LOCK_FILES

    runs: List[ListedRun] = []
    for entry in sorted(_directory_entries(reports, done="Nothing was read."),
                        key=lambda p: p.name, reverse=True):
        if entry.name in STORE_FILES + STORE_LOCK_FILES:
            continue
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
        try:
            is_directory = resolved.is_dir()
        except OSError as exc:
            runs.append(ListedRun(entry.name, None, (
                f"{entry} cannot be checked ({type(exc).__name__}: {exc.strerror or exc})")))
            continue
        if not is_directory:
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


def cohort_version_block(version: CohortVersionRead) -> Dict[str, Any]:
    """The registered version as a report records it: exactly the fields
    `src.evaluation.sidecar.COHORT_VERSION_FIELDS` names, aggregates only.

    `manifest_sha256` is the digest of the `cohort.json` bytes this read parsed,
    the ledger's cohort identity (M4 decision 11). No case id, exclusion entry or
    unmappable term goes in: those stay in the version, in the registry on the
    operator's machine (D7), and the ledger refuses a block naming them.

    **One derivation for both sides of the comparison.** The measurement writes
    this block into its report right after it resolves the version, and a
    registration compares a report's block with this block of the version the
    report store holds it under (`src/evaluation/registration.py`), so a report
    is registered only under the version it measured, with that version's counts.

    **By kind.** A supplied version's block is the fields above, as it always
    was. A KG edge set's (`KG_EDGES_BLOCK_FIELDS`) names its kind and role,
    the manifest digest, the graph and dataset record digests, the view, the
    split, the message-passing graph, the pairs' rule and digests, and the
    counts with their units: aggregates only, never an edge. A generated
    split's (`GENERATED_BLOCK_FIELDS`) names its kind and role, the manifest
    digest, the graph and dataset record digests, the samples' digest, the
    split manifest and the split's files by digest, the generation definition
    and allocation, and the counts with their units: never a sample. What a
    measurement of either records, and what a registration compares, is
    Stage 5's and 6's (research comparison §5).

    Raises:
        CohortVersionRefused: the manifest does not state its source format,
            which `resolve_version`'s shape checks do not cover.
    """
    manifest = version.manifest
    if version.kind == KG_EDGES:
        return {
            "label": manifest["label"],
            "role": manifest["role"],
            "kind": KG_EDGES,
            "manifest_sha256": version.identity.sha256,
            "graph": dict(manifest["graph"]),
            "dataset_record_sha256": manifest["dataset_record"]["sha256"],
            # Copies, so the block cannot alias the version read.
            **{key: copy.deepcopy(manifest[key]) for key in (
                "graph_view", "split", "message_passing", "pairs", "counts", "count_units")},
        }
    if version.kind == GENERATED:
        return {
            "label": manifest["label"],
            "role": manifest["role"],
            "kind": GENERATED,
            "manifest_sha256": version.identity.sha256,
            "graph": dict(manifest["graph"]),
            "dataset_record_sha256": manifest["dataset_record"]["sha256"],
            "samples_sha256": manifest["samples"]["sha256"],
            # Copies, so the block cannot alias the version read.
            **{key: copy.deepcopy(manifest[key]) for key in (
                "split_manifest", "split_artifacts", "generation", "allocation", "counts",
                "count_units")},
        }
    source_format = manifest["source"].get("format")
    if not isinstance(source_format, str) or not source_format:
        raise CohortVersionRefused(
            f"{version.identity.path} records no source format, so a report of it could "
            "not say what its source was. Nothing was written."
        )
    return {
        "label": manifest["label"],
        "role": manifest["role"],
        "manifest_sha256": version.identity.sha256,
        "source_sha256": manifest["source"]["sha256"],
        "source_format": source_format,
        "samples_sha256": manifest["samples"]["sha256"],
        "graph": dict(manifest["graph"]),
        "dataset_record_sha256": manifest["dataset_record"]["sha256"],
        # Copies, so the report cannot alias the version read.
        "mapping": copy.deepcopy(manifest["mapping"]),
        "counts": copy.deepcopy(manifest["counts"]),
    }


# ---------------------------------------------------------------------------
# Content references (E3)
# ---------------------------------------------------------------------------
#: Where a reference's bytes are kept, which is where `check_references` looks:
#: a file of the version's own directory; a file the registry keeps beside the
#: versions (a supplied source); a file of the workspace the version was made
#: from; the checkpoint it was taken from; an object rebuilt from the graph by a
#: recorded rule (a view, a split); a source named by the dataset record only.
IN_VERSION = "version"
IN_REGISTRY = "registry"
IN_WORKSPACE = "workspace"
IN_CHECKPOINT = "checkpoint"
REBUILT = "rebuilt"
NAMED = "named"

#: A reference's state, as `check_references` finds it: the bytes at the
#: location given hash to the recorded digest; they hash to another; there is
#: nothing there; it cannot be read; or it was not looked for (no location was
#: given, or it is rebuilt or only named), which is never a pass.
VERIFIED = "verified"
DIFFERS = "differs"
MISSING = "missing"
UNREADABLE = "unreadable"
NOT_CHECKED = "not checked"
REFERENCE_STATES: Tuple[str, ...] = (VERIFIED, DIFFERS, MISSING, UNREADABLE, NOT_CHECKED)


class ContentReference(NamedTuple):
    """One thing a version depends on, by content identity: its name in the
    version (``graph.kg``, ``source.mondo``, ``checkpoint``, ...), what it is,
    its SHA-256, where its bytes are kept (`IN_VERSION` ...) and, for a file,
    its name there. The name of a file is a location, never the identity."""

    name: str
    what: str
    sha256: str
    kept: str
    file: Optional[str] = None


class ReferenceState(NamedTuple):
    """A reference as `check_references` found it: its state
    (`REFERENCE_STATES`), the digest found when its bytes were read, and why,
    in words."""

    reference: ContentReference
    state: str
    found: Optional[str]
    reason: str


_GRAPH_WHAT = {
    "kg": ("the graph (kg.json): its node naming, node types, species, namespaces and external "
           "ids; every node index the version holds names a node of it"),
    "node_features": "the graph's node features (node_features.pt)",
    "edge_indices": "the graph's typed edges (edge_indices.pt)",
    "num_nodes": "the graph's node counts by type (num_nodes.json)",
}


def version_references(version: Any) -> Tuple[ContentReference, ...]:
    """What a version depends on, by content identity, read from its kind's
    own fields (a `CohortVersionRead` or a manifest): the one list of a
    version's dependency references, never a path.

    Every kind: the four graph files (``graph.<role>``) and the dataset record
    (``kg.provenance.json``), and each source the record names, by role
    (``source.<role>``; E4, any role it names). Then by kind:

    - a KG edge set: the graph view (its relations and the raw-to-view node
      mapping) and the edge split, both rebuilt from the graph by their
      recorded rules; the pairs (``pairs.bin``, in the version); and the
      checkpoint whose model record named the view and split, by its SHA-256;
    - a generated split: the workspace's split manifest and each split's
      samples file it binds (``split.<name>``), and the samples the version
      keeps (``samples.json``, which is the role's file of the workspace);
    - a supplied cohort: its case source, which the registry keeps
      (``sources/<sha256>.json``), and its mapped samples (``samples.json``).
    """
    from src.kg.artifacts import MANIFEST_FILENAME
    from src.kg.provenance import PROVENANCE_FILENAME

    manifest = version.manifest if isinstance(version, CohortVersionRead) else version
    kind = version_kind(manifest)
    record = manifest["dataset_record"]
    refs: List[ContentReference] = [
        ContentReference(f"graph.{role}", _GRAPH_WHAT[role], manifest["graph"][role],
                         IN_WORKSPACE, GRAPH_ARTIFACTS[role])
        for role in GRAPH_ARTIFACTS
    ]
    refs.append(ContentReference("dataset_record", "the dataset record: what the graph was "
                                 "built from", record["sha256"], IN_WORKSPACE,
                                 PROVENANCE_FILENAME))
    if kind == SUPPLIED:
        named = [(role, record[role]) for role in ("hpo", "mondo")
                 if isinstance(record.get(role), Mapping)]
    else:
        named = [(entry["role"], entry) for entry in record["sources"]]
    refs += [ContentReference(f"source.{role}", f"the {role} source the graph was built from, "
                              "as the dataset record names it", entry["digest"], NAMED)
             for role, entry in named]
    if kind == KG_EDGES:
        refs += [
            ContentReference("graph_view", "the graph view: its relations and the raw-to-view "
                             "node mapping, rebuilt from the graph by its recorded rule",
                             manifest["graph_view"]["digest"], REBUILT),
            ContentReference("split", "the edge split of that view, rebuilt by its recorded "
                             "rule and seed", manifest["split"]["digest"], REBUILT),
            ContentReference("pairs", "the fixed link-prediction pairs",
                             manifest["pairs"]["sha256"], IN_VERSION, VERSION_PAIRS),
            ContentReference("checkpoint", "the checkpoint whose model record named the view "
                             "and split", manifest["taken_from"]["checkpoint_sha256"],
                             IN_CHECKPOINT),
        ]
    elif kind == GENERATED:
        refs.append(ContentReference("split_manifest", "the workspace's split manifest: the "
                                     "generation definition, the allocation and the split's "
                                     "files", manifest["split_manifest"]["sha256"],
                                     IN_WORKSPACE, MANIFEST_FILENAME))
        refs += [ContentReference(f"split.{name}", f"the workspace's generated "
                                  f"{name[: -len('_samples')]} split", digest, IN_WORKSPACE,
                                  f"{name}.json")
                 for name, digest in manifest["split_artifacts"].items()]
        refs.append(ContentReference("samples", f"the {manifest['role']} split's samples, kept "
                                     "in the version", manifest["samples"]["sha256"],
                                     IN_VERSION, VERSION_SAMPLES))
    else:
        sha = manifest["source"]["sha256"]
        refs += [
            ContentReference("source", "the case source, kept by the registry", sha,
                             IN_REGISTRY, f"{SOURCES_DIR}/{sha}.json"),
            ContentReference("samples", "the mapped samples, kept in the version",
                             manifest["samples"]["sha256"], IN_VERSION, VERSION_SAMPLES),
        ]
    return tuple(refs)


def check_references(
    version: CohortVersionRead,
    *,
    workspace: Optional[Path] = None,
    checkpoint: Optional[Path] = None,
) -> Tuple[ReferenceState, ...]:
    """Each of the version's references (`version_references`) and its state,
    found by comparing the bytes at a location with the recorded digest.

    **Locations, not identities.** The version's own files are looked for in
    its directory, a supplied source in the registry it was resolved from, a
    workspace's files in `workspace` and the checkpoint at `checkpoint`, each
    only when given; nothing a version records is a path. A reference with no
    location given, one rebuilt from the graph by a recorded rule (checked by
    rebuilding, as the import and a measurement do) and one only named by the
    dataset record are `NOT_CHECKED`, with the reason: never a pass.

    **A missing dependency is a state**, never an exception: a file that is
    not there is `MISSING`, one that cannot be read (a directory, a file this
    account may not read) is `UNREADABLE`, other bytes are `DIFFERS`.

    **Authorisation is not this.** Which registry roots may be read is
    `validate_registry`'s (and the service's ``SHEPHERD_REGISTRY``), checked
    when the version was resolved; which workspace may be read is its
    caller's. This compares bytes with digests and reads only the files named
    by the references, writing nothing.
    """
    from src.utils.fingerprint import file_sha256

    registry_root = version.directory.parent.parent
    places = {IN_VERSION: version.directory, IN_REGISTRY: registry_root,
              IN_WORKSPACE: None if workspace is None else Path(workspace)}
    states: List[ReferenceState] = []
    for ref in version_references(version):
        if ref.kept == REBUILT:
            states.append(ReferenceState(ref, NOT_CHECKED, None, (
                "rebuilt from the graph by its recorded rule, and checked by that rebuild "
                "(the import and a measurement rebuild it), not by a file")))
            continue
        if ref.kept == NAMED:
            states.append(ReferenceState(ref, NOT_CHECKED, None, (
                "named by the dataset record; its file is an input of the KG build and is not "
                "kept with the version")))
            continue
        if ref.kept == IN_CHECKPOINT:
            path = None if checkpoint is None else Path(checkpoint)
            where = "no checkpoint location was given"
        else:
            place = places[ref.kept]
            path = None if place is None else place / ref.file
            where = f"no {ref.kept} location was given"
        if path is None:
            states.append(ReferenceState(ref, NOT_CHECKED, None, where))
            continue
        try:
            found = file_sha256(path)
        except OSError as exc:
            states.append(ReferenceState(ref, UNREADABLE, None, (
                f"{path} cannot be read ({type(exc).__name__}: {exc.strerror or exc})")))
            continue
        if found is None:
            states.append(ReferenceState(ref, MISSING, None, f"{path} is not there"))
        elif found != ref.sha256:
            states.append(ReferenceState(ref, DIFFERS, found, (
                f"{path} holds {found[:12]}..., not the {ref.sha256[:12]}... recorded")))
        else:
            states.append(ReferenceState(ref, VERIFIED, found, f"{path} holds the bytes recorded"))
    return tuple(states)


def is_run_id(name: Any) -> bool:
    """Whether `name` is a run directory's name as `create_report_dir` makes it."""
    return isinstance(name, str) and RUN_ID_PATTERN.fullmatch(name) is not None


def report_store_directory(
    version: CohortVersionRead, *, create: bool = False, done: str = "Nothing was written."
) -> Optional[Path]:
    """The version's report store, ``reports/``, resolved and checked, or `None`
    when it does not exist and `create` is false.

    **Resolved before it is read or written** (`_registry_directory`): the
    version directory must not resolve inside the repository, and ``reports/``
    must resolve inside the version directory. `resolve_version` checked the
    version directory against its registry, so whatever is read from or written
    into the store is inside the registry and never in the repository, whatever
    links are below it. `done` says what was written before a refusal.

    The one way to the store: `create_report_dir` publishes a run into it, and
    `src/evaluation/report_store.py` reads the runs and appends the usage
    history and investigation notes beside them.
    """
    # The resolved version directory is what reports/ must stay inside; one in
    # the repository is refused here, naming it rather than its reports/.
    directory = _registry_directory(version.directory, within=None, done=done)
    reports = _registry_directory(directory / REPORTS_DIR, within=directory, done=done)
    if create:
        reports.mkdir(parents=True, exist_ok=True)
    elif not reports.is_dir():
        return None
    return reports


def report_run_directory(reports: Path, run_id: str) -> Path:
    """One run directory of a report store (`report_store_directory`'s result),
    resolved and refused when a link takes it outside ``reports/`` or into the
    repository. Raises `ValueError` for a name that is not a run id."""
    if not is_run_id(run_id):
        raise ValueError(f"{run_id!r} is not a run id (<UTC time>-<12 hex>-<8 hex>)")
    return _registry_directory(reports / run_id, within=reports,
                               done="Nothing is read from there.")


def create_report_dir(
    version: CohortVersionRead, checkpoint_sha256: str, now: datetime
) -> Tuple[str, Path]:
    """A new directory for one run's report under the version (§2.1).

    The run id is ``<UTC time>-<checkpoint digest, 12>-<8 random hex>``
    (`RUN_ID_PATTERN`): sortable by time, readable as which weights, and unique
    without a lock. ``reports/`` is created when absent; the run directory is
    created with ``exist_ok=False``, so two runs never share one.

    **Every directory is resolved before it is written into**
    (`report_store_directory`, `_registry_directory`): the version directory
    must not resolve inside the repository, ``reports/`` must resolve inside the
    version directory, and the run directory inside ``reports/``.
    `resolve_version` checked the version directory against its registry, so
    the report lands inside the registry and never in the repository, whatever
    links are below it. The resolved run directory is returned, and the caller
    writes through it.
    """
    if not (isinstance(checkpoint_sha256, str) and _SHA256.fullmatch(checkpoint_sha256)):
        raise ValueError(f"{checkpoint_sha256!r} is not a SHA-256 hexdigest")
    run_id = (f"{_utc(now, '%Y%m%dT%H%M%SZ')}-{checkpoint_sha256[:12]}-"
              f"{secrets.token_hex(4)}")
    reports = report_store_directory(version, create=True)
    run_dir = reports / run_id
    run_dir.mkdir(exist_ok=False)
    run_dir = _registry_directory(
        run_dir, within=reports,
        done=f"Only the empty run directory {run_dir} was created; nothing was written into it.",
    )
    return run_id, run_dir


__all__ = [
    "CohortBindingMismatch",
    "CohortVersionChanged",
    "CohortVersionRead",
    "CohortVersionRefused",
    "ContentReference",
    "DEFAULT_PARTIAL_PHENOTYPES",
    "DIFFERS",
    "EXCLUSION_CATEGORIES",
    "GENERATED",
    "GENERATED_BLOCK_FIELDS",
    "GENERATED_COUNT_UNITS",
    "GENERATED_REQUIRED_FIELDS",
    "GENERATED_ROLES",
    "GENERATED_WHAT_THIS_IS",
    "IN_CHECKPOINT",
    "IN_REGISTRY",
    "IN_VERSION",
    "IN_WORKSPACE",
    "ImportPlan",
    "KG_EDGES",
    "KG_EDGES_BLOCK_FIELDS",
    "KG_EDGES_COUNT_UNITS",
    "KG_EDGES_REQUIRED_FIELDS",
    "KG_EDGES_ROLES",
    "KG_EDGES_TAKEN_FROM",
    "KG_EDGES_WHAT_THIS_IS",
    "KINDED_SCHEMA_VERSION",
    "KIND_SCHEMAS",
    "KgEdgePairsRead",
    "ListedRun",
    "ListedVersion",
    "MISSING",
    "MalformedSource",
    "MappedCohort",
    "NAMED",
    "NOT_CHECKED",
    "PARTIAL_PHENOTYPE_POLICIES",
    "RANKS_FILE",
    "REBUILT",
    "REFERENCE_STATES",
    "REPORTS_DIR",
    "REPORT_FILE",
    "REPO_ROOT",
    "REQUIRED_VERSION_FIELDS",
    "RUN_ID_PATTERN",
    "ReferenceState",
    "RegistryRefused",
    "SOURCES_DIR",
    "SOURCE_FORMAT",
    "SUPPLIED",
    "SourceCase",
    "UNREADABLE",
    "UNSUPPORTED_DISEASE_NAMESPACES",
    "UnsupportedDiseaseNamespace",
    "VERIFIED",
    "VERSIONS_DIR",
    "VERSION_KINDS",
    "VERSION_MANIFEST",
    "VERSION_PAIRS",
    "VERSION_SAMPLES",
    "VERSION_SCHEMAS",
    "VERSION_SCHEMA_VERSION",
    "WHAT_THIS_IS",
    "WorkspaceGraph",
    "build_generated_manifest",
    "build_kg_edges_manifest",
    "build_version_manifest",
    "check_references",
    "cohort_version_block",
    "commit_import",
    "create_report_dir",
    "encode_json",
    "is_run_id",
    "list_report_runs",
    "list_versions",
    "map_cases",
    "not_a_supplied_cohort",
    "parse_source",
    "prepare_generated_import",
    "prepare_import",
    "prepare_kg_edges_import",
    "preview_id",
    "read_kg_edge_pairs",
    "read_workspace_graph",
    "report_run_directory",
    "report_store_directory",
    "resolve_version",
    "validate_generated_role",
    "validate_kg_edges_role",
    "validate_label",
    "validate_partial_policy",
    "validate_registry",
    "validate_role",
    "verify_cohort_binding",
    "version_kind",
    "version_references",
    "version_words",
]
