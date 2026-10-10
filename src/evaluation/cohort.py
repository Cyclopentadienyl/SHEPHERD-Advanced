"""
What kind of cohort a split is, stated rather than inferred.
============================================================
Two kinds of evaluation cohort exist in this project and they are not variants of
one thing:

- a **generated** cohort — `train_samples.json` and `val_samples.json`, produced
  by `src/kg/sample_generator.py` from a disease allocation. It always has a
  `split_manifest.json` recording the cut it came from;
- a **supplied** cohort — an institutional or external patient set
  (`EVALUATION_COHORTS.md` §4). It carries no allocation, because nobody cut it;
  its relationship to training is a *measurement* (§1.6 reports the upstream
  team's: 109 of 319 UDN diseases present in their simulated cohort), not a
  contract.

**Why a stated kind and not a file-presence check.** Before this module, callers
wrote `if split_manifest.json exists: record it`. That conditional cannot
distinguish three situations it has to: a generated cohort (manifest present), a
supplied cohort (no manifest, and correctly so), and a workspace built before the
allocation step (no manifest, and it is a defect). Reading the first and third as
"absent, carry on" is exactly the backward compatibility that keeps a superseded
pipeline alive inside the current one.

**A supplied cohort may not be named `train` or `val`.** Those names belong to
the generator, and a supplied set written into `val_samples.json` would be
recorded as generated, inherit the manifest of a cut it was never part of, and
look like an ordinary validation number. Refusing the name is what makes §6.7's
role hazard unrepresentable rather than merely discouraged.

The kind is an operator assertion in the same sense as
`src/utils/provenance.py`'s deployment relationship — a bounded vocabulary, and
one no code can verify — but unlike that one it has checkable consequences, and
this module checks every one of them.

Module: src/evaluation/cohort.py
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    FrozenSet,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

# `src.kg` writes these files, so it owns their names; importing the map from
# there keeps the layer order right and gives the writer and the verifier one
# definition instead of two lists that can drift.
from src.kg.artifacts import (
    GRAPH_ARTIFACTS,
    MANIFEST_FILENAME,
    ManifestRead,
    read_split_manifest,
    verify_graph_artifacts,
)

if TYPE_CHECKING:
    from src.kg.storage.file_storage import SamplesRead

#: The generator's own split names. Reserved: a supplied cohort using one would be
#: indistinguishable from a generated one in every artifact downstream.
GENERATED_SPLITS: Tuple[str, ...] = ("train", "val")

#: A split name is an **identifier, not a path fragment**.
#:
#: Lifting the three-value ``choices`` list — so `mygene2` and
#: `institutional_acceptance` stay representable — turned the value into free text
#: that was then interpolated into ``data_dir / f"{split}_samples.json"``.
#: ``pathlib`` lets an absolute value replace the whole path, so ``/tmp/cohort``
#: escaped the workspace entirely and ``../x`` traversed out of it; the same
#: string reaches the measurement manifest and the ledger's ``cohort_role``, so a
#: path could re-enter an evidence artifact as an identity.
#:
#: The alphabet is bounded rather than the values enumerated, which is what keeps
#: arbitrary institutional roles reachable. Anchored, starts with an
#: alphanumeric — so ``.``, ``..``, ``-x`` and ``.hidden`` cannot form — and
#: admits no separator, colon or backslash, which rules out POSIX and Windows
#: traversal, drive letters and UNC paths in one condition rather than a list of
#: special cases someone has to keep complete.
#: ``fullmatch`` at the call site, not ``$``: ``$`` also matches before a final
#: newline, so ``"mygene2\n"`` would pass an alphabet that does not contain one.
SPLIT_NAME = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")

#: What a caller may assert a cohort is.
COHORT_KINDS: Tuple[str, ...] = ("generated", "supplied")

#: The default. Chosen so that the dangerous direction is the one that has to be
#: stated: a supplied cohort under a reserved name is refused either way, and a
#: generated one needs no ceremony.
DEFAULT_COHORT_KIND = "generated"



class CohortProvenance(NamedTuple):
    """One split, resolved: what it is and what describes it."""

    kind: str
    split: str
    samples: Path
    split_manifest: Optional[Path]
    """The manifest for a generated cohort; ``None`` for a supplied one.

    ``None`` here means "this kind of cohort has no allocation", which is a
    positive fact, not a missing file. A generated cohort whose manifest is
    absent never reaches this field — it is refused.
    """

    @property
    def is_generated(self) -> bool:
        return self.kind == "generated"


def validate_split_name(split: Any) -> str:
    """A split name that can only ever address a file inside the workspace.

    **Lowercase, and an uppercase form is refused rather than normalised.** The
    alphabet admitted uppercase while the reserved-name check compared ``train``
    and ``val`` exactly, so a supplied cohort called ``TRAIN`` passed — and on a
    case-insensitive filesystem ``TRAIN_samples.json`` *is* ``train_samples.json``,
    which is precisely the aliasing the reservation exists to prevent. Even where
    the filesystem keeps them apart, two roles differing only in case are two
    identities in the ledger that read as one to a person.

    Refused rather than silently lowercased: normalising would mean the name a
    caller passed and the name recorded in the manifest differ, and an operator
    reading the artifact would have no way to tell.

    Raises:
        ValueError: for anything that is not a bounded lowercase identifier
            token — absolute paths, ``.``/``..``, any name containing ``/``,
            ``\\`` or ``:``, uppercase, a trailing newline, and anything over 64
            characters.
    """
    if not isinstance(split, str) or not SPLIT_NAME.fullmatch(split):
        raise ValueError(
            f"split name must be an identifier — lowercase letters, digits, "
            f"underscore, hyphen or dot, starting with a letter or digit, at most "
            f"64 characters — got {split!r}. It names a cohort, and it is "
            "interpolated into a filename inside the workspace, so a path or a "
            "traversal is refused rather than resolved."
        )
    return split


def validate_kind(kind: Any) -> str:
    if kind not in COHORT_KINDS:
        raise ValueError(f"cohort kind must be one of {COHORT_KINDS}, got {kind!r}")
    return kind


def resolve_cohort(
    data_dir: Path, split: str, kind: str = DEFAULT_COHORT_KIND
) -> CohortProvenance:
    """Check that a split is the kind of cohort it is claimed to be.

    Raises:
        ValueError: on an unknown kind, a reserved name used for a supplied
            cohort, a non-reserved name claimed as generated, a missing samples
            file, or a generated cohort with no manifest.
    """
    validate_kind(kind)
    validate_split_name(split)
    samples = data_dir / f"{split}_samples.json"
    if not samples.is_file():
        raise ValueError(f"{samples} does not exist")

    if kind == "supplied":
        # Compared through casefold as well, defensively: the alphabet already
        # rules uppercase out, so this cannot fire today — and it is the check
        # that would still hold if the alphabet were ever widened.
        if split.casefold() in {name.casefold() for name in GENERATED_SPLITS}:
            raise ValueError(
                f"{split!r} is one of the generator's own split names "
                f"({', '.join(GENERATED_SPLITS)}), so a supplied cohort may not use "
                "it. Written there it would inherit the manifest of a cut it was "
                "never part of and be recorded as a generated cohort. Give it its "
                "own name, such as test_samples.json."
            )
        return CohortProvenance(kind, split, samples, None)

    if split not in GENERATED_SPLITS:
        raise ValueError(
            f"{split!r} is not one of the generator's splits "
            f"({', '.join(GENERATED_SPLITS)}), so it was not produced by "
            "src/kg/sample_generator.py. If it is an institutional or external "
            "cohort, say so with cohort kind 'supplied'."
        )
    manifest = data_dir / MANIFEST_FILENAME
    if not manifest.is_file():
        raise ValueError(
            f"{data_dir} has no {MANIFEST_FILENAME}, so its {split} cohort was "
            "generated before the disease allocation step and its disease sets "
            "overlap. Nothing reads such a workspace any more; rebuild it with "
            "scripts/build_knowledge_graph.py --generate-samples. If this is an "
            "external cohort, it is a supplied one and must not use a generated "
            "split name."
        )
    return CohortProvenance(kind, split, samples, manifest)


class GeneratedCohorts(NamedTuple):
    """A verified generated workspace: the manifest, what its files hold, and the
    scope that was actually verified.

    ``verified`` exists so a caller's artifact can describe the gate that ran.
    Reporting a scope the caller assumed, while the verifier checked something
    wider or narrower, is the artifact failing to describe itself.
    """

    manifest: Dict[str, Any]
    disease_sets: Dict[str, FrozenSet[int]]
    verified: Tuple[str, ...]
    disjointness_claim_checked: bool
    disjointness_measured: bool
    """Reported separately because they are separate facts, and a caller's
    artifact must be able to say which of them its gate performed. A single
    "disjointness was handled" flag cannot distinguish a scope that measured it
    from one that only read the manifest's word for it — or from one that did
    neither."""


def verify_generated_cohorts(
    data_dir: Path, splits: Sequence[str] = GENERATED_SPLITS
) -> GeneratedCohorts:
    """Refuse a workspace whose manifest does not describe the files beside it.

    **Existence is not binding.** ``resolve_cohort`` establishes that a manifest
    is present, which stops a pre-allocation workspace — but any
    ``split_manifest.json`` dropped into such a workspace would satisfy it, and
    training would then proceed on overlapping cohorts under a manifest that
    describes a different cut entirely. This is the check that makes the manifest
    mean something, and it runs at every entry point that consumes generated
    cohorts, not only in the audit.

    **Scoped to what the caller consumes.** A supplied-cohort overlap audit binds
    generated ``train`` and never reads generated ``val``; requiring ``val`` there
    would let a missing or corrupt file block an institutional measurement for a
    reason unrelated to it. Everything that *does* consume both — training,
    measurement, the fidelity audit, a generated train/val audit — passes both
    and gets the disjointness check with them. The returned value
    names the scope, so the caller's report can describe the gate that actually
    ran instead of the one it assumed.

    Four things are established, and they prove different facts:

    1. **The files are the bytes the manifest describes** — SHA-256 against
       ``artifacts.{split}_samples``. A disease-set digest alone cannot see this:
       phenotypes, genes, patient ids, row order and multiplicity can all change
       while the disease set is preserved.
    2. **The disease sets are the ones it recorded** — recomputed from the
       records through the shared reader (`read_samples`), against
       ``realised.*_digest``.
    3. **Its realised sets are what the allocation cut** — internal to the file,
       and what makes "these files are that allocation's cohorts" transitive.
    4. **The cohorts are disjoint** — both the measurement and the manifest's own
       ``disjoint`` claim, and **both only when both cohorts are in scope**. That
       field describes the generated train/val relationship, which a
       train-versus-supplied overlap does not consume; refusing on it at narrow
       scope would be the same over-validation this scoping removes, arriving
       through a claim instead of a file.

    Reads the sample files in scope, so it costs one pass over the cohorts the
    caller consumes. Called once per run, before hours of training or minutes of
    measurement.

    **The path form, a migration aid until contract M2.1's S9.** It hashes each
    sample file by path and parses it through a second read, so the digest and
    the disease sets can describe different bytes. `verify_cohort_reads` makes
    the same four checks from the samples a run already parsed, against its one
    manifest reading.

    Raises:
        ValueError: naming which of the four failed, and for which split.
    """
    from src.kg.storage.file_storage import read_samples
    from src.utils.fingerprint import file_sha256

    scope = tuple(splits)
    _require_scope(scope)
    cohorts = {split: resolve_cohort(data_dir, split) for split in scope}
    manifest_read = read_split_manifest(data_dir)
    manifest, manifest_path = manifest_read.manifest, manifest_read.identity.path

    disease_sets: Dict[str, FrozenSet[int]] = {}
    for split, cohort in cohorts.items():
        _check_samples_digest(manifest, manifest_path, split, cohort.samples,
                              file_sha256(cohort.samples))
        ids = frozenset(
            int(sample.disease_id) for sample in read_samples(data_dir, split).samples
        )
        disease_sets[split] = ids
        _check_disease_set(manifest, manifest_path, split, cohort.samples, ids)

    both_in_scope = _check_disjointness(
        manifest, manifest_path, scope, disease_sets,
        {split: cohort.samples for split, cohort in cohorts.items()},
    )
    return GeneratedCohorts(
        manifest, disease_sets, scope,
        disjointness_claim_checked=both_in_scope,
        disjointness_measured=both_in_scope,
    )


def verify_cohort_reads(
    manifest: ManifestRead, samples: Mapping[str, "SamplesRead"]
) -> GeneratedCohorts:
    """`verify_generated_cohorts`' four checks, from reads a run already made.

    **Compares, and reads nothing** (contract M2.1). `manifest` is the run's one
    reading of its manifest and `samples` maps each generated split in scope to
    what `read_samples` parsed and its identity. Each sample file's digest is the
    digest of the bytes those samples were parsed from, and the disease sets are
    recomputed from the same samples, so the check and the run's input cannot
    describe different bytes. A file replaced after the manifest read is refused
    by name.

    The scope is the splits passed, in the generator's order; the disjointness
    checks run only when both are in scope, as in the path form. The caller has
    already resolved its cohort (`resolve_cohort`) and read the manifest through
    `read_split_manifest`, which checked its schema.

    Raises:
        ValueError: naming which check failed, the file and the manifest.
    """
    manifest_dict, manifest_path = manifest.manifest, manifest.identity.path
    unknown = [split for split in samples if split not in GENERATED_SPLITS]
    scope = tuple(split for split in GENERATED_SPLITS if split in samples)
    _require_scope(scope if not unknown else tuple(samples))

    disease_sets: Dict[str, FrozenSet[int]] = {}
    for split in scope:
        read = samples[split]
        _check_samples_digest(manifest_dict, manifest_path, split, read.identity.path,
                              read.identity.sha256)
        ids = frozenset(int(sample.disease_id) for sample in read.samples)
        disease_sets[split] = ids
        _check_disease_set(manifest_dict, manifest_path, split, read.identity.path, ids)

    both_in_scope = _check_disjointness(
        manifest_dict, manifest_path, scope, disease_sets,
        {split: samples[split].identity.path for split in scope},
    )
    return GeneratedCohorts(
        manifest_dict, disease_sets, scope,
        disjointness_claim_checked=both_in_scope,
        disjointness_measured=both_in_scope,
    )


def _require_scope(scope: Tuple[str, ...]) -> None:
    unknown = [split for split in scope if split not in GENERATED_SPLITS]
    if unknown or not scope:
        raise ValueError(
            f"verification scope must be a non-empty subset of {GENERATED_SPLITS}, "
            f"got {scope!r}"
        )


def _check_samples_digest(
    manifest: Dict[str, Any], manifest_path: Path, split: str, path: Path, observed: Any
) -> None:
    """Check 1: the file is the bytes the manifest describes."""
    recorded = manifest.get("artifacts", {}).get(f"{split}_samples")
    if recorded != observed:
        raise ValueError(
            f"{path} is not the file {manifest_path} describes "
            f"({str(recorded)[:12]}... vs {str(observed)[:12]}...). Either the "
            "samples were replaced after the manifest was written, or the "
            "manifest came from another workspace."
        )


def _check_disease_set(
    manifest: Dict[str, Any], manifest_path: Path, split: str, path: Path,
    ids: FrozenSet[int],
) -> None:
    """Checks 2 and 3: the disease set is the one recorded, which is the one allocated."""
    from src.kg.disease_allocation import disease_set_digest

    realised = manifest.get("realised", {})
    allocated = manifest.get("allocation", {}).get("allocated", {})
    if disease_set_digest(sorted(ids)) != realised.get(f"{split}_digest"):
        raise ValueError(
            f"the {split} cohort's disease set ({path}) is not the one "
            f"{manifest_path} records as realised"
        )
    if realised.get(f"{split}_digest") != allocated.get(f"{split}_digest"):
        raise ValueError(
            f"{manifest_path} contradicts itself: its realised {split} digest "
            "is not its allocated one, so full coverage did not hold"
        )


def _check_disjointness(
    manifest: Dict[str, Any], manifest_path: Path, scope: Tuple[str, ...],
    disease_sets: Dict[str, FrozenSet[int]], paths: Dict[str, Path],
) -> bool:
    """Check 4, when both cohorts are in scope. Returns whether it ran.

    **Both disjointness checks are scoped, not just the measurement.** The
    manifest's `disjoint` field describes the generated train/val relationship.
    A train-versus-supplied overlap audit consumes neither that relationship nor
    `val`, so refusing it on that field would block an institutional measurement
    on the state of an input it never reads — the same over-validation the
    scoping was introduced to remove, arriving through a claim instead of a file.
    """
    if set(scope) != set(GENERATED_SPLITS):
        return False
    if manifest.get("disjoint") is not True:
        raise ValueError(
            f"{manifest_path} claims disjoint={manifest.get('disjoint')!r}. "
            "Disjointness is a contract of the allocation step, so this is a "
            "broken workspace, not a measurement."
        )
    shared = disease_sets["train"] & disease_sets["val"]
    if shared:
        raise ValueError(
            f"{paths['train'].parent} does not hold disease-disjoint cohorts: "
            f"{paths['train'].name} and {paths['val'].name} share {len(shared)} "
            f"diseases, though {manifest_path} claims they share none. "
            "Disjointness is a contract of the allocation step, so this is a "
            "broken workspace, not a measurement."
        )
    return True


__all__ = [
    "COHORT_KINDS",
    "DEFAULT_COHORT_KIND",
    "GENERATED_SPLITS",
    "MANIFEST_FILENAME",
    "CohortProvenance",
    "GRAPH_ARTIFACTS",
    "GeneratedCohorts",
    "resolve_cohort",
    "validate_split_name",
    "verify_cohort_reads",
    "verify_generated_cohorts",
    "verify_graph_artifacts",
    "validate_kind",
]
