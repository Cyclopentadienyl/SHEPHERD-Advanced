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

**Since contract M4 a supplied cohort is a registered cohort version.** It is
imported, mapped to one graph and published by `src/evaluation/cohort_registry.py`,
and measured and audited from there. The loose `<split>_samples.json` written into
a workspace, which this module used to resolve as `supplied`, is gone with its
`kind` argument: `resolve_cohort` resolves the generator's own splits only, and a
supplied cohort has one resolver, the registry's (`PLAN_TEST_RESULTS.md` §5,
"one resolver"). `COHORT_KINDS` stays as the vocabulary a measurement manifest
records.

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
    Tuple,
)

# `src.kg` writes these files, so it owns their names; importing the map from
# there keeps the layer order right and gives the writer and the verifier one
# definition instead of two lists that can drift.
from src.kg.artifacts import (
    GRAPH_ARTIFACTS,
    MANIFEST_FILENAME,
    ManifestRead,
    manifest_section,
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

#: What a measurement manifest records a cohort as: `generated` for the
#: generator's own splits, resolved here; `supplied` for a registered cohort
#: version, resolved by `src/evaluation/cohort_registry.py`.
COHORT_KINDS: Tuple[str, ...] = ("generated", "supplied")


class CohortProvenance(NamedTuple):
    """One generated split, resolved: what it is and what describes it."""

    kind: str
    split: str
    samples: Path
    split_manifest: Path
    """The workspace's manifest, which records the cut the split came from. A
    generated cohort whose manifest is absent never reaches this field — it is
    refused."""

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


def resolve_cohort(data_dir: Path, split: str) -> CohortProvenance:
    """Check that a split is one of the generator's own, and locate it.

    A supplied cohort is not resolved here: it is a registered cohort version
    (`src/evaluation/cohort_registry.resolve_version`), so a name this module
    does not know is refused with a pointer there rather than read as a loose
    file in the workspace.

    Raises:
        ValueError: on a name that is not an identifier, a name that is not one
            of the generator's splits, a missing samples file, or a workspace
            with no manifest.
    """
    validate_split_name(split)
    if split not in GENERATED_SPLITS:
        raise ValueError(
            f"{split!r} is not one of the generator's splits "
            f"({', '.join(GENERATED_SPLITS)}), so it was not produced by "
            "src/kg/sample_generator.py. An institutional or external cohort is "
            "imported as a registered cohort version (scripts/import_cohort.py) and "
            "measured with --cohort LABEL --registry DIR, not read from the workspace."
        )
    samples = data_dir / f"{split}_samples.json"
    if not samples.is_file():
        raise ValueError(f"{samples} does not exist")
    manifest = data_dir / MANIFEST_FILENAME
    if not manifest.is_file():
        raise ValueError(
            f"{data_dir} has no {MANIFEST_FILENAME}, so its {split} cohort was "
            "generated before the disease allocation step and its disease sets "
            "overlap. Nothing reads such a workspace any more; rebuild it with "
            "scripts/build_knowledge_graph.py --generate-samples."
        )
    return CohortProvenance("generated", split, samples, manifest)


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


def verify_cohort_reads(
    manifest: ManifestRead, samples: Mapping[str, "SamplesRead"]
) -> GeneratedCohorts:
    """Refuse a workspace whose manifest does not describe the cohorts a run read.

    **Existence is not binding.** ``resolve_cohort`` establishes that a manifest
    is present, which stops a pre-allocation workspace — but any
    ``split_manifest.json`` dropped into such a workspace would satisfy it, and
    training would then proceed on overlapping cohorts under a manifest that
    describes a different cut entirely. This is the check that makes the manifest
    mean something, and it runs at every entry point that consumes generated
    cohorts, not only in the audits.

    **Scoped to what the caller consumes.** A registered-cohort overlap audit binds
    generated ``train`` and never reads generated ``val``; requiring ``val`` there
    would let a missing or corrupt file block an institutional measurement for a
    reason unrelated to it. Everything that *does* consume both — training,
    measurement of a generated cohort, the fidelity audit, a generated train/val
    audit — passes both and gets the disjointness check with them. The returned
    value names the scope, so the caller's report can describe the gate that
    actually ran instead of the one it assumed.

    Four things are established, and they prove different facts:

    1. **The files are the bytes the manifest describes** — SHA-256 against
       ``artifacts.{split}_samples``. A disease-set digest alone cannot see this:
       phenotypes, genes, patient ids, row order and multiplicity can all change
       while the disease set is preserved.
    2. **The disease sets are the ones it recorded** — recomputed from the
       records the shared reader (`read_samples`) parsed, against
       ``realised.*_digest``.
    3. **Its realised sets are what the allocation cut** — internal to the file,
       and what makes "these files are that allocation's cohorts" transitive.
    4. **The cohorts are disjoint** — both the measurement and the manifest's own
       ``disjoint`` claim, and **both only when both cohorts are in scope**. That
       field describes the generated train/val relationship, which a
       train-versus-registered overlap does not consume.

    **Compares, and reads nothing** (contract M2.1). `manifest` is the run's one
    reading of its manifest and `samples` maps each generated split in scope to
    what `read_samples` parsed and its identity. Each sample file's digest is the
    digest of the bytes those samples were parsed from, and the disease sets are
    recomputed from the same samples, so the check and the run's input cannot
    describe different bytes. A file replaced after the manifest read is refused
    by name. Since S9 there is no path form beside it.

    The scope is the splits passed, in the order passed; the disjointness checks
    run only when both are in scope. The caller has already resolved its cohort
    (`resolve_cohort`) and read the manifest through `read_split_manifest`, which
    checked its schema.

    Raises:
        ValueError: naming which check failed and the manifest, and the samples
            file when the refusal is about one.
    """
    manifest_dict, manifest_path = manifest.manifest, manifest.identity.path
    scope = tuple(samples)
    _require_scope(scope)

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
    recorded = manifest_section(manifest, manifest_path, "artifacts").get(f"{split}_samples")
    if recorded is None:
        raise ValueError(
            f"{manifest_path} records no digest for {split}_samples, so nothing "
            f"says {path} is the cohort it describes. Rebuild the workspace."
        )
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

    realised = manifest_section(manifest, manifest_path, "realised")
    allocated = manifest_section(manifest, manifest_path, "allocation", "allocated")
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
    A train-versus-registered overlap audit consumes neither that relationship nor
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
    "GENERATED_SPLITS",
    "MANIFEST_FILENAME",
    "CohortProvenance",
    "GRAPH_ARTIFACTS",
    "GeneratedCohorts",
    "resolve_cohort",
    "validate_split_name",
    "verify_cohort_reads",
]
