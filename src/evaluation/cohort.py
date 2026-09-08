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

import json
from pathlib import Path
from typing import Any, Dict, FrozenSet, NamedTuple, Optional, Tuple

#: The generator's own split names. Reserved: a supplied cohort using one would be
#: indistinguishable from a generated one in every artifact downstream.
GENERATED_SPLITS: Tuple[str, ...] = ("train", "val")

#: What a caller may assert a cohort is.
COHORT_KINDS: Tuple[str, ...] = ("generated", "supplied")

#: The default. Chosen so that the dangerous direction is the one that has to be
#: stated: a supplied cohort under a reserved name is refused either way, and a
#: generated one needs no ceremony.
DEFAULT_COHORT_KIND = "generated"

MANIFEST_FILENAME = "split_manifest.json"


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
    samples = data_dir / f"{split}_samples.json"
    if not samples.is_file():
        raise ValueError(f"{samples} does not exist")

    if kind == "supplied":
        if split in GENERATED_SPLITS:
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
    """A verified generated workspace: the manifest, and what its files hold."""

    manifest: Dict[str, Any]
    disease_sets: Dict[str, FrozenSet[int]]


def verify_generated_cohorts(data_dir: Path) -> GeneratedCohorts:
    """Refuse a workspace whose manifest does not describe the files beside it.

    **Existence is not binding.** ``resolve_cohort`` establishes that a manifest
    is present, which stops a pre-allocation workspace — but any
    ``split_manifest.json`` dropped into such a workspace would satisfy it, and
    training would then proceed on overlapping cohorts under a manifest that
    describes a different cut entirely. This is the check that makes the manifest
    mean something, and it runs at every entry point that consumes generated
    cohorts, not only in the audit.

    Four things are established, and they prove different facts:

    1. **The files are the bytes the manifest describes** — SHA-256 against
       ``artifacts.{split}_samples``. A disease-set digest alone cannot see this:
       phenotypes, genes, patient ids, row order and multiplicity can all change
       while the disease set is preserved.
    2. **The disease sets are the ones it recorded** — recomputed from the
       records through the reader training uses, against ``realised.*_digest``.
    3. **Its realised sets are what the allocation cut** — internal to the file,
       and what makes "these files are that allocation's cohorts" transitive.
    4. **The cohorts are disjoint**, measured, and the manifest agrees.

    Reads both sample files, so it costs one pass over the workspace's cohorts.
    Called once per run, before hours of training or minutes of measurement.

    Raises:
        ValueError: naming which of the four failed, and for which split.
    """
    from src.kg.disease_allocation import disease_set_digest
    from src.kg.sample_generator import SPLIT_MANIFEST_SCHEMA_VERSION
    from src.kg.storage.file_storage import read_samples
    from src.utils.fingerprint import file_sha256

    cohorts = {split: resolve_cohort(data_dir, split) for split in GENERATED_SPLITS}
    manifest_path = data_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())

    version = manifest.get("schema_version")
    if version != SPLIT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"{manifest_path} is split-manifest schema {version!r}; this code reads "
            f"{SPLIT_MANIFEST_SCHEMA_VERSION}. Rebuild the workspace rather than "
            "reading its fields under rules they were not written to."
        )

    artifacts = manifest.get("artifacts", {})
    realised = manifest.get("realised", {})
    allocated = manifest.get("allocation", {}).get("allocated", {})
    disease_sets: Dict[str, FrozenSet[int]] = {}

    for split, cohort in cohorts.items():
        recorded = artifacts.get(f"{split}_samples")
        observed = file_sha256(cohort.samples)
        if recorded != observed:
            raise ValueError(
                f"{cohort.samples} is not the file {manifest_path} describes "
                f"({str(recorded)[:12]}... vs {str(observed)[:12]}...). Either the "
                "samples were replaced after the manifest was written, or the "
                "manifest came from another workspace."
            )
        ids = frozenset(int(sample.disease_id) for sample in read_samples(data_dir, split))
        disease_sets[split] = ids
        if disease_set_digest(sorted(ids)) != realised.get(f"{split}_digest"):
            raise ValueError(
                f"the {split} cohort's disease set is not the one {manifest_path} "
                "records as realised"
            )
        if realised.get(f"{split}_digest") != allocated.get(f"{split}_digest"):
            raise ValueError(
                f"{manifest_path} contradicts itself: its realised {split} digest "
                "is not its allocated one, so full coverage did not hold"
            )

    measured_disjoint = not (disease_sets["train"] & disease_sets["val"])
    if not measured_disjoint or manifest.get("disjoint") is not True:
        raise ValueError(
            f"{data_dir} does not hold disease-disjoint cohorts "
            f"(measured disjoint: {measured_disjoint}; manifest claims: "
            f"{manifest.get('disjoint')!r}). Disjointness is a contract of the "
            "allocation step, so this is a broken workspace, not a measurement."
        )
    return GeneratedCohorts(manifest, disease_sets)


__all__ = [
    "COHORT_KINDS",
    "DEFAULT_COHORT_KIND",
    "GENERATED_SPLITS",
    "MANIFEST_FILENAME",
    "CohortProvenance",
    "GeneratedCohorts",
    "resolve_cohort",
    "verify_generated_cohorts",
    "validate_kind",
]
