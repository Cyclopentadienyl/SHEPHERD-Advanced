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

from pathlib import Path
from typing import Any, NamedTuple, Optional, Tuple

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


__all__ = [
    "COHORT_KINDS",
    "DEFAULT_COHORT_KIND",
    "GENERATED_SPLITS",
    "MANIFEST_FILENAME",
    "CohortProvenance",
    "resolve_cohort",
    "validate_kind",
]
