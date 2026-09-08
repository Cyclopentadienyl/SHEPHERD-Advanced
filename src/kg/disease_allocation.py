"""
Disease allocation — cutting the disease universe, before anything is generated.
================================================================================

**Allocation is a separate step from generation, and that separation is the
point** (``docs/working/EVALUATION_COHORTS.md`` §6.2). The generator is handed a
partition; it does not decide one. That is the minimum seam letting a stratified
allocation later replace the uniform one here without redesigning generation.

The defect this exists to fix: ``sample_generator`` used to draw one pooled
sample set and slice it by index, which splits *patients* and never splits
*diseases*. Every disease with more than one patient landed on both sides, so
``EVIDENCE_M4.json`` recorded all 7,970 validation diseases as also present in
training. A metric measured on such a split reports recognition of new phenotype
subsets of diseases the model has labelled examples of — not generalisation to
diseases it has none for.

**Uniform only, deliberately.** The split feasibility audit measured the
alternative's value: at every fraction of practical interest no stratum comes
close to losing validation representation under a uniform draw — the worst case
across four stratifications was 2.9% for a 69-disease band at f = 0.05, and
0.0013% for the same band at f = 0.15. Stratified allocation is therefore not
built. This module is where it would go.

Module: src/kg/disease_allocation.py
"""
from __future__ import annotations

import hashlib
import logging
import math
import random
from typing import Any, Dict, NamedTuple, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Bumped when the allocation *algorithm* changes in a way that makes two
#: allocations of the same universe, fraction and seed differ. Recorded in the
#: split manifest so a reader can tell whether two workspaces were cut the same
#: way, which the digests alone cannot say.
ALLOCATION_ALGORITHM = "uniform-without-replacement"
ALLOCATION_ALGORITHM_VERSION = 1

#: A disease profile pair as ``build_eligible_disease_profiles`` returns it.
DiseaseProfile = Tuple[int, Dict[str, Any]]


def derive_stream(seed: int, name: str) -> random.Random:
    """An independent, reproducible random stream from a root seed and a name.

    **Independent streams, not one sequential generator.** With a single shared
    ``Random``, changing ``num_train`` shifts every draw that follows it, so the
    validation cohort would move whenever the training budget did — two runs
    meant to differ in one dimension would differ in two.

    ``random.Random`` seeds a string through SHA-512, not through Python's
    salted ``hash()``, so ``derive_stream(42, "train")`` is the same stream in
    every process and under every ``PYTHONHASHSEED``. Verified across seeds
    rather than assumed. The ``|`` separator cannot appear in a stream name, so
    the encoding is unambiguous.

    This guarantees stream *isolation*. It does not promise that a given
    high-level algorithm (``sample``, ``shuffle``) yields identical output across
    Python versions, and nothing here relies on that.
    """
    if "|" in name:
        raise ValueError(f"stream name must not contain '|', got {name!r}")
    return random.Random(f"{seed}|{name}")


def disease_set_digest(disease_ids: Sequence[int]) -> str:
    """SHA-256 of a disease set, order-independent.

    Sorted before hashing so the digest identifies the *set*, not the order it
    happened to be built in. This is what lets a later reader confirm two
    workspaces were cut identically without either of them listing a single
    disease id — which BACKLOG §5.2 forbids in evidence artifacts.
    """
    joined = ",".join(str(int(i)) for i in sorted(disease_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


class DiseaseAllocation(NamedTuple):
    """One immutable cut of the disease universe.

    Holds the two partitions as ``(disease_index, profile)`` pairs, ready to be
    handed to generation, plus the provenance a split manifest needs.

    **Allocated is not realised.** These are the diseases a partition *may* draw
    from. Whether every one of them received a sample is a fact about generation,
    derived there from the emitted records and never copied from here.
    """

    train: Tuple[DiseaseProfile, ...]
    val: Tuple[DiseaseProfile, ...]
    val_fraction_requested: float
    seed: int

    @property
    def train_ids(self) -> Tuple[int, ...]:
        return tuple(disease_id for disease_id, _ in self.train)

    @property
    def val_ids(self) -> Tuple[int, ...]:
        return tuple(disease_id for disease_id, _ in self.val)

    def provenance(self) -> Dict[str, Any]:
        """The allocation half of a split manifest. Counts and digests only."""
        return {
            "algorithm": ALLOCATION_ALGORITHM,
            "algorithm_version": ALLOCATION_ALGORITHM_VERSION,
            "seed": self.seed,
            "stream_derivation": "random.Random(f'{seed}|{stream_name}')",
            "val_fraction_requested": self.val_fraction_requested,
            "eligible_diseases": len(self.train) + len(self.val),
            "allocated": {
                "train_diseases": len(self.train),
                "val_diseases": len(self.val),
                "train_digest": disease_set_digest(self.train_ids),
                "val_digest": disease_set_digest(self.val_ids),
            },
        }


def withheld_count(n_eligible: int, val_fraction: float) -> int:
    """How many diseases to withhold: round half up, then keep both sides alive.

    The same rule the split feasibility audit reports against, so a fraction
    chosen from that audit's curve produces the partition size the audit
    described. A universe too small to supply both sides is refused rather than
    clamped into a shape it cannot hold.
    """
    if n_eligible < 2:
        raise ValueError(
            f"a disease universe of {n_eligible} cannot be cut into two non-empty "
            "partitions; at least 2 eligible diseases are required"
        )
    return min(max(math.floor(val_fraction * n_eligible + 0.5), 1), n_eligible - 1)


def allocate_diseases(
    eligible: Sequence[DiseaseProfile],
    val_fraction: float,
    seed: int,
) -> DiseaseAllocation:
    """Cut the eligible universe into disjoint training and validation partitions.

    Args:
        eligible: ``(disease_index, profile)`` pairs from
            ``build_eligible_disease_profiles``. The audit measures this same
            universe; a different one here would cost what the audit did not
            price.
        val_fraction: strictly between 0 and 1. **There is no train-only mode
            here** — ``train_only_allocation`` says that explicitly, because a
            fraction of zero standing in for "no validation partition" is a magic
            value a reader has to know about.
        seed: root seed. Allocation draws from its own stream, so changing a
            generation budget cannot move the cut.

    Returns:
        A ``DiseaseAllocation`` whose partitions are disjoint by construction.
    """
    if isinstance(val_fraction, bool) or not isinstance(val_fraction, (int, float)):
        raise ValueError(f"val_fraction must be a number, got {val_fraction!r}")
    if not math.isfinite(val_fraction) or not 0.0 < val_fraction < 1.0:
        raise ValueError(
            "val_fraction must be finite and strictly between 0 and 1 "
            f"(use train_only_allocation for no validation partition), got {val_fraction}"
        )

    ordered = sorted(eligible, key=lambda pair: pair[0])
    n_withheld = withheld_count(len(ordered), val_fraction)

    shuffled = list(ordered)
    derive_stream(seed, "allocation").shuffle(shuffled)
    val = tuple(shuffled[:n_withheld])
    train = tuple(shuffled[n_withheld:])

    logger.info(
        "Allocated %d diseases to train and %d to validation (requested %.4f, "
        "realised %.4f)",
        len(train), len(val), val_fraction, len(val) / len(ordered),
    )
    return DiseaseAllocation(
        train=train, val=val, val_fraction_requested=float(val_fraction), seed=seed
    )


def train_only_allocation(
    eligible: Sequence[DiseaseProfile], seed: int
) -> DiseaseAllocation:
    """Every eligible disease to training, no validation partition.

    Named rather than signalled by a fraction of zero. Used where a caller wants
    training samples and no validation cohort at all — a distinct intent from
    "withhold a very small fraction", and one a reader should not have to infer
    from a boundary value.
    """
    ordered = tuple(sorted(eligible, key=lambda pair: pair[0]))
    return DiseaseAllocation(
        train=ordered, val=(), val_fraction_requested=0.0, seed=seed
    )
