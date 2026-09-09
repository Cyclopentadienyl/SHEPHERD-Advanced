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

**Uniform only, and now on evidence rather than expectation.** The split
feasibility audit (``scripts/audit_split_feasibility.py``) exists to price the
alternative, and ``docs/working/EVIDENCE_split_feasibility_homelab.json`` —
schema 2, `deployment_relationship: identical-sibling`, over the graph whose
digest is ``6889ed11…`` — finds no stratum close to losing validation
representation under a uniform draw. Across four stratifications and 10,577
eligible diseases of 29,866 disease nodes, the worst case is a 69-disease
``gene_count 11+`` band: **2.9% at f = 0.05, 0.0013% at f = 0.15**, and below
2e-7 at every larger fraction. The worst *training*-side risk anywhere in the
report is 2e-92 at f = 0.05.

At the value this project uses, f = 0.15, one band in a thousand runs would go
unrepresented in validation. Stratifying to remove that would cost a fixed quota
per band and buy a difference of that size, so it is **not built** — and this
module is where it would go if a later artifact overturns the reading.

**What the artifact does and does not settle.** It measures one knowledge graph:
the eligible universe, the band populations and the probabilities all belong to
that MONDO vintage, and the institute's own graph is fetched days apart and will
differ slightly. What transfers is the shape of the answer, since the mechanism
is hypergeometric over band sizes and a few diseases either way cannot move a
margin of that order. A run on the institutional graph is worth having as a
second vintage rather than as a second machine — the operator states the two are
an identical hardware and software build, and these audits read files and count
integers, so no number here depends on which machine produced it.

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


def validate_allocation_seed(value: Any) -> None:
    """The root seed an allocation is cut with.

    **A seed is not only a source of randomness here; it is provenance.**
    ``derive_stream`` stringifies it, so almost anything produces *a* stream —
    but ``DiseaseAllocation`` keeps the object it was given and the manifest
    serialises it, so a value that is not JSON is discovered by ``json.dump``
    after the graph, both cohorts and part of the manifest are already on disk.
    A ``bytes`` seed, or an ``object()`` whose ``repr`` carries a process
    address, also makes the derived stream unreproducible while looking
    deterministic.

    Any Python integer is accepted, of any magnitude: the stream comes from the
    decimal form and nothing downstream packs it into a machine word. ``True``
    is refused for the usual reason — it is an ``int``, and "seed 1" is not what
    a caller writing ``seed=True`` meant.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"seed must be an integer, got {value!r}")


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


def universe_digest(profiles: Sequence["DiseaseProfile"]) -> str:
    """SHA-256 over the eligible universe's **ids and their profile contents**.

    An id-only digest binds an allocation to *which* diseases were cut, not to
    *what they contained*. Two knowledge graphs can agree on every disease index
    and disagree on every phenotype list, and generation would then draw samples
    from profiles the allocation never saw. This digest is what lets generation
    refuse that, by recomputing eligibility from the graph it was actually handed
    and comparing.

    Order-independent by construction: diseases sorted by id, and each profile's
    lists sorted before joining.
    """
    parts = []
    for disease_id, profile in sorted(profiles, key=lambda pair: pair[0]):
        phenotypes = ",".join(str(int(p)) for p in sorted(profile["phenotype_ids"]))
        genes = ",".join(str(int(g)) for g in sorted(profile["gene_ids"]))
        parts.append(f"{int(disease_id)}:{phenotypes}:{genes}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


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
    universe_digest: str

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
            "universe_digest": self.universe_digest,
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


def validate_allocation(allocation: DiseaseAllocation) -> DiseaseAllocation:
    """Check a ``DiseaseAllocation`` that may not have come from this module.

    **Immutability is not validity.** ``DiseaseAllocation`` is a public
    ``NamedTuple``, so any caller can build one by hand — with duplicate disease
    ids, or with the same id on both sides — and the "disjoint by construction"
    claim would then be a claim about a code path that was not taken. The
    checks run wherever an allocation is consumed, not only where one is built.

    A duplicate id is the sharp case: the same disease appearing in both
    partitions is exactly the defect this whole change removes, arriving through
    the front door instead.
    """
    for partition, name in ((allocation.train, "train"), (allocation.val, "val")):
        for disease_id, profile in partition:
            if isinstance(disease_id, bool) or not isinstance(disease_id, int):
                raise ValueError(
                    f"{name} disease id must be an integer, got {disease_id!r}"
                )
            if not isinstance(profile, dict) or "phenotype_ids" not in profile:
                raise ValueError(f"{name} profile for disease {disease_id} is malformed")

    train_ids, val_ids = allocation.train_ids, allocation.val_ids
    for ids, name in ((train_ids, "train"), (val_ids, "val")):
        if len(set(ids)) != len(ids):
            raise ValueError(f"{name} partition contains duplicate disease ids")
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise ValueError(
            f"{len(overlap)} disease(s) appear in both partitions; an allocation "
            "must be disjoint"
        )

    # **Self-consistency: the partitions must still be what the digest says.**
    # The recorded digest is compared against the graph elsewhere, but that only
    # catches an allocation cut from a *different* universe. Tampering with the
    # profiles *inside* an allocation leaves both the ids and the recorded digest
    # untouched, and the graph comparison passes. Recomputing over the current
    # contents is what closes that.
    observed = universe_digest(tuple(allocation.train) + tuple(allocation.val))
    if observed != allocation.universe_digest:
        raise ValueError(
            "allocation contents do not match its recorded universe digest "
            f"({allocation.universe_digest[:12]}... vs {observed[:12]}...); it has "
            "been modified since it was cut"
        )
    return allocation


def require_eligible(allocation: DiseaseAllocation, min_phenotypes: int) -> None:
    """Every allocated disease must satisfy the eligibility rule generation uses.

    An allocation cut at one ``min_phenotypes`` and generated at a stricter one
    would ask ``retained_phenotype_count`` for more phenotypes than a profile
    holds. The rule is checked against the value generation is about to apply,
    not against the one allocation happened to use.
    """
    for partition, name in ((allocation.train, "train"), (allocation.val, "val")):
        for disease_id, profile in partition:
            if len(profile["phenotype_ids"]) < min_phenotypes:
                raise ValueError(
                    f"{name} disease {disease_id} has "
                    f"{len(profile['phenotype_ids'])} phenotypes, below the "
                    f"min_phenotypes={min_phenotypes} generation will apply"
                )


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
    validate_allocation_seed(seed)
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
    return validate_allocation(DiseaseAllocation(
        train=train, val=val, val_fraction_requested=float(val_fraction), seed=seed,
        universe_digest=universe_digest(ordered),
    ))


def train_only_allocation(
    eligible: Sequence[DiseaseProfile], seed: int
) -> DiseaseAllocation:
    """Every eligible disease to training, no validation partition.

    Named rather than signalled by a fraction of zero. Used where a caller wants
    training samples and no validation cohort at all — a distinct intent from
    "withhold a very small fraction", and one a reader should not have to infer
    from a boundary value.
    """
    validate_allocation_seed(seed)
    ordered = tuple(sorted(eligible, key=lambda pair: pair[0]))
    return validate_allocation(DiseaseAllocation(
        train=ordered, val=(), val_fraction_requested=0.0, seed=seed,
        universe_digest=universe_digest(ordered),
    ))
