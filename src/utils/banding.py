"""
Banding — turning a population of integers into comparable bands.
=================================================================
Extracted from ``scripts/audit_split_feasibility.py`` when a second audit needed
the same mechanism. The same reason ``DEPLOYMENT_RELATIONSHIPS`` moved to
``src/utils/provenance.py`` and the measurement caveats moved to
``src/evaluation/caveats.py``: these reports are read together, and two copies of
a banding rule produce two reports whose bands look alignable and are not.

**Bands, not raw values, because BACKLOG §5.2 forbids per-disease lists in
evidence artifacts.** A band population is an aggregate; a list of per-disease
capacities is a list of diseases. The bound tuples themselves stay with the audit
that chooses them — they are a judgement about what resolution a question needs —
except ``CAPACITY_BANDS``, which two audits share and which therefore has to be
one definition or the reports cannot be compared.

No torch, no I/O, standard library only.

Module: src/utils/banding.py
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

#: Combination-count bands for ``C(P, k)`` — how many distinct phenotype subsets a
#: disease admits. Shared by the feasibility audit, which reports the capacity a
#: hypothetical allocation would have, and the generator audit, which reports how
#: many draws each capacity actually received. Comparing those two reports is the
#: point, and it requires identical bounds.
CAPACITY_BANDS: Tuple[int, ...] = (1, 2, 6, 21, 101, 1001)

#: Where a value that could not be computed is placed. Sorted **after** every
#: numeric band, ahead of any label tie-break, so bucket order is total.
MISSING_LABEL = "missing"


def band_label(value: Optional[int], bounds: Sequence[int]) -> str:
    """Which band a value falls in, as a stable label.

    ``None`` goes to the explicit missing bucket. It is never imputed and never
    silently dropped. The bucket exists so that the ordering rule is total rather
    than conditional on the data.
    """
    if value is None:
        return MISSING_LABEL
    for lower, upper in zip(bounds, list(bounds[1:]) + [None]):
        if value >= lower and (upper is None or value < upper):
            return f"{lower}+" if upper is None else (
                str(lower) if upper == lower + 1 else f"{lower}-{upper - 1}"
            )
    return MISSING_LABEL


def band_sort_key(label: str, bounds: Sequence[int]) -> Tuple[int, int, str]:
    """Canonical bucket order: ascending lower bound, missing last, then label.

    The first element separates numeric bands (0) from the missing bucket (1),
    which is what puts missing after every band regardless of its lower bound.
    The label is the final tie-break so the ordering is total even if two bands
    were ever given the same bound.
    """
    if label == MISSING_LABEL:
        return (1, 0, label)
    head = label.rstrip("+").split("-")[0]
    try:
        return (0, int(head), label)
    except ValueError:  # pragma: no cover - labels are generated, not parsed
        return (1, 0, label)


def bucket(values: Sequence[Optional[int]], bounds: Sequence[int]) -> List[Tuple[str, int]]:
    """Band populations, in canonical bucket order, empty bands included.

    Empty bands are emitted rather than omitted: a band absent from one report
    and present in another cannot be told apart from a band that was never
    defined, and the two reports stop being alignable.
    """
    counts: Dict[str, int] = {}
    for value in values:
        label = band_label(value, bounds)
        counts[label] = counts.get(label, 0) + 1
    for lower in bounds:
        counts.setdefault(band_label(lower, bounds), 0)
    counts.setdefault(MISSING_LABEL, 0)
    return sorted(counts.items(), key=lambda kv: band_sort_key(kv[0], bounds))


__all__ = ["CAPACITY_BANDS", "MISSING_LABEL", "band_label", "band_sort_key", "bucket"]
