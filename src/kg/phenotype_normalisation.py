"""
One rule for a case's phenotype list.
====================================
A case that lists one phenotype twice was scored differently from the same case
listing it once: every listed position enters the GNN mean, the shortest-path
mean distance and path search. Entry points disagreed about it — the WebUI removed
repeats by string, the API kept them, and the pipeline kept them and truncated
first. This is the one definition they all apply.

**The key is the graph node, after mapping.** Two source ids that map to one node
are one phenotype, which a string comparison cannot see. Callers therefore map
first and pass node identities here: graph indices in a sample file, KG node ids
in a served request.

**First occurrence, in input order.** The pooled scores do not depend on order,
but path search, candidate ties and explanations do, so the order kept has to be
fixed rather than whatever a set iterates in. First occurrence also keeps the
order a clinician entered.

**What it does not do.** It merges nothing but exact repeats of one node: not
different patients, not an ancestor with its descendant, not frequency, severity
or time. The formats it sees carry positive observations only.

The rule and its scope are recorded in docs/working/PLAN_PHENOTYPE_NORMALISATION.md.
No torch and no I/O, so every layer that needs it can import it.

Module: src/kg/phenotype_normalisation.py
"""
from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any, NamedTuple

#: Recorded wherever the rule's output is recorded, so a record can say which
#: rule produced the list it describes. Bumped when the rule changes what it keeps.
PHENOTYPE_NORMALISATION_VERSION = 1


class NormalisedPhenotypes(NamedTuple):
    """What the rule kept, where it came from, and what it removed."""

    #: The distinct identities, in first-occurrence order.
    kept: tuple[Any, ...]
    #: For each kept identity, its position in the caller's original list.
    kept_positions: tuple[int, ...]
    #: How many entries were repeats of an identity already kept.
    repeats_removed: int


def normalise_phenotypes(
    identities: Sequence[Hashable],
    positions: Sequence[int] | None = None,
) -> NormalisedPhenotypes:
    """Keep the first occurrence of each identity, in order.

    `positions` are the original positions of `identities` in the caller's list,
    for a caller that has already dropped entries — a served request loses its
    unknown ids in mapping, and a per-position field such as
    `phenotype_confidences` has to follow the positions that survive. Omitted, an
    identity's position is its index here.

    Raises:
        ValueError: if `positions` is given and is not one position per identity.
    """
    if positions is None:
        positions = range(len(identities))
    elif len(positions) != len(identities):
        raise ValueError(
            f"{len(positions)} positions for {len(identities)} phenotypes; "
            "each identity needs its original position"
        )

    seen = set()
    kept = []
    kept_positions = []
    for identity, position in zip(identities, positions, strict=True):
        if identity in seen:
            continue
        seen.add(identity)
        kept.append(identity)
        kept_positions.append(position)

    return NormalisedPhenotypes(
        kept=tuple(kept),
        kept_positions=tuple(kept_positions),
        repeats_removed=len(identities) - len(kept),
    )


__all__ = [
    "PHENOTYPE_NORMALISATION_VERSION",
    "NormalisedPhenotypes",
    "normalise_phenotypes",
]
