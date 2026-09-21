"""
An independent reference for the shortest-path primitive.
=========================================================
**This exists to disagree with the served implementation**, and it is worth
nothing if it cannot. `src/inference/sp_index.py` answers queries by building a
composite int64 key and binary-searching it; this answers them by putting every
row in a Python dictionary and looking triples up. The two share no data
structure, no ordering, no arithmetic and no file, so a test comparing them
compares two programs rather than one program with itself.

That failure mode is not hypothetical here. The equivalence tests this replaces
compared the indexed implementation against a scan that would, once dispatch
existed, have *been* the indexed implementation for the same input — which is
how a comparison passes for the wrong reason.

**Not the old scan, and deliberately not a copy of it.** The scan that shipped
until B-0.4's productionisation compared an int32 column against a Python int,
and torch wraps a scalar the dtype cannot hold rather than refusing it, so a
query for target `2**32` was answered with target 0's distance. Reproducing that
here would encode the defect as the expected answer. Python dictionary keys are
exact integers of arbitrary width, so the defect has nowhere to live: an id that
is not in the table is simply not in the dictionary.

**Correct, not fast.** It is O(rows) to build and O(phenotypes x candidates) to
query, in Python. It is used by tests and by `benchmark_sp_lookup.py` as the
baseline the index is measured against. **Nothing under `src/` imports it**, and
it imports nothing from `src.inference.sp_index` — either direction would
collapse the independence it exists for.

Module: scripts/sp_scan_reference.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
from torch import Tensor

__all__ = ["ReferenceTable", "build_reference_table", "sp_mean_distances_reference"]


@dataclass(frozen=True)
class ReferenceTable:
    """The rows and the hop bound, carried together.

    **Shaped so its query has the served primitive's exact signature.** The
    benchmark drives both through one caller factory, and the equivalence tests
    call them with the same arguments; a reference whose signature differed
    would need an adapter, and an adapter is a place for the two to be given
    different arguments without anyone noticing.
    """

    rows: Dict[Tuple[int, int, int], int]
    max_hops: int

    @property
    def unreachable_distance(self) -> float:
        return float(self.max_hops + 1)


def build_reference_table(
    phenotype: Tensor,
    target: Tensor,
    target_type: Tensor,
    distance: Tensor,
    max_hops: int,
) -> ReferenceTable:
    """Every row as a Python dictionary entry, keyed by the triple itself.

    No encoding, so nothing to collide. A later row with the same triple
    overwrites an earlier one; the served builder refuses duplicates outright,
    so a table where that matters is one the index would not have accepted, and
    the tests that exercise duplicates assert the refusal rather than compare
    answers.
    """
    return ReferenceTable(
        rows={
            (int(p), int(t), int(y)): int(d)
            for p, t, y, d in zip(
                phenotype.tolist(),
                target.tolist(),
                target_type.tolist(),
                distance.tolist(),
            )
        },
        max_hops=int(max_hops),
    )


def sp_mean_distances_reference(
    table: ReferenceTable,
    phenotype_indices: Sequence[int],
    target_indices: Sequence[int],
    target_type_idx: int,
) -> Tuple[Tensor, Tensor]:
    """The same contract as the served primitive, computed the obvious way.

    Returns ``(mean_distance, available)``, both ``(C,)`` and float64/bool.
    ``available`` is False only when there was nothing to measure from. A
    phenotype that cannot reach a candidate contributes ``max_hops + 1``.

    The mean is accumulated in Python floats and placed into a float64 tensor,
    which is what makes it an independent check of the primitive's float64
    contract rather than a restatement of it.
    """
    n_candidates = len(target_indices)
    distances = torch.zeros(n_candidates, dtype=torch.float64)
    available = torch.zeros(n_candidates, dtype=torch.bool)
    if not phenotype_indices or n_candidates == 0:
        return distances, available

    unreachable = table.unreachable_distance
    for position, target_idx in enumerate(target_indices):
        total = 0.0
        for phenotype_idx in phenotype_indices:
            hop = table.rows.get(
                (int(phenotype_idx), int(target_idx), int(target_type_idx))
            )
            total += unreachable if hop is None else float(hop)
        distances[position] = total / len(phenotype_indices)
        available[position] = True
    return distances, available
