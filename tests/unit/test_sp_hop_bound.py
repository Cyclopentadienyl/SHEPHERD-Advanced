"""The hop bound is provenance, not a default.

`max_hops` sets the unreachable sentinel every shortest-path score is measured
against, so a wrong one does not merely shift a number — it reorders candidates.
The loader used to read its sidecar inside `except Exception: pass`, which meant
a missing file, malformed JSON, an absent key, a string, a boolean and a value
the producer would never write all arrived at the same silent 5.

**What the fix is and is not.** The defect was the silence, not the number: every
operator-facing build path in this repository uses the producer's default of 5,
and the workspace inventories in `docs/` do not list the sidecar, so a real
5-hop table with no sidecar beside it is a shape this project's own
documentation produces. Refusing it would replace a correct score with a
different one. So a missing sidecar still assumes 5 — and says that it did.

*Superseded twice:* a missing sidecar then needed the bound stated
(`sp_hop_bound`), and since contract M2.5 a table without a sidecar is refused
outright: it records no binding to the graph, and the configured bound is retired.
The serving loader retired with the served mixture (K8-4), and its tests with
it; the counterexample below, which says why a wrong bound reorders candidates,
stays with the file until K8-5.

Module: tests/unit/test_sp_hop_bound.py
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


class TestTheRankingThisPrevents:
    """The counterexample the assumption could not survive, asserted as
    behaviour so the reasoning is not only in a commit message."""

    def test_reading_a_three_hop_table_against_five_flips_a_pair(self):
        """Not a claim about the loader — a claim about why its refusal matters.

        Two candidates over one 3-hop table: A's phenotype distances are
        [1, unreachable], B's are [3, 3], both embedding scores 0.5, eta 0.7.
        The sentinel is the only thing that differs between the two readings.
        """
        from src.inference.scoring import sp_scores_from_distances

        def mixture(max_hops):
            unreachable = float(max_hops + 1)
            means = torch.tensor(
                [(1.0 + unreachable) / 2, 3.0], dtype=torch.float64
            )
            sp = sp_scores_from_distances(means)
            return [0.7 * 0.5 + 0.3 * float(v) for v in sp]

        correct_a, correct_b = mixture(3)
        assumed_a, assumed_b = mixture(5)

        assert correct_a > correct_b, "with the real bound, A ranks above B"
        assert assumed_b > assumed_a, (
            "the assumption does not merely shift the scores, it swaps the pair"
        )
