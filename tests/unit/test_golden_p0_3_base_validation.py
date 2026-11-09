"""
P0-3: base validation's pairs, scores and output, as on the unchanged tree.
==========================================================================
`docs/working/research-comparison/README.md` §5 Stage 1 (golden P0-3): the LP
core moves to `src/models/bundle` and base validation must stay
byte-identical, with training rules 1 pinning its validation pairs. This pins
`src/training/lp_task.py`'s `BaseLPTask` on CPU for the toy view of
tests/unit/test_base_lp_task.py and for two demo-size synthetic workspaces
(`tests/fixtures/goldens/values.p0_3_values`):

- its fixed validation pairs `val_head`, `val_rel`, `val_tail` and
  `val_neg_tail` (values and the digest of their bytes) and `val_counts`, with
  the view and split they came from: always exact;
- the positive and negative raw scores `validate` computes, and the dict it
  returns, for weights drawn from one seeded generator (`fixed_weights`):
  exact on the CPU the golden records (`cpu_environment`: torch, machine, CPU
  kernels), where "old validation pairs and math hold in a fixed
  environment" (Checkpoint A); on another CPU, a float may differ in its last
  bits, so there they are compared to within 1e-5 relative (counts exactly),
  which still catches a refactor that changes a pair or a score.

Module: tests/unit/test_golden_p0_3_base_validation.py
"""
from __future__ import annotations

import math

import pytest

pytest.importorskip("torch")

from tests.fixtures.goldens import encode, golden, golden_bytes  # noqa: E402
from tests.fixtures.goldens.values import p0_3_values  # noqa: E402

#: How close a float must be on a CPU other than the golden's.
REL, ABS = 1e-5, 1e-6


def _close(got, want, where):
    """`got` equals `want`, except that two floats need only be within
    `REL`/`ABS` of each other; anything else (an int, a string, a shape, a
    count, None) is compared exactly."""
    if isinstance(want, dict):
        assert isinstance(got, dict) and sorted(got) == sorted(want), where
        for key in want:
            _close(got[key], want[key], f"{where}.{key}")
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), where
        for i, (a, b) in enumerate(zip(got, want, strict=True)):
            _close(a, b, f"{where}[{i}]")
    elif isinstance(want, float) and not isinstance(got, bool) and isinstance(got, float):
        assert math.isclose(got, want, rel_tol=REL, abs_tol=ABS), (where, got, want)
    else:
        assert got == want, (where, got, want)


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory):
    return p0_3_values(tmp_path_factory.mktemp("p0_3"))


def test_p0_3_the_fixed_validation_pairs_are_exact(regenerated):
    """P0-3: for each case, the view, split, task settings, the fixed pairs
    and their counts, and the weights' digest equal the golden exactly on any
    machine: they are integers and digests, and validation scores twice per
    call (positives, then negatives)."""
    expected = golden("p0_3_base_validation.json")
    assert sorted(regenerated["cases"]) == sorted(expected["cases"]) == [
        "base_workspace", "demo_workspace", "toy_view"]
    for name, want in expected["cases"].items():
        got = regenerated["cases"][name]
        for key in ("view_digest", "split", "task", "pairs", "weights"):
            assert got[key] == want[key], (name, key)
        assert got["scores"]["calls"] == want["scores"]["calls"] == 2, name
        assert want["pairs"]["val_head"]["shape"][0] == got["metrics"]["lp_positives"] > 0, name


def test_p0_3_validation_scores_and_its_returned_dict(regenerated):
    """P0-3: the scores `validate` computed and the dict it returned equal the
    golden: exactly, and as bytes, on the golden's CPU; to within `REL` on
    another (module docstring)."""
    expected = golden("p0_3_base_validation.json")
    same_cpu = regenerated["environment"] == expected["environment"]
    for name, want in expected["cases"].items():
        got = regenerated["cases"][name]
        if same_cpu:
            assert got["scores"] == want["scores"], name
            assert got["metrics"] == want["metrics"], name
        else:
            for side in ("positive", "negative"):
                _close(got["scores"][side]["values"], want["scores"][side]["values"],
                       f"{name}.scores.{side}")
                assert got["scores"][side]["shape"] == want["scores"][side]["shape"], name
            _close(got["metrics"], want["metrics"], f"{name}.metrics")
    if same_cpu:
        assert encode(regenerated) == golden_bytes("p0_3_base_validation.json")
