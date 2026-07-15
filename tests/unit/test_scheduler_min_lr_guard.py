"""
Unit tests for the scheduler min_lr_ratio guard (_validate_min_lr_ratio).

Balanced rule (matched to PyTorch's actual behaviour, not stricter):
  - negative ratio  -> invalid for cosine / onecycle / linear
  - onecycle        -> requires ratio > 0 (final_div_factor = 1 / min_lr_ratio)
  - cosine / linear -> ratio == 0 is allowed (decay-to-zero)
  - none            -> ratio is never consumed, so never rejected
"""
from __future__ import annotations

import pytest

# trainer imports torch at module load; skip the whole file where torch is absent.
pytest.importorskip("torch")

from src.training.trainer import _validate_min_lr_ratio  # noqa: E402


# ------------------------------------------------------------------ onecycle
def test_onecycle_zero_raises():
    with pytest.raises(ValueError):
        _validate_min_lr_ratio(0.0, "onecycle")


def test_onecycle_negative_raises():
    with pytest.raises(ValueError):
        _validate_min_lr_ratio(-0.5, "onecycle")


def test_onecycle_positive_ok():
    # no exception
    _validate_min_lr_ratio(0.01, "onecycle")


# ------------------------------------------------------------- cosine / linear
@pytest.mark.parametrize("sched", ["cosine", "linear"])
def test_cosine_linear_zero_allowed(sched):
    # decay-to-zero is legitimate for these schedulers
    _validate_min_lr_ratio(0.0, sched)


@pytest.mark.parametrize("sched", ["cosine", "linear"])
def test_cosine_linear_negative_raises(sched):
    with pytest.raises(ValueError):
        _validate_min_lr_ratio(-0.5, sched)


@pytest.mark.parametrize("sched", ["cosine", "onecycle", "linear"])
def test_positive_ratio_ok_everywhere(sched):
    _validate_min_lr_ratio(0.1, sched)


# ------------------------------------------------------------------------ none
@pytest.mark.parametrize("ratio", [-1.0, 0.0, 0.01])
def test_none_scheduler_never_rejects(ratio):
    # 'none' does not build a scheduler and never consumes the ratio.
    _validate_min_lr_ratio(ratio, "none")
