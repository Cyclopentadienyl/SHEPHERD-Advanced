"""
The seeding bootstrap, on input that is *sensitive* to randomness.
==================================================================
`tests/integration/test_legacy_equivalence.py` runs the whole calibration
procedure, but its workspace is deliberately built so sampling randomness cannot
change the answer (`tests/fixtures/synthetic_workspace.py`). That is the right
property for a comparison of two scorers — and it means that test says nothing
about the launcher's central claim, because it would pass with the seeding
removed entirely.

This file tests the claim directly, on draws that *do* move: same seed twice
gives identical output, a different seed gives different output. Without the
second half the first proves nothing — a constant is also reproducible.

**No graph fixture.** The probes below are a few lines of RNG and, for the worker
case, `DiagnosisDataset` on hand-written samples. Negative sampling needs no
graph, so none is built.

The bootstrap under test is `calibrate_mode_a.SEED_BOOTSTRAP`, invoked through
`calibrate_mode_a._run` — the real function the launcher uses, not a copy of it.
"""
import json
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("torch")

from scripts.calibrate_mode_a import ORACLE_NUM_WORKERS, _run  # noqa: E402

TIMEOUT = 300
SEED_A = 20260818
SEED_B = 20260819

RNG_PROBE = """
import json, random
import numpy, torch
print(json.dumps({
    "python": [random.random() for _ in range(3)],
    "numpy": numpy.random.rand(3).tolist(),
    "torch": torch.rand(3).tolist(),
}))
"""

# Negatives are drawn in `DiagnosisDataset.__getitem__` via `random.randint`
# (`src/kg/data_loader.py:667`), which in a multi-worker loader runs in the worker
# processes. `num_diseases` is large relative to the batch so the draws genuinely
# vary; the synthetic workspace's four diseases could not show this.
WORKER_PROBE = """
import json, sys
from torch.utils.data import DataLoader
from src.kg.data_loader import DiagnosisDataset, DiagnosisSample, diagnosis_collate_fn

workers = int(sys.argv[1])
samples = [
    DiagnosisSample(patient_id=f"P{i}", phenotype_ids=[i % 3, (i + 1) % 3], disease_id=i % 7)
    for i in range(24)
]
loader = DataLoader(
    DiagnosisDataset(samples=samples, num_diseases=500, num_negative_diseases=5),
    batch_size=4, num_workers=workers, shuffle=False, collate_fn=diagnosis_collate_fn,
)
print(json.dumps([b["negative_disease_ids"].tolist() for b in loader]))
"""


def _probe(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(source))
    return path


def _output(seed: int, script: Path, args, cwd: Path):
    """Run one probe through the launcher's own bootstrap and parse what it printed."""
    completed = _run(seed, script, [str(a) for a in args], cwd, TIMEOUT)
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("seeding")
    rng = _probe(root, "rng_probe.py", RNG_PROBE)
    worker = _probe(root, "worker_probe.py", WORKER_PROBE)
    w = ORACLE_NUM_WORKERS

    return {
        "rng_a1": _output(SEED_A, rng, [], root),
        "rng_a2": _output(SEED_A, rng, [], root),
        "rng_b": _output(SEED_B, rng, [], root),
        "workers_a1": _output(SEED_A, worker, [w], root),
        "workers_a2": _output(SEED_A, worker, [w], root),
        "workers_b": _output(SEED_B, worker, [w], root),
        "workers_a_serial": _output(SEED_A, worker, [0], root),
    }


# ---------------------------------------------------------------------------
# The claim
# ---------------------------------------------------------------------------
def test_the_same_seed_reproduces_across_processes(runs):
    """Two separate interpreter processes, same seed, identical draws from all
    three generators. This is what lets the frozen evaluator — which has no seed
    option and may not be given one — be run twice comparably."""
    assert runs["rng_a1"] == runs["rng_a2"]
    assert set(runs["rng_a1"]) == {"python", "numpy", "torch"}


@pytest.mark.parametrize("generator", ["python", "numpy", "torch"])
def test_a_different_seed_changes_every_generator(runs, generator):
    """The half that makes the half above mean something. A constant reproduces
    perfectly too; only this shows the output is actually seed-controlled — and
    per generator, because seeding two of the three and leaving one unseeded would
    otherwise pass."""
    assert runs["rng_a1"][generator] != runs["rng_b"][generator]


# ---------------------------------------------------------------------------
# Worker seeding — where the negatives are actually drawn
# ---------------------------------------------------------------------------
def test_worker_drawn_negatives_reproduce_at_the_fixed_worker_count(runs):
    """PyTorch seeds each worker as `base_seed + worker_id`, with `base_seed`
    drawn from the parent's torch RNG. Seeding the parent therefore reaches into
    the worker processes — which is the only reason a multi-worker run can be
    compared at all."""
    assert runs["workers_a1"] == runs["workers_a2"]
    assert runs["workers_a1"] != runs["workers_b"]


def test_the_worker_count_is_part_of_the_stream(runs):
    """Not a throughput setting. The same seed with a different worker count
    consumes a different random stream and produces different negatives, so the
    launcher must give the harness the worker count the frozen evaluator
    hardcodes (`EvalConfig.num_workers = 4`) rather than a convenient one.

    This is the finding that changed `measure_scorer.py`'s default from 0 to 4:
    on real data the previous default would have guaranteed divergence.
    """
    assert runs["workers_a_serial"] != runs["workers_a1"]
