"""`scripts/evaluate_model.py` is frozen, and this is what makes that true.

Its own header says "Do not modify this file. Its only value is being the
unmodified artefact that produced the reference numbers." Until now that was a
sentence, and BACKLOG item 9 depends on it holding: nothing oracle-only may be
deleted before Mode A's institutional calibration succeeds, and a calibration
against a modified oracle is a calibration against nothing.

**Why it is frozen rather than brought up to the current contract.** Every other
entry point now refuses a workspace whose cohorts were cut before the allocation
step. This one does not, and adding the preflight would edit the artefact being
compared against. The freeze wins: the script is non-authoritative and scheduled
for deletion, `scripts/calibrate_mode_a.py` drives it only as the reference
implementation, and nothing treats its output as a supported production result.
That exception is stated in EVALUATION_COHORTS §6.1 rather than left implicit in
a claim of "refused everywhere".

Module: tests/unit/test_frozen_evaluator.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN = REPO_ROOT / "scripts" / "evaluate_model.py"

#: The bytes that produced the reference numbers. Updating this constant is the
#: deliberate act that unfreezing requires -- it leaves a diff, and it belongs in
#: a commit that says why.
FROZEN_DIGEST = "162c9f3da9379aa9123b6b4aaef60402d7d62a15b63fad794fe06a42df0313f7"


def test_the_frozen_evaluator_is_unmodified():
    observed = hashlib.sha256(FROZEN.read_bytes()).hexdigest()

    assert observed == FROZEN_DIGEST, (
        "scripts/evaluate_model.py changed. It is the unmodified artefact Mode A "
        "is calibrated against, so an edit makes it no longer the thing being "
        "compared. If the change is deliberate, update FROZEN_DIGEST in the same "
        "commit and say why; if it is not, revert it."
    )


def test_nothing_in_src_imports_the_frozen_evaluator():
    """It is a script driven by subprocess, never a library. An import from `src`
    would put its behaviour on a path something else depends on, and the freeze
    would then be blocking that code too."""
    offenders = [
        path.relative_to(REPO_ROOT)
        for path in (REPO_ROOT / "src").rglob("*.py")
        if "evaluate_model" in path.read_text()
        and any(
            line.lstrip().startswith(("import ", "from "))
            and "evaluate_model" in line
            for line in path.read_text().splitlines()
        )
    ]

    assert not offenders, f"src modules importing the frozen evaluator: {offenders}"
