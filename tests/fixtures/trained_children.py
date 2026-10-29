"""A base and its finetune children, trained for real on CPU, for the tests that need one.

**The one trained-child helper** (K6 decision 10): a 1-epoch base on a small
spec, then one 1-epoch finetune child of it per task asked for, each trained by
`scripts/train_model.py`'s own `train` on a complete workspace, so a test that
serves, measures, lists or loads a task model reads files the production path
wrote, not a payload a test assembled. A gene or patients-like-me child trains
on the gene task's functional fixture, derived first through
`scripts/derive_task_fixture.py` when the workspace has none.

The runs are filed where the server's list finds them by default
(``<workspace>/runs/gatv2/<run>``, named by a start time so the listing's
order is base first, children after, in the order asked); a test that keeps
them elsewhere passes `runs`.

Module: tests/fixtures/trained_children.py
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"

#: The run directory names, by start time (`checkpoint_paths`' run layout).
BASE_RUN = "20261001-000000_base_aaaaaa"
CHILD_RUNS = {"disease": "20261002-000000_disease_bbbbbb",
              "gene": "20261003-000000_gene_cccccc",
              "patients_like_me": "20261004-000000_patients_like_me_dddddd"}

#: Each task's settings for a 1-epoch CPU child on the demo workspace; the
#: task's recipe fills the rest.
_CHILD_SETTINGS: Dict[str, Dict[str, Any]] = {
    "disease": {"batch_size": 8, "num_neighbors": [3, 3], "candidate_sample_size": 8,
                "sparse_sample_size": 8},
    "gene": {"batch_size": 4, "num_neighbors": [3, 3], "sparse_sample_size": 4},
    "patients_like_me": {"batch_size": 4, "num_neighbors": [3, 3], "sparse_sample_size": 4},
}


def small_base_spec() -> Any:
    """The reference base spec, shrunk to train in seconds on CPU (input 8,
    two layers of width 4 with 2 heads); every other field the reference's."""
    from src.models.bundle import reference_spec
    from src.models.bundle.spec import InputSpec

    spec = reference_spec("base")
    return dataclasses.replace(
        spec, input=InputSpec(dim=8),
        encoder=dataclasses.replace(spec.encoder, layers=2, hidden=4, output=4, heads=2)
    ).validate()


def _train(**settings: Any) -> None:
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    import train_model

    # Every setting given counts as the operator's, as a configuration file's
    # keys do (`load_config_with_explicit`).
    train_model.train(train_model.TrainConfig(**settings), set(settings) - {"data_dir",
                                                                            "run_dir"})


def train_children(data_dir: Path, tasks: Sequence[str] = ("disease",), *,
                   runs: Optional[Path] = None) -> Dict[str, Path]:
    """Train a base on `data_dir` (a complete workspace, `write_demo_workspace`)
    and one child of it per task in `tasks`, and return each one's
    ``checkpoints/last.pt`` by ``"base"`` and by task, beside ``"data_dir"``."""
    data_dir = Path(data_dir)
    runs = Path(runs) if runs is not None else data_dir / "runs" / "gatv2"
    base_run = runs / BASE_RUN
    _train(data_dir=str(data_dir), run_dir=str(base_run), learning_stage="base",
           model_spec=small_base_spec().to_dict(), num_epochs=1, batch_size=32,
           num_neighbors=[4, 4], device="cpu", save_top_k=1)
    base = base_run / "checkpoints" / "last.pt"
    out: Dict[str, Path] = {"data_dir": data_dir, "base": base}
    for task in tasks:
        from src.kg.task_fixture import FIXTURE_RECORD, FIXTURE_TASKS, fixture_dir

        if task in FIXTURE_TASKS and not (fixture_dir(data_dir) / FIXTURE_RECORD).exists():
            if str(_SCRIPTS) not in sys.path:
                sys.path.insert(0, str(_SCRIPTS))
            from derive_task_fixture import derive

            derive(data_dir, distractors=4)
        child_run = runs / CHILD_RUNS[task]
        _train(data_dir=str(data_dir), run_dir=str(child_run), learning_stage="finetune",
               task=task, finetune_from=str(base), num_epochs=1, device="cpu", save_top_k=1,
               **_CHILD_SETTINGS[task])
        out[task] = child_run / "checkpoints" / "last.pt"
    return out


def serve(trained: Dict[str, Path], task: str = "disease", **kwargs: Any) -> Any:
    """The pipeline the service builds for ``trained[task]`` over its workspace
    (`build_served_pipeline`, on the CPU): every served pipeline is a model
    read from its checkpoint (K7: no graph is served without one)."""
    from src.inference.pipeline import build_served_pipeline

    data_dir = Path(trained["data_dir"])
    return build_served_pipeline(data_dir / "kg.json", checkpoint_path=str(trained[task]),
                                 data_dir=str(data_dir), **{"device": "cpu", **kwargs})


__all__ = ["BASE_RUN", "CHILD_RUNS", "serve", "small_base_spec", "train_children"]
