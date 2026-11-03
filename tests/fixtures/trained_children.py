"""A base and its finetune children, trained for real on CPU, for the tests that need one.

**The one trained-child helper** (K6 decision 10): a 1-epoch base on a small
spec, then one 1-epoch finetune child of it per task asked for, each trained by
`scripts/train_model.py`'s own `train` on a complete workspace, so a test that
serves, measures, lists or loads a task model reads files the production path
wrote, not a payload a test assembled. A gene or patients-like-me child trains
on the gene task's functional fixture, derived first through
`scripts/derive_task_fixture.py` when the workspace has none.

The base's spec and sizes and the disease child's sizes are the demo's, read
from `scripts/setup_demo.py`'s `demo_task_training` (the one statement of the
demo's sizes), so a test's child is trained at the demo's sizes; the gene and
patients-like-me children's sizes, which the demo does not train, are this
helper's own.

The runs are filed where the server's list finds them by default
(``<workspace>/runs/gatv2/<run>``, named by a start time so the listing's
order is base first, children after, in the order asked); a test that keeps
them elsewhere passes `runs`.

A test that runs `train` itself on the demo workspace (with the Trainer
stubbed, or a reader patched) takes the module's `demo` and `workspace`
fixtures (the demo workspace and a base `train_children` trained on it, and a
copy of both per test) and `finetune_demo_base`, the configuration and call
of a disease finetune of that base at the demo's sizes.

Module: tests/fixtures/trained_children.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"

#: The run directory names, by start time (`checkpoint_paths`' run layout).
BASE_RUN = "20261001-000000_base_aaaaaa"
CHILD_RUNS = {"disease": "20261002-000000_disease_bbbbbb",
              "gene": "20261003-000000_gene_cccccc",
              "patients_like_me": "20261004-000000_patients_like_me_dddddd"}

#: The gene and patients-like-me tasks' settings for a 1-epoch CPU child on
#: the demo workspace; the task's recipe fills the rest. The disease child's
#: are the demo's (`demo_task_training`).
_CHILD_SETTINGS: Dict[str, Dict[str, Any]] = {
    "gene": {"batch_size": 4, "num_neighbors": [3, 3], "sparse_sample_size": 4},
    "patients_like_me": {"batch_size": 4, "num_neighbors": [3, 3], "sparse_sample_size": 4},
}


def _demo_training() -> Dict[str, Dict[str, Any]]:
    from scripts.setup_demo import demo_task_training

    return demo_task_training()


def small_base_spec() -> Any:
    """The demo's base spec, `demo_task_training()["base"]["model_spec"]`: the
    reference base spec shrunk to train in seconds on CPU, its sizes stated
    there only; every other field the reference's."""
    from src.models.bundle.spec import ModelSpec

    return ModelSpec.from_dict(_demo_training()["base"]["model_spec"]).validate()


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
    demo = _demo_training()
    child_settings = {**_CHILD_SETTINGS, "disease": demo["disease"]}
    base_run = runs / BASE_RUN
    _train(data_dir=str(data_dir), run_dir=str(base_run), learning_stage="base",
           num_epochs=1, device="cpu", save_top_k=1, **demo["base"])
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
               **child_settings[task])
        out[task] = child_run / "checkpoints" / "last.pt"
    return out


def finetune_demo_base(train_model: Any, data_dir: Path, run: Path, **overrides: Any) -> Any:
    """Train into `run` a disease finetune of the base `train_children`
    trained on `data_dir`, at the demo's disease sizes (`demo_task_training`),
    or, with `resume_from` in `overrides`, a resume of that finetune parent,
    which brings its own stage settings; `overrides` set on top. Through the
    caller's `train_model` module (`scripts.train_model`, the one its patches
    are on), which is told that exactly the fields set here are the
    operator's (contract TM1: the explicit set is stated, never inferred).
    Returns what `train` returns."""
    data_dir = Path(data_dir)
    fields: dict[str, Any] = {"data_dir": str(data_dir), "run_dir": str(run),
                              "learning_stage": "finetune", "task": "disease",
                              "device": "cpu"}
    if "resume_from" not in overrides:
        base = data_dir / "runs" / "gatv2" / BASE_RUN / "checkpoints" / "last.pt"
        fields.update(finetune_from=str(base), num_epochs=1, **_demo_training()["disease"])
    fields.update(overrides)
    return train_model.train(train_model.TrainConfig(**fields), set(fields))


@pytest.fixture(scope="module", name="demo")
def demo_fixture(tmp_path_factory: Any) -> Path:
    """The demo workspace and a 1-epoch base trained on it by `train_children`
    (no child). Shared by a module's tests, and only read: a test works on its
    own copy (`workspace`)."""
    from tests.fixtures.registered_cohort import write_demo_workspace

    data_dir = write_demo_workspace(tmp_path_factory.mktemp("demo") / "data")
    train_children(data_dir, tasks=())
    return data_dir


@pytest.fixture(name="workspace")
def workspace_fixture(demo: Path, tmp_path: Path) -> Path:
    """A copy of `demo`, its base included, for one test."""
    return Path(shutil.copytree(demo, tmp_path / "ws"))


def serve(trained: Dict[str, Path], task: str = "disease", **kwargs: Any) -> Any:
    """The pipeline the service builds for ``trained[task]`` over its workspace
    (`build_served_pipeline`, on the CPU): every served pipeline is a model
    read from its checkpoint (K7: no graph is served without one)."""
    from src.inference.pipeline import build_served_pipeline

    data_dir = Path(trained["data_dir"])
    return build_served_pipeline(data_dir / "kg.json", checkpoint_path=str(trained[task]),
                                 data_dir=str(data_dir), **{"device": "cpu", **kwargs})


__all__ = ["BASE_RUN", "CHILD_RUNS", "demo_fixture", "finetune_demo_base", "serve",
           "small_base_spec", "train_children", "workspace_fixture"]
