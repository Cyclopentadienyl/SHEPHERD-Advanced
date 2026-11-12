"""
CUDA serving and research jobs exclude each other across processes (I2, I3, I8)
==============================================================================
The service's CUDA inference holds the deployment's one GPU lock, the flock
every managed job and formal CLI takes, so the exclusion holds between
processes and not only inside the service:

- A CUDA load in this process holds the lock; `scripts/train_model.py`, run
  as its own process with the same workspaces root, exits before it reads
  anything, its refusal naming inference and how to stop it.
- After the one stop (`run_stop`), another process (the tests' lock holder,
  `tests/fixtures/job_processes`) takes the lock; a CUDA load here is then
  refused naming that holder, and loads once the holder has ended.

CUDA is faked by the shared serving-device resolver and the build is a stub
(`tests/fixtures/serving_lifecycle.py`); the lock is the real flock under the
test's own workspaces root, and each process started is stopped by its pid.

Module: tests/integration/test_serving_lock.py
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytest.importorskip("fastapi")
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the POSIX flock")

import src.api.main as api  # noqa: E402
import src.api.services.training_manager as tm  # noqa: E402
from src.api.routes.pipeline import PipelineStopRequest, run_stop  # noqa: E402
from src.utils.gpu_lock import SERVING_JOB_TYPE, WORKSPACES_ROOT_ENV, lock_state  # noqa: E402
from tests.fixtures.job_processes import REPO, start_lock_holder  # noqa: E402
from tests.fixtures.serving_lifecycle import (  # noqa: E402
    StubBuilds,
    checkpoint,
    fake_cuda,
    reload,
    served_workspace,
)


@pytest.fixture
def service(monkeypatch, tmp_path):
    """Nothing served, CUDA faked, builds stubbed, a runner of the test's own;
    afterwards whatever is served stopped and any lock still held released."""
    for name in ("pipeline", "kg", "_current_data_dir", "_current_checkpoint_path",
                 "not_serving", "serving_device", "served_sha256"):
        monkeypatch.setattr(api.app_state, name, None)
    monkeypatch.setattr(tm, "training_manager", tm.TrainingManager())
    fake_cuda(monkeypatch)
    StubBuilds().install(monkeypatch)
    yield served_workspace(tmp_path / "served")
    run_stop(PipelineStopRequest())
    if api.app_state.gpu_lock is not None:
        api.app_state.gpu_lock.release()
        api.app_state.gpu_lock = None


def test_a_training_cli_is_refused_naming_the_inference_that_holds_the_gpu(service, tmp_path):
    loaded = reload(service, checkpoint(service, "a"), "cuda")
    assert loaded.success, loaded.message
    assert lock_state()["holder"]["job_type"] == SERVING_JOB_TYPE

    cli = subprocess.run(
        [sys.executable, "scripts/train_model.py", "--data-dir", str(tmp_path / "ws")],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300,
        env={**os.environ, WORKSPACES_ROOT_ENV: os.environ[WORKSPACES_ROOT_ENV]})

    assert cli.returncode == 1, cli.stdout
    assert "train_model: refused: the GPU lock" in cli.stdout
    assert "is held by an inference job (the SHEPHERD service's CUDA inference" in cli.stdout
    assert "stop inference first with POST /api/v1/pipeline/stop" in cli.stdout
    assert not (tmp_path / "ws").exists(), "the refused CLI wrote something"
    assert api.app_state.pipeline is not None, "a job's start unloaded the model"


def test_after_a_stop_another_process_takes_the_lock_and_a_load_waits_for_it(service,
                                                                             tmp_path):
    assert reload(service, checkpoint(service, "a"), "cuda").success
    assert run_stop(PipelineStopRequest()).released
    assert lock_state()["held_by"] is None

    holder = start_lock_holder("sleep", {}, tmp_path)
    try:
        refused = reload(service, checkpoint(service, "b"), "cuda")
        assert refused.success is False
        assert "the GPU is in use by a training job (lock_holder.py" in refused.message
        assert api.app_state.pipeline is None and api.app_state.gpu_lock is None
    finally:
        holder.kill()
        holder.wait(10)

    loaded = reload(service, checkpoint(service, "b"), "cuda")
    assert loaded.success, loaded.message
    assert lock_state()["holder"]["pid"] == os.getpid()
