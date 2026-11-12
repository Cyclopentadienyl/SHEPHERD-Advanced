"""
The service's inference lifecycle, driven without a GPU or a model, for tests
=============================================================================
The lifecycle (`src.api.main.AppState`: one load or stop at a time,
admission, the GPU lock for CUDA serving) decides the lock by the device the
shared resolver gives, and publishes what the build returns. These pieces
stand in for the two parts a CPU test host cannot run:

- `fake_cuda(monkeypatch)`: the shared serving-device resolver
  (`src.inference.pipeline.resolve_serving_device`) answers "cuda" for a CUDA
  request (auto, cuda, cuda:N) as on a host with CUDA, and "cpu" for cpu.
- `StubBuilds`: `build_pipeline` replaced by a recorder that returns a built
  bundle whose served identity names the device in effect the resolver gave
  ("cuda:0" or "cpu") and the checkpoint's own digest; it can be told to fail
  or to wait, and it records what the GPU lock was doing when it ran.
- `served_workspace(root)`: a directory the reload's file check accepts, and
  `checkpoint(workspace, name)` a file whose bytes give each model its digest.

Everything else is the product's: the reload, the stop, the lease, the job
runner's slot, and a real flock under each test's own workspaces root.

Module: tests/fixtures/serving_lifecycle.py
"""
from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional


def fake_cuda(monkeypatch) -> None:
    """The shared resolver answers as on a host with CUDA."""
    import src.inference.pipeline as pipeline

    def resolve(requested: Optional[str]) -> str:
        requested = (requested or "").strip() or "auto"
        if requested == "cpu":
            return "cpu"
        return "cuda" if requested in ("auto", "cuda") else requested

    monkeypatch.setattr(pipeline, "resolve_serving_device", resolve)
    monkeypatch.delenv("SHEPHERD_DEVICE", raising=False)


def served_identity(device: str, sha256: str) -> Dict[str, Any]:
    """A coherent served identity (contract M3c) of a disease task model on
    `device`, its checkpoint `sha256`."""
    from src.core.types import (
        ServedCheckpoint,
        ServedDatasetRecord,
        ServedGraph,
        ServedIdentity,
        ServedSettings,
        ServedShortestPaths,
    )

    return ServedIdentity(
        scoring_mode="disease_task",
        checkpoint=ServedCheckpoint(state="verified", sha256=sha256),
        graph=ServedGraph(state="verified", kg_sha256="1" * 64, split_manifest_sha256="2" * 64),
        shortest_paths=ServedShortestPaths(state="absent"),
        dataset_record=ServedDatasetRecord(state="verified", origin="synthetic",
                                           sha256="3" * 64),
        settings=ServedSettings(
            device=device, device_requested=device.split(":")[0], eta=None,
            max_path_length=4, path_length_penalty=0.9, max_paths_per_candidate=10,
            use_direct_path_optimization=True, phenotype_normalisation_version=1,
        ),
    ).to_dict()


class StubPipeline:
    """A built pipeline as far as the reload, the status and the stop read it."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self.model = SimpleNamespace(spec=SimpleNamespace(encoder=SimpleNamespace(kind="gatv2")))

    def get_pipeline_config(self) -> Dict[str, Any]:
        return dict(self._config)


class StubBuilds:
    """`build_pipeline` replaced (`install`): each call recorded with the GPU
    lock's state at that moment (`lock_at_build`), answered with a bundle on
    the resolved device, or with `fail` raised, after waiting for `hold` when
    it is set (an `Event` the test sets)."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.lock_at_build: List[Optional[str]] = []
        self.fail: Optional[BaseException] = None
        self.hold: Optional[threading.Event] = None
        self.started = threading.Event()

    def install(self, monkeypatch) -> "StubBuilds":
        import src.api.main as api

        monkeypatch.setattr(api, "build_pipeline", self)
        return self

    def __call__(self, kg_path=None, checkpoint_path=None, data_dir=None, device=None,
                 expected_checkpoint_sha256=None, checkpoint_within=None):
        from src.config.serving_device import requested_serving_device
        from src.inference import pipeline
        from src.utils.gpu_lock import lock_state

        checkpoint_path = checkpoint_path or os.environ.get("SHEPHERD_CHECKPOINT_PATH")
        data_dir = data_dir or os.environ.get("SHEPHERD_DATA_DIR")
        self.calls.append({"checkpoint_path": checkpoint_path, "device": device})
        self.lock_at_build.append(lock_state()["held_by"])
        self.started.set()
        if self.hold is not None:
            assert self.hold.wait(30), "the test never let the build go on"
        if self.fail is not None:
            raise self.fail
        resolved = pipeline.resolve_serving_device(requested_serving_device(device))
        in_effect = "cuda:0" if resolved == "cuda" else resolved
        sha = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
        config = {"served_identity": served_identity(in_effect, sha), "gnn_ready": True,
                  "scoring_mode": "disease_task", "kg_nodes": 3, "kg_edges": 2,
                  "has_model": True}
        from src.api.main import PipelineBundle

        return PipelineBundle(kg=object(), pipeline=StubPipeline(config), config=config,
                              data_dir=data_dir, checkpoint_path=checkpoint_path)


def served_workspace(root: Path) -> Path:
    """A directory with every file the reload's file check requires."""
    from src.api.routes.pipeline import REQUIRED_DATA_FILES

    root.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_DATA_FILES:
        (root / name).write_bytes(b"{}")
    return root


def checkpoint(workspace: Path, name: str) -> Path:
    """A checkpoint file of its own bytes, so each model has its own digest."""
    path = workspace / f"{name}.pt"
    path.write_bytes(name.encode("utf-8"))
    return path


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reload(workspace: Path, path: Path, device: str, **extra: Any):
    """The one reload (`run_reload`), of an explicit checkpoint on `device`."""
    from src.api.routes.pipeline import PipelineReloadRequest, run_reload

    return run_reload(PipelineReloadRequest(data_dir=str(workspace), checkpoint_path=str(path),
                                            device=device, **extra))


def in_thread(fn: Callable[[], Any]) -> SimpleNamespace:
    """`fn` run in a thread of its own; `.result` once `.thread` has ended."""
    box = SimpleNamespace(result=None, error=None)

    def run() -> None:
        try:
            box.result = fn()
        except BaseException as exc:  # noqa: BLE001 - handed to the test
            box.error = exc

    box.thread = threading.Thread(target=run, daemon=True)
    box.thread.start()
    return box


__all__ = ["StubBuilds", "StubPipeline", "checkpoint", "fake_cuda", "in_thread", "reload",
           "served_identity", "served_workspace", "sha_of"]
