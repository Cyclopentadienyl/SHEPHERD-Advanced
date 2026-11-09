"""
Shared fixtures for every test, unit and integration.

Module: tests/conftest.py
"""
import sys

import pytest

from src.config.server_settings import (
    ENV_ACCEPT_UNAUTHENTICATED_NETWORK_BIND,
    ENV_BIND_HOST,
    ENV_DEPLOYMENT_MODE,
    ENV_REGISTRY,
)
from src.utils.gpu_lock import WORKSPACES_ROOT_ENV


@pytest.fixture(autouse=True, scope="session")
def _suite_gpu_lock(tmp_path_factory):
    """A GPU lock of the suite's own for fixtures wider than one test (a
    module's fixture that runs a CLI once), set before any of them runs and
    never the deployment's: the lock under a workspaces root of the suite's
    own, an empty directory. Each test then gets its own
    (`_no_deployment_gpu_lock`)."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(WORKSPACES_ROOT_ENV, str(tmp_path_factory.mktemp("suite-workspaces")))
        yield


@pytest.fixture(autouse=True)
def _no_deployment_gpu_lock(monkeypatch, tmp_path_factory):
    """Every test starts with a GPU lock of its own, never the deployment's.

    The deployment guide (§4.11) tells operators to load the service's
    environment in the shell where they run the CLIs; a test suite run from
    that shell would otherwise take the deployment's lock in every test that
    runs a CLI in process (failing while a real job holds it) and create the
    lock file in the real workspace root. Since the review of 8242b47 (P2-4)
    a formal CLI and the job runner refuse to run without a configured lock,
    and the workspaces root is the lock's one name (root-cause round,
    P24-S1), so each test is given a workspaces root of its own
    (`SHEPHERD_WORKSPACES_ROOT`, an empty directory), whose
    `.shepherd-gpu.lock` is its lock and under which the server lists no
    workspace. A test about another root, or none, sets or removes it with
    `monkeypatch`, and its child processes inherit only what it set.
    """
    monkeypatch.setenv(WORKSPACES_ROOT_ENV, str(tmp_path_factory.mktemp("workspaces")))


@pytest.fixture(autouse=True)
def _gpu_slot_open():
    """Every test starts with the job runner's GPU slot open.

    An application's shutdown closes the slot (root-cause review,
    P23-1: once the backend begins to stop, no job starts), and the runner is
    a process-wide singleton that a later test reuses, through an in-process
    client that runs no startup of its own, say. Reopened only where the
    runner's module is imported already, so nothing is imported for a test
    that does not use it.
    """
    module = sys.modules.get("src.api.services.training_manager")
    if module is not None:
        module.training_manager.open()


@pytest.fixture(autouse=True)
def _no_deployment_access_settings(monkeypatch):
    """Every test starts read-only, as an unconfigured deployment does.

    The shell the guide tells operators to prepare sets
    `SHEPHERD_DEPLOYMENT_MODE=M1`; a suite run from it would otherwise pass
    D8's check on every loopback request, and a test meant to see a refusal
    would see the control run. A test that needs mode M1, or the network-bind
    acceptance, sets it with `monkeypatch` (tests/fixtures/operator_access.py).
    The host the launcher passes to startup is cleared too: a suite run from a
    shell that holds one would have every application startup check it. It is
    set before it is removed so that the teardown removes it again whatever
    the test did: `src.api.main:main()` sets it in the process environment,
    which a plain ``delenv`` of an unset variable would not undo.
    """
    monkeypatch.delenv(ENV_DEPLOYMENT_MODE, raising=False)
    monkeypatch.delenv(ENV_ACCEPT_UNAUTHENTICATED_NETWORK_BIND, raising=False)
    monkeypatch.setenv(ENV_BIND_HOST, "")
    monkeypatch.delenv(ENV_BIND_HOST)


@pytest.fixture(scope="session")
def trained_disease_child(tmp_path_factory, _suite_gpu_lock):
    """A base and one disease child of it, trained for real on CPU once per
    session (per worker) on a demo workspace (`tests/fixtures/trained_children`):
    ``{"data_dir", "base", "disease"}``. The model every test serves where it
    once served a graph with no model, or a model supplied in memory: since K7
    a served pipeline is always a model read from its checkpoint. Read-only:
    a test that changes a file copies the workspace first."""
    from tests.fixtures.registered_cohort import write_demo_workspace
    from tests.fixtures.trained_children import train_children

    root = tmp_path_factory.mktemp("trained-disease-child")
    return train_children(write_demo_workspace(root / "workspace"), ("disease",), runs=root)


@pytest.fixture(scope="session")
def disease_pipeline(trained_disease_child):
    """The pipeline the service builds for `trained_disease_child`'s disease
    child, built once per session. Shared: a test that changes the pipeline
    builds its own (`tests.fixtures.trained_children.serve`)."""
    from tests.fixtures.trained_children import serve

    return serve(trained_disease_child)


@pytest.fixture(autouse=True)
def _no_deployment_registry(monkeypatch):
    """Every test starts with no deployment registry.

    A report is registered, and a ``--cohort`` run measures, only in the
    registry `SHEPHERD_REGISTRY` names (BACKLOG 37); a suite run from the shell
    the guide prepares (§4.11) would otherwise refuse every test's own registry
    as not the deployment's, and see "unset" nowhere. A test that registers or
    measures names its own registry with `monkeypatch` (or `environ=`), and its
    child processes inherit only what it set."""
    monkeypatch.delenv(ENV_REGISTRY, raising=False)
