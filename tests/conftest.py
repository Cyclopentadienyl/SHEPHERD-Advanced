"""
Shared fixtures for every test, unit and integration.

Module: tests/conftest.py
"""
import pytest

from src.config.server_settings import (
    ENV_ACCEPT_UNAUTHENTICATED_NETWORK_BIND,
    ENV_BIND_HOST,
    ENV_DEPLOYMENT_MODE,
    ENV_REGISTRY,
)
from src.utils.gpu_lock import LOCK_FILE_ENV, WORKSPACES_ROOT_ENV


@pytest.fixture(autouse=True)
def _no_deployment_gpu_lock(monkeypatch):
    """Every test starts without the deployment's GPU lock settings.

    The deployment guide (§4.11) tells operators to load the service's
    environment in the shell where they run the CLIs; a test suite run from
    that shell would otherwise take the deployment's lock in every test that
    runs a CLI in process (failing while a real job holds it) and create the
    lock file in the real workspace root. A test that needs a lock sets these
    with `monkeypatch`, and its child processes inherit only what it set.
    """
    monkeypatch.delenv(LOCK_FILE_ENV, raising=False)
    monkeypatch.delenv(WORKSPACES_ROOT_ENV, raising=False)


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
