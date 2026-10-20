"""
Shared fixtures for every test, unit and integration.

Module: tests/conftest.py
"""
import pytest

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
