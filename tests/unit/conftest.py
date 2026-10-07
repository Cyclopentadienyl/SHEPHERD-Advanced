"""
Shared fixtures for the unit tests.

Module: tests/unit/conftest.py
"""
import sys

import pytest


@pytest.fixture(autouse=True)
def _real_pipeline_request_is_per_test():
    """`app_state.real_pipeline_requested` is put back after every test.

    `build_pipeline` and the reload route set it as a side effect of being asked
    for a real pipeline, and a running service never clears it: a deployment
    that asked for one stays a deployment that did. In a test process that
    leaks. Every later test that reaches `/diagnose` with no pipeline published
    gets the 503 meant for a failed deployment instead of the mock path it was
    written against, so a file's result depended on which files ran before it.

    Lazy: a test that never imports the API does not import it here.
    """
    module = sys.modules.get("src.api.main")
    before = getattr(getattr(module, "app_state", None), "real_pipeline_requested", False)
    yield
    module = sys.modules.get("src.api.main")
    if module is not None:
        module.app_state.real_pipeline_requested = before
