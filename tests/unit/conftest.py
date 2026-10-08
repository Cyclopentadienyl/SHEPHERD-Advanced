"""
Shared fixtures for the unit tests.

Module: tests/unit/conftest.py
"""
import sys

import pytest

#: App-state flags the service sets as a side effect of a request, never reset by
#: a running service, and so never undone by a test that only drove the request.
_SIDE_EFFECT_FLAGS = (
    # Set by `build_pipeline` and the reload route whenever a real pipeline is
    # asked for; a deployment that asked for one stays a deployment that did.
    ("real_pipeline_requested", False),
    # Set by `publish_pipeline` whenever a pipeline is published.
    ("is_ready", False),
)


@pytest.fixture(autouse=True)
def _app_state_flags_are_per_test():
    """The flags in `_SIDE_EFFECT_FLAGS` are put back after every test.

    In a test process they leak. A leaked `real_pipeline_requested` sends every
    later test that reaches `/diagnose` with no pipeline published to the 503
    meant for a failed deployment instead of the mock path it was written
    against, so a file's result depended on which files ran before it. A leaked
    `is_ready` would do the same to `/ready`.

    Lazy: a test that never imports the API does not import it here.
    """
    module = sys.modules.get("src.api.main")
    state = getattr(module, "app_state", None)
    before = {name: getattr(state, name, default) for name, default in _SIDE_EFFECT_FLAGS}
    yield
    module = sys.modules.get("src.api.main")
    if module is not None:
        for name, value in before.items():
            setattr(module.app_state, name, value)
