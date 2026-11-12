"""
Shared fixtures for the unit tests.

Module: tests/unit/conftest.py
"""
import sys
import threading

import pytest

#: App-state fields the service sets as a side effect of a request, never reset
#: by a running service, and so never undone by a test that only drove the
#: request: the not-serving record and the lifecycle's own state.
_SIDE_EFFECT_FIELDS = (
    # Written by the lifecycle when a load or the start-up is refused, or a
    # stop ends, while nothing is served.
    ("not_serving", None),
    # Set by `publish_pipeline` whenever a pipeline is published.
    ("is_ready", False),
    ("serving_device", None),
    ("served_sha256", None),
    ("transition", None),
    ("_load_holds_gpu", False),
)


@pytest.fixture(autouse=True)
def _app_state_flags_are_per_test():
    """The fields in `_SIDE_EFFECT_FIELDS` are put back after every test, the
    leases forgotten, and **a GPU lock the service still holds released**.

    In a test process they leak. A leaked not-serving record gives every later
    test that reaches `/diagnose` with no pipeline published another test's
    refusal instead of the one for nothing loaded, so a file's result depended
    on which files ran before it; a leaked `is_ready` would do the same to
    `/ready`. A held `GpuLock` would keep the flock of a test's own lock root
    and be reused by the next CUDA load, and a leaked lease would make a later
    stop wait for a request that never ends. A lifecycle lock left taken (a
    test that failed inside a load) is replaced, so the next test is not
    refused as busy.

    Lazy: a test that never imports the API does not import it here.
    """
    module = sys.modules.get("src.api.main")
    state = getattr(module, "app_state", None)
    before = {name: getattr(state, name, default) for name, default in _SIDE_EFFECT_FIELDS}
    yield
    module = sys.modules.get("src.api.main")
    if module is None:
        return
    state = module.app_state
    held = getattr(state, "gpu_lock", None)
    if held is not None:
        held.release()
    state.gpu_lock = None
    state._leases = {}
    state._retired_on_gpu = set()
    if state._lifecycle.locked():
        state._lifecycle = threading.Lock()
    for name, value in before.items():
        setattr(state, name, value)
