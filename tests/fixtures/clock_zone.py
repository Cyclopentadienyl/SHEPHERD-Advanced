"""
The process's local time zone, set for one test (POSIX).
========================================================
The dashboard shows every clock time in UTC (`renderers.clock_utc`), converting
the server's local stamps, so a test that writes local stamps by hand states
the zone they are in: ``with local_zone("Asia/Taipei"): ...`` sets ``TZ``,
calls `time.tzset`, and puts the zone back afterwards.

Module: tests/fixtures/clock_zone.py
"""
import contextlib
import os
import time
from typing import Iterator

import pytest


@contextlib.contextmanager
def local_zone(name: str) -> Iterator[None]:
    """`name` as this process's local time zone inside the block."""
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is POSIX only")
    before = os.environ.get("TZ")
    os.environ["TZ"] = name
    time.tzset()
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = before
        time.tzset()


@pytest.fixture
def utc_clock():
    """The local zone UTC for the test, as its hand-written stamps assume."""
    with local_zone("UTC"):
        yield


__all__ = ["local_zone", "utc_clock"]
