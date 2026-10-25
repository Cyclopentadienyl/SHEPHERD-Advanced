"""
A cancel held for the steps that must happen together (M5-3 review EVAL-1).
==========================================================================
`src/utils/interrupts.py`: inside `interrupt_deferred` a SIGINT is held and
raised again, for the handler in place before, when the block ends; nested
blocks hold it until the outermost ends; outside the main thread nothing is
held or changed.

Module: tests/unit/test_interrupts.py
"""
import os
import signal
import sys
import threading
import time

import pytest

from src.utils.interrupts import interrupt_deferred

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")


def _interrupt_now():
    os.kill(os.getpid(), signal.SIGINT)
    time.sleep(0.2)          # delivered here unless held


def test_a_sigint_inside_is_held_then_raised_when_the_block_ends():
    done = []
    before = signal.getsignal(signal.SIGINT)
    with pytest.raises(KeyboardInterrupt):
        with interrupt_deferred():
            _interrupt_now()
            done.append("after the signal")
    assert done == ["after the signal"]
    assert signal.getsignal(signal.SIGINT) is before


def test_without_a_sigint_the_block_ends_quietly_and_restores_the_handler():
    before = signal.getsignal(signal.SIGINT)
    with interrupt_deferred():
        pass
    assert signal.getsignal(signal.SIGINT) is before


def test_nested_blocks_hold_it_until_the_outermost_ends():
    steps = []
    with pytest.raises(KeyboardInterrupt):
        with interrupt_deferred():
            with interrupt_deferred():
                _interrupt_now()
                steps.append("inner")
            steps.append("outer")
    assert steps == ["inner", "outer"]


def test_outside_the_main_thread_nothing_is_changed():
    seen = []

    def work():
        with interrupt_deferred():
            seen.append(signal.getsignal(signal.SIGINT))

    before = signal.getsignal(signal.SIGINT)
    thread = threading.Thread(target=work)
    thread.start()
    thread.join()
    assert seen == [before]
