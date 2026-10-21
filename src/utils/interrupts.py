"""
A cancel held for the few steps that must happen together.
==========================================================
The job runner cancels a test run with SIGINT (`TrainingManager`), and the
measurement and the registration each end with steps that belong together:
publishing a report, then appending it to its cohort version's usage history
and writing the run's outcome file; writing a ledger record, then its
"registered" history entry. A SIGINT landing between two of them left a kept
or registered result out of the usage history D4 and D7 require (M5-3 review
EVAL-1). Inside `interrupt_deferred` a SIGINT is held, and raised again as soon
as the block ends: the cancel still ends the run, after the steps, not between
them.

**Held in Python, in the main thread.** Python runs every signal handler in the
main thread, whichever thread the operating system delivered the signal to (a
signal mask would hold it only in threads that set it, and torch starts threads
of its own), so the block puts a handler in place that notes the SIGINT, puts
the previous handler back at its end, and then raises the signal again for it:
`default_int_handler`'s KeyboardInterrupt, or the default action. Outside the
main thread (a server's worker registering a report) a handler cannot be set,
and nothing is held: there, SIGINT is the server's own shutdown. A SIGTERM or
SIGKILL (the runner's later escalation) and a power loss are not held: the
runner says what the run left either way.

Module: src/utils/interrupts.py
"""
from __future__ import annotations

import contextlib
import signal
import threading
from typing import Iterator, List


@contextlib.contextmanager
def interrupt_deferred() -> Iterator[None]:
    """SIGINT held until the block ends, then raised again for the handler in
    place before it (main thread only; elsewhere a no-op). Nested blocks hold
    it until the outermost one ends."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    held: List[int] = []

    def hold(signum: int, frame: object) -> None:
        held.append(signum)

    previous = signal.signal(signal.SIGINT, hold)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
        if held:
            signal.raise_signal(signal.SIGINT)


__all__ = ["interrupt_deferred"]
