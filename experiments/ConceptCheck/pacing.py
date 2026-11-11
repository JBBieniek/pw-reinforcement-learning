"""Soft wall-clock rate limit for training and evaluation decisions."""

import math
import time


class StepPacer:
    """Limit completed decisions per second without accumulating catch-up work.

    Call ``wait()`` after each completed environment decision. This gives other
    applications regular gaps between our GPU work, but does not reserve GPU
    compute or guarantee a particular utilization percentage. A rate of zero
    disables pacing. Sleeping may take longer than requested on a busy desktop.
    """

    def __init__(self, max_steps_per_second=0.0, *, clock=time.perf_counter,
                 sleep=time.sleep):
        rate = float(max_steps_per_second)
        if not math.isfinite(rate) or rate < 0:
            raise ValueError("max_steps_per_second must be finite and non-negative")
        self._clock = clock
        self._sleep = sleep
        self._interval = 1.0 / rate if rate else 0.0
        self._next_deadline = clock() + self._interval if rate else None

    def wait(self):
        """Wait for the next decision slot and return requested sleep seconds."""
        if not self._interval:
            return 0.0
        now = self._clock()
        delay = max(0.0, self._next_deadline - now)
        if delay:
            self._sleep(delay)
            now = self._clock()
        # Reset from actual time after a stall or oversleep. Old unused slots
        # must not create a burst after checkpointing, evaluation or a pause.
        self._next_deadline = max(self._next_deadline, now) + self._interval
        return delay
