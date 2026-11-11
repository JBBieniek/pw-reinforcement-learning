"""Rate-limit behavior without real sleeps or hardware dependencies."""

import unittest

from experiments.ConceptCheck.pacing import StepPacer


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []
        self.oversleep = 0.0

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds + self.oversleep


class StepPacerTests(unittest.TestCase):
    def test_unlimited_does_not_read_clock_or_sleep(self):
        def unexpected(*args):
            self.fail("Unlimited pacing must not call clock or sleep")

        pacer = StepPacer(0, clock=unexpected, sleep=unexpected)
        self.assertEqual(pacer.wait(), 0)

    def test_work_time_counts_toward_rate_limit(self):
        clock = FakeClock()
        pacer = StepPacer(100, clock=clock.clock, sleep=clock.sleep)
        clock.now = 0.003
        self.assertAlmostEqual(pacer.wait(), 0.007)
        clock.now += 0.004
        self.assertAlmostEqual(pacer.wait(), 0.006)
        self.assertAlmostEqual(clock.now, 0.020)

    def test_checkpoint_stall_does_not_accumulate_catch_up_burst(self):
        clock = FakeClock()
        pacer = StepPacer(100, clock=clock.clock, sleep=clock.sleep)
        pacer.wait()
        clock.now += 20.0  # Expensive checkpoint or a paused desktop process.
        self.assertEqual(pacer.wait(), 0.0)
        clock.now += 0.001
        self.assertAlmostEqual(pacer.wait(), 0.009)
        clock.now += 0.001
        self.assertAlmostEqual(pacer.wait(), 0.009)

    def test_oversleep_does_not_create_catch_up_slots(self):
        clock = FakeClock()
        clock.oversleep = 0.050
        pacer = StepPacer(100, clock=clock.clock, sleep=clock.sleep)
        pacer.wait()
        clock.oversleep = 0.0
        clock.now += 0.002
        self.assertAlmostEqual(pacer.wait(), 0.008)

    def test_slow_work_does_not_add_unnecessary_sleep(self):
        clock = FakeClock()
        pacer = StepPacer(100, clock=clock.clock, sleep=clock.sleep)
        for _ in range(4):
            clock.now += 0.020
            self.assertEqual(pacer.wait(), 0.0)
        self.assertEqual(clock.sleeps, [])

    def test_invalid_rates_are_rejected(self):
        for rate in (-1, float("nan"), float("inf"), float("-inf")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                StepPacer(rate)


if __name__ == "__main__":
    unittest.main()
