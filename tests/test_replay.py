"""Storage equivalence, boundaries, RNG and checkpoint tests for frame replay."""

from collections import deque
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from experiments.ConceptCheck.replay import FrameReplayBuffer


def observation(values, shape=(3, 5)):
    return np.stack([np.full(shape, value, dtype=np.uint8) for value in values])


class FrameReplayTests(unittest.TestCase):
    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)

    def assertTransitionEqual(self, actual, expected):
        np.testing.assert_array_equal(actual[0], expected[0])
        self.assertEqual(actual[1:3], expected[1:3])
        np.testing.assert_array_equal(actual[3], expected[3])
        self.assertEqual(actual[4], expected[4])

    def test_wraparound_and_reset_boundaries_equal_full_transition_deque(self):
        replay = FrameReplayBuffer(7)
        reference = deque(maxlen=7)
        state = observation([1, 1, 1, 1])
        for index in range(34):
            # A full-game reset is discontinuous. Life loss still has a normal
            # shifted observation, while a truncation need not mask bootstrap.
            if index in (8, 17, 26):
                state = observation([index + 1] * 4)
            following = np.concatenate((state[1:], observation([index + 30])))
            terminal = index in (7, 16, 25) or index % 5 == 0
            transition = (state, index % 4, float(index % 8), following, terminal)
            replay.append(transition)
            reference.append(transition)
            self.assertEqual(len(replay), len(reference))
            self.assertEqual(replay.maxlen, 7)
            for actual, expected in zip(replay, reference):
                self.assertTransitionEqual(actual, expected)
            self.assertTransitionEqual(replay[-1], reference[-1])
            state = following
        for actual, expected in zip(replay[-4::2], list(reference)[-4::2]):
            self.assertTransitionEqual(actual, expected)

    def test_arbitrary_nonoverlapping_observations_and_single_frame_stacks(self):
        for stack_size in (1, 4):
            with self.subTest(stack_size=stack_size):
                replay = FrameReplayBuffer(3)
                for index in range(9):
                    state = observation(range(index, index + stack_size))
                    following = observation(range(index + 50, index + stack_size + 50))
                    transition = (state, index, 0.5, following, False)
                    replay.append(transition)
                    self.assertTransitionEqual(replay[-1], transition)

    def test_mutating_inputs_or_sample_outputs_does_not_change_stored_frames(self):
        replay = FrameReplayBuffer(5)
        state = observation([1, 2, 3, 4])
        following = observation([2, 3, 4, 5])
        expected = (state.copy(), 2, 7.0, following.copy(), False)
        replay.append((state, 2, 7.0, following, False))
        state[:] = 81
        following[:] = 82
        sampled = replay.sample(1)[0]
        sampled[0][:] = 90
        sampled[3][:] = 91
        self.assertTransitionEqual(replay[0], expected)
        # Reusing a mutated input object must not incorrectly reuse the cache.
        next_expected = (following.copy(), 1, 1.0, observation([83] * 4), True)
        replay.append(next_expected)
        self.assertTransitionEqual(replay[1], next_expected)
        self.assertTransitionEqual(replay[0], expected)

    def test_uniform_sampling_matches_random_sample_indices_and_preserves_rng(self):
        replay = FrameReplayBuffer(40)
        for index in range(60):
            replay.append((observation([index] * 4), index, 0.0,
                           observation([index + 1] * 4), False))
        random.seed(87301529)
        saved_rng = random.getstate()
        expected_indices = random.sample(range(len(replay)), 13)
        expected_rng = random.getstate()
        random.setstate(saved_rng)
        sample = replay.sample(13)
        self.assertEqual([item[1] for item in sample], [index + 20 for index in expected_indices])
        self.assertEqual(random.getstate(), expected_rng)
        self.assertEqual(len({item[1] for item in sample}), 13)
        self.assertEqual(replay.sample(0), [])

    def test_sequential_append_uses_one_new_frame_and_frees_evicted_frames(self):
        replay = FrameReplayBuffer(100)
        state = observation([0, 1, 2, 3], shape=(84, 84))
        for index in range(200):
            following = np.concatenate((state[1:], observation([index + 4], shape=(84, 84))))
            replay.append((state, index % 4, 0.0, following, False))
            state = following
        # 100 retained transitions cover 104 distinct frames, including context.
        self.assertEqual(replay.stored_frame_bytes, 104 * 84 * 84)
        naive_array_bytes = 2 * 100 * 4 * 84 * 84
        self.assertLess(replay.stored_frame_bytes, naive_array_bytes / 7)

    def test_torch_roundtrip_preserves_shared_storage_sampling_and_future_appends(self):
        replay = FrameReplayBuffer(11)
        state = observation([0, 1, 2, 3])
        for index in range(18):
            following = np.concatenate((state[1:], observation([index + 4])))
            replay.append((state, index % 4, float(index % 3), following, index == 9))
            state = following
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.pt"
            torch.save(replay, path, pickle_protocol=5)
            loaded = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(loaded.stored_frame_bytes, replay.stored_frame_bytes)
        self.assertEqual(loaded.capacity, replay.capacity)
        for actual, expected in zip(loaded, replay):
            self.assertTransitionEqual(actual, expected)
        random.seed(613509)
        rng_state = random.getstate()
        expected = replay.sample(8)
        random.setstate(rng_state)
        for actual, transition in zip(loaded.sample(8), expected):
            self.assertTransitionEqual(actual, transition)
        transition = (state, 3, 4.0, observation([19, 20, 21, 22]), True)
        replay.append(transition)
        loaded.append(transition)
        for actual, expected in zip(loaded, replay):
            self.assertTransitionEqual(actual, expected)
        self.assertEqual(loaded.stored_frame_bytes, replay.stored_frame_bytes)

    def test_invalid_append_is_atomic_and_does_not_draw_rng(self):
        replay = FrameReplayBuffer(2)
        valid = (observation([1, 2, 3, 4]), 0, 1.0, observation([2, 3, 4, 5]), False)
        replay.append(valid)
        initial_bytes = replay.stored_frame_bytes
        rng_state = random.getstate()
        invalid = [
            (), valid[:4],
            (valid[0].astype(np.float32), *valid[1:]),
            (valid[0], 0, 1.0, observation([2, 3]), False),
            (np.array([], dtype=np.uint8), 0, 1.0, np.array([], dtype=np.uint8), False),
            (valid[0], -1, 1.0, valid[3], False),
            (valid[0], 0.5, 1.0, valid[3], False),
            (valid[0], True, 1.0, valid[3], False),
            (valid[0], 0, float("nan"), valid[3], False),
            (valid[0], 0, "1.0", valid[3], False),
            (valid[0], 0, 1.0, valid[3], 1),
            (observation([1, 2, 3, 4], shape=(2, 2)), 0, 1.0,
             observation([2, 3, 4, 5], shape=(2, 2)), False),
        ]
        for transition in invalid:
            with self.subTest(transition_fields=len(transition)):
                with self.assertRaises(ValueError):
                    replay.append(transition)
                self.assertEqual(len(replay), 1)
                self.assertEqual(replay.stored_frame_bytes, initial_bytes)
                self.assertTransitionEqual(replay[0], valid)
                self.assertEqual(random.getstate(), rng_state)

    def test_invalid_capacity_sample_and_index_are_rejected(self):
        for capacity in (0, -1, 1.5, True, "5"):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                FrameReplayBuffer(capacity)
        replay = FrameReplayBuffer(1)
        self.assertEqual(list(replay), [])
        for batch_size in (-1, 1, 0.5, True):
            with self.subTest(batch_size=batch_size), self.assertRaises(ValueError):
                replay.sample(batch_size)
        for index in (0, -1):
            with self.subTest(index=index), self.assertRaises(IndexError):
                _ = replay[index]


if __name__ == "__main__":
    unittest.main()
