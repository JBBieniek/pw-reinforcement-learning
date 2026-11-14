"""Native-frame collision detection and bounded auxiliary reward semantics."""

import json
import unittest

import gymnasium as gym
import numpy as np

from experiments.ConceptCheck.rewards import PaddleBounceDetector, RallyRewardTracker


def frame(ball=None, paddle_x=70, paddle_width=16):
    image = np.zeros((210, 160, 3), dtype=np.uint8)
    image[189:193, paddle_x:paddle_x + paddle_width] = (200, 72, 72)
    if ball is not None:
        x, y = ball
        image[y:y + 4, x:x + 2] = (200, 72, 72)
    return image


class Frames(gym.Env):
    def __init__(self, events):
        self.events = events
        self.position = 0
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(0, 255, (210, 160, 3), np.uint8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = 0
        return frame(), {"lives": 5}

    def step(self, action):
        image, reward, lives, terminal, truncated = self.events[self.position]
        self.position += 1
        return image, reward, terminal, truncated, {"lives": lives}


def event(y=None, reward=0.0, lives=5, terminal=False, truncated=False, x=76):
    return frame(None if y is None else (x, y)), reward, lives, terminal, truncated


CONTACT = [event(y) for y in (179, 182, 185, 187, 185)]


class BounceDetectorTests(unittest.TestCase):
    def test_true_contact_counts_once_including_upward_frames(self):
        detector = PaddleBounceDetector()
        detections = [detector.observe(frame((76, y))) for y in
                      (179, 182, 185, 187, 185, 182, 179, 176)]
        self.assertEqual(detections, [False, False, False, False, True, False, False, False])

    def test_contact_can_be_hidden_in_paddle_for_a_few_native_frames(self):
        detector = PaddleBounceDetector()
        frames = [frame((76, y)) for y in (179, 182, 185, 187, 189, 190, 189, 187, 185, 182)]
        self.assertEqual(sum(detector.observe(image) for image in frames), 1)

    def test_brick_wall_nonoverlap_and_serve_are_not_paddle_contacts(self):
        cases = [
            [(76, y) for y in (100, 103, 106, 103, 100)],
            [(x, y) for x, y in ((10, 178), (9, 181), (10, 184), (12, 187))],
            [(20, y) for y in (179, 182, 185, 187, 185, 182)],
            [(76, y) for y in (187, 185, 182, 179)],
        ]
        for path in cases:
            with self.subTest(path=path):
                detector = PaddleBounceDetector()
                self.assertEqual(sum(detector.observe(frame(ball)) for ball in path), 0)

    def test_shrunken_paddle_is_measured_and_false_ball_in_paddle_row_is_separate(self):
        for width in (8, 12, 16):
            with self.subTest(width=width):
                detector = PaddleBounceDetector()
                images = [frame((73, y), paddle_width=width) for y in (179, 182, 185, 187, 185)]
                self.assertEqual(sum(detector.observe(image) for image in images), 1)
        _, paddle = PaddleBounceDetector.locate(frame((20, 189), paddle_width=8))
        self.assertEqual(paddle, (70, 189, 77, 192))

    def test_serialized_detector_resumes_pending_contact_exactly(self):
        detector = PaddleBounceDetector()
        for y in (179, 182, 185, 187):
            detector.observe(frame((76, y)))
        restored = PaddleBounceDetector()
        restored.set_state(json.loads(json.dumps(detector.get_state())))
        for y in (185, 182, 178):
            self.assertEqual(detector.observe(frame((76, y))), restored.observe(frame((76, y))))
        self.assertEqual(detector.get_state(), restored.get_state())

    def test_wrong_frame_format_fails_instead_of_silently_dropping_bonuses(self):
        with self.assertRaisesRegex(ValueError, "native 210x160"):
            PaddleBounceDetector().observe(np.zeros((84, 84), np.uint8))


class RallyRewardTests(unittest.TestCase):
    def test_raw_rewards_unchanged_incremental_bonus_capped_per_rally(self):
        events = CONTACT + [event(181, 1), event(177, 7), event(173, 100)]
        tracker = RallyRewardTracker(Frames(events))
        _, reset_info = tracker.reset()
        self.assertEqual(reset_info["rally_reward"]["bonus_total"], 0)
        snapshots = []
        for original in events:
            image, reward, terminal, truncated, info = tracker.step(0)
            np.testing.assert_array_equal(image, original[0])
            self.assertEqual((reward, terminal, truncated), (original[1], original[3], original[4]))
            snapshots.append(info["rally_reward"])
        self.assertEqual(snapshots[4]["bonus_total"], 1)
        self.assertAlmostEqual(snapshots[5]["bonus_total"], 1.2)
        self.assertEqual(snapshots[6]["bonus_total"], 2)
        self.assertEqual(snapshots[7]["bonus_total"], 2)
        self.assertEqual(snapshots[7]["raw_reward_total"], 108)
        self.assertEqual(snapshots[7]["bounces_total"], 1)

    def test_life_loss_and_full_reset_clear_rally_but_keep_totals(self):
        events = CONTACT + [event(181, 1), event(lives=4), event(120, 7, lives=4)]
        tracker = RallyRewardTracker(Frames(events))
        tracker.reset()
        for _ in events:
            tracker.step(0)
        self.assertAlmostEqual(tracker.bonus_total, 1.2)
        self.assertEqual(tracker.rally_points, 0)
        self.assertFalse(tracker.rally_active)
        _, info = tracker.reset()
        self.assertAlmostEqual(info["rally_reward"]["bonus_total"], 1.2)
        self.assertEqual(info["rally_reward"]["bounces_total"], 1)
        self.assertFalse(info["rally_reward"]["rally_active"])

    def test_new_contact_starts_a_new_bonus_budget(self):
        events = CONTACT + [event(y) for y in (181, 177, 173, 165, 160, 165, 173, 180, 184, 187, 185)]
        events[-1] = event(185, 7)
        tracker = RallyRewardTracker(Frames(events))
        tracker.reset()
        for _ in events:
            tracker.step(0)
        self.assertEqual(tracker.bounces_total, 2)
        self.assertEqual(tracker.bonus_total, 3)
        self.assertEqual(tracker.rally_points, 7)

    def test_points_before_first_contact_do_not_create_a_rally(self):
        tracker = RallyRewardTracker(Frames([event(120, 7)]))
        tracker.reset()
        _, reward, _, _, info = tracker.step(0)
        self.assertEqual(reward, 7)
        self.assertEqual(info["rally_reward"]["bonus_total"], 0)

    def test_tracker_roundtrip_preserves_cap_and_pending_detector_state(self):
        events = CONTACT + [event(181, 1), event(177, 7), event(173, 100)]
        original = RallyRewardTracker(Frames(events))
        original.reset()
        for _ in events[:4]:
            original.step(0)
        saved = json.loads(json.dumps(original.get_state()))
        restored = RallyRewardTracker(Frames(events))
        restored.reset()
        restored.env.position = original.env.position
        restored.set_state(saved)
        for _ in events[4:]:
            first, second = original.step(0), restored.step(0)
            self.assertEqual(first[1:], second[1:])
        self.assertEqual(original.get_state(), restored.get_state())
        with self.assertRaisesRegex(ValueError, "mismatch"):
            RallyRewardTracker(Frames(events), bounce_bonus=2).set_state(saved)

    def test_terminal_and_truncation_clear_the_rally_after_awarding_its_points(self):
        for terminal, truncated in ((True, False), (False, True)):
            with self.subTest(terminal=terminal):
                events = CONTACT + [event(181, 7, terminal=terminal, truncated=truncated)]
                tracker = RallyRewardTracker(Frames(events))
                tracker.reset()
                for _ in events:
                    tracker.step(0)
                self.assertEqual(tracker.bonus_total, 2)
                self.assertFalse(tracker.rally_active)

    def test_configuration_rejects_invalid_reward_scales(self):
        for kwargs in ({"bounce_bonus": -1}, {"bounce_bonus": True},
                       {"rally_bonus_step": float("nan")},
                       {"rally_bonus_max_multiplier": 0.5},
                       {"bounce_bonus": 1e308, "rally_bonus_max_multiplier": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RallyRewardTracker(Frames([]), **kwargs)


if __name__ == "__main__":
    unittest.main()
