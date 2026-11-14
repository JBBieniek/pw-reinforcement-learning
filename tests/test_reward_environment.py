"""Native ALE reward tracking preserves pixels, scores and checkpoint state."""

from collections import deque
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.ConceptCheck.checkpoints import (
    _capture_environment, _restore_environment,
    load_training_checkpoint, restore_training_checkpoint, save_training_checkpoint,
)
from experiments.ConceptCheck.neural_network import make_breakout_env
from experiments.ConceptCheck.rewards import PaddleBounceDetector, RallyRewardTracker


def tracker(env):
    while not isinstance(env, RallyRewardTracker):
        env = env.env
    return env


class RewardEnvironmentTests(unittest.TestCase):
    def test_tracker_preserves_observations_and_game_rewards_at_all_frame_skips(self):
        for skip in (2, 4, 8):
            with self.subTest(frame_skip=skip):
                plain = make_breakout_env(frame_skip=skip, sticky_action_probability=0.25)
                shaped = make_breakout_env(frame_skip=skip, sticky_action_probability=0.25,
                                          reward_shaping="rally")
                try:
                    left, a = plain.reset(seed=913783)
                    right, b = shaped.reset(seed=913783)
                    np.testing.assert_array_equal(left, right)
                    for action in [1, 2, 0, 3, 0, 2, 1] * 40:
                        left, reward_a, term_a, trunc_a, a = plain.step(action)
                        right, reward_b, term_b, trunc_b, b = shaped.step(action)
                        np.testing.assert_array_equal(left, right)
                        self.assertEqual((reward_a, term_a, trunc_a, a["lives"]),
                                         (reward_b, term_b, trunc_b, b["lives"]))
                        self.assertIn("rally_reward", b)
                        if term_a or trunc_a:
                            left, _ = plain.reset()
                            right, _ = shaped.reset()
                            np.testing.assert_array_equal(left, right)
                finally:
                    plain.close()
                    shaped.close()

    def test_real_contact_and_pending_rally_survive_full_checkpoint_restore(self):
        env = make_breakout_env(sticky_action_probability=0.0, reward_shaping="rally")
        restored_env = make_breakout_env(sticky_action_probability=0.0, reward_shaping="rally")
        try:
            observation, _ = env.reset(seed=317891)
            action = 1
            for _ in range(2500):
                observation, _, terminated, truncated, info = env.step(action)
                if tracker(env).bounces_total:
                    break
                if terminated or truncated:
                    observation, _ = env.reset()
                    action = 1
                    continue
                ball, paddle = PaddleBounceDetector.locate(env.unwrapped.ale.getScreenRGB())
                if ball is None or paddle is None:
                    action = 1
                else:
                    dx = (ball[0] + ball[2] - paddle[0] - paddle[2]) / 2
                    action = 2 if dx > 2 else 3 if dx < -2 else 0
            self.assertGreater(tracker(env).bounces_total, 0, "must exercise a real paddle bounce")
            saved_tracker = deepcopy(tracker(env).get_state())
            policy = torch.nn.Linear(1, 2)
            target = torch.nn.Linear(1, 2)
            optimizer = torch.optim.Adam(policy.parameters())
            totals = tracker(env).snapshot()
            counters = dict(episode_bonus=totals["bonus_total"], last_episode_bonus=0.0,
                            episode_bounces=totals["bounces_total"], last_episode_bounces=0,
                            total_bonus=totals["bonus_total"], total_bounces=totals["bounces_total"],
                            tracker_bonus_total=totals["bonus_total"],
                            tracker_bounces_total=totals["bounces_total"])
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.training.pt"
                save_training_checkpoint(
                    path, args={"algorithm": "double-dqn", "reward_shaping": "rally"},
                    policy_net=policy, target_net=target, optimizer=optimizer,
                    replay_buffer=deque(maxlen=8), state=np.array(observation), step=73,
                    epsilon=0.1, episode_reward=0.0, last_episode_reward=0.0,
                    episodes_completed=0, recent_loss=0.0, env=env,
                    previous_lives=info["lives"], reward_state=counters,
                )
                loaded = load_training_checkpoint(path)
                self.assertEqual(loaded["environment"]["kind"], "ale")
                restored_env.reset(seed=77417)
                result = restore_training_checkpoint(
                    loaded, policy_net=policy, target_net=target, optimizer=optimizer,
                    env=restored_env, device="cpu",
                )
                self.assertTrue(result["environment_restore_exact"])
                self.assertEqual(saved_tracker, tracker(restored_env).get_state())
                for action in [0, 2, 0, 3, 1] * 60:
                    left, reward_a, term_a, trunc_a, a = env.step(action)
                    right, reward_b, term_b, trunc_b, b = restored_env.step(action)
                    np.testing.assert_array_equal(left, right)
                    self.assertEqual((reward_a, term_a, trunc_a, a),
                                     (reward_b, term_b, trunc_b, b))
                    if term_a or trunc_a:
                        env.reset()
                        restored_env.reset()
        finally:
            env.close()
            restored_env.close()

    def test_unvalidated_modes_fail_before_constructing_environment(self):
        with self.assertRaisesRegex(ValueError, "selected modes"):
            make_breakout_env(mode=8, reward_shaping="rally")

    def test_bare_tracker_protocol_also_captures_the_underlying_emulator(self):
        import gymnasium as gym

        env = RallyRewardTracker(gym.make("ALE/Breakout-v5", frameskip=1,
                                          repeat_action_probability=0.0))
        try:
            env.reset(seed=771393)
            for action in [1, 2, 3, 0] * 10:
                env.step(action)
            saved = _capture_environment(env)
            self.assertEqual(saved["kind"], "ale")
            expected = env.step(2)
            self.assertTrue(_restore_environment(env, saved))
            actual = env.step(2)
            np.testing.assert_array_equal(expected[0], actual[0])
            self.assertEqual(expected[1:], actual[1:])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
