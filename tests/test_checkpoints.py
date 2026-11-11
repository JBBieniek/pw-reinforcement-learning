"""Snapshot integrity and exact ALE continuation, including sticky actions."""

from collections import deque
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.ConceptCheck import checkpoints


class ProtocolEnvironment:
    def __init__(self):
        self.position = 3

    def get_state(self):
        return {"position": self.position}

    def set_state(self, state):
        self.position = state["position"]


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.rng = checkpoints._capture_rng()
        self.addCleanup(checkpoints._restore_rng, self.rng)
        self.policy = torch.nn.Linear(1, 2)
        self.target = torch.nn.Linear(1, 2)
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.policy.parameters())
        self.policy(torch.ones(1, 1)).sum().backward()
        self.optimizer.step()

    def save(self, path, env=None, state=None, replay_buffer=None):
        checkpoints.save_training_checkpoint(
            path, args=SimpleNamespace(algorithm="double-dqn", seed=374795),
            policy_net=self.policy, target_net=self.target, optimizer=self.optimizer,
            replay_buffer=deque(maxlen=8) if replay_buffer is None else replay_buffer,
            state=np.array([3], dtype=np.uint8) if state is None else state,
            step=21, epsilon=0.2, episode_reward=2.0, last_episode_reward=4.0,
            episodes_completed=3, recent_loss=0.1,
            env=ProtocolEnvironment() if env is None else env,
            elapsed_seconds=1.25,
        )

    def test_protocol_resume_preserves_rng_replay_alias_and_optimizer(self):
        env = ProtocolEnvironment()
        one = np.array([1], dtype=np.uint8)
        two = np.array([2], dtype=np.uint8)
        replay = deque([(one, 0, 1.0, two, False), (two, 1, 0.0, one, False)], maxlen=8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.training.pt"
            self.save(path, env=env, replay_buffer=replay)
            expected = (random.random(), np.random.random(), torch.rand(3))
            env.position = 999
            loaded = checkpoints.load_training_checkpoint(path)
            restored = checkpoints.restore_training_checkpoint(
                loaded, policy_net=self.policy, target_net=self.target,
                optimizer=self.optimizer, env=env, device="cpu",
            )
            self.assertTrue(restored["environment_restored"])
            self.assertEqual(env.position, 3)
            self.assertEqual(restored["step"], 21)
            self.assertEqual(restored["episode_reward"], 2.0)
            self.assertEqual(restored["elapsed_seconds"], 1.25)
            self.assertEqual(len(self.optimizer.state), 2)
            self.assertIs(restored["replay_buffer"][0][3], restored["replay_buffer"][1][0])
            self.assertEqual(restored["replay_buffer"].maxlen, 8)
            self.assertEqual(random.random(), expected[0])
            self.assertEqual(np.random.random(), expected[1])
            torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)

    def test_atomic_failure_keeps_previous_snapshot_and_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.training.pt"
            self.save(path)
            original = path.read_bytes()
            with patch.object(checkpoints.torch, "save", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    self.save(path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_weights_only_and_unknown_version_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pth"
            torch.save(self.policy.state_dict(), path)
            with self.assertRaisesRegex(ValueError, "full training checkpoint"):
                checkpoints.load_training_checkpoint(path)
            torch.save({"format_version": 987123}, path)
            with self.assertRaisesRegex(ValueError, "version"):
                checkpoints.load_training_checkpoint(path)

    def test_real_ale_future_matches_after_restore_in_new_emulator(self):
        import gymnasium as gym
        from experiments.ConceptCheck.neural_network import make_breakout_env

        def make():
            return gym.wrappers.TimeLimit(make_breakout_env(), max_episode_steps=97)

        def future(env):
            result = []
            for _ in range(160):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                result.append((action, observation.copy(), reward, terminated,
                               truncated, dict(info), random.random(),
                               np.random.random(), torch.rand(1)))
                if terminated or truncated:
                    env.reset()
            return result

        original = make()
        self.addCleanup(original.close)
        original.action_space.seed(977347)
        observation, _ = original.reset(seed=574037)
        for action in [1, 1, 2, 3, 0, 1, 2] * 3:
            observation, *_ = original.step(action)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.training.pt"
            self.save(path, env=original, state=observation)
            expected = future(original)
            self.assertTrue(any(row[4] for row in expected), "Must exercise time-limit reset")
            restored_env = make()
            self.addCleanup(restored_env.close)
            restored_env.reset(seed=729487)
            restored_env.action_space.seed(999)
            restored = checkpoints.restore_training_checkpoint(
                checkpoints.load_training_checkpoint(path), policy_net=self.policy,
                target_net=self.target, optimizer=self.optimizer,
                env=restored_env, device="cpu",
            )
            self.assertTrue(restored["environment_restored"])
            np.testing.assert_array_equal(restored["state"], observation)
            actual = future(restored_env)
            for step, (left, right) in enumerate(zip(expected, actual)):
                with self.subTest(step=step):
                    self.assertEqual(left[0], right[0])
                    np.testing.assert_array_equal(left[1], right[1])
                    self.assertEqual(left[2:8], right[2:8])
                    torch.testing.assert_close(left[8], right[8], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
