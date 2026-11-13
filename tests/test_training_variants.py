"""Variant semantics and exact checkpoint continuation in a small full-game env."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from experiments.ConceptCheck import train
from experiments.ConceptCheck.checkpoints import load_training_checkpoint
from experiments.ConceptCheck.validate_stability import digest_value


class ActionSpace:
    n = 2

    def seed(self, seed):
        self._np_random = np.random.default_rng(seed)

    def sample(self):
        return int(self._np_random.integers(self.n))


class LifeEnvironment:
    def __init__(self):
        self.action_space = ActionSpace()
        self.position = 0
        self.resets = 0
        self.frames = np.zeros((4, 2, 2), dtype=np.uint8)

    def reset(self, seed=None):
        self.position = 0
        self.resets += 1
        self.frames.fill(0)
        return self.frames.copy(), {"lives": 3}

    def step(self, action):
        self.position += 1
        self.frames = np.concatenate((self.frames[1:], np.full((1, 2, 2), self.position + action, dtype=np.uint8)))
        return self.frames.copy(), 7.0, self.position == 6, False, {"lives": 3 - self.position // 2}

    def get_state(self):
        return {"position": self.position, "resets": self.resets, "frames": self.frames.copy()}

    def set_state(self, state):
        self.position, self.resets = state["position"], state["resets"]
        self.frames = state["frames"].copy()

    def close(self):
        pass


class SmallNetwork(nn.Module):
    def __init__(self, in_channels=4, n_actions=2):
        super().__init__()
        self.output = nn.Linear(1, n_actions)

    def forward(self, states):
        return self.output(states.float().reshape(len(states), -1).mean(dim=1, keepdim=True) / 255.0)


class TrainingVariantTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())

    def args(self, weights, algorithm="double-dqn"):
        return ["--algorithm", algorithm, "--seed", "81327019", "--device", "cpu",
                "--torch-threads", "1", "--steps", "12", "--batch-size", "2",
                "--buffer-capacity", "12", "--min-replay-size", "2", "--train-freq", "2",
                "--target-update-freq", "5", "--save-freq", "4", "--log-freq", "2",
                "--health-freq", "0", "--save-path", str(weights)]

    def run_training(self, args):
        env = LifeEnvironment()
        with patch.object(train, "make_breakout_env", return_value=env), \
                patch.object(train, "BreakoutBasicNetwork", SmallNetwork), \
                contextlib.redirect_stdout(io.StringIO()):
            train.train(args)
        return env

    def assert_tree_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        elif isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second)
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_tree_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(first, second)

    def test_life_mask_changes_only_training_terminal_not_game_or_raw_rewards(self):
        for storage in ("deque", "frames"):
            for life_terminal in (False, True):
                with self.subTest(storage=storage, life_terminal=life_terminal), tempfile.TemporaryDirectory() as directory:
                    weights = Path(directory) / "model.pth"
                    args = self.args(weights) + ["--replay-storage", storage]
                    if life_terminal:
                        args.append("--terminal-on-life-loss")
                    env = self.run_training(args)
                    saved = load_training_checkpoint(weights.with_suffix(".training.pt"))
                    expected = [False, True, False, True, False, True] if life_terminal else [False] * 5 + [True]
                    self.assertEqual([row[4] for row in saved["replay_buffer"]], expected * 2)
                    self.assertEqual([row[2] for row in saved["replay_buffer"]], [7.0] * 12)
                    self.assertEqual(saved["episodes_completed"], 2)
                    self.assertEqual(saved["last_episode_reward"], 42.0)
                    self.assertEqual(env.resets, 3)
                    self.assertEqual(saved["previous_lives"], 3)

    def test_frame_replay_and_lives_counter_resume_exactly_for_both_algorithms(self):
        for algorithm in ("dqn", "double-dqn"):
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as directory:
                direct = Path(directory) / "direct.pth"
                resumed = Path(directory) / "resumed.pth"
                extra = ["--replay-storage", "frames", "--terminal-on-life-loss"]
                self.run_training(self.args(direct, algorithm) + extra)
                self.run_training(self.args(resumed, algorithm) + extra + ["--stop-at", "3"])
                paused = load_training_checkpoint(resumed.with_suffix(".training.pt"))
                self.assertEqual(paused["previous_lives"], 2)
                self.run_training(["--resume", str(resumed.with_suffix(".training.pt")), "--device", "cpu"])
                expected = load_training_checkpoint(direct.with_suffix(".training.pt"))
                actual = load_training_checkpoint(resumed.with_suffix(".training.pt"))
                for name in ("policy_net", "target_net", "optimizer", "state", "previous_lives",
                             "episodes_completed", "epsilon", "recent_loss", "step"):
                    self.assert_tree_equal(expected[name], actual[name])
                self.assert_tree_equal(list(expected["replay_buffer"]), list(actual["replay_buffer"]))
                self.assert_tree_equal(expected["rng"]["python"], actual["rng"]["python"])
                self.assert_tree_equal(expected["rng"]["torch"], actual["rng"]["torch"])
                self.assertEqual(digest_value(expected["replay_buffer"]), digest_value(actual["replay_buffer"]))

    def test_resume_rejects_changing_training_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model.pth"
            self.run_training(self.args(weights) + ["--replay-storage", "frames", "--terminal-on-life-loss", "--stop-at", "3"])
            for option in (["--no-terminal-on-life-loss"], ["--replay-storage", "deque"], ["--buffer-capacity", "24"], ["--epsilon-decay-steps", "1000000"]):
                with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    train.parse_args(["--resume", str(weights.with_suffix(".training.pt"))] + option)


if __name__ == "__main__":
    unittest.main()
