"""Training contracts for raw scores, bounded bonuses and checkpoint resumes."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from experiments.ConceptCheck import train
from experiments.ConceptCheck.checkpoints import load_training_checkpoint


class ActionSpace:
    n = 2

    def seed(self, seed):
        self._np_random = np.random.default_rng(seed)

    def sample(self):
        return int(self._np_random.integers(self.n))


class BonusEnvironment:
    """Known game/bonus streams, including tracker activity during reset noops.

    Each six-decision game scores 12 raw points. Its first rally earns bonus 2,
    and the next earns 1.8. Reset activity is deliberately much larger so that
    accidentally including reset noops in replay or metrics is easy to detect.
    """

    RAW = (0.0, 1.0, 7.0, 0.0, 0.0, 4.0)
    BONUS = (1.0, 0.2, 0.8, 0.0, 1.0, 0.8)

    def __init__(self):
        self.action_space = ActionSpace()
        self.position = 0
        self.resets = 0
        self.bonus_total = 0.0
        self.bounces_total = 0
        self.frames = np.zeros((4, 2, 2), dtype=np.uint8)
        self.closed = False

    def info(self):
        lives = 0 if self.position == 6 else (2 if self.position >= 4 else 3)
        return {
            "lives": lives,
            "rally_reward": {
                "bonus_total": self.bonus_total,
                "bounces_total": self.bounces_total,
            },
        }

    def reset(self, seed=None):
        self.position = 0
        self.resets += 1
        self.frames.fill(0)
        self.bonus_total += 17.0
        self.bounces_total += 3
        return self.frames.copy(), self.info()

    def step(self, action):
        index = self.position
        self.position += 1
        self.bonus_total += self.BONUS[index]
        self.bounces_total += int(self.position in (1, 5))
        frame = np.full((1, 2, 2), self.position + action, dtype=np.uint8)
        self.frames = np.concatenate((self.frames[1:], frame))
        return self.frames.copy(), self.RAW[index], self.position == 6, False, self.info()

    def get_state(self):
        return {
            "position": self.position, "resets": self.resets,
            "bonus_total": self.bonus_total, "bounces_total": self.bounces_total,
            "frames": self.frames.copy(),
        }

    def set_state(self, state):
        for name in ("position", "resets", "bonus_total", "bounces_total"):
            setattr(self, name, state[name])
        self.frames = state["frames"].copy()

    def close(self):
        self.closed = True


class SmallNetwork(nn.Module):
    def __init__(self, in_channels=4, n_actions=2):
        super().__init__()
        self.output = nn.Linear(1, n_actions)

    def forward(self, states):
        value = states.float().reshape(len(states), -1).mean(dim=1, keepdim=True)
        return self.output(value / 255.0)


class RewardTrainingTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())

    def arguments(self, weights, algorithm="double-dqn", shaping="rally"):
        return [
            "--algorithm", algorithm, "--seed", "81327019", "--device", "cpu",
            "--torch-threads", "1", "--steps", "12", "--batch-size", "2",
            "--buffer-capacity", "12", "--min-replay-size", "2", "--train-freq", "2",
            "--lr", "0.00005", "--gamma", "0.995", "--target-update-freq", "5",
            "--save-freq", "4", "--log-freq", "2", "--health-freq", "0",
            "--replay-storage", "frames", "--terminal-on-life-loss",
            "--reward-shaping", shaping, "--save-path", str(weights),
        ]

    def run_training(self, arguments):
        env = BonusEnvironment()
        with patch.object(train, "make_breakout_env", return_value=env), \
                patch.object(train, "BreakoutBasicNetwork", SmallNetwork), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            train.train(arguments)
        self.assertTrue(env.closed)
        return env

    def assert_nested_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        elif isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second)
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                with self.subTest(key=key):
                    self.assert_nested_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(first, second)

    def test_replay_uses_raw_plus_bonus_but_scores_remain_raw_and_reset_noops_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model.pth"
            env = self.run_training(self.arguments(weights))
            saved = load_training_checkpoint(weights.with_suffix(".training.pt"))
            expected = np.array(BonusEnvironment.RAW) + np.array(BonusEnvironment.BONUS)
            np.testing.assert_allclose([row[2] for row in saved["replay_buffer"]],
                                       np.tile(expected, 2), rtol=0, atol=1e-12)
            self.assertEqual(saved["last_episode_reward"], 12.0)
            self.assertEqual(saved["episodes_completed"], 2)
            counters = saved["reward_state"]
            self.assertAlmostEqual(counters["last_episode_bonus"], 3.8)
            self.assertAlmostEqual(counters["total_bonus"], 7.6)
            self.assertEqual(counters["last_episode_bounces"], 2)
            self.assertEqual(counters["total_bounces"], 4)
            self.assertEqual(counters["episode_bonus"], 0)
            self.assertEqual(counters["tracker_bonus_total"], env.bonus_total)
            self.assertEqual(counters["tracker_bounces_total"], env.bounces_total)

            progress = json.loads(weights.with_suffix(".progress.json").read_text())
            self.assertEqual(progress["last_episode_reward"], 12)
            self.assertAlmostEqual(progress["last_episode_training_reward"], 15.8)
            self.assertAlmostEqual(progress["total_bonus"], 7.6)
            self.assertEqual(progress["total_bounces"], 4)
            with weights.with_suffix(".metrics.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(all(None not in row and None not in row.values() for row in rows))
            self.assertEqual(float(rows[-1]["last_episode_reward"]), 12)
            self.assertAlmostEqual(float(rows[-1]["last_episode_training_reward"]), 15.8)
            self.assertEqual(int(rows[-1]["total_bounces"]), 4)

    def test_mid_rally_resume_preserves_bonus_replay_optimizer_and_weights_for_both_algorithms(self):
        for algorithm in ("dqn", "double-dqn"):
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as directory:
                direct = Path(directory) / "direct" / "model.pth"
                resumed = Path(directory) / "resumed" / "model.pth"
                self.run_training(self.arguments(direct, algorithm))
                self.run_training(self.arguments(resumed, algorithm) + ["--stop-at", "2"])
                snapshot = resumed.with_suffix(".training.pt")
                paused = load_training_checkpoint(snapshot)
                self.assertEqual(paused["step"], 2)
                self.assertAlmostEqual(paused["reward_state"]["episode_bonus"], 1.2)
                self.assertEqual(paused["reward_state"]["episode_bounces"], 1)
                self.run_training(["--resume", str(snapshot), "--device", "cpu"])
                expected = load_training_checkpoint(direct.with_suffix(".training.pt"))
                actual = load_training_checkpoint(snapshot)
                for name in ("policy_net", "target_net", "optimizer", "state", "reward_state",
                             "previous_lives", "episodes_completed", "epsilon", "recent_loss", "step"):
                    with self.subTest(component=name):
                        self.assert_nested_equal(expected[name], actual[name])
                self.assert_nested_equal(list(expected["replay_buffer"]), list(actual["replay_buffer"]))
                self.assert_nested_equal(expected["environment"]["state"], actual["environment"]["state"])
                self.assert_nested_equal(expected["rng"]["python"], actual["rng"]["python"])
                self.assert_nested_equal(expected["rng"]["torch"], actual["rng"]["torch"])

    def test_resume_rejects_reward_and_learning_changes_before_touching_run_files(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model.pth"
            self.run_training(self.arguments(weights) + ["--stop-at", "2"])
            snapshot = weights.with_suffix(".training.pt")
            original = {path: path.read_bytes() for path in Path(directory).iterdir() if path.is_file()}
            changes = (
                ["--reward-shaping", "none"], ["--bounce-bonus", "2"],
                ["--rally-bonus-step", "0.4"], ["--rally-bonus-max-multiplier", "3"],
                ["--lr", "0.0001"], ["--gamma", "0.99"],
                ["--train-freq", "4"], ["--buffer-capacity", "24"],
            )
            for change in changes:
                with self.subTest(change=change), patch.object(train, "make_breakout_env") as make_env, \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    train.train(["--resume", str(snapshot)] + change)
                make_env.assert_not_called()
                self.assertEqual({path: path.read_bytes() for path in original}, original)

    def test_legacy_checkpoint_without_reward_fields_resumes_with_raw_rewards_only(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model.pth"
            self.run_training(self.arguments(weights, shaping="none") + ["--stop-at", "2"])
            snapshot = weights.with_suffix(".training.pt")
            saved = load_training_checkpoint(snapshot)
            for name in ("reward_shaping", "bounce_bonus", "rally_bonus_step", "rally_bonus_max_multiplier"):
                saved["config"].pop(name)
            saved.pop("reward_state")
            torch.save(saved, snapshot)
            args = train.parse_args(["--resume", str(snapshot)])
            self.assertEqual(args.reward_shaping, "none")
            self.assertEqual((args.bounce_bonus, args.rally_bonus_step,
                              args.rally_bonus_max_multiplier), (1.0, 0.2, 2.0))
            self.run_training(["--resume", str(snapshot), "--device", "cpu"])
            actual = load_training_checkpoint(snapshot)
            self.assertEqual([row[2] for row in actual["replay_buffer"]], list(BonusEnvironment.RAW) * 2)
            self.assertEqual(actual["last_episode_reward"], 12)
            progress = json.loads(weights.with_suffix(".progress.json").read_text())
            self.assertNotIn("total_bonus", progress)

    def test_invalid_reward_parameters_fail_before_environment_creation(self):
        options = {
            "--bounce-bonus": ("nan", "inf", "-1"),
            "--rally-bonus-step": ("nan", "inf", "-0.1"),
            "--rally-bonus-max-multiplier": ("nan", "inf", "0.9"),
        }
        for option, values in options.items():
            for value in values:
                with self.subTest(option=option, value=value), \
                        patch.object(train, "make_breakout_env") as make_env, \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    train.train(["--reward-shaping", "rally", option, value])
                make_env.assert_not_called()
        with patch.object(train, "make_breakout_env") as make_env, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train.train(["--bounce-bonus", "1e308", "--rally-bonus-max-multiplier", "2"])
        make_env.assert_not_called()


if __name__ == "__main__":
    unittest.main()
