"""Check that pausing training preserves learning, exploration and run history."""

from collections import deque
import contextlib
from copy import deepcopy
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


class RandomActionSpace:
    n = 2

    def seed(self, seed):
        self._np_random = np.random.default_rng(seed)

    def sample(self):
        return int(self._np_random.integers(self.n))


class StatefulEnvironment:
    """Exercise private and global RNGs, episode resets and truncations."""

    def __init__(self):
        self.action_space = RandomActionSpace()
        self._np_random = np.random.default_rng()
        self.total_steps = 0
        self.episode_steps = 0
        self.observation = 0
        self.closed = False

    def reset(self, seed=None):
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
            self.total_steps = 0
        self.episode_steps = 0
        self.observation = int(self._np_random.integers(50))
        return np.array([self.observation], dtype=np.uint8), {}

    def step(self, action):
        self.total_steps += 1
        self.episode_steps += 1
        self.observation = (
            self.observation + 3 * action + int(self._np_random.integers(1, 40))
            + int(np.random.randint(0, 13))
        ) % 251
        done = self.episode_steps == 5
        terminated = done and self.total_steps % 2 == 0
        truncated = done and not terminated
        return (
            np.array([self.observation], dtype=np.uint8),
            float(self.observation % 5 - 1 + action),
            terminated, truncated, {},
        )

    def get_state(self):
        return {
            "total_steps": self.total_steps,
            "episode_steps": self.episode_steps,
            "observation": self.observation,
            "rng": deepcopy(self._np_random.bit_generator.state),
        }

    def set_state(self, state):
        self.total_steps = state["total_steps"]
        self.episode_steps = state["episode_steps"]
        self.observation = state["observation"]
        self._np_random.bit_generator.state = deepcopy(state["rng"])

    def close(self):
        self.closed = True


class StochasticNetwork(nn.Module):
    """Dropout ensures restoring the Torch RNG matters for identical weights."""

    def __init__(self, in_channels=4, n_actions=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 8), nn.Tanh(), nn.Dropout(0.35), nn.Linear(8, n_actions),
        )

    def forward(self, states):
        return self.layers(states.float().reshape(states.shape[0], -1) / 255.0)


class ResumeTrainingTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())

    def arguments(self, weights, algorithm="double-dqn"):
        return [
            "--algorithm", algorithm, "--seed", "81327019",
            "--device", "cpu", "--torch-threads", "1", "--steps", "24",
            "--batch-size", "2", "--buffer-capacity", "9",
            "--min-replay-size", "2", "--train-freq", "2",
            "--target-update-freq", "11", "--epsilon-decay-steps", "24",
            "--save-freq", "4", "--log-freq", "3", "--save-path", str(weights),
        ]

    def run_training(self, arguments):
        env = StatefulEnvironment()
        with (
            patch.object(train, "make_breakout_env", return_value=env),
            patch.object(train, "BreakoutBasicNetwork", StochasticNetwork),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            train.train(arguments)
        self.assertTrue(env.closed)

    def assert_nested_equal(self, expected, actual):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, np.random.Generator):
            self.assert_nested_equal(expected.bit_generator.state, actual.bit_generator.state)
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected:
                with self.subTest(key=key):
                    self.assert_nested_equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple, deque)):
            self.assertEqual(len(expected), len(actual))
            if isinstance(expected, deque):
                self.assertEqual(expected.maxlen, actual.maxlen)
            for left, right in zip(expected, actual):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(expected, actual)

    def test_paused_and_uninterrupted_training_are_identical_for_both_algorithms(self):
        for algorithm in ("dqn", "double-dqn"):
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as directory:
                direct = Path(directory) / "direct" / "model.pth"
                resumed = Path(directory) / "resumed" / "model.pth"
                self.run_training(self.arguments(direct, algorithm))
                self.run_training(self.arguments(resumed, algorithm) + ["--stop-at", "7"])
                paused = load_training_checkpoint(resumed.with_suffix(".training.pt"))
                self.assertEqual(paused["step"], 7)
                self.assertEqual(paused["config"]["steps"], 24)
                self.assertAlmostEqual(paused["epsilon"], 1.0 - 0.9 * 7 / 24)
                self.assertEqual(paused["environment"]["state"]["episode_steps"], 2)
                self.run_training([
                    "--resume", str(resumed.with_suffix(".training.pt")),
                    "--device", "cpu", "--torch-threads", "1",
                ])
                expected = load_training_checkpoint(direct.with_suffix(".training.pt"))
                actual = load_training_checkpoint(resumed.with_suffix(".training.pt"))
                for key in (
                    "policy_net", "target_net", "optimizer", "replay_buffer", "state",
                    "step", "epsilon", "episode_reward", "last_episode_reward",
                    "episodes_completed", "recent_loss", "environment", "rng",
                ):
                    with self.subTest(component=key):
                        self.assert_nested_equal(expected[key], actual[key])
                self.assertEqual(actual["step"], 24)
                self.assertEqual(actual["config"]["algorithm"], algorithm)
                self.assertTrue(resumed.with_name("model.step-000000007.pth").exists())

    def test_resume_appends_history_and_preserves_original_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model.pth"
            self.run_training(self.arguments(weights) + ["--stop-at", "7"])
            config_path = weights.with_suffix(".config.json")
            metrics_path = weights.with_suffix(".metrics.csv")
            original_config = json.loads(config_path.read_text(encoding="utf-8"))
            original_metrics = metrics_path.read_text(encoding="utf-8")
            original = load_training_checkpoint(weights.with_suffix(".training.pt"))
            self.run_training([
                "--resume", str(weights.with_suffix(".training.pt")),
                "--device", "cpu", "--torch-threads", "1",
            ])
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["started_at_utc"], original_config["started_at_utc"])
            for key in ("algorithm", "seed", "steps", "epsilon_decay_steps", "lr", "buffer_capacity"):
                self.assertEqual(config[key], original_config[key])
            self.assertIn("resumed_at_utc", config)
            self.assertTrue(metrics_path.read_text(encoding="utf-8").startswith(original_metrics))
            with metrics_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            steps = [int(row["step"]) for row in rows]
            self.assertEqual(steps, sorted(steps))
            self.assertEqual(rows[-1]["status"], "completed")
            self.assertEqual(steps[-1], 24)
            final = load_training_checkpoint(weights.with_suffix(".training.pt"))
            self.assertGreaterEqual(final["elapsed_seconds"], original["elapsed_seconds"])

    def test_extending_target_preserves_existing_exploration_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model.pth"
            self.run_training(self.arguments(weights) + ["--stop-at", "7"])
            self.run_training([
                "--resume", str(weights.with_suffix(".training.pt")),
                "--steps", "36", "--stop-at", "12", "--device", "cpu",
                "--torch-threads", "1",
            ])
            checkpoint = load_training_checkpoint(weights.with_suffix(".training.pt"))
            self.assertEqual(checkpoint["step"], 12)
            self.assertEqual(checkpoint["config"]["steps"], 36)
            self.assertEqual(checkpoint["config"]["epsilon_decay_steps"], 24)
            self.assertAlmostEqual(checkpoint["epsilon"], 0.55)
            progress = json.loads(weights.with_suffix(".progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["status"], "paused")

    def test_algorithm_change_is_rejected_before_touching_run_files(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model.pth"
            self.run_training(self.arguments(weights) + ["--stop-at", "7"])
            paths = [weights, weights.with_suffix(".training.pt"),
                     weights.with_suffix(".config.json"), weights.with_suffix(".metrics.csv")]
            original = {path: path.read_bytes() for path in paths}
            stderr = io.StringIO()
            with (
                patch.object(train, "make_breakout_env") as make_env,
                contextlib.redirect_stderr(stderr),
                self.assertRaises(SystemExit),
            ):
                train.train(["--resume", str(weights.with_suffix(".training.pt")),
                             "--algorithm", "dqn"])
            make_env.assert_not_called()
            self.assertIn("--algorithm differs", stderr.getvalue())
            for path in paths:
                self.assertEqual(path.read_bytes(), original[path])

    def test_weights_only_file_is_rejected_with_full_checkpoint_guidance(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "old-model.pth"
            torch.save(StochasticNetwork().state_dict(), weights)
            original = weights.read_bytes()
            stderr = io.StringIO()
            with (
                patch.object(train, "make_breakout_env") as make_env,
                contextlib.redirect_stderr(stderr),
                self.assertRaises(SystemExit),
            ):
                train.train(["--resume", str(weights)])
            make_env.assert_not_called()
            self.assertIn(".training.pt", stderr.getvalue())
            self.assertIn("not .pth weights", stderr.getvalue())
            self.assertEqual(weights.read_bytes(), original)

    def test_activation_and_numerical_mode_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model.pth"
            self.run_training(self.arguments(weights) + ["--stop-at", "7"])
            snapshot = weights.with_suffix(".training.pt")
            original = snapshot.read_bytes()
            for option in (["--activation", "relu"], ["--no-deterministic"]):
                with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    train.parse_args(["--resume", str(snapshot)] + option)
            self.assertEqual(snapshot.read_bytes(), original)

    def test_legacy_snapshot_keeps_relu_and_original_numerical_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model.pth"
            self.run_training(self.arguments(weights) + ["--stop-at", "7"])
            snapshot = weights.with_suffix(".training.pt")
            saved = load_training_checkpoint(snapshot)
            del saved["config"]["activation"], saved["config"]["deterministic"]
            torch.save(saved, snapshot)
            args = train.parse_args(["--resume", str(snapshot)])
            self.assertEqual(args.activation, "relu")
            self.assertFalse(args.deterministic)

    def test_new_output_path_does_not_overwrite_source_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "original" / "model.pth"
            branch_weights = Path(directory) / "branched" / "model.pth"
            self.run_training(self.arguments(weights) + ["--stop-at", "7"])
            snapshot = weights.with_suffix(".training.pt")
            original = snapshot.read_bytes()
            self.run_training(["--resume", str(snapshot), "--save-path", str(branch_weights),
                               "--stop-at", "12", "--device", "cpu", "--torch-threads", "1"])
            self.assertEqual(snapshot.read_bytes(), original)
            branched = load_training_checkpoint(branch_weights.with_suffix(".training.pt"))
            self.assertEqual(branched["step"], 12)
            self.assertEqual(branched["config"]["checkpoint_path"], str(branch_weights.with_suffix(".training.pt")))


if __name__ == "__main__":
    unittest.main()
