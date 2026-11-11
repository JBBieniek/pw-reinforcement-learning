"""Training lifecycle checks using a small environment and network."""

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
from experiments.ConceptCheck.health import RepresentationCollapseError


class TinyNetwork(nn.Module):
    def __init__(self, in_channels=4, n_actions=2):
        super().__init__()
        self.output = nn.Linear(1, n_actions)

    def forward(self, states):
        return self.output(states.float().reshape(states.shape[0], -1) / 255.0)


class FakeActionSpace:
    n = 2

    def __init__(self):
        self.seed_value = None

    def seed(self, value):
        self.seed_value = value

    def sample(self):
        return 0


class FakeEnvironment:
    def __init__(self, fail_after=None):
        self.action_space = FakeActionSpace()
        self.fail_after = fail_after
        self.steps = 0
        self.reset_seed = None
        self.closed = False

    def reset(self, seed=None):
        if seed is not None:
            self.reset_seed = seed
        return np.array([0], dtype=np.uint8), {}

    def step(self, action):
        if self.fail_after is not None and self.steps >= self.fail_after:
            raise RuntimeError("simulated emulator failure")
        self.steps += 1
        return (
            np.array([self.steps], dtype=np.uint8),
            1.0,
            self.steps % 3 == 0,
            False,
            {},
        )

    def close(self):
        self.closed = True


class TrainingLifecycleTests(unittest.TestCase):
    def setUp(self):
        original_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, original_threads)

    def arguments(self, checkpoint):
        return [
            "--algorithm", "double-dqn",
            "--device", "cpu",
            "--torch-threads", "1",
            "--steps", "12",
            "--batch-size", "2",
            "--buffer-capacity", "12",
            "--min-replay-size", "2",
            "--train-freq", "4",
            "--target-update-freq", "8",
            "--save-freq", "8",
            "--log-freq", "4",
            "--save-path", str(checkpoint),
        ]

    @contextlib.contextmanager
    def environment(self, env):
        with (
            patch.object(train, "make_breakout_env", return_value=env),
            patch.object(train, "BreakoutBasicNetwork", TinyNetwork),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            yield

    def test_completed_run_persists_seed_metrics_and_loadable_weights(self):
        env = FakeEnvironment()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            with self.environment(env), patch.object(
                train, "compute_td_targets", wraps=train.compute_td_targets,
            ) as target_calculation:
                train.train(self.arguments(checkpoint) + ["--seed", "321759126"])

            self.assertEqual(target_calculation.call_count, 3)
            config = json.loads(checkpoint.with_suffix(".config.json").read_text())
            progress = json.loads(checkpoint.with_suffix(".progress.json").read_text())
            self.assertEqual(config["seed"], 321759126)
            self.assertEqual(config["activation"], "leaky-relu")
            self.assertTrue(config["deterministic"])
            self.assertTrue(config["numerical_runtime"]["deterministic_algorithms"])
            self.assertEqual(env.reset_seed, 321759126)
            self.assertEqual(env.action_space.seed_value, 321759126)
            self.assertEqual(progress["status"], "completed")
            self.assertEqual(progress["step"], 12)
            self.assertIsNone(progress["error"])
            with checkpoint.with_suffix(".metrics.csv").open(newline="") as handle:
                metrics = list(csv.DictReader(handle))
            self.assertTrue(metrics)
            self.assertEqual(metrics[-1]["status"], "completed")
            model = TinyNetwork()
            model.load_state_dict(torch.load(checkpoint, weights_only=True))
            self.assertTrue(torch.isfinite(model(torch.zeros(1, 1))).all())
        self.assertTrue(env.closed)

    def test_generated_seed_is_persisted_and_used_by_environment(self):
        env = FakeEnvironment()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            with self.environment(env), patch.object(
                train.secrets, "randbits", return_value=384901727,
            ) as random_seed:
                train.train(self.arguments(checkpoint))

            random_seed.assert_called_once_with(32)
            config = json.loads(checkpoint.with_suffix(".config.json").read_text())
            self.assertEqual(config["seed"], 384901727)
            self.assertEqual(env.reset_seed, 384901727)
            self.assertEqual(env.action_space.seed_value, 384901727)

    def test_environment_failure_is_saved_and_propagated(self):
        env = FakeEnvironment(fail_after=5)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            with self.environment(env), self.assertRaisesRegex(
                RuntimeError, "simulated emulator failure",
            ):
                train.train(self.arguments(checkpoint) + ["--seed", "83013462"])

            progress = json.loads(checkpoint.with_suffix(".progress.json").read_text())
            self.assertEqual(progress["status"], "failed")
            self.assertIn("simulated emulator failure", progress["error"])
            model = TinyNetwork()
            model.load_state_dict(torch.load(checkpoint, weights_only=True))
        self.assertTrue(env.closed)

    def test_health_failure_stops_at_completed_step_and_retains_snapshot(self):
        env = FakeEnvironment()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            with self.environment(env), patch.object(
                train.HealthMonitor, "check", side_effect=RepresentationCollapseError("dead CNN"),
            ), self.assertRaisesRegex(RepresentationCollapseError, "dead CNN"):
                train.train(self.arguments(checkpoint) + ["--seed", "83013462", "--health-freq", "4"])
            snapshot = load_training_checkpoint(checkpoint.with_suffix(".training.pt"))
            self.assertEqual(snapshot["step"], 4)
            self.assertEqual(len(snapshot["replay_buffer"]), 4)
            self.assertIn("health_state", snapshot)
            self.assertEqual(env.steps, 4)
            progress = json.loads(checkpoint.with_suffix(".progress.json").read_text())
            self.assertEqual(progress["status"], "failed")
            self.assertEqual(progress["step"], 4)
            self.assertTrue(checkpoint.with_name("model.step-000000004.pth").exists())
        self.assertTrue(env.closed)


if __name__ == "__main__":
    unittest.main()
