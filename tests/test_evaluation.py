"""Evaluation protocol checks without running the Atari emulator."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from experiments.ConceptCheck import test as evaluation


class TinyNetwork(nn.Module):
    def __init__(self, in_channels=4, n_actions=2):
        super().__init__()
        self.scores = nn.Parameter(torch.tensor([2.0, 0.0]))

    def forward(self, states):
        return self.scores.unsqueeze(0).expand(states.shape[0], -1)


class ActivationNetwork(TinyNetwork):
    def __init__(self, in_channels=4, n_actions=2):
        super().__init__(in_channels, n_actions)
        self.scores.data.copy_(torch.tensor([-2.0, -1.0]))
        self.activation = nn.ReLU()

    def forward(self, states):
        return self.activation(self.scores).unsqueeze(0).expand(states.shape[0], -1)


class FakeActionSpace:
    n = 2

    def __init__(self):
        self.seeds = []

    def seed(self, seed):
        self.seeds.append(seed)


class FakeEnvironment:
    def __init__(self, terminate=True):
        self.action_space = FakeActionSpace()
        self.seeds = []
        self.actions = []
        self.terminate = terminate
        self.closed = False

    def reset(self, seed=None):
        self.seeds.append(seed)
        self.step_number = 0
        return np.zeros(1, dtype=np.uint8), {"lives": 5}

    def step(self, action):
        self.step_number += 1
        self.actions.append(action)
        # Life is lost on the model's first action, requiring FIRE at step 3.
        return (np.zeros(1, dtype=np.uint8), 288.0,
                self.terminate and self.step_number == 3, False,
                {"lives": 5 if self.step_number < 2 else 4})

    def close(self):
        self.closed = True


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        self.enterContext(patch.dict(os.environ))
        self.addCleanup(torch.use_deterministic_algorithms,
                        torch.are_deterministic_algorithms_enabled(),
                        warn_only=torch.is_deterministic_algorithms_warn_only_enabled())
        for backend, attributes in (
            (torch.backends.cudnn, ("deterministic", "benchmark", "allow_tf32")),
            (torch.backends.cuda.matmul, ("allow_tf32",)),
        ):
            for attribute in attributes:
                self.addCleanup(setattr, backend, attribute, getattr(backend, attribute))

    def test_life_loss_triggers_fire_and_summary_counts_864(self):
        env = FakeEnvironment()
        report = evaluation.evaluate_policy(TinyNetwork(), env, episodes=2, seed=782165)
        self.assertEqual(env.actions, [1, 0, 1, 1, 0, 1])
        self.assertEqual(report["summary"]["wins_864"], 2)
        self.assertEqual(report["summary"]["win_rate_864"], 1.0)
        self.assertEqual(report["summary"]["mean_reward"], 864.0)
        self.assertEqual(report["summary"]["std_reward"], 0.0)
        self.assertEqual(report["summary"]["hit_step_limit"], 0)
        json.dumps(report)

    def test_fair_episode_seeds_wrap_and_no_auto_fire(self):
        left, right = FakeEnvironment(), FakeEnvironment()
        first = evaluation.evaluate_policy(TinyNetwork(), left, episodes=2, seed=2**32 - 1,
                                           auto_fire=False)
        second = evaluation.evaluate_policy(TinyNetwork(), right, episodes=2, seed=2**32 - 1)
        self.assertEqual(first["episode_seeds"], [2**32 - 1, 0])
        self.assertEqual(left.seeds, right.seeds)
        self.assertEqual(left.action_space.seeds, right.action_space.seeds)
        self.assertEqual(left.actions, [1, 0, 0, 1, 0, 0])
        self.assertEqual(first["episode_seeds"], second["episode_seeds"])

    def test_cap_includes_initial_fire(self):
        env = FakeEnvironment(terminate=False)
        report = evaluation.evaluate_policy(TinyNetwork(), env, episodes=1, seed=274690,
                                            max_steps=2)
        self.assertEqual(env.actions, [1, 0])
        self.assertEqual(report["episodes"][0]["steps"], 2)
        self.assertEqual(report["summary"]["hit_step_limit"], 1)
        self.assertFalse(report["episodes"][0]["terminated"])

    def test_missing_checkpoint_is_an_error_before_environment_created(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(evaluation, "make_breakout_env") as make_env:
                with self.assertRaises(FileNotFoundError):
                    evaluation.evaluate(Path(directory) / "missing.pth")
            make_env.assert_not_called()

    def test_load_weights_write_generated_seed_and_close_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            env = FakeEnvironment()
            model_path = Path(directory) / "model.pth"
            output_path = Path(directory) / "evaluation.json"
            torch.save(TinyNetwork().state_dict(), model_path)
            with (patch.object(evaluation, "make_breakout_env", return_value=env),
                  patch.object(evaluation, "BreakoutBasicNetwork", TinyNetwork),
                  patch.object(evaluation.secrets, "randbits", return_value=2816593307)):
                report = evaluation.evaluate(model_path, episodes=2, device="cpu",
                                              output_path=output_path)
            saved = json.loads(output_path.read_text())
            self.assertEqual(saved["evaluation_seed"], 2816593307)
            self.assertEqual(saved["episode_seeds"], [2816593307, 2816593308])
            self.assertEqual(saved["activation"], "relu")
            self.assertFalse(saved["deterministic"])
            self.assertFalse(saved["runtime_flags"]["deterministic_algorithms"])
            self.assertEqual(saved, report)
            self.assertTrue(env.closed)

    def test_sidecar_selects_activation_and_determinism_and_api_can_override(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pth"
            torch.save(ActivationNetwork().state_dict(), model_path)
            model_path.with_suffix(".config.json").write_text(json.dumps({
                "activation": "leaky-relu", "deterministic": True,
            }))
            environments = [FakeEnvironment(), FakeEnvironment()]
            with (patch.object(evaluation, "make_breakout_env", side_effect=environments),
                  patch.object(evaluation, "BreakoutBasicNetwork", ActivationNetwork)):
                inherited = evaluation.evaluate(model_path, episodes=1, seed=936071, device="cpu")
                overridden = evaluation.evaluate(model_path, episodes=1, seed=936071, device="cpu",
                                                   activation="relu", deterministic=False)
            self.assertEqual(inherited["activation"], "leaky-relu")
            self.assertTrue(inherited["deterministic"])
            self.assertTrue(inherited["runtime_flags"]["deterministic_algorithms"])
            self.assertEqual(environments[0].actions, [1, 1, 1])
            self.assertEqual(overridden["activation"], "relu")
            self.assertFalse(overridden["deterministic"])
            self.assertEqual(environments[1].actions, [1, 0, 1])

    def test_legacy_sidecar_without_runtime_fields_preserves_relu_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pth"
            torch.save(ActivationNetwork().state_dict(), model_path)
            model_path.with_suffix(".config.json").write_text(json.dumps({"mode": 0}))
            env = FakeEnvironment()
            with (patch.object(evaluation, "make_breakout_env", return_value=env),
                  patch.object(evaluation, "BreakoutBasicNetwork", ActivationNetwork)):
                report = evaluation.evaluate(model_path, episodes=1, seed=936071, device="cpu")
            self.assertEqual(report["activation"], "relu")
            self.assertFalse(report["deterministic"])
            self.assertEqual(env.actions, [1, 0, 1])

    def test_runtime_is_configured_before_cuda_selection_and_seed_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pth"
            torch.save(TinyNetwork().state_dict(), model_path)
            events = []

            def configure(enabled):
                events.append(("configure", enabled))
                return {"deterministic_algorithms": enabled}

            def cuda_available():
                events.append("cuda")
                return False

            def generated_seed(bits):
                events.append("seed")
                return 936071

            with (patch.object(evaluation, "make_breakout_env", return_value=FakeEnvironment()),
                  patch.object(evaluation, "BreakoutBasicNetwork", TinyNetwork),
                  patch.object(evaluation, "configure_determinism", side_effect=configure),
                  patch.object(torch.cuda, "is_available", side_effect=cuda_available),
                  patch.object(evaluation.secrets, "randbits", side_effect=generated_seed)):
                evaluation.evaluate(model_path, episodes=1, deterministic=True)
            self.assertEqual(events, [("configure", True), "cuda", "seed"])

    def test_full_checkpoint_requires_accompanying_policy_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "full.pth"
            torch.save({"policy_net": TinyNetwork().state_dict(), "optimizer": {}}, model_path)
            with patch.object(evaluation, "make_breakout_env") as make_env:
                with self.assertRaisesRegex(ValueError, "policy .pth"):
                    evaluation.evaluate(model_path, episodes=1, seed=583210, device="cpu")
            make_env.assert_not_called()

    def test_failed_output_write_propagates_and_environment_is_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            env = FakeEnvironment()
            model_path = Path(directory) / "model.pth"
            torch.save(TinyNetwork().state_dict(), model_path)
            output_parent = Path(directory) / "existing-file"
            output_parent.write_text("cannot be a directory")
            with (patch.object(evaluation, "make_breakout_env", return_value=env),
                  patch.object(evaluation, "BreakoutBasicNetwork", TinyNetwork)):
                with self.assertRaises(OSError):
                    evaluation.evaluate(model_path, episodes=1, seed=583210, device="cpu",
                                        output_path=output_parent / "evaluation.json")
            self.assertTrue(env.closed)

    def test_cli_environment_defaults_follow_sidecar_and_explicit_flags_win(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pth"
            model_path.with_suffix(".config.json").write_text(json.dumps({
                "mode": 3, "difficulty": 1, "frame_skip": 8,
                "sticky_action_probability": 0.0,
                "activation": "leaky-relu", "deterministic": True,
            }))
            args = evaluation.parse_args(["--model-path", str(model_path), "--mode", "0",
                                          "--no-auto-fire", "--seed", "12598"])
            self.assertEqual(args.mode, 0)
            self.assertEqual(args.difficulty, 1)
            self.assertEqual(args.frame_skip, 8)
            self.assertEqual(args.sticky_action_probability, 0.0)
            self.assertFalse(args.auto_fire)
            self.assertEqual(args.activation, "leaky-relu")
            self.assertTrue(args.deterministic)
            self.assertEqual(args.output, str(model_path.with_suffix(".evaluation.json")))
            overridden = evaluation.parse_args([
                "--model-path", str(model_path), "--seed", "12598",
                "--activation", "relu", "--no-deterministic",
            ])
            self.assertEqual(overridden.activation, "relu")
            self.assertFalse(overridden.deterministic)

    def test_cli_legacy_defaults_and_explicit_runtime_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "legacy.pth"
            default = evaluation.parse_args(["--model-path", str(model_path), "--seed", "12598"])
            self.assertEqual(default.activation, "relu")
            self.assertFalse(default.deterministic)
            explicit = evaluation.parse_args([
                "--model-path", str(model_path), "--seed", "12598",
                "--activation", "leaky-relu", "--deterministic",
            ])
            self.assertEqual(explicit.activation, "leaky-relu")
            self.assertTrue(explicit.deterministic)


if __name__ == "__main__":
    unittest.main()
