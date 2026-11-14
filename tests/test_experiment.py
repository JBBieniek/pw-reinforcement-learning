"""Controller checks with lightweight subprocess doubles, without Atari training."""

import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiments.ConceptCheck import experiment


ENVIRONMENT = {"mode": 0, "difficulty": 0, "frame_skip": 4,
               "sticky_action_probability": 0.25}


def option(command, name):
    return command[command.index(name) + 1]


class ChildFactory:
    def __init__(self, fail=None):
        self.commands = []
        self.fail = fail
        self.resume_checkpoints = []

    def __call__(self, command, cwd):
        command = list(command)
        self.commands.append(command)
        phase = "train" if "experiments.ConceptCheck.train" in command else "eval"
        factory = self

        class Child:
            pid = 10000 + len(factory.commands)

            def wait(self, timeout=None):
                if factory.fail == phase:
                    factory.fail = None
                    return 7
                if phase == "train":
                    factory.train(command)
                else:
                    factory.evaluate(command)
                return 0

            def poll(self):
                return 0

        return Child()

    def train(self, command):
        directory = Path(option(command, "--save-path")).parent
        checkpoint = directory / "model.training.pt"
        if "--resume" in command:
            previous = json.loads(checkpoint.read_text())
            self.resume_checkpoints.append(previous)
            algorithm, seed = previous["algorithm"], previous["seed"]
            settings = {name: previous.get(name, default)
                        for name, default in experiment.TRAINING_DEFAULTS.items()}
        else:
            algorithm, seed = option(command, "--algorithm"), int(option(command, "--seed"))
            settings = {
                "replay_storage": option(command, "--replay-storage"),
                "terminal_on_life_loss": "--terminal-on-life-loss" in command,
                "buffer_capacity": int(option(command, "--buffer-capacity")),
                "epsilon_decay_steps": int(option(command, "--epsilon-decay-steps")),
                "lr": float(option(command, "--lr")),
                "gamma": float(option(command, "--gamma")),
                "train_freq": int(option(command, "--train-freq")),
                "reward_shaping": option(command, "--reward-shaping"),
                "bounce_bonus": float(option(command, "--bounce-bonus")),
                "rally_bonus_step": float(option(command, "--rally-bonus-step")),
                "rally_bonus_max_multiplier": float(option(command, "--rally-bonus-max-multiplier")),
            }
        activation = option(command, "--activation")
        deterministic = "--deterministic" in command
        step = int(option(command, "--stop-at"))
        checkpoint.write_text(json.dumps({"step": step, "algorithm": algorithm,
                                         "seed": seed, "steps": int(option(command, "--steps")),
                                         "activation": activation, "deterministic": deterministic,
                                         **settings}))
        weights = directory / f"model.step-{step:09d}.pth"
        weights.write_bytes(f"{algorithm}:{seed}:{step}".encode())
        weights.with_suffix(".config.json").write_text(json.dumps(
            {**ENVIRONMENT, "activation": activation, "deterministic": deterministic, **settings}))
        (directory / "model.pth").write_bytes(weights.read_bytes())

    def evaluate(self, command):
        weights = Path(option(command, "--model-path"))
        score = int(weights.stem.rsplit("-", 1)[1])
        episodes = int(option(command, "--episodes"))
        seed = int(option(command, "--seed"))
        seeds = [(seed + index) % 2**32 for index in range(episodes)]
        report = {
            "completed_at_utc": "2026-11-11T12:00:00+00:00",
            "activation": option(command, "--activation"),
            "deterministic": "--deterministic" in command,
            "evaluation_seed": seed, "episode_seeds": seeds,
            "policy_epsilon": 0.0, "auto_fire_after_life_loss": True,
            "max_steps": int(option(command, "--max-steps")),
            "model_path": str(weights), "environment": ENVIRONMENT,
            "episodes": [{"seed": item, "reward": score} for item in seeds],
            "summary": {"episodes": episodes, "mean_reward": score,
                        "median_reward": score, "min_reward": score,
                        "max_reward": score, "std_reward": 0,
                        "wins_864": 0, "win_rate_864": 0,
                        "hit_step_limit": 0, "mean_steps": 10},
        }
        Path(option(command, "--output")).write_text(json.dumps(report))


class ExperimentTests(unittest.TestCase):
    def arguments(self, directory):
        return ["--run-dir", str(directory), "--milestones", "4,8,12",
                "--seed", "230849415", "--eval-seed", "3303497698",
                "--eval-episodes", "2", "--max-steps", "32", "--device", "cpu"]

    def run_with(self, factory, arguments):
        with patch.object(experiment.subprocess, "Popen", factory), patch.object(
                experiment, "_read_training_checkpoint",
                side_effect=lambda path: json.loads(path.read_text())):
            return experiment.run_experiment(arguments)

    def test_all_budgets_train_then_evaluate_and_resume_with_same_target(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            factory = ChildFactory()
            result = self.run_with(factory, self.arguments(directory))
            self.assertEqual(result["status"], "completed")
            self.assertIsNone(result["child_pid"])
            self.assertEqual(result["completed_milestones"], [4, 8, 12])
            self.assertEqual(len(factory.commands), 6)
            for index, budget in enumerate((4, 8, 12)):
                train, evaluate = factory.commands[2 * index:2 * index + 2]
                self.assertIn("experiments.ConceptCheck.train", train)
                self.assertEqual(option(train, "--steps"), "12")
                self.assertEqual(int(option(train, "--stop-at")), budget)
                self.assertEqual("--resume" in train, index > 0)
                for command in (train, evaluate):
                    self.assertEqual(option(command, "--activation"), "leaky-relu")
                    self.assertIn("--deterministic", command)
                    self.assertNotIn("--no-deterministic", command)
                self.assertEqual(option(evaluate, "--seed"), "3303497698")
                self.assertEqual(option(evaluate, "--max-steps-per-second"), "220.0")
            config = json.loads((directory / "experiment.config.json").read_text())
            self.assertEqual(config["activation"], "leaky-relu")
            self.assertIs(config["deterministic"], True)
            with (directory / "comparison.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row["decisions"]) for row in rows], [4, 8, 12])
            self.assertEqual([float(row["mean_reward"]) for row in rows], [4, 8, 12])

    def test_append_trains_only_new_budgets_from_unchanged_old_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self.run_with(ChildFactory(), self.arguments(directory))
            checkpoint = json.loads((directory / "model.training.pt").read_text())
            old_reports = {path: path.read_bytes() for path in directory.glob("eval-*.json")}
            factory = ChildFactory()
            result = self.run_with(factory, ["--run-dir", str(directory), "--resume-experiment",
                                             "--append-milestones", "16,20"])
            self.assertEqual(result["completed_milestones"], [4, 8, 12, 16, 20])
            self.assertEqual(len(factory.commands), 4)
            self.assertEqual(factory.resume_checkpoints[0], checkpoint)
            trains = factory.commands[::2]
            self.assertEqual([int(option(command, "--stop-at")) for command in trains], [16, 20])
            for command in trains:
                self.assertEqual(option(command, "--steps"), "20")
                self.assertIn("--resume", command)
                self.assertNotIn("--epsilon-decay-steps", command)
                self.assertNotIn("--seed", command)
            for path, before in old_reports.items():
                self.assertEqual(path.read_bytes(), before)
            config = json.loads((directory / "experiment.config.json").read_text())
            self.assertEqual(config["previous_targets"], [12])
            self.assertEqual(config["milestones"], [4, 8, 12, 16, 20])
            self.assertNotIn("append_milestones", config)
            self.assertEqual(config["milestone_extensions"][0]["from_milestones"], [4, 8, 12])
            with (directory / "comparison.csv").open(newline="") as stream:
                self.assertEqual([int(row["decisions"]) for row in csv.DictReader(stream)], [4, 8, 12, 16, 20])

    def test_append_requires_resume_and_preserves_immutable_milestones(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            with self.assertRaises(SystemExit):
                experiment.parse_args(["--run-dir", str(directory), "--append-milestones", "16,20"])
            self.run_with(ChildFactory(), self.arguments(directory))
            config = (directory / "experiment.config.json").read_bytes()
            for extra in (["--append-milestones", "12,16"], ["--append-milestones", "8"],
                          ["--append-milestones", "20,16"],
                          ["--append-milestones", "16,20", "--milestones", "4,8,12"],
                          ["--append-milestones", "16,20", "--milestones", "4,8,12,16,20"]):
                with self.subTest(extra=extra), self.assertRaises(SystemExit):
                    self.run_with(ChildFactory(), ["--run-dir", str(directory), "--resume-experiment", *extra])
            self.assertEqual((directory / "experiment.config.json").read_bytes(), config)

    def test_new_append_requires_completed_progress_and_final_checkpoint_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self.run_with(ChildFactory(), self.arguments(directory))
            progress_path = directory / "experiment.progress.json"
            checkpoint_path = directory / "model.training.pt"
            progress = json.loads(progress_path.read_text())
            checkpoint = json.loads(checkpoint_path.read_text())
            invalid_cases = [("failed", checkpoint), ("training", checkpoint),
                             ("completed", {**checkpoint, "step": 8}),
                             ("completed", {**checkpoint, "steps": 99})]
            for status, saved in invalid_cases:
                with self.subTest(status=status, saved=saved):
                    progress_path.write_text(json.dumps({**progress, "status": status}))
                    checkpoint_path.write_text(json.dumps(saved))
                    before = {path: path.read_bytes() for path in
                              (progress_path, checkpoint_path, directory / "experiment.config.json")}
                    factory = ChildFactory()
                    with self.assertRaises(ValueError):
                        self.run_with(factory, ["--run-dir", str(directory), "--resume-experiment",
                                                "--append-milestones", "16,20"])
                    self.assertEqual(factory.commands, [])
                    for path, data in before.items():
                        self.assertEqual(path.read_bytes(), data)

    def test_live_previous_controller_blocks_append_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self.run_with(ChildFactory(), self.arguments(directory))
            progress_path = directory / "experiment.progress.json"
            progress = json.loads(progress_path.read_text())
            progress["pid"] = 12345
            progress_path.write_text(json.dumps(progress))
            before = {path: path.read_bytes() for path in (progress_path, directory / "experiment.config.json")}
            factory = ChildFactory()
            with patch.object(experiment, "_process_alive", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "controller.*still running"):
                    self.run_with(factory, ["--run-dir", str(directory), "--resume-experiment",
                                            "--append-milestones", "16,20"])
            self.assertEqual(factory.commands, [])
            for path, data in before.items():
                self.assertEqual(path.read_bytes(), data)

    def test_held_controller_lock_blocks_append_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self.run_with(ChildFactory(), self.arguments(directory))
            paths = [directory / name for name in ("experiment.config.json", "experiment.progress.json")]
            before = {path: path.read_bytes() for path in paths}
            with experiment._experiment_lock(directory):
                with self.assertRaisesRegex(RuntimeError, "another experiment controller"):
                    self.run_with(ChildFactory(), ["--run-dir", str(directory), "--resume-experiment",
                                                  "--append-milestones", "16,20"])
            for path, data in before.items():
                self.assertEqual(path.read_bytes(), data)

    def test_changed_config_is_not_overwritten_after_argument_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self.run_with(ChildFactory(), self.arguments(directory))
            config_path = directory / "experiment.config.json"
            progress_path = directory / "experiment.progress.json"
            before_progress = progress_path.read_bytes()
            parse_args = experiment.parse_args

            def parse_then_external_write(arguments):
                args = parse_args(arguments)
                config = json.loads(config_path.read_text())
                config["max_steps_per_second"] = 123
                config_path.write_text(json.dumps(config))
                return args

            factory = ChildFactory()
            with patch.object(experiment, "parse_args", side_effect=parse_then_external_write):
                with self.assertRaisesRegex(RuntimeError, "configuration changed"):
                    self.run_with(factory, ["--run-dir", str(directory), "--resume-experiment",
                                            "--append-milestones", "16,20"])
            self.assertEqual(factory.commands, [])
            self.assertEqual(json.loads(config_path.read_text())["max_steps_per_second"], 123)
            self.assertEqual(progress_path.read_bytes(), before_progress)

    def test_append_config_write_crash_can_resume_with_or_without_same_request(self):
        for repeat_append in (False, True):
            with self.subTest(repeat_append=repeat_append), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                self.run_with(ChildFactory(), self.arguments(directory))
                checkpoint_path = directory / "model.training.pt"
                before = checkpoint_path.read_bytes()
                write_json = experiment._write_json

                def crash_after_config(path, data):
                    write_json(path, data)
                    if path.name == "experiment.config.json":
                        raise RuntimeError("simulated crash after config replacement")

                append = ["--run-dir", str(directory), "--resume-experiment", "--append-milestones", "16,20"]
                with patch.object(experiment, "_write_json", side_effect=crash_after_config):
                    with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                        self.run_with(ChildFactory(), append)
                self.assertEqual(checkpoint_path.read_bytes(), before)
                factory = ChildFactory()
                arguments = append if repeat_append else ["--run-dir", str(directory), "--resume-experiment"]
                result = self.run_with(factory, arguments)
                self.assertEqual(result["completed_milestones"], [4, 8, 12, 16, 20])
                self.assertEqual(factory.resume_checkpoints[0]["steps"], 12)
                config = json.loads((directory / "experiment.config.json").read_text())
                self.assertEqual(config["previous_targets"], [12])
                self.assertEqual(len(config["milestone_extensions"]), 1)

    def test_same_append_after_training_failure_and_completed_retries_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self.run_with(ChildFactory(), self.arguments(directory))
            append = ["--run-dir", str(directory), "--resume-experiment", "--append-milestones", "16,20"]
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_with(ChildFactory(fail="train"), append)
            self.run_with(ChildFactory(), append)
            factory = ChildFactory()
            self.run_with(factory, append)
            self.assertEqual(factory.commands, [])
            config = json.loads((directory / "experiment.config.json").read_text())
            self.assertEqual(config["previous_targets"], [12])
            self.assertEqual(len(config["milestone_extensions"]), 1)

    def test_only_recorded_previous_checkpoint_targets_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self.run_with(ChildFactory(), self.arguments(directory))
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_with(ChildFactory(fail="train"), ["--run-dir", str(directory), "--resume-experiment",
                                                         "--append-milestones", "16,20"])
            args = experiment.parse_args(["--run-dir", str(directory), "--resume-experiment"])
            checkpoint_path = directory / "model.training.pt"
            original = json.loads(checkpoint_path.read_text())
            with patch.object(experiment, "_read_training_checkpoint", return_value=original):
                self.assertEqual(experiment._validated_checkpoint(checkpoint_path, args), 12)
            for saved in ({**original, "steps": 16}, {**original, "step": 16}):
                with self.subTest(saved=saved), patch.object(experiment, "_read_training_checkpoint", return_value=saved):
                    with self.assertRaises(ValueError):
                        experiment._validated_checkpoint(checkpoint_path, args)

    def test_failed_evaluation_resumes_from_saved_training_without_repeating_it(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            factory = ChildFactory(fail="eval")
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_with(factory, self.arguments(directory))
            progress = json.loads((directory / "experiment.progress.json").read_text())
            self.assertEqual(progress["status"], "failed")
            self.assertEqual(progress["current_milestone"], 4)
            resumed = ChildFactory()
            result = self.run_with(resumed, ["--run-dir", str(directory), "--resume-experiment"])
            self.assertIn("experiments.ConceptCheck.test", resumed.commands[0])
            trained_budgets = [int(option(command, "--stop-at")) for command in resumed.commands
                               if "experiments.ConceptCheck.train" in command]
            self.assertEqual(trained_budgets, [8, 12])
            self.assertEqual(result["completed_milestones"], [4, 8, 12])

    def test_valid_completed_evaluations_are_reused_and_bad_seed_is_retested(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self.run_with(ChildFactory(), self.arguments(directory))
            factory = ChildFactory()
            self.run_with(factory, ["--run-dir", str(directory), "--resume-experiment"])
            self.assertEqual(factory.commands, [])
            report_path = directory / "eval-000000008.json"
            report = json.loads(report_path.read_text())
            report["evaluation_seed"] = 99
            report_path.write_text(json.dumps(report))
            factory = ChildFactory()
            self.run_with(factory, ["--run-dir", str(directory), "--resume-experiment"])
            self.assertEqual(len(factory.commands), 1)
            self.assertIn("model.step-000000008.pth", option(factory.commands[0], "--model-path"))
            with (directory / "comparison.csv").open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)

    def test_fresh_overwrite_and_changed_seed_are_rejected_without_new_children(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self.run_with(ChildFactory(), self.arguments(directory))
            before = (directory / "experiment.config.json").read_bytes()
            with self.assertRaises(SystemExit):
                self.run_with(ChildFactory(), self.arguments(directory))
            with self.assertRaises(SystemExit):
                self.run_with(ChildFactory(), ["--run-dir", str(directory),
                                             "--resume-experiment", "--seed", "123"])
            self.assertEqual((directory / "experiment.config.json").read_bytes(), before)

    def test_activation_and_determinism_are_immutable_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self.run_with(ChildFactory(), self.arguments(directory))
            config_path = directory / "experiment.config.json"
            before = config_path.read_bytes()
            for override in (["--activation", "relu"], ["--no-deterministic"]):
                with self.subTest(override=override):
                    factory = ChildFactory()
                    with self.assertRaises(SystemExit):
                        self.run_with(factory, ["--run-dir", str(directory),
                                                "--resume-experiment", *override])
                    self.assertEqual(factory.commands, [])
                    self.assertEqual(config_path.read_bytes(), before)

    def test_legacy_config_checkpoint_and_cached_reports_keep_old_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self.run_with(ChildFactory(), [*self.arguments(directory),
                                         "--activation", "relu", "--no-deterministic"])
            legacy_paths = [directory / "experiment.config.json", directory / "model.training.pt",
                            *directory.glob("model.step-*.config.json"),
                            *directory.glob("eval-*.json")]
            for path in legacy_paths:
                document = json.loads(path.read_text())
                document.pop("activation")
                document.pop("deterministic")
                path.write_text(json.dumps(document))
            resume = ["--run-dir", str(directory), "--resume-experiment"]
            for override in (["--activation", "leaky-relu"], ["--deterministic"]):
                with self.subTest(override=override):
                    with self.assertRaises(SystemExit):
                        experiment.parse_args([*resume, *override])
            parsed = experiment.parse_args(resume)
            self.assertEqual(parsed.activation, "relu")
            self.assertIs(parsed.deterministic, False)
            factory = ChildFactory()
            self.run_with(factory, resume)
            self.assertEqual(factory.commands, [])
            report_path = directory / "eval-000000004.json"
            report = json.loads(report_path.read_text())
            report["max_steps"] = 999
            report_path.write_text(json.dumps(report))
            self.run_with(factory, resume)
            self.assertEqual(len(factory.commands), 1)
            self.assertEqual(option(factory.commands[0], "--activation"), "relu")
            self.assertIn("--no-deterministic", factory.commands[0])

    def test_wrong_runtime_in_evaluation_cache_is_retested(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self.run_with(ChildFactory(), self.arguments(directory))
            for key, value in (("activation", "relu"), ("deterministic", False)):
                with self.subTest(key=key):
                    report_path = directory / "eval-000000008.json"
                    report = json.loads(report_path.read_text())
                    report[key] = value
                    report_path.write_text(json.dumps(report))
                    factory = ChildFactory()
                    self.run_with(factory, ["--run-dir", str(directory), "--resume-experiment"])
                    self.assertEqual(len(factory.commands), 1)
                    self.assertIn("model.step-000000008.pth", option(factory.commands[0], "--model-path"))

    def test_changed_checkpoint_runtime_is_rejected_before_starting_children(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self.run_with(ChildFactory(), self.arguments(directory))
            checkpoint_path = directory / "model.training.pt"
            original = json.loads(checkpoint_path.read_text())
            for key, value in (("activation", "relu"), ("deterministic", False)):
                with self.subTest(key=key):
                    checkpoint_path.write_text(json.dumps({**original, key: value}))
                    factory = ChildFactory()
                    with self.assertRaisesRegex(ValueError, "checkpoint does not match"):
                        self.run_with(factory, ["--run-dir", str(directory), "--resume-experiment"])
                    self.assertEqual(factory.commands, [])

    def test_missing_full_checkpoint_cannot_silently_restart_completed_training(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self.run_with(ChildFactory(), self.arguments(directory))
            (directory / "model.training.pt").unlink()
            factory = ChildFactory()
            with self.assertRaisesRegex(ValueError, "checkpoint is missing or older"):
                self.run_with(factory, ["--run-dir", str(directory), "--resume-experiment"])
            self.assertEqual(factory.commands, [])

    def test_training_failure_does_not_advance_to_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            factory = ChildFactory(fail="train")
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_with(factory, self.arguments(directory))
            self.assertEqual(len(factory.commands), 1)
            progress = json.loads((Path(directory) / "experiment.progress.json").read_text())
            self.assertEqual(progress["status"], "failed")
            self.assertEqual(progress["completed_milestones"], [])

    def test_default_plan_and_invalid_milestone_order(self):
        with tempfile.TemporaryDirectory() as directory:
            args = experiment.parse_args(["--run-dir", directory])
            self.assertEqual(args.activation, "leaky-relu")
            self.assertIs(args.deterministic, True)
            self.assertEqual(args.milestones, [1_000_000, 2_000_000, 4_000_000,
                                              7_000_000, 10_000_000, 15_000_000])
            with self.assertRaises(SystemExit):
                experiment.parse_args(["--run-dir", directory, "--milestones", "4,4,8"])

    def test_training_variants_are_forwarded_and_immutable(self):
        settings = ["--replay-storage", "frames", "--terminal-on-life-loss",
                    "--buffer-capacity", "500000", "--epsilon-decay-steps", "1000000",
                    "--lr", "0.00005", "--gamma", "0.995", "--train-freq", "2",
                    "--reward-shaping", "rally", "--bounce-bonus", "1.5",
                    "--rally-bonus-step", "0.3", "--rally-bonus-max-multiplier", "2.5"]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            factory = ChildFactory()
            self.run_with(factory, [*self.arguments(directory), *settings])
            first = factory.commands[0]
            self.assertEqual(option(first, "--replay-storage"), "frames")
            self.assertIn("--terminal-on-life-loss", first)
            self.assertEqual(option(first, "--buffer-capacity"), "500000")
            self.assertEqual(option(first, "--epsilon-decay-steps"), "1000000")
            self.assertEqual(float(option(first, "--lr")), 5e-5)
            self.assertEqual(float(option(first, "--gamma")), 0.995)
            self.assertEqual(option(first, "--train-freq"), "2")
            self.assertEqual(option(first, "--reward-shaping"), "rally")
            for name, expected in (("bounce_bonus", 1.5), ("rally_bonus_step", 0.3),
                                   ("rally_bonus_max_multiplier", 2.5)):
                self.assertEqual(float(option(first, "--" + name.replace("_", "-"))), expected)
            resume = ["--run-dir", str(directory), "--resume-experiment"]
            parsed = experiment.parse_args(resume)
            self.assertEqual(parsed.buffer_capacity, 500_000)
            self.assertTrue(parsed.terminal_on_life_loss)
            for document in (json.loads((directory / "experiment.config.json").read_text()),
                             json.loads((directory / "experiment.progress.json").read_text())["training_settings"],
                             json.loads((directory / "model.step-000000012.config.json").read_text())):
                for name in experiment.TRAINING_DEFAULTS:
                    self.assertEqual(document[name], getattr(parsed, name))
            factory = ChildFactory()
            self.run_with(factory, resume)
            self.assertEqual(factory.commands, [])
            for override in (["--replay-storage", "deque"], ["--no-terminal-on-life-loss"],
                             ["--buffer-capacity", "50000"], ["--epsilon-decay-steps", "250000"],
                             ["--lr", "0.0001"], ["--gamma", "0.99"], ["--train-freq", "4"],
                             ["--reward-shaping", "none"], ["--bounce-bonus", "1"],
                             ["--rally-bonus-step", "0.2"], ["--rally-bonus-max-multiplier", "2"]):
                with self.subTest(override=override), self.assertRaises(SystemExit):
                    experiment.parse_args([*resume, *override])

    def test_changed_checkpoint_training_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self.run_with(ChildFactory(), self.arguments(directory))
            checkpoint = directory / "model.training.pt"
            saved = json.loads(checkpoint.read_text())
            for name, value in (("replay_storage", "frames"), ("terminal_on_life_loss", True),
                                ("buffer_capacity", 200000), ("epsilon_decay_steps", 1000000), ("lr", 5e-5),
                                ("gamma", 0.995), ("train_freq", 2), ("reward_shaping", "rally"),
                                ("bounce_bonus", 2), ("rally_bonus_step", 0.3),
                                ("rally_bonus_max_multiplier", 3)):
                with self.subTest(name=name):
                    checkpoint.write_text(json.dumps({**saved, name: value}))
                    factory = ChildFactory()
                    with self.assertRaisesRegex(ValueError, "training settings"):
                        self.run_with(factory, ["--run-dir", str(directory), "--resume-experiment"])
                    self.assertEqual(factory.commands, [])

    def test_invalid_revision_settings_are_rejected_before_creating_a_run(self):
        invalid = [("--gamma", "nan"), ("--gamma", "1.1"), ("--gamma", "-0.1"),
                   ("--train-freq", "0"), ("--train-freq", "-1"),
                   ("--reward-shaping", "per"),
                   ("--bounce-bonus", "nan"), ("--bounce-bonus", "-1"),
                   ("--rally-bonus-step", "inf"), ("--rally-bonus-step", "-0.1"),
                   ("--rally-bonus-max-multiplier", "nan"),
                   ("--rally-bonus-max-multiplier", "0.9")]
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "not-created"
            for option_name, value in invalid:
                with self.subTest(option=option_name, value=value), self.assertRaises(SystemExit):
                    experiment.parse_args(["--run-dir", str(run), option_name, value])
            self.assertFalse(run.exists())

    def test_changed_reward_recipe_in_weights_invalidates_cached_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self.run_with(ChildFactory(), self.arguments(directory))
            weights = directory / "model.step-000000008.pth"
            training_path = weights.with_suffix(".config.json")
            training = json.loads(training_path.read_text())
            training["reward_shaping"] = "rally"
            training_path.write_text(json.dumps(training))
            args = experiment.parse_args(["--run-dir", str(directory), "--resume-experiment"])
            self.assertIsNone(experiment._evaluation_report(
                directory / "eval-000000008.json", weights, 8, args, cached=True))

    def test_missing_new_fields_resume_with_legacy_training_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self.run_with(ChildFactory(), self.arguments(directory))
            for path in (directory / "experiment.config.json", directory / "model.training.pt",
                         *directory.glob("model.step-*.config.json")):
                saved = json.loads(path.read_text())
                for name in experiment.TRAINING_DEFAULTS:
                    saved.pop(name, None)
                path.write_text(json.dumps(saved))
            factory = ChildFactory()
            self.run_with(factory, ["--run-dir", str(directory), "--resume-experiment"])
            self.assertEqual(factory.commands, [])

    def test_surviving_child_blocks_resume_without_touching_progress_or_config(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self.run_with(ChildFactory(), self.arguments(directory))
            progress_path = directory / "experiment.progress.json"
            previous = json.loads(progress_path.read_text())
            previous.update(status="training", child_pid=12345)
            progress_path.write_text(json.dumps(previous))
            before_progress = progress_path.read_bytes()
            before_config = (directory / "experiment.config.json").read_bytes()
            factory = ChildFactory()
            with patch.object(experiment, "_process_alive", return_value=True) as alive:
                with self.assertRaisesRegex(RuntimeError, "still running"):
                    self.run_with(factory, ["--run-dir", str(directory), "--resume-experiment"])
            alive.assert_called_once_with(12345)
            self.assertEqual(factory.commands, [])
            self.assertEqual(progress_path.read_bytes(), before_progress)
            self.assertEqual((directory / "experiment.config.json").read_bytes(), before_config)

    def test_process_check_recognizes_current_process(self):
        self.assertTrue(experiment._process_alive(os.getpid()))

    def test_interrupted_child_records_achieved_cleanup_scope(self):
        child = Mock(pid=12345)
        child.wait.side_effect = KeyboardInterrupt
        child.poll.return_value = None
        cleanup = {"cleanup_scope": "direct-child", "cleanup_method": "terminate",
                   "tree_cleanup_error": "access denied"}

        def clean(process):
            self.assertIs(process, child)
            child.poll.return_value = 1
            return cleanup

        updates = []
        with patch.object(experiment.subprocess, "Popen", return_value=child), patch.object(
                experiment, "terminate_process_tree", side_effect=clean):
            with self.assertRaises(KeyboardInterrupt):
                experiment._run_child(["owned-child"], lambda **changes: updates.append(changes))
        self.assertIn({"child_cleanup": cleanup}, updates)
        self.assertEqual(updates[-1], {"child_pid": None})

    def test_cleanup_failure_preserves_the_live_child_pid(self):
        child = Mock(pid=12345)
        child.wait.side_effect = KeyboardInterrupt
        child.poll.return_value = None
        updates = []
        with patch.object(experiment.subprocess, "Popen", return_value=child), patch.object(
                experiment, "terminate_process_tree", side_effect=RuntimeError("cleanup failed")):
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                experiment._run_child(["owned-child"], lambda **changes: updates.append(changes))
        self.assertEqual(updates, [{"child_pid": 12345, "child_cleanup": None}])


if __name__ == "__main__":
    unittest.main()
