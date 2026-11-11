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
            algorithm, seed = previous["algorithm"], previous["seed"]
        else:
            algorithm, seed = option(command, "--algorithm"), int(option(command, "--seed"))
        activation = option(command, "--activation")
        deterministic = "--deterministic" in command
        step = int(option(command, "--stop-at"))
        checkpoint.write_text(json.dumps({"step": step, "algorithm": algorithm,
                                         "seed": seed, "steps": int(option(command, "--steps")),
                                         "activation": activation, "deterministic": deterministic}))
        weights = directory / f"model.step-{step:09d}.pth"
        weights.write_bytes(f"{algorithm}:{seed}:{step}".encode())
        weights.with_suffix(".config.json").write_text(json.dumps(
            {**ENVIRONMENT, "activation": activation, "deterministic": deterministic}))
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
