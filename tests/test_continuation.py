"""Continuation queue checks without starting training or the Atari emulator."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from experiments.ConceptCheck import continue_experiment as continuation
from experiments.ConceptCheck import experiment


ENVIRONMENT = {"mode": 0, "difficulty": 0, "frame_skip": 4,
               "sticky_action_probability": 0.25}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.config_path = self.directory / "experiment.config.json"
        self.progress_path = self.directory / "experiment.progress.json"
        self.checkpoint = self.directory / "model.training.pt"
        self.queue = self.directory / "continuation"
        self.config = {
            "format_version": 1, "run_dir": str(self.directory.resolve()),
            "algorithm": "double-dqn", "seed": 391827561, "eval_seed": 107292681,
            "eval_episodes": 2, "max_steps": 32, "device": "cpu",
            "activation": "leaky-relu", "deterministic": True,
            "milestones": [4, 8, 12], "previous_targets": [], "milestone_extensions": [],
            "max_steps_per_second": 220.0, "created_at_utc": "2026-11-12T01:00:00+00:00",
        }
        write(self.config_path, self.config)
        write(self.checkpoint, self.saved_checkpoint(12, 12))
        self.original_checkpoint = self.checkpoint.read_bytes()
        self.progress = {"status": "completed", "pid": 7410, "child_pid": None,
                         "completed_milestones": [4, 8, 12]}
        write(self.progress_path, self.progress)
        self.evaluations(self.config["milestones"])
        self.commands = []
        self.live = set()
        self.enterContext(patch.object(experiment, "_read_training_checkpoint",
                                      side_effect=lambda path: read(path)))
        self.enterContext(patch.object(experiment, "_process_alive",
                                      side_effect=lambda pid: pid in self.live))

    def arguments(self, **overrides):
        options = {"--run-dir": str(self.directory), "--append-milestones": "16,20",
                   "--max-steps-per-second": "0", "--poll-seconds": "0.01", **overrides}
        return [item for pair in options.items() for item in pair]

    def saved_checkpoint(self, step, target):
        return {"step": step, "steps": target, "algorithm": self.config["algorithm"],
                "seed": self.config["seed"], "activation": self.config["activation"],
                "deterministic": self.config["deterministic"]}

    def evaluations(self, milestones):
        for milestone in milestones:
            weights = experiment._weights_path(self.directory, milestone)
            weights.write_bytes(f"policy-{milestone}".encode())
            write(weights.with_suffix(".config.json"), {**ENVIRONMENT,
                  "activation": self.config["activation"], "deterministic": self.config["deterministic"]})
            seeds = [self.config["eval_seed"] + index for index in range(self.config["eval_episodes"])]
            report = {
                "completed_at_utc": "2026-11-12T02:00:00+00:00",
                "activation": self.config["activation"], "deterministic": self.config["deterministic"],
                "evaluation_seed": self.config["eval_seed"], "episode_seeds": seeds,
                "policy_epsilon": 0.0, "auto_fire_after_life_loss": True,
                "max_steps": self.config["max_steps"], "model_path": str(weights.resolve()),
                "environment": ENVIRONMENT,
                "episodes": [{"seed": seed, "reward": milestone} for seed in seeds],
                "checkpoint_step": milestone, "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                "summary": {"episodes": len(seeds), "mean_reward": milestone, "median_reward": milestone,
                            "min_reward": milestone, "max_reward": milestone, "std_reward": 0.0,
                            "wins_864": 0, "win_rate_864": 0.0, "hit_step_limit": 0, "mean_steps": 10},
            }
            write(experiment._evaluation_path(self.directory, milestone), report)

    def extended_config(self):
        return {**self.config, "milestones": [4, 8, 12, 16, 20], "previous_targets": [12],
                "max_steps_per_second": 0.0, "milestone_extensions": [{
                    "from_milestones": [4, 8, 12], "appended_milestones": [16, 20],
                    "registered_at_utc": "2026-11-12T02:01:00+00:00",
                }]}

    def complete_child(self, command, cwd):
        self.commands.append(command)
        self.assertEqual(cwd, experiment.ROOT)
        self.assertEqual((self.directory / "model.training-at-000000012.pt").read_bytes(),
                         self.original_checkpoint)
        self.assertIn("--resume-experiment", command)
        self.assertEqual(command[command.index("--append-milestones") + 1], "16,20")
        self.assertEqual(command[command.index("--max-steps-per-second") + 1], "0.0")
        write(self.config_path, self.extended_config())
        write(self.checkpoint, self.saved_checkpoint(20, 20))
        self.evaluations([16, 20])
        write(self.progress_path, {"status": "completed", "pid": 8317, "child_pid": None,
                                   "completed_milestones": [4, 8, 12, 16, 20]})

        class Child:
            # Windows venv shim PID differs from the controller in progress.
            pid = 8318

            def poll(self):
                return 0

            def wait(self):
                return 0

        return Child()

    def run_with(self, factory=None, sleep=None, arguments=None):
        with (patch.object(continuation.subprocess, "Popen", side_effect=factory or self.complete_child),
              patch.object(continuation.time, "sleep", side_effect=sleep)):
            return continuation.run_continuation(arguments or self.arguments())

    def test_completed_baseline_preserved_then_extended_once_and_retry_is_noop(self):
        old_reports = {path: path.read_bytes() for path in self.directory.glob("eval-*.json")}
        result = self.run_with()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["expected_target"], 20)
        self.assertIsNone(result["child_pid"])
        self.assertEqual(len(self.commands), 1)
        request = (self.queue / "continuation.request.json").read_bytes()
        self.assertEqual(read(self.queue / "continuation.request.json")["baseline_config"], self.config)
        for path, original in old_reports.items():
            self.assertEqual(path.read_bytes(), original)
        repeated = self.run_with()
        self.assertEqual(repeated["status"], "completed")
        self.assertEqual(len(self.commands), 1)
        self.assertEqual((self.queue / "continuation.request.json").read_bytes(), request)
        self.assertEqual((self.directory / "model.training-at-000000012.pt").read_bytes(), self.original_checkpoint)

    def test_waits_for_completion_and_both_processes_without_touching_live_outputs(self):
        write(self.progress_path, {**self.progress, "status": "training", "child_pid": 7411,
                                   "completed_milestones": [4, 8]})
        self.live = {7410, 7411}
        config_before = self.config_path.read_bytes()
        sleeps = []

        def advance(seconds):
            sleeps.append(seconds)
            self.assertEqual(self.config_path.read_bytes(), config_before)
            self.assertEqual(self.checkpoint.read_bytes(), self.original_checkpoint)
            self.assertFalse((self.directory / "model.training-at-000000012.pt").exists())
            self.assertEqual(self.commands, [])
            if len(sleeps) == 1:
                write(self.progress_path, {**self.progress, "child_pid": 7411})
                self.live = {7411}
            else:
                self.live.clear()

        result = self.run_with(sleep=advance)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(sleeps, [0.01, 0.01])

    def test_original_controller_death_before_completion_fails_without_restart(self):
        write(self.progress_path, {**self.progress, "status": "training"})
        before = self.config_path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "exited before completing"):
            self.run_with()
        self.assertEqual(self.commands, [])
        self.assertEqual(read(self.queue / "continuation.progress.json")["status"], "failed")
        self.assertEqual(self.config_path.read_bytes(), before)
        self.assertFalse((self.directory / "model.training-at-000000012.pt").exists())

    def test_failed_original_and_invalid_cached_evaluation_do_not_start_children(self):
        for status in ("failed", "interrupted"):
            with self.subTest(status=status):
                write(self.progress_path, {**self.progress, "status": status, "error": "stopped"})
                with self.assertRaisesRegex(RuntimeError, status):
                    self.run_with()
        write(self.progress_path, self.progress)
        report = experiment._evaluation_path(self.directory, 8)
        invalid = read(report)
        invalid["checkpoint_sha256"] = "invalid"
        write(report, invalid)
        with self.assertRaisesRegex(ValueError, "evaluation at 8"):
            self.run_with()
        self.assertEqual(self.commands, [])

    def test_request_mismatch_and_changed_baseline_are_rejected_without_overwrite(self):
        write(self.progress_path, {**self.progress, "status": "failed"})
        with self.assertRaises(RuntimeError):
            self.run_with()
        request = (self.queue / "continuation.request.json").read_bytes()
        state = (self.queue / "continuation.progress.json").read_bytes()
        for override in ({"--append-milestones": "16,24"}, {"--max-steps-per-second": "300"}):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, "different continuation"):
                self.run_with(arguments=self.arguments(**override))
            self.assertEqual((self.queue / "continuation.request.json").read_bytes(), request)
            self.assertEqual((self.queue / "continuation.progress.json").read_bytes(), state)
        write(self.config_path, {**self.config, "seed": 9})
        with self.assertRaisesRegex(ValueError, "differs from this continuation"):
            self.run_with()
        self.assertEqual(self.commands, [])

    def test_explicit_retry_of_registered_extension_resumes_same_plan(self):
        def fail_after_registration(command, cwd):
            child = self.complete_child(command, cwd)
            write(self.checkpoint, self.saved_checkpoint(12, 12))
            write(self.progress_path, {"status": "failed", "pid": child.pid, "child_pid": None,
                                       "completed_milestones": [4, 8, 12]})
            child.poll = lambda: 7
            child.wait = lambda: 7
            return child

        with self.assertRaises(subprocess.CalledProcessError):
            self.run_with(factory=fail_after_registration)
        self.assertEqual(len(self.commands), 1)
        self.assertEqual(read(self.queue / "continuation.progress.json")["status"], "failed")
        result = self.run_with()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(self.commands), 2)
        self.assertEqual(read(self.config_path)["milestones"], [4, 8, 12, 16, 20])
        self.assertEqual(len(read(self.config_path)["milestone_extensions"]), 1)

    def test_live_registered_continuation_is_not_started_again(self):
        self.run_with()
        self.live.add(8317)
        with self.assertRaisesRegex(RuntimeError, "already has a live"):
            self.run_with()
        self.assertEqual(len(self.commands), 1)

    def test_duplicate_wrapper_lock_does_not_touch_queue(self):
        self.queue.mkdir()
        with experiment._experiment_lock(self.queue):
            with self.assertRaisesRegex(RuntimeError, "another experiment controller"):
                self.run_with()
        self.assertFalse((self.queue / "continuation.request.json").exists())
        self.assertEqual(self.commands, [])

    def test_interrupt_while_waiting_records_failure_without_touching_original(self):
        write(self.progress_path, {**self.progress, "status": "training"})
        self.live.add(7410)
        before = self.progress_path.read_bytes()
        with self.assertRaises(KeyboardInterrupt):
            self.run_with(sleep=KeyboardInterrupt)
        self.assertEqual(read(self.queue / "continuation.progress.json")["status"], "failed")
        self.assertEqual(self.progress_path.read_bytes(), before)
        self.assertEqual(self.commands, [])

    def test_interrupt_stops_only_the_owned_continuation_and_records_cleanup(self):
        class Child:
            pid = 8317
            returncode = None

            def poll(self):
                return self.returncode

        child = Child()
        cleanup = {"cleanup_scope": "process-tree", "cleanup_method": "test"}

        def terminate(owned):
            self.assertIs(owned, child)
            child.returncode = -15
            return cleanup

        with patch.object(experiment, "terminate_process_tree", side_effect=terminate) as stop:
            with self.assertRaises(KeyboardInterrupt):
                self.run_with(factory=lambda command, cwd: child, sleep=KeyboardInterrupt)
        stop.assert_called_once_with(child)
        saved = read(self.queue / "continuation.progress.json")
        self.assertEqual(saved["status"], "failed")
        self.assertIsNone(saved["child_pid"])
        self.assertEqual(saved["child_cleanup"], cleanup)
        self.assertEqual(self.commands, [])

    def test_missing_final_full_checkpoint_never_restarts_training(self):
        self.checkpoint.unlink()
        with self.assertRaisesRegex(ValueError, "full checkpoint"):
            self.run_with()
        self.assertEqual(read(self.queue / "continuation.progress.json")["status"], "failed")
        self.assertEqual(self.commands, [])

    def test_zero_cap_and_poll_validation(self):
        parsed = continuation.parse_args(self.arguments())
        self.assertEqual(parsed.max_steps_per_second, 0.0)
        for override in ({"--poll-seconds": "0"}, {"--poll-seconds": "nan"},
                         {"--max-steps-per-second": "-1"}, {"--max-steps-per-second": "inf"}):
            with self.subTest(override=override), self.assertRaises(SystemExit):
                continuation.parse_args(self.arguments(**override))


if __name__ == "__main__":
    unittest.main()
