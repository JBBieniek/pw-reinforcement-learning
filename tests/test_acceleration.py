"""Safe-boundary handoff checks without touching a real training process."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiments.ConceptCheck import accelerate_experiment as acceleration


class AccelerationTests(unittest.TestCase):
    def fixture(self, directory):
        directory = Path(directory).resolve()
        config = {"format_version": 1, "milestones": [8, 16], "algorithm": "double-dqn",
                  "seed": 230849415, "activation": "leaky-relu", "deterministic": True,
                  "max_steps_per_second": 2.0, "device": "cpu", "previous_targets": [],
                  "milestone_extensions": [], "created_at_utc": "2026-11-12T00:00:00Z"}
        identity = {"pid": 100, "creation_time": "2026-11-12T00:00:00Z", "executable": sys.executable,
                    "command_line": subprocess.list2cmdline([sys.executable, "-u", "-m",
                                    "experiments.ConceptCheck.experiment", "--run-dir", str(directory)])}
        request = {"format_version": 1, "run_dir": str(directory), "at_step": 8,
                   "then_append_milestones": [24, 32], "max_steps_per_second": 0.0,
                   "baseline_config": config,
                   "baseline_config_sha256": acceleration.continuation._identity(config),
                   "original_controller": identity}
        self.write(directory / "experiment.config.json", config)
        self.write(directory / "experiment.progress.json", {"pid": 100, "child_pid": 101,
                   "status": "evaluating", "current_milestone": 8})
        self.write(directory / "model.progress.json", {"status": "paused", "step": 8})
        (directory / "model.training.pt").write_bytes(b"bounded fake full snapshot at 8")
        weights = directory / "model.step-000000008.pth"
        weights.write_bytes(b"fake immutable weights")
        self.write(weights.with_suffix(".config.json"), {**config, "steps": 16})
        return directory, request

    def write(self, path, document):
        path.write_text(json.dumps(document), encoding="utf-8")

    def arguments(self, directory):
        return ["--run-dir", str(directory), "--at-step", "8", "--max-steps-per-second", "0",
                "--then-append-milestones", "24,32", "--poll-seconds", "1"]

    def boundary(self, directory, request):
        with patch.object(acceleration.experiment, "_validated_checkpoint", return_value=8):
            return acceleration._boundary(directory, request)

    def test_cli_rejects_invalid_threshold_pacing_and_polling(self):
        with tempfile.TemporaryDirectory() as directory:
            args = acceleration.parse_args(self.arguments(directory))
            self.assertEqual(args.max_steps_per_second, 0)
            self.assertEqual(args.then_append_milestones, [24, 32])
            for override in (["--at-step", "0"], ["--max-steps-per-second", "nan"],
                             ["--max-steps-per-second", "-1"], ["--poll-seconds", "0"]):
                with self.subTest(override=override), self.assertRaises(SystemExit):
                    acceleration.parse_args([*self.arguments(directory), *override])

    def test_identity_requires_exact_module_run_directory_and_creation_time(self):
        with tempfile.TemporaryDirectory() as directory:
            directory, request = self.fixture(directory)
            identity = request["original_controller"]
            acceleration._validate_controller(identity, directory)
            for identity_change in (
                    {"creation_time": "new unrelated process"},
                    {"command_line": identity["command_line"].replace("ConceptCheck.experiment", "ConceptCheck.train")},
                    {"command_line": identity["command_line"].replace(str(directory), str(directory / "other"))}):
                with self.subTest(change=identity_change), self.assertRaises((ValueError, RuntimeError)):
                    acceleration._same_identity(identity, {**identity, **identity_change}, directory)

    def test_paused_file_cannot_authorize_stopping_the_next_training_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            directory, request = self.fixture(directory)
            self.assertEqual(acceleration._paused_step(directory, request), 8)
            self.write(directory / "experiment.progress.json", {"pid": 100, "child_pid": 102,
                       "status": "training", "current_milestone": 16})
            self.assertIsNone(acceleration._paused_step(directory, request))
            self.write(directory / "model.progress.json", {"status": "running", "step": 9})
            self.assertIsNone(acceleration._paused_step(directory, request))

    def test_full_checkpoint_and_milestone_metadata_must_match_the_pause(self):
        with tempfile.TemporaryDirectory() as directory:
            directory, request = self.fixture(directory)
            with patch.object(acceleration.experiment, "_validated_checkpoint", return_value=7):
                with self.assertRaisesRegex(ValueError, "steps differ"):
                    acceleration._boundary(directory, request)
            sidecar = directory / "model.step-000000008.config.json"
            config = json.loads(sidecar.read_text())
            config["seed"] = 999
            self.write(sidecar, config)
            with patch.object(acceleration.experiment, "_validated_checkpoint", return_value=8):
                with self.assertRaisesRegex(ValueError, "seed"):
                    acceleration._boundary(directory, request)

    def test_archive_is_atomic_and_preserves_the_full_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            directory, request = self.fixture(directory)
            queue = directory / "acceleration"
            queue.mkdir()
            boundary = self.boundary(directory, request)
            result = acceleration._archive(directory, queue, request, boundary)
            self.assertEqual(Path(result["snapshot_path"]).read_bytes(), (directory / "model.training.pt").read_bytes())
            self.assertFalse(list(queue.glob("*.tmp")))
            self.assertEqual(result, acceleration._archive(directory, queue, request, boundary))

    def test_identity_mismatch_never_calls_taskkill(self):
        with tempfile.TemporaryDirectory() as directory:
            directory, request = self.fixture(directory)
            boundary = self.boundary(directory, request)
            changed = {**request["original_controller"], "creation_time": "different"}
            with patch.object(acceleration, "_process_identity", return_value=changed), patch.object(
                    acceleration.subprocess, "run") as run:
                with self.assertRaisesRegex(RuntimeError, "identity changed"):
                    acceleration._stop_original(directory, request, boundary)
            run.assert_not_called()

    def test_training_advancing_during_final_cim_query_never_calls_taskkill(self):
        with tempfile.TemporaryDirectory() as directory:
            directory, request = self.fixture(directory)
            boundary = self.boundary(directory, request)

            def query_then_advance(pid):
                self.write(directory / "model.progress.json", {"status": "running", "step": 9})
                self.write(directory / "experiment.progress.json", {"pid": 100, "child_pid": 102,
                           "status": "training", "current_milestone": 16})
                return request["original_controller"]

            with patch.object(acceleration, "_process_identity", side_effect=query_then_advance), patch.object(
                    acceleration.subprocess, "run") as run:
                self.assertFalse(acceleration._stop_original(directory, request, boundary))
            run.assert_not_called()

    def test_taskkill_failure_aborts_without_direct_child_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            directory, request = self.fixture(directory)
            boundary = self.boundary(directory, request)
            with patch.object(acceleration, "_process_identity", return_value=request["original_controller"]), patch.object(
                    acceleration.subprocess, "run", return_value=Mock(returncode=5, stderr="Access denied", stdout="")), patch.object(
                    acceleration.subprocess, "Popen") as launch:
                with self.assertRaisesRegex(RuntimeError, "Access denied"):
                    acceleration._stop_original(directory, request, boundary)
            launch.assert_not_called()

    def test_taskkill_targets_only_the_verified_real_controller_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            directory, request = self.fixture(directory)
            boundary = self.boundary(directory, request)
            with patch.object(acceleration, "_process_identity", return_value=request["original_controller"]), patch.object(
                    acceleration.subprocess, "run", return_value=Mock(returncode=0)) as run, patch.object(
                    acceleration.experiment, "_process_alive", return_value=False):
                self.assertTrue(acceleration._stop_original(directory, request, boundary))
            self.assertEqual(run.call_args.args[0][1:], ["/PID", "100", "/T", "/F"])

    def test_running_training_above_threshold_is_observed_without_stop_or_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            directory, request = self.fixture(directory)
            self.write(directory / "model.progress.json", {"status": "running", "step": 9})
            with patch.object(acceleration.experiment, "_process_alive", return_value=True), patch.object(
                    acceleration.time, "sleep", side_effect=KeyboardInterrupt), patch.object(
                    acceleration, "_stop_original") as stop, patch.object(acceleration, "_archive") as archive:
                with self.assertRaises(KeyboardInterrupt):
                    acceleration._wait_and_handoff(directory, directory, request, 1, Mock())
            stop.assert_not_called()
            archive.assert_not_called()

    def test_dead_original_before_boundary_fails_without_restarting(self):
        with tempfile.TemporaryDirectory() as directory:
            directory, request = self.fixture(directory)
            with patch.object(acceleration.experiment, "_process_alive", return_value=False), patch.object(
                    acceleration.subprocess, "Popen") as launch:
                with self.assertRaisesRegex(RuntimeError, "exited before handoff"):
                    acceleration._wait_and_handoff(directory, directory, request, 1, Mock())
            launch.assert_not_called()

    def test_existing_live_acceleration_child_blocks_a_duplicate_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            directory, request = self.fixture(directory)
            queue = directory / "acceleration"
            queue.mkdir()
            path = queue / "acceleration.progress.json"
            self.write(path, {"status": "resuming", "child_pid": 777})
            before = path.read_bytes()
            with patch.object(acceleration, "_process_identity", return_value=request["original_controller"]), patch.object(
                    acceleration.experiment, "_process_alive", return_value=True), patch.object(
                    acceleration.subprocess, "Popen") as launch:
                with self.assertRaisesRegex(RuntimeError, "live child"):
                    acceleration.run_acceleration(self.arguments(directory))
            self.assertEqual(path.read_bytes(), before)
            launch.assert_not_called()

    def test_unrecorded_live_replacement_controller_blocks_duplicate_after_a_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            directory, request = self.fixture(directory)
            queue = directory / "acceleration"
            queue.mkdir()
            self.write(queue / "acceleration.request.json", request)
            self.write(directory / "experiment.progress.json", {"pid": 200, "child_pid": None,
                       "status": "training", "current_milestone": 16})
            with patch.object(acceleration.experiment, "_process_alive", return_value=True), patch.object(
                    acceleration.subprocess, "Popen") as launch:
                with self.assertRaisesRegex(RuntimeError, "another experiment controller"):
                    acceleration.run_acceleration(self.arguments(directory))
            launch.assert_not_called()

    def test_completed_request_retry_verifies_results_without_launching_children(self):
        with tempfile.TemporaryDirectory() as directory:
            directory, request = self.fixture(directory)
            queue = directory / "acceleration"
            queue.mkdir()
            self.write(queue / "acceleration.request.json", request)
            config = {**acceleration._resumed_config(request), "milestones": [8, 16, 24, 32],
                      "previous_targets": [16], "milestone_extensions": [{"from_milestones": [8, 16],
                          "appended_milestones": [24, 32], "registered_at_utc": "later"}]}
            self.write(directory / "experiment.config.json", config)
            self.write(directory / "experiment.progress.json", {"pid": 300, "child_pid": None,
                       "status": "completed", "completed_milestones": [8, 16, 24, 32]})
            with patch.object(acceleration.experiment, "_process_alive", return_value=False), patch.object(
                    acceleration, "_check_complete") as complete, patch.object(
                    acceleration.subprocess, "Popen") as launch:
                result = acceleration.run_acceleration(self.arguments(directory))
            self.assertEqual(result["status"], "completed")
            complete.assert_called_once_with(directory, config)
            launch.assert_not_called()

    def test_dead_controller_after_persisted_handoff_intent_recovers_without_a_second_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            directory, request = self.fixture(directory)
            queue = directory / "acceleration"
            queue.mkdir()
            self.write(queue / "acceleration.request.json", request)
            boundary = self.boundary(directory, request)
            archive = acceleration._archive(directory, queue, request, boundary)
            self.write(queue / "acceleration.progress.json", {"status": "stopping_controller",
                       "handoff_intent": True, **archive})
            commands = []

            def phase(command, phase, poll_seconds, update):
                commands.append(command)
                config = acceleration._resumed_config(request)
                if phase == "continuing":
                    config = {**config, "milestones": [8, 16, 24, 32], "previous_targets": [16],
                              "milestone_extensions": [{"from_milestones": [8, 16],
                                  "appended_milestones": [24, 32], "registered_at_utc": "later"}]}
                self.write(directory / "experiment.config.json", config)
                self.write(directory / "experiment.progress.json", {"pid": 200, "child_pid": None,
                           "status": "completed", "completed_milestones": config["milestones"]})

            with patch.object(acceleration.experiment, "_process_alive", return_value=False), patch.object(
                    acceleration, "_run_phase", side_effect=phase), patch.object(
                    acceleration, "_check_complete"), patch.object(acceleration, "_wait_and_handoff") as handoff:
                result = acceleration.run_acceleration(self.arguments(directory))
            self.assertTrue(result["handoff_complete"])
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(commands), 2)
            handoff.assert_not_called()

    def test_unchanged_pacing_hyperparameters_and_final_extension_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            directory, request = self.fixture(directory)
            commands = []

            def handoff(directory, queue, request, poll_seconds, update):
                update(handoff_complete=True, status="ready_to_resume")

            def phase(command, phase, poll_seconds, update):
                commands.append(command)
                config = acceleration._resumed_config(request)
                if phase == "continuing":
                    config = {**config, "milestones": [8, 16, 24, 32], "previous_targets": [16],
                              "milestone_extensions": [{"from_milestones": [8, 16],
                                  "appended_milestones": [24, 32], "registered_at_utc": "later"}]}
                self.write(directory / "experiment.config.json", config)
                self.write(directory / "experiment.progress.json", {"pid": 200, "child_pid": None,
                           "status": "completed", "completed_milestones": config["milestones"]})

            with patch.object(acceleration, "_process_identity", return_value=request["original_controller"]), patch.object(
                    acceleration.experiment, "_process_alive", return_value=False), patch.object(
                    acceleration, "_wait_and_handoff", side_effect=handoff), patch.object(
                    acceleration, "_run_phase", side_effect=phase), patch.object(acceleration, "_check_complete"):
                result = acceleration.run_acceleration(self.arguments(directory))
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(commands), 2)
            self.assertIn("experiments.ConceptCheck.experiment", commands[0])
            self.assertIn("--resume-experiment", commands[0])
            self.assertIn("experiments.ConceptCheck.continue_experiment", commands[1])
            for command in commands:
                self.assertEqual(command[command.index("--max-steps-per-second") + 1], "0.0")
                self.assertNotIn("--epsilon-decay-steps", command)
                self.assertNotIn("--lr", command)


if __name__ == "__main__":
    unittest.main()
