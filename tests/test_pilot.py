"""Bounded conservative-variant queue checks without real Atari or GPU work."""

import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from experiments.ConceptCheck import experiment, pilot
from test_experiment import ChildFactory, option


class PilotTests(unittest.TestCase):
    def arguments(self, directory):
        return ["--run-dir", str(directory), "--milestones", "4,8",
                "--eval-episodes", "2", "--max-steps", "32", "--device", "cpu"]

    def run_with(self, arguments, *, fail_variant=None, controllers=None, children=None):
        controllers = [] if controllers is None else controllers
        children = ChildFactory() if children is None else children
        actual_run_child = experiment._run_child

        def run_child(command, update, *, cwd=None):
            if "experiments.ConceptCheck.experiment" not in command:
                return actual_run_child(command, update, cwd=cwd)
            controllers.append(command)
            self.assertEqual(Path(cwd), Path(option(command, "--run-dir")).parent / "source")
            self.assertIn("-B", command)
            variant = Path(option(command, "--run-dir")).name
            if variant == fail_variant:
                raise subprocess.CalledProcessError(7, command)
            update(child_pid=12345)
            experiment.run_experiment(command[command.index("experiments.ConceptCheck.experiment") + 1:])
            update(child_pid=None)

        with patch.object(experiment, "_run_child", side_effect=run_child), patch.object(
                experiment.subprocess, "Popen", children), patch.object(
                experiment, "_read_training_checkpoint", side_effect=lambda path: json.loads(path.read_text())), contextlib.redirect_stdout(io.StringIO()):
            return pilot.run_pilot(arguments)

    def test_four_variants_differ_one_setting_and_produce_verified_results(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            controllers = []
            result = self.run_with(self.arguments(directory), controllers=controllers)
            self.assertEqual(result["status"], "completed")
            self.assertIsNone(result["child_pid"])
            self.assertIsNone(result["current_variant"])
            names = [name for name, _ in pilot.VARIANTS]
            self.assertEqual(result["completed_variants"], names)
            self.assertEqual(len(controllers), 4)
            config = json.loads((directory / "pilot.config.json").read_text())
            base = config["variants"][0]["training"]
            self.assertEqual(base["replay_storage"], "frames")
            for variant in config["variants"][1:]:
                differences = [name for name, value in variant["training"].items() if value != base[name]]
                self.assertEqual(len(differences), 1)
            for command in controllers:
                self.assertEqual(option(command, "--seed"), "230849415")
                self.assertEqual(option(command, "--eval-seed"), "3303497698")
                self.assertEqual(option(command, "--max-steps-per-second"), "0.0")
            with (directory / "comparison.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 8)
            self.assertEqual([row["variant"] for row in rows], [name for name in names for _ in range(2)])
            archive = directory / "source.zip"
            self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), config["source_zip_sha256"])
            with zipfile.ZipFile(archive) as source:
                self.assertEqual(set(source.namelist()), set(config["source_files_sha256"]))
                for name, digest in config["source_files_sha256"].items():
                    self.assertEqual(hashlib.sha256(source.read(name)).hexdigest(), digest)
                    self.assertEqual(hashlib.sha256((directory / "source" / name).read_bytes()).hexdigest(), digest)

    def test_failed_variant_stops_queue_and_resume_reuses_completed_training(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            controllers = []
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_with(self.arguments(directory), fail_variant="life_loss", controllers=controllers)
            progress = json.loads((directory / "pilot.progress.json").read_text())
            self.assertEqual(progress["status"], "failed")
            self.assertEqual(progress["completed_variants"], ["control"])
            self.assertEqual(len(controllers), 2)
            before = (directory / "control" / "model.training.pt").read_bytes()
            children = ChildFactory()
            result = self.run_with(["--run-dir", str(directory), "--resume"], children=children)
            self.assertEqual(result["status"], "completed")
            self.assertEqual((directory / "control" / "model.training.pt").read_bytes(), before)
            trained_directories = {Path(option(command, "--save-path")).parent.name
                                   for command in children.commands if "experiments.ConceptCheck.train" in command}
            self.assertNotIn("control", trained_directories)
            self.assertEqual(trained_directories, {"life_loss", "replay200k", "slow_exploration"})

    def test_changed_plan_cannot_resume_but_live_tree_changes_do_not_change_frozen_source(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self.run_with(self.arguments(directory))
            before = (directory / "pilot.progress.json").read_bytes()
            with self.assertRaises(SystemExit):
                self.run_with(["--run-dir", str(directory), "--resume", "--seed", "123"])
            frozen = directory / "source" / "experiments" / "ConceptCheck" / "train.py"
            frozen_before = frozen.read_bytes()
            with patch.object(pilot, "_source_manifest", return_value={"changed": "source"}):
                result = self.run_with(["--run-dir", str(directory), "--resume"])
            self.assertEqual(result["status"], "completed")
            self.assertEqual(frozen.read_bytes(), frozen_before)

    def test_frozen_file_tamper_blocks_resume_before_starting_children(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self.run_with(self.arguments(directory))
            before = (directory / "pilot.progress.json").read_bytes()
            frozen = directory / "source" / "experiments" / "ConceptCheck" / "train.py"
            frozen.write_bytes(b"changed training implementation")
            controllers = []
            with self.assertRaisesRegex(ValueError, "frozen pilot source"):
                self.run_with(["--run-dir", str(directory), "--resume"], controllers=controllers)
            self.assertEqual(controllers, [])
            self.assertEqual((directory / "pilot.progress.json").read_bytes(), before)

    def test_source_manifest_paths_cannot_escape_frozen_directory(self):
        for relative in ("../outside.py", "/absolute.py", "C:/outside.py", "folder\\outside.py"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                archive = directory / "source.zip"
                data = b"untrusted"
                with zipfile.ZipFile(archive, "w") as source:
                    source.writestr(relative, data)
                config = {"source_zip_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                          "source_files_sha256": {relative: hashlib.sha256(data).hexdigest()}}
                with self.assertRaisesRegex(ValueError, "unsafe path"):
                    pilot._frozen_source(directory, config)
                self.assertFalse((directory / "source").exists())
                self.assertEqual(list(directory.iterdir()), [archive])

    def test_corrupt_archive_and_surviving_child_block_resume_without_progress_write(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self.run_with(self.arguments(directory))
            path = directory / "pilot.progress.json"
            previous = json.loads(path.read_text())
            previous["child_pid"] = 12345
            path.write_text(json.dumps(previous))
            before = path.read_bytes()
            with patch.object(experiment, "_process_alive", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "still running"):
                    self.run_with(["--run-dir", str(directory), "--resume"])
            self.assertEqual(path.read_bytes(), before)
            (directory / "source.zip").write_bytes(b"broken")
            with self.assertRaisesRegex(ValueError, "archive"):
                self.run_with(["--run-dir", str(directory), "--resume"])
            self.assertEqual(path.read_bytes(), before)

    def test_existing_baseline_and_duplicate_pilot_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            marker = directory / "experiment.config.json"
            marker.write_text("baseline")
            with self.assertRaisesRegex(ValueError, "already contains"):
                self.run_with(self.arguments(directory))
            self.assertEqual(marker.read_text(), "baseline")
            self.assertFalse((directory / "pilot.config.json").exists())
        with tempfile.TemporaryDirectory() as directory:
            self.run_with(self.arguments(directory))
            with self.assertRaises(SystemExit):
                self.run_with(self.arguments(directory))

    def test_lock_prevents_second_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            with experiment._experiment_lock(Path(directory)):
                with self.assertRaisesRegex(RuntimeError, "another experiment"):
                    self.run_with(self.arguments(directory))
            self.assertFalse((Path(directory) / "pilot.config.json").exists())


if __name__ == "__main__":
    unittest.main()
