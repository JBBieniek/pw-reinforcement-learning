"""Wait for a running experiment, preserve its final state, then extend it once."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

if __package__:
    from . import experiment
else:
    import experiment


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--append-milestones", type=experiment._milestones, required=True)
    parser.add_argument("--max-steps-per-second", type=float, default=0.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    args.run_dir = str(Path(args.run_dir).resolve())
    if not math.isfinite(args.max_steps_per_second) or args.max_steps_per_second < 0:
        parser.error("--max-steps-per-second must be finite and non-negative")
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        parser.error("--poll-seconds must be finite and positive")
    return args


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(config):
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _request(directory, queue, args):
    config = _read_json(directory / "experiment.config.json")
    path = queue / "continuation.request.json"
    if path.exists():
        request = _read_json(path)
        if (request.get("format_version") != 1 or request.get("run_dir") != args.run_dir
                or request.get("append_milestones") != args.append_milestones
                or request.get("max_steps_per_second") != args.max_steps_per_second
                or request.get("baseline_config_sha256") != _identity(request["baseline_config"])):
            raise ValueError("a different continuation request already exists; it was not overwritten")
    else:
        milestones = config["milestones"]
        if args.append_milestones[0] <= milestones[-1]:
            raise ValueError("new continuation milestones must be above the saved experiment target")
        experiment._validate_extension_history(
            milestones, config.get("previous_targets", []), config.get("milestone_extensions", []))
        request = {
            "format_version": 1, "run_dir": args.run_dir,
            "append_milestones": args.append_milestones,
            "max_steps_per_second": args.max_steps_per_second,
            "baseline_config": config, "baseline_config_sha256": _identity(config),
            "created_at_utc": experiment._timestamp(),
        }
        experiment._write_json(path, request)
    _config_stage(config, request)
    return request


def _config_stage(config, request):
    """Accept the original config or exactly this recorded extension, never a new plan."""
    baseline = request["baseline_config"]
    if config == baseline:
        return "baseline"
    expected_milestones = baseline["milestones"] + request["append_milestones"]
    history = config.get("milestone_extensions", [])
    previous = baseline.get("milestone_extensions", [])
    if (config.get("milestones") != expected_milestones or len(history) != len(previous) + 1
            or history[:-1] != previous
            or history[-1].get("from_milestones") != baseline["milestones"]
            or history[-1].get("appended_milestones") != request["append_milestones"]
            or not history[-1].get("registered_at_utc")
            or config.get("previous_targets") != baseline.get("previous_targets", []) + [baseline["milestones"][-1]]
            or config.get("max_steps_per_second") != request["max_steps_per_second"]):
        raise ValueError("experiment configuration differs from this continuation request")
    unchanged = {key: value for key, value in config.items() if key not in
                 ("milestones", "previous_targets", "milestone_extensions", "max_steps_per_second")}
    expected = {key: value for key, value in baseline.items() if key not in
                ("milestones", "previous_targets", "milestone_extensions", "max_steps_per_second")}
    if unchanged != expected:
        raise ValueError("experiment identity changed after the continuation was queued")
    experiment._validate_extension_history(
        expected_milestones, config["previous_targets"], history)
    return "extended"


def _settings(config):
    return SimpleNamespace(**{**config, "activation": config.get("activation", "relu"),
                              "deterministic": config.get("deterministic", False),
                              "previous_targets": config.get("previous_targets", [])})


def _live_processes(progress):
    pid = progress.get("pid")
    if pid is None:
        raise ValueError("experiment progress has no controller PID")
    controller_alive = experiment._process_alive(pid)
    child = progress.get("child_pid")
    child_alive = child is not None and experiment._process_alive(child)
    return controller_alive, child_alive


def _validate_evaluations(directory, config, progress, *, require_completed):
    milestones = config["milestones"]
    if require_completed and (progress.get("status") != "completed"
                              or progress.get("completed_milestones") != milestones):
        raise ValueError("experiment did not complete every requested milestone")
    args = _settings(config)
    for milestone in milestones:
        report = experiment._evaluation_report(
            experiment._evaluation_path(directory, milestone),
            experiment._weights_path(directory, milestone), milestone, args, cached=True)
        if report is None:
            raise ValueError(f"missing or incompatible completed evaluation at {milestone}")


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preserve_checkpoint(directory, request, *, extended):
    baseline = request["baseline_config"]
    final_step = baseline["milestones"][-1]
    destination = directory / f"model.training-at-{final_step:09d}.pt"
    args = _settings(baseline)
    if extended:
        if not destination.is_file() or experiment._validated_checkpoint(destination, args) != final_step:
            raise ValueError("the original final full checkpoint was not preserved before extension")
        return destination
    source = directory / "model.training.pt"
    if not source.is_file() or experiment._validated_checkpoint(source, args) != final_step:
        raise ValueError("continuation requires a full checkpoint at the original final milestone")
    if destination.exists():
        if _digest(source) != _digest(destination):
            raise ValueError("preserved full checkpoint differs from the original final checkpoint")
        return destination
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        shutil.copyfile(source, temporary)
        # Publish a complete copy before the new controller can replace the source.
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _wait_for_baseline(directory, request, poll_seconds, update):
    while True:
        config = _read_json(directory / "experiment.config.json")
        stage = _config_stage(config, request)
        progress = _read_json(directory / "experiment.progress.json")
        controller_alive, child_alive = _live_processes(progress)
        if stage == "extended":
            if controller_alive or child_alive:
                raise RuntimeError("the requested continuation already has a live controller or child")
            return config, progress, stage
        if progress.get("status") in ("failed", "interrupted"):
            raise RuntimeError(f"original experiment {progress['status']}: {progress.get('error')}")
        if progress.get("status") == "completed":
            if not controller_alive and not child_alive:
                _validate_evaluations(directory, config, progress, require_completed=True)
                return config, progress, stage
        elif not controller_alive:
            raise RuntimeError("original controller exited before completing the experiment; no training was restarted")
        update(status="waiting", waiting_for=progress.get("status"),
               original_controller_pid=progress["pid"], original_child_pid=progress.get("child_pid"),
               original_controller_alive=controller_alive, original_child_alive=child_alive)
        time.sleep(poll_seconds)


def run_continuation(argv=None):
    args = parse_args(argv)
    directory = Path(args.run_dir)
    if not (directory / "experiment.config.json").is_file():
        raise FileNotFoundError("continuation requires an existing experiment.config.json")
    queue = directory / "continuation"
    queue.mkdir(exist_ok=True)
    with experiment._experiment_lock(queue):
        request = _request(directory, queue, args)
        expected_milestones = request["baseline_config"]["milestones"] + request["append_milestones"]
        progress_path = queue / "continuation.progress.json"
        progress = {
            "status": "waiting", "pid": os.getpid(), "child_pid": None,
            "expected_target": expected_milestones[-1], "expected_milestones": expected_milestones,
            "baseline_config_sha256": request["baseline_config_sha256"],
            "started_at_utc": experiment._timestamp(), "error": None,
        }
        process = None
        last_message = None

        def update(**changes):
            nonlocal last_message
            progress.update(changes)
            progress["updated_at_utc"] = experiment._timestamp()
            experiment._write_json(progress_path, progress)
            message = (progress["status"], progress.get("waiting_for"), progress.get("error"))
            if message != last_message:
                print(f"Continuation {progress['status']} | target {progress['expected_target']:,} | "
                      f"{progress.get('error') or progress.get('waiting_for') or ''}", flush=True)
                last_message = message

        try:
            update()
            config, original, stage = _wait_for_baseline(directory, request, args.poll_seconds, update)
            with experiment._experiment_lock(directory):
                # Recheck under the experiment lock before reading/copying its final state.
                current = _read_json(directory / "experiment.config.json")
                if current != config:
                    raise RuntimeError("experiment configuration changed before continuation preflight")
                original = _read_json(directory / "experiment.progress.json")
                if any(_live_processes(original)):
                    raise RuntimeError("experiment controller or child became active before continuation")
                _validate_evaluations(directory, request["baseline_config"], original,
                                      require_completed=stage == "baseline")
                snapshot = _preserve_checkpoint(directory, request, extended=stage == "extended")
                update(snapshot_path=str(snapshot.resolve()), snapshot_sha256=_digest(snapshot))
                if stage == "extended" and original.get("status") == "completed":
                    _validate_evaluations(directory, config, original, require_completed=True)
                    if experiment._validated_checkpoint(directory / "model.training.pt", _settings(config)) != expected_milestones[-1]:
                        raise ValueError("completed continuation checkpoint has the wrong final step")
                    update(status="completed", waiting_for=None)
                    return progress
            command = [sys.executable, "-u", "-m", "experiments.ConceptCheck.experiment",
                       "--run-dir", str(directory), "--resume-experiment", "--append-milestones",
                       ",".join(map(str, request["append_milestones"])),
                       "--max-steps-per-second", str(args.max_steps_per_second)]
            process = subprocess.Popen(command, cwd=experiment.ROOT)
            update(status="continuing", waiting_for=None, child_pid=process.pid, command=command)
            while process.poll() is None:
                time.sleep(args.poll_seconds)
            returncode = process.wait()
            update(child_pid=None, child_returncode=returncode)
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
            config = _read_json(directory / "experiment.config.json")
            if _config_stage(config, request) != "extended":
                raise ValueError("continuation controller did not register the requested extension")
            final = _read_json(directory / "experiment.progress.json")
            # Windows venv launchers may be a shim whose PID differs from the
            # Python controller recorded in progress. Both must have exited.
            if any(_live_processes(final)):
                raise RuntimeError("continuation still has a live controller or child")
            _validate_evaluations(directory, config, final, require_completed=True)
            if experiment._validated_checkpoint(directory / "model.training.pt", _settings(config)) != expected_milestones[-1]:
                raise ValueError("continuation checkpoint has the wrong final step")
            update(status="completed")
            return progress
        except BaseException as error:
            cleanup = None
            if process is not None and process.poll() is None:
                try:
                    cleanup = experiment.terminate_process_tree(process)
                except Exception as cleanup_error:
                    cleanup = {"error": repr(cleanup_error)}
            update(status="failed", error=repr(error), child_cleanup=cleanup,
                   child_pid=process.pid if process is not None and process.poll() is None else None)
            raise


if __name__ == "__main__":
    run_continuation()
