"""Change pacing at a saved milestone, then supervise the requested extension."""

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

if __package__:
    from . import continue_experiment as continuation
    from . import experiment
else:
    import continue_experiment as continuation
    import experiment


class BoundaryAdvanced(RuntimeError):
    """The old controller left its saved milestone; wait without stopping it."""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--at-step", type=int, required=True)
    parser.add_argument("--max-steps-per-second", type=float, default=0.0)
    parser.add_argument("--then-append-milestones", type=experiment._milestones, required=True)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args(argv)
    args.run_dir = str(Path(args.run_dir).resolve())
    if args.at_step <= 0:
        parser.error("--at-step must be positive")
    if not math.isfinite(args.max_steps_per_second) or args.max_steps_per_second < 0:
        parser.error("--max-steps-per-second must be finite and non-negative")
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        parser.error("--poll-seconds must be finite and positive")
    return args


def _argv(command_line):
    if os.name != "nt":
        return shlex.split(command_line)
    count = ctypes.c_int()
    shell = ctypes.WinDLL("shell32", use_last_error=True)
    shell.CommandLineToArgvW.argtypes = (ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int))
    shell.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LocalFree.argtypes = (ctypes.c_void_p,)
    kernel.LocalFree.restype = ctypes.c_void_p
    arguments = shell.CommandLineToArgvW(command_line, ctypes.byref(count))
    if not arguments:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return [arguments[index] for index in range(count.value)]
    finally:
        kernel.LocalFree(arguments)


def _process_identity(pid):
    """Read the real interpreter's identity, without signaling any process."""
    if type(pid) is not int or not 0 < pid < 2**32:
        raise ValueError("invalid original controller PID")
    if os.name != "nt":
        raise RuntimeError("verified controller handoff currently requires Windows CIM")
    powershell = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                              "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
    script = (
        "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); "
        f"$taskProcess=Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}'; "
        "if ($null -eq $taskProcess) { exit 3 }; "
        "[pscustomobject]@{pid=[int]$taskProcess.ProcessId; "
        "parent_pid=[int]$taskProcess.ParentProcessId; command_line=$taskProcess.CommandLine; "
        "creation_time=$taskProcess.CreationDate.ToUniversalTime().ToString('o'); "
        "executable=$taskProcess.ExecutablePath} | ConvertTo-Json -Compress"
    )
    result = subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-Command", script],
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError(f"cannot verify controller {pid}: {result.stderr.strip() or result.returncode}")
    identity = json.loads(result.stdout.lstrip("\ufeff"))
    if (identity.get("pid") != pid or not identity.get("command_line")
            or not identity.get("creation_time") or not identity.get("executable")):
        raise ValueError("incomplete controller process identity")
    return identity


def _validate_controller(identity, directory):
    arguments = _argv(identity["command_line"])
    if ("-m" not in arguments or arguments[arguments.index("-m") + 1:][:1]
            != ["experiments.ConceptCheck.experiment"]):
        raise ValueError("controller PID does not run the experiment module")
    run_dirs = []
    for index, argument in enumerate(arguments):
        if argument == "--run-dir" and index + 1 < len(arguments):
            run_dirs.append(arguments[index + 1])
        elif argument.startswith("--run-dir="):
            run_dirs.append(argument.split("=", 1)[1])
    if len(run_dirs) != 1 or Path(run_dirs[0]).resolve() != directory.resolve():
        raise ValueError("controller PID belongs to a different run directory")


def _same_identity(expected, actual, directory):
    _validate_controller(actual, directory)
    if any(actual.get(key) != expected.get(key)
           for key in ("pid", "creation_time", "command_line", "executable")):
        raise RuntimeError("original controller process identity changed; no process was stopped")


def _resumed_config(request):
    baseline = dict(request["baseline_config"])
    baseline.setdefault("previous_targets", [])
    baseline.setdefault("milestone_extensions", [])
    baseline["max_steps_per_second"] = request["max_steps_per_second"]
    return baseline


def _config_stage(config, request):
    if config == request["baseline_config"]:
        return "original"
    resumed = _resumed_config(request)
    if config == resumed:
        return "resumed"
    extension_request = {"baseline_config": resumed,
                         "append_milestones": request["then_append_milestones"],
                         "max_steps_per_second": request["max_steps_per_second"]}
    if continuation._config_stage(config, extension_request) == "extended":
        return "extended"
    raise ValueError("experiment configuration differs from the acceleration request")


def _request(directory, queue, args):
    config = continuation._read_json(directory / "experiment.config.json")
    path = queue / "acceleration.request.json"
    if path.exists():
        request = continuation._read_json(path)
        if (request.get("format_version") != 1 or request.get("run_dir") != args.run_dir
                or request.get("at_step") != args.at_step
                or request.get("max_steps_per_second") != args.max_steps_per_second
                or request.get("then_append_milestones") != args.then_append_milestones
                or request.get("baseline_config_sha256") != continuation._identity(request["baseline_config"])):
            raise ValueError("a different acceleration request exists; it was not overwritten")
    else:
        milestones = config["milestones"]
        if args.at_step not in milestones[:-1] or args.then_append_milestones[0] <= milestones[-1]:
            raise ValueError("acceleration requires an intermediate milestone and later continuation budgets")
        original = continuation._read_json(directory / "experiment.progress.json")
        identity = _process_identity(original["pid"])
        _validate_controller(identity, directory)
        request = {"format_version": 1, "run_dir": args.run_dir, "at_step": args.at_step,
                   "then_append_milestones": args.then_append_milestones,
                   "max_steps_per_second": args.max_steps_per_second,
                   "baseline_config": config, "baseline_config_sha256": continuation._identity(config),
                   "original_controller": identity, "created_at_utc": experiment._timestamp()}
        experiment._write_json(path, request)
    _validate_controller(request["original_controller"], directory)
    _config_stage(config, request)
    return request


def _fingerprint(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def _paused_step(directory, request):
    progress = continuation._read_json(directory / "model.progress.json")
    step = progress.get("step")
    if (progress.get("status") != "paused" or type(step) is not int
            or step < request["at_step"]):
        return None
    if step not in request["baseline_config"]["milestones"][:-1]:
        raise ValueError("paused training is not at an intermediate experiment milestone")
    controller = continuation._read_json(directory / "experiment.progress.json")
    if controller.get("pid") != request["original_controller"]["pid"]:
        raise RuntimeError("original controller changed while checking a safe boundary")
    if (controller.get("status") not in ("evaluating", "between_milestones")
            or controller.get("current_milestone") != step):
        return None
    return step


def _boundary(directory, request):
    step = _paused_step(directory, request)
    if step is None:
        raise BoundaryAdvanced("training is running; no handoff is permitted")
    settings = continuation._settings(request["baseline_config"])
    source = directory / "model.training.pt"
    if experiment._validated_checkpoint(source, settings) != step:
        raise ValueError("paused training progress and full checkpoint steps differ")
    weights = experiment._weights_path(directory, step)
    sidecar = weights.with_suffix(".config.json")
    training = continuation._read_json(sidecar)
    for name in ("algorithm", "seed", "activation", "deterministic"):
        if training.get(name, {"activation": "relu", "deterministic": False}.get(name)) != getattr(settings, name):
            raise ValueError(f"milestone weights metadata differs: {name}")
    if training.get("steps") != settings.milestones[-1]:
        raise ValueError("milestone weights have a different overall training target")
    boundary = {"step": step, "files": {str(path): _fingerprint(path)
                for path in (source, weights, sidecar)}}
    if _paused_step(directory, request) != step:
        raise BoundaryAdvanced("training advanced during checkpoint validation")
    return boundary


def _still_at_boundary(directory, request, boundary):
    return (_paused_step(directory, request) == boundary["step"]
            and all(_fingerprint(Path(path)) == stamp for path, stamp in boundary["files"].items()))


def _archive(directory, queue, request, boundary):
    source = directory / "model.training.pt"
    destination = queue / f"model.training-at-{boundary['step']:09d}.pt"
    temporary = destination.with_suffix(".pt.tmp")
    try:
        if not destination.exists():
            shutil.copyfile(source, temporary)
            if not _still_at_boundary(directory, request, boundary):
                raise BoundaryAdvanced("training advanced while the checkpoint was being copied")
            os.replace(temporary, destination)
        digest = continuation._digest(destination)
        if not _still_at_boundary(directory, request, boundary):
            raise BoundaryAdvanced("training advanced while the archive was being checked")
        if digest != continuation._digest(source):
            raise ValueError("archived full checkpoint differs from the paused source")
        return {"snapshot_path": str(destination.resolve()), "snapshot_sha256": digest,
                "boundary": boundary}
    finally:
        temporary.unlink(missing_ok=True)


def _stop_original(directory, request, boundary):
    """Kill only the verified controller tree, immediately after the final pause guard."""
    expected = request["original_controller"]
    current = continuation._read_json(directory / "experiment.progress.json")
    if current.get("pid") != expected["pid"]:
        raise RuntimeError("experiment controller changed before handoff")
    _same_identity(expected, _process_identity(expected["pid"]), directory)
    # CIM and a multi-GB copy can take time. Never stop the next training phase.
    if not _still_at_boundary(directory, request, boundary):
        return False
    if os.name != "nt":
        raise RuntimeError("verified tree handoff currently requires Windows")
    taskkill = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "taskkill.exe")
    result = subprocess.run([taskkill, "/PID", str(expected["pid"]), "/T", "/F"],
                            capture_output=True, text=True, errors="replace", timeout=30,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError(f"verified controller tree was not stopped: {result.stderr.strip() or result.stdout.strip()}")
    deadline = time.monotonic() + 30
    while experiment._process_alive(expected["pid"]) and time.monotonic() < deadline:
        time.sleep(0.1)
    child = current.get("child_pid")
    if experiment._process_alive(expected["pid"]) or (child is not None and experiment._process_alive(child)):
        raise RuntimeError("old controller or child survived the requested tree stop")
    return True


def _wait_and_handoff(directory, queue, request, poll_seconds, update):
    expected = request["original_controller"]
    while True:
        if continuation._read_json(directory / "experiment.config.json") != request["baseline_config"]:
            raise RuntimeError("original experiment configuration changed before the safe boundary")
        original = continuation._read_json(directory / "experiment.progress.json")
        if original.get("pid") != expected["pid"]:
            raise RuntimeError("original experiment controller changed before the safe boundary")
        if not experiment._process_alive(expected["pid"]):
            raise RuntimeError("original controller exited before handoff; no training was restarted")
        if original.get("status") in ("failed", "interrupted"):
            raise RuntimeError("original controller failed before handoff")
        step = _paused_step(directory, request)
        update(status="waiting_checkpoint", original_controller_pid=expected["pid"],
               observed_training=continuation._read_json(directory / "model.progress.json"))
        if step is not None:
            try:
                _same_identity(expected, _process_identity(expected["pid"]), directory)
                boundary = _boundary(directory, request)
                archive = _archive(directory, queue, request, boundary)
                update(status="stopping_controller", handoff_intent=True, **archive)
                if _stop_original(directory, request, boundary):
                    update(status="ready_to_resume", handoff_complete=True)
                    return
                update(status="waiting_checkpoint", handoff_intent=False)
            except BoundaryAdvanced:
                update(status="waiting_checkpoint", handoff_intent=False)
        time.sleep(poll_seconds)


def _check_complete(directory, config):
    final = continuation._read_json(directory / "experiment.progress.json")
    if any(continuation._live_processes(final)):
        raise RuntimeError("a controller or child is still active; no duplicate was launched")
    continuation._validate_evaluations(directory, config, final, require_completed=True)
    if experiment._validated_checkpoint(directory / "model.training.pt", continuation._settings(config)) != config["milestones"][-1]:
        raise ValueError("completed experiment has a different final checkpoint step")


def _run_phase(command, phase, poll_seconds, update):
    process = subprocess.Popen(command, cwd=experiment.ROOT)
    update(status=phase, command=command, child_pid=process.pid)
    while process.poll() is None:
        time.sleep(poll_seconds)
    code = process.wait()
    update(child_pid=None, child_returncode=code)
    if code:
        raise subprocess.CalledProcessError(code, command)


def run_acceleration(argv=None):
    args = parse_args(argv)
    directory = Path(args.run_dir)
    queue = directory / "acceleration"
    queue.mkdir(exist_ok=True)
    with experiment._experiment_lock(queue):
        request = _request(directory, queue, args)
        progress_path = queue / "acceleration.progress.json"
        previous = continuation._read_json(progress_path) if progress_path.exists() else {}
        old_child = previous.get("child_pid")
        if old_child is not None and experiment._process_alive(old_child):
            raise RuntimeError("the acceleration request already has a live child; no duplicate was launched")
        progress = {**previous, "pid": os.getpid(), "child_pid": None, "error": None,
                    "target_step": args.at_step, "max_steps_per_second": args.max_steps_per_second,
                    "final_target": args.then_append_milestones[-1]}

        def update(**changes):
            progress.update(changes)
            progress["updated_at_utc"] = experiment._timestamp()
            experiment._write_json(progress_path, progress)

        try:
            current = continuation._read_json(directory / "experiment.progress.json")
            original_pid = request["original_controller"]["pid"]
            current_alive = any(continuation._live_processes(current))
            if current.get("pid") != original_pid and current_alive:
                raise RuntimeError("another experiment controller/child is active; no duplicate was launched")
            stage = _config_stage(continuation._read_json(directory / "experiment.config.json"), request)
            if stage == "extended":
                if current.get("status") == "completed":
                    _check_complete(directory, continuation._read_json(directory / "experiment.config.json"))
                    update(status="completed")
                    return progress
            elif not progress.get("handoff_complete"):
                if progress.get("handoff_intent") and not current_alive:
                    archived = Path(progress["snapshot_path"])
                    if continuation._digest(archived) != progress["snapshot_sha256"]:
                        raise ValueError("handoff recovery archive checksum differs")
                    if not _still_at_boundary(directory, request, progress["boundary"]):
                        raise RuntimeError("handoff recovery no longer matches the saved boundary")
                    update(handoff_complete=True, status="ready_to_resume")
                else:
                    _wait_and_handoff(directory, queue, request, args.poll_seconds, update)
            if stage != "extended":
                config = continuation._read_json(directory / "experiment.config.json")
                current = continuation._read_json(directory / "experiment.progress.json")
                if any(continuation._live_processes(current)):
                    raise RuntimeError("an experiment controller/child is still active before resume")
                if current.get("status") != "completed":
                    command = [sys.executable, "-u", "-m", "experiments.ConceptCheck.experiment",
                               "--run-dir", str(directory), "--resume-experiment",
                               "--max-steps-per-second", str(args.max_steps_per_second)]
                    _run_phase(command, "resuming", args.poll_seconds, update)
                    config = continuation._read_json(directory / "experiment.config.json")
                if config != _resumed_config(request):
                    raise ValueError("resumed controller did not retain the requested settings")
                _check_complete(directory, config)
            command = [sys.executable, "-u", "-m", "experiments.ConceptCheck.continue_experiment",
                       "--run-dir", str(directory), "--append-milestones",
                       ",".join(map(str, args.then_append_milestones)),
                       "--max-steps-per-second", str(args.max_steps_per_second),
                       "--poll-seconds", str(args.poll_seconds)]
            _run_phase(command, "continuing", args.poll_seconds, update)
            config = continuation._read_json(directory / "experiment.config.json")
            if _config_stage(config, request) != "extended":
                raise ValueError("continuation did not register the requested final target")
            _check_complete(directory, config)
            update(status="completed")
            return progress
        except BaseException as error:
            # Never interrupt an active trainer because its guardian failed.
            # Durable PID/state records prevent a retry from launching a duplicate.
            update(status="failed", error=repr(error))
            raise


if __name__ == "__main__":
    run_acceleration()
