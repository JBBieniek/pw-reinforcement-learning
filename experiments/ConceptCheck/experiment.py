"""Train, evaluate and resume one algorithm at fixed decision budgets."""

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys

if __package__:
    from .process_control import terminate_process_tree
else:
    from process_control import terminate_process_tree


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MILESTONES = "1000000,2000000,4000000,7000000,10000000,15000000"
SUMMARY_FIELDS = (
    "mean_reward", "median_reward", "min_reward", "max_reward", "std_reward",
    "wins_864", "win_rate_864", "hit_step_limit", "mean_steps", "episodes",
)
TRAINING_DEFAULTS = {
    "replay_storage": "deque", "terminal_on_life_loss": False,
    "buffer_capacity": 50_000, "epsilon_decay_steps": 250_000, "lr": 1e-4,
}


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _milestones(value):
    try:
        values = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("milestones must be comma-separated integers") from error
    if not values or any(item <= 0 for item in values) or values != sorted(set(values)):
        raise argparse.ArgumentTypeError("milestones must be positive and strictly increasing")
    return values


def _validate_extension_history(milestones, previous_targets, history):
    """Only durable, contiguous extensions can authorize an older checkpoint target."""
    if (not isinstance(previous_targets, list) or not isinstance(history, list)
            or any(type(target) is not int for target in previous_targets)):
        raise ValueError("invalid experiment extension history")
    targets = []
    last = None
    for extension in history:
        before = extension["from_milestones"]
        appended = extension["appended_milestones"]
        if (not isinstance(before, list) or not isinstance(appended, list)
                or not before or not appended
                or any(type(item) is not int or item <= 0 for item in before + appended)
                or before + appended != sorted(set(before + appended))
                or (last is not None and before != last)):
            raise ValueError("invalid experiment extension history")
        last = before + appended
        targets.append(before[-1])
    if targets != previous_targets or (last is not None and last != milestones):
        raise ValueError("experiment target history does not match its milestones")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-dir", required=True, help="Separate output directory for this algorithm")
    parser.add_argument("--algorithm", choices=["dqn", "double-dqn"], default="double-dqn")
    parser.add_argument("--activation", choices=["relu", "leaky-relu"], default="leaky-relu")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--replay-storage", choices=["deque", "frames"], default="deque")
    parser.add_argument("--terminal-on-life-loss", action=argparse.BooleanOptionalAction, default=False,
                        help="Mask the training TD target on life loss; evaluation remains full-game")
    parser.add_argument("--buffer-capacity", type=int, default=50_000)
    parser.add_argument("--epsilon-decay-steps", type=int, default=250_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--milestones", type=_milestones, default=_milestones(DEFAULT_MILESTONES))
    parser.add_argument("--append-milestones", type=_milestones,
                        help="Append later evaluation budgets to a completed experiment")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=3303497698)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=27_000, help="Evaluation decision cap per game")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps-per-second", type=float, default=220.0)
    parser.add_argument("--resume-experiment", action="store_true")
    arguments = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(arguments)
    if args.append_milestones is not None and not args.resume_experiment:
        parser.error("--append-milestones requires --resume-experiment")
    supplied = {item.split("=", 1)[0] for item in arguments if item.startswith("--")}
    if args.append_milestones is not None and "--milestones" in supplied:
        parser.error("use --append-milestones without --milestones when extending an experiment")
    args.run_dir = str(Path(args.run_dir).resolve())
    config_path = Path(args.run_dir) / "experiment.config.json"
    if args.resume_experiment:
        if not config_path.is_file():
            parser.error("--resume-experiment requires an existing experiment.config.json")
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if saved.get("format_version") != 1:
            parser.error("unsupported experiment config version")
        args._saved_config = saved
        previous_targets = saved.get("previous_targets", [])
        extension_history = saved.get("milestone_extensions", [])
        try:
            _validate_extension_history(saved["milestones"], previous_targets, extension_history)
        except (KeyError, TypeError, ValueError) as error:
            parser.error(str(error))
        args.previous_targets = list(previous_targets)
        args.milestone_extensions = list(extension_history)
        legacy_defaults = {"activation": "relu", "deterministic": False, **TRAINING_DEFAULTS}
        for name in ("algorithm", "activation", "deterministic", "milestones", "seed",
                     "eval_seed", "eval_episodes", "max_steps", *TRAINING_DEFAULTS):
            option = "--" + name.replace("_", "-")
            saved_value = saved.get(name, legacy_defaults[name]) if name in legacy_defaults else saved[name]
            explicitly_supplied = option in supplied or "--no-" + name.replace("_", "-") in supplied
            if explicitly_supplied and getattr(args, name) != saved_value:
                parser.error(f"{option} differs from the saved experiment ({saved_value!r})")
            setattr(args, name, saved_value)
        # Device and pacing are operational settings; changing either does not
        # alter the training or evaluation decision budget.
        for name in ("device", "max_steps_per_second"):
            if "--" + name.replace("_", "-") not in supplied:
                setattr(args, name, saved[name])
        args.created_at_utc = saved["created_at_utc"]
        args._append_replay = bool(args.milestone_extensions and args.append_milestones
                                  == args.milestone_extensions[-1]["appended_milestones"])
        if (args.append_milestones is not None and not args._append_replay
                and args.append_milestones[0] <= args.milestones[-1]):
            parser.error("--append-milestones must be strictly above the saved final milestone")
    elif config_path.exists():
        parser.error("run directory already contains an experiment; use --resume-experiment")
    else:
        args.created_at_utc = _timestamp()
        args.previous_targets = []
        args.milestone_extensions = []
        args._append_replay = False
    if args.seed is None:
        args.seed = secrets.randbits(32)
    if any(not 0 <= seed < 2**32 for seed in (args.seed, args.eval_seed)):
        parser.error("seeds must be between 0 and 4294967295")
    if args.eval_episodes <= 0 or args.max_steps <= 0:
        parser.error("--eval-episodes and --max-steps must be positive")
    if args.buffer_capacity < 20_000 or args.epsilon_decay_steps <= 0:
        parser.error("--buffer-capacity must be at least 20000 and --epsilon-decay-steps positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    if not math.isfinite(args.max_steps_per_second) or args.max_steps_per_second < 0:
        parser.error("--max-steps-per-second must be finite and non-negative")
    return args


@contextmanager
def _experiment_lock(directory):
    """OS lock is released on a crash, so stale PID files cannot block resume."""
    path = directory / "experiment.lock"
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("another experiment controller is using this run directory") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_training_checkpoint(path):
    if __package__:
        from .checkpoints import load_training_checkpoint
    else:
        from checkpoints import load_training_checkpoint
    saved = load_training_checkpoint(path)
    result = {"step": saved["step"], "algorithm": saved["algorithm"],
              "seed": saved["config"]["seed"], "steps": saved["config"]["steps"],
              "activation": saved["config"].get("activation", "relu"),
              "deterministic": saved["config"].get("deterministic", False)}
    result.update({name: saved["config"].get(name, default)
                   for name, default in TRAINING_DEFAULTS.items()})
    # Never retain the multi-GB replay while starting the next training child.
    del saved
    gc.collect()
    return result


def _validated_checkpoint(path, args):
    saved = _read_training_checkpoint(path)
    accepted_targets = [args.milestones[-1], *args.previous_targets]
    if (saved["algorithm"] != args.algorithm or saved["seed"] != args.seed
            or saved["steps"] not in accepted_targets
            or saved.get("activation", "relu") != args.activation
            or saved.get("deterministic", False) is not args.deterministic):
        raise ValueError("training checkpoint does not match this experiment")
    if any(saved.get(name, default) != getattr(args, name, default)
           for name, default in TRAINING_DEFAULTS.items()):
        raise ValueError("training checkpoint does not match this experiment's training settings")
    if saved["step"] > saved["steps"] or saved["step"] > args.milestones[-1]:
        raise ValueError("training checkpoint exceeds the experiment target")
    return saved["step"]


def _process_alive(pid):
    """Check a recorded child without sending signals on Windows."""
    if not isinstance(pid, int) or not 0 < pid < 2**32:
        raise ValueError("invalid child PID in previous experiment progress")
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # ERROR_INVALID_PARAMETER: PID no longer exists.
            return False
        if error == 5:  # Access denied is not evidence that the process exited.
            return True
        raise ctypes.WinError(error)
    try:
        exit_code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return exit_code.value == 259  # STILL_ACTIVE
    finally:
        kernel.CloseHandle(handle)


def _weights_path(directory, milestone):
    return directory / f"model.step-{milestone:09d}.pth"


def _evaluation_path(directory, milestone):
    return directory / f"eval-{milestone:09d}.json"


def _evaluation_report(path, weights, milestone, args, *, cached):
    if not path.is_file() or not weights.is_file():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        training = json.loads(weights.with_suffix(".config.json").read_text(encoding="utf-8"))
        expected_environment = {key: training[key] for key in
                                ("mode", "difficulty", "frame_skip", "sticky_action_probability")}
        expected_activation = training.get("activation", "relu")
        expected_deterministic = training.get("deterministic", False)
        if any(training.get(name, default) != getattr(args, name, default)
               for name, default in TRAINING_DEFAULTS.items()):
            return None
        expected_seeds = [(args.eval_seed + index) % 2**32 for index in range(args.eval_episodes)]
        if (not report.get("completed_at_utc")
                or expected_activation != args.activation
                or expected_deterministic is not args.deterministic
                or report.get("activation", "relu") != expected_activation
                or report.get("deterministic", False) is not expected_deterministic
                or report.get("evaluation_seed") != args.eval_seed
                or report.get("episode_seeds") != expected_seeds
                or report.get("policy_epsilon") != 0.0
                or report.get("max_steps") != args.max_steps
                or report.get("auto_fire_after_life_loss") is not True
                or report.get("environment") != expected_environment
                or len(report.get("episodes", [])) != args.eval_episodes
                or Path(report["model_path"]).resolve() != weights.resolve()
                or report["summary"]["episodes"] != args.eval_episodes):
            return None
        for name in SUMMARY_FIELDS:
            if not math.isfinite(float(report["summary"][name])):
                return None
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        if cached and (report.get("checkpoint_step") != milestone
                       or report.get("checkpoint_sha256") != digest):
            return None
        report.update(checkpoint_step=milestone, checkpoint_sha256=digest,
                      experiment_algorithm=args.algorithm)
        return report
    except (KeyError, TypeError, ValueError, OSError):
        return None


def _write_comparison(directory, results):
    path = directory / "comparison.csv"
    temporary = path.with_suffix(".csv.tmp")
    fields = ["decisions", "algorithm", *SUMMARY_FIELDS]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({"decisions": result["checkpoint_step"],
                             "algorithm": result["experiment_algorithm"],
                             **{name: result["summary"][name] for name in SUMMARY_FIELDS}})
    os.replace(temporary, path)


def _run_child(command, update, *, cwd=None):
    process = subprocess.Popen(command, cwd=ROOT if cwd is None else cwd)
    try:
        update(child_pid=process.pid, child_cleanup=None)
        returncode = process.wait()
    except BaseException:
        cleanup = terminate_process_tree(process)
        update(child_cleanup=cleanup)
        raise
    finally:
        if process.poll() is not None:
            update(child_pid=None)
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def run_experiment(argv=None):
    args = parse_args(argv)
    directory = Path(args.run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "model.training.pt"
    progress_path = directory / "experiment.progress.json"
    config_path = directory / "experiment.config.json"
    results = []
    progress = {
        "status": "starting", "pid": os.getpid(), "child_pid": None, "child_cleanup": None,
        "algorithm": args.algorithm, "seed": args.seed, "device": args.device,
        "activation": args.activation, "deterministic": args.deterministic,
        "training_settings": {name: getattr(args, name) for name in TRAINING_DEFAULTS},
        "milestones": args.milestones, "current_milestone": None,
        "next_milestone": args.milestones[0],
        "completed_milestones": [], "results": [], "error": None,
        "started_at_utc": args.created_at_utc,
    }

    def update(**changes):
        progress.update(changes)
        progress["updated_at_utc"] = _timestamp()
        _write_json(progress_path, progress)

    with _experiment_lock(directory):
        if args.resume_experiment and json.loads(config_path.read_text(encoding="utf-8")) != args._saved_config:
            raise RuntimeError("experiment configuration changed before the controller acquired its lock; retry")
        previous = {}
        if progress_path.exists():
            previous = json.loads(progress_path.read_text(encoding="utf-8"))
            previous_child = previous.get("child_pid")
            if previous_child is not None and _process_alive(previous_child):
                raise RuntimeError(f"previous training/evaluation child {previous_child} is still running; wait before resuming")
        saved_step = None
        if args.append_milestones is not None:
            previous_pid = previous.get("pid")
            # A same-process API call is no longer running the old controller:
            # it has already released the lock held by the previous invocation.
            if previous_pid is not None and previous_pid != os.getpid() and _process_alive(previous_pid):
                raise RuntimeError(f"previous experiment controller {previous_pid} is still running; wait before appending")
            if not checkpoint.is_file():
                raise ValueError("appending milestones requires the completed full training checkpoint")
            if not args._append_replay:
                if previous.get("status") != "completed":
                    raise ValueError("--append-milestones requires a completed experiment")
                saved_step = _validated_checkpoint(checkpoint, args)
                if saved_step != args.milestones[-1]:
                    raise ValueError("appending milestones requires a checkpoint at the saved final milestone")
                before = list(args.milestones)
                args.previous_targets = [*args.previous_targets, before[-1]]
                args.milestone_extensions = [*args.milestone_extensions, {
                    "from_milestones": before,
                    "appended_milestones": list(args.append_milestones),
                    "registered_at_utc": _timestamp(),
                }]
                args.milestones = before + args.append_milestones
            else:
                saved_step = _validated_checkpoint(checkpoint, args)
            progress["milestones"] = args.milestones
        # Recheck after taking the lock: another controller may have created it
        # between argument parsing and lock acquisition.
        if not args.resume_experiment and any(path.exists() for path in
                (config_path, checkpoint, directory / "model.pth", directory / "model.config.json")):
            raise ValueError("run directory already contains an experiment; use --resume-experiment")
        config = {key: value for key, value in vars(args).items()
                  if not key.startswith("_") and key not in ("resume_experiment", "append_milestones")}
        # Existing continuation requests identify the original configuration
        # exactly. Do not add redundant default fields to legacy configurations.
        if args.resume_experiment:
            for name, default in TRAINING_DEFAULTS.items():
                if name not in args._saved_config and getattr(args, name) == default:
                    config.pop(name)
        config["format_version"] = 1
        _write_json(config_path, config)
        update()
        try:
            if saved_step is None:
                saved_step = _validated_checkpoint(checkpoint, args) if checkpoint.exists() else 0
            if not args.resume_experiment and saved_step:
                raise ValueError("fresh experiment cannot overwrite an existing training checkpoint")
            for milestone in args.milestones:
                weights = _weights_path(directory, milestone)
                evaluation_path = _evaluation_path(directory, milestone)
                report = _evaluation_report(evaluation_path, weights, milestone, args, cached=True)
                if report is not None and milestone > saved_step:
                    raise ValueError("full training checkpoint is missing or older than an evaluated milestone")
                if report is None:
                    if saved_step < milestone:
                        update(status="training", current_milestone=milestone)
                        command = [sys.executable, "-B", "-u", "-m", "experiments.ConceptCheck.train",
                                   "--steps", str(args.milestones[-1]), "--stop-at", str(milestone),
                                   "--activation", args.activation,
                                   "--deterministic" if args.deterministic else "--no-deterministic",
                                   "--device", args.device, "--max-steps-per-second", str(args.max_steps_per_second)]
                        if saved_step:
                            command += ["--resume", str(checkpoint),
                                        "--save-path", str(directory / "model.pth"),
                                        "--checkpoint-path", str(checkpoint)]
                        else:
                            command += ["--algorithm", args.algorithm, "--seed", str(args.seed),
                                        "--save-path", str(directory / "model.pth")]
                            for name in TRAINING_DEFAULTS:
                                value = getattr(args, name)
                                option = "--" + name.replace("_", "-")
                                command += ([option if value else "--no-terminal-on-life-loss"]
                                            if name == "terminal_on_life_loss" else [option, str(value)])
                        print(f"Training {args.algorithm} to {milestone:,} decisions", flush=True)
                        _run_child(command, update)
                        saved_step = _validated_checkpoint(checkpoint, args)
                        if saved_step != milestone:
                            raise ValueError(f"training stopped at {saved_step}, expected {milestone}")
                    if not weights.is_file():
                        raise FileNotFoundError(f"milestone weights are missing: {weights}")
                    update(status="evaluating", current_milestone=milestone)
                    command = [sys.executable, "-B", "-u", "-m", "experiments.ConceptCheck.test",
                               "--model-path", str(weights), "--episodes", str(args.eval_episodes),
                               "--activation", args.activation,
                               "--deterministic" if args.deterministic else "--no-deterministic",
                               "--seed", str(args.eval_seed), "--max-steps", str(args.max_steps),
                               "--auto-fire", "--device", args.device, "--render-mode", "none",
                               "--max-steps-per-second", str(args.max_steps_per_second),
                               "--output", str(evaluation_path)]
                    print(f"Evaluating {milestone:,} decisions on {args.eval_episodes} games", flush=True)
                    _run_child(command, update)
                    report = _evaluation_report(evaluation_path, weights, milestone, args, cached=False)
                    if report is None:
                        raise ValueError(f"evaluation report is missing or incompatible: {evaluation_path}")
                    _write_json(evaluation_path, report)
                results.append(report)
                _write_comparison(directory, results)
                update(status="between_milestones", current_milestone=milestone,
                       next_milestone=next((value for value in args.milestones if value > milestone), None),
                       completed_milestones=[result["checkpoint_step"] for result in results],
                       results=[{"decisions": result["checkpoint_step"], **result["summary"]} for result in results])
            update(status="completed", current_milestone=None, next_milestone=None)
        except KeyboardInterrupt:
            update(status="interrupted", error="Controller interrupted; resume the experiment from its latest full checkpoint")
            raise
        except Exception as error:
            update(status="failed", error=repr(error))
            raise
    return progress


if __name__ == "__main__":
    run_experiment()
